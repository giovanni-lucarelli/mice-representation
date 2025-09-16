from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


from pathlib import Path
import xarray as xr

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xarray as xr

try:
    import zarr  # optional, required if out_format=='zarr'
except Exception:  # pragma: no cover — optional dependency
    zarr = None

# -------------------------
# Constants & small helpers
# -------------------------
AREANAME = {"VISp": "V1", "VISl": "LM", "VISal": "AL",
            "VISrl": "RL", "VISam": "AM", "VISpm": "PM"}

# ---------------------------------- helpers --------------------------------- #
def longest_true_run(mask_bool: Sequence[bool]) -> Optional[Tuple[int, int]]:
    """Return (start, end) indices (end exclusive) of the longest contiguous True run."""

    # find the longest contiguous list of True in mask_bool
    mask = np.asarray(mask_bool, dtype=bool)
    if not np.any(mask):
        return None
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])

def q75_higher(x: Sequence[int]) -> int:
    x = np.asarray(list(x))
    if x.size == 0:
        return 0
    try:
        return int(np.quantile(x, 0.75, method="higher"))  # NumPy ≥1.22
    except TypeError:  # back-compat
        return int(np.percentile(x, 75, interpolation="higher"))

# ---------------------------- main preprocessing ---------------------------- #
def trial_mean_over_window(
    ds: xr.Dataset,
    unit_idx: np.ndarray,
    t0: int,
    t1: int,
    time_dim: str = "time_relative_to_stimulus_onset",
    trial_dim: str = "trials",
    image_dim: str = "frame_id",
    unit_dim: str = "units",
) -> np.ndarray:
    
    """Average ds['neural_data'] across time bins [t0:t1) → (trials, images, units) float32."""

    da = ds["neural_data"].isel({unit_dim: unit_idx, time_dim: slice(t0, t1)}).mean(time_dim)
    da = da.transpose(trial_dim, image_dim, unit_dim)
    return da.astype("float32").values  # (T, F, U)

@dataclass
class PreprocConfig:
    apply_running_filter: bool = False
    window_median_thr: float = 0.3
    overwrite: bool = True
    skip_existing: bool = False
    equalize_to_p75: bool = False
    equalize_seed: int = 0


def _ensure_str(a):
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        try:
            return np.char.decode(a, "utf-8")
        except Exception:
            pass  # <-- fixed typo here
    return a.astype(str)

def preprocess_neuropixel(
    ds: xr.Dataset,
    out_root: str = "neuropixels_ns_preproc",
    index_name: str = "index.csv",
    out_format: str = "npy",   # <- default to npy if you like
    config: PreprocConfig = PreprocConfig(),
    save_design_matrices: bool = False,
    design_root: Optional[str] = None,
):
    os.makedirs(out_root, exist_ok=True)
    if save_design_matrices:
        design_root = design_root or os.path.join(out_root, "design_mats")
        os.makedirs(design_root, exist_ok=True)

    # --- coords & limits ---
    time_name = "time_relative_to_stimulus_onset"
    time_coord = np.asarray(ds[time_name].values)  # seconds
    tmask = (time_coord >= 0.0) & (time_coord < 0.25 + 1e-12)
    t_idx = np.flatnonzero(tmask)
    if t_idx.size == 0:
        raise ValueError("No time points in [0, 0.25) s")

    r_tu = ds["splithalf_r_mean"].values # (time, units)
    r_tu = r_tu[t_idx, :]  # window for search

    frame_vals = np.asarray(ds["frame_id"].values)
    specimen   = np.asarray(ds["specimen_id"].values).astype(int)
    area_code  = _ensure_str(ds["visual_area"].values)
    unit_ids   = np.asarray(ds["unit_id"].values).astype(int)

    # running filter (off by default for Neuropixels)
    # this was used in the original paper for calcium imaging data
    # signals if the mouse response doesn't change significantly with running speed

    # return a boolean mask of units to keep, used later
    if config.apply_running_filter and ("run_pval_ns" in ds):
        run_p = np.asarray(ds["run_pval_ns"].values)
        run_ok = np.logical_or(np.isnan(run_p), run_p > 0.05)
    else:
        run_ok = np.ones(ds.sizes["units"], dtype=bool)

    # group units
    groups: Dict[Tuple[int, str], List[int]] = {}
    for u, (s, a) in enumerate(zip(specimen, area_code)):
        groups.setdefault((int(s), str(a)), []).append(int(u))

    meta_rows: List[dict] = []
    counts_by_area: Dict[str, List[int]] = {}

    # pass 1: find windows & collect counts
    for (s, a), idxs in groups.items():
        idxs_arr = np.asarray(idxs, dtype=int)
        gate = run_ok[idxs_arr] & np.isfinite(r_tu[:, idxs_arr]).any(axis=0)
        sel = idxs_arr[gate]
        if sel.size == 0:
            continue

        med_t = np.nanmedian(r_tu[:, sel], axis=1)
        run = longest_true_run(med_t >= config.window_median_thr)
        if run is None:
            continue

        rel_t0, rel_t1 = run  # indices in masked axis, end-exclusive
        t0 = int(t_idx[rel_t0])
        t1 = int(t_idx[rel_t1 - 1]) + 1   # end-exclusive back on full axis
        t1 = min(t1, len(time_coord))     # clamp

        meta_rows.append({
            "specimen_id": int(s),
            "area_code": a,
            "sel_units": sel,
            "t0": t0, "t1": t1,
        })
        counts_by_area.setdefault(a, []).append(sel.size)

    # area-wise P75 thresholds
    p75: Dict[str, int] = {a: q75_higher(c) for a, c in counts_by_area.items()}

    # pass 2: save arrays + metadata, build index
    index_rows: List[dict] = []
    for row in meta_rows:
        s, a, sel, t0, t1 = row["specimen_id"], row["area_code"], row["sel_units"], row["t0"], row["t1"]
        if sel.size < p75[a]:
            continue

        K = p75[a]
        sel_eq = sel
        if config.equalize_to_p75 and sel.size > K:
            rng = np.random.default_rng(config.equalize_seed + int(s))
            sel_eq = np.sort(rng.choice(sel, size=K, replace=False))

        arr = trial_mean_over_window(ds, sel_eq, t0, t1, time_dim=time_name)

        keep_u = np.isfinite(arr).any(axis=(0, 1))
        arr, sel_eq = arr[:, :, keep_u], sel_eq[keep_u]
        if arr.shape[2] < p75[a]:
            continue

        area_name = AREANAME.get(a, a)
        base = f"{area_name}_{s}"

        # ---------- SAVE + INDEX ----------
        from os import fspath
        # compute time endpoints (seconds) safely
        dt = float(np.median(np.diff(time_coord))) if len(time_coord) > 1 else 0.01
        start_s = float(time_coord[t0])
        end_s   = float(time_coord[t1 - 1] + dt)
        end_s   = min(end_s, float(time_coord[-1] + dt))

        # choose store path by format
        if out_format == "zarr":
            if zarr is None:
                raise RuntimeError("zarr is not available; install it or choose a different out_format")
            store_path = os.path.join(out_root, f"{base}.zarr")
            if os.path.exists(store_path):
                if config.overwrite: shutil.rmtree(store_path, ignore_errors=True)
                elif config.skip_existing: pass
                else: raise FileExistsError(f"Exists: {store_path}")
            zarr.save(store_path, arr)

        elif out_format == "npy":
            store_path = os.path.join(out_root, f"{base}.npy")
            if os.path.exists(store_path):
                if config.overwrite: os.remove(store_path)
                elif config.skip_existing: pass
                else: raise FileExistsError(f"Exists: {store_path}")
            np.save(store_path, arr)

        else:
            raise ValueError(f"Unsupported out_format: {out_format}")

        store_path = fspath(store_path)
        meta_path  = fspath(os.path.join(out_root, f"{base}.json"))

        # write metadata JSON
        with open(meta_path, "w") as f:
            json.dump({
                "specimen_id": int(s),
                "visual_area_code": a,
                "visual_area": area_name,
                "n_trials": int(arr.shape[0]),
                "n_images": int(arr.shape[1]),
                "n_units": int(arr.shape[2]),
                "time_window_idx": [int(t0), int(t1)],     # [start, end)
                "time_window_s":   [start_s, end_s],       # [start, end)
                "unit_ids": unit_ids[sel_eq].tolist(),
                "frame_id_values": [int(x) for x in frame_vals[:arr.shape[1]]],
                "area_unit_threshold_p75": int(p75[a]),
                "window_median_thr": float(config.window_median_thr),
                "equalized_to_p75": bool(config.equalize_to_p75),
                "equalize_seed": int(config.equalize_seed),
            }, f)

        # optional design matrix (images × units), averaged across trials
        if save_design_matrices and design_root is not None:
            X = arr.mean(axis=0).astype("float32")  # (images, units)
            np.save(os.path.join(design_root, f"X_{s}_{area_name}.npy"), X)

        # add to index
        index_rows.append({
            "area": area_name,
            "area_code": a,
            "specimen_id": int(s),
            "n_units": int(arr.shape[2]),
            "store": store_path,
            "meta": meta_path,
        })
        # ---------- /SAVE + INDEX ----------

    # build and save index table
    index_df = pd.DataFrame(index_rows).sort_values(["area", "specimen_id"]).reset_index(drop=True)
    index_path = os.path.join(out_root, index_name)
    if index_name.lower().endswith(".json"):
        index_df.to_json(index_path, orient="records", indent=2)
    else:
        index_df.to_csv(index_path, index=False)

    print("Saved", len(index_df), "specimen-area arrays to", out_root)
    print("P75 thresholds (units/specimen) by area:", {AREANAME.get(k, k): int(v) for k, v in p75.items()})

    return index_df, p75



def save_all_stimuli_as_png(dataset: xr.Dataset, output_folder: str | Path):
    """
    Saves all stimuli from the xarray Dataset as PNG files.

    Args:
        dataset: The xarray Dataset containing the 'stimuli' DataArray.
        output_folder: The path to the folder where images will be saved.
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving stimuli to {output_path.resolve()}...")

    # Iterate through each stimulus along the first dimension
    for i, stimulus_image in enumerate(dataset['stimuli']):
        file_path = output_path / f"stimulus_{i}.png"
        # Use .values to get the numpy array from the DataArray and save it
        plt.imsave(file_path, stimulus_image.values, cmap='gray')

    print(f"Finished saving {len(dataset['stimuli'])} stimuli.")