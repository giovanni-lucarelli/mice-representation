import xarray as xr
import numpy as np
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISrl", "VISam"]

# ---------- helpers ----------

def _unit_mask(ds: xr.Dataset, specimen: int, area: str) -> np.ndarray:
    """Boolean mask over the 'units' axis for (specimen, area)."""
    return (ds.specimen_id.values == specimen) & (ds.visual_area.values == area)

def subset_S(ds: xr.Dataset, area: str) -> List[int]:
    """Get specimen IDs for a given visual area."""
    return np.unique(ds.specimen_id.values[ds.visual_area.values == area]).tolist()


# ---------- new / updated helpers ----------

def _reliable_unit_mask(
    ds: xr.Dataset,
    specimen: int,
    area: str,
    splithalf_r_thresh: float = 0.5,
    run_pval_ns_thresh: float | None = 0.05,
) -> np.ndarray:
    """
    Boolean mask over 'units' for (specimen, area) that keeps only:
      - units whose *max over time* of splithalf_r_mean >= splithalf_r_thresh
      - (optionally) units NOT significantly modulated by running:
            run_pval_ns > run_pval_ns_thresh OR NaN
    Returns a boolean array of length ds.dims['units'].
    """
    base = _unit_mask(ds, specimen, area)
    if not base.any():
        return base  # all False

    # reliability per unit: max over time for units in (specimen, area)
    rho = ds["splithalf_r_mean"].values  # (time_bins, units)
    # handle NaNs by treating them as -inf when taking max
    per_unit_max = np.nanmax(np.where(np.isfinite(rho), rho, -np.inf)[:, base], axis=0)
    reliable_local = per_unit_max >= splithalf_r_thresh

    # start with all-False, then place reliable flags at the selected unit indices
    out = np.zeros(ds.dims["units"], dtype=bool)
    idx = np.where(base)[0]
    out[idx[reliable_local]] = True

    # optional: running modulation filter (paper keeps units with p > 0.05 or NaN)
    if run_pval_ns_thresh is not None and "run_pval_ns" in ds:
        p = ds["run_pval_ns"].values  # per-unit array
        run_keep = (p > run_pval_ns_thresh) | ~np.isfinite(p)
        out &= run_keep

    return out

# ---------- updated rho / median / time-window ----------

def median(
    ds: xr.Dataset, 
    specimen: int, 
    area: str, 
    min_median_thresh: float = 0.3,
    splithalf_r_thresh: float = 0.5,
    run_pval_ns_thresh: float | None = 0.05,
) -> np.ndarray:
    """
    Median over *filtered* units (reliable and optionally non-running-modulated).
    Returns shape (time_bins,). Empty array if no units survive.
    """
    mask = _reliable_unit_mask(ds, specimen, area, splithalf_r_thresh, run_pval_ns_thresh)
    if not mask.any():
        return np.array([])

    rho = ds["splithalf_r_mean"].values  # (time_bins, units)
    return np.nanmedian(rho[:, mask], axis=1)

def time_window(
    ds: xr.Dataset, 
    specimen: int, 
    area: str, 
    min_median_thresh: float = 0.3,
    splithalf_r_thresh: float = 0.5,
    run_pval_ns_thresh: float | None = 0.05,
) -> Optional[Tuple[int, int]]:
    """
    Longest consecutive bins where median (across *filtered* units) >= min_median_thresh.
    """
    med = median(ds, specimen, area, min_median_thresh, splithalf_r_thresh, run_pval_ns_thresh)
    if med.size == 0:
        return None
    return _longest_true_run(med >= min_median_thresh)

# ---------- updated responses ----------

def pipeline(
    ds: xr.Dataset, 
    specimen: int, 
    area: str, 
    min_median_thresh: float = 0.3,
    splithalf_r_thresh: float = 0.5,
    run_pval_ns_thresh: float | None = 0.05,
) -> Optional[np.ndarray]:
    """
    Returns responses with shape (T, F, U) for all *filtered* units in (specimen, area),
    averaged over the reliable time window derived from filtered units.
    """
    # compute reliable unit mask first (like paper's r_cond [+ speed_cond])
    unit_mask = _reliable_unit_mask(ds, specimen, area, splithalf_r_thresh, run_pval_ns_thresh)
    if not unit_mask.any():
        return None

    # find longest reliable time window using *filtered* units
    tw = time_window(ds, specimen, area, min_median_thresh, splithalf_r_thresh, run_pval_ns_thresh)
    if tw is None:
        return None
    start_bin, end_bin = tw

    da = (
        ds["neural_data"]
        .isel(units=unit_mask, time_relative_to_stimulus_onset=slice(start_bin, end_bin + 1))
        .mean("time_relative_to_stimulus_onset")
        .transpose("trials", "frame_id", "units")
        .astype("float32")
    )
    # if no surviving units for any reason, bail
    if da.sizes.get("units", 0) == 0:
        return None
    return da.values  # (T, F, U)

def pipeline_all_specimens(
    ds: xr.Dataset, 
    area: str, 
    min_median_thresh: float = 0.3,
    splithalf_r_thresh: float = 0.5,
    run_pval_ns_thresh: float | None = 0.05,
) -> Optional[np.ndarray]:
    """
    Run pipeline for each specimen in area using *per-unit reliability >= splithalf_r_thresh*,
    then keep specimens with unit_count >= 75th percentile (post-filter), save selected,
    and return concatenation along unit axis.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    all_responses: Dict[int, np.ndarray] = {}

    for specimen in tqdm(subset_S(ds, area), desc=f"Processing specimens in {area}"):
        resp = pipeline(
            ds, specimen, area, 
            min_median_thresh=min_median_thresh, 
            splithalf_r_thresh=splithalf_r_thresh,
            run_pval_ns_thresh=run_pval_ns_thresh
        )
        if resp is not None:
            all_responses[specimen] = resp

    if not all_responses:
        return None

    unit_counts = {specimen: resp.shape[2] for specimen, resp in all_responses.items()}
    perc_thr = np.percentile(list(unit_counts.values()), 75)

    selected = {s: r for s, r in all_responses.items() if unit_counts[s] >= perc_thr}
    if not selected:
        return None

    index_filepath = os.path.join(data_dir, f"{area}_index.csv")
    with open(index_filepath, "w") as f:
        f.write("area,specimen_id,unit_count,filename\n")
        for specimen, resp in selected.items():
            unit_count = unit_counts[specimen]
            filename = f"{area}_{specimen}_responses.npy"
            filepath = os.path.join(data_dir, filename)
            np.save(filepath, resp)
            f.write(f"{area},{specimen},{unit_count},{filepath}\n")

    return np.concatenate(list(selected.values()), axis=2)


def _longest_true_run(x: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Fast longest consecutive True run in a 1-D boolean array.
    Returns (start, end) inclusive, or None.
    """
    if x.size == 0 or not x.any():
        return None
    # Add sentinels to detect edges
    dx = np.diff(np.concatenate(([0], x.view(np.int8), [0])))
    starts = np.where(dx == 1)[0]
    ends   = np.where(dx == -1)[0] - 1
    lengths = ends - starts + 1
    i = np.argmax(lengths)
    return int(starts[i]), int(ends[i])


def get_summary_df(index_df):
    rows = []
    for area, sub in index_df.groupby("area"):
        sids = sub['specimen_id'].astype(int).tolist()
        units = sub['unit_count'].astype(int).tolist()
        rows.append({
            "Area": area,
            "Number of Specimen IDs": len(sids),
            "Total Units": int(sum(units)),
            "Units per Specimen ID": units,
        })
    return pd.DataFrame(rows)

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