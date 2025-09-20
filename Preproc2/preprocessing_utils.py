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

# ---------- rho / median / time-window ----------

def median(ds: xr.Dataset, specimen: int, area: str) -> np.ndarray:
    """
    Compute median (over units) of split-half rho for (specimen, area).
    Vectorized: slice once and median along unit axis.
    Returns shape (time_bins,).
    """
    mask = _unit_mask(ds, specimen, area)
    # splithalf_r_mean expected dims: (time_bins, units)
    rho_ub = ds["splithalf_r_mean"].values  # ndarray
    if not mask.any():
        return np.array([])
    # Take only the masked units and median across units (axis=1)
    return np.median(rho_ub[:, mask], axis=1)

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

def time_window(ds: xr.Dataset, specimen: int, area: str, threshold: float = 0.3) -> Optional[Tuple[int, int]]:
    """
    Longest consecutive bins where median rho >= threshold.
    """
    med = median(ds, specimen, area)
    if med.size == 0:
        return None
    return _longest_true_run(med >= threshold)

# ---------- responses ----------

def pipeline(ds: xr.Dataset, specimen: int, area: str, threshold: float = 0.3) -> Optional[np.ndarray]:
    """
    Returns responses with shape (T, F, U) for all units in (specimen, area),
    averaged over the reliable time window. Fully vectorized over units.
    """
    tw = time_window(ds, specimen, area, threshold)
    if tw is None:
        return None
    start_bin, end_bin = tw

    mask = _unit_mask(ds, specimen, area)
    if not mask.any():
        return None

    # Slice all units + time window once, reduce over time, then order dims
    da = (
        ds["neural_data"]
        .isel(units=mask, time_relative_to_stimulus_onset=slice(start_bin, end_bin + 1))
        .mean("time_relative_to_stimulus_onset")
        .transpose("trials", "frame_id", "units")
        .astype("float32")
    )
    return da.values  # (T, F, U)

def pipeline_all_specimens(ds: xr.Dataset, area: str, threshold: float = 0.3) -> Optional[np.ndarray]:
    """
    Run pipeline for each specimen in area, then select specimens with
    unit_count >= 75th percentile. Save only the selected ones, and return
    concatenated responses across them along the unit axis.
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    all_responses: Dict[int, np.ndarray] = {}

    # Compute responses for each specimen
    for specimen in tqdm(subset_S(ds, area), desc=f"Processing specimens in {area}"):
        resp = pipeline(ds, specimen, area, threshold)
        if resp is not None:
            all_responses[specimen] = resp

    if not all_responses:
        return None

    # Compute unit counts and 75th percentile
    unit_counts = {specimen: resp.shape[2] for specimen, resp in all_responses.items()}
    perc_thr = np.percentile(list(unit_counts.values()), 75)

    # Filter specimens
    selected = {s: r for s, r in all_responses.items() if unit_counts[s] >= perc_thr}
    if not selected:
        return None

    # Save index + npy files ONLY for selected specimens
    index_filepath = os.path.join(data_dir, f"{area}_index.csv")
    with open(index_filepath, "w") as f:
        f.write("area,specimen_id,unit_count,filename\n")
        for specimen, resp in selected.items():
            unit_count = unit_counts[specimen]
            filename = f"{area}_{specimen}_responses.npy"
            filepath = os.path.join(data_dir, filename)
            np.save(filepath, resp)
            f.write(f"{area},{specimen},{unit_count},{filepath}\n")

    # Concatenate selected responses along unit axis (last axis)
    return np.concatenate(list(selected.values()), axis=2)

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