import xarray as xr
import numpy as np
from typing import Optional
from tqdm import tqdm

AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISrl", "VISam"]

def rho(ds: xr.Dataset, unit: int):
    """return for each unit the array of split-half correlation, indexed by time bin"""
    rho_ub = ds["splithalf_r_mean"].values
    return rho_ub[:,unit]

def subset_S(ds: xr.Dataset, area: str) -> list:
    """
    Get specimen IDs for a given visual area.
    """
    area_units_mask = ds.visual_area.values == area
    specimens_in_area = ds.specimen_id.values[area_units_mask]
    return np.unique(specimens_in_area).tolist()

# 1. calcolo la maschera che restituisce tutte le unità appartenenti ad (s,a)

def subset_U(ds: xr.Dataset, specimen: int, area: str) -> np.ndarray:
    """
    Get unit indices for a given specimen and visual area.
    """
    specimen_units = ds.specimen_id.values == specimen
    area_units = ds.visual_area.values == area
    
    mask = specimen_units & area_units
    
    return np.where(mask)[0]

# 2. calcolo la mediana di rho fra tute quelle unità

def median(ds: xr.Dataset, specimen: int, area: str) -> np.array:

    """compute the median (np.array indexed by time bin) of rho over all units in a given specimen and area"""

    units = subset_U(ds, specimen, area)
    rho_matrix = np.array([rho(ds, u) for u in units])
    median_rho = np.median(rho_matrix, axis=0)
    return median_rho

# 3. calcolo gli estremi della time window definita dalla più lunga sequenza di bin consecutivi in cui la mediana è >= 0.3

def time_window(ds: xr.Dataset, specimen: int, area: str, threshold: float = 0.3) -> Optional[tuple]:
    """
    Compute the time window defined by the longest sequence of consecutive bins where the median rho is >= threshold.
    Returns a tuple (start_bin, end_bin) or None if no such window exists.
    """
    median_rho = median(ds, specimen, area)
    above_threshold = median_rho >= threshold
    
    max_len = 0
    current_len = 0
    start_index = -1
    best_start = -1
    
    for i, val in enumerate(above_threshold):
        if val:
            if current_len == 0:
                start_index = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                best_start = start_index
        else:
            current_len = 0
    
    if max_len == 0:
        return None
    
    return (best_start, best_start + max_len - 1)


# calcolo per ogni unità la risposta media al trial in quella time window

def mean_response_in_window(ds: xr.Dataset, unit: int, start_bin: int, end_bin: int) -> np.ndarray:
    """
    Compute the response of a unit within a specified time window for each paired trial and frame_id.
    Returns a 2D numpy array of shape (T, F) where T is the number of trials and F is the number of frame_ids.
    """

    da = ds["neural_data"].isel({"units": unit, "time_relative_to_stimulus_onset": slice(start_bin, end_bin + 1)}).mean("time_relative_to_stimulus_onset")
    da = da.transpose("trials", "frame_id")
    return da.astype("float32").values  # (T, F)


def pipeline(ds: xr.Dataset, specimen: int, area: str, threshold: float = 0.3) -> Optional[np.ndarray]:
    """
    Complete pipeline to compute the trial mean responses for all units in a given specimen and area
    within the time window defined by the longest sequence of consecutive bins where the median rho is >= threshold.
    Returns a 3D numpy array of shape (U, T, F) where U is the number of units, T is the number of trials, and F is the number of frame_ids.
    Returns None if no valid time window exists.
    """
    time_win = time_window(ds, specimen, area, threshold)
    if time_win is None:
        return None
    
    start_bin, end_bin = time_win
    units = subset_U(ds, specimen, area)

    responses = np.array([mean_response_in_window(ds, u, start_bin, end_bin) for u in units])

    return responses  # (U, T, F)


def subset_S75pc(ds: xr.Dataset, area: str) -> list:
    """
    Get specimen IDs for a given visual area that have at least 75 percentile of the number of units in that area.
    """
    specimens = subset_S(ds, area)
    unit_counts = [len(subset_U(ds, specimen, area)) for specimen in specimens]
    threshold = np.percentile(unit_counts, 75)
    print(f"75th percentile of unit counts in {area}: {threshold}")
    selected_specimens = [specimen for specimen, count in zip(specimens, unit_counts) if count >= threshold]
    return selected_specimens


def pipeline_all(ds: xr.Dataset, area: str, threshold: float = 0.3) -> Optional[np.ndarray]:
    """
    use the above pipeline to compute the responses for all specimens in a given area then subselect those specimens with at least 75 percentile of units
    concatenate all the results along the unit dimension and return a single array of shape (U_total, T, F)
    Returns None if no valid specimens exist.
    """

    selected_specimens = subset_S75pc(ds, area)
    if not selected_specimens:
        return None
    
    all_responses = []
    for specimen in selected_specimens:
        responses = pipeline(ds, specimen, area, threshold)
        if responses is not None:
            all_responses.append(responses)
    
    if not all_responses:
        return None
    
    concatenated_responses = np.concatenate(all_responses, axis=0)  # concatenate along the unit dimension

    # print the number of unique units in concatenated_responses
    unique_units = set(concatenated_responses[:, 0, 0])  # assuming unit IDs are in the first dimension
    print(f"Number of unique units in concatenated_responses: {len(unique_units)}")

    return concatenated_responses  # (U_total, T, F)


def pipeline_all_correct_order(ds: xr.Dataset, area: str, threshold: float = 0.3) -> Optional[np.ndarray]:
    """
    first use the pipeline above to get all responses for the pair (specimen, area), then compute the new numeber of resulting units for each specimen, then subselect those specimens with at least 75 percentile of units.
    """

    all_responses = {}

    # for all specimens in area
    specimens = subset_S(ds, area)
    for specimen in tqdm(specimens, desc="Processing specimens"):
        # compute the avg response over the reliable time window (for each trial)
        resp = pipeline(ds, specimen, area, threshold)
        if resp is not None:
            all_responses[specimen] = (resp, resp.shape[0])  # specimen: (U, T, F), number of units

    if not all_responses:
        return None


    # Compute the number of resulting units for each specimen
    unit_counts = [count for _, count in all_responses.values()]
    print(f"Unit counts per specimen: {unit_counts}")
    if not unit_counts:
        return None
    
    # Compute the 75th percentile threshold
    percentile_threshold = np.percentile(unit_counts, 75)
    print(f"75th percentile of unit counts in {area}: {percentile_threshold}")

    # Subselect specimens with at least 75 percentile of units
    selected_responses = [resp for resp, count in all_responses.values() if count >= percentile_threshold]

    if not selected_responses:
        return None

    # Concatenate all the results along the unit dimension
    return np.concatenate(selected_responses, axis=0)