from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
import numpy as np

AREANAME = {'VISp':'V1','VISl':'LM','VISal':'AL','VISrl':'RL','VISam':'AM','VISpm':'PM'}
ROOT_DIR = '../Preproc2'  # keep relative to your repo; no trailing slash is fine

def load_index(index_csv_path):
    p = Path(index_csv_path).resolve()
    df = pd.read_csv(p)
    df.attrs["base_dir"] = str(p.parent)
    return df

def get_trials(index_df, spec_id, area, *, root_dir: str | os.PathLike | None = ROOT_DIR):
    """
    Load the (trials, images, units) array for a given specimen and area from an index CSV.

    Compatible with indexes that use any of these columns for the on-disk array path:
    'store' (old), 'filename' (new), 'filepath', or 'path'.

    Path resolution order for relative paths:
    1) index_df.attrs['base_dir']      (set by load_index)
    2) root_dir argument               (defaults to ROOT_DIR)
    3) $DATA_ROOT environment variable
    """
    def _norm_area(a):
        try:
            return AREANAME.get(a, a)
        except NameError:
            return a

    # Normalize area names (VISp->V1 etc)
    idx = index_df.copy()
    idx['area_norm'] = idx['area'].map(_norm_area)
    a_norm = _norm_area(area)

    df = idx[(idx['specimen_id'].astype(int) == int(spec_id)) & (idx['area_norm'] == a_norm)]
    if df.empty:
        avail = (index_df[index_df['specimen_id'].astype(int) == int(spec_id)]
                 [['specimen_id','area']].drop_duplicates())
        raise ValueError(
            f"No data for specimen_id={spec_id} area='{area}' (norm='{a_norm}').\n{avail}"
        )

    # Support multiple possible column names for the stored file path
    path_col_candidates = ['store', 'filename', 'filepath', 'path']
    path_col = next((c for c in path_col_candidates if c in df.columns), None)
    if path_col is None:
        raise KeyError(
            "Index is missing a file path column. Looked for columns: "
            + ", ".join(path_col_candidates)
        )

    store_rel = str(df.iloc[0][path_col]).strip()
    p = Path(store_rel)

    # Build base candidates (for resolving relative paths)
    candidates = []
    base_dir = index_df.attrs.get("base_dir")
    if base_dir:
        candidates.append(Path(base_dir))
    if root_dir is not None:
        candidates.append(Path(root_dir))
    env_root = os.environ.get("DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Resolve absolute or try each base candidate
    resolved = p if p.is_absolute() else next((b / p for b in candidates if (b / p).exists()), None)
    if resolved is None:
        tried = "\n  - " + "\n  - ".join(str(b / p) for b in candidates) if candidates else " (no base candidates)"
        raise FileNotFoundError(
            "Could not locate data file referenced by index.\n"
            f"Referenced path: {p}\n"
            f"Tried the following locations:{tried}"
        )

    path = str(resolved)
    # Load supported formats
    if path.endswith('.npy'):
        arr = np.load(path, mmap_mode=None)
    else:
        raise ValueError(f"Unsupported store type: {path}")

    return arr  # (trials, images, units)



def get_areas(index_df):
    areas = sorted(set(index_df['area'].tolist()))
    return areas

def get_specimen_ids(index_df, area):
    return index_df[index_df['area'] == area]['specimen_id'].tolist()

def get_units_count(index_df, spec_id, area):
    a = index_df[(index_df['specimen_id'].astype(int)==int(spec_id)) & (index_df['area']==area)]
    if a.empty:
        raise ValueError(f"No row for specimen_id={spec_id}, area={area}")
    return int(a.iloc[0]['unit_count'])

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

def load_memmap(path, shape, dtype=np.float32):
    return np.memmap(path, mode='r', dtype=dtype, shape=shape)