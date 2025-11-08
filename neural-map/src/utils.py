from __future__ import annotations
import pandas as pd

from pathlib import Path
import os

import numpy as np



def load_memmap(path, shape, dtype=np.float32):
    return np.memmap(path, mode='r', dtype=dtype, shape=shape)
from pathlib import Path
import pandas as pd
import os  # <-- you missed this import

AREANAME = {'VISp':'V1','VISl':'LM','VISal':'AL','VISrl':'RL','VISam':'AM','VISpm':'PM'}

def load_index(index_csv_path):
    p = Path(index_csv_path).resolve()
    df = pd.read_csv(p)
    df.attrs["base_dir"] = str(p.parent)
    return df

def find_repo_root(start: str | Path | None = None) -> Path | None:
    p = Path(start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return None

from pathlib import Path
import os

ROOT_DIR = '../PreprocData'  # keep relative to your repo; no trailing slash is fine

def get_trials(index_df, spec_id, area, *, root_dir: str | os.PathLike | None = ROOT_DIR):
    def _norm_area(a):
        try:
            return AREANAME.get(a, a)
        except NameError:
            return a

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

    store_rel = str(df.iloc[0]['store'])
    p = Path(store_rel)

    candidates = []

    # 1) explicit/root default
    if root_dir is not None:
        candidates.append(Path(root_dir))


    resolved = p if p.is_absolute() else next((b / p for b in candidates if (b / p).exists()), None)
    if resolved is None:
        tried = "\n  - " + "\n  - ".join(str(b / p) for b in candidates)
        raise FileNotFoundError(
            "Could not locate data file referenced by index_df['store'].\n"
            f"Requested: {store_rel}\n"
            f"Current working dir: {Path.cwd()}\n"
            "Paths tried:" + tried + "\n\n"
            "Fixes:\n"
            "  • Pass root_dir=... to get_trials\n"
            "  • or set DATA_ROOT\n"
            "  • or load index via load_index('.../index.csv') so base_dir is known"
        )

    path = str(resolved)
    if path.endswith('.zarr'):
        import zarr
        z = zarr.open(path, mode='r')
        arr = z[:] if hasattr(z, 'dtype') else z[( 'data' if 'data' in z.array_keys() else z.array_keys()[0] )][:]
    elif path.endswith('.npz'):
        import numpy as np
        with np.load(path) as npz:
            key = 'data' if 'data' in npz.files else npz.files[0]
            arr = npz[key]
    elif path.endswith('.npy'):
        import numpy as np
        arr = np.load(path, mmap_mode=None)
    elif path.endswith('.nc'):
        import xarray as xr
        arr = xr.load_dataarray(path).values
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
    return int(a.iloc[0]['n_units'])

def get_summary_df(index_df):
    rows = []
    for area, sub in index_df.groupby("area"):
        sids = sub['specimen_id'].astype(int).tolist()
        units = sub['n_units'].astype(int).tolist()
        rows.append({
            "Area": area,
            "Number of Specimen IDs": len(sids),
            "Total Units": int(sum(units)),
            "Units per Specimen ID": units,
        })
    return pd.DataFrame(rows)
