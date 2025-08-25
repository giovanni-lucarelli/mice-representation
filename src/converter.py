import sys, os, pandas as pd
import xarray as xr


def _flatten_multiindex(ds: xr.Dataset) -> xr.Dataset:
    try:
        import pandas as _pd
        # Reset any MultiIndex coordinates to regular coordinates
        for name, index in getattr(ds, "indexes", {}).items():
            if isinstance(index, _pd.MultiIndex):
                ds = ds.reset_index(name)
    except Exception:
        pass
    return ds


def _coerce_to_dataset(obj) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name="data")
    if hasattr(obj, "to_xarray"):
        xo = obj.to_xarray()
        if isinstance(xo, xr.Dataset):
            return xo
        if isinstance(xo, xr.DataArray):
            return xo.to_dataset(name="data")
    if isinstance(obj, dict):
        variables = {}
        datasets = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, xr.Dataset):
                datasets.append(v)
                continue
            if isinstance(v, xr.DataArray):
                variables[key] = v
                continue
            if hasattr(v, "to_xarray"):
                vx = v.to_xarray()
                if isinstance(vx, xr.Dataset):
                    datasets.append(vx)
                elif isinstance(vx, xr.DataArray):
                    variables[key] = vx
                continue
            try:
                import numpy as np
                import pandas as _pd
                if isinstance(v, _pd.DataFrame):
                    dfx = v.reset_index()
                    dset = dfx.to_xarray()
                    # Prefix to avoid collisions across dict entries
                    prefixed = {f"{key}_{name}": dset[name] for name in dset.data_vars}
                    datasets.append(xr.Dataset(prefixed))
                    continue
                if isinstance(v, _pd.Series):
                    s = v.reset_index(drop=False)
                    dfx = s.to_frame(name=key)
                    dset = dfx.to_xarray()
                    datasets.append(dset)
                    continue
                if isinstance(v, (list, tuple, np.ndarray)):
                    variables[key] = xr.DataArray(v)
                    continue
            except Exception:
                pass
        base = xr.Dataset(variables) if variables else xr.Dataset()
        if datasets:
            datasets.insert(0, base)
            return xr.merge(datasets, combine_attrs="drop_conflicts")
        return base
    return None

print(f"Converting {len(sys.argv[1:])} files")
for p in sys.argv[1:]:
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue
    obj = pd.read_pickle(p)
    ds = _coerce_to_dataset(obj)
    if ds is None:
        print(f"Not supported: {type(obj)} per {p}")
        continue
    ds = _flatten_multiindex(ds)
    stem = os.path.splitext(p)[0]
    out = stem + ".zarr"
    ds.to_zarr(out, mode="w")
    # Consolidate metadata to speed up reads and avoid open_zarr warnings
    try:
        import zarr as _zarr  # type: ignore
        _zarr.consolidate_metadata(out)
    except Exception:
        pass
    print(f"OK: {p} -> {out}")
    break

print("File successfully converted")