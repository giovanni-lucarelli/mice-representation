"""Allen packaged dataset (.zarr) viewer utilities.

This module provides a small CLI to:
- open a converted .zarr dataset (as produced from packaged PKL),
- print a concise summary of data variables and dimensions,
- visualize a grid of stimuli images,
- visualize neural responses, optionally filtered by visual area and trial subset.

Notes
-----
- We intentionally import xarray and zarr lazily inside functions that need them
  so regular training workflows do not pull extra deps at import time.
- All plotting is done with matplotlib and saved to disk unless --show is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import xarray as xr
import numpy as np


def _open_zarr(store_path: str):
    try:
        return xr.open_zarr(store_path, consolidated=True)
    except Exception:
        return xr.open_zarr(store_path, consolidated=False)
    
def _ensure_numpy(array_like):
    try:
        return np.asarray(array_like)
    except Exception:
        return array_like
    
def _check_path(path: str | list[str]):
    if isinstance(path, str):
        assert Path(path).exists(), f"Provided path does not exist: {path}"
    elif isinstance(path, list):
        for p in path:
            assert Path(p).exists(), f"Provided path does not exist: {p}"

class AllenDataViewer:
    def __init__(self, zarr_dir: str):
        
        _check_path(zarr_dir)
        
        self.zarr_dir = zarr_dir
        self.init_dataset()
        
    def init_dataset(self):
        self.ds = _open_zarr(self.zarr_dir)
        self.dims_summary = {k: int(v) for k, v in getattr(self.ds, "sizes", {}).items()}
        self.vars_list = list(getattr(self.ds, "data_vars", {}).keys())
        self.summarize_dataset()
        
    def summarize_dataset(self):
        self.summary = ""
        
        self.summary += f"Dataset opened from: {self.zarr_dir}\n"
        self.summary += f"Dims: {self.dims_summary}\n"
        self.summary += f"Variables: {self.vars_list}\n"
        # Best-effort peek
        for key in ("stimuli", "neural_data"):
            if key in self.ds:
                var = self.ds[key]
                shape = getattr(var, "shape", None)
                self.summary += f"{key}: shape={shape}\n"
        print(self.summary)
        return self.summary

    def plot_stimuli_grid(self, out_path: Optional[str] = None, max_images: int = 12, show: bool = False) -> str:
        # Lazy import to avoid hard dependency at module import time
        import matplotlib.pyplot as plt
        
        if "stimuli" not in self.ds:
            raise RuntimeError("Dataset does not contain a 'stimuli' variable")

        stim = self.ds["stimuli"].values  # Load into memory for simple viewing

        stim = _ensure_numpy(stim)
        num = int(min(max_images, stim.shape[0]))
        grid_x = int(np.ceil(np.sqrt(num)))
        grid_y = int(np.ceil(num / grid_x))
        fig_size = (2*grid_x, 2*grid_y)
        fig, axes = plt.subplots(grid_y, grid_x, figsize=fig_size, squeeze=False)
        for i in range(grid_x * grid_y):
            ax = axes.ravel()[i]
            ax.axis("off")
            if i >= num:
                continue

            img = stim[i]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            elif img.ndim == 3 and img.shape[-1] in (1, 3):
                if img.shape[-1] == 1:
                    ax.imshow(img[..., 0], cmap="gray")
                else:
                    ax.imshow(img)
            else:
                # Pad/reshape best-effort for unexpected layout (e.g., C,H,W)
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    ax.imshow(np.moveaxis(img, 0, -1).squeeze())
                else:
                    ax.imshow(img.squeeze(), cmap="gray")
        fig.tight_layout()

        out = out_path or "stimuli_grid.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved stimuli grid to {out}")
        return out
    
    def plot_neural_responses(self, out_path: Optional[str] = None, visual_area: Optional[str] = None, max_units: int = 64, max_trials: int = 100, show: bool = False) -> str:
        # Lazy import to avoid hard dependency at module import time
        import matplotlib.pyplot as plt

        if "neural_data" not in self.ds:
            raise RuntimeError("Dataset does not contain a 'neural_data' variable")

        nd = self.ds["neural_data"]
        data = nd
        dims = list(getattr(nd, "dims", []))

        # Prefer explicit names from the dataset layout when present
        if "unit_id" in dims:
            unit_dim = "unit_id"
        else:
            unit_dim_candidates: Sequence[str] = [d for d in dims if d.lower() in ("unit", "units", "neurons", "cells")]
            unit_dim = unit_dim_candidates[0] if unit_dim_candidates else (dims[0] if len(dims) >= 1 else None)

        trial_dim = None
        if "trial" in dims:
            trial_dim = "trial"
        elif "trials" in dims:
            trial_dim = "trials"
        else:
            trial_dim_candidates: Sequence[str] = [d for d in dims if d.lower() in ("trial", "trials")]
            trial_dim = trial_dim_candidates[0] if trial_dim_candidates else (dims[1] if len(dims) >= 2 else None)

        if visual_area is not None and unit_dim is not None and unit_dim in getattr(nd, "dims", ()): 
            try:
                # Prefer 'visual_area' provided at the variable level; fall back to dataset
                va = None
                if "visual_area" in getattr(nd, "coords", {}):
                    va = nd.coords["visual_area"].values
                elif "visual_area" in getattr(self.ds, "data_vars", {}):
                    va = self.ds["visual_area"].values
                if va is not None:
                    mask = np.array([str(v) == str(visual_area) for v in va])
                    # Wrap mask as DataArray for correct alignment
                    mask_da = xr.DataArray(mask, dims=(unit_dim,))
                    data = nd.where(mask_da, drop=True)
            except Exception:
                pass

        # Reduce to at most max_units along unit dimension early (if known)
        try:
            if unit_dim is not None and unit_dim in data.dims:
                data = data.isel({unit_dim: slice(0, max_units)})
        except Exception as e:
            pass

        # Identify non-unit dims and time-like dims
        def _is_time_dim(name: str) -> bool:
            name_l = name.lower()
            return any(token in name_l for token in ("time", "frame", "bin", "timestep", "ms", "sec"))

        other_dims = [d for d in list(getattr(data, "dims", [])) if d != unit_dim]
        time_dims = [d for d in other_dims if _is_time_dim(d)]

        # Average across time-like dims if present
        if len(time_dims) > 0:
            try:
                data = data.mean(dim=tuple(time_dims), keep_attrs=True)
            except Exception:
                pass

        other_dims = [d for d in list(getattr(data, "dims", [])) if d != unit_dim]
        if len(other_dims) == 0:
            try:
                data = data.expand_dims({"trial": 1})
            except Exception:
                pass
        elif len(other_dims) == 1 and other_dims[0] not in ("trial", "trials"):
            try:
                data = data.rename({other_dims[0]: "trial"})
            except Exception:
                pass
        else:
            try:
                data = data.stack(trial=tuple(other_dims))
            except Exception:
                # Best-effort fallback: leave as is
                pass

        try:
            if "trial" in data.dims:
                data = data.isel({"trial": slice(0, max_trials)})
        except Exception:
            pass

        try:
            if unit_dim is not None and unit_dim in data.dims and "trial" in data.dims:
                data = data.transpose(unit_dim, "trial", ...)
        except Exception:
            pass

        mat = _ensure_numpy(data.values)
        # Ensure 2D for imshow
        if mat.ndim > 2:
            mat = np.reshape(mat, (mat.shape[0], -1))
        if mat.ndim == 1:
            mat = mat[:, None]
        mat_plot = mat

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat_plot, aspect="auto", interpolation="nearest", cmap="viridis")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Units")
        title = "Neural responses"
        if visual_area:
            title += f" — area: {visual_area}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out = out_path or "neural_responses.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved neural responses plot to {out}")
        return out

    def plot_neuropixels_reliability(self,
                                     out_path: Optional[str] = None,
                                     visual_areas: Optional[Sequence[str]] = None,
                                     fraction_of_max: bool = False,
                                     time_step_ms: float = 1.0,
                                     show: bool = False) -> str:
        """Plot split-half reliability over time for each visual area.

        This mirrors the notebook helper that plots median ± SEM across units,
        computed per visual area, over time.

        Assumptions/best effort:
        - Looks for a variable named 'splithalf_r_mean' in the dataset.
        - Expects a unit-like dimension (e.g., 'unit'/'units'/'neurons'/'cells').
        - Expects a time-like dimension containing 'time'/'frame'/'bin' tokens.
        - The 'visual_area' coordinate is expected to be aligned to the unit dimension.
        """
        import matplotlib.pyplot as plt

        # Locate reliability variable (prefer exact key present in your store)
        if "splithalf_r_mean" in self.ds:
            rel = self.ds["splithalf_r_mean"]
        else:
            # Fallback: search approximately named variables
            reliability_var_name = None
            for name in getattr(self.ds, "data_vars", {}).keys():
                name_l = str(name).lower()
                if ("split" in name_l or "reliab" in name_l) and ("mean" in name_l or "r_" in name_l or name_l.endswith("_r")):
                    reliability_var_name = name
                    break
            if reliability_var_name is None:
                raise RuntimeError("Could not find a reliability variable (e.g., 'splithalf_r_mean') in the dataset")
            rel = self.ds[reliability_var_name]

        # Identify dimensions
        dims = list(getattr(rel, "dims", []))

        def _is_time_dim(name: str) -> bool:
            name_l = name.lower()
            return any(token in name_l for token in ("time", "frame", "bin", "timestep", "ms", "sec"))

        # Prefer explicit dimension names present in your zarr layout
        if "unit_id" in dims:
            unit_dim = "unit_id"
        else:
            unit_dim_candidates: Sequence[str] = [d for d in dims if d.lower() in ("unit", "units", "neurons", "cells")]
            unit_dim = unit_dim_candidates[0] if unit_dim_candidates else (dims[0] if len(dims) >= 1 else None)
        if "time_relative_to_stimulus_onset" in dims:
            time_dim = "time_relative_to_stimulus_onset"
        else:
            time_dim_candidates: Sequence[str] = [d for d in dims if _is_time_dim(d)]
            time_dim = time_dim_candidates[0] if time_dim_candidates else (dims[1] if len(dims) >= 2 else None)

        if unit_dim is None or time_dim is None:
            # Try alternative heuristic: assume two-dim array, pick non-time as unit
            if len(dims) == 2:
                if _is_time_dim(dims[0]):
                    time_dim, unit_dim = dims[0], dims[1]
                elif _is_time_dim(dims[1]):
                    unit_dim, time_dim = dims[0], dims[1]
        if unit_dim is None or time_dim is None:
            raise RuntimeError(f"Unable to infer unit/time dimensions for reliability variable with dims={dims}")

        # Visual areas
        if "visual_area" in getattr(rel, "coords", {}):
            va_coord = rel.coords["visual_area"]
        elif "visual_area" in getattr(self.ds, "data_vars", {}):
            va_coord = self.ds["visual_area"]
        else:
            raise RuntimeError("Reliability data lacks 'visual_area' variable/coord aligned to the unit dimension")

        # If no explicit visual_areas provided, derive sorted unique values
        if visual_areas is None:
            try:
                values = np.unique(np.asarray(va_coord).astype(str))
                visual_areas = list(sorted(values))
            except Exception:
                raise RuntimeError("Failed to derive list of visual areas from 'visual_area' coordinate")

        # Build color map for areas
        cmap = plt.cm.get_cmap("tab10", max(10, len(visual_areas)))
        color_map = {v: cmap(i % cmap.N) for i, v in enumerate(visual_areas)}

        # Build time axis
        if time_dim in getattr(rel, "coords", {}):
            try:
                time_values = np.asarray(rel.coords[time_dim]).astype(float)
            except Exception:
                time_values = np.arange(rel.sizes[time_dim], dtype=float) * float(time_step_ms)
        elif "time_relative_to_stimulus_onset" in getattr(self.ds, "data_vars", {}):
            try:
                time_values = np.asarray(self.ds["time_relative_to_stimulus_onset"]).astype(float)
            except Exception:
                time_values = np.arange(rel.sizes[time_dim], dtype=float) * float(time_step_ms)
        else:
            time_values = np.arange(rel.sizes[time_dim], dtype=float) * float(time_step_ms)

        # Infer unit: seconds when coming from dataset coordinates (as in Allen data),
        # otherwise treat as milliseconds when synthesized via time_step_ms
        use_seconds = False
        try:
            if (time_dim in getattr(rel, "coords", {})) or ("time_relative_to_stimulus_onset" in getattr(self.ds, "data_vars", {})):
                diffs = np.diff(np.asarray(time_values, dtype=float))
                median_step = float(np.median(diffs)) if diffs.size > 0 else None
                max_t = float(time_values[-1]) if len(time_values) > 0 else 0.0
                # Heuristic: dataset coords are in seconds (dt <= 1s and range <= 10s)
                use_seconds = (median_step is None) or (median_step <= 1.0 and max_t <= 10.0)
            else:
                use_seconds = False
        except Exception:
            use_seconds = False

        fig, ax = plt.subplots(figsize=(8, 5))

        # Compute median and SEM over units, per area
        for area in visual_areas:
            try:
                # Mask units by visual area
                mask = (va_coord.astype(str) == str(area))
                # Align mask to unit dimension as DataArray
                mask_da = xr.DataArray(np.asarray(mask), dims=(unit_dim,))
                rel_area = rel.where(mask_da, drop=True)

                # Median across units
                median_da = rel_area.median(dim=unit_dim, skipna=True)

                # SEM across units: std / sqrt(n)
                n_effective = rel_area.count(dim=unit_dim)
                std_da = rel_area.std(dim=unit_dim, skipna=True, ddof=1)
                sem_da = std_da / np.sqrt(n_effective.clip(min=1))

                median_vals = _ensure_numpy(median_da.transpose(time_dim, ...).values)
                sem_vals = _ensure_numpy(sem_da.transpose(time_dim, ...).values)

                if median_vals.ndim > 1:
                    median_vals = np.squeeze(median_vals)
                if sem_vals.ndim > 1:
                    sem_vals = np.squeeze(sem_vals)

                if fraction_of_max:
                    max_val = np.nanmax(median_vals) if np.size(median_vals) > 0 else 1.0
                    if max_val == 0 or not np.isfinite(max_val):
                        max_val = 1.0
                    median_vals = median_vals / max_val
                    sem_vals = sem_vals / max_val

                color = color_map.get(area, None)
                ax.fill_between(time_values,
                                median_vals + sem_vals,
                                median_vals - sem_vals,
                                label=str(area),
                                alpha=0.8,
                                color=color)
            except Exception as e:
                print(f"Skipping area '{area}' due to error: {e}")
                continue

        ax.set_xlabel("Time relative to stimulus onset (s)" if use_seconds else "Time relative to stimulus onset (ms)")
        ax.set_ylabel("Fraction of Maximum Split-half Reliability (Pearson's R)" if fraction_of_max else "Split-half Reliability (Pearson's R)")
        ax.legend(loc="upper left", frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Ticks every 40 ms (0.04 s) if possible, aligned to data range
        try:
            if len(time_values) > 0:
                min_t = float(time_values[0])
                max_t = float(time_values[-1])
                step = 0.04 if use_seconds else 40.0  # 40 ms in seconds vs milliseconds
                # Start and end ticks inside data bounds to avoid empty ticks at edges
                start = float(np.ceil((min_t + 1e-12) / step) * step)
                end = float(np.floor((max_t - 1e-12) / step) * step)
                if end < start:
                    # Fallback: single tick at nearest value within range
                    start = end = float(np.round(min_t / step) * step)
                tick_vals = np.arange(start, end + step * 0.5, step)
                ax.set_xticks(tick_vals)
                ax.set_xlim(min_t, max_t)
        except Exception:
            pass

        fig.tight_layout()
        out = out_path or ("neural_reliability_fraction.png" if fraction_of_max else "neural_reliability.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved reliability plot to {out}")
        return out

