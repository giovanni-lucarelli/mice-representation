import os, json
from typing import Optional, Dict
import numpy as np
from .mouse_params import PATCH_SIZE
from config import ROOT

def plot_csf_grid_log(
    log_path: str,
    metric: str = "total_error",
    sf: Optional[float] = None,
    cmap: str = "magma",
    annotate_min: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    patch_size: Optional[int] = None,
    plot_mode: str = "heatmap",
    facet_cols: int = 3,
    annotate_values: bool = False,
):
    """
    Plot results of CSF fitting from the detailed grid-search log JSON.

    Parameters:
    - log_path: path to "*_grid_log.json" created by fit_mouse_csf_params
    - metric: "total_error" (default) or "per_sf_error" to plot a specific SF error
    - sf: spatial frequency (e.g., 0.31) if metric == "per_sf_error"
    - cmap: matplotlib colormap name
    - annotate_min: whether to mark the minimum error location
    - save_path: optional output path for the plot image (PNG). If None, derive from log_path
    - show: whether to display the plot via plt.show()
    - patch_size: when plot_mode="heatmap", select which patch size to slice; if None, pick best
    - plot_mode: "heatmap" (2D slice over blur_sigma x noise_std),
                 "scatter3d" (3D scatter over blur_sigma x noise_std x patch_size),
                 or "facet" (multipanel heatmaps for each patch size)

    Returns:
    - (ax, Z): matplotlib Axes and the 2D array of scores (noise_std x blur_sigma)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required to plot the CSF grid log. Please install it.") from e

    with open(log_path, "r") as f:
        records = json.load(f)

    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Grid log JSON appears empty or malformed.")

    # If 3D scatter is requested, use all patch sizes; otherwise we may slice by patch
    all_patches = sorted({int(r.get("patch_size", PATCH_SIZE)) for r in records})
    if plot_mode == "heatmap":
        # If records contain multiple patch sizes, filter by requested or best
        if patch_size is not None:
            chosen_patch = int(patch_size)
        else:
            if len(all_patches) <= 1:
                chosen_patch = all_patches[0]
            else:
                # Choose the patch size with globally minimal metric
                # Define the metric helper now that we have all dependencies
                import numpy as _np
                def _get_per_sf_err(per_sf_error_dict: Dict[str, float], target_sf: float) -> float:
                    try:
                        pairs = [(_np.float64(k), float(v)) for k, v in per_sf_error_dict.items()]
                    except Exception:
                        key = str(target_sf)
                        if key in per_sf_error_dict:
                            return float(per_sf_error_dict[key])
                        raise
                    sfs = _np.array([p[0] for p in pairs], dtype=_np.float64)
                    vals = _np.array([p[1] for p in pairs], dtype=_np.float64)
                    idx = int(_np.argmin(_np.abs(sfs - target_sf)))
                    return float(vals[idx])

                def _metric_value(rec: Dict[str, float]) -> float:
                    if metric == "total_error":
                        return float(rec.get("total_error", np.nan))
                    elif metric == "per_sf_error":
                        if sf is None:
                            raise ValueError("When metric='per_sf_error', you must provide sf (e.g., 0.31)")
                        per_sf = rec.get("per_sf_error", {})
                        return _get_per_sf_err(per_sf, float(sf))
                    else:
                        raise ValueError("metric must be 'total_error' or 'per_sf_error'")

                best_rec = min(records, key=_metric_value)
                chosen_patch = int(best_rec.get("patch_size", PATCH_SIZE))
        records = [r for r in records if int(r.get("patch_size", PATCH_SIZE)) == chosen_patch]

    # Helper to fetch per-SF error with tolerance and metric value (for branches below)
    import numpy as _np
    def _get_per_sf_err(per_sf_error_dict: Dict[str, float], target_sf: float) -> float:
        try:
            pairs = [(_np.float64(k), float(v)) for k, v in per_sf_error_dict.items()]
        except Exception:
            key = str(target_sf)
            if key in per_sf_error_dict:
                return float(per_sf_error_dict[key])
            raise
        sfs = _np.array([p[0] for p in pairs], dtype=_np.float64)
        vals = _np.array([p[1] for p in pairs], dtype=_np.float64)
        idx = int(_np.argmin(_np.abs(sfs - target_sf)))
        return float(vals[idx])

    def _metric_value(rec: Dict[str, float]) -> float:
        if metric == "total_error":
            return float(rec.get("total_error", _np.nan))
        elif metric == "per_sf_error":
            if sf is None:
                raise ValueError("When metric='per_sf_error', you must provide sf (e.g., 0.31)")
            per_sf = rec.get("per_sf_error", {})
            return _get_per_sf_err(per_sf, float(sf))
        else:
            raise ValueError("metric must be 'total_error' or 'per_sf_error'")

    if plot_mode == "scatter3d":
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            raise RuntimeError("Matplotlib 3D toolkit is required for plot_mode='scatter3d'.") from e

        xs = _np.array([float(r["blur_sigma"]) for r in records], dtype=float)
        ys = _np.array([float(r["noise_std"]) for r in records], dtype=float)
        zs = _np.array([int(r.get("patch_size", PATCH_SIZE)) for r in records], dtype=float)
        cs = _np.array([_metric_value(r) for r in records], dtype=float)

        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=cs, cmap=cmap, s=35, depthshade=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("Error")
        ax.set_xlabel("Blur sigma (px)")
        ax.set_ylabel("Noise std")
        ax.set_zlabel("Patch size (px)")
        title = "CSF fit error (3D)"
        if metric == "per_sf_error" and sf is not None:
            title = f"CSF error at SF={sf} cpd (3D)"
        ax.set_title(title)
        fig.tight_layout()

        if save_path is None:
            base, _ = os.path.splitext(log_path)
            suffix = "total" if metric == "total_error" else f"sf_{sf}"
            save_path = base + f"_scatter3d_{suffix}.png"
        try:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved 3D scatter to {save_path}")
        except Exception:
            pass
        if show:
            plt.show()
        else:
            plt.close(fig)
        return ax, (xs, ys, zs, cs)

    if plot_mode == "facet":
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("Matplotlib is required to plot the CSF grid log. Please install it.") from e

        patches = all_patches
        n = len(patches)
        ncols = max(1, min(int(facet_cols), n))
        nrows = int(_np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), squeeze=False, sharex=True, sharey=True)

        # Collect unique axis values across all records for consistent axes
        blur_values_all = sorted({float(r["blur_sigma"]) for r in records})
        noise_values_all = sorted({float(r["noise_std"]) for r in records})
        blur_index = {v: j for j, v in enumerate(blur_values_all)}
        noise_index = {v: i for i, v in enumerate(noise_values_all)}

        # Pre-compute global vmin/vmax for consistent color scale
        Zs = []
        for idx, p in enumerate(patches):
            r_subset = [r for r in records if int(r.get("patch_size", PATCH_SIZE)) == int(p)]
            Z = _np.full((len(noise_values_all), len(blur_values_all)), _np.nan, dtype=_np.float32)
            for rec in r_subset:
                j = blur_index.get(float(rec["blur_sigma"]))
                i = noise_index.get(float(rec["noise_std"]))
                if i is None or j is None:
                    continue
                Z[i, j] = _metric_value(rec)
            Zs.append(Z)

        if len(Zs) == 0:
            raise ValueError("No records found in grid log for facet plotting.")
        vmin = _np.nanmin(_np.stack(Zs))
        vmax = _np.nanmax(_np.stack(Zs))

        # Draw each facet
        ims = []
        for idx, (p, Z) in enumerate(zip(patches, Zs)):
            ax = axes[idx // ncols][idx % ncols]
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(min(blur_values_all), max(blur_values_all), min(noise_values_all), max(noise_values_all)),
                interpolation='nearest',
            )
            ims.append(im)
            ax.set_title(f"patch={p}", fontsize=12, pad=6)
            ax.set_xlabel("Blur sigma (px)", labelpad=4, fontsize=10)
            ax.set_ylabel("Noise std", labelpad=4, fontsize=10)
            ax.tick_params(labelsize=8)

            # Thin ticks to avoid clutter
            def _thin_ticks(vals, max_ticks=5):
                if len(vals) <= max_ticks:
                    return vals
                step = max(1, int(np.ceil(len(vals) / max_ticks)))
                picked = [vals[k] for k in range(0, len(vals), step)]
                if picked[-1] != vals[-1]:
                    picked[-1] = vals[-1]
                return picked
            ax.set_xticks(_thin_ticks(blur_values_all))
            ax.set_yticks(_thin_ticks(noise_values_all))
            for tick in ax.get_xticklabels():
                tick.set_rotation(0)

            if annotate_values:
                # Write values centered in each cell
                xedges = _np.linspace(blur_values_all[0], blur_values_all[-1], len(blur_values_all) + 1)
                yedges = _np.linspace(noise_values_all[0], noise_values_all[-1], len(noise_values_all) + 1)
                xs = 0.5 * (xedges[:-1] + xedges[1:])
                ys = 0.5 * (yedges[:-1] + yedges[1:])
                for i in range(len(noise_values_all)):
                    for j in range(len(blur_values_all)):
                        val = Z[i, j]
                        if _np.isfinite(val):
                            ax.text(xs[j], ys[i], f"{val:.2f}", ha='center', va='center', fontsize=7, color='white', alpha=0.85)

            if annotate_min and _np.isfinite(Z).any():
                min_val = _np.nanmin(Z)
                min_idx = _np.unravel_index(int(_np.nanargmin(Z)), Z.shape)
                min_noise = noise_values_all[min_idx[0]]
                min_blur = blur_values_all[min_idx[1]]
                ax.scatter([min_blur], [min_noise], marker='o', s=60, c='cyan', edgecolor='k', linewidths=0.5, zorder=3)
                ax.text(min_blur, min_noise, f" {min_val:.3f}", fontsize=8, color='black', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        # Hide unused axes (if any)
        for k in range(len(Zs), nrows * ncols):
            axes[k // ncols][k % ncols].axis('off')

        # Shared colorbar placed OUTSIDE the grid to avoid overlap
        big_title = "CSF fit error faceted by patch size"
        if metric == "per_sf_error" and sf is not None:
            big_title = f"CSF error at SF={sf} cpd — faceted by patch"
        fig.suptitle(big_title)
        # Reserve space on the right for the colorbar and set nicer spacing
        fig.subplots_adjust(left=0.08, right=0.88, top=0.90, bottom=0.10, wspace=0.12, hspace=0.18)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(ims[0], cax=cbar_ax)
        cbar.set_label("Error")

        if save_path is None:
            basefolder = ROOT / "artifacts"
            suffix = "total" if metric == "total_error" else f"sf_{sf}"
            save_path = os.path.join(basefolder, f"csf_grid_facet_{suffix}.png")
        try:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved facet heatmaps to {save_path}")
        except Exception:
            pass
        if show:
            plt.show()
        else:
            plt.close(fig)
        return axes, None

    # Default: 2D heatmap for a chosen patch size
    # Collect unique axis values
    blur_values = sorted({float(r["blur_sigma"]) for r in records})
    noise_values = sorted({float(r["noise_std"]) for r in records})

    blur_index = {v: j for j, v in enumerate(blur_values)}
    noise_index = {v: i for i, v in enumerate(noise_values)}

    Z = _np.full((len(noise_values), len(blur_values)), _np.nan, dtype=_np.float32)
    for rec in records:
        s = float(rec["blur_sigma"])  # x-axis (columns)
        n = float(rec["noise_std"])   # y-axis (rows)
        i = noise_index.get(n, None)
        j = blur_index.get(s, None)
        if i is None or j is None:
            continue
        Z[i, j] = _metric_value(rec)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required to plot the CSF grid log. Please install it.") from e

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=(min(blur_values), max(blur_values), min(noise_values), max(noise_values)),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Error")

    ax.set_xlabel("Blur sigma (px)")
    ax.set_ylabel("Noise std")
    title = "CSF fit total error"
    if metric == "per_sf_error" and sf is not None:
        title = f"CSF fit error at SF={sf} cpd"
    ax.set_title(title)

    # Tick helpers (avoid too many labels)
    def _thin_ticks(vals, max_ticks=8):
        if len(vals) <= max_ticks:
            return vals
        step = max(1, len(vals) // max_ticks)
        picked = [vals[k] for k in range(0, len(vals), step)]
        if picked[-1] != vals[-1]:
            picked[-1] = vals[-1]
        return picked

    ax.set_xticks(_thin_ticks(blur_values))
    ax.set_yticks(_thin_ticks(noise_values))

    # Annotate global minimum
    if annotate_min and _np.isfinite(Z).any():
        min_val = _np.nanmin(Z)
        min_idx = _np.unravel_index(int(_np.nanargmin(Z)), Z.shape)
        min_noise = noise_values[min_idx[0]]
        min_blur = blur_values[min_idx[1]]
        ax.scatter([min_blur], [min_noise], marker='o', s=60, c='cyan', edgecolor='k', linewidths=0.5, zorder=3)
        ax.annotate(
            f"min={min_val:.3f}\n(σ={min_blur:.3g}, noise={min_noise:.3g})",
            xy=(min_blur, min_noise), xytext=(8, 8), textcoords='offset points',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.8),
        )

    fig.tight_layout()

    # Save if requested
    if save_path is None:
        base, _ = os.path.splitext(log_path)
        suffix = "total" if metric == "total_error" else f"sf_{sf}"
        save_path = base + f"_heatmap_{suffix}.png"
    try:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    except Exception:
        pass

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax, Z

def tensor_to_display_np(x):
    """
    Convert a tensor/dict/PIL/numpy image to a displayable numpy array in [0,1] or [0,255].
    Accepts:
    - dict with key 'imgs'
    - torch.Tensor in CHW or NCHW
    - PIL.Image.Image
    - numpy.ndarray
    """
    try:
        import torch
    except Exception:
        torch = None

    if isinstance(x, dict):
        x = x.get('imgs', x)

    if torch is not None and isinstance(x, torch.Tensor):
        t = x
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.size(0) in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.detach().cpu().clamp(0.0, 1.0).numpy()
        return t

    try:
        from PIL import Image as _Image
        if isinstance(x, _Image.Image):
            return np.array(x)
    except Exception:
        pass

    if isinstance(x, np.ndarray):
        return x

    raise TypeError(f"Unsupported image type: {type(x)}")

def plot_image_pair(img_left, img_right, title_left: str = 'Original', title_right: str = 'Mouse-transformed', figsize=(8, 4)):
    """Plot two images side by side. Inputs can be PIL/np/tensor or dict with 'imgs'."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required for plotting.") from e

    left = tensor_to_display_np(img_left)
    right = tensor_to_display_np(img_right)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(left)
    ax[0].set_title(title_left)
    ax[0].axis('off')

    ax[1].imshow(right)
    ax[1].set_title(title_right)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, ax

def print_affine_params(affine_params):
    """Pretty-print affine params tuple returned by the transform (angle, translations, scale, shear)."""
    if affine_params is None:
        return
    angle, translations, scale, shear = affine_params
    print('Affine params:')
    print(f'  angle={angle:.2f}°, translate(pixels)={translations}, scale={scale:.3f}, shear={shear}')

def plot_csf_thresholds(sf_list, thresholds_dict, target_dict):
    """
    Plot CSF thresholds vs spatial frequency (log-log), matching the notebook style.
    thresholds_dict: {sf -> threshold}
    target_dict: {sf -> target_threshold}
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required for plotting.") from e

    sf = np.array(list(sf_list), dtype=float)
    thr_sim = np.array([float(thresholds_dict[s]) for s in sf_list], dtype=float)
    thr_target = np.array([float(target_dict[s]) for s in sf_list], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.loglog(sf, 1.0 / thr_target, 'o--', label='Target (mouse data)')
    plt.loglog(sf, 1.0 / thr_sim, 's-', label='Simulated (this pipeline)')
    plt.xlabel('Spatial frequency (cycles/deg)')
    plt.ylabel('Contrast threshold (Michelson)')
    plt.title('Mouse CSF: Threshold vs Spatial Frequency (log-log)')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_psychometric_curves(curves_dict, sfs_to_plot=None, threshold_criterion: float = 0.75):
    """
    Plot psychometric curves (% correct vs contrast) for selected SFs.
    curves_dict: {sf -> (contrasts, accuracies)}
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required for plotting.") from e

    if sfs_to_plot is None:
        sfs_to_plot = list(curves_dict.keys())

    plt.figure(figsize=(6, 4))
    for sf in sfs_to_plot:
        c, a = curves_dict[sf]
        plt.plot(c, 100 * np.array(a), '-o', label=f'{sf:.2f} cpd')
    plt.axhline(threshold_criterion * 100, color='k', ls='--', lw=1, label=f'{int(threshold_criterion * 100)}% criterion')
    plt.xlabel('Contrast (Michelson)')
    plt.ylabel('Percent correct (%)')
    plt.title('Psychometric curves across spatial frequencies')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_grating(g, title: str | None = None, cmap: str = 'gray'):
    """Plot a single grating tensor (HxW) in [0,1]."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("Matplotlib is required for plotting.") from e

    arr = tensor_to_display_np(g)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()