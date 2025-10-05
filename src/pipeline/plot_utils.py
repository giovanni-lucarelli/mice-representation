import os, json
from typing import Optional, Dict
from pathlib import Path
import numpy as np
from .mouse_params import PATCH_SIZE

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