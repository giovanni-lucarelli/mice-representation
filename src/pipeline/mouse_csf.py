import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from .mouse_params import (
    MOUSE_CSF_TARGET, BLUR_SIGMA_GRID, NOISE_STD_GRID, CONTRAST_SWEEP,
    N_SAMPLES_PER_CLASS, PATCH_SIZE, THRESH_CRITERION, FOV_DEG, SEED
)

torch.manual_seed(SEED)
np.random.seed(SEED)

@dataclass
class CSFParams:
    blur_sigma: float
    noise_std: float
    scores: Dict[str, float]  # per-SF error

def _make_grating(size: Tuple[int, int], sf_cpd: float, contrast: float, phase: float, theta: float) -> torch.Tensor:
    """
    Generate sinusoidal grating in [0,1] range on uniform mid-gray background.
    """
    H, W = size
    y = torch.linspace(-0.5, 0.5, steps=H)
    x = torch.linspace(-0.5, 0.5, steps=W)
    Y, X = torch.meshgrid(y, x, indexing='ij')  # degrees normalized to half-FOV below
    # degrees grid scaled to actual FOV
    Xd = X * FOV_DEG[0]
    Yd = Y * FOV_DEG[1]
    kx = sf_cpd * math.cos(theta)
    ky = sf_cpd * math.sin(theta)
    g = torch.sin(2 * math.pi * (kx * Xd + ky * Yd) + phase)
    # Michelson contrast around mid-gray 0.5
    img = 0.5 + 0.5 * contrast * g
    return img.clamp(0.0, 1.0)

def _gauss_blur(img: torch.Tensor, sigma: float, k: int) -> torch.Tensor:
    if sigma <= 0: return img
    k = int(k if k % 2 == 1 else k + 1)
    # make separable Gaussian kernel
    radius = k // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel1d = torch.exp(-(x**2) / (2 * sigma**2))
    kernel1d /= kernel1d.sum()
    kernel2d = torch.outer(kernel1d, kernel1d)
    kernel2d = kernel2d[None, None, :, :].to(img.device)
    img4 = img[None, None, ...]  # 1x1xHxW
    pad = (radius, radius, radius, radius)
    img4 = F.pad(img4, pad, mode='reflect')
    out = F.conv2d(img4, kernel2d)
    return out[0, 0]

def _extract_patch_std(x: torch.Tensor, patch: int = PATCH_SIZE) -> np.ndarray:
    """
    Compute patchwise std on luminance (grayscale). Returns (patches,) vector.
    """
    if x.dim() == 3 and x.size(0) == 3:
        x = x.mean(0)  # grayscale
    x = x[None, None, ...]  # NCHW
    unfold = F.unfold(x, kernel_size=patch, stride=patch)  # 1 x (P*P) x (HW/P^2)
    patches = unfold.squeeze(0).transpose(0, 1)  # (#patches) x (P*P)
    stds = patches.std(dim=1)
    return stds.cpu().numpy()

def _simulate_detection(size: Tuple[int, int], sf: float, blur_sigma: float, noise_std: float,
                        contrasts: Sequence[float], pca_n: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train/test an SVM to discriminate gratings vs mid-gray, returning detection rate vs contrast.
    """
    n_train = N_SAMPLES_PER_CLASS
    n_test = max(200, N_SAMPLES_PER_CLASS // 5)

    def sample_set(n):
        X, y = [], []
        for i in range(n):
            is_grating = (i % 2 == 0)
            if is_grating:
                c = contrasts[np.random.randint(0, len(contrasts))]
                theta = np.random.uniform(0, math.pi)
                phase = np.random.uniform(0, 2 * math.pi)
                g = _make_grating(size, sf, c, phase, theta)
            else:
                g = torch.full(size, 0.5)

            g = _gauss_blur(g, blur_sigma, k=11)
            if noise_std > 0:
                g = (g + torch.randn_like(g) * noise_std).clamp(0.0, 1.0)
            feat = _extract_patch_std(g)
            X.append(feat)
            y.append(1 if is_grating else 0)
        return np.stack(X, 0), np.array(y, dtype=np.int32)

    Xtr, ytr = sample_set(n_train)
    Xte, yte = sample_set(n_test)

    steps = [StandardScaler(with_mean=True, with_std=True)]
    if pca_n is not None and pca_n > 0:
        steps.append(PCA(n_components=pca_n, whiten=True, random_state=SEED))
    steps.append(LinearSVC(dual='auto', class_weight='balanced', random_state=SEED, max_iter=5000))
    clf = make_pipeline(*steps)
    clf.fit(Xtr, ytr)

    # Evaluate detection vs contrast on test contrasts
    det = []
    for c in contrasts:
        Xt, Yt = [], []
        for _ in range(n_test // len(contrasts)):
            theta = np.random.uniform(0, math.pi)
            phase = np.random.uniform(0, 2 * math.pi)
            g = _make_grating(size, sf, c, phase, theta)
            g = _gauss_blur(g, blur_sigma, k=11)
            if noise_std > 0:
                g = (g + torch.randn_like(g) * noise_std).clamp(0.0, 1.0)
            Xt.append(_extract_patch_std(g))
            Yt.append(1)
            # negatives
            h = torch.full(size, 0.5)
            h = _gauss_blur(h, blur_sigma, k=11)
            if noise_std > 0:
                h = (h + torch.randn_like(h) * noise_std).clamp(0.0, 1.0)
            Xt.append(_extract_patch_std(h))
            Yt.append(0)
        Xt = np.stack(Xt, 0)
        Yt = np.array(Yt, dtype=np.int32)
        p = clf.decision_function(Xt)
        pred = (p > 0).astype(np.int32)
        det.append((pred == Yt).mean())
    return np.array(contrasts), np.array(det)

def _estimate_threshold(contrasts: np.ndarray, acc: np.ndarray, crit: float = THRESH_CRITERION) -> float:
    # Monotonic interp: find minimal contrast where accuracy >= crit
    idx = np.where(acc >= crit)[0]
    return float(contrasts[idx[0]]) if len(idx) > 0 else float(contrasts[-1])

def fit_mouse_csf_params(
    out_json: str,
    img_size: int = 224,
    sf_table: Dict[float, float] = MOUSE_CSF_TARGET,
    blur_grid: np.ndarray = BLUR_SIGMA_GRID,
    noise_grid: np.ndarray = NOISE_STD_GRID,
    pca_n: int | None = None,
) -> CSFParams:
    """
    Search blur sigma and noise std to minimize squared error between simulated thresholds
    and target mouse CSF thresholds across spatial frequencies.
    """
    size = (img_size, img_size)
    best = (None, None, float('inf'), {})
    for s in blur_grid:
        for n in noise_grid:
            errs, total = {}, 0.0
            for sf, target_thr in sf_table.items():
                contrasts, det = _simulate_detection(size, sf, s, n, CONTRAST_SWEEP, pca_n=pca_n)
                thr = _estimate_threshold(contrasts, det, THRESH_CRITERION)
                err = abs(thr - target_thr)  # keep L1 as you set at L163
                errs[str(sf)] = err
                total += err
            if total < best[2]:
                best = (s, n, total, errs.copy())

    params = CSFParams(blur_sigma=float(best[0]), noise_std=float(best[1]), scores=best[3])
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'blur_sigma': params.blur_sigma, 'noise_std': params.noise_std, 'per_sf_error': params.scores}, f, indent=2)
    return params

def load_or_fit_params(out_json: str) -> Tuple[float, float]:
    if os.path.isfile(out_json):
        with open(out_json, 'r') as f:
            d = json.load(f)
        return float(d['blur_sigma']), float(d['noise_std'])
    p = fit_mouse_csf_params(out_json, pca_n=32)
    return p.blur_sigma, p.noise_std