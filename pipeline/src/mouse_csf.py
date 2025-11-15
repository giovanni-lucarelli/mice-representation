import json
import math
import os
import multiprocessing
from functools import partial
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from tqdm import tqdm

from .mouse_params import (
    MOUSE_CSF_TARGET, BLUR_SIGMA_GRID, BLUR_KER_GRID, NOISE_STD_GRID, CONTRAST_SWEEP,
    N_SAMPLES_PER_CLASS, PATCH_SIZE, THRESH_CRITERION, FOV_DEG, SEED, PATCH_SIZE_GRID
)
 
# Enable cuDNN autotuner for faster convs when shapes are stable
try:
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# Simple caches to avoid rebuilding grids and kernels repeatedly
_XY_GRID_CACHE: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_GAUSS_KERNEL_CACHE: dict[tuple[float, int, str], torch.Tensor] = {}

def _get_xy_grid(size: Tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    H, W = size
    key = (H, W, str(device))
    cached = _XY_GRID_CACHE.get(key)
    if cached is not None and cached[0].device == device:
        return cached
    y = torch.linspace(-0.5, 0.5, steps=H, device=device, dtype=torch.float32)
    x = torch.linspace(-0.5, 0.5, steps=W, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    _XY_GRID_CACHE[key] = (Y, X)
    return Y, X

def _get_gaussian_kernel2d(sigma: float, k: int, device: torch.device) -> torch.Tensor:
    if sigma <= 0:
        # Return 1x1 identity kernel to skip convolution
        return torch.tensor([[[[1.0]]]], device=device, dtype=torch.float32)
    
    if k <= 0:
        # Dynamic kernel size based on sigma: cover ~±3σ
        kernel_size = int(2 * math.ceil(3.0 * float(sigma)) + 1)
    else:
        kernel_size = int(k)

    key = (float(sigma), kernel_size, str(device))
    ker = _GAUSS_KERNEL_CACHE.get(key)
    if ker is not None and ker.device == device:
        return ker
        
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel1d = torch.exp(-(x**2) / (2 * float(sigma)**2))
    kernel1d /= kernel1d.sum()
    kernel2d = torch.outer(kernel1d, kernel1d).view(1, 1, kernel_size, kernel_size)
    _GAUSS_KERNEL_CACHE[key] = kernel2d
    return kernel2d

# Use CUDA if available, with fallback to CPU
USE_CUDA = torch.cuda.is_available()

torch.manual_seed(SEED)
np.random.seed(SEED)

@dataclass
class CSFParams:
    blur_sigma: float
    blur_ker: int
    noise_std: float
    patch_size: int
    scores: Dict[str, float]  # per-SF error

@torch.no_grad()
def _make_grating_batch(size: Tuple[int, int], sf_cpd: float, contrast: torch.Tensor, phase: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Generate a BATCH of sinusoidal gratings in [0,1] range.
    contrast, phase, and theta are tensors of shape (N,).
    """
    N = contrast.shape[0]
    H, W = size
    device = contrast.device

    Y, X = _get_xy_grid((H, W), device)

    # Reshape for broadcasting: (N, 1, 1)
    contrast = contrast.view(N, 1, 1)
    phase = phase.view(N, 1, 1)
    theta = theta.view(N, 1, 1)

    kx = sf_cpd * torch.cos(theta)
    ky = sf_cpd * torch.sin(theta)

    g = torch.sin(2 * math.pi * (kx * (X * FOV_DEG[0]) + ky * (Y * FOV_DEG[1])) + phase)
    img = 0.5 + 0.5 * contrast * g
    return img.clamp(0.0, 1.0)

@torch.no_grad()
def _make_grating(size: Tuple[int, int], sf_cpd: float, contrast: float, phase: float, theta: float) -> torch.Tensor:
    """
    Generate sinusoidal grating in [0,1] range on uniform mid-gray background.
    """
    H, W = size
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    Y, X = _get_xy_grid((H, W), device)
    # degrees grid scaled to actual FOV
    Xd = X * FOV_DEG[0]
    Yd = Y * FOV_DEG[1]
    kx = sf_cpd * math.cos(theta)
    ky = sf_cpd * math.sin(theta)
    g = torch.sin(2 * math.pi * (kx * Xd + ky * Yd) + phase)
    # Michelson contrast around mid-gray 0.5
    img = 0.5 + 0.5 * contrast * g
    return img.clamp(0.0, 1.0)

@torch.no_grad()
def _gauss_blur(img: torch.Tensor, sigma: float, k: int, precomputed_kernel: torch.Tensor | None = None) -> torch.Tensor:
    if sigma <= 0:
        return img
    if img.dtype != torch.float32:
        img = img.to(torch.float32)
    kernel2d = precomputed_kernel if precomputed_kernel is not None else _get_gaussian_kernel2d(sigma, k, img.device)

    # Normalize input to N x 1 x H x W
    squeeze_back_2d = False
    squeeze_back_3d = False
    if img.dim() == 2:
        img4 = img.unsqueeze(0).unsqueeze(0)
        squeeze_back_2d = True
    elif img.dim() == 3:  # N x H x W
        img4 = img.unsqueeze(1)
        squeeze_back_3d = True
    elif img.dim() == 4:
        img4 = img
    else:
        raise ValueError("Unsupported image tensor shape for gaussian blur")

    ksz = int(kernel2d.size(-1))
    radius = ksz // 2
    pad = (radius, radius, radius, radius)
    img4 = F.pad(img4, pad, mode='reflect')
    out = F.conv2d(img4, kernel2d, groups=1)

    if squeeze_back_2d:
        return out[0, 0]
    if squeeze_back_3d:
        return out.squeeze(1)
    return out

@torch.no_grad()
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

@torch.no_grad()
def _extract_patch_std_batch(x_batch: torch.Tensor, patch: int = PATCH_SIZE) -> np.ndarray:
    """
    Compute patchwise std on a BATCH of luminance images.
    Returns array of shape (N, #patches).
    """
    if x_batch.dim() == 4 and x_batch.size(1) == 3:  # N,3,H,W -> grayscale
        x_batch = x_batch.mean(1, keepdim=True)
    elif x_batch.dim() == 3:  # N,H,W
        x_batch = x_batch.unsqueeze(1)

    unfold = F.unfold(x_batch, kernel_size=patch, stride=patch)  # N x (P*P) x (#patches)
    stds = unfold.std(dim=1)  # N x (#patches)
    return stds.cpu().numpy()

def _simulate_detection(size: Tuple[int, int], sf: float, blur_sigma: float, blur_ker: int, noise_std: float,
                        contrasts: Sequence[float], pca_n: int | None = None, patch_size: int = PATCH_SIZE,
                        seed: int | None = None, torch_gen: torch.Generator | None = None,
                        np_rng: np.random.Generator | None = None, deterministic: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    VECTORIZED version. Trains/tests an SVM. Data generation is done in batches on GPU.
    """
    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    if deterministic and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Local RNGs to make repeated calls identical with the same seed
    if torch_gen is None:
        torch_gen = torch.Generator(device=device)
        if seed is not None:
            torch_gen.manual_seed(int(seed))
    if np_rng is None and seed is not None:
        np_rng = np.random.default_rng(int(seed))

    n_train = N_SAMPLES_PER_CLASS
    n_test = max(200, N_SAMPLES_PER_CLASS // 5)
    gauss_kernel = _get_gaussian_kernel2d(blur_sigma, blur_ker, device)

    def _estimate_chunk_size(n_items: int) -> int:
        if device.type != "cuda":
            return min(512, n_items)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            # Rough per-image bytes: account for image, padded conv buffer, unfold buffer
            # image ~ H*W*4, conv ~ ~2x, unfold ~ ~ (P*P*#patches)*4 ~ approx same as image for stride=patch
            H, W = size
            bytes_per_img = (H * W * 4) * 3
            max_imgs = max(1, int((free_bytes * 0.5) // bytes_per_img))
            return int(max(1, min(512, max_imgs)))
        except Exception:
            return min(256, n_items)


    def sample_set_batch(n, is_training):
        half = n // 2
        rest = n - half
        chunk = _estimate_chunk_size(n)

        X_chunks = []
        y_chunks = []

        # Grating class (label=1)
        remaining = half
        while remaining > 0:
            b = min(chunk, remaining)
            if is_training:
                c_idx = torch.randint(0, len(contrasts), (b,), device=device, generator=torch_gen)
                c_vec = torch.tensor(contrasts, device=device, dtype=torch.float32)[c_idx]
            else:
                c_vec = torch.full((b,), float(c_test), device=device, dtype=torch.float32)
            theta = torch.rand(b, device=device, dtype=torch.float32, generator=torch_gen) * math.pi
            phase = torch.rand(b, device=device, dtype=torch.float32, generator=torch_gen) * (2 * math.pi)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and not deterministic)):
                g = _make_grating_batch(size, sf, c_vec, phase, theta)
                g = _gauss_blur(g, blur_sigma, k=blur_ker, precomputed_kernel=gauss_kernel)
                if noise_std > 0:
                    noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=torch_gen)
                    g = (g + noise * noise_std).clamp(0.0, 1.0)
            X_chunks.append(_extract_patch_std_batch(g, patch=patch_size))
            y_chunks.append(np.ones(b, dtype=np.int32))
            remaining -= b

        # Gray class (label=0)
        remaining = rest
        while remaining > 0:
            b = min(chunk, remaining)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and not deterministic)):
                h = torch.full((b, *size), 0.5, device=device, dtype=torch.float32)
                h = _gauss_blur(h, blur_sigma, k=blur_ker, precomputed_kernel=gauss_kernel)
                if noise_std > 0:
                    noise = torch.randn(h.shape, device=h.device, dtype=h.dtype, generator=torch_gen)
                    h = (h + noise * noise_std).clamp(0.0, 1.0)
            X_chunks.append(_extract_patch_std_batch(h, patch=patch_size))
            y_chunks.append(np.zeros(b, dtype=np.int32))
            remaining -= b

        X = np.concatenate(X_chunks, axis=0)
        y = np.concatenate(y_chunks, axis=0)

                # Shuffle the dataset
        perm = (np_rng.permutation(n) if np_rng is not None else np.random.permutation(n))
        return X[perm], y[perm]

    # --- Training ---
    c_test = 0  # Placeholder, not used for training sample generation
    Xtr, ytr = sample_set_batch(n_train, is_training=True)

    steps = [StandardScaler(with_mean=True, with_std=True)]
    if pca_n is not None and pca_n > 0:
        # Ensure PCA components do not exceed rank constraints
        max_components = min(Xtr.shape[0], Xtr.shape[1])
        n_comp = min(int(pca_n), int(max_components))
        if n_comp >= 1:
            steps.append(PCA(n_components=n_comp, whiten=True, random_state=SEED))
    steps.append(LinearSVC(dual='auto', class_weight='balanced', random_state=SEED, max_iter=2000))
    clf = make_pipeline(*steps)
    clf.fit(Xtr, ytr)

    # --- Evaluation ---
    det = []
    n_test_per_contrast = n_test // len(contrasts)
    for c_val in contrasts:
        c_test = c_val  # Set the global contrast for testing
        Xt, Yt = sample_set_batch(n_test_per_contrast, is_training=False)

        p = clf.decision_function(Xt)
        pred = (p > 0).astype(np.int32)
        det.append((pred == Yt).mean())

    return np.array(contrasts), np.array(det)

def _estimate_threshold(contrasts: np.ndarray, acc: np.ndarray, crit: float = THRESH_CRITERION) -> float:
    """
    Estimate threshold via a logistic fit on log-contrast, with robust fallbacks.

    - We fit: p(x) = gamma + (1 - gamma - lapse) / (1 + exp(-k (x - x0)))
      where x = log10(c), gamma = 0.5 (chance), and lapse is fitted in [0, 0.1].
    - If the fit fails, fall back to monotonic interpolation (first c with acc>=crit).
    """
    c = np.asarray(contrasts, dtype=float)
    a = np.asarray(acc, dtype=float)

    # Sort by contrast and enforce monotonic non-decreasing accuracy
    s = np.argsort(c)
    c = c[s]
    a = a[s]
    a = np.maximum.accumulate(a)

    # Guard rails for numeric stability
    c = np.clip(c, 1e-6, 1.0)
    a = np.clip(a, 0.0, 1.0)

    x = np.log10(c)
    gamma = 0.5

    def logistic(xv, x0, k, lapse):
        L = 1.0
        lapse = np.clip(lapse, 0.0, 0.1)
        return gamma + (L - gamma - lapse) / (1.0 + np.exp(-k * (xv - x0)))

    # Initial guesses
    x0_guess = x[int(np.clip(np.searchsorted(a, crit), 0, len(x) - 1))]
    p0 = [x0_guess, 8.0, 0.02]
    bounds = ([x.min() - 2.0, 0.1, 0.0], [x.max() + 2.0, 80.0, 0.1])

    try:
        popt, _ = curve_fit(logistic, x, a, p0=p0, bounds=bounds, maxfev=20000)
        x0, k, lapse = popt
        # Invert logistic at target criterion
        y = float(np.clip(crit, gamma + 1e-5, 1.0 - lapse - 1e-5))
        num = (1.0 - gamma - lapse) / (y - gamma) - 1.0
        thr_x = x0 - (1.0 / k) * np.log(num)
        thr_c = float(np.power(10.0, thr_x))
        return float(np.clip(thr_c, c.min(), c.max()))
    except Exception:
        # Fallback: first contrast meeting criterion
        idx = np.where(a >= crit)[0]
        return float(c[idx[0]]) if len(idx) > 0 else float(c[-1])

def _evaluate_params(params, sf_table, size, contrasts, crit, pca_n, early_stop_at: float | None = None, early_stop_margin: float = 1.10):
    """
    Worker function executed by a single process. Computes total absolute error
    for a given (blur_sigma, noise_std) pair.
    """
    p, s, k, n = params
    errs, total_err = {}, 0.0
    # Stable, per-combination seed for deterministic evaluation across runs
    p, s, k, n = params
    combo_seed = int(hash((int(p), float(s), int(k), float(n))) & 0x7FFFFFFF)
    for sf, target_thr in sf_table.items():
        # if sf < 0.1:
        #     current_patch_size = 32  # Patch più grande per basse SF
        # else:
        #     current_patch_size = int(p) # Patch di default per alte SF
        contrasts_res, det = _simulate_detection(
            size, sf, s, k, n, contrasts,
            pca_n=pca_n, patch_size=int(p), seed=combo_seed, deterministic=True
        )
        thr = _estimate_threshold(contrasts_res, det, crit)
        err = abs(np.log10(thr) - np.log10(target_thr))
        errs[str(sf)] = err
        total_err += err
        # Early pruning if clearly worse than best-so-far
        if early_stop_at is not None and total_err > (early_stop_at * early_stop_margin):
            break
    return int(p), float(s), int(k), float(n), float(total_err), errs

def fit_mouse_csf_params(
    out_json: str,
    img_size: int = 224,
    sf_table: Dict[float, float] = MOUSE_CSF_TARGET,
    blur_grid: np.ndarray = BLUR_SIGMA_GRID,
    blur_ker_grid: np.ndarray = BLUR_KER_GRID,
    noise_grid: np.ndarray = NOISE_STD_GRID,
    patch_grid: Optional[Sequence[int]] = None,
    pca_n: int | None = None,
    grid_search_log_path: Optional[str] = None,
) -> CSFParams:
    """
    Search blur sigma and noise std to minimize squared error between simulated thresholds
    and target mouse CSF thresholds across spatial frequencies.
    """
    size = (img_size, img_size)
    _patch_grid = np.array(patch_grid if patch_grid is not None else PATCH_SIZE_GRID, dtype=np.int32)
    param_combinations = [(p, s, k, n) for p in _patch_grid for s in blur_grid for k in blur_ker_grid for n in noise_grid]
    # Limit multiprocessing when CUDA is available to prevent multi-process GPU OOM
    if USE_CUDA and torch.cuda.is_available():
        num_processes = 1
    else:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Starting parallel fitting on {num_processes} processes for {len(param_combinations)} combinations...")

    worker_func = partial(
        _evaluate_params,
        sf_table=sf_table,
        size=size,
        contrasts=CONTRAST_SWEEP,
        crit=THRESH_CRITERION,
        pca_n=pca_n,
    )

    results = []
    if num_processes == 1:
        # Avoid multiprocessing when CUDA is active to prevent fork/CUDA init errors
        best_total_err_so_far = float('inf')
        for params in tqdm(param_combinations, total=len(param_combinations), desc="Fitting CSF Params"):
            p, s, k, n, total_err, errs = _evaluate_params(
                params,
                sf_table=sf_table,
                size=size,
                contrasts=CONTRAST_SWEEP,
                crit=THRESH_CRITERION,
                pca_n=pca_n,
                early_stop_at=best_total_err_so_far,
            )
            results.append((p, s, k, n, total_err, errs))
            if total_err < best_total_err_so_far:
                best_total_err_so_far = total_err
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use imap_unordered for efficient progress bar updates
            results_iterator = pool.imap_unordered(worker_func, param_combinations)
            for result in tqdm(results_iterator, total=len(param_combinations), desc="Fitting CSF Params"):
                results.append(result)

    # Optional detailed grid-search logging
    if grid_search_log_path is None:
        try:
            base, _ = os.path.splitext(out_json)
            grid_search_log_path = base + "_grid_log.json"
        except Exception:
            grid_search_log_path = None

    if grid_search_log_path is not None:
        try:
            log_dir = os.path.dirname(grid_search_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_data = [
                {
                    'patch_size': int(p),
                    'blur_sigma': float(s),
                    'blur_ker': int(k),
                    'noise_std': float(n),
                    'total_error': float(total_err),
                    'per_sf_error': {k: float(v) for k, v in errs.items()},
                }
                for p, s, k, n, total_err, errs in results
            ]
            with open(grid_search_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            print(f"Saved detailed grid search log to {grid_search_log_path}")
        except Exception as e:
            print(f"Warning: failed to write grid search log: {e}")

    best_p, best_s, best_k, best_n, min_err, best_errs = min(results, key=lambda x: x[4])
    print(f"\nBest result: Patch={int(best_p)}, Sigma={best_s:.2f}, Kernel={int(best_k)}, Noise={best_n:.3f}, Total Error={min_err:.4f}")

    params = CSFParams(blur_sigma=float(best_s), blur_ker=int(best_k), noise_std=float(best_n), patch_size=int(best_p), scores=best_errs)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'patch_size': int(params.patch_size), 'blur_sigma': params.blur_sigma, 'blur_ker': params.blur_ker, 'noise_std': params.noise_std, 'per_sf_error': params.scores}, f, indent=2)
    return params

def load_or_fit_params(out_json: str, force_fit: bool = False, pca_n: int | None = None) -> Tuple[float, int, float, int]:
    """
    Loads CSF parameters from a JSON file. If the file doesn't exist or force_fit is True,
    it runs the fitting process.
    """
    if not force_fit:
        if not os.path.isfile(out_json):
            print(f"No cached CSF params found at {out_json}")
            return None
        print(f"Loading cached CSF params from {out_json}")
        with open(out_json, 'r') as f:
            d = json.load(f)
        blur_sigma = float(d['blur_sigma'])
        blur_ker = int(d['blur_ker'])
        patch_size = int(d['patch_size'])
        noise_std = float(d['noise_std'])
        return blur_sigma, blur_ker, noise_std, patch_size

    print("Fitting CSF params (this may take a while)...")
    # Derive default log path next to out_json
    base, _ = os.path.splitext(out_json)
    log_path = base + "_grid_log.json"
    p = fit_mouse_csf_params(out_json, pca_n=pca_n, grid_search_log_path=log_path)
    return p.blur_sigma, p.blur_ker, p.noise_std, p.patch_size