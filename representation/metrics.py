import math, random
import numpy as np

# Pearson su vettori (con guardie)
def _corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size or a.size < 2:
        return np.nan
    a = a - a.mean(); b = b - b.mean()
    den = math.sqrt((a @ a) * (b @ b))
    return float((a @ b) / den) if den > 0 else np.nan

def _corr_vectorized_inplace(A, B, out):
    """
    MEMORY-OPTIMIZED: In-place vectorized correlation computation.
    A, B: (n_samples, n_units) arrays
    out: (n_units,) pre-allocated output array
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    
    if A.shape != B.shape or A.shape[0] < 2:
        out.fill(np.nan)
        return
    
    # MEMORY OPTIMIZATION: Use temporary arrays for centering
    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    
    # MEMORY OPTIMIZATION: In-place centering using broadcasting
    A_centered = A - A_mean
    B_centered = B - B_mean
    
    # MEMORY OPTIMIZATION: Compute correlations in-place
    np.sum(A_centered * B_centered, axis=0, out=out)  # numerator
    
    # Compute denominators
    den_a = np.sqrt(np.sum(A_centered * A_centered, axis=0))
    den_b = np.sqrt(np.sum(B_centered * B_centered, axis=0))
    denominator = den_a * den_b
    
    # MEMORY OPTIMIZATION: In-place division with zero check
    np.divide(out, denominator, out=out, where=denominator > 0)
    out[denominator <= 0] = np.nan

def spearman_brown(r, strict=False):
    if strict:
        if not np.isfinite(r) or r <= 0:
            return 0.0
        return float(min(2 * r / (1 + r), 1.0))
    # mouse-vision style (no guards, allow negatives/NaNs)
    return float(2 * r / (1 + r))

def spearman_brown_vectorized_inplace(r_array, out, strict=False):
    """
    MEMORY-OPTIMIZED: In-place vectorized spearman_brown for arrays.
    r_array: input array
    out: pre-allocated output array
    """
    r_array = np.asarray(r_array)
    if strict:
        # strict: clamp non-finite or r <= 0 to 0.0, cap at 1.0
        out.fill(0.0)
        finite_mask = np.isfinite(r_array) & (r_array > 0)
        if np.any(finite_mask):
            temp = 2 * r_array[finite_mask] / (1 + r_array[finite_mask])
            out[finite_mask] = np.minimum(temp, 1.0)
        return
    # mouse-vision style (no guards): out = 2r/(1+r)
    np.divide(2.0 * r_array, 1.0 + r_array, out=out)