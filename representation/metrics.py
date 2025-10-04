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

def spearman_brown(r):
    if not np.isfinite(r) or r <= 0: return 0.0
    return float(min(2*r/(1+r), 1.0))

def spearman_brown_vectorized_inplace(r_array, out):
    """
    MEMORY-OPTIMIZED: In-place vectorized spearman_brown for arrays.
    r_array: input array
    out: pre-allocated output array
    """
    r_array = np.asarray(r_array)
    
    # MEMORY OPTIMIZATION: In-place computation
    out.fill(0.0)  # Initialize with zeros
    
    # MEMORY OPTIMIZATION: Compute mask separately
    finite_mask = np.isfinite(r_array) & (r_array > 0)
    
    if np.any(finite_mask):
        # Compute spearman_brown only for valid values
        temp = 2 * r_array[finite_mask] / (1 + r_array[finite_mask])
        out[finite_mask] = np.minimum(temp, 1.0)

def _center_rows(X):  # centra sulle immagini (righe)
    return X - X.mean(axis=0, keepdims=True)

def _rdm_corr_distance(X):
    # RDM di distanza di correlazione tra immagini (righe)
    X = np.asarray(X, float)
    Xc = _center_rows(X)
    sd = Xc.std(axis=0, ddof=1, keepdims=True); sd[sd==0] = 1.0
    Z  = Xc / sd
    R  = np.corrcoef(Z)      # Pearson tra righe
    return 1.0 - R

def _vec_upper(M):
    iu = np.triu_indices_from(M, 1)
    return M[iu]

def rsa_pearson(X, Y):
    DX, DY = _rdm_corr_distance(X), _rdm_corr_distance(Y)
    vx, vy = _vec_upper(DX), _vec_upper(DY)
    C = np.corrcoef(vx, vy)
    return float(C[0,1])

def cka_linear(X, Y):
    Xc, Yc = _center_rows(np.asarray(X,float)), _center_rows(np.asarray(Y,float))
    XtY = Xc.T @ Yc
    num = np.sum(XtY * XtY)
    XX  = Xc.T @ Xc; YY = Yc.T @ Yc
    den = math.sqrt(np.sum(XX*XX) * np.sum(YY*YY))
    return float(num/den) if den > 0 else np.nan

def cka_linear_optimized(X, Y, chunk_size=1000):
    """
    Highly optimized CKA computation with hybrid memory/cache optimization:
    - chunk_size: Controls memory usage (how much to process at once)
    - block_size: Controls cache efficiency (how to organize data within chunks)
    """
    Xc, Yc = _center_rows(np.asarray(X, float)), _center_rows(np.asarray(Y, float))
    
    # CONSTANT PRECOMPUTATION and STRENGTH REDUCTION
    n_features, n_units, n_stimuli = Xc.shape[1], Yc.shape[1], Xc.shape[0]
    
    # HYBRID APPROACH: Separate memory management from cache optimization
    # chunk_size: Memory management (fits in RAM)
    # block_size: Cache optimization (fits in L1 cache)
    cache_block_size = min(1024, chunk_size)  # Safe for L1 cache (4KB)
    
    # COMMON SUBEXPRESSION ELIMINATION: Precompute YY
    if n_units <= 1000:
        YY = Yc.T @ Yc
        sum_YY_squared = np.sum(YY * YY)
    else:
        sum_YY_squared = _compute_YY_chunked(Yc, cache_block_size)
    
    # OPTIMIZED NUMERATOR with hybrid approach
    num = 0.0
    for i in range(0, n_features, chunk_size):
        end_i = min(i + chunk_size, n_features)
        Xc_chunk = Xc[:, i:end_i]
        
        # Process chunk in cache-optimized blocks
        for j in range(0, end_i - i, cache_block_size):
            end_j = min(j + cache_block_size, end_i - i)
            Xc_block = Xc_chunk[:, j:end_j]
            XtY_block = Xc_block.T @ Yc
            num += np.sum(XtY_block * XtY_block)
    
    # OPTIMIZED XX with hybrid approach
    sum_XX_squared = _compute_XX_hybrid(Xc, chunk_size, cache_block_size)
    
    den = math.sqrt(sum_XX_squared * sum_YY_squared)
    return float(num/den) if den > 0 else np.nan

def _compute_XX_hybrid(Xc, chunk_size, cache_block_size):
    """Helper function for hybrid XX computation with memory/cache separation."""
    n_features = Xc.shape[1]
    sum_XX_squared = 0.0
    
    # Process in memory-safe chunks
    for i in range(0, n_features, chunk_size):
        end_i = min(i + chunk_size, n_features)
        Xc_chunk_i = Xc[:, i:end_i]
        
        # Process diagonal blocks within chunk
        for j in range(0, end_i - i, cache_block_size):
            end_j = min(j + cache_block_size, end_i - i)
            Xc_block = Xc_chunk_i[:, j:end_j]
            XX_block = Xc_block.T @ Xc_block
            sum_XX_squared += np.sum(XX_block * XX_block)
        
        # Process cross blocks within chunk
        for j in range(0, end_i - i, cache_block_size):
            end_j = min(j + cache_block_size, end_i - i)
            Xc_block_j = Xc_chunk_i[:, j:end_j]
            
            for k in range(j + cache_block_size, end_i - i, cache_block_size):
                end_k = min(k + cache_block_size, end_i - i)
                Xc_block_k = Xc_chunk_i[:, k:end_k]
                XX_cross = Xc_block_j.T @ Xc_block_k
                sum_XX_squared += 2 * np.sum(XX_cross * XX_cross)
        
        # Process cross chunks (between different memory chunks)
        for k in range(i + chunk_size, n_features, chunk_size):
            end_k = min(k + chunk_size, n_features)
            Xc_chunk_k = Xc[:, k:end_k]
            
            # Cross terms between chunks
            for j in range(0, end_i - i, cache_block_size):
                end_j = min(j + cache_block_size, end_i - i)
                Xc_block_j = Xc_chunk_i[:, j:end_j]
                
                for l in range(0, end_k - k, cache_block_size):
                    end_l = min(l + cache_block_size, end_k - k)
                    Xc_block_l = Xc_chunk_k[:, l:end_l]
                    XX_cross = Xc_block_j.T @ Xc_block_l
                    sum_XX_squared += 2 * np.sum(XX_cross * XX_cross)
    
    return sum_XX_squared
def _compute_YY_chunked(Yc, block_size):
    """Helper function for chunked YY computation with optimizations."""
    n_units = Yc.shape[1]
    sum_YY_squared = 0.0
    
    # Process diagonal blocks
    for i in range(0, n_units, block_size):
        end_i = min(i + block_size, n_units)
        Yc_block = Yc[:, i:end_i]
        YY_block = Yc_block.T @ Yc_block
        sum_YY_squared += np.sum(YY_block * YY_block)
    
    # Process cross blocks (only upper triangle)
    for i in range(0, n_units, block_size):
        end_i = min(i + block_size, n_units)
        Yc_block_i = Yc[:, i:end_i]
        
        for j in range(i + block_size, n_units, block_size):
            end_j = min(j + block_size, n_units)
            Yc_block_j = Yc[:, j:end_j]
            YY_cross = Yc_block_i.T @ Yc_block_j
            sum_YY_squared += 2 * np.sum(YY_cross * YY_cross)
    
    return sum_YY_squared