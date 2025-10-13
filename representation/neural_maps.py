from __future__ import annotations
import random
import warnings
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from .metrics import spearman_brown, spearman_brown_vectorized_inplace, _corr, _corr_vectorized_inplace
import numpy as np
import math
from .utils import maybe_tqdm

def _make_image_splits(n_images, n_splits=10, seed=0):
    rng = random.Random(seed)
    idx = list(range(n_images))
    splits = []
    half = n_images // 2
    for _ in range(n_splits):
        rng.shuffle(idx)
        splits.append((np.array(idx[:half]), np.array(idx[half:])))
    return splits

def pls_corrected_single_source_to_B(
    XA_trials,
    YB_trials,
    n_components=25,
    n_splits=10,
    n_boot=100,
    seed=0,
    min_half_trials=3,
    progress: bool = False,
):
    rng = random.Random(seed)
    TA, F, p = XA_trials.shape
    TB, F2, q = YB_trials.shape
    assert F == F2, "Mismatch numero/ordine immagini"
    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)

    per_split = []  # ognuno è (q,) score per unità
    for train_idx, test_idx in maybe_tqdm(splits, enable=progress, total=len(splits), desc="PLS splits", leave=False):
        # accumulatori per media su bootstrap
        ssum = np.zeros(q, float); cnt = np.zeros(q, int)

        for _ in maybe_tqdm(range(n_boot), enable=progress, total=n_boot, desc="PLS bootstrap", leave=False):
            # split dei trial
            idxA = list(range(TA)); rng.shuffle(idxA)
            idxB = list(range(TB)); rng.shuffle(idxB)
            hA1, hA2 = np.array(idxA[:TA//2]), np.array(idxA[TA//2:])
            hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
            if min(hA1.size,hA2.size,hB1.size,hB2.size) < min_half_trials:
                continue

            # medie per metà, separando train/test (immagini)
            Xtr1 = XA_trials[hA1][:, train_idx].mean(0)  # (n_train, p)
            Ytr1 = YB_trials[hB1][:, train_idx].mean(0)  # (n_train, q)
            Xte1 = XA_trials[hA1][:, test_idx].mean(0)   # (n_test, p)
            Yte1 = YB_trials[hB1][:, test_idx].mean(0)   # (n_test, q)

            Xtr2 = XA_trials[hA2][:, train_idx].mean(0)
            Ytr2 = YB_trials[hB2][:, train_idx].mean(0)
            Xte2 = XA_trials[hA2][:, test_idx].mean(0)
            Yte2 = YB_trials[hB2][:, test_idx].mean(0)

            # Xtr1: (n_train × p), Ytr1: (n_train × q)
            max_nc = min(Xtr1.shape[0], Xtr1.shape[1])   # = min(n_train, p)
            nc = max(1, min(n_components, max_nc))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m1 = PLSRegression(n_components=nc, scale=False).fit(Xtr1, Ytr1)
                m2 = PLSRegression(n_components=nc, scale=False).fit(Xtr2, Ytr2)

            Yhat1 = m1.predict(Xte1)  # (n_test, q)
            Yhat2 = m2.predict(Xte2)

            # per-unità: numerator e reliabilities SB
            for j in range(q):
                num = _corr(Yhat1[:,j], Yte2[:,j])                 # cross-half as in the paper
                # num = 0.5*(_corr(Yhat1[:,j], Yte2[:,j]) + _corr(Yhat2[:,j], Yte1[:,j]))  # simmetrico
                map_rel = spearman_brown(_corr(Yhat1[:,j], Yhat2[:,j]))
                tar_rel = spearman_brown(_corr(Yte1[:,j],  Yte2[:,j]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    denom = np.sqrt(map_rel * tar_rel)
                # if np.isfinite(num) and np.isfinite(denom) and denom > 0: # keep commented, explained below
                # let's propagate NaN values if denom is infinite or zero (as Nayebi)
                ssum[j] += (num / denom); cnt[j] += 1

        # media su bootstrap (per unità)
        with np.errstate(invalid='ignore', divide='ignore'):
            per_split.append(np.where(cnt>0, ssum/np.maximum(cnt,1), np.nan))

    # media sui 10 split -> vettore (q,)
    per_unit = np.nanmean(np.vstack(per_split), axis=0)
    # aggregato finale (paper): mediana su unità
    return float(np.nanmedian(per_unit)), float(np.nanstd(per_unit))

def pls_corrected_model_to_B(
    X_model,          # (F, p) model design matrix for the chosen layer
    YB_trials,        # (T, F, q) trial-level neural data for target animal B
    n_components=25,
    n_splits=10,
    n_boot=100,
    seed=0,
    min_half_trials=3,
    symmetric_num=False,  # if True, average num over both cross-halves
    progress: bool = False,
):
    """
    MEMORY-OPTIMIZED: Paper-style corrected predictivity for PLS, model→animal B.
    
    Memory pre-allocation optimizations applied:
    - Pre-allocate all working arrays
    - Reuse memory buffers across iterations
    - Optimize memory layout for cache performance
    - Reduce garbage collection pressure
    """
    rng = np.random.default_rng(seed)
    F, p = X_model.shape
    TB, F2, q = YB_trials.shape
    assert F == F2, "Image count/order mismatch between model and neural target."

    # MEMORY PRE-ALLOCATION: Compute all dimensions upfront
    half_TB = TB // 2
    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)
    
    # MEMORY PRE-ALLOCATION: Pre-allocate result arrays
    per_split = np.zeros((n_splits, q), dtype=float)
    per_split.fill(np.nan)  # Initialize with NaN
    
    # MEMORY PRE-ALLOCATION: Pre-allocate working arrays for bootstrap loop
    idxB_buffer = np.arange(TB, dtype=int)  # Reusable trial index buffer
    hB1_buffer = np.empty(half_TB, dtype=int)  # Pre-allocated half-split buffer
    hB2_buffer = np.empty(TB - half_TB, dtype=int)  # Pre-allocated half-split buffer
    
    # MEMORY PRE-ALLOCATION: Pre-allocate correlation arrays
    corr_hat1_te2 = np.empty(q, dtype=float)
    corr_hat2_te1 = np.empty(q, dtype=float)
    map_corrs = np.empty(q, dtype=float)
    tar_corrs = np.empty(q, dtype=float)
    map_rel = np.empty(q, dtype=float)
    tar_rel = np.empty(q, dtype=float)
    denom = np.empty(q, dtype=float)
    valid_mask = np.empty(q, dtype=bool)

    for split_idx, (train_idx, test_idx) in enumerate(maybe_tqdm(splits, enable=progress, total=len(splits), desc="PLS splits", leave=False)):
        Xtr = X_model[train_idx]   # (n_train, p)
        Xte = X_model[test_idx]    # (n_test,  p)

        # OPTIMIZATION: Pre-compute max components once
        max_nc = min(Xtr.shape[0], Xtr.shape[1])
        nc = max(1, min(n_components, max_nc))

        # MEMORY PRE-ALLOCATION: Accumulators over bootstraps
        ssum = np.zeros(q, dtype=float)
        cnt  = np.zeros(q, dtype=int)

        for _ in maybe_tqdm(range(n_boot), enable=progress, total=n_boot, desc="PLS bootstrap", leave=False):
            # MEMORY OPTIMIZATION: Reuse pre-allocated buffers
            np.copyto(idxB_buffer, np.arange(TB))
            rng.shuffle(idxB_buffer)
            
            # MEMORY OPTIMIZATION: Use pre-allocated half-split buffers
            hB1_buffer[:] = idxB_buffer[:half_TB]
            hB2_buffer[:] = idxB_buffer[half_TB:]
            
            if min(hB1_buffer.size, hB2_buffer.size) < min_half_trials:
                continue

            # MEMORY OPTIMIZATION: Direct indexing without array creation
            Ytr1 = YB_trials[hB1_buffer][:, train_idx].mean(axis=0)  # (n_train, q)
            Ytr2 = YB_trials[hB2_buffer][:, train_idx].mean(axis=0)
            Yte1 = YB_trials[hB1_buffer][:, test_idx].mean(axis=0)   # (n_test, q)
            Yte2 = YB_trials[hB2_buffer][:, test_idx].mean(axis=0)

            # fit two independent PLS models (one per half)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m1 = PLSRegression(n_components=nc, scale=False).fit(Xtr, Ytr1)
                m2 = PLSRegression(n_components=nc, scale=False).fit(Xtr, Ytr2)

            Yhat1 = m1.predict(Xte)  # (n_test, q)
            Yhat2 = m2.predict(Xte)

            # MEMORY OPTIMIZATION: Reuse pre-allocated correlation arrays
            if symmetric_num:
                # Compute all correlations at once using pre-allocated buffers
                _corr_vectorized_inplace(Yhat1, Yte2, corr_hat1_te2)
                _corr_vectorized_inplace(Yhat2, Yte1, corr_hat2_te1)
                num = np.empty_like(corr_hat1_te2)
                np.add(corr_hat1_te2, corr_hat2_te1, out=num)
                num *= 0.5
            else:
                num = np.empty(q, dtype=float)
                _corr_vectorized_inplace(Yhat1, Yte2, num)

            # MEMORY OPTIMIZATION: Reuse pre-allocated reliability arrays
            _corr_vectorized_inplace(Yhat1, Yhat2, map_corrs)
            _corr_vectorized_inplace(Yte1, Yte2, tar_corrs)
            
            # MEMORY OPTIMIZATION: In-place spearman_brown calculations
            spearman_brown_vectorized_inplace(map_corrs, map_rel)
            spearman_brown_vectorized_inplace(tar_corrs, tar_rel)
            np.multiply(map_rel, tar_rel, out=denom)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                np.sqrt(denom, out=denom)

            # MEMORY OPTIMIZATION: Vectorized accumulation with pre-allocated mask
            np.isfinite(num, out=valid_mask)
            # valid_mask &= np.isfinite(denom) & (denom > 0) # keep commented, explained below
            # let's propagate NaN values if denom is infinite or zero (as Nayebi)
            
            # In-place accumulation
            temp = np.zeros_like(ssum)
            # previously here there was temp = np.empty_like(ssum), bug (?)
            
            np.divide(num, denom, out=temp, where=valid_mask)
            ssum += temp
            cnt += valid_mask.astype(int)

        # MEMORY OPTIMIZATION: Direct assignment to pre-allocated result array
        valid_cnt = cnt > 0
        per_split[split_idx, valid_cnt] = ssum[valid_cnt] / cnt[valid_cnt]

    # MEMORY OPTIMIZATION: Use pre-allocated arrays for final computation
    per_unit = np.nanmean(per_split, axis=0)  # (q,)
    return float(np.nanmedian(per_unit)), float(np.nanstd(per_unit))

# -------------------------------------------------------
# POOLED SOURCE → target B : PLS (paper-style corrected)
# -------------------------------------------------------
def pls_corrected_pooled_source_to_B(
    source_trials_list,   # list of (Ti, F, pi) arrays for all sources A≠B
    YB_trials,            # (TB, F, q)
    n_components=25,
    n_splits=10,
    n_boot=100,
    seed=0,
    min_half_trials=3,
    progress: bool = False):
    
    rng = random.Random(seed)
    TB, F, q = YB_trials.shape
    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)

    per_split = []  # each: (q,) per-unit scores
    for train_idx, test_idx in maybe_tqdm(splits, enable=progress, total=len(splits), desc="PLS splits", leave=False):
        ssum = np.zeros(q, float); cnt = np.zeros(q, int)

        for _ in maybe_tqdm(range(n_boot), enable=progress, total=n_boot, desc="PLS bootstrap", leave=False):
            # Build pooled source halves for this bootstrap + split
            pooled = _pooled_halves_sources(
                source_trials_list, train_idx, test_idx, rng=rng,
                min_half_trials=min_half_trials
            )
            if pooled is None:
                continue
            Xtr1, Xtr2, Xte1, Xte2 = pooled

            # Target B half-split
            idxB = list(range(TB)); rng.shuffle(idxB)
            hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
            if min(hB1.size, hB2.size) < min_half_trials:
                continue
            Ytr1 = YB_trials[hB1][:, train_idx].mean(0)  # (n_train, q)
            Ytr2 = YB_trials[hB2][:, train_idx].mean(0)
            Yte1 = YB_trials[hB1][:, test_idx].mean(0)   # (n_test,  q)
            Yte2 = YB_trials[hB2][:, test_idx].mean(0)

            # Fit two independent PLS maps
            max_nc = min(Xtr1.shape[0], Xtr1.shape[1])  # = min(n_train, p_pool)
            nc = max(1, min(n_components, max_nc))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m1 = PLSRegression(n_components=nc, scale=False).fit(Xtr1, Ytr1)
                m2 = PLSRegression(n_components=nc, scale=False).fit(Xtr2, Ytr2)

            Yhat1 = m1.predict(Xte1)  # (n_test, q)
            Yhat2 = m2.predict(Xte2)

            # Per-unit corrected score (paper Eq. 4 style)
            for j in range(q):
                num = _corr(Yhat1[:, j], Yte2[:, j])  # cross-half numerator
                map_rel = spearman_brown(_corr(Yhat1[:, j], Yhat2[:, j]))
                tar_rel = spearman_brown(_corr(Yte1[:, j],  Yte2[:, j]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    denom = np.sqrt(map_rel * tar_rel)
                # if np.isfinite(num) and np.isfinite(denom) and denom > 0: # keep commented, explained below
                # let's propagate NaN values if denom is infinite or zero (as Nayebi)
                ssum[j] += num / denom
                cnt[j]  += 1

        per_split.append(np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan))

    per_unit = np.nanmean(np.vstack(per_split), axis=0)  # (q,)
    return float(np.nanmedian(per_unit)), float(np.nanstd(per_unit))

def _pooled_halves_sources(source_trials_list, train_idx=None, test_idx=None,
                           rng=None, min_half_trials=3):
    """
    Build pooled source halves across multiple animals for one bootstrap.
    Each source animal is split into two halves of trials, averaged per half,
    then concatenated across animals along the feature axis.

    source_trials_list: list of arrays [ (Ti, F, pi), ... ]
    Returns:
      if train_idx/test_idx provided:
        (Xtr1, Xtr2, Xte1, Xte2) each shape: (n_imgs_split × p_pool)
      else (no image split for RSA/CKA):
        (X1, X2) each shape: (F × p_pool)
    """
    if rng is None:
        rng = random.Random(0)

    X1_chunks_train, X2_chunks_train = [], []
    X1_chunks_test,  X2_chunks_test  = [], []

    F_ref = None
    for Xi in source_trials_list:
        Ti, F, pi = Xi.shape
        if F_ref is None: F_ref = F
        assert F == F_ref, "All sources must share the same image order/count"

        idx = list(range(Ti)); rng.shuffle(idx)
        h1, h2 = np.array(idx[:Ti//2]), np.array(idx[Ti//2:])
        if min(h1.size, h2.size) < min_half_trials:
            # skip this source animal for this bootstrap
            continue

        if train_idx is None:
            X1 = Xi[h1].mean(0)  # (F, pi)
            X2 = Xi[h2].mean(0)
            X1_chunks_train.append(X1); X2_chunks_train.append(X2)
        else:
            # per split: train/test slices
            X1tr = Xi[h1][:, train_idx].mean(0)  # (n_train, pi)
            X2tr = Xi[h2][:, train_idx].mean(0)
            X1te = Xi[h1][:, test_idx].mean(0)   # (n_test, pi)
            X2te = Xi[h2][:, test_idx].mean(0)
            X1_chunks_train.append(X1tr); X2_chunks_train.append(X2tr)
            X1_chunks_test.append(X1te);  X2_chunks_test.append(X2te)

    if not X1_chunks_train:  # no contributing sources this round
        return None

    if train_idx is None:
        X1 = np.concatenate(X1_chunks_train, axis=1)  # (F, p_pool)
        X2 = np.concatenate(X2_chunks_train,  axis=1)
        return (X1, X2)
    else:
        Xtr1 = np.concatenate(X1_chunks_train, axis=1)  # (n_train, p_pool)
        Xtr2 = np.concatenate(X2_chunks_train,  axis=1)
        Xte1 = np.concatenate(X1_chunks_test,  axis=1)  # (n_test,  p_pool)
        Xte2 = np.concatenate(X2_chunks_test,   axis=1)
        return (Xtr1, Xtr2, Xte1, Xte2)