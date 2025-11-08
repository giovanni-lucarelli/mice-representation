from __future__ import annotations
import random
from sklearn.cross_decomposition import PLSRegression
from .metrics import cka_linear, rsa_pearson, spearman_brown, _corr
import numpy as np
import math

def sim_corrected_source_pair(
    XA_trials, XB_trials, metric='RSA', n_boot=100, seed=0, min_half_trials=3
):
    sim = rsa_pearson if metric.upper()=='RSA' else cka_linear
    rng = random.Random(seed)
    TA, F, _ = XA_trials.shape
    TB, F2, _ = XB_trials.shape
    assert F == F2, "Mismatch numero/ordine immagini"

    vals = []
    for _ in range(n_boot):
        idxA = list(range(TA)); rng.shuffle(idxA)
        idxB = list(range(TB)); rng.shuffle(idxB)
        hA1, hA2 = np.array(idxA[:TA//2]), np.array(idxA[TA//2:])
        hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
        if min(hA1.size,hA2.size,hB1.size,hB2.size) < min_half_trials:
            continue

        XA1 = XA_trials[hA1].mean(0)  # (immagini × featA)
        XA2 = XA_trials[hA2].mean(0)
        XB1 = XB_trials[hB1].mean(0)  # (immagini × featB)
        XB2 = XB_trials[hB2].mean(0)

        # numeratore cross-half simmetrico
        num  = 0.5*(sim(XA1, XB2) + sim(XA2, XB1))
        # reliabilities (stessa metrica) + SB
        relA = spearman_brown(sim(XA1, XA2))
        # print("relA:", relA)
        relB = spearman_brown(sim(XB1, XB2))
        # print("relB:", relB)
        denom = math.sqrt(relA * relB)
        if np.isfinite(num) and denom > 0:
            vals.append(num / denom)

    return float(np.nanmean(vals)) if vals else np.nan, float(np.nanstd(vals)) if vals else np.nan

def sim_corrected_model_to_B(X_model, YB_trials, metric='RSA', n_boot=100, seed=0, min_half_trials=3):
    sim = rsa_pearson if metric.upper()=='RSA' else cka_linear
    rng = random.Random(seed)
    TB, F, _ = YB_trials.shape
    assert X_model.shape[0] == F, "Mismatch numero/ordine immagini"
    vals = []
    for _ in range(n_boot):
        idxB = list(range(TB)); rng.shuffle(idxB)
        hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
        if min(hB1.size, hB2.size) < min_half_trials:
            continue
        Y1 = YB_trials[hB1].mean(0)  # (F,q)
        Y2 = YB_trials[hB2].mean(0)
        num  = sim(X_model, Y2)                          # cross-half (il modello non va splittato)
        relB = spearman_brown(sim(Y1, Y2))               # solo affidabilità del target
        denom = math.sqrt(relB)
        if np.isfinite(num) and denom > 0:
            vals.append(num/denom)
    return (float(np.nanmean(vals)) if vals else np.nan,
            float(np.nanstd(vals))  if vals else np.nan)

def _make_image_splits(n_images, n_splits=10, seed=0):
    rng = random.Random(seed)
    idx = list(range(n_images))
    splits = []
    for _ in range(n_splits):
        rng.shuffle(idx)
        half = n_images // 2
        splits.append((np.array(idx[:half]), np.array(idx[half:])))
    return splits

def pls_corrected_single_source_to_B(
    XA_trials, YB_trials, n_components=25, n_splits=10, n_boot=100, seed=0, min_half_trials=3
):
    rng = random.Random(seed)
    TA, F, p = XA_trials.shape
    TB, F2, q = YB_trials.shape
    assert F == F2, "Mismatch numero/ordine immagini"
    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)

    per_split = []  # ognuno è (q,) score per unità
    for train_idx, test_idx in splits:
        # accumulatori per media su bootstrap
        ssum = np.zeros(q, float); cnt = np.zeros(q, int)

        for _ in range(n_boot):
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

            m1 = PLSRegression(n_components=nc).fit(Xtr1, Ytr1)
            m2 = PLSRegression(n_components=nc).fit(Xtr2, Ytr2)

            Yhat1 = m1.predict(Xte1)  # (n_test, q)
            Yhat2 = m2.predict(Xte2)

            # per-unità: numeratore e reliabilities SB
            for j in range(q):
                num = _corr(Yhat1[:,j], Yte2[:,j])                 # cross-half as in the paper
                # num = 0.5*(_corr(Yhat1[:,j], Yte2[:,j]) + _corr(Yhat2[:,j], Yte1[:,j]))  # simmetrico
                map_rel = spearman_brown(_corr(Yhat1[:,j], Yhat2[:,j]))
                tar_rel = spearman_brown(_corr(Yte1[:,j],  Yte2[:,j]))
                denom = math.sqrt(map_rel * tar_rel)
                if np.isfinite(num) and denom > 0:
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
):
    """
    Paper-style corrected predictivity for PLS, model to animal B.

    Steps per image split (train/test):
      - Bootstrap half-splits on B's trials (independent halves h1, h2).
      - Fit two PLS maps on TRAIN images: (X_model → Ytr1) and (X_model → Ytr2).
      - Predict on TEST images to get Yhat1, Yhat2.
      - For each unit j: num = corr(Yhat1[:,j], Yte2[:,j])  [or symmetric average]
        and denom = sqrt( SB(corr(Yhat1[:,j], Yhat2[:,j])) * SB(corr(Yte1[:,j], Yte2[:,j])) ).
      - Accumulate num/denom across bootstraps, then average across bootstraps.
    Finally, average across splits and return median/std across units.
    """
    rng = random.Random(seed)
    F, p = X_model.shape
    TB, F2, q = YB_trials.shape
    assert F == F2, "Image count/order mismatch between model and neural target."

    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)
    per_split = []  # each element: (q,) unit-wise scores averaged over bootstraps

    for train_idx, test_idx in splits:
        Xtr = X_model[train_idx]   # (n_train, p)
        Xte = X_model[test_idx]    # (n_test,  p)

        # accumulators over bootstraps
        ssum = np.zeros(q, float)
        cnt  = np.zeros(q, int)

        for _ in range(n_boot):
            # half-split trials on target B
            idxB = list(range(TB)); rng.shuffle(idxB)
            hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
            if min(hB1.size, hB2.size) < min_half_trials:
                continue

            # averages per half, separating train/test images
            Ytr1 = YB_trials[hB1][:, train_idx].mean(0)  # (n_train, q)
            Ytr2 = YB_trials[hB2][:, train_idx].mean(0)
            Yte1 = YB_trials[hB1][:, test_idx].mean(0)   # (n_test, q)
            Yte2 = YB_trials[hB2][:, test_idx].mean(0)

            # fit two independent PLS models (one per half)
            max_nc = min(Xtr.shape[0], Xtr.shape[1])  # = min(n_train, p)
            nc = max(1, min(n_components, max_nc))

            m1 = PLSRegression(n_components=nc).fit(Xtr, Ytr1)
            m2 = PLSRegression(n_components=nc).fit(Xtr, Ytr2)

            Yhat1 = m1.predict(Xte)  # (n_test, q)
            Yhat2 = m2.predict(Xte)

            # per-unit corrected score
            for j in range(q):
                if symmetric_num:
                    num = 0.5*(_corr(Yhat1[:, j], Yte2[:, j]) +
                               _corr(Yhat2[:, j], Yte1[:, j]))
                else:
                    num = _corr(Yhat1[:, j], Yte2[:, j])  # as in paper (cross-half)

                map_rel = spearman_brown(_corr(Yhat1[:, j], Yhat2[:, j]))
                tar_rel = spearman_brown(_corr(Yte1[:, j],  Yte2[:, j]))
                denom = math.sqrt(map_rel * tar_rel)

                if np.isfinite(num) and denom > 0:
                    ssum[j] += num / denom
                    cnt[j]  += 1

        # average over bootstraps for this split
        per_split.append(np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan))

    # average across splits, then summarize across units
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
):
    rng = random.Random(seed)
    TB, F, q = YB_trials.shape
    splits = _make_image_splits(F, n_splits=n_splits, seed=seed)

    per_split = []  # each: (q,) per-unit scores
    for train_idx, test_idx in splits:
        ssum = np.zeros(q, float); cnt = np.zeros(q, int)

        for _ in range(n_boot):
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
            m1 = PLSRegression(n_components=nc).fit(Xtr1, Ytr1)
            m2 = PLSRegression(n_components=nc).fit(Xtr2, Ytr2)

            Yhat1 = m1.predict(Xte1)  # (n_test, q)
            Yhat2 = m2.predict(Xte2)

            # Per-unit corrected score (paper Eq. 4 style)
            for j in range(q):
                num = _corr(Yhat1[:, j], Yte2[:, j])  # cross-half numerator
                map_rel = spearman_brown(_corr(Yhat1[:, j], Yhat2[:, j]))
                tar_rel = spearman_brown(_corr(Yte1[:, j],  Yte2[:, j]))
                denom = math.sqrt(map_rel * tar_rel)
                if np.isfinite(num) and denom > 0:
                    ssum[j] += num / denom
                    cnt[j]  += 1

        per_split.append(np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan))

    per_unit = np.nanmean(np.vstack(per_split), axis=0)  # (q,)
    return float(np.nanmedian(per_unit)), float(np.nanstd(per_unit))


# ---------------------------------------------------------
# POOLED SOURCE → target B : RSA / CKA (SB-corrected)
# ---------------------------------------------------------
def sim_corrected_pooled_source_to_B(
    source_trials_list,   # list of (Ti, F, pi)
    YB_trials,            # (TB, F, q)
    metric='RSA',
    n_boot=100,
    seed=0,
    min_half_trials=3,
):
    sim = rsa_pearson if metric.upper() == 'RSA' else cka_linear
    rng = random.Random(seed)
    TB, F, _ = YB_trials.shape

    vals = []
    for _ in range(n_boot):
        # Pooled source halves (no train/test split for RSA/CKA)
        pooled = _pooled_halves_sources(source_trials_list, train_idx=None, test_idx=None,
                                        rng=rng, min_half_trials=min_half_trials)
        if pooled is None:
            continue
        XA1, XA2 = pooled  # (F × p_pool) each

        # Target halves
        idxB = list(range(TB)); rng.shuffle(idxB)
        hB1, hB2 = np.array(idxB[:TB//2]), np.array(idxB[TB//2:])
        if min(hB1.size, hB2.size) < min_half_trials:
            continue
        Y1 = YB_trials[hB1].mean(0)  # (F, q)
        Y2 = YB_trials[hB2].mean(0)

        # Symmetric numerator + SB correction on both sides
        num  = 0.5 * (sim(XA1, Y2) + sim(XA2, Y1))
        relA = spearman_brown(sim(XA1, XA2))
        relB = spearman_brown(sim(Y1,  Y2))
        den = math.sqrt(relA * relB)
        if np.isfinite(num) and den > 0:
            vals.append(num / den)

    return float(np.nanmean(vals)) if vals else np.nan, float(np.nanstd(vals)) if vals else np.nan
