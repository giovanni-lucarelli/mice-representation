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

def spearman_brown(r):
    if not np.isfinite(r) or r <= 0: return 0.0
    return float(min(2*r/(1+r), 1.0))

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
