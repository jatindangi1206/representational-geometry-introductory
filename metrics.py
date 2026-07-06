"""Geometry battery. All pairwise functions assume X, Y rows are the SAME
held-out states in the SAME order (guaranteed by extract.py).
CKA = global structure, mutual kNN = local, participation ratio = effective
dimensionality, jPCA = dynamical structure."""
from __future__ import annotations
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------
# Global: Centered Kernel Alignment
# --------------------------------------------------------------------------
def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC via cross-covariance (linear kernel), numerically stable form
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    hsic_xy = np.sum(xty ** 2)
    hsic_xx = np.sum(xtx ** 2)
    hsic_yy = np.sum(yty ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(hsic_xy / denom)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_frac: float = 0.5) -> float:
    def gram(Z):
        sq = np.sum(Z ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * (Z @ Z.T)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / (2 * (sigma_frac ** 2) * med + 1e-12))
    Kx, Ky = _center_gram(gram(X)), _center_gram(gram(Y))
    hsic = lambda A, B: np.sum(A * B)
    return float(hsic(Kx, Ky) / (np.sqrt(hsic(Kx, Kx) * hsic(Ky, Ky)) + 1e-12))


def cka(X, Y, kernel: str = "linear") -> float:
    return linear_cka(X, Y) if kernel == "linear" else rbf_cka(X, Y)


# --------------------------------------------------------------------------
# Local: mutual k-nearest-neighbor overlap (platonic-rep style)
# --------------------------------------------------------------------------
def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """(N, k) nearest-neighbor index matrix. Building the index once per
    representation (not per pair) keeps all-pairs comparisons O(L*N^2)
    instead of O(L^2*N^2)."""
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    return np.argpartition(d2, kth=k, axis=1)[:, :k]


def knn_overlap(nx: np.ndarray, ny: np.ndarray) -> float:
    """Mean fraction of shared neighbors between two precomputed indices."""
    k = nx.shape[1]
    overlaps = np.empty(nx.shape[0])
    for i in range(nx.shape[0]):
        overlaps[i] = len(set(nx[i]).intersection(ny[i])) / k
    return float(overlaps.mean())


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Mean fraction of shared neighbors between the two representations."""
    return knn_overlap(knn_indices(X, k), knn_indices(Y, k))


# --------------------------------------------------------------------------
# Effective dimensionality: participation ratio (single representation)
# --------------------------------------------------------------------------
def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0, None)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    return float(s1 ** 2 / (s2 + 1e-12))


# --------------------------------------------------------------------------
# Dynamical: jPCA rotational strength
# --------------------------------------------------------------------------
def jpca_rotational_strength(features: np.ndarray, spans: Sequence[tuple],
                             n_components: int = 6, horizon: int = 20) -> dict:
    """Skew-symmetric (rotational) dynamics fit, after Churchland et al. 2012:
    PCA-reduce, finite-difference dX, LS-fit dX ~ X M with M constrained skew.
    Returns rotational_r2, rot_frac (= rot R^2 / unconstrained R^2), top_freq.
    """
    Xs, dXs = [], []
    for (s, ln) in spans:
        if ln < 2:
            continue
        seg = features[s:s + min(ln, horizon)]
        Xs.append(seg[:-1])
        dXs.append(np.diff(seg, axis=0))
    if not Xs:
        return {"rotational_r2": 0.0, "rot_frac": 0.0, "top_freq": 0.0}
    X = np.concatenate(Xs, 0)
    dX = np.concatenate(dXs, 0)

    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    k -= k % 2 or 0
    k = max(k, 2)
    P = Vt[:k].T
    Xr = Xc @ P
    dXr = (dX - dX.mean(0, keepdims=True)) @ P

    M_full, *_ = np.linalg.lstsq(Xr, dXr, rcond=None)
    r2_full = _fit_r2(Xr, dXr, M_full)
    # Antisymmetric projection = the standard jPCA skew estimate.
    M_skew = 0.5 * (M_full - M_full.T)
    r2_skew = _fit_r2(Xr, dXr, M_skew)

    eig = np.linalg.eigvals(M_skew)
    top_freq = float(np.max(np.abs(eig.imag))) if eig.size else 0.0
    return {"rotational_r2": float(max(r2_skew, 0.0)),
            "rot_frac": float(max(r2_skew, 0.0) / (abs(r2_full) + 1e-9)),
            "top_freq": top_freq}


def _fit_r2(X, Y, M):
    pred = X @ M
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(0, keepdims=True)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------
# Convenience: the single-representation battery
# --------------------------------------------------------------------------
def single_battery(X: np.ndarray, spans, mcfg: dict) -> dict:
    j = mcfg["jpca"]
    out = {"participation_ratio": participation_ratio(X)}
    out.update(jpca_rotational_strength(X, spans, j["n_components"], j["horizon"]))
    return out
