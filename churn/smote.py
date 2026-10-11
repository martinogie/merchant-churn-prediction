"""Minimal SMOTE (Chawla et al., 2002): synthesise minority samples by interpolating to neighbours."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, ratio: float = 0.5, k: int = 5, seed: int = 0):
    """Oversample class 1 until minority/majority == ratio. Use on training data only."""
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X, float), np.asarray(y)
    minority = X[y == 1]
    n_needed = int(ratio * (y == 0).sum()) - len(minority)
    if n_needed <= 0 or len(minority) <= k:
        return X, y
    scale = X.std(axis=0) + 1e-9                       # neighbours in standardised space
    nn = NearestNeighbors(n_neighbors=k + 1).fit(minority / scale)
    _, idx = nn.kneighbors(minority / scale)
    base = rng.integers(0, len(minority), n_needed)
    neigh = idx[base, rng.integers(1, k + 1, n_needed)]
    gap = rng.random((n_needed, 1))
    synthetic = minority[base] + gap * (minority[neigh] - minority[base])
    return np.vstack([X, synthetic]), np.concatenate([y, np.ones(n_needed, dtype=y.dtype)])
