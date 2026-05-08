from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """Return k nearest neighbor indices for each coordinate row."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array")
    if len(coords) == 0:
        return np.empty((0, 0), dtype=int)
    if k <= 0:
        raise ValueError("k must be positive")
    n_neighbors = min(k + 1, len(coords))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)
    result = indices[:, 1:]
    if result.shape[1] < k:
        padded = np.full((len(coords), k), -1, dtype=int)
        padded[:, : result.shape[1]] = result
        return padded
    return result


def inverse_distance_weights(coords: np.ndarray, neighbors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute normalized inverse-distance weights for each spot and its neighbors."""
    coords = np.asarray(coords, dtype=float)
    neighbors = np.asarray(neighbors, dtype=int)
    weights = np.zeros(neighbors.shape, dtype=float)
    for i, row in enumerate(neighbors):
        valid_mask = row >= 0
        valid_neighbors = row[valid_mask]
        if len(valid_neighbors) == 0:
            continue
        deltas = coords[valid_neighbors] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        local = 1.0 / (distances + eps)
        weights[i, valid_mask] = local / local.sum()
    return weights
