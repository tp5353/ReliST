import numpy as np

from st_risk.risk.neighbors import inverse_distance_weights, knn_indices


def test_knn_indices_returns_expected_shape():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    idx = knn_indices(coords, k=1)
    assert idx.shape == (3, 1)


def test_inverse_distance_weights_normalizes_rows():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    neighbors = np.array([[1, 2], [0, 2], [1, 0]])
    weights = inverse_distance_weights(coords, neighbors)
    assert np.allclose(weights.sum(axis=1), 1.0)
