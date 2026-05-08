import numpy as np

from st_risk.risk.stability import (
    gene_subsample_stability,
    ridge_project_celltype_proportions,
    row_normalize,
    sample_gene_indices,
)


def test_row_normalize_preserves_row_sums():
    values = np.array([[1.0, 1.0], [2.0, 2.0]])
    normed = row_normalize(values)
    assert np.allclose(normed.sum(axis=1), 1.0)


def test_ridge_project_celltype_proportions_returns_nonnegative_rows():
    spot = np.array([[10.0, 0.0], [0.0, 10.0]])
    signatures = np.array([[1.0, 0.0], [0.0, 1.0]])
    projected = ridge_project_celltype_proportions(spot, signatures)
    assert projected.shape == (2, 2)
    assert np.all(projected >= 0.0)
    assert np.allclose(projected.sum(axis=1), 1.0)


def test_gene_subsample_stability_returns_repeat_stack():
    spot = np.array([[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]])
    signatures = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    runs = gene_subsample_stability(spot, signatures, repeats=2, gene_fraction=0.67, random_state=0)
    assert runs.shape == (2, 2, 2)


def test_sample_gene_indices_is_deterministic():
    first = sample_gene_indices(10, repeats=2, gene_fraction=0.5, random_state=7)
    second = sample_gene_indices(10, repeats=2, gene_fraction=0.5, random_state=7)
    assert len(first) == 2
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
