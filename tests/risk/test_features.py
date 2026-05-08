import numpy as np
import pandas as pd

from st_risk.models.base import BaseSpatialModelOutput
from st_risk.risk.features import (
    ambiguity_score,
    build_feature_table,
    confidence_proxy_score,
    local_residual_score,
    stability_score,
    uncertainty_score,
)


def test_local_residual_score_is_zero_for_identical_neighbors():
    abundance = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    neighbors = np.array([[1, 2], [0, 2], [0, 1]])
    score = local_residual_score(abundance, neighbors)
    assert np.allclose(score, 0.0)


def test_uncertainty_score_reduces_matrix_to_row_mean():
    uncertainty = np.array([[1.0, 3.0], [2.0, 4.0]])
    score = uncertainty_score(uncertainty)
    assert np.allclose(score, np.array([2.0, 3.0]))


def test_ambiguity_score_is_higher_for_mixed_abundance():
    abundance = np.array([[0.9, 0.1], [0.5, 0.5]])
    score = ambiguity_score(abundance)
    assert score[1] > score[0]


def test_confidence_proxy_defaults_to_ambiguity_without_uncertainty():
    abundance = np.array([[0.9, 0.1], [0.5, 0.5]])
    proxy = confidence_proxy_score(abundance)
    np.testing.assert_allclose(proxy, ambiguity_score(abundance))


def test_confidence_proxy_blends_uncertainty_and_ambiguity():
    abundance = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4]])
    uncertainty = np.array([0.1, 0.9, 0.2])
    proxy = confidence_proxy_score(abundance, uncertainty, uncertainty_weight=0.5)
    assert proxy.shape == (3,)
    assert proxy[1] > proxy[0]
    assert np.all(proxy >= 0.0)
    assert np.all(proxy <= 1.0)


def test_confidence_proxy_weight_changes_the_result():
    abundance = np.array([[0.9, 0.1], [0.5, 0.5], [0.6, 0.4]])
    uncertainty = np.array([0.9, 0.1, 0.2])
    low_weight = confidence_proxy_score(abundance, uncertainty, uncertainty_weight=0.1)
    high_weight = confidence_proxy_score(abundance, uncertainty, uncertainty_weight=0.9)
    assert not np.allclose(low_weight, high_weight)


def test_stability_score_returns_per_spot_variance():
    predictions = np.array(
        [
            [[0.0, 1.0], [1.0, 2.0]],
            [[0.0, 3.0], [1.0, 4.0]],
        ]
    )
    score = stability_score(predictions)
    assert np.allclose(score, np.array([1.0, 1.0]))


def test_build_feature_table_returns_expected_columns():
    abundance = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=["s1", "s2"], columns=["A", "B"])
    uncertainty = pd.Series([0.1, 0.2], index=abundance.index)
    output = BaseSpatialModelOutput(abundance=abundance, uncertainty=uncertainty)
    neighbors = np.array([[1], [0]])
    table = build_feature_table(output, neighbors=neighbors)
    assert list(table.columns) == ["phi_local", "phi_uncertainty", "phi_stability"]


def test_build_feature_table_uses_ambiguity_when_uncertainty_missing():
    abundance = pd.DataFrame([[0.9, 0.1], [0.5, 0.5]], index=["s1", "s2"], columns=["A", "B"])
    output = BaseSpatialModelOutput(abundance=abundance, uncertainty=None)
    neighbors = np.array([[1], [0]])
    table = build_feature_table(output, neighbors=neighbors)
    assert table.loc["s2", "phi_uncertainty"] > table.loc["s1", "phi_uncertainty"]
