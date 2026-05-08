import numpy as np
import pytest

from st_risk.risk.score import combine_risk_features, enumerate_weight_schemes, grouped_zscore, weight_scheme_name


def test_combine_risk_features_returns_unit_interval_scores():
    features = {
        "phi_local": np.array([0.0, 1.0, 2.0]),
        "phi_uncertainty": np.array([0.0, 0.5, 1.0]),
        "phi_stability": np.array([0.0, 0.25, 0.5]),
    }
    scores = combine_risk_features(features)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_weight_scheme_name_formats_feature_weights():
    name = weight_scheme_name({"phi_local": 1.0, "phi_uncertainty": 2.0, "phi_stability": 0.0})
    assert name == "wL1_U2_S0"


def test_enumerate_weight_schemes_skips_all_zero_scheme():
    schemes = enumerate_weight_schemes(levels=(0.0, 1.0))
    assert "wL0_U0_S0" not in schemes
    assert "wL1_U0_S0" in schemes
    assert "wL0_U1_S1" in schemes
    assert len(schemes) == 7


def test_grouped_zscore_normalizes_each_group_independently():
    values = np.array([1.0, 2.0, 10.0, 12.0])
    groups = np.array(["a", "a", "b", "b"])

    normalized = grouped_zscore(values, groups=groups)

    np.testing.assert_allclose(normalized[:2], np.array([-1.0, 1.0]))
    np.testing.assert_allclose(normalized[2:], np.array([-1.0, 1.0]))


def test_combine_risk_features_accepts_groupwise_normalization():
    features = {
        "phi_local": np.array([0.0, 1.0, 10.0, 11.0]),
        "phi_uncertainty": np.array([0.0, 1.0, 10.0, 11.0]),
        "phi_stability": np.array([0.0, 1.0, 10.0, 11.0]),
    }
    groups = np.array(["s1", "s1", "s2", "s2"])

    global_scores = combine_risk_features(features)
    grouped_scores = combine_risk_features(features, groups=groups)

    assert global_scores[0] < grouped_scores[0]
    np.testing.assert_allclose(grouped_scores[[0, 2]], grouped_scores[0])
    np.testing.assert_allclose(grouped_scores[[1, 3]], grouped_scores[1])


def test_grouped_zscore_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        grouped_zscore(np.array([1.0, 2.0]), groups=np.array(["a"]))
