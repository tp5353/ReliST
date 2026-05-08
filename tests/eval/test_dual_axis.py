import numpy as np
import pandas as pd
import pytest

from st_risk.eval.dual_axis import (
    assign_dual_axis_quadrants,
    build_dual_axis_scores,
    dual_axis_correlation_summary,
    dual_axis_filter_strategy_masks,
    dual_axis_filter_strategy_summary,
    dual_axis_quadrant_proxy_summary,
    normalized_axis_weights,
)


def test_build_dual_axis_scores_returns_two_unit_interval_axes():
    features = pd.DataFrame(
        {
            "phi_local": [0.1, 0.9, 0.2, 0.3],
            "phi_uncertainty": [0.2, 0.1, 0.8, 0.7],
            "phi_stability": [0.0, 0.1, 0.6, 0.7],
            "phi_reference": [0.3, 0.2, 0.7, 0.8],
        }
    )
    result = build_dual_axis_scores(features)
    assert list(result.columns) == ["structure_risk_score", "reference_risk_score"]
    assert ((result >= 0.0) & (result <= 1.0)).all().all()
    assert not np.allclose(result["structure_risk_score"], result["reference_risk_score"])


def test_build_dual_axis_scores_accepts_custom_weights():
    features = pd.DataFrame(
        {
            "phi_local": [0.1, 0.9, 0.2, 0.3],
            "phi_uncertainty": [0.2, 0.1, 0.8, 0.7],
            "phi_stability": [0.0, 0.1, 0.6, 0.7],
            "phi_reference": [0.3, 0.2, 0.7, 0.8],
        }
    )
    result = build_dual_axis_scores(
        features,
        structure_weights={"phi_local": 1.0, "phi_uncertainty": 0.0, "phi_stability": 0.0, "phi_reference": 0.0},
        reference_weights={"phi_local": 0.0, "phi_uncertainty": 0.0, "phi_stability": 0.0, "phi_reference": 1.0},
    )
    assert not np.allclose(result["structure_risk_score"], result["reference_risk_score"])


def test_normalized_axis_weights_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="non-zero"):
        normalized_axis_weights(
            {"phi_local": 0.0, "phi_uncertainty": 0.0, "phi_stability": 0.0, "phi_reference": 0.0},
            fallback={"phi_local": 1.0, "phi_uncertainty": 0.0, "phi_stability": 0.0, "phi_reference": 0.0},
        )


def test_build_dual_axis_scores_tolerates_missing_reference_feature_column():
    features = pd.DataFrame(
        {
            "phi_local": [0.1, 0.9, 0.2, 0.3],
            "phi_uncertainty": [0.2, 0.1, 0.8, 0.7],
            "phi_stability": [0.0, 0.1, 0.6, 0.7],
        }
    )
    result = build_dual_axis_scores(
        features,
        reference_weights={"phi_local": 0.0, "phi_uncertainty": 1.0, "phi_stability": 0.0, "phi_reference": 2.0},
    )
    assert list(result.columns) == ["structure_risk_score", "reference_risk_score"]


def test_dual_axis_correlation_summary_reports_global_and_per_sample_correlations():
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "structure_risk_score": [0.1, 0.2, 0.7, 0.8],
            "reference_risk_score": [0.2, 0.1, 0.8, 0.7],
        }
    )
    global_summary, sample_summary = dual_axis_correlation_summary(obs)
    assert global_summary.iloc[0]["scope"] == "global"
    assert set(sample_summary["sample_id"]) == {"A", "B"}


def test_assign_dual_axis_quadrants_splits_each_sample_relatively():
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "A", "A"],
            "structure_risk_score": [0.1, 0.2, 0.8, 0.9],
            "reference_risk_score": [0.1, 0.8, 0.2, 0.9],
        }
    )
    result = assign_dual_axis_quadrants(obs)
    assert set(result["dual_axis_quadrant"]) == {"low_both", "structure_only", "reference_only", "high_both"}


def test_dual_axis_quadrant_proxy_summary_reports_per_quadrant_means():
    obs = pd.DataFrame(
        {
            "dual_axis_quadrant": ["low_both", "structure_only", "reference_only", "high_both"],
            "structure_risk_score": [0.1, 0.8, 0.2, 0.9],
            "reference_risk_score": [0.1, 0.2, 0.8, 0.9],
            "proxy_a": [0.2, 0.5, 0.6, 0.9],
        }
    )
    result = dual_axis_quadrant_proxy_summary(obs, proxy_cols=("proxy_a",))
    assert list(result["quadrant"]) == ["low_both", "structure_only", "reference_only", "high_both"]
    assert np.isclose(result.loc[result["quadrant"] == "high_both", "mean_proxy"].iloc[0], 0.9)


def test_dual_axis_filter_strategy_summary_compares_low_both_to_matched_single_scores():
    obs = pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "structure_risk_score": [0.1, 0.8, 0.2, 0.9],
            "reference_risk_score": [0.1, 0.2, 0.8, 0.9],
            "dual_axis_quadrant": ["low_both", "structure_only", "reference_only", "high_both"],
            "proxy_a": [0.2, 0.5, 0.6, 0.9],
        }
    )
    result = dual_axis_filter_strategy_summary(obs, proxy_cols=("proxy_a",))
    assert set(result["strategy_name"]) == {
        "low_both",
        "single_risk_matched",
        "structure_axis_matched",
        "reference_axis_matched",
    }
    low_both = result.loc[result["strategy_name"] == "low_both"].iloc[0]
    assert np.isclose(low_both["coverage"], 0.25)
    assert np.isclose(low_both["retained_mean_proxy"], 0.2)


def test_dual_axis_filter_strategy_masks_exposes_unfiltered_and_matched_masks():
    obs = pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "structure_risk_score": [0.1, 0.8, 0.2, 0.9],
            "reference_risk_score": [0.1, 0.2, 0.8, 0.9],
            "dual_axis_quadrant": ["low_both", "structure_only", "reference_only", "high_both"],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    result = dual_axis_filter_strategy_masks(obs)
    assert set(result.columns) == {
        "unfiltered",
        "low_both",
        "single_risk_matched",
        "structure_axis_matched",
        "reference_axis_matched",
    }
    assert result["unfiltered"].all()
    assert result["low_both"].sum() == 1
    assert result["single_risk_matched"].sum() == 1
