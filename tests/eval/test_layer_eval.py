import numpy as np
import pandas as pd

from st_risk.eval.layer_eval import (
    calibrate_proxy_threshold,
    evaluate_proxy_threshold,
    bootstrap_metric_summary,
    paired_sample_metric_summary,
    boundary_enrichment_summary,
    boundary_spot_mask,
    celltype_risk_association_summary,
    cross_sample_layer_consistency_summary,
    dominant_layer_frequency_summary,
    high_risk_fraction_by_group,
    layer_celltype_risk_summary,
    layer_enrichment_by_strategy_summary,
    layer_profile_reproducibility_by_strategy_summary,
    neighbor_label_agreement,
    risk_stratified_layer_coherence_summary,
    sample_layer_risk_summary,
    sample_risk_summary,
    score_baseline_comparison_summary,
    score_proxy_comparison_summary,
    score_proxy_bin_monotonicity_summary,
    score_proxy_bin_summary,
    score_proxy_retention_curve,
    score_proxy_sample_retention_summary,
    score_proxy_retention_summary,
    score_stratified_coherence_summary,
    selective_celltype_shift_summary,
    selective_layer_retention_by_group,
    selective_layer_coherence_summary,
    selective_score_coherence_summary,
    selective_retention_summary,
)


def test_high_risk_fraction_by_group_returns_named_series():
    obs = pd.DataFrame(
        {
            "layer_guess": ["L1", "L1", "L2", "L2"],
            "risk_score": [0.1, 0.9, 0.2, 0.8],
        }
    )
    result = high_risk_fraction_by_group(obs, "layer_guess", quantile=0.5)
    assert set(result.index) == {"L1", "L2"}
    assert result.name == "high_risk_fraction"


def test_boundary_spot_mask_flags_label_changes():
    labels = pd.Series(["L1", "L1", "L2"])
    neighbors = np.array([[1], [0], [1]])
    mask = boundary_spot_mask(labels, neighbors)
    assert np.array_equal(mask, np.array([False, False, True]))


def test_neighbor_label_agreement_computes_same_label_fraction():
    labels = pd.Series(["L1", "L1", "L2"])
    neighbors = np.array([[1, 2], [0, 2], [1, -1]])
    agreement = neighbor_label_agreement(labels, neighbors)
    assert np.allclose(agreement, np.array([0.5, 0.5, 0.0]))


def test_boundary_enrichment_summary_reports_boundary_advantage():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.95, 0.99],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = boundary_enrichment_summary(obs, quantile=0.5)
    boundary_row = result.loc[result["region"] == "boundary"].iloc[0]
    non_boundary_row = result.loc[result["region"] == "non_boundary"].iloc[0]
    assert boundary_row["high_risk_fraction"] > non_boundary_row["high_risk_fraction"]
    assert boundary_row["high_risk_enrichment"] > 1.0


def test_selective_retention_summary_returns_requested_quantiles():
    obs = pd.DataFrame({"risk_score": [0.1, 0.2, 0.3, 0.4]})
    result = selective_retention_summary(obs, keep_quantiles=(0.5, 1.0))
    assert list(result["keep_quantile"]) == [0.5, 1.0]


def test_selective_layer_coherence_summary_tracks_agreement_and_boundary():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "neighbor_agreement": [1.0, 0.9, 0.3, 0.2],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = selective_layer_coherence_summary(obs, keep_quantiles=(0.5, 1.0))
    assert list(result["keep_quantile"]) == [0.5, 1.0]
    assert result.loc[0, "mean_neighbor_agreement"] > result.loc[1, "mean_neighbor_agreement"]
    assert result.loc[0, "boundary_fraction"] < result.loc[1, "boundary_fraction"]


def test_risk_stratified_layer_coherence_summary_separates_low_and_high_risk():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "neighbor_agreement": [0.9, 0.8, 0.3, 0.2],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = risk_stratified_layer_coherence_summary(obs, low_quantile=0.25, high_quantile=0.75)
    low_row = result.loc[result["risk_group"] == "low_risk"].iloc[0]
    high_row = result.loc[result["risk_group"] == "high_risk"].iloc[0]
    assert low_row["mean_neighbor_agreement"] > high_row["mean_neighbor_agreement"]
    assert low_row["boundary_fraction"] < high_row["boundary_fraction"]


def test_selective_score_coherence_summary_supports_alternative_score_columns():
    obs = pd.DataFrame(
        {
            "uncertainty_only": [0.1, 0.2, 0.8, 0.9],
            "neighbor_agreement": [1.0, 0.9, 0.3, 0.2],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = selective_score_coherence_summary(obs, score_col="uncertainty_only", keep_quantiles=(0.5, 1.0))
    assert list(result["score_name"]) == ["uncertainty_only", "uncertainty_only"]
    assert result.loc[0, "mean_neighbor_agreement"] > result.loc[1, "mean_neighbor_agreement"]


def test_score_stratified_coherence_summary_uses_requested_score_column():
    obs = pd.DataFrame(
        {
            "local_only": [0.05, 0.1, 0.7, 0.95],
            "neighbor_agreement": [0.9, 0.8, 0.4, 0.2],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = score_stratified_coherence_summary(obs, score_col="local_only", low_quantile=0.25, high_quantile=0.75)
    low_row = result.loc[result["risk_group"] == "low_risk"].iloc[0]
    high_row = result.loc[result["risk_group"] == "high_risk"].iloc[0]
    assert low_row["score_name"] == "local_only"
    assert low_row["mean_neighbor_agreement"] > high_row["mean_neighbor_agreement"]


def test_score_baseline_comparison_summary_ranks_better_scores_first():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "uncertainty_only": [0.2, 0.3, 0.7, 0.8],
            "neighbor_agreement": [0.95, 0.85, 0.35, 0.25],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = score_baseline_comparison_summary(
        obs,
        score_cols=("risk_score", "uncertainty_only"),
        keep_quantiles=(0.5, 1.0),
        low_quantile=0.25,
        high_quantile=0.75,
    )
    assert list(result["score_name"]) == ["risk_score", "uncertainty_only"]
    assert (result["low_vs_high_neighbor_agreement_gap"] > 0).all()


def test_score_proxy_comparison_summary_prefers_scores_aligned_with_proxy():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.95],
            "uncertainty_only": [0.2, 0.4, 0.6, 0.7],
            "proxy_error": [0.05, 0.1, 0.75, 0.9],
        }
    )
    result = score_proxy_comparison_summary(
        obs,
        score_cols=("risk_score", "uncertainty_only"),
        proxy_col="proxy_error",
        high_proxy_quantile=0.75,
        low_proxy_quantile=0.25,
        top_score_quantile=0.75,
    )
    assert result.iloc[0]["score_name"] == "risk_score"
    assert result.iloc[0]["score_proxy_corr"] >= result.iloc[1]["score_proxy_corr"]
    assert result.iloc[0]["top_vs_bottom_proxy_gap"] >= result.iloc[1]["top_vs_bottom_proxy_gap"]


def test_score_proxy_retention_curve_rewards_low_risk_retention():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "proxy_error": [0.05, 0.1, 0.7, 0.9],
        }
    )
    result = score_proxy_retention_curve(
        obs,
        score_col="risk_score",
        proxy_col="proxy_error",
        keep_quantiles=(0.5, 1.0),
        high_proxy_quantile=0.75,
    )
    assert result.loc[0, "retained_mean_proxy"] < result.loc[1, "retained_mean_proxy"]
    assert result.loc[0, "retained_mean_proxy_ratio"] < 1.0


def test_score_proxy_retention_summary_orders_lower_proxy_ratio_first():
    curve = pd.DataFrame(
        {
            "score_name": ["risk_score", "risk_score", "uncertainty_only", "uncertainty_only"],
            "proxy_name": ["proxy_error"] * 4,
            "keep_quantile": [0.5, 1.0, 0.5, 1.0],
            "retained_mean_proxy_ratio": [0.5, 1.0, 0.7, 1.0],
            "retained_high_proxy_fraction_ratio": [0.2, 1.0, 0.4, 1.0],
        }
    )
    result = score_proxy_retention_summary(curve, focus_quantiles=(0.5, 1.0))
    assert result.iloc[0]["score_name"] == "risk_score"
    assert result.iloc[0]["proxy_ratio_at_0p5"] == 0.5


def test_score_proxy_sample_retention_summary_returns_one_row_per_sample():
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "risk_score": [0.1, 0.8, 0.2, 0.9],
            "proxy_error": [0.05, 0.7, 0.1, 0.8],
        }
    )
    result = score_proxy_sample_retention_summary(
        obs,
        score_col="risk_score",
        proxy_col="proxy_error",
        keep_quantiles=(0.5, 1.0),
        focus_quantiles=(0.5, 1.0),
    )
    assert set(result["sample_id"]) == {"A", "B"}
    assert len(result) == 2
    assert (result["proxy_ratio_at_0p5"] < 1.0).all()


def test_calibrate_proxy_threshold_selects_maximum_feasible_coverage():
    obs = pd.DataFrame(
        {
            "risk_score": [0.10, 0.20, 0.30, 0.80, 0.90],
            "proxy_error": [0.01, 0.02, 0.60, 0.90, 1.00],
        }
    )
    result = calibrate_proxy_threshold(
        obs,
        score_col="risk_score",
        proxy_col="proxy_error",
        target_high_proxy_fraction=0.0,
        high_proxy_quantile=0.4,
    )
    assert result["target_met"] is True
    assert np.isclose(result["coverage"], 0.4)
    assert np.isclose(result["score_threshold"], 0.20)
    assert np.isclose(result["retained_high_proxy_fraction"], 0.0)


def test_evaluate_proxy_threshold_applies_fixed_threshold_on_new_subset():
    obs = pd.DataFrame(
        {
            "risk_score": [0.05, 0.15, 0.40, 0.90],
            "proxy_error": [0.01, 0.05, 0.70, 0.95],
        }
    )
    result = evaluate_proxy_threshold(
        obs,
        score_col="risk_score",
        proxy_col="proxy_error",
        score_threshold=0.20,
        target_high_proxy_fraction=0.05,
        high_proxy_quantile=0.75,
    )
    assert np.isclose(result["coverage"], 0.5)
    assert np.isclose(result["retained_mean_proxy"], 0.03)
    assert np.isclose(result["retained_high_proxy_fraction"], 0.0)
    assert result["target_met"] is True


def test_score_proxy_bin_summary_splits_sample_into_relative_risk_bins():
    obs = pd.DataFrame(
        {
            "risk_score": [0.05, 0.10, 0.20, 0.60, 0.80, 0.95],
            "proxy_error": [0.01, 0.03, 0.15, 0.40, 0.75, 0.90],
        }
    )
    result = score_proxy_bin_summary(obs, score_col="risk_score", proxy_col="proxy_error")
    assert list(result["risk_bin"]) == ["low_risk", "mid_risk", "high_risk"]
    assert np.allclose(result["coverage"], np.array([1 / 3, 1 / 3, 1 / 3]))
    assert result.loc[result["risk_bin"] == "low_risk", "mean_proxy"].iloc[0] < result.loc[
        result["risk_bin"] == "mid_risk", "mean_proxy"
    ].iloc[0]
    assert result.loc[result["risk_bin"] == "mid_risk", "mean_proxy"].iloc[0] < result.loc[
        result["risk_bin"] == "high_risk", "mean_proxy"
    ].iloc[0]


def test_score_proxy_bin_monotonicity_summary_detects_clean_to_dirty_gradient():
    obs = pd.DataFrame(
        {
            "risk_score": [0.05, 0.10, 0.20, 0.60, 0.80, 0.95],
            "proxy_error": [0.01, 0.03, 0.15, 0.40, 0.75, 0.90],
        }
    )
    bin_summary = score_proxy_bin_summary(obs, score_col="risk_score", proxy_col="proxy_error")
    result = score_proxy_bin_monotonicity_summary(bin_summary)
    row = result.iloc[0]
    assert bool(row["monotone_mean_proxy"]) is True
    assert row["high_minus_low_mean_proxy"] > 0
    assert row["low_risk_mean_proxy_ratio"] < row["high_risk_mean_proxy_ratio"]


def test_bootstrap_metric_summary_wraps_observed_mean_with_ci():
    obs = pd.DataFrame(
        {
            "run_label": ["v1", "v1", "v2", "v2"],
            "score_name": ["risk_score"] * 4,
            "proxy_name": ["proxy_error"] * 4,
            "sample_id": ["A", "B", "A", "B"],
            "mean_retained_proxy_ratio": [0.82, 0.86, 0.76, 0.79],
            "proxy_ratio_at_0p5": [0.80, 0.84, 0.72, 0.75],
        }
    )
    result = bootstrap_metric_summary(
        obs,
        group_cols=("run_label", "score_name", "proxy_name"),
        metric_cols=("mean_retained_proxy_ratio", "proxy_ratio_at_0p5"),
        n_bootstrap=64,
        random_state=0,
    )
    assert list(result["run_label"]) == ["v1", "v2"]
    for metric_col in ("mean_retained_proxy_ratio", "proxy_ratio_at_0p5"):
        assert (result[f"{metric_col}_ci_low"] <= result[metric_col]).all()
        assert (result[metric_col] <= result[f"{metric_col}_ci_high"]).all()


def test_paired_sample_metric_summary_reports_challenger_win_rate_and_diff():
    obs = pd.DataFrame(
        {
            "run_label": ["v1", "v1", "v2", "v2"],
            "score_name": ["risk_score"] * 4,
            "proxy_name": ["proxy_error"] * 4,
            "sample_id": ["A", "B", "A", "B"],
            "mean_retained_proxy_ratio": [0.90, 0.85, 0.80, 0.78],
            "proxy_ratio_at_0p5": [0.88, 0.82, 0.75, 0.74],
        }
    )
    result = paired_sample_metric_summary(
        obs,
        baseline_run="v1",
        challenger_run="v2",
        group_cols=("score_name", "proxy_name"),
        metric_cols=("mean_retained_proxy_ratio", "proxy_ratio_at_0p5"),
        n_bootstrap=64,
        random_state=0,
    )
    row = result.iloc[0]
    assert row["baseline_run"] == "v1"
    assert row["challenger_run"] == "v2"
    assert row["mean_retained_proxy_ratio_mean_diff"] < 0
    assert row["mean_retained_proxy_ratio_challenger_win_rate"] == 1.0
    assert row["proxy_ratio_at_0p5_challenger_win_rate"] == 1.0


def test_sample_risk_summary_orders_samples_by_mean_risk():
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "risk_score": [0.2, 0.3, 0.8, 0.9],
            "phi_local": [0.1, 0.1, 0.2, 0.2],
            "phi_uncertainty": [0.2, 0.2, 0.4, 0.4],
            "phi_stability": [0.3, 0.3, 0.5, 0.5],
            "neighbor_agreement": [0.9, 0.8, 0.4, 0.3],
            "boundary_mask": [False, False, True, True],
        }
    )
    result = sample_risk_summary(obs, quantile=0.5)
    assert list(result["sample_id"]) == ["B", "A"]
    assert result.loc[0, "high_risk_fraction"] > result.loc[1, "high_risk_fraction"]


def test_sample_layer_risk_summary_keeps_sample_layer_breakdown():
    obs = pd.DataFrame(
        {
            "sample_id": ["A", "A", "A", "B"],
            "layer_guess": ["L1", "L1", "L2", "L1"],
            "risk_score": [0.1, 0.9, 0.3, 0.8],
            "neighbor_agreement": [0.9, 0.2, 0.8, 0.3],
        }
    )
    result = sample_layer_risk_summary(obs, quantile=0.75)
    assert set(result["sample_id"]) == {"A", "B"}
    assert set(result["layer_guess"]) == {"L1", "L2"}
    assert "high_risk_fraction" in result.columns


def test_cross_sample_layer_consistency_summary_aggregates_by_layer():
    sample_layer = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B"],
            "layer_guess": ["L1", "L2", "L1", "L2"],
            "mean_risk_score": [0.8, 0.4, 0.7, 0.3],
            "high_risk_fraction": [0.5, 0.1, 0.4, 0.05],
            "mean_neighbor_agreement": [0.2, 0.8, 0.3, 0.9],
        }
    )
    result = cross_sample_layer_consistency_summary(sample_layer)
    assert list(result["layer_guess"]) == ["L1", "L2"]
    assert result.loc[0, "sample_count"] == 2


def test_dominant_layer_frequency_summary_counts_top_layers():
    sample_layer = pd.DataFrame(
        {
            "sample_id": ["A", "A", "B", "B", "C", "C"],
            "layer_guess": ["L1", "L2", "L1", "L2", "L2", "L3"],
            "mean_risk_score": [0.8, 0.4, 0.7, 0.6, 0.9, 0.1],
        }
    )
    result = dominant_layer_frequency_summary(sample_layer, top_k=2)
    top_row = result.iloc[0]
    assert top_row["layer_guess"] in {"L1", "L2"}
    assert "top1_count" in result.columns
    assert "top2_count" in result.columns


def test_celltype_risk_association_summary_ranks_high_risk_enriched_celltype_first():
    abundance = pd.DataFrame(
        {
            "C1": [0.1, 0.2, 0.8, 0.9],
            "C2": [0.9, 0.8, 0.2, 0.1],
        },
        index=["a", "b", "c", "d"],
    )
    obs = pd.DataFrame({"risk_score": [0.1, 0.2, 0.8, 0.9]}, index=abundance.index)
    result = celltype_risk_association_summary(abundance, obs, quantile=0.5)
    assert result.iloc[0]["celltype"] == "C1"
    assert result.iloc[0]["abundance_delta"] > 0


def test_layer_celltype_risk_summary_keeps_layer_specific_celltype_shift():
    abundance = pd.DataFrame(
        {
            "C1": [0.1, 0.9, 0.2, 0.8],
            "C2": [0.8, 0.1, 0.7, 0.2],
        },
        index=["a", "b", "c", "d"],
    )
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.9, 0.2, 0.8],
            "layer_guess": ["L1", "L1", "L2", "L2"],
        },
        index=abundance.index,
    )
    result = layer_celltype_risk_summary(abundance, obs, quantile=0.5)
    assert set(result["layer_guess"]) == {"L1", "L2"}
    assert set(result["celltype"]) == {"C1", "C2"}


def test_selective_layer_retention_by_group_tracks_retention_ratio():
    obs = pd.DataFrame(
        {
            "risk_score": [0.1, 0.2, 0.8, 0.9],
            "layer_guess": ["L1", "L1", "L2", "L2"],
        }
    )
    result = selective_layer_retention_by_group(obs, keep_quantiles=(0.5, 1.0))
    half = result[result["keep_quantile"] == 0.5]
    assert set(half["layer_guess"]) == {"L1", "L2"}
    assert float(half.loc[half["layer_guess"] == "L1", "retention_ratio"].iloc[0]) > 1.0
    assert float(half.loc[half["layer_guess"] == "L2", "retention_ratio"].iloc[0]) < 1.0


def test_selective_celltype_shift_summary_reports_retained_shift():
    abundance = pd.DataFrame(
        {
            "C1": [0.9, 0.8, 0.2, 0.1],
            "C2": [0.1, 0.2, 0.8, 0.9],
        },
        index=["a", "b", "c", "d"],
    )
    obs = pd.DataFrame({"risk_score": [0.1, 0.2, 0.8, 0.9]}, index=abundance.index)
    result = selective_celltype_shift_summary(abundance, obs, keep_quantiles=(0.5,))
    assert result.iloc[0]["celltype"] == "C1"
    assert result.iloc[0]["abundance_delta"] > 0


def test_layer_enrichment_by_strategy_summary_reports_target_layer_contrast():
    abundance = pd.DataFrame(
        {
            "C1": [0.9, 0.8, 0.2, 0.1],
            "C2": [0.1, 0.2, 0.8, 0.9],
        },
        index=["a", "b", "c", "d"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S1", "S1"],
            "layer_guess": ["L1", "L1", "L2", "L2"],
        },
        index=abundance.index,
    )
    strategy_masks = pd.DataFrame(
        {
            "unfiltered": [True, True, True, True],
            "single_risk": [True, True, False, False],
        },
        index=abundance.index,
    )
    targets = pd.DataFrame({"celltype": ["C1"], "target_layer": ["L1"]})
    result = layer_enrichment_by_strategy_summary(
        abundance,
        obs,
        strategy_masks,
        targets=targets,
    )
    assert set(result["strategy_name"]) == {"unfiltered"}
    row = result.iloc[0]
    assert row["target_layer"] == "L1"
    assert row["layer_enrichment_ratio"] > 1.0


def test_layer_profile_reproducibility_by_strategy_summary_reports_leave_one_out_correlations():
    abundance = pd.DataFrame(
        {
            "C1": [0.9, 0.8, 0.2, 0.1, 0.85, 0.75, 0.25, 0.15],
        },
        index=["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["S1"] * 4 + ["S2"] * 4,
            "layer_guess": ["L1", "L2", "L3", "L4"] * 2,
        },
        index=abundance.index,
    )
    strategy_masks = pd.DataFrame(
        {
            "unfiltered": [True] * 8,
            "single_risk": [True] * 8,
        },
        index=abundance.index,
    )
    targets = pd.DataFrame({"celltype": ["C1"], "target_layer": ["L1"]})
    result = layer_profile_reproducibility_by_strategy_summary(
        abundance,
        obs,
        strategy_masks,
        targets=targets,
        layer_order=("L1", "L2", "L3", "L4"),
        min_layers=3,
    )
    assert set(result["strategy_name"]) == {"unfiltered", "single_risk"}
    assert set(result["sample_id"]) == {"S1", "S2"}
    assert (result["reproducibility_corr"] > 0.9).all()
