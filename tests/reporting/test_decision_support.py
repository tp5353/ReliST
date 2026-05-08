from __future__ import annotations

from pathlib import Path

import pandas as pd

from st_risk.reporting.decision_support import (
    ABSTAIN_CAUTION,
    ABSTAIN_REVIEW,
    ABSTAIN_TRUSTED,
    build_consensus_primary_inputs,
    build_decision_support_summary,
    build_decision_validation_summary,
    build_deployment_table,
    cross_model_dominant_disagreement,
    render_decision_support_markdown,
    render_sample_interpretation_markdown,
    select_primary_model,
)


def test_cross_model_dominant_disagreement_builds_consensus_fraction():
    dominant_tables = {
        "a": pd.DataFrame({"dominant_celltype": ["Astro", "Excit_01"]}, index=["s1", "s2"]),
        "b": pd.DataFrame({"dominant_celltype": ["Astro", "Oligo_01"]}, index=["s1", "s2"]),
        "c": pd.DataFrame({"dominant_celltype": ["Astro", "Excit_01"]}, index=["s1", "s2"]),
    }

    result = cross_model_dominant_disagreement(dominant_tables)

    assert result.loc["s1", "consensus_label"] == "Astro"
    assert result.loc["s1", "consensus_fraction"] == 1.0
    assert result.loc["s2", "consensus_label"] == "Excit_01"
    assert round(float(result.loc["s2", "consensus_fraction"]), 3) == 0.667


def test_build_deployment_table_applies_abstention_rule():
    primary = pd.DataFrame(
        {
            "x_spatial": [1.0, 2.0, 3.0],
            "y_spatial": [4.0, 5.0, 6.0],
            "dominant_celltype": ["Astro", "Excit_01", "Oligo_01"],
        },
        index=["s1", "s2", "s3"],
    )
    risk = pd.DataFrame(
        {
            "sample_id": ["151508", "151508", "151508"],
            "risk_score": [0.1, 0.9, 0.8],
        },
        index=["s1", "s2", "s3"],
    )
    disagreement = pd.DataFrame(
        {
            "consensus_label": ["Astro", "Excit_01", "Astro"],
            "consensus_fraction": [1.0, 0.4, 0.8],
            "n_unique_labels": [1, 3, 2],
        },
        index=["s1", "s2", "s3"],
    )

    table = build_deployment_table(primary, risk, disagreement, sample_id="151508", high_risk_quantile=0.5, high_disagreement_threshold=0.6)

    assert table.loc["s1", "abstention_status"] == ABSTAIN_TRUSTED
    assert table.loc["s2", "abstention_status"] == ABSTAIN_REVIEW
    assert table.loc["s3", "abstention_status"] == ABSTAIN_CAUTION
    assert round(float(table.attrs["high_risk_quantile"]), 3) == 0.5


def test_build_decision_validation_summary_reorients_to_user_metrics():
    model_summary = pd.DataFrame(
        [
            {
                "display_name": "RCTD",
                "layer_guess_mean_proxy_ratio_at_0p8": 0.8,
                "Maynard_mean_proxy_ratio_at_0p8": 0.9,
                "layer_guess_mean_high_proxy_ratio_at_0p8": 0.6,
                "Maynard_mean_high_proxy_ratio_at_0p8": 0.7,
                "layer_guess_monotone_mean_proxy_rate": 1.0,
                "Maynard_monotone_mean_proxy_rate": 0.5,
                "layer_guess_mean_high_minus_low_proxy": 0.2,
                "Maynard_mean_high_minus_low_proxy": 0.1,
            }
        ]
    )

    summary = build_decision_validation_summary(model_summary)
    row = summary.iloc[0]

    assert row["display_name"] == "RCTD"
    assert round(float(row["mean_retained_proxy_ratio_at_0p8"]), 3) == 0.85
    assert round(float(row["mean_removed_high_proxy_fraction_at_0p8"]), 3) == 0.35
    assert round(float(row["mean_monotonicity"]), 3) == 0.75
    assert round(float(row["mean_high_vs_low_gap"]), 3) == 0.15


def test_select_primary_model_follows_goal():
    model_summary = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "mean_top1_margin": 0.3,
                "layer_guess_mean_best_auc": 0.8,
                "Maynard_mean_best_auc": 0.81,
                "marker_discordance_mean_best_auc": 0.7,
                "signature_residual_mean_best_auc": 0.9,
            },
            {
                "model_key": "stereoscope",
                "display_name": "Stereoscope",
                "mean_top1_margin": 0.01,
                "layer_guess_mean_best_auc": 0.93,
                "Maynard_mean_best_auc": 0.94,
                "marker_discordance_mean_best_auc": 0.95,
                "signature_residual_mean_best_auc": 0.81,
            },
        ]
    )

    assert select_primary_model(model_summary, goal="dominant") == "rctd"
    assert select_primary_model(model_summary, goal="structure") == "stereoscope"
    assert select_primary_model(model_summary, goal="marker") == "stereoscope"
    assert select_primary_model(model_summary, goal="reference") == "rctd"
    assert select_primary_model(model_summary, goal="consensus") == "consensus"


def test_build_consensus_primary_inputs_averages_ranked_risk():
    dominant_tables = {
        "a": pd.DataFrame(
            {"x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "dominant_celltype": ["Astro", "Excit_01"]},
            index=["s1", "s2"],
        ),
        "b": pd.DataFrame(
            {"x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "dominant_celltype": ["Astro", "Oligo_01"]},
            index=["s1", "s2"],
        ),
    }
    risk_tables = {
        "a": pd.DataFrame({"sample_id": ["151508", "151508"], "risk_score": [0.1, 0.9]}, index=["s1", "s2"]),
        "b": pd.DataFrame({"sample_id": ["151508", "151508"], "risk_score": [0.2, 0.8]}, index=["s1", "s2"]),
    }

    primary, consensus_risk, disagreement = build_consensus_primary_inputs(dominant_tables, risk_tables, sample_id="151508")

    assert primary.loc["s1", "dominant_celltype"] == "Astro"
    assert primary.loc["s2", "dominant_celltype"] == "uncertain"
    assert round(float(consensus_risk.loc["s1", "risk_score"]), 3) == 0.5
    assert round(float(consensus_risk.loc["s2", "risk_score"]), 3) == 1.0
    assert round(float(disagreement.loc["s2", "consensus_fraction"]), 3) == 0.5
    assert bool(disagreement.loc["s1", "is_consensus_resolved"])
    assert not bool(disagreement.loc["s2", "is_consensus_resolved"])


def test_build_consensus_primary_inputs_handles_partial_spot_overlap():
    dominant_tables = {
        "a": pd.DataFrame(
            {"x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "dominant_celltype": ["Astro", "Excit_01"]},
            index=["s1", "s2"],
        ),
        "b": pd.DataFrame(
            {"x_spatial": [1.0], "y_spatial": [3.0], "dominant_celltype": ["Astro"]},
            index=["s1"],
        ),
    }
    risk_tables = {
        "a": pd.DataFrame({"sample_id": ["151508", "151508"], "risk_score": [0.1, 0.9]}, index=["s1", "s2"]),
        "b": pd.DataFrame({"sample_id": ["151508"], "risk_score": [0.2]}, index=["s1"]),
    }

    primary, consensus_risk, disagreement = build_consensus_primary_inputs(dominant_tables, risk_tables, sample_id="151508")

    assert list(primary.index) == ["s1"]
    assert list(consensus_risk.index) == ["s1"]
    assert list(disagreement.index) == ["s1"]


def test_build_consensus_primary_inputs_aligns_on_coordinates_not_only_index():
    dominant_tables = {
        "a": pd.DataFrame(
            {"x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "dominant_celltype": ["Astro", "Excit_01"]},
            index=["a1", "a2"],
        ),
        "b": pd.DataFrame(
            {"x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "dominant_celltype": ["Astro", "Excit_01"]},
            index=["b1", "b2"],
        ),
    }
    risk_tables = {
        "a": pd.DataFrame(
            {"sample_id": ["151508", "151508"], "x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "risk_score": [0.1, 0.9]},
            index=["a1", "a2"],
        ),
        "b": pd.DataFrame(
            {"sample_id": ["151508", "151508"], "x_spatial": [1.0, 2.0], "y_spatial": [3.0, 4.0], "risk_score": [0.2, 0.8]},
            index=["b1", "b2"],
        ),
    }

    primary, consensus_risk, disagreement = build_consensus_primary_inputs(dominant_tables, risk_tables, sample_id="151508")

    assert list(primary.index) == ["a1", "a2"]
    assert list(consensus_risk.index) == ["a1", "a2"]
    assert round(float(disagreement.loc["a1", "consensus_fraction"]), 3) == 1.0
    assert round(float(consensus_risk.loc["a2", "risk_score"]), 3) == 1.0


def test_render_decision_support_markdown_contains_deployment_outputs(tmp_path: Path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "decision-run"
    start.mkdir(parents=True)
    figures = []
    for name in ("deconv.png", "risk.png", "filtered.png", "status.png", "disagreement.png"):
        path = start / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        figures.append(path)
    abstention_csv = start / "tables" / "abstention.csv"
    abstention_csv.parent.mkdir(parents=True, exist_ok=True)
    abstention_csv.write_text("spot,status\n", encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "sample_id": "151508",
                "primary_model": "RCTD",
                "selection_goal": "dominant",
                "high_risk_quantile": 0.8,
                "risk_threshold": 0.5,
                "disagreement_threshold": 0.6,
                "n_spots": 10,
                "trusted_fraction": 0.8,
                "caution_fraction": 0.1,
                "review_fraction": 0.1,
                "consensus_support_fraction": 0.7,
                "mean_consensus_fraction": 0.6,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "display_name": "RCTD",
                "mean_retained_proxy_ratio_at_0p8": 0.85,
                "mean_removed_high_proxy_fraction_at_0p8": 0.35,
                "mean_monotonicity": 0.75,
                "mean_high_vs_low_gap": 0.15,
            }
        ]
    )

    markdown = render_decision_support_markdown(
        sample_id="151508",
        primary_display_name="RCTD",
        selection_goal="dominant",
        deconvolution_figure=figures[0],
        risk_figure=figures[1],
        filtered_figure=figures[2],
        status_figure=figures[3],
        disagreement_figure=figures[4],
        summary_table=summary,
        validation_summary=validation,
        abstention_csv=abstention_csv,
        start=start,
    )

    assert "## Deployment Outputs（部署输出）" in markdown
    assert "## Abstention Rule（保留意见规则）" in markdown
    assert "## Decision-Oriented Validation（面向决策的验证）" in markdown
    assert "选择目标 selection goal" in markdown
    assert "disagreement map" in markdown


def test_render_decision_support_markdown_uses_consensus_specific_usage_note(tmp_path: Path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "decision-run"
    start.mkdir(parents=True)
    figures = []
    for name in ("deconv.png", "risk.png", "filtered.png", "status.png", "disagreement.png"):
        path = start / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        figures.append(path)
    abstention_csv = start / "tables" / "abstention.csv"
    abstention_csv.parent.mkdir(parents=True, exist_ok=True)
    abstention_csv.write_text("spot,status\n", encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "sample_id": "151508",
                "primary_model": "Consensus-first",
                "selection_goal": "consensus",
                "high_risk_quantile": 0.8,
                "risk_threshold": 0.5,
                "disagreement_threshold": 0.6,
                "n_spots": 10,
                "trusted_fraction": 0.8,
                "caution_fraction": 0.1,
                "review_fraction": 0.1,
                "consensus_support_fraction": 0.3,
                "mean_consensus_fraction": 0.31,
            }
        ]
    )
    validation = pd.DataFrame(
        [
            {
                "display_name": "RCTD",
                "mean_retained_proxy_ratio_at_0p8": 0.85,
                "mean_removed_high_proxy_fraction_at_0p8": 0.35,
                "mean_monotonicity": 0.75,
                "mean_high_vs_low_gap": 0.15,
            }
        ]
    )

    markdown = render_decision_support_markdown(
        sample_id="151508",
        primary_display_name="Consensus-first",
        selection_goal="consensus",
        deconvolution_figure=figures[0],
        risk_figure=figures[1],
        filtered_figure=figures[2],
        status_figure=figures[3],
        disagreement_figure=figures[4],
        summary_table=summary,
        validation_summary=validation,
        abstention_csv=abstention_csv,
        start=start,
    )

    assert "多数共识" in markdown
    assert "uncertain" in markdown


def test_render_sample_interpretation_markdown_mentions_read_order(tmp_path: Path):
    start = tmp_path / "results" / "model_comparison" / "runs" / "gallery-run"
    start.mkdir(parents=True)
    margin_summary = pd.DataFrame(
        [
            {"display_name": "RCTD", "mean_top1_margin": 0.3},
            {"display_name": "Tangram", "mean_top1_margin": 0.1},
            {"display_name": "Stereoscope", "mean_top1_margin": 0.01},
        ]
    )
    dominant_summary = pd.DataFrame(
        [
            {"display_name": "RCTD", "celltype": "Excit_01", "fraction_of_sample": 0.6},
            {"display_name": "Tangram", "celltype": "Astro", "fraction_of_sample": 0.2},
        ]
    )
    abundance_summary = pd.DataFrame(
        [
            {"display_name": "RCTD", "celltype": "Excit_01", "raw_p90": 0.8},
        ]
    )

    markdown = render_sample_interpretation_markdown(
        sample_id="151670",
        margin_summary=margin_summary,
        dominant_summary=dominant_summary,
        abundance_summary=abundance_summary,
        start=start,
    )

    assert "# Sample Interpretation（样本解读）151670" in markdown
    assert "阅读顺序" in markdown
    assert "certainty shape" in markdown
