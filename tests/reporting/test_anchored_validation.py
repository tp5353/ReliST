from __future__ import annotations

from pathlib import Path

import pandas as pd

from st_risk.models.base import BaseSpatialModelOutput
from st_risk.models.io import save_base_model_output
from st_risk.reporting.anchored_validation import (
    AnchoredModelSpec,
    best_score_per_proxy_sample,
    build_model_summary,
    render_anchored_validation_markdown,
    sharpness_summary,
)


def test_best_score_per_proxy_sample_keeps_highest_auc():
    auc_summary = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "score_name": "risk_score",
                "proxy_name": "snrna_signature_residual",
                "extreme_proxy_auc": 0.82,
                "score_proxy_corr": 0.3,
            },
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "extreme_proxy_auc": 0.91,
                "score_proxy_corr": 0.2,
            },
        ]
    )

    result = best_score_per_proxy_sample(auc_summary)

    assert result.shape[0] == 1
    row = result.iloc[0]
    assert row["score_name"] == "reference_risk_score"
    assert float(row["extreme_proxy_auc"]) == 0.91


def test_build_model_summary_combines_sharpness_auc_retention_and_calibration():
    sharpness = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "mean_top1_margin": 0.3,
                "low_margin_fraction": 0.2,
                "top_dominant_fraction": 0.7,
            },
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151508",
                "mean_top1_margin": 0.2,
                "low_margin_fraction": 0.3,
                "top_dominant_fraction": 0.6,
            },
        ]
    )
    auc_summary = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "extreme_proxy_auc": 0.9,
                "score_proxy_corr": 0.5,
            },
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151508",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "extreme_proxy_auc": 0.8,
                "score_proxy_corr": 0.4,
            },
        ]
    )
    retention = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "keep_quantile": 0.8,
                "retained_mean_proxy_ratio": 0.7,
                "retained_high_proxy_fraction_ratio": 0.6,
            },
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151508",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "keep_quantile": 0.8,
                "retained_mean_proxy_ratio": 0.8,
                "retained_high_proxy_fraction_ratio": 0.7,
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151507",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "monotone_mean_proxy": True,
                "high_minus_low_mean_proxy": 0.2,
            },
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "run_id": "run-a",
                "sample_id": "151508",
                "score_name": "reference_risk_score",
                "proxy_name": "snrna_signature_residual",
                "monotone_mean_proxy": False,
                "high_minus_low_mean_proxy": 0.1,
            },
        ]
    )

    summary = build_model_summary(sharpness, auc_summary, retention, calibration)

    row = summary.iloc[0]
    assert row["display_name"] == "RCTD"
    assert round(float(row["mean_top1_margin"]), 3) == 0.25
    assert round(float(row["signature_residual_mean_best_auc"]), 3) == 0.85
    assert row["signature_residual_best_score_mode"] == "reference_risk_score"
    assert round(float(row["signature_residual_mean_proxy_ratio_at_0p8"]), 3) == 0.75
    assert round(float(row["signature_residual_monotone_mean_proxy_rate"]), 3) == 0.5


def test_render_anchored_validation_markdown_mentions_rctd_readout():
    model_summary = pd.DataFrame(
        [
            {
                "model_key": "rctd",
                "display_name": "RCTD",
                "n_samples": 4,
                "mean_top1_margin": 0.3,
                "mean_low_margin_fraction": 0.2,
                "mean_top_dominant_fraction": 0.7,
                "layer_guess_mean_best_auc": 0.85,
                "Maynard_mean_best_auc": 0.84,
                "marker_discordance_mean_best_auc": 0.72,
                "signature_residual_mean_best_auc": 0.88,
                "layer_guess_mean_proxy_ratio_at_0p8": 0.8,
                "Maynard_mean_proxy_ratio_at_0p8": 0.81,
                "marker_discordance_mean_proxy_ratio_at_0p8": 0.85,
                "signature_residual_mean_proxy_ratio_at_0p8": 0.7,
            },
            {
                "model_key": "stereoscope",
                "display_name": "Stereoscope",
                "n_samples": 4,
                "mean_top1_margin": 0.01,
                "mean_low_margin_fraction": 1.0,
                "mean_top_dominant_fraction": 0.06,
                "layer_guess_mean_best_auc": 0.93,
                "Maynard_mean_best_auc": 0.94,
                "marker_discordance_mean_best_auc": 0.95,
                "signature_residual_mean_best_auc": 0.81,
                "layer_guess_mean_proxy_ratio_at_0p8": 0.75,
                "Maynard_mean_proxy_ratio_at_0p8": 0.76,
                "marker_discordance_mean_proxy_ratio_at_0p8": 0.8,
                "signature_residual_mean_proxy_ratio_at_0p8": 0.74,
            },
        ]
    )
    best_auc = pd.DataFrame(
        [
            {
                "display_name": "RCTD",
                "proxy_name": "snrna_signature_residual",
                "score_name": "reference_risk_score",
                "extreme_proxy_auc": 0.88,
            }
        ]
    )

    markdown = render_anchored_validation_markdown(model_summary, best_auc)

    assert "# Anchored Validation（外部锚点验证）：RCTD vs Hardness（更硬输出）" in markdown
    assert "## RCTD Readout（RCTD 专门解读）" in markdown
    assert "并不是所有锚点都最强" in markdown or "所以还不能写成“普遍最可靠”" in markdown
    assert "## Model Selection Guide（模型选择指引）" in markdown
    assert "需要更明确的 `winner`（第一名）" in markdown


def test_sharpness_summary_reads_abundance_and_risk_tables(tmp_path: Path):
    run_path = tmp_path / "results" / "run-a"
    abundance = pd.DataFrame(
        {
            "Astro": [0.9, 0.1, 0.2],
            "Excit_01": [0.1, 0.8, 0.7],
        },
        index=["spot1", "spot2", "spot3"],
    )
    save_base_model_output(BaseSpatialModelOutput(abundance=abundance), run_path)
    risk = pd.DataFrame(
        {
            "sample_id": ["151507", "151507", "151508"],
            "x_spatial": [1.0, 2.0, 3.0],
            "y_spatial": [4.0, 5.0, 6.0],
        },
        index=abundance.index,
    )
    tables_dir = run_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    risk.to_csv(tables_dir / "risk_table.csv")

    spec = AnchoredModelSpec(model_key="demo", display_name="Demo", run_id="run-a", run_path=run_path)
    summary = sharpness_summary(spec, sample_ids=("151507",))

    row = summary.iloc[0]
    assert row["display_name"] == "Demo"
    assert row["sample_id"] == "151507"
    assert round(float(row["mean_top1_margin"]), 3) == 0.789
    assert row["top_dominant_celltype"] == "Astro"
    assert round(float(row["top_dominant_fraction"]), 3) == 0.5
