from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run
from st_risk.risk.score import grouped_zscore, sigmoid


DEFAULT_RUN_ID = "2026-06-20-reference-perturbation-component-ablation-v2-donor-disjoint"
SCORE_DEFINITIONS = {
    "full_risk": {"phi_local": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0},
    "leave_out_reference": {"phi_local": 1.0, "phi_uncertainty": 1.0},
    "reference_only": {"phi_reference": 1.0},
    "local_only": {"phi_local": 1.0},
    "uncertainty_only": {"phi_uncertainty": 1.0},
    "local_reference": {"phi_local": 1.0, "phi_reference": 1.0},
    "uncertainty_reference": {"phi_uncertainty": 1.0, "phi_reference": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full and no-reference risk scores within each reference-perturbation scenario."
    )
    parser.add_argument(
        "--reference-perturbation-root",
        type=Path,
        default=project_root() / "results" / "revision_reference_perturbation",
        help="Reference perturbation result root or run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_reference_perturbation_component_ablation",
        help="Result root for perturbation component-ablation outputs.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    return parser.parse_args()


def _combine_features(table: pd.DataFrame, weights: dict[str, float], *, groups: pd.Series | None = None) -> pd.Series:
    linear = np.zeros(table.shape[0], dtype=float)
    total_weight = 0.0
    for feature, weight in weights.items():
        if feature not in table.columns or np.isclose(float(weight), 0.0):
            continue
        linear += float(weight) * grouped_zscore(table[feature].to_numpy(dtype=float), groups=groups)
        total_weight += abs(float(weight))
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one non-zero feature is required.")
    return pd.Series(sigmoid(linear / total_weight), index=table.index)


def _safe_corr(score: pd.Series, error: pd.Series, *, method: str) -> tuple[float, float]:
    valid = pd.concat([score, error], axis=1).dropna()
    if valid.shape[0] < 3 or valid.iloc[:, 0].nunique() <= 1 or valid.iloc[:, 1].nunique() <= 1:
        return np.nan, np.nan
    if method == "spearman":
        stat, pvalue = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    elif method == "pearson":
        stat, pvalue = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    else:
        raise ValueError(method)
    return float(stat), float(pvalue)


def _safe_auc(score: pd.Series, labels: pd.Series) -> tuple[float, float]:
    valid = pd.concat([score, labels], axis=1).dropna()
    if valid.shape[0] < 3 or valid.iloc[:, 1].nunique() < 2 or valid.iloc[:, 0].nunique() <= 1:
        return np.nan, np.nan
    return (
        float(roc_auc_score(valid.iloc[:, 1].astype(int), valid.iloc[:, 0].astype(float))),
        float(average_precision_score(valid.iloc[:, 1].astype(int), valid.iloc[:, 0].astype(float))),
    )


def _summary_for_score(block: pd.DataFrame, *, score_col: str, error_col: str = "total_variation_error") -> dict[str, float]:
    error = block[error_col].astype(float)
    score = block[score_col].astype(float)
    high20 = (error >= error.quantile(0.80)).astype(int)
    spearman, spearman_p = _safe_corr(score, error, method="spearman")
    pearson, pearson_p = _safe_corr(score, error, method="pearson")
    auc20, ap20 = _safe_auc(score, high20)
    low_mask = score <= score.quantile(0.20)
    high_mask = score >= score.quantile(0.80)
    keep_mask = score <= score.quantile(0.80)
    full_mean = float(error.mean())
    keep_mean = float(error.loc[keep_mask].mean())
    return {
        "spearman_error": spearman,
        "spearman_pvalue": spearman_p,
        "pearson_error": pearson,
        "pearson_pvalue": pearson_p,
        "auroc_top20_error": auc20,
        "average_precision_top20_error": ap20,
        "bottom20_score_mean_error": float(error.loc[low_mask].mean()),
        "top20_score_mean_error": float(error.loc[high_mask].mean()),
        "top_minus_bottom20_error": float(error.loc[high_mask].mean() - error.loc[low_mask].mean()),
        "keep80_mean_error": keep_mean,
        "keep80_error_reduction_vs_full": float(1.0 - keep_mean / full_mean) if full_mean > 0 else np.nan,
    }


def main() -> int:
    args = parse_args()
    source_run_dir = current_results_dir(args.reference_perturbation_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    table = pd.read_csv(source_run_dir / "tables" / "reference_perturbation_spot_metrics.csv")
    required = {"scenario", "sample_id", "total_variation_error", "phi_local", "phi_uncertainty", "phi_reference"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Reference perturbation table is missing required columns: {missing}")

    score_table = table.copy()
    groups = score_table["sample_id"].astype(str)
    for score_name, weights in SCORE_DEFINITIONS.items():
        score_table[score_name] = _combine_features(score_table, weights, groups=groups)

    rows: list[dict[str, object]] = []
    for scenario, block in score_table.groupby("scenario", sort=True):
        for score_name, weights in SCORE_DEFINITIONS.items():
            rows.append(
                {
                    "scenario": scenario,
                    "score_name": score_name,
                    "features_used": "+".join(weights.keys()),
                    "n_spots": int(block.shape[0]),
                    "mean_true_error": float(block["total_variation_error"].mean()),
                    **_summary_for_score(block, score_col=score_name),
                }
            )
    summary = pd.DataFrame(rows)

    full = summary.loc[summary["score_name"] == "full_risk"].set_index("scenario")
    no_ref = summary.loc[summary["score_name"] == "leave_out_reference"].set_index("scenario")
    ref_only = summary.loc[summary["score_name"] == "reference_only"].set_index("scenario")
    delta_rows = []
    for scenario in full.index:
        delta_rows.append(
            {
                "scenario": scenario,
                "full_auroc_top20_error": float(full.loc[scenario, "auroc_top20_error"]),
                "leave_out_reference_auroc_top20_error": float(no_ref.loc[scenario, "auroc_top20_error"]),
                "reference_only_auroc_top20_error": float(ref_only.loc[scenario, "auroc_top20_error"]),
                "delta_full_minus_leave_out_reference_auroc": float(
                    full.loc[scenario, "auroc_top20_error"] - no_ref.loc[scenario, "auroc_top20_error"]
                ),
                "full_spearman_error": float(full.loc[scenario, "spearman_error"]),
                "leave_out_reference_spearman_error": float(no_ref.loc[scenario, "spearman_error"]),
                "delta_full_minus_leave_out_reference_spearman": float(
                    full.loc[scenario, "spearman_error"] - no_ref.loc[scenario, "spearman_error"]
                ),
                "full_keep80_mean_error": float(full.loc[scenario, "keep80_mean_error"]),
                "leave_out_reference_keep80_mean_error": float(no_ref.loc[scenario, "keep80_mean_error"]),
                "delta_full_minus_leave_out_reference_keep80_mean_error": float(
                    full.loc[scenario, "keep80_mean_error"] - no_ref.loc[scenario, "keep80_mean_error"]
                ),
            }
        )
    delta = pd.DataFrame(delta_rows).sort_values("scenario")

    score_table.to_csv(results_file(run_dir, "tables", "reference_perturbation_component_score_table.csv"), index=False)
    summary.to_csv(results_file(run_dir, "tables", "reference_perturbation_component_summary.csv"), index=False)
    delta.to_csv(results_file(run_dir, "tables", "reference_perturbation_full_vs_no_reference.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "source_reference_perturbation_run_dir": str(source_run_dir),
        "score_definitions": SCORE_DEFINITIONS,
        "claim_boundary": (
            "The reference component should be interpreted by scenario. A near-neutral or negative contribution "
            "in matched-reference settings is compatible with contribution under explicit reference mismatch."
        ),
    }
    results_file(run_dir, "metadata", "reference_perturbation_component_ablation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Reference Perturbation Component Ablation",
        "",
        "## Purpose",
        "",
        "This run compares full risk against a no-reference variant within each controlled reference-perturbation scenario.",
        "",
        "## Output Tables",
        "",
        "- `tables/reference_perturbation_component_score_table.csv`",
        "- `tables/reference_perturbation_component_summary.csv`",
        "- `tables/reference_perturbation_full_vs_no_reference.csv`",
    ]
    (run_dir / "reference_perturbation_component_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reference perturbation component ablation to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
