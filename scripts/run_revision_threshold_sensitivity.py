from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run


DEFAULT_RUN_ID = "2026-06-20-threshold-sensitivity-v2-donor-disjoint"
DEFAULT_SCORE_COLUMNS = ("risk_score", "local_uncertainty_risk_score", "reference_risk_score")
HIGH_RISK_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
KEEP_FRACTIONS = (0.70, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold sensitivity on the revision known-composition benchmark.")
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_threshold_sensitivity",
        help="Result root for threshold sensitivity outputs.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    parser.add_argument(
        "--score-columns",
        default=",".join(DEFAULT_SCORE_COLUMNS),
        help="Comma-separated risk score columns to evaluate.",
    )
    return parser.parse_args()


def _parse_score_columns(raw: str, table: pd.DataFrame) -> list[str]:
    columns = [part.strip() for part in raw.split(",") if part.strip()]
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"Score columns not found in known-composition table: {missing}")
    return columns


def _high_risk_threshold_summary(
    table: pd.DataFrame,
    *,
    score_cols: list[str],
    error_col: str = "total_variation_error",
) -> pd.DataFrame:
    error = table[error_col].astype(float)
    full_mean = float(error.mean())
    high_error_threshold = float(error.quantile(0.80))
    high_error_mask = error >= high_error_threshold
    rows = []
    for score_col in score_cols:
        score = table[score_col].astype(float)
        for high_fraction in HIGH_RISK_FRACTIONS:
            threshold = float(score.quantile(1.0 - high_fraction))
            high_mask = score >= threshold
            low_mask = ~high_mask
            rows.append(
                {
                    "score_name": score_col,
                    "high_risk_fraction_target": float(high_fraction),
                    "risk_threshold": threshold,
                    "n_high_risk": int(high_mask.sum()),
                    "observed_high_risk_fraction": float(high_mask.mean()),
                    "mean_error_high_risk": float(error.loc[high_mask].mean()),
                    "mean_error_not_high_risk": float(error.loc[low_mask].mean()),
                    "high_vs_not_high_error_gap": float(error.loc[high_mask].mean() - error.loc[low_mask].mean()),
                    "high_risk_error_enrichment_vs_full": float(error.loc[high_mask].mean() / full_mean) if full_mean > 0 else np.nan,
                    "top20_true_error_capture_fraction": float((high_mask & high_error_mask).sum() / high_error_mask.sum()),
                    "top20_true_error_precision": float((high_mask & high_error_mask).sum() / max(high_mask.sum(), 1)),
                    "full_mean_error": full_mean,
                }
            )
    return pd.DataFrame(rows)


def _keep_fraction_summary(
    table: pd.DataFrame,
    *,
    score_cols: list[str],
    error_col: str = "total_variation_error",
) -> pd.DataFrame:
    error = table[error_col].astype(float)
    full_mean = float(error.mean())
    high_error_threshold = float(error.quantile(0.80))
    rows = []
    for score_col in score_cols:
        ordered = table.sort_values(score_col, ascending=True)
        score_values = ordered[score_col].astype(float)
        cumulative_error = ordered[error_col].astype(float).expanding().mean().to_numpy(dtype=float)
        coverage = np.arange(1, ordered.shape[0] + 1, dtype=float) / ordered.shape[0]
        aurc = float(np.trapezoid(cumulative_error, coverage) / (coverage[-1] - coverage[0]))
        for keep_fraction in KEEP_FRACTIONS:
            n_keep = max(1, int(round(ordered.shape[0] * keep_fraction)))
            kept = ordered.head(n_keep)
            abstained = ordered.iloc[n_keep:]
            kept_error = kept[error_col].astype(float)
            abstained_error = abstained[error_col].astype(float)
            rows.append(
                {
                    "policy": "score_based",
                    "score_name": score_col,
                    "keep_fraction": float(keep_fraction),
                    "abstain_fraction": float(1.0 - keep_fraction),
                    "n_kept": int(n_keep),
                    "n_abstained": int(max(ordered.shape[0] - n_keep, 0)),
                    "mean_error_kept": float(kept_error.mean()),
                    "mean_error_abstained": float(abstained_error.mean()) if not abstained.empty else np.nan,
                    "kept_error_reduction_vs_full": float(1.0 - kept_error.mean() / full_mean) if full_mean > 0 else np.nan,
                    "kept_top20_true_error_fraction": float((kept_error >= high_error_threshold).mean()),
                    "abstained_top20_true_error_fraction": float((abstained_error >= high_error_threshold).mean())
                    if not abstained.empty
                    else np.nan,
                    "full_mean_error": full_mean,
                    "aurc": aurc,
                }
            )
    random_top20 = float((error >= high_error_threshold).mean())
    oracle_ordered = table.sort_values(error_col, ascending=True)
    oracle_error_curve = oracle_ordered[error_col].astype(float).expanding().mean().to_numpy(dtype=float)
    oracle_coverage = np.arange(1, oracle_ordered.shape[0] + 1, dtype=float) / oracle_ordered.shape[0]
    oracle_aurc = float(np.trapezoid(oracle_error_curve, oracle_coverage) / (oracle_coverage[-1] - oracle_coverage[0]))
    for keep_fraction in KEEP_FRACTIONS:
        n_keep = max(1, int(round(table.shape[0] * keep_fraction)))
        oracle_kept = oracle_ordered.head(n_keep)
        oracle_abstained = oracle_ordered.iloc[n_keep:]
        for policy, score_name, mean_error, top20_fraction, abstained_top20_fraction, aurc in (
            ("random_reference", "random_reference", full_mean, random_top20, random_top20, full_mean),
            (
                "oracle_error",
                "oracle_error",
                float(oracle_kept[error_col].mean()),
                float((oracle_kept[error_col] >= high_error_threshold).mean()),
                float((oracle_abstained[error_col] >= high_error_threshold).mean()) if not oracle_abstained.empty else np.nan,
                oracle_aurc,
            ),
        ):
            rows.append(
                {
                    "policy": policy,
                    "score_name": score_name,
                    "keep_fraction": float(keep_fraction),
                    "abstain_fraction": float(1.0 - keep_fraction),
                    "n_kept": int(n_keep),
                    "n_abstained": int(max(table.shape[0] - n_keep, 0)),
                    "mean_error_kept": mean_error,
                    "mean_error_abstained": np.nan,
                    "kept_error_reduction_vs_full": float(1.0 - mean_error / full_mean) if full_mean > 0 else np.nan,
                    "kept_top20_true_error_fraction": top20_fraction,
                    "abstained_top20_true_error_fraction": abstained_top20_fraction,
                    "full_mean_error": full_mean,
                    "aurc": aurc,
                }
            )
    return pd.DataFrame(rows)


def _usage_guidance(keep_summary: pd.DataFrame, high_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    score_based = keep_summary.loc[keep_summary["policy"] == "score_based"].copy()
    for score_name, group in score_based.groupby("score_name", sort=True):
        for keep_fraction, label, usage in (
            (0.95, "illustrative_light_review", "示例审查预算：约 5% spot 进入快速人工复核"),
            (0.90, "illustrative_moderate_review", "示例审查预算：约 10% spot 进入低成本补充质控"),
            (0.80, "illustrative_balanced_review", "示例审查预算：约 20% spot 进入平衡复核"),
            (0.70, "illustrative_stringent_review", "示例审查预算：约 30% spot 进入更保守下游分析前复核"),
        ):
            keep_row = group.loc[group["keep_fraction"].round(6) == keep_fraction]
            high_row = high_summary.loc[
                (high_summary["score_name"] == score_name)
                & (high_summary["high_risk_fraction_target"].round(6) == round(1.0 - keep_fraction, 6))
            ]
            if keep_row.empty:
                continue
            keep_row = keep_row.iloc[0]
            high_capture = float(high_row.iloc[0]["top20_true_error_capture_fraction"]) if not high_row.empty else np.nan
            rows.append(
                {
                    "score_name": score_name,
                    "policy_label": label,
                    "usage_guidance": usage,
                    "keep_fraction": keep_fraction,
                    "abstain_fraction": 1.0 - keep_fraction,
                    "mean_error_kept": float(keep_row["mean_error_kept"]),
                    "kept_error_reduction_vs_full": float(keep_row["kept_error_reduction_vs_full"]),
                    "kept_top20_true_error_fraction": float(keep_row["kept_top20_true_error_fraction"]),
                    "top20_true_error_capture_fraction_in_abstained": high_capture,
                    "aurc": float(keep_row["aurc"]),
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    run_dir: Path,
    *,
    keep_summary: pd.DataFrame,
    high_summary: pd.DataFrame,
    guidance: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    default = guidance.loc[
        (guidance["score_name"] == "risk_score") & (guidance["policy_label"] == "illustrative_balanced_review")
    ].iloc[0]
    top10 = high_summary.loc[
        (high_summary["score_name"] == "risk_score") & (high_summary["high_risk_fraction_target"].round(6) == 0.10)
    ].iloc[0]
    top20 = high_summary.loc[
        (high_summary["score_name"] == "risk_score") & (high_summary["high_risk_fraction_target"].round(6) == 0.20)
    ].iloc[0]

    lines = [
        "# Revision Threshold Sensitivity",
        "",
        "## Purpose",
        "",
        "本运行评估 high-risk threshold（高风险阈值）和 keep fraction（保留比例）对 true deconvolution error（真实反卷积误差）的影响，用于回应 Reviewer #2 Comment 2。",
        "",
        "## Main Result Snapshot",
        "",
        f"- 示例 keep 80%（保留 80% 低风险点）审查预算下，`risk_score（风险分数）` 的 mean error（平均误差）为 `{default['mean_error_kept']:.3f}`，相对全量下降 `{default['kept_error_reduction_vs_full']:.3f}`，AURC（覆盖率-误差曲线下面积）为 `{default['aurc']:.3f}`。",
        f"- top 10% high-risk（最高风险 10%）区域捕获 top20 true-error spots（最高 20% 真实误差点）的比例为 `{top10['top20_true_error_capture_fraction']:.3f}`，precision（精确率）为 `{top10['top20_true_error_precision']:.3f}`。",
        f"- top 20% high-risk（最高风险 20%）区域捕获 top20 true-error spots 的比例为 `{top20['top20_true_error_capture_fraction']:.3f}`，precision 为 `{top20['top20_true_error_precision']:.3f}`。",
        "",
        "## Practical Guidance",
        "",
        "- 这些 keep fractions（保留比例）应写成 illustrative review budgets（示例审查预算），不是跨组织、跨平台或跨基础模型通用阈值。",
        "- 实践中应根据 coverage-risk curve（覆盖率-风险曲线）的拐点、人工复核预算和下游容错率选择阈值。",
        "- 这些阈值不是 truth-calibrated cutoff（真实标签校准截断），不能写成高风险点一定错误。",
        "",
        "## Output Tables",
        "",
        "- `tables/threshold_high_risk_summary.csv`",
        "- `tables/threshold_keep_fraction_summary.csv`",
        "- `tables/threshold_usage_guidance.csv`",
    ]
    (run_dir / "revision_threshold_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    known_run_dir = current_results_dir(args.known_composition_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    table = pd.read_csv(known_run_dir / "tables" / "known_composition_risk_error_table.csv", index_col=0)
    score_cols = _parse_score_columns(args.score_columns, table)

    high_summary = _high_risk_threshold_summary(table, score_cols=score_cols)
    keep_summary = _keep_fraction_summary(table, score_cols=score_cols)
    guidance = _usage_guidance(keep_summary, high_summary)

    high_summary.to_csv(results_file(run_dir, "tables", "threshold_high_risk_summary.csv"), index=False)
    keep_summary.to_csv(results_file(run_dir, "tables", "threshold_keep_fraction_summary.csv"), index=False)
    guidance.to_csv(results_file(run_dir, "tables", "threshold_usage_guidance.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_run_dir),
        "score_columns": score_cols,
        "high_risk_fractions": list(HIGH_RISK_FRACTIONS),
        "keep_fractions": list(KEEP_FRACTIONS),
        "primary_error_col": "total_variation_error",
        "claim_boundary": (
            "Thresholds are illustrative review budgets selected from coverage-risk curves; "
            "they are not universal cutoffs or truth-calibrated probabilities."
        ),
    }
    results_file(run_dir, "metadata", "revision_threshold_sensitivity.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(run_dir, keep_summary=keep_summary, high_summary=high_summary, guidance=guidance, metadata=metadata)

    print(f"Wrote revision threshold sensitivity to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "known_composition_run_dir": str(known_run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
