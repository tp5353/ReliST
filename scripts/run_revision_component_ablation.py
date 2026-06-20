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


DEFAULT_RUN_ID = "2026-06-20-component-ablation-v2-donor-disjoint"
FEATURE_COLUMNS = ("phi_local", "phi_uncertainty", "phi_reference")
ABLATION_SCHEMES = {
    "full_risk": {"phi_local": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0},
    "local_only": {"phi_local": 1.0},
    "uncertainty_only": {"phi_uncertainty": 1.0},
    "reference_only": {"phi_reference": 1.0},
    "local_uncertainty": {"phi_local": 1.0, "phi_uncertainty": 1.0},
    "local_reference": {"phi_local": 1.0, "phi_reference": 1.0},
    "uncertainty_reference": {"phi_uncertainty": 1.0, "phi_reference": 1.0},
    "leave_out_local": {"phi_uncertainty": 1.0, "phi_reference": 1.0},
    "leave_out_uncertainty": {"phi_local": 1.0, "phi_reference": 1.0},
    "leave_out_reference": {"phi_local": 1.0, "phi_uncertainty": 1.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run component ablation on the revision known-composition benchmark.")
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_component_ablation",
        help="Result root for component ablation outputs.",
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
    y_true = valid.iloc[:, 1].astype(int).to_numpy()
    y_score = valid.iloc[:, 0].astype(float).to_numpy()
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def _score_summary(table: pd.DataFrame, *, score_cols: list[str], error_col: str = "total_variation_error") -> pd.DataFrame:
    error = table[error_col].astype(float)
    high20 = (error >= error.quantile(0.80)).astype(int)
    high10 = (error >= error.quantile(0.90)).astype(int)
    rows = []
    for score_col in score_cols:
        score = table[score_col].astype(float)
        spearman, spearman_p = _safe_corr(score, error, method="spearman")
        pearson, pearson_p = _safe_corr(score, error, method="pearson")
        auc20, ap20 = _safe_auc(score, high20)
        auc10, ap10 = _safe_auc(score, high10)
        low_mask = score <= score.quantile(0.20)
        high_mask = score >= score.quantile(0.80)
        rows.append(
            {
                "score_name": score_col,
                "n_spots": int(score.notna().sum()),
                "features_used": "+".join(ABLATION_SCHEMES[score_col].keys()),
                "n_features_used": int(len(ABLATION_SCHEMES[score_col])),
                "spearman_error": spearman,
                "spearman_pvalue": spearman_p,
                "pearson_error": pearson,
                "pearson_pvalue": pearson_p,
                "auroc_top20_error": auc20,
                "average_precision_top20_error": ap20,
                "auroc_top10_error": auc10,
                "average_precision_top10_error": ap10,
                "bottom20_score_mean_error": float(error.loc[low_mask].mean()),
                "top20_score_mean_error": float(error.loc[high_mask].mean()),
                "top_minus_bottom20_error": float(error.loc[high_mask].mean() - error.loc[low_mask].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["auroc_top20_error", "spearman_error"], ascending=[False, False])


def _keep_fraction_summary(
    table: pd.DataFrame,
    *,
    score_cols: list[str],
    error_col: str = "total_variation_error",
    keep_fractions: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> pd.DataFrame:
    full_mean = float(table[error_col].mean())
    high_error_threshold = float(table[error_col].quantile(0.8))
    rows = []
    for score_col in score_cols:
        ordered = table.sort_values(score_col, ascending=True)
        for keep_fraction in keep_fractions:
            n_keep = max(1, int(round(ordered.shape[0] * keep_fraction)))
            kept = ordered.head(n_keep)
            mean_error = float(kept[error_col].mean())
            rows.append(
                {
                    "score_name": score_col,
                    "keep_fraction": float(keep_fraction),
                    "n_kept": int(n_keep),
                    "mean_error": mean_error,
                    "error_reduction_vs_full": float(1.0 - mean_error / full_mean) if full_mean > 0 else np.nan,
                    "high_error_fraction": float((kept[error_col] >= high_error_threshold).mean()),
                    "full_mean_error": full_mean,
                }
            )
    return pd.DataFrame(rows)


def _delta_vs_full(summary: pd.DataFrame, keep_summary: pd.DataFrame) -> pd.DataFrame:
    full = summary.loc[summary["score_name"] == "full_risk"].iloc[0]
    keep80 = keep_summary.loc[keep_summary["keep_fraction"].round(6) == 0.8].set_index("score_name")
    full_keep80 = keep80.loc["full_risk"]
    rows = []
    for _, row in summary.iterrows():
        score_name = str(row["score_name"])
        keep_row = keep80.loc[score_name]
        rows.append(
            {
                "score_name": score_name,
                "features_used": row["features_used"],
                "delta_auroc_top20_vs_full": float(row["auroc_top20_error"] - full["auroc_top20_error"]),
                "delta_spearman_vs_full": float(row["spearman_error"] - full["spearman_error"]),
                "delta_top_minus_bottom20_error_vs_full": float(
                    row["top_minus_bottom20_error"] - full["top_minus_bottom20_error"]
                ),
                "delta_keep80_mean_error_vs_full": float(keep_row["mean_error"] - full_keep80["mean_error"]),
                "delta_keep80_error_reduction_vs_full_score": float(
                    keep_row["error_reduction_vs_full"] - full_keep80["error_reduction_vs_full"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["delta_auroc_top20_vs_full", "delta_spearman_vs_full"], ascending=False)


def _write_report(
    run_dir: Path,
    *,
    summary: pd.DataFrame,
    keep_summary: pd.DataFrame,
    delta: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    full = summary.loc[summary["score_name"] == "full_risk"].iloc[0]
    best = summary.iloc[0]
    single = summary.loc[summary["n_features_used"] == 1].sort_values("auroc_top20_error", ascending=False).iloc[0]
    keep80 = keep_summary.loc[(keep_summary["score_name"] == "full_risk") & (keep_summary["keep_fraction"] == 0.8)].iloc[0]
    leave_ref = delta.loc[delta["score_name"] == "leave_out_reference"].iloc[0]

    lines = [
        "# Revision Component Ablation",
        "",
        "## Purpose",
        "",
        "本运行在 known-composition benchmark（已知组成基准）上比较 `phi_local（局部特征）`、`phi_uncertainty（输出模糊性）` 和 `phi_reference（参考特征）` 的贡献，用于回应 Reviewer #2 Comment 1。",
        "",
        "## Main Result Snapshot",
        "",
        f"- full risk（完整风险分数）top20 true-error AUROC（最高 20% 真实误差 AUROC）为 `{full['auroc_top20_error']:.3f}`，Spearman（秩相关）为 `{full['spearman_error']:.3f}`。",
        f"- 当前最佳 ablation score（消融分数）为 `{best['score_name']}`，AUROC 为 `{best['auroc_top20_error']:.3f}`。",
        f"- 最强 single component（单组件）为 `{single['score_name']}`，AUROC 为 `{single['auroc_top20_error']:.3f}`。",
        f"- full risk 保留低风险 80% 时 mean error（平均误差）为 `{keep80['mean_error']:.3f}`，相对全量下降 `{keep80['error_reduction_vs_full']:.3f}`。",
        f"- 去掉 `phi_reference（参考特征）` 后 AUROC 变化为 `{leave_ref['delta_auroc_top20_vs_full']:.3f}`；该 matched-reference benchmark（匹配参考基准）中 reference component（参考分量）应解释为 near-neutral / context-dependent（近中性 / 情境依赖），而不是稳定提升平均 AUROC 的分量。",
        "",
        "## Interpretation Boundary",
        "",
        "- ablation（消融）结果应写成 component contribution（组件贡献）和 redundancy（冗余性）分析，不应写成某个组件在所有组织、所有基础模型或所有 reference mismatch（参考不匹配）程度中普遍最优。",
        "- `phi_reference（参考特征）` 的贡献需要结合 reference perturbation component ablation（参考扰动组件消融）解释：在 matched-reference benchmark 中近中性并不排除它在 marker dropout（marker 丢失）、cell-type omission（细胞类型缺失）或 out-of-reference（参考外）场景中有价值。",
        "- 该结果基于 pseudo-spots（伪空间点）中的真实组成误差；DLPFC 自然组织仍应结合 anchored proxies（锚定代理指标）解释。",
        "",
        "## Output Tables",
        "",
        "- `tables/component_ablation_score_table.csv`",
        "- `tables/component_ablation_summary.csv`",
        "- `tables/component_ablation_keep_fraction_summary.csv`",
        "- `tables/component_ablation_delta_vs_full.csv`",
    ]
    (run_dir / "revision_component_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    known_run_dir = current_results_dir(args.known_composition_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    table = pd.read_csv(known_run_dir / "tables" / "known_composition_risk_error_table.csv", index_col=0)
    missing = [col for col in FEATURE_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"Known-composition risk table is missing feature columns: {missing}")

    groups = table["sample_id"].astype(str) if "sample_id" in table.columns else None
    score_table = table[["sample_id", "scenario", "total_variation_error", *FEATURE_COLUMNS]].copy()
    for score_name, weights in ABLATION_SCHEMES.items():
        score_table[score_name] = _combine_features(table, weights, groups=groups)

    score_cols = list(ABLATION_SCHEMES)
    summary = _score_summary(score_table, score_cols=score_cols)
    keep_summary = _keep_fraction_summary(score_table, score_cols=score_cols)
    delta = _delta_vs_full(summary, keep_summary)

    score_table.to_csv(results_file(run_dir, "tables", "component_ablation_score_table.csv"))
    summary.to_csv(results_file(run_dir, "tables", "component_ablation_summary.csv"), index=False)
    keep_summary.to_csv(results_file(run_dir, "tables", "component_ablation_keep_fraction_summary.csv"), index=False)
    delta.to_csv(results_file(run_dir, "tables", "component_ablation_delta_vs_full.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_run_dir),
        "feature_columns": list(FEATURE_COLUMNS),
        "ablation_schemes": ABLATION_SCHEMES,
        "primary_error_col": "total_variation_error",
        "claim_boundary": (
            "Component ablation describes contribution in the known-composition pseudo-spot benchmark; "
            "it is not a universal component ranking."
        ),
    }
    results_file(run_dir, "metadata", "revision_component_ablation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(run_dir, summary=summary, keep_summary=keep_summary, delta=delta, metadata=metadata)

    print(f"Wrote revision component ablation to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "known_composition_run_dir": str(known_run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
