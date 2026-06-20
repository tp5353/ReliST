from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run


DEFAULT_RUN_ID = "2026-06-20-uncertainty-baseline-v1"
RELIST_SCORES = ("risk_score", "reference_risk_score", "local_uncertainty_risk_score")
CONFIDENCE_BASELINES = (
    "abundance_entropy_risk",
    "inverse_top1_margin",
    "inverse_max_abundance",
    "phi_uncertainty",
)
COMPONENT_DIAGNOSTICS = (
    "phi_local",
    "phi_reference",
    "snrna_marker_discordance",
    "snrna_signature_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ReliST-vs-confidence baseline comparisons for the iScience revision."
    )
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--contract-fairness-run-dir",
        type=Path,
        default=project_root()
        / "results"
        / "model_contract_fairness_control"
        / "runs"
        / "2026-04-29-dlpfc-model-contract-fairness-control-v1",
        help="Existing DLPFC model-contract fairness control run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_uncertainty_baseline_eval",
        help="Result root for this revision baseline comparison.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    return parser.parse_args()


def _score_family(score_name: str) -> str:
    if score_name in RELIST_SCORES:
        return "ReliST score"
    if score_name in CONFIDENCE_BASELINES:
        return "confidence / uncertainty baseline"
    if score_name in COMPONENT_DIAGNOSTICS:
        return "component / proxy diagnostic"
    return "other"


def _load_known_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = run_dir / "tables"
    score_summary = pd.read_csv(tables_dir / "known_composition_score_error_summary.csv")
    selective = pd.read_csv(tables_dir / "known_composition_selective_error_curve.csv")
    score_summary["score_family"] = score_summary["score_name"].map(_score_family)
    selective["score_family"] = selective["score_name"].map(_score_family)
    return score_summary, selective


def _known_composition_key_comparison(score_summary: pd.DataFrame, selective: pd.DataFrame) -> pd.DataFrame:
    primary = score_summary.loc[score_summary["score_name"] == "risk_score"].copy()
    if primary.empty:
        raise ValueError("known_composition_score_error_summary.csv does not contain risk_score.")
    primary = primary.iloc[0]

    confidence = score_summary.loc[score_summary["score_name"].isin(CONFIDENCE_BASELINES)].copy()
    if confidence.empty:
        raise ValueError("No confidence baseline score is present in known-composition summary.")
    best_confidence = confidence.sort_values(
        ["auroc_top20_error", "spearman_error"],
        ascending=[False, False],
    ).iloc[0]

    relist_best = score_summary.loc[score_summary["score_name"].isin(RELIST_SCORES)].sort_values(
        ["auroc_top20_error", "spearman_error"],
        ascending=[False, False],
    ).iloc[0]

    keep80 = selective.loc[selective["keep_fraction"].round(6) == 0.8].copy()
    keep80_by_score = keep80.set_index("score_name")

    rows = []
    for label, row in (
        ("primary_risk_score", primary),
        ("best_relist_score", relist_best),
        ("best_confidence_baseline", best_confidence),
    ):
        keep_row = keep80_by_score.loc[row["score_name"]] if row["score_name"] in keep80_by_score.index else pd.Series(dtype=float)
        rows.append(
            {
                "comparison_block": "known_composition_true_error",
                "entry": label,
                "score_name": row["score_name"],
                "score_family": _score_family(str(row["score_name"])),
                "spearman_error": float(row["spearman_error"]),
                "auroc_top20_error": float(row["auroc_top20_error"]),
                "auroc_top10_error": float(row["auroc_top10_error"]),
                "top_minus_bottom20_error": float(row["top_minus_bottom20_error"]),
                "keep80_mean_error": float(keep_row.get("mean_error", np.nan)),
                "keep80_error_reduction_vs_full": float(keep_row.get("error_reduction_vs_full", np.nan)),
                "keep80_high_error_fraction": float(keep_row.get("high_error_fraction", np.nan)),
            }
        )

    delta_row = {
        "comparison_block": "known_composition_true_error",
        "entry": "primary_minus_best_confidence",
        "score_name": "risk_score - " + str(best_confidence["score_name"]),
        "score_family": "delta",
        "spearman_error": float(primary["spearman_error"] - best_confidence["spearman_error"]),
        "auroc_top20_error": float(primary["auroc_top20_error"] - best_confidence["auroc_top20_error"]),
        "auroc_top10_error": float(primary["auroc_top10_error"] - best_confidence["auroc_top10_error"]),
        "top_minus_bottom20_error": float(primary["top_minus_bottom20_error"] - best_confidence["top_minus_bottom20_error"]),
        "keep80_mean_error": float(
            keep80_by_score.loc["risk_score", "mean_error"] - keep80_by_score.loc[best_confidence["score_name"], "mean_error"]
        ),
        "keep80_error_reduction_vs_full": float(
            keep80_by_score.loc["risk_score", "error_reduction_vs_full"]
            - keep80_by_score.loc[best_confidence["score_name"], "error_reduction_vs_full"]
        ),
        "keep80_high_error_fraction": float(
            keep80_by_score.loc["risk_score", "high_error_fraction"]
            - keep80_by_score.loc[best_confidence["score_name"], "high_error_fraction"]
        ),
    }
    rows.append(delta_row)
    return pd.DataFrame(rows)


def _load_contract_fairness(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = run_dir / "tables"
    fairness = pd.read_csv(tables_dir / "contract_fairness_summary.csv")
    table5 = pd.read_csv(tables_dir / "table5_contract_fairness_control.csv")
    return fairness, table5


def _contract_fairness_aggregate(fairness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    blocks = {
        "all_dlpfc_anchored_proxies": fairness,
        "marker_discordance_only": fairness.loc[fairness["proxy_short_name"] == "marker_discordance"],
        "reference_residual_only": fairness.loc[fairness["proxy_short_name"] == "signature_residual"],
        "structure_proxies": fairness.loc[fairness["proxy_short_name"].isin(["layer_guess", "Maynard"])],
    }
    for block_name, block in blocks.items():
        if block.empty:
            continue
        rows.append(
            {
                "comparison_block": block_name,
                "n_rows": int(block.shape[0]),
                "n_models": int(block["model_key"].nunique()),
                "mean_risk_axis_auc": float(block["current_best_auc"].mean()),
                "mean_best_common_auc": float(block["best_common_mean_auc"].mean()),
                "mean_delta_auc": float(block["current_minus_common_auc"].mean()),
                "median_delta_auc": float(block["current_minus_common_auc"].median()),
                "fraction_delta_gt_0p05": float((block["current_minus_common_auc"] > 0.05).mean()),
                "fraction_delta_lt_minus_0p05": float((block["current_minus_common_auc"] < -0.05).mean()),
            }
        )
    return pd.DataFrame(rows)


def _contract_key_rows(fairness: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_key",
        "display_name",
        "proxy_short_name",
        "current_best_score",
        "current_best_auc",
        "best_common_score",
        "best_common_mean_auc",
        "current_minus_common_auc",
        "fairness_reading",
    ]
    table = fairness.loc[:, columns].copy()
    return table.sort_values(["proxy_short_name", "current_minus_common_auc"], ascending=[True, False]).reset_index(drop=True)


def _write_report(
    run_dir: Path,
    *,
    known_key: pd.DataFrame,
    contract_aggregate: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    primary = known_key.loc[known_key["entry"] == "primary_risk_score"].iloc[0]
    best_conf = known_key.loc[known_key["entry"] == "best_confidence_baseline"].iloc[0]
    delta = known_key.loc[known_key["entry"] == "primary_minus_best_confidence"].iloc[0]
    all_proxy = contract_aggregate.loc[contract_aggregate["comparison_block"] == "all_dlpfc_anchored_proxies"].iloc[0]
    marker = contract_aggregate.loc[contract_aggregate["comparison_block"] == "marker_discordance_only"].iloc[0]

    lines = [
        "# Revision Uncertainty Baseline Evaluation",
        "",
        "## Purpose",
        "",
        "本运行汇总 ReliST 与 confidence / uncertainty baselines（置信度 / 不确定性基线）的比较，用于回应 Reviewer #1 Comment 2。",
        "",
        "## Known-Composition True-Error Benchmark",
        "",
        f"- `risk_score（风险分数）` 对 true deconvolution error（真实反卷积误差）的 Spearman（秩相关）为 `{primary['spearman_error']:.3f}`，top20 true-error AUROC（最高 20% 真实误差 AUROC）为 `{primary['auroc_top20_error']:.3f}`。",
        f"- 最强 confidence baseline（置信度基线）为 `{best_conf['score_name']}`，top20 true-error AUROC 为 `{best_conf['auroc_top20_error']:.3f}`。",
        f"- `risk_score - best confidence baseline` 的 AUROC 差值为 `{delta['auroc_top20_error']:.3f}`；keep 80%（保留 80% 低风险点）时，mean error（平均误差）差值为 `{delta['keep80_mean_error']:.3f}`。",
        "",
        "## DLPFC Anchored Proxy Control",
        "",
        f"- 在所有 DLPFC anchored proxies（锚定代理指标）上，risk-axis score（风险轴分数）平均 AUROC 为 `{all_proxy['mean_risk_axis_auc']:.3f}`，最佳 common confidence feature（通用置信特征）平均 AUROC 为 `{all_proxy['mean_best_common_auc']:.3f}`，平均差值为 `{all_proxy['mean_delta_auc']:.3f}`。",
        f"- 在 marker discordance（marker 不一致）代理上，平均差值为 `{marker['mean_delta_auc']:.3f}`，`{marker['fraction_delta_gt_0p05']:.2f}` 的模型-代理组合差值超过 0.05。",
        "",
        "## Interpretation Boundary",
        "",
        "- 这不是把 ReliST 写成普遍优于所有 confidence score（置信度分数）的绝对声明；更稳的写法是：在 true-error pseudo-spots（真实误差伪空间点）和 DLPFC anchored proxies（DLPFC 锚定代理指标）上，ReliST 提供了不等同于单一输出置信度的可检验风险信号。",
        "- DLPFC 合同公平性控制仍应写成 current canonical output contract（当前统一输出接口）下的 observable support（可观测支持），不能写成 base model quality ranking（基础模型质量排名）。",
        "",
        "## Output Tables",
        "",
        "- `tables/known_composition_baseline_score_summary.csv`",
        "- `tables/known_composition_keep80_summary.csv`",
        "- `tables/known_composition_key_comparison.csv`",
        "- `tables/dlpfc_contract_fairness_key_rows.csv`",
        "- `tables/dlpfc_contract_fairness_aggregate.csv`",
        "- `tables/revision_uncertainty_baseline_summary.csv`",
    ]
    (run_dir / "revision_uncertainty_baseline_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    known_run_dir = current_results_dir(args.known_composition_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    score_summary, selective = _load_known_tables(known_run_dir)
    known_key = _known_composition_key_comparison(score_summary, selective)
    keep80 = selective.loc[selective["keep_fraction"].round(6) == 0.8].copy()

    fairness, table5 = _load_contract_fairness(args.contract_fairness_run_dir)
    fairness_aggregate = _contract_fairness_aggregate(fairness)
    fairness_key = _contract_key_rows(fairness)

    revision_summary = pd.concat(
        [
            known_key.assign(source="known_composition"),
            fairness_aggregate.rename(
                columns={
                    "mean_risk_axis_auc": "auroc_top20_error",
                    "mean_best_common_auc": "best_common_or_baseline_auc",
                    "mean_delta_auc": "delta_auc",
                }
            ).assign(source="dlpfc_contract_fairness"),
        ],
        ignore_index=True,
        sort=False,
    )

    score_summary.to_csv(results_file(run_dir, "tables", "known_composition_baseline_score_summary.csv"), index=False)
    keep80.to_csv(results_file(run_dir, "tables", "known_composition_keep80_summary.csv"), index=False)
    known_key.to_csv(results_file(run_dir, "tables", "known_composition_key_comparison.csv"), index=False)
    fairness_key.to_csv(results_file(run_dir, "tables", "dlpfc_contract_fairness_key_rows.csv"), index=False)
    fairness_aggregate.to_csv(results_file(run_dir, "tables", "dlpfc_contract_fairness_aggregate.csv"), index=False)
    table5.to_csv(results_file(run_dir, "tables", "dlpfc_contract_fairness_table5_source.csv"), index=False)
    revision_summary.to_csv(results_file(run_dir, "tables", "revision_uncertainty_baseline_summary.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_run_dir),
        "contract_fairness_run_dir": str(args.contract_fairness_run_dir),
        "relist_scores": list(RELIST_SCORES),
        "confidence_baselines": list(CONFIDENCE_BASELINES),
        "component_diagnostics": list(COMPONENT_DIAGNOSTICS),
        "claim_boundary": (
            "This table supports a reviewer-facing baseline comparison. It should not be phrased as an "
            "absolute base-model ranking or universal superiority claim."
        ),
    }
    results_file(run_dir, "metadata", "revision_uncertainty_baseline_eval.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(run_dir, known_key=known_key, contract_aggregate=fairness_aggregate, metadata=metadata)

    print(f"Wrote revision uncertainty baseline evaluation to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "known_composition_run_dir": str(known_run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
