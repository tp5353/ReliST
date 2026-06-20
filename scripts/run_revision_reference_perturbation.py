from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_revision_known_composition_benchmark import (  # noqa: E402
    _combine_any_features,
    _compute_local_heterogeneity,
    _error_table,
    _project_abundance,
)
from st_risk.eval.reference_eval import (  # noqa: E402
    compute_reference_marker_scores,
    reference_marker_discordance_proxy,
    reference_signature_residual_proxy,
    reference_subsampling_instability,
    select_signature_markers,
    subset_markers,
)
from st_risk.models.base import BaseSpatialModelOutput  # noqa: E402
from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run  # noqa: E402
from st_risk.risk.features import ambiguity_score, build_feature_table  # noqa: E402
from st_risk.risk.neighbors import inverse_distance_weights, knn_indices  # noqa: E402


DEFAULT_RUN_ID = "2026-06-20-reference-perturbation-v2-donor-disjoint"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference perturbation stress tests on known-composition pseudo-spots.")
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_reference_perturbation",
        help="Result root for reference perturbation outputs.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    parser.add_argument("--marker-top-k", type=int, default=25, help="Top markers per cell type for perturbed references.")
    parser.add_argument("--min-positive-markers", type=int, default=8, help="Minimum positive markers per cell type.")
    parser.add_argument("--reference-repeats", type=int, default=8, help="Reference marker subsampling repeats.")
    parser.add_argument("--reference-fraction", type=float, default=0.5, help="Reference marker subsampling fraction.")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge penalty if ridge projection is used.")
    parser.add_argument("--projection-method", default="nnls", choices=("nnls", "ridge"), help="Projection method.")
    parser.add_argument(
        "--contamination-fraction",
        type=float,
        default=0.20,
        help="Synthetic out-of-reference fraction mixed into pseudo-spot expression.",
    )
    parser.add_argument("--random-state", type=int, default=20260620, help="Random seed.")
    return parser.parse_args()


def _marker_genes(marker_table: pd.DataFrame, *, max_rank: int | None = None) -> set[str]:
    table = marker_table.copy()
    if max_rank is not None and "rank" in table.columns:
        table = table.loc[table["rank"].astype(float) <= max_rank]
    return set(table["gene"].astype(str))


def _coarsen_name(celltype: str) -> str:
    if str(celltype).startswith("Excit_"):
        return "Excit_coarse"
    if str(celltype).startswith("Inhib_"):
        return "Inhib_coarse"
    return str(celltype)


def _coarsen_signatures(signatures: pd.DataFrame) -> pd.DataFrame:
    groups = pd.Series([_coarsen_name(col) for col in signatures.columns], index=signatures.columns)
    columns = {}
    for name in pd.Index(groups).unique():
        cols = groups.index[groups == name].tolist()
        columns[str(name)] = signatures[cols].mean(axis=1)
    return pd.DataFrame(columns, index=signatures.index)


def _coarsen_truth(truth: pd.DataFrame) -> pd.DataFrame:
    groups = pd.Series([_coarsen_name(col) for col in truth.columns], index=truth.columns)
    columns = {}
    for name in pd.Index(groups).unique():
        cols = groups.index[groups == name].tolist()
        columns[str(name)] = truth[cols].sum(axis=1)
    return pd.DataFrame(columns, index=truth.index)


def _top_truth_celltypes(truth: pd.DataFrame, n: int = 3) -> list[str]:
    return truth.mean(axis=0).sort_values(ascending=False).head(n).index.astype(str).tolist()


def _add_out_of_reference_contamination(
    expression: pd.DataFrame,
    signatures: pd.DataFrame,
    *,
    contamination_fraction: float,
) -> pd.DataFrame:
    fraction = float(np.clip(contamination_fraction, 0.0, 0.8))
    contaminant_profile = signatures.quantile(0.95, axis=1).reindex(expression.columns).astype(float)
    contaminated = (1.0 - fraction) * expression.astype(float) + fraction * contaminant_profile.to_numpy(dtype=float)
    return pd.DataFrame(contaminated, index=expression.index, columns=expression.columns)


def _add_out_of_reference_truth(truth: pd.DataFrame, *, contamination_fraction: float) -> pd.DataFrame:
    fraction = float(np.clip(contamination_fraction, 0.0, 0.8))
    contaminated_truth = truth.astype(float) * (1.0 - fraction)
    contaminated_truth["out_of_reference_contamination"] = fraction
    return contaminated_truth


def _align_prediction_for_error(predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    aligned = predicted.reindex(columns=truth.columns, fill_value=0.0).copy()
    row_sums = aligned.sum(axis=1)
    zero_rows = np.isclose(row_sums.to_numpy(dtype=float), 0.0)
    if zero_rows.any():
        aligned.loc[zero_rows, :] = 1.0 / max(aligned.shape[1], 1)
    return aligned


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


def _score_summary(table: pd.DataFrame, *, score_col: str = "risk_score", error_col: str = "total_variation_error") -> dict:
    error = table[error_col].astype(float)
    score = table[score_col].astype(float)
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
        "n_spots": int(table.shape[0]),
        "mean_true_error": full_mean,
        "median_true_error": float(error.median()),
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
        "keep80_high_error_fraction": float((error.loc[keep_mask] >= error.quantile(0.80)).mean()),
    }


def _compute_risk_table(
    *,
    expression: pd.DataFrame,
    predicted: pd.DataFrame,
    signatures: pd.DataFrame,
    truth: pd.DataFrame,
    spot_table: pd.DataFrame,
    marker_top_k: int,
    min_positive_markers: int,
    reference_repeats: int,
    reference_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coords = spot_table[["x_spatial", "y_spatial"]].to_numpy(dtype=float)
    neighbors = knn_indices(coords, k=8)
    weights = inverse_distance_weights(coords, neighbors)
    heterogeneity = _compute_local_heterogeneity(expression.to_numpy(dtype=np.float32), neighbors)

    ambiguity = pd.Series(ambiguity_score(predicted), index=predicted.index, name="phi_uncertainty")
    model_output = BaseSpatialModelOutput(abundance=predicted, uncertainty=ambiguity)
    features = build_feature_table(
        model_output,
        neighbors=neighbors,
        weights=weights,
        heterogeneity=heterogeneity,
        stability_predictions=None,
        confidence_proxy_precomputed=True,
    )

    markers_all, marker_table = select_signature_markers(
        signatures,
        top_k=marker_top_k,
        min_positive_markers=min_positive_markers,
    )
    markers = subset_markers(markers_all, mode="odd")
    try:
        features["phi_reference"] = reference_subsampling_instability(
            predicted,
            expression,
            markers,
            repeats=reference_repeats,
            subset_fraction=reference_fraction,
            random_state=random_state,
        )
    except Exception:
        features["phi_reference"] = 0.0

    try:
        marker_scores = compute_reference_marker_scores(expression, markers)
        features["snrna_marker_discordance"] = reference_marker_discordance_proxy(predicted, marker_scores)
    except Exception:
        features["snrna_marker_discordance"] = np.nan

    try:
        features["snrna_signature_residual"] = reference_signature_residual_proxy(
            predicted,
            expression,
            signatures,
            genes=marker_table["gene"].astype(str).tolist(),
        )
    except Exception:
        features["snrna_signature_residual"] = np.nan

    groups = spot_table["sample_id"].astype(str) if "sample_id" in spot_table.columns else None
    features["risk_score"] = _combine_any_features(
        features,
        {"phi_local": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0},
        groups=groups,
    )
    spot_info = spot_table.copy()
    if "scenario" in spot_info.columns:
        spot_info = spot_info.rename(columns={"scenario": "pseudo_generation_scenario"})
    error = _error_table(_align_prediction_for_error(predicted, truth), truth)
    return pd.concat([spot_info, features, error], axis=1), marker_table


def _build_scenarios(
    *,
    expression: pd.DataFrame,
    signatures: pd.DataFrame,
    truth: pd.DataFrame,
    marker_table: pd.DataFrame,
    contamination_fraction: float,
) -> list[dict]:
    marker_drop = _marker_genes(marker_table, max_rank=12)
    genes_after_marker_dropout = [gene for gene in expression.columns if gene not in marker_drop]
    top_types = _top_truth_celltypes(truth, n=3)
    remaining_after_top_dropout = [celltype for celltype in signatures.columns if celltype not in top_types]
    contaminated_expression = _add_out_of_reference_contamination(
        expression,
        signatures,
        contamination_fraction=contamination_fraction,
    )
    contaminated_truth = _add_out_of_reference_truth(truth, contamination_fraction=contamination_fraction)

    scenarios = [
        {
            "scenario": "baseline_full_reference",
            "description": "Original reference signatures and full gene set.",
            "expression": expression,
            "signatures": signatures,
            "truth": truth,
            "error_truth": truth,
            "dropped_genes": [],
            "dropped_celltypes": [],
            "contamination_fraction": 0.0,
        },
        {
            "scenario": "marker_gene_dropout",
            "description": "Remove high-specificity reference marker genes before signature projection.",
            "expression": expression.loc[:, genes_after_marker_dropout],
            "signatures": signatures.loc[genes_after_marker_dropout, :],
            "truth": truth,
            "error_truth": truth,
            "dropped_genes": sorted(marker_drop),
            "dropped_celltypes": [],
            "contamination_fraction": 0.0,
        },
        {
            "scenario": "dominant_celltype_dropout",
            "description": "Drop the three cell types with the highest average true pseudo-spot abundance.",
            "expression": expression,
            "signatures": signatures.loc[:, remaining_after_top_dropout],
            "truth": truth.loc[:, remaining_after_top_dropout],
            "error_truth": truth,
            "dropped_genes": [],
            "dropped_celltypes": top_types,
            "contamination_fraction": 0.0,
        },
        {
            "scenario": "excit_inhib_label_coarsening",
            "description": "Merge Excit_* and Inhib_* reference labels into broad excitatory and inhibitory classes.",
            "expression": expression,
            "signatures": _coarsen_signatures(signatures),
            "truth": _coarsen_truth(truth),
            "error_truth": _coarsen_truth(truth),
            "dropped_genes": [],
            "dropped_celltypes": [],
            "contamination_fraction": 0.0,
        },
        {
            "scenario": "out_of_reference_contamination",
            "description": (
                "Mix a synthetic non-reference expression profile into pseudo-spots and evaluate against an "
                "explicit out-of-reference composition column."
            ),
            "expression": contaminated_expression,
            "signatures": signatures,
            "truth": truth,
            "error_truth": contaminated_truth,
            "dropped_genes": [],
            "dropped_celltypes": [],
            "contamination_fraction": float(np.clip(contamination_fraction, 0.0, 0.8)),
        },
    ]
    return scenarios


def _write_report(run_dir: Path, summary: pd.DataFrame) -> None:
    baseline = summary.loc[summary["scenario"] == "baseline_full_reference"].iloc[0]
    worst = summary.sort_values("mean_true_error", ascending=False).iloc[0]
    best_auc = summary.sort_values("auroc_top20_error", ascending=False).iloc[0]
    lines = [
        "# Revision Reference Perturbation Stress Test",
        "",
        "## Purpose",
        "",
        "本运行在 known-composition pseudo-spots（已知组成伪空间点）上做 reference perturbation / mismatch stress test（参考扰动 / 错配压力测试），用于补强 Reviewer #1 和 Reviewer #2 对风险分数是否响应参考扰动的疑问。",
        "",
        "## Main Result Snapshot",
        "",
        f"- baseline full reference（完整参考）中，`risk_score（风险分数）` vs true error（真实误差）的 Spearman（秩相关）为 `{baseline['spearman_error']:.3f}`，top20 true-error AUROC（最高 20% 真实误差 AUROC）为 `{baseline['auroc_top20_error']:.3f}`。",
        f"- mean true error（平均真实误差）最高的扰动为 `{worst['scenario']}`，平均误差 `{worst['mean_true_error']:.3f}`。",
        f"- risk-error AUROC 最高的扰动为 `{best_auc['scenario']}`，AUROC `{best_auc['auroc_top20_error']:.3f}`。",
        "",
        "## Interpretation Boundary",
        "",
        "- 该压力测试是 controlled reference perturbation（受控参考扰动）和 sanity check（合理性检查），不能写成自然组织 spot-level ground truth（空间点级真实标签）。",
        "- label coarsening（标签合并）使用 coarsened truth（合并后的真实组成）评估，因此只回答粒度变化下的 broad-level error（粗粒度误差），不与 full cell-type error（全细胞类型误差）直接等价。",
        "",
        "## Output Tables",
        "",
        "- `tables/reference_perturbation_spot_metrics.csv`",
        "- `tables/reference_perturbation_summary.csv`",
        "- `tables/reference_perturbation_marker_table.csv`",
        "- `tables/reference_perturbation_scenario_definitions.csv`",
    ]
    (run_dir / "revision_reference_perturbation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    known_run_dir = current_results_dir(args.known_composition_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    expression = pd.read_csv(known_run_dir / "tables" / "known_composition_expression_log_cp10k.csv", index_col=0)
    signatures = pd.read_csv(known_run_dir / "tables" / "reference_signatures_means.csv", index_col=0)
    truth = pd.read_csv(known_run_dir / "tables" / "known_composition_true_abundance.csv", index_col=0)
    spot_table = pd.read_csv(known_run_dir / "tables" / "known_composition_spot_table.csv", index_col=0)
    marker_table = pd.read_csv(known_run_dir / "tables" / "reference_signature_markers.csv")

    expression = expression.loc[:, signatures.index.astype(str).tolist()]
    truth = truth.loc[:, signatures.columns.astype(str).tolist()]

    scenario_rows = []
    spot_metric_frames = []
    marker_frames = []
    for scenario in _build_scenarios(
        expression=expression,
        signatures=signatures,
        truth=truth,
        marker_table=marker_table,
        contamination_fraction=args.contamination_fraction,
    ):
        scenario_name = scenario["scenario"]
        scenario_expression = scenario["expression"]
        scenario_signatures = scenario["signatures"]
        scenario_truth = scenario["truth"]
        error_truth = scenario["error_truth"]
        predicted_values = _project_abundance(
            scenario_expression.to_numpy(dtype=np.float32),
            scenario_signatures,
            method=args.projection_method,
            ridge_lambda=args.ridge_lambda,
        )
        predicted = pd.DataFrame(predicted_values, index=scenario_expression.index, columns=scenario_signatures.columns)
        metrics, perturbed_markers = _compute_risk_table(
            expression=scenario_expression,
            predicted=predicted,
            signatures=scenario_signatures,
            truth=error_truth,
            spot_table=spot_table,
            marker_top_k=args.marker_top_k,
            min_positive_markers=args.min_positive_markers,
            reference_repeats=args.reference_repeats,
            reference_fraction=args.reference_fraction,
            random_state=args.random_state,
        )
        metrics.insert(0, "scenario", scenario_name)
        metrics.insert(1, "evaluation_truth_space", "coarsened" if scenario_name.endswith("coarsening") else "full_celltype")
        spot_metric_frames.append(metrics)
        marker_frames.append(perturbed_markers.assign(scenario=scenario_name))
        scenario_rows.append(
            {
                "scenario": scenario_name,
                "description": scenario["description"],
                "n_genes": int(scenario_expression.shape[1]),
                "n_reference_celltypes": int(scenario_signatures.shape[1]),
                "n_error_celltypes": int(error_truth.shape[1]),
                "n_dropped_genes": int(len(scenario["dropped_genes"])),
                "dropped_genes_preview": ";".join(scenario["dropped_genes"][:20]),
                "dropped_celltypes": ";".join(scenario["dropped_celltypes"]),
                "contamination_fraction": float(scenario["contamination_fraction"]),
            }
        )

    spot_metrics = pd.concat(spot_metric_frames, ignore_index=False)
    marker_output = pd.concat(marker_frames, ignore_index=True) if marker_frames else pd.DataFrame()
    scenario_definitions = pd.DataFrame(scenario_rows)
    summary_rows = []
    for scenario_name, group in spot_metrics.groupby("scenario", sort=False):
        row = {"scenario": scenario_name, **_score_summary(group)}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).merge(scenario_definitions, on="scenario", how="left")

    spot_metrics.to_csv(results_file(run_dir, "tables", "reference_perturbation_spot_metrics.csv"))
    summary.to_csv(results_file(run_dir, "tables", "reference_perturbation_summary.csv"), index=False)
    marker_output.to_csv(results_file(run_dir, "tables", "reference_perturbation_marker_table.csv"), index=False)
    scenario_definitions.to_csv(results_file(run_dir, "tables", "reference_perturbation_scenario_definitions.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_run_dir),
        "projection_method": args.projection_method,
        "reference_repeats": int(args.reference_repeats),
        "reference_fraction": float(args.reference_fraction),
        "contamination_fraction": float(args.contamination_fraction),
        "random_state": int(args.random_state),
        "primary_error_col": "total_variation_error",
        "claim_boundary": (
            "Reference perturbation is a controlled stress test. It supports sensitivity to reference mismatch "
            "but should not be phrased as natural tissue ground truth."
        ),
    }
    results_file(run_dir, "metadata", "revision_reference_perturbation.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(run_dir, summary)

    print(f"Wrote revision reference perturbation stress test to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "known_composition_run_dir": str(known_run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
