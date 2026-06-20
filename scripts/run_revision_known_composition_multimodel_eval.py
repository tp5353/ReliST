from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from st_risk.eval.reference_eval import (
    compute_reference_marker_scores,
    reference_marker_discordance_proxy,
    reference_signature_residual_proxy,
    reference_subsampling_instability,
    select_signature_markers,
    subset_markers,
)
from st_risk.models.base import BaseSpatialModelOutput
from st_risk.models.io import load_saved_base_model_output
from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file, set_selected_run
from st_risk.risk.features import ambiguity_score, build_feature_table, confidence_proxy_score, local_residual_score
from st_risk.risk.neighbors import inverse_distance_weights, knn_indices
from st_risk.risk.score import grouped_zscore, sigmoid, zscore


DEFAULT_RUN_ID = "2026-06-20-dlpfc-known-composition-multimodel-v2-rctd-tangram"
RELIST_SCORES = (
    "risk_score",
    "reference_risk_score",
    "local_uncertainty_risk_score",
    "leave_out_reference",
    "reference_only",
    "risk_score_shuffled_local",
)
BASELINE_SCORES = (
    "abundance_entropy_risk",
    "inverse_top1_margin",
    "inverse_max_abundance",
    "phi_uncertainty",
    "native_uncertainty_risk",
    "cross_model_disagreement",
)
COMPONENT_SCORES = (
    "phi_local",
    "phi_local_shuffled_coordinates",
    "phi_reference",
    "snrna_marker_discordance",
    "snrna_signature_residual",
)
KEEP_FRACTIONS = tuple(np.round(np.linspace(0.5, 1.0, 11), 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple base-model outputs on the donor-disjoint known-composition benchmark."
    )
    parser.add_argument(
        "--known-composition-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Known-composition benchmark result root or run directory.",
    )
    parser.add_argument(
        "--model-run",
        action="append",
        default=[],
        help="Model run specification as model_key:/path/to/results_root_or_run_dir. Can be repeated.",
    )
    parser.add_argument(
        "--include-known-projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the projection already stored in known_composition_predicted_abundance.csv.",
    )
    parser.add_argument(
        "--projection-name",
        default="nnls_projection",
        help="Model key for the stored known-composition projection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_multimodel_eval",
        help="Result root for the multi-model known-composition evaluation.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    parser.add_argument("--marker-top-k", type=int, default=25, help="Top markers per cell type.")
    parser.add_argument("--min-positive-markers", type=int, default=10, help="Minimum positive markers per cell type.")
    parser.add_argument("--marker-subset-mode", default="odd", choices=("all", "odd", "even", "top_half", "bottom_half"))
    parser.add_argument("--reference-repeats", type=int, default=8, help="Reference marker subsampling repeats.")
    parser.add_argument("--reference-fraction", type=float, default=0.5, help="Reference marker subsampling fraction.")
    parser.add_argument("--bootstrap-repeats", type=int, default=500, help="Bootstrap repeats for CI tables.")
    parser.add_argument("--random-state", type=int, default=20260620, help="Random seed.")
    return parser.parse_args()


def _parse_model_runs(specs: list[str]) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--model-run must be model_key:path, got {spec!r}")
        name, raw_path = spec.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model key in --model-run {spec!r}")
        runs.append((name, current_results_dir(Path(raw_path.strip()))))
    return runs


def _normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = values.sum(axis=1, keepdims=True)
    safe = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
    return pd.DataFrame(values / safe, index=frame.index.astype(str), columns=frame.columns.astype(str))


def _native_uncertainty_to_risk(abundance: pd.DataFrame, uncertainty: pd.DataFrame | pd.Series | None) -> pd.Series | None:
    if uncertainty is None:
        return None
    if isinstance(uncertainty, pd.Series):
        return uncertainty.reindex(abundance.index).astype(float).rename("native_uncertainty_risk")
    aligned = uncertainty.reindex(index=abundance.index, columns=abundance.columns)
    values = aligned.to_numpy(dtype=float) / (abundance.to_numpy(dtype=float) + 1e-6)
    return pd.Series(np.nanmean(values, axis=1), index=abundance.index, name="native_uncertainty_risk")


def _abundance_baselines(predicted: pd.DataFrame) -> pd.DataFrame:
    probs = _normalize_rows(predicted).to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1] if probs.shape[1] else np.zeros(probs.shape[0], dtype=float)
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(probs.shape[0], dtype=float)
    entropy = np.zeros(probs.shape[0], dtype=float)
    if probs.shape[1] > 1:
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / np.log(probs.shape[1])
    return pd.DataFrame(
        {
            "abundance_entropy_risk": entropy,
            "inverse_top1_margin": 1.0 - (top1 - top2),
            "inverse_max_abundance": 1.0 - top1,
        },
        index=predicted.index,
    )


def _error_table(predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    shared = [celltype for celltype in truth.columns.astype(str) if celltype in predicted.columns]
    if len(shared) < 2:
        raise ValueError("At least two shared cell types are required for true-error evaluation.")
    pred = _normalize_rows(predicted.loc[truth.index, shared]).to_numpy(dtype=float)
    true = _normalize_rows(truth.loc[:, shared]).to_numpy(dtype=float)
    absolute = np.abs(pred - true)
    numerator = (pred * true).sum(axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return pd.DataFrame(
        {
            "n_shared_celltypes": int(len(shared)),
            "l1_error": absolute.sum(axis=1),
            "total_variation_error": 0.5 * absolute.sum(axis=1),
            "rmse_error": np.sqrt(np.mean((pred - true) ** 2, axis=1)),
            "cosine_distance": 1.0 - np.clip(cosine, -1.0, 1.0),
            "dominant_mismatch": pd.Index(shared)[pred.argmax(axis=1)].to_numpy()
            != pd.Index(shared)[true.argmax(axis=1)].to_numpy(),
        },
        index=truth.index,
    )


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


def _combine_features(table: pd.DataFrame, weights: dict[str, float], *, groups: pd.Series | None = None) -> pd.Series:
    linear = np.zeros(table.shape[0], dtype=float)
    total_weight = 0.0
    for feature, weight in weights.items():
        if feature not in table.columns or np.isclose(float(weight), 0.0):
            continue
        linear += float(weight) * grouped_zscore(table[feature].to_numpy(dtype=float), groups=groups)
        total_weight += abs(float(weight))
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one non-zero available feature is required.")
    return pd.Series(sigmoid(linear / total_weight), index=table.index)


def _aurc(score: pd.Series, error: pd.Series) -> float:
    valid = pd.concat([score, error], axis=1).dropna()
    if valid.empty:
        return np.nan
    ordered = valid.sort_values(valid.columns[0], ascending=True)
    errors = ordered.iloc[:, 1].to_numpy(dtype=float)
    coverages = np.arange(1, errors.size + 1, dtype=float) / errors.size
    risks = np.cumsum(errors) / np.arange(1, errors.size + 1, dtype=float)
    if errors.size == 1:
        return float(risks[0])
    return float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))


def _score_summary(table: pd.DataFrame, *, score_cols: list[str], error_col: str = "total_variation_error") -> pd.DataFrame:
    error = table[error_col].astype(float)
    high20 = (error >= error.quantile(0.80)).astype(int)
    high10 = (error >= error.quantile(0.90)).astype(int)
    rows: list[dict[str, object]] = []
    for score_col in score_cols:
        if score_col not in table.columns:
            continue
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
                "score_family": _score_family(score_col),
                "n_spots": int(score.notna().sum()),
                "spearman_error": spearman,
                "spearman_pvalue": spearman_p,
                "pearson_error": pearson,
                "pearson_pvalue": pearson_p,
                "auroc_top20_error": auc20,
                "average_precision_top20_error": ap20,
                "auroc_top10_error": auc10,
                "average_precision_top10_error": ap10,
                "top_minus_bottom20_error": float(error.loc[high_mask].mean() - error.loc[low_mask].mean()),
                "keep80_mean_error": _keep_mean_error(score, error, 0.80),
                "aurc": _aurc(score, error),
            }
        )
    return pd.DataFrame(rows).sort_values(["auroc_top20_error", "spearman_error"], ascending=[False, False])


def _score_family(score_name: str) -> str:
    if score_name in RELIST_SCORES:
        return "ReliST score"
    if score_name in BASELINE_SCORES:
        return "uncertainty/confidence baseline"
    if score_name in COMPONENT_SCORES:
        return "component/proxy diagnostic"
    return "other"


def _keep_mean_error(score: pd.Series, error: pd.Series, keep_fraction: float) -> float:
    valid = pd.concat([score, error], axis=1).dropna().sort_values(score.name, ascending=True)
    if valid.empty:
        return np.nan
    n_keep = max(1, int(round(valid.shape[0] * keep_fraction)))
    return float(valid.iloc[:n_keep, 1].mean())


def _selective_curve(table: pd.DataFrame, *, score_cols: list[str], error_col: str = "total_variation_error") -> pd.DataFrame:
    error = table[error_col].astype(float)
    full_mean = float(error.mean())
    high_error_threshold = float(error.quantile(0.80))
    rows: list[dict[str, object]] = []
    for score_col in score_cols:
        if score_col not in table.columns:
            continue
        ordered = table.sort_values(score_col, ascending=True)
        for keep_fraction in KEEP_FRACTIONS:
            n_keep = max(1, int(round(ordered.shape[0] * keep_fraction)))
            kept = ordered.head(n_keep)
            rows.append(
                {
                    "policy": "score",
                    "score_name": score_col,
                    "keep_fraction": float(keep_fraction),
                    "n_kept": int(n_keep),
                    "mean_error": float(kept[error_col].mean()),
                    "error_reduction_vs_full": float(1.0 - kept[error_col].mean() / full_mean) if full_mean > 0 else np.nan,
                    "high_error_fraction": float((kept[error_col] >= high_error_threshold).mean()),
                    "full_mean_error": full_mean,
                }
            )

    for policy, ordered in (
        ("random_reference", table.copy()),
        ("oracle_error", table.sort_values(error_col, ascending=True)),
    ):
        for keep_fraction in KEEP_FRACTIONS:
            n_keep = max(1, int(round(ordered.shape[0] * keep_fraction)))
            if policy == "random_reference":
                mean_error = full_mean
                high_error_fraction = float((error >= high_error_threshold).mean())
            else:
                kept = ordered.head(n_keep)
                mean_error = float(kept[error_col].mean())
                high_error_fraction = float((kept[error_col] >= high_error_threshold).mean())
            rows.append(
                {
                    "policy": policy,
                    "score_name": policy,
                    "keep_fraction": float(keep_fraction),
                    "n_kept": int(n_keep),
                    "mean_error": mean_error,
                    "error_reduction_vs_full": float(1.0 - mean_error / full_mean) if full_mean > 0 else np.nan,
                    "high_error_fraction": high_error_fraction,
                    "full_mean_error": full_mean,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_metric_rows(
    table: pd.DataFrame,
    *,
    model_key: str,
    score_cols: list[str],
    baseline_cols: list[str],
    repeats: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repeats <= 0:
        return pd.DataFrame(), pd.DataFrame()
    n = table.shape[0]
    error_full = table["total_variation_error"].astype(float)
    high20_threshold = float(error_full.quantile(0.80))
    best_baseline = None
    best_auc = -np.inf
    for col in baseline_cols:
        if col not in table.columns:
            continue
        auc, _ = _safe_auc(table[col].astype(float), (error_full >= high20_threshold).astype(int))
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_baseline = col

    draws: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    for repeat_idx in range(repeats):
        sample_idx = rng.integers(0, n, size=n)
        sample = table.iloc[sample_idx].copy()
        error = sample["total_variation_error"].astype(float)
        labels = (error >= high20_threshold).astype(int)
        for score_col in score_cols:
            if score_col not in sample.columns:
                continue
            score = sample[score_col].astype(float)
            spearman, _ = _safe_corr(score, error, method="spearman")
            auc, _ = _safe_auc(score, labels)
            draws.append(
                {
                    "model_key": model_key,
                    "score_name": score_col,
                    "bootstrap_id": repeat_idx,
                    "spearman_error": spearman,
                    "auroc_top20_error": auc,
                    "keep80_mean_error": _keep_mean_error(score, error, 0.80),
                    "aurc": _aurc(score, error),
                }
            )
        if best_baseline and "risk_score" in sample.columns:
            risk_score = sample["risk_score"].astype(float)
            baseline_score = sample[best_baseline].astype(float)
            risk_auc, _ = _safe_auc(risk_score, labels)
            base_auc, _ = _safe_auc(baseline_score, labels)
            deltas.append(
                {
                    "model_key": model_key,
                    "bootstrap_id": repeat_idx,
                    "primary_score": "risk_score",
                    "baseline_score": best_baseline,
                    "delta_auroc_top20": risk_auc - base_auc,
                    "delta_keep80_mean_error": _keep_mean_error(risk_score, error, 0.80)
                    - _keep_mean_error(baseline_score, error, 0.80),
                    "delta_aurc": _aurc(risk_score, error) - _aurc(baseline_score, error),
                }
            )
    return pd.DataFrame(draws), pd.DataFrame(deltas)


def _ci_table(draws: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    if draws.empty:
        return draws
    value_cols = [col for col in draws.columns if col not in group_cols and col != "bootstrap_id"]
    rows: list[dict[str, object]] = []
    for keys, group in draws.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys, strict=True))
        for value_col in value_cols:
            values = group[value_col].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            rows.append(
                {
                    **base,
                    "metric": value_col,
                    "mean": float(np.mean(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "n_bootstrap": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _regression_rows(table: pd.DataFrame, *, model_key: str, baseline_cols: list[str]) -> pd.DataFrame:
    error = table["total_variation_error"].astype(float)
    high20 = (error >= error.quantile(0.80)).astype(int)
    best_baseline = None
    best_auc = -np.inf
    for col in baseline_cols:
        if col not in table.columns:
            continue
        auc, _ = _safe_auc(table[col].astype(float), high20)
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_baseline = col
    if best_baseline is None or "risk_score" not in table.columns:
        return pd.DataFrame()

    y = error.to_numpy(dtype=float)
    y_centered = y - y.mean()
    tss = float(np.sum(y_centered**2))

    def fit_r2(columns: list[str]) -> tuple[float, np.ndarray]:
        x = np.column_stack([np.ones(table.shape[0]), *[zscore(table[col].to_numpy(dtype=float)) for col in columns]])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ beta
        rss = float(np.sum((y - pred) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        return float(r2), beta

    baseline_r2, baseline_beta = fit_r2([best_baseline])
    combined_r2, combined_beta = fit_r2([best_baseline, "risk_score"])
    return pd.DataFrame(
        [
            {
                "model_key": model_key,
                "best_baseline_score": best_baseline,
                "best_baseline_auroc_top20": best_auc,
                "baseline_only_r2": baseline_r2,
                "baseline_plus_relist_r2": combined_r2,
                "incremental_r2_for_relist": combined_r2 - baseline_r2,
                "baseline_coef_in_baseline_only": float(baseline_beta[1]),
                "baseline_coef_in_combined": float(combined_beta[1]),
                "relist_coef_in_combined": float(combined_beta[2]),
            }
        ]
    )


def _load_known_projection(known_run_dir: Path) -> BaseSpatialModelOutput:
    abundance = pd.read_csv(known_run_dir / "tables" / "known_composition_predicted_abundance.csv", index_col=0)
    return BaseSpatialModelOutput(
        abundance=abundance,
        uncertainty=None,
        metadata={"model_name": "known_projection", "integration_mode": "stored_projection"},
    )


def _load_inputs(known_run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    spot_table = pd.read_csv(known_run_dir / "tables" / "known_composition_spot_table.csv", index_col=0)
    truth = pd.read_csv(known_run_dir / "tables" / "known_composition_true_abundance.csv", index_col=0)
    expression = pd.read_csv(known_run_dir / "tables" / "known_composition_expression_log_cp10k.csv", index_col=0)
    signatures = pd.read_csv(known_run_dir / "tables" / "reference_signatures_means.csv", index_col=0)
    inclusion = pd.read_csv(known_run_dir / "tables" / "known_composition_celltype_inclusion.csv")
    metadata = json.loads((known_run_dir / "metadata" / "known_composition_benchmark.json").read_text(encoding="utf-8"))
    return spot_table, truth, expression, signatures, inclusion, metadata


def _evaluate_model(
    *,
    model_key: str,
    output: BaseSpatialModelOutput,
    spot_table: pd.DataFrame,
    truth: pd.DataFrame,
    expression: pd.DataFrame,
    signatures: pd.DataFrame,
    marker_top_k: int,
    min_positive_markers: int,
    marker_subset_mode: str,
    reference_repeats: int,
    reference_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    abundance = output.abundance.copy()
    abundance.index = abundance.index.astype(str)
    abundance.columns = abundance.columns.astype(str)
    abundance = abundance.reindex(index=truth.index).fillna(0.0)
    shared = [celltype for celltype in truth.columns.astype(str) if celltype in abundance.columns]
    abundance = _normalize_rows(abundance.loc[:, shared])

    native_uncertainty = _native_uncertainty_to_risk(abundance, output.uncertainty)
    if native_uncertainty is None:
        phi_uncertainty = pd.Series(ambiguity_score(abundance), index=abundance.index, name="phi_uncertainty")
        uncertainty_source = "ambiguity_only_no_native_uncertainty"
    else:
        phi_uncertainty = pd.Series(
            confidence_proxy_score(abundance, native_uncertainty, uncertainty_weight=0.5),
            index=abundance.index,
            name="phi_uncertainty",
        )
        uncertainty_source = str(output.metadata.get("uncertainty_source", "native_uncertainty_plus_ambiguity"))

    coords = spot_table.loc[truth.index, ["x_spatial", "y_spatial"]].to_numpy(dtype=float)
    neighbors = knn_indices(coords, k=8)
    weights = inverse_distance_weights(coords, neighbors)
    shuffled_coords = coords.copy()
    rng = np.random.default_rng(random_state)
    rng.shuffle(shuffled_coords, axis=0)
    shuffled_neighbors = knn_indices(shuffled_coords, k=8)
    shuffled_weights = inverse_distance_weights(shuffled_coords, shuffled_neighbors)

    expression_values = expression.loc[truth.index].to_numpy(dtype=float)
    heterogeneity = np.var(expression_values, axis=1) + 1e-6
    model_output = BaseSpatialModelOutput(abundance=abundance, uncertainty=phi_uncertainty, metadata=output.metadata)
    features = build_feature_table(
        model_output,
        neighbors=neighbors,
        weights=weights,
        heterogeneity=heterogeneity,
        confidence_proxy_precomputed=True,
    )
    features["phi_local_shuffled_coordinates"] = local_residual_score(
        abundance.to_numpy(dtype=float),
        shuffled_neighbors,
        weights=shuffled_weights,
        heterogeneity=heterogeneity,
    )

    sig_shared = signatures.loc[:, [celltype for celltype in shared if celltype in signatures.columns]].copy()
    markers_all, marker_table = select_signature_markers(
        sig_shared,
        top_k=marker_top_k,
        min_positive_markers=min_positive_markers,
    )
    markers = subset_markers(markers_all, mode=marker_subset_mode)
    marker_scores = compute_reference_marker_scores(expression, markers)
    features["phi_reference"] = reference_subsampling_instability(
        abundance,
        expression,
        markers,
        repeats=reference_repeats,
        subset_fraction=reference_fraction,
        random_state=random_state,
    )
    features["snrna_marker_discordance"] = reference_marker_discordance_proxy(abundance, marker_scores)
    features["snrna_signature_residual"] = reference_signature_residual_proxy(
        abundance,
        expression,
        sig_shared,
        genes=marker_table["gene"].astype(str).tolist(),
    )

    groups = spot_table.loc[truth.index, "sample_id"].astype(str)
    table = pd.concat([spot_table.loc[truth.index].copy(), features, _error_table(abundance, truth)], axis=1)
    table["native_uncertainty_risk"] = native_uncertainty.reindex(table.index) if native_uncertainty is not None else np.nan
    for column, values in _abundance_baselines(abundance).items():
        table[column] = values
    table["risk_score"] = _combine_features(table, {"phi_local": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0}, groups=groups)
    table["reference_risk_score"] = _combine_features(table, {"phi_uncertainty": 2.0, "phi_reference": 2.0}, groups=groups)
    table["local_uncertainty_risk_score"] = _combine_features(table, {"phi_local": 1.0, "phi_uncertainty": 1.0}, groups=groups)
    table["leave_out_reference"] = _combine_features(table, {"phi_local": 1.0, "phi_uncertainty": 1.0}, groups=groups)
    table["reference_only"] = _combine_features(table, {"phi_reference": 1.0}, groups=groups)
    table["risk_score_shuffled_local"] = _combine_features(
        table,
        {"phi_local_shuffled_coordinates": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0},
        groups=groups,
    )
    table.insert(0, "model_key", model_key)

    model_meta = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "model_name": output.metadata.get("model_name", model_key),
                "integration_mode": output.metadata.get("integration_mode", "unknown"),
                "uncertainty_source": uncertainty_source,
                "has_native_uncertainty": bool(native_uncertainty is not None),
                "n_spots": int(table.shape[0]),
                "n_shared_celltypes": int(len(shared)),
                "n_marker_celltypes": int(len(markers)),
            }
        ]
    )
    return table, model_meta


def _attach_cross_model_disagreement(tables: dict[str, pd.DataFrame]) -> None:
    if len(tables) < 2:
        for table in tables.values():
            table["cross_model_disagreement"] = np.nan
        return
    common_celltypes: set[str] | None = None
    abundance_by_model: dict[str, pd.DataFrame] = {}
    for model_key, table in tables.items():
        abundance_cols = [col for col in table.columns if col.startswith("abundance__")]
        if not abundance_cols:
            continue
        abundance = table.set_index(table.index)[abundance_cols].copy()
        abundance.columns = [col.replace("abundance__", "", 1) for col in abundance.columns]
        abundance_by_model[model_key] = abundance
        common_celltypes = set(abundance.columns) if common_celltypes is None else common_celltypes.intersection(abundance.columns)
    if not abundance_by_model or not common_celltypes:
        for table in tables.values():
            table["cross_model_disagreement"] = np.nan
        return

    ordered_celltypes = sorted(common_celltypes)
    stack = np.stack([_normalize_rows(abundance[ordered_celltypes]).to_numpy(dtype=float) for abundance in abundance_by_model.values()], axis=0)
    mean_abundance = stack.mean(axis=0)
    for idx, (model_key, table) in enumerate(tables.items()):
        if model_key not in abundance_by_model:
            table["cross_model_disagreement"] = np.nan
            continue
        model_values = _normalize_rows(abundance_by_model[model_key][ordered_celltypes]).to_numpy(dtype=float)
        table["cross_model_disagreement"] = 0.5 * np.abs(model_values - mean_abundance).sum(axis=1)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    known_run_dir = current_results_dir(args.known_composition_root)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    spot_table, truth, expression, signatures, inclusion, known_metadata = _load_inputs(known_run_dir)
    model_specs = _parse_model_runs(args.model_run)
    outputs: list[tuple[str, BaseSpatialModelOutput, str]] = []
    if args.include_known_projection:
        outputs.append((args.projection_name, _load_known_projection(known_run_dir), str(known_run_dir)))
    for model_key, run_path in model_specs:
        outputs.append((model_key, load_saved_base_model_output(run_path), str(run_path)))
    if not outputs:
        raise ValueError("No model outputs were provided.")

    model_tables: dict[str, pd.DataFrame] = {}
    metadata_rows: list[pd.DataFrame] = []
    for model_key, output, source_path in outputs:
        table, model_meta = _evaluate_model(
            model_key=model_key,
            output=output,
            spot_table=spot_table,
            truth=truth,
            expression=expression,
            signatures=signatures,
            marker_top_k=args.marker_top_k,
            min_positive_markers=args.min_positive_markers,
            marker_subset_mode=args.marker_subset_mode,
            reference_repeats=args.reference_repeats,
            reference_fraction=args.reference_fraction,
            random_state=args.random_state,
        )
        abundance = _normalize_rows(output.abundance.reindex(index=truth.index).fillna(0.0))
        for col in [celltype for celltype in truth.columns.astype(str) if celltype in abundance.columns]:
            table[f"abundance__{col}"] = abundance[col].to_numpy(dtype=float)
        table["source_run_dir"] = source_path
        model_tables[model_key] = table
        metadata_rows.append(model_meta.assign(source_run_dir=source_path))

    _attach_cross_model_disagreement(model_tables)

    score_cols = [
        *RELIST_SCORES,
        *BASELINE_SCORES,
        *COMPONENT_SCORES,
    ]
    baseline_cols = [col for col in BASELINE_SCORES]
    all_tables: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    selective_rows: list[pd.DataFrame] = []
    bootstrap_draws: list[pd.DataFrame] = []
    bootstrap_delta_draws: list[pd.DataFrame] = []
    regression_rows: list[pd.DataFrame] = []
    scenario_rows: list[pd.DataFrame] = []
    for model_key, table in model_tables.items():
        all_tables.append(table)
        model_score_cols = [col for col in score_cols if col in table.columns]
        summary_rows.append(_score_summary(table, score_cols=model_score_cols).assign(model_key=model_key))
        selective_rows.append(_selective_curve(table, score_cols=model_score_cols).assign(model_key=model_key))
        draws, deltas = _bootstrap_metric_rows(
            table,
            model_key=model_key,
            score_cols=model_score_cols,
            baseline_cols=[col for col in baseline_cols if col in table.columns],
            repeats=args.bootstrap_repeats,
            rng=rng,
        )
        bootstrap_draws.append(draws)
        bootstrap_delta_draws.append(deltas)
        regression_rows.append(_regression_rows(table, model_key=model_key, baseline_cols=[col for col in baseline_cols if col in table.columns]))
        scenario_rows.append(
            table.groupby("scenario", sort=True)
            .agg(
                n_spots=("scenario", "size"),
                mean_true_error=("total_variation_error", "mean"),
                full_risk_auroc_proxy=("risk_score", "mean"),
                leave_out_reference_mean=("leave_out_reference", "mean"),
                reference_only_mean=("reference_only", "mean"),
                phi_reference_mean=("phi_reference", "mean"),
            )
            .reset_index()
            .assign(model_key=model_key)
        )

    combined_table = pd.concat(all_tables, axis=0)
    summary = pd.concat(summary_rows, axis=0).loc[
        :,
        [
            "model_key",
            "score_name",
            "score_family",
            "n_spots",
            "spearman_error",
            "spearman_pvalue",
            "pearson_error",
            "pearson_pvalue",
            "auroc_top20_error",
            "average_precision_top20_error",
            "auroc_top10_error",
            "average_precision_top10_error",
            "top_minus_bottom20_error",
            "keep80_mean_error",
            "aurc",
        ],
    ]
    selective = pd.concat(selective_rows, axis=0)
    bootstrap = _ci_table(pd.concat(bootstrap_draws, axis=0), group_cols=["model_key", "score_name"])
    delta_ci = _ci_table(pd.concat(bootstrap_delta_draws, axis=0), group_cols=["model_key", "primary_score", "baseline_score"])
    regression = pd.concat(regression_rows, axis=0, ignore_index=True) if regression_rows else pd.DataFrame()
    scenario = pd.concat(scenario_rows, axis=0, ignore_index=True)
    model_metadata = pd.concat(metadata_rows, axis=0, ignore_index=True)

    combined_table.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_risk_error_table.csv"))
    summary.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_score_summary.csv"), index=False)
    selective.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_selective_curve.csv"), index=False)
    bootstrap.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_bootstrap_ci.csv"), index=False)
    delta_ci.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_delta_vs_best_baseline_ci.csv"), index=False)
    regression.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_independent_contribution.csv"), index=False)
    scenario.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_scenario_component_summary.csv"), index=False)
    model_metadata.to_csv(results_file(run_dir, "tables", "known_composition_multimodel_model_metadata.csv"), index=False)
    inclusion.to_csv(results_file(run_dir, "tables", "known_composition_celltype_inclusion.csv"), index=False)

    metadata = {
        "run_id": args.run_id,
        "known_composition_run_dir": str(known_run_dir),
        "known_composition_metadata": known_metadata,
        "model_runs": [{"model_key": key, "source": source} for key, _, source in outputs],
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "score_columns": score_cols,
        "baseline_columns": baseline_cols,
        "claim_boundary": (
            "This table evaluates ReliST and baseline scores against true pseudo-spot composition error. "
            "Natural-tissue conclusions still require anchored proxies and orthogonal validation."
        ),
    }
    results_file(run_dir, "metadata", "known_composition_multimodel_eval.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_by_model = (
        summary.loc[summary["score_name"].eq("risk_score"), ["model_key", "spearman_error", "auroc_top20_error", "aurc"]]
        .sort_values("model_key")
        .to_dict(orient="records")
    )
    report_lines = [
        "# Known-Composition Multi-Model Evaluation",
        "",
        "## Purpose",
        "",
        "This run evaluates the donor-disjoint known-composition pseudo-spots across stored base-model outputs and shared ReliST/baseline scores.",
        "",
        "## Primary Risk-Score Snapshot",
        "",
    ]
    for row in best_by_model:
        report_lines.append(
            f"- `{row['model_key']}`: Spearman={row['spearman_error']:.3f}; "
            f"AUROC(top20 true error)={row['auroc_top20_error']:.3f}; AURC={row['aurc']:.3f}."
        )
    report_lines.extend(
        [
            "",
            "## Output Tables",
            "",
            "- `tables/known_composition_multimodel_risk_error_table.csv`",
            "- `tables/known_composition_multimodel_score_summary.csv`",
            "- `tables/known_composition_multimodel_selective_curve.csv`",
            "- `tables/known_composition_multimodel_bootstrap_ci.csv`",
            "- `tables/known_composition_multimodel_delta_vs_best_baseline_ci.csv`",
            "- `tables/known_composition_multimodel_independent_contribution.csv`",
            "- `tables/known_composition_multimodel_scenario_component_summary.csv`",
            "- `tables/known_composition_multimodel_model_metadata.csv`",
        ]
    )
    (run_dir / "known_composition_multimodel_eval.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote known-composition multi-model evaluation to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "models": [key for key, _, _ in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
