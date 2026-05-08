from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from st_risk.config import load_config
from st_risk.data.io import open_h5ad
from st_risk.eval.reference_eval import (
    compute_reference_marker_scores,
    marker_subset_discordance_mean,
    reference_marker_discordance_proxy,
    reference_subsampling_instability,
    select_signature_markers,
    subset_markers,
)
from st_risk.eval.layer_eval import (
    boundary_enrichment_summary,
    boundary_spot_mask,
    celltype_risk_association_summary,
    cross_sample_layer_consistency_summary,
    dominant_layer_frequency_summary,
    high_risk_fraction_by_group,
    layer_celltype_risk_summary,
    neighbor_label_agreement,
    risk_stratified_layer_coherence_summary,
    sample_layer_risk_summary,
    sample_risk_summary,
    selective_celltype_shift_summary,
    selective_layer_retention_by_group,
    selective_layer_coherence_summary,
    selective_retention_summary,
)
from st_risk.models.io import load_saved_base_model_output
from st_risk.paths import current_results_dir, ensure_results_layout, project_root, resolve_results_file, results_file
from st_risk.risk.features import ambiguity_score, build_feature_table, confidence_proxy_score
from st_risk.risk.neighbors import inverse_distance_weights, knn_indices
from st_risk.risk.score import attach_risk_score
from st_risk.risk.stability import (
    cell2location_gene_subsample_stability,
    gene_subsample_stability,
    load_stability_predictions,
    row_normalize,
    save_stability_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ReliST risk features for a configured run.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "example_cell2location.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--input-results-dir",
        type=Path,
        default=None,
        help="Optional source run directory that provides saved base-model outputs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional output run_id under outputs.results_dir.",
    )
    parser.add_argument(
        "--risk-weights",
        type=str,
        default=None,
        help="Optional comma-separated weights for phi_local,phi_uncertainty,phi_stability. Example: 3,1,0",
    )
    return parser.parse_args()


def _compute_local_heterogeneity(counts, neighbors: np.ndarray, *, n_components: int = 12) -> np.ndarray:
    if sparse.issparse(counts):
        log_counts = counts.copy().astype(np.float32)
        log_counts.data = np.log1p(log_counts.data)
    else:
        log_counts = np.log1p(np.asarray(counts, dtype=np.float32))
    svd = TruncatedSVD(n_components=min(n_components, max(2, log_counts.shape[1] - 1)), random_state=0)
    embedding = svd.fit_transform(log_counts)
    heterogeneity = np.zeros(embedding.shape[0], dtype=float)
    for i, row in enumerate(neighbors):
        valid = row[row >= 0]
        if len(valid) == 0:
            heterogeneity[i] = 1.0
            continue
        local = embedding[valid]
        center = local.mean(axis=0, keepdims=True)
        heterogeneity[i] = float(np.mean(np.sum((local - center) ** 2, axis=1))) + 1e-6
    return heterogeneity


def _relative_uncertainty(abundance: pd.DataFrame, uncertainty: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(uncertainty, pd.Series):
        return uncertainty.reindex(abundance.index).astype(float).rename("phi_uncertainty")
    aligned_uncertainty = uncertainty.reindex(columns=abundance.columns)
    values = aligned_uncertainty.to_numpy() / (abundance.to_numpy() + 1e-6)
    return pd.Series(values.mean(axis=1), index=abundance.index, name="phi_uncertainty")


def _resolve_confidence_proxy_feature(
    abundance: pd.DataFrame,
    uncertainty: pd.DataFrame | pd.Series | None,
    *,
    missing_policy: str,
    uncertainty_weight: float,
) -> tuple[pd.Series, str]:
    ambiguity = pd.Series(ambiguity_score(abundance), index=abundance.index, name="phi_uncertainty")
    if uncertainty is not None:
        uncertainty_component = _relative_uncertainty(abundance, uncertainty)
        proxy = confidence_proxy_score(abundance, uncertainty_component, uncertainty_weight=uncertainty_weight)
        if np.allclose(uncertainty_component.to_numpy(dtype=float), 0.0):
            return ambiguity, "ambiguity_only_zero_uncertainty"
        return pd.Series(proxy, index=abundance.index, name="phi_uncertainty"), "uncertainty_plus_ambiguity"
    policy = missing_policy.strip().lower()
    if policy in {"zero", "ambiguity"}:
        return ambiguity, "ambiguity_only_missing_uncertainty"
    raise ValueError(
        "Base model outputs do not include uncertainty estimates and risk.missing_uncertainty_policy is not one of: zero, ambiguity."
    )


def _parse_risk_weights(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--risk-weights must contain exactly three comma-separated values.")
    local, uncertainty, stability = (float(part) for part in parts)
    return {
        "phi_local": local,
        "phi_uncertainty": uncertainty,
        "phi_stability": stability,
    }


def _risk_groups(sample_ids: pd.Series, normalization_scope: str) -> pd.Series | None:
    scope = normalization_scope.strip().lower()
    if scope == "global":
        return None
    if scope == "sample":
        return sample_ids.astype(str)
    raise ValueError(f"Unsupported risk.normalization_scope: {normalization_scope}")


def _resolve_dual_axis_weights(risk_cfg: dict) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    structure = risk_cfg.get("dual_axis_structure_weights")
    reference = risk_cfg.get("dual_axis_reference_weights")
    return structure, reference


def _build_reference_feature(
    visium,
    abundance: pd.DataFrame,
    *,
    layer: str,
    signatures_path: Path,
    feature_mode: str,
    marker_top_k: int,
    min_positive_markers: int,
    marker_subset_mode: str,
    subsampling_repeats: int,
    subsampling_fraction: float,
    subsampling_random_state: int,
) -> tuple[pd.Series | None, str]:
    if not signatures_path.exists():
        return None, "missing_reference_signatures"

    signatures = pd.read_csv(signatures_path, index_col=0)
    signatures.index = signatures.index.astype(str)
    markers, _ = select_signature_markers(
        signatures,
        top_k=marker_top_k,
        min_positive_markers=min_positive_markers,
    )
    markers = subset_markers(markers, mode=marker_subset_mode)
    marker_genes = sorted({gene for genes in markers.values() for gene in genes})
    if not marker_genes:
        return None, "empty_marker_subset"

    visium_var_names = pd.Index(visium.var_names.astype(str))
    visium_gene_lookup = {gene.lower(): gene for gene in visium_var_names}
    present_marker_genes = [gene for gene in marker_genes if gene.lower() in visium_gene_lookup]
    if not present_marker_genes:
        return None, "markers_not_in_visium"

    visium_genes = [visium_gene_lookup[gene.lower()] for gene in present_marker_genes]
    visium_view = visium[abundance.index, visium_genes]
    counts = visium_view.layers[layer][:]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = counts.astype(float, copy=False)

    library_size = counts.sum(axis=1, keepdims=True)
    safe_library = np.where(np.isclose(library_size, 0.0), 1.0, library_size)
    normalized = np.log1p((counts / safe_library) * 1e4)
    expression = pd.DataFrame(normalized, index=abundance.index, columns=present_marker_genes)
    marker_scores = compute_reference_marker_scores(expression, markers)
    if marker_scores.shape[1] < 2:
        return None, "insufficient_shared_marker_scores"
    normalized_mode = str(feature_mode).strip().lower()
    if normalized_mode == "marker_discordance":
        discordance = reference_marker_discordance_proxy(abundance, marker_scores)
        return discordance.rename("phi_reference"), f"marker_discordance_{marker_subset_mode}"
    if normalized_mode == "reference_subsampling_instability":
        instability = reference_subsampling_instability(
            abundance,
            expression,
            markers,
            repeats=subsampling_repeats,
            subset_fraction=subsampling_fraction,
            random_state=subsampling_random_state,
        )
        return instability, f"reference_subsampling_instability_{marker_subset_mode}"
    if normalized_mode == "marker_subset_discordance_mean":
        discordance_mean = marker_subset_discordance_mean(
            abundance,
            expression,
            markers,
            repeats=subsampling_repeats,
            subset_fraction=subsampling_fraction,
            random_state=subsampling_random_state,
        )
        return discordance_mean, f"marker_subset_discordance_mean_{marker_subset_mode}"
    raise ValueError(f"Unsupported reference feature mode: {feature_mode}")


def _copy_source_run_artifacts(source_dir: Path, output_dir: Path) -> None:
    relative_files = [
        ("tables", "base_model_abundance_means.csv"),
        ("tables", "base_model_abundance_stds.csv"),
        ("tables", "base_model_used_genes.csv"),
        ("tables", "cell2location_abundance_means.csv"),
        ("tables", "cell2location_abundance_stds.csv"),
        ("tables", "cell2location_used_genes.csv"),
        ("tables", "reference_signatures_means.csv"),
        ("metadata", "base_model_metadata.json"),
        ("metadata", "cell2location_metadata.json"),
    ]
    for category, filename in relative_files:
        source = resolve_results_file(source_dir, category, filename)
        if not source.exists():
            continue
        target = results_file(output_dir, category, filename)
        if source.resolve() == target.resolve():
            continue
        shutil.copy2(source, target)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    risk_cfg = config["risk"]
    model_cfg = config["model"]
    outputs_cfg = config.get("outputs", {})
    output_run_id = args.run_id or outputs_cfg.get("run_id")
    outputs_dir = current_results_dir(
        outputs_cfg["results_dir"],
        run_id=output_run_id,
        create=True,
    )
    ensure_results_layout(outputs_dir)
    source_results_dir = current_results_dir(args.input_results_dir) if args.input_results_dir is not None else outputs_dir
    _copy_source_run_artifacts(source_results_dir, outputs_dir)
    risk_weights = _parse_risk_weights(args.risk_weights)
    normalization_scope = str(risk_cfg.get("normalization_scope", "global"))
    confidence_proxy_uncertainty_weight = float(risk_cfg.get("confidence_proxy_uncertainty_weight", 0.5))
    dual_axis_structure_weights, dual_axis_reference_weights = _resolve_dual_axis_weights(risk_cfg)
    reference_feature_marker_top_k = int(risk_cfg.get("reference_feature_marker_top_k", 25))
    reference_feature_min_positive_markers = int(risk_cfg.get("reference_feature_min_positive_markers", 10))
    reference_feature_marker_subset = str(risk_cfg.get("reference_feature_marker_subset", "odd"))
    reference_feature_mode = str(risk_cfg.get("reference_feature_mode", "marker_discordance"))
    reference_feature_subsampling_repeats = int(risk_cfg.get("reference_feature_subsampling_repeats", 8))
    reference_feature_subsampling_fraction = float(risk_cfg.get("reference_feature_subsampling_fraction", 0.5))
    reference_feature_subsampling_random_state = int(risk_cfg.get("reference_feature_subsampling_random_state", 0))

    model_output = load_saved_base_model_output(source_results_dir)

    visium = open_h5ad(dataset_cfg["visium_h5ad"], backed=None)
    used_genes_path = resolve_results_file(source_results_dir, "tables", "base_model_used_genes.csv")
    if not used_genes_path.exists():
        used_genes_path = resolve_results_file(source_results_dir, "tables", "cell2location_used_genes.csv")
    if used_genes_path.exists():
        used_genes = pd.read_csv(used_genes_path)["gene"].tolist()
        visium = visium[:, used_genes].copy()
    else:
        used_genes = list(map(str, model_output.metadata.get("used_genes", visium.var_names.astype(str).tolist())))
        visium = visium[:, used_genes].copy()

    coords = np.asarray(visium.obsm["spatial"], dtype=float)
    neighbors = knn_indices(coords, k=int(risk_cfg.get("neighbor_k", 8)))
    weights = inverse_distance_weights(coords, neighbors)

    abundance_prop = pd.DataFrame(
        row_normalize(model_output.abundance.to_numpy()),
        index=model_output.abundance.index,
        columns=model_output.abundance.columns,
    )
    confidence_proxy, confidence_proxy_source = _resolve_confidence_proxy_feature(
        model_output.abundance,
        model_output.uncertainty,
        missing_policy=str(risk_cfg.get("missing_uncertainty_policy", "zero")),
        uncertainty_weight=confidence_proxy_uncertainty_weight,
    )
    heterogeneity = _compute_local_heterogeneity(visium.layers["counts"], neighbors)
    sample_ids = visium.obs["sample_id"].astype(str)

    stability_predictions = None
    signatures_path = resolve_results_file(source_results_dir, "tables", "reference_signatures_means.csv")
    reference_feature, reference_feature_source = _build_reference_feature(
        visium,
        abundance_prop,
        layer=model_cfg.get("layer", "counts"),
        signatures_path=signatures_path,
        feature_mode=reference_feature_mode,
        marker_top_k=reference_feature_marker_top_k,
        min_positive_markers=reference_feature_min_positive_markers,
        marker_subset_mode=reference_feature_marker_subset,
        subsampling_repeats=reference_feature_subsampling_repeats,
        subsampling_fraction=reference_feature_subsampling_fraction,
        subsampling_random_state=reference_feature_subsampling_random_state,
    )
    if signatures_path.exists() and int(risk_cfg.get("stability_repeats", 0)) > 0:
        signatures = pd.read_csv(signatures_path, index_col=0).loc[used_genes, abundance_prop.columns]
        stability_mode = str(risk_cfg.get("stability_mode", "proxy"))
        cache_path = resolve_results_file(source_results_dir, "artifacts", f"stability_predictions_{stability_mode}.npz")
        if cache_path.exists():
            stability_predictions, _, _ = load_stability_predictions(cache_path)
        elif stability_mode == "cell2location":
            if importlib.util.find_spec("cell2location") is None:
                raise RuntimeError(
                    "stability_mode=cell2location 需要在安装了 cell2location 的环境里运行。"
                )
            stability_predictions = cell2location_gene_subsample_stability(
                visium,
                signatures,
                layer=model_cfg.get("layer", "counts"),
                batch_key=model_cfg.get("visium_batch_key", "sample_id"),
                accelerator=model_cfg.get("accelerator", "gpu"),
                device=model_cfg.get("device", 1),
                repeats=int(risk_cfg.get("stability_repeats", 3)),
                gene_fraction=float(risk_cfg.get("stability_gene_fraction", 0.8)),
                random_state=int(risk_cfg.get("stability_random_state", 0)),
                spatial_max_epochs=int(risk_cfg.get("stability_spatial_max_epochs", 60)),
                spatial_batch_size=int(risk_cfg.get("stability_spatial_batch_size", 1024)),
                posterior_batch_size=int(model_cfg.get("posterior_batch_size", 1024)),
                spatial_posterior_samples=int(risk_cfg.get("stability_posterior_samples", 50)),
                n_cells_per_location=float(model_cfg.get("n_cells_per_location", 8.0)),
                detection_alpha=float(model_cfg.get("detection_alpha", 20.0)),
                early_stopping=bool(model_cfg.get("early_stopping", False)),
            )
            save_stability_predictions(
                results_file(outputs_dir, "artifacts", f"stability_predictions_{stability_mode}.npz"),
                stability_predictions,
                spot_index=abundance_prop.index.to_list(),
                celltypes=abundance_prop.columns.to_list(),
            )
        else:
            counts = visium.layers["counts"]
            if sparse.issparse(counts):
                counts = counts.toarray()
            stability_predictions = gene_subsample_stability(
                np.asarray(counts, dtype=np.float32),
                signatures.to_numpy(dtype=np.float32),
                repeats=int(risk_cfg.get("stability_repeats", 3)),
                gene_fraction=float(risk_cfg.get("stability_gene_fraction", 0.8)),
                ridge_lambda=float(risk_cfg.get("stability_ridge_lambda", 1e-3)),
                random_state=int(risk_cfg.get("stability_random_state", 0)),
            )
            save_stability_predictions(
                results_file(outputs_dir, "artifacts", f"stability_predictions_{stability_mode}.npz"),
                stability_predictions,
                spot_index=abundance_prop.index.to_list(),
                celltypes=abundance_prop.columns.to_list(),
            )

    feature_input = model_output
    feature_input.abundance = abundance_prop
    feature_input.uncertainty = confidence_proxy
    features = build_feature_table(
        feature_input,
        neighbors=neighbors,
        weights=weights,
        heterogeneity=heterogeneity,
        stability_predictions=stability_predictions,
        confidence_proxy_precomputed=True,
    )
    risk_table = attach_risk_score(
        features,
        weights=risk_weights,
        groups=_risk_groups(sample_ids, normalization_scope),
    )
    risk_table["sample_id"] = sample_ids.values
    risk_table["x_spatial"] = coords[:, 0]
    risk_table["y_spatial"] = coords[:, 1]
    if reference_feature is not None:
        risk_table["phi_reference"] = reference_feature.reindex(risk_table.index).astype(float)
    else:
        risk_table["phi_reference"] = 0.0

    risk_table.to_csv(results_file(outputs_dir, "tables", "risk_table.csv"))
    selective_retention_summary(risk_table).to_csv(results_file(outputs_dir, "tables", "selective_retention_summary.csv"), index=False)
    sample_summary = sample_risk_summary(risk_table, quantile=float(risk_cfg.get("high_risk_quantile", 0.9)))
    sample_summary.to_csv(results_file(outputs_dir, "tables", "sample_risk_summary.csv"), index=False)
    celltype_summary = celltype_risk_association_summary(
        abundance_prop, risk_table, quantile=float(risk_cfg.get("high_risk_quantile", 0.9))
    )
    celltype_summary.to_csv(results_file(outputs_dir, "tables", "celltype_risk_association_summary.csv"), index=False)
    selective_celltype_shift = selective_celltype_shift_summary(abundance_prop, risk_table, keep_quantiles=(0.8, 0.9, 1.0))
    selective_celltype_shift.to_csv(results_file(outputs_dir, "tables", "selective_celltype_shift_summary.csv"), index=False)

    has_layer_labels = "layer_guess" in visium.obs.columns
    if has_layer_labels:
        layer_labels = visium.obs["layer_guess"].astype(str)
        risk_table["layer_guess"] = layer_labels.values
        risk_table["boundary_mask"] = boundary_spot_mask(layer_labels, neighbors)
        risk_table["neighbor_agreement"] = neighbor_label_agreement(layer_labels, neighbors)
        risk_table.to_csv(results_file(outputs_dir, "tables", "risk_table.csv"))
        high_risk_fraction_by_group(risk_table, "layer_guess", quantile=float(risk_cfg.get("high_risk_quantile", 0.9))).to_csv(
            results_file(outputs_dir, "tables", "high_risk_fraction_by_layer.csv"), header=True
        )
        coherence_summary = selective_layer_coherence_summary(risk_table)
        coherence_summary.to_csv(results_file(outputs_dir, "tables", "selective_layer_coherence_summary.csv"), index=False)
        stratified_coherence = risk_stratified_layer_coherence_summary(
            risk_table,
            low_quantile=1.0 - float(risk_cfg.get("high_risk_quantile", 0.9)),
            high_quantile=float(risk_cfg.get("high_risk_quantile", 0.9)),
        )
        stratified_coherence.to_csv(results_file(outputs_dir, "tables", "risk_stratified_layer_coherence_summary.csv"), index=False)
        boundary_summary = boundary_enrichment_summary(
            risk_table, quantile=float(risk_cfg.get("high_risk_quantile", 0.9))
        )
        boundary_summary.to_csv(results_file(outputs_dir, "tables", "boundary_enrichment_summary.csv"), index=False)
        sample_layer_summary = sample_layer_risk_summary(risk_table, quantile=float(risk_cfg.get("high_risk_quantile", 0.9)))
        sample_layer_summary.to_csv(results_file(outputs_dir, "tables", "sample_layer_risk_summary.csv"), index=False)
        cross_sample_layer_summary = cross_sample_layer_consistency_summary(sample_layer_summary)
        cross_sample_layer_summary.to_csv(results_file(outputs_dir, "tables", "cross_sample_layer_consistency_summary.csv"), index=False)
        dominant_layer_summary = dominant_layer_frequency_summary(sample_layer_summary, top_k=2)
        dominant_layer_summary.to_csv(results_file(outputs_dir, "tables", "dominant_layer_frequency_summary.csv"), index=False)
        layer_celltype_summary = layer_celltype_risk_summary(
            abundance_prop, risk_table, quantile=float(risk_cfg.get("high_risk_quantile", 0.9))
        )
        layer_celltype_summary.to_csv(results_file(outputs_dir, "tables", "layer_celltype_risk_summary.csv"), index=False)
        selective_layer_retention = selective_layer_retention_by_group(risk_table, keep_quantiles=(0.8, 0.9, 1.0))
        selective_layer_retention.to_csv(results_file(outputs_dir, "tables", "selective_layer_retention_by_layer.csv"), index=False)

    summary = {
        "n_spots": int(risk_table.shape[0]),
        "risk_score_mean": float(risk_table["risk_score"].mean()),
        "risk_score_std": float(risk_table["risk_score"].std()),
        "high_risk_quantile": float(risk_cfg.get("high_risk_quantile", 0.9)),
        "stability_enabled": stability_predictions is not None,
        "stability_mode": str(risk_cfg.get("stability_mode", "proxy")),
        "source_results_dir": str(source_results_dir),
        "risk_feature_weights": risk_weights or {name: 1.0 for name in ("phi_local", "phi_uncertainty", "phi_stability")},
        "risk_normalization_scope": normalization_scope,
        "confidence_proxy_source": confidence_proxy_source,
        "confidence_proxy_uncertainty_weight": confidence_proxy_uncertainty_weight,
        "reference_feature_source": reference_feature_source,
        "dual_axis_structure_weights": dual_axis_structure_weights,
        "dual_axis_reference_weights": dual_axis_reference_weights,
        "highest_mean_risk_sample_id": str(sample_summary.iloc[0]["sample_id"]),
        "highest_mean_risk_sample_score": float(sample_summary.iloc[0]["mean_risk_score"]),
        "lowest_mean_risk_sample_id": str(sample_summary.iloc[-1]["sample_id"]),
        "lowest_mean_risk_sample_score": float(sample_summary.iloc[-1]["mean_risk_score"]),
        "top_risk_associated_celltype": str(celltype_summary.iloc[0]["celltype"]),
        "top_risk_associated_celltype_delta": float(celltype_summary.iloc[0]["abundance_delta"]),
        "top_retained_celltype_at_0p8": str(
            selective_celltype_shift.loc[selective_celltype_shift["keep_quantile"] == 0.8].iloc[0]["celltype"]
        ),
        "has_layer_labels": bool(has_layer_labels),
    }
    if has_layer_labels:
        summary.update(
            {
                "boundary_high_risk_enrichment": float(
                    boundary_summary.loc[boundary_summary["region"] == "boundary", "high_risk_enrichment"].iloc[0]
                ),
                "overall_mean_neighbor_agreement": float(risk_table["neighbor_agreement"].mean()),
                "low_risk_quartile_mean_neighbor_agreement": float(
                    selective_layer_coherence_summary(risk_table, keep_quantiles=(0.25,))["mean_neighbor_agreement"].iloc[0]
                ),
                "high_vs_low_neighbor_agreement_gap": float(
                    stratified_coherence.loc[stratified_coherence["risk_group"] == "high_risk", "mean_neighbor_agreement"].iloc[0]
                    - stratified_coherence.loc[stratified_coherence["risk_group"] == "low_risk", "mean_neighbor_agreement"].iloc[0]
                ),
                "most_consistent_high_risk_layer": str(cross_sample_layer_summary.iloc[0]["layer_guess"]),
                "most_consistent_high_risk_layer_mean_score": float(cross_sample_layer_summary.iloc[0]["mean_risk_score_mean"]),
                "top1_dominant_layer": str(dominant_layer_summary.iloc[0]["layer_guess"]),
                "top1_dominant_layer_count": int(dominant_layer_summary.iloc[0]["top1_count"]),
                "top_retained_layer_at_0p8": str(
                    selective_layer_retention.loc[selective_layer_retention["keep_quantile"] == 0.8].iloc[0]["layer_guess"]
                ),
            }
        )
    results_file(outputs_dir, "metadata", "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved risk outputs to {outputs_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
