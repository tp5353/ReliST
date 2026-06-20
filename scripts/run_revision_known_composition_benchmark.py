from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
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
from st_risk.paths import ensure_results_layout, project_root, results_file, set_selected_run
from st_risk.risk.features import ambiguity_score, build_feature_table
from st_risk.risk.neighbors import inverse_distance_weights, knn_indices
from st_risk.risk.score import grouped_zscore, sigmoid
from st_risk.risk.stability import gene_subsample_stability, ridge_project_celltype_proportions, row_normalize


DEFAULT_RUN_ID = "2026-06-20-dlpfc-known-composition-v2-donor-disjoint"


def parse_args() -> argparse.Namespace:
    default_source_root = project_root() / "results" / "dlpfc_rctd"
    selected_run = (default_source_root / "selected_run.txt").read_text(encoding="utf-8").strip()
    default_source_run = default_source_root / "runs" / selected_run

    parser = argparse.ArgumentParser(
        description="Build a DLPFC pseudo-spot known-composition benchmark for revision analyses."
    )
    parser.add_argument(
        "--reference-h5ad",
        type=Path,
        default=project_root() / "data" / "Human DLPFC" / "ready" / "dlpfc_snrna_ref.h5ad",
        help="snRNA reference h5ad used to draw pseudo-spots.",
    )
    parser.add_argument(
        "--reference-label-column",
        default="cellType_k",
        help="Reference cell-type column used as the known composition label.",
    )
    parser.add_argument(
        "--split-column",
        default="BrNum",
        help="Reference obs column used for donor-disjoint train/simulation split.",
    )
    parser.add_argument(
        "--split-mode",
        default="donor_disjoint",
        choices=("donor_disjoint", "cell_random"),
        help="How to split reference cells into signature-building and held-out simulation pools.",
    )
    parser.add_argument(
        "--min-split-cells-per-type",
        type=int,
        default=20,
        help="Minimum cells per retained type on each side of a donor-disjoint split.",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=default_source_run,
        help="Existing DLPFC run providing the selected gene list and optional cell-type order.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root() / "results" / "revision_known_composition_benchmark",
        help="Result root for this revision benchmark.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run id under output-root/runs/.")
    parser.add_argument("--n-spots", type=int, default=1200, help="Number of pseudo-spots to generate.")
    parser.add_argument("--n-regions", type=int, default=6, help="Number of smooth pseudo-spatial regions.")
    parser.add_argument("--cells-per-spot-min", type=int, default=4, help="Minimum cells per pseudo-spot.")
    parser.add_argument("--cells-per-spot-max", type=int, default=10, help="Maximum cells per pseudo-spot.")
    parser.add_argument("--min-cells-per-type", type=int, default=80, help="Minimum reference cells per retained type.")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="Fraction of cells used to build signatures.")
    parser.add_argument("--layer", default="counts", help="Reference h5ad layer containing raw counts.")
    parser.add_argument("--marker-top-k", type=int, default=25, help="Top signature markers per cell type.")
    parser.add_argument("--min-positive-markers", type=int, default=10, help="Minimum positive markers per cell type.")
    parser.add_argument(
        "--marker-subset-mode",
        default="odd",
        choices=("all", "odd", "even", "top_half", "bottom_half"),
        help="Marker subset used for reference_subsampling_instability.",
    )
    parser.add_argument("--reference-repeats", type=int, default=8, help="Marker subsampling repeats.")
    parser.add_argument("--reference-fraction", type=float, default=0.5, help="Marker subsampling fraction.")
    parser.add_argument(
        "--projection-method",
        default="nnls",
        choices=("nnls", "ridge"),
        help="Nonnegative projection method used as the lightweight deconvolution baseline.",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=0,
        help="Optional gene-subsampling stability repeats. Default 0 keeps this revision benchmark reference-centered.",
    )
    parser.add_argument("--stability-gene-fraction", type=float, default=0.8, help="Gene fraction if stability is enabled.")
    parser.add_argument("--ridge-lambda", type=float, default=1e-3, help="Ridge penalty for signature projection.")
    parser.add_argument("--random-state", type=int, default=20260620, help="Random seed.")
    return parser.parse_args()


def _read_gene_list(source_run_dir: Path) -> list[str]:
    used_genes_path = source_run_dir / "tables" / "base_model_used_genes.csv"
    if not used_genes_path.exists():
        used_genes_path = source_run_dir / "tables" / "cell2location_used_genes.csv"
    if used_genes_path.exists():
        return pd.read_csv(used_genes_path)["gene"].astype(str).tolist()

    signatures_path = source_run_dir / "tables" / "reference_signatures_means.csv"
    if not signatures_path.exists():
        raise FileNotFoundError(
            f"Could not find base_model_used_genes.csv or reference_signatures_means.csv under {source_run_dir}"
        )
    return pd.read_csv(signatures_path, index_col=0).index.astype(str).tolist()


def _read_source_celltype_order(source_run_dir: Path) -> list[str] | None:
    signatures_path = source_run_dir / "tables" / "reference_signatures_means.csv"
    if not signatures_path.exists():
        return None
    signatures = pd.read_csv(signatures_path, index_col=0, nrows=1)
    return signatures.columns.astype(str).tolist()


def _normalize_log_cp10k(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    values = np.asarray(matrix, dtype=np.float32)
    library = values.sum(axis=1, keepdims=True)
    safe_library = np.where(np.isclose(library, 0.0), 1.0, library)
    return np.log1p((values / safe_library) * 1e4).astype(np.float32)


def _as_dense_vector(matrix: np.ndarray | sparse.spmatrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32).reshape(-1)


def _select_reference_matrix(
    reference_h5ad: Path,
    *,
    layer: str,
    label_col: str,
    split_col: str | None,
    requested_genes: list[str],
    source_celltype_order: list[str] | None,
    min_cells_per_type: int,
) -> tuple[pd.DataFrame, sparse.spmatrix | np.ndarray, list[str], list[str], pd.DataFrame]:
    adata = ad.read_h5ad(reference_h5ad, backed="r")
    if label_col not in adata.obs.columns:
        raise KeyError(f"{label_col} is not present in {reference_h5ad}")
    if split_col and split_col not in adata.obs.columns:
        raise KeyError(f"{split_col} is not present in {reference_h5ad}")
    if layer not in adata.layers:
        raise KeyError(f"{layer} is not present in layers of {reference_h5ad}")

    var_lookup = {str(gene).lower(): str(gene) for gene in adata.var_names.astype(str)}
    genes = [var_lookup[str(gene).lower()] for gene in requested_genes if str(gene).lower() in var_lookup]
    if len(genes) < 50:
        raise ValueError(f"Only {len(genes)} requested genes were found in {reference_h5ad}; need at least 50.")

    labels_all = adata.obs[label_col].astype(str)
    counts_by_type = labels_all.value_counts()
    source_order = pd.Index(source_celltype_order or [], dtype=str)
    if source_celltype_order is None:
        celltypes = counts_by_type.loc[counts_by_type >= min_cells_per_type].index.astype(str).tolist()
    else:
        celltypes = [
            celltype
            for celltype in source_celltype_order
            if celltype in counts_by_type.index and int(counts_by_type[celltype]) >= min_cells_per_type
        ]
    if len(celltypes) < 3:
        raise ValueError("At least three retained cell types are required for this benchmark.")

    inclusion_table = (
        counts_by_type.rename_axis("celltype")
        .reset_index(name="n_reference_cells")
        .assign(
            in_source_celltype_order=lambda frame: frame["celltype"].isin(source_order).astype(bool),
            passes_min_cells=lambda frame: frame["n_reference_cells"].ge(min_cells_per_type),
            retained_before_split=lambda frame: frame["celltype"].isin(celltypes).astype(bool),
        )
        .sort_values(["retained_before_split", "n_reference_cells"], ascending=[False, False])
        .reset_index(drop=True)
    )

    keep_mask = labels_all.isin(celltypes).to_numpy()
    obs_columns = [label_col]
    if split_col:
        obs_columns.append(split_col)
    obs = adata.obs.loc[keep_mask, obs_columns].copy()
    obs[label_col] = obs[label_col].astype(str)
    if split_col:
        obs[split_col] = obs[split_col].astype(str)
    matrix = adata[keep_mask, genes].layers[layer]
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix, dtype=np.float32)
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs, matrix, genes, celltypes, inclusion_table


def _split_reference_cells(
    obs: pd.DataFrame,
    *,
    label_col: str,
    celltypes: list[str],
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train: dict[str, np.ndarray] = {}
    simulate: dict[str, np.ndarray] = {}
    fraction = float(np.clip(train_fraction, 0.2, 0.8))
    labels = obs[label_col].astype(str).to_numpy()
    for celltype in celltypes:
        positions = np.flatnonzero(labels == celltype)
        shuffled = rng.permutation(positions)
        n_train = int(round(len(shuffled) * fraction))
        n_train = min(max(n_train, 20), len(shuffled) - 1)
        train[celltype] = np.sort(shuffled[:n_train])
        simulate[celltype] = np.sort(shuffled[n_train:])
        if simulate[celltype].size == 0:
            simulate[celltype] = train[celltype]
    return train, simulate


def _split_reference_cells_donor_disjoint(
    obs: pd.DataFrame,
    *,
    label_col: str,
    split_col: str,
    celltypes: list[str],
    train_fraction: float,
    min_cells_per_side: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, object]]:
    labels = obs[label_col].astype(str).to_numpy()
    units = obs[split_col].astype(str).to_numpy()
    unique_units = np.asarray(pd.Index(units).unique().astype(str), dtype=object)
    if unique_units.size < 2:
        raise ValueError(f"Need at least two unique {split_col} values for donor-disjoint split.")

    fraction = float(np.clip(train_fraction, 0.2, 0.8))
    n_train_units = int(round(unique_units.size * fraction))
    n_train_units = min(max(n_train_units, 1), unique_units.size - 1)

    best: tuple[list[str], set[str], set[str]] | None = None
    best_score = (-1, -1)
    for _ in range(500):
        shuffled = rng.permutation(unique_units)
        train_units = set(map(str, shuffled[:n_train_units]))
        simulate_units = set(map(str, shuffled[n_train_units:]))
        train_mask = np.asarray([unit in train_units for unit in units], dtype=bool)
        simulate_mask = np.asarray([unit in simulate_units for unit in units], dtype=bool)
        retained: list[str] = []
        total_cells = 0
        for celltype in celltypes:
            label_mask = labels == celltype
            n_train = int((label_mask & train_mask).sum())
            n_sim = int((label_mask & simulate_mask).sum())
            if n_train >= min_cells_per_side and n_sim >= min_cells_per_side:
                retained.append(celltype)
                total_cells += n_train + n_sim
        score = (len(retained), total_cells)
        if score > best_score:
            best = (retained, train_units, simulate_units)
            best_score = score
        if len(retained) == len(celltypes):
            break

    if best is None:
        raise RuntimeError("Could not construct a donor-disjoint split.")
    retained_celltypes, train_units, simulate_units = best
    if len(retained_celltypes) < 3:
        raise ValueError(
            "Fewer than three cell types passed the donor-disjoint split filters; "
            "try lowering --min-split-cells-per-type or using --split-mode cell_random."
        )

    train: dict[str, np.ndarray] = {}
    simulate: dict[str, np.ndarray] = {}
    train_mask = np.asarray([unit in train_units for unit in units], dtype=bool)
    simulate_mask = np.asarray([unit in simulate_units for unit in units], dtype=bool)
    for celltype in retained_celltypes:
        label_mask = labels == celltype
        train[celltype] = np.flatnonzero(label_mask & train_mask)
        simulate[celltype] = np.flatnonzero(label_mask & simulate_mask)

    metadata = {
        "split_mode": "donor_disjoint",
        "split_column": split_col,
        "train_units": sorted(train_units),
        "simulate_units": sorted(simulate_units),
        "n_train_units": int(len(train_units)),
        "n_simulate_units": int(len(simulate_units)),
        "min_split_cells_per_type": int(min_cells_per_side),
        "n_retained_celltypes_after_split": int(len(retained_celltypes)),
        "dropped_after_split": [celltype for celltype in celltypes if celltype not in retained_celltypes],
    }
    return train, simulate, retained_celltypes, metadata


def _augment_inclusion_table(
    inclusion_table: pd.DataFrame,
    *,
    celltypes_after_split: list[str],
    train_cells: dict[str, np.ndarray],
    simulate_cells: dict[str, np.ndarray],
) -> pd.DataFrame:
    table = inclusion_table.copy()
    train_counts = {celltype: int(len(indices)) for celltype, indices in train_cells.items()}
    simulate_counts = {celltype: int(len(indices)) for celltype, indices in simulate_cells.items()}
    table["n_train_signature_cells"] = table["celltype"].map(train_counts).fillna(0).astype(int)
    table["n_heldout_simulation_cells"] = table["celltype"].map(simulate_counts).fillna(0).astype(int)
    table["retained_after_split"] = table["celltype"].isin(celltypes_after_split).astype(bool)
    table["exclusion_reason"] = "retained"
    table.loc[~table["passes_min_cells"], "exclusion_reason"] = "below_min_cells_per_type"
    table.loc[
        table["retained_before_split"] & ~table["retained_after_split"],
        "exclusion_reason",
    ] = "insufficient_train_or_heldout_cells_after_split"
    table.loc[~table["retained_before_split"] & table["passes_min_cells"], "exclusion_reason"] = "not_in_source_celltype_order"
    return table


def _compute_signatures(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    train_cells: dict[str, np.ndarray],
    genes: list[str],
    celltypes: list[str],
) -> pd.DataFrame:
    columns = {}
    for celltype in celltypes:
        normalized = _normalize_log_cp10k(matrix[train_cells[celltype], :])
        columns[celltype] = normalized.mean(axis=0)
    signatures = pd.DataFrame(columns, index=genes, dtype=float)
    return signatures


def _build_region_prototypes(celltypes: list[str], *, n_regions: int, rng: np.random.Generator) -> pd.DataFrame:
    n_types = len(celltypes)
    rows = []
    for region_id in range(n_regions):
        alpha = np.full(n_types, 0.08, dtype=float)
        dominant_count = min(4, n_types)
        dominant = rng.choice(n_types, size=dominant_count, replace=False)
        alpha[dominant] = 3.0
        prototype = rng.dirichlet(alpha)
        rows.append(prototype)
    return pd.DataFrame(rows, columns=celltypes, index=[f"region_{i + 1}" for i in range(n_regions)])


def _smooth_region_composition(
    *,
    y: int,
    height: int,
    prototypes: pd.DataFrame,
    rng: np.random.Generator,
    concentration: float,
) -> tuple[str, np.ndarray]:
    n_regions = prototypes.shape[0]
    scaled = ((y + 0.5) / max(height, 1)) * n_regions
    lower = int(np.floor(scaled))
    lower = min(max(lower, 0), n_regions - 1)
    upper = min(lower + 1, n_regions - 1)
    mix = scaled - lower
    base = (1.0 - mix) * prototypes.iloc[lower].to_numpy(dtype=float) + mix * prototypes.iloc[upper].to_numpy(dtype=float)
    alpha = np.clip(base * concentration, 0.02, None)
    composition = rng.dirichlet(alpha)
    return str(prototypes.index[lower]), composition


def _sample_pseudo_spots(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    simulate_cells: dict[str, np.ndarray],
    celltypes: list[str],
    genes: list[str],
    markers: dict[str, list[str]],
    n_spots: int,
    n_regions: int,
    cells_per_spot_min: int,
    cells_per_spot_max: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    width = int(np.ceil(np.sqrt(n_spots)))
    height = int(np.ceil(n_spots / width))
    prototypes = _build_region_prototypes(celltypes, n_regions=n_regions, rng=rng)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    scenario_names = np.asarray(["clean", "low_depth", "marker_dropout", "diffuse_mixture"], dtype=object)
    scenario_probs = np.asarray([0.55, 0.15, 0.15, 0.15], dtype=float)

    spot_rows: list[dict[str, object]] = []
    true_rows: list[np.ndarray] = []
    pseudo_counts = np.zeros((n_spots, len(genes)), dtype=np.float32)
    for spot_id in range(n_spots):
        x = spot_id % width
        y = spot_id // width
        scenario = str(rng.choice(scenario_names, p=scenario_probs))
        concentration = 12.0 if scenario == "diffuse_mixture" else 80.0
        region_name, composition = _smooth_region_composition(
            y=y,
            height=height,
            prototypes=prototypes,
            rng=rng,
            concentration=concentration,
        )
        if scenario == "diffuse_mixture":
            composition = row_normalize((0.65 * composition + 0.35 / len(celltypes))[None, :])[0]

        n_cells = int(rng.integers(cells_per_spot_min, cells_per_spot_max + 1))
        type_counts = rng.multinomial(n_cells, composition)
        if type_counts.sum() == 0:
            type_counts[int(np.argmax(composition))] = n_cells
        true_fraction = type_counts / max(type_counts.sum(), 1)

        selected_cell_positions: list[int] = []
        for celltype, count in zip(celltypes, type_counts, strict=True):
            if count <= 0:
                continue
            pool = simulate_cells[celltype]
            selected = rng.choice(pool, size=int(count), replace=True)
            selected_cell_positions.extend(int(value) for value in selected.tolist())

        if selected_cell_positions:
            spot_counts = _as_dense_vector(matrix[selected_cell_positions, :].sum(axis=0))
        else:
            spot_counts = np.zeros(len(genes), dtype=np.float32)

        if scenario == "low_depth":
            keep_probability = float(rng.uniform(0.15, 0.45))
            spot_counts = rng.binomial(np.maximum(spot_counts, 0).astype(np.int64), keep_probability).astype(np.float32)
        elif scenario == "marker_dropout":
            dominant_type = celltypes[int(np.argmax(true_fraction))]
            marker_genes = [gene for gene in markers.get(dominant_type, []) if gene in gene_to_idx]
            if marker_genes:
                marker_idx = np.asarray([gene_to_idx[gene] for gene in marker_genes], dtype=int)
                spot_counts[marker_idx] = rng.binomial(
                    np.maximum(spot_counts[marker_idx], 0).astype(np.int64),
                    0.25,
                ).astype(np.float32)

        pseudo_counts[spot_id, :] = spot_counts
        spot_name = f"pseudo_spot_{spot_id + 1:05d}"
        spot_rows.append(
            {
                "spot_id": spot_name,
                "sample_id": "dlpfc_known_composition",
                "x_spatial": float(x),
                "y_spatial": float(y),
                "pseudo_region": region_name,
                "scenario": scenario,
                "n_cells": int(n_cells),
                "library_size": float(spot_counts.sum()),
            }
        )
        true_rows.append(true_fraction)

    spot_table = pd.DataFrame(spot_rows).set_index("spot_id")
    true_abundance = pd.DataFrame(true_rows, index=spot_table.index, columns=celltypes)
    return spot_table, true_abundance, prototypes, pseudo_counts


def _compute_local_heterogeneity(expression: np.ndarray, neighbors: np.ndarray, *, n_components: int = 12) -> np.ndarray:
    n_components = min(n_components, max(2, expression.shape[1] - 1), max(2, expression.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    embedding = svd.fit_transform(expression)
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


def _project_abundance(
    expression: np.ndarray,
    signatures: pd.DataFrame,
    *,
    method: str,
    ridge_lambda: float,
) -> np.ndarray:
    normalized_method = method.strip().lower()
    signature_values = signatures.to_numpy(dtype=np.float32)
    if normalized_method == "ridge":
        return ridge_project_celltype_proportions(expression, signature_values, ridge_lambda=ridge_lambda)
    if normalized_method != "nnls":
        raise ValueError(f"Unsupported projection method: {method}")

    projected = np.zeros((expression.shape[0], signature_values.shape[1]), dtype=np.float32)
    for idx, y in enumerate(expression):
        weights, _ = nnls(signature_values, np.asarray(y, dtype=np.float64), maxiter=1000)
        projected[idx, :] = weights.astype(np.float32)
    return row_normalize(projected)


def _combine_any_features(
    table: pd.DataFrame,
    weights: dict[str, float],
    *,
    groups: pd.Series | None = None,
) -> pd.Series:
    linear = np.zeros(table.shape[0], dtype=float)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in table.columns or np.isclose(float(weight), 0.0):
            continue
        linear += float(weight) * grouped_zscore(table[name].to_numpy(dtype=float), groups=groups)
        total_weight += abs(float(weight))
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one non-zero available feature is required.")
    return pd.Series(sigmoid(linear / total_weight), index=table.index)


def _abundance_baselines(predicted: pd.DataFrame) -> pd.DataFrame:
    values = predicted.to_numpy(dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    probs = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1] if probs.shape[1] else np.zeros(probs.shape[0], dtype=float)
    top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(probs.shape[0], dtype=float)
    if probs.shape[1] <= 1:
        entropy = np.zeros(probs.shape[0], dtype=float)
    else:
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
    pred = predicted.loc[truth.index, truth.columns].to_numpy(dtype=float)
    true = truth.to_numpy(dtype=float)
    absolute = np.abs(pred - true)
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    numerator = (pred * true).sum(axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return pd.DataFrame(
        {
            "l1_error": absolute.sum(axis=1),
            "total_variation_error": 0.5 * absolute.sum(axis=1),
            "rmse_error": rmse,
            "cosine_distance": 1.0 - np.clip(cosine, -1.0, 1.0),
            "dominant_mismatch": predicted.idxmax(axis=1).ne(truth.idxmax(axis=1)).astype(int).to_numpy(),
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


def _score_error_summary(table: pd.DataFrame, *, score_cols: list[str], error_col: str) -> pd.DataFrame:
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
                "error_col": error_col,
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


def _selective_error_curve(
    table: pd.DataFrame,
    *,
    score_cols: list[str],
    error_col: str,
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
                    "abstain_fraction": float(1.0 - keep_fraction),
                    "n_kept": int(n_keep),
                    "mean_error": mean_error,
                    "median_error": float(kept[error_col].median()),
                    "error_reduction_vs_full": float(1.0 - (mean_error / full_mean)) if full_mean > 0 else np.nan,
                    "high_error_fraction": float((kept[error_col] >= high_error_threshold).mean()),
                    "full_mean_error": full_mean,
                }
            )
    return pd.DataFrame(rows)


def _scenario_summary(table: pd.DataFrame, *, score_col: str = "risk_score") -> pd.DataFrame:
    grouped = table.groupby("scenario", sort=True)
    return (
        grouped.agg(
            n_spots=("scenario", "size"),
            mean_true_error=("total_variation_error", "mean"),
            median_true_error=("total_variation_error", "median"),
            mean_risk_score=(score_col, "mean"),
            mean_phi_local=("phi_local", "mean"),
            mean_phi_uncertainty=("phi_uncertainty", "mean"),
            mean_phi_reference=("phi_reference", "mean"),
        )
        .reset_index()
        .sort_values("mean_true_error", ascending=False)
    )


def _write_benchmark_h5ad_inputs(
    run_dir: Path,
    *,
    obs: pd.DataFrame,
    matrix: sparse.spmatrix | np.ndarray,
    genes: list[str],
    train_cells: dict[str, np.ndarray],
    celltypes: list[str],
    spot_table: pd.DataFrame,
    pseudo_counts: np.ndarray,
    label_col: str,
    layer: str,
) -> tuple[Path, Path]:
    train_indices = np.sort(np.concatenate([train_cells[celltype] for celltype in celltypes]))
    reference_obs = obs.iloc[train_indices].copy()
    reference_var = pd.DataFrame(index=pd.Index(genes, name=None).astype(str))
    reference_counts = matrix[train_indices, :]
    reference = ad.AnnData(X=reference_counts.copy(), obs=reference_obs, var=reference_var)
    reference.layers[layer] = reference_counts.copy()

    pseudo_obs = spot_table.copy()
    pseudo_var = pd.DataFrame(index=pd.Index(genes, name=None).astype(str))
    pseudo = ad.AnnData(X=pseudo_counts.copy(), obs=pseudo_obs, var=pseudo_var)
    pseudo.layers[layer] = pseudo_counts.copy()
    pseudo.obsm["spatial"] = pseudo_obs[["x_spatial", "y_spatial"]].to_numpy(dtype=float)
    pseudo.obs["sample_id"] = pseudo.obs["sample_id"].astype(str)

    reference_path = results_file(run_dir, "artifacts", "known_composition_train_reference.h5ad")
    pseudo_path = results_file(run_dir, "artifacts", "known_composition_pseudo_visium.h5ad")
    reference.write_h5ad(reference_path)
    pseudo.write_h5ad(pseudo_path)
    return reference_path, pseudo_path


def _write_report(
    run_dir: Path,
    *,
    summary: pd.DataFrame,
    selective: pd.DataFrame,
    scenario: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    best = summary.iloc[0]
    risk_row = summary.loc[summary["score_name"] == "risk_score"]
    risk_text = "not available"
    if not risk_row.empty:
        row = risk_row.iloc[0]
        risk_text = (
            f"Spearman={row['spearman_error']:.3f}, "
            f"AUROC(top20 error)={row['auroc_top20_error']:.3f}, "
            f"top-bottom20 error gap={row['top_minus_bottom20_error']:.3f}"
        )

    keep80 = selective.loc[(selective["score_name"] == "risk_score") & (selective["keep_fraction"] == 0.8)]
    keep80_text = "not available"
    if not keep80.empty:
        row = keep80.iloc[0]
        keep80_text = (
            f"mean error={row['mean_error']:.3f}, "
            f"error reduction={row['error_reduction_vs_full']:.3f}, "
            f"high-error fraction={row['high_error_fraction']:.3f}"
        )

    lines = [
        "# Revision Known-Composition Benchmark",
        "",
        "## Purpose",
        "",
        "本运行生成 DLPFC pseudo-spots（伪空间点），保留 known cell-type composition（已知细胞类型组成），用于直接评估 ReliST risk score（ReliST 风险分数）与 true deconvolution error（真实反卷积误差）的关系。",
        "",
        "## Main Result Snapshot",
        "",
        f"- pseudo-spots（伪空间点）数量：`{metadata['n_spots']}`",
        f"- retained cell types（保留细胞类型）：`{metadata['n_celltypes']}`",
        f"- split mode（拆分方式）：`{metadata['split']['split_mode']}`，split column（拆分列）：`{metadata['split'].get('split_column', 'none')}`。",
        f"- primary `risk_score（风险分数）`：{risk_text}",
        f"- 低风险 keep 80%（保留 80% 低风险点）后：{keep80_text}",
        f"- 当前最佳 score（分数）：`{best['score_name']}`，AUROC(top20 error)={best['auroc_top20_error']:.3f}",
        "",
        "## Caveats",
        "",
        "- 这是 pseudo-spatial known-composition benchmark（伪空间已知组成基准），可直接回答审稿人关于 true error（真实误差）的核心问题，但仍不是自然组织中的 spot-level ground truth（空间点级真实标签）。",
        "- 当前默认不启用 base-model perturbation stability（基础模型扰动稳定性）；`phi_stability`（稳定性特征）保留为零列，避免偏离当前 manuscript boundary（手稿边界）。",
        "- `risk_score`（风险分数）在本运行中定义为 `phi_local（局部特征）`、`phi_uncertainty（输出模糊性）` 和 `phi_reference（参考特征）` 的等权标准化组合；高分表示更不可靠。",
        "",
        "## Output Tables",
        "",
        "- `tables/known_composition_spot_table.csv`",
        "- `tables/known_composition_true_abundance.csv`",
        "- `tables/known_composition_predicted_abundance.csv`",
        "- `tables/known_composition_risk_error_table.csv`",
        "- `tables/known_composition_celltype_inclusion.csv`",
        "- `tables/known_composition_score_error_summary.csv`",
        "- `tables/known_composition_selective_error_curve.csv`",
        "- `tables/known_composition_scenario_summary.csv`",
        "- `tables/reference_signatures_means.csv`",
        "- `tables/reference_signature_markers.csv`",
        "- `artifacts/known_composition_train_reference.h5ad`",
        "- `artifacts/known_composition_pseudo_visium.h5ad`",
    ]
    (run_dir / "revision_known_composition_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    run_dir = args.output_root / "runs" / args.run_id
    ensure_results_layout(run_dir)
    set_selected_run(args.output_root, args.run_id)

    requested_genes = _read_gene_list(args.source_run_dir)
    source_celltype_order = _read_source_celltype_order(args.source_run_dir)
    obs, matrix, genes, celltypes, inclusion_table = _select_reference_matrix(
        args.reference_h5ad,
        layer=args.layer,
        label_col=args.reference_label_column,
        split_col=args.split_column if args.split_mode == "donor_disjoint" else None,
        requested_genes=requested_genes,
        source_celltype_order=source_celltype_order,
        min_cells_per_type=args.min_cells_per_type,
    )
    if args.split_mode == "donor_disjoint":
        train_cells, simulate_cells, celltypes, split_metadata = _split_reference_cells_donor_disjoint(
            obs,
            label_col=args.reference_label_column,
            split_col=args.split_column,
            celltypes=celltypes,
            train_fraction=args.train_fraction,
            min_cells_per_side=args.min_split_cells_per_type,
            rng=rng,
        )
    else:
        train_cells, simulate_cells = _split_reference_cells(
            obs,
            label_col=args.reference_label_column,
            celltypes=celltypes,
            train_fraction=args.train_fraction,
            rng=rng,
        )
        split_metadata = {
            "split_mode": "cell_random",
            "split_column": None,
            "train_fraction": float(args.train_fraction),
            "n_retained_celltypes_after_split": int(len(celltypes)),
            "dropped_after_split": [],
        }
    inclusion_table = _augment_inclusion_table(
        inclusion_table,
        celltypes_after_split=celltypes,
        train_cells=train_cells,
        simulate_cells=simulate_cells,
    )
    signatures = _compute_signatures(matrix, train_cells=train_cells, genes=genes, celltypes=celltypes)
    markers_all, marker_table = select_signature_markers(
        signatures,
        top_k=args.marker_top_k,
        min_positive_markers=args.min_positive_markers,
    )
    markers = subset_markers(markers_all, mode=args.marker_subset_mode)

    spot_table, true_abundance, region_prototypes, pseudo_counts = _sample_pseudo_spots(
        matrix,
        simulate_cells=simulate_cells,
        celltypes=celltypes,
        genes=genes,
        markers=markers_all,
        n_spots=args.n_spots,
        n_regions=args.n_regions,
        cells_per_spot_min=args.cells_per_spot_min,
        cells_per_spot_max=args.cells_per_spot_max,
        rng=rng,
    )
    reference_h5ad, pseudo_visium_h5ad = _write_benchmark_h5ad_inputs(
        run_dir,
        obs=obs,
        matrix=matrix,
        genes=genes,
        train_cells=train_cells,
        celltypes=celltypes,
        spot_table=spot_table,
        pseudo_counts=pseudo_counts,
        label_col=args.reference_label_column,
        layer=args.layer,
    )
    expression = _normalize_log_cp10k(pseudo_counts)
    expression_df = pd.DataFrame(expression, index=spot_table.index, columns=genes)

    predicted_values = _project_abundance(
        expression,
        signatures,
        method=args.projection_method,
        ridge_lambda=args.ridge_lambda,
    )
    predicted_abundance = pd.DataFrame(predicted_values, index=spot_table.index, columns=celltypes)
    error = _error_table(predicted_abundance, true_abundance)

    coords = spot_table[["x_spatial", "y_spatial"]].to_numpy(dtype=float)
    neighbors = knn_indices(coords, k=8)
    weights = inverse_distance_weights(coords, neighbors)
    heterogeneity = _compute_local_heterogeneity(expression, neighbors)
    stability_predictions = None
    if args.stability_repeats > 0:
        stability_predictions = gene_subsample_stability(
            expression,
            signatures.to_numpy(dtype=np.float32),
            repeats=args.stability_repeats,
            gene_fraction=args.stability_gene_fraction,
            ridge_lambda=args.ridge_lambda,
            random_state=args.random_state,
        )

    ambiguity = pd.Series(ambiguity_score(predicted_abundance), index=predicted_abundance.index, name="phi_uncertainty")
    model_output = BaseSpatialModelOutput(abundance=predicted_abundance, uncertainty=ambiguity)
    features = build_feature_table(
        model_output,
        neighbors=neighbors,
        weights=weights,
        heterogeneity=heterogeneity,
        stability_predictions=stability_predictions,
        confidence_proxy_precomputed=True,
    )

    marker_scores = compute_reference_marker_scores(expression_df, markers)
    features["phi_reference"] = reference_subsampling_instability(
        predicted_abundance,
        expression_df,
        markers,
        repeats=args.reference_repeats,
        subset_fraction=args.reference_fraction,
        random_state=args.random_state,
    )
    reference_marker = reference_marker_discordance_proxy(predicted_abundance, marker_scores)
    reference_residual = reference_signature_residual_proxy(
        predicted_abundance,
        expression_df,
        signatures,
        genes=marker_table["gene"].astype(str).tolist(),
    )

    groups = spot_table["sample_id"].astype(str)
    risk_table = features.copy()
    risk_table["risk_score"] = _combine_any_features(
        risk_table,
        {"phi_local": 1.0, "phi_uncertainty": 1.0, "phi_reference": 1.0},
        groups=groups,
    )
    risk_table["reference_risk_score"] = _combine_any_features(
        risk_table,
        {"phi_uncertainty": 2.0, "phi_reference": 2.0},
        groups=groups,
    )
    risk_table["local_uncertainty_risk_score"] = _combine_any_features(
        risk_table,
        {"phi_local": 1.0, "phi_uncertainty": 1.0},
        groups=groups,
    )
    for column, values in _abundance_baselines(predicted_abundance).items():
        risk_table[column] = values
    risk_table["snrna_marker_discordance"] = reference_marker
    risk_table["snrna_signature_residual"] = reference_residual
    risk_table = pd.concat([spot_table, risk_table, error], axis=1)

    score_cols = [
        "risk_score",
        "reference_risk_score",
        "local_uncertainty_risk_score",
        "abundance_entropy_risk",
        "inverse_top1_margin",
        "inverse_max_abundance",
        "phi_local",
        "phi_uncertainty",
        "phi_reference",
        "snrna_marker_discordance",
        "snrna_signature_residual",
    ]
    if args.stability_repeats > 0:
        score_cols.append("phi_stability")

    score_summary = _score_error_summary(risk_table, score_cols=score_cols, error_col="total_variation_error")
    selective_curve = _selective_error_curve(risk_table, score_cols=score_cols, error_col="total_variation_error")
    scenario_summary = _scenario_summary(risk_table)

    spot_table.to_csv(results_file(run_dir, "tables", "known_composition_spot_table.csv"))
    true_abundance.to_csv(results_file(run_dir, "tables", "known_composition_true_abundance.csv"))
    predicted_abundance.to_csv(results_file(run_dir, "tables", "known_composition_predicted_abundance.csv"))
    risk_table.to_csv(results_file(run_dir, "tables", "known_composition_risk_error_table.csv"))
    inclusion_table.to_csv(results_file(run_dir, "tables", "known_composition_celltype_inclusion.csv"), index=False)
    score_summary.to_csv(results_file(run_dir, "tables", "known_composition_score_error_summary.csv"), index=False)
    selective_curve.to_csv(results_file(run_dir, "tables", "known_composition_selective_error_curve.csv"), index=False)
    scenario_summary.to_csv(results_file(run_dir, "tables", "known_composition_scenario_summary.csv"), index=False)
    region_prototypes.to_csv(results_file(run_dir, "tables", "known_composition_region_prototypes.csv"))
    signatures.to_csv(results_file(run_dir, "tables", "reference_signatures_means.csv"))
    marker_table.to_csv(results_file(run_dir, "tables", "reference_signature_markers.csv"), index=False)
    expression_df.to_csv(results_file(run_dir, "tables", "known_composition_expression_log_cp10k.csv"))

    metadata = {
        "run_id": args.run_id,
        "reference_h5ad": str(args.reference_h5ad),
        "reference_label_column": args.reference_label_column,
        "source_run_dir": str(args.source_run_dir),
        "n_spots": int(args.n_spots),
        "n_genes": int(len(genes)),
        "n_celltypes": int(len(celltypes)),
        "n_original_labels": int(inclusion_table.shape[0]),
        "n_retained_before_split": int(inclusion_table["retained_before_split"].sum()),
        "n_retained_after_split": int(inclusion_table["retained_after_split"].sum()),
        "celltypes": celltypes,
        "random_state": int(args.random_state),
        "train_fraction": float(args.train_fraction),
        "split": split_metadata,
        "train_reference_h5ad": str(reference_h5ad),
        "pseudo_visium_h5ad": str(pseudo_visium_h5ad),
        "cells_per_spot_min": int(args.cells_per_spot_min),
        "cells_per_spot_max": int(args.cells_per_spot_max),
        "pseudo_spot_scenarios": {
            "clean": "No extra degradation beyond pseudo-bulk sampling.",
            "low_depth": "Binomial thinning of counts with keep_probability sampled uniformly from 0.15 to 0.45.",
            "marker_dropout": "Dominant-type marker counts thinned to 25% keep probability.",
            "diffuse_mixture": "Composition smoothed toward a uniform mixture before cell sampling.",
        },
        "spatial_coordinate_generation": (
            "Pseudo-spots are placed on a regular grid. Smooth region prototypes vary along the y coordinate; "
            "therefore local-structure features should be interpreted with the shuffled-coordinate/null controls "
            "added in downstream revision scripts."
        ),
        "risk_score_definition": "equal-weight grouped-zscore combination of phi_local, phi_uncertainty, and phi_reference",
        "projection_method": args.projection_method,
        "stability_repeats": int(args.stability_repeats),
        "reference_feature_mode": "reference_subsampling_instability",
        "reference_marker_subset_mode": args.marker_subset_mode,
        "primary_error_col": "total_variation_error",
        "score_columns": score_cols,
        "manuscript_boundary": (
            "This benchmark is a known-composition validation of risk-error association. "
            "It does not make natural tissue spot-level truth claims."
        ),
    }
    results_file(run_dir, "metadata", "known_composition_benchmark.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(run_dir, summary=score_summary, selective=selective_curve, scenario=scenario_summary, metadata=metadata)

    print(f"Wrote known-composition benchmark to {run_dir}")
    print(json.dumps({"run_id": args.run_id, "n_spots": args.n_spots, "n_celltypes": len(celltypes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
