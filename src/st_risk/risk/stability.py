from __future__ import annotations

import numpy as np
import pandas as pd


def row_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) < eps, 1.0, row_sums)
    return values / row_sums


def sample_gene_indices(
    n_genes: int,
    *,
    repeats: int,
    gene_fraction: float,
    random_state: int = 0,
) -> list[np.ndarray]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0 < gene_fraction <= 1:
        raise ValueError("gene_fraction must be in (0, 1]")
    n_take = max(2, int(round(n_genes * gene_fraction)))
    rng = np.random.default_rng(random_state)
    return [np.sort(rng.choice(n_genes, size=n_take, replace=False)) for _ in range(repeats)]


def ridge_project_celltype_proportions(
    spot_gene_matrix: np.ndarray,
    signature_gene_matrix: np.ndarray,
    *,
    ridge_lambda: float = 1e-3,
) -> np.ndarray:
    """Project spot expression profiles onto signature profiles with nonnegative clipped ridge regression."""
    y = row_normalize(spot_gene_matrix)
    s = row_normalize(signature_gene_matrix.T).T
    gram = s.T @ s
    weights = y @ s @ np.linalg.inv(gram + ridge_lambda * np.eye(gram.shape[0]))
    weights = np.clip(weights, 0.0, None)
    return row_normalize(weights)


def gene_subsample_stability(
    spot_gene_matrix: np.ndarray,
    signature_gene_matrix: np.ndarray,
    *,
    repeats: int = 3,
    gene_fraction: float = 0.8,
    ridge_lambda: float = 1e-3,
    random_state: int = 0,
) -> np.ndarray:
    """Generate repeated projected compositions under random gene subsampling."""
    n_genes = spot_gene_matrix.shape[1]
    gene_subsets = sample_gene_indices(
        n_genes,
        repeats=repeats,
        gene_fraction=gene_fraction,
        random_state=random_state,
    )
    runs = []
    for gene_idx in gene_subsets:
        projected = ridge_project_celltype_proportions(
            spot_gene_matrix[:, gene_idx],
            signature_gene_matrix[gene_idx, :],
            ridge_lambda=ridge_lambda,
        )
        runs.append(projected)
    return np.stack(runs, axis=0)


def save_stability_predictions(
    path,
    predictions: np.ndarray,
    *,
    spot_index: list[str] | np.ndarray,
    celltypes: list[str] | np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        predictions=np.asarray(predictions, dtype=np.float32),
        spot_index=np.asarray(spot_index, dtype=object),
        celltypes=np.asarray(celltypes, dtype=object),
    )


def load_stability_predictions(path):
    with np.load(path, allow_pickle=True) as data:
        return data["predictions"], data["spot_index"], data["celltypes"]


def cell2location_gene_subsample_stability(
    visium_adata,
    signatures: pd.DataFrame,
    *,
    layer: str,
    batch_key: str,
    accelerator: str,
    device: int | str,
    repeats: int,
    gene_fraction: float,
    random_state: int,
    spatial_max_epochs: int,
    spatial_batch_size: int,
    posterior_batch_size: int,
    spatial_posterior_samples: int,
    n_cells_per_location: float,
    detection_alpha: float,
    early_stopping: bool,
) -> np.ndarray:
    from cell2location.models import Cell2location
    import torch

    torch.set_float32_matmul_precision("high")

    gene_subsets = sample_gene_indices(
        visium_adata.n_vars,
        repeats=repeats,
        gene_fraction=gene_fraction,
        random_state=random_state,
    )
    predictions: list[np.ndarray] = []
    for gene_idx in gene_subsets:
        gene_names = visium_adata.var_names[gene_idx]
        visium_sub = visium_adata[:, gene_names].copy()
        signature_sub = signatures.loc[gene_names, :]

        Cell2location.setup_anndata(visium_sub, layer=layer, batch_key=batch_key)
        model = Cell2location(
            visium_sub,
            cell_state_df=signature_sub.loc[visium_sub.var_names],
            N_cells_per_location=n_cells_per_location,
            detection_alpha=detection_alpha,
        )
        model.train(
            max_epochs=spatial_max_epochs,
            batch_size=spatial_batch_size,
            train_size=1,
            accelerator=accelerator,
            device=device,
            early_stopping=early_stopping,
        )
        visium_sub = model.export_posterior(
            visium_sub,
            sample_kwargs={
                "num_samples": spatial_posterior_samples,
                "batch_size": posterior_batch_size,
            },
            add_to_obsm=["means"],
        )
        abundance = visium_sub.obsm["means_cell_abundance_w_sf"]
        abundance_df = pd.DataFrame(
            np.asarray(abundance, dtype=np.float32),
            index=visium_sub.obs_names,
            columns=signatures.columns,
        )
        predictions.append(row_normalize(abundance_df.to_numpy()))
    return np.stack(predictions, axis=0)
