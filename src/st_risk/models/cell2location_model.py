from __future__ import annotations

import json
import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner
from st_risk.models.io import load_saved_base_model_output, save_base_model_output
from st_risk.paths import current_results_dir, ensure_results_layout, resolve_results_file, results_file

ABUNDANCE_PREFIX = "meanscell_abundance_w_sf_means_per_cluster_mu_fg_"
UNCERTAINTY_PREFIX = "stdscell_abundance_w_sf_means_per_cluster_mu_fg_"
SIGNATURE_PREFIX = "means_per_cluster_mu_fg_"


def _clean_prefixed_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [re.sub(rf"^{re.escape(prefix)}", "", str(col)) for col in renamed.columns]
    return renamed


def load_saved_cell2location_output(results_dir: str | Path) -> BaseSpatialModelOutput:
    results_path = current_results_dir(results_dir)
    canonical_abundance = resolve_results_file(results_path, "tables", "base_model_abundance_means.csv")
    if canonical_abundance.exists():
        output = load_saved_base_model_output(results_path)
        metadata = dict(output.metadata)
        metadata.setdefault("model_name", "cell2location")
        output.metadata = metadata
        return output
    abundance = pd.read_csv(resolve_results_file(results_path, "tables", "cell2location_abundance_means.csv"), index_col=0)
    abundance = _clean_prefixed_columns(abundance, ABUNDANCE_PREFIX)

    uncertainty_path = resolve_results_file(results_path, "tables", "cell2location_abundance_stds.csv")
    uncertainty = None
    if uncertainty_path.exists():
        uncertainty = pd.read_csv(uncertainty_path, index_col=0)
        uncertainty = _clean_prefixed_columns(uncertainty, UNCERTAINTY_PREFIX)

    metadata = {}
    metadata_path = resolve_results_file(results_path, "metadata", "cell2location_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return BaseSpatialModelOutput(abundance=abundance, uncertainty=uncertainty, metadata=metadata)


class Cell2LocationRunner(BaseSpatialModelRunner):
    model_name = "cell2location"

    def is_available(self) -> bool:
        return importlib.util.find_spec("cell2location") is not None

    def run(
        self,
        visium_path: str | Path,
        reference_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> BaseSpatialModelOutput:
        if not self.is_available():
            raise NotImplementedError(
                "cell2location is not installed in the current environment. "
                "Install the dependency before running the base model stage."
            )
        from cell2location.models import Cell2location, RegressionModel
        import torch

        torch.set_float32_matmul_precision("high")

        config = config or {}
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        preprocessing_cfg = config.get("preprocessing", {})
        outputs_cfg = config.get("outputs", {})
        output_dir = current_results_dir(
            outputs_cfg.get("results_dir", "results/cell2location"),
            run_id=outputs_cfg.get("run_id"),
            create=True,
        )
        ensure_results_layout(output_dir)

        visium = open_h5ad(visium_path, backed=None)
        reference = open_h5ad(reference_path, backed=None)

        shared_genes = intersect_gene_names(visium.var_names, reference.var_names)
        if preprocessing_cfg.get("use_hvg_only", True) and "is_top_hvg" in visium.var.columns:
            hvg = visium.var.index[visium.var["is_top_hvg"].fillna(False)]
            shared_genes = shared_genes.intersection(hvg)

        max_genes = preprocessing_cfg.get("max_genes")
        if max_genes is not None:
            shared_genes = shared_genes[: int(max_genes)]
        if len(shared_genes) == 0:
            raise ValueError("No shared genes remain after preprocessing filters.")

        celltype_col = dataset_cfg.get("reference_celltype_column") or choose_reference_celltype_column(
            reference.obs.columns
        )
        reference = reference[reference.obs[celltype_col].notna(), shared_genes].copy()
        visium = visium[:, shared_genes].copy()

        reference_batch_key = model_cfg.get("reference_batch_key", "Sample")
        visium_batch_key = model_cfg.get("visium_batch_key", "sample_id")
        accelerator = model_cfg.get("accelerator", "gpu")
        device = model_cfg.get("device", 1)
        posterior_batch_size = int(model_cfg.get("posterior_batch_size", 1024))

        reference, signatures = export_reference_signatures(
            reference,
            celltype_col=celltype_col,
            batch_key=reference_batch_key,
            layer=model_cfg.get("layer", "counts"),
            accelerator=accelerator,
            device=device,
            reference_max_epochs=int(model_cfg.get("reference_max_epochs", 50)),
            reference_batch_size=int(model_cfg.get("reference_batch_size", 2048)),
            reference_posterior_samples=int(model_cfg.get("reference_posterior_samples", 200)),
            posterior_batch_size=posterior_batch_size,
            early_stopping=bool(model_cfg.get("early_stopping", False)),
            signature_key=model_cfg.get("reference_signature_key", "means_per_cluster_mu_fg"),
        )
        signature_key = model_cfg.get("reference_signature_key", "means_per_cluster_mu_fg")

        Cell2location.setup_anndata(
            visium,
            layer=model_cfg.get("layer", "counts"),
            batch_key=visium_batch_key,
        )
        spatial_model = Cell2location(
            visium,
            cell_state_df=signatures.loc[visium.var_names],
            N_cells_per_location=float(model_cfg.get("n_cells_per_location", 8.0)),
            detection_alpha=float(model_cfg.get("detection_alpha", 20.0)),
        )
        spatial_model.train(
            max_epochs=int(model_cfg.get("spatial_max_epochs", 200)),
            batch_size=int(model_cfg.get("spatial_batch_size", 1024)),
            train_size=1,
            accelerator=accelerator,
            device=device,
            early_stopping=bool(model_cfg.get("early_stopping", False)),
        )
        visium = spatial_model.export_posterior(
            visium,
            sample_kwargs={
                "num_samples": int(model_cfg.get("spatial_posterior_samples", 200)),
                "batch_size": posterior_batch_size,
            },
        )

        abundance_key = model_cfg.get("abundance_key", "means_cell_abundance_w_sf")
        uncertainty_key = model_cfg.get("uncertainty_key", "stds_cell_abundance_w_sf")
        abundance = visium.obsm[abundance_key].copy()
        uncertainty = visium.obsm.get(uncertainty_key)
        if isinstance(uncertainty, pd.DataFrame):
            uncertainty_df = uncertainty.copy()
        elif uncertainty is None:
            uncertainty_df = None
        else:
            uncertainty_df = pd.DataFrame(uncertainty, index=visium.obs_names)

        metadata = {
            "model_name": self.model_name,
            "n_spots": int(visium.n_obs),
            "n_reference_cells": int(reference.n_obs),
            "n_genes": int(len(shared_genes)),
            "reference_celltype_column": celltype_col,
            "reference_signature_key": signature_key,
            "abundance_key": abundance_key,
            "uncertainty_key": uncertainty_key,
            "used_genes": list(map(str, shared_genes)),
        }

        abundance.to_csv(results_file(output_dir, "tables", "cell2location_abundance_means.csv"))
        if uncertainty_df is not None:
            uncertainty_df.to_csv(results_file(output_dir, "tables", "cell2location_abundance_stds.csv"))
        _clean_prefixed_columns(signatures, SIGNATURE_PREFIX).to_csv(results_file(output_dir, "tables", "reference_signatures_means.csv"))
        pd.Series(shared_genes, name="gene").to_csv(results_file(output_dir, "tables", "cell2location_used_genes.csv"), index=False)
        output = BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty_df,
            metadata=metadata,
        )
        save_base_model_output(output, output_dir)
        with results_file(output_dir, "metadata", "cell2location_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return output


def export_reference_signatures(
    reference,
    *,
    celltype_col: str,
    batch_key: str,
    layer: str,
    accelerator: str,
    device: int | str,
    reference_max_epochs: int,
    reference_batch_size: int,
    reference_posterior_samples: int,
    posterior_batch_size: int,
    early_stopping: bool,
    signature_key: str,
):
    from cell2location.models import RegressionModel

    RegressionModel.setup_anndata(
        reference,
        layer=layer,
        batch_key=batch_key,
        labels_key=celltype_col,
    )
    ref_model = RegressionModel(reference)
    ref_model.train(
        max_epochs=reference_max_epochs,
        batch_size=reference_batch_size,
        train_size=1,
        accelerator=accelerator,
        device=device,
        early_stopping=early_stopping,
    )
    reference = ref_model.export_posterior(
        reference,
        sample_kwargs={
            "num_samples": reference_posterior_samples,
            "batch_size": posterior_batch_size,
        },
    )
    if signature_key not in reference.varm:
        available = list(reference.varm.keys())
        raise KeyError(f"Reference signature key '{signature_key}' not found. Available: {available}")
    signatures = reference.varm[signature_key].copy()
    if not isinstance(signatures, pd.DataFrame):
        signatures = pd.DataFrame(signatures, index=reference.var_names)
    return reference, signatures
