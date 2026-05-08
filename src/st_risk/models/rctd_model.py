from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner
from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file


def _to_csc(matrix) -> sparse.csc_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsc()
    return sparse.csc_matrix(matrix)


def _write_lines(path: Path, values) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def _reference_signature_means(reference, *, celltype_col: str, layer_key: str) -> pd.DataFrame:
    matrix = reference.layers[layer_key] if layer_key in reference.layers else reference.X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    return (
        pd.DataFrame(matrix, index=reference.obs_names, columns=reference.var_names)
        .groupby(reference.obs[celltype_col].astype(str))
        .mean()
        .T
    )


def _align_rctd_outputs_to_visium(
    visium_index: pd.Index,
    abundance: pd.DataFrame,
    uncertainty: pd.Series,
    results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Index]:
    visium_index = pd.Index(visium_index.astype(str))
    abundance = abundance.copy()
    abundance.index = abundance.index.astype(str)
    uncertainty = uncertainty.copy()
    uncertainty.index = uncertainty.index.astype(str)
    results_df = results_df.copy()
    results_df.index = results_df.index.astype(str)

    returned_spots = visium_index.intersection(abundance.index)
    missing_spots = visium_index.difference(abundance.index)

    abundance = abundance.reindex(visium_index).fillna(0.0)
    uncertainty = uncertainty.reindex(visium_index).fillna(1.0)

    aligned_results_df = results_df.reindex(visium_index)
    aligned_results_df.insert(0, "returned_by_rctd", visium_index.isin(returned_spots))
    return abundance, uncertainty, aligned_results_df, missing_spots


class RCTDRunner(BaseSpatialModelRunner):
    model_name = "rctd"

    def is_available(self) -> bool:
        try:
            completed = subprocess.run(
                ["Rscript", "-e", "library(spacexr)"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return False
        return completed.returncode == 0

    def run(
        self,
        visium_path: str | Path,
        reference_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> BaseSpatialModelOutput:
        if not self.is_available():
            raise NotImplementedError(
                "spacexr is not available via Rscript. Install the R package before running the base model stage."
            )

        config = config or {}
        dataset_cfg = config.get("dataset", {})
        preprocessing_cfg = config.get("preprocessing", {})
        model_cfg = config.get("model", {})
        outputs_cfg = config.get("outputs", {})

        output_dir = current_results_dir(
            outputs_cfg.get("results_dir", "results/rctd"),
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
        layer_key = str(model_cfg.get("layer", "counts"))
        min_reference_cells = int(model_cfg.get("rctd_min_reference_cells_per_type", 25))
        max_reference_cells = model_cfg.get("rctd_max_reference_cells_per_type")
        if max_reference_cells is not None:
            max_reference_cells = int(max_reference_cells)
        reference_subsample_seed = int(model_cfg.get("rctd_reference_subsample_seed", 0))

        reference = reference[reference.obs[celltype_col].notna(), shared_genes].copy()
        visium = visium[:, shared_genes].copy()
        celltype_counts = reference.obs[celltype_col].astype(str).value_counts()
        keep_celltypes = celltype_counts.index[celltype_counts >= min_reference_cells]
        reference = reference[reference.obs[celltype_col].astype(str).isin(keep_celltypes)].copy()
        if max_reference_cells is not None and max_reference_cells > 0:
            reference_celltypes = reference.obs[celltype_col].astype(str).to_numpy()
            selected = np.zeros(reference.n_obs, dtype=bool)
            rng = np.random.default_rng(reference_subsample_seed)
            for cell_type in pd.unique(reference_celltypes):
                group_idx = np.flatnonzero(reference_celltypes == cell_type)
                if group_idx.size > max_reference_cells:
                    group_idx = np.sort(rng.choice(group_idx, size=max_reference_cells, replace=False))
                selected[group_idx] = True
            reference = reference[selected].copy()
        reference.var_names = reference.var_names.astype(str)
        reference.obs_names = reference.obs_names.astype(str)
        visium.var_names = visium.var_names.astype(str)
        visium.obs_names = visium.obs_names.astype(str)

        spatial_counts = _to_csc(
            (visium.layers[layer_key] if layer_key in visium.layers else visium.X).T
        )
        reference_counts = _to_csc(
            (reference.layers[layer_key] if layer_key in reference.layers else reference.X).T
        )

        if "spatial" not in visium.obsm:
            raise KeyError("Visium AnnData does not contain obsm['spatial'], required for RCTD.")
        spatial_coords = pd.DataFrame(
            visium.obsm["spatial"],
            index=visium.obs_names.astype(str),
            columns=["x", "y"],
        )
        reference_celltypes = pd.DataFrame(
            {"cell_type": reference.obs[celltype_col].astype(str).to_numpy()},
            index=reference.obs_names.astype(str),
        )
        reference_signatures = _reference_signature_means(reference, celltype_col=celltype_col, layer_key=layer_key)
        reference_signatures.to_csv(results_file(output_dir, "tables", "reference_signatures_means.csv"))

        r_script = project_root() / "scripts" / "run_rctd_native.R"
        with TemporaryDirectory(prefix="rctd_native_") as tmp_root:
            tmp = Path(tmp_root)
            spatial_mtx = tmp / "spatial_counts.mtx"
            spatial_genes = tmp / "spatial_genes.txt"
            spatial_barcodes = tmp / "spatial_barcodes.txt"
            spatial_coords_csv = tmp / "spatial_coords.csv"
            reference_mtx = tmp / "reference_counts.mtx"
            reference_genes = tmp / "reference_genes.txt"
            reference_cells = tmp / "reference_cells.txt"
            reference_celltypes_csv = tmp / "reference_celltypes.csv"
            weights_csv = tmp / "rctd_weights.csv"
            uncertainty_csv = tmp / "rctd_uncertainty.csv"
            results_df_csv = tmp / "rctd_results_df.csv"

            mmwrite(spatial_mtx, spatial_counts)
            _write_lines(spatial_genes, visium.var_names)
            _write_lines(spatial_barcodes, visium.obs_names)
            spatial_coords.to_csv(spatial_coords_csv, index=True)

            mmwrite(reference_mtx, reference_counts)
            _write_lines(reference_genes, reference.var_names)
            _write_lines(reference_cells, reference.obs_names)
            reference_celltypes.to_csv(reference_celltypes_csv, index=True)

            command = [
                "Rscript",
                str(r_script),
                str(spatial_mtx),
                str(spatial_genes),
                str(spatial_barcodes),
                str(spatial_coords_csv),
                str(reference_mtx),
                str(reference_genes),
                str(reference_cells),
                str(reference_celltypes_csv),
                str(weights_csv),
                str(uncertainty_csv),
                str(results_df_csv),
                str(int(model_cfg.get("rctd_max_cores", 8))),
                str(model_cfg.get("rctd_doublet_mode", "doublet")),
                str(float(model_cfg.get("rctd_umi_min", 100))),
                str(int(model_cfg.get("rctd_cell_min_instance", 25))),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "RCTD native run failed.\n"
                    f"STDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )

            abundance = pd.read_csv(weights_csv, index_col=0)
            uncertainty = pd.read_csv(uncertainty_csv, index_col=0).iloc[:, 0]
            uncertainty.name = "rctd_inverse_max_weight"
            results_df = pd.read_csv(results_df_csv, index_col=0)

        abundance.index = abundance.index.astype(str)
        abundance.columns = abundance.columns.astype(str)
        uncertainty.index = uncertainty.index.astype(str)
        results_df.index = results_df.index.astype(str)
        abundance, uncertainty, results_df, missing_spots = _align_rctd_outputs_to_visium(
            pd.Index(visium.obs_names.astype(str)),
            abundance,
            uncertainty,
            results_df,
        )

        results_df.to_csv(results_file(output_dir, "tables", "rctd_results_df.csv"))

        version_cmd = subprocess.run(
            ["Rscript", "-e", "cat(as.character(packageVersion('spacexr')))"],
            check=False,
            capture_output=True,
            text=True,
        )
        spacexr_version = version_cmd.stdout.strip() if version_cmd.returncode == 0 else "unknown"

        metadata = {
            "model_name": self.model_name,
            "integration_mode": "native",
            "backend": "spacexr",
            "spacexr_version": spacexr_version,
            "n_spots": int(visium.n_obs),
            "n_returned_spots": int(visium.n_obs - len(missing_spots)),
            "n_missing_spots_filled": int(len(missing_spots)),
            "n_reference_cells": int(reference.n_obs),
            "n_genes": int(len(shared_genes)),
            "reference_celltype_column": celltype_col,
            "uncertainty_source": "rctd_inverse_max_weight",
            "rctd_max_cores": int(model_cfg.get("rctd_max_cores", 8)),
            "rctd_doublet_mode": str(model_cfg.get("rctd_doublet_mode", "doublet")),
            "rctd_umi_min": float(model_cfg.get("rctd_umi_min", 100)),
            "rctd_cell_min_instance": int(model_cfg.get("rctd_cell_min_instance", 25)),
            "rctd_min_reference_cells_per_type": min_reference_cells,
            "rctd_max_reference_cells_per_type": max_reference_cells,
            "rctd_reference_subsample_seed": reference_subsample_seed,
            "has_uncertainty": True,
            "used_genes": list(map(str, shared_genes)),
        }
        return BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty,
            metadata=metadata,
        )
