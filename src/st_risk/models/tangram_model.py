from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner
from st_risk.paths import current_results_dir, ensure_results_layout, results_file


class TangramRunner(BaseSpatialModelRunner):
    model_name = "tangram"

    @staticmethod
    def _sample_gene_subsets(
        genes: pd.Index,
        *,
        repeats: int,
        gene_fraction: float,
        random_state: int,
    ) -> list[pd.Index]:
        if repeats <= 0:
            raise ValueError("tangram_ensemble_repeats must be positive.")
        if not 0 < gene_fraction <= 1:
            raise ValueError("tangram_gene_fraction must be in (0, 1].")
        if repeats == 1 and np.isclose(gene_fraction, 1.0):
            return [genes]
        n_take = max(2, int(round(len(genes) * gene_fraction)))
        rng = np.random.default_rng(random_state)
        subsets: list[pd.Index] = []
        gene_values = genes.to_numpy(dtype=object)
        for _ in range(repeats):
            picked = np.sort(rng.choice(gene_values, size=n_take, replace=False))
            subsets.append(pd.Index(picked.astype(str)))
        return subsets

    @staticmethod
    def _normalize_abundance(abundance: pd.DataFrame) -> pd.DataFrame:
        values = abundance.to_numpy(dtype=float)
        row_sums = values.sum(axis=1, keepdims=True)
        safe = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        return pd.DataFrame(values / safe, index=abundance.index, columns=abundance.columns)

    @classmethod
    def _aggregate_ensemble_predictions(
        cls,
        predictions: list[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if not predictions:
            raise ValueError("Tangram ensemble received no predictions.")
        aligned = [cls._normalize_abundance(pred).sort_index(axis=0).sort_index(axis=1) for pred in predictions]
        if len(aligned) == 1:
            return aligned[0], None
        stacked = np.stack([pred.to_numpy(dtype=float) for pred in aligned], axis=0)
        mean = pd.DataFrame(stacked.mean(axis=0), index=aligned[0].index, columns=aligned[0].columns)
        std = pd.DataFrame(stacked.std(axis=0), index=aligned[0].index, columns=aligned[0].columns)
        return cls._normalize_abundance(mean), std

    @staticmethod
    def _subsample_reference_by_celltype(
        reference,
        *,
        celltype_col: str,
        max_cells_per_type: int | None,
        random_state: int,
    ):
        if max_cells_per_type is None or max_cells_per_type <= 0:
            return reference
        sampled_obs: list[str] = []
        rng = np.random.default_rng(random_state)
        labels = reference.obs[celltype_col].astype(str)
        for celltype, obs_names in labels.groupby(labels, sort=False).groups.items():
            obs_index = np.asarray(list(obs_names), dtype=object)
            if len(obs_index) <= max_cells_per_type:
                sampled_obs.extend(obs_index.tolist())
                continue
            picked = rng.choice(obs_index, size=int(max_cells_per_type), replace=False)
            sampled_obs.extend(picked.tolist())
        return reference[sampled_obs, :].copy()

    @staticmethod
    def _resolve_device(raw_device: Any) -> str:
        if raw_device is None:
            return "cpu"
        if isinstance(raw_device, int):
            return "cpu" if raw_device < 0 else f"cuda:{raw_device}"
        text = str(raw_device).strip()
        if not text:
            return "cpu"
        if text.isdigit():
            return f"cuda:{text}"
        normalized = text.lower()
        if normalized in {"cpu", "cuda", "mps"}:
            return normalized
        return text

    def is_available(self) -> bool:
        try:
            import tangram  # noqa: F401
            return True
        except Exception:
            return False

    def run(
        self,
        visium_path: str | Path,
        reference_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> BaseSpatialModelOutput:
        if not self.is_available():
            raise NotImplementedError(
                "tangram is not installed in the current environment. "
                "Install the dependency before running the base model stage."
            )

        import anndata as ad
        import scanpy as sc
        import tangram as tg

        config = config or {}
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        preprocessing_cfg = config.get("preprocessing", {})
        outputs_cfg = config.get("outputs", {})

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

        celltype_col = dataset_cfg.get("reference_celltype_column") or choose_reference_celltype_column(reference.obs.columns)
        reference = reference[reference.obs[celltype_col].notna(), shared_genes].copy()
        visium = visium[:, shared_genes].copy()

        layer_key = model_cfg.get("layer", "counts")
        if layer_key in reference.layers:
            reference.X = reference.layers[layer_key].copy()
        if layer_key in visium.layers:
            visium.X = visium.layers[layer_key].copy()

        reference.var_names = reference.var_names.astype(str)
        visium.var_names = visium.var_names.astype(str)
        reference.obs_names = reference.obs_names.astype(str)
        visium.obs_names = visium.obs_names.astype(str)

        mode = str(model_cfg.get("tangram_mode", "clusters"))
        if mode != "clusters":
            raise ValueError("Tangram integration currently supports tangram_mode='clusters' only.")
        repeats = int(model_cfg.get("tangram_ensemble_repeats", 1))
        gene_fraction = float(model_cfg.get("tangram_gene_fraction", 1.0))
        base_seed = int(model_cfg.get("tangram_random_state", 0))
        reference_subsample_max_cells = model_cfg.get("tangram_reference_subsample_max_cells_per_type")
        if reference_subsample_max_cells is not None:
            reference_subsample_max_cells = int(reference_subsample_max_cells)
        gene_subsets = self._sample_gene_subsets(
            pd.Index(shared_genes.astype(str)),
            repeats=repeats,
            gene_fraction=gene_fraction,
            random_state=base_seed,
        )

        predictions: list[pd.DataFrame] = []
        for repeat_idx, gene_subset in enumerate(gene_subsets):
            ref_sub = self._subsample_reference_by_celltype(
                reference,
                celltype_col=celltype_col,
                max_cells_per_type=reference_subsample_max_cells,
                random_state=base_seed + repeat_idx,
            )
            ref_sub = ref_sub[:, gene_subset].copy()
            visium_sub = visium[:, gene_subset].copy()
            tg.pp_adatas(
                ref_sub,
                visium_sub,
                genes=list(gene_subset),
                gene_to_lowercase=bool(model_cfg.get("gene_to_lowercase", True)),
            )
            mapping = tg.map_cells_to_space(
                ref_sub,
                visium_sub,
                cluster_label=celltype_col,
                mode="clusters",
                device=self._resolve_device(model_cfg.get("device", "cpu")),
                learning_rate=float(model_cfg.get("tangram_learning_rate", 0.1)),
                num_epochs=int(model_cfg.get("tangram_num_epochs", 500)),
                scale=bool(model_cfg.get("tangram_scale", True)),
                lambda_d=float(model_cfg.get("tangram_lambda_d", 1.0)),
                lambda_g1=float(model_cfg.get("tangram_lambda_g1", 1.0)),
                lambda_g2=float(model_cfg.get("tangram_lambda_g2", 0.0)),
                lambda_r=float(model_cfg.get("tangram_lambda_r", 0.0)),
                density_prior=model_cfg.get("tangram_density_prior", "rna_count_based"),
                random_state=base_seed + repeat_idx,
                verbose=bool(model_cfg.get("tangram_verbose", True)),
            )
            tg.project_cell_annotations(mapping, visium_sub, annotation=celltype_col)
            abundance = visium_sub.obsm["tangram_ct_pred"].copy()
            if not isinstance(abundance, pd.DataFrame):
                abundance = pd.DataFrame(abundance, index=visium_sub.obs_names)
            abundance.index = abundance.index.astype(str)
            abundance.columns = abundance.columns.astype(str)
            predictions.append(abundance.loc[visium.obs_names])

        abundance, uncertainty = self._aggregate_ensemble_predictions(predictions)

        metadata = {
            "model_name": self.model_name,
            "integration_mode": "native",
            "n_spots": int(visium.n_obs),
            "n_reference_cells": int(reference.n_obs),
            "n_genes": int(len(shared_genes)),
            "reference_celltype_column": celltype_col,
            "tangram_mode": mode,
            "has_uncertainty": uncertainty is not None,
            "uncertainty_source": "tangram_ensemble_std" if uncertainty is not None else None,
            "tangram_ensemble_repeats": repeats,
            "tangram_gene_fraction": gene_fraction,
            "tangram_reference_subsample_max_cells_per_type": reference_subsample_max_cells,
            "used_genes": list(map(str, shared_genes)),
        }

        output_dir = current_results_dir(
            outputs_cfg.get("results_dir", "results/tangram"),
            run_id=outputs_cfg.get("run_id"),
            create=True,
        )
        ensure_results_layout(output_dir)
        reference_matrix = reference.X.toarray() if hasattr(reference.X, "toarray") else np.asarray(reference.X)
        reference_signatures = (
            pd.DataFrame(reference_matrix, index=reference.obs_names, columns=reference.var_names)
            .groupby(reference.obs[celltype_col].astype(str))
            .mean()
            .T
        )
        reference_signatures.to_csv(results_file(output_dir, "tables", "reference_signatures_means.csv"))

        return BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty,
            metadata=metadata,
        )
