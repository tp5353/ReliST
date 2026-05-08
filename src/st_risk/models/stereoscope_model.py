from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner
from st_risk.paths import current_results_dir, ensure_results_layout, project_root, results_file


def _reference_signature_means(reference, *, celltype_col: str, layer_key: str) -> pd.DataFrame:
    matrix = reference.layers[layer_key] if layer_key in reference.layers else reference.X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return (
        pd.DataFrame(matrix, index=reference.obs_names, columns=reference.var_names)
        .groupby(reference.obs[celltype_col].astype(str))
        .mean()
        .T
    )


class StereoscopeRunner(BaseSpatialModelRunner):
    model_name = "stereoscope"

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
            raise ValueError("Stereoscope ensemble received no predictions.")
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
        for _, obs_names in labels.groupby(labels, sort=False).groups.items():
            obs_index = np.asarray(list(obs_names), dtype=object)
            if len(obs_index) <= max_cells_per_type:
                sampled_obs.extend(obs_index.tolist())
                continue
            picked = rng.choice(obs_index, size=int(max_cells_per_type), replace=False)
            sampled_obs.extend(picked.tolist())
        return reference[sampled_obs, :].copy()

    @staticmethod
    def _resolve_devices(raw_device: Any) -> str:
        if raw_device is None:
            return "auto"
        if isinstance(raw_device, int):
            return str(raw_device)
        text = str(raw_device).strip()
        return text or "auto"

    def is_available(self) -> bool:
        env_name = "st-scvi"
        try:
            completed = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "-c",
                    "from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope",
                ],
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
        config = config or {}
        dataset_cfg = config.get("dataset", {})
        preprocessing_cfg = config.get("preprocessing", {})
        model_cfg = config.get("model", {})
        outputs_cfg = config.get("outputs", {})
        env_name = str(model_cfg.get("stereoscope_conda_env", "st-scvi"))

        try:
            completed = subprocess.run(
                [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    "-c",
                    "from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise NotImplementedError("conda is not available; cannot launch Stereoscope backend.") from exc
        if completed.returncode != 0:
            raise NotImplementedError(
                f"Stereoscope is not available in conda env '{env_name}'. "
                "Install scvi-tools with scvi.external.stereoscope support before running."
            )

        output_dir = current_results_dir(
            outputs_cfg.get("results_dir", "results/stereoscope"),
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

        min_reference_cells = int(model_cfg.get("stereoscope_min_reference_cells_per_type", 10))
        celltype_counts = reference.obs[celltype_col].astype(str).value_counts()
        keep_celltypes = celltype_counts.index[celltype_counts >= min_reference_cells]
        reference = reference[reference.obs[celltype_col].astype(str).isin(keep_celltypes)].copy()
        if reference.n_obs == 0:
            raise ValueError("No reference cells remain after Stereoscope cell-type filtering.")

        reference_max_cells = model_cfg.get("stereoscope_reference_max_cells_per_type")
        if reference_max_cells is not None:
            reference_max_cells = int(reference_max_cells)
        repeats = int(model_cfg.get("stereoscope_ensemble_repeats", 1))
        if repeats <= 0:
            raise ValueError("stereoscope_ensemble_repeats must be positive.")
        base_seed = int(model_cfg.get("stereoscope_random_state", 0))

        layer_key = str(model_cfg.get("layer", "counts"))
        reference.var_names = reference.var_names.astype(str)
        visium.var_names = visium.var_names.astype(str)
        reference.obs_names = reference.obs_names.astype(str)
        visium.obs_names = visium.obs_names.astype(str)

        reference_signatures = _reference_signature_means(reference, celltype_col=celltype_col, layer_key=layer_key)
        reference_signatures.to_csv(results_file(output_dir, "tables", "reference_signatures_means.csv"))

        native_script = project_root() / "scripts" / "run_stereoscope_native.py"
        with TemporaryDirectory(prefix="stereoscope_native_") as tmp_root:
            tmp_root_path = Path(tmp_root)
            visium_h5ad_tmp = tmp_root_path / "visium.h5ad"
            metadata_json = tmp_root_path / "stereoscope_metadata.json"

            visium.write_h5ad(visium_h5ad_tmp)
            predictions: list[pd.DataFrame] = []
            native_metadata: dict[str, Any] | None = None
            for repeat_idx in range(repeats):
                reference_h5ad_tmp = tmp_root_path / f"reference_{repeat_idx}.h5ad"
                abundance_csv = tmp_root_path / f"stereoscope_abundance_{repeat_idx}.csv"
                ref_sub = self._subsample_reference_by_celltype(
                    reference,
                    celltype_col=celltype_col,
                    max_cells_per_type=reference_max_cells,
                    random_state=base_seed + repeat_idx,
                )
                ref_sub.write_h5ad(reference_h5ad_tmp)

                command = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    str(native_script),
                    "--visium-h5ad",
                    str(visium_h5ad_tmp),
                    "--reference-h5ad",
                    str(reference_h5ad_tmp),
                    "--celltype-column",
                    celltype_col,
                    "--output-abundance-csv",
                    str(abundance_csv),
                    "--output-metadata-json",
                    str(metadata_json),
                    "--accelerator",
                    str(model_cfg.get("accelerator", "gpu")),
                    "--devices",
                    self._resolve_devices(model_cfg.get("device", "0")),
                    "--sc-max-epochs",
                    str(int(model_cfg.get("stereoscope_sc_max_epochs", 20))),
                    "--sc-lr",
                    str(float(model_cfg.get("stereoscope_sc_lr", 0.01))),
                    "--sc-batch-size",
                    str(int(model_cfg.get("stereoscope_sc_batch_size", 2048))),
                    "--sp-max-epochs",
                    str(int(model_cfg.get("stereoscope_sp_max_epochs", 50))),
                    "--sp-lr",
                    str(float(model_cfg.get("stereoscope_sp_lr", 0.01))),
                    "--sp-batch-size",
                    str(int(model_cfg.get("stereoscope_sp_batch_size", 1024))),
                    "--prior-weight",
                    str(model_cfg.get("stereoscope_prior_weight", "n_obs")),
                    "--random-state",
                    str(base_seed + repeat_idx),
                ]
                if layer_key:
                    command.extend(["--layer", layer_key])

                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Stereoscope native run failed.\n"
                        f"STDOUT:\n{completed.stdout}\n"
                        f"STDERR:\n{completed.stderr}"
                    )
                pred = pd.read_csv(abundance_csv, index_col=0)
                pred.index = pred.index.astype(str)
                pred.columns = pred.columns.astype(str)
                predictions.append(pred.reindex(visium.obs_names).fillna(0.0))
                if native_metadata is None:
                    native_metadata = json.loads(metadata_json.read_text(encoding="utf-8"))

        abundance, uncertainty = self._aggregate_ensemble_predictions(predictions)

        metadata = {
            "model_name": self.model_name,
            "integration_mode": "native",
            "backend": "scvi.external.stereoscope",
            "stereoscope_conda_env": env_name,
            "n_spots": int(visium.n_obs),
            "n_reference_cells": int(reference.n_obs),
            "n_genes": int(len(shared_genes)),
            "reference_celltype_column": celltype_col,
            "has_uncertainty": uncertainty is not None,
            "uncertainty_source": "stereoscope_ensemble_std" if uncertainty is not None else None,
            "used_genes": list(map(str, shared_genes)),
            "stereoscope_reference_max_cells_per_type": reference_max_cells,
            "stereoscope_min_reference_cells_per_type": min_reference_cells,
            "stereoscope_ensemble_repeats": repeats,
            **(native_metadata or {}),
        }
        return BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty,
            metadata=metadata,
        )
