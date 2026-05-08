from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner


class PrecomputedTableRunner(BaseSpatialModelRunner):
    model_name = "precomputed"

    def is_available(self) -> bool:
        return True

    def run(
        self,
        visium_path: str | Path,
        reference_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> BaseSpatialModelOutput:
        del reference_path  # external outputs are already computed
        config = config or {}
        model_cfg = config.get("model", {})
        abundance_path = model_cfg.get("precomputed_abundance_csv")
        if not abundance_path:
            raise ValueError(f"{self.model_name} requires model.precomputed_abundance_csv")

        abundance = pd.read_csv(abundance_path, index_col=0)
        uncertainty = None
        uncertainty_path = model_cfg.get("precomputed_uncertainty_csv")
        if uncertainty_path:
            uncertainty = pd.read_csv(uncertainty_path, index_col=0)
            if uncertainty.shape[1] == 1:
                uncertainty = uncertainty.iloc[:, 0]

        visium = open_h5ad(visium_path, backed="r")
        try:
            visium_index = pd.Index(visium.obs_names.astype(str))
        finally:
            if hasattr(visium, "file") and visium.file is not None:
                visium.file.close()

        abundance.index = abundance.index.astype(str)
        shared_index = visium_index.intersection(abundance.index)
        if shared_index.empty:
            raise ValueError(
                f"{self.model_name} abundance table does not share spot IDs with the Visium dataset."
            )
        abundance = abundance.loc[shared_index]
        if uncertainty is not None:
            uncertainty.index = uncertainty.index.astype(str)
            uncertainty = uncertainty.reindex(shared_index)

        metadata = {
            "model_name": self.model_name,
            "integration_mode": "precomputed_tables",
            "n_spots": int(len(shared_index)),
            "n_celltypes": int(abundance.shape[1]),
            "has_uncertainty": uncertainty is not None,
            "precomputed_abundance_csv": str(abundance_path),
        }
        if uncertainty_path:
            metadata["precomputed_uncertainty_csv"] = str(uncertainty_path)

        return BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty,
            metadata=metadata,
        )


class RCTDRunner(PrecomputedTableRunner):
    model_name = "rctd"


class TangramRunner(PrecomputedTableRunner):
    model_name = "tangram"


class DestVIRunner(PrecomputedTableRunner):
    model_name = "destvi"
