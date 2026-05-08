from __future__ import annotations

from st_risk.models.base import BaseSpatialModelRunner
from st_risk.models.cell2location_model import Cell2LocationRunner
from st_risk.models.destvi_model import DestVIRunner
from st_risk.models.precomputed_model import PrecomputedTableRunner
from st_risk.models.rctd_model import RCTDRunner
from st_risk.models.stereoscope_model import StereoscopeRunner
from st_risk.models.tangram_model import TangramRunner


def build_model_runner(model_name: str) -> BaseSpatialModelRunner:
    normalized = model_name.strip().lower()
    if normalized == "precomputed":
        return PrecomputedTableRunner()
    if normalized == "cell2location":
        return Cell2LocationRunner()
    if normalized == "rctd":
        return RCTDRunner()
    if normalized == "tangram":
        return TangramRunner()
    if normalized == "destvi":
        return DestVIRunner()
    if normalized == "stereoscope":
        return StereoscopeRunner()
    raise ValueError(
        f"Unsupported model.name '{model_name}'. Expected one of: precomputed, cell2location, rctd, tangram, destvi, stereoscope."
    )
