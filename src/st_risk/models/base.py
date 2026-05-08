from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class BaseSpatialModelOutput:
    abundance: pd.DataFrame
    uncertainty: pd.DataFrame | pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSpatialModelRunner:
    model_name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> BaseSpatialModelOutput:
        raise NotImplementedError
