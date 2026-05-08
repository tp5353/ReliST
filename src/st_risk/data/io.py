from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest at {path} must contain a JSON object.")
    return data


def open_h5ad(path: str | Path, *, backed: str | None = "r"):
    return ad.read_h5ad(Path(path), backed=backed)
