from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

DEFAULT_REFERENCE_CELLTYPE_COLUMNS = (
    "cellType_k",
    "cellType_broad_k",
    "cellType_hc",
    "cellType_broad_hc",
    "cellType_layer",
)


def intersect_gene_names(visium_index: pd.Index, ref_index: pd.Index) -> pd.Index:
    return visium_index.intersection(ref_index)


def choose_reference_celltype_column(
    available_columns: Sequence[str],
    preferred_columns: Sequence[str] = DEFAULT_REFERENCE_CELLTYPE_COLUMNS,
) -> str:
    available = set(available_columns)
    for column in preferred_columns:
        if column in available:
            return column
    raise ValueError(
        "No suitable reference cell type column found. "
        f"Checked columns: {list(preferred_columns)}"
    )
