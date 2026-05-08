import pandas as pd

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names


def test_intersect_gene_names_preserves_shared_order():
    visium = pd.Index(["A", "B", "C"])
    ref = pd.Index(["B", "C", "D"])
    shared = intersect_gene_names(visium, ref)
    assert list(shared) == ["B", "C"]


def test_choose_reference_celltype_column_prefers_celltype_k():
    column = choose_reference_celltype_column(["cellType_hc", "cellType_k"])
    assert column == "cellType_k"
