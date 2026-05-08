from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from st_risk.models.base import BaseSpatialModelOutput
from st_risk.models.cell2location_model import Cell2LocationRunner
from st_risk.models.destvi_model import (
    DestVIRunner,
    _destvi_sampled_proportion_summary,
    _normalize_destvi_sampled_proportions,
)
from st_risk.models.io import load_saved_base_model_output, save_base_model_output
from st_risk.models.precomputed_model import PrecomputedTableRunner
from st_risk.models.registry import build_model_runner
from st_risk.models.rctd_model import RCTDRunner, _align_rctd_outputs_to_visium
from st_risk.models.stereoscope_model import StereoscopeRunner
from st_risk.models.tangram_model import TangramRunner


def test_base_model_output_preserves_abundance_table():
    abundance = pd.DataFrame([[1.0, 2.0]], columns=["A", "B"], index=["spot_1"])
    output = BaseSpatialModelOutput(abundance=abundance, metadata={"model_name": "placeholder"})
    assert list(output.abundance.columns) == ["A", "B"]
    assert output.metadata["model_name"] == "placeholder"


def test_cell2location_runner_reports_missing_dependency():
    runner = Cell2LocationRunner()
    assert isinstance(runner.is_available(), bool)


def test_build_model_runner_returns_expected_runner_types():
    assert isinstance(build_model_runner("precomputed"), PrecomputedTableRunner)
    assert isinstance(build_model_runner("cell2location"), Cell2LocationRunner)
    assert isinstance(build_model_runner("rctd"), RCTDRunner)
    assert isinstance(build_model_runner("tangram"), TangramRunner)
    assert isinstance(build_model_runner("destvi"), DestVIRunner)
    assert isinstance(build_model_runner("stereoscope"), StereoscopeRunner)


def test_build_model_runner_rejects_unknown_model():
    try:
        build_model_runner("unknown")
    except ValueError as exc:
        assert "Unsupported model.name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported model.name")


def test_save_and_load_base_model_output_round_trip(tmp_path: Path):
    abundance = pd.DataFrame([[1.0, 2.0]], columns=["A", "B"], index=["spot_1"])
    uncertainty = pd.Series([0.2], index=["spot_1"], name="uncertainty")
    output = BaseSpatialModelOutput(
        abundance=abundance,
        uncertainty=uncertainty,
        metadata={"model_name": "placeholder"},
    )
    save_base_model_output(output, tmp_path)
    loaded = load_saved_base_model_output(tmp_path)
    assert loaded.metadata["model_name"] == "placeholder"
    assert loaded.abundance.equals(abundance)
    assert loaded.uncertainty.equals(uncertainty)


def test_precomputed_runner_aligns_to_visium_spots(tmp_path: Path):
    visium_path = tmp_path / "visium.h5ad"
    ref_path = tmp_path / "ref.h5ad"
    abundance_path = tmp_path / "abundance.csv"
    uncertainty_path = tmp_path / "uncertainty.csv"

    visium = ad.AnnData(X=np.ones((2, 2)))
    visium.obs_names = ["spot_1", "spot_2"]
    visium.var_names = ["g1", "g2"]
    visium.write_h5ad(visium_path)

    ref = ad.AnnData(X=np.ones((1, 2)))
    ref.obs_names = ["cell_1"]
    ref.var_names = ["g1", "g2"]
    ref.write_h5ad(ref_path)

    pd.DataFrame(
        {"Astro": [0.8, 0.2], "Excit": [0.2, 0.8]},
        index=["spot_2", "spot_1"],
    ).to_csv(abundance_path)
    pd.DataFrame({"uncertainty": [0.1, 0.3]}, index=["spot_2", "spot_1"]).to_csv(uncertainty_path)

    runner = PrecomputedTableRunner()
    runner.model_name = "precomputed"
    output = runner.run(
        visium_path,
        ref_path,
        config={
            "model": {
                "precomputed_abundance_csv": str(abundance_path),
                "precomputed_uncertainty_csv": str(uncertainty_path),
            }
        },
    )
    assert list(output.abundance.index) == ["spot_1", "spot_2"]
    assert output.metadata["model_name"] == "precomputed"
    assert output.metadata["has_uncertainty"] is True


def test_align_rctd_outputs_to_visium_fills_missing_spots():
    visium_index = pd.Index(["spot_1", "spot_2", "spot_3"])
    abundance = pd.DataFrame({"Astro": [0.8, 0.2]}, index=["spot_1", "spot_3"])
    uncertainty = pd.Series([0.1, 0.4], index=["spot_1", "spot_3"], name="uncertainty")
    results_df = pd.DataFrame({"spot_class": ["singlet", "doublet"]}, index=["spot_1", "spot_3"])

    aligned_abundance, aligned_uncertainty, aligned_results_df, missing = _align_rctd_outputs_to_visium(
        visium_index,
        abundance,
        uncertainty,
        results_df,
    )

    assert list(aligned_abundance.index) == ["spot_1", "spot_2", "spot_3"]
    assert float(aligned_abundance.loc["spot_2", "Astro"]) == 0.0
    assert float(aligned_uncertainty.loc["spot_2"]) == 1.0
    assert bool(aligned_results_df.loc["spot_2", "returned_by_rctd"]) is False
    assert list(missing) == ["spot_2"]


def test_tangram_runner_ensemble_gene_subsets_are_reproducible():
    genes = pd.Index(["g1", "g2", "g3", "g4"])
    subsets_a = TangramRunner._sample_gene_subsets(genes, repeats=3, gene_fraction=0.5, random_state=7)
    subsets_b = TangramRunner._sample_gene_subsets(genes, repeats=3, gene_fraction=0.5, random_state=7)
    assert [tuple(idx) for idx in subsets_a] == [tuple(idx) for idx in subsets_b]
    assert all(len(idx) >= 2 for idx in subsets_a)


def test_tangram_runner_aggregate_ensemble_predictions_returns_mean_and_std():
    preds = [
        pd.DataFrame([[0.8, 0.2], [0.2, 0.8]], index=["spot_1", "spot_2"], columns=["A", "B"]),
        pd.DataFrame([[0.6, 0.4], [0.4, 0.6]], index=["spot_1", "spot_2"], columns=["A", "B"]),
    ]
    mean, std = TangramRunner._aggregate_ensemble_predictions(preds)
    assert std is not None
    assert list(mean.index) == ["spot_1", "spot_2"]
    assert list(mean.columns) == ["A", "B"]
    assert np.allclose(mean.sum(axis=1).to_numpy(), 1.0)
    assert float(std.loc["spot_1", "A"]) > 0.0


def test_tangram_runner_reference_subsampling_is_reproducible(tmp_path: Path):
    adata = ad.AnnData(X=np.ones((6, 2)))
    adata.obs_names = [f"cell_{i}" for i in range(6)]
    adata.var_names = ["g1", "g2"]
    adata.obs["celltype"] = ["A", "A", "A", "B", "B", "B"]

    sub_a = TangramRunner._subsample_reference_by_celltype(
        adata,
        celltype_col="celltype",
        max_cells_per_type=2,
        random_state=3,
    )
    sub_b = TangramRunner._subsample_reference_by_celltype(
        adata,
        celltype_col="celltype",
        max_cells_per_type=2,
        random_state=3,
    )
    assert list(sub_a.obs_names) == list(sub_b.obs_names)
    counts = sub_a.obs["celltype"].value_counts().to_dict()
    assert counts == {"A": 2, "B": 2}


def test_stereoscope_runner_aggregate_ensemble_predictions_returns_mean_and_std():
    preds = [
        pd.DataFrame([[0.7, 0.3], [0.1, 0.9]], index=["spot_1", "spot_2"], columns=["A", "B"]),
        pd.DataFrame([[0.5, 0.5], [0.2, 0.8]], index=["spot_1", "spot_2"], columns=["A", "B"]),
    ]
    mean, std = StereoscopeRunner._aggregate_ensemble_predictions(preds)
    assert std is not None
    assert list(mean.index) == ["spot_1", "spot_2"]
    assert list(mean.columns) == ["A", "B"]
    assert np.allclose(mean.sum(axis=1).to_numpy(), 1.0)
    assert float(std.loc["spot_1", "A"]) > 0.0


def test_destvi_normalize_sampled_proportions_drops_additional_and_renormalizes():
    sampled_v = np.array(
        [
            [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]],
            [[0.3, 0.5, 0.2], [0.1, 0.5, 0.4]],
        ],
        dtype=float,
    )
    normalized = _normalize_destvi_sampled_proportions(
        sampled_v,
        add_celltypes=1,
        keep_additional=False,
        normalize=True,
    )
    assert normalized.shape == (2, 2, 2)
    assert np.allclose(normalized.sum(axis=2), 1.0)
    assert np.allclose(normalized[0, 0], [0.5, 0.5])


def test_destvi_sampled_proportion_summary_returns_mean_and_std():
    sampled_v = np.array(
        [
            [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]],
            [[0.3, 0.5, 0.2], [0.1, 0.5, 0.4]],
        ],
        dtype=float,
    )
    mean, std = _destvi_sampled_proportion_summary(
        sampled_v,
        index_names=pd.Index(["spot_1", "spot_2"]),
        column_names=["A", "B"],
        add_celltypes=1,
        keep_additional=False,
        normalize=True,
    )
    assert list(mean.index) == ["spot_1", "spot_2"]
    assert list(mean.columns) == ["A", "B"]
    assert list(std.columns) == ["A", "B"]
    assert np.allclose(mean.sum(axis=1).to_numpy(), 1.0)
    assert float(std.loc["spot_1", "A"]) > 0.0
