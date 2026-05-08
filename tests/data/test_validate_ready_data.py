import anndata as ad
import numpy as np

from st_risk.data.validate import validate_ready_directory, validate_ready_manifest


def test_validate_ready_manifest_requires_expected_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset": "Human DLPFC"}', encoding="utf-8")
    errors = validate_ready_manifest(manifest)
    assert any("ready_files" in error for error in errors)


def test_validate_ready_directory_accepts_sample_id_without_layer_guess(tmp_path):
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    manifest = ready_dir / "manifest.json"
    manifest.write_text(
        '{"dataset": "example", "ready_files": {"visium_h5ad": "spatial.h5ad", "snrna_h5ad": "reference.h5ad"}}',
        encoding="utf-8",
    )

    visium = ad.AnnData(X=np.ones((2, 3)))
    visium.obs_names = ["spot_1", "spot_2"]
    visium.var_names = ["g1", "g2", "g3"]
    visium.layers["counts"] = visium.X.copy()
    visium.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0]])
    visium.obs["sample_id"] = ["sample_a", "sample_a"]
    visium.write_h5ad(ready_dir / "spatial.h5ad")

    reference = ad.AnnData(X=np.ones((2, 3)))
    reference.obs_names = ["cell_1", "cell_2"]
    reference.var_names = ["g1", "g2", "g3"]
    reference.layers["counts"] = reference.X.copy()
    reference.obs["cell_type"] = ["A", "B"]
    reference.write_h5ad(ready_dir / "reference.h5ad")

    assert validate_ready_directory(ready_dir) == []


def test_validate_ready_directory_requires_sample_id(tmp_path):
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    manifest = ready_dir / "manifest.json"
    manifest.write_text(
        '{"dataset": "example", "ready_files": {"visium_h5ad": "spatial.h5ad", "snrna_h5ad": "reference.h5ad"}}',
        encoding="utf-8",
    )

    visium = ad.AnnData(X=np.ones((1, 2)))
    visium.obs_names = ["spot_1"]
    visium.var_names = ["g1", "g2"]
    visium.layers["counts"] = visium.X.copy()
    visium.obsm["spatial"] = np.array([[0.0, 0.0]])
    visium.write_h5ad(ready_dir / "spatial.h5ad")

    reference = ad.AnnData(X=np.ones((1, 2)))
    reference.obs_names = ["cell_1"]
    reference.var_names = ["g1", "g2"]
    reference.layers["counts"] = reference.X.copy()
    reference.obs["cell_type"] = ["A"]
    reference.write_h5ad(ready_dir / "reference.h5ad")

    errors = validate_ready_directory(ready_dir)
    assert any("sample_id" in error for error in errors)
