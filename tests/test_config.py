from st_risk.config import load_config


def test_load_config_reads_dataset_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset:\n"
        "  visium_h5ad: visium.h5ad\n"
        "  snrna_h5ad: ref.h5ad\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["dataset"]["visium_h5ad"] == "visium.h5ad"
    assert config["dataset"]["snrna_h5ad"] == "ref.h5ad"
