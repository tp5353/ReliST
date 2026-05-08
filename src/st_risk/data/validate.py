from __future__ import annotations

from pathlib import Path

from st_risk.data.io import load_manifest, open_h5ad

REQUIRED_READY_FILES = ("visium_h5ad", "snrna_h5ad")
REQUIRED_VISIUM_OBS_COLUMNS = ("sample_id",)
REQUIRED_REF_OBS_CANDIDATES = (
    "cell_type",
    "cellType_k",
    "cellType_broad_k",
    "cellType_hc",
    "cellType_broad_hc",
    "cellType_layer",
)


def validate_ready_manifest(path: str | Path) -> list[str]:
    errors: list[str] = []
    manifest = load_manifest(path)
    if "dataset" not in manifest:
        errors.append("Manifest missing required key: dataset")
    ready_files = manifest.get("ready_files")
    if not isinstance(ready_files, dict):
        errors.append("Manifest missing required key: ready_files")
        return errors
    for key in REQUIRED_READY_FILES:
        if key not in ready_files:
            errors.append(f"Manifest missing ready_files entry: {key}")
    return errors


def validate_ready_directory(ready_dir: str | Path) -> list[str]:
    ready_path = Path(ready_dir)
    manifest_path = ready_path / "manifest.json"
    errors = validate_ready_manifest(manifest_path)
    if errors:
        return errors

    manifest = load_manifest(manifest_path)
    ready_files = manifest["ready_files"]
    visium_path = ready_path / ready_files["visium_h5ad"]
    ref_path = ready_path / ready_files["snrna_h5ad"]

    for path in (visium_path, ref_path):
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    if errors:
        return errors

    errors.extend(_validate_visium(visium_path))
    errors.extend(_validate_reference(ref_path))
    return errors


def _validate_visium(path: Path) -> list[str]:
    errors: list[str] = []
    adata = open_h5ad(path, backed="r")
    try:
        if "counts" not in adata.layers.keys():
            errors.append(f"{path.name}: missing layers['counts']")
        if "spatial" not in adata.obsm.keys():
            errors.append(f"{path.name}: missing obsm['spatial']")
        missing_obs = [col for col in REQUIRED_VISIUM_OBS_COLUMNS if col not in adata.obs.columns]
        if missing_obs:
            errors.append(f"{path.name}: missing obs columns {missing_obs}")
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
    return errors


def _validate_reference(path: Path) -> list[str]:
    errors: list[str] = []
    adata = open_h5ad(path, backed="r")
    try:
        if "counts" not in adata.layers.keys():
            errors.append(f"{path.name}: missing layers['counts']")
        if not any(col in adata.obs.columns for col in REQUIRED_REF_OBS_CANDIDATES):
            errors.append(
                f"{path.name}: missing all candidate cell type columns {list(REQUIRED_REF_OBS_CANDIDATES)}"
            )
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
    return errors
