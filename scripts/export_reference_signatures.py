from __future__ import annotations

import argparse
from pathlib import Path

from st_risk.config import load_config
from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.cell2location_model import _clean_prefixed_columns, export_reference_signatures
from st_risk.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cell2location reference signatures for a configured dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "example_cell2location.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    preprocessing_cfg = config.get("preprocessing", {})
    outputs_dir = Path(config["outputs"]["results_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    visium = open_h5ad(dataset_cfg["visium_h5ad"], backed=None)
    reference = open_h5ad(dataset_cfg["snrna_h5ad"], backed=None)
    shared = intersect_gene_names(visium.var_names, reference.var_names)
    if preprocessing_cfg.get("use_hvg_only", True) and "is_top_hvg" in visium.var.columns:
        hvg = visium.var.index[visium.var["is_top_hvg"].fillna(False)]
        shared = shared.intersection(hvg)
    max_genes = preprocessing_cfg.get("max_genes")
    if max_genes is not None:
        shared = shared[: int(max_genes)]
    celltype_col = dataset_cfg.get("reference_celltype_column") or choose_reference_celltype_column(reference.obs.columns)
    reference = reference[reference.obs[celltype_col].notna(), shared].copy()

    _, signatures = export_reference_signatures(
        reference,
        celltype_col=celltype_col,
        batch_key=model_cfg.get("reference_batch_key", "Sample"),
        layer=model_cfg.get("layer", "counts"),
        accelerator=model_cfg.get("accelerator", "gpu"),
        device=model_cfg.get("device", 1),
        reference_max_epochs=int(model_cfg.get("reference_max_epochs", 50)),
        reference_batch_size=int(model_cfg.get("reference_batch_size", 2048)),
        reference_posterior_samples=int(model_cfg.get("reference_posterior_samples", 200)),
        posterior_batch_size=int(model_cfg.get("posterior_batch_size", 1024)),
        early_stopping=bool(model_cfg.get("early_stopping", False)),
        signature_key=model_cfg.get("reference_signature_key", "means_per_cluster_mu_fg"),
    )
    cleaned = _clean_prefixed_columns(signatures, "means_per_cluster_mu_fg_")
    cleaned.to_csv(outputs_dir / "reference_signatures_means.csv")
    print(f"Saved reference signatures to {outputs_dir / 'reference_signatures_means.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
