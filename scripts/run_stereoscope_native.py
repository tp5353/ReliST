from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd
import scvi
import torch
from scvi.external.stereoscope import RNAStereoscope, SpatialStereoscope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native Stereoscope inside an scvi-enabled environment.")
    parser.add_argument("--visium-h5ad", type=Path, required=True)
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--celltype-column", type=str, required=True)
    parser.add_argument("--output-abundance-csv", type=Path, required=True)
    parser.add_argument("--output-metadata-json", type=Path, required=True)
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--sc-max-epochs", type=int, default=20)
    parser.add_argument("--sc-lr", type=float, default=0.01)
    parser.add_argument("--sc-batch-size", type=int, default=2048)
    parser.add_argument("--sp-max-epochs", type=int, default=50)
    parser.add_argument("--sp-lr", type=float, default=0.01)
    parser.add_argument("--sp-batch-size", type=int, default=1024)
    parser.add_argument("--prior-weight", type=str, default="n_obs")
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def _resolve_devices(raw: str, *, accelerator: str) -> int | str:
    text = str(raw).strip()
    if not text:
        return "auto"
    normalized_accelerator = str(accelerator).strip().lower()
    if normalized_accelerator in {"gpu", "cuda"} and text.isdigit():
        # scvi/lightning expects the number of devices, not a CUDA ordinal.
        return 1
    if text.isdigit():
        return int(text)
    return text


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = int(args.random_state)
    scvi_layer = args.layer if args.layer else None

    reference = ad.read_h5ad(args.reference_h5ad)
    visium = ad.read_h5ad(args.visium_h5ad)
    reference.obs[args.celltype_column] = reference.obs[args.celltype_column].astype(str)

    RNAStereoscope.setup_anndata(reference, labels_key=args.celltype_column, layer=scvi_layer)
    sc_model = RNAStereoscope(reference)
    sc_model.train(
        max_epochs=args.sc_max_epochs,
        lr=args.sc_lr,
        accelerator=args.accelerator,
        devices=_resolve_devices(args.devices, accelerator=args.accelerator),
        batch_size=args.sc_batch_size,
        train_size=1.0,
        validation_size=None,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
    )

    SpatialStereoscope.setup_anndata(visium, layer=scvi_layer)
    spatial_model = SpatialStereoscope.from_rna_model(
        visium,
        sc_model,
        prior_weight=args.prior_weight,
    )
    spatial_model.train(
        max_epochs=args.sp_max_epochs,
        lr=args.sp_lr,
        accelerator=args.accelerator,
        devices=_resolve_devices(args.devices, accelerator=args.accelerator),
        batch_size=args.sp_batch_size,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
    )

    abundance = spatial_model.get_proportions(keep_noise=False)
    if not isinstance(abundance, pd.DataFrame):
        abundance = pd.DataFrame(abundance, index=visium.obs_names)
    abundance.index = abundance.index.astype(str)
    abundance.columns = abundance.columns.astype(str)
    abundance.to_csv(args.output_abundance_csv)

    metadata = {
        "scvi_version": __import__("scvi").__version__,
        "n_reference_cells": int(reference.n_obs),
        "n_spots": int(visium.n_obs),
        "n_genes": int(reference.n_vars),
        "stereoscope_sc_max_epochs": int(args.sc_max_epochs),
        "stereoscope_sp_max_epochs": int(args.sp_max_epochs),
        "stereoscope_sc_lr": float(args.sc_lr),
        "stereoscope_sp_lr": float(args.sp_lr),
        "stereoscope_sc_batch_size": int(args.sc_batch_size),
        "stereoscope_sp_batch_size": int(args.sp_batch_size),
        "stereoscope_prior_weight": str(args.prior_weight),
        "stereoscope_accelerator": str(args.accelerator),
        "stereoscope_devices": str(args.devices),
        "random_state": int(args.random_state),
    }
    args.output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
