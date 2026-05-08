from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from st_risk.data.harmonize import choose_reference_celltype_column, intersect_gene_names
from st_risk.data.io import open_h5ad
from st_risk.models.base import BaseSpatialModelOutput, BaseSpatialModelRunner
from st_risk.paths import current_results_dir, ensure_results_layout, results_file


def _resolve_scvi_devices(raw_device: Any) -> int | str:
    if raw_device is None:
        return "auto"
    if isinstance(raw_device, int):
        return raw_device
    text = str(raw_device).strip()
    if not text:
        return "auto"
    if text.isdigit():
        return int(text)
    return text


def _resolve_prior_device(accelerator: str, devices: int | str) -> str | None:
    normalized = str(accelerator).strip().lower()
    if normalized not in {"gpu", "cuda", "auto"}:
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    if isinstance(devices, int) and devices <= 0:
        return None
    if isinstance(devices, str) and devices.strip().lower() == "cpu":
        return None
    return "cuda:0"


def _reference_signature_means(reference, *, celltype_col: str, layer_key: str) -> pd.DataFrame:
    if layer_key in reference.layers:
        matrix = reference.layers[layer_key]
    else:
        matrix = reference.X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return (
        pd.DataFrame(matrix, index=reference.obs_names, columns=reference.var_names)
        .groupby(reference.obs[celltype_col].astype(str))
        .mean()
        .T
    )


def _normalize_destvi_sampled_proportions(
    sampled_v: np.ndarray,
    *,
    add_celltypes: int,
    keep_additional: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    values = np.asarray(sampled_v, dtype=float)
    if values.ndim != 3:
        raise ValueError("sampled_v must have shape (n_samples, n_spots, n_celltypes)")
    if not keep_additional and add_celltypes > 0:
        values = values[:, :, :-add_celltypes]
    if normalize:
        denom = values.sum(axis=2, keepdims=True)
        denom = np.where(np.isclose(denom, 0.0), 1.0, denom)
        values = values / denom
    return values


def _destvi_sampled_proportion_summary(
    sampled_v: np.ndarray,
    *,
    index_names: pd.Index,
    column_names: list[str],
    add_celltypes: int,
    keep_additional: bool = False,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = _normalize_destvi_sampled_proportions(
        sampled_v,
        add_celltypes=add_celltypes,
        keep_additional=keep_additional,
        normalize=normalize,
    )
    mean = pd.DataFrame(
        values.mean(axis=0),
        index=index_names,
        columns=column_names,
    )
    std = pd.DataFrame(
        values.std(axis=0, ddof=0),
        index=index_names,
        columns=column_names,
    )
    return mean, std


def _destvi_from_rna_model_compat(
    st_adata,
    sc_model,
    *,
    prior_device: str | None = None,
    vamp_prior_p: int | None = None,
    anndata_setup_kwargs: dict[str, Any] | None = None,
    **module_kwargs,
):
    import torch
    from scvi.data._constants import _SETUP_ARGS_KEY
    from scvi.model import DestVI
    from scvi.model.base._archesmixin import _get_loaded_data

    attr_dict, _, load_state_dict, _ = _get_loaded_data(sc_model)
    registry = attr_dict.pop("registry_")

    decoder_state_dict = OrderedDict(
        (name[8:], value) for name, value in load_state_dict.items() if name.split(".")[0] == "decoder"
    )
    px_decoder_state_dict = OrderedDict(
        (name[11:], value) for name, value in load_state_dict.items() if name.split(".")[0] == "px_decoder"
    )
    px_r = load_state_dict["px_r"]
    per_ct_bias = load_state_dict["per_ct_bias"]
    mapping = registry["field_registries"]["labels"]["state_registry"]["categorical_mapping"]

    init_kwargs = attr_dict["init_params_"].get("kwargs", {})
    module_init_kwargs = init_kwargs.get("module_kwargs", {})
    prior = module_init_kwargs.get("prior", "normal")
    dropout_decoder = attr_dict["init_params_"]["non_kwargs"]["dropout_rate"]

    target_device = torch.device(prior_device) if prior_device is not None else None

    def _to_tensor(values):
        if values is None:
            return None
        tensor = torch.as_tensor(values, dtype=px_r.dtype)
        return tensor.to(target_device) if target_device is not None else tensor

    if vamp_prior_p is None:
        mean_vprior = None
        var_vprior = None
        mp_vprior = None
    elif prior == "mog":
        mean_vprior = load_state_dict["prior_means"].clone().detach()
        var_vprior = torch.exp(load_state_dict["prior_log_std"]) ** 2
        mp_vprior = torch.nn.functional.softmax(load_state_dict["prior_logits"], dim=-1)
        if target_device is not None:
            mean_vprior = mean_vprior.to(target_device)
            var_vprior = var_vprior.to(target_device)
            mp_vprior = mp_vprior.to(target_device)
    else:
        vamp = sc_model.get_vamp_prior(sc_model.adata, p=vamp_prior_p)
        mean_vprior = _to_tensor(vamp["mean_vprior"])
        var_vprior = _to_tensor(vamp["var_vprior"])
        mp_vprior = _to_tensor(vamp["weights_vprior"])

    setup_kwargs = dict(registry[_SETUP_ARGS_KEY])
    if anndata_setup_kwargs:
        setup_kwargs.update(anndata_setup_kwargs)
    DestVI.setup_anndata(
        st_adata,
        source_registry=registry,
        extend_categories=True,
        **setup_kwargs,
    )
    return DestVI(
        st_adata,
        mapping,
        decoder_state_dict,
        px_decoder_state_dict,
        px_r,
        per_ct_bias,
        sc_model.module.n_hidden,
        sc_model.module.n_latent,
        sc_model.module.n_layers,
        mean_vprior=mean_vprior,
        var_vprior=var_vprior,
        mp_vprior=mp_vprior,
        dropout_decoder=dropout_decoder,
        **module_kwargs,
    )


def _destvi_posterior_proportion_summary(
    spatial_model,
    *,
    posterior_samples: int,
    batch_size: int | None = None,
    keep_additional: bool = False,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import torch

    spatial_model._check_if_trained()
    index_names = spatial_model.adata.obs.index
    if keep_additional:
        column_names = list(spatial_model.cell_type_mapping_extended)
    else:
        column_names = list(spatial_model.cell_type_mapping)

    stdl = spatial_model._make_data_loader(
        adata=spatial_model.adata,
        indices=None,
        batch_size=batch_size,
    )
    mean_chunks: list[pd.DataFrame] = []
    std_chunks: list[pd.DataFrame] = []
    current = 0
    for tensors in stdl:
        inference_inputs = spatial_model.module._get_inference_input(tensors)
        outputs = spatial_model.module.inference(**inference_inputs)
        qz = outputs["qz"]
        loc = qz.loc[0, ...]
        scale = qz.scale[0, ...]
        sampled_z = loc.unsqueeze(0) + scale.unsqueeze(0) * torch.randn(
            (posterior_samples, *loc.shape),
            device=loc.device,
            dtype=loc.dtype,
        )
        outputs = dict(outputs)
        outputs["z"] = sampled_z
        generative_inputs = spatial_model.module._get_generative_input(tensors, outputs)
        generative_outputs = spatial_model.module.generative(**generative_inputs)
        sampled_v = generative_outputs["v"].detach().cpu().numpy()
        batch_n = sampled_v.shape[1]
        batch_index = index_names[current : current + batch_n]
        current += batch_n
        mean_df, std_df = _destvi_sampled_proportion_summary(
            sampled_v,
            index_names=batch_index,
            column_names=column_names,
            add_celltypes=spatial_model.module.add_celltypes,
            keep_additional=keep_additional,
            normalize=normalize,
        )
        mean_chunks.append(mean_df)
        std_chunks.append(std_df)
    return pd.concat(mean_chunks), pd.concat(std_chunks)


class DestVIRunner(BaseSpatialModelRunner):
    model_name = "destvi"

    def is_available(self) -> bool:
        try:
            import scvi  # noqa: F401
            return True
        except Exception:
            return False

    def run(
        self,
        visium_path: str | Path,
        reference_path: str | Path,
        *,
        config: dict[str, Any] | None = None,
    ) -> BaseSpatialModelOutput:
        if not self.is_available():
            raise NotImplementedError(
                "scvi-tools is not installed in the current environment. "
                "Install the dependency before running the base model stage."
            )

        import torch
        from scvi.model import CondSCVI

        torch.set_float32_matmul_precision("high")

        config = config or {}
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        preprocessing_cfg = config.get("preprocessing", {})
        outputs_cfg = config.get("outputs", {})

        output_dir = current_results_dir(
            outputs_cfg.get("results_dir", "results/destvi"),
            run_id=outputs_cfg.get("run_id"),
            create=True,
        )
        ensure_results_layout(output_dir)

        visium = open_h5ad(visium_path, backed=None)
        reference = open_h5ad(reference_path, backed=None)

        shared_genes = intersect_gene_names(visium.var_names, reference.var_names)
        if preprocessing_cfg.get("use_hvg_only", True) and "is_top_hvg" in visium.var.columns:
            hvg = visium.var.index[visium.var["is_top_hvg"].fillna(False)]
            shared_genes = shared_genes.intersection(hvg)
        max_genes = preprocessing_cfg.get("max_genes")
        if max_genes is not None:
            shared_genes = shared_genes[: int(max_genes)]
        if len(shared_genes) == 0:
            raise ValueError("No shared genes remain after preprocessing filters.")

        celltype_col = dataset_cfg.get("reference_celltype_column") or choose_reference_celltype_column(
            reference.obs.columns
        )
        reference = reference[reference.obs[celltype_col].notna(), shared_genes].copy()
        visium = visium[:, shared_genes].copy()

        layer_key = str(model_cfg.get("layer", "counts"))
        if layer_key in reference.layers:
            reference.X = reference.layers[layer_key].copy()
        if layer_key in visium.layers:
            visium.X = visium.layers[layer_key].copy()

        reference.var_names = reference.var_names.astype(str)
        visium.var_names = visium.var_names.astype(str)
        reference.obs_names = reference.obs_names.astype(str)
        visium.obs_names = visium.obs_names.astype(str)

        reference_batch_key = str(model_cfg.get("reference_batch_key", "Sample"))
        visium_batch_key = str(model_cfg.get("visium_batch_key", "sample_id"))
        if reference_batch_key not in visium.obs.columns:
            if visium_batch_key not in visium.obs.columns:
                raise KeyError(
                    f"Neither visium batch key '{visium_batch_key}' nor reference batch key "
                    f"'{reference_batch_key}' exists in visium.obs."
                )
            visium.obs[reference_batch_key] = visium.obs[visium_batch_key].astype(str)

        accelerator = str(model_cfg.get("accelerator", "gpu"))
        devices = _resolve_scvi_devices(model_cfg.get("device", "auto"))
        prior_device = _resolve_prior_device(accelerator, devices)

        CondSCVI.setup_anndata(
            reference,
            layer=layer_key,
            batch_key=reference_batch_key,
            labels_key=celltype_col,
        )
        sc_model = CondSCVI(
            reference,
            n_hidden=int(model_cfg.get("condscvi_n_hidden", 128)),
            n_latent=int(model_cfg.get("condscvi_n_latent", 5)),
            n_layers=int(model_cfg.get("condscvi_n_layers", 2)),
            dropout_rate=float(model_cfg.get("condscvi_dropout_rate", 0.05)),
            prior=str(model_cfg.get("condscvi_prior", "normal")),
        )
        sc_model.train(
            max_epochs=int(model_cfg.get("condscvi_max_epochs", 100)),
            lr=float(model_cfg.get("condscvi_lr", 1e-3)),
            accelerator=accelerator,
            devices=devices,
            batch_size=int(model_cfg.get("condscvi_batch_size", 2048)),
            train_size=1,
            validation_size=None,
            check_val_every_n_epoch=None,
            enable_checkpointing=False,
            logger=False,
        )

        spatial_model = _destvi_from_rna_model_compat(
            visium,
            sc_model,
            prior_device=prior_device,
            vamp_prior_p=model_cfg.get("destvi_vamp_prior_p"),
            anndata_setup_kwargs={"batch_key": reference_batch_key, "layer": layer_key},
            amortization=str(model_cfg.get("destvi_amortization", "latent")),
            n_latent_amortization=int(model_cfg.get("destvi_n_latent_amortization", 32)),
            prior_mode=str(model_cfg.get("destvi_prior_mode", "normal")),
            n_states_per_label=int(model_cfg.get("destvi_n_states_per_label", 3)),
        )
        spatial_model.train(
            max_epochs=int(model_cfg.get("destvi_max_epochs", 200)),
            lr=float(model_cfg.get("destvi_lr", 3e-3)),
            accelerator=accelerator,
            devices=devices,
            batch_size=int(model_cfg.get("destvi_batch_size", 1024)),
            train_size=1,
            validation_size=None,
            n_epochs_kl_warmup=int(model_cfg.get("destvi_n_epochs_kl_warmup", 200)),
            check_val_every_n_epoch=None,
            enable_checkpointing=False,
            logger=False,
        )

        posterior_batch_size = int(model_cfg.get("posterior_batch_size", model_cfg.get("destvi_batch_size", 1024)))
        uncertainty_mode = str(model_cfg.get("destvi_uncertainty_mode", "latent")).strip().lower()
        if uncertainty_mode in {"posterior", "posterior_proportions", "posterior_abundance"}:
            posterior_samples = int(model_cfg.get("destvi_posterior_samples", 16))
            abundance, uncertainty = _destvi_posterior_proportion_summary(
                spatial_model,
                posterior_samples=posterior_samples,
                batch_size=posterior_batch_size,
                keep_additional=False,
                normalize=True,
            )
            uncertainty_source = "destvi_posterior_abundance_std"
        elif uncertainty_mode == "latent":
            abundance = spatial_model.get_proportions(batch_size=posterior_batch_size).copy()
            _, latent_var = spatial_model.get_latent_representation(
                return_dist=True,
                batch_size=posterior_batch_size,
            )
            uncertainty = pd.Series(
                latent_var.mean(axis=1),
                index=abundance.index.astype(str),
                name="destvi_latent_variance",
            )
            uncertainty_source = "destvi_latent_variance"
            posterior_samples = None
        else:
            raise ValueError(
                "Unsupported destvi_uncertainty_mode. Expected one of: latent, posterior_proportions."
            )
        abundance.index = abundance.index.astype(str)
        abundance.columns = abundance.columns.astype(str)
        if isinstance(uncertainty, pd.DataFrame):
            uncertainty.index = uncertainty.index.astype(str)
            uncertainty.columns = uncertainty.columns.astype(str)

        reference_signatures = _reference_signature_means(reference, celltype_col=celltype_col, layer_key=layer_key)
        reference_signatures.to_csv(results_file(output_dir, "tables", "reference_signatures_means.csv"))

        metadata = {
            "model_name": self.model_name,
            "integration_mode": "native",
            "backend": "scvi-tools",
            "n_spots": int(visium.n_obs),
            "n_reference_cells": int(reference.n_obs),
            "n_genes": int(len(shared_genes)),
            "reference_celltype_column": celltype_col,
            "reference_batch_key": reference_batch_key,
            "visium_batch_key": visium_batch_key,
            "uncertainty_source": uncertainty_source,
            "destvi_uncertainty_mode": uncertainty_mode,
            "destvi_posterior_samples": posterior_samples,
            "destvi_amortization": str(model_cfg.get("destvi_amortization", "latent")),
            "destvi_n_latent_amortization": int(model_cfg.get("destvi_n_latent_amortization", 32)),
            "destvi_prior_mode": str(model_cfg.get("destvi_prior_mode", "normal")),
            "destvi_vamp_prior_p": model_cfg.get("destvi_vamp_prior_p"),
            "compat_from_rna_model": True,
            "used_genes": list(map(str, shared_genes)),
            "has_uncertainty": True,
        }
        return BaseSpatialModelOutput(
            abundance=abundance,
            uncertainty=uncertainty,
            metadata=metadata,
        )
