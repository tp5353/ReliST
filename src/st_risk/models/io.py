from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from st_risk.models.base import BaseSpatialModelOutput
from st_risk.paths import current_results_dir, ensure_results_layout, resolve_results_file, results_file

CANONICAL_ABUNDANCE_FILENAME = "base_model_abundance_means.csv"
CANONICAL_UNCERTAINTY_FILENAME = "base_model_abundance_stds.csv"
CANONICAL_METADATA_FILENAME = "base_model_metadata.json"
CANONICAL_USED_GENES_FILENAME = "base_model_used_genes.csv"


def save_base_model_output(
    output: BaseSpatialModelOutput,
    results_dir: str | Path,
    *,
    extra_metadata: dict | None = None,
) -> None:
    results_path = current_results_dir(results_dir)
    ensure_results_layout(results_path)
    output.abundance.to_csv(results_file(results_path, "tables", CANONICAL_ABUNDANCE_FILENAME))
    if output.uncertainty is not None:
        if isinstance(output.uncertainty, pd.Series):
            output.uncertainty.to_frame(name=output.uncertainty.name or "uncertainty").to_csv(
                results_file(results_path, "tables", CANONICAL_UNCERTAINTY_FILENAME)
            )
        else:
            output.uncertainty.to_csv(results_file(results_path, "tables", CANONICAL_UNCERTAINTY_FILENAME))

    metadata = dict(output.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    with results_file(results_path, "metadata", CANONICAL_METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if "used_genes" in metadata:
        pd.Series(metadata["used_genes"], name="gene").to_csv(
            results_file(results_path, "tables", CANONICAL_USED_GENES_FILENAME),
            index=False,
        )


def load_saved_base_model_output(results_dir: str | Path) -> BaseSpatialModelOutput:
    results_path = current_results_dir(results_dir)
    abundance_path = resolve_results_file(results_path, "tables", CANONICAL_ABUNDANCE_FILENAME)
    if abundance_path.exists():
        abundance = pd.read_csv(abundance_path, index_col=0)
        abundance.columns = [re.sub(r"^meanscell_abundance_w_sf_means_per_cluster_mu_fg_", "", str(col)) for col in abundance.columns]
        uncertainty_path = resolve_results_file(results_path, "tables", CANONICAL_UNCERTAINTY_FILENAME)
        uncertainty = None
        if uncertainty_path.exists():
            uncertainty = pd.read_csv(uncertainty_path, index_col=0)
            uncertainty.columns = [
                re.sub(r"^stdscell_abundance_w_sf_means_per_cluster_mu_fg_", "", str(col)) for col in uncertainty.columns
            ]
            if uncertainty.shape[1] == 1:
                uncertainty = uncertainty.iloc[:, 0]
                uncertainty.name = str(uncertainty.name)

        metadata = {}
        metadata_path = resolve_results_file(results_path, "metadata", CANONICAL_METADATA_FILENAME)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        return BaseSpatialModelOutput(abundance=abundance, uncertainty=uncertainty, metadata=metadata)

    legacy_abundance_path = resolve_results_file(results_path, "tables", "cell2location_abundance_means.csv")
    if not legacy_abundance_path.exists():
        raise FileNotFoundError(f"No canonical or legacy base model abundance table found under {results_path}")

    abundance = pd.read_csv(legacy_abundance_path, index_col=0)
    abundance.columns = [re.sub(r"^meanscell_abundance_w_sf_means_per_cluster_mu_fg_", "", str(col)) for col in abundance.columns]

    legacy_uncertainty_path = resolve_results_file(results_path, "tables", "cell2location_abundance_stds.csv")
    uncertainty = None
    if legacy_uncertainty_path.exists():
        uncertainty = pd.read_csv(legacy_uncertainty_path, index_col=0)
        uncertainty.columns = [re.sub(r"^stdscell_abundance_w_sf_means_per_cluster_mu_fg_", "", str(col)) for col in uncertainty.columns]

    metadata = {}
    for metadata_name in (CANONICAL_METADATA_FILENAME, "cell2location_metadata.json"):
        metadata_path = resolve_results_file(results_path, "metadata", metadata_name)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            break

    metadata.setdefault("model_name", "cell2location")
    return BaseSpatialModelOutput(abundance=abundance, uncertainty=uncertainty, metadata=metadata)
