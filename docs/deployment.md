# Deployment Guide

This repository is a lightweight deployment package for ReliST. It keeps the runtime
package, base-model adapters, minimal command-line entry points, example configs, and
tests. Large datasets, generated outputs, dataset-specific analysis utilities, and
figure-building pipelines are intentionally excluded.

## Expected Inputs

ReliST expects ready-to-analyze `.h5ad` files:

- a spatial transcriptomics object with spot coordinates and counts
- a scRNA/snRNA reference object with cell-type labels
- optional precomputed base-model abundance and uncertainty-like tables

The repository does not include raw data or generated results.

## Minimal Workflow

Install the package:

```bash
pip install -e ".[dev]"
```

Install optional model backends separately. For `scvi-tools` based backends:

```bash
pip install -e ".[scvi]"
```

1. Validate a prepared data directory:

   ```bash
   PYTHONPATH=src python scripts/validate_ready_data.py --ready-dir data/example/ready
   ```

2. Run a configured base model:

   ```bash
   PYTHONPATH=src python scripts/run_base_model.py --config configs/example_rctd.yaml
   ```

   If abundance estimates were produced outside this package, use
   `configs/example_precomputed.yaml` and set `model.precomputed_abundance_csv`.

3. Score ReliST risk:

   ```bash
   PYTHONPATH=src python scripts/run_risk_scoring.py --config configs/example_rctd.yaml
   ```

## Optional Backend Helpers

- `scripts/run_rctd_native.R`: native RCTD backend helper.
- `scripts/run_stereoscope_native.py`: native Stereoscope backend helper.
- `scripts/export_reference_signatures.py`: export reference signatures for reference-aware risk features.

Backend packages such as `cell2location`, Tangram, Stereoscope, and R/RCTD are not
vendored by this repository and should be installed from their upstream projects.

## What Is Excluded

This deployment package excludes:

- figure generation pipelines
- modality-specific validation scripts
- dataset-specific ready-data builders
- annotation review utilities
- large `data/` and `results/` directories

Those materials should live in a separate analysis archive when needed.
