# ReliST

ReliST is a reliability-aware risk layer for spatial transcriptomics deconvolution.
It reads spatial expression data and base-model deconvolution outputs, then produces
spot-level risk scores that indicate where predictions should be trusted, reviewed,
down-weighted, or withheld from downstream interpretation.

This repository is the lightweight public deployment surface for the ReliST codebase.
It intentionally excludes raw data, large processed objects, generated result
directories, figure builders, and dataset-specific validation pipelines.

## Repository Layout

```text
src/st_risk/      Core Python package
scripts/          Minimal command-line entry points and model backend helpers
configs/          Example YAML configs for deployment
tests/            Unit and smoke tests for core package behavior
docs/             Deployment and data-layout notes
```

## Installation

Install the lightweight package and test dependencies:

```bash
pip install -e ".[dev]"
```

Optional deep-learning backends have their own dependency stacks. For `scvi-tools`
based adapters, start with:

```bash
pip install -e ".[scvi]"
```

Model-specific packages such as `cell2location`, Tangram, Stereoscope, and R/RCTD
should be installed following their upstream instructions.

## Core Workflow

1. Prepare ready-to-analyze `.h5ad` inputs.
2. Run a base deconvolution model through a YAML config.
3. Score ReliST risk features from the canonical model output.

Representative entry points:

```bash
PYTHONPATH=src python scripts/run_base_model.py --config configs/example_rctd.yaml
PYTHONPATH=src python scripts/run_risk_scoring.py --config configs/example_rctd.yaml
```

For externally computed abundance tables, use `model.name: precomputed` with
`model.precomputed_abundance_csv` in a config such as `configs/example_precomputed.yaml`.

See `docs/deployment.md` for the deployment workflow.

## Data Policy

Raw datasets and large generated outputs are not stored in this Git repository.
Expected local paths are documented in the YAML configs and in `docs/data_sources.md`.
For shared deployments, document each dataset source and keep large data files outside
the Git repository.

## Tests

Core tests can be run with:

```bash
PYTHONPATH=src pytest
```

Real-data deployment requires spatial transcriptomics inputs and any model-specific
dependencies needed by the selected backend, such as `cell2location`, `scvi-tools`,
Tangram, Stereoscope, or R/RCTD.

## Citation

Citation metadata can be added once a citable release is available.
