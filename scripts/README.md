# Scripts

This directory intentionally contains only lightweight deployment entry points and
backend helpers.

## Runtime Entry Points

- `run_base_model.py`: run one configured base model and save canonical outputs.
- `run_risk_scoring.py`: compute ReliST risk scores from a configured run.
- `validate_ready_data.py`: validate a prepared ready-data directory.

## Backend Helpers

- `run_rctd_native.R`: native RCTD helper used by the RCTD adapter.
- `run_stereoscope_native.py`: native Stereoscope helper used by the Stereoscope adapter.
- `export_reference_signatures.py`: export reference signatures for reference-aware risk features.

Figure generation, modality-specific validation, and dataset-specific data preparation
scripts are not part of this deployment package.
