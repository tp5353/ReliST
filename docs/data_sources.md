# Data Layout

This repository does not include raw spatial transcriptomics, scRNA/snRNA reference,
or generated result data.

## Expected Local Layout

The YAML configs expect a local project layout like:

```text
data/
  example_dataset/
    ready/
results/
```

Large files should remain outside Git. Use symlinks or local copies to connect public
datasets to these relative paths.

## Ready Data

Each ready-data directory should contain a manifest and the `.h5ad` objects expected
by the chosen config, for example:

- spatial `.h5ad`
- reference `.h5ad`
- manifest with dataset paths and metadata

Use `scripts/validate_ready_data.py` to check the manifest before running a model.
