# Configs

YAML configs define dataset inputs, base model parameters, risk settings, output
directories, and selected run identifiers. The configs in this lightweight package are
examples/templates for deployment, not a complete analysis archive.

## Examples

- `example_cell2location.yaml`: cell2location-style example.
- `example_precomputed.yaml`: precomputed abundance table example.
- `example_rctd.yaml`: RCTD example.
- `example_tangram.yaml`: Tangram example.
- `example_stereoscope.yaml`: Stereoscope example.
- `example_destvi.yaml`: DestVI example.

## Path Convention

Configs use repository-relative paths such as:

```yaml
dataset:
  visium_h5ad: data/<dataset>/ready/<file>.h5ad
outputs:
  results_dir: results/<experiment>
```

Raw data and generated outputs are intentionally ignored by Git. See
`docs/data_sources.md` before running these configs on a new machine.
