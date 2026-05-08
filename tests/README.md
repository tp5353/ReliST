# Tests

Run the core test suite with:

```bash
PYTHONPATH=src pytest
```

The tests focus on core package behavior: config loading, data validation helpers,
risk features, neighborhood logic, scoring, model interfaces, evaluation summaries,
and reporting helpers.

End-to-end deployment on real datasets requires the external `.h5ad` inputs and
model-specific environments described in `docs/deployment.md`.
