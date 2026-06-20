# ReliST v0.2.0 Revision Reproducibility Notes

This document records the code and source-data surface for the iScience revision
release. The revision analyses were run from the project root with repository
relative paths under `data/` and `results/`.

## Scope

The `v0.2.0-revision` release adds:

- donor-disjoint known-composition pseudo-spot validation;
- RCTD and Tangram true-error validation on the same pseudo-spot benchmark;
- paired bootstrap confidence intervals for AUROC, AURC, Spearman correlation,
  filtering deltas, and baseline deltas;
- native or model-derived uncertainty baselines where available;
- cross-model disagreement and incremental regression checks;
- shuffled-coordinate local null controls;
- random and oracle filtering reference curves;
- component ablation and reference-perturbation stress tests.

The release does not claim that the equal-weight full risk score dominates every
baseline in every setting. The revision benchmark supports a direct but modest
and model-dependent association between ReliST-derived risk and true
deconvolution error.

## Selected Runs

The selected revision runs are listed in
`revision/v0.2.0-revision/selected_runs.csv`.

The source-data tables used by the revised manuscript are copied to
`revision/v0.2.0-revision/source_tables/`. These are small reviewer-facing CSV
summaries, not the full generated result directories.

## Large Artifacts

Large generated files are excluded from Git by design:

- `results/revision_known_composition_benchmark/.../artifacts/known_composition_train_reference.h5ad`
- `results/revision_known_composition_benchmark/.../artifacts/known_composition_pseudo_visium.h5ad`
- generated figure PDF/PNG previews
- full per-spot intermediate result tables when they are not needed as source
  data for the revised manuscript

These artifacts can be regenerated from the scripts below after placing the
public input datasets in the expected `data/` layout.

## Reproduction Commands

Build the donor-disjoint pseudo-spot benchmark:

```bash
PYTHONPATH=src python scripts/run_revision_known_composition_benchmark.py \
  --split-mode donor_disjoint \
  --split-column BrNum \
  --run-id 2026-06-20-dlpfc-known-composition-v2-donor-disjoint
```

Run representative base deconvolution models:

```bash
PYTHONPATH=src python scripts/run_base_model.py --config configs/revision_known_composition_rctd.yaml
PYTHONPATH=src python scripts/run_base_model.py --config configs/revision_known_composition_tangram.yaml
```

Evaluate NNLS, RCTD, and Tangram outputs together:

```bash
PYTHONPATH=src python scripts/run_revision_known_composition_multimodel_eval.py \
  --model-run rctd:results/revision_known_composition_rctd \
  --model-run tangram:results/revision_known_composition_tangram \
  --run-id 2026-06-20-dlpfc-known-composition-multimodel-v2-rctd-tangram
```

Run supporting revision analyses:

```bash
PYTHONPATH=src python scripts/run_revision_uncertainty_baseline_eval.py
PYTHONPATH=src python scripts/run_revision_component_ablation.py
PYTHONPATH=src python scripts/run_revision_threshold_sensitivity.py
PYTHONPATH=src python scripts/run_revision_reference_perturbation.py
PYTHONPATH=src python scripts/run_revision_reference_perturbation_component_ablation.py
PYTHONPATH=src python scripts/build_revision_figures.py
```

The `configs/revision_known_composition_cell2location.yaml` file is provided as
a compatible configuration template, but the iScience revision benchmark used
RCTD and Tangram as the two representative additional base methods.

## Manuscript Boundary

The known-composition benchmark is a controlled validation of risk-error
association. It should not be described as natural tissue spot-level ground
truth. Thresholds are illustrative review budgets selected from coverage-risk
curves, not universal cutoffs across tissues, platforms, or base models.

The reference component should be interpreted by scenario. In the matched
reference benchmark it was near-neutral or unfavorable relative to
local-plus-ambiguity variants; therefore it should not be described as a
universal average-performance improvement.
