# v0.2.0-revision Source Data

This directory contains small source-data tables and run identifiers for the
iScience revision analyses. It is intended to support the frozen GitHub and
Zenodo release, while keeping large data objects and generated figures outside
Git.

- `selected_runs.csv`: selected run identifiers and their role in the revision.
- `run_metadata/`: sanitized selected-run metadata JSON files with local absolute
  path prefixes removed.
- `source_tables/`: reviewer-facing CSV summaries used in Supplementary Tables
  S6-S10 and Figure 7/Supplementary Figures S6-S10.
- `source_tables.sha256`: checksums for the copied source-data CSV files.

Supplementary Table S6 also includes
`table_sx_known_composition_seed_repeat_summary.csv`, a lightweight independent
simulation seed-repeat check for the donor-disjoint NNLS pseudo-spot benchmark.

The full generated result directories remain under local `results/revision_*`
directories and can be regenerated from the scripts documented in
`docs/revision_v0.2.0.md`.
