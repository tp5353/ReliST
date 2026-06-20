from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT / "results" / "revision_known_composition_benchmark"
REPEAT_ROOT = ROOT / "results" / "revision_known_composition_seed_repeats"
MANUSCRIPT_TABLE_DIR = ROOT / "manuscript" / "iscience" / "tables"
DEFAULT_RUN_ID = "2026-06-20-dlpfc-known-composition-seed-repeat-summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize donor-disjoint known-composition simulation seed repeats."
    )
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    parser.add_argument("--repeat-root", type=Path, default=REPEAT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--manuscript-table",
        type=Path,
        default=MANUSCRIPT_TABLE_DIR / "table_sx_known_composition_seed_repeat_summary.csv",
    )
    return parser.parse_args()


def selected_run(root: Path) -> Path:
    selected = (root / "selected_run.txt").read_text(encoding="utf-8").strip()
    return root / "runs" / selected


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def metric_value(rows: list[dict[str, str]], score_name: str, column: str) -> float:
    for row in rows:
        if row.get("score_name") == score_name:
            return float(row[column])
    raise KeyError(f"Missing {score_name} in rows")


def keep80_value(rows: list[dict[str, str]], score_name: str, column: str) -> float:
    for row in rows:
        if row.get("score_name") == score_name and abs(float(row.get("keep_fraction", "nan")) - 0.8) < 1e-9:
            return float(row[column])
    raise KeyError(f"Missing keep_fraction=0.8 for {score_name}")


def collect_run(run_dir: Path, source: str) -> dict[str, object]:
    metadata = json.loads((run_dir / "metadata" / "known_composition_benchmark.json").read_text(encoding="utf-8"))
    summary = read_csv(run_dir / "tables" / "known_composition_score_error_summary.csv")
    selective = read_csv(run_dir / "tables" / "known_composition_selective_error_curve.csv")
    scenario = read_csv(run_dir / "tables" / "known_composition_scenario_summary.csv")
    risk = "risk_score"
    full_mean_error = keep80_value(selective, risk, "full_mean_error")
    keep80_mean_error = keep80_value(selective, risk, "mean_error")
    scenario_counts = {}
    for row in scenario:
        scenario_counts[row["scenario"]] = int(float(row["n_spots"]))
    return {
        "source": source,
        "run_id": run_dir.name,
        "random_state": metadata["random_state"],
        "split_mode": metadata["split"]["split_mode"],
        "split_column": metadata["split"].get("split_column", ""),
        "n_train_units": metadata["split"].get("n_train_units", ""),
        "n_simulate_units": metadata["split"].get("n_simulate_units", ""),
        "n_spots": metadata["n_spots"],
        "n_celltypes": metadata["n_celltypes"],
        "cells_per_spot_min": metadata["cells_per_spot_min"],
        "cells_per_spot_max": metadata["cells_per_spot_max"],
        "spearman_error": metric_value(summary, risk, "spearman_error"),
        "auroc_top20_error": metric_value(summary, risk, "auroc_top20_error"),
        "top_minus_bottom20_error": metric_value(summary, risk, "top_minus_bottom20_error"),
        "keep80_mean_error": keep80_mean_error,
        "full_mean_error": full_mean_error,
        "keep80_error_reduction_vs_full": full_mean_error - keep80_mean_error,
        "clean_n_spots": scenario_counts.get("clean", 0),
        "low_depth_n_spots": scenario_counts.get("low_depth", 0),
        "marker_dropout_n_spots": scenario_counts.get("marker_dropout", 0),
        "diffuse_mixture_n_spots": scenario_counts.get("diffuse_mixture", 0),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def ci95_t(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    # t critical values for two-sided 95% intervals for df 1-10; df=2 for the default three repeats.
    t_critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(values) - 1, 1.96)
    half_width = t_critical * sample_sd(values) / math.sqrt(len(values))
    return mean(values) - half_width, mean(values) + half_width


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "spearman_error",
        "auroc_top20_error",
        "top_minus_bottom20_error",
        "keep80_mean_error",
        "keep80_error_reduction_vs_full",
    ]
    output = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        low, high = ci95_t(values)
        output.append(
            {
                "metric": metric,
                "n_independent_simulation_seeds": len(values),
                "mean": mean(values),
                "ci95_low": low,
                "ci95_high": high,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    run_dir = args.repeat_root / "runs" / args.run_id
    tables_dir = run_dir / "tables"
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = [collect_run(selected_run(args.main_root), "primary_selected_run")]
    repeat_runs = [
        path for path in sorted((args.repeat_root / "runs").iterdir()) if path.is_dir() and path.name != args.run_id
    ]
    rows.extend(collect_run(path, "independent_seed_repeat") for path in repeat_runs)

    per_seed_fields = [
        "source",
        "run_id",
        "random_state",
        "split_mode",
        "split_column",
        "n_train_units",
        "n_simulate_units",
        "n_spots",
        "n_celltypes",
        "cells_per_spot_min",
        "cells_per_spot_max",
        "spearman_error",
        "auroc_top20_error",
        "top_minus_bottom20_error",
        "keep80_mean_error",
        "full_mean_error",
        "keep80_error_reduction_vs_full",
        "clean_n_spots",
        "low_depth_n_spots",
        "marker_dropout_n_spots",
        "diffuse_mixture_n_spots",
    ]
    aggregate_fields = ["metric", "n_independent_simulation_seeds", "mean", "ci95_low", "ci95_high", "min", "max"]
    aggregate_rows = aggregate(rows)

    per_seed_path = tables_dir / "known_composition_seed_repeat_per_seed_summary.csv"
    aggregate_path = tables_dir / "known_composition_seed_repeat_metric_summary.csv"
    write_csv(per_seed_path, rows, per_seed_fields)
    write_csv(aggregate_path, aggregate_rows, aggregate_fields)

    manuscript_rows: list[dict[str, object]] = []
    for row in rows:
        manuscript_rows.append({"row_type": "per_seed", **row})
    for row in aggregate_rows:
        manuscript_rows.append({"row_type": "aggregate_metric", **row})
    manuscript_fields = ["row_type"] + per_seed_fields + aggregate_fields
    write_csv(args.manuscript_table, manuscript_rows, manuscript_fields)

    (args.repeat_root / "selected_run.txt").write_text(args.run_id + "\n", encoding="utf-8")
    report = [
        "# Known-Composition Seed Repeat Summary",
        "",
        "This run summarizes the primary donor-disjoint known-composition benchmark and two independent simulation seeds.",
        "It is a lightweight repeatability check for the NNLS pseudo-spot projection; the RCTD and Tangram true-error analyses remain the main multi-base-model validation.",
        "",
        f"- per-seed table: `{per_seed_path}`",
        f"- aggregate table: `{aggregate_path}`",
        f"- manuscript table: `{args.manuscript_table}`",
    ]
    (run_dir / "known_composition_seed_repeat_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {run_dir}")
    print(f"Wrote {args.manuscript_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
