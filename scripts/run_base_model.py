from __future__ import annotations

import argparse
from pathlib import Path

from st_risk.config import load_config
from st_risk.models.io import save_base_model_output
from st_risk.models.registry import build_model_runner
from st_risk.paths import current_results_dir, ensure_results_layout
from st_risk.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured base spatial model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "example_cell2location.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    outputs_cfg = config.get("outputs", {})
    runner = build_model_runner(config.get("model", {}).get("name", "cell2location"))
    output = runner.run(
        dataset_cfg["visium_h5ad"],
        dataset_cfg["snrna_h5ad"],
        config=config,
    )
    results_dir = current_results_dir(
        outputs_cfg.get("results_dir", "results/example_model"),
        run_id=outputs_cfg.get("run_id"),
        create=True,
    )
    ensure_results_layout(results_dir)
    save_base_model_output(output, results_dir)
    print(output.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
