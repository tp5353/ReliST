from __future__ import annotations

import argparse
import sys
from pathlib import Path

from st_risk.data.validate import validate_ready_directory
from st_risk.paths import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a prepared ready-data directory.")
    parser.add_argument(
        "--ready-dir",
        type=Path,
        default=project_root() / "data" / "example" / "ready",
        help="Path to the ready-data directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_ready_directory(args.ready_dir)
    if errors:
        print("READY DATA VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("READY DATA VALIDATION PASSED")
    print(f"Ready directory: {args.ready_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
