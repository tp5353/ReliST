from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


RESULTS_LAYOUT = {
    "tables": "tables",
    "figures": "figures",
    "metadata": "metadata",
    "artifacts": "artifacts",
}


def results_dir(path: str | Path) -> Path:
    return Path(path)


def selected_run_file(path: str | Path) -> Path:
    return results_dir(path) / "selected_run.txt"


def runs_dir(path: str | Path) -> Path:
    return results_dir(path) / "runs"


def get_selected_run(path: str | Path) -> str | None:
    pointer = selected_run_file(path)
    if not pointer.exists():
        return None
    value = pointer.read_text(encoding="utf-8").strip()
    return value or None


def set_selected_run(path: str | Path, run_id: str) -> None:
    base = results_dir(path)
    base.mkdir(parents=True, exist_ok=True)
    selected_run_file(base).write_text(f"{run_id}\n", encoding="utf-8")


def run_dir(path: str | Path, run_id: str) -> Path:
    return runs_dir(path) / run_id


def current_results_dir(path: str | Path, *, run_id: str | None = None, create: bool = False) -> Path:
    base = results_dir(path)
    if run_id is not None:
        target = run_dir(base, run_id)
        if create:
            target.mkdir(parents=True, exist_ok=True)
            set_selected_run(base, run_id)
        return target

    selected = get_selected_run(base)
    if selected is not None:
        target = run_dir(base, selected)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def results_subdir(path: str | Path, category: str) -> Path:
    base = results_dir(path)
    if category not in RESULTS_LAYOUT:
        raise KeyError(f"Unknown results category: {category}")
    return base / RESULTS_LAYOUT[category]


def ensure_results_layout(path: str | Path) -> Path:
    base = results_dir(path)
    base.mkdir(parents=True, exist_ok=True)
    for category in RESULTS_LAYOUT:
        results_subdir(base, category).mkdir(parents=True, exist_ok=True)
    return base


def results_file(path: str | Path, category: str, filename: str) -> Path:
    ensure_results_layout(path)
    return results_subdir(path, category) / filename


def resolve_results_file(path: str | Path, category: str, filename: str) -> Path:
    nested = results_subdir(path, category) / filename
    if nested.exists():
        return nested
    legacy = results_dir(path) / filename
    return legacy
