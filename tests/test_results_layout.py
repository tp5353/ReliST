from st_risk.paths import (
    current_results_dir,
    ensure_results_layout,
    resolve_results_file,
    results_file,
    run_dir,
    set_selected_run,
)


def test_results_file_creates_nested_layout(tmp_path):
    base = ensure_results_layout(tmp_path / "run")
    target = results_file(base, "tables", "risk_table.csv")

    assert target == base / "tables" / "risk_table.csv"
    assert (base / "tables").exists()
    assert (base / "figures").exists()
    assert (base / "metadata").exists()
    assert (base / "artifacts").exists()


def test_resolve_results_file_prefers_nested_then_legacy(tmp_path):
    base = ensure_results_layout(tmp_path / "run")
    legacy = base / "risk_table.csv"
    legacy.write_text("legacy", encoding="utf-8")
    assert resolve_results_file(base, "tables", "risk_table.csv") == legacy

    nested = base / "tables" / "risk_table.csv"
    nested.write_text("nested", encoding="utf-8")
    assert resolve_results_file(base, "tables", "risk_table.csv") == nested


def test_current_results_dir_uses_selected_run_when_present(tmp_path):
    experiment = tmp_path / "dlpfc_mvp"
    target = current_results_dir(experiment, run_id="2026-04-19-dlpfc-mvp-v1", create=True)

    assert target == run_dir(experiment, "2026-04-19-dlpfc-mvp-v1")
    assert current_results_dir(experiment) == target


def test_current_results_dir_falls_back_to_experiment_root_without_selected_run(tmp_path):
    experiment = tmp_path / "dlpfc_mvp"
    experiment.mkdir(parents=True, exist_ok=True)

    assert current_results_dir(experiment) == experiment


def test_set_selected_run_switches_current_run(tmp_path):
    experiment = tmp_path / "dlpfc_mvp"
    run_dir(experiment, "v1").mkdir(parents=True, exist_ok=True)
    run_dir(experiment, "v2").mkdir(parents=True, exist_ok=True)
    set_selected_run(experiment, "v2")

    assert current_results_dir(experiment) == run_dir(experiment, "v2")
