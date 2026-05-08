from st_risk.paths import project_root


def test_project_root_contains_readme():
    assert (project_root() / "README.md").exists()
