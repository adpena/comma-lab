from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ci_blocking_ruff_f821_covers_tools_tree() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert ".venv/bin/ruff check --force-exclude --select F821" in workflow
    ruff_f821_line = workflow.split(".venv/bin/ruff check --force-exclude --select F821", 1)[1].splitlines()[0]
    assert "tools/" in ruff_f821_line


def test_ruff_excludes_generated_experiment_artifacts() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "experiments/results" in pyproject
