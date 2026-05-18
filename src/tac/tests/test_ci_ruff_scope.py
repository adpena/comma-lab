import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_ci_blocking_ruff_f821_covers_tools_tree() -> None:
    workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert ".venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa" in workflow
    ruff_f821_line = workflow.split(
        ".venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa", 1
    )[1].splitlines()[0]
    assert "tools/" in ruff_f821_line
    assert "--exclude experiments/archive --exclude experiments/results" in ruff_f821_line


def test_ruff_excludes_generated_experiment_artifacts() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "experiments/results" in pyproject
    assert "force-exclude = true" in pyproject


def test_project_ruff_excludes_giant_runtime_style_noise_without_weakening_f821() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "submissions/robust_current/inflate_renderer.py" in pyproject

    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "ruff"),
            "check",
            "--force-exclude",
            "--select",
            "RUF100",
            "src/tac/tests/test_ci_ruff_scope.py",
            "submissions/robust_current/inflate_renderer.py",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Invalid `# noqa` directive" not in result.stdout
    assert "Invalid `# noqa` directive" not in result.stderr


def test_ruff_math_notation_is_ignored_without_weakening_f821() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"RUF001"' in pyproject
    assert '"RUF002"' in pyproject
    assert '"RUF003"' in pyproject
    assert '"F821"' not in pyproject.split("ignore = [", 1)[1].split("]", 1)[0]


def test_ruff_externalizes_legacy_wps_noqa_markers() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'external = [\n    "WPS"' in pyproject


def test_preflight_hook_forces_ruff_excludes_for_explicit_staged_paths() -> None:
    hook = (REPO_ROOT / "tools/preflight_hook.py").read_text(encoding="utf-8")

    assert '"--isolated"' in hook
    assert '"--force-exclude"' in hook
    assert '"--select",\n                "F821"' in hook
    assert '"--ignore-noqa"' in hook
    assert '"experiments/archive"' in hook
    assert '"experiments/results"' in hook


def _run_isolated_f821_probe(*, stdin_filename: str, source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "ruff"),
            "check",
            "--isolated",
            "--force-exclude",
            "--select",
            "F821",
            "--ignore-noqa",
            "--exclude",
            "experiments/archive",
            "--exclude",
            "experiments/results",
            "--stdin-filename",
            stdin_filename,
            "-",
        ],
        cwd=REPO_ROOT,
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )


def test_blocking_f821_probe_ignores_project_per_file_style_carveouts() -> None:
    result = _run_isolated_f821_probe(
        stdin_filename="src/tac/preflight.py",
        source="def f():\n    return missing_name\n",
    )

    assert result.returncode == 1
    assert "F821" in result.stdout
    assert "missing_name" in result.stdout


def test_blocking_f821_probe_still_force_excludes_generated_artifacts() -> None:
    result = _run_isolated_f821_probe(
        stdin_filename="experiments/results/generated.py",
        source="def f():\n    return missing_name\n",
    )

    assert result.returncode == 0


def test_blocking_f821_probe_still_covers_contest_runtime() -> None:
    result = _run_isolated_f821_probe(
        stdin_filename="submissions/robust_current/inflate_renderer.py",
        source="def f():\n    return missing_name\n",
    )

    assert result.returncode == 1
    assert "F821" in result.stdout
    assert "missing_name" in result.stdout
