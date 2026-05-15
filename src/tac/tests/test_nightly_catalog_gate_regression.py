# SPDX-License-Identifier: MIT
"""Tests for the canonical nightly catalog-gate regression CI helper + gate.

Catalog #284 (PHASE-3-CONTINUOUS-REGRESSION-CI-NIGHTLY-WORKFLOW 2026-05-15).
Sister of Catalog #245 ledger-discipline pattern.

Covers:
- Helper script invocations (run_preflight / run_pytest /
  run_meta_meta_drift_checks)
- Status JSON schema (required fields + types)
- HISTORICAL_PROVENANCE write discipline (per-day file, atomic write)
- Workflow YAML syntactically valid (yaml.safe_load(workflow_file))
- Workflow YAML has required schedule + steps
- Drift-detection branching (preflight fail -> exit 1; preflight pass +
  pytest pass -> exit 0)
- STRICT preflight gate Catalog #284 refuses workflow modifications outside
  the canonical helper
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Canonical helper: import via the file path (it lives under tools/, not
# under tac.*). The file is fully self-contained.
# ---------------------------------------------------------------------------


def _import_helper():
    import importlib.util

    helper_path = REPO_ROOT / "tools" / "run_nightly_catalog_gate_regression.py"
    spec = importlib.util.spec_from_file_location(
        "run_nightly_catalog_gate_regression_test_module", helper_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper schema / parsing tests
# ---------------------------------------------------------------------------


def test_helper_schema_version_pinned():
    mod = _import_helper()
    assert mod._STATUS_SCHEMA_VERSION == "nightly_catalog_gate_regression_status_v1"


def test_helper_meta_meta_gate_roster_canonical():
    mod = _import_helper()
    expected = {
        "check_claude_md_catalog_no_duplicate_numbers",
        "check_claude_md_catalog_text_matches_preflight_strict_value",
        "check_strict_preflight_callsites_have_claude_md_catalog_row",
        "check_strict_flipped_catalog_entries_have_live_count_zero",
    }
    assert set(mod._META_META_GATE_NAMES) == expected


def test_helper_state_dir_canonical_path():
    mod = _import_helper()
    assert str(mod._NIGHTLY_STATE_DIR_RELATIVE) == ".omx/state"


def test_helper_resolve_run_id_uses_gha_env(monkeypatch):
    mod = _import_helper()
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    assert mod._resolve_run_id(None) == "gha-12345"


def test_helper_resolve_run_id_uses_override(monkeypatch):
    mod = _import_helper()
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    assert mod._resolve_run_id("custom-id") == "custom-id"


def test_helper_resolve_run_id_local_fallback(monkeypatch):
    mod = _import_helper()
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    rid = mod._resolve_run_id(None)
    assert rid.startswith("local-")


def test_parse_pytest_summary_passed():
    mod = _import_helper()
    stdout = "==== 5 passed in 1.23s ===="
    counts = mod._parse_pytest_summary(stdout)
    assert counts["passed"] == 5
    assert counts["failed"] == 0
    assert counts["skipped"] == 0


def test_parse_pytest_summary_mixed():
    mod = _import_helper()
    stdout = "==== 3 failed, 5 passed, 2 skipped in 1.23s ===="
    counts = mod._parse_pytest_summary(stdout)
    assert counts["failed"] == 3
    assert counts["passed"] == 5
    assert counts["skipped"] == 2


def test_parse_pytest_summary_with_errors():
    mod = _import_helper()
    stdout = "==== 1 failed, 2 passed, 1 error in 0.5s ===="
    counts = mod._parse_pytest_summary(stdout)
    assert counts["failed"] == 1
    assert counts["passed"] == 2
    assert counts["errors"] == 1


def test_parse_pytest_summary_empty_stdout():
    mod = _import_helper()
    counts = mod._parse_pytest_summary("")
    assert counts == {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "warnings": 0,
    }


# ---------------------------------------------------------------------------
# Status JSON write tests (HISTORICAL_PROVENANCE per Catalog #110/#113)
# ---------------------------------------------------------------------------


def test_write_status_json_creates_per_day_file(tmp_path: Path):
    mod = _import_helper()
    out = mod.write_status_json(
        tmp_path,
        run_id="test-run-1",
        timestamp_utc="2026-05-15T22:00:00Z",
        preflight_passed=True,
        preflight_summary={"rc": 0, "elapsed_seconds": 1.0, "stdout_tail": "", "stderr_tail": ""},
        pytest_skipped=False,
        pytest_passed=True,
        pytest_summary={
            "rc": 0,
            "elapsed_seconds": 2.0,
            "counts": {"passed": 5, "failed": 0, "skipped": 0, "errors": 0},
            "stdout_tail": "",
            "stderr_tail": "",
        },
        meta_meta_no_drift=True,
        meta_meta_summary={"no_drift": True, "per_gate": {}},
    )
    assert out.parent == (tmp_path / ".omx" / "state").resolve()
    assert out.name.startswith("nightly_catalog_gate_regression_")
    assert out.name.endswith(".json")
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == "nightly_catalog_gate_regression_status_v1"
    assert payload["run_id"] == "test-run-1"
    assert payload["overall_pass"] is True


def test_write_status_json_failure_payload(tmp_path: Path):
    mod = _import_helper()
    out = mod.write_status_json(
        tmp_path,
        run_id="test-run-2",
        timestamp_utc="2026-05-15T22:00:00Z",
        preflight_passed=False,
        preflight_summary={"rc": 1, "elapsed_seconds": 1.0, "stdout_tail": "", "stderr_tail": "boom"},
        pytest_skipped=False,
        pytest_passed=False,
        pytest_summary={
            "rc": 1,
            "elapsed_seconds": 2.0,
            "counts": {"passed": 3, "failed": 2, "skipped": 0, "errors": 0},
            "stdout_tail": "",
            "stderr_tail": "",
        },
        meta_meta_no_drift=False,
        meta_meta_summary={"no_drift": False, "per_gate": {}},
    )
    payload = json.loads(out.read_text())
    assert payload["preflight_pass"] is False
    assert payload["pytest_pass"] is False
    assert payload["meta_meta_drift"] is True
    assert payload["overall_pass"] is False
    assert payload["passed"] == 3
    assert payload["failed"] == 2


def test_write_status_json_pytest_skipped(tmp_path: Path):
    mod = _import_helper()
    out = mod.write_status_json(
        tmp_path,
        run_id="test-run-3",
        timestamp_utc="2026-05-15T22:00:00Z",
        preflight_passed=True,
        preflight_summary={"rc": 0, "elapsed_seconds": 1.0, "stdout_tail": "", "stderr_tail": ""},
        pytest_skipped=True,
        pytest_passed=None,
        pytest_summary=None,
        meta_meta_no_drift=True,
        meta_meta_summary={"no_drift": True, "per_gate": {}},
    )
    payload = json.loads(out.read_text())
    assert payload["pytest_skipped"] is True
    assert payload["pytest_pass"] is None
    assert payload["pytest_summary"] is None
    # Pytest skip + preflight pass + no drift = overall_pass True.
    assert payload["overall_pass"] is True


def test_write_status_json_atomic_no_tmp_leak(tmp_path: Path):
    mod = _import_helper()
    mod.write_status_json(
        tmp_path,
        run_id="atomic-test",
        timestamp_utc="2026-05-15T22:00:00Z",
        preflight_passed=True,
        preflight_summary={"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""},
        pytest_skipped=True,
        pytest_passed=None,
        pytest_summary=None,
        meta_meta_no_drift=True,
        meta_meta_summary={"no_drift": True, "per_gate": {}},
    )
    state_dir = tmp_path / ".omx" / "state"
    leftovers = list(state_dir.glob("*.tmp.*"))
    assert leftovers == []


def test_write_status_json_idempotent_same_day_overwrite(tmp_path: Path):
    mod = _import_helper()
    out1 = mod.write_status_json(
        tmp_path,
        run_id="run-A",
        timestamp_utc="2026-05-15T01:00:00Z",
        preflight_passed=True,
        preflight_summary={"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""},
        pytest_skipped=True,
        pytest_passed=None,
        pytest_summary=None,
        meta_meta_no_drift=True,
        meta_meta_summary={"no_drift": True, "per_gate": {}},
    )
    out2 = mod.write_status_json(
        tmp_path,
        run_id="run-B",
        timestamp_utc="2026-05-15T05:00:00Z",
        preflight_passed=False,
        preflight_summary={"rc": 1, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""},
        pytest_skipped=True,
        pytest_passed=None,
        pytest_summary=None,
        meta_meta_no_drift=True,
        meta_meta_summary={"no_drift": True, "per_gate": {}},
    )
    # Same UTC day -> same file path; second write replaces first.
    assert out1 == out2
    payload = json.loads(out2.read_text())
    assert payload["run_id"] == "run-B"


# ---------------------------------------------------------------------------
# META-meta drift gate invocation tests
# ---------------------------------------------------------------------------


def test_run_meta_meta_drift_checks_returns_per_gate_dict():
    mod = _import_helper()
    no_drift, summary = mod.run_meta_meta_drift_checks(REPO_ROOT)
    assert isinstance(summary, dict)
    assert "per_gate" in summary
    assert set(summary["per_gate"].keys()) == set(mod._META_META_GATE_NAMES)
    # Every gate must report a status field.
    for name, detail in summary["per_gate"].items():
        assert "status" in detail, f"gate {name} missing status"
        assert detail["status"] in {"ok", "drift", "missing", "error"}


# ---------------------------------------------------------------------------
# Workflow YAML schema tests
# ---------------------------------------------------------------------------


def _read_workflow_text() -> str:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "nightly_catalog_gate_regression.yml"
    assert workflow_path.is_file(), f"workflow missing: {workflow_path}"
    return workflow_path.read_text(encoding="utf-8")


def test_workflow_yaml_parses():
    yaml = pytest.importorskip("yaml")
    text = _read_workflow_text()
    doc = yaml.safe_load(text)
    assert isinstance(doc, dict)


def test_workflow_yaml_has_schedule():
    yaml = pytest.importorskip("yaml")
    text = _read_workflow_text()
    doc = yaml.safe_load(text)
    # YAML 1.1 may parse the bare key `on` as bool True; accept both.
    on = doc.get("on") or doc.get(True)
    assert on is not None, "workflow missing 'on' trigger block"
    assert "schedule" in on
    schedules = on["schedule"]
    assert any(entry.get("cron") == "0 2 * * *" for entry in schedules)


def test_workflow_yaml_has_workflow_dispatch():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_read_workflow_text())
    on = doc.get("on") or doc.get(True)
    assert "workflow_dispatch" in on


def test_workflow_yaml_invokes_canonical_helper():
    text = _read_workflow_text()
    assert "tools/run_nightly_catalog_gate_regression.py" in text


def test_workflow_yaml_uploads_status_artifact():
    text = _read_workflow_text()
    assert "actions/upload-artifact" in text
    assert "nightly_catalog_gate_regression_" in text


def test_workflow_yaml_uses_python_3_12():
    text = _read_workflow_text()
    assert "python-version: '3.12'" in text


def test_workflow_yaml_uses_concurrency_no_cancel_in_progress():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_read_workflow_text())
    assert "concurrency" in doc
    assert doc["concurrency"]["cancel-in-progress"] is False


# ---------------------------------------------------------------------------
# main() exit-code / branching tests
# ---------------------------------------------------------------------------


def test_main_exits_zero_when_all_green(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    # Stub all 3 stages to "green".
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (True, {"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )
    monkeypatch.setattr(
        mod,
        "run_pytest",
        lambda root, **kw: (
            True,
            {"rc": 0, "elapsed_seconds": 0.1, "counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0}, "stdout_tail": "", "stderr_tail": "", "timed_out": False},
        ),
    )
    monkeypatch.setattr(
        mod, "run_meta_meta_drift_checks", lambda root: (True, {"no_drift": True, "per_gate": {}})
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-green"])
    assert rc == 0


def test_main_exits_one_when_preflight_fails(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (False, {"rc": 1, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )
    monkeypatch.setattr(
        mod,
        "run_pytest",
        lambda root, **kw: (
            True,
            {"rc": 0, "elapsed_seconds": 0.1, "counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0}, "stdout_tail": "", "stderr_tail": "", "timed_out": False},
        ),
    )
    monkeypatch.setattr(
        mod, "run_meta_meta_drift_checks", lambda root: (True, {"no_drift": True, "per_gate": {}})
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-pre-red", "--strict"])
    assert rc == 1


def test_main_exits_one_when_pytest_fails(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (True, {"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )
    monkeypatch.setattr(
        mod,
        "run_pytest",
        lambda root, **kw: (
            False,
            {"rc": 1, "elapsed_seconds": 0.1, "counts": {"passed": 1, "failed": 1, "skipped": 0, "errors": 0}, "stdout_tail": "", "stderr_tail": "", "timed_out": False},
        ),
    )
    monkeypatch.setattr(
        mod, "run_meta_meta_drift_checks", lambda root: (True, {"no_drift": True, "per_gate": {}})
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-py-red"])
    assert rc == 1


def test_main_exits_one_when_meta_meta_drifts(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (True, {"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )
    monkeypatch.setattr(
        mod,
        "run_pytest",
        lambda root, **kw: (
            True,
            {"rc": 0, "elapsed_seconds": 0.1, "counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0}, "stdout_tail": "", "stderr_tail": "", "timed_out": False},
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_meta_meta_drift_checks",
        lambda root: (False, {"no_drift": False, "per_gate": {"check_claude_md_catalog_no_duplicate_numbers": {"status": "drift", "violations": ["dup #100"]}}}),
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-meta-red"])
    assert rc == 1


def test_main_no_strict_always_zero(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (False, {"rc": 1, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )
    monkeypatch.setattr(
        mod,
        "run_pytest",
        lambda root, **kw: (
            False,
            {"rc": 1, "elapsed_seconds": 0.1, "counts": {"passed": 0, "failed": 1, "skipped": 0, "errors": 0}, "stdout_tail": "", "stderr_tail": "", "timed_out": False},
        ),
    )
    monkeypatch.setattr(
        mod, "run_meta_meta_drift_checks", lambda root: (True, {"no_drift": True, "per_gate": {}})
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-advisory", "--no-strict"])
    assert rc == 0


def test_main_skip_pytest_does_not_invoke_pytest(tmp_path: Path, monkeypatch):
    mod = _import_helper()
    monkeypatch.setattr(
        mod, "run_preflight", lambda root, **kw: (True, {"rc": 0, "elapsed_seconds": 0.1, "stdout_tail": "", "stderr_tail": "", "timed_out": False})
    )

    def _should_not_be_called(root, **kw):
        raise AssertionError("run_pytest must not be called when --skip-pytest is set")

    monkeypatch.setattr(mod, "run_pytest", _should_not_be_called)
    monkeypatch.setattr(
        mod, "run_meta_meta_drift_checks", lambda root: (True, {"no_drift": True, "per_gate": {}})
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--run-id", "test-skip-py", "--skip-pytest"])
    assert rc == 0
    # And the per-day status file MUST exist with pytest_skipped=True.
    state_dir = tmp_path / ".omx" / "state"
    files = list(state_dir.glob("nightly_catalog_gate_regression_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["pytest_skipped"] is True


# ---------------------------------------------------------------------------
# STRICT preflight gate Catalog #284 tests
# ---------------------------------------------------------------------------


def test_check_284_live_repo_passes():
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        strict=False, verbose=False
    )
    assert violations == []


def test_check_284_flags_workflow_missing_canonical_helper(tmp_path: Path):
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / "nightly_catalog_gate_regression.yml").write_text(
        "name: nightly_catalog_gate_regression\non:\n  schedule:\n    - cron: '0 2 * * *'\n"
        "jobs:\n  bad:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - name: do random thing\n        run: echo hi\n",
        encoding="utf-8",
    )
    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "tools/run_nightly_catalog_gate_regression.py" in violations[0]


def test_check_284_accepts_workflow_with_canonical_helper(tmp_path: Path):
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / "nightly_catalog_gate_regression.yml").write_text(
        "name: nightly_catalog_gate_regression\non:\n  schedule:\n    - cron: '0 2 * * *'\n"
        "jobs:\n  good:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: .venv/bin/python tools/run_nightly_catalog_gate_regression.py\n",
        encoding="utf-8",
    )
    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_284_strict_raises_on_violation(tmp_path: Path):
    from tac.preflight import (
        PreflightError,
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / "nightly_catalog_gate_regression.yml").write_text(
        "name: bad\non:\n  schedule:\n    - cron: '0 2 * * *'\n"
        "jobs:\n  bad:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: echo bypass\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="Catalog #284"):
        check_nightly_catalog_gate_regression_workflow_canonical_use(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_284_no_workflow_file_returns_empty(tmp_path: Path):
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    # No .github/ at all -> nothing to check, nothing to refuse.
    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_284_accepts_same_line_waiver(tmp_path: Path):
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / "nightly_catalog_gate_regression.yml").write_text(
        "# NIGHTLY_CATALOG_GATE_REGRESSION_OK:operator-approved-alternate-driver-2026-05-15\n"
        "name: nightly_catalog_gate_regression\non:\n  schedule:\n    - cron: '0 2 * * *'\n"
        "jobs:\n  alt:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: echo alt\n",
        encoding="utf-8",
    )
    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_284_rejects_placeholder_waiver(tmp_path: Path):
    from tac.preflight import (
        check_nightly_catalog_gate_regression_workflow_canonical_use,
    )

    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / "nightly_catalog_gate_regression.yml").write_text(
        "# NIGHTLY_CATALOG_GATE_REGRESSION_OK:<reason>\n"
        "name: nightly_catalog_gate_regression\non:\n  schedule:\n    - cron: '0 2 * * *'\n"
        "jobs:\n  alt:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: echo placeholder\n",
        encoding="utf-8",
    )
    violations = check_nightly_catalog_gate_regression_workflow_canonical_use(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_284_orchestrator_callsite_strict_true():
    """Catalog #284 wired strict=True in preflight_all() per Strict-flip atomicity rule."""
    text = (REPO_ROOT / "src" / "tac" / "preflight.py").read_text(encoding="utf-8")
    assert "check_nightly_catalog_gate_regression_workflow_canonical_use(" in text
    # Must appear in preflight_all with strict=True nearby (within 3 lines).
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "check_nightly_catalog_gate_regression_workflow_canonical_use(" in line:
            window = "\n".join(lines[idx : idx + 4])
            if "strict=True" in window:
                return
    pytest.fail("Catalog #284 callsite missing strict=True wiring in preflight_all()")
