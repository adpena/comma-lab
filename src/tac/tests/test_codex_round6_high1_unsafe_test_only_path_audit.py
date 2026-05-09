"""Tests for codex round 6 HIGH 1 fix: unsafe_test_only path audit.

Catalog #142 — refuses ``Phase3DispatchGate(unsafe_test_only=True)`` from
non-test code paths. The round-4 fix added ``unsafe_test_only=True`` as
an escape hatch, but any production caller could toggle the kwarg to
silently bypass every Phase 3 precondition. The round-6 fix detects
the calling frame and refuses the escape hatch from non-test paths.

Bug class: codex round 6 HIGH 1 (2026-05-09). Memory:
feedback_codex_round6_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tac.phase3.joint_scorer_renderer_codec import (
    Phase3DispatchGate,
    Phase3DispatchGateError,
    _caller_path_is_test,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# Frame-walker helper unit tests
# --------------------------------------------------------------------------


def test_caller_path_is_test_returns_true_for_this_test_file():
    """When invoked directly from a test path, the helper returns True."""
    is_test, fn = _caller_path_is_test()
    assert is_test is True, (
        f"expected True (this file is a test), got False; frame at {fn}"
    )
    assert "/tests/" in fn or "test_" in Path(fn).name


def test_caller_path_is_test_walks_past_synthesized_frames():
    """The helper must skip ``<string>`` frames (dataclass auto-init)."""
    # Constructing the gate here goes through the dataclass-synthesized
    # __init__ at "<string>" → __post_init__ → _caller_path_is_test.
    # The helper must walk past "<string>" and find this test as the
    # true caller.
    gate = Phase3DispatchGate(unsafe_test_only=True)
    assert gate.unsafe_test_only is True


# --------------------------------------------------------------------------
# unsafe_test_only allowed from test paths
# --------------------------------------------------------------------------


def test_unsafe_test_only_works_from_test_path():
    """Tests legitimately need a permissive gate."""
    gate = Phase3DispatchGate(unsafe_test_only=True)
    assert gate.unsafe_test_only is True
    # Defaults to False so the path-audit waiver is opt-in
    assert gate.unsafe_test_only_path_audit_waived is False


def test_full_precondition_set_works_from_test_path():
    """Production callers passing the full precondition set still work."""
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="/some/evidence.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="/some/distill.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="/some/aaf.json",
        phase3_council_deliberation_path="some_council.md",
    )
    assert gate.phase2_anchor_verified is True


# --------------------------------------------------------------------------
# unsafe_test_only REFUSED from non-test paths (the regression)
# --------------------------------------------------------------------------


def _run_python_snippet(snippet: str) -> tuple[int, str, str]:
    """Run a python snippet in a fresh subprocess; return (rc, stdout, stderr)."""
    py = sys.executable
    env = os.environ.copy()
    # Ensure the repo's src is on PYTHONPATH so `tac.phase3` imports.
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    proc = subprocess.run(
        [py, "-c", snippet], env=env, capture_output=True, text=True, timeout=30
    )
    return proc.returncode, proc.stdout, proc.stderr


def _make_non_test_tmpdir() -> Path:
    """Create a tmp path that contains NONE of `/tests/`, `/test_`, `_test.py`.

    pytest's ``tmp_path`` fixture creates paths like
    ``pytest-of-<user>/pytest-NN/test_<funcname>_M/...`` which contain
    ``/test_`` and trigger our path-audit guard erroneously. Use a
    dedicated mkdtemp under ``/tmp/round6_*`` instead.
    """
    return Path(tempfile.mkdtemp(prefix="round6_high1_nontest_"))


def test_unsafe_test_only_REFUSED_from_non_test_tools_path():
    """Production code under tools/-shaped paths must NOT bypass the gate."""
    # Write a "production-shaped" script under a non-test path.
    base = _make_non_test_tmpdir()
    try:
        prod_dir = base / "fake_tools"
        prod_dir.mkdir(parents=True)
        prod_script = prod_dir / "bad_dispatcher.py"
        prod_script.write_text(
            "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
            "gate = Phase3DispatchGate(unsafe_test_only=True)\n"
            "print('bypassed gate!')\n"
        )
        rc, stdout, stderr = _run_python_snippet(
            f"import sys; sys.path.insert(0, {str(prod_dir)!r}); "
            "import bad_dispatcher"
        )
        # Expect non-zero exit because Phase3DispatchGateError is raised
        assert rc != 0, (
            f"non-test caller MUST be refused; got rc=0\n"
            f"stdout={stdout}\nstderr={stderr}\npath={prod_script}"
        )
        assert "Phase3DispatchGateError" in stderr or "BLOCKED" in stderr, (
            f"expected Phase3DispatchGateError in stderr; got {stderr}"
        )
        assert "bypassed gate!" not in stdout, (
            "the gate was bypassed despite non-test caller"
        )
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


def test_unsafe_test_only_REFUSED_from_experiments_path():
    """experiments/ path must not bypass either."""
    base = _make_non_test_tmpdir()
    try:
        prod_dir = base / "fake_experiments"
        prod_dir.mkdir(parents=True)
        prod_script = prod_dir / "evil.py"
        prod_script.write_text(
            "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
            "gate = Phase3DispatchGate(unsafe_test_only=True)\n"
            "print('bypassed!')\n"
        )
        rc, stdout, stderr = _run_python_snippet(
            f"import sys; sys.path.insert(0, {str(prod_dir)!r}); "
            "import evil"
        )
        assert rc != 0, f"stdout={stdout}\nstderr={stderr}"
        assert "Phase3DispatchGateError" in stderr or "BLOCKED" in stderr
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


def test_unsafe_test_only_REFUSED_from_scripts_path():
    """scripts/ path must not bypass either."""
    base = _make_non_test_tmpdir()
    try:
        prod_dir = base / "fake_scripts"
        prod_dir.mkdir(parents=True)
        prod_script = prod_dir / "bad.py"
        prod_script.write_text(
            "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
            "Phase3DispatchGate(unsafe_test_only=True)\n"
        )
        rc, _, stderr = _run_python_snippet(
            f"import sys; sys.path.insert(0, {str(prod_dir)!r}); "
            "import bad"
        )
        assert rc != 0
        assert "Phase3DispatchGateError" in stderr or "BLOCKED" in stderr
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------
# Explicit operator-waiver path
# --------------------------------------------------------------------------


def test_unsafe_test_only_with_explicit_path_audit_waiver_works():
    """The double-opt-in (unsafe_test_only=True AND unsafe_test_only_path_audit_waived=True)
    must succeed even from a non-test path. This is the operator-review escape
    hatch for the rare legitimate fixture (e.g. an interactive REPL probe).
    """
    base = _make_non_test_tmpdir()
    try:
        prod_dir = base / "fake_tools"
        prod_dir.mkdir(parents=True)
        prod_script = prod_dir / "operator_repl.py"
        prod_script.write_text(
            "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
            "gate = Phase3DispatchGate(\n"
            "    unsafe_test_only=True,\n"
            "    unsafe_test_only_path_audit_waived=True,\n"
            ")\n"
            "print('opt-in waiver works')\n"
        )
        rc, stdout, stderr = _run_python_snippet(
            f"import sys; sys.path.insert(0, {str(prod_dir)!r}); "
            "import operator_repl"
        )
        assert rc == 0, (
            f"explicit double-opt-in MUST succeed; got rc={rc}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )
        assert "opt-in waiver works" in stdout
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------
# Default behaviour (no escape hatch) still required to satisfy preconditions
# --------------------------------------------------------------------------


def test_default_construction_still_requires_full_precondition_set():
    """No-kwarg construction (no unsafe_test_only) still must satisfy check()."""
    with pytest.raises(Phase3DispatchGateError) as exc_info:
        Phase3DispatchGate()
    msg = str(exc_info.value)
    assert "Phase 3 dispatch BLOCKED" in msg
    # The catalog-#142 message about path audit should NOT be in the
    # default-construction error (that's a separate code path)
    assert "is only permitted from test paths" not in msg


# --------------------------------------------------------------------------
# Preflight check #142 STRICT
# --------------------------------------------------------------------------


def test_preflight_check_142_passes_with_zero_violations():
    """The check should report zero live violations on a clean repo."""
    from tac.preflight import check_unsafe_test_only_restricted_to_test_paths

    violations = check_unsafe_test_only_restricted_to_test_paths(
        verbose=False, strict=False
    )
    # Live count expected: 0 — every existing call site is either a test
    # path or carries the explicit double-opt-in waiver.
    assert violations == [], (
        f"Catalog #142 preflight should be at 0 live violations; got "
        f"{len(violations)}:\n  " + "\n  ".join(violations[:5])
    )


def test_preflight_check_142_fires_on_simulated_violation(tmp_path: Path):
    """Simulating a non-test caller with unsafe_test_only=True must violate."""
    from tac.preflight import check_unsafe_test_only_restricted_to_test_paths

    # Build a fake repo root with a violating non-test file
    fake_repo = tmp_path
    (fake_repo / "tools").mkdir()
    bad_file = fake_repo / "tools" / "bad_dispatcher.py"
    bad_file.write_text(
        "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
        "Phase3DispatchGate(unsafe_test_only=True)\n"
    )
    # Also need to create the canonical scaffold to satisfy excluded-files
    # plus an empty src/tac dir
    (fake_repo / "src" / "tac" / "phase3").mkdir(parents=True)
    (fake_repo / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py").write_text(
        "# canonical module — excluded\n"
    )
    (fake_repo / "experiments").mkdir()
    (fake_repo / "scripts").mkdir()

    violations = check_unsafe_test_only_restricted_to_test_paths(
        repo_root=fake_repo, verbose=False, strict=False
    )
    assert len(violations) >= 1, (
        f"expected ≥1 violation on the simulated bad file; got {violations}"
    )
    assert "bad_dispatcher.py" in str(violations), (
        f"violation should mention bad_dispatcher.py; got {violations}"
    )


def test_preflight_check_142_accepts_test_path(tmp_path: Path):
    """Same kwarg in a test_*.py file must be allowed."""
    from tac.preflight import check_unsafe_test_only_restricted_to_test_paths

    fake_repo = tmp_path
    (fake_repo / "src" / "tac" / "tests").mkdir(parents=True)
    test_file = fake_repo / "src" / "tac" / "tests" / "test_fake.py"
    test_file.write_text(
        "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
        "def test_x():\n"
        "    Phase3DispatchGate(unsafe_test_only=True)\n"
    )
    (fake_repo / "src" / "tac" / "phase3").mkdir(parents=True)
    (fake_repo / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py").write_text(
        "# canonical module — excluded\n"
    )
    (fake_repo / "tools").mkdir()
    (fake_repo / "experiments").mkdir()
    (fake_repo / "scripts").mkdir()

    violations = check_unsafe_test_only_restricted_to_test_paths(
        repo_root=fake_repo, verbose=False, strict=False
    )
    assert violations == [], (
        f"test path must NOT violate; got {violations}"
    )


def test_preflight_check_142_accepts_explicit_double_opt_in_waiver(tmp_path: Path):
    """A non-test file passing the double-opt-in waiver must be allowed."""
    from tac.preflight import check_unsafe_test_only_restricted_to_test_paths

    fake_repo = tmp_path
    (fake_repo / "tools").mkdir()
    waiver_file = fake_repo / "tools" / "operator_repl.py"
    waiver_file.write_text(
        "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
        "gate = Phase3DispatchGate(\n"
        "    unsafe_test_only=True,\n"
        "    unsafe_test_only_path_audit_waived=True,\n"
        ")\n"
    )
    (fake_repo / "src" / "tac" / "phase3").mkdir(parents=True)
    (fake_repo / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py").write_text(
        "# canonical module — excluded\n"
    )
    (fake_repo / "experiments").mkdir()
    (fake_repo / "scripts").mkdir()

    violations = check_unsafe_test_only_restricted_to_test_paths(
        repo_root=fake_repo, verbose=False, strict=False
    )
    assert violations == [], (
        f"double-opt-in waiver must NOT violate; got {violations}"
    )


def test_preflight_check_142_accepts_same_line_waiver_marker(tmp_path: Path):
    """Same-line waiver marker # PHASE3_GATE_UNSAFE_PATH_WAIVED:<reason>."""
    from tac.preflight import check_unsafe_test_only_restricted_to_test_paths

    fake_repo = tmp_path
    (fake_repo / "tools").mkdir()
    waiver_file = fake_repo / "tools" / "documented.py"
    waiver_file.write_text(
        "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
        "Phase3DispatchGate(unsafe_test_only=True)  # PHASE3_GATE_UNSAFE_PATH_WAIVED:operator-repl-fixture\n"
    )
    (fake_repo / "src" / "tac" / "phase3").mkdir(parents=True)
    (fake_repo / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py").write_text(
        "# canonical module — excluded\n"
    )
    (fake_repo / "experiments").mkdir()
    (fake_repo / "scripts").mkdir()

    violations = check_unsafe_test_only_restricted_to_test_paths(
        repo_root=fake_repo, verbose=False, strict=False
    )
    assert violations == [], (
        f"same-line waiver marker must NOT violate; got {violations}"
    )


def test_preflight_check_142_strict_mode_raises():
    """STRICT mode must raise PreflightError on violations."""
    from tac.preflight import (
        PreflightError,
        check_unsafe_test_only_restricted_to_test_paths,
    )

    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td)
        (fake_repo / "tools").mkdir()
        bad = fake_repo / "tools" / "evil.py"
        bad.write_text(
            "from tac.phase3.joint_scorer_renderer_codec import Phase3DispatchGate\n"
            "Phase3DispatchGate(unsafe_test_only=True)\n"
        )
        (fake_repo / "src" / "tac" / "phase3").mkdir(parents=True)
        (fake_repo / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py").write_text("")
        (fake_repo / "experiments").mkdir()
        (fake_repo / "scripts").mkdir()

        with pytest.raises(PreflightError) as exc_info:
            check_unsafe_test_only_restricted_to_test_paths(
                repo_root=fake_repo, verbose=False, strict=True
            )
        assert "unsafe_test_only" in str(exc_info.value)
