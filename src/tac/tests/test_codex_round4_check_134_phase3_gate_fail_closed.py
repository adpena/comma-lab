# SPDX-License-Identifier: MIT
"""Tests for codex round-4 catalog #134 — Phase3DispatchGate fail-closed.

Bug class (codex round-4 MEDIUM 1, 2026-05-09): the previous
``Phase3DispatchGate.__post_init__`` was a no-op with a comment
"tests need to construct with a permissive gate". Any future
trainer/dispatcher could silently bypass every precondition by forgetting
to call ``gate.check()``. The fix made construction fail-closed with an
explicit ``unsafe_test_only=True`` opt-out for tests.

The gate (#134) refuses any ``Phase3DispatchGate(...)`` call site
outside the canonical scaffold module that does NOT pass
``unsafe_test_only=True`` AND does NOT pass the full production
precondition kwarg set.

Memory: feedback_codex_round4_findings_fix_with_self_protection_landed_20260509.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_phase3_dispatch_gate_fail_closed,
)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src" / "tac" / "phase3").mkdir(parents=True)
    (tmp_path / "tools").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    return tmp_path


# ── Live-repo sanity ────────────────────────────────────────────────────


def test_134_live_repo_clean():
    """Live-repo: catalog #134 must land at 0 violations after fix."""
    v = check_phase3_dispatch_gate_fail_closed(strict=False, verbose=False)
    assert v == [], (
        f"Catalog #134 landed with {len(v)} violations:\n"
        + "\n".join(v[:3])
    )


# ── Catch the unsafe construction pattern ──────────────────────────────


def test_134_catches_bare_gate_construction_in_production_code(tmp_path):
    """Bare `Phase3DispatchGate()` in tools/experiments/scripts (NOT tests)
    that does NOT set unsafe_test_only must be flagged.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "bad_dispatcher.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def dispatch():\n"
        "    gate = Phase3DispatchGate()  # bypasses every precondition\n"
        "    # ...trainer code...\n"
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "bad_dispatcher.py" in x]
    assert len(matches) == 1, f"Expected to catch bare construction; got {v}"


def test_134_catches_partial_precondition_construction(tmp_path):
    """A construction with SOME but not all preconditions must still be flagged
    (partial = unsafe by definition; the gate enforces all-or-nothing)."""
    root = _make_repo(tmp_path)
    (root / "experiments" / "partial.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def go():\n"
        "    g = Phase3DispatchGate(\n"
        "        phase2_anchor_verified=True,\n"
        "        phase2_anchor_score=0.140,\n"
        "        # missing other required kwargs\n"
        "    )\n"
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "partial.py" in x]
    assert len(matches) == 1


# ── Accept clean patterns ──────────────────────────────────────────────


def test_134_accepts_unsafe_test_only_construction(tmp_path):
    """Tests/utilities that explicitly opt out via unsafe_test_only=True
    are acceptable — though tests are also path-excluded."""
    root = _make_repo(tmp_path)
    (root / "tools" / "fixture.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def make_fixture():\n"
        "    return Phase3DispatchGate(unsafe_test_only=True)\n"
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "fixture.py" in x]
    assert len(matches) == 0


def test_134_accepts_full_production_precondition_set(tmp_path):
    """A construction with EVERY required precondition kwarg is the
    canonical production pattern — must pass.
    """
    root = _make_repo(tmp_path)
    (root / "tools" / "good_dispatcher.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def dispatch():\n"
        "    gate = Phase3DispatchGate(\n"
        "        phase2_anchor_verified=True,\n"
        "        phase2_anchor_score=0.140,\n"
        "        phase2_anchor_evidence_path='ev.json',\n"
        "        distillation_gap_estimate=0.025,\n"
        "        distillation_gap_evidence_path='dg.json',\n"
        "        operator_approved_gpu_budget_usd=800.0,\n"
        "        aaf68f37_verdict_clean=True,\n"
        "        aaf68f37_verdict_evidence_path='aaf.md',\n"
        "        phase3_council_deliberation_path='council.md',\n"
        "    )\n"
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "good_dispatcher.py" in x]
    assert len(matches) == 0


def test_134_accepts_canonical_scaffold_module():
    """The canonical scaffold module
    (`src/tac/phase3/joint_scorer_renderer_codec.py`) is excluded from
    scanning — it defines the gate and naturally references it.
    """
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "src" / "tac" / "phase3" / "joint_scorer_renderer_codec.py"
    assert target.exists()
    text = target.read_text()
    assert "Phase3DispatchGate" in text
    # Live-repo run already proved this file does not generate a violation


def test_134_accepts_test_files(tmp_path):
    """Test files (which legitimately mock the gate construction) MUST be
    excluded from scanning."""
    root = _make_repo(tmp_path)
    test_file = root / "src" / "tac" / "tests"
    test_file.mkdir(parents=True)
    (test_file / "test_phase3_xyz.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def test_x():\n"
        "    g = Phase3DispatchGate()\n"  # would be unsafe in prod, OK in test
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "test_phase3_xyz.py" in x]
    assert len(matches) == 0


# ── Waiver works ────────────────────────────────────────────────────────


def test_134_same_line_waiver_accepts(tmp_path):
    """`# PHASE3_GATE_OK:<reason>` waiver lets through rare integration
    points that need to construct with explicit precondition values."""
    root = _make_repo(tmp_path)
    (root / "tools" / "waived.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def builder():\n"
        "    return Phase3DispatchGate()  # PHASE3_GATE_OK: builder wraps with extra validation\n"
    )
    v = check_phase3_dispatch_gate_fail_closed(
        repo_root=root, strict=False, verbose=False
    )
    matches = [x for x in v if "waived.py" in x]
    assert len(matches) == 0


# ── Strict-mode round-trip ──────────────────────────────────────────────


def test_134_strict_raises(tmp_path):
    root = _make_repo(tmp_path)
    (root / "scripts" / "evil.py").write_text(
        "from tac.phase3 import Phase3DispatchGate\n"
        "def go():\n"
        "    g = Phase3DispatchGate()\n"
    )
    with pytest.raises(
        PreflightError, match="check_phase3_dispatch_gate_fail_closed"
    ):
        check_phase3_dispatch_gate_fail_closed(
            repo_root=root, strict=True, verbose=False
        )


# ── Real Phase 3 module integration tests ───────────────────────────────


def test_134_real_phase3_gate_default_construction_raises():
    """The real Phase3DispatchGate() default constructor must raise
    because the fix made the gate fail-closed by default."""
    from tac.phase3 import Phase3DispatchGate, Phase3DispatchGateError

    with pytest.raises(Phase3DispatchGateError):
        Phase3DispatchGate()


def test_134_real_phase3_gate_unsafe_test_only_works():
    """The unsafe_test_only=True opt-out must work for tests."""
    from tac.phase3 import Phase3DispatchGate

    gate = Phase3DispatchGate(unsafe_test_only=True)  # must not raise
    assert gate.unsafe_test_only is True


def test_134_real_phase3_gate_full_preconditions_construct(tmp_path):
    """Full production preconditions allow construction."""
    from tac.phase3 import Phase3DispatchGate

    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="ev.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="distill.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="aaf.md",
        phase3_council_deliberation_path="council.md",
    )
    assert gate.phase2_anchor_verified is True
    assert gate.unsafe_test_only is False


def test_134_real_scaffold_construction_with_permissive_gate_works():
    """The scaffold accepts an opted-out gate (for tests)."""
    from tac.phase3 import (
        JointScorerRendererCodecConfig,
        JointScorerRendererCodecScaffold,
        Phase3DispatchGate,
    )

    cfg = JointScorerRendererCodecConfig()
    gate = Phase3DispatchGate(unsafe_test_only=True)
    scaffold = JointScorerRendererCodecScaffold(config=cfg, gate=gate)
    assert scaffold.gate is gate
