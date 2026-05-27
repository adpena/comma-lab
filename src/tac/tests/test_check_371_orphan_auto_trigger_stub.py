# SPDX-License-Identifier: MIT
"""Tests for Catalog #371 — check_no_orphan_auto_trigger_stub_with_satisfied_condition.

Per STUB-AUDIT-AND-FIX wave 2026-05-27 (operator directive "fix all stubs and
continue iterating and optimizing and auditing"). The gate refuses re-introduction
of the orphan-auto-trigger-stub: a no-op ``auto_recalibrate_*`` that hardcodes
``equations_recalibrated=0`` / carries the follow-on-stub marker WITHOUT wiring
the canonical ``EVENT_RECALIBRATED`` emission path, unless a same-line
``# DEFERRED_STUB_OK:<reactivation-criteria>`` waiver is present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_orphan_auto_trigger_stub_with_satisfied_condition,
)


_FN = "auto_recalibrate_from_continual_learning_posterior"
_RELPATH = "src/tac/canonical_equations/registry.py"


# ---------------------------------------------------------------------------
# Synthetic repo fixtures
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, body: str) -> Path:
    """Create a synthetic repo with src/tac/canonical_equations/registry.py."""
    p = tmp_path / "src" / "tac" / "canonical_equations" / "registry.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return tmp_path


_WIRED_BODY = '''\
# SPDX-License-Identifier: MIT
EVENT_RECALIBRATED = "recalibrated"


def auto_recalibrate_from_continual_learning_posterior(equation_id=None, *, path=None):
    """Refit from anchors."""
    # emits EVENT_RECALIBRATED when 3+ anchors land
    _append_event_locked(EVENT_RECALIBRATED, updated, path=path)
    return RecalibrationReport(equations_checked=1, equations_recalibrated=1, new_anchors_absorbed=1)


def next_top_level():
    pass
'''

_NOOP_STUB_BODY = '''\
# SPDX-License-Identifier: MIT


def auto_recalibrate_from_continual_learning_posterior(equation_id=None, *, path=None):
    """Stub."""
    return RecalibrationReport(
        equations_checked=len(equations),
        equations_recalibrated=0,  # stub; auto-refit comes in a follow-on landing
        new_anchors_absorbed=0,
    )


def next_top_level():
    pass
'''

_NOOP_WITH_WAIVER_BODY = '''\
# SPDX-License-Identifier: MIT


def auto_recalibrate_from_continual_learning_posterior(equation_id=None, *, path=None):
    """Stub deferred pending the cathedral consumer landing."""
    return RecalibrationReport(
        equations_checked=len(equations),
        equations_recalibrated=0,  # stub; auto-refit comes in a follow-on landing  # DEFERRED_STUB_OK:reactivate when cathedral_equation_lookup_consumer lands the posterior-read path
        new_anchors_absorbed=0,
    )


def next_top_level():
    pass
'''

_NOOP_WITH_PLACEHOLDER_WAIVER_BODY = '''\
# SPDX-License-Identifier: MIT


def auto_recalibrate_from_continual_learning_posterior(equation_id=None, *, path=None):
    """Stub."""
    return RecalibrationReport(
        equations_recalibrated=0,  # stub; auto-refit comes in a follow-on landing  # DEFERRED_STUB_OK:<reactivation-criteria>
    )


def next_top_level():
    pass
'''

_MISSING_FN_BODY = '''\
# SPDX-License-Identifier: MIT
EVENT_RECALIBRATED = "recalibrated"


def some_other_function():
    pass
'''


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_passes_clean() -> None:
    """The live repo at landing has zero violations (the fix is wired)."""
    violations = check_no_orphan_auto_trigger_stub_with_satisfied_condition()
    assert violations == [], "\n  - ".join(["Live repo Catalog #371:"] + violations[:5])


def test_live_repo_strict_does_not_raise() -> None:
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(strict=True) == []


# ---------------------------------------------------------------------------
# Positive (caught) cases
# ---------------------------------------------------------------------------


def test_noop_stub_without_emission_or_waiver_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, _NOOP_STUB_BODY)
    violations = check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root)
    assert len(violations) >= 1
    assert any("EVENT_RECALIBRATED" in v for v in violations)


def test_noop_stub_placeholder_waiver_rejected(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, _NOOP_WITH_PLACEHOLDER_WAIVER_BODY)
    violations = check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root)
    assert len(violations) >= 1
    assert any("placeholder" in v.lower() for v in violations)


def test_missing_function_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, _MISSING_FN_BODY)
    violations = check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root)
    assert len(violations) == 1
    assert "not found" in violations[0]


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, _NOOP_STUB_BODY)
    with pytest.raises(PreflightError, match="Catalog #371"):
        check_no_orphan_auto_trigger_stub_with_satisfied_condition(
            repo_root=root, strict=True
        )


# ---------------------------------------------------------------------------
# Negative (accepted) cases
# ---------------------------------------------------------------------------


def test_wired_recalibrator_accepted(tmp_path: Path) -> None:
    """A recalibrator that wires EVENT_RECALIBRATED is NOT a no-op orphan."""
    root = _make_repo(tmp_path, _WIRED_BODY)
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root) == []


def test_noop_stub_with_real_waiver_accepted(tmp_path: Path) -> None:
    """A no-op stub with a substantive DEFERRED_STUB_OK reactivation criterion passes."""
    root = _make_repo(tmp_path, _NOOP_WITH_WAIVER_BODY)
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root) == []


def test_wired_body_with_residual_stub_comment_accepted(tmp_path: Path) -> None:
    """If EVENT_RECALIBRATED is wired, a leftover stub comment is historical-only."""
    body = (
        "# SPDX-License-Identifier: MIT\n"
        'EVENT_RECALIBRATED = "recalibrated"\n\n\n'
        f"def {_FN}(equation_id=None, *, path=None):\n"
        '    """Refit."""\n'
        "    # NOTE: auto-refit comes in a follow-on landing (historical comment)\n"
        "    _append_event_locked(EVENT_RECALIBRATED, eq, path=path)\n"
        "    return RecalibrationReport(equations_checked=1, equations_recalibrated=1, new_anchors_absorbed=1)\n\n\n"
        "def next_top_level():\n    pass\n"
    )
    root = _make_repo(tmp_path, body)
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root) == []


def test_missing_target_file_skips(tmp_path: Path) -> None:
    """No registry.py present -> skip (returns empty)."""
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=tmp_path) == []


def test_string_repo_root_accepted(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, _WIRED_BODY)
    assert check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=str(root)) == []


# ---------------------------------------------------------------------------
# Body extraction helper
# ---------------------------------------------------------------------------


def test_body_extractor_returns_none_when_fn_absent() -> None:
    from tac.preflight import _check_371_extract_recalibrator_body

    assert _check_371_extract_recalibrator_body("def foo():\n    pass\n") is None


def test_body_extractor_stops_at_next_top_level() -> None:
    from tac.preflight import _check_371_extract_recalibrator_body

    src = (
        f"def {_FN}(x):\n"
        "    return 1\n\n\n"
        "def other():\n"
        "    return 2\n"
    )
    body = _check_371_extract_recalibrator_body(src)
    assert body is not None
    assert "def other" not in body
    assert "return 1" in body


# ---------------------------------------------------------------------------
# Verbose + multi-marker aggregation
# ---------------------------------------------------------------------------


def test_verbose_does_not_crash(tmp_path: Path, capsys) -> None:
    root = _make_repo(tmp_path, _NOOP_STUB_BODY)
    check_no_orphan_auto_trigger_stub_with_satisfied_condition(repo_root=root, verbose=True)
    out = capsys.readouterr().out
    assert "orphan-auto-trigger-stub" in out


def test_orchestrator_wires_strict_true() -> None:
    """preflight_all() wires Catalog #371 at strict=True (regression guard)."""
    import inspect

    from tac import preflight

    src = inspect.getsource(preflight.preflight_all)
    assert "check_no_orphan_auto_trigger_stub_with_satisfied_condition(" in src
    assert "strict=True" in src.split(
        "check_no_orphan_auto_trigger_stub_with_satisfied_condition("
    )[1].split(")")[0] + ")"


def test_catalog_185_sister_callable() -> None:
    """Gate is callable via module globals (Catalog #185 sister regression)."""
    from tac import preflight

    fn = getattr(preflight, "check_no_orphan_auto_trigger_stub_with_satisfied_condition")
    assert callable(fn)
    assert fn(strict=False) == []
