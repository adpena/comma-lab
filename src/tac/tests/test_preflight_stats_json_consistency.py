# SPDX-License-Identifier: MIT
"""Tests for Check PCC3: stats.json producers must carry an internal-
consistency assertion comparing elapsed-like wall-clock to epochs-like
iteration count BEFORE the json.dump call.

Bug class this guards: internal-inconsistency in reporting — stats.json
said `epochs: 200, elapsed_sec: 3.47` (200 epochs in 3.5s impossible).
A 1-line `assert elapsed >= epochs * MIN_SEC` would have caught the
IMP cycle 0 = 1.98 [contest-CUDA] metabug at runtime.

Memory:
- feedback_grand_council_imp_permanent_fix_review_20260430.md (DD2 PCC3)
- feedback_grand_council_pcc3_stats_consistency_20260430.md (the check)
- feedback_grand_council_recursive_greenup_shannon_floor_20260501.md
  (LOW #10: this dedicated test file added by Round 1 council greenup)

Tests cover:
  - Real-codebase regression (STRICT @ 0)
  - Positive: legitimate producer with backing assertion passes
  - Negative: producer without backing assertion is caught
  - Inter-function waiver: PCC3-WAIVED-INTERFUNCTION marker auto-passes
  - Same-line waiver: PCC3-WAIVED marker auto-passes
  - Strict mode raises MetaBugViolation
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.preflight import (  # noqa: E402
    MetaBugViolation,
    check_stats_json_internal_consistency,
)


# ─────────────────────────────────────────────────────────────────────────────
# Real-codebase regression — STRICT @ 0 violations
# ─────────────────────────────────────────────────────────────────────────────


def test_real_codebase_passes() -> None:
    """[regression] Real codebase has 0 PCC3 violations.

    The canonical fix in ``experiments/train_imp_cycle.py:362-374`` provides
    the wall-clock floor assertion before the stats.json write at line ~393.
    Other train scripts (e.g. ``experiments/train_segmap.py`` line 456,
    ``experiments/train_segmap_film_canvas.py`` line 337) were also fixed
    in the PCC3 landing wave 2026-04-30.
    """
    violations = check_stats_json_internal_consistency(
        strict=False, verbose=False,
    )
    assert violations == [], (
        f"PCC3 expected 0 violations on the real codebase, got "
        f"{len(violations)}:\n  • " + "\n  • ".join(violations[:5])
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-fixture tests
# ─────────────────────────────────────────────────────────────────────────────


def _write_fixture(tmp_path: Path, content: str) -> Path:
    """Build a fake repo layout with the given Python content under
    ``scripts/foo.py`` so the check's scan dirs see it."""
    (tmp_path / "scripts").mkdir()
    fp = tmp_path / "scripts" / "fake_trainer.py"
    fp.write_text(textwrap.dedent(content))
    return fp


def test_legitimate_producer_with_backing_assertion_passes(tmp_path: Path) -> None:
    """A producer with a `elapsed >= epochs * MIN_SEC` assertion before
    the json.dump call is OK."""
    _write_fixture(
        tmp_path,
        '''
        import json, time

        def main():
            epochs = 200
            t0 = time.time()
            # ... do training ...
            elapsed_sec = time.time() - t0
            MIN_SEC = 0.05
            assert elapsed_sec >= epochs * MIN_SEC, "stub loop detected"
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "elapsed_sec": elapsed_sec}, f)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == [], f"expected 0 violations, got: {v}"


def test_producer_without_backing_assertion_is_caught(tmp_path: Path) -> None:
    """A producer that writes both epochs + elapsed_sec WITHOUT an assertion
    is the IMP cycle 0 = 1.98 metabug class — must be caught."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def main():
            epochs = 200
            elapsed_sec = 3.47  # the smoking gun
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "elapsed_sec": elapsed_sec}, f)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(v) == 1, f"expected 1 violation, got: {v}"
    assert "elapsed_sec" in v[0] and "epochs" in v[0]


def test_same_line_waiver_passes(tmp_path: Path) -> None:
    """`# PCC3-WAIVED: <reason>` same-line marker on the json.dump call
    auto-passes (legitimate cases like a smoke-only producer)."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def main():
            epochs = 0  # smoke mode
            elapsed_sec = 0.01
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "elapsed_sec": elapsed_sec}, f)  # PCC3-WAIVED: smoke-only producer always reports zero
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == [], f"PCC3-WAIVED should auto-pass, got: {v}"


def test_inter_function_waiver_passes(tmp_path: Path) -> None:
    """`# PCC3-WAIVED-INTERFUNCTION: <reason>` allows the backing assertion
    to live in the caller (e.g. main()) while the json.dump is in a helper
    (e.g. _save_state())."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def _save_state(epochs, elapsed_sec):
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "elapsed_sec": elapsed_sec}, f)  # PCC3-WAIVED-INTERFUNCTION: assertion lives in main()

        def main():
            epochs = 200
            elapsed_sec = 50.0
            assert elapsed_sec >= epochs * 0.05
            _save_state(epochs, elapsed_sec)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == [], f"PCC3-WAIVED-INTERFUNCTION should auto-pass, got: {v}"


def test_strict_mode_raises(tmp_path: Path) -> None:
    """In strict mode, a violation MUST raise MetaBugViolation (not just
    return a list). This is what makes PCC3 a commit-time blocker."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def main():
            with open("stats.json", "w") as f:
                json.dump({"epochs": 200, "elapsed_sec": 3.47}, f)
        ''',
    )
    with pytest.raises(MetaBugViolation):
        check_stats_json_internal_consistency(
            strict=True, verbose=False, repo_root=tmp_path,
        )


def test_only_epochs_no_elapsed_does_not_trigger(tmp_path: Path) -> None:
    """Producer with ONLY epochs (no elapsed-like key) does not trigger;
    the inconsistency check is irrelevant without both axes."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def main():
            epochs = 200
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "loss": 0.5}, f)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == [], f"expected 0 violations, got: {v}"


def test_only_elapsed_no_epochs_does_not_trigger(tmp_path: Path) -> None:
    """Producer with ONLY elapsed (no epochs-like key) does not trigger."""
    _write_fixture(
        tmp_path,
        '''
        import json
        import time

        def main():
            elapsed_sec = time.time()
            with open("stats.json", "w") as f:
                json.dump({"elapsed_sec": elapsed_sec, "device": "cuda"}, f)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert v == [], f"expected 0 violations, got: {v}"


def test_assertion_with_other_op_does_not_satisfy(tmp_path: Path) -> None:
    """A bare `assert epochs > 0` (no Mult/Div op) does NOT satisfy the
    backing-assertion requirement — the check specifically requires a
    `elapsed >= epochs * X` style with multiplication/division."""
    _write_fixture(
        tmp_path,
        '''
        import json

        def main():
            epochs = 200
            elapsed_sec = 3.47
            assert epochs > 0  # this does NOT count
            with open("stats.json", "w") as f:
                json.dump({"epochs": epochs, "elapsed_sec": elapsed_sec}, f)
        ''',
    )
    v = check_stats_json_internal_consistency(
        strict=False, verbose=False, repo_root=tmp_path,
    )
    assert len(v) == 1, (
        f"`assert epochs > 0` lacks Mult/Div op and should NOT count as a "
        f"backing assertion; expected 1 violation, got: {v}"
    )
