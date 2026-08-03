# SPDX-License-Identifier: MIT
"""ddm_dc1 2026-08-01 — the QA03 greedy-descent censoring guard.

The 2026-07-29 QA03 run stopped 51/120 instances (42.5%) on the hard
``--max-quanta`` cap of 4 rather than on its convergence test, and those censored
instances produced 64.7% of the realized flips.  The receipt recorded neither fact,
so a censored solve was consumed downstream as a converged one.

These tests are SCORER-FREE: they exercise the stopping-rule contract in
``tools/sb1_seg_batch.py`` by source inspection plus a faithful re-implementation
of the loop, never by running SegNet.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "tools" / "sb1_seg_batch.py"


def _source() -> str:
    return _SRC.read_text()


# --------------------------------------------------------------------------- #
# A faithful re-implementation of the loop under test (same control flow), so the
# stop_reason semantics can be exercised without a scorer.
# --------------------------------------------------------------------------- #
def _descend(flip_seq: list[int], max_quanta: int) -> tuple[int, str]:
    """Return (n_accepted_steps, stop_reason) for a scripted descent.

    ``flip_seq[i]`` is the best whole-pair flip count reachable at step i; the loop
    accepts while it strictly improves on ``cur``.
    """
    cur = flip_seq[0]
    accepted = 0
    stop_reason = "no_move"
    for step in range(max_quanta):
        nxt = flip_seq[step + 1] if step + 1 < len(flip_seq) else cur
        if nxt >= cur:
            stop_reason = "converged" if accepted else "no_move"
            break
        accepted += 1
        cur = nxt
    else:
        stop_reason = "cap"
    return accepted, stop_reason


def test_converged_when_no_improving_move_before_the_bound():
    # descends 100 -> 90 -> 85 then stalls; bound is generous
    assert _descend([100, 90, 85, 85], max_quanta=32) == (2, "converged")


def test_no_move_when_the_first_probe_already_fails():
    assert _descend([100, 100], max_quanta=32) == (0, "no_move")


def test_cap_when_still_descending_at_the_bound():
    # strictly descending for longer than the bound -> censored
    assert _descend([100, 90, 80, 70, 60, 50], max_quanta=4) == (4, "cap")


def test_the_old_default_of_4_would_have_censored_this_descent():
    """The regression the fix targets: the SAME descent is censored at 4, converged at 32."""
    seq = [100, 90, 80, 70, 60, 55, 52, 51, 51]
    assert _descend(seq, max_quanta=4) == (4, "cap")
    assert _descend(seq, max_quanta=32) == (7, "converged")


def test_max_quanta_default_is_not_the_censoring_value():
    """Guard the default itself: 4 is the value that produced the 42.5% censored run."""
    tree = ast.parse(_source())
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--max-quanta":
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                defaults["--max-quanta"] = kw.value.value
    assert "--max-quanta" in defaults, "--max-quanta argument disappeared from the CLI"
    assert defaults["--max-quanta"] > 4, (
        "--max-quanta default is back at the censoring value; the convergence test at "
        "sb1_seg_batch.py must be the terminator, not the bound"
    )


@pytest.mark.parametrize("token", ["stop_reason", "n_cap_saturated", "cap_saturated_frac"])
def test_receipt_surfaces_the_censoring(token: str):
    """A censored solve must be visible in the receipt, not recoverable only by forensics."""
    assert token in _source(), f"{token} missing: censoring would be invisible again"


def test_stop_reason_is_written_on_every_instance_row():
    src = _source()
    assert '"stop_reason": stop_reason' in src, (
        "per-instance rows must carry stop_reason so the step histogram is not the only "
        "evidence of censoring"
    )


def test_cap_branch_is_reached_via_for_else_not_a_post_hoc_length_check():
    """for/else is necessary to see 'ran out' at all -- but NOT sufficient to call it censoring.

    AMENDED ddm_sm1 2026-08-03. This test previously asserted the literal
    ``stop_reason = "cap"`` in the else-branch, and its docstring claimed for/else
    "distinguishes 'ran out' from 'converged at k'". MEASURED: it does not. Loop
    exhaustion conflates truncation with convergence that happens to land exactly on
    the bound, and on the shipped cap-4 store **7 of 12 "cap" labels (58.3%) were
    convergence coincidences** -- instances that took no further step when the bound
    was raised to 32. The else-branch must therefore PROBE once more before labelling.
    """
    src = _source()
    i = src.index("for _step in range(args.max_quanta):")
    window = src[i : i + 2000]
    assert "\n        else:" in window, "the cap branch must be the for/else, not a length compare"
    assert "_best_single_quantum" in window.split("\n        else:")[1], (
        "the else-branch must probe once more; loop exhaustion alone cannot tell "
        "truncation from convergence-at-the-bound"
    )
    assert '"converged_at_bound"' in window, (
        "convergence that lands exactly on the bound needs its own label, or "
        "cap_saturated_frac overstates censoring"
    )


# --------------------------------------------------------------------------- #
# ddm_sm1 2026-08-03 — classifying ALREADY-PERSISTED rows.
# --------------------------------------------------------------------------- #
def _classify():
    import sys

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.sb1_seg_batch import classify_persisted_stop_reason

    return classify_persisted_stop_reason


def test_persisted_row_prefers_its_own_stop_reason():
    c = _classify()
    row = {"accepted_steps": [[0, 1, 5]] * 4, "stop_reason": "converged_at_bound",
           "max_quanta_at_write": 4}
    assert c(row) == "converged_at_bound"


def test_persisted_row_uses_its_own_recorded_cap_not_the_readers():
    """The co-recorded bound governs, so a reader's --max-quanta cannot re-label history."""
    c = _classify()
    row = {"accepted_steps": [[0, 1, 5]] * 4, "max_quanta_at_write": 4}
    assert c(row) == "cap"
    assert c(row, legacy_cap=32) == "cap", "legacy_cap must not override a co-recorded bound"


def test_legacy_row_without_provenance_is_unknown_never_converged():
    """THE REGRESSION. The shipped fallback classified the WRITER's data with the
    READER's --max-quanta, so replaying the 120-row cap-4 store at the post-cure
    default of 32 reported n_cap_saturated=0 / cap_saturated_frac=0.000000 -- and the
    receipt's own note reads ">0 here means the solve is CENSORED, not solved", so 0.0
    reads as SOLVED. The cure (default 4 -> 32) thus silenced its own censoring
    detector on the very rows it was written to expose."""
    c = _classify()
    legacy = {"accepted_steps": [[0, 1, 5]] * 4}  # no stop_reason, no recorded cap
    assert c(legacy) == "unknown"
    assert c(legacy) != "converged", "silent 'converged' is the defect being fixed"
    # Declared provenance recovers the CLASS, but the row was still written by the
    # pre-cure solver, so the cap-hit keeps the conflated label rather than being
    # summed with post-cure censoring.
    assert c(legacy, legacy_cap=4) == "cap_conflated"


def test_legacy_no_move_row_is_unknown_without_provenance():
    c = _classify()
    assert _classify()({"accepted_steps": []}) == "unknown"
    assert c({"accepted_steps": []}, legacy_cap=4) == "no_move"


def test_pre_cure_cap_label_is_not_summed_with_post_cure_cap():
    """Found while REVIEWING the fix for the other two defects.

    Pre-cure rows carry ``stop_reason == "cap"`` under the OLD unconditional semantics
    (58.3% convergence coincidences); post-cure ``"cap"`` means genuinely still
    descending. Adding them would average two different quantities. They are
    distinguishable by the absence of ``max_quanta_at_write``.
    """
    c = _classify()
    pre_cure = {"accepted_steps": [[0, 1, 5]] * 4, "stop_reason": "cap"}
    post_cure = {"accepted_steps": [[0, 1, 5]] * 4, "stop_reason": "cap",
                 "max_quanta_at_write": 4}
    assert c(pre_cure) == "cap_conflated"
    assert c(post_cure) == "cap"
    assert c(pre_cure) != c(post_cure), "the two semantics must not collapse"


def test_explicit_null_recorded_cap_is_not_provenance():
    """An explicitly-null field is missing provenance, not a precise writer marker."""
    c = _classify()
    row = {"accepted_steps": [[0, 1, 5]] * 4, "max_quanta_at_write": None}
    assert c(row) == "unknown"
    assert c(row, legacy_cap=4) == "cap_conflated"


def _vacuous():
    import sys

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.sb1_seg_batch import legacy_cap_is_vacuous

    return legacy_cap_is_vacuous


def test_declared_legacy_cap_too_large_to_bind_is_vacuous():
    """THE THIRD-ORDER DEFECT. Declaring a cap no row can reach makes every legacy row
    'converged' by construction, so cap_saturated_frac reports a clean 0.0 having
    tested nothing -- reproducing, through the operator's own flag, the exact
    0.000000 this fix exists to prevent (m50: a check that cannot fail is not a check).
    """
    v = _vacuous()
    legacy = [{"accepted_steps": [[0, 1, 5]] * 4} for _ in range(10)]  # max 4 steps
    assert v(legacy, 32) is True, "cap 32 cannot bind rows with at most 4 steps"
    assert v(legacy, 4) is False, "cap 4 binds; the declaration is testable"
    assert v(legacy, None) is False


def test_vacuity_check_ignores_rows_that_carry_their_own_provenance():
    v = _vacuous()
    precise = [{"accepted_steps": [[0, 1, 5]] * 4, "stop_reason": "cap",
                "max_quanta_at_write": 4}]
    assert v(precise, 32) is False, "precise rows are not classified via legacy_cap"
    assert v([], 32) is False


def test_vacuous_declaration_is_downgraded_to_undetermined_at_the_callsite():
    src = _source()
    assert "legacy_cap_is_vacuous(_rows, args.legacy_cap)" in src
    assert "legacy_cap=None if _vacuous else args.legacy_cap" in src


def test_conflated_rows_are_excluded_from_the_censoring_count():
    src = _source()
    i = src.index('cls in ("unknown", "cap_conflated")')
    assert "n_cap_unknown += 1" in src[i : i + 320]


def test_receipt_fails_closed_when_censoring_is_undetermined():
    """cap_saturated_frac must be null, never 0.0, when any row is unclassifiable."""
    src = _source()
    assert '"n_cap_unknown"' in src
    assert '"censoring_determinable"' in src
    i = src.index('"cap_saturated_frac"')
    assert "None if n_cap_unknown" in src[i : i + 260], (
        "emitting 0.0 under undetermined censoring reads as 'solved'"
    )


def test_rows_co_record_the_bound_they_ran_under():
    """Provenance must travel WITH the data; that absence is what made the bug possible."""
    src = _source()
    assert '"max_quanta_at_write": int(args.max_quanta)' in src
