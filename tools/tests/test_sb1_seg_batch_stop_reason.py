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
    """for/else is the only construct that distinguishes 'ran out' from 'converged at k'."""
    src = _source()
    i = src.index("for _step in range(args.max_quanta):")
    window = src[i : i + 1200]
    assert "\n        else:" in window, "the cap branch must be the for/else, not a length compare"
    assert 'stop_reason = "cap"' in window
