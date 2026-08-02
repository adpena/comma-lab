# SPDX-License-Identifier: MIT
"""ddm_pb2 (#900) — guards for the symmetric-entry bracket.

The ddm_pw1 outward bracket commits to the FIRST improving entry probe and
``break``s, so ``-`` is never evaluated when ``+`` improves at all.  MEASURED
by ddm_lg2 on the shipped ``pw1_arms.jsonl``: 125 arm-instances over 109
distinct pairs (15.89% of the live arm-AB d_pose mass) commit to ``+``
untested, while among the pairs where ``-`` is allowed to compete it wins 60
to 31 in arm B.

These tests guard BEHAVIOUR, not constants.  Each carries a POSITIVE CONTROL:
an evaluator on which the asymmetric bracket demonstrably makes the wrong
choice and the symmetric one does not, so a test suite that passed with both
variants collapsed to the same code would fail here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, relpath: str):
    """Import a repo script by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    if spec is None or spec.loader is None:  # pragma: no cover - env guard
        pytest.skip(f"cannot load {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pb2():
    return _load("_pb2_bracket_direction_ab", "tools/pb2_bracket_direction_ab.py")


def _counting(fn):
    """Wrap an evaluator so a test can assert the number of forwards."""
    calls: list[float] = []

    def _ev(x):
        calls.append(float(x))
        return fn(x)

    _ev.calls = calls  # type: ignore[attr-defined]
    return _ev


# --------------------------------------------------------------------------- #
# THE MEASURED DEFECT: the ``break`` commits to ``+`` without seeing ``-``
# --------------------------------------------------------------------------- #
def test_asym_commits_to_plus_without_evaluating_minus(pb2):
    """POSITIVE CONTROL for the defect itself.

    ``+`` improves a little, ``-`` improves a lot.  The asymmetric bracket must
    take ``+`` after exactly ONE entry probe -- if this ever stops holding, the
    thing pb2 measures no longer exists and every number in the receipt is
    about something else.
    """
    def f(x):
        return {0.0: 10.0, 1.0: 9.0, -1.0: 1.0}.get(round(x, 9), 100.0)

    ev = _counting(f)
    x, d, n, probes, direction = pb2.bracket_asym(ev, 0.0, 10.0, 1.0, 0)
    assert direction == 1.0
    assert x == 1.0 and d == 9.0
    assert len([p for p in probes if p["phase"] == "probe"]) == 1
    assert -1.0 not in ev.calls, "the '-' probe must never have been evaluated"


def test_sym_evaluates_both_and_takes_the_better(pb2):
    """The cure: both entry probes always, commit to the better."""
    def f(x):
        return {0.0: 10.0, 1.0: 9.0, -1.0: 1.0}.get(round(x, 9), 100.0)

    ev = _counting(f)
    x, d, n, probes, direction = pb2.bracket_sym(ev, 0.0, 10.0, 1.0, 0)
    assert direction == -1.0
    assert x == -1.0 and d == 1.0
    assert len([p for p in probes if p["phase"] == "probe"]) == 2
    assert sorted(ev.calls) == [-1.0, 1.0]


def test_sym_costs_exactly_one_extra_entry_probe_when_plus_improves(pb2):
    """The pre-registered cost claim, as a test rather than an estimate."""
    def f(x):
        return {0.0: 10.0, 1.0: 9.0, -1.0: 11.0}.get(round(x, 9), 100.0)

    a = _counting(f)
    s = _counting(f)
    pb2.bracket_asym(a, 0.0, 10.0, 1.0, 0)
    pb2.bracket_sym(s, 0.0, 10.0, 1.0, 0)
    assert len(a.calls) == 1
    assert len(s.calls) == 2  # exactly +1 forward on a short-circuiting pair


def test_variants_agree_when_plus_does_not_improve(pb2):
    """The 491-pair majority: if ``+`` fails, asym already evaluated ``-``.

    This is why the honest scope is 109 pairs and not 600 -- the two variants
    are provably identical everywhere else.
    """
    def f(x):
        return {0.0: 10.0, 1.0: 11.0, -1.0: 4.0}.get(round(x, 9), 100.0)

    ra = pb2.bracket_asym(_counting(f), 0.0, 10.0, 1.0, 0)
    rs = pb2.bracket_sym(_counting(f), 0.0, 10.0, 1.0, 0)
    assert ra[:3] == rs[:3] and ra[4] == rs[4]


# --------------------------------------------------------------------------- #
# monotone safety + tie-breaking (what makes the delta trustworthy)
# --------------------------------------------------------------------------- #
def test_sym_is_monotone_safe_under_an_adversarial_evaluator(pb2):
    """No arm may report a win it did not realize."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        table = {round(v, 9): float(rng.normal(5.0, 3.0))
                 for v in np.arange(-40.0, 40.0, 0.5)}
        d0 = table[0.0]
        _, d, _, _, _ = pb2.bracket_sym(
            lambda x, _t=table: _t.get(round(x, 9), 1e9), 0.0, d0, 0.5, 12)
        assert d <= d0 + 0.0, "bracket_sym returned a worse point than x0"


def test_plus_wins_exact_ties_so_any_delta_is_a_strict_minus_win(pb2):
    """Tie-breaking is the reason a measured delta cannot be an artefact."""
    def f(x):
        return {0.0: 10.0, 1.0: 7.0, -1.0: 7.0}.get(round(x, 9), 100.0)

    _, _, _, _, direction = pb2.bracket_sym(f, 0.0, 10.0, 1.0, 0)
    assert direction == 1.0, "an exact tie must not flip the direction"


def test_neither_direction_improving_leaves_x0_untouched(pb2):
    def f(x):
        return {0.0: 1.0}.get(round(x, 9), 99.0)

    for fn in (pb2.bracket_asym, pb2.bracket_sym):
        x, d, _, _, direction = fn(f, 0.0, 1.0, 1.0, 12)
        assert (x, d, direction) == (0.0, 1.0, 0.0)


# --------------------------------------------------------------------------- #
# the expansion is shared, and the lattice is respected
# --------------------------------------------------------------------------- #
def test_both_variants_share_one_doubling_expansion(pb2):
    """Only the ENTRY rule differs; a divergent expansion would confound."""
    def f(x):
        return -abs(x) if x <= 0 else 100.0  # improves without bound on '-'

    xa, da, _, _, _ = pb2.bracket_asym(f, 0.0, 0.0, 1.0, 6)
    xs, ds, _, _, _ = pb2.bracket_sym(f, 0.0, 0.0, 1.0, 6)
    assert (xa, da) == (xs, ds)
    assert da < 0.0, "the expansion must actually have run"


def test_expansion_terminates_under_unbounded_improvement(pb2):
    """Termination is a proof, not a budget -- but max_expand still caps it."""
    calls = 0

    def f(x):
        nonlocal calls
        calls += 1
        return -abs(x)

    pb2.bracket_sym(f, 0.0, 0.0, 1.0, 5)
    assert calls <= 2 + 5


def test_quantize_collapsing_a_candidate_onto_x0_skips_that_probe(pb2):
    """A lattice that cannot represent the step must not burn a forward."""
    ev = _counting(lambda x: 0.0)
    pb2.bracket_sym(ev, 0.0, 1.0, 1e-12, 3, quantize=lambda x: 0.0)
    assert ev.calls == [], "collapsed candidates must not be evaluated"


def test_expansion_stays_on_the_quantization_lattice(pb2):
    seen: list[float] = []

    def q(x):
        return float(np.round(x * 4.0) / 4.0)

    def f(x):
        seen.append(x)
        return -abs(x)

    pb2.bracket_sym(f, 0.0, 0.0, 0.3, 4, quantize=q)
    assert all(v == q(v) for v in seen)


# --------------------------------------------------------------------------- #
# the memo (what makes running BOTH variants nearly free, and floor-free)
# --------------------------------------------------------------------------- #
def test_memo_counts_unique_forwards_only(pb2):
    m = pb2.MemoEval(_counting(lambda x: float(x) ** 2))
    for v in (1.0, 1.0, 2.0, 1.0, 2.0, 3.0):
        m(v)
    assert m.n_forward == 3


def test_memo_makes_the_second_variant_nearly_free(pb2):
    """Both variants on one memo => the delta carries no cross-run floor."""
    def f(x):
        return {0.0: 10.0, 1.0: 9.0, -1.0: 11.0}.get(round(x, 9), 100.0)

    inner = _counting(f)
    m = pb2.MemoEval(inner)
    pb2.bracket_asym(m, 0.0, 10.0, 1.0, 0)
    before = m.n_forward
    pb2.bracket_sym(m, 0.0, 10.0, 1.0, 0)
    assert before == 1
    assert m.n_forward - before == 1, "only the untested '-' probe is new"


# --------------------------------------------------------------------------- #
# VACUOUS is never PASS (memory: vacuity_is_indistinguishable_from_pass)
# --------------------------------------------------------------------------- #
def test_verify_input_on_an_empty_scope_is_vacuous_not_pass(pb2, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    out = pb2.verify_input(empty)
    assert out["verdict"] == "VACUOUS"
    assert out["denominator"] == 0


def test_the_entry_rule_does_not_dominate(pb2):
    """MEASURED on the real run (pair 326) and reproduced here synthetically.

    The entry probe is a ONE-STEP-LOOKAHEAD greedy: ``-`` can win the entry
    probe and still lose the doubling continuation.  A test suite that assumed
    "symmetric is always >= asymmetric" would be asserting something FALSE, so
    this guards the honest framing rather than the hoped-for one.
    """
    # '-' wins the entry probe (4 < 6) but '+' expands far past it.  The
    # doubling walks best_x +- 2, then +- 4, so the '+' chain is 1 -> 3 -> 7
    # and the '-' chain is -1 -> -3 -> -7.
    table = {0.0: 10.0,
             1.0: 6.0, 3.0: 0.5, 7.0: 0.4,
             -1.0: 4.0, -3.0: 3.9, -7.0: 3.8}

    def f(x):
        return table.get(round(x, 9), 100.0)

    _, d_asym, _, _, _ = pb2.bracket_asym(f, 0.0, 10.0, 1.0, 4)
    _, d_sym, _, _, _ = pb2.bracket_sym(f, 0.0, 10.0, 1.0, 4)
    assert d_sym > d_asym, "this fixture must exhibit the non-dominance"
    # both are still monotone-safe against the starting point
    assert d_asym <= 10.0 and d_sym <= 10.0


def test_falsifier_is_gated_by_the_positive_control(pb2):
    """L3 verdict clearance: an untrusted instrument yields NO verdict.

    A huge apparent win must NOT be reported as a win when the control failed.
    """
    assert pb2.falsifier_verdict(1e-9, 109, -1.0, -1.0) == "INSTRUMENT_UNTRUSTED"
    assert pb2.falsifier_verdict(0.0, 109, -1.0, -1.0) == "ASYMMETRY_PRICED"


def test_falsifier_requires_both_denominators_to_agree(pb2):
    """The population reading is the easier null; it must not decide alone."""
    # clean null on both readings
    assert pb2.falsifier_verdict(0.0, 109, 0.0, 0.0) == "NULL_PRICED_AT_ZERO"
    # a per-pair effect that the /600 population mean would have hidden
    assert pb2.falsifier_verdict(
        0.0, 109, -9.1e-7, -5e-6) == "BELOW_SCORE_RESOLUTION"
    # both agree it is real
    assert pb2.falsifier_verdict(0.0, 109, -1e-4, -6e-4) == "ASYMMETRY_PRICED"


def test_falsifier_on_an_empty_scope_is_vacuous_not_null(pb2):
    """VACUOUS != a clean null -- an empty scope proves nothing either way."""
    assert pb2.falsifier_verdict(0.0, 0, None, None) == "VACUOUS"
    assert pb2.falsifier_verdict(0.0, 0, 0.0, 0.0) == "VACUOUS"


def test_verify_input_reports_its_denominator(pb2, tmp_path):
    """A count without a denominator is unreadable; the schema must carry it."""
    import json
    row = {
        "pair": 0, "d_ctrl": 1.0, "arm_ab_d": 0.5,
        "arm_a_probes": [{"x": 1.0, "d": 0.5, "phase": "probe"}],
        "arm_b_probes": [{"x": 1.0, "d": 2.0, "phase": "probe"},
                         {"x": -1.0, "d": 0.4, "phase": "probe"}],
    }
    p = tmp_path / "one.jsonl"
    p.write_text(json.dumps(row) + "\n")
    out = pb2.verify_input(p)
    assert out["denominator"] == 1
    assert out["arm_a_short_circuit"] == 1
    assert out["arm_b_short_circuit"] == 0
    assert out["union_pairs"] == 1
    assert out["genuine_break_of_arm_instances"] == 1
