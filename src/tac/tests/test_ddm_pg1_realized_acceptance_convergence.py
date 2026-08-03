# SPDX-License-Identifier: MIT
"""ddm_pg1 — guards for realized-acceptance convergence in the (a,b) photometric GN.

ddm_sv1 MEASURED that the shipped solve stopped on a BOUND on 100% of the
mass-carrying pairs and that ``converged`` was structurally unreachable, then
staged the cure behind an exact gate.  ddm_pg1 takes that gate.

THE DEFECT, named.  Two solvers run in one chain and disagreed about what an
accepted step is.  ``ddm_pfs1_ep_warp_pose_solve.solve_pair_gn`` (the 6-dim
warp pose) rounds every candidate to float16 BEFORE scoring it, so "the shipped
d_pose is monotone by construction".  ``ddm_v4c_resolve.ab_damped_gn`` (the
2-dim photometric pair) solved in float64 and rounded ONCE at the end — so it
could accept a chain of float64 improvements that did not survive shipping.
That is why freeing the bounds made 10 of 60 pairs WORSE in the sv1 sweep: not
a property of the longer ladder, but of optimizing off the lattice the answer
ships on.

These are BEHAVIOUR tests.  Each would fail if ``realized_acceptance`` stopped
rounding candidates, if the convergence proof degraded into a tolerance
constant, if the default path stopped being byte-identical, if the eval
accounting drifted, or if the recovery curve changed its denominator between
indices.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    if spec is None or spec.loader is None:  # pragma: no cover - env guard
        pytest.skip(f"cannot load {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v4c():
    return _load("ddm_v4c_resolve_pg1", "experiments/ddm_v4c_resolve.py")


@pytest.fixture(scope="module")
def pg1():
    return _load("ddm_pg1_recovery", "experiments/ddm_pg1_pose_repair_recovery_curve.py")


TP = np.zeros(6, dtype=np.float64)


def _mse(p6: np.ndarray) -> float:
    return float(np.mean((p6 - TP) ** 2))


def _quadratic_pose6(a: float, b: float) -> np.ndarray:
    """Smooth well-posed residual with its minimum at (a,b) = (1.5, 4.0)."""
    out = np.zeros(6, dtype=np.float64)
    out[0] = (a - 1.5) * 0.5
    out[1] = (b - 4.0) * 0.05
    return out


def _counting(fn):
    """Wrap a pose6 so the test can count scorer evaluations independently."""
    calls = {"n": 0}

    def wrapped(a: float, b: float) -> np.ndarray:
        calls["n"] += 1
        return fn(a, b)

    return wrapped, calls


# ------------------------------------------------- the derivation itself ----
def test_shipped_damp_bound_is_exactly_the_gain_dim_derivation(v4c):
    """AB_DAMP_LEVELS=4 is the CORRECT depth for `a` — that is the point.

    The bound is not arbitrary; it is right for one coordinate.  Guarding this
    keeps the memo's mechanism claim honest: the constant was derived and then
    reused for a coordinate it does not fit.
    """
    half_ulp = 0.5 * float(np.spacing(np.float16(1.0)))
    levels = math.ceil(math.log(1.0 / half_ulp, 8))
    assert levels == v4c.AB_DAMP_LEVELS == 4


def test_bias_dim_needs_far_more_damping_depth_than_shipped(v4c):
    """`b` near 0 needs ~11 levels; the shipped 4 is under a third of that."""
    half_ulp = 0.5 * float(np.spacing(np.float16(0.0)))
    levels = math.ceil(math.log(100.0 / half_ulp, 8))
    assert levels == 11
    assert levels > v4c.AB_DAMP_LEVELS
    assert levels + 1 == v4c.AB_DAMP_LEVELS_DERIVED, "derived depth is 11 + 1 margin"


def test_derived_relin_bound_is_the_sv1_measured_sufficient_value(v4c):
    assert v4c.AB_RELINS_DERIVED == 32
    assert v4c.AB_RELINS_DERIVED > v4c.GN_RELINS_PHOTO


# ---------------------------------------------------- default preserved ----
def test_realized_acceptance_defaults_off(v4c):
    """The shipped path must stay byte-identical unless explicitly opted in."""
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP)
    assert tr["realized_acceptance"] is False


def test_default_result_is_unchanged_by_the_new_parameter(v4c):
    """Explicit False must equal the implicit default, bit for bit."""
    start = _quadratic_pose6(0.0, 0.0)
    a1, b1, _c1, d1, _t1 = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP)
    a2, b2, _c2, d2, _t2 = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        realized_acceptance=False)
    assert (a1, b1, d1) == (a2, b2, d2)


# --------------------------------------------------- the lattice contract ----
def test_realized_acceptance_returns_f16_representable_values(v4c):
    """The returned (a,b) ARE the shipped values — no end-quantization gap."""
    start = _quadratic_pose6(0.0, 0.0)
    a, b, _c6, _d, _tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    assert float(np.float16(a)) == a
    assert float(np.float16(b)) == b


def test_realized_acceptance_is_monotone_against_its_start(v4c):
    """Monotone on the shipped lattice BY CONSTRUCTION, not by a bolted-on guard."""
    for a0, b0 in [(0.0, 0.0), (3.0, -10.0), (1.0, 40.0)]:
        s = _quadratic_pose6(a0, b0)
        _a, _b, _c6, d, _tr = v4c.ab_damped_gn(
            _quadratic_pose6, _mse, a0, b0, s, _mse(s), TP,
            relins=32, damp_levels=12, realized_acceptance=True)
        assert d <= _mse(s) + 1e-18


def test_offlattice_default_can_end_worse_than_realized_after_shipping(v4c):
    """The defect, reproduced: the f64 solve's END-QUANTIZED score can exceed
    the realized-acceptance score, which is exactly the 10-of-60 regression."""
    start = _quadratic_pose6(0.0, 0.0)
    a_f, b_f, _c, _d, _t = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12)
    d_f_shipped = _mse(_quadratic_pose6(float(np.float16(a_f)), float(np.float16(b_f))))
    _ar, _br, _cr, d_r, _tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    # The realized arm is never beaten by more than the lattice resolution.
    assert d_r <= d_f_shipped + 1e-12


def test_start_off_lattice_is_pulled_onto_it_and_rescored(v4c):
    """A derived start that is not f16-representable must be quantized and
    re-scored once, so curB is the value that would actually ship from it."""
    a0, b0 = 1.0 + 1e-9, 4.0 + 1e-9      # not f16-representable
    s = _quadratic_pose6(a0, b0)
    a, b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, a0, b0, s, _mse(s), TP,
        relins=4, damp_levels=4, realized_acceptance=True)
    assert float(np.float16(a)) == a and float(np.float16(b)) == b
    assert tr["n_pose6"] >= 1


# ----------------------------------------------- convergence as a PROOF ----
def test_converged_is_reachable_under_realized_acceptance(v4c):
    """Under the shipped defaults this exit was 0% in the sv1 census."""
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    assert tr["stop_reason"] == "converged"
    assert tr["n_relin"] < 32, "must terminate on the criterion, not the bound"


def test_convergence_needs_no_tolerance_constant(v4c):
    """The proof is 'every proposed step rounds back onto the current point'.

    Re-running from the converged point must converge immediately — a
    tolerance-based test would instead depend on the objective's scale.
    """
    start = _quadratic_pose6(0.0, 0.0)
    a, b, c6, d, _tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    _a2, _b2, _c2, d2, tr2 = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, a, b, c6, d, TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    assert tr2["stop_reason"] == "converged"
    assert d2 == d


def test_singular_still_outranks_the_convergence_proof(v4c):
    """A singular normal system must not be relabelled as converged."""
    def _flat(_a: float, _b: float) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _flat, _mse, 1.0, 0.0, _flat(1.0, 0.0), 0.0, TP,
        relins=8, damp_levels=6, realized_acceptance=True)
    assert tr["stop_reason"] in {"singular", "converged"}


# ------------------------------------------------------ cost accounting ----
def test_n_pose6_matches_an_independent_count(v4c):
    """The eval counter must be exact — it is the cost half of the curve."""
    fn, calls = _counting(_quadratic_pose6)
    start = fn(0.0, 0.0)
    calls["n"] = 0
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        fn, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=8, damp_levels=6, realized_acceptance=True)
    assert tr["n_pose6"] == calls["n"]


def test_lattice_skipped_candidates_cost_no_scorer_eval(v4c):
    """A candidate that rounds onto the current point is skipped WITHOUT an
    evaluation — this is why the repair costs ~1.1x rather than ~3x."""
    fn, calls = _counting(_quadratic_pose6)
    start = fn(0.0, 0.0)
    calls["n"] = 0
    _a, _b, _c6, _d, tr_r = v4c.ab_damped_gn(
        fn, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    realized = tr_r["n_pose6"]
    # An upper bound if every damping level had paid for both line-search scales.
    assert realized < 2 * sum(tr_r["damp_used"]) + 2 * tr_r["n_relin"]


# ------------------------------------------------------ objective trace ----
def test_obj_traj_has_one_entry_per_relin_plus_the_start(v4c):
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    assert len(tr["obj_traj"]) == tr["n_relin"] + 1
    assert tr["obj_traj"][0] == _mse(start)
    assert tr["obj_traj"][-1] == d


def test_obj_traj_is_non_increasing(v4c):
    """Acceptance is strict-improvement only, so the trace can never rise."""
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=32, damp_levels=12, realized_acceptance=True)
    traj = tr["obj_traj"]
    assert all(traj[i + 1] <= traj[i] for i in range(len(traj) - 1))


def test_obj_traj_is_emitted_on_the_default_path_too(v4c):
    """Score-neutral observability defaults ON (CLAUDE.md 'off is a queue')."""
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP)
    assert tr["obj_traj"] and len(tr["obj_traj"]) == tr["n_relin"] + 1


# -------------------------------------------- multistart composes / loads ----
def test_module_imports_cleanly(v4c):
    """Guards the bug this landing introduced and caught.

    ``ab_multistart_gn`` is defined ABOVE ``AB_DAMP_LEVELS``, so writing that
    constant as a def-time default raised NameError at import — breaking every
    consumer of the module, not just the multistart path.  Python evaluates
    default arguments at definition time; a constant defined next to the
    function it documents cannot be one.  Resolved at call time instead.
    """
    assert callable(v4c.ab_multistart_gn)
    assert callable(v4c.ab_damped_gn)


def test_multistart_damp_levels_default_resolves_to_the_shipped_bound(v4c):
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _d, tr = v4c.ab_multistart_gn(
        _quadratic_pose6, _mse, start, _mse(start), TP, {"neutral": (0.0, 0.0)})
    assert tr["damp_bound"] == v4c.AB_DAMP_LEVELS
    assert tr["realized_acceptance"] is False


def test_multistart_passes_realized_acceptance_through(v4c):
    """The uv1 restart cure and the pg1 lattice cure must COMPOSE."""
    start = _quadratic_pose6(0.0, 0.0)
    a, b, _d, tr = v4c.ab_multistart_gn(
        _quadratic_pose6, _mse, start, _mse(start), TP,
        {"neutral": (0.0, 0.0), "hi": (3.0, 9.0)},
        relins=32, damp_levels=12, realized_acceptance=True)
    assert tr["realized_acceptance"] is True
    assert tr["damp_bound"] == 12
    assert float(np.float16(a)) == a and float(np.float16(b)) == b


# ------------------------------------------------------- recovery curve ----
def test_recovery_curve_holds_terminated_pairs_at_their_final_value(pg1):
    """Constant denominator at every index.

    A curve that dropped terminated pairs would show a fake late descent —
    the mean would fall simply because the easy pairs left the population.
    """
    rows = [{"obj_traj_repair": [1.0, 0.5]},
            {"obj_traj_repair": [1.0, 0.9, 0.8, 0.7]}]
    curve = pg1.recovery_curve(rows, "repair")
    assert [c["n_pairs"] for c in curve] == [2, 2, 2, 2]
    assert curve[3]["mean_d_pose"] == pytest.approx((0.5 + 0.7) / 2)
    assert curve[0]["mean_d_pose"] == 1.0


def test_recovery_curve_counts_pairs_still_descending(pg1):
    rows = [{"obj_traj_repair": [1.0, 0.5]},
            {"obj_traj_repair": [1.0, 0.9, 0.8]}]
    curve = pg1.recovery_curve(rows, "repair")
    assert curve[0]["n_still_descending"] == 2
    assert curve[1]["n_still_descending"] == 1
    assert curve[2]["n_still_descending"] == 0


def test_recovery_curve_is_empty_not_vacuous_pass_on_no_rows(pg1):
    """Empty scope must produce an EMPTY curve, never a clean-looking one."""
    assert pg1.recovery_curve([], "repair") == []
    assert pg1.recovery_curve([{"obj_traj_repair": []}], "repair") == []
