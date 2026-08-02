# SPDX-License-Identifier: MIT
"""ddm_sv1 — guards for the emitted termination of the (a,b) photometric GN.

The loop lived in byte-identical copies at ``experiments/ddm_v4c_resolve.py``
(rung B) and ``experiments/ddm_v4d_resolve.py`` (``_refit_ab``) and had three
distinct ways to stop that it collapsed onto one ``if not accepted: break``:
the damping ladder running out, the relinearization count running out, and a
singular normal system.  None was recorded, so a solve that RAN OUT was
indistinguishable in the receipt from one that CONVERGED.

MEASURED on the hardest 60 pairs (86.6% of post-GN pose mass), n600 v4c cache,
canary EXACT against the shipped ``d_rungB``: 0% converged, ~64% damp_cap,
~36% relin_cap -- the shipped solve stopped on a BOUND on 100% of the
mass-carrying pairs.

These are BEHAVIOUR tests.  Each drives the solver to a specific exit and
asserts the exit it reports; each would fail if the discriminator collapsed
back to a single branch, if the extraction changed the arithmetic, or if the
census counted an untraced row as converged.
"""

from __future__ import annotations

import importlib.util
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
    return _load("ddm_v4c_resolve_sv1", "experiments/ddm_v4c_resolve.py")


TP = np.zeros(6, dtype=np.float64)


def _mse(p6: np.ndarray) -> float:
    return float(np.mean((p6 - TP) ** 2))


def _quadratic_pose6(a: float, b: float) -> np.ndarray:
    """A smooth well-posed residual with its minimum at (a,b) = (1.5, 4.0)."""
    out = np.zeros(6, dtype=np.float64)
    out[0] = (a - 1.5) * 0.8
    out[1] = (b - 4.0) * 0.05
    return out


# ----------------------------------------------------------------- exits ----
def test_descending_problem_reports_relin_cap_not_convergence(v4c):
    """A problem still descending at the last relin must NOT read as converged."""
    a, b, _c6, curB, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, _quadratic_pose6(0.0, 0.0),
        _mse(_quadratic_pose6(0.0, 0.0)), TP, relins=2, damp_levels=4)
    assert tr["stop_reason"] == "relin_cap"
    assert tr["n_relin"] == 2
    assert curB < _mse(_quadratic_pose6(0.0, 0.0))
    assert tr["relins_bound"] == 2


def test_more_relins_descend_further_so_the_cap_was_censoring(v4c):
    """Raising only the relin bound must strictly improve -- proving it bound."""
    start = _quadratic_pose6(0.0, 0.0)
    _a1, _b1, _c1, d_short, t_short = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=1, damp_levels=4)
    _a2, _b2, _c2, d_long, _t_long = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=8, damp_levels=4)
    assert t_short["stop_reason"] == "relin_cap"
    assert d_long < d_short


def _cliff(a: float, b: float) -> np.ndarray:
    """Improves ONLY at the two FD probe points, so the linearization always lies.

    The Jacobian is large and points somewhere the function is not actually
    better, so no damping level can accept -- the classic case where a truncated
    ladder is indistinguishable from a converged solve.
    """
    out = np.ones(6, dtype=np.float64)
    if b == 0.0 and abs(a - 1.02) < 1e-15:
        out *= 0.1
    if a == 1.0 and abs(b - 2.0) < 1e-15:
        out *= 0.1
    return out


def test_truncated_damping_ladder_reports_damp_cap_not_convergence(v4c):
    """A ladder cut off while the step is still above f16 resolution is CENSORED."""
    start = _cliff(1.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _cliff, _mse, 1.0, 0.0, start, _mse(start), TP, relins=4, damp_levels=1)
    assert tr["stop_reason"] == "damp_cap"
    assert tr["n_relin"] == 1
    assert tr["damp_used"] == [1]


def test_a_long_enough_ladder_converts_damp_cap_into_a_proof(v4c):
    """THE CURE, stated as behaviour.

    Damping drives the step toward zero geometrically, so on the SAME problem a
    ladder long enough to push the step below f16 resolution stops reporting
    ``damp_cap`` and starts reporting a real ``converged``.  That is what makes
    ``damp_cap`` a precise censoring signal rather than a shrug: it means the
    ladder was too short, and the required length is bounded by
    log8(step/resolution).
    """
    start = _cliff(1.0, 0.0)
    short = v4c.ab_damped_gn(_cliff, _mse, 1.0, 0.0, start, _mse(start), TP,
                             relins=1, damp_levels=1)[4]
    long = v4c.ab_damped_gn(_cliff, _mse, 1.0, 0.0, start, _mse(start), TP,
                            relins=1, damp_levels=24)[4]
    assert short["stop_reason"] == "damp_cap"
    assert long["stop_reason"] == "converged"
    assert long["damp_used"][0] <= 24


def test_zero_jacobian_is_a_proof_not_a_cap(v4c):
    """A constant residual yields a zero step: no ladder length could help."""
    def const(_a: float, _b: float) -> np.ndarray:
        return np.ones(6, dtype=np.float64)

    start = const(1.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        const, _mse, 1.0, 0.0, start, _mse(start), TP, relins=4, damp_levels=4)
    assert tr["stop_reason"] == "converged"


def test_singular_normal_system_is_not_reported_as_converged(v4c, monkeypatch):
    def boom(*_a, **_k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", boom)
    start = _quadratic_pose6(0.0, 0.0)
    _a, _b, _c6, _d, tr = v4c.ab_damped_gn(
        _quadratic_pose6, _mse, 0.0, 0.0, start, _mse(start), TP,
        relins=4, damp_levels=4)
    assert tr["stop_reason"] == "singular"


def test_the_three_exits_are_actually_distinguished(v4c):
    """Mutation guard: a collapsed discriminator would make these all equal."""
    seen = set()
    for fn, a0, b0, damp in (
        (_quadratic_pose6, 0.0, 0.0, 4),      # still descending  -> relin_cap
        (_cliff, 1.0, 0.0, 1),                # ladder truncated  -> damp_cap
        (_cliff, 1.0, 0.0, 24),               # ladder exhausted  -> converged
    ):
        s = fn(a0, b0)
        seen.add(v4c.ab_damped_gn(fn, _mse, a0, b0, s, _mse(s), TP,
                                  relins=2, damp_levels=damp)[4]["stop_reason"])
    assert seen == {"relin_cap", "damp_cap", "converged"}


@pytest.mark.parametrize(
    ("a", "b", "step", "expect"),
    [
        (1.0, 0.0, None, False),                      # no step was ever computed
        (1e9, 0.0, (1e-9, 1e-9), False),              # f16 overflow -> inf -> nan
        (1.0, 0.0, (np.nan, np.nan), False),          # nonfinite step
        (1.0, 0.0, (0.0, 0.0), True),                 # zero step IS a proof
        (0.0, 0.0, (1e-4, 1e-4), False),              # tiny f16 spacing at 0
        (1.0, 0.0, (1.0, 0.0), False),                # needs BOTH halves small
    ],
)
def test_resolution_proof_never_falsely_claims_convergence(v4c, a, b, step, expect):
    """Every degenerate input must fail toward CENSORED, never toward converged."""
    s = None if step is None else np.asarray(step, dtype=np.float64)
    assert v4c._step_below_f16_resolution(a, b, s) is expect


# ------------------------------------------------- extraction equivalence ----
def _original_inline_loop(pose6, mse, a0, b0, cur6, curB, tp, relins, damp):
    """Verbatim pre-extraction loop, kept as the differential reference."""
    a_p, b_p = a0, b0
    lm = 1.0
    for _ in range(relins):
        p6a = pose6(a_p + 0.02, b_p)
        p6b = pose6(a_p, b_p + 2.0)
        jb = np.stack([(p6a - cur6) / 0.02, (p6b - cur6) / 2.0], 1)
        r = cur6 - tp
        accepted = False
        for _damp in range(damp):
            aa = jb.T @ jb + lm * np.diag(np.maximum(np.diag(jb.T @ jb), 1e-8))
            try:
                step = np.linalg.solve(aa, -(jb.T @ r))
            except np.linalg.LinAlgError:
                break
            for scale in (1.0, 0.5):
                ca, cb = a_p + scale * step[0], b_p + scale * step[1]
                c6 = pose6(ca, cb)
                cv = mse(c6)
                if cv < curB:
                    a_p, b_p, cur6, curB = ca, cb, c6, cv
                    lm = max(lm * 0.3, 1e-4)
                    accepted = True
                    break
            if accepted:
                break
            lm *= 8.0
        if not accepted:
            break
    return a_p, b_p, curB


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_extraction_is_bit_identical_to_the_original_loop(v4c, seed):
    """The refactor must not move a single bit of the shipped solve."""
    rng = np.random.default_rng(seed)
    coeff = rng.normal(size=(6, 2))
    centre = rng.normal(size=2) * np.array([0.5, 20.0]) + np.array([1.0, 0.0])

    def pose6(a: float, b: float) -> np.ndarray:
        return coeff @ np.array([a - centre[0], (b - centre[1]) * 0.05])

    a0, b0 = 1.0, 0.0
    start = pose6(a0, b0)
    ref = _original_inline_loop(pose6, _mse, a0, b0, start, _mse(start),
                               TP, v4c.GN_RELINS_PHOTO, v4c.AB_DAMP_LEVELS)
    got = v4c.ab_damped_gn(pose6, _mse, a0, b0, start, _mse(start), TP)
    assert got[0] == ref[0]
    assert got[1] == ref[1]
    assert got[3] == ref[2]


# ------------------------------------------------------------- the census ----
def test_census_counts_untraced_rows_as_unrecorded_not_converged(v4c):
    c = v4c.ab_stop_census([{"pair": 0}, {"pair": 1, "ab_stop": "damp_cap"}])
    assert c["n_rows"] == 2
    assert c["by_stop_reason"]["unrecorded"] == 1
    assert c["by_stop_reason"]["converged"] == 0
    assert c["n_stopped_on_bound"] == 1


def test_census_fraction_reports_the_denominator(v4c):
    rows = [{"ab_stop": "damp_cap"}] * 3 + [{"ab_stop": "converged"}]
    c = v4c.ab_stop_census(rows)
    assert c["n_rows"] == 4
    assert c["n_stopped_on_bound"] == 3
    assert c["frac_stopped_on_bound"] == pytest.approx(0.75)


def test_census_of_nothing_is_zero_rows_not_a_clean_bill(v4c):
    c = v4c.ab_stop_census([])
    assert c["n_rows"] == 0
    assert c["frac_stopped_on_bound"] == 0.0


# ------------------------------------------------------- the v4d call site ---
def test_v4d_refit_ab_returns_the_trace(v4c):
    v4d = _load("ddm_v4d_resolve_sv1", "experiments/ddm_v4d_resolve.py")
    import inspect
    src = inspect.getsource(v4d._refit_ab)
    assert "v4c.ab_damped_gn" in src, "v4d must use the shared solver, not a copy"
    assert src.rstrip().endswith("trace"), "v4d must return the trace"
