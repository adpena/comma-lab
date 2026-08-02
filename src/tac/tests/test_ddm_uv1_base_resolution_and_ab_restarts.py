# SPDX-License-Identifier: MIT
"""ddm_uv1 — guards for arbitrary base resolution and the derived (a,b) restart policy.

TWO surfaces, one arm.

1.  ``resolve_base`` — the pose solver's base set was a hardcoded 2-entry dict
    (``BASES``) while the BUILDER had already been parametrized, so a base the
    builder could SHIP could never be SOLVED against.  That one-sided bridge is
    what blocked the #827 composition row from being adjudicated on its own
    terms.  These tests pin the resolution contract, including both refusals.

2.  ``derive_ab_starts`` / ``ab_multistart_gn`` — ddm_sv1 measured that the
    START, not the bound, dominates this solver (restarts bought 1.70x the
    bound's gain from 0.07x the pairs, at 0/4 argmin agreement) but its 5-point
    displacement set was an explicit GENERIC control, not a policy.  These
    starts are DERIVED from state the solver already holds and cost zero scorer
    evaluations to construct.

    The load-bearing guard is ``test_neutral_only_is_byte_identical``: with the
    default policy the multistart wrapper must reproduce the shipped
    single-start solve EXACTLY, so enabling the surface cannot perturb the
    sealed config.  Without it, "opt-in" is a claim rather than a property.
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
    sys.path.insert(0, str(_REPO / "experiments"))
    try:
        return _load("ddm_v4c_resolve_uv1", "experiments/ddm_v4c_resolve.py")
    except Exception as exc:  # pragma: no cover - env guard
        pytest.skip(f"v4c not importable: {exc}")


# --------------------------------------------------------------------------- #
# 1. base resolution
# --------------------------------------------------------------------------- #
def test_registered_label_resolves_to_its_registered_path(v4c):
    for label, path in v4c.BASES.items():
        assert v4c.resolve_base(label) == (label, path)


def test_unknown_label_without_archive_refuses(v4c):
    with pytest.raises(SystemExit) as ei:
        v4c.resolve_base("w03_ep854_representative")
    assert "--base-archive" in str(ei.value)


def test_arbitrary_archive_resolves_under_any_label(v4c, tmp_path):
    arc = tmp_path / "some_base_archive.zip"
    arc.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    label, got = v4c.resolve_base("w03_ep854_representative", arc)
    assert (label, got) == ("w03_ep854_representative", arc)


def test_missing_archive_refuses_rather_than_resolving(v4c, tmp_path):
    with pytest.raises(SystemExit) as ei:
        v4c.resolve_base("x", tmp_path / "absent.zip")
    assert "does not exist" in str(ei.value)


def test_explicit_archive_overrides_a_registered_label(v4c, tmp_path):
    """A registered label must not silently win over an explicit path."""
    arc = tmp_path / "override.zip"
    arc.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    assert v4c.resolve_base("celldrop50", arc)[1] == arc


def test_dead_rs_betas_menu_is_gone(v4c):
    """The unconsumed 5-entry sweep must not read as a live, audited menu."""
    assert not hasattr(v4c, "RS_BETAS")
    assert v4c.RS_GLOBAL_G == (0.5, 1.0)


# --------------------------------------------------------------------------- #
# 2. the derived start set
# --------------------------------------------------------------------------- #
def test_neutral_is_always_present(v4c):
    f0 = np.zeros((4, 4), np.float64)   # degenerate: sd == 0
    starts = v4c.derive_ab_starts(f0, np.ones((4, 4), np.float64), 0, None)
    assert starts["neutral"] == (1.0, 0.0)


def test_degenerate_variance_omits_moment_match_instead_of_dividing_by_zero(v4c):
    starts = v4c.derive_ab_starts(np.full((4, 4), 7.0), np.arange(16.0).reshape(4, 4),
                                  0, None)
    assert "moment_match" not in starts
    assert starts == {"neutral": (1.0, 0.0)}


def test_moment_match_maps_the_first_two_moments_exactly(v4c):
    rng = np.random.default_rng(11)
    f0 = rng.normal(40.0, 12.0, (16, 16))
    f1 = rng.normal(110.0, 30.0, (16, 16))
    a, b = v4c.derive_ab_starts(f0, f1, 0, None)["moment_match"]
    out = a * f0 + b
    assert out.mean() == pytest.approx(f1.mean(), rel=1e-9)
    assert out.std() == pytest.approx(f1.std(), rel=1e-9)


def test_neighbour_start_is_the_previous_pair_regardless_of_selector(v4c):
    hist = [(0, 1.10, 5.0), (1, 0.90, -3.0)]
    starts = v4c.derive_ab_starts(np.zeros((2, 2)), np.zeros((2, 2)), 0, hist)
    assert starts["neighbour"] == (0.90, -3.0)


def test_sel_median_uses_only_the_matching_selector(v4c):
    hist = [(0, 1.0, 10.0), (1, 5.0, 500.0), (0, 3.0, 30.0), (1, 6.0, 600.0)]
    starts = v4c.derive_ab_starts(np.zeros((2, 2)), np.zeros((2, 2)), 0, hist)
    assert starts["sel_median"] == (2.0, 20.0)          # median of the sel==0 rows
    starts1 = v4c.derive_ab_starts(np.zeros((2, 2)), np.zeros((2, 2)), 1, hist)
    assert starts1["sel_median"] == (5.5, 550.0)


def test_no_history_yields_no_history_derived_starts(v4c):
    starts = v4c.derive_ab_starts(np.arange(16.0).reshape(4, 4),
                                  np.arange(16.0).reshape(4, 4), 0, [])
    assert "neighbour" not in starts and "sel_median" not in starts


# --------------------------------------------------------------------------- #
# 3. the multistart wrapper
# --------------------------------------------------------------------------- #
def _quadratic_problem(a_star: float, b_star: float):
    """A separable quadratic in (a,b) whose unique minimiser is (a_star,b_star)."""
    tp = np.zeros(6, np.float64)

    def pose6(a: float, b: float) -> np.ndarray:
        out = np.zeros(6, np.float64)
        out[0] = (a - a_star)
        out[1] = (b - b_star) * 0.01
        return out

    def mse(p6: np.ndarray) -> float:
        return float(np.mean((p6 - tp) ** 2))

    return pose6, mse, tp


def _bimodal_problem():
    """Two wells: a SHALLOW one at the neutral start, a DEEP one away from it.

    A separable quadratic is the wrong fixture for a restart test -- Gauss-Newton
    solves a linear residual exactly from ANY start, so every start ties and the
    test cannot distinguish a working policy from a broken one.  The regime sv1
    actually measured is multi-modal ("the per-start spread is large and
    multi-modal (pair 16: 0.638 -> 2.547 across starts)"), which is precisely
    when a start matters.  Objective, encoded through one residual component so
    ``mse`` reproduces it exactly:

        f(a,b) = 1.0 - 0.4*exp(-[(a-1)^2 + (b/50)^2]/0.2)      # shallow, at neutral
                     - 0.9*exp(-[(a-3)^2 + ((b-90)/50)^2]/0.2)  # deep, far away

    f(1,0) = 0.6 (local), f(3,90) = 0.1 (global); both strictly positive.
    """
    tp = np.zeros(6, np.float64)

    def f(a: float, b: float) -> float:
        g1 = np.exp(-(((a - 1.0) ** 2) + ((b - 0.0) / 50.0) ** 2) / 0.2)
        g2 = np.exp(-(((a - 3.0) ** 2) + ((b - 90.0) / 50.0) ** 2) / 0.2)
        return float(1.0 - 0.4 * g1 - 0.9 * g2)

    def pose6(a: float, b: float) -> np.ndarray:
        out = np.zeros(6, np.float64)
        out[0] = np.sqrt(6.0 * f(a, b))
        return out

    def mse(p6: np.ndarray) -> float:
        return float(np.mean((p6 - tp) ** 2))

    return pose6, mse, tp


def test_neutral_only_is_byte_identical_to_the_shipped_single_start(v4c):
    """The default policy must not perturb the sealed solve -- BIT-identical."""
    for a_star, b_star in ((1.4, 6.0), (0.7, -12.0), (1.0, 0.0), (2.5, 40.0)):
        pose6, mse, tp = _quadratic_problem(a_star, b_star)
        cur6 = pose6(1.0, 0.0)
        d0 = mse(cur6)
        a_s, b_s, _c, _v, tr_s = v4c.ab_damped_gn(pose6, mse, 1.0, 0.0, cur6, d0, tp)
        shipped = (float(np.float16(a_s)), float(np.float16(b_s)))
        a_m, b_m, d_m, tr_m = v4c.ab_multistart_gn(
            pose6, mse, cur6, d0, tp, {"neutral": (1.0, 0.0)})
        assert (a_m, b_m) == shipped
        assert d_m == mse(pose6(*shipped))
        assert tr_m["stop_reason"] == tr_s["stop_reason"]
        assert tr_m["start"] == "neutral"
        assert tr_m["starts_tried"] == ["neutral"]


def test_multistart_never_loses_to_neutral(v4c):
    """Acceptance is on the f16 values that ship, so it can only improve.

    Carries its own NON-VACUITY guard: on a bimodal objective at least one
    displaced start must strictly win, otherwise this test would pass just as
    well against a wrapper that ignored every start but ``neutral``.
    """
    rng = np.random.default_rng(5)
    strict_wins = 0
    for _ in range(8):
        pose6, mse, tp = _bimodal_problem()
        cur6 = pose6(1.0, 0.0)
        d0 = mse(cur6)
        cand = (float(rng.uniform(2.6, 3.4)), float(rng.uniform(75.0, 105.0)))
        _a, _b, d_neutral, _t = v4c.ab_multistart_gn(
            pose6, mse, cur6, d0, tp, {"neutral": (1.0, 0.0)})
        _a2, _b2, d_multi, _t2 = v4c.ab_multistart_gn(
            pose6, mse, cur6, d0, tp, {"neutral": (1.0, 0.0), "moment_match": cand})
        assert d_multi <= d_neutral + 1e-15
        strict_wins += int(d_multi < d_neutral - 1e-12)
    assert strict_wins > 0, "fixture is vacuous: no start ever strictly won"


def test_a_better_start_wins_and_is_named(v4c):
    """The deep well is unreachable from neutral; only a restart finds it."""
    pose6, mse, tp = _bimodal_problem()
    cur6 = pose6(1.0, 0.0)
    d0 = mse(cur6)
    _a, _b, d_neutral, tr_n = v4c.ab_multistart_gn(
        pose6, mse, cur6, d0, tp, {"neutral": (1.0, 0.0)})
    _a2, _b2, d_multi, tr = v4c.ab_multistart_gn(
        pose6, mse, cur6, d0, tp,
        {"neutral": (1.0, 0.0), "moment_match": (3.0, 90.0)})
    assert tr_n["start"] == "neutral"
    assert tr["start"] == "moment_match"
    assert sorted(tr["starts_tried"]) == ["moment_match", "neutral"]
    assert d_multi < d_neutral          # the restart is what found the deep well
    assert d_multi == pytest.approx(0.1, abs=1e-3)
    assert d_neutral == pytest.approx(0.6, abs=1e-3)


def test_ties_go_to_neutral_so_the_shipped_answer_is_preferred(v4c):
    pose6, mse, tp = _quadratic_problem(1.0, 0.0)   # neutral is already optimal
    cur6 = pose6(1.0, 0.0)
    d0 = mse(cur6)
    _a, _b, _d, tr = v4c.ab_multistart_gn(
        pose6, mse, cur6, d0, tp,
        {"neutral": (1.0, 0.0), "moment_match": (1.0, 0.0),
         "neighbour": (1.0, 0.0)})
    assert tr["start"] == "neutral"


# --------------------------------------------------------------------------- #
# 4. the activation ledger
# --------------------------------------------------------------------------- #
def test_start_census_reports_the_denominator(v4c):
    rows = [{"ab_start": "neutral"}, {"ab_start": "moment_match"},
            {"ab_start": "sel_median"}, {"ab_start": "neutral"}]
    cen = v4c._ab_start_census(rows)
    assert cen["n_rows"] == 4
    assert cen["by_start"] == {"neutral": 2, "moment_match": 1, "sel_median": 1}
    assert cen["n_non_neutral_winner"] == 2
    assert cen["frac_non_neutral_winner"] == pytest.approx(0.5)


def test_start_census_reads_untraced_rows_as_neutral_not_as_a_win(v4c):
    """Pre-policy caches were neutral-start solves; they must not inflate wins."""
    cen = v4c._ab_start_census([{}, {}, {"ab_start": "neighbour"}])
    assert cen["by_start"]["neutral"] == 2
    assert cen["n_non_neutral_winner"] == 1


def test_start_census_on_empty_scope_reports_zero_not_a_clean_bill(v4c):
    cen = v4c._ab_start_census([])
    assert cen["n_rows"] == 0
    assert cen["frac_non_neutral_winner"] == 0.0
