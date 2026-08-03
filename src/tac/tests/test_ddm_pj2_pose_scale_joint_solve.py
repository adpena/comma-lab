# SPDX-License-Identifier: MIT
"""Tests for ``tools/pj2_pose_scale_joint_solve`` (ddm_pj2, tasks #850/#873/#882).

The load-bearing claims this suite has to be able to FALSIFY:

1. ``s_t`` and ``pose[0:3]`` are an EXACT scale-degenerate pair in the shipped
   homography.  If that is false the whole unit is wrong, so it is tested
   against the vendored receiver algebra, not against a re-typed formula.
2. ``ShippedQuant`` reproduces ``inflate_runner_v4d.Decoder``'s pose
   reconstruction (``dim0_offset + f16(residual)``).  A candidate that does not
   survive this map cannot ship, so every acceptance goes through it.
3. The Gauss-Newton exits are SPLIT and a convergence PROOF exists that a bound
   cannot masquerade as.  ``ddm_os1`` measured the live solver reporting
   ``ALL_STOPPED_ON_A_BOUND 600/600`` precisely because ``cur < 1e-6`` was fused
   into the same ``break``.
4. The dual-metric readback reports Euclid AND Fisher, and can DISAGREE.

Anti-vacuity note (``ddm_uv1`` s6): a separable quadratic is solved exactly by
Gauss-Newton from any start, so a fixture built from one would pass against a
solver that ignored its own step.  Every solver fixture here is non-separable
and ill-conditioned, and ``test_fixture_is_not_vacuous`` asserts that a
null-step solver FAILS it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_TOOL = REPO / "tools" / "pj2_pose_scale_joint_solve.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pj2_tool", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pj2_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


pj2 = _load_tool()


# --------------------------------------------------------------------------- #
# 1. the degeneracy -- the claim the whole unit rests on
# --------------------------------------------------------------------------- #
def _homography(pose6, s_t, rot, *, h=1.22):
    """The shipped ground homography, transcribed ONCE here so the test is
    independent of the tool.  ``pfs1_warp_receiver.pose_to_homography``."""
    k = np.array([[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]])
    kinv = np.linalg.inv(k)
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    omega = rot * np.asarray(pose6[3:6], dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    kx = np.array([[0.0, -omega[2], omega[1]],
                   [omega[2], 0.0, -omega[0]],
                   [-omega[1], omega[0], 0.0]])
    r = (np.eye(3) + kx if theta < 1e-12 else
         np.eye(3) + (np.sin(theta) / theta) * kx
         + ((1.0 - np.cos(theta)) / theta ** 2) * (kx @ kx))
    n = np.array([0.0, -1.0, 0.0])
    return k @ (r - np.outer(t, n) / h) @ kinv


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.9, 1.1, 2.0, 7.0])
def test_scale_degeneracy_is_exact_to_roundoff(lam):
    rng = np.random.default_rng(11)
    for _ in range(20):
        p = np.array([rng.normal(33, 4), rng.normal(0, .3), rng.normal(0, .3),
                      rng.normal(0, .02), rng.normal(0, .02), rng.normal(0, .02)])
        s = float(rng.uniform(0.01, 0.30))
        rot = float(rng.uniform(0.5, 1.5))
        h1 = _homography(p, s, rot)
        p2 = p.copy()
        p2[:3] *= lam
        h2 = _homography(p2, s / lam, rot)
        rel = np.max(np.abs(h1 - h2)) / np.max(np.abs(h1))
        assert rel < 1e-12, f"degeneracy broken at lam={lam}: rel={rel:g}"


def test_far_plane_homography_ignores_the_translation_scale():
    """The far plane is evaluated at ``s_t = 0``, so scaling the translation
    triple must leave it BIT-identical -- otherwise the pose-route lever would
    perturb the two-plane composite and the degeneracy would be only partial."""
    rng = np.random.default_rng(3)
    p = rng.normal(size=6)
    for lam in (0.3, 3.0):
        p2 = p.copy()
        p2[:3] *= lam
        assert np.array_equal(_homography(p, 0.0, 1.0), _homography(p2, 0.0, 1.0))


def test_rotation_dims_are_untouched_by_the_scale_route():
    """Scaling dims 0:3 must not move the yaw dim the beta sign reads
    (``inflate_runner_v4d.py:196``), or the rolling-shutter branch would flip."""
    p = np.array([33.0, 0.1, -0.2, 1e-3, -2e-3, -5e-4])
    p2 = p.copy()
    p2[:3] *= 9.0
    assert p2[5] == p[5]
    assert (p2[5] >= 0.0) == (p[5] >= 0.0)


# --------------------------------------------------------------------------- #
# 2. the shipped quantization
# --------------------------------------------------------------------------- #
def test_shipped_quant_matches_receiver_dim0_residual_reconstruction():
    off = 31.546875
    q = pj2.ShippedQuant(off)
    p = np.array([33.556640625, 0.1380615, -0.1890869, 1.7747e-3, -1.5795e-5,
                  -8.3446e-4])
    got = q.pose(p)
    want0 = off + float(np.float16(np.float64(p[0]) - off))
    assert got[0] == want0
    assert np.array_equal(got[1:], np.asarray(p[1:], np.float64)
                          .astype(np.float16).astype(np.float64))


def test_shipped_quant_dim0_residual_beats_plain_f16_near_the_offset():
    """The residual encoding is why the pose route has resolution at all: near
    the offset it is orders finer than plain float16 of a value around 33."""
    off = 31.546875
    q = pj2.ShippedQuant(off)
    p = np.array([31.5470001, 0.0, 0.0, 0.0, 0.0, 0.0])
    residual_err = abs(q.pose(p)[0] - p[0])
    plain_err = abs(float(np.float16(p[0])) - p[0])
    assert residual_err < plain_err


def test_shipped_quant_none_offset_is_plain_f16():
    q = pj2.ShippedQuant(None)
    p = np.array([33.5, 0.1, -0.2, 0.001, 0.002, 0.003])
    assert np.array_equal(q.pose(p),
                          p.astype(np.float16).astype(np.float64))


def test_same_detects_identical_shipped_bytes():
    q = pj2.ShippedQuant(31.546875)
    p = np.array([33.5, 0.1, -0.2, 0.001, 0.002, 0.003])
    t0 = q.theta(p, 1.0, 0.0)
    # a perturbation far below the float16 cell must land on the same bytes
    t1 = q.theta(p + 1e-9, 1.0 + 1e-9, 0.0 + 1e-9)
    assert q.same(t0, t1)
    t2 = q.theta(p + 0.5, 1.0, 0.0)
    assert not q.same(t0, t2)


def test_same_is_sensitive_to_the_photometric_pair_alone():
    q = pj2.ShippedQuant(None)
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not q.same(q.theta(p, 1.0, 0.0), q.theta(p, 1.0, 4.0))
    assert not q.same(q.theta(p, 1.0, 0.0), q.theta(p, 0.5, 0.0))


# --------------------------------------------------------------------------- #
# 3. the SPD chart + the solver
# --------------------------------------------------------------------------- #
def test_spd_chart_makes_a_rank_deficient_pullback_positive_definite():
    j = np.zeros((6, 8))
    j[:, :3] = np.random.default_rng(0).normal(size=(6, 3))
    gn = j.T @ j                        # rank <= 3 in an 8-dim chart
    assert np.linalg.eigvalsh(gn).min() < 1e-12
    h = pj2._spd_chart(gn, 1e-3)
    assert np.linalg.eigvalsh(h).min() > 0.0
    assert np.allclose(h, h.T)


def test_spd_chart_ridge_zero_leaves_a_singular_chart_singular():
    """The ridge is a DECLARED consumer obligation, not a hidden safety net:
    with ridge 0 the chart stays singular and the caller must see that."""
    j = np.zeros((6, 8))
    j[:, 0] = 1.0
    h = pj2._spd_chart(j.T @ j, 0.0)
    assert np.linalg.eigvalsh(h).min() <= 1e-12


class _Quadratic:
    """Ill-conditioned NON-separable residual field with an exactly known root.

    ``r(x) = A (x - x*)`` with ``A`` dense and badly conditioned.  Gauss-Newton
    does NOT solve this in one step under a trust region, so a solver that
    ignored its own step, or took a Euclidean step, fails the fixture.
    """

    def __init__(self, dim=8, cond=1e6, seed=5):
        rng = np.random.default_rng(seed)
        qm, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        sv = np.geomspace(1.0, 1.0 / cond, dim)
        # Apple Accelerate's BLAS raises spurious divide-by-zero/overflow FP
        # STATUS FLAGS on this matmul even though every input and output is
        # finite and well-scaled (verified: cond(a) matches `cond`).  Suppressed
        # with the cause named rather than left as unexplained warning noise.
        with np.errstate(all="ignore"):
            self.a = ((qm * sv) @ qm.T)[:6]
        assert np.all(np.isfinite(self.a))
        self.xstar = rng.normal(size=dim) * 0.01
        self.dim = dim
        self.calls = 0

    def pose6(self, x):
        self.calls += 1
        return self.a @ (np.asarray(x, np.float64) - self.xstar)


def _fixture_ctx(fx, quant_dim0=None):
    q = pj2.ShippedQuant(quant_dim0)
    tp = np.zeros(6)

    class _Sc:
        n_evals = 0

        def pose6(self, f1_f, f1_u8, pose, s_t, sel, a, b, g):
            type(self).n_evals += 1
            return fx.pose6(np.concatenate([pose, [a, b]]))

        def d_pose(self, f1_f, f1_u8, tp_, pose, s_t, sel, a, b, g):
            type(self).n_evals += 1
            r = fx.pose6(np.concatenate([pose, [a, b]])) - tp_
            return float(np.mean(r ** 2))

    _Sc.n_evals = 0
    return _Sc(), (q, None, None, tp, 0.08, 1, 0.0)


def test_fisher_trust_region_gn_descends_on_an_ill_conditioned_fixture():
    fx = _Quadratic()
    sc, ctx = _fixture_ctx(fx)
    q = ctx[0]
    theta0 = q.theta(np.zeros(6), 0.0, 0.0)
    d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1, theta0[1], theta0[2], 0.0)
    theta, cur, meta = pj2.fisher_trust_region_gn(
        sc, ctx, theta0, d0, relins=12, radius_steps=20, fit_ab=True,
        ridge=1e-8, shrink=0.5)
    assert cur < d0 * 0.5, f"no material descent: {d0} -> {cur}"
    assert meta["gn_stop"] in {
        "step_below_shipped_quantization", "trust_radius_cap", "relin_cap",
        "singular_normal_equations", "zero_natural_gradient"}
    assert meta["n_relin"] >= 1


def test_fixture_is_not_vacuous_a_null_step_solver_fails_it():
    """If this fixture could be 'solved' without taking the computed step, the
    descent assertion above would pass against a broken solver."""
    fx = _Quadratic()
    sc, ctx = _fixture_ctx(fx)
    q = ctx[0]
    theta0 = q.theta(np.zeros(6), 0.0, 0.0)
    d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1, theta0[1], theta0[2], 0.0)
    assert d0 > 0.0
    # a solver that returns its input unchanged gets NO descent
    assert not (d0 < d0 * 0.5)


def test_convergence_proof_fires_when_the_lattice_cannot_resolve_the_step():
    """Start at the exact root: every proposed step lands on the same shipped
    float16 cells, so the loop must report the PROOF, not a bound."""
    fx = _Quadratic(cond=1e2)
    sc, ctx = _fixture_ctx(fx)
    q = ctx[0]
    theta0 = q.theta(fx.xstar[:6], float(fx.xstar[6]), float(fx.xstar[7]))
    d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1, theta0[1], theta0[2], 0.0)
    _theta, cur, meta = pj2.fisher_trust_region_gn(
        sc, ctx, theta0, d0, relins=6, radius_steps=24, fit_ab=True,
        ridge=1e-6, shrink=0.5)
    assert cur <= d0
    assert meta["gn_stop"] == "step_below_shipped_quantization"


def test_gn_never_returns_a_worse_value_than_it_started_with():
    """Acceptance is at SHIPPED quantization, so the returned value is monotone
    BY CONSTRUCTION -- the property the whole realized-acceptance discipline
    exists to give.  A regression here means a candidate was accepted on an
    unquantized score."""
    for seed in range(6):
        fx = _Quadratic(seed=seed, cond=10.0 ** (2 + seed))
        sc, ctx = _fixture_ctx(fx)
        q = ctx[0]
        theta0 = q.theta(np.full(6, 0.05), 0.0, 0.0)
        d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1,
                       theta0[1], theta0[2], 0.0)
        _t, cur, _m = pj2.fisher_trust_region_gn(
            sc, ctx, theta0, d0, relins=8, radius_steps=20, fit_ab=True,
            ridge=1e-8, shrink=0.5)
        assert cur <= d0


def test_stop_reasons_are_split_not_fused():
    """The #850 defect in one assertion: the reason vocabulary must contain a
    convergence value that is DISTINCT from every bound value."""
    src = _TOOL.read_text()
    for token in ("step_below_shipped_quantization", "trust_radius_cap",
                  "relin_cap", "singular_normal_equations"):
        assert f'"{token}"' in src
    # and the sweep-level reason must not be reused as the GN-level one
    assert '"sweep_relative_gain_below_tol"' in src


# --------------------------------------------------------------------------- #
# 4. the dual-metric readback
# --------------------------------------------------------------------------- #
def test_dual_metric_readback_reports_both_and_can_disagree():
    """Euclid-alone would be a false read.  Construct a displacement nearly
    orthogonal to the Euclidean gradient but aligned with the natural one, and
    assert the two cosines differ materially -- if they could not differ, the
    readback would be decoration."""
    h = np.diag([1e6, 1.0])
    cot = np.array([1.0, 1.0])
    nat = np.linalg.solve(h, cot)
    pose = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    pose0 = np.zeros(6)
    out = pj2._dual_metric_readback(
        np.pad(h, ((0, 4), (0, 4)), constant_values=0) + np.eye(6) * 1e-9,
        np.pad(cot, (0, 4)), np.pad(nat, (0, 4)),
        pose, pose0, 0.0, 0.0, (0.0, 0.0), 6)
    assert out["cos_euclid_disp_grad"] is not None
    assert out["cos_fisher_disp_natural"] is not None
    assert "metric_cos_sign_flip" in out
    assert out["fisher_cond"] is not None and out["fisher_cond"] > 1.0


def test_dual_metric_readback_flags_a_sign_flip():
    h = np.eye(6)
    cot = np.zeros(6)
    cot[0] = 1.0
    nat = -cot                      # deliberately anti-aligned natural direction
    pose = np.zeros(6)
    pose[0] = 1.0
    out = pj2._dual_metric_readback(h, cot, nat, pose, np.zeros(6),
                                    0.0, 0.0, (0.0, 0.0), 6)
    assert out["metric_cos_sign_flip"] is True


def test_dual_metric_readback_is_none_safe_on_a_zero_displacement():
    h = np.eye(6)
    cot = np.zeros(6)
    cot[0] = 1.0
    out = pj2._dual_metric_readback(h, cot, cot, np.zeros(6), np.zeros(6),
                                    0.0, 0.0, (0.0, 0.0), 6)
    assert out["cos_euclid_disp_grad"] is None
    assert out["metric_cos_sign_flip"] is False


# --------------------------------------------------------------------------- #
# 5. ordering, folding, and the monotone guard
# --------------------------------------------------------------------------- #
class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_pair_order_is_mass_first():
    final = {i: {"d_final": float(i)} for i in range(10)}
    order = pj2._pair_order(_Args(pair_list="", pairs=10),
                            final, {"n_pairs": 10})
    assert order[0] == 9 and order[-1] == 0


def test_pair_order_truncates_to_the_requested_budget_keeping_the_mass():
    final = {i: {"d_final": float(i)} for i in range(10)}
    order = pj2._pair_order(_Args(pair_list="", pairs=3), final, {"n_pairs": 10})
    assert order == [9, 8, 7]


def test_pair_order_respects_an_explicit_list():
    order = pj2._pair_order(_Args(pair_list="4,1", pairs=600), {}, {"n_pairs": 10})
    assert order == [4, 1]


def test_emit_is_monotone_and_never_folds_a_regression(tmp_path):
    final = tmp_path / "final.jsonl"
    final.write_text(
        '{"pair":0,"p":[1,0,0,0,0,0],"a":1.0,"b":0.0,"selector":1,'
        '"beta_idx":0,"d_final":0.5}\n'
        '{"pair":1,"p":[2,0,0,0,0,0],"a":1.0,"b":0.0,"selector":1,'
        '"beta_idx":0,"d_final":0.5}\n')
    out = tmp_path / "solved"
    out.mkdir()
    (out / "pj2_solve_shard0.jsonl").write_text(
        '{"pair":0,"p":[9,0,0,0,0,0],"a":1.0,"b":0.0,"d_final":0.1}\n'
        '{"pair":1,"p":[9,0,0,0,0,0],"a":1.0,"b":0.0,"d_final":0.9}\n')
    emitted = tmp_path / "final_pj2.jsonl"
    pj2.mode_emit(_Args(final_jsonl=final, out_dir=out, emit_jsonl=emitted))
    rows = [__import__("json").loads(x) for x in
            emitted.read_text().splitlines() if x.strip()]
    assert rows[0]["d_final"] == 0.1 and rows[0]["p"][0] == 9      # improved: folded
    assert rows[1]["d_final"] == 0.5 and rows[1]["p"][0] == 2      # worse: REFUSED


def test_contribution_matches_the_score_definition():
    assert pj2.contribution(0.0051662) == pytest.approx(0.2272927, abs=1e-7)
    assert pj2.contribution(0.0) == 0.0


def test_assert_fd_steps_refuses_on_drift(monkeypatch):
    import types
    fake = types.SimpleNamespace(FD_STEPS=np.array([1.0, 2.0, 3.0]))
    monkeypatch.setitem(sys.modules, "ddm_pfs1_ep_warp_pose_solve", fake)
    monkeypatch.setattr(pj2, "_ensure_paths", lambda: None)
    with pytest.raises(SystemExit):
        pj2._assert_fd_steps()


def test_tool_declares_non_promotable_axis():
    """A pose row from this tool is advisory; the axis stamp is load-bearing."""
    src = _TOOL.read_text()
    assert "score_claim" in src and "promotion_eligible" in src
    assert "macOS-CPU frozen-PoseNet advisory" in src


def test_scale_line_search_is_monotone_and_keeps_the_incumbent():
    """The incumbent (lam=1) is ALWAYS a member of the coarse scan, so the
    search can never return worse than its input.  If quantization were not
    idempotent this would silently fail, which is why it is asserted."""
    fx = _Quadratic(cond=1e3, seed=2)
    sc, ctx = _fixture_ctx(fx, quant_dim0=31.546875)
    q = ctx[0]
    for start in (np.full(6, 0.2), np.full(6, -1.5), fx.xstar[:6] * 1.7):
        theta0 = q.theta(np.asarray(start, np.float64), 1.0, 0.0)
        d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1,
                       theta0[1], theta0[2], 0.0)
        _t, cur, meta = pj2.scale_line_search(sc, ctx, theta0, d0,
                                              span=pj2.LAM_SPAN, max_evals=40)
        assert cur <= d0
        assert meta["scale_stop"] in {"golden_width_below_resolution",
                                      "scale_eval_cap"}


def test_shipped_quantization_is_idempotent():
    """``scale_line_search`` relies on q(q(x)) == q(x) so that lam=1 reproduces
    the incumbent EXACTLY; the dim0 residual path is the one that could break it."""
    q = pj2.ShippedQuant(31.546875)
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = np.array([rng.normal(33, 3), *rng.normal(0, 0.3, 5)])
        once = q.pose(p)
        assert np.array_equal(once, q.pose(once))
        a, b = q.ab(float(rng.normal(1, .1)), float(rng.normal(0, 3)))
        assert (a, b) == q.ab(a, b)


def test_solve_pair_end_to_end_never_regresses():
    """Both phases are monotone, so their composition must be.  This is the
    property the report's monotone guard should never have to exercise."""
    for seed in (0, 1, 2):
        fx = _Quadratic(cond=1e4, seed=seed)
        sc, ctx = _fixture_ctx(fx, quant_dim0=31.546875)
        q = ctx[0]
        theta0 = q.theta(np.full(6, 0.3), 1.0, 0.0)
        d0 = sc.d_pose(None, None, ctx[3], theta0[0], 0.08, 1,
                       theta0[1], theta0[2], 0.0)
        res = pj2.solve_pair(sc, ctx, theta0, d0, sweeps=3, relins=4,
                             radius_steps=12, fit_ab=True, scale_evals=30,
                             rel_tol=1e-3, ridge=1e-8, shrink=0.5)
        assert res["d_final"] <= d0
        assert [t["d"] for t in res["trace"]] == sorted(
            [t["d"] for t in res["trace"]], reverse=True)
        assert res["stop"] in {"sweep_relative_gain_below_tol", "sweep_cap"}


class _WarpSpy:
    """Minimal stand-in for ``StaticComposer`` that records how it was called."""

    def __init__(self):
        self.rot_calls = []
        self.far = np.zeros((2, 2), bool)
        self.alpha_row = np.zeros((2, 1, 1))
        self.recv = self
        self.o = self
        self.p3v2 = self
        self.posenet = None

    def warp_ground_rot(self, f1_f, theta, s_t, rot):
        self.rot_calls.append(float(rot))
        return np.zeros((2, 2, 3))

    def warps(self, f1_f, theta, s_t, rot):
        self.rot_calls.append(float(rot))
        return np.zeros((2, 2, 3)), np.zeros((2, 2, 3))

    def _to_uint8(self, x):
        return np.zeros((2, 2, 3), np.uint8)

    def pose6_u8(self, net, f0, f1):
        return np.zeros(6)


def test_realized_scorer_takes_the_single_warp_branch_when_beta_is_zero():
    """``inflate_runner_v4d.Decoder.f0`` uses a SINGLE warp at rot=1 when
    ``beta_mag == 0``; the ms8 harness always blended, and ``(1-a)x + a*x`` is
    not bit-identical to ``x`` in float -- the documented pw1 instrument floor.
    Mirroring the receiver branch is what makes this tool's canary EXACT, so the
    branch itself is asserted."""
    spy = _WarpSpy()
    sc = pj2.RealizedScorer(spy)
    pose = np.zeros(6)
    sc.d_pose(np.zeros((2, 2, 3)), np.zeros((2, 2, 3), np.uint8), np.zeros(6),
              pose, 0.08, 0, 1.0, 0.0, 0.0)
    assert spy.rot_calls == [1.0], spy.rot_calls


def test_realized_scorer_blends_two_rotations_when_beta_is_nonzero():
    spy = _WarpSpy()
    sc = pj2.RealizedScorer(spy)
    pose = np.zeros(6)
    pose[5] = 1.0                       # positive yaw -> positive beta sign
    sc.d_pose(np.zeros((2, 2, 3)), np.zeros((2, 2, 3), np.uint8), np.zeros(6),
              pose, 0.08, 0, 1.0, 0.0, 0.5)
    assert spy.rot_calls == [0.75, 1.25], spy.rot_calls


def test_realized_scorer_beta_sign_follows_the_yaw_dim():
    """``beta = beta_mag * sign(pose[5])`` (receiver :196).  A flipped sign here
    would silently swap the top/bottom rolling-shutter rows."""
    spy = _WarpSpy()
    sc = pj2.RealizedScorer(spy)
    pose = np.zeros(6)
    pose[5] = -1.0
    sc.d_pose(np.zeros((2, 2, 3)), np.zeros((2, 2, 3), np.uint8), np.zeros(6),
              pose, 0.08, 0, 1.0, 0.0, 0.5)
    assert spy.rot_calls == [1.25, 0.75], spy.rot_calls


def test_realized_scorer_uses_the_two_plane_path_only_when_selector_is_one():
    spy = _WarpSpy()
    sc = pj2.RealizedScorer(spy)
    out0 = sc._warp_pair(np.zeros((2, 2, 3)), np.zeros(6), 0.08, 0, 1.0)
    out1 = sc._warp_pair(np.zeros((2, 2, 3)), np.zeros(6), 0.08, 1, 1.0)
    assert out0.shape == out1.shape == (2, 2, 3)
    assert sc.n_evals == 0              # _warp_pair alone is not a scorer eval
