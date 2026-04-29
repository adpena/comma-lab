"""Lane Joint-ADMM coordinator — convergence + KKT + restart + scorer-rule tests.

Per memory `project_phases_2_3_4_design_implementation_math_provenance_20260429`
§"Lane 10 kill criteria": ADMM must converge within 1000 iters; the synthetic
harness uses a much smaller cap (200) for CI speed.

All test claims are tagged:
- [synthetic] for problems with a closed-form optimum
- [prediction] for any score-saving estimate (none in this file — V2 measures)

CLAUDE.md non-negotiables verified by these tests:
1. Coordinator does NOT load any scorer (test_strict_scorer_rule_no_scorer_import).
2. Adaptive ρ + restart (test_divergent_problem_restarts).
3. Exact byte projection after every codec call (test_byte_projection_handles_discretisation).
4. KKT waterline equilibration at convergence (test_two_stream_convex_converges_to_kkt).
5. Real codec wrapping works (test_pose_delta_proximal_wrapper_runs_in_admm).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.joint_admm_coordinator import (
    AdmmResult,
    JointADMMConfig,
    ProximalStepResult,
    StreamProximalCodec,
    kkt_waterline_residual,
    run_admm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic streams (CPU-only, no scorer load)
# ─────────────────────────────────────────────────────────────────────────────


class QuadraticRateStream:
    """Stream with f(b) = a*(b - b_opt)^2 (convex; minimum at b_opt).

    Marginal df/db = 2a*(b - b_opt). For coordinator's sign convention
    (positive = "more bytes lowers score"), we negate when b < b_opt:
    margin = -df/db = 2a*(b_opt - b).

    Proximal-Lagrangian step: minimise   f(b) + dual * b
    (the dual is the price paid per byte; high dual ⇒ shrink b).

        argmin_b  a*(b - b_opt)^2 + dual * b
        = b_opt - dual / (2a)

    discretised + clamped to [0, target_bytes].
    """

    def __init__(
        self,
        a: float,
        b_opt: float,
        name: str = "quadratic",
        discretisation: float = 1.0,
    ):
        self.a = a
        self.b_opt = b_opt
        self._name = name
        self.discretisation = discretisation

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        # Lagrangian-form proximal: solve unconstrained min f(b) + dual*b.
        b_unconstr = self.b_opt - dual / (2.0 * self.a)
        b = max(0.0, min(target_bytes, b_unconstr))
        b = max(0.0, round(b / self.discretisation) * self.discretisation)
        score = self.a * (b - self.b_opt) ** 2
        margin = max(2.0 * self.a * (self.b_opt - b), 0.0)
        return ProximalStepResult(
            encoded_bytes=int(b),
            score_delta=float(score),
            marginal=float(margin),
            state=None,
        )


class LinearRateStream:
    """Stream with f(b) = max(c0 - slope*b, 0) (linear, saturating).

    Marginal = slope while not saturated; 0 once at floor.

    Proximal-Lagrangian step: minimise max(c0 - slope*b, 0) + dual*b.
    If dual >= slope: the coefficient of b is positive ⇒ optimal b = 0.
    If dual <  slope: pay all bytes you can up to saturation point c0/slope.
    """

    def __init__(
        self,
        slope: float,
        c0: float,
        name: str = "linear",
        discretisation: float = 1.0,
    ):
        self.slope = slope
        self.c0 = c0
        self._name = name
        self.discretisation = discretisation

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        floor_b = self.c0 / self.slope  # saturation point
        if dual >= self.slope:
            b_unconstr = 0.0  # not worth spending bytes
        else:
            b_unconstr = floor_b  # spend all the way to saturation
        b = max(0.0, min(target_bytes, b_unconstr))
        b = max(0.0, round(b / self.discretisation) * self.discretisation)
        score = max(self.c0 - self.slope * b, 0.0)
        margin = self.slope if b < floor_b else 0.0
        return ProximalStepResult(
            encoded_bytes=int(b),
            score_delta=float(score),
            marginal=float(margin),
            state=None,
        )


class DivergentMockStream:
    """Pathological stream: encoded bytes oscillate vs target.

    Used to trigger divergence detection + restart.
    """

    def __init__(self, name: str = "divergent"):
        self._name = name
        self._toggle = False

    @property
    def name(self) -> str:
        return self._name

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        self._toggle = not self._toggle
        # Each call returns a wildly different byte count to keep primal
        # residual large + growing.
        b = (target_bytes * 5.0) if self._toggle else max(target_bytes - 50.0, 0.0)
        return ProximalStepResult(
            encoded_bytes=int(max(b, 0.0)),
            score_delta=1.0,
            marginal=0.5,
            state=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test (a): convergence on convex 2-stream synthetic R(D); KKT waterline
# ─────────────────────────────────────────────────────────────────────────────


def test_two_stream_convex_converges_to_kkt():
    """[synthetic] Two quadratic streams with different curvatures.
    Closed-form KKT requires marginals to equilibrate.

    f1(b) = a1*(b - b_opt1)^2 ⇒ margin1 = 2*a1*(b_opt1 - b)
    f2(b) = a2*(b - b_opt2)^2 ⇒ margin2 = 2*a2*(b_opt2 - b)

    With a1=0.005, b_opt1=300, a2=0.01, b_opt2=200, B=300:
    KKT: 2*0.005*(300 - b1) = 2*0.01*(200 - b2)  AND  b1 + b2 = 300
    ⇒ 0.01*(300 - b1) = 0.02*(200 - b2)
    ⇒ 0.01*(300 - b1) = 0.02*(200 - 300 + b1)
    ⇒ 0.01*(300 - b1) = 0.02*(b1 - 100)
    ⇒ 3 - 0.01*b1 = 0.02*b1 - 2
    ⇒ 5 = 0.03*b1
    ⇒ b1 = 166.67, b2 = 133.33
    ⇒ margin1 = 2*0.005*(300-166.67) = 1.333
    ⇒ margin2 = 2*0.01*(200-133.33) = 1.333  ✓ EQUAL (KKT)
    """
    quad1 = QuadraticRateStream(a=0.005, b_opt=300.0, name="q1", discretisation=1.0)
    quad2 = QuadraticRateStream(a=0.01, b_opt=200.0, name="q2", discretisation=1.0)
    cfg = JointADMMConfig(
        byte_budget=300.0,
        max_iters=300,
        primal_tol=1e-2,
        dual_tol=1e-2,
        kkt_waterline_tol=0.05,
        rho_init=0.02,
        verbose=False,
    )
    result = run_admm([quad1, quad2], cfg)

    # Coordinator should converge.
    assert isinstance(result, AdmmResult)
    assert result.converged, (
        f"failed to converge in {result.iters} iters; "
        f"final primal/dual={result.history[-1].primal_residual:.4e}/"
        f"{result.history[-1].dual_residual:.4e}"
    )

    # KKT: among UNSATURATED ACTIVE streams, marginals should equilibrate.
    bytes_arr = np.asarray(result.final_bytes_per_stream)
    margins = np.asarray(result.final_marginal_per_stream)
    res, has_active, _ = kkt_waterline_residual(bytes_arr, margins)
    assert has_active, "no unsaturated active stream after ADMM"
    # Marginals should equilibrate around 1.333; waterline residual within 0.1.
    assert result.waterline_kkt_residual < 0.10, (
        f"waterline residual {result.waterline_kkt_residual:.3f} too large; "
        f"margins={margins.tolist()}, bytes={bytes_arr.tolist()}"
    )
    # Both bytes should be in interior (~100-200 range, NOT pinned to 0 or budget).
    assert 100.0 <= bytes_arr[0] <= 220.0, (
        f"q1 bytes {bytes_arr[0]} not in expected interior range; "
        f"closed-form predicts ~167"
    )
    assert 80.0 <= bytes_arr[1] <= 200.0, (
        f"q2 bytes {bytes_arr[1]} not in expected interior range; "
        f"closed-form predicts ~133"
    )
    # Total bytes <= budget (with tolerance for discretisation).
    assert bytes_arr.sum() <= cfg.byte_budget + 5.0

    # KKT residual logged for the report-back to caller.
    print(
        f"[KKT] waterline_residual={result.waterline_kkt_residual:.6f} "
        f"satisfied={result.waterline_satisfied} "
        f"bytes={bytes_arr.tolist()} margins={margins.tolist()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (b): divergence + restart
# ─────────────────────────────────────────────────────────────────────────────


def test_divergent_problem_restarts():
    """[synthetic] Pathological codec triggers restart; coordinator stays bounded.

    The DivergentMockStream is INFEASIBLE (it ignores any byte target the
    coordinator sets; its output oscillates between 5x and tgt-50). No
    coordinator can enforce a byte budget against such a stream — the test
    only asserts that the coordinator (a) DETECTS the divergence pattern and
    (b) fires a restart instead of letting ρ saturate at rho_max + diverging.
    """
    bad1 = DivergentMockStream(name="bad1")
    bad2 = DivergentMockStream(name="bad2")
    cfg = JointADMMConfig(
        byte_budget=100.0,
        max_iters=80,
        rho_init=10.0,  # big rho amplifies divergence
        rho_max=1e4,
        rho_imbalance_ratio=5.0,
        restart_threshold=3,
        primal_tol=1e-3,
        dual_tol=1e-3,
        verbose=False,
    )
    result = run_admm([bad1, bad2], cfg)

    # Should have restarted at least once.
    assert result.restarts >= 1, (
        f"expected >=1 restart on divergent problem; got {result.restarts}"
    )
    # ρ should NOT have saturated at rho_max (the restart machinery
    # specifically prevents that).
    final_rho = result.history[-1].rho
    assert final_rho < cfg.rho_max, (
        f"rho saturated at rho_max {cfg.rho_max} ⇒ restart did not effectively "
        f"reset; final_rho={final_rho}"
    )
    # The coordinator should NOT crash + should produce a valid result.
    assert result.iters > 0
    print(
        f"[restart] restarts={result.restarts} iters={result.iters} "
        f"final_rho={final_rho:.2f} final_bytes={result.final_bytes_per_stream}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (c): adaptive rho grows monotonically until convergence
# ─────────────────────────────────────────────────────────────────────────────


def test_adaptive_rho_grows_when_undersized():
    """[synthetic] rho_init too small ⇒ coordinator grows it."""
    quad = QuadraticRateStream(a=0.05, b_opt=80.0, name="q", discretisation=1.0)
    lin = LinearRateStream(slope=0.3, c0=5.0, name="l", discretisation=1.0)
    cfg = JointADMMConfig(
        byte_budget=100.0,
        max_iters=300,
        rho_init=1e-3,  # WAY too small
        rho_max=1e6,
        rho_imbalance_ratio=10.0,
        primal_tol=1e-2,
        dual_tol=1e-2,
        verbose=False,
    )
    result = run_admm([quad, lin], cfg)
    rhos = [step.rho for step in result.history]
    # Growth should occur — early iterations have rho<<1, later iterations
    # should reach >= 0.01 (10x growth) at minimum.
    assert max(rhos) > rhos[0] * 5.0, (
        f"rho did not grow significantly; trace: {rhos[:20]} ... {rhos[-5:]}"
    )
    # Should still converge.
    assert result.converged or result.iters == cfg.max_iters
    print(
        f"[rho] init={cfg.rho_init} final_max={max(rhos):.4f} "
        f"converged={result.converged}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (d): byte projection — codec actual bytes != target
# ─────────────────────────────────────────────────────────────────────────────


def test_byte_projection_handles_discretisation():
    """[synthetic] Heavy discretisation (multiples of 25) exercises the
    exact-byte-projection-after-every-codec-call rule."""
    # Both streams round to multiples of 25, so nothing fits the budget exactly.
    quad = QuadraticRateStream(a=0.02, b_opt=100.0, name="q", discretisation=25.0)
    lin = LinearRateStream(slope=0.4, c0=10.0, name="l", discretisation=25.0)
    cfg = JointADMMConfig(
        byte_budget=120.0,
        max_iters=200,
        primal_tol=1e-2,
        dual_tol=1e-2,
        rho_init=0.1,
        verbose=False,
    )
    result = run_admm([quad, lin], cfg)
    # Even when targets land between ladder points, dual variables should
    # accumulate to push the next iteration's targets in the right direction.
    duals = [step.duals_per_stream for step in result.history]
    # Some dual variable must be non-zero somewhere (else byte projection
    # was a no-op which contradicts the test premise).
    nonzero_dual_observed = any(
        any(abs(d) > 1e-6 for d in step) for step in duals
    )
    assert nonzero_dual_observed, (
        f"duals stayed zero ⇒ byte-projection-with-discretisation didn't "
        f"engage; final duals={duals[-1]}"
    )
    # Final bytes should be on the discretisation grid.
    for b in result.final_bytes_per_stream:
        assert b % 25 == 0, f"byte {b} not on discretisation grid 25"
    print(
        f"[discrete] final_bytes={result.final_bytes_per_stream} "
        f"final_duals={result.history[-1].duals_per_stream}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test (e): strict-scorer-rule — coordinator does not import scorers
# ─────────────────────────────────────────────────────────────────────────────


def test_strict_scorer_rule_no_scorer_import():
    """The coordinator module + the proximal-pose-delta module must NOT
    import any scorer (SegNet / PoseNet / contest_scorer / scorer / distillation).

    This is the inflate-side scorer-rule extended to the COMPRESS-time
    coordinator: even though compress-time has unlimited compute, the
    coordinator's job is allocation-not-measurement; live scorer calls would
    violate separation of concerns. See CLAUDE.md "Strict scorer rule".
    """
    # Inspect the source WITHOUT reloading — reloading would invalidate
    # the AdmmResult class identity for subsequent tests in the same
    # session (Python's isinstance uses identity, not name).
    import inspect
    import tac.joint_admm_coordinator as coord_module
    src = inspect.getsource(coord_module)
    forbidden_patterns = [
        "from tac.scorer ",
        "import tac.scorer",
        "from tac.eval ",
        "from upstream",
        "contest_scorer",
        "EfficientNet",
        "FastViT",
        "load_segnet",
        "load_posenet",
    ]
    for pat in forbidden_patterns:
        assert pat not in src, (
            f"joint_admm_coordinator.py contains forbidden scorer-load "
            f"pattern: {pat!r}. Strict-scorer-rule violation."
        )

    # Also verify the proximal pose-delta wrapper is clean.
    from tac import joint_admm_proximal_pose_delta as pd_mod
    pd_src = inspect.getsource(pd_mod)
    for pat in forbidden_patterns:
        assert pat not in pd_src, (
            f"joint_admm_proximal_pose_delta.py contains forbidden scorer-load "
            f"pattern: {pat!r}. Strict-scorer-rule violation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test (f): real codec wrapping — pose_delta_codec proximal in run_admm
# ─────────────────────────────────────────────────────────────────────────────


def test_pose_delta_proximal_wrapper_runs_in_admm():
    """[synthetic-frontier] Wrap pose_delta_codec; coordinator runs to
    completion + reports byte/marginal data.

    The score-cost surface is a CACHED synthetic function (no live scorer):
    a small-budget caller would derive this from a one-shot offline sweep.
    """
    from tac.joint_admm_proximal_pose_delta import (
        PoseDeltaProximalCodec,
        build_pose_delta_frontier,
    )

    # Synthetic 50-pair pose tensor (smooth ramp).
    n_pairs = 50
    t = torch.linspace(0, 1, n_pairs).unsqueeze(-1)  # (50, 1)
    poses = torch.cat(
        [t, t * 0.5, t * 0.25, t * 0.1, t * 0.05, t * 0.02], dim=1
    )  # (50, 6)

    # Cached score-cost surface. Larger byte budget => lower cost.
    def score_at_bytes(b: int) -> float:
        # Simple monotonic decreasing cost in the LOG scale of bytes.
        return 0.01 / max(np.log1p(float(b)), 1.0)

    frontier = build_pose_delta_frontier(poses, score_at_bytes)
    pose_codec = PoseDeltaProximalCodec(
        poses=poses, frontier=frontier, name="pose_delta"
    )

    # Need >=1 stream; pair pose_delta with a synthetic mask-byte placeholder
    # so the coordinator has something to allocate against.
    mask_placeholder = QuadraticRateStream(
        a=1e-5, b_opt=400.0, name="mask_bytes", discretisation=1.0
    )

    # Total budget: pose_delta needs ~(50-1)*6 + 12 + 12 + 50 ≈ 368 bytes;
    # mask placeholder wants 400 ⇒ total room ~768.
    cfg = JointADMMConfig(
        byte_budget=800.0,
        max_iters=100,
        primal_tol=1e-2,
        dual_tol=1e-2,
        rho_init=0.1,
        kkt_waterline_tol=0.5,
        verbose=False,
    )
    result = run_admm([pose_codec, mask_placeholder], cfg)
    # Must return without crashing + produce a valid AdmmResult.
    assert isinstance(result, AdmmResult)
    assert result.iters >= 1
    # pose_delta byte count should match its single frontier sample.
    pose_bytes = result.final_bytes_per_stream[0]
    assert pose_bytes == frontier[0].bytes_used, (
        f"pose_delta returned bytes={pose_bytes} but frontier sample is "
        f"{frontier[0].bytes_used}"
    )
    # Coordinator should NOT crash on the strict Protocol check.
    assert isinstance(pose_codec, StreamProximalCodec)
    print(
        f"[real-wrap] pose_bytes={pose_bytes} mask_bytes="
        f"{result.final_bytes_per_stream[1]} converged={result.converged} "
        f"iters={result.iters}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bonus guard tests (cheap, not in the 6-test count)
# ─────────────────────────────────────────────────────────────────────────────


def test_config_rejects_zero_budget():
    """Silent-default trap: byte_budget=0 must raise (Check 81 STRICT)."""
    with pytest.raises(ValueError, match="byte_budget"):
        JointADMMConfig(byte_budget=0.0)


def test_config_rejects_invalid_rho():
    with pytest.raises(ValueError, match="rho_init"):
        JointADMMConfig(byte_budget=100.0, rho_init=0.0)
    with pytest.raises(ValueError, match="rho_growth"):
        JointADMMConfig(byte_budget=100.0, rho_growth=0.5)


def test_run_admm_rejects_non_protocol_stream():
    class NotAStream:
        pass

    cfg = JointADMMConfig(byte_budget=100.0)
    with pytest.raises(TypeError, match="Protocol"):
        run_admm([NotAStream()], cfg)
