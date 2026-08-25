#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pj2 -- the s_t codebook and the pose translation are ONE degenerate scale.

WHY THIS EXISTS
---------------
``ddm_ms8`` bought ``dS -0.0491770`` by re-fitting the 11-entry ``s_t`` scalar
codebook and paying ``+51 B`` for the widened index stream, and it left three
things owed (its own s11): the arms held the pose FIXED (one coordinate-descent
step), the 21-point evaluation support is "a floor, not a ceiling", and a joint
``(pose, s_t)`` re-solve "should win more".

The shipped ground homography is (``pfs1_warp_receiver.pose_to_homography``)::

    t = s_t * [p[2], p[1], p[0]]
    R = expmap(rot * [p[3], p[4], p[5]])
    H = K (R - t n^T / h) K^-1

so the translation enters ONLY through the PRODUCT ``s_t * p[0:3]``, and the
far-plane homography is evaluated at ``s_t = 0`` and does not see it at all.
``s_t`` and the pose translation triple are therefore an EXACT scale-degenerate
pair: ``H(p, s) == H(lam*p[0:3], s/lam)`` for every ``lam > 0``.  MEASURED here
at ``--mode degen``: max relative homography difference ``7.4e-16`` over 200
random draws, i.e. float64 roundoff.

Two consequences, both measured by this tool rather than argued:

1. The ms8 move is a REPARAMETERIZATION.  The same effective scale is reachable
   through the pose column, which the archive ALREADY ships -- so the widened
   ``st_coded`` stream is not the only way to buy it.
2. The pose column is stored as float16 (dim0 as an f16 residual from
   ``manifest["pose_dim0_offset"]``), whose relative resolution is ~5e-4 --
   two orders finer than any 16-entry codebook.  The "inter-column" optima ms8
   could not express are reachable here.

WHAT THIS TOOL DOES
-------------------
``--mode degen``   algebraic + realized positive control of the degeneracy.
``--mode solve``   per-pair coordinate descent on the REALIZED objective:
                   a bracketed scale line-search on ``lam`` alternating with a
                   damped Gauss-Newton over ``(p[0:6], a, b)``, every candidate
                   scored AT THE SHIPPED QUANTIZATION, with a real convergence
                   test (``step_below_shipped_quantization``) and a
                   ``stop_reason`` census -- the ``#850`` defect, cured rather
                   than described.  Resumable, sharded, wall-capped.
``--mode emit``    fold solved rows into a builder-ready final JSONL.
``--mode report``  aggregate n600 census + composed-S prediction.

CONVERGENCE, NOT A CAP
----------------------
``ddm_os1`` measured the live pose GN ``ALL_STOPPED_ON_A_BOUND 600/600, 100%
mass`` -- its ``cur < 1e-6`` criterion fired 0/600 and was FUSED with the bound
so no census could tell them apart.  Here the exits are SPLIT and every one is
recorded.  The criterion is not a tolerance constant: a step that moves no
shipped parameter off its float16 cell CANNOT change the shipped ``d_pose``
however long the ladder continues, so ``step_below_shipped_quantization`` is a
proof of local optimality ON THE LATTICE THAT SHIPS.  The bounds remain, but
they are now distinguishable from convergence and are reported per pair.

AXIS
----
``[macOS-CPU frozen-PoseNet advisory]``, ``score_claim=false``,
``promotion_eligible=false``.  No training, no paid dispatch, no pointer
mutation, no exact gate fired.  Composed S is a PREDICTION whose fidelity
anchors on this vehicle are the QA78 v4d gate residual (1.8e-6) and the pw1
gate residual (2.5e-6).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tac.canonical_equations.ddm_fs1_coordinate_fit_staleness_20260802 import (
    FIT_CONTEXT_KEY,
    stamp_fit_context,
)

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pj2_pose_scale_joint_solve.v1"
N_PAIRS = 600
ARCHIVE_DENOM = 37_545_489.0

#: The live own-vehicle frontier archive (post-ms8) and its predecessor.
LIVE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/"
                    "v4d_composed_ms8_archive.zip")
BASE_LABEL = "celldrop50"

#: Forward-difference steps per pose dim, inherited from the live solver
#: (``ddm_pfs1_ep_warp_pose_solve.FD_STEPS``: scaled to the measured per-dim
#: spread of the carried pose).  Re-derived, not re-typed: asserted equal to the
#: live module's table in ``_assert_fd_steps``.
FD_STEPS = np.array([0.08, 0.004, 0.004, 0.0015, 0.0015, 0.004], np.float64)
FD_GAIN, FD_BIAS = 0.02, 2.0           # ddm_v4c_resolve rung-B photometric steps

#: Scale line-search: multiplicative bracket around lam=1.  The widest incumbent
#: cell ratio in the ms8 fitted table is 0.16/0.14 = 1.143, and ms8 measured
#: 72/600 pairs landing OUTSIDE the incumbent codewords with abs_s_move_max 0.1
#: (a factor of 2.25 on s), so the bracket must span well past one cell.
#: The vendored warp receiver the whole live chain imports by bare name
#: (``WarpPoseOracle.__init__`` inserts this same directory).
PFS1_SUBMISSION = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/submission")

#: Trust-region radius ladder.  MEASURED (see fisher_trust_region_gn): the
#: Fisher condition number on this chart is ~1e11, so the ladder must span many
#: orders; 0.25 per rejection reaches 4^-steps of the natural length.
TR_SHRINK, TR_GROW = 0.25, 4.0

LAM_SPAN = 4.0
LAM_MIN_REL = 1e-4                     # golden-section stop on log-lam width


def _ensure_paths() -> None:
    """Put every bare-name import the live chain uses on the path, once.

    ``inflate_runner_v4d`` imports ``ddm_tr1_runtime`` / ``ddm_r7_token_coder``
    (under ``src/tac/optimization``) and ``pfs1_warp_receiver`` (vendored beside
    the D1 submission) by BARE NAME, so a consumer outside ``experiments/`` must
    supply all three roots or it fails at import with a misleading message.
    """
    for p in (REPO / "experiments", REPO / "src", REPO / "upstream",
              REPO / "src" / "tac" / "optimization", PFS1_SUBMISSION):
        if not p.exists():
            raise SystemExit(f"required import root missing: {p}")
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _utc() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * float(d_pose_mean)))


def _assert_fd_steps() -> None:
    """Refuse if the live solver's FD table drifted from the vendored copy."""
    _ensure_paths()
    import ddm_pfs1_ep_warp_pose_solve as pfs1
    live = np.asarray(pfs1.FD_STEPS, np.float64)
    if live.shape != FD_STEPS.shape or not np.array_equal(live, FD_STEPS):
        raise SystemExit(
            f"live FD_STEPS {live.tolist()} != vendored {FD_STEPS.tolist()}; "
            "the Jacobian scaling this tool uses is no longer the live one")


# --------------------------------------------------------------------------- #
# the shipped quantization -- the single place that knows how a candidate
# becomes archive bytes.  Everything is scored THROUGH this.
# --------------------------------------------------------------------------- #
class ShippedQuant:
    """float16 pose storage with dim0 carried as a residual from a fixed offset.

    ``ddm_v4d_build_composed_archive`` stores ``pose[:,0] - dim0_offset`` and the
    receiver reconstructs ``dim0_offset + f16(residual)``
    (``inflate_runner_v4d.py:140-143``); dims 1..5 and ``(a, b)`` are plain f16.
    A candidate that does not survive this map is not a candidate.
    """

    def __init__(self, dim0_offset: float | None) -> None:
        self.dim0_offset = None if dim0_offset is None else float(dim0_offset)

    def pose(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, np.float64).astype(np.float16).astype(np.float64)
        if self.dim0_offset is not None:
            q = q.copy()
            res = np.float16(np.float64(p[0]) - self.dim0_offset)
            q[0] = self.dim0_offset + float(res)
        return q

    def ab(self, a: float, b: float) -> tuple[float, float]:
        return (float(np.float16(a)), float(np.float16(b)))

    def theta(self, p: np.ndarray, a: float, b: float):
        qa, qb = self.ab(a, b)
        return self.pose(p), qa, qb

    def same(self, t0, t1) -> bool:
        """True when two candidates land on IDENTICAL shipped bytes."""
        return (np.array_equal(t0[0], t1[0])
                and t0[1] == t1[1] and t0[2] == t1[2])


# --------------------------------------------------------------------------- #
# the realized objective -- mirrors inflate_runner_v4d.Decoder.f0 branch for
# branch (including the beta_mag == 0 single-warp path, which the ms8 harness
# did NOT take: its always-blend path computes (1-a)x + a*x, which is not
# bit-identical to x in float and is the documented pw1 instrument floor).
# --------------------------------------------------------------------------- #
class RealizedScorer:
    def __init__(self, comp) -> None:
        self.c = comp
        self.n_evals = 0

    def _warp_pair(self, f1_f, pose, s_t, sel, rot):
        if sel == 0:
            return self.c.warp_ground_rot(f1_f, pose, s_t, rot)
        wg, wf = self.c.warps(f1_f, pose, s_t, rot)
        return np.where(self.c.far[..., None], wf, wg)

    def d_pose(self, f1_f, f1_u8, tp, pose, s_t, sel, a, b, beta_mag) -> float:
        self.n_evals += 1
        if beta_mag != 0.0:
            beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)
            f0f = ((1.0 - self.c.alpha_row)
                   * self._warp_pair(f1_f, pose, s_t, sel, 1.0 - beta / 2.0)
                   + self.c.alpha_row
                   * self._warp_pair(f1_f, pose, s_t, sel, 1.0 + beta / 2.0))
        else:
            f0f = self._warp_pair(f1_f, pose, s_t, sel, 1.0)
        if a != 1.0 or b != 0.0:
            f0f = a * f0f + b
        p6 = self.c.o.p3v2.pose6_u8(self.c.o.posenet,
                                    self.c.recv._to_uint8(f0f), f1_u8)
        return float(np.mean((p6 - tp) ** 2))

    def pose6(self, f1_f, f1_u8, pose, s_t, sel, a, b, beta_mag) -> np.ndarray:
        self.n_evals += 1
        if beta_mag != 0.0:
            beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)
            f0f = ((1.0 - self.c.alpha_row)
                   * self._warp_pair(f1_f, pose, s_t, sel, 1.0 - beta / 2.0)
                   + self.c.alpha_row
                   * self._warp_pair(f1_f, pose, s_t, sel, 1.0 + beta / 2.0))
        else:
            f0f = self._warp_pair(f1_f, pose, s_t, sel, 1.0)
        if a != 1.0 or b != 0.0:
            f0f = a * f0f + b
        return self.c.o.p3v2.pose6_u8(self.c.o.posenet,
                                      self.c.recv._to_uint8(f0f), f1_u8)


# --------------------------------------------------------------------------- #
# the solve
# --------------------------------------------------------------------------- #
GOLDEN = 0.5 * (3.0 - 5.0 ** 0.5)      # 0.381966


def scale_line_search(sc: RealizedScorer, ctx, theta, cur, *,
                      span: float, max_evals: int) -> tuple[tuple, float, dict]:
    """Bracketed golden-section on log(lam), lam scaling ``pose[0:3]``.

    The objective along this ray IS the ms8 curve (exact degeneracy), so it is
    smooth and deeply U-shaped; golden section on the log is the right shape.
    Every candidate is quantized to shipped bytes BEFORE scoring, so the
    returned value is realizable, and the search stops when the bracket is
    narrower than the shipped quantization can resolve.
    """
    q, f1_f, f1_u8, tp, s_t, sel, g = ctx
    pose, a, b = theta
    n0 = sc.n_evals

    def at(lam: float):
        p = pose.copy()
        p[:3] = p[:3] * lam
        t = q.theta(p, a, b)
        return t, sc.d_pose(f1_f, f1_u8, tp, t[0], s_t, sel, t[1], t[2], g)

    lo, hi = -np.log(span), np.log(span)
    # coarse scan first: the curve has 4 orders of magnitude of dynamic range
    # (ms8 pair 44: 36.2 -> 0.449 -> 54.4), so a pure golden section from a wide
    # bracket can be captured by a spurious local shoulder.
    ncoarse = 9
    xs = np.linspace(lo, hi, ncoarse)
    vals, thetas = [], []
    for x in xs:
        t, v = at(float(np.exp(x)))
        vals.append(v)
        thetas.append(t)
    j = int(np.argmin(vals))
    best_t, best_v = thetas[j], float(vals[j])
    if best_v > cur:                    # incumbent (lam=1) is inside the scan
        best_t, best_v = theta, cur
    a_x = xs[max(j - 1, 0)]
    b_x = xs[min(j + 1, ncoarse - 1)]
    # golden section inside the bracketing triple
    c_x = a_x + GOLDEN * (b_x - a_x)
    d_x = b_x - GOLDEN * (b_x - a_x)
    tc, fc = at(float(np.exp(c_x)))
    td, fd = at(float(np.exp(d_x)))
    reason = "golden_width_below_resolution"
    while sc.n_evals - n0 < max_evals:
        if (b_x - a_x) < LAM_MIN_REL:
            break
        if fc < fd:
            b_x, d_x, fd, td = d_x, c_x, fc, tc
            c_x = a_x + GOLDEN * (b_x - a_x)
            tc, fc = at(float(np.exp(c_x)))
        else:
            a_x, c_x, fc, tc = c_x, d_x, fd, td
            d_x = b_x - GOLDEN * (b_x - a_x)
            td, fd = at(float(np.exp(d_x)))
    else:
        reason = "scale_eval_cap"
    for t, v in ((tc, fc), (td, fd)):
        if v < best_v:
            best_t, best_v = t, v
    return best_t, best_v, {"scale_stop": reason,
                            "scale_evals": sc.n_evals - n0}


def _spd_chart(H: np.ndarray, ridge: float) -> np.ndarray:
    """The gauge-fixed SPD chart the canonical helper REQUIRES the consumer to own.

    ``fisher_natural_cotangent_trust_region_step`` accepts "only an already
    gauge-fixed SPD chart ... implicit damping here would silently change the
    geometry".  With 6 PoseNet residuals and up to 8 parameters the raw
    Gauss-Newton pullback ``J^T J`` has rank <= 6 and is singular in the ambient
    chart, so the ridge is a CONSUMER obligation, declared here and recorded in
    the receipt rather than hidden inside the metric solve.
    """
    d = np.maximum(np.diag(H), 0.0)
    floor = max(float(d.max()), 1e-30) * 1e-12
    return 0.5 * (H + H.T) + ridge * np.diag(np.maximum(d, floor))


def fisher_trust_region_gn(sc: RealizedScorer, ctx, theta, cur, *,
                           relins: int, radius_steps: int, fit_ab: bool,
                           ridge: float, shrink: float = TR_SHRINK) -> tuple:
    """Fisher-natural trust-region Gauss-Newton over (pose[0:6], a, b).

    METRIC-FIRST.  ``J`` is the Jacobian of the frozen PoseNet's 6-vector output
    with respect to the shipped parameters, so ``H = J^T J`` IS the Gauss-Newton
    pullback of the PoseNet quadratic -- the scorer's own metric on this chart,
    not a Euclidean surrogate.  The step is taken by the canonical helper
    ``tac.information_geometry.fisher_natural_trust_region`` and projected into
    an ``H``-NORM ball (a Fisher ball), never a Euclidean one.

    ``ddm_os1`` #1: the live 6-param pose GN fuses ``cur < 1e-6`` with
    ``not accepted`` into one ``break``, so 600/600 solves report a bound and no
    census can separate them.  Here each exit is its own reason, and the
    convergence test is a PROOF on the shipped lattice, not a tolerance
    constant: a step whose quantized candidate lands on the SAME float16 cells
    cannot change the shipped ``d_pose`` however long the loop continues.
    """
    from tac.information_geometry.bregman_v9_surfaces import GeometryValidationError
    from tac.information_geometry.fisher_natural_trust_region import (
        fisher_natural_cotangent_trust_region_step,
    )

    q, f1_f, f1_u8, tp, s_t, sel, g = ctx
    pose, a, b = theta
    pose0 = pose.copy()
    ab0 = (a, b)
    dim = 8 if fit_ab else 6
    steps = np.concatenate([FD_STEPS, [FD_GAIN, FD_BIAS]])[:dim]
    cur6 = sc.pose6(f1_f, f1_u8, pose, s_t, sel, a, b, g)
    reason = "relin_cap"
    n_relin = 0
    delta = None
    readback: dict = {}
    for _ in range(relins):
        n_relin += 1
        J = np.zeros((6, dim), np.float64)
        for k in range(dim):
            p2, a2, b2 = pose.copy(), a, b
            if k < 6:
                p2[k] += steps[k]
            elif k == 6:
                a2 = a + steps[k]
            else:
                b2 = b + steps[k]
            t2 = q.theta(p2, a2, b2)
            J[:, k] = (sc.pose6(f1_f, f1_u8, t2[0], s_t, sel, t2[1], t2[2], g)
                       - cur6) / steps[k]
        r = cur6 - tp
        gn = J.T @ J
        cot = -(J.T @ r)                       # descent cotangent (helper owns no sign)
        accepted = False
        all_below_quant = True
        # The RIDGE builds the SPD chart and nothing else -- it is a declared
        # gauge choice, not a search handle.  MEASURED: escalating it on
        # rejection (the LM coupling) is WORSE here, because with a Fisher
        # condition number of ~1e11 a large ridge collapses the natural step
        # onto steepest descent, which is precisely the direction the readback
        # shows the useful displacement is ORTHOGONAL to (cos_euclid ~ 0.00 vs
        # cos_fisher ~ 0.999).  A/B on pair 44: ridge-escalating 0.449 -> 0.327,
        # radius-only 0.449 -> 0.139.  So the RADIUS is the only handle, and it
        # needs enough RANGE: SHRINK=0.25 over `radius_steps` covers
        # 4^-steps of the natural length at one scorer evaluation per step.
        H = _spd_chart(gn, ridge)
        try:
            # radius=0 still returns the UNCONSTRAINED natural step and its
            # H-norm (the helper computes them before projecting), so this
            # probe costs one metric solve and no scorer evaluation.
            full = fisher_natural_cotangent_trust_region_step(H, cot, radius=0.0)
        except GeometryValidationError:
            readback["ridge_at_exit"] = float(ridge)
            reason = "singular_normal_equations"
            break
        nat = full.unconstrained_step
        nat_norm = full.unconstrained_hessian_norm
        if not np.isfinite(nat_norm) or nat_norm <= 0.0:
            readback["ridge_at_exit"] = float(ridge)
            reason = "zero_natural_gradient"
            break
        readback = _dual_metric_readback(H, cot, nat, pose, pose0, a, b, ab0, dim)
        readback["ridge_at_exit"] = float(ridge)
        step_radius = float(nat_norm) if delta is None \
            else min(float(delta), float(nat_norm))
        for _s in range(radius_steps):
            try:
                res = fisher_natural_cotangent_trust_region_step(
                    H, cot, radius=step_radius)
            except GeometryValidationError:
                reason = "singular_normal_equations"
                break
            step = res.step
            p2 = pose.copy()
            p2[:6] = p2[:6] + step[:6]
            a2 = a + (step[6] if dim > 6 else 0.0)
            b2 = b + (step[7] if dim > 7 else 0.0)
            t2 = q.theta(p2, a2, b2)
            if q.same(t2, (pose, a, b)):
                # below the shipped lattice: shrink, and do NOT spend a scorer
                # evaluation on a candidate that ships identical bytes.
                step_radius *= shrink
                continue
            all_below_quant = False
            val = sc.d_pose(f1_f, f1_u8, tp, t2[0], s_t, sel, t2[1], t2[2], g)
            if val < cur:
                pose, a, b, cur = t2[0], t2[1], t2[2], val
                cur6 = sc.pose6(f1_f, f1_u8, pose, s_t, sel, a, b, g)
                delta = step_radius / shrink
                accepted = True
                break
            step_radius *= shrink
        if reason == "singular_normal_equations":
            break
        if all_below_quant:
            # PROOF, not a tolerance: every radius the loop tried produced a
            # candidate on the SAME shipped float16 cells, so no continuation
            # can change the shipped d_pose.  Local optimality ON THE LATTICE
            # THAT SHIPS.
            reason = "step_below_shipped_quantization"
            break
        if not accepted:
            reason = "trust_radius_cap"
            break
    return (pose, a, b), cur, {"gn_stop": reason, "n_relin": n_relin,
                               **readback}


def _dual_metric_readback(H, cot, nat, pose, pose0, a, b, ab0, dim) -> dict:
    """Euclid-vs-Fisher cosines on the SAME displacement.

    Standing discipline (memory ``dual_metric_readback_euclid_cosine_vs_fisher``):
    the two cosines can SIGN-FLIP, so neither is reported alone.  ``cot`` is the
    Euclidean (co)gradient direction; ``nat = H^-1 cot`` is the Fisher-natural
    direction; ``disp`` is the displacement this solve has accumulated so far.
    """
    disp = np.zeros(dim, np.float64)
    disp[:6] = pose[:6] - pose0[:6]
    if dim > 6:
        disp[6] = a - ab0[0]
    if dim > 7:
        disp[7] = b - ab0[1]

    def _cos(u, v, M=None):
        if M is None:
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        else:
            nu = float(np.sqrt(max(u @ M @ u, 0.0)))
            nv = float(np.sqrt(max(v @ M @ v, 0.0)))
        if nu <= 0.0 or nv <= 0.0:
            return None
        num = float(u @ v) if M is None else float(u @ M @ v)
        return float(num / (nu * nv))

    ce = _cos(disp, cot)
    cf = _cos(disp, nat, H)
    out = {"cos_euclid_disp_grad": ce, "cos_fisher_disp_natural": cf,
           "metric_cos_sign_flip": (ce is not None and cf is not None
                                    and (ce > 0) != (cf > 0))}
    ev = np.linalg.eigvalsh(H)
    ev = ev[ev > 0]
    out["fisher_cond"] = float(ev.max() / ev.min()) if ev.size else None
    return out


def solve_pair(sc: RealizedScorer, ctx, theta0, d0, *,
               sweeps: int, relins: int, radius_steps: int, fit_ab: bool,
               scale_evals: int, rel_tol: float, ridge: float,
               shrink: float) -> dict:
    """Coordinate descent: scale ray, then GN, until neither improves.

    Emits the per-sweep descent curve, which is the measurement #850 asked for:
    *where does this solve actually converge, versus where the cap stopped it?*
    """
    theta, cur = theta0, d0
    trace = [{"sweep": 0, "d": float(cur), "phase": "start"}]
    stop = "sweep_cap"
    census: list[dict] = []
    for it in range(1, sweeps + 1):
        before = cur
        theta, cur, m1 = scale_line_search(sc, ctx, theta, cur,
                                           span=LAM_SPAN, max_evals=scale_evals)
        trace.append({"sweep": it, "d": float(cur), "phase": "scale"})
        theta, cur, m2 = fisher_trust_region_gn(
            sc, ctx, theta, cur, relins=relins, radius_steps=radius_steps,
            fit_ab=fit_ab, ridge=ridge, shrink=shrink)
        trace.append({"sweep": it, "d": float(cur), "phase": "gn"})
        census.append({**m1, **m2})
        rel = (before - cur) / max(before, 1e-30)
        if rel <= rel_tol:
            stop = "sweep_relative_gain_below_tol"
            break
    return {"d_final": float(cur), "theta": theta, "trace": trace,
            "stop": stop, "census": census, "n_evals": sc.n_evals}


# --------------------------------------------------------------------------- #
# archive-sourced shipped state (no transcription: the receiver is the source)
# --------------------------------------------------------------------------- #
def load_shipped(archive: Path) -> dict:
    import zipfile
    _ensure_paths()
    import tempfile

    import inflate_runner_v4d as rv4d
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with zipfile.ZipFile(archive) as z:
            z.extractall(d)
        dec = rv4d.Decoder(d)
        out = {
            "n_pairs": int(dec.n_pairs),
            "pose": np.asarray(dec.p_best, np.float64).copy(),
            "st_idx": np.asarray(dec.st_idx, np.int64).copy(),
            "st_vals": np.asarray(dec.st_vals, np.float64).copy(),
            "sel": np.asarray(dec.sel, np.int64).copy(),
            "ab": np.asarray(dec.ab, np.float64).copy(),
            "beta_idx": np.asarray(dec.beta_idx, np.int64).copy(),
            "beta_mags": tuple(float(x) for x in dec.beta_mags),
            "dim0_offset": (None if dec.dim0_offset is None
                            else float(dec.dim0_offset)),
        }
    return out


def build_context(base: str, base_archive: Path | None):
    _ensure_paths()
    import ddm_v4c_resolve as v4c
    oracle = v4c.build_oracle(base, s_r=1.0, archive=base_archive)
    comp = v4c.StaticComposer(oracle)

    def warp_ground_rot(f1_f, theta, s_t, rot):
        hg = comp.recv.pose_to_homography(theta, comp.K, comp.Kinv,
                                          float(s_t), float(rot), 0.0)
        return comp.recv.warp_rgb(f1_f, hg, comp.grid)

    comp.warp_ground_rot = warp_ground_rot
    return oracle, comp


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def mode_degen(args) -> None:
    """Algebraic + realized positive control of the scale degeneracy."""
    _assert_fd_steps()
    _ensure_paths()
    import torch
    torch.set_num_threads(int(args.threads))
    import pfs1_warp_receiver as recv

    K = recv.intrinsics_native()
    Ki = np.linalg.inv(K)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        p = np.array([rng.normal(33, 3), rng.normal(0, .2), rng.normal(0, .2),
                      rng.normal(0, .01), rng.normal(0, .01), rng.normal(0, .01)])
        s = float(rng.uniform(0.01, 0.3))
        lam = float(rng.uniform(0.2, 5.0))
        rot = float(rng.uniform(0.5, 1.5))
        h1 = recv.pose_to_homography(p, K, Ki, s, rot, 0.0)
        p2 = p.copy()
        p2[:3] *= lam
        h2 = recv.pose_to_homography(p2, K, Ki, s / lam, rot, 0.0)
        worst = max(worst, float(np.max(np.abs(h1 - h2))
                                 / max(1e-30, float(np.max(np.abs(h1))))))
    print(f"[degen] algebraic max rel |H(p,s)-H(lam*p,s/lam)| = {worst:.3e}")

    shipped = load_shipped(args.archive)
    oracle, comp = build_context(args.base, args.base_archive)
    sc = RealizedScorer(comp)
    q = ShippedQuant(shipped["dim0_offset"])
    final = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in args.final_jsonl.read_text().splitlines() if ln.strip()}
    pairs = [int(x) for x in args.pair_list.split(",")] if args.pair_list else \
        list(range(int(args.pairs)))
    rows = []
    for pidx in pairs:
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        tp = oracle.targets64[pidx].copy()
        s_t = float(shipped["st_vals"][shipped["st_idx"][pidx]])
        sel = int(shipped["sel"][pidx])
        g = float(shipped["beta_mags"][int(shipped["beta_idx"][pidx])])
        pose = shipped["pose"][pidx].copy()
        a, b = float(shipped["ab"][pidx][0]), float(shipped["ab"][pidx][1])
        d_ship = sc.d_pose(f1_f, f1_u8, tp, pose, s_t, sel, a, b, g)
        lam = float(args.degen_lam)
        # LEG 1 -- ALGEBRA: move lam and s_t inversely in EXACT arithmetic.
        # Isolates the degeneracy from the f16 lattice; must agree to roundoff.
        p2 = pose.copy()
        p2[:3] *= lam
        d_alt = sc.d_pose(f1_f, f1_u8, tp, p2, s_t / lam, sel, a, b, g)
        # LEG 2 -- REALIZABILITY: the move that actually ships.  To reach the
        # effective scale a NEW codeword ``lam*s_t`` would give, hold ``s_t`` at
        # the SHIPPED codeword and rescale the pose triple instead, quantized
        # exactly as the builder stores it.  If this tracks LEG 3, the ms8 win
        # is reachable through the pose column at zero index-stream cost.
        p3 = pose.copy()
        p3[:3] *= lam
        d_pose_route = sc.d_pose(f1_f, f1_u8, tp, q.pose(p3), s_t, sel, a, b, g)
        # LEG 3 -- the ms8 route to the same effective scale: move the codeword.
        d_st_route = sc.d_pose(f1_f, f1_u8, tp, pose, s_t * lam, sel, a, b, g)
        rows.append({"pair": pidx, "d_shipped": d_ship, "d_rescaled": d_alt,
                     "abs_err": abs(d_ship - d_alt),
                     "d_pose_route_quantized": d_pose_route,
                     "d_st_route": d_st_route,
                     "route_abs_err": abs(d_pose_route - d_st_route),
                     "route_rel_err": abs(d_pose_route - d_st_route)
                     / max(d_st_route, 1e-30),
                     "d_final_reported": float(final[pidx]["d_final"])
                     if pidx in final else None})
        r = rows[-1]
        print(f"  pair {pidx:3d}  d_ship {d_ship:.9f}  algebra|err| "
              f"{r['abs_err']:.3e}  pose-route {d_pose_route:.9f} vs "
              f"st-route {d_st_route:.9f}  rel {r['route_rel_err']:.3e}")
    errs = [r["abs_err"] for r in rows]
    rerrs = [r["route_rel_err"] for r in rows]
    canary = [abs(r["d_shipped"] - r["d_final_reported"]) for r in rows
              if r["d_final_reported"] is not None]
    print(f"[degen] LEG1 algebra   max |d(p,s) - d(lam*p, s/lam)| = "
          f"{max(errs):.3e}")
    print(f"[degen] LEG2/3 realizable  max rel |pose-route - st-route| = "
          f"{max(rerrs):.3e}")
    if canary:
        print(f"[degen] CANARY max |d_realized - d_final_reported| = "
              f"{max(canary):.3e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": SCHEMA, "mode": "degen", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer_moved": False,
        "algebraic_max_rel_H_err": worst, "degen_lam": float(args.degen_lam),
        "realized_max_abs_err": float(max(errs)),
        "route_max_rel_err": float(max(rerrs)),
        "canary_max_abs_err": float(max(canary)) if canary else None,
        "rows": rows,
    }, indent=1) + "\n")
    print(f"[degen] wrote {args.out}")


def mode_solve(args) -> None:
    _assert_fd_steps()
    for tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[tv] = str(args.threads)
    _ensure_paths()
    import torch
    torch.set_num_threads(int(args.threads))

    shipped = load_shipped(args.archive)
    oracle, comp = build_context(args.base, args.base_archive)
    q = ShippedQuant(shipped["dim0_offset"])
    final = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in args.final_jsonl.read_text().splitlines() if ln.strip()}

    order = _pair_order(args, final, shipped)
    seq = [p for i, p in enumerate(order) if i % args.nshards == args.shard]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jl = args.out_dir / f"pj2_solve_shard{args.shard}.jsonl"
    cache = {int(json.loads(ln)["pair"])
             for ln in (jl.read_text().splitlines() if jl.exists() else [])
             if ln.strip()}
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    print(f"[pj2 solve] shard {args.shard}/{args.nshards} pairs={len(seq)} "
          f"cached={len(cache)} sweeps={args.sweeps} relins={args.relins} "
          f"radius_steps={args.radius_steps} ridge={args.ridge} "
          f"fit_ab={not args.no_fit_ab}", flush=True)
    ndone = 0
    for pidx in seq:
        if pidx in cache:
            continue
        if (time.time() - t0) > args.max_minutes * 60.0:
            print(f"[pj2 solve] wall cap at pair {pidx}; rerun to resume",
                  flush=True)
            break
        f1_u8 = oracle.f1(pidx)
        f1_f = f1_u8.astype(np.float64)
        tp = oracle.targets64[pidx].copy()
        s_t = float(shipped["st_vals"][shipped["st_idx"][pidx]])
        sel = int(shipped["sel"][pidx])
        g = float(shipped["beta_mags"][int(shipped["beta_idx"][pidx])])
        pose0 = shipped["pose"][pidx].copy()
        a0, b0 = float(shipped["ab"][pidx][0]), float(shipped["ab"][pidx][1])
        theta0 = q.theta(pose0, a0, b0)
        sc = RealizedScorer(comp)
        ctx = (q, f1_f, f1_u8, tp, s_t, sel, g)
        d0 = sc.d_pose(f1_f, f1_u8, tp, theta0[0], s_t, sel,
                       theta0[1], theta0[2], g)
        res = solve_pair(sc, ctx, theta0, d0, sweeps=args.sweeps,
                         relins=args.relins, radius_steps=args.radius_steps,
                         fit_ab=not args.no_fit_ab,
                         scale_evals=args.scale_evals, rel_tol=args.rel_tol,
                         ridge=args.ridge, shrink=args.tr_shrink)
        pose, a, b = res["theta"]
        row = {
            "pair": int(pidx),
            "d_start_realized": float(d0),
            "d_final": float(res["d_final"]),
            "d_ms8_reported": float(final[pidx]["d_final"])
            if pidx in final else None,
            "p": [float(x) for x in pose], "a": float(a), "b": float(b),
            "selector": sel, "beta_idx": int(shipped["beta_idx"][pidx]),
            "beta_mag": g, "s_t": s_t,
            "st_idx": int(shipped["st_idx"][pidx]),
            "lam_effective": float(pose[0] / pose0[0]) if pose0[0] else None,
            "stop": res["stop"], "census": res["census"],
            "trace": res["trace"], "n_evals": int(res["n_evals"]),
            "source": "pj2_joint",
            # (a, b) is re-solved here JOINTLY with pose: the held partners the
            # solve saw are {selector, beta_mag, s_t} + base/archive identity.
            # pose is co-solved, so it is deliberately NOT stamped as a held
            # partner (that would manufacture a freshness claim); the emitted
            # "p" field records the co-solved state (a, b) is valid at.
            FIT_CONTEXT_KEY: stamp_fit_context(
                coefficient="ab_gain_bias",
                partners={"selector": sel, "beta_mag": g, "s_t": s_t},
                base=str(args.base),
                vehicle=args.archive.name,
                fit_sign=(1.0 if pose[5] >= 0.0 else -1.0)),
        }
        fj.write(json.dumps(row) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        ndone += 1
        if ndone % 5 == 1 or ndone < 4:
            print(f"  [pj2 {pidx:3d}] {d0:.6f} -> {res['d_final']:.6f} "
                  f"({res['stop']}, {res['n_evals']} evals, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    fj.close()
    print(f"[pj2 solve] shard {args.shard} done {ndone} in "
          f"{time.time()-t0:.0f}s", flush=True)


def _pair_order(args, final, shipped) -> list[int]:
    """Mass-first ordering: the tail carries the term.

    ms8 measured the post-refit residual concentration -- top 10 pairs 69.9% of
    d_pose mass, top 100 88.0% -- so a wall-capped run that walks pairs in index
    order spends its budget where the objective is already ~0.  Ordering by the
    incumbent per-pair d_pose puts every minute against mass.  The ordering is
    RECORDED so a partial run's coverage is auditable, and the *reported* n600
    mean always uses the shipped value for unsolved pairs (never a subset mean).
    """
    n = int(shipped["n_pairs"])
    if args.pair_list:
        return [int(x) for x in args.pair_list.split(",")]
    d = np.array([float(final[i]["d_final"]) if i in final else 0.0
                  for i in range(n)])
    idx = list(np.argsort(-d))
    return [int(i) for i in idx[:int(args.pairs)]]


def mode_report(args) -> None:
    shipped = load_shipped(args.archive)
    n = int(shipped["n_pairs"])
    final = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in args.final_jsonl.read_text().splitlines() if ln.strip()}
    solved: dict[int, dict] = {}
    for f in sorted(args.out_dir.glob("pj2_solve_shard*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                solved[int(r["pair"])] = r
    base = np.array([float(final[i]["d_final"]) for i in range(n)])
    new = base.copy()
    regressed = []
    for i, r in solved.items():
        # MONOTONE GUARD: never ship a pair the solve made worse.  sv1 measured
        # 10/60 pairs regressing under f16 quantization on the sister surface.
        if r["d_final"] < base[i]:
            new[i] = r["d_final"]
        elif r["d_final"] > base[i]:
            regressed.append((i, base[i], r["d_final"]))
    stops: dict[str, int] = {}
    gn_stops: dict[str, int] = {}
    for r in solved.values():
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
        for c in r["census"]:
            gn_stops[c["gn_stop"]] = gn_stops.get(c["gn_stop"], 0) + 1
    d_new, d_old = float(new.mean()), float(base.mean())
    rep = {
        "schema": SCHEMA, "mode": "report", "utc": _utc(),
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "promotion_eligible": False,
        "pointer_moved": False,
        "n_pairs": n, "n_solved": len(solved),
        "coverage_of_incumbent_mass": float(
            base[sorted(solved)].sum() / base.sum()) if solved else 0.0,
        "d_pose_incumbent": d_old,
        "d_pose_after": d_new,
        "contribution_incumbent": contribution(d_old),
        "contribution_after": contribution(d_new),
        "delta_S_pose": contribution(d_new) - contribution(d_old),
        "pairs_improved": int(sum(1 for i in solved if new[i] < base[i])),
        "pairs_regressed_guarded": [
            {"pair": i, "was": w, "got": gt} for i, w, gt in regressed],
        "sweep_stop_census": stops,
        "gn_stop_census": gn_stops,
        "total_evals": int(sum(r["n_evals"] for r in solved.values())),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "pairs_regressed_guarded"}, indent=1))
    print(f"[report] wrote {args.out}")


def mode_emit(args) -> None:
    """Fold solved rows into a builder-ready final JSONL (monotone-guarded)."""
    final = {int(json.loads(ln)["pair"]): json.loads(ln)
             for ln in args.final_jsonl.read_text().splitlines() if ln.strip()}
    solved: dict[int, dict] = {}
    for f in sorted(args.out_dir.glob("pj2_solve_shard*.jsonl")):
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                solved[int(r["pair"])] = r
    n = len(final)
    out, replaced = [], 0
    for i in range(n):
        row = dict(final[i])
        r = solved.get(i)
        if r is not None and r["d_final"] < float(row["d_final"]):
            row["p"] = r["p"]
            row["a"] = r["a"]
            row["b"] = r["b"]
            row["d_final"] = r["d_final"]
            row["source"] = "pj2_joint"
            # This stage carries a solved row forward without re-solving (a, b):
            # CARRY the upstream fit context rather than re-stamping.
            if FIT_CONTEXT_KEY in r:
                row[FIT_CONTEXT_KEY] = r[FIT_CONTEXT_KEY]
            replaced += 1
        out.append(row)
    args.emit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.emit_jsonl.write_text(
        "".join(json.dumps(r) + "\n" for r in out))
    d = np.array([float(r["d_final"]) for r in out])
    print(f"[emit] {args.emit_jsonl} rows={len(out)} replaced={replaced} "
          f"mean d_pose {d.mean():.8f} contribution {contribution(d.mean()):.7f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True,
                    choices=["degen", "solve", "report", "emit"])
    ap.add_argument("--archive", type=Path, default=LIVE_ARCHIVE)
    ap.add_argument("--base", default=BASE_LABEL)
    ap.add_argument("--base-archive", type=Path, default=None)
    ap.add_argument("--final-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_pj2_20260802"))
    ap.add_argument("--out", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_pj2_20260802"
                                 "/pj2_receipt.json"))
    ap.add_argument("--emit-jsonl", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_pj2_20260802"
                                 "/final_pj2.jsonl"))
    ap.add_argument("--pairs", type=int, default=N_PAIRS)
    ap.add_argument("--pair-list", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sweeps", type=int, default=6)
    ap.add_argument("--relins", type=int, default=8)
    ap.add_argument("--radius-steps", type=int, default=20)
    ap.add_argument("--tr-shrink", type=float, default=TR_SHRINK)
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="explicit ridge making the GN pullback an SPD chart; "
                         "the canonical Fisher helper refuses to damp for us")
    ap.add_argument("--scale-evals", type=int, default=40)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    ap.add_argument("--no-fit-ab", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--degen-lam", type=float, default=1.375)
    args = ap.parse_args()
    {"degen": mode_degen, "solve": mode_solve,
     "report": mode_report, "emit": mode_emit}[args.mode](args)


if __name__ == "__main__":
    main()
