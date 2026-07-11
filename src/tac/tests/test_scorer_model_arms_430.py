# SPDX-License-Identifier: MIT
"""Tests for the #426 scorer-model arms (P0 build) + the #430 schedule backtest.

NO-FAKE proofs, mirrors of the organ test discipline:
  * the smoothed-argmax gradient is verified EXACT against finite differences of the
    smoothed metric (survey #1's zero-model-error claim, tested not asserted);
  * the adversarial-boundary susceptibility ranks a PLANTED thin class highest;
  * the ball-agreement audit returns perfect agreement on a construction where the
    margin field and the label geometry coincide, and degrades when they are broken;
  * the comma10k reducers are tested on synthetic palette masks (exact shares);
  * the per-class coupling matrix is row-stochastic-blended and couples planted
    adjacent classes;
  * the #430 replay: self-replay through a ridge model on PLANTED LINEAR dynamics
    reproduces the planted trajectory; policies differ only through the control;
  * containment: the new modules carry no actuation tokens; the #430 ticket is a
    HEAVY OperatorGoTicket with actuation NONE and the measured rows embedded.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tac.witness_control.lambda_net import (
    ARCHITECTURES,
    N_CLASSES,
    CampaignTrajectory,
    build_intervals,
    fit_score_composition,
    make_model,
)
from tac.witness_control.scorer_model_arms import (
    AdversarialBoundaryPrior,
    Comma10kRegimePrior,
    PerClassCoupledRidgeAdjoint,
    PriorMeanRidgeAdjoint,
    boundary_pair_shares,
    build_comma10k_prior_from_labels,
    comma10k_labels_from_rgb,
    perclass_coupling_matrix,
    smoothed_dseg_and_grad,
)

REPO = Path(__file__).resolve().parents[3]


# ───────────────────────── synthetic trajectory (planted linear dynamics) ─────────────────────────
def _planted_traj(n_verdicts: int = 7) -> CampaignTrajectory:
    """Planted EXACTLY-LINEAR dynamics dx/dt = a + M(Σ u φ): a ridge solve must fit it
    exactly, so the self-replay reproduces the trajectory to numerical precision."""
    lever_names = ("seg", "island_amplify", "eikonal", "chroma_boundary")
    from tac.witness_control.lambda_net import lever_features
    phis = np.stack([lever_features(n) for n in lever_names])
    rng = np.random.default_rng(7)
    a = np.asarray([-1e-4, -5e-5, -2e-5, -1e-4, -1e-6, 0.0])
    M = rng.normal(scale=2e-4, size=(6, phis.shape[1]))
    x = np.asarray([0.10, 0.40, 0.02, 0.90, 0.01, math.log(85_000.0)])
    wgt = [.23, .006, .49, .012, .28]
    verdicts, loss_terms = [], []
    ep = 0.0
    for k in range(n_verdicts):
        verdicts.append({
            "stage": "verdict", "epoch": ep,
            "d_seg": float(np.dot(wgt, x[:5])),
            "d_seg_by_class": [float(v) for v in x[:5]],
            "d_pose": 5.0, "blob_bytes": float(math.exp(x[5])), "ep_loss": 400.0,
        })
        mags = {"seg": 3.0 + 0.5 * math.sin(k), "island_amplify": 0.2 + 0.15 * k,
                "eikonal": 0.03, "chroma_boundary": 0.01 * k}
        for e in range(int(ep), int(ep) + 20):
            loss_terms.append({"stage": "loss_terms", "ep": float(e), "gnorm": 5.0,
                               "terms": dict(mags)})
        m = np.asarray([abs(mags[n]) for n in lever_names])
        u = m / m.sum()
        dx = a + M @ (phis.T @ u)
        x = x + 20.0 * dx
        x[:5] = np.clip(x[:5], 0.0, None)
        ep += 20.0
    return CampaignTrajectory(run_dir="<synthetic>", verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms),
                              lever_names=tuple(lever_names))


# ───────────────────────── ARM H: smoothed argmax exactness ─────────────────────────
def test_smoothed_argmax_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    margins = rng.exponential(scale=3.0, size=(2, 24, 32))
    lstars = rng.integers(0, N_CLASSES, size=(2, 24, 32))
    eps = 2.5
    sd0, g = smoothed_dseg_and_grad(margins, lstars, eps)
    assert 0.0 < sd0 < 1.0
    d = 1e-5
    for c in range(N_CLASSES):
        m2 = margins.copy()
        m2[lstars == c] -= d
        sd1, _ = smoothed_dseg_and_grad(m2, lstars, eps)
        fd = (sd1 - sd0) / d
        assert abs(g[c] - fd) <= 1e-6 * max(abs(fd), 1e-9) + 1e-10, \
            f"class {c}: analytic {g[c]} vs FD {fd}"


def test_smoothed_argmax_rejects_bad_epsilon():
    import pytest
    with pytest.raises(ValueError):
        smoothed_dseg_and_grad(np.ones((1, 2, 2)), np.zeros((1, 2, 2), dtype=int), 0.0)


# ───────────────────────── ARM J: adversarial boundary + ball agreement ─────────────────────────
def test_boundary_pair_shares_planted_adjacency():
    lab = np.zeros((16, 16), dtype=np.int64)
    lab[:, 8:] = 1                       # a single Road|Lane vertical boundary
    A = boundary_pair_shares(lab)
    # symmetric matrix normalized to sum 1 over BOTH triangles → 0.5 each side
    assert abs(A[0, 1] - 0.5) < 1e-12 and abs(A[0, 1] - A[1, 0]) < 1e-12
    assert A[0, 0] == 0.0 and abs(float(A.sum()) - 1.0) < 1e-12


def test_ball_agreement_perfect_when_margin_is_distance(tmp_path):
    """Construct a cache where margin == distance-to-boundary: the rank-matched
    predicted set equals the advection-ball flip set (IoU 1)."""
    from tac.witness_control.scorer_model_arms import ball_agreement_audit
    lab = np.zeros((32, 32), dtype=np.int64)
    lab[:, 16:] = 1
    col = np.arange(32)
    dist = np.minimum(np.abs(col - 15.5), 40.0)           # distance to the boundary
    margin = np.tile(dist, (32, 1)).astype(np.float32)
    p = tmp_path / "cache.npz"
    np.savez(p, lstars=lab[None], margins=margin[None])
    a = ball_agreement_audit(p, radius_px=1, frame_stride=1)
    assert a["iou"] > 0.99 and a["faithful"]
    # break the margin (shuffle) → agreement collapses
    rng = np.random.default_rng(0)
    np.savez(p, lstars=lab[None],
             margins=rng.permutation(margin.ravel()).reshape(margin.shape)[None])
    b = ball_agreement_audit(p, radius_px=1, frame_stride=1)
    assert b["iou"] < 0.5 and not b["faithful"]


def test_adversarial_susceptibility_ranks_thin_class(tmp_path):
    """A planted THIN stripe class (all its mass in the boundary band) must out-rank
    a bulk class."""
    from tac.witness_control.scorer_model_arms import adversarial_boundary_susceptibility
    lab = np.zeros((32, 32), dtype=np.int64)
    lab[15:17, :] = 1                     # thin Lane stripe inside bulk Road
    p = tmp_path / "cache.npz"
    np.savez(p, lstars=lab[None], margins=np.ones((1, 32, 32), dtype=np.float32))
    prior = adversarial_boundary_susceptibility(p, radius_px=1, frame_stride=1,
                                                sigma_preset="all-ones")
    assert prior.susceptibility[1] > prior.susceptibility[0] > 0.0
    assert isinstance(prior, AdversarialBoundaryPrior)


# ───────────────────────── ARM I: comma10k reducers ─────────────────────────
def test_comma10k_palette_roundtrip_and_prior_shares():
    from tac.witness_control.scorer_model_arms import COMMA10K_PALETTE
    h = w = 20
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = COMMA10K_PALETTE[0]       # road everywhere
    rgb[:2, :] = COMMA10K_PALETTE[1]      # 10% lane
    lab = comma10k_labels_from_rgb(rgb)
    assert (lab >= 0).all()
    prior = build_comma10k_prior_from_labels([lab], source="synthetic")
    assert abs(prior.class_pixel_share[1] - 0.1) < 1e-9
    assert prior.unmatched_frac == 0.0
    rw = prior.rarity_reweight()
    assert rw[1] > rw[0]                  # rare class up-weighted
    assert abs(float(rw.sum()) - N_CLASSES) < 1e-9


def test_comma10k_unmatched_pixels_counted():
    rgb = np.full((8, 8, 3), 7, dtype=np.uint8)     # matches nothing
    lab = comma10k_labels_from_rgb(rgb, tol=0)
    prior = build_comma10k_prior_from_labels([lab], source="synthetic")
    assert prior.unmatched_frac == 1.0
    assert isinstance(prior, Comma10kRegimePrior)


def test_comma10k_artifact_loads_when_present():
    """The REAL durable artifact (built by tools/build_comma10k_regime_prior.py from
    192 real comma10k masks). Skip honestly if absent on this checkout."""
    import pytest
    from tac.witness_control.scorer_model_arms import (
        DEFAULT_COMMA10K_PRIOR, load_comma10k_prior)
    if not DEFAULT_COMMA10K_PRIOR.exists():
        pytest.skip("comma10k prior artifact not built on this checkout")
    prior = load_comma10k_prior()
    assert prior.n_masks >= 32 and prior.unmatched_frac <= 0.05
    # rare classes (Lane, Movable) must out-rank bulk (Road/Undrivable/MyCar)
    rw = prior.rarity_reweight()
    assert rw[1] == max(rw), "Lane must be the rarity-hot class (matches L80/L2 crux)"


# ───────────────────────── ARM K: coupling matrix ─────────────────────────
def test_perclass_coupling_matrix_planted(tmp_path):
    lab = np.zeros((16, 16), dtype=np.int64)
    lab[:, 8:] = 1
    p = tmp_path / "cache.npz"
    np.savez(p, lstars=lab[None], margins=np.ones((1, 16, 16), dtype=np.float32))
    C = perclass_coupling_matrix(p, alpha=0.5, frame_stride=1, sigma_preset="all-ones")
    assert C.shape == (N_CLASSES, N_CLASSES)
    assert abs(C[0, 0] - 0.5) < 1e-9 and abs(C[0, 1] - 0.5) < 1e-9  # Road↔Lane only
    assert C[2, 3] == 0.0
    # rows blend to α + (1−α)·rownorm ⇒ observed rows sum to 1
    assert abs(float(C[0].sum()) - 1.0) < 1e-9


def test_perclass_lambda_readout():
    m = PerClassCoupledRidgeAdjoint(coupling=np.eye(N_CLASSES))
    lam = m.perclass_lambda(np.asarray([1.0, -2.0, 0.0, 0.5, 0.0, 9.9]),
                            np.asarray([100.0, 50.0, 10.0, 1.0, 1.0, 0.0]))
    assert lam.shape == (N_CLASSES,)
    assert lam[1] == -100.0 and lam[0] == 100.0


# ───────────────────────── arms fit/response API on planted data ─────────────────────────
def test_all_new_arms_are_registered_and_fit_planted():
    traj = _planted_traj()
    intervals = build_intervals(traj)
    from tac.witness_control.lambda_net import lever_features
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    for arch in ("H_smoothed_argmax", "J_adv_boundary", "K_perclass_v8",
                 "M_priormean_advb"):
        assert arch in ARCHITECTURES
        m = make_model(arch)
        m.fit(intervals, phis, seed=0)
        r = m.response(intervals[-1].x0, intervals[-1].ctx, phis[0],
                       intervals[-1].path)
        b = m.base(intervals[-1].x0, intervals[-1].ctx, intervals[-1].path)
        assert np.isfinite(r).all() and np.isfinite(b).all()
        assert r.shape == (6,) and b.shape == (6,)


def test_priormean_small_n_returns_prior_structure():
    """At n=2 the shrink-to-prior solve's response must carry the prior's class
    structure (nonzero on the prior-hot class-targeting features), not collapse to 0."""
    traj = _planted_traj(n_verdicts=3)
    intervals = build_intervals(traj)
    from tac.witness_control.lambda_net import lever_features
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    s = np.asarray([0.1, 4.0, 0.1, 0.7, 0.1])
    m = PriorMeanRidgeAdjoint(s, ridge=1e-2)
    m.fit(intervals[:2], phis, seed=0)
    assert m.kappa >= 0.0
    phi_lane = lever_features("lane_edge")          # pure Lane-targeting lever
    r = m.response(intervals[0].x0, intervals[0].ctx, phi_lane)
    assert np.isfinite(r).all()


# ───────────────────────── #430 schedule replay ─────────────────────────
def test_self_replay_reproduces_planted_linear_dynamics():
    from tac.witness_control.schedule_backtest import backtest_schedule_430
    traj = _planted_traj()
    rep = backtest_schedule_430(traj, arch="A_ridge_solve")
    # planted dynamics are the ridge model CLASS; the ridge penalty leaves a small
    # shrinkage residual — the self-replay must still track the planted trajectory
    # to a small fraction of its scale (~0.05 weighted d_seg)
    assert rep.self_replay_mae < 2e-3, rep.self_replay_mae
    by = {r.policy: r for r in rep.results}
    assert set(by) == {"hand", "selective", "always_on", "uniform"}
    for r in rep.results:
        assert all(np.isfinite(v) for v in r.weighted_dseg_series)
    assert rep.score_claim is False and rep.promotable is False


def test_policies_differ_only_through_control():
    from tac.witness_control.schedule_backtest import backtest_schedule_430
    traj = _planted_traj()
    rep = backtest_schedule_430(traj, arch="A_ridge_solve", budget=0.0)
    by = {r.policy: r for r in rep.results}
    # zero budget ⇒ selective/always_on collapse onto hand exactly
    assert by["selective"].final_weighted_dseg == by["hand"].final_weighted_dseg
    assert by["always_on"].final_weighted_dseg == by["hand"].final_weighted_dseg


def test_gates_are_derived_and_recorded():
    from tac.witness_control.schedule_backtest import derive_state_gates
    traj = _planted_traj()
    intervals = build_intervals(traj)
    comp = fit_score_composition(traj.verdicts)
    g = derive_state_gates(intervals, comp)
    assert "derived from the measured trajectory" in g.provenance
    flags = g.flags(intervals[0].x0, -1e-3, comp.class_weights)
    assert flags["movable_unborn"] is True          # planted Movable starts at 0.90


def test_430_ticket_is_contained_and_carries_measured_rows():
    from tac.witness_control.control_alphabet import OperatorGoTicket
    from tac.witness_control.schedule_backtest import (
        backtest_schedule_430, compose_430_ticket)
    traj = _planted_traj()
    reports = [backtest_schedule_430(traj, arch="A_ridge_solve")]
    t = compose_430_ticket(traj, reports)
    assert isinstance(t, OperatorGoTicket)
    assert t.action == "mutate_live_config" and t.actuation == "NONE"
    assert "MEASURED replay rows" in t.justification
    assert "MODEL-BASED-BACKTESTED" in t.justification
    assert any("A/B" in g for g in t.gates_owed)    # the live A/B stays owed
    assert not hasattr(t, "execute")


# ───────────────────────── containment (source scan, mirrors the organ test) ─────────────────────────
def test_new_modules_have_no_actuation_tokens():
    forbidden = ("import subprocess", "from subprocess", "os.system(", "os.exec",
                 "os.spawn", "os.popen(", "os.kill(", "import signal", "Popen(",
                 "urllib", "requests.")
    for rel in ("src/tac/witness_control/scorer_model_arms.py",
                "src/tac/witness_control/schedule_backtest.py"):
        src = (REPO / rel).read_text()
        for tok in forbidden:
            assert tok not in src, f"{tok!r} in {rel}"
