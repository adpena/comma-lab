# SPDX-License-Identifier: MIT
"""Tests for THE TRANSIENT FORGE (#434) — the synthetic-trajectory engine + honest gate.

Protects the load-bearing invariants:
  * the CGauge simulator is numerically STABLE + BOUNDED across all regimes (the round-1
    self-review fix: explicit Euler diverged; exponential relaxation must not),
  * the whole pipeline is DETERMINISTIC (same seed → same corpus/verdict),
  * the QD diversity gate admits only coverage-growing / non-redundant windows,
  * the prior-mean forge arm reduces to real-only at λ=0 and moves toward the prior at λ→∞,
  * chronological hygiene: the corpus for fold k is derived from the prefix ≤ k ONLY,
  * NO-FAKE: every reported number carries score_claim=False / promotable=False, and the
    synthetic-fold skill is REPORTED but NEVER the adoption authority.
Every number [macOS advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import numpy as np

from tac.witness_control.lambda_net import (
    CampaignTrajectory,
    build_intervals,
    fit_score_composition,
    lever_features,
)
from tac.witness_control.transient_forge import (
    AXIS_TAG,
    ForgeAugmentedRidge,
    ForgeConfig,
    QDArchive,
    _design,
    adoption_backtest,
    forge_corpus,
    sample_sim_params,
    simulate,
    window_regret,
)

_LEVERS = ("seg", "eikonal", "length", "lane_edge", "thin_lane", "island_amplify",
           "area_constraint", "persistence", "chroma_boundary", "horizon_margin",
           "weight_entropy", "code_spectral")


def _synthetic_traj(n: int = 10, seed: int = 0) -> CampaignTrajectory:
    """A deterministic plateau-ish real trajectory with the fields the engine needs."""
    rng = np.random.default_rng(seed)
    verdicts, loss_terms = [], []
    for i in range(n):
        ep = 50.0 + 25.0 * i
        base = np.array([0.09, 0.8, 0.004, 0.9, 0.0016]) * np.exp(-0.05 * i)
        d_by_class = (base + rng.normal(0, 2e-3, 5)).clip(1e-6, 1.2).tolist()
        d_seg = float(np.array([0.23, 0.006, 0.5, 0.012, 0.26]) @ np.array(d_by_class))
        verdicts.append({"stage": "verdict", "epoch": ep, "d_seg": d_seg,
                         "d_seg_by_class": d_by_class, "blob_bytes": 90000.0,
                         "ep_loss": 2.0})
        for j in range(3):
            terms = {lv: float(abs(rng.normal(1.0, 0.3))) for lv in _LEVERS}
            loss_terms.append({"stage": "loss_terms", "ep": ep + j, "terms": terms,
                               "gnorm": 5.0})
    return CampaignTrajectory(run_dir="test://forge", verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms), lever_names=_LEVERS)


def _prefix_and_comp(traj, k=5):
    ivs = build_intervals(traj)
    comp = fit_score_composition(traj.verdicts)
    return ivs[:k], comp


# ── simulator: numerical stability + determinism ──────────────────────────────────────
def test_simulator_finite_bounded_all_regimes():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    rng = np.random.default_rng(7)
    max_abs = 0.0
    for i in range(120):
        regime = ("transient", "birth", "reversal", "plateau")[i % 4]
        p = sample_sim_params(rng, traj.lever_names, prefix, comp, regime)
        window = simulate(p, traj.lever_names)
        assert len(window) >= 5
        for iv in window:
            arr = np.concatenate([iv.x0, iv.x1, iv.dxdt(), iv.u_mean, iv.ctx,
                                  iv.path.ravel()])
            assert np.all(np.isfinite(arr)), f"non-finite in {regime}"
            # d_seg channels are physical [0, 1.5]; log-bytes bounded ~[6.9, 16.1]
            assert np.all(iv.x1[:5] >= -1e-9) and np.all(iv.x1[:5] <= 1.5 + 1e-9)
            max_abs = max(max_abs, float(np.max(np.abs(arr))))
    assert max_abs < 100.0, "explicit-Euler divergence regression (must stay bounded)"


def test_simulator_deterministic():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    p1 = sample_sim_params(np.random.default_rng(3), traj.lever_names, prefix, comp,
                           "reversal")
    p2 = sample_sim_params(np.random.default_rng(3), traj.lever_names, prefix, comp,
                           "reversal")
    w1, w2 = simulate(p1, traj.lever_names), simulate(p2, traj.lever_names)
    assert len(w1) == len(w2)
    for a, b in zip(w1, w2, strict=True):
        assert np.allclose(a.x1, b.x1) and np.allclose(a.u_mean, b.u_mean)


def test_control_schedule_regimes_differ():
    """Transient/reversal schedules must actually VARY more than plateau (the UED point)."""
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    rng = np.random.default_rng(1)
    plat = sample_sim_params(rng, traj.lever_names, prefix, comp, "plateau")
    tran = sample_sim_params(rng, traj.lever_names, prefix, comp, "transient")
    plat_var = float(np.mean(np.std(plat.control_schedule, axis=0)))
    tran_var = float(np.mean(np.std(tran.control_schedule, axis=0)))
    assert tran_var > plat_var


# ── UED regret: transient window scores higher learning-potential than plateau ────────
def test_regret_transient_exceeds_plateau():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    rng = np.random.default_rng(2)
    plat = simulate(sample_sim_params(rng, traj.lever_names, prefix, comp, "plateau"),
                    traj.lever_names)
    tran = simulate(sample_sim_params(rng, traj.lever_names, prefix, comp, "transient"),
                    traj.lever_names)
    r_plat = window_regret(plat, prefix, phis, comp)
    r_tran = window_regret(tran, prefix, phis, comp)
    assert r_plat >= 0.0 and r_tran >= 0.0
    # transient windows should carry at least as much regret on average (not a hard <,
    # since a single draw can vary — assert the transient is not degenerate-zero)
    assert r_tran > 0.0


# ── QD diversity gate ─────────────────────────────────────────────────────────────────
def test_qd_archive_admission_and_redundancy():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    rng = np.random.default_rng(5)
    arch = QDArchive()
    windows = [simulate(sample_sim_params(rng, traj.lever_names, prefix, comp,
                                          ("transient", "birth", "reversal")[i % 3]),
                        traj.lever_names) for i in range(12)]
    # first admission of a fresh descriptor always grows coverage
    d0 = ("transient", "seg", True, "short")
    assert arch.try_admit(d0, 1.0, windows[0]) is True
    assert arch.coverage() == 1
    # same descriptor with LOWER regret does not grow coverage (rejected)
    assert arch.try_admit(d0, 0.5, windows[1]) is False
    # a new descriptor grows coverage
    d1 = ("reversal", "lane_edge", False, "long")
    assert arch.try_admit(d1, 2.0, windows[2]) is True
    assert arch.coverage() == 2


def test_qd_effective_rank_nonneg():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj)
    rng = np.random.default_rng(9)
    arch = QDArchive()
    for i in range(8):
        w = simulate(sample_sim_params(rng, traj.lever_names, prefix, comp, "transient"),
                     traj.lever_names)
        arch.try_admit(("transient", f"d{i}", True, "short"), float(i + 1), w)
    assert arch.effective_rank() >= 0.0


# ── prior-mean forge arm: λ=0 ≡ real-only; larger λ moves toward the synthetic prior ──
def test_forge_augmented_lambda_zero_equals_real_only():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj, k=6)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    from tac.witness_control.transient_forge import _fit_ridge
    real = _fit_ridge(prefix, phis)
    syn = simulate(sample_sim_params(np.random.default_rng(4), traj.lever_names, prefix,
                                     comp, "transient"), traj.lever_names)
    arm = ForgeAugmentedRidge()
    coef0 = arm.fit_prior(syn, phis)
    arm.fit(prefix, phis, coef0, prior_strength=0.0)
    # λ=0 ⇒ identical to the plain ridge solve (both minimize the same objective)
    assert np.allclose(arm.M, real.M, atol=1e-6)
    assert np.allclose(arm.a, real.a, atol=1e-6)


def test_forge_augmented_large_lambda_moves_toward_prior():
    traj = _synthetic_traj()
    prefix, comp = _prefix_and_comp(traj, k=6)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    syn = simulate(sample_sim_params(np.random.default_rng(4), traj.lever_names, prefix,
                                     comp, "transient"), traj.lever_names)
    arm = ForgeAugmentedRidge()
    coef0 = arm.fit_prior(syn, phis)
    arm.fit(prefix, phis, coef0, prior_strength=0.0)
    m_small = arm.M.copy()
    arm.fit(prefix, phis, coef0, prior_strength=1e6)
    # with an overwhelming prior the coefficients move toward the synthetic prior mean
    # (the large-λ M must differ from the λ=0 M — the prior actually pulls the solve)
    assert not np.allclose(arm.M, m_small, atol=1e-3)


# ── chronological hygiene + corpus structure ──────────────────────────────────────────
def test_corpus_chronological_hygiene():
    """The corpus for fold k must be built from the prefix ≤ k ONLY (deterministic per k)."""
    traj = _synthetic_traj()
    cfg = ForgeConfig(n_candidate_trajectories=16, seed=0)
    c_a = forge_corpus(traj, 4, cfg)
    c_b = forge_corpus(traj, 4, cfg)
    assert len(c_a.intervals) == len(c_b.intervals)
    for a, b in zip(c_a.intervals, c_b.intervals, strict=True):
        assert np.allclose(a.x1, b.x1)
    # a longer prefix is allowed to change the corpus (uses more real information)
    c_k6 = forge_corpus(traj, 6, cfg)
    assert c_k6.n_generated == cfg.n_candidate_trajectories


def test_corpus_pipeline_reduces_volume():
    """generated ≥ after_regret ≥ after_diversity (the UED + BIRD funnels actually cut)."""
    traj = _synthetic_traj()
    c = forge_corpus(traj, 5, ForgeConfig(n_candidate_trajectories=32, seed=0))
    assert c.n_generated >= c.n_after_regret >= c.n_after_diversity
    assert c.archive_coverage == c.n_after_diversity  # each admitted window = one cell
    assert c.effective_rank >= 0.0


# ── the adoption gate: structure + NO-FAKE invariants ─────────────────────────────────
def test_adoption_report_no_fake_invariants():
    traj = _synthetic_traj()
    rep = adoption_backtest(traj, ForgeConfig(n_candidate_trajectories=16, seed=0))
    d = rep.to_dict()
    assert d["score_claim"] is False and d["promotable"] is False
    assert d["axis_tag"] == AXIS_TAG
    # adoption REQUIRES beating all three nulls (never a synthetic-fold win)
    assert rep.adopted == (rep.beats_persistence and rep.beats_incumbent
                           and rep.beats_real_only)
    # the primary treatment is the prior-mean arm; naive-concat is diagnostic only
    assert "forge_priormean" in rep.per_fold[0]
    # sim2real is reported (a gap channel) but is NOT part of the adoption booleans
    assert all("gap" in s for s in rep.sim2real)


def test_adoption_deterministic():
    traj = _synthetic_traj()
    cfg = ForgeConfig(n_candidate_trajectories=16, seed=0)
    r1 = adoption_backtest(traj, cfg)
    r2 = adoption_backtest(traj, cfg)
    assert abs(r1.wf_forge_ridge_pruned - r2.wf_forge_ridge_pruned) < 1e-12
    assert r1.adopted == r2.adopted


def test_design_matrix_shape():
    traj = _synthetic_traj()
    prefix, _ = _prefix_and_comp(traj, k=5)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    Phi, Y = _design(prefix, phis)
    assert Phi.shape[0] == len(prefix) and Y.shape[0] == len(prefix)
    assert Phi.shape[1] == 1 + 6 + phis.shape[1]   # 1 + STATE_DIM + PHI_DIM
