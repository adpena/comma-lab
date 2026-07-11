# SPDX-License-Identifier: MIT
"""Tests for the closed-form GP costate posterior arm (Warnick 2025 §3.3–3.4).

Protects the DERIVATION: the differentiated-RBF derivative posterior mean must equal a
finite-difference of the (independently computed) GP function posterior mean, the
posterior variance must be non-negative, the arm must be deterministic, and it must
plug into the organ backtest harness. Every number [macOS advisory] NON-PROMOTABLE.
"""
from __future__ import annotations

import numpy as np

from tac.witness_control.lambda_net import (
    ARCHITECTURES,
    STATE_DIM,
    GPCostatePosteriorAdjoint,
    backtest,
    build_intervals,
    make_model,
)


def _synthetic_traj(n: int = 8, seed: int = 0):
    """A small deterministic synthetic trajectory with the fields the arm needs."""
    rng = np.random.default_rng(seed)
    verdicts, loss_terms = [], []
    for i in range(n):
        ep = 50.0 + 25.0 * i
        # smooth decaying per-class d_seg + tiny noise (plateau-ish)
        base = np.array([0.006, 0.02, 0.001, 0.03, 5e-4]) * np.exp(-0.02 * i)
        d_by_class = (base + rng.normal(0, 1e-4, 5)).clip(1e-6).tolist()
        d_seg = float(np.array([0.23, 0.006, 0.5, 0.012, 0.26]) @ np.array(d_by_class))
        verdicts.append({"stage": "verdict", "epoch": ep, "d_seg": d_seg,
                         "d_seg_by_class": d_by_class, "blob_bytes": 85000.0,
                         "ep_loss": 1.0})
        for j in range(3):
            loss_terms.append({"stage": "loss_terms", "ep": ep + j,
                               "terms": {"seg": 1.0, "eikonal": 0.1}, "gnorm": 1.0})
    from tac.witness_control.lambda_net import CampaignTrajectory
    names = ("eikonal", "seg")
    return CampaignTrajectory(run_dir="synthetic", verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms), lever_names=names)


def test_arm_registered_and_constructible():
    assert "T_gp_costate_posterior" in ARCHITECTURES
    m = make_model("T_gp_costate_posterior")
    assert isinstance(m, GPCostatePosteriorAdjoint)


def test_fit_base_response_shapes_and_finite():
    traj = _synthetic_traj()
    ivs = build_intervals(traj)
    phis = np.stack([np.ones(13) for _ in traj.lever_names])
    m = make_model("T_gp_costate_posterior")
    m.fit(ivs, phis)
    b = m.base(ivs[-1].x0, ivs[-1].ctx, ivs[-1].path)
    assert b.shape == (STATE_DIM,)
    assert np.all(np.isfinite(b))
    # forecast-only arm: response is the honest zero field (no lever decomposition)
    r = m.response(ivs[-1].x0, ivs[-1].ctx, phis[0], ivs[-1].path)
    assert r.shape == (STATE_DIM,)
    assert np.allclose(r, 0.0)


def test_derivative_kernel_matches_finite_difference():
    """The differentiated-kernel closed form == finite-diff of the GP function posterior.

    This is the correctness proof of the Warnick §3.3 derivative blocks: the Bayes action
    Q̂ = (d/dep) μ*(ep) computed via the differentiated cross-covariance must equal the
    numerical derivative of the standard GP posterior mean.
    """
    traj = _synthetic_traj()
    ivs = build_intervals(traj)
    phis = np.stack([np.ones(13) for _ in traj.lever_names])
    m = make_model("T_gp_costate_posterior")
    m.fit(ivs, phis)
    u = float(m._eps[-2])  # interior-ish query for a clean central difference
    for c in range(STATE_DIM):
        ell, sf2, _ = m._hyp[c]
        ep = m._eps
        z = m._z[:, c] - m._zbar[c]
        Kinv = m._Kinv[c]

        def fpost(us, ell=ell, sf2=sf2, ep=ep, z=z, Kinv=Kinv):
            ks = sf2 * np.exp(-((us - ep) ** 2) / (2.0 * ell ** 2))
            return float(ks @ (Kinv @ z))

        h = 1e-3
        fd = (fpost(u + h) - fpost(u - h)) / (2.0 * h)
        cf, var = m._deriv_posterior(u, c)
        assert var >= 0.0  # closed-form posterior variance is non-negative
        assert abs(cf - fd) <= 1e-5 + 1e-4 * abs(fd)


def test_deterministic():
    traj = _synthetic_traj()
    ivs = build_intervals(traj)
    phis = np.stack([np.ones(13) for _ in traj.lever_names])
    a = make_model("T_gp_costate_posterior"); a.fit(ivs, phis)
    b = make_model("T_gp_costate_posterior"); b.fit(ivs, phis)
    va = a.base(ivs[-1].x0, ivs[-1].ctx, ivs[-1].path)
    vb = b.base(ivs[-1].x0, ivs[-1].ctx, ivs[-1].path)
    assert np.array_equal(va, vb)


def test_plugs_into_backtest_harness():
    """The arm runs through the organ backtest gate and yields a walk-forward number."""
    traj = _synthetic_traj(n=8)
    report, field = backtest(traj, architecture="T_gp_costate_posterior")
    assert report.architecture == "T_gp_costate_posterior"
    assert report.n_intervals >= 3
    assert np.isfinite(report.forecast_mae_model)
    # walk-forward computable at n>=4 intervals; field is stamped
    assert np.isfinite(report.walkforward_mae_model)
    assert field.status in ("BACKTESTED-PASS", "BACKTESTED-FAIL")
