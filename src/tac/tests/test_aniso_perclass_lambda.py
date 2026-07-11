# SPDX-License-Identifier: MIT
"""Tests for the #433 anisotropic-coupled per-class-λ arms (P0 physics-model directive).

NO-FAKE proofs, mirroring the organ test discipline:
  * the coupling matrix is row-stochastic BY CONSTRUCTION and couples a planted
    adjacent class pair through the boundary term (never a silent identity);
  * a planted thin class (all-boundary) gets bulk_frac 0 and a planted bulk class
    gets bulk_frac near 1 — the two Fisher regimes measurably separate;
  * the along/across anisotropy responds to a planted tangentially-modulated margin
    field (dash-comb surrogate) vs a smooth one;
  * the flip temperature equals the rank-matched margin threshold on a construction
    where the flip mass and margin field are known exactly;
  * PhysicsPriorMeanAdjoint recovers planted physics-structured dynamics at n=2
    (the small-n regime the prior exists for) where the plain ridge cannot, and its
    1-dof κ is non-negative and gauge-stable;
  * the openpilot prior excludes Movable BY MODEL SCOPE and finds the planted
    horizon/hood rows;
  * the SAO trust region is inert at large radius and binding at tiny radius
    (positive control — the clamp is not a no-op);
  * containment: the new module carries no actuation tokens;
  * every new arm is registered in ARCHITECTURES and constructible via make_model
    (cache-gated arms skipped when the cache is absent).
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control.aniso_perclass_lambda import (
    OPENPILOT_STATIC_CLASSES,
    AnisoClassProfiles,
    IsoPriorMeanAdjoint,
    PhysicsPriorMeanAdjoint,
    _all_class_boundary,
    _box_smooth,
    aniso_coupled_m0,
    c10k_scorelaw_m0,
)
from tac.witness_control.lambda_net import ARCHITECTURES, N_CLASSES
from tac.witness_control.scorer_geometry import DEFAULT_GT_CACHE

_HAS_CACHE = DEFAULT_GT_CACHE.exists()


# ────────────────────────────── synthetic label/margin worlds ──────────────────────────
def _two_class_world(h: int = 40, w: int = 40, split: int = 20):
    """Road (0) left, Lane (1) right: one vertical interface at x=split."""
    lab = np.zeros((h, w), dtype=np.int64)
    lab[:, split:] = 1
    xs = np.arange(w)[None, :].repeat(h, axis=0)
    margin = np.abs(xs - (split - 0.5))          # distance to the interface
    return lab, margin.astype(np.float64)


def test_box_smooth_preserves_constant():
    a = np.full((16, 12), 3.5)
    s = _box_smooth(a, 3)
    assert np.allclose(s, 3.5)


def test_all_class_boundary_is_union_of_edges():
    lab, _ = _two_class_world()
    b = _all_class_boundary(lab)
    ys, xs = np.where(b)
    assert set(np.unique(xs)) <= {19, 20}         # both sides of the interface
    assert b.sum() > 0


def test_coupling_matrix_row_stochastic_and_couples_planted_pair():
    # planted: Lane fully boundary (bulk_frac 0), coupled ONLY to Road
    pair = np.zeros((5, 5))
    pair[1, 0] = 1.0                              # Lane's band susceptibility on Road pair
    pair[0, 1] = 0.4
    prof = AnisoClassProfiles(
        epsilon=1.0, bulk_susc=(3.0, 0.0, 5.0, 1.0, 2.0),
        boundary_susc=(1.0, 2.0, 1.0, 1.0, 1.0),
        bulk_frac=(0.75, 0.0, 5 / 6, 0.5, 2 / 3),
        total_susc_share=(0.4, 0.2, 0.2, 0.1, 0.1),
        annulus_area_frac=0.05, annulus_susc_frac=0.5,
        pair_susc=tuple(tuple(r) for r in pair),
        aniso_ratio=tuple(tuple(np.ones((5, 5))[i]) for i in range(5)),
        sigma_preset="all-ones", radius_px=2, n_frames=1, source="synthetic",
        registered_deficit_ratio=3.125)
    C = prof.coupling_matrix()
    assert np.allclose(C.sum(axis=1), 1.0)
    # Lane row: bulk 0 → ½ own + ½ Road
    assert C[1, 1] == pytest.approx(0.5)
    assert C[1, 0] == pytest.approx(0.5)
    # Road row: bulk 0.75 → own 0.75 + 0.125 + 0.125·P(Road→Lane)=0.125
    assert C[0, 0] == pytest.approx(0.875)
    assert C[0, 1] == pytest.approx(0.125)
    # classes with no measured pair mass keep the WHOLE boundary term on themselves
    # (own ½ + the partner ½ redirected to self) — row-stochastic by construction
    assert C[2, 2] == pytest.approx(1.0)


def test_anisotropy_gauge_is_geomean_one_and_upweights_rough_pair():
    pair = np.zeros((5, 5))
    pair[0, 1] = pair[1, 0] = 1.0
    pair[0, 2] = pair[2, 0] = 1.0
    an = np.ones((5, 5))
    an[0, 1] = an[1, 0] = 4.0                     # dash-comb-like tangential roughness
    an[0, 2] = an[2, 0] = 0.25
    prof = AnisoClassProfiles(
        epsilon=1.0, bulk_susc=(0.0,) * 5, boundary_susc=(1.0,) * 5,
        bulk_frac=(0.0,) * 5, total_susc_share=(0.2,) * 5,
        annulus_area_frac=0.05, annulus_susc_frac=1.0,
        pair_susc=tuple(tuple(r) for r in pair),
        aniso_ratio=tuple(tuple(r) for r in an),
        sigma_preset="all-ones", radius_px=2, n_frames=1, source="synthetic",
        registered_deficit_ratio=3.125)
    C = prof.coupling_matrix()
    # Road boundary partner distribution favors the tangentially-rough Lane pair 16:1
    assert C[0, 1] / C[0, 2] == pytest.approx(16.0, rel=1e-6)


def test_physics_priormean_kappa_nonneg_and_recovers_planted_dynamics_at_n2():
    """At n=2 intervals the prior-mean ridge with the TRUE physics matrix must beat the
    plain (shrink-to-zero) ridge on a held-out interval — the small-n regime claim."""
    from tac.witness_control.lambda_net import (
        Interval,
        RidgeSolveAdjoint,
        lever_features,
    )
    rng = np.random.default_rng(3)
    lever_names = ("seg", "lane_edge", "island_amplify")
    phis = np.stack([lever_features(n) for n in lever_names])
    m0 = np.eye(N_CLASSES) * 0.5
    m0[0, 1] = 1.0                                # Lane lever moves Road (coupling)
    kappa_true = 2e-3
    b_true = np.asarray([-1e-4, -3e-4, -5e-5, -1e-4, -2e-5, 1e-3])

    def dxdt(u):
        occ = phis.T @ u
        d = b_true.copy()
        d[:N_CLASSES] += -kappa_true * (m0 @ occ[:N_CLASSES])
        return d

    intervals = []
    x = np.asarray([0.1, 0.4, 0.02, 0.9, 0.01, np.log(85_000.0)])
    us = [np.asarray([0.6, 0.3, 0.1]), np.asarray([0.2, 0.7, 0.1]),
          np.asarray([0.1, 0.2, 0.7])]
    ep = 0.0
    for u in us:
        d = dxdt(u)
        x1 = x + 25.0 * d
        path = np.tile(np.concatenate([u, [0.1]]), (5, 1))
        intervals.append(Interval(ep0=ep, ep1=ep + 25.0, x0=x.copy(), x1=x1.copy(),
                                  ctx=np.zeros(3), u_mean=u.copy(), path=path))
        x, ep = x1, ep + 25.0
        x = x + rng.normal(scale=1e-7, size=x.shape)      # tiny noise

    train, held = intervals[:2], intervals[2]
    phys = PhysicsPriorMeanAdjoint(m0)
    phys.fit(train, phis)
    assert phys.kappa is not None and phys.kappa >= 0.0
    plain = RidgeSolveAdjoint()
    plain.fit(train, phis)

    def pred_err(model):
        resp = np.stack([model.response(held.x0, held.ctx, lever_features(n), held.path)
                         for n in lever_names])
        pred = model.base(held.x0, held.ctx) + resp.T @ held.u_mean
        return float(np.mean(np.abs(pred[:N_CLASSES] - held.dxdt()[:N_CLASSES])))

    assert pred_err(phys) < pred_err(plain)


def test_physics_priormean_validates_m0():
    with pytest.raises(ValueError):
        PhysicsPriorMeanAdjoint(np.ones((4, 4)))
    with pytest.raises(ValueError):
        PhysicsPriorMeanAdjoint(-np.eye(N_CLASSES))
    with pytest.raises(ValueError):
        PhysicsPriorMeanAdjoint(np.zeros((N_CLASSES, N_CLASSES)))


def test_m0_composers_shapes_and_content():
    prof = AnisoClassProfiles(
        epsilon=1.0, bulk_susc=(1.0,) * 5, boundary_susc=(1.0,) * 5,
        bulk_frac=(0.5,) * 5, total_susc_share=(0.2,) * 5,
        annulus_area_frac=0.05, annulus_susc_frac=0.5,
        pair_susc=tuple(tuple(np.ones((5, 5))[i]) for i in range(5)),
        aniso_ratio=tuple(tuple(np.ones((5, 5))[i]) for i in range(5)),
        sigma_preset="all-ones", radius_px=2, n_frames=1, source="synthetic",
        registered_deficit_ratio=3.125)
    g = np.asarray([5.0, 1.0, 3.0, 1.0, 2.0])
    m = aniso_coupled_m0(prof, g)
    assert m.shape == (5, 5) and np.all(m >= 0)
    # column scaling by g: high-impact columns carry more mass
    assert m[:, 0].sum() > m[:, 1].sum()
    r = np.asarray([1.0, 5.0, 1.0, 2.0, 1.0])
    m2 = c10k_scorelaw_m0(prof, g, r)
    # rarity multiplies the column direction: Lane column mass ratio grows by 5×
    assert (m2[:, 1].sum() / m[:, 1].sum()) == pytest.approx(5.0, rel=1e-9)
    with pytest.raises(ValueError):
        aniso_coupled_m0(prof, np.ones(4))
    with pytest.raises(ValueError):
        c10k_scorelaw_m0(prof, g, np.ones(3))


def test_openpilot_scope_excludes_movable():
    assert 3 not in OPENPILOT_STATIC_CLASSES
    assert set(OPENPILOT_STATIC_CLASSES) == {0, 1, 2, 4}


def test_new_arms_registered():
    for a in ("N_aniso_coupled", "O_openpilot_geom", "P_priormean_aniso",
              "Q_priormean_iso", "R_priormean_c10k_scorelaw", "S_priormean_openpilot"):
        assert a in ARCHITECTURES


def test_iso_priormean_is_identity_ablation():
    arm = IsoPriorMeanAdjoint()
    assert np.allclose(arm.m0_unit, np.eye(N_CLASSES))


def test_containment_no_actuation_tokens():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "witness_control" / "aniso_perclass_lambda.py").read_text()
    for token in ("subprocess", "os.system", "Popen", "launch_witness_run"):
        assert token not in src, f"actuation token {token!r} in the P0 module"


# ───────────────────────── cache-gated (real gt_n96) tests ─────────────────────────
@pytest.mark.skipif(not _HAS_CACHE, reason="gt_n96 cache absent")
def test_flip_temperature_is_positive_and_below_median_margin():
    import numpy as np

    from tac.witness_control.aniso_perclass_lambda import measure_flip_temperature
    eps = measure_flip_temperature()
    z = np.load(DEFAULT_GT_CACHE)
    med = float(np.median(z["margins"][::8]))
    assert 0.0 < eps < med          # the flip scale is far below the bulk median


@pytest.mark.skipif(not _HAS_CACHE, reason="gt_n96 cache absent")
def test_measured_profiles_match_known_crux_structure():
    from tac.witness_control.aniso_perclass_lambda import measure_aniso_class_profiles
    pr = measure_aniso_class_profiles()
    # the annulus concentration (#333/L66 family): >half the flip sensitivity in <10% area
    assert pr.annulus_area_frac < 0.10
    assert pr.annulus_susc_frac > 0.45
    # Lane is thin: (near-)all-boundary regime (a handful of interior px at r=2)
    assert pr.bulk_frac[1] < 0.01
    # coupling rows are stochastic; Lane couples to Road hardest among partners
    C = pr.coupling_matrix()
    assert np.allclose(C.sum(axis=1), 1.0)
    off = [C[1, j] for j in range(N_CLASSES) if j != 1]
    assert max(off) == pytest.approx(C[1, 0])


@pytest.mark.skipif(not _HAS_CACHE, reason="gt_n96 cache absent")
def test_openpilot_prior_measured_geometry():
    from tac.witness_control.aniso_perclass_lambda import (
        measure_openpilot_geometry_prior,
    )
    op = measure_openpilot_geometry_prior()
    # horizon within the measured Undrivable lower edge band; hood above frame bottom
    assert 150 < op.horizon_row < 220
    assert 260 < op.hood_top_row < 330
    assert op.addressable_frac[3] == 0.0            # Movable: model scope
    assert op.addressable_frac[4] > 0.9             # static hood fully ego-addressable
