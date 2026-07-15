"""Tests for the score-neutral mod-dim dynamics telemetry (src/tac/boundary_math/mod_dim_dynamics.py).

Coverage: spectrum math on KNOWN matrices (rank/k90/entropy) · effective-rank cross-check against the
canonical PR helper · k-energy correctness · score-neutrality (no input mutation, no RNG-stream
consumption) · ξ-CCA math (identical/orthogonal blocks) · per-dim FiLM consumption · per-dim ξ-r² ·
truncate-bytes estimate · bit-allocation waterfill · ablation loop (mock render_fn) · row assembly /
default-ON emission shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math import mod_dim_dynamics as mdd
from tac.boundary_math.lever_b_levelset_generator import code_spectrum_participation_ratio


# ── spectrum math on known matrices ────────────────────────────────────────────────────────────────
def test_rank1_spectrum_collapse():
    # all rows a scalar multiple of one direction after centering => rank-1 => eff_rank ~ 1, k90 == 1.
    base = np.array([1.0, 2.0, -1.0, 0.5])
    code = np.outer(np.array([-2.0, -1.0, 1.0, 2.0, 0.0]), base)  # (5, 4), centered rank-1
    sv = mdd.centered_singular_values(code)
    assert mdd.effective_rank(sv) == pytest.approx(1.0, abs=1e-6)
    assert mdd.k_energy_cutoff(sv, 0.90) == 1
    assert mdd.spectral_entropy(sv)["normalized"] == pytest.approx(0.0, abs=1e-9)


def test_uniform_isotropic_spectrum():
    # orthonormal columns scaled equally => flat spectrum => eff_rank == D, entropy_norm == 1, k90 near D.
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((200, 6)))
    code = Q * 3.0  # 6 equal-energy orthogonal directions
    sv = mdd.centered_singular_values(code)
    assert mdd.effective_rank(sv) == pytest.approx(6.0, rel=0.02)
    assert mdd.spectral_entropy(sv)["normalized"] == pytest.approx(1.0, rel=0.02)
    assert mdd.k_energy_cutoff(sv, 0.90) in (5, 6)
    assert mdd.k_energy_cutoff(sv, 0.99) == 6


def test_effective_rank_matches_canonical_pr_helper():
    # eff_rank(svals) must equal code_spectrum_participation_ratio(code) (algebraic identity).
    rng = np.random.default_rng(7)
    code = rng.standard_normal((120, 10)) @ np.diag([5, 4, 3, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01])
    sv = mdd.centered_singular_values(code)
    assert mdd.effective_rank(sv) == pytest.approx(code_spectrum_participation_ratio(code), rel=1e-9)


def test_k_energy_monotone_and_bounds():
    rng = np.random.default_rng(3)
    sv = mdd.centered_singular_values(rng.standard_normal((80, 16)))
    k90, k99 = mdd.k_energy_cutoff(sv, 0.90), mdd.k_energy_cutoff(sv, 0.99)
    assert 1 <= k90 <= k99 <= 16
    assert len(mdd.top_k_energy_fracs(sv, 8)) == 8
    assert mdd.top_k_energy_fracs(sv, 8)[0] >= mdd.top_k_energy_fracs(sv, 8)[1]


def test_zero_spectrum_safe():
    code = np.zeros((10, 4))
    sv = mdd.centered_singular_values(code)
    assert mdd.k_energy_cutoff(sv, 0.90) == 0
    assert mdd.spectral_entropy(sv) == {"nats": 0.0, "normalized": 0.0}
    assert mdd.top_k_energy_fracs(sv, 4) == []


# ── score-neutrality (the byte-identity guarantee) ─────────────────────────────────────────────────
def test_row_build_does_not_mutate_inputs_or_consume_rng():
    rng = np.random.default_rng(1)
    code = rng.standard_normal((40, 8))  # (2P=40 => P=20, D=8)
    poses = rng.standard_normal((20, 6))
    film = rng.standard_normal((32, 8))
    code0, poses0, film0 = code.copy(), poses.copy(), film.copy()
    np.random.seed(12345)
    state_before = np.random.get_state()
    row = mdd.mod_dim_dynamics_row(
        code, poses, epoch=5, seg_form="stageTau", tau=1.5, mod_dim=8,
        code_bytes_full=4096, film_weights=film)
    state_after = np.random.get_state()
    # inputs untouched (read-only)
    assert np.array_equal(code, code0) and np.array_equal(poses, poses0) and np.array_equal(film, film0)
    # global legacy RNG stream untouched (no np.random.* draw anywhere in the path)
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1]) and state_before[2] == state_after[2]
    assert row["stage"] == "mod_dim_dynamics" and row["mod_dim"] == 8


def test_ablation_copies_and_does_not_mutate():
    code = np.ones((6, 3)) * 2.0
    code0 = code.copy()
    seen = {}

    def render_fn(c):
        # the callable must receive a COPY with column zeroed; the original stays intact.
        seen["zeroed_cols"] = [j for j in range(c.shape[1]) if np.allclose(c[:, j], 0.0)]
        return float(seen["zeroed_cols"][0]) * 0.01  # deterministic mock d_seg
    deltas = mdd.per_dim_dseg_ablation(code, [0, 2], render_fn, baseline_dseg=0.0)
    assert np.array_equal(code, code0)              # original never mutated
    assert deltas == pytest.approx([0.0, 0.02])     # dim0 -> 0.0, dim2 -> 0.02


def test_ablation_failure_records_nan_not_abort():
    code = np.ones((4, 3))

    def bad_fn(_c):
        raise RuntimeError("scorer hiccup")
    out = mdd.per_dim_dseg_ablation(code, [0, 1], bad_fn, baseline_dseg=0.1)
    assert len(out) == 2 and all(np.isnan(x) for x in out)


# ── ξ-CCA math ─────────────────────────────────────────────────────────────────────────────────────
def test_cca_identical_blocks_corr_one():
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((100, 6))
    # X spans the same subspace (a rotation of Y) => leading canonical correlations == 1.
    R, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    cca = mdd.latent_pose_cca(Y @ R, Y)
    assert cca[0] == pytest.approx(1.0, abs=1e-6)
    assert cca.min() == pytest.approx(1.0, abs=1e-6)


def test_cca_independent_blocks_corr_small():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((500, 8))
    Y = rng.standard_normal((500, 6))  # independent => canonical corrs modest
    cca = mdd.latent_pose_cca(X, Y)
    assert cca.shape == (6,)
    assert cca.max() < 0.35  # sampling noise only; well below a real redundancy


def test_cca_too_few_pairs_safe():
    assert np.array_equal(mdd.latent_pose_cca(np.zeros((1, 4)), np.zeros((1, 6))), np.zeros(4))


# ── per-dim primitives ──────────────────────────────────────────────────────────────────────────────
def test_per_dim_variance_and_film_consumption():
    code = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 5.0], [0.0, 1.0, -5.0], [0.0, -1.0, 0.0]])
    var = mdd.per_dim_variance(code)
    assert var[0] == pytest.approx(0.0)          # dim 0 constant
    assert var[2] > var[1] > 0.0                 # dim 2 varies most
    # FiLM consumption: column norms of an (out, D) weight; dim with larger column norm reads more.
    W = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # col0 norm 5, col1 norm 1, col2 0
    cons = mdd.per_dim_film_consumption(W, mod_dim=3)
    assert cons == pytest.approx([5.0, 1.0, 0.0])


def test_film_consumption_quadrature_and_width_guard():
    W1 = np.array([[3.0, 0.0]])           # (1, 2) col norms [3, 0]
    W2 = np.array([[4.0, 0.0]])           # (1, 2) col norms [4, 0]
    W_wrong = np.array([[9.0, 9.0, 9.0]])  # width 3 != mod_dim 2 => ignored
    cons = mdd.per_dim_film_consumption([W1, W2, W_wrong], mod_dim=2)
    assert cons == pytest.approx([5.0, 0.0])  # sqrt(3^2 + 4^2) = 5


def test_per_dim_pose_r2_names_redundant_dim():
    rng = np.random.default_rng(5)
    xi = rng.standard_normal((300, 6))
    # dim 0 == twist comp 2 exactly (r2 -> 1); dim 1 pure noise (r2 -> ~0).
    code = np.column_stack([xi[:, 2], rng.standard_normal(300)])
    r2 = mdd.per_dim_pose_r2(code, xi)
    assert r2.shape == (2, 6)
    assert r2[0, 2] == pytest.approx(1.0, abs=1e-6)
    assert r2[0].max() == pytest.approx(1.0, abs=1e-6)
    assert r2[1].max() < 0.1


def test_code_to_per_pair_averages_frames():
    code_table = np.array([[0.0, 10.0], [2.0, 20.0],   # pair 0 -> mean [1, 15]
                           [4.0, 40.0], [6.0, 60.0]])  # pair 1 -> mean [5, 50]
    pp = mdd.code_to_per_pair(code_table)
    assert pp == pytest.approx(np.array([[1.0, 15.0], [5.0, 50.0]]))


# ── exploitation hooks ────────────────────────────────────────────────────────────────────────────
def test_truncate_bytes_estimate():
    assert mdd.truncate_bytes_estimate(1000, 20, 32) == 625
    assert mdd.truncate_bytes_estimate(1000, 32, 32) == 1000  # full
    assert mdd.truncate_bytes_estimate(1000, 40, 32) == 1000  # clamped to mod_dim
    assert mdd.truncate_bytes_estimate(1000, 5, 0) == 1000     # mod_dim<=0 fail-safe


def test_bit_allocation_hint_waterfill():
    util = np.array([1.0, 1.0, 0.0])
    sens = np.array([2.0, 0.0, 5.0])  # dim0: 1*2=2 ; dim1: 0 ; dim2: 0*5=0
    hint = mdd.per_dim_bit_allocation_hint(util, sens)
    assert hint == pytest.approx([1.0, 0.0, 0.0])  # only dim0 has util AND sensitivity
    # degenerate -> uniform
    assert mdd.per_dim_bit_allocation_hint(np.zeros(3), np.zeros(3)) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# ── row assembly / default-ON emission shape ─────────────────────────────────────────────────────────
def test_dynamics_row_shape_and_json_safe():
    import json
    rng = np.random.default_rng(2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # spurious macOS-BLAS FPE flag
        code = rng.standard_normal((1200, 32)) @ np.diag(np.linspace(5, 0.1, 32))  # (2P=1200, D=32)
    poses = rng.standard_normal((600, 6))
    film = rng.standard_normal((256, 32))
    row = mdd.mod_dim_dynamics_row(
        code, poses, epoch=225, seg_form="stageTau", tau=0.8, mod_dim=32,
        code_bytes_full=8000, film_weights=film, top_k=8)
    s = json.dumps(row)  # must be JSON-serializable (no numpy scalars leak)
    assert '"stage": "mod_dim_dynamics"' in s
    assert 1.0 <= row["spectrum"]["effective_rank"] <= 32.0
    assert row["spectrum"]["k90"] <= row["spectrum"]["k99"] <= 32
    assert len(row["per_dim"]["variance"]) == 32
    assert len(row["per_dim"]["film_consumption"]) == 32
    assert len(row["per_dim"]["xi_max_r2"]) == 32
    assert len(row["latent_xi_cca"]["canonical_corrs"]) == 6
    assert row["k90_truncate_bytes_estimate"] <= row["code_bytes_full"]
    assert "NON-PROMOTABLE" in row["axis"]


def test_dynamics_row_film_optional():
    code = np.random.default_rng(4).standard_normal((20, 8))
    poses = np.random.default_rng(4).standard_normal((10, 6))
    row = mdd.mod_dim_dynamics_row(
        code, poses, epoch=1, seg_form=None, tau=None, mod_dim=8,
        code_bytes_full=100, film_weights=None)
    assert row["per_dim"]["film_consumption"] is None
    assert row["seg_form"] is None and row["tau"] is None


def test_ablation_row_shape():
    row = mdd.mod_dim_ablation_row(
        [0.01, float("nan"), 0.0], [0, 1, 2], [1.0, 0.5, 0.2],
        epoch=700, seg_form="stageMuon", k_sample=32)
    assert row["stage"] == "mod_dim_ablation"
    assert row["delta_d_seg"] == [0.01, None, 0.0]  # nan -> None (JSON-safe)
    assert row["bit_allocation_hint"] is not None and len(row["bit_allocation_hint"]) == 3
    assert row["k_sample"] == 32


def test_ablation_parallel_values_identical_to_sequential():
    """(#509) workers>=2 fans the per-dim calls across a thread pool with ordered
    aggregation — the returned deltas must equal the sequential ones exactly (the
    bit-identity claim), including per-dim nan semantics."""
    rng = np.random.default_rng(7)
    code = rng.standard_normal((12, 6))

    def render_fn(cc):
        if float(np.abs(cc[:, 3]).sum()) == 0.0:  # dim-3 arm deliberately fails
            raise RuntimeError("boom")
        return float(np.abs(cc).mean())

    base = float(np.abs(code).mean())
    dims = list(range(6))
    seq = mdd.per_dim_dseg_ablation(code, dims, render_fn, baseline_dseg=base)
    par = mdd.per_dim_dseg_ablation(code, dims, render_fn, baseline_dseg=base, workers=4)
    assert len(seq) == len(par) == 6
    for a, b in zip(seq, par):
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert float(a) == float(b)  # float-exact, not approx
    assert np.isnan(par[3])  # failure semantics preserved under the pool
