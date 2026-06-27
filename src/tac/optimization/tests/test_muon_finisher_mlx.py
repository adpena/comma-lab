# SPDX-License-Identifier: MIT
"""Tests for the MLX Muon finisher glue (PR95 stage-8 partition).

NO-FAKE focus: prove Newton-Schulz ACTUALLY orthogonalizes the real param
matrices (singular spectrum collapses toward 1), prove the realized optimizer
step orthogonalizes a 2-D weight while routing biases/heads to AdamW, and prove
the default-off predicate is bit-identical (AdamW throughout).
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.muon_finisher_mlx import (
    MUON_EXCLUDE_TOKENS,
    build_muon_finisher_optimizer,
    count_muon_adamw_split,
    muon_active_for_epoch,
    muon_finisher_param_filter,
)

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")


@pytest.fixture(autouse=True)
def _cpu_device():
    """Pin MLX to CPU (the GPU is owned by a critical-path decoder; tests never touch it)."""
    mx.set_default_device(mx.cpu)


def _spread_matrix(rng, m, n, svals):
    """A (m, n) matrix (m>=n) with prescribed singular values (spread spectrum)."""
    base = rng.normal(size=(m, n)).astype(np.float32)
    u, _, vt = np.linalg.svd(base, full_matrices=False)
    return (u * np.asarray(svals, dtype=np.float32)) @ vt


# ---------------------------------------------------------------------------
# Pure routing predicate
# ---------------------------------------------------------------------------
def test_param_filter_routes_2d_hidden_weights_to_muon():
    w2d = mx.zeros((8, 5))
    assert muon_finisher_param_filter("in_proj.weight", w2d) is True
    assert muon_finisher_param_filter("film.weight", w2d) is True
    assert muon_finisher_param_filter("hidden.0.weight", w2d) is True
    assert muon_finisher_param_filter("hidden.3.weight", w2d) is True


def test_param_filter_excludes_heads_biases_embedding():
    w2d = mx.zeros((5, 8))
    b1d = mx.zeros((8,))
    # final heads (K-class SDF + RGB texture) -> AdamW even though 2-D .weight
    assert muon_finisher_param_filter("out_sdf.weight", w2d) is False
    assert muon_finisher_param_filter("out_tex.weight", w2d) is False
    # biases / 1-D -> AdamW
    assert muon_finisher_param_filter("in_proj.bias", b1d) is False
    assert muon_finisher_param_filter("hidden.0.bias", b1d) is False
    # per-pair latent (a bare 2-D leaf, no ".weight" suffix) -> AdamW
    assert muon_finisher_param_filter("code", mx.zeros((48, 32))) is False
    # future-proofing exclusion tokens
    assert "out_sdf" in MUON_EXCLUDE_TOKENS and "rgb" in MUON_EXCLUDE_TOKENS


def test_param_filter_rejects_non_2d():
    assert muon_finisher_param_filter("hidden.0.weight", mx.zeros((8,))) is False  # 1-D
    assert muon_finisher_param_filter("conv.weight", mx.zeros((4, 4, 3, 3))) is False  # 4-D (witness has none)


# ---------------------------------------------------------------------------
# Pure switch predicate — DEFAULT-OFF must be bit-identical (AdamW throughout)
# ---------------------------------------------------------------------------
def test_muon_active_default_none_is_always_off():
    for ep in (1, 50, 500, 10_000):
        assert muon_active_for_epoch(ep, None) is False


def test_muon_active_switches_at_start_epoch_and_stays_on():
    assert muon_active_for_epoch(799, 800) is False
    assert muon_active_for_epoch(800, 800) is True   # fires AT the boundary
    assert muon_active_for_epoch(801, 800) is True
    assert muon_active_for_epoch(5000, 800) is True  # resume past boundary re-enters


# ---------------------------------------------------------------------------
# NO-FAKE: Newton-Schulz REALLY orthogonalizes (not a no-op stub)
# ---------------------------------------------------------------------------
def test_newton_schulz_collapses_singular_spectrum_toward_one():
    """The NS used by the finisher (mlx.optimizers.Muon) collapses the singular spectrum.

    The 5-step quintic (coeffs 3.4445/-4.7750/2.0315) is an APPROXIMATE orthogonalizer:
    it drives every singular value into a tight band around 1 (~[0.68, 1.13]) rather
    than exactly 1 -- that band IS the orthogonalization. Proof = an 8x-spread input
    spectrum collapses to ratio < 2.0 (and the output is NOT a rescale of the input).
    """
    rng = np.random.default_rng(0)
    m = _spread_matrix(rng, 8, 5, [4.0, 3.0, 2.0, 1.0, 0.5])  # spread ratio 8x
    in_ratio = 4.0 / 0.5
    muon = optim.Muon(learning_rate=1.0, momentum=0.0, weight_decay=0.0, nesterov=False, ns_steps=5)
    out = np.asarray(muon._zeropower_via_newtonschulz5(mx.array(m), steps=5))
    sv = np.linalg.svd(out, compute_uv=False)
    # All singular values pulled into the orthogonal band around 1.
    assert np.all(sv > 0.6) and np.all(sv < 1.4), f"svals not ~1: {sv}"
    out_ratio = float(sv.max() / sv.min())
    # Spectrum spread collapses dramatically vs input -> proves orthogonalization happened.
    assert out_ratio < 2.0 < in_ratio, f"ratio in={in_ratio} out={out_ratio}"
    # Falsify the no-op stub: the orthogonalized matrix is NOT a rescale of the input.
    in_norm = m / (np.linalg.norm(m) + 1e-7)
    out_norm = out / (np.linalg.norm(out) + 1e-7)
    assert np.linalg.norm(out_norm - in_norm) > 0.1, "NS output == normalized input (no-op stub!)"


def test_newton_schulz_maps_arbitrary_spectra_into_the_same_orthogonal_band():
    """Universality: WILDLY different input spectra all land in the same ~1 band.

    This is the defining property of an orthogonalizer (it projects onto the orthogonal
    manifold), and falsifies any passthrough/scale stub (a stub would preserve each
    input's distinct spread).
    """
    rng = np.random.default_rng(1)
    muon = optim.Muon(learning_rate=1.0, momentum=0.0, weight_decay=0.0, nesterov=False, ns_steps=5)
    spectra = ([16.0, 1.0, 1.0, 1.0, 1.0], [3.0, 2.5, 2.0, 1.5, 0.4], [1.05, 1.0, 0.98, 0.97, 0.95])
    out_ratios = []
    for sv_in in spectra:
        m = mx.array(_spread_matrix(rng, 9, 5, sv_in))
        sv_out = np.linalg.svd(np.asarray(muon._zeropower_via_newtonschulz5(m, steps=5)), compute_uv=False)
        assert np.all(sv_out > 0.6) and np.all(sv_out < 1.4), f"in={sv_in} -> out svals {sv_out}"
        out_ratios.append(float(sv_out.max() / sv_out.min()))
    # The 16x-spread input collapses to roughly the SAME band as the near-flat input.
    assert max(out_ratios) < 2.0, f"outputs not all in the orthogonal band: {out_ratios}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def test_build_finisher_is_multioptimizer_with_muon_and_adamw():
    opt = build_muon_finisher_optimizer(
        muon_lr=2e-3, muon_adamw_lr=1e-4, muon_momentum=0.95,
        muon_weight_decay=1e-4, muon_ns_steps=5, adamw_weight_decay=1e-4,
    )
    assert isinstance(opt, optim.MultiOptimizer)
    kinds = {type(o).__name__ for o in opt.optimizers}
    assert "Muon" in kinds and "AdamW" in kinds


@pytest.mark.parametrize("bad", [
    dict(muon_lr=0.0, muon_adamw_lr=1e-4),
    dict(muon_lr=2e-3, muon_adamw_lr=-1.0),
    dict(muon_lr=2e-3, muon_adamw_lr=1e-4, muon_ns_steps=0),
    dict(muon_lr=2e-3, muon_adamw_lr=1e-4, muon_momentum=1.0),
])
def test_build_finisher_rejects_invalid_hparams(bad):
    with pytest.raises(ValueError):
        build_muon_finisher_optimizer(**bad)


# ---------------------------------------------------------------------------
# NO-FAKE strongest: the REALIZED optimizer step orthogonalizes a 2-D weight
# (singular values become ~equal) while the bias follows AdamW (NOT orthogonalized)
# ---------------------------------------------------------------------------
def test_realized_step_orthogonalizes_weight_and_routes_bias_to_adamw():
    rng = np.random.default_rng(7)
    w = mx.array(np.zeros((8, 5), dtype=np.float32))
    b = mx.array(np.zeros((8,), dtype=np.float32))
    # Use a dict leaf path with NO numeric key (mirrors in_proj/film; a numeric list
    # index like "hidden.0" makes MultiOptimizer's split/merge ambiguous).
    params = {"in_proj": {"weight": w, "bias": b}}
    # gradient on the weight has a DELIBERATELY spread singular spectrum (ratio 8x).
    gw = _spread_matrix(rng, 8, 5, [4.0, 3.0, 2.0, 1.0, 0.5])
    gb = rng.normal(size=(8,)).astype(np.float32)
    grads = {"in_proj": {"weight": mx.array(gw), "bias": mx.array(gb)}}

    opt = build_muon_finisher_optimizer(
        muon_lr=1.0, muon_adamw_lr=1e-2, muon_momentum=0.0,
        muon_weight_decay=0.0, muon_ns_steps=5, adamw_weight_decay=0.0, nesterov=False,
    )
    opt.init(params)
    new = opt.apply_gradients(grads, params)
    mx.eval(new)

    # weight delta = -lr_eff * NS(grad); NS(grad) is orthogonalized so delta has a
    # near-flat singular spectrum (ratio < 2) EVEN THOUGH the input gradient ratio is 8x.
    dw = np.asarray(new["in_proj"]["weight"]) - np.asarray(w)
    assert np.linalg.norm(dw) > 1e-6, "weight did not update (Muon group inactive!)"
    sv_dw = np.linalg.svd(dw, compute_uv=False)
    grad_sv = np.linalg.svd(gw, compute_uv=False)
    grad_ratio = float(grad_sv.max() / grad_sv.min())
    dw_ratio = float(sv_dw.max() / sv_dw.min())
    assert grad_ratio > 5.0, f"test gradient not spread enough: {grad_ratio}"
    assert dw_ratio < 2.0, f"weight update NOT orthogonalized (ratio {dw_ratio}); Muon is a no-op stub!"

    # bias delta follows AdamW (1-D, never orthogonalized) -> just prove it moved.
    db = np.asarray(new["in_proj"]["bias"]) - np.asarray(b)
    assert np.linalg.norm(db) > 1e-6, "bias did not update (AdamW fallback inactive!)"


# ---------------------------------------------------------------------------
# Observability split count (matches the witness param tree)
# ---------------------------------------------------------------------------
def test_count_split_matches_witness_partition():
    params = {
        "in_proj": {"weight": mx.zeros((96, 40)), "bias": mx.zeros((96,))},
        "film": {"weight": mx.zeros((768, 32)), "bias": mx.zeros((768,))},
        "hidden": [
            {"weight": mx.zeros((96, 96)), "bias": mx.zeros((96,))},
            {"weight": mx.zeros((96, 96)), "bias": mx.zeros((96,))},
        ],
        "out_sdf": {"weight": mx.zeros((5, 96)), "bias": mx.zeros((5,))},
        "out_tex": {"weight": mx.zeros((3, 96)), "bias": mx.zeros((3,))},
        "code": mx.zeros((48, 32)),
    }
    n_muon, n_adamw = count_muon_adamw_split(params)
    # Muon: in_proj.weight, film.weight, hidden.0.weight, hidden.1.weight = 4
    assert n_muon == 4
    # AdamW: 6 biases + out_sdf.weight + out_tex.weight + code = 9
    assert n_adamw == 9
