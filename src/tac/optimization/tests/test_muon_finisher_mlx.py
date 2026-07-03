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
    _adamw_bias_correction_for,
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
    dict(muon_lr=2e-3, muon_adamw_lr=1e-4, muon_lr_final_frac=0.0),   # GAP 1: must be > 0
    dict(muon_lr=2e-3, muon_adamw_lr=1e-4, muon_lr_final_frac=1.5),   # GAP 1: must be <= 1
    dict(muon_lr=2e-3, muon_adamw_lr=1e-4, muon_anneal_steps=-1),     # GAP 1: must be >= 0
])
def test_build_finisher_rejects_invalid_hparams(bad):
    with pytest.raises(ValueError):
        build_muon_finisher_optimizer(**bad)


# ---------------------------------------------------------------------------
# GAP 1 (default-off): --muon-lr-final-frac cosine-decays the Muon-GROUP LR.
# Default (final_frac=1.0 OR anneal_steps<=0) => the Muon child keeps the LITERAL
# pre-change SCALAR construction => byte-identical.
# ---------------------------------------------------------------------------
def _muon_has_lr_schedule(opt):
    return "learning_rate" in getattr(opt.optimizers[0], "_schedulers", {})


@pytest.mark.parametrize("kw", [
    {},                                                       # pure default
    dict(muon_lr_final_frac=1.0, muon_anneal_steps=500),      # frac>=1 dominates
    dict(muon_lr_final_frac=0.1, muon_anneal_steps=0),        # anneal<=0 guard
])
def test_muon_lr_final_frac_default_off_is_scalar_byte_identical(kw):
    """Every 'off' combination builds the Muon child with the IDENTICAL scalar LR the
    pre-change code used (``optim.Muon(learning_rate=float(muon_lr))``) -> no schedule."""
    mlr = 3e-3
    pre_change_scalar = float(optim.Muon(learning_rate=float(mlr)).learning_rate)
    opt = build_muon_finisher_optimizer(muon_lr=mlr, muon_adamw_lr=1e-4, **kw)
    muon = opt.optimizers[0]
    assert not _muon_has_lr_schedule(opt), f"{kw} unexpectedly attached a schedule (not byte-identical)"
    assert float(muon.learning_rate) == pre_change_scalar, f"{kw} scalar drifted from pre-change build"


def test_muon_lr_final_frac_builds_cosine_decay_over_the_span():
    """final_frac<1.0 + anneal_steps>0 => the Muon GROUP LR cosine-decays from muon_lr
    (step 0) to muon_lr*final_frac (step==anneal_steps), monotone; the AdamW fallback
    keeps a scalar LR (only the Muon group anneals, mirroring the base trainer)."""
    mlr, frac, steps = 3e-3, 0.1, 100
    opt = build_muon_finisher_optimizer(
        muon_lr=mlr, muon_adamw_lr=1e-4, muon_lr_final_frac=frac, muon_anneal_steps=steps,
    )
    assert _muon_has_lr_schedule(opt), "Muon group must carry the cosine schedule"
    sched = opt.optimizers[0]._schedulers["learning_rate"]
    lr0, lr_mid, lr_end = (float(sched(mx.array(s))) for s in (0, steps // 2, steps))
    assert abs(lr0 - mlr) <= 1e-6 * mlr, f"schedule must start at muon_lr, got {lr0}"
    assert abs(lr_end - mlr * frac) <= 1e-6 * mlr, f"schedule must end at muon_lr*frac, got {lr_end}"
    assert lr0 > lr_mid > lr_end, f"schedule must decay monotonically: {lr0}, {lr_mid}, {lr_end}"
    # AdamW fallback group is NOT scheduled (only the Muon group decays).
    assert "learning_rate" not in getattr(opt.optimizers[1], "_schedulers", {})


# ---------------------------------------------------------------------------
# GAP 2 (default-off): --muon-warm-start-momentum seeds the fresh Muon child's
# momentum (state 'v') from the OUTGOING AdamW first-moment (state 'm'). The seeder
# lives in the base trainer (imported by the levelset launch path); this test proves
# it seeds the MultiOptimizer's Muon CHILD (opt.optimizers[0]) correctly.
# ---------------------------------------------------------------------------
def test_warm_start_seeds_muon_child_momentum_from_adamw():
    import sys
    from pathlib import Path

    _exp = str(Path(__file__).resolve().parents[4] / "experiments")
    if _exp not in sys.path:
        sys.path.insert(0, _exp)
    try:
        from train_witness_realized_through_R_mlx import _seed_muon_momentum_from_adam
    except Exception as e:  # base trainer unimportable in this env -> skip (not a helper failure)
        pytest.skip(f"base trainer not importable: {e}")

    from mlx.utils import tree_flatten

    params = {"hidden": {"weight": mx.ones((4, 4)), "bias": mx.zeros((4,))}}
    # An outgoing AdamW that has accumulated a nonzero first moment on the 2-D weight.
    adam = optim.AdamW(learning_rate=1e-3)
    adam.init(params)
    adam.apply_gradients({"hidden": {"weight": mx.full((4, 4), 0.5), "bias": mx.zeros((4,))}}, params)
    mx.eval(adam.state)

    opt = build_muon_finisher_optimizer(muon_lr=2e-3, muon_adamw_lr=1e-4)
    opt.init(params)
    mx.eval(opt.state)
    # cold init: Muon momentum is all zeros BEFORE seeding
    v_before = float(mx.sum(mx.abs(dict(tree_flatten(opt.optimizers[0].state))["hidden.weight.v"])))
    assert v_before == 0.0

    n = _seed_muon_momentum_from_adam(opt.optimizers[0], adam.state)
    mx.eval(opt.state)
    vmap = {k: v for k, v in tree_flatten(opt.optimizers[0].state) if k.endswith(".v")}
    mmap = {k[:-2]: v for k, v in tree_flatten(adam.state) if k.endswith(".m")}
    # exactly the one 2-D weight got seeded; its momentum now equals the outgoing AdamW m
    assert n == 1
    seeded = float(mx.sum(mx.abs(vmap["hidden.weight.v"])))
    src = float(mx.sum(mx.abs(mmap["hidden.weight"])))
    assert seeded > 0.0 and abs(seeded - src) < 1e-6
    # the 1-D bias is AdamW-routed -> NOT present in the Muon child's state
    assert "hidden.bias.v" not in vmap


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


# ---------------------------------------------------------------------------
# NO-FAKE (LENS-A adversarial review, FEED-fj): the REALIZED MultiOptimizer step
# on the ACTUAL production param tree, where `hidden` is a LIST (the real witness
# is ``self.hidden = [nn.Linear(...) for _ in range(n_hidden)]`` => leaf paths
# ``hidden.0.weight``, ``hidden.1.weight``). The realized-step test above
# deliberately uses a NO-LIST dict ("a numeric list index ... makes
# MultiOptimizer's split/merge ambiguous") -- so the PRODUCTION structure was
# UNTESTED. These two tests close that gap: they prove tree_flatten ->
# filter-split -> tree_unflatten -> tree_merge handles the numeric-list path
# correctly (Muon really orthogonalizes the list-indexed hidden weights; the
# merged tree keeps every leaf in the right list slot).
# ---------------------------------------------------------------------------
def _sv_ratio(a):
    s = np.linalg.svd(np.asarray(a), compute_uv=False)
    return float(s.max() / s.min())


def test_realized_step_on_list_indexed_hidden_routes_and_orthogonalizes():
    """apply_gradients on a tree with `hidden` AS A LIST (the real witness structure).

    Proves the production path the existing realized-step test avoided: the
    list-indexed ``hidden.0.weight`` / ``hidden.1.weight`` ARE routed to Muon and
    orthogonalized (spread spectrum -> ~flat), the structure round-trips (hidden
    stays a 2-list with every leaf), and biases/heads/code go to AdamW.
    """
    rng = np.random.default_rng(11)
    H = 16
    params = {
        "in_proj": {"weight": mx.zeros((H, 8)), "bias": mx.zeros((H,))},
        "hidden": [
            {"weight": mx.zeros((H, H)), "bias": mx.zeros((H,))},
            {"weight": mx.zeros((H, H)), "bias": mx.zeros((H,))},
        ],
        "out_sdf": {"weight": mx.zeros((5, H)), "bias": mx.zeros((5,))},
        "code": mx.zeros((4, 8)),
    }

    def _grad_like(p):
        if isinstance(p, dict):
            return {k: _grad_like(v) for k, v in p.items()}
        if isinstance(p, list):
            return [_grad_like(v) for v in p]
        if getattr(p, "ndim", 0) == 2:
            m, n = p.shape
            k = min(m, n)
            return mx.array(_spread_matrix(rng, m, n, [4.0 / (1 + i) for i in range(k)]))
        return mx.array(rng.normal(size=p.shape).astype(np.float32))

    grads = _grad_like(params)
    opt = build_muon_finisher_optimizer(
        muon_lr=1.0, muon_adamw_lr=1e-2, muon_momentum=0.0,
        muon_weight_decay=0.0, muon_ns_steps=5, adamw_weight_decay=0.0, nesterov=False,
    )
    opt.init(params)
    new = opt.apply_gradients(grads, params)
    mx.eval(new)

    # structure round-trip: hidden is still a 2-list with both leaves present.
    assert isinstance(new["hidden"], list) and len(new["hidden"]) == 2
    for i in (0, 1):
        assert "weight" in new["hidden"][i] and "bias" in new["hidden"][i]
    assert "code" in new and "out_sdf" in new

    # Muon group: list-indexed hidden weights orthogonalized (ratio < 2 from 4x-spread grad).
    for i in (0, 1):
        dw = np.asarray(new["hidden"][i]["weight"])
        assert np.linalg.norm(dw) > 1e-6, f"hidden.{i}.weight did NOT update (Muon dropped the list entry!)"
        assert _sv_ratio(dw) < 2.0, f"hidden.{i}.weight NOT orthogonalized (Muon skipped list path)"
    # AdamW group: head/bias/code moved.
    assert np.linalg.norm(np.asarray(new["out_sdf"]["weight"])) > 1e-6
    assert np.linalg.norm(np.asarray(new["code"])) > 1e-6
    for i in (0, 1):
        assert np.linalg.norm(np.asarray(new["hidden"][i]["bias"])) > 1e-6


def test_opt_update_on_nn_module_with_hidden_list_drives_muon_group():
    """End-to-end ``opt.update(model, grads)`` (the EXACT trainer loop call) on an
    nn.Module whose ``hidden`` is a Python list -> the production Muon path.

    The trainer calls ``opt.update(model, clipped)`` (not ``apply_gradients``);
    this proves that full path moves the list-indexed Muon-group weights.
    """
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.code = mx.zeros((4, 8))
            self.in_proj = nn.Linear(8, 16)
            self.hidden = [nn.Linear(16, 16) for _ in range(2)]
            self.out_sdf = nn.Linear(16, 5)

        def __call__(self, x):
            h = self.in_proj(x)
            for layer in self.hidden:
                h = layer(h)
            return self.out_sdf(h)

    model = _Tiny()
    mx.eval(model.parameters())
    before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    opt = build_muon_finisher_optimizer(
        muon_lr=0.5, muon_adamw_lr=1e-2, muon_momentum=0.0,
        muon_weight_decay=0.0, muon_ns_steps=5, adamw_weight_decay=0.0, nesterov=False,
    )
    opt.init(model.trainable_parameters())
    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(size=(32, 8)).astype(np.float32))
    y = mx.array(rng.normal(size=(32, 5)).astype(np.float32))

    def _loss(m, x, y):
        return ((m(x) - y) ** 2).mean()

    _, grads = nn.value_and_grad(model, _loss)(model, x, y)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)
    after = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    # the two list-indexed hidden weights + in_proj.weight MUST have moved (Muon group).
    for k in ("hidden.0.weight", "hidden.1.weight", "in_proj.weight"):
        assert np.linalg.norm(after[k] - before[k]) > 1e-7, f"{k} did NOT move via opt.update (production Muon path broken)"


# ---------------------------------------------------------------------------
# #224 Wave D (R4 non-blocking #2): adamw_beta2 threading + bias-correction gate
# ---------------------------------------------------------------------------
def _adamw_sub(opt):
    """Return the AdamW sub-optimizer of the MultiOptimizer (fallback = last)."""
    subs = getattr(opt, "optimizers", None) or getattr(opt, "_optimizers", None)
    assert subs is not None, "MultiOptimizer exposes no optimizers list"
    adamws = [s for s in subs if type(s).__name__ == "AdamW"]
    assert len(adamws) == 1, f"expected exactly one AdamW sub-optimizer, got {len(adamws)}"
    return adamws[0]


def test_adamw_bias_correction_gate_off_at_default():
    """The gate is OFF at exactly the MLX default 0.999 (byte-identical path)."""
    assert _adamw_bias_correction_for(0.999) is False


def test_adamw_bias_correction_gate_on_off_default():
    """The gate is ON for the all-levers small-n optimum (and any value != 0.999)."""
    assert _adamw_bias_correction_for(0.9999999) is True
    assert _adamw_bias_correction_for(0.95) is True
    # tolerant to float noise right at the default
    assert _adamw_bias_correction_for(0.999 + 1e-12) is False


def test_finisher_default_beta2_is_mlx_default_byte_identical():
    """Default build => AdamW rest-group carries betas [0.9, 0.999] + bias_correction False,
    which are the MLX AdamW defaults => byte-identical to the pre-Wave-D finisher construction."""
    opt = build_muon_finisher_optimizer(muon_lr=0.002, muon_adamw_lr=1e-4)
    adamw = _adamw_sub(opt)
    betas = [float(b) for b in adamw.betas]
    assert betas == [0.9, 0.999], f"default betas drifted: {betas}"
    assert bool(adamw.bias_correction) is False


def test_finisher_all_levers_beta2_threaded():
    """adamw_beta2=0.9999999 (the derived small-n optimum) is threaded into the AdamW
    rest-group AND turns bias_correction ON (the divergence fix)."""
    opt = build_muon_finisher_optimizer(
        muon_lr=0.002, muon_adamw_lr=1e-4, adamw_beta2=0.9999999,
    )
    adamw = _adamw_sub(opt)
    betas = [float(b) for b in adamw.betas]
    assert betas[0] == 0.9
    assert abs(betas[1] - 0.9999999) < 1e-12, f"beta2 not threaded: {betas}"
    assert bool(adamw.bias_correction) is True


def test_finisher_high_beta2_step_is_finite():
    """A finisher step at the high beta2 (+ bias correction) is FINITE (no ~100x LR blowup /
    NaN) on a tiny real model — the small-MLX smoke the Wave D task requires."""
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.code = mx.zeros((4, 8))
            self.in_proj = nn.Linear(8, 16)
            self.hidden = [nn.Linear(16, 16) for _ in range(2)]
            self.out_sdf = nn.Linear(16, 5)

        def __call__(self, x):
            h = self.in_proj(x)
            for layer in self.hidden:
                h = layer(h)
            return self.out_sdf(h)

    model = _Tiny()
    mx.eval(model.parameters())
    opt = build_muon_finisher_optimizer(
        muon_lr=0.002, muon_adamw_lr=1e-3, muon_momentum=0.0,
        muon_weight_decay=0.0, muon_ns_steps=5, adamw_weight_decay=0.0,
        adamw_beta2=0.9999999, nesterov=False,
    )
    opt.init(model.trainable_parameters())
    rng = np.random.default_rng(7)
    x = mx.array(rng.normal(size=(16, 8)).astype(np.float32))
    y = mx.array(rng.normal(size=(16, 5)).astype(np.float32))

    def _loss(m, x, y):
        return ((m(x) - y) ** 2).mean()

    for _ in range(3):
        _, grads = nn.value_and_grad(model, _loss)(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
    after = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
    for k, v in after.items():
        assert np.all(np.isfinite(v)), f"{k} became non-finite at high beta2 (divergence not gated)"
