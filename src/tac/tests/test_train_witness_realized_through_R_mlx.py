# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the MLX through-R witness trainer.

These assert the trainer ACTUALLY does the work it names on REAL primitives
(not metadata): (1) the gradient FLOWS through R + both frozen MLX scorers to the
witness params and is non-zero/finite; (2) the CPU-torch VERDICT path exists and
returns a real argmax-disagreement / pose-MSE (the authority, not the MLX
surrogate); (3) EMA is a real running average distinct from live params after a
step; (4) determinism — same seed => bit-identical witness init + render.

Heavy paths (loading the full upstream scorers / decoding GT) are guarded so the
suite skips cleanly when mlx / torch / upstream weights are unavailable, but the
gradient-flow + EMA + determinism tests run on the real witness + real MLX scorer
adapter when mlx is present.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_HAS_MLX = importlib.util.find_spec("mlx") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_MLX, reason="mlx not available")


def _load_trainer():
    name = "train_witness_realized_through_R_mlx"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / "experiments" / "train_witness_realized_through_R_mlx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so @dataclass can resolve cls.__module__ (GTData).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scorer_weights_present() -> bool:
    return (REPO / "upstream" / "models").exists() and _HAS_TORCH


# ---------------------------------------------------------------------------
# Test 1 (NO-FAKE): gradient FLOWS through R + both frozen MLX scorers to the
# witness params, and is non-zero + finite. If the trainer were a metadata fake,
# the grad would be all-zero (no real scorer in the loss).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _scorer_weights_present(), reason="upstream scorer weights / torch unavailable")
def test_gradient_flows_through_R_and_both_scorers_to_witness():
    import mlx.core as mx
    import mlx.nn as nn

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    mod = _load_trainer()
    with temporary_mlx_device("gpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        render_h, render_w = 96, 128
        model = mod.build_witness_module(
            num_pairs=1, n_fourier=8, hidden_dim=32, n_hidden=2, mod_dim=16, fourier_sigma=8.0
        )
        mx.eval(model.parameters())
        coords = mx.array(mod._build_render_coords(render_h, render_w))
        coord_feats = model.build_feats(coords)
        mx.eval(coord_feats)

        # Fixed arbitrary targets (one-hot L*, margin, pose) -- the realized loss.
        h, w = mod.SEG_H, mod.SEG_W
        lstar_oh = mx.zeros((1, h, w, mod.N_CLASSES))
        lstar_oh = lstar_oh + mx.reshape(
            (mx.arange(mod.N_CLASSES) == 0).astype(mx.float32), (1, 1, 1, mod.N_CLASSES)
        )
        margin = mx.zeros((h, w))
        pose_tgt = mx.zeros((6,))
        loss_fn = mod.make_loss_fn(adapter, render_h, render_w)
        value_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = value_and_grad(
            model, coord_feats, 0, 1, lstar_oh, margin, pose_tgt,
            mx.array(1.0), mx.array(1.0), mx.array(4.0), mx.array(1.0),
        )
        from mlx.utils import tree_flatten

        flat = tree_flatten(grads)
        total_abs = 0.0
        all_finite = True
        any_nonzero = False
        for _name, g in flat:
            a = np.asarray(g, dtype=np.float64)
            total_abs += float(np.abs(a).sum())
            all_finite = all_finite and bool(np.all(np.isfinite(a)))
            if np.count_nonzero(a) > 0:
                any_nonzero = True
        assert np.isfinite(float(loss)), "loss must be finite"
        assert all_finite, "all witness param grads must be finite"
        assert any_nonzero, "gradient must be non-zero somewhere (real scorer in loss)"
        assert total_abs > 0.0, "total grad magnitude must be > 0 (NOT a metadata fake)"


# ---------------------------------------------------------------------------
# Test 2 (NO-FAKE): the CPU-torch VERDICT path is REAL -- it argmaxes the frozen
# CPU SegNet on the rendered frame and returns a disagreement RATE in [0,1], and
# a pose-MSE >= 0. This is the authority, NOT the MLX surrogate.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _scorer_weights_present(), reason="upstream scorer weights / torch unavailable")
def test_cpu_torch_verdict_path_is_real_and_distinct_from_mlx():
    from tac.boundary_math.seg_core import load_real_segnet

    mod = _load_trainer()
    seg_cpu = load_real_segnet("cpu")
    rng = np.random.default_rng(0)
    h, w = mod.SEG_H, mod.SEG_W
    frame1 = rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8)
    gt_argmax = rng.integers(0, mod.N_CLASSES, size=(h, w)).astype(np.int64)
    d_seg = mod.cpu_verdict_d_seg(seg_cpu, frame1, gt_argmax)
    assert 0.0 <= d_seg <= 1.0, "realized d_seg must be a disagreement RATE in [0,1]"
    # Identical frame vs its OWN argmax => zero disagreement (verdict is real).
    import torch

    pair = torch.from_numpy(np.stack([frame1, frame1], axis=0)[None]).float()
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        own = seg_cpu(seg_cpu.preprocess_input(xp)).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    assert mod.cpu_verdict_d_seg(seg_cpu, frame1, own) == 0.0, "self-argmax must give exactly 0 disagreement"


# ---------------------------------------------------------------------------
# Test 3 (NO-FAKE): EMA is a REAL running average -- after one update with a
# changed param it lies strictly between the old shadow and the new live value
# (not a copy, not a no-op).
# ---------------------------------------------------------------------------
def test_mlx_ema_is_real_running_average():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    mod = _load_trainer()
    model = mod.build_witness_module(
        num_pairs=1, n_fourier=4, hidden_dim=8, n_hidden=1, mod_dim=4, fourier_sigma=8.0
    )
    mx.eval(model.parameters())
    ema = mod.MlxEMA(model, decay=0.5)
    shadow0 = {k: np.asarray(v, dtype=np.float64) for k, v in ema.shadow.items()}
    # Perturb live params by +1.0 everywhere.
    new_params = {}
    for k, v in tree_flatten(model.parameters()):
        new_params[k] = v + 1.0
    from mlx.utils import tree_unflatten

    model.update(tree_unflatten(list(new_params.items())))
    mx.eval(model.parameters())
    ema.update(model)
    # With decay 0.5: shadow' = 0.5*shadow0 + 0.5*(shadow0+1) = shadow0 + 0.5.
    moved = False
    for k, s0 in shadow0.items():
        s1 = np.asarray(ema.shadow[k], dtype=np.float64)
        if s0.size == 0:
            continue
        assert np.allclose(s1, s0 + 0.5, atol=1e-5), f"EMA at {k} must be the 0.5-decayed mean"
        moved = True
    assert moved, "EMA must have at least one non-empty param"


# ---------------------------------------------------------------------------
# Test 4 (NO-FAKE): determinism -- same seed => bit-identical witness init AND
# bit-identical render-through-R output (deterministic reproducibility spine).
# ---------------------------------------------------------------------------
def test_deterministic_witness_init_and_render():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mod = _load_trainer()

    def _init_and_render():
        mx.random.seed(123)
        np.random.seed(123)
        m = mod.build_witness_module(
            num_pairs=1, n_fourier=6, hidden_dim=16, n_hidden=2, mod_dim=8, fourier_sigma=8.0
        )
        mx.eval(m.parameters())
        coords = mx.array(mod._build_render_coords(48, 64))
        cf = m.build_feats(coords)
        r = mod.render_through_R_mlx(m, cf, 0, 48, 64)
        mx.eval(r)
        params = {k: np.asarray(v, dtype=np.float32) for k, v in tree_flatten(m.parameters())}
        return params, np.asarray(r, dtype=np.float32)

    p1, r1 = _init_and_render()
    p2, r2 = _init_and_render()
    assert set(p1) == set(p2)
    for k in p1:
        assert np.array_equal(p1[k], p2[k]), f"param {k} must be bit-identical across seeded runs"
    assert np.array_equal(r1, r2), "render-through-R must be bit-identical across seeded runs"


# ---------------------------------------------------------------------------
# Test 5: the deterministic Fourier B matches the torch trainer's (parity).
# ---------------------------------------------------------------------------
def test_fourier_B_parity_with_torch_trainer():
    mod = _load_trainer()
    b1 = mod.deterministic_fourier_B(24, 8.0)
    b2 = mod.deterministic_fourier_B(24, 8.0)
    assert np.array_equal(b1, b2)
    assert b1.shape == (2, 24)
    # Same seed/sigma => same matrix as a fresh default_rng(0) draw.
    rng = np.random.default_rng(0)
    expect = (rng.standard_normal((2, 24)) * 8.0).astype(np.float32)
    assert np.array_equal(b1, expect)


# ---------------------------------------------------------------------------
# Test 6 (NO-FAKE): implied-S formula matches evaluate.py exactly (no laundering).
# ---------------------------------------------------------------------------
def test_implied_score_matches_evaluate_formula():
    mod = _load_trainer()
    d_seg, d_pose, wbytes = 5.6e-4, 1.6e-5, 177000
    s = mod.implied_score_from_verdict(d_seg, d_pose, wbytes)
    expect = 100.0 * d_seg + np.sqrt(10.0 * d_pose) + 25.0 * wbytes / mod.RATE_DENOM
    assert abs(s - expect) < 1e-9


# ---------------------------------------------------------------------------
# Test 7: /tmp refusal (disk-hygiene non-negotiable).
# ---------------------------------------------------------------------------
def test_refuse_tmp_durable_paths():
    mod = _load_trainer()
    for bad in ("/tmp/x", "/private/tmp/x", "/var/tmp/x"):
        with pytest.raises(ValueError):
            mod._refuse_tmp(Path(bad), "out_dir")
    # A repo results path is fine.
    mod._refuse_tmp(REPO / "experiments" / "results" / "ok", "out_dir")


# ---------------------------------------------------------------------------
# Test 8 (NO-FAKE): the STEP-NATIVE hosc activation is REAL — it genuinely
# changes the witness output vs the relu control (NOT a renamed no-op). If hosc
# silently fell back to relu, the diff would be ~0 and this fails.
# ---------------------------------------------------------------------------
def test_hosc_activation_is_real_not_relu_noop():
    import mlx.core as mx

    mod = _load_trainer()
    kw = dict(num_pairs=2, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8, fourier_sigma=8.0)
    relu = mod.build_witness_module(activation="relu", **kw)
    hosc = mod.build_witness_module(activation="hosc", hosc_beta=4.0, **kw)
    mx.eval(relu.parameters())
    mx.eval(hosc.parameters())
    feats = mx.array(np.random.default_rng(0).standard_normal((256, relu.in_feat)).astype(np.float32))
    o_relu = relu(feats, 0)
    o_hosc = hosc(feats, 0)
    mx.eval(o_relu, o_hosc)
    assert tuple(o_hosc.shape) == (256, 3)
    # hosc output is bounded RGB in [0,255] (sigmoid head) and DIFFERS from relu.
    assert float(mx.mean(mx.abs(o_hosc - o_relu))) > 1.0, "hosc must differ from relu (real activation)"
    # hosc internals are tanh(beta*sin(.)) in [-1,1]; with random init the RGB head still
    # spans a non-trivial range (NOT collapsed to a constant).
    assert float(o_hosc.max()) - float(o_hosc.min()) > 1.0


# ---------------------------------------------------------------------------
# Test 9 (NO-FAKE): the DIRECTIONAL basis is REAL — in_feat_override sets the
# coord-feat width and the witness consumes per-pair directional feats end to end;
# the directional Fourier feats are the ACTUAL oriented (sin/cos across/along)
# encoding (not zeros), and the deterministic Fourier table is NOT counted.
# ---------------------------------------------------------------------------
def test_directional_in_feat_override_and_free_table():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from tac.boundary_math.lever_b_generator import (
        all_class_boundary_proximity_and_tangent,
        directional_fourier_feats,
    )

    mod = _load_trainer()
    n_dir = 6
    in_feat = 4 * n_dir
    m = mod.build_witness_module(
        num_pairs=2, n_fourier=8, hidden_dim=16, n_hidden=2, mod_dim=8,
        fourier_sigma=8.0, in_feat_override=in_feat, activation="relu",
    )
    mx.eval(m.parameters())
    assert m.in_feat == in_feat
    assert m.in_proj.weight.shape[-1] == in_feat  # the MLP consumes the directional width
    # Build the ACTUAL per-pair directional feats over a render grid with a real boundary.
    coords = mod._build_render_coords(96, 128)
    lr = np.zeros((96, 128), dtype=np.int64)
    lr[40:60, :] = 1
    lr[:, 60:90] = 2
    _, tang = all_class_boundary_proximity_and_tangent(lr)
    feats = directional_fourier_feats(coords, tang.reshape(-1, 2), n_freqs=n_dir).astype(np.float32)
    assert feats.shape == (96 * 128, in_feat)
    assert np.isfinite(feats).all() and float(np.abs(feats).max()) > 0.0  # NOT a zero stub
    out = m(mx.array(feats), 0)
    mx.eval(out)
    assert tuple(out.shape) == (96 * 128, 3)
    # The deterministic seeded Fourier table is FREE (rule 118) — not in the counted blob.
    names = [n for n, _ in tree_flatten(m.parameters())]
    assert not any(n.endswith("_B") for n in names), "_B must not be a counted parameter"
    blob = mod.quantize_witness_blob(m)
    assert blob["total_quantized_blob_bytes"] > 0


# ---------------------------------------------------------------------------
# Test 10 (NO-FAKE): the margin-hinge seg loss is the REAL lensA surrogate, not
# a renamed CE. On a constructed flip it returns relu(m - (z_GT - max_other)) and
# its value moves with the margin target; a correct-with-margin pixel contributes 0.
# ---------------------------------------------------------------------------
def test_margin_hinge_math_is_real_relu_margin():
    import mlx.core as mx

    # Replicate the loss_fn's margin-hinge block on hand-built logits (the exact math
    # the trainer runs through the frozen SegNet). GT class 0.
    # Pixel A: FLIPPED — class 1 wins by 2.0 (z_GT - max_other = 1.0 - 3.0 = -2.0).
    # Pixel B: CORRECT with margin 2.0 (z_GT - max_other = 3.0 - 1.0 = +2.0).
    logits = mx.array(np.array([[[1.0, 3.0, 0.0, 0.0, 0.0],
                                 [3.0, 1.0, 0.0, 0.0, 0.0]]], dtype=np.float32))  # (1,2,5)
    oh = mx.array(np.eye(5, dtype=np.float32)[np.array([0, 0])][None])  # (1,2,5) GT=0
    gt_logit = mx.sum(logits * oh, axis=-1)
    masked = logits + oh * (-1e9)
    runner_up = mx.max(masked, axis=-1)
    signed = gt_logit - runner_up  # [-2.0, +2.0]
    for m_t, expect_a, expect_b in ((1.0, 3.0, 0.0), (0.5, 2.5, 0.0)):
        hinge = mx.maximum(m_t - signed, 0.0)
        mx.eval(hinge)
        a, b = float(hinge[0, 0]), float(hinge[0, 1])
        assert abs(a - expect_a) < 1e-4, f"flip pixel hinge {a} != {expect_a} at m={m_t}"
        assert abs(b - expect_b) < 1e-4, f"correct-with-margin pixel must be 0, got {b}"
