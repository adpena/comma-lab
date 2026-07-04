"""BUILD #300 — SEED-ABSORPTION FIX: unit + $0 real-scorer confirmation tests.

Root cause of the CE plateau (memo ``.omx/research/plateau_disambiguator_results_20260704.md`` /
memory ``seed_compose_island_gradient_starvation_the_crutch_that_blocks_learning``): the island seed
(``--seed-islands``) is composited into the SegNet-scored frame1 (via ``_compose_chain``) and read by
EVERY realized-through-R seg lever, so once the seed satisfies the loss on the Lane+Movable island,
``dL/d(witness) ~= 0`` there and the witness never learns to FORM the islands itself -> deploy
(witness-alone) has ~0 island mass. The fix adds two coupled DEFAULT-OFF mechanisms:

  (a) ``--witness-alone-island-loss``: route the island-FORMATION levers (amplify + persistence)
      through the seed-EXCLUDED render so the witness gets the absorption gradient.
  (b) ``--seed-anneal-epochs``: ramp the seed COMPOSE WEIGHT full(1.0)->0.0 across CE (transfer).

Tests (all $0; MLX CPU is the authority device per [[mlx_gpu_not_bit_identical_crossprocess...]]):

  1. SCHEDULE (pure, exact): ``seed_compose_weight_at_epoch`` off=constant-1.0, linear/cosine full->0,
     monotone, clamped, edge cases. This is mechanism (b)'s schedule.
  2. COMPOSE BYTE-IDENTITY (mx arrays, exact): the seed-compose formula with the ``!= 1.0`` guard is
     BIT-IDENTICAL to ``rgb + res*mask`` at weight 1.0 (the DEFAULT / anneal-off path), removes the
     seed at weight 0.0 (anneal end == witness == deploy surface), and scales linearly in between.
  3. GRADIENT-FLOW CONFIRMATION (real frozen MLX SegNet + real render_through_R_mlx, slow, $0): the
     island-birth gradient w.r.t. the WITNESS is ~0 when the loss reads the SEED-COMPOSED margin (the
     seed satisfies it) but NONZERO when it reads the WITNESS-ALONE margin. This DIRECTLY proves the
     absorption pathway the fix restores (the memo's pending D1), on the exact primitives the trainer
     uses. Pointer 0.19110 UNMOVED (this is a $0 diagnosis-confirmation, not a score claim).

Run: .venv/bin/pytest experiments/test_seed_absorption_fix.py -v            (schedule + compose, fast)
     .venv/bin/pytest experiments/test_seed_absorption_fix.py -v -m ""      (incl. the slow real proof)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# MLX CPU is the equivalence AUTHORITY device (GPU is not bit-identical cross-process per
# [[mlx_gpu_not_bit_identical_crossprocess...]]; the sibling test_batched_seed_cograd.py pins the
# same convention).
mx.set_default_device(mx.cpu)
import mlx.nn as nn
from mlx.utils import tree_flatten

_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src"), str(_REPO / "upstream")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_GT_N6 = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"
_UPSTREAM = _REPO / "upstream"


def _lvl():
    """Lazy import of the trainer module (heavy: pulls the base trainer + tac boundary_math)."""
    import experiments.train_levelset_witness_realized_through_R_mlx as lvl
    return lvl


def _base():
    import experiments.train_witness_realized_through_R_mlx as base
    return base


# ===========================================================================
# TEST 1 — mechanism (b) SCHEDULE (pure, exact, no MLX render needed)
# ===========================================================================
def test_seed_anneal_off_is_constant_one():
    """--seed-anneal-epochs 0 (DEFAULT) => constant 1.0 at every epoch => _compose_chain byte-identical."""
    f = _lvl().seed_compose_weight_at_epoch
    for ep in (0, 1, 5, 100, 10_000):
        assert f(0, "linear", ep) == 1.0
        assert f(0, "cosine", ep) == 1.0
    # negative / nonsense anneal_epochs also treated as OFF (defensive).
    assert f(-3, "linear", 50) == 1.0


def test_seed_anneal_linear_full_to_zero():
    f = _lvl().seed_compose_weight_at_epoch
    N = 300
    assert f(N, "linear", 1) == 1.0            # start full
    assert f(N, "linear", N) == 0.0            # end zero (== tau-onset transfer complete)
    assert f(N, "linear", N + 50) == 0.0       # clamps past the end
    mid = f(N, "linear", (N + 1) // 2)
    assert abs(mid - 0.5) < 0.02               # ~halfway linear


def test_seed_anneal_cosine_full_to_zero():
    f = _lvl().seed_compose_weight_at_epoch
    N = 300
    assert f(N, "cosine", 1) == 1.0
    assert abs(f(N, "cosine", N)) < 1e-9       # end zero
    assert f(N, "cosine", N + 10) == 0.0       # clamp
    # cosine is above the linear ramp in the first half (slower initial decay), below in the second.
    lin_q = f(N, "linear", N // 4)
    cos_q = f(N, "cosine", N // 4)
    assert cos_q > lin_q


def test_seed_anneal_monotone_nonincreasing_and_bounded():
    f = _lvl().seed_compose_weight_at_epoch
    N = 250
    for shape in ("linear", "cosine"):
        prev = 2.0
        for ep in range(1, N + 5):
            w = f(N, shape, ep)
            assert 0.0 <= w <= 1.0
            assert w <= prev + 1e-9            # non-increasing
            prev = w


def test_seed_anneal_edge_anneal_epochs_one_no_div_by_zero():
    """anneal_epochs==1 is degenerate; must not divide-by-zero and stays in [0,1]."""
    f = _lvl().seed_compose_weight_at_epoch
    assert f(1, "linear", 1) == 1.0            # ep<=1 branch first
    assert f(1, "linear", 2) == 0.0            # ep>=anneal_epochs branch (no frac division)
    assert f(1, "cosine", 5) == 0.0


# ===========================================================================
# TEST 2 — mechanism (b) COMPOSE BYTE-IDENTITY + endpoints (mx arrays, exact)
# ===========================================================================
def _seed_compose(rgb, res, mask, compose_w):
    """Faithful replica of the trainer's _compose_chain seed step (the `!= 1.0` guard is the
    byte-identity contract: weight 1.0 => the extra multiply is NEVER emitted)."""
    sd = res * mask
    if compose_w != 1.0:
        sd = sd * compose_w
    return rgb + sd


def test_compose_weight_one_is_byte_identical():
    rng = np.random.default_rng(0)
    rgb = mx.array(rng.standard_normal((1, 8, 8, 3)).astype(np.float32))
    res = mx.array(rng.standard_normal((8, 8, 3)).astype(np.float32))
    mask = mx.array((rng.random((8, 8, 1)) > 0.5).astype(np.float32))
    got = _seed_compose(rgb, res, mask, 1.0)
    ref = rgb + res * mask                     # the pre-#300 seed compose, verbatim
    assert bool(mx.all(got == ref).item()), "compose_w=1.0 must be BIT-IDENTICAL to rgb + res*mask"


def test_compose_weight_zero_removes_seed():
    rng = np.random.default_rng(1)
    rgb = mx.array(rng.standard_normal((1, 6, 6, 3)).astype(np.float32))
    res = mx.array(rng.standard_normal((6, 6, 3)).astype(np.float32))
    mask = mx.array(np.ones((6, 6, 1), np.float32))
    got = _seed_compose(rgb, res, mask, 0.0)
    assert bool(mx.all(got == rgb).item()), "compose_w=0.0 (anneal end) must == witness render (no seed)"


def test_compose_weight_half_scales_linearly():
    rng = np.random.default_rng(2)
    rgb = mx.array(rng.standard_normal((1, 5, 7, 3)).astype(np.float32))
    res = mx.array(rng.standard_normal((5, 7, 3)).astype(np.float32))
    mask = mx.array((rng.random((5, 7, 1)) > 0.3).astype(np.float32))
    got = _seed_compose(rgb, res, mask, 0.5)
    ref = rgb + res * mask * 0.5
    assert float(mx.max(mx.abs(got - ref)).item()) < 1e-6


# ===========================================================================
# TEST 3 — mechanism (a) GRADIENT-FLOW CONFIRMATION (real frozen MLX SegNet, slow, $0)
#          The memo's pending D1: ∂L_island/∂witness ~0 through the SEED-COMPOSED margin, NONZERO
#          through the WITNESS-ALONE margin. Proven on the exact render + frozen scorer the trainer uses.
# ===========================================================================
def _grad_l2(tree) -> float:
    """Global L2 norm of a grad pytree (sum of per-leaf squared norms)."""
    s = 0.0
    for _k, g in tree_flatten(tree):
        s += float(mx.sum(mx.square(g)).item())
    return float(np.sqrt(s))


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(not _GT_N6.exists(), reason="gt_n6.npz cache missing")
@pytest.mark.skipif(not (_UPSTREAM / "modules.py").exists(), reason="upstream snapshot missing")
def test_witness_alone_island_gradient_is_nonzero_vs_seed_composed_zero():
    import torch
    import torch.nn.functional as tF

    from tac.boundary_math.island_protection import (
        build_island_masks,
        build_island_seed,
        identify_island_classes,
        island_birth_from_signed_mx,
        island_persistence_weight,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    lvl = _lvl()
    base = _base()
    RH, RW = 384, 512
    z = np.load(_GT_N6, allow_pickle=False)
    P = int(z["n_pairs"])
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(_UPSTREAM, device="cpu")

    model = lvl.build_levelset_rgb_witness(
        num_pairs=P, in_feat=12, hidden_dim=8, n_hidden=1, mod_dim=4, n_classes=5,
        activation="relu", softmax_temp=1.0, wire_w0=8.0, wire_s0=4.0,
        hosc_beta=4.0, hosc_omega=1.0, chroma=True)
    rng = np.random.default_rng(0)
    model.code = mx.array(rng.standard_normal((P * 2, 4)).astype(np.float32) * 0.3)
    mx.eval(model.parameters())

    ys, xs = np.meshgrid(np.linspace(-1, 1, RH, dtype=np.float32),
                         np.linspace(-1, 1, RW, dtype=np.float32), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        proj = coords @ (rng.standard_normal((6, 2)).astype(np.float32) * 2.0).T
    assert np.isfinite(proj).all()
    cf = mx.array(np.concatenate([np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], axis=1))
    mx.eval(cf)

    # self-detect the island classes over the full stack (trainer-faithful), then pick the pair with
    # the most island pixels so the gradient signal is unambiguous.
    st = np.stack([z["lstars"][i].astype(np.int64) for i in range(P)], axis=0)
    det = identify_island_classes(st, n_classes=5)
    masks = [build_island_masks(z["lstars"][i].astype(np.int64), det.lane_cls, det.movable_cls,
                                dilate_px=1) for i in range(P)]
    pbest = int(np.argmax([m.any_mask.sum() for m in masks]))
    im = masks[pbest]
    assert im.any_mask.sum() > 0, "no island pixels — cannot exercise the island gradient"
    lst0 = z["lstars"][pbest].astype(np.int64)                   # (RH, RW) at seg res
    c1 = 2 * pbest + 1                                            # the SegNet-scored (odd) frame code

    # GT frame1 down to render/seg res (the argmax grid); trainer-faithful interpolate.
    g1 = tF.interpolate(torch.from_numpy(z["gt_f1"][pbest].astype(np.float32)).permute(2, 0, 1)[None],
                        size=(RH, RW), mode="bilinear", align_corners=False
                        )[0].permute(1, 2, 0).numpy()

    # Build the seed so the COMPOSED (pre-R) frame == GT on the island: residual = GT - witness_preR.
    # This constructs, deterministically, the seed-satisfies-the-loss condition the REAL run reaches via
    # training (the memo's mechanism: composed correct => dL/d(composed) ~0 => dL/d(witness) ~0).
    witness_preR = np.asarray(mx.reshape(model(cf, c1), (RH, RW, 3)), np.float32)
    seed = build_island_seed(g1, im, base_render_segres=witness_preR, blend=1.0)
    seed_res = mx.array(seed.residual[None])                     # (1, RH, RW, 3)
    # FULL-FRAME GT compose for the composed case. WHY full-frame (not island-only): SegNet is REGIONAL
    # (EfficientNet-B2 stride-2 stem + deep conv; CLAUDE.md "SegNet sees REGIONS, not pixels"), so making
    # ONLY the island pixels correct while the CONTEXT (off-island) is a random untrained render still
    # misclassifies the island (the receptive field reads the garbage surround). In the REAL run the
    # TRAINED witness supplies the correct context and the island seed fixes the island -> composed
    # classified correctly -> gradient starved. Here we supply that correct-context condition
    # deterministically via a full-frame GT compose (residual = g1 - witness_preR over the whole frame),
    # isolating the HINGE-GATE mechanism the fix targets. The amplify loss is STILL island-weighted
    # (isl_w = 0 off-island), so the measured gradient is the ISLAND-formation gradient exactly.
    seed_res_full = mx.array((g1 - witness_preR)[None].astype(np.float32))   # (1, RH, RW, 3)
    isl_w = mx.array(island_persistence_weight(im.any_mask, kind="inverse_thickness")[None])  # (1,RH,RW)
    oh = mx.array(np.eye(5, dtype=np.float32)[lst0.ravel()].reshape(1, RH, RW, 5))
    mx.eval(seed_res, seed_res_full, isl_w, oh)

    # margin_target=0.0 => the island-birth hinge relu(0 - signed) fires ONLY where the island pixel is
    # MISCLASSIFIED (signed<0). On the composed (== GT) frame SegNet classifies the island correctly
    # (signed>=0) => hinge==0 in a neighborhood => the witness gradient is STARVED. On the witness-alone
    # (random-init) render the island is wrong => hinge>0 => the absorption gradient FLOWS.
    MTGT = 0.0

    def _compose_gt(rgb, code_idx):
        if int(code_idx) % 2 == 1:
            return rgb + seed_res_full[0]        # composed == g1 (correct-context seed carries the island)
        return rgb

    def _amplify_loss(m, compose_fn):
        f1 = base.render_through_R_mlx(m, cf, c1, RH, RW, compose_fn=compose_fn)
        slog = adapter.segnet(f1)
        sig_gt = mx.sum(slog * oh, axis=-1)
        sig_run = mx.max(slog + oh * (-1e9), axis=-1)
        signed = sig_gt - sig_run
        return island_birth_from_signed_mx(signed, isl_w, MTGT, form="hinge")

    loss_composed = nn.value_and_grad(model, lambda m: _amplify_loss(m, _compose_gt))
    loss_wa = nn.value_and_grad(model, lambda m: _amplify_loss(m, None))

    Lc, gc = loss_composed(model)
    Lw, gw = loss_wa(model)
    mx.eval(gc, gw)
    nc = _grad_l2(gc)
    nw = _grad_l2(gw)
    print(f"[BUILD#300 D1] seed-composed: L={float(Lc.item()):.5f} |grad|={nc:.3e}  "
          f"witness-alone: L={float(Lw.item()):.5f} |grad|={nw:.3e}  ratio={nw / (nc + 1e-30):.1f}")

    # The confirmation (memo D1): the witness-alone island gradient EXISTS (the absorption pathway the fix
    # routes through) while the seed-composed one is STARVED (the seed satisfies the loss -> hinge gate
    # closed -> ~0 witness gradient on the island). This is exactly what --witness-alone-island-loss
    # restores: amplify/persistence read _signed_wa (seed EXCLUDED) instead of the seed-composed _signed.
    assert nw > 1e-4, "witness-alone island gradient must be NONZERO (the restored absorption pathway)"
    assert nc < nw / 50.0, (
        f"seed-composed island grad ({nc:.3e}) must be STARVED relative to witness-alone ({nw:.3e}) — "
        "the seed-composed margin satisfies the island hinge so its witness gradient is ~0")
