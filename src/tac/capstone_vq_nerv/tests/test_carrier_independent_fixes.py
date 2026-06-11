# SPDX-License-Identifier: MIT
"""Behavioral tests for the carrier-INDEPENDENT correctness fixes (definitive audit).

Each test FAILS if its fix is reverted (NO-FAKE — they assert behavior, not
constants). The fixes (per
``.omx/research/capstone_pr95_fullstack_definitive_audit_synthesis_20260611T023748Z.md``):

  [B1] cosine LR schedule — both LRs, per-epoch, per-stage restart.
  [B2] per-stage optimizer-state / bias-correction reset.
  [A1] weight-EMA in the capstone (build/update/snapshot-restore-eval/EXPORT shadow).
  [A2] score the RELOADED int8 archive (the honest contest predictor).
  [A3] BICUBIC camera resize in the inflate (matches PR95 _decoded_to_camera).
  [A4] the dead bespoke ``_exact_d_pose`` is gone (mean_d_pose routes via the bridge).

These use a fast, well-conditioned color/luma-prototype frozen scorer (the same
stand-in the sister capstone tests use); a contest score needs
``upstream/evaluate.py`` on paired CUDA + Linux-x86_64 CPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    _HAVE_MLX = True
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _HAVE_MLX = False

skip_no_mlx = pytest.mark.skipif(not _HAVE_MLX, reason="mlx not available")


# ---------------------------------------------------------------------------
# Fast frozen proto scorer (color SegNet + luma PoseNet) — NOT the upstream net,
# so the C1 yuv6 assertion (real-PoseNet-only) is exempt here.
# ---------------------------------------------------------------------------


class _ColorProtoSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        protos = torch.tensor(
            [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]],
            dtype=torch.float32,
        )
        self.c = nn.Conv2d(3, 5, 1)
        self.c.weight.data = protos.reshape(5, 3, 1, 1) / 128.0
        self.c.bias.data = -(protos**2).sum(1) / (2 * 128.0 * 128.0)

    def forward(self, x):
        return self.c(x * 255.0)


class _LumaPose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(12, 6)
        torch.manual_seed(0)
        self.w.weight.data = torch.randn(6, 12) * 0.1
        self.w.bias.data = torch.zeros(6)

    def forward(self, x):
        return {"pose": self.w(x.mean(dim=(2, 3)))}


class _FrozenDNet(nn.Module):
    def __init__(self, *, with_pose: bool = False) -> None:
        super().__init__()
        self.segnet = _ColorProtoSeg()
        self.posenet = _LumaPose() if with_pose else None

    def preprocess_input(self, bhwc):
        last = bhwc[:, -1].permute(0, 3, 1, 2)
        first = bhwc[:, 0].permute(0, 3, 1, 2)
        pose_in = torch.cat([first.repeat(1, 2, 1, 1), last.repeat(1, 2, 1, 1)], dim=1)
        return pose_in, last / 255.0


def _frozen_dnet(*, with_pose: bool = False):
    dnet = _FrozenDNet(with_pose=with_pose).eval()
    for p in dnet.parameters():
        p.requires_grad = False
    return dnet


def _setup(n_pairs=6, h=48, w=64, seed=0, with_pose=True, K=16, base_ch=16):
    from tac.capstone_vq_nerv.capstone_trainer import (
        CapstoneTrainConfig,
        CapstoneTrainer,
    )
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    dnet = _frozen_dnet(with_pose=with_pose)
    seg_tgt = torch.zeros(n_pairs, h, w, dtype=torch.long)
    bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]
    for i in range(n_pairs):
        for cls, (r0, r1) in enumerate(bands):
            seg_tgt[i, r0:r1, :] = (cls + i) % 5
    rng = np.random.default_rng(seed)
    pose_tgt = (
        torch.from_numpy(rng.standard_normal((n_pairs, 6)).astype(np.float32))
        if with_pose
        else None
    )
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(
            num_pairs=n_pairs, base_channels=base_ch, codebook_size=K, seed=seed
        )
    )
    bridge = TorchScorerBridge(
        dnet, seg_tgt, pose_tgt, seg_loss_form="ce_seg_loss",
        seg_weight=100.0, pose_weight=1.0, eval_roundtrip=True, scorer_hw=(h, w),
    )
    pose_store = (
        pose_tgt.numpy().astype(np.float32)
        if pose_tgt is not None
        else np.zeros((n_pairs, 6), dtype=np.float32)
    )
    return bundle, bridge, pose_store, CapstoneTrainConfig, CapstoneTrainer


# ===========================================================================
# [B1] cosine LR schedule
# ===========================================================================


def test_b1_cosine_helper_matches_pr95_lr_lambda():
    """The cosine helper is the EXACT PR95 ``lr_lambda`` (0.5*(1+cos)) floored at eta_min."""
    from tac.local_acceleration.pr95_hnerv_mlx import pr95_cosine_lr_scale

    total = 100
    base_lr = 3e-5
    # PR95: eta_min_ratio = max(lr_floor_ratio/adamw_lr, 1e-3).
    eta_min = max(5e-6 / base_lr, 1e-3)
    # epoch 0 -> 1.0 (no decay at the start); final epoch -> floor.
    assert pr95_cosine_lr_scale(0, total, base_lr=base_lr) == pytest.approx(1.0)
    assert pr95_cosine_lr_scale(total, total, base_lr=base_lr) == pytest.approx(eta_min)
    # midpoint -> 0.5; matches the raw cosine where it exceeds the floor.
    assert pr95_cosine_lr_scale(50, total, base_lr=base_lr) == pytest.approx(0.5, abs=1e-9)
    for ep in (10, 25, 70):
        expected = max(0.5 * (1 + math.cos(math.pi * ep / total)), eta_min)
        assert pr95_cosine_lr_scale(ep, total, base_lr=base_lr) == pytest.approx(expected)
    # monotone non-increasing over the schedule.
    vals = [pr95_cosine_lr_scale(ep, total, base_lr=base_lr) for ep in range(total + 1)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


@skip_no_mlx
def test_b1_optimizer_step_scales_both_lrs_by_lr_scale():
    """``apply_pr95_mlx_optimizer_step(lr_scale=s)`` scales BOTH Muon + AdamW LRs by s.

    REVERT-CATCH: if the step ignores lr_scale, the lr_scale=0.5 update equals the
    lr_scale=1.0 update (the assertion that they DIFFER would fail).
    """
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.local_acceleration.pr95_hnerv_mlx import (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        build_parameter_group_lr_policy_fingerprint,
        pr95_mlx_parameter_shape_records,
    )

    def _one(scale):
        np.random.seed(0)
        mx.random.seed(0)
        b = CapstoneVqNervBundle(
            CapstoneVqNervConfig(num_pairs=4, base_channels=16, codebook_size=16, seed=0)
        )
        fp = build_parameter_group_lr_policy_fingerprint(
            pr95_mlx_parameter_shape_records(b.trainable_parameters())
        )
        # a fixed non-zero gradient (ones) so the step magnitude is deterministic.
        grads = {k: mx.ones_like(v) for k, v in tree_flatten(b.trainable_parameters())}
        from mlx.utils import tree_unflatten

        grads_tree = tree_unflatten(list(grads.items()))
        before = {k: mx.array(v) for k, v in tree_flatten(b.trainable_parameters())}
        cfg = Pr95MlxOptimizerConfig(use_muon=True, adamw_lr=2e-2, muon_lr=3e-2)
        st = Pr95MlxOptimizerState()
        summary = apply_pr95_mlx_optimizer_step(
            b, grads_tree, st, cfg, parameter_group_fingerprint=fp, lr_scale=scale
        )
        after = {k: mx.array(v) for k, v in tree_flatten(b.trainable_parameters())}
        # total L2 update magnitude across all params.
        mag = 0.0
        for k in before:
            d = after[k] - before[k]
            mag += float(mx.sum(d * d))
        return mag, summary

    mag_full, s_full = _one(1.0)
    mag_half, s_half = _one(0.5)
    # the summary reports the effective LRs scaled.
    assert s_full["lr_scale"] == pytest.approx(1.0)
    assert s_half["lr_scale"] == pytest.approx(0.5)
    assert s_half["effective_muon_lr"] == pytest.approx(0.5 * 3e-2)
    assert s_half["effective_adamw_lr"] == pytest.approx(0.5 * 2e-2)
    # REVERT-CATCH: a smaller lr_scale takes a strictly smaller step.
    assert mag_half < mag_full * 0.9, (
        f"lr_scale must shrink the step: full={mag_full} half={mag_half}"
    )


@skip_no_mlx
def test_b1_optimizer_step_rejects_nonpositive_lr_scale():
    """``lr_scale <= 0`` is rejected (a degenerate schedule, not silently accepted)."""
    from mlx.utils import tree_unflatten

    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.local_acceleration.pr95_hnerv_mlx import (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
    )

    b = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=2, base_channels=16, codebook_size=16, seed=0)
    )
    grads = tree_unflatten(
        [(k, mx.zeros_like(v)) for k, v in tree_flatten(b.trainable_parameters())]
    )
    cfg = Pr95MlxOptimizerConfig(use_muon=True)
    with pytest.raises(ValueError, match="lr_scale"):
        apply_pr95_mlx_optimizer_step(b, grads, Pr95MlxOptimizerState(), cfg, lr_scale=0.0)


@skip_no_mlx
def test_b1_effective_lr_decays_per_epoch_in_training_loop():
    """The trainer threads a DECAYING per-epoch cosine into the step (not constant).

    REVERT-CATCH: with cosine ON, the per-epoch lr_scale in the telemetry strictly
    decreases across the run; with cosine OFF it is constant 1.0.
    """
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=8, with_pose=False)
    cfg = Cfg(
        epochs=10, batch_size=4, eval_every=1, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, cosine_lr_schedule=True,
        use_ema_for_eval=False,
    )
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    out = trainer.train()
    lr_scales = [row["lr_scale"] for row in out["trajectory"]]
    assert lr_scales[0] == pytest.approx(1.0)
    assert lr_scales[-1] < 0.2, f"final lr_scale should be small: {lr_scales[-1]}"
    assert all(
        lr_scales[i] >= lr_scales[i + 1] - 1e-9 for i in range(len(lr_scales) - 1)
    ), f"cosine lr_scale must be non-increasing: {lr_scales}"
    assert out["lr_scale_final"] < 0.2

    # cosine OFF -> constant 1.0.
    bundle2, bridge2, pose2, Cfg2, Trainer2 = _setup(n_pairs=8, with_pose=False)
    cfg2 = Cfg2(
        epochs=10, batch_size=4, eval_every=1, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, cosine_lr_schedule=False,
        use_ema_for_eval=False,
    )
    out2 = Trainer2(bundle2, bridge2, pose2, cfg2).train()
    assert all(row["lr_scale"] == pytest.approx(1.0) for row in out2["trajectory"])


# ===========================================================================
# [B2] per-stage optimizer reset (bias-correction warmup + cosine restart)
# ===========================================================================


@skip_no_mlx
def test_b2_configure_stage_resets_optimizer_step_counter_and_cosine():
    """``configure_stage`` resets the optimizer step counter + cosine span (PR95 fresh opt)."""
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=4, with_pose=False)
    trainer = Trainer(bundle, bridge, pose_store, Cfg(epochs=2, batch_size=4, seed=0))
    # advance the optimizer state with a couple of steps.
    idx = np.arange(4)
    trainer.step(idx)
    trainer.step(idx)
    assert trainer.opt_state.step == 2
    # configure a stage -> the step counter resets (fresh optimizer per stage).
    stages = build_pr95_8stage_curriculum(total_epochs=16)
    trainer.configure_stage(stages[4], optimizer_schedule="muon_throughout")
    assert trainer.opt_state.step == 0, "B2: per-stage optimizer step must reset"
    # the cosine span is the stage's epoch count + base LR is the stage adamw_lr.
    assert trainer._cosine_total_epochs == stages[4].epochs
    assert trainer._cosine_base_lr == pytest.approx(stages[4].adamw_lr)
    assert trainer._current_epoch == 0
    # muon/adamw momentum buffers are cleared too (fresh momentum).
    assert trainer.opt_state.muon_buffers == {}
    assert trainer.opt_state.adamw_m == {}


# ===========================================================================
# [A1] weight-EMA: shadow != live, eval-on-shadow, EXPORT the shadow
# ===========================================================================


@skip_no_mlx
def test_a1_ema_shadow_diverges_from_live_and_export_bytes_shadow():
    """The EMA shadow diverges from live after training, and EXPORT bytes the shadow.

    REVERT-CATCH (export): if the export bytes LIVE weights, the exported render-basis
    will equal the live params and DIFFER from the shadow — the assertion fails.
    """
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=8, with_pose=False)
    cfg = Cfg(
        epochs=12, batch_size=4, eval_every=12, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, ema_decay=0.9, use_ema_for_eval=True,
        cosine_lr_schedule=False,
    )
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    trainer.train()

    live = {k: np.asarray(v) for k, v in tree_flatten(bundle.trainable_parameters())}
    shadow = {k: np.asarray(v) for k, v in trainer._ema.shadow.items()}
    # the shadow is a real EMA: it differs from the live final-step weights.
    max_div = max(
        float(np.max(np.abs(live[k] - shadow[k]))) for k in live if k in shadow
    )
    assert max_div > 1e-4, f"EMA shadow must diverge from live: {max_div}"

    # EXPORT bytes the SHADOW: every exported trainable tensor EQUALS the shadow.
    exported = trainer.export_render_weights()
    n_checked = 0
    n_differ_from_live = 0
    for k, ev in exported.items():
        if k in shadow:
            assert np.allclose(np.asarray(ev), shadow[k], atol=1e-5), (
                f"export must byte the EMA shadow for {k}"
            )
            if not np.allclose(np.asarray(ev), live[k], atol=1e-4):
                n_differ_from_live += 1
            n_checked += 1
    assert n_checked >= 3, "expected several decoder tensors to be checked against the shadow"
    # REVERT-CATCH: at least the diverged (decoder) tensors must NOT equal live —
    # if export byted LIVE, the shadow would equal live and n_differ_from_live==0.
    assert n_differ_from_live >= 1, (
        "export must byte the SHADOW (≠ live for the diverged decoder tensors); "
        "0 tensors differing from live means export byted the live weights"
    )


@skip_no_mlx
def test_a1_eval_under_ema_shadow_differs_from_live_eval():
    """``exact_d_seg(use_ema=True)`` evaluates the shadow (different from live eval)."""
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=8, with_pose=False)
    cfg = Cfg(
        epochs=12, batch_size=4, eval_every=12, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, ema_decay=0.9, use_ema_for_eval=True,
        cosine_lr_schedule=False,
    )
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    trainer.train()
    d_seg_shadow = trainer.exact_d_seg(use_ema=True)
    d_seg_live = trainer.exact_d_seg(use_ema=False)
    # the two are distinct measurements (shadow != live after training).
    assert d_seg_shadow != pytest.approx(d_seg_live, abs=1e-9) or d_seg_live == 0.0
    # and the eval is non-destructive: the live weights are restored after the
    # shadow eval (a second live eval matches the first).
    assert trainer.exact_d_seg(use_ema=False) == pytest.approx(d_seg_live, abs=1e-9)


@skip_no_mlx
def test_a1_codebook_ema_still_works_alongside_weight_ema():
    """The VQ codebook EMA (van den Oord) is INDEPENDENT of the new weight-EMA."""
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=8, with_pose=False)
    cb0 = np.asarray(bundle.quantizer._codebook).copy()
    cfg = Cfg(
        epochs=4, batch_size=4, eval_every=4, seed=0, muon_lr=3e-2, adamw_lr=2e-2,
        grad_clip=50.0, grad_clip_muon=50.0, ema_decay=0.9, use_ema_for_eval=True,
        cosine_lr_schedule=False,
    )
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    trainer.train()
    cb1 = np.asarray(bundle.quantizer._codebook)
    # the codebook moved (its OWN EMA ran), and it is NOT in the weight-EMA shadow.
    assert float(np.max(np.abs(cb1 - cb0))) > 0.0, "codebook EMA must still update"
    assert not any("codebook" in k.lower() for k in trainer._ema.shadow), (
        "the weight-EMA must NOT shadow the VQ codebook (it has its own EMA)"
    )


# ===========================================================================
# [A4] the dead bespoke ``_exact_d_pose`` is gone (mean_d_pose -> bridge.exact_d_pose)
# ===========================================================================


def test_a4_dead_exact_d_pose_function_removed():
    """The bespoke clamp-only ``_exact_d_pose`` module function is deleted (A4)."""
    import tac.capstone_vq_nerv.capstone_trainer as ct

    assert not hasattr(ct, "_exact_d_pose"), (
        "the dead clamp-only _exact_d_pose must be removed; mean_d_pose routes "
        "through bridge.exact_d_pose (roundtrip-consistent)"
    )


@skip_no_mlx
def test_a4_mean_d_pose_uses_roundtrip_bridge_path():
    """``mean_d_pose`` uses the roundtrip-consistent bridge path (>= clamp-only)."""
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=4, with_pose=True)
    trainer = Trainer(bundle, bridge, pose_store, Cfg(epochs=1, batch_size=4, seed=0))
    # the roundtrip d_pose (bridge.exact_d_pose) is what mean_d_pose returns.
    d_pose = trainer.mean_d_pose(use_ema=False)
    # compute clamp-only on the same render for a sanity bound (roundtrip >= clamp
    # is the eval_roundtrip discipline; here we assert the value is finite + the
    # method delegates to the bridge — no bespoke clamp-only path remains).
    assert math.isfinite(d_pose)
    # direct bridge call on the live render matches mean_d_pose's per-pair mean.
    idx = np.arange(4)
    render = trainer._render(mx.array(idx.astype(np.int32)), trainer._pose_mx(idx))
    mx.eval(render)
    direct = bridge.exact_d_pose(render, torch.from_numpy(idx.astype(np.int64)))
    assert d_pose == pytest.approx(direct, rel=1e-5, abs=1e-6)


# ===========================================================================
# [B4-FIX] EMA WARMUP decay — the shadow TRACKS the live weights from step 1, so
# exact_d_seg/exact_d_pose (and the EXPORT) reflect the trained weights, NOT a
# stale near-init shadow. The bug this guards (confirmed 2026-06-11,
# ``experiments/diag_curriculum_ema_lag.py``): with a constant decay 0.999, the
# shadow stayed ~init on a 25-epoch run, freezing exact_d_seg at 0.507 while the
# LIVE weights descended to 0.041 — a measurement+export poison falsely read as a
# "seg-capacity wall."
# ===========================================================================


@skip_no_mlx
def test_b4fix_ema_effective_decay_warms_up_from_low_to_cap():
    """``effective_decay`` ramps from ~0.1 at update 1 toward the cap as t grows.

    FAILS if the warmup is reverted to a constant decay (the lag bug): a constant
    decay would return the cap (0.997) at update 1 instead of ~0.18.
    """
    from tac.capstone_vq_nerv.capstone_trainer import _CapstoneWeightEMA

    bundle, *_ = _setup(n_pairs=4, with_pose=False)
    ema = _CapstoneWeightEMA(bundle, decay=0.997)
    # update 1: warmup (1+1)/(10+1)=0.1818 << cap. A constant-decay revert returns 0.997.
    ema._num_updates = 1
    assert ema.effective_decay() == pytest.approx(2.0 / 11.0, rel=1e-6)
    assert ema.effective_decay() < 0.5, "warmup must start far below the cap (shadow~live)"
    # update 30: still well below the cap, but climbing (the ramp is monotone).
    ema._num_updates = 30
    d30 = ema.effective_decay()
    assert 0.7 < d30 < 0.97
    # far future: saturates AT the cap (never exceeds it).
    ema._num_updates = 1_000_000
    assert ema.effective_decay() == pytest.approx(0.997, abs=1e-4)


@skip_no_mlx
def test_b4fix_ema_shadow_tracks_live_weights_on_short_run():
    """After a short run, the EMA-shadow WEIGHTS track the live weights.

    The fundamental, scorer-independent property the warmup restores: the shadow
    follows the live weights as they move from init. The constant-decay-0.999 bug
    left the shadow ~init (||shadow-live|| ~= ||live-init||); the warmup makes
    ||shadow-live|| a small fraction of the movement. Measured on the largest
    trainable decoder tensor (relative L2). FAILS if the warmup is reverted.
    """
    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=8, with_pose=True, base_ch=16)
    init_w = {k: np.asarray(v).copy() for k, v in tree_flatten(bundle.trainable_parameters())}
    cfg = Cfg(
        epochs=20, batch_size=8, eval_every=20, seed=0,
        muon_lr=2e-2, adamw_lr=2e-2, grad_clip=50.0, grad_clip_muon=50.0,
        ema_decay=0.999, use_ema_for_eval=True, cosine_lr_schedule=False,
    )
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    trainer.train()
    live_w = {k: np.asarray(v) for k, v in tree_flatten(bundle.trainable_parameters())}
    shadow_w = {k: np.asarray(v) for k, v in trainer._ema.shadow.items()}
    # pick the tensor that moved the MOST from init (the clearest tracking signal).
    moved = {k: float(np.linalg.norm(live_w[k] - init_w[k])) for k in live_w if k in init_w}
    key = max(moved, key=moved.get)
    movement = moved[key]
    assert movement > 1e-4, f"live weights must move from init (tensor {key})"
    shadow_to_live = float(np.linalg.norm(shadow_w[key] - live_w[key]))
    shadow_to_init = float(np.linalg.norm(shadow_w[key] - init_w[key]))
    # warmup: the shadow is much CLOSER to live than to init (it tracked).
    # constant-0.999 bug: the shadow stays ~init -> shadow_to_init ~ 0, shadow_to_live ~ movement.
    assert shadow_to_live < 0.5 * movement, (
        f"EMA shadow must TRACK live (warmup): tensor={key} ||shadow-live||="
        f"{shadow_to_live} ||shadow-init||={shadow_to_init} movement={movement}"
    )
    assert shadow_to_live < shadow_to_init, (
        "shadow must be closer to LIVE than to INIT (the lag bug inverts this)"
    )


# ===========================================================================
# [RECIPE-FIX BUG-A 2026-06-11] muon_throughout uses the CONFIG muon_lr/grad_clip,
# NOT the StageSpec's torch-tuned 2e-4/1.0 (the d_seg-wall recipe bug). The faithful
# pr95_adamw_then_muon path is byte-UNCHANGED (uses the StageSpec values).
# See .omx/research/pr95_seg_convergence_mechanism_and_recipe_gap_audit_20260611.md
# ===========================================================================


@skip_no_mlx
def test_bugA_muon_throughout_uses_config_muon_lr_not_stagespec():
    """Under muon_throughout, configure_stage must use cfg.muon_lr (the working 0.03),
    NOT the StageSpec's torch-tuned 2e-4 — the 150x-too-small value that walled d_seg."""
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=4, with_pose=False)
    # config carries the WORKING small-basis values (the muon-only arm reached 0.0037
    # with these); the StageSpec hardcodes muon_lr=2e-4, grad_clip_muon=1.0.
    cfg = Cfg(epochs=2, batch_size=4, seed=0, muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0)
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    stages = build_pr95_8stage_curriculum(total_epochs=16)
    # stage 1 (CE) under muon_throughout -> Muon active, config LR must win.
    trainer.configure_stage(stages[0], optimizer_schedule="muon_throughout")
    assert trainer.opt_config.muon_lr == pytest.approx(0.03), (
        "muon_throughout must use cfg.muon_lr (0.03), got "
        f"{trainer.opt_config.muon_lr} (the StageSpec 2e-4 would be the BUG)"
    )
    assert trainer.opt_config.grad_clip_muon == pytest.approx(50.0), (
        "muon_throughout must use cfg.grad_clip_muon (50), got "
        f"{trainer.opt_config.grad_clip_muon} (StageSpec 1.0 = the 100%-clip BUG)"
    )
    assert trainer.opt_config.grad_clip == pytest.approx(50.0)
    # NO-FAKE: the StageSpec's OWN value really is the small one (the bug we route around).
    assert stages[0].muon_lr == pytest.approx(2e-4)
    assert stages[0].grad_clip_muon == pytest.approx(1.0)


@skip_no_mlx
def test_bugA_faithful_pr95_schedule_still_uses_stagespec_values_byte_unchanged():
    """The FAITHFUL pr95_adamw_then_muon schedule is byte-UNCHANGED by the fix:
    it must still use the StageSpec's muon_lr/grad_clip (only stage 8 uses Muon)."""
    from tac.mlx_pr95_port.curriculum import build_pr95_8stage_curriculum

    bundle, bridge, pose_store, Cfg, Trainer = _setup(n_pairs=4, with_pose=False)
    cfg = Cfg(epochs=2, batch_size=4, seed=0, muon_lr=0.03, grad_clip=50.0, grad_clip_muon=50.0)
    trainer = Trainer(bundle, bridge, pose_store, cfg)
    stages = build_pr95_8stage_curriculum(total_epochs=16)
    # stage 8 muon_finetune under the FAITHFUL schedule -> StageSpec values must win
    # (config 0.03 must NOT leak in; the faithful path is unchanged by the fix).
    trainer.configure_stage(stages[7], optimizer_schedule="pr95_adamw_then_muon")
    assert trainer.opt_config.muon_lr == pytest.approx(stages[7].muon_lr), (
        "faithful schedule must use StageSpec muon_lr (byte-unchanged), got "
        f"{trainer.opt_config.muon_lr} vs spec {stages[7].muon_lr}"
    )
    assert trainer.opt_config.grad_clip_muon == pytest.approx(stages[7].grad_clip_muon)
    assert trainer.opt_config.grad_clip == pytest.approx(stages[7].grad_clip)
    assert trainer.opt_config.use_muon is True  # stage 8 IS the muon stage
