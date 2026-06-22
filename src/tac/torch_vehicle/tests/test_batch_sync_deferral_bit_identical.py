# SPDX-License-Identifier: MIT
"""NO-FAKE proof that ``defer_batch_sync`` is a PURE read-deferral — the trained
weights are BIT-IDENTICAL with the lever ON vs OFF.

The lever (``TorchVehicleConfig.defer_batch_sync``) replaces ~3 per-batch
device→host ``.item()`` syncs on the NON-SPLIT training step with a single
on-device-accumulated read at epoch end (a throughput win on MPS, where every
``.item()`` flushes the command buffer + blocks). It must NOT change the math:
the loss/pose/grad-norm reads are logging-only on the non-split path (they feed
``epoch_loss``/``epoch_pose`` for the per-epoch mean + the last-batch grad-norm
log — never the gradient, weights, EMA, or any control flow; the one place
pose_mse drives control flow, the APGC pose-cadence controller, is split-only).

This test runs the SAME tiny CPU curriculum twice from the SAME seed — once with
``defer_batch_sync=False`` (the legacy per-batch-sync path) and once with
``=True`` — and asserts the decoder + ema_decoder + latents + ema_latents are
byte-for-byte identical. If a future edit ever makes the deferral touch the
compute graph, this FAILS here, not in a multi-day basin run.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import SyntheticScorerContext


def _ce(seg_logits, targets_hard):
    return F.cross_entropy(seg_logits, targets_hard)


def _stage() -> StageSpec:
    # batch_size < n_pairs → multiple batches/epoch, so the per-batch accumulation
    # path is genuinely exercised (3 batches/epoch at n_pairs=12, batch_size=4).
    return StageSpec(
        name="defer_ce", epochs=6, seg_loss_fn=_ce, eval_every=3, batch_size=4,
        ema_decay=0.999, use_muon=False, adamw_lr=1e-3, muon_lr=2e-4,
        muon_weight_decay=0.0, latent_lr_mult=10.0, grad_clip=1.0,
        grad_clip_muon=1.0, lr_floor_ratio=5e-6, seg_weight=100.0, pose_weight=1.0,
        cat_lambda=0.0, cat_sigma=0.2, use_qat=False, init_latents_random=True,
    )


def _run_hash(out_dir, defer: bool) -> str:
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=out_dir, checkpoint_every_epochs=1,
        device="cpu", train_device="cpu", split_by_head=False, seed=0,
        defer_batch_sync=defer,
    )
    scorer = SyntheticScorerContext(n_pairs=12, device="cpu", seed=0, split_by_head=False)
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(), curriculum=[_stage()],
    )
    driver.run()
    ck = torch.load(out_dir / "torch_vehicle_checkpoint_state.pt",
                    map_location="cpu", weights_only=False)
    h = hashlib.sha256()
    for key in ("decoder", "ema_decoder", "latents", "ema_latents"):
        v = ck[key]
        if isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode())
                h.update(v[k].detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_defer_batch_sync_bit_identical_weights(tmp_path):
    """The decisive guard: defer ON vs OFF → byte-identical trained weights."""
    off = _run_hash(tmp_path / "off", defer=False)
    on = _run_hash(tmp_path / "on", defer=True)
    assert on == off, (
        "defer_batch_sync changed the trained weights — it is NOT a pure read-"
        f"deferral. defer=OFF sha={off} vs defer=ON sha={on}. A real edit must "
        "leave gradients/weights/EMA bit-identical (the lever only changes WHEN "
        "logging scalars are read)."
    )


def test_defer_batch_sync_default_is_off():
    """Convention: the throughput lever defaults OFF (byte-identical default)."""
    cfg = TorchVehicleConfig(base_channels=8, latent_dim=28, device="cpu",
                             train_device="cpu", split_by_head=False, seed=0)
    assert cfg.defer_batch_sync is False
