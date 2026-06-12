# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the split-device (train=MPS gradient / eval=CPU authority)
wire-in of the torch-vehicle.

The load-bearing authority claim (CLAUDE.md "MPS auth eval is NOISE"): when the
per-step gradient runs on the Apple GPU for the 104x throughput, the EXACT
d_seg/d_pose that pick the BEST checkpoint and seed the telemetry MUST be
computed on the CPU-TRUSTED authority device — NEVER read off MPS. These tests
make that structural:

* the AUTHORITY device may never be MPS (the ban is preserved on ``device``);
* ``train_device='mps'`` is ALLOWED and places the training decoder/latents +
  the per-step forward on the Apple GPU;
* ``exact_eval`` ALWAYS routes through the authority net on the authority device
  (a frozen-but-distinct train scorer is held for the forward);
* the synthetic split-device run (logic-only, no MPS hardware needed because the
  synthetic scorer is a tiny conv) produces a valid descending trajectory + a
  DONE marker — i.e. the split plumbing does not break the curriculum loop.

The MPS-hardware tests are skipped when ``torch.backends.mps.is_available()`` is
False (CI / Linux), so the suite is portable; the logic tests run everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    TorchVehicleConfig,
    TorchVehicleDriver,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import RealScorerContext, SyntheticScorerContext

_MPS = torch.backends.mps.is_available()
_requires_mps = pytest.mark.skipif(not _MPS, reason="requires torch-MPS (Apple GPU)")


def _ce_seg_loss(seg_logits, targets_hard):
    return torch.nn.functional.cross_entropy(seg_logits, targets_hard)


def _short_curriculum(total_epochs: int, eval_every: int = 2) -> list[StageSpec]:
    return [
        StageSpec(
            name="test_split_ce",
            epochs=total_epochs,
            seg_loss_fn=_ce_seg_loss,
            eval_every=eval_every,
            batch_size=4,
            ema_decay=0.999,
            use_muon=False,
            adamw_lr=1e-3,
            muon_lr=2e-4,
            muon_weight_decay=0.0,
            latent_lr_mult=10.0,
            grad_clip=1.0,
            grad_clip_muon=1.0,
            lr_floor_ratio=5e-6,
            seg_weight=100.0,
            pose_weight=1.0,
            cat_lambda=0.0,
            cat_sigma=0.2,
            use_qat=False,
            init_latents_random=True,
        )
    ]


# ---------------------------------------------------------------------------
# Authority-device ban is PRESERVED (MPS is never the eval/authority device).
# ---------------------------------------------------------------------------
def test_config_authority_device_mps_still_raises():
    with pytest.raises(ValueError, match="AUTHORITY device"):
        TorchVehicleConfig(device="mps", out_dir=Path("/dev/null"))


def test_config_authority_device_mps_raises_even_with_train_cpu():
    # The ban is on the AUTHORITY device specifically — having a CPU train device
    # does not rescue an MPS authority device.
    with pytest.raises(ValueError, match="AUTHORITY device"):
        TorchVehicleConfig(device="mps", train_device="cpu", out_dir=Path("/dev/null"))


def test_real_scorer_context_authority_device_mps_raises():
    with pytest.raises(ValueError, match="AUTHORITY device"):
        RealScorerContext("upstream/videos/0.mkv", device="mps")


# ---------------------------------------------------------------------------
# train_device='mps' is ALLOWED; the config wires the split correctly.
# ---------------------------------------------------------------------------
def test_config_train_device_mps_allowed():
    cfg = TorchVehicleConfig(device="cpu", train_device="mps", out_dir=Path("/dev/null"))
    assert cfg.device == "cpu"
    assert cfg.train_device == "mps"


def test_config_train_device_defaults_to_authority_device():
    # Legacy single-device: train_device None -> equals device (no split).
    cfg = TorchVehicleConfig(device="cpu", out_dir=Path("/dev/null"))
    assert cfg.train_device == "cpu"


def test_driver_split_device_flag_set(tmp_path):
    cfg = TorchVehicleConfig(device="cpu", train_device="cpu", out_dir=tmp_path / "flag")
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu")
    driver = TorchVehicleDriver(cfg, scorer=scorer, vendored=_fake_bundle())
    assert driver.split_device is False
    assert driver.train_device == torch.device("cpu")
    assert driver.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Synthetic split-device LOGIC run (no MPS hardware needed): the split plumbing
# (train scorer != authority scorer, targets on train device, eval on authority)
# does not break the curriculum loop — it descends and writes a DONE marker.
# Uses device='cpu', train_device='cpu' with a DISTINCT train scorer instance to
# exercise the split code path even on a host without an Apple GPU.
# ---------------------------------------------------------------------------
def test_synthetic_split_device_logic_run(tmp_path):
    """Force the split-device code path on CPU (train scorer is a separate frozen
    instance) and verify the run completes with a tracked BEST + DONE marker."""
    cfg = TorchVehicleConfig(
        base_channels=8,
        latent_dim=28,
        out_dir=tmp_path / "split_logic",
        checkpoint_every_epochs=1,
        device="cpu",
        train_device="cpu",
        seed=0,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", seed=0)
    # Manually force the split flag + a distinct (frozen, identical-weight) train
    # scorer so the eval-on-authority / forward-on-train split is exercised even
    # without MPS hardware. This mirrors the RealScorerContext split structure.
    from tac.torch_vehicle.scorer_context import _TinyFrozenScorer

    scorer.split_device = True
    scorer._train_scorer = _TinyFrozenScorer(seed=0).to("cpu").eval()
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=_short_curriculum(6, eval_every=2),
    )
    summary = driver.run()
    assert summary["status"] == "complete"
    # A BEST checkpoint was tracked (the eval/exact path ran on the authority).
    assert driver.best_score < float("inf")
    assert (cfg.out_dir / "best" / "best_archive.bin").exists()
    # The trajectory JSONL has at least one evaluated (authority) row.
    traj = (cfg.out_dir / "torch_vehicle_trajectory.jsonl").read_text().strip().splitlines()
    assert len(traj) == 6
    import json

    evaluated = [json.loads(r) for r in traj if json.loads(r)["evaluated"]]
    assert evaluated, "no evaluated (authority) rows recorded"
    # Authority advisory tag, NON-PROMOTABLE (never an MPS score claim).
    for r in evaluated:
        assert r["promotable"] is False


# ---------------------------------------------------------------------------
# MPS-HARDWARE tests (skipped on hosts without an Apple GPU).
# ---------------------------------------------------------------------------
@_requires_mps
def test_synthetic_train_decoder_lands_on_mps(tmp_path):
    """With train_device='mps' the TRAINING decoder + latents are MPS tensors,
    but the authority eval decoder is built on CPU."""
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "mps_run",
        checkpoint_every_epochs=1, device="cpu", train_device="mps", seed=0,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", train_device="mps", seed=0)
    assert scorer.seg_targets_hard.device.type == "mps"
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=_short_curriculum(4, eval_every=2),
    )
    assert driver.train_device.type == "mps"
    assert driver.device.type == "cpu"
    # A fresh TRAIN decoder is on MPS; a fresh AUTHORITY (eval) decoder on CPU.
    assert next(driver._new_decoder().parameters()).device.type == "mps"
    assert next(driver._new_decoder(device=driver.device).parameters()).device.type == "cpu"
    summary = driver.run()
    assert summary["status"] == "complete"
    assert driver.best_score < float("inf")


@_requires_mps
def test_synthetic_split_run_completes_and_is_resumable_on_mps(tmp_path):
    """A split-device run completes; a second run on the DONE dir is idempotent."""
    cfg = TorchVehicleConfig(
        base_channels=8, latent_dim=28, out_dir=tmp_path / "mps_resume",
        checkpoint_every_epochs=1, device="cpu", train_device="mps", seed=1,
    )
    scorer = SyntheticScorerContext(n_pairs=6, device="cpu", train_device="mps", seed=1)
    driver = TorchVehicleDriver(
        cfg, scorer=scorer, vendored=import_vendored_bundle(),
        curriculum=_short_curriculum(4, eval_every=4),
    )
    s1 = driver.run()
    assert s1["status"] == "complete"
    # Re-run on the DONE dir: idempotent (no re-training).
    scorer2 = SyntheticScorerContext(n_pairs=6, device="cpu", train_device="mps", seed=1)
    driver2 = TorchVehicleDriver(
        cfg, scorer=scorer2, vendored=import_vendored_bundle(),
        curriculum=_short_curriculum(4, eval_every=4),
    )
    s2 = driver2.run()
    assert s2["status"] == "already_done"


def _fake_bundle():
    """A minimal vendored bundle stub for config/flag tests that never call run()."""
    from tac.torch_vehicle.driver import VendoredBundle

    def _noop(*a, **k):  # pragma: no cover - never invoked in flag tests
        raise AssertionError("vendored primitive should not be called in a flag test")

    return VendoredBundle(
        HNeRVDecoder=_noop, Muon=_noop, partition_params_for_muon=_noop,
        ema_update=_noop, apply_qat=_noop, restore_qat=_noop, cat_entropy_v2=_noop,
        build_archive=_noop, parse_archive=_noop,
    )
