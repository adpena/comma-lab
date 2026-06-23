#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build a stage-8-positioned checkpoint that JUMPS the PR95 curriculum from a
preserved stage-5 checkpoint directly into the genuine stage-8 Muon finetune.

WHY THIS IS GENUINE, NOT A FAKE HARNESS
=======================================
The driver (``tac.torch_vehicle.driver.TorchVehicleDriver.run``) already runs the
IDENTICAL stage-8 code path on a normal resume:

* the resume loop is ``for stage_index in range(resume_pos.stage_index, len(...))``
  (driver.py ~3286); a checkpoint with ``stage_index=7`` runs ONLY stage 8;
* ``_restore_into`` (driver.py ~2546) ALWAYS restores decoder/latents/EMA from the
  checkpoint (lines 2550-2555), and SKIPS the optimizer/scheduler restore when
  ``merged["adamw"] is None`` (line 2578) — the existing, tested FORK-SEED protocol
  ("fresh optimizers from these weights at this curriculum position");
* ``_build_stage_runtime`` (driver.py ~1784) builds the Muon + AdamW optimizers
  FRESH per stage when ``spec.use_muon`` (stage 8 has ``use_muon=True``), exactly as
  a natural stage-7->8 boundary does.

So a checkpoint positioned at ``stage_index=7 / epoch_in_stage=0`` with the WEIGHTS
+ EMA + Lever-4 sensitivity EMA KEPT and the OPTIMIZER state CLEARED, fed to the
driver's NORMAL resume, runs the bit-identical stage-8 Muon-finetune code path —
just seeded from the stage-5 weights instead of stage-7 weights. The only thing
skipped is the long stages 5-7 rate sweep (the rate prize is ~0.005 of S, cheap +
recoverable from the small basis; the d_seg finisher Muon does is the binding term).

This is NOT a divergent harness: every byte of the stage-8 training/eval/export
runs through ``driver.run()`` unchanged. The surgery only sets the resume cursor +
clears the optimizers (the documented fork-seed signal).

The stage-8 resume guards this surgery must satisfy (verified at build time):
* ``base_channels`` / ``latent_dim`` / ``n_pairs`` / ``taper_channels`` UNCHANGED
  (else the resume basis-mismatch guards fail closed — driver.py ~3185-3215);
* ``muon_lr_floor_fix`` left at the saved value (the preserved ckpt has
  ``has_muon=False`` so its toggle guard passes pre-stage-8; the relaunch passes
  ``--muon-lr-floor-fix`` and the Muon floor is built fresh at stage 8).

Authority: infrastructure, not a score axis. $0, local CPU only (NO MPS — the live
run owns the MPS GPU). NON-PROMOTABLE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tac.torch_vehicle.checkpoint import (
    TorchCheckpointPosition,
    checkpoint_exists,
    load_checkpoint,
    read_manifest,
    save_checkpoint,
)

# stage_index 7 == the 8th (0-based) curriculum stage == stage8_muon_finetune.
_STAGE8_INDEX = 7
_STAGE8_NAME = "stage8_muon_finetune"
# sum of PR95 stages 0-6 default epochs (3000+5650+1500+500+9000+2000+3000).
# A resume at stage_index=7/epoch_in_stage=0 sets driver._global_epoch to exactly
# this value; the smoke verifier asserts it.
_EXPECTED_GLOBAL_EPOCH_AT_STAGE8 = 24650


def _carry_keys() -> tuple[str, ...]:
    """Blob keys whose VALUES we KEEP from the source stage-5 checkpoint (the
    weights/EMA-carry side of the natural stage boundary)."""
    return (
        "decoder",
        "latents",
        "ema_decoder",
        "ema_latents",
        # Lever-4 score-aware QAT per-tensor sensitivity EMA — a property of the
        # CARRIED decoder (driver.py ~3269), so it belongs on the carry side. Helps
        # stage-8 QAT continue the same quant-grid trajectory instead of resetting
        # to uniform-127 for its first steps. Empty/None on the default path.
        "tensor_sensitivity_ema",
    )


def build_stage8_jump_checkpoint(
    src_dir: Path,
    out_dir: Path,
    *,
    map_location: str = "cpu",
) -> dict[str, object]:
    """Read the preserved stage-5 checkpoint and write a stage-8-positioned one.

    Returns a small dict of build facts (for the caller / smoke verifier).
    """
    if not checkpoint_exists(src_dir):
        raise FileNotFoundError(f"no checkpoint (state + manifest) at {src_dir}")

    src_man = read_manifest(src_dir)
    merged = load_checkpoint(src_dir, map_location=map_location)

    # ---- carry the weights/EMA/sensitivity; CLEAR the optimizer/scheduler state --
    new_state: dict[str, object] = {}
    for key in _carry_keys():
        # Clone tensors / state_dicts so we never alias the source blob.
        val = merged.get(key)
        if isinstance(val, dict):
            new_state[key] = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in val.items()
            }
        elif torch.is_tensor(val):
            new_state[key] = val.detach().cpu().clone()
        else:
            new_state[key] = val  # None / plain dict (sensitivity EMA may be {})

    # FORK-SEED signal: optimizers None -> _restore_into SKIPS the optimizer/scheduler
    # restore -> stage 8 builds Muon+AdamW FRESH from the carried weights (the natural
    # stage-7->8 boundary behavior). RNG: fresh (None) is fine for a fork-seed.
    new_state["adamw"] = None
    new_state["muon"] = None
    new_state["adamw_sched"] = None
    new_state["muon_sched"] = None
    new_state["torch_rng"] = None
    new_state["numpy_rng"] = None

    # Reset the per-controller running state to defaults (no adaptive controllers are
    # active on this faithful basin; explicit defaults keep the jump checkpoint clean).
    new_state["ema_step"] = 0
    new_state["pose_floor"] = None
    new_state["pose_mse_hist"] = []
    new_state["last_pose_epoch"] = -1
    new_state["equimarginal_ctrl"] = None
    new_state["weight_entropy_penalty"] = None

    # ---- manifest scalar fields: architecture UNCHANGED (resume guards), best RESET --
    # These MUST equal the source so the resume basis guards pass.
    new_state["base_channels"] = int(src_man["base_channels"])
    new_state["latent_dim"] = int(src_man["latent_dim"])
    new_state["n_pairs"] = int(src_man["n_pairs"])
    new_state["taper_channels"] = (
        list(src_man["taper_channels"]) if src_man.get("taper_channels") is not None else None
    )
    new_state["ema_decay"] = float(src_man.get("ema_decay", 0.999))
    # Leave muon_lr_floor_fix at the SAVED value (has_muon will be False on this jump
    # checkpoint -> the resume floor-fix toggle guard passes regardless; the relaunch
    # sets --muon-lr-floor-fix and the floor is built fresh at stage 8).
    new_state["muon_lr_floor_fix"] = bool(src_man.get("muon_lr_floor_fix", False))
    new_state["stage_lr_warmup_frac"] = (
        None if src_man.get("stage_lr_warmup_frac") is None
        else float(src_man["stage_lr_warmup_frac"])
    )
    new_state["stage_lr_warmup_start_ratio"] = (
        None if src_man.get("stage_lr_warmup_start_ratio") is None
        else float(src_man["stage_lr_warmup_start_ratio"])
    )
    # Position at stage 8 entry; name the stage so dashboards/telemetry read correctly.
    new_state["stage_name"] = _STAGE8_NAME
    # Reset best-so-far so the jump run tracks ITS OWN best (not the stage-5 best).
    new_state["best_score"] = float("inf")
    new_state["best_ep"] = 0
    new_state["best_stage"] = -1

    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        new_state,
        out_dir,
        TorchCheckpointPosition(stage_index=_STAGE8_INDEX, epoch_in_stage=0),
    )

    new_man = read_manifest(out_dir)
    return {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "src_stage_index": int(src_man["stage_index"]),
        "src_stage_name": src_man.get("stage_name"),
        "src_epoch_in_stage": int(src_man["epoch_in_stage"]),
        "new_manifest": new_man,
        "expected_global_epoch_at_stage8": _EXPECTED_GLOBAL_EPOCH_AT_STAGE8,
        "checkpoint_exists": checkpoint_exists(out_dir),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="preserved stage-5 checkpoint dir (state + manifest)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="NEW_DIR to write the stage-8-positioned checkpoint into",
    )
    p.add_argument(
        "--map-location",
        default="cpu",
        help="torch.load map_location (CPU only; MPS is owned by the live run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    facts = build_stage8_jump_checkpoint(
        args.src_dir, args.out_dir, map_location=args.map_location
    )
    print(json.dumps(facts, indent=2, sort_keys=True))
    man = facts["new_manifest"]
    assert man["stage_index"] == _STAGE8_INDEX, man
    assert man["epoch_in_stage"] == 0, man
    assert man["stage_name"] == _STAGE8_NAME, man
    assert man["has_muon"] is False, man  # optimizers cleared -> no muon yet
    assert facts["checkpoint_exists"] is True
    print(
        f"\nOK: stage-8-positioned checkpoint written to {facts['out_dir']} "
        f"(stage_index=7/epoch_in_stage=0, has_muon=False, optimizers cleared). "
        f"On resume the driver runs the genuine stage-8 Muon finetune from the "
        f"carried stage-5 weights; _global_epoch starts at "
        f"{facts['expected_global_epoch_at_stage8']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
