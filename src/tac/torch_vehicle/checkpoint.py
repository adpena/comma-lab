# SPDX-License-Identifier: MIT
"""Stage+epoch checkpoint/resume for the P2 torch-vehicle curriculum driver.

Torch parity with the MLX ``tac.capstone_vq_nerv.checkpoint``: the $100 n600
PR95-curriculum run is a multi-day Modal/CUDA job, and the "Durable detached
daemons" + "LONG RESUMABLE SATURATION SWEEPS" non-negotiables require that a
death (SIGURG, OOM, host reboot, Modal preemption) costs at most ONE in-flight
checkpoint — not the whole run.

This module captures the COMPLETE driver state so a resumed run continues the
EXACT same descent trajectory:

* the decoder ``state_dict`` (the live training weights) + the per-pair
  ``latents`` tensor;
* the **EMA shadow** decoder ``state_dict`` + EMA latents — the inference/export
  bytes come from the shadow (the EMA non-negotiable), so it must survive a
  death bit-for-bit;
* the AdamW optimizer ``state_dict`` (per-param ``exp_avg`` / ``exp_avg_sq`` /
  ``step``) so bias correction resumes mid-stage;
* the Muon optimizer ``state_dict`` (the ``momentum_buffer`` per param) so the
  Newton-Schulz orthogonalized momentum resumes mid-stage;
* the LR-scheduler ``state_dict``s (the cosine ``last_epoch``);
* the curriculum position (stage index + epoch within stage) + the RNG states
  (torch + numpy) so the per-epoch ``randperm`` batch order is deterministic
  on resume;
* the best-so-far tracking (best score / best epoch / best stage).

State is written via ``torch.save`` to a single ``.pt`` blob; the manifest goes
in a JSON sidecar. The write is ATOMIC (tmp + ``os.replace``) so a crash
mid-write never corrupts the live checkpoint.

Authority: infrastructure, not a score axis. $0, local/Modal, no MPS.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_VERSION = 1
_STATE_NAME = "torch_vehicle_checkpoint_state.pt"
_MANIFEST_NAME = "torch_vehicle_checkpoint_manifest.json"
_DONE_MARKER = "torch_vehicle_run.DONE"


@dataclass(frozen=True)
class TorchCheckpointPosition:
    """Where in the curriculum a resumed run picks up.

    ``stage_index`` is the 0-based index into the curriculum stage tuple.
    ``epoch_in_stage`` is the number of epochs ALREADY COMPLETED in that stage
    (so the resumed stage runs ``range(epoch_in_stage, spec.epochs)``). When
    ``stage_index == len(stages)`` the curriculum is complete.
    """

    stage_index: int
    epoch_in_stage: int


def save_checkpoint(
    state: dict[str, Any],
    out_dir: str | os.PathLike[str],
    position: TorchCheckpointPosition,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a complete, atomically-replaced checkpoint of driver ``state``.

    ``state`` MUST contain (callers build it via
    :func:`tac.torch_vehicle.driver.capture_driver_state`):

    * ``decoder``: decoder ``state_dict``
    * ``latents``: per-pair latent tensor (CPU)
    * ``ema_decoder``: EMA shadow decoder ``state_dict``
    * ``ema_latents``: EMA latents tensor (CPU)
    * ``adamw``: AdamW ``state_dict`` (or ``None``)
    * ``muon``: Muon ``state_dict`` (or ``None`` for non-Muon stages)
    * ``adamw_sched`` / ``muon_sched``: LR-scheduler ``state_dict``s (or ``None``)
    * ``torch_rng`` / ``numpy_rng``: RNG states
    * scalar manifest fields: ``base_channels`` / ``latent_dim`` / ``n_pairs``
      / ``stage_name`` / ``ema_decay`` / ``best_score`` / ``best_ep`` /
      ``best_stage``

    Returns the checkpoint directory path (overwrites the single live
    checkpoint — the resume point is always the latest).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The big tensor/optimizer blob goes through torch.save.
    blob = {
        "decoder": state["decoder"],
        "latents": state["latents"],
        "ema_decoder": state["ema_decoder"],
        "ema_latents": state["ema_latents"],
        "adamw": state.get("adamw"),
        "muon": state.get("muon"),
        "adamw_sched": state.get("adamw_sched"),
        "muon_sched": state.get("muon_sched"),
        "torch_rng": state.get("torch_rng"),
        "numpy_rng": state.get("numpy_rng"),
        # Lever 4 (score-aware QAT) per-tensor sensitivity EMA. A plain
        # ``dict[str, float]`` (or ``None``/absent for a default / pre-Lever-4
        # checkpoint). MUST survive a death so a Lever-4-ON resume continues the
        # SAME quant-grid trajectory instead of resetting the EMA to empty (which
        # would fall back to uniform-127 for the post-resume steps — a resume-
        # fidelity drift, the sec/epoch/eval-row reset class). Absent on legacy
        # checkpoints → the loader yields no key → the driver rebuilds an empty
        # EMA, exactly today's behavior (backward-compatible).
        "tensor_sensitivity_ema": state.get("tensor_sensitivity_ema"),
        # EMA-warmup step counter (Lever: ema_warmup). MUST survive a death so a resumed
        # warmup run CONTINUES its decay schedule rather than snapping back to decay=0.1
        # at t=0 (a one-step shadow jolt). 0 on the default (ema_warmup off) path → the
        # loader yields 0 and the restore is a no-op (backward-compatible; old checkpoints
        # lack the key → merged.get('ema_step', 0) == 0 == today's behavior).
        "ema_step": int(state.get("ema_step", 0)),
        # APGC (Adaptive Pose-Gradient Controller) state. MUST survive a death so a
        # resumed adaptive run CONTINUES the same cadence rather than re-establishing
        # the floor + recomputing pose every epoch for the first k_max post-resume epochs
        # (a spurious cost spike + a floor re-seed off a possibly-drifted sample). All
        # default on the non-adaptive path (``pose_floor`` None, empty hist, epoch 0) →
        # the loader yields the defaults and the restore is a no-op (backward-compatible;
        # old checkpoints lack these keys → the driver's merged.get(...) yields the same
        # defaults == today's behavior).
        "pose_floor": state.get("pose_floor"),
        "pose_mse_hist": list(state.get("pose_mse_hist", [])),
        "last_pose_epoch": int(state.get("last_pose_epoch", -1)),
        # Lever A (equimarginal pose-weight controller) state: the ratio EMA + accumulated w_pose fraction +
        # step count. MUST survive a death so a resumed run CONTINUES the same w_pose trajectory rather than
        # snapping back to w_pose0 (a pose/seg balance jolt). None on the default path (controller off) → the
        # loader yields None and the restore is a no-op (backward-compatible; old checkpoints lack the key →
        # merged.get('equimarginal_ctrl') == None == today's behavior).
        "equimarginal_ctrl": state.get("equimarginal_ctrl"),
    }

    manifest: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "stage_index": int(position.stage_index),
        "epoch_in_stage": int(position.epoch_in_stage),
        "base_channels": int(state["base_channels"]),
        "latent_dim": int(state["latent_dim"]),
        "n_pairs": int(state["n_pairs"]),
        # Configurable-taper schedule (None on the vendored-decoder path). Persisted so a
        # resume into an out_dir trained with a DIFFERENT taper fails closed EXPLICITLY
        # (not only via an accidental state_dict shape mismatch). Backward-compatible:
        # old checkpoints lack the key → read as None == the vendored taper.
        "taper_channels": state.get("taper_channels"),
        # Stage-8 Muon own-floor flag. Persisted so a resume that TOGGLES it fails closed
        # (the saved Muon LambdaLR step-count is tuned to this flag's eta_min). Backward-
        # compatible: old checkpoints lack the key → read as False == the vendored shared-floor.
        "muon_lr_floor_fix": bool(state.get("muon_lr_floor_fix", False)),
        "stage_name": str(state.get("stage_name", "")),
        "ema_decay": float(state.get("ema_decay", 0.999)),
        "best_score": float(state.get("best_score", float("inf"))),
        "best_ep": int(state.get("best_ep", 0)),
        "best_stage": int(state.get("best_stage", -1)),
        "has_muon": bool(state.get("muon") is not None),
        "extra": extra or {},
    }

    # Atomic write: tmp + os.replace (crash-mid-write safe).
    tmp_state = out / (_STATE_NAME + ".tmp")
    tmp_manifest = out / (_MANIFEST_NAME + ".tmp")
    torch.save(blob, tmp_state)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_state, out / _STATE_NAME)
    os.replace(tmp_manifest, out / _MANIFEST_NAME)
    return out


def checkpoint_exists(out_dir: str | os.PathLike[str]) -> bool:
    """True iff a complete (state + manifest) checkpoint is present."""
    out = Path(out_dir)
    return (out / _STATE_NAME).exists() and (out / _MANIFEST_NAME).exists()


def read_manifest(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the checkpoint manifest JSON (raises if absent)."""
    return json.loads((Path(out_dir) / _MANIFEST_NAME).read_text())


def load_checkpoint(out_dir: str | os.PathLike[str], *, map_location: str = "cpu") -> dict[str, Any]:
    """Load the checkpoint blob + manifest. Returns a merged state dict.

    The caller (driver) reconstructs the architecture FIRST (same base_channels
    / latent_dim / n_pairs as the manifest), then restores the arrays. Raises
    on a basis mismatch is the CALLER's responsibility; this function just reads.
    Returns a dict with the blob keys + ``position`` + ``manifest``.
    """
    out = Path(out_dir)
    manifest = read_manifest(out)
    blob = torch.load(out / _STATE_NAME, map_location=map_location, weights_only=False)
    merged = dict(blob)
    merged["manifest"] = manifest
    merged["position"] = TorchCheckpointPosition(
        stage_index=int(manifest["stage_index"]),
        epoch_in_stage=int(manifest["epoch_in_stage"]),
    )
    return merged


def write_done_marker(out_dir: str | os.PathLike[str], summary: dict[str, Any]) -> Path:
    """Write the run-complete marker (the marker-on-exit non-negotiable)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / _DONE_MARKER
    marker.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return marker


def is_done(out_dir: str | os.PathLike[str]) -> bool:
    """True iff the run-complete marker is present."""
    return (Path(out_dir) / _DONE_MARKER).exists()


def restore_rng(merged: dict[str, Any]) -> None:
    """Restore torch + numpy RNG states from a loaded checkpoint (best-effort)."""
    trng = merged.get("torch_rng")
    if trng is not None:
        # torch.get_rng_state returns a ByteTensor; ensure CPU uint8.
        torch.set_rng_state(trng.cpu() if hasattr(trng, "cpu") else trng)
    nrng = merged.get("numpy_rng")
    if nrng is not None:
        np.random.set_state(nrng)


__all__ = [
    "CHECKPOINT_VERSION",
    "TorchCheckpointPosition",
    "checkpoint_exists",
    "is_done",
    "load_checkpoint",
    "read_manifest",
    "restore_rng",
    "save_checkpoint",
    "write_done_marker",
]
