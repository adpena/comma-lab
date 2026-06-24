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
# Per-stage PRESERVED snapshot dir (the prune/fork-from-any-stage artifact, the
# operator's explicit ask). Lives UNDER out_dir, sibling to the rolling checkpoint;
# each stage gets its own ``stageNN_<name>/`` subdir so a fork/prune can start from
# any curriculum stage. NEVER overwrites the rolling resume checkpoint (a different
# filename family in a different subdir) and is NEVER auto-deleted (the preserve
# contract). Default-OFF in the driver → byte-identical when the operator does not
# opt in (the live basin is unaffected).
_STAGE_SNAPSHOTS_DIR = "stage_snapshots"
# The certify-or-block preservation manifest (the "Local Disk / SSD spill /
# auto-cleanup / provenance" non-negotiable): a machine-readable record of the
# durable run artifacts so cold-store/move decisions are lossless.
_PRESERVATION_MANIFEST_NAME = "preservation_manifest.json"


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

    blob = _build_blob(state)
    manifest = _build_manifest(state, position, extra=extra)

    # Atomic write: tmp + os.replace (crash-mid-write safe).
    tmp_state = out / (_STATE_NAME + ".tmp")
    tmp_manifest = out / (_MANIFEST_NAME + ".tmp")
    torch.save(blob, tmp_state)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_state, out / _STATE_NAME)
    os.replace(tmp_manifest, out / _MANIFEST_NAME)
    return out


def _build_blob(state: dict[str, Any]) -> dict[str, Any]:
    """Build the torch.save tensor/optimizer blob from driver ``state`` (the
    COMPLETE serializable state — decoder + latents + EMA shadow + optimizer +
    scheduler + RNG + controller state). Shared by ``save_checkpoint`` (the rolling
    resume point) and ``save_stage_snapshot`` (the preserved per-stage fork point)
    so the two write the IDENTICAL state — no drift between the resume artifact and
    the snapshot artifact."""
    # The big tensor/optimizer blob goes through torch.save.
    return {
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


def _build_manifest(
    state: dict[str, Any],
    position: TorchCheckpointPosition,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON manifest sidecar for ``state`` at ``position``. Shared by the
    rolling checkpoint and the per-stage snapshot so the resume guards (base_ch /
    latent_dim / taper / muon-floor / warmup) apply IDENTICALLY to a snapshot a fork
    resumes from."""
    return {
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
        # E#5 per-stage LR warmup shape. Persisted so a resume that CHANGES it fails closed
        # mid-stage (the saved LambdaLR step-count is tuned to this warmup shape). Backward-
        # compatible: old checkpoints lack the keys → the driver guard reads them as the cfg
        # value (pass), so a legacy resume is never spuriously blocked.
        "stage_lr_warmup_frac": (
            None if state.get("stage_lr_warmup_frac") is None
            else float(state["stage_lr_warmup_frac"])
        ),
        "stage_lr_warmup_start_ratio": (
            None if state.get("stage_lr_warmup_start_ratio") is None
            else float(state["stage_lr_warmup_start_ratio"])
        ),
        "stage_name": str(state.get("stage_name", "")),
        "ema_decay": float(state.get("ema_decay", 0.999)),
        "best_score": float(state.get("best_score", float("inf"))),
        "best_ep": int(state.get("best_ep", 0)),
        "best_stage": int(state.get("best_stage", -1)),
        "has_muon": bool(state.get("muon") is not None),
        "extra": extra or {},
    }


def stage_snapshot_dir(
    out_dir: str | os.PathLike[str], stage_index: int, stage_name: str
) -> Path:
    """The canonical preserved-snapshot dir for a completed stage:
    ``<out_dir>/stage_snapshots/stageNN_<sanitized_name>/``. Zero-padded index so a
    lexical sort matches curriculum order; the name is sanitized to a filesystem-safe
    slug (alnum + ``_``)."""
    slug = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(stage_name))
    return (
        Path(out_dir)
        / _STAGE_SNAPSHOTS_DIR
        / f"stage{int(stage_index):02d}_{slug}"
    )


def save_stage_snapshot(
    state: dict[str, Any],
    out_dir: str | os.PathLike[str],
    position: TorchCheckpointPosition,
    *,
    stage_index: int,
    stage_name: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a PRESERVED per-stage snapshot (the prune/fork-from-any-stage artifact —
    the operator's explicit checkpoint-preservation ask + the MUONJUMP precedent).

    Writes the IDENTICAL complete state blob + manifest ``save_checkpoint`` writes,
    but into a stage-specific subdir (:func:`stage_snapshot_dir`) that is NEVER
    overwritten by the rolling resume checkpoint and NEVER auto-deleted — so a fork
    or the capacity-RD prune-path can restore the EXACT decoder + latents + EMA
    shadow + optimizer state at the boundary of ANY completed curriculum stage. The
    write is ATOMIC (tmp + ``os.replace``) so a crash mid-snapshot never corrupts it,
    and a re-run that re-completes the same stage REWRITES the same dir idempotently
    (no orphan accumulation across resumes — the snapshot count is bounded by the
    stage count, not the run count).

    The snapshot manifest carries an ``is_stage_snapshot`` marker in ``extra`` so a
    loader can distinguish it from the rolling resume checkpoint. It is loadable via
    the SAME :func:`load_checkpoint` (the layout is identical), so the prune-path /
    fork resumes a snapshot exactly as it would the rolling checkpoint.
    """
    snap_dir = stage_snapshot_dir(out_dir, stage_index, stage_name)
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_extra = dict(extra or {})
    snap_extra["is_stage_snapshot"] = True
    snap_extra["snapshot_stage_index"] = int(stage_index)
    snap_extra["snapshot_stage_name"] = str(stage_name)
    blob = _build_blob(state)
    manifest = _build_manifest(state, position, extra=snap_extra)
    tmp_state = snap_dir / (_STATE_NAME + ".tmp")
    tmp_manifest = snap_dir / (_MANIFEST_NAME + ".tmp")
    torch.save(blob, tmp_state)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_state, snap_dir / _STATE_NAME)
    os.replace(tmp_manifest, snap_dir / _MANIFEST_NAME)
    return snap_dir


def list_stage_snapshots(out_dir: str | os.PathLike[str]) -> list[Path]:
    """Return the preserved per-stage snapshot dirs (each holding a complete
    state+manifest), sorted in curriculum order. Empty when none were preserved
    (the default-OFF path)."""
    root = Path(out_dir) / _STAGE_SNAPSHOTS_DIR
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and checkpoint_exists(d)
    )


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


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Stream-hash a (possibly large) file for the preservation manifest."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_preservation_manifest(
    out_dir: str | os.PathLike[str],
    *,
    config: dict[str, Any] | None = None,
    rebuild_command: str | None = None,
    hash_files: bool = True,
) -> Path:
    """Write the certify-or-block preservation manifest (the "Local Disk / SSD spill /
    auto-cleanup / provenance" non-negotiable).

    Enumerates the run's durable bulk artifacts (the rolling checkpoint, the ``best/``
    EMA shadow + archive, every preserved stage snapshot) with per-file bytes + SHA-256
    + the rebuild command + config — so a future cold-store/move decision is LOSSLESS
    (the artifact can be deleted-after-move OR rebuilt-from-config, never silently
    lost). Re-callable: it overwrites the single manifest with the current artifact
    set (atomic tmp + os.replace). ``hash_files=False`` skips the (potentially slow on
    multi-GB checkpoints) SHA pass for a fast metadata-only manifest.

    This does NOT delete or move anything — it is the RECORD that makes a later
    lossless cleanup possible (certify-or-block: if this manifest is missing, the
    bytes must be kept).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    def _record(p: Path, kind: str) -> None:
        if not p.exists():
            return
        rec: dict[str, Any] = {
            "path": str(p),
            "rel_path": str(p.relative_to(out)) if out in p.parents or p == out else str(p),
            "bytes": int(p.stat().st_size),
            "kind": kind,
        }
        if hash_files:
            rec["sha256"] = _sha256_file(p)
        artifacts.append(rec)

    # Rolling resume checkpoint.
    _record(out / _STATE_NAME, "rolling_resume_state")
    _record(out / _MANIFEST_NAME, "rolling_resume_manifest")
    # Best (the prune-ready EMA shadow + archive).
    best = out / "best"
    for name, kind in (
        ("best_ema_decoder.pt", "best_ema_decoder"),
        ("best_ema_latents.pt", "best_ema_latents"),
        ("best_archive.bin", "best_archive"),
        ("best_meta.json", "best_meta"),
    ):
        _record(best / name, kind)
    # Preserved per-stage snapshots.
    for snap in list_stage_snapshots(out):
        _record(snap / _STATE_NAME, f"stage_snapshot_state:{snap.name}")
        _record(snap / _MANIFEST_NAME, f"stage_snapshot_manifest:{snap.name}")

    manifest = {
        "version": CHECKPOINT_VERSION,
        "out_dir": str(out),
        "rebuild_command": rebuild_command,
        "config": config or {},
        "total_bytes": sum(a["bytes"] for a in artifacts),
        "artifacts": artifacts,
        "authority": "[contest-CPU advisory] NON-PROMOTABLE — infrastructure custody record",
        "rebuildable": (
            "rebuildable from rebuild_command (the deterministic seeded curriculum); "
            "cold-store/move is LOSSLESS once this manifest exists (certify-or-block)."
        ),
    }
    tmp = out / (_PRESERVATION_MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, out / _PRESERVATION_MANIFEST_NAME)
    return out / _PRESERVATION_MANIFEST_NAME


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
    "list_stage_snapshots",
    "load_checkpoint",
    "read_manifest",
    "restore_rng",
    "save_checkpoint",
    "save_stage_snapshot",
    "stage_snapshot_dir",
    "write_done_marker",
    "write_preservation_manifest",
]
