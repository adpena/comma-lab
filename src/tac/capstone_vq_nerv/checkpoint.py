# SPDX-License-Identifier: MIT
"""Stage+epoch checkpoint/resume for the capstone curriculum trainer.

The capstone n600 PR95-curriculum run is a multi-day local job (per the
2026-06-11 throughput profile, ~14-20 min/epoch at 600 pairs). The CLAUDE.md
"Durable detached daemons" + "LONG RESUMABLE SATURATION SWEEPS" non-negotiables
require that a death (SIGURG, OOM, host reboot) costs at most one in-flight
checkpoint — the missing piece that lost the earlier 2x2 capacity-ablation arms.

This module captures the COMPLETE trainer state so a resumed run continues the
EXACT same descent trajectory:

* the bundle param tree (decoder + per-frame FiLM + latents), via
  ``mx.tree_flatten(bundle.parameters())``;
* the VQ EMA buffers (``quantizer._codebook`` / ``_ema_cluster_size`` /
  ``_ema_w``) — plain arrays, NOT in ``parameters()``, so captured explicitly;
* the weight-EMA shadow (the inference/export bytes come from it — the EMA
  non-negotiable) + its update counter (drives the warmup decay);
* the PR95 optimizer state (``step`` / ``muon_buffers`` / ``adamw_m`` /
  ``adamw_v``) so Muon momentum + AdamW bias-correction resume mid-stage;
* the curriculum position (stage index + epoch within stage) + ``_mech_step``
  (drives the QAT / sigma-noise / C1a RNG keys deterministically).

Arrays are written via ``mx.save_safetensors`` (the canonical MLX array store,
already used by ``pr95_hnerv_mlx_long_training``); scalars + the manifest go in a
JSON sidecar. The write is atomic (tmp + ``os.replace``) so a crash mid-write
never corrupts the live checkpoint (CLAUDE.md fcntl/atomic-write discipline).

Authority: this is infrastructure, not a score axis. $0, local, no MPS.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - import guard
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except Exception:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    tree_flatten = tree_unflatten = None  # type: ignore[assignment]

CHECKPOINT_VERSION = 1
_ARRAYS_NAME = "capstone_checkpoint_arrays.safetensors"
_MANIFEST_NAME = "capstone_checkpoint_manifest.json"
_DONE_MARKER = "capstone_run.DONE"


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover
        raise RuntimeError("tac.capstone_vq_nerv.checkpoint requires mlx.core.")


@dataclass(frozen=True)
class CheckpointPosition:
    """Where in the curriculum a resumed run picks up.

    ``stage_index`` is the 0-based index into the curriculum stage tuple.
    ``epoch_in_stage`` is the number of epochs ALREADY COMPLETED in that stage
    (so the resumed stage runs ``range(epoch_in_stage, spec.epochs)``). When
    ``stage_index == len(stages)`` the curriculum is complete.
    """

    stage_index: int
    epoch_in_stage: int


def _flat_arrays_with_prefix(prefix: str, tree: Any) -> dict[str, Any]:
    """Flatten an MLX param tree into a flat ``{prefix.key: array}`` dict."""
    return {f"{prefix}.{k}": v for k, v in tree_flatten(tree)}


def _unflatten_prefixed(prefix: str, flat: dict[str, Any]) -> Any:
    """Inverse of :func:`_flat_arrays_with_prefix` for one prefix."""
    plen = len(prefix) + 1
    items = [(k[plen:], v) for k, v in flat.items() if k.startswith(prefix + ".")]
    return tree_unflatten(items)


def save_checkpoint(
    trainer: Any,
    out_dir: str | os.PathLike[str],
    position: CheckpointPosition,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a complete, atomically-replaced checkpoint of ``trainer`` state.

    Returns the checkpoint directory path. Overwrites the single live checkpoint
    (the resume point is always the latest); callers that want a history may
    pass distinct ``out_dir`` per checkpoint.
    """
    _require_mlx()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {}
    # 1. Bundle full param/buffer tree (decoder + FiLM + latents).
    arrays.update(_flat_arrays_with_prefix("bundle", trainer.bundle.parameters()))
    # 2. VQ EMA buffers (plain arrays, not in parameters()).
    q = getattr(trainer.bundle, "quantizer", None)
    has_vq = q is not None and getattr(q, "_codebook", None) is not None
    if has_vq:
        arrays["vq._codebook"] = q._codebook
        arrays["vq._ema_cluster_size"] = q._ema_cluster_size
        arrays["vq._ema_w"] = q._ema_w
    # 3. Weight-EMA shadow.
    for k, v in trainer._ema.shadow.items():
        arrays[f"ema_shadow.{k}"] = v
    # 4. Optimizer state arrays.
    opt = trainer.opt_state
    for k, v in opt.muon_buffers.items():
        arrays[f"opt.muon.{k}"] = v
    for k, v in opt.adamw_m.items():
        arrays[f"opt.adamw_m.{k}"] = v
    for k, v in opt.adamw_v.items():
        arrays[f"opt.adamw_v.{k}"] = v

    # Materialize before writing (lazy arrays must be evaluated).
    mx.eval(list(arrays.values()))

    manifest: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "stage_index": int(position.stage_index),
        "epoch_in_stage": int(position.epoch_in_stage),
        "opt_step": int(opt.step),
        "mech_step": int(trainer._mech_step),
        "ema_num_updates": int(trainer._ema._num_updates),
        "ema_decay": float(trainer._ema.decay),
        "current_epoch": int(trainer._current_epoch),
        "cosine_base_lr": float(trainer._cosine_base_lr),
        "cosine_total_epochs": int(trainer._cosine_total_epochs),
        "n_pairs": int(trainer.n_pairs),
        "scorer_backend": str(trainer.cfg.scorer_backend),
        "has_vq": bool(has_vq),
        "muon_keys": sorted(opt.muon_buffers.keys()),
        "adamw_m_keys": sorted(opt.adamw_m.keys()),
        "adamw_v_keys": sorted(opt.adamw_v.keys()),
        "extra": extra or {},
    }

    # Atomic write: tmp files + os.replace (crash-mid-write safe). The tmp array
    # file MUST keep the ``.safetensors`` extension (mx.save_safetensors infers the
    # format from the suffix and rejects/renames otherwise), so the tmp marker is
    # an INFIX (``.tmp.safetensors``), not a trailing suffix.
    tmp_arrays = out / (_ARRAYS_NAME[: -len(".safetensors")] + ".tmp.safetensors")
    tmp_manifest = out / (_MANIFEST_NAME + ".tmp")
    mx.save_safetensors(str(tmp_arrays), arrays)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp_arrays, out / _ARRAYS_NAME)
    os.replace(tmp_manifest, out / _MANIFEST_NAME)
    return out


def checkpoint_exists(out_dir: str | os.PathLike[str]) -> bool:
    """True iff a complete (arrays + manifest) checkpoint is present in ``out_dir``."""
    out = Path(out_dir)
    return (out / _ARRAYS_NAME).exists() and (out / _MANIFEST_NAME).exists()


def read_manifest(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the checkpoint manifest JSON (raises if absent)."""
    return json.loads((Path(out_dir) / _MANIFEST_NAME).read_text())


def load_checkpoint(trainer: Any, out_dir: str | os.PathLike[str]) -> CheckpointPosition:
    """Restore ``trainer`` IN-PLACE from a checkpoint; return the resume position.

    The trainer must already be constructed with the SAME architecture (base_ch,
    carrier, tie_depth, n_pairs) — only the array contents are restored. Raises
    if the manifest's ``n_pairs`` disagrees (a different basis cannot resume).
    """
    _require_mlx()
    out = Path(out_dir)
    manifest = read_manifest(out)
    if int(manifest["n_pairs"]) != int(trainer.n_pairs):
        raise ValueError(
            f"checkpoint n_pairs={manifest['n_pairs']} != trainer n_pairs={trainer.n_pairs}; "
            "cannot resume a different basis"
        )
    flat = dict(mx.load(str(out / _ARRAYS_NAME)))

    # 1. Bundle params/buffers.
    trainer.bundle.update(_unflatten_prefixed("bundle", flat))
    # 2. VQ EMA buffers.
    if manifest.get("has_vq"):
        q = trainer.bundle.quantizer
        q._codebook = flat["vq._codebook"]
        q._ema_cluster_size = flat["vq._ema_cluster_size"]
        q._ema_w = flat["vq._ema_w"]
    # 3. Weight-EMA shadow.
    shadow = {
        k[len("ema_shadow.") :]: v for k, v in flat.items() if k.startswith("ema_shadow.")
    }
    trainer._ema.shadow = shadow
    trainer._ema._num_updates = int(manifest["ema_num_updates"])
    trainer._ema.decay = float(manifest["ema_decay"])
    # 4. Optimizer state.
    from tac.local_acceleration.pr95_hnerv_mlx import Pr95MlxOptimizerState

    opt = Pr95MlxOptimizerState()
    opt.step = int(manifest["opt_step"])
    opt.muon_buffers = {
        k[len("opt.muon.") :]: v for k, v in flat.items() if k.startswith("opt.muon.")
    }
    opt.adamw_m = {
        k[len("opt.adamw_m.") :]: v for k, v in flat.items() if k.startswith("opt.adamw_m.")
    }
    opt.adamw_v = {
        k[len("opt.adamw_v.") :]: v for k, v in flat.items() if k.startswith("opt.adamw_v.")
    }
    trainer.opt_state = opt
    # 5. Scalar trainer state.
    trainer._mech_step = int(manifest["mech_step"])
    trainer._current_epoch = int(manifest["current_epoch"])
    trainer._cosine_base_lr = float(manifest["cosine_base_lr"])
    trainer._cosine_total_epochs = int(manifest["cosine_total_epochs"])

    # Materialize the restored arrays so the next step sees concrete values.
    mx.eval([v for _, v in tree_flatten(trainer.bundle.parameters())])
    return CheckpointPosition(
        stage_index=int(manifest["stage_index"]),
        epoch_in_stage=int(manifest["epoch_in_stage"]),
    )


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


__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointPosition",
    "checkpoint_exists",
    "is_done",
    "load_checkpoint",
    "read_manifest",
    "save_checkpoint",
    "write_done_marker",
]
