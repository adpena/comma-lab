# SPDX-License-Identifier: MIT
"""Long-training adapter for the HPRC compact receiver.

This is the first executable HPRC train/export control arm.  It deliberately
keeps the receiver numpy-portable: training optimizes the compact receiver's
RDO gains over the archive-contained latent/residual components, while archive
export still goes through the HPRC byte-closed runtime bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import tac.substrates.hprc.archive_candidate as hprc_archive_candidate
from tac.framework_agnostic import is_mlx_runtime_available, require_mlx_core
from tac.substrates.hprc.archive import HprcPacketConfig, HprcSectionKind, pack_hprc_packet
from tac.substrates.hprc.learned_receiver import (
    COMPACT_NUMPY_DECODER_FAMILY_ID,
    COMPACT_RECEIVER_MODE,
    COMPACT_RGB_COLOR_TRANSFORM_ID,
    build_compact_receiver_packet_from_lowres_frames,
    decode_compact_receiver_packet,
    pack_compact_decoder,
    pack_compact_latents,
    pack_compact_receiver_state,
    pack_compact_residual,
    pack_compact_residual_protected,
    pack_compact_selectors,
)

HPRC_LONG_TRAINING_SUBSTRATE_ID = "hprc_compact_receiver"
HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA = "hprc_compact_receiver_training_export.v1"
HPRC_NATIVE_RATE_AWARE_TRAINING_SCHEMA = "hprc_native_rate_aware_training.v1"
HPRC_MLX_TRAIN_NUMPY_PORTABLE_SCHEMA = "hprc_mlx_trained_numpy_portable_export.v1"
HPRC_TRAINING_BACKENDS = frozenset({"auto", "numpy", "mlx"})


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ndarray_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return _sha256_bytes(arr.tobytes(order="C"))


def _state_dict_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            summary[key] = {
                "kind": "ndarray",
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
                "bytes": int(arr.nbytes),
                "sha256": _ndarray_sha256(arr),
            }
        else:
            summary[key] = value
    return summary


def _nearest_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    y_idx = (np.arange(height, dtype=np.int64) * src_h // height).clip(0, src_h - 1)
    x_idx = (np.arange(width, dtype=np.int64) * src_w // width).clip(0, src_w - 1)
    return frame[y_idx[:, None], x_idx[None, :], :]


def _downsample_sum_nearest_inverse(frame: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Sum full-resolution pixels into the residual cells used by ``_nearest_resize``."""

    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("frame must be HxWxC")
    src_h, src_w, channels = arr.shape
    y_idx = (np.arange(src_h, dtype=np.int64) * int(grid_h) // src_h).clip(0, int(grid_h) - 1)
    x_idx = (np.arange(src_w, dtype=np.int64) * int(grid_w) // src_w).clip(0, int(grid_w) - 1)
    cell = (y_idx[:, None] * int(grid_w) + x_idx[None, :]).reshape(-1)
    out = np.zeros((int(grid_h) * int(grid_w), int(channels)), dtype=np.float64)
    np.add.at(out, cell, arr.reshape((-1, int(channels))))
    return out.reshape((int(grid_h), int(grid_w), int(channels)))


def _nearest_resize_batch_mlx(mx: Any, frames: Any, height: int, width: int) -> Any:
    """Nearest-neighbor resize for NHWC MLX arrays.

    Training can use MLX/Metal, but the receiver runtime remains numpy-only.
    Keeping this helper local to the training adapter avoids pulling MLX into
    archive decode paths.
    """

    src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
    y_idx = (np.arange(int(height), dtype=np.int32) * src_h // int(height)).clip(0, src_h - 1)
    x_idx = (np.arange(int(width), dtype=np.int32) * src_w // int(width)).clip(0, src_w - 1)
    return frames[:, mx.array(y_idx), :, :][:, :, mx.array(x_idx), :]


def _training_backend_requested(value: str) -> str:
    backend = str(value or "auto").strip().lower()
    if backend not in HPRC_TRAINING_BACKENDS:
        raise ValueError(
            f"HPRC training backend must be one of {sorted(HPRC_TRAINING_BACKENDS)}, got {value!r}"
        )
    return backend


def _select_training_backend(value: str) -> str:
    requested = _training_backend_requested(value)
    if requested == "auto":
        return "mlx" if is_mlx_runtime_available() else "numpy"
    if requested == "mlx":
        require_mlx_core()
    return requested


def _mx_scalar(value: Any) -> float:
    return float(np.array(value).reshape(()))


def _normalize_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim == 5:
        pairs, gop, height, width, channels = arr.shape
        arr = arr.reshape((pairs * gop, height, width, channels))
    if arr.ndim != 4:
        raise ValueError("HPRC training frames must be FxHxWxC or Px2xHxWxC")
    if arr.shape[3] != 3:
        raise ValueError("HPRC compact receiver training requires RGB frames")
    if arr.shape[0] <= 0:
        raise ValueError("HPRC compact receiver training requires at least one frame")
    if arr.shape[0] > 0xFFFF:
        raise ValueError("HPRC compact receiver frame count exceeds u16 packet limit")
    if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.5:
        arr = arr * 255.0
    return np.asarray(arr, dtype=np.float32)


def _normalize_residual_protection(
    value: np.ndarray | None,
    *,
    residual_shape: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Normalize scorer protection weights to residual-token shape.

    Protection is in ``[0, 1]`` where ``1`` means protect from rate pressure and
    ``0`` means safest to shrink.  The train-time rate pressure is therefore
    ``1 - protection``.
    """

    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    frames, grid_h, grid_w, channels = residual_shape
    allowed = {
        (frames, grid_h, grid_w, channels),
        (frames, grid_h, grid_w),
        (frames, 1, 1, 1),
        (frames, 1, 1),
        (1, grid_h, grid_w, channels),
        (1, grid_h, grid_w),
    }
    if tuple(arr.shape) not in allowed:
        raise ValueError(
            "residual protection shape must broadcast to residual tokens; "
            f"got {arr.shape}, expected one of {sorted(allowed)}"
        )
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    if arr.ndim != 4:
        raise ValueError("residual protection must be 3D or 4D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("residual protection contains non-finite values")
    return np.broadcast_to(np.clip(arr, 0.0, 1.0), residual_shape).astype(np.float32, copy=True)


@dataclass(frozen=True)
class HprcGainBounds:
    """Hard projection bounds for compact receiver RDO gains."""

    latent: tuple[float, float] = (-4.0, 4.0)
    residual: tuple[float, float] = (-4.0, 4.0)
    receiver_state: tuple[float, float] = (-4.0, 4.0)
    protected_residual: tuple[float, float] = (-4.0, 4.0)

    def clamp(self, *, latent: float, residual: float, receiver_state: float) -> tuple[float, float, float]:
        return (
            float(np.clip(latent, self.latent[0], self.latent[1])),
            float(np.clip(residual, self.residual[0], self.residual[1])),
            float(np.clip(receiver_state, self.receiver_state[0], self.receiver_state[1])),
        )

    def clamp_protected_residual(self, value: float) -> float:
        return float(np.clip(float(value), self.protected_residual[0], self.protected_residual[1]))


class HprcCompactReceiverTrainingModel:
    """Small trainable state for a compact HPRC receiver.

    The heavy tensor fields are fixed archive components.  The trainable state is
    the RDO gain tuple consumed by the decode-only receiver runtime.
    """

    def __init__(
        self,
        frames: np.ndarray,
        *,
        basis_count: int = 3,
        residual_grid_h: int = 24,
        residual_grid_w: int = 32,
        protected_residual_grid_h: int | None = None,
        protected_residual_grid_w: int | None = None,
        protected_residual_mask: np.ndarray | None = None,
        source_manifest: Mapping[str, Any] | None = None,
        initial_latent_gain: float = 1.0,
        initial_residual_gain: float = 1.0,
        initial_protected_residual_gain: float = 1.0,
        initial_receiver_state_gain: float = 0.25,
        gain_bounds: HprcGainBounds | None = None,
    ) -> None:
        self.target_frames = _normalize_frames(frames)
        self.source_manifest = dict(source_manifest or {})
        self.gain_bounds = gain_bounds or HprcGainBounds()
        packet_bytes = build_compact_receiver_packet_from_lowres_frames(
            self.target_frames,
            basis_count=int(basis_count),
            residual_grid_h=int(residual_grid_h),
            residual_grid_w=int(residual_grid_w),
            protected_residual_grid_h=protected_residual_grid_h,
            protected_residual_grid_w=protected_residual_grid_w,
            protected_residual_mask=protected_residual_mask,
            source_manifest={
                **self.source_manifest,
                "training_adapter": HPRC_LONG_TRAINING_SUBSTRATE_ID,
            },
        )
        from tac.substrates.hprc.archive import parse_hprc_packet

        compact = decode_compact_receiver_packet(parse_hprc_packet(packet_bytes))
        self.packet_config = compact.packet.config
        self.mean = np.array(compact.decoder.mean, dtype=np.uint8, copy=True)
        self.basis = (
            compact.decoder.basis_q.astype(np.float32) * float(compact.decoder.basis_scale)
        )
        self.latents = (
            compact.latents.q.astype(np.float32) * float(compact.latents.scale)
        )
        self.selectors = compact.selectors.values.astype(np.float32) / 255.0
        self.residual = (
            compact.residual.q.astype(np.float32) * float(compact.residual.scale)
        )
        self.protected_residual = (
            None
            if compact.residual.protected_q is None
            else compact.residual.protected_q.astype(np.float32)
            * float(compact.residual.protected_scale)
        )
        self.receiver_state = (
            compact.receiver_state.q.astype(np.float32)
            * float(compact.receiver_state.scale)
        )
        self.latent_gain, self.residual_gain, self.receiver_state_gain = self.gain_bounds.clamp(
            latent=float(initial_latent_gain),
            residual=float(initial_residual_gain),
            receiver_state=float(initial_receiver_state_gain),
        )
        self.protected_residual_gain = self.gain_bounds.clamp_protected_residual(
            float(initial_protected_residual_gain)
        )
        self.training_backend_lineage: dict[str, Any] = {
            "schema": HPRC_MLX_TRAIN_NUMPY_PORTABLE_SCHEMA,
            "requested_training_backend": "numpy",
            "effective_training_backend": "numpy",
            "portable_runtime": "numpy",
            "contest_runtime_requires_mlx": False,
            "contest_runtime_requires_torch": False,
        }
        self.train_steps = 0

    @property
    def frame_count(self) -> int:
        return int(self.target_frames.shape[0])

    @property
    def pair_count(self) -> int:
        return math.ceil(self.frame_count / max(int(self.packet_config.gop_size), 1))

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "latent_gain": [float(self.latent_gain)],
            "residual_gain": [float(self.residual_gain)],
            "protected_residual_gain": [float(self.protected_residual_gain)],
            "receiver_state_gain": [float(self.receiver_state_gain)],
            "residual": self.residual.astype(np.float32, copy=True),
        }
        if self.protected_residual is not None:
            state["protected_residual"] = self.protected_residual.astype(np.float32, copy=True)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if isinstance(state.get("state_dict"), Mapping):
            wrapper = state
            state = wrapper["state_dict"]
            self.train_steps = int(wrapper.get("train_steps", self.train_steps))
        self.latent_gain = _state_scalar(state, "latent_gain", self.latent_gain)
        self.residual_gain = _state_scalar(state, "residual_gain", self.residual_gain)
        self.protected_residual_gain = _state_scalar(
            state,
            "protected_residual_gain",
            self.protected_residual_gain,
        )
        self.receiver_state_gain = _state_scalar(
            state, "receiver_state_gain", self.receiver_state_gain
        )
        self.latent_gain, self.residual_gain, self.receiver_state_gain = self.gain_bounds.clamp(
            latent=self.latent_gain,
            residual=self.residual_gain,
            receiver_state=self.receiver_state_gain,
        )
        self.protected_residual_gain = self.gain_bounds.clamp_protected_residual(
            self.protected_residual_gain
        )
        if "residual" in state:
            residual = np.asarray(state["residual"], dtype=np.float32)
            if tuple(residual.shape) != tuple(self.residual.shape):
                raise ValueError(
                    f"residual checkpoint shape {residual.shape} != model shape {self.residual.shape}"
                )
            self.residual = residual.copy()
        if "protected_residual" in state:
            if self.protected_residual is None:
                raise ValueError("checkpoint contains protected_residual but model has no pathway")
            protected = np.asarray(state["protected_residual"], dtype=np.float32)
            if tuple(protected.shape) != tuple(self.protected_residual.shape):
                raise ValueError(
                    "protected_residual checkpoint shape "
                    f"{protected.shape} != model shape {self.protected_residual.shape}"
                )
            self.protected_residual = protected.copy()

    def render_continuous(
        self,
        frame_index: int,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        frame = self.mean.astype(np.float32).copy()
        latent_component = np.tensordot(
            self.latents[frame_index],
            self.basis,
            axes=(0, 0),
        )
        residual_component = (
            float(self.selectors[frame_index])
            * _nearest_resize(
                self.residual[frame_index],
                int(self.packet_config.height),
                int(self.packet_config.width),
            )
        )
        protected_component = np.zeros_like(residual_component, dtype=np.float32)
        if self.protected_residual is not None:
            protected_component = (
                float(self.selectors[frame_index])
                * _nearest_resize(
                    self.protected_residual[frame_index],
                    int(self.packet_config.height),
                    int(self.packet_config.width),
                )
            )
            residual_component = residual_component + (
                float(self.protected_residual_gain) * protected_component
            )
        pair_index = min(
            self.receiver_state.shape[0] - 1,
            frame_index // max(int(self.packet_config.gop_size), 1),
        )
        state_component = np.zeros_like(frame, dtype=np.float32)
        state = self.receiver_state[pair_index]
        if state.shape[0] >= 3:
            state_component += state[:3].reshape((1, 1, 3))
        frame = (
            frame
            + self.latent_gain * latent_component
            + self.residual_gain * residual_component
            + self.receiver_state_gain * state_component
        )
        protected_gain_component = float(self.residual_gain) * protected_component
        return frame, (
            latent_component,
            residual_component,
            state_component,
            protected_gain_component,
        )

    def packet_bytes(self) -> bytes:
        rdo_plan = {
            "schema": "hprc_compact_receiver_rdo_plan.v1",
            "decoder_mode": COMPACT_RECEIVER_MODE,
            "training_adapter": HPRC_LONG_TRAINING_SUBSTRATE_ID,
            "latent_gain": float(self.latent_gain),
            "residual_gain": float(self.residual_gain),
            "protected_residual_gain": float(self.protected_residual_gain),
            "receiver_state_gain": float(self.receiver_state_gain),
            "basis_count": int(self.basis.shape[0]),
            "residual_grid_h": int(self.residual.shape[1]),
            "residual_grid_w": int(self.residual.shape[2]),
            "protected_residual_pathway": {
                "enabled": self.protected_residual is not None,
                "grid_h": (
                    0
                    if self.protected_residual is None
                    else int(self.protected_residual.shape[1])
                ),
                "grid_w": (
                    0
                    if self.protected_residual is None
                    else int(self.protected_residual.shape[2])
                ),
            },
            "output_resize": "bilinear",
            "output_resize_alignment": "bilinear_align_corners_false",
            "train_steps": int(self.train_steps),
            "training_backend": dict(self.training_backend_lineage),
            "score_claim": False,
            "promotion_eligible": False,
        }
        manifest = {
            "schema": "hprc_compact_receiver_manifest.v1",
            "hprc_receiver_mode": COMPACT_RECEIVER_MODE,
            "candidate_kind": "compact_numpy_receiver_with_trained_rdo_gains",
            "trained_renderer_export_ready": True,
            "z8_scorer_weighted_residual_sidecar_ready": False,
            "protected_highres_pose_pathway_ready": self.protected_residual is not None,
            "mamba_dreamer_stack_ready": False,
            "exact_cpu_cuda_authority_ready": False,
            "training_backend": dict(self.training_backend_lineage),
            "portable_runtime": "numpy",
            "training_adapter": HPRC_LONG_TRAINING_SUBSTRATE_ID,
            "target_frame_count": self.frame_count,
            "target_height": int(self.packet_config.height),
            "target_width": int(self.packet_config.width),
            "score_claim": False,
            "promotion_eligible": False,
            "source": dict(self.source_manifest),
        }
        sections = {
            HprcSectionKind.DECODER_QW: pack_compact_decoder(self.mean, self.basis),
            HprcSectionKind.LATENTS_RC: pack_compact_latents(self.latents),
            HprcSectionKind.SELECTORS_RC: pack_compact_selectors(
                np.rint(self.selectors * 255.0).clip(0, 255).astype(np.uint8)
            ),
            HprcSectionKind.RESIDUAL_RC: (
                pack_compact_residual(self.residual)
                if self.protected_residual is None
                else pack_compact_residual_protected(self.residual, self.protected_residual)
            ),
            HprcSectionKind.RDO_PLAN: _json_bytes(rdo_plan),
            HprcSectionKind.RECEIVER_STATE: pack_compact_receiver_state(self.receiver_state),
            HprcSectionKind.MANIFEST_JSON: _json_bytes(manifest),
        }
        return pack_hprc_packet(
            sections,
            config=HprcPacketConfig(
                frames=self.frame_count,
                pairs=self.pair_count,
                height=int(self.packet_config.height),
                width=int(self.packet_config.width),
                decoder_family_id=COMPACT_NUMPY_DECODER_FAMILY_ID,
                color_transform_id=COMPACT_RGB_COLOR_TRANSFORM_ID,
                gop_size=int(self.packet_config.gop_size),
            ),
        )


def _state_scalar(state: Mapping[str, Any], key: str, default: float) -> float:
    value = state.get(key, [default])
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


class HprcCompactReceiverLongTrainingAdapter:
    """Canonical long-training adapter for compact HPRC train/export smokes."""

    substrate_id: str = HPRC_LONG_TRAINING_SUBSTRATE_ID

    def __init__(
        self,
        frames: np.ndarray,
        *,
        basis_count: int = 3,
        residual_grid_h: int = 24,
        residual_grid_w: int = 32,
        source_manifest: Mapping[str, Any] | None = None,
        initial_latent_gain: float = 1.0,
        initial_residual_gain: float = 1.0,
        initial_protected_residual_gain: float = 1.0,
        initial_receiver_state_gain: float = 0.25,
        gain_bounds: HprcGainBounds | None = None,
        repo_root: str | Path | None = None,
        retain_receiver_proof_output: bool = False,
        emit_archive_bound_candidate_package: bool = True,
        native_rate_aware: bool = False,
        rate_aware_residual_l1_weight: float = 0.0,
        rate_aware_residual_prox_weight: float = 0.0,
        residual_protection: np.ndarray | None = None,
        protected_residual_mask: np.ndarray | None = None,
        enable_protected_residual_pathway: bool = False,
        protected_residual_grid_h: int | None = None,
        protected_residual_grid_w: int | None = None,
        training_backend: str = "auto",
    ) -> None:
        self.requested_training_backend = _training_backend_requested(training_backend)
        self.effective_training_backend = _select_training_backend(self.requested_training_backend)
        self._mlx = require_mlx_core() if self.effective_training_backend == "mlx" else None
        self.model = HprcCompactReceiverTrainingModel(
            frames,
            basis_count=basis_count,
            residual_grid_h=residual_grid_h,
            residual_grid_w=residual_grid_w,
            protected_residual_grid_h=protected_residual_grid_h
            if enable_protected_residual_pathway
            else None,
            protected_residual_grid_w=protected_residual_grid_w
            if enable_protected_residual_pathway
            else None,
            protected_residual_mask=(
                (
                    protected_residual_mask
                    if protected_residual_mask is not None
                    else residual_protection
                )
                if enable_protected_residual_pathway
                else None
            ),
            source_manifest=source_manifest,
            initial_latent_gain=initial_latent_gain,
            initial_residual_gain=initial_residual_gain,
            initial_protected_residual_gain=initial_protected_residual_gain,
            initial_receiver_state_gain=initial_receiver_state_gain,
            gain_bounds=gain_bounds,
        )
        self.model.training_backend_lineage = {
            "schema": HPRC_MLX_TRAIN_NUMPY_PORTABLE_SCHEMA,
            "requested_training_backend": self.requested_training_backend,
            "effective_training_backend": self.effective_training_backend,
            "portable_runtime": "numpy",
            "portable_archive_contract": "hprc_packet_plus_numpy_decode_only_receiver",
            "contest_runtime_requires_mlx": False,
            "contest_runtime_requires_torch": False,
            "contest_auth_eval_path": "inflate.sh -> numpy receiver -> upstream evaluate.py",
            "score_claim": False,
            "promotion_eligible": False,
        }
        self.repo_root = None if repo_root is None else Path(repo_root)
        self.retain_receiver_proof_output = bool(retain_receiver_proof_output)
        self.emit_archive_bound_candidate_package = bool(emit_archive_bound_candidate_package)
        self.native_rate_aware = bool(native_rate_aware)
        self.rate_aware_residual_l1_weight = max(0.0, float(rate_aware_residual_l1_weight))
        self.rate_aware_residual_prox_weight = max(0.0, float(rate_aware_residual_prox_weight))
        self.residual_protection = _normalize_residual_protection(
            residual_protection,
            residual_shape=tuple(int(v) for v in self.model.residual.shape),
        )
        self.protected_residual_mask = _normalize_residual_protection(
            protected_residual_mask,
            residual_shape=tuple(int(v) for v in self.model.residual.shape),
        )
        if self.native_rate_aware and (
            self.rate_aware_residual_l1_weight <= 0.0
            and self.rate_aware_residual_prox_weight <= 0.0
        ):
            raise ValueError("native rate-aware HPRC training requires a positive residual rate weight")

    def sample_batch(self, batch_size: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        pair_count = self.model.pair_count
        size = min(int(batch_size), pair_count)
        pair_indices = np.sort(rng.choice(pair_count, size=size, replace=False))
        frame_indices: list[int] = []
        for pair_index in pair_indices:
            start = int(pair_index) * max(int(self.model.packet_config.gop_size), 1)
            for offset in range(max(int(self.model.packet_config.gop_size), 1)):
                frame_index = start + offset
                if frame_index < self.model.frame_count:
                    frame_indices.append(frame_index)
        return {
            "pair_indices": pair_indices.astype(np.int32),
            "frame_indices": np.array(frame_indices, dtype=np.int32),
        }

    def loss_fn(
        self,
        model: HprcCompactReceiverTrainingModel,
        batch: Mapping[str, Any],
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        loss, grads, metrics = self._loss_and_grads(model, batch, loss_weights)
        return {
            "total": float(loss),
            **metrics,
            "latent_gain_grad": float(grads[0]),
            "residual_gain_grad": float(grads[1]),
            "receiver_state_gain_grad": float(grads[2]),
            "protected_residual_gain_grad": float(grads[3]),
        }

    def optimizer_step(self, model: Any, loss: Any, learning_rate: float) -> None:
        raise NotImplementedError(
            "HPRC compact receiver uses Style B train_step so the gain-gradient "
            "and hard projection happen atomically."
        )

    def train_step(
        self,
        batch: Mapping[str, Any],
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> Mapping[str, float]:
        if self.effective_training_backend == "mlx":
            loss, grads, metrics = self._loss_and_grads_mlx(self.model, batch, loss_weights)
        else:
            loss, grads, metrics = self._loss_and_grads(self.model, batch, loss_weights)
        latent = self.model.latent_gain - float(learning_rate) * grads[0]
        residual = self.model.residual_gain - float(learning_rate) * grads[1]
        state = self.model.receiver_state_gain - float(learning_rate) * grads[2]
        protected = self.model.protected_residual_gain - float(learning_rate) * grads[3]
        self.model.latent_gain, self.model.residual_gain, self.model.receiver_state_gain = (
            self.model.gain_bounds.clamp(
                latent=latent,
                residual=residual,
                receiver_state=state,
            )
        )
        self.model.protected_residual_gain = (
            self.model.gain_bounds.clamp_protected_residual(protected)
        )
        if self.effective_training_backend == "mlx":
            rate_update = self._apply_native_residual_rate_step_mlx(
                batch=batch,
                learning_rate=float(learning_rate),
                loss_weights=loss_weights,
            )
        else:
            rate_update = self._apply_native_residual_rate_step(
                batch=batch,
                learning_rate=float(learning_rate),
                loss_weights=loss_weights,
            )
        self.model.train_steps += 1
        return {
            "total": float(loss),
            **metrics,
            "latent_gain_grad": float(grads[0]),
            "residual_gain_grad": float(grads[1]),
            "receiver_state_gain_grad": float(grads[2]),
            "protected_residual_gain_grad": float(grads[3]),
            **rate_update,
        }

    def _loss_and_grads(
        self,
        model: HprcCompactReceiverTrainingModel,
        batch: Mapping[str, Any],
        loss_weights: Mapping[str, float],
    ) -> tuple[float, tuple[float, float, float, float], dict[str, float]]:
        frame_indices = np.asarray(batch["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            raise ValueError("HPRC training batch contains no frames")
        recon_weight = float(loss_weights.get("recon", 1.0))
        gain_l2_weight = float(loss_weights.get("gain_l2", 0.0))
        residual_l1_weight = self._residual_l1_weight(loss_weights)
        score_protection_recon_weight = self._score_protection_recon_weight(loss_weights)
        sse = 0.0
        count = 0
        grad = np.zeros((4,), dtype=np.float64)
        for frame_index in frame_indices:
            pred, components = model.render_continuous(int(frame_index))
            target = model.target_frames[int(frame_index)]
            diff = pred - target
            pixel_weight = self._score_protection_pixel_weight_np(
                int(frame_index),
                score_protection_recon_weight,
            )
            weighted_diff = diff if pixel_weight is None else diff * pixel_weight
            sse += float(np.sum(diff * weighted_diff))
            count += int(diff.size)
            inv_count = 2.0 / max(diff.size, 1)
            for i, component in enumerate(components):
                grad[i] += float(np.sum(weighted_diff * component) * inv_count)
        mse = sse / max(count, 1)
        residual_rate_l1 = self._residual_rate_l1(frame_indices)
        recon_objective = float(mse)
        grad /= float(frame_indices.size)
        if gain_l2_weight:
            defaults = np.array([1.0, 1.0, 0.25, 1.0], dtype=np.float64)
            gains = np.array(
                [
                    model.latent_gain,
                    model.residual_gain,
                    model.receiver_state_gain,
                    model.protected_residual_gain,
                ],
                dtype=np.float64,
            )
            delta = gains - defaults
            recon_objective += gain_l2_weight * float(np.sum(delta * delta))
            grad += 2.0 * gain_l2_weight * delta
        grad *= recon_weight
        total_loss = recon_weight * recon_objective + residual_l1_weight * residual_rate_l1
        return total_loss, (
            float(grad[0]),
            float(grad[1]),
            float(grad[2]),
            float(grad[3]),
        ), {
            "recon": float(mse),
            "residual_rate_l1_proxy": float(residual_rate_l1),
            "residual_rate_l1_weight": float(residual_l1_weight),
            "score_protection_recon_weight": float(score_protection_recon_weight),
            "score_protection_recon_active": float(
                score_protection_recon_weight > 0.0 and self.residual_protection is not None
            ),
            "residual_nonzero_fraction": float(np.count_nonzero(model.residual) / model.residual.size),
            "protected_residual_gain": float(model.protected_residual_gain),
        }

    def _mlx_common_tensors(
        self,
        model: HprcCompactReceiverTrainingModel,
        frame_indices: np.ndarray,
    ) -> dict[str, Any]:
        mx = self._mlx or require_mlx_core()
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        target = mx.array(model.target_frames[indices].astype(np.float32, copy=False))
        mean = mx.broadcast_to(mx.array(model.mean.astype(np.float32, copy=False)), target.shape)
        basis = mx.array(model.basis.astype(np.float32, copy=False))
        latents = mx.array(model.latents[indices].astype(np.float32, copy=False))
        latent_component = mx.tensordot(latents, basis, axes=([1], [0]))
        selector = mx.array(model.selectors[indices].astype(np.float32, copy=False)).reshape(
            (int(indices.size), 1, 1, 1)
        )
        pair_indices = (
            indices // max(int(model.packet_config.gop_size), 1)
        ).clip(0, model.receiver_state.shape[0] - 1)
        if model.receiver_state.shape[1] >= 3:
            state = mx.array(model.receiver_state[pair_indices, :3].astype(np.float32, copy=False))
            state_component = mx.broadcast_to(state.reshape((int(indices.size), 1, 1, 3)), target.shape)
        else:
            state_component = mx.zeros_like(target)
        protected_residual_component = mx.zeros_like(target)
        if model.protected_residual is not None:
            protected_batch = mx.array(
                model.protected_residual[indices].astype(np.float32, copy=False)
            )
            protected_residual_component = selector * _nearest_resize_batch_mlx(
                mx,
                protected_batch,
                int(model.packet_config.height),
                int(model.packet_config.width),
            )
        protection_component = None
        if self.residual_protection is not None:
            protection_grid = mx.array(
                self.residual_protection[indices].astype(np.float32, copy=False)
            )
            protection_component = _nearest_resize_batch_mlx(
                mx,
                protection_grid,
                int(model.packet_config.height),
                int(model.packet_config.width),
            )
        return {
            "mx": mx,
            "target": target,
            "mean": mean,
            "latent_component": latent_component,
            "selector": selector,
            "state_component": state_component,
            "protected_residual_component": protected_residual_component,
            "protection_component": protection_component,
            "height": int(model.packet_config.height),
            "width": int(model.packet_config.width),
            "frame_indices": indices,
        }

    def _mlx_prediction_from_residual(
        self,
        model: HprcCompactReceiverTrainingModel,
        common: Mapping[str, Any],
        residual_batch: Any,
    ) -> tuple[Any, Any]:
        mx = common["mx"]
        residual_component = common["selector"] * _nearest_resize_batch_mlx(
            mx,
            residual_batch,
            int(common["height"]),
            int(common["width"]),
        )
        residual_component = residual_component + (
            float(model.protected_residual_gain) * common["protected_residual_component"]
        )
        pred = (
            common["mean"]
            + float(model.latent_gain) * common["latent_component"]
            + float(model.residual_gain) * residual_component
            + float(model.receiver_state_gain) * common["state_component"]
        )
        return pred, residual_component

    def _loss_and_grads_mlx(
        self,
        model: HprcCompactReceiverTrainingModel,
        batch: Mapping[str, Any],
        loss_weights: Mapping[str, float],
    ) -> tuple[float, tuple[float, float, float, float], dict[str, float]]:
        frame_indices = np.asarray(batch["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            raise ValueError("HPRC training batch contains no frames")
        mx = self._mlx or require_mlx_core()
        common = self._mlx_common_tensors(model, frame_indices)
        residual_batch = mx.array(model.residual[frame_indices].astype(np.float32, copy=False))
        pred, residual_component = self._mlx_prediction_from_residual(model, common, residual_batch)
        diff = pred - common["target"]
        recon_weight = float(loss_weights.get("recon", 1.0))
        gain_l2_weight = float(loss_weights.get("gain_l2", 0.0))
        residual_l1_weight = self._residual_l1_weight(loss_weights)
        score_protection_recon_weight = self._score_protection_recon_weight(loss_weights)
        pixel_weight = self._score_protection_pixel_weight_mlx(common, score_protection_recon_weight)
        weighted_diff = diff if pixel_weight is None else diff * pixel_weight
        mse = mx.mean(diff * weighted_diff)
        grad_scale = 2.0 / float(np.prod(model.target_frames.shape[1:]) * frame_indices.size)
        grad_latent = mx.sum(weighted_diff * common["latent_component"]) * grad_scale
        grad_residual = mx.sum(weighted_diff * residual_component) * grad_scale
        grad_state = mx.sum(weighted_diff * common["state_component"]) * grad_scale
        grad_protected = (
            mx.sum(
                weighted_diff
                * (float(model.residual_gain) * common["protected_residual_component"])
            )
            * grad_scale
        )
        recon_objective = mse
        if gain_l2_weight:
            defaults = mx.array([1.0, 1.0, 0.25, 1.0], dtype=mx.float32)
            gains = mx.array(
                [
                    model.latent_gain,
                    model.residual_gain,
                    model.receiver_state_gain,
                    model.protected_residual_gain,
                ],
                dtype=mx.float32,
            )
            delta = gains - defaults
            recon_objective = recon_objective + float(gain_l2_weight) * mx.sum(delta * delta)
            grad_l2 = 2.0 * float(gain_l2_weight) * delta
            grad_latent = grad_latent + grad_l2[0]
            grad_residual = grad_residual + grad_l2[1]
            grad_state = grad_state + grad_l2[2]
            grad_protected = grad_protected + grad_l2[3]
        pressure = mx.array(self._residual_rate_pressure()[frame_indices].astype(np.float32, copy=False))
        residual_rate_l1 = mx.mean(mx.abs(residual_batch) * pressure)
        total_loss = float(recon_weight) * recon_objective + float(residual_l1_weight) * residual_rate_l1
        grads = (
            float(recon_weight) * grad_latent,
            float(recon_weight) * grad_residual,
            float(recon_weight) * grad_state,
            float(recon_weight) * grad_protected,
        )
        mx.eval(total_loss, mse, residual_rate_l1, *grads)
        return _mx_scalar(total_loss), tuple(_mx_scalar(g) for g in grads), {
            "recon": _mx_scalar(mse),
            "residual_rate_l1_proxy": _mx_scalar(residual_rate_l1),
            "residual_rate_l1_weight": float(residual_l1_weight),
            "score_protection_recon_weight": float(score_protection_recon_weight),
            "score_protection_recon_active": float(
                score_protection_recon_weight > 0.0 and self.residual_protection is not None
            ),
            "residual_nonzero_fraction": float(np.count_nonzero(model.residual) / model.residual.size),
            "protected_residual_gain": float(model.protected_residual_gain),
            "loss_backend_is_mlx": 1.0,
        }

    def _residual_l1_weight(self, loss_weights: Mapping[str, float]) -> float:
        explicit = loss_weights.get("residual_rate_l1")
        if explicit is not None:
            return max(0.0, float(explicit))
        if not self.native_rate_aware:
            return 0.0
        return self.rate_aware_residual_l1_weight

    def _residual_prox_weight(self, loss_weights: Mapping[str, float]) -> float:
        explicit = loss_weights.get("residual_rate_prox")
        if explicit is not None:
            return max(0.0, float(explicit))
        if not self.native_rate_aware:
            return 0.0
        return self.rate_aware_residual_prox_weight

    def _residual_recon_update_weight(self, loss_weights: Mapping[str, float]) -> float:
        return max(0.0, float(loss_weights.get("residual_recon_update", 0.0)))

    def _score_protection_recon_weight(self, loss_weights: Mapping[str, float]) -> float:
        return max(0.0, float(loss_weights.get("score_protection_recon", 0.0)))

    def _score_protection_pixel_weight_np(
        self,
        frame_index: int,
        weight: float,
    ) -> np.ndarray | None:
        if weight <= 0.0 or self.residual_protection is None:
            return None
        protection_grid = self.residual_protection[int(frame_index)]
        protection = _nearest_resize(
            protection_grid,
            int(self.model.packet_config.height),
            int(self.model.packet_config.width),
        ).astype(np.float64, copy=False)
        pixel_weight = 1.0 + float(weight) * protection
        mean = float(np.mean(pixel_weight))
        if mean > 0.0:
            pixel_weight = pixel_weight / mean
        return pixel_weight

    def _score_protection_pixel_weight_mlx(
        self,
        common: Mapping[str, Any],
        weight: float,
    ) -> Any | None:
        if weight <= 0.0 or common.get("protection_component") is None:
            return None
        mx = common["mx"]
        pixel_weight = 1.0 + float(weight) * common["protection_component"]
        return pixel_weight / mx.maximum(mx.mean(pixel_weight), 1e-6)

    def _residual_rate_pressure(self) -> np.ndarray:
        if self.residual_protection is None:
            return np.ones_like(self.model.residual, dtype=np.float32)
        return (1.0 - self.residual_protection).astype(np.float32, copy=False)

    def _residual_rate_l1(self, frame_indices: np.ndarray) -> float:
        residual = self.model.residual[frame_indices]
        pressure = self._residual_rate_pressure()[frame_indices]
        return float(np.mean(np.abs(residual) * pressure))

    def _apply_native_residual_rate_step(
        self,
        *,
        batch: Mapping[str, Any],
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> dict[str, float]:
        l1_weight = self._residual_l1_weight(loss_weights)
        prox_weight = self._residual_prox_weight(loss_weights)
        residual_recon_weight = self._residual_recon_update_weight(loss_weights)
        if l1_weight <= 0.0 and prox_weight <= 0.0 and residual_recon_weight <= 0.0:
            return {
                "native_rate_residual_recon_update_weight": 0.0,
                "native_rate_residual_update_l1_weight": 0.0,
                "native_rate_residual_update_prox_weight": 0.0,
                "native_rate_residual_mean_abs_delta": 0.0,
                "native_rate_residual_update_backend_is_mlx": 0.0,
            }
        frame_indices = np.asarray(batch["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            return {
                "native_rate_residual_recon_update_weight": float(residual_recon_weight),
                "native_rate_residual_update_l1_weight": float(l1_weight),
                "native_rate_residual_update_prox_weight": float(prox_weight),
                "native_rate_residual_mean_abs_delta": 0.0,
                "native_rate_residual_update_backend_is_mlx": 0.0,
            }
        before = self.model.residual[frame_indices].copy()
        pressure_all = self._residual_rate_pressure()
        recon_weight = float(residual_recon_weight)
        score_protection_recon_weight = self._score_protection_recon_weight(loss_weights)
        grid_h, grid_w = int(self.model.residual.shape[1]), int(self.model.residual.shape[2])
        for frame_index in frame_indices:
            idx = int(frame_index)
            pred, _components = self.model.render_continuous(idx)
            diff = pred - self.model.target_frames[idx]
            pixel_weight = self._score_protection_pixel_weight_np(
                idx,
                score_protection_recon_weight,
            )
            weighted_diff = diff if pixel_weight is None else diff * pixel_weight
            grad_grid = _downsample_sum_nearest_inverse(weighted_diff, grid_h, grid_w)
            grad_grid *= (
                recon_weight
                * float(self.model.selectors[idx])
                * float(self.model.residual_gain)
                * (2.0 / max(diff.size, 1))
            )
            pressure = pressure_all[idx].astype(np.float64)
            if l1_weight > 0.0:
                grad_grid += (
                    float(l1_weight)
                    * pressure
                    * np.sign(self.model.residual[idx]).astype(np.float64)
                    / max(self.model.residual[idx].size, 1)
                )
            updated = self.model.residual[idx].astype(np.float64) - float(learning_rate) * grad_grid
            if prox_weight > 0.0:
                shrink = float(learning_rate) * float(prox_weight) * pressure
                updated = np.sign(updated) * np.maximum(np.abs(updated) - shrink, 0.0)
            self.model.residual[idx] = np.nan_to_num(updated, copy=False).astype(np.float32)
        after = self.model.residual[frame_indices]
        return {
            "native_rate_residual_recon_update_weight": float(residual_recon_weight),
            "native_rate_residual_update_l1_weight": float(l1_weight),
            "native_rate_residual_update_prox_weight": float(prox_weight),
            "native_rate_residual_mean_abs_delta": float(np.mean(np.abs(after - before))),
            "native_rate_residual_update_backend_is_mlx": 0.0,
        }

    def _apply_native_residual_rate_step_mlx(
        self,
        *,
        batch: Mapping[str, Any],
        learning_rate: float,
        loss_weights: Mapping[str, float],
    ) -> dict[str, float]:
        l1_weight = self._residual_l1_weight(loss_weights)
        prox_weight = self._residual_prox_weight(loss_weights)
        residual_recon_weight = self._residual_recon_update_weight(loss_weights)
        if l1_weight <= 0.0 and prox_weight <= 0.0 and residual_recon_weight <= 0.0:
            return {
                "native_rate_residual_recon_update_weight": 0.0,
                "native_rate_residual_update_l1_weight": 0.0,
                "native_rate_residual_update_prox_weight": 0.0,
                "native_rate_residual_mean_abs_delta": 0.0,
                "native_rate_residual_update_backend_is_mlx": 1.0,
            }
        frame_indices = np.asarray(batch["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            return {
                "native_rate_residual_recon_update_weight": float(residual_recon_weight),
                "native_rate_residual_update_l1_weight": float(l1_weight),
                "native_rate_residual_update_prox_weight": float(prox_weight),
                "native_rate_residual_mean_abs_delta": 0.0,
                "native_rate_residual_update_backend_is_mlx": 1.0,
            }
        mx = self._mlx or require_mlx_core()
        before = self.model.residual[frame_indices].copy()
        common = self._mlx_common_tensors(self.model, frame_indices)
        residual_batch = mx.array(before.astype(np.float32, copy=False))
        pressure = mx.array(self._residual_rate_pressure()[frame_indices].astype(np.float32, copy=False))
        recon_weight = float(residual_recon_weight)
        score_protection_recon_weight = self._score_protection_recon_weight(loss_weights)
        pixel_weight = self._score_protection_pixel_weight_mlx(
            common,
            score_protection_recon_weight,
        )

        def residual_objective(residual_value: Any) -> Any:
            pred, _residual_component = self._mlx_prediction_from_residual(
                self.model,
                common,
                residual_value,
            )
            diff = pred - common["target"]
            weighted_diff = diff if pixel_weight is None else diff * pixel_weight
            recon = float(recon_weight) * mx.mean(diff * weighted_diff)
            rate = float(l1_weight) * mx.mean(mx.abs(residual_value) * pressure)
            return recon + rate

        loss_value, grad = mx.value_and_grad(residual_objective)(residual_batch)
        updated = residual_batch - float(learning_rate) * grad
        if prox_weight > 0.0:
            shrink = float(learning_rate) * float(prox_weight) * pressure
            updated = mx.sign(updated) * mx.maximum(mx.abs(updated) - shrink, 0.0)
        mx.eval(loss_value, updated)
        updated_np = np.nan_to_num(np.array(updated), copy=False).astype(np.float32, copy=False)
        self.model.residual[frame_indices] = updated_np
        after = self.model.residual[frame_indices]
        return {
            "native_rate_residual_recon_update_weight": float(residual_recon_weight),
            "native_rate_residual_update_l1_weight": float(l1_weight),
            "native_rate_residual_update_prox_weight": float(prox_weight),
            "native_rate_residual_mean_abs_delta": float(np.mean(np.abs(after - before))),
            "native_rate_residual_update_backend_is_mlx": 1.0,
        }

    def export_state_dict(self, model: HprcCompactReceiverTrainingModel, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = model.state_dict()
        metadata = {
            "schema": "hprc_compact_receiver_training_state.v1",
            "substrate_id": self.substrate_id,
            "state_format": "npz",
            "state_dict_summary": _state_dict_summary(state),
            "train_steps": int(model.train_steps),
            "frame_count": int(model.frame_count),
            "height": int(model.packet_config.height),
            "width": int(model.packet_config.width),
            "source_manifest": dict(model.source_manifest),
            "score_claim": False,
            "promotion_eligible": False,
        }
        arrays: dict[str, np.ndarray] = {
            "__metadata_json_utf8": np.frombuffer(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                dtype=np.uint8,
            ),
            "latent_gain": np.array([model.latent_gain], dtype=np.float64),
            "residual_gain": np.array([model.residual_gain], dtype=np.float64),
            "protected_residual_gain": np.array(
                [model.protected_residual_gain],
                dtype=np.float64,
            ),
            "receiver_state_gain": np.array([model.receiver_state_gain], dtype=np.float64),
            "residual": np.asarray(model.residual, dtype=np.float32),
        }
        if model.protected_residual is not None:
            arrays["protected_residual"] = np.asarray(model.protected_residual, dtype=np.float32)
        np.savez_compressed(path.with_suffix(path.suffix + ".npz"), **arrays)

    def import_state_dict(self, model: HprcCompactReceiverTrainingModel, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(bytes(data["__metadata_json_utf8"].tolist()).decode("utf-8"))
            state: dict[str, Any] = {
                "latent_gain": [float(data["latent_gain"].reshape(-1)[0])],
                "residual_gain": [float(data["residual_gain"].reshape(-1)[0])],
                "protected_residual_gain": [
                    float(data["protected_residual_gain"].reshape(-1)[0])
                ],
                "receiver_state_gain": [float(data["receiver_state_gain"].reshape(-1)[0])],
                "residual": np.asarray(data["residual"], dtype=np.float32),
            }
            if "protected_residual" in data.files:
                state["protected_residual"] = np.asarray(
                    data["protected_residual"],
                    dtype=np.float32,
                )
        model.load_state_dict(
            {
                "state_dict": state,
                "train_steps": int(metadata.get("train_steps", model.train_steps)),
            }
        )

    def export_archive(
        self,
        model: HprcCompactReceiverTrainingModel,
        output_dir: Path,
    ) -> tuple[Path, str, int] | None:
        output_dir.mkdir(parents=True, exist_ok=True)
        packet = model.packet_bytes()
        export_dir = output_dir / "hprc_compact_receiver_archive_export"
        archive_path, archive_sha256, archive_bytes = hprc_archive_candidate.export_hprc_archive_bytes(
            packet,
            export_dir,
            repo_root=self.repo_root,
            emit_archive_bound_candidate_package=self.emit_archive_bound_candidate_package,
            retain_receiver_proof_output=self.retain_receiver_proof_output,
        )
        export_manifest = {
            "schema": HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA,
            "archive_zip_path": archive_path.as_posix(),
            "archive_zip_sha256": archive_sha256,
            "archive_zip_bytes": int(archive_bytes),
            "hprc_packet_sha256": _sha256_bytes(packet),
            "state_dict_summary": _state_dict_summary(model.state_dict()),
            "receiver_proof_requested": bool(self.emit_archive_bound_candidate_package),
            "receiver_output_retained": bool(self.retain_receiver_proof_output),
            **hprc_archive_candidate.FALSE_AUTHORITY,
        }
        (output_dir / "hprc_compact_receiver_training_export.json").write_text(
            json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return archive_path, archive_sha256, int(archive_bytes)

    def score_aware_components(
        self,
        model: HprcCompactReceiverTrainingModel,
        batch: Mapping[str, Any],
    ) -> Mapping[str, float] | None:
        loss, _grads, _metrics = self._loss_and_grads(model, batch, {"recon": 1.0})
        return {
            "decoder_grid_mse_rgb255_advisory": float(loss),
            "archive_rate_term_advisory": 0.0,
            "native_rate_aware_training_enabled": bool(self.native_rate_aware),
            "protected_highres_residual_pathway_enabled": self.model.protected_residual is not None,
            "protected_residual_gain": float(self.model.protected_residual_gain),
            "native_residual_nonzero_fraction": float(
                np.count_nonzero(model.residual) / model.residual.size
            ),
        }

    def artifact_metadata(self) -> Mapping[str, Any]:
        packet = self.model.packet_bytes()
        return {
            "schema": "hprc_compact_receiver_training_metadata.v1",
            "receiver_mode": COMPACT_RECEIVER_MODE,
            "portable_runtime": "numpy",
            "training_backend": {
                "schema": HPRC_MLX_TRAIN_NUMPY_PORTABLE_SCHEMA,
                "requested_training_backend": self.requested_training_backend,
                "effective_training_backend": self.effective_training_backend,
                "portable_runtime": "numpy",
                "archive_export_runtime_requires_mlx": False,
                "contest_runtime_requires_mlx": False,
                "contest_runtime_requires_torch": False,
            },
            "training_surface": "rdo_gain_projection_with_trainable_protected_pose_gain",
            "native_rate_aware_training": {
                "schema": HPRC_NATIVE_RATE_AWARE_TRAINING_SCHEMA,
                "enabled": bool(self.native_rate_aware),
                "residual_l1_weight": float(self.rate_aware_residual_l1_weight),
                "residual_prox_weight": float(self.rate_aware_residual_prox_weight),
                "residual_protection_present": self.residual_protection is not None,
                "residual_protection_semantics": (
                    "1=protect_from_rate_pressure,0=safest_to_shrink"
                ),
                "protected_reconstruction_pathway": self.residual_protection is not None,
                "protected_reconstruction_semantics": (
                    "curriculum loss weight score_protection_recon upweights "
                    "P18/P19 protected residual cells during reconstruction and "
                    "residual-token repair updates"
                ),
                "residual_nonzero_fraction": float(
                    np.count_nonzero(self.model.residual) / self.model.residual.size
                ),
            },
            "protected_highres_residual_pathway": {
                "schema": "hprc_protected_highres_residual_pathway.v1",
                "enabled": self.model.protected_residual is not None,
                "grid_h": (
                    0
                    if self.model.protected_residual is None
                    else int(self.model.protected_residual.shape[1])
                ),
                "grid_w": (
                    0
                    if self.model.protected_residual is None
                    else int(self.model.protected_residual.shape[2])
                ),
                "channels": (
                    0
                    if self.model.protected_residual is None
                    else int(self.model.protected_residual.shape[3])
                ),
                "mask_source": (
                    "none"
                    if self.model.protected_residual is None
                    else "p18_p19_sparse_protected_mask"
                    if self.protected_residual_mask is not None
                    else "p18_p19_residual_protection"
                    if self.residual_protection is not None
                    else "unmasked_highres_residual"
                ),
                "receiver_storage": "residual_rc_v2_dense_or_v3_sparse_int8_protected_sidecar",
                "trainable_gain": True,
                "protected_residual_gain": float(self.model.protected_residual_gain),
                "training_backend": self.effective_training_backend,
                "portable_runtime": "numpy",
            },
            "packet_sha256_at_metadata": _sha256_bytes(packet),
            "frame_count": int(self.model.frame_count),
            "decoder_grid": {
                "height": int(self.model.packet_config.height),
                "width": int(self.model.packet_config.width),
            },
            "remaining_blockers": [
                "z8_scorer_weighted_residual_sidecar_missing",
                "full_video_p18_p19_allocator_not_bound_to_candidate",
                "local_cpu_full_video_replay_not_executed",
                "contest_cpu_cuda_exact_eval_not_executed",
            ],
        }


__all__ = [
    "HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA",
    "HPRC_LONG_TRAINING_SUBSTRATE_ID",
    "HPRC_MLX_TRAIN_NUMPY_PORTABLE_SCHEMA",
    "HPRC_NATIVE_RATE_AWARE_TRAINING_SCHEMA",
    "HPRC_TRAINING_BACKENDS",
    "HprcCompactReceiverLongTrainingAdapter",
    "HprcCompactReceiverTrainingModel",
    "HprcGainBounds",
]
