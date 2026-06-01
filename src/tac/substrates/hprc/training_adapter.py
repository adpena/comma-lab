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
    pack_compact_selectors,
)

HPRC_LONG_TRAINING_SUBSTRATE_ID = "hprc_compact_receiver"
HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA = "hprc_compact_receiver_training_export.v1"


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nearest_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    y_idx = (np.arange(height, dtype=np.int64) * src_h // height).clip(0, src_h - 1)
    x_idx = (np.arange(width, dtype=np.int64) * src_w // width).clip(0, src_w - 1)
    return frame[y_idx[:, None], x_idx[None, :], :]


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


@dataclass(frozen=True)
class HprcGainBounds:
    """Hard projection bounds for compact receiver RDO gains."""

    latent: tuple[float, float] = (-4.0, 4.0)
    residual: tuple[float, float] = (-4.0, 4.0)
    receiver_state: tuple[float, float] = (-4.0, 4.0)

    def clamp(self, *, latent: float, residual: float, receiver_state: float) -> tuple[float, float, float]:
        return (
            float(np.clip(latent, self.latent[0], self.latent[1])),
            float(np.clip(residual, self.residual[0], self.residual[1])),
            float(np.clip(receiver_state, self.receiver_state[0], self.receiver_state[1])),
        )


class HprcCompactReceiverTrainingModel:
    """Small trainable state for a compact HPRC receiver.

    The heavy tensor fields are fixed archive components.  The trainable state is
    the RDO gain triple consumed by the decode-only receiver runtime.
    """

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
        self.receiver_state = (
            compact.receiver_state.q.astype(np.float32)
            * float(compact.receiver_state.scale)
        )
        self.latent_gain, self.residual_gain, self.receiver_state_gain = self.gain_bounds.clamp(
            latent=float(initial_latent_gain),
            residual=float(initial_residual_gain),
            receiver_state=float(initial_receiver_state_gain),
        )
        self.train_steps = 0

    @property
    def frame_count(self) -> int:
        return int(self.target_frames.shape[0])

    @property
    def pair_count(self) -> int:
        return math.ceil(self.frame_count / max(int(self.packet_config.gop_size), 1))

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "latent_gain": [float(self.latent_gain)],
            "residual_gain": [float(self.residual_gain)],
            "receiver_state_gain": [float(self.receiver_state_gain)],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.latent_gain = _state_scalar(state, "latent_gain", self.latent_gain)
        self.residual_gain = _state_scalar(state, "residual_gain", self.residual_gain)
        self.receiver_state_gain = _state_scalar(
            state, "receiver_state_gain", self.receiver_state_gain
        )
        self.latent_gain, self.residual_gain, self.receiver_state_gain = self.gain_bounds.clamp(
            latent=self.latent_gain,
            residual=self.residual_gain,
            receiver_state=self.receiver_state_gain,
        )

    def render_continuous(self, frame_index: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        return frame, (latent_component, residual_component, state_component)

    def packet_bytes(self) -> bytes:
        rdo_plan = {
            "schema": "hprc_compact_receiver_rdo_plan.v1",
            "decoder_mode": COMPACT_RECEIVER_MODE,
            "training_adapter": HPRC_LONG_TRAINING_SUBSTRATE_ID,
            "latent_gain": float(self.latent_gain),
            "residual_gain": float(self.residual_gain),
            "receiver_state_gain": float(self.receiver_state_gain),
            "basis_count": int(self.basis.shape[0]),
            "residual_grid_h": int(self.residual.shape[1]),
            "residual_grid_w": int(self.residual.shape[2]),
            "output_resize": "bilinear",
            "output_resize_alignment": "bilinear_align_corners_false",
            "train_steps": int(self.train_steps),
            "score_claim": False,
            "promotion_eligible": False,
        }
        manifest = {
            "schema": "hprc_compact_receiver_manifest.v1",
            "hprc_receiver_mode": COMPACT_RECEIVER_MODE,
            "candidate_kind": "compact_numpy_receiver_with_trained_rdo_gains",
            "trained_renderer_export_ready": True,
            "z8_scorer_weighted_residual_sidecar_ready": False,
            "mamba_dreamer_stack_ready": False,
            "exact_cpu_cuda_authority_ready": False,
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
            HprcSectionKind.RESIDUAL_RC: pack_compact_residual(self.residual),
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
        initial_receiver_state_gain: float = 0.25,
        gain_bounds: HprcGainBounds | None = None,
        repo_root: str | Path | None = None,
        retain_receiver_proof_output: bool = False,
        emit_archive_bound_candidate_package: bool = True,
    ) -> None:
        self.model = HprcCompactReceiverTrainingModel(
            frames,
            basis_count=basis_count,
            residual_grid_h=residual_grid_h,
            residual_grid_w=residual_grid_w,
            source_manifest=source_manifest,
            initial_latent_gain=initial_latent_gain,
            initial_residual_gain=initial_residual_gain,
            initial_receiver_state_gain=initial_receiver_state_gain,
            gain_bounds=gain_bounds,
        )
        self.repo_root = None if repo_root is None else Path(repo_root)
        self.retain_receiver_proof_output = bool(retain_receiver_proof_output)
        self.emit_archive_bound_candidate_package = bool(emit_archive_bound_candidate_package)

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
        loss, grads = self._loss_and_grads(model, batch, loss_weights)
        return {
            "total": float(loss),
            "recon": float(loss),
            "latent_gain_grad": float(grads[0]),
            "residual_gain_grad": float(grads[1]),
            "receiver_state_gain_grad": float(grads[2]),
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
        loss, grads = self._loss_and_grads(self.model, batch, loss_weights)
        latent = self.model.latent_gain - float(learning_rate) * grads[0]
        residual = self.model.residual_gain - float(learning_rate) * grads[1]
        state = self.model.receiver_state_gain - float(learning_rate) * grads[2]
        self.model.latent_gain, self.model.residual_gain, self.model.receiver_state_gain = (
            self.model.gain_bounds.clamp(
                latent=latent,
                residual=residual,
                receiver_state=state,
            )
        )
        self.model.train_steps += 1
        return {
            "total": float(loss),
            "recon": float(loss),
            "latent_gain_grad": float(grads[0]),
            "residual_gain_grad": float(grads[1]),
            "receiver_state_gain_grad": float(grads[2]),
        }

    def _loss_and_grads(
        self,
        model: HprcCompactReceiverTrainingModel,
        batch: Mapping[str, Any],
        loss_weights: Mapping[str, float],
    ) -> tuple[float, tuple[float, float, float]]:
        frame_indices = np.asarray(batch["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.size == 0:
            raise ValueError("HPRC training batch contains no frames")
        recon_weight = float(loss_weights.get("recon", 1.0))
        gain_l2_weight = float(loss_weights.get("gain_l2", 0.0))
        sse = 0.0
        count = 0
        grad = np.zeros((3,), dtype=np.float64)
        for frame_index in frame_indices:
            pred, components = model.render_continuous(int(frame_index))
            target = model.target_frames[int(frame_index)]
            diff = pred - target
            sse += float(np.sum(diff * diff))
            count += int(diff.size)
            inv_count = 2.0 / max(diff.size, 1)
            for i, component in enumerate(components):
                grad[i] += float(np.sum(diff * component) * inv_count)
        mse = sse / max(count, 1)
        grad /= float(frame_indices.size)
        if gain_l2_weight:
            defaults = np.array([1.0, 1.0, 0.25], dtype=np.float64)
            gains = np.array(
                [
                    model.latent_gain,
                    model.residual_gain,
                    model.receiver_state_gain,
                ],
                dtype=np.float64,
            )
            delta = gains - defaults
            mse += gain_l2_weight * float(np.sum(delta * delta))
            grad += 2.0 * gain_l2_weight * delta
        grad *= recon_weight
        return recon_weight * mse, (float(grad[0]), float(grad[1]), float(grad[2]))

    def export_state_dict(self, model: HprcCompactReceiverTrainingModel, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "hprc_compact_receiver_training_state.v1",
            "substrate_id": self.substrate_id,
            "state_dict": model.state_dict(),
            "train_steps": int(model.train_steps),
            "frame_count": int(model.frame_count),
            "height": int(model.packet_config.height),
            "width": int(model.packet_config.width),
            "source_manifest": dict(model.source_manifest),
            "score_claim": False,
            "promotion_eligible": False,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
            "state_dict": model.state_dict(),
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
        loss, _grads = self._loss_and_grads(model, batch, {"recon": 1.0})
        return {
            "decoder_grid_mse_rgb255_advisory": float(loss),
            "archive_rate_term_advisory": 0.0,
        }

    def artifact_metadata(self) -> Mapping[str, Any]:
        packet = self.model.packet_bytes()
        return {
            "schema": "hprc_compact_receiver_training_metadata.v1",
            "receiver_mode": COMPACT_RECEIVER_MODE,
            "portable_runtime": "numpy",
            "training_surface": "rdo_gain_projection_over_fixed_compact_components",
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
    "HprcCompactReceiverLongTrainingAdapter",
    "HprcCompactReceiverTrainingModel",
    "HprcGainBounds",
]
