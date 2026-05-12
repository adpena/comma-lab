"""Ego-NeRV-as-renderer — Egocentric pose-conditioning NeRV substrate.

Per operator directive 2026-05-11 ("all nerv-family"). Ego-NeRV
conditions the decoder on EGOCENTRIC POSE (the contest video is
driving footage from a forward-facing camera; egomotion is the dominant
spatial structure). The substrate consumes a per-pair pose vector
(pre-computed from the scored video at compress-time, stored as a
small extra section in the archive) to drive the decoder via FiLM
modulation on the latent.

Why egocentric pose conditioning
--------------------------------
- **Driving video is egomotion-dominated**: the contest dataset
  (`upstream/videos/0.mkv`) is a continuous forward drive. Per-pair
  pose (yaw, pitch, roll, x, y, z velocity / acceleration) explains
  most of the inter-pair variation.
- **Foveation prior**: pose-conditioning lets the decoder allocate
  capacity to ego-relevant spatial regions (road ahead, lane lines)
  rather than uniformly across the frame.
- **Compose with NeRV-Enc/Dec + FiLM bolt-on**: the FiLM modulator is
  KK's bolt-on (`tac.nerv_pose_conditioning_bolton`) lifted to a
  substrate-level commitment.

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair as in Lane 12-v2.
- Pose table: (n_pairs, pose_dim=6) — extra archive section, FP16,
  small (600 * 6 * 2 = 7200 bytes).
- FiLM modulator: (B, 6) → (scale (B, 16), shift (B, 16)) via 2-layer
  MLP. Identity-init.
- Decoder: standard PixelShuffle stack on FiLM-modulated latent.
- Output: (B, 2, 3, 384, 512) RGB.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header
    + 5 length-prefixed sections: decoder_blob, scale_table, latent_blob,
    pose_table FP16, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_EGO_NERV with
    schema_keys_in_order = EgoNeRVRenderer.SCHEMA
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: ego_nerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: train_step_ego_nerv with diff rgb_to_yuv6 +
    load_differentiable_scorers + Lagrangian
  bolt_on_loc_budget: substrate_engineering (full pose-conditioned NeRV)
  no_op_detector_planned: export_ego_nerv_to_archive returns sha256

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; egocentric pose conditioning
exploits driving-video egomotion structure for ~1.25× param efficiency
at fixed bytes per Wang 2024 Ego-NeRV / Park 2023 driving-NeRF;
pose-axis marginal 2.71× SegNet at PR106 r2 frontier]``. NOT a score
claim until [contest-CUDA] anchor.

Format ID: 0x68.
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


EGO_NERV_MAGIC: bytes = b"eNRV"
EGO_NERV_FORMAT_VERSION: int = 1
EGO_NERV_FORMAT_ID: int = 0x68


ARCHIVE_GRAMMAR_EGO_NERV: dict = {
    "format_version": EGO_NERV_FORMAT_VERSION,
    "format_id": EGO_NERV_FORMAT_ID,
    "magic": EGO_NERV_MAGIC.decode("ascii"),
    "sections": [
        {"name": "header", "offset": 0, "length": 16, "kind": "fixed_header"},
        {"name": "decoder_blob", "offset_after": "header",
         "length_field_le_u32": True,
         "kind": "brotli_int8_codes_schema_driven"},
        {"name": "scale_table", "offset_after": "decoder_blob",
         "length_field_le_u32": True,
         "kind": "fp16_raw_one_per_schema_entry"},
        {"name": "latent_blob", "offset_after": "scale_table",
         "length_field_le_u32": True,
         "kind": "brotli_uint8_asym_delta_split"},
        {"name": "pose_table", "offset_after": "latent_blob",
         "length_field_le_u32": True,
         "kind": "fp16_raw_n_pairs_x_pose_dim"},
        {"name": "sidecar_blob", "offset_after": "pose_table",
         "length_field_le_u32": True,
         "kind": "brotli_optional_phase_b", "phase_a_empty": True},
    ],
    "schema_keys_in_order": "EgoNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "160_000 to 190_000 [predicted; pose table adds ~7KB]",
}


@dataclass(frozen=True)
class EgoNeRVConfig:
    latent_dim: int = 16
    pose_dim: int = 6
    film_hidden_dim: int = 64
    base_channels: int = 36
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    n_pairs: int = 600
    n_stages: int = 6
    frames_per_pair: int = 2
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive, got {self.pose_dim}")
        if self.film_hidden_dim <= 0:
            raise ValueError(f"film_hidden_dim must be positive, got {self.film_hidden_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_stages != 6:
            raise ValueError(f"Phase A pinned at n_stages=6, got {self.n_stages}")
        if self.frames_per_pair != 2:
            raise ValueError(f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}")


class _FiLMModulator(nn.Module):
    """Pose (B, pose_dim) → (scale (B, latent_dim), shift (B, latent_dim)).

    Identity-init: scale=1, shift=0 so training starts equivalent to
    "no modulation" per KK's bolt-on. Sister to
    `tac.nerv_pose_conditioning_bolton.FiLMModulator`.
    """

    def __init__(self, pose_dim: int, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(pose_dim, hidden_dim)
        self.fc2_scale = nn.Linear(hidden_dim, latent_dim)
        self.fc2_shift = nn.Linear(hidden_dim, latent_dim)
        # Identity init: outputs scale=1 + tanh(0)=1, shift=0.
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2_scale.weight)
        nn.init.zeros_(self.fc2_scale.bias)
        nn.init.zeros_(self.fc2_shift.weight)
        nn.init.zeros_(self.fc2_shift.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(pose))
        scale = 1.0 + torch.tanh(self.fc2_scale(h))  # init to 1.0
        shift = self.fc2_shift(h)  # init to 0.0
        return scale, shift


class EgoNeRVRenderer(nn.Module):
    """Ego-NeRV decoder. Forward: ``(z, pose) → (B, 2, 3, H, W)``."""

    def __init__(self, config: EgoNeRVConfig) -> None:
        super().__init__()
        self.config = config
        self.film = _FiLMModulator(
            pose_dim=config.pose_dim, latent_dim=config.latent_dim,
            hidden_dim=config.film_hidden_dim,
        )
        C = config.base_channels
        self.channels: list[int] = [
            C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)
        ]
        self.stem = nn.Linear(
            config.latent_dim, self.channels[0] * config.base_h * config.base_w
        )
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(config.n_stages):
            in_ch = self.channels[i]; out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected z (B, {self.config.latent_dim}); got {tuple(z.shape)}"
            )
        if pose.dim() != 2 or pose.shape[1] != self.config.pose_dim:
            raise ValueError(
                f"forward expected pose (B, {self.config.pose_dim}); got {tuple(pose.shape)}"
            )
        if pose.shape[0] != z.shape[0]:
            raise ValueError(
                f"z batch ({z.shape[0]}) != pose batch ({pose.shape[0]})"
            )
        # FiLM modulation on latent.
        scale, shift = self.film(pose)
        z_mod = z * scale + shift
        B = z.shape[0]
        x = self.stem(z_mod).view(
            B, self.channels[0], self.config.base_h, self.config.base_w
        )
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        keys = [
            "film.fc1.weight", "film.fc1.bias",
            "film.fc2_scale.weight", "film.fc2_scale.bias",
            "film.fc2_shift.weight", "film.fc2_shift.bias",
            "stem.weight", "stem.bias",
        ]
        for i in range(self.config.n_stages):
            keys += [f"blocks.{i}.weight", f"blocks.{i}.bias"]
        for i in range(self.config.n_stages):
            if isinstance(self.skips[i], nn.Conv2d):
                keys += [f"skips.{i}.weight", f"skips.{i}.bias"]
        keys += [
            "refine.0.weight", "refine.0.bias",
            "refine.1.weight", "refine.1.bias",
            "rgb_0.weight", "rgb_0.bias",
            "rgb_1.weight", "rgb_1.bias",
        ]
        for key in keys:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        return out


class EgoNeRVLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


class EgoNeRVPoseTable(nn.Module):
    """Per-pair pose table; shipped in archive as FP16."""

    def __init__(self, n_pairs: int, pose_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, pose_dim)
        nn.init.zeros_(self.embedding.weight)  # default: no ego motion

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        value = output["pose"]
    else:
        value = output
    if not torch.is_tensor(value):
        raise TypeError(f"pose output must be a tensor, got {type(value).__name__}")
    return value


def train_step_ego_nerv(
    *,
    renderer: EgoNeRVRenderer,
    latent_table: EgoNeRVLatentTable,
    pose_table: EgoNeRVPoseTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
) -> dict:
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z = latent_table(pair_indices)
    pose = pose_table(pair_indices)
    decoded = renderer(z, pose)
    B, F_pp, C, H_native, W_native = decoded.shape
    flat = decoded.reshape(B * F_pp, C, H_native, W_native)
    H_camera, W_camera = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(flat, size=(H_camera, W_camera), mode="bicubic", align_corners=False)
    up_uint8_ste = _eval_roundtrip_uint8_clamp(up)
    up_pairs = up_uint8_ste.reshape(B, F_pp, C, H_camera, W_camera)
    gt_pairs = gt_pairs_uint8.float()

    seg_pred_logits = scorer_seg(scorer_seg.preprocess_input(up_pairs))
    with torch.no_grad():
        seg_target_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred_logits, seg_target_logits)

    pose_pred = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(up_pairs)))
    with torch.no_grad():
        pose_target = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(gt_pairs)))
    loss_pose_unweighted = pose_surrogate(pose_pred, pose_target)

    loss_seg = lambda_seg * loss_seg_unweighted
    loss_pose = lambda_pose * loss_pose_unweighted
    loss = loss_seg + loss_pose
    return {
        "loss": loss,
        "loss_seg": loss_seg,
        "loss_pose": loss_pose,
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
    }


def _quantize_per_tensor_int8_with_fp16_scale(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(tensor.abs().max().item())
    scale = max(max_abs, 1e-8) / 127.0
    scale_fp16 = torch.tensor([scale], dtype=torch.float16)
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale_fp16


def _quantize_latent_table_uint8_delta_split(latents: torch.Tensor) -> bytes:
    import brotli
    n, d = latents.shape
    mins = latents.min(dim=0).values.to(torch.float16)
    maxs = latents.max(dim=0).values.to(torch.float16)
    scales = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)
    q = ((latents - mins.float()) / scales.float()).round().clamp(0, 255).to(torch.int32)
    delta = torch.zeros_like(q)
    delta[0] = q[0]
    delta[1:] = q[1:] - q[:-1]
    delta_zz = torch.where(delta >= 0, 2 * delta, -2 * delta - 1).to(torch.int32)
    delta_zz = delta_zz.clamp(0, 65535)
    delta_lo = (delta_zz & 0xFF).to(torch.uint8)
    delta_hi = ((delta_zz >> 8) & 0xFF).to(torch.uint8)
    buf = io.BytesIO()
    buf.write(struct.pack("<II", n, d))
    buf.write(mins.numpy().tobytes())
    buf.write(scales.numpy().tobytes())
    buf.write(delta_lo.numpy().tobytes())
    buf.write(delta_hi.numpy().tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def export_ego_nerv_to_archive(
    *,
    renderer: EgoNeRVRenderer,
    latent_table: EgoNeRVLatentTable,
    pose_table: EgoNeRVPoseTable,
    output_path: Path,
) -> str:
    import brotli
    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != expected {expected}")
    pose_shape = tuple(pose_table.embedding.weight.shape)
    expected_pose = (config.n_pairs, config.pose_dim)
    if pose_shape != expected_pose:
        raise ValueError(f"pose_table shape {pose_shape} != expected {expected_pose}")

    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        int8_chunks.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    decoder_blob = brotli.compress(b"".join(int8_chunks), quality=11)
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta_split(
        latent_table.embedding.weight.detach().cpu()
    )
    pose_blob = pose_table.embedding.weight.detach().cpu().to(torch.float16).numpy().tobytes()
    sidecar_blob = b""

    out = io.BytesIO()
    out.write(EGO_NERV_MAGIC)
    out.write(struct.pack("<H", EGO_NERV_FORMAT_VERSION))
    out.write(struct.pack("<H", EGO_NERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.pose_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.base_channels))
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(scale_table)))
    out.write(scale_table)
    out.write(struct.pack("<I", len(latent_blob)))
    out.write(latent_blob)
    out.write(struct.pack("<I", len(pose_blob)))
    out.write(pose_blob)
    out.write(struct.pack("<I", len(sidecar_blob)))
    out.write(sidecar_blob)

    archive_bytes = out.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)
    return hashlib.sha256(archive_bytes).hexdigest()


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:ego_nerv_phase_a_scaffold_smoke_test_only"""
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    gt_pairs = torch.randint(0, 256, (batch_size, 2, 3, 874, 1164), generator=g, dtype=torch.uint8)
    return pair_indices, gt_pairs


def default_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])
