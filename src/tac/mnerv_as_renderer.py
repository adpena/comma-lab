"""MNeRV-as-renderer — multi-scale hierarchical NeRV substrate.

Per operator directive 2026-05-11 + CLAUDE.md HNeRV parity discipline + Mallat
scattering council position: MNeRV adds multi-scale (3-level hierarchical)
processing to the NeRV substrate. Where Lane 12-v2 NeRV-as-renderer produces
RGB at native 384×512 in one pass, MNeRV decomposes the synthesis across:

  - Scale 1 (1/4 res, 96×128): coarse RGB from the latent
  - Scale 2 (1/2 res, 192×256): refinement using upsampled scale 1 + latent
  - Scale 3 (full res, 384×512): final refinement using upsampled scale 2 + latent

The hierarchical decomposition is a Mallat-scattering-style cascade where
each scale acts as a bandpass: low frequencies at scale 1, mid frequencies
at scale 2 residual, high frequencies at scale 3 residual. This composes
orthogonally with numpy_inverse_dwt for inflate-time multi-scale synthesis.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin with 16-byte fixed header +
    4 length-prefixed sections (decoder_blob_scale_1_2_3 INT8+brotli,
    scale_table FP16, latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_MNERV in this module with
    schema_keys_in_order = MNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200
    LOC contest-hermetic (hierarchical decoder ≤3× single-scale)
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: mnerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern with
    differentiable rgb_to_yuv6 + load_differentiable_scorers + Lagrangian
  bolt_on_loc_budget: substrate_engineering (full multi-scale renderer)
  no_op_detector_planned: export_to_archive returns sha256; trainer wires
    Phase 1 packet compiler integration in Phase B

CLAUDE.md compliance
--------------------
- Full RGB renderer at the contest's camera resolution (HNeRV parity lesson 5)
- Score-aware training via the Lane 12-v2 train_step pattern (lesson 1)
- Archive grammar declared at module level (lesson 2 + Catalog #124)
- Inflate budget ≤200 LOC (lesson 4 substrate-engineering waiver)
- Eval-roundtrip-aware via the trainer's train_step (lesson 8)

Cross-references
----------------
- Lane 12-v2 substrate: :mod:`tac.lane_12_v2_nerv_as_renderer`
- Phase 2 pre-design (T15 FiLM, T17 VQ codebook, T18 nonlinear transform)
- Mallat scattering: ``src/tac/wavelet_*`` modules (orthogonal composition)
- HNeRV retrospective: ``feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md``
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


__all__ = [
    "MNERV_MAGIC",
    "MNERV_FORMAT_VERSION",
    "ARCHIVE_GRAMMAR_MNERV",
    "MNeRVConfig",
    "MNeRVRenderer",
    "MNeRVLatentTable",
    "mnerv_train_step",
    "default_mnerv_seg_surrogate",
    "default_mnerv_pose_surrogate",
    "export_mnerv_to_archive",
]


MNERV_MAGIC: bytes = b"MNRV"
MNERV_FORMAT_VERSION: int = 1


ARCHIVE_GRAMMAR_MNERV: dict = {
    "format_version": MNERV_FORMAT_VERSION,
    "magic": MNERV_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("version", "<H", 2),
                ("latent_dim", "<H", 2),
                ("n_pairs", "<H", 2),
                ("base_channels", "<H", 2),
                ("n_scales", "<H", 2),
                ("reserved", "<H", 2),
            ],
        },
        {
            "name": "decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_3_scale_cascade",
        },
        {
            "name": "scale_table",
            "offset_after": "decoder_blob",
            "length_field_le_u32": True,
            "kind": "fp16_raw_one_per_schema_entry",
        },
        {
            "name": "latent_blob",
            "offset_after": "scale_table",
            "length_field_le_u32": True,
            "kind": "brotli_uint8_delta_zigzag",
        },
        {
            "name": "sidecar_blob",
            "offset_after": "latent_blob",
            "length_field_le_u32": True,
            "kind": "brotli_optional_phase_b",
            "phase_a_empty": True,
        },
    ],
    "schema_keys_in_order": "MNeRVRenderer.SCHEMA",
    "n_scales": 3,
    "scale_resolutions": [(96, 128), (192, 256), (384, 512)],
    "predicted_total_bytes": "165_000 to 200_000 [predicted; multi-scale 3-level vs PR100 174_786]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MNeRVConfig:
    """Frozen MNeRV multi-scale renderer config.

    Defaults: 3 scales (96×128 → 192×256 → 384×512), latent_dim=16,
    base_channels=24 (smaller per scale because 3 scales share work).
    """

    latent_dim: int = 16
    base_channels: int = 24
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    n_pairs: int = 600
    frames_per_pair: int = 2
    n_scales: int = 3
    cuda_required: bool = True
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_scales != 3:
            # Phase A pins 3 scales; Mallat scattering canonical bandpass count.
            raise ValueError(f"Phase A pins n_scales=3, got {self.n_scales}")
        if self.frames_per_pair != 2:
            raise ValueError(f"frames_per_pair must be 2, got {self.frames_per_pair}")


# ── Multi-scale renderer ──────────────────────────────────────────────────


class _MNeRVScaleStage(nn.Module):
    """One scale of the MNeRV hierarchical synthesis.

    Each stage consumes (z, prev_x) and outputs the next-scale feature map.
    """

    def __init__(self, latent_dim: int, prev_channels: int, out_channels: int,
                 out_size: tuple[int, int]) -> None:
        super().__init__()
        self.out_size = out_size
        # Latent → spatial map at out_size via linear stem + reshape.
        self.latent_proj = nn.Linear(latent_dim, out_channels)
        # Combine previous scale (if any) + latent.
        in_ch = prev_channels + out_channels if prev_channels > 0 else out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, prev: torch.Tensor | None) -> torch.Tensor:
        # Broadcast latent → spatial map at out_size.
        latent_spatial = self.latent_proj(z).unsqueeze(-1).unsqueeze(-1)
        latent_spatial = latent_spatial.expand(-1, -1, self.out_size[0], self.out_size[1])
        if prev is not None:
            prev_up = F.interpolate(
                prev, size=self.out_size, mode="bilinear", align_corners=False,
            )
            x = torch.cat([prev_up, latent_spatial], dim=1)
        else:
            x = latent_spatial
        # Residual: fused features + latent broadcast (Mallat scattering-style).
        return self.fuse(x) + latent_spatial


class MNeRVRenderer(nn.Module):
    """MNeRV multi-scale hierarchical RGB renderer.

    Forward signature: ``z (B, latent_dim) → (B, 2, 3, eval_h, eval_w)``.

    Architecture (3 scales, Mallat-scattering-style bandpass cascade):
      - Scale 1 (96, 128):  z → feature map at 1/4 res
      - Scale 2 (192, 256): scale_1 + z → feature map at 1/2 res
      - Scale 3 (384, 512): scale_2 + z → feature map at full res → 2× RGB heads
    """

    def __init__(self, config: MNeRVConfig) -> None:
        super().__init__()
        self.config = config

        C = config.base_channels
        # Per-scale resolutions (1/4, 1/2, 1×).
        eh, ew = config.eval_size
        scales: list[tuple[int, int]] = [
            (eh // 4, ew // 4),
            (eh // 2, ew // 2),
            (eh, ew),
        ]
        self.scales = scales

        self.stage_1 = _MNeRVScaleStage(config.latent_dim, 0, C, scales[0])
        self.stage_2 = _MNeRVScaleStage(config.latent_dim, C, C, scales[1])
        self.stage_3 = _MNeRVScaleStage(config.latent_dim, C, C, scales[2])

        self.rgb_0 = nn.Conv2d(C, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(C, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
            )
        x_1 = self.stage_1(z, None)
        x_2 = self.stage_2(z, x_1)
        x_3 = self.stage_3(z, x_2)
        f0 = torch.sigmoid(self.rgb_0(x_3)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x_3)) * 255.0
        return torch.stack([f0, f1], dim=1)

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        """Pinned state-dict (key, shape) order for archive packing."""
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        ordered_keys = [
            "stage_1.latent_proj.weight", "stage_1.latent_proj.bias",
            "stage_1.fuse.0.weight", "stage_1.fuse.0.bias",
            "stage_1.fuse.2.weight", "stage_1.fuse.2.bias",
            "stage_2.latent_proj.weight", "stage_2.latent_proj.bias",
            "stage_2.fuse.0.weight", "stage_2.fuse.0.bias",
            "stage_2.fuse.2.weight", "stage_2.fuse.2.bias",
            "stage_3.latent_proj.weight", "stage_3.latent_proj.bias",
            "stage_3.fuse.0.weight", "stage_3.fuse.0.bias",
            "stage_3.fuse.2.weight", "stage_3.fuse.2.bias",
            "rgb_0.weight", "rgb_0.bias",
            "rgb_1.weight", "rgb_1.bias",
        ]
        for key in ordered_keys:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        return out


# ── Latent table ──────────────────────────────────────────────────────────


class MNeRVLatentTable(nn.Module):
    """Per-pair learned latent embedding (mirrors Lane 12-v2 pattern)."""

    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── Score-aware loss step (mirrors Lane 12-v2 train_step pattern) ────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 bottleneck (STE) per CLAUDE.md eval_roundtrip non-negotiable."""
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        return output["pose"]
    if not torch.is_tensor(output):
        raise TypeError(f"pose output must be a tensor, got {type(output).__name__}")
    return output


def mnerv_train_step(
    *,
    renderer: MNeRVRenderer,
    latent_table: MNeRVLatentTable,
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
    """Score-aware training step for MNeRV (mirrors Lane 12-v2 contract).

    Identical loss surface as Lane 12-v2 except the renderer is MNeRV (multi-
    scale hierarchical) instead of single-scale PixelShuffle.
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false. Use the dedicated probe test."
        )
    z = latent_table(pair_indices)
    decoded = renderer(z)  # (B, 2, 3, H_native, W_native)

    B, F_pp, C, H_native, W_native = decoded.shape
    flat = decoded.reshape(B * F_pp, C, H_native, W_native)
    H_cam, W_cam = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(flat, size=(H_cam, W_cam), mode="bicubic", align_corners=False)

    up_uint8_ste = _eval_roundtrip_uint8_clamp(up)
    up_pairs = up_uint8_ste.reshape(B, F_pp, C, H_cam, W_cam)
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
        "loss": loss, "loss_seg": loss_seg, "loss_pose": loss_pose,
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
    }


def default_mnerv_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    """Default seg surrogate: KL on logits (Hinton T=2 distillation)."""
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_mnerv_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    """Default pose surrogate: MSE on first 6 dims."""
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])


# ── Archive export ────────────────────────────────────────────────────────


def _quantize_per_tensor_int8_with_fp16_scale(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(tensor.abs().max().item())
    scale = max(max_abs, 1e-8) / 127.0
    scale_fp16 = torch.tensor([scale], dtype=torch.float16)
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale_fp16


def _quantize_latent_table_uint8_delta(latents: torch.Tensor) -> bytes:
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


def export_mnerv_to_archive(
    *,
    renderer: MNeRVRenderer,
    latent_table: MNeRVLatentTable,
    output_path: Path,
) -> str:
    """Pack trained MNeRV + latents into monolithic 0.bin (returns sha256)."""
    import brotli

    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != renderer config {expected}")

    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from renderer state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        int8_chunks.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    decoder_blob = brotli.compress(b"".join(int8_chunks), quality=11)
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta(
        latent_table.embedding.weight.detach().cpu()
    )
    sidecar_blob = b""

    out = io.BytesIO()
    # Header (16 bytes)
    out.write(MNERV_MAGIC)
    out.write(struct.pack("<H", MNERV_FORMAT_VERSION))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.base_channels))
    out.write(struct.pack("<H", config.n_scales))
    out.write(struct.pack("<H", 0))  # reserved
    # Sections
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(scale_table)))
    out.write(scale_table)
    out.write(struct.pack("<I", len(latent_blob)))
    out.write(latent_blob)
    out.write(struct.pack("<I", len(sidecar_blob)))
    out.write(sidecar_blob)

    archive_bytes = out.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)
    return hashlib.sha256(archive_bytes).hexdigest()
