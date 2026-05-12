"""NeRVdc-as-renderer — NeRV with explicit decoder-conditioning substrate.

Per operator directive 2026-05-11 ("all nerv-family"). NeRVdc adds
PER-PAIR DECODER CONDITIONING beyond the standard (t, latent) input —
specifically a previous-frame conditioning vector that the decoder
consumes through a side-input. This breaks the i.i.d.-frame
independence assumption that limits Lane 12-v2 + KK's NeRV-family
substrates and exploits temporal correlation in driving video.

Why decoder conditioning
------------------------
- **Temporal smoothness**: driving video has high temporal coherence.
  Conditioning the decoder on the previous pair's reconstruction lets
  the model exploit this without paying for a recurrent latent.
- **Compose-with-other-NeRV**: the conditioning hook is a side-input
  that doesn't disturb the substrate's monolithic 0.bin contract.
- **No archive cost**: the conditioning vector is computed
  DETERMINISTICALLY at inflate-time (previous-frame reconstruction
  passed to the next pair's decode); zero archive bytes added.

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair as in Lane 12-v2.
- Decoder: stem(latent) + side-input(prev_frame_summary) → PixelShuffle stack.
- Conditioning summary: 1×1 conv-pool over the previous pair's last frame
  produces a (B, cond_dim=8) vector concatenated with latent before stem.
- Output: (B, 2, 3, 384, 512) RGB.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header
    + 4 length-prefixed sections: decoder_blob INT8+brotli,
    scale_table FP16, latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_NERVDC in this module with
    schema_keys_in_order = NeRVdcRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: nervdc_phase_a_monolithic_singlefile_0bin
  score_aware_loss: train_step_nervdc with diff rgb_to_yuv6 +
    load_differentiable_scorers + Lagrangian
  bolt_on_loc_budget: substrate_engineering (full conditioned NeRV)
  no_op_detector_planned: export_nervdc_to_archive returns sha256

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; decoder-conditioning exploits
temporal coherence in driving video for ~1.15× param efficiency at
fixed bytes per Wang 2024 NeRVdc; pose-axis marginal 2.71× SegNet at
PR106 r2 frontier]``. NOT a score claim until [contest-CUDA] anchor.

Format ID: 0x66.
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


NERVDC_MAGIC: bytes = b"NRVc"
NERVDC_FORMAT_VERSION: int = 1
NERVDC_FORMAT_ID: int = 0x66


ARCHIVE_GRAMMAR_NERVDC: dict = {
    "format_version": NERVDC_FORMAT_VERSION,
    "format_id": NERVDC_FORMAT_ID,
    "magic": NERVDC_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("version", "<H", 2),
                ("format_id", "<H", 2),
                ("latent_dim", "<H", 2),
                ("cond_dim", "<H", 2),
                ("n_pairs", "<H", 2),
                ("base_channels", "<H", 2),
            ],
        },
        {
            "name": "decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven",
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
            "kind": "brotli_uint8_asym_delta_split",
        },
        {
            "name": "sidecar_blob",
            "offset_after": "latent_blob",
            "length_field_le_u32": True,
            "kind": "brotli_optional_phase_b",
            "phase_a_empty": True,
        },
    ],
    "schema_keys_in_order": "NeRVdcRenderer.SCHEMA",
    "predicted_total_bytes": "150_000 to 180_000 [predicted; conditioning is decoder-internal]",
}


@dataclass(frozen=True)
class NeRVdcConfig:
    """Frozen config for NeRVdc-as-renderer Phase A."""
    latent_dim: int = 16
    cond_dim: int = 8
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
        if self.cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {self.cond_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_stages != 6:
            raise ValueError(f"Phase A pinned at n_stages=6, got {self.n_stages}")
        if self.frames_per_pair != 2:
            raise ValueError(f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}")


class _CondSummary(nn.Module):
    """Maps (B, 3, H, W) prev-frame → (B, cond_dim) conditioning vector.

    Pure-decoder side; no encoder; the prev-frame is always available at
    inflate-time (output of the previous pair's decode).
    """

    def __init__(self, cond_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(8, cond_dim)

    def forward(self, prev_frame_uint8: torch.Tensor) -> torch.Tensor:
        x = prev_frame_uint8.float() / 127.5 - 1.0
        h = self.stem(x).flatten(1)
        return self.head(h)


class NeRVdcRenderer(nn.Module):
    """NeRVdc decoder. Forward: ``(z, prev_frame) → (B, 2, 3, H, W)``."""

    def __init__(self, config: NeRVdcConfig) -> None:
        super().__init__()
        self.config = config
        self.cond_summary = _CondSummary(config.cond_dim)
        C = config.base_channels
        self.channels: list[int] = [
            C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)
        ]
        self.stem = nn.Linear(
            config.latent_dim + config.cond_dim,
            self.channels[0] * config.base_h * config.base_w,
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

    def forward(
        self, z: torch.Tensor, prev_frame_uint8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}); got {tuple(z.shape)}"
            )
        B = z.shape[0]
        if prev_frame_uint8 is None:
            # First-pair / no-prev case: use zero conditioning.
            cond = torch.zeros((B, self.config.cond_dim), dtype=z.dtype, device=z.device)
        else:
            if prev_frame_uint8.dim() != 4 or prev_frame_uint8.shape[1] != 3:
                raise ValueError(
                    f"prev_frame_uint8 expected (B, 3, H, W); got "
                    f"{tuple(prev_frame_uint8.shape)}"
                )
            cond = self.cond_summary(prev_frame_uint8)
        z_cond = torch.cat([z, cond], dim=-1)
        x = self.stem(z_cond).view(
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
            "cond_summary.stem.0.weight", "cond_summary.stem.0.bias",
            "cond_summary.head.weight", "cond_summary.head.bias",
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


class NeRVdcLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

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


def train_step_nervdc(
    *,
    renderer: NeRVdcRenderer,
    latent_table: NeRVdcLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    prev_frames_uint8: torch.Tensor | None,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
) -> dict:
    """Score-aware training step for NeRVdc.

    Caller supplies prev_frames_uint8 (the second frame of the previous
    pair, downsampled to camera resolution). At inflate-time this is
    deterministically reconstructed from the previous pair's decode.
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z = latent_table(pair_indices)
    decoded = renderer(z, prev_frame_uint8=prev_frames_uint8)
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


def export_nervdc_to_archive(
    *,
    renderer: NeRVdcRenderer,
    latent_table: NeRVdcLatentTable,
    output_path: Path,
) -> str:
    import brotli
    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != expected {expected}")

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
    sidecar_blob = b""

    out = io.BytesIO()
    out.write(NERVDC_MAGIC)
    out.write(struct.pack("<H", NERVDC_FORMAT_VERSION))
    out.write(struct.pack("<H", NERVDC_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.cond_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.base_channels))
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


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:nervdc_phase_a_scaffold_smoke_test_only"""
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
