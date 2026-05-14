# SPDX-License-Identifier: MIT
"""HiNeRV-as-renderer — hierarchical NeRV substrate.

Per operator directive 2026-05-11 (NeRV-family expansion) + CLAUDE.md HNeRV
parity discipline (lesson 5: full RGB renderer; lesson 4: inflate ≤200 LOC).
HiNeRV is a sister to MNeRV: both decompose synthesis across multiple scales,
but where MNeRV uses a Mallat scattering bandpass cascade, HiNeRV uses an
**explicit learned hierarchy** with separate RGB heads at each scale and a
GROUP of stage decoders that EACH receive the latent.

Why hierarchical
----------------
- **Coarse-to-fine training**: per-scale RGB heads provide an auxiliary loss
  at each level, forcing low-frequency content to be captured by lower levels
  and high-frequency content by higher levels — a strong inductive bias.
- **Multi-scale supervision**: training loss is a weighted sum over scales,
  encouraging the network to be useful at every scale.
- **Sister to MNeRV but distinct**: MNeRV applies the SAME decoder iteratively
  across scales (Mallat scattering); HiNeRV uses SEPARATE per-stage decoders
  + per-stage RGB heads (more params but also more expressive).

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair.
- 3 hierarchical stages:
    Stage 1: latent → (3, 96, 128) RGB at 1/4 resolution
    Stage 2: latent + upsampled stage 1 features → (3, 192, 256) at 1/2
    Stage 3: latent + upsampled stage 2 features → (3, 384, 512) full res
- Per-stage RGB heads (rgb_0_s1, rgb_1_s1, rgb_0_s2, …) — 6 RGB heads total.
- Inference: returns final-scale (B, 2, 3, 384, 512) RGB.
- Training: returns intermediate-scale RGBs as auxiliaries for multi-scale loss.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header +
    4 length-prefixed sections: decoder_stack INT8+brotli, scale_table FP16,
    latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_HINERV in this module with
    schema_keys_in_order = HiNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200
    LOC contest-hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: hinerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern + per-stage
    auxiliary losses
  bolt_on_loc_budget: substrate_engineering (full hierarchical renderer)
  no_op_detector_planned: export_to_archive returns sha256

CLAUDE.md compliance — same as Lane 12-v2.

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; explicit hierarchical decomposition
gives ~1.4× param efficiency at fixed bytes per Hu 2024 HiNeRV ablations;
sister to MNeRV scattering paradigm but with separate per-stage decoders +
per-stage RGB supervision; pose-axis marginal 2.71× SegNet at PR106 r2
frontier]``. NOT a score claim until [contest-CUDA] anchor lands.

Format ID: 0x63 (NeRV-family expansion magic 0xFE).
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


# ── Magic + format ────────────────────────────────────────────────────────


HINERV_MAGIC: bytes = b"HiNV"
HINERV_FORMAT_VERSION: int = 1
HINERV_FORMAT_ID: int = 0x63


# ── Archive grammar ──────────────────────────────────────────────────────


ARCHIVE_GRAMMAR_HINERV: dict = {
    "format_version": HINERV_FORMAT_VERSION,
    "format_id": HINERV_FORMAT_ID,
    "magic": HINERV_MAGIC.decode("ascii"),
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
                ("n_pairs", "<H", 2),
                ("base_channels", "<H", 2),
                ("n_levels", "<H", 2),
            ],
        },
        {
            "name": "decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven_per_level",
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
    "schema_keys_in_order": "HiNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "180_000 to 220_000 [predicted; per-level decoders ~3× single-scale params]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HiNeRVConfig:
    """Frozen config for HiNeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim
        Per-pair latent dim.
    base_channels
        Per-stage decoder base channel width.
    n_pairs
        Number of per-pair latents.
    n_levels
        Number of hierarchical stages. Default 3 (1/4, 1/2, 1).
    eval_size
        Final native render size. Default (384, 512).
    base_h, base_w
        Initial spatial dims after stem (smallest scale resolution AFTER stem
        but BEFORE per-stage upsamples). Default (24, 32) → after 2 stages
        becomes 48×64 then 96×128.
    aux_loss_weights
        Per-level auxiliary loss weights for training. Last level always 1.0;
        earlier levels are advisory weights for the multi-scale loss.
    lambda_seg, lambda_pose
        Score-aware loss weights.
    cuda_required
        If True (default), train_step raises if CUDA unavailable.
    """

    latent_dim: int = 16
    base_channels: int = 24
    n_pairs: int = 600
    n_levels: int = 3
    eval_size: tuple[int, int] = (384, 512)
    base_h: int = 96
    base_w: int = 128
    aux_loss_weights: tuple[float, ...] = (0.25, 0.5, 1.0)
    frames_per_pair: int = 2
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_levels < 2:
            raise ValueError(f"n_levels must be >=2, got {self.n_levels}")
        if len(self.aux_loss_weights) != self.n_levels:
            raise ValueError(
                f"aux_loss_weights len {len(self.aux_loss_weights)} != n_levels {self.n_levels}"
            )
        if self.aux_loss_weights[-1] <= 0:
            raise ValueError("final-level loss weight must be positive")
        if self.frames_per_pair != 2:
            raise ValueError("Phase A pinned at frames_per_pair=2")
        # Final size must equal stem * 2**(n_levels - 1).
        expected_final_h = self.base_h * (2 ** (self.n_levels - 1))
        expected_final_w = self.base_w * (2 ** (self.n_levels - 1))
        if (expected_final_h, expected_final_w) != self.eval_size:
            raise ValueError(
                f"eval_size {self.eval_size} != base * 2**(n_levels-1) "
                f"= ({expected_final_h}, {expected_final_w})"
            )


# ── Per-stage decoder ─────────────────────────────────────────────────────


class _HiNeRVStageDecoder(nn.Module):
    """One hierarchical stage: combines (latent, prev features) → features + RGB(2 frames)."""

    def __init__(self, latent_dim: int, prev_channels: int | None,
                 out_channels: int, init_h: int, init_w: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.init_h = init_h
        self.init_w = init_w
        # Stem from latent (flat) → (out_channels, init_h, init_w)
        self.stem = nn.Linear(latent_dim, out_channels * init_h * init_w)
        # Optional projection from prev-stage features (already at this stage's resolution)
        self.has_prev = prev_channels is not None
        if self.has_prev:
            self.prev_proj = nn.Conv2d(prev_channels, out_channels, 1)
        # Two refinement convs.
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # Two RGB heads (one per frame in pair).
        self.rgb_0 = nn.Conv2d(out_channels, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(out_channels, 3, 3, padding=1)

    def forward(self, z: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (features, rgb_pair) where rgb_pair: (B, 2, 3, H, W)."""
        B = z.shape[0]
        x = self.stem(z).view(B, self.out_channels, self.init_h, self.init_w)
        x = torch.sin(x)
        if self.has_prev and prev is not None:
            # Prev arrives already at this stage's resolution (caller resamples).
            x = x + self.prev_proj(prev)
        x = torch.sin(self.conv1(x))
        x = torch.sin(self.conv2(x)) + x  # residual
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return x, torch.stack([f0, f1], dim=1)


class HiNeRVRenderer(nn.Module):
    """HiNeRV — hierarchical multi-stage NeRV with per-stage RGB heads.

    Forward signature: ``z (B, latent_dim) → (B, 2, 3, H_final, W_final)`` by
    default. ``forward_with_aux=True`` returns the list of per-stage RGB
    outputs for multi-scale training.
    """

    def __init__(self, config: HiNeRVConfig) -> None:
        super().__init__()
        self.config = config
        # Channel taper: deeper stages can use slightly fewer channels per pixel
        # since spatial coverage scales (Hu 2024 HiNeRV recommendation).
        c_taper = [config.base_channels for _ in range(config.n_levels)]
        c_taper[-1] = max(8, int(config.base_channels * 0.75))
        self.stages = nn.ModuleList()
        for i in range(config.n_levels):
            scale = 2 ** i  # i=0: 1x base; i=1: 2x; i=2: 4x.
            stage_h = config.base_h * scale
            stage_w = config.base_w * scale
            prev_channels = c_taper[i - 1] if i > 0 else None
            self.stages.append(
                _HiNeRVStageDecoder(
                    latent_dim=config.latent_dim,
                    prev_channels=prev_channels,
                    out_channels=c_taper[i],
                    init_h=stage_h,
                    init_w=stage_w,
                )
            )

    def forward(self, z: torch.Tensor,
                forward_with_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
            )
        prev_features: torch.Tensor | None = None
        all_rgbs: list[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            if prev_features is not None:
                # Upsample previous features to this stage's resolution.
                stage_h = self.config.base_h * (2 ** i)
                stage_w = self.config.base_w * (2 ** i)
                prev_features = F.interpolate(
                    prev_features, size=(stage_h, stage_w),
                    mode="bilinear", align_corners=False,
                )
            features, rgb = stage(z, prev_features)
            prev_features = features
            all_rgbs.append(rgb)
        if forward_with_aux:
            return all_rgbs[-1], all_rgbs
        return all_rgbs[-1]

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        for i in range(self.config.n_levels):
            keys = [
                f"stages.{i}.stem.weight", f"stages.{i}.stem.bias",
            ]
            if i > 0:
                keys += [f"stages.{i}.prev_proj.weight", f"stages.{i}.prev_proj.bias"]
            keys += [
                f"stages.{i}.conv1.weight", f"stages.{i}.conv1.bias",
                f"stages.{i}.conv2.weight", f"stages.{i}.conv2.bias",
                f"stages.{i}.rgb_0.weight", f"stages.{i}.rgb_0.bias",
                f"stages.{i}.rgb_1.weight", f"stages.{i}.rgb_1.bias",
            ]
            for key in keys:
                if key in sd:
                    out.append((key, tuple(sd[key].shape)))
        return out


# ── Latent table ─────────────────────────────────────────────────────────


class HiNeRVLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── train_step (multi-scale auxiliary loss) ──────────────────────────────


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


def train_step_hinerv(
    *,
    renderer: HiNeRVRenderer,
    latent_table: HiNeRVLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    aux_loss_weights: tuple[float, ...] | None = None,
    eval_roundtrip: bool = True,
) -> dict:
    """Multi-scale score-aware training step.

    The scorer always sees the FINAL stage's RGB upsampled to camera res. The
    auxiliary intermediate-stage RGBs are pixel-supervised (L1 against GT
    downsampled to that stage's resolution) for hierarchical training signal.
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z = latent_table(pair_indices)
    final_decoded, all_rgbs = renderer(z, forward_with_aux=True)

    # Score-aware loss (final scale through scorer).
    B, F_pp, C, H_native, W_native = final_decoded.shape
    flat = final_decoded.reshape(B * F_pp, C, H_native, W_native)
    H_camera, W_camera = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(
        flat, size=(H_camera, W_camera), mode="bicubic", align_corners=False
    )
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
    loss_score = loss_seg + loss_pose

    # Auxiliary multi-scale L1 loss.
    weights = aux_loss_weights if aux_loss_weights is not None else renderer.config.aux_loss_weights
    aux_loss = torch.zeros(1, device=z.device, dtype=z.dtype).squeeze()
    aux_breakdown: list[float] = []
    for i, (rgb_stage, w) in enumerate(zip(all_rgbs, weights[:-1])):
        # Downsample GT to this stage's spatial size.
        stage_h, stage_w = rgb_stage.shape[-2:]
        gt_flat = gt_pairs_uint8.float().reshape(B * F_pp, C, H_camera, W_camera)
        gt_at_stage = F.interpolate(
            gt_flat, size=(stage_h, stage_w), mode="bicubic", align_corners=False
        ).reshape(B, F_pp, C, stage_h, stage_w)
        l1 = F.l1_loss(rgb_stage, gt_at_stage)
        aux_loss = aux_loss + float(w) * l1
        aux_breakdown.append(float(l1.detach()))

    loss = loss_score + aux_loss
    return {
        "loss": loss,
        "loss_seg": loss_seg,
        "loss_pose": loss_pose,
        "loss_score": loss_score,
        "loss_aux": aux_loss,
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
        "aux_breakdown": aux_breakdown,
    }


# ── Quantization + archive packing ───────────────────────────────────────


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


def export_hinerv_to_archive(
    *,
    renderer: HiNeRVRenderer,
    latent_table: HiNeRVLatentTable,
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
    out.write(HINERV_MAGIC)
    out.write(struct.pack("<H", HINERV_FORMAT_VERSION))
    out.write(struct.pack("<H", HINERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.base_channels))
    out.write(struct.pack("<H", config.n_levels))
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


# ── Smoke + default surrogates ──────────────────────────────────────────


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:hinerv_phase_a_scaffold_smoke_test_only"""
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    gt_pairs = torch.randint(
        0, 256, (batch_size, 2, 3, 874, 1164), generator=g, dtype=torch.uint8,
    )
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
