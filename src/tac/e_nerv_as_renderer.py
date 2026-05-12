"""E-NeRV-as-renderer — Encoder-NeRV substrate (NeRV-family completion).

Per operator directive 2026-05-11 ("all nerv-family"). Sister to KK's
NeRV-Enc/Dec bolt-on (`tac.nerv_enc_dec_separated`) but as a STANDALONE
substrate where the encoder is part of the substrate identity (not a
compose-able add-on). E-NeRV uses an explicit per-frame learned encoder
to map raw frames → latent at COMPRESS-TIME ONLY; at inflate-time the
decoder consumes pre-quantized latents from the archive.

Why E-NeRV as a substrate
-------------------------
KK's NeRV-Enc/Dec bolt-on lets ANY NeRV-family decoder accept a learned
latent producer. E-NeRV makes the encoder a FIRST-CLASS substrate
parameter so the trainer co-optimizes (encoder, decoder, latent
quantization) jointly. The separation matters at the operating point
because:

* Per-pair latents from a global encoder share representation across
  the dataset (Buchanan 2024 §3 — implicit codebook regularization).
* Joint encoder+decoder training avoids the latent-table independence
  assumption that limits Lane 12-v2 + KK's NeRV.
* Per CLAUDE.md HNeRV parity discipline lesson 5: full RGB renderer.
  E-NeRV's encoder is COMPRESS-TIME ONLY (NOT shipped); at inflate-time
  the decoder + latent table is the renderer.

Architecture (default config)
-----------------------------
- Encoder (compress-time only): 4 stride-2 conv stages + GroupNorm +
  global-avg-pool + linear head — maps (B, 2, 3, H, W) → (B, latent_dim).
  Encoder params NOT in archive; the latent_table from encoded outputs
  is what ships.
- Latent: (B, 16) per-pair, populated by encoder at compress-time, then
  quantized for archive.
- Decoder: PixelShuffle stack (Lane 12-v2 style) — (B, 16) → (B, 2, 3, H, W).
- Output: (B, 2, 3, 384, 512) RGB.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header
    + 4 length-prefixed sections: decoder_blob INT8+brotli,
    scale_table FP16, latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_E_NERV in this module with
    schema_keys_in_order = ENeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: e_nerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: train_step_e_nerv with diff rgb_to_yuv6 +
    load_differentiable_scorers + Lagrangian (encoder COMPRESS-TIME ONLY)
  bolt_on_loc_budget: substrate_engineering (full encoder+decoder
    NeRV substrate)
  no_op_detector_planned: export_e_nerv_to_archive returns sha256

CLAUDE.md compliance (lesson-by-lesson)
---------------------------------------
- L1 (score-aware): trainer routes through train_step with SegNet+PoseNet.
- L2 (export-first): ARCHIVE_GRAMMAR_E_NERV declared at module level.
- L4 (inflate ≤ 200 LOC): substrate-engineering target.
- L5 (full RGB renderer): forward returns (B, 2, 3, H, W).
- L6 (score-domain Lagrangian): train_step delegates to Lagrangian via
  same surrogate signatures.
- L8 (eval-roundtrip): train_step simulates uint8 STE before scorer.
- L11 (no-op detector): export_to_archive returns sha256.
- L13 (KILL is last resort): N/A.

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; co-optimized encoder+decoder
gives ~1.2× param efficiency at fixed bytes vs decoder-only-with-
table NeRV per Chen 2024 ENeRV; pose-axis marginal 2.71× SegNet at
PR106 r2 frontier]``. NOT a score claim until [contest-CUDA] anchor.

Format ID: 0x65 (NeRV-family completion magic 0xFE).
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


E_NERV_MAGIC: bytes = b"ENRV"
E_NERV_FORMAT_VERSION: int = 1
E_NERV_FORMAT_ID: int = 0x65


# ── Archive grammar (parser-section manifest) ────────────────────────────


ARCHIVE_GRAMMAR_E_NERV: dict = {
    "format_version": E_NERV_FORMAT_VERSION,
    "format_id": E_NERV_FORMAT_ID,
    "magic": E_NERV_MAGIC.decode("ascii"),
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
                ("reserved", "<H", 2),
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
    "schema_keys_in_order": "ENeRVRenderer.SCHEMA",
    "predicted_total_bytes": "150_000 to 180_000 [predicted; encoder NOT in archive]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ENeRVConfig:
    """Frozen config for E-NeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim
        Per-pair latent dim. Default 16 mirrors Lane 12-v2.
    base_channels
        NeRV decoder base channel width.
    encoder_base_channels
        Encoder base channel width (compress-time only; not in archive).
    base_h, base_w
        Initial spatial dims after stem. Default (6, 8).
    eval_size
        Native render size (H, W). Default (384, 512).
    n_pairs
        Number of per-pair latents.
    n_stages
        PixelShuffle stages. Default 6.
    encoder_n_stages
        Encoder stride-2 stages. Default 4.
    lambda_seg, lambda_pose
        Score-aware loss weights.
    cuda_required
        If True (default), train_step raises if CUDA unavailable.
    """

    latent_dim: int = 16
    base_channels: int = 36
    encoder_base_channels: int = 32
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    n_pairs: int = 600
    n_stages: int = 6
    encoder_n_stages: int = 4
    frames_per_pair: int = 2
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.encoder_base_channels <= 0:
            raise ValueError(
                f"encoder_base_channels must be positive, got {self.encoder_base_channels}"
            )
        if self.n_stages != 6:
            raise ValueError(f"Phase A pinned at n_stages=6, got {self.n_stages}")
        if self.encoder_n_stages < 2:
            raise ValueError(
                f"encoder_n_stages must be >= 2, got {self.encoder_n_stages}"
            )
        if self.frames_per_pair != 2:
            raise ValueError(f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}")


# ── Encoder (compress-time only — NOT in archive) ────────────────────────


class ENeRVEncoder(nn.Module):
    """Compress-time encoder: (B, 2, 3, H, W) → (B, latent_dim).

    Per CLAUDE.md "Strict scorer rule"-style discipline: this module is
    COMPRESS-TIME ONLY. The encoder IS NOT serialized into the archive;
    only the encoded latents (post-quantization) are. At inflate-time the
    decoder consumes the latent table directly.

    Architecture: stride-2 conv stack + GroupNorm + global-avg-pool +
    linear head. Sister to `tac.nerv_enc_dec_separated.NeRVEncoder` but
    explicitly tied to the E-NeRV substrate's `latent_dim`.
    """

    def __init__(self, config: ENeRVConfig) -> None:
        super().__init__()
        self.config = config
        c = config.encoder_base_channels
        # 2-frame pair → 6-channel input.
        layers: list[nn.Module] = [
            nn.Conv2d(6, c, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=c),
            nn.SiLU(),
        ]
        ch = c
        for _ in range(config.encoder_n_stages - 1):
            layers.extend([
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=ch * 2),
                nn.SiLU(),
            ])
            ch *= 2
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.stem = nn.Sequential(*layers)
        self.head = nn.Linear(ch, config.latent_dim)

    def forward(self, frame_pair_uint8: torch.Tensor) -> torch.Tensor:
        """``frame_pair_uint8 (B, 2, 3, H, W) → (B, latent_dim)``.

        Input frames are uint8 [0, 255]; we float them and rescale into
        [-1, 1] before the conv stack for activation stability.
        """
        if frame_pair_uint8.dim() != 5:
            raise ValueError(
                f"forward expected (B, 2, 3, H, W); got {tuple(frame_pair_uint8.shape)}"
            )
        B, T, C, H, W = frame_pair_uint8.shape
        if T != 2 or C != 3:
            raise ValueError(
                f"forward expected (B, 2, 3, H, W); got T={T} C={C}"
            )
        x = frame_pair_uint8.float() / 127.5 - 1.0
        # Stack the 2 frames along channel dim → 6-channel.
        x6 = x.reshape(B, T * C, H, W)
        h = self.stem(x6).flatten(1)  # (B, ch)
        return self.head(h)


# ── Renderer (decoder side; ships in archive) ────────────────────────────


class ENeRVRenderer(nn.Module):
    """E-NeRV decoder. Forward: ``z (B, latent_dim) → (B, 2, 3, H, W)``."""

    def __init__(self, config: ENeRVConfig) -> None:
        super().__init__()
        self.config = config
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
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}); got {tuple(z.shape)}"
            )
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.config.base_h, self.config.base_w)
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
        keys = ["stem.weight", "stem.bias"]
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


# ── Latent table (populated by encoder at compress-time) ─────────────────


class ENeRVLatentTable(nn.Module):
    """Per-pair latent table; populated by the encoder during training.

    At training time, after a few warmup epochs of joint encoder+decoder
    training, the trainer FREEZES the encoder, runs it once over each
    of the n_pairs pairs to populate the table, then continues fine-
    tuning the decoder + table jointly. Only the table is shipped.
    """

    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)

    def populate_from_encoder(
        self, encoder: ENeRVEncoder, all_frame_pairs_uint8: torch.Tensor,
    ) -> None:
        """Run the encoder once on all pairs to seed the latent table.

        This is the canonical "freeze encoder, transfer to table" step.
        """
        if all_frame_pairs_uint8.dim() != 5:
            raise ValueError(
                f"all_frame_pairs_uint8 expected (N, 2, 3, H, W); got "
                f"{tuple(all_frame_pairs_uint8.shape)}"
            )
        n = all_frame_pairs_uint8.shape[0]
        if n != self.embedding.weight.shape[0]:
            raise ValueError(
                f"all_frame_pairs_uint8 batch={n} != n_pairs={self.embedding.weight.shape[0]}"
            )
        encoder.eval()
        with torch.no_grad():
            BATCH = 8
            for start in range(0, n, BATCH):
                end = min(start + BATCH, n)
                z = encoder(all_frame_pairs_uint8[start:end])
                self.embedding.weight.data[start:end] = z


# ── train_step ────────────────────────────────────────────────────────────


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


def train_step_e_nerv(
    *,
    encoder: ENeRVEncoder,
    renderer: ENeRVRenderer,
    latent_table: ENeRVLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
    use_encoder: bool = True,
) -> dict:
    """Score-aware training step for E-NeRV.

    When ``use_encoder=True``, the encoder produces fresh latents for the
    batch (joint encoder+decoder training). When False, the latent table
    is the source (post-encoder-freeze fine-tuning).
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    if use_encoder:
        z = encoder(gt_pairs_uint8)
    else:
        z = latent_table(pair_indices)
    decoded = renderer(z)
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


# ── Quantization + archive packing (encoder NOT in archive) ──────────────


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


def export_e_nerv_to_archive(
    *,
    renderer: ENeRVRenderer,
    latent_table: ENeRVLatentTable,
    output_path: Path,
) -> str:
    """Pack decoder + latent table (NOT encoder) into monolithic 0.bin.

    Per HNeRV parity discipline lesson 5: the renderer is the decoder +
    latent table only. The encoder is COMPRESS-TIME ONLY and never
    shipped.
    """
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
    out.write(E_NERV_MAGIC)
    out.write(struct.pack("<H", E_NERV_FORMAT_VERSION))
    out.write(struct.pack("<H", E_NERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.base_channels))
    out.write(struct.pack("<H", 0))  # reserved (16-byte header alignment)
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


# ── Smoke synthetic helper + default surrogates ──────────────────────────


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:e_nerv_phase_a_scaffold_smoke_test_only"""
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
