# SPDX-License-Identifier: MIT
"""VQ-VAE-as-full-renderer substrate.

Per operator directive 2026-05-11 + HNeRV parity discipline lesson 5 (full
RGB renderer) + van den Oord 2017 (VQ-VAE) council position. T17 added a
SHARED VQ codebook as a BOLT-ON to T1's substrate (codebook-as-side-table
inside an existing renderer). THIS module makes VQ-VAE the FULL renderer:
codebook indices → renderer → RGB.

Architecture: encoder maps (z, t) → discrete codebook indices, decoder maps
codebook entries → RGB at full camera resolution. The codebook is the
PRIMARY representation; the renderer is conditioned solely on the looked-up
codebook entries.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin with 16-byte fixed header +
    5 length-prefixed sections (codebook_blob FP16 raw, decoder_blob
    INT8+brotli, scale_table FP16, indices_blob brotli uint16, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_VQVAE_FULL in this module with
    schema_keys_in_order = VQVAEFullRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — ≤200 LOC hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: vqvae_full_renderer_phase_a_monolithic_singlefile_0bin
  score_aware_loss: vqvae_train_step uses load_differentiable_scorers
    PoseNet + SegNet via Lagrangian λ_seg + λ_pose + commitment loss
  bolt_on_loc_budget: substrate_engineering (full codebook+decoder renderer)
  no_op_detector_planned: export_vqvae_to_archive returns sha256

van den Oord 2017 codebook EMA (canonical persistent-EMA buffer form):
  decay = 0.99 per VQ-VAE §3.2 (DIFFERENT from CLAUDE.md weight-EMA 0.997)
  N_i' = γ·N_i + (1-γ)·n_i      (count EMA)
  m_i' = γ·m_i + (1-γ)·sum_i    (sum EMA)
  e_i' = m_i' / N_i'            (entry update from running mean)

NN-2 perplexity gate (CLAUDE.md Phase 2 pre-design):
  perplexity ≥ 0.4 · num_entries (= 102.4 for N=256)
  Codebook-collapse pauses training; operator inspects for diagnostic context.

CLAUDE.md compliance
--------------------
- Full RGB renderer at camera resolution (HNeRV parity lesson 5)
- Score-aware Lagrangian (lesson 1)
- Archive grammar declared at module level (lesson 2 + Catalog #124)
- Codebook EMA at 0.99 (canonical VQ-VAE exception); WEIGHT EMA stays 0.997
- Eval-roundtrip-aware via the trainer's train_step (lesson 8)

Cross-references
----------------
- T17 shared codebook BOLT-ON: :mod:`tac.shared_vq_codebook`
- van den Oord 2017 — VQ-VAE persistent EMA codebook
- Lane 12-v2 substrate: :mod:`tac.lane_12_v2_nerv_as_renderer` (sister full renderer)
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
    "VQVAE_FULL_MAGIC",
    "VQVAE_FULL_FORMAT_VERSION",
    "ARCHIVE_GRAMMAR_VQVAE_FULL",
    "VQVAEFullConfig",
    "VQVAEFullEncoder",
    "VQVAEFullDecoder",
    "VQVAEFullRenderer",
    "VQVAEFullLatentTable",
    "vqvae_train_step",
    "default_vqvae_seg_surrogate",
    "default_vqvae_pose_surrogate",
    "compute_perplexity_from_indices",
    "assert_codebook_perplexity_ok",
    "VQVAECodebookCollapseError",
    "export_vqvae_to_archive",
]


VQVAE_FULL_MAGIC: bytes = b"VQVR"
VQVAE_FULL_FORMAT_VERSION: int = 1


ARCHIVE_GRAMMAR_VQVAE_FULL: dict = {
    "format_version": VQVAE_FULL_FORMAT_VERSION,
    "magic": VQVAE_FULL_MAGIC.decode("ascii"),
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
                ("num_entries", "<H", 2),
                ("entry_dim", "<H", 2),
                ("n_pairs", "<H", 2),
                ("tokens_per_pair", "<H", 2),
            ],
        },
        {
            "name": "codebook_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "fp16_raw_num_entries_x_entry_dim",
        },
        {
            "name": "decoder_blob",
            "offset_after": "codebook_blob",
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
            "name": "indices_blob",
            "offset_after": "scale_table",
            "length_field_le_u32": True,
            "kind": "brotli_uint16_per_token_per_pair",
        },
        {
            "name": "sidecar_blob",
            "offset_after": "indices_blob",
            "length_field_le_u32": True,
            "kind": "brotli_optional_phase_b",
            "phase_a_empty": True,
        },
    ],
    "schema_keys_in_order": "VQVAEFullRenderer.SCHEMA",
    "num_entries_default": 256,
    "entry_dim_default": 64,
    "codebook_ema_decay": 0.99,
    "nn2_perplexity_floor_ratio": 0.4,
    "predicted_total_bytes": "170_000 to 200_000 [predicted; 32KB codebook + ~130KB decoder + ~10KB indices]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VQVAEFullConfig:
    """VQ-VAE-as-full-renderer config.

    Defaults: 256-entry codebook × 64-dim @ FP16 (~32 KB), tokens_per_pair=8.
    """

    latent_dim: int = 16
    num_entries: int = 256
    entry_dim: int = 64
    tokens_per_pair: int = 8
    n_pairs: int = 600
    frames_per_pair: int = 2
    base_channels: int = 36
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    codebook_ema_decay: float = 0.99
    commitment_weight: float = 0.25
    nn2_perplexity_floor_ratio: float = 0.4
    cuda_required: bool = True
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.num_entries <= 0 or self.num_entries > 65535:
            raise ValueError(f"num_entries must be in (0, 65535], got {self.num_entries}")
        if self.entry_dim <= 0:
            raise ValueError(f"entry_dim must be positive, got {self.entry_dim}")
        if self.tokens_per_pair <= 0:
            raise ValueError(f"tokens_per_pair must be positive, got {self.tokens_per_pair}")
        if not (0.9 <= self.codebook_ema_decay < 1.0):
            raise ValueError(
                f"codebook_ema_decay must be in [0.9, 1.0) — van den Oord §3.2; "
                f"got {self.codebook_ema_decay}"
            )
        if not (0.0 <= self.nn2_perplexity_floor_ratio <= 1.0):
            raise ValueError(
                f"nn2_perplexity_floor_ratio must be in [0, 1], got "
                f"{self.nn2_perplexity_floor_ratio}"
            )
        if self.frames_per_pair != 2:
            raise ValueError(f"frames_per_pair must be 2, got {self.frames_per_pair}")


# ── Exceptions ────────────────────────────────────────────────────────────


class VQVAECodebookCollapseError(RuntimeError):
    """Raised by NN-2 gate when codebook perplexity < floor."""


# ── Encoder: latent → codebook indices ───────────────────────────────────


class VQVAEFullEncoder(nn.Module):
    """Maps per-pair latent → tokens_per_pair tokens of entry_dim each.

    Forward signature: ``z (B, latent_dim) → z_e (B, tokens_per_pair, entry_dim)``.
    """

    def __init__(self, config: VQVAEFullConfig) -> None:
        super().__init__()
        self.config = config
        # Linear projection to tokens × entry_dim.
        self.proj = nn.Linear(
            config.latent_dim, config.tokens_per_pair * config.entry_dim,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        out = self.proj(z)
        return out.view(B, self.config.tokens_per_pair, self.config.entry_dim)


# ── Codebook (van den Oord persistent EMA buffers) ───────────────────────


class _VQVAECodebook(nn.Module):
    """VQ-VAE codebook with persistent EMA buffers (van den Oord §3.2).

    Entries are BUFFERS, not parameters. Updated via explicit ``update_ema``
    after each training step. Forward implements straight-through quantization.
    """

    def __init__(self, num_entries: int, entry_dim: int, decay: float) -> None:
        super().__init__()
        self.num_entries = num_entries
        self.entry_dim = entry_dim
        self.decay = decay
        # Initialize entries from N(0, 1) for diversity.
        self.register_buffer("codebook", torch.randn(num_entries, entry_dim))
        self.register_buffer("ema_count", torch.zeros(num_entries))
        self.register_buffer("ema_sum", torch.zeros(num_entries, entry_dim))

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``z_e (..., entry_dim) → (z_q_ste, indices, commitment_loss)``."""
        flat = z_e.reshape(-1, self.entry_dim)
        # Squared distances to every entry.
        d = (flat.pow(2).sum(dim=1, keepdim=True)
             - 2 * flat @ self.codebook.t()
             + self.codebook.pow(2).sum(dim=1, keepdim=True).t())
        indices = d.argmin(dim=1)
        z_q = self.codebook[indices].reshape(z_e.shape)
        # Commitment loss: encoder commits to codebook.
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        # Straight-through: forward = z_q; backward = z_e.
        z_q_ste = z_e + (z_q - z_e).detach()
        return z_q_ste, indices.reshape(z_e.shape[:-1]), commitment_loss

    def update_ema(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """Update persistent EMA buffers (van den Oord §3.2)."""
        with torch.no_grad():
            flat_z = z_e.reshape(-1, self.entry_dim).detach()
            flat_idx = indices.reshape(-1)
            one_hot = F.one_hot(flat_idx, num_classes=self.num_entries).float()
            n_per = one_hot.sum(dim=0)
            sum_per = one_hot.t() @ flat_z
            self.ema_count.mul_(self.decay).add_(n_per, alpha=1 - self.decay)
            self.ema_sum.mul_(self.decay).add_(sum_per, alpha=1 - self.decay)
            # Laplace smoothing to avoid division by zero.
            laplace = 1e-5
            n = self.num_entries
            denom = self.ema_count + laplace
            denom = denom * (self.ema_count.sum() + n * laplace) / (
                self.ema_count.sum() + n * laplace
            )
            self.codebook.copy_(self.ema_sum / denom.unsqueeze(-1).clamp(min=1e-8))


# ── Decoder: codebook entries → RGB ──────────────────────────────────────


class VQVAEFullDecoder(nn.Module):
    """Decodes (tokens, entry_dim) codebook entries → RGB (2, 3, H, W).

    Architecture: linear stem fuses all tokens into a base spatial map,
    then PixelShuffle upsamples to camera resolution. Mirrors Lane 12-v2's
    decoder structure but conditioned on quantized codebook entries.
    """

    def __init__(self, config: VQVAEFullConfig) -> None:
        super().__init__()
        self.config = config

        C = config.base_channels
        # Channel taper similar to Lane 12-v2.
        self.channels: list[int] = [
            C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5),
        ]
        # Stem: flat tokens × entry_dim → channel_0 × base_h × base_w.
        flat_in = config.tokens_per_pair * config.entry_dim
        self.stem = nn.Linear(flat_in, self.channels[0] * config.base_h * config.base_w)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(6):  # 6 stages: 6×8 → 384×512
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
        self.ps = nn.PixelShuffle(2)

        final_ch = self.channels[-1]
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """``z_q (B, tokens, entry_dim) → (B, 2, 3, H, W)``."""
        B = z_q.shape[0]
        flat = z_q.reshape(B, -1)  # (B, tokens × entry_dim)
        x = self.stem(flat).view(
            B, self.channels[0], self.config.base_h, self.config.base_w,
        )
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)


# ── Full renderer (encoder + codebook + decoder) ─────────────────────────


class VQVAEFullRenderer(nn.Module):
    """Full VQ-VAE renderer: (latent → encoder → codebook → decoder → RGB).

    The codebook lookup IS the representation; the renderer is conditioned
    solely on the quantized codebook entries.
    """

    def __init__(self, config: VQVAEFullConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = VQVAEFullEncoder(config)
        self.codebook = _VQVAECodebook(
            config.num_entries, config.entry_dim, config.codebook_ema_decay,
        )
        self.decoder = VQVAEFullDecoder(config)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``z (B, latent_dim) → (decoded RGB, indices, commitment_loss)``."""
        z_e = self.encoder(z)
        z_q_ste, indices, commitment_loss = self.codebook(z_e)
        decoded = self.decoder(z_q_ste)
        return decoded, indices, commitment_loss

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        """Pinned (key, shape) order for archive packing of DECODER + ENCODER weights.

        The CODEBOOK is stored as a buffer separately (FP16 raw section).
        """
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        # Encoder weights.
        for key in ["encoder.proj.weight", "encoder.proj.bias"]:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        # Decoder weights: stem, 6 blocks/skips, 2 RGB heads.
        ordered: list[str] = ["decoder.stem.weight", "decoder.stem.bias"]
        for i in range(6):
            ordered.append(f"decoder.blocks.{i}.weight")
            ordered.append(f"decoder.blocks.{i}.bias")
            if isinstance(self.decoder.skips[i], nn.Conv2d):
                ordered.append(f"decoder.skips.{i}.weight")
                ordered.append(f"decoder.skips.{i}.bias")
        ordered.extend([
            "decoder.rgb_0.weight", "decoder.rgb_0.bias",
            "decoder.rgb_1.weight", "decoder.rgb_1.bias",
        ])
        for key in ordered:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        return out


# ── Latent table ──────────────────────────────────────────────────────────


class VQVAEFullLatentTable(nn.Module):
    """Per-pair latent embedding (mirrors Lane 12-v2 / MNeRV pattern)."""

    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── Score-aware loss step ────────────────────────────────────────────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    """Simulate uint8 bottleneck STE per CLAUDE.md eval_roundtrip non-negotiable."""
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        return output["pose"]
    if not torch.is_tensor(output):
        raise TypeError(f"pose output must be tensor, got {type(output).__name__}")
    return output


def vqvae_train_step(
    *,
    renderer: VQVAEFullRenderer,
    latent_table: VQVAEFullLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    commitment_weight: float = 0.25,
    eval_roundtrip: bool = True,
) -> dict:
    """Score-aware step + commitment loss (van den Oord §3.2)."""
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false. Use the dedicated probe test."
        )
    z = latent_table(pair_indices)
    decoded, indices, commitment_loss = renderer(z)

    B, F_pp, C, H_native, W_native = decoded.shape
    flat = decoded.reshape(B * F_pp, C, H_native, W_native)
    H_cam, W_cam = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(flat, size=(H_cam, W_cam), mode="bicubic", align_corners=False)

    up_uint8_ste = _eval_roundtrip_uint8_clamp(up)
    up_pairs = up_uint8_ste.reshape(B, F_pp, C, H_cam, W_cam)
    gt_pairs = gt_pairs_uint8.float()

    seg_pred = scorer_seg(scorer_seg.preprocess_input(up_pairs))
    with torch.no_grad():
        seg_target = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred, seg_target)

    pose_pred = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(up_pairs)))
    with torch.no_grad():
        pose_target = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(gt_pairs)))
    loss_pose_unweighted = pose_surrogate(pose_pred, pose_target)

    loss_seg = lambda_seg * loss_seg_unweighted
    loss_pose = lambda_pose * loss_pose_unweighted
    loss = loss_seg + loss_pose + commitment_weight * commitment_loss

    return {
        "loss": loss, "loss_seg": loss_seg, "loss_pose": loss_pose,
        "commitment_loss": commitment_loss,
        "indices": indices.detach(),
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
    }


def default_vqvae_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_vqvae_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])


# ── NN-2 perplexity gate (CLAUDE.md Phase 2 pre-design) ──────────────────


def compute_perplexity_from_indices(
    indices: torch.Tensor, num_entries: int,
) -> float:
    """Codebook perplexity = exp(H(P)) where P is empirical index distribution."""
    counts = torch.bincount(indices.reshape(-1), minlength=num_entries).float()
    p = counts / counts.sum().clamp(min=1)
    p = p[p > 0]  # avoid log(0)
    h = -(p * torch.log(p)).sum()
    return float(torch.exp(h).item())


def assert_codebook_perplexity_ok(
    indices: torch.Tensor,
    num_entries: int,
    floor_ratio: float = 0.4,
) -> dict:
    """NN-2: codebook-collapse gate (per CLAUDE.md Phase 2 pre-design)."""
    perplexity = compute_perplexity_from_indices(indices, num_entries)
    floor = floor_ratio * num_entries
    diag = {
        "perplexity": float(perplexity),
        "floor": float(floor),
        "floor_ratio": float(floor_ratio),
        "num_entries": int(num_entries),
        "passed": perplexity >= floor,
    }
    if not diag["passed"]:
        raise VQVAECodebookCollapseError(
            f"[vqvae] NN-2 codebook collapse: perplexity {perplexity:.1f} < "
            f"floor {floor:.1f} (ratio {floor_ratio} × N={num_entries}). "
            "Trainer paused for operator inspection. Re-init dead entries "
            "from recent encoder outputs and re-run; do not silently continue."
        )
    return diag


# ── Archive export ────────────────────────────────────────────────────────


def _quantize_per_tensor_int8_with_fp16_scale(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(tensor.abs().max().item())
    scale = max(max_abs, 1e-8) / 127.0
    scale_fp16 = torch.tensor([scale], dtype=torch.float16)
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale_fp16


def _encode_indices_to_bytes(indices_per_pair: torch.Tensor) -> bytes:
    """Pack (n_pairs, tokens_per_pair) uint16 indices → brotli bytes."""
    import brotli

    flat = indices_per_pair.reshape(-1).to(torch.int32).clamp(0, 65535)
    arr = flat.to(torch.int32).numpy().astype("<u2")  # little-endian uint16
    return brotli.compress(arr.tobytes(), quality=11)


def export_vqvae_to_archive(
    *,
    renderer: VQVAEFullRenderer,
    latent_table: VQVAEFullLatentTable,
    output_path: Path,
) -> str:
    """Pack VQ-VAE substrate into monolithic 0.bin (returns sha256).

    The codebook indices for all 600 pairs are computed at export time by
    a forward pass through the encoder + codebook (no_grad).
    """
    import brotli

    config = renderer.config

    # 1. Codebook (FP16 raw): num_entries × entry_dim.
    codebook_fp16 = renderer.codebook.codebook.detach().cpu().to(torch.float16)
    codebook_blob = codebook_fp16.numpy().tobytes()

    # 2. Decoder + encoder weights as INT8 per-tensor + FP16 scale.
    schema = renderer.schema
    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from state_dict")
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

    # 3. Indices for all pairs (computed via encoder + codebook).
    renderer.eval()
    with torch.no_grad():
        all_indices = []
        for start in range(0, config.n_pairs, 64):
            stop = min(start + 64, config.n_pairs)
            batch_ids = torch.arange(start, stop)
            z = latent_table(batch_ids.to(latent_table.embedding.weight.device))
            z_e = renderer.encoder(z)
            _, idx, _ = renderer.codebook(z_e)
            all_indices.append(idx.detach().cpu())
        indices_per_pair = torch.cat(all_indices, dim=0)
    indices_blob = _encode_indices_to_bytes(indices_per_pair)

    sidecar_blob = b""

    out = io.BytesIO()
    # Header (16 bytes)
    out.write(VQVAE_FULL_MAGIC)
    out.write(struct.pack("<H", VQVAE_FULL_FORMAT_VERSION))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.num_entries))
    out.write(struct.pack("<H", config.entry_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.tokens_per_pair))
    # Sections
    out.write(struct.pack("<I", len(codebook_blob)))
    out.write(codebook_blob)
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(scale_table)))
    out.write(scale_table)
    out.write(struct.pack("<I", len(indices_blob)))
    out.write(indices_blob)
    out.write(struct.pack("<I", len(sidecar_blob)))
    out.write(sidecar_blob)

    archive_bytes = out.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)
    return hashlib.sha256(archive_bytes).hexdigest()
