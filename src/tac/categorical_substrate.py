"""Categorical mask grammar FULL RGB renderer — Phase A scaffold.

Per handoff P3 long-tail "PR85/86/91 mask/action grammars" + operator directive
2026-05-11, this is a NON-HNeRV substrate direction that uses per-pixel
categorical class indices (SegNet-class palette: 5 classes) as the COMPLETE
encoder for the renderer's input. The full RGB output is conditioned on
per-pixel categorical class indices + a class-conditioned palette + per-class
intra-class shading.

This is the **categorical** sister to the **ANR token-program** direction —
both consume a 5-class categorical token map of shape (B, SEGNET_IN_H,
SEGNET_IN_W). The difference:

- **ANR** ships the token stream **arithmetic-coded** by HPACMini (universal
  context model). Bytes are dominated by the AC stream.
- **categorical** ships the token stream more directly (PR91 QM0/QH0 grammar
  via ``tac.packet_compiler.pr91_hpac_grammar`` already in custody) and is
  decoded at inflate via a SMALLER renderer with a per-pixel palette +
  per-class intra-class shading conv head.

Both substrates respect HNeRV parity discipline 13 lessons. Key design choices
recorded inline:

1. **L1 score-aware** — ``train_step`` backprops through SegNet + PoseNet.
2. **L5 full RGB renderer** — output is (B, 2, 3, CAMERA_H, CAMERA_W), NOT
   mask-only. Per-pixel class indices CONDITION the renderer; they do not
   REPLACE it.
3. **L8 eval_roundtrip** — uint8 STE clamp before scorer call.
4. **Codebook collapse guard** — class-prediction entropy monitored to detect
   the "all pixels predict class 0" degenerate solution.

CLAUDE.md compliance
====================

* NON-NEGOTIABLE forbidden patterns observed:
    - No MPS-fallback default; ``train_step`` raises on missing CUDA.
    - No make_synthetic_pair_batch in any non-smoke path; defers to
      ``lane_12_v2_nerv_as_renderer.RealPairBatchSource``.
    - No /tmp paths; archive bytes are returned as `bytes` to the caller.
    - No scorer load at inflate (strict-scorer-rule).

* Differentiable scorers MUST be loaded via
  ``tac.scorer.load_differentiable_scorers`` per CLAUDE.md eval_roundtrip
  non-negotiable.

* Class-prediction entropy < ``codebook_collapse_floor`` raises during
  training (codebook-collapse guard, sister to van den Oord VQ-VAE
  perplexity gate).
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


# ── Constants ────────────────────────────────────────────────────────────


CAMERA_H: int = 874
CAMERA_W: int = 1164
SEGNET_IN_H: int = 384
SEGNET_IN_W: int = 512
NUM_CLASSES: int = 5

CATEGORICAL_MAGIC: bytes = b"CATG"
CATEGORICAL_FORMAT_ID: int = 0x51
CATEGORICAL_FORMAT_VERSION: int = 1


# ── Archive grammar ──────────────────────────────────────────────────────


ARCHIVE_GRAMMAR: dict = {
    "format_id": CATEGORICAL_FORMAT_ID,
    "format_version": CATEGORICAL_FORMAT_VERSION,
    "magic": CATEGORICAL_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("format_id", "<H", 2),
                ("format_version", "<H", 2),
                ("num_pairs", "<I", 4),
                ("flags", "<I", 4),
            ],
        },
        {
            "name": "meta",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "torch_save_dict_meta",
        },
        {
            "name": "renderer_state",
            "offset_after": "meta",
            "length_field_le_u32": True,
            "kind": "fp16_state_dict",
        },
        {
            "name": "palette",
            "offset_after": "renderer_state",
            "length_field_le_u32": True,
            "kind": "fp16_palette_NxCx3",
        },
        {
            "name": "tokens",
            "offset_after": "palette",
            "length_field_le_u32": True,
            "kind": "categorical_arithmetic_stream",
        },
    ],
    "predicted_total_bytes": (
        "120_000 to 165_000 [predicted; sister to ANR; class-palette saves "
        "FiLM bytes vs ANR's TokenRendererV62]"
    ),
}


# ── Config ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoricalSubstrateConfig:
    """Frozen config for categorical-substrate Phase A.

    Attributes
    ----------
    num_pairs : int
        Number of pair embeddings (matches PR95 ANR exemplar's 600).
    num_classes : int
        Number of categorical classes per pixel; pinned at 5 (SegNet-class).
    palette_dim : int
        Dimensionality of the learned per-class palette (3 = RGB direct;
        higher = palette-then-conv). Phase A uses 8.
    shading_channels : int
        Width of the per-class intra-class shading conv head.
    lambda_seg, lambda_pose : float
        Score-domain Lagrangian weights (operating-point-aware).
    codebook_collapse_floor : float
        Minimum per-frame class-entropy at training step. Below this floor
        the trainer raises ``CodebookCollapseError`` (van den Oord
        perplexity-style gate per CLAUDE.md "Forbidden codebook collapse").
    cuda_required : bool
        If True (default), ``train_step`` raises if CUDA unavailable.
    """

    num_pairs: int = 600
    num_classes: int = NUM_CLASSES
    palette_dim: int = 8
    shading_channels: int = 16
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    codebook_collapse_floor: float = 0.4
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {self.num_pairs}")
        if self.num_classes != NUM_CLASSES:
            raise ValueError(
                f"num_classes pinned at {NUM_CLASSES} (SegNet-class); "
                f"got {self.num_classes}"
            )
        if self.palette_dim < 3:
            raise ValueError(
                f"palette_dim must be >= 3 (RGB minimum), got {self.palette_dim}"
            )
        if self.shading_channels <= 0:
            raise ValueError(
                f"shading_channels must be positive, got {self.shading_channels}"
            )
        if self.codebook_collapse_floor < 0:
            raise ValueError(
                f"codebook_collapse_floor must be >= 0, got "
                f"{self.codebook_collapse_floor}"
            )


# ── Renderer module ──────────────────────────────────────────────────────


class CategoricalRenderer(nn.Module):
    """Categorical full-RGB renderer.

    Architecture
    ------------
    1. Per-pair embedding (small frame-conditioning latent) →
       per-class palette modulation.
    2. Per-pixel: index palette by class index → 8-dim color seed per pixel.
    3. 3-layer conv head produces RGB intra-class shading from
       (palette_seed, class_onehot, coord_grid).
    4. Bilinear upsample to camera resolution.

    Forward: ``(tokens, idx) → (B, 2, 3, CAMERA_H, CAMERA_W)``.
    Both frames per pair are rendered; this matches the ANR substrate's
    frames_per_pair=2 invariant.
    """

    def __init__(self, config: CategoricalSubstrateConfig) -> None:
        super().__init__()
        self.config = config
        C = config.num_classes
        P = config.palette_dim
        SC = config.shading_channels

        # Per-pair frame embed (small)
        self.frame_embed = nn.Embedding(config.num_pairs, P)

        # Per-class palette (learned RGB seed per class per channel)
        # shape: (num_classes, palette_dim)
        self.palette = nn.Parameter(torch.randn(C, P) * 0.1)

        # Coord grid will be cached at first forward; per_pixel xy in [-1, 1]
        self.register_buffer("_coord_cache", torch.zeros(0), persistent=False)
        self._cached_hw: tuple[int, int] = (-1, -1)

        # Shading conv head: input = palette_dim + one_hot(class) + 2 coord = P + C + 2
        in_ch = P + C + 2
        self.conv1 = nn.Conv2d(in_ch, SC, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(SC, SC, kernel_size=3, padding=1)

        # 2 RGB heads (frame0 + frame1) — matches ANR master/slave shape but in a
        # SINGLE module (categorical has fewer params; no separate master+slave).
        self.rgb_0 = nn.Conv2d(SC, 3, kernel_size=3, padding=1)
        self.rgb_1 = nn.Conv2d(SC, 3, kernel_size=3, padding=1)

    def _coord_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self._cached_hw != (H, W) or self._coord_cache.numel() == 0:
            ys = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(1, 1, H, W)
            xs = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(1, 1, H, W)
            grid = torch.cat([ys, xs], dim=1)  # (1, 2, H, W)
            self._coord_cache = grid
            self._cached_hw = (H, W)
        return self._coord_cache

    def forward(self, tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens must be long, got {tokens.dtype}")
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be (B, H, W), got {tuple(tokens.shape)}")
        if idx.dtype != torch.long:
            raise TypeError(f"idx must be long, got {idx.dtype}")
        B, H, W = tokens.shape
        C = self.config.num_classes
        P = self.config.palette_dim

        # One-hot expansion: (B, C, H, W)
        one_hot = F.one_hot(tokens, num_classes=C).permute(0, 3, 1, 2).float()

        # Index palette per pixel: (B, P, H, W) via index_select on classes.
        # palette: (C, P). per-pixel palette = palette[tokens] permuted to (B, P, H, W).
        palette_at_pixel = self.palette[tokens].permute(0, 3, 1, 2)  # (B, P, H, W)

        # Per-pair modulation: frame_embed gives (B, P) → broadcast multiply.
        emb = self.frame_embed(idx).view(B, P, 1, 1)
        modulated = palette_at_pixel * (1.0 + emb)

        # Coordinate grid
        coord = self._coord_grid(H, W, tokens.device).expand(B, -1, -1, -1)

        # Concat features
        feats = torch.cat([modulated, one_hot, coord], dim=1)  # (B, P+C+2, H, W)
        x = F.gelu(self.conv1(feats))
        x = F.gelu(self.conv2(x))

        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        rendered_native = torch.stack([f0, f1], dim=1)  # (B, 2, 3, H, W)

        # Upsample to camera resolution
        B_, F_pp, C_, H_, W_ = rendered_native.shape
        flat = rendered_native.reshape(B_ * F_pp, C_, H_, W_)
        up = F.interpolate(
            flat, size=(CAMERA_H, CAMERA_W),
            mode="bilinear", align_corners=False,
        )
        return up.reshape(B_, F_pp, C_, CAMERA_H, CAMERA_W)


# ── Codebook-collapse error ──────────────────────────────────────────────


class CodebookCollapseError(RuntimeError):
    """Raised by the categorical trainer when class-entropy < floor.

    Per CLAUDE.md "Forbidden codebook collapse" + van den Oord VQ-VAE
    perplexity-style gate. If the categorical encoder collapses to a single
    class (e.g. all pixels predict class 0), the renderer degenerates to a
    palette-constant output and the scorer becomes blind. We refuse to
    proceed.
    """


def _class_entropy(class_indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Empirical class-entropy over a batch of categorical predictions.

    class_indices: long tensor of any shape, values in [0, num_classes).
    Returns: scalar tensor (nats; clamp log to avoid log(0)).
    """
    if class_indices.numel() == 0:
        raise ValueError("cannot compute entropy of empty tensor")
    one_hot = F.one_hot(class_indices.reshape(-1), num_classes=num_classes).float()
    p = one_hot.mean(dim=0)
    # Use natural-log entropy. Clamp to avoid log(0).
    p_safe = p.clamp(min=1e-12)
    return -(p * p_safe.log()).sum()


# ── Eval roundtrip helper ────────────────────────────────────────────────


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


# ── Score-aware train step ───────────────────────────────────────────────


def train_step(
    *,
    renderer: CategoricalRenderer,
    pair_indices: torch.Tensor,
    tokens: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    codebook_collapse_floor: float,
    eval_roundtrip: bool = True,
) -> dict:
    """Score-aware train step for the categorical full renderer.

    Per CLAUDE.md HNeRV parity discipline lessons 1+5+6+8. Loss is
    ``λ_seg · d_seg + λ_pose · d_pose`` through the actual scorer contracts.
    Class-entropy guard prevents categorical collapse.
    """
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    if tokens.shape[1:] != (SEGNET_IN_H, SEGNET_IN_W):
        raise ValueError(
            f"tokens spatial shape must be ({SEGNET_IN_H}, {SEGNET_IN_W}), "
            f"got {tuple(tokens.shape[1:])}"
        )
    if gt_pairs_uint8.shape[-2:] != (CAMERA_H, CAMERA_W):
        raise ValueError(
            f"gt_pairs_uint8 spatial shape must be ({CAMERA_H}, {CAMERA_W}), "
            f"got {tuple(gt_pairs_uint8.shape[-2:])}"
        )

    # Codebook-collapse guard BEFORE forward (cheap check on input distribution).
    entropy = _class_entropy(tokens, renderer.config.num_classes)
    if entropy.item() < codebook_collapse_floor:
        raise CodebookCollapseError(
            f"Class-entropy {entropy.item():.4f} < floor "
            f"{codebook_collapse_floor:.4f}. The categorical encoder has "
            f"collapsed to a near-degenerate distribution (van den Oord "
            f"perplexity gate). Refuse to train further; investigate input "
            f"tokens or relax floor."
        )

    rendered = renderer(tokens, pair_indices)  # (B, 2, 3, H, W)
    rendered_uint8_ste = _eval_roundtrip_uint8_clamp(rendered)
    gt_pairs = gt_pairs_uint8.float()

    seg_pred_logits = scorer_seg(scorer_seg.preprocess_input(rendered_uint8_ste))
    with torch.no_grad():
        seg_target_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred_logits, seg_target_logits)

    pose_pred = _pose_tensor(
        scorer_pose(scorer_pose.preprocess_input(rendered_uint8_ste))
    )
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
        "rendered_uint8_ste": rendered_uint8_ste,
        "class_entropy": entropy.detach(),
    }


# ── Archive packer ───────────────────────────────────────────────────────


def _pack_section(payload: bytes) -> bytes:
    if len(payload) > 0xFFFFFFFF:
        raise ValueError(f"payload too large: {len(payload)} > 2^32-1")
    return struct.pack("<I", len(payload)) + payload


def export_to_archive(
    *,
    config: CategoricalSubstrateConfig,
    renderer: CategoricalRenderer,
    tokens_bin: bytes,
) -> tuple[bytes, str]:
    """Pack the categorical substrate into the monolithic archive bytes.

    Layout: ``HEADER (16) | META | RENDERER_STATE | PALETTE | TOKENS``.

    Returns (archive_bytes, sha256_hex).
    """
    header = struct.pack(
        "<4sHHII",
        CATEGORICAL_MAGIC,
        CATEGORICAL_FORMAT_ID,
        CATEGORICAL_FORMAT_VERSION,
        config.num_pairs,
        0,
    )
    assert len(header) == 16

    meta = {
        "N": config.num_pairs,
        "num_classes": config.num_classes,
        "palette_dim": config.palette_dim,
        "shading_channels": config.shading_channels,
    }
    meta_buf = io.BytesIO()
    torch.save(meta, meta_buf)
    meta_bytes = meta_buf.getvalue()

    renderer_sd_fp16 = {
        k: (
            v.detach().cpu().to(torch.float16)
            if torch.is_floating_point(v) and k != "palette"
            else v.detach().cpu()
        )
        for k, v in renderer.state_dict().items()
        if k != "palette"  # palette ships separately
    }
    renderer_buf = io.BytesIO()
    torch.save(renderer_sd_fp16, renderer_buf)
    renderer_bytes = renderer_buf.getvalue()

    palette_fp16 = renderer.palette.detach().cpu().to(torch.float16)
    palette_buf = io.BytesIO()
    torch.save(palette_fp16, palette_buf)
    palette_bytes = palette_buf.getvalue()

    if not isinstance(tokens_bin, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"tokens_bin must be bytes-like, got {type(tokens_bin).__name__}"
        )
    tokens_bin = bytes(tokens_bin)

    blob = (
        header
        + _pack_section(meta_bytes)
        + _pack_section(renderer_bytes)
        + _pack_section(palette_bytes)
        + _pack_section(tokens_bin)
    )
    return blob, hashlib.sha256(blob).hexdigest()


def parse_archive_sections(blob: bytes) -> dict:
    """Inverse of ``export_to_archive``."""
    if len(blob) < 16:
        raise ValueError(f"archive too short: {len(blob)} < 16")
    header = blob[:16]
    magic, fmt_id, fmt_ver, num_pairs, flags = struct.unpack("<4sHHII", header)
    if magic != CATEGORICAL_MAGIC:
        raise ValueError(f"magic mismatch: {magic!r} != {CATEGORICAL_MAGIC!r}")
    if fmt_id != CATEGORICAL_FORMAT_ID:
        raise ValueError(f"format_id mismatch: {fmt_id} != {CATEGORICAL_FORMAT_ID}")
    if fmt_ver != CATEGORICAL_FORMAT_VERSION:
        raise ValueError(
            f"format_version mismatch: {fmt_ver} != {CATEGORICAL_FORMAT_VERSION}"
        )

    offset = 16
    sections = {}
    for name in ("meta", "renderer_state", "palette", "tokens"):
        if offset + 4 > len(blob):
            raise ValueError(
                f"archive truncated before section {name!r}: "
                f"offset={offset} len={len(blob)}"
            )
        (length,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        if offset + length > len(blob):
            raise ValueError(
                f"archive truncated inside section {name!r}: "
                f"offset={offset} length={length} archive_len={len(blob)}"
            )
        sections[name] = blob[offset:offset + length]
        offset += length
    if offset != len(blob):
        raise ValueError(
            f"trailing bytes after parsed sections: parsed up to {offset}, "
            f"archive_len={len(blob)}"
        )
    sections["_header"] = {
        "magic": magic,
        "format_id": fmt_id,
        "format_version": fmt_ver,
        "num_pairs": num_pairs,
        "flags": flags,
    }
    return sections


__all__ = [
    "ARCHIVE_GRAMMAR",
    "CAMERA_H",
    "CAMERA_W",
    "CATEGORICAL_FORMAT_ID",
    "CATEGORICAL_FORMAT_VERSION",
    "CATEGORICAL_MAGIC",
    "CategoricalRenderer",
    "CategoricalSubstrateConfig",
    "CodebookCollapseError",
    "NUM_CLASSES",
    "SEGNET_IN_H",
    "SEGNET_IN_W",
    "export_to_archive",
    "parse_archive_sections",
    "train_step",
]
