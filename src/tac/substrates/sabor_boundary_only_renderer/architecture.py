# SPDX-License-Identifier: MIT
"""SABOR boundary-only renderer architecture.

The substrate factors a per-pair frame into:

1. **Boundary mask** ``(B, 2, H, W)`` boolean tensor — union of (a) Canny-style
   gradient-magnitude edges at threshold ``edge_threshold`` AND (b) SegNet
   argmax 4-neighbor disagreement at the GT input. Typically 1-3% of pixels.
2. **Boundary RGB** ``(B, 2, 3, H, W)`` float in ``[0, 1]`` — the high-fidelity
   per-pixel RGB AT the boundary mask. Stored explicitly in the archive as
   int8 (scale=255).
3. **Class means** ``(5, 3)`` per-SegNet-class mean RGB (learned). Used at
   inflate time to texture-fill interior pixels: ``rgb[h, w] = class_means
   [segnet_argmax[h, w]]`` plus a per-pair small bias correction.
4. **Refinement decoder** — tiny FiLM-conditioned conv block that lifts the
   texture-filled RGB into the final per-pair output, conditioned on a
   per-pair embedding. Decoder weights live in the archive (~10-30 KB).

The renderer is **score-aware**: gradients flow from contest scorers through
the refinement decoder, class means, and per-pair bias to all parameters via
``apply_eval_roundtrip_during_training`` + ``patch_upstream_yuv6_globally``.
The boundary mask itself is a HARD selection (no gradients flow through the
mask; the gradient through interior pixels is unaffected because the boundary
pixels are passed through verbatim).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline":

* L1 score-aware substrate trains against contest video pixels.
* L4 inflate ≤ 200 LOC (substrate_engineering exception); torch + brotli closure.
* L5 architecture is the FULL renderer (RGB out from selected pairs).
* L7 substrate-engineering LOC budget (~900 total).
* L8 eval-roundtrip + diff yuv6 wired in the loss path.
* L12 single-LOC review discipline.

Mallat seat (grand council):
    The boundary set is the high-frequency wavelet-detail-band analog —
    pixels where the gradient operator returns large magnitude carry the
    structural signal. The interior set is the wavelet-approximation-band
    analog — smooth, low-rank, well-modeled by per-class mean + per-pair
    bias. SABOR is a non-wavelet boundary/interior factorization optimized
    for the SegNet argmax distortion contract.

Shannon seat (grand council LEAD):
    The φ1 audit measured H[interior] ≈ 0 conditional on
    SegNet argmax + per-class mean (interior pixels are
    argmax-stable so the SegNet term sees no entropy cost). The boundary
    term carries the entire rate; this is the
    information-theoretic lower bound on a score-aware factorization.

CLAUDE.md compliance:

* No silent device defaults (caller passes device explicitly).
* No scorer loading inside this module.
* No /tmp paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Contest constants — fixed at design time per substrate-grammar discipline.
SABOR_EVAL_H: int = 384
SABOR_EVAL_W: int = 512
SABOR_NUM_FRAMES: int = 1200
SABOR_NUM_PAIRS: int = SABOR_NUM_FRAMES // 2  # 600
SABOR_NUM_SEG_CLASSES: int = 5  # SegNet output classes (contest scorer)


@dataclass(frozen=True)
class SaborBoundaryOnlyConfig:
    """Static design-time parameters for SABOR substrate (L0 SKETCH)."""

    num_pairs: int = SABOR_NUM_PAIRS
    """Number of latent rows (600 for 1200-frame contest video)."""

    output_height: int = SABOR_EVAL_H
    output_width: int = SABOR_EVAL_W

    num_seg_classes: int = SABOR_NUM_SEG_CLASSES
    """SegNet output classes; class_means is (num_seg_classes, 3)."""

    edge_threshold: float = 0.04
    """Canny-style gradient-magnitude threshold (normalized to [0, 1]).

    A pixel is boundary iff
    ``sqrt(grad_x^2 + grad_y^2) > edge_threshold * (max - min)`` per frame.
    Default 0.04 was selected to land in the 1-3% boundary fraction band the
    φ1 audit measured as the SegNet argmax-disagreement fraction at ε=32.
    """

    refinement_hidden: int = 32
    """Hidden channels of the FiLM-conditioned refinement decoder."""

    refinement_blocks: int = 2
    """Number of FiLM-conditioned refinement blocks; small to keep
    archive size and inflate latency bounded."""

    embedding_dim: int = 8
    """Per-pair embedding dimensionality for FiLM conditioning."""

    bias_dim: int = 3
    """Per-pair RGB bias channels (small per-pair color correction)."""

    def __post_init__(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("output_height/output_width must be positive")
        if not (0.0 < self.edge_threshold < 1.0):
            raise ValueError("edge_threshold must be in (0, 1)")
        if self.refinement_hidden <= 0 or self.refinement_blocks <= 0:
            raise ValueError("refinement_hidden/refinement_blocks must be positive")
        if self.num_seg_classes <= 1:
            raise ValueError("num_seg_classes must be > 1 (multi-class segmentation)")


# ---------------------------------------------------------------------------
# Boundary detection (Canny-style gradient magnitude + SegNet argmax 4-nbr)
# ---------------------------------------------------------------------------


def _sobel_gradient_magnitude(rgb_bchw: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel Sobel gradient magnitude on luminance (BCHW -> B1HW).

    Returns gradient magnitude in [0, 1] approximately (clamped after
    sqrt(gx^2 + gy^2) and dividing by per-frame max).
    """
    if rgb_bchw.dim() != 4:
        raise ValueError(f"expected (B, C, H, W); got {tuple(rgb_bchw.shape)}")
    # Rec.601 luminance — gradient is computed on the luminance channel.
    weights = torch.tensor(
        [0.299, 0.587, 0.114], dtype=rgb_bchw.dtype, device=rgb_bchw.device
    )
    y = (rgb_bchw * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=rgb_bchw.dtype,
        device=rgb_bchw.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(y, sobel_x, padding=1)
    gy = F.conv2d(y, sobel_y, padding=1)
    mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    # Per-frame normalization to [0, 1].
    flat = mag.view(mag.shape[0], -1)
    mag_max = flat.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return (mag / mag_max.view(-1, 1, 1, 1)).clamp(0.0, 1.0)


def _segnet_argmax_4nbr_disagreement(seg_argmax_bhw: torch.Tensor) -> torch.Tensor:
    """Return boolean BHW mask of pixels disagreeing with any 4-neighbor.

    A pixel ``(h, w)`` is a boundary iff its class differs from any of its
    4-neighbors. The border pixels are conservatively flagged as boundary.
    """
    if seg_argmax_bhw.dim() != 3:
        raise ValueError(f"expected (B, H, W); got {tuple(seg_argmax_bhw.shape)}")
    # Shifts produce equal-on-most-pixels-except-boundary masks.
    up = seg_argmax_bhw[:, :-1, :] != seg_argmax_bhw[:, 1:, :]
    down = seg_argmax_bhw[:, 1:, :] != seg_argmax_bhw[:, :-1, :]
    left = seg_argmax_bhw[:, :, :-1] != seg_argmax_bhw[:, :, 1:]
    right = seg_argmax_bhw[:, :, 1:] != seg_argmax_bhw[:, :, :-1]
    mask = torch.zeros_like(seg_argmax_bhw, dtype=torch.bool)
    mask[:, :-1, :] |= up
    mask[:, 1:, :] |= down
    mask[:, :, :-1] |= left
    mask[:, :, 1:] |= right
    # Border conservatively flagged.
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def detect_boundary_mask_canny_segnet(
    rgb_bchw: torch.Tensor,
    seg_argmax_bhw: torch.Tensor | None,
    *,
    edge_threshold: float,
) -> torch.Tensor:
    """Compute the boundary mask as Canny-edges ∪ SegNet-argmax-4nbr-disagreement.

    Args:
        rgb_bchw: ``(B, 3, H, W)`` RGB in ``[0, 1]``.
        seg_argmax_bhw: ``(B, H, W)`` SegNet argmax class indices, or None
            (Canny-only mode for boundary detection during inference when
            scorers are unavailable).
        edge_threshold: gradient-magnitude threshold in ``[0, 1]``.

    Returns:
        ``(B, H, W)`` boolean boundary mask. True = boundary pixel
        (high-fidelity store); False = interior pixel (texture-fill).
    """
    mag = _sobel_gradient_magnitude(rgb_bchw).squeeze(1)  # (B, H, W)
    canny_mask = mag > edge_threshold
    if seg_argmax_bhw is None:
        return canny_mask
    seg_mask = _segnet_argmax_4nbr_disagreement(seg_argmax_bhw)
    return canny_mask | seg_mask


# ---------------------------------------------------------------------------
# FiLM-conditioned refinement decoder
# ---------------------------------------------------------------------------


class _FiLMRefineBlock(nn.Module):
    """One Conv + GELU block with FiLM (gamma, beta) modulation."""

    def __init__(self, hidden: int, embedding_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.film_gen = nn.Linear(embedding_dim, 2 * hidden)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        gamma_beta = self.film_gen(emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + gamma) + beta
        return self.act(h)


class SaborBoundaryOnlyRenderer(nn.Module):
    """SABOR boundary-only renderer.

    Forward signature mirrors sane_hnerv for trainer interop::

        forward(pair_indices, boundary_mask, boundary_rgb, segnet_argmax)
            -> (rgb_0, rgb_1) each (B, 3, H, W) in [0, 1]

    The boundary mask + boundary RGB + segnet_argmax are precomputed by the
    trainer (they are byte-payload inputs at archive time, not learned).
    The renderer's learned parameters are: ``class_means`` (per-class mean RGB),
    ``pair_embedding`` (per-pair FiLM conditioning), ``pair_bias`` (per-pair
    RGB bias correction), and the FiLM refinement decoder.

    At inflate time the deterministic algorithm is::

        1. Reconstruct interior pixels: ``rgb_interior = class_means[segnet_argmax] + pair_bias``
        2. Overlay boundary pixels: ``rgb_final[boundary_mask] = boundary_rgb[boundary_mask]``
        3. Refine via FiLM-conditioned decoder.
    """

    def __init__(self, cfg: SaborBoundaryOnlyConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Per-class mean RGB (learned).
        self.class_means = nn.Parameter(
            torch.empty(cfg.num_seg_classes, 3).uniform_(0.3, 0.7)
        )

        # Per-pair embedding (FiLM conditioning + bias generation).
        # We have num_pairs * 2 frames per pair, so embedding rows = 2 * num_pairs.
        self.pair_embedding = nn.Parameter(
            torch.empty(cfg.num_pairs * 2, cfg.embedding_dim).normal_(std=0.02)
        )

        # Per-pair RGB bias (small per-frame color correction).
        self.pair_bias = nn.Parameter(
            torch.zeros(cfg.num_pairs * 2, cfg.bias_dim)
        )

        # Refinement stem + blocks.
        self.stem = nn.Conv2d(3, cfg.refinement_hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                _FiLMRefineBlock(cfg.refinement_hidden, cfg.embedding_dim)
                for _ in range(cfg.refinement_blocks)
            ]
        )
        self.head_rgb = nn.Conv2d(cfg.refinement_hidden, 3, kernel_size=3, padding=1)

    def _texture_fill(
        self,
        segnet_argmax_bhw: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Interior texture fill: class_means[argmax] + per-frame bias.

        Args:
            segnet_argmax_bhw: ``(B, H, W)`` long class indices.
            frame_indices: ``(B,)`` long frame indices into pair_embedding.

        Returns:
            ``(B, 3, H, W)`` RGB tensor in approximately ``[0, 1]``.
        """
        if segnet_argmax_bhw.dim() != 3:
            raise ValueError(
                f"segnet_argmax_bhw expected (B, H, W); got {tuple(segnet_argmax_bhw.shape)}"
            )
        means = self.class_means[segnet_argmax_bhw]  # (B, H, W, 3)
        rgb = means.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        bias = self.pair_bias[frame_indices]  # (B, 3)
        rgb = rgb + bias.unsqueeze(-1).unsqueeze(-1)
        return rgb

    def forward(
        self,
        pair_indices: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_rgb: torch.Tensor,
        segnet_argmax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one batch of frame pairs.

        Args:
            pair_indices: ``(B,)`` long pair indices in ``[0, num_pairs)``.
            boundary_mask: ``(B, 2, H, W)`` boolean — True where boundary.
            boundary_rgb: ``(B, 2, 3, H, W)`` float in ``[0, 1]`` — the high-
                fidelity RGB at the boundary pixels (interior values are ignored).
            segnet_argmax: ``(B, 2, H, W)`` long — SegNet argmax class indices.

        Returns:
            ``(rgb_0, rgb_1)`` each ``(B, 3, H, W)`` float in ``[0, 1]``.
        """
        if pair_indices.dtype != torch.long:
            raise ValueError("pair_indices must be torch.long")
        if pair_indices.min().item() < 0 or pair_indices.max().item() >= self.cfg.num_pairs:
            raise ValueError(f"pair_indices out of range [0, {self.cfg.num_pairs})")
        if boundary_mask.shape != boundary_rgb.shape[:2] + boundary_rgb.shape[3:]:
            raise ValueError(
                f"boundary_mask {tuple(boundary_mask.shape)} vs boundary_rgb "
                f"{tuple(boundary_rgb.shape)} shape mismatch"
            )
        if segnet_argmax.shape != boundary_mask.shape:
            raise ValueError(
                f"segnet_argmax shape {tuple(segnet_argmax.shape)} != boundary_mask "
                f"{tuple(boundary_mask.shape)}"
            )
        if segnet_argmax.dtype != torch.long:
            raise ValueError("segnet_argmax must be torch.long")

        out: list[torch.Tensor] = []
        for frame_offset in (0, 1):
            frame_indices = 2 * pair_indices + frame_offset
            seg_t = segnet_argmax[:, frame_offset]
            mask_t = boundary_mask[:, frame_offset]
            brgb_t = boundary_rgb[:, frame_offset]
            interior = self._texture_fill(seg_t, frame_indices)  # (B, 3, H, W)
            # Overlay boundary pixels (boundary_rgb verbatim where mask True).
            mask_chw = mask_t.unsqueeze(1).expand_as(interior)
            composite = torch.where(mask_chw, brgb_t, interior)
            # FiLM-conditioned refinement on composite.
            emb = self.pair_embedding[frame_indices]  # (B, embedding_dim)
            h = self.stem(composite)
            for blk in self.blocks:
                h = blk(h, emb)
            refined = torch.sigmoid(self.head_rgb(h) + torch.logit(composite.clamp(1e-4, 1 - 1e-4)))
            out.append(refined)
        return out[0], out[1]

    def quantize_class_means_for_archive(self) -> torch.Tensor:
        """Quantize class_means to uint8 (clamp to [0, 1] then scale to 255)."""
        with torch.no_grad():
            q = (self.class_means.detach().clamp(0.0, 1.0) * 255.0).round()
        return q.to(torch.uint8)

    def quantize_boundary_rgb_for_archive(self, boundary_rgb: torch.Tensor) -> torch.Tensor:
        """Quantize boundary RGB to uint8 (clamp to [0, 1] then scale to 255)."""
        with torch.no_grad():
            q = (boundary_rgb.detach().clamp(0.0, 1.0) * 255.0).round()
        return q.to(torch.uint8)

    def runtime_state_dict_for_archive(self) -> dict[str, torch.Tensor]:
        """Return only inflate-time tensors for the SBO1 archive.

        class_means lives in its own uint8 section (quantized separately).
        pair_embedding + pair_bias + stem/blocks/head live in the
        decoder_state_dict section as fp16 brotli'd.
        """
        return {
            name: tensor.detach().clone()
            for name, tensor in self.state_dict().items()
            if name != "class_means"
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "SABOR_EVAL_H",
    "SABOR_EVAL_W",
    "SABOR_NUM_FRAMES",
    "SABOR_NUM_PAIRS",
    "SABOR_NUM_SEG_CLASSES",
    "SaborBoundaryOnlyConfig",
    "SaborBoundaryOnlyRenderer",
    "detect_boundary_mask_canny_segnet",
]
