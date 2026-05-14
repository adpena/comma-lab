# SPDX-License-Identifier: MIT
"""Lane SZ — szabolcs-cs PR#56 paradigm replica.

This module implements the architecture from szabolcs-cs's PR #56 (score 0.36)
that achieves competitive results WITHOUT storing segmentation masks in the
archive. Reference: project_szabolcs_full_re_20260426 + the disassembled
``inflate.py`` recovered to ``/tmp/szabolcs_re/``.

High-level paradigm
-------------------
The submission archive contains:
  * A monochrome AV1 video at 384x512 (luma only — no class labels stored).
  * A small renderer (``segmap_inference.pt``) packed in block-floating-point.
  * NO masks.mkv; the per-class probability map is *reconstructed at inflate
    time* from luma pixel values via a fixed Gaussian-softmax look-up.

The renderer pipeline (per frame):
  1. Decode luma -> ``(H=384, W=512)`` float32 in ``[0, 255]``.
  2. Round + clamp to ``long`` and look up
     ``softmax(exp(-(x - target)^2 / (2 * sigma^2)))`` over the five class
     targets ``[0, 255, 64, 192, 128]`` (sigma=15.0). Result: 5-channel soft
     class probability map.
  3. Build per-frame affine-warped 3-channel latent canvas:
       a. ``shared_latent_base`` (1, 3, 30, 40) bicubic-upsampled to
          ``ceil(1.25 * out_h), ceil(1.25 * out_w)``.
       b. ``frame_affine_embedding(idx)`` -> 6-DoF tanh-bounded affine
          (zoom delta, aspect delta, shear_x, shear_y, trans_x, trans_y).
       c. ``affine_grid + grid_sample`` produces a per-frame (3, out_h, out_w)
          canvas.
  4. Concat ``[probability_map (5ch), latent_canvas (3ch)]`` -> 8 channels.
  5. Tiny CNN: ``Conv2d(8, hidden, 1) -> N x ResidualBlock -> Conv2d(hidden, 3, 1)``.
  6. ``sigmoid * 255`` -> RGB at ``(384, 512)``; bicubic upscale to
     ``(874, 1164)`` for camera-resolution output.

Phase 1 scope (this file)
-------------------------
We provide:
  * ``SzabolcsRenderer`` (``nn.Module``) — exact reference architecture, with
    the sigmoid * 255 output. Reads inputs as float probability maps + frame
    indices; matches the upstream forward signature.
  * ``create_gaussian_softmax_lut`` — fixed (256, 5) table builder.
  * ``encode_luma_to_probability_map`` — convenience helper that takes a luma
    tensor and returns the 5-channel soft class map (float32 in [0, 1]).
  * ``build_szabolcs_renderer`` — friendly factory that returns the renderer
    plus a parameter-count summary, for budget pre-flight.

Out of scope (Phase 2+)
-----------------------
  * Block-floating-point (1.017 bits/weight) export/import.
  * tar.xz double compression of the archive payload.
  * Inflate-time pipeline glue (the canonical inflate.sh path).
  * Training script. The forward signature here intentionally matches the
    reference, so a training loop can drive (luma -> prob map) + frame indices
    -> RGB pair end-to-end without further architectural changes.

This module deliberately stays standalone. The existing ``PairGenerator``
contract takes ``(mask_t, mask_t1)`` long tensors; the szabolcs paradigm has
no mask input and instead conditions on per-frame indices. Forcing it through
``PairGenerator`` would lie about the data flow. Higher-level integration
(training profile, archive builder, inflate dispatcher) is staged for Phase 2
once the forward / parameter-count assumptions here are validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants (mirroring /tmp/szabolcs_re/inflate.py) ──────────────────────

CAMERA_SIZE: tuple[int, int] = (1164, 874)  # (W, H) — camera-native output
SEGMAP_INPUT_SIZE: tuple[int, int] = (512, 384)  # (W, H) — renderer working res
CLASS_TARGETS: tuple[int, ...] = (0, 255, 64, 192, 128)  # luma centroids per class
LUT_SIGMA: float = 15.0  # Gaussian width over luma values


# ── Building blocks ────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Two 3x3 conv residual block with SiLU activations.

    Mirrors the reference implementation byte-for-byte. The intermediate
    width (``block_hidden``) may differ from the residual width (``hidden``);
    the second conv projects back so the skip connection is shape-compatible.
    """

    def __init__(self, hidden: int, block_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, block_hidden, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(block_hidden, hidden, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        return self.act2(out + x)


class SzabolcsRenderer(nn.Module):
    """szabolcs-cs PR#56 SegMap renderer (Phase 1 replica).

    Args:
      hidden: residual stream width (reference default 32-64).
      block_hidden: intermediate width inside each residual block.
      num_blocks: number of residual blocks in the body.
      max_frame_index: size of the per-frame affine embedding table. The
        contest video has 1200 frames; submissions pass ``2 * idx`` and
        ``2 * idx + 1`` per pair, so this should be at least 1200.
      affine_max_zoom_delta / aspect / shear / translation: tanh bounds on
        each affine DoF (defaults match the reference).
      latent_input_scale: multiplier on the affine-latent channel before
        concat. Reference default is 1.0; kept as a knob because Quantizr
        showed FiLM/latent scale matters for SegNet sensitivity.
      shared_latent_height / shared_latent_width: spatial size of the
        learned shared latent canvas before bicubic upsample. Reference
        defaults are 30x40 (3600 params at 3 channels).
      shared_latent_channels: defaults to 3 (per reference). The 3-channel
        latent is concatenated with a 5-channel probability map -> 8ch.
      latent_canvas_scale: bicubic-upsample factor for the shared latent
        before affine grid_sample (reference 1.25x for crop slack).

    Forward signature
    -----------------
    ``forward(probability_map, frame_indices)`` where:
      * ``probability_map``: ``(B, 5, H, W)`` float in [0, 1]; produced by
        ``encode_luma_to_probability_map`` from a decoded luma frame.
      * ``frame_indices``: ``(B,)`` long; selects the per-frame affine row.

    Returns ``(B, 3, H, W)`` float in [0, 255] (sigmoid * 255). Caller is
    responsible for any final bicubic upscale to camera resolution; we keep
    the renderer working at ``SEGMAP_INPUT_SIZE`` because per-pixel layers
    at full res would inflate the FLOP count by ~6x.
    """

    def __init__(
        self,
        hidden: int = 32,
        block_hidden: int | None = None,
        num_blocks: int = 4,
        max_frame_index: int = 1200,
        affine_max_zoom_delta: float = 0.12,
        affine_max_aspect_delta: float = 0.03,
        affine_max_shear: float = 0.03,
        affine_max_translation: float = 0.08,
        latent_input_scale: float = 1.0,
        shared_latent_channels: int = 3,
        shared_latent_height: int = 30,
        shared_latent_width: int = 40,
        latent_canvas_scale: float = 1.25,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        if block_hidden is None:
            block_hidden = hidden
        self.h = SEGMAP_INPUT_SIZE[1]
        self.w = SEGMAP_INPUT_SIZE[0]
        self.hidden = hidden
        self.block_hidden = block_hidden
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.shared_latent_channels = shared_latent_channels
        self.shared_latent_height = shared_latent_height
        self.shared_latent_width = shared_latent_width
        self.latent_canvas_scale = latent_canvas_scale
        self.max_zoom_delta = affine_max_zoom_delta
        self.max_aspect_delta = affine_max_aspect_delta
        self.max_shear = affine_max_shear
        self.max_translation = affine_max_translation
        self.latent_input_scale = latent_input_scale
        self.max_frame_index = max_frame_index

        self.shared_latent_base = nn.Parameter(
            torch.empty(
                1,
                shared_latent_channels,
                shared_latent_height,
                shared_latent_width,
            )
        )
        self.frame_affine_embedding = nn.Embedding(max_frame_index, 6)
        self.layer_in = nn.Conv2d(
            num_classes + shared_latent_channels, hidden, kernel_size=1
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, block_hidden) for _ in range(num_blocks)]
        )
        self.layer_out = nn.Conv2d(hidden, 3, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Small-init for both the shared latent and affine deltas so the
        # network starts close to "identity affine + neutral latent". Without
        # this the tanh-bounded affines saturate immediately and grids drift
        # outside [-1, 1] before any optimization step.
        nn.init.normal_(self.shared_latent_base, mean=0.0, std=0.02)
        nn.init.zeros_(self.frame_affine_embedding.weight)
        # Conv layers: PyTorch default kaiming_uniform_ is fine; we just
        # zero the output projection bias so initial RGB ~ sigmoid(0)*255 = 127.5.
        nn.init.zeros_(self.layer_out.bias)

    # ── affine-latent channel ─────────────────────────────────────────────

    def _build_affine_latent_channel(
        self,
        frame_indices: torch.Tensor,
        output_height: int,
        output_width: int,
    ) -> torch.Tensor:
        """Bicubic-upsample shared latent -> per-frame affine warp."""
        batch_size = frame_indices.shape[0]
        canvas_height = math.ceil(output_height * self.latent_canvas_scale)
        canvas_width = math.ceil(output_width * self.latent_canvas_scale)
        shared_latent = F.interpolate(
            self.shared_latent_base,
            size=(canvas_height, canvas_width),
            mode="bicubic",
            align_corners=False,
        ).expand(batch_size, -1, -1, -1)

        affine_delta = self.frame_affine_embedding(frame_indices)
        zoom = 1.0 + self.max_zoom_delta * torch.tanh(affine_delta[:, 0:1])
        aspect = self.max_aspect_delta * torch.tanh(affine_delta[:, 1:2])
        shear_x = self.max_shear * torch.tanh(affine_delta[:, 2:3])
        shear_y = self.max_shear * torch.tanh(affine_delta[:, 3:4])
        trans_x = self.max_translation * torch.tanh(affine_delta[:, 4:5])
        trans_y = self.max_translation * torch.tanh(affine_delta[:, 5:6])
        scale_x = zoom + aspect
        scale_y = zoom - aspect
        theta = torch.cat(
            [scale_x, shear_x, trans_x, shear_y, scale_y, trans_y], dim=1
        ).view(-1, 2, 3)

        grid = F.affine_grid(
            theta,
            size=(
                batch_size,
                self.shared_latent_channels,
                output_height,
                output_width,
            ),
            align_corners=False,
        )
        return F.grid_sample(
            shared_latent,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        probability_map: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the SegMap renderer.

        Args:
          probability_map: (B, num_classes, H, W) float in [0, 1].
          frame_indices: (B,) long, in ``[0, max_frame_index)``.
        Returns:
          (B, 3, H, W) float in [0, 255].
        """
        if probability_map.dim() != 4:
            raise ValueError(
                f"probability_map must be (B, C, H, W); got {probability_map.shape}"
            )
        if probability_map.shape[1] != self.num_classes:
            raise ValueError(
                f"probability_map channel count {probability_map.shape[1]} != "
                f"num_classes {self.num_classes}"
            )
        if frame_indices.dim() != 1:
            raise ValueError(
                f"frame_indices must be 1-D (B,); got {frame_indices.shape}"
            )
        if frame_indices.shape[0] != probability_map.shape[0]:
            raise ValueError(
                f"frame_indices batch {frame_indices.shape[0]} != "
                f"probability_map batch {probability_map.shape[0]}"
            )

        affine_latent = self._build_affine_latent_channel(
            frame_indices, probability_map.shape[-2], probability_map.shape[-1]
        )
        feat = self.layer_in(
            torch.cat(
                [probability_map, affine_latent * self.latent_input_scale], dim=1
            )
        )
        for block in self.blocks:
            feat = block(feat)
        return torch.sigmoid(self.layer_out(feat)) * 255.0

    # ── budget helpers ────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict[str, int]:
        """Per-component parameter count for budget review."""
        latent = self.shared_latent_base.numel()
        affine = self.frame_affine_embedding.weight.numel()
        layer_in = sum(p.numel() for p in self.layer_in.parameters())
        layer_out = sum(p.numel() for p in self.layer_out.parameters())
        body = sum(
            sum(p.numel() for p in block.parameters()) for block in self.blocks
        )
        total = latent + affine + layer_in + layer_out + body
        return {
            "shared_latent": latent,
            "frame_affine_embedding": affine,
            "layer_in": layer_in,
            "blocks_total": body,
            "layer_out": layer_out,
            "total": total,
        }


# ── Gaussian softmax LUT (fixed, no learned parameters) ────────────────────


def create_gaussian_softmax_lut(
    class_targets: tuple[int, ...] = CLASS_TARGETS,
    sigma: float = LUT_SIGMA,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the (256, num_classes) probability table.

    For each luma value ``x in [0, 255]`` and each class target ``t``, we
    compute ``exp(-(x - t)^2 / (2 sigma^2))`` and softmax across classes.
    The result is purely deterministic and is recreated at inflate time
    rather than stored in the archive — the recipe is in code, not bytes.

    Returns:
      Tensor of shape ``(256, len(class_targets))``, rows sum to 1.
    """
    x = torch.arange(256, dtype=dtype, device=device).unsqueeze(1)
    targets = torch.tensor(class_targets, dtype=dtype, device=device).unsqueeze(0)
    squared_diff = (x - targets) ** 2
    logits = torch.exp(-squared_diff / (2.0 * sigma * sigma))
    return F.softmax(logits, dim=1)


def encode_luma_to_probability_map(
    luma: torch.Tensor,
    lut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert luma pixel values to a 5-channel soft probability map.

    Args:
      luma: ``(B, H, W)`` or ``(B, 1, H, W)`` float in ``[0, 255]``. Will be
        rounded + clamped + cast to ``long`` before lookup. The reference
        implementation uses bicubic-resized luma at ``SEGMAP_INPUT_SIZE``;
        callers should resize *before* calling this helper.
      lut: optional pre-built ``(256, C)`` LUT. Recomputed on the input
        device + dtype if not supplied.

    Returns:
      ``(B, C, H, W)`` float probability map.
    """
    if luma.dim() == 4 and luma.shape[1] == 1:
        luma = luma.squeeze(1)
    if luma.dim() != 3:
        raise ValueError(f"luma must be (B, H, W) or (B, 1, H, W); got {luma.shape}")

    if lut is None:
        lut = create_gaussian_softmax_lut(device=luma.device, dtype=luma.dtype)
    elif lut.device != luma.device or lut.dtype != luma.dtype:
        lut = lut.to(device=luma.device, dtype=luma.dtype)

    long_luma = luma.round().clamp(0, 255).long()
    # F.embedding expects 1-D index input flattened; we reshape after.
    probs = F.embedding(long_luma, lut)  # (B, H, W, C)
    return probs.permute(0, 3, 1, 2).contiguous()


# ── Friendly factory ───────────────────────────────────────────────────────


@dataclass
class SzabolcsBuildResult:
    """Bundle returned by ``build_szabolcs_renderer``.

    Attributes:
      model: The constructed ``SzabolcsRenderer``.
      lut: Pre-built (256, num_classes) Gaussian-softmax LUT (CPU, float32).
      param_breakdown: Per-component parameter dict (see ``param_breakdown``).
      total_params: Convenience copy of ``param_breakdown['total']``.
    """

    model: SzabolcsRenderer
    lut: torch.Tensor
    param_breakdown: dict[str, int]
    total_params: int


def build_szabolcs_renderer(
    hidden: int = 32,
    block_hidden: int | None = None,
    num_blocks: int = 4,
    max_frame_index: int = 1200,
    affine_max_zoom_delta: float = 0.12,
    affine_max_aspect_delta: float = 0.03,
    affine_max_shear: float = 0.03,
    affine_max_translation: float = 0.08,
    latent_input_scale: float = 1.0,
    shared_latent_height: int = 30,
    shared_latent_width: int = 40,
    shared_latent_channels: int = 3,
    latent_canvas_scale: float = 1.25,
    num_classes: int = 5,
    quiet: bool = False,
) -> SzabolcsBuildResult:
    """Construct a SzabolcsRenderer + Gaussian-softmax LUT bundle.

    Returns a ``SzabolcsBuildResult`` so callers (training script, archive
    builder, inflate path) get the LUT pre-baked with the architecture for
    free — there is exactly one canonical LUT, and it is ``CLASS_TARGETS`` +
    ``LUT_SIGMA`` derived. Storing it next to the model keeps the recipe in
    one place.
    """
    model = SzabolcsRenderer(
        hidden=hidden,
        block_hidden=block_hidden,
        num_blocks=num_blocks,
        max_frame_index=max_frame_index,
        affine_max_zoom_delta=affine_max_zoom_delta,
        affine_max_aspect_delta=affine_max_aspect_delta,
        affine_max_shear=affine_max_shear,
        affine_max_translation=affine_max_translation,
        latent_input_scale=latent_input_scale,
        shared_latent_channels=shared_latent_channels,
        shared_latent_height=shared_latent_height,
        shared_latent_width=shared_latent_width,
        latent_canvas_scale=latent_canvas_scale,
        num_classes=num_classes,
    )
    lut = create_gaussian_softmax_lut()
    breakdown = model.param_breakdown()
    if not quiet:
        print(
            f"[szabolcs_renderer] Built SzabolcsRenderer: "
            f"{breakdown['total']:,} params "
            f"(latent={breakdown['shared_latent']:,}, "
            f"affine={breakdown['frame_affine_embedding']:,}, "
            f"body={breakdown['blocks_total']:,}, "
            f"in/out={breakdown['layer_in'] + breakdown['layer_out']:,})"
        )
    return SzabolcsBuildResult(
        model=model,
        lut=lut,
        param_breakdown=breakdown,
        total_params=breakdown["total"],
    )


# ── SZv1 binary loader (Phase 2) ───────────────────────────────────────────


def load_szabolcs_renderer(
    data: "bytes | str | object",
    device: str | torch.device = "cpu",
) -> SzabolcsRenderer:
    """Inflate a SZv1 binary into a ready-to-run ``SzabolcsRenderer``.

    Args:
        data: Either raw SZv1 bytes (as produced by
            ``pack_szabolcs_archive``) or a path to a SZv1 file on disk.
        device: Target device for the loaded model (default CPU; the inflate
            wrapper passes the inflate-time device).

    Returns:
        A ``SzabolcsRenderer`` in eval mode with all weights restored from
        the block-FP-packed payload. No scorers are loaded — the szabolcs
        renderer reconstructs class probability maps from luma via the fixed
        Gaussian LUT, so strict-scorer-rule is satisfied trivially.

    The runtime banner printed by this loader (``[szabolcs] inflated …``) is
    used by the inflate dispatcher to tag the score lane.
    """
    # Imported lazily so that the szabolcs_renderer module remains importable
    # when only the architecture is needed (e.g. unit tests that don't touch
    # the archive packer).
    from tac.szabolcs_archive import unpack_szabolcs_archive

    contents = unpack_szabolcs_archive(data)
    cfg = contents.config

    model = SzabolcsRenderer(
        hidden=int(cfg["hidden"]),
        block_hidden=int(cfg.get("block_hidden") or cfg["hidden"]),
        num_blocks=int(cfg["num_blocks"]),
        max_frame_index=int(cfg["max_frame_index"]),
        affine_max_zoom_delta=float(cfg.get("affine_max_zoom_delta", 0.12)),
        affine_max_aspect_delta=float(cfg.get("affine_max_aspect_delta", 0.03)),
        affine_max_shear=float(cfg.get("affine_max_shear", 0.03)),
        affine_max_translation=float(cfg.get("affine_max_translation", 0.08)),
        latent_input_scale=float(cfg.get("latent_input_scale", 1.0)),
        shared_latent_channels=int(cfg.get("shared_latent_channels", 3)),
        shared_latent_height=int(cfg.get("shared_latent_height", 30)),
        shared_latent_width=int(cfg.get("shared_latent_width", 40)),
        latent_canvas_scale=float(cfg.get("latent_canvas_scale", 1.25)),
        num_classes=int(cfg.get("num_classes", 5)),
    )
    missing, unexpected = model.load_state_dict(contents.state_dict, strict=False)
    if missing or unexpected:
        # We tolerate a handful of "extra" keys (none expected today) but
        # missing keys are a hard error: the renderer would silently zero-init.
        if missing:
            raise RuntimeError(
                f"load_szabolcs_renderer: SZv1 state_dict missing keys "
                f"{sorted(missing)[:8]} — refusing to inflate a partially "
                f"initialized renderer."
            )
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    print(
        f"[szabolcs] inflated SZv1 renderer "
        f"({contents.header.get('param_count', 'n/a')} params, "
        f"packed_bytes={contents.header.get('tarxz_nbytes', 'n/a')}, "
        f"predicted_band={contents.header.get('predicted_band')})",
        flush=True,
    )
    return model


__all__ = [
    "CAMERA_SIZE",
    "SEGMAP_INPUT_SIZE",
    "CLASS_TARGETS",
    "LUT_SIGMA",
    "ResidualBlock",
    "SzabolcsRenderer",
    "SzabolcsBuildResult",
    "create_gaussian_softmax_lut",
    "encode_luma_to_probability_map",
    "build_szabolcs_renderer",
    "load_szabolcs_renderer",
]
