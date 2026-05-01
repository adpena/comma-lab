"""Selfcomp ``SegMap`` renderer (mask -> RGB) + matching trainer.

The ``SegMap`` architecture is the rate-attack core of the Selfcomp PR
paradigm — the inflate-time renderer that consumes a class-probability map
(produced by the grayscale-LUT pathway in ``tac.mask_grayscale_lut``) and a
per-frame index, and emits an RGB frame at SegMap-input resolution
(384 x 512). The bicubic upsample to camera resolution (1164 x 874) happens
downstream in ``submissions/robust_current/inflate_segmap.py``.

The model classes (``ResidualBlock``, ``SegMap``) are byte-faithful to
Selfcomp's reference inflate.py L31-134 — the comment block at the top of the
module documents that lineage so a future operator can audit any drift.

This module also adds the trainer that does NOT live in the reference (the
reference is decode-only). The trainer:

* Refuses ``eval_roundtrip=False`` and ``device="mps"`` at construction time
  (CLAUDE.md non-negotiables — proxy/auth gap is 2-11x without roundtrip,
  and MPS auth eval is invalid per the 2026-04-25 measurement).
* Optionally stacks Hinton-style KL-distill (T=2.0) on the SegNet logits
  via ``loss_mode == "kl_distill"`` — same auxiliary regime as Lane G v3.
* Exports an ``inference_state_dict`` shaped exactly the way the Selfcomp
  reference ``load_segmap`` consumes it (see L183-220 of the reference).

Note on training inputs: the trainer takes ``mask_pairs`` already in the
LUT-projected probability form (B, T, NUM_CLASSES, H, W) so we don't pull in
the LUT tensor here — caller (the lane script) is responsible for the
grayscale -> probability projection. ``gt_pairs`` are camera-resolution RGB
frame pairs (B, T, H, W, 3) suitable for ``scorer_forward_pair``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.losses import (
    kl_distill_segnet_only,
    scorer_forward_pair,
)
from tac.preflight import PreflightError
from tac.training import EMA, TrainConfig

# Match Selfcomp inflate.py constants (L25-29).
CAMERA_SIZE: tuple[int, int] = (1164, 874)
SEGMAP_INPUT_SIZE: tuple[int, int] = (512, 384)


# ── Architecture (byte-faithful to Selfcomp reference) ────────────────────


class ResidualBlock(nn.Module):
    """3x3 -> SiLU -> 3x3 -> SiLU(+residual). Mirrors Selfcomp ResidualBlock."""

    def __init__(self, hidden: int, block_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden, block_hidden, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(block_hidden, hidden, kernel_size=3, padding=1)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        return self.act2(out + residual)


class SegMap(nn.Module):
    """Mask-conditional RGB renderer with per-frame learned affine latent."""

    def __init__(
        self,
        hidden: int,
        block_hidden: int,
        num_blocks: int,
        max_frame_index: int,
        affine_max_zoom_delta: float = 0.12,
        affine_max_aspect_delta: float = 0.03,
        affine_max_shear: float = 0.03,
        affine_max_translation: float = 0.08,
        latent_input_scale: float = 1.0,
    ):
        super().__init__()
        self.h = SEGMAP_INPUT_SIZE[1]
        self.w = SEGMAP_INPUT_SIZE[0]
        self.hidden = hidden
        self.block_hidden = block_hidden
        self.num_blocks = num_blocks
        self.shared_latent_channels = 3
        self.shared_latent_height = 30
        self.shared_latent_width = 40
        self.latent_canvas_scale = 1.25
        self.max_zoom_delta = affine_max_zoom_delta
        self.max_aspect_delta = affine_max_aspect_delta
        self.max_shear = affine_max_shear
        self.max_translation = affine_max_translation
        self.latent_input_scale = latent_input_scale
        self.shared_latent_base = nn.Parameter(
            torch.empty(
                1,
                self.shared_latent_channels,
                self.shared_latent_height,
                self.shared_latent_width,
            )
        )
        nn.init.normal_(self.shared_latent_base, mean=0.0, std=0.02)
        self.frame_affine_embedding = nn.Embedding(max_frame_index, 6)
        nn.init.normal_(self.frame_affine_embedding.weight, mean=0.0, std=0.01)
        self.layer_in = nn.Conv2d(5 + self.shared_latent_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, block_hidden) for _ in range(num_blocks)]
        )
        self.layer_out = nn.Conv2d(hidden, 3, kernel_size=1)

    def _build_affine_latent_channel(
        self, frame_indices: torch.Tensor, output_height: int, output_width: int
    ) -> torch.Tensor:
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
            size=(batch_size, self.shared_latent_channels, output_height, output_width),
            align_corners=False,
        )
        return F.grid_sample(
            shared_latent,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        affine_latent = self._build_affine_latent_channel(
            frame_indices, x.shape[-2], x.shape[-1]
        )
        feat = self.layer_in(torch.cat([x, affine_latent * self.latent_input_scale], dim=1))
        for block in self.blocks:
            feat = block(feat)
        return torch.sigmoid(self.layer_out(feat)) * 255.0


class SegMapHomography(SegMap):
    """SegMap variant with 8-DOF perspective homography frame embeddings.

    The 6-DOF affine grid in the canonical SegMap is a strict subset of the
    8-DOF perspective homography. Adding 2 perspective parameters (the
    bottom-row entries g, h of the 3x3 homography matrix) lets the per-frame
    latent capture forward-zoom + tilt patterns that pure affine cannot,
    which matches the comma.ai dashcam viewing geometry better.

    Tensor delta from base SegMap: ``frame_affine_embedding`` is dim 8
    (was 6); two extra params per frame; layer_in/blocks/layer_out are
    UNCHANGED. Total parameter count change is tiny (~2 * max_frame_index).

    Lane HM-S (homography-augmented SegMap) uses this. Predicted band
    [0.32, 0.45] [contest-CUDA] (small improvement over Lane SC++ via
    better road-plane perspective tracking).
    """

    PERSPECTIVE_BOUND: float = 0.05  # max |g|, |h| after tanh activation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_frame_index = self.frame_affine_embedding.num_embeddings
        # Replace the dim-6 embedding with a dim-8 one (init small).
        self.frame_affine_embedding = nn.Embedding(max_frame_index, 8)
        nn.init.normal_(self.frame_affine_embedding.weight, mean=0.0, std=0.01)

    def _build_affine_latent_channel(
        self, frame_indices: torch.Tensor, output_height: int, output_width: int
    ) -> torch.Tensor:
        batch_size = frame_indices.shape[0]
        canvas_height = math.ceil(output_height * self.latent_canvas_scale)
        canvas_width = math.ceil(output_width * self.latent_canvas_scale)
        shared_latent = F.interpolate(
            self.shared_latent_base,
            size=(canvas_height, canvas_width),
            mode="bicubic",
            align_corners=False,
        ).expand(batch_size, -1, -1, -1)
        params = self.frame_affine_embedding(frame_indices)  # (B, 8)
        zoom = 1.0 + self.max_zoom_delta * torch.tanh(params[:, 0:1])
        aspect = self.max_aspect_delta * torch.tanh(params[:, 1:2])
        shear_x = self.max_shear * torch.tanh(params[:, 2:3])
        shear_y = self.max_shear * torch.tanh(params[:, 3:4])
        trans_x = self.max_translation * torch.tanh(params[:, 4:5])
        trans_y = self.max_translation * torch.tanh(params[:, 5:6])
        # 7th, 8th: perspective row [g, h] of the homography. tanh-bounded
        # so the homography stays well-conditioned (no division by ≈0).
        persp_x = self.PERSPECTIVE_BOUND * torch.tanh(params[:, 6:7])
        persp_y = self.PERSPECTIVE_BOUND * torch.tanh(params[:, 7:8])
        scale_x = zoom + aspect
        scale_y = zoom - aspect
        # Build the 3x3 homography per batch element.
        ones = torch.ones_like(zoom)
        H = torch.stack(
            [
                torch.cat([scale_x, shear_x, trans_x], dim=1),
                torch.cat([shear_y, scale_y, trans_y], dim=1),
                torch.cat([persp_x, persp_y, ones], dim=1),
            ],
            dim=1,
        )  # (B, 3, 3)

        # Build a normalized [-1, 1] sampling grid (target -> source coords).
        ys = torch.linspace(-1.0, 1.0, output_height, device=H.device, dtype=H.dtype)
        xs = torch.linspace(-1.0, 1.0, output_width, device=H.device, dtype=H.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        ones_grid = torch.ones_like(xx)
        target_pts = torch.stack([xx, yy, ones_grid], dim=-1)  # (H, W, 3)
        target_pts = target_pts.view(-1, 3).T  # (3, H*W)

        # Apply per-batch homography: source = H · target (normalized coords).
        # H: (B, 3, 3); target_pts: (3, H*W); product is (B, 3, H*W).
        src_pts = torch.matmul(H, target_pts.unsqueeze(0).expand(batch_size, -1, -1))
        # Normalize the homogeneous coordinate; clamp the divisor away from 0.
        denom = src_pts[:, 2:3, :].clamp(min=1e-6)
        src_pts = src_pts[:, :2, :] / denom
        # Reshape to grid_sample input layout (B, H, W, 2).
        src_pts = src_pts.permute(0, 2, 1).reshape(batch_size, output_height, output_width, 2)

        return F.grid_sample(
            shared_latent,
            src_pts,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


# ── Trainer (the lane-side contribution) ─────────────────────────────────


def _eval_roundtrip_chain(
    rgb_pair_btchw: torch.Tensor,
    noise_std: float = 0.5,
) -> torch.Tensor:
    """Apply the canonical 384 -> 874 -> uint8 -> 384 contest eval chain.

    Mirrors the established roundtrip pattern from src/tac/losses.py / training.
    The bicubic resize to camera-H + uint8 cast + bicubic-back to scorer-H
    simulates the lossy decode of the ``.raw`` rgb24 the contest evaluator
    sees. Without this step the proxy-auth gap is 2-11x on PoseNet
    (feedback_proxy_auth_math_useless).

    Round 1 review CRITICAL fix: noise_std (Hotz STE proxy) was previously
    hardcoded to 0 — the dead-code pattern that burned weeks of LANE-B runs.
    Now threaded as a kwarg defaulting to 0.5 (matches qat_finetune /
    optimize_poses canonical noise_std). Caller can override via
    SegMapTrainer.train_epoch(roundtrip_noise_std=...).
    """
    b, t, c, h, w = rgb_pair_btchw.shape
    # 384x512 (scorer in) -> 874x1164 (camera) -> uint8 -> 384x512.
    flat = rgb_pair_btchw.reshape(b * t, c, h, w)
    up = F.interpolate(flat, size=CAMERA_SIZE[::-1], mode="bicubic", align_corners=False)
    # CRITICAL: torch.tensor.round() has ZERO gradient. Using bare .round()
    # here severs the entire backprop chain to SegMap parameters → optimizer
    # steps but params don't move. This was the Lane DARTS-S V1 freeze bug
    # (5h on Vast.ai 4090, 400 epochs of identical loss=277.02). Lane SC++,
    # SA, SO, MM v2 all invalidated by the same bug. Use Uint8STE (forward
    # = clamp+round, backward = identity inside [0,255], zero outside).
    # See council_darts_s_freeze_audit_20260429.md for the full diagnosis.
    from tac.quantization import Uint8STE
    up_u8 = Uint8STE.apply(up)
    if noise_std > 0:
        up_u8 = up_u8 + noise_std * torch.randn_like(up_u8)
    back = F.interpolate(up_u8, size=(h, w), mode="bicubic", align_corners=False)
    return back.clamp(0, 255).reshape(b, t, c, h, w)


class SegMapTrainer:
    """Trainer for ``SegMap`` with eval_roundtrip + KL-distill stacking.

    The trainer enforces CLAUDE.md non-negotiables at construction time:

    * ``config.eval_roundtrip`` MUST be True. Raises ``PreflightError`` on
      False — there is no valid reason to disable the contest-eval roundtrip
      simulation in 2026-04 onward (every wasted run in this project had it
      off).
    * ``device != "mps"``. Raises ``PreflightError`` on MPS — the auth eval
      drift is 23x on PoseNet (verified 2026-04-25), so MPS training cannot
      produce numbers that translate to contest CUDA.

    Args:
        model: an instantiated SegMap (already moved to ``device``).
        config: a ``TrainConfig``; controls loss_mode / kl temperature.
        posenet, segnet: frozen scorer modules (already on ``device``).
        device: torch.device or string. CUDA only.
    """

    def __init__(
        self,
        model: SegMap,
        config: TrainConfig,
        posenet,
        segnet,
        device: str | torch.device = "cuda",
        learnable_class_targets: Optional[nn.Module] = None,
    ):
        # Enforce CLAUDE.md non-negotiables BEFORE any GPU spend.
        if not config.eval_roundtrip:
            raise PreflightError(
                "SegMapTrainer requires config.eval_roundtrip=True. "
                "Without the 384->874->uint8->384 chain the proxy/auth gap "
                "is 2-11x on PoseNet (CLAUDE.md non-negotiable; "
                "feedback_proxy_auth_math_useless)."
            )
        if config.loss_mode == "kl_distill":
            scope = getattr(config, "kl_distill_scope", "none")
            if scope != "segnet_aux":
                raise PreflightError(
                    "SegMapTrainer implements only kl_distill_scope='segnet_aux' "
                    "as a SegNet-only auxiliary on top of the standard scorer "
                    "loss. Got kl_distill_scope="
                    f"{scope!r}. The legacy primary_scorer KL path is "
                    "forensic-only and must not be routed through SegMapTrainer."
                )
        self.distillation_policy = config.distillation_policy()
        self.distillation_policy_provenance = self.distillation_policy.to_provenance()
        self.distillation_policy_sha256 = config.distillation_policy_sha256()
        device_str = str(device)
        if device_str.startswith("mps"):
            raise PreflightError(
                "SegMapTrainer refuses device='mps'. MPS auth eval drifts "
                "23x on PoseNet (CLAUDE.md non-negotiable; verified "
                "2026-04-25). CUDA only."
            )
        self.model = model
        self.config = config
        self.posenet = posenet
        self.segnet = segnet
        self.device = torch.device(device_str)
        # Lane LCT (van den Oord VQ-VAE prescription): optional learnable
        # class-target codebook. When supplied, its parameters are added to
        # the optimizer so the codebook adapts during training.
        self.learnable_class_targets = learnable_class_targets

        # ── Council C OOM-class deep fixes (DF2 + DF3) ─────────────────
        # Resolve the bf16 autocast context once at __init__ time so the
        # train_epoch hot loop has zero per-step branching cost. The
        # autocast device_type is hard-coded to "cuda" — CLAUDE.md
        # FORBIDDEN PATTERN: never silently fall back to MPS/CPU.
        # Lane G v3 inference loop (training.py:967) already uses
        # autocast on CUDA for the eval scorer call; this extends the
        # symmetry to the SegMap training scorer call.
        bf16_requested = bool(getattr(config, "bf16", False))
        if bf16_requested and self.device.type != "cuda":
            raise PreflightError(
                "TrainConfig.bf16=True requires device='cuda'. Got "
                f"device={self.device.type!r}. CLAUDE.md FORBIDDEN PATTERN: "
                "do NOT silently fall back to MPS or CPU when CUDA is "
                "unavailable. Council C OOM-class deep fix (DF2): bf16 "
                "autocast halves PoseNet FastViT attention-map memory "
                "(.omx/research/council_oom_class_deep_fix_20260429.md)."
            )
        self.use_bf16 = bf16_requested
        # 0 = no chunking; >=1 = split scorer call into chunks of N pairs.
        chunk_raw = int(getattr(config, "scorer_chunk", 0) or 0)
        if chunk_raw < 0:
            raise PreflightError(
                f"TrainConfig.scorer_chunk must be >= 0, got {chunk_raw}"
            )
        self.scorer_chunk = chunk_raw

        params = [p for p in model.parameters() if p.requires_grad]
        if learnable_class_targets is not None:
            params.extend(p for p in learnable_class_targets.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay
        )

    # ── Council C deep fix DF3: per-pair scorer chunking ───────────────
    def _scorer_forward_chunked(
        self,
        rt_btchw: torch.Tensor,
        gt_btchw: torch.Tensor,
    ) -> tuple[dict, torch.Tensor, dict, torch.Tensor]:
        """Run scorer_forward_pair on (rt_btchw, gt_btchw), splitting along
        the batch dimension into chunks of ``self.scorer_chunk`` pairs.

        Preserves autograd: slicing with ``[a:b]`` returns a view that
        retains the gradient graph back to the renderer; ``torch.cat`` on
        the per-chunk outputs is also autograd-aware.

        Returns ``(posenet_out, segnet_out, gt_pose_out, gt_seg_out)`` —
        same return shape as a single ``scorer_forward_pair`` invocation,
        so downstream loss code is unchanged.

        When ``self.scorer_chunk == 0`` OR ``mb <= self.scorer_chunk``,
        falls through to a single un-chunked call (zero memory penalty
        AND mathematically identical to the legacy path → preserves bit-
        identical behaviour for tests with ``scorer_chunk=0``).
        """
        mb = rt_btchw.shape[0]
        chunk = self.scorer_chunk
        if chunk <= 0 or chunk >= mb:
            posenet_out, segnet_out = scorer_forward_pair(
                rt_btchw, self.posenet, self.segnet
            )
            with torch.no_grad():
                gt_pose_out, gt_seg_out = scorer_forward_pair(
                    gt_btchw, self.posenet, self.segnet
                )
            return posenet_out, segnet_out, gt_pose_out, gt_seg_out

        pose_chunks: list[torch.Tensor] = []
        seg_chunks: list[torch.Tensor] = []
        gt_pose_chunks: list[torch.Tensor] = []
        gt_seg_chunks: list[torch.Tensor] = []
        for cs in range(0, mb, chunk):
            ce = min(cs + chunk, mb)
            rt_chunk = rt_btchw[cs:ce]
            gt_chunk = gt_btchw[cs:ce]
            pn_out, sn_out = scorer_forward_pair(
                rt_chunk, self.posenet, self.segnet
            )
            with torch.no_grad():
                gp_out, gs_out = scorer_forward_pair(
                    gt_chunk, self.posenet, self.segnet
                )
            pose_chunks.append(pn_out["pose"])
            seg_chunks.append(sn_out)
            gt_pose_chunks.append(gp_out["pose"])
            gt_seg_chunks.append(gs_out)

        posenet_out = {"pose": torch.cat(pose_chunks, dim=0)}
        segnet_out = torch.cat(seg_chunks, dim=0)
        gt_pose_out = {"pose": torch.cat(gt_pose_chunks, dim=0)}
        gt_seg_out = torch.cat(gt_seg_chunks, dim=0)
        return posenet_out, segnet_out, gt_pose_out, gt_seg_out

    def train_epoch(
        self,
        mask_pairs: torch.Tensor,
        gt_pairs: torch.Tensor,
        ema: Optional[EMA] = None,
        roundtrip_noise_std: float = 0.5,
        pair_weights: Optional[torch.Tensor] = None,
        batch_size: int = 8,
    ) -> dict[str, float]:
        """Run one pass over (mask_pairs, gt_pairs), CHUNKED in mini-batches.

        Args:
            mask_pairs: (B, T, NUM_CLASSES, H, W) class-probability maps from
                the grayscale-LUT projection. T=2 for the canonical pair
                shape.
            gt_pairs: (B, T, H, W, 3) HWC ground-truth RGB pairs (uint8 or
                float; converted to float internally). Spatial size must
                match the SegMap output (384x512).
            ema: optional EMA shadow to update after each step.
            pair_weights: optional (B,) float tensor of per-pair training
                weights (Lane WC-S Curator outlier weighting). Used to
                weight the pose / seg distortion contributions per pair
                BEFORE the mean reduction. Must have length == B if
                supplied; broadcasting is rejected to avoid silent bugs.
            batch_size: number of pairs per mini-batch forward. Default 8
                matches the legacy `--batch-size 8` CLI default. Each
                mini-batch processes batch_size*T scorer frames at once.
                BUG CLASS B fix (2026-04-29): the previous implementation
                ran ALL B pairs through a single forward pass (B*T frames
                = 1200 frames at full res), which OOM'd T4's 14.56GB of
                VRAM with a 7.03GB allocation. Chunking + gradient
                accumulation keeps VRAM bounded by batch_size*T, while
                preserving per-epoch optimizer.step() semantics.

        Returns:
            dict with keys ``loss``, ``pose_dist``, ``seg_dist``,
            ``num_steps``, ``kl_aux`` (0.0 if loss_mode != kl_distill).
            Loss aggregation is SUM-then-divide-by-N over mini-batches
            (per CLAUDE.md "validate at every boundary"); the optimizer
            step is taken ONCE per call after all mini-batches accumulate
            their gradients.
        """
        self.model.train()
        # Lane LCT support: accept either (B, T, NUM_CLASSES, H, W) one-hot
        # OR (B, T, H, W) uint8 grayscale (when learnable_class_targets is
        # supplied). The grayscale path projects through the LCT-aware LUT
        # to get the 5-channel probability map.
        if mask_pairs.dim() == 4:
            if self.learnable_class_targets is None:
                raise ValueError(
                    f"mask_pairs has 4 dims (B, T, H, W) — "
                    f"requires learnable_class_targets to project via LUT. "
                    f"Otherwise pass (B, T, NUM_CLASSES, H, W) one-hot."
                )
            from tac.mask_grayscale_lut import create_gaussian_softmax_lut, NUM_CLASSES as _NC
            b, t, h, w = mask_pairs.shape
            num_classes = _NC
            # Compute LUT WITH gradient for raw_values (so backprop reaches
            # the learnable codebook). Use the trainer's device.
            targets = self.learnable_class_targets()  # (NUM_CLASSES,) on whatever device
            targets = targets.to(self.device)
            # Build the LUT inline (5 classes, sigma=15 per design).
            x = torch.arange(256, device=self.device, dtype=torch.float32).unsqueeze(1)
            squared_diff = (x - targets.unsqueeze(0)) ** 2
            sigma = 15.0
            logits = -squared_diff / (2.0 * sigma * sigma)
            lut = torch.softmax(logits, dim=1)  # (256, NUM_CLASSES) — differentiable in targets
            # mask_pairs uint8 -> long indexer
            gray_idx = mask_pairs.to(self.device, dtype=torch.long).clamp(0, 255)
            # Embedding lookup: (B, T, H, W) -> (B, T, H, W, NUM_CLASSES)
            probs = lut[gray_idx]
            # Reshape to (B, T, NUM_CLASSES, H, W) channel-first
            mask_pairs = probs.permute(0, 1, 4, 2, 3).contiguous()
        b, t, num_classes, h, w = mask_pairs.shape
        if num_classes != 5:
            raise ValueError(
                f"mask_pairs expects NUM_CLASSES=5 channel dim, got {num_classes}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        # Validate pair_weights shape ONCE before chunking so the same
        # contract holds whether or not we hit the Lane WC-S path.
        pw_full: Optional[torch.Tensor] = None
        if pair_weights is not None:
            if pair_weights.numel() != b:
                raise ValueError(
                    f"pair_weights length {pair_weights.numel()} != batch {b}; "
                    f"refusing to broadcast (Lane WC-S contract)."
                )
            pw_full = pair_weights.to(self.device, dtype=torch.float32)

        # Pre-zero gradients ONCE; mini-batch backward calls accumulate.
        self.optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        pose_sum = 0.0
        seg_sum = 0.0
        kl_sum = 0.0
        steps = 0

        # Iterate over [0, b) in chunks of batch_size.
        for start in range(0, b, batch_size):
            stop = min(start + batch_size, b)
            mb = stop - start  # mini-batch size in pairs
            mb_mask = mask_pairs[start:stop]
            mb_gt = gt_pairs[start:stop]

            # Flatten the (mb, T) -> single forward; frame indices preserve
            # the original global index space so the SegMap's per-frame
            # affine embedding stays aligned with this pair's true frame
            # location.
            masks_flat = mb_mask.reshape(mb * t, num_classes, h, w).to(self.device)
            # Original-frame-index range for this slice: [2*start, 2*stop).
            frame_indices = torch.arange(
                start * t, stop * t, device=self.device, dtype=torch.long
            )

            # Council C deep fix DF2: bf16 autocast wraps the renderer
            # forward + scorer call. PoseNet+SegNet are FROZEN
            # (requires_grad=False on every param), so they backprop
            # gradients only through the rendered branch — bf16 cast
            # cleanly without GradScaler (bf16 has fp32 exponent range,
            # GradScaler is only needed for fp16). KL distill loss is
            # computed inside the same autocast block since T >= 2.0
            # (validated by TrainConfig.kl_distill clause) keeps softmax
            # well-conditioned in bf16. Loss assembly stays in autocast
            # context — bf16-precision losses backprop into the renderer
            # weights at fp32 master copy via PyTorch's standard autocast
            # gradient promotion.
            # Council C deep fix DF2: bf16 autocast wraps the renderer
            # forward + scorer call + loss assembly. PoseNet+SegNet are
            # FROZEN (requires_grad=False on every param), so they only
            # backprop gradients through the rendered branch — bf16 cast
            # cleanly without GradScaler (bf16 has fp32 exponent range,
            # GradScaler is only needed for fp16). KL distill (which
            # makes another scorer call) runs inside the same autocast
            # block; T >= 2.0 (validated by TrainConfig.kl_distill clause)
            # keeps softmax well-conditioned in bf16. PyTorch promotes
            # loss to fp32 master copies via standard autocast gradient
            # promotion — the .backward() call below is autocast-clean.
            autocast_ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            )
            with autocast_ctx:
                rendered = self.model(masks_flat, frame_indices)  # (mb*T, 3, H, W) in [0,255]
                rendered_btchw = rendered.reshape(mb, t, 3, h, w)
                # Apply roundtrip BEFORE the scorer call. CLAUDE.md non-negotiable.
                rt_btchw = _eval_roundtrip_chain(
                    rendered_btchw, noise_std=roundtrip_noise_std
                )

                # Convert GT pairs (mb, T, H, W, 3) -> (mb, T, 3, H, W).
                gt_btchw = mb_gt.permute(0, 1, 4, 2, 3).contiguous().to(
                    self.device, dtype=torch.float32
                )

                # Council C deep fix DF3: per-pair scorer chunking.
                # When self.scorer_chunk > 0, split the dual scorer call
                # into chunks of N pairs (cuts FastViT attention-map
                # allocation by ~chunk_size). When 0 (legacy default),
                # passes through unchunked → bit-identical legacy path.
                (
                    posenet_out,
                    segnet_out,
                    gt_pose_out,
                    gt_seg_out,
                ) = self._scorer_forward_chunked(rt_btchw, gt_btchw)

                pose_diff_sq = (
                    posenet_out["pose"][..., :6] - gt_pose_out["pose"][..., :6]
                ).pow(2)
                seg_logits_pred = segnet_out
                with torch.no_grad():
                    seg_argmax_gt = gt_seg_out.argmax(dim=1)
                seg_ce_per = F.cross_entropy(
                    seg_logits_pred, seg_argmax_gt, reduction="none"
                )

                if pw_full is not None:
                    pw_mb = pw_full[start:stop]
                    if pose_diff_sq.shape[0] != mb:
                        raise RuntimeError(
                            f"pose output batch dim {pose_diff_sq.shape[0]} != "
                            f"mb={mb}; cannot apply pair_weights safely."
                        )
                    if seg_ce_per.shape[0] != mb:
                        raise RuntimeError(
                            f"seg CE batch dim {seg_ce_per.shape[0]} != mb={mb}; "
                            f"cannot apply pair_weights safely."
                        )
                    pose_dist = (
                        pose_diff_sq.mean(dim=tuple(range(1, pose_diff_sq.ndim)))
                        * pw_mb
                    ).sum() / pw_mb.sum().clamp_min(1e-8)
                    seg_dist = (
                        seg_ce_per.mean(dim=tuple(range(1, seg_ce_per.ndim)))
                        * pw_mb
                    ).sum() / pw_mb.sum().clamp_min(1e-8)
                else:
                    pose_dist = pose_diff_sq.mean()
                    seg_dist = seg_ce_per.mean()

                # Standard scorer loss formula (mirrors training.py / losses.py):
                #   total = segnet_weight * seg + sqrt(10 * pose + 1e-8)
                loss = (
                    self.config.segnet_loss_weight * seg_dist
                    + torch.sqrt(10.0 * pose_dist + 1e-8)
                )

                kl_aux_value = 0.0
                if self.config.loss_mode == "kl_distill":
                    # Use the SegNet-only helper to avoid the double-PoseNet
                    # gradient path. CLAUDE.md feedback_kl_distill_uses_roundtripped_frames
                    # is satisfied because we pass the SAME rt_btchw used above.
                    rt_hwc = rt_btchw.permute(0, 1, 3, 4, 2).contiguous()
                    gt_hwc = gt_btchw.permute(0, 1, 3, 4, 2).contiguous()
                    kl_loss, kl_value = kl_distill_segnet_only(
                        rt_hwc, gt_hwc, self.segnet,
                        temperature=float(
                            getattr(
                                self.config,
                                "kl_distill_temperature",
                                self.config.temperature_start,
                            )
                        ),
                    )
                    # Round 7 Defect #2 fix: read the weight from
                    # config (Lane G v3 default 0.002), not hard-code.
                    # The previous literal silently overrode any operator
                    # passing --kl-distill-weight 0.001 / 0.005.
                    kl_weight = float(getattr(self.config, "kl_distill_weight", 0.002))
                    loss = loss + kl_weight * kl_loss
                    kl_aux_value = float(kl_value)

            # Gradient accumulation: scale by 1/N_minibatches so that the
            # SUMMED gradient over the epoch matches the gradient of the
            # MEAN-of-mini-batch losses. This makes mini-batched training
            # mathematically equivalent (modulo BN stats) to the legacy
            # single-forward path when batch_size == b.
            # IMPORTANT: backward runs OUTSIDE the autocast context per
            # PyTorch idiom (https://pytorch.org/docs/stable/amp.html);
            # autograd handles the bf16/fp32 mixed gradient promotion.
            n_minibatches_estimate = (b + batch_size - 1) // batch_size
            (loss / n_minibatches_estimate).backward()

            loss_sum += float(loss.item())
            pose_sum += float(pose_dist.item())
            seg_sum += float(seg_dist.item())
            kl_sum += kl_aux_value
            steps += 1

            # Free intermediate activations between mini-batches.
            del rendered, rendered_btchw, rt_btchw, gt_btchw
            del posenet_out, segnet_out, gt_pose_out, gt_seg_out
            del pose_diff_sq, seg_logits_pred, seg_ce_per, seg_argmax_gt
            del loss, pose_dist, seg_dist

        # Single optimizer.step() per epoch over accumulated gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if ema is not None:
            ema.update(self.model)

        return {
            "loss": loss_sum / max(steps, 1),
            "pose_dist": pose_sum / max(steps, 1),
            "seg_dist": seg_sum / max(steps, 1),
            "kl_aux": kl_sum / max(steps, 1),
            "num_steps": steps,
        }

    def export_inference_state_dict(self, ema: Optional[EMA] = None) -> dict:
        """Return a state-dict shaped for the Selfcomp ``load_segmap`` consumer.

        The reference ``load_segmap`` (Selfcomp inflate.py L183-220) reads:

        * ``shared_latent_base`` (raw float tensor)
        * ``frame_affine_embedding.weight`` (raw float tensor)
        * ``layer_in.weight`` / ``layer_in.bias``
        * ``layer_out.weight`` / ``layer_out.bias``
        * ``blocks.{i}.conv{1,2}.weight`` / ``.bias``

        The block_fp_codec layer wraps weights into the
        ``weight_qint`` / ``weight_exponents`` pair at archive time; that's
        handled by ``tac.block_fp_codec.pack_payload_tar_xz``, NOT here.
        This exporter just returns the float tensors.
        """
        if ema is not None:
            # Snapshot the live state, swap in EMA, dump, swap back. We avoid
            # mutating the caller's model in-place beyond the exit invariant.
            live = {k: v.clone() for k, v in self.model.state_dict().items()}
            ema.apply(self.model)
            try:
                state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
            finally:
                self.model.load_state_dict(live)
        else:
            state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        # Sanity: ensure every reference key is present.
        expected_top = {"shared_latent_base", "frame_affine_embedding.weight",
                        "layer_in.weight", "layer_in.bias",
                        "layer_out.weight", "layer_out.bias"}
        missing = expected_top - set(state.keys())
        if missing:
            raise RuntimeError(
                f"export_inference_state_dict missing required keys: {sorted(missing)}"
            )
        return state


__all__ = [
    "CAMERA_SIZE",
    "SEGMAP_INPUT_SIZE",
    "ResidualBlock",
    "SegMap",
    "SegMapHomography",
    "SegMapTrainer",
]
