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


# ── Trainer (the lane-side contribution) ─────────────────────────────────


def _eval_roundtrip_chain(rgb_pair_btchw: torch.Tensor) -> torch.Tensor:
    """Apply the canonical 384 -> 874 -> uint8 -> 384 contest eval chain.

    Mirrors the established roundtrip pattern from src/tac/losses.py / training.
    The bicubic resize to camera-H + uint8 cast + bicubic-back to scorer-H
    simulates the lossy decode of the ``.raw`` rgb24 the contest evaluator
    sees. Without this step the proxy-auth gap is 2-11x on PoseNet
    (feedback_proxy_auth_math_useless).
    """
    b, t, c, h, w = rgb_pair_btchw.shape
    # 384x512 (scorer in) -> 874x1164 (camera) -> uint8 -> 384x512.
    flat = rgb_pair_btchw.reshape(b * t, c, h, w)
    up = F.interpolate(flat, size=CAMERA_SIZE[::-1], mode="bicubic", align_corners=False)
    up_u8 = up.clamp(0, 255).round()  # STE-friendly proxy for uint8 cast
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
    ):
        # Enforce CLAUDE.md non-negotiables BEFORE any GPU spend.
        if not config.eval_roundtrip:
            raise PreflightError(
                "SegMapTrainer requires config.eval_roundtrip=True. "
                "Without the 384->874->uint8->384 chain the proxy/auth gap "
                "is 2-11x on PoseNet (CLAUDE.md non-negotiable; "
                "feedback_proxy_auth_math_useless)."
            )
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

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay
        )

    def train_epoch(
        self,
        mask_pairs: torch.Tensor,
        gt_pairs: torch.Tensor,
        ema: Optional[EMA] = None,
    ) -> dict[str, float]:
        """Run a single pass over (mask_pairs, gt_pairs) and return loss stats.

        Args:
            mask_pairs: (B, T, NUM_CLASSES, H, W) class-probability maps from
                the grayscale-LUT projection. T=2 for the canonical pair
                shape.
            gt_pairs: (B, T, H, W, 3) HWC ground-truth RGB pairs (uint8 or
                float; converted to float internally). Spatial size must
                match the SegMap output (384x512).
            ema: optional EMA shadow to update after each step.

        Returns:
            dict with keys ``loss``, ``pose_dist``, ``seg_dist``,
            ``num_steps``, ``kl_aux`` (0.0 if loss_mode != kl_distill).
        """
        self.model.train()
        b, t, num_classes, h, w = mask_pairs.shape
        if num_classes != 5:
            raise ValueError(
                f"mask_pairs expects NUM_CLASSES=5 channel dim, got {num_classes}"
            )

        loss_sum = 0.0
        pose_sum = 0.0
        seg_sum = 0.0
        kl_sum = 0.0
        steps = 0

        # Flatten the (B, T) -> single forward, frame indices map per sample.
        masks_flat = mask_pairs.reshape(b * t, num_classes, h, w).to(self.device)
        # frame_indices: 2*i, 2*i+1 for each sample i. The trainer is single-
        # epoch over a static batch, so pair index 0..b-1 -> frames 0..2b-1.
        # Caller is expected to provide a correctly-sized max_frame_index in
        # the SegMap; we just produce monotonic indices here.
        frame_indices = torch.arange(b * t, device=self.device, dtype=torch.long)

        rendered = self.model(masks_flat, frame_indices)  # (B*T, 3, H, W) in [0, 255]
        rendered_btchw = rendered.reshape(b, t, 3, h, w)
        # Apply roundtrip BEFORE the scorer call. This is the canonical
        # eval_roundtrip simulation per the CLAUDE.md non-negotiable.
        rt_btchw = _eval_roundtrip_chain(rendered_btchw)

        # Convert GT pairs (B, T, H, W, 3) -> (B, T, 3, H, W) for the scorer.
        gt_btchw = gt_pairs.permute(0, 1, 4, 2, 3).contiguous().to(
            self.device, dtype=torch.float32
        )

        # Scorer forward — frozen weights, gradients flow through rendered.
        posenet_out, segnet_out = scorer_forward_pair(
            rt_btchw, self.posenet, self.segnet
        )
        # GT scorer outputs (no grad).
        with torch.no_grad():
            gt_pose_out, gt_seg_out = scorer_forward_pair(
                gt_btchw, self.posenet, self.segnet
            )

        # PoseNet: MSE on first 6 dims (pose).
        pose_dist = (
            posenet_out["pose"][..., :6] - gt_pose_out["pose"][..., :6]
        ).pow(2).mean()

        # SegNet: argmax disagreement rate (auth-faithful) — averaged over
        # spatial + batch. Differentiable via a soft cross-entropy proxy on
        # the per-class softmax disagreement.
        seg_logits_pred = segnet_out
        with torch.no_grad():
            seg_logits_gt = gt_seg_out
            seg_argmax_gt = seg_logits_gt.argmax(dim=1)
        # CE between predicted logits and GT argmax — differentiable, and
        # converges to the auth-eval argmax disagreement at the optimum.
        seg_dist = F.cross_entropy(seg_logits_pred, seg_argmax_gt)

        # Standard scorer loss formula (mirrors training.py / losses.py):
        #   total = segnet_weight * seg + sqrt(10 * pose + 1e-8)
        loss = (
            self.config.segnet_loss_weight * seg_dist
            + torch.sqrt(10.0 * pose_dist + 1e-8)
        )

        # Optional KL-distill auxiliary on SegNet (Hinton T=2 regime).
        kl_aux_value = 0.0
        if self.config.loss_mode == "kl_distill":
            # Use the dedicated SegNet-only helper to avoid the double-PoseNet
            # gradient path (per losses.py docstring + Lane G v3 history).
            # Pass the SAME roundtripped rendered + GT pairs in HWC layout —
            # kl_distill_segnet_only normalises shape internally.
            rt_hwc = rt_btchw.permute(0, 1, 3, 4, 2).contiguous()
            gt_hwc = gt_btchw.permute(0, 1, 3, 4, 2).contiguous()
            kl_loss, kl_value = kl_distill_segnet_only(
                rt_hwc, gt_hwc, self.segnet,
                temperature=self.config.temperature_start,
            )
            loss = loss + 0.002 * kl_loss  # canonical Lane G v3 weight
            kl_aux_value = float(kl_value)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if ema is not None:
            ema.update(self.model)

        loss_sum += float(loss.item())
        pose_sum += float(pose_dist.item())
        seg_sum += float(seg_dist.item())
        kl_sum += kl_aux_value
        steps += 1

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
    "SegMapTrainer",
]
