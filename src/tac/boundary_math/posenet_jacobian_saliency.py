# SPDX-License-Identifier: MIT
"""PoseNet pixel-Jacobian saliency field — the MEASURED frozen-oracle Jacobian for PTNC (task #61).

THE PTNC INVENTION (vs the #57 amortized carrier): the #57 carrier trained with a DENSE pixel-MSE anchor
(weight >= 0.5 always), spending capacity reproducing pose-IRRELEVANT luma — the source of the d_pose
0.0036 ceiling. PTNC replaces the dense anchor with a **PoseNet-Jacobian-saliency-WEIGHTED recon anchor**
(USC IDSE, but with the EXACT measured atlas Jacobian of the FROZEN oracle, not a per-image Taylor
approximation). This module computes the per-pixel weight field:

    w_pix(x, y)  ~  || d pose6 / d (frame pixel at (x,y)) ||   (the pose OUTPUT sensitivity per input
                                                                pixel; high = pose-relevant, ~0 = pose-null)

THE SCORER FACTS (upstream/modules.py, verified):
  * PoseNet.preprocess_input: resize 384x512 (bilinear) -> rgb_to_yuv6 (HALF-res luma 4ch + 2x2-box
    chroma 2ch) -> 12ch (2 frames x 6); forward -> pose head; d_pose = MSE on FIRST 6 of the 12 dims.
  * The Jacobian here is of the 6 SCORED dims w.r.t. the camera-res RGB pixels. It is computed by
    backprop through the DIFFERENTIABLE yuv6 path (``tac.differentiable_eval_roundtrip``) because the
    upstream ``rgb_to_yuv6`` is ``@torch.no_grad()``/in-place and otherwise severs the gradient.
  * The saliency is the Jacobian NORM (sensitivity), NOT the gradient of d_pose at the GT operating point
    (which VANISHES because pose(comp)==pose(gt) there). The norm tells us which pixels the 6 scored dims
    READ; the carrier must paint luma there and is FREE to be wrong elsewhere (the pose-null).

FAIL-CLOSED (the YUV6-severed-gradient signature): if the computed Jacobian-norm field is identically
zero, the differentiable path is broken (gradient severed) and a non-reachable gradient must NEVER
masquerade as "all pose-null". :func:`compute_posenet_pixel_saliency` raises in that case.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): exact frozen CPU-torch PoseNet (NEVER mps). The torch path MEASURES
the field once at the GT operating point (the atlas Jacobian); the numpy-portable :func:`saliency_to_weight_map`
turns the measured field into the inflate-irrelevant TRAINING weight (the weight is a training-time
object; the inflate decode is the carrier numpy forward, scorer-free). Evidence ``[local CPU-torch
advisory]``. Non-promotable.

NO-FAKE (class 8): the saliency is the EXACT backprop Jacobian norm of the frozen PoseNet (not a
fabricated constant or a hand-drawn mask); :func:`saliency_to_weight_map` is a pure deterministic
transform of that measured field; the IDENTITY weight map (uniform) is provided explicitly so a test can
prove the Jacobian field is load-bearing (replacing it with identity recovers dense-MSE behavior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512  # PoseNet resize target (segnet_model_input_size = (W,H) = (512,384))


class PoseNetSaliencyError(ValueError):
    """Raised when the measured Jacobian field is degenerate (severed gradient)."""


@dataclass(frozen=True)
class PixelSaliencyField:
    """A measured per-pixel PoseNet sensitivity field for one frame slot of one pair.

    ``saliency`` is (H, W) float >= 0: the L2 norm over the 6 scored pose dims of the per-pixel input
    Jacobian (summed over the 3 RGB channels). High = pose-relevant; ~0 = pose-null (free to be wrong).
    The field is measured at the GT operating point (the frozen-oracle atlas Jacobian).
    """

    saliency: np.ndarray  # (H, W) float32, >= 0
    h: int
    w: int
    frame_slot: int  # 0 or 1 (which frame of the pair this Jacobian is w.r.t.)
    compute_path: str  # "cpu_torch" — NEVER mps
    nonzero_fraction: float
    max_value: float

    def to_summary(self) -> dict[str, float]:
        s = self.saliency
        return {
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "p90": float(np.percentile(s, 90)),
            "max": float(self.max_value),
            "nonzero_fraction": float(self.nonzero_fraction),
        }


def saliency_to_weight_map(
    field: PixelSaliencyField,
    *,
    floor: float = 0.02,
    gamma: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Turn a measured saliency field into the PTNC recon-loss weight map (H, W) float32.

    ``w = floor + (1 - floor) * (saliency / max(saliency))**gamma``. The ``floor`` keeps a small
    nonzero weight everywhere (numerical stability + a faint global anchor); the bulk of the weight
    concentrates on the pose-relevant pixels. ``gamma > 1`` sharpens onto the very-high-saliency tube.

    The mean is renormalised to 1.0 (when ``normalize``) so the PTNC loss magnitude is comparable to a
    plain mean-MSE — the weight REDISTRIBUTES capacity, it does not just rescale the loss.

    NO-FAKE: this is a pure deterministic transform of the MEASURED field; with a uniform input
    saliency it returns a uniform map (== dense MSE), which is exactly :func:`identity_weight_map`.
    """

    s = np.asarray(field.saliency, dtype=np.float64)
    smax = float(s.max())
    if smax <= 0.0:
        raise PoseNetSaliencyError(
            "saliency field is identically zero — severed-gradient signature; refusing to "
            "manufacture an all-pose-null weight map."
        )
    norm = (s / smax) ** float(gamma)
    w = float(floor) + (1.0 - float(floor)) * norm
    if normalize:
        w = w / float(w.mean())
    return w.astype(np.float32)


def identity_weight_map(h: int, w: int) -> np.ndarray:
    """Uniform weight map (the dense-MSE control). PTNC with this == the #57 dense anchor."""

    return np.ones((h, w), dtype=np.float32)


def compute_posenet_pixel_saliency(
    posenet,
    gt_frame0_chw,
    gt_frame1_chw,
    *,
    frame_slot: int = 0,
    n_probe_dirs: int = 0,
    seed: int = 0,
    coarse_grid: int = 0,
):
    """Measure the per-pixel PoseNet Jacobian-norm field for one frame slot of a pair.

    ``posenet``: a frozen ``upstream.modules.PoseNet`` on CPU with the differentiable yuv6 patch ACTIVE
    (the caller patches via ``tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally`` for the
    duration). ``gt_frameN_chw``: (3, H, W) float [0, 255] camera-res GT frames.

    Method (the exact atlas Jacobian, vectorised): backprop each of the 6 scored pose dims to the input
    frame, accumulate the per-pixel L2 norm over the 6 dims and 3 channels. For speed we use the standard
    Jacobian-norm estimator: sum of squared grads of the 6 outputs == diag of J^T J per pixel; we take 6
    backward passes (cheap on CPU for 6 outputs) and sum the squared per-pixel input grads, then sqrt.

    Returns a :class:`PixelSaliencyField` at CAMERA resolution (the resize-bilinear is differentiable so
    the grad flows back to camera-res pixels). FAIL-CLOSED if the field is identically zero.
    """

    import torch

    if frame_slot not in (0, 1):
        raise ValueError(f"frame_slot must be 0 or 1, got {frame_slot}")

    dev = torch.device("cpu")
    f0 = torch.as_tensor(gt_frame0_chw, dtype=torch.float32, device=dev).clone()
    f1 = torch.as_tensor(gt_frame1_chw, dtype=torch.float32, device=dev).clone()
    h, w = f0.shape[-2], f0.shape[-1]

    target = f0 if frame_slot == 0 else f1
    target.requires_grad_(True)

    # (1,2,3,H,W) — PoseNet.preprocess_input expects b t c h w.
    x = torch.stack([f0, f1]).unsqueeze(0)
    pin = posenet.preprocess_input(x)
    pose6 = posenet(pin)["pose"][..., :6].reshape(-1)  # (6,)

    sq_accum = torch.zeros_like(target)  # (3,H,W)
    for k in range(6):
        g = torch.autograd.grad(pose6[k], target, retain_graph=(k < 5), create_graph=False)[0]
        sq_accum = sq_accum + g.detach() ** 2
    # per-pixel L2 norm over 6 dims (already summed) and 3 channels.
    per_pixel = torch.sqrt(sq_accum.sum(dim=0))  # (H,W)
    sal = per_pixel.cpu().numpy().astype(np.float32)

    smax = float(sal.max())
    if smax <= 0.0:
        raise PoseNetSaliencyError(
            "PoseNet pixel-Jacobian field is identically zero — the differentiable yuv6 path is not "
            "active (gradient severed). Patch rgb_to_yuv6 before measuring; refusing to fake pose-null."
        )
    nz = float(np.mean(sal > (1e-6 * smax)))
    return PixelSaliencyField(
        saliency=sal, h=int(h), w=int(w), frame_slot=int(frame_slot),
        compute_path="cpu_torch", nonzero_fraction=nz, max_value=smax,
    )


def downsample_field(field: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a (H,W) field by ``factor`` (for a coarse training weight on a downsampled grid)."""

    if factor <= 1:
        return np.asarray(field, dtype=np.float32)
    h, w = field.shape
    hh, ww = h // factor, w // factor
    crop = field[: hh * factor, : ww * factor]
    return crop.reshape(hh, factor, ww, factor).mean(axis=(1, 3)).astype(np.float32)


__all__ = [
    "PixelSaliencyField",
    "PoseNetSaliencyError",
    "compute_posenet_pixel_saliency",
    "downsample_field",
    "identity_weight_map",
    "saliency_to_weight_map",
]
