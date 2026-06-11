# SPDX-License-Identifier: MIT
"""MEASURED correlation between the differentiable d_seg surrogates and the EXACT
non-differentiable argmax-flip d_seg set-functional, on REAL video frames.

The contest SegNet term is a NON-differentiable set functional
(``upstream/modules.py:111-113``)::

    d_seg = mean( argmax(SegNet(rendered_last_frame)) != argmax(SegNet(gt_last_frame)) )

Any smaller-basis retrain (Cool-Chic, IB carrier, KAN) or learned postfilter that
wants to descend the EXACT d_seg needs a DIFFERENTIABLE SURROGATE that *provably*
correlates with this rate. The PR95-faithful margin surrogates already live in
``live_segnet_loss.py`` and were proven to DESCEND exact d_seg during training
(``.omx/research/inert_loop_fix_20260610T193900Z.md``: 0.508 -> 0.081 on real
EfficientNet-B2). That is a *descent* proof on ONE trajectory.

This module is the orthogonal, decisive measurement the DAG (THREAD-B SHARED NEW
MATH) names: across a POPULATION of real rendered-vs-GT frame fidelities, does a
LOWER surrogate imply a LOWER exact d_seg? It reports the rank-correlation
(Spearman) + the OLS slope of exact-d_seg-on-surrogate. A surrogate whose
rank-correlation is not strongly positive is a FAKE training signal — minimizing
it would not move the score (the inert-loop bug class, ledger #75/#76).

Authority discipline: every number this module emits is ``[macOS-CPU advisory]``
(torch CPU, the EXACT authority decode path; NO MPS). GT decode is ONLY via
``upstream/frame_utils.yuv420_to_rgb`` (PyAV rgb24 manufactures phantom pose).
The exact d_seg it computes IS the quantity ``evaluate.py`` charges — it is not a
proxy of a proxy.

Reuse (search-and-familiarize, per CLAUDE.md):
- ``live_segnet_loss.py``  — the surrogates under test + ``exact_d_seg_from_logits``.
- ``targets.py``           — ``load_frozen_distortion_net`` (frozen, YUV6-patched).
- ``analysis/segnet_boundary_marginals.py`` — boundary-band weighting (the risk axis).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import einops
import torch

from tac.score_aware_loop.live_segnet_loss import (
    STAGE_SEG_LOSS_FNS,
    exact_d_seg_from_logits,
)

# ---------------------------------------------------------------------------
# Frame population: graded perturbations of REAL GT frames.
# ---------------------------------------------------------------------------
# We do NOT train (GPU busy + the descent proof already exists). Instead we build
# a POPULATION of "rendered" frames at known, graded fidelity by perturbing the
# real GT frames with degradations that mimic real decoder/codec error
# (additive luminance noise, spatial blur, uint8 requantization). Each
# perturbation level produces a frame whose EXACT d_seg vs the GT is a genuine,
# real-content argmax-flip rate. Correlating the surrogate against that exact rate
# IS the population correlation the smaller-basis nodes care about: "if my carrier
# lowers the surrogate, does the exact d_seg fall too?"


@dataclass(frozen=True)
class Perturbation:
    """A named, reproducible frame degradation parameterized by strength ``s``."""

    name: str
    fn: Callable[[torch.Tensor, float], torch.Tensor]


# Perturbations operate in the NATIVE scorer input space: uint8-range
# ``(H, W, C=3)`` float in [0, 255] (the raw ``yuv420_to_rgb`` output cast to
# float). This is exactly the space the contest scorer consumes; degrading here
# (not in a normalized 0-1 space) keeps the population faithful to real decoder
# error magnitudes.


def _additive_gauss(frame: torch.Tensor, s: float) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    noise = torch.empty_like(frame).normal_(0.0, 1.0, generator=g)
    return (frame + (s * 255.0) * noise).clamp(0.0, 255.0)


def _box_blur(frame: torch.Tensor, s: float) -> torch.Tensor:
    # s in [0,1] -> kernel radius up to 4 (odd kernel via reflect pad + avg pool).
    radius = round(s * 4.0)
    if radius < 1:
        return frame.clone()
    k = 2 * radius + 1
    x = einops.rearrange(frame, "h w c -> 1 c h w")
    x = torch.nn.functional.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=1)
    return einops.rearrange(x, "1 c h w -> h w c").clamp(0.0, 255.0)


def _uint8_requant(frame: torch.Tensor, s: float) -> torch.Tensor:
    # Coarsen the quantization grid: s in [0,1] -> 2..256 levels.
    levels = max(2, round(256 * (1.0 - 0.96 * s)))
    step = 255.0 / (levels - 1)
    return (torch.round(frame / step) * step).clamp(0.0, 255.0)


DEFAULT_PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("additive_gauss", _additive_gauss),
    Perturbation("box_blur", _box_blur),
    Perturbation("uint8_requant", _uint8_requant),
)

DEFAULT_STRENGTHS: tuple[float, ...] = (
    0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32, 0.45, 0.6,
)


@dataclass
class CorrelationResult:
    """Per-surrogate correlation verdict + the raw (surrogate, exact) cloud."""

    surrogate_name: str
    spearman_rho: float
    pearson_r: float
    ols_slope: float
    ols_intercept: float
    n_points: int
    # Suprafloor correlation: Spearman restricted to points where exact d_seg is
    # above the argmax-sampling-noise floor (where ordering is actually
    # meaningful — below the floor every render already passes, so the ranking is
    # noise, not signal). This is the number a training signal cares about.
    spearman_rho_suprafloor: float = float("nan")
    n_suprafloor: int = 0
    floor: float = 0.0
    surrogate_values: list[float] = field(default_factory=list)
    exact_d_seg_values: list[float] = field(default_factory=list)

    def is_strong_positive(self, rho_min: float = 0.9) -> bool:
        """A usable surrogate has strongly-positive rank-correlation with exact d_seg.

        Uses the suprafloor Spearman when available (the meaningful regime), else
        the full-cloud Spearman.
        """
        rho = (
            self.spearman_rho_suprafloor
            if self.n_suprafloor >= 8
            else self.spearman_rho
        )
        return rho >= rho_min and self.ols_slope > 0.0

    def as_jsonable(self) -> dict:
        return {
            "surrogate_name": self.surrogate_name,
            "spearman_rho": self.spearman_rho,
            "spearman_rho_suprafloor": self.spearman_rho_suprafloor,
            "n_suprafloor": self.n_suprafloor,
            "floor": self.floor,
            "pearson_r": self.pearson_r,
            "ols_slope": self.ols_slope,
            "ols_intercept": self.ols_intercept,
            "n_points": self.n_points,
            "is_strong_positive": self.is_strong_positive(),
            "surrogate_values": self.surrogate_values,
            "exact_d_seg_values": self.exact_d_seg_values,
        }


def _spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr

    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def _pearson_and_ols(x: list[float], y: list[float]) -> tuple[float, float, float]:
    tx = torch.tensor(x, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    if tx.numel() < 2 or tx.std() == 0 or ty.std() == 0:
        return float("nan"), float("nan"), float("nan")
    xc, yc = tx - tx.mean(), ty - ty.mean()
    r = float((xc @ yc) / (xc.norm() * yc.norm()))
    slope = float((xc @ yc) / (xc @ xc))
    intercept = float(ty.mean() - slope * tx.mean())
    return r, slope, intercept


def _segnet_logits(distortion_net: torch.nn.Module, frame_hwc: torch.Tensor) -> torch.Tensor:
    """Run the frozen SegNet on a single GT frame ``(H, W, C=3)`` -> logits ``(1, 5, h, w)``.

    Goes through ``DistortionNet.preprocess_input`` (which rearranges
    ``b t h w c -> b t c h w`` and slices the last frame, resizing to 512x384),
    exactly as the contest scorer does. ``frame_hwc`` is the raw
    ``frame_utils.yuv420_to_rgb`` output (uint8 ``(874, 1164, 3)``).
    """
    segnet = distortion_net.segnet
    # (1, T=1, H, W, C) -> upstream preprocess rearranges + resizes + slices last.
    x = frame_hwc.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, C)
    x = einops.rearrange(x, "b t h w c -> b t c h w", b=1, t=1, c=3).float()
    segnet_in = segnet.preprocess_input(x)
    return segnet(segnet_in)


def measure_surrogate_correlation(
    *,
    video_path: str = "upstream/videos/0.mkv",
    max_frames: int = 8,
    surrogate_names: tuple[str, ...] = tuple(STAGE_SEG_LOSS_FNS.keys()),
    perturbations: tuple[Perturbation, ...] = DEFAULT_PERTURBATIONS,
    strengths: tuple[float, ...] = DEFAULT_STRENGTHS,
    tau: float = 0.3,
    upstream_dir: str = "upstream",
    device: str = "cpu",
    held_out_split: bool = True,
) -> dict[str, CorrelationResult]:
    """Measure surrogate<->exact-d_seg correlation across a real-frame population.

    For each real GT frame, for each (perturbation, strength), build a degraded
    frame, run BOTH the exact scorer (true argmax-flip d_seg vs the GT argmax) and
    each surrogate (margin loss on the degraded SegNet logits vs the GT argmax
    targets). Accumulate the (surrogate, exact) cloud and report Spearman + OLS.

    ``held_out_split=True`` measures correlation on the first half of frames and
    re-checks Spearman on the held-out second half (anti-overfit / anti-artifact).

    Returns a dict ``{surrogate_name: CorrelationResult}`` (the train split). When
    ``held_out_split``, each result carries a sibling under key
    ``f"{name}__heldout"``.
    """
    if device != "cpu":
        raise ValueError("CPU-only (MPS never authority; CUDA GPU is busy).")
    root = Path(upstream_dir).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import av
    from frame_utils import yuv420_to_rgb

    from tac.score_aware_loop.targets import load_frozen_distortion_net

    dnet = load_frozen_distortion_net(upstream_dir=upstream_dir, device=device)

    # Decode real GT frames (last-frame-of-pair semantics not needed here: the
    # surrogate compares one degraded frame against its own GT, which is exactly
    # the rendered-vs-GT comparison d_seg charges per-pair).
    frames: list[torch.Tensor] = []
    container = av.open(str(video_path))
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            # Raw uint8 (H, W, C) -> float in [0, 255] (native scorer-input space).
            frames.append(yuv420_to_rgb(frame).to(device).float())
            if len(frames) >= max_frames:
                break
    container.close()
    if len(frames) < 2:
        raise ValueError("need >= 2 real frames for a correlation population")

    split = len(frames) // 2 if held_out_split else len(frames)
    frame_groups = {"train": frames[:split]}
    if held_out_split:
        frame_groups["heldout"] = frames[split:]

    out: dict[str, CorrelationResult] = {}
    for group_name, group_frames in frame_groups.items():
        # Per-surrogate clouds.
        clouds: dict[str, tuple[list[float], list[float]]] = {
            n: ([], []) for n in surrogate_names
        }
        for gt_frame in group_frames:
            # GT argmax targets (the d_seg reference) from the clean GT frame.
            with torch.inference_mode():
                gt_logits = _segnet_logits(dnet, gt_frame)
                gt_argmax = gt_logits.argmax(dim=1)  # (1, h, w)
            for pert in perturbations:
                for s in strengths:
                    degraded = pert.fn(gt_frame, float(s))
                    with torch.inference_mode():
                        deg_logits = _segnet_logits(dnet, degraded)
                        exact = exact_d_seg_from_logits(deg_logits, gt_argmax)
                    for name in surrogate_names:
                        fn = STAGE_SEG_LOSS_FNS[name]
                        with torch.inference_mode():
                            if name == "ce_seg_loss":
                                sval = float(fn(deg_logits, gt_argmax).item())
                            else:
                                sval = float(fn(deg_logits, gt_argmax, tau=tau).item())
                        clouds[name][0].append(sval)
                        clouds[name][1].append(exact)
        for name in surrogate_names:
            xs, ys = clouds[name]
            rho = _spearman(xs, ys)
            r, slope, intercept = _pearson_and_ols(xs, ys)
            # Suprafloor: keep points whose exact d_seg exceeds the argmax-sampling
            # floor (default 5e-3 ~ the contest operating-point neighborhood; below
            # this the render already "passes" and ordering is noise).
            floor = 5e-3
            sup = [(sv, ev) for sv, ev in zip(xs, ys, strict=True) if ev > floor]
            if len(sup) >= 8:
                sx = [p[0] for p in sup]
                sy = [p[1] for p in sup]
                rho_sup = _spearman(sx, sy)
            else:
                rho_sup = float("nan")
            key = name if group_name == "train" else f"{name}__heldout"
            out[key] = CorrelationResult(
                surrogate_name=key,
                spearman_rho=rho,
                spearman_rho_suprafloor=rho_sup,
                n_suprafloor=len(sup),
                floor=floor,
                pearson_r=r,
                ols_slope=slope,
                ols_intercept=intercept,
                n_points=len(xs),
                surrogate_values=xs,
                exact_d_seg_values=ys,
            )
    return out
