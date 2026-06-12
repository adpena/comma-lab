#!/usr/bin/env python3
"""R10 lever-interaction SIGN/monotonicity probe (the 5 Layer-2 levers).

The R10 adversarial lens: do levers 2 (seg surrogate) + 3 (pose-FiLM) + 5
(margin weight) COMPOSE (each gradient points the same way as it does alone) or
FIGHT (the combined gradient REVERSES a single lever's score-improving
direction)? The live distortion arm runs levers 2+3+5 ON together; if the margin
lever's boundary up-weighting opposed the pose-FiLM gradient, or the seg
surrogate cancelled the pose term, the multi-day descent would stall — and no
prior round MEASURED the SIGN of the combined-vs-individual gradient on the REAL
frozen scorer.

The math being tested (the falsifiable hypothesis):
  L_combined = w_seg * seg_l + w_pose * pose_l   (the driver's actual loss)
  g_combined = dL_combined/d(params)
  g_seg_only = w_seg * d(seg_l)/d(params)        (lever 2[+5] alone)
  g_pose_only = w_pose * d(pose_l)/d(params)     (the pose term alone)
COMPOSE  <=>  <g_combined, g_seg_only> >= 0  AND  <g_combined, g_pose_only> >= 0
i.e. taking the combined step does NOT move UP-hill on either individual term's
descent direction. (A negative inner product would mean the combined gradient
has a component that INCREASES that term = the levers FIGHT.)

Lever-5's specific test: g_seg_margin (margin-weighted) vs g_seg_plain
(unweighted) — the margin re-weighting must not REVERSE the plain seg descent
direction (cos > 0), only RESHAPE where the gradient concentrates.

Authority: real frozen scorer (EfficientNet-B2 SegNet + FastViT PoseNet via
load_frozen_distortion_net), CPU-TRUSTED. RESEARCH-ONLY tiny slice ->
[contest-CPU advisory] NON-PROMOTABLE. Gradient-DIRECTION claim only (not a score
claim). NO daemon touched (writes only .omx/tmp/r10_*).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tac.torch_vehicle.curriculum import StageSpec
from tac.torch_vehicle.driver import (
    _EVAL_H,
    _EVAL_W,
    TorchVehicleConfig,
    TorchVehicleDriver,
    _seg_loss_for_spec,
    import_vendored_bundle,
)
from tac.torch_vehicle.scorer_context import RealScorerContext

_VIDEO = "upstream/videos/0.mkv"


def _ce(s, t):
    return F.cross_entropy(s, t)


def _stage(**ov) -> StageSpec:
    base = dict(
        name="r10",
        epochs=2,
        seg_loss_fn=_ce,
        eval_every=1,
        batch_size=4,
        ema_decay=0.999,
        use_muon=False,
        adamw_lr=1e-3,
        muon_lr=2e-4,
        muon_weight_decay=0.0,
        latent_lr_mult=10.0,
        grad_clip=1e9,
        grad_clip_muon=1e9,
        lr_floor_ratio=5e-6,
        seg_weight=100.0,
        pose_weight=1.0,
        cat_lambda=0.0,
        cat_sigma=0.2,
        use_qat=False,
        init_latents_random=True,
    )
    base.update(ov)
    return StageSpec(**base)


def _render_roundtrip(decoder, latents, idx, *, film: bool):
    """The driver's exact render + uint8-roundtrip pipeline (lines 584-604)."""
    decoded_pair = decoder(latents[idx], idx) if film else decoder(latents[idx])
    B = len(idx)
    flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
    up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
    down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
    decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
    dc = decoded_bhwc.clamp(0, 255)
    dr = dc.round()
    return dc + (dr - dc).detach()  # straight-through round


def _grad_vec(params, loss):
    """Flatten dL/d(params) into a single vector (None grads -> zeros)."""
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    flat = []
    for p, g in zip(params, grads, strict=False):
        flat.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(flat)


def _cos(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--out", default=".omx/tmp/r10_lever_interaction")
    ap.add_argument("--margin-tau", type=float, default=2.0)
    ap.add_argument("--seg-temp", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # Real frozen scorer (cached targets -> instant); CPU authority (no MPS).
    sc = RealScorerContext(
        _VIDEO,
        device="cpu",
        max_pairs=args.n_pairs,
        targets_cache=str(out / "targets_cache"),
    )
    v = import_vendored_bundle()
    cfg = TorchVehicleConfig(
        base_channels=20,
        latent_dim=28,
        out_dir=str(out / "run"),
        device="cpu",
        pose_film_enabled=True,  # Lever 3 ON
        seed=0,
    )
    spec = _stage(seg_surrogate="soft_cosine", margin_weight_tau=args.margin_tau)
    driver = TorchVehicleDriver(cfg, scorer=sc, vendored=v, curriculum=[spec])

    # Build a FiLM-wrapped decoder + latents (the live arm's shape).
    decoder = driver._new_decoder(device=torch.device("cpu"))
    decoder.set_stored_pose(sc.pose_targets[: args.n_pairs].cpu())
    n = int(args.n_pairs)
    latents = torch.nn.Parameter(torch.randn(n, 28) * 0.1)
    idx = torch.arange(min(4, n))
    params = [p for p in decoder.parameters() if p.requires_grad] + [latents]

    def seg_pose(film: bool):
        frames = _render_roundtrip(decoder, latents, idx, film=film)
        seg_out, pose6 = sc.seg_pose_forward(frames)
        return seg_out, pose6

    # --- Individual lever gradient directions (each computed in isolation) ---
    # g_pose_only: the pose term alone (Lever-3 FiLM render conditions it).
    seg_out, pose6 = seg_pose(film=True)
    pose_mse = F.mse_loss(pose6, sc.pose_targets[idx])
    pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
    g_pose_only = _grad_vec(params, spec.pose_weight * pose_l)

    # g_seg_plain: the seg surrogate WITHOUT the margin lever (Lever-2 alone).
    seg_out, _ = seg_pose(film=True)
    spec_plain = _stage(seg_surrogate="soft_cosine", margin_weight_tau=None)
    seg_l_plain = _seg_loss_for_spec(
        spec_plain, seg_out, sc.seg_targets_hard[idx], temperature=args.seg_temp
    )
    g_seg_plain = _grad_vec(params, spec_plain.seg_weight * seg_l_plain)

    # g_seg_margin: the seg surrogate WITH the margin lever (Lever-2 + Lever-5).
    seg_out, _ = seg_pose(film=True)
    seg_l_margin = _seg_loss_for_spec(
        spec, seg_out, sc.seg_targets_hard[idx], temperature=args.seg_temp
    )
    g_seg_margin = _grad_vec(params, spec.seg_weight * seg_l_margin)

    # --- The COMBINED gradient (the driver's actual loss; levers 2+3+5 together) ---
    seg_out, pose6 = seg_pose(film=True)
    seg_l_c = _seg_loss_for_spec(
        spec, seg_out, sc.seg_targets_hard[idx], temperature=args.seg_temp
    )
    pose_mse_c = F.mse_loss(pose6, sc.pose_targets[idx])
    pose_l_c = torch.sqrt(10.0 * pose_mse_c + 1e-12)
    loss_c = spec.seg_weight * seg_l_c + spec.pose_weight * pose_l_c
    g_combined = _grad_vec(params, loss_c)

    # --- The SIGN tests ---
    # (1) Combined does NOT move up-hill on the pose term.
    ip_combined_pose = float(g_combined @ g_pose_only)
    cos_combined_pose = _cos(g_combined, g_pose_only)
    # (2) Combined does NOT move up-hill on the margin-weighted seg term.
    ip_combined_seg = float(g_combined @ g_seg_margin)
    cos_combined_seg = _cos(g_combined, g_seg_margin)
    # (3) Lever-5 (margin) RESHAPES but does not REVERSE the plain seg direction.
    cos_margin_vs_plain = _cos(g_seg_margin, g_seg_plain)
    # (4) The combined gradient is genuinely the SUM (sanity: it equals g_seg_margin
    #     + g_pose_only to numerical precision — the levers compose by gradient sum).
    g_sum = g_seg_margin + g_pose_only
    sum_decomposition_relerr = float(
        (g_combined - g_sum).norm() / (g_combined.norm() + 1e-12)
    )

    verdict = {
        "scorer_class": type(sc).__name__,
        "is_real_scorer": type(sc).__name__ == "RealScorerContext",
        "n_pairs": n,
        "seg_temp_tail": args.seg_temp,
        "margin_tau": args.margin_tau,
        # The four findings:
        "combined_dot_pose_only": ip_combined_pose,
        "combined_cos_pose_only": cos_combined_pose,
        "combined_descends_pose": cos_combined_pose > 0.0,
        "combined_dot_seg_margin": ip_combined_seg,
        "combined_cos_seg_margin": cos_combined_seg,
        "combined_descends_seg": cos_combined_seg > 0.0,
        "margin_vs_plain_cos": cos_margin_vs_plain,
        "margin_reshapes_not_reverses_seg": cos_margin_vs_plain > 0.0,
        "sum_decomposition_relerr": sum_decomposition_relerr,
        "composes_by_gradient_sum": sum_decomposition_relerr < 1e-4,
        # gradient norms (non-degenerate check)
        "norm_g_combined": float(g_combined.norm()),
        "norm_g_seg_margin": float(g_seg_margin.norm()),
        "norm_g_seg_plain": float(g_seg_plain.norm()),
        "norm_g_pose_only": float(g_pose_only.norm()),
    }
    verdict["ALL_COMPOSE"] = bool(
        verdict["combined_descends_pose"]
        and verdict["combined_descends_seg"]
        and verdict["margin_reshapes_not_reverses_seg"]
        and verdict["composes_by_gradient_sum"]
    )
    (out / "r10_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["ALL_COMPOSE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
