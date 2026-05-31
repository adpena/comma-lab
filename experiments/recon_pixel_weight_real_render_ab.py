# SPDX-License-Identifier: MIT
"""WAVE-5E real-contest-shaped A/B: the canonical ``recon_pixel_weight`` channel.

CONFIRMS (or falsifies) the sister #1587 detector-informed lever on a REAL
contest-shaped reconstruction, at the RECON-LOSS-WEIGHT surface (the codex-named
``recon_pixel_weight`` channel) rather than the sparse-delta packing surface.

THE SETUP (real, $0, NON-PROMOTABLE [macOS-CPU advisory]):
  * Decode REAL frames from ``upstream/videos/0.mkv``.
  * DEGRADE each to a moderate-d_seg render via a grayscale-LUT-style palette
    quantization (a real PR-class artifact: the natural operating point where the
    sister mechanistic boundary said the lever HELPS — moderate seg degradation,
    NOT the extreme-collapse regime).
  * Optimize a per-pixel correction ``delta`` (L_inf-clamped) by gradient descent
    on a RECON-WEIGHTED loss ``mean(w * (render+delta - gt)^2) / mean(w)`` for
    three weight maps W:
      - ``uniform``          : w = 1 everywhere (the canonical default-OFF recon).
      - ``s_uniward_texture``: w = S-UNIWARD texture cost (textured regions are
                               detector-invisible — the rate-attack prior).
      - ``full_grid_saliency``: w = |dL_seg/dpixel| measured SegNet response
                               (the #1587 winning detector-informed map).
  * Measure REAL contest d_seg (argmax-flip rate vs GT) of each corrected render
    at MATCHED iterations + MATCHED L_inf budget.

FALSIFIABLE CLAIM (Catalog #307): the full-grid-saliency recon-weight produces
LOWER real d_seg than uniform AND than s_uniward_texture at the moderate operating
point -> the recon_pixel_weight channel is a confirmed reusable lever for the
gradient-driven solver. If it does NOT beat uniform on a real render, the #1587
result was surface-specific (sparse-delta ranking only) -> report honestly.

This is a PURE-PYTORCH harness (the channel's MATH is the MLX bundle's
``_weighted_recon``; here we exercise the SAME re-weighting identity end-to-end
against the REAL SegNet, which is PyTorch -- no MLX renderer needed to confirm the
LEVER). The MLX bundle channel + this PyTorch confirm share the identical
re-weight formula ``mean(w*sq)/mean(w)``.

NON-PROMOTABLE: per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#341/#127/#323.
Reuses the canonical #1587 saliency primitive + S-UNIWARD texture (no rebuild).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
UPSTREAM = REPO / "upstream"

from tac.scorer import load_default_scorers
from tac.uniward_delta import compute_uniward_cost_map


def _decode_frames(mkv: Path, n: int, stride: int) -> np.ndarray:
    """Decode ``n`` real frames (stride apart) from the contest video."""
    import av

    container = av.open(str(mkv))
    frames = []
    want = {i * stride for i in range(n)}
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    container.close()
    return np.stack(frames, axis=0)  # (n, H, W, 3) uint8


def _segnet_logits(segnet, frame_uint8: np.ndarray, device) -> torch.Tensor:
    """REAL SegNet logits for a single (H,W,3) uint8 frame -> (1,5,384,512)."""
    x = (
        torch.from_numpy(frame_uint8.astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )  # (1,1,3,H,W)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(x)
        logits = segnet(seg_in)
    return logits


def _d_seg(segnet, gt_uint8: np.ndarray, recon_float: torch.Tensor, device) -> float:
    """REAL contest d_seg: argmax-flip rate between GT and recon SegNet outputs."""
    recon_uint8 = recon_float.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    lg_gt = _segnet_logits(segnet, gt_uint8, device)
    lg_re = _segnet_logits(segnet, recon_uint8, device)
    with torch.inference_mode():
        flip = (lg_gt.argmax(dim=1) != lg_re.argmax(dim=1)).float().mean()
    return float(flip.item())


def _full_grid_saliency(
    segnet, frame_hwc_float: torch.Tensor, device, H, W
) -> np.ndarray:
    """FULL-GRID SegNet response = |dL_seg/dpixel| (sum over RGB) at native H,W.

    The canonical #1587 saliency primitive (one backward pass; self-consistent
    cross-entropy of SegNet logits against their own argmax pseudo-labels).
    """
    x = frame_hwc_float.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
    x = x.detach().clone().requires_grad_(True)
    seg_in = segnet.preprocess_input(x)
    logits = segnet(seg_in)
    pseudo = logits.argmax(dim=1).detach()
    loss = F.cross_entropy(logits, pseudo)
    loss.backward()
    grad = x.grad.detach().abs()[0, 0]  # (3,H,W)
    sal = grad.sum(dim=0)  # (H,W)
    if sal.shape != (H, W):
        sal = F.interpolate(
            sal[None, None], size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
    return sal.cpu().numpy().astype(np.float32)


def _palette_degrade(gt_uint8: np.ndarray, n_levels: int) -> np.ndarray:
    """Grayscale-LUT-style palette quantization render (a real PR-class artifact).

    Quantizes each channel to ``n_levels`` (a deterministic posterize) -- the
    natural moderate-d_seg operating point a grayscale-LUT / palette renderer
    produces (NOT a synthetic spatial downsample). Fewer levels => higher d_seg.
    """
    g = gt_uint8.astype(np.float32)
    step = 255.0 / (n_levels - 1)
    q = np.round(g / step) * step
    return np.clip(q, 0, 255).round().astype(np.uint8)


def _mean_normalize(w: torch.Tensor) -> torch.Tensor:
    """Canonical 'mean' normalize: w / mean(w) (scale-preserving re-distribution).

    Identical to the MLX bundle ``recon_pixel_weight_normalize='mean'`` math.
    """
    return w / (w.mean() + 1e-12)


def _optimize_correction(
    render_f: torch.Tensor,  # (H,W,3) float32 [0,255]
    gt_f: torch.Tensor,  # (H,W,3) float32 [0,255]
    weight_hw: torch.Tensor,  # (H,W) float32 non-negative
    *,
    l_inf: float,
    iters: int,
    lr: float,
    device,
) -> torch.Tensor:
    """Optimize an L_inf-clamped per-pixel correction on the RECON-WEIGHTED loss.

    Minimizes ``mean(w * (render + delta - gt)^2) / mean(w)`` (the EXACT MLX
    bundle ``_weighted_recon`` 'mean'-normalize identity) over delta, projecting
    delta to ``[-l_inf, +l_inf]`` each step. The weight map is gradient-blocked
    (a fixed per-pixel emphasis) so the ONLY thing distinguishing the three arms
    is WHERE the correction is concentrated.

    Returns the corrected render ``(render + delta)`` clamped to [0,255].
    """
    render = render_f.to(device)
    gt = gt_f.to(device)
    w = weight_hw.to(device).unsqueeze(-1)  # (H,W,1)
    w = _mean_normalize(w)
    delta = torch.zeros_like(render, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        corrected = render + delta
        sq = (corrected - gt) ** 2  # (H,W,3)
        loss = (w * sq).mean() / (w.mean() + 1e-12)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-l_inf, l_inf)
    with torch.no_grad():
        return (render + delta).clamp(0, 255)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--palette-levels", type=int, nargs="+", default=[6, 10])
    ap.add_argument("--l-inf", type=float, default=16.0)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--out",
        default="experiments/results/recon_pixel_weight_real_render_ab_20260531/ab_output.json",
    )
    args = ap.parse_args()

    # CPU per CLAUDE.md "MPS auth eval is NOISE" (the d_seg measurement must not
    # drift); the optimization is small.
    device = torch.device(args.device)
    print(f"[ab] device={device} (NON-PROMOTABLE [macOS-CPU advisory])", flush=True)

    frames = _decode_frames(UPSTREAM / "videos" / "0.mkv", args.n_frames, args.stride)
    n, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"[ab] decoded {n} real frames {H}x{W} from 0.mkv", flush=True)
    segnet = load_default_scorers(str(UPSTREAM), device=device)[1]
    segnet.eval()
    print("[ab] loaded REAL SegNet (segnet.safetensors)", flush=True)

    per_palette: dict[str, dict] = {}
    for n_levels in args.palette_levels:
        print(f"\n[ab] ===== PALETTE n_levels={n_levels} =====", flush=True)
        degraded = np.stack(
            [_palette_degrade(frames[i], n_levels) for i in range(n)], axis=0
        )

        # Baseline (uncorrected degraded) d_seg.
        base_vals = []
        for p in range(n):
            deg_f = torch.from_numpy(degraded[p].astype(np.float32))
            base_vals.append(_d_seg(segnet, frames[p], deg_f, device))
        base_mean = float(np.mean(base_vals))
        print(f"[ab] baseline d_seg (uncorrected palette) = {base_mean:.6f}", flush=True)

        # Per-frame weight maps.
        deg_bchw = (
            torch.from_numpy(degraded.astype(np.float32))
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(device)
        )
        texture = compute_uniward_cost_map(deg_bchw).cpu().numpy()  # (n,H,W)
        saliency = np.zeros((n, H, W), dtype=np.float32)
        for p in range(n):
            saliency[p] = _full_grid_saliency(
                segnet,
                torch.from_numpy(degraded[p].astype(np.float32)),
                device,
                H,
                W,
            )
        sal_nz = float((saliency > 0).mean())
        print(
            f"[ab] full-grid saliency nonzero_frac={sal_nz:.4f} "
            f"mean={saliency.mean():.3e}",
            flush=True,
        )

        methods = {
            "uniform": np.ones((n, H, W), dtype=np.float32),
            "s_uniward_texture": texture,
            "full_grid_saliency": saliency,
        }

        results: dict[str, list[float]] = {m: [] for m in methods}
        for mname, wmap in methods.items():
            for p in range(n):
                render_f = torch.from_numpy(degraded[p].astype(np.float32))
                gt_f = torch.from_numpy(frames[p].astype(np.float32))
                w_hw = torch.from_numpy(np.maximum(wmap[p], 0.0).astype(np.float32))
                corrected = _optimize_correction(
                    render_f,
                    gt_f,
                    w_hw,
                    l_inf=args.l_inf,
                    iters=args.iters,
                    lr=args.lr,
                    device=device,
                )
                results[mname].append(
                    _d_seg(segnet, frames[p], corrected.cpu(), device)
                )
            mean_dseg = float(np.mean(results[mname]))
            print(
                f"[ab]   {mname:20s} corrected d_seg={mean_dseg:.6f} "
                f"(d vs base {base_mean - mean_dseg:+.6f})",
                flush=True,
            )

        d_uni = float(np.mean(results["uniform"]))
        d_tex = float(np.mean(results["s_uniward_texture"]))
        d_sal = float(np.mean(results["full_grid_saliency"]))
        saliency_beats_uniform = d_sal < d_uni
        saliency_beats_texture = d_sal < d_tex
        margin_vs_uniform = d_uni - d_sal  # >0 => saliency lower (better)
        margin_vs_texture = d_tex - d_sal
        verdict = (
            "LEVER_CONFIRMED"
            if (saliency_beats_uniform and saliency_beats_texture)
            else (
                "LEVER_PARTIAL_BEATS_UNIFORM_NOT_TEXTURE"
                if saliency_beats_uniform
                else "LEVER_NOT_CONFIRMED_ON_REAL_RENDER"
            )
        )
        print(
            f"[ab] VERDICT n_levels={n_levels}: {verdict} "
            f"(sal {d_sal:.6f} vs uni {d_uni:.6f} vs tex {d_tex:.6f}; "
            f"margin_vs_uni {margin_vs_uniform:+.6f})",
            flush=True,
        )
        per_palette[str(n_levels)] = {
            "n_levels": int(n_levels),
            "baseline_d_seg": base_mean,
            "d_seg_uniform": d_uni,
            "d_seg_s_uniward_texture": d_tex,
            "d_seg_full_grid_saliency": d_sal,
            "saliency_beats_uniform": bool(saliency_beats_uniform),
            "saliency_beats_texture": bool(saliency_beats_texture),
            "margin_saliency_vs_uniform": float(margin_vs_uniform),
            "margin_saliency_vs_texture": float(margin_vs_texture),
            "saliency_nonzero_fraction": sal_nz,
            "verdict": verdict,
            "per_frame_d_seg": {k: [float(x) for x in v] for k, v in results.items()},
        }

    any_confirmed = any(
        r["verdict"] == "LEVER_CONFIRMED" for r in per_palette.values()
    )
    any_beats_uniform = any(
        r["saliency_beats_uniform"] for r in per_palette.values()
    )
    payload = {
        "schema": "recon_pixel_weight_real_render_ab_v1",
        "axis_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "config": {
            "n_frames": n,
            "stride": args.stride,
            "palette_levels": list(args.palette_levels),
            "l_inf": args.l_inf,
            "iters": args.iters,
            "lr": args.lr,
            "device": str(device),
            "video_sha8": hashlib.sha256(
                (UPSTREAM / "videos" / "0.mkv").read_bytes()[: 1 << 20]
            ).hexdigest()[:8],
        },
        "per_palette": per_palette,
        "overall_any_lever_confirmed": bool(any_confirmed),
        "overall_any_beats_uniform": bool(any_beats_uniform),
        "channel_identity_note": (
            "the optimize_correction recon-weighted loss mean(w*sq)/mean(w) is "
            "the EXACT MLX bundle _weighted_recon 'mean'-normalize identity; this "
            "PyTorch harness confirms the SAME re-weight lever end-to-end against "
            "the real PyTorch SegNet."
        ),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[ab] wrote {out}", flush=True)
    print(
        f"[ab] OVERALL any_confirmed={any_confirmed} "
        f"any_beats_uniform={any_beats_uniform}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
