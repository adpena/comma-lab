# SPDX-License-Identifier: MIT
"""R3 review — gradient-direction correctness probe for the 5 Layer-2 levers.

The highest-value R3 check (lens A): does each lever's gradient actually point
toward LOWER S — i.e. does a gradient step on the lever's term actually REDUCE its
real target quantity (codec bytes / d_seg argmax-flip rate / d_pose)? A lever whose
gradient points the WRONG way is the worst silent bug (active mid-run, looks active,
actively harms the descent). All numbers are ``[macOS-CPU advisory]`` NON-PROMOTABLE.

Deploy-faithful: every byte number is the REAL vendored codec
(``encode_decoder(quantize_state_dict(sd))`` / ``build_archive``); every d_seg number is
the REAL argmax-flip rate on real 0.mkv pairs through the FROZEN contest SegNet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from tac.losses.rate_surrogate import (  # noqa: E402
    RateSurrogateConfig,
    conditional_weight_entropy,
    latent_delta_entropy,
)
from tac.torch_vehicle.vendored_imports import import_vendored  # noqa: E402

_BASIN = (
    _ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512


def _load_basin():
    codec = import_vendored("codec")
    model = import_vendored("model")
    ck = torch.load(_BASIN, map_location="cpu", weights_only=False)
    sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    lat = ck["ema_latents"].detach().float()
    return codec, model, sd, lat


def _build_decoder(model, sd):
    d = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    d.load_state_dict(sd)
    return d


def _decoder_bytes(codec, decoder) -> int:
    qsd = codec.quantize_state_dict(decoder.state_dict())
    return len(codec.encode_decoder(qsd))


def _full_bytes(codec, decoder, latents) -> int:
    meta = {"base_channels": 20, "latent_dim": int(latents.shape[1])}
    return len(codec.build_archive(decoder.state_dict(), latents.detach(), meta))


# --------------------------------------------------------------------------- #
# Lever 1a — weight rate term gradient must REDUCE real decoder bytes.
# --------------------------------------------------------------------------- #
def lever1a_weight_rate(codec, model, sd, *, steps: int, lr: float) -> dict:
    dec = _build_decoder(model, sd)
    cfg = RateSurrogateConfig(codec_scan_order=True)
    b0 = _decoder_bytes(codec, dec)
    opt = torch.optim.SGD(dec.parameters(), lr=lr)
    h0 = None
    for i in range(steps):
        opt.zero_grad()
        h = conditional_weight_entropy(dec, cfg)
        if i == 0:
            h0 = float(h.item())
        h.backward()
        opt.step()
    h1 = float(conditional_weight_entropy(dec, cfg).item())
    b1 = _decoder_bytes(codec, dec)
    return {
        "lever": "1a_weight_rate",
        "surrogate_before": round(h0, 5),
        "surrogate_after": round(h1, 5),
        "real_decoder_bytes_before": b0,
        "real_decoder_bytes_after": b1,
        "byte_delta": b1 - b0,
        "byte_pct": round(100.0 * (b1 - b0) / b0, 3),
        "surrogate_descended": h1 < h0,
        "bytes_descended": b1 < b0,
        "gradient_direction": "DOWN (correct)" if b1 < b0 else "UP (WRONG-WAY BUG)",
    }


# --------------------------------------------------------------------------- #
# Lever 1b — latent delta rate gradient must REDUCE real full-archive bytes.
# --------------------------------------------------------------------------- #
def lever1b_latent_rate(codec, model, sd, lat, *, steps: int, lr: float) -> dict:
    dec = _build_decoder(model, sd)
    z = lat.clone().requires_grad_(True)
    cfg = RateSurrogateConfig(codec_scan_order=True)
    f0 = _full_bytes(codec, dec, z)
    opt = torch.optim.SGD([z], lr=lr)
    r0 = None
    for i in range(steps):
        opt.zero_grad()
        r = latent_delta_entropy(z, cfg)
        if i == 0:
            r0 = float(r.item())
        r.backward()
        opt.step()
    r1 = float(latent_delta_entropy(z, cfg).item())
    f1 = _full_bytes(codec, dec, z)
    return {
        "lever": "1b_latent_rate",
        "surrogate_before": round(r0, 5),
        "surrogate_after": round(r1, 5),
        "real_full_bytes_before": f0,
        "real_full_bytes_after": f1,
        "byte_delta": f1 - f0,
        "byte_pct": round(100.0 * (f1 - f0) / f0, 3),
        "surrogate_descended": r1 < r0,
        "bytes_descended": f1 < f0,
        "gradient_direction": "DOWN (correct)" if f1 < f0 else "UP (WRONG-WAY BUG)",
    }


# --------------------------------------------------------------------------- #
# Lever 2 — seg surrogate gradient must REDUCE the real argmax-flip rate (d_seg).
# Lever 5 — margin-weighting must not REVERSE that descent.
# Run at BOTH a static high T and the annealed cold T (lens A: does the surrogate
# diverge from the hard argmax objective at high/annealed T?).
# --------------------------------------------------------------------------- #
def lever2_seg_descent(model, sd, lat, *, max_pairs: int, steps: int, lr: float) -> list:
    from tac.losses.core import segnet_surrogate_per_pixel
    from tac.torch_vehicle.scorer_context import RealScorerContext

    _SEG_C = 5
    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=max_pairs,
        targets_cache=str(_ROOT / ".omx/tmp/r3_seg_probe_targets"),
    )
    n = ctx.n_pairs
    idx = torch.arange(n)
    net = ctx.distortion_net

    def real_d_seg(decoder) -> float:
        with torch.no_grad():
            dp = decoder(lat[idx])
            flat = dp.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2).clamp(0, 255).round()
            _, seg_in = net.preprocess_input(bhwc)
            so = net.segnet(seg_in)
            return float((so.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item())

    def gt_logits():
        gt = F.one_hot(ctx.seg_targets_hard[idx].long(), num_classes=_SEG_C)
        return gt.permute(0, 3, 1, 2).float() * 30.0

    out = []
    # T variants: static-hot 1.0, annealed-cold 0.05; with/without margin weighting.
    variants = [
        ("static_T1.0_nomargin", 1.0, None),
        ("annealed_T0.05_nomargin", 0.05, None),
        ("static_T1.0_margin2.0", 1.0, 2.0),
    ]
    for label, temp, margin_tau in variants:
        dec = _build_decoder(model, sd)
        glog = gt_logits()
        d0 = real_d_seg(dec)
        opt = torch.optim.SGD(dec.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            dp = dec(lat[idx])
            flat = dp.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
            bhwc = bhwc.clamp(0, 255)
            bhwc = bhwc + (bhwc.round() - bhwc).detach()
            _, seg_in = net.preprocess_input(bhwc)
            so = net.segnet(seg_in)
            pp = segnet_surrogate_per_pixel(
                so, glog, surrogate="soft_cosine", temperature=temp,
                gt_already_probs=False, num_classes=_SEG_C,
            )
            if margin_tau is not None:
                top2, _ = torch.topk(so.detach(), k=2, dim=1, largest=True, sorted=True)
                marg = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
                w = torch.exp(-marg / max(margin_tau, 1e-6))
                seg_l = (pp * w).mean()
            else:
                seg_l = pp.mean()
            seg_l.backward()
            opt.step()
        d1 = real_d_seg(dec)
        out.append({
            "lever": f"2_seg_{label}",
            "real_d_seg_before": round(d0, 6),
            "real_d_seg_after": round(d1, 6),
            "d_seg_delta": round(d1 - d0, 6),
            "d_seg_descended": d1 < d0,
            "gradient_direction": "DOWN (correct)" if d1 < d0 else ("FLAT" if d1 == d0 else "UP (WRONG-WAY BUG)"),
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr-w", type=float, default=5e-3)
    ap.add_argument("--lr-lat", type=float, default=1e-2)
    ap.add_argument("--seg-steps", type=int, default=30)
    ap.add_argument("--seg-lr", type=float, default=2e-3)
    ap.add_argument("--seg-pairs", type=int, default=6)
    ap.add_argument("--levers", default="all", help="1,2 or all")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    codec, model, sd, lat = _load_basin()
    results = []
    if args.levers in ("all", "1"):
        results.append(lever1a_weight_rate(codec, model, sd, steps=args.steps, lr=args.lr_w))
        results.append(lever1b_latent_rate(codec, model, sd, lat, steps=args.steps, lr=args.lr_lat))
    if args.levers in ("all", "2"):
        results.extend(
            lever2_seg_descent(model, sd, lat, max_pairs=args.seg_pairs, steps=args.seg_steps, lr=args.seg_lr)
        )
    out = {"authority": "[macOS-CPU advisory] NON-PROMOTABLE", "results": results}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for r in out["results"]:
            print(f"=== {r['lever']} ===")
            for k, v in r.items():
                print(f"  {k}: {v}")
    print("PROBE_COMPLETE")
