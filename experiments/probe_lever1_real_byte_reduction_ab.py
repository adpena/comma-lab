# SPDX-License-Identifier: MIT
"""Lever-1 REAL-BYTE-REDUCTION A/B — does the rate surrogate actually shrink bytes?

The deploy-faithful version of the audit's MED-1 (now on the FIXED ``codec_scan_order``):
the MED-1 probe (`probe_lever1_entropy_vs_real_brotli.py`) proved the conditional-entropy
surrogate RANK-TRACKS real brotli bytes (Spearman 0.90 / Pearson 0.999) over a STATIC set
of weight configs. That answers "does the surrogate VALUE track real bytes?" — it does NOT
answer "does TRAINING with the surrogate as a loss term actually DRIVE the real byte count
DOWN at non-worse distortion?" This probe closes that: it is the empirical bit-spend proof
(CLAUDE.md Catalog #304) for Lever-1 as a training lever.

It trains TWO short arms from the SAME basin-EMA seed, on the SAME tiny real-0.mkv slice,
SAME epoch budget + RNG:

  * OFF arm : ``rate_lambda_w = 0`` -> the vendored loss (no rate term).
  * ON  arm : ``rate_lambda_w > 0`` -> adds the order-1 conditional weight entropy
    (``codec_scan_order=True``, the deploy-faithful density) to the loss.

Both arms byte-close through the REAL vendored codec and are scored on the REAL frozen
SegNet/PoseNet over the SAME real pairs. We report the REAL brotli decoder-blob delta AND
the advisory distortion deltas.

VERDICT:
  * BYTE_REDUCTION   — the ON arm's REAL brotli blob is SMALLER at non-worse advisory
    distortion (the surrogate, used as a loss term, actually drives real bytes down).
  * NO_BYTE_REDUCTION — the ON arm is NOT smaller (or distortion is worse): on this budget
    the surrogate does not reduce real bytes; report it honestly (the lever may need a
    larger ``rate_lambda_w`` / more epochs / a late-stage schedule, OR its EV is bounded by
    the rate-invariant-floor reality).

Authority: $0 / local / torch-CPU (TRUSTED for advisory). Every number is
``[macOS-CPU advisory]`` NON-PROMOTABLE (small real slice, NOT a 600-pair contest eval).

Usage::

    .venv/bin/python experiments/probe_lever1_real_byte_reduction_ab.py --max-pairs 12 --epochs 8 --rate-lambda-w 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _REPO_ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512
_N = 37_545_489


def _contest_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose + 1e-12) + 25.0 * archive_bytes / _N


def _advisory_distortion(decoder, latents, ctx, n: int) -> tuple[float, float]:
    idx = torch.arange(n)
    with torch.no_grad():
        decoded_pair = decoder(latents[idx])
        flat = decoded_pair.reshape(n * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(n, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
        decoded_bhwc = decoded_bhwc.clamp(0, 255).round()
        net = ctx.distortion_net
        posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_out = net.segnet(segnet_in)
        pose6 = net.posenet(posenet_in)["pose"][:, :6]
        d_seg = float(
            (seg_out.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item()
        )
        d_pose = float(F.mse_loss(pose6, ctx.pose_targets[idx]).item())
    return d_seg, d_pose


def _train_arm(
    *, rate_lambda_w: float, rate_lambda_lat: float, max_pairs: int, epochs: int,
    seed: int, lr: float,
):
    """Train ONE arm from the basin-EMA seed; ``rate_lambda_w/lat`` toggle Lever-1.
    Returns (ema_sd, ema_latents, ctx, n). Mirrors the driver inner loop (the rate
    surrogate is added to the loss exactly like ``_weight_regularizers``)."""
    from copy import deepcopy

    from tac.losses.rate_surrogate import RateSurrogateConfig, brotli_rate_surrogate
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    model = import_vendored("model")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    basin_sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    basin_latents = ck["ema_latents"].detach().float()

    ctx = RealScorerContext(
        str(_VIDEO),
        device="cpu",
        max_pairs=max_pairs,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/lever4_probe_targets"),
    )
    n = ctx.n_pairs

    torch.manual_seed(seed)
    decoder = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    decoder.load_state_dict(basin_sd)
    latents = torch.nn.Parameter(basin_latents[:n].clone())

    ema_decoder = deepcopy(decoder)
    ema_latents = latents.data.clone()
    ema_decay = 0.999

    opt = torch.optim.AdamW(
        [
            {"params": decoder.parameters(), "lr": lr},
            {"params": [latents], "lr": lr * 10.0},
        ],
        weight_decay=0.0,
    )
    rate_cfg = RateSurrogateConfig(codec_scan_order=True)
    bs = 4
    for _ep in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            B = len(idx)

            decoded_pair = decoder(latents[idx])
            flat = decoded_pair.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
            decoded_bhwc = down.reshape(B, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
            dc = decoded_bhwc.clamp(0, 255)
            decoded_bhwc = dc + (dc.round() - dc).detach()

            net = ctx.distortion_net
            posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
            seg_out = net.segnet(segnet_in)
            pose6 = net.posenet(posenet_in)["pose"][:, :6]

            seg_l = F.cross_entropy(seg_out, ctx.seg_targets_hard[idx])
            pose_mse = F.mse_loss(pose6, ctx.pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            loss = 100.0 * seg_l + pose_l

            # --- Lever-1 rate surrogate (the only difference between arms) ---
            if rate_lambda_w > 0 or rate_lambda_lat > 0:
                lat_arg = latents if rate_lambda_lat > 0 else None
                h_cond, r_lat = brotli_rate_surrogate(decoder, lat_arg, rate_cfg, device="cpu")
                if rate_lambda_w > 0:
                    loss = loss + rate_lambda_w * h_cond
                if rate_lambda_lat > 0:
                    loss = loss + rate_lambda_lat * r_lat

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*decoder.parameters(), latents], 1e9)
            opt.step()

            with torch.no_grad():
                for (_k, ev), (_k2, lv) in zip(
                    ema_decoder.state_dict().items(), decoder.state_dict().items(), strict=True
                ):
                    ev.mul_(ema_decay).add_(lv, alpha=1.0 - ema_decay)
                ema_latents.mul_(ema_decay).add_(latents.data, alpha=1.0 - ema_decay)

    ema_sd = {k: v.detach().clone() for k, v in ema_decoder.state_dict().items()}
    return ema_sd, ema_latents.detach().clone(), ctx, n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--rate-lambda-w", type=float, default=0.5)
    ap.add_argument("--rate-lambda-lat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not _BASIN_CKPT.exists():
        raise FileNotFoundError(f"basin checkpoint not found: {_BASIN_CKPT}")

    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    model = import_vendored("model")

    # OFF arm
    off_sd, off_lat, ctx, n = _train_arm(
        rate_lambda_w=0.0, rate_lambda_lat=0.0, max_pairs=args.max_pairs,
        epochs=args.epochs, seed=args.seed, lr=args.lr,
    )
    # ON arm
    on_sd, on_lat, _ctx2, _n2 = _train_arm(
        rate_lambda_w=args.rate_lambda_w, rate_lambda_lat=args.rate_lambda_lat,
        max_pairs=args.max_pairs, epochs=args.epochs, seed=args.seed, lr=args.lr,
    )

    meta = {"base_channels": 20, "latent_dim": 28}

    def _blob(sd):
        return len(codec.encode_decoder(codec.quantize_state_dict(sd)))

    def _archive(sd, lat):
        return len(codec.build_archive(sd, lat, meta))

    off_blob, on_blob = _blob(off_sd), _blob(on_sd)
    off_arch, on_arch = _archive(off_sd, off_lat), _archive(on_sd, on_lat)

    dec_off = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_off.load_state_dict(off_sd)
    off_dseg, off_dpose = _advisory_distortion(dec_off, off_lat, ctx, n)
    dec_on = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec_on.load_state_dict(on_sd)
    on_dseg, on_dpose = _advisory_distortion(dec_on, on_lat, ctx, n)

    off_score = _contest_score(off_dseg, off_dpose, off_arch)
    on_score = _contest_score(on_dseg, on_dpose, on_arch)
    blob_delta = on_blob - off_blob
    arch_delta = on_arch - off_arch
    score_delta = on_score - off_score

    SEG_TOL = 0.02
    seg_ok = (on_dseg - off_dseg) <= SEG_TOL
    if blob_delta < 0 and seg_ok:
        verdict = "BYTE_REDUCTION"
        note = (
            f"Lever-1 ON drives the REAL brotli blob DOWN {-blob_delta} B ({on_blob} vs "
            f"{off_blob}) at advisory d_seg within {SEG_TOL} (off {off_dseg:.4f} -> on "
            f"{on_dseg:.4f}). Catalog #304 bit-spend proof: POSITIVE on the deploy-faithful "
            "codec axis (TRAINING with the surrogate reduces real bytes, not just rank-tracks "
            "them). The 600-pair dual exact eval is still required before a SCORE claim."
        )
    else:
        reason = []
        if blob_delta >= 0:
            reason.append(f"blob NOT smaller (delta {blob_delta:+d} B)")
        if not seg_ok:
            reason.append(f"advisory d_seg WORSE by {on_dseg - off_dseg:+.4f} (> {SEG_TOL})")
        verdict = "NO_BYTE_REDUCTION"
        note = (
            f"Lever-1 ON did NOT reduce real bytes at non-worse distortion on this budget: "
            f"{'; '.join(reason)}. HONEST: the surrogate rank-TRACKS bytes (MED-1) but did not "
            f"DRIVE them down in {args.epochs} epochs on {n} pairs at rate_lambda_w="
            f"{args.rate_lambda_w}; needs a larger lambda / a late-stage schedule / more "
            "epochs, OR its EV is bounded by the rate-invariant-floor reality. Do NOT claim a "
            "Lever-1 byte win on this basis."
        )

    result = {
        "probe": "lever1_real_byte_reduction_ab",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — small real slice, NOT a 600-pair contest eval",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "max_pairs": n,
        "epochs": args.epochs,
        "rate_lambda_w": args.rate_lambda_w,
        "rate_lambda_lat": args.rate_lambda_lat,
        "lr": args.lr,
        "seed": args.seed,
        "off_decoder_blob_bytes": off_blob,
        "on_decoder_blob_bytes": on_blob,
        "decoder_blob_delta_bytes": blob_delta,
        "off_full_archive_bytes": off_arch,
        "on_full_archive_bytes": on_arch,
        "full_archive_delta_bytes": arch_delta,
        "advisory_d_seg_off": round(off_dseg, 6),
        "advisory_d_seg_on": round(on_dseg, 6),
        "advisory_d_pose_off": round(off_dpose, 8),
        "advisory_d_pose_on": round(on_dpose, 8),
        "advisory_score_off": round(off_score, 6),
        "advisory_score_on": round(on_score, 6),
        "advisory_score_delta": round(score_delta, 6),
        "verdict": verdict,
        "verdict_note": note,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 80)
        print("LEVER-1 REAL-BYTE-REDUCTION A/B — rate surrogate ON vs OFF (TRAINING)")
        print("=" * 80)
        print(f"basin: {result['basin_checkpoint']}")
        print(f"{n} real pairs, {args.epochs} epochs, lr={args.lr}, rate_lambda_w={args.rate_lambda_w}")
        print()
        print(f"  OFF blob {off_blob:>7d} B   archive {off_arch:>7d} B")
        print(f"  ON  blob {on_blob:>7d} B   archive {on_arch:>7d} B")
        print(f"  blob delta {blob_delta:>+7d} B   archive delta {arch_delta:>+7d} B   (negative = ON SMALLER)")
        print()
        print(f"  advisory d_seg  off {off_dseg:.5f} -> on {on_dseg:.5f}  (Δ {on_dseg-off_dseg:+.5f})")
        print(f"  advisory d_pose off {off_dpose:.6f} -> on {on_dpose:.6f}  (Δ {on_dpose-off_dpose:+.6f})")
        print(f"  advisory score  off {off_score:.6f} -> on {on_score:.6f}  (Δ {score_delta:+.6f})")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
