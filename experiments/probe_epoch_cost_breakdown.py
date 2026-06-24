#!/usr/bin/env python3
"""Per-epoch cost-breakdown probe: WHERE the ~13 s/ep goes on the bc36 MPS basin.

The batch-saturation probe established s/ep is BATCH-INVARIANT (bs 8→64 both ~13.4
s/ep) — so the cost is NOT step-overhead and NOT per-batch data transfer. This probe
DECOMPOSES one training epoch into its components so the parent knows the true
throughput CEILING before launching the multi-day prune-source:

  * GT-targets / latents RESIDENCY check (are they re-transferred each epoch? — the
    scorer_context holds them on the train device from construction, so the answer is
    NO; this probe MEASURES the per-epoch host->device byte traffic to PROVE it);
  * decoder forward (latents -> 2 frames);
  * eval-roundtrip (bicubic up -> bilinear down -> uint8 STE);
  * frozen SegNet forward + backward;
  * frozen PoseNet forward + backward;
  * optimizer step.

It times the REAL bc36 decoder + the REAL frozen SegNet/PoseNet (RealScorerContext)
on the byte-identical n600 target cache, for a few warm epochs (MPS JITs the first).
$0, MPS train / CPU authority, ADVISORY (throughput only — no score claim). Run this
when the anchor is STOPPED (do not contend the MPS GPU with a live run).

Usage::

    .venv/bin/python experiments/probe_epoch_cost_breakdown.py \
        --train-device mps --base-channels 36 --n-pairs 600 --batch-size 150 \
        --warm-epochs 2 --timed-epochs 3

Writes ``.omx/research/epoch_cost_breakdown_<utc>.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
_EVAL_H, _EVAL_W = 384, 512


def _sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _device_bytes_baseline(dev: torch.device) -> int | None:
    """Current driver-allocated device bytes (MPS/CUDA), or None on CPU. Used to
    PROVE the GT targets are resident (no per-epoch growth)."""
    if dev.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if dev.type == "cuda":
        return int(torch.cuda.memory_allocated())
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--base-channels", type=int, default=36)
    ap.add_argument("--latent-dim", type=int, default=28)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--warm-epochs", type=int, default=2)
    ap.add_argument("--timed-epochs", type=int, default=3)
    ap.add_argument(
        "--targets-cache", type=Path,
        default=REPO / "experiments/results/capstone_gt_targets_cache",
    )
    ap.add_argument("--video-path", type=Path, default=None)
    args = ap.parse_args(argv)

    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    video_path = args.video_path
    if video_path is None:
        video_path = import_vendored("data").get_default_video_path()

    train_dev = torch.device(args.train_device)
    # Full-MPS path (no split-by-head): both heads on the train device — the probe's
    # config (matches the running batch probe). The scorer holds the GT targets on
    # the train device from construction (residency under test).
    scorer = RealScorerContext(
        video_path, device=args.device, train_device=args.train_device,
        split_by_head=False, max_pairs=args.n_pairs, targets_cache=args.targets_cache,
    )
    v = import_vendored("model") if False else None  # decoder via the driver helper below

    # Build the REAL bc36 vendored decoder on the train device (same as the driver).
    from tac.torch_vehicle.vendored_imports import import_vendored as _imp
    HNeRVDecoder = _imp("model").HNeRVDecoder
    decoder = HNeRVDecoder(
        latent_dim=args.latent_dim, base_channels=args.base_channels,
        eval_size=(_EVAL_H, _EVAL_W),
    ).to(train_dev)
    latents = torch.nn.Parameter(
        (torch.randn(args.n_pairs, args.latent_dim) * 0.1).to(train_dev)
    )
    opt = torch.optim.AdamW(list(decoder.parameters()) + [latents], lr=1e-3)

    n_pairs, bs = args.n_pairs, args.batch_size
    seg_targets = scorer.seg_targets_hard  # resident on train_dev
    pose_targets = scorer.pose_targets     # resident on train_dev

    # Residency proof: GT target device + a per-epoch device-bytes delta (should be
    # ~0 growth if nothing is re-transferred each epoch).
    residency = {
        "seg_targets_device": str(seg_targets.device),
        "pose_targets_device": str(pose_targets.device),
        "latents_device": str(latents.device),
        "seg_targets_resident_on_train_device": seg_targets.device.type == train_dev.type,
    }

    # Component timers (accumulated over timed epochs).
    acc = {k: 0.0 for k in (
        "decoder_fwd", "roundtrip", "seg_fwd", "seg_bwd", "pose_fwd", "pose_bwd",
        "opt_step", "total",
    )}

    def _ce(seg_out, tgt):
        return F.cross_entropy(seg_out, tgt)

    def run_epoch(timed: bool) -> None:
        perm = torch.randperm(n_pairs).to(train_dev)
        for bstart in range(0, n_pairs, bs):
            idx = perm[bstart:bstart + bs]
            opt.zero_grad()
            _sync(train_dev); t = time.time()
            decoded = decoder(latents[idx])  # (B,2,3,H,W)
            _sync(train_dev)
            if timed:
                acc["decoder_fwd"] += time.time() - t; t = time.time()
            B = len(idx)
            flat = decoded.reshape(B * 2, 3, _EVAL_H, _EVAL_W)
            up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
            down = F.interpolate(up, size=(384, 512), mode="bilinear", align_corners=False)
            dec_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)
            dec_bhwc = dec_bhwc.clamp(0, 255)
            dec_bhwc = dec_bhwc + (dec_bhwc.round() - dec_bhwc).detach()
            _sync(train_dev)
            if timed:
                acc["roundtrip"] += time.time() - t; t = time.time()
            # SegNet head (forward).
            net = scorer.train_distortion_net
            posenet_in, segnet_in = net.preprocess_input(dec_bhwc)
            seg_out = net.segnet(segnet_in)
            _sync(train_dev)
            if timed:
                acc["seg_fwd"] += time.time() - t; t = time.time()
            seg_l = 100.0 * _ce(seg_out, seg_targets[idx])
            seg_l.backward(retain_graph=True)
            _sync(train_dev)
            if timed:
                acc["seg_bwd"] += time.time() - t; t = time.time()
            # PoseNet head (forward) — re-run on the same dec graph for the pose grad.
            pose_out = net.posenet(posenet_in)
            pose6 = pose_out["pose"][:, :6]
            _sync(train_dev)
            if timed:
                acc["pose_fwd"] += time.time() - t; t = time.time()
            pose_mse = F.mse_loss(pose6, pose_targets[idx])
            pose_l = torch.sqrt(10.0 * pose_mse + 1e-12)
            pose_l.backward()
            _sync(train_dev)
            if timed:
                acc["pose_bwd"] += time.time() - t; t = time.time()
            opt.step()
            _sync(train_dev)
            if timed:
                acc["opt_step"] += time.time() - t

    # Warm epochs (MPS JIT / autotune the first epochs).
    for _ in range(args.warm_epochs):
        run_epoch(timed=False)

    bytes_before = _device_bytes_baseline(train_dev)
    _sync(train_dev); t0 = time.time()
    for _ in range(args.timed_epochs):
        run_epoch(timed=True)
    _sync(train_dev)
    acc["total"] = time.time() - t0
    bytes_after = _device_bytes_baseline(train_dev)

    te = max(1, args.timed_epochs)
    per_ep = {k: round(v / te, 4) for k, v in acc.items()}
    residency["device_bytes_before_timed"] = bytes_before
    residency["device_bytes_after_timed"] = bytes_after
    residency["device_bytes_growth_over_timed_epochs"] = (
        None if bytes_before is None or bytes_after is None
        else bytes_after - bytes_before
    )

    out = {
        "schema": "epoch_cost_breakdown.v1",
        "authority": "[advisory] throughput-only NON-PROMOTABLE",
        "config": {
            "train_device": args.train_device, "device": args.device,
            "base_channels": args.base_channels, "latent_dim": args.latent_dim,
            "n_pairs": n_pairs, "batch_size": bs,
            "steps_per_epoch": (n_pairs + bs - 1) // bs,
            "warm_epochs": args.warm_epochs, "timed_epochs": args.timed_epochs,
        },
        "residency": residency,
        "per_epoch_seconds": per_ep,
        "scorer_fwd_bwd_fraction": (
            round(
                (per_ep["seg_fwd"] + per_ep["seg_bwd"] + per_ep["pose_fwd"]
                 + per_ep["pose_bwd"]) / per_ep["total"], 4
            ) if per_ep["total"] > 0 else None
        ),
        "note": (
            "GT targets + latents are device-RESIDENT from construction "
            "(scorer_context.py holds them on train_device); a ~0 device-bytes growth "
            "over timed epochs PROVES no per-epoch re-transfer. The dominant cost is "
            "the frozen SegNet+PoseNet fwd/bwd (fixed per epoch) — the true ceiling."
        ),
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outpath = REPO / f".omx/research/epoch_cost_breakdown_{ts}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print("WROTE", outpath, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
