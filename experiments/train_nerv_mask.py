#!/usr/bin/env python3
"""Lane 12 — NeRV mask codec trainer.

Trains a tiny coordinate-MLP (``tac.nerv_mask_codec.NeRVMaskCodec``) to overfit
the SegNet argmax mask sequence for a single 1200-frame video at 384×512. The
trained codec is encoded as an NRV2 self-describing payload and written
alongside provenance + metrics for the dispatch script to bundle into the
archive.

CLAUDE.md non-negotiables enforced:
    - CUDA-required default (``--device cuda``); MPS rejected at trainer
      construction. Explicit ``--device cpu`` opt-in for the unit-test path.
    - EMA decay 0.997 (canonical ``tac.training.EMA``); shadow is what gets
      shipped via ``trainer.encode()`` — NOT the live weights.
    - eval_roundtrip-aware: mask-CODEC layer uses cross-entropy on raw logits
      (no ``.round()`` zero-gradient bug). The eval-roundtrip 384→874→uint8→384
      simulation lives in the auth-eval stage that consumes the produced
      archive (delegated to ``experiments/contest_auth_eval.py``).
    - Auth eval at end: dispatch script (``scripts/remote_lane_nerv.sh``)
      runs CUDA auth eval after this trainer writes its outputs.
    - Provenance JSON written: git hash, GPU info, profile dict, byte counts.
    - No silent defaults: ``--profile`` required; CLI overrides are explicit.

Usage (dispatch via ``scripts/remote_lane_nerv.sh``):

    python experiments/train_nerv_mask.py \\
        --profile nerv_mask_lane_g_v3 \\
        --device cuda \\
        --gt-masks-source segnet \\
        --upstream upstream \\
        --output-dir results/lane_12_nerv

Outputs (under ``--output-dir``):
    masks.nrv          — NRV2 self-describing payload (~12-23 KB)
    provenance.json    — git hash, GPU, profile, training metrics
    train_metrics.csv  — per-eval-step disagreement rate
"""

# CLAUDE.md non-negotiable: CUDA-required default. NEVER fall back to MPS.
# An explicit `--device cpu` exists ONLY to keep the unit-test path
# deterministic-bytes acceptable; CPU is NOT used for production dispatch.
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src/ to path BEFORE any tac imports so this works in detached/Vast.ai
# bootstraps where the package is editable-installed under src/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.nerv_mask_codec import (  # noqa: E402
    NeRVMaskCodec,
    NeRVMaskTrainer,
    encode_nerv_codec,
    nerv_codec_bytes,
)
from tac.profiles import PROFILES  # noqa: E402


def _load_segnet_argmax_masks(
    upstream_dir: Path,
    device: str,
    num_frames: int = 1200,
) -> torch.Tensor:
    """Run SegNet over upstream/videos/0.mkv and return (T, H, W) argmax masks.

    This is the contest-compliant compress-time path: SegNet is loaded ONCE
    here at compress time to extract ground-truth argmax labels for the
    NeRV trainer to overfit. Per CLAUDE.md "Strict scorer rule" the loaded
    SegNet is NOT shipped in archive.zip — it stays on the compress-time
    machine.

    Args:
        upstream_dir: directory containing videos/0.mkv + scorers.
        device: "cuda" required for production. CPU possible for tests.
        num_frames: total frames to extract (default 1200).

    Returns:
        (T, H, W) long tensor on CPU with class IDs in [0, NUM_CLASSES).
    """
    # Lazy import — SegNet load is heavy.
    from tac.scorer import load_differentiable_scorers

    print(f"[lane-12] loading SegNet from {upstream_dir} on {device} ...", flush=True)
    posenet, segnet = load_differentiable_scorers(
        upstream_dir=str(upstream_dir),
        device=device,
    )
    segnet.eval()
    # Decode video frames at scorer resolution. Use ffmpeg-cpu pipe for
    # robustness; NVDEC is for batch training, not for one-shot extraction.
    import subprocess

    video = upstream_dir / "videos" / "0.mkv"
    if not video.exists():
        raise FileNotFoundError(f"upstream video not found: {video}")
    # Probe original size
    probe = subprocess.run(  # subprocess-no-check-OK: check=True is set on the same call (multi-line — scanner's regex doesn't span lines)
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    orig_w, orig_h = (int(x) for x in probe.stdout.strip().split(","))
    # Decode raw rgb24 then resize to scorer resolution per upstream evaluate.
    cmd = [
        "ffmpeg",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "error",
        "pipe:1",
    ]
    decode = subprocess.run(cmd, capture_output=True, timeout=600, check=True)
    raw = np.frombuffer(decode.stdout, dtype=np.uint8)
    frame_size = orig_h * orig_w * 3
    n_decoded = len(raw) // frame_size
    if n_decoded < num_frames:
        raise RuntimeError(
            f"video has {n_decoded} frames; expected at least {num_frames}"
        )
    frames = raw[: num_frames * frame_size].reshape(num_frames, orig_h, orig_w, 3)
    # Resize to scorer (384, 512) and run SegNet in batches.
    SCORER_H, SCORER_W = 384, 512
    masks_THW = torch.empty(num_frames, SCORER_H, SCORER_W, dtype=torch.long)
    BATCH = 16
    import torch.nn.functional as F  # noqa: N812

    print(
        f"[lane-12] extracting argmax masks for {num_frames} frames ...",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, num_frames, BATCH):
            end = min(start + BATCH, num_frames)
            chunk = (
                torch.from_numpy(frames[start:end].copy())
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
            )  # (B, 3, H, W)
            chunk = F.interpolate(
                chunk, size=(SCORER_H, SCORER_W), mode="bilinear", align_corners=False
            )
            # SegNet expects (B, T=1, C, H, W) per upstream spec
            seg_logits = segnet(chunk.unsqueeze(1))  # (B, 1, 5, H, W)
            seg_argmax = seg_logits.squeeze(1).argmax(dim=1).cpu().long()
            masks_THW[start:end] = seg_argmax
    return masks_THW


def main() -> int:
    p = argparse.ArgumentParser(description="Lane 12 NeRV mask codec trainer")
    p.add_argument("--profile", required=True, help="Profile name from tac.profiles.PROFILES")
    p.add_argument(
        "--device",
        required=True,
        choices=["cuda", "cpu"],
        help="cuda for production; cpu allowed for unit tests only",
    )
    p.add_argument(
        "--upstream",
        type=Path,
        default=_REPO_ROOT / "upstream",
        help="upstream directory containing videos/0.mkv + scorers",
    )
    p.add_argument(
        "--gt-masks-source",
        choices=["segnet", "amrc", "synthetic"],
        default="segnet",
        help=(
            "segnet (production: extract from upstream/videos/0.mkv via SegNet),"
            " amrc (load from existing AMRC payload),"
            " synthetic (unit-test stripes pattern)"
        ),
    )
    p.add_argument(
        "--amrc-path",
        type=Path,
        default=None,
        help="Path to existing masks.amrc (when --gt-masks-source=amrc)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where masks.nrv + provenance.json + train_metrics.csv land",
    )
    p.add_argument(
        "--num-frames", type=int, default=1200, help="Total frame count (default 1200)"
    )
    p.add_argument(
        "--mask-height", type=int, default=384, help="Scorer-resolution H (default 384)"
    )
    p.add_argument(
        "--mask-width", type=int, default=512, help="Scorer-resolution W (default 512)"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="SGD steps; if None, uses profile['nerv_steps']",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate disagreement every N steps; default = profile['nerv_eval_every']",
    )
    p.add_argument(
        "--weight-dtype",
        choices=["fp16", "int8"],
        default=None,
        help="Quantization for shipping; default = profile['nerv_weight_dtype']",
    )
    args = p.parse_args()

    if args.profile not in PROFILES:
        raise SystemExit(
            f"profile {args.profile!r} not in PROFILES; available: "
            f"{sorted(PROFILES.keys())}"
        )
    profile = PROFILES[args.profile]
    # Required NeRV knobs:
    nerv_keys = (
        "nerv_num_freqs",
        "nerv_hidden_dim",
        "nerv_depth",
        "nerv_num_classes",
        "nerv_learning_rate",
        "nerv_ema_decay",
        "nerv_batch_coords",
        "nerv_steps",
        "nerv_eval_every",
        "nerv_weight_dtype",
    )
    missing = [k for k in nerv_keys if k not in profile]
    if missing:
        raise SystemExit(
            f"profile {args.profile!r} missing NeRV keys: {missing}. "
            f"Use a profile registered for Lane 12 (e.g. nerv_mask_lane_g_v3)."
        )

    steps = int(args.steps if args.steps is not None else profile["nerv_steps"])
    eval_every = int(
        args.eval_every if args.eval_every is not None else profile["nerv_eval_every"]
    )
    weight_dtype = (
        args.weight_dtype if args.weight_dtype is not None else profile["nerv_weight_dtype"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determinism (CLAUDE.md canonical pipeline standard)
    seed = int(profile.get("seed", 12))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "FATAL: --device cuda requested but torch.cuda.is_available() is False. "
                "CLAUDE.md FORBIDDEN PATTERN: NEVER fall back to MPS/CPU silently."
            )
        torch.cuda.manual_seed_all(seed)

    # ── Stage 1: load ground-truth argmax masks ──
    if args.gt_masks_source == "segnet":
        masks_THW = _load_segnet_argmax_masks(
            args.upstream, device=args.device, num_frames=args.num_frames
        )
    elif args.gt_masks_source == "amrc":
        if args.amrc_path is None or not args.amrc_path.exists():
            raise SystemExit(
                "--gt-masks-source=amrc requires --amrc-path pointing to an "
                "existing masks.amrc file"
            )
        from tac.lossless.argmax_codec import decode_argmax_masks  # type: ignore

        masks_THW = decode_argmax_masks(args.amrc_path.read_bytes()).long()
        if masks_THW.shape[0] != args.num_frames:
            print(
                f"[lane-12] amrc has {masks_THW.shape[0]} frames; expected "
                f"{args.num_frames}. Continuing with the smaller count.",
                flush=True,
            )
    else:  # synthetic
        # 4×8×8 stripes for unit-test path (matches tests in test_nerv_mask_codec.py)
        T = min(args.num_frames, 4)
        H = min(args.mask_height, 8)
        W = min(args.mask_width, 8)
        masks_THW = torch.zeros(T, H, W, dtype=torch.long)
        for t in range(T):
            cols = (torch.arange(W) + t) % int(profile["nerv_num_classes"])
            masks_THW[t] = cols.unsqueeze(0).expand(H, W)

    print(
        f"[lane-12] masks shape: {tuple(masks_THW.shape)}, dtype={masks_THW.dtype}, "
        f"unique={sorted(masks_THW.unique().tolist())}",
        flush=True,
    )

    # ── Stage 2: build codec + trainer ──
    codec = NeRVMaskCodec(
        num_freqs=int(profile["nerv_num_freqs"]),
        hidden_dim=int(profile["nerv_hidden_dim"]),
        num_classes=int(profile["nerv_num_classes"]),
        depth=int(profile["nerv_depth"]),
        seed=seed,
    )
    trainer = NeRVMaskTrainer(
        codec=codec,
        device=args.device,
        learning_rate=float(profile["nerv_learning_rate"]),
        ema_decay=float(profile["nerv_ema_decay"]),
        seed=seed,
    )
    fp16_bytes = nerv_codec_bytes(codec, weight_dtype="fp16")
    int8_bytes = nerv_codec_bytes(codec, weight_dtype="int8")
    print(
        f"[lane-12] codec: params={codec.num_params()}, "
        f"fp16={fp16_bytes}B, int8={int8_bytes}B (excluding header+scale)",
        flush=True,
    )

    # ── Stage 3: training loop with periodic eval ──
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_writer = csv.writer(metrics_path.open("w", newline=""))
    metrics_writer.writerow(["step", "loss", "acc", "eval_disagreement_rate"])
    t0 = time.monotonic()
    best_disagreement = 1.0
    last_loss = float("nan")
    last_acc = float("nan")
    for step in range(1, steps + 1):
        m = trainer.step(masks_THW, batch_size=int(profile["nerv_batch_coords"]))
        last_loss, last_acc = m["loss"], m["acc"]
        if step % eval_every == 0 or step == steps:
            ev = trainer.evaluate_argmax_disagreement(masks_THW)
            dr = ev["disagreement_rate"]
            best_disagreement = min(best_disagreement, dr)
            metrics_writer.writerow([step, last_loss, last_acc, dr])
            print(
                f"[lane-12] step {step}/{steps} loss={last_loss:.4f} "
                f"train_acc={last_acc:.3f} eval_disagree={dr:.4f} "
                f"(best={best_disagreement:.4f}) elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )
        else:
            metrics_writer.writerow([step, last_loss, last_acc, ""])
    metrics_path.open("a").close()  # ensure flushed

    # ── Stage 4: encode EMA shadow + write artifacts ──
    blob = trainer.encode(weight_dtype=weight_dtype)
    out_payload = args.output_dir / "masks.nrv"
    out_payload.write_bytes(blob)
    elapsed = time.monotonic() - t0

    final_eval = trainer.evaluate_argmax_disagreement(masks_THW)

    # Round-trip sanity: decode the just-written payload + render argmax + compare
    from tac.nerv_mask_codec import decode_nerv_codec, render_mask_argmax  # noqa: E402

    decoded = decode_nerv_codec(blob)
    rendered = render_mask_argmax(
        decoded,
        num_frames=int(masks_THW.shape[0]),
        height=int(masks_THW.shape[1]),
        width=int(masks_THW.shape[2]),
        device=args.device,
    )
    rt_disagree = float((rendered.long() != masks_THW).float().mean().item())

    provenance = {
        "lane_name": "lane_12_nerv_mask_codec",
        "lane_script": "experiments/train_nerv_mask.py",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - elapsed)),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "git_hash": os.environ.get("GIT_HASH", "no-git"),
        "gpu_name": os.environ.get("GPU_NAME", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        "device": args.device,
        "profile": args.profile,
        "profile_nerv_keys": {k: profile[k] for k in nerv_keys},
        "seed": seed,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "num_frames": int(masks_THW.shape[0]),
        "mask_height": int(masks_THW.shape[1]),
        "mask_width": int(masks_THW.shape[2]),
        "gt_masks_source": args.gt_masks_source,
        "weight_dtype": weight_dtype,
        "nrv_payload_bytes": len(blob),
        "nrv_payload_path": str(out_payload),
        "codec_params": codec.num_params(),
        "fp16_weights_bytes": fp16_bytes,
        "int8_weights_bytes": int8_bytes,
        "final_loss": last_loss,
        "final_acc": last_acc,
        "final_eval_disagreement_rate": final_eval["disagreement_rate"],
        "best_eval_disagreement_rate": best_disagreement,
        "roundtrip_disagreement_rate": rt_disagree,
        "predicted_band_bytes": [23000, 80000],  # KB
        "kill_criterion_bytes": 100000,
        "kill_criterion_segnet_delta": 0.25,  # +25% vs Lane G v3
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(
        f"[lane-12] DONE: nrv={out_payload} ({len(blob)}B), "
        f"final_disagreement={final_eval['disagreement_rate']:.4f}, "
        f"roundtrip={rt_disagree:.4f}, "
        f"best={best_disagreement:.4f}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"RESULT_JSON {json.dumps({'lane': 'lane_12_nerv', 'bytes': len(blob), 'disagreement': final_eval['disagreement_rate']})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
