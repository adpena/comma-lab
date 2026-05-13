#!/usr/bin/env python3
"""Train the szabolcs no-masks SegMap renderer (Lane SZ Phase 2).

Pipeline (matches /tmp/szabolcs_re/inflate.py at training time)
---------------------------------------------------------------
1. Decode the GT video (1200 frames @ 1164x874).
2. Per training step, sample a random frame index ``idx``.
3. Convert the frame's luma -> bicubic to (384, 512) -> Gaussian softmax LUT
   -> 5-channel probability map.
4. Forward (probability_map, frame_index) through SzabolcsRenderer to get a
   reconstructed RGB at (384, 512); bilinear-upscale to (874, 1164) for the
   pixel L1 supervision.
5. L1 loss vs the GT RGB. We do NOT load PoseNet/SegNet — the szabolcs
   paradigm is a pure pixel reconstruction overfit; the per-frame affine
   embeddings memorize the necessary motion per frame.

Why this is enough
------------------
The renderer trains to overfit a single 1200-frame video. The per-frame
affine embedding is essentially a memorized 6-DoF zoom/shear per frame — it
absorbs the motion that PoseNet would otherwise need to penalize. Empirically
(per project_szabolcs_full_re_20260426) this produces auth scores in the
[0.30, 0.50] band on contest-CUDA.

Usage
-----
    python experiments/train_szabolcs.py \
        --device cuda \
        --total-epochs 1200 \
        --lr 5e-4 \
        --output-dir results/lane_sz_phase2

Outputs
-------
* ``szabolcs_best.pt`` — torch pickle ``{model_state_dict, config, meta}``.
* ``szabolcs_last.pt`` — final-epoch checkpoint.
* ``train_log.jsonl`` — per-epoch JSON metrics.

The SZv1 export is a separate tool (``experiments/export_szabolcs_archive.py``);
this script only produces a pickle. Strict-scorer-rule: nothing in the train
loop loads PoseNet or SegNet.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Path setup ──────────────────────────────────────────────────────────
_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo / "src"))

from tac.contrib.szabolcs_renderer import (  # noqa: E402
    CAMERA_SIZE,
    SEGMAP_INPUT_SIZE,
    build_szabolcs_renderer,
    encode_luma_to_probability_map,
)
from tac.data import decode_video  # noqa: E402
# Council D EMA wire-in (2026-04-29 PM): canonical EMA shadow with the
# Quantizr decay=0.997 default. Selfcomp PR#56 (#2 leaderboard at 0.38)
# uses EMA throughout its anchor → finetune → joint → QAT → final pipeline;
# this script (the direct replica of that pipeline) was missing it
# entirely per .omx/research/council_ema_audit_20260429.md §3.4.
from tac.training import EMA  # noqa: E402


# ── CLI ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the szabolcs SegMap renderer.")
    p.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cuda"],
        help="Training device. CUDA REQUIRED — szabolcs is a contest-bound "
        "training run, MPS/CPU produce numerically different scores. The "
        "CLI rejects --device cpu/mps explicitly per CLAUDE.md.",
    )
    p.add_argument(
        "--video",
        type=str,
        default=str(_repo / "upstream" / "videos" / "0.mkv"),
        help="Path to the GT video to overfit.",
    )
    p.add_argument("--total-epochs", type=int, default=1200,
                   help="Total epoch count. Each epoch covers all frames once.")
    p.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay (default 0.0 — szabolcs has very "
                   "few weights, decay just hurts convergence).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Frames per training step.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="L2 grad-norm clip; 0 to disable.")
    p.add_argument("--seed", type=int, default=1234)
    # Council D 2026-04-29 PM: EMA decay (Quantizr canonical 0.997). Per
    # CLAUDE.md "EMA — NON-NEGOTIABLE", every training path that ships a
    # checkpoint must EMA the model and save the EMA shadow.
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (Quantizr 0.997). Mandatory per "
                   "CLAUDE.md non-negotiable; the EMA shadow is what gets "
                   "saved as the inference checkpoint.")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for checkpoints, log, provenance.")
    p.add_argument("--tag", type=str, default="szabolcs",
                   help="Filename prefix for checkpoints.")
    # Architecture (defaults match the reference; override for sweeps).
    p.add_argument("--hidden", type=int, default=32,
                   help="Residual stream width (reference 32-64).")
    p.add_argument("--block-hidden", type=int, default=None,
                   help="Intermediate width inside each residual block "
                   "(default: hidden).")
    p.add_argument("--num-blocks", type=int, default=4,
                   help="Number of residual blocks (reference 4).")
    p.add_argument("--latent-input-scale", type=float, default=1.0)
    # Logging / smoke knobs.
    p.add_argument("--log-interval", type=int, default=10,
                   help="Print + log every N epochs (default 10).")
    p.add_argument("--max-frames", type=int, default=1200,
                   help="Truncate GT to first N frames (1200 = full video).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: forces total_epochs=5 and max-frames=8 "
                   "for fast CI / unit-test runs. The training loss is "
                   "expected to drop monotonically; it WON'T converge.")
    return p.parse_args()


# ── Reproducibility ─────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training step ───────────────────────────────────────────────────────


def make_luma_low(rgb_uint8: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Convert (B, H, W, 3) uint8 RGB to (B, target_h, target_w) float32 luma.

    Uses BT.601 luma weights, matching the reference inflate which feeds
    pyav-decoded gray planes through bicubic resize.
    """
    rgb = rgb_uint8.float()
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    luma = luma.unsqueeze(1)  # (B, 1, H, W)
    luma = F.interpolate(
        luma, size=(target_h, target_w), mode="bicubic", align_corners=False
    )
    return luma.squeeze(1).clamp(0, 255)


def upscale_to_camera(rgb_low: torch.Tensor) -> torch.Tensor:
    """Upscale renderer output (B, 3, segH, segW) to (B, 3, camH, camW)."""
    return F.interpolate(
        rgb_low,
        size=(CAMERA_SIZE[1], CAMERA_SIZE[0]),
        mode="bicubic",
        align_corners=False,
    ).clamp(0, 255)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.total_epochs = min(args.total_epochs, 5)
        args.max_frames = min(args.max_frames, 8)
        args.log_interval = 1

    if args.device != "cuda":  # defense in depth — argparse choices already enforce it
        raise SystemExit("--device cuda required (CLAUDE.md non-negotiable)")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable on this machine; szabolcs training requires "
            "CUDA per CLAUDE.md. Refusing CPU/MPS fallback."
        )
    device = torch.device("cuda", 0)

    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    best_path = out_dir / f"{args.tag}_best.pt"
    last_path = out_dir / f"{args.tag}_last.pt"
    prov_path = out_dir / "train_szabolcs_provenance.json"

    # ── Load data ────────────────────────────────────────────────────────
    print(f"[train_szabolcs] decoding {args.video} (max_frames={args.max_frames}) …",
          flush=True)
    frames_full = decode_video(args.video, target_h=CAMERA_SIZE[1], target_w=CAMERA_SIZE[0])
    frames = frames_full[: args.max_frames]
    if len(frames) < 2:
        raise SystemExit(f"need at least 2 frames; decoded {len(frames)}")
    n_frames = len(frames)
    # Stack once on CPU; index per step. (1200 * 874 * 1164 * 3) ≈ 3.7GB at uint8
    # which fits in host memory comfortably.
    frames_stack = torch.stack(frames)  # (N, H, W, 3) uint8
    print(f"[train_szabolcs] decoded {n_frames} frames at {frames_stack.shape[1:]}",
          flush=True)

    # max_frame_index needs slack: the reference uses 2 * idx + 1 indexing.
    max_frame_index = max(2 * n_frames, 1200)

    # ── Build model ─────────────────────────────────────────────────────
    bundle = build_szabolcs_renderer(
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=max_frame_index,
        latent_input_scale=args.latent_input_scale,
        quiet=False,
    )
    model = bundle.model.to(device)
    lut = bundle.lut.to(device)
    model.train()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── EMA shadow (Council D 2026-04-29 PM) ────────────────────────────
    # Mandatory per CLAUDE.md non-negotiable. update() is called AFTER
    # every optim.step(); apply() is called ONLY at checkpoint save with
    # snapshot+restore so the live model's gradient progress is never
    # squashed by the shadow (the "EMA shadows back into live"
    # antipattern documented in the Council D audit §6).
    ema = EMA(model, decay=args.ema_decay)
    print(f"[train_szabolcs] EMA enabled (decay={args.ema_decay})", flush=True)

    # ── Provenance ──────────────────────────────────────────────────────
    prov = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cuda",
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "video": args.video,
        "n_frames": n_frames,
        "total_epochs": args.total_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "hidden": args.hidden,
        "block_hidden": args.block_hidden,
        "num_blocks": args.num_blocks,
        "param_count": bundle.total_params,
        "max_frame_index": max_frame_index,
        "smoke": args.smoke,
        "ema_decay": args.ema_decay,
        "predicted_band_contest_cuda": [0.30, 0.50],
        "score_lane": "contest-CUDA",
        "rationale": (
            "szabolcs PR#56 reports 0.36 [contest-CUDA]. Our replica matches "
            "the architecture exactly; per-frame affine embeddings memorize "
            "motion. Predicted band [0.30, 0.50] reflects the architectural "
            "fidelity (low) and the absence of QAT/pose TTO refinement (high)."
        ),
    }
    prov_path.write_text(json.dumps(prov, indent=2))

    # ── Training loop ───────────────────────────────────────────────────
    seg_h, seg_w = SEGMAP_INPUT_SIZE[1], SEGMAP_INPUT_SIZE[0]
    cam_h, cam_w = CAMERA_SIZE[1], CAMERA_SIZE[0]
    best_loss = float("inf")
    epochs_per_pass = max(1, n_frames // args.batch_size)

    print(f"[train_szabolcs] starting {args.total_epochs} epochs × "
          f"{epochs_per_pass} steps each, batch={args.batch_size}, "
          f"params={bundle.total_params:,}", flush=True)
    t0 = time.monotonic()

    with open(log_path, "w") as log_file:
        for epoch in range(args.total_epochs):
            epoch_loss = 0.0
            n_steps = 0
            perm = torch.randperm(n_frames).tolist()
            for step in range(epochs_per_pass):
                lo = (step * args.batch_size) % n_frames
                idxs = perm[lo: lo + args.batch_size]
                if len(idxs) < args.batch_size:
                    # wrap to top of perm if we ran off the end
                    idxs = idxs + perm[: args.batch_size - len(idxs)]
                idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)

                rgb_full = frames_stack[idxs].to(device, non_blocking=True)
                luma_low = make_luma_low(rgb_full, seg_h, seg_w)
                prob_map = encode_luma_to_probability_map(luma_low, lut=lut)

                # frame_indices use the (2 * idx) convention per the reference
                # so the same trained embedding works at inflate.
                frame_idx_seq = 2 * idx_tensor

                pred_low = model(prob_map, frame_idx_seq)
                pred_full = upscale_to_camera(pred_low)

                # L1 vs GT RGB at camera resolution.
                gt_full = rgb_full.permute(0, 3, 1, 2).float()
                loss = F.l1_loss(pred_full, gt_full)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()
                # Council D 2026-04-29: EMA update AFTER optim.step(). Writes
                # only to ema.shadow; never touches the live model. CLAUDE.md
                # non-negotiable.
                ema.update(model)

                epoch_loss += float(loss.item())
                n_steps += 1

            mean = epoch_loss / max(n_steps, 1)
            elapsed = time.monotonic() - t0
            log_record = {
                "epoch": epoch,
                "mean_l1": mean,
                "elapsed_s": round(elapsed, 2),
                "best_so_far": min(mean, best_loss),
            }
            log_file.write(json.dumps(log_record) + "\n")
            log_file.flush()

            if mean < best_loss:
                best_loss = mean
                # Council D 2026-04-29 PM: ship EMA shadow as the
                # inference state_dict (Quantizr / Selfcomp pattern).
                # ``ema.state_dict()`` returns clones — the live model is
                # never mutated. CLAUDE.md non-negotiable: inference
                # bytes come from the EMA shadow, never the live model.
                # ``model_state_dict_live`` is preserved for diagnostic /
                # resume purposes.
                torch.save({
                    "model_state_dict": ema.state_dict(),
                    "model_state_dict_live": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "ema_decay": args.ema_decay,
                    "config": prov,
                    "epoch": epoch,
                    "loss": mean,
                }, best_path)

            if epoch % args.log_interval == 0 or epoch == args.total_epochs - 1:
                print(f"[train_szabolcs] epoch={epoch} mean_l1={mean:.4f} "
                      f"best={best_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    # Final-epoch checkpoint: ship EMA shadow as the inference state_dict
    # (Council D 2026-04-29 PM, CLAUDE.md non-negotiable). Live state
    # preserved for diagnostic / resume.
    torch.save({
        "model_state_dict": ema.state_dict(),
        "model_state_dict_live": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "ema_decay": args.ema_decay,
        "config": prov,
        "epoch": args.total_epochs - 1,
        "loss": mean,
    }, last_path)

    final_record = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_loss": best_loss,
        "final_loss": mean,
        "total_seconds": round(time.monotonic() - t0, 2),
    }
    (out_dir / "train_szabolcs_summary.json").write_text(json.dumps(final_record, indent=2))
    print(f"[train_szabolcs] DONE. best={best_loss:.4f} ckpt={best_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
