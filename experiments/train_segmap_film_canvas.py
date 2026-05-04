#!/usr/bin/env python
"""Train SegMap with Lane FC FiLM-Canvas hybrid (EUREKA #5).

Variant of experiments/train_segmap.py that uses
``tac.segmap_film_canvas_renderer.SegMapFilmCanvas`` instead of the vanilla
``SegMap`` — the only delta is the model class. The trainer (``SegMapTrainer``)
remains the canonical one in ``tac.segmap_renderer`` because the FiLM
modulation is purely architectural (forward-only), not loss-side.

CLAUDE.md compliance:
  * eval_roundtrip=True (inherited from SegMapTrainer).
  * --device cuda required.
  * KL-distill weight=0.002 (Lane G v3 canonical).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "upstream") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "upstream"))


def _seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--anchor-renderer", type=str, default=None)
    p.add_argument(
        "--anchor-poses",
        type=str,
        default="experiments/results/lane_a_landed/optimized_poses.pt",
    )
    p.add_argument(
        "--anchor-masks",
        type=str,
        default="experiments/results/lane_a_landed/iter_0/masks.mkv",
    )
    p.add_argument("--gt-video", type=str, default="upstream/videos/0.mkv")
    p.add_argument("--upstream", type=str, default="upstream")

    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--block-hidden", type=int, default=24)
    p.add_argument("--num-blocks", type=int, default=8)

    p.add_argument(
        "--variant",
        type=str,
        choices=("plain", "kl_distill"),
        default="kl_distill",
    )
    p.add_argument("--kl-distill-weight", type=float, default=0.002)
    p.add_argument("--kl-distill-temperature", type=float, default=2.0)

    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--roundtrip-noise-std", type=float, default=0.5)

    p.add_argument(
        "--device", type=str, default="cuda", choices=("cuda", "cpu")
    )

    # ── Council C OOM-class deep fixes (DF2 + DF3) ────────────────────────
    # Memory: .omx/research/council_oom_class_deep_fix_20260429.md.
    # Round 7 Defect #1: this script wraps the SAME SegMapTrainer that
    # train_segmap.py uses, so the SAME 21 GiB FastViT-attention OOM
    # applies. Check 87 STRICT now scans this file too; --bf16 +
    # --scorer-chunk are required on Modal A10G / RTX 4090 24 GB.
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable bf16 autocast around SegMapTrainer forward + scorer "
             "(Council C DF2). Halves PoseNet FastViT attention-map "
             "allocation. CUDA-only — raises if requested without CUDA.",
    )
    p.add_argument(
        "--scorer-chunk",
        type=int,
        default=0,
        help="Per-pair scorer chunk size for the dual scorer_forward_pair "
             "calls (Council C DF3). 0 = no chunking. N>0 = split each "
             "mini-batch's scorer call into chunks of N pairs along the "
             "batch dim. Recommended B*N <= 8 for 24 GB RTX 4090.",
    )

    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="lane_fc_film_canvas")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        print(
            "[train_segmap_fc] WARNING: --device cpu — bytes/score will DIFFER "
            "from contest-CUDA. Smoke only.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() is False."
        )
    return torch.device("cuda")


def _build_trainer_config(args: argparse.Namespace, device: torch.device):
    from tac.training import TrainConfig

    loss_mode = "standard" if args.variant == "plain" else "kl_distill"
    cfg_kwargs = dict(
        loss_mode=loss_mode,
        eval_roundtrip=True,
        roundtrip_noise_std=args.roundtrip_noise_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        tag=args.tag,
        output_dir=args.output_dir,
        # Council C OOM-class deep fixes (DF2 + DF3). The TrainConfig
        # validator rejects bf16=True without CUDA at SegMapTrainer.__init__
        # time (FORBIDDEN PATTERN: no silent MPS/CPU fallback).
        bf16=bool(getattr(args, "bf16", False)),
        scorer_chunk=int(getattr(args, "scorer_chunk", 0) or 0),
    )
    # Round 7 Defect #2 fix: ALWAYS pass kl_distill_weight (no longer
    # silently dropped — see comment in experiments/train_segmap.py).
    cfg_kwargs["kl_distill_weight"] = args.kl_distill_weight
    if loss_mode == "kl_distill":
        cfg_kwargs["kl_distill_scope"] = "segnet_aux"
    if hasattr(TrainConfig, "model_fields"):
        fields = set(TrainConfig.model_fields.keys())
        if "kl_distill_temperature" in fields:
            cfg_kwargs["kl_distill_temperature"] = args.kl_distill_temperature
        if "temperature_start" in fields and loss_mode == "kl_distill":
            cfg_kwargs.setdefault("temperature_start", args.kl_distill_temperature)
            cfg_kwargs.setdefault("temperature_end", 1.0)
    return TrainConfig(**cfg_kwargs)


def _load_pairs(args: argparse.Namespace, device: torch.device):
    from tac.mask_codec import decode_masks_auto
    from tac.data import load_gt_video

    masks_path = Path(args.anchor_masks)
    if not masks_path.exists():
        raise FileNotFoundError(f"--anchor-masks not found: {masks_path}")
    mask_classes = decode_masks_auto(masks_path)
    if mask_classes.ndim != 3:
        raise RuntimeError(
            f"masks.mkv decoded to ndim={mask_classes.ndim} (expected 3)."
        )
    h, w = mask_classes.shape[-2:]
    if (h, w) != (384, 512):
        raise RuntimeError(
            f"Mask resolution {h}x{w} != 384x512. CATASTROPHIC — refusing to train."
        )
    gt_video_path = Path(args.gt_video)
    if not gt_video_path.exists():
        raise FileNotFoundError(f"--gt-video not found: {gt_video_path}")
    gt_frames = load_gt_video(gt_video_path, n_frames=mask_classes.shape[0])
    return mask_classes, gt_frames


def _build_pair_tensors(mask_classes: torch.Tensor, gt_frames):
    from tac.mask_grayscale_lut import (
        encode_masks_grayscale,
        grayscale_to_probability_map,
    )

    n = mask_classes.shape[0]
    if n % 2 != 0:
        raise RuntimeError(
            f"Frame count {n} is odd; cannot form non-overlapping pairs."
        )
    half = n // 2
    gray = encode_masks_grayscale(mask_classes.long())
    soft = grayscale_to_probability_map(gray, sigma=15.0, channel_first=True)
    mask_pairs = soft.view(half, 2, 5, *soft.shape[-2:])
    if isinstance(gt_frames, list):
        gt_tensor = torch.stack(gt_frames, dim=0)
    else:
        gt_tensor = gt_frames
    if gt_tensor.ndim != 4:
        raise RuntimeError(f"gt_tensor ndim={gt_tensor.ndim} (expected 4)")
    gt_pairs = gt_tensor.view(half, 2, *gt_tensor.shape[-3:])
    return mask_pairs, gt_pairs


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    device = _resolve_device(args.device)

    from tac.segmap_film_canvas_renderer import SegMapFilmCanvas
    from tac.segmap_renderer import SegMapTrainer
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    cfg = _build_trainer_config(args, device)

    NUM_FRAMES = 1200
    model = SegMapFilmCanvas(
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=NUM_FRAMES,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train_segmap_fc] SegMapFilmCanvas hidden={args.hidden} "
        f"blocks={args.num_blocks} params={n_params:,}",
        flush=True,
    )

    if args.dry_run:
        meta = {
            "dry_run": True,
            "n_params": n_params,
            "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
            "distillation_policy": cfg.distillation_policy_provenance(),
            "device": str(device),
        }
        (output_dir / "segmap_fc_dry_run.json").write_text(json.dumps(meta, indent=2))
        print("[train_segmap_fc] DRY RUN OK")
        return 0

    posenet, segnet = load_differentiable_scorers(args.upstream, device=device)
    posenet.eval()
    segnet.eval()
    for net in (posenet, segnet):
        for prm in net.parameters():
            prm.requires_grad_(False)

    trainer = SegMapTrainer(
        model=model,
        config=cfg,
        posenet=posenet,
        segnet=segnet,
        device=device,
    )
    ema = EMA(model, decay=args.ema_decay)

    mask_classes, gt_frames = _load_pairs(args, device)
    mask_pairs, gt_pairs = _build_pair_tensors(mask_classes, gt_frames)
    print(
        f"[train_segmap_fc] training pairs: {mask_pairs.shape[0]} (non-overlapping)",
        flush=True,
    )

    history: list[dict] = []
    t_start = time.monotonic()
    for epoch in range(args.epochs):
        epoch_metrics = trainer.train_epoch(
            mask_pairs=mask_pairs,
            gt_pairs=gt_pairs,
            ema=ema,
        )
        epoch_metrics = dict(epoch_metrics)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        if epoch % max(1, args.epochs // 50) == 0 or epoch == args.epochs - 1:
            loss = epoch_metrics.get("loss", float("nan"))
            print(
                f"[train_segmap_fc] epoch={epoch} loss={loss:.6f} "
                f"seg={epoch_metrics.get('seg_dist', float('nan')):.6f} "
                f"pose={epoch_metrics.get('pose_dist', float('nan')):.6f}",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    print(f"[train_segmap_fc] training complete in {elapsed:.1f}s", flush=True)

    # PCC3 internal-consistency check (Council 2026-04-30 ~23:30 UTC, DD2/DD3
    # 10/10 vote): SegMap-FC training on T4 has the same per-epoch cost as
    # plain SegMap (~2 s/epoch) PLUS the FiLM table/canvas overhead (~0.2 s).
    # 0.5 s/epoch floor flags any 4× regression. To no-op the training, pass
    # --epochs 0; a stub-loop pretending to be real training will trip this.
    # Memory: feedback_grand_council_pcc3_stats_consistency_20260430.md.
    MIN_WALL_PER_EPOCH_SEC = 0.5  # see council vote
    if args.epochs > 0:
        expected_min = args.epochs * MIN_WALL_PER_EPOCH_SEC
        if elapsed < expected_min:
            raise RuntimeError(
                f"PCC3 STUB-LOOP DETECTED: claimed {args.epochs} epochs in "
                f"{elapsed:.2f}s — below floor {expected_min:.2f}s "
                f"({MIN_WALL_PER_EPOCH_SEC}s/epoch). This indicates the train "
                f"loop is not actually training. To run a deliberate no-op, "
                f"pass --epochs 0 instead of pretending. Memory: "
                f"feedback_grand_council_pcc3_stats_consistency_20260430.md."
            )

    train_ckpt_path = output_dir / "segmap_fc_train.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
            "distillation_policy": cfg.distillation_policy_provenance(),
            "epoch": args.epochs,
            "n_params": n_params,
        },
        train_ckpt_path,
    )

    # Inference state: dump the EMA-applied model (export_inference_state on
    # the SegMap subclass directly, mirroring SegMapTrainer.export_inference_state_dict
    # but capturing the FiLM table key).
    live = {k: v.clone() for k, v in model.state_dict().items()}
    ema.apply(model)
    try:
        inference_state = model.export_inference_state()
    finally:
        model.load_state_dict(live)
    inference_path = output_dir / "segmap_inference.pt"
    torch.save(inference_state, inference_path)

    summary = {
        "tag": args.tag,
        "n_params": n_params,
        "epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
        "distillation_policy": cfg.distillation_policy_provenance(),
        "history": history,
        "has_film_table": True,
    }
    (output_dir / "segmap_fc_train.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[train_segmap_fc] wrote {train_ckpt_path}, {inference_path}, summary.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
