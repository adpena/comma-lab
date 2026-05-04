#!/usr/bin/env python
"""Train SegMap renderer (Selfcomp 0.38 paradigm clone).

Forks the structure of `experiments/train_distill.py` but uses the new
``tac.segmap_renderer.SegMap`` architecture and ``SegMapTrainer``. The model
takes 5-class SegNet logits → produces RGB frames with a per-frame
learned affine latent.

Variants (selected via ``--variant``):
  * ``plain`` — standard scorer loss (eval_roundtrip=True, noise_std=0.5)
  * ``kl_distill`` — adds Quantizr KL-distill auxiliary on SegNet logits T=2.0
  * ``hessian_quant`` — same training as ``kl_distill``; the Hessian work
    happens at export time only.

CRITICAL CONSTRAINTS (CLAUDE.md):
  * eval_roundtrip MUST default True (Trainer raises if False).
  * Device MUST be cuda; mps is rejected (PreflightError raised by trainer).
  * Auth eval is the responsibility of the lane shell script.
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--anchor-renderer", type=str, default=None,
                   help="Optional renderer.bin reference (not loaded — SegMap "
                        "arch is independent).")
    p.add_argument("--anchor-poses", type=str,
                   default="experiments/results/lane_a_landed/optimized_poses.pt")
    p.add_argument("--anchor-masks", type=str,
                   default="experiments/results/lane_a_landed/iter_0/masks.mkv",
                   help="Pre-extracted full-res 384x512 masks. Half-res FORBIDDEN.")
    p.add_argument("--gt-video", type=str, default="upstream/videos/0.mkv")
    p.add_argument("--upstream", type=str, default="upstream")

    # Architecture sweep — note that SegMap hardcodes in_channels=5 and
    # out_channels=3; only hidden / block_hidden / num_blocks are tunable.
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--block-hidden", type=int, default=24)
    p.add_argument("--num-blocks", type=int, default=8)

    p.add_argument(
        "--variant",
        type=str,
        choices=("plain", "kl_distill", "hessian_quant"),
        default="plain",
    )
    p.add_argument(
        "--arch",
        type=str,
        choices=("segmap", "segmap_homography"),
        default="segmap",
        help="segmap=canonical 6-DOF affine; segmap_homography=8-DOF "
             "perspective homography frame embedding (Lane HM-S).",
    )
    p.add_argument("--kl-distill-weight", type=float, default=0.002,
                   help="Hinton-regime KL distill weight (matches Lane G v3).")
    p.add_argument("--kl-distill-temperature", type=float, default=2.0)

    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--roundtrip-noise-std", type=float, default=0.5,
                   help="Gaussian noise std after STE quant in eval roundtrip.")

    p.add_argument(
        "--pair-weights", type=str, default=None,
        help="Optional .pt file containing a (N_pairs,) float tensor of "
             "per-pair training weights (Lane WC-S Curator outlier weighting). "
             "When supplied, each pair's loss is multiplied by its weight; "
             "must have length == number of training pairs.",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="cuda for real training. cpu is an explicit smoke-test opt-in "
             "(banner will be printed; bytes/score will differ).",
    )

    # ── Council C OOM-class deep fixes (DF2 + DF3) ────────────────────
    # Memory: .omx/research/council_oom_class_deep_fix_20260429.md.
    # Required for SegMap-class lanes per Check 87 STRICT.
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable bf16 autocast around SegMapTrainer forward + scorer. "
             "Halves PoseNet FastViT attention-map allocation. CUDA-only — "
             "raises if requested without CUDA. Council C DF2.",
    )
    p.add_argument(
        "--scorer-chunk",
        type=int,
        default=0,
        help="Per-pair scorer chunk size for the dual scorer_forward_pair "
             "calls. 0 = no chunking (legacy unchunked path). N>0 = split "
             "each mini-batch's scorer call into chunks of N pairs along "
             "the batch dim. Cuts attention-map memory by ~chunk_size. "
             "Council C DF3 — recommended B*N <= 8 for 24 GB RTX 4090.",
    )

    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--tag", type=str, default="lane_segmap")

    p.add_argument("--dry-run", action="store_true",
                   help="Instantiate trainer + write metadata, then exit 0.")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    """CUDA-required default. CPU explicit opt-in with banner. mps FORBIDDEN."""
    if device_str == "cpu":
        print(
            "[train_segmap] WARNING: --device cpu — bytes/score will DIFFER "
            "from contest-CUDA. Use only for smoke tests.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() is False. "
            "This script CUDA-requires by design; pass --device cpu for a smoke test."
        )
    return torch.device("cuda")


def _build_trainer_config(args: argparse.Namespace, device: torch.device):
    """Map CLI args → tac.training.TrainConfig (the SegMapTrainer reuses it)."""
    from tac.training import TrainConfig

    loss_mode = "standard" if args.variant == "plain" else "kl_distill"
    # NOTE: the canonical TrainConfig has many fields; we set the ones the
    # SegMapTrainer reads (loss_mode, eval_roundtrip, roundtrip_noise_std,
    # kl_distill_*) and accept defaults for the rest.
    cfg_kwargs = dict(
        loss_mode=loss_mode,
        eval_roundtrip=True,  # NON-NEGOTIABLE
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
    # Round 7 Defect #2 fix: ALWAYS pass kl_distill_weight (no longer a
    # silent no-op when the field is missing — the field is now declared
    # on TrainConfig and the SegMapTrainer reads self.config.kl_distill_weight
    # instead of a hard-coded literal). The conditional plumbing pattern
    # was a silent-default-override foot-gun (memory:
    # feedback_silent_default_bug_class_findings_20260429.md).
    cfg_kwargs["kl_distill_weight"] = args.kl_distill_weight
    if loss_mode == "kl_distill":
        cfg_kwargs["kl_distill_scope"] = "segnet_aux"
    if hasattr(TrainConfig, "model_fields"):
        fields = set(TrainConfig.model_fields.keys())
        if "kl_distill_temperature" in fields:
            cfg_kwargs["kl_distill_temperature"] = args.kl_distill_temperature
        if "temperature_start" in fields and loss_mode == "kl_distill":
            # kl_distill validator requires temperature_start >= 2.0
            cfg_kwargs.setdefault("temperature_start", args.kl_distill_temperature)
            cfg_kwargs.setdefault("temperature_end", 1.0)
    return TrainConfig(**cfg_kwargs)


def _load_pairs(args: argparse.Namespace, device: torch.device):
    """Load (mask_classes, gt_frames) at full 384x512 resolution."""
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
            f"Mask resolution {h}x{w} != 384x512. CATASTROPHIC — see CLAUDE.md "
            f"'MASKS.MKV AT 48x64 DESTROYED THE SCORE'. Refusing to train."
        )

    gt_video_path = Path(args.gt_video)
    if not gt_video_path.exists():
        raise FileNotFoundError(f"--gt-video not found: {gt_video_path}")
    gt_frames = load_gt_video(gt_video_path, n_frames=mask_classes.shape[0])
    return mask_classes, gt_frames


def _build_pair_tensors(mask_classes: torch.Tensor, gt_frames):
    """Adjacent-pair construction matching upstream evaluate.py seq_len=2.

    Returns (mask_pairs, gt_pairs) shaped (P, 2, ...) — the SegMapTrainer
    expects (B, T, num_classes, H, W) for masks and (B, T, 3, H, W) for GT.
    """
    from tac.mask_grayscale_lut import (
        encode_masks_grayscale,
        grayscale_to_probability_map,
    )

    n = mask_classes.shape[0]
    if n % 2 != 0:
        raise RuntimeError(f"Frame count {n} is odd; cannot form non-overlapping pairs.")
    half = n // 2

    # Train on the exact analog distribution the inflate path will feed to
    # SegMap. Public Selfcomp #2 does not hard-argmax grayscale masks before
    # the renderer; it feeds the soft Gaussian-LUT probability map.
    gray = encode_masks_grayscale(mask_classes.long())
    soft = grayscale_to_probability_map(gray, sigma=15.0, channel_first=True)
    mask_pairs = soft.view(half, 2, 5, *soft.shape[-2:])

    # GT frames may be a list of (3, H, W) tensors. Stack into (N, 3, H, W).
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

    from tac.segmap_renderer import SegMap, SegMapHomography, SegMapTrainer
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    cfg = _build_trainer_config(args, device)

    # SegMap requires max_frame_index for the per-frame affine embedding.
    # Use 1200 frames (NUM_FRAMES) as the canonical upper bound.
    NUM_FRAMES = 1200
    arch_cls = SegMapHomography if args.arch == "segmap_homography" else SegMap
    model = arch_cls(
        hidden=args.hidden,
        block_hidden=args.block_hidden,
        num_blocks=args.num_blocks,
        max_frame_index=NUM_FRAMES,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_segmap] SegMap variant={args.variant} hidden={args.hidden} "
          f"blocks={args.num_blocks} params={n_params:,}", flush=True)

    if args.dry_run:
        meta = {
            "dry_run": True,
            "variant": args.variant,
            "n_params": n_params,
            "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
            "distillation_policy": cfg.distillation_policy_provenance(),
            "device": str(device),
        }
        (output_dir / "segmap_dry_run.json").write_text(json.dumps(meta, indent=2))
        print("[train_segmap] DRY RUN OK")
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
    print(f"[train_segmap] training pairs: {mask_pairs.shape[0]} (non-overlapping)",
          flush=True)

    # Lane WC-S Curator outlier pair weighting (optional). When supplied, the
    # weights are checked for shape match and broadcast into the pair loss
    # path. CLAUDE.md non-negotiable: validate at every boundary — a wrong-
    # length weight tensor must hard-error, not silently truncate.
    pair_weights = None
    if args.pair_weights:
        pw_path = Path(args.pair_weights)
        if not pw_path.exists():
            raise FileNotFoundError(f"--pair-weights not found: {pw_path}")
        pair_weights = torch.load(pw_path, map_location="cpu", weights_only=False)
        if isinstance(pair_weights, dict) and "weights" in pair_weights:
            pair_weights = pair_weights["weights"]
        if not isinstance(pair_weights, torch.Tensor):
            raise RuntimeError(
                f"--pair-weights file must contain a tensor or {{'weights': tensor}}, "
                f"got {type(pair_weights).__name__}"
            )
        if pair_weights.numel() != mask_pairs.shape[0]:
            raise RuntimeError(
                f"--pair-weights length {pair_weights.numel()} != n_pairs {mask_pairs.shape[0]}. "
                f"Refusing to silently broadcast or truncate."
            )
        pair_weights = pair_weights.to(torch.float32)
        if pair_weights.min() < 0:
            raise RuntimeError(
                f"--pair-weights must be non-negative; min={float(pair_weights.min())}"
            )
        print(
            f"[train_segmap] pair_weights loaded: n={pair_weights.numel()} "
            f"min={float(pair_weights.min()):.3f} max={float(pair_weights.max()):.3f}",
            flush=True,
        )

    history: list[dict] = []
    t_start = time.monotonic()
    # BUG CLASS B fix (2026-04-29): plumb --batch-size into train_epoch so
    # the legacy "all 600 pairs in one forward = 7.03 GiB OOM on T4" path
    # is closed. SegMapTrainer.train_epoch chunks pairs into mini-batches
    # of `batch_size` and accumulates gradients; one optimizer.step() per
    # epoch preserves the legacy semantics.
    import inspect as _inspect
    _train_sig = _inspect.signature(trainer.train_epoch)
    _supports_batch = "batch_size" in _train_sig.parameters
    _supports_pair_weights = "pair_weights" in _train_sig.parameters
    for epoch in range(args.epochs):
        kwargs = {}
        if pair_weights is not None:
            if _supports_pair_weights:
                kwargs["pair_weights"] = pair_weights
            else:
                # Fail loud on first iteration: silent drop of pair_weights
                # would invalidate the lane's hypothesis.
                if epoch == 0:
                    raise RuntimeError(
                        "--pair-weights supplied but SegMapTrainer.train_epoch "
                        "does not accept the 'pair_weights' kwarg. Wire it in "
                        "or remove --pair-weights."
                    )
        if _supports_batch:
            kwargs["batch_size"] = args.batch_size
        elif epoch == 0:
            # Older SegMapTrainer without batch chunking → loud warning so
            # operators see why VRAM may explode under big-batch eval.
            print(
                "[train_segmap] WARNING: SegMapTrainer.train_epoch lacks "
                "batch_size kwarg; running unchunked (T4 may OOM at 600 pairs)",
                flush=True,
            )
        epoch_metrics = trainer.train_epoch(
            mask_pairs=mask_pairs,
            gt_pairs=gt_pairs,
            ema=ema,
            **kwargs,
        )
        epoch_metrics = dict(epoch_metrics)
        epoch_metrics["epoch"] = epoch
        history.append(epoch_metrics)
        if epoch % max(1, args.epochs // 50) == 0 or epoch == args.epochs - 1:
            loss = epoch_metrics.get("loss", float("nan"))
            # SegMapTrainer.train_epoch returns canonical keys "seg_dist" +
            # "pose_dist". Earlier code read "seg"/"seg_loss"/"pose"/"pose_loss"
            # which are NEVER produced — display showed NaN for 5h of GPU
            # training (Lane DARTS-S V1 incident 2026-04-29 PM, $1.41 wasted
            # on inscrutable output). Check 85 STRICT prevents regression.
            seg = epoch_metrics.get("seg_dist", float("nan"))
            pose = epoch_metrics.get("pose_dist", float("nan"))
            print(
                f"[train_segmap] epoch={epoch} loss={loss:.6f} seg={seg:.6f} pose={pose:.6f}",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    print(f"[train_segmap] training complete in {elapsed:.1f}s", flush=True)

    # PCC3 internal-consistency check (Council 2026-04-30 ~23:30 UTC, DD2/DD3
    # 10/10 vote): producer-side assertion that elapsed wall-clock is at least
    # the lower bound a real epoch would consume. SegMap on T4 with 600 mask
    # pairs takes ~2 s/epoch (B=8 chunk × 75 chunks × ~0.025 s/chunk on T4 fp16);
    # a stub loop pretending to be SegMap training would come in at <0.1s/epoch.
    # The 0.5 s/epoch floor flags any 4× regression while leaving slack for
    # batch-size/device variation. CPU smoke runs with --device cpu still trip
    # this if epochs > 0; intentional smoke tests should pass --epochs 0 OR add
    # an explicit args.smoke flag (currently this script has none — that gap is
    # noted for a follow-up: dispatch scripts that want to no-op should pass
    # --epochs 0). Memory: feedback_grand_council_pcc3_stats_consistency_20260430.
    MIN_WALL_PER_EPOCH_SEC = 0.5  # see council vote
    if args.epochs > 0:
        expected_min = args.epochs * MIN_WALL_PER_EPOCH_SEC
        if elapsed < expected_min:
            raise RuntimeError(
                f"PCC3 STUB-LOOP DETECTED: claimed {args.epochs} epochs in "
                f"{elapsed:.2f}s — below floor {expected_min:.2f}s "
                f"({MIN_WALL_PER_EPOCH_SEC}s/epoch). This indicates the train "
                f"loop is not actually training (data loader empty? trainer "
                f"step is no-op? tensor-shape early-exit?). The IMP cycle 0 "
                f"= 1.98 metabug class. To run a deliberate no-op, pass "
                f"--epochs 0 instead of pretending. Memory: "
                f"feedback_grand_council_pcc3_stats_consistency_20260430.md."
            )

    train_ckpt_path = output_dir / "segmap_train.pt"
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

    inference_state = trainer.export_inference_state_dict(ema)
    inference_path = output_dir / "segmap_inference.pt"
    torch.save(inference_state, inference_path)

    summary = {
        "variant": args.variant,
        "tag": args.tag,
        "n_params": n_params,
        "epochs": args.epochs,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
        "distillation_policy": cfg.distillation_policy_provenance(),
        "history": history,
    }
    (output_dir / "segmap_train.json").write_text(json.dumps(summary, indent=2))
    print(f"[train_segmap] wrote {train_ckpt_path}, {inference_path}, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
