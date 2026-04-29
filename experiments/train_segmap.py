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
    )
    # KL-distill knobs only exist when tac.training.TrainConfig declares
    # them; check via field listing to avoid a hard error if missing.
    if hasattr(TrainConfig, "model_fields"):
        fields = set(TrainConfig.model_fields.keys())
        if "kl_distill_weight" in fields:
            cfg_kwargs["kl_distill_weight"] = args.kl_distill_weight
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
    from tac.mask_grayscale_lut import create_gaussian_softmax_lut

    n = mask_classes.shape[0]
    if n % 2 != 0:
        raise RuntimeError(f"Frame count {n} is odd; cannot form non-overlapping pairs.")
    half = n // 2

    # Convert class IDs to softmax-LUT logits (B, num_classes, H, W).
    lut = create_gaussian_softmax_lut(sigma=15.0)
    # Encode via the lookup: classes (0..4) → softmax over 5 classes. The LUT
    # is keyed on grayscale 0..255 values; we feed an inverse mapping that
    # maps class → grayscale-mean → 5-vector.
    # Simpler path: one-hot the class IDs and let the trainer do its own
    # softmax / smoothing if needed. The Selfcomp paradigm uses one-hot.
    one_hot = torch.nn.functional.one_hot(
        mask_classes.long(), num_classes=5
    ).permute(0, 3, 1, 2).float()
    # one_hot: (N, 5, H, W). Pairs: (half, 2, 5, H, W).
    mask_pairs = one_hot.view(half, 2, 5, *one_hot.shape[-2:])

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
    for epoch in range(args.epochs):
        kwargs = {}
        # Forward pair_weights only when the trainer accepts it. Newer
        # SegMapTrainer accepts the kwarg; older versions raise TypeError.
        # We probe once (via inspect) and cache the support flag.
        if pair_weights is not None:
            import inspect
            sig = inspect.signature(trainer.train_epoch)
            if "pair_weights" in sig.parameters:
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
            seg = epoch_metrics.get("seg", epoch_metrics.get("seg_loss", float("nan")))
            pose = epoch_metrics.get("pose", epoch_metrics.get("pose_loss", float("nan")))
            print(
                f"[train_segmap] epoch={epoch} loss={loss:.6f} seg={seg:.6f} pose={pose:.6f}",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    print(f"[train_segmap] training complete in {elapsed:.1f}s", flush=True)

    train_ckpt_path = output_dir / "segmap_train.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": cfg.model_dump() if hasattr(cfg, "model_dump") else dict(vars(cfg)),
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
        "history": history,
    }
    (output_dir / "segmap_train.json").write_text(json.dumps(summary, indent=2))
    print(f"[train_segmap] wrote {train_ckpt_path}, {inference_path}, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
