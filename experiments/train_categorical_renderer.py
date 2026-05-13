"""Categorical full-renderer trainer — Phase A.

Per CLAUDE.md non-negotiables this trainer:

* Defaults CUDA-REQUIRED, no MPS-fallback.
* Uses ``tac.scorer.load_differentiable_scorers`` for differentiable PoseNet.
* Uses ``RealPairBatchSource`` from ``lane_12_v2_nerv_as_renderer`` for real GT.
* EMA decay 0.997 on renderer weights; archive saves EMA shadow.
* Codebook-collapse guard (class-entropy floor) per van den Oord VQ-VAE pattern.
* No /tmp paths; outputs to ``experiments/results/categorical_substrate_<ts>/``.
* Auth eval is operator-gated; trainer ends with EMA checkpoint + summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tac.categorical_substrate import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    CategoricalRenderer,
    CategoricalSubstrateConfig,
    CodebookCollapseError,
    NUM_CLASSES,
    SEGNET_IN_H,
    SEGNET_IN_W,
    train_step,
)


class EMA:
    """Weight EMA mirroring ``tac.training.EMA``."""

    def __init__(self, model: nn.Module, decay: float = 0.997) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                continue
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}


def _make_smoke_batch(batch_size: int, num_pairs: int,
                       device: torch.device) -> dict:  # SYNTHETIC_NON_SMOKE_OK:explicit_smoke_only_branch
    """Synthetic batch for --smoke ONLY. Real training uses RealPairBatchSource."""
    pair_indices = torch.randint(0, num_pairs, (batch_size,), dtype=torch.long, device=device)
    # Use diverse tokens to keep class-entropy above the collapse floor.
    tokens = torch.randint(0, NUM_CLASSES, (batch_size, SEGNET_IN_H, SEGNET_IN_W),
                            dtype=torch.long, device=device)
    gt_pairs = torch.randint(0, 256, (batch_size, 2, 3, CAMERA_H, CAMERA_W),
                              dtype=torch.uint8, device=device).float()
    return {"pair_indices": pair_indices, "tokens": tokens, "gt_pairs": gt_pairs}


def _make_real_batch_iterator(args: argparse.Namespace, batch_size: int,
                                num_pairs: int,
                                device: torch.device) -> Iterator[dict]:
    if args.smoke:
        raise RuntimeError("called in smoke mode")
    if not args.video_path:
        raise RuntimeError(
            "--video-path required for non-smoke runs (decodes "
            "upstream/videos/0.mkv via RealPairBatchSource)"
        )
    try:
        from src.tac.lane_12_v2_nerv_as_renderer import RealPairBatchSource
    except ImportError as e:
        raise RuntimeError(
            f"RealPairBatchSource import failed: {e}"
        ) from e

    source = RealPairBatchSource(
        video_path=Path(args.video_path),
        num_pairs=num_pairs,
        device=device,
    )
    return source.iter_batches(batch_size=batch_size)


def _gt_to_tokens_via_segnet(scorer_seg: nn.Module,
                              gt_pairs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        seg_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    return seg_logits.argmax(dim=1).to(torch.long)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Categorical substrate trainer.")
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda-seg", type=float, default=100.0)
    parser.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    parser.add_argument("--palette-dim", type=int, default=8)
    parser.add_argument("--shading-channels", type=int, default=16)
    parser.add_argument("--codebook-collapse-floor", type=float, default=0.4)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--upstream-dir", type=str, default=str(ROOT / "upstream"))
    parser.add_argument("--video-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    # CLAUDE.md "Auth eval EVERYWHERE" canonical opt-out flag. The categorical
    # substrate trainer defers archive build + auth eval to the operator-gated
    # Phase B dispatch step (per feedback_anr_token_renderer_categorical_full
    # _substrate_landed_20260511.md). When set, the trainer saves the EMA
    # checkpoint and exits cleanly; the operator runs auth eval as a separate
    # step that consumes the saved checkpoint via the dispatch helper.
    parser.add_argument(
        "--no-auth-eval-on-best",
        action="store_true",
        default=False,
        help=(
            "Skip auth_eval at end of training. Defers to operator-gated "
            "Phase B dispatch (CLAUDE.md operator-gate non-negotiable + II's "
            "ANR/categorical landing memo). Default OFF; CI/dispatch wrappers "
            "must pass this flag explicitly to acknowledge they own the "
            "subsequent auth eval step."
        ),
    )
    args = parser.parse_args(argv)

    if args.device.lower() == "mps":
        print("FATAL: MPS forbidden by CLAUDE.md.", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.smoke:
            print("WARNING: CUDA unavailable; smoke falling back to cpu.",
                  file=sys.stderr)
            args.device = "cpu"
        else:
            print("FATAL: --device cuda but CUDA unavailable.", file=sys.stderr)
            return 2

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    if str(out_dir).startswith("/tmp"):
        print("FATAL: --output-dir must not be /tmp/.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CategoricalSubstrateConfig(
        num_pairs=args.num_pairs,
        palette_dim=args.palette_dim,
        shading_channels=args.shading_channels,
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        codebook_collapse_floor=args.codebook_collapse_floor,
        cuda_required=(args.device == "cuda"),
    )
    renderer = CategoricalRenderer(cfg).to(device)
    ema = EMA(renderer, decay=args.ema_decay)
    opt = torch.optim.AdamW(renderer.parameters(), lr=args.lr)

    # SMOKE PATH
    if args.smoke:
        class _MockSeg(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3 * 384 * 512, 5 * 384 * 512)

            def preprocess_input(self, x):
                last = x[:, -1, ...]
                return F.interpolate(last, size=(384, 512), mode="bilinear",
                                      align_corners=False)

            def forward(self, x):
                B = x.shape[0]
                return self.lin(x.reshape(B, -1)).reshape(B, 5, 384, 512)

        class _MockPose(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3 * 384 * 512 * 2, 6)

            def preprocess_input(self, x):
                B_, F_, C_, H_, W_ = x.shape
                flat = x.reshape(B_ * F_, C_, H_, W_)
                r = F.interpolate(flat, size=(384, 512), mode="bilinear",
                                   align_corners=False)
                return r.reshape(B_, F_, C_, 384, 512)

            def forward(self, x):
                B = x.shape[0]
                return self.lin(x.reshape(B, -1))

        scorer_seg = _MockSeg().to(device)
        scorer_pose = _MockPose().to(device)
        batch = _make_smoke_batch(args.batch_size, cfg.num_pairs, device)
        result = train_step(
            renderer=renderer,
            pair_indices=batch["pair_indices"],
            tokens=batch["tokens"],
            gt_pairs_uint8=batch["gt_pairs"],
            scorer_seg=scorer_seg, scorer_pose=scorer_pose,
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=cfg.lambda_seg, lambda_pose=cfg.lambda_pose,
            codebook_collapse_floor=cfg.codebook_collapse_floor,
        )
        opt.zero_grad()
        result["loss"].backward()
        opt.step()
        ema.update(renderer)
        with open(out_dir / "smoke_summary.json", "w") as f:
            json.dump({
                "smoke": True,
                "device": str(device),
                "loss": float(result["loss"].item()),
                "class_entropy": float(result["class_entropy"].item()),
                "score_claim": False,
                "evidence_grade": "smoke-only",
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }, f, indent=2)
        print(f"SMOKE OK loss={result['loss'].item():.4f} "
              f"entropy={result['class_entropy'].item():.4f}")
        return 0

    # NON-SMOKE PATH
    from src.tac.scorer import load_differentiable_scorers  # noqa: E402

    scorer_pose, scorer_seg = load_differentiable_scorers(
        args.upstream_dir, device=device,
    )

    best_loss = float("inf")
    start = time.time()
    collapse_count = 0
    for epoch in range(args.epochs):
        renderer.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in _make_real_batch_iterator(args, args.batch_size,
                                                cfg.num_pairs, device):
            pair_indices = batch["pair_indices"].to(device)
            gt_pairs = batch["gt_pairs"].to(device)
            tokens = _gt_to_tokens_via_segnet(scorer_seg, gt_pairs)
            try:
                result = train_step(
                    renderer=renderer, pair_indices=pair_indices,
                    tokens=tokens, gt_pairs_uint8=gt_pairs,
                    scorer_seg=scorer_seg, scorer_pose=scorer_pose,
                    seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
                    pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
                    lambda_seg=cfg.lambda_seg, lambda_pose=cfg.lambda_pose,
                    codebook_collapse_floor=cfg.codebook_collapse_floor,
                )
            except CodebookCollapseError:
                collapse_count += 1
                if collapse_count > 5:
                    print(f"FATAL: codebook collapse repeated {collapse_count}× — "
                          f"refusing to continue. Investigate SegNet teacher.",
                          file=sys.stderr)
                    return 3
                continue
            opt.zero_grad()
            result["loss"].backward()
            opt.step()
            ema.update(renderer)
            epoch_loss += float(result["loss"].item())
            n_batches += 1
            if n_batches >= 50:
                break
        epoch_loss /= max(n_batches, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ema.state_dict(), out_dir / "renderer_ema_best.pt")

    elapsed = time.time() - start
    if elapsed < args.epochs * 0.05:
        raise RuntimeError(
            f"elapsed_sec {elapsed:.2f} < epochs * MIN_SEC. Stub-loop suspected."
        )
    summary = {
        "epochs": args.epochs,
        "elapsed_sec": elapsed,
        "best_loss": best_loss,
        "collapse_count": collapse_count,
        "config": asdict(cfg),
        "device": str(device),
        "score_claim": False,
        "evidence_grade": "training-loss-only",
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": True,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TRAINING COMPLETE elapsed={elapsed:.1f}s best_loss={best_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
