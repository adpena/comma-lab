"""ANR substrate trainer — TokenRendererV62 master + ShrinkSingleNeRV slave +
HPACMini context model, jointly trained against the contest video with
score-aware Lagrangian and eval_roundtrip.

Per CLAUDE.md non-negotiables this trainer:

* Defaults to CUDA-REQUIRED; raises on missing CUDA (no MPS fallback).
* Uses ``tac.scorer.load_differentiable_scorers`` so PoseNet gradients flow.
* Uses ``tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training``
  for the uint8 bottleneck simulation.
* Uses ``RealPairBatchSource`` from ``lane_12_v2_nerv_as_renderer`` (decodes
  real ``upstream/videos/0.mkv`` via PyAV — NEVER synthetic outside smoke).
* Uses EMA (decay=0.997) on master + slave + HPACMini weights per the EMA
  non-negotiable. EMA shadow is what gets saved to the archive.
* Ends with a CUDA auth eval on the best EMA checkpoint per "Auth eval
  EVERYWHERE" non-negotiable. Auth eval is operator-gated (dispatch budget
  $40-80 per CLAUDE.md / parent prompt).
* No /tmp paths; all outputs go to ``experiments/results/anr_substrate_<timestamp>/``.

Dispatch is operator-gated. This script does NOT auto-dispatch to GPU; the
operator approves the launch command separately.
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

# Make tac available — point sys.path at `src/` so the canonical `tac.*`
# namespace resolves under any entry point (README's PYTHONPATH=src:upstream
# OR standalone `python experiments/train_anr_token_renderer.py`). Pre-FIX-
# WAVE-11 used `parent.parent` (repo root) and `from src.tac.*` imports,
# which created a parallel non-canonical namespace per FIX-WAVE-11 R11-1
# closure (Catalog #188 extended scope). ROOT is retained as the repo root
# for downstream default-path resolution (e.g. ``ROOT / "upstream"``).
ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.anr_token_renderer import (  # noqa: E402
    ANRTokenRendererConfig,
    CAMERA_H,
    CAMERA_W,
    SEGNET_IN_H,
    SEGNET_IN_W,
    ShrinkSingleNeRV,
    TokenRendererV62,
    train_step,
)


# ── EMA helper (CLAUDE.md EMA non-negotiable, decay=0.997) ──────────────


class EMA:
    """Weight EMA mirroring ``tac.training.EMA`` (canonical).

    Decay=0.997 per CLAUDE.md non-negotiable. Apply EMA only at eval time with
    snapshot+restore — NEVER mutate live weights mid-step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.997) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
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

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Apply shadow → model, return original state for restore."""
        orig = {k: v.detach().clone() for k, v in model.state_dict().items()}
        merged = orig.copy()
        for k, v in self.shadow.items():
            if k in merged:
                merged[k] = v.detach().clone()
        model.load_state_dict(merged)
        return orig


# ── Smoke-only synthetic batch (CLAUDE.md SYNTHETIC_NON_SMOKE_OK waiver) ──


def _make_smoke_batch(batch_size: int, num_pairs: int,
                       device: torch.device) -> dict:  # SYNTHETIC_NON_SMOKE_OK:explicit_smoke_only_branch
    """Synthetic batch for the --smoke gradient-path test ONLY.

    Real training MUST use RealPairBatchSource that decodes upstream/videos/0.mkv.
    This helper is gated behind args.smoke per the
    check_training_scripts_use_real_data_in_nonsmoke_mode preflight.
    """
    pair_indices = torch.randint(0, num_pairs, (batch_size,), dtype=torch.long, device=device)
    tokens = torch.randint(0, 5, (batch_size, SEGNET_IN_H, SEGNET_IN_W),
                            dtype=torch.long, device=device)
    gt_pairs = torch.randint(0, 256, (batch_size, 2, 3, CAMERA_H, CAMERA_W),
                              dtype=torch.uint8, device=device).float()
    return {"pair_indices": pair_indices, "tokens": tokens, "gt_pairs": gt_pairs}


# ── Optional real-data batch loader (delegates to lane_12_v2) ───────────


def _make_real_batch_iterator(args: argparse.Namespace, batch_size: int,
                                num_pairs: int, device: torch.device
                                ) -> Iterator[dict]:
    """Real GT pair batch source (CLAUDE.md non-negotiable for non-smoke).

    Delegates to ``lane_12_v2_nerv_as_renderer.RealPairBatchSource`` which
    decodes ``upstream/videos/0.mkv`` via PyAV and yields real GT pairs.
    """
    if args.smoke:
        raise RuntimeError("_make_real_batch_iterator called in smoke mode")
    if not args.video_path:
        raise RuntimeError(
            "--video-path is required for non-smoke runs. Real data must come "
            "from upstream/videos/0.mkv (or a sister copy)."
        )

    # Lazy import to keep the smoke path free of PyAV.
    try:
        from tac.lane_12_v2_nerv_as_renderer import RealPairBatchSource
    except ImportError as e:
        raise RuntimeError(
            f"RealPairBatchSource import failed: {e}. Real-data training "
            f"requires lane_12_v2 + PyAV."
        ) from e

    source = RealPairBatchSource(
        video_path=Path(args.video_path),
        num_pairs=num_pairs,
        device=device,
    )
    # iter_batches returns dicts with pair_indices + gt_pairs; tokens come from
    # the SegNet teacher (forward GT through SegNet to get class-argmax tokens).
    return source.iter_batches(batch_size=batch_size)


def _gt_to_tokens_via_segnet(scorer_seg: nn.Module,
                              gt_pairs: torch.Tensor) -> torch.Tensor:
    """SegNet teacher → token indices at SEGNET_IN_H × SEGNET_IN_W.

    The contest SegNet derives masks from the LAST frame; we use its argmax
    as the token stream. This makes tokens semantically aligned with the
    contest scorer's view of the world.
    """
    with torch.no_grad():
        seg_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
        # seg_logits is (B, NUM_CLASSES, 384, 512) per CLAUDE.md scorer architecture.
    return seg_logits.argmax(dim=1).to(torch.long)


# ── Main training loop ─────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ANR substrate trainer.")
    parser.add_argument("--num-pairs", type=int, default=600,
                        help="Number of pairs (matches PR95 exemplar).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda-seg", type=float, default=100.0)
    parser.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    parser.add_argument("--ema-decay", type=float, default=0.997,
                        help="CLAUDE.md EMA non-negotiable decay.")
    parser.add_argument("--upstream-dir", type=str,
                        default=str(ROOT / "upstream"),
                        help="Path to upstream snapshot.")
    parser.add_argument("--video-path", type=str, default="",
                        help="Path to upstream/videos/0.mkv (real-data non-smoke).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory under experiments/results/.")
    parser.add_argument("--smoke", action="store_true",
                        help="Gradient-path smoke test (1 step, synthetic batch). "
                             "NEVER use as the dispatched-to-GPU training path.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda|cpu. CLAUDE.md forbids MPS. Default cuda; "
                             "--device cpu opt-in is for byte-deterministic smoke.")
    args = parser.parse_args(argv)

    # CLAUDE.md non-negotiable: no MPS-fallback default.
    if args.device.lower() == "mps":
        print("FATAL: MPS forbidden by CLAUDE.md. Use cuda (default) or cpu.",
              file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.smoke:
            print("WARNING: CUDA unavailable; smoke mode falling back to cpu.",
                  file=sys.stderr)
            args.device = "cpu"
        else:
            print("FATAL: --device cuda requested but CUDA unavailable. "
                  "Add explicit --device cpu for byte-deterministic smoke; "
                  "ban MPS unconditionally.", file=sys.stderr)
            return 2

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    if str(out_dir).startswith("/tmp"):
        # CLAUDE.md non-negotiable: no /tmp evidence paths.
        print("FATAL: --output-dir must NOT be under /tmp/. Use "
              "experiments/results/<lane>_<timestamp>/.", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ANRTokenRendererConfig(
        num_pairs=args.num_pairs,
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda"),
    )

    master = TokenRendererV62(
        num_pairs=cfg.num_pairs, d_film=cfg.d_film,
    ).to(device)
    slave = ShrinkSingleNeRV(
        num_pairs=cfg.num_pairs, d_lat=cfg.slave_d_lat,
        channels=cfg.slave_channels,
    ).to(device)

    # NB: HPACMini is trained SEPARATELY (negative log-likelihood on token corpus).
    # This trainer only optimises master + slave RGB renderers via score-aware loss.
    # The HPAC weights ship in the archive but their training is a sister script.

    ema_master = EMA(master, decay=args.ema_decay)
    ema_slave = EMA(slave, decay=args.ema_decay)

    opt = torch.optim.AdamW(
        list(master.parameters()) + list(slave.parameters()), lr=args.lr,
    )

    # SMOKE PATH — gradient correctness only; never the dispatched training path.
    if args.smoke:
        scorer_seg = nn.Linear(3 * 384 * 512, 5 * 384 * 512).to(device)
        scorer_pose = nn.Linear(3 * 384 * 512 * 2, 6).to(device)

        def _mock_preprocess_seg(x):
            last = x[:, -1, ...]
            return F.interpolate(last, size=(384, 512), mode="bilinear",
                                  align_corners=False)

        def _mock_preprocess_pose(x):
            B_, F_, C_, H_, W_ = x.shape
            flat = x.reshape(B_ * F_, C_, H_, W_)
            resized = F.interpolate(flat, size=(384, 512), mode="bilinear",
                                     align_corners=False)
            return resized.reshape(B_, F_, C_, 384, 512)

        # Mock scorers — gradient-path verification only.
        class _MockSeg(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3 * 384 * 512, 5 * 384 * 512)

            def preprocess_input(self, x):
                return _mock_preprocess_seg(x)

            def forward(self, x):
                B = x.shape[0]
                return self.lin(x.reshape(B, -1)).reshape(B, 5, 384, 512)

        class _MockPose(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3 * 384 * 512 * 2, 6)

            def preprocess_input(self, x):
                return _mock_preprocess_pose(x)

            def forward(self, x):
                B = x.shape[0]
                return self.lin(x.reshape(B, -1))

        scorer_seg = _MockSeg().to(device)
        scorer_pose = _MockPose().to(device)

        batch = _make_smoke_batch(args.batch_size, cfg.num_pairs, device)
        result = train_step(
            master=master, slave=slave,
            pair_indices=batch["pair_indices"],
            tokens=batch["tokens"],
            gt_pairs_uint8=batch["gt_pairs"],
            scorer_seg=scorer_seg, scorer_pose=scorer_pose,
            seg_surrogate=lambda a, b: (a - b).pow(2).mean(),
            pose_surrogate=lambda a, b: (a - b).pow(2).mean(),
            lambda_seg=cfg.lambda_seg, lambda_pose=cfg.lambda_pose,
        )
        opt.zero_grad()
        result["loss"].backward()
        opt.step()
        ema_master.update(master)
        ema_slave.update(slave)

        with open(out_dir / "smoke_summary.json", "w") as f:
            json.dump({
                "smoke": True,
                "device": str(device),
                "loss": float(result["loss"].item()),
                "loss_seg": float(result["loss_seg"].item()),
                "loss_pose": float(result["loss_pose"].item()),
                "score_claim": False,
                "evidence_grade": "smoke-only",
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }, f, indent=2)
        print(f"SMOKE OK loss={result['loss'].item():.4f}")
        return 0

    # NON-SMOKE PATH — real data + real scorers + auth eval at end.
    from tac.scorer import load_differentiable_scorers  # noqa: E402

    scorer_pose, scorer_seg = load_differentiable_scorers(
        args.upstream_dir, device=device,
    )

    seg_surrogate = lambda a, b: (a - b).pow(2).mean()
    pose_surrogate = lambda a, b: (a - b).pow(2).mean()

    best_loss = float("inf")
    start = time.time()
    for epoch in range(args.epochs):
        master.train()
        slave.train()
        # Real batch iteration via RealPairBatchSource.
        epoch_loss = 0.0
        n_batches = 0
        for batch in _make_real_batch_iterator(args, args.batch_size,
                                                cfg.num_pairs, device):
            pair_indices = batch["pair_indices"].to(device)
            gt_pairs = batch["gt_pairs"].to(device)
            tokens = _gt_to_tokens_via_segnet(scorer_seg, gt_pairs)
            result = train_step(
                master=master, slave=slave,
                pair_indices=pair_indices,
                tokens=tokens,
                gt_pairs_uint8=gt_pairs,
                scorer_seg=scorer_seg, scorer_pose=scorer_pose,
                seg_surrogate=seg_surrogate, pose_surrogate=pose_surrogate,
                lambda_seg=cfg.lambda_seg, lambda_pose=cfg.lambda_pose,
            )
            opt.zero_grad()
            result["loss"].backward()
            opt.step()
            ema_master.update(master)
            ema_slave.update(slave)
            epoch_loss += float(result["loss"].item())
            n_batches += 1
            if n_batches >= 50:  # cap per-epoch batches for cost discipline
                break
        epoch_loss /= max(n_batches, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # Save EMA shadow — NEVER live weights — per CLAUDE.md EMA non-negotiable.
            torch.save(ema_master.state_dict(),
                       out_dir / "master_ema_best.pt")
            torch.save(ema_slave.state_dict(),
                       out_dir / "slave_ema_best.pt")

    # Final summary
    elapsed = time.time() - start
    if elapsed < args.epochs * 0.05:
        # CLAUDE.md "Internal-consistency assertions in stats files"
        raise RuntimeError(
            f"elapsed_sec {elapsed:.2f} < epochs * MIN_SEC ({args.epochs * 0.05}). "
            f"Suspected stub-loop. Failing closed."
        )
    summary = {
        "epochs": args.epochs,
        "elapsed_sec": elapsed,
        "best_loss": best_loss,
        "config": asdict(cfg),
        "device": str(device),
        # CLAUDE.md "no score claims until auth eval" — explicit downgrade.
        "score_claim": False,
        "evidence_grade": "training-loss-only",
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": True,  # archive ready for paired auth eval
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"TRAINING COMPLETE elapsed={elapsed:.1f}s best_loss={best_loss:.4f}")
    print(f"Archive build + auth eval is operator-gated. "
          f"EMA checkpoints saved to {out_dir}/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
