"""Lane J-NWC: train a neural weight codec on a corpus of renderer checkpoints.

Reference: arXiv 2510.11234 — "Neural Weight Compression for Language Models".
Trains a tiny VQ-VAE-style codec that maps each weight-tensor block →
codebook index. The corpus is the hundreds of small renderers we have already
trained (saved under experiments/results/).

Usage::

    python experiments/train_neural_weight_codec.py \
        --corpus-dir experiments/results \
        --output codec.pt \
        --num-steps 2000 \
        --device cpu

The output ``codec.pt`` is a torch.save dict::

    {
        "codec_state_dict": codec.state_dict(),
        "codec_config": dataclasses.asdict(codec.config),
        "training_loss_history": [...],
        "corpus_size_blocks": int,
    }

Then use ``tac.renderer_export.export_neural_compressed_checkpoint`` to apply
the trained codec to a final renderer.bin candidate.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

# Make ``tac`` importable when run directly without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.neural_weight_codec import (
    WeightCodec,
    WeightCodecConfig,
    build_corpus_from_checkpoints,
    train_codec,
)


def _discover_checkpoints(corpus_dir: Path, max_files: int) -> list[Path]:
    """Walk corpus_dir for .pt checkpoints (sorted, capped)."""
    if not corpus_dir.is_dir():
        raise SystemExit(f"FATAL: corpus dir does not exist: {corpus_dir}")
    candidates = sorted(corpus_dir.rglob("*.pt"))
    # Filter out optimizer-state-only and tiny files (< 1KB are useless)
    keepers = []
    for p in candidates:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz < 1024:
            continue
        keepers.append(p)
        if len(keepers) >= max_files:
            break
    if not keepers:
        raise SystemExit(
            f"FATAL: no .pt files >=1KB found under {corpus_dir} — pass a different "
            f"--corpus-dir or relax --max-corpus-files."
        )
    return keepers


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane J-NWC neural weight codec trainer")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory of .pt checkpoints to use as the codec training corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the trained codec checkpoint (codec.pt).",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device. NOTE: CLAUDE.md FORBIDS silent MPS fallback. "
             "Operator must pass --device explicitly.",
    )
    parser.add_argument("--block-size", type=int, default=16, help="Codec block size.")
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max-corpus-files", type=int, default=200)
    parser.add_argument("--max-blocks-per-ckpt", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    # Per CLAUDE.md: deterministic CUDA-or-explicit-CPU; no silent fallback.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() is False.\n"
            "  CLAUDE.md FORBIDS silent fallback to MPS/CPU. Pass --device cpu explicitly\n"
            "  to acknowledge that the codec training is byte-deterministic and CPU-OK."
        )

    print(f"[nwc-train-cli] discovering corpus under {args.corpus_dir}")
    ckpts = _discover_checkpoints(args.corpus_dir, args.max_corpus_files)
    print(f"[nwc-train-cli] using {len(ckpts)} checkpoints (cap={args.max_corpus_files})")

    print(f"[nwc-train-cli] building corpus at block_size={args.block_size}")
    corpus = build_corpus_from_checkpoints(
        [str(p) for p in ckpts],
        block_size=args.block_size,
        max_blocks_per_ckpt=args.max_blocks_per_ckpt,
    )
    print(f"[nwc-train-cli] corpus shape={tuple(corpus.shape)}")

    cfg = WeightCodecConfig(
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
    )
    codec = WeightCodec(cfg)
    print(f"[nwc-train-cli] codec config: {cfg}")
    print(
        f"[nwc-train-cli] codec param count: "
        f"{sum(p.numel() for p in codec.parameters()):,}"
    )

    codec, losses = train_codec(
        corpus,
        codec=codec,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "codec_state_dict": codec.cpu().state_dict(),
        "codec_config": dataclasses.asdict(cfg),
        "training_loss_history": losses,
        "corpus_size_blocks": int(corpus.shape[0]),
        "n_checkpoints": len(ckpts),
    }
    torch.save(payload, args.output)
    sz = args.output.stat().st_size
    print(
        f"[nwc-train-cli] wrote {args.output} ({sz:,} bytes) "
        f"final_loss={losses[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
