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
    train_codec,
)
from tac.neural_weight_corpus import (
    build_corpus_from_manifest,
    build_corpus_manifest_from_dir,
    canonical_manifest_json,
    load_corpus_manifest,
    sha256_file,
    write_corpus_manifest,
)


def _default_manifest_out(output: Path) -> Path:
    return output.with_suffix(".corpus_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lane J-NWC neural weight codec trainer")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory of .pt checkpoints to use as the codec training corpus.",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=None,
        help=(
            "Existing deterministic corpus manifest JSON to replay. If omitted, "
            "--corpus-dir is discovered and a manifest is emitted."
        ),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help=(
            "Path to write the corpus manifest used for this run. Defaults to "
            "<output>.corpus_manifest.json."
        ),
    )
    parser.add_argument(
        "--corpus-replay-root",
        type=Path,
        default=None,
        help=(
            "Root directory used to resolve selected manifest relative_path entries "
            "when replaying --corpus-manifest."
        ),
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
    parser.add_argument(
        "--min-checkpoint-bytes",
        type=int,
        default=1024,
        help="Exclude checkpoint files smaller than this many bytes during discovery.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    if args.corpus_manifest is None and args.corpus_dir is None:
        raise SystemExit("FATAL: pass either --corpus-dir or --corpus-manifest")
    if args.corpus_manifest is None and args.corpus_replay_root is not None:
        raise SystemExit("FATAL: --corpus-replay-root requires --corpus-manifest")
    if args.corpus_manifest is not None and args.corpus_dir is not None:
        print(
            "[nwc-train-cli] NOTE: --corpus-manifest was provided; "
            "--corpus-dir is recorded only by that manifest."
        )
    if args.corpus_replay_root is not None and not args.corpus_replay_root.is_dir():
        raise SystemExit(
            "FATAL: --corpus-replay-root is not a directory: "
            f"{args.corpus_replay_root}"
        )

    # Per CLAUDE.md: deterministic CUDA-or-explicit-CPU; no silent fallback.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but torch.cuda.is_available() is False.\n"
            "  CLAUDE.md FORBIDS silent fallback to MPS/CPU. Pass --device cpu explicitly\n"
            "  to acknowledge that the codec training is byte-deterministic and CPU-OK."
        )

    manifest_out = args.manifest_out or _default_manifest_out(args.output)
    if args.corpus_manifest is not None:
        print(f"[nwc-train-cli] replaying corpus manifest {args.corpus_manifest}")
        manifest = load_corpus_manifest(args.corpus_manifest)
        if args.corpus_replay_root is not None:
            print(f"[nwc-train-cli] replay root {args.corpus_replay_root}")
    else:
        print(f"[nwc-train-cli] discovering corpus under {args.corpus_dir}")
        manifest = build_corpus_manifest_from_dir(
            args.corpus_dir,
            block_size=args.block_size,
            max_files=args.max_corpus_files,
            max_blocks_per_ckpt=args.max_blocks_per_ckpt,
            min_checkpoint_bytes=args.min_checkpoint_bytes,
        )

    print(
        "[nwc-train-cli] corpus manifest totals: "
        f"files={manifest['totals']['selected_files']}/"
        f"{manifest['totals']['discovered_files']} "
        f"tensors={manifest['totals']['selected_tensors']} "
        f"blocks={manifest['totals']['selected_blocks']}"
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    if args.corpus_manifest is not None:
        source = args.corpus_manifest.resolve(strict=True)
        destination = manifest_out.resolve(strict=False)
        if destination != source:
            manifest_out.write_bytes(args.corpus_manifest.read_bytes())
    else:
        write_corpus_manifest(manifest, manifest_out)
    manifest_sha256 = sha256_file(manifest_out)
    print(
        f"[nwc-train-cli] wrote corpus manifest {manifest_out} "
        f"sha256={manifest_sha256}"
    )

    print(f"[nwc-train-cli] building manifest-backed corpus at block_size={args.block_size}")
    corpus = build_corpus_from_manifest(
        manifest,
        replay_root=args.corpus_replay_root,
    )
    print(f"[nwc-train-cli] corpus shape={tuple(corpus.shape)}")
    if int(corpus.shape[0]) != int(manifest["totals"]["selected_blocks"]):
        raise SystemExit(
            "FATAL: corpus block count does not match manifest "
            f"({corpus.shape[0]} vs {manifest['totals']['selected_blocks']})"
        )

    cfg = WeightCodecConfig(
        block_size=args.block_size,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        hidden=args.hidden,
    )
    # Seed before codec construction; codebook initialization is random and
    # must be part of the advertised deterministic seed contract.
    torch.manual_seed(int(args.seed))
    if args.device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
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
        "n_checkpoints": int(manifest["totals"]["selected_files"]),
        "corpus_manifest_path": str(manifest_out),
        "corpus_manifest_sha256": manifest_sha256,
        "corpus_manifest_totals": dict(manifest["totals"]),
        "corpus_manifest_json": canonical_manifest_json(manifest),
        "corpus_replay_root": (
            None if args.corpus_replay_root is None else str(args.corpus_replay_root)
        ),
        "seed": int(args.seed),
    }
    torch.save(payload, args.output)
    sz = args.output.stat().st_size
    print(
        f"[nwc-train-cli] wrote {args.output} ({sz:,} bytes) "
        f"final_loss={losses[-1]:.6f}"
    )


if __name__ == "__main__":
    main()
