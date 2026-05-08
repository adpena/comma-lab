"""CLI to dump PoseNet/SegNet per-layer activations for a frame pair.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE": this script writes a `[diagnostic-not-score]` introspection record;
it never produces a contest score. The CUDA path is wired (``--device cuda``)
but the operator dispatches it on a Linux x86_64 GPU instance via the
canonical remote bootstrap; this entry point is the same for CPU and CUDA so
the two records are byte-comparable.

Usage::

    .venv/bin/python experiments/dump_scorer_activations.py \\
        --upstream-dir upstream \\
        --frame-pair-idx 0 \\
        --device cpu \\
        --output-dir experiments/results/scorer_introspection_demo_20260508 \\
        --capture-mode fingerprint

Inputs
------

The toolkit dumps internals of PoseNet and SegNet using their canonical
upstream weights. It does NOT need an archive — what it inspects is the
scorer architecture, not a candidate submission's payload. ``--archive`` is
accepted as documentation/metadata only (the SHA-256 is recorded in the
introspection metadata so the operator can later cross-reference which
candidate was being analyzed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.diagnostics.scorer_introspection import (  # noqa: E402
    ScorerIntrospector,
    list_attention_like_layers,
)
from tac.scorer import load_scorers  # noqa: E402


def _gt_video_to_pair_input(
    upstream_dir: Path, frame_pair_idx: int, device: torch.device
) -> torch.Tensor:
    """Load a real (B=1, T=2, H, W, 3) GT frame pair from upstream/videos/0.mkv.

    Falls back to a deterministic synthetic pair if the GT video is not
    present (CI/laptop-without-videos friendly). The synthetic pair is
    explicitly tagged as such in the metadata so downstream consumers know
    it isn't a real-frame-pair signal.
    """
    video_path = upstream_dir / "videos" / "0.mkv"
    if not video_path.exists():
        # Synthetic deterministic fallback. Emit a marker so the metadata
        # records the source as "synthetic".
        torch.manual_seed(42 + frame_pair_idx)
        # camera_size = (1164, 874); we keep this exact shape to match contest.
        x = torch.randint(0, 256, (1, 2, 874, 1164, 3), dtype=torch.uint8)
        return x.float().to(device), "synthetic"

    # Use upstream's AVVideoDataset so we read frames the same way the contest
    # scorer does, then slice the requested pair.
    sys_path_added = str(upstream_dir) not in sys.path
    if sys_path_added:
        sys.path.insert(0, str(upstream_dir))
    try:
        from frame_utils import AVVideoDataset, seq_len, camera_size

        ds = AVVideoDataset(
            ["0.mkv"],
            data_dir=upstream_dir / "videos",
            batch_size=1,
            device=device,
        )
        ds.prepare_data()
        # Iterate to the requested pair.
        target = frame_pair_idx
        for _, _, batch in ds:
            target -= 1
            if target < 0:
                # batch is (B, T, H, W, 3) uint8 already.
                return batch.float().to(device), str(video_path)
        raise IndexError(
            f"frame_pair_idx={frame_pair_idx} exceeds available pairs in {video_path}"
        )
    finally:
        if sys_path_added and str(upstream_dir) in sys.path:
            sys.path.remove(str(upstream_dir))


def _archive_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO_ROOT / "upstream",
        help="Path to the pinned upstream snapshot (where modules.py lives).",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional candidate archive path (SHA-256 recorded in metadata only).",
    )
    parser.add_argument(
        "--frame-pair-idx",
        type=int,
        default=0,
        help="Index of the frame pair to feed through the scorers.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for the forward pass. CUDA path runs on Linux x86_64 GPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write {posenet,segnet}_record.pt and summary.json.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["fingerprint", "full"],
        default="fingerprint",
        help="'fingerprint' is the default and bounds memory; 'full' keeps tensors.",
    )
    parser.add_argument(
        "--full-threshold-elements",
        type=int,
        default=1 << 20,
        help="In full mode, layers with more elements fall back to fingerprint.",
    )
    parser.add_argument(
        "--scorer",
        choices=["both", "posenet", "segnet"],
        default="both",
        help="Which scorer(s) to dump.",
    )
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        # Per CLAUDE.md `forbidden_default_to_convenience_trap`, do NOT silently
        # fall back to CPU when CUDA was requested. Explicit hard failure.
        raise RuntimeError(
            "--device cuda requested but torch.cuda.is_available() is False. "
            "Run on a Linux x86_64 GPU instance per CLAUDE.md 'Submission "
            "auth eval — BOTH CPU AND CUDA' or rerun with --device cpu."
        )

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load scorers from upstream-pinned weights. We never modify upstream/.
    posenet_path = args.upstream_dir / "models" / "posenet.safetensors"
    segnet_path = args.upstream_dir / "models" / "segnet.safetensors"
    posenet, segnet = load_scorers(
        posenet_path=posenet_path,
        segnet_path=segnet_path,
        device=device,
        upstream_dir=args.upstream_dir,
    )

    # Load the frame pair.
    pair, source = _gt_video_to_pair_input(args.upstream_dir, args.frame_pair_idx, device)
    # pair: (1, 2, H, W, 3) float on device. Reshape to the (B, T, C, H, W)
    # the scorers' preprocess_input expects.
    import einops

    pair_chw = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    if pair_chw.shape[0] != 1:
        pair_chw = pair_chw[:1]

    # Preprocess inputs (these calls are upstream code; we don't modify it).
    posenet_in = posenet.preprocess_input(pair_chw)
    segnet_in = segnet.preprocess_input(pair_chw)

    # Common metadata block recorded into both records.
    common_meta: dict[str, Any] = {
        "tag": "[diagnostic-not-score]",
        "device": args.device,
        "frame_pair_idx": args.frame_pair_idx,
        "source": source,
        "capture_mode": args.capture_mode,
        "archive_path": str(args.archive) if args.archive else None,
        "archive_sha256": _archive_sha256(args.archive),
        "upstream_dir": str(args.upstream_dir),
        "timestamp": int(time.time()),
        "torch_version": torch.__version__,
    }

    summary: dict[str, Any] = {"common": common_meta, "scorers": {}}

    if args.scorer in ("both", "posenet"):
        attn_layers = list_attention_like_layers(posenet)
        insp = ScorerIntrospector(
            posenet,
            capture_mode=args.capture_mode,
            full_threshold_elements=args.full_threshold_elements,
        )
        with insp.session():
            record = insp.capture(posenet_in)
        record.metadata.update(common_meta)
        record.metadata["attention_like_layers"] = [
            {"name": n, "type": t} for n, t in attn_layers
        ]
        out_path = args.output_dir / "posenet_record.pt"
        record.to_disk(out_path)
        summary["scorers"]["posenet"] = {
            "record_path": str(out_path),
            "num_layers_recorded": len(record.layers),
            "num_attention_like_layers": len(attn_layers),
            "input_shape": list(posenet_in.shape),
        }
        print(
            f"[introspect] PoseNet: {len(record.layers)} layers, "
            f"{len(attn_layers)} attention-like, written to {out_path}"
        )

    if args.scorer in ("both", "segnet"):
        attn_layers = list_attention_like_layers(segnet)
        insp = ScorerIntrospector(
            segnet,
            capture_mode=args.capture_mode,
            full_threshold_elements=args.full_threshold_elements,
        )
        with insp.session():
            record = insp.capture(segnet_in)
        record.metadata.update(common_meta)
        record.metadata["attention_like_layers"] = [
            {"name": n, "type": t} for n, t in attn_layers
        ]
        out_path = args.output_dir / "segnet_record.pt"
        record.to_disk(out_path)
        summary["scorers"]["segnet"] = {
            "record_path": str(out_path),
            "num_layers_recorded": len(record.layers),
            "num_attention_like_layers": len(attn_layers),
            "input_shape": list(segnet_in.shape),
        }
        print(
            f"[introspect] SegNet: {len(record.layers)} layers, "
            f"{len(attn_layers)} attention-like, written to {out_path}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[introspect] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
