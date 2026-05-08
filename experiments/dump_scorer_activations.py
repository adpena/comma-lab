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

For CPU/CUDA xray work, prefer ``--shared-input-tensor`` with an artifact
written by ``tools/probe_eval_loader_drift.py --save-shared-input-dir``. That
path loads already-decoded RGB bytes with custody and avoids the invalid
``AVVideoDataset(device='cuda')`` trap.
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

SHARED_INPUT_TENSOR_SCHEMA = "eval_loader_shared_input_tensor.v1"
SHARED_INPUT_TENSOR_ROLE = "raw_rgb_uint8_before_posenet_segnet"
NON_PROMOTABLE_FIELDS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
    "dispatch_attempted",
)

from tac.diagnostics.scorer_introspection import (  # noqa: E402
    ScorerIntrospector,
    list_attention_like_layers,
)
from tac.scorer import load_scorers  # noqa: E402


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def _torch_load_tensor_artifact(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch without weights_only
        return torch.load(path, map_location="cpu")


def _load_shared_input_tensor(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a non-promotable shared RGB input tensor with strict custody."""
    payload = _torch_load_tensor_artifact(path)
    if not isinstance(payload, dict):
        raise ValueError("shared input tensor artifact must be a dict")
    if payload.get("schema") != SHARED_INPUT_TENSOR_SCHEMA:
        raise ValueError(
            "shared input tensor artifact schema must be "
            f"{SHARED_INPUT_TENSOR_SCHEMA}"
        )
    if payload.get("tensor_role") != SHARED_INPUT_TENSOR_ROLE:
        raise ValueError(
            "shared input tensor artifact tensor_role must be "
            f"{SHARED_INPUT_TENSOR_ROLE}"
        )
    for field in NON_PROMOTABLE_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"shared input tensor artifact requires {field}=false")
    tensor = payload.get("tensor")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("shared input tensor artifact missing tensor")
    if tensor.ndim != 5 or tensor.shape[-1] != 3 or tensor.shape[1] < 2:
        raise ValueError(
            "shared input tensor must have shape (B, T>=2, H, W, 3); got "
            f"{list(tensor.shape)}"
        )
    custody = payload.get("tensor_custody")
    if not isinstance(custody, dict):
        raise ValueError("shared input tensor artifact missing tensor_custody")
    actual_sha = _tensor_sha256(tensor)
    if custody.get("sha256") != actual_sha:
        raise ValueError("shared input tensor custody sha256 mismatch")
    if custody.get("dtype") != str(tensor.detach().to(device="cpu").dtype):
        raise ValueError("shared input tensor custody dtype mismatch")
    if custody.get("shape") != list(tensor.shape):
        raise ValueError("shared input tensor custody shape mismatch")
    metadata = {key: value for key, value in payload.items() if key != "tensor"}
    metadata["artifact_path"] = str(path)
    metadata["artifact_sha256"] = _archive_sha256(path)
    metadata["loaded_by"] = "experiments/dump_scorer_activations.py"
    metadata["shared_input_contract_valid"] = True
    return tensor.float().to(device), metadata


def _gt_video_to_pair_input(
    upstream_dir: Path, frame_pair_idx: int, device: torch.device
) -> tuple[torch.Tensor, str]:
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

    if device.type == "cuda":
        raise RuntimeError(
            "real-video CUDA scorer introspection must not instantiate "
            "AVVideoDataset(device='cuda'); use tools/probe_eval_loader_drift.py "
            "--run-forward-cells with tensor custody for CUDA/DALI xray, or "
            "provide an already-decoded shared input tensor in a future loader-cell mode"
        )

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
        "--shared-input-tensor",
        type=Path,
        default=None,
        help=(
            "Optional eval_loader_shared_input_tensor.v1 artifact from "
            "tools/probe_eval_loader_drift.py. When set, this decoded RGB "
            "tensor replaces GT video loading for CPU/CUDA shared-input xray."
        ),
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
    shared_input_meta: dict[str, Any] | None = None
    if args.shared_input_tensor is not None:
        pair, shared_input_meta = _load_shared_input_tensor(
            args.shared_input_tensor, device
        )
        source = f"shared_input_tensor:{args.shared_input_tensor}"
    else:
        pair, source = _gt_video_to_pair_input(
            args.upstream_dir, args.frame_pair_idx, device
        )
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
        "input_source_kind": (
            "shared_input_tensor"
            if args.shared_input_tensor is not None
            else "upstream_gt_video_or_synthetic"
        ),
        "shared_input_tensor": shared_input_meta,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "timestamp": int(time.time()),
        "torch_version": torch.__version__,
    }

    summary: dict[str, Any] = {
        "common": common_meta,
        "scorers": {},
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }

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
