#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a local QMA9 vertical block-copy escape byte screen.

This is a planning-only prototype. It decodes a bounded PR81 QMA9 frame prefix,
re-encodes that raw mask with the current QMA9 model and with the local QMB1
vertical block-copy format, verifies decode parity, and writes a manifest. It
does not edit runtime files, invoke the scorer, or dispatch remote work.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
for root in (SRC_ROOT, EXPERIMENTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from profile_qma9_range_mask_bitstream import parse_split_constants
from tac.qma9_range_mask_contract import (
    ORIGINAL_VIDEO_BYTES,
    QMA9ContractError,
    analyze_qma9_vertical_block_copy_opportunities,
    decode_qma9_prefix_frames,
    decode_qma9_vertical_block_escape_mask,
    encode_qma9_mask,
    encode_qma9_vertical_block_escape_mask,
    parse_qma9_header,
    parse_qma9_vertical_block_escape_header,
    read_single_member_zip,
    sha256_bytes,
    split_qma9_pr81_payload,
)


TOOL = "experiments/build_qma9_vertical_block_escape_candidate.py"
SCHEMA = "qma9_vertical_block_escape_candidate_screen_v1"
DEFAULT_PR81_DIR = REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex"
DEFAULT_PR81_ARCHIVE = DEFAULT_PR81_DIR / "archive.zip"
DEFAULT_PR81_INFLATE = DEFAULT_PR81_DIR / "replay_submission/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/qma9_range_mask_deconstruction_20260503_codex/candidates"


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_vertical_block_escape_screen(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    candidate_id: str,
    frames: int,
    block_width: int,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, custody = read_single_member_zip(archive_path)
    split = split_qma9_pr81_payload(
        payload,
        range_mask_bytes=constants["RANGE_MASK_BYTES"],
        model_bytes=constants["SPLIT_MODEL_REORDERED_BYTES"],
        pose_bytes=constants["POSE_STREAM_BYTES"],
        router_bytes=constants["ROUTER_ACTION_BYTES"],
    )
    source_header = parse_qma9_header(split.range_mask)
    subset_frames = int(frames)
    if subset_frames <= 0:
        raise QMA9ContractError("frames must be positive")
    if subset_frames > source_header.frame_count:
        raise QMA9ContractError(f"frames {subset_frames} exceeds source frame count {source_header.frame_count}")
    block_width = int(block_width)
    if block_width <= 0:
        raise QMA9ContractError("block width must be positive")

    raw = decode_qma9_prefix_frames(split.range_mask, frame_count=subset_frames)
    baseline_payload = encode_qma9_mask(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
    )
    candidate_payload = encode_qma9_vertical_block_escape_mask(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
        block_width=block_width,
    )
    decoded_candidate = decode_qma9_vertical_block_escape_mask(candidate_payload)
    if decoded_candidate.data != raw:
        raise QMA9ContractError("QMB1 candidate decode parity failed")

    candidate_header = parse_qma9_vertical_block_escape_header(candidate_payload)
    opportunities = analyze_qma9_vertical_block_copy_opportunities(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
        block_width=block_width,
    )
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    raw_path = candidate_dir / "decoded_prefix_mask.raw"
    baseline_path = candidate_dir / "baseline_subset.qma9"
    candidate_path = candidate_dir / "candidate_subset.qmb1"
    raw_path.write_bytes(raw)
    baseline_path.write_bytes(baseline_payload)
    candidate_path.write_bytes(candidate_payload)

    baseline_bytes = len(baseline_payload)
    candidate_bytes = len(candidate_payload)
    delta_bytes = candidate_bytes - baseline_bytes
    full_linear_projection_bytes = round(candidate_bytes * source_header.frame_count / subset_frames)
    reference_delta_bytes = full_linear_projection_bytes - source_header.packed_bytes
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "candidate": "qma9_vertical_block_escape_qmb1",
        "evidence_grade": "empirical/planning_only",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "source_qma9_header": asdict(source_header),
        "candidate_qmb1_header": asdict(candidate_header),
        "subset": {
            "frames": subset_frames,
            "decoded_pixels": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "raw_path": _repo_rel(raw_path),
            "baseline_subset_qma9_path": _repo_rel(baseline_path),
            "candidate_subset_qmb1_path": _repo_rel(candidate_path),
            "baseline_qma9_bytes": baseline_bytes,
            "candidate_qmb1_bytes": candidate_bytes,
            "delta_bytes_vs_subset_qma9": delta_bytes,
            "rate_score_delta_if_subset_scaled": delta_bytes * source_header.frame_count / subset_frames * 25.0 / ORIGINAL_VIDEO_BYTES,
            "decode_parity": True,
        },
        "block_copy_opportunities": opportunities,
        "full_stream_linear_projection": {
            "projection": "candidate_subset_bytes * source_frame_count / subset_frames",
            "candidate_range_mask_bytes": full_linear_projection_bytes,
            "reference_pr81_range_mask_bytes": source_header.packed_bytes,
            "delta_bytes_vs_pr81_range_mask": reference_delta_bytes,
            "rate_score_delta_if_components_unchanged": reference_delta_bytes * 25.0 / ORIGINAL_VIDEO_BYTES,
            "dispatchable": False,
            "reason": "linear projection from a bounded prefix; requires full deterministic re-encode and runtime integration",
        },
        "decision": {
            "local_screen_negative": candidate_bytes >= baseline_bytes,
            "reason": (
                "QMB1 must beat the exact subset QMA9 re-encode before any full-stream or runtime work is justified."
            ),
        },
        "notes": [
            "QMB1 charges copy flags inside the range-mask payload and falls back to the current QMA9 pixel tree.",
            "Copied blocks intentionally do not update the fallback QMA9 adaptive model; the byte screen measures that tradeoff directly.",
            "This artifact is not a score/component claim and did not invoke CUDA or a scorer.",
        ],
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR81_ARCHIVE)
    parser.add_argument("--split-constants-py", type=Path, default=DEFAULT_PR81_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="qma9_vertical_block_escape_len16_prefix1")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--block-width", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_vertical_block_escape_screen(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        frames=args.frames,
        block_width=args.block_width,
    )
    candidate_dir = args.output_dir / args.candidate_id
    subset = manifest["subset"]
    projection = manifest["full_stream_linear_projection"]
    print(f"wrote {candidate_dir / 'manifest.json'}")
    print(
        "subset_bytes "
        f"baseline_qma9={subset['baseline_qma9_bytes']} "
        f"candidate_qmb1={subset['candidate_qmb1_bytes']} "
        f"delta={subset['delta_bytes_vs_subset_qma9']}"
    )
    print(
        "linear_projection "
        f"candidate_range_mask_bytes={projection['candidate_range_mask_bytes']} "
        f"delta_vs_pr81={projection['delta_bytes_vs_pr81_range_mask']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
