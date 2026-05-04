#!/usr/bin/env python3
"""Build a local QMA9 context-conditioned copy/run escape byte screen.

This is a planning-only prototype. It decodes a bounded PR81 QMA9 frame
prefix, compares exact subset QMA9 re-encode bytes against a local ``QMC1``
format, verifies raw-mask parity, and writes a manifest. It does not edit
runtime files, invoke the scorer, or dispatch remote work.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
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
    _AdaptiveModel9Binary,
    _ArithmeticDecoder,
    _ArithmeticEncoder,
    _decode_qma9_base_pixel,
    _decode_symbol,
    _encode_qma9_base_pixel,
    _encode_symbol,
    _neighbours,
    _update_adaptive,
    _validate_raw_qma9_mask,
    decode_qma9_prefix_frames,
    encode_qma9_mask,
    parse_qma9_header,
    qma9_context_id,
    read_single_member_zip,
    sha256_bytes,
    split_qma9_pr81_payload,
)


TOOL = "experiments/build_qma9_context_run_escape_candidate.py"
SCHEMA = "qma9_context_run_escape_candidate_screen_v1"
QMC1_MAGIC = b"QMC1"
QMC1_HEADER = struct.Struct("<4sIIIIII")
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


def _run_bucket(run_len: int) -> int:
    return min(8, int(math.log2(max(1, int(run_len)))))


def _copy_flag_context_key(cls: int, run_len: int, left_matches_class: bool) -> tuple[int, int, int]:
    return int(cls), _run_bucket(run_len), 1 if left_matches_class else 0


def _flag_context(
    contexts: dict[tuple[int, int, int], list[int]],
    key: tuple[int, int, int],
) -> list[int]:
    state = contexts.get(key)
    if state is None:
        state = [1, 1]
        contexts[key] = state
    return state


def _row_above_run(
    data: bytes | bytearray,
    *,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    min_run_length: int,
    require_left_context: bool,
) -> tuple[int, int, bool] | None:
    if y <= 0 or xcoord >= height:
        return None
    base = t * frame_size + y * height
    prev_row = base - height
    cls = int(data[prev_row + xcoord])
    if xcoord > 0 and int(data[prev_row + xcoord - 1]) == cls:
        return None
    run_end = xcoord + 1
    while run_end < height and int(data[prev_row + run_end]) == cls:
        run_end += 1
    run_len = run_end - xcoord
    if run_len < min_run_length:
        return None
    left_matches_class = xcoord == 0 or int(data[base + xcoord - 1]) == cls
    if require_left_context and not left_matches_class:
        return None
    return cls, run_len, left_matches_class


def _update_base_model_for_known_pixel(
    *,
    model: _AdaptiveModel9Binary,
    data: bytes | bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    cls: int,
) -> None:
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        data, frame_size, t, y, width, height, xcoord
    )
    ctx = model.context(qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2))
    if cls == up:
        _update_adaptive(ctx.up_freq, 1)
        return
    _update_adaptive(ctx.up_freq, 0)
    if cls == left:
        _update_adaptive(ctx.left_freq, 1)
        return
    _update_adaptive(ctx.left_freq, 0)
    if cls == prev:
        _update_adaptive(ctx.prev_freq, 1)
        return
    _update_adaptive(ctx.prev_freq, 0)
    _update_adaptive(ctx.class_freq, int(cls))


def encode_qma9_context_run_escape_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    min_run_length: int = 64,
    require_left_context: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a local-only ``QMC1`` row-above run escape variant."""

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    min_run_length = int(min_run_length)
    if min_run_length <= 0:
        raise QMA9ContractError("min run length must be positive")
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)

    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    flag_contexts: dict[tuple[int, int, int], list[int]] = {}
    frame_size = width * height
    stats = {
        "eligible_run_starts": 0,
        "copied_runs": 0,
        "rejected_runs": 0,
        "copied_pixels": 0,
        "base_modeled_pixels": 0,
        "escaped_model_update_pixels": 0,
    }

    for t in range(frame_count):
        for y in range(width):
            xcoord = 0
            while xcoord < height:
                run = _row_above_run(
                    raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=width,
                    height=height,
                    xcoord=xcoord,
                    min_run_length=min_run_length,
                    require_left_context=require_left_context,
                )
                if run is not None:
                    cls, run_len, left_matches_class = run
                    base = t * frame_size + y * height
                    prev_row = base - height
                    copy_run = raw[base + xcoord:base + xcoord + run_len] == raw[prev_row + xcoord:prev_row + xcoord + run_len]
                    key = _copy_flag_context_key(cls, run_len, left_matches_class)
                    freq = _flag_context(flag_contexts, key)
                    flag = 1 if copy_run else 0
                    _encode_symbol(encoder, freq, flag)
                    _update_adaptive(freq, flag)
                    stats["eligible_run_starts"] += 1
                    if copy_run:
                        stats["copied_runs"] += 1
                        stats["copied_pixels"] += run_len
                        stats["escaped_model_update_pixels"] += run_len
                        for copied_x in range(xcoord, xcoord + run_len):
                            _update_base_model_for_known_pixel(
                                model=model,
                                data=raw,
                                frame_size=frame_size,
                                t=t,
                                y=y,
                                width=width,
                                height=height,
                                xcoord=copied_x,
                                cls=int(raw[base + copied_x]),
                            )
                        xcoord += run_len
                        continue
                    stats["rejected_runs"] += 1

                _encode_qma9_base_pixel(
                    encoder=encoder,
                    model=model,
                    raw=raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=width,
                    height=height,
                    xcoord=xcoord,
                )
                stats["base_modeled_pixels"] += 1
                xcoord += 1

    bitstream = encoder.finish()
    payload = QMC1_HEADER.pack(
        QMC1_MAGIC,
        frame_count,
        width,
        height,
        min_run_length,
        1 if require_left_context else 0,
        len(bitstream),
    ) + bitstream
    total_pixels = frame_count * frame_size
    stats.update(
        {
            "min_run_length": min_run_length,
            "require_left_context": bool(require_left_context),
            "flag_context_count": len(flag_contexts),
            "total_pixels": total_pixels,
            "copied_pixel_fraction": stats["copied_pixels"] / max(1, total_pixels),
            "eligible_acceptance_fraction": stats["copied_runs"] / max(1, stats["eligible_run_starts"]),
            "model_update_policy": "copied runs update the base QMA9 adaptive model without emitting base symbols",
        }
    )
    return payload, stats


def decode_qma9_context_run_escape_mask(payload: bytes) -> bytes:
    """Decode a local-only ``QMC1`` row-above run escape payload."""

    if len(payload) < QMC1_HEADER.size:
        raise QMA9ContractError("QMC1 payload is shorter than its header")
    magic, frame_count, width, height, min_run_length, require_left_context_u32, bitstream_bytes = QMC1_HEADER.unpack_from(payload, 0)
    if magic != QMC1_MAGIC:
        raise QMA9ContractError(f"expected QMC1 magic, got {magic!r}")
    packed_bytes = QMC1_HEADER.size + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMC1 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    if min(int(frame_count), int(width), int(height), int(min_run_length)) <= 0:
        raise QMA9ContractError("QMC1 dimensions and min run length must be positive")
    require_left_context = bool(require_left_context_u32)
    bitstream = payload[QMC1_HEADER.size:packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    flag_contexts: dict[tuple[int, int, int], list[int]] = {}
    frame_size = int(width) * int(height)
    out = bytearray(int(frame_count) * frame_size)

    for t in range(int(frame_count)):
        for y in range(int(width)):
            xcoord = 0
            while xcoord < int(height):
                run = _row_above_run(
                    out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=int(width),
                    height=int(height),
                    xcoord=xcoord,
                    min_run_length=int(min_run_length),
                    require_left_context=require_left_context,
                )
                if run is not None:
                    cls, run_len, left_matches_class = run
                    key = _copy_flag_context_key(cls, run_len, left_matches_class)
                    freq = _flag_context(flag_contexts, key)
                    flag = _decode_symbol(decoder, freq)
                    _update_adaptive(freq, flag)
                    if flag:
                        base = t * frame_size + y * int(height)
                        prev_row = base - int(height)
                        out[base + xcoord:base + xcoord + run_len] = out[prev_row + xcoord:prev_row + xcoord + run_len]
                        for copied_x in range(xcoord, xcoord + run_len):
                            _update_base_model_for_known_pixel(
                                model=model,
                                data=out,
                                frame_size=frame_size,
                                t=t,
                                y=y,
                                width=int(width),
                                height=int(height),
                                xcoord=copied_x,
                                cls=int(out[base + copied_x]),
                            )
                        xcoord += run_len
                        continue

                cls = _decode_qma9_base_pixel(
                    decoder=decoder,
                    model=model,
                    out=out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=int(width),
                    height=int(height),
                    xcoord=xcoord,
                )
                if cls < 0 or cls >= 5:
                    raise QMA9ContractError(f"decoded invalid QMC1 class symbol: {cls}")
                out[t * frame_size + y * int(height) + xcoord] = cls
                xcoord += 1
    return bytes(out)


def build_context_run_escape_screen(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    candidate_id: str,
    frames: int,
    min_run_length: int,
    require_left_context: bool,
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

    raw = decode_qma9_prefix_frames(split.range_mask, frame_count=subset_frames)
    baseline_payload = encode_qma9_mask(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
    )
    candidate_payload, opportunity_stats = encode_qma9_context_run_escape_mask(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
        min_run_length=min_run_length,
        require_left_context=require_left_context,
    )
    decoded_candidate = decode_qma9_context_run_escape_mask(candidate_payload)
    if decoded_candidate != raw:
        raise QMA9ContractError("QMC1 candidate decode parity failed")

    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    raw_path = candidate_dir / "decoded_prefix_mask.raw"
    baseline_path = candidate_dir / "baseline_subset.qma9"
    candidate_path = candidate_dir / "candidate_subset.qmc1"
    raw_path.write_bytes(raw)
    baseline_path.write_bytes(baseline_payload)
    candidate_path.write_bytes(candidate_payload)

    baseline_bytes = len(baseline_payload)
    candidate_bytes = len(candidate_payload)
    delta_bytes = candidate_bytes - baseline_bytes
    full_linear_projection_bytes = round(candidate_bytes * source_header.frame_count / subset_frames)
    reference_delta_bytes = full_linear_projection_bytes - source_header.packed_bytes
    local_negative = candidate_bytes >= baseline_bytes
    archive_relevant_state_change = int(opportunity_stats["copied_runs"]) > 0
    non_dispatchable_reasons = [
        "planning-only bounded prefix projection",
        "no contest runtime integration",
        "no CUDA auth eval",
    ]
    if not archive_relevant_state_change:
        non_dispatchable_reasons.insert(0, "no copied runs selected; candidate is a no-op except QMC1 header/wrapper bytes")
    if local_negative:
        non_dispatchable_reasons.insert(0, "candidate does not beat exact subset QMA9 re-encode")
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "candidate": "qma9_context_conditioned_run_escape_qmc1",
        "evidence_grade": "empirical/planning_only",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "source_qma9_header": asdict(source_header),
        "subset": {
            "frames": subset_frames,
            "decoded_pixels": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "raw_path": _repo_rel(raw_path),
            "baseline_subset_qma9_path": _repo_rel(baseline_path),
            "candidate_subset_qmc1_path": _repo_rel(candidate_path),
            "baseline_qma9_bytes": baseline_bytes,
            "candidate_qmc1_bytes": candidate_bytes,
            "delta_bytes_vs_subset_qma9": delta_bytes,
            "rate_score_delta_if_subset_scaled": delta_bytes * source_header.frame_count / subset_frames * 25.0 / ORIGINAL_VIDEO_BYTES,
            "decode_parity": True,
        },
        "context_run_escape": opportunity_stats,
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
            "local_screen_negative": local_negative,
            "dispatchable": False,
            "archive_relevant_state_change": archive_relevant_state_change,
            "non_dispatchable_reasons": non_dispatchable_reasons,
            "reason": (
                "QMC1 did not beat exact subset QMA9 re-encode; copy flags and headers exceed skipped base-symbol savings."
                if local_negative
                else "QMC1 beat this bounded subset only; still planning-only until full-stream runtime parity and CUDA auth eval."
            ),
        },
        "notes": [
            "QMC1 charges copy flags inside the range-mask payload at deterministic row-above run starts.",
            "Copied runs advance the base QMA9 adaptive model without emitting base pixel symbols, avoiding the QMB1 cold-model failure mode.",
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
    parser.add_argument("--candidate-id", default="qma9_context_run_escape_min64_prefix4")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--min-run-length", type=int, default=64)
    parser.add_argument("--no-require-left-context", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_context_run_escape_screen(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        frames=args.frames,
        min_run_length=args.min_run_length,
        require_left_context=not args.no_require_left_context,
    )
    candidate_dir = args.output_dir / args.candidate_id
    subset = manifest["subset"]
    projection = manifest["full_stream_linear_projection"]
    print(f"wrote {candidate_dir / 'manifest.json'}")
    print(
        "subset_bytes "
        f"baseline_qma9={subset['baseline_qma9_bytes']} "
        f"candidate_qmc1={subset['candidate_qmc1_bytes']} "
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
