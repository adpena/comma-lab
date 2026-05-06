#!/usr/bin/env python3
"""Build a local QMA9 horizontal run-tail escape byte screen.

This is a planning-only prototype. It decodes a bounded PR84/PR81 QMA9 frame
prefix, compares exact subset QMA9 re-encode bytes against a local ``QMH1``
horizontal run-tail format, verifies raw-mask parity, and writes a manifest.
It does not edit runtime files, invoke the scorer, dispatch remote work, or
claim score/component evidence.
"""
from __future__ import annotations

import argparse
import json
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

from profile_qma9_range_mask_bitstream import _split_qma9_public_payload, parse_split_constants
from tac.qma9_range_mask_contract import (
    ORIGINAL_VIDEO_BYTES,
    QMA9_CLASS_SYMBOLS,
    QMA9_SENTINEL,
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
)


TOOL = "experiments/build_qma9_horizontal_run_escape_candidate.py"
SCHEMA = "qma9_horizontal_run_escape_candidate_screen_v1"
QMH1_MAGIC = b"QMH1"
QMH1_HEADER = struct.Struct("<4sIIIIII")
QMH1_FLAG_REQUIRE_UP_DISAGREEMENT = 1
DEFAULT_PR84_DIR = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr84"
DEFAULT_PR84_ARCHIVE = DEFAULT_PR84_DIR / "archive.zip"
DEFAULT_PR84_INFLATE = DEFAULT_PR84_DIR / "sources/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/qma9_horizontal_run_escape_20260503_worker/candidates"


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


def _binary_context(
    contexts: dict[tuple[int, ...], list[int]],
    key: tuple[int, ...],
) -> list[int]:
    state = contexts.get(key)
    if state is None:
        state = [1, 1]
        contexts[key] = state
    return state


def _run_length_bucket(length: int) -> int:
    value = max(1, int(length))
    bucket = 0
    while value > 1 and bucket < 8:
        value >>= 1
        bucket += 1
    return bucket


def _is_horizontal_tail_candidate(
    data: bytes | bytearray,
    *,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    min_run_length: int,
    require_up_disagreement: bool,
) -> bool:
    if xcoord <= 0 or xcoord + min_run_length > height:
        return False
    base = t * frame_size + y * height
    left = int(data[base + xcoord - 1])
    if left < 0 or left >= QMA9_CLASS_SYMBOLS:
        return False
    if xcoord > 1 and int(data[base + xcoord - 2]) == left:
        return False
    if not require_up_disagreement:
        return True
    up = QMA9_SENTINEL if y == 0 else int(data[base - height + xcoord])
    return up != left


def _horizontal_tail_length(
    raw: bytes | bytearray,
    *,
    frame_size: int,
    t: int,
    y: int,
    height: int,
    xcoord: int,
) -> int:
    base = t * frame_size + y * height
    left = int(raw[base + xcoord - 1])
    run_end = xcoord
    while run_end < height and int(raw[base + run_end]) == left:
        run_end += 1
    return run_end - xcoord


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


def _run_flag_key(
    *,
    data: bytes | bytearray,
    frame_size: int,
    t: int,
    y: int,
    height: int,
    xcoord: int,
) -> tuple[int, ...]:
    base = t * frame_size + y * height
    left = int(data[base + xcoord - 1])
    up = QMA9_SENTINEL if y == 0 else int(data[base - height + xcoord])
    prev = QMA9_SENTINEL if t == 0 else int(data[(t - 1) * frame_size + y * height + xcoord])
    return left, 1 if y == 0 else 0, 1 if up == left else 0, 1 if prev == left else 0


def _length_flag_key(left: int, offset_from_tail_start: int) -> tuple[int, ...]:
    return int(left), _run_length_bucket(offset_from_tail_start)


def _encode_known_run_tail(
    *,
    encoder: _ArithmeticEncoder,
    length_contexts: dict[tuple[int, ...], list[int]],
    model: _AdaptiveModel9Binary,
    raw: bytes,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    run_len: int,
    min_run_length: int,
    stats: dict[str, Any],
) -> int:
    base = t * frame_size + y * height
    left = int(raw[base + xcoord - 1])
    copied = 0
    while copied < min_run_length:
        _update_base_model_for_known_pixel(
            model=model,
            data=raw,
            frame_size=frame_size,
            t=t,
            y=y,
            width=width,
            height=height,
            xcoord=xcoord + copied,
            cls=left,
        )
        copied += 1

    while copied < run_len:
        freq = _binary_context(length_contexts, _length_flag_key(left, copied))
        _encode_symbol(encoder, freq, 1)
        _update_adaptive(freq, 1)
        stats["extension_continue_symbols"] += 1
        _update_base_model_for_known_pixel(
            model=model,
            data=raw,
            frame_size=frame_size,
            t=t,
            y=y,
            width=width,
            height=height,
            xcoord=xcoord + copied,
            cls=left,
        )
        copied += 1

    if xcoord + copied < height:
        freq = _binary_context(length_contexts, _length_flag_key(left, copied))
        _encode_symbol(encoder, freq, 0)
        _update_adaptive(freq, 0)
        stats["extension_stop_symbols"] += 1
    return copied


def encode_qma9_horizontal_run_escape_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    min_run_length: int = 16,
    require_up_disagreement: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a local-only ``QMH1`` horizontal run-tail escape variant."""

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    min_run_length = int(min_run_length)
    if min_run_length <= 0:
        raise QMA9ContractError("min run length must be positive")
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)

    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    run_flag_contexts: dict[tuple[int, ...], list[int]] = {}
    length_contexts: dict[tuple[int, ...], list[int]] = {}
    frame_size = width * height
    stats: dict[str, Any] = {
        "candidate_positions": 0,
        "escaped_runs": 0,
        "rejected_positions": 0,
        "copied_pixels": 0,
        "base_modeled_pixels": 0,
        "extension_continue_symbols": 0,
        "extension_stop_symbols": 0,
        "max_escaped_tail_length": 0,
    }

    for t in range(frame_count):
        for y in range(width):
            xcoord = 0
            while xcoord < height:
                if _is_horizontal_tail_candidate(
                    raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=width,
                    height=height,
                    xcoord=xcoord,
                    min_run_length=min_run_length,
                    require_up_disagreement=require_up_disagreement,
                ):
                    run_len = _horizontal_tail_length(
                        raw,
                        frame_size=frame_size,
                        t=t,
                        y=y,
                        height=height,
                        xcoord=xcoord,
                    )
                    flag = 1 if run_len >= min_run_length else 0
                    freq = _binary_context(
                        run_flag_contexts,
                        _run_flag_key(data=raw, frame_size=frame_size, t=t, y=y, height=height, xcoord=xcoord),
                    )
                    _encode_symbol(encoder, freq, flag)
                    _update_adaptive(freq, flag)
                    stats["candidate_positions"] += 1
                    if flag:
                        copied = _encode_known_run_tail(
                            encoder=encoder,
                            length_contexts=length_contexts,
                            model=model,
                            raw=raw,
                            frame_size=frame_size,
                            t=t,
                            y=y,
                            width=width,
                            height=height,
                            xcoord=xcoord,
                            run_len=run_len,
                            min_run_length=min_run_length,
                            stats=stats,
                        )
                        stats["escaped_runs"] += 1
                        stats["copied_pixels"] += copied
                        stats["max_escaped_tail_length"] = max(stats["max_escaped_tail_length"], copied)
                        xcoord += copied
                        continue
                    stats["rejected_positions"] += 1

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
    flags = QMH1_FLAG_REQUIRE_UP_DISAGREEMENT if require_up_disagreement else 0
    payload = QMH1_HEADER.pack(QMH1_MAGIC, frame_count, width, height, min_run_length, flags, len(bitstream)) + bitstream
    total_pixels = frame_count * frame_size
    stats.update(
        {
            "min_run_length": min_run_length,
            "require_up_disagreement": bool(require_up_disagreement),
            "run_flag_context_count": len(run_flag_contexts),
            "length_context_count": len(length_contexts),
            "total_pixels": total_pixels,
            "copied_pixel_fraction": stats["copied_pixels"] / max(1, total_pixels),
            "candidate_acceptance_fraction": stats["escaped_runs"] / max(1, stats["candidate_positions"]),
            "model_update_policy": "escaped horizontal tails update the base QMA9 adaptive model without emitting base symbols",
        }
    )
    return payload, stats


def decode_qma9_horizontal_run_escape_mask(payload: bytes) -> bytes:
    """Decode a local-only ``QMH1`` horizontal run-tail payload."""

    if len(payload) < QMH1_HEADER.size:
        raise QMA9ContractError("QMH1 payload is shorter than its header")
    magic, frame_count, width, height, min_run_length, flags, bitstream_bytes = QMH1_HEADER.unpack_from(payload, 0)
    if magic != QMH1_MAGIC:
        raise QMA9ContractError(f"expected QMH1 magic, got {magic!r}")
    packed_bytes = QMH1_HEADER.size + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMH1 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    if min(int(frame_count), int(width), int(height), int(min_run_length)) <= 0:
        raise QMA9ContractError("QMH1 dimensions and min run length must be positive")

    require_up_disagreement = bool(int(flags) & QMH1_FLAG_REQUIRE_UP_DISAGREEMENT)
    bitstream = payload[QMH1_HEADER.size:packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    run_flag_contexts: dict[tuple[int, ...], list[int]] = {}
    length_contexts: dict[tuple[int, ...], list[int]] = {}
    frame_size = int(width) * int(height)
    out = bytearray(int(frame_count) * frame_size)

    for t in range(int(frame_count)):
        for y in range(int(width)):
            xcoord = 0
            while xcoord < int(height):
                if _is_horizontal_tail_candidate(
                    out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=int(width),
                    height=int(height),
                    xcoord=xcoord,
                    min_run_length=int(min_run_length),
                    require_up_disagreement=require_up_disagreement,
                ):
                    freq = _binary_context(
                        run_flag_contexts,
                        _run_flag_key(data=out, frame_size=frame_size, t=t, y=y, height=int(height), xcoord=xcoord),
                    )
                    flag = _decode_symbol(decoder, freq)
                    _update_adaptive(freq, flag)
                    if flag:
                        base = t * frame_size + y * int(height)
                        left = int(out[base + xcoord - 1])
                        copied = 0
                        while copied < int(min_run_length):
                            out[base + xcoord + copied] = left
                            _update_base_model_for_known_pixel(
                                model=model,
                                data=out,
                                frame_size=frame_size,
                                t=t,
                                y=y,
                                width=int(width),
                                height=int(height),
                                xcoord=xcoord + copied,
                                cls=left,
                            )
                            copied += 1
                        while xcoord + copied < int(height):
                            freq = _binary_context(length_contexts, _length_flag_key(left, copied))
                            cont = _decode_symbol(decoder, freq)
                            _update_adaptive(freq, cont)
                            if not cont:
                                break
                            out[base + xcoord + copied] = left
                            _update_base_model_for_known_pixel(
                                model=model,
                                data=out,
                                frame_size=frame_size,
                                t=t,
                                y=y,
                                width=int(width),
                                height=int(height),
                                xcoord=xcoord + copied,
                                cls=left,
                            )
                            copied += 1
                        xcoord += copied
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
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid QMH1 class symbol: {cls}")
                out[t * frame_size + y * int(height) + xcoord] = cls
                xcoord += 1
    return bytes(out)


def _parse_qmh1_header(payload: bytes) -> dict[str, Any]:
    if len(payload) < QMH1_HEADER.size:
        raise QMA9ContractError("QMH1 payload is shorter than its header")
    magic, frame_count, width, height, min_run_length, flags, bitstream_bytes = QMH1_HEADER.unpack_from(payload, 0)
    if magic != QMH1_MAGIC:
        raise QMA9ContractError(f"expected QMH1 magic, got {magic!r}")
    packed_bytes = QMH1_HEADER.size + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMH1 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    bitstream = payload[QMH1_HEADER.size:packed_bytes]
    return {
        "magic": magic.decode("ascii"),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "min_run_length": int(min_run_length),
        "flags": int(flags),
        "require_up_disagreement": bool(int(flags) & QMH1_FLAG_REQUIRE_UP_DISAGREEMENT),
        "bitstream_bytes": int(bitstream_bytes),
        "header_bytes": QMH1_HEADER.size,
        "packed_bytes": packed_bytes,
        "decoded_mask_bytes": int(frame_count) * int(width) * int(height),
        "bitstream_sha256": sha256_bytes(bitstream),
        "payload_sha256": sha256_bytes(payload[:packed_bytes]),
    }


def build_horizontal_run_escape_screen(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    candidate_id: str,
    frames: int,
    min_run_length: int,
    require_up_disagreement: bool,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, custody = read_single_member_zip(archive_path)
    split = _split_qma9_public_payload(payload, constants)
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
    candidate_payload, opportunity_stats = encode_qma9_horizontal_run_escape_mask(
        raw,
        frame_count=subset_frames,
        width=source_header.width,
        height=source_header.height,
        min_run_length=min_run_length,
        require_up_disagreement=require_up_disagreement,
    )
    decoded_candidate = decode_qma9_horizontal_run_escape_mask(candidate_payload)
    if decoded_candidate != raw:
        raise QMA9ContractError("QMH1 candidate decode parity failed")

    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    raw_path = candidate_dir / "decoded_prefix_mask.raw"
    baseline_path = candidate_dir / "baseline_subset.qma9"
    candidate_path = candidate_dir / "candidate_subset.qmh1"
    raw_path.write_bytes(raw)
    baseline_path.write_bytes(baseline_payload)
    candidate_path.write_bytes(candidate_payload)

    baseline_bytes = len(baseline_payload)
    candidate_bytes = len(candidate_payload)
    delta_bytes = candidate_bytes - baseline_bytes
    full_linear_projection_bytes = round(candidate_bytes * source_header.frame_count / subset_frames)
    reference_delta_bytes = full_linear_projection_bytes - source_header.packed_bytes
    local_negative = candidate_bytes >= baseline_bytes
    archive_relevant_state_change = int(opportunity_stats["escaped_runs"]) > 0
    non_dispatchable_reasons = [
        "planning-only bounded prefix projection",
        "no contest runtime integration",
        "no CUDA auth eval",
        "worker scope forbids remote dispatch",
    ]
    if not archive_relevant_state_change:
        non_dispatchable_reasons.insert(0, "no escaped runs selected; candidate is a no-op except QMH1 header/wrapper bytes")
    if local_negative:
        non_dispatchable_reasons.insert(0, "candidate does not beat exact subset QMA9 re-encode")

    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "candidate": "qma9_horizontal_run_tail_escape_qmh1",
        "evidence_grade": "empirical/planning_only",
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "segments": [asdict(segment) for segment in split.segments],
        "source_qma9_header": asdict(source_header),
        "candidate_qmh1_header": _parse_qmh1_header(candidate_payload),
        "subset": {
            "frames": subset_frames,
            "decoded_pixels": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "raw_path": _repo_rel(raw_path),
            "baseline_subset_qma9_path": _repo_rel(baseline_path),
            "candidate_subset_qmh1_path": _repo_rel(candidate_path),
            "baseline_qma9_bytes": baseline_bytes,
            "baseline_qma9_sha256": sha256_bytes(baseline_payload),
            "candidate_qmh1_bytes": candidate_bytes,
            "candidate_qmh1_sha256": sha256_bytes(candidate_payload),
            "delta_bytes_vs_subset_qma9": delta_bytes,
            "rate_score_delta_if_subset_scaled": delta_bytes * source_header.frame_count / subset_frames * 25.0 / ORIGINAL_VIDEO_BYTES,
            "decode_parity": True,
        },
        "horizontal_run_escape": opportunity_stats,
        "full_stream_linear_projection": {
            "projection": "candidate_subset_bytes * source_frame_count / subset_frames",
            "candidate_range_mask_bytes": full_linear_projection_bytes,
            "reference_source_range_mask_bytes": source_header.packed_bytes,
            "delta_bytes_vs_source_range_mask": reference_delta_bytes,
            "projected_archive_bytes_if_other_streams_unchanged": custody.archive_bytes + reference_delta_bytes,
            "rate_score_delta_if_components_unchanged": reference_delta_bytes * 25.0 / ORIGINAL_VIDEO_BYTES,
            "dispatchable": False,
            "reason": "linear projection from a bounded prefix; requires full deterministic re-encode and runtime integration",
        },
        "decision": {
            "local_screen_negative": local_negative,
            "local_byte_win": delta_bytes < 0,
            "raw_mask_parity": True,
            "dispatchable": False,
            "accepted_for_exact_eval_candidate": False,
            "archive_relevant_state_change": archive_relevant_state_change,
            "dispatch_gate": "planning_only/no_remote_dispatch",
            "non_dispatchable_reasons": non_dispatchable_reasons,
            "required_gates_before_dispatch": [
                "full 600-frame QMH1 encode/decode raw-mask parity",
                "runtime/inflate integration that consumes QMH1 bytes from archive",
                "deterministic archive closure and manifest with payload/runtime SHAs",
                "orchestrator lane claim before any remote/GPU eval",
                "exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py",
            ],
            "reason": (
                "QMH1 did not beat exact subset QMA9 re-encode; run flags/length bits exceed skipped base-symbol savings."
                if local_negative
                else "QMH1 beat this bounded subset only; still planning-only until full-stream runtime parity and CUDA auth eval."
            ),
        },
        "notes": [
            "QMH1 encodes a run flag only immediately after a horizontal transition or row-start pixel.",
            "Accepted escapes copy the same-left tail and advance the base QMA9 adaptive model without emitting base symbols.",
            "The default gate requires up != left, targeting horizontal tails where the dominant QMA9 up predictor is not already free.",
            "This artifact is not a score/component claim and did not invoke CUDA or a scorer.",
        ],
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR84_ARCHIVE)
    parser.add_argument("--split-constants-py", type=Path, default=DEFAULT_PR84_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="qma9_horizontal_run_escape_min16_prefix4")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--min-run-length", type=int, default=16)
    parser.add_argument("--allow-up-match", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_horizontal_run_escape_screen(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        frames=args.frames,
        min_run_length=args.min_run_length,
        require_up_disagreement=not args.allow_up_match,
    )
    candidate_dir = args.output_dir / args.candidate_id
    subset = manifest["subset"]
    projection = manifest["full_stream_linear_projection"]
    run = manifest["horizontal_run_escape"]
    print(f"wrote {candidate_dir / 'manifest.json'}")
    print(
        "subset_bytes "
        f"baseline_qma9={subset['baseline_qma9_bytes']} "
        f"candidate_qmh1={subset['candidate_qmh1_bytes']} "
        f"delta={subset['delta_bytes_vs_subset_qma9']}"
    )
    print(
        "qmh1_runs "
        f"candidates={run['candidate_positions']} "
        f"escaped={run['escaped_runs']} "
        f"copied_pixels={run['copied_pixels']}"
    )
    print(
        "linear_projection "
        f"candidate_range_mask_bytes={projection['candidate_range_mask_bytes']} "
        f"delta_vs_source={projection['delta_bytes_vs_source_range_mask']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
