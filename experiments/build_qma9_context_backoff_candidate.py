#!/usr/bin/env python3
"""Screen local QMA9 context-backoff byte candidates.

This is a planning-only byte screen. It decodes PR84/PR81 QMA9 range-mask
bytes, re-encodes a prefix or full stream with deterministic cold-context
backoff policies, verifies raw-mask parity with the matching local decoder,
and emits a manifest. It does not edit runtime files, invoke a scorer, train,
or dispatch remote work.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
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
    QMA9_HEADER_BYTES,
    QMA9_MAGIC,
    QMA9_SENTINEL,
    QMA9ContractError,
    _AdaptiveModel9Binary,
    _ArithmeticDecoder,
    _ArithmeticEncoder,
    _decode_symbol,
    _encode_symbol,
    _neighbours,
    _update_adaptive,
    _validate_raw_qma9_mask,
    decode_qma9_mask,
    decode_qma9_prefix_frames,
    encode_qma9_mask,
    parse_qma9_header,
    qma9_context_id,
    read_single_member_zip,
    sha256_bytes,
)


TOOL = "experiments/build_qma9_context_backoff_candidate.py"
SCHEMA = "qma9_context_backoff_candidate_screen_v1"
DEFAULT_PR84_DIR = REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr84"
DEFAULT_PR84_ARCHIVE = DEFAULT_PR84_DIR / "archive.zip"
DEFAULT_PR84_INFLATE = DEFAULT_PR84_DIR / "sources/inflate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/qma9_context_backoff_20260503_codex"
DEFAULT_MODES = "always_plu,warm4_plu,warm16_plu,always_lu,warm4_lu,warm16_lu"
CONTEXT_BACKOFF_HEADER = struct.Struct("<4sIIII")


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


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise QMA9ContractError("at least one context-backoff mode is required")
    return modes


def _parse_frames(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"all", "full", "none"}:
        return None
    frames = int(normalized)
    if frames <= 0:
        raise QMA9ContractError("frames must be positive or 'all'")
    return frames


def _mode_policy(mode_id: str) -> tuple[int | None, str]:
    if mode_id.startswith("always_"):
        return None, mode_id.removeprefix("always_")
    if mode_id.startswith("warm"):
        threshold_text, _, family = mode_id.partition("_")
        if not family:
            raise QMA9ContractError(f"context-backoff mode lacks family: {mode_id}")
        threshold = int(threshold_text.removeprefix("warm"))
        if threshold <= 0:
            raise QMA9ContractError(f"context-backoff warm threshold must be positive: {mode_id}")
        return threshold, family
    raise QMA9ContractError(
        f"unknown context-backoff mode {mode_id!r}; expected always_<family> or warmN_<family>"
    )


def _context_for_family(
    family: str,
    *,
    prev: int,
    left: int,
    up: int,
    up_left: int,
    up_right: int,
    prev_right: int,
    prev_down: int,
    up2: int,
    left2: int,
) -> int:
    if family == "full9":
        return qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2)
    if family == "plu":
        return qma9_context_id(
            prev,
            left,
            up,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    if family == "lu":
        return qma9_context_id(
            QMA9_SENTINEL,
            left,
            up,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    if family == "l":
        return qma9_context_id(
            QMA9_SENTINEL,
            left,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    if family == "u":
        return qma9_context_id(
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            up,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    raise QMA9ContractError(f"unknown context-backoff family: {family}")


def _select_context(
    *,
    mode_id: str,
    full_context_visits: Counter[int],
    prev: int,
    left: int,
    up: int,
    up_left: int,
    up_right: int,
    prev_right: int,
    prev_down: int,
    up2: int,
    left2: int,
) -> tuple[int, int, str, bool]:
    warm_threshold, family = _mode_policy(mode_id)
    full_ctx = _context_for_family(
        "full9",
        prev=prev,
        left=left,
        up=up,
        up_left=up_left,
        up_right=up_right,
        prev_right=prev_right,
        prev_down=prev_down,
        up2=up2,
        left2=left2,
    )
    if family == "full9" or (warm_threshold is not None and full_context_visits[full_ctx] >= warm_threshold):
        return full_ctx, full_ctx, "full9", False
    effective_ctx = _context_for_family(
        family,
        prev=prev,
        left=left,
        up=up,
        up_left=up_left,
        up_right=up_right,
        prev_right=prev_right,
        prev_down=prev_down,
        up2=up2,
        left2=left2,
    )
    return full_ctx, effective_ctx, family, True


def _encode_backoff_pixel(
    *,
    encoder: _ArithmeticEncoder,
    model: _AdaptiveModel9Binary,
    full_context_visits: Counter[int],
    raw: bytes | bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    mode_id: str,
    stats: dict[str, Any],
) -> None:
    base = t * frame_size + y * height
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        raw, frame_size, t, y, width, height, xcoord
    )
    full_ctx, effective_ctx, family_used, backed_off = _select_context(
        mode_id=mode_id,
        full_context_visits=full_context_visits,
        prev=prev,
        left=left,
        up=up,
        up_left=up_left,
        up_right=up_right,
        prev_right=prev_right,
        prev_down=prev_down,
        up2=up2,
        left2=left2,
    )
    cls = int(raw[base + xcoord])
    ctx = model.context(effective_ctx)
    if cls == up:
        _encode_symbol(encoder, ctx.up_freq, 1)
        _update_adaptive(ctx.up_freq, 1)
    else:
        _encode_symbol(encoder, ctx.up_freq, 0)
        _update_adaptive(ctx.up_freq, 0)
        if cls == left:
            _encode_symbol(encoder, ctx.left_freq, 1)
            _update_adaptive(ctx.left_freq, 1)
        else:
            _encode_symbol(encoder, ctx.left_freq, 0)
            _update_adaptive(ctx.left_freq, 0)
            if cls == prev:
                _encode_symbol(encoder, ctx.prev_freq, 1)
                _update_adaptive(ctx.prev_freq, 1)
            else:
                _encode_symbol(encoder, ctx.prev_freq, 0)
                _update_adaptive(ctx.prev_freq, 0)
                _encode_symbol(encoder, ctx.class_freq, cls)
                _update_adaptive(ctx.class_freq, cls)
    full_context_visits[full_ctx] += 1
    stats["pixels"] += 1
    stats["context_family_counts"][family_used] += 1
    if backed_off:
        stats["backoff_pixels"] += 1
    else:
        stats["full9_pixels"] += 1


def _decode_backoff_pixel(
    *,
    decoder: _ArithmeticDecoder,
    model: _AdaptiveModel9Binary,
    full_context_visits: Counter[int],
    out: bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    mode_id: str,
) -> int:
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        out, frame_size, t, y, width, height, xcoord
    )
    full_ctx, effective_ctx, _family_used, _backed_off = _select_context(
        mode_id=mode_id,
        full_context_visits=full_context_visits,
        prev=prev,
        left=left,
        up=up,
        up_left=up_left,
        up_right=up_right,
        prev_right=prev_right,
        prev_down=prev_down,
        up2=up2,
        left2=left2,
    )
    ctx = model.context(effective_ctx)
    bit = _decode_symbol(decoder, ctx.up_freq)
    _update_adaptive(ctx.up_freq, bit)
    if bit:
        cls = up
    else:
        bit = _decode_symbol(decoder, ctx.left_freq)
        _update_adaptive(ctx.left_freq, bit)
        if bit:
            cls = left
        else:
            bit = _decode_symbol(decoder, ctx.prev_freq)
            _update_adaptive(ctx.prev_freq, bit)
            if bit:
                cls = prev
            else:
                cls = _decode_symbol(decoder, ctx.class_freq)
                _update_adaptive(ctx.class_freq, cls)
    full_context_visits[full_ctx] += 1
    return int(cls)


def encode_qma9_context_backoff_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    mode_id: str,
) -> tuple[bytes, dict[str, Any]]:
    """Encode raw mask bytes with a deterministic QMA9 context-backoff policy."""

    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)
    _mode_policy(mode_id)
    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    full_context_visits: Counter[int] = Counter()
    frame_size = int(width) * int(height)
    stats: dict[str, Any] = {
        "mode_id": mode_id,
        "pixels": 0,
        "backoff_pixels": 0,
        "full9_pixels": 0,
        "context_family_counts": Counter(),
    }
    for t in range(int(frame_count)):
        for y in range(int(width)):
            for xcoord in range(int(height)):
                _encode_backoff_pixel(
                    encoder=encoder,
                    model=model,
                    full_context_visits=full_context_visits,
                    raw=raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=int(width),
                    height=int(height),
                    xcoord=xcoord,
                    mode_id=mode_id,
                    stats=stats,
                )

    bitstream = encoder.finish()
    stats["distinct_full_contexts_seen"] = len(full_context_visits)
    stats["distinct_effective_contexts_used"] = len(model._contexts)
    stats["backoff_pixel_fraction"] = stats["backoff_pixels"] / max(1, stats["pixels"])
    stats["context_family_counts"] = {
        str(key): int(value) for key, value in sorted(stats["context_family_counts"].items())
    }
    payload = CONTEXT_BACKOFF_HEADER.pack(QMA9_MAGIC, int(frame_count), int(width), int(height), len(bitstream)) + bitstream
    return payload, stats


def decode_qma9_context_backoff_mask(payload: bytes, *, mode_id: str) -> bytes:
    """Decode a local context-backoff QMA9 payload with the matching policy."""

    header = parse_qma9_header(payload)
    _mode_policy(mode_id)
    bitstream = payload[QMA9_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    full_context_visits: Counter[int] = Counter()
    frame_size = header.width * header.height
    out = bytearray(header.frame_count * frame_size)
    for t in range(header.frame_count):
        for y in range(header.width):
            base = t * frame_size + y * header.height
            for xcoord in range(header.height):
                cls = _decode_backoff_pixel(
                    decoder=decoder,
                    model=model,
                    full_context_visits=full_context_visits,
                    out=out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=header.width,
                    height=header.height,
                    xcoord=xcoord,
                    mode_id=mode_id,
                )
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid context-backoff class symbol: {cls}")
                out[base + xcoord] = cls
    return bytes(out)


def _load_raw_mask(
    *,
    qma9_payload: bytes,
    source_header: Any,
    frames: int | None,
    raw_mask_path: Path | None,
) -> tuple[bytes, dict[str, Any]]:
    if raw_mask_path is not None:
        raw = raw_mask_path.read_bytes()
        raw_frames = source_header.frame_count if frames is None else int(frames)
        _validate_raw_qma9_mask(
            raw,
            frame_count=raw_frames,
            width=source_header.width,
            height=source_header.height,
        )
        return raw, {
            "source": _repo_rel(raw_mask_path),
            "source_kind": "external_raw_mask_storage_order",
            "decoded_by_tool": False,
            "frames": raw_frames,
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
        }
    if frames is None:
        decoded = decode_qma9_mask(qma9_payload)
        return decoded.data, {
            "source": "archive_range_mask.qma9",
            "source_kind": "full_qma9_decode",
            "decoded_by_tool": True,
            "frames": source_header.frame_count,
            "raw_bytes": len(decoded.data),
            "raw_sha256": decoded.sha256,
        }
    raw = decode_qma9_prefix_frames(qma9_payload, frame_count=int(frames))
    return raw, {
        "source": "archive_range_mask.qma9",
        "source_kind": "prefix_qma9_decode",
        "decoded_by_tool": True,
        "frames": int(frames),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
    }


def _candidate_record(
    *,
    mode_id: str,
    candidate_payload: bytes | None,
    baseline_payload: bytes,
    raw: bytes,
    decoded: bytes | None,
    source_archive_bytes: int,
    stats: dict[str, Any] | None,
    path: Path | None,
    error: str | None = None,
) -> dict[str, Any]:
    parity_ok = decoded == raw if decoded is not None else False
    candidate_bytes = len(candidate_payload) if candidate_payload is not None else None
    delta = candidate_bytes - len(baseline_payload) if candidate_bytes is not None else None
    payload_changed = candidate_payload != baseline_payload if candidate_payload is not None else False
    model_changed_pixels = int((stats or {}).get("backoff_pixels", 0))
    archive_relevant_state_change = bool(payload_changed and model_changed_pixels > 0)
    local_byte_win = delta is not None and delta < 0
    rejection_reasons: list[str] = []
    if error is not None:
        rejection_reasons.append("candidate_encode_or_decode_failed")
    if not parity_ok:
        rejection_reasons.append("raw_mask_parity_failed")
    if not archive_relevant_state_change:
        rejection_reasons.append("no_op_or_source_preserving_transform")
    if not local_byte_win:
        rejection_reasons.append("no_local_byte_screen_win")
    selectable = parity_ok and archive_relevant_state_change and local_byte_win and error is None
    return {
        "mode_id": mode_id,
        "mode_family": "qma9_context_backoff_zero_side_cost",
        "payload_path": _repo_rel(path),
        "payload_bytes": candidate_bytes,
        "payload_sha256": sha256_bytes(candidate_payload) if candidate_payload is not None else None,
        "delta_bytes_vs_baseline": delta,
        "projected_archive_bytes_if_other_streams_unchanged": (
            source_archive_bytes + delta if delta is not None else None
        ),
        "rate_score_delta_if_components_unchanged": (
            delta * 25.0 / ORIGINAL_VIDEO_BYTES if delta is not None else None
        ),
        "raw_mask_parity": bool(parity_ok),
        "archive_relevant_state_change": archive_relevant_state_change,
        "no_op_status": "state_changed" if archive_relevant_state_change else "no_op_or_source_preserving",
        "selectable_for_local_followup": bool(selectable),
        "safe_for_remote_dispatch": False,
        "rejection_reasons": [] if selectable else rejection_reasons,
        "context_backoff": stats,
        "error": error,
    }


def build_context_backoff_screen(
    *,
    archive_path: Path,
    split_constants_path: Path,
    output_dir: Path,
    candidate_id: str,
    frames: int | None,
    modes: tuple[str, ...],
    raw_mask_path: Path | None = None,
    write_candidates: bool = True,
) -> dict[str, Any]:
    constants = parse_split_constants(split_constants_path)
    payload, custody = read_single_member_zip(archive_path)
    split = _split_qma9_public_payload(payload, constants)
    source_header = parse_qma9_header(split.range_mask)
    if frames is not None and int(frames) > source_header.frame_count:
        raise QMA9ContractError(f"frames {frames} exceeds source frame count {source_header.frame_count}")

    raw, raw_source = _load_raw_mask(
        qma9_payload=split.range_mask,
        source_header=source_header,
        frames=frames,
        raw_mask_path=raw_mask_path,
    )
    search_frames = int(raw_source["frames"])
    baseline_payload = (
        split.range_mask
        if frames is None and raw_mask_path is None
        else encode_qma9_mask(raw, frame_count=search_frames, width=source_header.width, height=source_header.height)
    )
    baseline_decoded = (
        raw
        if frames is None and raw_mask_path is None
        else (
            decode_qma9_mask(baseline_payload).data
            if frames is None
            else decode_qma9_prefix_frames(baseline_payload, frame_count=search_frames)
        )
    )
    if baseline_decoded != raw:
        raise QMA9ContractError("baseline QMA9 re-encode/decode parity failed")

    screen_dir = output_dir / candidate_id
    candidate_dir = screen_dir / "candidates"
    if write_candidates:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (screen_dir / "baseline.qma9").write_bytes(baseline_payload)
        if len(raw) <= 16 * 1024 * 1024:
            (screen_dir / "decoded_mask.raw").write_bytes(raw)

    candidates: list[dict[str, Any]] = []
    for mode_id in modes:
        path = candidate_dir / f"{mode_id}.qma9cb"
        try:
            candidate_payload, stats = encode_qma9_context_backoff_mask(
                raw,
                frame_count=search_frames,
                width=source_header.width,
                height=source_header.height,
                mode_id=mode_id,
            )
            decoded = decode_qma9_context_backoff_mask(candidate_payload, mode_id=mode_id)
            if write_candidates:
                path.write_bytes(candidate_payload)
            else:
                path = None
            record = _candidate_record(
                mode_id=mode_id,
                candidate_payload=candidate_payload,
                baseline_payload=baseline_payload,
                raw=raw,
                decoded=decoded,
                source_archive_bytes=custody.archive_bytes,
                stats=stats,
                path=path,
            )
        except Exception as exc:  # pragma: no cover - manifest robustness for real artifact screens
            record = _candidate_record(
                mode_id=mode_id,
                candidate_payload=None,
                baseline_payload=baseline_payload,
                raw=raw,
                decoded=None,
                source_archive_bytes=custody.archive_bytes,
                stats=None,
                path=None,
                error=str(exc),
            )
        candidates.append(record)

    selectable = [row for row in candidates if row["selectable_for_local_followup"]]
    best = min(selectable, key=lambda row: int(row["delta_bytes_vs_baseline"])) if selectable else None
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "evidence_grade": "empirical/planning_only",
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "archive": asdict(custody),
        "split_constants": constants,
        "segments": [asdict(segment) for segment in split.segments],
        "source_qma9_header": asdict(source_header),
        "raw_mask": raw_source,
        "baseline": {
            "payload_path": _repo_rel(screen_dir / "baseline.qma9") if write_candidates else None,
            "payload_bytes": len(baseline_payload),
            "payload_sha256": sha256_bytes(baseline_payload),
            "raw_mask_parity": True,
            "role": "reference_only",
            "no_op_status": "source_preserving_reference_not_selectable",
        },
        "mode_matrix": {
            "modes": list(modes),
            "candidate_count": len(candidates),
            "selectable_local_byte_wins": len(selectable),
        },
        "best_local_byte_win": best,
        "decision": {
            "safe_for_remote_dispatch": False,
            "archive_claim": False,
            "best_delta_bytes_vs_baseline": best["delta_bytes_vs_baseline"] if best else None,
            "required_before_dispatch": [
                "runtime consumes the context-backoff model from charged archive bytes",
                "candidate remains raw-parity clean on the full stream",
                "lane dispatch claim is active before any remote/GPU work",
                "exact CUDA auth eval through archive.zip -> inflate.sh -> upstream/evaluate.py",
            ],
            "reason": (
                "local parity byte win only; no runtime integration or scorer evidence"
                if best
                else "no tested context-backoff mode produced a parity-clean local byte win"
            ),
        },
        "candidates": candidates,
        "notes": [
            "Context-backoff uses deterministic decoder-side policy; no per-pixel flags or side tables are emitted.",
            "The .qma9cb files use QMA9-shaped local payloads but require this tool's matching decoder.",
            "No scorer, eval, training, GPU, remote dispatch, or archive promotion was performed.",
        ],
    }
    if write_candidates:
        _write_json(screen_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR84_ARCHIVE)
    parser.add_argument("--split-constants-py", type=Path, default=DEFAULT_PR84_INFLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="pr84_qma9_context_backoff_prefix32")
    parser.add_argument("--frames", default="32", help="Frame count to screen, or 'all' for full stream.")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--raw-mask", type=Path, default=None)
    parser.add_argument("--no-write-candidates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_context_backoff_screen(
        archive_path=args.archive,
        split_constants_path=args.split_constants_py,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        frames=_parse_frames(args.frames),
        modes=_parse_modes(args.modes),
        raw_mask_path=args.raw_mask,
        write_candidates=not args.no_write_candidates,
    )
    screen_dir = args.output_dir / args.candidate_id
    best = manifest["best_local_byte_win"]
    print(f"wrote {screen_dir / 'manifest.json'}")
    print(
        "baseline "
        f"bytes={manifest['baseline']['payload_bytes']} "
        f"frames={manifest['raw_mask']['frames']} "
        f"raw_sha256={manifest['raw_mask']['raw_sha256']}"
    )
    if best is None:
        print("best_local_byte_win=None")
    else:
        print(
            "best_local_byte_win "
            f"mode={best['mode_id']} bytes={best['payload_bytes']} "
            f"delta={best['delta_bytes_vs_baseline']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
