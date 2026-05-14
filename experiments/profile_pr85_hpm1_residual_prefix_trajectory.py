#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile PR85 QMA9 -> PR91 HPM1 residual re-encode prefixes.

This is a planning-only local profiler. It reads the decoded PR85 QMA9 token
source once, normalizes storage order ``N,W,H`` to render order ``N,H,W``, and
calls the existing PR91 HPM1 residual re-encode prototype for small requested
prefixes. It never runs scorers, exact eval, remote dispatch, or GPU work by
default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr91_hpm1_codec import (  # noqa: E402
    DEFAULT_HPAC_PROBABILITY_VARIANT,
    DEFAULT_PR85_QMA9_TOKEN_SOURCE,
    DEFAULT_PR91_ARCHIVE,
    EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256,
    EXPECTED_PR91_HPM1_HPAC_SHA256,
    EXPECTED_PR91_HPM1_MASK_BYTES,
    EXPECTED_PR91_HPM1_MASK_SHA256,
    EXPECTED_PR91_HPM1_TOKENS_SHA256,
    extract_pr91_hpm1_payload,
    prototype_reencode_hpm1_residual_from_raw_tokens,
)


TOOL = "experiments/profile_pr85_hpm1_residual_prefix_trajectory.py"
SCHEMA = "pr85_hpm1_residual_prefix_trajectory_v1"
DEFAULT_FRAME_COUNTS = "1,2,4"
DEFAULT_RAW_TOKEN_SHAPE = "600,512,384"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr85_hpm1_residual_prefix_trajectory_20260504_codex"
DEFAULT_JSON_OUT = DEFAULT_OUTPUT_DIR / "profile.json"
DEFAULT_MD_OUT = DEFAULT_OUTPUT_DIR / "profile.md"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return _repo_rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_frame_counts(text: str) -> list[int]:
    values: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid frame count {part!r}") from exc
        if value <= 0:
            raise ValueError(f"frame counts must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--frame-counts must include at least one positive integer")
    if values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("--frame-counts must be strictly increasing for marginal-byte accounting")
    return values


def _parse_shape(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("token shape must have exactly three comma-separated dimensions")
    shape = tuple(int(part) for part in parts)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"token shape dimensions must be positive, got {shape!r}")
    return shape  # type: ignore[return-value]


def load_raw_tokens(
    path: Path,
    *,
    shape_text: str = DEFAULT_RAW_TOKEN_SHAPE,
    layout: str = "qma9_storage_nwh_to_nhw",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load uint8 PR85 mask tokens and return contiguous render-order N,H,W."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"raw token source is missing: {path}")
    shape = _parse_shape(shape_text)
    raw = path.read_bytes()
    expected = shape[0] * shape[1] * shape[2]
    if len(raw) != expected:
        raise ValueError(
            f"raw token size mismatch for {path}: got {len(raw)} bytes, expected {expected}"
        )

    if layout in {"qma9_storage_nwh_to_nhw", "qma9_storage_wh_to_render_hw"}:
        storage_shape = [shape[0], shape[1], shape[2]]
        returned_shape = [shape[0], shape[2], shape[1]]
        transform = "reshape_NWH_transpose_to_NHW"
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape).transpose(0, 2, 1).copy()
    elif layout in {"nhw_render_order", "legacy_assume_nhw"}:
        storage_shape = [shape[0], shape[1], shape[2]]
        returned_shape = list(storage_shape)
        transform = "none"
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    else:
        raise ValueError(
            "unsupported raw token layout; expected qma9_storage_nwh_to_nhw or nhw_render_order"
        )

    observed_min = int(arr.min()) if arr.size else None
    observed_max = int(arr.max()) if arr.size else None
    if observed_min is None or observed_max is None or observed_min < 0 or observed_max > 4:
        raise ValueError(
            f"raw token class value out of range 0..4: min={observed_min} max={observed_max}"
        )
    storage_sha = _sha256_bytes(raw)
    normalized_bytes = arr.tobytes(order="C")
    return arr, {
        "path": _repo_rel(path),
        "bytes": len(raw),
        "sha256": storage_sha,
        "expected_pr85_qma9_token_source_sha256": EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256,
        "matches_expected_pr85_qma9_token_source": (
            storage_sha == EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256
        ),
        "dtype": "uint8",
        "semantic": "PR85 QMA9 decoded mask class token ids",
        "layout": layout,
        "storage_shape": storage_shape,
        "returned_shape": returned_shape,
        "storage_order": (
            "frame_major_header_width_by_header_height"
            if transform != "none"
            else "frame_major_header_height_by_header_width"
        ),
        "normalization": transform,
        "normalized_nhw_sha256": _sha256_bytes(normalized_bytes),
        "observed_range": {"min": observed_min, "max": observed_max},
    }


def _payload_custody(source_payload: Any, archive: Path) -> dict[str, Any]:
    archive_path = Path(getattr(source_payload, "archive_path", archive) or archive)
    archive_identity: dict[str, Any] = {"path": _repo_rel(archive_path), "exists": archive_path.is_file()}
    if archive_path.is_file():
        archive_identity.update({"bytes": archive_path.stat().st_size, "sha256": _sha256_path(archive_path)})

    contract = getattr(source_payload, "contract", None)
    metadata = dict(getattr(contract, "metadata", {}) or {})
    segment = getattr(source_payload, "segment", b"")
    tokens_blob = getattr(source_payload, "tokens_blob", b"")
    hpac_ppmd_blob = getattr(source_payload, "hpac_ppmd_blob", b"")
    return {
        "archive": archive_identity,
        "archive_report": dict(getattr(source_payload, "archive_report", {}) or {}),
        "bundle_report": dict(getattr(source_payload, "bundle_report", {}) or {}),
        "hpm1_segment": {
            "bytes": len(segment) if segment else getattr(contract, "bytes", None),
            "sha256": _sha256_bytes(segment) if segment else getattr(contract, "sha256", None),
            "expected_pr91_hpm1_bytes": EXPECTED_PR91_HPM1_MASK_BYTES,
            "expected_pr91_hpm1_sha256": EXPECTED_PR91_HPM1_MASK_SHA256,
            "matches_expected_pr91_hpm1": (
                bool(segment)
                and len(segment) == EXPECTED_PR91_HPM1_MASK_BYTES
                and _sha256_bytes(segment) == EXPECTED_PR91_HPM1_MASK_SHA256
            ),
            "metadata": metadata,
        },
        "hpm1_tokens_blob": {
            "bytes": len(tokens_blob),
            "sha256": _sha256_bytes(tokens_blob) if tokens_blob else None,
            "expected_pr91_hpm1_tokens_sha256": EXPECTED_PR91_HPM1_TOKENS_SHA256,
        },
        "hpm1_hpac_ppmd_blob": {
            "bytes": len(hpac_ppmd_blob),
            "sha256": _sha256_bytes(hpac_ppmd_blob) if hpac_ppmd_blob else None,
            "expected_pr91_hpm1_hpac_sha256": EXPECTED_PR91_HPM1_HPAC_SHA256,
        },
    }


PrototypeFn = Callable[..., Mapping[str, Any]]


def build_report(
    *,
    raw_tokens_nhw: np.ndarray,
    raw_token_source: Mapping[str, Any],
    source_payload: Any,
    source_archive: Path,
    frame_counts: list[int],
    probability_variant: str = DEFAULT_HPAC_PROBABILITY_VARIANT,
    device: str = "cpu",
    prototype_fn: PrototypeFn = prototype_reencode_hpm1_residual_from_raw_tokens,
) -> dict[str, Any]:
    started = time.time()
    if raw_tokens_nhw.ndim != 3:
        raise ValueError(f"raw tokens must be N,H,W, got shape {list(raw_tokens_nhw.shape)}")
    total_frames, height, width = (int(v) for v in raw_tokens_nhw.shape)
    for frame_count in frame_counts:
        if frame_count > total_frames:
            raise ValueError(
                f"requested frame count {frame_count} exceeds token source frames {total_frames}"
            )

    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for frame_count in frame_counts:
        row_started = time.time()
        prototype_report = dict(
            prototype_fn(
                raw_tokens_nhw,
                source_payload,
                max_frames=frame_count,
                probability_variant=probability_variant,
                device=device,
            )
        )
        candidate = dict(prototype_report.get("candidate_hpm1_segment", {}) or {})
        segment_bytes = int(candidate["bytes"])
        encoded_token_bytes = int(candidate.get("tokens_len", 0))
        hpac_bytes = int(candidate.get("hpac_len", 0))
        prefix = np.ascontiguousarray(raw_tokens_nhw[:frame_count])
        token_bytes = int(prefix.nbytes)
        row: dict[str, Any] = {
            "frame_count": frame_count,
            "token_bytes": token_bytes,
            "token_prefix_sha256": _sha256_bytes(prefix.tobytes(order="C")),
            "token_average_bytes_per_frame": round(token_bytes / frame_count, 6),
            "candidate_hpm1_segment_bytes": segment_bytes,
            "candidate_hpm1_segment_sha256": candidate.get("sha256"),
            "candidate_hpm1_token_stream_bytes": encoded_token_bytes,
            "candidate_hpm1_hpac_model_bytes": hpac_bytes,
            "segment_average_bytes_per_frame": round(segment_bytes / frame_count, 6),
            "encoded_token_stream_average_bytes_per_frame": round(
                encoded_token_bytes / frame_count, 6
            ),
            "prototype_status": prototype_report.get("status"),
            "prototype_elapsed_sec": prototype_report.get("elapsed_sec"),
            "call_elapsed_sec": round(time.time() - row_started, 3),
            "input_tokens_sha256": prototype_report.get("input_tokens_sha256"),
            "residual_symbols_sha256": prototype_report.get("residual_symbols_sha256"),
            "residual_roundtrip_raw_tokens_sha256": prototype_report.get(
                "residual_roundtrip_raw_tokens_sha256"
            ),
            "hpm1_encode": prototype_report.get("hpm1_encode", {}),
            "score_claim": False,
            "dispatch_unlocked": False,
        }
        if previous is None:
            row["marginal_vs_previous_prefix"] = None
        else:
            delta_frames = frame_count - int(previous["frame_count"])
            delta_segment = segment_bytes - int(previous["candidate_hpm1_segment_bytes"])
            delta_token_stream = encoded_token_bytes - int(
                previous["candidate_hpm1_token_stream_bytes"]
            )
            delta_raw_tokens = token_bytes - int(previous["token_bytes"])
            row["marginal_vs_previous_prefix"] = {
                "previous_frame_count": int(previous["frame_count"]),
                "added_frames": delta_frames,
                "raw_token_bytes_delta": delta_raw_tokens,
                "raw_token_bytes_per_added_frame": round(delta_raw_tokens / delta_frames, 6),
                "candidate_hpm1_segment_bytes_delta": delta_segment,
                "candidate_hpm1_segment_bytes_per_added_frame": round(
                    delta_segment / delta_frames, 6
                ),
                "candidate_hpm1_token_stream_bytes_delta": delta_token_stream,
                "candidate_hpm1_token_stream_bytes_per_added_frame": round(
                    delta_token_stream / delta_frames, 6
                ),
            }
        rows.append(row)
        previous = row

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "planning_only": True,
        "prototype_only": True,
        "score_claim": False,
        "dispatch_unlocked": False,
        "dispatch_performed": False,
        "gpu_or_remote_work": False,
        "local_only": True,
        "remote_dispatch": "not_performed",
        "exact_eval": "not_run",
        "safety": {
            "default_frame_counts": DEFAULT_FRAME_COUNTS,
            "full_600_not_default": True,
            "scorers_loaded": False,
            "remote_jobs_launched": False,
        },
        "inputs": {
            "frame_counts": frame_counts,
            "device": device,
            "probability_variant": probability_variant,
            "raw_token_source": dict(raw_token_source),
            "normalized_token_shape_nhw": [total_frames, height, width],
            "pr91_hpm1_payload": _payload_custody(source_payload, source_archive),
        },
        "trajectory": rows,
        "summary": {
            "profiled_prefix_count": len(rows),
            "min_frame_count": min(frame_counts),
            "max_frame_count": max(frame_counts),
            "max_candidate_hpm1_segment_bytes": max(
                int(row["candidate_hpm1_segment_bytes"]) for row in rows
            ),
            "max_raw_token_prefix_bytes": max(int(row["token_bytes"]) for row in rows),
            "score_claim": False,
            "dispatch_unlocked": False,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# PR85 -> HPM1 Residual Prefix Trajectory",
        "",
        f"- schema: `{report['schema']}`",
        f"- planning_only: `{str(report['planning_only']).lower()}`",
        f"- score_claim: `{str(report['score_claim']).lower()}`",
        f"- dispatch_unlocked: `{str(report['dispatch_unlocked']).lower()}`",
        f"- exact_eval: `{report['exact_eval']}`",
        "",
        "| frames | raw token bytes | HPM1 segment bytes | HPM1 token bytes | marginal segment B/frame | elapsed s | segment sha256 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["trajectory"]:
        marginal = row.get("marginal_vs_previous_prefix")
        marginal_text = (
            "n/a"
            if marginal is None
            else str(marginal["candidate_hpm1_segment_bytes_per_added_frame"])
        )
        lines.append(
            "| {frames} | {token_bytes} | {segment_bytes} | {token_stream_bytes} | "
            "{marginal} | {elapsed} | `{sha}` |".format(
                frames=row["frame_count"],
                token_bytes=row["token_bytes"],
                segment_bytes=row["candidate_hpm1_segment_bytes"],
                token_stream_bytes=row["candidate_hpm1_token_stream_bytes"],
                marginal=marginal_text,
                elapsed=row["call_elapsed_sec"],
                sha=str(row.get("candidate_hpm1_segment_sha256"))[:16],
            )
        )
    lines.extend(
        [
            "",
            "## Custody",
            "",
            f"- raw token source: `{report['inputs']['raw_token_source']['path']}`",
            f"- raw token sha256: `{report['inputs']['raw_token_source']['sha256']}`",
            f"- normalized NHW sha256: `{report['inputs']['raw_token_source']['normalized_nhw_sha256']}`",
            f"- PR91 archive: `{report['inputs']['pr91_hpm1_payload']['archive']['path']}`",
            "",
            "## Safety",
            "",
            "- This artifact is planning-only and local-only.",
            "- It does not claim score, unlock dispatch, run exact eval, or launch remote work.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PR91_ARCHIVE)
    parser.add_argument("--raw-token-bin", type=Path, default=DEFAULT_PR85_QMA9_TOKEN_SOURCE)
    parser.add_argument(
        "--raw-token-shape",
        default=DEFAULT_RAW_TOKEN_SHAPE,
        help="Shape before layout normalization. Default is PR85 storage order N,W,H.",
    )
    parser.add_argument(
        "--raw-token-layout",
        default="qma9_storage_nwh_to_nhw",
        choices=(
            "qma9_storage_nwh_to_nhw",
            "qma9_storage_wh_to_render_hw",
            "nhw_render_order",
            "legacy_assume_nhw",
        ),
    )
    parser.add_argument(
        "--frame-counts",
        default=DEFAULT_FRAME_COUNTS,
        help="Strictly increasing comma-separated local prefixes. Default stays small.",
    )
    parser.add_argument("--probability-variant", default=DEFAULT_HPAC_PROBABILITY_VARIANT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    frame_counts = parse_frame_counts(args.frame_counts)
    raw_tokens_nhw, raw_report = load_raw_tokens(
        args.raw_token_bin,
        shape_text=args.raw_token_shape,
        layout=args.raw_token_layout,
    )
    source_payload = extract_pr91_hpm1_payload(args.archive)
    report = build_report(
        raw_tokens_nhw=raw_tokens_nhw,
        raw_token_source=raw_report,
        source_payload=source_payload,
        source_archive=args.archive,
        frame_counts=frame_counts,
        probability_variant=args.probability_variant,
        device=args.device,
        prototype_fn=prototype_reencode_hpm1_residual_from_raw_tokens,
    )
    _write_json(args.json_out, report)
    write_markdown(args.md_out, report)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")
    print("score_claim=false dispatch_unlocked=false dispatch_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
