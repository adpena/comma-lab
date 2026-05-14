#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bit-level PR85-family archive comparison and rate Lagrangian profile.

This is a local-only observability tool. It parses PR85/PR91-style single
member ``x`` bundles, compares charged byte segments bit by bit, and emits the
rate term each delta would contribute under the official contest formula. It
does not inflate frames, run scorers, dispatch jobs, or make score claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from tac.pr85_bundle import SEGMENT_ORDER, infer_pr85_segment_contract, parse_pr85_bundle


SCHEMA = "pr85_family_bit_lagrangian_profile_v1"
TOOL = "experiments/profile_pr85_family_bit_lagrangian.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_LAMBDA = 25.0 / ORIGINAL_VIDEO_BYTES
NO_SCORE_CLAIM = (
    "Local byte/Lagrangian profile only. Exact score truth remains archive.zip -> "
    "inflate.sh -> upstream/evaluate.py on CUDA."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _entropy_bits_per_symbol(data: bytes) -> float:
    if not data:
        return 0.0
    total = float(len(data))
    entropy = 0.0
    for count in Counter(data).values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _first_diff(a: bytes, b: bytes) -> dict[str, Any] | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return {
                "byte_offset": idx,
                "bit_offset": idx * 8,
                "left_byte": a[idx],
                "right_byte": b[idx],
                "xor": a[idx] ^ b[idx],
            }
    if len(a) != len(b):
        return {
            "byte_offset": limit,
            "bit_offset": limit * 8,
            "left_byte": None if len(a) <= limit else a[limit],
            "right_byte": None if len(b) <= limit else b[limit],
            "xor": None,
            "reason": "length_mismatch_after_common_prefix",
        }
    return None


def _common_prefix_len(a: bytes, b: bytes) -> int:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if a[idx] != b[idx]:
            return idx
    return limit


def _hamming_bits_common_prefix(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b, strict=False))


def _safe_single_member_zip(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"{path} must contain exactly one member, got {len(infos)}")
        info = infos[0]
        if info.filename != "x":
            raise ValueError(f"{path} single member must be named 'x', got {info.filename!r}")
        data = zf.read(info.filename)
        return info.filename, data, {
            "member_name": info.filename,
            "file_size": int(info.file_size),
            "compress_size": int(info.compress_size),
            "compress_type": int(info.compress_type),
            "sha256": sha256_bytes(data),
        }


def _segment_profile(name: str, data: bytes, offset: int) -> dict[str, Any]:
    contract = infer_pr85_segment_contract(name, data)
    entropy_bits = _entropy_bits_per_symbol(data)
    return {
        "name": name,
        "offset_in_x_bytes": int(offset),
        "offset_in_x_bits": int(offset) * 8,
        "bytes": len(data),
        "bits": len(data) * 8,
        "sha256": sha256_bytes(data),
        "magic": contract.magic,
        "codec": contract.codec,
        "metadata": dict(contract.metadata),
        "zero_order_entropy_bits_per_byte": round(entropy_bits, 12),
        "zero_order_entropy_bytes": int(math.ceil(entropy_bits * len(data) / 8.0)),
        "rate_score_contribution": len(data) * RATE_LAMBDA,
    }


def _archive_profile(path: Path, *, label: str) -> dict[str, Any]:
    member_name, x_data, member = _safe_single_member_zip(path)
    bundle = parse_pr85_bundle(x_data)
    segments = {
        name: _segment_profile(name, bytes(bundle.segments[name]), bundle.segment_offsets[name])
        for name in SEGMENT_ORDER
    }
    zip_overhead = path.stat().st_size - len(x_data)
    return {
        "label": label,
        "path": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_bits": path.stat().st_size * 8,
        "archive_sha256": sha256_path(path),
        "score_claim": False,
        "promotion_eligible": False,
        "member": member,
        "bundle_format": bundle.format,
        "x_header_bytes": bundle.header_bytes,
        "zip_and_container_overhead_bytes": zip_overhead,
        "zip_and_container_overhead_bits": zip_overhead * 8,
        "segments": segments,
        "rate_score_contribution": path.stat().st_size * RATE_LAMBDA,
    }


def build_report(
    left_archive: Path,
    right_archive: Path,
    *,
    left_label: str,
    right_label: str,
    target_score_buffer: float | None = None,
    compliance_json: Path | None = None,
) -> dict[str, Any]:
    left = _archive_profile(left_archive, label=left_label)
    right = _archive_profile(right_archive, label=right_label)
    rows: list[dict[str, Any]] = []
    for name in SEGMENT_ORDER:
        left_data = parse_pr85_bundle(_safe_single_member_zip(left_archive)[1]).segments[name]
        right_data = parse_pr85_bundle(_safe_single_member_zip(right_archive)[1]).segments[name]
        left_prof = left["segments"][name]
        right_prof = right["segments"][name]
        delta_bytes = int(right_prof["bytes"]) - int(left_prof["bytes"])
        common_prefix = _common_prefix_len(left_data, right_data)
        hamming = _hamming_bits_common_prefix(left_data, right_data)
        rows.append(
            {
                "name": name,
                "left_bytes": left_prof["bytes"],
                "right_bytes": right_prof["bytes"],
                "delta_bytes_right_minus_left": delta_bytes,
                "delta_bits_right_minus_left": delta_bytes * 8,
                "delta_rate_score_right_minus_left": delta_bytes * RATE_LAMBDA,
                "same_sha256": left_prof["sha256"] == right_prof["sha256"],
                "left_codec": left_prof["codec"],
                "right_codec": right_prof["codec"],
                "left_sha256": left_prof["sha256"],
                "right_sha256": right_prof["sha256"],
                "common_prefix_bytes": common_prefix,
                "common_prefix_bits": common_prefix * 8,
                "hamming_bits_over_common_prefix": hamming,
                "first_diff": _first_diff(left_data, right_data),
                "left_hpm1_metadata": left_prof["metadata"] if left_prof["codec"] == "HPM1" else None,
                "right_hpm1_metadata": right_prof["metadata"] if right_prof["codec"] == "HPM1" else None,
            }
        )

    archive_delta = int(right["archive_bytes"]) - int(left["archive_bytes"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "no_score_claim": NO_SCORE_CLAIM,
        "rate_lambda_score_per_byte": RATE_LAMBDA,
        "rate_lambda_score_per_bit": RATE_LAMBDA / 8.0,
        "left": left,
        "right": right,
        "comparison": {
            "left_label": left_label,
            "right_label": right_label,
            "archive_delta_bytes_right_minus_left": archive_delta,
            "archive_delta_bits_right_minus_left": archive_delta * 8,
            "archive_delta_rate_score_right_minus_left": archive_delta * RATE_LAMBDA,
            "segment_rows": rows,
        },
    }
    if target_score_buffer is not None:
        bytes_needed = int(math.ceil(float(target_score_buffer) / RATE_LAMBDA))
        report["lagrangian_target"] = {
            "target_score_buffer": float(target_score_buffer),
            "neutral_bytes_needed_for_buffer": bytes_needed,
            "neutral_bits_needed_for_buffer": bytes_needed * 8,
            "interpretation": (
                "If SegNet/PoseNet remain unchanged, this many additional charged "
                "bytes must be removed to beat the target by the requested score buffer."
            ),
        }
    if compliance_json is not None and compliance_json.exists():
        payload = json.loads(compliance_json.read_text(encoding="utf-8"))
        report["compliance_signal"] = {
            "source_json": str(compliance_json),
            "status": payload.get("status"),
            "failure_stage": payload.get("failure_stage"),
            "failure_reason": payload.get("failure_reason"),
            "dispatch_unlocked": payload.get("dispatch_unlocked"),
            "local_decode_byte_parity_proven": payload.get("local_decode_byte_parity_proven"),
            "score_claim": False,
        }
    return report


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    comp = report["comparison"]
    lines = [
        "# PR85-Family Bit/Lagrangian Profile",
        "",
        str(report["no_score_claim"]),
        "",
        f"- Left: `{comp['left_label']}`",
        f"- Right: `{comp['right_label']}`",
        f"- Archive delta right-left: `{comp['archive_delta_bytes_right_minus_left']}` bytes "
        f"(`{comp['archive_delta_rate_score_right_minus_left']:.12f}` score rate)",
        "",
        "| segment | left bytes | right bytes | delta bytes | delta score | same sha | left codec | right codec |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in comp["segment_rows"]:
        lines.append(
            f"| `{row['name']}` | {row['left_bytes']} | {row['right_bytes']} | "
            f"{row['delta_bytes_right_minus_left']} | "
            f"{row['delta_rate_score_right_minus_left']:.12f} | "
            f"{row['same_sha256']} | `{row['left_codec']}` | `{row['right_codec']}` |"
        )
    if "lagrangian_target" in report:
        target = report["lagrangian_target"]
        lines += [
            "",
            "## Lagrangian Target",
            "",
            f"- Target score buffer: `{target['target_score_buffer']}`",
            f"- Neutral bytes needed: `{target['neutral_bytes_needed_for_buffer']}`",
        ]
    if "compliance_signal" in report:
        signal = report["compliance_signal"]
        lines += [
            "",
            "## Compliance Signal",
            "",
            f"- Source JSON: `{signal['source_json']}`",
            f"- Status: `{signal.get('status')}`",
            f"- Failure: `{signal.get('failure_stage')}` / `{signal.get('failure_reason')}`",
            f"- Dispatch unlocked: `{signal.get('dispatch_unlocked')}`",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left_archive", type=Path)
    parser.add_argument("right_archive", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--target-score-buffer", type=float, default=None)
    parser.add_argument("--compliance-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(
        args.left_archive,
        args.right_archive,
        left_label=args.left_label,
        right_label=args.right_label,
        target_score_buffer=args.target_score_buffer,
        compliance_json=args.compliance_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(_json_bytes(report))
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.output_md)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": str(args.output_json),
                "output_md": None if args.output_md is None else str(args.output_md),
                "archive_delta_bytes_right_minus_left": report["comparison"][
                    "archive_delta_bytes_right_minus_left"
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
