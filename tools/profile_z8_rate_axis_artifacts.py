#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile Z8HPC1 artifacts by contest-rate bytes and archive grammar section.

This is byte-only evidence. It does not inflate frames, load scorers, or promote
scores. Its job is to keep the Z8 rate-axis discussion grounded in the bytes the
contest archive actually pays for: the ZIP packet first, and the inner Z8HPC1
``0.bin`` grammar second.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    summarize_wavelet_blob_detail_codecs,
)

SCHEMA = "z8_rate_axis_artifact_profile.v1"
TOOL = "tools/profile_z8_rate_axis_artifacts.py"


@dataclass(frozen=True)
class _ArtifactSpec:
    label: str
    path: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_artifact_spec(raw: str) -> _ArtifactSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return _ArtifactSpec(label=label.strip(), path=Path(path).expanduser())
    path = Path(raw).expanduser()
    return _ArtifactSpec(label=path.parent.name or path.name, path=path)


def _section_profile(archive_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    sections = parse_z8hpc1_archive_bytes(archive_bytes)
    total = len(archive_bytes)
    rows: list[dict[str, Any]] = []
    wavelet_blob: bytes | None = None
    for name, (start, length) in sections.items():
        rows.append(
            {
                "section": name,
                "start": int(start),
                "bytes": int(length),
                "share_of_0bin": round(float(length) / float(total), 8) if total else 0.0,
                "contest_rate_term_if_paid_directly": contest_rate_term(int(length)),
            }
        )
        if name == "wavelet_blob":
            wavelet_blob = archive_bytes[start : start + length]
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    detail_summary = (
        summarize_wavelet_blob_detail_codecs(wavelet_blob)
        if wavelet_blob is not None
        else None
    )
    return rows, detail_summary


def _zip_profile(zip_path: Path, *, expected_member_sha256: str | None) -> dict[str, Any]:
    zip_bytes = zip_path.read_bytes()
    members: list[dict[str, Any]] = []
    total_compressed = 0
    total_uncompressed = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_error: str | None = None
            sha: str | None = None
            try:
                payload = zf.read(info.filename)
                sha = _sha256_bytes(payload)
            except zipfile.BadZipFile as exc:
                member_error = str(exc)
            total_compressed += int(info.compress_size)
            total_uncompressed += int(info.file_size)
            members.append(
                {
                    "filename": info.filename,
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "compression_type": int(info.compress_type),
                    "sha256": sha,
                    "read_error": member_error,
                    "matches_expected_member_sha256": (
                        sha == expected_member_sha256
                        if sha is not None and expected_member_sha256 is not None
                        else None
                    ),
                }
            )
    members.sort(key=lambda row: int(row["compressed_bytes"]), reverse=True)
    zero_bin_members = [row for row in members if row["filename"] == "0.bin"]
    member_read_errors = [row for row in members if row.get("read_error") is not None]
    zero_bin_member_matches_expected = (
        len(zero_bin_members) == 1
        and zero_bin_members[0].get("matches_expected_member_sha256") is True
    )
    return {
        "path": zip_path.as_posix(),
        "bytes": len(zip_bytes),
        "sha256": _sha256_bytes(zip_bytes),
        "contest_rate_term": contest_rate_term(len(zip_bytes)),
        "member_count": len(members),
        "members": members,
        "total_compressed_member_bytes": int(total_compressed),
        "total_uncompressed_member_bytes": int(total_uncompressed),
        "zip_container_overhead_bytes": len(zip_bytes) - int(total_compressed),
        "member_read_errors": member_read_errors,
        "zero_bin_member_count": len(zero_bin_members),
        "zero_bin_member_matches_expected": zero_bin_member_matches_expected,
        "zip_custody_ok": zero_bin_member_matches_expected and not member_read_errors,
    }


def _associated_zip(path: Path) -> Path | None:
    candidates = [
        path.with_name("archive.zip"),
        path.parent / "submission" / "archive.zip",
        path.parent.parent / "archive.zip",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _nearby_manifest_fields(path: Path) -> list[dict[str, Any]]:
    wanted_names = {
        "z8_joint_p18_p19_deadzone_manifest.json",
        "z8_joint_p18_p19_relinearized_search_manifest.json",
        "z8_hpc1_runtime_payload_bridge_report.json",
        "decode_benchmark.json",
        "local_submission_replay_summary.json",
    }
    rows: list[dict[str, Any]] = []
    for base in (path.parent, path.parent.parent):
        if not base.exists():
            continue
        for manifest in sorted(p for p in base.glob("*.json") if p.name in wanted_names):
            try:
                payload = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(
                {
                    "path": manifest.as_posix(),
                    "sha256": _sha256_file(manifest),
                    "schema": payload.get("schema"),
                    "score_claim": payload.get("score_claim"),
                    "ready_for_exact_eval_dispatch": payload.get(
                        "ready_for_exact_eval_dispatch"
                    ),
                    "promotion_eligible": payload.get("promotion_eligible"),
                    "blockers": payload.get("blockers"),
                }
            )
    return rows


def _profile_one(spec: _ArtifactSpec) -> dict[str, Any]:
    path = spec.path.resolve()
    if path.is_dir():
        path = path / "0.bin"
    archive_bytes = path.read_bytes()
    archive_sha = _sha256_bytes(archive_bytes)
    sections, detail_summary = _section_profile(archive_bytes)
    wavelet = next((row for row in sections if row["section"] == "wavelet_blob"), None)
    zip_path = _associated_zip(path)
    zip_record = (
        _zip_profile(zip_path, expected_member_sha256=archive_sha)
        if zip_path is not None
        else None
    )
    paid_bytes = (
        int(zip_record["bytes"])
        if zip_record is not None and zip_record.get("zip_custody_ok")
        else len(archive_bytes)
    )
    detail_payload_bytes = (
        int(detail_summary.get("total_detail_payload_bytes", 0))
        if isinstance(detail_summary, dict)
        else None
    )
    return {
        "label": spec.label,
        "path": path.as_posix(),
        "sha256": archive_sha,
        "z8_0bin_bytes": len(archive_bytes),
        "z8_0bin_contest_rate_term_if_paid_directly": contest_rate_term(len(archive_bytes)),
        "associated_zip": zip_record,
        "contest_zip_candidate_bytes": (
            int(zip_record["bytes"]) if zip_record is not None else None
        ),
        "contest_zip_custody_ok": (
            bool(zip_record.get("zip_custody_ok")) if zip_record is not None else None
        ),
        "contest_paid_bytes_best_available": paid_bytes,
        "contest_paid_rate_term_best_available": contest_rate_term(paid_bytes),
        "sections": sections,
        "dominant_section": sections[0]["section"] if sections else None,
        "dominant_section_bytes": sections[0]["bytes"] if sections else None,
        "wavelet_blob_share_of_0bin": (
            float(wavelet["share_of_0bin"]) if wavelet is not None else None
        ),
        "wavelet_detail_codec_summary": detail_summary,
        "detail_payload_bytes": detail_payload_bytes,
        "detail_payload_share_of_wavelet_blob": (
            round(detail_payload_bytes / float(wavelet["bytes"]), 8)
            if detail_payload_bytes is not None and wavelet is not None and wavelet["bytes"]
            else None
        ),
        "nearby_manifest_fields": _nearby_manifest_fields(path),
    }


def _opportunities(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not artifacts:
        return rows
    best_paid = min(artifacts, key=lambda row: int(row["contest_paid_bytes_best_available"]))
    best_0bin = min(artifacts, key=lambda row: int(row["z8_0bin_bytes"]))
    largest = max(artifacts, key=lambda row: int(row["contest_paid_bytes_best_available"]))
    valid_zip_artifacts = [
        row for row in artifacts if (row.get("associated_zip") or {}).get("zip_custody_ok")
    ]
    best_valid_zip = (
        min(valid_zip_artifacts, key=lambda row: int(row["associated_zip"]["bytes"]))
        if valid_zip_artifacts
        else None
    )
    target_rate_term_020_bytes = int(0.20 * CONTEST_ORIGINAL_BYTES / 25)
    rows.append(
        {
            "surface": "contest_zip_bytes_are_authority",
            "priority": 0,
            "finding": (
                "Rank candidates by archive.zip bytes when a valid sibling ZIP exists; "
                "inner 0.bin bytes can be the wrong objective when outer ZIP exploits "
                "structured zeros or is blocked by already-random entropy-coded payloads."
            ),
            "evidence": {
                "best_paid_label": best_paid["label"],
                "best_paid_bytes": best_paid["contest_paid_bytes_best_available"],
                "best_0bin_label": best_0bin["label"],
                "best_0bin_bytes": best_0bin["z8_0bin_bytes"],
                "best_valid_zip_label": (
                    best_valid_zip["label"] if best_valid_zip is not None else None
                ),
                "best_valid_zip_bytes": (
                    best_valid_zip["associated_zip"]["bytes"]
                    if best_valid_zip is not None
                    else None
                ),
                "largest_paid_label": largest["label"],
                "largest_paid_bytes": largest["contest_paid_bytes_best_available"],
            },
        }
    )
    if best_valid_zip is not None:
        rows.append(
            {
                "surface": "z8_rate_gap_to_competitive_range",
                "priority": 0,
                "finding": (
                    "The best custody-valid Z8 ZIP is still far above the byte range "
                    "where rate stops dominating. Incremental entropy-mode polishing "
                    "cannot close this alone; the next lever must remove or generate "
                    "most residual wavelet coefficients, not merely recode them."
                ),
                "evidence": {
                    "best_valid_zip_label": best_valid_zip["label"],
                    "best_valid_zip_bytes": best_valid_zip["associated_zip"]["bytes"],
                    "target_bytes_for_rate_term_0_20": target_rate_term_020_bytes,
                    "multiple_over_rate_term_0_20": round(
                        best_valid_zip["associated_zip"]["bytes"]
                        / float(target_rate_term_020_bytes),
                        2,
                    ),
                },
            }
        )
    for row in artifacts:
        detail = row.get("wavelet_detail_codec_summary") or {}
        zip_record = row.get("associated_zip") or {}
        if row.get("wavelet_blob_share_of_0bin", 0.0) and row["wavelet_blob_share_of_0bin"] > 0.90:
            rows.append(
                {
                    "surface": "wavelet_blob_dominates_0bin",
                    "priority": 1,
                    "label": row["label"],
                    "finding": (
                        "The Z8 rate axis is still a wavelet-pyramid byte problem; "
                        "decoder, Dreamer state, indices, and meta are second-order."
                    ),
                    "evidence": {
                        "wavelet_blob_share_of_0bin": row["wavelet_blob_share_of_0bin"],
                        "z8_0bin_bytes": row["z8_0bin_bytes"],
                    },
                }
            )
        if int(detail.get("float32_detail_subband_count", 0)) > 0:
            rows.append(
                {
                    "surface": "float32_detail_storage",
                    "priority": 2,
                    "label": row["label"],
                    "finding": (
                        "Raw float detail subbands remain: quantize/byteplane/RLE/range "
                        "mode selection is still a first-order compression opportunity, "
                        "but must be accepted by full replay because coefficients are scorer-active."
                    ),
                    "evidence": {
                        "float32_detail_subband_count": detail.get(
                            "float32_detail_subband_count"
                        ),
                        "total_detail_subbands": detail.get("total_detail_subbands"),
                    },
                }
            )
        methods = detail.get("detail_codec_method_counts") or {}
        if methods == {"qi16_static_range": detail.get("total_detail_subbands")}:
            cheaper = [
                other
                for other in artifacts
                if int(other["contest_paid_bytes_best_available"])
                < int(row["contest_paid_bytes_best_available"])
            ]
            if cheaper:
                rows.append(
                    {
                        "surface": "forced_range_coding_regression",
                        "priority": 1,
                        "label": row["label"],
                        "finding": (
                            "Forcing one entropy mode globally loses against the "
                            "per-subband portfolio. Sparse bands want run/byte-plane "
                            "structure; dense bands can use range coding."
                        ),
                        "evidence": {
                            "forced_range_paid_bytes": row[
                                "contest_paid_bytes_best_available"
                            ],
                            "cheapest_observed_label": min(
                                cheaper,
                                key=lambda other: int(
                                    other["contest_paid_bytes_best_available"]
                                ),
                            )["label"],
                        },
                    }
                )
        if (
            methods == {"qi16_zero_rle": detail.get("total_detail_subbands")}
            and row.get("contest_zip_custody_ok") is None
        ):
            rows.append(
                {
                    "surface": "smallest_inner_packet_missing_zip_receiver_proof",
                    "priority": 0,
                    "label": row["label"],
                    "finding": (
                        "The smallest observed inner Z8 packet is not yet a custody-valid "
                        "contest ZIP in this profile. Highest-EV next action is package, "
                        "inflate-proof, and full-replay it before inventing a new coder."
                    ),
                    "evidence": {
                        "z8_0bin_bytes": row["z8_0bin_bytes"],
                        "contest_rate_term_if_paid_directly": row[
                            "z8_0bin_contest_rate_term_if_paid_directly"
                        ],
                    },
                }
            )
        if zip_record and zip_record.get("zip_custody_ok"):
            zero_bin = next(
                (
                    member
                    for member in zip_record.get("members", [])
                    if member.get("filename") == "0.bin"
                ),
                None,
            )
            if zero_bin is not None:
                runtime_overhead = int(zip_record["bytes"]) - int(
                    zero_bin["compressed_bytes"]
                )
                if runtime_overhead > int(0.05 * CONTEST_ORIGINAL_BYTES / 25):
                    rows.append(
                        {
                            "surface": "runtime_payload_overhead_future_dominates",
                            "priority": 2,
                            "label": row["label"],
                            "finding": (
                                "The Python runtime bundle is small relative to today's "
                                "wavelet blob, but it is already larger than a 0.05 rate-term "
                                "budget. Once residual bytes collapse, runtime tree-shaking or "
                                "a thinner adapter becomes first-order."
                            ),
                            "evidence": {
                                "zip_bytes": zip_record["bytes"],
                                "compressed_0bin_member_bytes": zero_bin[
                                    "compressed_bytes"
                                ],
                                "runtime_and_zip_overhead_bytes": runtime_overhead,
                            },
                        }
                    )
        if zip_record and not zip_record.get("zip_custody_ok"):
            rows.append(
                {
                    "surface": "zip_custody_mismatch",
                    "priority": 0,
                    "label": row["label"],
                    "finding": (
                        "Associated archive.zip does not prove it contains the profiled 0.bin. "
                        "Treat the rate comparison as blocked until custody is repaired."
                    ),
                    "evidence": {
                        "zip_path": zip_record.get("path"),
                        "member_count": zip_record.get("member_count"),
                        "zero_bin_member_count": zip_record.get("zero_bin_member_count"),
                        "zero_bin_member_matches_expected": zip_record.get(
                            "zero_bin_member_matches_expected"
                        ),
                        "member_read_error_count": len(
                            zip_record.get("member_read_errors") or []
                        ),
                    },
                }
            )
    rows.sort(key=lambda item: int(item.get("priority", 99)))
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Z8 Rate-Axis Byte Profile",
        "",
        "Byte-only advisory profile. No scorer or promotion authority.",
        "",
        "## Contest Rate Targets",
        "",
        "| Rate term | Archive bytes |",
        "|---:|---:|",
    ]
    for rate_term in (0.05, 0.10, 0.20, 1.0):
        lines.append(
            f"| {rate_term:.2f} | {int(rate_term * CONTEST_ORIGINAL_BYTES / 25):,} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Ranking",
            "",
            "| Label | paid bytes | paid rate term | 0.bin bytes | dominant section | wavelet share | zip custody |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in sorted(
        report["artifacts"], key=lambda item: int(item["contest_paid_bytes_best_available"])
    ):
        zip_record = row.get("associated_zip") or {}
        zip_ok = zip_record.get("zip_custody_ok")
        lines.append(
            "| {label} | {paid:,} | {rate:.4f} | {inner:,} | {dom} | {share:.2%} | {zip_ok} |".format(
                label=row["label"],
                paid=int(row["contest_paid_bytes_best_available"]),
                rate=float(row["contest_paid_rate_term_best_available"]),
                inner=int(row["z8_0bin_bytes"]),
                dom=row.get("dominant_section"),
                share=float(row.get("wavelet_blob_share_of_0bin") or 0.0),
                zip_ok=zip_ok,
            )
        )
    lines.extend(["", "## Largest Sections", ""])
    for row in report["artifacts"]:
        lines.append(f"### {row['label']}")
        lines.append("")
        lines.append("| Section | bytes | share of 0.bin | rate term if direct |")
        lines.append("|---|---:|---:|---:|")
        for section in row["sections"][:8]:
            lines.append(
                "| {section} | {bytes:,} | {share:.2%} | {rate:.4f} |".format(
                    section=section["section"],
                    bytes=int(section["bytes"]),
                    share=float(section["share_of_0bin"]),
                    rate=float(section["contest_rate_term_if_paid_directly"]),
                )
            )
        detail = row.get("wavelet_detail_codec_summary") or {}
        if detail:
            lines.append("")
            lines.append(
                "Detail codec methods: "
                + json.dumps(detail.get("detail_codec_method_counts", {}), sort_keys=True)
            )
        lines.append("")
    lines.extend(["## Opportunities", "", "| Priority | Surface | Finding |", "|---:|---|---|"])
    for opp in report["opportunities"]:
        lines.append(
            f"| {opp.get('priority')} | {opp.get('surface')} | {str(opp.get('finding')).replace('|', '/')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_report(specs: list[_ArtifactSpec]) -> dict[str, Any]:
    artifacts = [_profile_one(spec) for spec in specs]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": "byte_profile_only",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "contest_original_bytes": CONTEST_ORIGINAL_BYTES,
        "artifacts": artifacts,
        "opportunities": _opportunities(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Artifact path or label=/path/to/0.bin. Directories resolve to directory/0.bin.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    report = build_report([_parse_artifact_spec(raw) for raw in args.artifact])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(_json_text(report), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_render_markdown(report), encoding="utf-8")
    best = min(
        report["artifacts"], key=lambda row: int(row["contest_paid_bytes_best_available"])
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "artifact_count": len(report["artifacts"]),
                "best_paid_label": best["label"],
                "best_paid_bytes": best["contest_paid_bytes_best_available"],
                "best_paid_rate_term": best["contest_paid_rate_term_best_available"],
                "out_json": args.out_json.as_posix(),
                "out_md": args.out_md.as_posix() if args.out_md else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
