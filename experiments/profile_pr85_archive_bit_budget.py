#!/usr/bin/env python3
"""Profile the PR85 archive bit budget without making score claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/profile_pr85_archive_bit_budget.py"
SCHEMA = "pr85_archive_bit_budget_profile_v1"
EVIDENCE_GRADE = "planning_only_static_archive_accounting"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

SEGMENT_ORDER = (
    "mask",
    "model",
    "pose",
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
FIXED_V5_LENGTHS = {"bias": 223, "region": 273}
POLICY_SEGMENTS = {
    "minus_post": ("post",),
    "minus_randmulti": ("randmulti",),
    "minus_motion_stack": ("shift", "frac", "frac2", "frac3"),
    "minus_post_motion": ("post", "shift", "frac", "frac2", "frac3"),
    "minus_all_safe_corrections": ("post", "shift", "frac", "frac2", "frac3", "randmulti"),
    "whole_post_deletion": ("post",),
    "whole_randmulti_deletion": ("randmulti",),
    "whole_motion_deletion": ("shift", "frac", "frac2", "frac3"),
    "whole_post_motion_deletion": ("post", "shift", "frac", "frac2", "frac3"),
    "protected_qpost_group_deletion": ("post", "shift", "frac", "frac2", "frac3"),
}

DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_PROFILE_JSON = DEFAULT_INTAKE_DIR / "profile_pr85_bundle.json"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr85_archive_bit_budget_20260504_codex"
DEFAULT_JSON_OUT = DEFAULT_OUT_DIR / "profile_pr85_archive_bit_budget.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT_DIR / "profile_pr85_archive_bit_budget.md"
DEFAULT_SIDECHANNEL_SUMMARIES = (
    REPO_ROOT / "experiments/results/public_pr85_sidechannel_ablations_20260503_codex/candidate_summary.json",
    REPO_ROOT / "experiments/results/public_pr85_sidechannel_recodes_20260503_codex/candidate_summary.json",
    REPO_ROOT / "experiments/results/pr85_randmulti_group_policy_candidates_20260504_codex/candidate_summary.json",
    REPO_ROOT / "experiments/results/pr85_post_motion_group_policy_candidates_20260504_codex/candidate_summary.json",
    REPO_ROOT / "experiments/results/pr85_bridge_sparse_action_candidates_20260504_codex/candidate_summary.json",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except (OSError, ValueError):
        if p.is_absolute():
            return f"<external>/{p.name}"
        return str(p)


def _rate_score(bytes_: int) -> float:
    return round(float(bytes_ * RATE_SCORE_PER_BYTE), 12)


def _safe_zip_member(name: str) -> str:
    path = Path(name)
    if name.startswith("/") or ".." in path.parts or not name:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_member_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [_safe_zip_member(info.filename) for info in infos]
        if len(infos) != 1:
            raise ValueError(f"{path}: expected exactly one non-directory member, got {names!r}")
        info = infos[0]
        raw = zf.read(info)
    archive_bytes = int(path.stat().st_size)
    member_bytes = int(len(raw))
    return (
        {
            "archive_path": _rel(path),
            "archive_bytes": archive_bytes,
            "archive_bits": int(archive_bytes * 8),
            "archive_sha256": _sha256_file(path),
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_bits": int(member_bytes * 8),
            "member_sha256": _sha256_bytes(raw),
            "member_crc32_hex": f"{info.CRC:08x}",
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
            "zip_container_overhead_bytes": int(archive_bytes - member_bytes),
            "zip_container_overhead_bits": int((archive_bytes - member_bytes) * 8),
        },
        raw,
    )


def _u24le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def _parse_pr85_v5_bundle(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(raw) < 24:
        raise ValueError("PR85 member is too short for the v5 micro header")
    lengths = {
        "mask": _u24le(raw, 0),
        "model": _u24le(raw, 3),
        "pose": _u24le(raw, 6),
        "post": _u24le(raw, 9),
        "shift": _u24le(raw, 12),
        "frac": _u24le(raw, 15),
        "frac2": _u24le(raw, 18),
        "frac3": _u24le(raw, 21),
        **FIXED_V5_LENGTHS,
    }
    pos = 24
    offsets: dict[str, int] = {}
    segments: dict[str, bytes] = {}
    for name in SEGMENT_ORDER[:-1]:
        size = int(lengths[name])
        if size <= 0:
            raise ValueError(f"invalid PR85 segment length for {name}: {size}")
        end = pos + size
        if end > len(raw):
            raise ValueError(f"truncated PR85 segment {name}")
        offsets[name] = pos
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise ValueError("PR85 member is missing the randmulti tail")
    offsets["randmulti"] = pos
    segments["randmulti"] = raw[pos:]
    return (
        {
            "format": "pr85_v5_micro_24bit_lengths_fixed_bias_region",
            "header_bytes": 24,
            "header_bits": 192,
            "fixed_length_segments": dict(FIXED_V5_LENGTHS),
            "segment_offsets": offsets,
            "segment_lengths": {name: len(segments[name]) for name in SEGMENT_ORDER},
        },
        segments,
    )


def _empty_flags() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sidechannel_candidate_ids": [],
            "deletion_candidate_ids": [],
            "deletion_screened": False,
            "exact_deletion_negative": False,
            "protected": False,
            "protected_reasons": [],
            "deletion_negative_context": [],
            "best_lossless_recode_delta_bytes": None,
            "lossless_recode_candidate_ids": [],
            "dispatch_preflight_blocker_ids": [],
        }
        for name in SEGMENT_ORDER
    }


def _append_unique(items: list[Any], value: Any) -> None:
    if value not in items:
        items.append(value)


def _segments_for_policy(policy_id: str | None, candidate: dict[str, Any]) -> tuple[str, ...]:
    if policy_id in POLICY_SEGMENTS:
        return POLICY_SEGMENTS[str(policy_id)]
    if isinstance(candidate.get("neutralized_segments"), list):
        return tuple(str(item) for item in candidate["neutralized_segments"] if item in SEGMENT_ORDER)
    if isinstance(candidate.get("changed_segments"), list):
        return tuple(str(item) for item in candidate["changed_segments"] if item in SEGMENT_ORDER)
    segments = []
    for transform in candidate.get("transforms", []):
        if isinstance(transform, dict) and transform.get("segment") in SEGMENT_ORDER:
            segments.append(str(transform["segment"]))
    return tuple(segments)


def _mark_negative_context(
    flags: dict[str, dict[str, Any]],
    *,
    context_id: str,
    context: dict[str, Any],
    source_summary: Path,
) -> None:
    for segment in POLICY_SEGMENTS.get(context_id, ()):
        row = flags[segment]
        row["exact_deletion_negative"] = True
        row["protected"] = True
        _append_unique(row["protected_reasons"], str(context.get("role", "whole-stream deletion negative")))
        _append_unique(
            row["deletion_negative_context"],
            {
                "context_id": context_id,
                "score": context.get("score"),
                "net_score_delta_vs_pr85": context.get("net_score_delta_vs_pr85"),
                "bytes_saved_vs_pr85": context.get("bytes_saved_vs_pr85"),
                "evidence_grade": context.get("evidence_grade"),
                "role": context.get("role"),
                "auth_json": context.get("auth_json"),
                "source_summary": _rel(source_summary),
            },
        )


def collect_sidechannel_flags(summary_paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    flags = _empty_flags()
    inputs: list[str] = []
    for path in summary_paths:
        if not path.is_file():
            continue
        inputs.append(str(_rel(path)))
        summary = _read_json(path)
        for candidate in summary.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            policy_id = candidate.get("policy_id") or candidate.get("candidate_id") or candidate.get("id")
            policy_id = str(policy_id) if policy_id is not None else "unknown_candidate"
            segments = _segments_for_policy(policy_id, candidate)
            candidate_text = " ".join(
                str(candidate.get(key, ""))
                for key in ("schema", "tool", "evidence_grade")
            ).lower()
            is_deletion_candidate = policy_id.startswith("minus_") or bool(candidate.get("neutralized_segments"))
            is_lossless_recode = "recode" in candidate_text and not is_deletion_candidate
            for segment in segments:
                row = flags[segment]
                _append_unique(row["sidechannel_candidate_ids"], policy_id)
                if is_lossless_recode and candidate.get("byte_delta_vs_source_archive") is not None:
                    delta = int(candidate["byte_delta_vs_source_archive"])
                    best = row["best_lossless_recode_delta_bytes"]
                    if best is None or delta < int(best):
                        row["best_lossless_recode_delta_bytes"] = delta
                    _append_unique(row["lossless_recode_candidate_ids"], policy_id)
            if is_deletion_candidate:
                for segment in segments:
                    row = flags[segment]
                    row["deletion_screened"] = True
                    _append_unique(row["deletion_candidate_ids"], policy_id)
            for context_id, context in candidate.get("whole_stream_negative_context", {}).items():
                if isinstance(context, dict):
                    _mark_negative_context(flags, context_id=str(context_id), context=context, source_summary=path)
            preflight = candidate.get("dispatch_preflight")
            if isinstance(preflight, dict):
                for blocker_id in preflight.get("blocker_ids", []):
                    blocker_id = str(blocker_id)
                    for segment in POLICY_SEGMENTS.get(blocker_id, ()):
                        row = flags[segment]
                        row["protected"] = True
                        _append_unique(row["dispatch_preflight_blocker_ids"], blocker_id)
                        _append_unique(row["protected_reasons"], blocker_id)
    for row in flags.values():
        row["sidechannel_candidate_ids"].sort()
        row["deletion_candidate_ids"].sort()
        row["lossless_recode_candidate_ids"].sort()
        row["dispatch_preflight_blocker_ids"].sort()
        row["protected_reasons"].sort()
        row["lossless_recode_non_improving"] = (
            row["best_lossless_recode_delta_bytes"] is not None
            and int(row["best_lossless_recode_delta_bytes"]) >= 0
        )
    return flags, inputs


def _profile_segments(
    *,
    segments: dict[str, bytes],
    bundle: dict[str, Any],
    archive_info: dict[str, Any],
    profile_json: dict[str, Any] | None,
    flags: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile_rows = {}
    if profile_json:
        profile_rows = {
            str(row.get("name")): row
            for row in profile_json.get("segments", [])
            if isinstance(row, dict) and row.get("name")
        }
    member_bytes = int(archive_info["member_file_size"])
    archive_bytes = int(archive_info["archive_bytes"])
    mismatches = []
    rows = []
    for name in SEGMENT_ORDER:
        raw = segments[name]
        profile_row = profile_rows.get(name, {})
        profile_bytes = profile_row.get("bytes")
        profile_sha = profile_row.get("sha256") or profile_row.get("raw_sha256")
        sha = _sha256_bytes(raw)
        if profile_bytes is not None and int(profile_bytes) != len(raw):
            mismatches.append({"segment": name, "field": "bytes", "profile": int(profile_bytes), "archive": len(raw)})
        if profile_sha is not None and str(profile_sha) != sha:
            mismatches.append({"segment": name, "field": "sha256", "profile": str(profile_sha), "archive": sha})
        row_flags = flags[name]
        rows.append(
            {
                "name": name,
                "offset": int(bundle["segment_offsets"][name]),
                "bytes": int(len(raw)),
                "bits": int(len(raw) * 8),
                "sha256": sha,
                "archive_byte_share": round(float(len(raw) / archive_bytes), 9),
                "member_byte_share": round(float(len(raw) / member_bytes), 9),
                "formula_only_rate_score_contribution": _rate_score(len(raw)),
                "known_flags": row_flags,
            }
        )
    rows_by_size = sorted(rows, key=lambda item: (-int(item["bytes"]), str(item["name"])))
    cumulative = 0
    for rank, row in enumerate(rows_by_size, start=1):
        cumulative += int(row["bytes"])
        row["rank_by_bytes"] = rank
        row["cumulative_bytes_by_desc"] = cumulative
        row["cumulative_bits_by_desc"] = cumulative * 8
        row["cumulative_archive_byte_share_by_desc"] = round(float(cumulative / archive_bytes), 9)
        row["cumulative_member_byte_share_by_desc"] = round(float(cumulative / member_bytes), 9)
    by_name = {row["name"]: row for row in rows_by_size}
    return [by_name[name] for name in SEGMENT_ORDER], mismatches


def _opportunity_class(name: str, row: dict[str, Any]) -> tuple[str, str]:
    flags = row["known_flags"]
    if name == "mask":
        return ("large_core_payload_entropy_surface", "QMA9 mask payload is the largest charged segment.")
    if name == "model":
        return ("large_core_payload_model_surface", "Joint frame model bytes are the second largest charged segment.")
    if name == "randmulti":
        return (
            "protected_group_waterfill_surface",
            "Whole randmulti deletion is negative; target group-level recoding or water-fill, not deletion.",
        )
    if flags["protected"]:
        return (
            "protected_recode_only_surface",
            "Deletion/preflight negatives protect this stream; only parity-preserving recodes are planning-safe.",
        )
    return ("micro_recode_surface", "Small charged stream; only deterministic parity-preserving byte work is justified.")


def _rank_opportunities(segment_rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(segment_rows, key=lambda item: (-int(item["bytes"]), str(item["name"])))
    out = []
    for rank, row in enumerate(ranked[:top_k], start=1):
        cls, rationale = _opportunity_class(str(row["name"]), row)
        out.append(
            {
                "rank": rank,
                "segment": row["name"],
                "bytes": row["bytes"],
                "bits": row["bits"],
                "archive_byte_share": row["archive_byte_share"],
                "formula_only_rate_score_contribution": row["formula_only_rate_score_contribution"],
                "opportunity_class": cls,
                "non_arbitrary_basis": rationale,
                "protected": row["known_flags"]["protected"],
                "exact_deletion_negative": row["known_flags"]["exact_deletion_negative"],
                "best_lossless_recode_delta_bytes": row["known_flags"]["best_lossless_recode_delta_bytes"],
            }
        )
    return out


def build_profile(
    archive: Path = DEFAULT_ARCHIVE,
    *,
    profile_json: Path | None = DEFAULT_PROFILE_JSON,
    sidechannel_summaries: list[Path] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    archive_info, raw = _read_single_member_archive(archive)
    bundle, segments = _parse_pr85_v5_bundle(raw)
    source_profile = _read_json(profile_json) if profile_json and profile_json.is_file() else None
    summaries = DEFAULT_SIDECHANNEL_SUMMARIES if sidechannel_summaries is None else tuple(sidechannel_summaries)
    flags, used_summaries = collect_sidechannel_flags(list(summaries))
    segment_rows, mismatches = _profile_segments(
        segments=segments,
        bundle=bundle,
        archive_info=archive_info,
        profile_json=source_profile,
        flags=flags,
    )
    archive_bytes = int(archive_info["archive_bytes"])
    member_bytes = int(archive_info["member_file_size"])
    header_bytes = int(bundle["header_bytes"])
    zip_overhead = int(archive_info["zip_container_overhead_bytes"])
    accounted_member_bytes = header_bytes + sum(int(row["bytes"]) for row in segment_rows)
    member_accounting_ok = accounted_member_bytes == member_bytes
    container_rows = [
        {
            "name": archive_info["member_name"],
            "kind": "zip_member",
            "bytes": member_bytes,
            "bits": member_bytes * 8,
            "archive_byte_share": round(float(member_bytes / archive_bytes), 9),
            "formula_only_rate_score_contribution": _rate_score(member_bytes),
        },
        {
            "name": "zip_container_overhead",
            "kind": "zip_overhead",
            "bytes": zip_overhead,
            "bits": zip_overhead * 8,
            "archive_byte_share": round(float(zip_overhead / archive_bytes), 9),
            "formula_only_rate_score_contribution": _rate_score(zip_overhead),
        },
        {
            "name": "pr85_v5_micro_header",
            "kind": "member_header",
            "bytes": header_bytes,
            "bits": header_bytes * 8,
            "member_byte_share": round(float(header_bytes / member_bytes), 9),
            "archive_byte_share": round(float(header_bytes / archive_bytes), 9),
            "formula_only_rate_score_contribution": _rate_score(header_bytes),
        },
    ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "deterministic": True,
        "score_rate_formula": {
            "formula_only": True,
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "score_claim_from_this_profile": False,
        },
        "inputs": {
            "archive": _rel(archive),
            "profile_json": _rel(profile_json) if profile_json else None,
            "sidechannel_summaries_used": used_summaries,
        },
        "archive": archive_info,
        "bundle": bundle,
        "container_rows": container_rows,
        "member_accounting": {
            "member_bytes": member_bytes,
            "header_bytes": header_bytes,
            "segment_bytes_total": int(sum(int(row["bytes"]) for row in segment_rows)),
            "accounted_member_bytes": accounted_member_bytes,
            "matches_member_file_size": member_accounting_ok,
            "profile_archive_mismatches": mismatches,
        },
        "segments": segment_rows,
        "opportunity_rankings": _rank_opportunities(segment_rows, top_k=top_k),
        "score_claim_refusal": {
            "reason": "Static byte accounting does not execute inflate or CUDA auth eval.",
            "score_claim": False,
            "dispatch_performed": False,
        },
    }


def render_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# PR85 Archive Bit-Budget Profile",
        "",
        f"- evidence_grade: `{profile['evidence_grade']}`",
        f"- planning_only: `{str(profile['planning_only']).lower()}`",
        f"- score_claim: `{str(profile['score_claim']).lower()}`",
        f"- dispatch_performed: `{str(profile['dispatch_performed']).lower()}`",
        f"- archive: `{profile['inputs']['archive']}`",
        f"- archive_bytes: `{profile['archive']['archive_bytes']}`",
        "",
        "## Top Byte Surfaces",
        "",
        "| rank | segment | bytes | share | protected | deletion-negative | basis |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in profile["opportunity_rankings"]:
        lines.append(
            "| {rank} | `{segment}` | {bytes} | {share:.6f} | `{protected}` | `{negative}` | {basis} |".format(
                rank=row["rank"],
                segment=row["segment"],
                bytes=row["bytes"],
                share=row["archive_byte_share"],
                protected=str(row["protected"]).lower(),
                negative=str(row["exact_deletion_negative"]).lower(),
                basis=row["non_arbitrary_basis"],
            )
        )
    lines.extend(
        [
            "",
            "## Segment Budget",
            "",
            "| rank | segment | bytes | bits | cumulative share | formula-only rate score |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(profile["segments"], key=lambda item: int(item["rank_by_bytes"])):
        lines.append(
            "| {rank} | `{name}` | {bytes} | {bits} | {share:.6f} | {rate:.12f} |".format(
                rank=row["rank_by_bytes"],
                name=row["name"],
                bytes=row["bytes"],
                bits=row["bits"],
                share=row["cumulative_archive_byte_share_by_desc"],
                rate=row["formula_only_rate_score_contribution"],
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--profile-json", type=Path, default=DEFAULT_PROFILE_JSON)
    parser.add_argument(
        "--sidechannel-summary",
        type=Path,
        action="append",
        default=None,
        help="Candidate summary JSON to fold into protected/deletion-negative flags. May be repeated.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)

    markdown_out = args.markdown_out
    if args.write_markdown and markdown_out is None:
        markdown_out = DEFAULT_MARKDOWN_OUT
    payload = build_profile(
        args.archive,
        profile_json=args.profile_json,
        sidechannel_summaries=args.sidechannel_summary,
        top_k=args.top_k,
    )
    text = _json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    if markdown_out:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(payload), encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
