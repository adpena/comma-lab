#!/usr/bin/env python3
"""Build a compact scorecard for public HNeRV frontier intake.

The scorecard joins exact CUDA replay artifacts with forensic payload profiles.
It does not evaluate archives and does not promote prediction-only claims; it is
only a deterministic decision table for production review and follow-on
optimization work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

ProfileIndex = dict[str, dict[str, dict[str, Any]]]
CandidateIndex = dict[str, dict[str, Any]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def maybe_load_json(path: Path | None) -> Any | None:
    if path is None or not path.is_file():
        return None
    return load_json(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_single_member_archive(archive: Path) -> dict[str, Any]:
    archive_blob = archive.read_bytes()
    with ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"{archive} has {len(infos)} file members; expected exactly one")
        member = infos[0]
        payload = zf.read(member.filename)
    return {
        "archive": str(archive),
        "archive_bytes": len(archive_blob),
        "archive_sha256": sha256_bytes(archive_blob),
        "member_name": member.filename,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_overhead_bytes": len(archive_blob) - len(payload),
    }


def profile_indexes(paths: list[Path] | None) -> ProfileIndex:
    by_archive_sha: dict[str, dict[str, Any]] = {}
    by_member_sha: dict[str, dict[str, Any]] = {}
    for path in paths or []:
        payload = maybe_load_json(path)
        if not payload:
            continue
        for item in payload:
            archive_sha = item.get("archive_sha256")
            member_sha = item.get("member_sha256")
            if isinstance(archive_sha, str):
                by_archive_sha[archive_sha] = item
            if isinstance(member_sha, str):
                by_member_sha.setdefault(member_sha, item)
    return {"archive_sha256": by_archive_sha, "member_sha256": by_member_sha}


def profile_by_sha(paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    return profile_indexes(paths)["archive_sha256"]


def candidate_indexes(paths: list[Path] | None) -> CandidateIndex:
    by_archive_sha: dict[str, dict[str, Any]] = {}
    for path in paths or []:
        payload = maybe_load_json(path)
        if not isinstance(payload, dict):
            continue
        candidate_sha = payload.get("candidate_archive_sha256")
        if isinstance(candidate_sha, str):
            by_archive_sha[candidate_sha] = {
                **payload,
                "candidate_manifest_path": str(path),
            }
    return by_archive_sha


def numeric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def inspect_eval_archive(
    path: Path,
    payload: dict[str, Any],
    archive_path: Path | None = None,
) -> dict[str, Any] | None:
    archive = archive_path or path.with_name("archive.zip")
    if not archive.is_file():
        return None
    inspected = inspect_single_member_archive(archive)
    provenance = payload.get("provenance") or {}
    expected_sha = provenance.get("archive_sha256")
    if isinstance(expected_sha, str) and inspected["archive_sha256"] != expected_sha:
        raise ValueError(
            f"{archive} SHA mismatch for {path}: "
            f"eval={expected_sha} archive={inspected['archive_sha256']}"
        )
    expected_bytes = payload.get("archive_size_bytes")
    if isinstance(expected_bytes, int) and inspected["archive_bytes"] != expected_bytes:
        raise ValueError(
            f"{archive} byte-size mismatch for {path}: "
            f"eval={expected_bytes} archive={inspected['archive_bytes']}"
        )
    return inspected


def matched_profile(
    archive_sha: str | None,
    inspected_archive: dict[str, Any] | None,
    indexes: ProfileIndex,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(archive_sha, str) and archive_sha in indexes["archive_sha256"]:
        return indexes["archive_sha256"][archive_sha], "archive_sha256"
    member_sha = inspected_archive.get("member_sha256") if inspected_archive else None
    if isinstance(member_sha, str) and member_sha in indexes["member_sha256"]:
        return indexes["member_sha256"][member_sha], "member_sha256"
    return {}, None


def normalize_profile_indexes(profiles: ProfileIndex | dict[str, dict[str, Any]]) -> ProfileIndex:
    if "archive_sha256" in profiles and "member_sha256" in profiles:
        return cast("ProfileIndex", profiles)
    return {
        "archive_sha256": cast("dict[str, dict[str, Any]]", profiles),
        "member_sha256": {},
    }


def _adjudication_meta(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("adjudication_provenance", "adjudication", "promotion"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _canonicality_blockers(payload: dict[str, Any], evidence_grade: str) -> list[str]:
    meta = _adjudication_meta(payload)
    blockers: list[str] = []
    lane_status = meta.get("lane_status") or payload.get("lane_status")
    if isinstance(lane_status, str) and "regression" in lane_status.lower():
        blockers.append(f"lane_status_{lane_status}")
    paper_claim_grade = meta.get("paper_claim_grade") or payload.get("paper_claim_grade")
    if isinstance(paper_claim_grade, str) and "negative" in paper_claim_grade.lower():
        blockers.append(f"paper_claim_grade_{paper_claim_grade}")
    evidence = str(payload.get("evidence_grade") or evidence_grade)
    if "negative" in evidence.lower():
        blockers.append(f"evidence_grade_{evidence}")
    if meta.get("promotion_eligible") is False or payload.get("promotion_eligible") is False:
        blockers.append("promotion_ineligible")
    if meta.get("regression_triggered") is True or payload.get("regression_triggered") is True:
        blockers.append("regression_triggered")
    if payload.get("score_claim_valid") is False:
        blockers.append("score_claim_invalid")
    return blockers


def row_from_eval(
    label: str,
    path: Path,
    profiles: ProfileIndex | dict[str, dict[str, Any]],
    candidates: CandidateIndex | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    provenance = payload.get("provenance") or {}
    sha = provenance.get("archive_sha256")
    indexes = normalize_profile_indexes(profiles)
    inspected_archive = inspect_eval_archive(path, payload)
    profile, profile_match_key = matched_profile(sha, inspected_archive, indexes)
    sections = profile.get("sections") or []
    largest_section = max(sections, key=lambda item: item.get("bytes", 0), default=None)
    top_sections = sorted(sections, key=lambda item: item.get("bytes", 0), reverse=True)[:3]
    payload_sha = (
        (inspected_archive or {}).get("member_sha256")
        or profile.get("member_sha256")
    )
    candidate_manifest = {}
    if isinstance(sha, str) and candidates:
        candidate_manifest = candidates.get(sha) or {}
    candidate_diff_audit = candidate_manifest.get("candidate_diff_audit")
    brotli_raw_equivalence = candidate_manifest.get("brotli_raw_equivalence")
    raw_equivalence_closed = (
        isinstance(brotli_raw_equivalence, list)
        and bool(brotli_raw_equivalence)
        and all(isinstance(item, dict) and item.get("raw_equal") is True for item in brotli_raw_equivalence)
    )
    evidence_grade = (
        "A++"
        if provenance.get("device") == "cuda"
        and provenance.get("gpu_t4_match")
        and payload.get("n_samples") == 600
        else "A"
    )
    canonicality_blockers = _canonicality_blockers(payload, evidence_grade)
    row = {
        "label": label,
        "evidence_grade": evidence_grade,
        "canonical_frontier_eligible": evidence_grade == "A++" and not canonicality_blockers,
        "canonicality_blockers": canonicality_blockers,
        "frontier_scope": "exact_local_cuda_custody",
        "score": numeric(payload, "score_recomputed_from_components"),
        "archive_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": sha,
        "avg_segnet_dist": numeric(payload, "avg_segnet_dist"),
        "avg_posenet_dist": numeric(payload, "avg_posenet_dist"),
        "score_seg_contribution": numeric(payload, "score_seg_contribution"),
        "score_pose_contribution": numeric(payload, "score_pose_contribution"),
        "score_rate_contribution": numeric(payload, "score_rate_contribution"),
        "gpu_model": provenance.get("gpu_model"),
        "runtime_tree_sha256": (provenance.get("inflate_runtime_manifest") or {}).get(
            "runtime_tree_sha256"
        ),
        "profile_kind": profile.get("kind"),
        "profile_match_key": profile_match_key,
        "profile_archive_sha256": profile.get("archive_sha256"),
        "profile_member_name": profile.get("member_name"),
        "zip_member": (inspected_archive or {}).get("member_name") or profile.get("member_name"),
        "payload_sha256": payload_sha,
        "member_bytes": (inspected_archive or {}).get("member_bytes") or profile.get("member_bytes"),
        "zip_overhead_bytes": (inspected_archive or {}).get("zip_overhead_bytes")
        or profile.get("zip_overhead_bytes"),
        "largest_payload_section": largest_section,
        "top_payload_sections": top_sections,
        "payload_sections": sections,
        "eval_artifact": str(path),
    }
    if candidate_manifest:
        row.update(
            {
                "candidate_manifest": candidate_manifest.get("candidate_manifest_path"),
                "candidate_manifest_match_key": "candidate_archive_sha256",
                "candidate_source_label": candidate_manifest.get("source_label"),
                "candidate_source_archive_sha256": candidate_manifest.get("source_archive_sha256"),
                "candidate_source_archive_bytes": candidate_manifest.get("source_archive_bytes"),
                "candidate_payload_sha256": candidate_manifest.get("candidate_payload_sha256"),
                "candidate_diff_audit": candidate_diff_audit,
                "brotli_raw_equivalence": brotli_raw_equivalence,
                "raw_equivalence_closed": raw_equivalence_closed,
                "frontier_scope": (
                    "exact_local_cuda_custody_lossless_repack_control"
                    if raw_equivalence_closed
                    else "exact_local_cuda_custody_candidate_manifest"
                ),
            }
        )
    return row


def equal_numeric(values: list[float | None]) -> bool:
    concrete = [value for value in values if value is not None]
    return len(concrete) == len(values) and len(set(concrete)) <= 1


def payload_equivalence_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload_sha = row.get("payload_sha256")
        if isinstance(payload_sha, str):
            grouped.setdefault(payload_sha, []).append(row)

    groups: list[dict[str, Any]] = []
    for payload_sha, group_rows in grouped.items():
        if len(group_rows) < 2:
            continue
        sorted_rows = sorted(group_rows, key=lambda item: item["label"])
        archive_bytes = [
            row.get("archive_bytes") for row in sorted_rows if isinstance(row.get("archive_bytes"), int)
        ]
        groups.append(
            {
                "payload_sha256": payload_sha,
                "labels": [row["label"] for row in sorted_rows],
                "archive_byte_span": max(archive_bytes) - min(archive_bytes) if archive_bytes else None,
                "same_seg_contribution": equal_numeric(
                    [row.get("score_seg_contribution") for row in sorted_rows]
                ),
                "same_pose_contribution": equal_numeric(
                    [row.get("score_pose_contribution") for row in sorted_rows]
                ),
                "archives": [
                    {
                        "label": row["label"],
                        "archive_sha256": row.get("archive_sha256"),
                        "archive_bytes": row.get("archive_bytes"),
                        "zip_member": row.get("zip_member"),
                        "zip_overhead_bytes": row.get("zip_overhead_bytes"),
                        "profile_match_key": row.get("profile_match_key"),
                    }
                    for row in sorted_rows
                ],
                "readiness": "byte-identical payload pair; use as repack custody/control only",
            }
        )
    return sorted(groups, key=lambda item: item["labels"])


def followup_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for row in rows:
        for section in row.get("top_payload_sections") or []:
            section_name = section.get("name")
            if not section_name:
                continue
            action = "payload grammar audit"
            lowered = str(section_name).lower()
            if "decoder" in lowered or "weight" in lowered:
                action = "decoder self-compression or weight-stream recoding fixture"
            elif "latent" in lowered or "sidecar" in lowered:
                action = "latent/sidecar arithmetic-coding parity fixture"
            targets.append(
                {
                    "label": row["label"],
                    "section": section_name,
                    "section_bytes": section.get("bytes"),
                    "entropy_bits_per_byte": section.get("entropy_bits_per_byte"),
                    "payload_sha256": row.get("payload_sha256"),
                    "required_next_gate": "build byte-different archive, then exact CUDA replay",
                    "suggested_action": action,
                }
            )
    return sorted(
        targets,
        key=lambda item: (
            -(item["section_bytes"] if isinstance(item.get("section_bytes"), int) else -1),
            item["label"],
            str(item["section"]),
        ),
    )


def frontier_eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("canonical_frontier_eligible") is True
        and isinstance(row.get("score"), int | float)
    ]


def current_frontier(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = frontier_eligible_rows(rows)
    if not eligible:
        return None
    row = min(eligible, key=lambda item: float(item["score"]))
    return {
        "label": row["label"],
        "score": row.get("score"),
        "archive_bytes": row.get("archive_bytes"),
        "archive_sha256": row.get("archive_sha256"),
        "frontier_scope": row.get("frontier_scope"),
        "evidence_grade": row.get("evidence_grade"),
        "eval_artifact": row.get("eval_artifact"),
    }


def _rate_mass_score(bytes_: Any) -> float | None:
    if not isinstance(bytes_, int):
        return None
    return 25.0 * bytes_ / 37_545_489


def hidden_gem_byte_mass_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank exact-evaluable byte targets by frontier proximity before byte mass."""
    frontier = current_frontier(rows)
    if not frontier or not isinstance(frontier.get("score"), int | float):
        return []
    frontier_score = float(frontier["score"])
    targets: list[dict[str, Any]] = []
    for row in frontier_eligible_rows(rows):
        score = float(row["score"])
        score_gap = score - frontier_score
        for section in row.get("payload_sections") or row.get("top_payload_sections") or []:
            section_name = section.get("name")
            section_bytes = section.get("bytes")
            if not section_name or not isinstance(section_bytes, int) or section_bytes <= 0:
                continue
            role = section_optimization_role(str(section_name))
            if role == "control_or_metadata":
                priority = "low"
            elif row.get("label") == frontier["label"]:
                priority = "current_frontier_primary"
            else:
                priority = "near_frontier_secondary"
            targets.append(
                {
                    "label": row["label"],
                    "section": section_name,
                    "optimization_role": role,
                    "section_bytes": section_bytes,
                    "section_sha256": section.get("sha256"),
                    "entropy_bits_per_byte": section.get("entropy_bits_per_byte"),
                    "score_gap_to_current_frontier": round(score_gap, 12),
                    "frontier_label": frontier["label"],
                    "frontier_score": frontier_score,
                    "archive_sha256": row.get("archive_sha256"),
                    "payload_sha256": row.get("payload_sha256"),
                    "rate_mass_score_if_removed": _rate_mass_score(section_bytes),
                    "exact_evaluable_next_gate": (
                        "build byte-different archive with old/new section SHA-256 and "
                        "charged-byte proof, then exact CUDA auth eval after lane claim"
                    ),
                    "priority": priority,
                    "score_claim": False,
                    "dispatch_attempted": False,
                }
            )
    return sorted(
        targets,
        key=lambda item: (
            float(item["score_gap_to_current_frontier"]),
            0 if item["priority"] == "current_frontier_primary" else 1,
            -(item["section_bytes"] if isinstance(item["section_bytes"], int) else -1),
            str(item["label"]),
            str(item["section"]),
        ),
    )


def next_exact_evaluable_target(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for target in hidden_gem_byte_mass_ranking(rows):
        if target.get("priority") != "low":
            return target
    return None


def section_optimization_role(section_name: str) -> str:
    """Classify a payload section into a byte-optimization role."""
    lowered = str(section_name).lower()
    if "header" in lowered or "meta" in lowered or "scale" in lowered:
        return "control_or_metadata"
    if "decoder" in lowered or "weight" in lowered:
        return "decoder_weight_stream"
    if "latent" in lowered:
        return "latent_stream"
    if "sidecar" in lowered or "correction" in lowered:
        return "sidecar_or_correction_stream"
    if "hist" in lowered or "range" in lowered or "ac_" in lowered or "arithmetic" in lowered:
        return "entropy_model_or_range_stream"
    return "opaque_payload_stream"


def payload_section_manifests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build deterministic payload-section manifests for byte-different follow-ups.

    The manifest is forensic optimizer input only. It records the exact section
    bytes and section roles that a future repack must change before dispatch can
    be considered meaningful.
    """
    manifests: list[dict[str, Any]] = []
    for row in rows:
        sections = row.get("top_payload_sections") or []
        all_sections = row.get("payload_sections") or sections
        section_rows = []
        for index, section in enumerate(all_sections):
            name = str(section.get("name") or f"section_{index}")
            section_rows.append(
                {
                    "index": index,
                    "name": name,
                    "start": section.get("start"),
                    "end": section.get("end"),
                    "bytes": section.get("bytes"),
                    "sha256": section.get("sha256"),
                    "entropy_bits_per_byte": section.get("entropy_bits_per_byte"),
                    "optimization_role": section_optimization_role(name),
                    "required_byte_change_proof": (
                        "future candidate must record old/new section SHA-256 "
                        "and old/new charged bytes before exact-eval dispatch"
                    ),
                }
            )
        manifests.append(
            {
                "label": row["label"],
                "archive_sha256": row.get("archive_sha256"),
                "archive_bytes": row.get("archive_bytes"),
                "zip_member": row.get("zip_member"),
                "payload_sha256": row.get("payload_sha256"),
                "member_bytes": row.get("member_bytes"),
                "profile_match_key": row.get("profile_match_key"),
                "sections": section_rows,
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_blockers": [
                    "payload_section_manifest_is_byte_forensics_only",
                    "requires_byte_different_candidate_archive",
                    "requires_exact_cuda_auth_eval_on_candidate",
                ],
            }
        )
    return sorted(manifests, key=lambda item: str(item["label"]))


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# HNeRV Frontier Scorecard",
        "",
        "| label | grade | canonical | scope | score | bytes | seg | pose | rate | largest section | archive sha |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["score"] if item["score"] is not None else 999):
        largest = row.get("largest_payload_section") or {}
        largest_text = f"{largest.get('name')}:{largest.get('bytes')}" if largest else "n/a"
        lines.append(
            "| {label} | {grade} | {canonical} | `{scope}` | {score:.12f} | {bytes_} | "
            "{seg:.9f} | {pose:.9f} | {rate:.9f} | `{largest}` | `{sha}` |".format(
                label=row["label"],
                grade=row["evidence_grade"],
                canonical="yes" if row.get("canonical_frontier_eligible") else "no",
                scope=row.get("frontier_scope") or "exact_local_cuda_custody",
                score=row["score"],
                bytes_=row["archive_bytes"],
                seg=row["score_seg_contribution"],
                pose=row["score_pose_contribution"],
                rate=row["score_rate_contribution"],
                largest=largest_text,
                sha=(row["archive_sha256"] or "")[:16],
            )
        )
    groups = payload_equivalence_groups(rows)
    if groups:
        lines.extend(
            [
                "",
                "## Byte-Identical Payload Groups",
                "",
                "| labels | archive byte span | same seg | same pose | payload sha |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for group in groups:
            lines.append(
                "| {labels} | {span} | {seg} | {pose} | `{sha}` |".format(
                    labels=", ".join(group["labels"]),
                    span=group["archive_byte_span"],
                    seg=str(group["same_seg_contribution"]).lower(),
                    pose=str(group["same_pose_contribution"]).lower(),
                    sha=group["payload_sha256"][:16],
                )
            )
    targets = followup_targets(rows)
    if targets:
        lines.extend(
            [
                "",
                "## Payload Follow-Up Targets",
                "",
                "| label | section | bytes | entropy b/B | required next gate | suggested action |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for target in targets[:12]:
            entropy = target.get("entropy_bits_per_byte")
            entropy_text = f"{entropy:.6f}" if isinstance(entropy, int | float) else "n/a"
            lines.append(
                "| {label} | `{section}` | {bytes_} | {entropy} | {gate} | {action} |".format(
                    label=target["label"],
                    section=target["section"],
                    bytes_=target["section_bytes"],
                    entropy=entropy_text,
                    gate=target["required_next_gate"],
                    action=target["suggested_action"],
                )
            )
    next_target = next_exact_evaluable_target(rows)
    ranking = hidden_gem_byte_mass_ranking(rows)
    if next_target:
        lines.extend(
            [
                "",
                "## Next Exact-Evaluable Target",
                "",
                "| frontier | target label | section | role | bytes | score gap | required next gate |",
                "|---|---|---|---|---:|---:|---|",
                "| {frontier} | {label} | `{section}` | `{role}` | {bytes_} | {gap:.12f} | {gate} |".format(
                    frontier=next_target["frontier_label"],
                    label=next_target["label"],
                    section=next_target["section"],
                    role=next_target["optimization_role"],
                    bytes_=next_target["section_bytes"],
                    gap=next_target["score_gap_to_current_frontier"],
                    gate=next_target["exact_evaluable_next_gate"],
                ),
                "",
                "This target is ranked by exact-frontier proximity first, then charged",
                "payload byte mass. It is a routing target only; it is not a new score",
                "claim.",
            ]
        )
    if ranking:
        lines.extend(
            [
                "",
                "## Hidden-Gem Byte-Mass Ranking",
                "",
                "| rank | label | section | role | bytes | score gap | priority |",
                "|---:|---|---|---|---:|---:|---|",
            ]
        )
        for rank, target in enumerate(ranking[:12], start=1):
            lines.append(
                "| {rank} | {label} | `{section}` | `{role}` | {bytes_} | {gap:.12f} | `{priority}` |".format(
                    rank=rank,
                    label=target["label"],
                    section=target["section"],
                    role=target["optimization_role"],
                    bytes_=target["section_bytes"],
                    gap=target["score_gap_to_current_frontier"],
                    priority=target["priority"],
                )
            )
    manifests = payload_section_manifests(rows)
    if manifests:
        lines.extend(
            [
                "",
                "## Payload Section Manifests",
                "",
                "| label | section | role | bytes | sha256 | required proof |",
                "|---|---|---|---:|---|---|",
            ]
        )
        for manifest in manifests:
            for section in manifest["sections"][:6]:
                lines.append(
                    "| {label} | `{section}` | `{role}` | {bytes_} | `{sha}` | {proof} |".format(
                        label=manifest["label"],
                        section=section["name"],
                        role=section["optimization_role"],
                        bytes_=section["bytes"],
                        sha=str(section.get("sha256") or "")[:16],
                        proof=section["required_byte_change_proof"],
                    )
                )
    lines.extend(
        [
            "",
            "Interpretation: score truth remains the exact CUDA replay JSON. Payload",
            "sections are forensic signals for the next compression action; they do",
            "not imply score deltas without a new exact archive eval.",
            "",
            "Guardrail: a lossless Brotli repack row is a local exact-custody",
            "byte-control. It does not supersede categorical/range-coded HNeRV",
            "families unless that exact candidate archive has a lower CUDA score",
            "under the same custody standard.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-json",
        action="append",
        type=Path,
        help="Payload profile JSON emitted by profile_hnerv_frontier_payloads.py.",
    )
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        type=Path,
        help=(
            "Candidate manifest emitted by a byte-transform builder. Rows match "
            "by candidate_archive_sha256 and remain exact-eval scored by JSON."
        ),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("evals", nargs="+", help="LABEL=path/to/contest_auth_eval.adjudicated.json")
    args = parser.parse_args()

    profiles = profile_indexes(args.profile_json)
    candidates = candidate_indexes(args.candidate_manifest)
    rows: list[dict[str, Any]] = []
    for item in args.evals:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        rows.append(row_from_eval(label, Path(raw_path), profiles, candidates))

    payload = {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "exact_cuda_auth_eval_json",
        "frontier_scope": "exact_local_cuda_custody",
        "interpretation_guardrails": [
            "lossless_brotli_repack_rows_are_local_byte_controls_not_categorical_frontier_claims",
            "categorical_or_range_coded_hnerv_rows_supersede_only_with_lower_exact_cuda_custody_score",
        ],
        "payload_equivalence_groups": payload_equivalence_groups(rows),
        "payload_section_manifests": payload_section_manifests(rows),
        "followup_targets": followup_targets(rows),
        "current_frontier": current_frontier(rows),
        "next_exact_evaluable_target": next_exact_evaluable_target(rows),
        "hidden_gem_byte_mass_ranking": hidden_gem_byte_mass_ranking(rows),
        "rows": sorted(rows, key=lambda item: item["score"] if item["score"] is not None else 999),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
