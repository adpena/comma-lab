"""Planning-only HNeRV payload-section repack targets.

This module consumes the HNeRV frontier scorecard section manifests and emits
no-op-resistant byte targets for later compressor work. It never rewrites an
archive, evaluates a score, or treats byte projections as evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
SCHEMA_VERSION = 1

ROLE_ACTIONS = {
    "decoder_weight_stream": "build decoder self-compression or weight-stream recoding fixture",
    "latent_stream": "build latent arithmetic-coding parity fixture",
    "sidecar_or_correction_stream": "build sidecar/correction entropy-coding parity fixture",
    "entropy_model_or_range_stream": "audit entropy model overhead and range stream packing",
    "control_or_metadata": "audit fixed header and metadata compaction",
    "opaque_payload_stream": "deconstruct payload grammar before proposing a transform",
}

ROLE_PRIORITY = {
    "decoder_weight_stream": 0,
    "latent_stream": 1,
    "sidecar_or_correction_stream": 2,
    "entropy_model_or_range_stream": 3,
    "control_or_metadata": 4,
    "opaque_payload_stream": 5,
}


class HnervSectionPlanError(ValueError):
    """Raised when a section scorecard is not safe planning input."""


def build_section_repack_plan(
    scorecard: Mapping[str, Any],
    *,
    labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic byte-target plan from payload section manifests."""
    manifests = scorecard.get("payload_section_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise HnervSectionPlanError("scorecard missing payload_section_manifests")
    label_filter = None if labels is None else {str(label) for label in labels}
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise HnervSectionPlanError("payload_section_manifest entries must be objects")
        label = str(manifest.get("label") or "")
        if not label:
            raise HnervSectionPlanError("payload_section_manifest missing label")
        if label_filter is not None and label not in label_filter:
            continue
        rows.extend(_rows_from_manifest(manifest))
    if not rows:
        raise HnervSectionPlanError("no payload sections matched the requested labels")
    rows.sort(
        key=lambda row: (
            ROLE_PRIORITY.get(str(row["optimization_role"]), 99),
            -int(row["section_bytes"]),
            str(row["label"]),
            str(row["section_name"]),
        )
    )
    role_counts = Counter(str(row["optimization_role"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_section_repack",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "planning_only_section_targets",
            "requires_byte_different_archive",
            "requires_old_new_section_sha256_proof",
            "requires_exact_cuda_auth_eval",
        ],
        "scorecard_tool": scorecard.get("tool"),
        "score_truth": scorecard.get("score_truth"),
        "selected_labels": sorted({str(row["label"]) for row in rows}),
        "role_counts": dict(sorted(role_counts.items())),
        "total_section_bytes": sum(int(row["section_bytes"]) for row in rows),
        "rows": rows,
    }


def candidate_diff_from_scorecard_manifests(
    scorecard: Mapping[str, Any],
    *,
    source_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Build a candidate section diff by comparing two scorecard labels."""
    manifests = _manifest_by_label(scorecard)
    try:
        source = manifests[str(source_label)]
        candidate = manifests[str(candidate_label)]
    except KeyError as exc:
        raise HnervSectionPlanError(f"missing payload section manifest label: {exc.args[0]}") from exc
    source_sections = _section_by_name(source)
    candidate_sections = _section_by_name(candidate)
    rows = []
    for section_name, source_section in source_sections.items():
        candidate_section = candidate_sections.get(section_name)
        if candidate_section is None:
            continue
        rows.append(
            {
                "label": str(source_label),
                "section_name": section_name,
                "source_section_sha256": source_section.get("sha256"),
                "candidate_section_sha256": candidate_section.get("sha256"),
                "source_bytes": source_section.get("bytes"),
                "candidate_bytes": candidate_section.get("bytes"),
                "candidate_label": str(candidate_label),
            }
        )
    if not rows:
        raise HnervSectionPlanError(
            f"no common section names between {source_label!r} and {candidate_label!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_section_repack.candidate_diff_from_scorecard_manifests",
        "score_claim": False,
        "dispatch_attempted": False,
        "source_label": str(source_label),
        "candidate_label": str(candidate_label),
        "source_archive_sha256": source.get("archive_sha256"),
        "candidate_archive_sha256": candidate.get("archive_sha256"),
        "source_payload_sha256": source.get("payload_sha256"),
        "candidate_payload_sha256": candidate.get("payload_sha256"),
        "sections": sorted(rows, key=lambda row: str(row["section_name"])),
    }


def render_markdown(plan: Mapping[str, Any]) -> str:
    """Render a section-repack plan as deterministic markdown."""
    rows = plan.get("rows")
    if not isinstance(rows, list):
        raise HnervSectionPlanError("plan rows must be a list")
    lines = [
        "# HNeRV Section Repack Plan",
        "",
        f"- score_claim: `{str(plan.get('score_claim') is True).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(plan.get('ready_for_exact_eval_dispatch') is True).lower()}`",
        f"- selected_labels: `{','.join(plan.get('selected_labels') or [])}`",
        f"- total_section_bytes: `{plan.get('total_section_bytes')}`",
        "",
        "| label | section | role | bytes | 1% rate gain | 5% rate gain | required proof | action |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | `{section}` | `{role}` | {bytes_} | {gain1:.9f} | {gain5:.9f} | {proof} | {action} |".format(
                label=row["label"],
                section=row["section_name"],
                role=row["optimization_role"],
                bytes_=row["section_bytes"],
                gain1=float(row["rate_score_gain_if_save_1pct"]),
                gain5=float(row["rate_score_gain_if_save_5pct"]),
                proof=row["candidate_required_proof"],
                action=row["recommended_next_action"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def audit_candidate_section_diff(
    plan: Mapping[str, Any],
    candidate_diff: Mapping[str, Any],
    *,
    require_raw_equivalence: bool = False,
) -> dict[str, Any]:
    """Audit a future section-transform diff against a section-repack plan.

    This is the no-op guard between a byte-target plan and exact-eval dispatch.
    It proves section bytes changed, but still does not claim any component
    score movement.
    """
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or not plan_rows:
        raise HnervSectionPlanError("plan rows must be a nonempty list")
    indexed = {
        (str(row.get("label")), str(row.get("section_name"))): row
        for row in plan_rows
        if isinstance(row, Mapping)
    }
    blockers: list[str] = []
    audited_rows: list[dict[str, Any]] = []
    candidate_archive_sha = candidate_diff.get("candidate_archive_sha256")
    if not _is_sha256(candidate_archive_sha):
        blockers.append("candidate_archive_sha256_missing_or_invalid")
    source_archive_sha = candidate_diff.get("source_archive_sha256")
    if source_archive_sha is not None and not _is_sha256(source_archive_sha):
        blockers.append("source_archive_sha256_invalid")
    sections = candidate_diff.get("sections")
    if not isinstance(sections, list) or not sections:
        blockers.append("candidate_diff_missing_sections")
        sections = []
    seen: set[tuple[str, str]] = set()
    changed_count = 0
    total_byte_delta = 0
    for section in sections:
        if not isinstance(section, Mapping):
            blockers.append("candidate_diff_section_not_object")
            continue
        key = (str(section.get("label") or ""), str(section.get("section_name") or ""))
        if key in seen:
            blockers.append(f"duplicate_candidate_section:{key[0]}:{key[1]}")
            continue
        seen.add(key)
        source = indexed.get(key)
        if source is None:
            blockers.append(f"candidate_section_not_in_plan:{key[0]}:{key[1]}")
            continue
        row, row_blockers = _audit_candidate_section_row(source, section)
        blockers.extend(row_blockers)
        if row["changed"]:
            changed_count += 1
        total_byte_delta += int(row["byte_delta"])
        audited_rows.append(row)
    if changed_count == 0:
        blockers.append("candidate_diff_has_no_changed_sections")
    if require_raw_equivalence:
        blockers.extend(_audit_raw_equivalence(candidate_diff, audited_rows))
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.hnerv_section_repack.audit_candidate_section_diff",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": not blockers,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "requires_archive_manifest_preflight",
            "requires_lane_dispatch_claim",
            "requires_exact_cuda_auth_eval",
        ],
        "blockers": blockers,
        "candidate_archive_sha256": candidate_archive_sha,
        "source_archive_sha256": source_archive_sha,
        "changed_section_count": changed_count,
        "total_byte_delta": total_byte_delta,
        "rate_score_delta_if_components_equal": round(total_byte_delta * RATE_SCORE_PER_BYTE, 12),
        "sections": audited_rows,
    }


def _rows_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("score_claim") is not False:
        raise HnervSectionPlanError("payload section manifest must have score_claim=false")
    if manifest.get("dispatch_attempted") is not False:
        raise HnervSectionPlanError("payload section manifest must have dispatch_attempted=false")
    sections = manifest.get("sections")
    if not isinstance(sections, list) or not sections:
        raise HnervSectionPlanError("payload section manifest missing sections")
    rows = []
    for section in sections:
        if not isinstance(section, Mapping):
            raise HnervSectionPlanError("payload section entries must be objects")
        rows.append(_row_from_section(manifest, section))
    return rows


def _row_from_section(manifest: Mapping[str, Any], section: Mapping[str, Any]) -> dict[str, Any]:
    role = str(section.get("optimization_role") or "opaque_payload_stream")
    section_bytes = int(section.get("bytes") or 0)
    if section_bytes <= 0:
        raise HnervSectionPlanError("payload section bytes must be positive")
    section_sha = section.get("sha256")
    if not isinstance(section_sha, str) or len(section_sha) != 64:
        raise HnervSectionPlanError("payload section must record a 64-char sha256")
    return {
        "label": str(manifest["label"]),
        "archive_sha256": manifest.get("archive_sha256"),
        "payload_sha256": manifest.get("payload_sha256"),
        "zip_member": manifest.get("zip_member"),
        "section_index": section.get("index"),
        "section_name": str(section.get("name") or ""),
        "section_start": section.get("start"),
        "section_end": section.get("end"),
        "section_bytes": section_bytes,
        "section_sha256": section_sha,
        "entropy_bits_per_byte": section.get("entropy_bits_per_byte"),
        "optimization_role": role,
        "recommended_next_action": ROLE_ACTIONS.get(role, ROLE_ACTIONS["opaque_payload_stream"]),
        "rate_score_gain_if_save_1pct": _rate_gain(section_bytes, 0.01),
        "rate_score_gain_if_save_5pct": _rate_gain(section_bytes, 0.05),
        "rate_score_gain_if_save_10pct": _rate_gain(section_bytes, 0.10),
        "candidate_required_proof": (
            "candidate manifest must include source section sha256, candidate "
            "section sha256, source bytes, candidate bytes, and exact archive sha256"
        ),
        "dispatchable": False,
        "score_claim": False,
    }


def _rate_gain(section_bytes: int, fraction: float) -> float:
    return round(max(1, int(section_bytes * fraction)) * RATE_SCORE_PER_BYTE, 12)


def _manifest_by_label(scorecard: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    manifests = scorecard.get("payload_section_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise HnervSectionPlanError("scorecard missing payload_section_manifests")
    out: dict[str, Mapping[str, Any]] = {}
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            raise HnervSectionPlanError("payload section manifest entries must be objects")
        label = str(manifest.get("label") or "")
        if not label:
            raise HnervSectionPlanError("payload section manifest missing label")
        if label in out:
            raise HnervSectionPlanError(f"duplicate payload section manifest label: {label}")
        out[label] = manifest
    return out


def _section_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    sections = manifest.get("sections")
    if not isinstance(sections, list) or not sections:
        raise HnervSectionPlanError("payload section manifest missing sections")
    out: dict[str, Mapping[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            raise HnervSectionPlanError("payload section entries must be objects")
        name = str(section.get("name") or "")
        if not name:
            raise HnervSectionPlanError("payload section missing name")
        if name in out:
            raise HnervSectionPlanError(f"duplicate payload section name: {name}")
        out[name] = section
    return out


def _audit_candidate_section_row(
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    label = str(source["label"])
    section_name = str(source["section_name"])
    source_sha = candidate.get("source_section_sha256")
    candidate_sha = candidate.get("candidate_section_sha256")
    if source_sha != source.get("section_sha256"):
        blockers.append(f"source_section_sha256_mismatch:{label}:{section_name}")
    if not _is_sha256(candidate_sha):
        blockers.append(f"candidate_section_sha256_missing_or_invalid:{label}:{section_name}")
    source_bytes = candidate.get("source_bytes")
    candidate_bytes = candidate.get("candidate_bytes")
    if int(source_bytes) != int(source["section_bytes"]):
        blockers.append(f"source_bytes_mismatch:{label}:{section_name}")
    if not isinstance(candidate_bytes, int) or candidate_bytes < 0:
        blockers.append(f"candidate_bytes_invalid:{label}:{section_name}")
        candidate_bytes = 0
    changed = candidate_sha != source.get("section_sha256") or int(candidate_bytes) != int(source["section_bytes"])
    if not changed:
        blockers.append(f"candidate_section_noop:{label}:{section_name}")
    byte_delta = int(candidate_bytes) - int(source["section_bytes"])
    return (
        {
            "label": label,
            "section_name": section_name,
            "optimization_role": source.get("optimization_role"),
            "source_section_sha256": source.get("section_sha256"),
            "candidate_section_sha256": candidate_sha,
            "source_bytes": int(source["section_bytes"]),
            "candidate_bytes": int(candidate_bytes),
            "byte_delta": byte_delta,
            "changed": changed,
            "score_claim": False,
        },
        blockers,
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _audit_raw_equivalence(
    candidate_diff: Mapping[str, Any],
    audited_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    rows = candidate_diff.get("brotli_raw_equivalence")
    if not isinstance(rows, list) or not rows:
        return ["brotli_raw_equivalence_missing"]
    by_name = {}
    blockers: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            blockers.append("brotli_raw_equivalence_row_not_object")
            continue
        name = str(row.get("section_name") or "")
        if not name:
            blockers.append("brotli_raw_equivalence_section_missing")
            continue
        by_name[name] = row
    for audited in audited_rows:
        section_name = str(audited.get("section_name") or "")
        if section_name not in {"decoder_packed_brotli", "latents_and_sidecar_brotli"}:
            continue
        raw = by_name.get(section_name)
        if raw is None:
            blockers.append(f"brotli_raw_equivalence_missing:{section_name}")
            continue
        if raw.get("raw_equal") is not True:
            blockers.append(f"brotli_raw_mismatch:{section_name}")
        if not _is_sha256(raw.get("source_raw_sha256")):
            blockers.append(f"source_raw_sha256_missing_or_invalid:{section_name}")
        if not _is_sha256(raw.get("candidate_raw_sha256")):
            blockers.append(f"candidate_raw_sha256_missing_or_invalid:{section_name}")
        if raw.get("source_raw_sha256") != raw.get("candidate_raw_sha256"):
            blockers.append(f"brotli_raw_sha256_mismatch:{section_name}")
        raw_bytes = raw.get("raw_bytes")
        if not isinstance(raw_bytes, int) or raw_bytes <= 0:
            blockers.append(f"brotli_raw_bytes_invalid:{section_name}")
    return blockers


__all__ = [
    "RATE_SCORE_PER_BYTE",
    "ROLE_ACTIONS",
    "ROLE_PRIORITY",
    "SCHEMA_VERSION",
    "HnervSectionPlanError",
    "audit_candidate_section_diff",
    "build_section_repack_plan",
    "candidate_diff_from_scorecard_manifests",
    "render_markdown",
]
