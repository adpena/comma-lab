# SPDX-License-Identifier: MIT
"""Scorer-conditional MDL ablation contracts.

This module implements the Z1 measurement gate from the 2026-05-14
zen-floor council. It is intentionally a measurement/planning surface, not a
codec and not a score authority.

The current implementation measures the byte entropy of parser-proven archive
sections and emits a conservative proxy for scorer-conditional MDL:

* unconditional payload entropy, ``H(payload)``;
* parser section / role conditioned entropy, ``H(payload | section_role)``;
* optional scalar-eval-feature grouped entropy when exact-eval JSON is
  available.

The third layer is explicitly marked as a proxy. A true
``H(payload | scorer_features)`` claim requires feature maps or component
response curves that bind byte regions to scorer outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.hnerv_packet_sections import (
    PARSER_AUTO,
    build_packet_section_manifest,
)
from tac.analysis.scorer_section_evidence import (
    SectionScorerEvidenceMap,
    axis_label,
    coerce_section_scorer_evidence_map,
)
from tac.hnerv_lowlevel_packer import read_strict_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_MAGIC,
    parse_pr106_sidecar_packet,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes

SCHEMA = "tac_scorer_conditional_mdl_ablation_v1"
SENSITIVITY_SCHEMA = "tac.sensitivity_map.scorer_conditional_entropy_map_v1"
TOOL = "tac.analysis.scorer_conditional_mdl"
SCORE_EVIDENCE_GRADE = "invalid_no_score"

DISPATCH_BLOCKERS = (
    "measurement_only_no_candidate_archive",
    "no_runtime_decoder_adapter",
    "no_exact_eval_score_claim",
    "true_scorer_conditional_entropy_requires_byte_to_scorer_feature_binding",
    "promotion_requires_byte_closed_codec_transform_and_exact_cuda_eval",
)

ROLE_WEIGHTS = {
    "decoder_weight_stream": 1.20,
    "latent_stream": 1.50,
    "entropy_model_or_range_stream": 0.90,
    "sidecar_or_correction_stream": 1.00,
    "control_or_metadata": 0.20,
    "decoder_byte_map_metadata": 0.70,
    "opaque_payload_stream": 0.60,
    "training_provenance_only": 0.0,
}

PROVENANCE_ONLY_ROLES = frozenset({"training_provenance_only"})


class ScorerConditionalMdlError(ValueError):
    """Raised when a scorer-conditional MDL manifest cannot be produced."""


@dataclass(frozen=True)
class ArchiveInput:
    """One archive plus optional scorer/eval side information."""

    label: str
    archive_path: Path
    parser: str = PARSER_AUTO
    eval_json_path: Path | None = None


@dataclass(frozen=True)
class _SectionBytes:
    archive_label: str
    section_name: str
    optimization_role: str
    runtime_effect: str
    score_affecting_at_inflate: bool
    start: int
    end: int
    data: bytes


def build_scorer_conditional_mdl_ablation(
    archives: Sequence[ArchiveInput],
    *,
    repo_root: str | Path | None = None,
    chunk_size: int = 1024,
    source_documents: Sequence[str] = (),
    section_scorer_evidence: SectionScorerEvidenceMap | Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Build a Z1 MDL ablation manifest for archive payloads.

    The returned payload is proxy-safe: it contains no score claim and is not
    ready for dispatch. It is meant to feed the sensitivity-map, bit allocator,
    and probe-disambiguator queues.
    """

    if not archives:
        raise ScorerConditionalMdlError("at least one archive is required")
    if chunk_size <= 0:
        raise ScorerConditionalMdlError("chunk_size must be positive")

    root = Path(repo_root or Path.cwd()).resolve()
    evidence_map = coerce_section_scorer_evidence_map(section_scorer_evidence)
    records: list[dict[str, Any]] = []
    section_rows: list[_SectionBytes] = []
    feature_rows: list[dict[str, Any]] = []

    for archive in archives:
        record, sections = _archive_record(archive, repo_root=root)
        records.append(record)
        section_rows.extend(sections)
        feature_rows.append(record["scorer_feature_summary"])

    payload_bytes = b"".join(row.data for row in section_rows)
    unconditional = _entropy_layer(
        "unconditional_payload",
        {"all_payload_bytes": payload_bytes},
        current_bytes=len(payload_bytes),
    )
    by_section_name = _entropy_layer(
        "parser_section_conditioned",
        _group_sections(section_rows, key="section_name"),
        current_bytes=len(payload_bytes),
    )
    by_role = _entropy_layer(
        "parser_role_conditioned",
        _group_sections(section_rows, key="optimization_role"),
        current_bytes=len(payload_bytes),
    )
    scorer_proxy = _scorer_feature_proxy_layer(section_rows, records, current_bytes=len(payload_bytes))
    section_evidence_layer = _section_scorer_evidence_layer(
        section_rows,
        evidence_map,
        current_bytes=len(payload_bytes),
        repo_root=root,
    )
    sensitivity_map = _sensitivity_map(section_rows, chunk_size=chunk_size, repo_root=root)
    verdict = _verdict(
        unconditional=unconditional,
        by_role=by_role,
        scorer_proxy=scorer_proxy,
        section_evidence=section_evidence_layer,
        records=records,
    )
    true_scorer_ready = bool(section_evidence_layer.get("true_scorer_conditioning_ready"))

    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "tool": TOOL,
        "created_at_utc": _utc_now(),
        "score_claim": False,
        "score_evidence_grade": SCORE_EVIDENCE_GRADE,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "true_scorer_conditional_entropy_claim": true_scorer_ready,
        "chunk_size": int(chunk_size),
        "archive_count": len(records),
        "archives": records,
        "source_documents": list(source_documents),
        "axis_labels": _manifest_axis_labels(records, section_evidence_layer),
        "measurement_layers": {
            "unconditional_payload": unconditional,
            "parser_section_conditioned": by_section_name,
            "parser_role_conditioned": by_role,
            "scorer_feature_proxy_conditioned": scorer_proxy,
            "scorer_section_evidence_conditioned": section_evidence_layer,
        },
        "section_scorer_evidence": section_evidence_layer,
        "sensitivity_map": sensitivity_map,
        "allocator_hook": _allocator_hook(sensitivity_map),
        "probe_disambiguator": _probe_disambiguator(verdict),
        "autopilot_rows": _autopilot_rows(verdict),
        "verdict": verdict,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "notes": [
            "This is Decision Z1 measurement output, not a candidate archive.",
            "Parser-role entropy is real parser-proven byte evidence.",
            "Scorer-feature grouping is a scalar-eval proxy unless feature maps or component-response curves are supplied.",
        ],
    }
    return manifest


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact lab-grade markdown summary."""

    layers = manifest.get("measurement_layers")
    if not isinstance(layers, Mapping):
        raise ScorerConditionalMdlError("manifest missing measurement_layers")
    verdict = manifest.get("verdict") if isinstance(manifest.get("verdict"), Mapping) else {}
    rows = []
    for key in (
        "unconditional_payload",
        "parser_section_conditioned",
        "parser_role_conditioned",
        "scorer_feature_proxy_conditioned",
        "scorer_section_evidence_conditioned",
    ):
        layer = layers.get(key)
        if isinstance(layer, Mapping):
            rows.append(
                "| `{}` | {} | {} | {} | {} |".format(
                    key,
                    layer.get("group_count"),
                    layer.get("floor_bytes_ceil"),
                    layer.get("gap_to_current_bytes_ceil"),
                    layer.get("claim_strength"),
                )
            )

    lines = [
        "# Z1 Scorer-Conditional MDL Ablation",
        "",
        f"- score_claim: `{str(manifest.get('score_claim')).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(manifest.get('ready_for_exact_eval_dispatch')).lower()}`",
        f"- true_scorer_conditional_entropy_claim: `{str(manifest.get('true_scorer_conditional_entropy_claim')).lower()}`",
        f"- archive_count: `{manifest.get('archive_count')}`",
        f"- verdict: `{verdict.get('headroom_class')}`",
        "",
        "| layer | groups | floor bytes ceil | gap bytes ceil | claim strength |",
        "|---|---:|---:|---:|---|",
        *rows,
        "",
        "## Interpretation",
        "",
        str(verdict.get("interpretation", "")),
        "",
        "## Next Actions",
        "",
    ]
    for action in verdict.get("next_actions", []) if isinstance(verdict.get("next_actions"), list) else []:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def dumps_manifest(payload: Mapping[str, Any]) -> str:
    """Return canonical JSON text."""

    return json_text(payload)


def _archive_record(
    archive: ArchiveInput,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[_SectionBytes]]:
    if not archive.label:
        raise ScorerConditionalMdlError("archive label must be nonempty")
    path = archive.archive_path
    if not path.is_file():
        raise ScorerConditionalMdlError(f"{archive.label}: archive missing: {path}")

    single = read_strict_single_member_zip(path)
    manifest = build_packet_section_manifest(
        path,
        label=archive.label,
        parser=archive.parser,
        repo_root=repo_root,
    )
    sections = manifest.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ScorerConditionalMdlError(f"{archive.label}: parser emitted no sections")
    parser_payload = _parser_payload_for_section_offsets(
        archive_label=archive.label,
        manifest=manifest,
        member_payload=single.payload,
    )

    section_bytes: list[_SectionBytes] = []
    section_records: list[dict[str, Any]] = []
    for section in sections:
        if not isinstance(section, Mapping):
            raise ScorerConditionalMdlError(f"{archive.label}: malformed section row")
        start = _as_int(section.get("start"), f"{archive.label}.section.start")
        end = _as_int(section.get("end"), f"{archive.label}.section.end")
        if start < 0 or end < start or end > len(parser_payload):
            raise ScorerConditionalMdlError(
                f"{archive.label}: section bounds outside parser payload: {start}:{end}"
            )
        data = parser_payload[start:end]
        name = str(section.get("name") or f"section_{len(section_bytes)}")
        role = str(section.get("optimization_role") or "opaque_payload_stream")
        runtime_effect = _runtime_effect_for_section(name, role)
        score_affecting = _score_affecting_at_inflate(role)
        entropy = _entropy_stats(data)
        section_bytes.append(
            _SectionBytes(
                archive_label=archive.label,
                section_name=name,
                optimization_role=role,
                runtime_effect=runtime_effect,
                score_affecting_at_inflate=score_affecting,
                start=start,
                end=end,
                data=data,
            )
        )
        section_records.append(
            {
                "name": name,
                "optimization_role": role,
                "runtime_effect": runtime_effect,
                "score_affecting_at_inflate": score_affecting,
                "start": start,
                "end": end,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "entropy_bits_per_byte": entropy["bits_per_symbol"],
                "entropy_floor_bytes_ceil": entropy["floor_bytes_ceil"],
                "gap_to_iid_floor_bytes_ceil": len(data) - entropy["floor_bytes_ceil"],
                "score_claim": False,
            }
        )

    eval_features = _eval_feature_summary(
        archive.eval_json_path,
        repo_root=repo_root,
        archive_sha256=single.archive_sha256,
        archive_size_bytes=single.archive_bytes,
    )
    return (
        {
            "label": archive.label,
            "archive_path": repo_relative(path, repo_root),
            "archive_bytes": single.archive_bytes,
            "archive_sha256": single.archive_sha256,
            "member_name": single.member_name,
            "member_bytes": single.member_bytes,
            "member_sha256": sha256_bytes(single.payload),
            "parser": manifest.get("parser"),
            "parser_input": manifest.get("parser_input"),
            "pr106_sidecar_wrapper": manifest.get("pr106_sidecar_wrapper"),
            "section_count": len(section_records),
            "sections": section_records,
            "scorer_feature_summary": eval_features,
            "score_claim": False,
        },
        section_bytes,
    )


def _parser_payload_for_section_offsets(
    *,
    archive_label: str,
    manifest: Mapping[str, Any],
    member_payload: bytes,
) -> bytes:
    parser_input = manifest.get("parser_input")
    if not isinstance(parser_input, Mapping):
        return member_payload
    kind = parser_input.get("kind")
    if kind == "member_payload":
        return member_payload
    if kind != "pr106_sidecar_inner_payload":
        raise ScorerConditionalMdlError(
            f"{archive_label}: unsupported parser_input kind {kind!r}"
        )
    if not member_payload or member_payload[0] != PR106_SIDECAR_MAGIC:
        raise ScorerConditionalMdlError(
            f"{archive_label}: manifest expects PR106 sidecar inner payload but member is not a sidecar"
        )
    try:
        packet = parse_pr106_sidecar_packet(member_payload)
    except ValueError as exc:
        raise ScorerConditionalMdlError(
            f"{archive_label}: PR106 sidecar wrapper parse failed: {exc}"
        ) from exc
    expected_sha = parser_input.get("sha256")
    actual_sha = sha256_bytes(packet.pr106_bytes)
    if isinstance(expected_sha, str) and expected_sha != actual_sha:
        raise ScorerConditionalMdlError(
            f"{archive_label}: parser_input sha mismatch: manifest={expected_sha} parsed={actual_sha}"
        )
    expected_bytes = parser_input.get("bytes")
    if isinstance(expected_bytes, int) and expected_bytes != len(packet.pr106_bytes):
        raise ScorerConditionalMdlError(
            f"{archive_label}: parser_input byte mismatch: manifest={expected_bytes} parsed={len(packet.pr106_bytes)}"
        )
    return packet.pr106_bytes


def _eval_feature_summary(
    path: Path | None,
    *,
    repo_root: Path,
    archive_sha256: str,
    archive_size_bytes: int,
) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "feature_class": "missing_eval_json",
            "score_axis": "unknown",
            "score_axis_label": axis_label("unknown"),
            "claim_strength": "none",
        }
    if not path.is_file():
        raise ScorerConditionalMdlError(f"eval json missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ScorerConditionalMdlError(f"eval json is not an object: {path}")
    custody = _validate_eval_json_custody(
        payload,
        path=path,
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size_bytes,
    )
    pose = _maybe_float(payload.get("avg_posenet_dist"))
    seg = _maybe_float(payload.get("avg_segnet_dist"))
    bytes_value = _maybe_int(payload.get("archive_size_bytes"))
    score = _maybe_float(
        payload.get("score_recomputed_from_components")
        or payload.get("canonical_score")
        or payload.get("final_score")
    )
    rate_contrib = _maybe_float(payload.get("score_rate_contribution"))
    if rate_contrib is None and bytes_value is not None:
        rate_contrib = 25.0 * float(bytes_value) / 37_545_489.0
    pose_contrib = _maybe_float(payload.get("score_pose_contribution"))
    if pose_contrib is None and pose is not None:
        pose_contrib = math.sqrt(max(0.0, 10.0 * pose))
    seg_contrib = _maybe_float(payload.get("score_seg_contribution"))
    if seg_contrib is None and seg is not None:
        seg_contrib = 100.0 * seg
    contributions = {
        "pose": pose_contrib,
        "seg": seg_contrib,
        "rate": rate_contrib,
    }
    present = {key: value for key, value in contributions.items() if value is not None}
    dominant = max(present, key=present.get) if present else "unknown"
    score_axis = _infer_score_axis(path, payload)
    return {
        "available": True,
        "feature_class": "scalar_eval_components_proxy",
        "claim_strength": "proxy_not_true_scorer_feature_map",
        "path": repo_relative(path, repo_root),
        "sha256": _sha256_path(path),
        "score_axis": score_axis,
        "score_axis_label": axis_label(score_axis),
        "score": score,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "archive_size_bytes": bytes_value,
        "score_contributions": contributions,
        "dominant_score_component": dominant,
        "pose_log10_bucket": _log10_bucket(pose),
        "seg_log10_bucket": _log10_bucket(seg),
        "custody_match": custody["match"],
        "custody_strength": custody["strength"],
        "custody_blockers": custody["blockers"],
    }


def _validate_eval_json_custody(
    payload: Mapping[str, Any],
    *,
    path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
) -> dict[str, Any]:
    archive_meta = payload.get("archive")
    provenance = payload.get("provenance")
    eval_archive_sha: Any = None
    eval_archive_bytes: Any = payload.get("archive_size_bytes")
    if isinstance(archive_meta, Mapping):
        eval_archive_sha = archive_meta.get("archive_sha256") or archive_meta.get("sha256")
        eval_archive_bytes = archive_meta.get("archive_size_bytes") or eval_archive_bytes
    if isinstance(provenance, Mapping):
        eval_archive_sha = provenance.get("archive_sha256") or eval_archive_sha
        eval_archive_bytes = provenance.get("archive_size_bytes") or eval_archive_bytes
    if isinstance(eval_archive_sha, str) and eval_archive_sha and eval_archive_sha != archive_sha256:
        raise ScorerConditionalMdlError(
            f"eval json archive sha mismatch for {path}: expected {archive_sha256}, got {eval_archive_sha}"
        )
    if eval_archive_bytes is not None:
        try:
            eval_bytes_int = int(eval_archive_bytes)
        except (TypeError, ValueError) as exc:
            raise ScorerConditionalMdlError(
                f"eval json archive_size_bytes is not an integer for {path}: {eval_archive_bytes!r}"
            ) from exc
        if eval_bytes_int != int(archive_size_bytes):
            raise ScorerConditionalMdlError(
                f"eval json archive_size_bytes mismatch for {path}: expected {archive_size_bytes}, got {eval_bytes_int}"
            )
    blockers: list[str] = []
    if not (isinstance(eval_archive_sha, str) and eval_archive_sha):
        blockers.append("missing_eval_json_archive_sha256")
    if eval_archive_bytes is None:
        blockers.append("missing_eval_json_archive_size_bytes")
    return {
        "match": not blockers,
        "strength": "archive_sha256_and_bytes" if not blockers else "missing_archive_identity",
        "blockers": blockers,
    }


def _scorer_feature_proxy_layer(
    sections: Sequence[_SectionBytes],
    records: Sequence[Mapping[str, Any]],
    *,
    current_bytes: int,
) -> dict[str, Any]:
    features_by_label = {
        str(record["label"]): record.get("scorer_feature_summary", {})
        for record in records
        if isinstance(record.get("scorer_feature_summary"), Mapping)
    }
    groups: dict[str, bytes] = {}
    missing = 0
    for section in sections:
        feature = features_by_label.get(section.archive_label, {})
        if not feature.get("available"):
            missing += 1
        key = "|".join(
            [
                section.optimization_role,
                str(feature.get("score_axis") or "unknown"),
                str(feature.get("dominant_score_component") or "unknown"),
                str(feature.get("pose_log10_bucket") or "pose_unknown"),
                str(feature.get("seg_log10_bucket") or "seg_unknown"),
            ]
        )
        groups[key] = groups.get(key, b"") + section.data
    layer = _entropy_layer(
        "scorer_feature_proxy_conditioned",
        groups,
        current_bytes=current_bytes,
    )
    layer["claim_strength"] = "proxy_not_true_scorer_conditional_entropy"
    layer["missing_eval_feature_section_count"] = missing
    layer["axis_labels"] = sorted(
        {
            str(feature.get("score_axis_label") or axis_label(str(feature.get("score_axis") or "unknown")))
            for feature in features_by_label.values()
        }
    )
    layer["blockers"] = [
        "scalar_archive_level_eval_features_do_not_bind_bytes_to_scorer_features",
        "true_scorer_conditioning_requires_penultimate_saliency_or_component_response_byte_map",
    ]
    return layer


def _section_scorer_evidence_layer(
    sections: Sequence[_SectionBytes],
    evidence_map: SectionScorerEvidenceMap | None,
    *,
    current_bytes: int,
    repo_root: Path,
) -> dict[str, Any]:
    if evidence_map is None:
        unbound = [f"{section.archive_label}:{section.section_name}" for section in sections]
        return {
            "label": "scorer_section_evidence_conditioned",
            "claim_strength": "proxy_not_true_scorer_conditional_entropy",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "current_bytes": int(current_bytes),
            "section_count": len(sections),
            "bound_section_count": 0,
            "unbound_section_count": len(unbound),
            "unbound_sections": unbound,
            "component_binding_count": 0,
            "true_scorer_conditioning_ready": False,
            "axis_labels": ["[axis-unknown]"],
            "sections": [],
            "evidence_map": {
                "available": False,
                "score_claim": False,
            },
            "blockers": [
                "section_scorer_evidence_map_missing",
                "proxy_not_true_scorer_conditional_entropy",
            ],
        }

    section_rows: list[dict[str, Any]] = []
    binding_count = 0
    ready_count = 0
    ready_section_count = 0
    unbound_sections: list[str] = []
    blockers: list[str] = []
    axis_labels: set[str] = set()
    for section in sections:
        bindings = evidence_map.bindings_for(section.archive_label, section.section_name)
        binding_rows = [binding.to_manifest(repo_root=repo_root) for binding in bindings]
        binding_count += len(binding_rows)
        ready_count += sum(1 for row in binding_rows if row.get("true_scorer_ready") is True)
        section_ready = any(row.get("true_scorer_ready") is True for row in binding_rows)
        if section_ready:
            ready_section_count += 1
        if not binding_rows:
            unbound_sections.append(f"{section.archive_label}:{section.section_name}")
        for row in binding_rows:
            axis_labels.add(str(row.get("axis_label") or "[axis-unknown]"))
            for blocker in row.get("blockers", []):
                blockers.append(
                    f"{section.archive_label}:{section.section_name}:{row.get('component')}:{blocker}"
                )
        section_rows.append(
            {
                "archive_label": section.archive_label,
                "section_name": section.section_name,
                "optimization_role": section.optimization_role,
                "runtime_effect": section.runtime_effect,
                "score_affecting_at_inflate": section.score_affecting_at_inflate,
                "start": section.start,
                "end": section.end,
                "bytes": len(section.data),
                "sha256": sha256_bytes(section.data),
                "component_bindings": binding_rows,
                "component_binding_count": len(binding_rows),
                "components": sorted({str(row.get("component")) for row in binding_rows}),
                "axis_labels": sorted({str(row.get("axis_label")) for row in binding_rows}),
                "true_scorer_ready": section_ready,
                "score_claim": False,
            }
        )

    evidence_manifest = evidence_map.to_manifest(repo_root=repo_root)
    blockers.extend(str(item) for item in evidence_manifest.get("blockers", []))
    if binding_count == 0:
        blockers.append("section_scorer_evidence_no_bindings_matched_parser_sections")
    if unbound_sections:
        blockers.append("unbound_sections")
    true_ready = bool(sections) and ready_section_count == len(sections) and not blockers
    if true_ready:
        claim_strength = "true_scorer_section_evidence_bound_planning_no_score_claim"
    elif binding_count:
        claim_strength = "section_bound_scorer_evidence_incomplete_not_true_scorer_conditional_entropy"
    else:
        claim_strength = "proxy_not_true_scorer_conditional_entropy"

    return {
        "label": "scorer_section_evidence_conditioned",
        "claim_strength": claim_strength,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "current_bytes": int(current_bytes),
        "section_count": len(sections),
        "bound_section_count": sum(1 for row in section_rows if row["component_binding_count"]),
        "unbound_section_count": len(unbound_sections),
        "unbound_sections": sorted(unbound_sections),
        "component_binding_count": binding_count,
        "true_scorer_ready_binding_count": ready_count,
        "true_scorer_ready_section_count": ready_section_count,
        "true_scorer_conditioning_ready": true_ready,
        "axis_labels": sorted(axis_labels) or ["[axis-unknown]"],
        "sections": section_rows,
        "evidence_map": evidence_manifest,
        "blockers": sorted(set(blockers)),
    }


def _entropy_layer(
    label: str,
    groups: Mapping[str, bytes],
    *,
    current_bytes: int,
) -> dict[str, Any]:
    rows = []
    total_bits = 0.0
    total_symbols = 0
    for key, data in sorted(groups.items()):
        stats = _entropy_stats(data)
        total_bits += float(stats["floor_bits"])
        total_symbols += int(stats["symbol_count"])
        rows.append(
            {
                "group": key,
                "bytes": len(data),
                "sha256": sha256_bytes(data) if data else None,
                **stats,
            }
        )
    floor_bytes = total_bits / 8.0
    floor_ceil = _ceil_bits_to_bytes(total_bits)
    return {
        "label": label,
        "claim_strength": "parser_proven_empirical_entropy_floor",
        "group_count": len(rows),
        "symbol_count": total_symbols,
        "current_bytes": int(current_bytes),
        "floor_bits": _round(total_bits),
        "floor_bytes": _round(floor_bytes),
        "floor_bytes_ceil": floor_ceil,
        "gap_to_current_bytes": _round(current_bytes - floor_bytes),
        "gap_to_current_bytes_ceil": int(current_bytes) - int(floor_ceil),
        "groups": rows,
    }


def _sensitivity_map(
    sections: Sequence[_SectionBytes],
    *,
    chunk_size: int,
    repo_root: Path,
) -> dict[str, Any]:
    del repo_root  # reserved for future source bindings
    rows: list[dict[str, Any]] = []
    for section in sections:
        role_weight = ROLE_WEIGHTS.get(section.optimization_role, 0.50)
        for chunk_index, start in enumerate(range(0, len(section.data), chunk_size)):
            chunk = section.data[start : start + chunk_size]
            stats = _entropy_stats(chunk)
            gap = len(chunk) - int(stats["floor_bytes_ceil"])
            rows.append(
                {
                    "archive_label": section.archive_label,
                    "section_name": section.section_name,
                    "optimization_role": section.optimization_role,
                    "runtime_effect": section.runtime_effect,
                    "score_affecting_at_inflate": section.score_affecting_at_inflate,
                    "chunk_index": chunk_index,
                    "section_start": section.start,
                    "chunk_start": section.start + start,
                    "chunk_end": section.start + start + len(chunk),
                    "bytes": len(chunk),
                    "sha256": sha256_bytes(chunk),
                    "entropy_bits_per_byte": stats["bits_per_symbol"],
                    "entropy_floor_bytes_ceil": stats["floor_bytes_ceil"],
                    "compression_opportunity_bytes_ceil": gap,
                    "role_weight": role_weight,
                    "allocator_priority_score": _round(max(0, gap) * role_weight),
                    "score_claim": False,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["allocator_priority_score"]),
            str(row["archive_label"]),
            str(row["section_name"]),
            int(row["chunk_index"]),
        )
    )
    return {
        "schema": SENSITIVITY_SCHEMA,
        "schema_version": 1,
        "tool": TOOL,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "row_count": len(rows),
        "rows": rows,
        "top_rows": rows[:25],
        "role_weights": ROLE_WEIGHTS,
        "limitations": [
            "allocator priority is entropy/opportunity proxy",
            "not a scorer gradient or component-response sensitivity map",
        ],
    }


def _allocator_hook(sensitivity_map: Mapping[str, Any]) -> dict[str, Any]:
    top_rows = sensitivity_map.get("top_rows") if isinstance(sensitivity_map, Mapping) else []
    return {
        "schema": "scorer_conditional_mdl_allocator_hook_v1",
        "consumer": "tac.composition.registry.allocate_bits",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "recommended_priority_rows": top_rows,
        "blockers_before_binding": [
            "replace_entropy_proxy_with_component_response_or_penultimate_feature_map",
            "emit_byte_closed_codec_candidate",
            "run_exact_cuda_eval_before_pareto_binding",
        ],
    }


def _probe_disambiguator(verdict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "zen_floor_probe_disambiguator_v1",
        "tool_expected": "tools/probe_zen_floor_disambiguator.py",
        "score_claim": False,
        "static_floor_interpretation": "source_scorer_runtime_encoder_class_property",
        "dynamic_floor_interpretation": "substrate_engineering_scope_conditional",
        "z1_headroom_class": verdict.get("headroom_class"),
        "next_required_probe": verdict.get("next_required_probe"),
    }


def _autopilot_rows(verdict: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "lane_zen_floor_scorer_conditional_mdl_ablation_20260514",
            "family": "zen_floor_measurement",
            "predicted_score_delta": 0.0,
            "expected_information_gain": 4.0,
            "estimated_dispatch_cost_usd": 0.0,
            "target_axis": "mixed",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "blockers": [
                "measurement_complete_but_not_archive_candidate",
                str(verdict.get("next_required_probe") or "true_scorer_feature_binding_required"),
            ],
            "notes": (
                "[Z1 measurement; no score claim] consumes A1/PR106 payload entropy "
                f"headroom_class={verdict.get('headroom_class')}"
            ),
        },
        {
            "candidate_id": "lane_balle_conditional_entropy_bolt_on_20260520",
            "family": "balle_hyperprior_staircase_step_1",
            "predicted_score_delta": -0.003,
            "expected_information_gain": 2.5,
            "estimated_dispatch_cost_usd": 2.0,
            "target_axis": "rate",
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "promotion_eligible": False,
            "blockers": [
                "requires_z1_hyperprior_target_selection",
                "requires_byte_closed_archive_builder",
                "requires_lane_dispatch_claim_before_gpu",
            ],
            "notes": "[prediction; Z3 staircase step 1 queued after Z1 measurement]",
        },
    ]


def _verdict(
    *,
    unconditional: Mapping[str, Any],
    by_role: Mapping[str, Any],
    scorer_proxy: Mapping[str, Any],
    section_evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    role_gain = int(by_role.get("gap_to_current_bytes_ceil") or 0)
    proxy_gain = int(scorer_proxy.get("gap_to_current_bytes_ceil") or 0)
    true_feature_available = bool(section_evidence.get("true_scorer_conditioning_ready"))
    if role_gain > 4096:
        headroom = "parser_conditioned_headroom_large"
    elif role_gain > 1024:
        headroom = "parser_conditioned_headroom_measurable"
    else:
        headroom = "parser_conditioned_headroom_small"
    if not true_feature_available:
        next_probe = "byte_to_scorer_feature_binding_required"
    else:
        next_probe = "build_conditional_entropy_codec_candidate"
    return {
        "headroom_class": headroom,
        "unconditional_floor_bytes_ceil": unconditional.get("floor_bytes_ceil"),
        "parser_role_conditioned_floor_bytes_ceil": by_role.get("floor_bytes_ceil"),
        "parser_role_conditioned_gain_bytes_ceil": role_gain,
        "scorer_feature_proxy_floor_bytes_ceil": scorer_proxy.get("floor_bytes_ceil"),
        "scorer_feature_proxy_gain_bytes_ceil": proxy_gain,
        "scorer_section_evidence_claim_strength": section_evidence.get("claim_strength"),
        "scorer_section_evidence_binding_count": section_evidence.get("component_binding_count"),
        "scorer_section_evidence_axis_labels": section_evidence.get("axis_labels"),
        "true_scorer_feature_available": true_feature_available,
        "next_required_probe": next_probe,
        "archive_labels": [str(record.get("label")) for record in records],
        "interpretation": (
            "Parser-conditioned entropy estimates quantify real archive-grammar headroom. "
            "They do not prove a scorer-conditional coding gain yet. A true cooperative-"
            "receiver MDL estimate requires binding byte regions to scorer penultimate "
            "features, component-response curves, or a differentiable byte/section ablation."
        ),
        "next_actions": [
            "Use top sensitivity rows to choose Balle/hyperprior target sections for Z3.",
            "Attach scorer penultimate-feature saliency or component-response curves to convert proxy entropy into true scorer-conditioned entropy.",
            "Do not dispatch from this artifact directly; build a byte-closed codec candidate first.",
        ],
    }


def _manifest_axis_labels(
    records: Sequence[Mapping[str, Any]],
    section_evidence: Mapping[str, Any],
) -> list[str]:
    labels = set()
    for record in records:
        feature = record.get("scorer_feature_summary")
        if isinstance(feature, Mapping):
            labels.add(
                str(
                    feature.get("score_axis_label")
                    or axis_label(str(feature.get("score_axis") or "unknown"))
                )
            )
    for label in section_evidence.get("axis_labels", []):
        labels.add(str(label))
    return sorted(labels) or ["[axis-unknown]"]


def _group_sections(sections: Sequence[_SectionBytes], *, key: str) -> dict[str, bytes]:
    groups: dict[str, bytes] = {}
    for section in sections:
        group_key = getattr(section, key)
        groups[group_key] = groups.get(group_key, b"") + section.data
    return groups


def _score_affecting_at_inflate(optimization_role: str) -> bool:
    return optimization_role not in PROVENANCE_ONLY_ROLES


def _runtime_effect_for_section(section_name: str, optimization_role: str) -> str:
    if optimization_role in PROVENANCE_ONLY_ROLES:
        return "training_provenance_only_not_score_affecting"
    if section_name.endswith("header") or "header" in section_name:
        return "runtime_parse_metadata"
    if optimization_role == "control_or_metadata":
        return "runtime_control_metadata"
    if optimization_role == "decoder_weight_stream":
        return "score_affecting_decoder_weights"
    if optimization_role == "latent_stream":
        return "score_affecting_latents"
    return "runtime_consumption_unclassified"


def _entropy_stats(data: bytes) -> dict[str, Any]:
    n = len(data)
    if n == 0:
        return {
            "symbol_count": 0,
            "alphabet_size": 0,
            "bits_per_symbol": 0.0,
            "floor_bits": 0.0,
            "floor_bytes": 0.0,
            "floor_bytes_ceil": 0,
        }
    counts = Counter(data)
    h = 0.0
    for count in counts.values():
        p = count / n
        h -= p * math.log2(p)
    bits = h * n
    return {
        "symbol_count": n,
        "alphabet_size": len(counts),
        "bits_per_symbol": _round(h),
        "floor_bits": _round(bits),
        "floor_bytes": _round(bits / 8.0),
        "floor_bytes_ceil": _ceil_bits_to_bytes(bits),
    }


def _ceil_bits_to_bytes(bits: float) -> int:
    if bits <= 0.0:
        return 0
    return int(math.ceil(bits / 8.0))


def _round(value: float) -> float:
    return round(float(value), 12)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScorerConditionalMdlError(f"{label} must be an integer")
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_score_axis(path: Path, payload: Mapping[str, Any]) -> str:
    axis = payload.get("score_axis")
    if isinstance(axis, str) and axis:
        return axis
    text = path.as_posix().lower()
    if "modal_auth_eval_cpu" in text or "contest-cpu" in text:
        return "contest_cpu"
    if "modal_auth_eval" in text or "contest-cuda" in text:
        return "contest_cuda"
    if "lightning_batch" in text and "exact_eval" in text:
        return "contest_cuda"
    if "mps" in text:
        return "mps"
    if "macos" in text:
        return "macos_cpu_advisory"
    return "unknown"


def _log10_bucket(value: float | None) -> str:
    if value is None or value <= 0.0:
        return "unknown"
    exponent = math.floor(math.log10(value))
    return f"1e{exponent}"


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_archive_spec(value: str) -> ArchiveInput:
    """Parse ``label=path[,parser=...]`` CLI syntax."""

    if "=" not in value:
        raise ScorerConditionalMdlError(
            "--archive must use label=path or label=path,parser=name syntax"
        )
    label, rest = value.split("=", 1)
    parser = PARSER_AUTO
    path_text = rest
    if ",parser=" in rest:
        path_text, parser = rest.split(",parser=", 1)
    return ArchiveInput(label=label.strip(), archive_path=Path(path_text), parser=parser.strip())


def attach_eval_jsons(
    archives: Iterable[ArchiveInput],
    eval_specs: Iterable[str],
) -> list[ArchiveInput]:
    """Attach ``label=eval.json`` paths to archive specs."""

    eval_by_label: dict[str, Path] = {}
    for spec in eval_specs:
        if "=" not in spec:
            raise ScorerConditionalMdlError("--eval-json must use label=path syntax")
        label, path = spec.split("=", 1)
        eval_by_label[label.strip()] = Path(path)
    out = []
    for archive in archives:
        out.append(
            ArchiveInput(
                label=archive.label,
                archive_path=archive.archive_path,
                parser=archive.parser,
                eval_json_path=eval_by_label.get(archive.label),
            )
        )
    return out
