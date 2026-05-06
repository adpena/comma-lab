"""Meta-Lagrangian atom ranking for contest archive optimization.

This module is the lightweight bridge between byte-only opportunities and
scorer-aware hard-pair repair work. It does not claim score evidence; it ranks
candidate atoms under the official contest score formula so downstream tools
can choose which archive builders deserve exact CUDA evaluation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
SCHEMA_VERSION = 2
NON_RANKABLE_EVIDENCE_GRADES = {
    "invalid",
    "prediction",
    "external",
}
PROXY_EVIDENCE_GRADE_MARKERS = (
    "planning",
    "proxy",
    "prediction",
)
PARETO_EPS = 1e-12
PARETO_MINIMIZE_OBJECTIVES = (
    "expected_total_score_delta",
    "byte_delta",
    "expected_seg_dist_delta",
    "expected_pose_dist_delta",
)
SELECTION_PENALTIES = {
    "non_rankable_atom": 1_000.0,
    "missing_byte_closed_archive_manifest": 100.0,
    "proxy_row": 50.0,
    "pareto_ineligible_atom": 25.0,
    "kkt_not_ready_for_field_planning": 10.0,
    "pareto_dominated_atom": 5.0,
    "requested_dispatchable_refused": 1.0,
}


class MetaLagrangianError(ValueError):
    """Raised when atom allocation inputs are invalid."""


def rate_score_delta(byte_delta: int | float) -> float:
    """Return official rate-term score delta for a charged byte delta."""

    return float(byte_delta) * RATE_SCORE_PER_BYTE


def pose_score_delta(base_pose_dist: float, pose_dist_delta: float) -> float:
    """Return exact official pose-term delta around ``base_pose_dist``."""

    base = float(base_pose_dist)
    delta = float(pose_dist_delta)
    if base < 0:
        raise MetaLagrangianError("base_pose_dist must be non-negative")
    if base + delta < 0:
        raise MetaLagrangianError("pose_dist_delta makes pose distance negative")
    return math.sqrt(10.0 * (base + delta)) - math.sqrt(10.0 * base)


def _rankable_atom(atom: Mapping[str, Any], evidence_grade: str, confidence: float) -> tuple[bool, list[str]]:
    """Return whether an atom can participate in planning priority order."""

    blockers: list[str] = []
    grade = evidence_grade.strip().lower()
    if grade in NON_RANKABLE_EVIDENCE_GRADES or "invalid" in grade:
        blockers.append("non_rankable_evidence_grade")
    if atom.get("raw_equal") is False:
        blockers.append("raw_output_not_byte_equivalent")
    if atom.get("rankable") is False:
        blockers.append("source_marked_non_rankable")
    if atom.get("allocation_inference") is True:
        blockers.append("allocated_global_response_not_rankable")
    if confidence <= 0.0 and int(atom.get("byte_delta", 0)) < 0:
        blockers.append("byte_savings_without_trusted_equivalence")
    return not blockers, blockers


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _sorted_unique_string_list(value: Any) -> list[str]:
    return sorted(set(_string_list(value)))


def _unique_ordered_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _optional_non_negative_float(atom: Mapping[str, Any], keys: Iterable[str], *, label: str) -> tuple[float, list[str]]:
    for key in keys:
        if key not in atom or atom.get(key) is None:
            continue
        value = atom.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MetaLagrangianError(f"{label} must be numeric when provided")
        out = float(value)
        if out < 0.0:
            raise MetaLagrangianError(f"{label} must be non-negative when provided")
        return round(out, 12), [key]
    return 0.0, []


def _uncertainty_fields(atom: Mapping[str, Any]) -> dict[str, Any]:
    """Carry nearby Bayesian-design uncertainty fields into atom rows."""

    eig, eig_sources = _optional_non_negative_float(
        atom,
        ("expected_information_gain_nats", "information_gain_nats"),
        label="expected_information_gain_nats",
    )
    variance, variance_sources = _optional_non_negative_float(
        atom,
        ("expected_score_variance", "predicted_score_variance", "score_variance"),
        label="expected_score_variance",
    )
    noise, noise_sources = _optional_non_negative_float(
        atom,
        ("observation_noise_variance",),
        label="observation_noise_variance",
    )
    reduction = atom.get("family_uncertainty_reduction")
    reduction_total_variance = 0.0
    reduction_sources: list[str] = []
    if isinstance(reduction, Mapping):
        nested_eig, nested_eig_sources = _optional_non_negative_float(
            reduction,
            ("total_information_gain_nats",),
            label="family_uncertainty_reduction.total_information_gain_nats",
        )
        nested_variance, nested_variance_sources = _optional_non_negative_float(
            reduction,
            ("total_variance_reduction",),
            label="family_uncertainty_reduction.total_variance_reduction",
        )
        if not eig_sources:
            eig = nested_eig
            eig_sources = [f"family_uncertainty_reduction.{source}" for source in nested_eig_sources]
        reduction_total_variance = nested_variance
        reduction_sources = [
            f"family_uncertainty_reduction.{source}" for source in nested_variance_sources
        ]
    source_fields = [*eig_sources, *variance_sources, *noise_sources, *reduction_sources]
    return {
        "expected_information_gain_nats": eig,
        "expected_score_variance": variance,
        "observation_noise_variance": noise,
        "expected_uncertainty_reduction": {
            "total_variance_reduction": reduction_total_variance,
            "source_fields": source_fields,
            "has_uncertainty_signal": bool(eig > 0.0 or variance > 0.0 or reduction_total_variance > 0.0),
        },
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _archive_manifest_payload_custody(path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "json_valid": False,
            "archive_sha256": "",
            "archive_sha256_valid": False,
            "archive_bytes": None,
            "archive_bytes_valid": False,
            "score_claim": None,
            "score_claim_false": False,
            "blockers": ["archive_manifest_json_invalid"],
        }
    if not isinstance(payload, dict):
        return {
            "json_valid": True,
            "archive_sha256": "",
            "archive_sha256_valid": False,
            "archive_bytes": None,
            "archive_bytes_valid": False,
            "score_claim": None,
            "score_claim_false": False,
            "blockers": ["archive_manifest_not_object"],
        }

    archive = payload.get("archive")
    archive_obj = archive if isinstance(archive, Mapping) else {}
    archive_sha256 = str(
        archive_obj.get("sha256")
        or payload.get("archive_sha256")
        or payload.get("candidate_archive_sha256")
        or ""
    )
    archive_bytes = (
        archive_obj.get("bytes")
        if "bytes" in archive_obj
        else payload.get("archive_bytes", payload.get("candidate_archive_bytes"))
    )
    archive_sha256_valid = _is_sha256(archive_sha256)
    archive_bytes_valid = isinstance(archive_bytes, int) and not isinstance(archive_bytes, bool) and archive_bytes > 0
    if not archive_sha256_valid:
        blockers.append("archive_manifest_archive_sha256_missing_or_invalid")
    if not archive_bytes_valid:
        blockers.append("archive_manifest_archive_bytes_missing_or_invalid")
    score_claim = payload.get("score_claim")
    score_claim_false = score_claim is not True
    if not score_claim_false:
        blockers.append("archive_manifest_score_claim_true")
    return {
        "json_valid": True,
        "archive_sha256": archive_sha256,
        "archive_sha256_valid": archive_sha256_valid,
        "archive_bytes": archive_bytes if isinstance(archive_bytes, int) and not isinstance(archive_bytes, bool) else None,
        "archive_bytes_valid": archive_bytes_valid,
        "score_claim": score_claim,
        "score_claim_false": score_claim_false,
        "blockers": blockers,
    }


def _archive_manifest_custody(path_value: str, sha256_value: str) -> dict[str, Any]:
    if not path_value and not sha256_value:
        return {
            "path": "",
            "sha256": "",
            "exists": False,
            "is_file": False,
            "sha256_valid": False,
            "sha256_matches": False,
            "manifest_json_valid": False,
            "archive_sha256": "",
            "archive_sha256_valid": False,
            "archive_bytes": None,
            "archive_bytes_valid": False,
            "score_claim_false": False,
            "verified": False,
            "blockers": [],
        }
    blockers: list[str] = []
    if not path_value:
        blockers.append("archive_manifest_path_missing")
    if not _is_sha256(sha256_value):
        blockers.append("archive_manifest_sha256_missing_or_invalid")
    path = Path(path_value) if path_value else Path()
    exists = bool(path_value and path.exists())
    is_file = bool(exists and path.is_file())
    actual_sha = ""
    payload_custody: dict[str, Any] = {
        "json_valid": False,
        "archive_sha256": "",
        "archive_sha256_valid": False,
        "archive_bytes": None,
        "archive_bytes_valid": False,
        "score_claim_false": False,
        "blockers": [],
    }
    if path_value and not exists:
        blockers.append("archive_manifest_path_missing")
    elif exists and not is_file:
        blockers.append("archive_manifest_path_not_file")
    elif is_file and _is_sha256(sha256_value):
        actual_sha = sha256_file(path)
        if actual_sha != sha256_value:
            blockers.append("archive_manifest_sha256_mismatch")
        else:
            payload_custody = _archive_manifest_payload_custody(path)
            blockers.extend(payload_custody["blockers"])
    sha256_matches = bool(actual_sha and actual_sha == sha256_value)
    verified = bool(
        is_file
        and sha256_matches
        and payload_custody["json_valid"]
        and payload_custody["archive_sha256_valid"]
        and payload_custody["archive_bytes_valid"]
        and payload_custody["score_claim_false"]
        and not blockers
    )
    return {
        "path": path_value,
        "sha256": sha256_value,
        "exists": exists,
        "is_file": is_file,
        "sha256_valid": _is_sha256(sha256_value),
        "sha256_actual": actual_sha,
        "sha256_matches": sha256_matches,
        "manifest_json_valid": bool(payload_custody["json_valid"]),
        "archive_sha256": str(payload_custody["archive_sha256"]),
        "archive_sha256_valid": bool(payload_custody["archive_sha256_valid"]),
        "archive_bytes": payload_custody["archive_bytes"],
        "archive_bytes_valid": bool(payload_custody["archive_bytes_valid"]),
        "score_claim_false": bool(payload_custody["score_claim_false"]),
        "verified": verified,
        "blockers": _unique_ordered_strings(blockers),
    }


def _pareto_objectives(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "expected_total_score_delta": float(row["expected_total_score_delta"]),
        "byte_delta": float(row["byte_delta"]),
        "expected_seg_dist_delta": float(row["expected_seg_dist_delta"]),
        "expected_pose_dist_delta": float(row["expected_pose_dist_delta"]),
        "confidence": float(row["confidence"]),
        "archive_ready_for_stack_review": float(bool(row["archive_ready_for_stack_review"])),
    }


def _dominates_pareto(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Return whether ``a`` dominates ``b`` within one Pareto scope.

    Lower is better for score, byte, SegNet, and PoseNet deltas. Higher is
    better for confidence and archive-custody readiness, so an uncustodied
    prediction cannot hide a byte-closed candidate with weaker proxy deltas.
    """

    if not bool(a["pareto_eligible"]) or not bool(b["pareto_eligible"]):
        return False
    if str(a["pareto_scope"]) != str(b["pareto_scope"]):
        return False
    strictly_better = False
    for objective in PARETO_MINIMIZE_OBJECTIVES:
        av = float(a[objective])
        bv = float(b[objective])
        if av > bv + PARETO_EPS:
            return False
        strictly_better = strictly_better or av < bv - PARETO_EPS
    for objective in ("confidence", "archive_ready_for_stack_review"):
        av = float(a[objective])
        bv = float(b[objective])
        if av < bv - PARETO_EPS:
            return False
        strictly_better = strictly_better or av > bv + PARETO_EPS
    return strictly_better


def _annotate_pareto_frontier(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominated_count = 0
    frontier_count = 0
    scope_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        row["pareto_eligible"] = bool(
            row["rankable"]
            and row["byte_closed_archive_manifest_attached"]
            and not row["proxy_row"]
        )
        row["pareto_frontier"] = bool(row["pareto_eligible"])
        row["pareto_dominated_by"] = []
        row["pareto_objectives"] = _pareto_objectives(row)
    for row in rows:
        if not bool(row["pareto_eligible"]):
            row["pareto_frontier"] = False
            continue
        dominators = [
            str(other["atom_id"])
            for other in rows
            if other is not row and _dominates_pareto(other, row)
        ]
        if dominators:
            row["pareto_frontier"] = False
            row["pareto_dominated_by"] = sorted(dominators)
            dominated_count += 1
        else:
            frontier_count += 1
        scope = str(row["pareto_scope"])
        stats = scope_counts.setdefault(scope, {"rankable": 0, "frontier": 0, "dominated": 0})
        stats["rankable"] += 1
        stats["frontier"] += int(bool(row["pareto_frontier"]))
        stats["dominated"] += int(not bool(row["pareto_frontier"]))
    return {
        "objective_direction": {
            "expected_total_score_delta": "min",
            "byte_delta": "min",
            "expected_seg_dist_delta": "min",
            "expected_pose_dist_delta": "min",
            "confidence": "max",
            "archive_ready_for_stack_review": "max",
        },
        "scope_default": "family_group",
        "eligibility": "rankable_verified_byte_closed_archive_manifest_and_non_proxy",
        "rankable_frontier_count": frontier_count,
        "rankable_dominated_count": dominated_count,
        "scope_counts": dict(sorted(scope_counts.items())),
    }


def _selection_penalty_terms(row: Mapping[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    if row.get("rankable") is not True:
        terms["non_rankable_atom"] = SELECTION_PENALTIES["non_rankable_atom"]
    if row.get("byte_closed_archive_manifest_attached") is not True:
        terms["missing_byte_closed_archive_manifest"] = SELECTION_PENALTIES[
            "missing_byte_closed_archive_manifest"
        ]
    if row.get("proxy_row") is True:
        terms["proxy_row"] = SELECTION_PENALTIES["proxy_row"]
    if row.get("pareto_eligible") is not True:
        terms["pareto_ineligible_atom"] = SELECTION_PENALTIES["pareto_ineligible_atom"]
    elif row.get("pareto_frontier") is not True:
        terms["pareto_dominated_atom"] = SELECTION_PENALTIES["pareto_dominated_atom"]
    if row.get("kkt_ready_for_field_planning") is not True:
        terms["kkt_not_ready_for_field_planning"] = SELECTION_PENALTIES[
            "kkt_not_ready_for_field_planning"
        ]
    if row.get("requested_dispatchable") is True and row.get("dispatchable") is not True:
        terms["requested_dispatchable_refused"] = SELECTION_PENALTIES[
            "requested_dispatchable_refused"
        ]
    return terms


def _annotate_selection_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        terms = _selection_penalty_terms(row)
        penalty = sum(terms.values())
        row["selection_penalty_terms"] = {key: round(value, 12) for key, value in sorted(terms.items())}
        row["selection_penalty_score_delta"] = round(penalty, 12)
        row["selection_score_delta"] = round(
            float(row["expected_total_score_delta"]) + penalty,
            12,
        )
        row["selection_blockers"] = list(row["selection_penalty_terms"])
        row["selection_policy"] = (
            "planning-only; rankable, non-proxy, byte-closed Pareto frontier rows "
            "sort before raw expected-score deltas; information gain is a deterministic tie-breaker"
        )


def _proxy_row(evidence_grade: str, atom: Mapping[str, Any]) -> bool:
    grade = evidence_grade.strip().lower()
    return bool(
        atom.get("proxy_row") is True
        or any(marker in grade for marker in PROXY_EVIDENCE_GRADE_MARKERS)
    )


def expected_atom_score_delta(
    atom: Mapping[str, Any],
    *,
    base_pose_dist: float,
) -> dict[str, Any]:
    """Compute a planning-only contest-score delta for one atom.

    Expected SegNet/PoseNet deltas are confidence-weighted development
    predictions. The rate delta is exact arithmetic for the charged byte delta.
    """

    atom_id = str(atom.get("atom_id") or "")
    if not atom_id:
        raise MetaLagrangianError("atom missing atom_id")
    byte_delta = int(atom.get("byte_delta", 0))
    confidence = float(atom.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise MetaLagrangianError(f"{atom_id}: confidence must be in [0, 1]")
    expected_seg_delta = float(atom.get("expected_seg_dist_delta", 0.0))
    expected_pose_delta = float(atom.get("expected_pose_dist_delta", 0.0))
    evidence_grade = str(atom.get("evidence_grade", "prediction"))
    uncertainty = _uncertainty_fields(atom)
    family = str(atom.get("family") or atom.get("atom_family") or atom.get("family_group") or "unknown")
    family_group = str(atom.get("family_group") or family)
    pareto_scope = str(atom.get("pareto_scope") or family_group)
    conflicts_with_families = _sorted_unique_string_list(atom.get("conflicts_with_families"))
    conflicts_with_atoms = _sorted_unique_string_list(atom.get("conflicts_with_atoms"))
    interaction_assumptions = _sorted_unique_string_list(atom.get("interaction_assumptions"))
    rankable, rank_blockers = _rankable_atom(atom, evidence_grade, confidence)
    if family == "unknown":
        rank_blockers.append("missing_atom_family")
        rankable = False
    archive_manifest_path = str(atom.get("archive_manifest_path") or "")
    archive_manifest_sha256 = str(atom.get("archive_manifest_sha256") or "")
    archive_manifest_custody = _archive_manifest_custody(
        archive_manifest_path,
        archive_manifest_sha256,
    )
    byte_closed_archive_manifest_attached = bool(archive_manifest_custody["verified"])
    proxy_row = _proxy_row(evidence_grade, atom)
    requested_dispatchable = bool(
        atom.get("dispatchable") is True or atom.get("ready_for_exact_eval_dispatch") is True
    )
    dispatch_blockers = [
        "planning_only_lagrangian_atom",
        "requires_exact_cuda_auth_eval",
        *rank_blockers,
        *archive_manifest_custody["blockers"],
    ]
    if not byte_closed_archive_manifest_attached:
        dispatch_blockers.append("requires_byte_closed_archive")
    if proxy_row:
        dispatch_blockers.append("proxy_row_not_dispatchable")
    if not interaction_assumptions:
        dispatch_blockers.append("missing_interaction_assumptions")
    if requested_dispatchable and not byte_closed_archive_manifest_attached:
        dispatch_blockers.append("requested_dispatchable_without_byte_closed_archive_manifest")
    if requested_dispatchable and proxy_row:
        dispatch_blockers.append("requested_dispatchable_proxy_row_refused")
    if requested_dispatchable:
        dispatch_blockers.append("requested_dispatchable_requires_external_exact_readiness_artifact")
    if atom.get("score_claim") is True:
        dispatch_blockers.append("source_atom_score_claim_true")
    kkt_blockers = [
        *rank_blockers,
        *archive_manifest_custody["blockers"],
    ]
    if not byte_closed_archive_manifest_attached:
        kkt_blockers.append("missing_byte_closed_archive_manifest")
    if proxy_row:
        kkt_blockers.append("proxy_evidence_not_kkt_ready")
    if not interaction_assumptions:
        kkt_blockers.append("missing_interaction_assumptions")
    kkt_ready_for_field_planning = bool(not kkt_blockers)
    seg_score = 100.0 * expected_seg_delta
    pose_score = pose_score_delta(base_pose_dist, expected_pose_delta)
    component_score = confidence * (seg_score + pose_score)
    rate_score = rate_score_delta(byte_delta)
    total = component_score + rate_score
    row = {
        "atom_id": atom_id,
        "family": family,
        "family_group": family_group,
        "pareto_scope": pareto_scope,
        "conflicts_with_families": conflicts_with_families,
        "conflicts_with_atoms": conflicts_with_atoms,
        "interaction_assumptions": interaction_assumptions,
        "score_claim": False,
        "evidence_grade": evidence_grade,
        "proxy_row": proxy_row,
        "expected_information_gain_nats": uncertainty["expected_information_gain_nats"],
        "expected_score_variance": uncertainty["expected_score_variance"],
        "observation_noise_variance": uncertainty["observation_noise_variance"],
        "expected_uncertainty_reduction": uncertainty["expected_uncertainty_reduction"],
        "byte_delta": byte_delta,
        "rate_score_delta": round(rate_score, 12),
        "expected_seg_dist_delta": expected_seg_delta,
        "expected_pose_dist_delta": expected_pose_delta,
        "seg_score_delta_confidence_weighted": round(confidence * seg_score, 12),
        "pose_score_delta_confidence_weighted": round(confidence * pose_score, 12),
        "expected_total_score_delta": round(total, 12),
        "rankable": rankable,
        "confidence": confidence,
        "pair_support": list(atom.get("pair_support") or []),
        "hard_pair_support": list(atom.get("hard_pair_support") or []),
        "class_support": list(atom.get("class_support") or []),
        "geometry_priors": list(atom.get("geometry_priors") or []),
        "openpilot_priors": list(atom.get("openpilot_priors") or []),
        "research_basis_ids": list(atom.get("research_basis_ids") or []),
        "allocation_inference": bool(atom.get("allocation_inference", False)),
        "evidence_source_path": atom.get("evidence_source_path", ""),
        "evidence_source_sha256": atom.get("evidence_source_sha256", ""),
        "source_archive_sha256": atom.get("source_archive_sha256", ""),
        "archive_manifest_path": archive_manifest_path,
        "archive_manifest_sha256": archive_manifest_sha256,
        "archive_manifest_custody": archive_manifest_custody,
        "byte_closed_archive_manifest_attached": byte_closed_archive_manifest_attached,
        "requested_dispatchable": requested_dispatchable,
        "archive_ready_for_stack_review": bool(rankable and byte_closed_archive_manifest_attached and not proxy_row),
        "pareto_eligible": bool(rankable and byte_closed_archive_manifest_attached and not proxy_row),
        "pareto_frontier": bool(rankable and byte_closed_archive_manifest_attached and not proxy_row),
        "pareto_dominated_by": [],
        "pareto_objectives": {},
        "kkt_ready_for_field_planning": kkt_ready_for_field_planning,
        "kkt_blockers": _unique_ordered_strings(kkt_blockers),
        "selection_penalty_terms": {},
        "selection_penalty_score_delta": 0.0,
        "selection_score_delta": round(total, 12),
        "selection_blockers": [],
        "selection_policy": "pending_pareto_annotation",
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "dispatch_blockers": _unique_ordered_strings(dispatch_blockers),
    }
    row["pareto_objectives"] = _pareto_objectives(row)
    _annotate_selection_scores([row])
    return row


def build_atom_ledger(
    atoms: Iterable[Mapping[str, Any]],
    *,
    base_pose_dist: float,
    source: str,
) -> dict[str, Any]:
    """Rank atoms by expected contest-score delta."""

    rows = [expected_atom_score_delta(atom, base_pose_dist=base_pose_dist) for atom in atoms]
    pareto_summary = _annotate_pareto_frontier(rows)
    _annotate_selection_scores(rows)
    rows.sort(
        key=lambda row: (
            not bool(row["rankable"]),
            not bool(row["byte_closed_archive_manifest_attached"]),
            bool(row["proxy_row"]),
            not bool(row["pareto_eligible"]),
            not bool(row["pareto_frontier"]),
            not bool(row["kkt_ready_for_field_planning"]),
            float(row["selection_score_delta"]),
            -float(row["expected_information_gain_nats"]),
            int(row["byte_delta"]),
            str(row["family_group"]),
            str(row["pareto_scope"]),
            str(row["atom_id"]),
        )
    )
    family_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    archive_manifest_attached_count = 0
    requested_dispatchable_refused_count = 0
    pareto_eligible_count = 0
    kkt_ready_count = 0
    proxy_row_count = 0
    for row in rows:
        family = str(row["family_group"])
        family_counts[family] = family_counts.get(family, 0) + 1
        archive_manifest_attached_count += int(bool(row["byte_closed_archive_manifest_attached"]))
        pareto_eligible_count += int(bool(row["pareto_eligible"]))
        kkt_ready_count += int(bool(row["kkt_ready_for_field_planning"]))
        proxy_row_count += int(bool(row["proxy_row"]))
        requested_dispatchable_refused_count += int(
            bool(row["requested_dispatchable"]) and not bool(row["dispatchable"])
        )
        for conflict in row["conflicts_with_families"]:
            conflict_counts[conflict] = conflict_counts.get(conflict, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.optimization.meta_lagrangian_allocator.build_atom_ledger",
        "source": source,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "base_pose_dist": float(base_pose_dist),
        "atom_count": len(rows),
        "family_group_counts": dict(sorted(family_counts.items())),
        "conflict_family_counts": dict(sorted(conflict_counts.items())),
        "pareto_summary": pareto_summary,
        "byte_closed_archive_manifest_attached_count": archive_manifest_attached_count,
        "pareto_eligible_count": pareto_eligible_count,
        "kkt_ready_for_field_planning_count": kkt_ready_count,
        "proxy_row_count": proxy_row_count,
        "requested_dispatchable_refused_count": requested_dispatchable_refused_count,
        "byte_closed_manifest_required_for_pareto": True,
        "byte_closed_manifest_required_for_kkt": True,
        "proxy_rows_dispatchable": False,
        "selection_policy": (
            "penalize_non_rankable_proxy_non_byte_closed_dominated_and_kkt_blocked_rows"
        ),
        "selection_penalties": dict(sorted(SELECTION_PENALTIES.items())),
        "rows": rows,
        "dispatch_blockers": [
            "planning_only_atom_ranking",
            "requires_stack_interaction_review",
            "requires_exact_cuda_auth_eval",
        ],
    }


def atoms_from_hnerv_decoder_recode_profile(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a structural-recode profile into rate-only allocation atoms."""

    label = str(profile.get("source_label") or "hnerv")
    source_archive_sha256 = str(profile.get("source_archive_sha256") or "")
    atoms = []
    for row in profile.get("variants") or []:
        if not isinstance(row, Mapping):
            continue
        variant = str(row.get("variant") or "variant")
        atoms.append(
            {
                "atom_id": f"{label}:decoder_recode:{variant}",
                "family": "hnerv_decoder_rate_recode",
                "family_group": "hnerv_rate_equivalent_recode",
                "conflicts_with_families": [],
                "conflicts_with_atoms": [],
                "byte_delta": int(row.get("byte_delta_vs_source_section", 0)),
                "expected_seg_dist_delta": 0.0,
                "expected_pose_dist_delta": 0.0,
                "confidence": 1.0 if row.get("raw_equal") is True else 0.0,
                "evidence_grade": "empirical_byte_raw_equal" if row.get("raw_equal") else "invalid",
                "hard_pair_support": [],
                "pair_support": [],
                "class_support": [],
                "geometry_priors": [],
                "openpilot_priors": [],
                "source_archive_sha256": source_archive_sha256,
            }
        )
    return atoms


__all__ = [
    "CONTEST_ORIGINAL_BYTES",
    "RATE_SCORE_PER_BYTE",
    "SCHEMA_VERSION",
    "MetaLagrangianError",
    "atoms_from_hnerv_decoder_recode_profile",
    "build_atom_ledger",
    "expected_atom_score_delta",
    "pose_score_delta",
    "rate_score_delta",
]
