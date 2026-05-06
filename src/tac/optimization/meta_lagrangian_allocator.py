"""Meta-Lagrangian atom ranking for contest archive optimization.

This module is the lightweight bridge between byte-only opportunities and
scorer-aware hard-pair repair work. It does not claim score evidence; it ranks
candidate atoms under the official contest score formula so downstream tools
can choose which archive builders deserve exact CUDA evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.repo_io import sha256_file

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25 / CONTEST_ORIGINAL_BYTES
SCHEMA_VERSION = 1
NON_RANKABLE_EVIDENCE_GRADES = {
    "invalid",
    "prediction",
    "external",
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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _archive_manifest_custody(path_value: str, sha256_value: str) -> dict[str, Any]:
    if not path_value and not sha256_value:
        return {
            "path": "",
            "sha256": "",
            "exists": False,
            "is_file": False,
            "sha256_valid": False,
            "sha256_matches": False,
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
    if path_value and not exists:
        blockers.append("archive_manifest_path_missing")
    elif exists and not is_file:
        blockers.append("archive_manifest_path_not_file")
    elif is_file and _is_sha256(sha256_value):
        actual_sha = sha256_file(path)
        if actual_sha != sha256_value:
            blockers.append("archive_manifest_sha256_mismatch")
    verified = bool(is_file and actual_sha == sha256_value)
    return {
        "path": path_value,
        "sha256": sha256_value,
        "exists": exists,
        "is_file": is_file,
        "sha256_valid": _is_sha256(sha256_value),
        "sha256_actual": actual_sha,
        "sha256_matches": verified,
        "verified": verified,
        "blockers": blockers,
    }


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
    rankable, rank_blockers = _rankable_atom(atom, evidence_grade, confidence)
    family = str(atom.get("family") or atom.get("atom_family") or "unknown")
    family_group = str(atom.get("family_group") or family)
    conflicts_with_families = _string_list(atom.get("conflicts_with_families"))
    conflicts_with_atoms = _string_list(atom.get("conflicts_with_atoms"))
    archive_manifest_path = str(atom.get("archive_manifest_path") or "")
    archive_manifest_sha256 = str(atom.get("archive_manifest_sha256") or "")
    archive_manifest_custody = _archive_manifest_custody(
        archive_manifest_path,
        archive_manifest_sha256,
    )
    byte_closed_archive_manifest_attached = bool(archive_manifest_custody["verified"])
    requested_dispatchable = bool(
        atom.get("dispatchable") is True or atom.get("ready_for_exact_eval_dispatch") is True
    )
    dispatch_blockers = [
        "planning_only_lagrangian_atom",
        "requires_byte_closed_archive",
        "requires_exact_cuda_auth_eval",
        *rank_blockers,
        *archive_manifest_custody["blockers"],
    ]
    if requested_dispatchable and not byte_closed_archive_manifest_attached:
        dispatch_blockers.append("requested_dispatchable_without_byte_closed_archive_manifest")
    if atom.get("score_claim") is True:
        dispatch_blockers.append("source_atom_score_claim_true")
    seg_score = 100.0 * expected_seg_delta
    pose_score = pose_score_delta(base_pose_dist, expected_pose_delta)
    component_score = confidence * (seg_score + pose_score)
    rate_score = rate_score_delta(byte_delta)
    total = component_score + rate_score
    return {
        "atom_id": atom_id,
        "family": family,
        "family_group": family_group,
        "conflicts_with_families": conflicts_with_families,
        "conflicts_with_atoms": conflicts_with_atoms,
        "score_claim": False,
        "evidence_grade": evidence_grade,
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
        "allocation_inference": bool(atom.get("allocation_inference", False)),
        "evidence_source_path": atom.get("evidence_source_path", ""),
        "evidence_source_sha256": atom.get("evidence_source_sha256", ""),
        "source_archive_sha256": atom.get("source_archive_sha256", ""),
        "archive_manifest_path": archive_manifest_path,
        "archive_manifest_sha256": archive_manifest_sha256,
        "archive_manifest_custody": archive_manifest_custody,
        "byte_closed_archive_manifest_attached": byte_closed_archive_manifest_attached,
        "requested_dispatchable": requested_dispatchable,
        "archive_ready_for_stack_review": bool(rankable and byte_closed_archive_manifest_attached),
        "dispatchable": False,
        "dispatch_blockers": dispatch_blockers,
    }


def build_atom_ledger(
    atoms: Iterable[Mapping[str, Any]],
    *,
    base_pose_dist: float,
    source: str,
) -> dict[str, Any]:
    """Rank atoms by expected contest-score delta."""

    rows = [expected_atom_score_delta(atom, base_pose_dist=base_pose_dist) for atom in atoms]
    rows.sort(
        key=lambda row: (
            not bool(row["rankable"]),
            float(row["expected_total_score_delta"]),
            int(row["byte_delta"]),
            str(row["atom_id"]),
        )
    )
    family_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    archive_manifest_attached_count = 0
    requested_dispatchable_refused_count = 0
    for row in rows:
        family = str(row["family_group"])
        family_counts[family] = family_counts.get(family, 0) + 1
        archive_manifest_attached_count += int(bool(row["byte_closed_archive_manifest_attached"]))
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
        "byte_closed_archive_manifest_attached_count": archive_manifest_attached_count,
        "requested_dispatchable_refused_count": requested_dispatchable_refused_count,
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
