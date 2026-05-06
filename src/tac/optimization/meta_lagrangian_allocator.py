"""Meta-Lagrangian atom ranking for contest archive optimization.

This module is the lightweight bridge between byte-only opportunities and
scorer-aware hard-pair repair work. It does not claim score evidence; it ranks
candidate atoms under the official contest score formula so downstream tools
can choose which archive builders deserve exact CUDA evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

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
    seg_score = 100.0 * expected_seg_delta
    pose_score = pose_score_delta(base_pose_dist, expected_pose_delta)
    component_score = confidence * (seg_score + pose_score)
    rate_score = rate_score_delta(byte_delta)
    total = component_score + rate_score
    return {
        "atom_id": atom_id,
        "family": atom.get("family", "unknown"),
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
        "dispatchable": False,
        "dispatch_blockers": [
            "planning_only_lagrangian_atom",
            "requires_byte_closed_archive",
            "requires_exact_cuda_auth_eval",
            *rank_blockers,
        ],
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
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.optimization.meta_lagrangian_allocator.build_atom_ledger",
        "source": source,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "base_pose_dist": float(base_pose_dist),
        "atom_count": len(rows),
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
    atoms = []
    for row in profile.get("variants") or []:
        if not isinstance(row, Mapping):
            continue
        variant = str(row.get("variant") or "variant")
        atoms.append(
            {
                "atom_id": f"{label}:decoder_recode:{variant}",
                "family": "hnerv_decoder_rate_recode",
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
