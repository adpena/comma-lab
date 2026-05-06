"""LA-POSE-style motion atom planning for contest-faithful water filling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from tac.analysis.lapose_paper_contract import LAPOSE_PAPER_REFERENCE
from tac.optimization.meta_lagrangian_allocator import build_atom_ledger

SCHEMA_VERSION = 1


class LaposeMotionAtomError(ValueError):
    """Raised when LA-POSE motion atom planning input is invalid."""


def build_motion_atom_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    base_pose_dist: float,
    source: str,
    target_average_degree: float = 2.0,
    max_atoms: int | None = None,
) -> dict[str, Any]:
    """Build a planning-only sparse motion-atom manifest.

    Each record represents one contest pair/window and may include a
    LA-POSE-style latent action vector, hard-pair support, class support,
    expected SegNet/PoseNet deltas, charged byte estimate, confidence, and
    openpilot/camera geometry priors. The output is a sparse graph plus a
    meta-Lagrangian atom ledger. It never builds archives or claims scores.
    """

    normalized = [_normalize_record(record) for record in records]
    if not normalized:
        raise LaposeMotionAtomError("at least one motion record is required")
    if target_average_degree < 0:
        raise LaposeMotionAtomError("target_average_degree must be non-negative")
    if max_atoms is not None and max_atoms <= 0:
        raise LaposeMotionAtomError("max_atoms must be positive when provided")
    edges = _sparse_edges(normalized, target_average_degree=target_average_degree)
    all_atoms = [_record_to_atom(record) for record in normalized]
    ledger = build_atom_ledger(
        all_atoms,
        base_pose_dist=base_pose_dist,
        source=source,
    )
    atoms = _ranked_atoms_from_ledger(all_atoms, ledger)
    if max_atoms is not None:
        atoms = atoms[:max_atoms]
        ledger = _truncate_ledger(ledger, max_atoms=max_atoms)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.analysis.lapose_motion_atoms.build_motion_atom_manifest",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source": source,
        "paper_reference": LAPOSE_PAPER_REFERENCE,
        "base_pose_dist": float(base_pose_dist),
        "record_count": len(normalized),
        "record_sha256": _sha256_json(normalized),
        "target_average_degree": float(target_average_degree),
        "graph": {
            "node_count": len(normalized),
            "edge_count": len(edges),
            "average_degree": round((2 * len(edges)) / len(normalized), 6),
            "construction": "deterministic_nearest_motion_edges",
            "edges": edges,
        },
        "source_atom_count": len(all_atoms),
        "atoms": atoms,
        "atom_ledger": ledger,
        "dispatch_blockers": [
            "planning_only_lapose_motion_atoms",
            "lapose_lite_is_not_paper_faithful_lapose_model",
            "requires_charged_payload_or_builder_policy",
            "requires_noop_controls",
            "requires_exact_cuda_auth_eval",
        ],
    }


def records_from_json_payload(payload: Any) -> list[Mapping[str, Any]]:
    """Extract motion records from either a list or ``{"records": [...]}``."""

    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        records = payload.get("records")
        if isinstance(records, list):
            return records
    raise LaposeMotionAtomError("motion record JSON must be a list or contain records")


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    pair_index = _int_field(record, "pair_index")
    latent = tuple(_float_sequence(record.get("latent_action"), "latent_action"))
    if not latent:
        raise LaposeMotionAtomError(f"pair {pair_index}: latent_action must be nonempty")
    byte_delta = int(record.get("byte_delta", record.get("estimated_charged_bytes", 0)))
    if "byte_delta" not in record and byte_delta < 0:
        raise LaposeMotionAtomError(
            f"pair {pair_index}: estimated_charged_bytes must be non-negative"
        )
    confidence = float(record.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise LaposeMotionAtomError(f"pair {pair_index}: confidence must be in [0, 1]")
    expected_seg = float(record.get("expected_seg_dist_delta", 0.0))
    expected_pose = float(record.get("expected_pose_dist_delta", 0.0))
    return {
        "pair_index": pair_index,
        "hard_pair_rank": _optional_int_field(record, "hard_pair_rank"),
        "latent_action": list(latent),
        "latent_norm": math.sqrt(sum(value * value for value in latent)),
        "hard_pair_score": float(record.get("hard_pair_score", 0.0)),
        "byte_delta": byte_delta,
        "expected_seg_dist_delta": expected_seg,
        "expected_pose_dist_delta": expected_pose,
        "confidence": confidence,
        "class_support": _int_list(record.get("class_support") or []),
        "pair_support": _int_list(record.get("pair_support") or [pair_index]),
        "hard_pair_support": _int_list(record.get("hard_pair_support") or []),
        "geometry_priors": _str_list(record.get("geometry_priors") or []),
        "openpilot_priors": _str_list(record.get("openpilot_priors") or []),
        "evidence_grade": str(record.get("evidence_grade") or "prediction"),
        "allocation_inference": bool(record.get("allocation_inference", False)),
        "evidence_source_path": str(record.get("evidence_source_path") or ""),
        "evidence_source_sha256": str(record.get("evidence_source_sha256") or ""),
        "source_archive_sha256": str(record.get("source_archive_sha256") or ""),
    }


def _record_to_atom(record: Mapping[str, Any]) -> dict[str, Any]:
    pair_index = int(record["pair_index"])
    return {
        "atom_id": f"lapose_motion_pair:{pair_index}",
        "family": "lapose_motion_atom",
        "hard_pair_rank": record["hard_pair_rank"],
        "byte_delta": int(record["byte_delta"]),
        "expected_seg_dist_delta": float(record["expected_seg_dist_delta"]),
        "expected_pose_dist_delta": float(record["expected_pose_dist_delta"]),
        "confidence": float(record["confidence"]),
        "evidence_grade": record["evidence_grade"],
        "pair_support": list(record["pair_support"]),
        "hard_pair_support": list(record["hard_pair_support"]),
        "class_support": list(record["class_support"]),
        "geometry_priors": list(record["geometry_priors"]),
        "openpilot_priors": list(record["openpilot_priors"]),
        "hard_pair_score": float(record["hard_pair_score"]),
        "latent_norm": float(record["latent_norm"]),
        "allocation_inference": bool(record["allocation_inference"]),
        "evidence_source_path": record["evidence_source_path"],
        "evidence_source_sha256": record["evidence_source_sha256"],
        "source_archive_sha256": record["source_archive_sha256"],
    }


def _ranked_atoms_from_ledger(
    atoms: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(atom["atom_id"]): dict(atom) for atom in atoms}
    ranked = []
    for row in ledger.get("rows") or []:
        if not isinstance(row, Mapping):
            continue
        atom_id = str(row.get("atom_id") or "")
        atom = by_id.get(atom_id)
        if atom is not None:
            ranked.append(atom)
    return ranked


def _truncate_ledger(ledger: Mapping[str, Any], *, max_atoms: int) -> dict[str, Any]:
    out = dict(ledger)
    rows = list(ledger.get("rows") or [])
    out["source_atom_count"] = len(rows)
    out["rows"] = rows[:max_atoms]
    out["atom_count"] = len(out["rows"])
    out["truncation"] = {
        "method": "rank_full_ledger_then_truncate",
        "max_atoms": max_atoms,
        "dropped_atom_count": max(len(rows) - max_atoms, 0),
    }
    return out


def _sparse_edges(
    records: Sequence[Mapping[str, Any]],
    *,
    target_average_degree: float,
) -> list[dict[str, Any]]:
    if len(records) <= 1 or target_average_degree == 0:
        return []
    target_edges = max(len(records) - 1, round(len(records) * target_average_degree / 2))
    latent = _latent_matrix(records)
    left_indices, right_indices = np.triu_indices(len(records), k=1)
    pair_ids = np.asarray([int(record["pair_index"]) for record in records], dtype=np.int64)
    distances = np.linalg.norm(latent[left_indices] - latent[right_indices], axis=1)
    order = np.lexsort(
        (
            pair_ids[right_indices],
            pair_ids[left_indices],
            distances,
        )
    )
    candidates = [
        (
            float(distances[index]),
            int(pair_ids[left_indices[index]]),
            int(pair_ids[right_indices[index]]),
        )
        for index in order
    ]
    edges = []
    parent = {int(record["pair_index"]): int(record["pair_index"]) for record in records}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    selected: set[tuple[int, int]] = set()
    for dist, left, right in candidates:
        if union(left, right):
            selected.add((left, right))
            edges.append(_edge(left, right, dist, role="connectivity"))
        if len(edges) == len(records) - 1:
            break
    for dist, left, right in candidates:
        if len(edges) >= target_edges:
            break
        key = (left, right)
        if key in selected:
            continue
        selected.add(key)
        edges.append(_edge(left, right, dist, role="nearest_extra"))
    return edges


def _edge(left: int, right: int, distance: float, *, role: str) -> dict[str, Any]:
    return {
        "left_pair_index": left,
        "right_pair_index": right,
        "motion_distance": round(distance, 12),
        "role": role,
    }


def _latent_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = [record["latent_action"] for record in records]
    try:
        matrix = np.asarray(rows, dtype=np.float64)
    except Exception as exc:
        raise LaposeMotionAtomError("latent_action rows must be rectangular numeric vectors") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or matrix.shape[1] == 0:
        raise LaposeMotionAtomError("latent_action rows must be a nonempty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise LaposeMotionAtomError("latent_action contains non-finite values")
    return matrix


def _int_field(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise LaposeMotionAtomError(f"{key} must be an integer")
    return value


def _optional_int_field(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        pair_index = record.get("pair_index", "<unknown>")
        raise LaposeMotionAtomError(f"pair {pair_index}: {key} must be an integer when provided")
    return value


def _float_sequence(value: Any, key: str) -> list[float]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError(f"{key} must be a list")
    out = [float(item) for item in value]
    if any(not math.isfinite(item) for item in out):
        raise LaposeMotionAtomError(f"{key} contains non-finite values")
    return out


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError("expected list")
    out = []
    for item in value:
        if not isinstance(item, int):
            raise LaposeMotionAtomError("expected integer list")
        out.append(item)
    return out


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError("expected list")
    return [str(item) for item in value]


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "LaposeMotionAtomError",
    "build_motion_atom_manifest",
    "records_from_json_payload",
]
