# SPDX-License-Identifier: MIT
"""Evidence-derived LA-POSE motion record construction.

This module bridges exact/component CUDA artifacts into LA-POSE-style motion
atom planning inputs. It is deliberately planning-only: component-response
curves and pair opportunities can rank where to spend charged bytes, but they
cannot dispatch or claim a score without a concrete archive builder and exact
CUDA auth eval.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from tac.analysis.lapose_motion_atoms import LaposeMotionAtomError
from tac.analysis.lapose_paper_contract import LAPOSE_PAPER_REFERENCE

SCHEMA_VERSION = 1


def records_from_component_response(
    component_response: Mapping[str, Any],
    *,
    latent_actions: Sequence[Mapping[str, Any]],
    pair_opportunities: Sequence[Mapping[str, Any]],
    evidence_source_path: str,
    evidence_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Create LA-POSE motion records from CUDA response and pair opportunities."""

    if not evidence_source_path:
        raise LaposeMotionAtomError("evidence_source_path is required")
    source_sha = evidence_source_sha256 or _sha256_json(component_response)
    baseline_archive = _mapping(component_response.get("baseline_archive"), "baseline_archive")
    baseline_sha = str(baseline_archive.get("sha256") or "")
    if not baseline_sha:
        raise LaposeMotionAtomError("component response missing baseline_archive.sha256")
    device = str(component_response.get("device") or "")
    if "cuda" not in device.lower():
        raise LaposeMotionAtomError("component response must be CUDA evidence")
    if component_response.get("score_claim") is True:
        raise LaposeMotionAtomError("component response must not be a score claim")

    response_delta = _best_response_delta(component_response)
    latent_by_pair = {_int_field(item, "pair_index"): item for item in latent_actions}
    opportunities = [_normalize_opportunity(item) for item in pair_opportunities]
    if not opportunities:
        raise LaposeMotionAtomError("at least one pair opportunity is required")
    missing = [item["pair_index"] for item in opportunities if item["pair_index"] not in latent_by_pair]
    if missing:
        raise LaposeMotionAtomError(f"missing latent_action for pair_index {missing[0]}")

    mass = sum(max(float(item["opportunity_mass"]), 0.0) for item in opportunities)
    if mass <= 0:
        raise LaposeMotionAtomError("pair opportunity mass must be positive")
    total_byte_delta = int(response_delta["byte_delta"])
    shares = [max(float(item["opportunity_mass"]), 0.0) / mass for item in opportunities]
    byte_deltas = _allocate_signed_integer_total(total_byte_delta, shares)
    records: list[dict[str, Any]] = []
    for item, share, byte_delta in zip(opportunities, shares, byte_deltas, strict=True):
        pair_index = int(item["pair_index"])
        latent = latent_by_pair[pair_index]
        record = {
            "pair_index": pair_index,
            "hard_pair_rank": item.get("hard_pair_rank"),
            "latent_action": _float_sequence(latent.get("latent_action"), "latent_action"),
            "byte_delta": byte_delta,
            "expected_seg_dist_delta": float(response_delta["seg_dist_delta"]) * share,
            "expected_pose_dist_delta": float(response_delta["pose_dist_delta"]) * share,
            "confidence": float(item["confidence"]) * float(response_delta["confidence"]),
            "hard_pair_score": float(item["hard_pair_score"]),
            "class_support": list(item["class_support"]),
            "pair_support": [pair_index],
            "hard_pair_support": list(item["hard_pair_support"]),
            "geometry_priors": list(item["geometry_priors"]),
            "openpilot_priors": list(item["openpilot_priors"]),
            "evidence_grade": "diagnostic_cuda_global_response_allocated",
            "allocation_inference": True,
            "evidence_source_path": evidence_source_path,
            "evidence_source_sha256": source_sha,
            "source_archive_sha256": baseline_sha,
        }
        records.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tac.analysis.lapose_motion_evidence.records_from_component_response",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_source_path": evidence_source_path,
        "evidence_source_sha256": source_sha,
        "source_archive_sha256": baseline_sha,
        "device": device,
        "paper_reference": LAPOSE_PAPER_REFERENCE,
        "allocation": {
            "method": "opportunity_mass_weighted_global_component_response",
            "allocation_inference": True,
            "measurement_scope": "global_component_response_allocated_to_pairs_by_opportunity_mass",
            "response_atom": response_delta,
            "pair_opportunity_count": len(opportunities),
        },
        "records": records,
        "dispatch_blockers": [
            "planning_only_allocated_component_response",
            "lapose_lite_is_not_paper_faithful_lapose_model",
            "requires_lapose_or_motion_builder_charging_bytes",
            "requires_noop_controls",
            "requires_exact_cuda_auth_eval",
        ],
    }


def _best_response_delta(component_response: Mapping[str, Any]) -> dict[str, Any]:
    points = component_response.get("points")
    if not isinstance(points, list) or not points:
        raise LaposeMotionAtomError("component response missing points")
    baseline = None
    for point in points:
        if isinstance(point, Mapping) and float(point.get("epsilon", math.nan)) == 0.0:
            baseline = point
            break
    if baseline is None:
        raise LaposeMotionAtomError("component response missing epsilon=0 baseline point")
    base_values = _mapping(baseline.get("values"), "baseline values")
    base_archive = _mapping(baseline.get("archive"), "baseline archive")
    base_combined = float(base_values.get("combined"))
    base_seg = float(base_values.get("segnet"))
    base_pose = float(base_values.get("posenet"))
    base_bytes = int(base_archive.get("bytes"))

    best_point = None
    best_delta = math.inf
    for point in points:
        if not isinstance(point, Mapping):
            continue
        values = _mapping(point.get("values"), "point values")
        combined_delta = float(values.get("combined")) - base_combined
        if combined_delta < best_delta:
            best_delta = combined_delta
            best_point = point
    if best_point is None:
        raise LaposeMotionAtomError("component response has no valid points")
    # Audit Finding 2 (2026-05-06): raise instead of silently returning
    # baseline as "best". If best_delta >= 0, every non-baseline point is
    # worse than or equal to baseline. Producing zero-delta atoms from this
    # would silently pollute the planning ledger with no-op atoms.
    if best_delta >= 0.0:
        raise LaposeMotionAtomError(
            f"component response has no improving point: best combined_delta="
            f"{best_delta:.6g} >= 0 (baseline already optimal or all "
            f"non-baseline points regress). Refusing to produce zero-delta atoms."
        )
    best_values = _mapping(best_point.get("values"), "best values")
    best_archive = _mapping(best_point.get("archive"), "best archive")
    return {
        "epsilon": float(best_point.get("epsilon")),
        "combined_score_delta": best_delta,
        "seg_dist_delta": float(best_values.get("segnet")) - base_seg,
        "pose_dist_delta": float(best_values.get("posenet")) - base_pose,
        "byte_delta": int(best_archive.get("bytes")) - base_bytes,
        # [prediction]: 0.5/0.8 thresholds are engineering judgments, not
        # empirical calibrations.
        "confidence": 0.5 if component_response.get("promotion_eligible") is False else 0.8,
    }


def _normalize_opportunity(item: Mapping[str, Any]) -> dict[str, Any]:
    pair_index = _int_field(item, "pair_index")
    confidence = float(item.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise LaposeMotionAtomError(f"pair {pair_index}: confidence must be in [0, 1]")
    mass = float(item.get("opportunity_mass", item.get("hard_pair_score", 0.0)))
    if not math.isfinite(mass):
        raise LaposeMotionAtomError(f"pair {pair_index}: opportunity_mass must be finite")
    return {
        "pair_index": pair_index,
        "hard_pair_rank": item.get("hard_pair_rank"),
        "opportunity_mass": mass,
        "hard_pair_score": float(item.get("hard_pair_score", mass)),
        "hard_pair_support": _int_list(item.get("hard_pair_support") or [pair_index]),
        "confidence": confidence,
        "class_support": _int_list(item.get("class_support") or []),
        "geometry_priors": _str_list(item.get("geometry_priors") or []),
        "openpilot_priors": _str_list(item.get("openpilot_priors") or []),
    }


def _allocate_signed_integer_total(total: int, shares: Sequence[float]) -> list[int]:
    """Allocate an integer total across shares while preserving the exact sum."""

    if not shares:
        return []
    sign = -1 if total < 0 else 1
    magnitude = abs(int(total))
    raw = [magnitude * max(float(share), 0.0) for share in shares]
    base = [math.floor(value) for value in raw]
    remainder = magnitude - sum(base)
    order = sorted(
        range(len(shares)),
        key=lambda index: (-(raw[index] - base[index]), index),
    )
    for index in order[:remainder]:
        base[index] += 1
    return [sign * int(value) for value in base]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaposeMotionAtomError(f"{name} must be an object")
    return value


def _int_field(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int):
        raise LaposeMotionAtomError(f"{key} must be an integer")
    return value


def _float_sequence(value: Any, key: str) -> list[float]:
    if not isinstance(value, list | tuple):
        raise LaposeMotionAtomError(f"{key} must be a list")
    out = [float(item) for item in value]
    if any(not math.isfinite(item) for item in out):
        raise LaposeMotionAtomError(f"{key} contains non-finite values")
    if not out:
        raise LaposeMotionAtomError(f"{key} must be nonempty")
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


__all__ = ["SCHEMA_VERSION", "records_from_component_response"]
