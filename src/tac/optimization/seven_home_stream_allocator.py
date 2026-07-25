# SPDX-License-Identifier: MIT
"""Deterministic allocation over EV2's seven independent counted homes.

The allocator consumes identity-bearing :class:`AppliedActionReceipt` rows.
It never manufactures pair/cell byte ownership from EV2's 162 distortion
cells and never adds independently measured deltas.  A multi-home allocation
is admissible only when a measured composed receipt closes the exact nonlinear
score transition and explicitly links its component action identities.

This is research-only planning apparatus.  It does not run a receiver or
scorer, mutate an archive, or promote a score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final

from tac.analysis.applied_action_receipt import (
    ApplicationStatus,
    AppliedActionReceipt,
    AppliedActionReceiptError,
)
from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score

ALLOCATION_PLAN_SCHEMA: Final = "tac.seven_home_allocation_plan.v1"
RECEIPT_MANIFEST_SCHEMA: Final = "tac.seven_home_receipt_manifest.v1"
ADAPTER_MANIFEST_SCHEMA: Final = "tac.applied_action_adapter_manifest.v1"
EV2_PARTITION_SCHEMA: Final = "ddm_ev2_coarse_stream_partition.v1"
EV2_HOME_SCHEMA: Final = "ddm_ev2_coarse_stream_home.v1"
REFERENCE_BYTES: Final = CONTEST_REFERENCE_BYTES

# These identities are re-derived from EV2's exact construction-lineage rows.
# The order is semantic and stable; input row order has no authority.
SEVEN_HOME_IDS: Final = (
    "manifest",
    "v15_predictor_zip_outer_home",
    "g1_movable_worldsheet_outer_home",
    "receiver_realization_profile",
    "solved_template_outer_home",
    "central_directory_and_eocd",
    "lane_program_seed",
)


class SevenHomeAllocationError(ValueError):
    """Raised when allocator inputs lose identity or exact accounting."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SevenHomeAllocationError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SevenHomeAllocationError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise SevenHomeAllocationError(f"{name} must be finite")
    return result


def load_dynamic_target(pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Read the competitive target from canonical pointer metadata."""

    effective = pointer.get("effective_frontier")
    if not isinstance(effective, Mapping):
        raise SevenHomeAllocationError("pointer lacks effective_frontier metadata")
    score = _finite(effective.get("score"), "effective_frontier.score")
    if score < 0.0:
        raise SevenHomeAllocationError("effective frontier score must be non-negative")
    return {
        "score": score,
        "axis": effective.get("axis"),
        "custody": effective.get("custody"),
        "evidence_grade": effective.get("evidence_grade"),
        "source": effective.get("source"),
        "source_kind": effective.get("source_kind"),
    }


def derive_seven_homes(ev2: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Re-derive the seven owners from EV2 exact construction lineage."""

    partition = ev2.get("coarse_lawful_partition")
    if not isinstance(partition, Mapping) or partition.get("schema") != EV2_PARTITION_SCHEMA:
        raise SevenHomeAllocationError("EV2 coarse lawful partition schema differs")
    rows = partition.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEVEN_HOME_IDS):
        raise SevenHomeAllocationError("EV2 must contain exactly seven coarse stream homes")

    by_id: dict[str, dict[str, Any]] = {}
    ranged: list[tuple[int, int, str]] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("schema") != EV2_HOME_SCHEMA:
            raise SevenHomeAllocationError("EV2 stream-home row schema differs")
        home_id = str(raw.get("stream") or "")
        if not home_id or home_id in by_id:
            raise SevenHomeAllocationError("EV2 stream-home identities must be unique")
        counted = raw.get("counted_bytes")
        if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
            raise SevenHomeAllocationError(f"EV2 home {home_id} has invalid counted bytes")
        if raw.get("derivation_method") != "EXACT_CONSTRUCTION_LINEAGE":
            raise SevenHomeAllocationError(f"EV2 home {home_id} lacks construction lineage")
        if raw.get("same_object") is not True:
            raise SevenHomeAllocationError(f"EV2 home {home_id} is not on the C1 object")
        byte_range = raw.get("byte_range")
        normalized_range: list[int] | None = None
        if byte_range is not None:
            if (
                not isinstance(byte_range, list)
                or len(byte_range) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in byte_range)
            ):
                raise SevenHomeAllocationError(f"EV2 home {home_id} byte range is malformed")
            start, stop = byte_range
            if start < 0 or stop < start or stop - start != counted:
                raise SevenHomeAllocationError(f"EV2 home {home_id} byte range does not reconcile")
            normalized_range = [start, stop]
            ranged.append((start, stop, home_id))
        elif home_id != "lane_program_seed":
            raise SevenHomeAllocationError("only lane_program_seed may be a separate home")
        by_id[home_id] = {
            "home_id": home_id,
            "counted_bytes": counted,
            "byte_range": normalized_range,
            "archive_member": raw.get("archive_member"),
            "typed_home": raw.get("typed_home"),
            "source_home": raw.get("source_home"),
            "derivation_method": raw.get("derivation_method"),
        }

    if set(by_id) != set(SEVEN_HOME_IDS):
        missing = sorted(set(SEVEN_HOME_IDS) - set(by_id))
        extra = sorted(set(by_id) - set(SEVEN_HOME_IDS))
        raise SevenHomeAllocationError(f"EV2 seven-home identities differ: missing={missing}, extra={extra}")
    ordered_ranges = sorted(ranged)
    if not ordered_ranges or ordered_ranges[0][0] != 0:
        raise SevenHomeAllocationError("EV2 ranged homes must begin at byte zero")
    for (_, left_stop, left_id), (right_start, _, right_id) in pairwise(ordered_ranges):
        if left_stop != right_start:
            raise SevenHomeAllocationError(
                f"EV2 ranged homes are not contiguous: {left_id}, {right_id}"
            )
    total = sum(row["counted_bytes"] for row in by_id.values())
    if partition.get("counted_bytes") != total:
        raise SevenHomeAllocationError("EV2 seven-home mass does not conserve")
    ranged_mass = total - by_id["lane_program_seed"]["counted_bytes"]
    if ordered_ranges[-1][1] != ranged_mass:
        raise SevenHomeAllocationError("EV2 ranged and separate-home mass does not reconcile")
    return tuple(by_id[home_id] for home_id in SEVEN_HOME_IDS)


@dataclass(frozen=True)
class ReceiptEnvelope:
    """Allocator metadata around one immutable AppliedActionReceipt."""

    receipt: AppliedActionReceipt
    opportunity_pool_id: str = "UNSPECIFIED_NONADDITIVE_POOL"
    alternative_group_id: str | None = None
    component_receipt_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.opportunity_pool_id, str) or not self.opportunity_pool_id.strip():
            raise SevenHomeAllocationError("opportunity_pool_id must be non-empty")
        if self.alternative_group_id is not None and not self.alternative_group_id.strip():
            raise SevenHomeAllocationError("alternative_group_id must be non-empty when present")
        if len(set(self.component_receipt_ids)) != len(self.component_receipt_ids):
            raise SevenHomeAllocationError("component receipt identities must be unique")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReceiptEnvelope:
        wrapped = payload.get("receipt")
        receipt_payload = wrapped if isinstance(wrapped, Mapping) else payload
        try:
            receipt = AppliedActionReceipt.from_dict(receipt_payload)
        except (AppliedActionReceiptError, TypeError, ValueError) as exc:
            raise SevenHomeAllocationError(f"invalid AppliedActionReceipt: {exc}") from exc
        components = payload.get("component_receipt_ids", ()) if wrapped is not None else ()
        if not isinstance(components, list | tuple):
            raise SevenHomeAllocationError("component_receipt_ids must be an array")
        return cls(
            receipt=receipt,
            opportunity_pool_id=str(payload.get("opportunity_pool_id") or "UNSPECIFIED_NONADDITIVE_POOL"),
            alternative_group_id=(
                str(payload["alternative_group_id"]) if payload.get("alternative_group_id") is not None else None
            ),
            component_receipt_ids=tuple(str(item) for item in components),
        )


def envelopes_from_manifest(manifest: Mapping[str, Any]) -> tuple[ReceiptEnvelope, ...]:
    """Parse either one receipt or a strict allocator receipt manifest."""

    if manifest.get("schema") == RECEIPT_MANIFEST_SCHEMA:
        rows = manifest.get("receipts")
        if not isinstance(rows, list):
            raise SevenHomeAllocationError("receipt manifest receipts must be an array")
        if any(not isinstance(row, Mapping) for row in rows):
            raise SevenHomeAllocationError("receipt manifest row must be an object")
        return tuple(ReceiptEnvelope.from_mapping(row) for row in rows)
    if manifest.get("schema") == ADAPTER_MANIFEST_SCHEMA:
        results = manifest.get("results")
        if not isinstance(results, list):
            raise SevenHomeAllocationError("adapter manifest results must be an array")
        envelopes: list[ReceiptEnvelope] = []
        for row in results:
            if not isinstance(row, Mapping):
                raise SevenHomeAllocationError("adapter result must be an object")
            receipt = row.get("receipt")
            if receipt is None:
                if row.get("ok") is True:
                    raise SevenHomeAllocationError("successful adapter result lacks a receipt")
                continue
            if row.get("ok") is not True:
                raise SevenHomeAllocationError("blocked adapter result must not carry a receipt")
            if not isinstance(receipt, Mapping):
                raise SevenHomeAllocationError("adapter result receipt must be an object")
            envelopes.append(
                ReceiptEnvelope.from_mapping(
                    {
                        "receipt": receipt,
                        "opportunity_pool_id": str(row.get("source_kind") or "UNSPECIFIED_NONADDITIVE_POOL"),
                    }
                )
            )
        return tuple(envelopes)
    if manifest.get("schema") == "tac.applied_action_receipt.v1":
        return (ReceiptEnvelope.from_mapping(manifest),)
    raise SevenHomeAllocationError("input is neither an applied receipt nor a seven-home receipt manifest")


def _identity_blockers(
    receipt: AppliedActionReceipt,
    homes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    home_id = receipt.stream_home.byte_home_id
    if home_id not in homes_by_id:
        blockers.append("STREAM_HOME_NOT_IN_EV2_SEVEN_HOME_PARTITION")
    elif receipt.stream_home.bytes_before != homes_by_id[home_id]["counted_bytes"]:
        blockers.append("STREAM_HOME_BASE_BYTES_DIFFER_FROM_EV2_OWNER")
    if not receipt.pair_ids:
        blockers.append("PAIR_IDENTITY_ABSENT")
    if receipt.support_sha256 is None:
        blockers.append("SUPPORT_IDENTITY_ABSENT")
    if receipt.action_effect.old_d_seg is None or receipt.action_effect.old_d_pose is None:
        blockers.append("BASE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.old_bytes is None:
        blockers.append("BASE_ARCHIVE_BYTES_ABSENT")
    if receipt.action_effect.new_d_seg is None or receipt.action_effect.new_d_pose is None:
        blockers.append("CANDIDATE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.new_bytes is None:
        blockers.append("CANDIDATE_ARCHIVE_BYTES_ABSENT")
    if receipt.status not in {ApplicationStatus.DOWNHILL_FINITE, ApplicationStatus.UPHILL_NULL}:
        blockers.append(f"STATUS_NOT_FINITE:{receipt.status.value}")
    return blockers


def _score_point(receipt: AppliedActionReceipt, *, candidate: bool) -> dict[str, Any]:
    effect = receipt.action_effect
    d_seg = effect.new_d_seg if candidate else effect.old_d_seg
    d_pose = effect.new_d_pose if candidate else effect.old_d_pose
    archive_bytes = effect.new_bytes if candidate else effect.old_bytes
    if d_seg is None or d_pose is None or archive_bytes is None:
        raise SevenHomeAllocationError("score endpoint is incomplete")
    score = contest_score(d_seg, d_pose, archive_bytes, reference_bytes=REFERENCE_BYTES)
    return {"d_seg": d_seg, "d_pose": d_pose, "archive_bytes": archive_bytes, "score": score}


def _baseline_choice(home: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "home_id": home["home_id"],
        "selection": "BASELINE_OWNER",
        "owner": home["source_home"],
        "coder_id": "EV2_EXISTING_OWNER",
        "bytes_before": home["counted_bytes"],
        "bytes_after": home["counted_bytes"],
        "delta_bytes": 0,
        "receipt_id": None,
        "physical_edge_id": None,
    }


def _component_composition_blockers(
    envelope: ReceiptEnvelope,
    by_receipt_id: Mapping[str, ReceiptEnvelope],
    homes_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], tuple[ReceiptEnvelope, ...]]:
    receipt = envelope.receipt
    blockers: list[str] = []
    components: list[ReceiptEnvelope] = []
    for receipt_id in envelope.component_receipt_ids:
        component = by_receipt_id.get(receipt_id)
        if component is None:
            blockers.append(f"COMPOSED_COMPONENT_ABSENT:{receipt_id}")
        else:
            components.append(component)
    if tuple(receipt.action_effect.composed_action_ids) != tuple(
        item.receipt.action_id for item in components
    ):
        blockers.append("COMPOSED_ACTION_IDENTITY_ORDER_MISMATCH")
    if receipt.action_effect.interaction_or_commutator is None:
        blockers.append("MEASURED_INTERACTION_OR_COMMUTATOR_ABSENT")
    if not receipt.pair_ids:
        blockers.append("COMPOSED_PAIR_IDENTITY_ABSENT")
    if receipt.support_sha256 is None:
        blockers.append("COMPOSED_SUPPORT_IDENTITY_ABSENT")
    if receipt.action_effect.old_d_seg is None or receipt.action_effect.old_d_pose is None:
        blockers.append("COMPOSED_BASE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.old_bytes is None:
        blockers.append("COMPOSED_BASE_ARCHIVE_BYTES_ABSENT")
    if receipt.action_effect.new_d_seg is None or receipt.action_effect.new_d_pose is None:
        blockers.append("COMPOSED_CANDIDATE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.new_bytes is None:
        blockers.append("COMPOSED_CANDIDATE_ARCHIVE_BYTES_ABSENT")
    if not components:
        blockers.append("COMPOSED_RECEIPT_HAS_NO_COMPONENTS")
        return blockers, tuple(components)

    component_receipts = [item.receipt for item in components]
    bases = {item.base_archive_sha256 for item in component_receipts}
    bases.add(receipt.base_archive_sha256)
    from_edges = {item.edge_from_state_id for item in component_receipts}
    from_edges.add(receipt.edge_from_state_id)
    if len(bases) != 1:
        blockers.append("CROSS_BASE_COMPOSITION_REFUSED")
    if len(from_edges) != 1:
        blockers.append("CROSS_EDGE_FROM_STATE_COMPOSITION_REFUSED")
    aggregate_base = (
        receipt.action_effect.old_d_seg,
        receipt.action_effect.old_d_pose,
        receipt.action_effect.old_bytes,
    )
    component_bases = {
        (
            item.action_effect.old_d_seg,
            item.action_effect.old_d_pose,
            item.action_effect.old_bytes,
        )
        for item in component_receipts
    }
    component_bases.add(aggregate_base)
    if len(component_bases) != 1:
        blockers.append("COMPOSED_BASE_SCORE_ENDPOINTS_DIFFER")
    physical_edges = [item.physical_edge_id for item in component_receipts]
    if len(set(physical_edges)) != len(physical_edges):
        blockers.append("DUPLICATE_PHYSICAL_EDGE_IN_COMPOSITION")
    component_homes = [item.stream_home.byte_home_id for item in component_receipts]
    if any(home_id not in homes_by_id for home_id in component_homes):
        blockers.append("COMPOSED_COMPONENT_HOME_OUTSIDE_EV2_PARTITION")
    if len(set(component_homes)) != len(component_homes):
        blockers.append("MULTIPLE_CODERS_OWN_ONE_HOME_IN_COMPOSITION")
    if sum(item.stream_home.delta_bytes for item in component_receipts) != receipt.action_effect.delta_bytes:
        blockers.append("COMPOSED_STREAM_HOME_DELTAS_DO_NOT_RECONCILE")
    pair_union = sorted({pair for item in component_receipts for pair in item.pair_ids})
    if tuple(pair_union) != tuple(sorted(receipt.pair_ids)):
        blockers.append("COMPOSED_PAIR_IDENTITY_UNION_MISMATCH")
    if receipt.status is not ApplicationStatus.DOWNHILL_FINITE:
        blockers.append("COMPOSED_TRANSITION_IS_NOT_DOWNHILL")
    return blockers, tuple(components)


def build_allocation_plan(
    *,
    ev2: Mapping[str, Any],
    pointer: Mapping[str, Any],
    envelopes: Sequence[ReceiptEnvelope],
) -> dict[str, Any]:
    """Build the exact deterministic singleton/composed allocation plan."""

    homes = derive_seven_homes(ev2)
    homes_by_id = {row["home_id"]: row for row in homes}
    target = load_dynamic_target(pointer)
    ordered = sorted(envelopes, key=lambda item: item.receipt.receipt_id)
    if len({item.receipt.receipt_id for item in ordered}) != len(ordered):
        raise SevenHomeAllocationError("receipt_id identities must be unique")
    by_receipt_id = {item.receipt.receipt_id: item for item in ordered}

    rejected: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, ReceiptEnvelope, tuple[ReceiptEnvelope, ...]]] = []
    bases: set[tuple[str, str]] = set()
    base_score_endpoints: set[tuple[float, float, int]] = set()
    for envelope in ordered:
        receipt = envelope.receipt
        if envelope.component_receipt_ids:
            blockers, components = _component_composition_blockers(
                envelope, by_receipt_id, homes_by_id
            )
            # The composed receipt's synthetic aggregate home is not an EV2
            # owner; ownership is carried by its exact components.
            blockers.extend(
                blocker
                for component in components
                for blocker in _identity_blockers(component.receipt, homes_by_id)
            )
        else:
            blockers = _identity_blockers(receipt, homes_by_id)
            components = (envelope,)
            if receipt.action_effect.composed_action_ids:
                blockers.append("COMPOSED_ACTION_REQUIRES_EXPLICIT_COMPONENT_RECEIPT_IDS")
        if blockers:
            rejected.append(
                {"receipt_id": receipt.receipt_id, "action_id": receipt.action_id, "blockers": sorted(set(blockers))}
            )
            continue
        bases.add((receipt.base_archive_sha256, receipt.edge_from_state_id))
        before = _score_point(receipt, candidate=False)
        base_score_endpoints.add(
            (before["d_seg"], before["d_pose"], before["archive_bytes"])
        )
        if receipt.status is ApplicationStatus.DOWNHILL_FINITE:
            candidates.append((_score_point(receipt, candidate=True)["score"], receipt.receipt_id, envelope, components))
        else:
            rejected.append(
                {
                    "receipt_id": receipt.receipt_id,
                    "action_id": receipt.action_id,
                    "blockers": ["FINITE_BUT_NON_DOWNHILL"],
                }
            )

    global_blockers: list[str] = []
    if len({base for base, _ in bases}) > 1:
        global_blockers.append("CROSS_BASE_CANDIDATE_SET_REFUSED")
        candidates = []
    if len(base_score_endpoints) > 1:
        global_blockers.append("BASE_SCORE_ENDPOINTS_DIFFER_FOR_SHARED_PRICE_SET")
        candidates = []
    # Distinct physical edges can be compared as singleton alternatives, but
    # they are never added.  This note makes the non-additivity explicit.
    if len({edge for _, edge in bases}) > 1:
        global_blockers.append("CROSS_EDGE_PRICES_NOT_COMPOSED;SINGLETON_COMPARISON_ONLY")

    selected_choices = {row["home_id"]: _baseline_choice(row) for row in homes}
    selected_identity: dict[str, Any] | None = None
    transition: dict[str, Any] | None = None
    if candidates:
        _, _, winner, components = min(candidates, key=lambda row: (row[0], row[1]))
        receipt = winner.receipt
        before = _score_point(receipt, candidate=False)
        after = _score_point(receipt, candidate=True)
        for component in components:
            item = component.receipt
            home_id = item.stream_home.byte_home_id
            selected_choices[home_id] = {
                "home_id": home_id,
                "selection": "APPLIED_ACTION_RECEIPT",
                "owner": item.stream_home.coder_owner,
                "coder_id": item.stream_home.coder_id,
                "bytes_before": item.stream_home.bytes_before,
                "bytes_after": item.stream_home.bytes_after,
                "delta_bytes": item.stream_home.delta_bytes,
                "receipt_id": item.receipt_id,
                "physical_edge_id": item.physical_edge_id,
                "pair_ids": list(item.pair_ids),
                "support_sha256": item.support_sha256,
            }
        selected_identity = {
            "receipt_id": receipt.receipt_id,
            "action_id": receipt.action_id,
            "component_receipt_ids": list(winner.component_receipt_ids),
            "opportunity_pool_id": winner.opportunity_pool_id,
            "alternative_group_id": winner.alternative_group_id,
            "physical_edge_id": receipt.physical_edge_id,
        }
        transition = {
            "before": before,
            "after": after,
            "delta_score_total": after["score"] - before["score"],
            "beats_dynamic_target": after["score"] < target["score"],
            "authority_axis": receipt.authority_axis,
        }
        selected_ids = {component.receipt.receipt_id for component in components}
        selected_ids.add(receipt.receipt_id)
        for _, _, alternative, _ in candidates:
            if alternative.receipt.receipt_id not in selected_ids:
                rejected.append(
                    {
                        "receipt_id": alternative.receipt.receipt_id,
                        "action_id": alternative.receipt.action_id,
                        "blockers": ["LOWER_EXACT_JOINT_SCORE_ALTERNATIVE_SELECTED"],
                    }
                )

    unpriced_homes = [
        home_id for home_id in SEVEN_HOME_IDS if selected_choices[home_id]["selection"] == "BASELINE_OWNER"
    ]
    pools: dict[str, list[str]] = {}
    for envelope in ordered:
        pools.setdefault(envelope.opportunity_pool_id, []).append(envelope.receipt.receipt_id)
    pool_rows = [
        {
            "opportunity_pool_id": pool_id,
            "receipt_ids": sorted(receipt_ids),
            "composition_law": "COMPETE_NEVER_ADD_WITHOUT_MEASURED_COMPOSED_RECEIPT",
        }
        for pool_id, receipt_ids in sorted(pools.items())
    ]
    status = "ALLOCATED_MEASURED_TRANSITION" if selected_identity is not None else "BLOCKED_NO_VALID_APPLIED_TRANSITION"
    result: dict[str, Any] = {
        "schema": ALLOCATION_PLAN_SCHEMA,
        "status": status,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "dynamic_target": target,
        "seven_home_derivation": {
            "authority": "EV2_EXACT_CONSTRUCTION_LINEAGE",
            "home_count": len(homes),
            "counted_bytes": sum(row["counted_bytes"] for row in homes),
            "rows": list(homes),
        },
        "selected_identity": selected_identity,
        "selected_home_owners": [selected_choices[home_id] for home_id in SEVEN_HOME_IDS],
        "rejected": sorted(rejected, key=lambda row: row["receipt_id"]),
        "opportunity_pools": pool_rows,
        "interaction_or_commutator_blockers": sorted(global_blockers),
        "unpriced_homes": unpriced_homes,
        "exact_score_transition": transition,
        "policy": {
            "exactly_one_owner_per_home": True,
            "coder_alternatives_same_home_mutually_exclusive": True,
            "pf3_coordinate_and_wf7_stream_pools_additive": False,
            "independent_component_thresholds_used": False,
            "linearized_pose_used": False,
            "cross_base_price_transfer_used": False,
            "cross_edge_price_transfer_used": False,
        },
    }
    result["plan_content_sha256"] = _sha256_json(result)
    return result


__all__ = [
    "ADAPTER_MANIFEST_SCHEMA",
    "ALLOCATION_PLAN_SCHEMA",
    "RECEIPT_MANIFEST_SCHEMA",
    "SEVEN_HOME_IDS",
    "ReceiptEnvelope",
    "SevenHomeAllocationError",
    "build_allocation_plan",
    "derive_seven_homes",
    "envelopes_from_manifest",
    "load_dynamic_target",
]
