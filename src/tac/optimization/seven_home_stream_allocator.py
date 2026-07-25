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
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.score_geometry import CONTEST_REFERENCE_BYTES, contest_score

ALLOCATION_PLAN_SCHEMA: Final = "tac.seven_home_allocation_plan.v1"
RECEIPT_MANIFEST_SCHEMA: Final = "tac.seven_home_receipt_manifest.v1"
ADAPTER_MANIFEST_SCHEMA: Final = "tac.applied_action_adapter_manifest.v1"
EV2_PARTITION_SCHEMA: Final = "ddm_ev2_coarse_stream_partition.v1"
EV2_HOME_SCHEMA: Final = "ddm_ev2_coarse_stream_home.v1"
REFERENCE_BYTES: Final = CONTEST_REFERENCE_BYTES
EV2_BASE_ARCHIVE_BYTES: Final = 134_211
EV2_PARTITION_KEYS: Final = frozenset({"schema", "partition_level", "counted_bytes", "rows"})
EV2_HOME_KEYS: Final = frozenset(
    {
        "schema",
        "stream",
        "typed_home",
        "source_home",
        "archive_member",
        "byte_range",
        "counted_bytes",
        "same_object",
        "derivation_method",
        "assignment_status",
        "pair_id",
        "cell_key",
        "unallocated_reason",
    }
)
ADAPTER_RESULT_SCHEMA: Final = "tac.applied_action_adapter_result.v1"
CONTEST_AUTHORITY_AXES: Final = frozenset(
    {"contest_cpu", "contest_cuda", "contest-CPU", "contest-CUDA", "[contest-CPU]", "[contest-CUDA]"}
)

EV2_HOME_CONTRACTS: Final = {
    "manifest": {
        "type": "PROGRAM",
        "layer_home": "L1_program",
        "stream_type": StreamType.SKELETON,
        "receipt_layer_home": LayerHome.L1_PROGRAM,
        "archive_member": "manifest.json",
    },
    "v15_predictor_zip_outer_home": {
        "type": "CONTEXT",
        "layer_home": "L2_chart_grammar",
        "stream_type": StreamType.CONNECTION,
        "receipt_layer_home": LayerHome.L2_CHART,
        "archive_member": "predictor.zip",
    },
    "g1_movable_worldsheet_outer_home": {
        "type": "CONTEXT",
        "layer_home": "L2_chart_grammar",
        "stream_type": StreamType.CONNECTION,
        "receipt_layer_home": LayerHome.L2_CHART,
        "archive_member": "predict/movable_polygon_worldsheet.g1s",
    },
    "receiver_realization_profile": {
        "type": "PROGRAM",
        "layer_home": "L1_program",
        "stream_type": StreamType.SKELETON,
        "receipt_layer_home": LayerHome.L1_PROGRAM,
        "archive_member": "render/receiver_realization.ddrp",
    },
    "solved_template_outer_home": {
        "type": "FIBER",
        "layer_home": "L4_scorer_feature",
        "stream_type": StreamType.FIBER,
        "receipt_layer_home": LayerHome.L4_SCORER_FEATURE,
        "archive_member": "render/scorer_solved_templates.ddst",
    },
    "central_directory_and_eocd": {
        "type": "PROGRAM",
        "layer_home": "L1_program",
        "stream_type": StreamType.SKELETON,
        "receipt_layer_home": LayerHome.L1_PROGRAM,
        "archive_member": "__central_directory_and_eocd__",
    },
    "lane_program_seed": {
        "type": "CONTEXT",
        "layer_home": "L2_chart_grammar",
        "stream_type": StreamType.CONNECTION,
        "receipt_layer_home": LayerHome.L2_CHART,
        "archive_member": None,
    },
}

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
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SevenHomeAllocationError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise SevenHomeAllocationError(f"{name} must be finite")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SevenHomeAllocationError(f"{name} must be a non-negative exact integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SevenHomeAllocationError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise SevenHomeAllocationError(f"{name} must be lowercase SHA-256 hex")
    return text


def _contest_compatible_axis(axis: Any) -> bool:
    return isinstance(axis, str) and axis in CONTEST_AUTHORITY_AXES


def _target_is_contest_compatible(effective: Mapping[str, Any]) -> bool:
    if _contest_compatible_axis(effective.get("axis")):
        return True
    return (
        effective.get("axis") == "official_leaderboard"
        and effective.get("source_kind") == "external_public_leaderboard_target"
    )


def load_dynamic_target(pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Read the competitive target from canonical pointer metadata."""

    effective = pointer.get("effective_frontier")
    if not isinstance(effective, Mapping):
        raise SevenHomeAllocationError("pointer lacks effective_frontier metadata")
    score = _finite(effective.get("score"), "effective_frontier.score")
    if score < 0.0:
        raise SevenHomeAllocationError("effective frontier score must be non-negative")
    axis = _text(effective.get("axis"), "effective_frontier.axis")
    custody = _text(effective.get("custody"), "effective_frontier.custody")
    evidence_grade = _text(
        effective.get("evidence_grade"), "effective_frontier.evidence_grade"
    )
    source = _text(effective.get("source"), "effective_frontier.source")
    source_kind = _text(effective.get("source_kind"), "effective_frontier.source_kind")
    return {
        "score": score,
        "axis": axis,
        "custody": custody,
        "evidence_grade": evidence_grade,
        "source": source,
        "source_kind": source_kind,
        "contest_compatible": _target_is_contest_compatible(effective),
    }


def derive_seven_homes(ev2: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Re-derive the seven owners from EV2 exact construction lineage."""

    partition = ev2.get("coarse_lawful_partition")
    if (
        not isinstance(partition, Mapping)
        or set(partition) != EV2_PARTITION_KEYS
        or partition.get("schema") != EV2_PARTITION_SCHEMA
        or partition.get("partition_level") != "LP1_TYPED_STREAM_HOME"
    ):
        raise SevenHomeAllocationError("EV2 coarse lawful partition schema differs")
    rows = partition.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEVEN_HOME_IDS):
        raise SevenHomeAllocationError("EV2 must contain exactly seven coarse stream homes")

    by_id: dict[str, dict[str, Any]] = {}
    ranged: list[tuple[int, int, str]] = []
    for raw in rows:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != EV2_HOME_KEYS
            or raw.get("schema") != EV2_HOME_SCHEMA
        ):
            raise SevenHomeAllocationError("EV2 stream-home row schema differs")
        home_id = _text(raw.get("stream"), "EV2 stream-home identity")
        if home_id in by_id:
            raise SevenHomeAllocationError("EV2 stream-home identities must be unique")
        counted = raw.get("counted_bytes")
        if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
            raise SevenHomeAllocationError(f"EV2 home {home_id} has invalid counted bytes")
        if raw.get("derivation_method") != "EXACT_CONSTRUCTION_LINEAGE":
            raise SevenHomeAllocationError(f"EV2 home {home_id} lacks construction lineage")
        if raw.get("same_object") is not True:
            raise SevenHomeAllocationError(f"EV2 home {home_id} is not on the C1 object")
        contract = EV2_HOME_CONTRACTS.get(home_id)
        if contract is None:
            raise SevenHomeAllocationError(f"EV2 home {home_id} has no sealed typed-home contract")
        typed_home = raw.get("typed_home")
        if not isinstance(typed_home, Mapping) or set(typed_home) != {"type", "layer_home"}:
            raise SevenHomeAllocationError(f"EV2 home {home_id} typed_home structure differs")
        if typed_home.get("type") != contract["type"]:
            raise SevenHomeAllocationError(f"EV2 home {home_id} typed stream type differs")
        if typed_home.get("layer_home") != contract["layer_home"]:
            raise SevenHomeAllocationError(f"EV2 home {home_id} typed layer home differs")
        if raw.get("archive_member") != contract["archive_member"]:
            raise SevenHomeAllocationError(f"EV2 home {home_id} archive member differs")
        if raw.get("assignment_status") != (
            "UNALLOCATED_NO_EXCLUSIVE_FINAL_BYTE_PAIR_AND_CELL_FOREIGN_KEY"
        ):
            raise SevenHomeAllocationError(f"EV2 home {home_id} allocation structure differs")
        if raw.get("pair_id") is not None or raw.get("cell_key") is not None:
            raise SevenHomeAllocationError(f"EV2 home {home_id} invents pair/cell ownership")
        source_home = _text(raw.get("source_home"), f"EV2 home {home_id} source_home")
        _text(raw.get("unallocated_reason"), f"EV2 home {home_id} unallocated_reason")
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
            "typed_home": dict(typed_home),
            "receipt_stream_type": contract["stream_type"].value,
            "receipt_layer_home": contract["receipt_layer_home"].value,
            "source_home": source_home,
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
    if total != EV2_BASE_ARCHIVE_BYTES:
        raise SevenHomeAllocationError("EV2 seven-home base partition must equal 134211 bytes")
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
        if self.alternative_group_id is not None and (
            not isinstance(self.alternative_group_id, str)
            or not self.alternative_group_id.strip()
        ):
            raise SevenHomeAllocationError("alternative_group_id must be non-empty when present")
        if not isinstance(self.component_receipt_ids, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.component_receipt_ids
        ):
            raise SevenHomeAllocationError(
                "component receipt identities must be a tuple of non-empty strings"
            )
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
        if any(not isinstance(item, str) or not item.strip() for item in components):
            raise SevenHomeAllocationError("component_receipt_ids must contain non-empty strings")
        opportunity_pool_id = payload.get(
            "opportunity_pool_id", "UNSPECIFIED_NONADDITIVE_POOL"
        )
        if not isinstance(opportunity_pool_id, str) or not opportunity_pool_id.strip():
            raise SevenHomeAllocationError("opportunity_pool_id must be a non-empty string")
        alternative_group_id = payload.get("alternative_group_id")
        if alternative_group_id is not None and (
            not isinstance(alternative_group_id, str) or not alternative_group_id.strip()
        ):
            raise SevenHomeAllocationError("alternative_group_id must be a non-empty string")
        return cls(
            receipt=receipt,
            opportunity_pool_id=opportunity_pool_id.strip(),
            alternative_group_id=(
                alternative_group_id.strip() if alternative_group_id is not None else None
            ),
            component_receipt_ids=tuple(item.strip() for item in components),
        )


def _validate_false_authority(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("research_only") is not True:
        raise SevenHomeAllocationError(f"{name} research_only must be true")
    if payload.get("promotion_eligible") is not False:
        raise SevenHomeAllocationError(f"{name} promotion_eligible must be false")
    if payload.get("score_claim") is not False:
        raise SevenHomeAllocationError(f"{name} score_claim must be false")


def _validate_manifest_content_sha256(manifest: Mapping[str, Any], name: str) -> str:
    declared = _sha256(manifest.get("content_sha256"), f"{name}.content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("content_sha256", None)
    if _sha256_json(unhashed) != declared:
        raise SevenHomeAllocationError(f"{name} content_sha256 does not reconcile")
    return declared


def _validate_adapter_manifest(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _validate_false_authority(manifest, "adapter manifest")
    _validate_manifest_content_sha256(manifest, "adapter manifest")
    results = manifest.get("results")
    if not isinstance(results, list):
        raise SevenHomeAllocationError("adapter manifest results must be an array")
    receipt_count = _nonnegative_int(
        manifest.get("receipt_count"), "adapter manifest receipt_count"
    )
    blocked_count = _nonnegative_int(
        manifest.get("blocked_source_count"), "adapter manifest blocked_source_count"
    )
    observed_receipts = 0
    observed_blocked = 0
    normalized: list[Mapping[str, Any]] = []
    for row in results:
        if not isinstance(row, Mapping):
            raise SevenHomeAllocationError("adapter result must be an object")
        if row.get("schema") != ADAPTER_RESULT_SCHEMA:
            raise SevenHomeAllocationError("adapter result schema differs")
        _validate_false_authority(row, "adapter result")
        _text(row.get("source_kind"), "adapter result source_kind")
        _text(row.get("source_schema"), "adapter result source_schema")
        _text(row.get("source_id"), "adapter result source_id")
        ok = row.get("ok")
        if not isinstance(ok, bool):
            raise SevenHomeAllocationError("adapter result ok must be boolean")
        receipt = row.get("receipt")
        blockers = row.get("blockers")
        if not isinstance(blockers, list):
            raise SevenHomeAllocationError("adapter result blockers must be an array")
        if ok:
            observed_receipts += 1
            if not isinstance(receipt, Mapping) or blockers:
                raise SevenHomeAllocationError(
                    "successful adapter result must carry one receipt and no blockers"
                )
        else:
            observed_blocked += 1
            if receipt is not None or not blockers:
                raise SevenHomeAllocationError(
                    "blocked adapter result must carry blockers and no receipt"
                )
            for blocker in blockers:
                if not isinstance(blocker, Mapping):
                    raise SevenHomeAllocationError("adapter blocker must be an object")
                for field in ("code", "source_key", "owed_field", "detail"):
                    _text(blocker.get(field), f"adapter blocker {field}")
        normalized.append(row)
    if receipt_count != observed_receipts or blocked_count != observed_blocked:
        raise SevenHomeAllocationError("adapter manifest result counts do not reconcile")
    if receipt_count + blocked_count != len(results):
        raise SevenHomeAllocationError("adapter manifest total count does not reconcile")
    return normalized


def _validate_receipt_manifest(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _validate_false_authority(manifest, "receipt manifest")
    _validate_manifest_content_sha256(manifest, "receipt manifest")
    rows = manifest.get("receipts")
    if not isinstance(rows, list):
        raise SevenHomeAllocationError("receipt manifest receipts must be an array")
    if any(not isinstance(row, Mapping) for row in rows):
        raise SevenHomeAllocationError("receipt manifest row must be an object")
    if _nonnegative_int(
        manifest.get("receipt_count"), "receipt manifest receipt_count"
    ) != len(rows):
        raise SevenHomeAllocationError("receipt manifest count does not reconcile")
    return rows


def envelopes_from_manifest(manifest: Mapping[str, Any]) -> tuple[ReceiptEnvelope, ...]:
    """Parse either one receipt or a strict allocator receipt manifest."""

    if manifest.get("schema") == RECEIPT_MANIFEST_SCHEMA:
        rows = _validate_receipt_manifest(manifest)
        return tuple(ReceiptEnvelope.from_mapping(row) for row in rows)
    if manifest.get("schema") == ADAPTER_MANIFEST_SCHEMA:
        results = _validate_adapter_manifest(manifest)
        envelopes: list[ReceiptEnvelope] = []
        for row in results:
            receipt = row.get("receipt")
            if receipt is None:
                continue
            source_kind = _text(row.get("source_kind"), "adapter result source_kind")
            envelopes.append(
                ReceiptEnvelope.from_mapping(
                    {
                        "receipt": receipt,
                        "opportunity_pool_id": source_kind,
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
    blockers = [f"SOURCE_RECEIPT_BLOCKER:{value}" for value in receipt.blockers]
    blockers.extend(
        f"SOURCE_ACTION_EFFECT_BLOCKER:{value}" for value in receipt.action_effect.blockers
    )
    home_id = receipt.stream_home.byte_home_id
    if home_id not in homes_by_id:
        blockers.append("STREAM_HOME_NOT_IN_EV2_SEVEN_HOME_PARTITION")
    else:
        home = homes_by_id[home_id]
        if receipt.stream_home.bytes_before != home["counted_bytes"]:
            blockers.append("STREAM_HOME_BASE_BYTES_DIFFER_FROM_EV2_OWNER")
        if receipt.stream_home.stream_type.value != home["receipt_stream_type"]:
            blockers.append("STREAM_HOME_TYPE_DIFFERS_FROM_EV2_TYPED_HOME")
        if receipt.stream_home.layer_home.value != home["receipt_layer_home"]:
            blockers.append("STREAM_HOME_LAYER_DIFFERS_FROM_EV2_TYPED_HOME")
    if not receipt.pair_ids:
        blockers.append("PAIR_IDENTITY_ABSENT")
    if receipt.support_sha256 is None:
        blockers.append("SUPPORT_IDENTITY_ABSENT")
    if receipt.action_effect.old_d_seg is None or receipt.action_effect.old_d_pose is None:
        blockers.append("BASE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.old_bytes is None:
        blockers.append("BASE_ARCHIVE_BYTES_ABSENT")
    elif receipt.action_effect.old_bytes != EV2_BASE_ARCHIVE_BYTES:
        blockers.append("BASE_ARCHIVE_BYTES_DIFFER_FROM_EV2_PARTITION")
    if receipt.action_effect.new_d_seg is None or receipt.action_effect.new_d_pose is None:
        blockers.append("CANDIDATE_DISTORTION_ENDPOINT_ABSENT")
    if receipt.action_effect.new_bytes is None:
        blockers.append("CANDIDATE_ARCHIVE_BYTES_ABSENT")
    if receipt.status not in {ApplicationStatus.DOWNHILL_FINITE, ApplicationStatus.UPHILL_NULL}:
        blockers.append(f"STATUS_NOT_FINITE:{receipt.status.value}")
    return blockers


def _aggregate_identity_blockers(
    receipt: AppliedActionReceipt,
    homes_by_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers = [f"SOURCE_RECEIPT_BLOCKER:{value}" for value in receipt.blockers]
    blockers.extend(
        f"SOURCE_ACTION_EFFECT_BLOCKER:{value}" for value in receipt.action_effect.blockers
    )
    if receipt.action_effect.old_bytes is None:
        blockers.append("COMPOSED_BASE_ARCHIVE_BYTES_ABSENT")
    elif receipt.action_effect.old_bytes != EV2_BASE_ARCHIVE_BYTES:
        blockers.append("COMPOSED_BASE_ARCHIVE_BYTES_DIFFER_FROM_EV2_PARTITION")
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
    blockers = _aggregate_identity_blockers(receipt, homes_by_id)
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
    authorities = {item.authority_axis for item in component_receipts}
    authorities.add(receipt.authority_axis)
    if len(bases) != 1:
        blockers.append("CROSS_BASE_COMPOSITION_REFUSED")
    if len(from_edges) != 1:
        blockers.append("CROSS_EDGE_FROM_STATE_COMPOSITION_REFUSED")
    if len(authorities) != 1:
        blockers.append("CROSS_AUTHORITY_AXIS_COMPOSITION_REFUSED")
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
    if sum(item.stream_home.delta_bytes for item in component_receipts) != receipt.action_effect.delta_bytes:
        blockers.append("COMPOSED_STREAM_HOME_DELTAS_DO_NOT_RECONCILE")
    pair_union = sorted({pair for item in component_receipts for pair in item.pair_ids})
    if tuple(pair_union) != tuple(sorted(receipt.pair_ids)):
        blockers.append("COMPOSED_PAIR_IDENTITY_UNION_MISMATCH")
    if receipt.status is not ApplicationStatus.DOWNHILL_FINITE:
        blockers.append("COMPOSED_TRANSITION_IS_NOT_DOWNHILL")
    return blockers, tuple(components)


def _dependency_cycle_nodes(
    by_receipt_id: Mapping[str, ReceiptEnvelope],
) -> frozenset[str]:
    states: dict[str, int] = {}
    stack: list[str] = []
    cycle_nodes: set[str] = set()

    def visit(receipt_id: str) -> None:
        state = states.get(receipt_id, 0)
        if state == 2:
            return
        if state == 1:
            cycle_nodes.update(stack[stack.index(receipt_id) :])
            return
        states[receipt_id] = 1
        stack.append(receipt_id)
        for dependency_id in by_receipt_id[receipt_id].component_receipt_ids:
            if dependency_id in by_receipt_id:
                visit(dependency_id)
        stack.pop()
        states[receipt_id] = 2

    for receipt_id in sorted(by_receipt_id):
        visit(receipt_id)
    return frozenset(cycle_nodes)


def _resolve_admissibility(
    envelope: ReceiptEnvelope,
    *,
    by_receipt_id: Mapping[str, ReceiptEnvelope],
    homes_by_id: Mapping[str, Mapping[str, Any]],
    cycle_nodes: frozenset[str],
    memo: dict[str, tuple[tuple[str, ...], tuple[ReceiptEnvelope, ...]]],
) -> tuple[tuple[str, ...], tuple[ReceiptEnvelope, ...]]:
    receipt_id = envelope.receipt.receipt_id
    if receipt_id in memo:
        return memo[receipt_id]
    if receipt_id in cycle_nodes:
        result = (("DEPENDENCY_CYCLE_DETECTED",), ())
        memo[receipt_id] = result
        return result

    receipt = envelope.receipt
    if not envelope.component_receipt_ids:
        blockers = _identity_blockers(receipt, homes_by_id)
        if receipt.action_effect.composed_action_ids:
            blockers.append("COMPOSED_ACTION_REQUIRES_EXPLICIT_COMPONENT_RECEIPT_IDS")
        result = (tuple(sorted(set(blockers))), (envelope,))
        memo[receipt_id] = result
        return result

    blockers, direct_components = _component_composition_blockers(
        envelope, by_receipt_id, homes_by_id
    )
    leaves: list[ReceiptEnvelope] = []
    for component in direct_components:
        child_blockers, child_leaves = _resolve_admissibility(
            component,
            by_receipt_id=by_receipt_id,
            homes_by_id=homes_by_id,
            cycle_nodes=cycle_nodes,
            memo=memo,
        )
        if child_blockers:
            blockers.append(
                f"DEPENDENCY_NOT_INDEPENDENTLY_ADMISSIBLE:{component.receipt.receipt_id}"
            )
        leaves.extend(child_leaves)

    leaf_ids = [item.receipt.receipt_id for item in leaves]
    if len(leaf_ids) != len(set(leaf_ids)):
        blockers.append("NESTED_DEPENDENCY_CLOSURE_REUSES_RECEIPT")
    leaf_action_ids = [item.receipt.action_id for item in leaves]
    if len(leaf_action_ids) != len(set(leaf_action_ids)):
        blockers.append("NESTED_DEPENDENCY_CLOSURE_REUSES_ACTION")
    leaf_homes = [item.receipt.stream_home.byte_home_id for item in leaves]
    if any(home_id not in homes_by_id for home_id in leaf_homes):
        blockers.append("NESTED_DEPENDENCY_HOME_OUTSIDE_EV2_PARTITION")
    if len(leaf_homes) != len(set(leaf_homes)):
        blockers.append("NESTED_DEPENDENCY_CLOSURE_DUPLICATES_HOME")
    leaf_edges = [item.receipt.physical_edge_id for item in leaves]
    if len(leaf_edges) != len(set(leaf_edges)):
        blockers.append("NESTED_DEPENDENCY_CLOSURE_DUPLICATES_PHYSICAL_EDGE")
    if sum(item.receipt.stream_home.delta_bytes for item in leaves) != (
        receipt.action_effect.delta_bytes
    ):
        blockers.append("NESTED_DEPENDENCY_STREAM_HOME_DELTAS_DO_NOT_RECONCILE")
    pair_union = sorted({pair for item in leaves for pair in item.receipt.pair_ids})
    if tuple(pair_union) != tuple(sorted(receipt.pair_ids)):
        blockers.append("NESTED_DEPENDENCY_PAIR_IDENTITY_UNION_MISMATCH")
    authorities = {receipt.authority_axis}
    authorities.update(item.receipt.authority_axis for item in leaves)
    if len(authorities) != 1:
        blockers.append("NESTED_DEPENDENCY_AUTHORITY_AXIS_DIFFERS")

    result = (tuple(sorted(set(blockers))), tuple(leaves))
    memo[receipt_id] = result
    return result


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
    cycle_nodes = _dependency_cycle_nodes(by_receipt_id)
    admissibility_memo: dict[
        str, tuple[tuple[str, ...], tuple[ReceiptEnvelope, ...]]
    ] = {}

    rejected: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, ReceiptEnvelope, tuple[ReceiptEnvelope, ...]]] = []
    bases: set[tuple[str, str]] = set()
    base_score_endpoints: set[tuple[float, float, int]] = set()
    authority_axes: set[str] = set()
    for envelope in ordered:
        receipt = envelope.receipt
        blockers, components = _resolve_admissibility(
            envelope,
            by_receipt_id=by_receipt_id,
            homes_by_id=homes_by_id,
            cycle_nodes=cycle_nodes,
            memo=admissibility_memo,
        )
        if blockers:
            rejected.append(
                {
                    "receipt_id": receipt.receipt_id,
                    "action_id": receipt.action_id,
                    "blockers": list(blockers),
                }
            )
            continue
        bases.add((receipt.base_archive_sha256, receipt.edge_from_state_id))
        authority_axes.add(receipt.authority_axis)
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
    if len(authority_axes) > 1:
        global_blockers.append("MIXED_AUTHORITY_AXES_REFUSED")
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
            "leaf_component_receipt_ids": [
                component.receipt.receipt_id for component in components
            ],
            "opportunity_pool_id": winner.opportunity_pool_id,
            "alternative_group_id": winner.alternative_group_id,
            "physical_edge_id": receipt.physical_edge_id,
        }
        transition = {
            "before": before,
            "after": after,
            "delta_score_total": after["score"] - before["score"],
            "beats_dynamic_target": (
                after["score"] < target["score"]
                if _contest_compatible_axis(receipt.authority_axis)
                and target["contest_compatible"]
                else None
            ),
            "dynamic_target_comparison_eligible": (
                _contest_compatible_axis(receipt.authority_axis)
                and target["contest_compatible"]
            ),
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
            "mixed_authority_axis_comparison_used": False,
            "dynamic_target_requires_contest_compatible_axis": True,
            "dependency_dag_cycle_refusal": True,
            "nested_dependency_leaf_closure_required": True,
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
