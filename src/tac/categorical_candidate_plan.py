# SPDX-License-Identifier: MIT
"""Deterministic planning helpers for categorical mask candidates.

The construction plan produced here is intentionally not dispatch evidence.
It grounds categorical candidate work in the canonical comma10k/contest labels,
charged archive members, and declared conditioning priors while preserving a
hard boundary between planning artifacts and byte-closed archive parity.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from tac.categorical_label_atoms import (
    OPENPILOT_PRIOR_HINTS,
    build_categorical_typed_label_atoms,
    semantic_priority_weight_ppm,
)
from tac.categorical_openpilot_mask_prior_contract import (
    RUNTIME_LABEL_CONTRACT,
    audit_categorical_openpilot_mask_priors,
)
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    SEMANTIC_QUANTIZATION_DEFAULT_BITS,
)

SCHEMA_VERSION = 1
CATEGORICAL_CLASS_CODEBOOK_KIND = "categorical_class_codebook"
CATEGORICAL_CLASS_CODEBOOK_CONTRACT = "categorical_class_codebook_v1"
CATEGORICAL_CONSTRUCTION_PLAN_KIND = "categorical_charged_label_construction_plan"
CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT = "categorical_charged_label_construction_plan_v1"
REQUIRED_PLAN_MEMBER_ROLES = (
    "categorical_payload",
    "decoder_or_runtime_consumer",
)
OPTIONAL_PLAN_MEMBER_ROLES = ("decoder_table",)
PRIMARY_LABEL_SOURCES: tuple[dict[str, str], ...] = (
    {
        "name": "comma10k README",
        "url": "https://github.com/commaai/comma10k/blob/master/README.md",
        "used_for": "class ids, label names, and mask colors",
    },
    {
        "name": "comma.ai crowdsourced Segnet blog",
        "url": "https://blog.comma.ai/crowdsourced-segnet-you-can-help/",
        "used_for": "five-label openpilot semantic grouping",
    },
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _safe_member_name(name: Any) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if "\x00" in name or "\\" in name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    return (
        not path.is_absolute()
        and ".." not in parts
        and all(part not in {"", ".", "__MACOSX"} for part in parts)
        and not any(part.startswith(".") for part in parts)
    )


def _normalize_charged_members(charged_members: Any) -> list[dict[str, Any]]:
    if not isinstance(charged_members, list):
        return []
    normalized: list[dict[str, Any]] = []
    for record in charged_members:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        role = record.get("role")
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        normalized.append(
            {
                "name": name if isinstance(name, str) else "",
                "role": role if isinstance(role, str) else "",
                "bytes": byte_count if isinstance(byte_count, int) else None,
                "sha256": digest if isinstance(digest, str) else "",
            }
        )
    return normalized


def _first_member_with_role(records: list[dict[str, Any]], role: str) -> str:
    names = sorted(
        record["name"]
        for record in records
        if record.get("role") == role and _safe_member_name(record.get("name"))
    )
    return names[0] if names else ""


def _member_role_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    role_to_members: dict[str, list[str]] = {}
    for record in records:
        role = record.get("role")
        name = record.get("name")
        if not isinstance(role, str) or not _safe_member_name(name):
            continue
        role_to_members.setdefault(role, []).append(name)
    for names in role_to_members.values():
        names.sort()
    required_role_status = {
        role: {
            "present": bool(role_to_members.get(role)),
            "members": role_to_members.get(role, []),
        }
        for role in REQUIRED_PLAN_MEMBER_ROLES
    }
    optional_role_status = {
        role: {
            "present": bool(role_to_members.get(role)),
            "members": role_to_members.get(role, []),
        }
        for role in OPTIONAL_PLAN_MEMBER_ROLES
    }
    return {
        "required_roles": required_role_status,
        "optional_roles": optional_role_status,
        "role_to_members": dict(sorted(role_to_members.items())),
    }


def _class_row(
    *,
    class_id: int,
    charged_label_member: str,
    payload_member: str,
) -> dict[str, Any]:
    item = CONTEST_SEGNET_CLASSES[class_id]
    return {
        "class_id": item.class_id,
        "comma10k_id": item.comma10k_id,
        "name": item.name,
        "comma10k_color": item.comma10k_color,
        "selfcomp_gray": SELFCOMP_CLASS_TO_GRAY[item.class_id],
        "default_quant_bits": SEMANTIC_QUANTIZATION_DEFAULT_BITS[item.class_id],
        "semantic_priority_weight_ppm": semantic_priority_weight_ppm(item.class_id),
        "openpilot_prior_hint": OPENPILOT_PRIOR_HINTS[item.name],
        "charged_label_member": charged_label_member,
        "categorical_payload_member": payload_member,
    }


def build_categorical_class_codebook() -> dict[str, Any]:
    """Return the deterministic charged class-codebook payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CATEGORICAL_CLASS_CODEBOOK_KIND,
        "class_codebook_contract": CATEGORICAL_CLASS_CODEBOOK_CONTRACT,
        "class_id_contract": "contest_zero_based_comma10k_order",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "primary_label_sources": list(PRIMARY_LABEL_SOURCES),
        "classes": [
            {
                "class_id": item.class_id,
                "comma10k_id": item.comma10k_id,
                "name": item.name,
                "comma10k_color": item.comma10k_color,
                "selfcomp_gray": SELFCOMP_CLASS_TO_GRAY[item.class_id],
                "default_quant_bits": SEMANTIC_QUANTIZATION_DEFAULT_BITS[item.class_id],
                "openpilot_prior_hint": OPENPILOT_PRIOR_HINTS[item.name],
            }
            for item in CONTEST_SEGNET_CLASSES
        ],
        "typed_label_atoms": build_categorical_typed_label_atoms(),
    }


def build_categorical_charged_label_plan(
    *,
    source_archive_sha256: str,
    charged_members: Any,
    conditioning_priors: Any,
    candidate_archive_sha256: str = "",
    archive_member_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Build a deterministic, non-dispatchable categorical construction plan."""

    records = _normalize_charged_members(charged_members)
    member_names = [record["name"] for record in records if _safe_member_name(record.get("name"))]
    role_summary = _member_role_summary(records)
    label_member = _first_member_with_role(records, "decoder_table")
    payload_member = _first_member_with_role(records, "categorical_payload")
    runtime_member = _first_member_with_role(records, "decoder_or_runtime_consumer")
    conditioning_prior_contract = audit_categorical_openpilot_mask_priors(
        conditioning_priors,
        charged_member_names=member_names,
        charged_members=records,
    )

    plan_blockers: list[str] = []
    if not _is_sha256(source_archive_sha256):
        plan_blockers.append("source_archive_sha256_missing_or_invalid")
    if candidate_archive_sha256 and not _is_sha256(candidate_archive_sha256):
        plan_blockers.append("candidate_archive_sha256_invalid")
    if archive_member_manifest_sha256 and not _is_sha256(archive_member_manifest_sha256):
        plan_blockers.append("archive_member_manifest_sha256_invalid")
    if not label_member:
        plan_blockers.append("charged_label_codebook_member_missing")
    if not payload_member:
        plan_blockers.append("categorical_payload_member_missing")
    if not runtime_member:
        plan_blockers.append("runtime_consumer_member_missing")
    for role, status in role_summary["required_roles"].items():
        if status["present"] is not True:
            plan_blockers.append(f"required_charged_member_role_missing:{role}")
    if conditioning_prior_contract["passed"] is not True:
        plan_blockers.extend(conditioning_prior_contract["dispatch_blockers"])

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CATEGORICAL_CONSTRUCTION_PLAN_KIND,
        "construction_plan_contract": CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "evidence_grade": "planning_manifest",
        "source_archive_sha256": source_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "archive_member_manifest_sha256": archive_member_manifest_sha256,
        "semantic_class_order": list(CONTEST_SEGNET_CLASS_NAME_TUPLE),
        "primary_label_sources": list(PRIMARY_LABEL_SOURCES),
        "semantic_weight_source": "SEMANTIC_QUANTIZATION_DEFAULT_BITS",
        "class_rows": [
            _class_row(
                class_id=item.class_id,
                charged_label_member=label_member,
                payload_member=payload_member,
            )
            for item in CONTEST_SEGNET_CLASSES
        ],
        "typed_label_atoms": build_categorical_typed_label_atoms(),
        "charged_member_summary": role_summary,
        "runtime_consumer_member": runtime_member,
        "conditioning_priors": conditioning_priors if isinstance(conditioning_priors, list) else [],
        "conditioning_prior_contract": conditioning_prior_contract,
        "candidate_construction_ready": not plan_blockers,
        "plan_blockers": plan_blockers,
        "byte_closed_archive_parity": {
            "required": True,
            "proven": False,
            "proof_path": "",
        },
        "dispatch_blockers": [
            "categorical_construction_plan_not_dispatchable",
            "real_byte_closed_archive_parity_missing",
        ],
        "next_required_proofs": [
            "replace_fixture_payload_with_real_charged_categorical_payload",
            "full_decode_reencode_parity_for_charged_categorical_payload",
            "runtime_loader_parity_against_archive_member",
            "exact_cuda_auth_eval_only_after_lane_claim",
        ],
    }


def audit_categorical_charged_label_plan(
    plan: Any,
    *,
    charged_member_names: list[str],
    charged_members: Any = None,
) -> dict[str, Any]:
    """Audit a declared categorical construction plan without making it dispatchable."""

    if plan is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": CATEGORICAL_CONSTRUCTION_PLAN_KIND,
            "declared": False,
            "accepted": True,
            "ready_for_exact_eval_dispatch": False,
            "validation_blockers": [],
            "planning_dispatch_blockers": [],
            "class_rows": [],
        }
    validation_blockers: list[str] = []
    if not isinstance(plan, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": CATEGORICAL_CONSTRUCTION_PLAN_KIND,
            "declared": True,
            "accepted": False,
            "ready_for_exact_eval_dispatch": False,
            "validation_blockers": ["construction_plan_not_object"],
            "planning_dispatch_blockers": [],
            "class_rows": [],
        }

    if plan.get("schema_version") != SCHEMA_VERSION:
        validation_blockers.append("schema_version_missing_or_invalid")
    if plan.get("kind") != CATEGORICAL_CONSTRUCTION_PLAN_KIND:
        validation_blockers.append("kind_missing_or_invalid")
    if plan.get("construction_plan_contract") != CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT:
        validation_blockers.append("contract_missing_or_invalid")
    if plan.get("score_claim") is not False:
        validation_blockers.append("score_claim_must_be_false")
    if plan.get("dispatch_attempted") is not False:
        validation_blockers.append("dispatch_attempted_must_be_false")
    if plan.get("ready_for_exact_eval_dispatch") is not False:
        validation_blockers.append("ready_for_exact_eval_dispatch_must_be_false")
    if plan.get("semantic_class_order") != list(CONTEST_SEGNET_CLASS_NAME_TUPLE):
        validation_blockers.append("semantic_class_order_mismatch")
    if plan.get("typed_label_atoms") != build_categorical_typed_label_atoms():
        validation_blockers.append("typed_label_atoms_mismatch")

    class_rows = plan.get("class_rows")
    expected_rows = [
        _class_row(
            class_id=item.class_id,
            charged_label_member="",
            payload_member="",
        )
        for item in CONTEST_SEGNET_CLASSES
    ]
    expected_by_id = {row["class_id"]: row for row in expected_rows}
    normalized_class_rows: list[dict[str, Any]] = []
    charged_names = {name for name in charged_member_names if _safe_member_name(name)}
    if not isinstance(class_rows, list) or len(class_rows) != len(CONTEST_SEGNET_CLASSES):
        validation_blockers.append("class_rows_missing_or_wrong_count")
    else:
        seen_ids: list[int] = []
        for row in class_rows:
            if not isinstance(row, dict):
                validation_blockers.append("class_row_not_object")
                continue
            class_id = row.get("class_id")
            if not isinstance(class_id, int) or class_id not in expected_by_id:
                validation_blockers.append("class_row_id_invalid")
                continue
            seen_ids.append(class_id)
            expected = expected_by_id[class_id]
            for key in (
                "comma10k_id",
                "name",
                "comma10k_color",
                "selfcomp_gray",
                "default_quant_bits",
                "semantic_priority_weight_ppm",
                "openpilot_prior_hint",
            ):
                if row.get(key) != expected[key]:
                    validation_blockers.append(f"class_row_{class_id}_{key}_mismatch")
            for key in ("charged_label_member", "categorical_payload_member"):
                member = row.get(key)
                if not _safe_member_name(member):
                    validation_blockers.append(f"class_row_{class_id}_{key}_missing_or_unsafe")
                elif member not in charged_names:
                    validation_blockers.append(f"class_row_{class_id}_{key}_not_charged:{member}")
            normalized_class_rows.append(row)
        if sorted(seen_ids) != list(range(len(CONTEST_SEGNET_CLASSES))):
            validation_blockers.append("class_row_ids_not_dense")

    runtime_member = plan.get("runtime_consumer_member")
    if not _safe_member_name(runtime_member):
        validation_blockers.append("runtime_consumer_member_missing_or_unsafe")
    elif runtime_member not in charged_names:
        validation_blockers.append(f"runtime_consumer_member_not_charged:{runtime_member}")

    byte_closed_archive_parity = plan.get("byte_closed_archive_parity")
    if not isinstance(byte_closed_archive_parity, dict):
        validation_blockers.append("byte_closed_archive_parity_missing")
    elif byte_closed_archive_parity.get("required") is not True:
        validation_blockers.append("byte_closed_archive_parity_required_must_be_true")

    recomputed_conditioning = audit_categorical_openpilot_mask_priors(
        plan.get("conditioning_priors"),
        charged_member_names=sorted(charged_names),
        charged_members=charged_members,
    )
    if recomputed_conditioning["passed"] is not True:
        validation_blockers.extend(recomputed_conditioning["dispatch_blockers"])

    planning_dispatch_blockers = plan.get("dispatch_blockers")
    if not isinstance(planning_dispatch_blockers, list):
        validation_blockers.append("dispatch_blockers_missing")
        planning_dispatch_blockers = []
    elif "real_byte_closed_archive_parity_missing" not in planning_dispatch_blockers:
        validation_blockers.append("real_byte_closed_archive_parity_blocker_missing")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CATEGORICAL_CONSTRUCTION_PLAN_KIND,
        "declared": True,
        "accepted": not validation_blockers,
        "contract": plan.get("construction_plan_contract", ""),
        "ready_for_exact_eval_dispatch": False,
        "candidate_construction_ready": plan.get("candidate_construction_ready") is True,
        "class_count": len(normalized_class_rows),
        "class_rows": normalized_class_rows,
        "runtime_consumer_member": runtime_member if isinstance(runtime_member, str) else "",
        "conditioning_prior_contract": recomputed_conditioning,
        "byte_closed_archive_parity": byte_closed_archive_parity
        if isinstance(byte_closed_archive_parity, dict)
        else {},
        "validation_blockers": validation_blockers,
        "planning_dispatch_blockers": planning_dispatch_blockers,
    }


__all__ = [
    "CATEGORICAL_CLASS_CODEBOOK_CONTRACT",
    "CATEGORICAL_CLASS_CODEBOOK_KIND",
    "CATEGORICAL_CONSTRUCTION_PLAN_CONTRACT",
    "CATEGORICAL_CONSTRUCTION_PLAN_KIND",
    "PRIMARY_LABEL_SOURCES",
    "RUNTIME_LABEL_CONTRACT",
    "SCHEMA_VERSION",
    "audit_categorical_charged_label_plan",
    "build_categorical_charged_label_plan",
    "build_categorical_class_codebook",
]
