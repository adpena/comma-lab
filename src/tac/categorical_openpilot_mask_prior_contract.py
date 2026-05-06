"""Byte-closed contract for categorical CLADE/SPADE/openpilot mask priors."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
KIND = "categorical_openpilot_mask_prior_contract"

KNOWN_CONDITIONING_PRIOR_FAMILIES = (
    "qma9",
    "clade_spade",
    "openpilot_priors",
)
COMPRESSION_TIME_ONLY_USAGE = "compression_time_atom_ranking_only"
RUNTIME_CONDITIONING_USAGES = (
    "decode_runtime_conditioning",
    "inflate_runtime_conditioning",
    "runtime_conditioning",
)
FAMILIES_REQUIRING_CHARGED_RUNTIME_MEMBER = (
    "clade_spade",
    "openpilot_priors",
)


def _safe_member_name(name: Any) -> bool:
    if not isinstance(name, str) or not name:
        return False
    path = PurePosixPath(name)
    parts = path.parts
    return (
        not path.is_absolute()
        and ".." not in parts
        and all(part not in {"", ".", "__MACOSX"} for part in parts)
        and not any(part.startswith(".") for part in parts)
    )


def _record_id(index: int, family: str, name: str) -> str:
    return f"{family}:{name}" if family and name else f"record_{index}"


def audit_categorical_openpilot_mask_priors(
    records: Any,
    *,
    charged_member_names: Iterable[str],
) -> dict[str, Any]:
    """Audit declared categorical/openpilot priors against charged archive members.

    Openpilot/supercombo-derived state may rank atoms at compression time without
    being charged. If any derived prior is consumed by inflate/runtime, the
    manifest must point at a declared charged member. CLADE/SPADE parameters are
    runtime conditioning by construction, so they also require charged custody.
    """

    charged_names = {name for name in charged_member_names if isinstance(name, str)}
    blockers: list[str] = []
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []

    if records is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "declared": False,
            "passed": True,
            "runtime_consumed_count": 0,
            "compression_time_only_count": 0,
            "records": [],
            "dispatch_blockers": [],
            "warnings": [],
        }
    if not isinstance(records, list):
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "declared": True,
            "passed": False,
            "runtime_consumed_count": 0,
            "compression_time_only_count": 0,
            "records": [],
            "dispatch_blockers": ["conditioning_priors_not_list"],
            "warnings": [],
        }

    runtime_count = 0
    compression_only_count = 0
    seen_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            blockers.append(f"conditioning_prior_{index}_not_object")
            continue
        family = record.get("family")
        name = record.get("name")
        usage = record.get("usage")
        charged_member = record.get("charged_member")
        family_str = family if isinstance(family, str) else ""
        name_str = name if isinstance(name, str) else ""
        usage_str = usage if isinstance(usage, str) else ""
        record_id = _record_id(index, family_str, name_str)
        if family_str not in KNOWN_CONDITIONING_PRIOR_FAMILIES:
            blockers.append(f"conditioning_prior_unknown_family:{record_id}")
        if not name_str:
            blockers.append(f"conditioning_prior_name_missing:{record_id}")
        if usage_str not in (*RUNTIME_CONDITIONING_USAGES, COMPRESSION_TIME_ONLY_USAGE):
            blockers.append(f"conditioning_prior_usage_unknown:{record_id}")

        runtime_consumed = (
            record.get("runtime_consumed") is True
            or usage_str in RUNTIME_CONDITIONING_USAGES
            or family_str == "clade_spade"
        )
        compression_time_only = (
            usage_str == COMPRESSION_TIME_ONLY_USAGE
            and record.get("runtime_consumed") is not True
            and family_str != "clade_spade"
        )
        requires_charged_member = runtime_consumed and (
            family_str in FAMILIES_REQUIRING_CHARGED_RUNTIME_MEMBER
            or usage_str in RUNTIME_CONDITIONING_USAGES
        )

        charged_member_str = charged_member if isinstance(charged_member, str) else ""
        if requires_charged_member:
            if not _safe_member_name(charged_member_str):
                blockers.append(f"conditioning_prior_charged_member_missing_or_unsafe:{record_id}")
            elif charged_member_str not in charged_names:
                blockers.append(
                    "conditioning_prior_charged_member_not_declared:"
                    f"{record_id}:{charged_member_str}"
                )
        elif charged_member_str and charged_member_str not in charged_names:
            blockers.append(
                "conditioning_prior_charged_member_not_declared:"
                f"{record_id}:{charged_member_str}"
            )

        if runtime_consumed:
            runtime_count += 1
        if compression_time_only:
            compression_only_count += 1
        seen_ids.append(record_id)
        normalized.append(
            {
                "index": index,
                "family": family_str,
                "name": name_str,
                "usage": usage_str,
                "runtime_consumed": runtime_consumed,
                "compression_time_only": compression_time_only,
                "requires_charged_member": requires_charged_member,
                "charged_member": charged_member_str,
            }
        )

    duplicate_ids = sorted({item for item in seen_ids if seen_ids.count(item) > 1})
    if duplicate_ids:
        blockers.append("conditioning_prior_duplicate_ids")
        warnings.append(f"duplicate conditioning prior ids: {duplicate_ids}")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "declared": True,
        "passed": not blockers,
        "runtime_consumed_count": runtime_count,
        "compression_time_only_count": compression_only_count,
        "records": normalized,
        "dispatch_blockers": blockers,
        "warnings": warnings,
    }


__all__ = [
    "COMPRESSION_TIME_ONLY_USAGE",
    "FAMILIES_REQUIRING_CHARGED_RUNTIME_MEMBER",
    "KIND",
    "KNOWN_CONDITIONING_PRIOR_FAMILIES",
    "RUNTIME_CONDITIONING_USAGES",
    "SCHEMA_VERSION",
    "audit_categorical_openpilot_mask_priors",
]
