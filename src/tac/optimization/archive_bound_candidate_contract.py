# SPDX-License-Identifier: MIT
"""Common archive-bound candidate contract for entropy/archive materializers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    ordered_unique,
    require_no_truthy_authority_fields,
)
from tac.repo_io import sha256_file

ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA = "tac_archive_bound_candidate_contract.v1"
ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA = (
    "tac_archive_bound_candidate_contract_surface.v1"
)
ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA = (
    "tac_archive_bound_candidate_adapter_package.v1"
)
ARCHIVE_BOUND_CANDIDATE_CONTRACT_PAYLOAD_KEYS = frozenset(
    {
        "archive_bound_candidate_contract",
        "archive_bound_candidate_contract_surface",
        "archive_bound_candidate_contract_schema",
        "archive_bound_candidate_contract_surface_schema",
    }
)
FALSE_AUTHORITY: dict[str, bool] = {
    **PROXY_FALSE_AUTHORITY_FIELDS,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


class ArchiveBoundCandidateContractError(ValueError):
    """Raised when archive-bound candidate contract payloads are invalid."""


_DUPLICATE_BOOL_CONTRACT_FIELDS = (
    "archive_bound_candidate_ready",
    "archive_bound_candidate_ready_for_exact_handoff",
    "byte_closed_candidate_materialized",
    "candidate_archive_materialized",
    "runtime_consumption_proof_ready",
    "receiver_contract_satisfied",
    "runtime_adapter_ready",
    "contest_runtime_decoder_adapter_ready",
    "ready_for_exact_eval_dispatch",
    "score_claim",
    "promotion_eligible",
    "rank_or_kill_eligible",
)


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _safe_int(value: Any, *, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _anti_pattern_id_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        direct = _first_text(
            value.get("anti_pattern_id"),
            value.get("canonical_anti_pattern_id"),
            value.get("pattern_id"),
            value.get("id"),
        )
        return [direct] if direct else []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            out.extend(_anti_pattern_id_values(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _canonical_anti_pattern_ids(candidate: Mapping[str, Any]) -> list[str]:
    coverage = _mapping(candidate.get("archive_entropy_substrate_coverage"))
    return ordered_unique(
        [
            *_anti_pattern_id_values(candidate.get("anti_pattern_id")),
            *_anti_pattern_id_values(candidate.get("anti_pattern_ids")),
            *_anti_pattern_id_values(candidate.get("canonical_anti_pattern_id")),
            *_anti_pattern_id_values(candidate.get("canonical_anti_pattern_ids")),
            *_anti_pattern_id_values(candidate.get("anti_pattern_matches")),
            *_anti_pattern_id_values(candidate.get("anti_pattern_protections")),
            *_anti_pattern_id_values(coverage.get("anti_pattern_protections")),
            *_anti_pattern_id_values(coverage.get("anti_pattern_matches")),
        ]
    )


def _anti_pattern_acquisition_penalty(anti_pattern_ids: Sequence[str]) -> float:
    penalty = 0.0
    for raw_id in anti_pattern_ids:
        anti_pattern_id = str(raw_id or "").lower()
        if not anti_pattern_id:
            continue
        if (
            "proxy_or_advisory" in anti_pattern_id
            or "masquerades_as_score" in anti_pattern_id
        ):
            penalty += 0.18
        elif (
            "probe_only_side_report" in anti_pattern_id
            or "orphaned_from_optimizer" in anti_pattern_id
        ):
            penalty += 0.14
        elif "without_receiver_consumption" in anti_pattern_id:
            penalty += 0.12
        elif "zero_order_entropy" in anti_pattern_id:
            penalty += 0.10
        elif (
            "entropy_coder_order_cargo_cult" in anti_pattern_id
            or "canonical_default_plateau_substrate" in anti_pattern_id
        ):
            penalty += 0.08
        elif "micro_optimization_without_macro_escape" in anti_pattern_id:
            penalty += 0.06
        elif "hnerv_pr95_language_anchoring" in anti_pattern_id:
            penalty += 0.04
        elif "cargo_cult" in anti_pattern_id or "false_authority" in anti_pattern_id:
            penalty += 0.08
        elif "orphan" in anti_pattern_id:
            penalty += 0.07
        elif (
            "probe" in anti_pattern_id
            or "prototype" in anti_pattern_id
            or "stale" in anti_pattern_id
            or "mismatch" in anti_pattern_id
        ):
            penalty += 0.04
    return min(0.35, penalty)


def _archive_field_value(
    row: Mapping[str, Any],
    *,
    object_key: str,
    direct_keys: Sequence[str],
    nested_keys: Sequence[str],
) -> Any:
    for key in direct_keys:
        if key in row and row.get(key) not in ("", None):
            return row.get(key)
    nested = _mapping(row.get(object_key))
    for key in nested_keys:
        if key in nested and nested.get(key) not in ("", None):
            return nested.get(key)
    return None


def _archive_path_values_differ(row_value: Any, contract_value: Any) -> bool:
    row_text = str(row_value).strip().replace("\\", "/")
    contract_text = str(contract_value).strip().replace("\\", "/")
    if not row_text or not contract_text:
        return False
    row_norm = row_text.lstrip("./")
    contract_norm = contract_text.lstrip("./")
    if row_norm == contract_norm:
        return False
    if row_text.startswith("/") and not contract_text.startswith("/"):
        return not row_text.endswith(f"/{contract_norm}")
    if contract_text.startswith("/") and not row_text.startswith("/"):
        return not contract_text.endswith(f"/{row_norm}")
    return True


def archive_bound_candidate_contract_stale_field_blockers(
    row: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return duplicate-field mismatches between a row and its contract.

    Legacy emitters still carry historic readiness/archive fields next to the
    shared contract. Consumers must fail closed when those duplicate fields
    disagree instead of picking whichever one is convenient.
    """

    resolved_contract = _mapping(
        contract
        if contract is not None
        else row.get("archive_bound_candidate_contract")
    )
    if not resolved_contract:
        return []
    blockers: list[str] = []
    embedded_schema = row.get("archive_bound_candidate_contract_schema")
    if (
        embedded_schema not in ("", None)
        and embedded_schema != ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    ):
        blockers.append("archive_bound_contract_stale_duplicate_field:schema")
    for key in _DUPLICATE_BOOL_CONTRACT_FIELDS:
        if (
            key in row
            and isinstance(row.get(key), bool)
            and isinstance(resolved_contract.get(key), bool)
            and row.get(key) != resolved_contract.get(key)
        ):
            blockers.append(f"archive_bound_contract_stale_duplicate_field:{key}")
    candidate_archive = _mapping(resolved_contract.get("candidate_archive"))
    source_archive = _mapping(resolved_contract.get("source_archive"))
    duplicate_specs = (
        (
            "candidate_archive_path",
            _archive_field_value(
                row,
                object_key="candidate_archive",
                direct_keys=("candidate_archive_path", "archive_path"),
                nested_keys=("path", "archive_path"),
            ),
            candidate_archive.get("path"),
        ),
        (
            "candidate_archive_sha256",
            _archive_field_value(
                row,
                object_key="candidate_archive",
                direct_keys=("candidate_archive_sha256", "archive_sha256"),
                nested_keys=("sha256", "archive_sha256"),
            ),
            candidate_archive.get("sha256"),
        ),
        (
            "candidate_archive_bytes",
            _archive_field_value(
                row,
                object_key="candidate_archive",
                direct_keys=("candidate_archive_bytes", "archive_bytes"),
                nested_keys=("bytes", "archive_bytes"),
            ),
            candidate_archive.get("bytes"),
        ),
        (
            "source_archive_path",
            _archive_field_value(
                row,
                object_key="source_archive",
                direct_keys=("source_archive_path",),
                nested_keys=("path", "archive_path"),
            ),
            source_archive.get("path"),
        ),
        (
            "source_archive_sha256",
            _archive_field_value(
                row,
                object_key="source_archive",
                direct_keys=("source_archive_sha256",),
                nested_keys=("sha256", "archive_sha256"),
            ),
            source_archive.get("sha256"),
        ),
        (
            "source_archive_bytes",
            _archive_field_value(
                row,
                object_key="source_archive",
                direct_keys=("source_archive_bytes",),
                nested_keys=("bytes", "archive_bytes"),
            ),
            source_archive.get("bytes"),
        ),
    )
    for field, row_value, contract_value in duplicate_specs:
        if row_value in ("", None) or contract_value in ("", None):
            continue
        if field.endswith("_path"):
            if _archive_path_values_differ(row_value, contract_value):
                blockers.append(f"archive_bound_contract_stale_duplicate_field:{field}")
            continue
        if str(row_value) != str(contract_value):
            blockers.append(f"archive_bound_contract_stale_duplicate_field:{field}")
    return ordered_unique(blockers)


def require_fresh_archive_bound_candidate_contract_row(
    row: Mapping[str, Any],
    *,
    label: str = "archive_bound_candidate_contract_row",
) -> None:
    blockers = archive_bound_candidate_contract_stale_field_blockers(row)
    if blockers:
        raise ArchiveBoundCandidateContractError(f"{label}: {', '.join(blockers)}")


def _validated_contract_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    require_no_truthy_authority_fields(payload, context=label)
    if payload.get("schema") != ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA:
        raise ArchiveBoundCandidateContractError(f"{label} schema mismatch")
    return dict(payload)


def archive_bound_candidate_contracts_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str = "archive_bound_candidate_contract_payload",
) -> list[dict[str, Any]]:
    """Extract validated archive-bound contracts from any shared payload shape."""

    require_no_truthy_authority_fields(payload, context=label)
    schema = payload.get("schema")
    if schema == ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA:
        rows = payload.get("candidate_rows")
        if isinstance(rows, Sequence) and not isinstance(rows, bytes | bytearray):
            for row_index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise ArchiveBoundCandidateContractError(
                        f"{label} candidate_rows[{row_index}] must be object"
                    )
                require_fresh_archive_bound_candidate_contract_row(
                    row,
                    label=f"{label} candidate_rows[{row_index}]",
                )
        nested_surfaces = payload.get("archive_bound_candidate_contract_surfaces")
        if not isinstance(nested_surfaces, Sequence) or isinstance(
            nested_surfaces, bytes | bytearray
        ):
            raise ArchiveBoundCandidateContractError(
                f"{label} adapter package lacks archive_bound_candidate_contract_surfaces[]"
            )
        contracts: list[dict[str, Any]] = []
        for index, surface in enumerate(nested_surfaces):
            if not isinstance(surface, Mapping):
                raise ArchiveBoundCandidateContractError(
                    f"{label} archive_bound_candidate_contract_surfaces[{index}] must be object"
                )
            contracts.extend(
                archive_bound_candidate_contracts_from_payload(
                    surface,
                    label=(
                        f"{label} archive_bound_candidate_contract_surfaces[{index}]"
                    ),
                )
            )
        return contracts
    if schema == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA:
        return [_validated_contract_from_payload(payload, label=label)]
    if schema == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA:
        contracts = payload.get("candidate_contracts")
        if not isinstance(contracts, Sequence) or isinstance(
            contracts, bytes | bytearray
        ):
            raise ArchiveBoundCandidateContractError(
                f"{label} candidate_contracts[] missing"
            )
        out: list[dict[str, Any]] = []
        for index, contract in enumerate(contracts):
            if not isinstance(contract, Mapping):
                raise ArchiveBoundCandidateContractError(
                    f"{label} candidate_contracts[{index}] must be object"
                )
            out.append(
                _validated_contract_from_payload(
                    contract,
                    label=f"{label} candidate_contracts[{index}]",
                )
            )
        return out
    embedded_contract = _optional_mapping(
        payload.get("archive_bound_candidate_contract")
    )
    if embedded_contract is not None:
        require_fresh_archive_bound_candidate_contract_row(payload, label=label)
        return [_validated_contract_from_payload(embedded_contract, label=label)]
    embedded_surface = _optional_mapping(
        payload.get("archive_bound_candidate_contract_surface")
    )
    if embedded_surface is not None:
        require_fresh_archive_bound_candidate_contract_row(payload, label=label)
        return archive_bound_candidate_contracts_from_payload(
            embedded_surface,
            label=f"{label} archive_bound_candidate_contract_surface",
        )
    raise ArchiveBoundCandidateContractError(f"{label} schema mismatch: {schema!r}")


def has_archive_bound_candidate_contract_payload(payload: Mapping[str, Any]) -> bool:
    """Return true when ``payload`` carries any shared contract surface."""

    return any(key in payload for key in ARCHIVE_BOUND_CANDIDATE_CONTRACT_PAYLOAD_KEYS)


def selected_archive_bound_candidate_contract_from_payload(
    payload: Mapping[str, Any],
    *,
    label: str = "archive_bound_candidate_contract_payload",
) -> dict[str, Any] | None:
    """Return the selected contract from any shared payload shape.

    This is the consumer-side helper for queue/readiness code that needs one
    canonical custody/readiness contract and must inherit stale-field rejection
    from ``archive_bound_candidate_contracts_from_payload``.
    """

    contracts = archive_bound_candidate_contracts_from_payload(payload, label=label)
    selected = [
        contract
        for contract in contracts
        if contract.get("selected_archive_transform_variant") is True
    ]
    return dict(selected[0] if selected else contracts[0] if contracts else {})


def source_archive_bound_candidate_contract_from_row(
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the source archive-bound contract snapshot carried by a promoted row."""

    source_contract = row.get("source_archive_bound_candidate_contract")
    if isinstance(source_contract, Mapping):
        return source_contract
    embedded_contract = row.get("archive_bound_candidate_contract")
    if isinstance(embedded_contract, Mapping):
        return embedded_contract
    return {}


def source_archive_bound_contract_candidate_archive(
    row: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return candidate-archive custody from a source contract snapshot."""

    return _mapping(
        source_archive_bound_candidate_contract_from_row(row).get("candidate_archive")
    )


def source_archive_bound_contract_snapshot_blockers(
    row: Mapping[str, Any],
    *,
    required: bool | None = None,
) -> list[str]:
    """Validate a promoted exact-ready row's source contract snapshot.

    The source snapshot is evidence about the upstream byte-closed materializer,
    not a new dispatch-authority contract.  Therefore raw exact-ready fields on
    the promoted row may be true, but the snapshot must still prove archive
    readiness, exact-handoff readiness, and archive SHA/byte custody.
    """

    snapshot_required = (
        row.get("source_archive_bound_candidate_contract_required") is True
        if required is None
        else required
    )
    contract = source_archive_bound_candidate_contract_from_row(row)
    if snapshot_required and not contract:
        return ["source_archive_bound_candidate_contract_missing"]
    if not contract:
        return []

    blockers: list[str] = []
    if contract.get("schema") != ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA:
        blockers.append("source_archive_bound_candidate_contract_schema_mismatch")
    if contract.get("archive_bound_candidate_ready") is not True:
        blockers.append("source_archive_bound_candidate_contract_not_ready")
    if contract.get("archive_bound_candidate_ready_for_exact_handoff") is not True:
        blockers.append(
            "source_archive_bound_candidate_contract_not_ready_for_exact_handoff"
        )
    custody = _mapping(contract.get("archive_file_custody"))
    if (snapshot_required and not custody) or (
        custody and custody.get("custody_complete") is not True
    ):
        blockers.append("source_archive_bound_candidate_contract_custody_incomplete")

    archive = _mapping(contract.get("candidate_archive"))
    contract_sha = archive.get("sha256") or archive.get("archive_sha256")
    row_sha = row.get("candidate_archive_sha256") or row.get("archive_sha256")
    if isinstance(contract_sha, str) and len(contract_sha.strip()) == 64:
        if (
            isinstance(row_sha, str)
            and len(row_sha.strip()) == 64
            and contract_sha.strip().lower() != row_sha.strip().lower()
        ):
            blockers.append("source_archive_bound_candidate_contract_sha256_mismatch")
    elif snapshot_required:
        blockers.append("source_archive_bound_candidate_contract_sha256_missing")

    contract_bytes = archive.get("bytes") or archive.get("archive_bytes")
    row_bytes = row.get("candidate_archive_bytes") or row.get("archive_bytes")
    if isinstance(contract_bytes, int) and not isinstance(contract_bytes, bool):
        if (
            isinstance(row_bytes, int)
            and not isinstance(row_bytes, bool)
            and contract_bytes != row_bytes
        ):
            blockers.append("source_archive_bound_candidate_contract_bytes_mismatch")
    elif snapshot_required:
        blockers.append("source_archive_bound_candidate_contract_bytes_missing")
    return ordered_unique(blockers)


def _resolve(path: str | Path, repo_root: str | Path | None) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute() or repo_root is None:
        return value
    return Path(repo_root) / value


def _repo_rel(path: Path, repo_root: str | Path | None) -> str:
    if repo_root is None:
        return path.as_posix()
    repo = Path(repo_root)
    try:
        return (
            path.resolve(strict=False)
            .relative_to(repo.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def archive_substrate_tags_for_transform_kind(transform_kind: str) -> list[str]:
    """Classify an archive transform into reusable acquisition substrates."""

    kind = str(transform_kind or "").lower()
    tags: list[str] = []
    if "public_frontier" in kind:
        tags.append("public_frontier")
    if "byte_range" in kind:
        tags.append("byte_range")
    if "dqs1" in kind or "pairset" in kind:
        tags.extend(["dqs1", "selector"])
    if "fec" in kind:
        tags.extend(["fec", "selector_stream"])
    if "selector" in kind:
        tags.append("selector")
    if "huffman" in kind or "fixed_k" in kind or "fec5" in kind or "fec6" in kind:
        tags.append("huffman")
    if "header" in kind:
        tags.append("header")
    if (
        "range_coder" in kind
        or "arithmetic" in kind
        or "lc_ac" in kind
        or "entropy_recode" in kind
    ):
        tags.extend(["range_coder", "entropy_coder"])
    if "ans_coder" in kind:
        tags.extend(["ans_coder", "entropy_coder"])
    if "zip" in kind or "repack" in kind:
        tags.extend(["zip_ordering", "zip_container"])
    if "packet_member" in kind:
        tags.extend(["zip_member", "member_payload"])
    neural_predictive_archive = (
        "z7" in kind
        or "mamba" in kind
        or "predictive_coding" in kind
        or "neural_archive" in kind
        or "ssd_reference" in kind
        or ("mlx" in kind and ("archive" in kind or "substrate" in kind))
    )
    if neural_predictive_archive:
        tags.extend(
            [
                "neural_archive",
                "predictive_coding",
                "mlx_substrate",
            ]
        )
        if "z7" in kind or "mamba" in kind:
            tags.append("z7_mamba2")
        if "z6" in kind or "rao_ballard" in kind:
            tags.append("z6_v2")
        if "z5" in kind:
            tags.append("z5")
        if "ssd" in kind or "ssd_reference" in kind:
            tags.append("ssd_reference")
    if "prototype" in kind:
        tags.append("receiver_adapter")
    if not tags:
        tags.append("archive_transform")
    return ordered_unique(tags)


def entropy_position_label_for_transform_kind(transform_kind: str) -> str:
    """Return the entropy-pipeline position for a transform kind."""

    kind = str(transform_kind or "").lower()
    if not kind:
        return "archive_transform_unknown_entropy_position"
    if "zip_header" in kind or "header_elide" in kind:
        return "after_entropy_coder"
    if "zip_order" in kind:
        return "after_entropy_coder"
    if "dqs1" in kind or "pairset" in kind or "selector" in kind or "fec" in kind:
        return "before_entropy_coder"
    neural_predictive_archive = (
        "z7" in kind
        or "mamba" in kind
        or "predictive_coding" in kind
        or "neural_archive" in kind
        or "ssd_reference" in kind
        or ("mlx" in kind and ("archive" in kind or "substrate" in kind))
    )
    if neural_predictive_archive:
        return "before_entropy_coder"
    if any(
        token in kind
        for token in (
            "range",
            "arithmetic",
            "ans",
            "huffman",
            "entropy_recode",
            "recompress",
            "brotli",
            "lzma",
            "zip_repack",
            "packet_member_merge",
            "tensor_factorize",
            "section_entropy",
        )
    ):
        return "at_entropy_coder"
    return "archive_transform_unknown_entropy_position"


def _file_custody(
    *,
    path_text: str,
    expected_sha256: str | None,
    expected_bytes: int | None,
    repo_root: str | Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not path_text:
        blockers.append("archive_bound_candidate_path_missing")
        return {
            "schema": "tac_archive_bound_candidate_file_custody.v1",
            "path": None,
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "sha256_matches": False,
            "bytes_match": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    path = _resolve(path_text, repo_root)
    if not path.is_file():
        blockers.append("archive_bound_candidate_file_missing")
        return {
            "schema": "tac_archive_bound_candidate_file_custody.v1",
            "path": _repo_rel(path, repo_root),
            "present": False,
            "sha256": None,
            "bytes": None,
            "expected_sha256": expected_sha256,
            "expected_bytes": expected_bytes,
            "sha256_matches": False,
            "bytes_match": False,
            "custody_complete": False,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    actual_sha = sha256_file(path)
    actual_bytes = path.stat().st_size
    sha_matches = bool(expected_sha256 and actual_sha == expected_sha256)
    bytes_match = expected_bytes is None or actual_bytes == expected_bytes
    if not expected_sha256:
        blockers.append("archive_bound_candidate_sha256_missing")
    elif not sha_matches:
        blockers.append("archive_bound_candidate_sha256_mismatch")
    if expected_bytes is not None and not bytes_match:
        blockers.append("archive_bound_candidate_bytes_mismatch")
    return {
        "schema": "tac_archive_bound_candidate_file_custody.v1",
        "path": _repo_rel(path, repo_root),
        "present": True,
        "sha256": actual_sha,
        "bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "sha256_matches": sha_matches,
        "bytes_match": bytes_match,
        "custody_complete": not blockers,
        "blockers": ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _runtime_consumption_proof_ready(row: Mapping[str, Any]) -> bool:
    if row.get("runtime_consumption_proof_ready") is True:
        return True
    if row.get("runtime_consumption_proof_passed") is True:
        return True
    if row.get("receiver_contract_satisfied") is not True:
        return False
    return str(row.get("runtime_consumption_proof_status") or "") in {
        "present",
        "revalidated",
        "archive_bound_proof_custody_present",
    }


def _runtime_adapter_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(_mapping(row.get("runtime_adapter_manifest")))
    for key in (
        "runtime_tree_sha256",
        "runtime_content_tree_sha256",
        "candidate_runtime_tree_sha256",
        "expected_runtime_tree_sha256",
        "candidate_runtime_dir",
        "runtime_dir",
        "source_runtime_dir",
        "submission_dir",
    ):
        if key in row and key not in manifest:
            manifest[key] = row[key]
    if row.get("runtime_adapter_ready") is True:
        manifest["runtime_adapter_ready"] = True
    return manifest


def archive_bound_candidate_contract_fields_for_row(
    row: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    selected_transform_kind: str | None = None,
    source_context: Mapping[str, Any] | None = None,
    family_id: str | None = None,
    typed_response_id: str | None = None,
    candidate_chain_id: str | None = None,
    entropy_position_label: str | None = None,
    entropy_stage_order: int | None = None,
    byte_credit_budget: int | None = None,
) -> dict[str, Any]:
    """Return canonical archive-bound contract fields for a generic row.

    This is the migration shim for existing emitters: rows may still use
    historic names such as ``candidate_archive_path`` or nested
    ``candidate_archive`` objects, but downstream consumers receive one shared
    contract surface.
    """

    candidate_archive = _mapping(row.get("candidate_archive"))
    source_archive = _mapping(row.get("source_archive"))
    transform_kind = _first_text(
        selected_transform_kind,
        row.get("archive_native_transform_kind"),
        row.get("target_kind"),
        row.get("materializer_id"),
        row.get("schema"),
    )
    runtime_manifest = _runtime_adapter_manifest(row)
    candidate = {
        "archive_native_transform_kind": transform_kind,
        "path": _first_text(
            row.get("candidate_archive_path"),
            row.get("archive_path"),
            candidate_archive.get("path"),
        ),
        "sha256": _first_text(
            row.get("candidate_archive_sha256"),
            row.get("archive_sha256"),
            candidate_archive.get("sha256"),
            candidate_archive.get("archive_sha256"),
        ),
        "bytes": _first_int(
            row.get("candidate_archive_bytes"),
            row.get("archive_bytes"),
            candidate_archive.get("bytes"),
            candidate_archive.get("archive_bytes"),
        ),
        "source_archive_path": _first_text(
            row.get("source_archive_path"),
            source_archive.get("path"),
        ),
        "source_archive_sha256": _first_text(
            row.get("source_archive_sha256"),
            source_archive.get("sha256"),
            source_archive.get("archive_sha256"),
        ),
        "source_archive_bytes": _first_int(
            row.get("source_archive_bytes"),
            source_archive.get("bytes"),
            source_archive.get("archive_bytes"),
        ),
        "materialized": (
            row.get("byte_closed_candidate_emitted") is True
            or row.get("byte_closed_candidate_materialized") is True
            or row.get("candidate_archive_materialized") is True
        ),
        "runtime_consumption_proof_ready": _runtime_consumption_proof_ready(row),
        "runtime_consumption_proof_path": _first_text(
            row.get("runtime_consumption_proof_path"),
            _mapping(row.get("runtime_consumption_proof")).get("path"),
        ),
        "receiver_contract_kind": row.get("receiver_contract_kind"),
        "receiver_contract_satisfied": row.get("receiver_contract_satisfied") is True,
        "runtime_adapter_ready": (
            row.get("runtime_adapter_ready") is True
            or runtime_manifest.get("runtime_adapter_ready") is True
        ),
        "runtime_adapter_manifest": runtime_manifest,
        "contest_runtime_decoder_adapter_ready": (
            row.get("contest_runtime_decoder_adapter_ready") is True
            or runtime_manifest.get("contest_runtime_decoder_adapter_ready") is True
            or bool(row.get("runtime_content_tree_sha256"))
        ),
        "semantic_payload_changed": row.get("semantic_payload_changed") is True,
        "score_affecting_payload_changed": (
            row.get("score_affecting_payload_changed") is True
        ),
        "exact_axis_score_affecting_adjudication_required": (
            row.get("exact_axis_score_affecting_adjudication_required") is True
        ),
        "charged_bits_changed": row.get("charged_bits_changed") is True,
        "prototype_only": row.get("prototype_only") is True,
        "entropy_probe_path": row.get("entropy_probe_path"),
        "saved_bytes": row.get("realized_saved_bytes") or row.get("saved_bytes"),
        "estimated_zero_order_savings_bytes": row.get(
            "estimated_zero_order_savings_bytes"
        ),
        "blockers": ordered_unique(
            [
                *_string_list(row.get("blockers")),
                *_string_list(row.get("readiness_blockers")),
                *_string_list(row.get("dispatch_blockers")),
            ]
        ),
        "anti_pattern_id": row.get("anti_pattern_id"),
        "anti_pattern_ids": row.get("anti_pattern_ids"),
        "canonical_anti_pattern_id": row.get("canonical_anti_pattern_id"),
        "canonical_anti_pattern_ids": row.get("canonical_anti_pattern_ids"),
        "anti_pattern_matches": row.get("anti_pattern_matches"),
        "anti_pattern_protections": row.get("anti_pattern_protections"),
        "archive_entropy_substrate_coverage": row.get(
            "archive_entropy_substrate_coverage"
        ),
    }
    resolved_source_context = {
        **dict(_mapping(source_context)),
        **{
            key: value
            for key, value in {
                "path": candidate["source_archive_path"],
                "sha256": candidate["source_archive_sha256"],
                "bytes": candidate["source_archive_bytes"],
            }.items()
            if value not in ("", None)
        },
    }
    surface = build_archive_bound_candidate_contract_surface(
        candidates=[candidate],
        selected_transform_kind=transform_kind,
        repo_root=repo_root,
        source_context=resolved_source_context,
        family_id=family_id
        or _first_text(row.get("family_id"), row.get("candidate_family")),
        typed_response_id=typed_response_id
        or _first_text(row.get("typed_response_id")),
        candidate_chain_id=(
            candidate_chain_id
            or _first_text(row.get("candidate_chain_id"), row.get("candidate_id"))
        ),
        entropy_position_label=(
            entropy_position_label
            or _first_text(row.get("entropy_position_label"))
            or entropy_position_label_for_transform_kind(transform_kind)
        ),
        entropy_stage_order=entropy_stage_order,
        byte_credit_budget=byte_credit_budget,
    )
    contract = dict(_mapping(surface.get("selected_candidate_contract")))
    return {
        "archive_bound_candidate_contract_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
        ),
        "archive_bound_candidate_contract": contract,
        "archive_bound_candidate_contract_surface_schema": (
            ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
        ),
        "archive_bound_candidate_contract_surface": surface,
    }


def _contract_penalty(
    *,
    blockers: Sequence[str],
    anti_pattern_ids: Sequence[str],
    materialized: bool,
    receiver_proof_ready: bool,
    receiver_contract_satisfied: bool,
    byte_credit_exhausted: bool,
    prototype_only: bool,
    probe_only: bool,
) -> float:
    penalty = 0.0
    if not materialized:
        penalty += 0.35
    if not receiver_proof_ready:
        penalty += 0.20
    if not receiver_contract_satisfied:
        penalty += 0.18
    if byte_credit_exhausted:
        penalty += 0.12
    if prototype_only:
        penalty += 0.04
    if probe_only:
        penalty += 0.16
    blocker_text = " ".join(blockers)
    if (
        "materializer_missing" in blocker_text
        or "target_entropy_coder_prototype_missing" in blocker_text
    ):
        penalty += 0.12
    if (
        "runtime_proof_missing" in blocker_text
        or "receiver_runtime_proof_missing" in blocker_text
    ):
        penalty += 0.10
    if "orphan" in blocker_text or "probe_only" in blocker_text:
        penalty += 0.08
    penalty += _anti_pattern_acquisition_penalty(anti_pattern_ids)
    return round(min(0.95, penalty), 6)


def build_archive_bound_candidate_contract(
    *,
    candidate: Mapping[str, Any],
    repo_root: str | Path | None = None,
    selected: bool = False,
    source_context: Mapping[str, Any] | None = None,
    family_id: str | None = None,
    typed_response_id: str | None = None,
    candidate_chain_id: str | None = None,
    entropy_position_label: str | None = None,
    entropy_stage_order: int | None = None,
    byte_credit_budget: int | None = None,
) -> dict[str, Any]:
    """Normalize any archive/entropy materializer output into one contract."""

    source = dict(_mapping(source_context))
    transform_kind = str(candidate.get("archive_native_transform_kind") or "")
    resolved_entropy_position_label = (
        entropy_position_label
        if isinstance(entropy_position_label, str) and entropy_position_label.strip()
        else entropy_position_label_for_transform_kind(transform_kind)
    )
    materialized = candidate.get("materialized") is True
    proof_ready = candidate.get("runtime_consumption_proof_ready") is True
    receiver_satisfied = candidate.get("receiver_contract_satisfied") is True
    probe_only = (
        bool(str(candidate.get("entropy_probe_path") or "").strip())
        and not materialized
    )
    prototype_only = candidate.get("prototype_only") is True
    anti_pattern_ids = _canonical_anti_pattern_ids(candidate)
    anti_pattern_penalty = round(_anti_pattern_acquisition_penalty(anti_pattern_ids), 6)
    expected_sha = str(candidate.get("sha256") or "").strip() or None
    expected_bytes = candidate.get("bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool):
        expected_bytes = None
    file_custody = _file_custody(
        path_text=str(candidate.get("path") or ""),
        expected_sha256=expected_sha,
        expected_bytes=expected_bytes,
        repo_root=repo_root,
    )
    source_bytes = _safe_int(
        candidate.get("source_archive_bytes") or source.get("bytes"),
        default=0,
    )
    candidate_bytes = _safe_int(
        file_custody.get("bytes") or candidate.get("bytes"),
        default=0,
    )
    byte_delta_vs_source = (
        candidate_bytes - source_bytes if source_bytes and candidate_bytes else None
    )
    saved_bytes = (
        source_bytes - candidate_bytes
        if source_bytes and candidate_bytes
        else _safe_int(candidate.get("saved_bytes"))
    )
    byte_credit_exhausted = (
        byte_credit_budget is not None
        and candidate_bytes > 0
        and candidate_bytes > max(0, byte_credit_budget)
    )
    blockers = ordered_unique(
        [
            *_string_list(candidate.get("blockers")),
            *_string_list(file_custody.get("blockers")),
            *([] if materialized else ["archive_bound_candidate_not_materialized"]),
            *([] if proof_ready else ["archive_bound_receiver_runtime_proof_missing"]),
            *(
                []
                if receiver_satisfied
                else ["archive_bound_receiver_contract_not_satisfied"]
            ),
            *(
                []
                if not byte_credit_exhausted
                else ["archive_bound_candidate_byte_credit_exhausted"]
            ),
            "contest_cpu_or_cuda_exact_axis_payload_required",
            "lane_dispatch_claim_required_before_exact_eval",
        ]
    )
    archive_bound_ready = bool(
        materialized
        and proof_ready
        and receiver_satisfied
        and file_custody.get("custody_complete") is True
    )
    acquisition_penalty = _contract_penalty(
        blockers=blockers,
        anti_pattern_ids=anti_pattern_ids,
        materialized=materialized,
        receiver_proof_ready=proof_ready,
        receiver_contract_satisfied=receiver_satisfied,
        byte_credit_exhausted=byte_credit_exhausted,
        prototype_only=prototype_only,
        probe_only=probe_only,
    )
    identity = {
        "schema": "tac_archive_bound_candidate_contract_identity.v1",
        "family_id": family_id,
        "typed_response_id": typed_response_id,
        "candidate_chain_id": candidate_chain_id,
        "archive_native_transform_kind": transform_kind,
        "candidate_archive_sha256": file_custody.get("sha256")
        or candidate.get("sha256"),
        "runtime_consumption_proof_path": candidate.get(
            "runtime_consumption_proof_path"
        ),
    }
    contract = {
        "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
        "contract_key": _stable_sha256(identity),
        "contract_identity": identity,
        "family_id": family_id,
        "typed_response_id": typed_response_id,
        "candidate_chain_id": candidate_chain_id,
        "entropy_position_label": resolved_entropy_position_label,
        "entropy_stage_order": entropy_stage_order,
        "archive_native_transform_kind": transform_kind,
        "archive_substrate_tags": archive_substrate_tags_for_transform_kind(
            transform_kind
        ),
        "canonical_anti_pattern_ids": anti_pattern_ids,
        "anti_pattern_protection_count": len(anti_pattern_ids),
        "anti_pattern_acquisition_penalty": anti_pattern_penalty,
        "selected_archive_transform_variant": selected,
        "archive_bound_candidate_ready": archive_bound_ready,
        "archive_bound_candidate_ready_for_exact_handoff": archive_bound_ready,
        "byte_closed_candidate_materialized": materialized,
        "candidate_archive_materialized": materialized,
        "runtime_consumption_proof_ready": proof_ready,
        "receiver_contract_kind": candidate.get("receiver_contract_kind"),
        "receiver_contract_satisfied": receiver_satisfied,
        "runtime_adapter_manifest": dict(
            _mapping(candidate.get("runtime_adapter_manifest"))
        ),
        "runtime_adapter_ready": candidate.get("runtime_adapter_ready") is True
        or _mapping(candidate.get("runtime_adapter_manifest")).get(
            "runtime_adapter_ready"
        )
        is True,
        "contest_runtime_decoder_adapter_ready": (
            candidate.get("contest_runtime_decoder_adapter_ready") is True
            or _mapping(candidate.get("runtime_adapter_manifest")).get(
                "contest_runtime_decoder_adapter_ready"
            )
            is True
        ),
        "archive_file_custody": file_custody,
        "candidate_archive": {
            "path": file_custody.get("path") or candidate.get("path"),
            "sha256": file_custody.get("sha256") or candidate.get("sha256"),
            "bytes": file_custody.get("bytes") or candidate.get("bytes"),
        },
        "source_archive": {
            "path": candidate.get("source_archive_path") or source.get("path"),
            "sha256": candidate.get("source_archive_sha256") or source.get("sha256"),
            "bytes": candidate.get("source_archive_bytes") or source.get("bytes"),
        },
        "runtime_consumption_proof_path": candidate.get(
            "runtime_consumption_proof_path"
        ),
        "semantic_payload_changed_observed": candidate.get("semantic_payload_changed")
        is True,
        "score_affecting_payload_changed_observed": (
            candidate.get("score_affecting_payload_changed") is True
        ),
        "exact_axis_score_affecting_adjudication_required_observed": (
            candidate.get("exact_axis_score_affecting_adjudication_required") is True
        ),
        "charged_bits_changed_observed": candidate.get("charged_bits_changed") is True,
        "prototype_only": prototype_only,
        "probe_only_entropy_signal": probe_only,
        "saved_bytes": saved_bytes,
        "byte_delta_vs_source": byte_delta_vs_source,
        "byte_credit_budget": byte_credit_budget,
        "byte_credit_exhausted": byte_credit_exhausted,
        "estimated_zero_order_savings_bytes": _safe_int(
            candidate.get("estimated_zero_order_savings_bytes")
        ),
        "acquisition_penalty": acquisition_penalty,
        "acquisition_score_hint": round(
            (1.0 if archive_bound_ready else 0.25 if materialized else 0.05)
            * (1.0 - acquisition_penalty)
            + max(0, saved_bytes) / 1_000_000.0,
            9,
        ),
        "blockers": blockers,
        "allowed_use": "archive_bound_candidate_acquisition_and_exact_handoff_planning_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        contract,
        context=f"archive_bound_candidate_contract:{transform_kind}",
    )
    return contract


def build_archive_bound_candidate_contract_surface(
    *,
    candidates: Sequence[Mapping[str, Any]],
    selected_transform_kind: str,
    repo_root: str | Path | None = None,
    source_context: Mapping[str, Any] | None = None,
    family_id: str | None = None,
    typed_response_id: str | None = None,
    candidate_chain_id: str | None = None,
    entropy_position_label: str | None = None,
    entropy_stage_order: int | None = None,
    byte_credit_budget: int | None = None,
) -> dict[str, Any]:
    """Build acquisition-ready contracts for every candidate variant."""

    contracts = [
        build_archive_bound_candidate_contract(
            candidate=candidate,
            repo_root=repo_root,
            selected=str(candidate.get("archive_native_transform_kind") or "")
            == selected_transform_kind,
            source_context=source_context,
            family_id=family_id,
            typed_response_id=typed_response_id,
            candidate_chain_id=candidate_chain_id,
            entropy_position_label=entropy_position_label,
            entropy_stage_order=entropy_stage_order,
            byte_credit_budget=byte_credit_budget,
        )
        for candidate in candidates
    ]
    selected_contracts = [
        contract
        for contract in contracts
        if contract.get("selected_archive_transform_variant") is True
    ]
    acquisition_sorted = sorted(
        contracts,
        key=lambda contract: (
            contract.get("archive_bound_candidate_ready") is not True,
            _safe_float(contract.get("acquisition_penalty")),
            -_safe_float(contract.get("acquisition_score_hint")),
            _safe_int(
                _mapping(contract.get("candidate_archive")).get("bytes"), default=10**18
            ),
            str(contract.get("archive_native_transform_kind") or ""),
        ),
    )
    selected = selected_contracts[0] if selected_contracts else {}
    best = acquisition_sorted[0] if acquisition_sorted else {}
    surface = {
        "schema": ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
        "selected_archive_transform_kind": selected_transform_kind,
        "candidate_contract_count": len(contracts),
        "archive_bound_ready_contract_count": sum(
            1
            for contract in contracts
            if contract.get("archive_bound_candidate_ready") is True
        ),
        "runtime_adapter_ready_contract_count": sum(
            1 for contract in contracts if contract.get("runtime_adapter_ready") is True
        ),
        "receiver_contract_satisfied_count": sum(
            1
            for contract in contracts
            if contract.get("receiver_contract_satisfied") is True
        ),
        "probe_only_contract_count": sum(
            1
            for contract in contracts
            if contract.get("probe_only_entropy_signal") is True
        ),
        "prototype_contract_count": sum(
            1 for contract in contracts if contract.get("prototype_only") is True
        ),
        "archive_substrate_tags": ordered_unique(
            tag
            for contract in contracts
            for tag in _string_list(contract.get("archive_substrate_tags"))
        ),
        "canonical_anti_pattern_ids": ordered_unique(
            anti_pattern_id
            for contract in contracts
            for anti_pattern_id in _string_list(
                contract.get("canonical_anti_pattern_ids")
            )
        ),
        "anti_pattern_protection_count": sum(
            _safe_int(contract.get("anti_pattern_protection_count"))
            for contract in contracts
        ),
        "anti_pattern_acquisition_penalty_sum": round(
            sum(
                _safe_float(contract.get("anti_pattern_acquisition_penalty"))
                for contract in contracts
            ),
            6,
        ),
        "selected_candidate_contract": dict(selected),
        "best_acquisition_contract": dict(best),
        "candidate_contracts": contracts,
        "candidate_contracts_sha256": _stable_sha256(
            {
                "schema": "tac_archive_bound_candidate_contract_surface_hash.v1",
                "candidate_contracts": contracts,
            }
        ),
        "acquisition_penalty_sum": round(
            sum(
                _safe_float(contract.get("acquisition_penalty"))
                for contract in contracts
            ),
            6,
        ),
        "blockers": ordered_unique(
            blocker
            for contract in contracts
            for blocker in _string_list(contract.get("blockers"))
        ),
        "allowed_use": "archive_bound_candidate_contract_acquisition_surface_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        surface,
        context="archive_bound_candidate_contract_surface",
    )
    return surface


__all__ = [
    "ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_CONTRACT_PAYLOAD_KEYS",
    "ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA",
    "ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA",
    "ArchiveBoundCandidateContractError",
    "archive_bound_candidate_contract_fields_for_row",
    "archive_bound_candidate_contract_stale_field_blockers",
    "archive_bound_candidate_contracts_from_payload",
    "archive_substrate_tags_for_transform_kind",
    "build_archive_bound_candidate_contract",
    "build_archive_bound_candidate_contract_surface",
    "entropy_position_label_for_transform_kind",
    "has_archive_bound_candidate_contract_payload",
    "require_fresh_archive_bound_candidate_contract_row",
    "selected_archive_bound_candidate_contract_from_payload",
    "source_archive_bound_candidate_contract_from_row",
    "source_archive_bound_contract_candidate_archive",
    "source_archive_bound_contract_snapshot_blockers",
]
