#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a parse-back-stable research-only EV2 seven-home allocation plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.seven_home_stream_allocator import (  # noqa: E402
    ADAPTER_MANIFEST_SCHEMA,
    RECEIPT_MANIFEST_SCHEMA,
    ReceiptEnvelope,
    SevenHomeAllocationError,
    build_allocation_plan,
    envelopes_from_manifest,
)

DEFAULT_EV2 = Path(".omx/research/ddm_ev2_per_pair_allocation_20260725T041933Z/allocation_table.json")
DEFAULT_POINTER = Path(".omx/state/canonical_frontier_pointer.json")
DEFAULT_CC3 = Path(".omx/research/ddm_cc3_mixed_coder_receiver_receipt_20260725.json")
DEFAULT_E5A = Path(
    ".omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/"
    "ddm_e5a_midcampaign_runtime_export_receipt.json"
)


def _repo_path(path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    if not resolved.is_relative_to(REPO):
        raise SevenHomeAllocationError(f"path must stay inside repository: {path}")
    return resolved


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SevenHomeAllocationError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SevenHomeAllocationError(f"JSON root must be an object: {path}")
    return value


def _input_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _exact_int(value: Any, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SevenHomeAllocationError(f"{name} must be an exact integer")
    if nonnegative and value < 0:
        raise SevenHomeAllocationError(f"{name} must be non-negative")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SevenHomeAllocationError(f"{name} must be lowercase SHA-256 hex")
    return value


def _receipt_manifest_reconciliation(
    manifest: Mapping[str, Any], parsed: tuple[ReceiptEnvelope, ...]
) -> dict[str, Any]:
    schema = manifest.get("schema")
    if schema == ADAPTER_MANIFEST_SCHEMA:
        results = manifest["results"]
        receipt_count = _exact_int(
            manifest["receipt_count"], "adapter manifest receipt_count", nonnegative=True
        )
        blocked_count = _exact_int(
            manifest["blocked_source_count"],
            "adapter manifest blocked_source_count",
            nonnegative=True,
        )
        return {
            "declared_receipt_count": receipt_count,
            "declared_blocked_source_count": blocked_count,
            "result_count": len(results),
            "parsed_receipt_count": len(parsed),
            "counts_reconciled": (
                receipt_count == len(parsed)
                and receipt_count + blocked_count == len(results)
            ),
            "content_sha256_reconciled": True,
            "false_authority_reconciled": True,
        }
    if schema == RECEIPT_MANIFEST_SCHEMA:
        receipt_count = _exact_int(
            manifest["receipt_count"], "receipt manifest receipt_count", nonnegative=True
        )
        return {
            "declared_receipt_count": receipt_count,
            "parsed_receipt_count": len(parsed),
            "counts_reconciled": receipt_count == len(parsed),
            "content_sha256_reconciled": True,
            "false_authority_reconciled": True,
        }
    return {
        "declared_receipt_count": 1,
        "parsed_receipt_count": len(parsed),
        "counts_reconciled": len(parsed) == 1,
        "content_sha256_reconciled": None,
        "false_authority_reconciled": None,
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() and path.read_bytes() == encoded:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def build_from_paths(
    *,
    ev2_path: Path,
    pointer_path: Path,
    cc3_path: Path,
    e5a_path: Path,
    receipt_manifest_paths: list[Path],
) -> dict[str, Any]:
    resolved_ev2 = _repo_path(ev2_path)
    resolved_pointer = _repo_path(pointer_path)
    resolved_cc3 = _repo_path(cc3_path)
    resolved_e5a = _repo_path(e5a_path)
    ev2 = _load_mapping(resolved_ev2)
    pointer = _load_mapping(resolved_pointer)
    cc3 = _load_mapping(resolved_cc3)
    e5a = _load_mapping(resolved_e5a)
    if cc3.get("schema") != "ddm_cc3_mixed_coder_receiver_integration_mirror.v1":
        raise SevenHomeAllocationError("CC3 receipt schema differs")
    if e5a.get("schema") != "ddm_e5a_midcampaign_runtime_export_receipt.v1":
        raise SevenHomeAllocationError("E5A receipt schema differs")
    envelopes: list[ReceiptEnvelope] = []
    sources: list[dict[str, Any]] = []
    for path in receipt_manifest_paths:
        resolved = _repo_path(path)
        manifest = _load_mapping(resolved)
        parsed = envelopes_from_manifest(manifest)
        envelopes.extend(parsed)
        blocked_results = []
        if manifest.get("schema") == ADAPTER_MANIFEST_SCHEMA:
            blocked_results = [
                {
                    "source_kind": row["source_kind"],
                    "source_id": row["source_id"],
                    "blockers": row["blockers"],
                }
                for row in manifest["results"]
                if row["ok"] is False
            ]
        sources.append(
            {
                **_input_identity(resolved),
                "schema": manifest.get("schema"),
                "declared_content_sha256": manifest.get("content_sha256"),
                "receipt_count": len(parsed),
                "blocked_results": blocked_results,
                "reconciliation": _receipt_manifest_reconciliation(manifest, parsed),
            }
        )
    plan = build_allocation_plan(ev2=ev2, pointer=pointer, envelopes=envelopes)
    plan["input_artifacts"] = {
        "ev2": _input_identity(resolved_ev2),
        "pointer": _input_identity(resolved_pointer),
        "cc3": _input_identity(resolved_cc3),
        "e5a": _input_identity(resolved_e5a),
    }
    cc3_source = cc3.get("source_archive")
    cc3_candidate = cc3.get("archive")
    e5a_archive = e5a.get("archive")
    e5a_rate = e5a.get("rate")
    e5a_framing = e5a.get("la1_framing")
    if not all(
        isinstance(value, Mapping)
        for value in (cc3_source, cc3_candidate, e5a_archive, e5a_rate, e5a_framing)
    ):
        raise SevenHomeAllocationError("CC3/E5A evidence shape differs")
    cc3_source_bytes = _exact_int(
        cc3_source.get("bytes"), "CC3 source archive bytes", nonnegative=True
    )
    cc3_candidate_bytes = _exact_int(
        cc3_candidate.get("bytes"), "CC3 candidate archive bytes", nonnegative=True
    )
    cc3_source_sha256 = _sha256(cc3_source.get("sha256"), "CC3 source archive sha256")
    cc3_candidate_sha256 = _sha256(
        cc3_candidate.get("sha256"), "CC3 candidate archive sha256"
    )
    e5a_archive_bytes = _exact_int(
        e5a_archive.get("bytes"), "E5A archive bytes", nonnegative=True
    )
    e5a_archive_sha256 = _sha256(e5a_archive.get("sha256"), "E5A archive sha256")
    packed_archive_bytes = _exact_int(
        e5a_rate.get("packed_archive_bytes"), "E5A packed archive bytes", nonnegative=True
    )
    unpacked_state_bytes = _exact_int(
        e5a_rate.get("unpacked_state_bytes"), "E5A unpacked state bytes", nonnegative=True
    )
    packed_minus_unpacked = _exact_int(
        e5a_rate.get("packed_minus_unpacked_bytes"), "E5A packed-minus-unpacked bytes"
    )
    complete_packet_bytes = _exact_int(
        e5a_framing.get("complete_packet_bytes"),
        "E5A complete packet bytes",
        nonnegative=True,
    )
    components = e5a_framing.get("components")
    if not isinstance(components, list) or any(
        not isinstance(component, Mapping) for component in components
    ):
        raise SevenHomeAllocationError("E5A framing components must be an array of objects")
    if e5a_framing.get("receiver_closed") is not True:
        raise SevenHomeAllocationError("E5A framing receiver_closed must be true")
    if packed_archive_bytes != e5a_archive_bytes or complete_packet_bytes != e5a_archive_bytes:
        raise SevenHomeAllocationError("E5A packed/archive/framing byte counts do not reconcile")
    if packed_archive_bytes - unpacked_state_bytes != packed_minus_unpacked:
        raise SevenHomeAllocationError("E5A packed-minus-unpacked bytes do not reconcile")
    plan["legacy_real_evidence"] = {
        "cc3": {
            "schema": cc3["schema"],
            "source_archive": {"bytes": cc3_source_bytes, "sha256": cc3_source_sha256},
            "candidate_archive": {
                "bytes": cc3_candidate_bytes,
                "sha256": cc3_candidate_sha256,
            },
            "measured_delta_bytes": cc3_candidate_bytes - cc3_source_bytes,
            "allocation_authority": False,
            "blocker": "NO_SEVEN_HOME_APPLIED_ACTION_RECEIPT_FOREIGN_KEYS",
        },
        "e5a": {
            "schema": e5a["schema"],
            "archive": {"bytes": e5a_archive_bytes, "sha256": e5a_archive_sha256},
            "packed_minus_unpacked_bytes": packed_minus_unpacked,
            "la1_component_count": len(components),
            "receiver_closed": True,
            "allocation_authority": False,
            "blocker": "E5A_COMPONENTS_ARE_NOT_EV2_HOME_APPLIED_ACTION_RECEIPTS",
        },
        "disposition": (
            "MEASURED_LEGACY_RATE_AND_RECEIVER_EVIDENCE; "
            "NOT_SELECTABLE_WITHOUT_IDENTITY_COMPLETE_APPLIED_ACTION_RECEIPTS"
        ),
    }
    plan["input_receipt_manifests"] = sources
    plan["receipt_manifest_count"] = len(sources)
    plan["applied_action_receipt_count"] = len(envelopes)
    if not envelopes:
        plan["interaction_or_commutator_blockers"] = sorted(
            set(plan["interaction_or_commutator_blockers"])
            | {"NO_VALID_APPLIED_ACTION_RECEIPT_ROWS_AVAILABLE"}
        )
    plan.pop("plan_content_sha256", None)
    plan["plan_content_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ev2", type=Path, default=DEFAULT_EV2)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--cc3", type=Path, default=DEFAULT_CC3)
    parser.add_argument("--e5a", type=Path, default=DEFAULT_E5A)
    parser.add_argument(
        "--receipt-manifest",
        type=Path,
        action="append",
        default=[],
        help="repeat for each tac.applied_action_receipt.v1 or strict receipt manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = build_from_paths(
            ev2_path=args.ev2,
            pointer_path=args.pointer,
            cc3_path=args.cc3,
            e5a_path=args.e5a,
            receipt_manifest_paths=args.receipt_manifest,
        )
        output = _repo_path(args.output)
        _write_atomic(output, plan)
    except SevenHomeAllocationError as exc:
        parser.error(str(exc))
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
