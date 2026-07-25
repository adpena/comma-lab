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
        if manifest.get("schema") == "tac.applied_action_adapter_manifest.v1":
            raw_results = manifest.get("results")
            if isinstance(raw_results, list):
                blocked_results = [
                    {
                        "source_kind": row.get("source_kind"),
                        "source_id": row.get("source_id"),
                        "blockers": row.get("blockers"),
                    }
                    for row in raw_results
                    if isinstance(row, Mapping) and row.get("ok") is False
                ]
        sources.append(
            {
                **_input_identity(resolved),
                "schema": manifest.get("schema"),
                "declared_content_sha256": manifest.get("content_sha256"),
                "receipt_count": len(parsed),
                "blocked_results": blocked_results,
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
    plan["legacy_real_evidence"] = {
        "cc3": {
            "schema": cc3["schema"],
            "source_archive": dict(cc3_source),
            "candidate_archive": dict(cc3_candidate),
            "measured_delta_bytes": int(cc3_candidate["bytes"]) - int(cc3_source["bytes"]),
            "allocation_authority": False,
            "blocker": "NO_SEVEN_HOME_APPLIED_ACTION_RECEIPT_FOREIGN_KEYS",
        },
        "e5a": {
            "schema": e5a["schema"],
            "archive": {"bytes": e5a_archive.get("bytes"), "sha256": e5a_archive.get("sha256")},
            "packed_minus_unpacked_bytes": e5a_rate.get("packed_minus_unpacked_bytes"),
            "la1_component_count": len(e5a_framing.get("components") or ()),
            "receiver_closed": e5a_framing.get("receiver_closed"),
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
