#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministically reseal the J7 #366 ticket and fail closed on fake starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    J7_PROGRAM_SHA256,
    J7_W_JOINT_PROGRAM_SHA256,
    J7_W_SEG_PROGRAM_SHA256,
    DirectDescriptionJointDescentTypedConfigV1,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402

AUTHORITY_SHA256 = "8dac31beda848b94b8bd42f43ffd7008cd024fcf916c0a14149307f68085907e"
AUTHORITY_BYTES = 7497
PROGRAM_ID = "ddm_j7_366_pose_gate_history_reseal_n600_seed0"
VERDICT_BATCH = 32
WARM_START_RECEIPT = (
    REPO
    / ".omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json"
)
PROGRAM_SHA_BY_WARM_START = {
    "inherited_v15_control": J7_PROGRAM_SHA256,
    "W_seg": J7_W_SEG_PROGRAM_SHA256,
    "W_joint": J7_W_JOINT_PROGRAM_SHA256,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO,
        text=True,
    ).strip()


def ws1_launchable_archive(candidate_id: str) -> dict[str, Any]:
    """Return exact archive custody or refuse endpoint-only WS1 evidence."""

    receipt = json.loads(WARM_START_RECEIPT.read_bytes())
    row = receipt["archive_custody"][candidate_id]
    required = ("archive_path", "archive_sha256", "archive_bytes")
    missing = [field for field in required if field not in row]
    if missing:
        raise DirectDescriptionError(
            "WS1_START_NOT_LAUNCHABLE_ENDPOINT_ONLY: "
            f"{candidate_id} lacks {','.join(missing)}; "
            "the measured camera transform is neither a receiver-closed archive "
            "nor a live optimizer state"
        )
    path = Path(row["archive_path"])
    if not path.is_file() or path.is_symlink():
        raise DirectDescriptionError(
            f"WS1_START_ARCHIVE_UNAVAILABLE: {candidate_id}: {path}"
        )
    actual_bytes = path.stat().st_size
    actual_sha = _sha256_file(path)
    if actual_bytes != int(row["archive_bytes"]) or actual_sha != row["archive_sha256"]:
        raise DirectDescriptionError(
            f"WS1_START_ARCHIVE_CUSTODY_DIFFERS: {candidate_id}"
        )
    return {
        "kind": "receiver_closed_ws1_archive",
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "receipt_path": str(WARM_START_RECEIPT.relative_to(REPO)),
        "optimizer_state_loadable": False,
    }


def reseal(
    *,
    ticket_path: Path,
    authority_path: Path,
    memory_receipt: Path | None,
    selected_warm_start: str,
) -> dict[str, Any]:
    if _sha256_file(authority_path) != AUTHORITY_SHA256 or authority_path.stat().st_size != AUTHORITY_BYTES:
        raise DirectDescriptionError("J7 delegated authority custody differs")
    ticket = json.loads(ticket_path.read_bytes())
    semantic = ticket["semantic_program"]
    semantic["program_id"] = PROGRAM_ID
    semantic["telemetry"]["verdict_batch"] = VERDICT_BATCH
    semantic["value_provenance"]["verdict_batch"] = (
        "P0 J7 authority: exact n600 frozen CPU scorer verdicts use batch32"
    )
    if selected_warm_start != "inherited_v15_control":
        semantic["warm_start"] = ws1_launchable_archive(selected_warm_start)
    semantic_sha = hashlib.sha256(rfc8785_canonicalize(semantic)).hexdigest()
    expected_semantic_sha = PROGRAM_SHA_BY_WARM_START[selected_warm_start]
    if semantic_sha != expected_semantic_sha:
        raise DirectDescriptionError(
            f"J7 semantic hash differs: {semantic_sha} != {expected_semantic_sha}"
        )
    ticket["authority"].update(
        {
            "delegation_prompt_path": str(authority_path),
            "delegation_prompt_sha256": AUTHORITY_SHA256,
            "delegation_prompt_bytes": AUTHORITY_BYTES,
            "source_commit": _source_commit(),
        }
    )
    ticket["compile_custody"].update(
        {
            "semantic_program_sha256": semantic_sha,
            "claim": "HASH_SEALED_EXECUTABLE_TYPED_J7_POSE_HISTORY_RESEAL",
        }
    )
    sources = ticket["execution_custody"]["source_files"]
    for name, relative in {
        "consumer": "src/tac/optimization/direct_description_joint_descent.py",
        "launcher": "tools/launch_ddm_joint_descent.py",
    }.items():
        path = REPO / relative
        sources[name] = {"path": relative, "sha256": _sha256_file(path)}
    memory = ticket["execution_custody"]["worst_geometry_memory_receipt"]
    if memory_receipt is None:
        memory["sha256"] = None
    else:
        if not memory_receipt.is_file() or memory_receipt.is_symlink():
            raise DirectDescriptionError("J7 memory receipt is unavailable")
        memory.update(
            {"path": str(memory_receipt), "sha256": _sha256_file(memory_receipt)}
        )
    _atomic_json(ticket_path, ticket)
    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    ticket = json.loads(ticket_path.read_bytes())
    ticket["compile_custody"]["typed_config_hash"] = config.typed_config_hash()
    ticket["compile_custody"]["existing_compiler_accepts_schema"] = True
    ticket["compile_custody"]["existing_governed_launcher_accepts_named_config"] = True
    _atomic_json(ticket_path, ticket)
    result = {
        "schema": "ddm_j7_366_ticket_reseal.v1",
        "ticket_path": str(ticket_path),
        "ticket_sha256": _sha256_file(ticket_path),
        "semantic_program_sha256": semantic_sha,
        "typed_config_hash": config.typed_config_hash(),
        "selected_warm_start": selected_warm_start,
        "verdict_batch": config.verdict_batch,
        "memory_receipt_sealed": memory_receipt is not None,
        "source_files": sources,
        "score_claim": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument(
        "--base-ticket",
        type=Path,
        help="Initialize an absent candidate ticket from this sealed J7 ticket.",
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--memory-receipt", type=Path)
    parser.add_argument(
        "--selected-warm-start",
        choices=("inherited_v15_control", "W_seg", "W_joint"),
        required=True,
    )
    args = parser.parse_args()
    try:
        ticket_path = args.ticket.resolve()
        if not ticket_path.exists():
            if args.base_ticket is None:
                raise DirectDescriptionError(
                    "absent candidate ticket requires --base-ticket"
                )
            base_ticket = args.base_ticket.resolve()
            if not base_ticket.is_file() or base_ticket.is_symlink():
                raise DirectDescriptionError(
                    "J7 base ticket is unavailable"
                )
            _atomic_json(ticket_path, json.loads(base_ticket.read_bytes()))
        reseal(
            ticket_path=ticket_path,
            authority_path=args.authority.resolve(),
            memory_receipt=(
                None if args.memory_receipt is None else args.memory_receipt.resolve()
            ),
            selected_warm_start=args.selected_warm_start,
        )
    except DirectDescriptionError as exc:
        print(json.dumps({"verdict": "REFUSE", "reason": str(exc)}), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
