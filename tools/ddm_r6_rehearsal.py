#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only R6 rehearsal and copied-checkpoint resume audit for DDM campaigns.

The source campaign directory is sacred.  This tool first snapshots it to the
governed SSD proof tier, then performs all inspection and the launcher
``--resume-proof`` against that copy.  It never invokes an n600 scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from costate_digest import (  # noqa: E402
    discover_latest_ddm_campaign_run,
    read_ddm_campaign_observability,
)

from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)

SCHEMA = "ddm_r6_campaign_rehearsal_receipt.v1"
DEFAULT_PROOF_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ct1_campaign_telemetry_encode_20260725"
)


class RehearsalError(ValueError):
    """The copied-run rehearsal failed before it could emit a typed receipt."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": stat.st_size,
                # RFC 8785/I-JSON integers stop at 2**53-1; nanosecond mtimes
                # are custody labels, not arithmetic, so serialize them exactly.
                "mtime_ns": str(stat.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
        )
    canonical = rfc8785_canonicalize({"files": rows})
    return {
        "file_count": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": rows,
    }


def _fingerprint_summary(fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_count": fingerprint["file_count"],
        "bytes": fingerprint["bytes"],
        "tree_sha256": fingerprint["tree_sha256"],
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = rfc8785_canonicalize(payload) + b"\n"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _ticket_from_launch_manifest(run_dir: Path) -> Path:
    launch = json.loads((run_dir / "launch_manifest.json").read_bytes())
    argv = launch.get("argv")
    if not isinstance(argv, list) or "--ticket" not in argv:
        raise RehearsalError("launch manifest lacks typed --ticket argv")
    index = argv.index("--ticket") + 1
    if index >= len(argv):
        raise RehearsalError("launch manifest has empty --ticket argv")
    ticket = Path(str(argv[index]))
    if not ticket.is_absolute():
        ticket = (REPO / ticket).resolve()
        ticket.relative_to(REPO)
    if not ticket.is_file():
        raise RehearsalError(f"typed ticket is absent: {ticket}")
    return ticket


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    with np.load(checkpoint, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise RehearsalError("checkpoint lacks canonical metadata")
        payload = json.loads(
            np.asarray(archive["metadata"], dtype=np.uint8).tobytes()
        )
        member_names = sorted(archive.files)
    if not isinstance(payload, dict):
        raise RehearsalError("checkpoint metadata is not a JSON object")
    payload["_npz_members"] = member_names
    return payload


def _copy_source_snapshot(source: Path, destination: Path) -> dict[str, Any]:
    source_fingerprint = _tree_fingerprint(source)
    if destination.exists():
        copied_fingerprint = _tree_fingerprint(destination)
        if copied_fingerprint["tree_sha256"] != source_fingerprint["tree_sha256"]:
            raise RehearsalError(
                "existing source-run copy differs; choose a fresh proof root"
            )
        status = "REUSED_BIT_IDENTICAL_COPY"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        copied_fingerprint = _tree_fingerprint(destination)
        if copied_fingerprint["tree_sha256"] != source_fingerprint["tree_sha256"]:
            raise RehearsalError("source-run copy is not bit-identical")
        status = "COPIED_BIT_IDENTICAL_BEFORE_REHEARSAL"
    return {
        "status": status,
        "source": _fingerprint_summary(source_fingerprint),
        "copy": _fingerprint_summary(copied_fingerprint),
    }


def _run_resume_proof(
    *,
    python_executable: Path,
    ticket: Path,
    memory_receipt: Path,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not memory_receipt.is_file():
        raise RehearsalError("copied sealed admission-memory receipt is absent")
    argv = [
        str(python_executable),
        "tools/launch_ddm_joint_descent.py",
        "--ticket",
        str(ticket),
        "--out-dir",
        str(output_dir),
        "--memory-receipt",
        str(memory_receipt),
        "--resume-proof",
        "--resume-from",
        str(checkpoint),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt_path = output_dir / "process_boundary_resume_proof.json"
    receipt = (
        json.loads(receipt_path.read_bytes())
        if receipt_path.is_file()
        else None
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "stderr_tail": completed.stderr[-2000:],
        "receipt_path": str(receipt_path),
        "receipt": receipt,
        "status": (
            "PASS_FRESH_PROCESS_COPIED_CHECKPOINT"
            if completed.returncode == 0
            and isinstance(receipt, dict)
            and receipt.get("verdict") == "FRESH_PROCESS_RESUME_PROOF_GREEN"
            else "FAIL_FRESH_PROCESS_COPIED_CHECKPOINT"
        ),
    }


def _copy_sealed_admission_memory_receipt(
    copied_run: Path,
    destination: Path,
) -> dict[str, Any]:
    full_receipt = json.loads((copied_run / "full_run_receipt.json").read_bytes())
    binding = full_receipt.get("admission_memory_receipt")
    if not isinstance(binding, dict):
        raise RehearsalError("full-run receipt lacks admission-memory binding")
    source = Path(str(binding.get("path")))
    expected_sha256 = str(binding.get("sha256"))
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise RehearsalError("sealed admission-memory receipt custody differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != expected_sha256:
            raise RehearsalError("existing admission-memory copy differs")
        status = "REUSED_BIT_IDENTICAL_COPY"
    else:
        shutil.copy2(source, destination)
        if _sha256_file(destination) != expected_sha256:
            raise RehearsalError("admission-memory copy differs")
        status = "COPIED_BIT_IDENTICAL"
    return {
        "status": status,
        "source_path": str(source),
        "copy_path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": expected_sha256,
    }


def rehearse(
    *,
    receipt_path: Path,
    proof_root: Path = DEFAULT_PROOF_ROOT,
    python_executable: Path = Path(sys.executable),
) -> dict[str, Any]:
    run_dir = discover_latest_ddm_campaign_run()
    if run_dir is None:
        raise RehearsalError("no canonical DDM campaign run found")
    before = _tree_fingerprint(run_dir)
    observability = read_ddm_campaign_observability(run_dir)
    copied_run = proof_root / "source_run_copy"
    copy_receipt = _copy_source_snapshot(run_dir, copied_run)

    accepted = sorted((copied_run / "checkpoints").glob("*_accepted_global*.npz"))
    if not accepted:
        raise RehearsalError("copied campaign has no accepted-step checkpoint")
    checkpoint = accepted[-1]
    metadata = _checkpoint_metadata(checkpoint)
    if metadata.get("schema") != "ddm_joint_descent_stage_checkpoint.v1":
        raise RehearsalError("latest accepted checkpoint schema differs")
    ticket = _ticket_from_launch_manifest(copied_run)
    memory_copy = _copy_sealed_admission_memory_receipt(
        copied_run,
        proof_root / "sealed_admission_memory_receipt_copy.json",
    )

    # E5 accepts a receiver-closed WS1 archive plus exactly two typed grammar
    # streams.  The campaign checkpoint contains optimizer/cursor state and only
    # an archive identity, not the realized archive bytes themselves.  Crossing
    # that boundary would fabricate a state, so R6 refuses before export.
    realized_archive = metadata.get("realized_archive") or {}
    checkpoint_has_realized_archive_bytes = any(
        name in metadata["_npz_members"]
        for name in ("archive", "archive_bytes", "realized_archive")
    )
    exporter_blocker = (
        "R6_BLOCKED_E5_MIDCAMP_CHECKPOINT_ADAPTER_ABSENT"
        if not checkpoint_has_realized_archive_bytes
        else None
    )
    resume_proof = _run_resume_proof(
        python_executable=python_executable,
        ticket=ticket,
        memory_receipt=Path(memory_copy["copy_path"]),
        checkpoint=checkpoint,
        output_dir=proof_root / "resume_proof_output",
    )
    after = _tree_fingerprint(run_dir)
    source_unchanged = before["tree_sha256"] == after["tree_sha256"]
    copied_after = _tree_fingerprint(copied_run)
    copy_unchanged = (
        copy_receipt["copy"]["tree_sha256"] == copied_after["tree_sha256"]
    )

    eta = next(
        row
        for row in observability["rows"]
        if row["row_id"] == "schedule_endpoint_eta"
    )
    result = {
        "schema": SCHEMA,
        "status": (
            "BLOCKED_TYPED_EXPORTER_ADAPTER_WITH_RESUME_PROOF_GREEN"
            if exporter_blocker
            and resume_proof["status"] == "PASS_FRESH_PROCESS_COPIED_CHECKPOINT"
            and source_unchanged
            and copy_unchanged
            else "FAIL_REHEARSAL_INVARIANT"
        ),
        "source_campaign": {
            "path": str(run_dir),
            "before": _fingerprint_summary(before),
            "after": _fingerprint_summary(after),
            "bit_identical_before_after": source_unchanged,
            "mutated_by_rehearsal": False if source_unchanged else None,
        },
        "source_copy": {
            **copy_receipt,
            "path": str(copied_run),
            "post_rehearsal": _fingerprint_summary(copied_after),
            "bit_identical_after_rehearsal": copy_unchanged,
            "liveness_authority": False,
        },
        "latest_accepted_checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
            "schema": metadata.get("schema"),
            "global_step": int(metadata.get("step", -1)),
            "npz_members": metadata["_npz_members"],
            "realized_archive_identity": realized_archive,
        },
        "resume_meta_audit": resume_proof,
        "sealed_admission_memory_receipt_copy": memory_copy,
        "r6_pipeline": {
            "lead_time_signal": {
                "epistemic_status": eta["epistemic_status"],
                "eta_hours": eta["eta_hours"],
                "bucket": "approximately_35h",
                "counterfactual_after_governed_stop": eta[
                    "counterfactual_after_governed_stop"
                ],
            },
            "steps": [
                {
                    "step": "copy_latest_accepted_checkpoint_out_of_source_run",
                    "status": "PASS_BIT_IDENTICAL",
                },
                {
                    "step": "adapt_checkpoint_to_e5_ws1_archive_input",
                    "status": exporter_blocker,
                },
                {
                    "step": "e5_export_archive_bytes",
                    "status": "NOT_RUN_UPSTREAM_TYPED_BLOCKER",
                },
                {
                    "step": "archive_parse_back",
                    "status": "NOT_RUN_UPSTREAM_TYPED_BLOCKER",
                },
                {
                    "step": "ic1_ic2_local_exact",
                    "status": "NOT_RUN_UPSTREAM_TYPED_BLOCKER",
                    "d_seg": None,
                    "d_pose": None,
                    "archive_bytes": None,
                },
            ],
            "exporter_adapter": {
                "verdict": exporter_blocker,
                "accepted_input_contract": (
                    "receiver_closed_ws1_archive_bytes_plus_exactly_two_typed_streams"
                ),
                "observed_input_contract": (
                    "ddm_joint_descent_stage_checkpoint_npz_optimizer_cursor_and_archive_identity"
                ),
                "checkpoint_has_realized_archive_bytes": (
                    checkpoint_has_realized_archive_bytes
                ),
                "verdict_scope": (
                    "E5 mid-campaign checkpoint adapter only; high first-cut bytes "
                    "remain an optimization-ladder measurement, not a family rejection"
                ),
            },
        },
        "observability_digest": observability,
        "actuation": "NONE",
        "campaign_launched": False,
        "n600_scorer_invoked": False,
        "execution_allowed": False,
        "research_only": True,
        "score_claim": False,
        "main_landing_review_required": True,
    }
    _atomic_json(receipt_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--proof-root", type=Path, default=DEFAULT_PROOF_ROOT)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
    )
    args = parser.parse_args(argv)
    result = rehearse(
        receipt_path=args.receipt,
        proof_root=args.proof_root,
        python_executable=args.python_executable,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"].startswith("BLOCKED_TYPED_") else 4


if __name__ == "__main__":
    raise SystemExit(main())
