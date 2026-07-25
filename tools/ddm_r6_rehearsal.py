#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only R6 rehearsal and copied-checkpoint E5 export audit for DDM campaigns.

The source campaign directory is sacred.  This tool first snapshots it to the
governed SSD proof tier, then performs all inspection and the launcher
``--resume-proof`` against that copy.  The E5A path materializes the copied
checkpoint's live resume-state shadow and invokes only the existing E5 compiler.
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
from remeasure_ddm_e4_ws1_packet import (  # noqa: E402
    DDME5APacketRemeasureConfigV1,
)
from remeasure_ddm_e4_ws1_packet import (  # noqa: E402
    remeasure as remeasure_e5a_packet,
)

from tac.optimization.ddm_e5a_midcampaign_adapter import (  # noqa: E402
    DDME5AMidcampaignCheckpointAdapterConfigV1,
    materialize_midcampaign_checkpoint,
)
from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    DDME5AMidcampaignRuntimeExporterConfigV1,
    derive_ws1_grammar_stream_manifest,
    export_ws1_runtime,
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


def _materialize_decoded_raw(
    *,
    stage_root: Path,
    destination: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Concatenate preserved E5 output stages without rerendering them."""

    stages = sorted(stage_root.glob("pairs_*.raw"))
    if len(stages) != 19:
        raise RehearsalError("E5 output identity lacks 19 preserved raw stages")
    expected_names = [
        f"pairs_{start:04d}_{min(start + 32, 600):04d}.raw"
        for start in range(0, 600, 32)
    ]
    if [path.name for path in stages] != expected_names:
        raise RehearsalError("E5 output stage order differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as output:
            for stage in stages:
                state_path = stage.with_suffix(".json")
                state = json.loads(state_path.read_bytes())
                if (
                    state.get("bytes") != stage.stat().st_size
                    or state.get("sha256") != _sha256_file(stage)
                ):
                    raise RehearsalError(
                        f"preserved E5 output stage custody differs: {stage.name}"
                    )
                with stage.open("rb") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    observed = (destination.stat().st_size, _sha256_file(destination))
    if observed != (expected_bytes, expected_sha256):
        raise RehearsalError(
            f"concatenated E5 output identity differs: {observed!r}"
        )
    return {
        "bytes": observed[0],
        "path": str(destination),
        "sha256": observed[1],
        "source_stage_count": len(stages),
        "status": "PASS_CONCATENATED_FROM_PRESERVED_STAGES",
    }


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

    realized_archive = metadata.get("realized_archive") or {}
    if (
        not isinstance(realized_archive, dict)
        or realized_archive.get("parameter_shadow") != "live_resume_state"
        or not isinstance(realized_archive.get("lane_programs_materialized"), bool)
        or not isinstance(realized_archive.get("bytes"), int)
        or not isinstance(realized_archive.get("sha256"), str)
    ):
        raise RehearsalError(
            "checkpoint lacks a typed live-resume realized-archive identity"
        )
    adapter_root = proof_root / "e5a_adapter"
    adapter_config = DDME5AMidcampaignCheckpointAdapterConfigV1(
        ticket_path=str(ticket),
        ticket_sha256=_sha256_file(ticket),
        checkpoint_path=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=_sha256_file(checkpoint),
        expected_stage_id=str(metadata["stage_id"]),
        expected_global_step=int(metadata["step"]),
        expected_lane_programs_materialized=bool(
            realized_archive["lane_programs_materialized"]
        ),
        expected_state_bytes=int(realized_archive["bytes"]),
        expected_state_sha256=str(realized_archive["sha256"]),
        output_state_path=str(
            adapter_root / "W_joint_step50_live.receiver_closed.zip.receipt-bytes"
        ),
        output_receipt_path=str(adapter_root / "receipt.json"),
    )
    adapter = materialize_midcampaign_checkpoint(adapter_config)
    state_path = Path(str(adapter["state"]["path"]))
    state_payload = state_path.read_bytes()
    stream_manifest = derive_ws1_grammar_stream_manifest(state_payload)
    adapter_receipt_path = Path(adapter_config.output_receipt_path)
    exporter_config_path = (
        REPO
        / ".omx/research/configs/ddm_e5a_midcampaign_runtime_export_20260725.json"
    )
    exporter_config = DDME5AMidcampaignRuntimeExporterConfigV1(
        source_archive_path=str(state_path),
        source_archive_bytes=len(state_payload),
        source_archive_sha256=hashlib.sha256(state_payload).hexdigest(),
        stream_manifest=list(stream_manifest),
        state_name=(
            f"W_joint:campaign_global_step_{int(metadata['step']):06d}_live_resume_state"
        ),
        output_directory=(
            ".omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/packet"
        ),
        proof_root=str(proof_root / "e5a_runtime"),
        minimum_free_bytes=8 * 1024 * 1024 * 1024,
        adapter_receipt_path=str(adapter_receipt_path),
        adapter_receipt_sha256=_sha256_file(adapter_receipt_path),
    )
    _atomic_json(
        exporter_config_path,
        exporter_config.model_dump(mode="json", by_alias=True),
    )
    export_result, export_receipt_path = export_ws1_runtime(
        exporter_config,
        config_path=exporter_config_path,
    )
    decoded_raw = _materialize_decoded_raw(
        stage_root=Path(
            str(export_result["output_identity"]["resume"]["checkpoint_root"])
        ),
        destination=proof_root / "e5a_n600" / "decoded_step50_live.raw",
        expected_bytes=int(export_result["output_identity"]["bytes"]),
        expected_sha256=str(export_result["output_identity"]["sha256"]),
    )
    menu1_config_path = (
        REPO / ".omx/research/configs/ddm_menu1_realized_flip_menu_20260723.json"
    )
    remeasure_config_path = (
        REPO / ".omx/research/configs/ddm_e5a_packet_remeasure_20260725.json"
    )
    remeasure_config = DDME5APacketRemeasureConfigV1(
        export_receipt_path=str(export_receipt_path.relative_to(REPO)),
        export_receipt_sha256=_sha256_file(export_receipt_path),
        menu1_config_path=str(menu1_config_path.relative_to(REPO)),
        menu1_config_sha256=_sha256_file(menu1_config_path),
        decoded_raw_path=decoded_raw["path"],
        decoded_raw_sha256=decoded_raw["sha256"],
        checkpoint_root=str(proof_root / "e5a_n600" / "scorer_rows"),
    )
    _atomic_json(
        remeasure_config_path,
        remeasure_config.model_dump(mode="json", by_alias=True),
    )
    remeasure_receipt_path = (
        REPO
        / ".omx/research/ddm_e5a_midcampaign_e5_adapter_20260725/"
        "ddm_e5a_packet_batch32_remeasure.json"
    )
    n600 = remeasure_e5a_packet(
        remeasure_config_path,
        remeasure_receipt_path,
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
            "PASS_E5A_MIDCAMP_CHECKPOINT_TO_PACKET"
            if resume_proof["status"] == "PASS_FRESH_PROCESS_COPIED_CHECKPOINT"
            and source_unchanged
            and copy_unchanged
            and export_result["grammar_admission"][
                "packet_parseback_source_byte_identical"
            ]
            and n600["verdict"]
            == "E5A_STEP50_LIVE_ENDPOINT_EXACT_BATCH32_MEASURED"
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
                    "status": "PASS_RECEIVER_CLOSED_BYTE_IDENTICAL",
                },
                {
                    "step": "e5_export_archive_bytes",
                    "status": "PASS",
                    "archive_bytes": export_result["archive"]["bytes"],
                    "archive_sha256": export_result["archive"]["sha256"],
                },
                {
                    "step": "archive_parse_back",
                    "status": "PASS_SOURCE_BYTE_IDENTICAL",
                },
                {
                    "step": "e5a_local_exact_n600",
                    "status": n600["verdict"],
                    "d_seg": n600["endpoint"]["d_seg"],
                    "d_pose": n600["endpoint"]["d_pose"],
                    "archive_bytes": export_result["archive"]["bytes"],
                },
            ],
            "exporter_adapter": {
                "verdict": "PASS_TYPED_MIDCAMP_CHECKPOINT_ADAPTER",
                "accepted_input_contract": (
                    "receiver_closed_ws1_archive_bytes_plus_exactly_two_typed_streams"
                ),
                "observed_input_contract": (
                    "ddm_joint_descent_stage_checkpoint_npz_optimizer_cursor_and_archive_identity"
                ),
                "adapter_receipt": adapter,
                "export_receipt_path": str(export_receipt_path),
                "export_receipt_sha256": _sha256_file(export_receipt_path),
                "n600_receipt_path": str(remeasure_receipt_path),
                "n600_receipt_sha256": _sha256_file(remeasure_receipt_path),
                "verdict_scope": (
                    "E5 mid-campaign checkpoint adapter only; high first-cut bytes "
                    "remain an optimization-ladder measurement, not a family rejection"
                ),
            },
        },
        "observability_digest": observability,
        "actuation": "NONE",
        "campaign_launched": False,
        "n600_scorer_invoked": True,
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
    return 0 if result["status"].startswith("PASS_") else 4


if __name__ == "__main__":
    raise SystemExit(main())
