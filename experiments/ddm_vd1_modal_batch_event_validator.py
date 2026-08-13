#!/usr/bin/env python3
"""Dispatch one retained, resumable T4 batch validation of EC1 singleton events.

This module is the provider wrapper.  The exact scorer work is delegated to
``ddm_vd1_batch_event_validator_worker.py`` under the upstream cu128 lock
environment.  It deliberately cannot score locally.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import modal

from experiments.modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image
from tac.deploy.modal.auth_eval import (
    ClaimSpec,
    claim_modal_auth_eval_dispatch,
    function_call_id,
    terminal_modal_auth_eval_claim,
    write_spawn_metadata,
)
from tac.deploy.modal.call_id_ledger import (
    register_dispatched_call_id_fail_closed,
    update_call_id_outcome,
)
from tac.deploy.modal.single_flight import assert_modal_single_flight
from tac.repo_io import sha256_file

REPO: Final = Path(__file__).resolve().parents[1]
APP_NAME: Final = "comma-ddm-vd1-event-validator"
LANE_LABEL: Final = "ddm_vd1_modal_batch_event_validator"
AXIS: Final = "[contest-CUDA T4 exact-upstream affected-pair n600 delta]"
CP135_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_ARCHIVE_BYTES: Final = 186_252
EXPECTED_EVENTS: Final = 200
N_PAIRS: Final = 600
POSE_STACK_BUDGET_GLOBAL: Final = 1.3e-7
POSE_STACK_EQUIVALENT_EVENTS: Final = 44

# Measured on the current CP135/JS7 T4 chain.  The fixed charge is the full
# decode+full-scorer wall time and therefore overcharges this affected-pair
# validator.  The event charge is 10x the measured per-pair render+score mean.
CONTEST_LIMIT_SECONDS: Final = 1_800.0
FIXED_CONSERVATIVE_SECONDS: Final = 393.566
FULL_MASTER_RENDER_SECONDS: Final = 33.3
FULL_SCORER_SECONDS: Final = 39.405
EVENT_SAFETY_FACTOR: Final = 10.0
RESERVE_SECONDS: Final = 300.0

DEFAULT_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
DEFAULT_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime"
)
DEFAULT_EVENT_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/"
    "follow_on/realized_acceptance_200"
)
DEFAULT_JO1_ANALYSIS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_jo1_20260812/10_ANALYSIS.json"
)
DEFAULT_OUTPUT: Final = REPO / ".omx/state/ddm_vd1_modal_batch_event_validator"
REMOTE_REPO: Final = Path("/workspace/pact")
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_vd1_batch_event_validator_worker.py"
VOLUME_NAME: Final = "comma-ddm-vd1-event-validator-retained"
VOLUME_ROOT: Final = Path("/ddm_vd1_retained")
COMMIT_PERIOD_SECONDS: Final = 20.0


class VD1Error(RuntimeError):
    """A bundle, custody, dispatch, or recovery invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for SHA-bound transport."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_member(relative: Path) -> bool:
    """Admit source/runtime files and reject generated or metadata residue."""
    parts = relative.parts
    name = relative.name
    return not (
        "__pycache__" in parts
        or name.startswith("._")
        or name.endswith((".pyc", ".pyo"))
        or name == ".DS_Store"
        or ".git" in parts
        or name == "archive.zip"
    )


def _zip_bytes(files: list[tuple[str, bytes, int]]) -> bytes:
    """Build a deterministic, executable-bit-preserving ZIP."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload, mode in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (mode & 0o777) << 16
            archive.writestr(info, payload)
    return stream.getvalue()


def build_runtime_bundle(runtime_root: Path) -> tuple[bytes, dict[str, Any]]:
    root = runtime_root.resolve()
    if not root.is_dir():
        raise VD1Error(f"missing CP135 adapted runtime: {root}")
    files: list[tuple[str, bytes, int]] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not _safe_member(relative):
            continue
        payload = path.read_bytes()
        mode = path.stat().st_mode & 0o777
        files.append((relative.as_posix(), payload, mode))
        manifest_rows.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "mode": mode,
            }
        )
    required = {"inflate.sh", "runtime/f26_inflate.py", "cpr1/inflate.py"}
    present = {row["relative_path"] for row in manifest_rows}
    if not required.issubset(present):
        raise VD1Error(f"runtime bundle misses required files: {sorted(required - present)}")
    manifest = {
        "schema": "ddm_vd1_runtime_bundle.v1",
        "source_root": str(root),
        "file_count": len(manifest_rows),
        "files": manifest_rows,
        "excluded_generated_residue": True,
    }
    files.append(("VD1_RUNTIME_MANIFEST.json", canonical_json_bytes(manifest), 0o644))
    bundle = _zip_bytes(files)
    manifest["bundle_bytes"] = len(bundle)
    manifest["bundle_sha256"] = sha256_bytes(bundle)
    return bundle, manifest


def _verify_record(record: dict[str, Any], root: Path) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise VD1Error(f"event payload escapes proposal store: {path}") from exc
    if not path.is_file():
        raise VD1Error(f"missing retained event payload: {path}")
    actual = file_record(path)
    if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
        raise VD1Error(f"event payload receipt differs: {path}")
    return path


def decode_event_header(payload: bytes) -> dict[str, int]:
    if len(payload) < 13 or payload[:8] != b"EC1PROP1":
        raise VD1Error("invalid EC1 event header")
    frame = int.from_bytes(payload[8:10], "little")
    return {
        "pair": frame,
        "source_class": payload[10],
        "target_class": payload[11],
        "event_type_id": payload[12],
    }


def _jo1_order(rows: list[dict[str, Any]], analysis_path: Path) -> list[str]:
    """Return JO1's +3B direct-object rate order, with a total fallback."""
    analysis = json.loads(analysis_path.read_text())
    features = analysis.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_EVENTS:
        raise VD1Error("JO1 analysis does not contain the sealed 200-row feature census")
    store_ids = {str(row["proposal_id"]) for row in rows}

    def key(row: dict[str, Any]) -> tuple[float, int, int]:
        ratio = row.get("bytes_per_projected_robust_flip")
        return (
            math.inf if ratio is None else float(ratio),
            int(row.get("projected_robust_delta_flips", 0)),
            int(row.get("ordinal", EXPECTED_EVENTS)),
        )

    ordered = [str(row["proposal_id"]) for row in sorted(features, key=key)]
    if len(ordered) != EXPECTED_EVENTS or set(ordered) != store_ids:
        raise VD1Error("JO1 +3B rate order differs from the proposal-store ID set")
    return ordered


def build_event_bundle(
    event_store: Path,
    jo1_analysis: Path,
    *,
    k: int,
) -> tuple[bytes, dict[str, Any]]:
    root = event_store.resolve()
    state = json.loads((root / "state.json").read_text())
    if (
        state.get("schema") != "ddm_js5_realized_acceptance_200_store.v1"
        or int(state.get("proposal_count", -1)) != EXPECTED_EVENTS
        or int(state.get("receiver_effective_count", -1)) != EXPECTED_EVENTS
        or bool(state.get("acceptance_tested"))
        or state.get("source_archive_sha256") != CP135_ARCHIVE_SHA256
    ):
        raise VD1Error("EC1 proposal-store state differs from the sealed 200-event source")
    index_path = _verify_record(state["proposal_index"], root)
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_EVENTS or len({str(row["proposal_id"]) for row in rows}) != EXPECTED_EVENTS:
        raise VD1Error("EC1 proposal index is not a unique 200-row census")
    if not 1 <= k <= EXPECTED_EVENTS:
        raise VD1Error(f"K must be in [1,{EXPECTED_EVENTS}], got {k}")
    by_id = {str(row["proposal_id"]): row for row in rows}
    order = [str(row["proposal_id"]) for row in rows]
    selection_mode = "full_200_census"
    if k < EXPECTED_EVENTS:
        order = _jo1_order(rows, jo1_analysis)
        selection_mode = "jo1_plus3B_rate_order"
    selected = [by_id[value] for value in order[:k]]
    zip_rows: list[tuple[str, bytes, int]] = []
    event_rows = []
    for ordinal, row in enumerate(selected):
        proposal_id = str(row["proposal_id"])
        payload_path = _verify_record(row["consumer_payloads"]["event.ec1p"], root)
        payload = payload_path.read_bytes()
        header = decode_event_header(payload)
        if int(row["pair"]) != header["pair"]:
            raise VD1Error(f"event pair differs from proposal receipt: {proposal_id}")
        member = f"events/{ordinal:04d}_{proposal_id}.ec1p"
        zip_rows.append((member, payload, 0o644))
        event_rows.append(
            {
                "ordinal": ordinal,
                "source_ordinal": int(row.get("ordinal", int(proposal_id.split("_")[1]))),
                "proposal_id": proposal_id,
                "pair": header["pair"],
                "source_class": header["source_class"],
                "target_class": header["target_class"],
                "event_type_id": header["event_type_id"],
                "event_type": row.get("event_type"),
                "member": member,
                "payload_bytes": len(payload),
                "payload_sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema": "ddm_vd1_event_bundle.v1",
        "source_archive_sha256": CP135_ARCHIVE_SHA256,
        "proposal_store": str(root),
        "proposal_index_sha256": sha256_file(index_path),
        "available_events": EXPECTED_EVENTS,
        "selected_events": k,
        "selection_mode": selection_mode,
        "jo1_analysis": file_record(jo1_analysis),
        "events": event_rows,
    }
    zip_rows.append(("VD1_EVENT_MANIFEST.json", canonical_json_bytes(manifest), 0o644))
    bundle = _zip_bytes(zip_rows)
    manifest["bundle_bytes"] = len(bundle)
    manifest["bundle_sha256"] = sha256_bytes(bundle)
    return bundle, manifest


def k_arithmetic() -> dict[str, Any]:
    event_seconds = EVENT_SAFETY_FACTOR * (
        (FULL_MASTER_RENDER_SECONDS + FULL_SCORER_SECONDS) / N_PAIRS
    )
    k_max = math.floor(
        (CONTEST_LIMIT_SECONDS - FIXED_CONSERVATIVE_SECONDS - RESERVE_SECONDS) / event_seconds
    )
    projected_200 = FIXED_CONSERVATIVE_SECONDS + EXPECTED_EVENTS * event_seconds
    return {
        "schema": "ddm_vd1_k_arithmetic.v1",
        "limit_seconds": CONTEST_LIMIT_SECONDS,
        "fixed_conservative_seconds": FIXED_CONSERVATIVE_SECONDS,
        "measured_full_master_render_seconds": FULL_MASTER_RENDER_SECONDS,
        "measured_full_scorer_seconds": FULL_SCORER_SECONDS,
        "per_event_safety_factor": EVENT_SAFETY_FACTOR,
        "charged_seconds_per_event": event_seconds,
        "reserve_seconds": RESERVE_SECONDS,
        "k_max_with_reserve": k_max,
        "target_k": EXPECTED_EVENTS,
        "projected_target_seconds_without_reserve": projected_200,
        "projected_target_seconds_with_reserve": projected_200 + RESERVE_SECONDS,
        "full_200_fits": k_max >= EXPECTED_EVENTS,
        "epistemic_status": "DERIVED_FROM_MEASURED_CP135_JS7_T4_COMPONENT_TIMES_NOT_YET_VALIDATOR_MEASURED",
    }


def prepare_request(
    *,
    archive: Path,
    runtime: Path,
    event_store: Path,
    jo1_analysis: Path,
    k: int,
    run_id: str,
    resume_from: str,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    if not resume_from:
        raise VD1Error("--resume-from is mandatory, including for a first launch")
    archive = archive.resolve()
    if file_record(archive)["sha256"] != CP135_ARCHIVE_SHA256 or archive.stat().st_size != CP135_ARCHIVE_BYTES:
        raise VD1Error("base archive differs from exact CP135 custody")
    runtime_bundle, runtime_manifest = build_runtime_bundle(runtime)
    event_bundle, event_manifest = build_event_bundle(event_store, jo1_analysis, k=k)
    archive_bytes = archive.read_bytes()
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO
    )
    request = {
        "schema": "ddm_vd1_modal_request.v1",
        "axis": AXIS,
        "run_id": run_id,
        "resume_from": resume_from,
        "k": k,
        "seed": 1234,
        "source_git_head": git_head,
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": sha256_bytes(git_status),
        "dispatcher_source_sha256": sha256_file(Path(__file__)),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_vd1_batch_event_validator_worker.py"
        ),
        "archive": {"bytes": len(archive_bytes), "sha256": sha256_bytes(archive_bytes)},
        "runtime_bundle": {
            "bytes": len(runtime_bundle),
            "sha256": sha256_bytes(runtime_bundle),
            "manifest": runtime_manifest,
        },
        "event_bundle": {
            "bytes": len(event_bundle),
            "sha256": sha256_bytes(event_bundle),
            "manifest": event_manifest,
        },
        "k_arithmetic": k_arithmetic(),
        "retention_volume": VOLUME_NAME,
        "retention_volume_run_path": str(VOLUME_ROOT / run_id),
        "resume_required": True,
        "per_stage_checkpoints": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    return archive_bytes, runtime_bundle, event_bundle, request


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
validator_image = (
    eval_image.add_local_file(
        "experiments/ddm_vd1_batch_event_validator_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=True,
    )
    .add_local_python_source(  # MODAL_ENTRYPOINT_SELF_MOUNT_OK:include_source=False requires this dispatcher module
        "experiments.ddm_vd1_modal_batch_event_validator",
        copy=True,
    )
)


@app.function(
    image=validator_image,
    gpu="T4",
    timeout=int(CONTEST_LIMIT_SECONDS),
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_validator(
    archive_bytes: bytes,
    runtime_bundle: bytes,
    event_bundle: bytes,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Persist inputs, run the locked worker, and commit while it advances."""
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "archive.zip": archive_bytes,
        "runtime_bundle.zip": runtime_bundle,
        "event_bundle.zip": event_bundle,
        "REQUEST.json": canonical_json_bytes(request),
    }
    for name, payload in inputs.items():
        path = run_root / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256_file(path) != sha256_bytes(payload):
                raise VD1Error(f"resume input differs: {path}")
        else:
            staging = path.with_name(path.name + ".partial")
            staging.write_bytes(payload)
            os.replace(staging, path)
    retained_volume.commit()

    command = [
        f"{UPSTREAM_LOCKED_VENV}/bin/python",
        "-u",
        str(REMOTE_WORKER),
        "--run-root",
        str(run_root),
        "--resume-from",
        str(request["resume_from"]),
    ]
    log_path = run_root / "worker.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(COMMIT_PERIOD_SECONDS)
            retained_volume.commit()
        returncode = process.wait()
    retained_volume.commit()
    final_path = run_root / "FINAL_RESULT.json"
    if returncode or not final_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
        return {
            "schema": "ddm_vd1_modal_return.v1",
            "passed": False,
            "returncode": returncode,
            "axis": AXIS,
            "run_id": run_id,
            "volume_name": VOLUME_NAME,
            "volume_path": str(run_root),
            "worker_log_tail": tail,
            "score_claim": False,
            "promotion_eligible": False,
        }
    result_bytes = final_path.read_bytes()
    return {
        "schema": "ddm_vd1_modal_return.v1",
        "passed": True,
        "returncode": 0,
        "axis": AXIS,
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "volume_download_command": (
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/ ./modal_{run_id}/"
        ),
        "final_result_sha256": sha256_bytes(result_bytes),
        "artifacts": {"VD1_FINAL_RESULT.json": result_bytes},
        "score_claim": False,
        "promotion_eligible": False,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    staging.write_bytes(canonical_json_bytes(value))
    os.replace(staging, path)


def recover(output_dir: Path, *, timeout_seconds: float = 0.0) -> int:
    """Harvest a detached validator call without misclassifying it as a score row."""
    output_dir = output_dir.resolve()
    spawn = json.loads((output_dir / "modal_auth_eval_spawn.json").read_text())
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise VD1Error(f"remote return is not a dict: {type(result).__name__}")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise VD1Error(f"unsafe or non-bytes returned artifact: {name!r}")
        path = output_dir / name
        path.write_bytes(payload)
    _atomic_json(output_dir / "modal_vd1_result.json", result)
    passed = bool(result.get("passed"))
    terminal_status = "harvested" if passed else "failed"
    update_call_id_outcome(
        call_id=call_id,
        status=terminal_status,
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_affected_pair_delta",
        evidence_grade="contest-CUDA T4 exact-upstream affected-pair n600 delta",
        lane_id=str(spawn["lane_id"]),
        label=LANE_LABEL,
        gpu="T4",
        agent=str(spawn["claim_agent"]),
        harvest_result={k: v for k, v in result.items() if k != "worker_log_tail"},
    )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(spawn["lane_id"]),
            instance_job_id=str(spawn["instance_job_id"]),
            agent=str(spawn["claim_agent"]),
            force=True,
        ),
        status=(
            "completed_modal_event_validator_recovered"
            if passed
            else "failed_modal_event_validator_recovered"
        ),
        notes=f"VD1 recovered; call_id={call_id}; result={output_dir / 'modal_vd1_result.json'}",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


@app.local_entrypoint()
def main(
    archive: str = str(DEFAULT_ARCHIVE),
    runtime: str = str(DEFAULT_RUNTIME),
    event_store: str = str(DEFAULT_EVENT_STORE),
    jo1_analysis: str = str(DEFAULT_JO1_ANALYSIS),
    output_dir: str = str(DEFAULT_OUTPUT),
    k: int = EXPECTED_EVENTS,
    run_id: str = "ddm_vd1_20260812",
    resume_from: str = "ddm_vd1_20260812",
    lane_id: str = "ddm_vd1_modal_batch_event_validator",
    instance_job_id: str = "modal:ddm_vd1_20260812",
    claim_agent: str = "codex:ddm_vd1",
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    """Validate custody and dispatch exactly one governed T4 job."""
    if detach and not provider_detach_ack:
        raise SystemExit("FATAL: --detach requires --provider-detach-ack")
    budget = k_arithmetic()
    if int(k) > int(budget["k_max_with_reserve"]):
        raise SystemExit(
            "FATAL: requested K exceeds the conservative 30-minute budget; "
            "use the JO1 +3B ordered top-K at or below k_max_with_reserve"
        )
    archive_bytes, runtime_bundle, event_bundle, request = prepare_request(
        archive=Path(archive),
        runtime=Path(runtime),
        event_store=Path(event_store),
        jo1_analysis=Path(jo1_analysis),
        k=int(k),
        run_id=run_id,
        resume_from=resume_from,
    )
    output = Path(output_dir).resolve()
    _atomic_json(output / "VD1_LOCAL_REQUEST.json", request)
    spec = ClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        notes=(
            "VD1 one-job exact-upstream affected-pair n600 delta validator; "
            f"K={k}; base_sha={CP135_ARCHIVE_SHA256}; retained_volume={VOLUME_NAME}/{run_id}"
        ),
    )
    claim_modal_auth_eval_dispatch(repo_root=REPO, spec=spec, status="active_modal_event_validator_spawning")
    assert_modal_single_flight(label=LANE_LABEL, lane_id=lane_id, repo_root=REPO)
    call_args = (archive_bytes, runtime_bundle, event_bundle, request)
    if detach:
        call = run_validator.spawn(*call_args)
        call_id = function_call_id(call)
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=lane_id,
            label=LANE_LABEL,
            platform="modal",
            gpu="T4",
            expected_axis="contest_cuda_affected_pair_delta",
            recipe="experiments/ddm_vd1_modal_batch_event_validator.py::main",
            max_seconds=int(CONTEST_LIMIT_SECONDS),
            agent=claim_agent,
            archive_sha256=CP135_ARCHIVE_SHA256,
            event_count=int(k),
            volume_name=VOLUME_NAME,
            volume_run_id=run_id,
        )
        write_spawn_metadata(
            out_dir=output,
            tool="experiments/ddm_vd1_modal_batch_event_validator.py",
            app=APP_NAME,
            axis="contest_cuda_affected_pair_delta",
            call_id=call_id,
            local_request=request,
            result_json_name="modal_vd1_result.json",
            recover_tool="experiments/ddm_vd1_modal_batch_event_validator.py recover",
            extra={
                "lane_id": lane_id,
                "instance_job_id": instance_job_id,
                "claim_agent": claim_agent,
                "claim_platform": "modal",
                "volume_name": VOLUME_NAME,
                "volume_run_id": run_id,
            },
        )
        claim_modal_auth_eval_dispatch(
            repo_root=REPO,
            spec=ClaimSpec(
                lane_id=lane_id,
                instance_job_id=instance_job_id,
                agent=claim_agent,
                force=True,
                notes=f"VD1 detached spawn accepted; call_id={call_id}; output={output}",
            ),
            status="active_modal_event_validator_spawned",
        )
        print(f"DISPATCHED call_id={call_id}")
        print(f"RECOVER .venv/bin/python {__file__} recover --output-dir {output}")
        return
    result = run_validator.remote(*call_args)
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        (output / name).write_bytes(payload)
    _atomic_json(output / "modal_vd1_result.json", result)
    if not result.get("passed"):
        raise SystemExit("FATAL: VD1 remote validator failed; inspect modal_vd1_result.json")


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover a detached VD1 Modal validator.")
    sub = parser.add_subparsers(dest="command", required=True)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "recover":
        return recover(args.output_dir, timeout_seconds=args.timeout_seconds)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_main_cli())
