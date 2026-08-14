#!/usr/bin/env python3
"""Seal, dispatch, harvest, and consume the JS1C T4 Stage-0 custody repeat.

The remote leg reuses the already-proven RE1T candidate-only T4 function and
its legacy SegNet-only worker mode.  This file supplies a new immutable T1R1
request and a matched-axis consumer.  The consumer never scores: it operates
only on retained contest-CUDA argmax fields and keeps every field it consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final

import modal
import numpy as np

try:
    from experiments import ddm_js1_stage0_per_edge as stage0
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b
    from experiments import ddm_re1t_modal_t4_sign_gate as re1t
except (ImportError, ModuleNotFoundError):
    import ddm_js1_stage0_per_edge as stage0  # type: ignore[no-redef]
    import ddm_js1b_modal_cuda_argmax_field_materializer as js1b  # type: ignore[no-redef]
    import ddm_re1t_modal_t4_sign_gate as re1t  # type: ignore[no-redef]

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

REPO: Final = Path(__file__).resolve().parents[1]
STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "stage0_per_edge/contest_cuda/ddm_js1c_20260814"
)
RUN_ID: Final = "ddm_js1c_cuda_custody_stage0_20260814_r1"
LANE_ID: Final = "ddm_js1c_cuda_custody_stage0_20260814"
LANE_LABEL: Final = "ddm_js1c_cuda_custody_stage0"
INSTANCE_JOB_ID: Final = f"modal:{RUN_ID}"
CLAIM_AGENT: Final = "main:ddm_js1c"
AXIS: Final = re1t.AXIS

CANDIDATE_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/ddm_t1r1/retained/adapted_runtime"
)
CANDIDATE_ARCHIVE: Final = CANDIDATE_RUNTIME / "archive.zip"
CANDIDATE_ARCHIVE_RECORD: Final = {
    "bytes": 187_046,
    "sha256": "12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80",
}
CANDIDATE_RUNTIME_TREE_SHA256: Final = (
    "3bb8da9ffed161566458dd9bcd5ffc38bb6f7aa7c54b5f102df9f5e31c2e78d4"
)
CANDIDATE_RUNTIME_FILE_COUNT: Final = 25

STAGE0_SOURCE_RECEIPT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "stage0_per_edge/STAGE0_RESULT.json"
)
STAGE0_SOURCE_RECEIPT_RECORD: Final = {
    "bytes": 31_009,
    "sha256": "4247320a2e824e03ed94ac9fbe6a96a97d2b125dadb4fbdfe5e59fd813d7d76f",
}
STAGE0_MEMO: Final = REPO / ".omx/research/ddm_js1_stage0_per_edge_20260812.md"
STAGE0_MEMO_SHA256: Final = (
    "60e6a9113f65b1f1da0a104e6e68c78efceb11d8d525c54dbeb01fd477062448"
)
PRIOR_ADJUDICATION_MEMO: Final = (
    REPO / ".omx/research/ddm_js1b_cuda_custody_adjudication_20260813.md"
)
PRIOR_ADJUDICATION_MEMO_SHA256: Final = (
    "eb1a9efd69585145f3d3d0e0bd827bbb6f378090933be59300aaa733ae414c15"
)
PROVEN_DISPATCHER: Final = REPO / "experiments/ddm_re1t_modal_t4_sign_gate.py"
PROVEN_DISPATCHER_SHA256: Final = (
    "b00f3ffc1eb5e8f4680eb8f301bd5c83921728f0c90d68e90c23799157983ec9"
)
SEAL_MODEL: Final = REPO / "experiments/ddm_re1_pose_leg_seal.py"
SEAL_MODEL_SHA256: Final = (
    "486cb7e92083bee0c1a7cc078654518da689a672129a6390a33e960cddf6ca63"
)
LEGACY_WORKER: Final = (
    REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
)
LEGACY_WORKER_SHA256: Final = (
    "03dc9e81a21409f5881cff642d5dc334a8f04deae5b008f31cd2719bba4a14fb"
)

PRIOR_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "stage0_per_edge/contest_cuda/ddm_js1b_20260813b"
)
PRIOR_FINAL: Final = PRIOR_ROOT / "FINAL_RESULT.json"
PRIOR_FINAL_RECORD: Final = {
    "bytes": 9_600,
    "sha256": "5fd65b946e2e1a5683e123554761c4216f8245a4d1cec46da2ee95b925c93a0c",
}
PRIOR_FIELD_RECORDS: Final = {
    "gt": {
        "bytes": 117_964_928,
        "sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    },
    "cp135_base": {
        "bytes": 117_964_928,
        "sha256": "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727",
    },
    "c1_target": {
        "bytes": 117_964_928,
        "sha256": "a9c4936c41bc6634477f9c060be3d170542bd2a1d4d0cd04d5afcd0912fb3908",
    },
    "prior_t1r1_candidate": {
        "bytes": 117_964_928,
        "sha256": "7c3750d9e44b44fb6cbd8dc9c9907714532d13364bb705c8183f5cff6b9a184a",
    },
}
PRIOR_FIELD_PATHS: Final = {
    "gt": PRIOR_ROOT / "retained/fields/gt_argmax_n600.npy",
    "cp135_base": PRIOR_ROOT / "retained/fields/cp135_base_argmax_n600.npy",
    "c1_target": PRIOR_ROOT / "retained/fields/c1_target_argmax_n600.npy",
    "prior_t1r1_candidate": (
        PRIOR_ROOT / "retained/fields/t1r1_c1_composed_argmax_n600.npy"
    ),
}

SEALED_REQUEST: Final = STORE / "JS1C_T4_REQUEST.json"
SEALED_FIRE_ORDER: Final = STORE / "SEALED_FIRE_ORDER.json"
FIRE_INPUT_DIR: Final = STORE / "fire_inputs"
DISPATCH_OUTPUT: Final = STORE / "dispatch"
RETAINED_FIELDS: Final = STORE / "retained/fields"
DECOMPOSITION_ROOT: Final = STORE / "decomposition"
REMOTE_RESULT: Final = DISPATCH_OUTPUT / "RE1T_T4_REMOTE_RESULT.json"
REMOTE_FIELD_PATH: Final = (
    re1t.VOLUME_ROOT / RUN_ID / "retained/fields/candidate_argmax_n600.npy"
)
LOCAL_CANDIDATE_DOWNLOAD: Final = RETAINED_FIELDS / "candidate_argmax_n600.npy"

RHO_GATE: Final = 0.827795
BASE_FLIPS: Final = 34_970
TARGET_FLIPS: Final = 27_330
TOTAL_PIXELS: Final = 600 * 384 * 512
REMOTE_PAYLOAD_BYTES: Final = 7_555_248_128
REMOTE_RESERVE_BYTES: Final = 4 * 1024**3
LOCAL_RESERVE_BYTES: Final = 512 * 1024**2
EXPECTED_LOCAL_FIELD_BYTES: Final = 4 * 117_964_928
ESTIMATED_COST_USD: Final = 0.16

EDGE_CONDITIONED_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned"
)


class JS1CError(RuntimeError):
    """A JS1C sealing, custody, dispatch, or adjudication invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_record(path: Path, record: dict[str, Any], *, label: str) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise JS1CError(f"{label} is missing: {resolved}")
    if resolved.stat().st_size != int(record["bytes"]):
        raise JS1CError(f"{label} byte count differs: {resolved}")
    if sha256_file(resolved) != str(record["sha256"]):
        raise JS1CError(f"{label} SHA-256 differs: {resolved}")


def atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    """Copy one retained payload without replacing a different existing payload."""
    source_record = js1b.file_record(source.resolve())
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        require_record(destination, source_record, label="existing retained field")
        return js1b.file_record(destination)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        require_record(temporary, source_record, label="temporary retained field")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return js1b.file_record(destination)


def source_records() -> dict[str, dict[str, Any]]:
    records = {
        "stage0_memo": {
            "bytes": STAGE0_MEMO.stat().st_size,
            "sha256": STAGE0_MEMO_SHA256,
        },
        "prior_adjudication_memo": {
            "bytes": PRIOR_ADJUDICATION_MEMO.stat().st_size,
            "sha256": PRIOR_ADJUDICATION_MEMO_SHA256,
        },
        "prior_final": PRIOR_FINAL_RECORD,
        "stage0_source_receipt": STAGE0_SOURCE_RECEIPT_RECORD,
        "proven_re1_dispatcher": {
            "bytes": PROVEN_DISPATCHER.stat().st_size,
            "sha256": PROVEN_DISPATCHER_SHA256,
        },
        "re1_pose_seal_model": {
            "bytes": SEAL_MODEL.stat().st_size,
            "sha256": SEAL_MODEL_SHA256,
        },
        "segnet_legacy_worker": {
            "bytes": LEGACY_WORKER.stat().st_size,
            "sha256": LEGACY_WORKER_SHA256,
        },
    }
    require_record(STAGE0_MEMO, records["stage0_memo"], label="Stage-0 source memo")
    require_record(
        PRIOR_ADJUDICATION_MEMO,
        records["prior_adjudication_memo"],
        label="prior CUDA custody adjudication memo",
    )
    require_record(PRIOR_FINAL, PRIOR_FINAL_RECORD, label="prior T4 field receipt")
    require_record(
        STAGE0_SOURCE_RECEIPT,
        STAGE0_SOURCE_RECEIPT_RECORD,
        label="Stage-0 source receipt",
    )
    require_record(
        PROVEN_DISPATCHER,
        records["proven_re1_dispatcher"],
        label="proven RE1 dispatcher",
    )
    require_record(SEAL_MODEL, records["re1_pose_seal_model"], label="RE1 pose seal model")
    require_record(
        LEGACY_WORKER,
        records["segnet_legacy_worker"],
        label="SegNet-only legacy worker",
    )
    return records


def build_request() -> tuple[dict[str, bytes], dict[str, Any]]:
    """Create the exact, scorer-free, resumable T1R1 request."""
    archive = CANDIDATE_ARCHIVE.resolve()
    runtime = CANDIDATE_RUNTIME.resolve()
    require_record(archive, CANDIDATE_ARCHIVE_RECORD, label="T1R1 candidate archive")
    runtime_tree = re1t.tree_record(runtime)
    if runtime_tree["tree_sha256"] != CANDIDATE_RUNTIME_TREE_SHA256:
        raise JS1CError("T1R1 candidate runtime-tree SHA-256 differs")
    if runtime_tree["file_count"] != CANDIDATE_RUNTIME_FILE_COUNT:
        raise JS1CError("T1R1 candidate runtime file count differs")
    runtime_pin = re1t.verify_runtime_archive_pin(runtime, js1b.file_record(archive))
    runtime_bundle, runtime_manifest = js1b.build_runtime_bundle(
        runtime, label="js1c_t1r1_candidate"
    )
    evidence = STAGE0_SOURCE_RECEIPT.read_bytes()
    payloads = {
        "candidate_archive.zip": archive.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        # The proven worker's Seg-only compatibility slot has this fixed name.
        # The request records its actual JS1 Stage-0 role and exact source path.
        "RE1X_FULL_N600_BLOCKER.json": evidence,
    }
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    k = re1t.k_arithmetic()
    request = {
        "schema": "ddm_js1c_cuda_custody_stage0_request.v1",
        "axis": AXIS,
        "run_id": RUN_ID,
        "resume_from": RUN_ID,
        "lane_id": LANE_ID,
        "instance_job_id": INSTANCE_JOB_ID,
        "claim_agent": CLAIM_AGENT,
        "seed": 1234,
        "batch_size": 16,
        "candidate_archive": js1b.file_record(archive),
        "candidate_runtime": runtime_tree,
        "runtime_archive_pin": runtime_pin,
        "inputs": {name: js1b.payload_record(value) for name, value in payloads.items()},
        "legacy_evidence_slot": {
            "input_name": "RE1X_FULL_N600_BLOCKER.json",
            "semantic_role": "JS1 Stage-0 source receipt; not an RE1X blocker",
            "source": js1b.file_record(STAGE0_SOURCE_RECEIPT),
        },
        "runtime_manifest": runtime_manifest,
        "source_records": source_records(),
        "retained_t4_controls": {
            "volume_run_path": str(re1t.VOLUME_ROOT / "ddm_js1b_20260813b"),
            "gt_field": PRIOR_FIELD_RECORDS["gt"],
            "cp135_base_field": PRIOR_FIELD_RECORDS["cp135_base"],
            "c1_target_field_local_custody": PRIOR_FIELD_RECORDS["c1_target"],
            "base_flips": BASE_FLIPS,
            "target_flips": TARGET_FLIPS,
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "pose_gate_note": (
            "Zero is a placeholder, not a measurement. JS1C is a SegNet-only Stage-0 "
            "component measurement and makes no complete-score or Pose claim."
        ),
        "k_arithmetic": k,
        "storage_preflight_contract": {
            "expected_total_retained_payload_bytes": REMOTE_PAYLOAD_BYTES,
            "reserve_bytes": REMOTE_RESERVE_BYTES,
            "cleanup_policy": "block and retain; never delete generated payloads",
            "resumable_same_run_id": True,
            "per_stage_checkpoints": True,
        },
        "source_git_head": git_head,
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": js1b.sha256_bytes(git_status),
        "seal_source_sha256": sha256_file(Path(__file__)),
        "dispatcher_source_sha256": sha256_file(PROVEN_DISPATCHER),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(LEGACY_WORKER),
        "retention_volume": re1t.VOLUME_NAME,
        "retention_volume_run_path": str(re1t.VOLUME_ROOT / RUN_ID),
        "resume_required": True,
        "per_stage_checkpoints": True,
        "remote_scope": "exact public decode plus n600 frozen T4 SegNet field",
        "local_scope": "matched T4 field-only per-edge Stage-0 decomposition",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    return payloads, request


def validate_request(request: dict[str, Any]) -> None:
    if request.get("schema") != "ddm_js1c_cuda_custody_stage0_request.v1":
        raise JS1CError("sealed request schema differs")
    if request.get("run_id") != RUN_ID or request.get("resume_from") != RUN_ID:
        raise JS1CError("sealed request lost its fresh resumable run identity")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", RUN_ID):
        raise JS1CError("run id is not one safe path component")
    if request.get("lane_id") != LANE_ID or request.get("instance_job_id") != INSTANCE_JOB_ID:
        raise JS1CError("sealed request lane identity differs")
    if request.get("axis") != AXIS or request.get("batch_size") != 16:
        raise JS1CError("sealed request T4 axis differs")
    if request.get("local_pose_delta") != 0.0 or request.get("pose_unmeasured") is not True:
        raise JS1CError("sealed request lost the explicit Pose-unknown placeholder law")
    if any(request.get(key) is not False for key in ("score_claim", "promotion_eligible", "pointer_moved")):
        raise JS1CError("sealed request crossed its component-only authority")
    if request.get("candidate_archive", {}).get("sha256") != CANDIDATE_ARCHIVE_RECORD["sha256"]:
        raise JS1CError("sealed request candidate identity differs")
    if request.get("candidate_runtime", {}).get("tree_sha256") != CANDIDATE_RUNTIME_TREE_SHA256:
        raise JS1CError("sealed request runtime identity differs")
    if request.get("runtime_archive_pin", {}).get("passed") is not True:
        raise JS1CError("sealed request lacks a runtime/archive pin proof")
    if set(request.get("inputs", {})) != {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "RE1X_FULL_N600_BLOCKER.json",
    }:
        raise JS1CError("sealed request input census differs")
    expected_sources = {
        "seal_source_sha256": sha256_file(Path(__file__)),
        "dispatcher_source_sha256": sha256_file(PROVEN_DISPATCHER),
        "worker_source_sha256": sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": sha256_file(LEGACY_WORKER),
    }
    for key, expected in expected_sources.items():
        if request.get(key) != expected:
            raise JS1CError(f"sealed request source drift: {key}")


def load_sealed_inputs(
    sealed_request: Path, fire_input_dir: Path, expected_request_sha256: str
) -> tuple[dict[str, bytes], dict[str, Any]]:
    require_record(
        sealed_request,
        {"bytes": sealed_request.stat().st_size, "sha256": expected_request_sha256},
        label="sealed JS1C request",
    )
    request = json.loads(sealed_request.read_text(encoding="utf-8"))
    validate_request(request)
    payloads: dict[str, bytes] = {}
    for name, record in request["inputs"].items():
        if Path(name).name != name:
            raise JS1CError(f"unsafe fire-input name: {name!r}")
        path = fire_input_dir / name
        require_record(path, record, label="sealed JS1C fire input")
        payloads[name] = path.read_bytes()
    return payloads, request


def fire_command(request_sha256: str) -> list[str]:
    return [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_js1c_cuda_custody_stage0.py::modal_main",
        "--sealed-request",
        str(SEALED_REQUEST),
        "--fire-input-dir",
        str(FIRE_INPUT_DIR),
        "--expected-request-sha256",
        request_sha256,
        "--output-dir",
        str(DISPATCH_OUTPUT),
        "--provider-detach-ack",
    ]


def prepare() -> dict[str, Any]:
    payloads, request = build_request()
    STORE.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(STORE)
    payload_bytes = sum(len(value) for value in payloads.values())
    required_free_bytes = (
        payload_bytes + EXPECTED_LOCAL_FIELD_BYTES + LOCAL_RESERVE_BYTES
    )
    preflight = {
        "schema": "ddm_js1c_local_storage_preflight.v1",
        "tier": str(STORE),
        "free_bytes": usage.free,
        "fire_input_payload_bytes": payload_bytes,
        "expected_retained_field_bytes": EXPECTED_LOCAL_FIELD_BYTES,
        "reserve_bytes": LOCAL_RESERVE_BYTES,
        "required_free_bytes": required_free_bytes,
        "passed": usage.free >= required_free_bytes,
        "cleanup_policy": "certify-or-block; all sealed inputs and harvested fields persist",
    }
    js1b.atomic_json(STORE / "LOCAL_STORAGE_PREFLIGHT.json", preflight)
    if not preflight["passed"]:
        raise JS1CError("local storage preflight failed")
    for name, payload in payloads.items():
        destination = FIRE_INPUT_DIR / name
        if destination.is_file():
            require_record(
                destination, js1b.payload_record(payload), label="existing sealed fire input"
            )
        else:
            js1b.atomic_bytes(destination, payload)
    js1b.atomic_json(SEALED_REQUEST, request)
    request_record = js1b.file_record(SEALED_REQUEST)
    command = fire_command(str(request_record["sha256"]))
    order = {
        "schema": "ddm_js1c_cuda_custody_stage0_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(STORE),
        "fire_trigger": (
            "the sole n600 scorer/Modal lane is clear, every sealed SHA verifies, and MAIN "
            "executes exact_command_argv once"
        ),
        "fresh_run_id": RUN_ID,
        "lane_id": LANE_ID,
        "request": request_record,
        "exact_command_argv": command,
        "exact_command": shlex.join(command),
        "estimated_cost_usd": ESTIMATED_COST_USD,
        "budget_ledger": "#381",
        "remote_retention": {
            "volume": re1t.VOLUME_NAME,
            "run_path": str(re1t.VOLUME_ROOT / RUN_ID),
            "worker_result": str(re1t.VOLUME_ROOT / RUN_ID / "FINAL_RESULT.json"),
            "candidate_raw": str(re1t.VOLUME_ROOT / RUN_ID / "retained/raw/candidate/0.raw"),
            "candidate_field": str(REMOTE_FIELD_PATH),
        },
        "recover_command_argv": [
            ".venv/bin/python",
            str(Path(__file__).relative_to(REPO)),
            "recover",
            "--output-dir",
            str(DISPATCH_OUTPUT),
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }
    js1b.atomic_json(SEALED_FIRE_ORDER, order)
    return {
        "schema": "ddm_js1c_cuda_custody_stage0_prepare_result.v1",
        "status": "READY_TO_FIRE_BY_MAIN",
        "request": request_record,
        "fire_order": js1b.file_record(SEALED_FIRE_ORDER),
        "storage_preflight": preflight,
        "modal_fired": False,
        "score_claim": False,
    }


# The imported app/function are the exact proven RE1T dispatcher surface.
app = re1t.app


@app.local_entrypoint()
def modal_main(
    sealed_request: str,
    fire_input_dir: str,
    expected_request_sha256: str,
    output_dir: str = str(DISPATCH_OUTPUT),
    provider_detach_ack: bool = False,
) -> None:
    """MAIN-only detached entry point for the immutable JS1C request."""
    if not provider_detach_ack:
        raise SystemExit("FATAL: detached provider acknowledgement is required")
    payloads, request = load_sealed_inputs(
        Path(sealed_request), Path(fire_input_dir), expected_request_sha256
    )
    if request.get("k_arithmetic", {}).get("fits_30_minutes") is not True:
        raise SystemExit("FATAL: the sealed one-candidate projection exceeds 30 minutes")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    js1b.atomic_json(output / "REQUEST.json", request)
    for name, payload in payloads.items():
        js1b.atomic_bytes(output / "fire_inputs" / name, payload)
    spec = ClaimSpec(
        lane_id=LANE_ID,
        instance_job_id=INSTANCE_JOB_ID,
        agent=CLAIM_AGENT,
        notes=f"JS1C fresh T1R1 Stage-0 T4 field repeat; volume={re1t.VOLUME_NAME}/{RUN_ID}",
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO, spec=spec, status="active_js1c_t4_measurement_spawning"
    )
    assert_modal_single_flight(label=LANE_LABEL, lane_id=LANE_ID, repo_root=REPO)
    call = re1t.run_gate.spawn(payloads, request)
    call_id = function_call_id(call)
    register_dispatched_call_id_fail_closed(
        call_id=call_id,
        lane_id=LANE_ID,
        label=LANE_LABEL,
        platform="modal",
        gpu="T4",
        expected_axis="contest_cuda_js1c_stage0_argmax_field_component",
        recipe="experiments/ddm_js1c_cuda_custody_stage0.py::modal_main",
        max_seconds=int(re1t.CONTEST_LIMIT_SECONDS),
        agent=CLAIM_AGENT,
        base_archive_sha256=re1t.BASE_SHA256,
        composed_archive_sha256=str(CANDIDATE_ARCHIVE_RECORD["sha256"]),
        archive_count=1,
        volume_name=re1t.VOLUME_NAME,
        volume_run_id=RUN_ID,
    )
    write_spawn_metadata(
        out_dir=output,
        tool="experiments/ddm_js1c_cuda_custody_stage0.py",
        app=re1t.APP_NAME,
        axis="contest_cuda_js1c_stage0_argmax_field_component",
        call_id=call_id,
        local_request=request,
        result_json_name="modal_js1c_t4_result.json",
        recover_tool="experiments/ddm_js1c_cuda_custody_stage0.py recover",
        extra={
            "lane_id": LANE_ID,
            "instance_job_id": INSTANCE_JOB_ID,
            "claim_agent": CLAIM_AGENT,
            "claim_platform": "modal",
            "volume_name": re1t.VOLUME_NAME,
            "volume_run_id": RUN_ID,
        },
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=LANE_ID,
            instance_job_id=INSTANCE_JOB_ID,
            agent=CLAIM_AGENT,
            force=True,
            notes=f"JS1C detached T4 call accepted; call_id={call_id}; output={output}",
        ),
        status="active_js1c_t4_measurement_spawned",
    )
    print(f"DISPATCHED call_id={call_id}")


def recover(output_dir: Path, timeout_seconds: float = 0.0) -> int:
    output = output_dir.resolve()
    spawn = json.loads((output / "modal_auth_eval_spawn.json").read_text(encoding="utf-8"))
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise JS1CError(f"remote return is not a dict: {type(result).__name__}")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise JS1CError(f"unsafe returned artifact: {name!r}")
        js1b.atomic_bytes(output / name, payload)
    js1b.atomic_json(output / "modal_js1c_t4_result.json", result)
    complete = bool(result.get("measurement_complete"))
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if complete else "failed",
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_js1c_stage0_argmax_field_component",
        evidence_grade="contest-CUDA T4 frozen-SegNet argmax field n600 batch16",
        lane_id=LANE_ID,
        label=LANE_LABEL,
        gpu="T4",
        agent=CLAIM_AGENT,
        harvest_result={key: value for key, value in result.items() if key != "worker_log_tail"},
    )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=LANE_ID,
            instance_job_id=INSTANCE_JOB_ID,
            agent=CLAIM_AGENT,
            force=True,
        ),
        status=(
            "completed_js1c_t4_measurement_recovered"
            if complete
            else "failed_js1c_t4_measurement_recovered"
        ),
        notes=f"JS1C T4 result recovered; call_id={call_id}; output={output}",
    )
    download = [
        ".venv/bin/modal",
        "volume",
        "get",
        "--force",
        re1t.VOLUME_NAME,
        str(REMOTE_FIELD_PATH.relative_to(re1t.VOLUME_ROOT)),
        str(RETAINED_FIELDS),
    ]
    harvest = {
        "schema": "ddm_js1c_cuda_custody_stage0_harvest.v1",
        "status": "REMOTE_COMPLETE_FIELD_DOWNLOAD_REQUIRED" if complete else "REMOTE_FAILED",
        "call_id": call_id,
        "remote": result,
        "candidate_field_download_argv": download,
        "candidate_field_download_command": shlex.join(download),
        "download_destination": str(LOCAL_CANDIDATE_DOWNLOAD),
        "consume_argv": [
            ".venv/bin/python",
            str(Path(__file__).relative_to(REPO)),
            "consume",
        ],
        "score_claim": False,
    }
    js1b.atomic_json(output / "HARVEST_REQUEST.json", harvest)
    print(json.dumps(harvest, indent=2, sort_keys=True))
    return 0 if complete else 1


def verify_remote_measurement(
    measurement: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    if measurement.get("execution_status") != "MEASUREMENT_COMPLETE":
        raise JS1CError("remote worker result is incomplete")
    if measurement.get("axis") != AXIS:
        raise JS1CError("remote worker axis differs")
    if measurement.get("candidate_archive") != request["inputs"]["candidate_archive.zip"]:
        raise JS1CError("remote candidate archive differs from the sealed request")
    if any(
        measurement.get(key) is not False
        for key in ("score_claim", "promotion_eligible", "pointer_moved")
    ):
        raise JS1CError("remote result crossed its component-only authority")
    retained = measurement.get("retained_prior_fields", {})
    for name, expected in {
        "gt": PRIOR_FIELD_RECORDS["gt"],
        "cp135_base": PRIOR_FIELD_RECORDS["cp135_base"],
    }.items():
        observed = retained.get(name, {})
        if {key: observed.get(key) for key in ("bytes", "sha256")} != expected:
            raise JS1CError(f"remote retained prior field differs: {name}")
    metrics = measurement.get("field_measurement", {})
    if int(metrics.get("denominator_pixels", -1)) != TOTAL_PIXELS:
        raise JS1CError("remote population denominator differs")
    if int(metrics.get("base_flips_vs_gt", -1)) != BASE_FLIPS:
        raise JS1CError("remote base control differs")
    if metrics.get("adjudicated_remotely") is not False:
        raise JS1CError("remote worker crossed the local-adjudication boundary")
    candidate_record = measurement.get("retention", {}).get("candidate_argmax_field")
    if not isinstance(candidate_record, dict):
        raise JS1CError("remote result lacks the retained candidate field record")
    return candidate_record


def road_hub(summary: dict[str, Any]) -> dict[str, Any]:
    directed = [
        row
        for row in summary["directed_cells"]
        if row["gt_class"] == "Road" or row["rendered_class"] == "Road"
    ]
    undirected = [
        row for row in summary["undirected_edges"] if row["edge"].startswith("Road<->")
    ]
    return {
        "directed_cells": directed,
        "undirected_interfaces": undirected,
        "incident_flips": int(sum(int(row["flips"]) for row in undirected)),
        "incident_share": summary["road_incident_share"],
    }


def stage0_result(
    *,
    summaries: dict[str, dict[str, Any]],
    field_records: dict[str, dict[str, Any]],
    request: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    base = summaries["cp135_base"]
    candidate = summaries["candidate"]
    target = summaries["c1_target"]
    denominator = int(base["total_flips"]) - int(target["total_flips"])
    if denominator <= 0:
        raise JS1CError("matched T4 Stage-0 denominator is non-positive")
    rho = (int(base["total_flips"]) - int(candidate["total_flips"])) / denominator
    gate_passed = rho >= RHO_GATE
    maps = {name: stage0.edge_map(value) for name, value in summaries.items()}
    edge_rows = []
    for edge in sorted(maps["cp135_base"]):
        base_flips = int(maps["cp135_base"][edge]["flips"])
        candidate_flips = int(maps["candidate"][edge]["flips"])
        target_flips = int(maps["c1_target"][edge]["flips"])
        edge_denominator = base_flips - target_flips
        edge_rows.append(
            {
                "edge": edge,
                "base_flips": base_flips,
                "candidate_flips": candidate_flips,
                "target_flips": target_flips,
                "base_to_candidate_flip_gain": base_flips - candidate_flips,
                "rho_measured": (
                    (base_flips - candidate_flips) / edge_denominator
                    if edge_denominator > 0
                    else None
                ),
                "rho_undefined_reason": (
                    None if edge_denominator > 0 else "base-minus-target edge denominator <= 0"
                ),
            }
        )
    edge_rows.sort(key=lambda row: (-abs(row["base_to_candidate_flip_gain"]), row["edge"]))
    repeat_equal = (
        field_records["candidate"]["sha256"]
        == PRIOR_FIELD_RECORDS["prior_t1r1_candidate"]["sha256"]
    )
    status = "ADMITTED_RHO_GATE" if gate_passed else "NOT_ADMITTED_RHO_GATE"
    return {
        "schema": "ddm_js1c_cuda_custody_stage0_result.v1",
        "status": status,
        "axis": AXIS,
        "selection_mode": "full population, no sampling, all 600 non-overlapping pairs",
        "denominators": {
            "pairs": 600,
            "scorer_pixels_per_pair": 384 * 512,
            "total_scorer_pixels": TOTAL_PIXELS,
            "directed_edge_cells": 20,
            "undirected_interfaces": 10,
            "rho_base_minus_target_flips": denominator,
        },
        "verdict_scope": (
            "INSTANCE: exact T1R1 archive 12a5b181 through runtime tree 3bb8da9f on the "
            "retained contest-CUDA T4 batch-16 field instrument"
        ),
        "field_records": field_records,
        "remote_measurement": js1b.file_record(REMOTE_RESULT),
        "remote_volume_run_root": str(re1t.VOLUME_ROOT / RUN_ID),
        "sealed_request": js1b.file_record(SEALED_REQUEST),
        "objects": {
            "candidate_archive": request["candidate_archive"],
            "candidate_receiver": measurement["receiver"],
            "candidate_scorer": measurement["scorer"],
        },
        "decompositions": summaries,
        "road_hub_map": {name: road_hub(value) for name, value in summaries.items()},
        "comparison": {
            "base_flips": int(base["total_flips"]),
            "candidate_flips": int(candidate["total_flips"]),
            "target_flips": int(target["total_flips"]),
            "base_to_candidate_flip_gain": (
                int(base["total_flips"]) - int(candidate["total_flips"])
            ),
            "rho_measured": rho,
            "rho_required": RHO_GATE,
            "rho_gate_passed": gate_passed,
            "edge_rows": edge_rows,
        },
        "determinism": {
            "fresh_candidate_field_byte_identical_to_prior_js1b_run": repeat_equal,
            "prior_candidate_field_record": PRIOR_FIELD_RECORDS["prior_t1r1_candidate"],
            "fresh_candidate_field_record": field_records["candidate"],
        },
        "measured": (
            "full-n600 contest-CUDA T4 SegNet argmax fields, all 20 directed cells, all 10 "
            "undirected interfaces, matched-axis Road-hub attribution, and Stage-0 rho"
        ),
        "not_measured": (
            "PoseNet, complete score, contest-CPU, V0-V5 effects, trained receiver, coupled "
            "multi-token realization, or a new exact archive row"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def follow_on_receipts(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gate_passed = bool(result["comparison"]["rho_gate_passed"])
    v0_v5 = {
        "schema": "ddm_js1c_v0_v5_disposition.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER" if gate_passed else "FOLDED",
        "owner": "JS1 V0-V5 ladder owner",
        "consumer_store": str(STORE / "v0_v5"),
        "fire_trigger": (
            f"retained JS1C rho >= {RHO_GATE} on contest-CUDA custody"
            if gate_passed
            else f"not fireable: retained JS1C rho is below {RHO_GATE}"
        ),
        "action": (
            "execute the already-specified V0-V5 ladder from this retained field map"
            if gate_passed
            else "do not execute V0-V5 for this T1R1 instance"
        ),
        "stage0_result": js1b.file_record(STORE / "STAGE0_RESULT.json"),
    }
    task_1043 = {
        "schema": "ddm_js1c_task_1043_trigger_receipt.v1",
        "task_id": 1043,
        "trigger_satisfied": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "JS8 implicit joint distortion-conditioning successor",
        "consumer_store": str(EDGE_CONDITIONED_STORE),
        "fire_trigger": (
            "already satisfied: the full-n600 CUDA Stage-0 decomposition and field payloads "
            "are retained under the JS1C store"
        ),
        "action": (
            "build the implicit decoder-derived edge-state conditioning consumer; retain every "
            "payload and do not ship an explicit edge mask"
        ),
        "stage0_result": js1b.file_record(STORE / "STAGE0_RESULT.json"),
    }
    reroutes = {
        "schema": "ddm_js1c_failed_rho_reroutes.v1",
        "active": not gate_passed,
        "routes": [
            {
                "task_id": 982,
                "disposition": "QUEUED-WITH-A-FIRE-ORDER" if not gate_passed else "FOLDED",
                "owner": "JS1 trained-receiver successor",
                "consumer_store": str(EDGE_CONDITIONED_STORE / "trained_receiver"),
                "fire_trigger": (
                    "already satisfied by the retained JS1C rho failure"
                    if not gate_passed
                    else "not satisfied because the T1R1 Stage-0 gate passed"
                ),
                "action": "train a receiver that changes the realization map; do not retry T1R1",
            },
            {
                "task_id": 978,
                "disposition": "QUEUED-WITH-A-FIRE-ORDER" if not gate_passed else "FOLDED",
                "owner": "JS1 coupled multi-token successor",
                "consumer_store": str(EDGE_CONDITIONED_STORE / "coupled_multi_token"),
                "fire_trigger": (
                    "already satisfied by the retained JS1C rho failure"
                    if not gate_passed
                    else "not satisfied because the T1R1 Stage-0 gate passed"
                ),
                "action": (
                    "build coupled multi-token realization against the retained map; do not retry "
                    "frozen-receiver singleton composition"
                ),
            },
        ],
    }
    return {"v0_v5": v0_v5, "task_1043": task_1043, "reroutes": reroutes}


def consume() -> dict[str, Any]:
    request = json.loads(SEALED_REQUEST.read_text(encoding="utf-8"))
    validate_request(request)
    measurement = json.loads(REMOTE_RESULT.read_text(encoding="utf-8"))
    candidate_remote_record = verify_remote_measurement(measurement, request)
    require_record(
        LOCAL_CANDIDATE_DOWNLOAD,
        candidate_remote_record,
        label="downloaded fresh candidate field",
    )
    require_record(PRIOR_FINAL, PRIOR_FINAL_RECORD, label="prior T4 field receipt")
    prior_receipt = json.loads(PRIOR_FINAL.read_text(encoding="utf-8"))
    if prior_receipt.get("execution_status") != "COMPLETE":
        raise JS1CError("prior T4 field receipt is not complete")
    retained_sources = {}
    for name in ("gt", "cp135_base", "c1_target"):
        require_record(PRIOR_FIELD_PATHS[name], PRIOR_FIELD_RECORDS[name], label=f"prior {name} field")
        retained_sources[name] = atomic_copy(
            PRIOR_FIELD_PATHS[name], RETAINED_FIELDS / f"{name}_argmax_n600.npy"
        )
    retained_sources["candidate"] = js1b.file_record(LOCAL_CANDIDATE_DOWNLOAD)
    paths = {
        "gt": RETAINED_FIELDS / "gt_argmax_n600.npy",
        "cp135_base": RETAINED_FIELDS / "cp135_base_argmax_n600.npy",
        "c1_target": RETAINED_FIELDS / "c1_target_argmax_n600.npy",
        "candidate": LOCAL_CANDIDATE_DOWNLOAD,
    }
    arrays = {}
    for name, path in paths.items():
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.shape != (600, 384, 512) or value.dtype != np.uint8:
            raise JS1CError(f"retained {name} field shape/dtype differs")
        arrays[name] = value
    gt = arrays.pop("gt")
    summaries = {
        name: stage0.decomposition(
            value, gt, DECOMPOSITION_ROOT / f"{name}_per_pair.jsonl"
        )
        for name, value in arrays.items()
    }
    if int(summaries["cp135_base"]["total_flips"]) != BASE_FLIPS:
        raise JS1CError("retained T4 base field no longer has 34,970 flips")
    if int(summaries["c1_target"]["total_flips"]) != TARGET_FLIPS:
        raise JS1CError("retained T4 C1 target field no longer has 27,330 flips")
    result_path = STORE / "STAGE0_RESULT.json"
    # The result references its path only through follow-on receipts, so it can
    # be written once without self-hash recursion.
    result = stage0_result(
        summaries=summaries,
        field_records=retained_sources,
        request=request,
        measurement=measurement,
    )
    js1b.atomic_json(result_path, result)
    receipts = follow_on_receipts(result)
    js1b.atomic_json(STORE / "V0_V5_DISPOSITION.json", receipts["v0_v5"])
    js1b.atomic_json(STORE / "TASK_1043_TRIGGER_RECEIPT.json", receipts["task_1043"])
    js1b.atomic_json(STORE / "REROUTE_FIRE_ORDERS.json", receipts["reroutes"])
    return {
        "schema": "ddm_js1c_cuda_custody_stage0_consume_result.v1",
        "status": result["status"],
        "stage0_result": js1b.file_record(result_path),
        "v0_v5_disposition": js1b.file_record(STORE / "V0_V5_DISPOSITION.json"),
        "task_1043_trigger": js1b.file_record(STORE / "TASK_1043_TRIGGER_RECEIPT.json"),
        "reroute_fire_orders": js1b.file_record(STORE / "REROUTE_FIRE_ORDERS.json"),
        "retained_fields": retained_sources,
        "score_claim": False,
        "pointer_moved": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="scorer-free seal; never dispatches")
    recover_parser = commands.add_parser("recover", help="harvest a detached Modal call")
    recover_parser.add_argument("--output-dir", type=Path, default=DISPATCH_OUTPUT)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    commands.add_parser("consume", help="consume retained fields without a scorer rerun")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2, sort_keys=True))
        return 0
    if args.command == "recover":
        return recover(args.output_dir, args.timeout_seconds)
    if args.command == "consume":
        print(json.dumps(consume(), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
