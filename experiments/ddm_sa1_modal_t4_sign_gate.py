#!/usr/bin/env python3
"""Governed Modal dispatcher for the candidate-only SA1 T4 sign gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import modal

from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch

try:
    from experiments.modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image
except ModuleNotFoundError:
    from modal_auth_eval import UPSTREAM_LOCKED_VENV, eval_image  # type: ignore[no-redef]
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
REMOTE_REPO: Final = Path("/workspace/pact")
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_sa1_t4_sign_gate_worker.py"
REMOTE_JS1B_WORKER: Final = (
    REMOTE_REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
)
APP_NAME: Final = "comma-ddm-sa1-t4-sign-gate"
LANE_LABEL: Final = "ddm_sa1_t4_sign_gate"
AXIS: Final = "[contest-CUDA T4 frozen-SegNet argmax field, n600, batch=16] COMPONENT-ONLY"
VOLUME_NAME: Final = js1b_dispatch.VOLUME_NAME
VOLUME_ROOT: Final = js1b_dispatch.VOLUME_ROOT
COMMIT_PERIOD_SECONDS: Final = 20.0
CONTEST_LIMIT_SECONDS: Final = 1_800.0
BASE_ARCHIVE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip"
)
BASE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_BYTES: Final = 186_252
DEFAULT_RESULT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/FINAL_RESULT.json")
DEFAULT_OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/t4_sign_gate_v3")


class SA1DispatchError(RuntimeError):
    """A custody, resume, dispatch, or recovery invariant failed."""


def k_arithmetic() -> dict[str, Any]:
    projected = 466.0 + 39.405 + 300.0
    return {
        "schema": "ddm_sa1_t4_k_arithmetic.v1",
        "k_archives": 1,
        "measured_seconds_per_decode": 466.0,
        "scorer_passes": 1,
        "measured_seconds_per_full_scorer_pass": 39.405,
        "reserve_seconds": 300.0,
        "projected_seconds_with_reserve": projected,
        "contest_limit_seconds": CONTEST_LIMIT_SECONDS,
        "headroom_seconds": CONTEST_LIMIT_SECONDS - projected,
        "fits_30_minutes": projected <= CONTEST_LIMIT_SECONDS,
        "epistemic_status": "DERIVED_FROM_MEASURED_PRIOR_T4_COMPONENT_TIMES",
    }


def _require_candidate(
    candidate_archive: Path,
    candidate_runtime: Path,
    final_result: Path,
) -> tuple[dict[str, Any], float]:
    js1b_dispatch.require_exact(
        BASE_ARCHIVE,
        size=BASE_BYTES,
        digest=BASE_SHA256,
        label="CP135 base archive",
    )
    if not final_result.is_file():
        raise SA1DispatchError(f"missing SA1 final result: {final_result}")
    result = json.loads(final_result.read_text())
    if result.get("status") != "READY_TO_FIRE":
        raise SA1DispatchError("SA1 result is not READY_TO_FIRE")
    expected = result["winner"]["archive"]
    observed = js1b_dispatch.file_record(candidate_archive)
    if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
        raise SA1DispatchError("candidate archive differs from SA1 selected winner")
    runtime_archive = candidate_runtime / "archive.zip"
    runtime_record = js1b_dispatch.file_record(runtime_archive)
    if runtime_record["bytes"] != observed["bytes"] or runtime_record["sha256"] != observed["sha256"]:
        raise SA1DispatchError("candidate runtime archive differs from selected winner")
    if observed["bytes"] - BASE_BYTES > 2_048:
        raise SA1DispatchError("candidate exceeds the SA1 counted-byte box")
    with zipfile.ZipFile(BASE_ARCHIVE) as base_zip, zipfile.ZipFile(candidate_archive) as candidate_zip:
        if base_zip.namelist() != ["p"]:
            raise SA1DispatchError("CP135 base archive grammar differs")
        if candidate_zip.namelist() != ["p", "sa1_conditioner.br"]:
            raise SA1DispatchError("SA1 candidate archive grammar differs")
        if base_zip.read("p") != candidate_zip.read("p"):
            raise SA1DispatchError("SA1 candidate changed CP135's counted pose-carrier member")
    return observed, float(result["winner"]["local_ordering"]["pose_delta_stratified_n32"])


def prepare_request(
    *,
    candidate_archive: Path,
    candidate_runtime: Path,
    sa1_final_result: Path,
    run_id: str,
    resume_from: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise SA1DispatchError("--run-id must be one safe path component")
    if resume_from != run_id:
        raise SA1DispatchError("--resume-from is mandatory and must equal --run-id")
    archive_record, pose_delta = _require_candidate(
        candidate_archive.resolve(), candidate_runtime.resolve(), sa1_final_result.resolve()
    )
    runtime_bundle, runtime_manifest = js1b_dispatch.build_runtime_bundle(
        candidate_runtime,
        label="sa1_candidate",
    )
    payloads = {
        "sa1_candidate_archive.zip": candidate_archive.read_bytes(),
        "sa1_candidate_runtime.zip": runtime_bundle,
    }
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_sa1_t4_modal_request.v1",
        "axis": AXIS,
        "run_id": run_id,
        "resume_from": resume_from,
        "seed": 1234,
        "batch_size": 16,
        "candidate_archive": archive_record,
        "local_pose_delta_stratified_n32": pose_delta,
        "source_git_head": git_head,
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": js1b_dispatch.sha256_bytes(git_status),
        "dispatcher_source_sha256": js1b_dispatch.sha256_file(Path(__file__)),
        "worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_sa1_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "inputs": {
            name: js1b_dispatch.payload_record(payload) for name, payload in payloads.items()
        },
        "runtime_manifest": runtime_manifest,
        "k_arithmetic": k_arithmetic(),
        "retention_volume": VOLUME_NAME,
        "retention_volume_run_path": str(VOLUME_ROOT / run_id),
        "prior_t4_field_run_path": str(VOLUME_ROOT / "ddm_js1b_20260813b"),
        "resume_required": True,
        "per_stage_checkpoints": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    return payloads, request


def persist_local_fire_inputs(
    output_dir: Path,
    payloads: dict[str, bytes],
    request: dict[str, Any],
) -> None:
    root = output_dir / "fire_inputs"
    for name, payload in payloads.items():
        path = root / name
        if path.is_file():
            if js1b_dispatch.sha256_file(path) != js1b_dispatch.sha256_bytes(payload):
                raise SA1DispatchError(f"retained fire input differs: {path}")
        else:
            js1b_dispatch.atomic_bytes(path, payload)
    request_path = output_dir / "SA1_T4_LOCAL_REQUEST.json"
    request_payload = js1b_dispatch.canonical_json_bytes(request)
    if request_path.is_file():
        if request_path.read_bytes() != request_payload:
            raise SA1DispatchError(f"retained fire request differs: {request_path}")
    else:
        js1b_dispatch.atomic_bytes(request_path, request_payload)


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
gate_image = (
    eval_image.add_local_file(
        "experiments/ddm_sa1_t4_sign_gate_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py",
        remote_path=str(REMOTE_JS1B_WORKER),
        copy=False,
    )
    .add_local_python_source("ddm_sa1_modal_t4_sign_gate")
)


@app.function(
    image=gate_image,
    gpu="T4",
    timeout=int(CONTEST_LIMIT_SECONDS),
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_gate(payloads: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    remote_inputs = {**payloads, "REQUEST.json": js1b_dispatch.canonical_json_bytes(request)}
    for name, payload in remote_inputs.items():
        path = run_root / "inputs" / name
        if path.is_file():
            if js1b_dispatch.sha256_file(path) != js1b_dispatch.sha256_bytes(payload):
                raise SA1DispatchError(f"resume input differs: {path}")
        else:
            js1b_dispatch.atomic_bytes(path, payload)
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
    retained_volume.commit()
    final_path = run_root / "FINAL_RESULT.json"
    if returncode or not final_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8_000:]
        return {
            "schema": "ddm_sa1_t4_modal_return.v1",
            "passed": False,
            "returncode": returncode,
            "axis": AXIS,
            "run_id": run_id,
            "volume_name": VOLUME_NAME,
            "volume_path": str(run_root),
            "worker_log_tail": tail,
            "resume_same_run_id": run_id,
            "score_claim": False,
        }
    final_bytes = final_path.read_bytes()
    return {
        "schema": "ddm_sa1_t4_modal_return.v1",
        "passed": True,
        "returncode": 0,
        "axis": AXIS,
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "volume_download_commands": [
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/FINAL_RESULT.json ./modal_{run_id}/FINAL_RESULT.json",
            f".venv/bin/modal volume get --force {VOLUME_NAME} {run_id}/retained/fields/ ./modal_{run_id}/retained/fields/",
        ],
        "final_result_sha256": js1b_dispatch.sha256_bytes(final_bytes),
        "artifacts": {"SA1_T4_FINAL_RESULT.json": final_bytes},
        "score_claim": False,
        "promotion_eligible": False,
    }


def recover(output_dir: Path, *, timeout_seconds: float = 0.0) -> int:
    output = output_dir.resolve()
    spawn = json.loads((output / "modal_auth_eval_spawn.json").read_text())
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise SA1DispatchError(f"remote return is not a dict: {type(result).__name__}")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise SA1DispatchError(f"unsafe returned artifact: {name!r}")
        js1b_dispatch.atomic_bytes(output / name, payload)
    js1b_dispatch.atomic_json(output / "modal_sa1_t4_result.json", result)
    passed = bool(result.get("passed"))
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if passed else "failed",
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_sa1_argmax_field_component",
        evidence_grade="contest-CUDA T4 frozen-SegNet argmax field n600 batch16",
        lane_id=str(spawn["lane_id"]),
        label=LANE_LABEL,
        gpu="T4",
        agent=str(spawn["claim_agent"]),
        harvest_result={key: value for key, value in result.items() if key != "worker_log_tail"},
    )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(spawn["lane_id"]),
            instance_job_id=str(spawn["instance_job_id"]),
            agent=str(spawn["claim_agent"]),
            force=True,
        ),
        status="completed_sa1_t4_gate_recovered" if passed else "failed_sa1_t4_gate_recovered",
        notes=f"SA1 T4 gate recovered; call_id={call_id}; output={output}",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


@app.local_entrypoint()
def main(
    candidate_archive: str,
    candidate_runtime: str,
    sa1_final_result: str = str(DEFAULT_RESULT),
    output_dir: str = str(DEFAULT_OUTPUT),
    run_id: str = "ddm_sa1_t4_sign_20260813",
    resume_from: str = "ddm_sa1_t4_sign_20260813",
    lane_id: str = "ddm_sa1_t4_sign_gate",
    instance_job_id: str = "modal:ddm_sa1_t4_sign_20260813",
    claim_agent: str = "main:ddm_sa1",
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    if detach and not provider_detach_ack:
        raise SystemExit("FATAL: --detach requires --provider-detach-ack")
    if not k_arithmetic()["fits_30_minutes"]:
        raise SystemExit("FATAL: candidate-only projection exceeds 30 minutes")
    payloads, request = prepare_request(
        candidate_archive=Path(candidate_archive),
        candidate_runtime=Path(candidate_runtime),
        sa1_final_result=Path(sa1_final_result),
        run_id=run_id,
        resume_from=resume_from,
    )
    output = Path(output_dir).resolve()
    persist_local_fire_inputs(output, payloads, request)
    spec = ClaimSpec(
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        agent=claim_agent,
        notes=f"SA1 candidate-only T4 field gate; run={run_id}; volume={VOLUME_NAME}/{run_id}",
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=spec,
        status="active_sa1_t4_gate_spawning",
    )
    assert_modal_single_flight(label=LANE_LABEL, lane_id=lane_id, repo_root=REPO)
    if detach:
        call = run_gate.spawn(payloads, request)
        call_id = function_call_id(call)
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=lane_id,
            label=LANE_LABEL,
            platform="modal",
            gpu="T4",
            expected_axis="contest_cuda_sa1_argmax_field_component",
            recipe="experiments/ddm_sa1_modal_t4_sign_gate.py::main",
            max_seconds=int(CONTEST_LIMIT_SECONDS),
            agent=claim_agent,
            base_archive_sha256=BASE_SHA256,
            composed_archive_sha256=request["candidate_archive"]["sha256"],
            archive_count=1,
            volume_name=VOLUME_NAME,
            volume_run_id=run_id,
        )
        write_spawn_metadata(
            out_dir=output,
            tool="experiments/ddm_sa1_modal_t4_sign_gate.py",
            app=APP_NAME,
            axis="contest_cuda_sa1_argmax_field_component",
            call_id=call_id,
            local_request=request,
            result_json_name="modal_sa1_t4_result.json",
            recover_tool="experiments/ddm_sa1_modal_t4_sign_gate.py recover",
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
                notes=f"SA1 detached T4 gate accepted; call_id={call_id}; output={output}",
            ),
            status="active_sa1_t4_gate_spawned",
        )
        print(f"DISPATCHED call_id={call_id}")
        print(f"RECOVER .venv/bin/python {__file__} recover --output-dir {output}")
        return
    result = run_gate.remote(payloads, request)
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        js1b_dispatch.atomic_bytes(output / name, payload)
    js1b_dispatch.atomic_json(output / "modal_sa1_t4_result.json", result)
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(lane_id=lane_id, instance_job_id=instance_job_id, agent=claim_agent, force=True),
        status="completed_sa1_t4_gate_synchronous" if result.get("passed") else "failed_sa1_t4_gate_synchronous",
        notes=f"SA1 synchronous T4 gate returned; output={output}; resume={run_id}",
    )
    if not result.get("passed"):
        raise SystemExit("FATAL: SA1 T4 gate failed; resume the same run id")


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover a detached SA1 T4 sign gate.")
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
