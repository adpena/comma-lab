#!/usr/bin/env python3
"""MAIN-owned JO1 dispatch gate.

Local preparation is pure and scorer-free.  The materializer entrypoint is a
real, retained T4 producer and is deliberately independent of JO1's training
backend.  The memory-preflight and train entrypoints remain fail-closed until
the rc2 fresh-Schur receiver-close backend exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import modal

_SOURCE_REPO = Path(__file__).resolve().parents[1]
REPO = (
    Path("/workspace/pact")
    if Path("/workspace/pact/experiments").is_dir()
    else _SOURCE_REPO
)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

design = importlib.import_module("experiments.ddm_jo1_joint_objective_design")
worker = importlib.import_module("experiments.ddm_jo1_joint_objective_worker")
materializer_worker = importlib.import_module(
    "experiments.ddm_jo1_payload_materializer_worker"
)
auth_eval = importlib.import_module("experiments.modal_auth_eval")
from tac.deploy.modal.auth_eval import prepare_modal_auth_eval_request

APP_NAME = "comma-ddm-jo1-joint-objective"
LANE_LABEL = "ddm_jo1_joint_objective"
app = modal.App(APP_NAME, include_source=False)
materializer_image = (
    auth_eval.eval_image
    .add_local_file(
        "experiments/__init__.py",
        remote_path="/workspace/pact/experiments/__init__.py",
    )
    .add_local_file(
        "experiments/modal_auth_eval.py",
        remote_path="/workspace/pact/experiments/modal_auth_eval.py",
    )
    .add_local_file(
        "experiments/ddm_jo1_joint_objective_design.py",
        remote_path="/workspace/pact/experiments/ddm_jo1_joint_objective_design.py",
    )
    .add_local_file(
        "experiments/ddm_jo1_joint_objective_worker.py",
        remote_path="/workspace/pact/experiments/ddm_jo1_joint_objective_worker.py",
    )
    .add_local_file(
        "experiments/ddm_jo1_payload_materializer_worker.py",
        remote_path="/workspace/pact/experiments/ddm_jo1_payload_materializer_worker.py",
    )
    .add_local_file(
        "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py",
        remote_path=(
            "/workspace/pact/experiments/"
            "ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    )
    .add_local_python_source("ddm_jo1_modal_joint_objective")
)


class JO1DispatchError(RuntimeError):
    """Fail-closed dispatch authorization or implementation error."""


def prepare_local(config: design.CompiledConfig, output: Path) -> dict[str, Any]:
    """Compile readiness/fire artifacts without importing scorers or spawning."""
    result = design.prepare(config, destination=output.resolve())
    if result.get("dispatch_performed") is not False:
        raise JO1DispatchError("local preparation unexpectedly reported a dispatch")
    return result


def dispatch_request(
    *,
    entrypoint: str,
    compiled_config: Path,
    expected_config_sha256: str,
    main_owned_dispatch_authorization: bool,
    detach: bool,
    provider_detach_ack: bool,
) -> dict[str, Any]:
    if entrypoint not in {"materialize_scorer_payloads", "memory_preflight", "train"}:
        raise JO1DispatchError(f"unknown JO1 entrypoint: {entrypoint}")
    if not main_owned_dispatch_authorization:
        raise JO1DispatchError("JO1 dispatch requires explicit MAIN-owned authorization")
    if not detach or not provider_detach_ack:
        raise JO1DispatchError("JO1 dispatch requires detach and provider-detach acknowledgement")
    config = design.load_compiled_config(compiled_config, expected_config_sha256)
    expected_action = {
        "materialize_scorer_payloads": "materialize_scorer_payloads",
        "memory_preflight": "memory_preflight",
        "train": "train",
    }[entrypoint]
    # A scorer-free ``prepare`` seal is the umbrella prerequisite seal.  After
    # each asynchronous harvest MAIN must reseal with the concrete next action;
    # action-specific seals may only invoke their matching entrypoint.
    if config.action not in {"prepare", expected_action}:
        raise JO1DispatchError(
            f"compiled action differs for {entrypoint}: {config.action} != {expected_action}"
        )
    readiness = (
        design.materializer_readiness(config)
        if entrypoint == "materialize_scorer_payloads"
        else design.readiness(config)
    )
    # Scorer materialization is the prerequisite that may run while readiness
    # is blocked on missing fields.  Memory and train require their predecessor.
    if entrypoint == "memory_preflight" and any(
        "FIELD" in blocker or "POSE6" in blocker or "TOKENS" in blocker
        for blocker in readiness["blockers"]
    ):
        raise JO1DispatchError("memory preflight is ordered after scorer payload harvest")
    if entrypoint == "train" and readiness["status"] != "READY_TO_FIRE":
        raise JO1DispatchError(f"training readiness is blocked: {readiness['blockers']}")
    if entrypoint == "materialize_scorer_payloads" and readiness["status"] != "READY_TO_FIRE":
        raise JO1DispatchError(f"materializer readiness is blocked: {readiness['blockers']}")
    return {
        "schema": "ddm_jo1_dispatch_request.v1",
        "entrypoint": entrypoint,
        "compiled_action": config.action,
        "lane_id": config.dispatch.lane_id,
        "claim_agent": "MAIN",
        "platform": "modal",
        "gpu": "T4",
        "compiled_config": str(compiled_config.resolve()),
        "compiled_config_sha256": expected_config_sha256,
        "workload_config_sha256": config.workload_config_sha256,
        "readiness": readiness,
    }


def governed_spawn(
    *,
    request: dict[str, Any],
    spawn: Callable[[], Any],
    output: Path,
) -> str:
    """Canonical claim/single-flight/call-id path for a closed remote backend.

    This function is real and deliberately separate from local preparation.
    The retained materializer calls it; memory preflight and training remain
    named-blocked below.
    """
    from tac.deploy.modal.auth_eval import (
        ClaimSpec,
        claim_modal_auth_eval_dispatch,
        function_call_id,
        write_spawn_metadata,
    )
    from tac.deploy.modal.call_id_ledger import register_dispatched_call_id_fail_closed
    from tac.deploy.modal.single_flight import assert_modal_single_flight

    output.mkdir(parents=True, exist_ok=True)
    design.atomic_json(output / "DISPATCH_REQUEST.json", request)
    spec = ClaimSpec(
        lane_id=str(request["lane_id"]),
        instance_job_id=f"modal:{request['workload_config_sha256'][:16]}",
        agent="MAIN",
        notes=f"JO1 {request['entrypoint']} governed dispatch",
    )
    claim_modal_auth_eval_dispatch(
        repo_root=Path(__file__).resolve().parents[1],
        spec=spec,
        status=f"active_jo1_{request['entrypoint']}_spawning",
    )
    assert_modal_single_flight(
        label=f"{LANE_LABEL}_{request['entrypoint']}",
        lane_id=str(request["lane_id"]),
        repo_root=Path(__file__).resolve().parents[1],
    )
    call = spawn()
    call_id = function_call_id(call)
    register_dispatched_call_id_fail_closed(
        call_id=call_id,
        lane_id=str(request["lane_id"]),
        label=f"{LANE_LABEL}_{request['entrypoint']}",
        platform="modal",
        gpu="T4",
        expected_axis="contest_cuda_jo1_component",
        recipe="experiments/ddm_jo1_modal_joint_objective.py",
        max_seconds=10_800,
        agent="MAIN",
        base_archive_sha256=str(request["archive_sha256"]),
        composed_archive_sha256=str(request["archive_sha256"]),
        archive_count=1,
        volume_name=str(request["remote_volume_name"]),
        volume_run_id=str(request["remote_volume_run_id"]),
    )
    write_spawn_metadata(
        out_dir=output,
        tool="experiments/ddm_jo1_modal_joint_objective.py",
        app=APP_NAME,
        axis="contest_cuda_jo1_component",
        call_id=call_id,
        local_request=request,
        result_json_name="modal_jo1_result.json",
        recover_tool="tools/modal_endpoint_close.py",
        extra={"lane_id": request["lane_id"], "entrypoint": request["entrypoint"]},
    )
    return call_id


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_control_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Write control-plane evidence without depending on a worker exception class."""
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _remote_inputs_seen(
    *, request: dict[str, Any], archive_bytes: bytes, runtime_zip_bytes: bytes
) -> dict[str, Any]:
    return {
        "request_schema": request.get("schema"),
        "request_keys": sorted(str(key) for key in request),
        "remote_volume_run_id": request.get("remote_volume_run_id"),
        "resume_from": request.get("resume_from"),
        "workload_config_sha256": request.get("workload_config_sha256"),
        "expected_runtime_tree_sha256": request.get("expected_runtime_tree_sha256"),
        "source_git_head": request.get("source_git_head"),
        "dispatcher_source_sha256": request.get("dispatcher_source_sha256"),
        "worker_source_sha256": request.get("worker_source_sha256"),
        "archive": {
            "bytes": len(archive_bytes),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        },
        "runtime_bundle": {
            "bytes": len(runtime_zip_bytes),
            "sha256": hashlib.sha256(runtime_zip_bytes).hexdigest(),
        },
    }


def _remote_run_binding(request: dict[str, Any]) -> tuple[Path, str | None]:
    """Return a safe volume path plus a named request-identity blocker, if any."""
    run_id = request.get("remote_volume_run_id") if isinstance(request, dict) else None
    if isinstance(run_id, str) and re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", run_id):
        return auth_eval.AUTH_CACHE_VOLUME_ROOT / run_id, None
    request_digest = hashlib.sha256(repr(request).encode("utf-8")).hexdigest()[:16]
    return (
        auth_eval.AUTH_CACHE_VOLUME_ROOT / f"invalid_jo1_remote_request_{request_digest}",
        f"remote_volume_run_id is invalid: {run_id!r}",
    )


def _execute_with_remote_failure_receipt(
    *,
    run_root: Path,
    inputs_seen: dict[str, Any],
    operation: Callable[[dict[str, str]], dict[str, Any]],
    commit: Callable[[], Any],
    diagnostic_probe: bool = False,
) -> dict[str, Any]:
    """Execute one remote operation with an immutable, volume-backed black box."""
    attempt_id = f"{time.time_ns()}_{os.getpid()}"
    stage = {"name": "entrypoint_entered"}
    start = {
        "schema": "ddm_jo1_remote_start.v1",
        "attempt_id": attempt_id,
        "stage": stage["name"],
        "inputs_seen": inputs_seen,
        "diagnostic_probe": diagnostic_probe,
        "score_claim": False,
        "promotion_eligible": False,
    }
    try:
        _atomic_control_json(run_root / f"attempts/{attempt_id}.start.json", start)
        _atomic_control_json(run_root / "REMOTE_START.json", start)
        commit()
        result = operation(stage)
        stage["name"] = "success_commit"
        commit()
        return result
    except Exception as error:
        failure = {
            "schema": "ddm_jo1_payload_materializer_failure.v2",
            "attempt_id": attempt_id,
            "stage": stage["name"],
            "inputs_seen": inputs_seen,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_repr": repr(error),
            "traceback": traceback.format_exc(),
            "diagnostic_probe": diagnostic_probe,
            "retention_policy": (
                "no cleanup performed; every input or intermediate written before "
                "failure remains on the mounted volume"
            ),
            "complete_payload_set": False,
            "score_claim": False,
            "promotion_eligible": False,
        }
        immutable_path = run_root / f"failures/{attempt_id}.json"
        try:
            immutable_record = _atomic_control_json(immutable_path, failure)
            # Commit the immutable black-box record before writing the mutable
            # convenience pointer. A pointer-write failure cannot erase the
            # receipt that explains it.
            commit()
            _atomic_control_json(
                run_root / "REMOTE_FAILURE.json",
                {**failure, "immutable_failure_receipt": immutable_record},
            )
            commit()
        except Exception as receipt_error:
            raise RuntimeError(
                "JO1 remote operation failed and its failure recorder also failed; "
                f"stage={stage['name']}; original={type(error).__name__}: {error}; "
                f"recorder={type(receipt_error).__name__}: {receipt_error}"
            ) from error
        raise RuntimeError(
            "JO1 remote operation failed; "
            f"stage={stage['name']}; receipt={immutable_path}; "
            f"cause={type(error).__name__}: {error}"
        ) from error


def _spawn_materializer(config: design.CompiledConfig, request: dict[str, Any]) -> str:
    materializer = config.materializer
    if materializer is None:
        raise JO1DispatchError("materializer config is absent after readiness")
    prepared = prepare_modal_auth_eval_request(
        archive=materializer.archive.path,
        output_dir=Path(materializer.harvest_root) / "dispatch",
        inflate_sh="inflate.sh",
        submission_dir=materializer.runtime.path,
        default_output_root=design.MATERIALIZER_OUTPUT_ROOT,
    )
    if (
        prepared.archive_size_bytes != materializer.archive.bytes
        or prepared.archive_sha256 != materializer.archive.sha256
    ):
        raise JO1DispatchError("prepared materializer archive differs from the seal")
    if prepared.submission_dir_zip is None or prepared.submission_dir_zip_sha256 is None:
        raise JO1DispatchError("materializer runtime transport bundle was not created")
    observed_runtime_tree, runtime_content_tree = (
        auth_eval._validate_uploaded_runtime_tree_expectation(
            expected_runtime_tree_sha256=materializer.expected_runtime_tree_sha256,
            submission_dir_path=prepared.submission_dir_path,
            inflate_sh_rel=prepared.inflate_sh_rel,
        )
    )
    source_pose = config.inputs.source_pose6_targets
    gt_argmax = config.inputs.gt_argmax_field
    worker_source = config.inputs.materializer_worker_source
    if source_pose is None or gt_argmax is None or worker_source is None:
        raise JO1DispatchError("materializer source bindings disappeared after readiness")
    archive_upload = {
        "bytes": len(prepared.archive_bytes),
        "sha256": hashlib.sha256(prepared.archive_bytes).hexdigest(),
    }
    runtime_upload = {
        "bytes": len(prepared.submission_dir_zip),
        "sha256": prepared.submission_dir_zip_sha256,
    }
    retained_upload_root = prepared.output_dir / "retained_uploads"
    local_archive_upload = design._atomic_bytes(
        retained_upload_root / "archive.zip", prepared.archive_bytes
    )
    local_runtime_upload = design._atomic_bytes(
        retained_upload_root / "submission_dir.zip", prepared.submission_dir_zip
    )
    if (
        local_archive_upload["bytes"] != archive_upload["bytes"]
        or local_archive_upload["sha256"] != archive_upload["sha256"]
        or local_runtime_upload["bytes"] != runtime_upload["bytes"]
        or local_runtime_upload["sha256"] != runtime_upload["sha256"]
    ):
        raise JO1DispatchError("locally retained upload payloads differ")
    remote_request = {
        "schema": materializer_worker.REQUEST_SCHEMA,
        "resume_from": config.workload_config_sha256,
        "vehicle_id": materializer.vehicle_id,
        "workload_config_sha256": config.workload_config_sha256,
        "remote_volume_name": materializer.remote_volume_name,
        "remote_volume_run_id": materializer.remote_volume_run_id,
        "expected_runtime_tree_sha256": observed_runtime_tree,
        "runtime_content_tree_sha256": runtime_content_tree,
        "batch_pairs": materializer.batch_pairs,
        "chunk_pair_limit": materializer.chunk_pair_limit,
        "uploads": {
            "archive.zip": archive_upload,
            "submission_dir.zip": runtime_upload,
        },
        "source_targets": {
            "gt_argmax_field": gt_argmax.model_dump(mode="json"),
            "source_pose6_targets": source_pose.model_dump(mode="json"),
        },
        "source_git_head": _git_head(),
        "dispatcher_source_sha256": _file_sha256(Path(__file__).resolve()),
        "worker_source_sha256": worker_source.sha256,
        "score_claim": False,
        "promotion_eligible": False,
    }
    request.update(
        {
            "archive_sha256": prepared.archive_sha256,
            "archive_bytes": prepared.archive_size_bytes,
            "runtime_tree_sha256": observed_runtime_tree,
            "runtime_content_tree_sha256": runtime_content_tree,
            "runtime_transport_sha256": prepared.submission_dir_zip_sha256,
            "remote_volume_name": materializer.remote_volume_name,
            "remote_volume_run_id": materializer.remote_volume_run_id,
            "harvest_root": str(Path(materializer.harvest_root).resolve()),
            "locally_retained_uploads": {
                "archive": local_archive_upload,
                "runtime_bundle": local_runtime_upload,
            },
        }
    )
    return governed_spawn(
        request=request,
        output=prepared.output_dir,
        spawn=lambda: run_payload_materializer.spawn(
            request=remote_request,
            archive_bytes=prepared.archive_bytes,
            runtime_zip_bytes=prepared.submission_dir_zip,
        ),
    )


@app.function(
    image=materializer_image,
    gpu="T4",
    timeout=4800,
    volumes={str(auth_eval.AUTH_CACHE_VOLUME_ROOT): auth_eval.auth_cache_vol},
)
def run_payload_materializer(
    *,
    request: dict[str, Any],
    archive_bytes: bytes,
    runtime_zip_bytes: bytes,
) -> dict[str, Any]:
    run_root, request_blocker = _remote_run_binding(request)
    inputs_seen = _remote_inputs_seen(
        request=request,
        archive_bytes=archive_bytes,
        runtime_zip_bytes=runtime_zip_bytes,
    )

    def operation(stage: dict[str, str]) -> dict[str, Any]:
        stage["name"] = "validate_remote_request"
        if request_blocker is not None:
            raise ValueError(request_blocker)
        stage["name"] = "stage_uploaded_inputs"
        materializer_worker.stage_uploaded_inputs(
            run_root=run_root,
            request=request,
            archive_bytes=archive_bytes,
            runtime_zip_bytes=runtime_zip_bytes,
        )
        stage["name"] = "materializer_run"
        return materializer_worker.run(run_root, str(request["resume_from"]))

    return _execute_with_remote_failure_receipt(
        run_root=run_root,
        inputs_seen=inputs_seen,
        operation=operation,
        commit=auth_eval.auth_cache_vol.commit,
    )


@app.function(
    image=materializer_image,
    cpu=1,
    timeout=300,
    memory=1024,
    volumes={str(auth_eval.AUTH_CACHE_VOLUME_ROOT): auth_eval.auth_cache_vol},
)
def run_control_plane_probe(
    *, run_id: str, expected_source_sha256: dict[str, str]
) -> dict[str, Any]:
    """Prove package closure and the failure black box without GPU or scorers."""
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", run_id) is None:
        raise RuntimeError(f"control-plane probe run_id is invalid: {run_id!r}")
    run_root = auth_eval.AUTH_CACHE_VOLUME_ROOT / run_id
    modules = {
        "design": importlib.import_module("experiments.ddm_jo1_joint_objective_design"),
        "worker": importlib.import_module("experiments.ddm_jo1_joint_objective_worker"),
        "materializer_worker": importlib.import_module(
            "experiments.ddm_jo1_payload_materializer_worker"
        ),
        "retained_worker": importlib.import_module(
            "experiments.ddm_js1b_cuda_argmax_field_materializer_worker"
        ),
        "modal_auth_eval": importlib.import_module("experiments.modal_auth_eval"),
    }
    module_records = {
        name: {
            "path": str(Path(module.__file__).resolve()),
            "sha256": _file_sha256(Path(module.__file__).resolve()),
        }
        for name, module in modules.items()
    }
    observed_sources = {
        **{name: record["sha256"] for name, record in module_records.items()},
        "dispatcher": _file_sha256(Path(__file__).resolve()),
    }
    if observed_sources != expected_source_sha256:
        raise RuntimeError(
            f"control-plane source closure differs: {observed_sources} != {expected_source_sha256}"
        )
    package_root = Path("/workspace/pact/experiments")
    if any(
        package_root not in Path(record["path"]).parents
        for record in module_records.values()
    ):
        raise RuntimeError(f"control-plane package route differs: {module_records}")

    def sentinel(stage: dict[str, str]) -> dict[str, Any]:
        stage["name"] = "diagnostic_injected_failure"
        raise ValueError("JO1U2_DIAGNOSTIC_FAILURE_SENTINEL")

    loud_error = ""
    try:
        _execute_with_remote_failure_receipt(
            run_root=run_root,
            inputs_seen={
                "probe": "package closure plus structured failure recorder",
                "expected_source_sha256": expected_source_sha256,
            },
            operation=sentinel,
            commit=auth_eval.auth_cache_vol.commit,
            diagnostic_probe=True,
        )
    except RuntimeError as error:
        loud_error = str(error)
    if not loud_error or "diagnostic_injected_failure" not in loud_error:
        raise RuntimeError("diagnostic failure did not re-raise a loud built-in exception")
    failure_path = run_root / "REMOTE_FAILURE.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        failure.get("stage") != "diagnostic_injected_failure"
        or failure.get("error_type") != "ValueError"
        or "JO1U2_DIAGNOSTIC_FAILURE_SENTINEL" not in failure.get("traceback", "")
    ):
        raise RuntimeError(f"diagnostic failure receipt is incomplete: {failure}")
    result = {
        "schema": "ddm_jo1u2_control_plane_probe.v1",
        "passed": True,
        "run_id": run_id,
        "module_records": module_records,
        "observed_source_sha256": observed_sources,
        "failure_receipt": _file_record_for_control(failure_path),
        "loud_error": loud_error,
        "gpu_requested": False,
        "scorer_loaded": False,
        "payload_materialized": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_control_json(run_root / "PROBE_RESULT.json", result)
    auth_eval.auth_cache_vol.commit()
    return result


def _file_record_for_control(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


@app.local_entrypoint()
def probe_control_plane(
    output_receipt: str,
    run_id: str = "ddm_jo1u2_control_plane_probe_r1",
    diagnostic_authorization: bool = False,
) -> None:
    """Fire the one charter-authorized no-GPU diagnostic probe."""
    if not diagnostic_authorization:
        raise JO1DispatchError("control-plane probe requires diagnostic authorization")
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", run_id) is None:
        raise JO1DispatchError(f"control-plane probe run_id is invalid: {run_id!r}")
    output_path = Path(output_receipt).expanduser().resolve()
    if output_path.exists():
        raise JO1DispatchError(f"control-plane probe receipt already exists: {output_path}")
    source_paths = {
        "design": Path(design.__file__).resolve(),
        "worker": Path(worker.__file__).resolve(),
        "materializer_worker": Path(materializer_worker.__file__).resolve(),
        "retained_worker": Path(materializer_worker.retained.__file__).resolve(),
        "modal_auth_eval": Path(auth_eval.__file__).resolve(),
        "dispatcher": Path(__file__).resolve(),
    }
    expected_sources = {name: _file_sha256(path) for name, path in source_paths.items()}
    call = run_control_plane_probe.spawn(
        run_id=run_id,
        expected_source_sha256=expected_sources,
    )
    from tac.deploy.modal.auth_eval import function_call_id
    from tac.deploy.modal.call_id_ledger import (
        register_dispatched_call_id_fail_closed,
        update_call_id_outcome,
    )

    call_id = function_call_id(call)
    lane_id = "ddm_jo1u2_materializer_probe"
    register_dispatched_call_id_fail_closed(
        call_id=call_id,
        lane_id=lane_id,
        label="ddm_jo1u2_control_plane_probe",
        platform="modal",
        gpu="CPU",
        expected_axis="diagnostic_control_plane_no_scorer",
        recipe="experiments/ddm_jo1_modal_joint_objective.py::probe_control_plane",
        max_seconds=300,
        agent="codex:ddm_jo1u2",
        volume_name=design.AUTH_CACHE_VOLUME_NAME,
        volume_run_id=run_id,
    )
    started = time.monotonic()
    try:
        result = call.get(timeout=420)
    except Exception as error:
        failure = {
            "schema": "ddm_jo1u2_control_plane_probe_local.v1",
            "passed": False,
            "call_id": call_id,
            "lane_id": lane_id,
            "run_id": run_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_repr": repr(error),
            "elapsed_seconds": time.monotonic() - started,
            "score_claim": False,
        }
        update_call_id_outcome(
            call_id=call_id,
            status="failed",
            harvest_result=failure,
            rc=1,
            elapsed_seconds=failure["elapsed_seconds"],
            evidence_grade="diagnostic control-plane probe; no scorer",
            agent="codex:ddm_jo1u2",
            lane_id=lane_id,
            label="ddm_jo1u2_control_plane_probe",
            gpu="CPU",
        )
        design.atomic_json(output_path, failure)
        raise JO1DispatchError(
            f"control-plane probe failed; call_id={call_id}; {failure['error_repr']}"
        ) from error
    receipt = {
        "schema": "ddm_jo1u2_control_plane_probe_local.v1",
        "passed": bool(result.get("passed")),
        "call_id": call_id,
        "lane_id": lane_id,
        "run_id": run_id,
        "elapsed_seconds": time.monotonic() - started,
        "remote_result": result,
        "score_claim": False,
        "promotion_eligible": False,
    }
    update_call_id_outcome(
        call_id=call_id,
        status="harvested",
        harvest_result=receipt,
        rc=0,
        elapsed_seconds=receipt["elapsed_seconds"],
        evidence_grade="diagnostic control-plane probe; no scorer",
        agent="codex:ddm_jo1u2",
        lane_id=lane_id,
        label="ddm_jo1u2_control_plane_probe",
        gpu="CPU",
    )
    design.atomic_json(output_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def _blocked_entrypoint(
    *,
    entrypoint: str,
    compiled_config: str,
    expected_config_sha256: str,
    main_owned_dispatch_authorization: bool,
    detach: bool,
    provider_detach_ack: bool,
) -> None:
    request = dispatch_request(
        entrypoint=entrypoint,
        compiled_config=Path(compiled_config),
        expected_config_sha256=expected_config_sha256,
        main_owned_dispatch_authorization=main_owned_dispatch_authorization,
        detach=detach,
        provider_detach_ack=provider_detach_ack,
    )
    raise JO1DispatchError(
        f"{worker.TRAINING_IMPLEMENTATION_BLOCKER}; no claim or spawn performed; "
        f"request={json.dumps(request, sort_keys=True)}"
    )


@app.local_entrypoint()
def materialize_scorer_payloads(
    compiled_config: str,
    expected_config_sha256: str,
    main_owned_dispatch_authorization: bool = False,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    request = dispatch_request(
        entrypoint="materialize_scorer_payloads",
        compiled_config=Path(compiled_config),
        expected_config_sha256=expected_config_sha256,
        main_owned_dispatch_authorization=main_owned_dispatch_authorization,
        detach=detach,
        provider_detach_ack=provider_detach_ack,
    )
    config = design.load_compiled_config(
        Path(compiled_config), expected_config_sha256
    )
    call_id = _spawn_materializer(config, request)
    print(
        json.dumps(
            {
                "status": "DISPATCHED",
                "entrypoint": "materialize_scorer_payloads",
                "call_id": call_id,
                "remote_volume_name": request["remote_volume_name"],
                "remote_volume_run_id": request["remote_volume_run_id"],
                "harvest_root": request["harvest_root"],
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def memory_preflight(
    compiled_config: str,
    expected_config_sha256: str,
    main_owned_dispatch_authorization: bool = False,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    _blocked_entrypoint(
        entrypoint="memory_preflight",
        compiled_config=compiled_config,
        expected_config_sha256=expected_config_sha256,
        main_owned_dispatch_authorization=main_owned_dispatch_authorization,
        detach=detach,
        provider_detach_ack=provider_detach_ack,
    )


@app.local_entrypoint()
def train(
    compiled_config: str,
    expected_config_sha256: str,
    main_owned_dispatch_authorization: bool = False,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    _blocked_entrypoint(
        entrypoint="train",
        compiled_config=compiled_config,
        expected_config_sha256=expected_config_sha256,
        main_owned_dispatch_authorization=main_owned_dispatch_authorization,
        detach=detach,
        provider_detach_ack=provider_detach_ack,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = design.CompiledConfig.model_validate_json(args.author_config.read_text())
        result = prepare_local(config, args.output)
    except (OSError, ValueError, design.JO1Error, JO1DispatchError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_TO_FIRE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
