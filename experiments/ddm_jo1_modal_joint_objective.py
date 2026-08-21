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
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import modal

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo1_joint_objective_worker as worker
from experiments import ddm_jo1_payload_materializer_worker as materializer_worker
from experiments import modal_auth_eval as auth_eval
from tac.deploy.modal.auth_eval import prepare_modal_auth_eval_request

APP_NAME = "comma-ddm-jo1-joint-objective"
LANE_LABEL = "ddm_jo1_joint_objective"
app = modal.App(APP_NAME, include_source=False)
materializer_image = (
    auth_eval.eval_image
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
    run_root = auth_eval.AUTH_CACHE_VOLUME_ROOT / str(request["remote_volume_run_id"])
    try:
        materializer_worker.stage_uploaded_inputs(
            run_root=run_root,
            request=request,
            archive_bytes=archive_bytes,
            runtime_zip_bytes=runtime_zip_bytes,
        )
        result = materializer_worker.run(run_root, str(request["resume_from"]))
    except Exception as error:
        materializer_worker.retained.atomic_json(
            run_root / "REMOTE_FAILURE.json",
            {
                "schema": "ddm_jo1_payload_materializer_failure.v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "retention_policy": (
                    "no cleanup performed; every input or intermediate written "
                    "before failure remains on the mounted volume"
                ),
                "complete_payload_set": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
        )
        auth_eval.auth_cache_vol.commit()
        raise
    auth_eval.auth_cache_vol.commit()
    return result


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
