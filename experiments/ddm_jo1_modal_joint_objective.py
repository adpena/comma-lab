#!/usr/bin/env python3
"""MAIN-owned JO1 dispatch gate.

Local preparation is pure and scorer-free.  The three named Modal entrypoints
exist so the sealed fire order is stable, but this no-launch build refuses them
before claiming a lane because the rc2 fresh-Schur receiver-close backend is
not yet implemented.  That refusal is intentional: a dispatch surface that
cannot perform its named work must not spawn a paid job.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import modal

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo1_joint_objective_worker as worker

APP_NAME = "comma-ddm-jo1-joint-objective"
LANE_LABEL = "ddm_jo1_joint_objective"
app = modal.App(APP_NAME)


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
    readiness = design.readiness(config)
    # Scorer materialization is the prerequisite that may run while readiness
    # is blocked on missing fields.  Memory and train require their predecessor.
    if entrypoint == "memory_preflight" and any(
        "FIELD" in blocker or "POSE6" in blocker or "TOKENS" in blocker
        for blocker in readiness["blockers"]
    ):
        raise JO1DispatchError("memory preflight is ordered after scorer payload harvest")
    if entrypoint == "train" and readiness["status"] != "READY_TO_FIRE":
        raise JO1DispatchError(f"training readiness is blocked: {readiness['blockers']}")
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
    The current entrypoints never call it because their backend closure is
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
        base_archive_sha256=design.RC2_ARCHIVE_SHA256,
        composed_archive_sha256=design.RC2_ARCHIVE_SHA256,
        archive_count=1,
        volume_name="pact-jo1-artifacts",
        volume_run_id=request["workload_config_sha256"],
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
    _blocked_entrypoint(
        entrypoint="materialize_scorer_payloads",
        compiled_config=compiled_config,
        expected_config_sha256=expected_config_sha256,
        main_owned_dispatch_authorization=main_owned_dispatch_authorization,
        detach=detach,
        provider_detach_ack=provider_detach_ack,
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
