#!/usr/bin/env python3
"""MAIN-only Modal transport for a sealed QS1 dual-axis measurement.

Preparation happens in ``ddm_qs1_frame0_schur_coupled_solve.py``.  This module
accepts only its hash-sealed request, claims the named single-flight lane, and
runs the extended RE1T worker with Pose-vector retention enabled.  It never
adjudicates or promotes remotely.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import modal

try:
    from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1b
except (ImportError, ModuleNotFoundError):
    import ddm_js1b_cuda_argmax_field_materializer_worker as js1b  # type: ignore[no-redef]

try:
    from experiments import ddm_re1t_modal_t4_sign_gate as substrate
except (ImportError, ModuleNotFoundError):
    import ddm_re1t_modal_t4_sign_gate as substrate  # type: ignore[no-redef]
from tac.deploy.dispatch_axis_screen import assert_distortion_axis_locally_screened
from tac.deploy.modal.auth_eval import (
    ClaimSpec,
    claim_modal_auth_eval_dispatch,
    function_call_id,
    terminal_modal_auth_eval_claim,
)
from tac.deploy.modal.call_id_ledger import (
    register_dispatched_call_id_fail_closed,
    update_call_id_outcome,
)
from tac.deploy.modal.single_flight import assert_modal_single_flight

REPO: Final = Path(__file__).resolve().parents[1]
APP_NAME: Final = "ddm-qs1-dual-axis"
VOLUME_NAME: Final = substrate.VOLUME_NAME
VOLUME_ROOT: Final = substrate.VOLUME_ROOT
REMOTE_WORKER: Final = substrate.REMOTE_WORKER
LANE_LABEL: Final = "ddm_qs1_dual_axis_n600"
AXIS: Final = (
    "[contest-CUDA T4 frozen-SegNet field + PoseNet first6 vectors, n600] COMPONENT-ONLY"
)


class QS1DispatchError(RuntimeError):
    """The sealed request, retained input, lane, or remote return differed."""


def _parse_pose_screen_evidence(payload: bytes | None) -> dict[str, Any] | None:
    """Parse the sealed pose-screen evidence, fail-closed on anything unusable.

    An unparseable or non-object evidence file yields ``None``, which makes the
    axis-screen census treat pose as UNSCREENED — the safe direction.
    """
    if not payload:
        return None
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def load_sealed_inputs(
    sealed_request: Path, fire_input_dir: Path, expected_request_sha256: str
) -> tuple[dict[str, bytes], dict[str, Any]]:
    request_path = sealed_request.resolve()
    if js1b.sha256_file(request_path) != expected_request_sha256:
        raise QS1DispatchError("sealed request SHA-256 differs")
    request = json.loads(request_path.read_text())
    if (
        request.get("schema") != "ddm_qs1_t4_dual_axis_request.v1"
        or request.get("resume_from") != request.get("run_id")
        or request.get("retain_pose_vectors") is not True
        or request.get("score_claim") is not False
        or request.get("promotion_eligible") is not False
    ):
        raise QS1DispatchError("sealed request contract differs")
    expected_names = {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "POSE_SCREEN_RESULT.json",
    }
    if set(request.get("inputs", {})) != expected_names:
        raise QS1DispatchError("sealed request input census differs")
    payloads: dict[str, bytes] = {}
    for name in sorted(expected_names):
        path = fire_input_dir.resolve() / name
        js1b.require_record(path, request["inputs"][name])
        payloads[name] = path.read_bytes()
    # A paid row may not fire when EVERY distortion axis is an assertion; the
    # ps1u r2 REFUSE (+1.686e-02 S) was bought with an unscreened pose leg.
    # Fires here, so it refuses at BOTH in-process seal validation and at the
    # real `modal run` entrypoint below.
    assert_distortion_axis_locally_screened(
        request, _parse_pose_screen_evidence(payloads.get("POSE_SCREEN_RESULT.json"))
    )
    return payloads, request


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = substrate.gate_image.add_local_python_source(
    "ddm_qs1_modal_t4_dual_axis",
    "ddm_js1b_cuda_argmax_field_materializer_worker",
)


@app.function(
    image=image,
    gpu="T4",
    timeout=int(substrate.CONTEST_LIMIT_SECONDS),
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_dual_axis(
    payloads: dict[str, bytes], request: dict[str, Any]
) -> dict[str, Any]:
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    remote_inputs = {
        **payloads,
        "REQUEST.json": js1b.canonical_json_bytes(request),
    }
    for name, payload in remote_inputs.items():
        path = run_root / "inputs" / name
        if path.is_file():
            if js1b.sha256_file(path) != js1b.sha256_bytes(payload):
                raise QS1DispatchError(f"resume input differs: {path}")
        else:
            js1b.atomic_bytes(path, payload)
    retained_volume.commit()
    command = [
        f"{substrate.UPSTREAM_LOCKED_VENV}/bin/python",
        "-u",
        str(REMOTE_WORKER),
        "--run-root",
        str(run_root),
        "--resume-from",
        run_id,
        "--retain-pose-vectors",
    ]
    log_path = run_root / "worker.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        while process.poll() is None:
            time.sleep(substrate.COMMIT_PERIOD_SECONDS)
            retained_volume.commit()
        returncode = process.wait()
    retained_volume.commit()
    retained_volume.commit()
    final_path = run_root / "FINAL_RESULT.json"
    if returncode or not final_path.is_file():
        return {
            "schema": "ddm_qs1_t4_modal_return.v1",
            "measurement_complete": False,
            "returncode": returncode,
            "run_id": run_id,
            "volume_name": VOLUME_NAME,
            "volume_path": str(run_root),
            "worker_log_tail": log_path.read_text(errors="replace")[-8_000:],
            "score_claim": False,
        }
    final = final_path.read_bytes()
    return {
        "schema": "ddm_qs1_t4_modal_return.v1",
        "measurement_complete": True,
        "returncode": 0,
        "axis": AXIS,
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "final_result_sha256": js1b.sha256_bytes(final),
        "artifacts": {"QS1_T4_REMOTE_RESULT.json": final},
        "score_claim": False,
        "promotion_eligible": False,
    }


def _persist_return(output: Path, result: dict[str, Any]) -> None:
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise QS1DispatchError(f"unsafe returned artifact: {name!r}")
        js1b.atomic_bytes(output / name, payload)
    js1b.atomic_json(output / "modal_qs1_t4_result.json", result)


def recover(output_dir: Path, timeout_seconds: float = 0.0) -> int:
    output = output_dir.resolve()
    spawn = json.loads((output / "modal_auth_eval_spawn.json").read_text())
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(
            timeout=timeout_seconds
        )
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}))
        return 4
    if not isinstance(result, dict):
        raise QS1DispatchError("remote return is not a dictionary")
    complete = bool(result.get("measurement_complete"))
    _persist_return(output, result)
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if complete else "failed",
        rc=0 if complete else 1,
        score_axis="contest_cuda_qs1_dual_axis_component",
        evidence_grade=AXIS,
        lane_id=str(spawn["lane_id"]),
        label=LANE_LABEL,
        gpu="T4",
        agent=str(spawn["claim_agent"]),
        harvest_result={"measurement_complete": complete},
    )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(spawn["lane_id"]),
            instance_job_id=str(spawn["instance_job_id"]),
            agent=str(spawn["claim_agent"]),
            force=True,
        ),
        status="completed_qs1_dual_axis_recovered" if complete else "failed_qs1_dual_axis_recovered",
        notes=f"QS1 dual-axis return harvested; call_id={call_id}; output={output}",
    )
    return 0 if complete else 1


@app.local_entrypoint()
def main(
    sealed_request: str,
    fire_input_dir: str,
    expected_request_sha256: str,
    output_dir: str,
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    if detach and not provider_detach_ack:
        raise SystemExit("FATAL: --detach requires --provider-detach-ack")
    payloads, request = load_sealed_inputs(
        Path(sealed_request), Path(fire_input_dir), expected_request_sha256
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    js1b.atomic_json(output / "REQUEST.json", request)
    spec = ClaimSpec(
        lane_id=str(request["lane_id"]),
        instance_job_id=str(request["instance_job_id"]),
        agent=str(request["claim_agent"]),
        notes=f"QS1 one-candidate dual-axis T4 measurement; run={request['run_id']}",
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO, spec=spec, status="active_qs1_dual_axis_spawning"
    )
    assert_modal_single_flight(
        label=LANE_LABEL, lane_id=str(request["lane_id"]), repo_root=REPO
    )
    if detach:
        call = run_dual_axis.spawn(payloads, request)
        call_id = function_call_id(call)
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=str(request["lane_id"]),
            label=LANE_LABEL,
            platform="modal",
            gpu="T4",
            expected_axis="contest_cuda_qs1_dual_axis_component",
            recipe="experiments/ddm_qs1_modal_t4_dual_axis.py::main",
            max_seconds=int(substrate.CONTEST_LIMIT_SECONDS),
            agent=str(request["claim_agent"]),
            base_archive_sha256=substrate.BASE_SHA256,
            composed_archive_sha256=request["candidate_archive"]["sha256"],
            archive_count=1,
            volume_name=VOLUME_NAME,
            volume_run_id=str(request["run_id"]),
        )
        js1b.atomic_json(
            output / "modal_auth_eval_spawn.json",
            {
                "call_id": call_id,
                "lane_id": request["lane_id"],
                "instance_job_id": request["instance_job_id"],
                "claim_agent": request["claim_agent"],
                "run_id": request["run_id"],
            },
        )
        print(f"DISPATCHED call_id={call_id}")
        print(
            f"RECOVER .venv/bin/python {__file__} recover --output-dir {output}"
        )
        return
    result = run_dual_axis.remote(payloads, request)
    _persist_return(output, result)
    complete = bool(result.get("measurement_complete"))
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(request["lane_id"]),
            instance_job_id=str(request["instance_job_id"]),
            agent=str(request["claim_agent"]),
            force=True,
        ),
        status="completed_qs1_dual_axis_sync" if complete else "failed_qs1_dual_axis_sync",
        notes=f"QS1 synchronous dual-axis return; output={output}",
    )
    if not complete:
        raise SystemExit("FATAL: QS1 T4 worker failed; resume the same run id")


def _main_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "recover":
        return recover(args.output_dir, args.timeout_seconds)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main_cli())
