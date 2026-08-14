#!/usr/bin/env python3
"""Seal the RE1 Round-1 dual-axis POSE-leg request for the QS1 T4 transport.

The RE1T r2 dispatch measured the seg leg of the RE1 Round-1 candidate
(archive 7be3eb94..., 186,252 B) on the frozen T4 SegNet field:
seg delta S = -1.6954210069444444e-06 at rate delta S = 0.0, verdict
PROVISIONALLY_ADMITTED_SEG_SIGN_GATE_POSE_MEASUREMENT_REQUIRED with
pose_job_may_fire = true.  This module builds the hash-sealed
``ddm_qs1_t4_dual_axis_request.v1`` that buys the missing pose leg on the
SAME candidate bytes through the SAME worker family (retain_pose_vectors
mode), honoring the worker placeholder law (local_pose_delta must be the
literal 0.0 with pose_unmeasured true).

Preparation only: no Modal spend, no scorer forward, no claims.  MAIN
fires the emitted command after the toy gate and single-flight checks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Final

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1bw
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1bm

REPO: Final = Path(__file__).resolve().parents[1]
STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race"
    "/ddm_re1_20260813/full_n600_exact/round_01_singleton_best"
)
RE1T_REQUEST: Final = STORE / "RE1T_T4_REQUEST.json"
RE1T_FIRE_INPUTS: Final = STORE / "re1t_t4_fire_inputs"
SEG_ADJUDICATION: Final = STORE / "re1t_t4_dispatch/LOCAL_ADJUDICATION.json"
OUTPUT: Final = STORE / "re1_dual_axis_pose"
RUN_ID: Final = "ddm_re1_dual_axis_pose_20260814_r1"
LANE_ID: Final = "ddm_re1_dual_axis_pose_n600_20260814"
EXPECTED_CANDIDATE_SHA: Final = (
    "7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa"
)
EXPECTED_CANDIDATE_BYTES: Final = 186_252


class RE1PoseSealError(RuntimeError):
    """The retained seg-leg custody chain does not support this seal."""


def _require_seg_gate(adjudication: dict[str, Any]) -> None:
    if adjudication.get("status") != (
        "PROVISIONALLY_ADMITTED_SEG_SIGN_GATE_POSE_MEASUREMENT_REQUIRED"
    ):
        raise RE1PoseSealError("seg sign-gate adjudication is not provisionally admitted")
    if adjudication.get("pose_job_may_fire") is not True:
        raise RE1PoseSealError("seg adjudication does not authorize the pose leg")
    if adjudication.get("local_pose_delta_placeholder") != 0.0:
        raise RE1PoseSealError("seg adjudication lost the pose placeholder law")


def seal() -> dict[str, Any]:
    re1t_request = json.loads(RE1T_REQUEST.read_text(encoding="utf-8"))
    adjudication = json.loads(SEG_ADJUDICATION.read_text(encoding="utf-8"))
    _require_seg_gate(adjudication)

    archive_path = RE1T_FIRE_INPUTS / "candidate_archive.zip"
    runtime_path = RE1T_FIRE_INPUTS / "candidate_runtime.zip"
    js1bw.require_record(archive_path, re1t_request["inputs"]["candidate_archive.zip"])
    js1bw.require_record(runtime_path, re1t_request["inputs"]["candidate_runtime.zip"])
    archive_record = js1bm.file_record(archive_path)
    if (
        archive_record["sha256"] != EXPECTED_CANDIDATE_SHA
        or archive_record["bytes"] != EXPECTED_CANDIDATE_BYTES
    ):
        raise RE1PoseSealError("candidate archive differs from the seg-admitted object")

    screen_payload = js1bm.canonical_json_bytes(
        {
            "schema": "ddm_re1_pose_screen_evidence.v1",
            "role": (
                "seg-leg evidence for the dual-axis pose measurement; pose is "
                "UNMEASURED locally by the worker placeholder law"
            ),
            "seg_adjudication": adjudication,
            "seg_adjudication_file": js1bm.file_record(SEG_ADJUDICATION),
            "local_pose_delta": 0.0,
            "pose_unmeasured": True,
            "score_claim": False,
        }
    )
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_path.read_bytes(),
        "POSE_SCREEN_RESULT.json": screen_payload,
    }
    input_root = OUTPUT / "fire_inputs"
    for name, payload in payloads.items():
        js1bm.atomic_bytes(input_root / name, payload)

    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": RUN_ID,
        "resume_from": RUN_ID,
        "lane_id": LANE_ID,
        "instance_job_id": f"modal:{RUN_ID}",
        "claim_agent": "MAIN",
        "seed": 1234,
        "batch_size": 16,
        "retain_pose_vectors": True,
        "candidate_archive": archive_record,
        "candidate_runtime": re1t_request["candidate_runtime"],
        "runtime_manifest": re1t_request["runtime_manifest"],
        "inputs": {
            name: js1bm.payload_record(payload) for name, payload in payloads.items()
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "seg_leg_provenance": {
            "re1t_run_id": str(re1t_request["run_id"]),
            "seg_delta_s_exact_t4_field": adjudication["seg_delta_s_exact_t4_field"],
            "rate_delta_s_exact_archive": adjudication["rate_delta_s_exact_archive"],
            "verdict_scope": adjudication["verdict_scope"],
        },
        "source_git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": js1bm.sha256_bytes(git_status),
        "dispatcher_source_sha256": js1bm.sha256_file(
            REPO / "experiments/ddm_qs1_modal_t4_dual_axis.py"
        ),
        "worker_source_sha256": js1bm.sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": js1bm.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_path = OUTPUT / "SEALED_REQUEST.json"
    js1bm.atomic_json(request_path, request)
    request_record = js1bm.file_record(request_path)

    from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher

    dispatcher.load_sealed_inputs(request_path, input_root, request_record["sha256"])
    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        str(request_path),
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str((OUTPUT / "dispatch" / RUN_ID).resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_re1_pose_leg_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(OUTPUT.resolve()),
        "fire_trigger": (
            "MAIN confirms no active Modal scorer lane, the toy gate passes on this "
            "store, and every sealed SHA verifies"
        ),
        "fresh_run_id": RUN_ID,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "remote_scope": (
            "one candidate; unchanged worker retains n600 T4 Seg field, official Pose "
            "first-six vectors for GT plus two candidate passes, inputs, outputs, and "
            "the deterministic repeat"
        ),
        "post_harvest_rule": (
            "compose pose delta S with the banked seg leg -1.6954210069444444e-06 at "
            "rate 0.0; admit RE1 Round-1 only when the realized matched-instrument "
            "NET delta S < 0; the pose leg is a measurement, never a score claim"
        ),
        "dispatcher_validation_passed": True,
        "modal_fired": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    js1bm.atomic_json(OUTPUT / "SEALED_FIRE_ORDER.json", order)
    js1bm.atomic_json(OUTPUT / "checkpoints/stage_50_sealed_order.json", order)
    return order


def main() -> int:
    order = seal()
    print(json.dumps({"request": order["request"], "argv": order["exact_command_argv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
