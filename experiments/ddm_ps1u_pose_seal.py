#!/usr/bin/env python3
"""Seal the ps1u dual-axis POSE request for the QS1 T4 transport.

WHAT THIS SEALS
---------------
The ps1u candidate: archive ``97048f9f…`` @183,347 B (base ``80d9c8c6…`` @182,759 B plus a
626 B ``P1D1`` frame-0 carrier-delta section over the 60 top-mass pairs; real archive delta
**+588 B**, because brotli absorbs part of the section). The container writer proved
byte-identity, parse-back and repeat-identity
(``CONTAINER_WRITER_PROOF.json`` → ``CONTAINER_WRITER_PROVEN``).

Adapted from the banked apparatus ``experiments/ddm_re1_pose_leg_seal.py`` — same request
schema, same transport, same placeholder law. Preparation only: no Modal spend, no scorer
forward, no claims. MAIN fires the emitted command.

THE SEG LEG IS ASSERTED, NOT MEASURED — AND THAT IS RECORDED AS SUCH
--------------------------------------------------------------------
re1 could populate ``seg_leg_provenance`` from a real RE1T seg run. **ps1u has no such run**,
so this module MUST NOT borrow one. It writes ``re1t_run_id =
"NONE_ps1u_seg_asserted_decode_identical"`` and ``seg_delta_s_exact_t4_field = 0.0`` with an
explicit basis string. The assertion is sound in mechanism — the edit touches only the frame-0
carrier and SegNet reads the LAST frame — but it is an ASSERTION, and any T4 seg drift is
signal, not noise.

CONSUMPTION AUDIT (done at source before writing this, per the STOP clause)
--------------------------------------------------------------------------
``seg_leg_provenance`` is **never read by any worker**: it appears only in the seal builders.
The worker enforces the placeholder law on ``local_pose_delta`` / ``pose_unmeasured``
(``ddm_re1t_modal_t4_sign_gate.py:663-665``) and computes ``seg_delta_s_exact_t4_field`` and
``rate_delta_s_exact_archive`` ITSELF from the real archive (``:704``, ``:706``). So the
provenance block is metadata and cannot misstate the worker's output arithmetic. Had it been
consumed numerically, this module would have refused to ship rather than emit a wrong number.

AXIS ``[preparation only — no dispatch, no score]``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Final

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1bw
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1bm

REPO: Final = Path(__file__).resolve().parents[1]
STORE: Final = Path("/Volumes/APDataStore/pact/ddm_ps1u_uncapped_pose_20260816")
CANDIDATE_ARCHIVE: Final = STORE / "retained/writer/candidate_build1.zip"
CANDIDATE_GENERATION: Final = STORE / "retained/candidate_generation"
WRITER_PROOF: Final = STORE / "CONTAINER_WRITER_PROOF.json"
RECEIVER_CONTRACT: Final = STORE / "RECEIVER_P1D1_CONTRACT.json"
DECODE_VERDICT: Final = STORE / "DECODE_AXIS_VERDICT.json"
FIRE_ORDER: Final = STORE / "SEALED_T4_FIRE_ORDER.json"
OUTPUT: Final = STORE / "dual_axis_pose"

RUN_ID: Final = "ddm_ps1u_dual_axis_pose_20260816_r1"
LANE_ID: Final = "ddm_ps1u_dual_axis_pose_n600_20260816"
RUNTIME_LABEL: Final = "ps1u_p1d1_candidate"
EXPECTED_CANDIDATE_SHA: Final = (
    "97048f9fe1845a2b0b602dbdaf5f85e87fb19dee0e6cc57503fe5fd60096bef8"
)
EXPECTED_CANDIDATE_BYTES: Final = 183_347
BASE_ARCHIVE_SHA: Final = (
    "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
)
RATE_DELTA_S_EXACT: Final = 0.00039152506443583535
SEG_BASIS: Final = (
    "frame-0-only edit; SegNet reads frame_1; T4 drift is signal"
)


class PS1USealError(RuntimeError):
    """The retained ps1u custody chain does not support this seal."""


def _require_proofs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    proof = json.loads(WRITER_PROOF.read_text(encoding="utf-8"))
    contract = json.loads(RECEIVER_CONTRACT.read_text(encoding="utf-8"))
    order = json.loads(FIRE_ORDER.read_text(encoding="utf-8"))
    if proof.get("verdict") != "CONTAINER_WRITER_PROVEN":
        raise PS1USealError("container writer is not proven")
    for leg in ("byte_identity_control", "repeat_identical", "parse_back"):
        if not proof[leg]["passed"]:
            raise PS1USealError(f"container writer proof leg failed: {leg}")
    if contract.get("verdict") != "RECEIVER_CONTRACT_VERIFIED":
        raise PS1USealError("receiver P1D1 contract is not verified")
    if order.get("status") != "SEALED":
        raise PS1USealError("ps1u fire order is not sealed")
    if order["candidate"]["archive_sha256"] != EXPECTED_CANDIDATE_SHA:
        raise PS1USealError("fire order candidate differs from the sealed archive")
    return proof, contract, order


def seal() -> dict[str, Any]:
    proof, contract, order = _require_proofs()

    archive_record = js1bm.file_record(CANDIDATE_ARCHIVE)
    if (
        archive_record["sha256"] != EXPECTED_CANDIDATE_SHA
        or archive_record["bytes"] != EXPECTED_CANDIDATE_BYTES
    ):
        raise PS1USealError("candidate archive differs from the proven writer output")

    # The runtime bundle EXCLUDES the embedded base archive.zip; the worker installs
    # OUR candidate as runtime_root/archive.zip via install_archive().
    runtime_bytes, runtime_manifest = js1bm.build_runtime_bundle(
        CANDIDATE_GENERATION, label=RUNTIME_LABEL
    )

    decode = json.loads(DECODE_VERDICT.read_text(encoding="utf-8"))
    screen_payload = js1bm.canonical_json_bytes(
        {
            "schema": "ddm_ps1u_pose_screen_evidence.v1",
            "role": (
                "ps1u evidence for the dual-axis pose measurement; pose is UNMEASURED "
                "locally by the worker placeholder law"
            ),
            "local_pose_delta": 0.0,
            "pose_unmeasured": True,
            "score_claim": False,
            "candidate": {
                "archive_sha256": EXPECTED_CANDIDATE_SHA,
                "archive_bytes": EXPECTED_CANDIDATE_BYTES,
                "base_archive_sha256": BASE_ARCHIVE_SHA,
                "delta_bytes": EXPECTED_CANDIDATE_BYTES - 182_759,
                "section_format": "P1D1",
                "section_bytes": proof["section_bytes"],
                "section_sha256": proof["section_sha256"],
                "pairs_edited": 60,
            },
            "container_writer_proof": {
                "verdict": proof["verdict"],
                "byte_identity_control": proof["byte_identity_control"]["passed"],
                "parse_back": proof["parse_back"]["passed"],
                "repeat_identical": proof["repeat_identical"]["passed"],
                "file": js1bm.file_record(WRITER_PROOF),
            },
            "receiver_contract": {
                "verdict": contract["verdict"],
                "file": js1bm.file_record(RECEIVER_CONTRACT),
            },
            "DEVICE_DEPENDENT_DECODE_WARNING": {
                "verdict": decode["verdict"],
                "cpu_raw_sha256": decode["cpu"]["raw_sha256"],
                "cuda_raw_sha256": decode["cuda"]["raw_sha256"],
                "meaning": (
                    "the CPU and CUDA inflate paths emit DIFFERENT frames from the same "
                    "archive; the local advisory pose reduction was solved against the "
                    "CPU-decode object and its CUDA-axis value is UNMEASURED. This row "
                    "measures the transfer. The advisory row cannot admit."
                ),
                "file": js1bm.file_record(DECODE_VERDICT),
            },
            "sealed_fire_order_record": js1bm.file_record(FIRE_ORDER),
            "pre_registered_admission": order["PRE_REGISTERED_ADMISSION"],
        }
    )

    payloads = {
        "candidate_archive.zip": CANDIDATE_ARCHIVE.read_bytes(),
        "candidate_runtime.zip": runtime_bytes,
        "POSE_SCREEN_RESULT.json": screen_payload,
    }
    input_root = OUTPUT / "fire_inputs"
    for name, payload in payloads.items():
        js1bm.atomic_bytes(input_root / name, payload)
    js1bw.require_record(
        input_root / "candidate_archive.zip", js1bm.payload_record(payloads["candidate_archive.zip"])
    )

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
        "candidate_runtime": js1bm.payload_record(runtime_bytes),
        "runtime_manifest": runtime_manifest,
        "inputs": {
            name: js1bm.payload_record(payload) for name, payload in payloads.items()
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "scorer_input_cache_hashes_requested": True,
        "scorer_input_cache_hash_batch_pairs": 8,
        "seg_leg_provenance": {
            "re1t_run_id": "NONE_ps1u_seg_asserted_decode_identical",
            "seg_delta_s_exact_t4_field": 0.0,
            "seg_delta_basis": SEG_BASIS,
            "seg_leg_measured": False,
            "rate_delta_s_exact_archive": RATE_DELTA_S_EXACT,
            "verdict_scope": order["PRE_REGISTERED_ADMISSION"].get(
                "verdict_scope", "instance: this candidate on this vehicle"
            ),
            "consumption_audit": (
                "seg_leg_provenance is metadata only — no worker reads it; the worker "
                "computes seg_delta_s_exact_t4_field and rate_delta_s_exact_archive itself "
                "from the real archive (ddm_re1t_modal_t4_sign_gate.py:704,:706)"
            ),
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

    # Validate the seal through the REAL dispatcher, in-process, before emitting anything.
    from experiments import ddm_qs1_modal_t4_dual_axis as dispatcher

    dispatcher.load_sealed_inputs(request_path, input_root, request_record["sha256"])

    command = [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_qs1_modal_t4_dual_axis.py::main",
        "--sealed-request",
        str(request_path.resolve()),
        "--fire-input-dir",
        str(input_root.resolve()),
        "--expected-request-sha256",
        request_record["sha256"],
        "--output-dir",
        str((OUTPUT / "dispatch" / RUN_ID).resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order_out = {
        "schema": "ddm_ps1u_pose_leg_sealed_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(OUTPUT.resolve()),
        "fire_trigger": (
            "MAIN confirms no active Modal scorer lane, the toy gate passes on this store, "
            "and every sealed SHA verifies"
        ),
        "fresh_run_id": RUN_ID,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "candidate": {
            "archive_sha256": EXPECTED_CANDIDATE_SHA,
            "archive_bytes": EXPECTED_CANDIDATE_BYTES,
            "delta_bytes_vs_base": EXPECTED_CANDIDATE_BYTES - 182_759,
        },
        "pre_registered_admission": order["PRE_REGISTERED_ADMISSION"],
        "post_harvest_rule": (
            "recompute S from measured components on the exact archive bytes; ADMIT only "
            "when S < 0.15959729295498598 with the pose leg MEASURED. The seg leg is "
            "ASSERTED decode-identical (0.029611) — any measured seg drift is SIGNAL and "
            "must be priced, not absorbed. The advisory row cannot admit."
        ),
        "piggyback_decode_localization": order["piggyback_decode_localization"],
        "dispatcher_validation_passed": True,
        "modal_fired": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    js1bm.atomic_json(OUTPUT / "SEALED_FIRE_ORDER.json", order_out)
    return order_out


def main() -> int:
    order_out = seal()
    print(json.dumps(order_out, indent=2, sort_keys=True))
    print("\nMAIN fires (this module dispatched nothing):\n")
    print(" ".join(order_out["exact_command_argv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
