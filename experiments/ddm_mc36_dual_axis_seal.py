#!/usr/bin/env python3
"""Build the dispatcher-conformant MC36 Variant-C dual-axis fire seal.

The MC36 arm retained a real candidate archive and runtime bundle, but its
``SEALED_REQUEST.json`` is fire-order metadata rather than the schema consumed
by :mod:`experiments.ddm_qs1_modal_t4_dual_axis`.  This preparation-only module
copies the already pinned payloads into a new immutable-looking seal directory,
adds the local advisory evidence under the worker's explicit Pose-unknown
placeholder law, and proves acceptance through the real dispatcher loader.

No scorer, Modal call, promotion, or score claim is performed here.  MAIN owns
the emitted fire order.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1bw
from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1bm

REPO: Final = _REPO
STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105"
)
SOURCE_FIRE_ORDER: Final = STORE / "fire_order"
SOURCE_ADVISORY: Final = STORE / "LOCAL_ADVISORY_RECOUNT.json"
SOURCE_COMPILED: Final = STORE / "COMPILED_ARCHIVE.json"
OUTPUT: Final = STORE / "dispatcher_conformant_seal"
RUN_ID: Final = "ddm_mc36_dual_axis_t4_r1"
LANE_ID: Final = "ddm_mc36_dual_axis_t4_n600_20260814"
EXPECTED_CANDIDATE_SHA: Final = (
    "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
)
EXPECTED_CANDIDATE_BYTES: Final = 186_269
EXPECTED_RUNTIME_SHA: Final = (
    "64e4642d30b436e6393d5573efcb579a13f922726566790efad40bc2ca117545"
)
EXPECTED_RUNTIME_BYTES: Final = 238_713
EXPECTED_NET_FLIP_GAIN: Final = 37
EXPECTED_DELTA_BYTES: Final = 17
EXPECTED_DELTA_DPOSE: Final = -1.4632967835484165e-10


class MC36DualAxisSealError(RuntimeError):
    """The retained MC36 candidate does not support the named fire seal."""


def _require_source_evidence(
    source_order: dict[str, Any], advisory: dict[str, Any]
) -> None:
    if (
        source_order.get("schema") != "ddm_mc36_dual_axis_fire_order.v1"
        or source_order.get("variant") != "successor_drop532_pair105"
        or source_order.get("remote_dispatched") is not False
    ):
        raise MC36DualAxisSealError("source fire order identity differs")
    archive = source_order.get("candidate_archive", {})
    runtime = source_order.get("candidate_runtime", {})
    if (
        archive.get("sha256") != EXPECTED_CANDIDATE_SHA
        or archive.get("bytes") != EXPECTED_CANDIDATE_BYTES
        or runtime.get("sha256") != EXPECTED_RUNTIME_SHA
        or runtime.get("bytes") != EXPECTED_RUNTIME_BYTES
    ):
        raise MC36DualAxisSealError("source fire-order payload pins differ")
    if advisory.get("all_gates_passed") is not True:
        raise MC36DualAxisSealError("MC36 local advisory gates are not all passed")
    if (
        advisory.get("seg", {}).get("net_flip_gain") != EXPECTED_NET_FLIP_GAIN
        or advisory.get("rate", {}).get("delta_bytes") != EXPECTED_DELTA_BYTES
        or advisory.get("pose", {}).get("delta_dpose") != EXPECTED_DELTA_DPOSE
    ):
        raise MC36DualAxisSealError("MC36 local advisory triple differs")
    if (
        advisory.get("score_claim") is not False
        or advisory.get("promotion_eligible") is not False
    ):
        raise MC36DualAxisSealError("local advisory crossed its authority boundary")


def _runtime_manifest(
    compiled: dict[str, Any], runtime_payload: bytes
) -> dict[str, Any]:
    runtime_tree = compiled.get("runtime_tree")
    if not isinstance(runtime_tree, dict) or not runtime_tree.get("files"):
        raise MC36DualAxisSealError("compiled receipt has no runtime tree")
    return {
        "schema": "ddm_mc36_deterministic_runtime_bundle.v1",
        "label": "ddm_mc36_successor_drop532_pair105",
        "source_root": runtime_tree["root"],
        "file_count": runtime_tree["file_count"],
        "files": [
            {
                "relative_path": row["relative_path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
            }
            for row in runtime_tree["files"]
        ],
        "includes_candidate_archive": True,
        "bundle": js1bm.payload_record(runtime_payload),
        "builder": "experiments.ddm_mc35_micro35_union_build:deterministic_runtime_zip",
    }


def seal() -> dict[str, Any]:
    source_order_path = STORE / "SEALED_FIRE_ORDER.json"
    source_order = json.loads(source_order_path.read_text(encoding="utf-8"))
    advisory = json.loads(SOURCE_ADVISORY.read_text(encoding="utf-8"))
    compiled = json.loads(SOURCE_COMPILED.read_text(encoding="utf-8"))
    _require_source_evidence(source_order, advisory)

    archive_path = SOURCE_FIRE_ORDER / "fire_inputs/candidate_archive.zip"
    runtime_path = SOURCE_FIRE_ORDER / "fire_inputs/candidate_runtime.zip"
    js1bw.require_record(archive_path, source_order["candidate_archive"])
    js1bw.require_record(runtime_path, source_order["candidate_runtime"])
    js1bm.require_exact(
        archive_path,
        size=EXPECTED_CANDIDATE_BYTES,
        digest=EXPECTED_CANDIDATE_SHA,
        label="MC36 Variant-C archive",
    )
    js1bm.require_exact(
        runtime_path,
        size=EXPECTED_RUNTIME_BYTES,
        digest=EXPECTED_RUNTIME_SHA,
        label="MC36 Variant-C runtime",
    )

    advisory_payload = js1bm.canonical_json_bytes(
        {
            "schema": "ddm_mc36_pose_screen_evidence.v1",
            "role": (
                "local advisory evidence for a fresh dual-axis T4 measurement; "
                "Pose is UNMEASURED by the worker placeholder law"
            ),
            "axis": advisory["axis"],
            "selection_mode": advisory["selection_mode"],
            "local_advisory": {
                "net_seg_flip_gain": advisory["seg"]["net_flip_gain"],
                "delta_archive_bytes": advisory["rate"]["delta_bytes"],
                "delta_dpose": advisory["pose"]["delta_dpose"],
                "projected_delta_s": advisory["projected_delta_s"],
            },
            "base_instrument": {
                "seg_flips": advisory["seg"]["base_flips"],
                "dpose_recomputed": advisory["pose"]["base_dpose_recomputed"],
                "archive_bytes": advisory["rate"]["base_archive_bytes"],
            },
            "local_advisory_file": js1bm.file_record(SOURCE_ADVISORY),
            "local_pose_delta": 0.0,
            "pose_unmeasured": True,
            "score_claim": False,
            "promotion_eligible": False,
        }
    )
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_path.read_bytes(),
        "POSE_SCREEN_RESULT.json": advisory_payload,
    }
    input_root = OUTPUT / "fire_inputs"
    for name, payload in payloads.items():
        js1bm.atomic_bytes(input_root / name, payload)

    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    runtime_payload = payloads["candidate_runtime.zip"]
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
        "candidate_archive": js1bm.file_record(archive_path),
        "candidate_runtime": compiled["runtime_tree"],
        "runtime_manifest": _runtime_manifest(compiled, runtime_payload),
        "inputs": {
            name: js1bm.payload_record(payload) for name, payload in payloads.items()
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "local_advisory_provenance": {
            "net_seg_flip_gain": advisory["seg"]["net_flip_gain"],
            "delta_archive_bytes": advisory["rate"]["delta_bytes"],
            "delta_dpose": advisory["pose"]["delta_dpose"],
            "projected_delta_s": advisory["projected_delta_s"],
            "verdict_scope": "local advisory admission only; fresh T4 worker is the verdict",
        },
        "source_fire_order": js1bm.file_record(source_order_path),
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

    loaded_payloads, loaded_request = dispatcher.load_sealed_inputs(
        request_path, input_root, request_record["sha256"]
    )
    if loaded_request != request or set(loaded_payloads) != set(payloads):
        raise MC36DualAxisSealError("dispatcher returned a different sealed object")

    dispatch_output = STORE / "dispatch" / RUN_ID
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
        str(dispatch_output.resolve()),
        "--detach",
        "--provider-detach-ack",
    ]
    order = {
        "schema": "ddm_mc36_dispatcher_conformant_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(dispatch_output.resolve()),
        "fire_trigger": (
            "MAIN confirms no active full-n600 Modal scorer lane, claims "
            f"{LANE_ID}, and every sealed request/input SHA verifies"
        ),
        "fresh_run_id": RUN_ID,
        "request": request_record,
        "fire_inputs": str(input_root.resolve()),
        "exact_command_argv": command,
        "estimated_cost_usd": 0.16,
        "remote_scope": (
            "one candidate; unchanged RE1T/JS1B worker retains the decoded raw, "
            "n600 T4 Seg field, official Pose first-six vectors for GT/base/candidate, "
            "inputs, outputs, and deterministic repeat"
        ),
        "post_harvest_rule": (
            "adjudicate the fresh matched-instrument Seg/Pose/rate components on the "
            "exact 186269-byte archive; this seal is preparation, never a score claim"
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
