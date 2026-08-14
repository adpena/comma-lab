#!/usr/bin/env python3
"""Seal and dispatch the RE1 Round-1 candidate-only T4 sign measurement.

Preparation is entirely local and scorer-free.  It rehashes the predecessor
blocker, verifies the exact candidate archive/runtime and the runtime's own
archive pin, persists every upload payload, and emits one immutable request.
MAIN may later dispatch that exact request.  The remote worker only decodes and
measures the retained SegNet field; mixed-axis adjudication happens here after
harvest and never becomes a score claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import modal

try:
    from experiments import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch
except (ImportError, ModuleNotFoundError):
    import ddm_js1b_modal_cuda_argmax_field_materializer as js1b_dispatch  # type: ignore[no-redef]

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
REMOTE_WORKER: Final = REMOTE_REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
REMOTE_JS1B_WORKER: Final = (
    REMOTE_REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
)
APP_NAME: Final = "comma-ddm-re1-round1-t4-sign-gate"
LANE_LABEL: Final = "ddm_re1_round1_t4_gate"
AXIS: Final = "[contest-CUDA T4 frozen-SegNet argmax field, n600, batch=16] COMPONENT-ONLY"
VOLUME_NAME: Final = js1b_dispatch.VOLUME_NAME
VOLUME_ROOT: Final = js1b_dispatch.VOLUME_ROOT
COMMIT_PERIOD_SECONDS: Final = 20.0
CONTEST_LIMIT_SECONDS: Final = 1_800.0

RUN_ID: Final = "ddm_re1_round1_t4_gate_20260813"
LANE_ID: Final = "ddm_re1_round1_t4_gate"
INSTANCE_JOB_ID: Final = f"modal:{RUN_ID}"
CLAIM_AGENT: Final = "main:ddm_re1t"

CONSUMER_STORE: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/"
    "ddm_re1_20260813/full_n600_exact/round_01_singleton_best"
)
CANDIDATE_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/"
    "ddm_re1_20260813/retained/candidates/round_01_singleton_best/primary/adapted_runtime"
)
CANDIDATE_ARCHIVE: Final = CANDIDATE_RUNTIME / "archive.zip"
BLOCKER_RECEIPT: Final = CONSUMER_STORE / "RE1X_FULL_N600_BLOCKER.json"
SEALED_REQUEST: Final = CONSUMER_STORE / "RE1T_T4_REQUEST.json"
SEALED_FIRE_ORDER: Final = CONSUMER_STORE / "RE1T_T4_FIRE_ORDER.json"
FIRE_INPUT_DIR: Final = CONSUMER_STORE / "re1t_t4_fire_inputs"
DEFAULT_DISPATCH_OUTPUT: Final = CONSUMER_STORE / "re1t_t4_dispatch"

CANDIDATE_SHA256: Final = "7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa"
CANDIDATE_BYTES: Final = 186_252
CANDIDATE_RUNTIME_TREE_SHA256: Final = (
    "63b93187e83cb310d68031a2b08b65b1a5e2103e830cede4941a7d3df604dc75"
)
CANDIDATE_RUNTIME_FILE_COUNT: Final = 25
BLOCKER_SHA256: Final = "197cfd2883e2c23c9f3e39cbe4fe1ce1b24953ebc81467de851a8281e4273a76"
BLOCKER_BYTES: Final = 11_846
BLOCKER_DISTINCT_RECORDS: Final = 28

BASE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_BYTES: Final = 186_252
BASE_FLIPS: Final = 34_970
DENOMINATOR: Final = 600 * 384 * 512
RATE_PRICE: Final = 25.0 / 37_545_489.0
EXPECTED_REMOTE_PAYLOAD_BYTES: Final = 7_555_248_128
REMOTE_STORAGE_RESERVE_BYTES: Final = 4 * 1024**3
LOCAL_STORAGE_RESERVE_BYTES: Final = 64 * 1024**2
ESTIMATED_COST_USD: Final = 0.16
BUDGET_SPENT_TO_DATE_USD_APPROX: Final = 2.4

GT_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
}
BASE_FIELD_RECORD: Final = {
    "bytes": 117_964_928,
    "sha256": "7648ad42e9f21942f86e81b97cabf46b710af747bba0909f7837ef3891232727",
}


class RE1TDispatchError(RuntimeError):
    """A custody, sealing, resume, dispatch, or adjudication invariant failed."""


def tree_record(root: Path) -> dict[str, Any]:
    """Hash one runtime tree exactly as the predecessor RE1X evaluator did."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RE1TDispatchError(f"candidate runtime is missing: {resolved}")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name.startswith("._"):
            continue
        relative = path.relative_to(resolved).as_posix()
        record = js1b_dispatch.file_record(path)
        row = {
            "relative_path": relative,
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        rows.append(row)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\n")
    return {
        "root": str(resolved),
        "file_count": len(rows),
        "tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def verify_runtime_archive_pin(runtime_root: Path, archive_record: dict[str, Any]) -> dict[str, Any]:
    """Verify the public entry point pins and consumes this candidate identity."""
    inflate_path = runtime_root.resolve() / "inflate.py"
    if not inflate_path.is_file():
        raise RE1TDispatchError(f"candidate runtime lacks inflate.py: {inflate_path}")
    source = inflate_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(inflate_path))
    constants: dict[str, Any] = {}
    verify_function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    verify_calls = 0
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}:
                    constants[target.id] = ast.literal_eval(value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_verify_input":
            verify_function = node
    if constants != {
        "ARCHIVE_SHA256": str(archive_record["sha256"]),
        "ARCHIVE_BYTES": int(archive_record["bytes"]),
    }:
        raise RE1TDispatchError(
            "candidate runtime inflate pin differs from the candidate archive: "
            f"observed={constants} expected={archive_record}"
        )
    if verify_function is None:
        raise RE1TDispatchError("candidate runtime has no _verify_input pin guard")
    loaded_names = {
        node.id
        for node in ast.walk(verify_function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    if not {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}.issubset(loaded_names):
        raise RE1TDispatchError("candidate runtime declares but does not consume both archive pins")
    compared_names = {
        node.id
        for compare in ast.walk(verify_function)
        if isinstance(compare, ast.Compare)
        for node in ast.walk(compare)
        if isinstance(node, ast.Name)
    }
    if not {"ARCHIVE_SHA256", "ARCHIVE_BYTES"}.issubset(compared_names):
        raise RE1TDispatchError("candidate runtime does not compare both archive pins")
    if sum(isinstance(node, ast.Raise) for node in ast.walk(verify_function)) < 2:
        raise RE1TDispatchError("candidate runtime pin guard does not fail closed on both mismatches")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            verify_calls += int(node.func.id == "_verify_input")
    if verify_calls < 1:
        raise RE1TDispatchError("candidate runtime never invokes _verify_input")
    return {
        "schema": "ddm_re1t_runtime_archive_pin.v1",
        "inflate_source": js1b_dispatch.file_record(inflate_path),
        "archive_sha256": constants["ARCHIVE_SHA256"],
        "archive_bytes": constants["ARCHIVE_BYTES"],
        "verify_input_calls": verify_calls,
        "verification_method": (
            "AST constants plus live _verify_input comparisons, fail-closed raises, and callsite"
        ),
        "passed": True,
    }


def iter_file_records(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested path/bytes/SHA records from a machine receipt."""
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from iter_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_file_records(child)


def verify_blocker_receipt(path: Path) -> dict[str, Any]:
    """Rehash the predecessor blocker's complete distinct record set."""
    js1b_dispatch.require_exact(
        path,
        size=BLOCKER_BYTES,
        digest=BLOCKER_SHA256,
        label="RE1X full-n600 blocker receipt",
    )
    blocker = json.loads(path.read_text(encoding="utf-8"))
    if blocker.get("verdict") != "BLOCKED_HASH_PINNED_PUBLIC_FRONT_DOOR_REQUIRES_CUDA":
        raise RE1TDispatchError("RE1X receipt is not the expected CUDA-lock blocker")
    archive = blocker.get("candidate_archive", {})
    runtime = blocker.get("candidate_runtime", {})
    if archive.get("sha256") != CANDIDATE_SHA256 or archive.get("bytes") != CANDIDATE_BYTES:
        raise RE1TDispatchError("RE1X blocker candidate archive identity differs")
    if runtime.get("tree_sha256") != CANDIDATE_RUNTIME_TREE_SHA256:
        raise RE1TDispatchError("RE1X blocker candidate runtime identity differs")
    distinct: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in iter_file_records(blocker):
        key = (str(record["path"]), int(record["bytes"]), str(record["sha256"]))
        distinct.setdefault(key, record)
    if len(distinct) != BLOCKER_DISTINCT_RECORDS:
        raise RE1TDispatchError(
            f"RE1X blocker record census differs: {len(distinct)} != {BLOCKER_DISTINCT_RECORDS}"
        )
    for record in distinct.values():
        js1b_dispatch.require_exact(
            Path(str(record["path"])),
            size=int(record["bytes"]),
            digest=str(record["sha256"]),
            label="RE1X blocker nested record",
        )
    return {
        "schema": "ddm_re1t_blocker_rehash.v1",
        "receipt": js1b_dispatch.file_record(path),
        "distinct_records_rehashed": len(distinct),
        "expected_cuda_lock_error": blocker["failure"]["runtime_error"],
        "passed": True,
    }


def k_arithmetic() -> dict[str, Any]:
    projected = 466.0 + 39.405 + 300.0
    return {
        "schema": "ddm_re1t_t4_k_arithmetic.v1",
        "k_archives": 1,
        "measured_prior_seconds_per_decode": 466.0,
        "scorer_passes": 1,
        "measured_prior_seconds_per_full_scorer_pass": 39.405,
        "reserve_seconds": 300.0,
        "projected_seconds_with_reserve": projected,
        "contest_limit_seconds": CONTEST_LIMIT_SECONDS,
        "headroom_seconds": CONTEST_LIMIT_SECONDS - projected,
        "fits_30_minutes": projected <= CONTEST_LIMIT_SECONDS,
        "epistemic_status": "DERIVED_FROM_PROVEN_SA1_T4_COMPONENT_TIMES",
    }


def prepare_request(
    *,
    candidate_archive: Path = CANDIDATE_ARCHIVE,
    candidate_runtime: Path = CANDIDATE_RUNTIME,
    blocker_receipt: Path = BLOCKER_RECEIPT,
    run_id: str = RUN_ID,
    resume_from: str = RUN_ID,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Build the exact upload payload set and immutable dispatch request."""
    if run_id != RUN_ID and not re.fullmatch(re.escape(RUN_ID) + r"r[0-9]{1,2}", run_id):
        # Versioned suffix (e.g. "...r2") is the RESEAL path: a later commit changed a
        # pinned worker source, the remote volume retains the ORIGINAL seal's inputs
        # under the canonical run id (byte-identity resume guard, run_gate:747-749),
        # and the payload law forbids mutating that namespace. A reseal therefore gets
        # its own run namespace; the old one stays retained for forensics.
        raise RE1TDispatchError(f"fresh RE1T run id must be {RUN_ID!r} or {RUN_ID!r}+'rN'")
    if resume_from != run_id:
        raise RE1TDispatchError("--resume-from is mandatory and must equal --run-id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise RE1TDispatchError("--run-id must be one safe path component")

    archive_path = candidate_archive.resolve()
    runtime_root = candidate_runtime.resolve()
    js1b_dispatch.require_exact(
        archive_path,
        size=CANDIDATE_BYTES,
        digest=CANDIDATE_SHA256,
        label="RE1 Round-1 candidate archive",
    )
    archive_record = js1b_dispatch.file_record(archive_path)
    runtime_archive = runtime_root / "archive.zip"
    js1b_dispatch.require_exact(
        runtime_archive,
        size=CANDIDATE_BYTES,
        digest=CANDIDATE_SHA256,
        label="RE1 Round-1 runtime archive",
    )
    runtime_tree = tree_record(runtime_root)
    if runtime_tree["tree_sha256"] != CANDIDATE_RUNTIME_TREE_SHA256:
        raise RE1TDispatchError("RE1 Round-1 runtime-tree SHA-256 differs")
    if runtime_tree["file_count"] != CANDIDATE_RUNTIME_FILE_COUNT:
        raise RE1TDispatchError("RE1 Round-1 runtime file count differs")
    runtime_pin = verify_runtime_archive_pin(runtime_root, archive_record)
    blocker_rehash = verify_blocker_receipt(blocker_receipt.resolve())
    runtime_bundle, runtime_manifest = js1b_dispatch.build_runtime_bundle(
        runtime_root,
        label="re1_round1_candidate",
    )
    payloads = {
        "candidate_archive.zip": archive_path.read_bytes(),
        "candidate_runtime.zip": runtime_bundle,
        "RE1X_FULL_N600_BLOCKER.json": blocker_receipt.resolve().read_bytes(),
    }
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO)
    request = {
        "schema": "ddm_re1t_t4_modal_request.v1",
        "axis": AXIS,
        "run_id": run_id,
        "resume_from": resume_from,
        "lane_id": LANE_ID,
        "instance_job_id": INSTANCE_JOB_ID,
        "claim_agent": CLAIM_AGENT,
        "seed": 1234,
        "batch_size": 16,
        "candidate_archive": archive_record,
        "candidate_runtime": runtime_tree,
        "runtime_archive_pin": runtime_pin,
        "blocker_rehash": blocker_rehash,
        "inputs": {
            name: js1b_dispatch.payload_record(payload) for name, payload in payloads.items()
        },
        "runtime_manifest": runtime_manifest,
        "retained_t4_controls": {
            "volume_run_path": str(VOLUME_ROOT / "ddm_js1b_20260813b"),
            "gt_field": GT_FIELD_RECORD,
            "cp135_base_field": BASE_FIELD_RECORD,
            "base_flips": BASE_FLIPS,
            "base_archive_bytes": BASE_BYTES,
            "base_archive_sha256": BASE_SHA256,
        },
        "local_pose_delta": 0.0,
        "pose_unmeasured": True,
        "pose_gate_note": (
            "A Seg sign-gate admission is provisional only; pz4r-law retained PoseNet-vector "
            "measurement is required before any composition or complete-S claim. If the candidate "
            "Seg field is identical to CP135, the instance is dead and no pose job fires."
        ),
        "k_arithmetic": k_arithmetic(),
        "storage_preflight_contract": {
            "expected_total_retained_payload_bytes": EXPECTED_REMOTE_PAYLOAD_BYTES,
            "reserve_bytes": REMOTE_STORAGE_RESERVE_BYTES,
            "cleanup_policy": "block and retain; never delete generated payloads",
            "volatile_measurement_is_not_checkpointed": True,
        },
        "source_git_head": git_head,
        "source_git_dirty": bool(git_status),
        "source_git_status_sha256": js1b_dispatch.sha256_bytes(git_status),
        "dispatcher_source_sha256": js1b_dispatch.sha256_file(Path(__file__)),
        "worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
        "retention_volume": VOLUME_NAME,
        "retention_volume_run_path": str(VOLUME_ROOT / run_id),
        "resume_required": True,
        "per_stage_checkpoints": True,
        "remote_scope": "exact CUDA-locked public decode plus frozen SegNet forward and field reduction",
        "local_scope": "mixed-axis sign-gate adjudication after harvest",
        "score_claim": False,
        "promotion_eligible": False,
    }
    return payloads, request


def local_storage_preflight(root: Path, payloads: dict[str, bytes]) -> dict[str, Any]:
    """Fail closed before retaining the local fire payload set."""
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    payload_bytes = sum(len(payload) for payload in payloads.values())
    required = payload_bytes + LOCAL_STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_re1t_local_storage_preflight.v1",
        "tier": str(root.resolve()),
        "free_bytes": usage.free,
        "fire_input_payload_bytes": payload_bytes,
        "reserve_bytes": LOCAL_STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; sealed fire inputs are durable and retained",
    }
    js1b_dispatch.atomic_json(root / "RE1T_LOCAL_STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise RE1TDispatchError("local fire-input storage preflight failed")
    return result


def fire_command(*, request_sha256: str, dispatch_output: Path) -> list[str]:
    """Return MAIN's exact detached provider command."""
    return [
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/ddm_re1t_modal_t4_sign_gate.py::main",
        "--sealed-request",
        str(SEALED_REQUEST),
        "--fire-input-dir",
        str(FIRE_INPUT_DIR),
        "--expected-request-sha256",
        request_sha256,
        "--output-dir",
        str(dispatch_output.resolve()),
        "--detach",
        "--provider-detach-ack",
    ]


def build_fire_order(request: dict[str, Any], request_record: dict[str, Any]) -> dict[str, Any]:
    """Build the sealed handoff; this function never dispatches."""
    command = fire_command(
        request_sha256=str(request_record["sha256"]),
        dispatch_output=DEFAULT_DISPATCH_OUTPUT,
    )
    run_id = str(request["run_id"])
    return {
        "schema": "ddm_re1t_t4_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN sole scorer-lane router",
        "consumer_store": str(CONSUMER_STORE),
        "fire_trigger": (
            "MAIN confirms no competing full-n600 scorer or Modal single-flight, verifies the "
            "sealed request plus candidate archive/runtime SHAs, and executes the exact command."
        ),
        "fresh_run_id": run_id,
        "lane_id": request["lane_id"],
        "instance_job_id": request["instance_job_id"],
        "request": request_record,
        "fire_inputs": str(FIRE_INPUT_DIR),
        "pre_fire_verify": {
            "candidate_archive_sha256": CANDIDATE_SHA256,
            "candidate_archive_bytes": CANDIDATE_BYTES,
            "candidate_runtime_tree_sha256": CANDIDATE_RUNTIME_TREE_SHA256,
            "runtime_archive_pin": request["runtime_archive_pin"],
            "blocker_distinct_records_rehashed": request["blocker_rehash"][
                "distinct_records_rehashed"
            ],
        },
        "exact_command_argv": command,
        "exact_command": shlex.join(command),
        "estimated_cost_usd": ESTIMATED_COST_USD,
        "budget_ledger": "#381",
        "budget_spent_to_date_usd_approx": BUDGET_SPENT_TO_DATE_USD_APPROX,
        "budget_spend_status": "RECALLED_FROM_CHARTER_AT_SEAL; approximate, not re-priced",
        "remote_scope": request["remote_scope"],
        "local_adjudication_after_harvest": True,
        "harvest": {
            "remote_volume": VOLUME_NAME,
            "remote_run_path": str(VOLUME_ROOT / run_id),
            "local_dispatch_output": str(DEFAULT_DISPATCH_OUTPUT),
            "recover_command": (
                ".venv/bin/python experiments/ddm_re1t_modal_t4_sign_gate.py recover "
                f"--output-dir {shlex.quote(str(DEFAULT_DISPATCH_OUTPUT))}"
            ),
            "retained_worker_result": str(VOLUME_ROOT / run_id / "FINAL_RESULT.json"),
            "retained_candidate_raw": str(VOLUME_ROOT / run_id / "retained/raw/candidate/0.raw"),
            "retained_candidate_field": str(
                VOLUME_ROOT / run_id / "retained/fields/candidate_argmax_n600.npy"
            ),
            "local_adjudication": str(DEFAULT_DISPATCH_OUTPUT / "LOCAL_ADJUDICATION.json"),
        },
        "post_harvest_rule": (
            "Identical candidate/base Seg field => DEAD at INSTANCE scope and no pose job. "
            "Any provisional Seg admission remains score_claim:false and requires a retained "
            "PoseNet-vector measurement before composition."
        ),
        "score_claim": False,
        "promotion_eligible": False,
    }


def persist_seal(
    root: Path,
    payloads: dict[str, bytes],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Persist all upload bytes, the request, preflight, and exact fire order."""
    if root.resolve() != CONSUMER_STORE.resolve():
        raise RE1TDispatchError(f"seal root must be the named consumer store: {CONSUMER_STORE}")
    preflight = local_storage_preflight(root, payloads)
    for name, payload in payloads.items():
        path = FIRE_INPUT_DIR / name
        if path.is_file():
            if js1b_dispatch.sha256_file(path) != js1b_dispatch.sha256_bytes(payload):
                raise RE1TDispatchError(f"retained fire input differs: {path}")
        else:
            js1b_dispatch.atomic_bytes(path, payload)
    js1b_dispatch.atomic_json(SEALED_REQUEST, request)
    request_record = js1b_dispatch.file_record(SEALED_REQUEST)
    order = build_fire_order(request, request_record)
    js1b_dispatch.atomic_json(SEALED_FIRE_ORDER, order)
    return {
        "schema": "ddm_re1t_t4_prepare_result.v1",
        "status": "READY_TO_FIRE_BY_MAIN",
        "storage_preflight": preflight,
        "request": request_record,
        "fire_order": js1b_dispatch.file_record(SEALED_FIRE_ORDER),
        "score_claim": False,
        "modal_fired": False,
    }


def load_sealed_inputs(
    *,
    sealed_request: Path,
    fire_input_dir: Path,
    expected_request_sha256: str,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Load only the exact prepared request and payload set."""
    request_path = sealed_request.resolve()
    if js1b_dispatch.sha256_file(request_path) != expected_request_sha256:
        raise RE1TDispatchError("sealed request SHA-256 differs from the fire order")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    run_id = str(request.get("run_id", ""))
    if run_id != RUN_ID and not re.fullmatch(re.escape(RUN_ID) + r"r[0-9]{1,2}", run_id):
        raise RE1TDispatchError("sealed request does not use the fresh RE1T run id")
    if request.get("resume_from") != run_id:
        raise RE1TDispatchError("sealed request resume_from must equal its run_id")
    if request.get("lane_id") != LANE_ID or request.get("instance_job_id") != INSTANCE_JOB_ID:
        raise RE1TDispatchError("sealed request lane identity differs")
    if request.get("score_claim") is not False or request.get("promotion_eligible") is not False:
        raise RE1TDispatchError("sealed request crossed its component-only authority boundary")
    if request.get("local_pose_delta") != 0.0 or request.get("pose_unmeasured") is not True:
        raise RE1TDispatchError("sealed request lost the explicit Pose-unknown placeholder law")
    if request.get("candidate_archive") != {
        "path": str(CANDIDATE_ARCHIVE.resolve()),
        "bytes": CANDIDATE_BYTES,
        "sha256": CANDIDATE_SHA256,
    }:
        raise RE1TDispatchError("sealed request candidate archive differs")
    if (
        request["candidate_runtime"]["tree_sha256"] != CANDIDATE_RUNTIME_TREE_SHA256
        or request["candidate_runtime"]["file_count"] != CANDIDATE_RUNTIME_FILE_COUNT
    ):
        raise RE1TDispatchError("sealed request candidate runtime differs")
    if request.get("runtime_archive_pin", {}).get("passed") is not True:
        raise RE1TDispatchError("sealed request lacks the candidate runtime pin proof")
    current_source_shas = {
        "dispatcher_source_sha256": js1b_dispatch.sha256_file(Path(__file__)),
        "worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_re1t_t4_sign_gate_worker.py"
        ),
        "js1b_worker_source_sha256": js1b_dispatch.sha256_file(
            REPO / "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py"
        ),
    }
    for key, observed in current_source_shas.items():
        if request.get(key) != observed:
            raise RE1TDispatchError(f"sealed request source drift: {key}")
    expected_inputs = {
        "candidate_archive.zip",
        "candidate_runtime.zip",
        "RE1X_FULL_N600_BLOCKER.json",
    }
    if set(request.get("inputs", {})) != expected_inputs:
        raise RE1TDispatchError("sealed request input census differs")
    if request["inputs"]["candidate_archive.zip"] != {
        "bytes": CANDIDATE_BYTES,
        "sha256": CANDIDATE_SHA256,
    }:
        raise RE1TDispatchError("sealed upload archive differs from candidate identity")
    payloads: dict[str, bytes] = {}
    for name, record in request["inputs"].items():
        if Path(name).name != name:
            raise RE1TDispatchError(f"unsafe sealed input name: {name!r}")
        path = fire_input_dir.resolve() / name
        js1b_dispatch.require_exact(
            path,
            size=int(record["bytes"]),
            digest=str(record["sha256"]),
            label="sealed RE1T fire input",
        )
        payloads[name] = path.read_bytes()
    return payloads, request


def adjudicate_measurement(measurement: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Apply SA1 mixed-axis arithmetic locally, with the Pose-unknown boundary explicit."""
    if measurement.get("execution_status") != "MEASUREMENT_COMPLETE":
        raise RE1TDispatchError("remote RE1T result is not a complete measurement")
    if (
        measurement.get("score_claim") is not False
        or measurement.get("promotion_eligible") is not False
        or measurement.get("pointer_moved") is not False
        or measurement.get("remote_adjudication_performed") is not False
    ):
        raise RE1TDispatchError("remote RE1T measurement crossed its component-only authority")
    if measurement.get("candidate_archive") != request["inputs"]["candidate_archive.zip"]:
        raise RE1TDispatchError("remote candidate archive record differs from the sealed request")
    if measurement.get("axis") != AXIS:
        raise RE1TDispatchError("remote measurement axis differs from the sealed T4 component axis")
    retained_fields = measurement.get("retained_prior_fields", {})
    for name, expected in {"gt": GT_FIELD_RECORD, "cp135_base": BASE_FIELD_RECORD}.items():
        observed = retained_fields.get(name, {})
        if {key: observed.get(key) for key in ("bytes", "sha256")} != expected:
            raise RE1TDispatchError(f"remote retained-field identity differs: {name}")
    metrics = measurement["field_measurement"]
    candidate_flips = int(metrics["candidate_flips_vs_gt"])
    if int(metrics.get("denominator_pixels", -1)) != DENOMINATOR:
        raise RE1TDispatchError("remote full-population denominator differs")
    if not 0 <= candidate_flips <= DENOMINATOR:
        raise RE1TDispatchError("remote candidate flip count is outside the population")
    if int(metrics["base_flips_vs_gt"]) != BASE_FLIPS:
        raise RE1TDispatchError("remote retained base control no longer has 34,970 flips")
    if int(metrics.get("candidate_minus_base_flips", DENOMINATOR + 1)) != (
        candidate_flips - BASE_FLIPS
    ):
        raise RE1TDispatchError("remote candidate/base flip delta is inconsistent")
    changed_pixels = int(metrics.get("candidate_changed_pixels_vs_cp135", -1))
    if not 0 <= changed_pixels <= DENOMINATOR:
        raise RE1TDispatchError("remote candidate/base change count is outside the population")
    candidate_identical = bool(metrics["candidate_field_identical_to_cp135"])
    if candidate_identical != (changed_pixels == 0):
        raise RE1TDispatchError("remote candidate/base identity flag is inconsistent")
    if metrics.get("adjudicated_remotely") is not False:
        raise RE1TDispatchError("remote result crossed the local-adjudication boundary")
    if candidate_identical and candidate_flips != BASE_FLIPS:
        raise RE1TDispatchError("identical candidate/base fields report different GT flips")

    delta_flips = candidate_flips - BASE_FLIPS
    delta_bytes = int(request["candidate_archive"]["bytes"]) - BASE_BYTES
    local_pose_delta = float(request["local_pose_delta"])
    pose_unmeasured = bool(request["pose_unmeasured"])
    if local_pose_delta != 0.0 or not pose_unmeasured:
        raise RE1TDispatchError("RE1T must preserve the explicit zero-placeholder Pose-unknown law")
    seg_delta_s = 100.0 * delta_flips / DENOMINATOR
    pose_delta_s_placeholder = 0.0
    rate_delta_s = RATE_PRICE * delta_bytes
    mixed_delta_s = seg_delta_s + pose_delta_s_placeholder + rate_delta_s
    per_flip_s = 100.0 / DENOMINATOR
    minimum_reduction = max(1, math.floor(rate_delta_s / per_flip_s) + 1)
    ceiling = BASE_FLIPS - minimum_reduction
    provisional_admission = candidate_flips <= ceiling and mixed_delta_s < 0.0

    if candidate_identical:
        verdict = "DEAD_INSTANCE_RECEIVER_NULL_IDENTICAL_TO_CP135"
        disposition = "FOLDED"
        pose_follow_up_required = False
    elif provisional_admission:
        verdict = "PROVISIONALLY_ADMITTED_SEG_SIGN_GATE_POSE_MEASUREMENT_REQUIRED"
        disposition = "QUEUED-WITH-A-FIRE-ORDER"
        pose_follow_up_required = True
    else:
        verdict = "DEAD_INSTANCE_NO_T4_SEG_SIGN_GAIN"
        disposition = "FOLDED"
        pose_follow_up_required = False
    return {
        "schema": "ddm_re1t_local_mixed_axis_adjudication.v1",
        "status": verdict,
        "verdict": verdict,
        "verdict_scope": "INSTANCE: RE1 Round-1 archive 7be3eb94 through runtime 63b93187",
        "disposition": disposition,
        "axis": AXIS,
        "base_flips": BASE_FLIPS,
        "candidate_flips": candidate_flips,
        "delta_flips": delta_flips,
        "candidate_field_identical_to_cp135": candidate_identical,
        "base_archive_bytes": BASE_BYTES,
        "candidate_archive_bytes": int(request["candidate_archive"]["bytes"]),
        "delta_bytes": delta_bytes,
        "local_pose_delta_placeholder": local_pose_delta,
        "pose_unmeasured": pose_unmeasured,
        "seg_delta_s_exact_t4_field": seg_delta_s,
        "pose_delta_s_placeholder_not_measurement": pose_delta_s_placeholder,
        "rate_delta_s_exact_archive": rate_delta_s,
        "mixed_axis_delta_s_gate_only": mixed_delta_s,
        "minimum_required_flip_reduction_without_pose_measurement": minimum_reduction,
        "maximum_admissible_candidate_flips_without_pose_measurement": ceiling,
        "provisional_seg_sign_admission": provisional_admission,
        "pose_follow_up_required_before_composition": pose_follow_up_required,
        "pose_job_may_fire": pose_follow_up_required,
        "gate_note": request["pose_gate_note"],
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


app = modal.App(APP_NAME, include_source=False)
retained_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
gate_image = (
    eval_image.add_local_file(
        "experiments/ddm_re1t_t4_sign_gate_worker.py",
        remote_path=str(REMOTE_WORKER),
        copy=False,
    )
    .add_local_file(
        "experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py",
        remote_path=str(REMOTE_JS1B_WORKER),
        copy=False,
    )
    .add_local_python_source(
        "ddm_re1t_modal_t4_sign_gate",
        "ddm_js1b_modal_cuda_argmax_field_materializer",
    )
)


@app.function(
    image=gate_image,
    gpu="T4",
    timeout=int(CONTEST_LIMIT_SECONDS),
    memory=16_384,
    volumes={str(VOLUME_ROOT): retained_volume},
)
def run_gate(payloads: dict[str, bytes], request: dict[str, Any]) -> dict[str, Any]:
    """Run only the CUDA-locked decode and SegNet measurement remotely."""
    run_id = str(request["run_id"])
    run_root = VOLUME_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    remote_inputs = {**payloads, "REQUEST.json": js1b_dispatch.canonical_json_bytes(request)}
    for name, payload in remote_inputs.items():
        path = run_root / "inputs" / name
        if path.is_file():
            if js1b_dispatch.sha256_file(path) != js1b_dispatch.sha256_bytes(payload):
                raise RE1TDispatchError(f"resume input differs: {path}")
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
            "schema": "ddm_re1t_t4_modal_return.v1",
            "measurement_complete": False,
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
        "schema": "ddm_re1t_t4_modal_return.v1",
        "measurement_complete": True,
        "returncode": 0,
        "axis": AXIS,
        "run_id": run_id,
        "volume_name": VOLUME_NAME,
        "volume_path": str(run_root),
        "final_result_sha256": js1b_dispatch.sha256_bytes(final_bytes),
        "artifacts": {"RE1T_T4_REMOTE_RESULT.json": final_bytes},
        "score_claim": False,
        "promotion_eligible": False,
    }


def recover(output_dir: Path, *, timeout_seconds: float = 0.0) -> int:
    """Harvest the retained measurement, then adjudicate its scalars locally."""
    output = output_dir.resolve()
    spawn = json.loads((output / "modal_auth_eval_spawn.json").read_text(encoding="utf-8"))
    call_id = str(spawn["call_id"])
    try:
        result = modal.functions.FunctionCall.from_id(call_id).get(timeout=timeout_seconds)
    except TimeoutError:
        print(json.dumps({"status": "pending", "call_id": call_id}, sort_keys=True))
        return 4
    if not isinstance(result, dict):
        raise RE1TDispatchError(f"remote return is not a dict: {type(result).__name__}")
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        if Path(name).name != name or not isinstance(payload, bytes):
            raise RE1TDispatchError(f"unsafe returned artifact: {name!r}")
        js1b_dispatch.atomic_bytes(output / name, payload)
    js1b_dispatch.atomic_json(output / "modal_re1t_t4_result.json", result)
    complete = bool(result.get("measurement_complete"))
    adjudication: dict[str, Any] | None = None
    if complete:
        request = json.loads((output / "REQUEST.json").read_text(encoding="utf-8"))
        measurement = json.loads(
            (output / "RE1T_T4_REMOTE_RESULT.json").read_text(encoding="utf-8")
        )
        adjudication = adjudicate_measurement(measurement, request)
        js1b_dispatch.atomic_json(output / "LOCAL_ADJUDICATION.json", adjudication)
    update_call_id_outcome(
        call_id=call_id,
        status="harvested" if complete else "failed",
        rc=int(result.get("returncode", 1)),
        score_axis="contest_cuda_re1_round1_argmax_field_component",
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
        status="completed_re1t_t4_measurement_recovered" if complete else "failed_re1t_t4_measurement_recovered",
        notes=f"RE1T T4 measurement recovered; call_id={call_id}; output={output}",
    )
    print(json.dumps({"remote": result, "local_adjudication": adjudication}, indent=2, sort_keys=True))
    return 0 if complete else 1


@app.local_entrypoint()
def main(
    sealed_request: str,
    fire_input_dir: str,
    expected_request_sha256: str,
    output_dir: str = str(DEFAULT_DISPATCH_OUTPUT),
    detach: bool = False,
    provider_detach_ack: bool = False,
) -> None:
    """MAIN-only entry point for the exact pre-sealed request."""
    if detach and not provider_detach_ack:
        raise SystemExit("FATAL: --detach requires --provider-detach-ack")
    payloads, request = load_sealed_inputs(
        sealed_request=Path(sealed_request),
        fire_input_dir=Path(fire_input_dir),
        expected_request_sha256=expected_request_sha256,
    )
    if not request["k_arithmetic"]["fits_30_minutes"]:
        raise SystemExit("FATAL: candidate-only projection exceeds 30 minutes")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    js1b_dispatch.atomic_json(output / "REQUEST.json", request)
    for name, payload in payloads.items():
        js1b_dispatch.atomic_bytes(output / "fire_inputs" / name, payload)
    spec = ClaimSpec(
        lane_id=str(request["lane_id"]),
        instance_job_id=str(request["instance_job_id"]),
        agent=str(request["claim_agent"]),
        notes=(
            f"RE1 Round-1 candidate-only T4 sign measurement; run={request['run_id']}; "
            f"volume={VOLUME_NAME}/{request['run_id']}"
        ),
    )
    claim_modal_auth_eval_dispatch(
        repo_root=REPO,
        spec=spec,
        status="active_re1t_t4_measurement_spawning",
    )
    assert_modal_single_flight(label=LANE_LABEL, lane_id=str(request["lane_id"]), repo_root=REPO)
    if detach:
        call = run_gate.spawn(payloads, request)
        call_id = function_call_id(call)
        register_dispatched_call_id_fail_closed(
            call_id=call_id,
            lane_id=str(request["lane_id"]),
            label=LANE_LABEL,
            platform="modal",
            gpu="T4",
            expected_axis="contest_cuda_re1_round1_argmax_field_component",
            recipe="experiments/ddm_re1t_modal_t4_sign_gate.py::main",
            max_seconds=int(CONTEST_LIMIT_SECONDS),
            agent=str(request["claim_agent"]),
            base_archive_sha256=BASE_SHA256,
            composed_archive_sha256=CANDIDATE_SHA256,
            archive_count=1,
            volume_name=VOLUME_NAME,
            volume_run_id=str(request["run_id"]),
        )
        write_spawn_metadata(
            out_dir=output,
            tool="experiments/ddm_re1t_modal_t4_sign_gate.py",
            app=APP_NAME,
            axis="contest_cuda_re1_round1_argmax_field_component",
            call_id=call_id,
            local_request=request,
            result_json_name="modal_re1t_t4_result.json",
            recover_tool="experiments/ddm_re1t_modal_t4_sign_gate.py recover",
            extra={
                "lane_id": request["lane_id"],
                "instance_job_id": request["instance_job_id"],
                "claim_agent": request["claim_agent"],
                "claim_platform": "modal",
                "volume_name": VOLUME_NAME,
                "volume_run_id": request["run_id"],
            },
        )
        claim_modal_auth_eval_dispatch(
            repo_root=REPO,
            spec=ClaimSpec(
                lane_id=str(request["lane_id"]),
                instance_job_id=str(request["instance_job_id"]),
                agent=str(request["claim_agent"]),
                force=True,
                notes=f"RE1T detached T4 measurement accepted; call_id={call_id}; output={output}",
            ),
            status="active_re1t_t4_measurement_spawned",
        )
        print(f"DISPATCHED call_id={call_id}")
        print(f"RECOVER .venv/bin/python {__file__} recover --output-dir {output}")
        return
    result = run_gate.remote(payloads, request)
    artifacts = result.pop("artifacts", {})
    for name, payload in artifacts.items():
        js1b_dispatch.atomic_bytes(output / name, payload)
    js1b_dispatch.atomic_json(output / "modal_re1t_t4_result.json", result)
    if result.get("measurement_complete"):
        measurement = json.loads(
            (output / "RE1T_T4_REMOTE_RESULT.json").read_text(encoding="utf-8")
        )
        js1b_dispatch.atomic_json(
            output / "LOCAL_ADJUDICATION.json",
            adjudicate_measurement(measurement, request),
        )
    terminal_modal_auth_eval_claim(
        repo_root=REPO,
        spec=ClaimSpec(
            lane_id=str(request["lane_id"]),
            instance_job_id=str(request["instance_job_id"]),
            agent=str(request["claim_agent"]),
            force=True,
        ),
        status=(
            "completed_re1t_t4_measurement_synchronous"
            if result.get("measurement_complete")
            else "failed_re1t_t4_measurement_synchronous"
        ),
        notes=f"RE1T synchronous T4 measurement returned; output={output}; resume={RUN_ID}",
    )
    if not result.get("measurement_complete"):
        raise SystemExit("FATAL: RE1T T4 measurement failed; resume the same run id")


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="scorer-free dry-run seal; never dispatches")
    prepare_parser.add_argument("--candidate-archive", type=Path, default=CANDIDATE_ARCHIVE)
    prepare_parser.add_argument("--candidate-runtime", type=Path, default=CANDIDATE_RUNTIME)
    prepare_parser.add_argument("--blocker-receipt", type=Path, default=BLOCKER_RECEIPT)
    prepare_parser.add_argument("--consumer-store", type=Path, default=CONSUMER_STORE)
    prepare_parser.add_argument("--run-id", default=RUN_ID)
    prepare_parser.add_argument("--resume-from", default=RUN_ID)
    recover_parser = sub.add_parser("recover", help="harvest remote measurement and adjudicate locally")
    recover_parser.add_argument("--output-dir", type=Path, required=True)
    recover_parser.add_argument("--timeout-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payloads, request = prepare_request(
            candidate_archive=args.candidate_archive,
            candidate_runtime=args.candidate_runtime,
            blocker_receipt=args.blocker_receipt,
            run_id=args.run_id,
            resume_from=args.resume_from,
        )
        result = persist_seal(args.consumer_store, payloads, request)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "recover":
        return recover(args.output_dir, timeout_seconds=args.timeout_seconds)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(_main_cli())
