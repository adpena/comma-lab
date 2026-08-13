#!/usr/bin/env python3
"""Retained, resumable full-n600 advisory evaluation for RE1 Round 1.

The candidate is decoded only through its shipped ``inflate.sh`` front door.
The exact decoded raw, SegNet inputs/logits/argmax, and two independent
PoseNet input/output/vector passes are retained on Vertigo.  CP135 and GT
fields come from the completed PZ4R matched-instrument run and are rehashed
and recomputed before use.  This is a macOS-CPU advisory gate, never a score
claim or pointer update.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

sys.dont_write_bytecode = True
_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

import numpy as np

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as retained

REPO: Final = Path(__file__).resolve().parents[1]
UPSTREAM: Final = REPO / "upstream"
RUN_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best"
)
CANDIDATE_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/"
    "probability_object_race/ddm_re1_20260813/retained/candidates/"
    "round_01_singleton_best/primary"
)
CANDIDATE_RUNTIME: Final = CANDIDATE_ROOT / "adapted_runtime"
CANDIDATE_ARCHIVE: Final = CANDIDATE_RUNTIME / "archive.zip"
CANDIDATE_ARCHIVE_BYTES: Final = 186_252
CANDIDATE_ARCHIVE_SHA256: Final = "7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa"
CANDIDATE_RUNTIME_TREE_SHA256: Final = "63b93187e83cb310d68031a2b08b65b1a5e2103e830cede4941a7d3df604dc75"
BASE_ARCHIVE_BYTES: Final = 186_252
BASE_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
PZ4R_RUN_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval"
)
PZ4R_RESULT: Final = PZ4R_RUN_ROOT / "PZ4R_FULL_N600_RESULT.json"
PZ4R_RESULT_BYTES: Final = 26_049
PZ4R_RESULT_SHA256: Final = "99f3361767145299221389843b8a435f8581442855e3e1da62a5b094544faa0c"
UNCOMPRESSED_BYTES: Final = 37_545_489
PAIR_COUNT: Final = 600
SEG_PIXELS_PER_PAIR: Final = 384 * 512
TOTAL_SEG_PIXELS: Final = PAIR_COUNT * SEG_PIXELS_PER_PAIR
GAP_TO_SUB015: Final = 0.01195513827824177
AXIS: Final = "[macOS-CPU advisory, frozen CPU-torch SegNet+PoseNet, n600] NON-PROMOTABLE"
RESUME_ID: Final = "ddm_re1x_round1_full_n600_eval_20260813"
RESERVE_BYTES: Final = 8 * 1024**3
EXPECTED_PAYLOAD_BYTES: Final = (
    retained.RAW_BYTES
    + retained.SEG_INPUT_BYTES
    + retained.LOGIT_BYTES
    + retained.FIELD_DATA_BYTES
    + 2 * (PAIR_COUNT * 12 * 384 * 512 * 4 + PAIR_COUNT * 12 * 4 + PAIR_COUNT * 6 * 4)
)


class RE1XEvalError(RuntimeError):
    """A custody, retention, resume, or metric invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_file(path: Path, *, size: int, digest: str) -> dict[str, Any]:
    if not path.is_file():
        raise RE1XEvalError(f"required input is missing: {path}")
    record = file_record(path)
    if record["bytes"] != size or record["sha256"] != digest:
        raise RE1XEvalError(f"required input custody differs: {record}")
    return record


def tree_record(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc" or path.name.startswith("._"):
            continue
        relative = path.relative_to(root).as_posix()
        record = file_record(path)
        rows.append({"relative_path": relative, **record})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\n")
    return {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def retained_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def atomic_json(path: Path, value: Any) -> None:
    retained.atomic_json(path, value)


def storage_preflight(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    already = retained_bytes(run_root)
    remaining = max(0, EXPECTED_PAYLOAD_BYTES - already)
    required = remaining + RESERVE_BYTES
    result = {
        "schema": "ddm_re1x_round1_full_n600_storage_preflight.v1",
        "tier": str(run_root.resolve()),
        "free_bytes": usage.free,
        "already_retained_bytes": already,
        "expected_total_retained_payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "remaining_payload_bytes": remaining,
        "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; retain every raw/scorer/vector payload; never delete",
    }
    atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise RE1XEvalError(f"storage preflight failed: free={usage.free}, required={required}")
    return result


def git_provenance() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v2"], cwd=REPO)
    return {
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "runner": file_record(Path(__file__)),
        "retained_worker": file_record(Path(retained.__file__)),
    }


def require_runtime_tree() -> dict[str, Any]:
    record = tree_record(CANDIDATE_RUNTIME)
    if record["tree_sha256"] != CANDIDATE_RUNTIME_TREE_SHA256 or record["file_count"] != 25:
        raise RE1XEvalError(f"candidate runtime tree differs: files={record['file_count']} sha={record['tree_sha256']}")
    return record


def load_matched_base_reference() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result_record = require_file(PZ4R_RESULT, size=PZ4R_RESULT_BYTES, digest=PZ4R_RESULT_SHA256)
    value = json.loads(PZ4R_RESULT.read_text())
    if value.get("execution_status") != "COMPLETE" or value.get("axis") != AXIS:
        raise RE1XEvalError("PZ4R matched-base result is not the sealed complete advisory row")
    if value["inputs"]["base_archive"]["sha256"] != BASE_ARCHIVE_SHA256:
        raise RE1XEvalError("PZ4R matched-base archive identity differs from CP135")
    if value["environment"] != {
        "platform": "macOS-26.4-arm64-arm-64bit",
        "python": "3.11.16 (main, Aug 12 2026, 23:03:19) [Clang 21.0.0 (clang-2100.1.1.101)]",
        "torch": "2.10.0",
        "numpy": "2.3.4",
        "cpu_threads": 4,
    }:
        raise RE1XEvalError("PZ4R matched-base instrument tuple differs")
    retained.require_record(Path(retained.__file__), value["inputs"]["git"]["retained_worker"])
    for record in value["inputs"]["upstream"].values():
        retained.require_record(Path(record["path"]), record)
    records = {
        "base_raw": value["inputs"]["base_raw"],
        "gt_argmax": value["retention"]["gt_argmax"],
        "base_argmax": value["retention"]["base_argmax"],
        "gt_pose_vectors": value["retention"]["gt_pose_vectors"],
        "base_pose_vectors": value["retention"]["base_pose_vectors"],
    }
    for record in records.values():
        retained.require_record(Path(record["path"]), record)
    return result_record, records, value["environment"]


def require_matched_environment(expected: dict[str, Any]) -> dict[str, Any]:
    import torch

    actual = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    if actual != expected:
        raise RE1XEvalError(f"current scorer environment differs from matched PZ4R instrument: {actual}")
    return actual


def field_metrics(gt_path: Path, value_path: Path) -> dict[str, Any]:
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    value = np.load(value_path, mmap_mode="r", allow_pickle=False)
    if gt.shape != (PAIR_COUNT, 384, 512) or value.shape != gt.shape:
        raise RE1XEvalError("SegNet argmax field geometry differs")
    flips = int(np.count_nonzero(gt != value))
    return {
        "flips": flips,
        "denominator_pixels": TOTAL_SEG_PIXELS,
        "d_seg": flips / TOTAL_SEG_PIXELS,
        "gt_field": file_record(gt_path),
        "value_field": file_record(value_path),
    }


def pose_metrics(gt_path: Path, value_path: Path) -> dict[str, Any]:
    gt = np.asarray(np.load(gt_path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    value = np.asarray(np.load(value_path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    if gt.shape != (PAIR_COUNT, 6) or value.shape != gt.shape:
        raise RE1XEvalError("PoseNet first-six vector geometry differs")
    error = value - gt
    return {
        "pairs": PAIR_COUNT,
        "components_per_pair": 6,
        "d_pose": float(np.mean(error * error, dtype=np.float32)),
        "gt_vectors": file_record(gt_path),
        "value_vectors": file_record(value_path),
    }


def iter_file_records(value: Any):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            yield value
        for child in value.values():
            yield from iter_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_file_records(child)


def score(d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    terms = {
        "seg": 100.0 * d_seg,
        "pose": math.sqrt(10.0 * d_pose),
        "rate": 25.0 * archive_bytes / UNCOMPRESSED_BYTES,
    }
    terms["total"] = sum(terms.values())
    return terms


def score_delta(base: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    result = {name: candidate[name] - base[name] for name in ("seg", "pose", "rate")}
    result["total"] = sum(result.values())
    result["gap_share"] = -result["total"] / GAP_TO_SUB015
    return result


def adjudicate(delta_total: float, base_d_pose: float, candidate_d_pose: float) -> dict[str, Any]:
    pose_held = candidate_d_pose <= base_d_pose
    earned = delta_total < 0.0 and pose_held
    return {
        "pose_held": pose_held,
        "t4_confirmation_earned": earned,
        "verdict": (
            "ADVISORY_COMPLETE_S_BETTER_POSE_HELD_T4_CONFIRMATION_EARNED"
            if earned
            else "DEAD_INSTANCE_NO_COMPLETE_S_GAIN_OR_POSE_NOT_HELD"
        ),
        "disposition": "QUEUED-WITH-A-FIRE-ORDER" if earned else "FOLDED",
    }


def write_fire_order(run_root: Path, final: dict[str, Any]) -> dict[str, Any] | None:
    if not final["adjudication"]["t4_confirmation_earned"]:
        return None
    result_record = file_record(run_root / "RE1X_FULL_N600_RESULT.json")
    order = {
        "schema": "ddm_re1x_round1_t4_fire_order.v1",
        "sealed": True,
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str(run_root.resolve()),
        "fire_trigger": (
            "MAIN verifies this result and its retained payload records, claims Modal #381 "
            "single-flight, and evaluates the exact archive/runtime pair on contest-CUDA T4 n600"
        ),
        "estimated_cost_usd": 0.16,
        "budget_ledger": "#381",
        "candidate_archive": final["inputs"]["candidate_archive"],
        "candidate_runtime_tree_sha256": CANDIDATE_RUNTIME_TREE_SHA256,
        "local_gate_result": result_record,
        "local_delta_s": final["score"]["delta_candidate_minus_base"]["total"],
        "score_claim": False,
        "promotion_eligible": False,
    }
    path = run_root / "T4_FIRE_ORDER.json"
    atomic_json(path, order)
    return {"path": str(path), **file_record(path)}


def record_public_front_door_blocker(run_root: Path, error: BaseException | str) -> dict[str, Any]:
    """Seal a receiver failure without misreporting a complete-S row."""
    log_path = run_root / "receivers/re1_round_01_candidate/inflate.log"
    if not log_path.is_file():
        raise RE1XEvalError("receiver failed without a retained public-front-door log")
    log_text = log_path.read_text(errors="replace")
    cuda_requirement = "F26 inflation requires a CUDA-capable GPU"
    if cuda_requirement not in log_text:
        raise RE1XEvalError("receiver failure is not the known hash-pinned CUDA boundary")
    extracted_payload = run_root / "work/re1_round_01_candidate/archive/p"
    if not extracted_payload.is_file():
        raise RE1XEvalError("receiver failure did not retain its extracted archive payload")
    inputs_path = run_root / "INPUTS.json"
    if not inputs_path.is_file():
        raise RE1XEvalError("receiver failure occurred without an immutable input receipt")
    blocker = {
        "schema": "ddm_re1x_round1_full_n600_blocker.v1",
        "execution_status": "BLOCKED_PRE_SCORE",
        "verdict": "BLOCKED_HASH_PINNED_PUBLIC_FRONT_DOOR_REQUIRES_CUDA",
        "verdict_scope": (
            "INSTANCE AND EXECUTION SURFACE: RE1 Round 1 archive 7be3eb94 through "
            "runtime tree 63b93187 on the charter-authorized local macOS CPU/Metal host"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "candidate_score": None,
        "delta_candidate_minus_base": None,
        "t4_confirmation_earned": False,
        "input_receipt": file_record(inputs_path),
        "candidate_archive": require_file(
            CANDIDATE_ARCHIVE,
            size=CANDIDATE_ARCHIVE_BYTES,
            digest=CANDIDATE_ARCHIVE_SHA256,
        ),
        "candidate_runtime": require_runtime_tree(),
        "failure": {
            "exception": str(error),
            "runtime_error": cuda_requirement,
            "public_front_door_attempted": True,
            "public_front_door_completed": False,
            "raw_materialized": False,
            "scorer_started": False,
            "log": file_record(log_path),
        },
        "retention": {
            "run_root": str(run_root.resolve()),
            "extracted_archive_payload": file_record(extracted_payload),
            "public_front_door_log": file_record(log_path),
            "cleanup_policy": "certify-or-block; attempted payload and failure evidence retained",
        },
        "adjudication": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN sole scorer-lane router",
            "consumer_store": str(run_root.resolve()),
            "fire_trigger": (
                "operator explicitly authorizes a CUDA public-front-door receiver execution "
                "or supplies the exact candidate raw plus receiver receipt under custody; "
                "then resume the matched local n600 Seg/Pose scorer"
            ),
            "forbidden_shortcuts": [
                "patching the hash-pinned runtime to CPU or MPS",
                "side-loading CP135 raw as though the candidate public front door produced it",
                "inferring PoseNet equality from entropy or token parse-back identity",
            ],
        },
    }
    path = run_root / "RE1X_FULL_N600_BLOCKER.json"
    atomic_json(path, blocker)
    retained.checkpoint_once(run_root / "checkpoints/stage_09_public_front_door_blocked.json", blocker)
    return blocker


def validate_resumed_inputs(inputs: dict[str, Any]) -> None:
    if inputs.get("resume_from") != RESUME_ID or inputs.get("axis") != AXIS:
        raise RE1XEvalError("resumed stage-0 identity differs")
    archive_record = require_file(
        CANDIDATE_ARCHIVE,
        size=CANDIDATE_ARCHIVE_BYTES,
        digest=CANDIDATE_ARCHIVE_SHA256,
    )
    runtime_record = require_runtime_tree()
    _, expected_records, expected_environment = load_matched_base_reference()
    if inputs.get("candidate_archive") != archive_record:
        raise RE1XEvalError("resumed candidate archive record differs")
    if inputs.get("candidate_runtime") != runtime_record:
        raise RE1XEvalError("resumed candidate runtime record differs")
    if inputs.get("matched_base_records") != expected_records:
        raise RE1XEvalError("resumed matched-base records differ")
    if inputs.get("matched_environment") != expected_environment:
        raise RE1XEvalError("resumed matched scorer environment differs")
    require_matched_environment(expected_environment)


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    if resume_from != RESUME_ID:
        raise RE1XEvalError(f"resume id must be {RESUME_ID!r}")
    run_root.mkdir(parents=True, exist_ok=True)
    final_path = run_root / "RE1X_FULL_N600_RESULT.json"
    final_checkpoint = run_root / "checkpoints/stage_40_final.json"
    if final_path.is_file() and final_checkpoint.is_file():
        final = json.loads(final_path.read_text())
        if final_checkpoint.read_bytes() != retained.canonical_json_bytes(final):
            raise RE1XEvalError("completed result and final checkpoint differ")
        for record in iter_file_records(final):
            retained.require_record(Path(record["path"]), record)
        write_fire_order(run_root, final)
        return final

    storage = storage_preflight(run_root)
    inputs_path = run_root / "INPUTS.json"
    inputs_checkpoint = run_root / "checkpoints/stage_00_inputs.json"
    if inputs_path.is_file() and inputs_checkpoint.is_file():
        inputs = json.loads(inputs_path.read_text())
        validate_resumed_inputs(inputs)
        checkpoint_inputs = {key: value for key, value in inputs.items() if key != "storage_preflight"}
        if inputs_checkpoint.read_bytes() != retained.canonical_json_bytes(checkpoint_inputs):
            raise RE1XEvalError("INPUTS.json and immutable stage-0 checkpoint differ")
        archive_record = inputs["candidate_archive"]
        base_records = inputs["matched_base_records"]
        environment = inputs["matched_environment"]
    else:
        archive_record = require_file(
            CANDIDATE_ARCHIVE,
            size=CANDIDATE_ARCHIVE_BYTES,
            digest=CANDIDATE_ARCHIVE_SHA256,
        )
        runtime_tree = require_runtime_tree()
        pz4r_result_record, base_records, environment = load_matched_base_reference()
        require_matched_environment(environment)
        inputs = {
            "schema": "ddm_re1x_round1_full_n600_inputs.v1",
            "resume_from": resume_from,
            "axis": AXIS,
            "candidate_archive": archive_record,
            "candidate_runtime": runtime_tree,
            "matched_base_result": pz4r_result_record,
            "matched_base_records": base_records,
            "matched_environment": environment,
            "upstream": {
                "evaluate_py": file_record(UPSTREAM / "evaluate.py"),
                "frame_utils_py": file_record(UPSTREAM / "frame_utils.py"),
                "modules_py": file_record(UPSTREAM / "modules.py"),
                "segnet_weights": file_record(UPSTREAM / "models/segnet.safetensors"),
                "posenet_weights": file_record(UPSTREAM / "models/posenet.safetensors"),
                "gt_video": file_record(UPSTREAM / "videos/0.mkv"),
            },
            "git": git_provenance(),
            "storage_preflight": storage,
            "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
            "chunking": {"batch_size": retained.BATCH_SIZE, "maximum_allowed": 120},
        }
        atomic_json(inputs_path, inputs)
        retained.checkpoint_once(
            inputs_checkpoint,
            {key: value for key, value in inputs.items() if key != "storage_preflight"},
        )

    retained.UPSTREAM = UPSTREAM
    receiver = retained.decode_exact_receiver(
        name="re1_round_01_candidate",
        archive_path=CANDIDATE_ARCHIVE,
        archive_record=archive_record,
        runtime_root=CANDIDATE_RUNTIME,
        run_root=run_root,
    )
    retained.checkpoint_once(
        run_root / "checkpoints/stage_10_candidate_decoded.json",
        {"stage": "candidate_decoded", "receiver": receiver, "complete": True},
    )

    import torch

    torch.set_num_threads(4)
    environment = require_matched_environment(environment)
    torch.manual_seed(retained.SEED)
    np.random.seed(retained.SEED)
    device = torch.device("cpu")

    segnet = retained.load_segnet(device)
    candidate_seg_receipt = retained.score_argmax_field(
        source="re1_round_01_candidate",
        raw_root=Path(receiver["raw"]["path"]).parent,
        raw_record=receiver["raw"],
        scorer=segnet,
        device=device,
        run_root=run_root,
    )
    retained.checkpoint_once(
        run_root / "checkpoints/stage_20_candidate_seg.json",
        {"stage": "candidate_seg", "result": candidate_seg_receipt, "complete": True},
    )
    del segnet

    posenet = retained.load_posenet(device)
    pose_receipts: dict[str, dict[str, Any]] = {}
    for ordinal, source in enumerate(("re1_round_01_candidate", "re1_round_01_candidate_repeat"), start=30):
        pose_receipts[source] = retained.score_pose_vectors(
            source=source,
            raw_root=Path(receiver["raw"]["path"]).parent,
            raw_record=receiver["raw"],
            scorer=posenet,
            device=device,
            run_root=run_root,
        )
        retained.checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal}_{source}_pose.json",
            {"stage": f"{source}_pose", "result": pose_receipts[source], "complete": True},
        )
    del posenet

    gt_field = Path(base_records["gt_argmax"]["path"])
    base_field = Path(base_records["base_argmax"]["path"])
    candidate_field = Path(candidate_seg_receipt["argmax"]["path"])
    base_seg = field_metrics(gt_field, base_field)
    candidate_seg = field_metrics(gt_field, candidate_field)
    gt_pose_path = Path(base_records["gt_pose_vectors"]["path"])
    base_pose_path = Path(base_records["base_pose_vectors"]["path"])
    candidate_pose_path = Path(pose_receipts["re1_round_01_candidate"]["first6_vectors"]["path"])
    repeat_pose_path = Path(pose_receipts["re1_round_01_candidate_repeat"]["first6_vectors"]["path"])
    base_pose = pose_metrics(gt_pose_path, base_pose_path)
    candidate_pose = pose_metrics(gt_pose_path, candidate_pose_path)
    repeat_pose = pose_metrics(candidate_pose_path, repeat_pose_path)
    base_score = score(base_seg["d_seg"], base_pose["d_pose"], BASE_ARCHIVE_BYTES)
    candidate_score = score(candidate_seg["d_seg"], candidate_pose["d_pose"], CANDIDATE_ARCHIVE_BYTES)
    delta = score_delta(base_score, candidate_score)
    adjudication = adjudicate(delta["total"], base_pose["d_pose"], candidate_pose["d_pose"])

    final = {
        "schema": "ddm_re1x_round1_full_n600_eval_result.v1",
        "execution_status": "COMPLETE",
        "verdict": adjudication["verdict"],
        "verdict_scope": (
            "INSTANCE: RE1 Round 1 archive 7be3eb94 through runtime tree 63b93187 "
            "on the matched macOS-CPU advisory instrument"
        ),
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "inputs": inputs,
        "receiver": receiver,
        "raw_identity_to_cp135": {
            "candidate": receiver["raw"],
            "base": base_records["base_raw"],
            "byte_identical": (receiver["raw"]["sha256"] == base_records["base_raw"]["sha256"]),
        },
        "seg": {
            "base": base_seg,
            "candidate": candidate_seg,
            "candidate_receipt": candidate_seg_receipt,
        },
        "pose": {
            "base": base_pose,
            "candidate": candidate_pose,
            "candidate_repeat_vs_first": repeat_pose,
            "repeat_bit_identical": bool(repeat_pose_path.read_bytes() == candidate_pose_path.read_bytes()),
            "receipts": pose_receipts,
        },
        "score": {
            "base": base_score,
            "candidate": candidate_score,
            "delta_candidate_minus_base": delta,
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "gap_to_sub015_reference": GAP_TO_SUB015,
        },
        "adjudication": adjudication,
        "cross_axis_reference_not_used_in_delta": {
            "axis": "[contest-CUDA T4, n600]",
            "base_score": 0.16195513827824176,
            "base_flips": 34_970,
            "base_d_seg": 34_970 / TOTAL_SEG_PIXELS,
            "base_d_pose": 6.885642960696714e-6,
            "base_archive_bytes": BASE_ARCHIVE_BYTES,
            "reason": "candidate was not measured on T4; mixing these values into the macOS delta is forbidden",
        },
        "retention": {
            "run_root": str(run_root.resolve()),
            "candidate_raw": receiver["raw"],
            "gt_argmax": base_records["gt_argmax"],
            "base_argmax": base_records["base_argmax"],
            "candidate_argmax": candidate_seg_receipt["argmax"],
            "gt_pose_vectors": base_records["gt_pose_vectors"],
            "base_pose_vectors": base_records["base_pose_vectors"],
            "candidate_pose_vectors": pose_receipts["re1_round_01_candidate"]["first6_vectors"],
            "candidate_repeat_pose_vectors": pose_receipts["re1_round_01_candidate_repeat"]["first6_vectors"],
            "cleanup_policy": "certify-or-block; no generated payload deleted or moved",
        },
        "environment": environment,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(final_path, final)
    retained.checkpoint_once(final_checkpoint, final)
    write_fire_order(run_root, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--resume-from", default=RESUME_ID)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "RUN.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"RE1X physical run lock is held: {lock_path}") from exc
        atomic_json(
            run_root / "RUN_LOCK_ACQUIRED.json",
            {
                "schema": "ddm_re1x_physical_run_lock.v1",
                "path": str(lock_path),
                "pid": os.getpid(),
                "resume_from": args.resume_from,
                "acquired": True,
            },
        )
        try:
            result = run(run_root, args.resume_from)
        except retained.WorkerError as error:
            record_public_front_door_blocker(run_root, error)
            raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
