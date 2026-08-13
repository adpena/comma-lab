#!/usr/bin/env python3
"""Retained, resumable macOS-CPU full-n600 evaluation for PZ4R direct-v6.

The run uses the shipped ``inflate.sh`` front door, retains the exact decoded
raw video, and evaluates PZ4R plus the retained CP135 base raw through matched
frozen CPU-torch SegNet and PoseNet instruments.  Every scorer input/output
payload is retained; the result is advisory and never promoted to contest
authority.
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
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval"
)
PZ4R_RUNTIME: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/submission"
)
PZ4R_ARCHIVE: Final = PZ4R_RUNTIME / "archive.zip"
PZ4R_ARCHIVE_BYTES: Final = 183_137
PZ4R_ARCHIVE_SHA256: Final = "c408adf9101bb19a363039a5e0f7185aabce8f31edb6787e2deaf6d0fe6738f4"
BASE_ARCHIVE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip")
BASE_ARCHIVE_BYTES: Final = 186_252
BASE_ARCHIVE_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_RAW: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/candidates/cp135_base/retained/0.raw"
)
BASE_RAW_BYTES: Final = 3_662_409_600
BASE_RAW_SHA256: Final = "a641d1ef149f8da8f06af3da9234d6d2f6be9702c3f606b7acf838b4b298ed47"
UNCOMPRESSED_BYTES: Final = 37_545_489
PAIR_COUNT: Final = 600
SEG_PIXELS_PER_PAIR: Final = 384 * 512
TOTAL_SEG_PIXELS: Final = PAIR_COUNT * SEG_PIXELS_PER_PAIR
GAP_TO_SUB015: Final = 0.01195513827824177
AXIS: Final = "[macOS-CPU advisory, frozen CPU-torch SegNet+PoseNet, n600] NON-PROMOTABLE"
RESUME_ID: Final = "ddm_pz4r_full_n600_eval_20260813"
EXPECTED_PAYLOAD_BYTES: Final = (
    retained.RAW_BYTES
    + retained.RAW_BYTES
    + 3 * (retained.SEG_INPUT_BYTES + retained.LOGIT_BYTES + retained.FIELD_DATA_BYTES + 128)
    + 4 * (PAIR_COUNT * 12 * 384 * 512 * 4 + PAIR_COUNT * 12 * 4 + PAIR_COUNT * 6 * 4)
)
RESERVE_BYTES: Final = 8 * 1024**3


class PZ4REvalError(RuntimeError):
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
        raise PZ4REvalError(f"required input is missing: {path}")
    record = file_record(path)
    if record["bytes"] != size or record["sha256"] != digest:
        raise PZ4REvalError(f"required input custody differs: {record}")
    return record


def atomic_json(path: Path, value: Any) -> None:
    retained.atomic_json(path, value)


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


def storage_preflight(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    already = retained_bytes(run_root)
    remaining = max(0, EXPECTED_PAYLOAD_BYTES - already)
    required = remaining + RESERVE_BYTES
    result = {
        "schema": "ddm_pz4r_full_n600_storage_preflight.v1",
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
        raise PZ4REvalError(f"storage preflight failed: free={usage.free}, required={required}")
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


def validate_resumed_inputs(inputs: dict[str, Any]) -> None:
    """Bind a resume to immutable stage-0 custody, not volatile git status."""
    if inputs.get("resume_from") != RESUME_ID or inputs.get("axis") != AXIS:
        raise PZ4REvalError("resumed stage-0 identity differs")
    for name, path in (
        ("candidate_archive", PZ4R_ARCHIVE),
        ("base_archive", BASE_ARCHIVE),
        ("base_raw", BASE_RAW),
    ):
        retained.require_record(path, inputs[name])
    upstream_records = inputs["upstream"]
    upstream_paths = {
        "evaluate_py": UPSTREAM / "evaluate.py",
        "frame_utils_py": UPSTREAM / "frame_utils.py",
        "modules_py": UPSTREAM / "modules.py",
        "segnet_weights": UPSTREAM / "models/segnet.safetensors",
        "posenet_weights": UPSTREAM / "models/posenet.safetensors",
        "gt_video": UPSTREAM / "videos/0.mkv",
    }
    for name, path in upstream_paths.items():
        retained.require_record(path, upstream_records[name])


def launch_contract(run_root: Path) -> dict[str, Any]:
    deps = Path(os.environ.get("PR130_RUNTIME_DEPS_DIR", "")).resolve()
    result = {
        "schema": "ddm_pz4r_full_n600_launch_contract.v1",
        "python": sys.executable,
        "receiver_python": os.environ.get("PYTHON", "python3 resolved through runner PATH"),
        "runtime_dependencies": str(deps),
        "brotli_cli": os.environ.get("PR130_BROTLI_CLI", ""),
        "inflate_device": os.environ.get("PR130_INFLATE_DEVICE", ""),
        "token_cache": os.environ.get("PR130_TOKEN_CACHE", ""),
        "token_receipt": os.environ.get("PR130_TOKEN_RECEIPT", ""),
        "threads": {
            name: os.environ.get(name, "")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "retention": "dependency closure and every generated receiver/scorer payload remain under the run root",
    }
    if deps == Path(".").resolve() or run_root not in deps.parents:
        raise PZ4REvalError("runtime dependency closure must be a retained child of the run root")
    atomic_json(run_root / "LAUNCH_CONTRACT.json", result)
    retained.checkpoint_once(run_root / "checkpoints/stage_05_launch_contract.json", result)
    return result


def field_metrics(gt_path: Path, value_path: Path) -> dict[str, Any]:
    gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    value = np.load(value_path, mmap_mode="r", allow_pickle=False)
    if gt.shape != (PAIR_COUNT, 384, 512) or value.shape != gt.shape:
        raise PZ4REvalError("SegNet argmax field geometry differs")
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
        raise PZ4REvalError("PoseNet first-six vector geometry differs")
    error = value - gt
    return {
        "pairs": PAIR_COUNT,
        "components_per_pair": 6,
        "d_pose": float(np.mean(error * error, dtype=np.float32)),
        "gt_vectors": file_record(gt_path),
        "value_vectors": file_record(value_path),
    }


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


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    if resume_from != RESUME_ID:
        raise PZ4REvalError(f"resume id must be {RESUME_ID!r}")
    run_root.mkdir(parents=True, exist_ok=True)
    final_path = run_root / "PZ4R_FULL_N600_RESULT.json"
    final_checkpoint = run_root / "checkpoints/stage_40_final.json"
    if final_path.is_file() and final_checkpoint.is_file():
        final = json.loads(final_path.read_text())
        if final_checkpoint.read_bytes() != retained.canonical_json_bytes(final):
            raise PZ4REvalError("completed result and final checkpoint differ")
        for record in final["retention"].values():
            if isinstance(record, dict) and {"path", "bytes", "sha256"} <= set(record):
                retained.require_record(Path(record["path"]), record)
        return final
    storage = storage_preflight(run_root)
    inputs_path = run_root / "INPUTS.json"
    inputs_checkpoint = run_root / "checkpoints/stage_00_inputs.json"
    if inputs_path.is_file() and inputs_checkpoint.is_file():
        inputs = json.loads(inputs_path.read_text())
        validate_resumed_inputs(inputs)
        checkpoint_inputs = {key: value for key, value in inputs.items() if key != "storage_preflight"}
        if inputs_checkpoint.read_bytes() != retained.canonical_json_bytes(checkpoint_inputs):
            raise PZ4REvalError("INPUTS.json and immutable stage-0 checkpoint differ")
        archive_record = inputs["candidate_archive"]
        base_raw_record = inputs["base_raw"]
    else:
        archive_record = require_file(PZ4R_ARCHIVE, size=PZ4R_ARCHIVE_BYTES, digest=PZ4R_ARCHIVE_SHA256)
        base_archive_record = require_file(BASE_ARCHIVE, size=BASE_ARCHIVE_BYTES, digest=BASE_ARCHIVE_SHA256)
        base_raw_record = require_file(BASE_RAW, size=BASE_RAW_BYTES, digest=BASE_RAW_SHA256)
        inputs = {
            "schema": "ddm_pz4r_full_n600_inputs.v1",
            "resume_from": resume_from,
            "axis": AXIS,
            "candidate_archive": archive_record,
            "candidate_runtime": tree_record(PZ4R_RUNTIME),
            "base_archive": base_archive_record,
            "base_raw": base_raw_record,
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
        }
        atomic_json(inputs_path, inputs)
        retained.checkpoint_once(
            inputs_checkpoint,
            {key: value for key, value in inputs.items() if key != "storage_preflight"},
        )
    launch = launch_contract(run_root)

    retained.UPSTREAM = UPSTREAM
    receiver = retained.decode_exact_receiver(
        name="pz4r_candidate",
        archive_path=PZ4R_ARCHIVE,
        archive_record=archive_record,
        runtime_root=PZ4R_RUNTIME,
        run_root=run_root,
    )
    retained.checkpoint_once(
        run_root / "checkpoints/stage_10_candidate_decoded.json",
        {"stage": "candidate_decoded", "receiver": receiver, "complete": True},
    )
    runtime_closure = tree_record(Path(launch["runtime_dependencies"]))
    retained.checkpoint_once(
        run_root / "checkpoints/stage_11_runtime_closure.json",
        {"stage": "runtime_closure", "runtime_closure": runtime_closure, "complete": True},
    )

    import torch

    torch.set_num_threads(4)
    torch.manual_seed(retained.SEED)
    np.random.seed(retained.SEED)
    device = torch.device("cpu")

    segnet = retained.load_segnet(device)
    seg: dict[str, dict[str, Any]] = {}
    seg_sources = (
        ("gt", None, None),
        ("cp135_base", BASE_RAW.parent, base_raw_record),
        ("pz4r_candidate", Path(receiver["raw"]["path"]).parent, receiver["raw"]),
    )
    for ordinal, (name, raw_root, raw_record) in enumerate(seg_sources, start=20):
        seg[name] = retained.score_argmax_field(
            source=name,
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=segnet,
            device=device,
            run_root=run_root,
        )
        retained.checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal}_{name}_seg.json",
            {"stage": f"{name}_seg", "result": seg[name], "complete": True},
        )
    del segnet

    posenet = retained.load_posenet(device)
    pose: dict[str, dict[str, Any]] = {}
    pose_sources = (
        ("gt", None, None),
        ("cp135_base", BASE_RAW.parent, base_raw_record),
        ("pz4r_candidate", Path(receiver["raw"]["path"]).parent, receiver["raw"]),
        ("pz4r_candidate_repeat", Path(receiver["raw"]["path"]).parent, receiver["raw"]),
    )
    for ordinal, (name, raw_root, raw_record) in enumerate(pose_sources, start=30):
        pose[name] = retained.score_pose_vectors(
            source=name,
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=posenet,
            device=device,
            run_root=run_root,
        )
        retained.checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal}_{name}_pose.json",
            {"stage": f"{name}_pose", "result": pose[name], "complete": True},
        )
    del posenet

    gt_field = Path(seg["gt"]["argmax"]["path"])
    base_field = Path(seg["cp135_base"]["argmax"]["path"])
    candidate_field = Path(seg["pz4r_candidate"]["argmax"]["path"])
    base_seg = field_metrics(gt_field, base_field)
    candidate_seg = field_metrics(gt_field, candidate_field)
    gt_pose = Path(pose["gt"]["first6_vectors"]["path"])
    base_pose_path = Path(pose["cp135_base"]["first6_vectors"]["path"])
    candidate_pose_path = Path(pose["pz4r_candidate"]["first6_vectors"]["path"])
    repeat_pose_path = Path(pose["pz4r_candidate_repeat"]["first6_vectors"]["path"])
    base_pose = pose_metrics(gt_pose, base_pose_path)
    candidate_pose = pose_metrics(gt_pose, candidate_pose_path)
    repeat_pose = pose_metrics(candidate_pose_path, repeat_pose_path)
    base_score = score(base_seg["d_seg"], base_pose["d_pose"], BASE_ARCHIVE_BYTES)
    candidate_score = score(candidate_seg["d_seg"], candidate_pose["d_pose"], PZ4R_ARCHIVE_BYTES)
    delta = score_delta(base_score, candidate_score)
    if delta["total"] < 0.0:
        verdict = "ADVISORY_BETTER_THAN_CP135_SAME_INSTRUMENT_T4_CONFIRMATION_EARNED"
        disposition = "QUEUED-WITH-A-FIRE-ORDER"
    else:
        verdict = "INSTANCE_CLOSED_REALIZED_DISTORTION_EXCEEDS_RATE_CREDIT"
        disposition = "FOLDED"

    final = {
        "schema": "ddm_pz4r_full_n600_eval_result.v1",
        "execution_status": "COMPLETE",
        "verdict": verdict,
        "verdict_scope": "INSTANCE: PZ4R direct_v6 archive c408adf9 on the macOS-CPU advisory instrument",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "inputs": inputs,
        "launch_contract": launch,
        "runtime_closure": runtime_closure,
        "receiver": receiver,
        "seg": {"base": base_seg, "candidate": candidate_seg, "receipts": seg},
        "pose": {
            "base": base_pose,
            "candidate": candidate_pose,
            "candidate_repeat_vs_first": repeat_pose,
            "repeat_bit_identical": bool(repeat_pose_path.read_bytes() == candidate_pose_path.read_bytes()),
            "receipts": pose,
        },
        "score": {
            "base": base_score,
            "candidate": candidate_score,
            "delta_candidate_minus_base": delta,
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "gap_to_sub015_reference": GAP_TO_SUB015,
        },
        "cross_axis_reference_not_used_in_delta": {
            "axis": "[contest-CUDA T4, n600]",
            "base_flips": 34_970,
            "base_d_seg": 34_970 / TOTAL_SEG_PIXELS,
            "base_d_pose": 6.885642960696714e-6,
            "base_archive_bytes": BASE_ARCHIVE_BYTES,
            "reason": "candidate was not measured on T4; mixing these values into the macOS delta is forbidden",
        },
        "charter_rate_reconciliation": {
            "candidate_vs_cp135_bytes": PZ4R_ARCHIVE_BYTES - BASE_ARCHIVE_BYTES,
            "candidate_vs_cp135_rate_delta_s": 25.0 * (PZ4R_ARCHIVE_BYTES - BASE_ARCHIVE_BYTES) / UNCOMPRESSED_BYTES,
            "candidate_vs_lc2_bytes": PZ4R_ARCHIVE_BYTES - 187_226,
            "note": "-4,089 B is the LC2 comparison; the charter's 186,252 B CP135 comparator gives -3,115 B",
        },
        "follow_on_disposition": disposition,
        "retention": {
            "run_root": str(run_root.resolve()),
            "candidate_raw": receiver["raw"],
            "gt_argmax": seg["gt"]["argmax"],
            "base_argmax": seg["cp135_base"]["argmax"],
            "candidate_argmax": seg["pz4r_candidate"]["argmax"],
            "gt_pose_vectors": pose["gt"]["first6_vectors"],
            "base_pose_vectors": pose["cp135_base"]["first6_vectors"],
            "candidate_pose_vectors": pose["pz4r_candidate"]["first6_vectors"],
            "candidate_repeat_pose_vectors": pose["pz4r_candidate_repeat"]["first6_vectors"],
            "cleanup_policy": "certify-or-block; no generated payload deleted or moved",
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(final_path, final)
    retained.checkpoint_once(final_checkpoint, final)
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
            raise SystemExit(f"PZ4R physical run lock is held: {lock_path}") from exc
        atomic_json(
            run_root / "RUN_LOCK_ACQUIRED.json",
            {
                "schema": "ddm_pz4r_physical_run_lock.v1",
                "path": str(lock_path),
                "pid": os.getpid(),
                "resume_from": args.resume_from,
                "acquired": True,
            },
        )
        result = run(run_root, args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
