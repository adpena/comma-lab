#!/usr/bin/env python3
"""Decode and score the selected JS8 candidate at full n600 with retention.

The runner is crash-resumable at the receiver token checkpoint and every
30-pair scorer chunk.  It retains all materialized raw frames, scorer inputs,
SegNet logits/argmax fields, and PoseNet vectors.  The selected JS8 candidate
still uses MC36's frame 0, so its row is an uncompensated diagnostic rather
than the charter's final QS5-compensated admission candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js8_implicit_edge_conditioning as build

OUTPUT: Final = build.BULK_ROOT / "full_n600_v1"
SWEEP_RESULT: Final = build.BULK_ROOT / "scale_sweep_v1/SWEEP_RESULT.json"
BASE_RAW: Final = Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/output/0.raw")
BASE_RAW_SHA256: Final = "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9"
UPSTREAM: Final = REPO / "upstream"
VIDEO_NAMES: Final = UPSTREAM / "public_test_video_names.txt"
VIDEOS: Final = UPSTREAM / "videos"
N: Final = 600
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
EVAL_H: Final = 384
EVAL_W: Final = 512
CHUNK_PAIRS: Final = 30
BATCH_PAIRS: Final = 30
SEED: Final = build.SEED
RATE_DENOMINATOR: Final = 37_545_489
AXIS: Final = "[macOS-CPU frozen-SegNet+PoseNet advisory, n600] UNCOMPENSATED DIAGNOSTIC"


class JS8FullError(RuntimeError):
    """A full decode, retention, resume, or scorer invariant failed."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def selected() -> tuple[dict[str, Any], Path, Path]:
    result = json.loads(SWEEP_RESULT.read_text())
    row = result["selected"]
    if row["label"] != "scale_0p125" or row["adapter_scale"] != 0.125:
        raise JS8FullError("selected JS8 sweep row differs")
    archive = Path(row["payloads"]["archive"]["path"])
    runtime = Path(row["runtime"])
    if file_record(archive) != row["payloads"]["archive"]:
        raise JS8FullError("selected archive custody differs")
    if not (runtime / "inflate.sh").is_file():
        raise JS8FullError("selected adapted runtime is incomplete")
    return row, archive, runtime


def storage_preflight() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(OUTPUT)
    required = 45 * 1024**3
    result = {
        "schema": "ddm_js8_full_storage_preflight.v1",
        "store": str(OUTPUT),
        "free_bytes": usage.free,
        "required_free_bytes": required,
        "admitted": usage.free >= required,
        "retention": "candidate raw plus all target/base/candidate scorer payloads; certify-or-block",
    }
    atomic_json(OUTPUT / "STORAGE_PREFLIGHT.json", result)
    if not result["admitted"]:
        raise JS8FullError("full JS8 storage preflight refused")
    return result


def decode_candidate(row: dict[str, Any], archive: Path, runtime: Path) -> dict[str, Any]:
    receipt_path = OUTPUT / "decode/DECODE_RESULT.json"
    raw_path = OUTPUT / "decode/output/0.raw"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("complete") and file_record(raw_path) == receipt["raw"]:
            return receipt
        raise JS8FullError("existing decode receipt is incomplete or differs")
    input_dir = OUTPUT / "decode/input"
    output_dir = raw_path.parent
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist() != ["p", "ec1_latent.br", "js8_edge_gate.br"]:
            raise JS8FullError("selected archive member surface differs")
        atomic_bytes(input_dir / "p", bundle.read("p"))
    file_list = OUTPUT / "decode/file_list.txt"
    atomic_bytes(file_list, b"0.mkv\n")
    command = [str(runtime / "inflate.sh"), str(input_dir), str(output_dir), str(file_list)]
    prior_failure = OUTPUT / "decode/DECODE_FAILURE.json"
    attempt_index = 2 if prior_failure.is_file() else 1
    attempt_root = OUTPUT / f"decode/attempt_{attempt_index:04d}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PATH"] = f"{REPO / '.venv/bin'}:{environment.get('PATH', '')}"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=runtime,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    stdout_path = attempt_root / "inflate.stdout.log"
    stderr_path = attempt_root / "inflate.stderr.log"
    atomic_bytes(stdout_path, completed.stdout.encode())
    atomic_bytes(stderr_path, completed.stderr.encode())
    if completed.returncode:
        failure = {
            "schema": "ddm_js8_decode_failure.v1",
            "complete": False,
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "stdout": file_record(stdout_path),
            "stderr": file_record(stderr_path),
            "checkpoint_dir": str(output_dir / ".f26_cpu_checkpoints"),
        }
        atomic_json(attempt_root / "DECODE_FAILURE.json", failure)
        raise JS8FullError(f"selected JS8 receiver failed: rc={completed.returncode}")
    expected = N * 2 * CAMERA_H * CAMERA_W * 3
    if not raw_path.is_file() or raw_path.stat().st_size != expected:
        raise JS8FullError("selected JS8 raw byte count differs")
    receipt = {
        "schema": "ddm_js8_decode_result.v1",
        "complete": True,
        "axis": "[macOS-CPU 4-thread receiver decode]",
        "command": command,
        "elapsed_seconds": elapsed,
        "archive": file_record(archive),
        "runtime": str(runtime),
        "raw": file_record(raw_path),
        "stdout": file_record(stdout_path),
        "stderr": file_record(stderr_path),
        "checkpoint_payloads": [
            file_record(path) for path in sorted((output_dir / ".f26_cpu_checkpoints").rglob("*")) if path.is_file()
        ],
        "selected_scale": row["adapter_scale"],
    }
    atomic_json(receipt_path, receipt)
    return receipt


def import_upstream() -> tuple[Any, Any]:
    sys.path.insert(0, str(UPSTREAM))
    try:
        import frame_utils
        import modules
    finally:
        sys.path.pop(0)
    return frame_utils, modules


def scorer_models(modules: Any) -> tuple[Any, Any]:
    segnet = modules.SegNet().eval()
    posenet = modules.PoseNet().eval()
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device="cpu"))
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device="cpu"))
    return segnet, posenet


def specs(size: int) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    result = {}
    for role in ("target", "base", "candidate_uncompensated"):
        result.update(
            {
                f"{role}_camera.uint8.npy": (np.dtype(np.uint8), (size, 2, CAMERA_H, CAMERA_W, 3)),
                f"{role}_seg_input.float32.npy": (np.dtype(np.float32), (size, 3, EVAL_H, EVAL_W)),
                f"{role}_seg_logits.float32.npy": (np.dtype(np.float32), (size, 5, EVAL_H, EVAL_W)),
                f"{role}_argmax.uint8.npy": (np.dtype(np.uint8), (size, EVAL_H, EVAL_W)),
                f"{role}_pose_input.float32.npy": (np.dtype(np.float32), (size, 12, EVAL_H // 2, EVAL_W // 2)),
                f"{role}_pose.float32.npy": (np.dtype(np.float32), (size, 6)),
            }
        )
    return result


def open_arrays(root: Path, size: int) -> dict[str, np.memmap]:
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, (dtype, shape) in specs(size).items():
        path = root / name
        if path.exists():
            raise JS8FullError(f"active chunk payload already exists: {path}")
        arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return arrays


def fresh_chunk_root(pair_start: int) -> Path:
    """Preserve interrupted arrays and select a never-written attempt directory."""
    base = OUTPUT / f"scorer/chunks/pairs_{pair_start:04d}_{pair_start + CHUNK_PAIRS - 1:04d}"
    if not base.exists():
        return base
    if (base / "CHUNK_RESULT.json").is_file():
        raise JS8FullError(f"completed chunk exists outside resume ledger: {base}")
    partials = [file_record(path) for path in sorted(base.glob("*.npy")) if path.is_file()]
    interrupted = {
        "schema": "ddm_js8_interrupted_scorer_chunk.v1",
        "complete": False,
        "reason": "operator stopped after the prior durable chunk to increase underutilized CPU batch size",
        "pair_start": pair_start,
        "payloads_preserved": partials,
        "disposition": "RESIDUE-rebuildable-but-retained; excluded from scorer aggregation",
    }
    receipt = base / "INTERRUPTED_PARTIAL.json"
    if not receipt.exists():
        atomic_json(receipt, interrupted)
    attempt = 2
    while True:
        candidate = base.with_name(f"{base.name}_attempt_{attempt:04d}")
        if not candidate.exists():
            return candidate
        attempt += 1


def score_role(frames: torch.Tensor, segnet: Any, posenet: Any) -> dict[str, np.ndarray]:
    value = frames.permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        seg_input = segnet.preprocess_input(value)
        seg_logits = segnet(seg_input)
        pose_input = posenet.preprocess_input(value)
        pose = posenet(pose_input)["pose"][..., :6]
    return {
        "seg_input.float32.npy": seg_input.numpy().astype(np.float32, copy=False),
        "seg_logits.float32.npy": seg_logits.numpy().astype(np.float32, copy=False),
        "argmax.uint8.npy": seg_logits.argmax(dim=1).numpy().astype(np.uint8, copy=False),
        "pose_input.float32.npy": pose_input.numpy().astype(np.float32, copy=False),
        "pose.float32.npy": pose.numpy().astype(np.float32, copy=False),
    }


def finalize_chunk(root: Path, size: int) -> dict[str, Any]:
    records = {name: file_record(root / name) for name in specs(size)}
    receipt = {"schema": "ddm_js8_scorer_chunk.v1", "complete": True, "payloads": records}
    atomic_json(root / "CHUNK_RESULT.json", receipt)
    return {**receipt, "receipt": file_record(root / "CHUNK_RESULT.json")}


def score_n600(decode: dict[str, Any], archive_bytes: int) -> dict[str, Any]:
    if not BASE_RAW.is_file() or BASE_RAW.stat().st_size != N * 2 * CAMERA_H * CAMERA_W * 3:
        raise JS8FullError("MC36 base raw custody is missing or has the wrong byte count")
    base_raw = file_record(BASE_RAW)
    if base_raw["sha256"] != BASE_RAW_SHA256:
        raise JS8FullError("MC36 base raw custody hash differs")
    progress_path = OUTPUT / "scorer/PROGRESS.json"
    result_path = OUTPUT / "FULL_RESULT.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        if (
            existing.get("schema") != "ddm_js8_full_n600_advisory.v1"
            or existing.get("status") != "UNCOMPENSATED_FULL_N600_DIAGNOSTIC_COMPLETE"
            or existing.get("axis") != AXIS
            or existing.get("decode", {}).get("raw") != decode["raw"]
            or existing.get("rows", {}).get("base", {}).get("archive_bytes") != build.BASE_ARCHIVE_BYTES
            or existing.get("rows", {}).get("candidate_uncompensated", {}).get("archive_bytes") != archive_bytes
            or int(progress.get("completed_pairs", -1)) != N
            or existing.get("scorer_progress") != file_record(progress_path)
        ):
            raise JS8FullError("existing full result is incomplete or differs; preserve it")
        return existing
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.is_file()
        else {
            "schema": "ddm_js8_scorer_progress.v1",
            "completed_pairs": 0,
            "chunks": [],
        }
    )
    completed_pairs = int(progress["completed_pairs"])
    if completed_pairs % CHUNK_PAIRS or completed_pairs < 0 or completed_pairs > N:
        raise JS8FullError("scorer resume offset differs")
    frame_utils, modules = import_upstream()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    segnet, posenet = scorer_models(modules)
    names = VIDEO_NAMES.read_text().splitlines()
    if names != ["0.mkv"]:
        raise JS8FullError("public video census differs")
    datasets = [
        frame_utils.AVVideoDataset(
            names,
            data_dir=VIDEOS,
            batch_size=BATCH_PAIRS,
            device=torch.device("cpu"),
            num_threads=2,
            seed=SEED,
            prefetch_queue_depth=2,
        ),
        frame_utils.TensorVideoDataset(
            names,
            data_dir=BASE_RAW.parent,
            batch_size=BATCH_PAIRS,
            device=torch.device("cpu"),
            num_threads=2,
            seed=SEED,
            prefetch_queue_depth=2,
        ),
        frame_utils.TensorVideoDataset(
            names,
            data_dir=Path(decode["raw"]["path"]).parent,
            batch_size=BATCH_PAIRS,
            device=torch.device("cpu"),
            num_threads=2,
            seed=SEED,
            prefetch_queue_depth=2,
        ),
    ]
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0) for dataset in datasets]
    global_pair = 0
    arrays: dict[str, np.memmap] | None = None
    chunk_root: Path | None = None
    chunk_start = completed_pairs
    for target_row, base_row, candidate_row in zip(*loaders, strict=True):
        target_frames = target_row[2]
        base_frames = base_row[2]
        candidate_frames = candidate_row[2]
        batch = len(target_frames)
        if not (len(base_frames) == len(candidate_frames) == batch):
            raise JS8FullError("paired scorer batch sizes differ")
        if global_pair + batch <= completed_pairs:
            global_pair += batch
            continue
        if global_pair < completed_pairs:
            raise JS8FullError("resume offset is not batch-aligned")
        if arrays is None:
            chunk_start = global_pair
            chunk_root = fresh_chunk_root(chunk_start)
            arrays = open_arrays(chunk_root, CHUNK_PAIRS)
        local = global_pair - chunk_start
        if local + batch > CHUNK_PAIRS:
            raise JS8FullError("scorer batch crosses chunk boundary")
        for role, frames in (
            ("target", target_frames),
            ("base", base_frames),
            ("candidate_uncompensated", candidate_frames),
        ):
            arrays[f"{role}_camera.uint8.npy"][local : local + batch] = frames.numpy()
            for suffix, value in score_role(frames, segnet, posenet).items():
                arrays[f"{role}_{suffix}"][local : local + batch] = value
        for value in arrays.values():
            value.flush()
        global_pair += batch
        if global_pair - chunk_start == CHUNK_PAIRS:
            del arrays
            arrays = None
            if chunk_root is None:
                raise AssertionError("chunk root missing")
            receipt = finalize_chunk(chunk_root, CHUNK_PAIRS)
            progress["completed_pairs"] = global_pair
            progress["chunks"].append(receipt)
            progress["updated_at_utc"] = utc_now()
            atomic_json(progress_path, progress)
            chunk_root = None
    if global_pair != N or arrays is not None or progress["completed_pairs"] != N:
        raise JS8FullError(f"scorer population incomplete: {global_pair}")

    confusion = {"base": np.zeros((5, 5), dtype=np.int64), "candidate_uncompensated": np.zeros((5, 5), dtype=np.int64)}
    flip_counts = {"base": 0, "candidate_uncompensated": 0}
    pose_squared = {"base": 0.0, "candidate_uncompensated": 0.0}
    pose_values = 0
    for chunk in progress["chunks"]:
        payloads = chunk["payloads"]
        target_argmax = np.load(payloads["target_argmax.uint8.npy"]["path"], mmap_mode="r")
        target_pose = np.load(payloads["target_pose.float32.npy"]["path"], mmap_mode="r")
        for role in ("base", "candidate_uncompensated"):
            predicted = np.load(payloads[f"{role}_argmax.uint8.npy"]["path"], mmap_mode="r")
            pose = np.load(payloads[f"{role}_pose.float32.npy"]["path"], mmap_mode="r")
            flip_counts[role] += int(np.count_nonzero(predicted != target_argmax))
            pose_squared[role] += float(np.sum((pose.astype(np.float64) - target_pose.astype(np.float64)) ** 2))
            flat = target_argmax.astype(np.int64).ravel() * 5 + predicted.astype(np.int64).ravel()
            confusion[role] += np.bincount(flat, minlength=25).reshape(5, 5)
        pose_values += int(target_pose.size)
    rows = {}
    for role, bytes_value in (("base", build.BASE_ARCHIVE_BYTES), ("candidate_uncompensated", archive_bytes)):
        d_seg = flip_counts[role] / (N * EVAL_H * EVAL_W)
        d_pose = pose_squared[role] / pose_values
        rows[role] = {
            "flips": flip_counts[role],
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_value,
            "seg_term": 100.0 * d_seg,
            "pose_term": math.sqrt(10.0 * d_pose),
            "rate_term": 25.0 * bytes_value / RATE_DENOMINATOR,
            "S": 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * bytes_value / RATE_DENOMINATOR,
            "confusion_gt_by_rendered": confusion[role].tolist(),
        }
    result = {
        "schema": "ddm_js8_full_n600_advisory.v1",
        "status": "UNCOMPENSATED_FULL_N600_DIAGNOSTIC_COMPLETE",
        "axis": AXIS,
        "selection_mode": "full population n600; 30-pair retained chunks; candidate selected by prior n32 Seg+rate screen",
        "rows": rows,
        "delta_candidate_minus_base": {
            key: rows["candidate_uncompensated"][key] - rows["base"][key]
            for key in ("flips", "d_seg", "d_pose", "archive_bytes", "seg_term", "pose_term", "rate_term", "S")
        },
        "decode": decode,
        "base_raw": base_raw,
        "scorer_progress": file_record(progress_path),
        "boundaries": {
            "real_receiver_decode": True,
            "full_n600": True,
            "pose_compensation": False,
            "score_claim": False,
            "pointer_moved": False,
            "verdict_scope": "INSTANCE diagnostic only; cannot admit or family-kill before QS5 compensation",
        },
    }
    atomic_json(result_path, result)
    atomic_json(build.LOGICAL_ROOT / "FULL_POINTER.json", {"result": file_record(result_path)})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--stage", choices=("decode", "score", "all"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume_from.resolve() != (OUTPUT / "RESUME.json").resolve():
        raise JS8FullError("--resume-from must name the canonical JS8 resume receipt")
    storage = storage_preflight()
    row, archive, runtime = selected()
    resume = {
        "schema": "ddm_js8_full_resume.v1",
        "storage": storage,
        "selected": row["label"],
        "updated_at_utc": utc_now(),
    }
    atomic_json(args.resume_from, resume)
    decode = decode_candidate(row, archive, runtime)
    if args.stage == "decode":
        print(json.dumps(decode, indent=2, sort_keys=True))
        return 0
    result = score_n600(decode, int(row["archive_bytes"]))
    print(
        json.dumps(
            {key: result[key] for key in ("status", "axis", "rows", "delta_candidate_minus_base", "boundaries")},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
