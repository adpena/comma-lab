#!/usr/bin/env python3
"""Locked-cu128 T4 worker for retained JS1B n600 SegNet argmax fields.

The worker runs each archive's exact adapted ``inflate.sh`` receiver, retains
both full raw videos, and evaluates GT plus both candidate videos through the
frozen upstream SegNet at the promoted batch size of 16.  Every materialized
RGB GT batch, SegNet input, logit tensor, and argmax field remains on the
mounted Modal volume.  Immutable stage checkpoints and per-batch progress make
the run resumable from disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np

REMOTE_REPO: Final = Path("/workspace/pact")
UPSTREAM: Final = REMOTE_REPO / "upstream"
AXIS: Final = "[contest-CUDA T4 frozen-SegNet argmax fields, n600, batch=16] COMPONENT-ONLY"
N_PAIRS: Final = 600
BATCH_SIZE: Final = 16
SEG_HEIGHT: Final = 384
SEG_WIDTH: Final = 512
CAMERA_HEIGHT: Final = 874
CAMERA_WIDTH: Final = 1164
CLASSES: Final = 5
SEED: Final = 1234
RAW_BYTES: Final = N_PAIRS * 2 * CAMERA_HEIGHT * CAMERA_WIDTH * 3
FIELD_DATA_BYTES: Final = N_PAIRS * SEG_HEIGHT * SEG_WIDTH
SEG_INPUT_BYTES: Final = N_PAIRS * 3 * SEG_HEIGHT * SEG_WIDTH * 4
LOGIT_BYTES: Final = N_PAIRS * CLASSES * SEG_HEIGHT * SEG_WIDTH * 4
EXPECTED_RETAINED_PAYLOAD_BYTES: Final = (
    2 * RAW_BYTES
    + RAW_BYTES
    + 3 * SEG_INPUT_BYTES
    + 3 * LOGIT_BYTES
    + 4 * (FIELD_DATA_BYTES + 128)
)
STORAGE_RESERVE_BYTES: Final = 4 * 1024**3
CP135_SHA256: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
CP135_BYTES: Final = 186_252
T1R1_SHA256: Final = "12a5b181fef4e15ad8a752161c744347beca0b5a1224c5d3d542ab148f6ece80"
T1R1_BYTES: Final = 187_046
C1_TARGET_SHA256: Final = "a9c4936c41bc6634477f9c060be3d170542bd2a1d4d0cd04d5afcd0912fb3908"
C1_TARGET_BYTES: Final = 117_964_928
EXPECTED_CP135_FLIPS: Final = 34_964
EXPECTED_C1_TARGET_FLIPS: Final = 17_926


class WorkerError(RuntimeError):
    """A receiver, scorer, retention, or resume invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json_bytes(value))


def atomic_npy(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    with staging.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    return file_record(path)


def checkpoint_once(path: Path, value: Any) -> None:
    """Create an immutable checkpoint or require byte-identical resume state."""
    payload = canonical_json_bytes(value)
    if path.is_file():
        if path.read_bytes() != payload:
            raise WorkerError(f"resume checkpoint differs: {path}")
        return
    atomic_bytes(path, payload)


def require_record(path: Path, record: dict[str, Any]) -> None:
    if not path.is_file():
        raise WorkerError(f"retained payload is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise WorkerError(f"retained payload size differs: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise WorkerError(f"retained payload SHA-256 differs: {path}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise WorkerError(f"unsafe ZIP member: {info.filename}")
            if info.is_dir():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(target, archive.read(info))
            os.chmod(target, (info.external_attr >> 16) & 0o777 or 0o644)


def extract_zip_once(path: Path, destination: Path, marker_name: str) -> dict[str, Any]:
    marker = destination / marker_name
    source = file_record(path)
    if marker.is_file():
        result = json.loads(marker.read_text())
        if result.get("source") != source or not result.get("complete"):
            raise WorkerError(f"extraction checkpoint differs: {marker}")
        return result
    _safe_extract(path, destination)
    result = {"source": source, "complete": True}
    atomic_json(marker, result)
    return result


def current_retained_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def storage_preflight(run_root: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(run_root)
    already_retained = current_retained_bytes(run_root)
    remaining_payload = max(0, EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained)
    required_free = remaining_payload + STORAGE_RESERVE_BYTES
    result = {
        "schema": "ddm_js1b_storage_preflight.v1",
        "tier": str(run_root),
        "free_bytes": usage.free,
        "already_retained_bytes": already_retained,
        "expected_total_retained_payload_bytes": EXPECTED_RETAINED_PAYLOAD_BYTES,
        "remaining_payload_bytes": remaining_payload,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required_free,
        "passed": usage.free >= required_free,
        "cleanup_policy": (
            "block rather than delete; all source, raw, scorer, and field payloads retained"
        ),
    }
    atomic_json(run_root / "STORAGE_PREFLIGHT.json", result)
    if not result["passed"]:
        raise WorkerError(
            f"storage preflight failed: free={usage.free}, required={required_free}"
        )
    return result


def install_archive(runtime_root: Path, archive_path: Path, expected: dict[str, Any]) -> None:
    require_record(archive_path, expected)
    destination = runtime_root / "archive.zip"
    if destination.is_file():
        require_record(destination, expected)
        return
    atomic_bytes(destination, archive_path.read_bytes())
    require_record(destination, expected)


def decode_exact_receiver(
    *,
    name: str,
    archive_path: Path,
    archive_record: dict[str, Any],
    runtime_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Run one unmodified adapted inflate.sh and retain its full raw output."""
    receipt_path = run_root / f"receivers/{name}/RECEIVER_RESULT.json"
    raw_path = run_root / f"retained/raw/{name}/0.raw"
    if receipt_path.is_file():
        result = json.loads(receipt_path.read_text())
        require_record(raw_path, result["raw"])
        return result

    if raw_path.exists():
        # The exact receiver refuses overwrite.  A killed prior stage may have
        # left either a partial or a complete-but-unreceipted raw.  Preserve it
        # losslessly as a distinct attempt before the deterministic rerun.
        attempt_record = file_record(raw_path)
        attempt_base = (
            run_root
            / f"retained/raw/{name}/failed_attempts/"
            / f"0.raw.unreceipted.{attempt_record['sha256'][:16]}"
        )
        attempt_path = attempt_base
        suffix = 1
        while attempt_path.exists():
            attempt_path = attempt_base.with_name(f"{attempt_base.name}.{suffix:03d}")
            suffix += 1
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw_path, attempt_path)
        atomic_json(
            attempt_path.with_name(attempt_path.name + ".json"),
            {
                "schema": "ddm_js1b_unreceipted_receiver_attempt.v1",
                "reason": "resume found receiver bytes without a completed receiver receipt",
                "payload": file_record(attempt_path),
                "disposition": "retained losslessly; deterministic exact-receiver rerun",
            },
        )

    require_record(archive_path, archive_record)
    bound_archive_record = file_record(archive_path)
    install_archive(runtime_root, archive_path, archive_record)
    data_dir = run_root / f"work/{name}/archive"
    data_marker = data_dir / "ARCHIVE_EXTRACTED.json"
    if not data_marker.is_file():
        _safe_extract(archive_path, data_dir)
        atomic_json(data_marker, {"archive": bound_archive_record, "complete": True})
    file_list = run_root / f"work/{name}/public_test_video_names.txt"
    video_names = (UPSTREAM / "public_test_video_names.txt").read_bytes()
    if video_names.splitlines() != [b"0.mkv"]:
        raise WorkerError("upstream public video-name census differs from the pinned one-object set")
    atomic_bytes(file_list, video_names)
    output_dir = raw_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        str(runtime_root / "inflate.sh"),
        str(data_dir),
        str(output_dir),
        str(file_list),
    ]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=runtime_root,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
            "PYTHONHASHSEED": str(SEED),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started
    log_path = run_root / f"receivers/{name}/inflate.log"
    atomic_bytes(
        log_path,
        (completed.stdout + "\n--- STDERR ---\n" + completed.stderr).encode(
            "utf-8", errors="replace"
        ),
    )
    if completed.returncode:
        raise WorkerError(f"exact receiver failed for {name}; inspect {log_path}")
    if not raw_path.is_file() or raw_path.stat().st_size != RAW_BYTES:
        raise WorkerError(f"exact receiver raw size differs for {name}: {raw_path}")
    result = {
        "schema": "ddm_js1b_exact_receiver_result.v1",
        "name": name,
        "archive": bound_archive_record,
        "runtime_inflate_sh": file_record(runtime_root / "inflate.sh"),
        "argv": command,
        "cwd": str(runtime_root),
        "seed": SEED,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "raw": file_record(raw_path),
        "log": file_record(log_path),
        "exact_receiver_entrypoint_used": True,
        "complete": True,
    }
    atomic_json(receipt_path, result)
    return result


def load_segnet(device: Any) -> Any:
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import SegNet, segnet_sd_path
    finally:
        sys.path.pop(0)
    network = SegNet().eval().to(device=device)
    network.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    return network


def load_posenet(device: Any) -> Any:
    """Load the exact upstream PoseNet used by ``upstream/evaluate.py``."""
    from safetensors.torch import load_file

    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().to(device=device)
    network.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def _dataset(source: str, raw_root: Path | None, device: Any) -> Any:
    sys.path.insert(0, str(UPSTREAM))
    try:
        from frame_utils import AVVideoDataset, DaliVideoDataset, TensorVideoDataset
    finally:
        sys.path.pop(0)
    names = (UPSTREAM / "public_test_video_names.txt").read_text().splitlines()
    dataset_class = (
        (DaliVideoDataset if device.type == "cuda" else AVVideoDataset)
        if source == "gt"
        else TensorVideoDataset
    )
    data_dir = UPSTREAM / "videos" if source == "gt" else raw_root
    dataset = dataset_class(
        names,
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        device=device,
        num_threads=2,
        seed=SEED,
        prefetch_queue_depth=4,
    )
    dataset.prepare_data()
    return dataset


def scorer_axis(device: Any, scorer_name: str) -> str:
    """Label the actual scorer instrument instead of inheriting the T4 worker name."""
    batch = f"n600, batch={BATCH_SIZE}"
    if device.type == "cuda":
        return f"[contest-CUDA T4 frozen-{scorer_name} {batch}] COMPONENT-ONLY"
    if device.type == "cpu" and platform.system() == "Darwin":
        return f"[macOS-CPU advisory frozen-{scorer_name} {batch}] NON-PROMOTABLE"
    if device.type == "cpu":
        return f"[CPU advisory frozen-{scorer_name} {batch}] NON-PROMOTABLE"
    return f"[diagnostic-{device.type} frozen-{scorer_name} {batch}] NON-PROMOTABLE"


def _progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"completed_batches": 0, "completed_pairs": 0}
    value = json.loads(path.read_text())
    if int(value.get("completed_batches", -1)) < 0 or int(value.get("completed_pairs", -1)) < 0:
        raise WorkerError(f"invalid score progress: {path}")
    return value


def score_argmax_field(
    *,
    source: str,
    raw_root: Path | None,
    raw_record: dict[str, Any] | None,
    scorer: Any,
    device: Any,
    run_root: Path,
) -> dict[str, Any]:
    """Retain all scorer payloads and one complete n600 uint8 argmax plane."""
    import einops
    import torch

    scorer_root = run_root / f"retained/scorer/{source}"
    result_path = scorer_root / "SCORER_RESULT.json"
    field_path = run_root / f"retained/fields/{source}_argmax_n600.npy"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        require_record(field_path, result["argmax"])
        return result

    progress_path = scorer_root / "PROGRESS.json"
    progress = _progress(progress_path)
    completed_batches = int(progress["completed_batches"])
    completed_pairs = int(progress["completed_pairs"])
    partial_field = field_path.with_name(field_path.name + ".inprogress.npy")
    field_storage = field_path if field_path.is_file() else partial_field
    if field_storage.is_file():
        field = np.lib.format.open_memmap(field_storage, mode="r+")
        if field.shape != (N_PAIRS, SEG_HEIGHT, SEG_WIDTH) or field.dtype != np.uint8:
            raise WorkerError(f"partial argmax field shape/dtype differs: {field_storage}")
    else:
        if completed_batches or completed_pairs:
            raise WorkerError(f"score progress exists without argmax payload: {source}")
        partial_field.parent.mkdir(parents=True, exist_ok=True)
        field = np.lib.format.open_memmap(
            partial_field,
            mode="w+",
            dtype=np.uint8,
            shape=(N_PAIRS, SEG_HEIGHT, SEG_WIDTH),
        )

    dataset = _dataset(source, raw_root, device)
    cursor = 0
    batch_rows: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for ordinal, (_path, _index, batch) in enumerate(dataset):
            batch_size = int(batch.shape[0])
            batch_root = scorer_root / f"batches/batch_{ordinal:04d}"
            row_path = batch_root / "BATCH_RESULT.json"
            source_record: dict[str, Any]
            if source == "gt":
                gt_batch_path = (
                    run_root / f"retained/gt_rgb_batches/batch_{ordinal:04d}.uint8.npy"
                )
                batch_cpu = batch.cpu().numpy()
                if gt_batch_path.is_file():
                    retained_batch = np.load(gt_batch_path, mmap_mode="r", allow_pickle=False)
                    if not np.array_equal(retained_batch, batch_cpu):
                        raise WorkerError(f"resumed GT RGB batch differs: {gt_batch_path}")
                    source_record = file_record(gt_batch_path)
                else:
                    source_record = atomic_npy(gt_batch_path, batch_cpu)
            else:
                if raw_root is None or raw_record is None:
                    raise WorkerError(f"candidate source has no retained raw root: {source}")
                source_record = raw_record

            if ordinal < completed_batches:
                if not row_path.is_file():
                    raise WorkerError(f"resume batch receipt is missing: {row_path}")
                row = json.loads(row_path.read_text())
                if int(row["pair_start"]) != cursor or int(row["pair_end"]) != cursor + batch_size:
                    raise WorkerError(f"resume batch geometry differs: {row_path}")
                if row.get("source_payload") != source_record:
                    raise WorkerError(f"resume source payload differs: {row_path}")
                batch_rows.append(row)
                cursor += batch_size
                continue

            batch_device = batch.to(device)
            tensor = einops.rearrange(
                batch_device,
                "b t h w c -> b t c h w",
                b=batch_size,
                t=2,
                c=3,
            ).float()
            seg_input = scorer.preprocess_input(tensor)
            logits = scorer(seg_input)
            argmax = logits.argmax(dim=1).to(torch.uint8)
            seg_input_record = atomic_npy(
                batch_root / "seg_input.float32.npy", seg_input.cpu().numpy()
            )
            logits_record = atomic_npy(
                batch_root / "logits.float32.npy", logits.cpu().numpy()
            )
            argmax_cpu = argmax.cpu().numpy()
            field[cursor : cursor + batch_size] = argmax_cpu
            field.flush()
            row = {
                "schema": "ddm_js1b_scorer_batch.v1",
                "source": source,
                "ordinal": ordinal,
                "pair_start": cursor,
                "pair_end": cursor + batch_size,
                "source_payload": source_record,
                "seg_input": seg_input_record,
                "logits": logits_record,
                "argmax_storage": str(field_storage),
                "argmax_slice_bytes": int(argmax_cpu.nbytes),
                "argmax_slice_sha256": sha256_bytes(argmax_cpu.tobytes()),
                "complete": True,
            }
            atomic_json(row_path, row)
            batch_rows.append(row)
            cursor += batch_size
            completed_batches = ordinal + 1
            completed_pairs = cursor
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_js1b_scorer_progress.v1",
                    "source": source,
                    "completed_batches": completed_batches,
                    "completed_pairs": completed_pairs,
                    "argmax_inprogress": {
                        "path": str(field_storage),
                        "bytes": field_storage.stat().st_size,
                        "complete_sha256": None,
                    },
                },
            )
            del batch_device, tensor, seg_input, logits, argmax, argmax_cpu

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if cursor != N_PAIRS or completed_pairs != N_PAIRS:
        raise WorkerError(
            f"scorer census differs for {source}: cursor={cursor}, progress={completed_pairs}"
        )
    field.flush()
    del field
    if field_storage != field_path:
        os.replace(field_storage, field_path)
    batches_path = scorer_root / "BATCH_RESULTS.jsonl"
    atomic_bytes(batches_path, b"".join(canonical_json_bytes(row) for row in batch_rows))
    result = {
        "schema": "ddm_js1b_scorer_result.v1",
        "axis": scorer_axis(device, "SegNet argmax fields"),
        "source": source,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "pairs": N_PAIRS,
        "argmax": file_record(field_path),
        "retained_batch_receipts": file_record(batches_path),
        "retained_gt_rgb_batches": source == "gt",
        "retained_seg_inputs": True,
        "retained_logits": True,
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    atomic_json(result_path, result)
    return result


def score_pose_vectors(
    *,
    source: str,
    raw_root: Path | None,
    raw_record: dict[str, Any] | None,
    scorer: Any,
    device: Any,
    run_root: Path,
) -> dict[str, Any]:
    """Retain exact PoseNet inputs, full outputs, and the scored first six values.

    A distinct ``source`` names every repeat.  Recreating the upstream dataset
    for each call makes decoded-repeat noise an observed same-job quantity,
    while immutable per-batch receipts keep an interrupted pass resumable.
    """
    import einops
    import torch

    scorer_root = run_root / f"retained/pose/{source}"
    result_path = scorer_root / "POSE_RESULT.json"
    vector_path = run_root / f"retained/pose_vectors/{source}_first6_n600.npy"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        require_record(vector_path, result["first6_vectors"])
        return result

    progress_path = scorer_root / "PROGRESS.json"
    progress = _progress(progress_path)
    completed_batches = int(progress["completed_batches"])
    completed_pairs = int(progress["completed_pairs"])
    partial_vectors = vector_path.with_name(vector_path.name + ".inprogress.npy")
    vector_storage = vector_path if vector_path.is_file() else partial_vectors
    if vector_storage.is_file():
        vectors = np.lib.format.open_memmap(vector_storage, mode="r+")
        if vectors.shape != (N_PAIRS, 6) or vectors.dtype != np.float32:
            raise WorkerError(f"partial PoseNet vector shape/dtype differs: {vector_storage}")
    else:
        if completed_batches or completed_pairs:
            raise WorkerError(f"pose progress exists without vector payload: {source}")
        partial_vectors.parent.mkdir(parents=True, exist_ok=True)
        vectors = np.lib.format.open_memmap(
            partial_vectors,
            mode="w+",
            dtype=np.float32,
            shape=(N_PAIRS, 6),
        )

    dataset_source = "gt" if source == "gt" else "candidate"
    dataset = _dataset(dataset_source, raw_root, device)
    cursor = 0
    batch_rows: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for ordinal, (_path, _index, batch) in enumerate(dataset):
            batch_size = int(batch.shape[0])
            batch_root = scorer_root / f"batches/batch_{ordinal:04d}"
            row_path = batch_root / "BATCH_RESULT.json"
            if dataset_source == "gt":
                gt_batch_path = run_root / f"retained/gt_rgb_batches/batch_{ordinal:04d}.uint8.npy"
                batch_cpu = batch.cpu().numpy()
                if gt_batch_path.is_file():
                    retained_batch = np.load(gt_batch_path, mmap_mode="r", allow_pickle=False)
                    if not np.array_equal(retained_batch, batch_cpu):
                        raise WorkerError(f"resumed GT RGB batch differs: {gt_batch_path}")
                    source_record = file_record(gt_batch_path)
                else:
                    source_record = atomic_npy(gt_batch_path, batch_cpu)
            else:
                if raw_root is None or raw_record is None:
                    raise WorkerError(f"candidate source has no retained raw root: {source}")
                source_record = raw_record

            if ordinal < completed_batches:
                if not row_path.is_file():
                    raise WorkerError(f"resume PoseNet batch receipt is missing: {row_path}")
                row = json.loads(row_path.read_text())
                if int(row["pair_start"]) != cursor or int(row["pair_end"]) != cursor + batch_size:
                    raise WorkerError(f"resume PoseNet batch geometry differs: {row_path}")
                if row.get("source_payload") != source_record:
                    raise WorkerError(f"resume PoseNet source payload differs: {row_path}")
                for key in ("pose_input", "pose_output_full"):
                    require_record(Path(row[key]["path"]), row[key])
                batch_rows.append(row)
                cursor += batch_size
                continue

            batch_device = batch.to(device)
            tensor = einops.rearrange(
                batch_device,
                "b t h w c -> b t c h w",
                b=batch_size,
                t=2,
                c=3,
            ).float()
            pose_input = scorer.preprocess_input(tensor)
            pose_output = scorer(pose_input)["pose"]
            pose_input_record = atomic_npy(
                batch_root / "pose_input.float32.npy", pose_input.cpu().numpy()
            )
            pose_output_cpu = pose_output.cpu().numpy().astype(np.float32, copy=False)
            pose_output_record = atomic_npy(
                batch_root / "pose_output_full.float32.npy", pose_output_cpu
            )
            first6 = np.ascontiguousarray(pose_output_cpu[:, :6])
            vectors[cursor : cursor + batch_size] = first6
            vectors.flush()
            row = {
                "schema": "ddm_js1b_pose_batch.v1",
                "source": source,
                "ordinal": ordinal,
                "pair_start": cursor,
                "pair_end": cursor + batch_size,
                "source_payload": source_record,
                "pose_input": pose_input_record,
                "pose_output_full": pose_output_record,
                "first6_slice_bytes": int(first6.nbytes),
                "first6_slice_sha256": sha256_bytes(first6.tobytes()),
                "vector_storage": str(vector_storage),
                "complete": True,
            }
            atomic_json(row_path, row)
            batch_rows.append(row)
            cursor += batch_size
            completed_batches = ordinal + 1
            completed_pairs = cursor
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_js1b_pose_progress.v1",
                    "source": source,
                    "completed_batches": completed_batches,
                    "completed_pairs": completed_pairs,
                    "vectors_inprogress": {
                        "path": str(vector_storage),
                        "bytes": vector_storage.stat().st_size,
                        "complete_sha256": None,
                    },
                },
            )
            del batch_device, tensor, pose_input, pose_output, pose_output_cpu, first6

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if cursor != N_PAIRS or completed_pairs != N_PAIRS:
        raise WorkerError(
            f"PoseNet census differs for {source}: cursor={cursor}, progress={completed_pairs}"
        )
    vectors.flush()
    del vectors
    if vector_storage != vector_path:
        os.replace(vector_storage, vector_path)
    batches_path = scorer_root / "BATCH_RESULTS.jsonl"
    atomic_bytes(batches_path, b"".join(canonical_json_bytes(row) for row in batch_rows))
    result = {
        "schema": "ddm_js1b_pose_result.v1",
        "axis": scorer_axis(device, "PoseNet first6 vectors"),
        "source": source,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "pairs": N_PAIRS,
        "first6_vectors": file_record(vector_path),
        "retained_batch_receipts": file_record(batches_path),
        "retained_gt_rgb_batches": dataset_source == "gt",
        "retained_pose_inputs": True,
        "retained_full_pose_outputs": True,
        "elapsed_seconds": time.time() - started,
        "complete": True,
    }
    atomic_json(result_path, result)
    return result


def materialize_c1_target(run_root: Path) -> dict[str, Any]:
    bundle = run_root / "inputs/c1_target_argmax_n600.zip"
    expected = {"bytes": C1_TARGET_BYTES, "sha256": C1_TARGET_SHA256}
    target = run_root / "retained/fields/c1_target_argmax_n600.npy"
    if not target.is_file():
        with zipfile.ZipFile(bundle) as archive:
            if archive.namelist() != ["c1_target_argmax_n600.npy"]:
                raise WorkerError("C1 target transport ZIP members differ")
            atomic_bytes(target, archive.read("c1_target_argmax_n600.npy"))
    require_record(target, expected)
    value = np.load(target, mmap_mode="r", allow_pickle=False)
    if value.shape != (N_PAIRS, SEG_HEIGHT, SEG_WIDTH) or value.dtype != np.uint8:
        raise WorkerError("C1 target argmax shape/dtype differs")
    return file_record(target)


def adjudicate_fields(run_root: Path) -> dict[str, Any]:
    fields = {
        name: run_root / f"retained/fields/{name}_argmax_n600.npy"
        for name in ("gt", "cp135_base", "t1r1_c1_composed", "c1_target")
    }
    values = {
        name: np.load(path, mmap_mode="r", allow_pickle=False)
        for name, path in fields.items()
    }
    gt = values["gt"]
    flips = {
        name: int(np.count_nonzero(value != gt))
        for name, value in values.items()
        if name != "gt"
    }
    cp_match = flips["cp135_base"] == EXPECTED_CP135_FLIPS
    c1_match = flips["c1_target"] == EXPECTED_C1_TARGET_FLIPS
    admitted = cp_match and c1_match
    return {
        "status": "ADMITTED" if admitted else "BLOCKED_AXIS_MISMATCH",
        "admitted_for_js1_stage0": admitted,
        "flips_vs_gt": flips,
        "cp135_control": {
            "expected_flips": EXPECTED_CP135_FLIPS,
            "observed_flips": flips["cp135_base"],
            "matches": cp_match,
        },
        "c1_target_control": {
            "expected_flips": EXPECTED_C1_TARGET_FLIPS,
            "observed_flips": flips["c1_target"],
            "matches": c1_match,
        },
        "disposition": (
            "consume fields in JS1 Stage 0 per-edge decomposition"
            if admitted
            else "stop; treat the discrepancy as a CUDA field or custody question"
        ),
        "fields": {name: file_record(path) for name, path in fields.items()},
    }


def run(run_root: Path, resume_from: str) -> dict[str, Any]:
    import timm
    import torch
    import torchvision

    from tac.contest_compliance import compute_upstream_snapshot_sha256

    started = time.time()
    request = json.loads((run_root / "inputs/REQUEST.json").read_text())
    if resume_from != str(request["resume_from"]) or resume_from != str(request["run_id"]):
        raise WorkerError("resume token differs from the retained run identity")
    final_path = run_root / "FINAL_RESULT.json"
    if final_path.is_file():
        result = json.loads(final_path.read_text())
        if result.get("execution_status") != "COMPLETE":
            raise WorkerError("retained final result is not complete")
        return result

    storage_preflight(run_root)
    input_records = request["inputs"]
    for filename, record in input_records.items():
        require_record(run_root / f"inputs/{filename}", record)
    checkpoint_once(
        run_root / "checkpoints/stage_00_inputs_and_storage.json",
        {
            "schema": "ddm_js1b_stage_checkpoint.v1",
            "stage": "inputs_and_storage",
            "inputs": input_records,
            "storage_preflight_passed": True,
            "expected_total_retained_payload_bytes": EXPECTED_RETAINED_PAYLOAD_BYTES,
            "storage_reserve_bytes": STORAGE_RESERVE_BYTES,
            "resume_from": resume_from,
            "complete": True,
        },
    )

    work_root = run_root / "work"
    runtimes = {}
    for name in ("cp135_base", "t1r1_c1_composed"):
        runtime_root = work_root / f"runtime/{name}"
        extract_zip_once(
            run_root / f"inputs/{name}_runtime.zip",
            runtime_root,
            "RUNTIME_EXTRACTED.json",
        )
        runtimes[name] = runtime_root
    checkpoint_once(
        run_root / "checkpoints/stage_05_runtimes_extracted.json",
        {
            "schema": "ddm_js1b_stage_checkpoint.v1",
            "stage": "runtimes_extracted",
            "runtime_roots": {name: str(path) for name, path in runtimes.items()},
            "complete": True,
        },
    )

    archives = {
        "cp135_base": {
            "path": run_root / "inputs/cp135_base_archive.zip",
            "record": {"bytes": CP135_BYTES, "sha256": CP135_SHA256},
        },
        "t1r1_c1_composed": {
            "path": run_root / "inputs/t1r1_c1_composed_archive.zip",
            "record": {"bytes": T1R1_BYTES, "sha256": T1R1_SHA256},
        },
    }
    receivers = {}
    for ordinal, name in enumerate(("cp135_base", "t1r1_c1_composed"), start=10):
        receivers[name] = decode_exact_receiver(
            name=name,
            archive_path=archives[name]["path"],
            archive_record=archives[name]["record"],
            runtime_root=runtimes[name],
            run_root=run_root,
        )
        checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal:02d}_{name}_decoded.json",
            {
                "schema": "ddm_js1b_stage_checkpoint.v1",
                "stage": f"{name}_decoded",
                "receiver": receivers[name],
                "complete": True,
            },
        )

    c1_target = materialize_c1_target(run_root)
    upstream_snapshot_sha256 = compute_upstream_snapshot_sha256(
        UPSTREAM,
        upstream_subdir=".",
        reject_executable_artifacts=True,
    )
    if not upstream_snapshot_sha256:
        raise WorkerError("remote canonical upstream snapshot is missing")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    scorer = load_segnet(device)
    scorer_results = {}
    score_sources = (
        ("gt", None, None),
        (
            "cp135_base",
            Path(receivers["cp135_base"]["raw"]["path"]).parent,
            receivers["cp135_base"]["raw"],
        ),
        (
            "t1r1_c1_composed",
            Path(receivers["t1r1_c1_composed"]["raw"]["path"]).parent,
            receivers["t1r1_c1_composed"]["raw"],
        ),
    )
    for ordinal, (name, raw_root, raw_record) in enumerate(score_sources, start=20):
        scorer_results[name] = score_argmax_field(
            source=name,
            raw_root=raw_root,
            raw_record=raw_record,
            scorer=scorer,
            device=device,
            run_root=run_root,
        )
        checkpoint_once(
            run_root / f"checkpoints/stage_{ordinal:02d}_{name}_scored.json",
            {
                "schema": "ddm_js1b_stage_checkpoint.v1",
                "stage": f"{name}_scored",
                "scorer": scorer_results[name],
                "complete": True,
            },
        )

    adjudication = adjudicate_fields(run_root)
    final = {
        "schema": "ddm_js1b_cuda_argmax_field_materializer_result.v1",
        "execution_status": "COMPLETE",
        "status": adjudication["status"],
        "axis": AXIS,
        "selection_mode": "full population, no sampling, 600 non-overlapping pairs",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "receivers": receivers,
        "scorers": scorer_results,
        "c1_target_argmax": c1_target,
        "axis_adjudication": adjudication,
        "fields": adjudication["fields"],
        "retention": {
            "volume_run_root": str(run_root),
            "both_exact_receiver_raw_videos": True,
            "all_materialized_gt_rgb_batches": True,
            "all_materialized_seg_inputs": True,
            "all_materialized_logits": True,
            "gt_and_both_candidate_argmax_fields": True,
            "c1_target_control_field": True,
            "expected_payload_bytes_before_metadata": EXPECTED_RETAINED_PAYLOAD_BYTES,
        },
        "provenance": {
            "source_git_head": request["source_git_head"],
            "source_git_dirty_at_dispatch": request["source_git_dirty"],
            "source_git_status_sha256": request["source_git_status_sha256"],
            "dispatcher_source_sha256": request["dispatcher_source_sha256"],
            "worker_source_sha256": request["worker_source_sha256"],
            "upstream_snapshot_sha256": upstream_snapshot_sha256,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "timm_version": timm.__version__,
            "numpy_version": np.__version__,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "elapsed_seconds": time.time() - started,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "boundaries": {
            "measured": (
                "exact adapted receiver raw outputs and frozen SegNet argmax fields on "
                "contest-CUDA T4 at batch 16"
            ),
            "not_measured": (
                "PoseNet, archive score, contest-CPU, full public evaluator, or any new archive"
            ),
        },
    }
    checkpoint_once(final_path, final)
    checkpoint_once(run_root / "checkpoints/stage_30_final.json", final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", required=True)
    args = parser.parse_args(argv)
    result = run(args.run_root.resolve(), args.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
