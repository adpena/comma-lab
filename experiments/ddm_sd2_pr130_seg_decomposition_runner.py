#!/usr/bin/env python3
"""Retain and decompose paired PR130 scorer evidence without scalar loss.

The runner materializes the pinned public receiver, decodes the PR130 control
and one candidate through that receiver, and runs the frozen upstream scorers
in chunks.  Every decoded frame, SegNet argmax, full SegNet logit field, and
full PoseNet output remains reachable from a bytes-plus-SHA-256 manifest.
Decomposition is performed only after re-opening the retained argmax chunks.

This is an advisory measurement runner, not a contest score authority.  The
``--plan-only`` path performs the storage, writer, receiver, and retention-gate
preflight without loading or running a scorer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
TIER1_ROOT = Path("/Volumes/VertigoDataTier/pact")
APDATA_VOLUME = Path("/Volumes/APDataStore")
BULK_ROOT = APDATA_VOLUME / "pact"
DEFAULT_OUT = BULK_ROOT / "ddm_sd2_20260810/matched_local_n600"
DEFAULT_QUEUE = REPO_ROOT / ".omx/research/ddm_sg2_20260810/SG2_SCORER_QUEUE.json"
DEFAULT_BASE_ARCHIVE = TIER1_ROOT / "ddm_pr130_reproduce_20260809/reproduction/archive.zip"
DEFAULT_CANDIDATE_ARCHIVE = TIER1_ROOT / "ddm_sd1_semantic_20260809/cpu_screen/archives/selected_mixed_n600.zip"
DEFAULT_CHALLENGE_ROOT = REPO_ROOT / "upstream"

BASE_ID = "pr130_q4_control"
CANDIDATE_ID = "sd1_selected_mixed_q3q4_n600"
BASE_ARCHIVE_BYTES = 191_052
BASE_ARCHIVE_SHA256 = "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
CANDIDATE_ARCHIVE_BYTES = 190_204
CANDIDATE_ARCHIVE_SHA256 = "010a8a5273ae87595191ffc03447fa36e61978ae9f827c2def46dea7075dfa67"
RUNTIME_COMMIT = "58f62cd22ff07562c0534c999d705fb9edfe5279"
RUNTIME_SOURCE_ROOT = "src/tac/pr130_runtime/dv1_cpu_runtime"
RUNTIME_FILE_SHA256 = {
    "carrier_codec.py": "d2f14402374b4e622b7f981d736389fb04f0ca0165180e4c75f3a32ffe996bed",
    "hpac_integer.py": "6e6b4f4d0b293fb60cc1b751958756a4cd6c2ce7bcff68c6f03e20277856803f",
    "hpac_integer_sparse.py": "2240ee32c53fe949b560d316d349e0bbdccc0ceb78787307cd4d530623d42a0c",
    "inflate.py": "9a42628e6306ddaa4682c915db31196ffdace8fa502c6322a4586e3c4a7562a2",
    "inflate.sh": "bc92880ef9c038c6adfe4968a4b6206b8e565501e839634e1d6762a704421915",
    "integer_model_io.py": "6f91c91ed4785d203aa3570af362fbe9c6a64bb2249599f8554adb31174b80a5",
    "receiver.py": "6239649cc81e9c5a86273502be0beff19805720854b980f167bb71a0a80c3a42",
    "runtime-dependencies.json": "55397f16f270472e0f0bde1e69d2c5c5a2e015f2cc051a31e19ce2dbfc8cfe07",
}

PAIR_COUNT = 600
SEQ_LEN = 2
CAMERA_H = 874
CAMERA_W = 1164
EVAL_H = 384
EVAL_W = 512
NUM_CLASSES = 5
POSE_OUTPUTS = 12
SCORED_POSE_OUTPUTS = 6
ORIGINAL_VIDEO_BYTES = 37_545_489
CANONICAL_CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

REQUIRED_WRITER_PLAN = {
    "archive.zip": "retain_archive",
    "archive.repeat.zip": "retain_archive_repeat",
    "decoded camera-frame chunks with bytes and SHA-256": "retain_camera_chunks",
    "SegNet argmax chunks with bytes and SHA-256": "retain_argmax_chunks",
    "PoseNet output chunks with bytes and SHA-256": "retain_pose_chunks",
    "directed target-to-prediction edge matrix": "write_decomposition",
    "atomic progress and final receipt": "write_atomic_receipts",
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file_range(path: Path, offset: int, length: int) -> str:
    if offset < 0 or length < 0 or offset + length > path.stat().st_size:
        raise ValueError(f"invalid retained range for {path}: {offset}+{length}")
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as handle:
        handle.seek(offset)
        while remaining:
            block = handle.read(min(8 << 20, remaining))
            if not block:
                raise EOFError(f"short retained range read from {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def range_artifact(path: Path, offset: int, length: int) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "offset_bytes": offset,
        "bytes": length,
        "sha256": sha256_file_range(path, offset, length),
        "storage": "byte range of the retained raw receiver output",
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists():
        if sha256_file(path) != sha256_bytes(payload):
            raise ValueError(f"immutable retained payload differs: {path}")
    else:
        atomic_write_bytes(path, payload)
    return artifact(path)


def atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    source_record = artifact(source)
    if destination.exists():
        destination_record = artifact(destination)
        if destination_record["sha256"] != source_record["sha256"]:
            raise ValueError(f"retained copy differs: {destination}")
        return destination_record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    copied = artifact(destination)
    if copied["sha256"] != source_record["sha256"]:
        raise ValueError(f"copy hash differs: {destination}")
    return copied


def require_bulk_store(path: Path) -> Path:
    if not APDATA_VOLUME.is_mount():
        raise RuntimeError(f"charter-mandated bulk volume is not mounted: {APDATA_VOLUME}")
    resolved = path.resolve()
    try:
        resolved.relative_to(BULK_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"bulk evidence must remain below {BULK_ROOT}: {resolved}") from error
    return resolved


def require_pinned_file(path: Path, byte_count: int, sha256: str) -> dict[str, Any]:
    record = artifact(path)
    if record["bytes"] != byte_count or record["sha256"] != sha256:
        raise ValueError(f"source pin differs for {path}: bytes={record['bytes']} sha256={record['sha256']}")
    return record


def validate_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    retention = queue.get("required_retention")
    if not isinstance(retention, Mapping):
        raise ValueError("SG2 queue lacks required_retention")
    declared = set(retention.get("per_candidate", ()))
    missing = set(REQUIRED_WRITER_PLAN) - declared
    if missing:
        raise ValueError(f"SG2 retention declarations missing: {sorted(missing)}")
    if int(retention.get("chunk_pair_limit", 0)) != 120:
        raise ValueError("SG2 queue chunk_pair_limit must remain 120")
    if retention.get("resume_required") is not True:
        raise ValueError("SG2 queue no longer requires resume")
    if retention.get("do_not_launch_if_any_payload_writer_is_absent") is not True:
        raise ValueError("SG2 queue no longer fails closed on missing writers")
    base = queue.get("baseline", {})
    candidate = queue.get("candidate", {})
    if (
        int(base.get("archive_bytes", -1)) != BASE_ARCHIVE_BYTES
        or base.get("archive_sha256") != BASE_ARCHIVE_SHA256
        or int(candidate.get("archive_bytes", -1)) != CANDIDATE_ARCHIVE_BYTES
        or candidate.get("archive_sha256") != CANDIDATE_ARCHIVE_SHA256
        or candidate.get("receiver_commit") != RUNTIME_COMMIT
    ):
        raise ValueError("SG2 queue source or receiver pins differ")
    return dict(retention)


def storage_projection(pair_count: int, chunk_pairs: int) -> dict[str, int]:
    receiver_raw = PAIR_COUNT * SEQ_LEN * CAMERA_H * CAMERA_W * 3
    target_camera = pair_count * SEQ_LEN * CAMERA_H * CAMERA_W * 3
    logits = 3 * pair_count * NUM_CLASSES * EVAL_H * EVAL_W * 4
    argmax = 3 * pair_count * EVAL_H * EVAL_W
    pose = 3 * pair_count * POSE_OUTPUTS * 4
    token_checkpoints_upper = 2 * (PAIR_COUNT * EVAL_H * EVAL_W + 16_000_000)
    runtime_dependencies_and_receipts = 512_000_000
    archive_copies = 2 * (BASE_ARCHIVE_BYTES + CANDIDATE_ARCHIVE_BYTES)
    projected_final = (
        2 * receiver_raw
        + target_camera
        + logits
        + argmax
        + pose
        + token_checkpoints_upper
        + runtime_dependencies_and_receipts
        + archive_copies
    )
    active_chunk = (
        chunk_pairs * SEQ_LEN * CAMERA_H * CAMERA_W * 3
        + 3 * chunk_pairs * NUM_CLASSES * EVAL_H * EVAL_W * 4
        + 3 * chunk_pairs * EVAL_H * EVAL_W
        + 3 * chunk_pairs * POSE_OUTPUTS * 4
    )
    return {
        "base_receiver_raw_bytes": receiver_raw,
        "candidate_receiver_raw_bytes": receiver_raw,
        "target_decoded_camera_bytes": target_camera,
        "retained_segnet_logits_bytes": logits,
        "retained_segnet_argmax_bytes": argmax,
        "retained_posenet_outputs_bytes": pose,
        "token_checkpoint_upper_bytes": token_checkpoints_upper,
        "runtime_dependencies_and_receipts_allowance_bytes": (runtime_dependencies_and_receipts),
        "archive_copy_bytes": archive_copies,
        "projected_final_bytes": projected_final,
        "active_chunk_bytes_already_in_final_projection": active_chunk,
        "failed_decode_attempt_contingency_bytes": receiver_raw,
    }


def retained_projection_credit(out_dir: Path, projected_final: int) -> int:
    """Count only paths that satisfy a future final-payload obligation.

    Failed decode attempts deliberately receive no credit: their bytes consume
    free space but do not reduce the remaining final footprint.
    """

    candidates: list[Path] = []
    for relative in (
        "retained/candidates",
        "retained/runtime",
        "preflight",
    ):
        root = out_dir / relative
        if root.exists():
            candidates.extend(path for path in root.rglob("*") if path.is_file())
    for candidate_id in (BASE_ID, CANDIDATE_ID):
        decode_root = out_dir / f"retained/decode/{candidate_id}"
        raw = decode_root / "0.raw"
        if raw.is_file():
            candidates.append(raw)
        token_root = decode_root / "token_cache"
        if token_root.exists():
            candidates.extend(path for path in token_root.rglob("*") if path.is_file())
    credited = sum(path.stat().st_size for path in set(candidates))
    return min(credited, projected_final)


def materialize_runtime(runtime_dir: Path) -> list[dict[str, Any]]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "cat-file", "-e", f"{RUNTIME_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    records = []
    for name, expected_sha in RUNTIME_FILE_SHA256.items():
        git_path = f"{RUNTIME_COMMIT}:{RUNTIME_SOURCE_ROOT}/{name}"
        completed = subprocess.run(
            ["git", "show", git_path],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        payload = completed.stdout
        if sha256_bytes(payload) != expected_sha:
            raise ValueError(f"Git object hash differs for {git_path}")
        record = retain_bytes(runtime_dir / name, payload)
        record["source"] = git_path
        records.append(record)
    (runtime_dir / "inflate.sh").chmod(0o755)
    (runtime_dir / "inflate.py").chmod(0o755)
    return records


def retain_archive(*, source: Path, destination_dir: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    source_record = require_pinned_file(source, expected_bytes, expected_sha256)
    archive_record = atomic_copy(source, destination_dir / "archive.zip")
    repeat_record = atomic_copy(source, destination_dir / "archive.repeat.zip")
    if archive_record["sha256"] != repeat_record["sha256"]:
        raise ValueError("archive repeat differs")
    with zipfile.ZipFile(destination_dir / "archive.zip") as handle:
        names = handle.namelist()
        if names != ["p"]:
            raise ValueError(f"archive member census differs: {names}")
        payload = handle.read("p")
    member_record = retain_bytes(destination_dir / "decode_input/p", payload)
    return {
        "source": source_record,
        "archive": archive_record,
        "archive_repeat": repeat_record,
        "archive_repeat_byte_identical": True,
        "member_p": member_record,
    }


def payload_retention_preflight(
    runner_path: Path, runtime_records: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from tac.payload_retention_gate import scan_paths

    paths = [runner_path]
    test_path = REPO_ROOT / "experiments/tests/test_ddm_sd2_pr130_seg_decomposition_runner.py"
    if test_path.is_file():
        paths.append(test_path)
    paths.extend(Path(record["path"]) for record in runtime_records if str(record["path"]).endswith(".py"))
    findings = scan_paths(paths)
    receipt = {
        "schema": "ddm_sd2.payload_retention_preflight.v1",
        "checked_at_utc": utc_now(),
        "files_scanned": len(paths),
        "paths": [str(path.resolve()) for path in paths],
        "findings": [finding.render() for finding in findings],
        "status": "PASS" if not findings else "REFUSED",
        "writer_reachability_note": (
            "The static gate is paired with writer-plan validation and live retained "
            "byte/numpy/range probes; hash-only reachability is not accepted."
        ),
    }
    atomic_write_json(output, receipt)
    if findings:
        raise RuntimeError("payload retention gate refused the runner/runtime")
    return {**receipt, "receipt": artifact(output)}


def writer_preflight(out_dir: Path, retention: Mapping[str, Any]) -> dict[str, Any]:
    declared = set(retention["per_candidate"])
    missing = set(REQUIRED_WRITER_PLAN) - declared
    writer_bindings: dict[str, tuple[Any, ...]] = {
        "retain_archive": (retain_archive, retain_bytes, atomic_copy),
        "retain_archive_repeat": (atomic_copy,),
        "retain_camera_chunks": (
            open_chunk_arrays,
            flush_chunk_arrays,
            finalize_chunk_arrays,
            range_artifact,
        ),
        "retain_argmax_chunks": (
            open_chunk_arrays,
            flush_chunk_arrays,
            finalize_chunk_arrays,
        ),
        "retain_pose_chunks": (
            open_chunk_arrays,
            flush_chunk_arrays,
            finalize_chunk_arrays,
        ),
        "write_decomposition": (atomic_write_json,),
        "write_atomic_receipts": (atomic_write_json,),
    }
    absent = [
        name
        for name, writer_name in REQUIRED_WRITER_PLAN.items()
        if writer_name not in writer_bindings or not all(callable(writer) for writer in writer_bindings[writer_name])
    ]
    if missing or absent:
        raise RuntimeError(f"payload writer preflight refused: missing={sorted(missing)} absent={absent}")
    probe_dir = out_dir / "preflight/writer_probe"
    byte_probe = retain_bytes(probe_dir / "payload_writer.bin", b"ddm_sd2_payload_writer_reachable_v1\n")
    probe_chunk_dir = probe_dir / "chunk_writer"
    probe_specs = {
        "probe/payload": (np.dtype(np.uint8), (2, 2)),
    }
    probe_arrays = open_chunk_arrays(probe_chunk_dir, probe_specs)
    probe_arrays["probe/payload"][:] = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    flush_chunk_arrays(probe_arrays)
    del probe_arrays
    numpy_probe = finalize_chunk_arrays(probe_chunk_dir, probe_specs)["probe/payload"]
    range_probe = range_artifact(Path(byte_probe["path"]), 4, 7)
    receipt = {
        "schema": "ddm_sd2.writer_preflight.v1",
        "checked_at_utc": utc_now(),
        "required_writer_plan": REQUIRED_WRITER_PLAN,
        "callable_writer_bindings": {
            name: [writer.__name__ for writer in writers] for name, writers in writer_bindings.items()
        },
        "declared_retention": sorted(declared),
        "byte_writer_probe": byte_probe,
        "numpy_writer_probe": numpy_probe,
        "range_writer_probe": range_probe,
        "status": "PASS",
    }
    receipt_path = probe_dir / "WRITER_PREFLIGHT.json"
    atomic_write_json(receipt_path, receipt)
    return {**receipt, "receipt": artifact(receipt_path)}


def axis_label(device: torch.device) -> str:
    host = platform.system().lower()
    if device.type == "mps":
        return "[macOS-MPS diagnostic; never score authority]"
    if device.type == "cpu" and host == "darwin":
        return "[macOS-CPU advisory]"
    if device.type == "cpu":
        return "[Linux-CPU advisory; not contest authority]"
    return "[CUDA local research axis; not contest authority without 1:1 custody]"


def environment_provenance() -> dict[str, Any]:
    package_versions = {}
    for name in ("av", "numpy", "safetensors", "torch"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head_at_fire": head,
        "runner": artifact(Path(__file__).resolve()),
        "python": sys.version,
        "executable": str(Path(sys.executable).absolute()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package_versions": package_versions,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
    }


def fire_command(args: argparse.Namespace) -> str:
    command = [
        str(Path(sys.executable).absolute()),
        str(Path(__file__).resolve()),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--resume-from",
        str(args.resume_from.resolve()),
        "--queue",
        str(args.queue.resolve()),
        "--base-archive",
        str(args.base_archive.resolve()),
        "--candidate-archive",
        str(args.candidate_archive.resolve()),
        "--challenge-root",
        str(args.challenge_root.resolve()),
        "--video-names-file",
        str(args.video_names_file.resolve()),
        "--uncompressed-dir",
        str(args.uncompressed_dir.resolve()),
        "--device",
        args.device,
        "--decode-device",
        args.decode_device,
        "--batch-size",
        str(args.batch_size),
        "--chunk-pairs",
        str(args.chunk_pairs),
        "--pair-count",
        str(args.pair_count),
        "--seed",
        str(args.seed),
        "--cpu-threads",
        str(args.cpu_threads),
        "--num-threads",
        str(args.num_threads),
        "--prefetch-queue-depth",
        str(args.prefetch_queue_depth),
        "--minimum-free-bytes",
        str(args.minimum_free_bytes),
    ]
    return shlex.join(command)


def decoder_attempts_root(out_dir: Path, candidate_id: str) -> Path:
    return out_dir / f"retained/decode/{candidate_id}/attempts"


def next_attempt_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in root.glob("attempt_*"):
        try:
            numbers.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    return root / f"attempt_{max(numbers, default=0) + 1:04d}"


def inspect_retained_token_codec(
    runtime_dir: Path,
    member_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse the retained payload with the exact materialized receiver.

    The caller must not infer checkpoint capability from the candidate name.
    The codec discriminator belongs to the retained archive bytes and the
    pinned receiver is the authority that interprets it.
    """

    member_path = Path(str(member_record["path"]))
    if artifact(member_path) != dict(member_record):
        raise ValueError(f"retained archive member differs before codec inspection: {member_path}")
    receiver_path = runtime_dir / "receiver.py"
    receiver_record = artifact(receiver_path)
    if receiver_record["sha256"] != RUNTIME_FILE_SHA256["receiver.py"]:
        raise ValueError("materialized receiver differs before codec inspection")
    module_name = f"ddm_sd2_pinned_receiver_{receiver_record['sha256'][:12]}"
    receiver = sys.modules.get(module_name)
    if receiver is None:
        spec = importlib.util.spec_from_file_location(module_name, receiver_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned receiver from {receiver_path}")
        receiver = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = receiver
        try:
            spec.loader.exec_module(receiver)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    parts = receiver.split_payload(member_path.read_bytes())
    if parts.token_codec not in ("range", "ans"):
        raise ValueError(f"pinned receiver returned an unsupported token codec: {parts.token_codec!r}")
    return {
        "token_codec": parts.token_codec,
        "model_codec": parts.model_codec,
        "archive_member": dict(member_record),
        "receiver": receiver_record,
        "inspection": "pinned receiver.split_payload on the retained archive member",
    }


def token_checkpoint_policy(
    *,
    decode_root: Path,
    codec_inspection: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind checkpoint requests to the actual token coder's capabilities."""

    token_codec = str(codec_inspection["token_codec"])
    cache_path = decode_root / "token_cache/tokens.npz"
    receipt_path = decode_root / "token_cache/TOKEN_RECEIPT.json"
    progress_path = cache_path.with_name("tokens.progress.npz")
    if token_codec == "ans":
        return {
            "token_codec": token_codec,
            "intra_decode_checkpointing": "ENABLED_ANS_STACK",
            "token_cache": str(cache_path.resolve()),
            "token_receipt": str(receipt_path.resolve()),
            "token_progress": str(progress_path.resolve()),
            "environment": {
                "PR130_TOKEN_CACHE": str(cache_path.resolve()),
                "PR130_TOKEN_RECEIPT": str(receipt_path.resolve()),
            },
            "resume_boundary": (
                "ANS preserves periodic token progress and the completed token stage; "
                "the retained 60-pair scorer chunks remain the outer resume boundary."
            ),
        }
    if token_codec != "range":
        raise ValueError(f"unsupported checkpoint policy token codec: {token_codec!r}")
    stale = [path for path in (cache_path, receipt_path, progress_path) if path.exists()]
    if stale:
        raise ValueError(
            "Range archive has token-checkpoint artifacts that cannot be silently ignored: "
            + ", ".join(str(path) for path in stale)
        )
    return {
        "token_codec": token_codec,
        "intra_decode_checkpointing": "DISABLED_RANGE_SEQUENTIAL_REPLAY",
        "token_cache": None,
        "token_receipt": None,
        "token_progress": None,
        "environment": {},
        "resume_boundary": (
            "Range has no compatible periodic token checkpoint in the pinned receiver. "
            "A failed receiver attempt is retained and the next attempt replays the full "
            "Range decode; the retained 60-pair scorer chunks remain resumable."
        ),
    }


def promote_successful_attempt(
    attempt: Path,
    final_raw: Path,
    expected_raw_bytes: int,
    *,
    candidate_id: str,
    archive_sha256: str,
    decode_device: str,
) -> dict[str, Any] | None:
    success_path = attempt / "subprocess_success.json"
    command_path = attempt / "command.json"
    attempt_raw = attempt / "inflated/0.raw"
    if not success_path.is_file() or not command_path.is_file() or not attempt_raw.is_file():
        return None
    success = json.loads(success_path.read_text())
    command = json.loads(command_path.read_text())
    if int(success.get("returncode", -1)) != 0:
        raise ValueError(f"successful decode receipt has nonzero return code: {success_path}")
    if (
        command.get("candidate_id") != candidate_id
        or command.get("receiver_commit") != RUNTIME_COMMIT
        or command.get("decode_device") != decode_device
        or command.get("archive", {}).get("sha256") != archive_sha256
    ):
        raise ValueError(f"decode attempt binding differs: {command_path}")
    if attempt_raw.stat().st_size != expected_raw_bytes:
        raise ValueError(f"successful decode has wrong raw size: {attempt_raw}")
    raw_record = artifact(attempt_raw)
    promotion = {
        "schema": "ddm_sd2.decode_promotion.v1",
        "recorded_before_lossless_move": True,
        "source": raw_record,
        "destination": str(final_raw.resolve()),
        "reason": "completed real receiver output promoted into retained custody",
    }
    atomic_write_json(attempt / "promotion.json", promotion)
    final_raw.parent.mkdir(parents=True, exist_ok=True)
    if final_raw.exists():
        if artifact(final_raw)["sha256"] != raw_record["sha256"]:
            raise ValueError(f"existing final decode differs: {final_raw}")
    else:
        os.replace(attempt_raw, final_raw)
    return artifact(final_raw)


def ensure_real_decode(
    *,
    out_dir: Path,
    candidate_id: str,
    archive_record: Mapping[str, Any],
    runtime_dir: Path,
    video_names_file: Path,
    decode_device: str,
) -> dict[str, Any]:
    expected_raw_bytes = PAIR_COUNT * SEQ_LEN * CAMERA_H * CAMERA_W * 3
    decode_root = out_dir / f"retained/decode/{candidate_id}"
    final_raw = decode_root / "0.raw"
    final_receipt = decode_root / "DECODE_RECEIPT.json"
    if final_receipt.is_file():
        receipt = json.loads(final_receipt.read_text())
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("receiver_commit") != RUNTIME_COMMIT
            or receipt.get("decode_device") != decode_device
            or receipt.get("archive", {}).get("archive", {}).get("sha256") != archive_record["archive"]["sha256"]
        ):
            raise ValueError(f"retained decode receipt binding differs for {candidate_id}")
        current = artifact(final_raw)
        if current != receipt["raw"]:
            raise ValueError(f"retained decode receipt differs for {candidate_id}")
        return receipt

    attempts_root = decoder_attempts_root(out_dir, candidate_id)
    for attempt in sorted(attempts_root.glob("attempt_*"), reverse=True):
        promoted = promote_successful_attempt(
            attempt,
            final_raw,
            expected_raw_bytes,
            candidate_id=candidate_id,
            archive_sha256=str(archive_record["archive"]["sha256"]),
            decode_device=decode_device,
        )
        if promoted is not None:
            receipt = {
                "schema": "ddm_sd2.real_decode.v1",
                "candidate_id": candidate_id,
                "completed_at_utc": utc_now(),
                "receiver_commit": RUNTIME_COMMIT,
                "receiver_source": RUNTIME_SOURCE_ROOT,
                "archive": archive_record,
                "raw": promoted,
                "attempt": str(attempt.resolve()),
                "decode_device": decode_device,
                "resumed_promotion": True,
            }
            atomic_write_json(final_receipt, receipt)
            return receipt

    codec_inspection = inspect_retained_token_codec(runtime_dir, archive_record["member_p"])
    checkpoint_policy = token_checkpoint_policy(
        decode_root=decode_root,
        codec_inspection=codec_inspection,
    )
    attempt = next_attempt_dir(attempts_root)
    attempt.mkdir(parents=True)
    command = [
        "bash",
        str((runtime_dir / "inflate.sh").resolve()),
        str(Path(archive_record["member_p"]["path"]).parent),
        str((attempt / "inflated").resolve()),
        str(video_names_file.resolve()),
    ]
    command_receipt = {
        "schema": "ddm_sd2.decode_attempt.v1",
        "candidate_id": candidate_id,
        "created_at_utc": utc_now(),
        "command": command,
        "receiver_commit": RUNTIME_COMMIT,
        "decode_device": decode_device,
        "archive": archive_record["archive"],
        "codec_inspection": codec_inspection,
        "token_checkpoint_policy": {
            key: value for key, value in checkpoint_policy.items() if key != "environment"
        },
        "resumability": checkpoint_policy["resume_boundary"],
    }
    atomic_write_json(attempt / "command.json", command_receipt)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(Path(sys.executable).absolute()),
            "PR130_INFLATE_DEVICE": decode_device,
            "PR130_RUNTIME_DEPS_DIR": str((out_dir / "retained/runtime/dependencies").resolve()),
        }
    )
    env.update(checkpoint_policy["environment"])
    if checkpoint_policy["token_cache"] is not None:
        Path(str(checkpoint_policy["token_cache"])).parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log_path = attempt / "decode.log"
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        failure = {
            "schema": "ddm_sd2.decode_failure.v1",
            "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "log": artifact(log_path),
            "token_codec": codec_inspection["token_codec"],
            "checkpoint_policy": checkpoint_policy["intra_decode_checkpointing"],
            "disposition": (
                "RETAINED_FAILED_ATTEMPT; retry resumes the ANS token stage"
                if codec_inspection["token_codec"] == "ans"
                else "RETAINED_FAILED_ATTEMPT; retry replays sequential Range decode"
            ),
        }
        atomic_write_json(attempt / "failure.json", failure)
        raise RuntimeError(f"real receiver failed for {candidate_id}: {failure}")
    success = {
        "schema": "ddm_sd2.decode_subprocess_success.v1",
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "log": artifact(log_path),
        "token_codec": codec_inspection["token_codec"],
        "checkpoint_policy": checkpoint_policy["intra_decode_checkpointing"],
    }
    atomic_write_json(attempt / "subprocess_success.json", success)
    promoted = promote_successful_attempt(
        attempt,
        final_raw,
        expected_raw_bytes,
        candidate_id=candidate_id,
        archive_sha256=str(archive_record["archive"]["sha256"]),
        decode_device=decode_device,
    )
    if promoted is None:
        raise RuntimeError("decode succeeded but could not be promoted")
    receipt = {
        "schema": "ddm_sd2.real_decode.v1",
        "candidate_id": candidate_id,
        "completed_at_utc": utc_now(),
        "receiver_commit": RUNTIME_COMMIT,
        "receiver_source": RUNTIME_SOURCE_ROOT,
        "archive": archive_record,
        "raw": promoted,
        "attempt": str(attempt.resolve()),
        "decode_device": decode_device,
        "codec_inspection": codec_inspection,
        "token_checkpoint_policy": {
            key: value for key, value in checkpoint_policy.items() if key != "environment"
        },
        "subprocess": success,
    }
    atomic_write_json(final_receipt, receipt)
    return receipt


def initialize_progress(
    path: Path,
    *,
    configuration: Mapping[str, Any],
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    if path.exists():
        progress = json.loads(path.read_text())
        if progress.get("configuration") != configuration:
            raise ValueError("resume configuration differs")
        if progress.get("fingerprints") != fingerprints:
            prior_fingerprints = progress.get("fingerprints")
            if not isinstance(prior_fingerprints, Mapping):
                raise ValueError("resume fingerprints are not an object")
            differing = {
                key
                for key in set(prior_fingerprints) | set(fingerprints)
                if prior_fingerprints.get(key) != fingerprints.get(key)
            }
            out_dir = Path(str(configuration["out_dir"]))
            final_decode_paths = [
                out_dir / f"retained/decode/{candidate_id}/0.raw"
                for candidate_id in (BASE_ID, CANDIDATE_ID)
            ]
            decode_receipts = [
                path.parent / "DECODE_RECEIPT.json" for path in final_decode_paths
            ]
            chunks_root = out_dir / "retained/chunks"
            chunk_payload_exists = chunks_root.exists() and any(
                path.is_file() for path in chunks_root.rglob("*")
            )
            safe_runner_only_migration = (
                differing == {"runner"}
                and progress.get("completed_stages") == ["retention_preflight"]
                and progress.get("chunks") == []
                and progress.get("active_chunk") is None
                and not any(path.exists() for path in final_decode_paths + decode_receipts)
                and not chunk_payload_exists
            )
            if not safe_runner_only_migration:
                raise ValueError(
                    "resume fingerprints differ outside the safe pre-decode runner-only migration: "
                    f"{sorted(differing)}"
                )
            migration = {
                "schema": "ddm_sd2.progress_runner_migration.v1",
                "migrated_at_utc": utc_now(),
                "from_runner_sha256": prior_fingerprints["runner"],
                "to_runner_sha256": fingerprints["runner"],
                "admission": "PRE_DECODE_ONLY_NO_RETAINED_SCORER_CHUNKS",
                "reason": (
                    "caller repair binds token checkpoint requests to the retained "
                    "archive's actual token codec"
                ),
            }
            progress.setdefault("migrations", []).append(migration)
            progress["fingerprints"] = dict(fingerprints)
            progress["updated_at_utc"] = utc_now()
            atomic_write_json(path, progress)
        return progress
    progress = {
        "schema": "ddm_sd2.progress.v1",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "configuration": dict(configuration),
        "fingerprints": dict(fingerprints),
        "completed_stages": [],
        "chunks": [],
        "active_chunk": None,
    }
    atomic_write_json(path, progress)
    return progress


def validate_progress_chunks(progress: Mapping[str, Any], *, pair_count: int, chunk_pairs: int) -> int:
    """Fail closed unless retained progress is one contiguous population prefix."""

    chunks = progress.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("resume progress chunks must be a list")
    cursor = 0
    for row in chunks:
        if not isinstance(row, Mapping):
            raise ValueError("resume progress contains a non-object chunk")
        start = int(row["pair_start"])
        end = int(row["pair_end_exclusive"])
        expected_end = min(start + chunk_pairs, pair_count)
        if start != cursor or end != expected_end or end <= start:
            raise ValueError(
                "resume chunks are not the exact contiguous configured prefix: "
                f"cursor={cursor} range=({start},{end}) expected_end={expected_end}"
            )
        manifest_record = row.get("manifest")
        if not isinstance(manifest_record, Mapping):
            raise ValueError("resume chunk lacks its manifest artifact record")
        manifest_path = Path(str(manifest_record["path"]))
        if artifact(manifest_path) != dict(manifest_record):
            raise ValueError(f"resume chunk manifest differs on disk: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if (
            int(manifest.get("pair_start", -1)) != start
            or int(manifest.get("pair_end_exclusive", -1)) != end
            or int(manifest.get("pair_count", -1)) != end - start
        ):
            raise ValueError(f"resume chunk manifest range differs: {manifest_path}")
        cursor = end
    active = progress.get("active_chunk")
    if active is not None:
        if not isinstance(active, Mapping):
            raise ValueError("resume active_chunk must be an object or null")
        start = int(active["pair_start"])
        end = int(active["pair_end_exclusive"])
        filled = int(active["filled_pairs"])
        expected_end = min(start + chunk_pairs, pair_count)
        if start != cursor or end != expected_end or end <= start or filled < 0 or filled > end - start:
            raise ValueError(
                "resume active chunk is not the exact next configured range: "
                f"cursor={cursor} range=({start},{end}) filled={filled} "
                f"expected_end={expected_end}"
            )
    if cursor > pair_count:
        raise ValueError("resume chunks exceed the configured population")
    return cursor


def load_progress_manifests(progress: Mapping[str, Any], *, pair_count: int) -> list[dict[str, Any]]:
    manifests = [json.loads(Path(row["manifest"]["path"]).read_text()) for row in progress["chunks"]]
    if sum(int(row["pair_count"]) for row in manifests) != pair_count:
        raise ValueError("retained chunk population denominator differs")
    return manifests


def mark_stage(progress: dict[str, Any], progress_path: Path, stage: str) -> None:
    progress["completed_stages"] = sorted(set(progress["completed_stages"]) | {stage})
    progress["updated_at_utc"] = utc_now()
    atomic_write_json(progress_path, progress)


def import_upstream(challenge_root: Path) -> tuple[Any, Any]:
    root_text = str(challenge_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    frame_utils = importlib.import_module("frame_utils")
    modules = importlib.import_module("modules")
    if Path(frame_utils.__file__).resolve().parent != challenge_root.resolve():
        raise ImportError("frame_utils did not resolve from the pinned challenge root")
    if Path(modules.__file__).resolve().parent != challenge_root.resolve():
        raise ImportError("modules did not resolve from the pinned challenge root")
    return frame_utils, modules


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def scorer_forward(
    frames: torch.Tensor,
    *,
    segnet: Any,
    posenet: Any,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if frames.dtype != torch.uint8:
        raise ValueError(f"camera frames must be uint8, got {frames.dtype}")
    if tuple(frames.shape[1:]) != (SEQ_LEN, CAMERA_H, CAMERA_W, 3):
        raise ValueError(f"camera frame geometry differs: {tuple(frames.shape)}")
    value = frames.to(device=device, dtype=torch.float32)
    value = value.permute(0, 1, 4, 2, 3).contiguous()
    with torch.inference_mode():
        seg_input = segnet.preprocess_input(value)
        seg_logits = segnet(seg_input)
        pose_input = posenet.preprocess_input(value)
        pose_output = posenet(pose_input)["pose"]
    synchronize(device)
    logits = seg_logits.detach().to(device="cpu", dtype=torch.float32).numpy()
    argmax = seg_logits.argmax(dim=1).detach().to(device="cpu", dtype=torch.uint8).numpy()
    pose = pose_output.detach().to(device="cpu", dtype=torch.float32).numpy()
    if tuple(logits.shape[1:]) != (NUM_CLASSES, EVAL_H, EVAL_W):
        raise ValueError(f"SegNet logit geometry differs: {logits.shape}")
    if tuple(argmax.shape[1:]) != (EVAL_H, EVAL_W):
        raise ValueError(f"SegNet argmax geometry differs: {argmax.shape}")
    if tuple(pose.shape[1:]) != (POSE_OUTPUTS,):
        raise ValueError(f"PoseNet output geometry differs: {pose.shape}")
    return {"segnet_logits": logits, "segnet_argmax": argmax, "posenet": pose}


def chunk_array_specs(chunk_pairs: int) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    specs: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
        "target/camera_frames": (
            np.dtype(np.uint8),
            (chunk_pairs, SEQ_LEN, CAMERA_H, CAMERA_W, 3),
        )
    }
    for source in ("target", "base", "candidate"):
        specs[f"{source}/segnet_logits"] = (
            np.dtype(np.float32),
            (chunk_pairs, NUM_CLASSES, EVAL_H, EVAL_W),
        )
        specs[f"{source}/segnet_argmax"] = (
            np.dtype(np.uint8),
            (chunk_pairs, EVAL_H, EVAL_W),
        )
        specs[f"{source}/posenet"] = (
            np.dtype(np.float32),
            (chunk_pairs, POSE_OUTPUTS),
        )
    return specs


def array_paths(chunk_dir: Path, key: str) -> tuple[Path, Path]:
    source, field = key.split("/", maxsplit=1)
    final = chunk_dir / source / f"{field}.npy"
    partial = final.with_name(f".{final.name}.partial.npy")
    return partial, final


def open_chunk_arrays(
    chunk_dir: Path,
    specs: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]],
) -> dict[str, np.memmap]:
    arrays = {}
    for key, (dtype, shape) in specs.items():
        partial, final = array_paths(chunk_dir, key)
        partial.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            arrays[key] = np.lib.format.open_memmap(final, mode="r+")
        elif partial.exists():
            arrays[key] = np.lib.format.open_memmap(partial, mode="r+")
        else:
            arrays[key] = np.lib.format.open_memmap(
                partial,
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
        if arrays[key].dtype != dtype or tuple(arrays[key].shape) != shape:
            raise ValueError(f"retained chunk array spec differs for {key}")
    return arrays


def flush_chunk_arrays(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
        path = Path(array.filename)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def finalize_chunk_arrays(
    chunk_dir: Path,
    specs: Mapping[str, tuple[np.dtype[Any], tuple[int, ...]]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key in specs:
        partial, final = array_paths(chunk_dir, key)
        if not final.exists():
            if not partial.exists():
                raise FileNotFoundError(f"chunk payload missing: {key}")
            os.replace(partial, final)
        records[key] = artifact(final)
    return records


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError("boundary labels must have shape (N,H,W)")
    boundary = np.zeros(values.shape, dtype=bool)
    horizontal = values[:, :, 1:] != values[:, :, :-1]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical = values[:, 1:, :] != values[:, :-1, :]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary


def open_verified_npy(record: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(record["path"]))
    if artifact(path) != dict(record):
        raise ValueError(f"retained numpy artifact differs on disk: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def class_signatures(target_chunks: Sequence[np.ndarray]) -> list[dict[str, float]]:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    y_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    x_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    top_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    bottom_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    center_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels = 0
    yy = np.arange(EVAL_H, dtype=np.float64)[:, None]
    xx = np.arange(EVAL_W, dtype=np.float64)[None, :]
    for chunk in target_chunks:
        total_pixels += int(chunk.size)
        for class_index in range(NUM_CLASSES):
            mask = chunk == class_index
            count = int(mask.sum())
            counts[class_index] += count
            y_sums[class_index] += float((mask * yy).sum())
            x_sums[class_index] += float((mask * xx).sum())
            top_counts[class_index] += int(mask[:, : EVAL_H // 4].sum())
            bottom_counts[class_index] += int(mask[:, 3 * EVAL_H // 4 :].sum())
            center_counts[class_index] += int(mask[:, :, int(0.35 * EVAL_W) : int(0.65 * EVAL_W)].sum())
    if np.any(counts == 0):
        raise ValueError(f"class self-detection found empty classes: {counts.tolist()}")
    return [
        {
            "index": class_index,
            "area_fraction": float(counts[class_index] / total_pixels),
            "centroid_y": float(y_sums[class_index] / counts[class_index] / (EVAL_H - 1)),
            "centroid_x": float(x_sums[class_index] / counts[class_index] / (EVAL_W - 1)),
            "top_quarter_share": float(top_counts[class_index] / counts[class_index]),
            "bottom_quarter_share": float(bottom_counts[class_index] / counts[class_index]),
            "center_band_share": float(center_counts[class_index] / counts[class_index]),
        }
        for class_index in range(NUM_CLASSES)
    ]


def self_detect_class_order(signatures: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    by_index = {int(row["index"]): row for row in signatures}
    mycar = max(by_index, key=lambda idx: by_index[idx]["bottom_quarter_share"])
    undrivable = max(by_index, key=lambda idx: by_index[idx]["top_quarter_share"])
    remaining = set(by_index) - {mycar, undrivable}
    lane = min(remaining, key=lambda idx: by_index[idx]["area_fraction"])
    remaining.remove(lane)
    movable = max(remaining, key=lambda idx: by_index[idx]["centroid_x"])
    remaining.remove(movable)
    if len(remaining) != 1:
        raise ValueError("class self-detection did not leave exactly one Road index")
    road = remaining.pop()
    detected = {
        "Road": road,
        "Lane": lane,
        "Undrivable": undrivable,
        "Movable": movable,
        "MyCar": mycar,
    }
    guards = {
        "MyCar_bottom_quarter_share_gt_0p90": by_index[mycar]["bottom_quarter_share"] > 0.90,
        "MyCar_centroid_y_gt_0p80": by_index[mycar]["centroid_y"] > 0.80,
        "Undrivable_top_quarter_share_gt_0p45": (by_index[undrivable]["top_quarter_share"] > 0.45),
        "Undrivable_centroid_y_lt_0p35": by_index[undrivable]["centroid_y"] < 0.35,
        "Lane_area_fraction_lt_0p02": by_index[lane]["area_fraction"] < 0.02,
        "Movable_centroid_x_gt_0p65": by_index[movable]["centroid_x"] > 0.65,
        "Road_centroid_y_between_0p50_and_0p75": (0.50 < by_index[road]["centroid_y"] < 0.75),
    }
    canonical = {name: index for index, name in enumerate(CANONICAL_CLASS_NAMES)}
    if not all(guards.values()) or detected != canonical:
        raise ValueError(
            "SegNet class self-detection refused canonical labeling: "
            f"detected={detected} guards={guards} signatures={signatures}"
        )
    return {
        "method": (
            "Self-detected from population area, vertical centroid, horizontal "
            "centroid, and top/bottom spatial shares; no luma sort."
        ),
        "detected_name_to_index": detected,
        "canonical_name_to_index": canonical,
        "guards": guards,
        "signatures": list(signatures),
        "status": "PASS",
    }


def gini(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.sum() == 0:
        return 0.0
    ordered = np.sort(array)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * ordered) / (ordered.size * ordered.sum())) - (ordered.size + 1.0) / ordered.size)


def concentration_summary(frame_errors: np.ndarray) -> dict[str, Any]:
    values = np.asarray(frame_errors, dtype=np.int64)
    total = int(values.sum())
    ordered = np.sort(values)[::-1]

    def top_share(fraction: float) -> float:
        count = max(1, math.ceil(fraction * len(ordered)))
        return float(ordered[:count].sum() / total) if total else 0.0

    top10 = top_share(0.10)
    value_gini = gini(values)
    if top10 >= 0.25 and value_gini >= 0.40:
        classification = "HEAVY_TAIL_BY_PREREGISTERED_CONCENTRATION_RULE"
    elif top10 <= 0.15 and value_gini <= 0.20:
        classification = "DIFFUSE_BY_PREREGISTERED_CONCENTRATION_RULE"
    else:
        classification = "INTERMEDIATE_BY_PREREGISTERED_CONCENTRATION_RULE"
    return {
        "frame_count": len(values),
        "error_pixels": total,
        "gini": value_gini,
        "top_1pct_frame_share": top_share(0.01),
        "top_5pct_frame_share": top_share(0.05),
        "top_10pct_frame_share": top10,
        "top_25pct_frame_share": top_share(0.25),
        "classification": classification,
        "classification_rule": (
            "heavy tail iff top-10% share >=0.25 and Gini >=0.40; diffuse iff "
            "top-10% share <=0.15 and Gini <=0.20; otherwise intermediate"
        ),
    }


def matrix_rows(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    mismatch_denominator = int(matrix.sum() - np.trace(matrix))
    pixel_denominator = int(matrix.sum())
    for target in range(NUM_CLASSES):
        for prediction in range(NUM_CLASSES):
            count = int(matrix[target, prediction])
            rows.append(
                {
                    "target_index": target,
                    "target_class": CANONICAL_CLASS_NAMES[target],
                    "prediction_index": prediction,
                    "prediction_class": CANONICAL_CLASS_NAMES[prediction],
                    "pixels": count,
                    "share_of_all_pixels": count / pixel_denominator,
                    "share_of_mismatches": (
                        count / mismatch_denominator if target != prediction and mismatch_denominator else 0.0
                    ),
                }
            )
    return rows


def decompose_retained_argmax(chunk_manifests: Sequence[Mapping[str, Any]], prediction_source: str) -> dict[str, Any]:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    boundary_matrix = np.zeros_like(matrix)
    interior_matrix = np.zeros_like(matrix)
    frame_rows: list[dict[str, Any]] = []
    target_chunks: list[np.ndarray] = []
    boundary_sites = 0
    interior_sites = 0
    pair_cursor = 0
    for manifest in chunk_manifests:
        arrays = manifest["arrays"]
        target = open_verified_npy(arrays["target/segnet_argmax"])
        prediction = open_verified_npy(arrays[f"{prediction_source}/segnet_argmax"])
        if target.shape != prediction.shape or target.dtype != np.uint8 or prediction.dtype != np.uint8:
            raise ValueError("retained argmax chunks differ in geometry or dtype")
        if (
            int(np.asarray(target).min()) < 0
            or int(np.asarray(target).max()) >= NUM_CLASSES
            or int(np.asarray(prediction).min()) < 0
            or int(np.asarray(prediction).max()) >= NUM_CLASSES
        ):
            raise ValueError("retained argmax chunks contain out-of-range classes")
        target_chunks.append(np.asarray(target))
        bmask = boundary_mask(target)
        boundary_sites += int(bmask.sum())
        interior_sites += int((~bmask).sum())
        flat_target = np.asarray(target, dtype=np.int64).reshape(-1)
        flat_prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
        flat_boundary = bmask.reshape(-1)
        joint = flat_target * NUM_CLASSES + flat_prediction
        matrix += np.bincount(joint, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        boundary_matrix += np.bincount(joint[flat_boundary], minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
        interior_matrix += np.bincount(joint[~flat_boundary], minlength=NUM_CLASSES**2).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
        errors = np.asarray(target) != np.asarray(prediction)
        for local_index in range(len(target)):
            count = int(errors[local_index].sum())
            boundary_error = int((errors[local_index] & bmask[local_index]).sum())
            frame_rows.append(
                {
                    "pair_index": pair_cursor + local_index,
                    "error_pixels": count,
                    "d_seg": count / (EVAL_H * EVAL_W),
                    "boundary_error_pixels": boundary_error,
                    "interior_error_pixels": count - boundary_error,
                }
            )
        pair_cursor += len(target)
    class_detection = self_detect_class_order(class_signatures(target_chunks))
    pixel_denominator = int(matrix.sum())
    mismatch_denominator = int(matrix.sum() - np.trace(matrix))
    if pixel_denominator != pair_cursor * EVAL_H * EVAL_W:
        raise ValueError("retained matrix denominator differs from retained pair count")
    directed_edges = []
    symmetric_edges = []
    for target in range(NUM_CLASSES):
        for prediction in range(NUM_CLASSES):
            if target == prediction:
                continue
            count = int(matrix[target, prediction])
            directed_edges.append(
                {
                    "edge": (f"{CANONICAL_CLASS_NAMES[target]}->{CANONICAL_CLASS_NAMES[prediction]}"),
                    "pixels": count,
                    "share_of_mismatches": (count / mismatch_denominator if mismatch_denominator else 0.0),
                    "boundary_pixels": int(boundary_matrix[target, prediction]),
                    "interior_pixels": int(interior_matrix[target, prediction]),
                }
            )
    for left in range(NUM_CLASSES):
        for right in range(left + 1, NUM_CLASSES):
            count = int(matrix[left, right] + matrix[right, left])
            symmetric_edges.append(
                {
                    "edge": (f"{CANONICAL_CLASS_NAMES[left]}<->{CANONICAL_CLASS_NAMES[right]}"),
                    "pixels": count,
                    "share_of_mismatches": (count / mismatch_denominator if mismatch_denominator else 0.0),
                    "forward_pixels": int(matrix[left, right]),
                    "reverse_pixels": int(matrix[right, left]),
                }
            )
    directed_edges.sort(key=lambda row: (-row["pixels"], row["edge"]))
    symmetric_edges.sort(key=lambda row: (-row["pixels"], row["edge"]))
    frame_rows.sort(key=lambda row: row["pair_index"])
    ranked_frames = sorted(frame_rows, key=lambda row: (-row["error_pixels"], row["pair_index"]))
    return {
        "prediction_source": prediction_source,
        "class_order": class_detection,
        "denominators": {
            "pairs": pair_cursor,
            "evaluated_pixels": pixel_denominator,
            "exact_mismatch_pixels": mismatch_denominator,
            "target_boundary_pixels": boundary_sites,
            "target_interior_pixels": interior_sites,
        },
        "d_seg": mismatch_denominator / pixel_denominator,
        "directed_target_rows_prediction_columns": matrix.tolist(),
        "directed_matrix_rows": matrix_rows(matrix),
        "directed_mismatch_edges": directed_edges,
        "symmetric_edges": symmetric_edges,
        "boundary_interior": {
            "definition": (
                "boundary means either side of a 4-neighbor target-class transition; interior is its exact complement"
            ),
            "directed_boundary_matrix": boundary_matrix.tolist(),
            "directed_interior_matrix": interior_matrix.tolist(),
            "boundary_mismatch_pixels": int(boundary_matrix.sum() - np.trace(boundary_matrix)),
            "interior_mismatch_pixels": int(interior_matrix.sum() - np.trace(interior_matrix)),
            "boundary_error_rate": (
                (boundary_matrix.sum() - np.trace(boundary_matrix)) / boundary_sites if boundary_sites else 0.0
            ),
            "interior_error_rate": (
                (interior_matrix.sum() - np.trace(interior_matrix)) / interior_sites if interior_sites else 0.0
            ),
        },
        "per_frame": frame_rows,
        "top_error_frames": ranked_frames[: min(60, len(ranked_frames))],
        "frame_error_concentration": concentration_summary(
            np.asarray([row["error_pixels"] for row in frame_rows], dtype=np.int64)
        ),
    }


def score_from_retained(
    *,
    chunk_manifests: Sequence[Mapping[str, Any]],
    prediction_source: str,
    d_seg: float,
    archive_bytes: int,
    uncompressed_bytes: int,
) -> dict[str, Any]:
    squared_error_sum = 0.0
    pair_count = 0
    for manifest in chunk_manifests:
        arrays = manifest["arrays"]
        target = open_verified_npy(arrays["target/posenet"])
        prediction = open_verified_npy(arrays[f"{prediction_source}/posenet"])
        if target.shape != prediction.shape or target.shape[1] != POSE_OUTPUTS:
            raise ValueError("retained PoseNet chunks differ")
        delta = np.asarray(target[:, :SCORED_POSE_OUTPUTS], dtype=np.float64) - np.asarray(
            prediction[:, :SCORED_POSE_OUTPUTS], dtype=np.float64
        )
        squared_error_sum += float(np.square(delta).sum())
        pair_count += len(target)
    d_pose = squared_error_sum / (pair_count * SCORED_POSE_OUTPUTS)
    rate = archive_bytes / uncompressed_bytes
    score = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * rate
    return {
        "pair_count": pair_count,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "uncompressed_bytes": uncompressed_bytes,
        "rate": rate,
        "seg_contribution": 100.0 * d_seg,
        "pose_contribution": math.sqrt(10.0 * d_pose),
        "rate_contribution": 25.0 * rate,
        "recomputed_s": score,
    }


def finalize_active_chunk(
    *,
    args: argparse.Namespace,
    progress: dict[str, Any],
    progress_path: Path,
    base_decode: Mapping[str, Any],
    candidate_decode: Mapping[str, Any],
    scorer_axis: str,
) -> dict[str, Any]:
    active = progress.get("active_chunk")
    if not isinstance(active, Mapping):
        raise ValueError("no active chunk is available for finalization")
    chunk_start = int(active["pair_start"])
    chunk_end = int(active["pair_end_exclusive"])
    chunk_size = chunk_end - chunk_start
    if int(active["filled_pairs"]) != chunk_size:
        raise ValueError("active chunk is not fully persisted")
    range_key = (chunk_start, chunk_end)
    existing_ranges = {(int(row["pair_start"]), int(row["pair_end_exclusive"])) for row in progress["chunks"]}
    if range_key in existing_ranges:
        raise ValueError(f"chunk range was already finalized: {range_key}")
    chunk_dir = args.out_dir / f"retained/chunks/pairs_{chunk_start:04d}_{chunk_end - 1:04d}"
    specs = chunk_array_specs(chunk_size)
    array_records = finalize_chunk_arrays(chunk_dir, specs)
    frame_bytes = SEQ_LEN * CAMERA_H * CAMERA_W * 3
    manifest = {
        "schema": "ddm_sd2.retained_chunk.v1",
        "axis": scorer_axis,
        "pair_start": chunk_start,
        "pair_end_exclusive": chunk_end,
        "pair_count": chunk_size,
        "arrays": array_records,
        "decoded_camera_ranges": {
            "base": range_artifact(
                Path(base_decode["raw"]["path"]),
                chunk_start * frame_bytes,
                chunk_size * frame_bytes,
            ),
            "candidate": range_artifact(
                Path(candidate_decode["raw"]["path"]),
                chunk_start * frame_bytes,
                chunk_size * frame_bytes,
            ),
            "target": array_records["target/camera_frames"],
        },
        "retention_note": (
            "Base and candidate camera chunks are byte ranges of the exact retained "
            "public-receiver raw files; target camera frames are physical NPY chunks."
        ),
        "completed_at_utc": utc_now(),
    }
    manifest_path = chunk_dir / "CHUNK_MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    progress["chunks"].append(
        {
            "pair_start": chunk_start,
            "pair_end_exclusive": chunk_end,
            "manifest": artifact(manifest_path),
        }
    )
    progress["active_chunk"] = None
    progress["updated_at_utc"] = utc_now()
    atomic_write_json(progress_path, progress)
    return manifest


def run_scorer_chunks(
    *,
    args: argparse.Namespace,
    progress: dict[str, Any],
    progress_path: Path,
    base_decode: Mapping[str, Any],
    candidate_decode: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frame_utils, modules = import_upstream(args.challenge_root)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS scorer fire was requested but Metal is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA scorer fire was requested but CUDA is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    scorer_axis = axis_label(device)
    completed_pairs = validate_progress_chunks(
        progress,
        pair_count=args.pair_count,
        chunk_pairs=args.chunk_pairs,
    )
    active = progress.get("active_chunk")
    if active is not None:
        active_size = int(active["pair_end_exclusive"]) - int(active["pair_start"])
        active_filled = int(active["filled_pairs"])
        if active_filled == active_size:
            finalize_active_chunk(
                args=args,
                progress=progress,
                progress_path=progress_path,
                base_decode=base_decode,
                candidate_decode=candidate_decode,
                scorer_axis=scorer_axis,
            )
            completed_pairs = validate_progress_chunks(
                progress,
                pair_count=args.pair_count,
                chunk_pairs=args.chunk_pairs,
            )
    if completed_pairs == args.pair_count:
        return load_progress_manifests(progress, pair_count=args.pair_count)

    segnet = modules.SegNet().eval().to(device)
    posenet = modules.PoseNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))

    video_names = args.video_names_file.read_text().splitlines()
    if video_names != ["0.mkv"]:
        raise ValueError(f"public video-name census differs: {video_names}")
    gt_class = frame_utils.DaliVideoDataset if device.type == "cuda" else frame_utils.AVVideoDataset
    gt_dataset = gt_class(
        video_names,
        data_dir=args.uncompressed_dir,
        batch_size=args.batch_size,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    base_dataset = frame_utils.TensorVideoDataset(
        video_names,
        data_dir=Path(base_decode["raw"]["path"]).parent,
        batch_size=args.batch_size,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    candidate_dataset = frame_utils.TensorVideoDataset(
        video_names,
        data_dir=Path(candidate_decode["raw"]["path"]).parent,
        batch_size=args.batch_size,
        device=device,
        num_threads=args.num_threads,
        seed=args.seed,
        prefetch_queue_depth=args.prefetch_queue_depth,
    )
    for dataset in (gt_dataset, base_dataset, candidate_dataset):
        dataset.prepare_data()
    loaders = [
        torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
        for dataset in (gt_dataset, base_dataset, candidate_dataset)
    ]

    active = progress.get("active_chunk")
    skip_pairs = completed_pairs + (int(active["filled_pairs"]) if active else 0)
    global_pair = 0
    arrays: dict[str, np.memmap] | None = None
    specs: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] | None = None
    chunk_dir: Path | None = None

    for gt_row, base_row, candidate_row in zip(*loaders, strict=True):
        gt_path, gt_batch_index, gt_frames = gt_row
        base_path, base_batch_index, base_frames = base_row
        candidate_path, candidate_batch_index, candidate_frames = candidate_row
        if not (
            Path(gt_path).stem == Path(base_path).stem == Path(candidate_path).stem == "0"
            and gt_batch_index == base_batch_index == candidate_batch_index
        ):
            raise ValueError("paired dataset identity or batch index differs")
        batch_pairs = len(gt_frames)
        if global_pair >= args.pair_count:
            break
        if global_pair + batch_pairs > args.pair_count:
            keep = args.pair_count - global_pair
            gt_frames = gt_frames[:keep]
            base_frames = base_frames[:keep]
            candidate_frames = candidate_frames[:keep]
            batch_pairs = keep
        if global_pair + batch_pairs <= skip_pairs:
            global_pair += batch_pairs
            continue
        if global_pair < skip_pairs:
            raise ValueError("resume offset is not aligned to the declared batch size")

        if active is None:
            chunk_start = global_pair
            chunk_end = min(chunk_start + args.chunk_pairs, args.pair_count)
            active = {
                "pair_start": chunk_start,
                "pair_end_exclusive": chunk_end,
                "filled_pairs": 0,
                "created_at_utc": utc_now(),
            }
            progress["active_chunk"] = active
            progress["updated_at_utc"] = utc_now()
            atomic_write_json(progress_path, progress)
        chunk_start = int(active["pair_start"])
        chunk_end = int(active["pair_end_exclusive"])
        chunk_size = chunk_end - chunk_start
        if global_pair + batch_pairs > chunk_end:
            raise ValueError("batch crosses a chunk boundary; choose divisible chunk_pairs")
        if arrays is None:
            chunk_dir = args.out_dir / f"retained/chunks/pairs_{chunk_start:04d}_{chunk_end - 1:04d}"
            specs = chunk_array_specs(chunk_size)
            arrays = open_chunk_arrays(chunk_dir, specs)
        local_start = int(active["filled_pairs"])
        local_end = local_start + batch_pairs
        arrays["target/camera_frames"][local_start:local_end] = gt_frames.detach().cpu().numpy()
        for source, frames in (
            ("target", gt_frames),
            ("base", base_frames),
            ("candidate", candidate_frames),
        ):
            outputs = scorer_forward(
                frames,
                segnet=segnet,
                posenet=posenet,
                device=device,
            )
            for field, value in outputs.items():
                arrays[f"{source}/{field}"][local_start:local_end] = value
        flush_chunk_arrays(arrays)
        active["filled_pairs"] = local_end
        active["last_batch_completed_at_utc"] = utc_now()
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(progress_path, progress)
        global_pair += batch_pairs

        if local_end == chunk_size:
            del arrays
            arrays = None
            if chunk_dir is None or specs is None:
                raise AssertionError("chunk finalization state missing")
            finalize_active_chunk(
                args=args,
                progress=progress,
                progress_path=progress_path,
                base_decode=base_decode,
                candidate_decode=candidate_decode,
                scorer_axis=scorer_axis,
            )
            active = None
            chunk_dir = None
            specs = None

    if global_pair != args.pair_count or progress.get("active_chunk") is not None:
        raise ValueError(f"scorer population incomplete: processed={global_pair} requested={args.pair_count}")
    return load_progress_manifests(progress, pair_count=args.pair_count)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--candidate-archive", type=Path, default=DEFAULT_CANDIDATE_ARCHIVE)
    parser.add_argument("--challenge-root", type=Path, default=DEFAULT_CHALLENGE_ROOT)
    parser.add_argument("--video-names-file", type=Path, default=None)
    parser.add_argument("--uncompressed-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--decode-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-pairs", type=int, default=60)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--prefetch-queue-depth", type=int, default=4)
    parser.add_argument("--minimum-free-bytes", type=int, default=5_000_000_000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.video_names_file is None:
        args.video_names_file = args.challenge_root / "public_test_video_names.txt"
    if args.uncompressed_dir is None:
        args.uncompressed_dir = args.challenge_root / "videos"
    if not 1 <= args.pair_count <= PAIR_COUNT:
        parser.error("--pair-count must be in [1,600]")
    if not 1 <= args.chunk_pairs <= 120:
        parser.error("--chunk-pairs must be in [1,120]")
    if args.chunk_pairs % args.batch_size or args.pair_count % args.batch_size:
        parser.error("--chunk-pairs and --pair-count must be divisible by --batch-size")
    if args.pair_count < PAIR_COUNT and not args.plan_only:
        parser.error(
            "reduced populations are not verdicts; use a separate stratified-random "
            "pair-id charter rather than a prefix"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = require_bulk_store(args.out_dir)
    args.resume_from = require_bulk_store(args.resume_from)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise RuntimeError(
            f"bulk custody unavailable at the charter-mandated APDataStore path: {args.out_dir}"
        ) from error
    if args.resume_from.parent != args.out_dir:
        raise ValueError("--resume-from must be directly inside --out-dir")
    queue = json.loads(args.queue.read_text())
    retention = validate_queue(queue)
    if args.chunk_pairs > int(retention["chunk_pair_limit"]):
        raise ValueError("requested chunk size exceeds the SG2 retention contract")
    if args.minimum_free_bytes < int(retention["minimum_storage_preflight_bytes"]):
        raise ValueError("minimum-free-bytes is below the SG2 queue floor")

    base_source = require_pinned_file(args.base_archive, BASE_ARCHIVE_BYTES, BASE_ARCHIVE_SHA256)
    candidate_source = require_pinned_file(
        args.candidate_archive,
        CANDIDATE_ARCHIVE_BYTES,
        CANDIDATE_ARCHIVE_SHA256,
    )
    video_names = args.video_names_file.read_text().splitlines()
    if video_names != ["0.mkv"]:
        raise ValueError(f"public video-name census differs: {video_names}")
    source_video = artifact(args.uncompressed_dir / video_names[0])
    if int(source_video["bytes"]) != ORIGINAL_VIDEO_BYTES:
        raise ValueError(f"uncompressed video denominator differs: {source_video}")
    queue_record = artifact(args.queue)
    projection = storage_projection(args.pair_count, args.chunk_pairs)
    free_before = shutil.disk_usage(args.out_dir).free
    projection_credit = retained_projection_credit(args.out_dir, projection["projected_final_bytes"])
    final_raw_paths = [
        args.out_dir / f"retained/decode/{candidate_id}/0.raw" for candidate_id in (BASE_ID, CANDIDATE_ID)
    ]
    decode_contingency = (
        projection["failed_decode_attempt_contingency_bytes"]
        if any(not path.is_file() for path in final_raw_paths)
        else 0
    )
    projected_remaining = max(
        projection["projected_final_bytes"] - projection_credit,
        0,
    )
    required_free = projected_remaining + decode_contingency + args.minimum_free_bytes
    if free_before < required_free:
        raise RuntimeError(
            "storage preflight refused: "
            f"free={free_before} required={required_free} "
            f"projected_final={projection['projected_final_bytes']} "
            f"reserve={args.minimum_free_bytes}"
        )

    writer_receipt = writer_preflight(args.out_dir, retention)
    runtime_dir = args.out_dir / "retained/runtime/receiver_58f62cd22f"
    runtime_records = materialize_runtime(runtime_dir)
    gate_receipt = payload_retention_preflight(
        Path(__file__).resolve(),
        runtime_records,
        args.out_dir / "preflight/PAYLOAD_RETENTION_GATE.json",
    )
    archives = {
        "base": retain_archive(
            source=args.base_archive,
            destination_dir=args.out_dir / f"retained/candidates/{BASE_ID}",
            expected_bytes=BASE_ARCHIVE_BYTES,
            expected_sha256=BASE_ARCHIVE_SHA256,
        ),
        "candidate": retain_archive(
            source=args.candidate_archive,
            destination_dir=args.out_dir / f"retained/candidates/{CANDIDATE_ID}",
            expected_bytes=CANDIDATE_ARCHIVE_BYTES,
            expected_sha256=CANDIDATE_ARCHIVE_SHA256,
        ),
    }
    decode_capability_preflight = {}
    for role, candidate_id in (
        ("base", BASE_ID),
        ("candidate", CANDIDATE_ID),
    ):
        codec_inspection = inspect_retained_token_codec(
            runtime_dir,
            archives[role]["member_p"],
        )
        checkpoint_policy = token_checkpoint_policy(
            decode_root=args.out_dir / f"retained/decode/{candidate_id}",
            codec_inspection=codec_inspection,
        )
        decode_capability_preflight[role] = {
            "candidate_id": candidate_id,
            "codec_inspection": codec_inspection,
            "checkpoint_policy": {
                key: value
                for key, value in checkpoint_policy.items()
                if key != "environment"
            },
            "checkpoint_environment_keys": sorted(checkpoint_policy["environment"]),
        }
    environment = environment_provenance()
    preflight = {
        "schema": "ddm_sd2.retention_preflight.v1",
        "status": "READY_FOR_MAIN_FIRE",
        "score_claim": False,
        "scorer_ran": False,
        "checked_at_utc": utc_now(),
        "queue": queue_record,
        "required_retention": retention,
        "writer_preflight": writer_receipt,
        "payload_retention_gate": gate_receipt,
        "sources": {
            "base": base_source,
            "candidate": candidate_source,
            "uncompressed_video": source_video,
        },
        "environment": environment,
        "archives": archives,
        "receiver": {
            "commit": RUNTIME_COMMIT,
            "source_path": RUNTIME_SOURCE_ROOT,
            "materialized_files": runtime_records,
            "decode_capability_preflight": decode_capability_preflight,
        },
        "storage": {
            "free_bytes_before_preflight": free_before,
            "projection": projection,
            "credited_retained_final_bytes": projection_credit,
            "projected_remaining_bytes": projected_remaining,
            "decode_failure_contingency_required_bytes": decode_contingency,
            "minimum_reserve_bytes": args.minimum_free_bytes,
            "required_free_bytes_at_admission": required_free,
            "admitted": True,
        },
        "exact_main_fire_command": fire_command(args),
        "runtime_projection": {
            "decode": (
                "Two serial CPU public-receiver passes; prior same-family n600 decode "
                "measured about 1,011 seconds for one archive."
            ),
            "scorer": (
                "One paired n600 target/base/candidate pass in 60-pair retained chunks; "
                "runtime is unmeasured for this three-way full-retention runner."
            ),
            "honest_wallclock_band": "approximately 45-90 minutes on MAIN Metal, unmeasured",
        },
        "boundaries": [
            "No scorer ran during preflight.",
            "The full fire remains advisory and cannot promote a contest score.",
            "The runner retains full SegNet logits although SG2 only required argmax.",
            "The committed receiver is materialized from Git, not read from the dirty runtime worktree.",
        ],
    }
    preflight_path = args.out_dir / "SD2_RETENTION_PREFLIGHT.json"
    atomic_write_json(preflight_path, preflight)
    if args.plan_only:
        print(json.dumps({**preflight, "receipt": artifact(preflight_path)}, indent=2))
        return 0

    configuration = {
        "out_dir": str(args.out_dir),
        "resume_from": str(args.resume_from),
        "queue": str(args.queue.resolve()),
        "base_archive": str(args.base_archive.resolve()),
        "candidate_archive": str(args.candidate_archive.resolve()),
        "challenge_root": str(args.challenge_root.resolve()),
        "video_names_file": str(args.video_names_file.resolve()),
        "uncompressed_dir": str(args.uncompressed_dir.resolve()),
        "device": args.device,
        "decode_device": args.decode_device,
        "batch_size": args.batch_size,
        "chunk_pairs": args.chunk_pairs,
        "pair_count": args.pair_count,
        "selection_mode": "full_population_n600",
        "seed": args.seed,
        "cpu_threads": args.cpu_threads,
        "num_threads": args.num_threads,
        "prefetch_queue_depth": args.prefetch_queue_depth,
        "minimum_free_bytes": args.minimum_free_bytes,
        "determinism": "seeded; torch deterministic algorithms required; no random selection",
    }
    fingerprints = {
        "runner": sha256_file(Path(__file__).resolve()),
        "queue": queue_record["sha256"],
        "base_archive": BASE_ARCHIVE_SHA256,
        "candidate_archive": CANDIDATE_ARCHIVE_SHA256,
        "receiver_commit": RUNTIME_COMMIT,
        "upstream_modules": sha256_file(args.challenge_root / "modules.py"),
        "upstream_frame_utils": sha256_file(args.challenge_root / "frame_utils.py"),
        "segnet_weights": sha256_file(args.challenge_root / "models/segnet.safetensors"),
        "posenet_weights": sha256_file(args.challenge_root / "models/posenet.safetensors"),
        "video_names": sha256_file(args.video_names_file),
        "uncompressed_video": source_video["sha256"],
        "software_environment": sha256_bytes(
            json.dumps(
                {
                    key: environment[key]
                    for key in (
                        "python",
                        "executable",
                        "platform",
                        "machine",
                        "package_versions",
                        "torch_mps_available",
                        "torch_cuda_available",
                    )
                },
                sort_keys=True,
            ).encode()
        ),
    }
    progress = initialize_progress(
        args.resume_from,
        configuration=configuration,
        fingerprints=fingerprints,
    )
    mark_stage(progress, args.resume_from, "retention_preflight")
    base_decode = ensure_real_decode(
        out_dir=args.out_dir,
        candidate_id=BASE_ID,
        archive_record=archives["base"],
        runtime_dir=runtime_dir,
        video_names_file=args.video_names_file,
        decode_device=args.decode_device,
    )
    mark_stage(progress, args.resume_from, "base_real_decode")
    candidate_decode = ensure_real_decode(
        out_dir=args.out_dir,
        candidate_id=CANDIDATE_ID,
        archive_record=archives["candidate"],
        runtime_dir=runtime_dir,
        video_names_file=args.video_names_file,
        decode_device=args.decode_device,
    )
    mark_stage(progress, args.resume_from, "candidate_real_decode")
    manifests = run_scorer_chunks(
        args=args,
        progress=progress,
        progress_path=args.resume_from,
        base_decode=base_decode,
        candidate_decode=candidate_decode,
    )
    mark_stage(progress, args.resume_from, "retained_scorer_chunks")

    base_decomposition = decompose_retained_argmax(manifests, "base")
    candidate_decomposition = decompose_retained_argmax(manifests, "candidate")
    uncompressed_bytes = sum(path.stat().st_size for path in args.uncompressed_dir.rglob("*") if path.is_file())
    if uncompressed_bytes != ORIGINAL_VIDEO_BYTES:
        raise ValueError(f"uncompressed video denominator differs: {uncompressed_bytes}")
    base_score = score_from_retained(
        chunk_manifests=manifests,
        prediction_source="base",
        d_seg=base_decomposition["d_seg"],
        archive_bytes=BASE_ARCHIVE_BYTES,
        uncompressed_bytes=uncompressed_bytes,
    )
    candidate_score = score_from_retained(
        chunk_manifests=manifests,
        prediction_source="candidate",
        d_seg=candidate_decomposition["d_seg"],
        archive_bytes=CANDIDATE_ARCHIVE_BYTES,
        uncompressed_bytes=uncompressed_bytes,
    )
    decomposition_payload = {
        "schema": "ddm_sd2.seg_decomposition.v1",
        "axis": axis_label(torch.device(args.device)),
        "score_claim": False,
        "computed_only_from_retained_argmax": True,
        "base": base_decomposition,
        "candidate": candidate_decomposition,
    }
    decomposition_path = args.out_dir / "SD2_SEG_DECOMPOSITION.json"
    atomic_write_json(decomposition_path, decomposition_payload)
    mark_stage(progress, args.resume_from, "final_decomposition_ready")
    final = {
        "schema": "ddm_sd2.result.v1",
        "status": "COMPLETE_ADVISORY_RETAINED_PAIRED_N600",
        "axis": axis_label(torch.device(args.device)),
        "score_claim": False,
        "pointer_moved": False,
        "completed_at_utc": utc_now(),
        "configuration": configuration,
        "fingerprints": fingerprints,
        "preflight": artifact(preflight_path),
        "progress": artifact(args.resume_from),
        "decode": {"base": base_decode, "candidate": candidate_decode},
        "chunk_manifests": [row["manifest"] for row in progress["chunks"]],
        "decomposition": artifact(decomposition_path),
        "scores_recomputed_from_retained_outputs": {
            "base": base_score,
            "candidate": candidate_score,
            "delta_candidate_minus_base": {
                "d_seg": candidate_score["d_seg"] - base_score["d_seg"],
                "d_pose": candidate_score["d_pose"] - base_score["d_pose"],
                "archive_bytes": CANDIDATE_ARCHIVE_BYTES - BASE_ARCHIVE_BYTES,
                "S": candidate_score["recomputed_s"] - base_score["recomputed_s"],
            },
        },
        "boundaries": [
            "This is not contest-CPU or contest-CUDA authority.",
            "No rounded displayed d_seg value was inverted; exact retained mismatch counts were used.",
            "Base/candidate camera chunks are ranges of retained exact receiver raw files.",
            "Class names were admitted only after spatial-signature self-detection matched canonical order.",
        ],
    }
    final_path = args.out_dir / "SD2_RESULT.json"
    atomic_write_json(final_path, final)
    print(json.dumps({**final, "result": artifact(final_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
