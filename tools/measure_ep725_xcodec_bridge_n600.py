#!/usr/bin/env python3
"""Measure one exact full-n600 same-object bridge row for the ep725 runtime.

The runner is deliberately staged and crash-resumable:

1. reopen a terminal G22 full-population equality receipt;
2. extract and render the requested exact archive through the frozen generic
   runtime in immutable pair chunks;
3. require the whole realized uint8 SHA-256 to equal G22's reference witness;
4. run the deterministic full-precision CPU-Torch frozen scorer;
5. materialize a coupled score envelope against the live canonical pointer;
6. certify the large raw witness as reproducible, then remove only that scratch.

The result is a macOS-CPU advisory scientific bridge, never a contest score or
pointer mutation.  A different same-state recode (for example G25) is accepted
only with an evidence JSON that contains the exact archive identity, and its
realized output must still equal the terminal G22 reference byte-for-byte.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tac.witness_dsl.taskspace_ep725_bridge_eval import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    PAIR_COUNT,
    RAW_BYTES,
    TaskspaceBridgeEvalError,
    build_bridge_receipt,
    file_identity,
    read_json_evidence,
    validate_terminal_g22,
)
from tools.measure_r1b_boundary_generator_n600 import _score_raw_cpu  # noqa: E402

REPO_ROOT = _BOOTSTRAP_ROOT
RESEARCH_ROOT = REPO_ROOT / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
DEFAULT_G22 = (
    RESEARCH_ROOT
    / "g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json"
)
DEFAULT_ARCHIVE = (
    RESEARCH_ROOT
    / "ep725_lossless_xcodec_recode_20260726/ep725_lossless_xcodec_recode.not_a_candidate.zip"
)
DEFAULT_POINTER = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
DEFAULT_RUN_ROOT = (
    Path("/Volumes/VertigoDataTier/pact")
    / "g28_ep725_xcodec_full_n600_bridge_eval_20260726"
)
DEFAULT_RECEIPT = RESEARCH_ROOT / "g28_ep725_xcodec_full_n600_bridge_eval_20260726.json"
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)
SCHEMA_MANIFEST = "tac.g28_ep725_bridge_manifest.v1"
SCHEMA_CHECKPOINT = "tac.g28_ep725_bridge_decode_checkpoint.v1"
SCHEMA_SCORER = "tac.g28_ep725_bridge_scorer_stage.v1"
SCHEMA_CLEANUP = "tac.g28_ep725_bridge_cleanup_certificate.v1"
SCHEMA_CLEANUP_COMPLETE = "tac.g28_ep725_bridge_cleanup_complete.v1"


class G28Error(TaskspaceBridgeEvalError):
    """Fail-closed execution, custody, or resumability error."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_range(path: Path, offset: int, length: int) -> str:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise G28Error(f"range source is not a regular file: {path}")
    if offset < 0 or length < 0 or offset + length > metadata.st_size:
        raise G28Error("range exceeds raw witness")
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        cursor = offset
        remaining = length
        while remaining:
            chunk = os.pread(fd, min(1 << 20, remaining), cursor)
            if not chunk:
                raise G28Error(f"short range read from {path}")
            digest.update(chunk)
            cursor += len(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _sha256_file(path: Path, expected_bytes: int | None = None) -> str:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise G28Error(f"file size drift: {path}")
    return _sha256_range(path, 0, path.stat().st_size)


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise G28Error(f"immutable artifact differs on resume: {path}")
        return
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with partial.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(partial, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise G28Error(f"immutable artifact race differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        partial.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_once(path, _canonical_json(value) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    evidence = read_json_evidence(path)
    return dict(evidence.value)


def _file_row(path: Path) -> dict[str, Any]:
    return file_identity(path)


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    base = root.resolve(strict=False)
    return resolved == base or base in resolved.parents


def _storage_preflight(run_root: Path, reserve_bytes: int) -> dict[str, Any]:
    resolved = run_root.expanduser().resolve(strict=False)
    if not any(_is_under(resolved, root) for root in SSD_ROOTS):
        raise G28Error("run root must be under the canonical SSD waterfall")
    resolved.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(resolved).free
    raw_missing = not (resolved / "scratch/0.raw").exists()
    required_new = RAW_BYTES if raw_missing else 0
    if free_bytes < required_new + reserve_bytes:
        raise G28Error(
            f"storage preflight refused: {free_bytes} free < {required_new + reserve_bytes} required"
        )
    return {
        "schema": "tac.g28_storage_preflight.v1",
        "tier": str(resolved),
        "required_new_allocation_bytes": required_new,
        "reserve_bytes": reserve_bytes,
        "free_bytes_at_preflight": free_bytes,
        "passed": True,
    }


def _preallocate(path: Path, expected_bytes: int) -> None:
    if path.exists():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise G28Error(f"partial or wrong-sized raw preallocation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, expected_bytes)
        else:
            zero = b"\0" * (8 * 1024 * 1024)
            cursor = 0
            while cursor < expected_bytes:
                block = zero[: min(len(zero), expected_bytes - cursor)]
                written = os.pwrite(fd, block, cursor)
                if written != len(block):
                    raise G28Error("short physical preallocation write")
                cursor += written
        os.fsync(fd)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


def _extract_single_member(archive: Path) -> tuple[bytes, dict[str, Any]]:
    archive_raw = archive.read_bytes()
    try:
        with zipfile.ZipFile(archive) as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) != 1 or infos[0].filename != "0.bin":
                raise G28Error("scored archive must contain exactly one 0.bin member")
            member = zf.read(infos[0])
            info = infos[0]
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise G28Error("scored archive is not a valid single-member ZIP") from exc
    return member, {
        "archive": {
            "path": str(archive.resolve()),
            "bytes": len(archive_raw),
            "sha256": _sha256_bytes(archive_raw),
        },
        "member": {
            "name": info.filename,
            "bytes": len(member),
            "sha256": _sha256_bytes(member),
            "compressed_bytes": info.compress_size,
            "compress_type": info.compress_type,
        },
    }


def _proof_contains_archive(value: Any, *, archive_bytes: int, archive_sha256: str) -> bool:
    if isinstance(value, Mapping):
        hashes = [value.get("sha256"), value.get("archive_sha256")]
        sizes = [value.get("bytes"), value.get("archive_bytes")]
        if archive_sha256 in hashes and archive_bytes in sizes:
            return True
        return any(
            _proof_contains_archive(child, archive_bytes=archive_bytes, archive_sha256=archive_sha256)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(
            _proof_contains_archive(child, archive_bytes=archive_bytes, archive_sha256=archive_sha256)
            for child in value
        )
    return False


def _partition(values: Sequence[int], workers: int) -> list[list[int]]:
    count = max(1, min(workers, len(values)))
    quotient, remainder = divmod(len(values), count)
    rows: list[list[int]] = []
    cursor = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        rows.append(list(values[cursor : cursor + width]))
        cursor += width
    return rows


def _worker_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    env["INFLATE_FP32"] = "0"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _assert_no_upstream_bytecode(upstream: Path, *, stage: str) -> dict[str, Any]:
    """Fail closed rather than silently changing the recursive auth-eval tree."""

    resolved = upstream.resolve(strict=True)
    offenders = sorted(
        str(path.relative_to(resolved))
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix in {".pyc", ".pyo"}
    )
    if offenders:
        preview = offenders[:8]
        raise G28Error(f"upstream bytecode contamination at {stage}: {preview}")
    return {
        "schema": "tac.g28_upstream_bytecode_guard.v1",
        "stage": stage,
        "upstream": str(resolved),
        "bytecode_file_count": 0,
        "python_dont_write_bytecode": sys.dont_write_bytecode,
        "passed": True,
    }


def _run_worker(worker_tool: Path, plan: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(worker_tool), "--internal-plan", str(plan)],
        cwd=REPO_ROOT,
        env=_worker_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise G28Error(
            f"receiver worker failed: stdout={completed.stdout[-2000:]!r} "
            f"stderr={completed.stderr[-4000:]!r}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise G28Error("receiver worker emitted no valid receipt") from exc
    return {
        "payload": payload,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
        "returncode": completed.returncode,
    }


def _inspect_receiver(
    *,
    run_root: Path,
    worker_tool: Path,
    worker_sha256: str,
    runtime: Path,
    runtime_sha256: str,
    member: Path,
    member_sha256: str,
) -> dict[str, Any]:
    plan_path = run_root / "plans/inspect.json"
    plan = {
        "schema": "tac.g22_ep725_xcodec_internal_plan.v1",
        "action": "inspect",
        "tool_sha256": worker_sha256,
        "runtime_path": str(runtime),
        "runtime_sha256": runtime_sha256,
        "member_path": str(member),
        "member_sha256": member_sha256,
    }
    _write_once_json(plan_path, plan)
    payload = _run_worker(worker_tool, plan_path)["payload"]
    expected = {
        "n_pairs": PAIR_COUNT,
        "camera_h": CAMERA_H,
        "camera_w": CAMERA_W,
        "dtype": "uint8",
        "channel_order": "RGB",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": payload.get(key)}
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if payload.get("frame_bytes") != CAMERA_H * CAMERA_W * 3:
        mismatches["frame_bytes"] = payload.get("frame_bytes")
    if mismatches:
        raise G28Error(f"receiver metadata mismatch: {mismatches}")
    return payload


def _chunk_rows(chunk_pairs: int, frame_bytes: int) -> list[dict[str, Any]]:
    if chunk_pairs <= 0:
        raise G28Error("chunk_pairs must be positive")
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, PAIR_COUNT, chunk_pairs)):
        stop = min(PAIR_COUNT, start + chunk_pairs)
        pair_ids = list(range(start, stop))
        rows.append(
            {
                "index": index,
                "pair_ids": pair_ids,
                "byte_offset": start * 2 * frame_bytes,
                "byte_length": len(pair_ids) * 2 * frame_bytes,
            }
        )
    return rows


def _validate_checkpoint(
    path: Path,
    *,
    row: Mapping[str, Any],
    manifest_sha256: str,
    raw_path: Path,
) -> dict[str, Any]:
    value = _read_json(path)
    expected = {
        "schema": SCHEMA_CHECKPOINT,
        "manifest_sha256": manifest_sha256,
        "index": row["index"],
        "pair_ids": row["pair_ids"],
        "byte_offset": row["byte_offset"],
        "byte_length": row["byte_length"],
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    actual_sha = _sha256_range(raw_path, row["byte_offset"], row["byte_length"])
    if value.get("range_sha256") != actual_sha:
        mismatches["range_sha256"] = {"expected": value.get("range_sha256"), "actual": actual_sha}
    if mismatches:
        raise G28Error(f"decode checkpoint drift: {path}: {mismatches}")
    return value


def _render_chunks(
    *,
    run_root: Path,
    raw_path: Path,
    rows: Sequence[Mapping[str, Any]],
    workers: int,
    worker_tool: Path,
    worker_sha256: str,
    runtime: Path,
    runtime_sha256: str,
    member: Path,
    member_sha256: str,
    receiver_metadata: Mapping[str, Any],
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for row in rows:
        checkpoint = run_root / "checkpoints" / f"chunk-{row['index']:04d}.json"
        if checkpoint.exists():
            completed.append(
                _validate_checkpoint(
                    checkpoint,
                    row=row,
                    manifest_sha256=manifest_sha256,
                    raw_path=raw_path,
                )
            )
            continue
        started = time.monotonic()
        plans: list[Path] = []
        for worker_index, pair_ids in enumerate(_partition(row["pair_ids"], workers)):
            plan_path = run_root / "plans" / f"chunk-{row['index']:04d}.worker-{worker_index:02d}.json"
            plan = {
                "schema": "tac.g22_ep725_xcodec_internal_plan.v1",
                "action": "render",
                "tool_sha256": worker_sha256,
                "runtime_path": str(runtime),
                "runtime_sha256": runtime_sha256,
                "member_path": str(member),
                "member_sha256": member_sha256,
                "output_path": str(raw_path),
                "output_bytes": RAW_BYTES,
                "receiver_metadata": dict(receiver_metadata),
                "pair_ids": pair_ids,
            }
            _write_once_json(plan_path, plan)
            plans.append(plan_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(plans)) as executor:
            worker_rows = list(executor.map(lambda plan: _run_worker(worker_tool, plan), plans))
        value = {
            "schema": SCHEMA_CHECKPOINT,
            "manifest_sha256": manifest_sha256,
            "index": row["index"],
            "pair_ids": row["pair_ids"],
            "byte_offset": row["byte_offset"],
            "byte_length": row["byte_length"],
            "range_sha256": _sha256_range(raw_path, row["byte_offset"], row["byte_length"]),
            "workers": worker_rows,
            "wall_seconds": time.monotonic() - started,
        }
        _write_once_json(checkpoint, value)
        completed.append(
            _validate_checkpoint(
                checkpoint,
                row=row,
                manifest_sha256=manifest_sha256,
                raw_path=raw_path,
            )
        )
    return completed


def _scorer_hashes(upstream: Path) -> dict[str, dict[str, Any]]:
    return {
        label: _file_row(upstream / relative)
        for label, relative in (
            ("modules", "modules.py"),
            ("frame_utils", "frame_utils.py"),
            ("posenet", "models/posenet.safetensors"),
            ("segnet", "models/segnet.safetensors"),
            ("source_video", "videos/0.mkv"),
        )
    }


def _score_stage(
    *,
    path: Path,
    raw_path: Path,
    raw_sha256: str,
    archive_row: Mapping[str, Any],
    runtime_row: Mapping[str, Any],
    upstream: Path,
    batch_size: int,
    cpu_threads: int,
    seed: int,
) -> dict[str, Any]:
    scorer_files = _scorer_hashes(upstream)
    scorer_impl = _file_row(REPO_ROOT / "tools/measure_r1b_boundary_generator_n600.py")
    stable = {
        "raw_sha256": raw_sha256,
        "archive": dict(archive_row),
        "runtime": dict(runtime_row),
        "scorer_files": scorer_files,
        "scorer_implementation": scorer_impl,
        "batch_size": batch_size,
        "cpu_threads": cpu_threads,
        "seed": seed,
    }
    if path.exists():
        value = _read_json(path)
        if value.get("schema") != SCHEMA_SCORER or value.get("stable_contract") != stable:
            raise G28Error("scorer stage checkpoint contract drift")
        return value
    started = time.monotonic()
    scorer_row = _score_raw_cpu(
        raw_path=raw_path,
        upstream=upstream,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
        seed=seed,
    )
    scorer_row["raw_sha256"] = raw_sha256
    value = {
        "schema": SCHEMA_SCORER,
        "stable_contract": stable,
        "scorer_row": scorer_row,
        "wall_seconds": time.monotonic() - started,
    }
    _write_once_json(path, value)
    return _read_json(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _recompute_receipt_identity(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("receipt_identity_sha256", None)
    value["receipt_identity_sha256"] = _sha256_bytes(_canonical_json(value))
    return value


def _finalize_cleanup(
    *,
    run_root: Path,
    receipt_path: Path,
    raw_path: Path,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    precleanup_path = run_root / "receipts/precleanup.json"
    certificate_path = run_root / "cleanup/raw_cleanup_certificate.json"
    completion_path = run_root / "cleanup/raw_cleanup_complete.json"
    if not precleanup_path.is_file():
        raise G28Error("precleanup scientific receipt is missing")
    precleanup = _read_json(precleanup_path)
    if certificate_path.exists():
        certificate = _read_json(certificate_path)
        if certificate.get("schema") != SCHEMA_CLEANUP:
            raise G28Error("cleanup certificate schema drift")
    else:
        if not raw_path.is_file() or raw_path.stat().st_size != RAW_BYTES:
            raise G28Error("raw witness missing before cleanup certification")
        immediate_sha = _sha256_file(raw_path, RAW_BYTES)
        if immediate_sha != expected_raw_sha256:
            raise G28Error("raw witness drifted immediately before cleanup certification")
        certificate = {
            "schema": SCHEMA_CLEANUP,
            "raw": {
                "path": str(raw_path),
                "bytes": RAW_BYTES,
                "sha256": immediate_sha,
            },
            "precleanup_receipt": _file_row(precleanup_path),
            "rebuild_argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "rebuildable": True,
            "cleanup_reason": "success-only full-n600 scorer raw scratch",
            "delete_authorized_after_rehash": True,
        }
        _write_once_json(certificate_path, certificate)
    if raw_path.exists():
        immediate_sha = _sha256_file(raw_path, RAW_BYTES)
        if immediate_sha != expected_raw_sha256:
            raise G28Error("raw witness drifted immediately before unlink")
        raw_path.unlink()
        directory_fd = os.open(raw_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    if not completion_path.exists():
        completion = {
            "schema": SCHEMA_CLEANUP_COMPLETE,
            "certificate": _file_row(certificate_path),
            "raw_path": str(raw_path),
            "raw_absent": not raw_path.exists(),
            "checkpoints_preserved": True,
        }
        if completion["raw_absent"] is not True:
            raise G28Error("raw witness remains after certified cleanup")
        _write_once_json(completion_path, completion)
    completion = _read_json(completion_path)
    if completion.get("raw_absent") is not True or raw_path.exists():
        raise G28Error("cleanup completion does not prove raw absence")

    final = dict(precleanup)
    run_custody = dict(final.get("run_custody", {}))
    run_custody["cleanup"] = {
        "completed": True,
        "success_only": True,
        "raw_absent": True,
        "certificate": _file_row(certificate_path),
        "completion": _file_row(completion_path),
        "decode_checkpoints_preserved": True,
        "scorer_stage_preserved": True,
    }
    final["run_custody"] = run_custody
    final = _recompute_receipt_identity(final)
    _write_once_json(receipt_path, final)
    return _read_json(receipt_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g22-receipt", type=Path, default=DEFAULT_G22)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--archive-proof",
        type=Path,
        help="required when --archive differs from G22 selected; must contain exact archive identity",
    )
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--resume-from", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--chunk-pairs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reserve-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--confirm-full-n600", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute_reviewed or not args.confirm_full_n600:
        raise G28Error("full n600 execution requires --execute-reviewed --confirm-full-n600")
    if args.workers <= 0 or args.cpu_threads <= 0 or args.batch_size != 16:
        raise G28Error("workers/cpu_threads must be positive and batch_size must be canonical 16")

    upstream = args.upstream.expanduser().resolve(strict=True)
    bytecode_preflight = _assert_no_upstream_bytecode(upstream, stage="preflight")
    run_root = args.resume_from.expanduser().resolve(strict=False)
    receipt_path = args.receipt.expanduser().resolve(strict=False)
    if receipt_path.exists():
        final = _read_json(receipt_path)
        if final.get("schema") != "tac.taskspace_ep725_same_object_bridge_eval.v1":
            raise G28Error("existing final receipt schema drift")
        print(json.dumps({"receipt": str(receipt_path), "verdict": final.get("verdict")}, sort_keys=True))
        return 0

    precleanup_path = run_root / "receipts/precleanup.json"
    if precleanup_path.exists():
        precleanup = _read_json(precleanup_path)
        expected_raw_sha256 = precleanup["artifact_identity"]["selected_raw_sha256"]
        final = _finalize_cleanup(
            run_root=run_root,
            receipt_path=receipt_path,
            raw_path=run_root / "scratch/0.raw",
            expected_raw_sha256=expected_raw_sha256,
        )
        print(json.dumps({"receipt": str(receipt_path), "verdict": final["verdict"]}, sort_keys=True))
        return 0

    g22 = read_json_evidence(args.g22_receipt.expanduser().resolve(strict=True))
    g22_closure = validate_terminal_g22(g22)
    pointer_start = read_json_evidence(args.pointer.expanduser().resolve(strict=True))
    archive = args.archive.expanduser().resolve(strict=True)
    archive_row = _file_row(archive)
    g22_archive = g22_closure["selected_archive"]
    archive_proof: dict[str, Any] | None = None
    if archive_row != g22_archive:
        if args.archive_proof is None:
            raise G28Error("non-G22 archive requires --archive-proof")
        proof_evidence = read_json_evidence(args.archive_proof.expanduser().resolve(strict=True))
        if not _proof_contains_archive(
            proof_evidence.value,
            archive_bytes=archive_row["bytes"],
            archive_sha256=archive_row["sha256"],
        ):
            raise G28Error("archive proof does not contain exact scored archive identity")
        archive_proof = {
            "path": proof_evidence.path,
            "bytes": proof_evidence.bytes,
            "sha256": proof_evidence.sha256,
        }

    current_preflight = _storage_preflight(run_root, args.reserve_bytes)
    preflight_path = run_root / "storage_preflight.json"
    if not preflight_path.exists():
        _write_once_json(preflight_path, current_preflight)
    launch_preflight = _read_json(preflight_path)
    if (
        launch_preflight.get("schema") != "tac.g28_storage_preflight.v1"
        or launch_preflight.get("tier") != str(run_root)
        or launch_preflight.get("reserve_bytes") != args.reserve_bytes
        or launch_preflight.get("passed") is not True
    ):
        raise G28Error("preserved storage preflight contract drift")
    member_raw, reopened = _extract_single_member(archive)
    if reopened["archive"] != archive_row:
        raise G28Error("archive changed while reopening member")

    tool_row = _file_row(Path(__file__).resolve())
    worker_row = _file_row(Path(g22.value["reproducibility"]["tool"]["path"]))
    if worker_row["sha256"] != g22.value["reproducibility"]["tool"]["sha256"]:
        raise G28Error("G22 worker tool drifted from terminal receipt")
    runtime_source = Path(g22_closure["runtime"]["path"])
    runtime_raw = runtime_source.read_bytes()
    runtime_path = run_root / "inputs/frozen_inflate.py"
    member_path = run_root / "inputs/scored.0.bin"
    _write_once(runtime_path, runtime_raw)
    _write_once(member_path, member_raw)
    runtime_row = _file_row(runtime_path)
    member_row = _file_row(member_path)
    if runtime_row["sha256"] != g22_closure["runtime"]["sha256"]:
        raise G28Error("custody runtime differs from G22")
    if member_row["sha256"] != reopened["member"]["sha256"]:
        raise G28Error("custody member differs from scored archive")

    receiver_metadata = _inspect_receiver(
        run_root=run_root,
        worker_tool=Path(worker_row["path"]),
        worker_sha256=worker_row["sha256"],
        runtime=runtime_path,
        runtime_sha256=runtime_row["sha256"],
        member=member_path,
        member_sha256=member_row["sha256"],
    )
    rows = _chunk_rows(args.chunk_pairs, int(receiver_metadata["frame_bytes"]))
    stable_contract = {
        "schema": SCHEMA_MANIFEST,
        "lane_id": "lane_g28_ep725_xcodec_full_n600_bridge_eval_20260726",
        "g22": g22_closure,
        "scored_archive": archive_row,
        "archive_proof": archive_proof,
        "reopened_member": reopened["member"],
        "runtime": runtime_row,
        "worker_tool": worker_row,
        "g28_tool": tool_row,
        "receiver_metadata": receiver_metadata,
        "pair_ids": list(range(PAIR_COUNT)),
        "raw_bytes": RAW_BYTES,
        "chunk_pairs": args.chunk_pairs,
        "chunk_count": len(rows),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        "storage_preflight": _file_row(preflight_path),
        "storage_requirements": {
            "raw_bytes": RAW_BYTES,
            "reserve_bytes": args.reserve_bytes,
            "ssd_root_required": True,
        },
        "upstream_bytecode_preflight": bytecode_preflight,
        "authority": {
            "axis": "[macOS-CPU frozen-scorer advisory]",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
    }
    manifest_path = run_root / "manifest.json"
    _write_once_json(manifest_path, stable_contract)
    manifest_row = _file_row(manifest_path)

    raw_path = run_root / "scratch/0.raw"
    scorer_path = run_root / "checkpoints/scorer.json"
    if precleanup_path.exists():
        precleanup = _read_json(precleanup_path)
        expected_raw_sha256 = precleanup["artifact_identity"]["selected_raw_sha256"]
        final = _finalize_cleanup(
            run_root=run_root,
            receipt_path=receipt_path,
            raw_path=raw_path,
            expected_raw_sha256=expected_raw_sha256,
        )
        print(json.dumps({"receipt": str(receipt_path), "verdict": final["verdict"]}, sort_keys=True))
        return 0

    _preallocate(raw_path, RAW_BYTES)
    checkpoints = _render_chunks(
        run_root=run_root,
        raw_path=raw_path,
        rows=rows,
        workers=args.workers,
        worker_tool=Path(worker_row["path"]),
        worker_sha256=worker_row["sha256"],
        runtime=runtime_path,
        runtime_sha256=runtime_row["sha256"],
        member=member_path,
        member_sha256=member_row["sha256"],
        receiver_metadata=receiver_metadata,
        manifest_sha256=manifest_row["sha256"],
    )
    raw_sha256 = _sha256_file(raw_path, RAW_BYTES)
    if raw_sha256 != g22_closure["selected_raw_sha256"]:
        raise G28Error("scored archive output differs from terminal G22 reference witness")

    scorer_stage = _score_stage(
        path=scorer_path,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        archive_row=archive_row,
        runtime_row=runtime_row,
        upstream=upstream,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
        seed=args.seed,
    )
    scorer_evidence = read_json_evidence(scorer_path)
    bytecode_postscore = _assert_no_upstream_bytecode(upstream, stage="postscore")
    run_custody = {
        "run_root": str(run_root),
        "manifest": manifest_row,
        "decode_checkpoints": [_file_row(run_root / "checkpoints" / f"chunk-{row['index']:04d}.json") for row in rows],
        "decode_checkpoint_count": len(checkpoints),
        "raw_before_cleanup": {
            "path": str(raw_path),
            "bytes": RAW_BYTES,
            "sha256": raw_sha256,
        },
        "scorer_stage": _file_row(scorer_path),
        "git_head": _git_head(),
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "storage_preflight": {
            "launch": _file_row(preflight_path),
            "launch_observation": launch_preflight,
            "current_resume_observation": current_preflight,
        },
        "pointer_observations": {
            "start": {
                "path": pointer_start.path,
                "bytes": pointer_start.bytes,
                "sha256": pointer_start.sha256,
            },
        },
        "upstream_bytecode_guard": {
            "preflight": bytecode_preflight,
            "postscore": bytecode_postscore,
        },
        "cleanup": {"completed": False, "policy": "certify, rehash, then success-only unlink"},
    }
    pointer_end = read_json_evidence(args.pointer.expanduser().resolve(strict=True))
    run_custody["pointer_observations"]["end"] = {
        "path": pointer_end.path,
        "bytes": pointer_end.bytes,
        "sha256": pointer_end.sha256,
    }
    receipt = build_bridge_receipt(
        g22=g22,
        pointer=pointer_end,
        scorer_row=scorer_stage["scorer_row"],
        scorer_evidence=scorer_evidence,
        run_custody=run_custody,
        scored_archive=archive_row,
    )
    _write_once_json(precleanup_path, receipt)
    final = _finalize_cleanup(
        run_root=run_root,
        receipt_path=receipt_path,
        raw_path=raw_path,
        expected_raw_sha256=raw_sha256,
    )
    print(json.dumps({"receipt": str(receipt_path), "verdict": final["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
