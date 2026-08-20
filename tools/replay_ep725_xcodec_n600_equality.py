#!/usr/bin/env python3
"""Resumable exact receiver replay for the frozen ep725 lossless-xcodec rewrite.

This is deliberately a research-only equality harness.  It reopens the exact
G20 ArchiveArtifact bytes, executes the frozen private ``_setup`` /
``_render_pair`` receiver path, and produces an append-only DecodeReceipt.  It
does not run the scorer, move the frontier pointer, or create a candidate.

The output lifecycle is explicit:

    counted ArchiveArtifact -> free generic decoder -> DecodeReceipt -> witness

The two large raw witnesses are scratch.  They are removed only after an
immutable, machine-readable cleanup certificate proves that every byte is
deterministically rebuildable from the preserved contract and chunk receipts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
import platform
import resource
import shlex
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.canonical_frontier_pointer import (
    POINTER_SCHEMA_VERSION,
    CanonicalFrontierPointer,
    recompute_effective_frontier,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
)
G17_SPEC = RESEARCH_ROOT / "SPEC_g17_unified_production_envelope_20260726.md"
G20_SPEC = RESEARCH_ROOT / "SPEC_g20_ep725_lossless_xcodec_recode_20260726.md"
G20_ROOT = RESEARCH_ROOT / "ep725_lossless_xcodec_recode_20260726"
G20_RECEIPT = G20_ROOT / "receipt.json"
G20_ARCHIVE = G20_ROOT / "ep725_lossless_xcodec_recode.not_a_candidate.zip"
G20_MODULE = REPO_ROOT / "src/tac/witness_dsl/ep725_lossless_xcodec_recode.py"
G20_TOOL = REPO_ROOT / "tools/materialize_ep725_lossless_xcodec_recode.py"
FRONTIER_POINTER = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"
FRONTIER_MODULE = REPO_ROOT / "src/tac/canonical_frontier_pointer.py"

FROZEN = {
    "g17_spec_sha256": "f315c8c0ad3708394e96cbbf40de9bb6af7d6072989bb28ea38a226f5354953b",
    "g20_spec_sha256": "5388f47daaa0b9dfa7510c12ae56a73f704375068fc1d9cf29410fa746b1d5ca",
    "g20_receipt_sha256": "02ccb8a6209c79651b64fa93b15aa1ed6155b03d9709f5f18b4ff98edfe25c8c",
    "g20_archive_sha256": "8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8",
    "g20_archive_bytes": 81027,
    "g20_member_sha256": "4789bf6b5f15272cc5f8a573f25137a9daf7e21755e81aa48a8fba84947b5634",
    "g20_member_bytes": 81738,
    "source_archive_sha256": "149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3",
    "source_archive_bytes": 83838,
    "source_member_sha256": "f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c",
    "source_member_bytes": 84536,
    "runtime_sha256": "4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224",
    "runtime_bytes": 56814,
    "g20_module_sha256": "7a54d13fc1fc98916997b655007ef7c5e66085f1cba3bfa3d0de28978c1b45de",
    "g20_tool_sha256": "b1b31baeef79f662ae4108379282f3a5fe8ebb76c380996aaf68506d47b16e86",
    "frontier_module_sha256": "502b2c77d37bce0767fb3e764aad57ac154b5bafd8bc5ca5ec12a3eb690c1994",
    "state_sha256": "5485d0d94c5c834e059837e74ae5320fe9d2b526604c47008a6bfdb74144adf6",
    "bounded_pair0_raw_sha256": "22b994567d3db018df29a95c597606053a115c631d98e89b72ec7eeba93666b3",
}

SCHEMA_MANIFEST = "tac.g22_ep725_xcodec_replay_manifest.v1"
SCHEMA_CHUNK = "tac.g22_ep725_xcodec_chunk_checkpoint.v1"
SCHEMA_PRECLEANUP = "tac.g22_ep725_xcodec_decode_receipt_precleanup.v1"
SCHEMA_CLEANUP = "tac.g22_ep725_xcodec_cleanup_certificate.v1"
SCHEMA_CLEANUP_COMPLETE = "tac.g22_ep725_xcodec_cleanup_complete.v1"
SCHEMA_FINAL = "tac.g22_ep725_xcodec_decode_receipt.v1"


class ReplayError(RuntimeError):
    """Fail-closed G22 replay error."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_read(path: Path) -> bytes:
    """Read one regular non-link file while binding path and descriptor identity."""

    path = path.absolute()
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ReplayError(f"required file is absent: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ReplayError(f"required path is not a real regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ReplayError(f"path identity changed before open: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after_fd = (
        after_fd.st_dev,
        after_fd.st_ino,
        after_fd.st_size,
        after_fd.st_mtime_ns,
    )
    identity_after_path = (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    )
    if identity_before != identity_after_fd or identity_before != identity_after_path:
        raise ReplayError(f"file drifted while being read: {path}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise ReplayError(f"short stable read: {path}")
    return raw


def _file_row(path: Path, *, expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    raw = _stable_read(path)
    digest = _sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ReplayError(f"SHA-256 drift for {path}: {digest} != {expected_sha256}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise ReplayError(f"byte-size drift for {path}: {len(raw)} != {expected_bytes}")
    return {"path": str(path.absolute()), "bytes": len(raw), "sha256": digest}


def _write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _stable_read(path) != raw:
            raise ReplayError(f"append-only artifact differs: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(partial, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ReplayError(f"short write: {partial}")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.exists():
        partial.unlink()
        if _stable_read(path) != raw:
            raise ReplayError(f"append-only write race differs: {path}")
        return
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_once(path, _canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_stable_read(path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplayError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root must be an object: {path}")
    return value


def _competitive_target_identity(pointer_raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the semantic target from constituents, never the cached row."""

    try:
        decoded = json.loads(pointer_raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReplayError("canonical frontier pointer JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ReplayError("canonical frontier pointer root must be an object")
    try:
        pointer = CanonicalFrontierPointer.from_dict(decoded)
        selected = recompute_effective_frontier(pointer)
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayError("cannot recompute competitive target from pointer constituents") from exc
    if pointer.schema_version != POINTER_SCHEMA_VERSION or not isinstance(selected, dict):
        raise ReplayError("canonical pointer has no supported recomputable competitive target")
    identity = {
        "schema": "tac.competitive_target_identity.v1",
        "target_score": selected.get("score"),
        "selected_axis": selected.get("axis"),
        "selected_source": selected.get("source"),
        "selected_source_kind": selected.get("source_kind"),
        "selected_score_precision": selected.get("score_precision"),
        "selected_custody": selected.get("custody"),
        "selected_evidence_grade": selected.get("evidence_grade"),
        "selection_rule": selected.get("selection_rule"),
        "selected_archive_sha256": selected.get("archive_sha256"),
        "selected_lane_id": selected.get("lane_id"),
        "selected_hardware_substrate": selected.get("hardware_substrate"),
        "selected_submission_name": selected.get("submission_name"),
        "selected_pr_number": selected.get("pr_number"),
        "selected_pr_url": selected.get("pr_url"),
        "selected_leaderboard_rank": selected.get("leaderboard_rank"),
    }
    return identity, {
        "last_refreshed_utc": decoded.get("last_refreshed_utc"),
        "source_snapshot_at_utc": selected.get("snapshot_at_utc"),
        "serialized_effective_frontier_matches_recomputed": decoded.get("effective_frontier") == selected,
    }


def _preserve_pointer_observation(run_root: Path, pointer_raw: bytes) -> dict[str, Any]:
    digest = _sha256_bytes(pointer_raw)
    preserved_path = run_root / "pointer_observations" / f"pointer.{digest}.json"
    _write_once(preserved_path, pointer_raw)
    metadata = FRONTIER_POINTER.lstat()
    target, provenance = _competitive_target_identity(pointer_raw)
    return {
        "schema": "tac.pointer_artifact_observation.v1",
        "artifact": {
            "source_path": str(FRONTIER_POINTER),
            "bytes": len(pointer_raw),
            "sha256": digest,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mtime_ns": metadata.st_mtime_ns,
            "preserved_bytes": _file_row(preserved_path),
        },
        "competitive_target": target,
        "provenance": provenance,
    }


def _observe_live_pointer(run_root: Path) -> dict[str, Any]:
    return _preserve_pointer_observation(run_root, _stable_read(FRONTIER_POINTER))


def _validate_pointer_observation(observation: Mapping[str, Any]) -> None:
    artifact = observation.get("artifact", {})
    preserved = artifact.get("preserved_bytes", {})
    _validate_preserved_row(preserved, "historical pointer observation")
    raw = _stable_read(Path(preserved["path"]))
    target, provenance = _competitive_target_identity(raw)
    if (
        artifact.get("bytes") != len(raw)
        or artifact.get("sha256") != _sha256_bytes(raw)
        or observation.get("competitive_target") != target
        or observation.get("provenance") != provenance
    ):
        raise ReplayError("historical pointer observation drifted")


def _pointer_change(start: Mapping[str, Any], end: Mapping[str, Any]) -> dict[str, Any]:
    artifact_changed = start.get("artifact", {}).get("sha256") != end.get("artifact", {}).get("sha256")
    target_changed = start.get("competitive_target") != end.get("competitive_target")
    return {
        "pointer_artifact_changed": artifact_changed,
        "competitive_target_changed": target_changed,
        "rebase_required_before_admission": target_changed,
        "decode_equality_invalidated": False,
    }


def _aggregate_pointer_change(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observations = [end] + [row["pointer_observation_after_chunk"] for row in checkpoints]
    changes = [_pointer_change(start, observation) for observation in observations]
    return {
        "pointer_artifact_changed": any(row["pointer_artifact_changed"] for row in changes),
        "competitive_target_changed": any(row["competitive_target_changed"] for row in changes),
        "rebase_required_before_admission": any(
            row["rebase_required_before_admission"] for row in changes
        ),
        "decode_equality_invalidated": False,
    }


def _extract_single_member(archive_raw: bytes, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_raw), "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir():
                raise ReplayError(f"{label} archive must contain exactly one regular 0.bin")
            info = infos[0]
            if info.flag_bits & 0x1:
                raise ReplayError(f"{label} member may not be encrypted")
            member = archive.read(info)
    except ReplayError:
        raise
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ReplayError(f"cannot reopen exact {label} archive bytes") from exc
    return member, {
        "name": "0.bin",
        "bytes": len(member),
        "sha256": _sha256_bytes(member),
        "zip_crc32": f"{info.CRC:08x}",
        "compress_type": info.compress_type,
        "compressed_bytes": info.compress_size,
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _is_ssd_tier(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    allowed = (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    )
    return any(resolved == root or root in resolved.parents for root in allowed)


def _validate_run_root(path: Path, *, allow_non_ssd: bool) -> Path:
    absolute = path.absolute()
    if absolute.exists() and (absolute.is_symlink() or not absolute.is_dir()):
        raise ReplayError("--resume-from must name one real directory")
    if not allow_non_ssd and not _is_ssd_tier(absolute):
        raise ReplayError("--resume-from must live on the configured SSD tier")
    absolute.mkdir(parents=True, exist_ok=True)
    if absolute.is_symlink():
        raise ReplayError("--resume-from may not be a symlink")
    return absolute


def _assert_no_partial_files(root: Path) -> None:
    partials = sorted(
        str(path) for path in root.rglob("*") if ".partial" in path.name or path.name.endswith(".tmp")
    )
    if partials:
        raise ReplayError(f"partial files require operator review: {partials[:5]}")


def _rusage_snapshot() -> dict[str, Any]:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_user_seconds": self_usage.ru_utime,
        "self_system_seconds": self_usage.ru_stime,
        "self_maxrss_native_units": self_usage.ru_maxrss,
        "children_user_seconds": child_usage.ru_utime,
        "children_system_seconds": child_usage.ru_stime,
        "children_maxrss_native_units": child_usage.ru_maxrss,
        "platform": platform.platform(),
    }


def _resource_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: after[key] - before[key]
        for key in (
            "self_user_seconds",
            "self_system_seconds",
            "children_user_seconds",
            "children_system_seconds",
        )
    } | {
        "self_maxrss_native_units": after["self_maxrss_native_units"],
        "children_maxrss_native_units": after["children_maxrss_native_units"],
        "platform": after["platform"],
    }


def _sha256_range(path: Path, offset: int, length: int) -> str:
    if offset < 0 or length < 0:
        raise ReplayError("negative range")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or offset + length > metadata.st_size:
            raise ReplayError(f"range exceeds regular file: {path}")
        os.lseek(fd, offset, os.SEEK_SET)
        remaining = length
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                raise ReplayError(f"short range read: {path}")
            digest.update(block)
            remaining -= len(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _compare_range(left: Path, right: Path, offset: int, length: int) -> tuple[str, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    left_fd = os.open(left, flags)
    right_fd = os.open(right, flags)
    left_hash = hashlib.sha256()
    right_hash = hashlib.sha256()
    try:
        os.lseek(left_fd, offset, os.SEEK_SET)
        os.lseek(right_fd, offset, os.SEEK_SET)
        remaining = length
        cursor = offset
        while remaining:
            want = min(1024 * 1024, remaining)
            a = os.read(left_fd, want)
            b = os.read(right_fd, want)
            if len(a) != want or len(b) != want:
                raise ReplayError("short witness range during exact comparison")
            if a != b:
                first = next(index for index, (av, bv) in enumerate(zip(a, b, strict=True)) if av != bv)
                raise ReplayError(f"source/selected uint8 mismatch at raw byte {cursor + first}")
            left_hash.update(a)
            right_hash.update(b)
            cursor += want
            remaining -= want
    finally:
        os.close(left_fd)
        os.close(right_fd)
    return left_hash.hexdigest(), right_hash.hexdigest()


def _sha256_file_streaming(path: Path, expected_bytes: int) -> str:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
        raise ReplayError(f"witness file shape drift: {path}")
    return _sha256_range(path, 0, expected_bytes)


def _preallocate(path: Path, expected_bytes: int) -> None:
    if path.exists():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise ReplayError(f"partial or wrong-sized preallocation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(fd, 0, expected_bytes)
        else:
            # macOS does not expose posix_fallocate in Python.  A truncate-only
            # sparse file would make the storage gate fictional, so explicitly
            # write every block.  The fixed buffer keeps memory bounded even for
            # the 3.66 GB n600 witness.
            zero = b"\0" * (8 * 1024 * 1024)
            cursor = 0
            while cursor < expected_bytes:
                block = zero[: min(len(zero), expected_bytes - cursor)]
                written = os.pwrite(fd, block, cursor)
                if written != len(block):
                    raise ReplayError(f"short physical preallocation write: {path}")
                cursor += written
        os.fsync(fd)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


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


def _load_private_runtime(runtime_path: Path, module_tag: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_tag, runtime_path)
    if spec is None or spec.loader is None:
        raise ReplayError("cannot import frozen runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "_setup", None)) or not callable(getattr(module, "_render_pair", None)):
        raise ReplayError("frozen runtime private receiver path is absent")
    if getattr(module, "_FP32", True) is not False:
        raise ReplayError("frozen replay must use portable fp64, never INFLATE_FP32")
    return module


def _internal_plan(plan_path: Path) -> int:
    plan = _read_json(plan_path)
    if plan.get("schema") != "tac.g22_ep725_xcodec_internal_plan.v1":
        raise ReplayError("internal plan schema drift")
    tool_path = Path(__file__).resolve()
    _file_row(tool_path, expected_sha256=plan.get("tool_sha256"))
    runtime = Path(plan["runtime_path"])
    member = Path(plan["member_path"])
    _file_row(runtime, expected_sha256=plan["runtime_sha256"])
    _file_row(member, expected_sha256=plan["member_sha256"])
    if os.environ.get("INFLATE_FP32", "0") != "0":
        raise ReplayError("INFLATE_FP32 must be exactly 0")
    for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ReplayError(f"worker environment drift: {name}")
    module = _load_private_runtime(runtime, f"g22_frozen_{os.getpid()}_{time.time_ns()}")
    module._setup(str(member))
    metadata = {
        "n_pairs": int(module._G["m"]["n_pairs"]),
        "render_h": int(module._G["rh"]),
        "render_w": int(module._G["rw"]),
        "camera_h": int(module._G["ch"]),
        "camera_w": int(module._G["cw"]),
        "frame_bytes": int(module._G["framebytes"]),
        "dtype": "uint8",
        "channel_order": "RGB",
    }
    if plan["action"] == "inspect":
        print(json.dumps(metadata, sort_keys=True))
        return 0
    if plan["action"] != "render":
        raise ReplayError("unknown internal plan action")
    pair_ids = plan.get("pair_ids")
    if not isinstance(pair_ids, list) or not pair_ids or pair_ids != list(range(pair_ids[0], pair_ids[-1] + 1)):
        raise ReplayError("internal pair population must be a non-empty ordered contiguous range")
    expected_metadata = plan.get("receiver_metadata")
    if metadata != expected_metadata:
        raise ReplayError("receiver metadata drifted between inspection and render")
    output = Path(plan["output_path"])
    output_metadata = output.lstat()
    if output.is_symlink() or not stat.S_ISREG(output_metadata.st_mode):
        raise ReplayError("worker output is not one real preallocated file")
    if output_metadata.st_size != int(plan["output_bytes"]):
        raise ReplayError("worker output preallocation size drift")
    module._G["dst"] = str(output)
    started = time.monotonic()
    before = _rusage_snapshot()
    for pair_id in pair_ids:
        if module._render_pair(pair_id) != pair_id:
            raise ReplayError("private receiver returned a different pair id")
    fd = os.open(output, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    after = _rusage_snapshot()
    print(
        json.dumps(
            {
                "pair_ids": pair_ids,
                "wall_seconds": time.monotonic() - started,
                "resource_usage": _resource_delta(before, after),
            },
            sort_keys=True,
        )
    )
    return 0


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["INFLATE_FP32"] = "0"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_internal(plan_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--internal-plan", str(plan_path)],
        cwd=REPO_ROOT,
        env=_worker_environment(),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReplayError(
            f"frozen private receiver worker failed ({plan_path}): "
            f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-4000:]!r}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ReplayError(f"worker emitted no canonical receipt: {plan_path}") from exc
    return {
        "payload": payload,
        "returncode": completed.returncode,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
    }


def _inspect(runtime_path: Path, member_path: Path, tool_sha256: str, plan_path: Path) -> dict[str, Any]:
    plan = {
        "schema": "tac.g22_ep725_xcodec_internal_plan.v1",
        "action": "inspect",
        "tool_sha256": tool_sha256,
        "runtime_path": str(runtime_path),
        "runtime_sha256": FROZEN["runtime_sha256"],
        "member_path": str(member_path),
        "member_sha256": _sha256_file_streaming(member_path, member_path.stat().st_size),
    }
    _write_once_json(plan_path, plan)
    result = _run_internal(plan_path)["payload"]
    if not isinstance(result, dict):
        raise ReplayError("receiver inspection is malformed")
    return result


def _validate_frozen_contract() -> dict[str, Any]:
    rows = {
        "g17_spec": _file_row(G17_SPEC, expected_sha256=FROZEN["g17_spec_sha256"]),
        "g20_spec": _file_row(G20_SPEC, expected_sha256=FROZEN["g20_spec_sha256"]),
        "g20_receipt": _file_row(G20_RECEIPT, expected_sha256=FROZEN["g20_receipt_sha256"]),
        "g20_module": _file_row(G20_MODULE, expected_sha256=FROZEN["g20_module_sha256"]),
        "g20_materializer": _file_row(G20_TOOL, expected_sha256=FROZEN["g20_tool_sha256"]),
        "frontier_module": _file_row(
            FRONTIER_MODULE,
            expected_sha256=FROZEN["frontier_module_sha256"],
        ),
        "selected_archive": _file_row(
            G20_ARCHIVE,
            expected_sha256=FROZEN["g20_archive_sha256"],
            expected_bytes=FROZEN["g20_archive_bytes"],
        ),
    }
    receipt = _read_json(G20_RECEIPT)
    if receipt.get("schema") != "tac.ep725_lossless_xcodec_recode.v1":
        raise ReplayError("G20 receipt schema drift")
    source_archive = Path(receipt["source_paths"]["archive"])
    runtime = Path(receipt["source_paths"]["runtime"])
    rows["source_archive"] = _file_row(
        source_archive,
        expected_sha256=FROZEN["source_archive_sha256"],
        expected_bytes=FROZEN["source_archive_bytes"],
    )
    rows["runtime"] = _file_row(
        runtime,
        expected_sha256=FROZEN["runtime_sha256"],
        expected_bytes=FROZEN["runtime_bytes"],
    )
    required_truth = receipt.get("truth", {})
    if (
        required_truth.get("candidate_claim") is not False
        or required_truth.get("score_claim") is not False
        or required_truth.get("research_only") is not True
        or receipt.get("artifact", {}).get("classification") != "not_a_candidate"
        or receipt.get("proof", {}).get("full_quantized_state_equal") is not True
        or receipt.get("proof", {}).get("source_state_sha256") != FROZEN["state_sha256"]
        or receipt.get("proof", {}).get("selected_state_sha256") != FROZEN["state_sha256"]
    ):
        raise ReplayError("G20 truth or exact decoded-state proof drift")
    return {"rows": rows, "receipt": receipt, "source_archive": source_archive, "runtime": runtime}


def _materialize_custody_inputs(run_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    inputs = run_root / "inputs"
    runtime_raw = _stable_read(contract["runtime"])
    source_archive_raw = _stable_read(contract["source_archive"])
    selected_archive_raw = _stable_read(G20_ARCHIVE)
    if _sha256_bytes(runtime_raw) != FROZEN["runtime_sha256"]:
        raise ReplayError("runtime drifted between contract validation and custody copy")
    if _sha256_bytes(source_archive_raw) != FROZEN["source_archive_sha256"]:
        raise ReplayError("source archive drifted between validation and member extraction")
    if _sha256_bytes(selected_archive_raw) != FROZEN["g20_archive_sha256"]:
        raise ReplayError("selected archive drifted between validation and member extraction")
    source_member, source_member_row = _extract_single_member(source_archive_raw, label="source")
    selected_member, selected_member_row = _extract_single_member(selected_archive_raw, label="selected")
    if (
        source_member_row["sha256"] != FROZEN["source_member_sha256"]
        or source_member_row["bytes"] != FROZEN["source_member_bytes"]
        or selected_member_row["sha256"] != FROZEN["g20_member_sha256"]
        or selected_member_row["bytes"] != FROZEN["g20_member_bytes"]
    ):
        raise ReplayError("reopened exact archive member bytes drifted")
    paths = {
        "runtime": inputs / "frozen_inflate.py",
        "source_member": inputs / "source.0.bin",
        "selected_member": inputs / "selected.0.bin",
    }
    _write_once(paths["runtime"], runtime_raw)
    _write_once(paths["source_member"], source_member)
    _write_once(paths["selected_member"], selected_member)
    return {
        "paths": paths,
        "runtime": _file_row(paths["runtime"], expected_sha256=FROZEN["runtime_sha256"]),
        "source_member": source_member_row,
        "selected_member": selected_member_row,
    }


def _chunk_rows(pair_count: int, chunk_pairs: int, frame_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, pair_count, chunk_pairs)):
        stop = min(pair_count, start + chunk_pairs)
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


def _render_object_chunk(
    *,
    label: str,
    row: Mapping[str, Any],
    workers: int,
    run_root: Path,
    tool_sha256: str,
    runtime_path: Path,
    member_path: Path,
    member_sha256: str,
    output_path: Path,
    output_bytes: int,
    receiver_metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    plans: list[Path] = []
    for worker_index, pair_ids in enumerate(_partition(row["pair_ids"], workers)):
        plan_path = run_root / "plans" / f"chunk-{row['index']:04d}.{label}.worker-{worker_index:02d}.json"
        plan = {
            "schema": "tac.g22_ep725_xcodec_internal_plan.v1",
            "action": "render",
            "tool_sha256": tool_sha256,
            "runtime_path": str(runtime_path),
            "runtime_sha256": FROZEN["runtime_sha256"],
            "member_path": str(member_path),
            "member_sha256": member_sha256,
            "output_path": str(output_path),
            "output_bytes": output_bytes,
            "receiver_metadata": receiver_metadata,
            "pair_ids": pair_ids,
        }
        _write_once_json(plan_path, plan)
        plans.append(plan_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plans)) as executor:
        futures = [executor.submit(_run_internal, path) for path in plans]
        return [future.result() for future in futures]


def _checkpoint_path(run_root: Path, index: int) -> Path:
    return run_root / "checkpoints" / f"chunk-{index:04d}.json"


def _validate_checkpoint(
    path: Path,
    *,
    row: Mapping[str, Any],
    manifest_sha256: str,
    source_raw: Path,
    selected_raw: Path,
) -> dict[str, Any]:
    checkpoint = _read_json(path)
    if (
        checkpoint.get("schema") != SCHEMA_CHUNK
        or checkpoint.get("manifest_sha256") != manifest_sha256
        or checkpoint.get("chunk", {}).get("index") != row["index"]
        or checkpoint.get("chunk", {}).get("pair_ids") != row["pair_ids"]
        or checkpoint.get("chunk", {}).get("byte_offset") != row["byte_offset"]
        or checkpoint.get("chunk", {}).get("byte_length") != row["byte_length"]
        or checkpoint.get("uint8_exact_equal") is not True
    ):
        raise ReplayError(f"chunk checkpoint contract mismatch: {path}")
    pointer = checkpoint.get("pointer_observation_after_chunk")
    if not isinstance(pointer, Mapping):
        raise ReplayError(f"chunk checkpoint lacks typed pointer observation: {path}")
    _validate_pointer_observation(pointer)
    source_hash, selected_hash = _compare_range(
        source_raw, selected_raw, row["byte_offset"], row["byte_length"]
    )
    if (
        checkpoint.get("source_range_sha256") != source_hash
        or checkpoint.get("selected_range_sha256") != selected_hash
    ):
        raise ReplayError(f"completed chunk bytes drifted: {path}")
    return checkpoint


def _validate_checkpoint_prefix(run_root: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    existing = {int(path.stem.split("-")[1]): path for path in (run_root / "checkpoints").glob("chunk-*.json")}
    unknown = sorted(set(existing) - {row["index"] for row in rows})
    if unknown:
        raise ReplayError(f"unexpected chunk checkpoints: {unknown}")
    completed = 0
    while completed in existing:
        completed += 1
    if any(index >= completed for index in existing):
        raise ReplayError("completed chunks are not an immutable prefix")
    return completed


def _cleanup_certified_raws(
    *,
    run_root: Path,
    source_raw: Path,
    selected_raw: Path,
    precleanup_path: Path,
    precleanup: Mapping[str, Any],
) -> dict[str, Any]:
    certificate_path = run_root / "cleanup" / "raw_witness_cleanup_certificate.json"
    complete_path = run_root / "cleanup" / "raw_witness_cleanup_complete.json"
    targets = precleanup["output_witness"]["scratch_files"]
    existing_certificate = _read_json(certificate_path) if certificate_path.exists() else None
    certificate = {
        "schema": SCHEMA_CLEANUP,
        "certified_at_utc": (
            existing_certificate["certified_at_utc"]
            if existing_certificate is not None
            else dt.datetime.now(dt.UTC).isoformat()
        ),
        "precleanup_receipt": _file_row(precleanup_path),
        "targets": targets,
        "rebuild_command": precleanup["reproducibility"]["exact_resume_command"],
        "rebuildable_from": {
            "run_manifest": precleanup["decode_receipt"]["run_manifest"],
            "chunk_checkpoints": precleanup["decode_receipt"]["chunk_checkpoints"],
            "archive_artifact": precleanup["archive_artifact"],
            "generic_decoder_runtime": precleanup["generic_decoder_runtime"],
        },
        "reason": "successful exact source/selected equality replay; raw witnesses are deterministically rebuildable scratch",
        "delete_authorized": True,
        "score_authority": False,
    }
    _write_once_json(certificate_path, certificate)
    # Resume deletion from either per-file boundary, but never delete a surviving
    # file without immediately revalidating it against the prior certificate.
    for target in targets:
        path = Path(target["path"])
        if path.exists():
            row = _file_row(path, expected_sha256=target["sha256"], expected_bytes=target["bytes"])
            if row["path"] != target["path"]:
                raise ReplayError("cleanup target path drift")
            path.unlink()
    if source_raw.exists() or selected_raw.exists():
        raise ReplayError("certified scratch cleanup is incomplete")
    existing_complete = _read_json(complete_path) if complete_path.exists() else None
    complete = {
        "schema": SCHEMA_CLEANUP_COMPLETE,
        "completed_at_utc": (
            existing_complete["completed_at_utc"]
            if existing_complete is not None
            else dt.datetime.now(dt.UTC).isoformat()
        ),
        "certificate": _file_row(certificate_path),
        "removed_paths": [str(source_raw), str(selected_raw)],
        "all_absent_after_cleanup": True,
        "preserved_checkpoint_count": len(precleanup["decode_receipt"]["chunk_checkpoints"]),
    }
    _write_once_json(complete_path, complete)
    return {"certificate": _file_row(certificate_path), "completion": _file_row(complete_path)}


def _write_final_from_precleanup(
    *,
    receipt_path: Path,
    precleanup_path: Path,
    precleanup: Mapping[str, Any],
    cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    final = json.loads(json.dumps(precleanup))
    final["schema"] = SCHEMA_FINAL
    final["cleanup"] = {
        "completed": True,
        "success_only": True,
        "certificate": cleanup["certificate"],
        "completion": cleanup["completion"],
        "scratch_absent": True,
        "per_chunk_checkpoints_preserved": True,
    }
    final["precleanup_receipt"] = _file_row(precleanup_path)
    _write_once_json(receipt_path, final)
    return final


def _resume_from_precleanup(
    *,
    precleanup_path: Path,
    receipt_path: Path,
    run_root: Path,
    manifest_row: Mapping[str, Any],
    source_raw: Path,
    selected_raw: Path,
    output_bytes: int,
    pair_ids: Sequence[int],
) -> dict[str, Any] | None:
    if not precleanup_path.exists():
        return None
    precleanup = _read_json(precleanup_path)
    decode = precleanup.get("decode_receipt", {})
    authority = precleanup.get("authority_pointer_status", {})
    witness = precleanup.get("output_witness", {})
    if (
        precleanup.get("schema") != SCHEMA_PRECLEANUP
        or Path(precleanup.get("run_root", "")) != run_root
        or decode.get("run_manifest") != manifest_row
        or decode.get("pair_ids") != list(pair_ids)
        or witness.get("raw_bytes_each") != output_bytes
        or witness.get("uint8_exact_equal") is not True
    ):
        raise ReplayError("pre-cleanup recovery receipt differs from current immutable run")
    _validate_preserved_row(decode["run_manifest"], "pre-cleanup run manifest")
    _validate_pointer_observation(authority.get("pointer_start", {}))
    _validate_pointer_observation(authority.get("pointer_end", {}))
    checkpoint_payloads: list[dict[str, Any]] = []
    for index, row in enumerate(decode.get("chunk_checkpoints", [])):
        _validate_preserved_row(row, f"pre-cleanup chunk checkpoint {index}")
        checkpoint_payloads.append(_read_json(Path(row["path"])))
    aggregate = _aggregate_pointer_change(
        authority["pointer_start"],
        authority["pointer_end"],
        checkpoint_payloads,
    )
    if any(authority.get(key) is not value for key, value in aggregate.items()):
        raise ReplayError("pre-cleanup pointer change classification drifted")
    targets = witness.get("scratch_files", [])
    if [row.get("path") for row in targets] != [str(source_raw), str(selected_raw)]:
        raise ReplayError("pre-cleanup scratch target paths drifted")
    cleanup = _cleanup_certified_raws(
        run_root=run_root,
        source_raw=source_raw,
        selected_raw=selected_raw,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
    )
    return _write_final_from_precleanup(
        receipt_path=receipt_path,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
        cleanup=cleanup,
    )


def _validate_preserved_row(row: Mapping[str, Any], label: str) -> None:
    path = Path(row.get("path", ""))
    if not path.is_absolute():
        raise ReplayError(f"{label} path is not absolute")
    _file_row(
        path,
        expected_sha256=row.get("sha256"),
        expected_bytes=row.get("bytes"),
    )


def _finalize_existing(
    receipt_path: Path,
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not receipt_path.exists():
        return None
    receipt = _read_json(receipt_path)
    if receipt.get("schema") != SCHEMA_FINAL:
        raise ReplayError("existing final receipt schema drift")
    if Path(receipt.get("run_root", "")) != run_root:
        raise ReplayError("existing final receipt binds another run root")
    if receipt.get("cleanup", {}).get("completed") is not True:
        raise ReplayError("existing final receipt lacks cleanup closure")
    _validate_frozen_contract()
    tool = receipt.get("reproducibility", {}).get("tool", {})
    _validate_preserved_row(tool, "existing receipt tool")
    if Path(tool.get("path", "")).resolve() != Path(__file__).resolve():
        raise ReplayError("existing final receipt binds another replay harness")
    authority = receipt.get("authority_pointer_status", {})
    _validate_pointer_observation(authority.get("pointer_start", {}))
    _validate_pointer_observation(authority.get("pointer_end", {}))
    decode = receipt.get("decode_receipt", {})
    manifest_row = decode.get("run_manifest", {})
    _validate_preserved_row(manifest_row, "completed run manifest")
    manifest = _read_json(Path(manifest_row["path"]))
    config = manifest.get("config", {})
    if (
        config.get("pair_start") != args.pair_start
        or config.get("pair_count") != args.pair_count
        or config.get("chunk_pairs") != args.chunk_pairs
        or config.get("workers") != args.workers
    ):
        raise ReplayError("completed replay invocation differs from immutable run manifest")
    checkpoint_payloads: list[dict[str, Any]] = []
    for index, row in enumerate(decode.get("chunk_checkpoints", [])):
        _validate_preserved_row(row, f"completed chunk checkpoint {index}")
        checkpoint_payloads.append(_read_json(Path(row["path"])))
    aggregate = _aggregate_pointer_change(
        authority["pointer_start"],
        authority["pointer_end"],
        checkpoint_payloads,
    )
    if any(authority.get(key) is not value for key, value in aggregate.items()):
        raise ReplayError("completed pointer change classification drifted")
    _validate_preserved_row(receipt.get("precleanup_receipt", {}), "pre-cleanup receipt")
    cleanup = receipt["cleanup"]
    _validate_preserved_row(cleanup.get("certificate", {}), "cleanup certificate")
    _validate_preserved_row(cleanup.get("completion", {}), "cleanup completion")
    for scratch_row in receipt.get("output_witness", {}).get("scratch_files", []):
        if Path(scratch_row.get("path", "")).exists():
            raise ReplayError("completed cleanup conflicts with a surviving raw witness")
    selected = receipt.get("archive_artifact", {}).get("selected", {})
    if (
        authority.get("candidate_claim") is not False
        or authority.get("score_claim") is not False
        or authority.get("decode_equality_independent_of_pointer_change") is not True
        or selected.get("counted_rate_bytes") != FROZEN["g20_archive_bytes"]
    ):
        raise ReplayError("completed receipt type/authority truth drift")
    if receipt.get("resource_custody", {}).get("storage_preflight", {}).get("reserve_bytes") != args.reserve_bytes:
        raise ReplayError("completed replay storage reserve differs from invocation")
    return receipt


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute_reviewed:
        raise ReplayError("execution is fail-closed; pass --execute-reviewed after reviewing the exact command")
    if args.pair_start != 0:
        raise ReplayError("this exact receiver writes global offsets; bounded replay must be ordered prefix from pair 0")
    if args.pair_count < 1 or args.pair_count > 600:
        raise ReplayError("--pair-count must be in 1..600")
    if args.chunk_pairs < 1 or args.workers < 1:
        raise ReplayError("chunk and worker counts must be positive")
    if args.pair_count == 600 and not args.confirm_full_n600:
        raise ReplayError("full heavy replay requires --confirm-full-n600")
    if args.confirm_full_n600 and args.pair_count != 600:
        raise ReplayError("--confirm-full-n600 is valid only for exactly pair IDs 0..599")
    run_root = _validate_run_root(args.resume_from, allow_non_ssd=args.allow_non_ssd)
    receipt_path = args.receipt.absolute()
    existing_final = _finalize_existing(receipt_path, run_root, args)
    if existing_final is not None:
        return existing_final
    _assert_no_partial_files(run_root)
    contract = _validate_frozen_contract()
    invocation_pointer = _observe_live_pointer(run_root)
    tool = _file_row(Path(__file__).resolve())
    custody = _materialize_custody_inputs(run_root, contract)
    inspect_source = _inspect(
        custody["paths"]["runtime"],
        custody["paths"]["source_member"],
        tool["sha256"],
        run_root / "plans" / "inspect.source.json",
    )
    inspect_selected = _inspect(
        custody["paths"]["runtime"],
        custody["paths"]["selected_member"],
        tool["sha256"],
        run_root / "plans" / "inspect.selected.json",
    )
    if inspect_source != inspect_selected:
        raise ReplayError("source/selected receiver metadata differ")
    if inspect_source["n_pairs"] != 600:
        raise ReplayError("frozen receiver PairPopulation is not exactly n600")
    pair_ids = list(range(args.pair_count))
    if args.pair_count == 600 and pair_ids != list(range(600)):
        raise ReplayError("full PairPopulation must be exact ordered pair IDs 0..599")
    frame_bytes = inspect_source["frame_bytes"]
    output_bytes = args.pair_count * 2 * frame_bytes
    rows = _chunk_rows(args.pair_count, args.chunk_pairs, frame_bytes)
    config = {
        "pair_start": 0,
        "pair_count": args.pair_count,
        "pair_ids": pair_ids,
        "full_n600": args.pair_count == 600,
        "chunk_pairs": args.chunk_pairs,
        "workers": args.workers,
        "portable_fp64": True,
        "receiver_private_calls": ["_setup", "_render_pair"],
        "receiver_metadata": inspect_source,
        "output_bytes_per_object": output_bytes,
    }
    config_sha256 = _sha256_bytes(_canonical_json(config))
    manifest_path = run_root / "run_manifest.json"
    existing_manifest = _read_json(manifest_path) if manifest_path.exists() else None
    pointer_start = (
        existing_manifest["pointer_start"]
        if existing_manifest is not None
        else invocation_pointer
    )
    _validate_pointer_observation(pointer_start)
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "created_at_utc": (
            existing_manifest["created_at_utc"]
            if existing_manifest is not None
            else dt.datetime.now(dt.UTC).isoformat()
        ),
        "run_root": str(run_root),
        "git_head": _git_head(),
        "tool": tool,
        "config": config,
        "config_sha256": config_sha256,
        "frozen_contract": contract["rows"],
        "custody_inputs": {
            "runtime": custody["runtime"],
            "source_member": custody["source_member"],
            "selected_member": custody["selected_member"],
        },
        "pointer_start": pointer_start,
        "invocation_pointer_at_manifest_open": (
            existing_manifest["invocation_pointer_at_manifest_open"]
            if existing_manifest is not None
            else invocation_pointer
        ),
        "research_only": True,
        "exact_eval_invoked": False,
    }
    _write_once_json(manifest_path, manifest)
    manifest_row = _file_row(manifest_path)
    # A resume is accepted only when all execution-critical bytes still match
    # the original write-once manifest.
    loaded_manifest = _read_json(manifest_path)
    if loaded_manifest != manifest:
        raise ReplayError("resume manifest differs from current exact invocation")
    scratch = run_root / "scratch"
    source_raw = scratch / "source.raw"
    selected_raw = scratch / "selected.raw"
    precleanup_path = run_root / "receipts" / "decode_receipt.pre_cleanup.json"
    recovered = _resume_from_precleanup(
        precleanup_path=precleanup_path,
        receipt_path=receipt_path,
        run_root=run_root,
        manifest_row=manifest_row,
        source_raw=source_raw,
        selected_raw=selected_raw,
        output_bytes=output_bytes,
        pair_ids=pair_ids,
    )
    if recovered is not None:
        return recovered
    free = shutil.disk_usage(run_root).free
    missing_allocation = sum(output_bytes for path in (source_raw, selected_raw) if not path.exists())
    if free < missing_allocation + args.reserve_bytes:
        raise ReplayError(
            f"storage preflight failed: free={free}, required={missing_allocation + args.reserve_bytes}"
        )
    _preallocate(source_raw, output_bytes)
    _preallocate(selected_raw, output_bytes)
    completed_prefix = _validate_checkpoint_prefix(run_root, rows)
    checkpoints: list[dict[str, Any]] = []
    total_started = time.monotonic()
    total_before = _rusage_snapshot()
    for index, row in enumerate(rows):
        path = _checkpoint_path(run_root, index)
        if index < completed_prefix:
            checkpoints.append(
                _validate_checkpoint(
                    path,
                    row=row,
                    manifest_sha256=manifest_row["sha256"],
                    source_raw=source_raw,
                    selected_raw=selected_raw,
                )
            )
            continue
        chunk_started = time.monotonic()
        before = _rusage_snapshot()
        source_workers = _render_object_chunk(
            label="source",
            row=row,
            workers=args.workers,
            run_root=run_root,
            tool_sha256=tool["sha256"],
            runtime_path=custody["paths"]["runtime"],
            member_path=custody["paths"]["source_member"],
            member_sha256=FROZEN["source_member_sha256"],
            output_path=source_raw,
            output_bytes=output_bytes,
            receiver_metadata=inspect_source,
        )
        selected_workers = _render_object_chunk(
            label="selected",
            row=row,
            workers=args.workers,
            run_root=run_root,
            tool_sha256=tool["sha256"],
            runtime_path=custody["paths"]["runtime"],
            member_path=custody["paths"]["selected_member"],
            member_sha256=FROZEN["g20_member_sha256"],
            output_path=selected_raw,
            output_bytes=output_bytes,
            receiver_metadata=inspect_selected,
        )
        source_hash, selected_hash = _compare_range(
            source_raw, selected_raw, row["byte_offset"], row["byte_length"]
        )
        chunk_pointer = _observe_live_pointer(run_root)
        pointer_change = _pointer_change(pointer_start, chunk_pointer)
        after = _rusage_snapshot()
        checkpoint = {
            "schema": SCHEMA_CHUNK,
            "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "manifest_sha256": manifest_row["sha256"],
            "chunk": dict(row),
            "frame_ids": list(range(row["pair_ids"][0] * 2, (row["pair_ids"][-1] + 1) * 2)),
            "source_range_sha256": source_hash,
            "selected_range_sha256": selected_hash,
            "uint8_exact_equal": True,
            "pointer_observation_after_chunk": chunk_pointer,
            "pointer_change_from_run_start": pointer_change,
            "source_workers": source_workers,
            "selected_workers": selected_workers,
            "wall_seconds": time.monotonic() - chunk_started,
            "resource_usage": _resource_delta(before, after),
        }
        _write_once_json(path, checkpoint)
        checkpoints.append(_validate_checkpoint(
            path,
            row=row,
            manifest_sha256=manifest_row["sha256"],
            source_raw=source_raw,
            selected_raw=selected_raw,
        ))
    source_full_sha = _sha256_file_streaming(source_raw, output_bytes)
    selected_full_sha = _sha256_file_streaming(selected_raw, output_bytes)
    if source_full_sha != selected_full_sha:
        raise ReplayError("full output witness hashes differ after chunk equality")
    if args.pair_count == 1 and source_full_sha != FROZEN["bounded_pair0_raw_sha256"]:
        raise ReplayError("real bounded pair-0 replay differs from the frozen G20 proof")
    pointer_end = _observe_live_pointer(run_root)
    end_contract = _validate_frozen_contract()
    if end_contract["rows"] != contract["rows"]:
        raise ReplayError("frozen contract drifted during replay")
    if _file_row(Path(__file__).resolve()) != tool:
        raise ReplayError("replay harness drifted during execution")
    _file_row(custody["paths"]["runtime"], expected_sha256=FROZEN["runtime_sha256"])
    _file_row(custody["paths"]["source_member"], expected_sha256=FROZEN["source_member_sha256"])
    _file_row(custody["paths"]["selected_member"], expected_sha256=FROZEN["g20_member_sha256"])
    total_after = _rusage_snapshot()
    checkpoint_rows = [_file_row(_checkpoint_path(run_root, row["index"])) for row in rows]
    aggregate_pointer_change = _aggregate_pointer_change(pointer_start, pointer_end, checkpoints)
    resume_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--resume-from",
        str(run_root),
        "--receipt",
        str(receipt_path),
        "--pair-start",
        "0",
        "--pair-count",
        str(args.pair_count),
        "--chunk-pairs",
        str(args.chunk_pairs),
        "--workers",
        str(args.workers),
        "--reserve-bytes",
        str(args.reserve_bytes),
        "--execute-reviewed",
    ]
    if args.pair_count == 600:
        resume_command.append("--confirm-full-n600")
    if args.allow_non_ssd:
        resume_command.append("--allow-non-ssd")
    existing_precleanup = _read_json(precleanup_path) if precleanup_path.exists() else None
    precleanup = {
        "schema": SCHEMA_PRECLEANUP,
        "generated_at_utc": (
            existing_precleanup["generated_at_utc"]
            if existing_precleanup is not None
            else dt.datetime.now(dt.UTC).isoformat()
        ),
        "run_root": str(run_root),
        "artifact_lifecycle": [
            "ArchiveArtifact",
            "generic_decoder_runtime",
            "DecodeReceipt",
            "output_witness",
        ],
        "archive_artifact": {
            "artifact_type": "counted_video_specific_archive_and_state",
            "source": {
                **contract["rows"]["source_archive"],
                "member": custody["source_member"],
            },
            "selected": {
                **contract["rows"]["selected_archive"],
                "member": custody["selected_member"],
                "decoded_quantized_state_sha256": FROZEN["state_sha256"],
                "counted_rate_bytes": FROZEN["g20_archive_bytes"],
                "classification": "not_a_candidate",
            },
        },
        "generic_decoder_runtime": {
            "type": "generic_decoder_source",
            **contract["rows"]["runtime"],
            "custody_critical": True,
            "contest_rate_cost_bytes": 0,
            "free_under_rule_118": True,
            "private_receiver_path_executed": ["_setup", "_render_pair"],
            "portable_fp64": True,
        },
        "encoder_only_rewrite_evidence": {
            "type": "encoder_only_exact_recode_proof",
            "g20_receipt": contract["rows"]["g20_receipt"],
            "implementation": {
                "module": contract["rows"]["g20_module"],
                "materializer": contract["rows"]["g20_materializer"],
            },
            "source_state_sha256": FROZEN["state_sha256"],
            "selected_state_sha256": FROZEN["state_sha256"],
            "decoded_state_equal": True,
            "selection_surface": "exact complete archive.zip bytes",
            "encoder_only": True,
        },
        "decode_receipt": {
            "type": "frozen_receiver_full_or_bounded_equality_replay",
            "run_manifest": manifest_row,
            "chunk_checkpoints": checkpoint_rows,
            "chunk_count": len(rows),
            "all_chunks_immutable_and_validated": True,
            "pair_ids": pair_ids,
            "pair_population_exact_ordered_0_through_599": args.pair_count == 600,
        },
        "output_witness": {
            "type": "realized_uint8_receiver_output_witness",
            "axis": "[macOS-CPU frozen receiver structural proof]",
            "dtype": "uint8",
            "layout": "ordered frames [pair0_frame0,pair0_frame1,...] in RGB HWC bytes",
            "camera_h": inspect_source["camera_h"],
            "camera_w": inspect_source["camera_w"],
            "pairs_compared": args.pair_count,
            "frames_compared": args.pair_count * 2,
            "all_bytes_directly_compared": True,
            "uint8_exact_equal": True,
            "source_raw_sha256": source_full_sha,
            "selected_raw_sha256": selected_full_sha,
            "raw_bytes_each": output_bytes,
            "scratch_files": [
                {"path": str(source_raw), "bytes": output_bytes, "sha256": source_full_sha},
                {"path": str(selected_raw), "bytes": output_bytes, "sha256": selected_full_sha},
            ],
        },
        "authority_pointer_status": {
            "type": "authority_and_frontier_status",
            "pointer_start": pointer_start,
            "pointer_end": pointer_end,
            "pointer_artifact_changed": aggregate_pointer_change["pointer_artifact_changed"],
            "competitive_target_changed": aggregate_pointer_change["competitive_target_changed"],
            "rebase_required_before_admission": aggregate_pointer_change[
                "rebase_required_before_admission"
            ],
            "decode_equality_invalidated": False,
            "decode_equality_independent_of_pointer_change": True,
            "candidate_claim": False,
            "score_claim": False,
            "exact_eval_invoked": False,
            "promotion_eligible": False,
            "research_only": True,
            "full_n600_receiver_replay_owed": args.pair_count != 600,
            "contest_cpu_cuda_same_bytes_owed": True,
        },
        "resource_custody": {
            "wall_seconds": time.monotonic() - total_started,
            "resource_usage": _resource_delta(total_before, total_after),
            "storage_preflight": {
                "tier": str(run_root),
                "free_bytes_before_preallocation": free,
                "required_new_allocation_bytes": missing_allocation,
                "reserve_bytes": args.reserve_bytes,
                "passed": True,
            },
        },
        "reproducibility": {
            "git_head": _git_head(),
            "tool": tool,
            "python": sys.version,
            "platform": platform.platform(),
            "worker_environment": dict.fromkeys(
                ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
                "1",
            )
            | {"INFLATE_FP32": "0", "PYTHONHASHSEED": "0"},
            "exact_resume_argv": resume_command,
            "exact_resume_command": shlex.join(resume_command),
        },
    }
    _write_once_json(precleanup_path, precleanup)
    cleanup = _cleanup_certified_raws(
        run_root=run_root,
        source_raw=source_raw,
        selected_raw=selected_raw,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
    )
    return _write_final_from_precleanup(
        receipt_path=receipt_path,
        precleanup_path=precleanup_path,
        precleanup=precleanup,
        cleanup=cleanup,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path, help="SSD run directory; created or resumed in place")
    parser.add_argument("--receipt", type=Path, help="append-only durable final receipt path")
    parser.add_argument("--pair-start", type=int, default=0)
    parser.add_argument("--pair-count", type=int, default=600)
    parser.add_argument("--chunk-pairs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reserve-bytes", type=int, default=1024**3)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--confirm-full-n600", action="store_true")
    parser.add_argument("--allow-non-ssd", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-plan", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.internal_plan is not None:
            return _internal_plan(args.internal_plan)
        if args.resume_from is None or args.receipt is None:
            raise ReplayError("--resume-from and --receipt are required")
        receipt = _execute(args)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except (ReplayError, OSError, KeyError, ValueError, TypeError) as exc:
        print(f"G22_REPLAY_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
