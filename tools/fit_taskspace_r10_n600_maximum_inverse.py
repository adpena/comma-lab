#!/usr/bin/env python3
"""Resumable source-only production launcher for the G32 R10 inverse fitter.

The launcher never imports or runs a frozen evaluator.  It materializes the
own-lineage selected realization and encoder-only source on an SSD, persists
immutable fitted stage state, and emits a research-only counted packet plus
physical-custody receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from tac.witness_dsl.taskspace_r10_feature_texture_relay import parse_r10_packet, serialize_r10_packet
from tac.witness_dsl.taskspace_r10_n600_maximum_inverse_fitter import (
    G32_AUTHORITY_BLOCKER,
    R10BoundedInverseConfigV1,
    R10BoundedInverseError,
    compile_r10_bounded_inverse,
    result_receipt_dict,
    streaming_realization_sha256,
    strict_json_loads,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
RESEARCH_ROOT: Final = REPO_ROOT / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
R10_MODULE: Final = REPO_ROOT / "src/tac/witness_dsl/taskspace_r10_feature_texture_relay.py"
FITTER_MODULE: Final = REPO_ROOT / "src/tac/witness_dsl/taskspace_r10_n600_maximum_inverse_fitter.py"
G22_RECEIPT: Final = RESEARCH_ROOT / "g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json"

FROZEN: Final = {
    "source_video_sha256": "2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9",
    "source_video_bytes": 37_545_489,
    "selected_archive_sha256": "8e9c7ba0fdd1fc0fdff696c639821d6e64a3110bb8744f47ae0ab3d287cd70d8",
    "selected_archive_bytes": 81_027,
    "selected_member_sha256": "4789bf6b5f15272cc5f8a573f25137a9daf7e21755e81aa48a8fba84947b5634",
    "selected_member_bytes": 81_738,
    "runtime_sha256": "4b54d512565f7275c53f697a931dd087222a36a69495b6e536a6b65dede36224",
    "runtime_bytes": 56_814,
    "r10_module_sha256": "13cd771d10c333a458c9977f8b21b916a4baf80b063bb4f849f001a6f660e11d",
    "r10_module_bytes": 65_411,
    "g22_receipt_sha256": "3a01e81abfd19a78db86e5851f1b0c453ff553c1fe7d5fad830f95bcd5ec3efd",
    "g22_receipt_bytes": 24_210,
    "full_n600_base_realization_sha256": "8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae",
}

HEIGHT: Final = 874
WIDTH: Final = 1164
CHANNELS: Final = 3
STAGES: Final = (
    "000_custody",
    "010_selected_base",
    "020_pair_index",
    "030_geometry",
    "040_xip2",
    "050_base_feature",
    "060_texture",
    "070_shooting_knot",
    "080_dash1",
    "090_pullback_polygon",
    "100_stratified_flow",
    "110_joint_refit",
    "120_packet_adapter",
    "130_bounded_decode_receipt",
    "140_cleanup_certificate",
)
SSD_ROOTS: Final = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))
THREAD_ENV: Final = {
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "INFLATE_FP32": "0",
    "PYTHONHASHSEED": "0",
}
G32_LANE_ID: Final = "lane_g32_r10_n600_inverse_fitter_20260726"
CLAIM_LEDGER: Final = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
GOVERNED_MARKER_ENV: Final = "TAC_GOVERNED_ADMISSION"
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
STAGE_RECORD_KEYS: Final = frozenset(
    {
        "schema",
        "stage",
        "stage_index",
        "binding_sha256",
        "predecessor_record_sha256",
        "payload_sha256",
        "payload",
    }
)
RANGE_RECORD_KEYS: Final = frozenset(
    {
        "schema",
        "binding_sha256",
        "kind",
        "first_pair",
        "stop_pair",
        "path",
        "byte_offset",
        "byte_length",
        "range_sha256",
        "predecessor_record_sha256",
        "telemetry",
    }
)
FIT_RANGE_STAGES: Final = (
    "040_xip2",
    "050_base_feature",
    "060_texture",
    "070_shooting_knot",
    "100_stratified_flow",
    "110_joint_base_feature",
    "110_joint_texture",
    "110_joint_shooting_knot",
    "110_joint_stratified_flow",
)
FIT_RANGE_RECORD_KEYS: Final = frozenset(
    {
        "schema",
        "binding_sha256",
        "fit_stage",
        "first_pair",
        "stop_pair",
        "predecessor_record_sha256",
        "payload_sha256",
        "payload",
    }
)
INTERNAL_RUNTIME_PLAN_KEYS: Final = frozenset(
    {
        "schema",
        "binding_sha256",
        "run_root",
        "runtime",
        "member",
        "output",
        "pair_count",
        "chunk_pairs",
        "first_pair",
        "receiver_metadata",
        "governed_claim_job_id",
        "governed_claim_platform",
        "governed_claim_sha256",
    }
)


class G32LaunchError(RuntimeError):
    """A production launch, custody, resume, storage, or cleanup invariant failed."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_read(path: Path) -> bytes:
    absolute = path.absolute()
    before = absolute.lstat()
    if absolute.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise G32LaunchError(f"required path is not a real regular file: {absolute}")
    fd = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise G32LaunchError(f"path identity changed before open: {absolute}")
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = absolute.lstat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after_fd.st_dev, after_fd.st_ino, after_fd.st_size, after_fd.st_mtime_ns):
        raise G32LaunchError(f"file drifted while open: {absolute}")
    if identity != (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns):
        raise G32LaunchError(f"file path drifted while read: {absolute}")
    payload = b"".join(blocks)
    if len(payload) != before.st_size:
        raise G32LaunchError(f"short stable read: {absolute}")
    return payload


def _file_row(path: Path, expected_sha256: str, expected_bytes: int) -> dict[str, Any]:
    payload = _stable_read(path)
    digest = _sha256_bytes(payload)
    if digest != expected_sha256 or len(payload) != expected_bytes:
        raise G32LaunchError(
            f"custody drift for {path}: ({len(payload)},{digest}) != ({expected_bytes},{expected_sha256})"
        )
    return {"path": str(path.absolute()), "bytes": len(payload), "sha256": digest}


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _stable_read(path) != payload:
            raise G32LaunchError(f"immutable artifact differs: {path}")
        return
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    fd = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise G32LaunchError(f"short immutable write: {partial}")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.exists():
        partial.unlink()
        if _stable_read(path) != payload:
            raise G32LaunchError(f"immutable write race differs: {path}")
        return
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_file_streaming(path: Path, *, read_bytes: int = 16 << 20) -> str:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise G32LaunchError(f"streaming hash target is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(read_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_range(path: Path, offset: int, length: int) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(fd)
        if offset < 0 or length < 0 or offset + length > metadata.st_size:
            raise G32LaunchError("checkpoint range escapes retained raw file")
        os.lseek(fd, offset, os.SEEK_SET)
        remaining = length
        while remaining:
            block = os.read(fd, min(1 << 20, remaining))
            if not block:
                raise G32LaunchError("short range during checkpoint validation")
            digest.update(block)
            remaining -= len(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _write_range(path: Path, offset: int, payload: bytes) -> None:
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        view = memoryview(payload)
        cursor = offset
        while view:
            count = os.pwrite(fd, view, cursor)
            if count <= 0:
                raise G32LaunchError(f"short range write: {path}")
            view = view[count:]
            cursor += count
        os.fsync(fd)
    finally:
        os.close(fd)


def _preallocate(path: Path, expected_bytes: int) -> None:
    if path.exists():
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_bytes:
            raise G32LaunchError(f"retained raw has wrong identity or size: {path}")
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
            zeros = b"\0" * (8 << 20)
            cursor = 0
            while cursor < expected_bytes:
                block = zeros[: min(len(zeros), expected_bytes - cursor)]
                count = os.pwrite(fd, block, cursor)
                if count != len(block):
                    raise G32LaunchError(f"short physical preallocation: {path}")
                cursor += count
        os.fsync(fd)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)


def _path_is_under(path: Path, roots: Sequence[Path]) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in roots)


def validate_run_root(path: Path, *, allow_local_test_only: bool) -> Path:
    absolute = path.absolute()
    if absolute.exists() and (absolute.is_symlink() or not absolute.is_dir()):
        raise G32LaunchError("--resume-from must name one real directory")
    if not allow_local_test_only and not _path_is_under(absolute, SSD_ROOTS):
        raise G32LaunchError("--resume-from must live under the configured SSD waterfall")
    absolute.mkdir(parents=True, exist_ok=True)
    if absolute.is_symlink():
        raise G32LaunchError("--resume-from may not be a symlink")
    return absolute


def storage_preflight(run_root: Path, *, pair_count: int, reserve_bytes: int) -> dict[str, int]:
    pair_bytes = pair_count * 2 * HEIGHT * WIDTH * CHANNELS
    label_bytes = pair_count * HEIGHT * WIDTH
    required = 5 * pair_bytes + label_bytes + reserve_bytes
    usage = shutil.disk_usage(run_root)
    if usage.free < required:
        raise G32LaunchError(f"storage preflight failed: free={usage.free} required={required}")
    return {"free_bytes": usage.free, "required_bytes": required, "reserve_bytes": reserve_bytes}


class ImmutableStageStore:
    """Content-addressed, contiguous, write-once stage checkpoint registry."""

    def __init__(self, root: Path, binding_sha256: str):
        self.root = root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding_sha256 = binding_sha256

    def _matches(self, stage: str) -> list[Path]:
        return sorted(self.root.glob(f"{stage}.*.json"))

    @staticmethod
    def _record_digest(path: Path, raw: bytes, *, prefix: str) -> str:
        digest = _sha256_bytes(raw)
        if path.name != f"{prefix}.{digest}.json":
            raise G32LaunchError(f"checkpoint filename/content digest drifted: {path}")
        return digest

    def _load_records(self) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
        out: dict[str, Mapping[str, Any]] = {}
        digests: dict[str, str] = {}
        gap = False
        predecessor: str | None = None
        for stage in STAGES:
            matches = self._matches(stage)
            if len(matches) > 1:
                raise G32LaunchError(f"duplicate immutable checkpoint stage: {stage}")
            if not matches:
                gap = True
                continue
            if gap:
                raise G32LaunchError(f"non-contiguous stage checkpoint appeared at {stage}")
            raw = _stable_read(matches[0])
            record_digest = self._record_digest(matches[0], raw, prefix=stage)
            record = strict_json_loads(raw)
            if not isinstance(record, dict) or set(record) != STAGE_RECORD_KEYS:
                raise G32LaunchError(f"checkpoint record schema/key set drifted: {matches[0]}")
            payload = record["payload"]
            if (
                record["schema"] != "tac.g35_r10_immutable_stage.v2"
                or record["stage"] != stage
                or record["stage_index"] != STAGES.index(stage)
                or record["binding_sha256"] != self.binding_sha256
                or record["predecessor_record_sha256"] != predecessor
                or not isinstance(payload, dict)
                or record["payload_sha256"] != _sha256_bytes(_canonical_json(payload))
            ):
                raise G32LaunchError(f"checkpoint identity, predecessor, or payload drifted: {matches[0]}")
            out[stage] = payload
            digests[stage] = record_digest
            predecessor = record_digest
        return out, digests

    def publish(self, stage: str, payload: Mapping[str, Any]) -> Path:
        if stage not in STAGES:
            raise G32LaunchError(f"unknown checkpoint stage: {stage}")
        normalized = strict_json_loads(_canonical_json(payload))
        if not isinstance(normalized, dict):
            raise G32LaunchError("checkpoint payload must normalize to one JSON object")
        existing, digests = self._load_records()
        stage_index = STAGES.index(stage)
        if stage in existing:
            matches = self._matches(stage)
            if existing[stage] != normalized or len(matches) != 1:
                raise G32LaunchError(f"immutable stage has conflicting retained state: {stage}")
            return matches[0]
        if stage_index != len(existing):
            raise G32LaunchError(
                "non-contiguous immutable stage publication must be exactly next in chain: "
                f"stage={stage} prefix={tuple(existing)}"
            )
        predecessor = None if stage_index == 0 else digests[STAGES[stage_index - 1]]
        payload_raw = _canonical_json(normalized)
        payload_hash = _sha256_bytes(payload_raw)
        record = {
            "schema": "tac.g35_r10_immutable_stage.v2",
            "stage": stage,
            "stage_index": stage_index,
            "binding_sha256": self.binding_sha256,
            "predecessor_record_sha256": predecessor,
            "payload_sha256": payload_hash,
            "payload": normalized,
        }
        raw = _canonical_json(record)
        path = self.root / f"{stage}.{_sha256_bytes(raw)}.json"
        matches = self._matches(stage)
        if matches and matches != [path]:
            raise G32LaunchError(f"immutable stage has conflicting retained state: {stage}")
        _write_once(path, raw)
        self.load_prefix()
        return path

    def load_prefix(self) -> dict[str, Mapping[str, Any]]:
        return self._load_records()[0]


class ChunkStore:
    """Immutable raw-range checkpoints used by selected/source materialization."""

    def __init__(self, root: Path, binding_sha256: str):
        self.root = root / "chunk_checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding_sha256 = binding_sha256

    def _prefix(self, kind: str, first_pair: int, stop_pair: int) -> str:
        return f"{kind}.{first_pair:04d}-{stop_pair:04d}"

    def _load_kind_chain(self, kind: str, raw_path: Path) -> list[tuple[Path, Mapping[str, Any], str]]:
        if not kind or not re.fullmatch(r"[a-z][a-z0-9_]*", kind):
            raise G32LaunchError("range checkpoint kind is not canonical")
        pattern = re.compile(rf"^{re.escape(kind)}\.(\d{{4}})-(\d{{4}})\.([0-9a-f]{{64}})\.json$")
        rows: list[tuple[Path, Mapping[str, Any], str]] = []
        expected_first = 0
        predecessor: str | None = None
        canonical_path = str(raw_path.absolute())
        for path in sorted(self.root.glob(f"{kind}.*.json")):
            match = pattern.fullmatch(path.name)
            if match is None:
                raise G32LaunchError(f"range checkpoint filename is not canonical: {path}")
            first_pair, stop_pair = int(match.group(1)), int(match.group(2))
            raw = _stable_read(path)
            digest = _sha256_bytes(raw)
            if digest != match.group(3):
                raise G32LaunchError(f"range checkpoint filename/content digest drifted: {path}")
            record = strict_json_loads(raw)
            if not isinstance(record, dict) or set(record) != RANGE_RECORD_KEYS:
                raise G32LaunchError(f"range checkpoint schema/key set drifted: {path}")
            if (
                record["schema"] != "tac.g35_r10_raw_range_checkpoint.v2"
                or record["binding_sha256"] != self.binding_sha256
                or record["kind"] != kind
                or record["first_pair"] != first_pair
                or record["stop_pair"] != stop_pair
                or record["path"] != canonical_path
                or record["predecessor_record_sha256"] != predecessor
                or first_pair != expected_first
                or stop_pair <= first_pair
                or not isinstance(record["telemetry"], dict)
                or not isinstance(record["byte_offset"], int)
                or not isinstance(record["byte_length"], int)
                or record["byte_offset"] < 0
                or record["byte_length"] <= 0
                or not isinstance(record["range_sha256"], str)
                or SHA256_RE.fullmatch(record["range_sha256"]) is None
            ):
                raise G32LaunchError(f"range checkpoint identity, coordinates, or predecessor drifted: {path}")
            rows.append((path, record, digest))
            expected_first = stop_pair
            predecessor = digest
        return rows

    def lookup(self, kind: str, first_pair: int, stop_pair: int, raw_path: Path) -> Mapping[str, Any] | None:
        rows = self._load_kind_chain(kind, raw_path)
        matching = [row for _path, row, _digest in rows if row["first_pair"] == first_pair]
        if not matching:
            return None
        if len(matching) != 1 or matching[0]["stop_pair"] != stop_pair:
            raise G32LaunchError("range checkpoint requested coordinates drifted")
        record = matching[0]
        if _hash_range(raw_path, record["byte_offset"], record["byte_length"]) != record["range_sha256"]:
            raise G32LaunchError("retained raw range differs from checkpoint")
        return record

    def publish(
        self,
        kind: str,
        first_pair: int,
        stop_pair: int,
        raw_path: Path,
        byte_offset: int,
        byte_length: int,
        telemetry: Mapping[str, Any],
    ) -> Path:
        normalized_telemetry = strict_json_loads(_canonical_json(telemetry))
        if not isinstance(normalized_telemetry, dict):
            raise G32LaunchError("range checkpoint telemetry must normalize to an object")
        chain = self._load_kind_chain(kind, raw_path)
        existing = [row for _path, row, _digest in chain if row["first_pair"] == first_pair]
        if existing:
            if len(existing) != 1 or existing[0]["stop_pair"] != stop_pair:
                raise G32LaunchError("range checkpoint conflicts with retained coordinates")
            return next(path for path, row, _digest in chain if row == existing[0])
        expected_first = 0 if not chain else int(chain[-1][1]["stop_pair"])
        if first_pair != expected_first or stop_pair <= first_pair:
            raise G32LaunchError("range checkpoint publication must extend one contiguous prefix")
        predecessor = None if not chain else chain[-1][2]
        record = {
            "schema": "tac.g35_r10_raw_range_checkpoint.v2",
            "binding_sha256": self.binding_sha256,
            "kind": kind,
            "first_pair": first_pair,
            "stop_pair": stop_pair,
            "path": str(raw_path.absolute()),
            "byte_offset": byte_offset,
            "byte_length": byte_length,
            "range_sha256": _hash_range(raw_path, byte_offset, byte_length),
            "predecessor_record_sha256": predecessor,
            "telemetry": normalized_telemetry,
        }
        raw = _canonical_json(record)
        path = self.root / f"{self._prefix(kind, first_pair, stop_pair)}.{_sha256_bytes(raw)}.json"
        _write_once(path, raw)
        self.lookup(kind, first_pair, stop_pair, raw_path)
        return path


class FitRangeStore:
    """Content-addressed, predecessor-linked per-stage fit-range state."""

    def __init__(self, root: Path, binding_sha256: str):
        self.root = root / "fit_range_checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding_sha256 = binding_sha256

    def load_stage(self, fit_stage: str) -> tuple[Mapping[str, Any], ...]:
        if fit_stage not in FIT_RANGE_STAGES:
            raise G32LaunchError(f"unknown fit range stage: {fit_stage}")
        pattern = re.compile(rf"^{re.escape(fit_stage)}\.(\d{{4}})-(\d{{4}})\.([0-9a-f]{{64}})\.json$")
        expected_first = 0
        predecessor: str | None = None
        rows: list[Mapping[str, Any]] = []
        for path in sorted(self.root.glob(f"{fit_stage}.*.json")):
            match = pattern.fullmatch(path.name)
            if match is None:
                raise G32LaunchError(f"fit range checkpoint filename is not canonical: {path}")
            first_pair, stop_pair = int(match.group(1)), int(match.group(2))
            raw = _stable_read(path)
            record_sha256 = _sha256_bytes(raw)
            record = strict_json_loads(raw)
            payload = record.get("payload") if isinstance(record, dict) else None
            if (
                record_sha256 != match.group(3)
                or not isinstance(record, dict)
                or set(record) != FIT_RANGE_RECORD_KEYS
                or record["schema"] != "tac.g35_r10_fit_range_checkpoint.v1"
                or record["binding_sha256"] != self.binding_sha256
                or record["fit_stage"] != fit_stage
                or record["first_pair"] != first_pair
                or record["stop_pair"] != stop_pair
                or first_pair != expected_first
                or stop_pair <= first_pair
                or record["predecessor_record_sha256"] != predecessor
                or not isinstance(payload, dict)
                or record["payload_sha256"] != _sha256_bytes(_canonical_json(payload))
            ):
                raise G32LaunchError(f"fit range identity, coordinates, predecessor, or payload drifted: {path}")
            rows.append({"first_pair": first_pair, "stop_pair": stop_pair, "payload": payload})
            expected_first = stop_pair
            predecessor = record_sha256
        return tuple(rows)

    def load_all(self) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
        return {stage: rows for stage in FIT_RANGE_STAGES if (rows := self.load_stage(stage))}

    def publish(
        self,
        fit_stage: str,
        first_pair: int,
        stop_pair: int,
        payload: Mapping[str, Any],
    ) -> Path:
        normalized = strict_json_loads(_canonical_json(payload))
        if not isinstance(normalized, dict):
            raise G32LaunchError("fit range payload must normalize to one object")
        chain = self.load_stage(fit_stage)
        for record in chain:
            if record["first_pair"] == first_pair:
                if record["stop_pair"] != stop_pair or record["payload"] != normalized:
                    raise G32LaunchError("fit range conflicts with retained immutable state")
                prefix = f"{fit_stage}.{first_pair:04d}-{stop_pair:04d}."
                matches = sorted(self.root.glob(f"{prefix}*.json"))
                if len(matches) != 1:
                    raise G32LaunchError("fit range retained coordinate has duplicate records")
                return matches[0]
        expected_first = 0 if not chain else int(chain[-1]["stop_pair"])
        if first_pair != expected_first or stop_pair <= first_pair:
            raise G32LaunchError("fit range publication must extend one contiguous prefix")
        predecessor: str | None = None
        if chain:
            previous = sorted(self.root.glob(f"{fit_stage}.{chain[-1]['first_pair']:04d}-{expected_first:04d}.*.json"))
            if len(previous) != 1:
                raise G32LaunchError("fit range predecessor record is ambiguous")
            predecessor = previous[0].name.rsplit(".", 2)[1]
        record = {
            "schema": "tac.g35_r10_fit_range_checkpoint.v1",
            "binding_sha256": self.binding_sha256,
            "fit_stage": fit_stage,
            "first_pair": first_pair,
            "stop_pair": stop_pair,
            "predecessor_record_sha256": predecessor,
            "payload_sha256": _sha256_bytes(_canonical_json(normalized)),
            "payload": normalized,
        }
        raw = _canonical_json(record)
        path = self.root / f"{fit_stage}.{first_pair:04d}-{stop_pair:04d}.{_sha256_bytes(raw)}.json"
        _write_once(path, raw)
        self.load_stage(fit_stage)
        return path


def validate_execution_flags(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("resume_from", "video", "selected_archive", "runtime", "pair_count")
        if getattr(args, name) is None
    ]
    if missing:
        raise G32LaunchError(f"launch is missing required arguments: {missing}")
    if not args.execute_reviewed:
        raise G32LaunchError("execution requires --execute-reviewed")
    if not (1 <= args.pair_count <= 600):
        raise G32LaunchError("--pair-count must be in [1,600]")
    governed_values = (args.governed_claim_job_id, args.governed_claim_platform)
    if args.pair_count == 600 and any(not isinstance(value, str) or not value for value in governed_values):
        raise G32LaunchError("n600 requires exact --governed-claim-job-id and --governed-claim-platform")
    if args.pair_count != 600 and any(value is not None for value in governed_values):
        raise G32LaunchError("governed claim coordinates are only valid for exactly 600 pairs")
    if not (1 <= args.chunk_pairs <= args.pair_count):
        raise G32LaunchError("--chunk-pairs must be in [1,pair-count]")
    if args.reserve_bytes < 0:
        raise G32LaunchError("--reserve-bytes must be nonnegative")


def require_governed_n600_execution(
    args: argparse.Namespace,
    *,
    expected_claim_sha256: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Revalidate the canonical live claim; caller-authored booleans have no authority."""

    if args.pair_count != 600:
        return {"required": False, "verified": False}
    effective_environment = os.environ if environment is None else environment
    if (effective_environment.get(GOVERNED_MARKER_ENV, "") or "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise G32LaunchError(
            "n600 refused: missing active TAC_GOVERNED_ADMISSION marker; direct/raw execution is forbidden"
        )
    from tools import claim_lane_dispatch

    try:
        ledger_raw = _stable_read(CLAIM_LEDGER)
        claims = claim_lane_dispatch._parse_claims(ledger_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise G32LaunchError(f"cannot read canonical governed claim ledger: {exc}") from exc
    latest = claim_lane_dispatch._latest_claims_by_job(claims).get((G32_LANE_ID, args.governed_claim_job_id))
    if latest is None or claim_lane_dispatch._is_terminal(latest.status):
        raise G32LaunchError("n600 refused: no active canonical G32 lane dispatch claim")
    if latest.platform != args.governed_claim_platform:
        raise G32LaunchError("n600 governed claim platform differs from the requested platform")
    now_utc = claim_lane_dispatch._utc_now()
    claim_utc = claim_lane_dispatch._parse_utc(latest.timestamp_utc)
    age_hours = claim_lane_dispatch._claim_age_hours(now_utc, latest)
    if claim_utc is None or claim_utc > now_utc or age_hours is None or age_hours > 24.0:
        raise G32LaunchError("n600 governed claim is malformed, future-dated, or older than 24h")
    claim = dict(latest.__dict__)
    claim_sha256 = _sha256_bytes(_canonical_json(claim))
    if expected_claim_sha256 is not None and claim_sha256 != expected_claim_sha256:
        raise G32LaunchError("n600 governed claim record changed after launch binding")
    return {
        "required": True,
        "verified": True,
        "marker_env": GOVERNED_MARKER_ENV,
        "lane_id": G32_LANE_ID,
        "claim": claim,
        "claim_record_sha256": claim_sha256,
        "claim_age_hours": age_hours,
        "claim_ledger_path": str(CLAIM_LEDGER),
        "claim_ledger_sha256_observed": _sha256_bytes(ledger_raw),
    }


def validate_custody(video: Path, selected_archive: Path, runtime: Path) -> tuple[dict[str, Any], bytes]:
    rows = {
        "source_video": _file_row(video, FROZEN["source_video_sha256"], FROZEN["source_video_bytes"]),
        "selected_archive": _file_row(
            selected_archive,
            FROZEN["selected_archive_sha256"],
            FROZEN["selected_archive_bytes"],
        ),
        "runtime": _file_row(runtime, FROZEN["runtime_sha256"], FROZEN["runtime_bytes"]),
        "r10_receiver": _file_row(R10_MODULE, FROZEN["r10_module_sha256"], FROZEN["r10_module_bytes"]),
        "g22_full_n600_receipt": _file_row(
            G22_RECEIPT,
            FROZEN["g22_receipt_sha256"],
            FROZEN["g22_receipt_bytes"],
        ),
    }
    archive_raw = _stable_read(selected_archive)
    try:
        with zipfile.ZipFile(selected_archive, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "0.bin" or infos[0].is_dir():
                raise G32LaunchError("selected archive must contain exactly one regular 0.bin")
            member = archive.read(infos[0])
    except zipfile.BadZipFile as exc:
        raise G32LaunchError("selected archive failed exact ZIP reopen") from exc
    if _sha256_bytes(archive_raw) != rows["selected_archive"]["sha256"]:
        raise G32LaunchError("selected archive drifted across reopen")
    if len(member) != FROZEN["selected_member_bytes"] or _sha256_bytes(member) != FROZEN["selected_member_sha256"]:
        raise G32LaunchError("selected member identity drifted")
    rows["selected_member"] = {
        "member_name": "0.bin",
        "bytes": len(member),
        "sha256": _sha256_bytes(member),
    }
    return rows, member


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    return environment


def _load_private_runtime(runtime: Path, tag: str) -> Any:
    spec = importlib.util.spec_from_file_location(tag, runtime)
    if spec is None or spec.loader is None:
        raise G32LaunchError("cannot load frozen selected runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "_setup", None)) or not callable(getattr(module, "_render_pair", None)):
        raise G32LaunchError("frozen runtime lacks selected private receiver entrypoints")
    if getattr(module, "_FP32", True) is not False:
        raise G32LaunchError("selected realization requires portable FP64 runtime")
    return module


def _run_internal_runtime_plan(plan_path: Path) -> int:
    plan_raw = _stable_read(plan_path)
    plan_digest = _sha256_bytes(plan_raw)
    if plan_path.name != f"selected_population.{plan_digest}.json":
        raise G32LaunchError("internal runtime plan filename/content digest drifted")
    plan = strict_json_loads(plan_raw)
    if (
        not isinstance(plan, dict)
        or set(plan) != INTERNAL_RUNTIME_PLAN_KEYS
        or plan.get("schema") != "tac.g35_r10_internal_runtime_plan.v2"
    ):
        raise G32LaunchError("internal runtime plan schema drifted")
    binding_sha256 = plan["binding_sha256"]
    if not isinstance(binding_sha256, str) or SHA256_RE.fullmatch(binding_sha256) is None:
        raise G32LaunchError("internal runtime plan binding is not canonical SHA-256")
    run_root = Path(plan["run_root"])
    runtime = Path(plan["runtime"])
    member = Path(plan["member"])
    output = Path(plan["output"])
    pair_count = plan["pair_count"]
    chunk_pairs = plan["chunk_pairs"]
    first_pair = plan["first_pair"]
    if (
        type(pair_count) is not int
        or type(chunk_pairs) is not int
        or type(first_pair) is not int
        or not (1 <= pair_count <= 600)
        or not (1 <= chunk_pairs <= pair_count)
        or not (0 <= first_pair < pair_count)
        or first_pair % chunk_pairs != 0
    ):
        raise G32LaunchError("internal runtime population/range coordinates drifted")
    governed_args = argparse.Namespace(
        pair_count=pair_count,
        governed_claim_job_id=plan["governed_claim_job_id"],
        governed_claim_platform=plan["governed_claim_platform"],
    )
    governed_context = require_governed_n600_execution(
        governed_args,
        expected_claim_sha256=plan["governed_claim_sha256"],
    )
    _file_row(runtime, FROZEN["runtime_sha256"], FROZEN["runtime_bytes"])
    _file_row(member, FROZEN["selected_member_sha256"], FROZEN["selected_member_bytes"])
    for name, value in THREAD_ENV.items():
        if os.environ.get(name) != value:
            raise G32LaunchError(f"internal runtime environment drifted: {name}")
    process_started = time.monotonic()
    module = _load_private_runtime(runtime, f"g35_runtime_{os.getpid()}_{time.time_ns()}")
    setup_started = time.monotonic()
    module._setup(str(member))
    setup_wall_seconds = time.monotonic() - setup_started
    metadata = {
        "n_pairs": int(module._G["m"]["n_pairs"]),
        "render_height": int(module._G["rh"]),
        "render_width": int(module._G["rw"]),
        "camera_height": int(module._G["ch"]),
        "camera_width": int(module._G["cw"]),
        "frame_bytes": int(module._G["framebytes"]),
    }
    if metadata != plan["receiver_metadata"]:
        raise G32LaunchError(f"selected runtime geometry drifted: {metadata}")
    pair_bytes = 2 * HEIGHT * WIDTH * CHANNELS
    if output.stat().st_size != pair_count * pair_bytes:
        raise G32LaunchError("internal runtime output byte extent drifted")
    chunk_store = ChunkStore(run_root, binding_sha256)
    module._G["dst"] = str(output)
    completed_ranges = 0
    for first in range(first_pair, pair_count, chunk_pairs):
        stop = min(pair_count, first + chunk_pairs)
        if chunk_store.lookup("selected", first, stop, output) is not None:
            raise G32LaunchError("internal runtime tail starts inside an already-published range")
        range_started = time.monotonic()
        for pair_id in range(first, stop):
            if module._render_pair(pair_id) != pair_id:
                raise G32LaunchError("selected runtime returned a different pair coordinate")
        output_fd = os.open(output, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        chunk_store.publish(
            "selected",
            first,
            stop,
            output,
            first * pair_bytes,
            (stop - first) * pair_bytes,
            {
                "process_plan_sha256": plan_digest,
                "process_first_pair": first_pair,
                "setup_wall_seconds": setup_wall_seconds,
                "range_wall_seconds": time.monotonic() - range_started,
                "one_setup_for_population_tail": True,
                "governed_claim_sha256": governed_context.get("claim_record_sha256"),
                "claim_ledger_sha256_observed": governed_context.get("claim_ledger_sha256_observed"),
            },
        )
        completed_ranges += 1
    print(
        json.dumps(
            {
                "receiver_metadata": metadata,
                "first_pair": first_pair,
                "stop_pair": pair_count,
                "completed_ranges": completed_ranges,
                "setup_calls": 1,
                "setup_wall_seconds": setup_wall_seconds,
                "wall_seconds": time.monotonic() - process_started,
            },
            sort_keys=True,
        )
    )
    return 0


def _subprocess_json(command: Sequence[str], *, environment: Mapping[str, str] | None = None) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=None if environment is None else dict(environment),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise G32LaunchError(
            f"subprocess failed: argv={list(command)!r} stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-4000:]!r}"
        )
    try:
        value = strict_json_loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, R10BoundedInverseError) as exc:
        raise G32LaunchError("subprocess emitted no strict JSON receipt") from exc
    if not isinstance(value, dict):
        raise G32LaunchError("subprocess receipt root is not an object")
    return value


def materialize_selected_base(
    raw_path: Path,
    runtime: Path,
    member: Path,
    *,
    pair_count: int,
    chunk_pairs: int,
    run_root: Path,
    chunk_store: ChunkStore,
    governed_context: Mapping[str, Any] | None = None,
) -> None:
    pair_bytes = 2 * HEIGHT * WIDTH * CHANNELS
    expected_bytes = pair_count * pair_bytes
    _preallocate(raw_path, expected_bytes)
    metadata = {
        "n_pairs": 600,
        "render_height": 384,
        "render_width": 512,
        "camera_height": HEIGHT,
        "camera_width": WIDTH,
        "frame_bytes": HEIGHT * WIDTH * CHANNELS,
    }
    first_missing: int | None = None
    for first in range(0, pair_count, chunk_pairs):
        stop = min(pair_count, first + chunk_pairs)
        existing = chunk_store.lookup("selected", first, stop, raw_path)
        if existing is None and first_missing is None:
            first_missing = first
        elif existing is not None and first_missing is not None:
            raise G32LaunchError("selected range chain contains state after its first missing range")
    if first_missing is None:
        return
    plan = {
        "schema": "tac.g35_r10_internal_runtime_plan.v2",
        "binding_sha256": chunk_store.binding_sha256,
        "run_root": str(run_root),
        "runtime": str(runtime),
        "member": str(member),
        "output": str(raw_path),
        "pair_count": pair_count,
        "chunk_pairs": chunk_pairs,
        "first_pair": first_missing,
        "receiver_metadata": metadata,
        "governed_claim_job_id": None
        if governed_context is None
        else governed_context.get("claim", {}).get("instance_job_id"),
        "governed_claim_platform": None
        if governed_context is None
        else governed_context.get("claim", {}).get("platform"),
        "governed_claim_sha256": None if governed_context is None else governed_context.get("claim_record_sha256"),
    }
    plan_raw = _canonical_json(plan)
    plan_path = run_root / "plans" / f"selected_population.{_sha256_bytes(plan_raw)}.json"
    _write_once(plan_path, plan_raw)
    receipt = _subprocess_json(
        [sys.executable, str(Path(__file__).resolve()), "--internal-runtime-plan", str(plan_path)],
        environment=_worker_environment(),
    )
    if receipt.get("setup_calls") != 1 or receipt.get("first_pair") != first_missing:
        raise G32LaunchError("selected runtime tail receipt does not prove exactly one setup")
    for first in range(0, pair_count, chunk_pairs):
        stop = min(pair_count, first + chunk_pairs)
        if chunk_store.lookup("selected", first, stop, raw_path) is None:
            raise G32LaunchError("selected runtime tail returned without a complete range chain")


def materialize_source_pairs(
    raw_path: Path,
    video: Path,
    *,
    pair_count: int,
    chunk_pairs: int,
    chunk_store: ChunkStore,
) -> None:
    """Decode the source population once while publishing immutable ranges.

    A per-range ``ffmpeg -i ... select=between(n,...)`` loop is not merely
    process overhead: without input seeking it decodes the source prefix again
    for every range.  At n600/chunk_pairs=2 that is 180,600 decoded frames for
    a 1,200-frame source.  The single-pass stream below preserves the same
    range-level resume boundary while making source work linear in population
    size.  On resume, already-published ranges are streamed again only once and
    compared byte-for-byte by digest before later missing ranges are admitted.
    """

    pair_bytes = 2 * HEIGHT * WIDTH * CHANNELS
    expected_bytes = pair_count * pair_bytes
    _preallocate(raw_path, expected_bytes)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise G32LaunchError("ffmpeg is required for encoder-only source extraction")

    ranges: list[tuple[int, int, Mapping[str, Any] | None]] = []
    for first in range(0, pair_count, chunk_pairs):
        stop = min(pair_count, first + chunk_pairs)
        ranges.append((first, stop, chunk_store.lookup("source", first, stop, raw_path)))
    if all(existing is not None for _first, _stop, existing in ranges):
        return

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-frames:v",
        str(2 * pair_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    process_started = time.monotonic()
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        if process.stdout is None:
            process.terminate()
            process.wait()
            raise G32LaunchError("single-pass source decoder has no stdout stream")
        raw_fd = os.open(raw_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        try:
            for first, stop, existing in ranges:
                range_started = time.monotonic()
                expected_chunk = (stop - first) * pair_bytes
                remaining = expected_chunk
                cursor = first * pair_bytes
                digest = hashlib.sha256()
                while remaining:
                    block = process.stdout.read(min(8 << 20, remaining))
                    if not block:
                        raise G32LaunchError(
                            f"single-pass source extraction ended inside pairs [{first},{stop}): missing={remaining}"
                        )
                    digest.update(block)
                    if existing is None:
                        view = memoryview(block)
                        while view:
                            count = os.pwrite(raw_fd, view, cursor)
                            if count <= 0:
                                raise G32LaunchError(f"short source range write: {raw_path}")
                            view = view[count:]
                            cursor += count
                    remaining -= len(block)
                range_sha256 = digest.hexdigest()
                if existing is not None:
                    if range_sha256 != existing.get("range_sha256"):
                        raise G32LaunchError(f"single-pass source replay differs from retained pairs [{first},{stop})")
                    continue
                os.fsync(raw_fd)
                chunk_store.publish(
                    "source",
                    first,
                    stop,
                    raw_path,
                    first * pair_bytes,
                    expected_chunk,
                    {
                        "argv": command,
                        "decoder_process_mode": "ONE_LINEAR_STREAM_FOR_COMPLETE_POPULATION",
                        "population_pair_count": pair_count,
                        "range_wall_seconds": time.monotonic() - range_started,
                        "stdout_sha256": range_sha256,
                    },
                )
            if process.stdout.read(1):
                raise G32LaunchError("single-pass source decoder emitted bytes beyond the requested population")
            returncode = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
            if returncode != 0:
                raise G32LaunchError(
                    f"single-pass source extraction failed: return={returncode} "
                    f"wall_seconds={time.monotonic() - process_started:.6f} stderr={stderr[-2000:]!r}"
                )
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            os.close(raw_fd)


def cleanup_certify_or_block(
    run_root: Path,
    paths: Sequence[Path],
    *,
    rebuild_command: Sequence[str],
    custody: Mapping[str, Any],
    preserve_scratch: bool,
) -> Mapping[str, Any]:
    artifact_root = run_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    requested_paths = list(dict.fromkeys(str(path.absolute()) for path in paths))
    existing_intents = sorted(artifact_root.glob("cleanup_intent.*.json"))
    if len(existing_intents) > 1:
        raise G32LaunchError("cleanup blocked because multiple immutable intents exist")
    if existing_intents:
        intent_path = existing_intents[0]
        intent_raw = _stable_read(intent_path)
        intent_sha256 = _sha256_bytes(intent_raw)
        if intent_path.name != f"cleanup_intent.{intent_sha256}.json":
            raise G32LaunchError("cleanup intent filename/content digest drifted")
        intent = strict_json_loads(intent_raw)
        if (
            not isinstance(intent, dict)
            or intent.get("schema") != "tac.g35_r10_cleanup_intent.v2"
            or intent.get("requested_paths") != requested_paths
            or not isinstance(intent.get("artifacts"), list)
        ):
            raise G32LaunchError("cleanup intent identity or requested path set drifted")
        rows = intent["artifacts"]
    else:
        rows: list[dict[str, Any]] = []
        for path_text in requested_paths:
            absolute = Path(path_text)
            if not absolute.exists():
                raise G32LaunchError(
                    f"cleanup cannot certify an absent scratch path without a prior intent: {absolute}"
                )
            metadata = absolute.lstat()
            if absolute.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise G32LaunchError(f"cleanup blocked on non-regular scratch: {absolute}")
            rows.append(
                {
                    "original_path": str(absolute),
                    "bytes": metadata.st_size,
                    "sha256": _sha256_file_streaming(absolute),
                    "cold_store_destination": None,
                    "false_authority_flags": {
                        "candidate": False,
                        "score": False,
                        "promotion": False,
                        "research_only": True,
                    },
                    "rebuildable_reason": "deterministic exact source extraction or selected/runtime replay",
                }
            )
        ffmpeg = shutil.which("ffmpeg")
        ffmpeg_path = None if ffmpeg is None else Path(ffmpeg).resolve()
        intent = {
            "schema": "tac.g35_r10_cleanup_intent.v2",
            "certify_or_block": True,
            "requested_paths": requested_paths,
            "artifacts": rows,
            "rebuild": {
                "argv": list(rebuild_command),
                "environment": THREAD_ENV,
                "source_sha256": custody["source_video"]["sha256"],
                "runtime_sha256": custody["runtime"]["sha256"],
                "selected_archive_sha256": custody["selected_archive"]["sha256"],
                "r10_receiver_sha256": custody["r10_receiver"]["sha256"],
                "g22_receipt_sha256": custody["g22_full_n600_receipt"]["sha256"],
                "launcher_sha256": _sha256_file_streaming(Path(__file__).resolve()),
                "fitter_sha256": _sha256_file_streaming(FITTER_MODULE),
                "ffmpeg_path": None if ffmpeg_path is None else str(ffmpeg_path),
                "ffmpeg_sha256": None if ffmpeg_path is None else _sha256_file_streaming(ffmpeg_path),
            },
        }
        intent_raw = _canonical_json(intent)
        intent_sha256 = _sha256_bytes(intent_raw)
        intent_path = artifact_root / f"cleanup_intent.{intent_sha256}.json"
        _write_once(intent_path, intent_raw)

    absent_before: list[str] = []
    present_before: list[str] = []
    for row in rows:
        target = Path(row["original_path"])
        if target.exists():
            if target.is_symlink() or _sha256_file_streaming(target) != row["sha256"]:
                raise G32LaunchError(f"cleanup blocked because retained scratch drifted: {target}")
            present_before.append(str(target))
        else:
            absent_before.append(str(target))
    if preserve_scratch and absent_before:
        raise G32LaunchError("cleanup preserve mode cannot certify scratch already absent after a deletion intent")
    removed: list[str] = []
    if not preserve_scratch:
        for path_text in present_before:
            target = Path(path_text)
            target.unlink()
            removed.append(path_text)
    outcome = {
        "schema": "tac.g35_r10_cleanup_certificate.v2",
        "intent_path": str(intent_path),
        "intent_sha256": intent_sha256,
        "preserve_scratch": preserve_scratch,
        "certified_artifacts": len(rows),
        "removed_paths": sorted([*absent_before, *removed]) if not preserve_scratch else [],
        "preserved_paths": sorted(present_before) if preserve_scratch else [],
    }
    outcome_raw = _canonical_json(outcome)
    certificate_path = artifact_root / f"cleanup_certificate.{_sha256_bytes(outcome_raw)}.json"
    _write_once(certificate_path, outcome_raw)
    return {
        "certificate_path": str(certificate_path),
        "certificate_sha256": _sha256_bytes(outcome_raw),
        **{
            key: outcome[key]
            for key in ("intent_path", "intent_sha256", "certified_artifacts", "removed_paths", "preserved_paths")
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--selected-archive", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--pair-count", type=int)
    parser.add_argument("--chunk-pairs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--reserve-bytes", type=int, default=2 << 30)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--governed-claim-job-id")
    parser.add_argument("--governed-claim-platform")
    parser.add_argument("--preserve-scratch", action="store_true")
    parser.add_argument("--allow-local-test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-runtime-plan", type=Path, help=argparse.SUPPRESS)
    return parser


def _binding(
    args: argparse.Namespace,
    custody: Mapping[str, Any],
    governed_context: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    value = {
        "schema": "tac.g32_r10_launch_binding.v1",
        "seed": args.seed,
        "pair_count": args.pair_count,
        "height": HEIGHT,
        "width": WIDTH,
        "chunk_pairs": args.chunk_pairs,
        "sample_stride": args.sample_stride,
        "custody": custody,
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": _sha256_file_streaming(Path(__file__).resolve()),
        "fitter_module_path": str(FITTER_MODULE),
        "fitter_module_sha256": _sha256_file_streaming(FITTER_MODULE),
        "governed_claim": (
            None
            if not governed_context.get("required")
            else {
                "lane_id": governed_context["lane_id"],
                "record": governed_context["claim"],
                "record_sha256": governed_context["claim_record_sha256"],
            }
        ),
    }
    return _sha256_bytes(_canonical_json(value)), value


def _verify_retained_result_artifacts(
    run_root: Path,
    stage_130: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path, Path, Path]:
    if set(stage_130) != {"executed", "receipt_path", "receipt_sha256", "blocker"}:
        raise G32LaunchError("retained bounded-decode stage key set drifted")
    receipt_path = Path(stage_130["receipt_path"])
    if not _path_is_under(receipt_path, (run_root / "artifacts",)):
        raise G32LaunchError("retained receipt escaped the run artifact root")
    receipt_raw = _stable_read(receipt_path)
    receipt_sha256 = _sha256_bytes(receipt_raw)
    if stage_130["receipt_sha256"] != receipt_sha256 or receipt_path.name != f"receipt.{receipt_sha256}.json":
        raise G32LaunchError("retained receipt filename/content identity drifted")
    receipt = strict_json_loads(receipt_raw)
    if (
        not isinstance(receipt, dict)
        or not isinstance(receipt.get("artifact_paths"), dict)
        or set(receipt["artifact_paths"]) != {"packet", "wrapper_zip"}
    ):
        raise G32LaunchError("retained receipt root or artifact map drifted")
    packet_path = Path(receipt["artifact_paths"]["packet"])
    wrapper_path = Path(receipt["artifact_paths"]["wrapper_zip"])
    for artifact in (packet_path, wrapper_path):
        if not _path_is_under(artifact, (run_root / "artifacts",)):
            raise G32LaunchError("retained packet artifact escaped the run artifact root")
    packet_raw = _stable_read(packet_path)
    wrapper_raw = _stable_read(wrapper_path)
    packet_sha256 = _sha256_bytes(packet_raw)
    wrapper_sha256 = _sha256_bytes(wrapper_raw)
    packet_receipt = receipt.get("packet")
    wrapper_receipt = receipt.get("wrapper_zip")
    if (
        not isinstance(packet_receipt, dict)
        or not isinstance(wrapper_receipt, dict)
        or packet_path.name != f"r10_packet.{packet_sha256}.bin"
        or packet_receipt.get("bytes") != len(packet_raw)
        or packet_receipt.get("sha256") != packet_sha256
        or wrapper_path.name != f"r10_packet_wrapper.{wrapper_sha256}.zip"
        or wrapper_receipt.get("bytes") != len(wrapper_raw)
        or wrapper_receipt.get("sha256") != wrapper_sha256
        or wrapper_receipt.get("member") != "r10.packet"
    ):
        raise G32LaunchError("retained packet/wrapper bytes differ from their receipt")
    if serialize_r10_packet(parse_r10_packet(packet_raw)) != packet_raw:
        raise G32LaunchError("retained packet failed strict parse/re-emission")
    try:
        with zipfile.ZipFile(wrapper_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "r10.packet" or archive.read(infos[0]) != packet_raw:
                raise G32LaunchError("retained wrapper does not contain the exact packet once")
    except zipfile.BadZipFile as exc:
        raise G32LaunchError("retained wrapper ZIP failed strict reopen") from exc
    return receipt, receipt_path, packet_path, wrapper_path


def _verify_retained_cleanup(run_root: Path, cleanup_stage: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "certificate_path",
        "certificate_sha256",
        "intent_path",
        "intent_sha256",
        "certified_artifacts",
        "removed_paths",
        "preserved_paths",
    }
    if set(cleanup_stage) != required:
        raise G32LaunchError("retained cleanup stage key set drifted")
    certificate_path = Path(cleanup_stage["certificate_path"])
    intent_path = Path(cleanup_stage["intent_path"])
    if not all(_path_is_under(path, (run_root / "artifacts",)) for path in (certificate_path, intent_path)):
        raise G32LaunchError("retained cleanup evidence escaped the run artifact root")
    certificate_raw = _stable_read(certificate_path)
    intent_raw = _stable_read(intent_path)
    certificate_sha256 = _sha256_bytes(certificate_raw)
    intent_sha256 = _sha256_bytes(intent_raw)
    if (
        certificate_path.name != f"cleanup_certificate.{certificate_sha256}.json"
        or intent_path.name != f"cleanup_intent.{intent_sha256}.json"
        or cleanup_stage["certificate_sha256"] != certificate_sha256
        or cleanup_stage["intent_sha256"] != intent_sha256
    ):
        raise G32LaunchError("retained cleanup evidence filename/content identity drifted")
    certificate = strict_json_loads(certificate_raw)
    intent = strict_json_loads(intent_raw)
    if (
        not isinstance(certificate, dict)
        or certificate.get("schema") != "tac.g35_r10_cleanup_certificate.v2"
        or certificate.get("intent_sha256") != intent_sha256
        or not isinstance(intent, dict)
        or intent.get("schema") != "tac.g35_r10_cleanup_intent.v2"
    ):
        raise G32LaunchError("retained cleanup intent/certificate schema or linkage drifted")
    expected_stage = {
        "certificate_path": str(certificate_path),
        "certificate_sha256": certificate_sha256,
        **{
            key: certificate[key]
            for key in ("intent_path", "intent_sha256", "certified_artifacts", "removed_paths", "preserved_paths")
        },
    }
    if expected_stage != cleanup_stage:
        raise G32LaunchError("retained cleanup stage differs from exact certificate fields")
    artifacts_by_path = {row["original_path"]: row for row in intent["artifacts"]}
    for path_text in certificate["removed_paths"]:
        if path_text not in artifacts_by_path or Path(path_text).exists():
            raise G32LaunchError("retained cleanup removed-path proof is false")
    for path_text in certificate["preserved_paths"]:
        row = artifacts_by_path.get(path_text)
        if row is None or not Path(path_text).exists() or _sha256_file_streaming(Path(path_text)) != row["sha256"]:
            raise G32LaunchError("retained cleanup preserved-path proof is false")
    return expected_stage


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    validate_execution_flags(args)
    governed_context = require_governed_n600_execution(args)
    run_root = validate_run_root(args.resume_from, allow_local_test_only=args.allow_local_test_only)
    custody, selected_member = validate_custody(args.video, args.selected_archive, args.runtime)
    binding_sha256, binding = _binding(args, custody, governed_context)
    stages = ImmutableStageStore(run_root, binding_sha256)
    chunks = ChunkStore(run_root, binding_sha256)
    fit_ranges = FitRangeStore(run_root, binding_sha256)
    completed = stages.load_prefix()
    if "140_cleanup_certificate" in completed:
        stage_130 = completed["130_bounded_decode_receipt"]
        _receipt, receipt_path, packet_path, wrapper_path = _verify_retained_result_artifacts(run_root, stage_130)
        cleanup = _verify_retained_cleanup(run_root, completed["140_cleanup_certificate"])
        return {
            "schema": "tac.g32_r10_launch_result.v1",
            "run_root": str(run_root),
            "packet_path": str(packet_path),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": _sha256_file_streaming(packet_path),
            "wrapper_zip_path": str(wrapper_path),
            "wrapper_zip_bytes": wrapper_path.stat().st_size,
            "wrapper_zip_sha256": _sha256_file_streaming(wrapper_path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": stage_130["receipt_sha256"],
            "cleanup": cleanup,
            "pointer_delta": False,
            "score_claim": None,
            "authority_blocker": G32_AUTHORITY_BLOCKER,
            "resumed_complete": True,
        }
    if "130_bounded_decode_receipt" in completed:
        stage_130 = completed["130_bounded_decode_receipt"]
        receipt, receipt_path, packet_path, wrapper_path = _verify_retained_result_artifacts(run_root, stage_130)
        cleanup = cleanup_certify_or_block(
            run_root,
            [Path(path) for path in receipt.get("scratch_paths", [])],
            rebuild_command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            custody=custody,
            preserve_scratch=args.preserve_scratch,
        )
        stages.publish("140_cleanup_certificate", cleanup)
        return {
            "schema": "tac.g32_r10_launch_result.v1",
            "run_root": str(run_root),
            "packet_path": str(packet_path),
            "packet_bytes": packet_path.stat().st_size,
            "packet_sha256": _sha256_file_streaming(packet_path),
            "wrapper_zip_path": str(wrapper_path),
            "wrapper_zip_bytes": wrapper_path.stat().st_size,
            "wrapper_zip_sha256": _sha256_file_streaming(wrapper_path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": stage_130["receipt_sha256"],
            "cleanup": cleanup,
            "pointer_delta": False,
            "score_claim": None,
            "authority_blocker": G32_AUTHORITY_BLOCKER,
            "resumed_cleanup": True,
        }
    storage = storage_preflight(run_root, pair_count=args.pair_count, reserve_bytes=args.reserve_bytes)
    if "000_custody" not in completed:
        stages.publish("000_custody", {"binding": binding, "storage": storage})

    inputs = run_root / "inputs"
    member_path = inputs / f"selected_0bin.{FROZEN['selected_member_sha256']}.bin"
    _write_once(member_path, selected_member)
    raw_bytes = args.pair_count * 2 * HEIGHT * WIDTH * CHANNELS
    selected_raw = run_root / "scratch" / "selected_pairs.rgb24"
    source_raw = run_root / "scratch" / "source_pairs.rgb24"
    materialize_selected_base(
        selected_raw,
        args.runtime,
        member_path,
        pair_count=args.pair_count,
        chunk_pairs=args.chunk_pairs,
        run_root=run_root,
        chunk_store=chunks,
        governed_context=governed_context if governed_context.get("required") else None,
    )
    governed_context = require_governed_n600_execution(
        args,
        expected_claim_sha256=governed_context.get("claim_record_sha256"),
    )
    materialize_source_pairs(
        source_raw,
        args.video,
        pair_count=args.pair_count,
        chunk_pairs=args.chunk_pairs,
        chunk_store=chunks,
    )
    governed_context = require_governed_n600_execution(
        args,
        expected_claim_sha256=governed_context.get("claim_record_sha256"),
    )
    base_realization = streaming_realization_sha256(
        selected_raw,
        pair_count=args.pair_count,
        height=HEIGHT,
        width=WIDTH,
    )
    if args.pair_count == 600 and base_realization != FROZEN["full_n600_base_realization_sha256"]:
        raise G32LaunchError("full-n600 selected realization differs from frozen G22 equality receipt")
    selected_state = {
        "selected_raw": {"path": str(selected_raw), "bytes": raw_bytes, "realization_sha256": base_realization},
        "source_raw": {"path": str(source_raw), "bytes": raw_bytes, "sha256": _sha256_file_streaming(source_raw)},
        "pair_count": args.pair_count,
    }
    if "010_selected_base" not in completed:
        stages.publish("010_selected_base", selected_state)
    elif completed["010_selected_base"] != selected_state:
        raise G32LaunchError("retained selected/source population differs from immutable stage")
    base = np.memmap(
        selected_raw,
        mode="r",
        dtype=np.uint8,
        shape=(args.pair_count, 2, HEIGHT, WIDTH, CHANNELS),
        order="C",
    )
    source = np.memmap(
        source_raw,
        mode="r",
        dtype=np.uint8,
        shape=(args.pair_count, 2, HEIGHT, WIDTH, CHANNELS),
        order="C",
    )
    config = R10BoundedInverseConfigV1(
        seed=args.seed,
        pair_count=args.pair_count,
        height=HEIGHT,
        width=WIDTH,
        chunk_pairs=args.chunk_pairs,
        sample_stride=args.sample_stride,
        execute_reviewed=args.execute_reviewed,
    )
    resumed = stages.load_prefix()

    def checkpoint(stage: str, payload: Mapping[str, Any]) -> None:
        require_governed_n600_execution(
            args,
            expected_claim_sha256=governed_context.get("claim_record_sha256"),
        )
        stages.publish(stage, payload)

    def range_checkpoint(stage: str, first_pair: int, stop_pair: int, payload: Mapping[str, Any]) -> None:
        require_governed_n600_execution(
            args,
            expected_claim_sha256=governed_context.get("claim_record_sha256"),
        )
        fit_ranges.publish(stage, first_pair, stop_pair, payload)

    result = compile_r10_bounded_inverse(
        base,
        source,
        source_sha256=FROZEN["source_video_sha256"],
        config=config,
        base_realization_sha256=base_realization,
        execute_bounded_decode=args.pair_count <= 2,
        scratch_directory=run_root / "scratch",
        resume_stages=resumed,
        stage_callback=checkpoint,
        resume_ranges=fit_ranges.load_all(),
        range_callback=range_checkpoint,
        governance_callback=(
            None
            if args.pair_count != 600
            else lambda: require_governed_n600_execution(
                args,
                expected_claim_sha256=governed_context["claim_record_sha256"],
            )
        ),
    )
    artifact_root = run_root / "artifacts"
    packet_path = artifact_root / f"r10_packet.{hashlib.sha256(result.packet_bytes).hexdigest()}.bin"
    wrapper_path = artifact_root / f"r10_packet_wrapper.{hashlib.sha256(result.wrapper_zip_bytes).hexdigest()}.zip"
    _write_once(packet_path, result.packet_bytes)
    _write_once(wrapper_path, result.wrapper_zip_bytes)
    receipt = result_receipt_dict(result)
    receipt["launch_binding_sha256"] = binding_sha256
    receipt["artifact_paths"] = {"packet": str(packet_path), "wrapper_zip": str(wrapper_path)}
    receipt["scratch_paths"] = [
        str(selected_raw),
        str(source_raw),
        *result.scratch_paths,
    ]
    receipt_raw = _canonical_json(receipt)
    receipt_path = artifact_root / f"receipt.{_sha256_bytes(receipt_raw)}.json"
    _write_once(receipt_path, receipt_raw)
    stages.publish(
        "130_bounded_decode_receipt",
        {
            "executed": args.pair_count <= 2,
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256_bytes(receipt_raw),
            "blocker": None if args.pair_count <= 2 else "BOUNDED_DECODE_SKIPPED_FOR_STREAMING_POPULATION",
        },
    )
    del base, source
    cleanup = cleanup_certify_or_block(
        run_root,
        [selected_raw, source_raw, *(Path(path) for path in result.scratch_paths)],
        rebuild_command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        custody=custody,
        preserve_scratch=args.preserve_scratch,
    )
    stages.publish("140_cleanup_certificate", cleanup)
    return {
        "schema": "tac.g32_r10_launch_result.v1",
        "run_root": str(run_root),
        "packet_path": str(packet_path),
        "packet_bytes": len(result.packet_bytes),
        "packet_sha256": _sha256_bytes(result.packet_bytes),
        "wrapper_zip_path": str(wrapper_path),
        "wrapper_zip_bytes": len(result.wrapper_zip_bytes),
        "wrapper_zip_sha256": _sha256_bytes(result.wrapper_zip_bytes),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_bytes(receipt_raw),
        "cleanup": cleanup,
        "pointer_delta": False,
        "score_claim": None,
        "authority_blocker": G32_AUTHORITY_BLOCKER,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.internal_runtime_plan is not None:
        return _run_internal_runtime_plan(args.internal_runtime_plan)
    try:
        result = run(args)
    except (G32LaunchError, R10BoundedInverseError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FROZEN",
    "STAGES",
    "FitRangeStore",
    "G32LaunchError",
    "ImmutableStageStore",
    "build_parser",
    "cleanup_certify_or_block",
    "main",
    "materialize_selected_base",
    "materialize_source_pairs",
    "run",
    "storage_preflight",
    "validate_custody",
    "validate_execution_flags",
    "validate_run_root",
]
