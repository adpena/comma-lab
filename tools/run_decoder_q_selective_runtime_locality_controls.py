#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run non-authoritative locality controls for a DQS1 selective runtime.

This tool compares three official inflate outputs:

* parent packet
* full-video global decoder-q mutation
* selective DQS1 runtime packet

It does not run the scorer and never emits score authority.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.decoder_q_selective_runtime_packet import (  # noqa: E402
    affected_frames_for_pairs,
    unpack_dqs1_payload,
)

SCHEMA = "decoder_q_selective_runtime_locality_controls.v1"
PRODUCER = "tools/run_decoder_q_selective_runtime_locality_controls.py"
DEFAULT_VIDEO_NAMES_FILE = REPO_ROOT / "upstream" / "public_test_video_names.txt"
DEFAULT_FRAME_COUNT = 1200
INFLATE_MANIFEST_NAME = ".locality_inflate_manifest.json"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SelectiveRuntimeControlError(ValueError):
    """Raised when the locality controls cannot run safely."""


@dataclass(frozen=True)
class InflateTarget:
    label: str
    runtime_dir: Path
    archive_zip: Path
    archive_source: str


class PhaseRecorder:
    def __init__(self, progress_jsonl: Path | None = None) -> None:
        self.phases: list[dict[str, Any]] = []
        self.progress_jsonl = progress_jsonl
        self._lock = threading.Lock()

    def append_event(self, event: dict[str, Any]) -> None:
        if self.progress_jsonl is None:
            return
        row = {
            "schema": "decoder_q_selective_runtime_locality_progress.v1",
            "producer": PRODUCER,
            "written_epoch_seconds": time.time(),
            **event,
        }
        with self._lock:
            self.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    @contextlib.contextmanager
    def timed(self, name: str, **metadata: Any):
        started = time.monotonic()
        record: dict[str, Any] = {
            "name": name,
            "status": "running",
            "started_epoch_seconds": time.time(),
            **metadata,
        }
        with self._lock:
            self.phases.append(record)
        self.append_event({"event": "phase_started", **record})
        try:
            yield record
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            raise
        else:
            record["status"] = "succeeded"
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["ended_epoch_seconds"] = time.time()
            self.append_event({"event": "phase_finished", **record})


class GlobalDeadline:
    def __init__(self, timeout_seconds: int | None) -> None:
        self.timeout_seconds = timeout_seconds
        self.started_monotonic = time.monotonic()
        self.deadline_monotonic = (
            None
            if timeout_seconds is None or timeout_seconds <= 0
            else self.started_monotonic + timeout_seconds
        )

    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def subprocess_timeout(self, requested_seconds: int) -> int:
        if requested_seconds <= 0:
            raise SelectiveRuntimeControlError("--timeout-seconds must be positive")
        remaining = self.remaining_seconds()
        if remaining is None:
            return requested_seconds
        if remaining <= 0:
            raise SelectiveRuntimeControlError(
                f"global timeout expired after {self.timeout_seconds}s"
            )
        return max(1, min(requested_seconds, math.floor(remaining)))

    def check(self, phase: str) -> None:
        remaining = self.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise SelectiveRuntimeControlError(
                f"{phase} exceeded global timeout {self.timeout_seconds}s"
            )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _as_int_list(raw: object, *, label: str) -> list[int]:
    if not isinstance(raw, list):
        raise SelectiveRuntimeControlError(f"{label} must be a list")
    values: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise SelectiveRuntimeControlError(f"{label}[{index}] must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise SelectiveRuntimeControlError(
                f"{label}[{index}] must be an integer"
            ) from exc
        if parsed != value and not (isinstance(value, str) and str(parsed) == value):
            raise SelectiveRuntimeControlError(f"{label}[{index}] must be integral")
        values.append(parsed)
    return values


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise SelectiveRuntimeControlError(f"{label} must keep {key}=false")


def parse_selected_pairs(raw: str) -> list[int]:
    pairs: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            pair = int(token)
        except ValueError as exc:
            raise SelectiveRuntimeControlError(
                f"selected pair token is not an integer: {token!r}"
            ) from exc
        if pair < 0:
            raise SelectiveRuntimeControlError(f"selected pair must be non-negative: {pair}")
        pairs.append(pair)
    if not pairs:
        raise SelectiveRuntimeControlError("at least one selected pair is required")
    if len(set(pairs)) != len(pairs):
        raise SelectiveRuntimeControlError("selected pairs contain duplicates")
    return sorted(pairs)


def selected_frame_indices_for_pairs(
    selected_pairs: list[int],
    *,
    frame_policy: str,
) -> list[int]:
    return affected_frames_for_pairs(selected_pairs, frame_policy=frame_policy)


def raw_relpath_for_video_name(video_name: str) -> Path:
    clean = video_name.strip()
    if not clean:
        raise SelectiveRuntimeControlError("empty video name in file list")
    rel = Path(clean).with_suffix(".raw")
    if rel.is_absolute() or ".." in rel.parts:
        raise SelectiveRuntimeControlError(f"unsafe video name in file list: {video_name!r}")
    return rel


def read_video_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise SelectiveRuntimeControlError(f"video names file is empty: {path}")
    return names


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelectiveRuntimeControlError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def read_single_stored_member_payload(archive_zip: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_zip) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise SelectiveRuntimeControlError(
                f"{archive_zip}: expected one ZIP member, found {len(infos)}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise SelectiveRuntimeControlError(
                f"{archive_zip}: member {info.filename!r} is not ZIP_STORED"
            )
        return info.filename, zf.read(info.filename)


def parse_dqs1_tail_from_archive(archive_zip: Path) -> dict[str, Any]:
    member_name, member_data = read_single_stored_member_payload(archive_zip)
    min_tail_len = 11
    max_tail_len = min(len(member_data), min_tail_len + 600 * 2)
    matches: list[tuple[int, dict[str, Any]]] = []
    for tail_len in range(min_tail_len, max_tail_len + 1):
        start = len(member_data) - tail_len
        if member_data[start : start + 4] != b"DQS1":
            continue
        try:
            parsed = unpack_dqs1_payload(member_data[start:])
        except ValueError:
            continue
        matches.append((start, parsed))
    if len(matches) != 1:
        raise SelectiveRuntimeControlError(
            f"{archive_zip}: expected exactly one DQS1 member-tail payload, "
            f"found {len(matches)}"
        )
    tail_offset, parsed = matches[0]
    tail = member_data[tail_offset:]
    return {
        "member_name": member_name,
        "tail_offset": tail_offset,
        "tail_bytes": len(tail),
        "tail_sha256": sha256_bytes(tail),
        **parsed,
    }


def validate_selective_runtime_contract(
    *,
    selective_submission_dir: Path,
    selected_pairs: list[int],
    frame_policy: str,
) -> dict[str, Any]:
    """Cross-check CLI locality expectations against manifest and archive bytes."""

    manifest_path = selective_submission_dir / "selective_runtime_manifest.json"
    if not manifest_path.is_file():
        raise SelectiveRuntimeControlError(
            f"selective runtime manifest missing: {manifest_path}"
        )
    manifest = read_json_object(manifest_path)
    _require_false_authority(manifest, label="selective runtime manifest")
    manifest_dqs1 = manifest.get("dqs1_payload")
    if not isinstance(manifest_dqs1, dict):
        raise SelectiveRuntimeControlError("selective runtime manifest dqs1_payload missing")

    archive_dqs1 = parse_dqs1_tail_from_archive(_submission_archive(selective_submission_dir))
    manifest_pairs = _as_int_list(
        manifest_dqs1.get("pair_indices"),
        label="selective runtime manifest dqs1_payload.pair_indices",
    )
    manifest_frames = _as_int_list(
        manifest_dqs1.get("affected_frame_indices"),
        label="selective runtime manifest dqs1_payload.affected_frame_indices",
    )
    archive_pairs = _as_int_list(
        archive_dqs1.get("pair_indices"),
        label="archive DQS1 pair_indices",
    )
    expected_frames = selected_frame_indices_for_pairs(
        selected_pairs,
        frame_policy=frame_policy,
    )
    checks = {
        "manifest frame_policy": manifest_dqs1.get("frame_policy"),
        "archive frame_policy": archive_dqs1.get("frame_policy"),
    }
    for label, value in checks.items():
        if value != frame_policy:
            raise SelectiveRuntimeControlError(
                f"{label} mismatch: expected {frame_policy!r}, got {value!r}"
            )
    if manifest_pairs != selected_pairs:
        raise SelectiveRuntimeControlError(
            f"manifest DQS1 selected pairs mismatch: expected {selected_pairs}, "
            f"got {manifest_pairs}"
        )
    if archive_pairs != selected_pairs:
        raise SelectiveRuntimeControlError(
            f"archive DQS1 selected pairs mismatch: expected {selected_pairs}, got {archive_pairs}"
        )
    if manifest_frames != expected_frames:
        raise SelectiveRuntimeControlError(
            f"manifest DQS1 affected frames mismatch: expected {expected_frames}, "
            f"got {manifest_frames}"
        )
    if archive_dqs1["tail_bytes"] != manifest_dqs1.get("payload_bytes"):
        raise SelectiveRuntimeControlError("archive DQS1 payload length mismatch vs manifest")
    if archive_dqs1["tail_sha256"] != manifest_dqs1.get("payload_sha256"):
        raise SelectiveRuntimeControlError("archive DQS1 payload SHA mismatch vs manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "archive_zip": str(_submission_archive(selective_submission_dir).resolve()),
        "frame_policy": frame_policy,
        "selected_pair_indices": selected_pairs,
        "selected_frame_indices": expected_frames,
        "dqs1_tail_offset": archive_dqs1["tail_offset"],
        "dqs1_tail_bytes": archive_dqs1["tail_bytes"],
        "dqs1_tail_sha256": archive_dqs1["tail_sha256"],
        "manifest_payload_sha256": manifest_dqs1.get("payload_sha256"),
    }


def _child_path(root: Path, rel: Path) -> Path:
    root_resolved = root.resolve()
    path = (root_resolved / rel).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise SelectiveRuntimeControlError(f"path escapes root: {rel}")
    return path


def _empty_region_hashes() -> dict[str, hashlib._Hash]:
    return {
        "parent": hashlib.sha256(),
        "global_mutated": hashlib.sha256(),
        "selective": hashlib.sha256(),
    }


def _hex_region_hashes(region_hashes: dict[str, hashlib._Hash]) -> dict[str, str]:
    return {name: digest.hexdigest() for name, digest in region_hashes.items()}


def _digest_region_hashes(region_hashes: dict[str, hashlib._Hash]) -> dict[str, str]:
    return {name: digest.hexdigest() for name, digest in region_hashes.items()}


def _native_raw_compare_binary() -> Path | None:
    env_path = os.environ.get("PACT_RAW_LOCALITY_COMPARE_BIN")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            REPO_ROOT / "runtime-rs" / "target" / "release" / "raw-locality-compare",
            REPO_ROOT / "runtime-rs" / "target" / "debug" / "raw-locality-compare",
        ]
    )
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _native_raw_compare(
    *,
    parent_raw: Path,
    global_mutated_raw: Path,
    selective_raw: Path,
    selected_frame_indices: list[int],
    frame_count: int,
    frame_bytes: int | None,
    rel_path: str,
    sample_limit: int,
    deadline: GlobalDeadline | None,
    backend: str,
) -> dict[str, Any] | None:
    binary = _native_raw_compare_binary()
    if binary is None:
        if backend == "rust":
            raise SelectiveRuntimeControlError(
                "raw compare backend 'rust' requested but runtime-rs raw-locality-compare "
                "binary is not built"
            )
        return None
    cmd = [
        str(binary),
        "--parent",
        str(parent_raw),
        "--global-mutated",
        str(global_mutated_raw),
        "--selective",
        str(selective_raw),
        "--selected-frames",
        ",".join(str(index) for index in selected_frame_indices),
        "--frame-count",
        str(frame_count),
        "--raw-path",
        rel_path,
        "--sample-limit",
        str(sample_limit),
    ]
    if frame_bytes is not None:
        cmd.extend(["--frame-bytes", str(frame_bytes)])
    timeout = None
    if deadline is not None:
        remaining = deadline.remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                raise SelectiveRuntimeControlError(
                    f"native raw compare exceeded global timeout {deadline.timeout_seconds}s"
                )
            timeout = remaining
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SelectiveRuntimeControlError(
            f"native raw compare timed out after {timeout:.1f}s"
        ) from exc
    if proc.returncode != 0:
        raise SelectiveRuntimeControlError(
            f"native raw compare failed with exit {proc.returncode}: {proc.stderr[-2000:]}"
        )
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise SelectiveRuntimeControlError("native raw compare emitted non-object JSON")
    payload["raw_compare_backend"] = "rust"
    payload["raw_compare_binary"] = str(binary.resolve())
    return payload


def _hash_frame_sample(
    *,
    frame_index: int,
    expected_role: str,
    expected: bytes,
    actual: bytes,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "expected_role": expected_role,
        "expected_sha256": sha256_bytes(expected),
        "actual_selective_sha256": sha256_bytes(actual),
    }


def compare_raw_triplet(
    *,
    parent_raw: Path,
    global_mutated_raw: Path,
    selective_raw: Path,
    selected_frame_indices: list[int],
    frame_count: int,
    frame_bytes: int | None = None,
    rel_path: str = "0.raw",
    sample_limit: int = 8,
    deadline: GlobalDeadline | None = None,
    raw_compare_backend: str = "auto",
) -> dict[str, Any]:
    """Compare one parent/global/selective raw triplet frame by frame."""

    if raw_compare_backend not in {"auto", "python", "rust"}:
        raise SelectiveRuntimeControlError(
            f"unsupported raw compare backend: {raw_compare_backend!r}"
        )
    if raw_compare_backend in {"auto", "rust"}:
        native = _native_raw_compare(
            parent_raw=parent_raw,
            global_mutated_raw=global_mutated_raw,
            selective_raw=selective_raw,
            selected_frame_indices=selected_frame_indices,
            frame_count=frame_count,
            frame_bytes=frame_bytes,
            rel_path=rel_path,
            sample_limit=sample_limit,
            deadline=deadline,
            backend=raw_compare_backend,
        )
        if native is not None:
            return native

    if frame_count <= 0:
        raise SelectiveRuntimeControlError("frame_count must be positive")
    for path in (parent_raw, global_mutated_raw, selective_raw):
        if not path.is_file():
            return {
                "raw_path": rel_path,
                "raw_byte_sizes_match": False,
                "frame_bytes": frame_bytes,
                "frame_count": frame_count,
                "selected_frame_indices": selected_frame_indices,
                "hashes": {},
                "mismatch_counts": {
                    "missing_raw_file_count": 1,
                    "raw_size_mismatch_count": 1,
                    "selected_frame_mismatch_count": 0,
                    "unselected_frame_mismatch_count": 0,
                },
                "mismatch_samples": [],
                "blockers": [f"missing_raw_file:{rel_path}:{path}"],
            }

    sizes = {
        "parent": parent_raw.stat().st_size,
        "global_mutated": global_mutated_raw.stat().st_size,
        "selective": selective_raw.stat().st_size,
    }
    blockers: list[str] = []
    mismatch_counts = {
        "missing_raw_file_count": 0,
        "raw_size_mismatch_count": 0,
        "selected_frame_mismatch_count": 0,
        "unselected_frame_mismatch_count": 0,
    }

    if len(set(sizes.values())) != 1:
        file_hashes = {
            "parent": sha256_file(parent_raw),
            "global_mutated": sha256_file(global_mutated_raw),
            "selective": sha256_file(selective_raw),
        }
        mismatch_counts["raw_size_mismatch_count"] = 1
        blockers.append(f"raw_size_mismatch:{rel_path}")
        return {
            "raw_path": rel_path,
            "raw_byte_sizes_match": False,
            "frame_bytes": frame_bytes,
            "frame_count": frame_count,
            "selected_frame_indices": selected_frame_indices,
            "hashes": {"raw_files": file_hashes},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
        }

    raw_size = sizes["parent"]
    if frame_bytes is None:
        if raw_size % frame_count:
            file_hashes = {
                "parent": sha256_file(parent_raw),
                "global_mutated": sha256_file(global_mutated_raw),
                "selective": sha256_file(selective_raw),
            }
            mismatch_counts["raw_size_mismatch_count"] = 1
            blockers.append(f"raw_size_not_divisible_by_frame_count:{rel_path}")
            return {
                "raw_path": rel_path,
                "raw_byte_sizes_match": False,
                "frame_bytes": None,
                "frame_count": frame_count,
                "selected_frame_indices": selected_frame_indices,
                "hashes": {"raw_files": file_hashes},
                "raw_bytes": sizes,
                "mismatch_counts": mismatch_counts,
                "mismatch_samples": [],
                "blockers": blockers,
            }
        frame_bytes = raw_size // frame_count
    if frame_bytes <= 0:
        raise SelectiveRuntimeControlError("frame_bytes must be positive")

    expected_size = frame_bytes * frame_count
    if raw_size != expected_size:
        file_hashes = {
            "parent": sha256_file(parent_raw),
            "global_mutated": sha256_file(global_mutated_raw),
            "selective": sha256_file(selective_raw),
        }
        mismatch_counts["raw_size_mismatch_count"] = 1
        blockers.append(f"raw_size_does_not_match_frame_geometry:{rel_path}")
        return {
            "raw_path": rel_path,
            "raw_byte_sizes_match": False,
            "frame_bytes": frame_bytes,
            "frame_count": frame_count,
            "selected_frame_indices": selected_frame_indices,
            "hashes": {"raw_files": file_hashes},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
        }

    invalid_frames = [
        frame for frame in selected_frame_indices if frame < 0 or frame >= frame_count
    ]
    if invalid_frames:
        file_hashes = {
            "parent": sha256_file(parent_raw),
            "global_mutated": sha256_file(global_mutated_raw),
            "selective": sha256_file(selective_raw),
        }
        blockers.append(f"selected_frame_index_out_of_range:{invalid_frames}")
        return {
            "raw_path": rel_path,
            "raw_byte_sizes_match": True,
            "frame_bytes": frame_bytes,
            "frame_count": frame_count,
            "selected_frame_indices": selected_frame_indices,
            "hashes": {"raw_files": file_hashes},
            "raw_bytes": sizes,
            "mismatch_counts": mismatch_counts,
            "mismatch_samples": [],
            "blockers": blockers,
        }

    selected_set = set(selected_frame_indices)
    raw_file_hashes = _empty_region_hashes()
    selected_hashes = _empty_region_hashes()
    unselected_hashes = _empty_region_hashes()
    selected_samples: list[dict[str, Any]] = []
    unselected_samples: list[dict[str, Any]] = []

    with (
        parent_raw.open("rb") as parent_handle,
        global_mutated_raw.open("rb") as global_handle,
        selective_raw.open("rb") as selective_handle,
    ):
        for frame_index in range(frame_count):
            if deadline is not None and frame_index % 16 == 0:
                deadline.check(f"compare:{rel_path}")
            parent_frame = parent_handle.read(frame_bytes)
            global_frame = global_handle.read(frame_bytes)
            selective_frame = selective_handle.read(frame_bytes)
            if (
                len(parent_frame) != frame_bytes
                or len(global_frame) != frame_bytes
                or len(selective_frame) != frame_bytes
            ):
                blockers.append(f"short_raw_read:{rel_path}:{frame_index}")
                break
            raw_file_hashes["parent"].update(parent_frame)
            raw_file_hashes["global_mutated"].update(global_frame)
            raw_file_hashes["selective"].update(selective_frame)

            if frame_index in selected_set:
                selected_hashes["parent"].update(parent_frame)
                selected_hashes["global_mutated"].update(global_frame)
                selected_hashes["selective"].update(selective_frame)
                if selective_frame != global_frame:
                    mismatch_counts["selected_frame_mismatch_count"] += 1
                    if len(selected_samples) < sample_limit:
                        selected_samples.append(
                            _hash_frame_sample(
                                frame_index=frame_index,
                                expected_role="global_mutated",
                                expected=global_frame,
                                actual=selective_frame,
                            )
                        )
            else:
                unselected_hashes["parent"].update(parent_frame)
                unselected_hashes["global_mutated"].update(global_frame)
                unselected_hashes["selective"].update(selective_frame)
                if selective_frame != parent_frame:
                    mismatch_counts["unselected_frame_mismatch_count"] += 1
                    if len(unselected_samples) < sample_limit:
                        unselected_samples.append(
                            _hash_frame_sample(
                                frame_index=frame_index,
                                expected_role="parent",
                                expected=parent_frame,
                                actual=selective_frame,
                            )
                        )

    if mismatch_counts["selected_frame_mismatch_count"]:
        blockers.append(f"selected_frame_locality_mismatch:{rel_path}")
    if mismatch_counts["unselected_frame_mismatch_count"]:
        blockers.append(f"unselected_frame_parent_regression:{rel_path}")

    return {
        "raw_path": rel_path,
        "raw_byte_sizes_match": True,
        "frame_bytes": frame_bytes,
        "frame_count": frame_count,
        "selected_frame_indices": selected_frame_indices,
        "compared_selected_frame_count": len(selected_set),
        "compared_unselected_frame_count": frame_count - len(selected_set),
        "hashes": {
            "raw_files": _digest_region_hashes(raw_file_hashes),
            "selected_frames": _hex_region_hashes(selected_hashes),
            "unselected_frames": _hex_region_hashes(unselected_hashes),
        },
        "raw_bytes": sizes,
        "mismatch_counts": mismatch_counts,
        "mismatch_samples": selected_samples + unselected_samples,
        "blockers": blockers,
    }


def compare_inflated_outputs(
    *,
    parent_dir: Path,
    global_mutated_dir: Path,
    selective_dir: Path,
    video_names: list[str],
    selected_frame_indices: list[int],
    frame_count: int,
    frame_bytes: int | None,
    deadline: GlobalDeadline | None = None,
    raw_compare_backend: str = "auto",
) -> dict[str, Any]:
    raw_results: list[dict[str, Any]] = []
    aggregate_counts = {
        "missing_raw_file_count": 0,
        "raw_size_mismatch_count": 0,
        "selected_frame_mismatch_count": 0,
        "unselected_frame_mismatch_count": 0,
    }
    blockers: list[str] = []
    for video_name in video_names:
        rel = raw_relpath_for_video_name(video_name)
        result = compare_raw_triplet(
            parent_raw=_child_path(parent_dir, rel),
            global_mutated_raw=_child_path(global_mutated_dir, rel),
            selective_raw=_child_path(selective_dir, rel),
            selected_frame_indices=selected_frame_indices,
            frame_count=frame_count,
            frame_bytes=frame_bytes,
            rel_path=rel.as_posix(),
            deadline=deadline,
            raw_compare_backend=raw_compare_backend,
        )
        raw_results.append(result)
        for key in aggregate_counts:
            aggregate_counts[key] += int(result["mismatch_counts"][key])
        blockers.extend(str(blocker) for blocker in result["blockers"])

    locality_passed = not blockers and all(value == 0 for value in aggregate_counts.values())
    return {
        "locality_controls_passed": locality_passed,
        "mismatch_counts": aggregate_counts,
        "raw_results": raw_results,
        "blockers": sorted(set(blockers)),
    }


def _validate_zip_member_name(name: str) -> None:
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts:
        raise SelectiveRuntimeControlError(f"zip-slip archive member rejected: {name!r}")
    if "\\" in name or ":" in name:
        raise SelectiveRuntimeControlError(f"non-portable archive member rejected: {name!r}")


def extract_zip_safely(archive_zip: Path, dest: Path) -> list[str]:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    with zipfile.ZipFile(archive_zip, "r") as zf:
        for info in zf.infolist():
            _validate_zip_member_name(info.filename)
            if info.is_dir():
                continue
            target = (dest / info.filename).resolve()
            if dest not in target.parents:
                raise SelectiveRuntimeControlError(
                    f"zip-slip archive member rejected: {info.filename!r}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            members.append(info.filename)
    return members


def _submission_archive(submission_dir: Path) -> Path:
    archive = submission_dir / "archive.zip"
    if not archive.is_file():
        raise SelectiveRuntimeControlError(f"submission archive missing: {archive}")
    return archive


def _runtime_entrypoint(runtime_dir: Path) -> tuple[list[str], str, Path]:
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_py = runtime_dir / "inflate.py"
    if inflate_sh.is_file():
        return ["bash", str(inflate_sh)], "inflate.sh", inflate_sh
    if inflate_py.is_file():
        return [sys.executable, str(inflate_py)], "inflate.py", inflate_py
    raise SelectiveRuntimeControlError(
        f"runtime has neither inflate.sh nor inflate.py: {runtime_dir}"
    )


def _inflate_manifest_path(work_root: Path, label: str) -> Path:
    return work_root / label / INFLATE_MANIFEST_NAME


def _target_identity(
    target: InflateTarget,
    *,
    video_names_file: Path,
    entry_kind: str,
    entry_path: Path,
) -> dict[str, Any]:
    return {
        "label": target.label,
        "runtime_dir": str(target.runtime_dir.resolve()),
        "archive_zip": str(target.archive_zip.resolve()),
        "archive_source": target.archive_source,
        "archive_bytes": target.archive_zip.stat().st_size,
        "archive_sha256": sha256_file(target.archive_zip),
        "entrypoint_kind": entry_kind,
        "entrypoint_path": str(entry_path.resolve()),
        "entrypoint_sha256": sha256_file(entry_path),
        "video_names_file": str(video_names_file.resolve()),
    }


def _expected_output_paths(output_dir: Path, video_names: list[str]) -> list[Path]:
    return [_child_path(output_dir, raw_relpath_for_video_name(name)) for name in video_names]


def _raw_output_sizes(output_dir: Path, video_names: list[str]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name in video_names:
        rel = raw_relpath_for_video_name(name)
        path = _child_path(output_dir, rel)
        if not path.is_file():
            raise SelectiveRuntimeControlError(f"missing inflated raw output: {path}")
        sizes[rel.as_posix()] = path.stat().st_size
    return sizes


def _manifest_outputs_match(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    video_names: list[str],
) -> bool:
    raw_sizes = manifest.get("raw_output_bytes")
    if not isinstance(raw_sizes, dict):
        return False
    try:
        current_sizes = _raw_output_sizes(output_dir, video_names)
    except SelectiveRuntimeControlError:
        return False
    parsed_sizes: dict[str, int] = {}
    try:
        for key, value in raw_sizes.items():
            if isinstance(value, bool):
                return False
            parsed_sizes[str(key)] = int(value)
    except (TypeError, ValueError):
        return False
    return parsed_sizes == current_sizes


def _archive_dir_matches_target(target: InflateTarget, archive_dir: Path) -> bool:
    if not archive_dir.is_dir():
        return False
    try:
        member_name, member_data = read_single_stored_member_payload(target.archive_zip)
    except (OSError, zipfile.BadZipFile, SelectiveRuntimeControlError):
        return False
    extracted = archive_dir / member_name
    return extracted.is_file() and sha256_file(extracted) == sha256_bytes(member_data)


def run_inflate_target(
    target: InflateTarget,
    *,
    work_root: Path,
    video_names_file: Path,
    video_names: list[str],
    timeout_seconds: int,
    deadline: GlobalDeadline | None = None,
    reuse_existing_inflates: bool = False,
    recorder: PhaseRecorder | None = None,
) -> dict[str, Any]:
    if not target.runtime_dir.is_dir():
        raise SelectiveRuntimeControlError(
            f"{target.label} runtime dir missing: {target.runtime_dir}"
        )
    if not target.archive_zip.is_file():
        raise SelectiveRuntimeControlError(
            f"{target.label} archive missing: {target.archive_zip}"
        )

    archive_dir = work_root / target.label / "archive"
    output_dir = work_root / target.label / "inflated"
    entry_cmd, entry_kind, entry_path = _runtime_entrypoint(target.runtime_dir)
    identity = _target_identity(
        target,
        video_names_file=video_names_file,
        entry_kind=entry_kind,
        entry_path=entry_path,
    )
    manifest_path = _inflate_manifest_path(work_root, target.label)
    if reuse_existing_inflates and manifest_path.is_file():
        manifest = read_json_object(manifest_path)
        if (
            manifest.get("schema")
            == "decoder_q_selective_runtime_locality_inflate_manifest.v1"
            and manifest.get("target_identity") == identity
            and _manifest_outputs_match(
                manifest,
                output_dir=output_dir,
                video_names=video_names,
            )
        ):
            cached = dict(manifest["run"])
            cached["reused_existing_inflate"] = True
            cached["reuse_mode"] = "manifest_verified"
            cached["manifest_path"] = str(manifest_path.resolve())
            if recorder is not None:
                recorder.append_event(
                    {
                        "event": "inflate_reused",
                        "target": target.label,
                        "reuse_mode": "manifest_verified",
                    }
                )
            return cached

    if reuse_existing_inflates and all(
        path.is_file() and path.stat().st_size > 0
        for path in _expected_output_paths(output_dir, video_names)
    ) and _archive_dir_matches_target(target, archive_dir):
        raw_sizes = _raw_output_sizes(output_dir, video_names)
        cached = {
            **identity,
            "archive_member_count": 1,
            "command": None,
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "stdout_sha256": None,
            "stderr_sha256": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "output_dir": str(output_dir),
            "reused_existing_inflate": True,
            "reuse_mode": "legacy_archive_member_verified",
            "raw_output_bytes": raw_sizes,
        }
        manifest = {
            "schema": "decoder_q_selective_runtime_locality_inflate_manifest.v1",
            "target_identity": identity,
            "run": cached,
            "raw_output_bytes": raw_sizes,
            "legacy_reuse_without_prior_manifest": True,
            **FALSE_AUTHORITY,
        }
        write_json(manifest_path, manifest)
        if recorder is not None:
            recorder.append_event(
                {
                    "event": "inflate_reused",
                    "target": target.label,
                    "reuse_mode": "legacy_archive_member_verified",
                }
            )
        return {**cached, "manifest_path": str(manifest_path.resolve())}

    if deadline is not None:
        deadline.check(f"inflate:{target.label}")

    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    members = extract_zip_safely(target.archive_zip, archive_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [*entry_cmd, str(archive_dir), str(output_dir), str(video_names_file)]
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PACT_PYTHON_BIN", sys.executable)
    env.setdefault("PYTHON", sys.executable)
    env.setdefault("PYTHON_BIN", sys.executable)
    env.setdefault("COMMA_CHALLENGE_ROOT", str(REPO_ROOT / "upstream"))
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(work_root / target.label / "uv_env"))

    effective_timeout = (
        deadline.subprocess_timeout(timeout_seconds)
        if deadline is not None
        else timeout_seconds
    )
    active_recorder = recorder if recorder is not None else PhaseRecorder()
    with active_recorder.timed(
        "inflate",
        target=target.label,
        timeout_seconds=effective_timeout,
        requested_timeout_seconds=timeout_seconds,
    ):
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(target.runtime_dir),
                env=env,
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SelectiveRuntimeControlError(
                f"{target.label} inflate timed out after {effective_timeout}s"
            ) from exc
        elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise SelectiveRuntimeControlError(
            f"{target.label} {entry_kind} failed with exit {proc.returncode}: "
            f"{proc.stderr[-2000:]}"
        )

    raw_sizes = _raw_output_sizes(output_dir, video_names)
    run = {
        **identity,
        "archive_member_count": len(members),
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "output_dir": str(output_dir),
        "raw_output_bytes": raw_sizes,
        "reused_existing_inflate": False,
        "reuse_mode": "fresh",
    }
    write_json(
        manifest_path,
        {
            "schema": "decoder_q_selective_runtime_locality_inflate_manifest.v1",
            "target_identity": identity,
            "run": run,
            "raw_output_bytes": raw_sizes,
            **FALSE_AUTHORITY,
        },
    )
    return {**run, "manifest_path": str(manifest_path.resolve())}


def run_inflate_targets(
    targets: list[InflateTarget],
    *,
    work_root: Path,
    video_names_file: Path,
    video_names: list[str],
    timeout_seconds: int,
    deadline: GlobalDeadline | None,
    max_parallelism: int,
    reuse_existing_inflates: bool,
    recorder: PhaseRecorder,
) -> dict[str, dict[str, Any]]:
    if max_parallelism <= 0:
        raise SelectiveRuntimeControlError("--max-inflate-parallelism must be positive")
    runs: dict[str, dict[str, Any]] = {}
    with recorder.timed(
        "inflate_targets",
        target_count=len(targets),
        max_parallelism=max_parallelism,
        reuse_existing_inflates=reuse_existing_inflates,
    ):
        workers = min(max_parallelism, len(targets))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_inflate_target,
                    target,
                    work_root=work_root,
                    video_names_file=video_names_file,
                    video_names=video_names,
                    timeout_seconds=timeout_seconds,
                    deadline=deadline,
                    reuse_existing_inflates=reuse_existing_inflates,
                    recorder=recorder,
                ): target.label
                for target in targets
            }
            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                if deadline is not None:
                    deadline.check(f"inflate:{label}")
                runs[label] = future.result()
    return runs


def build_report(
    *,
    parent_submission_dir: Path,
    global_mutated_submission_dir: Path | None,
    global_mutated_archive: Path | None,
    selective_submission_dir: Path,
    selected_pairs: list[int],
    frame_policy: str,
    video_names_file: Path,
    frame_count: int,
    frame_bytes: int | None,
    work_root: Path,
    timeout_seconds: int,
    work_dir_preserved: bool,
    global_timeout_seconds: int | None = None,
    max_inflate_parallelism: int = 1,
    reuse_existing_inflates: bool = False,
    raw_compare_backend: str = "auto",
) -> dict[str, Any]:
    if (global_mutated_submission_dir is None) == (global_mutated_archive is None):
        raise SelectiveRuntimeControlError(
            "provide exactly one of --global-mutated-submission-dir or "
            "--global-mutated-archive"
        )
    parent_submission_dir = parent_submission_dir.resolve()
    selective_submission_dir = selective_submission_dir.resolve()
    video_names_file = video_names_file.resolve()
    work_root = work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    progress_jsonl = work_root / "locality_controls_progress.jsonl"
    recorder = PhaseRecorder(progress_jsonl=progress_jsonl)
    deadline = GlobalDeadline(global_timeout_seconds)
    global_mutated_submission_dir = (
        None
        if global_mutated_submission_dir is None
        else global_mutated_submission_dir.resolve()
    )
    global_mutated_archive = (
        None if global_mutated_archive is None else global_mutated_archive.resolve()
    )

    with recorder.timed("load_inputs"):
        selected_frame_indices = selected_frame_indices_for_pairs(
            selected_pairs,
            frame_policy=frame_policy,
        )
        video_names = read_video_names(video_names_file)
    with recorder.timed("validate_selective_runtime_contract"):
        selective_contract = validate_selective_runtime_contract(
            selective_submission_dir=selective_submission_dir,
            selected_pairs=selected_pairs,
            frame_policy=frame_policy,
        )

    parent_target = InflateTarget(
        label="parent",
        runtime_dir=parent_submission_dir,
        archive_zip=_submission_archive(parent_submission_dir),
        archive_source="parent_submission_dir/archive.zip",
    )
    if global_mutated_submission_dir is not None:
        global_target = InflateTarget(
            label="global_mutated",
            runtime_dir=global_mutated_submission_dir,
            archive_zip=_submission_archive(global_mutated_submission_dir),
            archive_source="global_mutated_submission_dir/archive.zip",
        )
    else:
        assert global_mutated_archive is not None
        global_target = InflateTarget(
            label="global_mutated",
            runtime_dir=parent_submission_dir,
            archive_zip=global_mutated_archive,
            archive_source="global_mutated_archive_with_parent_runtime",
        )
    selective_target = InflateTarget(
        label="selective",
        runtime_dir=selective_submission_dir,
        archive_zip=_submission_archive(selective_submission_dir),
        archive_source="selective_submission_dir/archive.zip",
    )

    targets = [parent_target, global_target, selective_target]
    runs = run_inflate_targets(
        targets,
        work_root=work_root,
        video_names_file=video_names_file,
        video_names=video_names,
        timeout_seconds=timeout_seconds,
        deadline=deadline,
        max_parallelism=max_inflate_parallelism,
        reuse_existing_inflates=reuse_existing_inflates,
        recorder=recorder,
    )
    deadline.check("compare")
    with recorder.timed("compare_inflated_outputs"):
        comparison = compare_inflated_outputs(
            parent_dir=Path(runs["parent"]["output_dir"]),
            global_mutated_dir=Path(runs["global_mutated"]["output_dir"]),
            selective_dir=Path(runs["selective"]["output_dir"]),
            video_names=video_names,
            selected_frame_indices=selected_frame_indices,
            frame_count=frame_count,
            frame_bytes=frame_bytes,
            deadline=deadline,
            raw_compare_backend=raw_compare_backend,
        )

    blockers = [
        "score_claim_false_locality_control_only",
        "promotion_requires_exact_contest_auth_eval",
    ]
    blockers.extend(comparison["blockers"])
    report_runs = {label: dict(run) for label, run in runs.items()}
    if not work_dir_preserved:
        for run in report_runs.values():
            run["output_dir"] = None
    recorder.append_event(
        {
            "event": "report_built",
            "locality_controls_passed": comparison["locality_controls_passed"],
            "blocker_count": len(set(blockers)),
        }
    )
    return {
        "schema": SCHEMA,
        "producer": PRODUCER,
        **FALSE_AUTHORITY,
        "evidence_grade": "inflate-locality-control",
        "score_axis": "[locality-control no-score]",
        "locality_controls_passed": comparison["locality_controls_passed"],
        "selected_pair_indices": selected_pairs,
        "frame_policy": frame_policy,
        "selected_frame_indices": selected_frame_indices,
        "selective_contract": selective_contract,
        "frame_count": frame_count,
        "frame_bytes": frame_bytes,
        "video_names_file": str(video_names_file),
        "video_names": video_names,
        "work_dir": str(work_root) if work_dir_preserved else None,
        "work_dir_preserved": work_dir_preserved,
        "progress_jsonl": str(progress_jsonl) if work_dir_preserved else None,
        "timeout_policy": {
            "per_inflate_timeout_seconds": timeout_seconds,
            "global_timeout_seconds": global_timeout_seconds,
            "max_inflate_parallelism": max_inflate_parallelism,
            "reuse_existing_inflates": reuse_existing_inflates,
            "raw_compare_backend": raw_compare_backend,
        },
        "phase_timings": recorder.phases,
        "targets": report_runs,
        "hashes": {
            result["raw_path"]: result["hashes"]
            for result in comparison["raw_results"]
        },
        "mismatch_counts": comparison["mismatch_counts"],
        "raw_results": comparison["raw_results"],
        "blockers": sorted(set(blockers)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-submission-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--global-mutated-submission-dir", type=Path)
    group.add_argument("--global-mutated-archive", type=Path)
    parser.add_argument("--selective-submission-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-pairs",
        required=True,
        help="Comma- or whitespace-separated pair indices, e.g. '501' or '501,502'.",
    )
    parser.add_argument(
        "--frame-policy",
        choices=["pair_all_frames", "segnet_last_frame_only"],
        required=True,
    )
    parser.add_argument("--video-names-file", type=Path, default=DEFAULT_VIDEO_NAMES_FILE)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument(
        "--frame-bytes",
        type=int,
        help="Bytes per RGB frame. If omitted, infer from raw size / frame-count.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--global-timeout-seconds",
        type=int,
        default=0,
        help="Whole-run wall-clock budget. 0 disables the runner-level global deadline.",
    )
    parser.add_argument(
        "--max-inflate-parallelism",
        type=int,
        default=1,
        help="Number of parent/global/selective inflates to run concurrently.",
    )
    parser.add_argument(
        "--reuse-existing-inflates",
        action="store_true",
        help="Reuse completed per-target inflate outputs in --work-dir when identity checks pass.",
    )
    parser.add_argument(
        "--raw-compare-backend",
        choices=["auto", "python", "rust"],
        default="auto",
        help="Raw locality comparison backend. auto uses Rust when the binary is available.",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.timeout_seconds <= 0:
            raise SelectiveRuntimeControlError("--timeout-seconds must be positive")
        if args.global_timeout_seconds < 0:
            raise SelectiveRuntimeControlError("--global-timeout-seconds must be non-negative")
        if args.max_inflate_parallelism <= 0:
            raise SelectiveRuntimeControlError("--max-inflate-parallelism must be positive")
        selected_pairs = parse_selected_pairs(args.selected_pairs)
        if args.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="dqs1_locality_controls_") as tmp:
                report = build_report(
                    parent_submission_dir=args.parent_submission_dir,
                    global_mutated_submission_dir=args.global_mutated_submission_dir,
                    global_mutated_archive=args.global_mutated_archive,
                    selective_submission_dir=args.selective_submission_dir,
                    selected_pairs=selected_pairs,
                    frame_policy=args.frame_policy,
                    video_names_file=args.video_names_file,
                    frame_count=args.frame_count,
                    frame_bytes=args.frame_bytes,
                    work_root=Path(tmp),
                    timeout_seconds=args.timeout_seconds,
                    work_dir_preserved=False,
                    global_timeout_seconds=(
                        args.global_timeout_seconds
                        if args.global_timeout_seconds > 0
                        else None
                    ),
                    max_inflate_parallelism=args.max_inflate_parallelism,
                    reuse_existing_inflates=args.reuse_existing_inflates,
                    raw_compare_backend=args.raw_compare_backend,
                )
        else:
            args.work_dir.mkdir(parents=True, exist_ok=True)
            report = build_report(
                parent_submission_dir=args.parent_submission_dir,
                global_mutated_submission_dir=args.global_mutated_submission_dir,
                global_mutated_archive=args.global_mutated_archive,
                selective_submission_dir=args.selective_submission_dir,
                selected_pairs=selected_pairs,
                frame_policy=args.frame_policy,
                video_names_file=args.video_names_file,
                frame_count=args.frame_count,
                frame_bytes=args.frame_bytes,
                work_root=args.work_dir,
                timeout_seconds=args.timeout_seconds,
                work_dir_preserved=True,
                global_timeout_seconds=(
                    args.global_timeout_seconds
                    if args.global_timeout_seconds > 0
                    else None
                ),
                max_inflate_parallelism=args.max_inflate_parallelism,
                reuse_existing_inflates=args.reuse_existing_inflates,
                raw_compare_backend=args.raw_compare_backend,
            )
    except (OSError, json.JSONDecodeError, SelectiveRuntimeControlError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    output = dumps_json(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["locality_controls_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
