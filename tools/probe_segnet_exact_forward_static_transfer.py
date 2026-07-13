#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Process-static ABBA timing probe for the exact frozen-SegNet CPU forward.

The canary tournament is selection-only.  Terminal evidence is produced by
four fresh measurement children in baseline/selected/selected/baseline order
and four separate full-replay children.  Every child binds Torch intra-op and
inter-op threads before model load and never changes them afterwards.

This is local macOS-CPU advisory evidence for the frozen training-forward
component.  It never invokes the evaluator, MPS, CUDA, a provider, or a live
trainer and cannot move a score/frontier pointer.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = Path(__file__).resolve()
V4_PATH = REPO_ROOT / "tools/probe_segnet_exact_forward_transfer.py"
for _root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def _load_v4() -> Any:
    spec = importlib.util.spec_from_file_location("probe_segnet_exact_forward_transfer_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import reviewed v4 probe: {V4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = _load_v4()
base = v4.base

SCHEMA = "frozen_segnet_exact_forward_static_transfer_probe_v2"
CHECKPOINT_SCHEMA = "frozen_segnet_exact_forward_static_transfer_checkpoint_v2"
STAGE_SCHEMA = "frozen_segnet_exact_forward_static_transfer_stage_v2"
WORKER_CONFIG_SCHEMA = "frozen_segnet_exact_forward_static_transfer_worker_config_v2"
FAILURE_SCHEMA = "frozen_segnet_exact_forward_static_transfer_failure_v2"
LATEST_SCHEMA = "frozen_segnet_exact_forward_static_transfer_latest_v2"
RUN_IDENTITY_SCHEMA = "frozen_segnet_exact_forward_static_transfer_run_identity_v1"
LOCK_ACQUISITION_SCHEMA = "frozen_segnet_exact_forward_static_transfer_lock_acquisition_v1"
STAGE_ORDER = ("baseline_rep0", "selected_rep0", "selected_rep1", "baseline_rep1")
WORKER_MODES = ("measurement", "replay")

# ASSUMED recovery envelope: at most 25 newly computed pairs are lost on a
# process failure.  The value and provenance are also carried by the DSL and
# terminal receipt; it is not a measured performance constant.
CHECKPOINT_INTERVAL = 25
CHECKPOINT_INTERVAL_PROVENANCE = "ASSUMED_RECOVERY_ENVELOPE_MAX_25_PAIR_RECOMPUTE"
VERDICT_PAIR_CARDINALITY = v4.VERDICT_PAIR_CARDINALITY
# Operator-sealed v4 false-positive budget.  This is an admission policy, not
# a measured number and not an inferred scientific constant.
MATCHED_SIGN_ALPHA = v4.MATCHED_SIGN_ALPHA
MATCHED_SIGN_ALPHA_PROVENANCE = "OPERATOR_SEALED_TRANSFER_V4_FALSE_POSITIVE_BUDGET"
METHOD = "fresh_child_process_static_threads"
AUTHORITY = {"score_claim": False, "pointer_moved": False, "promotion_eligible": False}
AXIS = "[macOS-CPU advisory; process-static torch-fp32 training-forward; no MPS/CUDA]"
EVIDENCE_LABELS = {
    "timing_and_sha": "MEASURED",
    "zero_flip_from_sha": "DERIVED",
    "canary_count": "ASSUMED_HEURISTIC_SCREEN_ONLY",
    "checkpoint_interval": "ASSUMED_RECOVERY_ENVELOPE",
}
THREAD_ENV_KEYS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def active_python_executable() -> str:
    """Preserve a venv launcher path; resolving its symlink drops venv sys.path."""

    return str(Path(sys.executable).absolute())


def sha256_file(path: Path) -> str:
    return v4.sha256_file(path)


def sha256_json(payload: Mapping[str, Any]) -> str:
    return v4.sha256_json(payload)


def _without_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "sha256"}


def _with_sha(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _without_sha(payload)
    result["sha256"] = sha256_json(result)
    return result


def _validate_digest(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("sha256") != sha256_json(_without_sha(payload)):
        raise RuntimeError(f"{label} digest mismatch")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def checkpoint_namespace(out: Path) -> Path:
    return out.parent / f"{out.stem}_static_checkpoints"


def load_or_create_run_identity(
    *, checkpoint_dir: Path, raw_argv: Sequence[str], lock_owner: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist initial provenance once so a new lock PID can resume it."""

    path = checkpoint_dir / "run_identity.json"
    expected_argv = [active_python_executable(), str(TOOL_PATH), *raw_argv]
    if path.is_file():
        identity = json.loads(path.read_text())
        _validate_digest(identity, label="run identity")
        if identity.get("schema") != RUN_IDENTITY_SCHEMA or identity.get("argv") != expected_argv:
            raise RuntimeError("run identity argv/schema mismatch on resume")
        return identity
    identity = _with_sha(
        {
            "schema": RUN_IDENTITY_SCHEMA,
            "run_id": uuid.uuid4().hex,
            "created_at_utc": v4.utc_now(),
            "argv": expected_argv,
            "initial_git": _git_custody(),
            "initial_lock_owner": dict(lock_owner),
            "authority": dict(AUTHORITY),
        }
    )
    v4._atomic_checkpoint(path, identity)
    return json.loads(path.read_text())


def record_lock_acquisition(
    *, checkpoint_dir: Path, run_identity: Mapping[str, Any], lock_owner: Mapping[str, Any]
) -> Path:
    """Append the current acquisition without changing the immutable resume key."""

    payload = _with_sha(
        {
            "schema": LOCK_ACQUISITION_SCHEMA,
            "written_at_utc": v4.utc_now(),
            "run_identity_sha256": run_identity["sha256"],
            "lock_owner": dict(lock_owner),
            "git_at_acquisition": _git_custody(),
            "authority": dict(AUTHORITY),
        }
    )
    path = checkpoint_dir / f"lock_acquisition_{time.time_ns()}_{uuid.uuid4().hex[:12]}.json"
    v4._atomic_checkpoint(path, payload)
    return path


@contextlib.contextmanager
def exclusive_run_lock(out: Path):
    """Reuse the reviewed nonblocking lock under the static namespace."""

    original = v4.checkpoint_namespace
    v4.checkpoint_namespace = checkpoint_namespace
    try:
        with v4.exclusive_run_lock(out) as owner:
            yield owner
    finally:
        v4.checkpoint_namespace = original


def stage_strategy(stage: str, selected: Mapping[str, Any]) -> tuple[str, int]:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown static stage: {stage}")
    if stage.startswith("baseline"):
        return "eager_nchw_autograd", int(selected["baseline_threads"])
    return str(selected["strategy"]), int(selected["threads"])


def is_distinct_arm(selected: Mapping[str, Any]) -> bool:
    return not (
        selected["strategy"] == "eager_nchw_autograd"
        and int(selected["threads"]) == int(selected["baseline_threads"])
    )


def static_admission(
    *,
    n_pairs: int,
    selected: Mapping[str, Any],
    sequence_sha256: Mapping[str, str],
    total_flip_count: int | None,
    baseline_median_ms: float,
    selected_median_ms: float,
    matched_sign_pvalue: float,
    replay_complete: bool = False,
    replay_sequence_sha256: Mapping[str, str] | None = None,
    process_segments_per_pass: Mapping[str, int] | None = None,
    independent_processes: bool = False,
) -> bool:
    measurement_shas = [str(sequence_sha256.get(stage, "")) for stage in STAGE_ORDER]
    replay = replay_sequence_sha256 or {}
    replay_shas = [str(replay.get(stage, "")) for stage in STAGE_ORDER]
    segment_counts = process_segments_per_pass or {}
    return (
        n_pairs == VERDICT_PAIR_CARDINALITY
        and bool(measurement_shas[0])
        and len(set(measurement_shas + replay_shas)) == 1
        and total_flip_count == 0
        and replay_complete
        and independent_processes
        and all(int(segment_counts.get(stage, 0)) == 1 for stage in STAGE_ORDER)
        and is_distinct_arm(selected)
        and selected_median_ms < baseline_median_ms
        and matched_sign_pvalue <= MATCHED_SIGN_ALPHA
    )


def diagnostic_verdict(*, admitted: bool, n_pairs: int) -> str:
    if n_pairs != VERDICT_PAIR_CARDINALITY:
        return "DIAGNOSTIC_ONLY"
    return "GO" if admitted else "NO-GO"


def receipt_verdict_scope(n_pairs: int) -> str:
    return (
        f"fresh-child process-static ABBA formulation over first {n_pairs} receiver-realized pairs "
        "on the fingerprinted local macOS CPU/Torch build only; n<600 diagnostic; no transfer to "
        "another host/build/model/input set, backward, full training, contest-CPU/CUDA, evaluator, "
        "d_seg, d_pose, archive, score, or promotion"
    )


def _failure_scope(build: str | None = None) -> str:
    build_suffix = f" Torch build {build}" if build else " fingerprinted local Torch build"
    return (
        "fresh-child process-static ABBA timing formulation on the fingerprinted local macOS CPU/"
        f"{build_suffix} only; not a mechanism-family, trainer, backward, contest-CPU/CUDA, "
        "d_seg, d_pose, archive, evaluator, score, or promotion verdict"
    )


def write_failure_checkpoint(
    *,
    checkpoint_dir: Path,
    fingerprint_sha256: str,
    stage: str,
    pair_index: int,
    expected_sha256: str | None,
    observed_sha256: str | None,
    flip_count: int | None,
    reason: str,
    torch_build: str | None = None,
) -> Path:
    """Atomically append a unique false-authority failure marker."""

    payload = {
        "schema": FAILURE_SCHEMA,
        "written_at_utc": v4.utc_now(),
        "fingerprint_sha256": fingerprint_sha256,
        "stage": stage,
        "pair_index": pair_index,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed_sha256,
        "argmax_flip_count": flip_count,
        "argmax_flip_count_authority": (
            "MEASURED" if flip_count is not None else "UNAVAILABLE_SHA_MISMATCH_FAIL_CLOSED"
        ),
        "reason": reason,
        "verdict": "FAIL-CLOSED",
        "verdict_scope": _failure_scope(torch_build),
        "authority": dict(AUTHORITY),
        "research_only": True,
    }
    payload = _with_sha(payload)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    nonce = f"{time.time_ns()}_{uuid.uuid4().hex[:12]}"
    path = checkpoint_dir / f"failure_{stage}_{max(pair_index, 0):06d}_{nonce}.json"
    v4._atomic_checkpoint(path, payload)
    return path


def _stage_path(checkpoint_dir: Path, stage: str, mode: str) -> Path:
    if stage not in STAGE_ORDER or mode not in WORKER_MODES:
        raise ValueError("unknown stage/mode")
    return checkpoint_dir / f"stage_{STAGE_ORDER.index(stage):02d}_{stage}_{mode}.json"


def _worker_config_path(checkpoint_dir: Path, stage: str, mode: str) -> Path:
    return checkpoint_dir / f"worker_config_{STAGE_ORDER.index(stage):02d}_{stage}_{mode}.json"


def _thread_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def build_run_fingerprint(
    *,
    raw: Path,
    n_pairs: int,
    seed: int,
    interop_threads: int,
    canary_indices: Sequence[int],
    selection_key: Mapping[str, Any],
    selected: Mapping[str, Any],
    start_custody: Mapping[str, Any],
    policy_contracts: Mapping[str, Any],
    out: Path,
    raw_argv: Sequence[str],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "raw_path": str(raw),
        "raw_bytes": int(start_custody["raw_bytes"]),
        "raw_sha256": str(start_custody["raw_sha256"]),
        "weights_sha256": str(start_custody["weights_sha256"]),
        "tool_sha256": sha256_file(TOOL_PATH),
        "v4_tool_sha256": sha256_file(V4_PATH),
        "dependency_sha256": str(start_custody["dependency_sha256"]),
        "python_executable": active_python_executable(),
        "python_executable_resolved_target": str(Path(sys.executable).resolve()),
        "n_pairs": n_pairs,
        "seed": seed,
        "interop_threads": interop_threads,
        "canary_indices": list(canary_indices),
        "stage_order": list(STAGE_ORDER),
        "selected_arm": dict(selected),
        "selection_key_sha256": selection_key["sha256"],
        "policy_contracts_sha256": sha256_json(policy_contracts),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_interval_provenance": CHECKPOINT_INTERVAL_PROVENANCE,
        "matched_sign_alpha": MATCHED_SIGN_ALPHA,
        "matched_sign_alpha_provenance": MATCHED_SIGN_ALPHA_PROVENANCE,
        "method": METHOD,
        "worker_pythonhashseed": str(seed),
        "out_path": str(out),
        "argv": [active_python_executable(), str(TOOL_PATH), *raw_argv],
        "run_identity": dict(run_identity),
        "run_identity_sha256": run_identity["sha256"],
        "thread_environment": _thread_environment(),
    }
    return {"payload": payload, "sha256": sha256_json(payload)}


def _stage_config(
    *,
    checkpoint_dir: Path,
    raw: Path,
    n_pairs: int,
    seed: int,
    interop_threads: int,
    stage: str,
    selected: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    start_custody: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    strategy, intraop_threads = stage_strategy(stage, selected)
    measurement_path = _stage_path(checkpoint_dir, stage, "measurement")
    payload: dict[str, Any] = {
        "schema": WORKER_CONFIG_SCHEMA,
        "mode": mode,
        "stage": stage,
        "raw_path": str(raw),
        "raw_bytes": int(start_custody["raw_bytes"]),
        "expected_raw_sha256": str(start_custody["raw_sha256"]),
        "n_pairs": n_pairs,
        "seed": seed,
        "strategy": strategy,
        "intraop_threads": intraop_threads,
        "interop_threads": interop_threads,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_interval_provenance": CHECKPOINT_INTERVAL_PROVENANCE,
        "fingerprint_sha256": fingerprint["sha256"],
        "expected_weights_sha256": str(start_custody["weights_sha256"]),
        "expected_torch_build": str(start_custody["build"]["torch_runtime"]["version"]),
        "expected_python_executable": active_python_executable(),
        "expected_tool_sha256": sha256_file(TOOL_PATH),
        "expected_v4_tool_sha256": sha256_file(V4_PATH),
        "pythonhashseed": str(seed),
        "output_path": str(_stage_path(checkpoint_dir, stage, mode)),
        "measurement_path": str(measurement_path),
        "measurement_sha256": (
            sha256_file(measurement_path) if mode == "replay" and measurement_path.is_file() else None
        ),
        "method": METHOD,
        "thread_environment": _thread_environment(),
    }
    return _with_sha(payload)


def _load_worker_config(path: Path, *, verify_raw_sha: bool = True) -> dict[str, Any]:
    config = json.loads(path.read_text())
    _validate_digest(config, label="worker config")
    if config.get("schema") != WORKER_CONFIG_SCHEMA:
        raise RuntimeError("worker config schema mismatch")
    if config.get("mode") not in WORKER_MODES or config.get("stage") not in STAGE_ORDER:
        raise RuntimeError("worker config stage/mode mismatch")
    if config.get("method") != METHOD:
        raise RuntimeError("worker config process method mismatch")
    if config["expected_python_executable"] != active_python_executable():
        raise RuntimeError("worker Python executable mismatch")
    if config.get("expected_tool_sha256") != sha256_file(TOOL_PATH):
        raise RuntimeError("worker static tool SHA mismatch")
    if config.get("expected_v4_tool_sha256") != sha256_file(V4_PATH):
        raise RuntimeError("worker v4 tool SHA mismatch")
    raw = Path(config["raw_path"])
    if not raw.is_file() or raw.stat().st_size != int(config["raw_bytes"]):
        raise RuntimeError("worker raw input size/path mismatch")
    if verify_raw_sha and sha256_file(raw) != config.get("expected_raw_sha256"):
        raise RuntimeError("worker raw input SHA mismatch")
    if verify_raw_sha and os.environ.get("PYTHONHASHSEED") != config.get("pythonhashseed"):
        raise RuntimeError("worker PYTHONHASHSEED does not match the fingerprinted seed")
    output = base.validate_durable_output(Path(config["output_path"]))
    if output != Path(config["output_path"]).resolve():
        raise RuntimeError("worker output path is not canonical")
    return config


def _bind_static_threads(config: Mapping[str, Any]) -> tuple[Any, dict[str, int]]:
    """Bind once, before model load; no other function may mutate threads."""

    import torch

    torch.set_num_interop_threads(int(config["interop_threads"]))
    torch.set_num_threads(int(config["intraop_threads"]))
    binding = {
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
    }
    expected = {
        "intraop_threads": int(config["intraop_threads"]),
        "interop_threads": int(config["interop_threads"]),
    }
    if binding != expected:
        raise RuntimeError(f"static worker thread binding failed: expected={expected} observed={binding}")
    return torch, binding


def _assert_static_binding(torch_module: Any, expected: Mapping[str, int]) -> dict[str, int]:
    observed = {
        "intraop_threads": int(torch_module.get_num_threads()),
        "interop_threads": int(torch_module.get_num_interop_threads()),
    }
    if observed != dict(expected):
        raise RuntimeError(f"static worker thread binding drift: expected={dict(expected)} observed={observed}")
    return observed


def _forward_fixed(model: Any, model_input: Any, *, strategy: str) -> tuple[np.ndarray, float]:
    import torch

    sample = model_input.detach().clone().requires_grad_(True)
    if strategy == "eager_channels_last_autograd":
        sample = sample.to(memory_format=torch.channels_last)
    started = time.perf_counter_ns()
    logits = model(sample)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if not logits.requires_grad or logits.grad_fn is None:
        raise RuntimeError("process-static forward did not preserve the input-gradient graph")
    array = np.ascontiguousarray(logits.detach().argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False))
    return array, elapsed_ms


def _empty_stage_record(config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(config["mode"])
    return {
        "schema": STAGE_SCHEMA,
        "mode": mode,
        "stage": config["stage"],
        "strategy": config["strategy"],
        "intraop_threads": int(config["intraop_threads"]),
        "interop_threads": int(config["interop_threads"]),
        "fingerprint_sha256": config["fingerprint_sha256"],
        "config_sha256": config["sha256"],
        "completed_pairs": 0,
        "timings_ms": [],
        "pair_sha256": [],
        "process_segments": [],
        "measurement_complete": False,
        "replay_complete": False,
        "sequence_sha256": None,
        "pair78_sha256": None,
        "total_argmax_pixels": 0,
        "argmax_tensor_bytes_persisted": 0,
    }


def validate_stage_record(
    record: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    require_complete: bool = False,
) -> int:
    _validate_digest(record, label=f"{config['stage']} {config['mode']} stage")
    checks = {
        "schema": record.get("schema") == STAGE_SCHEMA,
        "mode": record.get("mode") == config["mode"],
        "stage": record.get("stage") == config["stage"],
        "strategy": record.get("strategy") == config["strategy"],
        "intraop": record.get("intraop_threads") == int(config["intraop_threads"]),
        "interop": record.get("interop_threads") == int(config["interop_threads"]),
        "fingerprint": record.get("fingerprint_sha256") == config["fingerprint_sha256"],
        "config": record.get("config_sha256") == config["sha256"],
        "no_argmax_bulk": record.get("argmax_tensor_bytes_persisted") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"stage custody mismatch {config['stage']}/{config['mode']}: {failed}")
    completed = record.get("completed_pairs")
    n_pairs = int(config["n_pairs"])
    if not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= n_pairs:
        raise RuntimeError("stage completed_pairs invalid")
    pair_shas = record.get("pair_sha256")
    timings = record.get("timings_ms")
    if not isinstance(pair_shas, list) or len(pair_shas) != completed or not all(
        _is_sha256(value) for value in pair_shas
    ):
        raise RuntimeError("stage pair SHA vector invalid")
    expected_timing_count = completed if config["mode"] == "measurement" else 0
    if not isinstance(timings, list) or len(timings) != expected_timing_count or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in timings
    ):
        raise RuntimeError("stage timing vector invalid")
    segments = record.get("process_segments")
    if not isinstance(segments, list) or not all(isinstance(row, Mapping) for row in segments):
        raise RuntimeError("stage process segments invalid")
    expected_binding = {
        "intraop_threads": int(config["intraop_threads"]),
        "interop_threads": int(config["interop_threads"]),
    }
    child_ids: set[str] = set()
    for index, segment in enumerate(segments):
        child_id = segment.get("child_id")
        pid = segment.get("pid")
        started_from = segment.get("started_from_completed_pairs")
        if (
            not isinstance(child_id, str)
            or not child_id
            or child_id in child_ids
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(started_from, int)
            or isinstance(started_from, bool)
            or not 0 <= started_from <= completed
            or segment.get("binding") != expected_binding
            or segment.get("python_executable") != config["expected_python_executable"]
            or segment.get("pythonhashseed") != config["pythonhashseed"]
            or not isinstance(segment.get("started_at_utc"), str)
        ):
            raise RuntimeError(f"stage process segment {index} custody invalid")
        child_ids.add(child_id)
    complete_field = "measurement_complete" if config["mode"] == "measurement" else "replay_complete"
    complete = record.get(complete_field) is True
    if complete:
        if completed != n_pairs or not _is_sha256(record.get("sequence_sha256")):
            raise RuntimeError("complete stage lacks terminal cardinality/sequence SHA")
        expected_pixels = n_pairs * base.SEG_H * base.SEG_W
        if record.get("total_argmax_pixels") != expected_pixels:
            raise RuntimeError("stage total argmax pixels mismatch")
        if n_pairs > 78 and not _is_sha256(record.get("pair78_sha256")):
            raise RuntimeError("stage pair78 SHA missing")
        binding_before = record.get("binding_before")
        binding_after = record.get("binding_after")
        if binding_before != expected_binding or binding_after != expected_binding:
            raise RuntimeError("stage terminal binding custody mismatch")
        if not segments:
            raise RuntimeError("complete stage needs process custody")
        terminal = segments[-1]
        if (
            record.get("terminal_child_id") != terminal["child_id"]
            or record.get("terminal_pid") != terminal["pid"]
            or terminal.get("completed_pairs") != n_pairs
            or not isinstance(terminal.get("completed_at_utc"), str)
        ):
            raise RuntimeError("terminal child identity does not match final process segment")
    elif record.get("sequence_sha256") is not None:
        raise RuntimeError("incomplete stage cannot carry terminal sequence SHA")
    if require_complete and not complete:
        raise RuntimeError("terminal stage is incomplete")
    return completed


def _atomic_stage(path: Path, record: Mapping[str, Any]) -> None:
    v4._atomic_checkpoint(path, _with_sha(record))


def _load_or_initialize_stage(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["output_path"])
    if path.is_file():
        record = json.loads(path.read_text())
        validate_stage_record(record, config=config)
        return _without_sha(record)
    return _empty_stage_record(config)


def _persist_process_launch(
    *, record: dict[str, Any], segment: Mapping[str, Any], output_path: Path
) -> None:
    """Bank process identity before prefix replay, warm-up, or measured work."""

    record["process_segments"].append(dict(segment))
    record["written_at_utc"] = v4.utc_now()
    record["sequence_sha256"] = None
    record["measurement_complete"] = False
    record["replay_complete"] = False
    _atomic_stage(output_path, record)


def _complete_process_segment(
    *, record: dict[str, Any], child_id: str, pid: int, n_pairs: int, completed_at_utc: str
) -> None:
    """Complete the exact segment copy previously banked in the stage record."""

    if not record["process_segments"]:
        raise RuntimeError("cannot complete a missing process segment")
    terminal_segment = record["process_segments"][-1]
    if terminal_segment.get("child_id") != child_id or terminal_segment.get("pid") != pid:
        raise RuntimeError("banked process segment identity drift")
    terminal_segment["completed_at_utc"] = completed_at_utc
    terminal_segment["completed_pairs"] = n_pairs


def _measurement_reference(config: Mapping[str, Any]) -> dict[str, Any] | None:
    if config["mode"] != "replay":
        return None
    path = Path(config["measurement_path"])
    if not path.is_file() or sha256_file(path) != config.get("measurement_sha256"):
        raise RuntimeError("replay measurement stage file custody mismatch")
    measurement_config_path = _worker_config_path(path.parent, str(config["stage"]), "measurement")
    measurement_config = _load_worker_config(measurement_config_path, verify_raw_sha=False)
    record = json.loads(path.read_text())
    validate_stage_record(record, config=measurement_config, require_complete=True)
    return record


def run_worker(config_path: Path) -> dict[str, Any]:
    """Execute one process-static measurement or replay stage."""

    config = _load_worker_config(config_path.resolve())
    torch, binding = _bind_static_threads(config)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    raw = Path(config["raw_path"])
    model, weights = base._load_model()
    if sha256_file(weights) != config["expected_weights_sha256"]:
        raise RuntimeError("worker model-weight SHA mismatch")
    if str(torch.__version__) != config["expected_torch_build"]:
        raise RuntimeError("worker Torch build mismatch")
    stage_model = v4._strategy_model(model, str(config["strategy"]))
    _assert_static_binding(torch, binding)
    record = _load_or_initialize_stage(config)
    reference = _measurement_reference(config)
    output_path = Path(config["output_path"])
    completed = validate_stage_record(_with_sha(record), config=config)
    child_id = uuid.uuid4().hex
    segment = {
        "child_id": child_id,
        "pid": os.getpid(),
        "started_at_utc": v4.utc_now(),
        "started_from_completed_pairs": completed,
        "binding": dict(binding),
        "python_executable": active_python_executable(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    _persist_process_launch(record=record, segment=segment, output_path=output_path)

    # Recompute any prefix inside this same fixed-binding child.  This both
    # authenticates resume bytes and reconstructs the raw ordered digest.
    digest = hashlib.sha256()
    for pair_index in range(completed):
        inp = base._model_input(model, base._read_frame1(raw, pair_index))
        array, _ = _forward_fixed(stage_model, inp, strategy=str(config["strategy"]))
        observed = hashlib.sha256(array.tobytes()).hexdigest()
        expected = record["pair_sha256"][pair_index]
        if observed != expected:
            write_failure_checkpoint(
                checkpoint_dir=output_path.parent,
                fingerprint_sha256=str(config["fingerprint_sha256"]),
                stage=f"{config['stage']}_{config['mode']}_resume",
                pair_index=pair_index,
                expected_sha256=expected,
                observed_sha256=observed,
                flip_count=None,
                reason="process-static resume prefix SHA drift",
                torch_build=str(torch.__version__),
            )
            raise RuntimeError(f"resume replay drift at pair {pair_index}")
        if reference is not None and observed != reference["pair_sha256"][pair_index]:
            raise RuntimeError(f"independent replay differs from measurement at pair {pair_index}")
        digest.update(array.tobytes())

    # A single unmeasured warm-up establishes the selected memory layout.  It
    # occurs only for a brand-new stage and is never included in terminal time.
    if completed == 0:
        warm_input = base._model_input(model, base._read_frame1(raw, 0))
        _forward_fixed(stage_model, warm_input, strategy=str(config["strategy"]))
        record["warmup_pairs"] = 1
        record["warmup_authority"] = "DERIVED_ONE_UNMEASURED_FIRST_PAIR_PER_FRESH_STAGE"

    n_pairs = int(config["n_pairs"])
    interval = int(config["checkpoint_interval"])
    for pair_index in range(completed, n_pairs):
        _assert_static_binding(torch, binding)
        inp = base._model_input(model, base._read_frame1(raw, pair_index))
        array, elapsed_ms = _forward_fixed(stage_model, inp, strategy=str(config["strategy"]))
        pair_sha = hashlib.sha256(array.tobytes()).hexdigest()
        if reference is not None and pair_sha != reference["pair_sha256"][pair_index]:
            write_failure_checkpoint(
                checkpoint_dir=output_path.parent,
                fingerprint_sha256=str(config["fingerprint_sha256"]),
                stage=f"{config['stage']}_replay",
                pair_index=pair_index,
                expected_sha256=reference["pair_sha256"][pair_index],
                observed_sha256=pair_sha,
                flip_count=None,
                reason="independent full replay pair SHA drift",
                torch_build=str(torch.__version__),
            )
            raise RuntimeError(f"independent replay drift at pair {pair_index}")
        record["pair_sha256"].append(pair_sha)
        if config["mode"] == "measurement":
            record["timings_ms"].append(elapsed_ms)
        record["completed_pairs"] = pair_index + 1
        digest.update(array.tobytes())
        # The n_pairs checkpoint is intentionally incomplete until the raw
        # sequence digest and terminal bindings exist.  A crash here resumes.
        if (pair_index + 1) % interval == 0 or pair_index + 1 == n_pairs:
            record["written_at_utc"] = v4.utc_now()
            record["sequence_sha256"] = None
            record["measurement_complete"] = False
            record["replay_complete"] = False
            _atomic_stage(output_path, record)

    terminal_binding = _assert_static_binding(torch, binding)
    record["sequence_sha256"] = digest.hexdigest()
    record["pair78_sha256"] = record["pair_sha256"][78] if n_pairs > 78 else None
    record["total_argmax_pixels"] = n_pairs * base.SEG_H * base.SEG_W
    record["binding_before"] = dict(binding)
    record["binding_after"] = terminal_binding
    record["measurement_complete"] = config["mode"] == "measurement"
    record["replay_complete"] = config["mode"] == "replay"
    record["terminal_child_id"] = child_id
    record["terminal_pid"] = os.getpid()
    record["written_at_utc"] = v4.utc_now()
    _complete_process_segment(
        record=record,
        child_id=child_id,
        pid=os.getpid(),
        n_pairs=n_pairs,
        completed_at_utc=record["written_at_utc"],
    )
    _atomic_stage(output_path, record)
    persisted = json.loads(output_path.read_text())
    validate_stage_record(persisted, config=config, require_complete=True)
    return persisted


def _launch_worker(config_path: Path) -> dict[str, Any]:
    command = [active_python_executable(), str(TOOL_PATH), "--_worker-config", str(config_path)]
    config = json.loads(config_path.read_text())
    child_env = os.environ.copy()
    child_env["PYTHONHASHSEED"] = str(config["pythonhashseed"])
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=child_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"static child failed rc={completed.returncode} config={config_path}: "
            f"stdout={completed.stdout[-2000:]} stderr={completed.stderr[-4000:]}"
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"static child emitted malformed terminal output: {completed.stdout[-2000:]}") from exc
    stage_path = Path(result["stage_path"])
    if not stage_path.is_file() or sha256_file(stage_path) != result["stage_file_sha256"]:
        raise RuntimeError("static child terminal stage output custody mismatch")
    return result


def per_pair_replica_medians(stage_records: Mapping[str, Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    baseline = [
        statistics.median(values)
        for values in zip(
            stage_records["baseline_rep0"]["timings_ms"],
            stage_records["baseline_rep1"]["timings_ms"],
            strict=True,
        )
    ]
    selected = [
        statistics.median(values)
        for values in zip(
            stage_records["selected_rep0"]["timings_ms"],
            stage_records["selected_rep1"]["timings_ms"],
            strict=True,
        )
    ]
    return baseline, selected


def _pair_sha_evidence(
    measurements: Mapping[str, Mapping[str, Any]],
    replays: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    n_pairs = int(measurements[STAGE_ORDER[0]]["completed_pairs"])
    mismatches: list[dict[str, Any]] = []
    for pair_index in range(n_pairs):
        values = {
            f"measurement:{stage}": measurements[stage]["pair_sha256"][pair_index]
            for stage in STAGE_ORDER
        } | {
            f"replay:{stage}": replays[stage]["pair_sha256"][pair_index]
            for stage in STAGE_ORDER
        }
        if len(set(values.values())) != 1:
            mismatches.append({"pair_index": pair_index, "sha256": values})
    all_equal = not mismatches
    return {
        "all_pair_sha256_equal": all_equal,
        "mismatch_pair_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "derived_argmax_flip_count": 0 if all_equal else None,
        "flip_count_derivation": (
            "DERIVED_ZERO_FROM_EIGHT_WAY_EXACT_PER_PAIR_SHA_EQUALITY"
            if all_equal
            else "UNAVAILABLE_SHA_MISMATCH_FAIL_CLOSED_NO_RAW_PRIOR_TENSOR"
        ),
    }


def _read_stage_bundle(
    checkpoint_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    measurements: dict[str, dict[str, Any]] = {}
    replays: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        for mode, target in (("measurement", measurements), ("replay", replays)):
            config_path = _worker_config_path(checkpoint_dir, stage, mode)
            config = _load_worker_config(config_path, verify_raw_sha=False)
            record_path = _stage_path(checkpoint_dir, stage, mode)
            record = json.loads(record_path.read_text())
            validate_stage_record(record, config=config, require_complete=True)
            target[stage] = record
            configs[f"{stage}:{mode}"] = config
    return measurements, replays, configs


def _derive_measurement(
    *,
    args: argparse.Namespace,
    selected: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, Any]],
    replays: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    baseline_values, selected_values = per_pair_replica_medians(measurements)
    baseline_summary = v4.summarize_ms(baseline_values)
    selected_summary = v4.summarize_ms(selected_values)
    sign_test = v4.matched_sign_test(baseline_values, selected_values)
    measurement_shas = {stage: measurements[stage]["sequence_sha256"] for stage in STAGE_ORDER}
    replay_shas = {stage: replays[stage]["sequence_sha256"] for stage in STAGE_ORDER}
    pair_evidence = _pair_sha_evidence(measurements, replays)
    segment_counts = {stage: len(measurements[stage]["process_segments"]) for stage in STAGE_ORDER}
    all_sequences_equal = len({*measurement_shas.values(), *replay_shas.values()}) == 1
    replay_complete = all(
        replays[stage]["replay_complete"] is True
        and len(replays[stage]["process_segments"]) == 1
        for stage in STAGE_ORDER
    )
    measurement_child_ids = [measurements[stage]["terminal_child_id"] for stage in STAGE_ORDER]
    replay_child_ids = [replays[stage]["terminal_child_id"] for stage in STAGE_ORDER]
    measurement_pids = [measurements[stage]["terminal_pid"] for stage in STAGE_ORDER]
    replay_pids = [replays[stage]["terminal_pid"] for stage in STAGE_ORDER]
    independent_processes = (
        len({*measurement_child_ids, *replay_child_ids}) == 8
        and len({*measurement_pids, *replay_pids}) == 8
    )
    admitted = static_admission(
        n_pairs=args.n_pairs,
        selected=selected,
        sequence_sha256=measurement_shas,
        replay_sequence_sha256=replay_shas,
        total_flip_count=pair_evidence["derived_argmax_flip_count"],
        baseline_median_ms=float(baseline_summary["median_ms"]),
        selected_median_ms=float(selected_summary["median_ms"]),
        matched_sign_pvalue=float(sign_test["one_sided_exact_binomial_pvalue"]),
        replay_complete=replay_complete,
        process_segments_per_pass=segment_counts,
        independent_processes=independent_processes,
    )
    pass_receipts = {}
    for stage in STAGE_ORDER:
        measurement = measurements[stage]
        replay = replays[stage]
        measurement_path = _stage_path(Path(args.checkpoint_dir), stage, "measurement")
        replay_path = _stage_path(Path(args.checkpoint_dir), stage, "replay")
        pass_receipts[stage] = {
            "measurement_child_id": measurement["terminal_child_id"],
            "measurement_pid": measurement["terminal_pid"],
            "measurement_process_segments": measurement["process_segments"],
            "replay_child_ids": [row["child_id"] for row in replay["process_segments"]],
            "replay_pids": [row["pid"] for row in replay["process_segments"]],
            "replay_process_segments": replay["process_segments"],
            "strategy": measurement["strategy"],
            "intraop_threads": measurement["intraop_threads"],
            "interop_threads": measurement["interop_threads"],
            "measurement_complete": measurement["measurement_complete"],
            "replay_complete": replay["replay_complete"],
            "measurement_sequence_sha256": measurement["sequence_sha256"],
            "replay_sequence_sha256": replay["sequence_sha256"],
            "binding_before": measurement["binding_before"],
            "binding_after": measurement["binding_after"],
            "replay_binding_before": replay["binding_before"],
            "replay_binding_after": replay["binding_after"],
            "terminal_stage_file": {
                "path": str(measurement_path),
                "sha256": sha256_file(measurement_path),
            },
            "terminal_replay_file": {
                "path": str(replay_path),
                "sha256": sha256_file(replay_path),
            },
            "derived_argmax_flip_count": (
                0
                if measurement["pair_sha256"] == replay["pair_sha256"]
                and pair_evidence["all_pair_sha256_equal"]
                else None
            ),
        }
    pair78_measurement = {
        stage: measurements[stage]["pair78_sha256"] for stage in STAGE_ORDER
    }
    pair78_replay = {stage: replays[stage]["pair78_sha256"] for stage in STAGE_ORDER}
    pair78_stable = args.n_pairs > 78 and len(
        {*pair78_measurement.values(), *pair78_replay.values()}
    ) == 1
    measurement = {
        "label": "MEASURED",
        "method": METHOD,
        "n_real_pairs": args.n_pairs,
        "total_argmax_pixels": args.n_pairs * base.SEG_H * base.SEG_W,
        "stage_order": list(STAGE_ORDER),
        "child_passes": list(STAGE_ORDER),
        "thread_binding": "fresh process per measurement and replay; intra/inter-op immutable after pre-model binding",
        "baseline_per_pair_replica_median": baseline_summary,
        "selected_per_pair_replica_median": selected_summary,
        "static_paired_speedup_x": float(baseline_summary["median_ms"])
        / float(selected_summary["median_ms"]),
        "matched_sign_test": sign_test,
        "matched_sign_alpha": MATCHED_SIGN_ALPHA,
        "matched_sign_alpha_provenance": MATCHED_SIGN_ALPHA_PROVENANCE,
        "sequence_sha256": measurement_shas,
        "replay_sequence_sha256": replay_shas,
        "all_sequence_shas_equal": all_sequences_equal,
        "argmax_flip_count": pair_evidence["derived_argmax_flip_count"],
        "derived_argmax_flip_count": pair_evidence["derived_argmax_flip_count"],
        "argmax_flip_rate": 0.0 if pair_evidence["derived_argmax_flip_count"] == 0 else None,
        "pair_sha_evidence": pair_evidence,
        "input_gradient_graph_preserved": True,
        "process_segments_per_pass": segment_counts,
        "pass_receipts": pass_receipts,
        "independent_full_replays": {
            "count": 4,
            "per_arm_count": 2,
            "complete": replay_complete,
            "independent_processes": independent_processes,
            "unique_child_id_count": len({*measurement_child_ids, *replay_child_ids}),
            "unique_pid_count": len({*measurement_pids, *replay_pids}),
            "sha_equal": all_sequences_equal and pair_evidence["all_pair_sha256_equal"],
        },
        "pair78": {
            "index": 78 if args.n_pairs > 78 else None,
            "prior_margin": 2.384185791015625e-7 if args.n_pairs > 78 else None,
            "prior_margin_label": "MEASURED_PRIOR_ALTERNATING_ORDER_DIAGNOSTIC",
            "prior_location_yx": [275, 356] if args.n_pairs > 78 else None,
            "prior_classes": [0, 1] if args.n_pairs > 78 else None,
            "per_pass_sha256": pair78_measurement,
            "per_replay_sha256": pair78_replay,
            "stable": pair78_stable,
            "resolved": pair78_stable and all_sequences_equal and pair_evidence["all_pair_sha256_equal"],
            "resolution_scope": (
                "alternating in-process thread-switch order confound only; mechanism transfer remains local advisory"
            ),
        },
    }
    return measurement, admitted


def _validate_latest(
    latest: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    fingerprint: Mapping[str, Any],
) -> None:
    _validate_digest(latest, label="latest manifest")
    if latest.get("schema") != LATEST_SCHEMA or latest.get("fingerprint_sha256") != fingerprint["sha256"]:
        raise RuntimeError("latest manifest schema/fingerprint mismatch")
    expected = {}
    for stage in STAGE_ORDER:
        for mode in WORKER_MODES:
            path = _stage_path(checkpoint_dir, stage, mode)
            expected[f"{stage}:{mode}"] = {"path": str(path), "sha256": sha256_file(path)}
    if latest.get("terminal_stage_files") != expected or latest.get("complete") is not True:
        raise RuntimeError("latest manifest does not bind every terminal stage file")


def validate_static_plan_contract(
    *,
    receipt: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    expected_selected: Mapping[str, Any],
) -> None:
    """Bind receipt claims and every child config to the fingerprinted plan."""

    selected = receipt.get("selected_arm")
    fingerprint_selected = fingerprint["payload"]["selected_arm"]
    run_identity = fingerprint["payload"]["run_identity"]
    runtime = receipt.get("runtime")
    custody = receipt.get("custody")
    checks: dict[str, bool] = {
        "selected_fingerprint": selected == fingerprint_selected,
        "selected_canary": selected == expected_selected,
        "axis": receipt.get("axis") == AXIS,
        "verdict_scope": receipt.get("verdict_scope")
        == receipt_verdict_scope(int(fingerprint["payload"]["n_pairs"])),
        "evidence_labels": receipt.get("labels") == EVIDENCE_LABELS,
        "authority": receipt.get("authority") == AUTHORITY,
        "runtime_false_authority": isinstance(runtime, Mapping)
        and runtime.get("mps_used") is False
        and runtime.get("cuda_used") is False
        and runtime.get("contest_cpu_timing_measured") is False,
        "runtime_python": isinstance(runtime, Mapping)
        and runtime.get("python") == sys.version
        and runtime.get("python_executable") == fingerprint["payload"]["python_executable"]
        and runtime.get("python_executable_resolved_target")
        == fingerprint["payload"]["python_executable_resolved_target"],
        "runtime_host": isinstance(runtime, Mapping)
        and runtime.get("platform") == platform.platform()
        and runtime.get("machine") == platform.machine(),
        "custody_argv": isinstance(custody, Mapping)
        and custody.get("argv") == run_identity["argv"],
        "custody_git": isinstance(custody, Mapping)
        and custody.get("git") == run_identity["initial_git"],
        "custody_lock": isinstance(custody, Mapping)
        and custody.get("lock_owner") == run_identity["initial_lock_owner"],
    }
    for key, config in configs.items():
        stage, mode = key.split(":")
        expected_strategy, expected_threads = stage_strategy(stage, fingerprint_selected)
        checks[f"config-plan:{key}"] = (
            config.get("stage") == stage
            and config.get("mode") == mode
            and config.get("strategy") == expected_strategy
            and config.get("intraop_threads") == expected_threads
            and config.get("interop_threads") == fingerprint["payload"]["interop_threads"]
            and config.get("n_pairs") == fingerprint["payload"]["n_pairs"]
            and config.get("seed") == fingerprint["payload"]["seed"]
            and config.get("pythonhashseed") == fingerprint["payload"]["worker_pythonhashseed"]
            and config.get("expected_raw_sha256") == fingerprint["payload"]["raw_sha256"]
            and config.get("expected_weights_sha256") == fingerprint["payload"]["weights_sha256"]
            and config.get("fingerprint_sha256") == fingerprint["sha256"]
            and config.get("method") == METHOD
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"static plan contract validation failed: {failed}")


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    current_custody: Mapping[str, Any],
    latest: Mapping[str, Any],
) -> None:
    if receipt.get("schema") != SCHEMA:
        raise RuntimeError("receipt schema mismatch")
    if receipt.get("authority") != AUTHORITY or receipt.get("research_only") is not True:
        raise RuntimeError("receipt authority escalation")
    resume = receipt.get("resume")
    if not isinstance(resume, Mapping):
        raise RuntimeError("receipt resume block missing")
    fingerprint = resume.get("fingerprint")
    if not isinstance(fingerprint, Mapping) or fingerprint.get("sha256") != sha256_json(fingerprint["payload"]):
        raise RuntimeError("receipt fingerprint invalid")
    checkpoint_dir = Path(resume["checkpoint_dir"])
    run_identity_path = Path(resume["run_identity_path"])
    if (
        not run_identity_path.is_file()
        or sha256_file(run_identity_path) != resume["run_identity_file_sha256"]
    ):
        raise RuntimeError("run identity file custody mismatch")
    run_identity = json.loads(run_identity_path.read_text())
    _validate_digest(run_identity, label="run identity")
    if (
        run_identity != fingerprint["payload"]["run_identity"]
        or run_identity["sha256"] != fingerprint["payload"]["run_identity_sha256"]
    ):
        raise RuntimeError("run identity differs from resume fingerprint")
    lock_acquisition_path = Path(resume["lock_acquisition_path"])
    if (
        not lock_acquisition_path.is_file()
        or sha256_file(lock_acquisition_path) != resume["lock_acquisition_file_sha256"]
    ):
        raise RuntimeError("lock acquisition file custody mismatch")
    lock_acquisition = json.loads(lock_acquisition_path.read_text())
    _validate_digest(lock_acquisition, label="lock acquisition")
    if (
        lock_acquisition.get("schema") != LOCK_ACQUISITION_SCHEMA
        or lock_acquisition.get("run_identity_sha256") != run_identity["sha256"]
        or receipt["custody"].get("current_lock_acquisition") != lock_acquisition
    ):
        raise RuntimeError("current lock acquisition is not bound to run identity/receipt")
    _validate_latest(latest, checkpoint_dir=checkpoint_dir, fingerprint=fingerprint)
    measurements, replays, configs = _read_stage_bundle(checkpoint_dir)
    selected = receipt["selected_arm"]
    fingerprint_selected = fingerprint["payload"]["selected_arm"]
    canary_path = Path(resume["canary_path"])
    if not canary_path.is_file() or sha256_file(canary_path) != resume["canary_sha256"]:
        raise RuntimeError("receipt canary file custody mismatch")
    canary = json.loads(canary_path.read_text())
    expected_selected = v4.validate_canary_state(
        canary,
        fingerprint=fingerprint,
        expected_arms=receipt["selection_key"]["payload"]["candidate_arms"],
        baseline_threads=int(fingerprint_selected["baseline_threads"]),
    )
    validate_static_plan_contract(
        receipt=receipt,
        fingerprint=fingerprint,
        configs=configs,
        expected_selected=expected_selected,
    )
    args_proxy = argparse.Namespace(n_pairs=int(fingerprint["payload"]["n_pairs"]), checkpoint_dir=checkpoint_dir)
    derived, admitted = _derive_measurement(
        args=args_proxy,
        selected=selected,
        measurements=measurements,
        replays=replays,
    )
    expected_verdict = diagnostic_verdict(admitted=admitted, n_pairs=args_proxy.n_pairs)
    expected_stage_files = latest["terminal_stage_files"]
    checks = {
        "verdict": receipt.get("verdict") == expected_verdict,
        "measurement": receipt.get("measurement") == derived,
        "stage_files": resume.get("terminal_stage_files") == expected_stage_files,
        "checkpoint_interval": resume.get("checkpoint_interval_pairs") == CHECKPOINT_INTERVAL,
        "checkpoint_provenance": resume.get("checkpoint_interval_provenance")
        == CHECKPOINT_INTERVAL_PROVENANCE,
        "custody": receipt["custody"]["start"] == receipt["custody"]["end"] == current_custody,
        "runtime_torch": receipt["runtime"]["torch"]
        == current_custody["build"]["torch_runtime"]["version"],
        "runtime_cpu": receipt["runtime"]["mps_used"] is False
        and receipt["runtime"]["cuda_used"] is False
        and receipt["runtime"]["contest_cpu_timing_measured"] is False,
        "axis": receipt.get("axis") == AXIS,
        "method": fingerprint["payload"]["method"] == METHOD,
        "policy_contracts": sha256_json(receipt["policy_contracts"])
        == fingerprint["payload"]["policy_contracts_sha256"],
        "selection": receipt["selection_key"]["sha256"]
        == fingerprint["payload"]["selection_key_sha256"],
        "selection_digest": receipt["selection_key"]["sha256"]
        == sha256_json(receipt["selection_key"]["payload"]),
        "selected_fingerprint": selected == fingerprint_selected,
        "selected_canary": selected == expected_selected,
        "canary_rows": receipt.get("canary_tournament") == canary["rows"],
        "topology": receipt.get("topology")
        == receipt["selection_key"]["payload"]["forward_signature"]["thread_topology"],
        "latest_file": resume.get("latest_sha256") == sha256_file(Path(resume["latest_path"])),
    }
    # Config SHA/file custody is part of the terminal proof, not merely an
    # implementation detail.
    for key, config in configs.items():
        stage, mode = key.split(":")
        config_path = _worker_config_path(checkpoint_dir, stage, mode)
        checks[f"config:{key}"] = sha256_file(config_path) == resume["worker_config_file_sha256"][key]
        checks[f"config-digest:{key}"] = config["sha256"] == sha256_json(_without_sha(config))
        expected_strategy, expected_threads = stage_strategy(stage, fingerprint_selected)
        checks[f"config-plan:{key}"] = (
            config["stage"] == stage
            and config["mode"] == mode
            and config["strategy"] == expected_strategy
            and config["intraop_threads"] == expected_threads
            and config["interop_threads"] == fingerprint["payload"]["interop_threads"]
            and config["n_pairs"] == fingerprint["payload"]["n_pairs"]
            and config["seed"] == fingerprint["payload"]["seed"]
            and config["pythonhashseed"] == fingerprint["payload"]["worker_pythonhashseed"]
            and config["expected_raw_sha256"] == fingerprint["payload"]["raw_sha256"]
            and config["expected_weights_sha256"] == fingerprint["payload"]["weights_sha256"]
            and config["method"] == METHOD
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"static transfer receipt validation failed: {failed}")


def _git_custody() -> dict[str, Any]:
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip(),
    }


def _write_worker_config(path: Path, config: Mapping[str, Any]) -> None:
    v4._atomic_checkpoint(path, config)
    persisted = json.loads(path.read_text())
    _validate_digest(persisted, label="persisted worker config")


def _best_effort_unhandled_failure(
    *, checkpoint_dir: Path, reason: str, stage: str, config_path: Path | None = None
) -> None:
    """Preserve a scoped marker without masking the original exception."""

    with contextlib.suppress(Exception):
        fingerprint_sha = "unavailable"
        torch_build = None
        if config_path is not None and config_path.is_file():
            config = json.loads(config_path.read_text())
            fingerprint_sha = str(config.get("fingerprint_sha256", fingerprint_sha))
            torch_build = config.get("expected_torch_build")
        elif (checkpoint_dir / "stage_canary.json").is_file():
            canary = json.loads((checkpoint_dir / "stage_canary.json").read_text())
            fingerprint_sha = str(canary.get("fingerprint_sha256", fingerprint_sha))
        write_failure_checkpoint(
            checkpoint_dir=checkpoint_dir,
            fingerprint_sha256=fingerprint_sha,
            stage=stage,
            pair_index=-1,
            expected_sha256=None,
            observed_sha256=None,
            flip_count=None,
            reason=reason,
            torch_build=str(torch_build) if torch_build is not None else None,
        )


def run_probe(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    *,
    lock_owner: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    raw = args.raw.expanduser().resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)
    available_pairs = base._raw_pair_count(raw)
    if args.n_pairs > available_pairs:
        raise ValueError(f"requested {args.n_pairs} pairs but raw has {available_pairs}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_interop_threads(args.interop_threads)
    baseline_threads = int(torch.get_num_threads())
    topology = v4.runtime_thread_topology(torch)
    arms = v4.derive_candidate_arms(topology)
    model, weights = base._load_model()
    first_input = base._model_input(model, base._read_frame1(raw, 0))
    start_custody = dict(v4.measurement_custody(raw, weights, torch)) | {
        "static_tool_sha256": sha256_file(TOOL_PATH),
        "v4_tool_sha256": sha256_file(V4_PATH),
    }
    policy_contracts = v4.compile_policy_contracts(topology, first_input)
    for contract in policy_contracts.values():
        lifecycle = contract.get("process_lifecycle")
        if not isinstance(lifecycle, Mapping) or lifecycle.get("method") != METHOD:
            raise RuntimeError("typed DSL lacks the process-static lifecycle contract")
        if contract.get("checkpoint_interval_pairs") != CHECKPOINT_INTERVAL:
            raise RuntimeError("typed DSL checkpoint interval differs from runtime")
    signature = v4.forward_signature(
        model,
        first_input,
        torch,
        weights,
        topology=topology,
        runtime_custody=start_custody["build"],
    )
    selection_key = v4.selection_key(signature, arms, policy_contracts=policy_contracts)
    canary_count = v4.derive_canary_count(topology, n_pairs=args.n_pairs)
    canary_indices = base.evenly_spaced_indices(args.n_pairs, canary_count)
    out = base.validate_durable_output(args.out)
    checkpoint_dir = checkpoint_namespace(out)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preflight = v4.storage_preflight(checkpoint_dir, n_pairs=args.n_pairs)
    canary_path = checkpoint_dir / "stage_canary.json"
    run_identity = load_or_create_run_identity(
        checkpoint_dir=checkpoint_dir,
        raw_argv=raw_argv,
        lock_owner=lock_owner,
    )
    lock_acquisition_path = record_lock_acquisition(
        checkpoint_dir=checkpoint_dir,
        run_identity=run_identity,
        lock_owner=lock_owner,
    )
    lock_acquisition = json.loads(lock_acquisition_path.read_text())

    if canary_path.is_file():
        canary = json.loads(canary_path.read_text())
        selected = dict(canary["selected"])
    else:
        rows = v4._canary_tournament(
            model=model,
            raw=raw,
            indices=canary_indices,
            arms=arms,
            baseline_threads=baseline_threads,
        )
        selected = v4.select_canary_arm(rows)
        selected["baseline_threads"] = baseline_threads
        canary = {"schema": v4.CHECKPOINT_SCHEMA, "rows": rows, "selected": selected}
    fingerprint = build_run_fingerprint(
        raw=raw,
        n_pairs=args.n_pairs,
        seed=args.seed,
        interop_threads=args.interop_threads,
        canary_indices=canary_indices,
        selection_key=selection_key,
        selected=selected,
        start_custody=start_custody,
        policy_contracts=policy_contracts,
        out=out,
        raw_argv=raw_argv,
        run_identity=run_identity,
    )
    if canary_path.is_file():
        if canary.get("fingerprint_sha256") != fingerprint["sha256"]:
            raise RuntimeError("canary fingerprint mismatch; refusing stale selection")
        selected = v4.validate_canary_state(
            canary,
            fingerprint=fingerprint,
            expected_arms=arms,
            baseline_threads=baseline_threads,
        )
    else:
        canary["fingerprint_sha256"] = fingerprint["sha256"]
        canary["sha256"] = sha256_json(canary)
        v4._atomic_checkpoint(canary_path, canary)
        selected = v4.validate_canary_state(
            canary,
            fingerprint=fingerprint,
            expected_arms=arms,
            baseline_threads=baseline_threads,
        )

    latest_path = checkpoint_dir / "latest.json"
    for stage in STAGE_ORDER:
        for mode in WORKER_MODES:
            config = _stage_config(
                checkpoint_dir=checkpoint_dir,
                raw=raw,
                n_pairs=args.n_pairs,
                seed=args.seed,
                interop_threads=args.interop_threads,
                stage=stage,
                selected=selected,
                fingerprint=fingerprint,
                start_custody=start_custody,
                mode=mode,
            )
            config_path = _worker_config_path(checkpoint_dir, stage, mode)
            if config_path.is_file():
                existing = json.loads(config_path.read_text())
                if existing != config:
                    raise RuntimeError(f"worker config drift on resume: {stage}/{mode}")
            else:
                _write_worker_config(config_path, config)
            stage_path = _stage_path(checkpoint_dir, stage, mode)
            if stage_path.is_file():
                record = json.loads(stage_path.read_text())
                try:
                    validate_stage_record(record, config=config, require_complete=True)
                    continue
                except RuntimeError:
                    # Incomplete checkpoints are resumed by a fresh static
                    # process.  Malformed/custody failures still fail closed.
                    validate_stage_record(record, config=config, require_complete=False)
            try:
                _launch_worker(config_path)
            except Exception as exc:
                _best_effort_unhandled_failure(
                    checkpoint_dir=checkpoint_dir,
                    config_path=config_path,
                    stage=f"{stage}_{mode}_child_failure",
                    reason=f"fresh child failed closed: {type(exc).__name__}: {exc}",
                )
                raise
            record = json.loads(stage_path.read_text())
            validate_stage_record(record, config=config, require_complete=True)

    terminal_stage_files = {}
    for stage in STAGE_ORDER:
        for mode in WORKER_MODES:
            path = _stage_path(checkpoint_dir, stage, mode)
            terminal_stage_files[f"{stage}:{mode}"] = {"path": str(path), "sha256": sha256_file(path)}
    latest = _with_sha(
        {
            "schema": LATEST_SCHEMA,
            "written_at_utc": v4.utc_now(),
            "fingerprint_sha256": fingerprint["sha256"],
            "terminal_stage_files": terminal_stage_files,
            "complete": True,
        }
    )
    v4._atomic_checkpoint(latest_path, latest)
    latest = json.loads(latest_path.read_text())
    _validate_latest(latest, checkpoint_dir=checkpoint_dir, fingerprint=fingerprint)

    end_custody = dict(v4.measurement_custody(raw, weights, torch)) | {
        "static_tool_sha256": sha256_file(TOOL_PATH),
        "v4_tool_sha256": sha256_file(V4_PATH),
    }
    if end_custody != start_custody:
        write_failure_checkpoint(
            checkpoint_dir=checkpoint_dir,
            fingerprint_sha256=fingerprint["sha256"],
            stage="terminal_custody",
            pair_index=-1,
            expected_sha256=sha256_json(start_custody),
            observed_sha256=sha256_json(end_custody),
            flip_count=None,
            reason="start/end parent custody drift",
            torch_build=str(torch.__version__),
        )
        raise RuntimeError("measurement custody changed during static ABBA probe")

    measurements, replays, _ = _read_stage_bundle(checkpoint_dir)
    args.checkpoint_dir = checkpoint_dir
    measurement, admitted = _derive_measurement(
        args=args,
        selected=selected,
        measurements=measurements,
        replays=replays,
    )
    verdict = diagnostic_verdict(admitted=admitted, n_pairs=args.n_pairs)
    worker_config_hashes = {
        f"{stage}:{mode}": sha256_file(_worker_config_path(checkpoint_dir, stage, mode))
        for stage in STAGE_ORDER
        for mode in WORKER_MODES
    }
    receipt = {
        "schema": SCHEMA,
        "completed_at_utc": v4.utc_now(),
        "verdict": verdict,
        "verdict_scope": receipt_verdict_scope(args.n_pairs),
        "labels": dict(EVIDENCE_LABELS),
        "authority": dict(AUTHORITY),
        "research_only": True,
        "axis": AXIS,
        "topology": topology,
        "policy_contracts": policy_contracts,
        "selection_key": selection_key,
        "canary_tournament": canary["rows"],
        "selected_arm": selected,
        "measurement": measurement,
        "storage": {
            "preflight": preflight,
            "large_artifacts_created": False,
            "argmax_tensor_bytes_persisted": 0,
            "cleanup_policy": "only small atomic JSON custody artifacts; no bulk scratch created",
        },
        "resume": {
            "fingerprint": fingerprint,
            "checkpoint_interval_pairs": CHECKPOINT_INTERVAL,
            "checkpoint_interval_provenance": CHECKPOINT_INTERVAL_PROVENANCE,
            "checkpoint_dir": str(checkpoint_dir),
            "run_identity_path": str(checkpoint_dir / "run_identity.json"),
            "run_identity_file_sha256": sha256_file(checkpoint_dir / "run_identity.json"),
            "lock_acquisition_path": str(lock_acquisition_path),
            "lock_acquisition_file_sha256": sha256_file(lock_acquisition_path),
            "canary_path": str(canary_path),
            "canary_sha256": sha256_file(canary_path),
            "latest_path": str(latest_path),
            "latest_sha256": sha256_file(latest_path),
            "terminal_stage_files": terminal_stage_files,
            "worker_config_file_sha256": worker_config_hashes,
            "argmax_tensor_bytes_persisted": 0,
        },
        "custody": {
            "argv": run_identity["argv"],
            "git": run_identity["initial_git"],
            "lock_owner": run_identity["initial_lock_owner"],
            "current_lock_acquisition": lock_acquisition,
            "start": start_custody,
            "end": end_custody,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": active_python_executable(),
            "python_executable_resolved_target": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "mps_used": False,
            "cuda_used": False,
            "contest_cpu_timing_measured": False,
        },
        "validation": {"status": "self-validated-from-terminal-child-bytes-before-and-after-write"},
    }
    validate_receipt(receipt, current_custody=end_custody, latest=latest)
    base.atomic_write_json(out, receipt)
    persisted = json.loads(out.read_text())
    validate_receipt(persisted, current_custody=end_custody, latest=latest)
    if verdict == "NO-GO":
        first_mismatch = measurement["pair_sha_evidence"]["first_mismatch"]
        write_failure_checkpoint(
            checkpoint_dir=checkpoint_dir,
            fingerprint_sha256=fingerprint["sha256"],
            stage="terminal_n600_no_go",
            pair_index=(first_mismatch or {}).get("pair_index", -1),
            expected_sha256=None,
            observed_sha256=None,
            flip_count=measurement["derived_argmax_flip_count"],
            reason="terminal n600 admission predicate did not pass; receipt preserves exact failed gate fields",
            torch_build=str(torch.__version__),
        )
    return persisted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--n-pairs", type=int, default=VERDICT_PAIR_CARDINALITY)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_pairs < 1:
        parser.error("--n-pairs must be positive")
    if args.interop_threads < 1 or args.seed < 0:
        parser.error("interop threads must be positive and seed non-negative")
    args.out = base.validate_durable_output(args.out)
    if args.out.suffix != ".json":
        parser.error("--out must end in .json")
    return args


def _parse_worker_path(argv: Sequence[str]) -> Path | None:
    if "--_worker-config" not in argv:
        return None
    index = list(argv).index("--_worker-config")
    if index + 1 >= len(argv) or len(argv) != 2:
        raise SystemExit("--_worker-config requires exactly one path")
    return Path(argv[index + 1])


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    worker_path = _parse_worker_path(raw_argv)
    if worker_path is not None:
        try:
            record = run_worker(worker_path)
        except Exception as exc:
            with contextlib.suppress(Exception):
                raw_config = json.loads(worker_path.read_text())
                _best_effort_unhandled_failure(
                    checkpoint_dir=Path(raw_config["output_path"]).parent,
                    config_path=worker_path,
                    stage=f"{raw_config.get('stage', 'unknown')}_{raw_config.get('mode', 'unknown')}_worker_exception",
                    reason=f"worker exception: {type(exc).__name__}: {exc}",
                )
            raise
        stage_path = Path(_load_worker_config(worker_path, verify_raw_sha=False)["output_path"])
        print(
            json.dumps(
                {
                    "stage": record["stage"],
                    "mode": record["mode"],
                    "stage_path": str(stage_path),
                    "stage_file_sha256": sha256_file(stage_path),
                },
                sort_keys=True,
            )
        )
        return 0
    args = parse_args(raw_argv)
    try:
        with exclusive_run_lock(args.out) as lock_owner:
            receipt = run_probe(args, raw_argv, lock_owner=lock_owner)
    except Exception as exc:
        _best_effort_unhandled_failure(
            checkpoint_dir=checkpoint_namespace(args.out),
            stage="parent_unhandled_exception",
            reason=f"parent exception: {type(exc).__name__}: {exc}",
        )
        raise
    print(json.dumps({"verdict": receipt["verdict"], "measurement": receipt["measurement"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
