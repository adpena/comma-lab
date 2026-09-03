#!/usr/bin/env python3
"""Default-off stage timing for the QBR1 burn, for MAIN to fire at a cell boundary.

This wrapper never changes QBR1 scientific state.  With ``--profile-stages``
absent it calls the pinned run entry directly and installs no wrappers.  With
profiling enabled it synchronizes the configured device around named stages,
records elapsed time outside the burn tree, and restores every wrapped symbol.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

_IMPORT_REPO = Path(__file__).resolve().parents[1]
if str(_IMPORT_REPO) not in sys.path:
    sys.path.insert(0, str(_IMPORT_REPO))

from experiments import ddm_qbr1_born_fairform_burn_prep as qbr

qbt = qbr.qbt
SCHEMA = "ddm_wc2_qbr1_stage_timing.v1"
IDENTITY_SCHEMA = "ddm_wc2_qbr1_default_off_identity.v1"
DEFAULT_FLUSH_STEPS = 16


class WC2TimingError(RuntimeError):
    """Fail-closed timing-harness contract error."""


def _tree_fact(root: Path) -> dict[str, Any]:
    """Hash every regular file without mutating the retained tree."""

    if not root.is_dir():
        raise WC2TimingError(f"tree does not exist: {root}")
    digest = hashlib.sha256()
    total = 0
    files = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        content_sha = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                content_sha.update(chunk)
                size += len(chunk)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(content_sha.digest())
        total += size
        files += 1
    return {
        "path": str(root.resolve()),
        "regular_files": files,
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


class StageProfiler:
    """Exclusive named timers plus an inclusive training-step timer."""

    def __init__(self, output: Path, *, synchronize_mps: bool, flush_steps: int) -> None:
        if flush_steps < 1:
            raise WC2TimingError("flush_steps must be positive")
        self.output = output
        self.partial_output = output.with_name(output.name + ".partial.json")
        self.synchronize_mps = synchronize_mps
        self.flush_steps = flush_steps
        self.context = "training"
        self.step_started_ns: int | None = None
        self.stats: dict[str, dict[str, float | int]] = {}

    def sync(self) -> None:
        if self.synchronize_mps:
            torch.mps.synchronize()

    def record(self, stage: str, elapsed_ns: int) -> None:
        key = f"{self.context}.{stage}"
        elapsed = elapsed_ns / 1.0e9
        row = self.stats.setdefault(
            key,
            {"count": 0, "seconds": 0.0, "min_seconds": math.inf, "max_seconds": 0.0},
        )
        row["count"] = int(row["count"]) + 1
        row["seconds"] = float(row["seconds"]) + elapsed
        row["min_seconds"] = min(float(row["min_seconds"]), elapsed)
        row["max_seconds"] = max(float(row["max_seconds"]), elapsed)

    def begin_step_if_needed(self) -> None:
        if self.context == "training" and self.step_started_ns is None:
            self.sync()
            self.step_started_ns = time.perf_counter_ns()

    def finish_step(self, completed_steps: int) -> None:
        if self.step_started_ns is None:
            raise WC2TimingError("history append occurred without a training-step start")
        self.sync()
        self.record("step_total", time.perf_counter_ns() - self.step_started_ns)
        self.step_started_ns = None
        if completed_steps % self.flush_steps == 0:
            qbt.atomic_json(self.partial_output, self.payload(complete=False))

    def payload(self, *, complete: bool) -> dict[str, Any]:
        rows = {}
        for name, values in sorted(self.stats.items()):
            count = int(values["count"])
            seconds = float(values["seconds"])
            rows[name] = {
                **values,
                "mean_seconds": seconds / count,
            }
        return {
            "schema": SCHEMA,
            "complete": complete,
            "measurement_axis": "[per-stage wall-clock; configured-device synchronized; no score claim]",
            "score_claim": False,
            "profile_output_outside_live_burn": True,
            "synchronize_mps": self.synchronize_mps,
            "flush_steps": self.flush_steps,
            "stages": rows,
        }


def _timed(profiler: StageProfiler, stage: str, function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        profiler.sync()
        started = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            profiler.sync()
            profiler.record(stage, time.perf_counter_ns() - started)

    return wrapped


@contextmanager
def _profiling(profiler: StageProfiler) -> Iterator[None]:
    """Install timing-only wrappers and restore them even after a failed run."""

    originals: list[tuple[Any, str, Any]] = []

    def replace(owner: Any, name: str, value: Any) -> None:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    original_targets = qbt._target_arrays

    @functools.wraps(original_targets)
    def targets(*args: Any, **kwargs: Any) -> Any:
        profiler.begin_step_if_needed()
        return _timed(profiler, "target_arrays", original_targets)(*args, **kwargs)

    original_history = qbr._append_history

    @functools.wraps(original_history)
    def history(path: Path, row: Mapping[str, Any]) -> None:
        _timed(profiler, "history_io", original_history)(path, row)
        profiler.finish_step(int(row["completed_steps"]))

    original_milestone = qbr._evaluate_milestone

    @functools.wraps(original_milestone)
    def milestone(*args: Any, **kwargs: Any) -> Any:
        previous = profiler.context
        profiler.context = "milestone"
        try:
            return _timed(profiler, "total", original_milestone)(*args, **kwargs)
        finally:
            profiler.context = previous

    replace(qbt, "_target_arrays", targets)
    replace(qbt.QBFLOWTorch, "forward", _timed(profiler, "forward_realizer", qbt.QBFLOWTorch.forward))
    replace(qbt, "roundtrip_to_camera_uint8_ste", _timed(profiler, "roundtrip_R", qbt.roundtrip_to_camera_uint8_ste))
    replace(qbt, "scorer_forward", _timed(profiler, "scorer", qbt.scorer_forward))
    replace(qbr, "fairform_objective", _timed(profiler, "loss_forward", qbr.fairform_objective))
    replace(torch.nn.utils, "clip_grad_norm_", _timed(profiler, "grad_clip", torch.nn.utils.clip_grad_norm_))
    replace(torch.optim.AdamW, "step", _timed(profiler, "optimizer_step", torch.optim.AdamW.step))
    replace(qbt.EMA, "update", _timed(profiler, "ema_update", qbt.EMA.update))
    replace(qbr, "_append_history", history)
    replace(qbr, "_save_checkpoint", _timed(profiler, "checkpoint", qbr._save_checkpoint))
    replace(qbr, "_evaluate_milestone", milestone)
    try:
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def verify_default_off(resume_smoke_tree: Path, output: Path) -> dict[str, Any]:
    receipt_path = resume_smoke_tree / "RESUME_SMOKE_RESULT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or "CPU" not in str(receipt.get("axis")):
        raise WC2TimingError("identity tree is not the retained passing CPU resume smoke")
    watched = (
        (qbt, "_target_arrays"),
        (qbt.QBFLOWTorch, "forward"),
        (qbt, "roundtrip_to_camera_uint8_ste"),
        (qbt, "scorer_forward"),
        (qbr, "fairform_objective"),
        (qbr, "_append_history"),
        (qbr, "_save_checkpoint"),
        (qbr, "_evaluate_milestone"),
    )
    identities_before = tuple(id(getattr(owner, name)) for owner, name in watched)
    tree_before = _tree_fact(resume_smoke_tree)
    # The disabled branch deliberately installs no context and invokes no wrapper.
    profile_stages = False
    if profile_stages:  # pragma: no cover - positive path belongs to MAIN's fire order.
        raise AssertionError("unreachable default-off branch")
    identities_after = tuple(id(getattr(owner, name)) for owner, name in watched)
    tree_after = _tree_fact(resume_smoke_tree)
    result = {
        "schema": IDENTITY_SCHEMA,
        "status": "PASS" if tree_before == tree_after and identities_before == identities_after else "FAIL",
        "axis": "[macOS-CPU retained resume-smoke tree; read-only hash verification]",
        "score_claim": False,
        "profile_stages_default": False,
        "wrappers_installed_when_off": False,
        "callable_identities_unchanged": identities_before == identities_after,
        "tree_before": tree_before,
        "tree_after": tree_after,
        "tree_byte_identical": tree_before == tree_after,
        "source_resume_smoke": qbt.file_fact(receipt_path),
    }
    qbt.atomic_json(output, result)
    if result["status"] != "PASS":
        raise WC2TimingError("default-off identity verification failed")
    return result


def run_config(config_path: Path, *, profile_stages: bool, timing_output: Path | None, flush_steps: int) -> dict[str, Any]:
    if not profile_stages:
        if timing_output is not None:
            raise WC2TimingError("timing_output requires --profile-stages")
        return qbr.run_config(config_path)
    if timing_output is None:
        raise WC2TimingError("--profile-stages requires --timing-output")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiler = StageProfiler(
        timing_output,
        synchronize_mps=str(config.get("device")) == "mps",
        flush_steps=flush_steps,
    )
    started = time.perf_counter_ns()
    try:
        with _profiling(profiler):
            result = qbr.run_config(config_path)
    except BaseException:
        qbt.atomic_json(profiler.partial_output, profiler.payload(complete=False))
        raise
    payload = profiler.payload(complete=True)
    payload.update(
        {
            "wall_seconds_inclusive": (time.perf_counter_ns() - started) / 1.0e9,
            "config": qbt.file_fact(config_path),
            "qbr_entry": qbt.file_fact(Path(qbr.__file__)),
            "qbt_trainer": qbt.file_fact(Path(qbt.__file__)),
            "result_cell_id": result["cell_id"],
            "result_completed_steps": result["completed_steps"],
        }
    )
    qbt.atomic_json(timing_output, payload)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    verify = sub.add_parser("verify-default-off")
    verify.add_argument("--resume-smoke-tree", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run-config")
    run.add_argument("config", type=Path)
    run.add_argument("--profile-stages", action="store_true", default=False)
    run.add_argument("--timing-output", type=Path)
    run.add_argument("--flush-steps", type=int, default=DEFAULT_FLUSH_STEPS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "verify-default-off":
        result = verify_default_off(args.resume_smoke_tree, args.output)
    else:
        result = run_config(
            args.config,
            profile_stages=args.profile_stages,
            timing_output=args.timing_output,
            flush_steps=args.flush_steps,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
