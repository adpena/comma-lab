"""Read-only live ETA instrument for the JO1 r8/r9 critical path.

The instrument never creates, edits, opens for write, renames, or deletes an
artifact below the run directory.  It derives progress from atomic cursors and
completed receipt mtimes.  Unexecuted tail work stays labeled as an allowance;
it is never promoted to a measured current-run rate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

DEFAULT_RUN_DIR = Path(
    "experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final"
)
N_PAIRS = 600


class WallclockError(RuntimeError):
    """The run artifacts cannot support an honest ETA."""


@dataclass(frozen=True)
class RateBand:
    lower: float
    median: float
    upper: float
    sample_denominator: int
    source: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WallclockError(f"JSON root is not an object: {path}")
    return value


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise WallclockError("cannot derive a quantile from an empty sample")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _infer_compiled_config(run_dir: Path) -> Path:
    log = run_dir / "train.log"
    if not log.is_file():
        raise WallclockError(f"train log is absent: {log}")
    match = re.search(r"--compiled-config\s+(\S+)", log.read_text(encoding="utf-8"))
    if match is None:
        raise WallclockError("train log does not name --compiled-config")
    result = Path(match.group(1)).resolve()
    if not result.is_file():
        raise WallclockError(f"compiled config is absent: {result}")
    return result


def _pair_rate(stage_root: Path, fallback: RateBand) -> tuple[RateBand, int, int | None, int | None]:
    pointer_path = stage_root / "FRESH_SCHUR_POINTER.json"
    if not pointer_path.is_file():
        return fallback, 0, None, None
    pointer = _read_json(pointer_path)
    output = Path(str(pointer.get("output", ""))).resolve()
    if not output.is_dir():
        return fallback, 0, None, None
    receipts: list[tuple[int, int]] = []
    for result_path in output.glob("pairs/pair_*/RESULT.json"):
        try:
            pair = int(result_path.parent.name.removeprefix("pair_"))
        except ValueError:
            continue
        receipts.append((pair, result_path.stat().st_mtime_ns))
    receipts.sort(key=lambda value: value[1])
    completed = len(receipts)
    if completed < 2:
        return (
            fallback,
            completed,
            receipts[-1][0] if receipts else None,
            receipts[-1][1] if receipts else None,
        )
    # Completion-time intervals measure aggregate pair throughput for both the
    # current serial implementation and a possible whole-pair process layout.
    # A pair's candidate work and retention are already included.
    intervals = [
        (right_time - left_time) / 1e9
        for (_left_pair, left_time), (_right_pair, right_time) in pairwise(receipts)
        if right_time > left_time
    ]
    recent = intervals[-64:]
    if len(recent) < 2:
        return fallback, completed, receipts[-1][0], receipts[-1][1]
    median = statistics.median(recent)
    # Keep only gross pauses above 5x the median out of the computational-rate
    # sample. Ordinary hard-pair variation remains in the 10/90 band.
    computational = [value for value in recent if value <= 5.0 * median]
    if len(computational) < 2:
        computational = recent
    return (
        RateBand(
            lower=_quantile(computational, 0.10),
            median=statistics.median(computational),
            upper=_quantile(computational, 0.90),
            sample_denominator=len(computational),
            source="current-run completed fresh-Schur pair throughput receipts (recent<=64)",
        ),
        completed,
        receipts[-1][0],
        receipts[-1][1],
    )


def _materialization_seconds(run_dir: Path, current_stage_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    cursor_path = current_stage_root / "retained/MATERIALIZE_CURSOR.json"
    if cursor_path.is_file():
        cursor = _read_json(cursor_path)
        path = Path(str(cursor.get("candidate_master_path", "")))
        if int(cursor.get("next_pair", 0)) == N_PAIRS and path.is_file():
            stat = path.stat()
            birth = getattr(stat, "st_birthtime", stat.st_ctime)
            result["candidate_master_full_seconds"] = max(0.0, cursor_path.stat().st_mtime - birth)
    frame_cursor_path = run_dir / "retained/FX5_FRAME0_CURSOR.json"
    if frame_cursor_path.is_file():
        cursor = _read_json(frame_cursor_path)
        path = Path(str(cursor.get("retained_field", "")))
        if int(cursor.get("next_pair", 0)) == N_PAIRS and path.is_file():
            stat = path.stat()
            birth = getattr(stat, "st_birthtime", stat.st_ctime)
            result["frame0_full_seconds"] = max(0.0, frame_cursor_path.stat().st_mtime - birth)
    return result


def derive_eta(
    run_dir: Path,
    *,
    compiled_config: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise WallclockError(f"run directory is absent: {run_dir}")
    config_path = (compiled_config or _infer_compiled_config(run_dir)).resolve()
    config = _read_json(config_path)
    stages = list(config.get("stages", []))
    if not stages:
        raise WallclockError("compiled config has no stages")
    preflight_record = config.get("inputs", {}).get("memory_preflight_receipt")
    if not isinstance(preflight_record, dict):
        raise WallclockError("compiled config has no bound memory-preflight receipt")
    preflight_path = Path(str(preflight_record.get("path", ""))).resolve()
    preflight = _read_json(preflight_path)
    projection = preflight.get("wall_clock_projection", {})
    components = projection.get("components", {})
    step_seconds = float(projection["one_real_training_step_seconds"])
    carrier_low = float(components["carrier_resolve_lower"]) / (len(stages) * N_PAIRS)
    carrier_high = float(components["carrier_resolve_upper"]) / (len(stages) * N_PAIRS)
    fallback = RateBand(
        lower=carrier_low,
        median=(carrier_low + carrier_high) / 2.0,
        upper=carrier_high,
        sample_denominator=1,
        source="sealed preflight carrier allowance (fallback; not a current-run rate)",
    )
    resume_path = run_dir / "checkpoints/RESUME_LATEST.json"
    resume = _read_json(resume_path) if resume_path.is_file() else {}
    stage_roots = [run_dir / "stages" / f"{index + 1:02d}_{stage['stage_id']}" for index, stage in enumerate(stages)]
    admitted = [
        (root / "STAGE_RESULT.json").is_file() and _read_json(root / "STAGE_RESULT.json").get("status") == "ADMITTED"
        for root in stage_roots
    ]
    current_index = next((index for index, done in enumerate(admitted) if not done), len(stages))
    observed_at = now or datetime.now(UTC)
    if current_index == len(stages):
        return {
            "schema": "ddm_wc2_jo1_live_eta.v1",
            "status": "COMPLETE",
            "observed_at_utc": observed_at.isoformat().replace("+00:00", "Z"),
            "run_dir": str(run_dir),
            "remaining_seconds_band": [0.0, 0.0],
            "endpoint_utc_band": [
                observed_at.isoformat().replace("+00:00", "Z"),
                observed_at.isoformat().replace("+00:00", "Z"),
            ],
        }
    current_stage = stages[current_index]
    current_root = stage_roots[current_index]
    pair_rate, completed_pairs, last_pair, last_pair_mtime_ns = _pair_rate(current_root, fallback)
    current_stage_name = str(current_stage["stage_id"])
    resume_stage = str(resume.get("stage_id", ""))
    resume_step = int(resume.get("step", 0)) if resume_stage == current_stage_name else 0
    current_training_remaining = max(0, int(current_stage["fail_safe_steps"]) - resume_step)
    future_training_steps = sum(int(stage["fail_safe_steps"]) for stage in stages[current_index + 1 :])
    training_steps_remaining = current_training_remaining + future_training_steps
    current_pairs_remaining = (
        0
        if (current_root / "FRESH_SCHUR_POINTER.json").is_file()
        and _read_json(current_root / "FRESH_SCHUR_POINTER.json").get("status") == "COMPLETE"
        else max(0, N_PAIRS - completed_pairs)
    )
    future_pair_solves = N_PAIRS * len(stages[current_index + 1 :])
    pair_solves_remaining = current_pairs_remaining + future_pair_solves
    materialization = _materialization_seconds(run_dir, current_root)
    master_seconds = materialization.get("candidate_master_full_seconds", 0.0)
    frame0_seconds = materialization.get("frame0_full_seconds", 0.0)
    future_stage_count = len(stages) - current_index - 1
    # The current candidate master is already complete if its cursor says 600.
    current_master_complete = (current_root / "retained/MATERIALIZE_CURSOR.json").is_file() and int(
        _read_json(current_root / "retained/MATERIALIZE_CURSOR.json").get("next_pair", 0)
    ) == N_PAIRS
    materialization_remaining = (
        (0.0 if current_master_complete else master_seconds)
        + future_stage_count * master_seconds
        + (len(stages) - current_index) * frame0_seconds
    )
    field_seconds_per_stage = step_seconds * N_PAIRS
    tail_stage_count = len(stages) - current_index
    field_seconds_remaining = tail_stage_count * field_seconds_per_stage
    receiver_coder_upper_per_stage = float(components.get("receiver_and_real_coder_upper", 0.0)) / len(stages)
    receiver_coder_upper_remaining = tail_stage_count * receiver_coder_upper_per_stage
    common = training_steps_remaining * step_seconds + field_seconds_remaining + materialization_remaining
    lower_remaining = common + pair_solves_remaining * pair_rate.lower
    median_remaining = common + pair_solves_remaining * pair_rate.median + receiver_coder_upper_remaining / 2.0
    upper_remaining = common + pair_solves_remaining * pair_rate.upper + receiver_coder_upper_remaining
    return {
        "schema": "ddm_wc2_jo1_live_eta.v1",
        "status": "ACTIVE_OR_STALLED_ARTIFACT_STATE",
        "process_liveness_verified": False,
        "process_liveness_boundary": "process table is sandbox-blocked; freshness is artifact-derived",
        "observed_at_utc": observed_at.isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "compiled_config": str(config_path),
        "current_stage": current_stage_name,
        "current_stage_ordinal": current_index + 1,
        "stage_denominator": len(stages),
        "resume_step": resume_step,
        "completed_fresh_schur_pairs": completed_pairs,
        "last_completed_pair": last_pair,
        "last_pair_receipt_utc": (
            datetime.fromtimestamp(last_pair_mtime_ns / 1e9, UTC).isoformat().replace("+00:00", "Z")
            if last_pair_mtime_ns is not None
            else None
        ),
        "last_pair_receipt_age_seconds": (
            max(0.0, observed_at.timestamp() - last_pair_mtime_ns / 1e9) if last_pair_mtime_ns is not None else None
        ),
        "pair_rate_seconds": asdict(pair_rate),
        "one_training_step_seconds": {
            "value": step_seconds,
            "source": str(preflight_path),
            "evidence": "MEASURED [macOS-CPU real-config preflight; no score authority]",
        },
        "materialization_seconds": materialization,
        "work_remaining": {
            "training_steps": training_steps_remaining,
            "fresh_schur_pairs": pair_solves_remaining,
            "field_scoring_stage_equivalents": tail_stage_count,
            "receiver_coder_unexecuted_stage_equivalents": tail_stage_count,
        },
        "receiver_coder_tail": {
            "lower_seconds": 0.0,
            "upper_seconds": receiver_coder_upper_remaining,
            "source": "sealed r8 preflight allowance; current-run receiver/coder tail unexecuted",
            "evidence": "DERIVED BOUND, not a measured current-run rate",
        },
        "remaining_seconds": {
            "lower": lower_remaining,
            "median_hybrid": median_remaining,
            "upper": upper_remaining,
        },
        "remaining_seconds_band": [lower_remaining, upper_remaining],
        "endpoint_utc_band": [
            (observed_at + timedelta(seconds=lower_remaining)).isoformat().replace("+00:00", "Z"),
            (observed_at + timedelta(seconds=upper_remaining)).isoformat().replace("+00:00", "Z"),
        ],
        "all_inputs_read_only": True,
        "score_claim": False,
        "frontier_moved": False,
    }


def _duration(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{hours:.2f}h"


def summary(result: dict[str, Any]) -> str:
    if result["status"] == "COMPLETE":
        return "JO1 ETA COMPLETE: all configured stages have admitted receipts"
    lower, upper = result["remaining_seconds_band"]
    rate = result["pair_rate_seconds"]
    endpoint = result["endpoint_utc_band"]
    return (
        f"JO1 ETA {result['current_stage']} pair {result['completed_fresh_schur_pairs']}/{N_PAIRS}; "
        f"live pair median {rate['median']:.3f}s (n={rate['sample_denominator']}); "
        f"remaining {_duration(lower)}-{_duration(upper)}; endpoint {endpoint[0]}..{endpoint[1]}; "
        "receiver/coder tail includes a sealed unexecuted allowance"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    result.add_argument("--compiled-config", type=Path)
    result.add_argument("--json", action="store_true", dest="emit_json")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = derive_eta(args.run_dir, compiled_config=args.compiled_config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, WallclockError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True) if args.emit_json else summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
