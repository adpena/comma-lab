#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run unattended DQS1 local-first candidate tranches.

This is the higher-level control loop around ``run_dqs1_local_first_autopilot``:
execute one disk-bounded candidate, harvest/retain it, rebuild the canonical
observation and portfolio surfaces, rewrite the checked-in queue under a SHA
guard, then repeat.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.artifact_retention import DEFAULT_RETENTION_KINDS  # noqa: E402
from comma_lab.scheduler.dqs1_local_first_harvest import DEFAULT_QUEUE_PATH  # noqa: E402
from comma_lab.scheduler.dqs1_local_first_queue import DEFAULT_RESULTS_ROOT  # noqa: E402
from comma_lab.storage_tiers import (  # noqa: E402
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    bytes_from_gib,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.frontier_scan import build_frontier_scan_payload  # noqa: E402
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402

TRANCHE_SCHEMA = "dqs1_local_first_tranche_result.v1"
INACTIVE_PAIRSET_OBSERVATION_MODEL_MARKER = "pairset observation response model inactive"
DEFAULT_EXECUTOR_MODE = "dqs1_local_first"
DEFAULT_STORAGE_WORKLOAD_SUBDIR = "experiments/results/dqs1_local_first"
DEFAULT_PROACTIVE_CLEANUP_ROOTS = (
    "experiments/results",
    ".omx/tmp",
    "submissions/robust_current/eval_runs",
)
DEFAULT_BASELINE_ADVISORY = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "dqs1_top32_cpu_advisory_venv.json"
)
DEFAULT_DYNAMIC_OBSERVATIONS = (
    "experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/"
    "pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl"
)
DEFAULT_PORTFOLIO_ROOT = "experiments/results/cross_family_candidate_portfolio"
DEFAULT_CONTEST_CPU_FRONTIER = "0.19202828295713675"
DEFAULT_CONTEST_CUDA_FRONTIER = "0.2053300290"


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(REPO_ROOT))
    except ValueError:
        return str(value)


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _parse_bytes(value: str) -> int:
    raw = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(raw)


def _run(command: list[str], *, check: bool = True) -> CommandResult:
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    result = CommandResult(
        command=command,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_seconds=time.monotonic() - started,
    )
    if check and proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")
    return result


def _json_from_stdout(result: CommandResult, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: command did not emit JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}: expected JSON object")
    return payload


def _run_portfolio_with_exploratory_fallback(
    command: list[str],
) -> tuple[CommandResult, bool]:
    """Run strict portfolio planning, then fall back when calibration is not active."""

    result = _run(command, check=False)
    if result.returncode == 0:
        return result, False
    combined = f"{result.stdout}\n{result.stderr}"
    if (
        INACTIVE_PAIRSET_OBSERVATION_MODEL_MARKER not in combined
        or "--require-active-pairset-observation-model" not in command
    ):
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    exploratory = [
        item
        for item in command
        if item != "--require-active-pairset-observation-model"
    ]
    return _run(exploratory), True


def _free_disk_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free / float(1024**3)


def _latest_pairset_acquisition(results_root: Path) -> Path:
    root = results_root / "pairset_acquisition"
    candidates = sorted(
        root.glob("dqs1_pairset_acquisition_full_drop_two_*.json"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        raise RuntimeError(f"{root}: no full drop-two DQS1 acquisition JSON found")
    return candidates[-1]


def _portfolio_dir_name(stamp: str, row_count: int) -> str:
    return f"{stamp}_full_drop_two_local_harvest{row_count}"


def _choose_results_root(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.results_root is not None:
        explicit = _resolve(args.results_root)
        return explicit, {
            "mode": "explicit_results_root",
            "selected_workload_root": str(explicit),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    if not args.storage_waterfall:
        fallback = _resolve(DEFAULT_RESULTS_ROOT)
        return fallback, {
            "mode": "storage_waterfall_disabled",
            "selected_workload_root": str(fallback),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    tier_specs = parse_storage_tier_specs(
        args.storage_tier,
        repo_root=REPO_ROOT,
        reserve_free_gb=args.storage_reserve_gb,
        allow_local_disk=args.allow_local_storage_tier,
    )
    plan = plan_experiment_storage(
        tier_specs,
        workload_subdir=args.storage_workload_subdir,
        requested_bytes=args.storage_expected_bytes,
        min_free_bytes=bytes_from_gib(args.min_free_disk_gb),
        create=args.execute,
        probe_writable=True,
    )
    selected = require_selected_storage(plan)
    payload = plan.to_dict()
    payload["mode"] = "storage_waterfall"
    payload["executor_mode"] = args.executor_mode
    payload["source_families"] = list(args.source_family)
    return selected, payload


def _cold_store_root(args: argparse.Namespace, storage_payload: dict[str, Any]) -> str | None:
    if args.retention_cold_store_root:
        return str(args.retention_cold_store_root[0])
    if args.retention_action != "move":
        return None
    selected_root = storage_payload.get("selected_root")
    if not isinstance(selected_root, str) or not selected_root:
        return None
    return str(Path(selected_root) / args.cold_store_subdir)


def _cold_store_roots(args: argparse.Namespace) -> list[Path]:
    if args.retention_cold_store_root:
        return [_resolve(path) for path in args.retention_cold_store_root]
    tier_specs = parse_storage_tier_specs(
        args.storage_tier,
        repo_root=REPO_ROOT,
        reserve_free_gb=args.storage_reserve_gb,
        allow_local_disk=False,
    )
    roots: list[Path] = []
    for spec in tier_specs:
        resolved = spec.root.expanduser().resolve(strict=False)
        if not str(resolved).startswith("/Volumes/"):
            continue
        roots.append(resolved / args.cold_store_subdir)
    return roots


def _completed_results_roots(args: argparse.Namespace) -> list[str]:
    roots = [DEFAULT_RESULTS_ROOT]
    try:
        tier_specs = parse_storage_tier_specs(
            args.storage_tier,
            repo_root=REPO_ROOT,
            reserve_free_gb=args.storage_reserve_gb,
            allow_local_disk=False,
        )
    except StorageTierError:
        tier_specs = ()
    for spec in tier_specs:
        roots.append(str(spec.root.expanduser().resolve(strict=False) / args.storage_workload_subdir))
    out: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        out.append(root)
    return out


def _frontier_scores() -> tuple[dict[str, float], dict[str, Any]]:
    payload = build_frontier_scan_payload(REPO_ROOT)
    best = payload.get("best_per_axis")
    scores: dict[str, float] = {}
    if isinstance(best, dict):
        for axis in ("contest_cpu", "contest_cuda"):
            row = best.get(axis)
            if isinstance(row, dict):
                score = row.get("score")
                if isinstance(score, int | float) and not isinstance(score, bool):
                    scores[axis] = float(score)
    scores.setdefault("contest_cpu", float(DEFAULT_CONTEST_CPU_FRONTIER))
    scores.setdefault("contest_cuda", float(DEFAULT_CONTEST_CUDA_FRONTIER))
    return scores, payload


def _queue_build_common_args(
    args: argparse.Namespace,
    *,
    results_root: Path,
    dqs1_observation_jsonl: Sequence[str | Path] = (),
) -> list[str]:
    completed_args: list[str] = []
    for completed_root in _completed_results_roots(args):
        completed_args.extend(["--completed-results-root", completed_root])
    materializer_feedback_args: list[str] = []
    for feedback_path in args.materializer_feedback:
        materializer_feedback_args.extend(["--materializer-feedback", feedback_path])
    dqs1_observation_args: list[str] = []
    for observation_path in [*args.dqs1_observation_jsonl, *dqs1_observation_jsonl]:
        dqs1_observation_args.extend(
            ["--dqs1-observation-jsonl", _display_path(observation_path)]
        )
    base_args = [
        "--results-root",
        str(results_root),
        *completed_args,
        *materializer_feedback_args,
        *dqs1_observation_args,
        "--candidate-limit",
        str(args.queue_candidate_limit),
        "--local-cpu-concurrency",
        str(args.local_cpu_concurrency),
        "--local-io-concurrency",
        str(args.local_io_concurrency),
    ]
    if args.include_mlx_local_advisory_debug:
        base_args.append("--include-mlx-local-advisory-debug")
    if args.allow_large_mlx_cache:
        base_args.append("--allow-large-mlx-cache")
    base_args.extend(["--mlx-reference-cache-dir", args.mlx_reference_cache_dir])
    base_args.extend(["--mlx-device", args.mlx_device])
    base_args.extend(["--mlx-batch-pairs", str(args.mlx_batch_pairs)])
    base_args.extend(["--mlx-cache-batch-pairs", str(args.mlx_cache_batch_pairs)])
    if args.skip_mlx_retention_plan:
        base_args.append("--skip-mlx-retention-plan")
    if not args.storage_waterfall:
        return base_args
    preflight_args = [
        "--include-scheduler-preflight",
        "--scheduler-storage-workload-subdir",
        args.storage_workload_subdir,
        "--scheduler-storage-expected-workload-root",
        str(results_root),
        "--scheduler-storage-reserve-free-gb",
        str(args.storage_reserve_gb),
        "--scheduler-storage-expected-bytes",
        str(args.storage_expected_bytes),
        "--scheduler-proactive-cleanup-action",
        args.proactive_cleanup_action,
        "--scheduler-proactive-cleanup-min-bytes",
        str(args.proactive_cleanup_min_bytes),
        "--scheduler-proactive-cleanup-cold-store-reserve-gb",
        str(args.storage_reserve_gb),
    ]
    for tier in args.storage_tier:
        preflight_args.extend(["--scheduler-storage-tier", tier])
    for root in args.proactive_cleanup_root:
        preflight_args.extend(["--scheduler-proactive-cleanup-root", root])
    for root in _cold_store_roots(args):
        preflight_args.extend(["--scheduler-proactive-cleanup-cold-store-root", str(root)])
    preflight_args.append("--scheduler-proactive-cleanup-execute")
    return [*base_args, *preflight_args]


def _prepare_queue_for_results_root(
    *,
    args: argparse.Namespace,
    queue_path: Path,
    results_root: Path,
    storage_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not args.storage_working_queue:
        return queue_path, {
            "mode": "configured_queue",
            "queue_path": str(queue_path),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    try:
        results_root.relative_to(REPO_ROOT)
        return queue_path, {
            "mode": "repo_relative_queue",
            "queue_path": str(queue_path),
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    except ValueError:
        pass
    selected_root_raw = storage_payload.get("selected_root")
    selected_root = Path(selected_root_raw) if isinstance(selected_root_raw, str) else results_root.parent
    queue_root = (
        _resolve(args.storage_queue_root)
        if args.storage_queue_root is not None
        else selected_root / "queues"
    )
    queue_root.mkdir(parents=True, exist_ok=True)
    working_queue = queue_root / queue_path.name
    working_queue_id = f"{queue_path.stem}_storage"
    command = [
        ".venv/bin/python",
        "tools/build_dqs1_local_first_queue.py",
        "--action-summary",
        args.initial_action_summary,
        "--output",
        str(working_queue),
        "--queue-id",
        working_queue_id,
        "--write",
        *_queue_build_common_args(args, results_root=results_root),
    ]
    expected_sha = None
    if working_queue.exists():
        expected_sha = _sha256_file(working_queue)
        command.extend(["--overwrite-output", "--expected-output-sha256", expected_sha])
    result = _run(command)
    payload = _json_from_stdout(result, label="storage working queue")
    return working_queue, {
        "mode": "storage_working_queue",
        "source_queue_path": str(queue_path),
        "queue_path": str(working_queue),
        "queue_root": str(queue_root),
        "queue_id": working_queue_id,
        "refresh_command": command,
        "refresh_elapsed_seconds": result.elapsed_seconds,
        "refresh_result": payload,
        "expected_existing_sha256": expected_sha,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _run_proactive_cleanup(
    *,
    args: argparse.Namespace,
    stamp: str,
    round_index: int,
) -> dict[str, Any] | None:
    if not args.proactive_cleanup:
        return None
    roots = [_resolve(path) for path in args.proactive_cleanup_root]
    existing_roots = [path for path in roots if path.exists()]
    if not existing_roots:
        return {
            "schema": "dqs1_local_first_proactive_cleanup.v1",
            "round_index": round_index,
            "terminal": "no_existing_roots",
            "roots": [str(path) for path in roots],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    output_path = (
        REPO_ROOT
        / ".omx/research"
        / f"dqs1_proactive_artifact_retention_{stamp}_round{round_index:03d}.json"
    )
    command = [
        ".venv/bin/python",
        "tools/compact_experiment_artifacts.py",
        *[_display_path(path) for path in existing_roots],
        "--min-bytes",
        str(args.proactive_cleanup_min_bytes),
        "--json-output",
        _display_path(output_path),
    ]
    if args.proactive_cleanup_include_mlx_cache:
        for kind in sorted(set(DEFAULT_RETENTION_KINDS) | {"mlx_scorer_input_cache"}):
            command.extend(["--include-kind", kind])
    if args.execute:
        command.extend(["--execute", "--action", args.proactive_cleanup_action])
    if args.proactive_cleanup_action == "move":
        cold_roots = _cold_store_roots(args)
        if args.execute:
            _ensure_directories(cold_roots)
        command.extend(["--cold-store-reserve-gb", str(args.storage_reserve_gb)])
        for root in cold_roots:
            command.extend(["--cold-store-root", str(root)])
    result = _run(command)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{output_path}: proactive cleanup did not write a JSON object")
    expected_output_sha = _sha256_file(output_path)
    payload["schema"] = "dqs1_local_first_proactive_cleanup.v1"
    payload["round_index"] = round_index
    payload["command"] = command
    payload["elapsed_seconds"] = result.elapsed_seconds
    payload["artifact_path"] = str(output_path)
    payload["expected_existing_sha256_before_enrichment"] = expected_output_sha
    payload["score_claim"] = False
    payload["promotion_eligible"] = False
    payload["rank_or_kill_eligible"] = False
    payload["ready_for_exact_eval_dispatch"] = False
    try:
        write_json_artifact(
            output_path,
            payload,
            allow_overwrite=True,
            expected_existing_sha256=expected_output_sha,
        )
    except ArtifactWriteError as exc:
        raise RuntimeError(f"{output_path}: failed to persist enriched cleanup payload: {exc}") from exc
    return payload


def _round_harvest(autopilot_payload: dict[str, Any]) -> dict[str, Any] | None:
    harvests = _round_harvests(autopilot_payload)
    return harvests[0] if harvests else None


def _round_harvests(autopilot_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rounds = autopilot_payload.get("rounds")
    if not isinstance(rounds, list):
        return []
    for round_payload in reversed(rounds):
        if not isinstance(round_payload, dict):
            continue
        harvests = round_payload.get("harvests")
        if isinstance(harvests, list):
            return [row for row in harvests if isinstance(row, dict)]
        harvest = round_payload.get("harvest")
        if isinstance(harvest, dict):
            return [harvest]
    return []


def _round_retention(autopilot_payload: dict[str, Any]) -> dict[str, Any] | None:
    rounds = autopilot_payload.get("rounds")
    if not isinstance(rounds, list):
        return None
    for round_payload in reversed(rounds):
        if not isinstance(round_payload, dict):
            continue
        retention = round_payload.get("post_harvest_retention")
        if isinstance(retention, dict):
            return retention
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor-mode", default=DEFAULT_EXECUTOR_MODE)
    parser.add_argument(
        "--source-family",
        action="append",
        default=["decoder_q_pairset", "nerv_family_ready", "non_nerv_ready"],
        help="Family tags for downstream telemetry; execution is DQS1-local-first today.",
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--initial-action-summary", default="latest")
    parser.add_argument(
        "--storage-working-queue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="when results are external, refresh a queue under the selected storage tier.",
    )
    parser.add_argument("--storage-queue-root", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--max-steps-per-worker", type=int, default=8)
    parser.add_argument("--min-free-disk-gb", type=float, default=40.0)
    parser.add_argument("--retention-action", choices=("delete", "move"), default="move")
    parser.add_argument(
        "--retention-cold-store-root",
        action="append",
        default=[],
        help="cold-store root for move retention; repeat for tiered water-bucket moves",
    )
    parser.add_argument(
        "--cold-store-subdir",
        default="cold_store",
        help="cold-store directory under the selected storage tier when --retention-action=move",
    )
    parser.add_argument("--retention-min-bytes", default="1")
    parser.add_argument(
        "--results-root",
        default=None,
        help="explicit result root; when omitted, use the storage waterfall.",
    )
    parser.add_argument(
        "--storage-waterfall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="auto-place bulky work on VertigoDataTier, then APDataStore; local disk is opt-in.",
    )
    parser.add_argument(
        "--storage-tier",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override storage tiers in priority order; defaults to VertigoDataTier then APDataStore.",
    )
    parser.add_argument(
        "--storage-workload-subdir",
        default=DEFAULT_STORAGE_WORKLOAD_SUBDIR,
        help=f"relative workload path under the selected tier (default: {DEFAULT_STORAGE_WORKLOAD_SUBDIR})",
    )
    parser.add_argument(
        "--storage-reserve-gb",
        type=float,
        default=DEFAULT_RESERVE_FREE_GB,
        help="free-space reserve each tier must keep after the workload allocation",
    )
    parser.add_argument(
        "--storage-expected-bytes",
        type=int,
        default=0,
        help="optional estimated bytes needed before considering a tier eligible",
    )
    parser.add_argument(
        "--allow-local-storage-tier",
        action="store_true",
        help="allow the repo filesystem as a final spill tier; default is fail closed before local bloat.",
    )
    parser.add_argument(
        "--proactive-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="plan, and under --execute run, certified cleanup before launching workers.",
    )
    parser.add_argument(
        "--proactive-cleanup-frequency",
        choices=("once", "each-round"),
        default="once",
    )
    parser.add_argument(
        "--proactive-cleanup-root",
        action="append",
        default=list(DEFAULT_PROACTIVE_CLEANUP_ROOTS),
        help="root to scan for certified rebuildable bulk before worker launch.",
    )
    parser.add_argument(
        "--proactive-cleanup-action",
        choices=("move", "delete"),
        default="move",
        help="action used for proactive cleanup under --execute.",
    )
    parser.add_argument(
        "--proactive-cleanup-min-bytes",
        type=_parse_bytes,
        default=1 << 30,
    )
    parser.add_argument(
        "--proactive-cleanup-include-mlx-cache",
        action="store_true",
        help="reserved for explicit MLX-cache cleanup; default cleanup keeps caches blocked.",
    )
    parser.add_argument("--pairset-acquisition", default="latest")
    parser.add_argument(
        "--pairset-acquisition-root",
        default=DEFAULT_RESULTS_ROOT,
        help="root to search when --pairset-acquisition=latest; separate from output storage.",
    )
    parser.add_argument(
        "--mlx-selection",
        action="append",
        default=[],
        help="Additional MLX selection JSON from a promising run; passed to portfolio planning.",
    )
    parser.add_argument(
        "--hfv2-manifest",
        action="append",
        default=[],
        help="Additional HFV/HNeRV-family candidate manifest; passed to portfolio planning.",
    )
    parser.add_argument(
        "--candidate-json",
        action="append",
        default=[],
        help="Manual or post-training candidate JSON; passed to portfolio planning.",
    )
    parser.add_argument(
        "--family-beliefs",
        default=None,
        help="Optional family belief JSON for post-training portfolio priors.",
    )
    parser.add_argument("--baseline-advisory", default=DEFAULT_BASELINE_ADVISORY)
    parser.add_argument("--baseline-archive-size-bytes", type=int, default=178560)
    parser.add_argument("--baseline-candidate-id", default="dqs1_top32_gap_uleb")
    parser.add_argument(
        "--dynamic-observation-jsonl",
        action="append",
        default=[DEFAULT_DYNAMIC_OBSERVATIONS],
    )
    parser.add_argument(
        "--materializer-feedback",
        action="append",
        default=[],
        help=(
            "family-agnostic materializer feedback sweep/observation JSON to forward "
            "into generated DQS1 local-first queues"
        ),
    )
    parser.add_argument(
        "--dqs1-observation-jsonl",
        "--dqs1-observations",
        action="append",
        default=[],
        dest="dqs1_observation_jsonl",
        help=(
            "existing DQS1 local-first harvest observation JSONL to forward into "
            "generated queue feedback bridges; current-round harvest observations "
            "are added automatically after each harvest."
        ),
    )
    parser.add_argument("--portfolio-root", default=None)
    parser.add_argument("--incumbent-score", default=None)
    parser.add_argument(
        "--incumbent-score-by-axis",
        action="append",
        default=[],
    )
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument("--top-actions", type=int, default=64)
    parser.add_argument("--queue-candidate-limit", type=int, default=2)
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument("--local-io-concurrency", type=int, default=2)
    parser.add_argument(
        "--include-mlx-local-advisory-debug",
        action="store_true",
        help="add local MLX scorer-cache/response steps to generated DQS1 queues",
    )
    parser.add_argument(
        "--allow-large-mlx-cache",
        action="store_true",
        help="required with --include-mlx-local-advisory-debug for full local caches",
    )
    parser.add_argument(
        "--mlx-reference-cache-dir",
        default=(
            "experiments/results/"
            "mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"
        ),
    )
    parser.add_argument("--mlx-device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--mlx-batch-pairs", type=int, default=1)
    parser.add_argument("--mlx-cache-batch-pairs", type=int, default=8)
    parser.add_argument("--skip-mlx-retention-plan", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--stop-on-eureka",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop when a harvested candidate creates an exact auth request",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.queue_candidate_limit < 1:
        raise SystemExit("--queue-candidate-limit must be >= 1")
    if args.local_cpu_concurrency < 1:
        raise SystemExit("--local-cpu-concurrency must be >= 1")
    if args.local_io_concurrency < 1:
        raise SystemExit("--local-io-concurrency must be >= 1")
    if args.mlx_batch_pairs < 1:
        raise SystemExit("--mlx-batch-pairs must be >= 1")
    if args.mlx_cache_batch_pairs < 1:
        raise SystemExit("--mlx-cache-batch-pairs must be >= 1")
    queue_path = _resolve(args.queue)
    frontier_scores, frontier_payload = _frontier_scores()
    incumbent_score = args.incumbent_score or str(frontier_scores["contest_cuda"])
    incumbent_score_by_axis = list(args.incumbent_score_by_axis) or [
        f"contest_cpu={frontier_scores['contest_cpu']}",
        f"contest_cuda={frontier_scores['contest_cuda']}",
    ]
    try:
        results_root, storage_payload = _choose_results_root(args)
    except StorageTierError as exc:
        raise SystemExit(str(exc)) from exc
    queue_path, queue_storage_payload = _prepare_queue_for_results_root(
        args=args,
        queue_path=queue_path,
        results_root=results_root,
        storage_payload=storage_payload,
    )
    cold_store_roots = _cold_store_roots(args) if args.retention_action == "move" else []
    if args.execute and cold_store_roots:
        _ensure_directories(cold_store_roots)
    cold_store_root = str(cold_store_roots[0]) if cold_store_roots else _cold_store_root(args, storage_payload)
    pairset_acquisition_root = _resolve(args.pairset_acquisition_root)
    pairset_acquisition = (
        _latest_pairset_acquisition(pairset_acquisition_root)
        if args.pairset_acquisition == "latest"
        else _resolve(args.pairset_acquisition)
    )
    portfolio_root = (
        results_root / "cross_family_candidate_portfolio"
        if args.portfolio_root is None
        else _resolve(args.portfolio_root)
    )
    tranche_stamp = _utc_stamp()
    json_out = (
        _resolve(args.json_out)
        if args.json_out is not None
        else REPO_ROOT / ".omx/research" / f"dqs1_local_first_tranche_{tranche_stamp}.json"
    )
    rounds: list[dict[str, Any]] = []
    stop_reason = "round_limit_reached"
    for index in range(args.rounds):
        proactive_cleanup = None
        if args.proactive_cleanup_frequency == "each-round" or index == 0:
            proactive_cleanup = _run_proactive_cleanup(
                args=args,
                stamp=tranche_stamp,
                round_index=index,
            )
        if _free_disk_gb(results_root) < args.min_free_disk_gb:
            if proactive_cleanup is not None:
                rounds.append(
                    {
                        "round_index": index,
                        "terminal": "insufficient_free_disk_after_proactive_cleanup",
                        "proactive_cleanup": proactive_cleanup,
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                )
            stop_reason = "insufficient_free_disk"
            break
        worker_step_budget = args.max_steps_per_worker * args.queue_candidate_limit
        autopilot_cmd = [
            ".venv/bin/python",
            "tools/run_dqs1_local_first_autopilot.py",
            "--queue",
            _display_path(queue_path),
            "--max-candidates",
            str(args.queue_candidate_limit),
            "--max-total-steps",
            str(worker_step_budget),
            "--max-steps-per-worker",
            str(worker_step_budget),
            "--max-worker-experiments",
            str(args.queue_candidate_limit),
            "--min-free-disk-gb",
            str(args.min_free_disk_gb),
            "--retention-action",
            args.retention_action,
            "--retention-min-bytes",
            str(args.retention_min_bytes),
        ]
        if args.execute:
            autopilot_cmd.append("--execute")
        for root in cold_store_roots:
            autopilot_cmd.extend(["--retention-cold-store-root", str(root)])
        autopilot_cmd.extend(["--results-root", str(results_root)])
        autopilot_result = _run(autopilot_cmd)
        autopilot_payload = _json_from_stdout(autopilot_result, label="autopilot")
        harvests = _round_harvests(autopilot_payload)
        harvest = harvests[0] if harvests else None
        retention = _round_retention(autopilot_payload)
        round_record: dict[str, Any] = {
            "round_index": index,
            "autopilot": {
                "command": autopilot_cmd,
                "elapsed_seconds": autopilot_result.elapsed_seconds,
                "stop_reason": autopilot_payload.get("stop_reason"),
                "total_steps_started": autopilot_payload.get("total_steps_started"),
                "candidates_harvested": autopilot_payload.get("candidates_harvested"),
            },
            "harvest": harvest,
            "harvests": harvests,
            "harvest_count": len(harvests),
            "proactive_cleanup": proactive_cleanup,
            "post_harvest_retention": retention,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        if not harvests:
            round_record["terminal"] = "no_harvest"
            rounds.append(round_record)
            stop_reason = str(autopilot_payload.get("stop_reason") or "no_harvest")
            break

        refresh_stamp = _utc_stamp()
        observation_jsonl = (
            REPO_ROOT / ".omx/research" / f"dqs1_local_first_harvest_observations_{refresh_stamp}.jsonl"
        )
        observation_summary = observation_jsonl.with_suffix(".summary.json")
        observation_md = observation_jsonl.with_suffix(".md")
        obs_result = _run(
            [
                ".venv/bin/python",
                "tools/build_dqs1_local_first_harvest_observations.py",
                "--harvest",
                ".omx/research/dqs1_local_first_harvest_*.json",
                "--pairset-acquisition",
                _display_path(pairset_acquisition),
                "--baseline-advisory",
                args.baseline_advisory,
                "--baseline-archive-size-bytes",
                str(args.baseline_archive_size_bytes),
                "--baseline-candidate-id",
                args.baseline_candidate_id,
                "--jsonl-out",
                _display_path(observation_jsonl),
                "--summary-json-out",
                _display_path(observation_summary),
                "--md-out",
                _display_path(observation_md),
            ]
        )
        observation_payload = _json_from_stdout(obs_result, label="harvest observations")
        row_count = int(observation_payload.get("row_count") or 0)
        portfolio_dir = portfolio_root / _portfolio_dir_name(refresh_stamp, row_count)
        action_summary = portfolio_dir / "action_summary.json"
        portfolio_cmd = [
            ".venv/bin/python",
            "tools/plan_cross_family_candidate_portfolio.py",
            "--incumbent-score",
            str(incumbent_score),
            "--pairset-acquisition",
            _display_path(pairset_acquisition),
        ]
        for mlx_selection in args.mlx_selection:
            portfolio_cmd.extend(["--mlx-selection", mlx_selection])
        for hfv2_manifest in args.hfv2_manifest:
            portfolio_cmd.extend(["--hfv2-manifest", hfv2_manifest])
        for candidate_json in args.candidate_json:
            portfolio_cmd.extend(["--candidate-json", candidate_json])
        if args.family_beliefs:
            portfolio_cmd.extend(["--family-beliefs", args.family_beliefs])
        for axis_score in incumbent_score_by_axis:
            portfolio_cmd.extend(["--incumbent-score-by-axis", str(axis_score)])
        for observation in args.dynamic_observation_jsonl:
            portfolio_cmd.extend(["--observation-jsonl", observation])
        portfolio_cmd.extend(
            [
                "--observation-jsonl",
                _display_path(observation_jsonl),
                "--require-active-pairset-observation-model",
                "--top-k",
                str(args.top_k),
                "--top-actions",
                str(args.top_actions),
                "--json-out",
                _display_path(portfolio_dir / "portfolio.json"),
                "--md-out",
                _display_path(portfolio_dir / "portfolio.md"),
                "--summary-json-out",
                _display_path(action_summary),
            ]
        )
        portfolio_result, exploratory_portfolio_fallback = (
            _run_portfolio_with_exploratory_fallback(portfolio_cmd)
        )
        _json_from_stdout(portfolio_result, label="portfolio")

        queue_sha = _sha256_file(queue_path)
        queue_result = _run(
            [
                ".venv/bin/python",
                "tools/build_dqs1_local_first_queue.py",
                "--action-summary",
                _display_path(action_summary),
                "--output",
                _display_path(queue_path),
                "--write",
                "--overwrite-output",
                "--expected-output-sha256",
                queue_sha,
                "--queue-id",
                str(queue_storage_payload.get("queue_id") or "dqs1_pairset_local_first"),
                *_queue_build_common_args(
                    args,
                    results_root=results_root,
                    dqs1_observation_jsonl=(observation_jsonl,),
                ),
            ]
        )
        queue_payload = _json_from_stdout(queue_result, label="queue rebuild")
        validate_result = _run(
            [
                ".venv/bin/python",
                "tools/experiment_queue.py",
                "--queue",
                _display_path(queue_path),
                "validate",
            ]
        )
        validate_payload = _json_from_stdout(validate_result, label="queue validate")
        round_record["observation"] = {
            "jsonl": str(observation_jsonl),
            "summary_json": str(observation_summary),
            "markdown": str(observation_md),
            "row_count": row_count,
            "best_local_advisory": observation_payload.get("best_local_advisory"),
        }
        round_record["portfolio"] = {
            "action_summary": str(action_summary),
            "portfolio_json": str(portfolio_dir / "portfolio.json"),
            "portfolio_md": str(portfolio_dir / "portfolio.md"),
            "exploratory_pairset_observation_model_fallback": (
                exploratory_portfolio_fallback
            ),
        }
        round_record["queue_rebuild"] = queue_payload
        round_record["queue_validate"] = validate_payload
        rounds.append(round_record)
        if args.stop_on_eureka and any(
            row.get("exact_auth_request_path") for row in harvests
        ):
            stop_reason = "exact_auth_anchor_request_created"
            break
    payload = {
        "schema": TRANCHE_SCHEMA,
        "created_at_utc": tranche_stamp,
        "executor_mode": args.executor_mode,
        "source_families": list(args.source_family),
        "execute": bool(args.execute),
        "queue_path": str(queue_path),
        "queue_storage": queue_storage_payload,
        "results_root": str(results_root),
        "storage": storage_payload,
        "pairset_acquisition_root": str(pairset_acquisition_root),
        "pairset_acquisition": str(pairset_acquisition),
        "portfolio_root": str(portfolio_root),
        "retention_cold_store_root": cold_store_root,
        "frontier_scores": frontier_scores,
        "frontier_scan_schema": frontier_payload.get("schema"),
        "incumbent_score": incumbent_score,
        "incumbent_score_by_axis": incumbent_score_by_axis,
        "round_limit": args.rounds,
        "rounds_completed": len(rounds),
        "stop_reason": stop_reason,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rounds": rounds,
    }
    try:
        write_json_artifact(json_out, payload)
    except ArtifactWriteError as exc:
        raise RuntimeError(str(exc)) from exc
    payload["json_out"] = str(json_out)
    _json_print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
