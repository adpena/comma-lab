#!/usr/bin/env python3
"""Profile local preflight and test-adjacent latency without dispatching jobs.

The profiler is intentionally observational: it does not change preflight
semantics, does not write review-tracker state, and does not treat a failed
surface as a profiler failure unless ``--fail-on-surface-failure`` is passed.
Use the JSON output to pick evidence-backed Python/Rust acceleration targets.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as _dt
import functools
import importlib
import importlib.util
import inspect
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)

from tac.repo_io import json_text, repo_relative  # noqa: E402

PROFILE_SCHEMA = "pact.preflight_latency_profile.v1"
SURFACES = (
    "preflight-all-codebase",
    "dispatch-hazards",
    "review-tracker-scan",
    "all-lanes",
)


class SurfaceTimeoutError(TimeoutError):
    """A profiled surface exceeded the requested local timing budget."""


@dataclass(frozen=True)
class StepTiming:
    """One timed unit inside a profiled surface."""

    surface: str
    name: str
    elapsed_s: float
    status: str
    detail: str = ""
    path: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        row: dict[str, object] = {
            "surface": self.surface,
            "name": self.name,
            "elapsed_s": round(float(self.elapsed_s), 6),
            "status": self.status,
        }
        if self.detail:
            row["detail"] = self.detail
        if self.path:
            row["path"] = self.path
        if self.metadata:
            row["metadata"] = self.metadata
        return row


@dataclass(frozen=True)
class SurfaceProfile:
    """Profile result for one developer-facing surface."""

    name: str
    status: str
    elapsed_s: float
    steps: list[StepTiming]
    metadata: dict[str, object] = field(default_factory=dict)
    error_type: str = ""
    error: str = ""

    def to_json(self, *, max_step_records: int | None = None) -> dict[str, object]:
        steps = sorted(self.steps, key=_step_sort_key)
        if max_step_records is not None and max_step_records >= 0:
            steps = steps[:max_step_records]
        row: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "elapsed_s": round(float(self.elapsed_s), 6),
            "step_count": len(self.steps),
            "steps": [step.to_json() for step in steps],
        }
        if self.metadata:
            row["metadata"] = self.metadata
        if self.error_type:
            row["error_type"] = self.error_type
        if self.error:
            row["error"] = self.error
        return row


def _step_sort_key(step: StepTiming) -> tuple[float, str, str, str]:
    return (-float(step.elapsed_s), step.surface, step.name, step.path)


def _load_tool_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _timeout_after(seconds: float | None, *, label: str):
    if seconds is None or seconds <= 0:
        yield
        return

    def handler(_signum: int, _frame: object) -> None:
        raise SurfaceTimeoutError(f"{label} exceeded timeout_s={seconds:g}")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _called_preflight_check_names(func: Callable[..., object]) -> list[str]:
    """Return check/preflight function names called by ``preflight_all``.

    This intentionally discovers the current call surface from source so adding
    a new strict check automatically becomes visible in timing profiles.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    names: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if not isinstance(callee, ast.Name):
            continue
        name = callee.id
        if name == "preflight_all":
            continue
        if not (name.startswith("check_") or name.startswith("preflight_")):
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _profile_preflight_all_codebase(
    *,
    use_fs_cache: bool,
    timeout_s: float | None,
) -> SurfaceProfile:
    surface = "preflight-all-codebase"
    started = time.perf_counter()
    steps: list[StepTiming] = []
    originals: dict[str, Callable[..., object]] = {}
    status = "passed"
    error_type = ""
    error = ""

    preflight = importlib.import_module("tac.preflight")
    check_names = _called_preflight_check_names(preflight.preflight_all)

    def wrap(name: str, func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            step_started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                steps.append(
                    StepTiming(
                        surface=surface,
                        name=name,
                        elapsed_s=time.perf_counter() - step_started,
                        status="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
            steps.append(
                StepTiming(
                    surface=surface,
                    name=name,
                    elapsed_s=time.perf_counter() - step_started,
                    status="passed",
                )
            )
            return result

        return wrapper

    for name in check_names:
        func = getattr(preflight, name, None)
        if callable(func):
            originals[name] = func
            setattr(preflight, name, wrap(name, func))

    try:
        with _timeout_after(timeout_s, label=surface):
            preflight.preflight_all(
                check_codebase=True,
                verbose=False,
                use_fs_cache=use_fs_cache,
            )
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        error = str(exc)
    finally:
        for name, func in originals.items():
            setattr(preflight, name, func)

    return SurfaceProfile(
        name=surface,
        status=status,
        elapsed_s=time.perf_counter() - started,
        steps=steps,
        metadata={
            "wrapped_check_count": len(originals),
            "use_fs_cache": use_fs_cache,
            "timeout_s": timeout_s if timeout_s is not None else 0,
        },
        error_type=error_type,
        error=error,
    )


def _profile_dispatch_hazards() -> SurfaceProfile:
    surface = "dispatch-hazards"
    started = time.perf_counter()
    steps: list[StepTiming] = []
    hazards: list[object] = []
    module = _load_tool_module(
        REPO / "tools" / "check_dispatch_cli_shell_hazards.py",
        "profile_dispatch_cli_shell_hazards",
    )

    discovery_started = time.perf_counter()
    files = module.iter_scan_files(REPO, module.DEFAULT_SCAN_PATHS)
    steps.append(
        StepTiming(
            surface=surface,
            name="discover scan files",
            elapsed_s=time.perf_counter() - discovery_started,
            status="passed",
            metadata={"file_count": len(files)},
        )
    )

    for file_path in files:
        step_started = time.perf_counter()
        rel = repo_relative(file_path, REPO)
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            file_hazards = module.scan_text(file_path, text, root=REPO)
        except OSError as exc:
            steps.append(
                StepTiming(
                    surface=surface,
                    name=rel,
                    path=rel,
                    elapsed_s=time.perf_counter() - step_started,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        hazards.extend(file_hazards)
        steps.append(
            StepTiming(
                surface=surface,
                name=rel,
                path=rel,
                elapsed_s=time.perf_counter() - step_started,
                status="failed" if file_hazards else "passed",
                detail=f"{len(file_hazards)} hazard(s)" if file_hazards else "",
            )
        )

    return SurfaceProfile(
        name=surface,
        status="failed" if hazards else "passed",
        elapsed_s=time.perf_counter() - started,
        steps=steps,
        metadata={
            "file_count": len(files),
            "hazard_count": len(hazards),
            "scan_paths": list(module.DEFAULT_SCAN_PATHS),
        },
    )


def _profile_review_tracker_scan() -> SurfaceProfile:
    surface = "review-tracker-scan"
    started = time.perf_counter()
    steps: list[StepTiming] = []
    module = _load_tool_module(REPO / "tools" / "review_tracker.py", "profile_review_tracker")

    discovery_started = time.perf_counter()
    files = module._tracked_reviewable_python_files()
    steps.append(
        StepTiming(
            surface=surface,
            name="discover reviewable Python files",
            elapsed_s=time.perf_counter() - discovery_started,
            status="passed",
            metadata={"file_count": len(files)},
        )
    )

    compute_complexity = os.environ.get("REVIEW_TRACKER_COMPLEXITY") == "1"
    entity_count = 0
    failed_files = 0
    for file_path in files:
        step_started = time.perf_counter()
        rel = repo_relative(file_path, REPO)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                entities = module.extract_entities(file_path, compute_complexity=compute_complexity)
            entity_count += len(entities)
            status = "passed"
            detail = ""
        except Exception as exc:  # pragma: no cover - defensive wrapper.
            failed_files += 1
            status = "failed"
            detail = f"{type(exc).__name__}: {exc}"
        steps.append(
            StepTiming(
                surface=surface,
                name=rel,
                path=rel,
                elapsed_s=time.perf_counter() - step_started,
                status=status,
                detail=detail,
            )
        )

    return SurfaceProfile(
        name=surface,
        status="failed" if failed_files else "passed",
        elapsed_s=time.perf_counter() - started,
        steps=steps,
        metadata={
            "file_count": len(files),
            "entity_count": entity_count,
            "failed_file_count": failed_files,
            "compute_complexity": compute_complexity,
        },
    )


def _profile_all_lanes(*, jobs: int | None) -> SurfaceProfile:
    surface = "all-lanes"
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pact_all_lanes_timing_") as tmp:
        timing_path = Path(tmp) / "timings.json"
        cmd = [
            sys.executable,
            str(REPO / "tools" / "all_lanes_preflight.py"),
            "--timings-json",
            str(timing_path),
        ]
        if jobs is not None:
            cmd.extend(["--jobs", str(jobs)])
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        elapsed_s = time.perf_counter() - started
        steps: list[StepTiming] = []
        metadata: dict[str, object] = {
            "returncode": proc.returncode,
            "jobs": jobs if jobs is not None else "default",
        }
        try:
            payload = json_loads_path(timing_path)
        except (OSError, ValueError) as exc:
            return SurfaceProfile(
                name=surface,
                status="failed",
                elapsed_s=elapsed_s,
                steps=[],
                metadata={
                    **metadata,
                    "stdout_tail": _tail(proc.stdout),
                    "stderr_tail": _tail(proc.stderr),
                },
                error_type=type(exc).__name__,
                error=str(exc),
            )
        for row in payload.get("steps", []):
            if not isinstance(row, dict):
                continue
            section = str(row.get("section") or "")
            number = row.get("number")
            name = str(row.get("name") or "")
            steps.append(
                StepTiming(
                    surface=surface,
                    name=f"{section} #{number}: {name}",
                    elapsed_s=float(row.get("elapsed_s", 0.0) or 0.0),
                    status="passed" if row.get("passed") else "failed",
                    metadata={
                        "section": section,
                        "number": number if isinstance(number, int) else str(number),
                        "forensic_only": bool(row.get("forensic_only")),
                        "local_smoke_only": bool(row.get("local_smoke_only")),
                    },
                )
            )
        metadata.update({
            "max_workers": payload.get("max_workers"),
            "wall_elapsed_s": payload.get("wall_elapsed_s"),
            "serial_elapsed_s": payload.get("serial_elapsed_s"),
            "parallel_speedup_estimate": payload.get("parallel_speedup_estimate"),
            "slowest_step_elapsed_s": payload.get("slowest_step_elapsed_s"),
            "slow_step_count": payload.get("slow_step_count"),
            "step_count": payload.get("step_count"),
        })
        failed_steps = [step for step in steps if step.status != "passed"]
        if failed_steps:
            metadata["failed_step_count"] = len(failed_steps)
            metadata["failed_steps"] = [step.name for step in failed_steps[:12]]
        error = ""
        if proc.returncode != 0:
            error = _failed_step_summary(failed_steps) or _tail(proc.stdout + proc.stderr)
        return SurfaceProfile(
            name=surface,
            status="passed" if proc.returncode == 0 else "failed",
            elapsed_s=elapsed_s,
            steps=steps,
            metadata=metadata,
            error_type="" if proc.returncode == 0 else "CalledProcessError",
            error=error,
        )


def json_loads_path(path: Path) -> dict[str, object]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _tail(text: str, *, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _failed_step_summary(steps: list[StepTiming], *, max_items: int = 8) -> str:
    if not steps:
        return ""
    labels = [f"{step.name} ({step.status})" for step in steps[:max_items]]
    suffix = f"; ... +{len(steps) - max_items} more" if len(steps) > max_items else ""
    return "Failed all-lanes step(s): " + "; ".join(labels) + suffix


def _build_report(
    surfaces: list[SurfaceProfile],
    *,
    top: int,
    max_step_records: int | None,
    generated_at: str | None = None,
) -> dict[str, object]:
    generated = generated_at or _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")
    all_steps = [step for surface in surfaces for step in surface.steps]
    hot_steps = sorted(all_steps, key=_step_sort_key)[:top]
    return {
        "schema": PROFILE_SCHEMA,
        "generated_at": generated,
        "repo_root": str(REPO),
        "surface_count": len(surfaces),
        "failed_surface_count": sum(1 for surface in surfaces if surface.status != "passed"),
        "total_elapsed_s": round(sum(surface.elapsed_s for surface in surfaces), 6),
        "surfaces": [
            surface.to_json(max_step_records=max_step_records)
            for surface in sorted(surfaces, key=lambda item: item.name)
        ],
        "hot_steps": [step.to_json() for step in hot_steps],
    }


def _print_report(report: dict[str, object], *, top: int) -> None:
    print("Preflight latency profile")
    print(f"schema: {report['schema']}")
    print(f"repo: {report['repo_root']}")
    print()
    print("Surfaces:")
    for surface in report["surfaces"]:
        if not isinstance(surface, dict):
            continue
        meta = surface.get("metadata")
        detail = ""
        if isinstance(meta, dict):
            useful = []
            for key in (
                "step_count",
                "file_count",
                "entity_count",
                "hazard_count",
                "wall_elapsed_s",
                "serial_elapsed_s",
                "max_workers",
                "parallel_speedup_estimate",
                "slow_step_count",
                "failed_step_count",
            ):
                if key in meta:
                    useful.append(f"{key}={meta[key]}")
            if useful:
                detail = " (" + ", ".join(useful) + ")"
        print(
            f"  {float(surface['elapsed_s']):8.3f}s  "
            f"{str(surface['status']).upper():6s}  {surface['name']}{detail}"
        )
        if surface.get("error"):
            print(f"            error: {surface['error']}")
    print()
    print(f"Slowest {top} steps:")
    for step in report["hot_steps"]:
        if not isinstance(step, dict):
            continue
        label = step.get("path") or step["name"]
        print(
            f"  {float(step['elapsed_s']):8.3f}s  "
            f"{str(step['status']).upper():6s}  {step['surface']} :: {label}"
        )


def _run_surface(name: str, args: argparse.Namespace) -> SurfaceProfile:
    if name == "preflight-all-codebase":
        return _profile_preflight_all_codebase(
            use_fs_cache=not args.no_fs_cache,
            timeout_s=args.preflight_timeout_s,
        )
    if name == "dispatch-hazards":
        return _profile_dispatch_hazards()
    if name == "review-tracker-scan":
        return _profile_review_tracker_scan()
    if name == "all-lanes":
        return _profile_all_lanes(jobs=args.all_lanes_jobs)
    raise ValueError(f"unsupported surface: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        choices=SURFACES,
        action="append",
        default=[],
        help=(
            "Surface to profile. Repeatable. Default: preflight-all-codebase. "
            "Use --all-surfaces for the local scanner/test-adjacent suite."
        ),
    )
    parser.add_argument(
        "--all-surfaces",
        action="store_true",
        help="Profile every local surface, including all-lanes preflight.",
    )
    parser.add_argument("--json-out", type=Path, help="Write deterministic JSON report.")
    parser.add_argument("--top", type=int, default=12, help="Number of slow steps to print/store.")
    parser.add_argument(
        "--max-step-records",
        type=int,
        default=500,
        help="Maximum per-surface steps stored in JSON after hot sorting; use -1 for all.",
    )
    parser.add_argument(
        "--no-fs-cache",
        action="store_true",
        help="Disable the preflight_all filesystem cache for comparison runs.",
    )
    parser.add_argument(
        "--preflight-timeout-s",
        type=float,
        default=None,
        help=(
            "Stop preflight-all-codebase after this many seconds and still emit "
            "the partial timing report. Default: no timeout."
        ),
    )
    parser.add_argument(
        "--all-lanes-jobs",
        type=int,
        default=None,
        help="Forward --jobs N when profiling tools/all_lanes_preflight.py.",
    )
    parser.add_argument(
        "--fail-on-surface-failure",
        action="store_true",
        help="Exit non-zero when any profiled surface reports failed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be >= 1")
    if args.max_step_records < -1:
        parser.error("--max-step-records must be >= -1")
    if args.all_lanes_jobs is not None and args.all_lanes_jobs < 1:
        parser.error("--all-lanes-jobs must be >= 1")
    if args.preflight_timeout_s is not None and args.preflight_timeout_s <= 0:
        parser.error("--preflight-timeout-s must be > 0")

    surfaces = list(SURFACES) if args.all_surfaces else (args.surface or ["preflight-all-codebase"])
    profiles = [_run_surface(name, args) for name in surfaces]
    max_step_records = None if args.max_step_records == -1 else args.max_step_records
    report = _build_report(
        profiles,
        top=args.top,
        max_step_records=max_step_records,
    )
    _print_report(report, top=args.top)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(report), encoding="utf-8")
        print(f"\nJSON report: {args.json_out}")
    if args.fail_on_surface_failure and report["failed_surface_count"]:
        return int(report["failed_surface_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
