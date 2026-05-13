#!/usr/bin/env python3
"""Memory-safe relaunch supervisor for the macOS-CPU advisory canvas sweep.

This tool is deliberately advisory-only infrastructure. It does not create a
score claim, does not promote candidates, and does not dispatch GPU work. Its
job is to prevent the local macOS sweep from becoming an orphaned process tree
or memory-pressure source while preserving completed per-archive JSON evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = (
    REPO_ROOT
    / "experiments/results/lane_macos_cpu_substrate_canvas_sweep_20260513_20260513T162636Z"
)


def _run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return proc.stdout.strip()


def _matching_pids(sweep_dir: Path) -> list[int]:
    pattern = str(sweep_dir)
    out = _run_text(
        [
            "pgrep",
            "-f",
            rf"{pattern}|sweep_runner.py|contest_auth_eval.py|upstream/evaluate.py|inflate.sh",
        ]
    )
    pids: list[int] = []
    self_pid = os.getpid()
    for token in out.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid != self_pid:
            pids.append(pid)
    return sorted(set(pids))


def _terminate_pids(pids: list[int], *, grace_sec: float = 2.0) -> None:
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_exists(pid)]
        if not alive:
            return
        time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _macos_available_memory_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    page_size = 16_384
    vm_stat = _run_text(["vm_stat"])
    pages: dict[str, int] = {}
    for raw_line in vm_stat.splitlines():
        line = raw_line.strip().rstrip(".")
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            pages[key] = int(digits)
    return (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
    ) * page_size


def _clean_stale_workdirs(sweep_dir: Path) -> list[str]:
    cleaned: list[str] = []
    per_archive = sweep_dir / "per_archive"
    if not per_archive.is_dir():
        return cleaned
    for stale_dir in per_archive.glob("*/work/*"):
        if stale_dir.name not in {"inflated", "extracted"} or not stale_dir.is_dir():
            continue
        subprocess.run(["rm", "-rf", str(stale_dir)], check=False)
        cleaned.append(str(stale_dir.relative_to(REPO_ROOT)))
    return cleaned


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launch_runner(args: argparse.Namespace, log_path: Path) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        str(args.sweep_dir / "sweep_runner.py"),
        "--targets-json",
        str(args.sweep_dir / "tier12_targets.json"),
        "--concurrency",
        str(args.concurrency),
        "--timeout",
        str(args.timeout),
    ]
    env = os.environ.copy()
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'upstream'}:{env.get('PYTHONPATH', '')}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n[safe-relaunch] {dt.datetime.now(dt.UTC).isoformat()} cmd={' '.join(cmd)}\n")
    log_file.flush()
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _supervise(args: argparse.Namespace) -> int:
    if args.stop_orphans:
        _terminate_pids(_matching_pids(args.sweep_dir))
    cleaned = _clean_stale_workdirs(args.sweep_dir) if args.clean_stale_workdirs else []
    log_path = args.sweep_dir / "safe_relaunch.log"
    child = _launch_runner(args, log_path)
    manifest_path = args.sweep_dir / "safe_relaunch_manifest.json"
    _write_manifest(
        manifest_path,
        {
            "schema": "macos_cpu_canvas_safe_relaunch_manifest_v1",
            "evidence_grade": "macOS-CPU-advisory",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "supervisor_pid": os.getpid(),
            "runner_pid": child.pid,
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "sweep_dir": str(args.sweep_dir.relative_to(REPO_ROOT)),
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
            "min_available_memory_gb": args.min_available_memory_gb,
            "cleaned_stale_dirs_count": len(cleaned),
            "cleaned_stale_dirs": cleaned[:200],
            "log_path": str(log_path.relative_to(REPO_ROOT)),
        },
    )
    min_bytes = int(args.min_available_memory_gb * (1024**3))
    while child.poll() is None:
        available = _macos_available_memory_bytes()
        if available is not None and available < min_bytes:
            os.killpg(child.pid, signal.SIGTERM)
            time.sleep(2)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            _write_manifest(
                manifest_path,
                {
                    "schema": "macos_cpu_canvas_safe_relaunch_manifest_v1",
                    "evidence_grade": "macOS-CPU-advisory",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "supervisor_pid": os.getpid(),
                    "runner_pid": child.pid,
                    "stopped_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "status": "stopped_memory_pressure",
                    "available_memory_bytes": available,
                    "min_available_memory_bytes": min_bytes,
                    "log_path": str(log_path.relative_to(REPO_ROOT)),
                },
            )
            return 77
        time.sleep(args.poll_seconds)
    rc = child.returncode
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
        }
    )
    _write_manifest(manifest_path, payload)
    return int(rc or 0)


def _detach(args: argparse.Namespace) -> int:
    cmd = [sys.executable, __file__, "--supervise"]
    cmd.extend(["--sweep-dir", str(args.sweep_dir)])
    cmd.extend(["--concurrency", str(args.concurrency)])
    cmd.extend(["--timeout", str(args.timeout)])
    cmd.extend(["--min-available-memory-gb", str(args.min_available_memory_gb)])
    cmd.extend(["--poll-seconds", str(args.poll_seconds)])
    if args.stop_orphans:
        cmd.append("--stop-orphans")
    if args.clean_stale_workdirs:
        cmd.append("--clean-stale-workdirs")
    log_path = args.sweep_dir / "safe_relaunch_supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(
        json.dumps(
            {
                "schema": "macos_cpu_canvas_safe_relaunch_detach_v1",
                "supervisor_pid": proc.pid,
                "sweep_dir": str(args.sweep_dir.relative_to(REPO_ROOT)),
                "log_path": str(log_path.relative_to(REPO_ROOT)),
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--min-available-memory-gb", type=float, default=16.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stop-orphans", action="store_true")
    parser.add_argument("--clean-stale-workdirs", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args(argv)
    args.sweep_dir = args.sweep_dir.resolve()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.concurrency > 1 and args.min_available_memory_gb < 32:
        raise SystemExit("concurrency > 1 requires --min-available-memory-gb >= 32")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.detach:
        return _detach(args)
    if args.supervise:
        return _supervise(args)
    cleaned = _clean_stale_workdirs(args.sweep_dir) if args.clean_stale_workdirs else []
    print(json.dumps({"cleaned_stale_dirs_count": len(cleaned), "score_claim": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
