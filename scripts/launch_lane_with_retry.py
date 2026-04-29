#!/usr/bin/env python3
"""launch_lane_with_retry.py — phase1+phase2 with auto-retry on Vast.ai failures.

Wraps `launch_lane_on_vastai.py phase1` + the phase2 split (wait+scp+extract+launch).
On any phase2 failure (NVDEC_BAD, SCP_FAIL, extract timeout), destroys the
instance and retries with a fresh host. Up to --max-retries (default 3).

Why: 2026-04-29 dispatches showed ~80% NVDEC_BAD rate on Vast.ai 4090.
Each manual retry cost ~5-10 min of operator time. This wrapper makes
dispatch resilient by design.

Usage:
  .venv/bin/python scripts/launch_lane_with_retry.py \\
    --lane-script scripts/remote_lane_X.sh --label lane_X --max-dph 0.40 \\
    --predicted-band 0.85 1.10 --estimated-cost 4.00 --max-retries 3

Exit codes:
  0 = lane successfully launched (instance running, lane_script tmux'd)
  1 = max retries exhausted
  2 = invalid args / pre-flight failure
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "scripts" / "launch_lane_on_vastai.py"
PYBIN = ".venv/bin/python"


def run_stage(args: list[str], timeout: int = 300) -> tuple[int, str]:
    """Run a launcher stage. Returns (returncode, combined output)."""
    try:
        proc = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s"


def parse_instance_id(stdout: str) -> int | None:
    for line in stdout.splitlines():
        if line.startswith("INSTANCE_ID="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def destroy(instance_id: int) -> None:
    """Best-effort destroy. Does not raise."""
    try:
        # subprocess-no-check-OK: best-effort destroy; ignore failures
        subprocess.run(
            ["bash", "-c", f"echo y | .venv/bin/vastai destroy instance {instance_id}"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
            check=False,
        )
    except Exception:
        pass


def attempt_dispatch(args: argparse.Namespace, attempt: int) -> tuple[bool, int | None, str]:
    """Single phase1+phase2 attempt. Returns (success, instance_id, log)."""
    log: list[str] = []
    log.append(f"=== ATTEMPT {attempt} ===")

    # phase1 — find offer + create instance
    label = f"{args.label}_a{attempt}"
    p1_args = [
        PYBIN, str(LAUNCHER), "phase1",
        "--lane-script", args.lane_script,
        "--label", label,
        "--max-dph", str(args.max_dph),
    ]
    if args.predicted_band:
        p1_args += ["--predicted-band", str(args.predicted_band[0]), str(args.predicted_band[1])]
    if args.estimated_cost is not None:
        p1_args += ["--estimated-cost", str(args.estimated_cost)]

    rc, out = run_stage(p1_args, timeout=120)
    log.append(f"[phase1] rc={rc}")
    if rc != 0:
        log.append(out[-500:])
        return False, None, "\n".join(log)

    iid = parse_instance_id(out)
    if iid is None:
        log.append("FAIL: phase1 succeeded but no INSTANCE_ID parsed")
        log.append(out[-500:])
        return False, None, "\n".join(log)
    log.append(f"  instance_id={iid}")

    # phase2-wait — may take 3-5 min
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-wait", "--instance-id", str(iid)],
        timeout=300,
    )
    log.append(f"[phase2-wait] rc={rc}")
    if rc != 0:
        log.append(out[-300:])
        destroy(iid)
        return False, iid, "\n".join(log)

    # phase2-scp — build tarball + ship to remote
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-scp", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=300,
    )
    log.append(f"[phase2-scp] rc={rc}")
    if rc != 0:
        log.append(out[-300:])
        destroy(iid)
        return False, iid, "\n".join(log)

    # phase2-extract — extract on remote + CUDA probe (auto-destroy on fail)
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-extract", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=240,
    )
    log.append(f"[phase2-extract] rc={rc}")
    if rc != 0:
        # phase2-extract auto-destroys on failure; no manual destroy needed
        log.append(out[-500:])
        return False, iid, "\n".join(log)

    # phase2-launch — subshell-detach lane (auto-destroys on NVDEC_BAD detection)
    rc, out = run_stage(
        [PYBIN, str(LAUNCHER), "phase2-launch", "--instance-id", str(iid),
         "--lane-script", args.lane_script],
        timeout=120,
    )
    log.append(f"[phase2-launch] rc={rc}")
    if rc != 0:
        log.append(out[-500:])
        # phase2-launch's NVDEC_BAD check destroys the instance itself
        return False, iid, "\n".join(log)

    log.append(f"✓ SUCCESS — instance {iid} running lane {args.label}")
    return True, iid, "\n".join(log)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lane-script", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--max-dph", type=float, default=0.40)
    p.add_argument("--predicted-band", nargs=2, type=float, metavar=("LOW", "HIGH"))
    p.add_argument("--estimated-cost", type=float)
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max attempts before giving up (default 3).")
    p.add_argument("--retry-delay", type=int, default=15,
                   help="Seconds between retries (default 15).")
    args = p.parse_args()

    if not (REPO_ROOT / args.lane_script).exists():
        print(f"FATAL: lane script missing: {args.lane_script}", file=sys.stderr)
        return 2

    print(f"=== launch_lane_with_retry: {args.label} (max {args.max_retries} attempts) ===")
    for attempt in range(1, args.max_retries + 1):
        success, iid, log = attempt_dispatch(args, attempt)
        print(log)
        if success:
            print(f"\n✓ DISPATCHED: instance={iid} label={args.label} attempts={attempt}")
            return 0
        if attempt < args.max_retries:
            print(f"  retrying in {args.retry_delay}s...\n")
            time.sleep(args.retry_delay)

    print(f"\n✗ FAILED: {args.max_retries} attempts exhausted for {args.label}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
