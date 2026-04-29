#!/usr/bin/env python3
"""Single-instance lane diagnosis: SSH, heartbeat, recent logs, archive bytes.

This tool answers the most common operator question — "what is lane X
actually doing right now?" — with a single command. Instead of N
separate `vastai show` / `ssh root@... tail run.log` invocations, this
prints a single audit block.

Outputs (per instance):
  - Vast.ai metadata (label, host, port, gpu_util, dph, accrued cost)
  - Heartbeat freshness (age in minutes; missing => still in setup)
  - Last 20 lines of run.log + setup.log + lane.log + auth_eval.log
  - GPU utilization sample
  - Archive bytes if `archive*.zip` exists on disk
  - Final score in run_record.json if present

Exit codes:
  0 — instance reachable + diagnosis printed
  1 — instance not in tracker
  2 — instance reachable but no logs visible (early setup OR misconfigured)
  3 — SSH failed (instance gone OR unreachable)

Usage:
    python tools/diagnose_lane.py 35759655
    python tools/diagnose_lane.py 35759655 --tail-lines 50
    python tools/diagnose_lane.py 35759655 --json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKER_PATH = REPO_ROOT / ".omx/state/vastai_active_instances.json"
VASTAI_BIN = REPO_ROOT / ".venv/bin/vastai"
SSH_BASE = [
    "ssh",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]
DEFAULT_TAIL_LINES = 20


@dataclass
class LaneDiagnosis:
    instance_id: str
    label: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    actual_status: str | None = None
    gpu_util_pct: float | None = None
    dph: float | None = None
    accrued_cost_usd: float | None = None
    heartbeat_age_minutes: float | None = None
    log_tails: dict[str, str] = field(default_factory=dict)
    archive_bytes: int | None = None
    final_score: float | None = None
    ssh_failed_reason: str | None = None


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run subprocess with explicit timeout/exception classification.

    Returns (returncode, stdout, stderr). returncode -1 distinguishes
    timeout (-1, "", "TIMEOUT") from generic exception (-1, "", "EXC: ...").
    Pattern matches scripts/verify_vast_instances.py for consistency.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except (OSError, ValueError) as e:
        return -1, "", f"EXC: {e}"


def _load_tracker() -> dict:
    if not TRACKER_PATH.exists():
        return {}
    try:
        return json.loads(TRACKER_PATH.read_text()) or {}
    except json.JSONDecodeError as e:
        print(f"[diagnose_lane] WARN: tracker JSON corrupt: {e}", file=sys.stderr)
        return {}


def _vast_show(instance_id: str) -> dict | None:
    rc, out, _ = _run([str(VASTAI_BIN), "show", "instance", instance_id, "--raw"])
    if rc != 0:
        return None
    try:
        d = json.loads(out)
        return d[0] if isinstance(d, list) and d else d if isinstance(d, dict) else None
    except (json.JSONDecodeError, IndexError):
        return None


def _ssh_collect(host: str, port: int, tail_lines: int) -> dict[str, str]:
    """Single SSH invocation that gathers all the diagnostic info.

    Reduces fan-out: 1 SSH call instead of N. Each section delimited
    by a `===<NAME>===` marker for parsing.
    """
    cmd = SSH_BASE + ["-p", str(port), f"root@{host}", (
        "cd /workspace/pact 2>/dev/null || { echo MISSING_REPO; exit 1; }; "
        "echo '===HEARTBEAT==='; "
        "find . -name 'heartbeat.log' -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1; "
        "echo '===RUN_LOG==='; "
        f"find . -name 'run.log' 2>/dev/null | head -1 | xargs -r tail -n {tail_lines} 2>/dev/null; "
        "echo '===SETUP_LOG==='; "
        f"find . -name 'setup.log' -o -name 'setup_full.log' 2>/dev/null | head -1 | xargs -r tail -n {tail_lines} 2>/dev/null; "
        "echo '===LANE_LOG==='; "
        f"find . -name 'lane.log' -o -name 'lane_*.log' 2>/dev/null | head -1 | xargs -r tail -n {tail_lines} 2>/dev/null; "
        "echo '===AUTH_EVAL==='; "
        f"find . -name 'auth_eval*.log' -o -name 'eval.log' 2>/dev/null | head -1 | xargs -r tail -n {tail_lines} 2>/dev/null; "
        "echo '===ARCHIVE==='; "
        "find . -name 'archive*.zip' -printf '%s %p\\n' 2>/dev/null | sort -rn | head -3; "
        "echo '===SCORE==='; "
        "find . -name 'run_record.json' -o -name 'auth_eval_result.json' 2>/dev/null | head -3 | xargs -r tail -n 100 2>/dev/null; "
        "echo '===GPU==='; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'NO_GPU'; "
        "echo '===END==='"
    )]
    rc, out, err = _run(cmd, timeout=30)
    if rc != 0:
        return {"_ssh_error": err.strip()[:200] or "unknown"}
    sections: dict[str, str] = {}
    cur_name = None
    cur_lines: list[str] = []
    for line in out.splitlines():
        if line.startswith("===") and line.endswith("==="):
            if cur_name:
                sections[cur_name] = "\n".join(cur_lines).strip()
            cur_name = line.strip("=")
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_name and cur_name != "END":
        sections[cur_name] = "\n".join(cur_lines).strip()
    return sections


def _heartbeat_age_minutes(heartbeat_section: str) -> float | None:
    if not heartbeat_section:
        return None
    last = heartbeat_section.splitlines()[-1].strip()
    if not last:
        return None
    try:
        # Format: "<unix-mtime> <path>"
        mtime = float(last.split()[0])
        return (datetime.now(timezone.utc).timestamp() - mtime) / 60.0
    except (ValueError, IndexError):
        return None


def _archive_bytes(archive_section: str) -> int | None:
    if not archive_section:
        return None
    first = archive_section.splitlines()[0].strip()
    try:
        return int(first.split()[0])
    except (ValueError, IndexError):
        return None


def _final_score(score_section: str) -> float | None:
    if not score_section:
        return None
    try:
        # Best-effort: try to parse a JSON object from the section.
        # Look for '"score":' or '"total":' or '"auth_score":'.
        for token in ('"score"', '"total"', '"auth_score"', '"final_score"'):
            idx = score_section.find(token)
            if idx >= 0:
                # Find value after ':'
                tail = score_section[idx:].split(":", 1)
                if len(tail) == 2:
                    val_str = tail[1].lstrip().split(",", 1)[0].split("}", 1)[0]
                    return float(val_str.strip().strip('"'))
    except (ValueError, IndexError):
        pass
    return None


def diagnose(instance_id: str, tail_lines: int = DEFAULT_TAIL_LINES) -> LaneDiagnosis:
    diag = LaneDiagnosis(instance_id=instance_id)
    tracker = _load_tracker()
    info = tracker.get(instance_id, {})
    diag.label = info.get("label")

    # Vast.ai metadata.
    vast_info = _vast_show(instance_id)
    if vast_info is None:
        diag.ssh_failed_reason = "vastai show returned no data (instance may be GONE)"
        return diag

    diag.actual_status = vast_info.get("actual_status")
    diag.gpu_util_pct = vast_info.get("gpu_util")
    diag.dph = vast_info.get("dph_total") or vast_info.get("dph")
    diag.accrued_cost_usd = vast_info.get("inet_up_cost")  # placeholder
    diag.ssh_host = vast_info.get("ssh_host")
    raw_port = vast_info.get("ssh_port")
    if raw_port is not None:
        try:
            diag.ssh_port = int(raw_port)
        except (TypeError, ValueError):
            diag.ssh_failed_reason = f"non-integer ssh_port: {raw_port!r}"
            return diag

    if not diag.ssh_host or not diag.ssh_port:
        diag.ssh_failed_reason = (
            f"ssh_host={diag.ssh_host!r} ssh_port={diag.ssh_port!r} — "
            "Vast.ai hasn't propagated SSH info yet (instance may be booting)"
        )
        return diag

    sections = _ssh_collect(diag.ssh_host, diag.ssh_port, tail_lines)
    if "_ssh_error" in sections:
        diag.ssh_failed_reason = sections["_ssh_error"]
        return diag

    diag.heartbeat_age_minutes = _heartbeat_age_minutes(sections.get("HEARTBEAT", ""))
    diag.log_tails = {
        k: v for k, v in sections.items()
        if k in ("RUN_LOG", "SETUP_LOG", "LANE_LOG", "AUTH_EVAL", "GPU")
        and v.strip()
    }
    diag.archive_bytes = _archive_bytes(sections.get("ARCHIVE", ""))
    diag.final_score = _final_score(sections.get("SCORE", ""))
    return diag


def render(diag: LaneDiagnosis) -> str:
    lines = []
    lines.append(f"=== diagnose_lane: instance {diag.instance_id} ===")
    lines.append(f"  label:        {diag.label or '(not in tracker)'}")
    lines.append(f"  status:       {diag.actual_status or '(unknown)'}")
    if diag.dph is not None:
        lines.append(f"  cost:         ${diag.dph:.4f}/hr")
    if diag.gpu_util_pct is not None:
        lines.append(f"  gpu_util:     {diag.gpu_util_pct}%")
    if diag.ssh_host and diag.ssh_port:
        lines.append(f"  ssh:          {diag.ssh_host}:{diag.ssh_port}")
    if diag.heartbeat_age_minutes is not None:
        flag = " STALE" if diag.heartbeat_age_minutes > 30 else " FRESH"
        lines.append(f"  heartbeat:    {diag.heartbeat_age_minutes:.1f} min ago{flag}")
    elif not diag.ssh_failed_reason:
        lines.append("  heartbeat:    MISSING (lane likely still in setup_full.sh)")
    if diag.archive_bytes is not None:
        rate = 25 * diag.archive_bytes / 37_545_489
        lines.append(f"  archive:      {diag.archive_bytes:,}B (rate ~{rate:.4f})")
    if diag.final_score is not None:
        lines.append(f"  final_score:  {diag.final_score:.4f}")
    if diag.ssh_failed_reason:
        lines.append(f"  SSH_FAILED:   {diag.ssh_failed_reason}")
    lines.append("")
    if diag.log_tails:
        for log_name, content in diag.log_tails.items():
            if not content:
                continue
            lines.append(f"  --- {log_name} ---")
            for ln in content.splitlines()[-25:]:
                lines.append(f"    {ln}")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-call diagnosis of a Vast.ai lane instance."
    )
    parser.add_argument("instance_id", type=str)
    parser.add_argument(
        "--tail-lines", type=int, default=DEFAULT_TAIL_LINES,
        help=f"Number of log lines to tail (default: {DEFAULT_TAIL_LINES})",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    diag = diagnose(args.instance_id, tail_lines=args.tail_lines)
    if args.json:
        print(json.dumps(asdict(diag), indent=2, default=str))
    else:
        print(render(diag))

    if diag.ssh_failed_reason:
        return 3
    if diag.heartbeat_age_minutes is None and not diag.log_tails:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
