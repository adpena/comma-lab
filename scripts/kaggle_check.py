#!/usr/bin/env python3
"""Check all Kaggle kernel statuses and show errors for failed kernels.

Usage:
    python scripts/kaggle_check.py
    python scripts/kaggle_check.py --download-logs  # also save logs locally
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

KERNEL_SLUGS = [
    "adpena/pr101-bias-refine",
    "adpena/pr101-proxy-sweep",
    "adpena/comma-lab-asym-warp-base",
    "adpena/comma-lab-asym-warp-raft-only",
    "adpena/comma-lab-asym-warp-supervised",
    "adpena/comma-lab-constrained-gen-smoke",
    "adpena/comma-lab-debug-mount",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "reports" / "raw" / "kaggle_logs"


def _kaggle_cmd() -> list[str]:
    venv_kaggle = REPO_ROOT / ".venv" / "bin" / "kaggle"
    if venv_kaggle.exists():
        return [str(venv_kaggle)]
    import shutil
    found = shutil.which("kaggle")
    if found:
        return [found]
    if shutil.which("uv"):
        return ["uv", "run", "--with", "kaggle", "kaggle"]
    print(f"{RED}kaggle CLI not found. Install with: uv pip install kaggle{RESET}")
    sys.exit(1)


def get_kernel_status(kaggle_cmd: list[str], slug: str, *, timeout_s: int = 10) -> str | None:
    """Return status string or None if kernel not found."""
    try:
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "status", slug],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"STATUS_TIMEOUT_AFTER_{timeout_s}S"
    if result.returncode != 0:
        return None
    # Output format: 'slug has status "STATUS"'
    stdout = result.stdout.strip()
    if '"' in stdout:
        return stdout.split('"')[1]
    return stdout


def get_kernel_log(
    kaggle_cmd: list[str],
    slug: str,
    download_dir: Path | None = None,
    *,
    timeout_s: int = 20,
) -> list[str]:
    """Download kernel output and return stderr lines from the log."""
    # Kaggle output can contain nested directories created or touched by the
    # CLI while a timeout/error path is unwinding. A failed cleanup is not a
    # status-check failure; the actionable signal is the kernel status/log.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        out_dir = Path(tmpdir)
        try:
            result = subprocess.run(
                [*kaggle_cmd, "kernels", "output", slug, "-p", str(out_dir)],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return [f"(log fetch timed out after {timeout_s}s for {slug})"]
        if result.returncode != 0:
            return [f"(could not fetch log: {result.stderr.strip()})"]

        # Kaggle outputs a JSON log file
        log_lines: list[str] = []
        for log_file in sorted(p for p in out_dir.rglob("*") if p.is_file()):
            content = log_file.read_text(errors="replace")

            # Save locally if requested
            if download_dir is not None:
                download_dir.mkdir(parents=True, exist_ok=True)
                slug_name = slug.split("/")[-1]
                rel_name = "_".join(log_file.relative_to(out_dir).parts)
                dest = download_dir / f"{slug_name}_{rel_name}"
                dest.write_text(content)

            # Try to parse as JSON log (Kaggle format: list of {data, stream} objects)
            try:
                entries = json.loads(content)
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("stream") == "stderr":
                            log_lines.append(entry.get("data", "").rstrip())
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Fallback: treat as plain text.
            # Inject the filename as a log line so marker files (e.g.
            # P100_RETRY_NEEDED) are discoverable by name, not just content.
            log_lines.append(f"[file: {log_file.relative_to(out_dir)}]")
            log_lines.extend(content.splitlines())

        return log_lines


def format_status(status: str | None) -> str:
    if status is None:
        return f"{YELLOW}NOT FOUND{RESET}"
    s = status.upper()
    if "RUNNING" in s:
        return f"{GREEN}{BOLD}{s}{RESET}"
    elif any(token in s for token in ("ERROR", "CANCELACKNOWLEDGED", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL")):
        return f"{RED}{BOLD}{s}{RESET}"
    elif "COMPLETE" in s:
        return f"{BLUE}{BOLD}{s}{RESET}"
    elif "TIMEOUT" in s:
        return f"{YELLOW}{BOLD}{s}{RESET}"
    else:
        return f"{YELLOW}{s}{RESET}"


def is_error_status(status: str | None) -> bool:
    if status is None:
        return False
    s = status.upper()
    return any(
        token in s
        for token in ("ERROR", "CANCELACKNOWLEDGED", "CANCEL_ACKNOWLEDGED", "CANCELLED", "TIMEOUT")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-logs",
        action="store_true",
        help="Save logs locally to reports/raw/kaggle_logs/",
    )
    parser.add_argument(
        "--log-timeout-s",
        type=int,
        default=20,
        help="Per-kernel timeout for `kaggle kernels output` when fetching failed-kernel logs.",
    )
    parser.add_argument(
        "--status-timeout-s",
        type=int,
        default=10,
        help="Per-kernel timeout for `kaggle kernels status`.",
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="Additional Kaggle kernel slug to check. May be repeated.",
    )
    parser.add_argument(
        "--only-kernel",
        action="append",
        default=[],
        help="Check only this Kaggle kernel slug. May be repeated.",
    )
    args = parser.parse_args()
    kaggle_cmd = _kaggle_cmd()
    download_dir = LOG_DIR if args.download_logs else None
    slugs = list(dict.fromkeys(args.only_kernel or [*KERNEL_SLUGS, *args.kernel]))

    print(f"\n{BOLD}Kaggle Kernel Status Check{RESET}")
    print(f"{'=' * 60}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    error_slugs: list[str] = []
    for slug in slugs:
        short = slug.split("/")[-1]
        status = get_kernel_status(kaggle_cmd, slug, timeout_s=args.status_timeout_s)
        status_str = format_status(status)
        print(f"  {short:<45s} {status_str}")

        if is_error_status(status):
            error_slugs.append(slug)

    if error_slugs:
        print(f"\n{'=' * 60}")
        print(f"{RED}{BOLD}Error logs:{RESET}\n")
        for slug in error_slugs:
            short = slug.split("/")[-1]
            lines = get_kernel_log(kaggle_cmd, slug, download_dir=download_dir, timeout_s=args.log_timeout_s)

            # Detect P100 retry marker in output files.
            # The bootstrap writes a marker file to /kaggle/working/P100_RETRY_NEEDED
            # whose content mentions compute capability. Also match the stdout
            # message "is unsupported" from the bootstrap preamble.
            is_p100 = any(
                "P100_RETRY_NEEDED" in line
                or "compute capability" in line.lower()
                or ("is unsupported" in line and "sm_" in line)
                for line in lines
            )
            if is_p100:
                print(f"  {YELLOW}--- {short} (P100 — retry needed) ---{RESET}")
                print(f"    {YELLOW}Kernel got assigned a P100 (sm_60). Not a real error.{RESET}")
                print(f"    {YELLOW}Re-run the kernel to get a T4/V100 assignment.{RESET}")
            else:
                print(f"  {RED}--- {short} ---{RESET}")
                # Show last 10 stderr lines
                tail = lines[-10:] if len(lines) > 10 else lines
                for line in tail:
                    print(f"    {line}")
                if len(lines) > 10:
                    print(f"    {YELLOW}... ({len(lines) - 10} earlier lines omitted){RESET}")
            print()

    if args.download_logs:
        print(f"  Logs saved to: {LOG_DIR}")

    print()
    return 1 if error_slugs else 0


if __name__ == "__main__":
    raise SystemExit(main())
