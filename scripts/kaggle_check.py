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
    "adpena/comma-lab-asym-warp-base",
    "adpena/comma-lab-asym-warp-raft-only",
    "adpena/comma-lab-asym-warp-supervised",
    "adpena/comma-lab-constrained-gen-smoke",
    "adpena/comma-lab-debug-mount",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "reports" / "raw" / "kaggle_logs"


def _kaggle_bin() -> str:
    venv_kaggle = REPO_ROOT / ".venv" / "bin" / "kaggle"
    if venv_kaggle.exists():
        return str(venv_kaggle)
    import shutil
    found = shutil.which("kaggle")
    if found:
        return found
    print(f"{RED}kaggle CLI not found. Install with: uv pip install kaggle{RESET}")
    sys.exit(1)


def get_kernel_status(kaggle: str, slug: str) -> str | None:
    """Return status string or None if kernel not found."""
    result = subprocess.run(
        [kaggle, "kernels", "status", slug],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    # Output format: 'slug has status "STATUS"'
    stdout = result.stdout.strip()
    if '"' in stdout:
        return stdout.split('"')[1]
    return stdout


def get_kernel_log(kaggle: str, slug: str, download_dir: Path | None = None) -> list[str]:
    """Download kernel output and return stderr lines from the log."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        result = subprocess.run(
            [kaggle, "kernels", "output", slug, "-p", str(out_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return [f"(could not fetch log: {result.stderr.strip()})"]

        # Kaggle outputs a JSON log file
        log_lines: list[str] = []
        for log_file in sorted(out_dir.iterdir()):
            content = log_file.read_text(errors="replace")

            # Save locally if requested
            if download_dir is not None:
                download_dir.mkdir(parents=True, exist_ok=True)
                slug_name = slug.split("/")[-1]
                dest = download_dir / f"{slug_name}_{log_file.name}"
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
            log_lines.append(f"[file: {log_file.name}]")
            log_lines.extend(content.splitlines())

        return log_lines


def format_status(status: str | None) -> str:
    if status is None:
        return f"{YELLOW}NOT FOUND{RESET}"
    s = status.upper()
    if s == "RUNNING":
        return f"{GREEN}{BOLD}{s}{RESET}"
    elif s in ("ERROR", "CANCELACKNOWLEDGED", "CANCEL"):
        return f"{RED}{BOLD}{s}{RESET}"
    elif s == "COMPLETE":
        return f"{BLUE}{BOLD}{s}{RESET}"
    else:
        return f"{YELLOW}{s}{RESET}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-logs",
        action="store_true",
        help="Save logs locally to reports/raw/kaggle_logs/",
    )
    args = parser.parse_args()
    kaggle = _kaggle_bin()
    download_dir = LOG_DIR if args.download_logs else None

    print(f"\n{BOLD}Kaggle Kernel Status Check{RESET}")
    print(f"{'=' * 60}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    error_slugs: list[str] = []
    for slug in KERNEL_SLUGS:
        short = slug.split("/")[-1]
        status = get_kernel_status(kaggle, slug)
        status_str = format_status(status)
        print(f"  {short:<45s} {status_str}")

        if status and status.upper() in ("ERROR", "CANCELACKNOWLEDGED"):
            error_slugs.append(slug)

    if error_slugs:
        print(f"\n{'=' * 60}")
        print(f"{RED}{BOLD}Error logs:{RESET}\n")
        for slug in error_slugs:
            short = slug.split("/")[-1]
            lines = get_kernel_log(kaggle, slug, download_dir=download_dir)

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
