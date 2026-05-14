#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and run the Zig exact-substring source scanner.

The Zig binary is a native acceleration leaf for source/preflight scans. It is
not a source of truth: Python tests compare its output to the Python oracle
before any preflight check can rely on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tool_bootstrap import repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tools.tool_bootstrap import repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ZIG_SOURCE = REPO_ROOT / "native" / "zig" / "source_needle_scan.zig"
DEFAULT_BINARY = REPO_ROOT / ".omx" / "cache" / "zig" / "source_needle_scan"
SCHEMA = "pact.zig_source_needle_scan.v1"


class ZigSourceScanError(RuntimeError):
    """Raised when the native scanner cannot be built or run."""


def zig_available() -> bool:
    return shutil.which("zig") is not None


def build_zig_source_scanner(
    *,
    binary_path: Path = DEFAULT_BINARY,
    optimize: str = "ReleaseFast",
) -> Path:
    """Build the native scanner when the source is newer than the binary."""

    if not zig_available():
        raise ZigSourceScanError("zig executable not found on PATH")
    if not ZIG_SOURCE.is_file():
        raise ZigSourceScanError(f"Zig source missing: {ZIG_SOURCE}")
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        binary_path.is_file()
        and binary_path.stat().st_mtime_ns >= ZIG_SOURCE.stat().st_mtime_ns
    ):
        return binary_path
    cmd = [
        "zig",
        "build-exe",
        "-O",
        optimize,
        str(ZIG_SOURCE),
        f"-femit-bin={binary_path}",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ZigSourceScanError(
            "Zig source scanner build failed:\n"
            + proc.stdout
            + proc.stderr
        )
    return binary_path


def run_zig_source_scan(
    *,
    root: Path = REPO_ROOT,
    dirs: list[str] | tuple[str, ...] = (".",),
    suffixes: list[str] | tuple[str, ...] = (".py",),
    needles: list[str] | tuple[str, ...],
    require_all: bool = False,
    binary_path: Path = DEFAULT_BINARY,
    build: bool = True,
    max_file_bytes: int = 16 * 1024 * 1024,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run the native source scanner and return its parsed JSON output."""

    if not needles:
        raise ZigSourceScanError("at least one needle is required")
    if max_file_bytes < 1:
        raise ZigSourceScanError("max_file_bytes must be >= 1")
    binary = build_zig_source_scanner(binary_path=binary_path) if build else binary_path
    if not binary.is_file():
        raise ZigSourceScanError(f"native scanner binary missing: {binary}")
    cmd = [str(binary), "--root", str(root)]
    for item in dirs:
        cmd.extend(["--dir", item])
    for suffix in suffixes:
        cmd.extend(["--suffix", suffix])
    for needle in needles:
        cmd.extend(["--needle", needle])
    if require_all:
        cmd.append("--require-all")
    cmd.extend(["--max-file-bytes", str(max_file_bytes)])
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise ZigSourceScanError(proc.stdout + proc.stderr)
    payload = json.loads(proc.stdout)
    if payload.get("schema") != SCHEMA:
        raise ZigSourceScanError(f"unexpected scanner schema: {payload.get('schema')!r}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--suffix", action="append", default=[])
    parser.add_argument("--needle", action="append", required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--binary-path", type=Path, default=DEFAULT_BINARY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_zig_source_scan(
            root=args.root,
            dirs=tuple(args.dir or ["."]),
            suffixes=tuple(args.suffix or [".py"]),
            needles=tuple(args.needle),
            require_all=args.require_all,
            binary_path=args.binary_path,
            build=not args.no_build,
            max_file_bytes=args.max_file_bytes,
        )
    except (OSError, subprocess.SubprocessError, ValueError, ZigSourceScanError) as exc:
        print(f"zig source scan failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
