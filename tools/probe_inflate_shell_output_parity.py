#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe exact inflate.sh output parity between two contest packets."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, sha256_file, write_json  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402


def _unsafe_zip_name(name: str) -> str | None:
    if not name:
        return "empty_member_name"
    if "\\" in name or "\x00" in name or any(ord(ch) < 32 for ch in name):
        return "unsafe_member_name"
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts:
        return "zip_slip_member_name"
    if any(part.startswith(".") for part in member.parts):
        return "hidden_member_name"
    return None


def _extract_archive(archive: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            reason = _unsafe_zip_name(info.filename)
            if reason is not None:
                raise ValueError(f"{archive}: unsafe ZIP member {info.filename!r}: {reason}")
            payload = zf.read(info)
            target = out_dir / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            rows.append(
                {
                    "name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "crc": int(info.CRC),
                    "sha256": sha256_file(target),
                }
            )
    return rows


def _output_records(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*"), key=lambda item: item.relative_to(output_dir).as_posix()):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _run_inflate(
    *,
    label: str,
    archive: Path,
    inflate_sh: Path,
    video_names: list[str],
    work_root: Path,
    timeout_s: float,
    env: dict[str, str],
) -> dict[str, Any]:
    archive_dir = work_root / f"{label}_archive"
    output_dir = work_root / f"{label}_output"
    archive_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    file_list = work_root / f"{label}_file_list.txt"
    file_list.write_text("".join(f"{name}\n" for name in video_names), encoding="utf-8")
    members = _extract_archive(archive, archive_dir)
    start = time.monotonic()
    proc = subprocess.run(
        [str(inflate_sh), str(archive_dir), str(output_dir), str(file_list)],
        cwd=inflate_sh.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.monotonic() - start
    outputs = _output_records(output_dir)
    return {
        "label": label,
        "archive": {
            "path": archive.as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "members": members,
        },
        "inflate_sh": {
            "path": inflate_sh.as_posix(),
            "bytes": inflate_sh.stat().st_size,
            "sha256": sha256_file(inflate_sh),
        },
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "outputs": outputs,
    }


def _output_map(record: dict[str, Any]) -> dict[str, tuple[int, str]]:
    return {
        str(row["relative_path"]): (int(row["bytes"]), str(row["sha256"]))
        for row in record.get("outputs", [])
        if isinstance(row, dict)
    }


def _write_python_shim(shim_dir: Path, name: str, python_bin: Path) -> Path:
    shim = shim_dir / name
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"exec {str(python_bin)!r} \"$@\"\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def build_report(args: argparse.Namespace, *, raw_argv: list[str]) -> dict[str, Any]:
    video_names = args.video_name or ["0.mkv"]
    source_archive = args.source_archive.resolve()
    source_inflate_sh = args.source_inflate_sh.resolve()
    candidate_archive = args.candidate_archive.resolve()
    candidate_inflate_sh = args.candidate_inflate_sh.resolve()
    keep_root = args.keep_work_dir
    if keep_root is not None:
        work_cm = None
        work_root = keep_root
        work_root.mkdir(parents=True, exist_ok=True)
        if any(work_root.iterdir()) and not args.allow_nonempty_keep_work_dir:
            raise ValueError(f"--keep-work-dir must be empty unless explicitly allowed: {work_root}")
    else:
        work_cm = tempfile.TemporaryDirectory(prefix="pact_inflate_shell_parity_")
        work_root = Path(work_cm.name)
    env = {**os.environ, "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", "1")}
    python_env_record: dict[str, Any] = {
        "python_bin_supplied": args.python_bin.as_posix() if args.python_bin is not None else None,
        "python_shim_dir": None,
        "python_shims": {},
        "python_shim_sha256s": {},
    }
    if args.python_bin is not None:
        python_bin = args.python_bin if args.python_bin.is_absolute() else Path.cwd() / args.python_bin
        python_bin = python_bin.absolute()
        shim_dir = work_root / "python_path_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shims = {
            name: _write_python_shim(shim_dir, name, python_bin)
            for name in ("python", "python3")
        }
        env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
        python_env_record = {
            "python_bin_supplied": python_bin.as_posix(),
            "python_shim_dir": shim_dir.as_posix(),
            "python_shims": {name: path.as_posix() for name, path in shims.items()},
            "python_shim_sha256s": {name: sha256_file(path) for name, path in shims.items()},
            "python_shim_sha256": sha256_file(shims["python"]),
        }
    try:
        source = _run_inflate(
            label="source",
            archive=source_archive,
            inflate_sh=source_inflate_sh,
            video_names=video_names,
            work_root=work_root,
            timeout_s=args.timeout_s,
            env=env,
        )
        candidate = _run_inflate(
            label="candidate",
            archive=candidate_archive,
            inflate_sh=candidate_inflate_sh,
            video_names=video_names,
            work_root=work_root,
            timeout_s=args.timeout_s,
            env=env,
        )
        source_outputs = _output_map(source)
        candidate_outputs = _output_map(candidate)
        mismatches: list[dict[str, Any]] = []
        for name in sorted(set(source_outputs) | set(candidate_outputs)):
            source_row = source_outputs.get(name)
            candidate_row = candidate_outputs.get(name)
            if source_row != candidate_row:
                mismatches.append(
                    {
                        "relative_path": name,
                        "source": source_row,
                        "candidate": candidate_row,
                    }
                )
        report: dict[str, Any] = {
            "schema": "pact.inflate_shell_output_parity_v1",
            "parity_method": "exact_inflate_sh_archive_dir_output_dir_file_list",
            "video_names": video_names,
            "passed": (
                source["returncode"] == 0
                and candidate["returncode"] == 0
                and not mismatches
                and bool(source_outputs)
            ),
            "score_claim": False,
            "dispatch_attempted": False,
            "work_dir_retained": keep_root.as_posix() if keep_root is not None else None,
            "runtime_environment": {
                "python_resolution": python_env_record,
                "python_no_user_site": env.get("PYTHONNOUSERSITE"),
            },
            "source": source,
            "candidate": candidate,
            "output_mismatches": mismatches,
        }
        return attach_tool_run_manifest(
            report,
            tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
            argv=raw_argv,
            input_paths=[
                source_archive,
                source_inflate_sh,
                candidate_archive,
                candidate_inflate_sh,
            ],
            repo_root=REPO_ROOT,
            output_path=args.json_out,
        )
    finally:
        if work_cm is not None:
            work_cm.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-inflate-sh", required=True, type=Path)
    parser.add_argument("--candidate-archive", required=True, type=Path)
    parser.add_argument("--candidate-inflate-sh", required=True, type=Path)
    parser.add_argument("--video-name", action="append", default=[])
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--keep-work-dir", type=Path)
    parser.add_argument("--allow-nonempty-keep-work-dir", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        report = build_report(args, raw_argv=raw_argv)
    except (OSError, ValueError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        print(f"FATAL: inflate shell parity probe failed: {exc}", file=sys.stderr)
        return 2
    if args.json_out is not None:
        write_json(args.json_out, report)
    else:
        print(json_text(report), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
