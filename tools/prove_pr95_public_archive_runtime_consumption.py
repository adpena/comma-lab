#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove a PR95-compatible archive member is consumed by the public inflate runtime."""

from __future__ import annotations

import argparse
import os
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    FALSE_AUTHORITY,
    parse_pr95_public_archive_zip,
)
from tac.repo_io import write_json_artifact  # noqa: E402

DEFAULT_INFLATE_SH = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
    / "source/submissions/hnerv_muon/inflate.sh"
)
CAMERA_H = 874
CAMERA_W = 1164
RGB_CHANNELS = 3


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _runtime_files(inflate_sh: Path) -> list[dict[str, Any]]:
    submission_dir = inflate_sh.resolve().parent
    paths = [
        inflate_sh,
        submission_dir / "inflate.py",
        submission_dir / "src/model.py",
        submission_dir / "src/codec.py",
    ]
    return [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def _expected_raw_bytes(meta: dict[str, Any]) -> int:
    return int(meta["n_pairs"]) * 2 * CAMERA_H * CAMERA_W * RGB_CHANNELS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--inflate-sh", type=Path, default=DEFAULT_INFLATE_SH)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Working directory. Defaults to <output-json parent>/runtime_consumption_work.",
    )
    parser.add_argument("--member-name", default="0.bin")
    parser.add_argument("--file-base", default="0")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Refuse predicted raw output above this size unless --allow-large-output.",
    )
    parser.add_argument("--allow-large-output", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    archive_zip = args.archive_zip.resolve()
    inflate_sh = args.inflate_sh.resolve()
    if not inflate_sh.is_file():
        raise SystemExit(f"inflate.sh not found: {inflate_sh}")
    packet = parse_pr95_public_archive_zip(archive_zip, member_name=args.member_name)
    expected_bytes = _expected_raw_bytes(packet.meta)
    if expected_bytes > args.max_output_bytes and not args.allow_large_output:
        raise SystemExit(
            f"refusing predicted raw output {expected_bytes} bytes above "
            f"--max-output-bytes {args.max_output_bytes}; pass --allow-large-output "
            "for intentional full-packet runtime proof"
        )

    output_json = args.output_json.resolve()
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else output_json.parent / "runtime_consumption_work"
    )
    data_dir = work_dir / "data"
    raw_dir = work_dir / "raw"
    file_list = work_dir / "file_list.txt"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip) as zf:
        member_bytes = zf.read(args.member_name)
    staged_member = data_dir / f"{args.file_base}.bin"
    staged_member.write_bytes(member_bytes)
    file_list.write_text(f"{args.file_base}.mkv\n", encoding="utf-8")

    env = os.environ.copy()
    venv_bin = REPO_ROOT / ".venv/bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    started = datetime.now(UTC)
    proc = subprocess.run(
        ["bash", inflate_sh.as_posix(), data_dir.as_posix(), raw_dir.as_posix(), file_list.as_posix()],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    raw_path = raw_dir / f"{args.file_base}.raw"
    raw_exists = raw_path.is_file()
    raw_bytes = raw_path.stat().st_size if raw_exists else 0
    runtime_consumption_proven = (
        proc.returncode == 0 and raw_exists and raw_bytes == expected_bytes
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pr95_hnerv_public_runtime_consumption_proof.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "started_utc": started.isoformat(),
        "lane_id": "lane_pr95_hnerv_mlx_reproduction",
        "source_pr": 95,
        "submission": "hnerv_muon",
        "archive_packet": packet.custody_manifest(),
        "inflate_sh": inflate_sh.as_posix(),
        "runtime_files": _runtime_files(inflate_sh),
        "work_dir": work_dir.as_posix(),
        "staged_member": staged_member.as_posix(),
        "file_list": file_list.as_posix(),
        "raw_output_path": raw_path.as_posix(),
        "expected_raw_bytes": expected_bytes,
        "raw_output_bytes": raw_bytes,
        "raw_output_sha256": _sha256_file(raw_path) if raw_exists else None,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "returncode": proc.returncode,
        "runtime_consumption_proven": runtime_consumption_proven,
        "full_frame_inflate_parity": False,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "runtime_consumption_smoke_is_not_score_authority",
                "full_frame_inflate_parity_against_source_runtime_not_run",
                "requires_exact_cpu_cuda_auth_eval_before_score_claim",
            ],
        },
        **FALSE_AUTHORITY,
    }
    write_json_artifact(output_json, payload)
    if not runtime_consumption_proven:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
