#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove two submission packets match at the shell-level inflate contract.

This runs each packet's ``inflate.sh archive_dir output_dir file_list`` entry
point against extracted archive contents, hashes the emitted raw file, compares
the byte streams, and writes a small JSON/Markdown proof. Large extracted data
and raw outputs are deleted by default after hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShellInflateSide:
    label: str
    archive: str
    archive_bytes: int
    archive_sha256: str
    submission_dir: str
    inflate_sh: str
    output_raw_bytes: int
    output_raw_sha256: str
    inflate_seconds: float


@dataclass(frozen=True)
class ShellInflateParityProof:
    schema: str
    generated_at_utc: str
    file_list_entry: str
    output_basename: str
    python_bin: str
    left: ShellInflateSide
    right: ShellInflateSide
    output_bytes_match: bool
    output_sha256_match: bool
    cmp_equal: bool
    scratch_retained: bool
    scratch_dir: str
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _safe_extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (output_dir / info.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"refusing unsafe zip member path: {info.filename!r}")
        archive.extractall(output_dir)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_fh, right.open("rb") as right_fh:
        while True:
            left_chunk = left_fh.read(1 << 20)
            right_chunk = right_fh.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _run_inflate(
    *,
    label: str,
    archive: Path,
    submission_dir: Path,
    file_list: Path,
    output_dir: Path,
    data_dir: Path,
    python_bin: str,
    output_basename: str,
) -> ShellInflateSide:
    _safe_extract_zip(archive, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = submission_dir / "inflate.sh"
    if not inflate_sh.is_file():
        raise FileNotFoundError(f"missing inflate.sh: {inflate_sh}")
    env = os.environ.copy()
    env["PACT_PYTHON_BIN"] = python_bin
    start = time.perf_counter()
    subprocess.run(
        [str(inflate_sh), str(data_dir), str(output_dir), str(file_list)],
        check=True,
        env=env,
    )
    elapsed = time.perf_counter() - start
    raw_path = output_dir / output_basename
    if not raw_path.is_file():
        raise FileNotFoundError(f"inflate did not write expected raw output: {raw_path}")
    return ShellInflateSide(
        label=label,
        archive=_repo_rel(archive),
        archive_bytes=archive.stat().st_size,
        archive_sha256=_sha256_file(archive),
        submission_dir=_repo_rel(submission_dir),
        inflate_sh=_repo_rel(inflate_sh),
        output_raw_bytes=raw_path.stat().st_size,
        output_raw_sha256=_sha256_file(raw_path),
        inflate_seconds=elapsed,
    )


def build_proof(
    *,
    left_archive: Path,
    left_submission_dir: Path,
    right_archive: Path,
    right_submission_dir: Path,
    output_dir: Path,
    file_list_entry: str,
    python_bin: str,
    keep_scratch: bool,
) -> ShellInflateParityProof:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_list = output_dir / "file_list.txt"
    file_list.write_text(file_list_entry.rstrip("\n") + "\n", encoding="utf-8")
    output_basename = f"{Path(file_list_entry).stem}.raw"
    scratch = output_dir / "scratch"
    left_data = scratch / "left_data"
    right_data = scratch / "right_data"
    left_out = scratch / "left_out"
    right_out = scratch / "right_out"
    left = _run_inflate(
        label="left",
        archive=left_archive,
        submission_dir=left_submission_dir,
        file_list=file_list,
        output_dir=left_out,
        data_dir=left_data,
        python_bin=python_bin,
        output_basename=output_basename,
    )
    right = _run_inflate(
        label="right",
        archive=right_archive,
        submission_dir=right_submission_dir,
        file_list=file_list,
        output_dir=right_out,
        data_dir=right_data,
        python_bin=python_bin,
        output_basename=output_basename,
    )
    left_raw = left_out / output_basename
    right_raw = right_out / output_basename
    cmp_equal = _files_equal(left_raw, right_raw)
    proof = ShellInflateParityProof(
        schema="shell_inflate_parity_proof_v1",
        generated_at_utc=_utc_iso(),
        file_list_entry=file_list_entry,
        output_basename=output_basename,
        python_bin=python_bin,
        left=left,
        right=right,
        output_bytes_match=left.output_raw_bytes == right.output_raw_bytes,
        output_sha256_match=left.output_raw_sha256 == right.output_raw_sha256,
        cmp_equal=cmp_equal,
        scratch_retained=keep_scratch,
        scratch_dir=_repo_rel(scratch),
    )
    if not keep_scratch:
        shutil.rmtree(scratch)
    return proof


def render_markdown(proof: ShellInflateParityProof) -> str:
    return "\n".join(
        [
            "# Shell Inflate Parity Proof",
            "",
            f"- Generated UTC: {proof.generated_at_utc}",
            f"- File list entry: `{proof.file_list_entry}`",
            f"- Output raw basename: `{proof.output_basename}`",
            f"- Python bin: `{proof.python_bin}`",
            f"- Left archive: `{proof.left.archive}`",
            f"- Right archive: `{proof.right.archive}`",
            f"- Left archive SHA-256: `{proof.left.archive_sha256}`",
            f"- Right archive SHA-256: `{proof.right.archive_sha256}`",
            f"- Left raw bytes: {proof.left.output_raw_bytes}",
            f"- Right raw bytes: {proof.right.output_raw_bytes}",
            f"- Left raw SHA-256: `{proof.left.output_raw_sha256}`",
            f"- Right raw SHA-256: `{proof.right.output_raw_sha256}`",
            f"- Output bytes match: {str(proof.output_bytes_match).lower()}",
            f"- Output SHA-256 match: {str(proof.output_sha256_match).lower()}",
            f"- cmp equal: {str(proof.cmp_equal).lower()}",
            f"- Scratch retained: {str(proof.scratch_retained).lower()}",
            "- Score claim: false",
            "- Promotion eligible: false",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-archive", type=Path, required=True)
    parser.add_argument("--left-submission-dir", type=Path, required=True)
    parser.add_argument("--right-archive", type=Path, required=True)
    parser.add_argument("--right-submission-dir", type=Path, required=True)
    parser.add_argument("--file-list-entry", default="0.mkv")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / f"shell_inflate_parity_{_utc_stamp()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    proof = build_proof(
        left_archive=args.left_archive,
        left_submission_dir=args.left_submission_dir,
        right_archive=args.right_archive,
        right_submission_dir=args.right_submission_dir,
        output_dir=args.output_dir,
        file_list_entry=args.file_list_entry,
        python_bin=args.python_bin,
        keep_scratch=args.keep_scratch,
    )
    payload = json.dumps(proof.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "shell_inflate_parity.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "shell_inflate_parity.md").write_text(
        render_markdown(proof),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0 if proof.output_sha256_match and proof.cmp_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
