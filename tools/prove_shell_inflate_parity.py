#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove two submission packets match at the shell-level inflate contract.

This runs each packet's ``inflate.sh archive_dir output_dir file_list`` entry
point against extracted archive contents, hashes the emitted raw files, compares
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

SHELL_PARITY_SCOPE_DECLARED_FILE_LIST = "declared_file_list"
SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE = "contest_full_sample"


@dataclass(frozen=True)
class ShellInflateOutput:
    file_list_entry: str
    output_basename: str
    output_raw_bytes: int
    output_raw_sha256: str


@dataclass(frozen=True)
class ShellInflateSide:
    label: str
    archive: str
    archive_bytes: int
    archive_sha256: str
    submission_dir: str
    submission_tree_file_count: int
    submission_tree_sha256: str
    inflate_sh: str
    inflate_sh_sha256: str
    output_count: int
    output_manifest_sha256: str
    output_raw_bytes: int
    output_raw_sha256: str
    outputs: tuple[ShellInflateOutput, ...]
    inflate_seconds: float


@dataclass(frozen=True)
class ShellInflateParityProof:
    schema: str
    generated_at_utc: str
    file_list_entries: tuple[str, ...]
    file_list_entry_count: int
    file_list_sha256: str
    full_frame_file_list_source: str | None
    expected_full_frame_file_list_sha256: str | None
    expected_full_frame_entry_count: int | None
    full_frame_file_list_sha256_match: bool | None
    full_frame_entry_count_match: bool | None
    parity_scope_kind: str
    contest_full_sample_claim: bool
    contest_full_sample_parity_claim: bool
    output_count: int
    file_list_entry: str | None
    output_basename: str | None
    python_bin: str
    left: ShellInflateSide
    right: ShellInflateSide
    output_bytes_match: bool
    output_sha256_match: bool
    output_manifest_sha256_match: bool
    cmp_equal: bool
    full_frame_file_list_claim: bool
    full_frame_inflate_output_parity_claim: bool
    blockers: tuple[str, ...]
    scratch_retained: bool
    scratch_dir: str
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    promotable: bool = False

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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    return None


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


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


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"refusing symlink output dir: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(f"refusing non-empty output dir: {output_dir}")
    else:
        output_dir.mkdir(parents=True)


def _submission_tree_record(path: Path) -> tuple[int, str]:
    records: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        rel = child.relative_to(path).as_posix()
        if child.is_symlink():
            records.append(
                {
                    "path": rel,
                    "kind": "symlink",
                    "target": os.readlink(child),
                }
            )
        elif child.is_file():
            records.append(
                {
                    "path": rel,
                    "kind": "file",
                    "bytes": child.stat().st_size,
                    "sha256": _sha256_file(child),
                }
            )
    return len(records), _canonical_json_sha256(records)


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


def _expected_output_basename(file_list_entry: str) -> str:
    return f"{Path(file_list_entry).stem}.raw"


def _ordered_file_list_entries(entries: list[str]) -> tuple[str, ...]:
    out = tuple(entry.strip() for entry in entries if entry.strip())
    if not out:
        raise ValueError("file list must contain at least one non-empty entry")
    basenames = [_expected_output_basename(entry) for entry in out]
    if len(set(basenames)) != len(basenames):
        raise ValueError(
            "file list entries produce duplicate raw output basenames: "
            + ", ".join(basenames)
        )
    return out


def _load_file_list_entries(path: Path) -> tuple[str, ...]:
    return _ordered_file_list_entries(path.read_text(encoding="utf-8").splitlines())


def _output_manifest_sha256(outputs: tuple[ShellInflateOutput, ...]) -> str:
    return _canonical_json_sha256([asdict(output) for output in outputs])


def _run_inflate(
    *,
    label: str,
    archive: Path,
    submission_dir: Path,
    file_list: Path,
    output_dir: Path,
    data_dir: Path,
    python_bin: str,
    file_list_entries: tuple[str, ...],
) -> ShellInflateSide:
    _safe_extract_zip(archive, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inflate_sh = submission_dir / "inflate.sh"
    if not inflate_sh.is_file():
        raise FileNotFoundError(f"missing inflate.sh: {inflate_sh}")
    env = os.environ.copy()
    env["PACT_PYTHON_BIN"] = python_bin
    python_path = Path(python_bin)
    if python_path.is_absolute() or python_path.parent != Path("."):
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        # Preserve venv shims/symlinks: resolving the executable can jump out of
        # the environment that owns the installed packages.
        python_dir = str(python_path.parent.resolve())
        env["PATH"] = f"{python_dir}{os.pathsep}{env.get('PATH', '')}"
    start = time.perf_counter()
    data_arg = str(data_dir.resolve())
    output_arg = str(output_dir.resolve())
    file_list_arg = str(file_list.resolve())
    inflate_arg = str(inflate_sh.resolve())
    command = (
        [inflate_arg, data_arg, output_arg, file_list_arg]
        if os.access(inflate_sh, os.X_OK)
        else ["bash", inflate_arg, data_arg, output_arg, file_list_arg]
    )
    subprocess.run(
        command,
        check=True,
        env=env,
    )
    elapsed = time.perf_counter() - start
    outputs: list[ShellInflateOutput] = []
    for entry in file_list_entries:
        output_basename = _expected_output_basename(entry)
        raw_path = output_dir / output_basename
        if not raw_path.is_file():
            raise FileNotFoundError(
                f"inflate did not write expected raw output: {raw_path}"
            )
        outputs.append(
            ShellInflateOutput(
                file_list_entry=entry,
                output_basename=output_basename,
                output_raw_bytes=raw_path.stat().st_size,
                output_raw_sha256=_sha256_file(raw_path),
            )
        )
    output_rows = tuple(outputs)
    output_raw_bytes = sum(row.output_raw_bytes for row in output_rows)
    output_raw_sha256 = (
        output_rows[0].output_raw_sha256
        if len(output_rows) == 1
        else _output_manifest_sha256(output_rows)
    )
    tree_file_count, tree_sha256 = _submission_tree_record(submission_dir)
    return ShellInflateSide(
        label=label,
        archive=_repo_rel(archive),
        archive_bytes=archive.stat().st_size,
        archive_sha256=_sha256_file(archive),
        submission_dir=_repo_rel(submission_dir),
        submission_tree_file_count=tree_file_count,
        submission_tree_sha256=tree_sha256,
        inflate_sh=_repo_rel(inflate_sh),
        inflate_sh_sha256=_sha256_file(inflate_sh),
        output_count=len(output_rows),
        output_manifest_sha256=_output_manifest_sha256(output_rows),
        output_raw_bytes=output_raw_bytes,
        output_raw_sha256=output_raw_sha256,
        outputs=output_rows,
        inflate_seconds=elapsed,
    )


def build_proof(
    *,
    left_archive: Path,
    left_submission_dir: Path,
    right_archive: Path,
    right_submission_dir: Path,
    output_dir: Path,
    python_bin: str,
    keep_scratch: bool,
    file_list_entries: tuple[str, ...] = (),
    file_list_entry: str | None = None,
    full_frame_file_list_claim: bool = False,
    expected_full_frame_file_list_sha256: str | None = None,
    expected_full_frame_entry_count: int | None = None,
    full_frame_file_list_source: str | None = None,
    parity_scope_kind: str = SHELL_PARITY_SCOPE_DECLARED_FILE_LIST,
    contest_full_sample_claim: bool = False,
) -> ShellInflateParityProof:
    _prepare_output_dir(output_dir)
    if not file_list_entries:
        file_list_entries = _ordered_file_list_entries([file_list_entry or "0.mkv"])
    else:
        file_list_entries = _ordered_file_list_entries(list(file_list_entries))
    file_list = output_dir / "file_list.txt"
    file_list.write_text("\n".join(file_list_entries) + "\n", encoding="utf-8")
    file_list_sha256 = _sha256_file(file_list)
    expected_file_list_sha256 = _canonical_sha256(
        expected_full_frame_file_list_sha256
    )
    file_list_source = str(full_frame_file_list_source or "").strip() or None
    scope_kind = (
        str(parity_scope_kind or "").strip() or SHELL_PARITY_SCOPE_DECLARED_FILE_LIST
    )
    file_list_sha256_match = (
        None
        if expected_file_list_sha256 is None
        else file_list_sha256 == expected_file_list_sha256
    )
    entry_count_match = (
        None
        if expected_full_frame_entry_count is None
        else len(file_list_entries) == expected_full_frame_entry_count
    )
    output_basename = (
        _expected_output_basename(file_list_entries[0])
        if len(file_list_entries) == 1
        else None
    )
    scratch = output_dir / "scratch"
    left_data = scratch / "left_data"
    right_data = scratch / "right_data"
    left_out = scratch / "left_out"
    right_out = scratch / "right_out"
    try:
        left = _run_inflate(
            label="left",
            archive=left_archive,
            submission_dir=left_submission_dir,
            file_list=file_list,
            output_dir=left_out,
            data_dir=left_data,
            python_bin=python_bin,
            file_list_entries=file_list_entries,
        )
        right = _run_inflate(
            label="right",
            archive=right_archive,
            submission_dir=right_submission_dir,
            file_list=file_list,
            output_dir=right_out,
            data_dir=right_data,
            python_bin=python_bin,
            file_list_entries=file_list_entries,
        )
        cmp_results = [
            _files_equal(
                left_out / _expected_output_basename(entry),
                right_out / _expected_output_basename(entry),
            )
            for entry in file_list_entries
        ]
        cmp_equal = all(cmp_results)
        output_bytes_match = all(
            left_output.output_raw_bytes == right_output.output_raw_bytes
            for left_output, right_output in zip(left.outputs, right.outputs, strict=True)
        )
        output_sha256_match = all(
            left_output.output_raw_sha256 == right_output.output_raw_sha256
            for left_output, right_output in zip(left.outputs, right.outputs, strict=True)
        )
        output_manifest_sha256_match = (
            left.output_manifest_sha256 == right.output_manifest_sha256
        )
        blockers: list[str] = []
        if not full_frame_file_list_claim:
            blockers.append("full_frame_file_list_claim_missing")
        if full_frame_file_list_claim:
            if expected_file_list_sha256 is None:
                blockers.append("expected_full_frame_file_list_sha256_missing")
            elif not file_list_sha256_match:
                blockers.append("expected_full_frame_file_list_sha256_mismatch")
            if expected_full_frame_entry_count is None:
                blockers.append("expected_full_frame_entry_count_missing")
            elif expected_full_frame_entry_count < 1:
                blockers.append("expected_full_frame_entry_count_invalid")
            elif not entry_count_match:
                blockers.append("expected_full_frame_entry_count_mismatch")
            if file_list_source is None:
                blockers.append("full_frame_file_list_source_missing")
        if (
            contest_full_sample_claim
            and scope_kind != SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE
        ):
            blockers.append("contest_full_sample_scope_kind_mismatch")
        if not (
            output_bytes_match
            and output_sha256_match
            and output_manifest_sha256_match
            and cmp_equal
        ):
            blockers.append("shell_inflate_output_parity_failed")
        full_frame_claim = full_frame_file_list_claim and not blockers
        contest_full_sample_parity_claim = (
            full_frame_claim
            and contest_full_sample_claim
            and scope_kind == SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE
        )
        return ShellInflateParityProof(
            schema="shell_inflate_parity_proof_v2",
            generated_at_utc=_utc_iso(),
            file_list_entries=file_list_entries,
            file_list_entry_count=len(file_list_entries),
            file_list_sha256=file_list_sha256,
            full_frame_file_list_source=file_list_source,
            expected_full_frame_file_list_sha256=expected_file_list_sha256,
            expected_full_frame_entry_count=expected_full_frame_entry_count,
            full_frame_file_list_sha256_match=file_list_sha256_match,
            full_frame_entry_count_match=entry_count_match,
            parity_scope_kind=scope_kind,
            contest_full_sample_claim=contest_full_sample_claim,
            contest_full_sample_parity_claim=contest_full_sample_parity_claim,
            output_count=len(file_list_entries),
            file_list_entry=file_list_entries[0] if len(file_list_entries) == 1 else None,
            output_basename=output_basename,
            python_bin=python_bin,
            left=left,
            right=right,
            output_bytes_match=output_bytes_match,
            output_sha256_match=output_sha256_match,
            output_manifest_sha256_match=output_manifest_sha256_match,
            cmp_equal=cmp_equal,
            full_frame_file_list_claim=full_frame_file_list_claim,
            full_frame_inflate_output_parity_claim=full_frame_claim,
            blockers=tuple(blockers),
            scratch_retained=keep_scratch,
            scratch_dir=_repo_rel(scratch),
        )
    finally:
        if not keep_scratch and scratch.exists() and not scratch.is_symlink():
            shutil.rmtree(scratch)


def render_markdown(proof: ShellInflateParityProof) -> str:
    return "\n".join(
        [
            "# Shell Inflate Parity Proof",
            "",
            f"- Generated UTC: {proof.generated_at_utc}",
            f"- File list entries: {proof.file_list_entry_count}",
            f"- File list SHA-256: `{proof.file_list_sha256}`",
            f"- Full-frame file-list source: `{proof.full_frame_file_list_source}`",
            f"- Expected full-frame file-list SHA-256: `{proof.expected_full_frame_file_list_sha256}`",
            f"- Expected full-frame entry count: {proof.expected_full_frame_entry_count}",
            f"- Full-frame file-list SHA-256 match: {str(proof.full_frame_file_list_sha256_match).lower()}",
            f"- Full-frame entry count match: {str(proof.full_frame_entry_count_match).lower()}",
            f"- Parity scope kind: `{proof.parity_scope_kind}`",
            f"- Contest full-sample claim: {str(proof.contest_full_sample_claim).lower()}",
            f"- Contest full-sample parity claim: {str(proof.contest_full_sample_parity_claim).lower()}",
            f"- Full-frame file-list claim: {str(proof.full_frame_file_list_claim).lower()}",
            f"- Full-frame inflate parity claim: {str(proof.full_frame_inflate_output_parity_claim).lower()}",
            f"- Output count: {proof.output_count}",
            f"- Single output raw basename: `{proof.output_basename}`",
            f"- Python bin: `{proof.python_bin}`",
            f"- Left archive: `{proof.left.archive}`",
            f"- Right archive: `{proof.right.archive}`",
            f"- Left archive SHA-256: `{proof.left.archive_sha256}`",
            f"- Right archive SHA-256: `{proof.right.archive_sha256}`",
            f"- Left output bytes: {proof.left.output_raw_bytes}",
            f"- Right output bytes: {proof.right.output_raw_bytes}",
            f"- Left output manifest SHA-256: `{proof.left.output_manifest_sha256}`",
            f"- Right output manifest SHA-256: `{proof.right.output_manifest_sha256}`",
            f"- Output bytes match: {str(proof.output_bytes_match).lower()}",
            f"- Output SHA-256 match: {str(proof.output_sha256_match).lower()}",
            f"- Output manifest SHA-256 match: {str(proof.output_manifest_sha256_match).lower()}",
            f"- cmp equal: {str(proof.cmp_equal).lower()}",
            f"- Blockers: {', '.join(proof.blockers) if proof.blockers else 'none'}",
            f"- Scratch retained: {str(proof.scratch_retained).lower()}",
            "- Score claim: false",
            "- Promotion eligible: false",
            "- Rank/kill eligible: false",
            "- Ready for exact eval dispatch: false",
            "- Promotable: false",
            "",
        ]
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-archive", type=Path, required=True)
    parser.add_argument("--left-submission-dir", type=Path, required=True)
    parser.add_argument("--right-archive", type=Path, required=True)
    parser.add_argument("--right-submission-dir", type=Path, required=True)
    file_list_group = parser.add_mutually_exclusive_group()
    file_list_group.add_argument("--file-list", type=Path)
    file_list_group.add_argument("--file-list-entry", action="append")
    parser.add_argument("--full-frame-file-list-claim", action="store_true")
    parser.add_argument("--expected-full-frame-file-list-sha256")
    parser.add_argument("--expected-full-frame-entry-count", type=int)
    parser.add_argument("--full-frame-file-list-source")
    parser.add_argument(
        "--parity-scope-kind",
        default=SHELL_PARITY_SCOPE_DECLARED_FILE_LIST,
        help=(
            "Semantic scope of the provided file list. Use 'contest_full_sample' "
            "only when the file list covers the complete contest sample."
        ),
    )
    parser.add_argument(
        "--contest-full-sample-claim",
        action="store_true",
        help="Require the proof to be interpreted as complete contest-sample parity.",
    )
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
    if args.file_list is not None:
        file_list_entries = _load_file_list_entries(args.file_list)
    else:
        file_list_entries = _ordered_file_list_entries(args.file_list_entry or ["0.mkv"])
    proof = build_proof(
        left_archive=args.left_archive,
        left_submission_dir=args.left_submission_dir,
        right_archive=args.right_archive,
        right_submission_dir=args.right_submission_dir,
        output_dir=args.output_dir,
        file_list_entries=file_list_entries,
        python_bin=args.python_bin,
        keep_scratch=args.keep_scratch,
        full_frame_file_list_claim=args.full_frame_file_list_claim,
        expected_full_frame_file_list_sha256=(
            args.expected_full_frame_file_list_sha256
        ),
        expected_full_frame_entry_count=args.expected_full_frame_entry_count,
        full_frame_file_list_source=args.full_frame_file_list_source,
        parity_scope_kind=args.parity_scope_kind,
        contest_full_sample_claim=args.contest_full_sample_claim,
    )
    payload = json.dumps(proof.to_dict(), indent=2, sort_keys=True) + "\n"
    (args.output_dir / "shell_inflate_parity.json").write_text(payload, encoding="utf-8")
    (args.output_dir / "shell_inflate_parity.md").write_text(
        render_markdown(proof),
        encoding="utf-8",
    )
    sys.stdout.write(payload)
    return 0 if proof.full_frame_inflate_output_parity_claim else 1


if __name__ == "__main__":
    raise SystemExit(main())
