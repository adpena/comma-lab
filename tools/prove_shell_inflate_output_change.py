#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove a receiver/runtime patch changes full-frame shell inflate output.

This is the complement of ``prove_shell_inflate_parity.py``. It runs both
submission packets through the same shell inflate contract, keeps scratch just
long enough to compute byte-level output deltas, then writes a fail-closed proof
that the change is real but carries no score or promotion authority.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.scorer_region_waterfill import (  # noqa: E402
    selected_archive_from_scorer_region_chain_report,
)

try:
    from tools.prove_shell_inflate_parity import (
        SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE,
        SHELL_PARITY_SCOPE_DECLARED_FILE_LIST,
        ShellInflateParityProof,
        _expected_output_basename,
        _load_file_list_entries,
        _ordered_file_list_entries,
        build_proof,
    )
except ModuleNotFoundError:  # pragma: no cover
    from prove_shell_inflate_parity import (
        SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE,
        SHELL_PARITY_SCOPE_DECLARED_FILE_LIST,
        ShellInflateParityProof,
        _expected_output_basename,
        _load_file_list_entries,
        _ordered_file_list_entries,
        build_proof,
    )

SHELL_INFLATE_OUTPUT_CHANGE_PROOF_SCHEMA = "shell_inflate_output_change_proof_v1"


@dataclass(frozen=True)
class OutputDiffRow:
    file_list_entry: str
    output_basename: str
    left_bytes: int
    right_bytes: int
    bytes_match: bool
    sha256_match: bool
    differing_byte_count: int
    first_differing_offsets: tuple[int, ...]


@dataclass(frozen=True)
class ShellInflateOutputChangeProof:
    schema: str
    generated_at_utc: str
    parity_probe_schema: str
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
    contest_full_sample_change_claim: bool
    output_count: int
    output_bytes_match: bool
    output_sha256_match: bool
    output_manifest_sha256_match: bool
    cmp_equal: bool
    differing_output_count: int
    differing_byte_count: int
    output_change_observed: bool
    raw_shape_preserving_output_change_observed: bool
    full_frame_file_list_claim: bool
    full_frame_output_change_claim: bool
    output_diffs: tuple[OutputDiffRow, ...]
    blockers: tuple[str, ...]
    python_bin: str
    left: dict[str, Any]
    right: dict[str, Any]
    scratch_retained: bool
    scratch_dir: str
    parity_probe: dict[str, Any]
    score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    promotable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _validate_output_change_overwrite_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve(strict=False)
    protected = {
        Path.cwd().resolve(strict=False),
        Path.home().resolve(strict=False),
        Path("/").resolve(strict=False),
    }
    if resolved in protected:
        raise ValueError(f"refusing dangerous overwrite output dir: {output_dir}")
    markers = {
        "shell_inflate_output_change.json",
        "shell_inflate_output_change.md",
        "scratch",
    }
    if not any((output_dir / marker).exists() for marker in markers):
        raise ValueError(
            "refusing overwrite output dir without output-change marker: "
            f"{output_dir}"
        )


def _atomic_overwrite_backup_path(output_dir: Path) -> Path:
    parent = output_dir.parent
    base = f".{output_dir.name}.overwrite-backup-{_utc_stamp()}-{os.getpid()}"
    candidate = parent / base
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = parent / f"{base}-{suffix}"
    return candidate


def _begin_atomic_output_dir_overwrite(
    output_dir: Path,
    *,
    overwrite: bool,
) -> Path | None:
    if not overwrite or not output_dir.exists() or not any(output_dir.iterdir()):
        return None
    _validate_output_change_overwrite_dir(output_dir)
    backup = _atomic_overwrite_backup_path(output_dir)
    shutil.move(str(output_dir), str(backup))
    output_dir.mkdir(parents=True)
    return backup


def _restore_atomic_output_dir_overwrite(output_dir: Path, backup: Path | None) -> None:
    if backup is None:
        return
    if output_dir.exists():
        if output_dir.is_symlink():
            raise ValueError(f"refusing symlink output dir during restore: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()
    shutil.move(str(backup), str(output_dir))


def _finish_atomic_output_dir_overwrite(backup: Path | None) -> None:
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _count_differing_bytes(
    left: Path,
    right: Path,
    *,
    first_offsets_limit: int = 32,
    chunk_size: int = 8 << 20,
) -> tuple[int, tuple[int, ...]]:
    left_size = left.stat().st_size
    right_size = right.stat().st_size
    differing = 0
    first_offsets: list[int] = []
    offset = 0
    with left.open("rb") as left_fh, right.open("rb") as right_fh:
        while True:
            left_chunk = left_fh.read(chunk_size)
            right_chunk = right_fh.read(chunk_size)
            if not left_chunk and not right_chunk:
                break
            if left_chunk != right_chunk:
                common = min(len(left_chunk), len(right_chunk))
                for index in range(common):
                    if left_chunk[index] != right_chunk[index]:
                        differing += 1
                        if len(first_offsets) < first_offsets_limit:
                            first_offsets.append(offset + index)
                extra = abs(len(left_chunk) - len(right_chunk))
                if extra:
                    differing += extra
                    if len(first_offsets) < first_offsets_limit:
                        first_offsets.append(offset + common)
            offset += max(len(left_chunk), len(right_chunk))
    if left_size != right_size and differing < abs(left_size - right_size):
        differing += abs(left_size - right_size)
    return differing, tuple(first_offsets)


def _diff_outputs(
    *,
    parity: ShellInflateParityProof,
    scratch_dir: Path,
) -> tuple[OutputDiffRow, ...]:
    rows: list[OutputDiffRow] = []
    left_out = scratch_dir / "left_out"
    right_out = scratch_dir / "right_out"
    for left_row, right_row in zip(parity.left.outputs, parity.right.outputs, strict=True):
        if left_row.file_list_entry != right_row.file_list_entry:
            raise ValueError("left/right output manifest entry mismatch")
        basename = _expected_output_basename(left_row.file_list_entry)
        left_path = left_out / basename
        right_path = right_out / basename
        differing, first_offsets = _count_differing_bytes(left_path, right_path)
        rows.append(
            OutputDiffRow(
                file_list_entry=left_row.file_list_entry,
                output_basename=basename,
                left_bytes=left_row.output_raw_bytes,
                right_bytes=right_row.output_raw_bytes,
                bytes_match=left_row.output_raw_bytes == right_row.output_raw_bytes,
                sha256_match=left_row.output_raw_sha256 == right_row.output_raw_sha256,
                differing_byte_count=differing,
                first_differing_offsets=first_offsets,
            )
        )
    return tuple(rows)


def _build_change_proof(
    *,
    parity: ShellInflateParityProof,
    output_diffs: tuple[OutputDiffRow, ...],
    keep_scratch: bool,
) -> ShellInflateOutputChangeProof:
    differing_output_count = sum(1 for row in output_diffs if row.differing_byte_count > 0)
    differing_byte_count = sum(row.differing_byte_count for row in output_diffs)
    output_change_observed = differing_output_count > 0 and not parity.cmp_equal
    raw_shape_preserving = output_change_observed and parity.output_bytes_match

    blockers: list[str] = []
    if not parity.full_frame_file_list_claim:
        blockers.append("full_frame_file_list_claim_missing")
    if parity.full_frame_file_list_claim:
        if parity.expected_full_frame_file_list_sha256 is None:
            blockers.append("expected_full_frame_file_list_sha256_missing")
        elif parity.full_frame_file_list_sha256_match is not True:
            blockers.append("expected_full_frame_file_list_sha256_mismatch")
        if parity.expected_full_frame_entry_count is None:
            blockers.append("expected_full_frame_entry_count_missing")
        elif parity.full_frame_entry_count_match is not True:
            blockers.append("expected_full_frame_entry_count_mismatch")
        if parity.full_frame_file_list_source is None:
            blockers.append("full_frame_file_list_source_missing")
    if (
        parity.contest_full_sample_claim
        and parity.parity_scope_kind != SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE
    ):
        blockers.append("contest_full_sample_scope_kind_mismatch")
    if not output_change_observed:
        blockers.append("shell_inflate_output_change_not_observed")
    if not raw_shape_preserving:
        blockers.append("raw_shape_preserving_output_change_not_observed")

    full_frame_output_change_claim = (
        raw_shape_preserving
        and parity.full_frame_file_list_claim
        and parity.full_frame_file_list_sha256_match is not False
        and parity.full_frame_entry_count_match is not False
        and parity.full_frame_file_list_source is not None
        and not blockers
    )
    contest_full_sample_change_claim = (
        full_frame_output_change_claim
        and parity.contest_full_sample_claim
        and parity.parity_scope_kind == SHELL_PARITY_SCOPE_CONTEST_FULL_SAMPLE
    )
    return ShellInflateOutputChangeProof(
        schema=SHELL_INFLATE_OUTPUT_CHANGE_PROOF_SCHEMA,
        generated_at_utc=_utc_iso(),
        parity_probe_schema=parity.schema,
        file_list_entries=parity.file_list_entries,
        file_list_entry_count=parity.file_list_entry_count,
        file_list_sha256=parity.file_list_sha256,
        full_frame_file_list_source=parity.full_frame_file_list_source,
        expected_full_frame_file_list_sha256=parity.expected_full_frame_file_list_sha256,
        expected_full_frame_entry_count=parity.expected_full_frame_entry_count,
        full_frame_file_list_sha256_match=parity.full_frame_file_list_sha256_match,
        full_frame_entry_count_match=parity.full_frame_entry_count_match,
        parity_scope_kind=parity.parity_scope_kind,
        contest_full_sample_claim=parity.contest_full_sample_claim,
        contest_full_sample_change_claim=contest_full_sample_change_claim,
        output_count=parity.output_count,
        output_bytes_match=parity.output_bytes_match,
        output_sha256_match=parity.output_sha256_match,
        output_manifest_sha256_match=parity.output_manifest_sha256_match,
        cmp_equal=parity.cmp_equal,
        differing_output_count=differing_output_count,
        differing_byte_count=differing_byte_count,
        output_change_observed=output_change_observed,
        raw_shape_preserving_output_change_observed=raw_shape_preserving,
        full_frame_file_list_claim=parity.full_frame_file_list_claim,
        full_frame_output_change_claim=full_frame_output_change_claim,
        output_diffs=output_diffs,
        blockers=tuple(dict.fromkeys(blockers)),
        python_bin=parity.python_bin,
        left=asdict(parity.left),
        right=asdict(parity.right),
        scratch_retained=keep_scratch,
        scratch_dir=parity.scratch_dir,
        parity_probe=parity.to_dict(),
    )


def render_markdown(proof: ShellInflateOutputChangeProof) -> str:
    return "\n".join(
        [
            "# Shell Inflate Output-Change Proof",
            "",
            f"- Generated UTC: {proof.generated_at_utc}",
            f"- File list entries: {proof.file_list_entry_count}",
            f"- File list SHA-256: `{proof.file_list_sha256}`",
            f"- Full-frame file-list source: `{proof.full_frame_file_list_source}`",
            f"- Parity scope kind: `{proof.parity_scope_kind}`",
            f"- Contest full-sample claim: {str(proof.contest_full_sample_claim).lower()}",
            f"- Contest full-sample change claim: {str(proof.contest_full_sample_change_claim).lower()}",
            f"- Output bytes match: {str(proof.output_bytes_match).lower()}",
            f"- Output SHA-256 match: {str(proof.output_sha256_match).lower()}",
            f"- cmp equal: {str(proof.cmp_equal).lower()}",
            f"- Differing outputs: {proof.differing_output_count}",
            f"- Differing bytes: {proof.differing_byte_count}",
            f"- Output change observed: {str(proof.output_change_observed).lower()}",
            f"- Shape-preserving change observed: {str(proof.raw_shape_preserving_output_change_observed).lower()}",
            f"- Full-frame output-change claim: {str(proof.full_frame_output_change_claim).lower()}",
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
    left_archive_group = parser.add_mutually_exclusive_group(required=True)
    left_archive_group.add_argument("--left-archive", type=Path)
    left_archive_group.add_argument(
        "--left-selected-archive-chain-report",
        type=Path,
        help=(
            "Use selected_local_survivor_archive.path from a scorer-region "
            "chain report as the left archive. This keeps output-change proofs "
            "aligned with the selected survivor instead of a fixed intermediate."
        ),
    )
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
    )
    parser.add_argument("--contest-full-sample-claim", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--require-output-change", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results") / "shell_inflate_output_change",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    left_archive = args.left_archive
    if args.left_selected_archive_chain_report is not None:
        left_archive, _left_archive_source = selected_archive_from_scorer_region_chain_report(
            args.left_selected_archive_chain_report,
            repo_root=REPO_ROOT,
            context="shell_inflate_output_change_left_selected_archive_chain_report",
        )
    file_list_entries = (
        _load_file_list_entries(args.file_list)
        if args.file_list is not None
        else _ordered_file_list_entries(args.file_list_entry or ["0.mkv"])
    )
    backup = _begin_atomic_output_dir_overwrite(
        args.output_dir,
        overwrite=args.overwrite,
    )
    try:
        parity = build_proof(
            left_archive=left_archive,
            left_submission_dir=args.left_submission_dir,
            right_archive=args.right_archive,
            right_submission_dir=args.right_submission_dir,
            output_dir=args.output_dir,
            file_list_entries=file_list_entries,
            python_bin=args.python_bin,
            keep_scratch=True,
            full_frame_file_list_claim=args.full_frame_file_list_claim,
            expected_full_frame_file_list_sha256=(
                args.expected_full_frame_file_list_sha256
            ),
            expected_full_frame_entry_count=args.expected_full_frame_entry_count,
            full_frame_file_list_source=args.full_frame_file_list_source,
            parity_scope_kind=args.parity_scope_kind,
            contest_full_sample_claim=args.contest_full_sample_claim,
            overwrite=False,
        )
        scratch = args.output_dir / "scratch"
        output_diffs = _diff_outputs(parity=parity, scratch_dir=scratch)
        proof = _build_change_proof(
            parity=parity,
            output_diffs=output_diffs,
            keep_scratch=args.keep_scratch,
        )
        if not args.keep_scratch and scratch.exists() and not scratch.is_symlink():
            shutil.rmtree(scratch)
        payload = json.dumps(proof.to_dict(), indent=2, sort_keys=True) + "\n"
        (args.output_dir / "shell_inflate_output_change.json").write_text(
            payload,
            encoding="utf-8",
        )
        (args.output_dir / "shell_inflate_output_change.md").write_text(
            render_markdown(proof),
            encoding="utf-8",
        )
    except Exception:
        _restore_atomic_output_dir_overwrite(args.output_dir, backup)
        raise
    else:
        _finish_atomic_output_dir_overwrite(backup)
    sys.stdout.write(payload)
    if args.require_output_change:
        return 0 if proof.full_frame_output_change_claim else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
