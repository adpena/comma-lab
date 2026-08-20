#!/usr/bin/env python3
"""Materialize terminal G14 -> canonical G18/G19 research-only receipts.

This CLI has no scorer, evaluator, dispatcher, candidate, or pointer mutation
surface.  It reads only explicitly named immutable inputs and writes one
atomic, write-once/equal JSON receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Final

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for import_root in (REPO, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tac.witness_control.taskspace_feedback_costate_materializer_v1 import (  # noqa: E402
    TaskspaceFeedbackCostateMaterializerError,
    materialization_receipt_bytes,
    materialize_taskspace_feedback_costate_v1,
)


def _stable_read_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Exclude atime, which the read itself may lawfully mutate."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TaskspaceFeedbackCostateMaterializerError(f"cannot open {label} as non-symlink regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        stable_identity = _stable_read_identity(metadata)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
            raise TaskspaceFeedbackCostateMaterializerError(f"{label} must be a nonempty regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise TaskspaceFeedbackCostateMaterializerError(f"{label} short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if _stable_read_identity(os.fstat(descriptor)) != stable_identity:
            raise TaskspaceFeedbackCostateMaterializerError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular(path, label="existing output") != payload:
            raise TaskspaceFeedbackCostateMaterializerError("output already exists with different immutable bytes")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_regular(path, label="raced output") != payload:
                raise TaskspaceFeedbackCostateMaterializerError(
                    "concurrent output differs from materialized receipt"
                ) from None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g14-final-receipt", type=Path, required=True)
    parser.add_argument("--pair-population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-sha256-hint")
    parser.add_argument("--g20-placement-receipt", type=Path)
    parser.add_argument("--g22-receiver-equality-receipt", type=Path)
    parser.add_argument("--g25-population-global-recode-receipt", type=Path)
    parser.add_argument("--g28-same-object-score-receipt", type=Path)
    parser.add_argument("--prior-controller-state", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prior = None
        if args.prior_controller_state is not None:
            prior_raw = _read_regular(args.prior_controller_state, label="prior controller state")
            prior = json.loads(prior_raw)
            if type(prior) is not dict:
                raise TaskspaceFeedbackCostateMaterializerError("prior controller state root must be one object")
        result = materialize_taskspace_feedback_costate_v1(
            _read_regular(args.g14_final_receipt, label="G14 final_receipt"),
            _read_regular(args.pair_population, label="PairPopulation"),
            population_sha256_hint=args.population_sha256_hint,
            prior_controller_state=prior,
            g20_placement_receipt=(
                None
                if args.g20_placement_receipt is None
                else _read_regular(args.g20_placement_receipt, label="G20 placement receipt")
            ),
            g22_receiver_equality_receipt=(
                None
                if args.g22_receiver_equality_receipt is None
                else _read_regular(
                    args.g22_receiver_equality_receipt,
                    label="G22 receiver equality receipt",
                )
            ),
            g25_population_global_recode_receipt=(
                None
                if args.g25_population_global_recode_receipt is None
                else _read_regular(
                    args.g25_population_global_recode_receipt,
                    label="G25 population-global recode receipt",
                )
            ),
            g28_same_object_score_receipt=(
                None
                if args.g28_same_object_score_receipt is None
                else _read_regular(
                    args.g28_same_object_score_receipt,
                    label="G28 same-object score receipt",
                )
            ),
        )
        payload = materialization_receipt_bytes(result)
        _write_once_or_equal(args.output, payload)
    except (TaskspaceFeedbackCostateMaterializerError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "bytes": len(payload)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
