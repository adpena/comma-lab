# pyc-recovery: human-reconstructed from src/tac/archive_byte_profile.py.pyc
# This is the canonical main-repo content as of 2026-05-05.
# Recovery spec preserved at: archive_byte_profile.recovery_spec.json
# Original STUB has been replaced with this canonical version.
"""Deterministic ZIP byte attribution for contest archive research.

The profiler is intentionally byte-only: it does not extract archive payloads,
inflate contest outputs, load scorer models, or make score claims.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/archive_byte_profile.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: ``contest_rate_term`` (the only symbol consumed by other
``tac.*`` modules in this codebase) is reconstructed exactly. Other helpers
that produce CLI markdown / multi-archive profiles are stubbed to
``NotImplementedError`` because the bytecode disassembly contains intricate
nested generator expressions that pycdc cannot fully decompile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

SCHEMA = "archive_byte_profile_collection_v1"
ARCHIVE_SCHEMA = "archive_byte_profile_v1"
TOOL = "experiments/profile_archive_bytes.py"
EVIDENCE_GRADE = "byte_profile_only"
CONTEST_ORIGINAL_BYTES = 37545489
RATE_TERM_COEFFICIENT = 25 / CONTEST_ORIGINAL_BYTES

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/archive_byte_profile.recovery_spec.json"
)


class ArchiveByteProfileError(ValueError):
    """Raised when an archive cannot be safely profiled."""

    pass


def contest_rate_term(byte_count: int) -> float:
    """Return the contest formula rate contribution for ``byte_count``."""
    return 25 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_zip_member_name(name: str) -> str:
    if not name:
        raise ArchiveByteProfileError("archive member name is empty")
    if "\x00" in name:
        raise ArchiveByteProfileError(
            f"archive member contains NUL byte: {name!r}"
        )
    if "\\" in name:
        raise ArchiveByteProfileError(
            f"archive member uses backslashes: {name!r}"
        )
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")
    parts = posix_path.parts
    if not parts or any(p == ".." for p in parts):
        raise ArchiveByteProfileError(f"zip-slip archive member path: {name!r}")
    return name


def _zip_method_name(compress_type: int) -> str:
    names = {
        zipfile.ZIP_DEFLATED: "deflated",
        zipfile.ZIP_STORED: "stored",
    }
    if hasattr(zipfile, "ZIP_BZIP2"):
        names[zipfile.ZIP_BZIP2] = "bzip2"
    if hasattr(zipfile, "ZIP_LZMA"):
        names[zipfile.ZIP_LZMA] = "lzma"
    return names.get(compress_type, f"unknown_{compress_type}")


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate nested generator "
        f"expressions that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


def profile_archive(path: Path | str) -> dict[str, Any]:
    """Profile one ZIP archive without extracting it."""
    raise _rehydration_failure("profile_archive")


def invalid_archive_record(path: Path | str, error: str) -> dict[str, Any]:
    """Return a structured byte-only record for an archive that failed profiling."""
    raise _rehydration_failure("invalid_archive_record")


def build_profile_collection(
    paths: Iterable[Path | str], *, continue_on_error: bool = False
) -> dict[str, Any]:
    raise _rehydration_failure("build_profile_collection")


def render_markdown(profile: dict[str, Any]) -> str:
    raise _rehydration_failure("render_markdown")


def write_outputs(
    profile: dict[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> None:
    raise _rehydration_failure("write_outputs")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile ZIP archive byte attribution without extracting payloads."
    )
    parser.add_argument(
        "archives", nargs="+", type=Path, help="archive.zip paths to profile"
    )
    parser.add_argument(
        "--json-out", type=Path, help="write deterministic JSON profile"
    )
    parser.add_argument(
        "--markdown-out", type=Path, help="write markdown summary"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record invalid/nonstandard archives instead of aborting the whole collection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    profile = build_profile_collection(
        args.archives, continue_on_error=args.continue_on_error
    )
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
