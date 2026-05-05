# pyc-recovery: human-reconstructed from src/tac/endgame_archive_decision.py.pyc
# This is the canonical main-repo content as of 2026-05-05.
# Recovery spec preserved at: endgame_archive_decision.recovery_spec.json
# Original STUB has been replaced with this canonical version.
"""Byte-level endgame decision support for PR85-family archives.

This module is local and deterministic. It validates ZIP custody, slices
PR85-family single-member bundles, probes cheap codec contracts, and estimates
rate-only transplant deltas against a named frontier archive. It never inflates
contest videos, loads scorers, dispatches jobs, or makes score claims.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/endgame_archive_decision.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: Module-level constants, error class, helpers, and the
``main`` CLI entry point are reconstructed. Many internal probes
(``_brotli_probe``, ``_qma9_probe``, ``_hpm1_probe``, ``_stbm1br_probe``,
``_parse_rmb1``, ``_parse_rsb1``, ``_profile_one_archive``, ``_comparison``,
``build_endgame_decision_profile``, etc.) are stubbed to ``NotImplementedError``
because the bytecode disassembly contains many nested closures, lambdas, and
generator expressions that pycdc cannot fully decompile. Public CLI entry
``main`` will fail loud at the first deferred internal call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from tac.archive_byte_profile import contest_rate_term
from tac.pr85_bundle import (
    HPM1_MAGIC,
    PR85_HEADERLESS_RANDMULTI_SPECS,
    Pr85BundleError,
    SEGMENT_ORDER,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
)

SCHEMA = "endgame_archive_decision_profile_v1"
TOOL = "experiments/profile_endgame_archive_decision.py"
EVIDENCE_GRADE = "byte_level_decision_support_only"
LOCAL_FILE_HEADER = 0x04034B50  # 67324752
QMA9_HEADER_BYTES = 20
STBM1BR_MAGIC = b"STBM1BR\x00"
RMB1_MAGIC = b"RMB1"
RSB1_MAGIC = b"RSB1"

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/endgame_archive_decision.recovery_spec.json"
)


class EndgameArchiveDecisionError(ValueError):
    """Raised when endgame archive profiling inputs are malformed."""

    pass


@dataclass(frozen=True)
class _MemberPayload:
    """ZIP member byte payload and structural metadata."""

    name: str
    raw_name: bytes
    bytes_compressed: int
    bytes_uncompressed: int
    crc32: int
    compress_type: int
    payload: bytes
    sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ArchiveProfile:
    """Profiled archive byte attribution."""

    path: str
    total_bytes: int
    sha256: str
    member_payloads: Sequence[_MemberPayload]
    extra: Mapping[str, Any] = field(default_factory=dict)


def _json_text(payload: Any) -> str:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _byte_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )


def _display_ascii(data: bytes, *, limit: int = 16) -> str:
    chars = []
    for b in data[:limit]:
        if 32 <= b < 127:
            chars.append(chr(b))
        else:
            chars.append(".")
    return "".join(chars)


def _safe_member_blockers(name: str) -> list[str]:
    blockers: list[str] = []
    if not name:
        blockers.append("empty_name")
        return blockers
    if "\x00" in name:
        blockers.append("nul_byte")
    if "\\" in name:
        blockers.append("backslash")
    posix_path = PurePosixPath(name)
    if posix_path.is_absolute():
        blockers.append("absolute_posix")
    windows_path = PureWindowsPath(name)
    if windows_path.is_absolute() or windows_path.drive:
        blockers.append("absolute_windows")
    if any(p == ".." for p in posix_path.parts):
        blockers.append("dotdot")
    return blockers


def _decode_zip_name(raw: bytes, flag_bits: int) -> str:
    if flag_bits & 0x800:
        return raw.decode("utf-8")
    return raw.decode("cp437")


def _local_header_name(buf: bytes, offset: int) -> bytes:
    if offset + 30 > len(buf):
        raise EndgameArchiveDecisionError(
            f"local header offset {offset} exceeds buffer length {len(buf)}"
        )
    sig = int.from_bytes(buf[offset : offset + 4], "little")
    if sig != LOCAL_FILE_HEADER:
        raise EndgameArchiveDecisionError(
            f"local header at {offset} has wrong signature: {sig:#x}"
        )
    name_len = int.from_bytes(buf[offset + 26 : offset + 28], "little")
    name_start = offset + 30
    return buf[name_start : name_start + name_len]


def _import_brotli():
    try:
        import brotli  # noqa: F401

        return brotli
    except ImportError as exc:  # pragma: no cover - env dependent
        raise EndgameArchiveDecisionError(
            "endgame archive profiling requires the brotli package"
        ) from exc


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate nested closures "
        f"and generators that pycdc cannot fully decompile; original bytecode "
        f"preserved in {_QUARANTINE_SPEC}"
    )


# --- Probes (deferred) ---


def _brotli_probe(data: bytes, *, name: str = "", limit: int = 0) -> dict[str, Any]:
    raise _rehydration_failure("_brotli_probe")


def _qma9_probe(data: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_qma9_probe")


def _hpm1_probe(data: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_hpm1_probe")


def _stbm1br_probe(data: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_stbm1br_probe")


def _parse_rmb1(data: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_parse_rmb1")


def _parse_rsb1(data: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_parse_rsb1")


def _codec_label(name: str, payload: bytes) -> str:
    raise _rehydration_failure("_codec_label")


def _segment_validation(name: str, segment: bytes) -> dict[str, Any]:
    raise _rehydration_failure("_segment_validation")


def _segment_row(name: str, segment: bytes, *args: Any) -> dict[str, Any]:
    raise _rehydration_failure("_segment_row")


def _side_member_row(payload: _MemberPayload) -> dict[str, Any]:
    raise _rehydration_failure("_side_member_row")


def _member_payloads(archive_path: Path) -> Sequence[_MemberPayload]:
    raise _rehydration_failure("_member_payloads")


def _try_primary_bundle(payload: _MemberPayload) -> Any:
    raise _rehydration_failure("_try_primary_bundle")


def _profile_one_archive(archive_path: Path) -> _ArchiveProfile:
    raise _rehydration_failure("_profile_one_archive")


def _segment_delta_rows(reference: Any, candidate: Any) -> list[dict[str, Any]]:
    raise _rehydration_failure("_segment_delta_rows")


def _comparison(reference: Any, candidate: Any) -> dict[str, Any]:
    raise _rehydration_failure("_comparison")


def _rank_actions(actions: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    raise _rehydration_failure("_rank_actions")


def _parse_label_path(label_path: str) -> dict[str, Any]:
    raise _rehydration_failure("_parse_label_path")


def build_endgame_decision_profile(
    args: argparse.Namespace,
) -> dict[str, Any]:
    raise _rehydration_failure("build_endgame_decision_profile")


def render_markdown(profile: dict[str, Any]) -> str:
    raise _rehydration_failure("render_markdown")


def write_outputs(
    profile: dict[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
) -> None:
    raise _rehydration_failure("write_outputs")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Byte-level endgame decision support for PR85-family archives."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="reference frontier archive (PR85 single-member zip)",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="candidate archive (may be passed multiple times)",
    )
    parser.add_argument(
        "--label",
        type=str,
        action="append",
        default=[],
        help="optional label for each candidate (positional 1:1 with --candidate)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="write deterministic JSON decision profile",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        help="write markdown summary",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = build_endgame_decision_profile(args)
    write_outputs(
        profile, json_out=args.json_out, markdown_out=args.markdown_out
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
