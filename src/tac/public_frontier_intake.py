"""Static intake and byte diffing for public frontier archives.

This module is intentionally offline and byte-only. It validates ZIP custody,
identifies PR85-family bundle segments, records charged side-channel hashes,
and supports simple cross-baseline diffing for triage.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/public_frontier_intake.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: error class and CLI surface are exposed; internal
helpers and the diff/profile loops are deferred to ``NotImplementedError``
(intricate generator/lambda/closures that pycdc cannot fully decompile).
There are no live consumers in the current codebase.
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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "public_frontier_intake_profile_v1"
TOOL = "experiments/profile_public_frontier_intake.py"
EVIDENCE_GRADE = "byte_intake_only"

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/public_frontier_intake.recovery_spec.json"
)


class PublicFrontierIntakeError(ValueError):
    """Raised when public frontier intake inputs are malformed."""

    pass


@dataclass(frozen=True)
class _ParsedArchive:
    path: str
    total_bytes: int
    sha256: str
    member_payloads: Sequence[Mapping[str, Any]]
    extra: Mapping[str, Any] = field(default_factory=dict)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


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
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains intricate generator/lambda "
        f"closures pycdc cannot decompile; original bytecode preserved in "
        f"{_QUARANTINE_SPEC}"
    )


def profile_public_frontier_archive(
    archive: Path | str, *, baselines: Sequence[Path] | None = None,
    inspect_segments: bool = True,
) -> dict[str, Any]:
    """Profile one public frontier archive byte-by-byte (deferred)."""
    raise _rehydration_failure("profile_public_frontier_archive")


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
        description="Static intake / byte diffing for public frontier archives."
    )
    parser.add_argument("archive", type=Path, help="archive.zip to intake")
    parser.add_argument(
        "--baseline", type=Path, action="append", default=[],
        help="baseline archive(s) to byte-diff against (may repeat)",
    )
    parser.add_argument(
        "--no-segments", action="store_true",
        help="skip per-PR85-segment inspection",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profile = profile_public_frontier_archive(
        args.archive,
        baselines=args.baseline,
        inspect_segments=not args.no_segments,
    )
    write_outputs(profile, json_out=args.json_out, markdown_out=args.markdown_out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
