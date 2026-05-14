# SPDX-License-Identifier: MIT
"""Local PR85 QMA9 alternate-grammar screening helpers.

This module intentionally does not modify or import the contest inflate
runtime.  It wraps an audited local range-mask codec binary as a byte-screen
oracle, records deterministic payload custody, and marks every non-current
runtime grammar as dispatch-locked until the public runtime explicitly learns
that grammar.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

QMA9_CURRENT_RUNTIME_MODE = "adaptive9bin"
QMA9_CURRENT_RUNTIME_MAGIC = "QMA9"

DEFAULT_ALT_GRAMMAR_MODES: tuple[str, ...] = (
    "adaptive6pr",
    "adaptive7prpd",
    "adaptive8prpdpl",
    "adaptive8prpdpu",
    "adaptive8prpdleft2",
    "adaptive8prpdup2",
    "adaptive8prpdpdl",
    "adaptive8prpdpdr",
    "adaptive9up2left2",
    "adaptive9up2pu",
    "adaptive9up2pl",
    QMA9_CURRENT_RUNTIME_MODE,
)


class QMA9AltGrammarError(RuntimeError):
    """Raised when a local alternate-grammar byte screen is not trustworthy."""


@dataclass(frozen=True)
class CompiledCodec:
    """Compiled local codec helper metadata."""

    binary: Path
    source: Path
    source_sha256: str
    compile_command: tuple[str, ...]


@dataclass(frozen=True)
class AltGrammarRun:
    """One successful local codec run and payload custody."""

    mode: str
    payload_path: Path
    payload_bytes: int
    payload_sha256: str
    payload_magic: str
    raw_bytes: int
    bitstream_bytes: int | None
    model_bytes: int | None
    stdout_json: Mapping[str, Any]
    run_command: tuple[str, ...]


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for in-memory bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ascii_magic(payload: bytes) -> str:
    """Return a printable four-byte payload magic, or a hex fallback."""

    magic = bytes(payload[:4])
    try:
        text = magic.decode("ascii")
    except UnicodeDecodeError:
        return magic.hex()
    if all(32 <= ord(ch) < 127 for ch in text):
        return text
    return magic.hex()


def parse_modes(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse comma-separated or iterable mode names into a stable tuple."""

    if isinstance(value, str):
        modes = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        modes = tuple(str(part).strip() for part in value if str(part).strip())
    if not modes:
        raise QMA9AltGrammarError("at least one alternate grammar mode is required")
    return modes


def mode_family(mode: str) -> str:
    """Classify an audited range-mask codec mode for manifests."""

    if mode == QMA9_CURRENT_RUNTIME_MODE:
        return "current_qma9_runtime_reference"
    if mode.startswith("adaptive6") or mode.startswith("adaptive7") or mode.startswith("adaptive8"):
        return "table_reduction_context_family"
    if mode.startswith("adaptive9"):
        return "qma9_neighbor_table_variant"
    return "alternate_range_mask_grammar"


def compile_local_codec(source: Path, work_dir: Path) -> CompiledCodec:
    """Compile an audited local codec helper for deterministic byte screening."""

    source = Path(source)
    if not source.is_file():
        raise QMA9AltGrammarError(f"local codec source is missing: {source}")
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise QMA9AltGrammarError("no C++ compiler found on PATH for QMA9 alt-grammar screen")
    binary = Path(work_dir) / "qma9_alt_grammar_codec"
    command = (compiler, "-O3", "-std=c++17", str(source), "-o", str(binary))
    proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise QMA9AltGrammarError(f"alt-grammar codec compile failed: {proc.stderr[-4000:]}")
    return CompiledCodec(
        binary=binary,
        source=source,
        source_sha256=sha256_file(source),
        compile_command=command,
    )


def _parse_codec_stdout(stdout: str, *, mode: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise QMA9AltGrammarError(f"codec mode {mode!r} emitted no JSON stdout")
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise QMA9AltGrammarError(f"codec mode {mode!r} stdout was not JSON: {lines[-1]!r}") from exc
    if not isinstance(parsed, dict):
        raise QMA9AltGrammarError(f"codec mode {mode!r} stdout JSON is not an object")
    return parsed


def run_codec_mode(
    codec: CompiledCodec,
    *,
    raw_tokens: Path,
    frame_count: int,
    width: int,
    height: int,
    output_path: Path,
    mode: str,
    timeout_seconds: int,
) -> AltGrammarRun:
    """Run one local codec mode.

    The audited helper exits nonzero if its own decode check does not reproduce
    the input tokens.  This wrapper treats a zero exit plus size/SHA custody as
    a strict local token-parity proof for the emitted payload.
    """

    raw_tokens = Path(raw_tokens)
    if not raw_tokens.is_file():
        raise QMA9AltGrammarError(f"raw token source is missing: {raw_tokens}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        str(codec.binary),
        str(raw_tokens),
        str(int(frame_count)),
        str(int(width)),
        str(int(height)),
        str(output_path),
        str(mode),
    )
    proc = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=int(timeout_seconds),
    )
    if proc.returncode != 0:
        raise QMA9AltGrammarError(
            f"codec mode {mode!r} failed with rc={proc.returncode}: {proc.stderr[-4000:]}"
        )
    if not output_path.is_file():
        raise QMA9AltGrammarError(f"codec mode {mode!r} did not write {output_path}")
    stdout_json = _parse_codec_stdout(proc.stdout, mode=mode)
    payload = output_path.read_bytes()
    packed_bytes = int(stdout_json.get("packed_bytes", len(payload)))
    if packed_bytes != len(payload):
        raise QMA9AltGrammarError(
            f"codec mode {mode!r} reported {packed_bytes} packed bytes but wrote {len(payload)}"
        )
    raw_bytes = int(stdout_json.get("raw_bytes", raw_tokens.stat().st_size))
    if raw_bytes != raw_tokens.stat().st_size:
        raise QMA9AltGrammarError(
            f"codec mode {mode!r} reported {raw_bytes} raw bytes but token file has {raw_tokens.stat().st_size}"
        )
    return AltGrammarRun(
        mode=str(mode),
        payload_path=output_path,
        payload_bytes=len(payload),
        payload_sha256=sha256_bytes(payload),
        payload_magic=ascii_magic(payload),
        raw_bytes=raw_bytes,
        bitstream_bytes=(
            int(stdout_json["bitstream_bytes"]) if "bitstream_bytes" in stdout_json else None
        ),
        model_bytes=(int(stdout_json["model_bytes"]) if "model_bytes" in stdout_json else None),
        stdout_json=stdout_json,
        run_command=command,
    )


def runtime_custody_contract(
    *,
    mode: str,
    payload_magic: str,
    live_runtime_cpp: Path,
    replay_codec_cpp: Path,
) -> dict[str, Any]:
    """Return the fail-closed runtime contract for a candidate payload."""

    mode = str(mode)
    payload_magic = str(payload_magic)
    live_runtime_cpp = Path(live_runtime_cpp)
    replay_codec_cpp = Path(replay_codec_cpp)
    live_runtime_supported = mode == QMA9_CURRENT_RUNTIME_MODE and payload_magic == QMA9_CURRENT_RUNTIME_MAGIC

    required_changes: list[str] = []
    if not live_runtime_supported:
        if payload_magic == QMA9_CURRENT_RUNTIME_MAGIC:
            required_changes.append(
                "allocate a new magic or charged mode-id header; current robust runtime interprets QMA9 as adaptive9bin"
            )
        else:
            required_changes.append(
                f"extend robust_current range_mask_codec.cpp to recognize {payload_magic} and decode mode {mode}"
            )
            required_changes.append(
                f"extend inflate/unpack QMA9 mask detection to admit {payload_magic} without changing fixed-slice custody"
            )
        required_changes.extend(
            [
                "add runtime output-parity tests against the exact raw token SHA before any score eval",
                "record the updated inflate runtime tree SHA in exact-eval provenance",
                "only dispatch after a fresh Level-2 lane claim and archive.zip -> inflate.sh -> upstream/evaluate.py CUDA gate",
            ]
        )

    return {
        "live_runtime_supported": live_runtime_supported,
        "current_live_runtime_contract": {
            "source": str(live_runtime_cpp),
            "source_sha256": sha256_file(live_runtime_cpp) if live_runtime_cpp.is_file() else None,
            "accepted_magic": QMA9_CURRENT_RUNTIME_MAGIC,
            "accepted_mode": QMA9_CURRENT_RUNTIME_MODE,
            "charged_mode_id_header": False,
        },
        "screen_codec_contract": {
            "source": str(replay_codec_cpp),
            "source_sha256": sha256_file(replay_codec_cpp) if replay_codec_cpp.is_file() else None,
            "mode": mode,
            "payload_magic": payload_magic,
            "local_token_parity": "codec self-decode must equal raw token input before payload is accepted",
        },
        "dispatch_unlocked": False,
        "safe_for_remote_dispatch": False,
        "required_runtime_changes_before_dispatch": required_changes,
    }
