"""Explicit Rust bridge for STBM1BR semantic mask decode.

This module intentionally does not auto-discover or build a native decoder.
Contest/runtime callers must pass a decoder path explicitly, or set
``PACT_STBM1BR_RUST_DECODER`` to a binary carried by the experiment/runtime.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class STBM1BRRustBridgeError(RuntimeError):
    """Raised when the explicit Rust STBM1BR decoder path cannot be used."""


def resolve_stbm1br_rust_decoder(decoder_path: str | Path | None = None) -> Path:
    """Resolve an explicit STBM1BR Rust decoder path and fail closed."""

    raw_path = (
        decoder_path
        if decoder_path is not None
        else os.environ.get("PACT_STBM1BR_RUST_DECODER")
    )
    if raw_path is None or str(raw_path).strip() == "":
        raise STBM1BRRustBridgeError(
            "STBM1BR Rust decode requested, but no decoder path was supplied. "
            "Pass decoder_path or set PACT_STBM1BR_RUST_DECODER."
        )
    path = Path(raw_path)
    if not path.is_file():
        raise STBM1BRRustBridgeError(f"STBM1BR Rust decoder is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise STBM1BRRustBridgeError(f"STBM1BR Rust decoder is not executable: {path}")
    return path


def decode_stbm1br_mask_segment_via_rust(
    segment: bytes,
    *,
    expected_shape: tuple[int, int, int],
    decoder_path: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> np.ndarray:
    """Decode ``STBM1BR`` bytes with the explicit Rust CLI and return uint8 masks."""

    frames, height, width = (int(v) for v in expected_shape)
    if frames <= 0 or height <= 0 or width <= 0:
        raise STBM1BRRustBridgeError(f"invalid expected STBM1BR shape: {expected_shape!r}")
    decoder = resolve_stbm1br_rust_decoder(decoder_path)
    expected_bytes = frames * height * width
    with tempfile.TemporaryDirectory(prefix="stbm1br-rust-") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "segment.stbm1br"
        output_path = tmpdir / "masks.raw"
        input_path.write_bytes(segment)
        cmd = [
            str(decoder),
            "decode",
            str(input_path),
            str(output_path),
            "--expected-frames",
            str(frames),
            "--expected-height",
            str(height),
            "--expected-width",
            str(width),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            raise STBM1BRRustBridgeError(
                "STBM1BR Rust decoder failed "
                f"(returncode={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        raw = output_path.read_bytes()
    if len(raw) != expected_bytes:
        raise STBM1BRRustBridgeError(
            f"STBM1BR Rust decoder wrote {len(raw)} bytes, expected {expected_bytes}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(frames, height, width)
