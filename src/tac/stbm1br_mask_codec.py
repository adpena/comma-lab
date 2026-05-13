"""Standalone decoder for PR90 ``STBM1BR`` semantic topband mask segments.

The public PR90 qrepro archive stores its semantic masks as::

    STBM1BR\\x00 + brotli(QTBM* topband/road-boundary stream)

This module reimplements the narrow decode surface needed to prove and replay
that mask stream inside PR85-family local candidate pipelines. It intentionally
does not import from the PR90 source checkout or touch any scorer/runtime model.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/stbm1br_mask_codec.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: Module-level constants, error class, dataclass, and trivial
helpers are reconstructed exactly. The range decoder, QTBM frame topband decoder,
and sparse-big unpack helpers (~1500 lines of disassembly with bit-level state
machines) are left as ``NotImplementedError`` stubs that fail loud rather than
return wrong masks. The production path uses the Rust bridge
(``tac.stbm1br_rust_bridge.decode_stbm1br_mask_segment_via_rust``) gated on
``PACT_STBM1BR_RUST_DECODER`` env var.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - hard dep in production
    import brotli  # noqa: F401
except ImportError:  # pragma: no cover
    brotli = None  # type: ignore[assignment]

import numpy as np

# --- module constants (verbatim from recovery spec) ---
STBM1BR_MAGIC = b"STBM1BR\x00"
QTBM_MAGICS = (b"QTBM5\x00", b"QTBM4\x00", b"QTBM3\x00", b"QTBM2\x00", b"QTBM1\x00")
N_CLASSES = 5
N_SYM = N_CLASSES - 1
DEFAULT_SHAPE = (600, 384, 512)

# Feature ids used by the STBM1BR context model
FEAT_DIAG_TLTL = 0
FEAT_LEFT_LEFT = 1
FEAT_TOP_TOP_TOP = 2
FEAT_PREV_PREV_PREV = 3
FEAT_DIAG_TRTR = 4
FEAT_PREV_LEFT = 5
FEAT_PREV_RIGHT = 6
FEAT_PREV_TOP = 7
FEAT_PREV_BOTTOM = 8
FEAT_PREV2_LEFT = 9
FEAT_PREV2_RIGHT = 10
FEAT_PREV_BOTTOM_RIGHT = 11
FEAT_PREV_BOTTOM_LEFT = 12
FEAT_PREV_TOP_RIGHT = 13
FEAT_PREV_BOTTOM2 = 14
FEAT_PREV_RIGHT2 = 15
FEAT_X_BIN5 = 16
FEAT_Y_BIN5 = 17
FEAT_X_BIN5_SHIFT = 20
FEAT_PEEL_DIST42 = 30
FEAT_PEEL_BOUND5 = 31
FEAT_PEEL_SLOPE5 = 32

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/stbm1br_mask_codec.recovery_spec.json"
)


class STBM1BRError(ValueError):
    """Raised when an STBM1BR payload violates the self-contained contract."""

    pass


@dataclass(frozen=True)
class STBM1BRMetadata:
    """Parsed metadata for an STBM1BR mask segment."""

    n_frames: int
    height: int
    width: int
    n_classes: int
    qtbm_magic: bytes
    encoded_bytes: int
    decoded_bytes: int
    encoded_sha256: str
    decoded_sha256: str
    extra: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[name-defined]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise STBM1BRError(message)


def parse_stbm1br_metadata(payload: bytes) -> STBM1BRMetadata:
    """Parse the leading STBM1BR header and return structural metadata.

    Returns a metadata object even if the inner QTBM topband payload is
    not decoded (which requires the full range-decoder, deferred to the
    Rust bridge).
    """
    if not payload.startswith(STBM1BR_MAGIC):
        raise STBM1BRError(
            f"STBM1BR mask payload does not start with magic; got {payload[:8]!r}"
        )
    # The PR90 STBM1BR layout is:
    #   magic (8) | brotli(QTBM<N> stream)
    # The QTBM stream itself begins with one of QTBM_MAGICS.
    body = payload[len(STBM1BR_MAGIC) :]
    if brotli is None:
        raise STBM1BRError(
            "STBM1BR metadata parse requires the brotli package"
        )
    try:
        decoded = brotli.decompress(body)
    except brotli.error as exc:  # pragma: no cover - depends on payload
        raise STBM1BRError(
            f"STBM1BR brotli stream is not decodable: {exc!r}"
        ) from exc
    qtbm_magic = b""
    for cand in QTBM_MAGICS:
        if decoded.startswith(cand):
            qtbm_magic = cand
            break
    if not qtbm_magic:
        raise STBM1BRError(
            f"STBM1BR inner stream has unknown magic: {decoded[:8]!r}"
        )
    # Extract a lightweight (n_frames, height, width) probe from the
    # QTBM header. The full QTBM grammar is not parsed here; we read the
    # first three uint32s after the magic which by convention are
    # (n_frames, height, width). This is documented in the original
    # bytecode (parse_stbm1br_metadata varnames + dis).
    if len(decoded) < len(qtbm_magic) + 12:
        raise STBM1BRError("STBM1BR QTBM stream too short for header")
    n_frames, height, width = struct.unpack_from(
        "<III", decoded, len(qtbm_magic)
    )
    return STBM1BRMetadata(
        n_frames=int(n_frames),
        height=int(height),
        width=int(width),
        n_classes=N_CLASSES,
        qtbm_magic=qtbm_magic,
        encoded_bytes=len(payload),
        decoded_bytes=len(decoded),
        encoded_sha256=sha256_bytes(payload),
        decoded_sha256=sha256_bytes(decoded),
        extra={},
    )


def metadata_as_dict(metadata: STBM1BRMetadata) -> dict[str, Any]:
    """Return ``metadata`` as a JSON-serialisable dict."""
    return {
        "n_frames": int(metadata.n_frames),
        "height": int(metadata.height),
        "width": int(metadata.width),
        "n_classes": int(metadata.n_classes),
        "qtbm_magic": metadata.qtbm_magic.decode("latin-1"),
        "encoded_bytes": int(metadata.encoded_bytes),
        "decoded_bytes": int(metadata.decoded_bytes),
        "encoded_sha256": metadata.encoded_sha256,
        "decoded_sha256": metadata.decoded_sha256,
        "extra": dict(metadata.extra),
    }


# --- Functions deferred to Rust bridge / future hand-rehydration ---


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} requires bit-level range-decoder + QTBM "
        f"context model that pycdc cannot fully decompile; original bytecode preserved "
        f"in {_QUARANTINE_SPEC}. Production path: set PACT_STBM1BR_RUST_DECODER and "
        f"call tac.stbm1br_rust_bridge.decode_stbm1br_mask_segment_via_rust."
    )


class _RangeDecoder:  # REHYDRATED stub
    """Bitwise range decoder used by the QTBM topband entropy model.

    Full reconstruction deferred — see :func:`_rehydration_failure`.
    """

    TOP = 0xFFFFFFFF
    HALF = 0x80000000
    QUARTER = 0x40000000
    THREE_QUARTER = 0xC0000000

    def __init__(self, data: bytes) -> None:
        raise _rehydration_failure("_RangeDecoder")

    def _read_bit(self) -> int:
        raise _rehydration_failure("_RangeDecoder._read_bit")

    def decode_target(self, total: int) -> int:
        raise _rehydration_failure("_RangeDecoder.decode_target")

    def advance(self, cum_low: int, cum_high: int, total: int) -> None:
        raise _rehydration_failure("_RangeDecoder.advance")


def _m5_ctx(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_m5_ctx")


def _leb128_decode_big_deltas(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_leb128_decode_big_deltas")


def decode_boundary_mask_payload(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("decode_boundary_mask_payload")


def decode_topband_payload(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("decode_topband_payload")


def unpack_sparse_big(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("unpack_sparse_big")


def unpack_sparse_big_plain(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("unpack_sparse_big_plain")


def unpack_sparse_big_plain_colsfirst(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("unpack_sparse_big_plain_colsfirst")


def _decode_frame_topband(*args: Any, **kwargs: Any) -> Any:
    raise _rehydration_failure("_decode_frame_topband")


def _parse_qtbm_blob(blob: bytes) -> Any:
    raise _rehydration_failure("_parse_qtbm_blob")


def decode_qtbm_topband_blob(blob: bytes) -> np.ndarray:
    raise _rehydration_failure("decode_qtbm_topband_blob")


def decode_stbm1br_mask_segment(
    payload: bytes,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Decode a complete STBM1BR mask segment to a ``(N, H, W)`` int array.

    REHYDRATION STATUS: DEFERRED. The full Python decoder requires the
    range-decoder + QTBM context model (~1500 lines of bit-level state machine
    that pycdc cannot fully decompile). Production path uses the Rust bridge.
    """
    if os.environ.get("PACT_STBM1BR_RUST_DECODER"):
        from tac.stbm1br_rust_bridge import (  # noqa: PLC0415 - optional native bridge
            STBM1BRRustBridgeError,
            decode_stbm1br_mask_segment_via_rust,
        )

        try:
            return decode_stbm1br_mask_segment_via_rust(
                payload,
                expected_shape=expected_shape,
            )
        except STBM1BRRustBridgeError as exc:
            raise STBM1BRError(str(exc)) from exc
    raise _rehydration_failure("decode_stbm1br_mask_segment")


def decode_stbm1br_mask_file(
    path: Path | str,
    *,
    expected_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Read and decode an STBM1BR mask file from disk."""
    payload = Path(path).read_bytes()
    return decode_stbm1br_mask_segment(payload, expected_shape=expected_shape)


# Local Mapping import (deferred to avoid runtime cost / circular concerns)
from typing import Mapping  # noqa: E402  - placed late to match original module layout
