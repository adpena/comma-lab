# SPDX-License-Identifier: MIT
"""STC-Dasher decoder - substrate-agnostic ``decode(bytes, sigma) -> bytes``.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #6.
Inverse of :mod:`tac.codecs.stc_dasher.encoder`.

Decoder contract
----------------
``decode(encoded_bytes, sigma) -> bytes`` returns the original residual
bytes (byte-stable roundtrip for ``sigma=0`` v1 scaffold).

The decoder also independently re-derives the syndrome from the
recovered residual bytes and verifies it matches the on-the-wire
syndrome payload - this is the no-op detector per CLAUDE.md "No-op
detector" non-negotiable + Catalog #105 / #139 sister.

Lane: ``lane_stc_dasher_scaffold_v1_20260515``.
"""
from __future__ import annotations

import dataclasses
import struct
from typing import Final

import numpy as np

from tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism import (
    build_default_stc_parity_matrix,
    stc_encode_to_syndrome,
)

__all__ = (
    "STCDasherDecodeError",
    "STCDasherDecodeResult",
    "STCDasherDecoder",
    "decode_stream",
)

# Internal envelope constants (kept in sync with __init__ + encoder).
_STC_DASHER_MAGIC: Final[bytes] = b"STCD\x01"
_STC_DASHER_SCHEMA_VERSION: Final[int] = 1
_HEADER_LEN: Final[int] = (
    len(_STC_DASHER_MAGIC) + 1 + 1 + 1 + 1 + 4 + 4 + 4
)  # = 5 + 1 + 1 + 1 + 1 + 4 + 4 + 4 = 21


class STCDasherDecodeError(ValueError):
    """Raised when the on-the-wire envelope fails magic / schema / parity check."""


@dataclasses.dataclass(frozen=True)
class STCDasherDecodeResult:
    """Result of one STC-Dasher decode pass."""

    decoded_bytes: bytes
    n_input_symbols: int
    n_syndrome_bits: int
    syndrome_verified: bool
    constraint_length: int
    context_length: int
    sigma_int8: int


def _bytes_to_binary_symbols(payload: bytes) -> np.ndarray:
    """Mirror of :func:`tac.codecs.stc_dasher.encoder._bytes_to_binary_symbols`."""
    if not payload:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(payload, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)


def _pack_syndrome_bits(syndrome: np.ndarray) -> bytes:
    """Mirror of :func:`tac.codecs.stc_dasher.encoder._pack_syndrome_bits`."""
    if syndrome.size == 0:
        return b""
    sy = np.ascontiguousarray(syndrome.astype(np.uint8))
    pad = (-sy.size) % 8
    if pad:
        sy = np.concatenate([sy, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(sy, bitorder="big").tobytes()


def decode_stream(
    encoded_bytes: bytes,
    sigma: float,
    *,
    verify_syndrome: bool = True,
) -> STCDasherDecodeResult:
    """Stateless functional API: decode an STC-Dasher v1 envelope.

    Parameters
    ----------
    encoded_bytes
        The envelope produced by
        :func:`tac.codecs.stc_dasher.encoder.encode_stream`.
    sigma
        Operator-supplied sigma. MUST match the encode-time ``sigma`` (the
        envelope stores ``sigma_int8 = clamp(round(sigma), -128, 127)``).
        For v1 scaffold (``sigma=0`` only), mismatch raises.
    verify_syndrome
        If True (default), independently re-derives the STC syndrome from
        the recovered bytes and verifies it matches the on-the-wire
        payload (no-op detector per CLAUDE.md non-negotiable).

    Returns
    -------
    :class:`STCDasherDecodeResult`
        The recovered bytes plus diagnostic counters.

    Raises
    ------
    TypeError
        If ``encoded_bytes`` is not ``bytes`` or ``bytearray``.
    STCDasherDecodeError
        If the envelope fails magic / schema / length / parity / sigma
        check.
    """
    if not isinstance(encoded_bytes, (bytes, bytearray)):
        raise TypeError("encoded_bytes must be bytes or bytearray")
    blob = bytes(encoded_bytes)
    if len(blob) < _HEADER_LEN:
        raise STCDasherDecodeError(
            f"envelope too short ({len(blob)} bytes; expected >= {_HEADER_LEN})"
        )

    magic = blob[: len(_STC_DASHER_MAGIC)]
    if magic != _STC_DASHER_MAGIC:
        raise STCDasherDecodeError(
            f"magic mismatch: got {magic!r}, expected {_STC_DASHER_MAGIC!r}"
        )

    cursor = len(_STC_DASHER_MAGIC)
    schema_version = blob[cursor]
    cursor += 1
    if schema_version != _STC_DASHER_SCHEMA_VERSION:
        raise STCDasherDecodeError(
            f"schema_version mismatch: got {schema_version}, "
            f"expected {_STC_DASHER_SCHEMA_VERSION}"
        )

    sigma_int8 = struct.unpack(">b", blob[cursor : cursor + 1])[0]
    cursor += 1
    constraint_length = blob[cursor]
    cursor += 1
    context_length = blob[cursor]
    cursor += 1
    n_input_symbols, n_syndrome_bits, syndrome_payload_len = struct.unpack(
        ">III", blob[cursor : cursor + 12]
    )
    cursor += 12

    # Verify operator sigma matches encode-time sigma.
    expected_sigma_int8 = max(-128, min(127, round(sigma)))
    if expected_sigma_int8 != sigma_int8:
        raise STCDasherDecodeError(
            f"sigma mismatch: caller supplied {sigma} (int8={expected_sigma_int8}), "
            f"envelope was encoded with int8={sigma_int8}"
        )

    if cursor + syndrome_payload_len > len(blob):
        raise STCDasherDecodeError(
            f"syndrome payload truncated: header claims {syndrome_payload_len} "
            f"bytes but only {len(blob) - cursor} remain"
        )
    syndrome_bytes = blob[cursor : cursor + syndrome_payload_len]
    cursor += syndrome_payload_len

    if cursor + 4 > len(blob):
        raise STCDasherDecodeError("residual envelope length prefix truncated")
    residual_len = struct.unpack(">I", blob[cursor : cursor + 4])[0]
    cursor += 4
    if cursor + residual_len > len(blob):
        raise STCDasherDecodeError(
            f"residual envelope truncated: header claims {residual_len} "
            f"bytes but only {len(blob) - cursor} remain"
        )
    residual = blob[cursor : cursor + residual_len]
    cursor += residual_len

    # Trailing bytes are forbidden by the v1 envelope spec.
    if cursor != len(blob):
        raise STCDasherDecodeError(
            f"trailing bytes after envelope: cursor={cursor}, len={len(blob)}"
        )

    # No-op detector: re-derive syndrome from recovered bytes and compare
    # against the on-the-wire payload. Per CLAUDE.md "Bit-level
    # deconstruction and entropy discipline" + Catalog #105 / #139.
    syndrome_verified = True
    if verify_syndrome and n_input_symbols > 0:
        bits = _bytes_to_binary_symbols(residual)
        if bits.size != n_input_symbols:
            raise STCDasherDecodeError(
                f"recovered residual has {bits.size} bits but envelope claims "
                f"n_input_symbols={n_input_symbols}"
            )
        from_bits_payload_bits = max(n_input_symbols // 4, 1)
        # NOTE: payload_bit_ratio is fixed at 4 for v1 scaffold per the
        # encoder default. If the encoder default changes, the decoder
        # must read it from the envelope too.
        parity = build_default_stc_parity_matrix(
            n_cover=n_input_symbols,
            payload_bits=from_bits_payload_bits,
            constraint_length=constraint_length,
        )
        rederived = stc_encode_to_syndrome(symbols=bits, parity=parity)
        rederived_bytes = _pack_syndrome_bits(rederived.syndrome)
        if rederived_bytes != syndrome_bytes:
            raise STCDasherDecodeError(
                "syndrome verification failed: re-derived syndrome does not "
                "match on-the-wire payload (envelope corrupted or parity "
                "mismatch)"
            )

    return STCDasherDecodeResult(
        decoded_bytes=residual,
        n_input_symbols=n_input_symbols,
        n_syndrome_bits=n_syndrome_bits,
        syndrome_verified=bool(syndrome_verified),
        constraint_length=constraint_length,
        context_length=context_length,
        sigma_int8=sigma_int8,
    )


@dataclasses.dataclass(frozen=True)
class STCDasherDecoder:
    """Stateful decoder facade. Identical contract to :func:`decode_stream`."""

    verify_syndrome: bool = True

    def decode(self, encoded_bytes: bytes, sigma: float) -> bytes:
        """Decode an envelope and return the recovered residual bytes."""
        return self.decode_with_diagnostics(encoded_bytes, sigma).decoded_bytes

    def decode_with_diagnostics(
        self, encoded_bytes: bytes, sigma: float
    ) -> STCDasherDecodeResult:
        """Decode and return the full :class:`STCDasherDecodeResult`."""
        return decode_stream(
            encoded_bytes,
            sigma,
            verify_syndrome=self.verify_syndrome,
        )
