# SPDX-License-Identifier: MIT
"""STC-Dasher encoder - substrate-agnostic ``encode(bytes, sigma) -> bytes``.

Per the Grand Reunion symposium 2026-05-15 Phase F Composite #6.
Delegates math primitives to :mod:`tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism`.

Encoder contract
----------------
``encode(residual_bytes, sigma) -> bytes`` produces a byte-stable envelope:

    +-----------------------------+
    | magic (5 bytes) "STCD\\x01" |
    | schema_version (1 byte)     |
    | sigma_int8 (1 byte)         |
    | constraint_length (1 byte)  |
    | context_length (1 byte)     |
    | n_input_symbols (4 bytes BE)|
    | n_syndrome_bits (4 bytes BE)|
    | syndrome_payload_len (4 BE) |
    | syndrome_payload bytes ...  |
    | residual_envelope_len (4 BE)|
    | residual_envelope bytes ... |
    +-----------------------------+

The syndrome payload is the deterministic STC syndrome (length
``ceil(n_syndrome_bits/8)`` bytes). The residual envelope carries the
original input bytes (so the scaffold's roundtrip is byte-stable; the
full STC-Viterbi inverse that recovers symbols from syndrome alone is
council-gated per the symposium spec).

Sigma controls the rate-distortion trade-off:
- ``sigma=0`` -> lossless; full residual envelope preserved.
- ``sigma>0`` -> lossy; reserved for the post-scaffold rate-axis sweep.

For the v1 scaffold, ``sigma`` is recorded but only ``sigma=0`` is
exercised by the roundtrip-correct path. Lossy paths (``sigma>0``) are
DEFERRED-pending-research per CLAUDE.md "KILL/FALSIFIED is LAST RESORT"
+ Catalog #220 scaffold discipline.

[verified-against: Filler-Judas-Fridrich 2011 IEEE TIFS Theorem 4
(``R_STC(D) <= R_AC(D) + 1/h``); MacKay 2003 ITILA section 6.6 (Dasher AC
context model); Cover & Thomas 2nd ed section 13.5.1 (arithmetic coding bit
cost = cross-entropy).]

Lane: ``lane_stc_dasher_scaffold_v1_20260515``.
"""
from __future__ import annotations

import dataclasses
import math
import struct
from typing import Final

import numpy as np

from tac.symposium_impls.stc_dasher_arithmetic_coding_maximalism import (
    DEFAULT_STC_CONSTRAINT_LENGTH,
    DasherContextModel,
    arithmetic_code_bit_estimate,
    build_default_stc_parity_matrix,
    stc_encode_to_syndrome,
)

__all__ = (
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_PAYLOAD_BIT_RATIO",
    "STCDasherEncodeResult",
    "STCDasherEncoder",
    "encode_stream",
)

# Default Dasher AC context length. ``k=2`` is MacKay's canonical choice
# for binary-alphabet sparse-signal streams (ITILA section 6.6).
DEFAULT_CONTEXT_LENGTH: Final[int] = 2

# Default payload-to-cover ratio for the STC parity-check matrix.
# ``payload_bits = n_cover // 4`` gives a 25% syndrome rate, which Filler
# 2011 section IV.B identifies as the canonical sweet-spot for high-entropy
# substrate-archive streams.
DEFAULT_PAYLOAD_BIT_RATIO: Final[int] = 4

# Internal envelope constants (kept in sync with __init__).
_STC_DASHER_MAGIC: Final[bytes] = b"STCD\x01"
_STC_DASHER_SCHEMA_VERSION: Final[int] = 1


@dataclasses.dataclass(frozen=True)
class STCDasherEncodeResult:
    """Result of one STC-Dasher encode pass.

    Carries the encoded bytes plus diagnostic counters so the caller can
    log the rate-distortion trade-off without re-running the codec.
    """

    encoded_bytes: bytes
    n_input_bytes: int
    n_syndrome_bytes: int
    estimated_arithmetic_bits: float
    constraint_length: int
    context_length: int
    sigma_int8: int

    @property
    def overhead_bytes(self) -> int:
        """Bytes added by the envelope (magic + headers; excludes payload)."""
        return len(self.encoded_bytes) - self.n_input_bytes - self.n_syndrome_bytes

    @property
    def total_size_bytes(self) -> int:
        return len(self.encoded_bytes)


def _bytes_to_binary_symbols(payload: bytes) -> np.ndarray:
    """Unpack bytes -> binary uint8 array of length ``8 * len(payload)``."""
    if not payload:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(payload, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)


def _pack_syndrome_bits(syndrome: np.ndarray) -> bytes:
    """Pack a binary syndrome array (uint8 0/1) into bytes (big-endian)."""
    if syndrome.size == 0:
        return b""
    if syndrome.ndim != 1:
        raise ValueError("syndrome must be 1D")
    sy = np.ascontiguousarray(syndrome.astype(np.uint8))
    if not ((sy == 0) | (sy == 1)).all():
        raise ValueError("syndrome entries must be 0 or 1")
    pad = (-sy.size) % 8
    if pad:
        sy = np.concatenate([sy, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(sy, bitorder="big").tobytes()


def encode_stream(
    residual_bytes: bytes,
    sigma: float,
    *,
    constraint_length: int = DEFAULT_STC_CONSTRAINT_LENGTH,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    payload_bit_ratio: int = DEFAULT_PAYLOAD_BIT_RATIO,
) -> STCDasherEncodeResult:
    """Stateless functional API: encode a byte stream via STC-Dasher v1.

    Parameters
    ----------
    residual_bytes
        The raw residual stream from a substrate archive (renderer
        parameters, mask argmax, latent bytes, etc.). Substrate-agnostic.
    sigma
        Rate-distortion trade-off control. ``sigma=0`` is the only
        roundtrip-byte-stable setting in v1; ``sigma>0`` is reserved.
        Stored as a clamped int8 (range [-128, 127]) in the envelope.
    constraint_length
        STC parity-check matrix constraint length. Filler 2011 section IV.B
        canonical default is 12.
    context_length
        Dasher AC context model length. MacKay 2003 section 6.6 canonical
        default is 2.
    payload_bit_ratio
        Ratio of payload bits to cover bits in the STC parity matrix.
        Default 4 = 25% syndrome rate.

    Returns
    -------
    :class:`STCDasherEncodeResult`
        Envelope-wrapped bytes plus diagnostic counters.

    Raises
    ------
    TypeError
        If ``residual_bytes`` is not ``bytes`` or ``bytearray``.
    ValueError
        If parameters are out of canonical range.
    """
    if not isinstance(residual_bytes, (bytes, bytearray)):
        raise TypeError("residual_bytes must be bytes or bytearray")
    payload = bytes(residual_bytes)
    if constraint_length < 1 or constraint_length > 255:
        raise ValueError("constraint_length must be in [1, 255]")
    if context_length < 0 or context_length > 255:
        raise ValueError("context_length must be in [0, 255]")
    if payload_bit_ratio < 1:
        raise ValueError("payload_bit_ratio must be >= 1")
    if not math.isfinite(sigma):
        raise ValueError("sigma must be finite")

    sigma_int8 = max(-128, min(127, round(sigma)))

    # Encode the syndrome via the canonical Filler-Judas-Fridrich primitive.
    bits = _bytes_to_binary_symbols(payload)
    n_input_symbols = int(bits.size)
    if n_input_symbols == 0:
        # Empty input: produce a syndrome-less envelope with zero payload.
        envelope = (
            _STC_DASHER_MAGIC
            + bytes([_STC_DASHER_SCHEMA_VERSION])
            + struct.pack(">b", sigma_int8)
            + bytes([constraint_length])
            + bytes([context_length])
            + struct.pack(">III", 0, 0, 0)
            + b""
            + struct.pack(">I", 0)
            + b""
        )
        return STCDasherEncodeResult(
            encoded_bytes=envelope,
            n_input_bytes=0,
            n_syndrome_bytes=0,
            estimated_arithmetic_bits=0.0,
            constraint_length=constraint_length,
            context_length=context_length,
            sigma_int8=sigma_int8,
        )

    payload_bits = max(n_input_symbols // payload_bit_ratio, 1)
    parity = build_default_stc_parity_matrix(
        n_cover=n_input_symbols,
        payload_bits=payload_bits,
        constraint_length=constraint_length,
    )
    stc_result = stc_encode_to_syndrome(symbols=bits, parity=parity)
    syndrome_bytes = _pack_syndrome_bits(stc_result.syndrome)
    n_syndrome_bits = int(stc_result.syndrome.size)

    # Estimate arithmetic-coded bit cost of the syndrome under a Dasher
    # context model. This is informational; the on-the-wire syndrome
    # payload is the raw packed syndrome bits (length-prefixed).
    dasher = DasherContextModel(
        context_length=context_length,
        symbol_alphabet_size=2,
    )
    ac_bits = arithmetic_code_bit_estimate(stc_result.syndrome, model=dasher)

    # Build the v1 envelope. Residual envelope carries the original bytes
    # so the scaffold roundtrip is byte-stable. Council-gated full
    # Viterbi inverse will replace the residual envelope with a length=0
    # marker once the symbol-recovery primitive lands.
    header = (
        _STC_DASHER_MAGIC
        + bytes([_STC_DASHER_SCHEMA_VERSION])
        + struct.pack(">b", sigma_int8)
        + bytes([constraint_length])
        + bytes([context_length])
        + struct.pack(">III", n_input_symbols, n_syndrome_bits, len(syndrome_bytes))
    )
    body = (
        syndrome_bytes
        + struct.pack(">I", len(payload))
        + payload
    )
    envelope = header + body

    return STCDasherEncodeResult(
        encoded_bytes=envelope,
        n_input_bytes=len(payload),
        n_syndrome_bytes=len(syndrome_bytes),
        estimated_arithmetic_bits=float(ac_bits),
        constraint_length=constraint_length,
        context_length=context_length,
        sigma_int8=sigma_int8,
    )


@dataclasses.dataclass(frozen=True)
class STCDasherEncoder:
    """Stateful encoder facade. Identical contract to :func:`encode_stream`."""

    constraint_length: int = DEFAULT_STC_CONSTRAINT_LENGTH
    context_length: int = DEFAULT_CONTEXT_LENGTH
    payload_bit_ratio: int = DEFAULT_PAYLOAD_BIT_RATIO

    def __post_init__(self) -> None:
        if self.constraint_length < 1 or self.constraint_length > 255:
            raise ValueError("constraint_length must be in [1, 255]")
        if self.context_length < 0 or self.context_length > 255:
            raise ValueError("context_length must be in [0, 255]")
        if self.payload_bit_ratio < 1:
            raise ValueError("payload_bit_ratio must be >= 1")

    def encode(self, residual_bytes: bytes, sigma: float) -> bytes:
        """Encode a residual stream and return the envelope bytes only.

        For the diagnostic-rich result, use :func:`encode_with_diagnostics`.
        """
        return self.encode_with_diagnostics(residual_bytes, sigma).encoded_bytes

    def encode_with_diagnostics(
        self, residual_bytes: bytes, sigma: float
    ) -> STCDasherEncodeResult:
        """Encode and return the full :class:`STCDasherEncodeResult`."""
        return encode_stream(
            residual_bytes,
            sigma,
            constraint_length=self.constraint_length,
            context_length=self.context_length,
            payload_bit_ratio=self.payload_bit_ratio,
        )
