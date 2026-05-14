# SPDX-License-Identifier: MIT
"""PR101 GOLD ``decoder_byte_maps`` sign-encoding strategy selector.

This module ports the **per-tensor sign-encoding strategy primitive**
from the PR101 GOLD-medal submission
(``submissions/hnerv_ft_microcodec/src/codec.py``, lines 57-62 + 225-239)
into a typed, golden-vector-backed transducer.

Mechanism (PR101 source): ::

    DECODER_BYTE_MAPS = {
        9: "negzig",
        14: "negzig",
        20: "twos",
        27: "off",
    }

    def decode_mapped_u8(arr_u8, byte_map):
        if byte_map == "zig":
            return zigzag_decode_u8(arr_u8)
        if byte_map == "negzig":
            return (-zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
        if byte_map == "off":
            return (arr_u8.astype(np.int16) - 128).astype(np.int8)
        if byte_map == "twos":
            return arr_u8.view(np.int8)
        raise ValueError(f"unknown decoder byte map: {byte_map}")

PR101 selects one of FOUR sign-encoding strategies per-tensor:

* ``zig``   — zigzag encoding (default; symmetric around 0)
* ``negzig``— negated zigzag (zigzag of -x)
* ``twos``  — two's complement (raw int8 reinterpret as uint8)
* ``off``   — signed-byte-offset (q + 128 ∈ [0, 256))

Each strategy minimises entropy under a different weight distribution:

* ``zig`` for distributions roughly symmetric around 0
* ``negzig`` for distributions with the opposite-side skew vs zig
* ``twos`` for symmetric-around-zero with high-frequency low-magnitude
* ``off`` for highly-skewed-negative (so the +128 offset peaks near 128
  rather than at the edges)

The encoder chooses per tensor offline via entropy comparison; the
decoder dispatches via the per-tensor ``byte_map`` string.

The reusable primitive here is the **strategy taxonomy + dispatcher**,
NOT the specific PR101 table. Per CLAUDE.md "HNeRV parity discipline"
lesson 2 (Export-first design): this is a GRAMMAR primitive, not a
method claim.

This primitive forms a unified taxonomy with PR96
(``pr96_rem2_uint8_offset_128``) which uses only the ``off`` strategy,
and PR103's ``zigzag`` used in ``pr103_arithmetic_coding``. Per the
sister-landing operator decision surfaced 2026-05-12, the 5-strategy
unified taxonomy (negzig, zig, twos, off, raw-uint8) should be
consolidated under a single module. This module is the first concrete
step in that consolidation: it implements 4 of the 5 strategies and
exposes them via a typed strategy enum.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src/codec.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake;
Catalog #109).

PR101 GOLD anchor data
======================

* PR101 archive score: ``0.193 [contest-CUDA]`` (public claim; not
  replayed internally yet)
* DECODER_BYTE_MAPS covers 4 of PR101's 28 tensors with non-default
  strategies (indices 9, 14, 20, 27); the remaining 24 tensors default
  to ``zig``.

target_substrate_hint
=====================

``hnerv_lc_family`` — specifically the PR95-derived HNeRV-LC tensor
layout where indices [9, 14, 20, 27] benefit from non-default sign
strategies. This primitive's MECHANISM (strategy dispatch) IS reusable
across substrates; the SPECIFIC PR101 table is not.

For non-HNeRV substrates (PSD / Quantizr / sane_hnerv etc.), the
per-tensor optimal strategy must be re-derived via offline entropy
comparison over the 4 strategies per tensor.

predicted_ev_per_byte
=====================

* Rank: ``#7 EV/byte`` at PR106 r2 frontier per
  ``.omx/research/public_pr_mining_pr81_104_typed_rows_20260512.json``.
* Basis: ``[predicted; PR101 GOLD byte trace; not yet measured on
  internal substrate]`` — per-tensor sign strategy traces to PR101 GOLD
  archive bytes; estimated ~50-150 bytes total per archive.

CLAUDE.md compliance
====================

* No scorer load — pure numpy + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* OSS-friendly: public surface is the 5 names re-exported from
  ``tac.packet_compiler``.
* Pure functional transducers — no global mutable state.
* No archive bytes mutated by this module — it is byte-grammar plumbing
  only.

[empirical:src/tac/packet_compiler/golden_vectors/pr101_decoder_byte_maps_v1.json]
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Mapping

import numpy as np


# ── Strategy enum (string-typed to match PR101 wire format) ────────────────


ByteMapStrategy = Literal["zig", "negzig", "twos", "off"]

VALID_BYTE_MAP_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"zig", "negzig", "twos", "off"}
)
"""The 4 sign-encoding strategies PR101 supports.

These match PR101 source line 232-239 exactly. The set is frozen at
module-import time to prevent runtime mutation.
"""


# ── PR101 canonical anchor table (for golden vector + reference only) ─────


PR101_DECODER_BYTE_MAPS: Mapping[int, ByteMapStrategy] = MappingProxyType({
    9:  "negzig",
    14: "negzig",
    20: "twos",
    27: "off",
})
"""The exact 4-entry per-tensor byte-map table PR101 uses.

This is **anchor data**, NOT a default. Downstream consumers must derive
their own table per architecture via offline entropy comparison over the
4 strategies per tensor.

Tensors not in the table default to ``zig`` per PR101 source line 279
(``DECODER_BYTE_MAPS.get(idx, "zig")``).
"""


# ── Public schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecoderByteMapsSchema:
    """A per-tensor byte-map strategy table with built-in invariants.

    Attributes
    ----------
    byte_maps:
        Mapping ``tensor_index -> strategy_name``. Each strategy must be
        one of the 4 supported values (see :data:`VALID_BYTE_MAP_STRATEGIES`).
    default_strategy:
        Strategy used for tensor indices NOT in ``byte_maps``. Defaults
        to ``"zig"`` per PR101 source convention.

    Notes
    -----
    The validation invariants (strategy in valid set, non-negative
    int keys) match PR101's wire-format contract. Violations raise
    ``ValueError`` at construction time so downstream encoders see
    structural bugs at table-definition time, not at runtime.
    """

    byte_maps: Mapping[int, ByteMapStrategy]
    default_strategy: ByteMapStrategy = "zig"

    @classmethod
    def from_table(
        cls,
        byte_maps: Mapping[int, str],
        *,
        default_strategy: str = "zig",
    ) -> "DecoderByteMapsSchema":
        """Build a validated schema from a (possibly untyped) table.

        Parameters
        ----------
        byte_maps:
            Mapping ``tensor_index -> strategy_name``.
        default_strategy:
            Strategy for unlisted tensors (defaults to ``"zig"``).

        Raises
        ------
        ValueError
            On any invalid key or strategy.
        TypeError
            On non-mapping input.
        """
        if not isinstance(byte_maps, Mapping):
            raise TypeError(
                f"byte_maps must be a Mapping; got {type(byte_maps)!r}"
            )
        if default_strategy not in VALID_BYTE_MAP_STRATEGIES:
            raise ValueError(
                f"default_strategy {default_strategy!r} not in "
                f"{sorted(VALID_BYTE_MAP_STRATEGIES)}"
            )
        validated: dict[int, ByteMapStrategy] = {}
        for k, v in byte_maps.items():
            if not isinstance(k, int) or isinstance(k, bool):
                raise ValueError(
                    f"byte_maps key must be int; got {type(k)!r}"
                )
            if k < 0:
                raise ValueError(
                    f"byte_maps key must be >= 0; got {k}"
                )
            if not isinstance(v, str):
                raise ValueError(
                    f"byte_maps[{k}] must be str; got {type(v)!r}"
                )
            if v not in VALID_BYTE_MAP_STRATEGIES:
                raise ValueError(
                    f"byte_maps[{k}] = {v!r} not in "
                    f"{sorted(VALID_BYTE_MAP_STRATEGIES)}"
                )
            # Cast OK after membership check.
            validated[k] = v  # type: ignore[assignment]
        return cls(
            byte_maps=MappingProxyType(validated),
            default_strategy=default_strategy,  # type: ignore[arg-type]
        )

    def strategy_for(self, tensor_index: int) -> ByteMapStrategy:
        """Look up the strategy for ``tensor_index``; default otherwise.

        Mirrors PR101 source line 279: ``DECODER_BYTE_MAPS.get(idx, "zig")``.
        """
        if not isinstance(tensor_index, int) or isinstance(tensor_index, bool):
            raise TypeError(
                f"tensor_index must be int; got {type(tensor_index)!r}"
            )
        return self.byte_maps.get(tensor_index, self.default_strategy)


# ── Transducers (bytes <-> int8) ────────────────────────────────────────────


def _zigzag_encode_i8(arr_i8: np.ndarray) -> np.ndarray:
    """Zigzag encode int8 -> uint8 (PR101 inverse of ``zigzag_decode_u8``)."""
    # Standard zigzag: (x << 1) ^ (x >> 7) on signed 8-bit, masked to uint8.
    arr32 = arr_i8.astype(np.int32)
    enc = (arr32 << 1) ^ (arr32 >> 31)  # python-side zigzag on int32
    return (enc & 0xFF).astype(np.uint8)


def _zigzag_decode_u8(arr_u8: np.ndarray) -> np.ndarray:
    """Zigzag decode uint8 -> int8. Mirrors PR101 source line 226-227."""
    arr32 = arr_u8.astype(np.int32)
    return np.where(arr32 % 2 == 0, arr32 // 2, -(arr32 // 2) - 1).astype(
        np.int8
    )


def encode_byte_map(
    arr_i8: np.ndarray,
    strategy: ByteMapStrategy,
) -> bytes:
    """Encode an int8 array under one of the 4 PR101 sign-encoding strategies.

    Parameters
    ----------
    arr_i8:
        1D int8 numpy array (post-quantisation weights).
    strategy:
        One of ``{"zig", "negzig", "twos", "off"}``.

    Returns
    -------
    bytes
        The encoded uint8 byte stream.

    Raises
    ------
    ValueError
        On non-1D input or unknown strategy.
    """
    if strategy not in VALID_BYTE_MAP_STRATEGIES:
        raise ValueError(
            f"unknown byte_map strategy: {strategy!r} "
            f"(valid: {sorted(VALID_BYTE_MAP_STRATEGIES)})"
        )
    arr = np.asarray(arr_i8)
    if arr.dtype != np.int8:
        raise ValueError(f"arr_i8 dtype must be int8; got {arr.dtype}")
    if arr.ndim != 1:
        raise ValueError(f"arr_i8 must be 1D; got shape {arr.shape}")

    if strategy == "zig":
        return _zigzag_encode_i8(arr).tobytes()
    if strategy == "negzig":
        # negzig encodes -x via zigzag. To round-trip:
        # decode_negzig(u8) = -decode_zig(u8) (per PR101 source line 234).
        # Therefore encode_negzig(x) = encode_zig(-x).
        #
        # Note: negzig is NOT a bijection over the full int8 range.
        # The value `-128` and `0` both encode to byte `0` (because
        # `-(-128) mod 256 = 0` under int8 wrap-around). PR101's training
        # script never produces `-128` in the int8 weight quantisation
        # output, which is why this is admissible at PR101's anchor
        # operating point. Downstream consumers using negzig MUST bound
        # their post-quantisation distribution to ``[-127, 127]``.
        neg = (-arr.astype(np.int16)).astype(np.int32)
        enc = (neg << 1) ^ (neg >> 31)
        return (enc & 0xFF).astype(np.uint8).tobytes()
    if strategy == "off":
        # PR101 source line 236: decode = (u8 - 128).view(int8). Therefore
        # encode = (x + 128) & 0xFF.
        return ((arr.astype(np.int16) + 128) & 0xFF).astype(np.uint8).tobytes()
    # strategy == "twos"
    # PR101 source line 238: decode = u8.view(int8). Therefore
    # encode = x.view(uint8) (raw bit reinterpret).
    return arr.view(np.uint8).tobytes()


def decode_byte_map(
    payload: bytes,
    strategy: ByteMapStrategy,
) -> np.ndarray:
    """Decode a uint8 byte stream under a PR101 sign-encoding strategy.

    Mirrors PR101 ``decode_mapped_u8`` (source lines 230-239) exactly.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_byte_map`.
    strategy:
        Must match the encoder strategy.

    Returns
    -------
    np.ndarray
        1D int8 array.

    Raises
    ------
    ValueError
        On unknown strategy.
    TypeError
        On non-bytes input.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"payload must be bytes-like; got {type(payload)!r}"
        )
    if strategy not in VALID_BYTE_MAP_STRATEGIES:
        raise ValueError(
            f"unknown byte_map strategy: {strategy!r} "
            f"(valid: {sorted(VALID_BYTE_MAP_STRATEGIES)})"
        )
    arr_u8 = np.frombuffer(bytes(payload), dtype=np.uint8)

    if strategy == "zig":
        return _zigzag_decode_u8(arr_u8)
    if strategy == "negzig":
        # PR101 source line 234: (-zigzag_decode_u8(arr).astype(int16)).astype(int8)
        return (-_zigzag_decode_u8(arr_u8).astype(np.int16)).astype(np.int8)
    if strategy == "off":
        # PR101 source line 236.
        return (arr_u8.astype(np.int16) - 128).astype(np.int8)
    # strategy == "twos" — PR101 source line 238.
    return arr_u8.view(np.int8).copy()


__all__ = [
    "ByteMapStrategy",
    "DecoderByteMapsSchema",
    "PR101_DECODER_BYTE_MAPS",
    "VALID_BYTE_MAP_STRATEGIES",
    "decode_byte_map",
    "encode_byte_map",
]
