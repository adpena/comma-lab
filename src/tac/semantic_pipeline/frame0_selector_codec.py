# SPDX-License-Identifier: MIT
"""Compress-side encoder for the shipped F26 frame-0 selector blob.

The shipped runtime is DECODE-ONLY: ``submissions/semantic_joint_ctxmix/runtime/
frame0_selector.py`` carries ``decode_selector`` and ``apply_pixel_mode`` but no
inverse, so no arm could ever write a re-selected selector into an archive.  This
module is that inverse and nothing else.

THE FORMAT, READ OUT OF THE DECODER RATHER THAN RECALLED
--------------------------------------------------------
``decode_selector`` (frame0_selector.py:91-110) accepts exactly::

    struct.Struct("<4sBH")      magic b"F0E1", version 1, active count k   (7 B)
    rank                        big-endian, ceil(bit_length(C(600,k)-1)/8) B
    labels                      3 bits per active position, MSB-first, zero-padded

* the rank is the colex combinatorial rank of the SORTED active positions:
  ``_combination_unrank`` (:48-67) subtracts ``comb(positions[w-1], w)`` for
  ``w = k..1`` and refuses a non-zero remainder, so the forward map is
  ``rank = sum_i C(p_i, i+1)`` over ascending ``p_0 < ... < p_{k-1}``;
* each stored label is ``mode_index - 1`` and ``_unpack_labels`` (:85-87) refuses
  ``label >= len(SPARSE_PIXEL_MODES) - 1``, so mode 0 (IDENTITY) is NOT
  representable as a label -- identity is expressed by ABSENCE from the position
  set.  An all-identity selector therefore has no encoding at all (the header
  refuses ``count == 0`` at :96) and this module says so rather than inventing one;
* the label padding bits MUST be zero (:74-75).

WHY THE ENCODER VERIFIES ITSELF THROUGH THE SHIPPED DECODER
-----------------------------------------------------------
A round trip against a locally re-implemented decoder proves only that this file
agrees with itself.  ``encode_selector`` therefore decodes its own output through
the SHIPPED ``decode_selector`` and refuses to return bytes whose parse-back
differs from the requested choices.  The verification is on by default; turning it
off is an explicit caller decision, never a default.

AUTHORITY
---------
This module writes bytes.  It makes no score claim: only ``upstream/evaluate.py``
on the exact archive bytes is a score.
"""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: ``runtime/frame0_selector.py`` SPARSE_MAGIC / SPARSE_VERSION / _HEADER.
SELECTOR_MAGIC = b"F0E1"
SELECTOR_VERSION = 1
SELECTOR_HEADER = struct.Struct("<4sBH")

#: ``decode_selector`` builds a 600-entry choice vector (frame0_selector.py:108).
FRAME_COUNT = 600

#: ``len(SPARSE_PIXEL_MODES)`` -- identity plus seven integer pixel operations.
MODE_COUNT = 8

#: The 5-byte prefix ``residual_archive.SPARSE_SELECTOR_PREFIX`` re-attaches when
#: it lifts the selector out of the stored carrier tail (residual_archive.py:53).
STORED_PREFIX = SELECTOR_MAGIC + bytes((SELECTOR_VERSION,))

_REPO = Path(__file__).resolve().parents[3]
_SHIPPED_DECODER = (
    _REPO / "submissions" / "semantic_joint_ctxmix" / "runtime" / "frame0_selector.py"
)


class Frame0SelectorEncodeError(ValueError):
    """The requested selector cannot be written in the shipped format."""


def load_shipped_decoder(path: Path | None = None) -> Any:
    """Import the frozen ``frame0_selector`` module by path, under a private name.

    The shipped module is loaded standalone (it imports only stdlib plus numpy) and
    registered under a name of this module's own choosing, so it never shadows the
    ``runtime`` package a caller may already have imported from another generation.
    Registration is required, not cosmetic: ``frame0_selector`` defines a frozen
    dataclass, and ``dataclasses`` reads ``sys.modules[cls.__module__]`` while
    processing it.
    """
    source = Path(path) if path is not None else _SHIPPED_DECODER
    if not source.is_file():
        raise Frame0SelectorEncodeError(
            f"shipped frame-0 selector decoder is absent at {source}; refusing to "
            "encode bytes that cannot be verified against the receiver"
        )
    key = f"tac_shipped_frame0_selector::{source}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, source)
    if spec is None or spec.loader is None:
        raise Frame0SelectorEncodeError(f"cannot load shipped decoder from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(key, None)
        raise
    return module


def selector_blob_length(active_count: int, frames: int = FRAME_COUNT) -> int:
    """Exact encoded length in bytes -- the receiver's own closed formula.

    ``7 + ceil(bit_length(C(frames, k) - 1) / 8) + ceil(3k / 8)``, read off
    ``decode_selector`` (frame0_selector.py:98-101).  The control that makes this
    the receiver's formula rather than a model: at ``k = 5`` it returns 14, which
    is exactly what the afr1 archive carries.
    """
    count = int(active_count)
    if not 1 <= count <= frames:
        raise Frame0SelectorEncodeError(
            f"active count {count} is outside the encodable range 1..{frames}"
        )
    rank_bytes = ((math.comb(frames, count) - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    return SELECTOR_HEADER.size + rank_bytes + label_bytes


def combination_rank(positions: Sequence[int], frames: int = FRAME_COUNT) -> int:
    """Colex rank of a strictly ascending position set -- inverse of ``_combination_unrank``."""
    values = [int(p) for p in positions]
    if not values:
        raise Frame0SelectorEncodeError("cannot rank an empty position set")
    if any(not 0 <= p < frames for p in values):
        raise Frame0SelectorEncodeError(
            f"selector position out of range 0..{frames - 1}: {values}"
        )
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise Frame0SelectorEncodeError(
            "selector positions must be strictly ascending and unique"
        )
    return sum(math.comb(p, index + 1) for index, p in enumerate(values))


def pack_labels(labels: Sequence[int]) -> bytes:
    """Pack 3-bit labels MSB-first with zero padding -- inverse of ``_unpack_labels``."""
    values = [int(v) for v in labels]
    if not values:
        raise Frame0SelectorEncodeError("cannot pack an empty label set")
    if any(not 0 <= v < MODE_COUNT - 1 for v in values):
        raise Frame0SelectorEncodeError(
            f"stored label out of range 0..{MODE_COUNT - 2}: {values}"
        )
    bits: list[int] = []
    for value in values:
        bits.extend((value >> shift) & 1 for shift in range(2, -1, -1))
    bits.extend([0] * (-len(bits) % 8))
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()


def encode_selector(
    choices: Sequence[int] | np.ndarray,
    *,
    frames: int = FRAME_COUNT,
    verify: bool = True,
    decoder_path: Path | None = None,
) -> bytes:
    """Write the sparse selector blob the shipped receiver parses.

    ``choices`` is one mode index per frame; 0 means IDENTITY and is expressed by
    absence.  With ``verify`` (the default) the bytes are decoded back through the
    SHIPPED ``decode_selector`` and refused unless the parse-back equals
    ``choices`` exactly, so this function cannot hand out bytes it has not proved.
    """
    values = np.asarray(choices, dtype=np.int64)
    if values.ndim != 1 or values.size != frames:
        raise Frame0SelectorEncodeError(
            f"choices must be a ({frames},) vector, got shape {values.shape}"
        )
    if values.min(initial=0) < 0 or values.max(initial=0) >= MODE_COUNT:
        raise Frame0SelectorEncodeError(
            f"choices must lie in 0..{MODE_COUNT - 1}, got "
            f"[{int(values.min(initial=0))}, {int(values.max(initial=0))}]"
        )
    positions = np.flatnonzero(values).astype(np.int64)
    count = int(positions.size)
    if count == 0:
        raise Frame0SelectorEncodeError(
            "an all-identity selector has no encoding: decode_selector refuses "
            "count == 0 (frame0_selector.py:96), so a body with no active pair "
            "must omit the selector blob rather than carry an empty one"
        )
    rank = combination_rank(positions.tolist(), frames=frames)
    limit = math.comb(frames, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    payload = (
        SELECTOR_HEADER.pack(SELECTOR_MAGIC, SELECTOR_VERSION, count)
        + rank.to_bytes(rank_bytes, "big")
        + pack_labels((values[positions] - 1).tolist())
    )
    expected = selector_blob_length(count, frames=frames)
    if len(payload) != expected:
        raise Frame0SelectorEncodeError(
            f"encoded {len(payload)} B but the receiver's formula says {expected} B"
        )
    if verify:
        module = load_shipped_decoder(decoder_path)
        try:
            _modes, parsed = module.decode_selector(payload)
        except Exception as error:  # the receiver refused our own bytes
            raise Frame0SelectorEncodeError(
                f"shipped decoder refused the encoded selector: {error}"
            ) from error
        if not np.array_equal(np.asarray(parsed, dtype=np.int64), values):
            differing = int(np.count_nonzero(np.asarray(parsed, dtype=np.int64) != values))
            raise Frame0SelectorEncodeError(
                f"encoded selector parses back to {differing} differing choices; "
                "refusing to return unverified bytes"
            )
    return payload


def stored_tail(blob: bytes) -> bytes:
    """The bytes the archive actually stores: the blob minus its 5-byte prefix.

    ``residual_archive._decode_rx1_models`` (:247) rebuilds the selector as
    ``SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]``, so the magic and
    version are implied by the container and never transmitted twice.
    """
    if not blob.startswith(STORED_PREFIX):
        raise Frame0SelectorEncodeError("selector blob does not carry the F0E1 prefix")
    return blob[len(STORED_PREFIX) :]
