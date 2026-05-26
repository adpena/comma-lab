"""SISTER inflate-side decoder for FEC7 adaptive 0-order range-coded selector stream.

This is the inflate-time consumer; it re-exports ``decode_fec7_arith_selector``
from the canonical encoder module so the encoder + decoder cannot drift
(single source of truth per CLAUDE.md "Beauty, simplicity, and developer experience"
+ Catalog #287 canonical-helper discipline).

To swap an existing FEC6 ``inflate.py`` to the FEC7 variant: replace
``unpack_fec6_fixed_huffman_codes`` callsite with::

    from fec7_arith_decoder import decode_fec7_arith_selector
    if selector_payload[:4] == b"FEC7":
        codes = decode_fec7_arith_selector(selector_payload)
    elif selector_payload[:4] == b"FEC6":
        ... existing FEC6 path ...

The K=16 mode palette (``FEC6_FIXED_K16_MODE_IDS``) is shared between FEC6 and
FEC7 wire formats; only the entropy-coding layer differs.

# SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow the encoder module to live as the canonical implementation.
_HERE = Path(__file__).resolve().parent
_ENCODER_DIR = _HERE.parent / "encoder"
if str(_ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODER_DIR))

from build_pr101_frame_exploit_selector_packet_arith import (  # type: ignore[import-not-found]  # noqa: E402
    FEC7_MAGIC,
    PALETTE_K,
    decode_fec7_arith_selector,
)

__all__ = [
    "FEC7_MAGIC",
    "PALETTE_K",
    "decode_fec7_arith_selector",
]
