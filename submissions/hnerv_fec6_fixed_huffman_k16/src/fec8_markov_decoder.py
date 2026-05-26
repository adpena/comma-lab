"""SISTER inflate-side decoder for FEC8 1st-order Markov context-coded selector stream.

This is the inflate-time consumer; it re-exports ``decode_fec8_markov_selector``
from the canonical encoder module so the encoder + decoder cannot drift
(single source of truth per CLAUDE.md "Beauty, simplicity, and developer experience"
+ Catalog #287 canonical-helper discipline; matches the FEC7 sister pattern).

To swap an existing FEC6/FEC7 ``inflate.py`` to add FEC8 support: extend the
selector_payload magic dispatch with::

    from fec8_markov_decoder import decode_fec8_markov_selector
    magic = selector_payload[:4]
    if magic == b"FEC8":
        codes = decode_fec8_markov_selector(selector_payload)
    elif magic == b"FEC7":
        ... existing FEC7 path ...
    elif magic == b"FEC6":
        ... existing FEC6 path ...

The K=16 mode palette (``FEC6_FIXED_K16_MODE_IDS``) is shared between FEC6, FEC7,
and FEC8 wire formats; only the entropy-coding layer differs.

The FEC8 variant byte at offset 4..5 selects between the STATIC seed (b"\\x00\\x01"
— hard-coded ``EMPIRICAL_TRANSITION_COUNTS`` table baked into source) and the
ADAPTIVE seed (b"\\x00\\x02" — uniform Laplace prior, online convergence). Both
variants are decoded by the same ``decode_fec8_markov_selector`` entry point.

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

from build_pr101_frame_exploit_selector_packet_markov import (  # type: ignore[import-not-found]  # noqa: E402
    FEC8_MAGIC,
    FEC8_VARIANT_ADAPTIVE,
    FEC8_VARIANT_STATIC,
    PALETTE_K,
    decode_fec8_markov_selector,
)

__all__ = [
    "FEC8_MAGIC",
    "FEC8_VARIANT_STATIC",
    "FEC8_VARIANT_ADAPTIVE",
    "PALETTE_K",
    "decode_fec8_markov_selector",
]
