"""SISTER inflate-side decoder for FEC10 hybrid adaptive-blend selector stream.

Re-exports ``decode_fec10_hybrid_selector`` from the canonical encoder module
so encoder + decoder cannot drift (single source of truth per CLAUDE.md +
sister to ``fec8_markov_decoder.py`` pattern).

# SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow the encoder module to live as canonical implementation.
_HERE = Path(__file__).resolve().parent
_ENCODER_DIR = _HERE.parent / "encoder"
if str(_ENCODER_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODER_DIR))

from build_pr101_frame_exploit_selector_packet_fec10_hybrid import (  # type: ignore[import-not-found]  # noqa: E402
    ALPHA_DEFAULT,
    FECA_MAGIC,
    FECA_VARIANT_ADAPTIVE_BLEND,
    PALETTE_K,
    decode_fec10_hybrid_selector,
)

__all__ = [
    "ALPHA_DEFAULT",
    "FECA_MAGIC",
    "FECA_VARIANT_ADAPTIVE_BLEND",
    "PALETTE_K",
    "decode_fec10_hybrid_selector",
]
