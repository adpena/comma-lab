# Source Generated with Decompyle++
# File: seg_sparse_m10_codec.cpython-312.pyc (Python 3.12)

__doc__ = 'Sparse-M10 seg_targets codec: M4 / M5 / M7 with sparse-M10 override on frames 2+.\n\nExtends the shipped ma1 a1 codec (frame 0 M4, frame 1 M5, frames 2+ M7) with a\nsparse-M10 context override on frames 2+. For each frame-2+ pixel we compute a\n10-dim context (M7 base + diag_tltl, left_left, top_top_top). If that M10 ctx\nlanded on the fired-list at encode time we use its empirical row (shipped as a\ndelta-indexlist); otherwise we fall back to the dense M7 row.\n\nPrune criterion (per `feedback_ctw_prune_0p01pct_retains_98pct.md`):\n    ship row iff Σ_s c[ctx,s] × log2( P_emp_q[ctx,s] / P_M7_parent[s] ) > 64 bits\n\nFire rate on ma1: 166 / 9.77M (0.0017%). Teacher-forced Δ_net = −8,277 B vs the\ndeployed ma1 a1 archive.\n\nThree new features, all raster-causal:\n  - diag_tltl    : (y-2, x-2)\n  - left_left    : (y,   x-2)\n  - top_top_top  : (y-3, x)\n\nBlob layout (little-endian, extends the shipped a1 layout):\n\n  <HHHBBB>   n_pairs, H, W, precision, peel_class, mask_format\n  <I>        mask_payload_len\n  <mask_payload>\n  <HHI>     spatial_size, m5_size, m6_size\n  spatial_freqs     uint16[5^4, N_SYM]\n  m5_freqs          uint16[5^5, N_SYM]\n  m6_freqs          uint16[5^7, N_SYM]\n  # sparse-M10 extension section:\n  <B>        m10_version  (=1)\n  <B>        n_feats\n  <B>*n_feats feat_ids   (0=diag_tltl, 1=left_left, 2=top_top_top, 3=prev_prev_prev)\n  <H>        threshold_bits_q8 (informative; decoder ignores)\n  <I>        m10_compressed_len\n  <m10_compressed_bytes>   # zlib(-15) of: <II>(n_fired, n_ctx) || uint32[n_fired] deltas\n                           #              || uint16[n_fired, N_SYM] freqs\n  <I>        bs_len\n  <bitstream>\n\nDecoder entry point (matches shipped signature for drop-in inflate.py swap):\n\n    decode_seg_split_m10(path: Path) -> np.ndarray           # returns (n_pairs, H, W) uint8\n'
from __future__ import annotations
import argparse
import struct
import sys
import time
import zlib
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent.parent
from range_coder import RangeDecoder, RangeEncoder
from encode_seg_c2split_purepy import BORDER, MASK_FORMAT_BZ2_PACKBITS, MASK_FORMAT_BZ2_RAW, MASK_FORMAT_LZMA_PACKBITS, MASK_FORMAT_LZMA_RAW, N_CLASSES, PRECISION, compute_spatial_contexts, decode_mask_payload, encode_mask_best, load_seg_targets_lzma, make_remap_tables, quantize_freqs
from encode_seg_c2split_a1_purepy import build_spatial_counts_m4, build_temporal_counts_m5, build_temporal_counts_m6, compute_tt
from encode_seg_c2split_tr_purepy import compute_tr
N_SYM = N_CLASSES - 1
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
# WARNING: Decompyle incomplete
