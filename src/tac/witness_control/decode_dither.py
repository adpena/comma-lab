# SPDX-License-Identifier: MIT
"""B19 decode-side seeded dither: the minimal transform for the uint8 quantization site.

LAW + BAND PROVENANCE (T5 crucible, DRAFT_OPTIMAL_STACK_v6 SS0.3 + SS5 + SS7c): the P-DZ
deadzone census measured 1.5795495775e-3 d_seg-equivalent of flip mass (38.366% of subset
flip mass, 43.70% of the n600 decoded residual) locked SUB-QUANTUM under the decode's
round-to-uint8 -- far-range lane rows 176-224, horizon shadow edges, hood boundary. B19 is
the #149-class lever in its cheapest deterministic form: seeded ordered dither injected at
the uint8 quantization of the byte-close/inflate decode path (v6 SS5: "0 archive bytes,
~15-25 LOC, no train-path coupling, byte-close-selectable"). Dither converts sub-quantum
boundary placement into deterministic-in-seed multi-quantum spatial texture; the scorer's
bilinear downsample (874x1164 -> 512x384) then averages it back toward the sub-quantum
mean (dither + averaging = effective bit-depth extension).

GATE (v6 SS7c P-DITHER, pre-registered): fire Delta d_seg <= -1e-5 at unchanged bytes;
kill THIS FORM Delta d_seg >= 0. Gate driver: tools/witness_dither_decode_ab.py.

RULE-118 / NO-FAKE BOUNDARY: everything here is a GENERIC deterministic algorithm --
the Bayer matrix is recursive closed form, the roll offsets come from PCG64(seed, pair,
frame); seed + amplitude are config SCALARS. NO video-derived table, NO learned content.
Archive bytes are unchanged by construction (decode-side; the rate term sizes archive.zip
only, upstream/evaluate.py:63).

All consumers are advisory-local until a byte-closed exact-eval row exists; the frontier
pointer (0.19110) moves only through upstream/evaluate.py.
"""
from __future__ import annotations

import numpy as np

#: pre-registered A/B config (probe_tau2_dither_20260708.md): Bayer-8, amp=1.0 quantum.
DEFAULT_MODE = "bayer8"
DEFAULT_AMP = 1.0
DEFAULT_SEED = 0xB19
_MODES = ("bayer8", "white")


def bayer_matrix(n: int) -> np.ndarray:
    """n x n Bayer ordered-dither index matrix (n a power of 2), values 0..n*n-1.

    Recursive closed form: B_{2n} = [[4B+0, 4B+2], [4B+3, 4B+1]]. Generic algorithm
    (rule-118 free); deterministic.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"bayer_matrix size must be a power of 2 >= 1; got {n}")
    if n == 1:
        return np.zeros((1, 1), dtype=np.int64)
    m = bayer_matrix(n // 2)
    return np.block([[4 * m + 0, 4 * m + 2], [4 * m + 3, 4 * m + 1]])


def dither_unit_field(
    h: int, w: int, mode: str, seed: int, pair_idx: int, frame_idx: int
) -> np.ndarray:
    """Deterministic (h, w, 3) float32 unit dither field in [0, 1).

    ``bayer8``: the 8x8 Bayer matrix cell-centered ((B + 0.5)/64), tiled, with a seeded
    per-(pair, frame, channel) integer roll offset (decorrelates channels/pairs while
    keeping the ordered high-frequency structure the scorer's downsample averages best).
    ``white``: PCG64 uniform noise. Both are keyed by (seed, pair_idx, frame_idx) ONLY --
    same inputs, same field, any host (SeedSequence entropy pooling is platform-stable).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")
    rng = np.random.default_rng([int(seed), int(pair_idx), int(frame_idx)])
    if mode == "white":
        return rng.random((h, w, 3), dtype=np.float32)
    b = (bayer_matrix(8).astype(np.float32) + 0.5) / 64.0  # cell-centered in (0,1)
    tile = np.tile(b, (h // 8 + 1, w // 8 + 1))[:h, :w]
    field = np.empty((h, w, 3), dtype=np.float32)
    for c in range(3):
        dy, dx = (int(x) for x in rng.integers(0, 8, size=2))
        field[..., c] = np.roll(np.roll(tile, dy, axis=0), dx, axis=1)
    return field


def dither_offset(
    h: int,
    w: int,
    amp: float = DEFAULT_AMP,
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    pair_idx: int = 0,
    frame_idx: int = 0,
) -> np.ndarray:
    """The additive pre-round offset: amp * (u - 0.5), (h, w, 3) float32.

    Injected between the decode's bicubic upsample and its round-to-uint8
    (``inflate._R``): ``uint8 = clip(round(up + dither_offset(...)), 0, 255)``.
    amp = 1.0 spans +/-0.5 = exactly the rounding deadzone (the pre-registered A/B
    amplitude); amp = 0.0 is the identity (OFF-identical, byte-close-selectable).
    """
    if amp == 0.0:
        return np.zeros((h, w, 3), dtype=np.float32)
    u = dither_unit_field(h, w, mode, seed, pair_idx, frame_idx)
    return (float(amp) * (u - np.float32(0.5))).astype(np.float32)
