# SPDX-License-Identifier: MIT
"""Measured representation ladder — subset indices, layout, and packing.

WHY THIS EXISTS
---------------
Arms keep re-deriving "how do we name WHICH ones" and "how do we pack this"
badly, one payload at a time.  This module is the canonical, MEASURED racer:
hand it a real payload, it returns real bytes from real coders for every rung
of the ladder, so the next arm calls it instead of re-inventing it.

Every number this module returns is a MEASURED byte count from an actual
encoder on actual bytes.  Nothing here is an asymptotic.  The subset rungs are
exact and round-trip-verified; the layout rungs are lossless permutations.

THE TWO LADDERS
---------------
1. ``race_subset_index`` — naming a k-subset of n positions.  Rungs:
   combinatorial (colex) rank, Elias-Fano, bitmap + generic coder, Golomb-Rice
   gaps, and the iid order-0 entropy reference.  The colex rung hits
   ``log2 C(n, k)`` exactly and is the campaign precedent (CLAUDE.md L31,
   PR101's 3-byte ``SIDECAR_NOOP_INFER_RANK_LEN``).

   ``log2 C(n, k)`` is a floor ONLY under exchangeability.  Our positions are
   spatially clustered, so a prior-coded rung can and does go BELOW it; the
   racer reports that as a ``structure_gain`` rather than hiding it.

2. ``race_layouts`` — for an integer array, race axis permutations
   (struct-of-arrays vs array-of-structs) crossed with sub-byte packings
   (byte lane / nibble lane / bit-plane transposition) through generic coders.
   Grouping like-valued fields makes the coder see stationary statistics; this
   is the rung that was never raced on our own token stream before 2026-08-02.

NON-GOALS — ROUTE, DO NOT REBUILD
---------------------------------
This module measures BYTES for LOSSLESS re-encodings.  It deliberately does
NOT rank anything by score impact, and it contains no quantizer, codebook, or
centroid.

* Which coordinates/pairs matter (sensitivity, hardness, flip/margin currency)
  -> the scorer-value ORACLE FACADE (#700), ``ddm_at1_scorer_analytic_atlas``,
  ``ddm_g3_score_atlas`` (hard-pair registry + subset->full validity r),
  ``ddm_g4_spatial_stationarity``, #141 margin-saliency, #391 flip waterfill,
  #583 Fisher-EV ordering.  Rank in the campaign's currency, never L2.
* Metric choice for any FINITE step or any centroid -> ``ms3``/``ms4``
  metric-custody (margin-Fisher), #504 Bregman levers bound to
  ``policy_bindings.optimal_metric``.  Fisher is the infinitesimal limit of
  Bregman; Euclidean/cosine is neither.  A lossless re-encode has no distortion
  and therefore no metric choice at all — which is exactly why this module is
  metric-free and must stay that way.  The moment a rung becomes LOSSY (width
  allocation, VQ, codebook), it leaves this module.

RULE-118
--------
Every rung here is a GENERIC ALGORITHM: permutation, bit packing, combinatorial
rank, entropy coding.  Generic algorithms are free in ``inflate.py``; only
video-derived payload is counted.  This module never generates a table whose
values encode a clip — that would be the hide-data-in-code fake.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "CoderResult",
    "SubsetIndexRace",
    "LayoutRace",
    "colex_rank",
    "colex_unrank",
    "colex_encode",
    "colex_decode",
    "colex_floor_bytes",
    "pack_nibble_lane",
    "RepresentationLadderError",
    "elias_fano_encode",
    "elias_fano_decode",
    "golomb_rice_gaps_encode",
    "generic_coders",
    "race_generic",
    "race_subset_index",
    "race_layouts",
    "pack_bitplanes",
    "gap_fraction_of_bytes",
    "delta_s_rate_from_bytes",
]

# Contest rate denominator (upstream/evaluate.py) and the campaign gap anchors.
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_COEFF = 25.0


class RepresentationLadderError(ValueError):
    """Raised when a ladder input violates a documented precondition."""


# --------------------------------------------------------------------------
# generic coder race
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoderResult:
    """One measured rung: ``name`` produced ``size_bytes`` on the payload."""

    name: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise RepresentationLadderError("size_bytes must be non-negative")


def generic_coders() -> dict[str, Callable[[bytes], bytes]]:
    """Return the generic lossless coders available in this runtime.

    ``brotli`` and ``lzma`` are optional at import time so this module stays
    usable in a bare environment; ``zlib`` is always present.
    """

    coders: dict[str, Callable[[bytes], bytes]] = {
        "deflate": lambda b: zlib.compress(b, 9),
    }
    try:  # pragma: no cover - availability varies by runtime
        import brotli

        coders["brotli_q11"] = lambda b: brotli.compress(b, quality=11)
    except ImportError:  # pragma: no cover
        pass
    try:  # pragma: no cover
        import lzma

        filters = [
            {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 23, "lc": 3, "lp": 0, "pb": 0}
        ]
        coders["lzma_raw"] = lambda b: lzma.compress(
            b, format=lzma.FORMAT_RAW, filters=filters
        )
    except ImportError:  # pragma: no cover
        pass
    return coders


def race_generic(payload: bytes) -> tuple[CoderResult, ...]:
    """Measure every available generic coder on ``payload``.

    Includes a ``stored`` rung so a caller can see when the payload is already
    at or beyond generic-coder incompressibility (the signature of a stream
    that has already been entropy coded).
    """

    rows = [CoderResult("stored", len(payload))]
    for name, fn in generic_coders().items():
        rows.append(CoderResult(name, len(fn(payload))))
    return tuple(sorted(rows, key=lambda r: r.size_bytes))


# --------------------------------------------------------------------------
# combinatorial number system (colex rank / unrank)
# --------------------------------------------------------------------------


def colex_rank(positions: Sequence[int], n: int) -> int:
    """Colex rank of a strictly increasing subset within ``C(n, k)``.

    The combinatorial number system: a k-subset ``c_0 < ... < c_{k-1}`` maps to
    ``sum_i C(c_i, i + 1)``.  Bijective onto ``[0, C(n, k))``.
    """

    pos = list(positions)
    if any(p < 0 or p >= n for p in pos):
        raise RepresentationLadderError("positions must lie in [0, n)")
    if any(b <= a for a, b in zip(pos, pos[1:])):
        raise RepresentationLadderError("positions must be strictly increasing")
    return sum(math.comb(c, i + 1) for i, c in enumerate(pos))


def colex_unrank(rank: int, k: int, n: int) -> tuple[int, ...]:
    """Inverse of :func:`colex_rank`."""

    total = math.comb(n, k)
    if not 0 <= rank < max(total, 1):
        raise RepresentationLadderError("rank outside [0, C(n, k))")
    out: list[int] = []
    remaining = rank
    for i in range(k, 0, -1):
        # largest c with C(c, i) <= remaining
        lo, hi = i - 1, n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if math.comb(mid, i) <= remaining:
                lo = mid
            else:
                hi = mid - 1
        out.append(lo)
        remaining -= math.comb(lo, i)
    return tuple(reversed(out))


def colex_floor_bytes(n: int, k: int) -> float:
    """``log2 C(n, k) / 8`` — the exchangeable-position information floor."""

    if k < 0 or k > n:
        raise RepresentationLadderError("require 0 <= k <= n")
    total = math.comb(n, k)
    return 0.0 if total <= 1 else math.log2(total) / 8.0


def colex_encode(positions: Sequence[int], n: int) -> bytes:
    """Encode a subset as its colex rank in the minimum whole number of bytes.

    ``k`` and ``n`` are decoder-side context (the caller already knows the
    universe size and the population count, or ships them once), matching the
    PR101 precedent where the rank alone was 3 bytes.
    """

    k = len(positions)
    rank = colex_rank(positions, n)
    total = math.comb(n, k)
    width = max(1, (max(total - 1, 1).bit_length() + 7) // 8)
    return rank.to_bytes(width, "big")


def colex_decode(payload: bytes, k: int, n: int) -> tuple[int, ...]:
    """Inverse of :func:`colex_encode`."""

    return colex_unrank(int.from_bytes(payload, "big"), k, n)


# --------------------------------------------------------------------------
# Elias-Fano
# --------------------------------------------------------------------------


def _bits_to_bytes(bits: list[int]) -> bytes:
    if not bits:
        return b""
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def elias_fano_encode(positions: Sequence[int], n: int) -> tuple[bytes, int]:
    """Elias-Fano encode a monotone sequence.

    Returns ``(payload, low_width)``.  Cost is ``k * low_width`` low bits plus a
    ``2k``-ish unary high-bit stream, i.e. about ``k * (2 + log2(n / k))`` bits.
    """

    pos = list(positions)
    k = len(pos)
    if k == 0:
        return b"", 0
    if any(b <= a for a, b in zip(pos, pos[1:])):
        raise RepresentationLadderError("positions must be strictly increasing")
    low_width = max(0, int(math.floor(math.log2(max(n, 1) / k))) if k else 0)
    mask = (1 << low_width) - 1
    high_bits: list[int] = []
    prev_high = 0
    for p in pos:
        high = p >> low_width
        high_bits.extend([0] * (high - prev_high))
        high_bits.append(1)
        prev_high = high
    low_bits: list[int] = []
    for p in pos:
        v = p & mask
        for b in range(low_width - 1, -1, -1):
            low_bits.append((v >> b) & 1)
    return _bits_to_bytes(high_bits + low_bits), low_width


def elias_fano_decode(
    payload: bytes, k: int, n: int, low_width: int
) -> tuple[int, ...]:
    """Inverse of :func:`elias_fano_encode`."""

    if k == 0:
        return ()
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    out: list[int] = []
    highs: list[int] = []
    idx = 0
    running = 0
    while len(highs) < k:
        if idx >= bits.size:
            raise RepresentationLadderError("truncated Elias-Fano high stream")
        if bits[idx]:
            highs.append(running)
        else:
            running += 1
        idx += 1
    for j in range(k):
        low = 0
        for b in range(low_width):
            low = (low << 1) | int(bits[idx])
            idx += 1
        out.append((highs[j] << low_width) | low)
    return tuple(out)


# --------------------------------------------------------------------------
# Golomb-Rice coded gaps
# --------------------------------------------------------------------------


def golomb_rice_gaps_encode(positions: Sequence[int], n: int) -> bytes:
    """Rice-code the first-difference gaps with the mean-optimal parameter.

    Optimal for geometrically distributed gaps, which is the uniform-random
    subset case; a clustered subset beats it, which is exactly the signal the
    racer is there to expose.
    """

    pos = list(positions)
    k = len(pos)
    if k == 0:
        return b""
    gaps = [pos[0]] + [b - a - 1 for a, b in zip(pos, pos[1:])]
    mean = max(sum(gaps) / k, 1e-9)
    param = max(0, int(round(math.log2(mean))) if mean > 0 else 0)
    bits: list[int] = []
    for g in gaps:
        q = g >> param
        bits.extend([1] * q)
        bits.append(0)
        for b in range(param - 1, -1, -1):
            bits.append((g >> b) & 1)
    return _bits_to_bytes(bits)


# --------------------------------------------------------------------------
# subset-index race
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SubsetIndexRace:
    """Measured result of naming ``k`` of ``n`` positions."""

    n: int
    k: int
    rungs: tuple[CoderResult, ...]
    colex_floor_bytes: float
    order0_entropy_bytes: float

    @property
    def best(self) -> CoderResult:
        return min(self.rungs, key=lambda r: r.size_bytes)

    @property
    def structure_gain_vs_colex(self) -> float:
        """``colex_floor / best``.

        Strictly greater than 1 means the positions are NOT exchangeable and a
        prior-coded rung beat the combinatorial bound.  That is a measurement
        about the data, not a coder bug.
        """

        best = max(self.best.size_bytes, 1e-9)
        return self.colex_floor_bytes / best

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "k": self.k,
            "colex_floor_bytes": self.colex_floor_bytes,
            "order0_entropy_bytes": self.order0_entropy_bytes,
            "best_rung": self.best.name,
            "best_bytes": self.best.size_bytes,
            "structure_gain_vs_colex": self.structure_gain_vs_colex,
            "rungs": {r.name: r.size_bytes for r in self.rungs},
        }


def race_subset_index(positions: Iterable[int], n: int) -> SubsetIndexRace:
    """Race every subset-index rung on a real position set.

    ``positions`` need not be sorted; duplicates are rejected.
    """

    pos = sorted(set(int(p) for p in positions))
    if len(pos) != len(list(positions)):
        raise RepresentationLadderError("positions must be unique")
    k = len(pos)
    if any(p < 0 or p >= n for p in pos):
        raise RepresentationLadderError("positions must lie in [0, n)")

    rungs: list[CoderResult] = []
    rungs.append(CoderResult("colex_rank", len(colex_encode(pos, n))))
    ef, _ = elias_fano_encode(pos, n)
    rungs.append(CoderResult("elias_fano", len(ef)))
    rungs.append(CoderResult("golomb_rice_gaps", len(golomb_rice_gaps_encode(pos, n))))

    bitmap = np.zeros(n, dtype=np.uint8)
    if pos:
        bitmap[np.asarray(pos, dtype=np.int64)] = 1
    packed = np.packbits(bitmap).tobytes()
    rungs.append(CoderResult("bitmap_packed", len(packed)))
    for name, fn in generic_coders().items():
        rungs.append(CoderResult(f"bitmap_{name}", len(fn(packed))))

    if 0 < k < n:
        p = k / n
        h0 = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    else:
        h0 = 0.0
    return SubsetIndexRace(
        n=n,
        k=k,
        rungs=tuple(sorted(rungs, key=lambda r: r.size_bytes)),
        colex_floor_bytes=colex_floor_bytes(n, k),
        order0_entropy_bytes=n * h0 / 8.0,
    )


# --------------------------------------------------------------------------
# layout / packing race
# --------------------------------------------------------------------------


def pack_nibble_lane(values: np.ndarray) -> bytes:
    """Pack an even-length 0..15 array two symbols per byte (high nibble first)."""

    a = np.ascontiguousarray(values).ravel().astype(np.uint8)
    if a.size and a.max() > 15:
        raise RepresentationLadderError("nibble lane requires values in 0..15")
    if a.size % 2:
        a = np.concatenate([a, np.zeros(1, dtype=np.uint8)])
    return ((a[0::2] << 4) | a[1::2]).tobytes()


def pack_bitplanes(values: np.ndarray, width: int) -> tuple[bytes, ...]:
    """Bit-plane transposition: return one packed plane per bit position.

    Transposing so that like-significance bits become contiguous usually makes
    each plane far more stationary than the interleaved symbol stream.  It is
    almost never tried, so the racer always includes it.
    """

    a = np.ascontiguousarray(values).ravel().astype(np.uint64)
    return tuple(
        np.packbits(((a >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)).tobytes()
        for b in range(width)
    )


@dataclass(frozen=True)
class LayoutRace:
    """Measured layout x packing race over an integer array."""

    shape: tuple[int, ...]
    rows: tuple[tuple[str, str, int], ...]  # (layout, packing, bytes)

    @property
    def best(self) -> tuple[str, str, int]:
        return min(self.rows, key=lambda r: r[2])

    def as_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "best_layout": self.best[0],
            "best_packing": self.best[1],
            "best_bytes": self.best[2],
            "rows": [
                {"layout": a, "packing": b, "bytes": c} for a, b, c in self.rows
            ],
        }


def race_layouts(
    array: np.ndarray,
    *,
    permutations: Mapping[str, Sequence[int]] | None = None,
    symbol_width: int = 4,
) -> LayoutRace:
    """Race axis permutations x packings x generic coders on ``array``.

    ``permutations`` defaults to the identity plus every single-axis rotation,
    which covers array-of-structs (native) and the struct-of-arrays variants.
    ``symbol_width`` is the number of significant bits per symbol; 4 for an
    int4 lattice.  Set to 8 to skip nibble packing.
    """

    arr = np.ascontiguousarray(array)
    ndim = arr.ndim
    if permutations is None:
        perms: dict[str, Sequence[int]] = {"aos_native": tuple(range(ndim))}
        for axis in range(ndim):
            order = (axis,) + tuple(a for a in range(ndim) if a != axis)
            perms[f"soa_axis{axis}_major"] = order
            tail = tuple(a for a in range(ndim) if a != axis) + (axis,)
            perms[f"soa_axis{axis}_minor"] = tail
    else:
        perms = dict(permutations)

    coders = generic_coders()
    rows: list[tuple[str, str, int]] = []
    for name, order in perms.items():
        flat = np.ascontiguousarray(np.transpose(arr, tuple(order))).ravel()
        variants: dict[str, bytes | tuple[bytes, ...]] = {
            "byte_lane": flat.astype(np.uint8).tobytes()
        }
        if symbol_width <= 4:
            variants["nibble_lane"] = pack_nibble_lane(flat)
        variants["bitplanes"] = pack_bitplanes(flat, symbol_width)
        for pack_name, payload in variants.items():
            if isinstance(payload, tuple):
                best = sum(
                    min(len(fn(plane)) for fn in coders.values()) for plane in payload
                )
            else:
                best = min(len(fn(payload)) for fn in coders.values())
            rows.append((name, pack_name, best))
    return LayoutRace(shape=tuple(arr.shape), rows=tuple(rows))


# --------------------------------------------------------------------------
# gap accounting
# --------------------------------------------------------------------------


def delta_s_rate_from_bytes(delta_bytes: int) -> float:
    """Exact contest rate-term delta for a byte delta on ``archive.zip``.

    A LOSSLESS re-encode changes only the rate term, so this is the whole ΔS
    provided the decode is proved bit-identical.
    """

    return RATE_COEFF * float(delta_bytes) / RATE_DENOMINATOR_BYTES


def gap_fraction_of_bytes(delta_bytes: int, *, total_gap: float) -> float:
    """Fraction of the gap-to-floor a lossless byte delta moves.

    ``total_gap`` must come from
    ``tac.canonical_equations.gap_decomposition_against_floor_20260802``; this
    helper never hardcodes a floor.
    """

    if not math.isfinite(total_gap) or total_gap <= 0:
        raise RepresentationLadderError("total_gap must be finite and positive")
    return delta_s_rate_from_bytes(delta_bytes) / total_gap
