# SPDX-License-Identifier: MIT
"""Curve-relative signed-offset residual coder (v8 T2 Lever-2; DEFAULT-OFF).

The v8 rate ledger's named enemy is the **residual sidecar** (the 17-30% of each inter-class
separatrix the parsimonious generator misses): 0.079 S = 1.9x the whole remaining gap to
sub-0.15 (`v8_movable_residual_rollup_20260709.md`).  The generic residual coder pays for the
uncovered pixels as a **2-D scattered point set** (~0.4-0.6 B/px, near its 2-D coordinate
entropy; chain-code buys only -13%).  S1's deep-math (position_S1 s4) derives that the residual
is NOT intrinsically 2-D: it is a **signed NORMAL OFFSET n(s)** of the true boundary vs the
generator curve, a 1-D signal along arc-length s.  Coding n(s) instead of absolute (row,col) is a
**2-D -> 1-D dimensional reduction**: n has a tiny alphabet (+/- a few px) and s is dense/sequential,
so the offset stream compresses far below the absolute-coordinate baseline.

This module is the buildable, bit-exact realization of that chart transform + coder, plus the two
$0 measurements the P8 rate row needs:
  * ``measure_curve_relative`` -> per-edge {delta_s entropy, Haar N-term, curve-relative coded bytes
    vs the absolute-2-D baseline coded bytes} on the frozen SegNet-argmax cache.

**Chart (bit-exact by integer construction).** For a curve parameterized by one image axis
(``axis='col'`` -> the horizon, a near-horizontal y(x); ``axis='row'`` -> lanes, near-vertical
x(y)), a residual pixel is coded as ``(seg_id, s, n)`` where ``s`` is the param-axis coordinate and
``n`` is the SIGNED offset on the other axis from the segment's rounded center at ``s``.
Reconstruction ``other = round(center_seg[s]) + n`` is EXACT integer arithmetic -> no rounding loss.
Pixels whose param coordinate has NO segment support (occluded / off-curve) go to an explicit
absolute-coded ``exceptions`` list (the honest lossless top-up).  Multi-valued junction segments are
handled by explicit monotone segmentation of each generator curve + nearest-center ``seg_id``
assignment (S1's flagged degradation case).

**NO-FAKE.** The coder actually reproduces the boundary it claims: ``decode_curve_relative`` returns
the exact per-frame residual pixel set (verified bit-for-bit in tests, incl. junction/off-support
edge cases).  All numbers are numpy-fp32 geometry on the frozen argmax -> ``[macOS-CPU advisory ·
NON-PROMOTABLE]``; a coder moves no pointer.  Axis label travels with every measured byte count.

Cross-refs: `position_S1_deepmath_20260709.md` s4 (R(D) framing) · `position_S6_structureblind`
s2 (residual chart) · `SYNTHESIS_DRAFT_v8_20260709.md` A.2/E.R2 · `v8_movable_residual_rollup`.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:  # brotli q11 is the shipped coder (matches the rollup's shared residual coder)
    import brotli as _brotli  # type: ignore

    _HAVE_BROTLI = True
except Exception:  # pragma: no cover - brotli is a hard dep of the coder path but guard anyway
    _brotli = None
    _HAVE_BROTLI = False

_MAGIC_ABS = b"ABS2"
_MAGIC_CRV = b"CRV1"


# ---------------------------------------------------------------------------
# entropy backend (brotli q11 preferred, zlib-9 fallback; both real + bit-exact)
# ---------------------------------------------------------------------------
def _pack(raw: bytes) -> bytes:
    if _HAVE_BROTLI:
        return _brotli.compress(raw, quality=11)  # type: ignore[union-attr]
    return zlib.compress(raw, 9)


def _unpack(blob: bytes) -> bytes:
    if _HAVE_BROTLI:
        try:
            return _brotli.decompress(blob)  # type: ignore[union-attr]
        except Exception:
            return zlib.decompress(blob)
    return zlib.decompress(blob)


def _zigzag(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return ((x << 1) ^ (x >> 63)).astype(np.uint64)


def _unzigzag(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.uint64)
    return ((z >> np.uint64(1)).astype(np.int64)) ^ -(z & np.uint64(1)).astype(np.int64)


def _u32(n: int) -> bytes:
    return struct.pack("<I", int(n))


def _ru32(blob: bytes, off: int) -> tuple[int, int]:
    (v,) = struct.unpack("<I", blob[off:off + 4])
    return v, off + 4


# ---------------------------------------------------------------------------
# BASELINE: absolute 2-D flat-index coder (the "current 2-D chain-code baseline")
# ---------------------------------------------------------------------------
def _pack_fields(fields: list[bytes]) -> bytes:
    """Concatenate raw field buffers with u32 length prefixes, then a SINGLE brotli/zlib pass.

    One entropy stream (low fixed overhead) instead of one brotli blob per field -- so small
    residual sets are not dominated by per-stream headers.
    """
    raw = b"".join(_u32(len(f)) + f for f in fields)
    return _pack(raw)


def _unpack_fields(blob: bytes, n: int) -> list[bytes]:
    raw = _unpack(blob)
    out, off = [], 0
    for _ in range(n):
        ln, off = _ru32(raw, off)
        out.append(raw[off:off + ln])
        off += ln
    return out


def encode_absolute_2d(frames_flat_idx: Sequence[np.ndarray], grid_h: int, grid_w: int) -> bytes:
    """Per-frame sorted flat-index residual, temporal/spatial-delta + zigzag + single brotli q11.

    This is the generic absolute-coordinate residual coder (the rollup's "shared residual coder"
    pattern; the ~0.4-0.6 B/px baseline the curve-relative coder must beat).  Bit-exact.
    """
    n_frames = len(frames_flat_idx)
    counts = np.empty(n_frames, dtype=np.int64)
    all_deltas: list[np.ndarray] = []
    prev = 0  # spatial+temporal running index (delta across the whole concatenated stream)
    for i, fi in enumerate(frames_flat_idx):
        idx = np.unique(np.asarray(fi, dtype=np.int64))  # sorted, dedup
        counts[i] = idx.size
        if idx.size:
            seq = np.concatenate([[prev], idx])
            all_deltas.append(np.diff(seq))
            prev = int(idx[-1])
    deltas = np.concatenate(all_deltas) if all_deltas else np.zeros(0, dtype=np.int64)
    body = _pack_fields([counts.astype(np.int64).tobytes(), _zigzag(deltas).tobytes()])
    head = _MAGIC_ABS + struct.pack("<III", n_frames, int(grid_h), int(grid_w))
    return head + body


def decode_absolute_2d(blob: bytes) -> tuple[int, int, list[np.ndarray]]:
    """Inverse of ``encode_absolute_2d`` -> (grid_h, grid_w, per-frame sorted flat idx). Bit-exact."""
    if blob[:4] != _MAGIC_ABS:
        raise ValueError("bad ABS2 magic")
    n_frames, grid_h, grid_w = struct.unpack("<III", blob[4:16])
    cnt_raw, d_raw = _unpack_fields(blob[16:], 2)
    counts = np.frombuffer(cnt_raw, dtype=np.int64)
    deltas = _unzigzag(np.frombuffer(d_raw, dtype=np.uint64))
    frames: list[np.ndarray] = []
    d_off = 0
    prev = 0
    for c in counts:
        c = int(c)
        if c == 0:
            frames.append(np.zeros(0, dtype=np.int64))
            continue
        seg = deltas[d_off:d_off + c]
        d_off += c
        idx = prev + np.cumsum(seg)
        frames.append(idx.astype(np.int64))
        prev = int(idx[-1])
    return int(grid_h), int(grid_w), frames


# ---------------------------------------------------------------------------
# GeneratorCurve: a monotone polyline parameterized by one image axis
# ---------------------------------------------------------------------------
@dataclass
class GeneratorCurve:
    """One monotone generator-curve segment, param by ``axis`` ('col' or 'row').

    ``support`` are the integer param-axis coordinates present; ``center`` the ROUNDED integer
    coordinate on the OTHER axis at each support coord.  A residual px at param ``p`` is coded as
    the signed offset ``n = other - center[p]``; reconstruction ``other = center[p] + n`` is exact.
    """

    axis: str            # 'col' -> param=column, offset=row ; 'row' -> param=row, offset=column
    support: np.ndarray  # int32, sorted unique param coords
    center: np.ndarray   # int32, rounded center coord on the offset axis (len == support)
    seg_id: int = 0

    def __post_init__(self) -> None:
        if self.axis not in ("col", "row"):
            raise ValueError(f"axis must be 'col' or 'row'; got {self.axis!r}")
        self.support = np.asarray(self.support, dtype=np.int32)
        self.center = np.asarray(self.center, dtype=np.int32)
        if self.support.shape != self.center.shape:
            raise ValueError("support/center length mismatch")
        # build param -> center lookup (sparse, via searchsorted on sorted support)
        if self.support.size and np.any(np.diff(self.support) <= 0):
            order = np.argsort(self.support, kind="stable")
            self.support = self.support[order]
            self.center = self.center[order]
            # dedup param collisions (keep first) -> guarantees strict monotone support
            keep = np.concatenate([[True], np.diff(self.support) > 0])
            self.support = self.support[keep]
            self.center = self.center[keep]

    def center_at(self, p: np.ndarray) -> np.ndarray:
        """Center coord at param values ``p`` (int); returns -1 where ``p`` is off-support."""
        p = np.asarray(p, dtype=np.int64)
        pos = np.searchsorted(self.support, p)
        out = np.full(p.shape, -1, dtype=np.int64)
        in_range = pos < self.support.size
        hit = np.zeros(p.shape, dtype=bool)
        hit[in_range] = self.support[pos[in_range]] == p[in_range]
        out[hit] = self.center[pos[hit]]
        return out


def curve_from_column_function(y_of_x: np.ndarray, *, seg_id: int = 0) -> GeneratorCurve:
    """Horizon-style curve: ``y_of_x`` is an int array length W, ``-1`` = column has no curve point.

    Param axis = column (``axis='col'``), offset = row.  Reproduces the dominant horizon arc chart.
    """
    y = np.asarray(y_of_x, dtype=np.int64)
    xs = np.where(y >= 0)[0].astype(np.int32)
    return GeneratorCurve(axis="col", support=xs, center=y[xs].astype(np.int32), seg_id=seg_id)


def curves_from_coverage_mask(
    cov_mask: np.ndarray, *, axis: str = "row", min_len: int = 8,
) -> list[GeneratorCurve]:
    """Extract monotone generator polylines from a boolean coverage band (lane-style).

    For ``axis='row'`` (near-vertical lanes): per row, split the covered columns into contiguous
    runs; each run's center is a curve point.  Runs are threaded across rows into segments by
    nearest-center continuity (gap<=2 rows, |dcenter|<=6 px).  This yields explicit monotone
    (row-parameterized) segments -> the junction/multi-valued case is resolved into single-valued
    pieces (S1's degradation handling).  ``axis='col'`` is the transpose (per-column runs).
    """
    if axis not in ("row", "col"):
        raise ValueError("axis must be 'row' or 'col'")
    m = np.asarray(cov_mask, dtype=bool)
    if axis == "col":
        m = m.T
    n_par = m.shape[0]  # param axis length (rows for 'row')
    # active segments: list of dict(param list, center list, last_par, last_center)
    open_segs: list[dict] = []
    done_segs: list[dict] = []
    for p in range(n_par):
        line = m[p]
        cols = np.where(line)[0]
        centers_here: list[int] = []
        if cols.size:
            # contiguous runs
            brk = np.where(np.diff(cols) > 1)[0]
            starts = np.concatenate([[0], brk + 1])
            ends = np.concatenate([brk + 1, [cols.size]])
            for s0, e0 in zip(starts, ends):
                run = cols[s0:e0]
                centers_here.append(int(round(float(run.mean()))))
        # match centers to open segments (greedy nearest, gap<=2, |d|<=6)
        used = [False] * len(centers_here)
        still_open: list[dict] = []
        for seg in open_segs:
            if p - seg["last_par"] > 2:
                done_segs.append(seg)
                continue
            best_j, best_d = -1, 7
            for j, c in enumerate(centers_here):
                if used[j]:
                    continue
                d = abs(c - seg["last_center"])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                seg["par"].append(p)
                seg["cen"].append(centers_here[best_j])
                seg["last_par"] = p
                seg["last_center"] = centers_here[best_j]
                still_open.append(seg)
            else:
                done_segs.append(seg)
        for j, c in enumerate(centers_here):
            if not used[j]:
                still_open.append({"par": [p], "cen": [c], "last_par": p, "last_center": c})
        open_segs = still_open
    done_segs.extend(open_segs)
    curves: list[GeneratorCurve] = []
    sid = 0
    for seg in done_segs:
        if len(seg["par"]) < min_len:
            continue
        curves.append(
            GeneratorCurve(
                axis=axis,
                support=np.asarray(seg["par"], dtype=np.int32),
                center=np.asarray(seg["cen"], dtype=np.int32),
                seg_id=sid,
            )
        )
        sid += 1
    return curves


# ---------------------------------------------------------------------------
# chart transform: residual pixels -> (seg_id, s, n) + exceptions
# ---------------------------------------------------------------------------
@dataclass
class ChartCoords:
    """Curve-relative coords for one frame's residual pixels (bit-exact reconstructable)."""

    grid_h: int
    grid_w: int
    seg_id: np.ndarray   # int32 per coded px
    s: np.ndarray        # int32 param coord per coded px
    n: np.ndarray        # int32 signed offset per coded px
    exceptions: np.ndarray  # int64 flat idx of off-support px (absolute-coded)
    axis_by_seg: list[str] = field(default_factory=list)


def chart_transform(
    residual_flat_idx: np.ndarray, curves: Sequence[GeneratorCurve], grid_h: int, grid_w: int,
) -> ChartCoords:
    """Assign each residual px to the nearest supporting segment; emit ``(seg_id, s, n)``.

    Off-support px (no segment supports its param coord) -> ``exceptions``.  Deterministic:
    nearest-center tie broken by lowest seg_id.
    """
    idx = np.unique(np.asarray(residual_flat_idx, dtype=np.int64))
    rows = (idx // grid_w).astype(np.int64)
    cols = (idx % grid_w).astype(np.int64)
    seg_ids = np.full(idx.size, -1, dtype=np.int64)
    s_arr = np.zeros(idx.size, dtype=np.int64)
    n_arr = np.zeros(idx.size, dtype=np.int64)
    best_dist = np.full(idx.size, np.iinfo(np.int64).max, dtype=np.int64)
    axis_by_seg: list[str] = []
    for ci, cv in enumerate(curves):
        axis_by_seg.append(cv.axis)
        if cv.axis == "col":
            param, other = cols, rows
        else:
            param, other = rows, cols
        center = cv.center_at(param)
        supported = center >= 0
        if not supported.any():
            continue
        off = np.where(supported, other - center, 0)
        dist = np.abs(off)
        take = supported & (dist < best_dist)
        seg_ids[take] = cv.seg_id
        s_arr[take] = param[take]
        n_arr[take] = off[take]
        best_dist[take] = dist[take]
    coded = seg_ids >= 0
    exceptions = idx[~coded]
    return ChartCoords(
        grid_h=int(grid_h), grid_w=int(grid_w),
        seg_id=seg_ids[coded].astype(np.int32),
        s=s_arr[coded].astype(np.int32),
        n=n_arr[coded].astype(np.int32),
        exceptions=exceptions.astype(np.int64),
        axis_by_seg=axis_by_seg,
    )


def reconstruct_from_chart(cc: ChartCoords, curves: Sequence[GeneratorCurve]) -> np.ndarray:
    """(seg_id,s,n)+exceptions -> exact residual flat-idx set. Inverse of ``chart_transform``."""
    seg_by_id = {int(cv.seg_id): cv for cv in curves}
    flat = []
    for sid, s, n in zip(cc.seg_id.tolist(), cc.s.tolist(), cc.n.tolist()):
        cv = seg_by_id[int(sid)]
        center = int(cv.center_at(np.array([s]))[0])
        if center < 0:
            raise ValueError("reconstruct hit off-support param (corrupt chart)")
        other = center + int(n)
        if cv.axis == "col":
            r, c = other, s
        else:
            r, c = s, other
        flat.append(int(r) * cc.grid_w + int(c))
    out = np.array(flat, dtype=np.int64)
    out = np.concatenate([out, np.asarray(cc.exceptions, dtype=np.int64)])
    return np.unique(out)


# ---------------------------------------------------------------------------
# curve-relative coder (bit-exact): encode/decode the (seg,s,n)+exceptions stream
# ---------------------------------------------------------------------------
def encode_curve_relative(charts: Sequence[ChartCoords]) -> bytes:
    """Code n (zigzag, tiny alphabet) + s (delta) + seg_id + exceptions across frames. Bit-exact."""
    n_frames = len(charts)
    if n_frames == 0:
        return _MAGIC_CRV + struct.pack("<III", 0, 0, 0)
    grid_h, grid_w = charts[0].grid_h, charts[0].grid_w
    counts = np.array([c.seg_id.size for c in charts], dtype=np.int64)
    exc_counts = np.array([c.exceptions.size for c in charts], dtype=np.int64)
    seg_all = np.concatenate([c.seg_id for c in charts]) if counts.sum() else np.zeros(0, np.int32)
    n_all = np.concatenate([c.n for c in charts]) if counts.sum() else np.zeros(0, np.int32)
    # s delta-coded per frame (reset each frame) after sorting by (seg,s) for locality
    s_deltas: list[np.ndarray] = []
    for c in charts:
        if c.s.size == 0:
            continue
        order = np.lexsort((c.s, c.seg_id))
        # NOTE: seg/n must be re-emitted in the SAME order for decode; re-sort them here too
        s_sorted = c.s[order].astype(np.int64)
        # per-seg delta of s
        seg_sorted = c.seg_id[order]
        dseg = np.concatenate([[1], np.diff(seg_sorted).astype(np.int64)])
        s_prev = np.concatenate([[0], s_sorted[:-1]])
        s_prev[dseg != 0] = 0  # reset at seg boundary
        s_deltas.append(s_sorted - s_prev)
    # rebuild seg_all / n_all in the SAME lexsorted order
    seg_ord, n_ord = [], []
    for c in charts:
        if c.s.size == 0:
            continue
        order = np.lexsort((c.s, c.seg_id))
        seg_ord.append(c.seg_id[order])
        n_ord.append(c.n[order])
    seg_all = np.concatenate(seg_ord) if seg_ord else np.zeros(0, np.int32)
    n_all = np.concatenate(n_ord) if n_ord else np.zeros(0, np.int32)
    s_all = np.concatenate(s_deltas) if s_deltas else np.zeros(0, np.int64)
    exc_sorted = []
    prev = 0
    for c in charts:
        if c.exceptions.size == 0:
            continue
        e = np.unique(np.asarray(c.exceptions, dtype=np.int64))
        seq = np.concatenate([[prev], e])
        exc_sorted.append(np.diff(seq))
        prev = int(e[-1])
    exc_deltas = np.concatenate(exc_sorted) if exc_sorted else np.zeros(0, np.int64)

    body = _pack_fields([
        counts.tobytes(),
        exc_counts.tobytes(),
        seg_all.astype(np.uint16).tobytes(),
        _zigzag(n_all).tobytes(),
        _zigzag(s_all).tobytes(),
        _zigzag(exc_deltas).tobytes(),
    ])
    head = _MAGIC_CRV + struct.pack("<III", n_frames, int(grid_h), int(grid_w))
    return head + body


def decode_curve_relative(
    blob: bytes, curves: Sequence[GeneratorCurve],
) -> tuple[int, int, list[np.ndarray]]:
    """Inverse -> (grid_h, grid_w, per-frame residual flat idx). Bit-exact (tested)."""
    if blob[:4] != _MAGIC_CRV:
        raise ValueError("bad CRV1 magic")
    n_frames, grid_h, grid_w = struct.unpack("<III", blob[4:16])
    if n_frames == 0:
        return int(grid_h), int(grid_w), []
    cnt_raw, exc_cnt_raw, seg_raw, n_raw, s_raw, exc_raw = _unpack_fields(blob[16:], 6)
    counts = np.frombuffer(cnt_raw, dtype=np.int64)
    exc_counts = np.frombuffer(exc_cnt_raw, dtype=np.int64)
    seg_all = np.frombuffer(seg_raw, dtype=np.uint16).astype(np.int64)
    n_all = _unzigzag(np.frombuffer(n_raw, dtype=np.uint64))
    s_all = _unzigzag(np.frombuffer(s_raw, dtype=np.uint64))
    exc_deltas = _unzigzag(np.frombuffer(exc_raw, dtype=np.uint64))
    seg_by_id = {int(cv.seg_id): cv for cv in curves}

    frames: list[np.ndarray] = []
    p_off = 0
    e_off = 0
    exc_prev = 0
    for fi in range(n_frames):
        c = int(counts[fi])
        seg = seg_all[p_off:p_off + c]
        n = n_all[p_off:p_off + c]
        s_d = s_all[p_off:p_off + c]
        p_off += c
        # invert per-seg s delta
        s = np.zeros(c, dtype=np.int64)
        if c:
            dseg = np.concatenate([[1], np.diff(seg).astype(np.int64)])
            acc = 0
            for i in range(c):
                if dseg[i] != 0:
                    acc = 0
                acc += int(s_d[i])
                s[i] = acc
        flat = []
        for sid, sv, nv in zip(seg.tolist(), s.tolist(), n.tolist()):
            cv = seg_by_id[int(sid)]
            center = int(cv.center_at(np.array([sv]))[0])
            other = center + int(nv)
            if cv.axis == "col":
                r, col = other, sv
            else:
                r, col = sv, other
            flat.append(int(r) * grid_w + int(col))
        ec = int(exc_counts[fi])
        e_seg = exc_deltas[e_off:e_off + ec]
        e_off += ec
        if ec:
            seq = np.concatenate([[exc_prev], e_seg])
            e_idx = np.cumsum(seq)[1:]
            exc_prev = int(e_idx[-1])
            flat.extend(e_idx.tolist())
        frames.append(np.unique(np.array(flat, dtype=np.int64)))
    return int(grid_h), int(grid_w), frames


# ---------------------------------------------------------------------------
# delta_s compressibility measurement (entropy + Haar N-term)
# ---------------------------------------------------------------------------
def _empirical_entropy_bits(vals: np.ndarray) -> float:
    vals = np.asarray(vals)
    if vals.size == 0:
        return 0.0
    _, counts = np.unique(vals, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _haar_nterm_fraction(sig: np.ndarray, *, energy: float = 0.99) -> float:
    """Fraction of Haar coeffs needed to retain ``energy`` of the signal's L2 energy (N-term)."""
    x = np.asarray(sig, dtype=np.float64)
    if x.size < 2:
        return 1.0
    # single-level+ Haar via iterated averaging/differencing to nearest pow2
    n = 1 << int(np.floor(np.log2(x.size)))
    x = x[:n] - x[:n].mean()
    coeffs = [x.copy()]
    cur = x.copy()
    while cur.size > 1:
        a = (cur[0::2] + cur[1::2]) / np.sqrt(2.0)
        d = (cur[0::2] - cur[1::2]) / np.sqrt(2.0)
        coeffs.append(d)
        cur = a
    allc = np.concatenate([cur] + coeffs[1:]) if len(coeffs) > 1 else cur
    e = np.sort(allc**2)[::-1]
    tot = e.sum()
    if tot <= 0:
        return 0.0
    cum = np.cumsum(e) / tot
    k = int(np.searchsorted(cum, energy) + 1)
    return float(k) / float(allc.size)


def delta_s_spectrum(charts: Sequence[ChartCoords]) -> dict:
    """Entropy (bits/sample) of the offset alphabet n + Haar N-term of n(s) per segment (averaged)."""
    n_all = np.concatenate([c.n for c in charts]) if any(c.n.size for c in charts) else np.zeros(0)
    ent = _empirical_entropy_bits(n_all)
    # per (frame,seg) 1-D signal n(s), sorted by s
    nterms: list[float] = []
    for c in charts:
        if c.seg_id.size == 0:
            continue
        for sid in np.unique(c.seg_id):
            m = c.seg_id == sid
            order = np.argsort(c.s[m])
            sig = c.n[m][order]
            if sig.size >= 8:
                nterms.append(_haar_nterm_fraction(sig))
    return {
        "offset_alphabet_entropy_bits": ent,
        "offset_abs_max_px": int(np.abs(n_all).max()) if n_all.size else 0,
        "offset_abs_mean_px": float(np.abs(n_all).mean()) if n_all.size else 0.0,
        "haar_nterm_frac_mean": float(np.mean(nterms)) if nterms else float("nan"),
        "n_segments_measured": len(nterms),
    }


# ---------------------------------------------------------------------------
# $0 MEASUREMENT 2: curve-relative coded bytes vs absolute-2-D baseline, per edge
# ---------------------------------------------------------------------------
def measure_curve_relative(
    residual_by_frame: Sequence[np.ndarray],
    curves_by_frame: Sequence[Sequence[GeneratorCurve]],
    grid_h: int,
    grid_w: int,
    *,
    edge_name: str = "edge",
) -> dict:
    """MEASURE one edge: curve-relative coded bytes vs absolute-2-D baseline + delta(s) compressibility.

    ``residual_by_frame[i]`` = flat idx of that frame's uncovered separatrix px; ``curves_by_frame[i]``
    = the generator segments to code against.  Both coders are REAL + bit-exact (asserted here:
    decode(encode)==input); the curve-relative number INCLUDES the off-support absolute exceptions
    (the honest lossless top-up).  Returns per-edge bytes + savings ratio + the n(s) spectrum.
    """
    charts = [
        chart_transform(residual_by_frame[i], curves_by_frame[i], grid_h, grid_w)
        for i in range(len(residual_by_frame))
    ]
    # --- bit-exact self-verification (NO-FAKE: the coder reproduces the boundary it claims) ---
    blob_crv = encode_curve_relative(charts)
    # per-frame decode needs that frame's curves; verify frame-by-frame
    bit_exact = True
    for i, ch in enumerate(charts):
        _, _, dec = decode_curve_relative(encode_curve_relative([ch]), curves_by_frame[i])
        want = np.unique(np.asarray(residual_by_frame[i], dtype=np.int64))
        got = dec[0] if dec else np.zeros(0, dtype=np.int64)
        if not np.array_equal(want, got):
            bit_exact = False
            break
    blob_abs = encode_absolute_2d([np.asarray(r, np.int64) for r in residual_by_frame], grid_h, grid_w)
    # baseline bit-exact check
    _, _, dec_abs = decode_absolute_2d(blob_abs)
    abs_exact = all(
        np.array_equal(np.unique(np.asarray(residual_by_frame[i], np.int64)), dec_abs[i])
        for i in range(len(residual_by_frame))
    )
    n_px = int(sum(np.unique(np.asarray(r, np.int64)).size for r in residual_by_frame))
    n_exc = int(sum(c.exceptions.size for c in charts))
    spec = delta_s_spectrum(charts)
    b_crv = len(blob_crv)
    b_abs = len(blob_abs)
    return {
        "edge": edge_name,
        "n_residual_px": n_px,
        "n_exceptions_off_support": n_exc,
        "frac_on_curve": 1.0 - float(n_exc) / float(max(1, n_px)),
        "bytes_absolute_2d_baseline": int(b_abs),
        "bytes_curve_relative": int(b_crv),
        "savings_ratio": float(b_abs) / float(max(1, b_crv)),
        "S_absolute_2d": 25.0 * b_abs / 37_545_489.0,
        "S_curve_relative": 25.0 * b_crv / 37_545_489.0,
        "bytes_per_px_absolute": float(b_abs) / float(max(1, n_px)),
        "bytes_per_px_curve_relative": float(b_crv) / float(max(1, n_px)),
        "delta_s_spectrum": spec,
        "curve_relative_bit_exact": bool(bit_exact),
        "absolute_bit_exact": bool(abs_exact),
        "axis_label": "[macOS-CPU advisory · NON-PROMOTABLE]",
    }
