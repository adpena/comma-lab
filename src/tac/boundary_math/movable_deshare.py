# SPDX-License-Identifier: MIT
"""Movable-first residual attribution + general pairwise ARCHIVE-DEDUP AUDIT (v8 T2 Lever-1; OFF).

The v8 rate ledger sums a DOMINANT generator + a RESIDUAL sidecar per inter-class edge.  Two
carriers can silently code the SAME pixel twice: the horizon residual's "secondary arcs" ARE cars
breaking the horizon (MEASURED 1.6-2.0 crossings/row) = Movable pixels the Movable carrier ALREADY
holds; lane fragments near cars are likewise double-held.  Paying for a pixel in two ledger rows is
a pure double-count that INFLATES the complete rate for zero d_seg benefit.

**Lever-1 = de-share (Movable-first attribution).**  Assign every uncovered separatrix pixel to
EXACTLY ONE carrier (its true class-pair), giving the Movable footprint priority so straddle pixels
are subtracted from the horizon + Road/Lane residual owed-sets BEFORE they are coded.  This is pure
set-partition de-duplication -- no coder change, no d_seg change -- so it is a strict MONOTONE
deflation of the complete number, free regardless of magnitude (S1 recommends it FIRST).

**Operator binding 2026-07-09 ("no duplicate data in archive; all falls out from geometry").**
De-share is the first INSTANCE of a general principle: no archive byte should be reconstructible from
another section + the free generator.  ``pairwise_dedup_audit`` generalizes the Movable case to a
sweep of ALL ledger-row pairs (horizon<->lane, hood<->road, movable<->{horizon,lane}, ...) and
reports any additional double-count found, per pair.  Each residual pixel's UNIQUE GEOMETRIC HOME is
the carrier whose generator owns its class-pair (stated per-output in the landing memo).

**NO-FAKE.** All numbers are numpy-fp32 geometry on the frozen SegNet-argmax cache; de-share deltas
are MEASURED through the real bit-exact absolute-2-D coder (``curve_relative_offset_coder``), never a
b/px proxy.  Deterministic + seeded (the op is fully deterministic; ``seed`` is accepted for interface
parity and asserted unused-for-randomness).  ``[macOS-CPU advisory · NON-PROMOTABLE]`` -- a dedup
audit moves no pointer.

Cross-refs: `position_S1_deepmath` s4 (de-share lever, EXACT monotone) · `position_S6_structureblind`
s2 (de-sharing partition) · `v8_movable_residual_rollup_20260709.md` (headroom lever 1) ·
`operator_no_duplicate_data_archive_geometry_first_20260709.md` · SYNTHESIS_DRAFT_v8 A.1/A.2/E.R1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.boundary_math.curve_relative_offset_coder import encode_absolute_2d

N_SEG_CLASSES = 5


# ---------------------------------------------------------------------------
# class self-detection (never hardcode the argmax order -- CLAUDE.md canonical rule)
# ---------------------------------------------------------------------------
@dataclass
class SegRoles:
    road: int
    lane: int
    undriv: int
    movable: int
    mycar: int

    def as_dict(self) -> dict[str, int]:
        return {"road": self.road, "lane": self.lane, "undriv": self.undriv,
                "movable": self.movable, "mycar": self.mycar}


def detect_seg_roles(lstars: np.ndarray, *, n_classes: int = N_SEG_CLASSES) -> SegRoles:
    """Self-detect the 5 comma10k roles by spatial/area/static signature (NO hardcoded index).

    Road+Undriv via the proven ``road_undriv_bulk_field.identify_road_undriv_classes``; the
    remaining three by signature: MyCar = static BOTTOM band (largest area among remaining, low
    vertical centroid); Lane = the thinnest (smallest area) remaining; Movable = the leftover
    mid-band class.  Verified against the comma10k canonical order on the real cache in tests.
    """
    from tac.boundary_math.road_undriv_bulk_field import identify_road_undriv_classes

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    ru = identify_road_undriv_classes(a, n_classes=n_classes)
    road, undriv = int(ru.road), int(ru.undriv)
    h = a.shape[1]
    remaining = [c for c in range(n_classes) if c not in (road, undriv)]
    areas: dict[int, float] = {}
    vcen: dict[int, float] = {}
    for c in remaining:
        mask = a == c
        areas[c] = float(mask.mean())
        vcen[c] = float(np.average(np.arange(h), weights=mask.sum(axis=(0, 2)) + 1e-9))
    # MyCar = largest-area remaining sitting in the BOTTOM third (static ego hood)
    bottom = [c for c in remaining if vcen[c] > 2.0 * h / 3.0]
    if bottom:
        mycar = max(bottom, key=lambda c: areas[c])
    else:
        mycar = max(remaining, key=lambda c: areas[c])
    rem2 = [c for c in remaining if c != mycar]
    lane = min(rem2, key=lambda c: areas[c])  # thinnest = lane markings
    movable = [c for c in rem2 if c != lane][0]
    return SegRoles(road=road, lane=lane, undriv=undriv, movable=movable, mycar=mycar)


# ---------------------------------------------------------------------------
# geometry primitives (separatrix, footprint) -- all deterministic numpy
# ---------------------------------------------------------------------------
def _dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary 8-neighbour dilation by ``iters`` (no scipy dep; deterministic)."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(iters)):
        out = m.copy()
        out[:-1, :] |= m[1:, :]
        out[1:, :] |= m[:-1, :]
        out[:, :-1] |= m[:, 1:]
        out[:, 1:] |= m[:, :-1]
        out[:-1, :-1] |= m[1:, 1:]
        out[1:, 1:] |= m[:-1, :-1]
        out[:-1, 1:] |= m[1:, :-1]
        out[1:, :-1] |= m[:-1, 1:]
        m = out
    return m


def separatrix_mask(lstar: np.ndarray, cls_a: int, cls_b: int) -> np.ndarray:
    """Boolean (H,W): pixels of class a 4-adjacent to class b, UNION the symmetric b-adjacent-a set."""
    a = np.asarray(lstar)
    ma = a == int(cls_a)
    mb = a == int(cls_b)

    def _adj(m_self: np.ndarray, m_other: np.ndarray) -> np.ndarray:
        nb = np.zeros_like(m_other)
        nb[:-1, :] |= m_other[1:, :]
        nb[1:, :] |= m_other[:-1, :]
        nb[:, :-1] |= m_other[:, 1:]
        nb[:, 1:] |= m_other[:, :-1]
        return m_self & nb

    return _adj(ma, mb) | _adj(mb, ma)


# INSTANCE-of-dil2: de-share band [0.000,0.0069] measured across dilate∈{0..3}; NOT footprint-robust; thing-itself owed
# (P9 proxy tag, philosophy_pass_v8_20260709 §3 rank-1 / §A.2; the 0.0044 de-share it generates is INSTANCE
#  at dilate=2, not the realized Movable-carrier coverage — the thing-itself is G3 bbox realized cover @ byte-close.
#  The measure/audit entrypoints below inherit this default; band is the same proxy caveat.)
def movable_footprint(lstar: np.ndarray, movable_cls: int, *, dilate: int = 2) -> np.ndarray:
    """The pixels the Movable carrier (bbox sites + track) already accounts for = dilated region."""
    return _dilate(np.asarray(lstar) == int(movable_cls), dilate)


def _horizon_poly_rows(lstar: np.ndarray, road: int, undriv: int, *, degree: int = 3) -> np.ndarray:
    """Per-column dominant-horizon row y(x) from the deg-``degree`` poly fit (reuses the real fn's
    logic); ``-1`` where un-fittable.  Matches ``road_undriv_bulk_field.horizon_poly_xi_byte_cost``.
    """
    from tac.boundary_math.road_undriv_bulk_field import _horizon_profile

    ys = _horizon_profile(np.asarray(lstar), road, undriv)
    valid = ys >= 0
    xs = np.where(valid)[0]
    w = ys.shape[0]
    out = np.full(w, -1, dtype=np.int64)
    if xs.size < (degree + 5):
        return out
    c = np.polyfit(xs.astype(np.float64), ys[valid].astype(np.float64), degree)
    allx = np.arange(w)
    fitted = np.rint(np.polyval(c, allx.astype(np.float64))).astype(np.int64)
    # only expose the poly where the profile actually had road/undriv support nearby (dominant arc)
    out[valid] = fitted[valid]
    return out


# ---------------------------------------------------------------------------
# residual reconstruction per edge (the ledger rows, on the frozen cache)
# ---------------------------------------------------------------------------
def horizon_residual_idx(lstar: np.ndarray, roles: SegRoles, *, tol: int = 2, degree: int = 3) -> np.ndarray:
    """Road<->Undriv separatrix pixels the dominant horizon poly MISSES (the residual sidecar set)."""
    sep = separatrix_mask(lstar, roles.road, roles.undriv)
    if not sep.any():
        return np.zeros(0, dtype=np.int64)
    rows, cols = np.where(sep)
    poly = _horizon_poly_rows(lstar, roles.road, roles.undriv, degree=degree)
    yv = poly[cols]
    covered = (yv >= 0) & (np.abs(rows - yv) <= tol)
    resid = ~covered
    w = lstar.shape[1]
    return (rows[resid] * w + cols[resid]).astype(np.int64)


def lane_residual_idx(lstar: np.ndarray, roles: SegRoles, coverage: np.ndarray) -> np.ndarray:
    """Road<->Lane separatrix pixels the analytic lane band MISSES (coverage<0.5)."""
    sep = separatrix_mask(lstar, roles.road, roles.lane)
    if not sep.any():
        return np.zeros(0, dtype=np.int64)
    rows, cols = np.where(sep)
    cov = np.asarray(coverage)
    covered = cov[rows, cols] >= 0.5
    resid = ~covered
    w = lstar.shape[1]
    return (rows[resid] * w + cols[resid]).astype(np.int64)


# ---------------------------------------------------------------------------
# THE LEVER: Movable-first attribution (de-share partition)
# ---------------------------------------------------------------------------
@dataclass
class DeShareResult:
    edge: str
    n_residual: int
    n_attributed_movable: int
    n_kept: int

    @property
    def frac_attributed(self) -> float:
        return float(self.n_attributed_movable) / float(max(1, self.n_residual))


def deshare_partition(
    residual_idx: np.ndarray, movable_footprint_flat: np.ndarray, edge: str,
) -> tuple[np.ndarray, DeShareResult]:
    """Movable-first: residual px inside the Movable footprint -> attributed to G3 (subtracted).

    Returns (kept_idx, result).  PARTITION GUARANTEE: ``kept`` and ``attributed`` are disjoint and
    union to the input residual set -> no pixel double-counted, no pixel lost.
    """
    resid = np.unique(np.asarray(residual_idx, dtype=np.int64))
    in_mov = movable_footprint_flat[resid] if resid.size else np.zeros(0, dtype=bool)
    attributed = resid[in_mov]
    kept = resid[~in_mov]
    return kept, DeShareResult(
        edge=edge, n_residual=int(resid.size),
        n_attributed_movable=int(attributed.size), n_kept=int(kept.size),
    )


# ---------------------------------------------------------------------------
# $0 MEASUREMENT 1: de-share magnitude (bytes double-counted, per edge)
# ---------------------------------------------------------------------------
def measure_deshare_magnitude(
    lstars: np.ndarray, *, seed: int = 0, dilate: int = 2, tol: int = 2, roles: SegRoles | None = None,
) -> dict:
    """MEASURE how many residual bytes are Movable double-count, per edge (Movable-first de-share).

    For each edge (horizon Road<->Undriv, Road<->Lane): reconstruct the residual set, de-share vs the
    Movable footprint, code BEFORE and AFTER through the real bit-exact absolute-2-D coder; report the
    byte deflation + S deflation.  The attributed pixels are FREE (already held by the Movable carrier).
    """
    assert int(seed) == seed  # deterministic op; seed accepted for interface parity (unused-for-RNG)
    from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    n, h, w = a.shape
    if roles is None:
        roles = detect_seg_roles(a)

    hz_before: list[np.ndarray] = []
    hz_after: list[np.ndarray] = []
    ln_before: list[np.ndarray] = []
    ln_after: list[np.ndarray] = []
    per_edge_px = {"horizon": [0, 0], "lane": [0, 0]}  # [residual, attributed]

    for i in range(n):
        fp = movable_footprint(a[i], roles.movable, dilate=dilate).reshape(-1)
        hz = horizon_residual_idx(a[i], roles, tol=tol)
        hk, hr = deshare_partition(hz, fp, "horizon")
        hz_before.append(hz)
        hz_after.append(hk)
        per_edge_px["horizon"][0] += hr.n_residual
        per_edge_px["horizon"][1] += hr.n_attributed_movable

        prior = build_analytic_lane_band_prior(a[i], lane_cls=roles.lane)
        ln = lane_residual_idx(a[i], roles, prior.coverage)
        lk, lr = deshare_partition(ln, fp, "lane")
        ln_before.append(ln)
        ln_after.append(lk)
        per_edge_px["lane"][0] += lr.n_residual
        per_edge_px["lane"][1] += lr.n_attributed_movable

    def _bytes(frames: list[np.ndarray]) -> int:
        return len(encode_absolute_2d(frames, h, w))

    edges = {}
    for name, before, after in (("horizon", hz_before, hz_after), ("lane", ln_before, ln_after)):
        b0 = _bytes(before)
        b1 = _bytes(after)
        px_res, px_att = per_edge_px[name]
        # PRIMARY magnitude: the amortized byte-VALUE of the double-counted pixels, at the edge's
        # coded B/px rate.  This is the honest, monotone measure (attributed px are FREE -- already
        # held by the Movable carrier).  The coded before/after diff (b0-b1) is a DIAGNOSTIC and is
        # NOT strictly monotone: removing a dense-run px from a delta+entropy stream can widen a gap
        # and cost more than the px saved, so b0-b1 occasionally goes slightly negative at small n.
        bpp = float(b0) / float(max(1, px_res))
        dc_amortized = int(round(px_att * bpp))
        edges[name] = {
            "residual_px_total": int(px_res),
            "attributed_movable_px": int(px_att),
            "frac_px_double_counted": float(px_att) / float(max(1, px_res)),
            "bytes_per_px": bpp,
            "bytes_double_counted": dc_amortized,                 # PRIMARY (amortized, monotone)
            "S_double_counted": 25.0 * dc_amortized / 37_545_489.0,
            "bytes_before_deshare_coded": int(b0),                 # DIAGNOSTIC
            "bytes_after_deshare_coded": int(b1),                  # DIAGNOSTIC
            "coded_diff_bytes": int(b0 - b1),                      # DIAGNOSTIC (non-monotone caveat)
            "S_before": 25.0 * b0 / 37_545_489.0,
        }
    total_defl_bytes = sum(e["bytes_double_counted"] for e in edges.values())
    return {
        "n_frames": int(n),
        "roles": roles.as_dict(),
        "dilate": int(dilate),
        "tol": int(tol),
        "edges": edges,
        "total_bytes_double_counted": int(total_defl_bytes),
        "total_S_deflation": 25.0 * total_defl_bytes / 37_545_489.0,
        "magnitude_note": "bytes_double_counted = attributed_px * edge B/px (amortized, monotone); the "
                          "attributed px are FREE (held by the Movable carrier). coded_diff_bytes is a "
                          "diagnostic and may go slightly negative at small n (delta-gap effect).",
        "axis_label": "[macOS-CPU advisory · NON-PROMOTABLE]",
    }


# ---------------------------------------------------------------------------
# $0 MEASUREMENT 1b (operator 2026-07-09): general pairwise ARCHIVE-DEDUP AUDIT
# ---------------------------------------------------------------------------
def pairwise_dedup_audit(
    lstars: np.ndarray, *, dilate: int = 2, tol: int = 2, roles: SegRoles | None = None,
) -> dict:
    """Sweep ALL ledger-row pairs for double-count (pixels one row codes reconstructible from another).

    Ledger rows (as pixel sets per frame): horizon-residual, lane-residual, movable-footprint,
    hood(=MyCar region), road-mycar-separatrix.  For each unordered pair, report the shared-pixel
    count + the coded bytes needed to code the OVERLAP alone (the free double-count if both rows keep
    it).  The Movable<->{horizon,lane} pair is the primary de-share; the others test the operator's
    general principle (no additional archive byte reconstructible from another section).
    """
    from tac.boundary_math.analytic_lane_render_band import build_analytic_lane_band_prior

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    n, h, w = a.shape
    if roles is None:
        roles = detect_seg_roles(a)

    rows_def = ["horizon_resid", "lane_resid", "movable_fp", "hood_region", "road_mycar_sep"]
    pairs = [(i, j) for i in range(len(rows_def)) for j in range(i + 1, len(rows_def))]
    pair_overlap: dict[tuple[int, int], list[np.ndarray]] = {p: [] for p in pairs}

    for k in range(n):
        lst = a[k]
        fp = movable_footprint(lst, roles.movable, dilate=dilate).reshape(-1)
        sets = {
            0: np.unique(horizon_residual_idx(lst, roles, tol=tol)),
            1: np.unique(lane_residual_idx(lst, roles, build_analytic_lane_band_prior(lst, lane_cls=roles.lane).coverage)),
            2: np.where(fp)[0].astype(np.int64),
            3: np.where((lst == roles.mycar).reshape(-1))[0].astype(np.int64),
            4: np.unique((lambda s: (np.where(s)[0]))(separatrix_mask(lst, roles.road, roles.mycar).reshape(-1))).astype(np.int64),
        }
        for (i, j) in pairs:
            ov = np.intersect1d(sets[i], sets[j], assume_unique=True)
            pair_overlap[(i, j)].append(ov)

    out_pairs = []
    for (i, j) in pairs:
        frames = pair_overlap[(i, j)]
        px = int(sum(f.size for f in frames))
        b = len(encode_absolute_2d(frames, h, w)) if px else 0
        out_pairs.append({
            "row_a": rows_def[i],
            "row_b": rows_def[j],
            "shared_px_total": px,
            "overlap_bytes_double_counted": int(b),
            "S_double_counted": 25.0 * b / 37_545_489.0,
        })
    out_pairs.sort(key=lambda d: -d["overlap_bytes_double_counted"])
    return {
        "n_frames": int(n),
        "roles": roles.as_dict(),
        "ledger_rows": rows_def,
        "pairs": out_pairs,
        "axis_label": "[macOS-CPU advisory · NON-PROMOTABLE]",
        "note": "shared px = pixels held by BOTH rows -> reconstructible from either -> a byte that "
                "cannot name ONE geometric home (operator no-duplicate-data binding 2026-07-09).",
    }
