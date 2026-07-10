# SPDX-License-Identifier: MIT
"""movable_site_coder — the v8 Movable-class SPARSE-SITE geometric carrier.

#394 UNIT A, deliverable (b). Movable (class 3 — cars/pedestrians, ~1.56% area, mid-band, IoU
0.903) is SPARSE: a handful of compact object blobs per frame, not a dense field. The v8 rate
thesis (SPEC_v8.1 §I clause-B, mature-codec audit lever #2) is to store Movable as a **sparse
site list** (per-object position + extent) instead of a whole-scene bitmap — the last of the 5
whole-scene edges not yet measured-geometric.

This module:

* **extract** — connected-component object sites from the GT ``L*`` Movable mask per frame
  (:func:`extract_movable_sites`), each a compact ``(cx, cy, w, h)`` axis-aligned box + area.
* **track** — Hungarian temporal correspondence across frames (:func:`track_sites`), so a site
  keeps its slot for its lifetime and the coded stream is a small per-track temporal delta (the
  #234 correspondence-first discipline, reused conceptually; the LAP is the same
  ``scipy.optimize.linear_sum_assignment``). LOSSLESS: only the slot index changes.
* **byte-account** — the COUNTED rate of the site stream (:func:`byte_account_sites`): quantised
  ``(cx,cy,w,h)`` per site + presence, temporal-delta + zigzag + a REAL coder (zlib, the v8
  byte-close coder family) — exact measured bytes, not asserted. The site GENERATOR (draw a box)
  is rule-118 FREE; only the fitted site coords are counted.
* **render** — rasterise the sites back to a Movable mask (:func:`render_sites_to_mask`) so the
  GEOMETRY-coverage loss of the sparse-site approximation vs the GT Movable is MEASURABLE (how
  much GT-Movable area the boxes recover; the honest lossy tell).

NO-FAKE / REUSE-not-rederive: connected components via ``scipy.ndimage.label``; tracking via the
same LAP as :mod:`tac.boundary_math.lane_track_and_smooth`; coding via stdlib zlib (a real
deterministic coder). No GT masks / scorer weights ship — only the fitted site coords are counted.
Deterministic (fixed argsort tie-breaks; zlib level fixed). The through-R d_seg contribution is
measured by the sibling :mod:`tac.through_r.roadlane_texture_generator` composed generator (this
module owns the site GEOMETRY + its rate). ``[macOS-CPU advisory . NON-PROMOTABLE]``; the pointer
(0.19110) moves only through byte-closed exact eval.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.through_r.resolution_chain import SEG_H, SEG_W

__all__ = [
    "MOVABLE_CLASS",
    "MovableSiteCoderError",
    "SiteStreamBytes",
    "TrackedSites",
    "byte_account_sites",
    "extract_movable_sites",
    "render_sites_to_mask",
    "track_sites",
]

MOVABLE_CLASS = 3  # comma10k canonical order (CLAUDE.md): 0=Road 1=Lane 2=Undriv 3=Movable 4=MyCar
_MAX_SITES_PER_FRAME = 64  # sanity cap; > this in one frame -> raise (NO-FAKE, no silent drop)
_ADVISORY_LABEL = "[macOS-CPU advisory . NON-PROMOTABLE]"


class MovableSiteCoderError(ValueError):
    """Raised on a mis-shaped / toy / degenerate site-coder input."""


# --------------------------------------------------------------------------- #
# Extraction.                                                                  #
# --------------------------------------------------------------------------- #
def extract_movable_sites(
    lab: np.ndarray,
    *,
    movable_class: int = MOVABLE_CLASS,
    min_area_px: int = 4,
    h: int = SEG_H,
    w: int = SEG_W,
) -> np.ndarray:
    """Connected-component Movable object sites from a label map -> ``(S, 5)`` float.

    Each row is ``(cx, cy, bw, bh, area)`` in seg-grid px: box centre, box width/height, pixel
    area. Components smaller than ``min_area_px`` are dropped (sub-site noise; a Movable blob below
    a few px cannot survive the stride-2 stem anyway). Deterministic (row-major component labels,
    sorted by descending area then (cy,cx) for a stable slot order). ``S`` may be 0 (no Movable).
    """

    from scipy import ndimage

    lab = np.asarray(lab)
    if lab.shape != (h, w):
        raise MovableSiteCoderError(f"lab must be ({h},{w}); got {lab.shape}")
    mask = lab == int(movable_class)
    if not mask.any():
        return np.zeros((0, 5), dtype=np.float64)
    labeled, n = ndimage.label(mask)
    if n == 0:
        return np.zeros((0, 5), dtype=np.float64)
    rows: list[tuple[float, float, float, float, float]] = []
    slices = ndimage.find_objects(labeled)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        ys, xs = sl
        comp = labeled[ys, xs] == i
        area = int(comp.sum())
        if area < int(min_area_px):
            continue
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        bh = float(y1 - y0)
        bw = float(x1 - x0)
        cy = 0.5 * (y0 + y1 - 1)
        cx = 0.5 * (x0 + x1 - 1)
        rows.append((cx, cy, bw, bh, float(area)))
    if len(rows) > _MAX_SITES_PER_FRAME:
        raise MovableSiteCoderError(
            f"{len(rows)} Movable sites in one frame > {_MAX_SITES_PER_FRAME}; refusing "
            "(NO-FAKE: min_area_px too small -> over-fragmentation, or a mask anomaly). Investigate."
        )
    if not rows:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.array(rows, dtype=np.float64)
    # stable slot order: descending area, then (cy, cx)
    order = np.lexsort((arr[:, 0], arr[:, 1], -arr[:, 4]))
    return arr[order]


# --------------------------------------------------------------------------- #
# Tracking (correspondence-first; reuse the LAP from lane_track_and_smooth).   #
# --------------------------------------------------------------------------- #
@dataclass
class TrackedSites:
    """Bounded-K temporally-coherent site slots: ``M`` ``(P, K*4)`` (cx,cy,bw,bh per slot) + presence."""

    M: np.ndarray            # (P, K*4) float64 -- slot-major (cx,cy,bw,bh), carry-forward hold
    presence: np.ndarray     # (P, K) bool
    K: int
    n_matched: int
    provenance: dict[str, Any] = field(default_factory=dict)


def track_sites(
    per_frame_sites: list[np.ndarray], *, gate_dist_px: float = 48.0, max_gap: int = 8
) -> TrackedSites:
    """Hungarian per-frame site correspondence -> bounded-K coherent slots.

    Association cost = centre L2 distance (px); a match is accepted only if <= ``gate_dist_px``
    (else the site takes a free slot / births). ``K`` = max concurrent sites over the clip; a slot
    is REUSED across a death+birth (correspondence-first: a persistent car keeps its slot -> zero
    temporal delta). Deterministic (stable extract order + LAP). Frames with no sites hold the
    carry-forward. Only the box coords (cx,cy,bw,bh) are tracked (area is derived, not coded).
    """

    from scipy.optimize import linear_sum_assignment

    P = len(per_frame_sites)
    K = max((int(np.asarray(s).shape[0]) for s in per_frame_sites), default=0)
    if K == 0:
        return TrackedSites(
            M=np.zeros((P, 0)), presence=np.zeros((P, 0), bool), K=0, n_matched=0,
            provenance={"tier": "bounded_K_site_track", "reason": "no_movable_sites"},
        )
    D = K * 4
    M = np.zeros((P, D), np.float64)
    presence = np.zeros((P, K), dtype=bool)
    prev = np.zeros((K, 4), np.float64)   # held slot boxes (cx,cy,bw,bh)
    ever = np.zeros(K, dtype=bool)
    last_seen = np.full(K, -(10**9), np.int64)
    n_matched = 0
    for t in range(P):
        sites = np.asarray(per_frame_sites[t], dtype=np.float64)
        n = sites.shape[0]
        if n == 0:
            M[t] = prev.reshape(-1)
            continue
        boxes = sites[:, :4]  # (n,4)
        cen = boxes[:, :2]    # (n,2)
        # candidate slots: alive within max_gap OR never-used (free)
        slot_cen = prev[:, :2]
        C = np.sqrt(((cen[:, None, :] - slot_cen[None, :, :]) ** 2).sum(axis=2))  # (n, K)
        # forbid matching to a slot that is stale (unseen > max_gap) AND was ever used
        stale = ever & ((t - last_seen) > int(max_gap))
        C = C + np.where(stale[None, :], 1e6, 0.0)
        # bias new lanes toward genuinely-free slots over evicting a live one
        C = C + np.where(ever[None, :], 0.0, 1e-3)
        ri, ci = linear_sum_assignment(C)
        assigned: set[int] = set()
        for r, c in zip(ri.tolist(), ci.tolist(), strict=True):
            if float(C[r, c]) > float(gate_dist_px) and ever[c] and not stale[c]:
                continue  # too far from a live slot -> leave r for a free slot below
            prev[c] = boxes[r]
            if ever[c]:
                n_matched += 1
            ever[c] = True
            last_seen[c] = t
            presence[t, c] = True
            assigned.add(r)
        # any unassigned site takes the cheapest free (never-used) slot if one remains
        free = [k for k in range(K) if not presence[t, k]]
        fi = 0
        for r in range(n):
            if r in assigned:
                continue
            if fi >= len(free):
                break
            c = free[fi]
            fi += 1
            prev[c] = boxes[r]
            ever[c] = True
            last_seen[c] = t
            presence[t, c] = True
        M[t] = prev.reshape(-1)
    return TrackedSites(
        M=M, presence=presence, K=int(K), n_matched=int(n_matched),
        provenance={"tier": "bounded_K_site_track", "gate_dist_px": float(gate_dist_px),
                    "max_gap": int(max_gap)},
    )


# --------------------------------------------------------------------------- #
# Byte accounting (the COUNTED site-stream rate; real coder = zlib).           #
# --------------------------------------------------------------------------- #
@dataclass
class SiteStreamBytes:
    """The measured COUNTED bytes of the tracked-site stream + a raw-per-frame control."""

    tracked_bytes: int          # tracked + temporal-delta + zigzag + zlib
    raw_perframe_bytes: int     # per-frame independent (no tracking), zlib -- the control
    presence_bytes: int         # presence bitmap zlib
    n_sites_total: int
    K: int
    P: int
    quant_px: int
    label: str = _ADVISORY_LABEL
    provenance: dict[str, Any] = field(default_factory=dict)


def _zigzag(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return ((a << 1) ^ (a >> 63)).astype(np.uint64)


def byte_account_sites(
    tracked: TrackedSites,
    per_frame_sites: list[np.ndarray],
    *,
    quant_px: int = 1,
    zlib_level: int = 9,
) -> SiteStreamBytes:
    """Measure the COUNTED bytes of the site geometry (tracked temporal-delta vs raw-per-frame).

    Tracked path: quantise each slot box (cx,cy,bw,bh) to ``quant_px``, temporal-delta down each
    slot's present-frame series (carry-forward held frames contribute 0), zigzag, concat, +
    presence bitmap; zlib (a REAL coder, the v8 byte-close family) each and sum. Raw control: each
    frame's site boxes quantised + zlib, no tracking (the "how much does correspondence buy" A/B).
    Only the fitted site coords are COUNTED; the box-raster GENERATOR is rule-118 FREE.
    """

    P = tracked.M.shape[0]
    K = int(tracked.K)
    q = max(1, int(quant_px))

    if K == 0:
        empty = zlib.compress(b"", zlib_level)
        return SiteStreamBytes(
            tracked_bytes=len(empty), raw_perframe_bytes=len(empty), presence_bytes=len(empty),
            n_sites_total=0, K=0, P=P, quant_px=q,
            provenance={"reason": "no_movable_sites"},
        )

    # tracked temporal-delta stream (slot-major)
    Q = np.round(tracked.M / q).astype(np.int64).reshape(P, K, 4)
    presence = tracked.presence
    delta_syms: list[int] = []
    for k in range(K):
        pres_t = np.where(presence[:, k])[0]
        if pres_t.size == 0:
            continue
        seq = Q[pres_t, k, :]  # (n_present, 4)
        d = np.vstack([seq[0:1], np.diff(seq, axis=0)])  # first frame absolute, then deltas
        delta_syms.extend(int(v) for v in d.reshape(-1))
    zz = _zigzag(np.array(delta_syms, dtype=np.int64)) if delta_syms else np.zeros(0, np.uint64)
    tracked_payload = zz.astype("<u4").tobytes()  # 4-byte little-endian symbols (zlib re-packs)
    tracked_bytes = len(zlib.compress(tracked_payload, zlib_level))

    pres_bytes = len(zlib.compress(np.packbits(presence.reshape(-1)).tobytes(), zlib_level))

    # raw per-frame control (no tracking): each frame's boxes, quantised, concatenated
    raw_syms: list[int] = []
    n_sites_total = 0
    for s in per_frame_sites:
        s = np.asarray(s, dtype=np.float64)
        n_sites_total += int(s.shape[0])
        if s.shape[0] == 0:
            continue
        qb = np.round(s[:, :4] / q).astype(np.int64)
        raw_syms.extend(int(v) for v in qb.reshape(-1))
    raw_zz = _zigzag(np.array(raw_syms, dtype=np.int64)) if raw_syms else np.zeros(0, np.uint64)
    raw_bytes = len(zlib.compress(raw_zz.astype("<u4").tobytes(), zlib_level))

    return SiteStreamBytes(
        tracked_bytes=tracked_bytes,
        raw_perframe_bytes=raw_bytes,
        presence_bytes=pres_bytes,
        n_sites_total=n_sites_total,
        K=K,
        P=P,
        quant_px=q,
        provenance={
            "tracked_total_with_presence": tracked_bytes + pres_bytes,
            "n_matched": tracked.n_matched,
            "zlib_level": int(zlib_level),
        },
    )


# --------------------------------------------------------------------------- #
# Render (the geometry-coverage tell).                                         #
# --------------------------------------------------------------------------- #
def render_sites_to_mask(
    sites: np.ndarray, *, h: int = SEG_H, w: int = SEG_W
) -> np.ndarray:
    """Rasterise ``(S,>=4)`` boxes (cx,cy,bw,bh) to an ``(h,w)`` bool Movable mask (axis-aligned).

    The site GENERATOR: draw each box as a filled rectangle. Deterministic; clipped to the frame.
    Used to MEASURE the geometry-coverage loss of the sparse-site approximation vs the GT Movable
    (IoU / recall), the honest lossy tell — the site coder trades bitmap rate for a box-approx mask.
    """

    sites = np.asarray(sites, dtype=np.float64)
    mask = np.zeros((h, w), dtype=bool)
    if sites.shape[0] == 0:
        return mask
    for row in sites:
        cx, cy, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        # Invert the extract convention EXACTLY (cx = 0.5*(x0+x1-1), bw = x1-x0) so an
        # integer box round-trips to IoU 1: x0 = cx - (bw-1)/2, x1 = x0 + bw.
        wbox = round(bw)
        hbox = round(bh)
        x0 = round(cx - (wbox - 1) / 2.0)
        y0 = round(cy - (hbox - 1) / 2.0)
        x0c = int(np.clip(x0, 0, w - 1))
        y0c = int(np.clip(y0, 0, h - 1))
        x1c = int(np.clip(x0 + wbox, x0c + 1, w))
        y1c = int(np.clip(y0 + hbox, y0c + 1, h))
        mask[y0c:y1c, x0c:x1c] = True
    return mask
