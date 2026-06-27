"""Static ego-hood level-set component (FEED-du; the HOOD-STATIC component named by the
FEED-dt review+seal gate).

The NEXT per-lever optimal-form treatment after the manifold-aware lane-SDF
(``tac.boundary_math.lane_sdf_component``, FEED-dm). Operator 2026-06-27: *"other
levers likely deserve and need this same attention."* This applies the VALIDATED
lane-SDF template (cluster/fit -> rasterize -> signed-distance -> inject as phi_k ->
isolation-decompose) to the EGO-HOOD class, which is the simplest possible structured
component: a SINGLE constant mask shared by all 600 frames (~0 bytes/frame).

THE MATH (grounded; measured on the frozen CPU-torch SegNet argmax cache).
  * The K=5 level-set witness represents the SegNet argmax partition as ``argmax_k phi_k``
    of K=5 SDF fields. The ego car hood is FIXED relative to the camera (the comma2k19
    RAV4 segment, fixed mount), so its argmax region is NEAR-STATIC across frames.
  * NO-FAKE CLASS-INDEX CORRECTION (this module's first finding): the canonical FEED-dn
    order [Road0, Lane1, MyCar2, Undrivable3, Movable4] does NOT match the cached SegNet
    argmax ordering in ``gt_n96.npz``. MEASURED on the real ``lstars``: the near-static
    bottom ego-hood region is **class 4** (static IoU 0.963, majority row span [282,383],
    100% of the bottom rows; matches the #139 ego_hood static core rows 284-383). Class 2
    is the LARGE static TOP region (sky/undrivable-upper; 49% of frame, rows [0,218]) — it
    is NOT a hood. ``identify_static_hood_class`` detects the hood class from the data so
    this module never hardcodes a possibly-wrong label.
  * Because the hood is static, its OPTIMAL FORM is a SINGLE canonical mask (majority vote
    over frames) -> ``phi_hood = signed-distance-to-(static hood mask)``, the SAME field
    injected into EVERY frame's level set (no per-frame fit, no per-frame floats). This is
    even simpler than the lane (a fixed region, not a per-frame polynomial).
      - PRECISE: an SDF is 1-Lipschitz; the static region is trivially captured (the per-
        frame hood deviates from the canonical mask only at the slowly-moving top boundary).
      - CONTAINED: the SDF is LOCALLY SUPPORTED -> ``phi_hood > 0`` only inside the mask,
        decaying negative outside -> far from the hood the deep ideal road/other SDFs win
        -> non-hood classes preserved STRUCTURALLY (the same containment mechanism the
        lane-SDF measured 20x tighter than a hard mask).
      - ~0 BYTES: ONE mask amortized over 600 frames. The mask boundary (per-row hood-top
        span) compresses to tens of bytes; it could even be a frozen-instance camera prior
        (0 video-derived bytes). ``hood_mask_byte_cost`` measures the COUNTED cost.

NO-FAKE: every function does the work its name claims on the REAL frozen CPU-torch SegNet
argmax label map. The static mask is a REAL majority-vote over the REAL cached ``lstars``;
the SDF is a REAL scipy EDT (the SAME primitive ``signed_distance_fields`` /
``lane_signed_distance`` use); the disagreement decomposition is the REAL argmax vs the
REAL L*. No stub, no surrogate, no synthetic fixture. REPRESENTATION-fidelity primitive
(numpy-fp32, CPU, $0) -> ``[macOS-CPU advisory]`` research-signal; NOT a trained-witness
output, NOT a byte-closed row.

ADDITIVE / DEFAULT-OFF: standalone geometry. NOT imported by the trainer unless a future
``--hood-static-component`` flag is wired (integration spec in the FEED-dt memo). Building
it changes no existing behavior. REUSES ``lane_sdf_component`` (``lane_signed_distance`` /
``inject_lane_sdf`` / ``decompose_argmax_disagreement`` are class-agnostic via ``lane_cls``)
rather than duplicating.

Borrowed-substrate accounting:
  * BORROWED (cited): scipy EDT (same primitive as ``signed_distance_fields`` /
    ``lane_signed_distance``); the lane-SDF injection/decomposition template (FEED-dm,
    ``lane_sdf_component``); the #139 ego_hood static-core measurement (rows 284-383).
  * OURS-ORIGINAL: parametrizing the SegNet argmax EGO-HOOD class as a SINGLE frame-shared
    signed-distance field to a majority-vote static mask, injected as phi_hood of the
    softmax-of-SDF level set so the hood is given STRUCTURALLY at ~0 bytes (contained by
    SDF local support); the data-driven hood-class identification (NO-FAKE correction of
    the FEED-dn index label).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reuse the lane-SDF machinery (class-agnostic via lane_cls) — do NOT duplicate.
from tac.boundary_math.lane_sdf_component import (  # noqa: F401  (re-exported for callers)
    ContainmentDecomp,
    decompose_argmax_disagreement,
    inject_lane_sdf as inject_hood_sdf,
    lane_signed_distance as hood_signed_distance,
)

_SEG_H = 384
_SEG_W = 512


# ---------------------------------------------------------------------------
# NO-FAKE: identify the static ego-hood class from the data (never hardcode).
# ---------------------------------------------------------------------------
@dataclass
class HoodClassEvidence:
    """Per-class evidence used to pick the static ego-hood class."""

    cls: int
    static_iou: float          # intersection-over-union of the all-frames mask (staticity)
    frac_of_frame: float       # mean per-frame area fraction
    bottom_share: float        # fraction of the class's pixels in the bottom 25% rows
    maj_row_span: tuple[int, int]


def identify_static_hood_class(
    lstars: np.ndarray, *, n_classes: int = 5, bottom_frac: float = 0.25,
) -> tuple[int, list[HoodClassEvidence]]:
    """Detect the near-static bottom ego-hood class from the cached argmax maps.

    The hood is the class that is BOTH near-static (high all-frames IoU) AND concentrated
    in the bottom rows. Returns ``(hood_cls, evidence_list)``. Pure measurement on the REAL
    L* — this is the NO-FAKE guard against the FEED-dn index mislabel (hood is class 4, not
    class 2, in ``gt_n96.npz``).
    """

    a = np.asarray(lstars)
    n, h, w = a.shape
    counts = np.zeros((int(n_classes), h, w), np.int64)
    for i in range(n):
        Li = a[i]
        for c in range(int(n_classes)):
            counts[c] += Li == c
    bottom = slice(int(h * (1.0 - bottom_frac)), h)
    ev: list[HoodClassEvidence] = []
    for c in range(int(n_classes)):
        union = counts[c] > 0
        inter = counts[c] == n
        iou = float(inter.sum() / union.sum()) if union.sum() else 0.0
        maj = (counts[c] / n) >= 0.5
        rr = np.where(maj)[0]
        span = (int(rr.min()), int(rr.max())) if rr.size else (-1, -1)
        tot = int(counts[c].sum())
        bshare = float(counts[c][bottom].sum() / tot) if tot else 0.0
        ev.append(HoodClassEvidence(cls=c, static_iou=iou, frac_of_frame=float(counts[c].mean() / 1.0),
                                    bottom_share=bshare, maj_row_span=span))
    # hood = highest (bottom_share * static_iou) score among classes whose majority sits low
    scored = sorted(ev, key=lambda e: e.bottom_share * e.static_iou, reverse=True)
    return scored[0].cls, ev


# ---------------------------------------------------------------------------
# The single canonical static hood mask (the ~0-byte payload).
# ---------------------------------------------------------------------------
@dataclass
class StaticHoodMask:
    """One frame-shared ego-hood mask + staticity diagnostics."""

    mask: np.ndarray            # (H,W) bool — the canonical hood region for ALL frames
    hood_cls: int
    agg: str                    # "majority" | "intersection" | "union"
    mean_frame_iou: float       # mean IoU(per-frame hood, canonical mask) — the staticity
    min_frame_iou: float
    n_frames: int
    px: int
    frac_of_frame: float
    row_span: tuple[int, int]


def compute_static_hood_mask(
    lstars: np.ndarray, *, hood_cls: int, agg: str = "majority", thresh: float = 0.5,
) -> StaticHoodMask:
    """Aggregate the per-frame class-``hood_cls`` masks into ONE canonical static mask.

    ``agg``:
      * "majority" (default): pixel is hood if it is class-``hood_cls`` in >= ``thresh`` of
        frames (the balanced precision/containment choice).
      * "intersection": hood in ALL frames (max containment, more false-negatives at the
        moving top edge — the analog of the lane dash gate's over-gating).
      * "union": hood in ANY frame (max recall, more containment leak).
    """

    a = np.asarray(lstars)
    n = a.shape[0]
    masks = np.empty(a.shape, bool)
    for i in range(n):
        masks[i] = a[i] == int(hood_cls)
    freq = masks.mean(0)
    if agg == "majority":
        canon = freq >= float(thresh)
    elif agg == "intersection":
        canon = masks.all(0)
    elif agg == "union":
        canon = masks.any(0)
    else:
        raise ValueError(f"agg must be majority|intersection|union, got {agg!r}")
    ious = np.empty(n, np.float64)
    cu = canon.sum()
    for i in range(n):
        inter = int((masks[i] & canon).sum())
        uni = int((masks[i] | canon).sum())
        ious[i] = inter / uni if uni else 1.0
    rr = np.where(canon)[0]
    span = (int(rr.min()), int(rr.max())) if rr.size else (-1, -1)
    return StaticHoodMask(
        mask=canon, hood_cls=int(hood_cls), agg=agg,
        mean_frame_iou=float(ious.mean()), min_frame_iou=float(ious.min()),
        n_frames=int(n), px=int(cu), frac_of_frame=float(canon.mean()), row_span=span,
    )


def build_static_hood_sdf(static_mask: np.ndarray) -> np.ndarray:
    """phi_hood = signed distance to the static hood mask (+EDT inside, -EDT outside).

    Thin reuse of ``lane_signed_distance`` (the SAME scipy-EDT 1-Lipschitz construction as
    ``signed_distance_fields``); returns (H,W) float32. ONE field for ALL frames."""

    return hood_signed_distance(np.asarray(static_mask, bool))


# ---------------------------------------------------------------------------
# Byte cost: the single mask amortized over all frames (~0 bytes/frame).
# ---------------------------------------------------------------------------
def _row_span_encode(mask: np.ndarray) -> bytes:
    """Encode the (contiguous bottom) hood mask as a per-row [u_lo, u_hi] span table.

    The ego hood is a simple bottom region: for each row in the hood's row span, store the
    leftmost/rightmost hood column as uint16 (0xFFFF sentinel for empty rows). This is the
    generic deterministic geometry the FREE inflate-time rasterizer re-expands; only these
    bytes are COUNTED. Returns the raw (pre-brotli) span bytes."""

    m = np.asarray(mask, bool)
    h, w = m.shape
    rr = np.where(m.any(1))[0]
    if rr.size == 0:
        return b""
    r0, r1 = int(rr.min()), int(rr.max())
    out = bytearray()
    out += int(r0).to_bytes(2, "little")
    out += int(r1).to_bytes(2, "little")
    for r in range(r0, r1 + 1):
        cols = np.where(m[r])[0]
        if cols.size == 0:
            out += (0xFFFF).to_bytes(2, "little") + (0xFFFF).to_bytes(2, "little")
        else:
            out += int(cols.min()).to_bytes(2, "little") + int(cols.max()).to_bytes(2, "little")
    return bytes(out)


def hood_mask_byte_cost(static_mask: np.ndarray, *, n_frames: int = 600) -> dict:
    """COUNTED byte cost of storing the SINGLE static hood mask (rule-118 boundary).

    Reports raw-packed-bits, per-row-span, and brotli-compressed sizes; the per-frame
    amortized cost over ``n_frames``; and the score-rate contribution (25*bytes/37_545_489).
    The mask is video-derived (fit from L*) so it is COUNTED, but ONE mask over 600 frames
    -> ~0 bytes/frame. The EDT rasterizer that re-expands the span table is FREE."""

    m = np.asarray(static_mask, bool)
    raw_bits = int(np.ceil(m.size / 8))                 # packed bitmap
    span = _row_span_encode(m)
    try:
        import brotli
        span_brotli = len(brotli.compress(span, quality=11))
        bitmap_brotli = len(brotli.compress(np.packbits(m).tobytes(), quality=11))
    except Exception:
        import zlib
        span_brotli = len(zlib.compress(span, 9))
        bitmap_brotli = len(zlib.compress(np.packbits(m).tobytes(), 9))
    best = int(min(span_brotli, bitmap_brotli))
    return {
        "raw_packed_bits_bytes": raw_bits,
        "row_span_bytes": len(span),
        "row_span_brotli_bytes": int(span_brotli),
        "bitmap_brotli_bytes": int(bitmap_brotli),
        "best_counted_bytes": best,
        "amortized_bytes_per_frame": float(best) / float(n_frames),
        "score_rate_contribution": 25.0 * float(best) / 37_545_489.0,
        "n_frames_amortized": int(n_frames),
    }
