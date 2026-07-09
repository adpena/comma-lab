# SPDX-License-Identifier: MIT
"""Road+Undrivable edge-centric bulk-boundary field — v8 increment-1 SCAFFOLD.

``research_only=True`` — **SCAFFOLD (v8 increment-1, P-C-gated).** This is the composed
SKELETON + byte-close stub for the ONE new carrier of v8 increment-1 (the design draft
``.omx/research/v8_increment1_design_draft_20260709.md`` §2). It is **NOT the increment-1
build**: the real build is gated on probe **P-C** (the "interiors near-free" go/no-go, UNRUN,
memory-gated behind #205). Nothing here is trained, byte-closed against ``upstream/evaluate.py``,
or promotable. All numbers are ``[macOS-CPU advisory · NON-PROMOTABLE]`` numpy-fp32 geometry on
the frozen SegNet argmax cache. Pointer contest-CPU 0.19110 is UNMOVED — a SCAFFOLD moves nothing.

THE THESIS (draft §0). Increment-1 de-shares the **Road separatrix** into ONE edge-centric
bulk-boundary field carrying {Road, Undrivable} — the two classes whose shared boundary (the
road-edge / horizon curve) is ONE tie locus. Instead of two independent region fields (which pay
for the same curve twice), a SINGLE SDF-gauged scalar ``phi_bulk`` over the Road/Undriv bulk is
LIFTED to the two per-class SDF channels by a signed scale.

THE SIGNED LIFT (draft §2, VERIFIED). ``phi_Road = +s_R * phi_bulk`` and
``phi_Undriv = -s_U * phi_bulk`` where the SIGN of ``phi_bulk`` is the Road/Undriv side of the
road-edge/horizon curve (``phi_bulk > 0`` on the Road side, ``< 0`` on the Undrivable side). The
argmax over the two lifted channels is ``argmax(s_R*phi, -s_U*phi) == Road  iff  (s_R+s_U)*phi > 0
iff phi > 0`` for ANY ``s_R, s_U > 0`` — so within the Road∪Undriv bulk the lift reproduces the
Road/Undriv labels EXACTLY (``bulk_signed_lift_argmax``), and the per-side scales ``s_R, s_U`` +
per-class biases ``b_c`` are the only per-side freedom under the ``|grad phi| = 1`` eikonal gauge.

MULTI-COMPONENT ROAD (draft §2, the binding scaffold constraint). Undrivable is single-connected
(background), but **Road is multi-component in ~37% of frames (up to 3 blobs)**. A *signed* SDF
represents multi-blob Road correctly — its zero-set is simply multiple closed curves and the
``+EDT`` interior is positive inside EVERY blob — so the lift stays valid, but this scaffold is
**multi-component-Road-aware BY CONSTRUCTION** (it never parametrizes Road as one blob; the field
is a signed distance to the Road MASK, blob-count-agnostic). ``road_component_stats`` MEASURES the
multi-blob fraction so the assumption is verified, not asserted.

NO-FAKE. Road + Undrivable class indices are **SELF-DETECTED** from the spatial/area/static
signature (``road_horizon_component.classify_segnet_regions``; the ``sky`` role == comma10k class-2
Undrivable which INCLUDES sky) — NEVER hardcoded. The signed distance is the REAL scipy EDT
(the same ``lane_signed_distance`` primitive); the argmax-parity checks are the REAL popcount vs
the REAL cached ``lstars``; the byte cost is the REAL brotli of the REAL Road-mask temporal stack.
No stub returns canonical markers in place of the work its name claims.

BYTE COST (draft §1, review-F). The 20-50 KB in the draft table is **CONJECTURED** (a GUESS,
~1 order-of-magnitude; NO RD curve ``d_bulk(B)`` fitted). ``bulk_boundary_byte_cost`` MEASURES the
COUNTED cost of the full Road-vs-Undriv boundary representation (brotli over the temporal mask
stack, the conservative FULL-boundary number) and labels it MEASURED; the coarse-grid + INR-annulus
reduction (draft §2, #308 "interiors near-free") that would push it toward the low end is
CONJECTURED here and is exactly what P-C must measure.

DECOUPLING GUARD (draft §2 review-E). The per-class tie bias ``b_c`` (``compute_bulk_tie_bias``,
riding ``laguerre_logit_offset.damped_newton_ot_offsets``) MUST be calibrated OUTSIDE the
scorer-gradient loop — else the global-bias coupling re-enters as theft-like behavior. This
scaffold computes ``b_c`` closed-form (no scorer gradient) by construction.

BORROWED-SUBSTRATE (NO-FAKE #7):
  * BORROWED (cited): scipy EDT via ``lane_sdf_component.lane_signed_distance`` /
    ``lever_b_levelset_generator.signed_distance_fields``; the self-detect region classifier
    (``road_horizon_component.classify_segnet_regions``, FEED-dw); the Laguerre / semi-discrete-OT
    per-class offset (``laguerre_logit_offset``, #218); the ``_row_span_encode`` byte-close pattern
    (``hood_static_component``, FEED-du); comma10k 5-class semantics.
  * OURS-ORIGINAL: representing the SHARED Road/Undrivable separatrix as ONE signed bulk-boundary
    scalar lifted to the two per-class SDF channels by a signed per-side scale (the edge-centric
    de-sharing of the Road hub), multi-component-Road-aware by construction; the full-boundary
    byte-cost MEASUREMENT that turns the draft's CONJECTURED 20-50 KB into a real number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COMPOSE the in-tree substrate (do NOT reinvent the math — import + call).
from tac.boundary_math.lane_sdf_component import lane_signed_distance
from tac.boundary_math.laguerre_logit_offset import (
    apply_offset_to_sdf_bias,
    damped_newton_ot_offsets,
    power_diagram_argmax,
)
from tac.boundary_math.lever_b_levelset_generator import signed_distance_fields
from tac.boundary_math.road_horizon_component import (
    RegionRoles,
    classify_segnet_regions,
)

_SEG_H = 384
_SEG_W = 512


class RoadUndrivBulkFieldError(ValueError):
    """Raised on malformed bulk-boundary-field inputs."""


# ---------------------------------------------------------------------------
# NO-FAKE self-detection: Road + Undrivable class indices from the data.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadUndrivRoles:
    """The self-detected Road + Undrivable class indices (+ the full role map)."""

    road: int
    undriv: int
    roles: RegionRoles

    def as_dict(self) -> dict[str, int]:
        return {"road": self.road, "undriv": self.undriv}


def identify_road_undriv_classes(lstars: np.ndarray, *, n_classes: int = 5) -> RoadUndrivRoles:
    """SELF-DETECT the Road + Undrivable class indices from the cached argmax maps (NO hardcode).

    Reuses ``road_horizon_component.classify_segnet_regions`` (spatial/area/static self-detection).
    The comma10k canonical order is Road0 / Lane1 / Undriv2 / Movable3 / MyCar4, but the trained
    net's cached ordering must never be trusted as a constant — the classifier detects Road as the
    large drivable background band and Undrivable as the static TOP region (the ``sky`` role, which
    INCLUDES sky per comma10k class-2 semantics). Returns ``RoadUndrivRoles``.
    """

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise RoadUndrivBulkFieldError(f"lstars must be (N,H,W) or (H,W); got shape {a.shape}")
    roles = classify_segnet_regions(a, n_classes=int(n_classes))
    # Undrivable (comma10k class 2, includes sky) == the self-detected static-TOP `sky` role.
    return RoadUndrivRoles(road=int(roles.road), undriv=int(roles.sky), roles=roles)


# ---------------------------------------------------------------------------
# Multi-component Road measurement (verify the "37% of frames multi-blob" claim).
# ---------------------------------------------------------------------------
def road_component_stats(
    lstars: np.ndarray, *, road_cls: int, min_area_frac: float = 0.002,
) -> dict:
    """MEASURE the connected-component structure of the Road region per frame.

    A component counts as SIGNIFICANT if its area >= ``min_area_frac`` of the frame (drops the
    speckle a raw connected-component count would over-report). Returns a dict with the fraction of
    frames that are multi-blob, the mean/max significant blob count, and the per-frame histogram —
    the MEASURED verification of the draft §2 "Road multi-component in 37.2% of frames" claim that
    forces the field to be multi-blob-aware.
    """

    from scipy import ndimage

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    n, h, w = a.shape
    thresh_px = float(min_area_frac) * float(h * w)
    counts: list[int] = []
    for i in range(n):
        road = a[i] == int(road_cls)
        lab, nlab = ndimage.label(road)
        if nlab == 0:
            counts.append(0)
            continue
        sizes = np.bincount(lab.ravel())[1:]  # drop background bin 0
        counts.append(int((sizes >= thresh_px).sum()))
    c = np.asarray(counts, np.int64)
    multi = c >= 2
    hist = {int(k): int((c == k).sum()) for k in range(int(c.max()) + 1)} if c.size else {}
    return {
        "n_frames": int(n),
        "road_cls": int(road_cls),
        "min_area_frac": float(min_area_frac),
        "frac_frames_multi_component": float(multi.mean()) if c.size else 0.0,
        "mean_significant_blobs": float(c.mean()) if c.size else 0.0,
        "max_significant_blobs": int(c.max()) if c.size else 0,
        "blob_count_histogram": hist,
    }


# ---------------------------------------------------------------------------
# The ONE bulk-boundary field + the VERIFIED signed lift.
# ---------------------------------------------------------------------------
def build_bulk_boundary_field(
    lstar: np.ndarray, *, road_cls: int, undriv_cls: int,
) -> tuple[np.ndarray, dict]:
    """Build ``phi_bulk`` (H,W) float32 — the ONE signed bulk-boundary scalar.

    ``phi_bulk = signed distance to the Road MASK`` (``+EDT`` inside every Road blob, ``-EDT``
    outside). Within the Road∪Undrivable bulk the only boundary is the Road/Undrivable separatrix
    (Undrivable is the sole large non-Road region there), so ``phi_bulk``'s sign IS the Road/Undriv
    side and its zero-set IS the road-edge/horizon curve. Multi-component Road is handled BY
    CONSTRUCTION — the ``+EDT`` interior is positive inside EVERY blob, so no single-blob assumption
    is made. Returns ``(phi_bulk, meta)``; ``meta`` carries the bulk-region area fractions.
    """

    a = np.asarray(lstar)
    if a.ndim != 2:
        raise RoadUndrivBulkFieldError(f"lstar must be (H,W); got shape {a.shape}")
    road = a == int(road_cls)
    undriv = a == int(undriv_cls)
    phi_bulk = lane_signed_distance(road)  # +EDT inside road (all blobs), -EDT outside
    meta = {
        "road_cls": int(road_cls),
        "undriv_cls": int(undriv_cls),
        "road_frac": float(road.mean()),
        "undriv_frac": float(undriv.mean()),
        "bulk_frac": float((road | undriv).mean()),
    }
    return phi_bulk.astype(np.float32), meta


def signed_lift(
    phi_bulk: np.ndarray, *, s_road: float = 1.0, s_undriv: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The VERIFIED signed lift: ``(phi_Road, phi_Undriv) = (+s_road*phi_bulk, -s_undriv*phi_bulk)``.

    ``s_road, s_undriv`` must be > 0 (the per-side scales under the eikonal gauge). Returns the two
    per-class SDF channels; ``bulk_signed_lift_argmax`` proves the argmax reproduces the labels.
    """

    if not (float(s_road) > 0.0 and float(s_undriv) > 0.0):
        raise RoadUndrivBulkFieldError("s_road and s_undriv must be > 0 (per-side eikonal scales)")
    p = np.asarray(phi_bulk, np.float32)
    return (float(s_road) * p).astype(np.float32), (-float(s_undriv) * p).astype(np.float32)


def bulk_signed_lift_argmax(
    phi_bulk: np.ndarray, *, road_cls: int, undriv_cls: int,
    s_road: float = 1.0, s_undriv: float = 1.0,
) -> np.ndarray:
    """Two-channel argmax of the signed lift -> the Road/Undrivable label map (H,W) int64.

    ``argmax(s_road*phi, -s_undriv*phi) == Road iff phi > 0`` for any positive scales, so this
    returns ``road_cls`` where ``phi_bulk >= 0`` and ``undriv_cls`` elsewhere. Within the Road∪
    Undrivable bulk this equals the true labels EXACTLY (the draft §2 VERIFIED property).
    """

    p_road, p_undriv = signed_lift(phi_bulk, s_road=s_road, s_undriv=s_undriv)
    stacked = np.stack([p_road, p_undriv], axis=-1)  # (H,W,2): 0->road-side, 1->undriv-side
    idx = np.argmax(stacked, axis=-1)
    out = np.where(idx == 0, int(road_cls), int(undriv_cls)).astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Composition: inject the lifted Road/Undriv channels into the K-field stack.
# ---------------------------------------------------------------------------
def inject_bulk_field(
    phi_hwk: np.ndarray, phi_bulk: np.ndarray, *, road_cls: int, undriv_cls: int,
    s_road: float = 1.0, s_undriv: float = 1.0, mode: str = "replace",
    bulk_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inject the lifted Road/Undrivable channels into the K-field ``phi_hwk`` (H,W,K).

    Mirrors the ``lane_sdf_component.inject_lane_sdf`` injection pattern.
      * ``mode="replace"``: set ``phi[...,road] = +s_road*phi_bulk`` and
        ``phi[...,undriv] = -s_undriv*phi_bulk`` (the de-shared bulk field REPLACES both channels).
      * ``mode="bias"``: ADD the lifted channels to the existing ones (prior/bias option).

    THE BULK MASK (the ±phi lift's scope; round-1 self-review fix). The signed lift is inherently a
    WITHIN-BULK representation: ``phi_undriv = -s_U*phi_bulk`` goes spuriously large-positive deep in
    Lane/Movable/MyCar territory (all non-Road reads as the "Undriv side" of ``phi_bulk``), so a raw
    whole-frame replace would let Undrivable steal the hood. In the increment-1 composition the bulk
    EXTENT is resolved by the OTHER carriers' tropical argmax; here, pass ``bulk_mask`` (True on the
    Road∪Undrivable bulk) so OUTSIDE the bulk BOTH lifted channels are set deeply negative
    (``-max(H,W)``) and the thin-class carriers win. Without ``bulk_mask`` this is the raw whole-frame
    lift (correct only for the SCOPED 2-way ``bulk_signed_lift_argmax`` use, NOT full-field argmax).
    """

    out = np.asarray(phi_hwk, np.float32).copy()
    if out.ndim != 3:
        raise RoadUndrivBulkFieldError(f"phi_hwk must be (H,W,K); got shape {out.shape}")
    h, w, k = out.shape
    if not (0 <= int(road_cls) < k and 0 <= int(undriv_cls) < k):
        raise RoadUndrivBulkFieldError(f"road/undriv class out of [0,K={k})")
    p_road, p_undriv = signed_lift(phi_bulk, s_road=s_road, s_undriv=s_undriv)
    if bulk_mask is not None:
        m = np.asarray(bulk_mask, bool)
        if m.shape != (h, w):
            raise RoadUndrivBulkFieldError(f"bulk_mask shape {m.shape} != (H,W)=({h},{w})")
        deep = -float(max(h, w))
        p_road = np.where(m, p_road, deep).astype(np.float32)
        p_undriv = np.where(m, p_undriv, deep).astype(np.float32)
    if mode == "replace":
        out[..., int(road_cls)] = p_road
        out[..., int(undriv_cls)] = p_undriv
    elif mode == "bias":
        out[..., int(road_cls)] = out[..., int(road_cls)] + p_road
        out[..., int(undriv_cls)] = out[..., int(undriv_cls)] + p_undriv
    else:
        raise RoadUndrivBulkFieldError(f"mode must be 'replace' or 'bias', got {mode!r}")
    return out


def build_ideal_kfield_with_bulk(
    lstar: np.ndarray, *, n_classes: int = 5, road_cls: int, undriv_cls: int,
    s_road: float = 1.0, s_undriv: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: build the ideal K-field from ``lstar`` then inject the bulk field.

    Returns ``(phi_hwk, phi_bulk)``. ``argmax(phi_hwk)`` reproduces ``lstar`` (the injection test).
    """

    a = np.asarray(lstar)
    phi_hwk = signed_distance_fields(a, int(n_classes))
    phi_bulk, _ = build_bulk_boundary_field(a, road_cls=road_cls, undriv_cls=undriv_cls)
    bulk_mask = (a == int(road_cls)) | (a == int(undriv_cls))  # the bulk EXTENT (composition input)
    injected = inject_bulk_field(
        phi_hwk, phi_bulk, road_cls=road_cls, undriv_cls=undriv_cls,
        s_road=s_road, s_undriv=s_undriv, mode="replace", bulk_mask=bulk_mask,
    )
    return injected, phi_bulk


# ---------------------------------------------------------------------------
# Decoupling guard (review-E): the per-class tie bias b_c, OUT of the scorer loop.
# ---------------------------------------------------------------------------
def compute_bulk_tie_bias(
    phi_hwk: np.ndarray, target_masses: np.ndarray, *, tau: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Closed-form per-class tie bias ``b_c`` (Laguerre / semi-discrete OT) — OUT of scorer loop.

    Rides ``laguerre_logit_offset.damped_newton_ot_offsets`` (a REAL Newton solve, NOT a sweep, NOT
    a scorer-gradient step) so the global-bias coupling the draft §2 review-E flags NEVER re-enters
    as theft-like behavior. Fold ``b_c`` into ``out_sdf.bias`` byte-free via
    ``laguerre_logit_offset.apply_offset_to_sdf_bias``. Returns ``(b, info)``.
    """

    p = np.asarray(phi_hwk, np.float64)
    if p.ndim < 1:
        raise RoadUndrivBulkFieldError("phi_hwk must have a class axis")
    b, info = damped_newton_ot_offsets(p, np.asarray(target_masses, np.float64), tau=float(tau))
    return b, info


def apply_bulk_tie_bias(
    params: dict[str, np.ndarray], offsets: np.ndarray, *, bias_key: str = "out_sdf.bias",
) -> dict[str, np.ndarray]:
    """Byte-free fold of the tie bias into ``out_sdf.bias`` (thin reuse of the Laguerre helper)."""

    return apply_offset_to_sdf_bias(params, offsets, bias_key=bias_key)


# ---------------------------------------------------------------------------
# Byte-close stub — MEASURED full-boundary representation cost (rule-118 boundary).
# ---------------------------------------------------------------------------
def _road_row_span_encode(road_mask: np.ndarray) -> bytes:
    """Encode a (multi-blob) Road mask as a per-row run-length span table.

    For each row, store the number of Road runs (uint8) then each run's ``[u_lo, u_hi]`` as uint16.
    Multi-blob Road produces multiple runs per row — this encoding is blob-count-agnostic (the
    scaffold's multi-component-awareness at the byte-close surface). The FREE inflate-time
    rasterizer re-expands the spans; only these bytes are COUNTED. Returns the raw (pre-brotli)
    span bytes. Round-trips bit-exactly via ``_road_row_span_decode``.
    """

    m = np.asarray(road_mask, bool)
    h, w = m.shape
    out = bytearray()
    out += int(h).to_bytes(2, "little")
    out += int(w).to_bytes(2, "little")
    for r in range(h):
        row = m[r]
        # run starts/ends via diff on the padded row
        padded = np.concatenate([[False], row, [False]])
        d = np.diff(padded.astype(np.int8))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1  # inclusive last True index
        n_runs = int(starts.size)
        out += int(min(n_runs, 255)).to_bytes(1, "little")
        for s, e in zip(starts[:255], ends[:255]):
            out += int(s).to_bytes(2, "little") + int(e).to_bytes(2, "little")
    return bytes(out)


def _road_row_span_decode(data: bytes) -> np.ndarray:
    """Inverse of ``_road_row_span_encode`` -> the bool Road mask (H,W). Bit-exact round-trip."""

    mv = memoryview(data)
    h = int.from_bytes(mv[0:2], "little")
    w = int.from_bytes(mv[2:4], "little")
    m = np.zeros((h, w), bool)
    off = 4
    for r in range(h):
        n_runs = int(mv[off]); off += 1
        for _ in range(n_runs):
            s = int.from_bytes(mv[off:off + 2], "little"); off += 2
            e = int.from_bytes(mv[off:off + 2], "little"); off += 2
            m[r, s:e + 1] = True
    return m


def bulk_boundary_byte_cost(
    lstars: np.ndarray, *, road_cls: int, undriv_cls: int, n_frames: int = 600,
) -> dict:
    """MEASURE the COUNTED byte cost of the full Road-vs-Undrivable boundary representation.

    The bulk field only needs the Road-vs-Undrivable SIGN per frame (the Road MASK; the bulk
    EXTENT vs Lane/Movable/MyCar is supplied by the OTHER carriers in composition). Two encodings
    are measured and the best reported, brotli-compressed over the TEMPORAL stack (the Road region
    drifts slowly -> temporally compressible):
      * packed bitmap of the Road mask, all frames concatenated -> brotli;
      * per-row multi-run span RLE (multi-blob-aware), all frames concatenated -> brotli.

    Returns per-frame amortized bytes over ``n_frames`` + the score-rate contribution
    (``25 * bytes / 37_545_489``). This is the conservative FULL-boundary number, labelled MEASURED.
    The coarse-grid + INR-annulus reduction (draft §2, #308) that would push it lower is CONJECTURED
    and is exactly what probe P-C must measure — it is NOT applied here.
    """

    a = np.asarray(lstars)
    if a.ndim == 2:
        a = a[None]
    n = a.shape[0]
    bitmap_parts: list[bytes] = []
    span_parts: list[bytes] = []
    for i in range(n):
        road = (a[i] == int(road_cls))
        bitmap_parts.append(np.packbits(road).tobytes())
        span_parts.append(_road_row_span_encode(road))
    bitmap_blob = b"".join(bitmap_parts)
    span_blob = b"".join(span_parts)
    try:
        import brotli

        bitmap_comp = len(brotli.compress(bitmap_blob, quality=11))
        span_comp = len(brotli.compress(span_blob, quality=11))
        coder = "brotli"
    except Exception:
        import zlib

        bitmap_comp = len(zlib.compress(bitmap_blob, 9))
        span_comp = len(zlib.compress(span_blob, 9))
        coder = "zlib"
    best_measured = int(min(bitmap_comp, span_comp))
    per_frame = float(best_measured) / float(max(1, n))
    full = int(round(per_frame * n_frames))
    return {
        "n_frames_measured": int(n),
        "coder": coder,
        "bitmap_brotli_bytes_measured": int(bitmap_comp),
        "row_span_brotli_bytes_measured": int(span_comp),
        "best_measured_bytes": best_measured,
        "measured_bytes_per_frame": per_frame,
        "full_bytes_at_n_frames_MEASURED": full,
        "score_rate_contribution_MEASURED": 25.0 * float(full) / 37_545_489.0,
        "n_frames_amortized": int(n_frames),
        "conjectured_note": (
            "FULL-boundary cost (MEASURED). The coarse-grid + INR-annulus 'interiors near-free' "
            "reduction (draft §2 #308) is CONJECTURED, NOT applied here, and is what P-C measures."
        ),
        "draft_conjectured_band_kb": (20, 50),
    }


# ---------------------------------------------------------------------------
# Full-frame Laguerre-reweighted argmax (thin reuse; observability).
# ---------------------------------------------------------------------------
def bulk_argmax_with_bias(phi_hwk: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``argmax_k(phi_k + b_k)`` over the injected K-field (thin reuse of the Laguerre helper)."""

    return power_diagram_argmax(np.asarray(phi_hwk), np.asarray(offsets))
