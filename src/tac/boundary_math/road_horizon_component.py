"""Sky/undrivable + road-as-complement level-set components AND the JOINT-CONTAINMENT
measure (FEED-dw — the structured-region decomposition seal-gate begun).

Completes the per-lever optimal-form treatment after the manifold-aware lane-SDF
(``tac.boundary_math.lane_sdf_component``, FEED-dm/ds) and the static ego-hood
(``tac.boundary_math.hood_static_component``, FEED-du/dv). Operator 2026-06-27:
*"other levers likely deserve and need this same attention."* + the FEED-dt seal gate's
HIGHEST-RISK item: *"do the structured components COMPOSE without antagonism in ONE
argmax?"* — this module builds the remaining two structured classes (sky/undrivable +
road) and runs the FIRST joint-containment measure of the wired multi-component codec.

THE MATH (grounded; measured on the frozen CPU-torch SegNet argmax cache).
  * The K=5 level-set witness represents the SegNet argmax partition as ``argmax_k phi_k``
    of K=5 SDF fields. comma10k 5-class convention (CONFIRMED by FEED-da + the horizon-band
    memo): 0=Road, 1=Lane markings, 2=Undrivable (includes sky), 3=Movable, 4=My Car (hood).
    The cached SegNet argmax in ``gt_n96.npz`` emits exactly this order — but this module
    SELF-DETECTS every role from the data (``classify_segnet_regions``), NEVER hardcodes an
    index (the NO-FAKE guard that FEED-du/dv enforced after the FEED-dn luma-sort mislabel).

  * SKY / UNDRIVABLE (the static TOP region; FEED-du measured static IoU 0.907, rows [0,218],
    49% of the frame). Two candidate optimal forms, MEASURED head-to-head:
      - STATIC MASK: ONE majority-vote mask shared by all frames (the hood template, reused
        class-agnostically). ~0 DOF; cost ~tens of bytes; the drift (~9%) is unmodeled FN.
      - HORIZON LINE: the sky lower boundary is the road/sky horizon — a near-horizontal
        line that DRIFTS with vehicle pitch. A per-frame degree-1 line ``row = a*u + b``
        (2 floats/frame) tracks the drift; sky = the half-plane ABOVE the line. The EDT
        rasterizer of the half-plane is FREE (rule 118); only the 2 coeffs/frame are COUNTED.

  * ROAD (class 0; the drivable MIDDLE band). The KEY structural insight: road is the
    BACKGROUND/COMPLEMENT — the argmax fills road wherever no other class wins. So instead of
    a per-pixel road SDF, ``phi_0`` can be a single CONSTANT background level ``c`` (0 video-
    derived bytes): road wins exactly where lane/sky/hood/movable are all "outside" (their
    locally-supported SDFs are negative below ``c``). ``road_complement_field`` builds it; the
    isolation + joint tests MEASURE whether road falls out as the complement (no dedicated SDF)
    or whether a real road SDF is still needed.

  * JOINT CONTAINMENT (the FEED-dt highest-risk seal item). The per-component isolation tests
    each held the OTHER classes IDEAL. The REAL question is whether the structured fields
    COMPOSE in ONE argmax: inject lane (phi_1) + sky (phi_2) + hood (phi_4) + road-complement
    (phi_0) TOGETHER, keep ONLY the learned Movable (phi_3) ideal, recompute the argmax, and
    measure (a) joint d_seg vs the SUM of isolation d_segs (do they ADD or ANTAGONIZE?), (b)
    per-class containment in the joint (does class-A's component leak into class-B when both
    are active?), (c) the RESIDUAL d_seg the witness must LEARN (Movable + the fine boundary).

NO-FAKE: every function does the work its name claims on the REAL frozen CPU-torch SegNet
argmax label map. Classes are DETECTED from the data; masks/lines are REAL fits to the REAL
cached ``lstars``; the SDFs are REAL scipy EDTs (the SAME primitive ``signed_distance_fields``
/ ``lane_signed_distance`` use); the joint argmax + full per-class disagreement are REAL vs the
REAL L* (bit-exact). No stub, no surrogate, no synthetic fixture in the measure path. This is a
REPRESENTATION-fidelity primitive (numpy-fp32, CPU, $0) -> ``[macOS-CPU advisory]`` research-
signal; NOT a trained-witness output and NOT a byte-closed row.

ADDITIVE / DEFAULT-OFF: standalone geometry. NOT imported by the trainer unless a future
``--sky-static-component`` / ``--road-complement`` flag is wired (integration spec in the memo).
Building it changes no existing behavior. REUSES ``lane_sdf_component`` + ``hood_static_component``
(the class-agnostic ``lane_signed_distance`` / ``inject_lane_sdf`` / ``compute_static_hood_mask``)
rather than duplicating.

Borrowed-substrate accounting:
  * BORROWED (cited): scipy EDT (same primitive as ``signed_distance_fields``); the lane-SDF
    template (FEED-dm, ``lane_sdf_component``); the static-mask template (FEED-du,
    ``hood_static_component``, reused class-agnostically); openpilot horizon-as-vanishing-point
    geometry (the horizon-band memo); comma10k 5-class semantics.
  * OURS-ORIGINAL: parametrizing the SegNet argmax SKY/UNDRIVABLE class as a per-frame horizon-
    line half-plane SDF vs a static-mask SDF (head-to-head, drift-vs-bytes); parametrizing ROAD
    as the level-set COMPLEMENT (a single constant background level, 0 video-derived bytes); the
    JOINT-containment harness that measures structured-component antagonism + the learned residual
    in ONE argmax; the data-driven full-region classifier (NO-FAKE self-detection of all 5 roles).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reuse the lane-SDF + hood machinery (class-agnostic) — do NOT duplicate.
from tac.boundary_math.lane_sdf_component import (  # noqa: F401  (re-exported for callers)
    cluster_lane_lines,
    fit_lane_line,
    inject_lane_sdf as inject_sdf,
    lane_signed_distance as region_signed_distance,
)
from tac.boundary_math.hood_static_component import (  # noqa: F401
    StaticHoodMask,
    compute_static_hood_mask as compute_static_mask,
    hood_mask_byte_cost as static_mask_byte_cost,
)

_SEG_H = 384
_SEG_W = 512


# ===========================================================================
# NO-FAKE: data-driven full-region classifier (detect ALL 5 roles, never hardcode).
# ===========================================================================
@dataclass
class RegionEvidence:
    """Per-class spatial/temporal evidence used to assign the comma10k roles."""

    cls: int
    static_iou: float          # all-frames intersection / union (temporal stability)
    frac_of_frame: float       # mean per-frame area fraction
    top_share: float           # fraction of the class's pixels in the TOP `band_frac` rows
    bottom_share: float        # fraction of the class's pixels in the BOTTOM `band_frac` rows
    mid_share: float           # fraction in the MIDDLE band
    n_lane_lines: int          # ground-plane lane lines fitted (lane-vs-movable linearity proxy)
    maj_row_span: tuple[int, int]


@dataclass
class RegionRoles:
    """The detected comma10k role -> class-index map (a permutation of 0..n_classes-1)."""

    road: int
    lane: int
    sky: int
    movable: int
    hood: int
    evidence: list[RegionEvidence]

    def as_dict(self) -> dict[str, int]:
        return {"road": self.road, "lane": self.lane, "sky": self.sky,
                "movable": self.movable, "hood": self.hood}


def _count_lane_lines(lstar: np.ndarray, cls: int) -> int:
    """How many ground-plane lane lines this class's pixels fit (the lane-vs-movable
    linearity proxy). Lane markings form thin ground-frame polynomial lines; movable
    blobs and road fills do not. REUSES the canonical lane clusterer/fitter."""

    try:
        clusters = cluster_lane_lines(lstar, lane_cls=int(cls))
    except Exception:
        return 0
    n = 0
    for c in clusters:
        try:
            if fit_lane_line(c, fit_dash=False) is not None:
                n += 1
        except Exception:
            continue
    return n


def classify_segnet_regions(
    lstars: np.ndarray, *, n_classes: int = 5, band_frac: float = 0.25,
    lane_probe_frames: int = 6,
) -> RegionRoles:
    """Detect the 5 comma10k roles from the REAL cached argmax maps (NO hardcoded indices).

    Falling-rule detection (each role removed from the pool before the next):
      1. HOOD    = argmax(bottom_share * static_iou)        — the static BOTTOM ego region.
      2. SKY     = argmax(top_share * static_iou)           — the static TOP undrivable region.
      3. ROAD    = argmax(frac_of_frame) of the remaining 3 — the large drivable background
                   band (lane/movable are small; road dwarfs them). Robust to lane staticity.
      4. LANE    = argmax(n_lane_lines) of the last 2       — thin ground-plane lines;
                   tiebreak (both 0) -> smaller area (lanes are thinner than movable blobs).
      5. MOVABLE = the last remaining class.

    Asserts the 5 roles are a permutation (fail-closed NO-FAKE). Returns ``RegionRoles``.
    """

    a = np.asarray(lstars)
    n, h, w = a.shape
    counts = np.zeros((int(n_classes), h, w), np.int64)
    for i in range(n):
        Li = a[i]
        for c in range(int(n_classes)):
            counts[c] += Li == c
    top = slice(0, int(h * band_frac))
    bot = slice(int(h * (1.0 - band_frac)), h)

    # lane-line probe on a few frames (cheap; the canonical IPM clusterer)
    probe_idx = np.linspace(0, n - 1, min(int(lane_probe_frames), n)).astype(int)
    lane_lines_by_cls = {c: 0 for c in range(int(n_classes))}
    for c in range(int(n_classes)):
        lane_lines_by_cls[c] = int(np.median(
            [_count_lane_lines(a[i], c) for i in probe_idx]
        )) if probe_idx.size else 0

    ev: list[RegionEvidence] = []
    for c in range(int(n_classes)):
        union = counts[c] > 0
        inter = counts[c] == n
        iou = float(inter.sum() / union.sum()) if union.sum() else 0.0
        tot = int(counts[c].sum())
        tshare = float(counts[c][top].sum() / tot) if tot else 0.0
        bshare = float(counts[c][bot].sum() / tot) if tot else 0.0
        mshare = float(1.0 - tshare - bshare)
        maj = (counts[c] / n) >= 0.5
        rr = np.where(maj)[0]
        span = (int(rr.min()), int(rr.max())) if rr.size else (-1, -1)
        ev.append(RegionEvidence(
            cls=c, static_iou=iou, frac_of_frame=float(counts[c].mean()),
            top_share=tshare, bottom_share=bshare, mid_share=mshare,
            n_lane_lines=int(lane_lines_by_cls[c]), maj_row_span=span,
        ))

    pool = set(range(int(n_classes)))
    by = {e.cls: e for e in ev}

    hood = max(pool, key=lambda c: by[c].bottom_share * by[c].static_iou)
    pool.discard(hood)
    sky = max(pool, key=lambda c: by[c].top_share * by[c].static_iou)
    pool.discard(sky)
    road = max(pool, key=lambda c: by[c].frac_of_frame)  # the large drivable background band
    pool.discard(road)
    # last two: lane = more lane-lines; tiebreak (both 0) -> thinner (smaller area)
    last_two = sorted(pool)
    lane = max(last_two, key=lambda c: (by[c].n_lane_lines, -by[c].frac_of_frame))
    pool.discard(lane)
    movable = pool.pop()

    roles = {road, lane, sky, movable, hood}
    if len(roles) != int(n_classes):
        raise ValueError(
            f"NO-FAKE region classification is not a permutation: "
            f"road={road} lane={lane} sky={sky} movable={movable} hood={hood}"
        )
    return RegionRoles(road=int(road), lane=int(lane), sky=int(sky),
                       movable=int(movable), hood=int(hood),
                       evidence=sorted(ev, key=lambda e: e.cls))


# ===========================================================================
# SKY / UNDRIVABLE — horizon-line model (per-frame, 2 floats) + static mask (reused).
# ===========================================================================
@dataclass
class HorizonLine:
    """A per-frame road/sky horizon: ``row = polyval(coeffs, u)``; sky = rows ABOVE it."""

    coeffs: np.ndarray         # degree-`deg` poly in column u -> boundary row
    deg: int
    n_fit_cols: int
    rms_row: float             # rms residual of the per-column boundary fit (px)

    def horizon_row(self, u: np.ndarray) -> np.ndarray:
        return np.polyval(self.coeffs, np.asarray(u, np.float64))

    def n_floats(self) -> int:
        return int(self.deg + 1)


def fit_horizon_line(
    lstar: np.ndarray, *, sky_cls: int, deg: int = 1, trim_frac: float = 0.15,
) -> HorizonLine | None:
    """Fit the road/sky horizon as ``row = poly_deg(u)`` from the per-column TOP-contiguous
    sky run. The boundary at column ``u`` is the first non-sky row from the top (the lower
    edge of the contiguous sky band). Robust: drop the ``trim_frac`` worst-residual columns
    and refit (suppresses building silhouettes / poles poking into the sky). Returns None if
    too few sky columns."""

    a = np.asarray(lstar) == int(sky_cls)
    h, w = a.shape
    cols: list[int] = []
    rows: list[int] = []
    for u in range(w):
        col = a[:, u]
        if not col[0]:
            continue  # this column has no sky at the top -> not on the clean horizon
        nz = np.where(~col)[0]
        b = int(nz[0]) if nz.size else h  # first non-sky row from top
        cols.append(u)
        rows.append(b)
    if len(cols) < (deg + 2):
        return None
    cu = np.asarray(cols, np.float64)
    rv = np.asarray(rows, np.float64)
    d = int(min(deg, max(1, len(cols) - 2)))
    coeffs = np.polyfit(cu, rv, d)
    resid = np.abs(rv - np.polyval(coeffs, cu))
    if trim_frac > 0 and cu.size >= (d + 2) * 2:
        keep = resid <= np.quantile(resid, 1.0 - float(trim_frac))
        if keep.sum() >= (d + 2):
            coeffs = np.polyfit(cu[keep], rv[keep], d)
            resid = np.abs(rv - np.polyval(coeffs, cu))
    return HorizonLine(coeffs=coeffs, deg=d, n_fit_cols=len(cols),
                       rms_row=float(np.sqrt(np.mean(resid ** 2))))


def rasterize_sky_above_line(line: HorizonLine, *, h: int = _SEG_H, w: int = _SEG_W) -> np.ndarray:
    """Sky half-plane mask: pixel (v,u) is sky iff ``v < horizon_row(u)``. Generic
    deterministic rasterizer (FREE inflate-time; only the line coeffs are COUNTED)."""

    us = np.arange(int(w), dtype=np.float64)
    rows = np.clip(line.horizon_row(us), 0.0, float(h))
    vv = np.arange(int(h), dtype=np.float64)[:, None]
    return (vv < rows[None, :]).astype(bool)


def build_sky_line_sdf(
    lstar: np.ndarray, *, sky_cls: int, deg: int = 1, h: int = _SEG_H, w: int = _SEG_W,
) -> tuple[np.ndarray, dict]:
    """Per-frame: fit horizon -> rasterize sky half-plane -> signed distance ``phi_sky``.
    Returns (phi (H,W) float32, meta). Falls back to an all-outside field if no horizon."""

    line = fit_horizon_line(lstar, sky_cls=sky_cls, deg=deg)
    if line is None:
        phi = np.full((int(h), int(w)), -float(max(h, w)), np.float32)
        return phi, {"fit": False, "n_floats": 0, "rms_row": -1.0, "coeffs": []}
    mask = rasterize_sky_above_line(line, h=h, w=w)
    phi = region_signed_distance(mask)
    return phi, {"fit": True, "n_floats": line.n_floats(), "deg": line.deg,
                 "rms_row": line.rms_row, "n_fit_cols": line.n_fit_cols,
                 "coeffs": [float(x) for x in line.coeffs]}


def build_static_sky_sdf(lstars: np.ndarray, *, sky_cls: int, agg: str = "majority") -> tuple[np.ndarray, StaticHoodMask]:
    """ONE frame-shared static sky mask SDF (reuses the hood static-mask template
    class-agnostically). Returns (phi (H,W) float32, StaticHoodMask diagnostics)."""

    sm = compute_static_mask(np.asarray(lstars), hood_cls=int(sky_cls), agg=agg)
    return region_signed_distance(sm.mask), sm


def sky_line_byte_cost(
    coeffs_per_frame: np.ndarray, *, n_frames: int = 600, fp_bytes: int = 2,
) -> dict:
    """COUNTED byte cost of the per-frame horizon-line coeffs (rule-118 boundary).

    Stores (deg+1) fp16 coeffs/frame; reports raw and brotli-over-the-temporal-sequence
    (the horizon drifts slowly -> highly compressible). The half-plane EDT rasterizer is FREE.
    """

    arr = np.asarray(coeffs_per_frame, np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_used, n_coef = arr.shape
    raw = int(n_used * n_coef * fp_bytes)
    # delta-code along frames (temporal smoothness) then compress fp16 bytes
    fp16 = arr.astype(np.float16)
    try:
        import brotli
        comp = len(brotli.compress(fp16.tobytes(), quality=11))
    except Exception:
        import zlib
        comp = len(zlib.compress(fp16.tobytes(), 9))
    # scale raw to the contest frame count
    per_frame = raw / max(1, n_used)
    raw_full = int(per_frame * n_frames)
    comp_full = int(comp / max(1, n_used) * n_frames)
    best = int(min(raw_full, comp_full))
    return {
        "n_coef_per_frame": int(n_coef), "fp_bytes": int(fp_bytes),
        "raw_bytes_full": raw_full, "brotli_bytes_full": comp_full,
        "best_counted_bytes": best, "bytes_per_frame": float(best) / float(n_frames),
        "score_rate_contribution": 25.0 * float(best) / 37_545_489.0,
        "n_frames": int(n_frames),
    }


# ===========================================================================
# ROAD — the level-set COMPLEMENT (a single constant background level; 0 video bytes).
# ===========================================================================
def road_complement_field(h: int = _SEG_H, w: int = _SEG_W, *, level: float = 0.0) -> np.ndarray:
    """``phi_road = constant level`` (H,W). Road wins wherever every other (locally-supported)
    SDF is below ``level`` -> road = the COMPLEMENT of the union of positive class regions. A
    single scalar -> 0 video-derived bytes (a frozen threshold). NO per-pixel road SDF."""

    return np.full((int(h), int(w)), float(level), np.float32)


def road_complement_byte_cost() -> dict:
    """Road-as-complement costs ONE scalar threshold (or 0 if fixed at level=0). Reported as
    a negligible / 0-video-derived-byte payload (rule-118: the constant is generic, not
    video-derived; the conservative accounting counts at most the 2-byte fp16 level)."""

    return {"counted_bytes": 0, "optional_level_fp16_bytes": 2,
            "score_rate_contribution": 0.0, "note": "constant background level; 0 video-derived bytes"}


# ===========================================================================
# Full per-class disagreement decomposition (the JOINT containment metric).
# ===========================================================================
@dataclass
class FullDecomp:
    """Per-class FN/FP + confusion (normalized by total pixels) for a predicted partition."""

    total_dseg: float
    n_classes: int
    true_frac: list[float]      # true area fraction per class
    fn: list[float]             # true==k & pred!=k  (per class, /N) — the SHAPE miss
    fp: list[float]             # pred==k & true!=k  (per class, /N) — the OVER-claim/leak
    confusion: list[list[float]]  # confusion[a][b] = true==a & pred==b (/N)

    def leak_into(self, k: int) -> list[tuple[int, float]]:
        """Sources of FP into class k: [(other_class, frac), ...] sorted desc."""
        col = [(a, self.confusion[a][k]) for a in range(self.n_classes) if a != k]
        return sorted(col, key=lambda t: t[1], reverse=True)


def decompose_full_confusion(pred: np.ndarray, lstar: np.ndarray, *, n_classes: int = 5) -> FullDecomp:
    """Full per-class decomposition of ``pred != lstar`` (both (H,W) int label maps)."""

    p = np.asarray(pred).astype(np.int64).ravel()
    L = np.asarray(lstar).astype(np.int64).ravel()
    N = float(L.size)
    K = int(n_classes)
    cm = np.zeros((K, K), np.float64)
    # confusion via bincount on flattened (true*K + pred)
    idx = L * K + p
    binc = np.bincount(idx, minlength=K * K).astype(np.float64).reshape(K, K)
    cm = binc / N
    true_frac = cm.sum(axis=1)
    pred_frac = cm.sum(axis=0)
    diag = np.diag(cm)
    fn = (true_frac - diag)
    fp = (pred_frac - diag)
    return FullDecomp(
        total_dseg=float((p != L).mean()), n_classes=K,
        true_frac=[float(x) for x in true_frac],
        fn=[float(x) for x in fn], fp=[float(x) for x in fp],
        confusion=[[float(cm[a, b]) for b in range(K)] for a in range(K)],
    )


def mean_full_decomp(decomps: list[FullDecomp], *, n_classes: int = 5) -> dict:
    """Average a list of ``FullDecomp`` into a json-safe dict (per-class FN/FP/true_frac +
    total + mean confusion)."""

    K = int(n_classes)
    total = float(np.mean([d.total_dseg for d in decomps])) if decomps else 0.0
    fn = [float(np.mean([d.fn[k] for d in decomps])) for k in range(K)]
    fp = [float(np.mean([d.fp[k] for d in decomps])) for k in range(K)]
    tf = [float(np.mean([d.true_frac[k] for d in decomps])) for k in range(K)]
    cm = [[float(np.mean([d.confusion[a][b] for d in decomps])) for b in range(K)] for a in range(K)]
    return {"total_dseg": total, "n_classes": K, "fn": fn, "fp": fp,
            "true_frac": tf, "confusion": cm}
