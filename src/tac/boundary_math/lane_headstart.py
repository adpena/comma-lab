# SPDX-License-Identifier: MIT
"""Wyner-Ziv lane-centerline HEAD-START for the v2 task-space witness (openpilot lane prior).

WHAT THIS IS (operator: "we need the openpilot head start too").
  The v2 binding term is the LANE-SURVIVAL d_seg residual (F4): sub-0.15 ⟺
  trained-through-R lane d_seg ≤ 1.23e-3; sub-0.19 ⟺ ≤ 1.63e-3. The frozen contest
  SegNet argmax has a thin, ragged class-1 (Lane) band whose POSITION is a smooth
  ground-frame polynomial (openpilot represents lanes natively as degree-4 polys,
  ``POLY_PATH_DEGREE = 4``). a99f41f0 measured the necessary-condition floor: a
  polynomial fit DIRECTLY to the frozen-SegNet lane argmax (an ORACLE; openpilot's
  own lanes can only be worse) reproduces the lane CENTERLINE to sub-pixel and
  recovers ~64% of the lane d_seg (drop-all-lanes 0.005885 -> centerline-base
  0.00214), but cannot reach sub-0.15 alone (the remaining ~0.00214 is the ragged
  ±1px boundary = the ~8-dim lane-orbit manifold a TRAINED generator must encode).

  This module turns that finding into USABLE infrastructure: it is the
  **Wyner-Ziv source-coding-with-side-information-at-the-decoder** decomposition for
  the lane class.  inflate.py's decoder has FREE side information ``Y`` = the lane
  centerline (a GENERIC rasterizer over stored polynomial coeffs), correlated with
  the source ``X`` = the frozen-SegNet lane argmax.  Wyner-Ziv (1976) says we code
  only the CONDITIONAL residual ``X - E[X|Y]``.  So the eventual trained-through-R
  witness STARTS from the conditioned base (lane d_seg 0.00214) and only learns the
  residual down toward ≤1.23e-3 -- a ~64% head-start over learning ``X`` from
  scratch (0.00588).

  This is the ResidualGauge cell ``CONDITIONAL_ON_LANE_PRIOR`` (Wyner-Ziv).

THE THREE PIECES (Part 1 of the openpilot-head-start build):
  1. ``fit_centerlines`` -- the COMPRESS-TIME analyzer (the base ``Y``): fit a
     low-order polynomial per lane component to the frozen-SegNet lane argmax.
     VIDEO-DERIVED -> COUNTED payload (the coeffs), tiny (see ``estimate_base_bytes``).
  2. ``rasterize_centerlines`` -- the DETERMINISTIC base rasterizer: the FREE
     GENERIC algorithm that lives in inflate.py (read coeffs -> evaluate poly ->
     paint band).  ``build_lane_headstart`` CONFIRMS this base alone reproduces the
     ~0.00214 lane d_seg floor (the NO-FAKE reproduce-it-yourself check).
  3. ``compute_lane_residual`` / ``apply_lane_residual`` -- the conditional-residual
     pipeline: ``X - Y`` (signed int8 ∈ {-1,0,+1}; what the witness must learn
     through R) and its EXACT inverse ``base ⊕ residual -> X`` (round-trips
     bit-exactly).  This DEFINES the witness's job for the GPU run: the small
     residual from the sub-pixel base, not the full lane from scratch.

rule-118 (COMPLIANCE BOUNDARY, binding):
  * the centerline RASTERIZER is a GENERIC algorithm -> FREE in inflate.py.
  * the stored centerline COEFFS are VIDEO-DERIVED -> COUNTED in archive.zip (tiny).
  * openpilot is a COMPRESS-TIME ANALYZER only (it may SOURCE the coeffs offline);
    supercombo.onnx (~30 MB neural weights) is NEVER shipped in inflate.py (Part 2
    confirms the clean compress-time-analyzer path; path (b) ship-the-net is AVOIDED).
  * the LEARNED ragged-boundary residual is the real (PENDING-GPU) counted budget.

NO-FAKE: every function does the work its name claims on the REAL frozen CPU-torch
SegNet argmax label map (``lstars`` in the GT cache).  The polynomials are REAL
``numpy.polyfit`` fits to the REAL class-1 pixels; the base d_seg is the REAL
symmetric XOR vs the REAL argmax; the residual round-trip is bit-exact (tested).
No stub, no surrogate, no synthetic fixture, no fabricated number.  This is a
REPRESENTATION-fidelity / coding primitive (numpy-fp32, CPU, $0) -> ``[macOS-CPU
advisory] / [macOS research-signal]`` -- it is NOT a trained-witness output and NOT
a byte-closed exact-eval row.  score_claim=false; promotable=false.  Pointer UNMOVED
contest-CPU 0.19110.  This is MEANS toward a byte-closed exact row, never an end.

Borrowed-substrate accounting:
  * BORROWED (cited): openpilot's degree-4 lane-polynomial representation
    (``POLY_PATH_DEGREE``, commaai/openpilot); numpy.polyfit; scipy connected
    components / morphology; the oracle-floor measurement method + dash-bridge knob
    from ``tools/measure_lane_polynomial_shape_floor.py`` (a99f41f0); Wyner-Ziv 1976
    source coding with side information at the decoder; comma10k class-1 = Lane.
  * OURS-ORIGINAL: packaging the oracle centerline as a reusable Wyner-Ziv
    conditioning base + the exact signed-residual (X-Y) pipeline that DEFINES the
    trained witness's reduced job (encode only X-E[X|Y]), the exact-preserving
    centerline-delta byte/entropy estimate, and the ResidualGauge cost cell.

Sister surfaces (anti-duplication): ``tac.boundary_math.margin_conditional_residual``
conditions on the decoder-RECOMPUTED MARGIN field (a different side-information
source); ``tac.boundary_math.lane_sdf_component`` is the level-set SDF form of the
same lane geometry.  This module is the openpilot-centerline Wyner-Ziv side-info
specifically, and is complementary to both.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

try:  # scipy is available (1.17.1); used only at COMPRESS time (fit), never inflate.
    from scipy import ndimage
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "lane_headstart requires scipy.ndimage (compress-time component labelling)"
    ) from exc

# canonical comma10k order [Road, Lane, Undrivable, Movable, MyCar] -> Lane = 1.
LANE_CLASS: int = 1
# F4 thresholds: sub-0.15 / sub-0.19 <=> trained-through-R lane d_seg <= these.
SUB015_LANE_DSEG: float = 1.23e-3
SUB019_LANE_DSEG: float = 1.63e-3
# rate term denominator (upstream/evaluate.py: 25 * archive_bytes / this).
_RATE_DENOM: int = 37_545_489


# ───────────────────────────── the COUNTED payload ─────────────────────────────
@dataclass(frozen=True)
class LaneCenterline:
    """One lane component as the openpilot-native polynomial form (the COUNTED base).

    ``axis='col_of_row'`` fits ``col = poly(row)`` (near-vertical strokes, the common
    case); ``axis='row_of_col'`` fits ``row = poly(col)`` (near-horizontal, distant
    lanes by the vanishing point).  ``coeffs`` is the ``numpy.polyfit`` vector (length
    ``eff_degree + 1``).  ``half_width`` is the (compress-time chosen) paint radius.
    """

    axis: str  # 'col_of_row' | 'row_of_col'
    x_start: int
    x_end: int
    coeffs: tuple[float, ...]
    half_width: int

    @property
    def eff_degree(self) -> int:
        return len(self.coeffs) - 1

    def samples(self) -> np.ndarray:
        """Rounded integer centerline coordinate per consecutive x (the FREE eval)."""
        xs = np.arange(self.x_start, self.x_end + 1, dtype=np.float64)
        ys = np.polyval(np.asarray(self.coeffs, dtype=np.float64), xs)
        return np.rint(ys).astype(np.int64)


# ───────────────────────── compress-time analyzer (the base Y) ──────────────────
def _fit_one(
    rows: np.ndarray, cols: np.ndarray, degree: int
) -> tuple[np.ndarray, str, int, int]:
    """Fit one component; return (coeffs, axis, x_min, x_max). Deterministic."""
    row_ext = int(rows.max() - rows.min())
    col_ext = int(cols.max() - cols.min())
    if row_ext >= col_ext:
        axis, x, y = "col_of_row", rows.astype(np.float64), cols.astype(np.float64)
    else:
        axis, x, y = "row_of_col", cols.astype(np.float64), rows.astype(np.float64)
    eff_deg = int(min(degree, max(1, np.unique(x).size - 1)))
    coeffs = np.polyfit(x, y, eff_deg)
    return coeffs, axis, int(x.min()), int(x.max())


def _paint(cl: LaneCenterline, grid_h: int, grid_w: int, out: np.ndarray) -> None:
    """Paint one centerline band onto ``out`` (bool HxW). The FREE generic algorithm."""
    xs = np.arange(cl.x_start, cl.x_end + 1)
    ys = cl.samples()
    hw = cl.half_width
    if cl.axis == "col_of_row":  # x=row, y=col
        for x, y in zip(xs, ys):
            if 0 <= x < grid_h:
                c0, c1 = max(0, y - hw), min(grid_w - 1, y + hw)
                if c0 <= c1:
                    out[x, c0 : c1 + 1] = True
    else:  # x=col, y=row
        for x, y in zip(xs, ys):
            if 0 <= x < grid_w:
                r0, r1 = max(0, y - hw), min(grid_h - 1, y + hw)
                if r0 <= r1:
                    out[r0 : r1 + 1, x] = True


def rasterize_centerlines(
    centerlines: list[LaneCenterline], grid_h: int, grid_w: int
) -> np.ndarray:
    """Deterministic base lane mask from stored centerlines (the inflate-time FREE op).

    This is the GENERIC rasterizer that lives in inflate.py.  Given only the COUNTED
    coeffs it reproduces the conditioning base ``Y`` bit-exactly on any host.
    """
    out = np.zeros((grid_h, grid_w), dtype=bool)
    for cl in centerlines:
        _paint(cl, grid_h, grid_w, out)
    return out


def fit_centerlines(
    lane_mask: np.ndarray,
    degree: int = 4,
    *,
    dash_bridge_rows: int = 1,
    min_component_pixels: int = 12,
    max_half_width: int = 4,
) -> list[LaneCenterline]:
    """COMPRESS-TIME: fit the centerline base ``Y`` to a frozen-SegNet lane mask.

    ``dash_bridge_rows=1`` (default) = the per-dash variant (each dashed stroke is its
    own component; no gap-overpaint) -- the a99f41f0 0.00214 head-start.  A larger
    value bridges dashes into continuous lines (the 0.00341 variant).  ``half_width``
    is chosen per component as the ORACLE width minimising the local XOR vs the true
    lane pixels (a compress-time decision, stored in the COUNTED payload).
    """
    if lane_mask.dtype != np.bool_:
        lane_mask = lane_mask.astype(bool)
    gh, gw = lane_mask.shape
    if lane_mask.sum() == 0:
        return []
    if dash_bridge_rows > 1:
        bridge = np.ones((dash_bridge_rows, 1), dtype=bool)
        bridged = ndimage.binary_closing(lane_mask, structure=bridge)
    else:
        bridged = lane_mask
    labels, nlab = ndimage.label(bridged)

    centerlines: list[LaneCenterline] = []
    for lab in range(1, nlab + 1):
        comp = lane_mask & (labels == lab)  # fit to ORIGINAL (unbridged) pixels
        rr, cc = np.where(comp)
        if rr.size < min_component_pixels:
            continue
        coeffs, axis, x_min, x_max = _fit_one(rr, cc, degree)
        # ORACLE half-width search (compress-time), local bbox for a fair XOR.
        pad = max_half_width + 1
        br0, br1 = max(0, int(rr.min()) - pad), min(gh, int(rr.max()) + pad + 1)
        bc0, bc1 = max(0, int(cc.min()) - pad), min(gw, int(cc.max()) + pad + 1)
        true_local = comp[br0:br1, bc0:bc1]
        best_hw, best_xor = 0, None
        for hw in range(0, max_half_width + 1):
            trial = LaneCenterline(
                axis=axis,
                x_start=x_min,
                x_end=x_max,
                coeffs=tuple(float(c) for c in coeffs),
                half_width=hw,
            )
            m = np.zeros((gh, gw), dtype=bool)
            _paint(trial, gh, gw, m)
            xv = int(np.logical_xor(true_local, m[br0:br1, bc0:bc1]).sum())
            if best_xor is None or xv < best_xor:
                best_xor, best_hw = xv, hw
        centerlines.append(
            LaneCenterline(
                axis=axis,
                x_start=x_min,
                x_end=x_max,
                coeffs=tuple(float(c) for c in coeffs),
                half_width=best_hw,
            )
        )
    return centerlines


def lane_base_from_argmax(
    lstars_frame: np.ndarray, degree: int = 4, **fit_kw
) -> tuple[np.ndarray, np.ndarray, list[LaneCenterline]]:
    """Convenience: argmax frame -> (true lane mask, base lane mask, centerlines)."""
    lane = lstars_frame == LANE_CLASS
    gh, gw = lane.shape
    centerlines = fit_centerlines(lane, degree, **fit_kw)
    base = rasterize_centerlines(centerlines, gh, gw)
    return lane, base, centerlines


# ───────────────────── the conditional-residual pipeline (X - E[X|Y]) ───────────
def compute_lane_residual(base_lane: np.ndarray, target_lane: np.ndarray) -> np.ndarray:
    """The Wyner-Ziv conditional residual ``X - Y`` as signed int8 ∈ {-1,0,+1}.

    ``+1`` = a lane pixel the base MISSED (false-negative; witness must ADD).
    ``-1`` = a base pixel that is NOT lane (false-positive; witness must REMOVE).
    `` 0`` = base already correct.  The nonzero support fraction is EXACTLY the
    conditioned base lane d_seg -- the residual the trained-through-R witness must
    fix (start 0.00214, target ≤1.23e-3).  This is the witness's reduced job.
    """
    x = target_lane.astype(np.int8)
    y = base_lane.astype(np.int8)
    return (x - y).astype(np.int8)


def apply_lane_residual(base_lane: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Exact inverse: ``base ⊕ residual -> X`` (the reconstructed lane mask).

    ``X = Y + (X - Y)``.  Round-trips ``compute_lane_residual`` bit-exactly because
    ``base ∈ {0,1}`` and ``residual ∈ {-1,0,+1}`` always sum back into ``{0,1}``.
    """
    recon = base_lane.astype(np.int8) + residual.astype(np.int8)
    return recon.astype(bool)


def residual_support_frac(residual: np.ndarray) -> float:
    """Nonzero fraction of the residual = the conditioned base lane d_seg."""
    return float(np.count_nonzero(residual)) / float(residual.size)


# ───────────────────────────── byte / entropy estimate ─────────────────────────
def _shannon_entropy_bits(values: np.ndarray) -> float:
    """Empirical Shannon entropy (bits/symbol) of an integer array."""
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p)).sum())


def serialize_centerlines_delta(
    centerlines_per_frame: list[list[LaneCenterline]],
) -> bytes:
    """Deterministic exact-preserving byte serialization of the centerline-delta form.

    Per component: axis(1B) + x_start(int16) + length(int16) + half_width(1B) +
    col0(int16) + (length-1) Δ (int16).  Concatenated across the corpus, frame-
    delimited.  This is a REAL byte stream whose zlib size MEASURES the achievable
    centerline rate INCLUDING cross-frame temporal redundancy (lanes near-static).
    Exact-preserving: deserializing + rasterizing reproduces the base bit-exactly.
    """
    buf = bytearray()
    for frame in centerlines_per_frame:
        buf += struct.pack("<H", len(frame))  # n components this frame
        for cl in frame:
            ys = cl.samples()
            buf += struct.pack("<B", 0 if cl.axis == "col_of_row" else 1)
            buf += struct.pack("<h", int(cl.x_start))
            buf += struct.pack("<H", int(ys.size))
            buf += struct.pack("<B", int(cl.half_width))
            buf += struct.pack("<h", int(ys[0]))
            if ys.size >= 2:
                d = np.clip(np.diff(ys), -32768, 32767).astype("<i2")
                buf += d.tobytes()
    return bytes(buf)


def deserialize_and_rasterize(
    raw: bytes, grid_h: int, grid_w: int
) -> list[np.ndarray]:
    """Decode the centerline-delta byte stream back to per-frame base masks.

    The inverse of :func:`serialize_centerlines_delta`.  This IS a candidate
    inflate.py decode path: read the COUNTED byte stream -> rasterize the FREE
    base.  Used to PROVE the serialization is exact-preserving (tested).
    """
    out: list[np.ndarray] = []
    off = 0
    while off < len(raw):
        (ncomp,) = struct.unpack_from("<H", raw, off)
        off += 2
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        for _ in range(ncomp):
            axis_b = raw[off]
            off += 1
            (x_start,) = struct.unpack_from("<h", raw, off)
            off += 2
            (length,) = struct.unpack_from("<H", raw, off)
            off += 2
            hw = raw[off]
            off += 1
            (col0,) = struct.unpack_from("<h", raw, off)
            off += 2
            if length >= 2:
                deltas = np.frombuffer(raw, dtype="<i2", count=length - 1, offset=off)
                off += (length - 1) * 2
                ys = np.concatenate(([col0], col0 + np.cumsum(deltas))).astype(np.int64)
            else:
                ys = np.array([col0], dtype=np.int64)
            xs = np.arange(x_start, x_start + length)
            if axis_b == 0:  # col_of_row
                for x, y in zip(xs, ys):
                    if 0 <= x < grid_h:
                        c0, c1 = max(0, y - hw), min(grid_w - 1, y + hw)
                        if c0 <= c1:
                            mask[x, c0 : c1 + 1] = True
            else:  # row_of_col
                for x, y in zip(xs, ys):
                    if 0 <= x < grid_w:
                        r0, r1 = max(0, y - hw), min(grid_h - 1, y + hw)
                        if r0 <= r1:
                            mask[r0 : r1 + 1, x] = True
        out.append(mask)
    return out


def estimate_base_bytes(
    centerlines_per_frame: list[list[LaneCenterline]], n_target_frames: int = 600
) -> dict:
    """COUNTED-byte estimate for the centerline base over a corpus.

    Two clean, HONEST estimates (both ``[macOS research-signal]`` advisory):
      * ``parametric_*`` -- the openpilot-native form: ``(eff_degree+1)`` coeffs/comp
        at 16 bits + a small per-component header.  Fixed-bit upper bound.
      * ``delta_entropy_*`` -- EXACT-PRESERVING: store the rounded centerline as
        ``col0`` + first-differences (Δ along the curve), entropy-coded at the
        MEASURED empirical Shannon entropy of the Δ stream.  This reconstructs the
        rasterized base bit-exactly (so it preserves the 0.00214 floor EXACTLY) and
        gives a measured iid entropy floor.  This is the recommended cell number.

    The further temporal/ground-frame optimization (lanes near-static in the ground
    frame; per-frame coeffs ~rank-few -> ~5-15% of iid) reaching ~0.5-5 KB for 600
    frames is the a99f41f0 estimate (MED confidence, NOT measured here).
    """
    n_obs = max(1, len(centerlines_per_frame))
    # gather streams across the corpus
    comp_counts: list[int] = []
    coeff_counts: list[int] = []
    hw_vals: list[int] = []
    deltas: list[int] = []
    col0s: list[int] = []
    lengths: list[int] = []
    for frame in centerlines_per_frame:
        comp_counts.append(len(frame))
        for cl in frame:
            coeff_counts.append(cl.eff_degree + 1)
            hw_vals.append(cl.half_width)
            ys = cl.samples()
            lengths.append(int(ys.size))
            col0s.append(int(ys[0]))
            if ys.size >= 2:
                deltas.extend(np.diff(ys).tolist())

    comps_pf = float(np.mean(comp_counts)) if comp_counts else 0.0
    coeffs_pf = float(np.sum(coeff_counts)) / n_obs
    samples_pf = float(np.sum(lengths)) / n_obs

    # parametric (openpilot-native) fixed-bit: coeffs@16b + header(axis1 + x_start9
    # + length9 + half_width3) per component.
    parametric_bits_pf = coeffs_pf * 16.0 + comps_pf * (1.0 + 9.0 + 9.0 + 3.0)
    parametric_bytes_pf = parametric_bits_pf / 8.0

    # delta-entropy (exact-preserving): per component header(axis1 + x_start9 +
    # length9 + half_width3 + col0_9) + (length-1) Δ @ H(Δ).
    h_delta = _shannon_entropy_bits(np.asarray(deltas, dtype=np.int64))
    delta_payload_pf = max(0.0, (samples_pf - comps_pf)) * h_delta
    delta_bits_pf = comps_pf * (1.0 + 9.0 + 9.0 + 3.0 + 9.0) + delta_payload_pf
    delta_bytes_pf = delta_bits_pf / 8.0

    # MEASURED temporal-aware estimate: zlib over the exact-preserving delta stream
    # (captures cross-frame redundancy -- lanes near-static over the highway clip).
    raw = serialize_centerlines_delta(centerlines_per_frame)
    zbytes = len(zlib.compress(raw, 9))
    zbytes_pf = zbytes / n_obs
    achievable_bytes_pf = min(parametric_bytes_pf, delta_bytes_pf, zbytes_pf)
    return {
        "n_frames_observed": len(centerlines_per_frame),
        "components_per_frame_mean": comps_pf,
        "coeffs_per_frame_mean": coeffs_pf,
        "centerline_samples_per_frame_mean": samples_pf,
        "delta_entropy_bits_per_symbol": h_delta,
        "parametric_bytes_per_frame": parametric_bytes_pf,
        "parametric_bytes_600": parametric_bytes_pf * n_target_frames,
        "delta_entropy_bytes_per_frame": delta_bytes_pf,
        "delta_entropy_bytes_600": delta_bytes_pf * n_target_frames,
        "zlib_temporal_bytes_per_frame": zbytes_pf,
        "zlib_temporal_bytes_600": zbytes_pf * n_target_frames,
        "achievable_iid_bytes_600": achievable_bytes_pf * n_target_frames,
        "achievable_iid_rate_term_600": (
            25.0 * achievable_bytes_pf * n_target_frames / _RATE_DENOM
        ),
        "temporal_groundframe_target_bytes_600": "0.5-5 KB (a99f41f0 estimate, MED, NOT measured here)",
        "authority": "[macOS research-signal] advisory; score_claim=false",
    }


# ───────────────────────────── the driver + gauge cell ─────────────────────────
@dataclass
class LaneHeadstartCorpusResult:
    n_frames: int
    grid_h: int
    grid_w: int
    degree: int
    dash_bridge_rows: int
    from_scratch_lane_dseg: float  # drop-all-lanes baseline (witness w/ no lane signal)
    base_lane_dseg: float  # conditioned base (the a99f41f0 ~0.00214 head-start)
    recovered_frac: float  # (from_scratch - base) / from_scratch  (~0.64)
    residual_support_frac_mean: float  # == base_lane_dseg (the witness's reduced job)
    roundtrip_exact: bool  # base ⊕ residual == target on every frame
    iou_mean: float
    base_passes_sub015: bool
    base_passes_sub019: bool
    bytes: dict
    centerlines_per_frame: list = field(default_factory=list, repr=False)


def build_lane_headstart(
    lstars: np.ndarray,
    degree: int = 4,
    *,
    dash_bridge_rows: int = 1,
    min_component_pixels: int = 12,
    max_half_width: int = 4,
    n_target_frames: int = 600,
) -> LaneHeadstartCorpusResult:
    """Build + measure the full lane head-start over a frozen-SegNet argmax corpus.

    Reproduces the conditioned base lane d_seg (the NO-FAKE check), measures the
    Wyner-Ziv residual support + verifies the exact round-trip, and estimates the
    COUNTED centerline bytes.  Deterministic; ``[macOS research-signal]`` advisory.
    """
    n, gh, gw = lstars.shape
    total = gh * gw
    xors: list[float] = []
    lane_fracs: list[float] = []
    ious: list[float] = []
    supports: list[float] = []
    roundtrip_ok = True
    centerlines_per_frame: list[list[LaneCenterline]] = []

    for i in range(n):
        lane, base, cls = lane_base_from_argmax(
            lstars[i],
            degree,
            dash_bridge_rows=dash_bridge_rows,
            min_component_pixels=min_component_pixels,
            max_half_width=max_half_width,
        )
        centerlines_per_frame.append(cls)
        lane_fracs.append(float(lane.mean()))

        residual = compute_lane_residual(base, lane)
        recon = apply_lane_residual(base, residual)
        if not np.array_equal(recon, lane):
            roundtrip_ok = False
        supports.append(residual_support_frac(residual))

        xor = np.logical_xor(lane, base)
        xors.append(float(xor.sum()) / total)
        inter = int(np.logical_and(lane, base).sum())
        union = int(np.logical_or(lane, base).sum())
        ious.append(float(inter) / union if union else 1.0)

    base_dseg = float(np.mean(xors))
    from_scratch = float(np.mean(lane_fracs))
    recovered = (from_scratch - base_dseg) / from_scratch if from_scratch else 0.0
    bytes_est = estimate_base_bytes(centerlines_per_frame, n_target_frames)

    return LaneHeadstartCorpusResult(
        n_frames=n,
        grid_h=gh,
        grid_w=gw,
        degree=degree,
        dash_bridge_rows=dash_bridge_rows,
        from_scratch_lane_dseg=from_scratch,
        base_lane_dseg=base_dseg,
        recovered_frac=recovered,
        residual_support_frac_mean=float(np.mean(supports)) if supports else 0.0,
        roundtrip_exact=roundtrip_ok,
        iou_mean=float(np.mean(ious)) if ious else 0.0,
        base_passes_sub015=base_dseg <= SUB015_LANE_DSEG,
        base_passes_sub019=base_dseg <= SUB019_LANE_DSEG,
        bytes=bytes_est,
        centerlines_per_frame=centerlines_per_frame,
    )


def gauge_cost_cell(result: LaneHeadstartCorpusResult) -> dict:
    """The ResidualGauge ``CONDITIONAL_ON_LANE_PRIOR`` (Wyner-Ziv) cost cell.

    The learned-residual cost is PENDING-GPU and is NOT fabricated here.
    """
    return {
        "gauge": "CONDITIONAL_ON_LANE_PRIOR",
        "lens": "Wyner-Ziv source coding with side info at decoder (1976)",
        "from_scratch_lane_dseg": result.from_scratch_lane_dseg,
        "base_lane_dseg": result.base_lane_dseg,
        "recovered_frac": result.recovered_frac,
        "residual_dseg_to_fix": result.base_lane_dseg,
        "residual_dseg_target_sub015": SUB015_LANE_DSEG,
        "residual_dseg_target_sub019": SUB019_LANE_DSEG,
        "base_bytes_600_achievable": result.bytes.get("achievable_iid_bytes_600"),
        "base_bytes_600_temporal_target": result.bytes.get(
            "temporal_groundframe_target_bytes_600"
        ),
        "learned_residual_cost": "PENDING-GPU",
        "roundtrip_exact": result.roundtrip_exact,
        "authority": "[macOS research-signal] advisory; score_claim=false; pointer UNMOVED 0.19110",
    }
