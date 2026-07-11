"""v8 LANE rate crux — ground-frame anisotropic factorization of the lane carrier.

THE CRUX (v8 Laguerre feasibility memo, FEED-lag 2026-07-10): the Lane class is the
expensive carrier — 97 B/frame image-space (poly runs model) at only 0.835 recall /
0.601 IoU, and Road–Lane is 60% of all boundary pixels. The lane is a 1-D curve in
2-D image space, elongated along the road tangent: image-space coding FIGHTS the
anisotropy. This module exploits it instead, in three measured factors:

  1. GEOMETRY — undo the anisotropy. IPM to the ground frame (openpilot flat-ground
     chart, ``tac.boundary_math.lane_sdf_component`` — v_h=174 #327-optimal), where
     each lane line is a low-order polynomial. Parametrize each per-frame line by
     LATERAL KNOTS: lateral (m) at fixed forward distances — bijective with the
     deg-3 centerline polynomial (4 knots <-> 4 coeffs), temporally smooth, and
     every dimension is in meters (units-consistent for covariance coding).
  2. FACTORIZATION — separate static from moving. Track lane lines ACROSS frames by
     ground lateral (a line's knots drift slowly; the shared drift IS the ego screw
     xi, which the archive already stores for pose — dual-use, amortized ~0). Each
     track is a small (n_obs, 8) coefficient matrix.
  3. SPD-CONE ANISOTROPIC CODING — the per-track coefficient covariance is extremely
     anisotropic (temporal smoothness + inter-knot correlation), so the SPD-cone /
     Hilbert reverse-water-filling codec that won -27% on the pose section
     (``tac.torch_vehicle.pose_spd_codec``, commit 348ac229f) applies DIRECTLY:
     normalize dims, water-fill the covariance spectrum, delta-zigzag + brotli.
     The Hilbert distance d_H = log(lmax/lmin) measures the exploitable anisotropy.

Dashes + visibility ride the same factorization: dash paint is STATIC in the WORLD
frame (dash phase = ego distance, #215), so each track's forward-occupancy stream is
coded LOSSLESSLY as world-aligned XOR deltas (shift by the shared per-frame travel
S(t), XOR against the previous observation, brotli) — a wrong shift only costs
bytes, never fidelity. S(t) is the integrated ego screw xi: dual-use with the
stored pose section (amortized ~0), also priced standalone honestly.

AUTHORITY: everything here is measured against the CACHED SegNet argmax label maps
(numpy load — NO scorer forward, NO model inference). ``[macOS-MLX advisory]``
geometric feasibility, NOT byte-closed through R + the frozen scorer, NOT a score.

Borrowed-substrate accounting:
  * BORROWED (cited): openpilot flat-ground IPM constants (via lane_sdf_component);
    numpy polyfit/SVD; brotli; the classic reverse-water-filling allocation.
  * OURS-ORIGINAL: the knot re-parametrization of the ground lane manifold, the
    lateral tracker + visibility runs, the track-matrix SPD-cone application, and
    the linear-phase dash factorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.boundary_math.lane_sdf_component import (
    _V_HORIZON,
    LaneLine,
    cluster_lane_lines,
    fit_lane_line,
)

# Fixed forward knots (m) for the lateral-knot parametrization. 4 knots <-> deg-3
# poly (bijective). Chosen inside the reliable IPM mid-field (cluster_lane_lines
# uses forward 6..45 m); generic constants of the algorithm (rule 118: not counted).
FORWARD_KNOTS_M = np.array([8.0, 14.0, 24.0, 40.0], dtype=np.float64)
# Half-width sample rows (image rows; hw is a deg-1 poly of v -> 2 samples bijective).
HALFWIDTH_KNOT_ROWS = np.array([240.0, 340.0], dtype=np.float64)
# Coeff vector layout: 4 lateral knots + 2 halfwidth knots + fwd range (lo, hi).
COEFF_DIM = 8
# Reference forward distance (m) at which tracks are associated by lateral offset.
F_REF_M = 15.0
# Forward occupancy grid (m) for the world-static dash factorization: dashes are
# STATIC ground paint, so occupancy(world) is one interval set per track + a single
# shared per-frame forward-travel scalar S(t) (= the ego screw xi, dual-use with the
# stored pose section -> amortized ~0 in a real archive). Generic constants (rule 118).
OCC_FWD_LO_M = 2.5
OCC_FWD_HI_M = 55.0
OCC_BIN_M = 0.5
OCC_N_BINS = int(round((OCC_FWD_HI_M - OCC_FWD_LO_M) / OCC_BIN_M))
# Max plausible ego travel per pair-frame (0.1 s at ~40 m/s) in bins, per frame gap.
_MAX_SHIFT_BINS_PER_FRAME = 8
# Alignment uses only the near field where one image row resolves <= ~2 bins
# (dfwd/drow = fwd^2 / (cam_h * fy) <= 1 m at fwd ~ 22 m).
_ALIGN_MAX_BIN = int(round((22.0 - OCC_FWD_LO_M) / OCC_BIN_M))


@dataclass
class GroundLaneObs:
    """One lane line observed in one frame, in the ground-frame knot chart."""

    frame: int
    vec: np.ndarray                # (COEFF_DIM,) float64 knot vector
    lat_ref: float                 # lateral (m) at F_REF_M — the tracker key
    n_pixels: int
    dash_period_m: float = 0.0
    dash_phase_m: float = 0.0
    dash_duty: float = 0.5
    occ: np.ndarray | None = None  # (OCC_N_BINS,) bool forward occupancy


def obs_from_lane_line(frame: int, ln: LaneLine) -> GroundLaneObs:
    """Re-parametrize a fitted :class:`LaneLine` into the knot chart (exact for
    deg<=3 centerlines and deg<=1 half-widths — same polynomial family)."""
    # Degenerate-extrapolation clamps (a cubic fitted on a short forward span can
    # blow up at the far knot): lateral beyond +-25 m / half-width beyond 20 px is
    # geometric nonsense; clamping keeps the coding ranges physical. The effect is
    # MEASURED through the raster metrics, never assumed.
    lat_knots = np.clip(np.polyval(ln.centerline_coeffs, FORWARD_KNOTS_M), -25.0, 25.0)
    hw_knots = np.clip(np.polyval(ln.halfwidth_coeffs, HALFWIDTH_KNOT_ROWS), 0.5, 20.0)
    vec = np.concatenate(
        [lat_knots, hw_knots, np.asarray(ln.forward_range, np.float64)]
    ).astype(np.float64)
    return GroundLaneObs(
        frame=int(frame),
        vec=vec,
        lat_ref=float(np.polyval(ln.centerline_coeffs, F_REF_M)),
        n_pixels=int(ln.n_pixels),
        dash_period_m=float(ln.dash_period_m),
        dash_phase_m=float(ln.dash_phase_m),
        dash_duty=float(ln.dash_duty),
    )


def lane_line_from_vec(
    vec: np.ndarray, *, dash: tuple[float, float, float] = (0.0, 0.0, 0.5)
) -> LaneLine:
    """Inverse of :func:`obs_from_lane_line`: knot vector -> :class:`LaneLine`."""
    v = np.asarray(vec, np.float64)
    c_coeffs = np.polyfit(FORWARD_KNOTS_M, v[:4], 3)
    h_coeffs = np.polyfit(HALFWIDTH_KNOT_ROWS, v[4:6], 1)
    per, ph, duty = dash
    return LaneLine(
        centerline_coeffs=c_coeffs,
        halfwidth_coeffs=h_coeffs,
        dash_period_m=float(per),
        dash_phase_m=float(ph),
        dash_duty=float(duty),
        forward_range=(float(v[6]), float(v[7])),
        n_pixels=0,
    )


def fit_frame_ground_lanes(
    lstar: np.ndarray,
    frame: int,
    *,
    lane_cls: int = 1,
    v_h: float = _V_HORIZON,
    fit_dash: bool = True,
) -> list[GroundLaneObs]:
    """Cluster + fit the frame's lane pixels in the ground frame (reuses the
    lane_sdf_component pipeline) and return knot-chart observations."""
    from tac.boundary_math.lane_sdf_component import image_to_ground

    clusters = cluster_lane_lines(np.asarray(lstar), lane_cls=lane_cls, v_h=v_h)
    out: list[GroundLaneObs] = []
    for c in clusters:
        ln = fit_lane_line(c, centerline_deg=3, fit_dash=fit_dash, v_h=v_h)
        if ln is None:
            continue
        obs = obs_from_lane_line(frame, ln)
        if not np.all(np.isfinite(obs.vec)):
            continue
        # Forward occupancy of the cluster's REAL pixels (the dash pattern).
        # Built from per-ROW forward INTERVALS (a row v covers fwd(v+0.5)..fwd(v-0.5))
        # so the far field — where one image row spans several meters — is filled,
        # not aliased into scattered single-bin hits.
        rows = np.unique(c[:, 0]).astype(np.float64)
        f_hi, _ = image_to_ground(np.zeros_like(rows), rows - 0.5, v_h=v_h)
        f_lo, _ = image_to_ground(np.zeros_like(rows), rows + 0.5, v_h=v_h)
        occ = np.zeros(OCC_N_BINS, bool)
        for lo_m, hi_m in zip(f_lo, f_hi):
            if not (np.isfinite(lo_m) and np.isfinite(hi_m)):
                continue
            b0 = int(np.floor((lo_m - OCC_FWD_LO_M) / OCC_BIN_M))
            b1 = int(np.floor((hi_m - OCC_FWD_LO_M) / OCC_BIN_M))
            b0 = max(b0, 0)
            b1 = min(b1, OCC_N_BINS - 1)
            if b1 >= b0:
                occ[b0 : b1 + 1] = True
        obs.occ = occ
        out.append(obs)
    return out


# ---------------------------------------------------------------------------
# Tracking: associate lane lines across frames by ground lateral at F_REF_M.
# ---------------------------------------------------------------------------
@dataclass
class LaneTrack:
    """A lane line tracked across frames: the factorization unit."""

    frames: np.ndarray             # (n_obs,) int64 frame indices (sorted)
    coeffs: np.ndarray             # (n_obs, COEFF_DIM) float64 knot vectors
    dash_obs: np.ndarray           # (n_obs, 3) per-obs (period_m, phase_m, duty)
    lat_ref: np.ndarray            # (n_obs,) float64
    occ: np.ndarray | None = field(default=None)   # (n_obs, OCC_N_BINS) bool
    # Filled by the codecs:
    decoded: np.ndarray | None = field(default=None)       # coeff round-trip
    occ_decoded: np.ndarray | None = field(default=None)   # occupancy round-trip

    @property
    def n_obs(self) -> int:
        return int(self.frames.size)


def track_ground_lanes(
    per_frame: list[list[GroundLaneObs]],
    *,
    max_jump_m: float = 0.6,
    max_gap: int = 8,
    min_obs: int = 2,
) -> list[LaneTrack]:
    """Greedy nearest-lateral association of per-frame observations into tracks.

    An active track carries its last ``lat_ref``; each frame's observations are
    matched to the nearest active track within ``max_jump_m`` (one obs per track
    per frame, nearest-first). Unmatched observations open new tracks; a track
    idle for more than ``max_gap`` frames retires. Tracks shorter than ``min_obs``
    are dropped (their pixels become honest recall loss)."""
    active: list[dict] = []
    done: list[dict] = []
    for fi, obs_list in enumerate(per_frame):
        # retire stale tracks
        still = []
        for tr in active:
            if fi - tr["last_frame"] > max_gap:
                done.append(tr)
            else:
                still.append(tr)
        active = still
        # nearest-first greedy matching
        pairs: list[tuple[float, int, int]] = []
        for oi, ob in enumerate(obs_list):
            for ti, tr in enumerate(active):
                d = abs(ob.lat_ref - tr["lat"])
                if d <= max_jump_m:
                    pairs.append((d, oi, ti))
        pairs.sort()
        used_o: set[int] = set()
        used_t: set[int] = set()
        for d, oi, ti in pairs:
            if oi in used_o or ti in used_t:
                continue
            used_o.add(oi)
            used_t.add(ti)
            tr = active[ti]
            ob = obs_list[oi]
            tr["obs"].append(ob)
            tr["last_frame"] = fi
            tr["lat"] = ob.lat_ref
        for oi, ob in enumerate(obs_list):
            if oi not in used_o:
                active.append({"obs": [ob], "last_frame": fi, "lat": ob.lat_ref})
    done.extend(active)

    tracks: list[LaneTrack] = []
    for tr in done:
        if len(tr["obs"]) < min_obs:
            continue
        obs = sorted(tr["obs"], key=lambda o: o.frame)
        occ = None
        if all(o.occ is not None for o in obs):
            occ = np.stack([o.occ for o in obs])
        tracks.append(
            LaneTrack(
                frames=np.array([o.frame for o in obs], np.int64),
                coeffs=np.stack([o.vec for o in obs]),
                dash_obs=np.array(
                    [[o.dash_period_m, o.dash_phase_m, o.dash_duty] for o in obs],
                    np.float64,
                ),
                lat_ref=np.array([o.lat_ref for o in obs], np.float64),
                occ=occ,
            )
        )
    return tracks


# ---------------------------------------------------------------------------
# World-static dash factorization: intervals(world) x shared ego travel S(t).
# ---------------------------------------------------------------------------
def _best_shift_bins(prev: np.ndarray, cur: np.ndarray, max_shift: int) -> tuple[int, float]:
    """Best forward shift (bins) aligning ``cur`` to ``prev`` (dash pattern moves
    TOWARD the ego: cur[k] ~ prev[k + shift]). Returns (shift, score)."""
    best_s, best_score = 0, -1.0
    for s in range(0, max_shift + 1):
        a = prev[s:]
        b = cur[: a.size] if s else cur
        n = min(a.size, b.size)
        if n < 8:
            continue
        a, b = a[:n], b[:n]
        denom = float(min(a.sum(), b.sum()))
        if denom < 3:
            continue
        score = float((a & b).sum()) / denom
        if score > best_score:
            best_score, best_s = score, s
    return best_s, best_score


def estimate_global_shifts(tracks: list[LaneTrack], n_frames: int) -> np.ndarray:
    """Estimate the SHARED per-frame ego forward travel S(t) (meters, S(0)=0) by
    aligning each dashed track's forward-occupancy pattern between consecutive
    observations, then pooling across tracks (weighted by match quality).

    S(t) IS the ego screw xi integrated forward — in a real archive it rides the
    stored pose section for free (dual-use); here it is estimated from the lane
    pixels alone (cached data only) and its byte cost is ALSO reported standalone."""
    num = np.zeros(n_frames)
    den = np.zeros(n_frames)
    for tr in tracks:
        if tr.occ is None:
            continue
        occ = tr.occ
        # dashed tracks only: a solid (all-on) pattern carries no alignment signal
        frac_on = occ.mean(axis=1)
        for i in range(1, tr.n_obs):
            gap = int(tr.frames[i] - tr.frames[i - 1])
            if gap < 1 or gap > 4:
                continue
            if frac_on[i - 1] > 0.9 or frac_on[i] > 0.9:
                continue
            s, score = _best_shift_bins(
                occ[i - 1][:_ALIGN_MAX_BIN],
                occ[i][:_ALIGN_MAX_BIN],
                min(_MAX_SHIFT_BINS_PER_FRAME * gap, 40),
            )
            if score < 0.4:
                continue
            ds = s * OCC_BIN_M / gap
            for f in range(int(tr.frames[i - 1]) + 1, int(tr.frames[i]) + 1):
                if 0 <= f < n_frames:
                    num[f] += score * ds
                    den[f] += score
    dS = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    # fill unobserved transitions with the median observed travel (steady speed prior)
    if np.isfinite(dS).any():
        med = float(np.nanmedian(dS[np.isfinite(dS)]))
    else:
        med = 0.0
    dS = np.where(np.isfinite(dS), dS, med)
    dS[0] = 0.0
    return np.cumsum(dS)


def shifts_to_bins(shifts: np.ndarray) -> np.ndarray:
    """Quantize the shared travel S(t) to integer occupancy bins (the resolution
    the occupancy codec aligns at)."""
    return np.round(np.asarray(shifts, np.float64) / OCC_BIN_M).astype(np.int64)


def encode_shift_stream(shift_bins: np.ndarray) -> int:
    """MEASURED byte cost of the shared S(t) stream coded standalone: per-frame
    delta bins (small non-negative ints) -> brotli. In a real archive S(t) is
    dual-use with the stored pose xi section (amortized ~0); this is the honest
    standalone price."""
    import brotli

    d = np.diff(np.asarray(shift_bins, np.int64))
    d = np.clip(d, 0, 255).astype(np.uint8)
    return len(brotli.compress(d.tobytes(), quality=11))


def encode_tracks_occupancy(
    tracks: list[LaneTrack], shift_bins: np.ndarray, *, code_bins: int = OCC_N_BINS
) -> dict:
    """World-aligned occupancy codec (the dash/visibility factor); LOSSLESS over
    the coded near field, solid beyond.

    Dashes and visibility are ~static ground paint: aligning each observation's
    forward-occupancy to the WORLD frame (shift by S(t) bins) makes consecutive
    observations nearly identical, so their XOR is sparse and brotli crushes it.
    A wrong shift only costs BYTES, never fidelity (the XOR stays exact).

    ``code_bins`` codes only the first (nearest) bins — the far field, where one
    image row smears several meters and the pattern churns, is decoded as SOLID
    (always on). The fidelity effect of that truncation is MEASURED through the
    decoded gate, never assumed.

    Stream per track: first obs raw, then per-obs XOR against the world-aligned
    predecessor; all tracks concatenated, packbits, one brotli(q11) blob.
    Fills ``track.occ_decoded`` from the REAL decoded stream (Catalog #304: the
    round-trip IS the fidelity proof) and returns the measured bytes."""
    import brotli

    nb = int(min(max(code_bins, 1), OCC_N_BINS))
    bit_rows: list[np.ndarray] = []
    for tr in tracks:
        if tr.occ is None:
            continue
        prev = None
        prev_k = 0
        for i in range(tr.n_obs):
            k = int(shift_bins[tr.frames[i]])
            cur = tr.occ[i][:nb]
            if prev is None:
                bit_rows.append(cur)
            else:
                dk = k - prev_k
                pred = _world_aligned_predictor(prev, dk)
                bit_rows.append(cur ^ pred)
            prev, prev_k = cur, k
    if not bit_rows:
        return {"occ_bytes": 0}
    bits = np.concatenate(bit_rows)
    blob = brotli.compress(np.packbits(bits).tobytes(), quality=11)

    # REAL decode: reverse the stream and fill occ_decoded (far field solid).
    unpacked = np.unpackbits(
        np.frombuffer(brotli.decompress(blob), dtype=np.uint8)
    )[: bits.size].astype(bool)
    pos = 0
    for tr in tracks:
        if tr.occ is None:
            continue
        dec = np.ones_like(tr.occ)
        prev = None
        prev_k = 0
        for i in range(tr.n_obs):
            k = int(shift_bins[tr.frames[i]])
            row = unpacked[pos : pos + nb]
            pos += nb
            if prev is None:
                dec[i, :nb] = row
            else:
                dk = k - prev_k
                dec[i, :nb] = row ^ _world_aligned_predictor(prev, dk)
            prev, prev_k = dec[i, :nb], k
        tr.occ_decoded = dec
    return {
        "occ_bytes": len(blob),
        "occ_bits_raw": int(bits.size),
        "code_bins": nb,
    }


def _world_aligned_predictor(prev_occ: np.ndarray, dk: int) -> np.ndarray:
    """Previous observation re-indexed to the current forward frame: the world
    bin at current fwd-bin ``b`` was at fwd-bin ``b + dk`` one observation ago
    (the ego moved forward by ``dk`` bins). Out-of-window bins predict 0."""
    pred = np.zeros_like(prev_occ)
    if dk >= 0:
        if dk < prev_occ.size:
            pred[: prev_occ.size - dk] = prev_occ[dk:]
    else:
        pred[-dk:] = prev_occ[: prev_occ.size + dk]
    return pred


def smooth_track_coeffs(coeffs: np.ndarray, *, window: int = 9) -> np.ndarray:
    """Centered moving-average of a track's knot trajectories — the SKELETON arm:
    drop the per-frame SegNet argmax jitter (which the v8 SPEC assigns to the
    INR/annulus budget) and keep only the smooth ego-geometry motion. Smooth
    trajectories delta-code to almost nothing."""
    c = np.asarray(coeffs, np.float64)
    n = c.shape[0]
    w = int(max(1, min(window, n)))
    if w <= 1:
        return c.copy()
    kernel = np.ones(w) / w
    out = np.empty_like(c)
    for d in range(c.shape[1]):
        pad = np.concatenate([np.full(w // 2, c[0, d]), c[:, d], np.full(w - 1 - w // 2, c[-1, d])])
        out[:, d] = np.convolve(pad, kernel, mode="valid")
    return out


def build_static_world_occupancy(
    tracks: list[LaneTrack], shift_bins: np.ndarray, *, vote: float = 0.4
) -> dict:
    """The fully-STATIC occupancy arm, factored as PAINT x VISIBILITY.

    Occupancy(frame, fwd) = static_paint(world) AND visibility(fwd): the lane
    PAINT is static ground truth in the world frame, but the VISIBLE EXTENT is
    ego-relative (SegNet resolves lane only inside a forward window). The vote
    is taken only over observations whose visible span covers the world bin, so
    the two factors do not corrupt each other. Stored: one static world bitmap
    per track (brotli) + a per-track forward window (4 B). Per-frame flicker is
    NOT reproduced — the fidelity cost is MEASURED through the raster."""
    import brotli

    payload = bytearray()
    n_tracks = 0
    for tr in tracks:
        if tr.occ is None:
            continue
        ks = shift_bins[tr.frames].astype(np.int64)
        k0, k1 = int(ks.min()), int(ks.max())
        n_bins = (k1 - k0) + OCC_N_BINS
        on = np.zeros(n_bins)
        seen = np.zeros(n_bins)
        # per-obs visible span (first..last on bin) -> the ego-relative window
        firsts, lasts = [], []
        for i in range(tr.n_obs):
            nz = np.flatnonzero(tr.occ[i])
            if nz.size == 0:
                continue
            firsts.append(int(nz[0]))
            lasts.append(int(nz[-1]))
        if not firsts:
            tr.occ_decoded = np.zeros_like(tr.occ)
            continue
        vis_lo = int(np.percentile(firsts, 20))
        vis_hi = int(np.percentile(lasts, 80))
        for i in range(tr.n_obs):
            off = int(ks[i]) - k0
            sl = slice(off + vis_lo, off + vis_hi + 1)
            on[sl] += tr.occ[i][vis_lo : vis_hi + 1]
            seen[sl] += 1.0
        static = np.zeros(n_bins, bool)
        cov = seen > 0
        static[cov] = (on[cov] / np.maximum(seen[cov], 1e-9)) >= vote
        dec = np.zeros_like(tr.occ)
        for i in range(tr.n_obs):
            off = int(ks[i]) - k0
            row = static[off : off + OCC_N_BINS].copy()
            row[:vis_lo] = False
            row[vis_hi + 1 :] = False
            dec[i] = row
        tr.occ_decoded = dec
        payload += np.packbits(static).tobytes()
        n_tracks += 1
    blob = brotli.compress(bytes(payload), quality=11)
    return {
        "occ_bytes": len(blob) + 4 * n_tracks,  # +u16 world offset +2x u8 window
        "occ_bits_raw": 8 * len(payload),
        "static": True,
    }


# ---------------------------------------------------------------------------
# Reconstruction + metrics (shared raster for the solid and occupancy-gated
# variants; NO +1 column padding — measured to cost ~0.15 precision).
# ---------------------------------------------------------------------------
def rasterize_tracks(
    tracks: list[LaneTrack],
    frame: int,
    *,
    occ_gate: bool = False,
    h: int = 384,
    w: int = 512,
    v_h: float = _V_HORIZON,
) -> np.ndarray:
    """Rasterize the frame's DECODED tracks. ``occ_gate=True`` gates each row by
    the track's decoded forward-occupancy (dashes + visibility); rows outside
    the occupancy grid default ON (protects near/far-field recall)."""
    from tac.boundary_math.lane_sdf_component import ground_to_image_row

    band = np.zeros((int(h), int(w)), bool)
    rows = np.arange(int(h), dtype=np.float64)
    vr = rows[rows > (v_h + 1.0)]
    for tr in tracks:
        if tr.decoded is None:
            raise ValueError("tracks must be encoded (call encode_tracks_spd) first")
        idx = np.searchsorted(tr.frames, frame)
        if idx >= tr.frames.size or tr.frames[idx] != frame:
            continue
        ln = lane_line_from_vec(tr.decoded[idx])
        fwd, u_c = ground_to_image_row(vr, ln.lateral_of_forward, v_h=v_h)
        hw = ln.halfwidth_of_v(vr)
        on = np.ones(vr.size, bool)
        if occ_gate:
            occ = tr.occ_decoded if tr.occ_decoded is not None else tr.occ
            if occ is not None:
                o = occ[idx]
                bins = np.floor((fwd - OCC_FWD_LO_M) / OCC_BIN_M).astype(np.int64)
                inb = (bins >= 0) & (bins < OCC_N_BINS)
                on[inb] = o[bins[inb]]
                # beyond-grid continuation: the nearest coded bin decides
                on[bins < 0] = bool(o[0])
                on[bins >= OCC_N_BINS] = bool(o[-1])
            in_range = np.ones(vr.size, bool)  # occupancy OWNS visibility
        else:
            f0, f1 = ln.forward_range
            in_range = (fwd >= f0 - 1.0) & (fwd <= f1 + 5.0)
        for j, vv in enumerate(vr):
            if not (on[j] and in_range[j]):
                continue
            lo = int(max(0, np.floor(u_c[j] - hw[j])))
            hi = int(min(w, np.ceil(u_c[j] + hw[j])))
            if hi > lo:
                band[int(vv), lo:hi] = True
    return band


def evaluate_tracks_raster(
    tracks: list[LaneTrack],
    labels: np.ndarray,
    frame_indices: np.ndarray,
    *,
    lane_cls: int = 1,
    occ_gate: bool = False,
    v_h: float = _V_HORIZON,
) -> dict:
    """Average band metrics of the decoded reconstruction over ``frame_indices``
    (positions into the fitted stack)."""
    recalls, precisions, ious = [], [], []
    h, w = labels.shape[1], labels.shape[2]
    for pos in frame_indices:
        band = rasterize_tracks(
            tracks, int(pos), occ_gate=occ_gate, h=h, w=w, v_h=v_h
        )
        m = lane_band_metrics(band, labels[pos] == lane_cls)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        ious.append(m["iou"])
    return {
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "iou": float(np.mean(ious)),
        "n_frames": int(len(frame_indices)),
    }


# ---------------------------------------------------------------------------
# Codec: per-track SPD-cone water-filled coding of the coefficient matrix.
# ---------------------------------------------------------------------------
def unit_dim_scales() -> np.ndarray:
    """Physical per-dim coding scales — GENERIC constants (rule 118, not
    video-derived): lateral knots in meters, half-width knots in pixels,
    forward range in meters. With these, the SPD ``water_level`` theta is a
    physical squared-error target (theta = 1e-2 -> ~0.1 m/px RMS per mode)."""
    return np.ones(COEFF_DIM, dtype=np.float64)


def robust_dim_scales(tracks: list[LaneTrack], *, floor: float = 1e-3) -> np.ndarray:
    """Per-dimension robust scale (std of pooled obs) used to normalize the knot
    vector before covariance coding (units-free water-filling)."""
    pooled = np.concatenate([t.coeffs for t in tracks], axis=0)
    s = pooled.std(axis=0)
    return np.maximum(s, floor)


def pooled_hilbert_distance(tracks: list[LaneTrack], scales: np.ndarray) -> float:
    """Hilbert projective distance d_H = log(lmax/lmin) of the pooled normalized
    coefficient covariance — the measured anisotropy the codec exploits."""
    from tac.torch_vehicle.pose_spd_codec import hilbert_projective_distance

    pooled = np.concatenate([t.coeffs for t in tracks], axis=0) / scales[None, :]
    pooled = pooled - pooled.mean(axis=0, keepdims=True)
    if pooled.shape[0] <= pooled.shape[1]:
        return 0.0
    # Some numpy/BLAS builds emit spurious divide/overflow/invalid warnings on the
    # float64 matmul SIMD path even when the result is finite and exact (the same
    # documented false positive as pose_spd_codec.decode_pose_section_spd).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cov = pooled.T @ pooled / (pooled.shape[0] - 1)
    eigs = np.linalg.eigvalsh(cov)
    return hilbert_projective_distance(eigs)


def _track_meta_bytes(track: LaneTrack) -> int:
    """Honest per-track metadata: first frame u16 + span u16 + flag byte, plus a
    visibility bitmap over the span when the track has gaps."""
    span = int(track.frames.max() - track.frames.min() + 1)
    meta = 2 + 2 + 1
    if span != track.n_obs:
        meta += (span + 7) // 8
    return meta


def encode_tracks_spd(
    tracks: list[LaneTrack],
    *,
    water_level: float,
    scales: np.ndarray | None = None,
    min_spd_obs: int = 8,
    n_dims: int = COEFF_DIM,
) -> dict:
    """Encode every track's coefficient matrix; fill ``track.decoded`` (denormalized);
    return the measured byte breakdown.

    Tracks with >= ``min_spd_obs`` observations go through the SPD-cone codec
    (:func:`tac.torch_vehicle.pose_spd_codec.encode_pose_section_spd` on the
    dim-normalized matrix, REAL encode + round-trip decode — Catalog #304: the
    encode IS the bit-spend). Short tracks are stored raw float16.

    ``n_dims`` codes only the first dims (e.g. 6 = shape only, no forward-range —
    valid when the occupancy gate owns visibility); un-coded dims decode as the
    per-track mean, priced at 2 float16 per track."""
    import torch

    from tac.torch_vehicle.pose_spd_codec import (
        decode_pose_section_spd,
        encode_pose_section_spd,
    )

    nd = int(min(max(n_dims, 1), COEFF_DIM))
    if scales is None:
        scales = robust_dim_scales(tracks)
    coeff_bytes = 0
    meta_bytes = 0
    n_spd = 0
    n_raw = 0
    for tr in tracks:
        meta_bytes += _track_meta_bytes(tr)
        if nd < COEFF_DIM:
            meta_bytes += (COEFF_DIM - nd) * 2  # per-track float16 means
        if tr.n_obs >= min_spd_obs:
            norm = (tr.coeffs[:, :nd] / scales[None, :nd]).astype(np.float32)
            section = encode_pose_section_spd(
                torch.from_numpy(norm), water_level=water_level
            )
            rec = decode_pose_section_spd(section).numpy().astype(np.float64)
            dec = np.empty_like(tr.coeffs)
            dec[:, :nd] = rec * scales[None, :nd]
            if nd < COEFF_DIM:
                mean_tail = tr.coeffs[:, nd:].mean(axis=0)
                dec[:, nd:] = mean_tail.astype(np.float16).astype(np.float64)[None, :]
            tr.decoded = dec
            coeff_bytes += len(section)
            n_spd += 1
        else:
            tr.decoded = tr.coeffs.astype(np.float16).astype(np.float64)
            coeff_bytes += tr.n_obs * nd * 2
            n_raw += 1
    header_bytes = scales.size * 4  # global per-dim scales (video-derived: counted)
    return {
        "coeff_bytes": int(coeff_bytes),
        "meta_bytes": int(meta_bytes),
        "header_bytes": int(header_bytes),
        "total_bytes": int(coeff_bytes + meta_bytes + header_bytes),
        "n_tracks": len(tracks),
        "n_spd": n_spd,
        "n_raw": n_raw,
    }


def lane_band_metrics(band: np.ndarray, lane_mask: np.ndarray) -> dict:
    """recall / precision / IoU of a rendered band vs the true lane pixels — the
    SAME definition the v8 feasibility probe used for the 97 B/frame baseline."""
    band = np.asarray(band, bool)
    lane = np.asarray(lane_mask, bool)
    n_lane = int(lane.sum())
    inter = int((band & lane).sum())
    union = int((band | lane).sum())
    return {
        "recall": inter / n_lane if n_lane else 1.0,
        "precision": inter / max(1, int(band.sum())),
        "iou": inter / union if union else 1.0,
        "lane_px": n_lane,
    }


# ---------------------------------------------------------------------------
# S1 comparator: per-frame-independent quantized ground coding (no factorization).
# ---------------------------------------------------------------------------
def per_frame_quantized_ground(
    per_frame: list[list[GroundLaneObs]], *, bits: int = 12
) -> tuple[list[list[GroundLaneObs]], float]:
    """Quantize every observation independently to ``bits`` per dimension inside
    the global per-dim range (round-trip applied in place of the raw vectors).
    Returns (quantized per-frame obs, bytes_per_frame) — the un-factorized
    ground-frame comparator, plus 30 dash bits per dashed line."""
    all_vecs = np.concatenate(
        [np.stack([o.vec for o in obs]) for obs in per_frame if obs], axis=0
    )
    lo = all_vecs.min(axis=0)
    hi = all_vecs.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    levels = float(2**bits - 1)
    total_bits = 0.0
    out: list[list[GroundLaneObs]] = []
    for obs_list in per_frame:
        row: list[GroundLaneObs] = []
        for o in obs_list:
            q = np.round((o.vec - lo) / span * levels)
            deq = q / levels * span + lo
            row.append(
                GroundLaneObs(
                    frame=o.frame,
                    vec=deq,
                    lat_ref=o.lat_ref,
                    n_pixels=o.n_pixels,
                    dash_period_m=o.dash_period_m,
                    dash_phase_m=o.dash_phase_m,
                    dash_duty=o.dash_duty,
                )
            )
            total_bits += COEFF_DIM * bits
            if o.dash_period_m > 0.0:
                total_bits += 30
        out.append(row)
    n_frames = max(1, len(per_frame))
    return out, (total_bits / 8.0 + 8 * COEFF_DIM) / n_frames


__all__ = [
    "COEFF_DIM",
    "FORWARD_KNOTS_M",
    "F_REF_M",
    "OCC_N_BINS",
    "GroundLaneObs",
    "LaneTrack",
    "encode_shift_stream",
    "encode_tracks_occupancy",
    "encode_tracks_spd",
    "estimate_global_shifts",
    "evaluate_tracks_raster",
    "fit_frame_ground_lanes",
    "lane_band_metrics",
    "lane_line_from_vec",
    "obs_from_lane_line",
    "per_frame_quantized_ground",
    "pooled_hilbert_distance",
    "rasterize_tracks",
    "robust_dim_scales",
    "build_static_world_occupancy",
    "shifts_to_bins",
    "smooth_track_coeffs",
    "track_ground_lanes",
    "unit_dim_scales",
]
