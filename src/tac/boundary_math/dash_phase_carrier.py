# SPDX-License-Identifier: MIT
"""DASH-PHASE CARRIER (#425 STORE side) — curve-domain per-dash δ(s) phase codec.

THE NAMED REACTIVATION (`temporal_advection_stratified_20260715.md` verdict, reactivation
criterion (c)): raster transport-conditional coding LOSES (amortization 0.71 < 1) because
the residual is scattered small-support boundary jitter; the CURVE-DOMAIN δ(s) carrier with
an explicit birth/death (event) model is the formulation the measured jitter prior
(d=0 40.4% · ≤1px 72.3% · ≤2px 79.8% · >2px 20.2%, screw transport) prices at ~2.2 bits/site
— PER SITE. This module moves the unit from raster SITES to lane DASHES (world objects,
~20.6/frame, `lane_channel_deep_refactorization_20260716.md` §3): one δ(s) pair per matched
dash per frame instead of ~54 straddle sites, plus EXPLICIT birth/death/rebirth event codes
for the ~9.4/step churn the transport cannot predict (§4 of the same memo).

WHAT IS CODED (the store-what-doesn't-generalize split):

  * per-frame lane DASH observations (8-conn Lane-class islands, area ≥ min_area,
    interior) are tracked across scored frames by the ξ-transport prediction (the SAME
    ground-plane homography A_ξ Force-1 / #359 use — ``homography_from_xi_numpy`` on the
    cross-scored-frame screw ``cross_scored_frame_xi_interp``);
  * MATCHED dash: a curve-relative residual ``(δs, δn)`` — the observed centroid minus the
    ξ-advected prediction, rotated into the dash's own frame (s = along the dash's
    principal axis = the PHASE coordinate; n = across) — quantized to ``q_px`` and coded
    with a PRIOR-DERIVED canonical Huffman code over {0, ±1, ±2, ESC} whose probabilities
    ARE the measured jitter prior (the code is derived from the prior, not hand-picked;
    the 6 code lengths ship IN THE HEADER so no video-derived table hides in decoder code);
  * DEATH: 1 alive-bit per live track per frame (measured churn ≈ 0.5 ⇒ H ≈ 1 bit — the
    prior says a flat bit is near-optimal);
  * BIRTH: full anchor (quantized centroid u16×2 + tilt(4b) + varint area);
  * REBIRTH: a varint index into the ξ-advected DORMANT pool + (δs, δn) — the world-frame
    amortization: a dash that blinks out and back (saddle-node re-crossing of the SAME
    world paint, memo §4 "blink-back fraction — unmeasured") re-anchors for ~2 B instead
    of ~6 B. The encoder's rebirth rate IS the first measurement of that fraction.

PRE-REGISTERED EXPECTATION (stated before measuring, from the prior alone): canonical
Huffman on {p0=0.404, p±1=0.1595 ea, p±2=0.0375 ea, pESC=0.202} costs 2.267 bits/symbol
(entropy 2.193); a matched dash = 2 symbols ⇒ **≈ 4.53 bits/dash expected** (+ ESC varint
payload on the 20.2% tail), vs ~2.2 bits × ~54 sites/frame in the per-site formulation.

RULE-118 boundary (NO-FAKE #6):
  * COUNTED (archive.zip): the section bytes — header (incl. the 6 Huffman code lengths),
    per-frame event stream (alive bits, δ symbols, birth anchors, rebirth indices), and
    (optionally, ``include_xi``) the fp16 ξ twists. ξ is ALREADY shipped for d_pose (L68
    dxi 7.2 KB banked) so composition credits it at 0 marginal; both accountings reported.
  * FREE (inflate.py): the ξ-transport point advection (generic geometry), the canonical
    Huffman decoder (generic given header lengths), the dash rasterizer downstream. The
    encoder-side island extraction/matching never runs at decode.
  * The decoder consumes EVERY seed byte (strict cursor/bit accounting; trailing slack
    beyond byte padding refuses) and the encoder runs the FULL decoder and REFUSES unless
    the reconstructed track states are EXACTLY the closed-loop states (NO-FAKE self-check).

DETERMINISM: numpy-fp64 + stdlib only on the codec path (inflate-portable); greedy
matching with total ordering (distance, track id, observation raster order). Same inputs
⇒ same bytes on any host. NEVER a score authority: this module produces BYTES; recovery
is measured at the partition/label level by ``tools/measure_dash_phase_carrier_n600.py``
(honestly labelled), and the pointer moves only through ``upstream/evaluate.py``.

Relationship to sisters (composition, not duplication):
  * `phase_residual_carrier.py` (#359): the RASTER tie-field residual carrier (per-site).
    This module is the curve/object-domain alternative the advection memo's verdict names.
  * `curve_relative_offset_coder.py` (#386 v8-T2): the WITHIN-FRAME spatial chart
    (residual px → 1-D n(s) along generator curves). Real, continued: this module is its
    TEMPORAL complement (per-dash phase across frames); the within-frame shape residual
    stays #386's job. Neither is superseded.
  * `phase_primitives.py` (#424/#360, Arm B): supplies ``cross_scored_frame_xi_interp``
    (imported); the event definition here (no valid transported reference ⇒ birth) mirrors
    Arm B's event-fallback semantics (straddle with no advected reference) at the island
    level — same event concept, object-level granularity.
"""
from __future__ import annotations

import heapq
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.boundary_math.phase_primitives import cross_scored_frame_xi_interp
from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    homography_from_xi_numpy,
)

__all__ = [
    "DASH_PHASE_MAGIC",
    "DashObs",
    "DashPhaseConfig",
    "DashPhaseError",
    "DashPhaseReport",
    "DecodedDash",
    "JITTER_PRIOR_SYMBOL_PROBS",
    "build_prior_huffman_lengths",
    "decode_dash_phase_carrier",
    "encode_dash_phase_carrier",
    "expected_bits_per_symbol",
    "extract_dash_observations",
    "dash_phase_carrier_report",
]

DASH_PHASE_MAGIC = b"DASH1\x00"

#: The measured screw-transport separatrix jitter prior (temporal_advection_stratified
#: n600, `edge_jitter_prior_screw`): P(d=0)=0.404, P(d≤1)=0.723, P(d≤2)=0.798, P(>2)=0.202.
#: Mapped to a per-component symbol alphabet {0, +1, −1, +2, −2, ESC} by splitting each
#: magnitude band evenly over sign (ASSUMED isotropic-in-sign; stated in the memo).
JITTER_PRIOR_SYMBOL_PROBS: dict[str, float] = {
    "0": 0.404,
    "+1": (0.723 - 0.404) / 2.0,
    "-1": (0.723 - 0.404) / 2.0,
    "+2": (0.798 - 0.723) / 2.0,
    "-2": (0.798 - 0.723) / 2.0,
    "ESC": 1.0 - 0.798 + 0.0,  # the >2px escape band
}
_SYMBOL_ORDER: tuple[str, ...] = ("0", "+1", "-1", "+2", "-2", "ESC")
_SYMBOL_VALUE: dict[str, int] = {"0": 0, "+1": 1, "-1": -1, "+2": 2, "-2": -2}


class DashPhaseError(ValueError):
    """Raised on a corrupt payload, an unconsumed seed byte, or a NO-FAKE self-check failure."""


# --------------------------------------------------------------------------- #
# 0. prior → code (derived, not hand-picked)                                    #
# --------------------------------------------------------------------------- #
def build_prior_huffman_lengths(probs: dict[str, float] | None = None) -> dict[str, int]:
    """Canonical-Huffman code LENGTHS derived from the measured jitter prior.

    Deterministic Huffman (heap with total tie order: probability, then symbol order).
    The lengths — not the prior — ship in the section header, so the decoder is generic
    given the header (no video-derived table in decoder code; rule-118 clean).
    """
    p = probs or JITTER_PRIOR_SYMBOL_PROBS
    if set(p) != set(_SYMBOL_ORDER):
        raise DashPhaseError(f"prior symbols must be {_SYMBOL_ORDER}, got {sorted(p)}")
    # heap items: (prob, tiebreak, leaves)
    heap: list[tuple[float, int, tuple[str, ...]]] = []
    for i, s in enumerate(_SYMBOL_ORDER):
        heapq.heappush(heap, (float(p[s]), i, (s,)))
    depth: dict[str, int] = {s: 0 for s in _SYMBOL_ORDER}
    tie = len(_SYMBOL_ORDER)
    while len(heap) > 1:
        pa, _, la = heapq.heappop(heap)
        pb, _, lb = heapq.heappop(heap)
        for s in la + lb:
            depth[s] += 1
        heapq.heappush(heap, (pa + pb, tie, la + lb))
        tie += 1
    return depth


def expected_bits_per_symbol(
    lengths: dict[str, int] | None = None, probs: dict[str, float] | None = None
) -> float:
    """E[len] under the prior — the pre-registered per-component cost (≈ 2.267 bits)."""
    p = probs or JITTER_PRIOR_SYMBOL_PROBS
    ln = lengths or build_prior_huffman_lengths(p)
    return float(sum(p[s] * ln[s] for s in _SYMBOL_ORDER))


def _canonical_codes(lengths: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Canonical Huffman codes (code, nbits) from lengths (deterministic symbol order)."""
    items = sorted(_SYMBOL_ORDER, key=lambda s: (lengths[s], _SYMBOL_ORDER.index(s)))
    codes: dict[str, tuple[int, int]] = {}
    code = 0
    prev_len = 0
    for s in items:
        ln = int(lengths[s])
        code <<= ln - prev_len
        codes[s] = (code, ln)
        code += 1
        prev_len = ln
    return codes


class _BitWriter:
    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, nbits: int) -> None:
        for i in range(nbits - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def write_bit(self, b: int) -> None:
        self._bits.append(int(b) & 1)

    def write_varint(self, v: int) -> None:
        if v < 0:
            raise DashPhaseError("varint must be non-negative")
        while True:
            chunk = v & 0x7F
            v >>= 7
            self.write_bit(1 if v else 0)  # continuation-first (bit-level varint)
            self.write(chunk, 7)
            if not v:
                break

    @property
    def bit_count(self) -> int:
        return len(self._bits)

    def getvalue(self) -> bytes:
        out = bytearray()
        acc = 0
        n = 0
        for b in self._bits:
            acc = (acc << 1) | b
            n += 1
            if n == 8:
                out.append(acc)
                acc = 0
                n = 0
        if n:
            out.append(acc << (8 - n))
        return bytes(out)


class _BitReader:
    def __init__(self, blob: bytes) -> None:
        self._blob = blob
        self._pos = 0  # bit cursor

    def read_bit(self) -> int:
        byte = self._pos >> 3
        if byte >= len(self._blob):
            raise DashPhaseError("bitstream exhausted (corrupt payload)")
        b = (self._blob[byte] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return b

    def read(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.read_bit()
        return v

    def read_varint(self) -> int:
        v = 0
        shift = 0
        while True:
            cont = self.read_bit()
            chunk = self.read(7)
            v |= chunk << shift
            shift += 7
            if not cont:
                return v

    @property
    def bits_read(self) -> int:
        return self._pos

    def assert_fully_consumed(self) -> None:
        """Every seed byte consumed: only sub-byte zero padding may remain."""
        remaining = len(self._blob) * 8 - self._pos
        if remaining >= 8:
            raise DashPhaseError(
                f"{remaining} unread bits (> byte padding): unconsumed seed bytes — NO-FAKE refusal"
            )
        # the padding bits must be zero (written by _BitWriter.getvalue)
        while self._pos < len(self._blob) * 8:
            if self.read_bit():
                raise DashPhaseError("nonzero padding bits (corrupt payload)")


def _zigzag_int(v: int) -> int:
    return (v << 1) ^ (v >> 63) if v >= 0 else ((-v) << 1) - 1


def _unzigzag_int(u: int) -> int:
    return (u >> 1) if (u & 1) == 0 else -((u + 1) >> 1)


def _write_delta_symbol(bw: _BitWriter, codes: dict[str, tuple[int, int]], v: int) -> str:
    """Write one quantized residual component; returns the symbol used (telemetry)."""
    if -2 <= v <= 2:
        s = {0: "0", 1: "+1", -1: "-1", 2: "+2", -2: "-2"}[v]
        c, n = codes[s]
        bw.write(c, n)
        return s
    c, n = codes["ESC"]
    bw.write(c, n)
    bw.write_varint(_zigzag_int(int(v)))
    return "ESC"


def _read_delta_symbol(br: _BitReader, decode_tree: dict[tuple[int, int], str]) -> int:
    code = 0
    nbits = 0
    while True:
        code = (code << 1) | br.read_bit()
        nbits += 1
        if nbits > 32:
            raise DashPhaseError("Huffman decode overrun (corrupt payload)")
        s = decode_tree.get((code, nbits))
        if s is None:
            continue
        if s == "ESC":
            return _unzigzag_int(br.read_varint())
        return _SYMBOL_VALUE[s]


# --------------------------------------------------------------------------- #
# 1. dash observation extraction (encoder-side only; FREE — never runs at decode)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashObs:
    """One observed lane dash (island) in a scored frame."""

    centroid_rc: tuple[float, float]  # (row, col), fp64 pixel mean
    area: int
    tilt: float                       # principal-axis angle in [0, pi) from the +col axis
    pixel_rows: np.ndarray            # int32 — kept for the recovery measurement only
    pixel_cols: np.ndarray


def _label_islands(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic 8-conn labeling (raster-order flood fill, stdlib only)."""
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    rows_idx, cols_idx = np.nonzero(mask)
    for r0, c0 in zip(rows_idx.tolist(), cols_idx.tolist()):
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        rr: list[int] = []
        cc: list[int] = []
        while stack:
            r, c = stack.pop()
            rr.append(r)
            cc.append(c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    r2, c2 = r + dr, c + dc
                    if 0 <= r2 < H and 0 <= c2 < W and mask[r2, c2] and not seen[r2, c2]:
                        seen[r2, c2] = True
                        stack.append((r2, c2))
        out.append((np.asarray(rr, dtype=np.int32), np.asarray(cc, dtype=np.int32)))
    # deterministic order: first pixel in raster order
    out.sort(key=lambda rc: (int(rc[0][0]) * W + int(rc[1][0])))
    return out


def _principal_tilt(rr: np.ndarray, cc: np.ndarray) -> float:
    """Principal-axis angle in [0, pi) from the +col axis (second central moments)."""
    x = cc.astype(np.float64) - float(cc.mean())
    y = rr.astype(np.float64) - float(rr.mean())
    mu20 = float((x * x).mean())
    mu02 = float((y * y).mean())
    mu11 = float((x * y).mean())
    ang = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    return float(ang % np.pi)


def extract_dash_observations(
    lstar: np.ndarray, cfg: "DashPhaseConfig"
) -> list[DashObs]:
    """Lane-class islands of one scored frame (8-conn, area/interior filtered)."""
    lstar = np.asarray(lstar)
    H, W = lstar.shape
    mask = lstar == cfg.lane_class
    obs: list[DashObs] = []
    b = int(cfg.border_px)
    for rr, cc in _label_islands(mask):
        if rr.size < cfg.min_area:
            continue
        if (rr.min() < b or cc.min() < b or rr.max() >= H - b or cc.max() >= W - b):
            continue  # interior islands only (memo §3/§4 convention)
        obs.append(
            DashObs(
                centroid_rc=(float(rr.mean()), float(cc.mean())),
                area=int(rr.size),
                tilt=_principal_tilt(rr, cc),
                pixel_rows=rr,
                pixel_cols=cc,
            )
        )
    return obs


# --------------------------------------------------------------------------- #
# 2. ξ point advection (the SAME homography as the field warp; forward map = H) #
# --------------------------------------------------------------------------- #
def _advect_points_rc(
    pts_rc: np.ndarray, xi_cross: np.ndarray, geom: GroundHomographyGeom
) -> np.ndarray:
    """Forward-advect (row, col) points by the cross-frame screw.

    The field warp samples ``warped[target] = src[H⁻¹ @ target]`` (warp_frame0_native),
    so a SOURCE point p maps forward to target ``H @ p`` — the exact same homography,
    point form. Off-frame / behind-camera → PERSIST fallback (same accounting)."""
    pts = np.asarray(pts_rc, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return pts
    H = homography_from_xi_numpy(np.asarray(xi_cross, dtype=np.float64), geom)
    hom = np.stack([pts[:, 1], pts[:, 0], np.ones(pts.shape[0])], axis=0)  # (u=col, v=row, 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tgt = H @ hom
        z = tgt[2]
        u = tgt[0] / z
        v = tgt[1] / z
    Hh, Ww = geom.native_hw
    ok = np.isfinite(u) & np.isfinite(v) & (z > 0) & (u >= 0) & (u <= Ww - 1) & (v >= 0) & (v <= Hh - 1)
    out = pts.copy()
    out[ok, 0] = v[ok]
    out[ok, 1] = u[ok]
    return out


def _cross_xi(xi_dec: np.ndarray, p: int, gap_xi: str) -> np.ndarray:
    """Scored-frame p−1 → p screw (same closed-loop convention as phase_residual_carrier)."""
    if gap_xi != "interp":
        raise DashPhaseError(f"gap_xi={gap_xi!r} not implemented (only 'interp')")
    a = xi_dec[p - 1]
    b = xi_dec[p] if p < xi_dec.shape[0] else xi_dec[p - 1]
    return cross_scored_frame_xi_interp(a, b)


# --------------------------------------------------------------------------- #
# 3. config / report / decoded-state types                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DashPhaseConfig:
    """Dash-phase carrier config (decoder-visible; ships in the section header)."""

    lane_class: int = 1
    min_area: int = 3
    border_px: int = 6
    match_radius_px: float = 6.0
    q_px: float = 1.0
    tilt_bins: int = 16
    dormant_max_frames: int = 30
    gap_xi: str = "interp"
    # pose->xi calibration (encode-side only; the STORED fp16 xi is post-calibration, so the
    # decoder never sees these). Defaults = the MEASURED grok/advection-memo fit on THIS video
    # (temporal_advection_stratified_20260715: s_t=-0.00322, s_r=0, pitch=-0.01, fit on the
    # first 100 transitions Road+Lane) — NOT the raw s_t=1 scale (the PoseNet 6-vec is xi only
    # up-to-affine; the s_t=1 scale mis-advects, measured in the n20 smoke: gt2 67% vs 20%).
    s_t: float = -0.00322
    s_r: float = 0.0
    pitch: float = -0.01
    include_xi: bool = True  # False when composing with the already-banked dxi section (L68)


@dataclass(frozen=True)
class DecodedDash:
    """Decoder output: one live dash at one frame (input to the FREE rasterizer)."""

    track_id: int
    centroid_rc: tuple[float, float]
    tilt: float
    area: int
    born: bool  # first frame of (re)appearance


@dataclass
class _Track:
    tid: int
    centroid: np.ndarray  # (2,) fp64 decoded (row, col)
    tilt: float           # dequantized anchor tilt (fixed over track life)
    area: int
    alive: bool
    dormant_for: int = 0


@dataclass(frozen=True)
class DashPhaseReport:
    """MEASURED accounting of one encode (bytes, events, prior-vs-realized code cost)."""

    n_frames: int
    section_bytes: int
    section_bytes_excl_xi: int
    xi_bytes: int
    header_bytes: int
    stream_bytes: int
    n_tracks_total: int
    n_matched: int
    n_births: int
    n_rebirths: int
    n_deaths: int
    blink_back_fraction: float  # rebirths / (births + rebirths) after frame 0
    alive_bits: int
    delta_bits: int
    birth_bits: int
    rebirth_bits: int
    esc_rate: float
    expected_bits_per_dash_prior: float  # pre-registered: 2 × E[len] under the prior
    measured_bits_per_matched_dash: float
    symbol_histogram: dict[str, int]
    mean_abs_delta_px: float
    zlib9_delta_stream_bytes: int  # the same (δs, δn) ints through the shared zlib9 stage
    prior_code_delta_bytes: int    # our prior-derived code, byte-rounded
    reconstruction_bit_identical: bool
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 4. encode (closed loop: the encoder predicts from what the decoder will see)  #
# --------------------------------------------------------------------------- #
def _tilt_quantize(tilt: float, bins: int) -> int:
    q = int(np.round(tilt / (np.pi / bins))) % bins
    return q


def _tilt_dequantize(q: int, bins: int) -> float:
    return float(q) * (np.pi / bins)


def _greedy_match(
    preds: list[tuple[int, np.ndarray]], obs: list[DashObs], radius: float
) -> dict[int, int]:
    """Greedy nearest matching pred-index→obs-index with total deterministic order."""
    cands: list[tuple[float, int, int]] = []
    for pi, (_, pc) in enumerate(preds):
        for oi, o in enumerate(obs):
            d = float(np.hypot(pc[0] - o.centroid_rc[0], pc[1] - o.centroid_rc[1]))
            if d <= radius:
                cands.append((d, pi, oi))
    cands.sort()
    used_p: set[int] = set()
    used_o: set[int] = set()
    out: dict[int, int] = {}
    for d, pi, oi in cands:
        if pi in used_p or oi in used_o:
            continue
        used_p.add(pi)
        used_o.add(oi)
        out[pi] = oi
    return out


def _frame_axes(tilt: float) -> tuple[np.ndarray, np.ndarray]:
    """(e_s, e_n) in (row, col) coords for a dash tilted `tilt` from the +col axis."""
    e_s = np.array([np.sin(tilt), np.cos(tilt)], dtype=np.float64)  # (drow, dcol)
    e_n = np.array([np.cos(tilt), -np.sin(tilt)], dtype=np.float64)
    return e_s, e_n


def encode_dash_phase_carrier(
    lstars: np.ndarray,
    xi_twists: np.ndarray,
    cfg: DashPhaseConfig | None = None,
    *,
    geom: GroundHomographyGeom | None = None,
    telemetry: list[dict[str, Any]] | None = None,
) -> tuple[bytes, DashPhaseReport, list[list[DecodedDash]]]:
    """Encode the dash-phase section from cached argmax + per-pair twists.

    ``lstars`` (P,H,W) int argmax per scored frame (f1 sequence); ``xi_twists`` (P,6)
    rho-first per-pair ego twists (the dual-use pose ξ). Returns
    ``(section, report, decoded_frames)`` where ``decoded_frames`` is the (verified
    bit-identical) decoder output — the phase-correct dash states per frame.

    ``telemetry`` (optional, measurement-only — never coded): appended per-event dicts
    ``{frame, kind: match|rebirth|birth, track_id, obs_index, pred_rc, obs_rc, dec_rc}``
    where ``pred_rc`` is the ξ-transport prediction BEFORE the δ correction — the
    transport-only arm of the recovery A/B."""
    cfg = cfg or DashPhaseConfig()
    lstars = np.asarray(lstars)
    if lstars.ndim != 3:
        raise DashPhaseError(f"lstars must be (P,H,W); got {lstars.shape}")
    P, H, W = lstars.shape
    if H >= 65536 or W >= 65536:
        raise DashPhaseError("frame dims must fit u16 anchors")
    xi_twists = np.asarray(xi_twists, dtype=np.float64).reshape(-1, 6)
    if xi_twists.shape[0] < P:
        raise DashPhaseError(f"need >= {P} twists, got {xi_twists.shape[0]}")
    geom = geom or GroundHomographyGeom.eon(native_hw=(H, W), pitch=cfg.pitch)
    # decoder-visible ξ (fp16 roundtrip — closed loop, same as phase_residual_carrier)
    xi_dec = xi_twists[:P].astype(np.float16).astype(np.float64)

    lengths = build_prior_huffman_lengths()
    codes = _canonical_codes(lengths)
    q = float(cfg.q_px)

    observations = [extract_dash_observations(lstars[p], cfg) for p in range(P)]

    bw = _BitWriter()
    tracks: list[_Track] = []
    next_tid = 0
    n_matched = n_births = n_rebirths = n_deaths = 0
    alive_bits = delta_bits = birth_bits = rebirth_bits = 0
    sym_hist: dict[str, int] = {s: 0 for s in _SYMBOL_ORDER}
    delta_ints: list[int] = []
    abs_delta_px: list[float] = []
    closed_loop_frames: list[list[DecodedDash]] = []

    def _write_birth_anchor(o: DashObs) -> tuple[np.ndarray, float, int]:
        rq = int(np.round(o.centroid_rc[0] / q))
        cq = int(np.round(o.centroid_rc[1] / q))
        tq = _tilt_quantize(o.tilt, cfg.tilt_bins)
        bw.write(rq, 16)
        bw.write(cq, 16)
        bw.write(tq, 4)
        bw.write_varint(int(o.area))
        dec_c = np.array([rq * q, cq * q], dtype=np.float64)
        return dec_c, _tilt_dequantize(tq, cfg.tilt_bins), int(o.area)

    def _write_delta(o: DashObs, pred_c: np.ndarray, tilt: float) -> np.ndarray:
        nonlocal delta_bits
        e_s, e_n = _frame_axes(tilt)
        d = np.array([o.centroid_rc[0] - pred_c[0], o.centroid_rc[1] - pred_c[1]])
        ds = int(np.round(float(d @ e_s) / q))
        dn = int(np.round(float(d @ e_n) / q))
        b0 = bw.bit_count
        s1 = _write_delta_symbol(bw, codes, ds)
        s2 = _write_delta_symbol(bw, codes, dn)
        delta_bits += bw.bit_count - b0
        sym_hist[s1] += 1
        sym_hist[s2] += 1
        delta_ints.extend([ds, dn])
        abs_delta_px.append(float(np.hypot(d[0], d[1])))
        return pred_c + (ds * q) * e_s + (dn * q) * e_n

    for p in range(P):
        obs = observations[p]
        frame_out: list[DecodedDash] = []
        if p > 0:
            xi_c = _cross_xi(xi_dec, p, cfg.gap_xi)
            live = [t for t in tracks if t.alive]
            dormant = [t for t in tracks if not t.alive and t.dormant_for <= cfg.dormant_max_frames]
            # advect ALL track centroids forward (live for prediction, dormant for rebirth)
            for t in live + dormant:
                t.centroid = _advect_points_rc(t.centroid[None], xi_c, geom)[0]
            live_match = _greedy_match(
                [(t.tid, t.centroid) for t in live], obs, cfg.match_radius_px
            )
            matched_obs = set(live_match.values())
            # (a) alive bit per live track, in creation order; survivors write δ
            for li, t in enumerate(live):
                if li in live_match:
                    bw.write_bit(1)
                    alive_bits += 1
                    o = obs[live_match[li]]
                    pred_c = t.centroid.copy()
                    t.centroid = _write_delta(o, t.centroid, t.tilt)
                    n_matched += 1
                    if telemetry is not None:
                        telemetry.append({
                            "frame": p, "kind": "match", "track_id": t.tid,
                            "obs_index": live_match[li],
                            "pred_rc": (float(pred_c[0]), float(pred_c[1])),
                            "obs_rc": o.centroid_rc,
                            "dec_rc": (float(t.centroid[0]), float(t.centroid[1])),
                        })
                    frame_out.append(
                        DecodedDash(t.tid, (float(t.centroid[0]), float(t.centroid[1])), t.tilt, t.area, False)
                    )
                else:
                    bw.write_bit(0)
                    alive_bits += 1
                    t.alive = False
                    t.dormant_for = 0
                    n_deaths += 1
            # (b) new observations: rebirth-from-dormant or full birth
            new_obs = [oi for oi in range(len(obs)) if oi not in matched_obs]
            bw.write_varint(len(new_obs))
            dorm_match = _greedy_match(
                [(t.tid, t.centroid) for t in dormant], [obs[oi] for oi in new_obs], cfg.match_radius_px
            )
            inv = {ni: di for di, ni in dorm_match.items()}
            reborn_d: set[int] = set()
            for ni, oi in enumerate(new_obs):
                o = obs[oi]
                di = inv.get(ni)
                if di is not None and di not in reborn_d:
                    reborn_d.add(di)
                    t = dormant[di]
                    bw.write_bit(1)  # rebirth
                    b0 = bw.bit_count
                    bw.write_varint(di)
                    rebirth_bits += bw.bit_count - b0 + 1  # flag + index (δ counted in delta_bits)
                    pred_c = t.centroid.copy()
                    t.centroid = _write_delta(o, t.centroid, t.tilt)
                    t.alive = True
                    t.dormant_for = 0
                    n_rebirths += 1
                    if telemetry is not None:
                        telemetry.append({
                            "frame": p, "kind": "rebirth", "track_id": t.tid, "obs_index": oi,
                            "pred_rc": (float(pred_c[0]), float(pred_c[1])),
                            "obs_rc": o.centroid_rc,
                            "dec_rc": (float(t.centroid[0]), float(t.centroid[1])),
                        })
                    frame_out.append(
                        DecodedDash(t.tid, (float(t.centroid[0]), float(t.centroid[1])), t.tilt, t.area, True)
                    )
                else:
                    bw.write_bit(0)  # full birth
                    b0 = bw.bit_count
                    dec_c, tilt, area = _write_birth_anchor(o)
                    birth_bits += bw.bit_count - b0 + 1
                    tracks.append(_Track(next_tid, dec_c, tilt, area, True))
                    frame_out.append(DecodedDash(next_tid, (float(dec_c[0]), float(dec_c[1])), tilt, area, True))
                    if telemetry is not None:
                        telemetry.append({
                            "frame": p, "kind": "birth", "track_id": next_tid, "obs_index": oi,
                            "pred_rc": None, "obs_rc": o.centroid_rc,
                            "dec_rc": (float(dec_c[0]), float(dec_c[1])),
                        })
                    next_tid += 1
                    n_births += 1
            # (c) age the dormant pool (unreborn only)
            for di, t in enumerate(dormant):
                if di not in reborn_d:
                    t.dormant_for += 1
        else:
            bw.write_varint(len(obs))
            for oi, o in enumerate(obs):
                b0 = bw.bit_count
                dec_c, tilt, area = _write_birth_anchor(o)
                birth_bits += bw.bit_count - b0
                tracks.append(_Track(next_tid, dec_c, tilt, area, True))
                frame_out.append(DecodedDash(next_tid, (float(dec_c[0]), float(dec_c[1])), tilt, area, True))
                if telemetry is not None:
                    telemetry.append({
                        "frame": 0, "kind": "birth", "track_id": next_tid, "obs_index": oi,
                        "pred_rc": None, "obs_rc": o.centroid_rc,
                        "dec_rc": (float(dec_c[0]), float(dec_c[1])),
                    })
                next_tid += 1
                n_births += 1
        closed_loop_frames.append(frame_out)

    stream = bw.getvalue()
    hdr = {
        "version": 1,
        "n_frames": P,
        "height": int(H),
        "width": int(W),
        "lane_class": int(cfg.lane_class),
        "min_area": int(cfg.min_area),
        "border_px": int(cfg.border_px),
        "match_radius_px": float(cfg.match_radius_px),
        "q_px": float(cfg.q_px),
        "tilt_bins": int(cfg.tilt_bins),
        "dormant_max_frames": int(cfg.dormant_max_frames),
        "gap_xi": str(cfg.gap_xi),
        "pitch": float(cfg.pitch),
        "include_xi": bool(cfg.include_xi),
        "code_lengths": [int(lengths[s]) for s in _SYMBOL_ORDER],  # the COUNTED code table
        "stream_bits": bw.bit_count,
    }
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    buf = bytearray()
    buf += DASH_PHASE_MAGIC
    buf += struct.pack("<I", len(hj))
    buf += hj
    xi_bytes = 0
    if cfg.include_xi:
        xb = np.asarray(xi_twists[:P], dtype=np.float16).tobytes()
        xi_bytes = len(xb)
        buf += xb
    buf += struct.pack("<I", len(stream))
    buf += stream
    section = bytes(buf)

    # -------- NO-FAKE self-check: full decode must equal the closed loop exactly ------
    dec_frames = decode_dash_phase_carrier(
        section, xi_twists_external=(None if cfg.include_xi else xi_dec), geom=geom
    )
    if len(dec_frames) != len(closed_loop_frames):
        raise DashPhaseError("NO-FAKE: decode frame count mismatch")
    for fa, fb in zip(dec_frames, closed_loop_frames):
        if len(fa) != len(fb):
            raise DashPhaseError("NO-FAKE: decode dash count mismatch")
        for da, db in zip(fa, fb):
            if (
                da.track_id != db.track_id
                or da.centroid_rc != db.centroid_rc
                or da.tilt != db.tilt
                or da.area != db.area
                or da.born != db.born
            ):
                raise DashPhaseError("NO-FAKE: decode != closed-loop reconstruction (codec bug)")

    # -------- accounting ---------------------------------------------------------------
    n_delta_events = n_matched + n_rebirths
    total_syms = sum(sym_hist.values())
    from tac.boundary_math.xi_spline_residual_coder import measure_residual_schemes

    d_arr = np.asarray(delta_ints, dtype=np.int64).reshape(-1, 1)
    zlib9_bytes = int(measure_residual_schemes(d_arr)["zlib9"]) if d_arr.size else 0
    report = DashPhaseReport(
        n_frames=P,
        section_bytes=len(section),
        section_bytes_excl_xi=len(section) - xi_bytes,
        xi_bytes=xi_bytes,
        header_bytes=len(DASH_PHASE_MAGIC) + 4 + len(hj),
        stream_bytes=len(stream),
        n_tracks_total=next_tid,
        n_matched=n_matched,
        n_births=n_births,
        n_rebirths=n_rebirths,
        n_deaths=n_deaths,
        blink_back_fraction=(
            n_rebirths / max(1, (n_births - len(observations[0])) + n_rebirths)
        ),
        alive_bits=alive_bits,
        delta_bits=delta_bits,
        birth_bits=birth_bits,
        rebirth_bits=rebirth_bits,
        esc_rate=(sym_hist["ESC"] / total_syms) if total_syms else 0.0,
        expected_bits_per_dash_prior=2.0 * expected_bits_per_symbol(lengths),
        measured_bits_per_matched_dash=(delta_bits / n_delta_events) if n_delta_events else 0.0,
        symbol_histogram=dict(sym_hist),
        mean_abs_delta_px=float(np.mean(abs_delta_px)) if abs_delta_px else 0.0,
        zlib9_delta_stream_bytes=zlib9_bytes,
        prior_code_delta_bytes=(delta_bits + 7) // 8,
        reconstruction_bit_identical=True,
    )
    return section, report, dec_frames


# --------------------------------------------------------------------------- #
# 5. decode (numpy + stdlib ONLY — inflate-portable; consumes every seed byte)  #
# --------------------------------------------------------------------------- #
def decode_dash_phase_carrier(
    section: bytes,
    *,
    xi_twists_external: np.ndarray | None = None,
    geom: GroundHomographyGeom | None = None,
) -> list[list[DecodedDash]]:
    """Decode the section into per-frame phase-correct dash states.

    When the section was written with ``include_xi=False`` the (already-shipped, L68)
    fp16 twists MUST be passed via ``xi_twists_external`` (fp16-roundtripped)."""
    if section[: len(DASH_PHASE_MAGIC)] != DASH_PHASE_MAGIC:
        raise DashPhaseError("bad dash-phase magic")
    off = len(DASH_PHASE_MAGIC)
    (hlen,) = struct.unpack_from("<I", section, off)
    off += 4
    hdr = json.loads(section[off : off + hlen].decode("utf-8"))
    off += hlen
    P = int(hdr["n_frames"])
    H, W = int(hdr["height"]), int(hdr["width"])
    q = float(hdr["q_px"])
    tilt_bins = int(hdr["tilt_bins"])
    gap_xi = str(hdr["gap_xi"])
    dormant_max = int(hdr["dormant_max_frames"])
    if bool(hdr["include_xi"]):
        xi_dec = (
            np.frombuffer(section[off : off + P * 6 * 2], dtype=np.float16)
            .astype(np.float64)
            .reshape(P, 6)
        )
        off += P * 6 * 2
    else:
        if xi_twists_external is None:
            raise DashPhaseError("include_xi=False section needs xi_twists_external (the L68 dxi)")
        xi_dec = np.asarray(xi_twists_external, dtype=np.float64).reshape(-1, 6)[:P]
    (slen,) = struct.unpack_from("<I", section, off)
    off += 4
    stream = section[off : off + slen]
    off += slen
    if off != len(section):
        raise DashPhaseError("trailing bytes after stream (unconsumed seed bytes — NO-FAKE refusal)")

    lengths = {s: int(v) for s, v in zip(_SYMBOL_ORDER, hdr["code_lengths"])}
    codes = _canonical_codes(lengths)
    decode_tree = {(c, n): s for s, (c, n) in codes.items()}
    geom = geom or GroundHomographyGeom.eon(native_hw=(H, W), pitch=float(hdr["pitch"]))

    br = _BitReader(stream)
    tracks: list[_Track] = []
    next_tid = 0
    out: list[list[DecodedDash]] = []

    def _read_birth_anchor() -> tuple[np.ndarray, float, int]:
        rq = br.read(16)
        cq = br.read(16)
        tq = br.read(4)
        area = br.read_varint()
        return (
            np.array([rq * q, cq * q], dtype=np.float64),
            _tilt_dequantize(tq, tilt_bins),
            int(area),
        )

    def _read_delta_apply(pred_c: np.ndarray, tilt: float) -> np.ndarray:
        e_s = np.array([np.sin(tilt), np.cos(tilt)], dtype=np.float64)
        e_n = np.array([np.cos(tilt), -np.sin(tilt)], dtype=np.float64)
        ds = _read_delta_symbol(br, decode_tree)
        dn = _read_delta_symbol(br, decode_tree)
        return pred_c + (ds * q) * e_s + (dn * q) * e_n

    for p in range(P):
        frame_out: list[DecodedDash] = []
        if p > 0:
            xi_c = _cross_xi(xi_dec, p, gap_xi)
            live = [t for t in tracks if t.alive]
            dormant = [t for t in tracks if not t.alive and t.dormant_for <= dormant_max]
            for t in live + dormant:
                t.centroid = _advect_points_rc(t.centroid[None], xi_c, geom)[0]
            for t in live:
                if br.read_bit():
                    t.centroid = _read_delta_apply(t.centroid, t.tilt)
                    frame_out.append(
                        DecodedDash(t.tid, (float(t.centroid[0]), float(t.centroid[1])), t.tilt, t.area, False)
                    )
                else:
                    t.alive = False
                    t.dormant_for = 0
            n_new = br.read_varint()
            reborn_d: set[int] = set()
            for _ in range(n_new):
                if br.read_bit():
                    di = br.read_varint()
                    if di >= len(dormant) or di in reborn_d:
                        raise DashPhaseError("rebirth index out of range (corrupt payload)")
                    reborn_d.add(di)
                    t = dormant[di]
                    t.centroid = _read_delta_apply(t.centroid, t.tilt)
                    t.alive = True
                    t.dormant_for = 0
                    frame_out.append(
                        DecodedDash(t.tid, (float(t.centroid[0]), float(t.centroid[1])), t.tilt, t.area, True)
                    )
                else:
                    dec_c, tilt, area = _read_birth_anchor()
                    tracks.append(_Track(next_tid, dec_c, tilt, area, True))
                    frame_out.append(DecodedDash(next_tid, (float(dec_c[0]), float(dec_c[1])), tilt, area, True))
                    next_tid += 1
            for di, t in enumerate(dormant):
                if di not in reborn_d:
                    t.dormant_for += 1
        else:
            n0 = br.read_varint()
            for _ in range(n0):
                dec_c, tilt, area = _read_birth_anchor()
                tracks.append(_Track(next_tid, dec_c, tilt, area, True))
                frame_out.append(DecodedDash(next_tid, (float(dec_c[0]), float(dec_c[1])), tilt, area, True))
                next_tid += 1
        out.append(frame_out)
    br.assert_fully_consumed()
    return out


# --------------------------------------------------------------------------- #
# 6. end-to-end convenience (cached authority → section + report)               #
# --------------------------------------------------------------------------- #
def dash_phase_carrier_report(
    lstars: np.ndarray,
    gt_poses: np.ndarray,
    cfg: DashPhaseConfig | None = None,
) -> tuple[bytes, DashPhaseReport]:
    """Cached ``(lstars, gt_poses)`` → encoded section + measured report.

    ξ is calibrated from the poses with the SAME ``xi_from_pose_calibration`` the pose /
    phase carriers use (dual-use ξ — no separate ξ is derived)."""
    from tac.boundary_math.warp_real_luma_frame0 import xi_from_pose_calibration

    cfg = cfg or DashPhaseConfig()
    lstars = np.asarray(lstars)
    P = lstars.shape[0]
    xi = np.stack(
        [
            xi_from_pose_calibration(np.asarray(gt_poses)[p], s_t=cfg.s_t, s_r=cfg.s_r, pitch=cfg.pitch)
            for p in range(P)
        ]
    )
    section, report, _ = encode_dash_phase_carrier(lstars, xi, cfg)
    return section, report
