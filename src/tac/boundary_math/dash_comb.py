# SPDX-License-Identifier: MIT
"""Max-plus/tropical DASH-COMB corrector (#287) — the cell-problem corrector of the
dash-erasure homogenization law (``tac.canonical_equations.dash_erasure_homogenization_20260707``,
hunt memo ``.omx/research/viscosity_theory_alignment_hunt_20260707.md`` §3).

THE OBJECT. Lane dashes are a δ-periodic microstructure ALONG the lane tangent, with the
period constant in GROUND meters (so RANGE-DEPENDENT in image pixels under perspective —
#215; a uniform image-space period is WRONG) and the PHASE transported by the ego screw ξ
(dash positions are STATIC in the ground/world frame; per-pair phase = cumulative
ego-forward-distance — MEASURED, #215 dashgap deep-dive). Two-scale expansion
``u ≈ ū + δ·v(x/δ)``: the coarse witness chart holds ū (the homogenized solid band); THIS
module is the analytic corrector ``δ·v(x/δ)`` — a periodic gate

    comb(w) = [ (w - w0) mod T  <  duty·T ],      w = E_k + forward(v)

over the WORLD forward coordinate ``w`` (``E_k`` = cumulative ego forward distance at pair
``k``; ``forward(v)`` = flat-ground IPM distance of image row ``v``). The comb MODULATES the
analytic lane render-band alpha (``tac.boundary_math.analytic_lane_render_band``); it never
replaces it. "Max-plus/tropical": the band union stays a max over lines (tropical sum) and
the comb multiplies each dashed line's gate (tropical: adds the gap penalty −∞ in the gaps);
the smooth form is the AA clip of the signed phase-distance into the ON cell.

RULE-118 / RATE: the comb is a GENERIC deterministic algorithm (FREE in inflate.py). The
counted payload is 3 global scalars (period T, duty, ego scale) + one phase offset w0 per
lane slot (~2-6 floats total) — the per-pair per-line dash PHASE floats the fitted gate
would store are replaced by phase-from-ξ (the ξ carrier is already counted elsewhere).

NO LEARNED PARAMS. Deterministic, no RNG. numpy fp32 reference is the bit-identical
authority; the MLX twin (``comb_gate_of_world_mlx``) is parity-gated by tests.

NO-FAKE: every function does the work its name claims on the real fitted ``LaneLine``
geometry + the real PoseNet ego channel. ``fit_dash_comb`` is honestly a FIT (deterministic
grid search + circular statistics), documented as such; its ``mean_concentration`` output
MEASURES how well phase-from-ξ explains the per-pair fitted phases (resultant length; 1.0 =
perfect ego transport, ~0 = refuted) — the fit reports its own quality instead of assuming
the transport claim.

Borrowed-substrate accounting:
  * BORROWED (cited): openpilot flat-ground IPM row->forward mapping (via
    ``lane_sdf_component``); circular-mean phase statistics (standard directional stats);
    LPV/Braides homogenization corrector FRAMING (theory, cited in the equation module).
  * OURS-ORIGINAL: the ego-transported world-frame dash comb as a render-band alpha
    modulator with per-slot phase + global (T, duty, scale) fit by concentration search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.boundary_math.analytic_lane_render_band import (
    DEFAULT_DASH_FORWARD_MAX_M,
    LaneBandPrior,
    _line_row_params,
)
from tac.boundary_math.lane_sdf_component import (
    _CAM_H,
    _FY,
    _SEG_H,
    _SEG_W,
    _V_HORIZON,
    LaneLine,
    cluster_lane_lines,
    fit_lane_line,
)

# Slot width (m) for matching a lane line to its world-phase offset across pairs: lane
# lines live near lateral multiples of ~half a lane width; quantizing the mid-field
# lateral at 1.8 m separates ego-left/ego-right/adjacent lines deterministically.
DEFAULT_SLOT_WIDTH_M: float = 1.8
# Mid-field forward (m) at which a line's lateral is evaluated for slot assignment
# (the IPM lateral is most stable mid-field; matches cluster_lane_lines' 6-45 m window).
SLOT_EVAL_FORWARD_M: float = 20.0


# ---------------------------------------------------------------------------
# The comb gate (numpy fp32 reference = bit-identical authority; MLX twin below).
# ---------------------------------------------------------------------------
def comb_gate_of_world(
    w_m: np.ndarray, *, period_m: float, duty: float, phase0_m: float,
    softness_m: float = 0.0,
) -> np.ndarray:
    """Periodic dash gate over the WORLD forward coordinate ``w_m`` (meters), fp32 in [0,1].

    ON cell = ``{ w : (w - phase0_m) mod period_m < duty*period_m }``. ``softness_m = 0``
    -> hard {0,1} gate. ``softness_m > 0`` -> AA ramp ``clip(d/softness + 0.5, 0, 1)`` on
    the signed distance ``d`` (m) into the ON cell (positive inside), computed within the
    principal period cell (exact for softness << gap length; the clip saturates before the
    wrap approximation matters at realistic softness ~0.3 m vs gaps ~ meters).
    """
    if period_m <= 0.0:
        raise ValueError(f"period_m must be > 0, got {period_m}")
    if not (0.0 < duty < 1.0):
        raise ValueError(f"duty must be in (0,1), got {duty}")
    w = np.asarray(w_m, np.float64)
    T = float(period_m)
    u = np.mod(w - float(phase0_m), T)
    if softness_m <= 0.0:
        # hard gate: the canonical strict convention (matches _line_row_params' fitted gate)
        return (u < float(duty) * T).astype(np.float32)
    half_on = 0.5 * float(duty) * T
    d = half_on - np.abs(u - half_on)  # signed distance (m) into the ON cell
    return np.clip(d / float(softness_m) + 0.5, 0.0, 1.0).astype(np.float32)


def comb_gate_of_world_mlx(
    w_m: Any, *, period_m: float, duty: float, phase0_m: float, softness_m: float = 0.0,
) -> Any:
    """MLX twin of :func:`comb_gate_of_world` (elementwise; mx.compile-friendly).

    Parity vs the numpy fp32 reference is test-gated (``tests/test_dash_comb.py``);
    numpy stays the bit-identical authority per the deterministic-repro spine.
    """
    import mlx.core as mx

    if period_m <= 0.0:
        raise ValueError(f"period_m must be > 0, got {period_m}")
    if not (0.0 < duty < 1.0):
        raise ValueError(f"duty must be in (0,1), got {duty}")
    T = float(period_m)
    u = (w_m - float(phase0_m)) % T
    if softness_m <= 0.0:
        return (u < float(duty) * T).astype(mx.float32)
    half_on = 0.5 * float(duty) * T
    d = half_on - mx.abs(u - half_on)
    return mx.clip(d / float(softness_m) + 0.5, 0.0, 1.0).astype(mx.float32)


def ego_cumulative_distance(fwd_raw: np.ndarray, scale: float) -> np.ndarray:
    """Cumulative ego forward distance (m) per pair: ``E = scale * cumsum(fwd_raw)``.

    ``fwd_raw`` is the raw PoseNet forward channel per pair (``gt_poses[:, 0]``,
    up-to-affine per LDM Thm 1); ``scale`` is the affine calibration fit by
    :func:`fit_dash_comb` (the constant offset is absorbed into the per-slot phase w0).
    """
    return float(scale) * np.cumsum(np.asarray(fwd_raw, np.float64))


def line_slot(line: LaneLine, *, slot_width_m: float = DEFAULT_SLOT_WIDTH_M) -> int:
    """Deterministic slot key for matching a lane line across pairs: the centerline
    lateral (m) at the mid-field forward, quantized at ``slot_width_m``."""
    lat_mid = float(np.polyval(line.centerline_coeffs, SLOT_EVAL_FORWARD_M))
    return int(np.round(lat_mid / float(slot_width_m)))


# ---------------------------------------------------------------------------
# The global comb fit (deterministic; reports its own quality).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DashCombFit:
    """The fitted comb: ONE shared ego scale + per-slot (period, duty) + SPARSE strided
    world-phase anchors per slot.

    Three MEASURED structure facts shape this (all n600, [macOS-CPU advisory]):
    (1) the cell problem is PER LINE FAMILY — each slot carries its own period/duty
    (ego-lane slots T≈7.5 m vs adjacent slots T≈19-21 m; one global period destroys the
    phase statistic); (2) LOCAL ego transport is REAL — consecutive-pair fitted-phase
    deltas concentrate at ``-ds mod T_slot`` (``mean_pairwise_concentration``, the
    transport-law verdict; null ≈ 1/sqrt(n)); (3) GLOBAL single-anchor coherence is
    WEAK (~0.26 at best joint (T, scale) refinement) — over ~260 dash periods the phase
    decoheres at achievable period precision. Hence the comb's phase payload = SPARSE
    ANCHORS every ``anchor_window`` pairs (circular mean of ``(E_j + phase_j) mod
    T_slot`` within the window), transported WITHIN each window by ego distance —
    still ~``anchor_window``x fewer floats than the per-pair fitted phase."""

    scale: float                       # ego affine scale: E = scale * cumsum(fwd_raw)
    period_m: float                    # fallback period T (m) (median over all fits)
    duty: float                        # fallback ON fraction
    phase0_by_slot: dict[int, float]   # per-slot GLOBAL world phase w0 (m) (fallback)
    period_by_slot: dict[int, float]   # per-slot dash period T_slot (m)
    duty_by_slot: dict[int, float]     # per-slot ON fraction
    transported_by_slot: dict[int, bool]  # per-slot: ego-transported vs STATIC periodic
    anchor_window: int                 # pairs per anchor (>= 1)
    anchor_phase0_by_slot: dict[int, list[float]]  # per-slot strided anchors (NaN = no data)
    concentration_by_slot: dict[int, float]        # cumulative/global (weak-form) conc
    anchored_concentration_by_slot: dict[int, float]  # within-window conc (the comb's form)
    pairwise_concentration_by_slot: dict[int, float]  # local-transport verdict
    mean_concentration: float          # count-weighted global (weak-form) concentration
    mean_pairwise_concentration: float  # count-weighted local-transport concentration
    concentration_at_zero_scale: float  # anchored score at scale 0 (the no-transport null)
    n_dashed_fits: int
    n_pairs: int
    slot_width_m: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def n_anchor_floats(self) -> int:
        """The comb's phase payload size (floats) — the rate-accounting number."""
        return int(sum(int(np.sum(np.isfinite(np.asarray(v, np.float64)))) for v in
                       self.anchor_phase0_by_slot.values()))

    def params_for(self, line: LaneLine,
                   pair_idx: int | None = None) -> tuple[float, float, float, bool]:
        """(period_m, duty, phase0_m, transported) for a line via its slot. With
        ``pair_idx`` the phase comes from the pair's strided anchor (NaN anchors fall
        back to the nearest finite anchor, then the slot's global w0); slots never fit
        fall back to the global medians + count-weighted circular-mean phase.
        ``transported`` False => the slot's periodic structure is STATIC (MEASURED:
        outer slots) and the gate must NOT be advanced by ego distance."""
        slot = line_slot(line, slot_width_m=self.slot_width_m)
        if slot not in self.phase0_by_slot:
            return (self.period_m, self.duty,
                    float(self.provenance.get("global_phase0_m", 0.0)), False)
        T_s = self.period_by_slot.get(slot, self.period_m)
        duty_s = self.duty_by_slot.get(slot, self.duty)
        w0 = self.phase0_by_slot[slot]
        anchors = self.anchor_phase0_by_slot.get(slot)
        if pair_idx is not None and anchors and self.anchor_window > 0:
            a = np.asarray(anchors, np.float64)
            ai = min(int(pair_idx) // int(self.anchor_window), a.shape[0] - 1)
            if np.isfinite(a[ai]):
                w0 = float(a[ai])
            else:
                finite = np.where(np.isfinite(a))[0]
                if finite.size:
                    w0 = float(a[finite[int(np.argmin(np.abs(finite - ai)))]])
        return (T_s, duty_s, w0, bool(self.transported_by_slot.get(slot, False)))


def fit_dash_comb(
    per_pair_lines: list[list[LaneLine]],
    fwd_raw: np.ndarray,
    *,
    slot_width_m: float = DEFAULT_SLOT_WIDTH_M,
    ds_max_m: float = 4.0,
    n_scale: int = 321,
    min_slot_count: int = 20,
    anchor_window: int = 24,
) -> DashCombFit:
    """Fit the comb (ONE shared ego scale + per-slot period/duty/w0) from per-pair
    fitted lines + the raw PoseNet forward channel. DETERMINISTIC (grid search +
    circular stats, no RNG).

    Model: per-pair fitted dash phase ``phase_k`` of a line in slot ``s`` satisfies
    ``w0_s ≡ E_k(a) + phase_k (mod T_s)`` when dashes are static in the world frame and
    ``E_k(a) = a·cumsum(fwd_raw)`` is the ego distance. ``T_s``/``duty_s`` = per-slot
    medians of the per-pair fits (the cell problem is per line FAMILY — ego-lane vs
    adjacent-lane lines carry different measured periods; one global period destroys
    the statistic). The shared scale ``a`` is fit from the PAIRWISE local-transport
    statistic (consecutive-entry phase deltas; robust to long-range period-precision
    decoherence), on a period-aware symmetric grid (|mean ds| <= ``ds_max_m`` m/pair) +
    local refinement. The phase payload = strided anchors every ``anchor_window``
    pairs. The concentrations are the measured verdicts on the transport claim
    (reported, never assumed): ``mean_pairwise_concentration`` (local transport, THE
    law verdict), ``concentration_by_slot`` (weak global single-anchor form),
    ``anchored_concentration_by_slot`` (the comb's operating form).
    """
    fwd_raw = np.asarray(fwd_raw, np.float64).reshape(-1)
    P = int(fwd_raw.shape[0])
    if len(per_pair_lines) != P:
        raise ValueError(f"per_pair_lines has {len(per_pair_lines)} pairs, fwd_raw has {P}")

    ks: list[int] = []
    slots: list[int] = []
    phases: list[float] = []
    periods: list[float] = []
    duties: list[float] = []
    for k, lines in enumerate(per_pair_lines):
        for ln in lines:
            if ln.dash_period_m > 0.0:
                ks.append(k)
                slots.append(line_slot(ln, slot_width_m=slot_width_m))
                phases.append(float(ln.dash_phase_m))
                periods.append(float(ln.dash_period_m))
                duties.append(float(ln.dash_duty))
    n_dashed = len(ks)
    if n_dashed < max(8, min_slot_count):
        raise ValueError(
            f"fit_dash_comb: only {n_dashed} dashed line fits across {P} pairs; "
            "not enough signal to fit the comb (need >= max(8, min_slot_count))."
        )
    k_arr = np.asarray(ks, np.int64)
    slot_arr = np.asarray(slots, np.int64)
    phase_arr = np.asarray(phases, np.float64)
    period_arr = np.asarray(periods, np.float64)
    duty_arr = np.asarray(duties, np.float64)
    T_fallback = float(np.median(periods))
    duty_fallback = float(np.clip(np.median(duties), 0.1, 0.9))

    csum = np.cumsum(fwd_raw)
    base = float(np.mean(np.abs(fwd_raw[1:]))) if P > 1 else 0.0
    a_max = ds_max_m / base if base > 1e-12 else 0.0

    uniq_slots = np.unique(slot_arr)
    counts = {int(s): int((slot_arr == s).sum()) for s in uniq_slots}
    live_slots = [s for s in uniq_slots if counts[int(s)] >= int(min_slot_count)]
    if not live_slots:
        live_slots = list(uniq_slots)  # fall back to all (small-P tests); counts still weight

    slot_masks = [slot_arr == s for s in live_slots]
    slot_counts = np.asarray([float(m.sum()) for m in slot_masks])
    csum_e = csum[k_arr]
    # PER-SLOT period/duty (the cell problem is per line family — MEASURED n600:
    # ego-lane slots T~7.6 m, adjacent slots T~12-14 m; one global T destroys the
    # circular statistic, the second fit class bug):
    T_by_slot = {int(s): float(np.median(period_arr[m])) for s, m in zip(live_slots, slot_masks)}
    duty_by_slot = {int(s): float(np.clip(np.median(duty_arr[m]), 0.1, 0.9))
                    for s, m in zip(live_slots, slot_masks)}
    # ---- (1) SHARED EGO SCALE from the ANCHORED (within-window) statistic. ---------
    # The comb's operating form is "sparse anchors + within-window ego transport", so
    # the scale objective IS the anchored concentration: at the true scale the
    # residuals (E_k + phase_k) mod T_slot hold still WITHIN each window; at a wrong
    # scale they wrap ~W*ds/T periods and the windows decohere. This objective is
    # (a) robust to long-range period-precision decoherence (MEASURED: the global
    # statistic caps at ~0.26 over ~260 periods even after joint (T, a) refinement)
    # and (b) sharply peaked in the scale, unlike the pairwise-delta statistic which
    # is nearly FLAT in the scale at constant speed (dphase is itself ~constant).
    W = int(anchor_window)
    if W < 1:
        raise ValueError(f"anchor_window must be >= 1, got {anchor_window}")
    n_anchors = int(np.ceil(P / W))
    n_live = len(live_slots)
    T_min = float(min(T_by_slot.values())) if T_by_slot else T_fallback
    # per-entry (live only) period + segment id (slot x anchor-block)
    live_any = np.zeros(csum_e.shape[0], bool)
    ent_T = np.full(csum_e.shape[0], T_fallback, np.float64)
    seg_id = np.zeros(csum_e.shape[0], np.int64)
    for si, (s, m) in enumerate(zip(live_slots, slot_masks)):
        live_any |= m
        ent_T[m] = T_by_slot[int(s)]
        seg_id[m] = si * n_anchors + (k_arr[m] // W)
    csum_l = csum_e[live_any]
    phase_l = phase_arr[live_any]
    T_l = ent_T[live_any]
    seg_l = seg_id[live_any]
    n_seg = n_live * n_anchors
    slot_n = np.maximum(np.asarray([float(c) for c in slot_counts]), 1.0)
    n_live_entries = float(live_any.sum())

    def _slot_sums(a: float) -> np.ndarray:
        """Per-slot sum over anchor windows of |window phase resultant| at scale a
        (dividing by the slot count gives that slot's anchored concentration)."""
        ang = 2.0 * np.pi * ((a * csum_l + phase_l) / T_l)
        zr = np.bincount(seg_l, weights=np.cos(ang), minlength=n_seg)
        zi = np.bincount(seg_l, weights=np.sin(ang), minlength=n_seg)
        return np.hypot(zr, zi).reshape(n_live, n_anchors).sum(axis=1)

    # PER-SLOT TRANSPORTED-vs-STATIC structure (MEASURED n600: ego-lane slots ±1 are
    # ego-transported — anchored conc 0.41-0.47 at the true scale vs 0.19-0.21 static —
    # while outer slots ±3/±4/±5 are STATIC periodic structure — 0.57-0.80 at zero
    # scale, collapsing under transport). The scale objective is therefore the
    # TRANSPORT GAIN: sum over slots of max(anchored(a) - anchored(0), 0); static
    # slots contribute zero and are classified below.
    sums0 = _slot_sums(0.0)

    def _gain(a: float) -> float:
        return float(np.maximum(_slot_sums(a) - sums0, 0.0).sum())

    # resolution: da * (W * mean|fwd_raw|) < T_min/32 within a window
    if a_max > 0.0 and base > 0.0:
        n_needed = int(np.ceil(2.0 * a_max * W * base * 32.0 / T_min)) + 1
        n_grid = int(min(max(int(n_scale), n_needed), 50_000))
        grid = np.linspace(-a_max, a_max, n_grid)
        scores = np.asarray([_gain(float(a)) for a in grid])
        bi = int(np.argmax(scores))
        step = grid[1] - grid[0]
        fine = np.linspace(grid[bi] - 2.0 * step, grid[bi] + 2.0 * step, 129)
        best_a = float(fine[int(np.argmax([_gain(float(a)) for a in fine]))])
    else:
        best_a = 0.0
    sums_best = _slot_sums(best_a)
    # a slot is TRANSPORTED iff transport buys it a real concentration margin
    transported: dict[int, bool] = {}
    for si, s in enumerate(live_slots):
        transported[int(s)] = bool((sums_best[si] - sums0[si]) / slot_n[si] > 0.02)
    if not any(transported.values()):
        best_a = 0.0  # no transported slot -> the scale is unidentified; comb all-static
    anch_best = float(np.where([transported[int(s)] for s in live_slots],
                               sums_best, sums0).sum() / max(n_live_entries, 1.0))
    anch_null = float(sums0.sum() / max(n_live_entries, 1.0))

    # ---- (1b) PAIRWISE local-transport concentration (reported verdict). -----------
    pair_conc_by_slot: dict[int, float] = {}
    pair_num = 0.0
    pair_den = 0.0
    for s, m in zip(live_slots, slot_masks):
        idx = np.where(m)[0]
        idx = idx[np.argsort(k_arr[idx], kind="stable")]
        kk = k_arr[idx]
        step_ok = np.where((np.diff(kk) >= 1) & (np.diff(kk) <= 3))[0]
        if step_ok.size == 0:
            continue
        dph = phase_arr[idx[step_ok + 1]] - phase_arr[idx[step_ok]]
        dcs = csum_e[idx[step_ok + 1]] - csum_e[idx[step_ok]]
        a_eff = best_a if transported.get(int(s), False) else 0.0
        zc = np.exp(2j * np.pi * (dph + a_eff * dcs) / T_by_slot[int(s)]).mean()
        pair_conc_by_slot[int(s)] = float(np.abs(zc))
        pair_num += float(np.abs(zc)) * step_ok.size
        pair_den += step_ok.size
    pair_mean = pair_num / pair_den if pair_den else float("nan")

    # ---- (2) GLOBAL w0 (weak-form fallback) + SPARSE STRIDED ANCHORS, per-slot mode. --
    E_e = best_a * csum_e
    conc_by_slot: dict[int, float] = {}
    w0_by_slot: dict[int, float] = {}
    anchors_by_slot: dict[int, list[float]] = {}
    anch_conc_by_slot: dict[int, float] = {}
    mean_num = 0.0
    for s, m, c in zip(live_slots, slot_masks, slot_counts):
        Ts = T_by_slot[int(s)]
        E_s = E_e[m] if transported.get(int(s), False) else 0.0
        zs = np.exp(2j * np.pi * ((E_s + phase_arr[m]) / Ts))
        zm = zs.mean()
        conc_by_slot[int(s)] = float(np.abs(zm))
        w0_by_slot[int(s)] = float(np.mod(np.angle(zm) * Ts / (2.0 * np.pi), Ts))
        mean_num += float(np.abs(zm)) * c
        kk = k_arr[m]
        blocks = kk // W
        anch = np.full(n_anchors, np.nan)
        a_num = 0.0
        a_den = 0.0
        for ai in range(n_anchors):
            bm = blocks == ai
            if bm.any():
                zb = zs[bm].mean()
                anch[ai] = float(np.mod(np.angle(zb) * Ts / (2.0 * np.pi), Ts))
                a_num += float(np.abs(zb)) * float(bm.sum())
                a_den += float(bm.sum())
        anchors_by_slot[int(s)] = [float(x) for x in anch]
        anch_conc_by_slot[int(s)] = a_num / a_den if a_den else float("nan")
    mean_conc = mean_num / max(float(slot_counts.sum()), 1.0)

    # count-weighted global circular mean at the fallback period (only for never-fit slots)
    ang_best = 2.0 * np.pi * ((E_e + phase_arr) / T_fallback)
    zg = np.exp(1j * ang_best).mean()
    global_phase0 = float(np.mod(np.angle(zg) * T_fallback / (2.0 * np.pi), T_fallback))

    return DashCombFit(
        scale=best_a,
        period_m=T_fallback,
        duty=duty_fallback,
        phase0_by_slot=w0_by_slot,
        period_by_slot=T_by_slot,
        duty_by_slot=duty_by_slot,
        transported_by_slot=transported,
        anchor_window=W,
        anchor_phase0_by_slot=anchors_by_slot,
        concentration_by_slot=conc_by_slot,
        anchored_concentration_by_slot=anch_conc_by_slot,
        pairwise_concentration_by_slot=pair_conc_by_slot,
        mean_concentration=float(mean_conc),
        mean_pairwise_concentration=float(pair_mean),
        concentration_at_zero_scale=float(anch_null),
        n_dashed_fits=n_dashed,
        n_pairs=P,
        slot_width_m=float(slot_width_m),
        provenance={
            "global_phase0_m": global_phase0,
            "ds_max_m": float(ds_max_m),
            "n_scale": int(n_scale),
            "min_slot_count": int(min_slot_count),
            "mean_abs_ds_m_at_best_scale": float(abs(best_a) * base),
            "slot_counts": {int(s): counts[int(s)] for s in uniq_slots},
            "n_pairwise": int(pair_den),
            "anchored_score_best": float(anch_best),
            "anchored_score_null": float(anch_null),
        },
    )


# ---------------------------------------------------------------------------
# Row gate + combed coverage raster (mirrors the range-dependent raster; the comb
# replaces the per-pair fitted phase with the ego-transported world phase).
# ---------------------------------------------------------------------------
def comb_row_gate(
    v_rows: np.ndarray, *, ego_dist_m: float, period_m: float, duty: float,
    phase0_m: float, forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M,
    softness_m: float = 0.0, cam_h: float = _CAM_H, fy: float = _FY,
    v_h: float = _V_HORIZON,
) -> np.ndarray:
    """Per-image-row comb gate (fp32, [0,1]): ``comb(E_k + forward(v))`` where
    ``forward(v)`` is the flat-ground IPM distance of row ``v``; rows with forward >=
    ``forward_max_m`` are NOT gated (gate 1.0) per the #215 SegNet-Nyquist range rule
    (beyond ~55 m the net reads a smeared continuous line; gating there creates FN)."""
    v = np.asarray(v_rows, np.float64)
    forward = cam_h * fy / np.maximum(v - v_h, 1e-3)
    g = comb_gate_of_world(
        forward + float(ego_dist_m), period_m=period_m, duty=duty,
        phase0_m=phase0_m, softness_m=softness_m,
    )
    return np.where(forward < float(forward_max_m), g, np.float32(1.0)).astype(np.float32)


def rasterize_lane_coverage_combed(
    lines: list[LaneLine], fit: DashCombFit, ego_dist_m: float, *,
    pair_idx: int | None = None,
    h: int = _SEG_H, w: int = _SEG_W, softness: float = 1.0,
    dash_forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M,
    comb_softness_m: float = 0.3, v_h: float = _V_HORIZON, cx: float | None = None,
) -> np.ndarray:
    """AA-SDF lane coverage (H,W) with the EGO-PHASE COMB as the dash gate.

    Identical geometry to ``analytic_lane_render_band.rasterize_lane_coverage_range_dependent``
    (same AA-SDF lateral coverage, same forward-range validity, same max-union over lines)
    EXCEPT the dash gate: a line the per-pair fit found DASHED (``dash_period_m > 0``) is
    gated by the comb (per-slot period/duty/world-phase from ``fit``, shared ego scale,
    transported to this pair by ``ego_dist_m``) instead of its per-pair fitted phase;
    non-dashed (solid) lines are never gated. The comb MODULATES the band; solid geometry
    is untouched."""
    H, W = int(h), int(w)
    cxx = float(W / 2.0) if cx is None else float(cx)
    cov = np.zeros((H, W), np.float32)
    if not lines:
        return cov
    rows = np.arange(H, dtype=np.float64)
    below = rows > (v_h + 1.0)
    if not below.any():
        return cov
    vr = rows[below]
    col = np.arange(W, dtype=np.float64)[None, :]
    soft = max(float(softness), 1e-6)
    acc = np.zeros((int(below.sum()), W), np.float64)
    for ln in lines:
        # dash_gate=False -> gate carries ONLY the forward-range validity; the comb
        # supplies the dash structure (ego-transported phase) below.
        u_c, hw_r, gate = _line_row_params(
            ln, vr, dash_gate=False, dash_forward_max_m=dash_forward_max_m, cx=cxx, v_h=v_h,
        )
        if ln.dash_period_m > 0.0:
            T_s, duty_s, w0_s, transported = fit.params_for(ln, pair_idx=pair_idx)
            gate = gate * comb_row_gate(
                vr, ego_dist_m=(ego_dist_m if transported else 0.0),
                period_m=T_s, duty=duty_s,
                phase0_m=w0_s, forward_max_m=dash_forward_max_m,
                softness_m=comb_softness_m, v_h=v_h,
            ).astype(np.float64)
        s = hw_r[:, None] - np.abs(col - u_c[:, None])
        cov_l = np.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, None]
        acc = np.maximum(acc, cov_l)
    cov[below] = acc.astype(np.float32)
    return cov


# ---------------------------------------------------------------------------
# End-to-end builder (the trainer wire-in entry): fit lines per pair, fit the comb
# once globally, rasterize combed coverage per pair.
# ---------------------------------------------------------------------------
def fit_lines_per_pair(
    lstars: np.ndarray, *, lane_cls: int = 1, centerline_deg: int = 3,
    v_h: float = _V_HORIZON,
) -> list[list[LaneLine]]:
    """Cluster + fit lane lines (WITH the per-pair dash fit — the comb consumes the
    fitted period/duty and the fitted phase as calibration data) for every pair."""
    out: list[list[LaneLine]] = []
    for pi in range(int(np.asarray(lstars).shape[0])):
        a = np.asarray(lstars[pi])
        lines: list[LaneLine] = []
        for c in cluster_lane_lines(a, lane_cls=lane_cls, v_h=v_h):
            ln = fit_lane_line(c, centerline_deg=centerline_deg, fit_dash=True, v_h=v_h)
            if ln is not None:
                lines.append(ln)
        out.append(lines)
    return out


def build_combed_lane_band_priors(
    lstars: np.ndarray, gt_poses: np.ndarray, *, lane_cls: int = 1,
    softness: float = 1.0, dash_forward_max_m: float = DEFAULT_DASH_FORWARD_MAX_M,
    comb_softness_m: float = 0.3, centerline_deg: int = 3, v_h: float = _V_HORIZON,
    slot_width_m: float = DEFAULT_SLOT_WIDTH_M,
) -> tuple[dict[int, LaneBandPrior], DashCombFit]:
    """Per-pair ``LaneBandPrior`` dict (keyed by PAIR index) with the EGO-PHASE COMB as
    the dash gate + the global :class:`DashCombFit`. The trainer wire-in duplicates each
    pair's prior to both code indices (2*pi, 2*pi+1), mirroring the fitted-gate path."""
    lst = np.asarray(lstars)
    P = int(lst.shape[0])
    gp = np.asarray(gt_poses, np.float64)
    if gp.shape[0] != P:
        raise ValueError(f"gt_poses pairs {gp.shape[0]} != lstars pairs {P}")
    per_pair = fit_lines_per_pair(lst, lane_cls=lane_cls, centerline_deg=centerline_deg, v_h=v_h)
    fit = fit_dash_comb(per_pair, gp[:, 0], slot_width_m=slot_width_m)
    E = ego_cumulative_distance(gp[:, 0], fit.scale)
    priors: dict[int, LaneBandPrior] = {}
    for pi in range(P):
        a = lst[pi]
        hh, ww = a.shape
        cov = rasterize_lane_coverage_combed(
            per_pair[pi], fit, float(E[pi]), pair_idx=pi, h=hh, w=ww, softness=softness,
            dash_forward_max_m=dash_forward_max_m, comb_softness_m=comb_softness_m, v_h=v_h,
        )
        is_lane = a == int(lane_cls)
        nlane = int(is_lane.sum())
        priors[pi] = LaneBandPrior(
            coverage=cov,
            lines=per_pair[pi],
            n_lines=len(per_pair[pi]),
            total_floats=int(sum(ln.n_floats() for ln in per_pair[pi])),
            n_dash_modeled=int(sum(1 for ln in per_pair[pi] if ln.dash_period_m > 0.0)),
            band_recall=(float((cov[is_lane] >= 0.5).mean()) if nlane else float("nan")),
            gt_lane_frac=float(nlane) / float(a.size),
        )
    return priors, fit


__all__ = [
    "DEFAULT_SLOT_WIDTH_M",
    "DashCombFit",
    "SLOT_EVAL_FORWARD_M",
    "build_combed_lane_band_priors",
    "comb_gate_of_world",
    "comb_gate_of_world_mlx",
    "comb_row_gate",
    "ego_cumulative_distance",
    "fit_dash_comb",
    "fit_lines_per_pair",
    "line_slot",
    "rasterize_lane_coverage_combed",
]
