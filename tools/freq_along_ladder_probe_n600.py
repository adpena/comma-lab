#!/usr/bin/env python
"""$0 freq_along LADDER oracle probe (Mallat/Balle review row 2) -- n600, frozen ep650.

[macOS-CPU advisory] NON-PROMOTABLE. Arbitrates parabolic-scaling-ceiling vs config
coincidence for the measured 3.2x along-tangent deficit, in the honest ORACLE-CAPACITY
form (form b): the frozen witness render is composited with the analytic lane band whose
dash gate is the FOURIER-TRUNCATED ego-phase comb at along-bandwidth f in {DC, 8, 16, 25,
32} cyc/unit (U=192 px/unit, the basis's own y-chart), plus the full comb endpoint.
Form (a) (re-render the frozen field at a changed ``freq_along``) is declared UNSUPPORTED:
``dir_feats`` keeps the feature DIMENSION but swaps basis functions under weights calibrated
to the trained fa -> an untrained field, not added bandwidth. Pre-registration (design
written BEFORE measurement): .omx/research/freq_along_ladder_oracle_probe_20260707.md.

Per rung the truncated gate is, per line (slot params T, D, w0 + ego transport identical to
``rasterize_lane_coverage_combed``):

    g_f(w; v) = clip(D + sum_{k<=K(v)} (2/(k pi)) sin(k pi D) cos(2 pi k ((w-w0)/T - D/2)), 0, 1)
    K(v) = floor(f * delta_px(v) / U),  delta_px(v) = T (v - v_h)^2 / (cam_h fy)

K=0 -> g=D = the homogenized Gamma-limit band; f->inf -> the hard comb. Rows with forward
>= 55 m ungated (gate 1.0), matching ``comb_row_gate``.

INSTRUMENT-VALIDITY (binding, per the 4db610af2 review that OVERTURNED the tau probe's
FLAT-H verdict: H computed on the GT labels reads 0.7015 ~= the witness's 0.666-0.677 ->
zero dynamic range): this probe carries ctrl_GT (realized := lstars through the exact
metric machinery = the perfect-field endpoint + region-misfit floor) and cSOLID (the
FEED-08c c2 degraded endpoint), and scores each band ONLY if the dash-structure CONTRAST
(= r_mark - r_gap, a difference, NOT the range-dead ratio H) separates cCOMB from cSOLID
by >= 0.10 with ctrl_GT >= cCOMB (else INDETERMINATE-at-this-resolution). Closure:
C_b(f) = (contrast(c_f)-contrast(cSOLID)) / (contrast(cCOMB)-contrast(cSOLID)). AUTHORITY:
every number uses the ONE probe-render instrument (dcp.Renderer + _torch_R_to_camera_uint8
+ seg_argmax_batch, frozen CPU torch); no trainer-verdict numbers are mixed in.

MEASURES per pair per condition (mark/gap regions identical to the tau probe -- GT-derived,
condition-independent): d_seg, lane recall/FP/FN, dash-gap FP, r_mark, r_gap, contrast,
all also per forward band; H reported only for FEED-08e comparability (demoted).

THIN WRAPPER discipline: imports the dash-comb probe's Renderer + verdict path + cached
line fits/comb fit; live #205 run READ-ONLY (frozen ep650 snapshot reused). Chunked
resumable foreground (atomic tmp+replace state); free-RAM >= 20 GiB gate; VBATCH <= 6;
NO MPS. Pointer 0.19110 UNMOVED (this is a means).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import dash_comb_probe_n600 as dcp  # noqa: E402  (sets env caps, sys.path, torch threads)
import numpy as np  # noqa: E402

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    _line_row_params,
    rasterize_lane_coverage_range_dependent,
)
from tac.boundary_math.dash_comb import (  # noqa: E402
    ego_cumulative_distance,
    rasterize_lane_coverage_combed,
)
from tac.boundary_math.lane_sdf_component import (  # noqa: E402
    _CAM_H,
    _FY,
    _V_HORIZON,
)
from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _torch_R_to_camera_uint8,
)

OUT_DIR = REPO / "experiments/results/freq_along_ladder_probe_20260707"
STATE = OUT_DIR / "probe_state.ckpt.npz"
OUT = OUT_DIR / "freq_along_ladder_n600_20260707.json"

# px per coordinate unit along a near-vertical tangent in the basis's own chart:
# coords_grid normalizes y in [-1,1] over 384 rows -> 192 px/unit (pre-registered primary;
# the older memos' 256 px/unit x-convention is reported as a secondary mapping only).
U_PX_PER_UNIT = 192.0
RUNGS = [("cDC", 0.0), ("cF8", 8.0), ("cF16", 16.0), ("cF25", 25.0), ("cF32", 32.0)]
# ctrl_GT is realized := lstars (no render/SegNet); the SegNet-forward conditions follow.
CONDS = ["ctrl_GT", "c1_witness", "cSOLID"] + [lbl for lbl, _ in RUNGS] + ["cCOMB"]
SEG_CONDS = [c for c in CONDS if c != "ctrl_GT"]
DYNRANGE_FLOOR = 0.10  # pre-registered per-band contrast(cCOMB)-contrast(cSOLID) floor
VBATCH = int(dcp.os.environ.get("FALADDER_VBATCH", "6"))
SAVE_EVERY = int(dcp.os.environ.get("FALADDER_SAVE_EVERY", "24"))
METRICS = ["dseg", "rec", "fn", "fp", "gapfp",
           "mark_ct", "mark_lane_ct", "gap_ct", "gap_lane_ct"]


def fourier_truncated_row_gate(
    v_rows: np.ndarray, *, ego_dist_m: float, period_m: float, duty: float,
    phase0_m: float, f_along: float, forward_max_m: float,
    u_px_per_unit: float = U_PX_PER_UNIT, cam_h: float = _CAM_H, fy: float = _FY,
    v_h: float = _V_HORIZON,
) -> np.ndarray:
    """Per-image-row FOURIER-TRUNCATED comb gate (fp32 in [0,1]).

    The square-wave dash gate of ``comb_row_gate`` replaced by its Fourier series
    truncated at along-bandwidth ``f_along`` cyc/unit: only harmonics whose LOCAL image
    frequency k*nu1(v) <= f_along are kept (nu1(v) = u_px_per_unit / delta_px(v)).
    K(v)=0 -> the homogenized duty-amplitude band (Gamma-limit); truncation overshoot
    (Gibbs) is clipped to [0,1]. Rows with forward >= forward_max_m are NOT gated (1.0),
    matching ``comb_row_gate``'s #215 range rule.
    """
    if period_m <= 0.0:
        raise ValueError(f"period_m must be > 0, got {period_m}")
    if not (0.0 < duty < 1.0):
        raise ValueError(f"duty must be in (0,1), got {duty}")
    v = np.asarray(v_rows, np.float64)
    forward = cam_h * fy / np.maximum(v - v_h, 1e-3)
    w = forward + float(ego_dist_m)
    T = float(period_m)
    D = float(duty)
    x = (w - float(phase0_m)) / T - 0.5 * D  # phase in cycles, centered on the ON cell
    # px spanned by one world period at row v (the in-tree perspective mapping):
    delta_px = T * (v - v_h) ** 2 / (cam_h * fy)
    K = np.floor(float(f_along) * delta_px / float(u_px_per_unit)).astype(np.int64)
    K = np.maximum(K, 0)
    g = np.full(v.shape, D, np.float64)
    kmax = int(min(int(K.max()) if K.size else 0, 512))
    for k in range(1, kmax + 1):
        coef = (2.0 / (k * np.pi)) * np.sin(k * np.pi * D)
        g += np.where(K >= k, coef * np.cos(2.0 * np.pi * k * x), 0.0)
    g = np.clip(g, 0.0, 1.0)
    return np.where(forward < float(forward_max_m), g, 1.0).astype(np.float32)


def rasterize_lane_coverage_fourier_truncated(
    lines, fit, ego_dist_m: float, *, pair_idx: int | None, f_along: float,
    h: int, w: int, softness: float, dash_forward_max_m: float,
    v_h: float = _V_HORIZON, cx: float | None = None,
) -> np.ndarray:
    """AA-SDF lane coverage with the FOURIER-TRUNCATED comb as the dash gate.

    Mirror of ``rasterize_lane_coverage_combed`` (same geometry, same slot params via
    ``fit.params_for``, same ego transport) EXCEPT the gate: the truncated series above
    instead of the hard/AA comb. Dashed lines only; solid lines never gated.
    """
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
        u_c, hw_r, gate = _line_row_params(
            ln, vr, dash_gate=False, dash_forward_max_m=dash_forward_max_m, cx=cxx, v_h=v_h,
        )
        if ln.dash_period_m > 0.0:
            T_s, duty_s, w0_s, transported = fit.params_for(ln, pair_idx=pair_idx)
            gate = gate * fourier_truncated_row_gate(
                vr, ego_dist_m=(ego_dist_m if transported else 0.0),
                period_m=T_s, duty=duty_s, phase0_m=w0_s, f_along=f_along,
                forward_max_m=dash_forward_max_m, v_h=v_h,
            ).astype(np.float64)
        s = hw_r[:, None] - np.abs(col - u_c[:, None])
        cov_l = np.clip(s / soft + 0.5, 0.0, 1.0) * gate[:, None]
        acc = np.maximum(acc, cov_l)
    cov[below] = acc.astype(np.float32)
    return cov


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-seconds", type=float, default=520.0,
                    help="exit cleanly (state saved) after this many seconds; re-invoke to resume.")
    ap.add_argument("--num-pairs", type=int, default=600, help="n600 discipline: keep 600.")
    args = ap.parse_args()

    free = dcp._free_gib()
    if free < dcp.MIN_FREE_GIB:
        print(f"REFUSE: free RAM {free:.1f} GiB < {dcp.MIN_FREE_GIB} GiB "
              f"(live #205 run protection)", flush=True)
        sys.exit(3)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not dcp.CKPT_NPZ.exists():
        raise SystemExit(f"missing frozen ep650 snapshot {dcp.CKPT_NPZ} (run the FEED-08c "
                         f"probe's snapshot step first); live run dir stays READ-ONLY")

    params, code, m, cfg = dcp.load_ckpt()
    R = dcp.Renderer(params, code, m)
    seg = dcp.load_real_segnet("cpu")
    z = np.load(dcp.CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    gt_poses = np.asarray(z["gt_poses"], np.float64)
    P = min(int(args.num_pairs), int(z["n_pairs"]))
    per_pair_lines, fit = dcp._stage0_lines_and_fit(lst_all, gt_poses, P)
    E = ego_cumulative_distance(gt_poses[:P, 0], fit.scale)
    H_, W_ = m["render_h"], m["render_w"]
    fwd_rows = dcp._forward_of_rows(H_)
    band_row_idx = np.full(H_, -1, np.int64)
    for bi, (lo, hi) in enumerate(dcp.FWD_BANDS):
        band_row_idx[(fwd_rows >= lo) & (fwd_rows < hi)] = bi
    NB = len(dcp.FWD_BANDS)
    print(f"[{time.time() - t0:.1f}s] ckpt ep{cfg.get('__epoch', -1)} "
          f"(freq_along={m['so_freq_along']:g} n_dir_freqs={m['n_dir_freqs']} -> dyadic along "
          f"ladder {[m['so_freq_along'] * 2 ** k for k in range(m['n_dir_freqs'])]}) + segnet "
          f"+ cache + comb fit loaded; P={P} conds={CONDS}", flush=True)

    st: dict[str, np.ndarray] = {}
    done = np.zeros(P, bool)
    for c in CONDS:
        for mt in METRICS:
            st[f"{c}__{mt}"] = np.full(P, np.nan)
        for mt in ("mark_ct", "mark_lane_ct", "gap_ct", "gap_lane_ct"):
            st[f"{c}__{mt}_band"] = np.full((P, NB), np.nan)
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for k in st:
            if k in ck.files:
                st[k] = ck[k]
        done = ck["done"].astype(bool)
        print(f"[resume] {int(done.sum())}/{P} done", flush=True)

    def save():
        payload = dict(st)
        payload["done"] = done
        tmp = STATE.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(STATE)

    def score_condition(c, pi, r, mark, gap):
        gt_l = np.asarray(lst_all[pi], np.int64)
        n = float(gt_l.size)
        is_lane = gt_l == 1
        nlane = int(is_lane.sum())
        r_lane = r == 1
        fp_px = (~is_lane) & r_lane
        st[f"{c}__dseg"][pi] = float(np.count_nonzero(r != gt_l)) / n
        st[f"{c}__rec"][pi] = (float(np.count_nonzero(r[is_lane] == 1)) / nlane) if nlane else np.nan
        st[f"{c}__fn"][pi] = float((is_lane & ~r_lane).sum()) / n
        st[f"{c}__fp"][pi] = float(fp_px.sum()) / n
        st[f"{c}__gapfp"][pi] = float((fp_px & gap).sum()) / n
        st[f"{c}__mark_ct"][pi] = float(mark.sum())
        st[f"{c}__mark_lane_ct"][pi] = float((r_lane & mark).sum())
        st[f"{c}__gap_ct"][pi] = float(gap.sum())
        st[f"{c}__gap_lane_ct"][pi] = float((r_lane & gap).sum())
        for bi in range(NB):
            bm = band_row_idx == bi
            st[f"{c}__mark_ct_band"][pi, bi] = float(mark[bm].sum())
            st[f"{c}__mark_lane_ct_band"][pi, bi] = float((r_lane & mark)[bm].sum())
            st[f"{c}__gap_ct_band"][pi, bi] = float(gap[bm].sum())
            st[f"{c}__gap_lane_ct_band"][pi, bi] = float((r_lane & gap)[bm].sum())

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s0 in range(0, len(todo), VBATCH):
        chunk = todo[s0:s0 + VBATCH]
        frames = {c: [] for c in SEG_CONDS}
        regions = []
        for pi in chunk:
            bulk, lane = R.render_pair(pi)
            lines = per_pair_lines[pi]
            cov_solid = rasterize_lane_coverage_range_dependent(
                lines, h=H_, w=W_, softness=dcp.BAND_SOFTNESS_PX, dash_gate=False)
            cov_comb = rasterize_lane_coverage_combed(
                lines, fit, float(E[pi]), pair_idx=pi, h=H_, w=W_,
                softness=dcp.BAND_SOFTNESS_PX,
                dash_forward_max_m=dcp.DASH_FORWARD_MAX_M,
                comb_softness_m=dcp.COMB_SOFTNESS_M)
            # mark/gap regions: GT-derived, condition-INDEPENDENT (identical to the tau probe)
            mark = (cov_solid >= 0.5) & (cov_comb >= 0.5)
            gap = (cov_solid >= 0.5) & (cov_comb < 0.5)
            regions.append((mark, gap))
            # ctrl_GT: the perfect-field validity control (realized := lstars; no SegNet)
            score_condition("ctrl_GT", pi, np.asarray(lst_all[pi], np.int64), mark, gap)

            def comp(cov):
                a = cov[..., None].astype(np.float32)
                return _torch_R_to_camera_uint8(((1 - a) * bulk + a * lane).astype(np.float64))

            frames["c1_witness"].append(_torch_R_to_camera_uint8(bulk.astype(np.float64)))
            frames["cSOLID"].append(comp(cov_solid))
            for lbl, f_along in RUNGS:
                cov_f = rasterize_lane_coverage_fourier_truncated(
                    lines, fit, float(E[pi]), pair_idx=pi, f_along=f_along,
                    h=H_, w=W_, softness=dcp.BAND_SOFTNESS_PX,
                    dash_forward_max_m=dcp.DASH_FORWARD_MAX_M)
                frames[lbl].append(comp(cov_f))
            frames["cCOMB"].append(comp(cov_comb))

        for c in SEG_CONDS:
            realized = dcp.seg_argmax_batch(seg, frames[c])
            for j, pi in enumerate(chunk):
                mark, gap = regions[j]
                score_condition(c, pi, realized[j], mark, gap)
        for pi in chunk:
            done[pi] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | "
                  f"c1={np.nanmean(st['c1_witness__dseg']):.5f} "
                  f"DC={np.nanmean(st['cDC__gapfp']):.6f} "
                  f"f8={np.nanmean(st['cF8__gapfp']):.6f} "
                  f"f32={np.nanmean(st['cF32__gapfp']):.6f} "
                  f"comb={np.nanmean(st['cCOMB__gapfp']):.6f} "
                  f"| rss={dcp._peak_rss_gib():.1f}GiB", flush=True)
        if time.time() - t0 > float(args.chunk_seconds) and nd < P:
            save()
            print(f"[chunk-exit] {nd}/{P} done at {time.time() - t0:.1f}s; re-invoke to "
                  f"resume. peak_rss={dcp._peak_rss_gib():.2f}GiB", flush=True)
            sys.exit(0)
    save()

    def pooled(c, num, den, bi=None):
        a = st[f"{c}__{num}" + ("_band" if bi is not None else "")]
        b = st[f"{c}__{den}" + ("_band" if bi is not None else "")]
        if bi is not None:
            a, b = a[:, bi], b[:, bi]
        sa, sb = float(np.nansum(a)), float(np.nansum(b))
        return sa / sb if sb > 0 else float("nan")

    def h_of(c, bi=None):
        rm = pooled(c, "mark_lane_ct", "mark_ct", bi)
        rg = pooled(c, "gap_lane_ct", "gap_ct", bi)
        return (rg / rm if rm and rm > 0 else float("nan")), rm, rg

    def contrast_of(c, bi=None):
        rm = pooled(c, "mark_lane_ct", "mark_ct", bi)
        rg = pooled(c, "gap_lane_ct", "gap_ct", bi)
        return (rm - rg) if (np.isfinite(rm) and np.isfinite(rg)) else float("nan")

    table = []
    for c in CONDS:
        Hh, rm, rg = h_of(c)
        table.append({
            "cond": c,
            "f_along": dict(RUNGS).get(c, None),
            "n_done": int(done.sum()),
            "d_seg": float(np.nanmean(st[f"{c}__dseg"])),
            "lane_recall": float(np.nanmean(st[f"{c}__rec"])),
            "lane_fp": float(np.nanmean(st[f"{c}__fp"])),
            "lane_fn": float(np.nanmean(st[f"{c}__fn"])),
            "dash_gap_fp": float(np.nanmean(st[f"{c}__gapfp"])),
            "r_mark": rm, "r_gap": rg,
            "contrast": contrast_of(c),
            "H_index_secondary": Hh,
            "contrast_by_band": [contrast_of(c, bi) for bi in range(NB)],
            "H_by_band_secondary": [h_of(c, bi)[0] for bi in range(NB)],
            "r_mark_by_band": [h_of(c, bi)[1] for bi in range(NB)],
            "r_gap_by_band": [h_of(c, bi)[2] for bi in range(NB)],
        })

    # pre-registered dynamic-range gate + contrast closure per band (GT-condition AMENDED
    # after the 2-pair runnability smoke, BEFORE the n600 measurement, on structural
    # grounds: the analytic band paints its mark footprint at r~1 while GT lane labels are
    # thin within the fitted band, so contrast(composite) can legitimately EXCEED
    # contrast(GT); the GT validity requirement is separation from the DEGRADED endpoint):
    #   scoreable iff contrast(cCOMB)-contrast(cSOLID) >= 0.10
    #             AND contrast(ctrl_GT) >= contrast(cSOLID) + 0.05
    #   C_b(f) = (contrast(c_f) - contrast(cSOLID)) / (contrast(cCOMB) - contrast(cSOLID))
    ctr = {c: [contrast_of(c, bi) for bi in range(NB)] for c in CONDS}
    band_scoreable, band_gate_reason = [], []
    for bi in range(NB):
        den = ctr["cCOMB"][bi] - ctr["cSOLID"][bi]
        gt_sep = ctr["ctrl_GT"][bi] - ctr["cSOLID"][bi]
        ok = (np.isfinite(den) and den >= DYNRANGE_FLOOR
              and np.isfinite(gt_sep) and gt_sep >= 0.05)
        band_scoreable.append(bool(ok))
        band_gate_reason.append(
            f"den={den:.4f} (floor {DYNRANGE_FLOOR}), gt_sep={gt_sep:.4f} (floor 0.05)"
            if np.isfinite(den) else "den=nan")
    closure = {}
    for lbl, _f in RUNGS:
        row = []
        for bi in range(NB):
            if not band_scoreable[bi]:
                row.append(None)  # INDETERMINATE-at-this-resolution (pre-registered gate)
            else:
                den = ctr["cCOMB"][bi] - ctr["cSOLID"][bi]
                row.append(float((ctr[lbl][bi] - ctr["cSOLID"][bi]) / den))
        closure[lbl] = row

    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "task": "freq_along ladder ORACLE probe (Mallat/Balle row 2): Fourier-truncated "
                "ego-phase comb at along-bandwidth f, composited over the frozen ep650 "
                "witness, n600 through exact R + frozen CPU SegNet",
        "pre_registration": ".omx/research/freq_along_ladder_oracle_probe_20260707.md",
        "form": "ORACLE-CAPACITY (form b); form a (re-render frozen field at changed "
                "freq_along) declared UNSUPPORTED: dir_feats keeps the feature DIMENSION but "
                "swaps basis functions under weights calibrated to the trained fa",
        "config_discovery": {
            "ckpt_freq_along_base": m["so_freq_along"],
            "ckpt_n_dir_freqs": m["n_dir_freqs"],
            "ckpt_along_ladder_cyc_per_unit": [m["so_freq_along"] * 2 ** k
                                               for k in range(m["n_dir_freqs"])],
            "note": "the as-built ep650 basis is a DYADIC along ladder reaching 64, not 8; "
                    "the FEED-08f '8 = sqrt(64)' arithmetic keyed on the base config value "
                    "(true of the ep200 vehicle: n_dir_freqs=2, fa=4 -> {4,8})",
        },
        "authority": "probe-render instrument ONLY (dcp.Renderer + _torch_R_to_camera_uint8 "
                     "+ seg_argmax_batch, frozen CPU torch); trainer-verdict numbers never "
                     "mixed (the two authorities disagree ~5% state-dependently per the "
                     "4db610af2 review)",
        "n_pairs": P,
        "u_px_per_unit": U_PX_PER_UNIT,
        "rungs_cyc_per_unit": [f for _, f in RUNGS],
        "table": table,
        "band_scoreable": band_scoreable,
        "band_gate_reason": band_gate_reason,
        "closure_C_b_f": closure,
        "closure_definition": "contrast = r_mark - r_gap; C_b(f) = (contrast(c_f)-"
                              "contrast(cSOLID)) / (contrast(cCOMB)-contrast(cSOLID)); "
                              "None = band failed the pre-registered dynamic-range gate "
                              "(den >= 0.10 AND contrast(ctrl_GT) >= contrast(cSOLID)+0.05; "
                              "GT-condition amended pre-measurement, see module docstring)",
        "H_definition_secondary": "H = r_gap/r_mark -- DEMOTED per the 4db610af2 review "
                                  "(zero dynamic range: GT labels read 0.7015); reported "
                                  "for FEED-08e comparability only",
        "forward_bands_m": [[lo, hi] for lo, hi in dcp.FWD_BANDS],
        "R_path": "torch bicubic^ 874x1164 -> round/clamp/uint8 -> SegNet.preprocess_input "
                  "(contest bilinear) -> argmax vs GT lstars (frozen CPU-torch; never MPS)",
        "peak_rss_gib": dcp._peak_rss_gib(),
        "note": "means not ends: advisory row; pointer 0.19110 moves only via "
                "upstream/evaluate.py on exact archive bytes",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE -> {OUT}", flush=True)
    print(f"  band_scoreable={band_scoreable}", flush=True)
    for r in table:
        print(f"  {r['cond']:11s} dseg={r['d_seg']:.5f} rec={r['lane_recall']:.4f} "
              f"gapfp={r['dash_gap_fp']:.6f} contrast={r['contrast']:.4f} "
              f"cb={['%.3f' % x if np.isfinite(x) else 'nan' for x in r['contrast_by_band']]}",
              flush=True)


if __name__ == "__main__":
    main()
