#!/usr/bin/env python
"""$0 DASH-COMB PROBES (#287) on the FROZEN mod32cap EMA-best checkpoint — n600.

[macOS-CPU advisory] NON-PROMOTABLE. The two owed probes of the dash-erasure
homogenization law (``tac.canonical_equations.dash_erasure_homogenization_20260707``
reactivation criteria; hunt memo §9.1):

PROBE 1 (corrector A/B, render-time so NO training): render the frozen witness and
composite the analytic lane band under four gates, through the EXACT contest R
(bicubic^ to 874x1164 -> round/clamp/uint8 -> SegNet contest bilinear -> argmax),
ALL 600 pairs, vs GT lstars:

  c1_witness      -- witness alone (baseline; must reproduce the trained ~0.0034).
  c2_band_solid   -- witness + SOLID band (dash_gate OFF = the HOMOGENIZED band).
  c3_band_comb    -- witness + band x EGO-PHASE COMB (tac.boundary_math.dash_comb:
                     global period/duty/ego-scale + per-slot world phase from xi).
  c4_band_fitgate -- witness + band x PER-PAIR FITTED range-dependent dash gate
                     (the existing #215 gate; the byte-expensive phase reference).

  All band conditions composite WITHOUT the witness-uncertainty mask (isolating the
  dash-gate contrast; the u-mask is a separate, separately-measured lever).

PROBE 2 (tau/smoothing crossover, pre-registered): blur the c3 comb-gated band ALPHA
with a Gaussian of sigma_px in {1,2,4,8} before compositing (the render-side analog
of the training flow's smoothing scale) and measure dash-gap FP vs sigma/delta_along
per range band. PREDICTION (homogenization): as sigma/delta_along crosses ~O(1) the
gap FP rises from the combed level toward the solid-band (homogenized) level.
HONEST NOTE: this sweeps the render-side smoothing of the CORRECTOR, not the training
flow's tau; it measures the crossover SHAPE of the R+SegNet leg.

MEASURES per pair per condition: d_seg, lane recall, lane FN/FP, dash-gap FP
(realized lane where GT!=lane inside the SOLID band footprint but in a comb GAP,
near field), and FP/FN per forward-range band.

Resumable per-pair (atomic tmp+replace state); chunked foreground (exits cleanly
after --chunk-seconds; kills become progress). Free-RAM >= 20 GiB gate at start;
verdict inference chunked at VBATCH pairs (never a full-P batched scorer forward).
NO MPS. numpy/CPU authority. Pointer 0.19110 UNMOVED (this is a means).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

from tac.local_acceleration import torch_levelset_inflate as tli  # noqa: E402
from tac.local_acceleration.torch_levelset_inflate import _torch_act  # noqa: E402
from tac.boundary_math.seg_core import load_real_segnet  # noqa: E402
from tac.boundary_math.lane_sdf_component import (  # noqa: E402
    _CAM_H,
    _FY,
    _V_HORIZON,
    LaneLine,
)
from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    rasterize_lane_coverage_range_dependent,
)
from tac.boundary_math.dash_comb import (  # noqa: E402
    DashCombFit,
    ego_cumulative_distance,
    fit_dash_comb,
    fit_lines_per_pair,
    rasterize_lane_coverage_combed,
)
from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _torch_R_to_camera_uint8,
)

RUN_DIR = REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"
SRC_CKPT = RUN_DIR / "levelset_witness_ema_BEST.npz"
CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_DIR = REPO / "experiments/results/dash_comb_probe_20260707"
CKPT_NPZ = OUT_DIR / "frozen_ckpt_ema_BEST.npz"       # read-only snapshot (live run safety)
LINES_JSON = OUT_DIR / "lines_and_comb_fit.json"
STATE = OUT_DIR / "probe_state.ckpt.npz"
OUT = OUT_DIR / "dash_comb_probe_n600_20260707.json"

VBATCH = int(os.environ.get("DASHCOMB_VBATCH", "6"))
SAVE_EVERY = int(os.environ.get("DASHCOMB_SAVE_EVERY", "24"))
SIGMAS = (1.0, 2.0, 4.0, 8.0)
CONDS = ["c1_witness", "c2_band_solid", "c3_band_comb", "c4_band_fitgate"] + [
    f"c3_blur_s{int(s)}" for s in SIGMAS
]
FWD_BANDS = [(4.0, 10.0), (10.0, 20.0), (20.0, 35.0), (35.0, 55.0), (55.0, 1e9)]
DASH_FORWARD_MAX_M = 55.0
COMB_SOFTNESS_M = 0.3
BAND_SOFTNESS_PX = 1.0
MIN_FREE_GIB = 20.0


def _free_gib() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return float("inf")


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def load_ckpt():
    z = np.load(CKPT_NPZ, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    code = params.pop("code")
    m = dict(
        render_h=384, render_w=512, n_pairs=code.shape[0] // 2,
        bank_n_scales=int(cfg["__bank_n_scales"]), bank_n_orient0=int(cfg["__bank_n_orient0"]),
        bank_f0=float(cfg["__bank_f0"]), bank_base=float(cfg["__bank_base"]),
        bank_n_iso=int(cfg["__bank_n_iso"]), max_bank_freq=float(cfg["__cfg_max_bank_freq"]),
        self_orient=bool(int(cfg["__cfg_self_orient"])), n_dir_freqs=int(cfg["__cfg_n_dir_freqs"]),
        so_iters=4, so_freq_along=float(cfg["__cfg_freq_along"]),
        so_freq_across=float(cfg["__cfg_freq_across"]), so_tau=4.0,
        activation=str(cfg["__cfg_activation"]), wire_w0=float(cfg["__cfg_wire_w0"]),
        wire_s0=float(cfg["__cfg_wire_s0"]), hosc_beta=float(cfg["__cfg_hosc_beta"]),
        hosc_omega=float(cfg["__cfg_hosc_omega"]), n_hidden=int(cfg["__cfg_n_hidden"]),
        hidden_dim=int(cfg["__cfg_hidden_dim"]), softmax_temp=float(cfg["__cfg_softmax_temp"]),
        chroma=bool(int(cfg["__cfg_chroma"])),
    )
    return params, code, m, cfg


class Renderer:
    """Frozen-witness renderer (op-for-op the canonical torch inflate primitives;
    mirrors tools/levelset_render_side_sizing_l7best_n600.py::Renderer, no supersample)."""

    def __init__(self, params, code, m):
        self.m = m
        self.rh, self.rw = m["render_h"], m["render_w"]
        self.nH, self.hd = m["n_hidden"], m["hidden_dim"]
        self.td = torch.float32
        self.P = {k: torch.as_tensor(v, dtype=self.td) for k, v in params.items()}
        self.code = torch.as_tensor(code, dtype=self.td)
        self.pal1 = self.P["palette"][1]
        self.B = tli.curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"],
                                m["bank_base"], m["bank_n_iso"], m["max_bank_freq"])
        self.coords_n = tli.coords_grid(self.rh, self.rw)
        self.curv_n = tli.curvelet_feats(self.coords_n, self.B)
        self.kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])

    def _self_orient_native(self, pi):
        m = self.m
        dirf = np.zeros((self.curv_n.shape[0], 4 * m["n_dir_freqs"]), np.float32)
        prev = None
        am = None
        for _ in range(m["so_iters"]):
            feats = np.concatenate([self.curv_n, dirf], axis=-1)
            ft = torch.as_tensor(feats, dtype=self.td)
            h0 = tli.torch_in_proj_h0(self.P, ft, m)
            phi, _ = tli.torch_outputs_from_h0(self.P, h0, self.code[2 * pi + 1], m, False)
            am = phi.argmax(-1).reshape(self.rh, self.rw).cpu().numpy().astype(np.int64)
            if prev is not None and np.array_equal(am, prev):
                break
            dirf = tli.dir_feats(self.coords_n, am, m["n_dir_freqs"], m["so_freq_along"],
                                 m["so_freq_across"], m["so_tau"])
            prev = am
        return np.concatenate([self.curv_n, dirf], axis=-1)

    def render_pair(self, pi):
        """-> (rgb_bulk (h,w,3) float, rgb_lane (h,w,3) float) at the render grid."""
        m = self.m
        feats_np = self._self_orient_native(pi) if m["self_orient"] else self.curv_n
        code_row = self.code[2 * pi + 1]
        ft = torch.as_tensor(feats_np, dtype=self.td)
        h0 = tli.torch_in_proj_h0(self.P, ft, m)
        film = (code_row @ self.P["film.weight"].T + self.P["film.bias"]).reshape(self.nH, 2, self.hd)
        hh = h0
        for li in range(self.nH):
            hh = _torch_act((hh @ self.P["hidden.%d.weight" % li].T + self.P["hidden.%d.bias" % li])
                            * (1.0 + film[li, 0]) + film[li, 1], *self.kw)
        phi = hh @ self.P["out_sdf.weight"].T + self.P["out_sdf.bias"]
        tex = hh @ self.P["out_tex.weight"].T + self.P["out_tex.bias"]
        z = phi / float(m["softmax_temp"])
        z = z - z.max(-1, keepdim=True).values
        soft = torch.exp(z)
        soft = soft / soft.sum(-1, keepdim=True)
        rgb_bulk = torch.sigmoid(soft @ self.P["palette"] + tex) * 255.0
        rgb_lane = torch.sigmoid(self.pal1[None, :] + tex) * 255.0
        if not m["chroma"]:
            def _lum(r):
                lu = 0.299 * r[:, 0:1] + 0.587 * r[:, 1:2] + 0.114 * r[:, 2:3]
                return torch.cat([lu, lu, lu], dim=-1)
            rgb_bulk, rgb_lane = _lum(rgb_bulk), _lum(rgb_lane)
        return (rgb_bulk.reshape(self.rh, self.rw, 3).numpy(),
                rgb_lane.reshape(self.rh, self.rw, 3).numpy())


def _gauss_blur_hw(a_hw: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an (H,W) map (torch conv; reflect padding)."""
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    kt = torch.as_tensor(k, dtype=torch.float32)
    t = torch.as_tensor(a_hw, dtype=torch.float32)[None, None]
    t = torch.nn.functional.pad(t, (r, r, 0, 0), mode="reflect")
    t = torch.nn.functional.conv2d(t, kt.reshape(1, 1, 1, -1))
    t = torch.nn.functional.pad(t, (0, 0, r, r), mode="reflect")
    t = torch.nn.functional.conv2d(t, kt.reshape(1, 1, -1, 1))
    return t[0, 0].numpy()


def _lines_to_json(lines: list[LaneLine]) -> list[dict]:
    return [dict(c=[float(x) for x in ln.centerline_coeffs],
                 h=[float(x) for x in ln.halfwidth_coeffs],
                 T=float(ln.dash_period_m), ph=float(ln.dash_phase_m), du=float(ln.dash_duty),
                 fr=[float(ln.forward_range[0]), float(ln.forward_range[1])],
                 n=int(ln.n_pixels)) for ln in lines]


def _lines_from_json(rows: list[dict]) -> list[LaneLine]:
    return [LaneLine(centerline_coeffs=np.asarray(r["c"], np.float64),
                     halfwidth_coeffs=np.asarray(r["h"], np.float64),
                     dash_period_m=r["T"], dash_phase_m=r["ph"], dash_duty=r["du"],
                     forward_range=(r["fr"][0], r["fr"][1]), n_pixels=r["n"]) for r in rows]


def _stage0_lines_and_fit(lst_all, gt_poses, P):
    """Fit lane lines per pair + the global comb; cache to JSON (deterministic)."""
    _INT_KEYED = ("phase0_by_slot", "period_by_slot", "duty_by_slot", "transported_by_slot",
                  "anchor_phase0_by_slot", "concentration_by_slot",
                  "anchored_concentration_by_slot", "pairwise_concentration_by_slot")
    if LINES_JSON.exists():
        d = json.loads(LINES_JSON.read_text())
        per_pair = [_lines_from_json(rows) for rows in d["per_pair_lines"]]
        f = dict(d["fit"])
        for key in _INT_KEYED:
            f[key] = {int(k): v for k, v in f[key].items()}
        fit = DashCombFit(**f)
        return per_pair, fit
    t0 = time.time()
    per_pair = fit_lines_per_pair(np.asarray(lst_all[:P]))
    fit = fit_dash_comb(per_pair, np.asarray(gt_poses[:P, 0], np.float64))
    LINES_JSON.write_text(json.dumps({
        "per_pair_lines": [_lines_to_json(ls) for ls in per_pair],
        "fit": {"scale": fit.scale, "period_m": fit.period_m, "duty": fit.duty,
                "phase0_by_slot": fit.phase0_by_slot,
                "period_by_slot": fit.period_by_slot,
                "duty_by_slot": fit.duty_by_slot,
                "transported_by_slot": fit.transported_by_slot,
                "anchor_window": fit.anchor_window,
                "anchor_phase0_by_slot": fit.anchor_phase0_by_slot,
                "concentration_by_slot": fit.concentration_by_slot,
                "anchored_concentration_by_slot": fit.anchored_concentration_by_slot,
                "pairwise_concentration_by_slot": fit.pairwise_concentration_by_slot,
                "mean_concentration": fit.mean_concentration,
                "mean_pairwise_concentration": fit.mean_pairwise_concentration,
                "concentration_at_zero_scale": fit.concentration_at_zero_scale,
                "n_dashed_fits": fit.n_dashed_fits, "n_pairs": fit.n_pairs,
                "slot_width_m": fit.slot_width_m, "provenance": fit.provenance},
    }))
    print(f"[stage0 {time.time() - t0:.1f}s] comb fit: scale={fit.scale:.5f} "
          f"(mean|ds|={fit.provenance['mean_abs_ds_m_at_best_scale']:.2f} m/pair) "
          f"transported_slots={[s for s, t in sorted(fit.transported_by_slot.items()) if t]} "
          f"anchored={fit.provenance['anchored_score_best']:.3f} null={fit.concentration_at_zero_scale:.3f} "
          f"pairwise={fit.mean_pairwise_concentration:.3f} n_dashed={fit.n_dashed_fits} "
          f"anchor_floats={fit.n_anchor_floats()}", flush=True)
    return per_pair, fit


def seg_argmax_batch(seg, frames_cam_uint8):
    arr = np.stack([np.asarray(f)[None] for f in frames_cam_uint8], axis=0)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        return seg(seg.preprocess_input(xp)).argmax(dim=1).cpu().numpy().astype(np.int64)


def _forward_of_rows(H):
    rows = np.arange(H, dtype=np.float64)
    return np.where(rows > _V_HORIZON + 1.0,
                    _CAM_H * _FY / np.maximum(rows - _V_HORIZON, 1e-3), np.inf)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-seconds", type=float, default=520.0,
                    help="exit cleanly (state saved) after this many seconds; re-invoke to resume.")
    ap.add_argument("--num-pairs", type=int, default=600, help="n600 discipline: keep 600.")
    args = ap.parse_args()

    free = _free_gib()
    if free < MIN_FREE_GIB:
        print(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB} GiB (live #205 run protection)", flush=True)
        sys.exit(3)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_NPZ.exists():
        shutil.copy2(SRC_CKPT, CKPT_NPZ)  # read-only snapshot of the live run's EMA-best
        print(f"snapshotted {SRC_CKPT.name} -> {CKPT_NPZ}", flush=True)

    params, code, m, cfg = load_ckpt()
    R = Renderer(params, code, m)
    seg = load_real_segnet("cpu")
    z = np.load(CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    gt_poses = np.asarray(z["gt_poses"], np.float64)
    P = min(int(args.num_pairs), int(z["n_pairs"]))
    per_pair_lines, fit = _stage0_lines_and_fit(lst_all, gt_poses, P)
    E = ego_cumulative_distance(gt_poses[:P, 0], fit.scale)
    H, W = m["render_h"], m["render_w"]
    fwd_rows = _forward_of_rows(H)
    band_row_idx = np.full(H, -1, np.int64)
    for bi, (lo, hi) in enumerate(FWD_BANDS):
        band_row_idx[(fwd_rows >= lo) & (fwd_rows < hi)] = bi
    NB = len(FWD_BANDS)
    print(f"[{time.time() - t0:.1f}s] ckpt ep{cfg.get('__epoch', -1)} + segnet + cache loaded; "
          f"P={P} conds={len(CONDS)}", flush=True)

    metrics = ["dseg", "rec", "fn", "fp", "gapfp"]
    st = {f"{c}__{mt}": np.full(P, np.nan) for c in CONDS for mt in metrics}
    st.update({f"{c}__fpband": np.full((P, NB), np.nan) for c in CONDS})
    st.update({f"{c}__fnband": np.full((P, NB), np.nan) for c in CONDS})
    done = np.zeros(P, bool)
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

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s0 in range(0, len(todo), VBATCH):
        chunk = todo[s0:s0 + VBATCH]
        lst = [np.asarray(lst_all[i], np.int64) for i in chunk]
        frames = {c: [] for c in CONDS}
        aux = []  # (solid_cov, comb_gate_hw) per pair for the dash-gap FP measure
        for j, pi in enumerate(chunk):
            bulk, lane = R.render_pair(pi)
            lines = per_pair_lines[pi]
            cov_solid = rasterize_lane_coverage_range_dependent(
                lines, h=H, w=W, softness=BAND_SOFTNESS_PX, dash_gate=False)
            cov_fit = rasterize_lane_coverage_range_dependent(
                lines, h=H, w=W, softness=BAND_SOFTNESS_PX, dash_gate=True,
                dash_forward_max_m=DASH_FORWARD_MAX_M)
            cov_comb = rasterize_lane_coverage_combed(
                lines, fit, float(E[pi]), pair_idx=pi, h=H, w=W, softness=BAND_SOFTNESS_PX,
                dash_forward_max_m=DASH_FORWARD_MAX_M, comb_softness_m=COMB_SOFTNESS_M)
            # dash-gap region (observability only, does not enter any composite):
            # inside the SOLID band footprint but comb-gated OFF (per-line, per-slot).
            gap_region = (cov_solid >= 0.5) & (cov_comb < 0.5)
            aux.append((cov_solid, gap_region))

            def comp(cov):
                a = cov[..., None].astype(np.float32)
                return _torch_R_to_camera_uint8(((1 - a) * bulk + a * lane).astype(np.float64))

            frames["c1_witness"].append(_torch_R_to_camera_uint8(bulk.astype(np.float64)))
            frames["c2_band_solid"].append(comp(cov_solid))
            frames["c3_band_comb"].append(comp(cov_comb))
            frames["c4_band_fitgate"].append(comp(cov_fit))
            for sg in SIGMAS:
                frames[f"c3_blur_s{int(sg)}"].append(comp(_gauss_blur_hw(cov_comb, sg)))

        for c in CONDS:
            realized = seg_argmax_batch(seg, frames[c])
            for j, pi in enumerate(chunk):
                gt_l = lst[j]
                r = realized[j]
                n = float(gt_l.size)
                is_lane = gt_l == 1
                nlane = int(is_lane.sum())
                fp_px = (~is_lane) & (r == 1)
                fn_px = is_lane & (r != 1)
                st[f"{c}__dseg"][pi] = float(np.count_nonzero(r != gt_l)) / n
                st[f"{c}__rec"][pi] = (float(np.count_nonzero(r[is_lane] == 1)) / nlane) if nlane else np.nan
                st[f"{c}__fn"][pi] = float(fn_px.sum()) / n
                st[f"{c}__fp"][pi] = float(fp_px.sum()) / n
                cov_solid, gap_region = aux[j]
                st[f"{c}__gapfp"][pi] = float((fp_px & gap_region).sum()) / n
                for bi in range(NB):
                    bm = band_row_idx == bi
                    st[f"{c}__fpband"][pi, bi] = float(fp_px[bm].sum()) / n
                    st[f"{c}__fnband"][pi, bi] = float(fn_px[bm].sum()) / n
        for pi in chunk:
            done[pi] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | "
                  f"c1={np.nanmean(st['c1_witness__dseg']):.5f} "
                  f"solid={np.nanmean(st['c2_band_solid__dseg']):.5f} "
                  f"comb={np.nanmean(st['c3_band_comb__dseg']):.5f} "
                  f"fit={np.nanmean(st['c4_band_fitgate__dseg']):.5f} "
                  f"| rss={_peak_rss_gib():.1f}GiB", flush=True)
        if time.time() - t0 > float(args.chunk_seconds) and nd < P:
            save()
            print(f"[chunk-exit] {nd}/{P} done at {time.time() - t0:.1f}s; re-invoke to resume. "
                  f"peak_rss={_peak_rss_gib():.2f}GiB", flush=True)
            sys.exit(0)
    save()

    n = int(done.sum())

    def stat(a):
        a = np.asarray(a, np.float64)
        a = a[~np.isnan(a)]
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "n": int(a.size)}

    c1m = float(np.nanmean(st["c1_witness__dseg"]))
    # delta_along in image rows per range band (for the probe-2 crossover axis):
    # a dash period of T meters spans ~ T*(v - v_h)^2 / (cam_h*fy) image rows at row v.
    # T taken from the DOMINANT (max-count) fitted slot (the ego-lane line family).
    slot_counts = {int(k): v for k, v in fit.provenance.get("slot_counts", {}).items()}
    dom_slot = max(fit.period_by_slot, key=lambda s: slot_counts.get(s, 0))
    T_dom = float(fit.period_by_slot[dom_slot])
    band_delta_px = {}
    for bi, (lo, hi) in enumerate(FWD_BANDS):
        bm = band_row_idx == bi
        if bm.any():
            v_mid = float(np.mean(np.arange(H)[bm]))
            band_delta_px[f"band{bi}_{lo:g}-{hi:g}m"] = float(
                T_dom * (v_mid - _V_HORIZON) ** 2 / (_CAM_H * _FY))
    band_delta_px["dominant_slot"] = dom_slot
    band_delta_px["dominant_slot_period_m"] = T_dom
    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "task": "#287 dash-comb corrector A/B (probe 1) + tau/smoothing crossover (probe 2) on the "
                "FROZEN mod32cap EMA-best checkpoint; the two owed anchors of "
                "dash_erasure_homogenization_v1",
        "n_pairs": n,
        "checkpoint": str(SRC_CKPT.relative_to(REPO)),
        "checkpoint_epoch": int(cfg.get("__epoch", -1)),
        "comb_fit": json.loads(LINES_JSON.read_text())["fit"],
        "conditions": {
            "c1_witness": "witness alone (baseline)",
            "c2_band_solid": "witness + SOLID analytic band (the homogenized band; no dash gate)",
            "c3_band_comb": f"witness + band x EGO-PHASE comb (softness {COMB_SOFTNESS_M} m)",
            "c4_band_fitgate": "witness + band x per-pair FITTED range-dependent dash gate (#215)",
            "c3_blur_s*": "probe 2: c3 alpha Gaussian-blurred at sigma_px before compositing",
            "u_mask": "OFF for all band conditions (isolates the dash-gate contrast)",
        },
        "d_seg": {c: stat(st[f"{c}__dseg"]) for c in CONDS},
        "delta_dseg_vs_witness": {c: float(np.nanmean(st[f"{c}__dseg"]) - c1m) for c in CONDS},
        "lane_recall": {c: float(np.nanmean(st[f"{c}__rec"])) for c in CONDS},
        "lane_fn_frac": {c: float(np.nanmean(st[f"{c}__fn"])) for c in CONDS},
        "lane_fp_frac": {c: float(np.nanmean(st[f"{c}__fp"])) for c in CONDS},
        "dash_gap_fp_frac": {c: float(np.nanmean(st[f"{c}__gapfp"])) for c in CONDS},
        "fp_by_forward_band": {c: [float(x) for x in np.nanmean(st[f"{c}__fpband"], axis=0)]
                               for c in CONDS},
        "fn_by_forward_band": {c: [float(x) for x in np.nanmean(st[f"{c}__fnband"], axis=0)]
                               for c in CONDS},
        "forward_bands_m": [[lo, hi] for lo, hi in FWD_BANDS],
        "probe2_delta_along_px_per_band": band_delta_px,
        "probe2_sigmas_px": list(SIGMAS),
        "R_path": "torch bicubic^ 874x1164 -> round/clamp/uint8 -> SegNet.preprocess_input "
                  "(contest bilinear) -> argmax vs GT lstars (frozen CPU-torch; never MPS)",
        "peak_rss_gib": _peak_rss_gib(),
        "note": "means not ends: advisory row; pointer 0.19110 moves only via upstream/evaluate.py "
                "on exact archive bytes",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE n={n} -> {OUT}", flush=True)
    for c in CONDS:
        print(f"  {c:16s} d_seg={np.nanmean(st[f'{c}__dseg']):.5f} "
              f"d(vs c1)={np.nanmean(st[f'{c}__dseg']) - c1m:+.5f} "
              f"rec={np.nanmean(st[f'{c}__rec']):.4f} fp={np.nanmean(st[f'{c}__fp']):.5f} "
              f"gapfp={np.nanmean(st[f'{c}__gapfp']):.5f}", flush=True)


if __name__ == "__main__":
    main()
