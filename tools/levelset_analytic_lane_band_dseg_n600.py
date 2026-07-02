#!/usr/bin/env python
"""$0 n600 Delta d_seg of the NON-NAIVE analytic-lane render-band (FEED-dv, #203/#213/#215).

[macOS-CPU advisory] NON-PROMOTABLE. Question: does the NON-NAIVE analytic-lane band
(AA-SDF coverage x RANGE-DEPENDENT dash gate x WITNESS-UNCERTAINTY mask) NET-NEGATIVE
Delta d_seg vs the witness alone, where the NAIVE band HURT (+0.00082, sizing c3)?

Ablation ladder on the SAME l7-best levelset EMA-shadow ckpt (fp32), n600 (all 600
pairs), realized through the ACTUAL contest R operator + frozen CPU-torch SegNet argmax
(NEVER MPS -- numpy/CPU authority):

  c1_default   -- witness bulk alone. SANITY: reproduce the trained ~0.00333.
  c_naive      -- band, UNIFORM dash gate, NO uncertainty mask (= the naive form).
  c_range      -- band, RANGE-DEPENDENT dash gate (#215), NO uncertainty. Isolates #215.
  c_full_gt    -- band, range-dep + WITNESS-UNCERTAINTY from the FROZEN GT SegNet margin
                  (the boundary annulus; cache['margins']). The full non-naive form.
  c_full_wit   -- band, range-dep + uncertainty from the WITNESS'S OWN softmax margin.

Band prior: tac.boundary_math.analytic_lane_render_band.build_analytic_lane_band_prior
(reuses lane_sdf_component cluster/fit; range-dependent rasterizer). Uncertainty:
tac.boundary_math.analytic_lane_render_band.witness_uncertainty_mask (rides #141 margin).
Lane appearance = the witness's OWN per-pixel sigmoid(palette[1]+tex)*255 (self-consistent,
byte-free, apples-to-apples with sizing c3).

R-PATH (cited, IDENTICAL to sizing): experiments/train_witness_realized_through_R_mlx.py::
_torch_R_to_camera_uint8 -> SegNet.preprocess_input (contest bilinear) -> argmax vs GT lstars.
Render REUSES tac.local_acceleration.torch_levelset_inflate primitives (parity-gated).

Resumable (ckpt every SAVE_EVERY pairs; atomic tmp+replace). NO MPS.
"""
from __future__ import annotations

import json
import os
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
from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    build_analytic_lane_band_prior,
    witness_uncertainty_mask,
    decompose_lane_dseg,
)
from experiments.train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8  # noqa: E402

CKPT_DIR = REPO / "experiments/results/levelset_n600_v2_attrclean_20260630T194549Z"
CKPT_NPZ = CKPT_DIR / "levelset_witness_ema_BEST.npz"
CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
STATE = REPO / "reports/levelset_analytic_lane_band_dseg_n600.ckpt.npz"
OUT = REPO / "reports/levelset_analytic_lane_band_dseg_n600_20260701.json"

NUM_PAIRS = int(os.environ.get("BAND_NUM_PAIRS", "600"))
VBATCH = int(os.environ.get("BAND_VBATCH", "8"))
SAVE_EVERY = int(os.environ.get("BAND_SAVE_EVERY", "40"))
SOFTNESS = float(os.environ.get("BAND_SOFTNESS", "1.0"))
DASH_FWD_MAX = float(os.environ.get("BAND_DASH_FWD_MAX", "55.0"))
TAU = float(os.environ.get("BAND_TAU", "0.85"))     # witness margin (PROBABILITY [0,1]) threshold
EPS = float(os.environ.get("BAND_EPS", "0.35"))     # witness margin ramp
TAU_GT = float(os.environ.get("BAND_TAU_GT", "2.0"))  # GT SegNet margin (LOGIT ~[0,13]) threshold
EPS_GT = float(os.environ.get("BAND_EPS_GT", "1.5"))  # GT margin ramp

CONDS = ["c1_default", "c_naive", "c_range", "c_full_gt", "c_full_wit"]


def load_ckpt():
    z = np.load(CKPT_NPZ, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    code = params.pop("code")
    m = dict(
        render_h=384, render_w=512, n_pairs=code.shape[0] // 2,
        bank_n_scales=int(cfg["__bank_n_scales"]), bank_n_orient0=int(cfg["__bank_n_orient0"]),
        bank_f0=float(cfg["__bank_f0"]), bank_base=float(cfg["__bank_base"]), bank_n_iso=int(cfg["__bank_n_iso"]),
        max_bank_freq=float(cfg["__cfg_max_bank_freq"]),
        self_orient=bool(int(cfg["__cfg_self_orient"])), n_dir_freqs=int(cfg["__cfg_n_dir_freqs"]),
        so_iters=4, so_freq_along=float(cfg["__cfg_freq_along"]),
        so_freq_across=float(cfg["__cfg_freq_across"]), so_tau=4.0,
        activation=str(cfg["__cfg_activation"]), wire_w0=float(cfg["__cfg_wire_w0"]), wire_s0=float(cfg["__cfg_wire_s0"]),
        hosc_beta=float(cfg["__cfg_hosc_beta"]), hosc_omega=float(cfg["__cfg_hosc_omega"]),
        n_hidden=int(cfg["__cfg_n_hidden"]), hidden_dim=int(cfg["__cfg_hidden_dim"]),
        softmax_temp=float(cfg["__cfg_softmax_temp"]), chroma=bool(int(cfg["__cfg_chroma"])),
    )
    return params, code, m, cfg


class Renderer:
    """Lean witness renderer: returns bulk RGB, lane-appearance RGB, and the WITNESS
    softmax decision margin (top1-top2). Op-for-op the sizing tool's render, minus the
    supersample (AA is on the analytic band, not the witness)."""

    def __init__(self, params, code, m):
        self.m = m
        self.rh, self.rw = m["render_h"], m["render_w"]
        self.nH, self.hd = m["n_hidden"], m["hidden_dim"]
        self.td = torch.float32
        self.P = {k: torch.as_tensor(v, dtype=self.td) for k, v in params.items()}
        self.code = torch.as_tensor(code, dtype=self.td)
        self.pal1 = self.P["palette"][1]
        self.B = tli.curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"],
                                m["bank_n_iso"], m["max_bank_freq"])
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
        """Returns (bulk (h,w,3), lane (h,w,3), witness_margin (h,w))."""
        m = self.m
        feats_np = self._self_orient_native(pi)
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
        # witness decision margin (top1-top2 of soft)
        top2 = torch.topk(soft, 2, dim=-1).values
        wit_margin = (top2[:, 0] - top2[:, 1]).reshape(self.rh, self.rw).cpu().numpy().astype(np.float32)
        rgb_bulk = torch.sigmoid(soft @ self.P["palette"] + tex) * 255.0
        rgb_lane = torch.sigmoid(self.pal1[None, :] + tex) * 255.0
        if not m["chroma"]:
            def _lum(r):
                lu = 0.299 * r[:, 0:1] + 0.587 * r[:, 1:2] + 0.114 * r[:, 2:3]
                return torch.cat([lu, lu, lu], dim=-1)
            rgb_bulk, rgb_lane = _lum(rgb_bulk), _lum(rgb_lane)
        return (rgb_bulk.reshape(self.rh, self.rw, 3).numpy(),
                rgb_lane.reshape(self.rh, self.rw, 3).numpy(), wit_margin)


def _composite(bulk, lane, coverage, u_mask):
    a = coverage if u_mask is None else coverage * u_mask
    a = a[..., None]
    return bulk * (1.0 - a) + lane * a


def seg_argmax_batch(seg, frames_cam_uint8):
    arr = np.stack([np.asarray(f)[None] for f in frames_cam_uint8], axis=0)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        return seg(seg.preprocess_input(xp)).argmax(dim=1).cpu().numpy().astype(np.int64)


def main():
    t0 = time.time()
    params, code, m, cfg = load_ckpt()
    R = Renderer(params, code, m)
    seg = load_real_segnet("cpu")
    z = np.load(CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    mgn_all = z["margins"]  # FROZEN GT SegNet top1-top2 margin
    P = min(NUM_PAIRS, int(z["n_pairs"]))
    print(f"[{time.time() - t0:.1f}s] ckpt+segnet+cache loaded P={P} tau={TAU} eps={EPS} "
          f"soft={SOFTNESS} dash_fwd_max={DASH_FWD_MAX}", flush=True)

    dseg = {c: np.full(P, np.nan) for c in CONDS}
    rec = {c: np.full(P, np.nan) for c in CONDS}
    fnf = {c: np.full(P, np.nan) for c in CONDS}
    fpf = {c: np.full(P, np.nan) for c in CONDS}
    band_rec = np.full(P, np.nan)
    band_lines = np.full(P, np.nan)
    done = np.zeros(P, bool)
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for c in CONDS:
            if c in ck.files:
                dseg[c] = ck[c]
            for suf, tgt in (("__rec", rec), ("__fn", fnf), ("__fp", fpf)):
                if c + suf in ck.files:
                    tgt[c] = ck[c + suf]
        if "band_rec" in ck.files:
            band_rec = ck["band_rec"]
        if "band_lines" in ck.files:
            band_lines = ck["band_lines"]
        done = ck["done"].astype(bool)
        print(f"[resume] {int(done.sum())}/{P} done", flush=True)

    def save():
        payload = {}
        for c in CONDS:
            payload[c] = dseg[c]
            payload[c + "__rec"] = rec[c]
            payload[c + "__fn"] = fnf[c]
            payload[c + "__fp"] = fpf[c]
        payload["band_rec"] = band_rec
        payload["band_lines"] = band_lines
        payload["done"] = done
        tmp = STATE.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(STATE)

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s in range(0, len(todo), VBATCH):
        chunk = todo[s:s + VBATCH]
        frames = {c: [] for c in CONDS}
        lst_chunk = []
        for pi in chunk:
            lst = np.asarray(lst_all[pi], np.int64)
            gt_margin = np.asarray(mgn_all[pi], np.float32)
            lst_chunk.append(lst)
            bulk, lane, wit_margin = R.render_pair(pi)
            # --- band priors: uniform (naive) vs range-dependent ---
            prior_uni = build_analytic_lane_band_prior(
                lst, lane_cls=1, softness=SOFTNESS, dash_gate=True, dash_forward_max_m=1e9)
            prior_rng = build_analytic_lane_band_prior(
                lst, lane_cls=1, softness=SOFTNESS, dash_gate=True, dash_forward_max_m=DASH_FWD_MAX)
            band_lines[pi] = prior_rng.n_lines
            band_rec[pi] = prior_rng.band_recall
            u_gt = witness_uncertainty_mask(gt_margin, tau=TAU_GT, eps=EPS_GT)  # GT margin = LOGIT scale
            u_wit = witness_uncertainty_mask(wit_margin, tau=TAU, eps=EPS)      # witness margin = PROB scale
            comp_naive = _composite(bulk, lane, prior_uni.coverage, None)
            comp_range = _composite(bulk, lane, prior_rng.coverage, None)
            comp_full_gt = _composite(bulk, lane, prior_rng.coverage, u_gt)
            comp_full_wit = _composite(bulk, lane, prior_rng.coverage, u_wit)
            frames["c1_default"].append(_torch_R_to_camera_uint8(bulk.astype(np.float64)))
            frames["c_naive"].append(_torch_R_to_camera_uint8(comp_naive.astype(np.float64)))
            frames["c_range"].append(_torch_R_to_camera_uint8(comp_range.astype(np.float64)))
            frames["c_full_gt"].append(_torch_R_to_camera_uint8(comp_full_gt.astype(np.float64)))
            frames["c_full_wit"].append(_torch_R_to_camera_uint8(comp_full_wit.astype(np.float64)))
        for c in CONDS:
            realized = seg_argmax_batch(seg, frames[c])
            for j, pi in enumerate(chunk):
                d = decompose_lane_dseg(realized[j], lst_chunk[j], lane_cls=1)
                dseg[c][pi] = d.d_seg
                rec[c][pi] = d.lane_recall
                fnf[c][pi] = d.lane_fn_frac
                fpf[c][pi] = d.lane_fp_frac
        for pi in chunk:
            done[pi] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            msg = " ".join(f"{c.split('_')[-1] if c!='c1_default' else 'c1'}={np.nanmean(dseg[c]):.5f}" for c in CONDS)
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | {msg}", flush=True)
    save()

    n = int(done.sum())

    def stat(a):
        a = np.asarray(a)
        a = a[~np.isnan(a)]
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max()), "n": int(a.size)}

    c1m = float(np.nanmean(dseg["c1_default"]))
    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "task": "FEED-dv #203/#213/#215: NON-NAIVE analytic-lane render-band Delta d_seg (AA-SDF x "
                "range-dependent dash gate x witness-uncertainty mask) vs witness-alone, n600 through R.",
        "n_pairs": n,
        "checkpoint": str(CKPT_NPZ.relative_to(REPO)),
        "checkpoint_epoch": int(cfg.get("__epoch", -1)),
        "params": {"softness": SOFTNESS, "dash_forward_max_m": DASH_FWD_MAX,
                   "tau_witness_prob": TAU, "eps_witness": EPS, "tau_gt_logit": TAU_GT, "eps_gt": EPS_GT},
        "conditions": {
            "c1_default": "witness bulk alone (sanity ~0.00333)",
            "c_naive": "band, UNIFORM dash gate, NO uncertainty (naive)",
            "c_range": "band, RANGE-DEPENDENT dash gate (#215), NO uncertainty",
            "c_full_gt": "band, range-dep + uncertainty from FROZEN GT SegNet margin",
            "c_full_wit": "band, range-dep + uncertainty from WITNESS softmax margin",
        },
        "d_seg": {c: stat(dseg[c]) for c in CONDS},
        "delta_vs_default": {c: float(np.nanmean(dseg[c]) - c1m) for c in CONDS},
        "lane_recall": {c: float(np.nanmean(rec[c])) for c in CONDS},
        "lane_fn_frac": {c: float(np.nanmean(fnf[c])) for c in CONDS},
        "lane_fp_frac": {c: float(np.nanmean(fpf[c])) for c in CONDS},
        "analytic_band_fit": {
            "band_vs_gt_lane_recall_mean": float(np.nanmean(band_rec)),
            "n_lines_mean": float(np.nanmean(band_lines)),
        },
        "sizing_reference": {"c1_default": 0.0033304680718315976, "c3_naive_lane": 0.004149373372395833,
                             "c3_delta": 0.0008189053005642358, "note": "sizing c3 (naive band) HURT +0.00082"},
        "sanity": {"c1_default_mean_d_seg": c1m,
                   "note": "c1 must reproduce the sizing c1 ~0.00333 (same ckpt/render/R)."},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE n={n}. wrote {OUT.name}", flush=True)
    for c in CONDS:
        print(f"  {c:12s} d_seg={np.nanmean(dseg[c]):.5f}  d(vs c1)={np.nanmean(dseg[c]) - c1m:+.5f}  "
              f"rec={np.nanmean(rec[c]):.3f}  fn={np.nanmean(fnf[c]):.5f}  fp={np.nanmean(fpf[c]):.5f}", flush=True)


if __name__ == "__main__":
    main()
