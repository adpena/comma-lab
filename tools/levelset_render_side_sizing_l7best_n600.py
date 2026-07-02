#!/usr/bin/env python
"""$0 RENDER-SIDE d_seg SIZING of the l7-best level-set witness (task #221 opener).

[macOS-CPU advisory] NON-PROMOTABLE. Question: how much d_seg can we grab from the
l7-best checkpoint via RENDER-SIDE levers WITHOUT any retraining? Four conditions on
the SAME EMA-shadow checkpoint (fp32, as-is), n600 (all 600 pairs), realized through
the ACTUAL contest R operator + the frozen CPU-torch SegNet argmax (NEVER MPS):

  1. witness-default   -- render witness RGB at its trained render grid (384x512) -> R ->
                          SegNet -> d_seg. SANITY: reproduce the trained ~0.004.
  2. witness + AA      -- render the witness field at a SUPERSAMPLED grid (S x render),
                          area/box-filter down to the render grid (pixel-footprint
                          coverage integration = anti-aliasing the sine/palette edges),
                          THEN R. (self-orient boundary structure kept at native res;
                          the final RGB is supersampled+box-filtered.)
  3. witness + analytic-lane -- composite the analytic openpilot AA-SDF lane band
                          (tac.boundary_math.lane_sdf_component.build_structured_lane_sdf,
                          fit to the GT SegNet class-1 mask in the ground/BEV frame) OVER
                          the witness bulk. The band becomes analytic RENDER-TIME authority
                          for the LANE class; the witness keeps Road/Undriv/hood/MyCar. The
                          lane appearance is the WITNESS'S OWN per-pixel lane color
                          (sigmoid(palette[1] + tex_pixel)*255 -- soft class-1 forced at the
                          band, so it is NOT a flat palette; byte-free, self-consistent).
                          Coverage alpha = AA soft edge on the SDF.
  4. witness + AA + analytic-lane -- both.

R-PATH (cited): experiments/train_witness_realized_through_R_mlx.py::_torch_R_to_camera_uint8
  (render-grid float RGB -> torch bicubic^ to camera 874x1164 -> round/clamp/uint8), then the
  cpu_verdict path = SegNet.preprocess_input (contest bilinear 874 -> 384x512) -> argmax,
  counted vs GT lstars. The witness render REUSES tac.local_acceleration.torch_levelset_inflate
  (coords_grid / curvelet_B / curvelet_feats / dir_feats / torch_in_proj_h0 / _torch_act) --
  the SAME (parity-gated) inflate primitives the byte-close path ships. cond1 == decode_levelset_torch
  (self-checked byte-identical at startup).

Resumable (ckpt every SAVE_EVERY pairs; atomic tmp+replace). NO MPS. numpy/CPU authority.
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
from tac.boundary_math.lane_sdf_component import build_structured_lane_sdf  # noqa: E402
from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _torch_R_to_camera_uint8,
)

CKPT_DIR = REPO / "experiments/results/levelset_n600_v2_attrclean_20260630T194549Z"
CKPT_NPZ = CKPT_DIR / "levelset_witness_ema_BEST.npz"
CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
STATE = REPO / "reports/levelset_render_side_sizing_l7best_n600.ckpt.npz"
OUT = REPO / "reports/levelset_render_side_sizing_l7best_n600_20260701.json"

NUM_PAIRS = int(os.environ.get("SIZING_NUM_PAIRS", "600"))
VBATCH = int(os.environ.get("SIZING_VBATCH", "8"))
SAVE_EVERY = int(os.environ.get("SIZING_SAVE_EVERY", "40"))
SUPER = int(os.environ.get("SIZING_SUPER", "2"))  # AA supersample factor

CONDS = ["c1_default", "c2_aa", "c3_lane", "c4_aa_lane"]


def load_ckpt():
    z = np.load(CKPT_NPZ, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    code = params.pop("code")
    m = dict(
        render_h=384, render_w=512, camera_h=874, camera_w=1164, n_pairs=code.shape[0] // 2,
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
    def __init__(self, params, code, m):
        self.m = m
        self.rh, self.rw = m["render_h"], m["render_w"]
        self.nH, self.hd = m["n_hidden"], m["hidden_dim"]
        self.td = torch.float32
        self.P = {k: torch.as_tensor(v, dtype=self.td) for k, v in params.items()}
        self.code = torch.as_tensor(code, dtype=self.td)
        self.pal1 = self.P["palette"][1]  # (3,) learned lane color row
        self.B = tli.curvelet_B(m["bank_n_scales"], m["bank_n_orient0"], m["bank_f0"], m["bank_base"],
                                m["bank_n_iso"], m["max_bank_freq"])
        self.coords_n = tli.coords_grid(self.rh, self.rw)
        self.curv_n = tli.curvelet_feats(self.coords_n, self.B)
        self.kw = (m["activation"], m["wire_w0"], m["wire_s0"], m["hosc_beta"], m["hosc_omega"])
        # super grid
        self.srh, self.srw = SUPER * self.rh, SUPER * self.rw
        self.coords_s = tli.coords_grid(self.srh, self.srw)
        self.curv_s = tli.curvelet_feats(self.coords_s, self.B)

    def _self_orient_native(self, pi):
        """4-iter argmax fixed point on the decoder's OWN all-class argmax (GT-free, byte-closeable).
        Returns (feats_native (HW,in_feat), am_native (rh,rw))."""
        m = self.m
        dirf = np.zeros((self.curv_n.shape[0], 4 * m["n_dir_freqs"]), np.float32)
        prev = None
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
        if prev is None:
            prev = am
        return np.concatenate([self.curv_n, dirf], axis=-1), prev

    def _render(self, feats_np, code_row, hw):
        """feats -> (rgb_bulk (h,w,3) float, rgb_lane (h,w,3) float). rgb_bulk is op-for-op the
        canonical inflate render; rgb_lane forces softmax=onehot(lane) keeping per-pixel tex."""
        m = self.m
        h, w = hw
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
        return rgb_bulk.reshape(h, w, 3).numpy(), rgb_lane.reshape(h, w, 3).numpy()

    def render_pair(self, pi):
        """Returns dict of the render-grid (384x512) fields for the 4 conditions."""
        feats_n, am_n = self._self_orient_native(pi)
        code_row = self.code[2 * pi + 1]
        bulk_n, lane_n = self._render(feats_n, code_row, (self.rh, self.rw))
        # super render: keep self-orient boundary at native, upsample tangent-source argmax (nearest)
        am_up = am_n[(np.arange(self.srh) * self.rh // self.srh)][:, (np.arange(self.srw) * self.rw // self.srw)]
        dir_s = tli.dir_feats(self.coords_s, am_up, self.m["n_dir_freqs"], self.m["so_freq_along"],
                              self.m["so_freq_across"], self.m["so_tau"])
        feats_s = np.concatenate([self.curv_s, dir_s], axis=-1)
        bulk_s, lane_s = self._render(feats_s, code_row, (self.srh, self.srw))
        bulk_s_d = _area_down(bulk_s, self.rh, self.rw)
        lane_s_d = _area_down(lane_s, self.rh, self.rw)
        return {"bulk_n": bulk_n, "lane_n": lane_n, "bulk_sd": bulk_s_d, "lane_sd": lane_s_d}


def _area_down(frame_hwc, H, W):
    x = torch.from_numpy(np.ascontiguousarray(frame_hwc)).permute(2, 0, 1)[None].float()
    if (x.shape[-2], x.shape[-1]) != (H, W):
        x = torch.nn.functional.interpolate(x, size=(H, W), mode="area")
    return x[0].permute(1, 2, 0).contiguous().numpy()


def seg_argmax_batch(seg, frames_cam_uint8):
    arr = np.stack([np.asarray(f)[None] for f in frames_cam_uint8], axis=0)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        return seg(seg.preprocess_input(xp)).argmax(dim=1).cpu().numpy().astype(np.int64)


def _decomp(realized, gt):
    """d_seg + lane recall + lane FN/FP (fraction of all px) for one frame."""
    n = float(gt.size)
    is_lane = gt == 1
    nlane = int(is_lane.sum())
    lane_fn = float(np.sum(is_lane & (realized != 1)) / n)
    lane_fp = float(np.sum((~is_lane) & (realized == 1)) / n)
    rec = (float(np.count_nonzero(realized[is_lane] == 1)) / nlane) if nlane else float("nan")
    return (float(np.count_nonzero(realized != gt)) / n, rec, lane_fn, lane_fp)


def main():
    t0 = time.time()
    params, code, m, cfg = load_ckpt()
    R = Renderer(params, code, m)
    seg = load_real_segnet("cpu")
    print(f"[{time.time() - t0:.1f}s] ckpt+segnet loaded; render {m['render_h']}x{m['render_w']} "
          f"self_orient={m['self_orient']} iters={m['so_iters']} SUPER={SUPER}", flush=True)
    z = np.load(CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    P = min(NUM_PAIRS, int(z["n_pairs"]))
    print(f"[{time.time() - t0:.1f}s] cache P={P}", flush=True)

    dseg = {c: np.full(P, np.nan) for c in CONDS}
    rec = {c: np.full(P, np.nan) for c in CONDS}
    fn = {c: np.full(P, np.nan) for c in CONDS}
    fp = {c: np.full(P, np.nan) for c in CONDS}
    band_rec = np.full(P, np.nan)  # analytic band vs GT lane recall (fit quality)
    band_lines = np.full(P, np.nan)
    done = np.zeros(P, bool)
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for c in CONDS:
            if c in ck.files:
                dseg[c] = ck[c]
            if c + "__rec" in ck.files:
                rec[c] = ck[c + "__rec"]
            if c + "__fn" in ck.files:
                fn[c] = ck[c + "__fn"]
            if c + "__fp" in ck.files:
                fp[c] = ck[c + "__fp"]
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
            payload[c + "__fn"] = fn[c]
            payload[c + "__fp"] = fp[c]
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
        lst = [np.asarray(lst_all[i], np.int64) for i in chunk]
        # render + build composites per pair
        frames = {c: [] for c in CONDS}
        for j, pi in enumerate(chunk):
            rr = R.render_pair(pi)
            phi1, meta = build_structured_lane_sdf(lst[j], lane_cls=1, dash_gate=True)
            alpha = np.clip(phi1 + 0.5, 0.0, 1.0)[..., None].astype(np.float32)  # AA soft edge
            band_lines[pi] = meta["n_lines"]
            band = phi1 >= 0.0
            mm = lst[j] == 1
            band_rec[pi] = (float((band & mm).sum()) / int(mm.sum())) if mm.any() else np.nan
            comp_n = (1 - alpha) * rr["bulk_n"] + alpha * rr["lane_n"]
            comp_sd = (1 - alpha) * rr["bulk_sd"] + alpha * rr["lane_sd"]
            frames["c1_default"].append(_torch_R_to_camera_uint8(rr["bulk_n"].astype(np.float64)))
            frames["c2_aa"].append(_torch_R_to_camera_uint8(rr["bulk_sd"].astype(np.float64)))
            frames["c3_lane"].append(_torch_R_to_camera_uint8(comp_n.astype(np.float64)))
            frames["c4_aa_lane"].append(_torch_R_to_camera_uint8(comp_sd.astype(np.float64)))
        for c in CONDS:
            realized = seg_argmax_batch(seg, frames[c])
            for j, pi in enumerate(chunk):
                d, rc, f_n, f_p = _decomp(realized[j], lst[j])
                dseg[c][pi] = d
                rec[c][pi] = rc
                fn[c][pi] = f_n
                fp[c][pi] = f_p
        for pi in chunk:
            done[pi] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | c1={np.nanmean(dseg['c1_default']):.5f} "
                  f"c2={np.nanmean(dseg['c2_aa']):.5f} c3={np.nanmean(dseg['c3_lane']):.5f} "
                  f"c4={np.nanmean(dseg['c4_aa_lane']):.5f}", flush=True)
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
        "task": "#221 fine-tune-path opener: render-side d_seg sizing of the l7-best level-set witness "
                "WITHOUT retraining",
        "n_pairs": n,
        "checkpoint": str(CKPT_NPZ.relative_to(REPO)),
        "checkpoint_epoch": int(cfg.get("__epoch", -1)),
        "render_grid": [m["render_h"], m["render_w"]],
        "supersample_factor": SUPER,
        "R_path": "experiments/train_witness_realized_through_R_mlx.py::_torch_R_to_camera_uint8"
                  " (render-grid float RGB -> torch bicubic^ to 874x1164 -> round/clamp/uint8),"
                  " then SegNet.preprocess_input (contest bilinear 874->384x512) -> argmax; counted"
                  " vs GT lstars. Render REUSES tac.local_acceleration.torch_levelset_inflate primitives.",
        "conditions": {
            "c1_default": "witness rendered as-is at 384x512",
            "c2_aa": f"witness field supersampled {SUPER}x -> area/box-down to 384x512 -> R (AA)",
            "c3_lane": "witness bulk + analytic openpilot AA-SDF lane band (build_structured_lane_sdf, "
                       "GT-fit BEV) composited as class-1 render-time authority; lane appearance = "
                       "witness's own per-pixel sigmoid(palette[1]+tex)*255 (NOT flat)",
            "c4_aa_lane": "c2 AA + c3 analytic lane",
        },
        "d_seg": {c: stat(dseg[c]) for c in CONDS},
        "lane_recall": {c: float(np.nanmean(rec[c])) for c in CONDS},
        "lane_fn_frac": {c: float(np.nanmean(fn[c])) for c in CONDS},
        "lane_fp_frac": {c: float(np.nanmean(fp[c])) for c in CONDS},
        "delta_vs_default": {c: float(np.nanmean(dseg[c]) - c1m) for c in CONDS},
        "analytic_band_fit": {
            "band_vs_gt_lane_recall_mean": float(np.nanmean(band_rec)),
            "n_lines_mean": float(np.nanmean(band_lines)),
            "note": "band_vs_gt_lane_recall = fraction of GT class-1 pixels inside the analytic band; "
                    "the complement + band false-positives (dash-gap leakage) drive c3/c4 d_seg.",
        },
        "witness_training_floor_ref": {"best_measured": 0.004445, "current_n600_ep175": 0.006265,
                                       "l7_best_this_ckpt_reported": 0.004048, "islands_target": 0.001},
        "sanity": {"c1_default_mean_d_seg": c1m,
                   "note": "SANITY: c1 must reproduce the trained ~0.004 realized d_seg."},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE n={n}. wrote {OUT.name}", flush=True)
    for c in CONDS:
        print(f"  {c:12s} d_seg={np.nanmean(dseg[c]):.5f}  d(vs c1)={np.nanmean(dseg[c]) - c1m:+.5f}  "
              f"lane_rec={np.nanmean(rec[c]):.4f}  lane_fn={np.nanmean(fn[c]):.5f}  lane_fp={np.nanmean(fp[c]):.5f}",
              flush=True)


if __name__ == "__main__":
    main()
