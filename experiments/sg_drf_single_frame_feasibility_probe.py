# SPDX-License-Identifier: MIT
"""SG-DRF $0 single-frame FEASIBILITY PROBE (DAG FEED-bg gate).

Decide whether a deterministic rectified-flow backbone's ITERATIVE DEPTH beats a
matched-param coord-INR on ONE seg-frame through the EXACT contest R, ring-free.

Both arms are IDENTICAL except the backbone:
  * ARM A (SG-DRF): tiny conv velocity field v_theta(x,t|z) + per-frame latent
    (d_z=16, FiLM) integrated by an N-step deterministic Euler ODE from seeded
    noise; trained END-TO-END through R by the realized softargmax-CE; reflowed 1x.
  * ARM B (coord-INR control): the in-repo single-forward coord-MLP witness
    (``RGBWitnessMLX`` from the through-R trainer) at MATCHED params, same FiLM
    conditioning, SAME exact R, SAME loss.

R-FIDELITY (NO-FAKE): training gradient uses the trainer's CONTEST-EXACT MLX R
(``apply_contest_faithful_roundtrip_nhwc``: render -> bicubic up to camera 874x1164
-> uint8 STE @ camera -> bilinear down to 384x512). The d_seg VERDICT is recomputed
on the FROZEN CPU-torch SegNet via the trainer's torch authority path
(``_torch_R_to_camera_uint8`` + ``cpu_verdict_d_seg``), which calls the REAL
``SegNet.preprocess_input`` (modules.py:108-113) + argmax-disagreement
(compute_distortion, modules.py:112) == upstream/evaluate.py's d_seg. MLX is the
fast fp32 TRAINING-gradient device ONLY; CPU-torch is the verdict authority.
MPS is NEVER used. Evidence tag: [contest-CPU advisory] (single-frame, exact
scorer, not the 600-sample harness) -> promotion_eligible=False, NO score claim.

Disk: evidence -> experiments/results/<dated>/; NEVER /tmp.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the trainer's CONTEST-EXACT R + CPU-torch authority verdict (the whole
# point of R-fidelity: no re-implementation of the knife-edge R).
import train_witness_realized_through_R_mlx as twr  # noqa: E402
from train_witness_realized_through_R_mlx import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    SEG_H,
    SEG_W,
    MlxEMA,
    _render_rgb_render_res,
    _torch_R_to_camera_uint8,
    build_witness_module,
    cpu_verdict_d_seg,
)

N_CLASSES = 5
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{path!r} is a /tmp-class path; use experiments/results/ per CLAUDE.md.")


# ---------------------------------------------------------------------------
# GT: ONE real odd seg-frame + its frozen CPU-torch SegNet argmax L* (+ margin).
# ---------------------------------------------------------------------------
def load_one_frame(pair_idx: int):
    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    seg_cpu = load_real_segnet("cpu")
    f0 = f1 = None
    for idx, a, b in decode_gt_frame1_pairs(n_pairs=pair_idx + 1):
        if idx == pair_idx:
            f0, f1 = np.asarray(a), np.asarray(b)
            break
    if f1 is None:
        raise RuntimeError(f"pair_idx {pair_idx} not decoded")
    lstar, margin = segnet_argmax_and_margin(seg_cpu, f1)
    return seg_cpu, f0, f1, np.asarray(lstar).astype(np.int64), np.asarray(margin).astype(np.float32)


# ---------------------------------------------------------------------------
# Ring-free quantification: split the realized-vs-L* flip rate into the boundary
# annulus (localization error, expected) vs the INTERIOR (spurious Gibbs rings).
# ---------------------------------------------------------------------------
def boundary_band_mask(lstar: np.ndarray, radius: int = 2) -> np.ndarray:
    """True where a pixel is within ``radius`` of a class boundary in L*."""
    edge = np.zeros_like(lstar, dtype=bool)
    edge[:, :-1] |= lstar[:, :-1] != lstar[:, 1:]
    edge[:, 1:] |= lstar[:, :-1] != lstar[:, 1:]
    edge[:-1, :] |= lstar[:-1, :] != lstar[1:, :]
    edge[1:, :] |= lstar[:-1, :] != lstar[1:, :]
    band = edge.copy()
    for _ in range(radius - 1):
        nb = np.zeros_like(band)
        nb[:, :-1] |= band[:, 1:]
        nb[:, 1:] |= band[:, :-1]
        nb[:-1, :] |= band[1:, :]
        nb[1:, :] |= band[:-1, :]
        band |= nb
    return band


def flip_decomposition(realized: np.ndarray, lstar: np.ndarray, radius: int = 2) -> dict[str, float]:
    band = boundary_band_mask(lstar, radius=radius)
    flips = realized != lstar
    d_seg = float(flips.mean())
    in_band = float((flips & band).sum()) / max(1, int(band.sum()))
    interior = ~band
    out_band = float((flips & interior).sum()) / max(1, int(interior.sum()))
    return {
        "d_seg": d_seg,
        "flip_in_band": in_band,            # boundary-localization error (expected for both)
        "flip_out_band_interior": out_band,  # spurious interior flips == Gibbs ring signature
        "band_frac": float(band.mean()),
    }


# ---------------------------------------------------------------------------
# Shared seg-only realized loss (IDENTICAL across both arms). Through-R frame1
# logits from the frozen MLX SegNet -> margin-weighted CE vs L*.
# ---------------------------------------------------------------------------
def make_seg_ce(adapter, hinge: float):
    import mlx.core as mx

    def seg_ce(seg_logits, lstar_oh, margin):
        logsum = mx.logsumexp(seg_logits, axis=-1)            # (1,H,W)
        tgt = mx.sum(seg_logits * lstar_oh, axis=-1)          # (1,H,W)
        ce = logsum - tgt
        w = 1.0 + hinge * mx.exp(-mx.clip(margin, 0.0, 1e9))  # margin-weighted (small-margin px)
        return mx.mean(ce * w[None])

    return seg_ce


# ---------------------------------------------------------------------------
# ARM A: tiny conv velocity field v_theta(x,t|z) + deterministic Euler ODE.
# ---------------------------------------------------------------------------
def build_conv_vf(C: int, d_z: int, t_dim: int = 16):
    import mlx.core as mx
    import mlx.nn as nn

    class ConvVF(nn.Module):
        """v_theta(x,t|z): conv U-block velocity field. FiLM(t-embed||z) per block."""

        def __init__(self) -> None:
            super().__init__()
            self.C = C
            self.d_z = d_z
            self.t_dim = t_dim
            self.z = mx.zeros((d_z,))  # trainable per-frame latent (COUNTED in the real witness)
            self.head = nn.Conv2d(3, C, 3, padding=1)
            self.enc = nn.Conv2d(C, C, 3, padding=1)
            self.down = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)
            self.mid = nn.Conv2d(2 * C, 2 * C, 3, padding=1)
            self.up = nn.Conv2d(2 * C, C, 3, padding=1)
            self.dec = nn.Conv2d(C, C, 3, padding=1)
            self.tail = nn.Conv2d(C, 3, 3, padding=1)
            cond = t_dim + d_z
            self.film_enc = nn.Linear(cond, 2 * C)
            self.film_mid = nn.Linear(cond, 2 * 2 * C)
            self.film_dec = nn.Linear(cond, 2 * C)

        def _t_embed(self, t: float):
            k = mx.arange(self.t_dim // 2)
            freqs = mx.exp(-np.log(10000.0) * k / max(1, self.t_dim // 2))
            ang = t * freqs
            return mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1)  # (t_dim,)

        @staticmethod
        def _film(h, params):
            # h: (1,H,W,Ch); params: (2*Ch,) -> scale,shift broadcast over channels.
            ch = h.shape[-1]
            scale = params[:ch]
            shift = params[ch:]
            return h * (1.0 + scale) + shift

        def __call__(self, x, t: float):
            from tac.local_acceleration.pr95_hnerv_mlx_training import (
                resize_nhwc_align_corners_false,
            )

            c = mx.concatenate([self._t_embed(t), self.z], axis=-1)
            h = nn.relu(self.head(x))
            e = nn.relu(self._film(self.enc(h), self.film_enc(c)))         # (1,H,W,C)
            d = nn.relu(self._film(self.mid(self.down(e)), self.film_mid(c)))  # (1,H/2,W/2,2C)
            u = resize_nhwc_align_corners_false(d, size=(e.shape[1], e.shape[2]), mode="bilinear")
            u = nn.relu(self.up(u))
            u = u + e                                                       # skip
            u = nn.relu(self._film(self.dec(u), self.film_dec(c)))
            return self.tail(u)                                            # (1,H,W,3) velocity

    return ConvVF()


def ode_sample(vf, x0, n_steps: int):
    """Deterministic Euler ODE from seeded noise x0: x_{t+dt} = x_t + dt*v(x_t,t)."""
    x = x0
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t = i * dt
        x = x + dt * vf(x, t)
    return x


def count_params(model) -> int:
    from mlx.utils import tree_flatten

    return int(sum(np.asarray(v).size for _, v in tree_flatten(model.parameters())))


# ---------------------------------------------------------------------------
# Authority verdict helpers (FROZEN CPU-torch; identical R for both arms).
# ---------------------------------------------------------------------------
def verdict_from_render_np(rgb_render_np: np.ndarray, seg_cpu, lstar: np.ndarray) -> dict[str, float]:
    """render-res float RGB -> torch R (bicubic up to camera + uint8) -> real
    SegNet.preprocess_input + argmax -> realized d_seg vs L* (== evaluate.py)."""
    f1_cam = _torch_R_to_camera_uint8(rgb_render_np)
    import torch

    r = np.asarray(f1_cam)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = seg_cpu.preprocess_input(xp)
        realized = seg_cpu(seg_in).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return {"realized": realized, **flip_decomposition(realized, lstar)}


# ---------------------------------------------------------------------------
# ARM B: coord-INR control (single-forward RGBWitnessMLX), through-R CE overfit.
# ---------------------------------------------------------------------------
def run_arm_b(args, adapter, seg_cpu, lstar, margin, render_h, render_w) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    coords = mx.array(twr._build_render_coords(render_h, render_w))
    model = build_witness_module(
        num_pairs=1, n_fourier=args.b_n_fourier, hidden_dim=args.b_hidden,
        n_hidden=args.b_n_hidden, mod_dim=args.b_mod_dim, fourier_sigma=8.0,
    )
    mx.eval(model.parameters())
    n_params = count_params(model)
    feats = model.build_feats(coords)
    mx.eval(feats)
    lstar_oh = mx.array(np.eye(N_CLASSES, dtype=np.float32)[lstar][None])  # (1,H?,W?,5) at SEG res
    margin_mx = mx.array(margin)
    seg_ce = make_seg_ce(adapter, args.hinge)
    ema = MlxEMA(model, decay=0.997)

    def loss_fn(m):
        f1 = twr.render_through_R_mlx(m, feats, 1, render_h, render_w)  # code idx 1 = odd/frame1
        logits = adapter.segnet(f1)  # (1,384,512,5)
        return seg_ce(logits, lstar_oh, margin_mx)

    vag = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)
    best = {"d_seg": 1.0}
    t0 = time.time()
    for ep in range(1, args.b_epochs + 1):
        loss, grads = vag(model)
        grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        ema.update(model)
        if ep % args.eval_every == 0 or ep == args.b_epochs:
            saved = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
            model.update(ema.shadow_tree())
            mx.eval(model.parameters())
            rgb = _render_rgb_render_res(model, feats, 1, render_h, render_w)
            model.update(tree_unflatten(list(saved.items())))
            mx.eval(model.parameters())
            v = verdict_from_render_np(rgb, seg_cpu, lstar)
            print(json.dumps({"arm": "B", "ep": ep, "loss": round(float(loss), 4),
                              "d_seg": round(v["d_seg"], 6),
                              "interior_ring": round(v["flip_out_band_interior"], 6)}), flush=True)
            if v["d_seg"] < best["d_seg"]:
                best = {k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in v.items()}
                best["realized"] = v["realized"]
                best["epoch"] = ep
    best["n_params"] = n_params
    best["secs"] = round(time.time() - t0, 1)
    return best


# ---------------------------------------------------------------------------
# ARM A: SG-DRF conv flow, through-R CE overfit end-to-end + reflow 1x.
# ---------------------------------------------------------------------------
def run_arm_a(args, adapter, seg_cpu, lstar, margin, render_h, render_w, b_params: int) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    # Choose C so ARM A params <= ARM B params (NO param advantage for the flow).
    chosen_C = 8
    for C in range(8, 80):
        vf_try = build_conv_vf(C, args.d_z)
        mx.eval(vf_try.parameters())
        if count_params(vf_try) <= b_params:
            chosen_C = C
        else:
            break
    vf = build_conv_vf(chosen_C, args.d_z)
    mx.eval(vf.parameters())
    n_params = count_params(vf)

    # Seeded noise in data space (reproducible; reconstructible from a fixed seed -> FREE).
    mx.random.seed(args.seed + 7)
    x0 = mx.random.normal((1, render_h, render_w, 3)) * 40.0 + 128.0
    mx.eval(x0)

    lstar_oh = mx.array(np.eye(N_CLASSES, dtype=np.float32)[lstar][None])
    margin_mx = mx.array(margin)
    seg_ce = make_seg_ce(adapter, args.hinge)
    ema = MlxEMA(vf, decay=0.997)

    def loss_fn(m):
        x1 = ode_sample(m, x0, args.n_steps)
        from tac.local_acceleration.pr95_hnerv_mlx_training import (
            apply_contest_faithful_roundtrip_nhwc,
        )
        f1 = apply_contest_faithful_roundtrip_nhwc(x1, output_hw=(SEG_H, SEG_W), ste_round=True)
        logits = adapter.segnet(f1)
        return seg_ce(logits, lstar_oh, margin_mx)

    vag = nn.value_and_grad(vf, loss_fn)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=1e-4)

    from mlx.utils import tree_flatten, tree_unflatten

    def eval_vf(use_ema: bool) -> dict[str, float]:
        saved = {k: mx.array(v) for k, v in tree_flatten(vf.parameters())}
        if use_ema:
            vf.update(ema.shadow_tree())
            mx.eval(vf.parameters())
        x1 = ode_sample(vf, x0, args.n_steps)
        mx.eval(x1)
        rgb = np.asarray(x1[0], dtype=np.float32)
        vf.update(tree_unflatten(list(saved.items())))
        mx.eval(vf.parameters())
        return verdict_from_render_np(rgb, seg_cpu, lstar)

    best = {"d_seg": 1.0}
    t0 = time.time()
    for ep in range(1, args.a_epochs + 1):
        loss, grads = vag(vf)
        grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
        opt.update(vf, grads)
        mx.eval(vf.parameters(), opt.state)
        ema.update(vf)
        if ep % args.eval_every == 0 or ep == args.a_epochs:
            v = eval_vf(use_ema=True)
            print(json.dumps({"arm": "A", "ep": ep, "loss": round(float(loss), 4),
                              "C": chosen_C, "d_seg": round(v["d_seg"], 6),
                              "interior_ring": round(v["flip_out_band_interior"], 6)}), flush=True)
            if v["d_seg"] < best["d_seg"]:
                best = dict(v)
                best["epoch"] = ep
                best["phase"] = "pre_reflow"
    pre_reflow = dict(best)

    # --- Reflow 1x: straighten v_theta toward the (x0 -> x1) couplings ---
    if args.reflow_steps > 0:
        x1_fixed = mx.stop_gradient(ode_sample(vf, x0, args.n_steps))
        mx.eval(x1_fixed)
        vstar = x1_fixed - x0  # straight-line target velocity
        rng = np.random.default_rng(args.seed)

        def reflow_loss_t(m, tval):
            xt = (1.0 - tval) * x0 + tval * x1_fixed
            return mx.mean(mx.square(m(xt, tval) - vstar))

        rvag = nn.value_and_grad(vf, reflow_loss_t)
        ropt = optim.AdamW(learning_rate=args.lr * 0.5, weight_decay=1e-4)
        for rs in range(1, args.reflow_steps + 1):
            tval = float(rng.uniform())
            rloss, rgrads = rvag(vf, tval)
            rgrads, _ = optim.clip_grad_norm(rgrads, args.grad_clip)
            ropt.update(vf, rgrads)
            mx.eval(vf.parameters(), ropt.state)
            ema.update(vf)
        v_post = eval_vf(use_ema=True)
        print(json.dumps({"arm": "A", "phase": "post_reflow", "d_seg": round(v_post["d_seg"], 6),
                          "interior_ring": round(v_post["flip_out_band_interior"], 6)}), flush=True)
        if v_post["d_seg"] < best["d_seg"]:
            best = dict(v_post)
            best["phase"] = "post_reflow"

    best["n_params"] = n_params
    best["C"] = chosen_C
    best["n_steps"] = args.n_steps
    best["secs"] = round(time.time() - t0, 1)
    best["pre_reflow"] = {k: pre_reflow.get(k) for k in ("d_seg", "flip_in_band", "flip_out_band_interior", "epoch")}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="SG-DRF $0 single-frame feasibility probe")
    ap.add_argument("--out-dir", type=Path, default=REPO / "experiments" / "results" / f"sg_drf_probe_{_utc().replace(':', '').replace('-', '')}")
    ap.add_argument("--pair-idx", type=int, default=0)
    ap.add_argument("--render-h", type=int, default=256)
    ap.add_argument("--render-w", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hinge", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    # ARM A (SG-DRF)
    ap.add_argument("--a-epochs", type=int, default=250)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--d-z", type=int, default=16)
    ap.add_argument("--reflow-steps", type=int, default=60)
    # ARM B (coord-INR)
    ap.add_argument("--b-epochs", type=int, default=250)
    ap.add_argument("--b-n-fourier", type=int, default=64)
    ap.add_argument("--b-hidden", type=int, default=96)
    ap.add_argument("--b-n-hidden", type=int, default=4)
    ap.add_argument("--b-mod-dim", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end sanity (few epochs)")
    args = ap.parse_args()

    if args.smoke:
        args.a_epochs = 4
        args.b_epochs = 4
        args.reflow_steps = 3
        args.eval_every = 2

    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    print(json.dumps({"stage": "load_frame", "pair_idx": args.pair_idx, "utc": _utc()}), flush=True)
    seg_cpu, f0, f1, lstar, margin = load_one_frame(args.pair_idx)
    print(json.dumps({"stage": "gt", "seg_shape": list(lstar.shape),
                      "classes_present": np.unique(lstar).tolist()}), flush=True)

    # R-FIDELITY check: GT frame1 through the SAME torch authority R must give d_seg==0
    # (a witness identical to GT scores exactly 0 -> proves R reproduces evaluate.py).
    # Push GT camera-res frame straight into cpu_verdict_d_seg (real preprocess_input).
    d_seg_gt = cpu_verdict_d_seg(seg_cpu, f1.astype(np.uint8), lstar)
    print(json.dumps({"stage": "R_fidelity", "d_seg_GT_vs_Lstar": round(float(d_seg_gt), 8),
                      "note": "0.0 == authority R reproduces evaluate.py d_seg (modules.py:108-113/112)"}), flush=True)

    with temporary_mlx_device("gpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
        print(json.dumps({"stage": "arm_b_start", "utc": _utc()}), flush=True)
        b = run_arm_b(args, adapter, seg_cpu, lstar, margin, args.render_h, args.render_w)
        print(json.dumps({"stage": "arm_a_start", "utc": _utc()}), flush=True)
        a = run_arm_a(args, adapter, seg_cpu, lstar, margin, args.render_h, args.render_w, int(b["n_params"]))

    # ---- VERDICT ----
    a_dseg, b_dseg = float(a["d_seg"]), float(b["d_seg"])
    a_ring = float(a["flip_out_band_interior"])
    b_ring = float(b["flip_out_band_interior"])
    param_ok = int(a["n_params"]) <= int(b["n_params"])
    beats = a_dseg < b_dseg
    ring_free = a_ring < b_ring
    if beats and ring_free and param_ok:
        verdict = "PASS"
    elif beats and param_ok:
        verdict = "PASS_DSEG_ONLY"  # beats d_seg but not strictly ring-free
    else:
        verdict = "FAIL"

    summary = {
        "utc": _utc(),
        "evidence_grade": "contest-CPU advisory (single-frame, exact scorer, NOT 600-sample harness)",
        "promotion_eligible": False,
        "score_claim": False,
        "render_res": [args.render_h, args.render_w],
        "R_fidelity_d_seg_GT_vs_Lstar": float(d_seg_gt),
        "arm_a_sg_drf": {k: a[k] for k in a if k != "realized"},
        "arm_b_coord_inr": {k: b[k] for k in b if k != "realized"},
        "head_to_head": {
            "a_d_seg": a_dseg, "b_d_seg": b_dseg, "a_minus_b": a_dseg - b_dseg,
            "a_interior_ring": a_ring, "b_interior_ring": b_ring,
            "a_params": int(a["n_params"]), "b_params": int(b["n_params"]),
            "param_ok_a_le_b": param_ok, "a_beats_b_dseg": beats, "a_ring_free_vs_b": ring_free,
        },
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "argmax_maps.npz",
        lstar=lstar,
        arm_a_realized=np.asarray(a["realized"]),
        arm_b_realized=np.asarray(b["realized"]),
    )
    print(json.dumps({"stage": "VERDICT", **summary["head_to_head"], "verdict": verdict}), flush=True)
    print(json.dumps({"stage": "DONE", "out_dir": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
