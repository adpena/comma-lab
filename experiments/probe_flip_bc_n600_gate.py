# SPDX-License-Identifier: MIT
"""#386 crucible-3 N-1 REFORMULATION gate ($0, read-only, CPU): the flip-weighted b_c 3-arm gate.

N-1 (n600 CONFIRMED) MEASURED that OT AREA-mass-matching HURTS realized d_seg (no_offset 0.0031436 <
menon 0.0033119 < ot_newton 0.0048921; verdict_scope FORMULATION — the SOLVER is exact, the OBJECTIVE
was wrong). This gate measures the two FLIP-mass reformulations against the same no_offset baseline on
the SAME harness / protocol / checkpoint as the N-1 verdict (reuses
``probe_laguerre_logit_offset_sweep``'s caching + realized-through-R machinery verbatim):

    * ``flip_weighted`` — the SAME damped-Newton OT solve, target = per-class FLIP SHARE (not GT area).
      UN-ANALYZED (crucible-3 P3 F3): may re-inherit N-1's cell-inflation. Expected split outcome.
    * ``flip_median``   — S1's Hamming-optimal per-edge MEDIAN threshold (a distinct closed-form path).

AUTHORITY: realized-through-R on the frozen CPU SegNet (NOT MPS, NOT MLX), fp32 EMA render.
[macOS-CPU advisory] — NON-PROMOTABLE until byte-closed exact eval. Pointer 0.19110 UNMOVED (means).
NO FAKE: real ckpt, real GT, real scorer; each mode computes what it claims on real inputs.

The SegNet forward is CHUNKED (``--verdict-batch``, per the n600-verdict-OOM law). All 600 pairs
(``--num-pairs 600``) — never a subset for the decision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "experiments", REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from probe_laguerre_logit_offset_sweep import (  # noqa: E402  (reuse the N-1 harness verbatim)
    _cache_phi_tex,
    _compose_rgb,
    _load_ckpt,
    _segnet_argmax_batch,
)
from train_witness_realized_through_R_mlx import (  # noqa: E402
    _build_render_coords,
    _torch_R_to_camera_uint8,
    load_gt_from_cache,
)

from tac.boundary_math.laguerre_logit_offset import (  # noqa: E402
    CLASS_NAMES,
    per_class_disagreement,
    solve_head_offsets,
)
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    curvelet_directional_B,
    curvelet_feats,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--so-iters", type=int, default=3)
    ap.add_argument("--verdict-batch", type=int, default=32,
                    help="chunk size for the SegNet forward (n600-verdict-OOM law; 0 = single batch).")
    ap.add_argument("--ot-tau", type=float, default=1.0,
                    help="flip_weighted OT solve softmax temperature (matches the N-1 ot_newton tau).")
    ap.add_argument("--out-json", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()
    params, cfg = _load_ckpt(Path(args.ckpt))
    rh, rw = int(cfg["render_hw"][0]), int(cfg["render_hw"][1])
    tsm = float(cfg["cfg_softmax_temp"])
    palette = params["palette"].astype(np.float64)

    gt, seg_cpu, _pn = load_gt_from_cache(Path(args.gt_cache), args.num_pairs)
    p = gt.n_pairs
    coords = _build_render_coords(rh, rw)
    bank = CurveletBankConfig(
        n_scales=int(cfg["bank_n_scales"]), n_orient0=int(cfg["bank_n_orient0"]),
        f0=float(cfg["bank_f0"]), base=float(cfg["bank_base"]), n_iso=int(cfg["bank_n_iso"]))
    big_b = curvelet_directional_B(bank, max_freq=float(cfg["cfg_max_bank_freq"]))
    curv_feats = curvelet_feats(coords, big_b).astype(np.float32)
    print(json.dumps({"stage": "loaded", "n_pairs": p, "epoch": int(cfg["epoch"]),
                      "render_hw": [rh, rw], "softmax_temp": tsm,
                      "verdict_batch": int(args.verdict_batch), "secs": round(time.time() - t0, 1)}),
          flush=True)

    # --- cache phi + tex per pair (frame1) — the expensive witness trunk, computed ONCE ---
    phis, texs = [], []
    for pi in range(p):
        phi, tex = _cache_phi_tex(params, curv_feats, coords, cfg, pi, rh, rw, args.so_iters, palette, tsm)
        phis.append(phi)
        texs.append(tex)
        if (pi + 1) % 50 == 0:
            print(json.dumps({"stage": "cache", "done": pi + 1, "secs": round(time.time() - t0, 1)}),
                  flush=True)
    gt_l = [gt.lstars[pi].astype(np.int64) for pi in range(p)]
    gt_arr = np.stack(gt_l, axis=0)
    phi_stack = np.stack(phis, axis=0).reshape(-1, 5).astype(np.float64)  # (N*H*W, K)
    gt_flat = gt_arr.reshape(-1).astype(np.int64)

    vb = int(args.verdict_batch)

    def realized_argmax(offset: np.ndarray) -> np.ndarray:
        f1s = [_torch_R_to_camera_uint8(_compose_rgb(phis[pi], texs[pi], palette, tsm, offset))
               for pi in range(p)]
        return _segnet_argmax_batch(seg_cpu, f1s, chunk=(vb if vb > 0 else 32))  # (p, h, w)

    def dseg_pc(am: np.ndarray):
        d = float(np.mean([np.count_nonzero(am[i] != gt_l[i]) / gt_l[i].size for i in range(p)]))
        pc = per_class_disagreement(am.reshape(-1), gt_arr.reshape(-1), 5)
        return d, pc

    # --- baseline (no_offset) realized argmax FIRST: it is both the no_offset arm AND the flip-
    # identifying prediction (pred) for the solvers. The flip that matters for d_seg is the REALIZED
    # SegNet-argmax-vs-GT residual, NOT the raw phi-argmax (which disagrees ~50x more for this
    # witness). render_hw == SegNet output res, so pred maps 1:1 to phi pixels. ---
    am_base = realized_argmax(np.zeros(5))
    base_pred = am_base.reshape(-1).astype(np.int64)
    base, base_pc = dseg_pc(am_base)
    print(json.dumps({"stage": "arm", "name": "no_offset", "realized_d_seg": base,
                      "secs": round(time.time() - t0, 1)}), flush=True)

    # --- solve the two reformulation offsets on the cached phi stack ($0, closed-form), flips
    # identified by the REALIZED baseline argmax (pred=base_pred) ---
    fw_b, fw_info = solve_head_offsets("flip_weighted", phi=phi_stack, gt=gt_flat, pred=base_pred,
                                       tau=float(args.ot_tau))
    fm_b, fm_info = solve_head_offsets("flip_median", phi=phi_stack, gt=gt_flat, pred=base_pred)
    print(json.dumps({"stage": "solved",
                      "flip_weighted_offsets": {int(c): round(float(fw_b[c]), 4) for c in range(5)},
                      "flip_weighted_info": {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 6))
                                             for k, v in fw_info.items()},
                      "flip_median_offsets": {int(c): round(float(fm_b[c]), 4) for c in range(5)},
                      "flip_median_info": {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 6))
                                           for k, v in fm_info.items()},
                      "secs": round(time.time() - t0, 1)}), flush=True)

    # --- realize the two offset arms through R (frozen CPU SegNet) ---
    arms: dict[str, dict] = {"no_offset": {
        "realized_d_seg": base, "offsets": {int(c): 0.0 for c in range(5)},
        "per_class": {int(k): v for k, v in base_pc.items()}}}
    for name, b in (("flip_weighted", fw_b), ("flip_median", fm_b)):
        d, pc = dseg_pc(realized_argmax(b))
        arms[name] = {"realized_d_seg": d, "offsets": {int(c): float(b[c]) for c in range(5)},
                      "per_class": {int(k): v for k, v in pc.items()}}
        print(json.dumps({"stage": "arm", "name": name, "realized_d_seg": d,
                          "secs": round(time.time() - t0, 1)}), flush=True)

    base = arms["no_offset"]["realized_d_seg"]
    out = {
        "advisory": "[macOS-CPU advisory . REALIZED-through-R CPU-SegNet authority; fp32 EMA render; "
                    "NON-PROMOTABLE until byte-closed exact eval; pointer 0.19110 UNMOVED]",
        "task": "#386 flip-weighted b_c reformulation gate (crucible-3 N-1 open arm)",
        "ckpt": str(args.ckpt), "n_pairs": p, "epoch": int(cfg["epoch"]),
        "class_names": list(CLASS_NAMES), "ot_tau": float(args.ot_tau),
        "n1_reference": {"no_offset": 0.003143556382921007, "menon": 0.003311852349175347,
                         "ot_newton": 0.0048921034071180555,
                         "note": "N-1 n600 verdict (ot_offset_n600_verdict_20260709.md); lower=better"},
        "three_arm_gate": {
            "no_offset": base,
            "flip_weighted": arms["flip_weighted"]["realized_d_seg"],
            "flip_median": arms["flip_median"]["realized_d_seg"],
            "delta_flip_weighted": arms["flip_weighted"]["realized_d_seg"] - base,
            "delta_flip_median": arms["flip_median"]["realized_d_seg"] - base,
            "note": "realized-through-R d_seg on the frozen CPU SegNet; lower=better; "
                    "relative-significance = |delta| / 0.0411 (remaining gap to sub-0.15).",
        },
        "arms": arms,
        "flip_weighted_info": {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
                               for k, v in fw_info.items()},
        "flip_median_info": {k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
                             for k, v in fm_info.items()},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({"stage": "done", "secs": round(time.time() - t0, 1),
                      "three_arm": out["three_arm_gate"], "out": args.out_json}), flush=True)


if __name__ == "__main__":
    main()
