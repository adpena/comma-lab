#!/usr/bin/env python3
"""ddm_jd4 endpoint n600 BOTH-BASES probe (task #963, endpoint-chain step 1).

Derived from the FIXED endpoint instrument (6e4a6e24fe lineage,
``experiments/ddm_jd1_endpoint_verdict.py``) — NOT from the stale SSD jd3
variant whose axis literal had to be corrected post-hoc (RR1-C2-R2-F1).

Measures the jd4 continuation FINAL checkpoint (ep1526) on ALL 600 pairs,
under BOTH parameter bases per the typed-split rule (MAIN-R5X/R6):

    ema  — the ema_shadow channel (the SHIPPED basis; A1/hold safety +
           Case-A pose adjudication read this)
    live — the raw trainable params (the plateau-slope/Case-0 policy channel)

Baselines for deltas (jd3 n600 both-bases receipt, endpoint_ep1405 — the
physical state jd4 resumed from; reanchor sets shadow:=live at entry so
entry rows are derivable, not re-measured):

    ep1405 live: d_seg 0.007150336  d_pose 0.574092  pose_term 2.396021
    ep1405 ema:  d_seg 0.005747986  d_pose 0.128853  pose_term 1.135135

Positive control (FREE — sliced from the n600 per-pair arrays, no extra
compute): the run's own 36 a1-gate pair ids under the ema basis must land
near the final a1_gate telemetry row (same pairs, same basis, same adapter).

Physics identical to the fixed instrument: pose pair =
(render(max(idx-1,0)), render(idx)) each through _apply_R -> yuv6 -> frozen
MLX PoseNet -> first-6 MSE vs gt_poses[idx][:6]; seg = argmax(segnet(f1))
vs lstar. Axis: [macOS-CPU frozen-scorer advisory], score_claim=false.
#855 caveat: MLX conv adapter ~76px argmax drift vs CPU-torch — deltas
across ckpts/bases share the adapter and cancel to first order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

RUN_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1406")
GATE_PAIR_IDS = [447, 448, 449, 450, 550, 360, 162, 42, 9, 367, 326, 153, 566, 507,
                 350, 23, 457, 232, 316, 175, 529, 1, 509, 482, 19, 393, 433, 470,
                 373, 290, 100, 104, 423, 484, 327, 289]
BASELINE_EP1405 = {
    "live": {"d_seg_mean": 0.007150336371527777, "d_pose_mean": 0.5740917290074666},
    "ema": {"d_seg_mean": 0.00574798583984375, "d_pose_mean": 0.12885309147362226},
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--ckpt-name", default="stage_joint_pose_finish_final.npz")
    ap.add_argument("--ckpt-tag", default="final_ep1526")
    ap.add_argument("--gt-cache", type=Path,
                    default=Path("/Users/adpena/Projects/pact/experiments/results/"
                                 "mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/"
                                 "jd4_endpoint_n600_both_bases.json"))
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    import dataclasses

    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # dt1: CPU is run-to-run bit-identical

    from experiments.train_tr1_partition_renderer_mlx import (
        TR1Config, build_module, ema_snapshot_swap, load_checkpoint,
    )
    from experiments.train_witness_realized_through_R_mlx import _apply_R
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )
    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    cfg_json = json.loads((args.run_dir / "tr1_config.json").read_text())
    field_names = {f.name for f in dataclasses.fields(TR1Config)}
    cfg = TR1Config(**{k: v for k, v in cfg_json.items() if k in field_names})

    adapter = load_mlx_distortion_scorer_adapter_from_upstream(
        str(REPO / "upstream"), device="cpu")
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    gt_poses = open_stored_npy_memmap(args.gt_cache, "gt_poses")

    def yuv12(f0, f1):
        pair = mx.stack([f0[0], f1[0]], axis=0)[None]
        yuv = rgb_to_yuv6_mlx(pair)
        b, t, h2, w2, c6 = yuv.shape
        return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))

    pair_ids = list(range(600))
    receipt: dict = {
        "schema": "ddm_jd4_endpoint_n600_both_bases.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] all 600 pair ids (0..599), "
                "training-vehicle endpoint, both bases, NON-PROMOTABLE",
        "score_claim": False,
        "task": 963,
        "instrument_lineage": "experiments/ddm_jd1_endpoint_verdict.py @ 6e4a6e24fe "
                              "(axis label derived, not literal)",
        "pose_semantics": "training-vehicle window objective; NOT shipped-archive "
                          "pair semantics (byte-close owns those)",
        "ckpt": args.ckpt_name,
        "baseline_ep1405": BASELINE_EP1405,
        "bases": {},
        "status": "running",
    }
    t0 = time.time()
    ckpt_path = args.run_dir / "checkpoints" / args.ckpt_name
    for basis in ("ema", "live"):
        model = build_module(cfg)
        ck = load_checkpoint(ckpt_path, model)
        if basis == "ema":
            if not ck["ema"]:
                raise RuntimeError("ema basis requested but checkpoint has no EMA")
            ema_snapshot_swap(model, ck["ema"])
        d_segs, d_poses = [], []
        for n, idx in enumerate(pair_ids):
            f1 = _apply_R(model.render_frame(int(idx)))
            f0 = _apply_R(model.render_frame(max(int(idx) - 1, 0)))
            logits = adapter.segnet(f1)
            mx.eval(logits)
            realized = np.asarray(mx.argmax(logits[0], axis=-1), dtype=np.int64)
            lstar = np.asarray(lstars[idx], dtype=np.int64)
            d_segs.append(float(np.count_nonzero(realized != lstar)) / lstar.size)
            pose_out = adapter.posenet(yuv12(f0, f1))
            pose = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            mx.eval(pose)
            p6 = np.asarray(pose, dtype=np.float64).ravel()[:6]
            tgt = np.asarray(gt_poses[idx], dtype=np.float64).ravel()[:6]
            d_poses.append(float(np.mean((p6 - tgt) ** 2)))
            if (n + 1) % 50 == 0:
                receipt["progress"] = {"basis": basis, "pairs_done": n + 1,
                                       "elapsed_s": round(time.time() - t0, 1)}
                args.out.write_text(json.dumps(receipt, indent=1))
        gate_seg = [d_segs[i] for i in GATE_PAIR_IDS]
        gate_pose = [d_poses[i] for i in GATE_PAIR_IDS]
        row = {
            "basis": basis,
            "d_seg_mean": float(np.mean(d_segs)),
            "d_pose_mean": float(np.mean(d_poses)),
            "d_pose_median": float(np.median(d_poses)),
            "pose_term_sqrt10": float(np.sqrt(10.0 * float(np.mean(d_poses)))),
            "seg_S_100x": 100.0 * float(np.mean(d_segs)),
            "delta_vs_ep1405_same_basis": {
                "d_seg": float(np.mean(d_segs)) - BASELINE_EP1405[basis]["d_seg_mean"],
                "d_pose": float(np.mean(d_poses)) - BASELINE_EP1405[basis]["d_pose_mean"],
            },
            "gate36_positive_control": {
                "d_seg_mean": float(np.mean(gate_seg)),
                "d_pose_mean": float(np.mean(gate_pose)),
                "note": "ema basis must land near the run's final a1_gate row "
                        "(same 36 pairs, same basis, same adapter)",
            },
            "d_seg_per_pair": d_segs,
            "d_pose_per_pair": d_poses,
        }
        receipt["bases"][basis] = row
        receipt["elapsed_s"] = round(time.time() - t0, 1)
        args.out.write_text(json.dumps(receipt, indent=1))
        print(f"{basis}: d_seg {row['d_seg_mean']:.7f}  d_pose {row['d_pose_mean']:.6f}  "
              f"pose_term {row['pose_term_sqrt10']:.4f}  "
              f"Δd_seg {row['delta_vs_ep1405_same_basis']['d_seg']:+.7f}  "
              f"Δd_pose {row['delta_vs_ep1405_same_basis']['d_pose']:+.6f}  "
              f"({receipt['elapsed_s']}s)")

    receipt["case_a_strict_bar"] = {
        "d_pose_bar": 0.00144, "pose_term_bar": 0.12,  # sqrt(10*0.00144) exactly
        "evaluated_on": "ema (the shipped basis)",
        "met": bool(receipt["bases"]["ema"]["d_pose_mean"] <= 0.00144),
    }
    receipt["status"] = "complete"
    args.out.write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt["case_a_strict_bar"], indent=1))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
