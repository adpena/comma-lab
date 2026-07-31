#!/usr/bin/env python
"""ddm_pa1r (#793) — full-n600 endpoint verdict for a Pool-A race arm (additive-S currency).

SCRATCHPAD (rebuildable from checkpoint + gt_n600 + committed code; NOT committed).  The trainer's
in-loop a1_gate d_seg is a SUBSET (fd2 geometry, ~36 pairs) — insufficient for the additive-S
verdict (B's gate 0.004986 vs nv1's full-n600 0.005114).  This runs the AUTHORITATIVE full-n600
realized d_seg through R on the shipped EMA shadow (chunked <=120, cpu-torch frozen SegNet argmax vs
gt_n600 lstars) + the real SMEVR byte-close (counted_bytes_ledger) -> additive S + c telemetry.

Reuses committed trainer functions ONLY.  Validated on B (must reproduce nv1: d_seg 0.005114475,
tokens 255,907, S 0.6842).  score_claim=False; [macOS-CPU advisory]; pointer 0.1910828242 UNMOVED.

Usage: ddm_pa1r_endpoint_verdict.py <arm_dir> [--ckpt <name>] [--chunk 100]
       ddm_pa1r_endpoint_verdict.py --b-validate   # re-derive B's nv1 anchor
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

GT_CACHE = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CONTEST_N = 37_545_489
B_DIR = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/control"


def _cfg_from_json(cfg_path: Path):
    import experiments.train_tr1_partition_renderer_mlx as T
    d = json.loads(cfg_path.read_text())
    fields = {f for f in T.TR1Config.__dataclass_fields__}
    kw = {k: v for k, v in d.items() if k in fields}
    return T.TR1Config(**kw)


def endpoint_verdict(arm_dir: str, ckpt_name: str = "stage_seg_trunk_tau_final.npz",
                     chunk: int = 100) -> dict:
    import mlx.core as mx

    import experiments.train_tr1_partition_renderer_mlx as T
    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet

    adir = Path(arm_dir)
    cfg = _cfg_from_json(adir / "tr1_config.json")
    cfg = replace(cfg, num_pairs=600)
    ckpt = adir / "checkpoints" / ckpt_name

    model = T.build_module(cfg)
    st = T.load_checkpoint(ckpt, model)
    # ship the EMA SHADOW (the shipped state; EMA non-negotiable). Swap it in and keep it.
    T.ema_snapshot_swap(model, st["ema"])
    # §3.3(a) resume-past-knee: re-engage the STE quantization so render_frame emits the SHIPPED
    # QUANTIZED lattice (uint8 codes), NOT continuous tokens (the trainer's resume block does this;
    # WITHOUT it the verdict under-reports d_seg by ~1e-4 = the quantization gap). at_knee + a
    # post-CE stage => engaged (matches the byte-closed shipped decode nv1 measured).
    stage = st.get("meta", {}).get("stage", "")
    if cfg.token_quant_anneal == "at_knee" and stage != "seg_trunk_ce":
        model._quant_engaged = True
    mx.eval(model.parameters())

    lstars = open_stored_npy_memmap(GT_CACHE, "lstars")
    seg_cpu = load_real_segnet("cpu")

    dsegs: list[float] = []
    ids = list(range(600))
    for s in range(0, 600, chunk):
        block = ids[s:s + chunk]
        frames = []
        with mx.stream(mx.cpu):
            for i in block:
                rgb = model.render_frame(int(i))
                mx.eval(rgb)
                frames.append(np.asarray(rgb, dtype=np.float32)[0])
        cams = [_torch_R_to_camera_uint8(f) for f in frames]
        gts = [np.asarray(lstars[i], dtype=np.int64) for i in block]
        ds, _ = cpu_verdict_d_seg_argmax_batch(seg_cpu, cams, gts)
        dsegs.extend(ds)
        del frames, cams, gts

    d_seg = float(np.mean(dsegs))
    ledger = T.counted_bytes_ledger(model, cfg)
    total = int(ledger["total_counted_bytes"])
    tok = int(ledger.get("tokens_bytes_smevr", ledger["tokens_bytes"]))
    seg_term = 100.0 * d_seg
    rate_term = 25.0 * total / CONTEST_N
    s_add = seg_term + rate_term
    c = seg_term * rate_term
    return {
        "arm_dir": str(adir), "ckpt": ckpt_name, "n600_d_seg": d_seg,
        "d_seg_per_pair_max": float(np.max(dsegs)),
        "tokens_bytes_smevr": tok, "total_counted_bytes": total,
        "seg_term": seg_term, "rate_term": rate_term,
        "s_additive": s_add, "c_product_telemetry": c,
        "renderer_bytes": int(ledger.get("renderer_bytes", 0)),
        "selector_ledger_bytes": int(ledger.get("selector_ledger_bytes", 0)),
        "rowband_spec_bytes": int(ledger.get("rowband_spec_bytes", 0)),
        "score_claim": False, "axis": "macOS-CPU advisory",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir", nargs="?", default=None)
    ap.add_argument("--ckpt", default="stage_seg_trunk_tau_final.npz")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--b-validate", action="store_true")
    args = ap.parse_args()
    target = B_DIR if args.b_validate else args.arm_dir
    if target is None:
        ap.error("give an arm_dir or --b-validate")
    out = endpoint_verdict(target, args.ckpt, args.chunk)
    if args.b_validate:
        out["nv1_anchor_check"] = {
            "d_seg_nv1": 0.005114475, "d_seg_drift": out["n600_d_seg"] - 0.005114475,
            "tokens_nv1": 255907, "tokens_drift": out["tokens_bytes_smevr"] - 255907,
            "s_nv1": 0.6842, "s_drift": out["s_additive"] - 0.6842}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
