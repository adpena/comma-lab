#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#336 SPARC-GRAIN APPLY — per-class-weighted / margin-aware sensitivity bit-alloc on the witness.

MEANS. pointer 0.19110 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE — a bounded-subset
compression measurement, never a score. NO-FAKE: real frozen witness checkpoint, real #202
byte-close packing grammar (per-tensor int8 symmetric + brotli-11 = REAL brotli stream lengths of
the exact blob the archive carries), real frozen CPU-torch SegNet argmax through the contest R on
real GT pairs.

WHY THIS EXISTS (the SPARC grain, `.omx/research/papers_checked_stac_sparc_taskaware_compression_
20260709.md`): the shipped #336 apply (`tools/apply_sensitivity_bitalloc_witness.py`, FEED-07k)
drives the #157 KKT reverse-water-fill off the AGGREGATE d_seg response — the contest metric is a
uniform per-pixel argmax-mean, so aggregate sensitivity directly targets it, BUT an aggregate
equal-marginal allocation STARVES the weights that support rare classes by construction (SPARC's
"tilted rate loss" phenomenon): Lane is 0.64% of pixels, so a tensor that matters mostly for Lane
shows a tiny AGGREGATE Δd_seg and gets compressed away, even though Lane carries ~19% of the flip
mass and is the unstable orbit (IoU 0.263). This tool adds the SPARC-honored functional: a
PER-CLASS-EQUAL-WEIGHTED d_seg response ``D_pc = mean_k missrate_k`` (each of the 5 classes
contributes its OWN within-class miss rate with weight 1/5, so Lane counts 20% not 0.64%). It runs
BOTH sensitivities head-to-head through the SAME #157 waterfill and re-measures the TRUE aggregate
d_seg (the contest metric) for each allocation — the DECISION metric is
``ΔS = 100·Δd_seg_aggregate + 25·Δbytes/37_545_489`` measured on the true aggregate for BOTH arms.

REUSE, not re-derivation: the realization grammar (`_realize_alloc`, `_int8_realize`,
`_brotli_bytes`, `_intn_qdq_numpy`), the render authority (`measure_contour_string_flip_coding`),
the checkpoint loader + render ctx (`levelset_byte_close_and_eval`), and the KKT allocator
(`tac.frontier_exact_bitalloc.waterfill_bit_allocation` / `lam_for_target_mean_bits` /
`CombinedTensorSensitivity`) are all imported AS-IS from the shipped surfaces. This tool adds ONLY
the per-class response measurement + the equal-per-class weighting of the sensitivity coefficient.

Honest bounds: probe/eval subsets < n600 are ADVISORY (evenly-spaced pairs; labeled). A promotable
verdict needs the full n600 byte-close + exact eval (reported as `what_n600_needs`). This is a
compress-half rate instrument: it changes archive bytes, never the trained weights.

Usage (mod32cap ep650 BEST snapshot, work copy):
  TAC_GOVERNED_ADMISSION=1 .venv/bin/python tools/safe_run.py -- \
    .venv/bin/python tools/apply_perclass_bitalloc_witness.py \
      --ckpt-dir experiments/results/perclass_bitalloc_witness_20260710 \
      --npz-name mod32cap_ep650_BEST.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --probe-pairs 16 --eval-pairs 96 --mean-bits 6 5 --torch-threads 2 \
      --out experiments/results/perclass_bitalloc_witness_20260710/perclass_bitalloc_n96.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream", _REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RATE_DENOM = 37_545_489.0
N_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def per_pair_class_counts(realized: np.ndarray, gt: np.ndarray,
                          n_classes: int = N_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    """Per-class (miss_count, pixel_count) for ONE pair.

    ``realized`` / ``gt`` are (H, W) int argmax label grids. For each GT class k:
    ``miss_k`` = # pixels where gt==k and realized!=gt; ``npx_k`` = # pixels where gt==k. Returned
    as two length-``n_classes`` int arrays so per-class miss RATES aggregate order-independently over
    pairs (sum miss / sum npx), matching the resumable per-pair cache."""
    realized = np.asarray(realized, np.int64)
    gt = np.asarray(gt, np.int64)
    if realized.shape != gt.shape:
        raise ValueError(f"per_pair_class_counts: shape mismatch {realized.shape} vs {gt.shape}")
    miss = np.zeros((n_classes,), np.int64)
    npx = np.zeros((n_classes,), np.int64)
    wrong = realized != gt
    for k in range(n_classes):
        mask = gt == k
        npx[k] = int(mask.sum())
        miss[k] = int(np.logical_and(mask, wrong).sum())
    return miss, npx


def aggregate_perclass(pair_records: list[dict], n_classes: int = N_CLASSES,
                       ) -> tuple[float, np.ndarray]:
    """(aggregate flip rate, per-class within-class miss-rate vector) from per-pair records.

    ``aggregate`` = total misses / total pixels (the CONTEST metric = uniform per-pixel argmax-mean).
    ``per_class[k]`` = Σ_pairs miss_k / Σ_pairs npx_k (within-class miss rate; a class absent from all
    pairs stays 0.0). Order-independent, so a resumed run is value-identical."""
    tot_miss = 0
    tot_px = 0
    cls_miss = np.zeros((n_classes,), np.int64)
    cls_px = np.zeros((n_classes,), np.int64)
    for rec in pair_records:
        m = np.asarray(rec["miss"], np.int64)
        n = np.asarray(rec["npx"], np.int64)
        cls_miss += m
        cls_px += n
        tot_miss += int(m.sum())
        tot_px += int(n.sum())
    agg = (tot_miss / tot_px) if tot_px > 0 else 0.0
    pc = np.where(cls_px > 0, cls_miss / np.maximum(cls_px, 1), 0.0).astype(np.float64)
    return float(agg), pc


def perclass_functional(per_class_missrate: np.ndarray) -> float:
    """SPARC-honored de-starving functional: EQUAL-per-class mean of within-class miss rates.

    ``D_pc = mean_k missrate_k`` — each class contributes 1/K regardless of pixel count, so Lane
    (0.64% of pixels) counts 20% not 0.64%. This is the "tilted" objective the sensitivity probe
    uses to keep the waterfill from starving rare-class-supporting weights. It is NOT the contest
    metric (that stays aggregate); it is the ALLOCATION sensitivity only."""
    v = np.asarray(per_class_missrate, np.float64)
    return float(v.mean()) if v.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", required=True, help="FROZEN snapshot dir (never the live run).")
    ap.add_argument("--npz-name", required=True)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--probe-pairs", type=int, default=16)
    ap.add_argument("--eval-pairs", type=int, default=96)
    ap.add_argument("--probe-bits", type=int, default=5)
    ap.add_argument("--mean-bits", type=float, nargs="+", default=[6.0, 5.0])
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--mem-floor-mb", type=int, default=8192,
                    help="P0: never endanger the live owed-16 trainer — abort clean below this.")
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=8.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--ckpt-provenance", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-json", default="")
    ap.add_argument("--chunk-seconds", type=float, default=0.0,
                    help="wall-clock chunk budget: exit rc=7 (resumable, mid-unit safe) once "
                    "elapsed exceeds this (0 = no budget). Drive mode for bounded foreground.")
    args = ap.parse_args()

    import levelset_byte_close_and_eval as bc
    import torch
    from apply_sensitivity_bitalloc_witness import _brotli_bytes, _realize_alloc
    from measure_contour_string_flip_coding import (
        _rss_str,
        build_render_ctx,
        render_frame1_both,
        segnet_argmax,
    )

    from tac.boundary_math.seg_core import load_real_segnet
    from tac.frontier_exact_bitalloc import (
        CombinedTensorSensitivity,
        lam_for_target_mean_bits,
        waterfill_bit_allocation,
    )

    torch.set_num_threads(max(1, int(args.torch_threads)))
    t_start = time.time()
    deadline = (t_start + float(args.chunk_seconds)) if float(args.chunk_seconds) > 0 else None

    def _mem_guard() -> None:
        # CLASS-1 fix: reclaimable-aware basis (raw psutil .available over-trusts dirty inactive anon).
        try:
            from tools.mem_basis import conservative_free_gib
        except Exception:
            from mem_basis import conservative_free_gib  # type: ignore
        _gib = conservative_free_gib(default=float("inf"))
        avail = (1 << 40) if _gib == float("inf") else int(_gib * 1024)
        if avail < int(args.mem_floor_mb):
            print(f"[#336-pc] MEM-GUARD: available {avail}MB < floor {args.mem_floor_mb}MB — "
                  "aborting clean (P0: never endanger the live trainer).", flush=True)
            sys.exit(7)

    params, cfg = bc._load_levelset_ckpt(Path(args.ckpt_dir), args.npz_name)
    so = bc.detect_self_orient(cfg, {
        "freq_across": float(args.so_freq_across), "freq_along": float(args.so_freq_along),
        "tau": float(args.so_tau), "iters": int(args.so_iters)})
    code_fp = np.asarray(params.pop("code"), np.float32)
    params_fp = {k: np.asarray(v, np.float32) for k, v in params.items()}

    alloc_names = [k for k, v in params_fp.items() if v.ndim >= 2] + ["code"]
    numel = {k: int(params_fp[k].size) for k in params_fp if params_fp[k].ndim >= 2}
    numel["code"] = int(code_fp.size)
    absmax = {k: float(np.abs(params_fp[k]).max()) for k in params_fp if params_fp[k].ndim >= 2}
    absmax["code"] = float(np.abs(code_fp).max())
    n_cls = int(cfg["n_classes"])
    print(f"[#336-pc] allocatable tensors: {alloc_names}", flush=True)

    manifest = {
        "n_pairs": int(cfg["n_pairs"]), "n_classes": n_cls,
        "hidden_dim": int(cfg["hidden_dim"]), "n_hidden": int(cfg["n_hidden"]),
        "mod_dim": int(cfg["mod_dim"]), "activation": str(cfg["activation"]),
        "softmax_temp": float(cfg["softmax_temp"]), "chroma": bool(cfg["chroma"]),
        "wire_w0": float(cfg["wire_w0"]), "wire_s0": float(cfg["wire_s0"]),
        "hosc_beta": float(cfg["hosc_beta"]), "hosc_omega": float(cfg["hosc_omega"]),
        "bank_n_scales": int(cfg["bank_n_scales"]), "bank_n_orient0": int(cfg["bank_n_orient0"]),
        "bank_f0": float(cfg["bank_f0"]), "bank_base": float(cfg["bank_base"]),
        "bank_n_iso": int(cfg["bank_n_iso"]), "max_bank_freq": cfg["max_bank_freq"],
        "render_h": int(cfg["render_h"]), "render_w": int(cfg["render_w"]),
        "camera_h": bc.CAMERA_H, "camera_w": bc.CAMERA_W,
        "self_orient": bool(so["self_orient"]), "n_dir_freqs": int(so.get("n_dir_freqs", 0)),
        "so_freq_across": float(so.get("freq_across", 0.0)),
        "so_freq_along": float(so.get("freq_along", 0.0)),
        "so_tau": float(so.get("tau", 4.0)), "so_iters": int(so.get("iters", 0)),
        "lane_render_band": None,
    }
    ctx = build_render_ctx(bc, {}, code_fp, manifest, None)
    z = np.load(args.gt_cache, allow_pickle=False)
    lstars_all = z["lstars"]
    n_avail = len(lstars_all)
    probe_ids = sorted({round(i * n_avail / args.probe_pairs) % n_avail
                        for i in range(args.probe_pairs)})
    eval_ids = sorted({round(i * n_avail / args.eval_pairs) % n_avail
                       for i in range(args.eval_pairs)})
    seg_cpu = load_real_segnet("cpu")

    state_path = Path(args.state_json) if args.state_json else Path(str(args.out) + ".state.json")
    state: dict = (json.loads(state_path.read_text()) if state_path.exists() else {})
    state.setdefault("units", {})   # unit_key -> {pair_id -> {"miss":[...], "npx":[...]}}

    def _save_state() -> None:
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(state_path)

    def _measure_full(unit_key: str, pair_ids, params_dq, code_dq) -> tuple[float, np.ndarray, dict]:
        """Render pair_ids through R, return (aggregate flip rate, per-class miss-rate vec, per-class
        pixel-count dict). Resumable via per-pair state cache; mid-unit deadline safe."""
        c = dict(ctx)
        c["params"] = params_dq
        c["code"] = code_dq
        rh, rw, ch, cw = c["rh"], c["rw"], c["ch"], c["cw"]
        cache = state["units"].setdefault(unit_key, {})
        for j, pi in enumerate(pair_ids):
            key = str(pi)
            if key in cache:
                continue
            if deadline is not None and time.time() > deadline:
                _save_state()
                print(f"[#336-pc] CHUNK-BUDGET mid-unit ({j}/{len(pair_ids)} done) — resumable "
                      f"exit rc=7. {_rss_str()}", flush=True)
                sys.exit(7)
            rgb, _ = render_frame1_both(bc, c, pi)
            realized = segnet_argmax(seg_cpu, bc._torch_R_reference(rgb, rh, rw, ch, cw))
            gt = np.asarray(lstars_all[pi], np.int64)
            miss, npx = per_pair_class_counts(realized, gt, n_cls)
            cache[key] = {"miss": miss.tolist(), "npx": npx.tolist()}
            if (j + 1) % 8 == 0:
                print(f"    [{unit_key} render {j + 1}/{len(pair_ids)}] {_rss_str()}", flush=True)
                _save_state()
        _save_state()
        recs = [cache[str(pi)] for pi in pair_ids]
        agg, pc = aggregate_perclass(recs, n_cls)
        cls_px = np.sum([np.asarray(r["npx"]) for r in recs], axis=0)
        return agg, pc, {CLASS_NAMES[k]: int(cls_px[k]) for k in range(n_cls)}

    # ---- baseline (shipped int8) ------------------------------------------------------------
    _mem_guard()
    dq0, code_dq0, base_b0, code_b0, distinct0 = _realize_alloc(params_fp, code_fp, {})
    bytes0 = _brotli_bytes(base_b0, code_b0)
    d0a_probe, d0pc_probe, _ = _measure_full("d0_probe", probe_ids, dq0, code_dq0)
    print(f"[#336-pc] baseline int8: weights_total={bytes0['weights_total']}B "
          f"agg_probe={d0a_probe:.6f} pc_probe={perclass_functional(d0pc_probe):.6f} "
          f"per-class={[round(float(x), 5) for x in d0pc_probe]} {_rss_str()}", flush=True)

    # ---- per-tensor sensitivity probe: BOTH aggregate + per-class ---------------------------
    pb = int(args.probe_bits)
    delta_scale = (2.0 ** -pb) - (2.0 ** -8)
    sens_rows: dict[str, dict] = {}
    s_agg: dict[str, float] = {}
    s_pc: dict[str, float] = {}
    for name in alloc_names:
        _mem_guard()
        dq, code_dq, _, _, _ = _realize_alloc(params_fp, code_fp, {name: pb})
        a_t, pc_t, _ = _measure_full(f"probe:{name}", probe_ids, dq, code_dq)
        d_pc_t = perclass_functional(pc_t)
        delta_agg = a_t - d0a_probe
        delta_pc = d_pc_t - perclass_functional(d0pc_probe)
        c_agg = max(0.0, delta_agg) / delta_scale
        c_pc = max(0.0, delta_pc) / delta_scale
        denom = absmax[name] * np.sqrt(numel[name])
        s_agg[name] = float(c_agg / denom) if denom > 0 else 0.0
        s_pc[name] = float(c_pc / denom) if denom > 0 else 0.0
        sens_rows[name] = {
            "delta_agg": float(delta_agg), "delta_pc": float(delta_pc),
            "c_agg": float(c_agg), "c_pc": float(c_pc),
            "per_class_missrate_at_probe": [round(float(x), 6) for x in pc_t],
            "numel": numel[name], "absmax": absmax[name]}
        print(f"  [probe] {name:16s} int{pb}: Δagg={delta_agg:+.6f} c_agg={c_agg:.5f} | "
              f"Δpc={delta_pc:+.6f} c_pc={c_pc:.5f} {_rss_str()}", flush=True)

    def _mk_sens(per_tensor: dict[str, float]) -> CombinedTensorSensitivity:
        return CombinedTensorSensitivity(
            per_tensor=dict(per_tensor), g_seg=dict(per_tensor),
            g_pose=dict.fromkeys(per_tensor, 0.0),
            absmax={k: absmax[k] for k in per_tensor}, numel={k: numel[k] for k in per_tensor},
            w_seg=1.0, w_pose=0.0)

    sens_agg = _mk_sens(s_agg)
    sens_pc = _mk_sens(s_pc)

    # ---- baseline eval (true aggregate + per-class, the contest surface) ---------------------
    _mem_guard()
    d0a_eval, d0pc_eval, cls_px_eval = _measure_full("d0_eval", eval_ids, dq0, code_dq0)
    print(f"[#336-pc] baseline int8 eval: agg={d0a_eval:.6f} per-class-miss="
          f"{[round(float(x), 5) for x in d0pc_eval]} {_rss_str()}", flush=True)

    def _op_for(sens, tag: str, mb: float) -> dict:
        lam = lam_for_target_mean_bits(sens, float(mb), b_min=2, b_max=8)
        alloc = waterfill_bit_allocation(sens, lam, b_min=2, b_max=8)
        dq, code_dq, base_b, code_b, distinct = _realize_alloc(params_fp, code_fp, alloc.nbits)
        by = _brotli_bytes(base_b, code_b)
        d_a, d_pc, _ = _measure_full(f"{tag}:mb{mb}", eval_ids, dq, code_dq)
        delta_bytes = int(by["weights_total"] - bytes0["weights_total"])
        delta_seg_S = 100.0 * (d_a - d0a_eval)
        delta_rate_S = 25.0 * delta_bytes / RATE_DENOM
        return {
            "sensitivity": tag, "target_mean_bits": float(mb), "lam": float(alloc.lam),
            "nbits": dict(alloc.nbits), "bytes": by,
            "d_seg_eval_aggregate": float(d_a),
            "per_class_missrate_eval": [round(float(x), 6) for x in d_pc],
            "delta_d_seg_vs_int8": float(d_a - d0a_eval),
            "delta_bytes_vs_int8": delta_bytes,
            "delta_seg_S": float(delta_seg_S), "delta_rate_S": float(delta_rate_S),
            "net_delta_S_advisory": float(delta_seg_S + delta_rate_S),
            "distinct_symbols": distinct,
        }

    operating_points = []
    for mb in args.mean_bits:
        _mem_guard()
        row_agg = _op_for(sens_agg, "aggregate", mb)
        _mem_guard()
        row_pc = _op_for(sens_pc, "per_class", mb)
        head = {
            "target_mean_bits": float(mb),
            "aggregate": row_agg, "per_class": row_pc,
            "perclass_minus_aggregate_net_S": float(
                row_pc["net_delta_S_advisory"] - row_agg["net_delta_S_advisory"]),
            "lane_missrate_aggregate_alloc": row_agg["per_class_missrate_eval"][1],
            "lane_missrate_perclass_alloc": row_pc["per_class_missrate_eval"][1],
        }
        operating_points.append(head)
        print(f"[#336-pc] mb={mb}: AGG net ΔS={row_agg['net_delta_S_advisory']:+.5f} "
              f"({row_agg['bytes']['weights_total']}B d_seg={row_agg['d_seg_eval_aggregate']:.6f}) "
              f"| PC net ΔS={row_pc['net_delta_S_advisory']:+.5f} "
              f"({row_pc['bytes']['weights_total']}B d_seg={row_pc['d_seg_eval_aggregate']:.6f}) "
              f"| PC-AGG={head['perclass_minus_aggregate_net_S']:+.5f}", flush=True)
        print(f"        Lane miss-rate: AGG-alloc={head['lane_missrate_aggregate_alloc']:.5f} "
              f"PC-alloc={head['lane_missrate_perclass_alloc']:.5f}", flush=True)

    result = {
        "task": "#336 SPARC-grain — per-class-weighted vs aggregate #157 KKT waterfill on witness",
        "authority": (f"[macOS-CPU advisory] NON-PROMOTABLE — probe n{len(probe_ids)} / "
                      f"eval n{len(eval_ids)} evenly-spaced subsets, witness-alone surface"),
        "pointer": "0.19110 UNMOVED (MEANS)",
        "utc": datetime.now(UTC).isoformat(),
        "ckpt": {"dir": str(args.ckpt_dir), "npz": args.npz_name,
                 "provenance": args.ckpt_provenance},
        "sparc_grain": ("per-class-equal-weighted functional D_pc = mean_k missrate_k (each of 5 "
                        "classes weight 1/5) de-starves Lane (0.64% pixels); DECISION metric stays "
                        "TRUE aggregate d_seg ΔS on the contest surface"),
        "reused_157_surfaces": ["tac.frontier_exact_bitalloc.waterfill_bit_allocation",
                                "tac.frontier_exact_bitalloc.lam_for_target_mean_bits",
                                "tac.frontier_exact_bitalloc.CombinedTensorSensitivity",
                                "tools.apply_sensitivity_bitalloc_witness._realize_alloc"],
        "probe_bits": pb,
        "baseline_int8": {"bytes": bytes0, "d_seg_eval_aggregate": float(d0a_eval),
                          "per_class_missrate_eval": [round(float(x), 6) for x in d0pc_eval],
                          "class_pixel_counts_eval": cls_px_eval,
                          "distinct_symbols": distinct0},
        "per_tensor_sensitivity": sens_rows,
        "operating_points": operating_points,
        "probe_pair_ids": probe_ids, "eval_pair_ids": eval_ids,
        "what_n600_needs": ("a full #202 byte-close of the winning allocation (same blob grammar; "
                            "reader unchanged) + n600 verdict through tools/"
                            "levelset_byte_close_and_eval.py, then contest-CPU/CUDA exact eval — the "
                            "ONLY promotable row. This report is the bounded advisory apply."),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[#336-pc] wrote {args.out} ({result['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
