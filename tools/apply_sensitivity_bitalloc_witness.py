#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#336 SENSITIVITY BIT-ALLOC APPLY — point the #157 KKT reverse-water-fill at the WITNESS npz.

MEANS. pointer 0.19110 UNMOVED. Authority: [macOS-CPU advisory] NON-PROMOTABLE — a bounded-subset
compression measurement, never a score. NO-FAKE: real frozen witness checkpoint, real #202
byte-close packing grammar (per-tensor int8 symmetric + brotli-11 — bytes are REAL brotli stream
lengths of the exact blob the archive would carry), real frozen CPU-torch SegNet d_seg through the
contest R on real GT pairs.

WHAT THIS APPLIES (FEED-07b row 5): the #157 reverse-water-fill tool
(``tac.frontier_exact_bitalloc.waterfill_bit_allocation`` + ``lam_for_target_mean_bits``) was
built + verified on the FRONTIER torch decoder and never pointed at our own witness checkpoint —
the compress-half of train-big-compress-small. The frontier tool's sensitivity step (torch
backward through the decoder) does not fit the witness (MLX/numpy coord-INR forward, no torch
graph), so THIS adapter supplies the sensitivity by DIRECT MEASUREMENT instead of a gradient
proxy: per tensor, quantize THAT tensor to the int-``probe_bits`` grid (others stay at the shipped
int8) and measure the d_seg response through R on a bounded probe subset. The measured response
calibrates the KKT model ``D = Σ c_t·2^{-b_t}`` (the same separable model the frontier solve
uses): ``c_t = Δd_seg_t / (2^{-probe_bits} − 2^{-8})``, fed to the UNCHANGED waterfill via
``CombinedTensorSensitivity`` (s_t = c_t/(absmax_t·√numel_t) so the allocator's internal
``c_t = s·absmax·√numel`` reproduces the measured coefficient). Reuse, not re-derivation: the
allocator, the λ-bisection, and the b_t* closed form are imported as-is.

REALIZATION (the rate win, frontier method): each tensor is qdq'd at its OWN allocated nbits ON
TOP of the shipped per-tensor int8+scale grammar — fewer distinct int8 symbols → smaller brotli.
No decoder/inflate change; the blob grammar is byte-identical in FORM (same reader), only the
symbol distribution changes. bytes_before/bytes_after = real brotli lengths (base stream + code
stream), d_seg before/after = measured through R on the eval subset.

Honest bounds: probe/eval subsets < n600 are ADVISORY (evenly-spaced pairs; labeled). A promotable
verdict needs the full n600 byte-close + exact eval (reported as `what_n600_needs`).

Usage (mod32cap snapshot):
  .venv/bin/python tools/apply_sensitivity_bitalloc_witness.py \
      --ckpt-dir <snapshot dir> --npz-name snapshot_ema_BEST.npz \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --probe-pairs 16 --eval-pairs 96 --mean-bits 6 5 --out <scratch>/bitalloc_witness.json
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


def _intn_qdq_numpy(a: np.ndarray, nbits: int) -> np.ndarray:
    """Symmetric per-tensor int-N quantize->dequantize (numpy mirror of the frontier intn_qdq)."""
    a = np.asarray(a, np.float32)
    qmax = float(2 ** (int(nbits) - 1) - 1)
    absmax = float(np.abs(a).max())
    if absmax <= 0.0 or qmax <= 0:
        return a.copy()
    scale = absmax / qmax
    return (np.clip(np.round(a / scale), -qmax, qmax) * scale).astype(np.float32)


def _int8_realize(params_fp: dict[str, np.ndarray], code_fp: np.ndarray,
                  ) -> tuple[dict[str, np.ndarray], np.ndarray, bytes, bytes]:
    """The shipped realization: per-tensor int8 symmetric quant of every base tensor + code,
    returning (dequantized params, dequantized code, base_int8_concat, code_int8_bytes) — the
    EXACT #202 build_levelset_blob grammar (base = concat int8 -> one brotli stream; code ->
    a second)."""
    from tac.boundary_math.lever_b_levelset_generator import _int8_symmetric
    dq: dict[str, np.ndarray] = {}
    chunks: list[bytes] = []
    for name, a in params_fp.items():
        q, scale = _int8_symmetric(np.asarray(a, np.float32))
        dq[name] = (q.astype(np.float32) * scale)
        chunks.append(q.astype(np.int8).tobytes())
    qc, cs = _int8_symmetric(np.asarray(code_fp, np.float32))
    code_dq = (qc.astype(np.float32) * cs).reshape(code_fp.shape)
    return dq, code_dq, b"".join(chunks), qc.astype(np.int8).tobytes()


def _realize_alloc(params_fp: dict[str, np.ndarray], code_fp: np.ndarray,
                   nbits: dict[str, int]) -> tuple[dict, np.ndarray, bytes, bytes, dict]:
    """Apply the allocation: per-tensor int-N qdq FIRST (the allocation), then the shipped int8
    grammar on top (unchanged reader). Tensors absent from ``nbits`` stay fp32 pre-int8 (= the
    shipped int8-only path). Returns realized (params_dq, code_dq, base_bytes, code_bytes,
    distinct_symbol_counts)."""
    pf = {}
    distinct: dict[str, int] = {}
    for name, a in params_fp.items():
        nb = nbits.get(name)
        pf[name] = _intn_qdq_numpy(a, nb) if nb is not None else np.asarray(a, np.float32)
    cf = (_intn_qdq_numpy(code_fp, nbits["code"]) if "code" in nbits
          else np.asarray(code_fp, np.float32))
    dq, code_dq, base_b, code_b = _int8_realize(pf, cf)
    arr = np.frombuffer(base_b, dtype=np.int8)
    distinct["base_stream"] = int(np.unique(arr).size)
    distinct["code_stream"] = int(np.unique(np.frombuffer(code_b, dtype=np.int8)).size)
    return dq, code_dq, base_b, code_b, distinct


def _brotli_bytes(base_b: bytes, code_b: bytes) -> dict[str, int]:
    import brotli
    nb = len(brotli.compress(base_b, quality=11))
    nc = len(brotli.compress(code_b, quality=11))
    return {"base_brotli": nb, "code_brotli": nc, "weights_total": nb + nc}


def measure_d_seg(bc, ctx, seg_cpu, lstars_all, pair_ids, params_dq, code_dq,
                  *, pair_cache: dict | None = None, deadline: float | None = None,
                  persist=None) -> float:
    """Mean d_seg over ``pair_ids`` through the contest R + frozen CPU SegNet (witness-alone
    surface: quantization attribution is clean of the lane-band composite, which is
    weight-independent). REUSES the #307 tool's render path (tools/measure_contour_string_
    flip_coding.py) — one render authority, not a re-type.

    Chunked-foreground drive support (2026-07-07): ``pair_cache`` (a state-dict sub-map, str(pi)
    -> per-pair d_seg) makes the unit resumable MID-way; when ``deadline`` (epoch seconds) passes
    at a pair boundary, ``persist()`` is called and the process exits rc=7 (resumable). The mean
    over per-pair values is order-independent, so a resumed unit is value-identical."""
    from measure_contour_string_flip_coding import _rss_str, render_frame1_both, segnet_argmax
    c = dict(ctx)
    c["params"] = params_dq
    c["code"] = code_dq
    rh, rw, ch, cw = c["rh"], c["rw"], c["ch"], c["cw"]
    vals = []
    for j, pi in enumerate(pair_ids):
        key = str(pi)
        if pair_cache is not None and key in pair_cache:
            vals.append(float(pair_cache[key]))
            continue
        if deadline is not None and time.time() > deadline:
            if persist is not None:
                persist()
            print(f"[#336] CHUNK-BUDGET reached mid-unit ({j}/{len(pair_ids)} pair-values done) "
                  f"— exiting resumable (rc=7). {_rss_str()}", flush=True)
            sys.exit(7)
        rgb, _ = render_frame1_both(bc, c, pi)
        realized = segnet_argmax(seg_cpu, bc._torch_R_reference(rgb, rh, rw, ch, cw))
        gt = np.asarray(lstars_all[pi], np.int64)
        v = float((realized != gt).mean())
        vals.append(v)
        if pair_cache is not None:
            pair_cache[key] = v
        if (j + 1) % 8 == 0:  # spike localization (2026-07-07 rc=137 diagnosis)
            print(f"    [render {j + 1}/{len(pair_ids)}] {_rss_str()}", flush=True)
            if persist is not None:
                persist()  # survive an external SIGKILL mid-unit (lose <=8 pair-values)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", required=True, help="FROZEN snapshot dir (never the live run).")
    ap.add_argument("--npz-name", required=True)
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--probe-pairs", type=int, default=16,
                    help="evenly-spaced pairs for the per-tensor sensitivity probe (advisory).")
    ap.add_argument("--eval-pairs", type=int, default=96,
                    help="evenly-spaced pairs for the before/after d_seg re-measure (advisory).")
    ap.add_argument("--probe-bits", type=int, default=5,
                    help="per-tensor probe grid (int-N while others stay int8).")
    ap.add_argument("--mean-bits", type=float, nargs="+", default=[6.0, 5.0],
                    help="waterfill operating points (param-weighted mean bits); each is paired "
                    "with a uniform int-round(mean) control at the ~same budget.")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--mem-floor-mb", type=int, default=6144)
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=8.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--ckpt-provenance", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-json", default="",
                    help="resumable per-unit state file (default: <out>.state.json). Every "
                    "completed measurement unit (baseline probe / per-tensor probe / baseline "
                    "eval / per-operating-point eval) is persisted atomically so the run can be "
                    "driven in bounded foreground chunks — the 2026-07-07 daemon-context kill "
                    "(silent SIGKILL of detached trees at ~5 min, tool RSS MEASURED flat 2.4 GiB) "
                    "makes chunked-foreground the robust drive mode.")
    ap.add_argument("--max-units", type=int, default=0,
                    help="exit rc=7 (resumable) after completing this many NEW units (0 = all).")
    ap.add_argument("--chunk-seconds", type=float, default=0.0,
                    help="wall-clock chunk budget: exit rc=7 (resumable, mid-unit safe via the "
                    "per-pair cache) once elapsed exceeds this (0 = no budget). Drive mode for "
                    "bounded foreground calls.")
    args = ap.parse_args()

    import levelset_byte_close_and_eval as bc
    import psutil
    import torch
    from measure_contour_string_flip_coding import build_render_ctx

    from tac.boundary_math.seg_core import load_real_segnet
    from tac.frontier_exact_bitalloc import (
        CombinedTensorSensitivity,
        lam_for_target_mean_bits,
        waterfill_bit_allocation,
    )

    torch.set_num_threads(max(1, int(args.torch_threads)))
    t_start = time.time()

    def _mem_guard():
        avail = psutil.virtual_memory().available // 1048576
        if avail < int(args.mem_floor_mb):
            print(f"[#336] MEM-GUARD: available {avail}MB < floor {args.mem_floor_mb}MB — "
                  "aborting cleanly (P0: never endanger the live trainer).", flush=True)
            sys.exit(7)

    params, cfg = bc._load_levelset_ckpt(Path(args.ckpt_dir), args.npz_name)
    so = bc.detect_self_orient(cfg, {
        "freq_across": float(args.so_freq_across), "freq_along": float(args.so_freq_along),
        "tau": float(args.so_tau), "iters": int(args.so_iters)})
    code_fp = np.asarray(params.pop("code"), np.float32)
    params_fp = {k: np.asarray(v, np.float32) for k, v in params.items()}

    # allocatable = dim>=2 base tensors + code (biases/1-D stay at the shipped int8; they are
    # ~0.4% of params — allocation value negligible, attribution noise not worth it).
    alloc_names = [k for k, v in params_fp.items() if v.ndim >= 2] + ["code"]
    numel = {k: int(params_fp[k].size) for k in params_fp if params_fp[k].ndim >= 2}
    numel["code"] = int(code_fp.size)
    absmax = {k: float(np.abs(params_fp[k]).max()) for k in params_fp if params_fp[k].ndim >= 2}
    absmax["code"] = float(np.abs(code_fp).max())
    print(f"[#336] allocatable tensors: {alloc_names} (total {sum(numel.values())} params of "
          f"{sum(v.size for v in params_fp.values()) + code_fp.size})", flush=True)

    # render authority context (witness-alone; band off => lane_pairs None)
    manifest = {
        "n_pairs": int(cfg["n_pairs"]), "n_classes": int(cfg["n_classes"]),
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
    # ctx params/code are swapped per config by measure_d_seg
    ctx = build_render_ctx(bc, {}, code_fp, manifest, None)
    z = np.load(args.gt_cache, allow_pickle=False)
    lstars_all = z["lstars"]
    n_avail = len(lstars_all)
    probe_ids = sorted({round(i * n_avail / args.probe_pairs) % n_avail
                        for i in range(args.probe_pairs)})
    eval_ids = sorted({round(i * n_avail / args.eval_pairs) % n_avail
                       for i in range(args.eval_pairs)})
    seg_cpu = load_real_segnet("cpu")

    # ---- resumable per-unit state (chunked-foreground drive mode) ---------------------------
    state_path = Path(args.state_json) if args.state_json else Path(str(args.out) + ".state.json")
    state: dict = (json.loads(state_path.read_text())
                   if state_path.exists() else {"probes": {}, "ops": {}})
    state.setdefault("pair_vals", {})
    units_done = 0
    deadline = (t_start + float(args.chunk_seconds)) if float(args.chunk_seconds) > 0 else None

    def _save_state():
        tmp = state_path.with_suffix(state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(state_path)

    def _measure_unit(unit_key: str, pair_ids, params_dq_u, code_dq_u) -> float:
        return measure_d_seg(
            bc, ctx, seg_cpu, lstars_all, pair_ids, params_dq_u, code_dq_u,
            pair_cache=state["pair_vals"].setdefault(unit_key, {}),
            deadline=deadline, persist=_save_state)

    def _unit_done():
        nonlocal units_done
        _save_state()
        units_done += 1
        if int(args.max_units) and units_done >= int(args.max_units):
            print(f"[#336] BOUNDED-CHUNK: completed {units_done} new units — exiting resumable "
                  f"(rc=7); state={state_path}", flush=True)
            sys.exit(7)

    # ---- baseline (the shipped int8 realization) --------------------------------------------
    _mem_guard()
    dq0, code_dq0, base_b0, code_b0, distinct0 = _realize_alloc(params_fp, code_fp, {})
    bytes0 = _brotli_bytes(base_b0, code_b0)
    from measure_contour_string_flip_coding import _rss_str
    if "d0_probe" in state:
        d0_probe = float(state["d0_probe"])
    else:
        d0_probe = _measure_unit("d0_probe", probe_ids, dq0, code_dq0)
        state["d0_probe"] = d0_probe
        _unit_done()
    print(f"[#336] baseline int8: weights_total={bytes0['weights_total']}B "
          f"d_seg(probe n{len(probe_ids)})={d0_probe:.6f} {_rss_str()}", flush=True)

    # ---- per-tensor MEASURED sensitivity probe ----------------------------------------------
    pb = int(args.probe_bits)
    delta_scale = (2.0 ** -pb) - (2.0 ** -8)
    sens_rows = {}
    per_tensor_s = {}
    for name in alloc_names:
        if name in state["probes"]:
            row = state["probes"][name]
            d_t = float(row[f"d_seg_probe_at_int{pb}"])
        else:
            _mem_guard()
            nb = {name: pb}
            dq, code_dq, _, _, _ = _realize_alloc(params_fp, code_fp, nb)
            d_t = _measure_unit(f"probe:{name}", probe_ids, dq, code_dq)
            row = None
        delta = d_t - d0_probe
        c_t = max(0.0, delta) / delta_scale
        s_t = (c_t / (absmax[name] * np.sqrt(numel[name]))
               if absmax[name] > 0 and numel[name] > 0 else 0.0)
        per_tensor_s[name] = float(s_t)
        sens_rows[name] = {f"d_seg_probe_at_int{pb}": float(d_t),
                           "delta_d_seg": float(delta), "c_t": float(c_t),
                           "numel": numel[name], "absmax": absmax[name]}
        print(f"  [probe] {name:16s} int{pb}: d_seg={d_t:.6f} (Δ={delta:+.6f}) c_t={c_t:.5f} "
              f"{_rss_str()}", flush=True)
        if row is None:
            state["probes"][name] = sens_rows[name]
            _unit_done()

    sens = CombinedTensorSensitivity(
        per_tensor=per_tensor_s, g_seg=dict(per_tensor_s),
        g_pose=dict.fromkeys(per_tensor_s, 0.0),
        absmax={k: absmax[k] for k in per_tensor_s},
        numel={k: numel[k] for k in per_tensor_s},
        w_seg=1.0, w_pose=0.0)

    # ---- allocate + realize + re-measure at each operating point ----------------------------
    if "d0_eval" in state:
        d0_eval = float(state["d0_eval"])
    else:
        _mem_guard()
        d0_eval = _measure_unit("d0_eval", eval_ids, dq0, code_dq0)
        state["d0_eval"] = d0_eval
        _unit_done()
    print(f"[#336] baseline int8 d_seg(eval n{len(eval_ids)})={d0_eval:.6f} {_rss_str()}",
          flush=True)
    operating_points = []
    for mb in args.mean_bits:
        op_key = f"mb{mb}"
        if op_key in state["ops"]:
            operating_points.append(state["ops"][op_key])
            continue
        _mem_guard()
        lam = lam_for_target_mean_bits(sens, float(mb), b_min=2, b_max=8)
        alloc = waterfill_bit_allocation(sens, lam, b_min=2, b_max=8)
        dq, code_dq, base_b, code_b, distinct = _realize_alloc(params_fp, code_fp, alloc.nbits)
        by = _brotli_bytes(base_b, code_b)
        d_w = _measure_unit(f"mb{mb}:wf", eval_ids, dq, code_dq)
        # uniform control at the same nominal budget
        u = round(mb)
        nb_u = dict.fromkeys(alloc_names, u)
        dqu, code_dqu, base_bu, code_bu, _ = _realize_alloc(params_fp, code_fp, nb_u)
        by_u = _brotli_bytes(base_bu, code_bu)
        d_u = _measure_unit(f"mb{mb}:u", eval_ids, dqu, code_dqu)
        row = {
            "target_mean_bits": float(mb), "lam": float(alloc.lam),
            "nbits": dict(alloc.nbits),
            "waterfill": {"bytes": by, "d_seg_eval": float(d_w),
                          "delta_d_seg_vs_int8": float(d_w - d0_eval),
                          "delta_bytes_vs_int8": int(by["weights_total"]
                                                     - bytes0["weights_total"]),
                          "delta_rate_S": 25.0 * (by["weights_total"]
                                                  - bytes0["weights_total"]) / RATE_DENOM,
                          "delta_seg_S": 100.0 * (d_w - d0_eval),
                          "distinct_symbols": distinct},
            "uniform_control": {"nbits_uniform": u, "bytes": by_u, "d_seg_eval": float(d_u),
                                "delta_d_seg_vs_int8": float(d_u - d0_eval)},
        }
        row["waterfill"]["net_delta_S_advisory"] = (row["waterfill"]["delta_rate_S"]
                                                    + row["waterfill"]["delta_seg_S"])
        operating_points.append(row)
        print(f"[#336] mean_bits={mb}: waterfill {by['weights_total']}B d_seg={d_w:.6f} "
              f"(Δd={d_w - d0_eval:+.6f}, Δbytes={row['waterfill']['delta_bytes_vs_int8']:+d}, "
              f"netΔS_adv={row['waterfill']['net_delta_S_advisory']:+.5f}) | uniform int{u} "
              f"{by_u['weights_total']}B d_seg={d_u:.6f} {_rss_str()}", flush=True)
        print(f"        nbits={alloc.nbits}", flush=True)
        state["ops"][op_key] = row
        _unit_done()

    result = {
        "task": "#336 / FEED-07b row 5 — #157-APPLY sensitivity bit-alloc on witness weights",
        "authority": (f"[macOS-CPU advisory] NON-PROMOTABLE — probe n{len(probe_ids)} / "
                      f"eval n{len(eval_ids)} evenly-spaced subsets, witness-alone surface"),
        "pointer": "0.19110 UNMOVED (MEANS)",
        "utc": datetime.now(UTC).isoformat(),
        "ckpt": {"dir": str(args.ckpt_dir), "npz": args.npz_name,
                 "provenance": args.ckpt_provenance},
        "reused_157_surfaces": ["tac.frontier_exact_bitalloc.waterfill_bit_allocation",
                                "tac.frontier_exact_bitalloc.lam_for_target_mean_bits",
                                "tac.frontier_exact_bitalloc.CombinedTensorSensitivity"],
        "sensitivity_probe": {"probe_bits": pb, "probe_pair_ids": probe_ids,
                              "baseline_d_seg_probe": float(d0_probe), "per_tensor": sens_rows,
                              "method": (f"MEASURED int8->int{pb} single-tensor d_seg response "
                                         f"(not a gradient proxy); c_t = max(0,Δd)/(2^-{pb} - 2^-8)")},
        "baseline_int8": {"bytes": bytes0, "d_seg_eval": float(d0_eval),
                          "distinct_symbols": distinct0},
        "operating_points": operating_points,
        "eval_pair_ids": eval_ids,
        "what_n600_needs": ("a full #202 byte-close of the chosen allocation (same blob grammar; "
                            "reader unchanged) + n600 verdict through tools/"
                            "levelset_byte_close_and_eval.py, then contest-CPU/CUDA exact eval — "
                            "the ONLY promotable row. This report is the bounded advisory apply."),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[#336] wrote {args.out} ({result['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
