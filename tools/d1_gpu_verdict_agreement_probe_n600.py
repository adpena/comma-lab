#!/usr/bin/env python
"""D1 GPU-vs-CPU verdict AGREEMENT probe — n600, device-forward drift.

[macOS-MLX research-signal] / [macOS-CPU advisory] NON-PROMOTABLE. Pointer 0.19110 UNMOVED.

QUESTION (operator crux-engineering, 2026-07-08): can the MLX-GPU verdict serve as the
IN-TRAINING ADVISORY sensor, replacing the slow (~80-min-stale) CPU-torch anchor cadence
with a minutes cadence? Fit-for-advisory IFF the GPU-vs-CPU-induced error in the sensor
readings (d_seg, d_pose, and the low-margin annulus statistics the part_frac / within_flip /
plateau triggers read) is FAR BELOW the sensor's decision granularity.

WHAT THIS MEASURES (the device-forward DRIFT, checkpoint-independent). The trainer's
verdict scores WITNESS-rendered frames; the DIFFERENCE between the CPU and GPU flavours is
NOT the witness — it is the frozen-scorer FORWARD numerics (torch-CPU vs MLX-GPU) on the
SAME input frames (the preprocess is the SAME torch ``preprocess_input`` in both flavours —
see gpu_verdict_d_seg_argmax_batch / gpu_verdict_d_pose_batch; ONLY the forward kernel
differs). That drift is a property of (scorer weights, input frames, device), governed at
the argmax by LOW-MARGIN pixels. We measure it on the n600 GT-reference frames
(gt_n600.npz) — the most realistic possible frames, sitting EXACTLY at the reference
separatrix, with cached per-pixel margins so we resolve the disagreement BY MARGIN BIN (the
sensor granularity). On GT frames the CPU verdict is ~0 by construction (lstars / gt_poses
ARE the frozen CPU-torch argmax / pose), so every reported DELTA is the pure device drift,
and the low-margin-bin disagreement RATE bounds the induced sensor error at ANY operating
point INCLUDING the witness's low-margin annulus.

SCOPE (honest, per the verdict-scope ladder — verdict_scope=formulation). This is the
device-forward-drift FORMULATION of the agreement question. It does NOT re-render the two
frozen witness checkpoints (mod32cap EMA-BEST + run-1 EMA-BEST): no reusable checkpoint->
render->verdict driver exists in-tree, and reconstructing the full compose (self-orient +
chroma + lane-render-band + structured-init) faithfully BESIDE the untouchable live run
(pid 63069) is a fidelity + GPU-contention + memory risk that would not sharpen the
device-drift answer (which is checkpoint-independent). The witness-frame sensor deltas at
the actual witness operating point are the OWED confirmation (reactivation: build the
render driver). See the landing memo for the full rationale.

CONFOUND CONTROL / AUTHORITY. Two flavours, SAME rendered/GT frames, SAME torch preprocess;
only the forward device differs. Both are ADVISORY (CLAUDE.md: MLX/MPS is NEVER a score) —
only a byte-closed upstream/evaluate.py n600 exact row moves the pointer. Instrument-validity
gates (pre-registered): (a) GPU forward run-to-run bit-identity (double-forward on the first
chunk) — else the GPU verdict is non-reproducible and unfit regardless; (b) CPU verdict ~0
vs the cache (confirms the GT-reference construction / no cache drift).

Governed-launch P0: routes through tools/safe_run.py (system memory governor) — this script
calls assert_governed_admission() and REFUSES a raw launch when enforce is armed. Chunked-
resumable foreground (atomic tmp+replace state); free-RAM gate before each chunk. Wall-clock
is NOT a deliverable (contended machine).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_DIR = REPO / "experiments/results/d1_gpu_verdict_agreement_probe_20260708"
STATE = OUT_DIR / "probe_state.ckpt.npz"
OUT = OUT_DIR / "d1_gpu_verdict_agreement_n600_20260708.json"

# margin-bin edges = the sensor granularity (low bins = the annulus regime the triggers read)
MARGIN_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])
NBINS = len(MARGIN_EDGES) - 1
NCLASS = 5  # comma10k canonical order [Road, Lane, Undrivable, Movable, MyCar]
VBATCH = int(os.environ.get("D1_VBATCH", "8"))
SAVE_EVERY = int(os.environ.get("D1_SAVE_EVERY", "24"))
MIN_FREE_GIB = float(os.environ.get("D1_MIN_FREE_GIB", "12.0"))  # footprint ~6 GiB + headroom;
# the pre-registered 15 GiB guard was sized for the #205 ~66 GiB unchunked verdict spike — this
# probe's resident is the ~4.8 GiB GT cache + scorers, so 12 GiB (footprint + ~6 GiB headroom)
# protects the live run; safe_run's memory governor is the AUTHORITATIVE P0 gate on top.

# ── pre-registered fit-for-advisory thresholds (written BEFORE measurement) ──
# Sensor operating point: witness verdict d_seg ~0.005; the sensor reads d_seg trends near
# ~1e-4 and the annulus part_frac / within_flip are low-margin (margin<1) statistics.
FIT_MEAN_ABS_DSEG = 5e-5        # device d_seg drift < ~1% of the 0.005 operating point
FIT_LOWMARGIN_DISAGREE_RATE = 1e-3   # < 0.1% of margin<1 pixels flip between devices
FIT_MEAN_ABS_DPOSE = 1e-6
FIT_MAX_ABS_DPOSE = 1e-4


def _free_gib() -> float:
    try:
        try:
            from tools.mem_basis import conservative_free_gib
        except Exception:
            from mem_basis import conservative_free_gib  # type: ignore
        return conservative_free_gib(default=float("inf"))
    except Exception:
        return float("inf")


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-seconds", type=float, default=480.0,
                    help="exit cleanly (state saved) after this many seconds; re-invoke to resume.")
    ap.add_argument("--num-pairs", type=int, default=600, help="n600 discipline: keep 600.")
    args = ap.parse_args()

    # P0 governed-admission gate (raw launch fails closed when enforce is armed).
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("d1_gpu_verdict_agreement_probe_n600")

    free = _free_gib()
    if free < MIN_FREE_GIB:
        print(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB} GiB (live #205 run protection)",
              flush=True)
        sys.exit(3)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx  # noqa: E402

    from train_witness_realized_through_R_mlx import (  # noqa: E402  (exact trainer primitives)
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_argmax_batch,
        gpu_verdict_d_pose_batch,
        gpu_verdict_d_seg_argmax_batch,
        load_gt_from_cache,
    )
    from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    # Replicate the trainer's --mlx-device gpu: the GPU verdict forwards run on the process
    # default MLX device. (Verdict is forward-only — the fused-R scatter-VJP determinism wall,
    # memory L70, applies to the R backward, NOT to the scorer forward.)
    mx.set_default_device(mx.Device(mx.gpu))
    dev = str(mx.default_device())

    gt, seg_cpu, posenet_cpu = load_gt_from_cache(CACHE, int(args.num_pairs))
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")
    P = int(gt.n_pairs)
    print(f"[{time.time() - t0:.1f}s] loaded gt + cpu scorers + mlx adapter; P={P} "
          f"mlx_device={dev} vbatch={VBATCH} peak_rss={_peak_rss_gib():.1f}GiB", flush=True)

    st: dict[str, np.ndarray] = {
        "cpu_dseg": np.full(P, np.nan),
        "gpu_dseg": np.full(P, np.nan),
        "cpu_dpose": np.full(P, np.nan),
        "gpu_dpose": np.full(P, np.nan),
        "disagree_px": np.full(P, np.nan),           # cpu_realized != gpu_realized (device drift)
        "cpu_vs_lstar_px": np.full(P, np.nan),       # validity: should be ~0 on GT frames
        "total_px": np.full(P, np.nan),
        "disagree_bin": np.full((P, NBINS), np.nan), # disagreement pixels per margin bin
        "binpx": np.full((P, NBINS), np.nan),        # denominator: pixels per margin bin
        "disagree_class": np.full((P, NCLASS), np.nan),
        "classpx": np.full((P, NCLASS), np.nan),
    }
    done = np.zeros(P, bool)
    determinism = {"checked": False, "bit_identical": None}
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for k in st:
            if k in ck.files:
                st[k] = ck[k]
        done = ck["done"].astype(bool)
        if "determinism_ok" in ck.files:
            determinism = {"checked": True, "bit_identical": bool(ck["determinism_ok"])}
        print(f"[resume] {int(done.sum())}/{P} done", flush=True)

    def save() -> None:
        payload = dict(st)
        payload["done"] = done
        if determinism["checked"]:
            payload["determinism_ok"] = np.asarray(bool(determinism["bit_identical"]))
        tmp = STATE.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(STATE)

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s0 in range(0, len(todo), VBATCH):
        if _free_gib() < MIN_FREE_GIB:
            save()
            print(f"[mem-abort] free RAM < {MIN_FREE_GIB} GiB mid-run; state saved, "
                  f"{int(done.sum())}/{P} done. Re-invoke when memory frees.", flush=True)
            sys.exit(3)
        chunk = todo[s0:s0 + VBATCH]
        f1s = [gt.gt_f1[pi] for pi in chunk]
        f0s = [gt.gt_f0[pi] for pi in chunk]
        lst = [gt.lstars[pi] for pi in chunk]
        mrg = [gt.margins[pi] for pi in chunk]
        poses = [gt.gt_poses[pi] for pi in chunk]

        # --- d_seg: CPU-torch forward (authority) vs MLX-GPU forward, SAME frames ---
        c_ds, c_real = cpu_verdict_d_seg_argmax_batch(seg_cpu, f1s, lst)
        g_ds, g_real = gpu_verdict_d_seg_argmax_batch(adapter.segnet, seg_cpu, f1s, lst)
        if not determinism["checked"]:
            # GPU run-to-run bit-identity floor (re-forward the IDENTICAL chunk).
            _g_ds2, g_real2 = gpu_verdict_d_seg_argmax_batch(adapter.segnet, seg_cpu, f1s, lst)
            bit = all(bool(np.array_equal(np.asarray(a), np.asarray(b)))
                      for a, b in zip(g_real, g_real2)) and (list(g_ds) == list(_g_ds2))
            determinism = {"checked": True, "bit_identical": bool(bit)}
            print(f"[validity] gpu double-forward bit_identical={bit}", flush=True)
            del g_real2, _g_ds2

        # --- d_pose: CPU-torch vs MLX-GPU PoseNet, SAME pairs ---
        c_dp = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, poses)
        g_dp = gpu_verdict_d_pose_batch(adapter.posenet, posenet_cpu, f0s, f1s, poses)

        for j, pi in enumerate(chunk):
            cr = np.asarray(c_real[j], np.int64)
            gr = np.asarray(g_real[j], np.int64)
            ls = np.asarray(lst[j], np.int64)
            mg = np.asarray(mrg[j], np.float64)
            dis = cr != gr
            st["cpu_dseg"][pi] = float(c_ds[j])
            st["gpu_dseg"][pi] = float(g_ds[j])
            st["cpu_dpose"][pi] = float(c_dp[j])
            st["gpu_dpose"][pi] = float(g_dp[j])
            st["disagree_px"][pi] = float(np.count_nonzero(dis))
            st["cpu_vs_lstar_px"][pi] = float(np.count_nonzero(cr != ls))
            st["total_px"][pi] = float(cr.size)
            for bi in range(NBINS):
                sel = (mg >= MARGIN_EDGES[bi]) & (mg < MARGIN_EDGES[bi + 1])
                st["binpx"][pi, bi] = float(sel.sum())
                st["disagree_bin"][pi, bi] = float((dis & sel).sum())
            for c in range(NCLASS):
                selc = ls == c
                st["classpx"][pi, c] = float(selc.sum())
                st["disagree_class"][pi, c] = float((dis & selc).sum())
            done[pi] = True

        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd
            dpx = float(np.nansum(st["disagree_px"]))
            tpx = float(np.nansum(st["total_px"]))
            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | seg_disagree_frac={dpx / max(tpx, 1):.3e} "
                  f"| rss={_peak_rss_gib():.1f}GiB", flush=True)
        if time.time() - t0 > float(args.chunk_seconds) and nd < P:
            save()
            print(f"[chunk-exit] {nd}/{P} done at {time.time() - t0:.1f}s; re-invoke to resume. "
                  f"peak_rss={_peak_rss_gib():.2f}GiB", flush=True)
            sys.exit(0)
    save()

    # ── finalize agreement metrics ──
    cpu_dseg = st["cpu_dseg"]
    gpu_dseg = st["gpu_dseg"]
    cpu_dpose = st["cpu_dpose"]
    gpu_dpose = st["gpu_dpose"]
    d_dseg = gpu_dseg - cpu_dseg
    d_dpose = gpu_dpose - cpu_dpose

    disagree_bin_tot = np.nansum(st["disagree_bin"], axis=0)  # (NBINS,)
    binpx_tot = np.nansum(st["binpx"], axis=0)
    bin_rate = [float(disagree_bin_tot[b] / binpx_tot[b]) if binpx_tot[b] > 0 else None
                for b in range(NBINS)]
    disagree_class_tot = np.nansum(st["disagree_class"], axis=0)
    classpx_tot = np.nansum(st["classpx"], axis=0)
    class_rate = [float(disagree_class_tot[c] / classpx_tot[c]) if classpx_tot[c] > 0 else None
                  for c in range(NCLASS)]
    total_disagree = float(np.nansum(st["disagree_px"]))
    total_px = float(np.nansum(st["total_px"]))
    # low-margin (margin<1.0) = annulus regime = bins 0,1,2
    lowm_disagree = float(disagree_bin_tot[:3].sum())
    lowm_px = float(binpx_tot[:3].sum())
    lowm_rate = lowm_disagree / lowm_px if lowm_px > 0 else float("nan")

    mean_abs_dseg = float(np.nanmean(np.abs(d_dseg)))
    max_abs_dseg = float(np.nanmax(np.abs(d_dseg)))
    dseg_exact_equal = int(np.count_nonzero(np.abs(d_dseg) < 1e-12))
    mean_abs_dpose = float(np.nanmean(np.abs(d_dpose)))
    max_abs_dpose = float(np.nanmax(np.abs(d_dpose)))
    dpose_exact_equal = int(np.count_nonzero(np.abs(d_dpose) < 1e-12))
    worst_pair_dseg = int(np.nanargmax(np.abs(d_dseg)))
    worst_pair_dpose = int(np.nanargmax(np.abs(d_dpose)))

    class_names = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
    fit = (mean_abs_dseg < FIT_MEAN_ABS_DSEG
           and lowm_rate < FIT_LOWMARGIN_DISAGREE_RATE
           and mean_abs_dpose < FIT_MEAN_ABS_DPOSE
           and max_abs_dpose < FIT_MAX_ABS_DPOSE
           and bool(determinism["bit_identical"]))
    verdict = ("FIT_FOR_ADVISORY_SENSOR" if fit
               else "NOT_FIT_current_mlx_gpu_verdict_formulation_on_this_chip")

    out = {
        "schema": "d1_gpu_verdict_agreement.v1",
        "axis_tags": ["[macOS-MLX research-signal]", "[macOS-CPU advisory]"],
        "promotable": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "task": "GPU(MLX)-vs-CPU(torch) frozen-scorer FORWARD drift on n600 GT-reference frames "
                "(device-forward-drift formulation of the verdict-agreement question)",
        "verdict_scope": "formulation (device-forward drift on GT-reference frames; witness-frame "
                         "render at the two frozen checkpoints is OWED — no reusable render driver)",
        "substrate": str(CACHE.relative_to(REPO)),
        "n_pairs": P,
        "mlx_device": dev,
        "cpu_is_authority_note": "on GT frames the CPU verdict is ~0 by construction (lstars / "
                                 "gt_poses ARE the frozen CPU-torch argmax / pose), so every DELTA "
                                 "below is the pure CPU->GPU device-forward drift",
        "instrument_validity": {
            "gpu_double_forward_bit_identical": bool(determinism["bit_identical"]),
            "cpu_vs_cached_lstar_mean_px": float(np.nanmean(st["cpu_vs_lstar_px"])),
            "cpu_dseg_mean": float(np.nanmean(cpu_dseg)),
            "cpu_dpose_mean": float(np.nanmean(cpu_dpose)),
        },
        "d_seg_agreement": {
            "cpu_mean": float(np.nanmean(cpu_dseg)),
            "gpu_mean": float(np.nanmean(gpu_dseg)),
            "mean_abs_delta_per_pair": mean_abs_dseg,
            "max_abs_delta_per_pair": max_abs_dseg,
            "worst_pair_index": worst_pair_dseg,
            "worst_pair_abs_delta": float(abs(d_dseg[worst_pair_dseg])),
            "n_pairs_exact_equal": dseg_exact_equal,
        },
        "d_pose_agreement": {
            "cpu_mean": float(np.nanmean(cpu_dpose)),
            "gpu_mean": float(np.nanmean(gpu_dpose)),
            "mean_abs_delta_per_pair": mean_abs_dpose,
            "max_abs_delta_per_pair": max_abs_dpose,
            "worst_pair_index": worst_pair_dpose,
            "worst_pair_abs_delta": float(abs(d_dpose[worst_pair_dpose])),
            "n_pairs_exact_equal": dpose_exact_equal,
        },
        "argmax_disagreement": {
            "total_disagree_px": total_disagree,
            "total_px": total_px,
            "total_disagree_fraction": total_disagree / max(total_px, 1),
            "margin_bin_edges": [float(x) for x in MARGIN_EDGES[:-1]] + ["inf"],
            "disagree_rate_by_margin_bin": bin_rate,
            "disagree_px_by_margin_bin": [float(x) for x in disagree_bin_tot],
            "px_by_margin_bin": [float(x) for x in binpx_tot],
            "low_margin_lt1_disagree_rate": lowm_rate,
            "disagree_rate_by_class": {class_names[c]: class_rate[c] for c in range(NCLASS)},
            "disagree_px_by_class": {class_names[c]: float(disagree_class_tot[c])
                                     for c in range(NCLASS)},
        },
        "sensor_level_bridge": {
            "note": "the low-margin (margin<1) disagreement RATE bounds the induced error in the "
                    "annulus part_frac / within_flip triggers at ANY operating point; the witness "
                    "verdict d_seg operates at ~0.005 with the sensor reading trends near ~1e-4",
            "induced_dseg_sensor_error_bound": mean_abs_dseg,
            "annulus_regime_disagree_rate": lowm_rate,
        },
        "pre_registered_thresholds": {
            "mean_abs_dseg_lt": FIT_MEAN_ABS_DSEG,
            "low_margin_disagree_rate_lt": FIT_LOWMARGIN_DISAGREE_RATE,
            "mean_abs_dpose_lt": FIT_MEAN_ABS_DPOSE,
            "max_abs_dpose_lt": FIT_MAX_ABS_DPOSE,
            "require_gpu_double_forward_bit_identical": True,
        },
        "fit_for_advisory_sensor": bool(fit),
        "verdict": verdict,
        "proposed_v7_hybrid_cadence": (
            "IF FIT: GPU verdict at the FAST cadence (--verdict-device gpu, minutes) + a CPU-torch "
            "ANCHOR (paired_anchor_verdict) at the SLOW cadence (checkpoint epochs, e.g. --verdict-"
            "anchor-every N) as the positive-control sentinel + comparability baseline; the "
            "controllers (nucleus-guard / ladder-homotopy) stay on CPU authority per gpu_verdict_"
            "conflicts. IF NOT FIT: keep the CPU verdict; the failing metric is the reactivation "
            "target."),
        "peak_rss_gib": _peak_rss_gib(),
        "note": "means not ends: advisory device-drift row; pointer 0.19110 moves ONLY via "
                "upstream/evaluate.py on exact archive bytes",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE -> {OUT}", flush=True)
    print(f"  verdict={verdict} mean|Δdseg|={mean_abs_dseg:.3e} lowm_rate={lowm_rate:.3e} "
          f"mean|Δdpose|={mean_abs_dpose:.3e} gpu_bit_identical={determinism['bit_identical']}",
          flush=True)


if __name__ == "__main__":
    main()
