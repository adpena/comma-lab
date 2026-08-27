# SPDX-License-Identifier: MIT
"""DECISIVE saturation test: does the M5 Max GPU+unified-memory support a PARALLEL
FLEET of MLX through-R witness arms, or is one arm already compute-saturating?

Operator 2026-06-25: "We have 128 GB and I doubt it's saturated." The single N=1
through-R arm uses ~11GB of 128GB; ~98GB is free. This probe MEASURES (NO-FAKE)
whether concurrent independent MLX arms extract MORE aggregate throughput from the
idle GPU+memory:

  for K in [1, 2, 4, 8]:
    launch K identical SHORT timing arms (n24, shared GT cache, gpu) CONCURRENTLY
    measure: aggregate pairs/s = K * n_pairs * epochs / max(per-arm wall)
             peak RSS across the K arms (proxy for unified-memory footprint)

If aggregate pairs/s SCALES with K -> the GPU is under-utilized by one arm, a fleet
wins (report N* where it plateaus / memory caps). If aggregate pairs/s is FLAT ->
one arm already compute-saturates the GPU; the idle memory's best use is a single
BIGGER-capacity arm, not a fleet.

The arms run ON TOP OF the 2 live arms (witcap_devAB_cpu + mlx_throughR_n600) --
so the measurement reflects the REAL marginal headroom, not an idle machine.

Disk: per-arm out-dirs under experiments/results/_fleet_conc_probe/, auto-cleaned.
NO score claim; this is a throughput measurement only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "experiments/train_witness_realized_through_R_mlx.py"
_EP_RE = re.compile(r'"ep_wall_s":\s*([0-9.]+)')
_STAGE_RE = re.compile(r'"stage":\s*"(gt_cache_load|gt_precompute)".*?"secs":\s*([0-9.]+)')


def _free_gb() -> float:
    """AVAILABLE (reclaimable) memory in GB. On macOS raw 'Pages free' is kept near
    zero by design; the true headroom = free + inactive + speculative + purgeable
    (all reclaimable without paging out anonymous memory)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout  # subprocess-no-check-OK: best-effort vm_stat read; failure degrades to zero-headroom (conservative)
    m = re.search(r"page size of (\d+) bytes", out)
    pg = int(m.group(1)) if m else 16384
    vals = {"free": 0, "inactive": 0, "speculative": 0, "purgeable": 0}
    for line in out.splitlines():
        if line.startswith("Pages free:"):
            vals["free"] = int(line.split()[-1].rstrip("."))
        elif line.startswith("Pages inactive:"):
            vals["inactive"] = int(line.split()[-1].rstrip("."))
        elif line.startswith("Pages speculative:"):
            vals["speculative"] = int(line.split()[-1].rstrip("."))
        elif line.startswith("Pages purgeable:"):
            vals["purgeable"] = int(line.split()[-1].rstrip("."))
    return sum(vals.values()) * pg / 1e9


def _peak_rss_gb(pids: list[int]) -> float:
    """Sum of RSS (GB) across the given pids, sampled now (ps -o rss)."""
    if not pids:
        return 0.0
    try:
        out = subprocess.run(  # subprocess-no-check-OK: best-effort ps RSS sample; failure degrades via the except arm
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True,
        ).stdout
        kb = sum(int(x) for x in out.split() if x.strip().isdigit())
        return kb / 1e6
    except Exception:
        return 0.0


def _arm_cmd(out_dir: Path, args: argparse.Namespace, seed: int) -> list[str]:
    return [
        ".venv/bin/python", "-u", str(TRAINER),
        "--out-dir", str(out_dir),
        "--num-pairs", str(args.num_pairs),
        "--epochs", str(args.epochs),
        "--eval-every", str(args.epochs + 1),  # skip the expensive CPU verdict; we only time the GPU loop
        "--render-h", str(args.render_h), "--render-w", str(args.render_w),
        "--hidden-dim", str(args.hidden_dim), "--n-hidden", str(args.n_hidden),
        "--mod-dim", str(args.mod_dim), "--n-fourier", str(args.n_fourier),
        "--gt-cache", str(args.gt_cache),
        "--mlx-device", "gpu", "--seed", str(seed),
    ]


def run_concurrency(K: int, args: argparse.Namespace, base_dir: Path) -> dict:
    import os

    env = dict(os.environ)
    env["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    procs = []
    logs = []
    for i in range(K):
        od = base_dir / f"K{K}_arm{i}"
        od.mkdir(parents=True, exist_ok=True)
        lf = open(od / "arm.log", "w")
        logs.append((od, lf))
        p = subprocess.Popen(_arm_cmd(od, args, seed=i), cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p)

    pids = [p.pid for p in procs]
    # Sample peak RSS while running.
    peak_rss = 0.0
    min_free = 999.0
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        # include child python pids (Popen pid may be the wrapper); sample full tree via pgrep is overkill --
        # the Popen pid IS the python process here (direct exec, no shell).
        peak_rss = max(peak_rss, _peak_rss_gb([p.pid for p in procs if p.poll() is None]))
        min_free = min(min_free, _free_gb())
        time.sleep(2.0)
        if time.time() - t0 > args.timeout_s:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            break
    for p in procs:
        p.wait()
    for _od, lf in logs:
        lf.close()

    # Parse per-arm ep_wall_s (median) + GT-load secs.
    arm_eps = []
    gt_secs = []
    for od, _lf in logs:
        txt = (od / "arm.log").read_text()
        eps = [float(x) for x in _EP_RE.findall(txt)]
        if eps:
            eps_sorted = sorted(eps)
            arm_eps.append(eps_sorted[len(eps_sorted) // 2])  # median
        gm = _STAGE_RE.search(txt)
        if gm:
            gt_secs.append(float(gm.group(2)))
    if not arm_eps:
        return {"K": K, "error": "no ep_wall_s parsed (arms may have crashed)", "peak_rss_gb": round(peak_rss, 2)}

    # Aggregate throughput: each arm does n_pairs gradient steps per epoch.
    # per-arm pairs/s = n_pairs / median_ep_wall_s ; aggregate = sum over arms.
    per_arm_pps = [args.num_pairs / e for e in arm_eps]
    agg_pps = sum(per_arm_pps)
    return {
        "K": K,
        "median_ep_wall_s_per_arm": round(float(sum(arm_eps) / len(arm_eps)), 3),
        "per_arm_pairs_per_s": round(float(sum(per_arm_pps) / len(per_arm_pps)), 2),
        "aggregate_pairs_per_s": round(agg_pps, 2),
        "n_arms_reporting": len(arm_eps),
        "gt_load_secs_median": round(float(sorted(gt_secs)[len(gt_secs) // 2]), 1) if gt_secs else None,
        "peak_rss_gb": round(peak_rss, 2),
        "min_free_gb_during": round(min_free, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MLX fleet concurrency scaling probe (saturation test)")
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--render-h", type=int, default=192)
    ap.add_argument("--render-w", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=192)
    ap.add_argument("--n-hidden", type=int, default=6)
    ap.add_argument("--mod-dim", type=int, default=64)
    ap.add_argument("--n-fourier", type=int, default=24)
    ap.add_argument("--concurrencies", type=str, default="1,2,4,8")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--out-json", type=str, default="experiments/results/_fleet_conc_probe/scaling.json")
    args = ap.parse_args(argv)

    base = REPO / "experiments/results/_fleet_conc_probe"
    base.mkdir(parents=True, exist_ok=True)
    Ks = [int(x) for x in args.concurrencies.split(",")]
    rows = []
    print(json.dumps({"probe": "mlx_fleet_concurrency_scaling", "free_gb_start": round(_free_gb(), 1),
                      "concurrencies": Ks, "num_pairs": args.num_pairs, "epochs": args.epochs}), flush=True)
    for K in Ks:
        free = _free_gb()
        if free < 15.0:
            print(json.dumps({"K": K, "skipped": "fail-closed: free < 15GB", "free_gb": round(free, 1)}), flush=True)
            break
        row = run_concurrency(K, args, base)
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Scaling verdict: compare aggregate pairs/s at max K vs K=1.
    verdict = {"scaling": "indeterminate"}
    if len(rows) >= 2 and "aggregate_pairs_per_s" in rows[0] and "aggregate_pairs_per_s" in rows[-1]:
        base_pps = rows[0]["aggregate_pairs_per_s"]
        max_pps = max(r.get("aggregate_pairs_per_s", 0) for r in rows)
        speedup = max_pps / base_pps if base_pps else 0
        best_K = max(rows, key=lambda r: r.get("aggregate_pairs_per_s", 0))["K"]
        verdict = {
            "scaling": "FLEET_WINS" if speedup >= 1.3 else "COMPUTE_SATURATED_BY_ONE_ARM",
            "aggregate_speedup_maxK_over_K1": round(speedup, 2),
            "best_concurrency_N_star": best_K,
            "base_K1_aggregate_pps": base_pps,
            "best_aggregate_pps": max_pps,
        }
    out = {"rows": rows, "verdict": verdict, "free_gb_end": round(_free_gb(), 1)}
    op = REPO / args.out_json
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print("\n=== SCALING VERDICT ===")
    print(json.dumps(verdict, indent=2))
    # Clean up the per-arm scratch dirs (the scaling.json is the durable artifact).
    for K in Ks:
        for d in base.glob(f"K{K}_arm*"):
            shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
