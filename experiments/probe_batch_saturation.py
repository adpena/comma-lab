#!/usr/bin/env python3
"""Batch-saturation throughput probe: s/ep vs batch_size on 128GB MPS.

Runs the REAL launcher at several batch sizes with a tiny epoch budget on scratch
dirs, records seconds/epoch + OOM, and writes a JSON throughput surface. $0, MPS,
advisory (throughput only, no score claim). The anchor is stopped during this.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGET = 30            # ~30 total epochs across stages (mostly stage1) — enough for s/ep
BATCHES = [64, 150, 300, 600]
OUT = REPO / ".omx/research/batch_saturation_throughput_20260623.json"
results = []
for bs in BATCHES:
    scratch = REPO / f".omx/tmp/batchprobe_bs{bs}"
    cmd = [
        ".venv/bin/python", "-u", "experiments/launch_split_by_head_basin.py",
        "--no-split-by-head", "--train-device", "mps", "--device", "cpu",
        "--base-channels", "36", "--latent-dim", "28", "--n-pairs", "600",
        "--targets-cache", "experiments/results/capstone_gt_targets_cache",
        "--batch-size", str(bs), "--batch-lr-scale", "sqrt",
        "--muon-lr-floor-fix", "--defer-batch-sync",
        "--total-epoch-budget", str(BUDGET),
        "--checkpoint-every-epochs", "999999",  # no checkpoint during probe
        "--out-dir", str(scratch),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = time.time() - t0
    oom = "out of memory" in (p.stderr + p.stdout).lower() or "mps" in (p.stderr).lower() and "alloc" in p.stderr.lower()
    rc = p.returncode
    spe = None; n_ep = None; last_dseg = None
    summ = scratch / "torch_vehicle_summary.json"
    traj = scratch / "torch_vehicle_trajectory.jsonl"
    if summ.exists():
        sm = json.loads(summ.read_text())
        n_ep = sm.get("n_records"); wc = sm.get("wall_clock_s")
        if n_ep and wc: spe = wc / n_ep
        le = sm.get("last_eval") or {}; last_dseg = le.get("d_seg")
    results.append({"batch_size": bs, "rc": rc, "oom": bool(oom), "wall_s": round(wall,1),
                    "n_epochs": n_ep, "s_per_ep": round(spe,3) if spe else None,
                    "steps_per_epoch": (600 + bs - 1)//bs, "last_d_seg": last_dseg,
                    "stderr_tail": p.stderr[-400:] if rc != 0 else ""})
    print(json.dumps(results[-1]), flush=True)
    # clean scratch
    import shutil
    if scratch.exists(): shutil.rmtree(scratch, ignore_errors=True)
OUT.write_text(json.dumps({"schema":"batch_saturation_throughput.v1","authority":"[advisory] throughput-only NON-PROMOTABLE",
                           "base_channels":36,"n_pairs":600,"budget":BUDGET,"results":results}, indent=2))
print("WROTE", OUT, flush=True)
