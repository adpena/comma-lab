# SPDX-License-Identifier: MIT
# HISTORICAL_RECIPE_ONLY: frozen launcher record for a completed lane; not a live deploy path (codebase-drift relocation 2026-08-25).
"""Launch the PARALLEL FLEET of MLX through-R witness arms (operator 2026-06-25).

Decisive saturation test (probe_mlx_fleet_concurrency_scaling.py) verdict:
FLEET_WINS -- aggregate GPU throughput scales from ~1-4 pairs/s (1 arm) to
~10-11 pairs/s and plateaus around concurrency N*~4 (K=8 adds memory but little
throughput, and at K=8 peak RSS hit ~49GB with available near the 15GB floor).
So the idle 98GB+GPU DOES support a fleet: run ~4 concurrent arms ON TOP OF the
2 live arms to explore the seg/pose-trade-resolution + capacity space in parallel
(min wall-clock), each SHARING one GT cache (no per-arm 480s recompute).

This launcher enumerates a focused config grid (the trade-resolution levers that
matter: capacity hidden_dim x the w_seg/w_pose balance x render-res), then spawns
each as a DURABLE detached daemon (tools/spawn_durable_daemon.py -- no orphan).
It launches at most --max-concurrent NEW arms; the rest are recorded as a queued
plan (a follow-on wave launches them as the first wave's arms finish).

MEMORY GOVERNANCE (review-fix CRITICAL A -- this launcher used to BYPASS the
governed-admission apparatus): every arm's spawn now passes
``--projected-peak-gib`` (from --per-arm-est-gb) AND ``--rss-cap-mb``
(projection x 1.3 safety factor) to spawn_durable_daemon, so each arm (a) is
admission-gated by the AUTHORITATIVE system memory governor (the P0 SUM-over-RAM
crash gate + TOCTOU-locked pending reservation), (b) lands in the registry with
a REAL projected_peak_gib (the governor's growth accounting no longer assumes
zero future growth for fleet arms), and (c) carries a per-arm safe_run RSS cap
(defense-in-depth layer 3). The old hand-rolled vm_stat available-vs-flat-est
check is DEMOTED TO ADVISORY (printed, never gating): the daemon's
_system_admission_gate is the single authority; a daemon REFUSAL (nonzero rc)
queues the arm for the next wave exactly like any other launch failure.

NO score claim: each arm's realized d_seg/d_pose VERDICT is the FROZEN CPU-torch
authority (in the trainer). The fleet EXPLORES; the verdict per arm is advisory
until a byte-closed exact-eval row lands. Pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = "experiments/train_witness_realized_through_R_mlx.py"
SPAWN = "tools/spawn_durable_daemon.py"

# Per-arm RSS cap = projected peak x this factor (headroom above the projection so the cap is a
# runaway backstop, not a scheduler): a fleet arm that balloons past 1.3x its own projection is a
# bug and safe_run SIGKILLs its group before it can threaten the machine.
RSS_CAP_SAFETY_FACTOR = 1.3


def _available_gb() -> float:
    """ADVISORY-ONLY vm_stat read (review-fix CRITICAL A: this hand-rolled check used to be the
    fleet's ONLY memory gate; the daemon's _system_admission_gate is now the authority)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    pg = int(m.group(1)) if m else 16384
    tot = 0
    for key in ("Pages free:", "Pages inactive:", "Pages speculative:", "Pages purgeable:"):
        for line in out.splitlines():
            if line.startswith(key):
                tot += int(line.split()[-1].rstrip("."))
                break
    return tot * pg / 1e9


@dataclass(frozen=True)
class ArmCfg:
    label: str
    hidden_dim: int
    n_hidden: int
    mod_dim: int
    render_h: int
    render_w: int
    w_seg: float
    w_pose: float
    seed: int
    witness_bytes: int

    def out_dir(self) -> str:
        return f"experiments/results/mlx_fleet_{self.label}"

    def cmd(self, num_pairs: int, epochs: int, eval_every: int, gt_cache: str) -> list[str]:
        return [
            ".venv/bin/python", "-u", TRAINER,
            "--out-dir", self.out_dir(),
            "--num-pairs", str(num_pairs),
            "--epochs", str(epochs),
            "--eval-every", str(eval_every),
            "--render-h", str(self.render_h), "--render-w", str(self.render_w),
            "--hidden-dim", str(self.hidden_dim), "--n-hidden", str(self.n_hidden),
            "--mod-dim", str(self.mod_dim), "--n-fourier", "24",
            "--w-seg", str(self.w_seg), "--w-pose", str(self.w_pose),
            "--witness-bytes", str(self.witness_bytes),
            "--gt-cache", gt_cache,
            "--mlx-device", "gpu", "--seed", str(self.seed),
        ]


def build_grid() -> list[ArmCfg]:
    """The trade-resolution grid: capacity x w_seg/w_pose balance x render-res.

    Capacity (the 'start big' lever, now memory-affordable): hidden_dim in
    {192, 256, 384} at n_hidden 6. The w_seg/w_pose balance resolves the seg/pose
    trade: {1/1 balanced, 3/1 seg-lean, 1/3 pose-lean}. One higher render-res arm
    (288x384) probes whether the realized argmax survives R better at finer grid.
    """
    grid: list[ArmCfg] = []
    # Core capacity x trade sweep at the canonical 192x256 render-res, seed 0.
    caps = [(192, 6, 64, 150_000), (256, 6, 72, 220_000), (384, 6, 96, 360_000)]
    trades = [(1.0, 1.0, "bal"), (3.0, 1.0, "seglean"), (1.0, 3.0, "poselean")]
    for (hd, nh, md, wb), (ws, wp, tname) in itertools.product(caps, trades):
        grid.append(ArmCfg(
            label=f"h{hd}_{tname}_r192", hidden_dim=hd, n_hidden=nh, mod_dim=md,
            render_h=192, render_w=256, w_seg=ws, w_pose=wp, seed=0, witness_bytes=wb,
        ))
    # One finer-render-res arm (balanced, mid capacity) to probe R-survival at 288x384.
    grid.append(ArmCfg(
        label="h256_bal_r288", hidden_dim=256, n_hidden=6, mod_dim=72,
        render_h=288, render_w=384, w_seg=1.0, w_pose=1.0, seed=0, witness_bytes=220_000,
    ))
    return grid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch the MLX through-R witness FLEET")
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument("--num-pairs", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-concurrent", type=int, default=4, help="N* from the scaling probe")
    ap.add_argument("--min-free-gb", type=float, default=18.0,
                    help="ADVISORY low-memory warning threshold (the daemon's system admission "
                         "gate is the authority and can REFUSE an arm regardless)")
    ap.add_argument("--per-arm-est-gb", type=float, default=8.0,
                    help="per-arm projected PEAK RSS (GiB) — registered with the governor via "
                         "spawn_durable_daemon --projected-peak-gib and used to derive the "
                         "per-arm safe_run RSS cap (x%.1f). For launch.sh-based runs prefer "
                         "tools/witness_memory_preflight.project_from_launch_sh; the fleet builds "
                         "trainer argv directly, so this explicit estimate is the projection "
                         "source." % RSS_CAP_SAFETY_FACTOR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not Path(REPO / args.gt_cache).exists():
        print(json.dumps({"error": f"gt-cache not found: {args.gt_cache}"}))
        return 2

    grid = build_grid()
    launched, queued = [], []
    avail = _available_gb()
    print(json.dumps({"fleet": "mlx_witness_through_R", "available_gb": round(avail, 1),
                      "grid_size": len(grid), "max_concurrent": args.max_concurrent,
                      "num_pairs": args.num_pairs, "epochs": args.epochs}), flush=True)

    # Per-arm governance numbers (review-fix CRITICAL A): the projection registers with the
    # governor (growth accounting + admission), the cap bounds the arm's own process group.
    per_arm_proj_gib = float(args.per_arm_est_gb)
    per_arm_rss_cap_mb = int(round(per_arm_proj_gib * RSS_CAP_SAFETY_FACTOR * 1024.0))

    n_active = 0
    for cfg in grid:
        avail = _available_gb()
        if n_active >= args.max_concurrent:
            queued.append(cfg.label)
            continue
        if avail - per_arm_proj_gib < args.min_free_gb:
            # ADVISORY only (review-fix CRITICAL A): the daemon's _system_admission_gate is the
            # authority — it sums REAL used RAM + all tracked jobs' growth-to-peak and REFUSES
            # (rc!=0 -> the arm is queued below) when this arm's projection would not fit.
            print(json.dumps({"advisory_low_memory": cfg.label, "available_gb": round(avail, 1),
                              "advisory_floor_gb": args.min_free_gb,
                              "note": "proceeding — daemon admission gate is the authority"}),
                  flush=True)
        cmd = cfg.cmd(args.num_pairs, args.epochs, args.eval_every, args.gt_cache)
        out_dir = REPO / cfg.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        spawn_cmd = [
            ".venv/bin/python", SPAWN,
            "--log", f"{cfg.out_dir()}/daemon.log",
            "--label", f"fleet_{cfg.label}",
            # governed admission: real projection registered with the governor (admission gate +
            # growth accounting) + the legacy free-floor preflight gets the same accurate number.
            "--projected-peak-gib", str(per_arm_proj_gib),
            "--projected-gb", str(per_arm_proj_gib),
            # defense-in-depth layer 3: per-arm safe_run RSS cap (projection x safety factor).
            "--rss-cap-mb", str(per_arm_rss_cap_mb),
            "--", "env", "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1", *cmd,
        ]
        if args.dry_run:
            print(json.dumps({"dry_run_launch": cfg.label, "cmd": " ".join(spawn_cmd)}), flush=True)
            launched.append(cfg.label)
            n_active += 1
            continue
        r = subprocess.run(spawn_cmd, cwd=str(REPO), capture_output=True, text=True)
        ok = r.returncode == 0
        print(json.dumps({"launched": cfg.label, "ok": ok, "msg": (r.stdout or r.stderr).strip()[-160:]}), flush=True)
        if ok:
            launched.append(cfg.label)
            n_active += 1
        else:
            queued.append(cfg.label)

    plan = {
        "launched": launched, "queued_for_next_wave": queued,
        "available_gb_end": round(_available_gb(), 1),
        "note": "queued arms launch as wave-1 arms finish; re-run this launcher to start the next wave.",
    }
    (REPO / "experiments/results/mlx_fleet_launch_plan.json").write_text(json.dumps(plan, indent=2))
    print("\n=== FLEET LAUNCH PLAN ===")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
