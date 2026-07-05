#!/usr/bin/env python
"""$0 lambda_pre HVP PROBE (EIK-STAB build 4; litsweep DOMAIN-2 rank-1 action, sweep lever #1).

Measures the Adam-PRECONDITIONED sharpness lambda_pre = lambda_max(P^-1/2 H P^-1/2) on the ep100
resume snapshot (n24 slice; moments RESTORED — the probe's design point) via the LEVELSET
trainer's own ``--lambda-pre-probe-iters`` mode (preconditioned power iteration, forward-diff
HVPs over the full 24-pair batch gradient, fp64 accumulation, central-difference consistency
check; the trainer EXITS before any training step).

THE LAW UNDER TEST (Cohen et al. arXiv 2207.14484, Adam-family EoS): stability iff
    eta * lambda_pre  <~  2*(1+b1)/(1-b1)  =  38   (b1 = 0.9)
The measured lr bracket [5e-4 stable-fastest, 9.1e-4 unstable] predicts the ep100 basin sits at
    lambda_pre in [38/9.1e-4, 38/5e-4] = [4.2e4, 7.6e4]   (a factor-1.8 window).
IN-WINDOW => the bracket upgrades to the candidate law `eos_adam_preconditioned_threshold_v1`
(FORMALIZATION_PENDING; register ONLY on a clean anchor) and per-stage lr caps become DERIVED:
eta_max(t) ~= 38/lambda_pre(t) * margin.

AXIS / NO-FAKE: [n24 advisory -- mechanism probe, NOT n600 evidence]. The probe reuses the
trainer's OWN resume + loss path (nothing reimplemented); pointer 0.19110 UNMOVED (means).

Foreground-only; no daemons. Usage:
    .venv/bin/python experiments/probe_lambda_pre_hvp.py \
        --out-dir experiments/results/eik_stab_build_20260705/lambda_pre [--iters 12] \
        [--mlx-device gpu|cpu]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_stepping_probe():
    p = REPO / "experiments" / "probe_resume_stepping_instability.py"
    spec = importlib.util.spec_from_file_location("_stepping_probe_for_lambda_pre", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_stepping_probe_for_lambda_pre"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "experiments/results/eik_stab_build_20260705/lambda_pre")
    ap.add_argument("--iters", type=int, default=12, help="power iterations (each = 1 FD grad)")
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    ap.add_argument("--mlx-device", type=str, default="gpu", choices=("gpu", "cpu"),
                    help="gpu = fast (FD noise checked by the central-diff consistency row); "
                    "cpu = the bit-exact fallback if fwd_vs_central_rel is large.")
    ap.add_argument("--accum-pairs", type=int, default=8)
    ap.add_argument("--timeout-s", type=int, default=5400)
    args = ap.parse_args(argv)

    sp = _load_stepping_probe()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sp.SNAPSHOT.exists():
        print(f"FATAL: snapshot missing at {sp.SNAPSHOT}", file=sys.stderr)
        return 2
    snap24 = out_dir / "resume_state_ep100_n24.npz"
    stats = sp.slice_snapshot(sp.SNAPSHOT, snap24, 24, drop_opt=False)
    print(json.dumps({"stage": "snapshot_slice", "with_opt": stats}), flush=True)

    # baseline_v3 semantics (bd 0.2, seed compose 0 at ep>=101, moments restored, lr as-scheduled
    # ~9.1e-4 at ep101 — eta is REPORTED by the trainer row) + the probe flags.
    cfg = sp.ARMS["baseline_v3"]
    argv_t = sp._base_argv(
        out_dir / "trainer_out", snap24, epochs=103, accum=args.accum_pairs,
        seed_anneal_epochs=cfg["seed_anneal"], bd_weight=cfg["bd"], lr=cfg["lr"],
        lr_end=cfg["lr_end"], tau_start=cfg["tau_start"], band_start=cfg["band_start"],
        persist_warmup=cfg["persist_warmup"],
        extra=["--lambda-pre-probe-iters", str(args.iters),
               "--lambda-pre-probe-fd-eps", f"{args.fd_eps:g}"])
    # device override (the base argv pins gpu)
    di = argv_t.index("--mlx-device")
    argv_t[di + 1] = args.mlx_device
    (out_dir / "probe.argv.json").write_text(json.dumps(argv_t, indent=1))
    log_path = out_dir / "probe.log"
    env = dict(os.environ)
    env.setdefault("TAC_MLX_CUSTOM_GROUPED_BACKWARD", "1")
    print(json.dumps({"stage": "probe_start", "log": str(log_path),
                      "device": args.mlx_device, "iters": args.iters}), flush=True)
    with log_path.open("w") as lf:
        try:
            rc = subprocess.run(argv_t, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=args.timeout_s, env=env, cwd=str(REPO)).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            print(json.dumps({"stage": "probe_timeout"}), flush=True)

    iters_rows, final = [], None
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stage") == "lambda_pre_iter":
            iters_rows.append(row)
        elif row.get("stage") == "lambda_pre":
            final = row
    report = {
        "axis": "[n24 advisory -- mechanism probe, NOT n600 evidence]",
        "rc": rc, "device": args.mlx_device, "iters": iters_rows, "final": final,
        "pointer": "0.19110 UNMOVED",
    }
    rp = out_dir / "lambda_pre_probe_report.json"
    rp.write_text(json.dumps(report, indent=1))
    print(json.dumps({"stage": "report_written", "path": str(rp)}), flush=True)
    if final is not None:
        print(f"\nlambda_pre = {final['lambda_pre']:.6g}  (central check {final['lambda_pre_central_check']:.6g}, "
              f"rel {final.get('fwd_vs_central_rel')})")
        print(f"eta(ep101) = {final['eta']:.6g}  ->  pi_EoS = eta*lambda_pre/38 = {final['pi_eos']:.4g}")
        print(f"bracket [4.2e4, 7.6e4]: in_window = {final['in_window']}")
        print(f"eta_max from law = {final.get('eta_max_from_law')}")
    else:
        print("NO lambda_pre row parsed -- inspect probe.log", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
