#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""witness_exact_ab — ZERO-NOISE exact-attribution A/B for the level-set witness (task #350, #183).

Runs TWO trainer configs from the SAME seed/start for K epochs under the ``--fused-r-kernel``
determinism guarantee (task #348: the full witness step is cross-process BIT-IDENTICAL with
fused-R ON, confirmed at op-class scope by STAGE 0). Because the numerical-noise floor is ZERO,
ANY weight difference between arm A and arm B is 100% ATTRIBUTABLE to the flag(s) that differ —
there is no run-to-run-noise confound to disentangle. This is the exact A/B that #183 asked for.

TWO reads per run pair:
  * NULL TEST  (A vs A, identical configs): must diverge NOWHERE — final EMA weights BIT-IDENTICAL
    AND per-epoch d_seg history identical. This validates the harness + re-confirms the #348
    guarantee on THIS config (a positive control that fails loudly if determinism ever regresses).
  * FLIP TEST  (A vs A+flag): the divergence is LOCATED — the first epoch where the per-epoch
    d_seg history diverges (the divergence STEP) + the per-tensor L2 diff of the final EMA weights
    (WHICH tensors the flag moved, and by how much).

Reuses the STAGE-0 composite config (``_composite_trainer_argv`` — the mod32cap n600 levers) via
``tools/safe_run.py`` (governed admission + RSS cap — NOT a raw bypass). Bounded pairs/epochs keep
it foreground + memory-safe; determinism + attribution are pair-count-invariant.

AUTHORITY: ``[macOS-MLX research-signal]`` (weight bit-identity) + ``[macOS-CPU advisory]`` (the
per-epoch d_seg history is the CPU-torch verdict). NEVER a score. Pointer UNMOVED.

Usage:
    # NULL TEST (harness + determinism positive control):
    .venv/bin/python tools/witness_exact_ab.py --null --pairs 8 --epochs 3 --device gpu
    # FLIP TEST (attribute one lever):
    .venv/bin/python tools/witness_exact_ab.py --flip-flag --length-weight 0.0 --pairs 8 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tools"))

from mlx_gpu_determinism_probe import (  # noqa: E402
    _composite_trainer_argv,
    _hash_npz,
)


def _run_arm(scratch: Path, label: str, *, pairs: int, epochs: int, gt_cache: str,
             device: str, fused_r: bool, extra_flags: list[str], timeout: int) -> dict:
    """Run one trainer arm K epochs; return {ema_hash, per_tensor(dict), history(list), out_dir}."""
    import numpy as np

    out_dir = scratch / f"arm_{label}"
    argv = _composite_trainer_argv(
        str(out_dir), num_pairs=pairs, epochs=epochs, gt_cache=gt_cache, device=device,
        fused_r=fused_r, timeout_s=max(30, timeout - 12), eval_every=1, extra_flags=extra_flags)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "src"), str(_REPO / "experiments"), str(_REPO / "upstream")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    ema = out_dir / "levelset_witness_ema_mlx.npz"
    # The back-to-back second arm can transiently trip the system memory-governor's accounting
    # fail-safe (psutil-vs-conservative disagreement) — NOT a real OOM. Settle + retry a couple of
    # times so a transient system condition does not masquerade as a divergence result.
    t0 = time.time()
    last_err = "no-ema"
    for attempt in range(3):
        if attempt:
            time.sleep(20)
        p = subprocess.run(argv, env=env, cwd=str(_REPO), capture_output=True, text=True,
                           timeout=timeout)
        if p.returncode == 0 and ema.exists():
            break
        last_err = (p.stderr or "no-ema")[-600:]
    if not ema.exists():
        return {"label": label, "error": last_err, "argv_tail": argv[-8:]}
    per_tensor = {}
    with np.load(ema, allow_pickle=True) as z:
        for k in sorted(z.files):
            if not k.startswith("__") and z[k].dtype.kind in "fiu":
                per_tensor[k] = np.ascontiguousarray(np.asarray(z[k], np.float32))
    history = []
    tr = out_dir / "levelset_train_result.json"
    if tr.exists():
        d = json.loads(tr.read_text())
        history = [(int(h.get("epoch", -1)), float(h.get("d_seg", float("nan"))))
                   for h in d.get("history", [])]
    return {"label": label, "ema_hash": _hash_npz(str(ema)), "per_tensor": per_tensor,
            "history": history, "wall_s": round(time.time() - t0, 1)}


def _compare(a: dict, b: dict) -> dict:
    import numpy as np

    if a.get("error") or b.get("error"):
        return {"error": {"A": a.get("error"), "B": b.get("error")}}
    ident = a["ema_hash"] == b["ema_hash"]
    # divergence STEP from the per-epoch d_seg history (first epoch whose d_seg differs).
    step = None
    ha, hb = dict(a["history"]), dict(b["history"])
    for ep in sorted(set(ha) & set(hb)):
        if ha[ep] != hb[ep]:
            step = ep
            break
    # per-tensor L2 of the final EMA weight diff (attribution: which tensors the flag moved).
    diffs = {}
    for k in sorted(set(a["per_tensor"]) & set(b["per_tensor"])):
        ta, tb = a["per_tensor"][k], b["per_tensor"][k]
        if ta.shape == tb.shape:
            d = float(np.sqrt(np.sum((ta - tb) ** 2)))
            if d > 0.0:
                diffs[k] = d
    return {
        "final_ema_bit_identical": ident,
        "divergence_step_epoch": step,
        "n_tensors_diverged": len(diffs),
        "per_tensor_l2_diff": dict(sorted(diffs.items(), key=lambda kv: -kv[1])[:12]),
        "history_A_tail": a["history"][-4:], "history_B_tail": b["history"][-4:],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--scratch", default=str(_REPO / "experiments" / "results" / "witness_exact_ab_350"))
    ap.add_argument("--timeout", type=int, default=560)
    ap.add_argument("--null", action="store_true", help="NULL TEST: run the SAME config twice (must be 0-divergence)")
    ap.add_argument("--flip-flag", action="store_true", help="FLIP TEST: arm B = arm A + the trailing flags")
    ap.add_argument("--out", default=None)
    ap.add_argument("flip_flags", nargs="*", help="extra trainer flags appended to arm B (flip test)")
    args = ap.parse_args(argv)

    fused_r = args.device == "gpu"  # fused-R is GPU-only; CPU is deterministic by reference path
    gt = args.gt_cache if os.path.isabs(args.gt_cache) else str(_REPO / args.gt_cache)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    b_flags = [] if args.null else list(args.flip_flags)
    if not args.null and not b_flags:
        print("[error] FLIP test needs trailing flags (or use --null)", file=sys.stderr)
        return 2

    a = _run_arm(scratch, "A", pairs=args.pairs, epochs=args.epochs, gt_cache=gt,
                 device=args.device, fused_r=fused_r, extra_flags=[], timeout=args.timeout)
    b = _run_arm(scratch, "B", pairs=args.pairs, epochs=args.epochs, gt_cache=gt,
                 device=args.device, fused_r=fused_r, extra_flags=b_flags, timeout=args.timeout)
    cmp = _compare(a, b)
    mode = "NULL" if args.null else "FLIP"
    report = {"mode": mode, "device": args.device, "fused_r": fused_r, "pairs": args.pairs,
              "epochs": args.epochs, "b_flags": b_flags, "compare": cmp,
              "wall_s": {"A": a.get("wall_s"), "B": b.get("wall_s")},
              "authority": "[macOS-MLX research-signal]/[macOS-CPU advisory]; NOT a score"}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    if cmp.get("error"):
        return 2
    if args.null:
        ok = cmp["final_ema_bit_identical"] and cmp["divergence_step_epoch"] is None
        print(f"[null-test] {'PASS (0 divergence)' if ok else 'FAIL — determinism regressed!'}",
              file=sys.stderr)
        return 0 if ok else 1
    print(f"[flip-test] divergence step epoch={cmp['divergence_step_epoch']} "
          f"tensors={cmp['n_tensors_diverged']} (attributed to {b_flags})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
