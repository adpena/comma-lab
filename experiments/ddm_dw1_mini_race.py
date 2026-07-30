"""ddm_dw1 — QA75 loss-FORM mini-race (guards 1+2: own-optimum + attack raced).

Race the distill loss form x attack-weighting on the REAL regime (resume from the E2 endpoint,
n600, matched base config = the burn continuation) for a short bounded window, so the winning
form is picked at ITS optimum in the SAME regime the full windows operate in (a fresh/n96 race
would rank forms in the wrong regime — the student is already near its floor here, the residual
is the boundary annulus).  Sequential (ONE scorer job at a time); harvest the gate d_seg
trajectory + SMEVR bytes per config; the winner = lowest end-window gate d_seg at acceptable rate.

score_claim=false; advisory [macOS-CPU/MLX]; pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = "experiments/train_tr1_partition_renderer_mlx.py"
E2 = "/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/checkpoints/stage_seg_trunk_tau_final.npz"
MASK = "/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/qa24_grid_keep_mask_50.npy"
GT = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CACHE = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache/distill_logits.f16.npy"
E2_EPOCH = 400
FORMS = ("kd_logits", "margin_field", "argmax_ce")
ATTACKS = (0.0, 1.0)


def _base_argv(out_dir: str, epochs: int, gate_every: int, max_wall: float) -> list[str]:
    """The matched window-base config (= burn continuation) WITHOUT distill/full-confirm."""
    return [TRAINER,
            "--variant", "lotto", "--num-pairs", "600", "--grid-downsample", "16",
            "--code-width", "4", "--renderer-width", "24", "--token-quant-levels", "16",
            "--token-ste", "round", "--token-temporal-mode", "shared_base",
            "--seg-form-start", "ce", "--w-seg", "100.0", "--class-weight-lane", "1.0",
            "--margin-target", "1.0", "--token-init-mode", "solve_project",
            "--basin-handoff", "off", "--gate-every", str(gate_every),
            "--token-cell-mask", MASK, "--margin-weighted-loss", "on",
            "--margin-weight-temp", "1.0", "--token-quant-anneal", "at_knee",
            "--w-rate", "0.05", "--rate-model", "entropy", "--byte-ledger-coder", "smevr",
            "--lotto-seed", "118", "--lotto-mask-density-init", "0.5",
            "--batch-pairs", "8", "--lr", "0.002", "--epochs", str(epochs),
            "--max-wall-minutes", str(max_wall), "--gt-cache", GT, "--resume-from", E2,
            "--out-dir", out_dir]


def _harvest(out_dir: Path) -> dict:
    tel = out_dir / "telemetry.jsonl"
    if not tel.is_file():
        return {"status": "no_telemetry"}
    rows = [json.loads(ln) for ln in tel.read_text().splitlines() if ln.strip()]
    gates = [r for r in rows if r.get("event") == "a1_gate"]
    eps = [r for r in rows if r.get("event") == "epoch"]
    refuse = [r for r in rows if r.get("event") == "a1_stage_exit_refuse"]
    traj = [{"epoch": g["epoch"], "dseg": g["realized_gate_dseg_mean"],
             "basis": g.get("gate_params"), "bytes": g.get("total_counted_bytes"),
             "a1": g.get("a1_classification")} for g in gates]
    final = gates[-1] if gates else None
    return {
        "status": "ok" if gates else "no_gates",
        "n_epochs_trained": len(eps),
        "final_gate_dseg": final["realized_gate_dseg_mean"] if final else None,
        "final_total_bytes": final.get("total_counted_bytes") if final else None,
        "final_basis": final.get("gate_params") if final else None,
        "a1_refused": bool(refuse),
        "last_ep_loss": eps[-1]["ep_loss"] if eps else None,
        "trajectory": traj,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/mini_race")
    ap.add_argument("--window-epochs", type=int, default=12)
    ap.add_argument("--gate-every", type=int, default=3)
    ap.add_argument("--max-wall", type=float, default=30.0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    epochs = E2_EPOCH + 1 + args.window_epochs
    env = {"PYTHONPATH": f"{REPO}/src:{REPO}:{REPO}/upstream",
           "TAC_MLX_CUSTOM_GROUPED_BACKWARD": "1", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    import os
    env = {**os.environ, **env}

    results: list[dict] = []
    started = time.time()
    for form in FORMS:
        for attack in ATTACKS:
            tag = f"{form}_a{attack:g}"
            out_dir = out_root / tag
            argv = _base_argv(str(out_dir), epochs, args.gate_every, args.max_wall)
            argv += ["--distill-field-cache", CACHE, "--distill-weight", "100.0",
                     "--distill-temp", "2.0", "--distill-form", form,
                     "--distill-attack-temp", str(attack)]
            print(f"[mini-race] START {tag} @ +{time.time()-started:.0f}s", flush=True)
            t0 = time.time()
            r = subprocess.run([f"{REPO}/.venv/bin/python", *argv], cwd=str(REPO), env=env,
                               capture_output=True, text=True)
            h = _harvest(out_dir)
            h.update({"tag": tag, "form": form, "attack": attack, "rc": r.returncode,
                      "wall_s": round(time.time() - t0, 1),
                      "stderr_tail": (r.stderr or "")[-400:] if r.returncode != 0 else ""})
            results.append(h)
            print(f"[mini-race] DONE {tag} rc={r.returncode} final_dseg={h.get('final_gate_dseg')} "
                  f"bytes={h.get('final_total_bytes')} wall={h['wall_s']}s", flush=True)
            (out_root / "race_summary.json").write_text(json.dumps({
                "schema": "ddm_dw1_mini_race.v1", "window_epochs": args.window_epochs,
                "resume_from": E2, "n_pairs": 600, "forms": list(FORMS), "attacks": list(ATTACKS),
                "score_claim": False, "authority": "advisory [macOS-CPU/MLX]",
                "pointer": "0.1910828242 [contest-CPU] UNMOVED", "results": results}, indent=2) + "\n")

    ok = [r for r in results if r["status"] == "ok" and not r["a1_refused"]
          and r["final_gate_dseg"] is not None]
    winner = min(ok, key=lambda r: r["final_gate_dseg"]) if ok else None
    print(f"[mini-race] WINNER: {winner['tag'] if winner else 'NONE'} "
          f"dseg={winner['final_gate_dseg'] if winner else None}", flush=True)
    summary = json.loads((out_root / "race_summary.json").read_text())
    summary["winner"] = winner
    (out_root / "race_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
