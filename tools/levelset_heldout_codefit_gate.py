#!/usr/bin/env python3
"""HELD-OUT code-fit GATE — the decisive amortization test (operator 2026-06-27, DAG FEED-ep).

The level-set witness factors into a SHARED decoder + per-(pair,frame) latent codes. The
amortization pivot trains a decoder on a STRIDED SUBSET of the 600 pairs (every Nth, spanning the
drive). The decisive question: does the FROZEN subset decoder GENERALIZE to the pairs it NEVER
trained on, via the code channel alone?

This wrapper, given a decoder checkpoint (a level-set EMA/deploy npz, or a run dir), constructs and
optionally runs the trainer in ``--freeze-decoder-fit-codes`` mode against the HELD-OUT (non-subset)
GT cache: the decoder is FROZEN (only ``code`` trains) and the per-pair codes for the held-out pairs
are fit through the frozen render+R+frozen-SegNet. The reported REALIZED-through-R d_seg over the
held-out pairs (CPU-torch authority verdict, EMA shadow, the ONE deploy codepath) is compared to the
JOINT-trained reference (mod-32/n600 ~0.00124 at convergence). If held-out d_seg is competitive the
frozen decoder amortizes -> fit all 600 codes -> byte-close -> exact eval. If it fails -> honest
DEFER + resume the preserved n600 row.

Front-end note: the freeze path requires the curvelet[+self-orient] feature width to equal the
decoder's in_proj width (the trainer fail-closes on mismatch). The decoder's __cfg_in_feat is read
here as a PRE-CHECK; the self-orient SCALARS (n_dir_freqs/freq_across/freq_along) are NOT stored in
the npz, so they default to the SEALED n600 config (overridable). The in_feat pre-check + the
trainer's own guard jointly prevent a width-mismatched (NO-FAKE) code fit.

Modes:  --dry-run (default; print + persist the exact command) · --smoke (tiny foreground run that
verifies the freeze path wires end-to-end and emits a d_seg) · --launch (durable resumable daemon
for the real gate; the monitoring loop runs this at the decoder's stage checkpoints).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
JOINT_REFERENCE_DSEG = 0.00124  # mod-32 / n600 joint-trained realized d_seg at convergence (operator)


def _resolve_ckpt(p: Path) -> Path:
    p = Path(p)
    if p.is_dir():
        for name in ("levelset_witness_ema_mlx.npz", "levelset_resume_state.npz"):
            c = p / name
            if c.exists():
                return c
        raise FileNotFoundError(f"decoder ckpt dir {p} has no levelset_witness_ema_mlx.npz / resume npz.")
    if p.exists():
        return p
    raise FileNotFoundError(f"decoder ckpt {p} does not exist (NO-FAKE: refusing to fabricate).")


def _ckpt_in_feat(ckpt: Path) -> int | None:
    z = np.load(ckpt, allow_pickle=False)
    if "in_proj.weight" in z.files:
        return int(z["in_proj.weight"].shape[1])
    if "__cfg_in_feat" in z.files:
        return int(z["__cfg_in_feat"])
    return None


def _build_cmd(args, ckpt: Path, out_dir: Path, num_pairs: int, epochs: int,
               verdict_pairs: int, stage_ckpts: bool) -> list[str]:
    """The trainer argv for a held-out code-fit. SEALED front-end defaults (overridable) so the
    curvelet[+self-orient] width matches the decoder in_proj; --freeze-decoder-fit-codes freezes
    everything but ``code``. No --curriculum (a code-fit uses a FIXED final-stage seg loss so the
    curriculum-boundary validator is not tripped at small epoch budgets)."""
    cmd = [
        "env", "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1",
        ".venv/bin/python", "-u",
        "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "--out-dir", str(out_dir),
        "--freeze-decoder-fit-codes", str(ckpt),
        "--gt-cache", str(args.heldout_gt),
        "--num-pairs", str(num_pairs),
        "--epochs", str(epochs),
        "--render-h", "384", "--render-w", "512",
        "--hidden-dim", str(args.hidden_dim), "--mod-dim", str(args.mod_dim),
        "--activation", args.activation, "--siren-init",
        "--softmax-temp-start", "1.0", "--softmax-temp-end", "0.05",
        "--palette-anchor",
        "--self-orient", "--reorient-every", str(args.reorient_every),
        "--freq-across", str(args.freq_across), "--n-dir-freqs", str(args.n_dir_freqs),
        "--freq-along", str(args.freq_along), "--max-bank-freq", str(args.max_bank_freq),
        "--chroma",
        "--seg-loss", args.seg_loss, "--no-curriculum",
        "--lane-edge-weight", str(args.lane_edge_weight),
        "--lane-edge-class", "1", "--lane-margin-target", "0.5",
        "--w-seg", "100", "--w-pose", str(args.w_pose),
        "--eikonal-weight", "0.01", "--length-weight", "0.001",
        "--ema-decay", "0.997", "--accum-pairs", str(args.accum_pairs), "--grad-clip", "1.0",
        "--verdict-pairs", str(verdict_pairs),
        "--eval-every", str(args.eval_every), "--ckpt-every", str(args.ckpt_every),
        "--mlx-device", "gpu",
    ]
    cmd += (["--stage-checkpoints"] if stage_ckpts else ["--no-stage-checkpoints"])
    if args.async_verdict:
        cmd += ["--async-verdict"]
    return cmd


def _verdict_from_result(out_dir: Path) -> dict | None:
    rp = out_dir / "levelset_train_result.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text())
    hist = r.get("history") or []
    if not hist:
        return None
    last = hist[-1]
    return {"epoch": last.get("epoch"), "d_seg": last.get("d_seg"), "d_pose": last.get("d_pose"),
            "implied_S": last.get("implied_S")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoder-ckpt", type=Path, required=True,
                    help="frozen decoder: a level-set EMA/deploy npz OR a run dir.")
    ap.add_argument("--heldout-gt", type=Path,
                    default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_heldout_n400.npz",
                    help="held-out (non-subset) GT cache to fit codes on.")
    ap.add_argument("--num-pairs", type=int, default=400, help="held-out pairs to fit codes for.")
    ap.add_argument("--epochs", type=int, default=600, help="code-fit epochs (frozen decoder; converges fast).")
    ap.add_argument("--verdict-pairs", type=int, default=0, help="realized-d_seg verdict subset (0=ALL held-out).")
    ap.add_argument("--reference-dseg", type=float, default=JOINT_REFERENCE_DSEG,
                    help="joint-trained reference realized d_seg (mod-32/n600 ~0.00124).")
    ap.add_argument("--competitive-mult", type=float, default=1.5,
                    help="PASS if held-out d_seg <= reference * this (competitive band).")
    # front-end (SEALED n600 defaults; must reproduce the decoder's in_feat).
    ap.add_argument("--activation", default="hosc")
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--mod-dim", type=int, default=32)
    ap.add_argument("--n-dir-freqs", type=int, default=2)
    ap.add_argument("--freq-across", type=float, default=32.0)
    ap.add_argument("--freq-along", type=float, default=4.0)
    ap.add_argument("--max-bank-freq", type=float, default=64.0)
    ap.add_argument("--reorient-every", type=int, default=50)
    ap.add_argument("--seg-loss", default="l7_softplus",
                    help="FIXED code-fit seg loss (no curriculum). l7_softplus = sharp margin stage.")
    ap.add_argument("--lane-edge-weight", type=float, default=0.0,
                    help="optional realized lane hinge during code-fit (0=off; the frozen decoder "
                    "already encodes lane structure; the d_seg metric captures lane flips regardless).")
    ap.add_argument("--w-pose", type=float, default=1.0)
    ap.add_argument("--accum-pairs", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--async-verdict", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    # mode
    ap.add_argument("--dry-run", action="store_true", help="(default) print + persist the command; do not run.")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny FOREGROUND run (num-pairs 2, epochs 3) verifying the freeze path emits a d_seg.")
    ap.add_argument("--launch", action="store_true",
                    help="launch the REAL gate as a durable resumable daemon (spawn_durable_daemon).")
    ap.add_argument("--label", default="levelset_heldout_gate")
    args = ap.parse_args(argv)

    ckpt = _resolve_ckpt(args.decoder_ckpt)
    dec_in_feat = _ckpt_in_feat(ckpt)
    print(json.dumps({"stage": "ckpt", "path": str(ckpt), "decoder_in_feat": dec_in_feat,
                      "heldout_gt": str(args.heldout_gt),
                      "reference_dseg": args.reference_dseg,
                      "pass_threshold": round(args.reference_dseg * args.competitive_mult, 6)}), flush=True)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    if args.smoke:
        out_dir = args.out_dir or (REPO / f"experiments/results/heldout_gate_SMOKE_{ts}")
        cmd = _build_cmd(args, ckpt, out_dir, num_pairs=2, epochs=3, verdict_pairs=2, stage_ckpts=False)
        print(json.dumps({"stage": "smoke_run", "cmd": " ".join(cmd)}), flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        v = _verdict_from_result(out_dir)
        ok = v is not None and isinstance(v.get("d_seg"), (int, float))
        print(json.dumps({"stage": "smoke_result", "rc": proc.returncode, "secs": round(time.time() - t0, 1),
                          "verdict": v, "SMOKE_OK": bool(ok and proc.returncode == 0)}), flush=True)
        if not (ok and proc.returncode == 0):
            print("=== smoke stdout tail ===\n" + tail, flush=True)
            print("=== smoke stderr tail ===\n" + "\n".join((proc.stderr or "").splitlines()[-12:]), flush=True)
            return 1
        return 0

    out_dir = args.out_dir or (REPO / f"experiments/results/heldout_gate_n{args.num_pairs}_{ts}")
    cmd = _build_cmd(args, ckpt, out_dir, num_pairs=args.num_pairs, epochs=args.epochs,
                     verdict_pairs=args.verdict_pairs, stage_ckpts=True)
    # in_feat pre-check (advisory; the trainer fail-closes authoritatively).
    if dec_in_feat is not None:
        print(json.dumps({"stage": "in_feat_precheck",
                          "decoder_in_feat": dec_in_feat,
                          "note": "the trainer recomputes the curvelet[+self-orient] width and fail-closes "
                          "if it != decoder in_proj; ensure --n-dir-freqs/--max-bank-freq match the decoder config."}),
              flush=True)

    cmd_path = REPO / ".omx/tmp" / f"heldout_gate_cmd_{ts}.txt"
    cmd_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_path.write_text(" ".join(cmd) + "\n")

    if args.launch:
        log = REPO / ".omx/tmp" / f"{args.label}_{ts}.log"
        spawn = [
            ".venv/bin/python", "tools/spawn_durable_daemon.py",
            "--label", args.label, "--log", str(log),
            "--rss-cap-mb", "90000", "--min-free-gb", "10", "--walltime-cap-s", "172800",
            "--projected-gb", "15", "--",
        ] + cmd
        print(json.dumps({"stage": "launch", "label": args.label, "log": str(log),
                          "out_dir": str(out_dir), "cmd_file": str(cmd_path)}), flush=True)
        r = subprocess.run(spawn, cwd=str(REPO), capture_output=True, text=True)
        print((r.stdout or "")[-800:], flush=True)
        if r.returncode != 0:
            print("=== spawn stderr ===\n" + (r.stderr or "")[-800:], flush=True)
        return r.returncode

    # default: dry-run
    print(json.dumps({"stage": "dry_run", "out_dir": str(out_dir), "cmd_file": str(cmd_path),
                      "run_with": "add --launch (durable daemon) or --smoke (tiny verify)",
                      "cmd": " ".join(cmd)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
