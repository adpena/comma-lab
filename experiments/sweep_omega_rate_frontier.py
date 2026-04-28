#!/usr/bin/env python3
"""Lane Ω-V3: explicit rate-distortion frontier sweep over Lagrangian
target_bits values.

V1 / V2 used a SINGLE hand-picked target (600,000 bits ≈ 2.5 bits/weight
for the ~290KB ASYM renderer). V3 turns that single point into a
rate-distortion frontier by sweeping {300K, 450K, 600K, 750K, 900K}
total bits and running Lane Ω-V2's Lagrangian QAT at each budget. The
operator inspects the resulting frontier .csv post-hoc and picks the
budget that minimises the contest score (or pre-commits to the best-
from-historical-data budget).

This tool is an ORCHESTRATION wrapper — every sub-run is the canonical
``experiments/qat_omega_lagrangian.py`` invocation, identical in every
respect except the ``--target-bits`` value.

Lagrangian framing
------------------
Each sub-run solves the inner problem::

    min_{θ}  D(θ)  s.t.  R(θ) ≤ B_k

via the Lagrangian dual::

    L_k(θ, λ) = D(θ) + λ · max(0, R(θ) - B_k)

with λ annealed by the inner Lagrangian schedule (lambda-start →
lambda-end across the lambda-ramp-start-frac fraction). The OUTER loop
(this script) is the parametric-programming / sensitivity-analysis
sweep over ``B_k`` — exactly the ``ε-constraint`` scalarisation of the
multi-objective problem (Boyd & Vandenberghe §4.7.5). The frontier
``{(R*(B_k), D*(B_k))}`` traces the Pareto curve of the rate-distortion
duality.

Usage
-----
::

    python experiments/sweep_omega_rate_frontier.py \\
        --checkpoint experiments/results/lane_a_landed/iter_0/renderer.bin \\
        --video upstream/videos/0.mkv \\
        --masks-mkv experiments/results/lane_a_landed/iter_0/masks.mkv \\
        --poses experiments/results/lane_a_landed/iter_0/optimized_poses.pt \\
        --upstream upstream \\
        --output-dir lane_omega_v3_results \\
        --target-bits-per-weight 1.25,1.875,2.5,3.125,3.75 \\
        --total-epochs 200 \\
        --device cuda

Each sub-run writes its own ``$output-dir/budget_<bpw>/`` directory.
The aggregated frontier is written to ``$output-dir/frontier.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QAT_SCRIPT = REPO / "experiments" / "qat_omega_lagrangian.py"


# ──────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Lane A renderer .bin to anchor every sub-run on")
    p.add_argument("--video", required=True,
                   help="GT video .mkv (passed to every sub-run)")
    p.add_argument("--masks-mkv", required=True,
                   help="Masks .mkv (passed to every sub-run)")
    p.add_argument("--poses", default=None,
                   help="Optimized poses .pt (REQUIRED if model has FiLM)")
    p.add_argument("--upstream", default="upstream",
                   help="Upstream dir (passed through)")
    p.add_argument("--output-dir", required=True,
                   help="Top-level output dir; sub-runs land in subdirs")
    p.add_argument(
        "--target-bits-per-weight",
        default="1.25,1.875,2.5,3.125,3.75",
        help="Comma-separated bits-per-weight targets to sweep "
             "(default '1.25,1.875,2.5,3.125,3.75' = "
             "{300K, 450K, 600K, 750K, 900K} for a 240K-weight model). "
             "qat_omega_lagrangian's --target-bits is in BITS PER WEIGHT.",
    )
    p.add_argument("--total-epochs", type=int, default=200,
                   help="Per-sub-run total epochs (passed through)")
    p.add_argument("--lr", type=float, default=2.5e-6,
                   help="Per-sub-run lr (passed through)")
    p.add_argument("--bits-lr-scale", type=float, default=0.1,
                   help="Per-sub-run bits-lr-scale (passed through)")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="Per-sub-run roundtrip noise std (passed through)")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="Per-sub-run seg loss weight (passed through)")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="Per-sub-run pose loss weight (passed through)")
    p.add_argument("--lambda-start", type=float, default=0.0)
    p.add_argument("--lambda-end", type=float, default=1.0)
    p.add_argument("--lambda-ramp-start-frac", type=float, default=0.3)
    p.add_argument("--init-bits", type=float, default=8.0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="CUDA strongly preferred per CLAUDE.md MPS-CUDA "
                        "drift rule (no MPS fallback in this orchestrator).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for sub-runs")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned commands without executing")
    return p.parse_args()


def parse_bpw_csv(s: str) -> list[float]:
    """Parse the comma-separated bits-per-weight list with validation."""
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError as exc:
            raise SystemExit(
                f"--target-bits-per-weight: cannot parse {tok!r} as float "
                f"({exc})"
            ) from exc
        if not (0.5 <= v <= 8.0):
            raise SystemExit(
                f"--target-bits-per-weight: {v} out of range [0.5, 8.0] "
                f"(0.5 = aggressive prune, 8.0 = essentially no compression)"
            )
        out.append(v)
    if len(out) < 2:
        raise SystemExit(
            f"--target-bits-per-weight must list ≥ 2 budgets (the whole "
            f"point of the sweep is a frontier; got {out!r})"
        )
    return out


def build_subrun_cmd(
    args: argparse.Namespace, bpw: float, subdir: Path,
) -> list[str]:
    """Construct the canonical qat_omega_lagrangian.py invocation for
    this budget. EVERY flag here is verified against the QAT script's
    argparse before launch — see assert_flags_real()."""
    cmd = [
        args.python, "-u", str(QAT_SCRIPT),
        "--checkpoint", args.checkpoint,
        "--video", args.video,
        "--masks-mkv", args.masks_mkv,
        "--upstream", args.upstream,
        "--output-dir", str(subdir),
        "--init-bits", str(args.init_bits),
        "--target-bits", str(bpw),
        "--lambda-start", str(args.lambda_start),
        "--lambda-end", str(args.lambda_end),
        "--lambda-ramp-start-frac", str(args.lambda_ramp_start_frac),
        "--total-epochs", str(args.total_epochs),
        "--lr", str(args.lr),
        "--bits-lr-scale", str(args.bits_lr_scale),
        "--noise-std", str(args.noise_std),
        "--seg-weight", str(args.seg_weight),
        "--pose-weight", str(args.pose_weight),
        "--device", args.device,
        "--seed", str(args.seed),
        "--log-every", str(args.log_every),
    ]
    if args.poses:
        cmd.extend(["--poses", args.poses])
    return cmd


def assert_flags_real(cmd: list[str]) -> None:
    """Pre-flight: every '--flag' in the planned command must exist in
    qat_omega_lagrangian.py's argparse. Catches the dead-flag class of
    bugs (CLAUDE.md non-negotiable). Mirrors the existing in-script
    preflight — cheap to run twice."""
    qat_src = QAT_SCRIPT.read_text()
    real = set(re.findall(
        r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", qat_src,
    ))
    used = {tok[2:] for tok in cmd if tok.startswith("--")}
    invented = used - real
    if invented:
        raise SystemExit(
            f"INVENTED FLAGS for qat_omega_lagrangian.py: {sorted(invented)} "
            f"— this is the dead-flag bug class CLAUDE.md's "
            f"`NEVER invent CLI flags` rule was created to prevent."
        )


def parse_qat_log_for_metrics(log_path: Path) -> dict:
    """Extract final bits/weight + best scorer / pose / seg from a
    qat_omega_lagrangian log. Robust to log-format drift: if a key
    can't be found the value is recorded as None and the operator
    sees the gap in the frontier.csv rather than a silent miss."""
    metrics: dict = {
        "final_avg_bits_per_weight": None,
        "final_scorer_loss": None,
        "final_pose_loss": None,
        "final_seg_loss": None,
    }
    if not log_path.exists():
        return metrics
    txt = log_path.read_text(errors="replace")
    # Look for the canonical "[qat] epoch N | scorer=... | bits=... | ..."
    # lines emitted by qat_omega_lagrangian.py — pick the LAST line.
    last_match = None
    for m in re.finditer(
        r"epoch\s+(\d+).*?scorer\s*=\s*([0-9.eE+-]+).*?bits\s*=\s*([0-9.]+)",
        txt,
    ):
        last_match = m
    if last_match:
        metrics["final_scorer_loss"] = float(last_match.group(2))
        metrics["final_avg_bits_per_weight"] = float(last_match.group(3))
    # Pose / seg breakdown if present
    pose_m = re.findall(r"pose[_=\s]+([0-9.eE+-]+)", txt)
    if pose_m:
        try:
            metrics["final_pose_loss"] = float(pose_m[-1])
        except ValueError:
            pass
    seg_m = re.findall(r"seg[_=\s]+([0-9.eE+-]+)", txt)
    if seg_m:
        try:
            metrics["final_seg_loss"] = float(seg_m[-1])
        except ValueError:
            pass
    return metrics


def main() -> int:
    args = parse_args()
    bpw_list = parse_bpw_csv(args.target_bits_per_weight)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Provenance: the entire sweep's plan as JSON.
    plan = {
        "tool": "experiments/sweep_omega_rate_frontier.py",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budgets_bits_per_weight": bpw_list,
        "checkpoint": args.checkpoint,
        "video": args.video,
        "masks_mkv": args.masks_mkv,
        "poses": args.poses,
        "total_epochs": args.total_epochs,
        "init_bits": args.init_bits,
        "lambda_start": args.lambda_start,
        "lambda_end": args.lambda_end,
        "lambda_ramp_start_frac": args.lambda_ramp_start_frac,
        "lr": args.lr,
        "bits_lr_scale": args.bits_lr_scale,
        "noise_std": args.noise_std,
        "seg_weight": args.seg_weight,
        "pose_weight": args.pose_weight,
        "device": args.device,
        "seed": args.seed,
        "lagrangian_target": "rate-distortion frontier (epsilon-constraint scalarisation, Boyd §4.7.5)",
        "predicted_band": [0.55, 1.10],
    }
    (out_root / "plan.json").write_text(json.dumps(plan, indent=2))
    print(f"[sweep] wrote plan to {out_root / 'plan.json'}")

    rows: list[dict] = []
    for bpw in bpw_list:
        subdir = out_root / f"budget_bpw_{bpw:.4f}"
        subdir.mkdir(parents=True, exist_ok=True)
        cmd = build_subrun_cmd(args, bpw, subdir)
        assert_flags_real(cmd)
        log_path = subdir / "qat.log"
        print(f"[sweep] === budget bpw={bpw:.4f} → {subdir} ===", flush=True)
        print(f"[sweep] cmd: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            print("[sweep] DRY-RUN — skipping subprocess.run", flush=True)
            rc = 0
        else:
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, check=False,
                )
                rc = proc.returncode
        metrics = parse_qat_log_for_metrics(log_path) if not args.dry_run else {}
        renderer_bin = subdir / "renderer.bin"
        renderer_bytes = (
            renderer_bin.stat().st_size
            if renderer_bin.exists()
            else None
        )
        row = {
            "target_bits_per_weight": bpw,
            "subdir": str(subdir),
            "returncode": rc,
            "renderer_bytes": renderer_bytes,
            **metrics,
        }
        rows.append(row)
        # Stream-write the frontier CSV after every sub-run so an
        # operator can monitor progress in another shell.
        write_frontier_csv(out_root / "frontier.csv", rows)
        if rc != 0 and not args.dry_run:
            print(
                f"[sweep] WARNING: budget {bpw} sub-run exited {rc} — "
                f"frontier.csv has the partial result; continuing.",
                flush=True,
            )

    print(f"[sweep] DONE — frontier at {out_root / 'frontier.csv'}")
    return 0


def write_frontier_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())
