# SPDX-License-Identifier: MIT
"""Local MPS smoke launcher for the Cool-Chic substrate (mirrors the validated
HNeRV ``launch_split_by_head_basin.py`` device-split pattern).

The split (CLAUDE.md "MPS auth eval is NOISE"):
  * TRAIN/GRADIENT device = ``mps`` (the Apple GPU; the per-step decoder+scorer
    forward/backward — the fast local lever).
  * AUTHORITY device = ``cpu`` (the EXACT d_seg/d_pose that pick BEST + the
    byte-closed ``upstream/evaluate.py`` advisory eval). MPS is NEVER score
    authority — every number this launcher reports as a score is a CPU-authority
    number tagged ``[contest-CPU advisory]`` NON-PROMOTABLE.

What it does (the REAL deliverable — no fake):
  1. Runs ``experiments/train_substrate_cool_chic.py`` in MPS-split mode
     (``--train-device mps --device cpu``) on the FULL 600-pair latent grid
     (so the archive inflates to all 1200 frames) for a few epochs, with the
     in-trainer CUDA auth-eval skipped (that path is CUDA-gated).
  2. Runs the byte-closed ``archive.zip`` through the contest CPU authority eval
     (``experiments/contest_auth_eval.py --device cpu``): inflate.sh ->
     upstream/evaluate.py on the EXACT submitted bytes.
  3. Reports the advisory ``S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/N`` from
     the CPU authority components, plus the MPS-vs-CPU per-epoch wall-clock.

This is a SMOKE, not a frontier run: a few epochs on a fresh init will NOT beat
the frontier. The point is to prove MPS training runs end-to-end, byte-closes,
and authority-evals on CPU — the local-MPS enablement, validated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINER = REPO_ROOT / "experiments" / "train_substrate_cool_chic.py"
AUTH_EVAL = REPO_ROOT / "experiments" / "contest_auth_eval.py"
DEFAULT_VIDEO = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM = REPO_ROOT / "upstream"
CONTEST_NORMALIZER = 37_545_489.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="run dir for the smoke (trainer writes archive + best.pt here)")
    p.add_argument("--epochs", type=int, default=6,
                   help="training epochs (few — this is a smoke, not a frontier run)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-pairs", type=int, default=600,
                   help="number of pairs to decode + latent rows (600 = full 1200 "
                        "frames so the archive inflates to a full contest eval). A "
                        "smaller value will fail the contest partial-inflate check.")
    p.add_argument("--lr", type=float, default=5e-3,
                   help="AdamW lr (higher than the 5e-4 full default so the short "
                        "smoke moves visibly).")
    p.add_argument("--val-every-epochs", type=int, default=2)
    p.add_argument("--val-pair-count", type=int, default=16)
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--cpu-compare-epochs", type=int, default=1,
        help="run this many epochs on CPU-ONLY (train_device=cpu) too, for an "
             "MPS-vs-CPU per-epoch wall-clock comparison. 0 to skip.",
    )
    p.add_argument(
        "--skip-authority-eval", action="store_true",
        help="skip the CPU authority eval (just train + byte-close). The advisory "
             "S will not be reported.",
    )
    return p


def _run(cmd: list[str], *, label: str) -> tuple[int, float]:
    print(f"\n[launch] {label}: {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    print(f"[launch] {label} rc={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return proc.returncode, elapsed


def _train_cmd(out_dir: Path, *, train_device: str, epochs: int, args) -> list[str]:
    return [
        sys.executable, str(TRAINER),
        "--video-path", str(args.video_path),
        "--output-dir", str(out_dir),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--val-every-epochs", str(args.val_every_epochs),
        "--val-pair-count", str(args.val_pair_count),
        "--max-pairs", str(args.n_pairs),
        "--upstream-dir", str(args.upstream_dir),
        "--device", "cpu",            # CPU authority (advisory)
        "--train-device", train_device,
        "--seed", str(args.seed),
        "--skip-auth-eval",           # the in-trainer auth-eval is CUDA-gated
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "axis": "[contest-CPU advisory] NON-PROMOTABLE",
        "train_device": "mps",
        "authority_device": "cpu",
        "n_pairs": args.n_pairs,
        "epochs": args.epochs,
    }

    # ---- 1. MPS-split training (the real deliverable) ----
    rc, mps_elapsed = _run(
        _train_cmd(out_dir, train_device="mps", epochs=args.epochs, args=args),
        label="MPS-split training",
    )
    if rc != 0:
        print("[launch] MPS training FAILED — aborting.", file=sys.stderr)
        return rc
    summary["mps_total_train_sec"] = round(mps_elapsed, 1)
    summary["mps_sec_per_epoch"] = round(mps_elapsed / max(1, args.epochs), 2)

    # ---- 2. CPU-only wall-clock comparison (optional, tiny) ----
    if args.cpu_compare_epochs > 0:
        cpu_dir = out_dir.parent / (out_dir.name + "_cpu_compare")
        rc_cpu, cpu_elapsed = _run(
            _train_cmd(
                cpu_dir, train_device="cpu", epochs=args.cpu_compare_epochs, args=args
            ),
            label=f"CPU-only compare ({args.cpu_compare_epochs}ep)",
        )
        if rc_cpu == 0:
            cpu_per_epoch = cpu_elapsed / max(1, args.cpu_compare_epochs)
            summary["cpu_sec_per_epoch"] = round(cpu_per_epoch, 2)
            mps_per_epoch = mps_elapsed / max(1, args.epochs)
            if mps_per_epoch > 0:
                summary["mps_speedup_vs_cpu"] = round(cpu_per_epoch / mps_per_epoch, 2)
        else:
            summary["cpu_compare_error"] = f"rc={rc_cpu}"

    # ---- 3. CPU authority eval on the byte-closed archive ----
    archive_zip = out_dir / "archive.zip"
    inflate_sh = out_dir / "submission" / "inflate.sh"
    if not args.skip_authority_eval and archive_zip.is_file() and inflate_sh.is_file():
        auth_json = out_dir / "contest_auth_eval_cpu_advisory.json"
        rc_auth, auth_elapsed = _run(
            [
                sys.executable, str(AUTH_EVAL),
                "--archive", str(archive_zip),
                "--inflate-sh", str(inflate_sh),
                "--upstream-dir", str(args.upstream_dir),
                "--device", "cpu",
                "--json-out", str(auth_json),
            ],
            label="CPU authority eval (inflate.sh -> evaluate.py)",
        )
        summary["authority_eval_sec"] = round(auth_elapsed, 1)
        if rc_auth == 0 and auth_json.is_file():
            payload = json.loads(auth_json.read_text(encoding="utf-8"))
            d_seg = payload.get("avg_segnet_dist")
            d_pose = payload.get("avg_posenet_dist")
            archive_bytes = payload.get("archive_size_bytes")
            advisory_s = payload.get("canonical_score", payload.get("final_score"))
            summary.update(
                {
                    "advisory_S": advisory_s,
                    "d_seg": d_seg,
                    "d_pose": d_pose,
                    "archive_bytes": archive_bytes,
                    "score_seg_contribution": payload.get("score_seg_contribution"),
                    "score_pose_contribution": payload.get("score_pose_contribution"),
                    "score_rate_contribution": payload.get("score_rate_contribution"),
                    "authority_eval_json": str(auth_json),
                }
            )
        else:
            summary["authority_eval_error"] = f"rc={rc_auth}"
    else:
        summary["authority_eval_skipped"] = True

    (out_dir / "mps_smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\n" + "=" * 72)
    print("[launch] COOL-CHIC MPS SMOKE SUMMARY (all scores [contest-CPU advisory]):")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
