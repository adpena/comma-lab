#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Thin CLI for the canonical BOTH-TERMS speedup acceptance gate.

Every future gradient-throughput speedup (the custom Metal grouped-conv backward,
``mx.compile`` fusion, native-conv recovery, fp16/bf16 scorer fwd+bwd, NAX, ...)
MUST pass THIS gate at the REAL n (600) before it drives a real basin run. This
CLI is the operator-facing surface: feed it the baseline (trusted-gradient) and
candidate (faster-gradient) per-epoch EXACT-(torch-CPU) trajectories and it prints
+ returns the gate verdict, exiting nonzero on REJECT so a CI/operator harness
can gate a dispatch on it.

The gate exists because of the n600 incident: a kernel validated on d_seg ONLY at
n8 passed and then DIVERGED on the unmeasured pose axis (d_pose 0.8 -> 7 -> 36).
This CLI structurally refuses a d_seg-only trajectory and rejects a d_pose-
divergent candidate even when d_seg is perfect. See
``.omx/research/mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md``
and ``.omx/research/mlx_gpu_throughput_plan_20260612.md``.

Input formats (either):
  * ``--from-ab <json>``: a single JSON written by
    ``experiments/measure_descent_equivalence.py`` (keys ``arm_torch_cpu`` /
    ``arm_mlx_gpu`` + ``config.max_pairs``). The torch-CPU arm is the baseline,
    the mlx_gpu arm is the candidate.
  * ``--baseline <json> --candidate <json>``: two separate trajectory JSONs, each
    a list of ``{epoch, d_seg|exact_d_seg, d_pose|mean_d_pose}`` records (or a
    dict with a ``trajectory`` / ``arm`` list under those keys).

Authority: the d_seg/d_pose fed in MUST be torch-CPU exact for BOTH arms (the
candidate's GRADIENT is research-signal; its REPORTED metric is recomputed on
torch-CPU). This CLI never reads a score off a fast backend. $0, local.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.mlx_pr95_port.speedup_acceptance_gate import (  # noqa: E402
    GateConfig,
    evaluate_descent_equivalence,
)


def _extract_trajectory(obj) -> list[dict]:
    """Pull a list of epoch records out of a loaded JSON object."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("trajectory", "arm", "records", "traj"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    raise ValueError(
        "could not find a trajectory list in the JSON; expected a list of "
        "{epoch, d_seg, d_pose} records or a dict with a 'trajectory'/'arm' list."
    )


def _load_ab(path: Path) -> tuple[list[dict], list[dict], int]:
    blob = json.loads(path.read_text())
    if "arm_torch_cpu" not in blob or "arm_mlx_gpu" not in blob:
        raise ValueError(
            f"{path} is not a measure_descent_equivalence A/B JSON "
            "(missing 'arm_torch_cpu'/'arm_mlx_gpu')."
        )
    n = int(blob.get("config", {}).get("max_pairs", 0)) or int(
        blob.get("config", {}).get("n_pairs", 0)
    )
    if n <= 0:
        raise ValueError(
            f"{path} has no config.max_pairs/n_pairs — cannot determine the n the "
            "A/B ran at (the gate flags a PASS below n600 as provisional)."
        )
    return blob["arm_torch_cpu"], blob["arm_mlx_gpu"], n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-ab",
        type=Path,
        default=None,
        help="a measure_descent_equivalence.py A/B JSON (torch-CPU=baseline, mlx_gpu=candidate).",
    )
    ap.add_argument("--baseline", type=Path, help="baseline (trusted-gradient) trajectory JSON.")
    ap.add_argument("--candidate", type=Path, help="candidate (faster-gradient) trajectory JSON.")
    ap.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="the n the A/B ran at (required with --baseline/--candidate; read from --from-ab).",
    )
    ap.add_argument("--seg-abs-tol", type=float, default=GateConfig.seg_abs_tol)
    ap.add_argument("--seg-rel-tol", type=float, default=GateConfig.seg_rel_tol)
    ap.add_argument("--pose-abs-tol", type=float, default=GateConfig.pose_abs_tol)
    ap.add_argument("--pose-rel-tol", type=float, default=GateConfig.pose_rel_tol)
    ap.add_argument(
        "--min-trustworthy-n",
        type=int,
        default=GateConfig.min_trustworthy_n,
        help="a PASS below this n is flagged provisional (default 600 — the n600 lesson).",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="optional verdict JSON sink.")
    args = ap.parse_args(argv)

    if args.from_ab is not None:
        if args.baseline is not None or args.candidate is not None:
            ap.error("supply --from-ab OR --baseline/--candidate, not both.")
        baseline, candidate, n_pairs = _load_ab(args.from_ab)
    else:
        if args.baseline is None or args.candidate is None:
            ap.error("supply --from-ab OR both --baseline and --candidate.")
        if args.n_pairs is None:
            ap.error("--n-pairs is required with --baseline/--candidate.")
        baseline = _extract_trajectory(json.loads(args.baseline.read_text()))
        candidate = _extract_trajectory(json.loads(args.candidate.read_text()))
        n_pairs = int(args.n_pairs)

    cfg = GateConfig(
        seg_abs_tol=args.seg_abs_tol,
        seg_rel_tol=args.seg_rel_tol,
        pose_abs_tol=args.pose_abs_tol,
        pose_rel_tol=args.pose_rel_tol,
        min_trustworthy_n=args.min_trustworthy_n,
    )
    verdict = evaluate_descent_equivalence(baseline, candidate, n_pairs=n_pairs, config=cfg)

    print("=== SPEEDUP ACCEPTANCE GATE (BOTH-TERMS, exact torch-CPU d_seg AND d_pose) ===")
    print(f"n_pairs={verdict.n_pairs}  epochs_compared={verdict.epochs_compared}  axis={verdict.axis}")
    print(f"seg : {verdict.seg.reason}")
    print(f"pose: {verdict.pose.reason}")
    print(f"\nVERDICT: {'PASS' if verdict.passed else 'REJECT'}")
    if verdict.generalization_warning:
        print("  (PROVISIONAL — passed below min_trustworthy_n; re-run at the real n)")
    for r in verdict.reasons:
        print(f"  - {r}")

    if args.out_json is not None:
        out = {
            "axis": verdict.axis,
            "passed": verdict.passed,
            "n_pairs": verdict.n_pairs,
            "epochs_compared": verdict.epochs_compared,
            "generalization_warning": verdict.generalization_warning,
            "reasons": list(verdict.reasons),
            "seg": {
                "tracks": verdict.seg.tracks_within_tol,
                "diverged": verdict.seg.diverged,
                "final_abs_gap": verdict.seg.final_abs_gap,
                "reason": verdict.seg.reason,
            },
            "pose": {
                "tracks": verdict.pose.tracks_within_tol,
                "diverged": verdict.pose.diverged,
                "diverged_at_epoch": verdict.pose.diverged_at_epoch,
                "final_abs_gap": verdict.pose.final_abs_gap,
                "reason": verdict.pose.reason,
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out_json}")

    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
