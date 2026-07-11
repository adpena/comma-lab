#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or explicitly execute the provider-neutral V9 CGauge CUDA plan.

Default operation is local and plan-only.  Remote mutation requires both
``--execute`` and the exact operator token; scaffold providers always refuse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.deploy.witness_cloud_launcher import build_plan, render_command  # noqa: E402

OPERATOR_TOKEN = "GO-CLOUD-381"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", choices=("modal", "aws", "gcp"), default="modal")
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--label", default="v9_cgauge_cuda_438")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--resume-from")
    ap.add_argument(
        "--gt-cache-sha256",
        help="Required for execution; exact SHA-256 of the staged scorer-oracle cache",
    )
    ap.add_argument("--output")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--operator-go-token")
    args = ap.parse_args(argv)

    plan = build_plan(
        provider=args.provider,
        gt_cache=args.gt_cache,
        label=args.label,
        gpu=args.gpu,
        epochs=args.epochs,
        num_pairs=args.num_pairs,
        resume_from=args.resume_from,
        gt_cache_sha256=args.gt_cache_sha256,
    )
    payload = plan.to_dict()
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("asset_stage:", render_command(plan.asset_stage_argv))
    print("dispatch:", render_command(plan.dispatch_argv))
    print("harvest:", render_command(plan.harvest_argv))
    if not args.execute:
        print("PLAN ONLY: no provider contacted and no paid resource dispatched.")
        return 0
    if args.operator_go_token != OPERATOR_TOKEN:
        raise SystemExit("REFUSED: --execute requires --operator-go-token GO-CLOUD-381")
    if not plan.execution_allowed:
        raise SystemExit(f"REFUSED: provider {plan.provider} is {plan.status}, not executable")
    local_cache = REPO / plan.local_gt_cache
    if not local_cache.is_file():
        raise SystemExit(f"REFUSED: asset missing: {local_cache}")
    if not args.gt_cache_sha256:
        raise SystemExit("REFUSED: --execute requires --gt-cache-sha256 custody")
    digest = hashlib.sha256()
    with local_cache.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != args.gt_cache_sha256.lower():
        raise SystemExit(
            f"REFUSED: GT cache SHA-256 mismatch: {actual_sha256} != {args.gt_cache_sha256}"
        )
    # Asset staging and dispatch are separate checked transactions.  The
    # dispatch itself uses Modal .spawn() inside modal_train_lane.py; losing
    # the local CLI therefore does not cancel the remote run.
    subprocess.run(plan.asset_stage_argv, cwd=REPO, check=True)
    subprocess.run(plan.dispatch_argv, cwd=REPO, check=True)
    print("DISPATCHED: harvest through the printed canonical ledger command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
