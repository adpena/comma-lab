#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed ``top_k`` queue from optimizer/sweep artifacts.

This is an adapter, not a dispatcher. It accepts local planning artifacts from
CMA-ES/Optuna CodecOp searches, A1/PR101 bias-coordinate sweeps, local training
runtime profiles, completed materializer chains, and existing meta-Lagrangian
reports, then emits the JSON shape consumed by ``tools/parallel_dispatch_top_k.py``. Rows stay
``ready_for_exact_eval_dispatch=false`` unless a separate exact-readiness gate
has already promoted them; proxy, macOS CPU, and forensic rankings are never
promoted here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimizer.candidate_queue import build_candidate_queue  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help=(
            "JSON source artifact. May be repeated. Supported sources include "
            "constrained_coord_search rollup.json, sweep_m5max_hnerv_cluster "
            "manifest, codec_op_cma/optuna reports, codec_op_param_sweep "
            "manifests, trainer_runtime_profile_observation.v1, representation "
            "training manifests with runtime_profile/runtime_profiles, "
            "byte_shaving_campaign_plan.v1, byte/inverse materializer chain "
            "manifests, family-agnostic materializer candidate manifests, and "
            "meta_lagrangian reports."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit emitted top_k rows after deterministic ranking.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used for relative path resolution.",
    )
    args = parser.parse_args(argv)

    if args.top_k is not None and args.top_k < 1:
        raise SystemExit("--top-k must be >= 1 when provided")
    missing = [path for path in args.source if not path.is_file()]
    if missing:
        raise SystemExit(
            "source path(s) do not exist: "
            + ", ".join(path.as_posix() for path in missing)
        )

    queue = build_candidate_queue(
        args.source,
        repo_root=args.repo_root,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(queue, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} "
        f"(n_candidates={queue['n_candidates']}, top_k={queue['top_k_count']}, "
        f"dispatch_ready={queue['dispatch_ready_count']})"
    )
    if queue["dispatch_ready_count"] == 0:
        print(
            "queue is planning-only: existing dispatch actuators should refuse "
            "until an exact readiness gate promotes rows"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
