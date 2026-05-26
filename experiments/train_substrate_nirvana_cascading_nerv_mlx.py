#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""nirvana_cascading_nerv MLX smoke trainer (L0 SCAFFOLD stub).

Path 3 candidate #G smoke trainer stub per operator directive 2026-05-26:
*"design the substrate and curriculum and then optimize the design the
whole stack around it for extreme optimization and performance and
optimal score lowering"*

L0 SCAFFOLD scope: this trainer scaffolds the MLX-local smoke training
interface; ``_full_main`` raises NotImplementedError per Catalog #240
+ Catalog #325 per-substrate council symposium requirement BEFORE any
paid CUDA dispatch authorization.

Per CLAUDE.md "MLX portable-local-substrate authority":
- Tag every output with ``[macOS-MLX research-signal]``
- ``score_claim=false`` + ``promotion_eligible=false`` +
  ``ready_for_exact_eval_dispatch=false``
- MLX results NEVER become contest score claims without paired
  Linux x86_64 + NVIDIA CUDA paired auth eval per the
  "Submission auth eval - BOTH CPU AND CUDA" non-negotiable.

Per Catalog #1265 MLX↔PyTorch parity gate (threshold 0.001 contest-units;
margin 90× over empirical anchor 0.000011): every substrate MUST pass
this gate BEFORE any paid CUDA dispatch authorization. The
``tools/gate_mlx_candidate_contest_equivalence.py`` gate is hardwired for
PR95 grammar; a sister gate
``tools/gate_mlx_candidate_contest_equivalence_nirvana.py`` is queued as
op-routable for NIRVANA1 grammar.

Smoke command (when L1+ landed):

    .venv/bin/python experiments/train_substrate_nirvana_cascading_nerv_mlx.py \\
        --smoke --epochs 5 --num-pairs 8

Stub behavior at L0:
- ``--help`` works and prints usage
- ``--smoke`` argparse path is wired but ``_full_main`` raises
  NotImplementedError per Catalog #240
- Importing this script does NOT trigger MLX import
"""

from __future__ import annotations

import argparse
import sys

# Import trainer-required-flags manifest per Catalog #151 (operator-wrapper
# threads Tier-1 required flags). At L0 SCAFFOLD posture, this is a stub
# manifest; Phase 2+ will populate real flags as the trainer grows.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict] = {
    # Placeholder; populated at L1+ when real training flags land
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_substrate_nirvana_cascading_nerv_mlx",
        description=(
            "NIRVANA cascading NeRV MLX smoke trainer (L0 SCAFFOLD stub). "
            "Phase 2 council symposium per Catalog #325 + Catalog #1265 "
            "MLX↔PyTorch parity gate REQUIRED before any paid-CUDA dispatch."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke training (NotImplementedError at L0 per Catalog #240)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Smoke training epoch count (≤5 at L0)",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=8,
        help="Smoke training pair count (≤8 at L0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Smoke output directory. MUST be under experiments/results/ or "
            ".omx/tmp/ per CLAUDE.md 'Forbidden /tmp paths' Catalog #113."
        ),
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke training entrypoint. Per Catalog #240 L0 SCAFFOLD posture:
    raises NotImplementedError until Phase 2 council symposium lands."""
    raise NotImplementedError(
        "nirvana_cascading_nerv smoke trainer NOT YET IMPLEMENTED — L0 "
        "SCAFFOLD posture per Catalog #240. Phase 2 council symposium per "
        "Catalog #325 + Catalog #1265 MLX↔PyTorch parity gate (sister "
        "tools/gate_mlx_candidate_contest_equivalence_nirvana.py) REQUIRED "
        "before any paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_g_nirvana_cascading_nerv_substrate_design_20260526.md "
        "for the Phase 2+ roadmap."
    )


def _full_main(args: argparse.Namespace) -> int:
    """Full training entrypoint per Catalog #240 L0 SCAFFOLD posture."""
    raise NotImplementedError(
        "nirvana_cascading_nerv full main NOT YET IMPLEMENTED — L0 SCAFFOLD "
        "posture per Catalog #240. See design memo for Phase 2+ roadmap."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
