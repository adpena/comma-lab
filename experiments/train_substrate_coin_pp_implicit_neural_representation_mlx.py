# SPDX-License-Identifier: MIT
"""COIN++ implicit neural representation MLX-first smoke trainer (L0 SCAFFOLD).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Path 3 candidate #K per operator directive 2026-05-26 verbatim:
*"The MLX first requirement might also force us out of the issue we were
having before where we had great ideas but we're building them as Boltons
to the same substrates over and over again; we want to design the
substrate and curriculum and then optimize the design the whole stack
around it for extreme optimization and performance and optimal score
lowering"*

L0 SCAFFOLD posture per Catalog #240: ``_full_main`` raises
``NotImplementedError``; the smoke path (≤5 epochs, ≤8 pairs) is the only
operator-runnable surface and is gated by manual flag passing per
``--smoke`` argparse plumbing.

All training emits artifacts tagged ``[macOS-MLX research-signal]`` +
``score_claim=false`` + ``promotion_eligible=false`` +
``ready_for_exact_eval_dispatch=false`` per Catalog #127 + #192 + #317 +
#341 non-promotable markers.

Per Catalog #1265 MLX-first contest-equivalence gate (threshold 0.001
contest-units; 90× margin over empirical anchor 0.000011): paid CUDA
dispatch authorization is GATED on this MLX trainer's smoke output passing
the canonical gate BEFORE any operator-routable paid GPU work fires.

Per Catalog #325 per-substrate symposium: L1+ promotion requires the
6-step adversarial grand council symposium contract documented in
``.omx/research/path_3_k_coin_pp_substrate_design_20260526.md`` Phase 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Avoid MLX import at module load per axis 3 portability (numpy reference
# remains operable on non-Apple-Silicon test rigs).


def _full_main(argv: list[str] | None = None) -> int:
    """Phase 2+ full trainer; raises NotImplementedError per Catalog #240 (c).

    The L0 SCAFFOLD ships smoke-only surface; full training requires:
    - Per-substrate symposium per Catalog #325
    - MLX-first contest-equivalence gate verdict per Catalog #1265
    - Operator-authorized paid CUDA budget per CLAUDE.md "Long-burn score-lowering campaign default"
    """
    raise NotImplementedError(
        "coin_pp_implicit_neural_representation full training NOT YET IMPLEMENTED "
        "per Catalog #240 (c) L0 SCAFFOLD posture. "
        "Phase 2 per-substrate symposium per Catalog #325 + Catalog #1265 "
        "MLX↔PyTorch contest-equivalence gate verdict REQUIRED before any "
        "paid-CUDA dispatch authorization. See "
        ".omx/research/path_3_k_coin_pp_substrate_design_20260526.md "
        "for the Phase 2+ roadmap."
    )


def _smoke_main(argv: list[str]) -> int:
    """MLX-local smoke trainer (≤5ep ≤8pairs); no paid dispatch.

    Emits non-promotable research-signal artifacts per Catalog #341 routing
    markers + Catalog #317 local-research-signal stamping. Output destination
    is operator-configurable; default to ``.omx/research/path_3_k_coin_pp_smoke/``.

    This is a STUB at L0: full smoke implementation is queued for the L1
    promotion path. The structure here establishes the canonical CLI surface
    + manifest emission contract.
    """
    parser = argparse.ArgumentParser(
        description="COIN++ MLX-local smoke trainer (L0 SCAFFOLD stub)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=2, help="Smoke epoch count (max 5 enforced)"
    )
    parser.add_argument(
        "--num-pairs", type=int, default=4, help="Smoke pair count (max 8 enforced)"
    )
    parser.add_argument(
        "--mod-dim", type=int, default=64, help="Per-pair modulation dim (default 64)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".omx/research/path_3_k_coin_pp_smoke",
        help="Output dir for smoke artifacts (NOT /tmp per Catalog #208)",
    )
    args = parser.parse_args(argv)

    # Enforce smoke bounds per L0 SCAFFOLD posture
    if args.num_epochs > 5:
        print(
            f"ERROR: smoke epoch count {args.num_epochs} > 5; use --num-epochs <= 5 at L0",
            file=sys.stderr,
        )
        return 2
    if args.num_pairs > 8:
        print(
            f"ERROR: smoke pair count {args.num_pairs} > 8; use --num-pairs <= 8 at L0",
            file=sys.stderr,
        )
        return 2

    # Lazy import; substrate package is portable on non-Apple-Silicon
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
        estimate_archive_bytes,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig(
        mod_dim=args.mod_dim, num_pairs=args.num_pairs
    )
    estimated_bytes = estimate_archive_bytes(cfg)

    # Catalog #229 PV: refuse if output_dir is under /tmp per CLAUDE.md FORBIDDEN_PATTERN
    output_dir = Path(args.output_dir)
    output_dir_str = str(output_dir.resolve())
    if output_dir_str.startswith("/tmp/") or output_dir_str.startswith("/private/tmp/"):
        print(
            f"ERROR: output-dir {args.output_dir} under /tmp per CLAUDE.md "
            f"FORBIDDEN_PATTERN 'Forbidden /tmp paths in any persisted artifact'",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    # Emit smoke manifest with non-promotable markers per Catalog #341
    # routing-markers + Catalog #317 + Catalog #192 + Catalog #127.
    smoke_manifest = {
        "schema_version": "coin_pp_smoke_manifest_v1_20260526",
        "substrate_id": "coin_pp_implicit_neural_representation",
        "lane_id": "lane_path_3_k_coin_pp_implicit_neural_representation_20260526",
        "config": {
            "mod_dim": cfg.mod_dim,
            "pos_dim": cfg.pos_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_pairs": cfg.num_pairs,
            "eval_h": cfg.eval_h,
            "eval_w": cfg.eval_w,
            "modulation_quant_bits": cfg.modulation_quant_bits,
        },
        "smoke_bounds": {"num_epochs": args.num_epochs, "num_pairs": args.num_pairs},
        "estimated_archive_bytes": int(estimated_bytes),
        # CANONICAL NON-PROMOTABLE MARKERS per Catalog #127/#192/#317/#341
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[macOS-MLX research-signal]",
        "predicted_delta_adjustment": 0.0,
        # OBSERVABILITY per Catalog #305
        "operator_routable_next_steps": [
            "Catalog #1265 MLX-first contest-equivalence gate invocation",
            "Catalog #325 per-substrate symposium (6-step contract)",
            "MOD_DIM sweep {16, 32, 64, 128, 256} empirical paired comparison",
            "int8 vs int16 modulation quantization paired sweep + Catalog #324 Tier-C re-measurement",
        ],
        # L0 SCAFFOLD STATUS
        "l0_scaffold_status": "smoke_stub_only_full_training_pending_phase_2",
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[coin_pp smoke L0 SCAFFOLD] manifest written to: {manifest_path}", file=sys.stderr)
    print(
        f"[coin_pp smoke L0 SCAFFOLD] config: mod_dim={cfg.mod_dim} num_pairs={cfg.num_pairs} "
        f"estimated_archive_bytes={estimated_bytes}",
        file=sys.stderr,
    )
    print(
        "[coin_pp smoke L0 SCAFFOLD] [macOS-MLX research-signal] non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. --smoke gates _smoke_main; default calls _full_main (NotImplementedError)."""
    args = list(argv if argv is not None else sys.argv[1:])
    if "--smoke" in args:
        args = [a for a in args if a != "--smoke"]
        return _smoke_main(args)
    return _full_main(args)


__all__ = ["_full_main", "_smoke_main", "main"]


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
