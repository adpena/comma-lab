# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 MLX-FIRST score-aware trainer SCAFFOLD — L1 SCAFFOLD pending migration.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_blocked_per_optimal_form_dispatch_until_mlx_module_migration_lands
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only
# OPTIMAL_FORM_DISPATCH_OK:scaffold_only_full_main_raises_NotImplementedError_per_catalog_240_research_only_true_dispatch_enabled_false

Per the operator mandate 2026-05-28 (Slot 1 of cap=2 atomic-pairing): this is
the MLX-FIRST sister trainer for Z7-Mamba-2 within the cooperative-receiver
paradigm class (sister of Z6-v2 canonical pose-axis pattern per CLAUDE.md
"INDIVIDUALLY-FRACTAL" 11th standing directive 2026-05-27).

The Z6-v2 Wave N+5 anchor empirically established pose-axis 3.74x reduction at
50ep/600pair/MLX-LOCAL via the canonical mlx_score_aware harness + Hinton-
distilled scorer-bound gradient. THIS trainer is the canonical Z7-Mamba-2
sister-architecture probe per Catalog #308 (N>=3 alternative-probe-methodology)
within the SAME paradigm class.

**SCAFFOLD STATE (Catalog #240 + #325 OPTIMAL_FORM_DISPATCH_OK):**

This trainer is L1 SCAFFOLD per Catalog #240 (recipe-vs-trainer-state
consistency) + Catalog #220 (operational mechanism status:
``pre_build_substrate_engineering``). The ``_full_main`` body RAISES
``NotImplementedError`` because the canonical harness requires
``mlx.nn.value_and_grad(self.model, _loss_fn_inner)`` per
``src/tac/substrates/_shared/mlx_score_aware/adapter.py`` line 161, but the
existing ``Z7Mamba2MLXNativeRenderer`` at
``src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py`` is a plain
Python class — it does NOT extend ``mlx.nn.Module`` and does NOT expose
``.parameters()`` per the harness contract.

**OPERATOR-ROUTABLE REACTIVATION CRITERIA** per the landing memo at
``.omx/research/z7_mamba2_state_space_hinton_distill_600pair_long_mlx_landed_20260528.md``
section 10 "Operator-routable next":

1. Author ``src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_module.py``
   with ``class Z7Mamba2MLXModule(mlx.nn.Module)`` wrapping the existing
   ``Z7Mamba2MLXNativeRenderer`` parameters as either:
   (a) nested ``mlx.nn.Linear`` + ``mlx.nn.Conv2d`` submodules (most canonical;
       mirrors ``Z6V2SubstrateMLX`` at
       ``src/tac/substrates/z6_v2_cargo_cult_unwind/mlx_renderer.py:252``),
   (b) ``mx.array`` attributes registered via ``self.update({...})`` (less
       canonical but lower-touch).
2. Implement ``reconstruct_pair(pair_indices_np) -> (rgb_0, rgb_1, latents)``
   delegating to the existing forward logic.
3. Sister tests at
   ``src/tac/substrates/time_traveler_l5_z7_mamba2/tests/test_mlx_module.py``
   verifying byte-parity vs ``Z7Mamba2MLXNativeRenderer.export_state_dict``.
4. Replace the ``_full_main`` body's NotImplementedError with the canonical
   Z6-v2 pattern from
   ``experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`` lines
   116-272 (substitute ``Z7Mamba2MLXModule`` for ``Z6V2SubstrateMLX`` and
   ``Z7Mamba2MLXRenderConfig`` for ``Z6V2Config``; keep all canonical harness
   + Hinton-distilled SegNet + PoseNet teacher wiring UNCHANGED).
5. Author ``src/tac/substrates/time_traveler_l5_z7_mamba2/archive_candidate.py``
   sister of ``src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py``
   (~100 LOC) for MLX state_dict -> Z7MCM2 archive bridge.
6. Run 600-pair MLX-LOCAL ~2000 epoch training on M5 Max ($0 GPU).
7. Run per-axis decomposition vs Z6-v2 paired-anchor comparison.
8. APPEND empirical anchor to ``z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1``
   canonical equation in
   ``.omx/state/canonical_equations_registry.jsonl``.

**Smoke command (RUNNABLE)::**

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py \\
        --output-dir .omx/research/z7_mamba2_mlx_smoke_<utc> --smoke

The smoke validates that the existing ``Z7Mamba2MLXNativeRenderer`` constructs
end-to-end + a single forward pass produces correctly-shaped output. The smoke
WORKS today (the renderer is functional; only the MLX-FIRST training harness
binding is blocked).

**Full command (BLOCKED per Catalog #240 until migration completes)::**

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py \\
        --full --output-dir .omx/research/z7_mamba2_mlx_long_<utc> \\
        --num-pairs 600 --epochs 2000

[verified-against: experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py canonical pattern]
[verified-against: src/tac/substrates/_shared/mlx_score_aware/adapter.py harness contract]
[verified-against: src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py Z7Mamba2MLXNativeRenderer API]
[verified-against: .omx/research/z7_mamba2_state_space_hinton_distill_600pair_long_mlx_landed_20260528.md landing memo §10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Canonical Tier-1 operator-required flags manifest per Catalog #151:
# every operator-authorize wrapper must thread these env vars -> CLI flags per
# Catalog #152 required-input validation. The MLX-LOCAL trainer is research-
# only per Catalog #240, so this manifest is a SCAFFOLD declaration; the
# canonical Z6-v2 sister at
# `experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py` is the
# reference shape for the post-migration full manifest.
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict] = {
    "--num-pairs": {
        "env": "Z7_MAMBA2_MLX_NUM_PAIRS",
        "rationale": (
            "Contest pair count for 600-pair MLX-LOCAL long-run training "
            "(Catalog #325 per-substrate symposium-evidence sister of Z6-v2 "
            "Wave N+5 anchor 3.74x pose-axis reduction at 50ep/600pair)."
        ),
        "default": "600",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "Z7_MAMBA2_MLX_EPOCHS",
        "rationale": (
            "Number of MLX-LOCAL training epochs. LONG run smoke -> short "
            "(e.g. 50 for paired-comparison vs Z6-v2 Wave N+5 anchor); LONG "
            "production target ~2000 epochs per CLAUDE.md MLX-FIRST $0 GPU."
        ),
        "default": "100",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "Z7_MAMBA2_MLX_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full score-aware training (Catalog #114; "
            "real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _full_main(args: argparse.Namespace) -> int:
    """Run the canonical MLX-first score-aware ``_full_main`` body for Z7-Mamba-2.

    **BLOCKED PER CATALOG #240 + LANDING MEMO §10 OPERATOR-ROUTABLE.**

    The canonical Z6-v2 sister pattern at
    ``experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py`` lines
    116-272 routes through the canonical
    ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
    harness, which calls ``mlx.nn.value_and_grad(self.model, _loss_fn_inner)``
    at ``src/tac/substrates/_shared/mlx_score_aware/adapter.py`` line 161.

    The existing ``Z7Mamba2MLXNativeRenderer`` at
    ``src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py`` does NOT
    extend ``mlx.nn.Module`` and does NOT expose ``.parameters()``. The
    canonical harness call fails closed because ``mlx.nn.value_and_grad``
    cannot differentiate a non-Module model.

    **REACTIVATION CRITERIA** per landing memo §10:

    1. Land ``src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_module.py``
       with ``Z7Mamba2MLXModule(mlx.nn.Module)`` wrapping the existing
       parameter set.
    2. Land sister tests verifying byte-parity vs the existing reference.
    3. Replace this NotImplementedError body with the canonical Z6-v2
       trainer pattern (substitute ``Z7Mamba2MLXModule`` for
       ``Z6V2SubstrateMLX``).
    4. Land ``src/tac/substrates/time_traveler_l5_z7_mamba2/archive_candidate.py``
       MLX-state_dict -> Z7MCM2 archive bridge.
    """
    raise NotImplementedError(
        "Z7-Mamba-2 MLX-FIRST `_full_main` is BLOCKED per Catalog #240 + "
        "landing memo at .omx/research/"
        "z7_mamba2_state_space_hinton_distill_600pair_long_mlx_landed_20260528.md "
        "section 10 'Operator-routable next'. The canonical harness requires "
        "`mlx.nn.value_and_grad(self.model, _loss_fn_inner)`, but the existing "
        "`Z7Mamba2MLXNativeRenderer` is a plain Python class — it does NOT "
        "extend `mlx.nn.Module`. Reactivation criteria: (1) author "
        "`src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_module.py` with "
        "`Z7Mamba2MLXModule(mlx.nn.Module)`; (2) byte-parity tests; (3) replace "
        "this NotImplementedError body with the canonical Z6-v2 pattern at "
        "`experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py` "
        "lines 116-272; (4) author `archive_candidate.py` bridge. See landing "
        "memo §10 for full migration plan + estimated wall-clock (~2-4h "
        "migration + ~12-24h MLX-LOCAL 2000-epoch training on M5 Max; $0 GPU)."
    )


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-local smoke manifest (config + renderer + 1 forward; no training).

    Validates that the existing ``Z7Mamba2MLXNativeRenderer`` constructs
    end-to-end and produces correctly-shaped output. This smoke WORKS today
    (the renderer is functional; only the MLX-FIRST training harness binding
    is blocked per ``_full_main`` above).
    """
    try:
        import mlx.core as mx
    except Exception as exc:
        print(
            f"ERROR: MLX is not available on this host: {exc!r}. The MLX-local "
            "smoke requires Apple Silicon with the ``mlx`` package installed.",
            file=sys.stderr,
        )
        return 2

    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXNativeRenderer,
        Z7Mamba2MLXRenderConfig,
    )

    # Per Catalog #192/#317/#341 + canonical sister Z6V2SubstrateMLX literal:
    # MLX-LOCAL artifacts are NON-PROMOTABLE [macOS-MLX research-signal].
    MLX_EVIDENCE_GRADE = "[macOS-MLX research-signal]"
    SCHEMA_VERSION = "z7_mamba2_mlx_native_v1_20260518"

    cfg = Z7Mamba2MLXRenderConfig(num_pairs=min(int(args.num_pairs), 8))
    renderer = Z7Mamba2MLXNativeRenderer(cfg, seed=int(args.seed))
    # Single forward to validate the architecture binds end-to-end.
    idx_np = list(range(min(4, cfg.num_pairs)))
    import numpy as np
    rgb_0, rgb_1 = renderer(np.array(idx_np, dtype=np.int64))
    mx.eval(rgb_0)
    mx.eval(rgb_1)
    out_shape_0 = tuple(int(s) for s in rgb_0.shape)
    out_shape_1 = tuple(int(s) for s in rgb_1.shape)
    expected_shape = (
        min(4, cfg.num_pairs), 3,
        int(cfg.output_height), int(cfg.output_width),
    )
    if out_shape_0 != expected_shape or out_shape_1 != expected_shape:
        print(
            f"ERROR: MLX renderer output shape mismatch — got rgb_0={out_shape_0}, "
            f"rgb_1={out_shape_1}, expected {expected_shape}",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(
        args.output_dir
        or ".omx/research/z7_mamba2_mlx_local_smoke"
    )
    output_dir_str = str(output_dir.resolve())
    if output_dir_str.startswith(("/tmp/", "/private/tmp/")):
        print(
            f"ERROR: output-dir {output_dir} under /tmp per CLAUDE.md "
            "FORBIDDEN_PATTERN 'Forbidden /tmp paths in any persisted artifact'",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_manifest = {
        "schema_version": (
            "z7_mamba2_mlx_local_smoke_manifest_v1_20260528"
        ),
        "substrate_id": "time_traveler_l5_z7_mamba2_mlx_local",
        "lane_id": (
            "lane_slot_1_z7_mamba2_hinton_distill_600pair_long_mlx_20260528"
        ),
        "renderer_module": (
            "tac.substrates.time_traveler_l5_z7_mamba2.mlx_native"
        ),
        "renderer_schema_version": SCHEMA_VERSION,
        "renderer_num_parameters": "deferred_pending_mlx_module_migration",
        "config": {
            "latent_dim": int(cfg.latent_dim),
            "ego_motion_dim": int(cfg.ego_motion_dim),
            "d_model": int(cfg.d_model),
            "d_state": int(cfg.d_state),
            "expand": int(cfg.expand),
            "d_conv": int(cfg.d_conv),
            "num_pairs": int(cfg.num_pairs),
            "decoder_embed_dim": int(cfg.decoder_embed_dim),
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": int(cfg.decoder_num_upsample_blocks),
            "output_height": int(cfg.output_height),
            "output_width": int(cfg.output_width),
            "stateful": bool(cfg.stateful),
        },
        "forward_smoke": {
            "input_indices": idx_np,
            "rgb_0_shape": list(out_shape_0),
            "rgb_1_shape": list(out_shape_1),
            "rgb_0_min": float(mx.min(rgb_0)),
            "rgb_0_max": float(mx.max(rgb_0)),
            "rgb_0_mean": float(mx.mean(rgb_0)),
            "rgb_1_min": float(mx.min(rgb_1)),
            "rgb_1_max": float(mx.max(rgb_1)),
            "rgb_1_mean": float(mx.mean(rgb_1)),
        },
        "forward_convention": "reconstruct_pair_nchw01",
        "evidence_grade": MLX_EVIDENCE_GRADE,
        "axis_tag": MLX_EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "predicted_delta_adjustment": 0.0,
        "scaffold_state": "L1_scaffold_full_main_blocked_pending_mlx_module_migration",
        "operator_routable_reactivation_criteria_memo": (
            ".omx/research/"
            "z7_mamba2_state_space_hinton_distill_600pair_long_mlx_landed_20260528.md"
        ),
        "canonical_provenance": {
            "kind": "predicted_from_model",
            "evidence_grade": "predicted",
            "axis_tag": MLX_EVIDENCE_GRADE,
            "score_claim_valid": False,
            "promotable": False,
            "rationale": (
                "MLX-local smoke produces no score; this manifest documents "
                "renderer construction + single forward pass only. Non-promotable "
                "by construction per Catalog #192/#317/#341. Full training path "
                "BLOCKED pending Z7Mamba2MLXNativeRenderer -> mlx.nn.Module "
                "migration per landing memo §10."
            ),
        },
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[z7_mamba2_mlx_local smoke] manifest written to: {manifest_path} "
        f"(rgb_0_shape={out_shape_0}; {MLX_EVIDENCE_GRADE}) "
        f"non-promotable per Catalog #341; full path BLOCKED per landing memo §10",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Z7-Mamba-2 state-space MLX-first score-aware trainer "
            "(L1 SCAFFOLD; sister of Z6-v2 cooperative-receiver per "
            "Catalog #311 + #312)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the MLX-local smoke (renderer construction + 1 forward pass).",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run the full MLX-LOCAL training body. BLOCKED per Catalog #240 + "
            "landing memo §10 until Z7Mamba2MLXNativeRenderer -> mlx.nn.Module "
            "migration completes."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-pairs", type=int, default=600)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--distillation-weight", type=float, default=0.5)
    p.add_argument("--pose-distillation-weight", type=float, default=1.0)
    p.add_argument(
        "--allow-mock-scorer-teacher",
        action="store_true",
        help=(
            "Allow scorer-blind mock teacher (reconstruction-proxy distill); "
            "default is REAL SegNet + PoseNet teacher per Catalog #164."
        ),
    )
    p.add_argument("--full-lr", type=float, default=1e-3)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.full and args.smoke:
        print(
            "ERROR: --full and --smoke are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    # Default to smoke when neither flag is set (no surprises on Apple Silicon
    # operator workstations).
    return _smoke_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
