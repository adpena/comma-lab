# SPDX-License-Identifier: MIT
"""Z6-v2 cargo-cult-unwind MLX-first score-aware trainer — L1 LONG-RUN MLX-LOCAL.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

Z6-V2 L0→L1 LONG RUN MLX-LOCAL 2026-05-28: promotes Z6-v2 from L0 SCAFFOLD
(commit ``afa5ba837``) to L1 LONG-RUN MLX-LOCAL via the canonical pattern
landed by sister PACT-NeRV-IA3 (commit ``9ecc75a2d``) + sister PACT-NeRV-
SELECTOR cascade (V2 ``fee801ac7`` / V3 ``2f69d0ea6`` / V4 ``f013736de``).

Per operator NON-NEGOTIABLE 2026-05-26 8th MLX-FIRST standing directive
REINFORCED 2026-05-28 ("always prefer MLX first always") + CLAUDE.md
"PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" + the
11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27.

Z6-v2 distinguishing primitives (per Catalog #272 contract; from architecture.py):

1. 2-level Rao-Ballard hierarchical FiLM-ego-motion predictor (depth=3, ~307K
   params per design memo Candidate 1; level-0 micro = blocks 0-2; level-1
   meso = blocks 3-6).
2. FoE (focus-of-expansion) ego-motion prior conditioning (per-pair 6-dim
   tx/ty/tz/rx/ry/rz vector feeds FiLM at every block).
3. Atick-Redlich cooperative-receiver gradient binding at score_aware_loss.py
   per Catalog #311 sister Z4 routing.

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD: separate from sister
PACT-NeRV cascade trainers (no shared-helper shortcut at the trainer layer).
The canonical mlx_score_aware harness is ADOPT_CANONICAL per Catalog #290.

Canonical-vs-unique decision per layer (Catalog #290):

- ADOPT_CANONICAL: training loop / EMA / score-aware loss harness / Provenance
  / posterior anchor (``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``).
- ADOPT_CANONICAL: HNeRV-class base decoder backbone (DepthSep + SIREN +
  PixelShuffle x7) — same as PACT-NeRV cascade per the empirically validated
  PR95/PR101/PR110 medal-class topology.
- FORK_BECAUSE_PRINCIPLED_MISMATCH: 2-level Rao-Ballard FiLM-ego-motion
  predictor + FoE ego conditioning + Atick-Redlich cooperative-receiver
  (Z6-v2's UNIQUE substrate-distinguishing primitives per Catalog #272).

Dispatch gating (Catalog #325): MLX-LOCAL ONLY ($0 M5 Max); the harness
fails closed on a non-MLX host (NO CPU/CUDA paid-dispatch leak per Catalog
#1 + #317). The matching PyTorch-sister recipe stays
``dispatch_enabled: false`` + ``research_only: true``; output from this
MLX-LOCAL trainer is non-promotable ``[macOS-MLX research-signal]`` per
Catalog #192/#341. Per-substrate symposium per Catalog #325 + MLX→PyTorch
bridge + paired CUDA/CPU anchor remain DEFERRED to the PyTorch sister L2 /
paid-dispatch path.

Usage
-----

Smoke (CPU/MLX, manifest only)::

    .venv/bin/python experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py \\
        --output-dir experiments/results/z6_v2_mlx_smoke_<utc> --smoke

Full LONG run (MLX-LOCAL M5 Max, real video, score-aware)::

    .venv/bin/python experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py \\
        --full --output-dir experiments/results/z6_v2_mlx_long_<utc> \\
        --epochs 2000 --num-pairs 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Catalog #151 manifest (ast.AnnAssign per Catalog #168).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "Z6_V2_MLX_OUTPUT_DIR",
        "rationale": (
            "Output dir for Z6-v2 MLX-local training artifacts: training_artifact "
            "JSON + EMA checkpoint + observability surface (NOT /tmp per "
            "Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "Z6_V2_MLX_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. LONG run smoke -> short "
            "follow-up runs; 100-2000ep is the canonical pre-paid-dispatch "
            "research-signal window per Catalog #325."
        ),
        "default": "100",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "Z6_V2_MLX_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full score-aware training (Catalog #114; "
            "real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _full_main(args: argparse.Namespace) -> int:
    """Run the canonical MLX-first score-aware ``_full_main`` body for Z6-v2.

    Routes through the canonical substrate-AGNOSTIC harness
    ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
    (sister of ``pact_nerv_ia3_mlx_local`` / ``pact_nerv_selector_v[234]_mlx_local``).

    Per CLAUDE.md "MLX portable-local-substrate authority": the harness auto-
    stamps the canonical non-promotable markers (``score_claim=False``,
    ``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``).

    Distillation defaults to ``0.0`` (pure reconstruction); operator opts INTO
    the gradient-reachable Hinton-KL T=2.0 scorer surrogate per Catalog #164.
    """
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        DEFAULT_POSE_DIMS,
        DEFAULT_SEGNET_CLASSES,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )
    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import Z6V2Config
    from tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate import (
        export_z6_v2_mlx_archive,
    )
    from tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer import (
        Z6V2SubstrateMLX,
    )

    cfg = Z6V2Config(num_pairs=int(args.num_pairs))
    model = Z6V2SubstrateMLX(cfg)
    out_h, out_w = int(cfg.output_height), int(cfg.output_width)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )

    # Canonical Hinton-distilled scorer surrogate wiring per IA3 sister commit
    # b551bfd34 + SELECTOR-V3 sister commit ab650cc78 + V2/V4/VQ sister cascade
    # commit 1860ea2ac + V2+V4+VQ 600-pair Hinton landed commit 84a4893e4. Per
    # 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27: this is Z6-v2's
    # OWN engineering pass — the cooperative-receiver paradigm (Rao-Ballard
    # hierarchical predictive coding + ego-motion FoE conditioning per Catalog
    # #311) under Hinton-distilled scorer-bound gradient binds the canonical
    # Hinton-Vinyals-Dean 2014 KL T=2.0 + pose-MSE composition to Z6-v2's
    # specific substrate-distinguishing primitives per Catalog #272.
    #
    # Cross-family hypothesis (operator-routable from V2+V4+VQ 600-pair parity
    # landing memo 2026-05-28): does the COOPERATIVE-RECEIVER paradigm under
    # Hinton-distilled scorer-bound gradient produce empirically DIFFERENT
    # convergence signature than the PACT-NeRV per-method-saturated parity
    # floor 3.40? Canonical equation #1 anchor 15 -> 16 (cross-family scope
    # expansion fires Catalog #371 auto-recalibration trigger).
    #
    # When --distillation-weight > 0 AND NOT --allow-mock-scorer-teacher we
    # bind the REAL SegNet + REAL PoseNet teacher caches + learnable student
    # heads per canonical Hinton-Vinyals-Dean 2014 KL T=2.0 + pose-MSE
    # composition. Per CLAUDE.md "SegNet vs PoseNet importance" + Catalog #164
    # the harness bundle.__post_init__ fail-closes on missing pose teacher
    # when distillation_weight > 0 (C6 IBPS / DreamerV3 scorer-blindness
    # lesson). Z6-v2 output (384, 512) matches canonical SegNet/PoseNet eval
    # resolution exactly (zero adapter).
    scorer_teacher = None
    pose_scorer_teacher = None
    learnable_student_head = None
    learnable_pose_student_head = None
    pose_distillation_weight = 0.0
    if (
        float(args.distillation_weight) > 0.0
        and not bool(args.allow_mock_scorer_teacher)
    ):
        bundle_no_teacher = RendererBundle(
            model=model,
            target_rgb_0=target_rgb_0,
            target_rgb_1=target_rgb_1,
            num_pairs=int(args.num_pairs),
            forward_convention="call_b2chw_255",
            distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_dims=DEFAULT_POSE_DIMS,
        )
        scorer_teacher = build_mlx_segnet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",  # CLAUDE.md "MPS auth eval is NOISE" - CPU teacher only.
        )
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",
        )
        learnable_student_head = build_learnable_student_head(
            num_classes=DEFAULT_SEGNET_CLASSES,
            in_channels=3,
            seed=int(args.seed),
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=DEFAULT_POSE_DIMS,
            seed=int(args.seed),
        )
        pose_distillation_weight = float(args.pose_distillation_weight)

    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="call_b2chw_255",
        distillation_weight=float(args.distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        pose_distillation_weight=pose_distillation_weight,
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=DEFAULT_POSE_DIMS,
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        export_archive_fn=export_z6_v2_mlx_archive,
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="z6_v2_cargo_cult_unwind_mlx_local",
        lane_id="lane_z6_v2_cargo_cult_unwind_l1_long_run_mlx_local_20260528",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        pr95_faithful_curriculum_enabled=bool(
            getattr(args, "pr95_faithful_curriculum_enabled", False)
        ),
        pr95_curriculum_total_epochs=getattr(
            args, "pr95_curriculum_total_epochs", None
        ),
        notes=(
            "Z6-v2 cargo-cult-unwind MLX-first score-aware LONG-RUN training "
            "via canonical mlx_score_aware harness; real contest video + "
            "reconstruction + optional Hinton-KL T=2.0 scorer surrogate; "
            "2-level Rao-Ballard hierarchical FiLM-ego-motion predictor "
            "(depth=3 ~307K params; level-0 micro blocks 0-2 + level-1 meso "
            "blocks 3-6) is the substrate-distinguishing primitive per "
            "Catalog #272 with Atick-Redlich cooperative-receiver gradient "
            "binding per Catalog #311; non-promotable [macOS-MLX research-"
            "signal] per Catalog #192/#317/#341; per-axis + MLX->PyTorch "
            "bridge + paired CUDA/CPU anchor DEFERRED to sister L2 + per-"
            "substrate symposium Catalog #325."
        ),
    )
    print(
        f"[z6_v2_mlx_local:_full_main] DONE "
        f"epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-local smoke manifest (config + renderer + 1 forward; no training)."""
    try:  # pragma: no cover — exercised on Apple Silicon with MLX installed.
        import mlx.core as mx
    except Exception as exc:
        print(
            f"ERROR: MLX is not available on this host: {exc!r}. The MLX-local "
            "smoke requires Apple Silicon with the ``mlx`` package installed.",
            file=sys.stderr,
        )
        return 2

    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import Z6V2Config
    from tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        Z6V2SubstrateMLX,
    )

    cfg = Z6V2Config(num_pairs=min(int(args.num_pairs), 8))
    model = Z6V2SubstrateMLX(cfg)
    num_params = int(model.num_parameters())
    # Single forward to validate the architecture binds end-to-end.
    idx = mx.array(list(range(min(4, cfg.num_pairs))), dtype=mx.int32)
    output = model(idx)
    mx.eval(output)
    output_shape = tuple(int(s) for s in output.shape)
    expected_shape = (
        min(4, cfg.num_pairs), 2, 3,
        int(cfg.output_height), int(cfg.output_width),
    )
    if output_shape != expected_shape:
        print(
            f"ERROR: MLX renderer output shape mismatch — got {output_shape}, "
            f"expected {expected_shape}",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(
        args.output_dir
        or ".omx/research/z6_v2_cargo_cult_unwind_mlx_local_smoke"
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
        "schema_version": "z6_v2_cargo_cult_unwind_mlx_smoke_manifest_v1_20260528",
        "substrate_id": "z6_v2_cargo_cult_unwind_mlx_local",
        "lane_id": "lane_z6_v2_cargo_cult_unwind_l1_long_run_mlx_local_20260528",
        "renderer_module": "tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer",
        "renderer_schema_version": SCHEMA_VERSION,
        "renderer_num_parameters": num_params,
        "config": {
            "latent_dim": int(cfg.latent_dim),
            "ego_dim": int(cfg.ego_dim),
            "embed_dim": int(cfg.embed_dim),
            "initial_grid_h": int(cfg.initial_grid_h),
            "initial_grid_w": int(cfg.initial_grid_w),
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": float(cfg.sin_frequency),
            "num_upsample_blocks": int(cfg.num_upsample_blocks),
            "rao_ballard_level_boundary": int(cfg.rao_ballard_level_boundary),
            "film_generator_depth": int(cfg.film_generator_depth),
            "film_hidden_width": int(cfg.film_hidden_width),
            "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
            "num_pairs": int(cfg.num_pairs),
            "output_height": int(cfg.output_height),
            "output_width": int(cfg.output_width),
        },
        "forward_smoke": {
            "input_indices": [int(v) for v in idx.tolist()],
            "output_shape": list(output_shape),
            "output_min": float(mx.min(output)),
            "output_max": float(mx.max(output)),
            "output_mean": float(mx.mean(output)),
        },
        "forward_convention": "call_b2chw_255",
        "evidence_grade": MLX_EVIDENCE_GRADE,
        "axis_tag": MLX_EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "predicted_delta_adjustment": 0.0,
        "canonical_provenance": {
            "kind": "predicted_from_model",
            "evidence_grade": "predicted",
            "axis_tag": MLX_EVIDENCE_GRADE,
            "score_claim_valid": False,
            "promotable": False,
            "rationale": (
                "MLX-local smoke produces no score; this manifest documents "
                "renderer construction + single forward pass only. Non-promotable "
                "by construction per Catalog #192/#317/#341."
            ),
        },
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[z6_v2_mlx_local smoke] manifest written to: {manifest_path} "
        f"(num_params={num_params}; output_shape={output_shape}) "
        f"{MLX_EVIDENCE_GRADE} non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Z6-v2 cargo-cult-unwind MLX-first score-aware trainer "
            "(L1 LONG-RUN MLX-LOCAL 2026-05-28)."
        )
    )
    p.add_argument("--smoke", action="store_true", help="Emit smoke manifest only.")
    p.add_argument(
        "--full", action="store_true",
        help="Run full MLX score-aware training via the canonical harness.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-pairs", type=int, default=32,
        help="Trainable pair count (32 for LONG; smoke caps at 8).",
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="MLX score-aware epochs (--full).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Output dir (NOT /tmp).",
    )
    p.add_argument(
        "--video-path", type=Path, default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full score-aware training (Catalog #114).",
    )
    p.add_argument("--full-lr", type=float, default=1e-3)
    p.add_argument(
        "--distillation-weight", type=float, default=0.0,
        help=(
            "Weight on the gradient-reachable Hinton-KL T=2.0 scorer surrogate "
            "term in the --full score-aware loss (0.0 disables). >0 + NOT "
            "--allow-mock-scorer-teacher binds the REAL SegNet + REAL PoseNet "
            "teacher cache via canonical "
            "build_mlx_segnet_pair_teacher/build_mlx_posenet_pair_teacher per "
            "the IA3 sister commit b551bfd34 + V3 sister commit ab650cc78 + "
            "V2/V4/VQ sister cascade commit 1860ea2ac + V2+V4+VQ 600-pair "
            "Hinton landed commit 84a4893e4 + Catalog #164."
        ),
    )
    p.add_argument(
        "--pose-distillation-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the POSE-MSE distillation term per CLAUDE.md 'SegNet "
            "vs PoseNet importance' operating-point-dependent discipline "
            "(pose is DOMINANT at frontier). Default 1.0 wires both scorers "
            "(PoseNet REQUIRED at frontier unless allow_segnet_only_research "
            "is opted into). Used only when --distillation-weight > 0 AND "
            "NOT --allow-mock-scorer-teacher."
        ),
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help=(
            "Upstream repo path containing SegNet + PoseNet safetensors for "
            "the real teacher cache build (canonical for the Hinton-distilled "
            "scorer surrogate wire-in)."
        ),
    )
    p.add_argument(
        "--allow-mock-scorer-teacher", action="store_true",
        help=(
            "EXPLICIT opt-in to the scorer-BLIND deterministic-cosine mock "
            "teacher when --distillation-weight > 0 AND no real scorer_teacher "
            "is wired. Default OFF — the harness fails closed otherwise. Set "
            "ONLY for a $0 no-real-SegNet smoke that explicitly accepts the "
            "result is reconstruction-proxy (NOT scorer-bound)."
        ),
    )
    p.add_argument(
        "--pr95-faithful-curriculum-enabled",
        action="store_true",
        default=False,
        help=(
            "Opt-in to PR95-faithful 8-stage Muon+AdamW canonical curriculum "
            "per CLAUDE.md 'HNeRV / leaderboard-implementation parity "
            "discipline' L14 (canonical 8-stage 29,650-epoch curriculum) + L15 "
            "(Muon optimizer in final stage only) + the optimizer stack "
            "research memo Option A MINIMUM-VIABLE + the m9-v3 canonical "
            "helper (commit c91481212). Default OFF preserves the legacy "
            "default-on AdamW behavior. When ON, the canonical harness routes "
            "per-stage optimizer state through apply_pr95_mlx_optimizer_step "
            "via PR95FaithfulCurriculumFactory so each canonical stage "
            "actually uses its declared optimizer + loss_family + sigma + "
            "lambda + qat hyperparameters per CLAUDE.md 'NO FAKE "
            "IMPLEMENTATIONS' non-negotiable."
        ),
    )
    p.add_argument(
        "--pr95-curriculum-total-epochs",
        type=int,
        default=None,
        help=(
            "Total epoch budget for the PR95-faithful 8-stage curriculum; "
            "defaults to the canonical 29,650 per L14 (sum of 3000+5650+1500+"
            "500+9000+2000+3000+5000). Smaller budgets proportionally scale "
            "each stage's epoch share per the canonical ratio. Used only "
            "when --pr95-faithful-curriculum-enabled."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    _build_parser().print_help()
    return 1


__all__ = ["_full_main", "_smoke_main", "main"]


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
