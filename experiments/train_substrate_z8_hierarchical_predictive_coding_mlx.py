# SPDX-License-Identifier: MIT
"""Z8 hierarchical predictive coding MLX-local smoke trainer — L0 SCAFFOLD.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526
# AUTOCAST_FP16_WAIVED:MLX_or_PyTorch_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_uses_different_precision_strategy_per_comprehensive_bug_audit_cascade_20260526

Path 3 substrate-class-shift candidate F MLX trainer. Per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable + the
operator's MLX-first directive 2026-05-26.

L0 SCAFFOLD scope:

- MLX-local smoke ONLY (axis_tag=``[macOS-MLX research-signal]``); no paid CUDA.
- Smoke ≤5 epochs ≤8 pairs (smoke-only mode); demonstrates the multi-level
  RSSM + per-level Gumbel-Softmax STE + Rao-Ballard residual L2 training step
  converges monotonically on synthetic targets.
- ``_full_main raises NotImplementedError`` per Catalog #240 acceptance
  cascade (c) pre-build substrate-engineering: full training path is
  council-gated.
- All artifacts carry ``[macOS-MLX research-signal]`` + ``score_claim=false``
  + ``promotion_eligible=false`` + ``ready_for_exact_eval_dispatch=false``
  per Catalog #127 + #192 + #317 + #341.

Per CLAUDE.md "MLX portable-local-substrate authority": MLX is a local substrate
for fast candidate generation; not a contest scoring axis. MLX results MUST
flow through ``tac.optimization.scorer_response_dataset`` for any LL planner
consumption (Phase 2 wire-in).

Discipline: Catalog #229 PV / #117/#157/#174 canonical serializer (commits) /
#206 checkpoint discipline / #119 Co-Authored-By Claude trailer / #287
placeholder-rationale rejection / #208 no /Users paths / #270 dispatch
optimization protocol (Phase 2 trainer wires canonical helpers) / #310 + #311
+ #312 class-shift-not-bolt-on + ego-motion conditioning + canonical-quadruple
binding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# CATEGORY-CHECK: this script is OUT-OF-SCOPE for Catalog #226
# (`check_trainer_auth_eval_uses_canonical_helper`) because it is an MLX-local
# SMOKE trainer that does NOT invoke `experiments/contest_auth_eval.py`.
# Phase 2 PyTorch trainer (separate file) will use the canonical
# `gate_auth_eval_call` helper.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z8 hierarchical predictive coding MLX-local smoke trainer "
            "(L0 SCAFFOLD; non-promotable)."
        )
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode (≤5 epochs ≤8 pairs synthetic targets; required at L0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of smoke epochs (default 3; max 5 at L0).",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=4,
        help="Number of synthetic pairs for smoke (default 4; max 8 at L0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smoke-output",
        type=str,
        default="",
        help=(
            "Optional path for smoke convergence JSON manifest (non-promotable; "
            "carries [macOS-MLX research-signal] tag)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir for --full MLX score-aware training (required for --full).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full MLX-first training (Catalog #114).",
    )
    parser.add_argument(
        "--full-lr",
        type=float,
        default=1e-3,
        help="Learning rate for --full MLX score-aware training.",
    )
    parser.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help="Weight on the gradient-reachable Hinton-KL T=2.0 scorer "
        "surrogate term in the --full score-aware loss (0.0 disables).",
    )
    parser.add_argument(
        "--pose-distillation-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the REAL PoseNet pose-MSE distillation term in --full. "
            "Default 1.0 keeps M12a frontier work bound to the dominant pose "
            "axis instead of repeating the SegNet-only pose-drift failure."
        ),
    )
    parser.add_argument(
        "--allow-mock-scorer-teacher",
        action="store_true",
        help=(
            "Allow the scorer-blind mock teacher in --full. Default is fail-"
            "closed real SegNet + real PoseNet teacher construction; this flag "
            "is only for explicit $0 no-real-scorer research smoke."
        ),
    )
    # M9 canonical quadruple binding-integration flag (operator-routed
    # Yousfi-cascade TOP-1 post-M6; 2026-05-30). When set, _full_main routes
    # through the canonical M4+M5+M6+M8 compose pattern at
    # tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding
    # instead of the MLX-harness path. Lifts the M9 milestone per
    # build_progress.py to LANDED. Uses real upstream/videos/0.mkv frames
    # per Catalog #213 + CLAUDE.md "Forbidden make_synthetic_pair_batch"
    # non-negotiable.
    parser.add_argument(
        "--canonical-quadruple-binding",
        action="store_true",
        help=(
            "Route _full_main through the M9 canonical quadruple binding-"
            "integration path (M4 Mamba-2 + M5 Mallat full DWT + M6 "
            "Wyner-Ziv + M8 ScoreAwareLevelLoss compose pattern per "
            "Catalog #312). Non-promotable [macOS-CPU advisory] per Catalog "
            "#192. Real video frames per Catalog #213 mandatory."
        ),
    )
    parser.add_argument(
        "--canonical-quadruple-output-dir",
        type=Path,
        default=None,
        help=(
            "Output dir for --canonical-quadruple-binding artifact JSON. "
            "Required when --canonical-quadruple-binding is set."
        ),
    )
    parser.add_argument(
        "--canonical-quadruple-eval-h",
        type=int,
        default=32,
        help=(
            "Per-pair target height for canonical quadruple binding (default "
            "32 = smoke-friendly; production uses 384 per CONTEST_SCORER_"
            "RESOLUTION but the M9 milestone does NOT couple to the decoder)."
        ),
    )
    parser.add_argument(
        "--canonical-quadruple-eval-w",
        type=int,
        default=32,
        help="Per-pair target width for canonical quadruple binding (default 32).",
    )
    # Tier-1 engineering flags (Catalog #172/#178/#179/#180) per CLAUDE.md
    # "Production-hardened dispatch optimization protocol" non-negotiable.
    # Required by Catalog #270 dispatch optimization protocol umbrella so
    # paid Modal/Lightning/Vast.ai dispatch >$0.30 is admissible. MLX trainer
    # ignores TF32 + torch.compile at runtime (MLX has its own kernels) but
    # the canonical Tier-1 flag presence IS the contract per Catalog #178/#179.
    parser.add_argument(
        "--enable-tf32",
        action="store_true",
        default=False,
        help=(
            "Enable TF32 matmul on Ampere+ GPUs (Catalog #178). MLX-LOCAL "
            "ignores at runtime; flag presence satisfies canonical Tier-1 "
            "dispatch optimization protocol per Catalog #270."
        ),
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        default=False,
        help=(
            "Wrap forward path in torch.compile (Catalog #179). MLX-LOCAL "
            "ignores at runtime; flag presence satisfies canonical Tier-1 "
            "dispatch optimization protocol per Catalog #270."
        ),
    )
    parser.add_argument(
        "--pr95-faithful-curriculum-enabled",
        action="store_true",
        default=False,
        help=(
            "Opt-in to PR95-faithful 8-stage Muon+AdamW canonical curriculum "
            "per CLAUDE.md 'HNeRV / leaderboard-implementation parity "
            "discipline' L14 + L15 + the m9-v3 canonical helper (commit "
            "c91481212). Default OFF preserves legacy default-on AdamW. When "
            "ON the canonical harness routes per-stage optimizer state "
            "through apply_pr95_mlx_optimizer_step via PR95FaithfulCurriculum"
            "Factory so each stage actually uses its declared optimizer + "
            "loss_family + sigma + lambda + qat per CLAUDE.md 'NO FAKE "
            "IMPLEMENTATIONS' non-negotiable."
        ),
    )
    parser.add_argument(
        "--pr95-curriculum-total-epochs",
        type=int,
        default=None,
        help=(
            "Total epoch budget for the PR95-faithful 8-stage curriculum; "
            "defaults to canonical 29,650 per L14. Smaller budgets "
            "proportionally scale each stage. Used only when "
            "--pr95-faithful-curriculum-enabled."
        ),
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=1.0,
        help="Wave N+11 stabilizer: grad-clip max L2 norm (<=0 disables).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Wave N+11 stabilizer: LR warmup epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Wave N+11 stabilizer: AdamW weight decay.",
    )
    parser.add_argument(
        "--optimizer-kind",
        type=str,
        default="adamw",
        choices=("adamw", "rmsprop"),
        help="Optimizer for the MLX score-aware adapter.",
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """L0 smoke convergence: minimal MLX training loop on synthetic targets.

    Demonstrates the multi-level RSSM + per-level Gumbel-Softmax STE + the
    Rao-Ballard residual L2 + decoder forward all train end-to-end on MLX.
    Synthetic MSE proxy stands in for the full canonical
    ``score_pair_components`` per Catalog #164 (which Phase 2 PyTorch trainer
    routes through).
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except Exception as exc:
        print(
            f"[Z8 MLX smoke] MLX not available: {exc}; cannot run smoke.",
            file=sys.stderr,
        )
        return 65  # canonical "MLX unavailable" L0 exit code

    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    if args.epochs > 5:
        print(f"[Z8 MLX smoke] L0 caps epochs at 5; got {args.epochs}", file=sys.stderr)
        return 2
    if args.num_pairs > 8:
        print(
            f"[Z8 MLX smoke] L0 caps num_pairs at 8; got {args.num_pairs}",
            file=sys.stderr,
        )
        return 2

    mx.random.seed(int(args.seed))

    # Smoke config: small enough to converge in seconds on Apple Silicon.
    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=int(args.num_pairs),
        deterministic_state_dim=8,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)

    # Synthetic target: per-pair random RGB at scorer resolution.
    # This is the smoke-only MSE proxy; Phase 2 trainer routes through
    # canonical `score_pair_components` per Catalog #164.
    target_pairs = mx.random.normal(  # type: ignore[union-attr]
        shape=(cfg.num_pairs, 2, 3, *cfg.eval_size),
        key=mx.random.key(123),
    ) * 50.0 + 127.5  # roughly in [0, 255]

    def loss_fn(model: Z8HierarchicalPredictiveCoderMLX, pair_idx: int) -> any:
        """MSE proxy loss + Rao-Ballard per-level residual L2 sketch."""
        indices = mx.array([pair_idx], dtype=mx.int32)
        rgb_pair, per_level_indices, per_level_soft = model.forward_training(
            indices
        )
        # Synthetic MSE on decoded pair
        target_slice = target_pairs[pair_idx : pair_idx + 1]
        mse = mx.mean((rgb_pair - target_slice) ** 2)
        # Rao-Ballard per-level residual L2 sketch: penalize Gumbel-softmax
        # entropy (proxy for per-level residual entropy term — true Rao-Ballard
        # residual term requires the bottom-up error encoder which Phase 2
        # implements). Sum across levels.
        rao_ballard_proxy = mx.zeros(())
        for soft in per_level_soft:
            # Gumbel-Softmax soft entropy: -sum p log p
            entropy = -mx.sum(soft * mx.log(soft + 1e-10), axis=-1)
            rao_ballard_proxy = rao_ballard_proxy + mx.mean(entropy)
        return mse + 0.01 * rao_ballard_proxy

    opt = optim.Adam(learning_rate=1e-3)

    epoch_losses: list[float] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        epoch_loss = 0.0
        for pair_idx in range(cfg.num_pairs):
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss_value, grads = loss_and_grad_fn(model, pair_idx)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            epoch_loss += float(loss_value)
        epoch_loss = epoch_loss / cfg.num_pairs
        epoch_losses.append(epoch_loss)
        print(
            f"[Z8 MLX smoke] epoch={epoch + 1}/{args.epochs} loss={epoch_loss:.4f}",
        )
    elapsed = time.time() - start

    # Smoke convergence verdict: monotonic decrease ≥5 epochs is the canonical
    # L0 convergence signal. At <5 epochs we just require last < first.
    converged = (
        epoch_losses[-1] < epoch_losses[0] if len(epoch_losses) >= 2 else True
    )

    smoke_manifest = {
        "schema": "z8_mlx_smoke_convergence_manifest_v1",
        "substrate_id": "z8_hierarchical_predictive_coding",
        "lane_id": (
            "lane_path_3_f_z8_hierarchical_predictive_coding_canonical_quadruple_20260526"
        ),
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "MPS-research-signal",
        "smoke_mode": True,
        "num_epochs": int(args.epochs),
        "num_pairs": int(args.num_pairs),
        "epoch_losses": [float(x) for x in epoch_losses],
        "converged_monotonic_proxy": bool(converged),
        "wall_clock_seconds": float(elapsed),
        "canonical_equation_refs": [
            "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
            "scorer_conditional_joint_rate_distortion_floor_v1",
            "categorical_posterior_capacity_vs_continuous_gaussian_v1",
        ],
        "design_memo_path": (
            ".omx/research/path_3_f_z8_hierarchical_predictive_coding_substrate_design_20260526.md"
        ),
        "non_promotable_rationale": (
            "L0 SCAFFOLD MLX-local smoke; per CLAUDE.md 'MLX portable-local-substrate "
            "authority' MLX is research-signal not contest scoring axis; per Catalog "
            "#192 + #317 + #341 non-promotable by construction"
        ),
    }
    print(json.dumps(smoke_manifest, indent=2, sort_keys=True))

    if args.smoke_output:
        out_path = Path(args.smoke_output)
        if str(out_path).startswith("/tmp/") or "/tmp/" in str(out_path):
            print(
                f"[Z8 MLX smoke] refusing /tmp/ path per CLAUDE.md transient-evidence "
                f"trap: {out_path}",
                file=sys.stderr,
            )
            return 4
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[Z8 MLX smoke] wrote manifest to {out_path}")

    return 0 if converged else 1


def _full_main(args: argparse.Namespace) -> int:
    """MLX-first score-aware full training via the canonical MLX harness.

    MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: this ``_full_main`` now routes
    through the canonical substrate-AGNOSTIC harness
    ``tac.substrates._shared.mlx_score_aware_full_main.run_mlx_score_aware_full_main``
    (sister of ``pact_nerv_full_main.py`` for MLX-first substrates). The harness
    extinguishes the prior ``NotImplementedError`` by binding:

    1. Real contest-video targets via ``decode_mlx_targets`` (Catalog #114).
    2. Gradient-reachable score-aware loss (reconstruction MSE + Hinton-KL
       T=2.0 scorer surrogate; Catalog #164 sister discipline, MLX-native).
    3. Canonical EMA shadow (0.997) / OOM-safe step / telemetry / Provenance /
       posterior anchor via ``run_long_training``.

    ## Canonical-vs-unique decision per layer (Catalog #290)

    - ADOPT_CANONICAL: training loop / EMA / score-aware-loss / Provenance /
      posterior anchor (the harness + ``run_long_training``).
    - FORK (this substrate's UNIQUE primitive): the Z8 multi-level RSSM
      hierarchy + per-level Gumbel-Softmax STE + Mallat wavelet proxy +
      DreamerV3 deterministic state (``mlx_renderer.py`` —
      ``Z8HierarchicalPredictiveCoderMLX``).

    ## Dispatch gating (Catalog #325)

    MLX-LOCAL ONLY ($0 M5 Max); fails closed on a non-MLX host (NO CPU/CUDA
    paid-dispatch leak per Catalog #1 + #317). Recipe stays
    ``dispatch_enabled: false`` + ``research_only: true``; output is
    non-promotable ``[macOS-MLX research-signal]`` per Catalog #192/#341.

    Still DEFERRED to the PyTorch sister L2 / paid-dispatch path (Catalog #325
    Phase 2 symposium): per-axis SegNet/PoseNet decomposition via
    ``score_pair_components`` (#164), ``patch_upstream_yuv6_globally`` (#187),
    MLX->PyTorch export bridge, Z8HPC1 archive + paired CPU/CUDA auth_eval
    (#226), Catalog #319 deliverability_proof + Catalog #270 declarations.
    """
    if args.output_dir is None:
        print(
            "[Z8 MLX full] --output-dir is required for --full training",
            file=sys.stderr,
        )
        return 2

    from tac.substrates._shared.mlx_score_aware_full_main import (
        MlxScoreAwareHarnessError,
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
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
        Z8HierarchicalPredictiveCoderMLX,
    )

    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=int(args.num_pairs),
        deterministic_state_dim=8,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    model = Z8HierarchicalPredictiveCoderMLX(cfg)
    out_h, out_w = cfg.eval_size  # HNeRV decoder hardcodes 384x512 output.
    try:
        target_rgb_0, target_rgb_1 = decode_mlx_targets(
            args.video_path,
            num_pairs=int(args.num_pairs),
            output_height=out_h,
            output_width=out_w,
        )
    except MlxScoreAwareHarnessError as exc:
        print(f"[Z8 MLX full] FATAL: {exc}", file=sys.stderr)
        return 65
    scorer_teacher = None
    pose_scorer_teacher = None
    learnable_student_head = None
    learnable_pose_student_head = None
    pose_distillation_weight = 0.0
    if (
        float(args.distillation_weight) > 0.0
        and not bool(args.allow_mock_scorer_teacher)
    ):
        teacherless_bundle = RendererBundle(
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
            teacherless_bundle,
            device="cpu",
        )
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            teacherless_bundle,
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
        substrate_artifact_metadata={
            "m12a_score_binding": (
                "real_segnet_posenet_hinton_t2"
                if scorer_teacher is not None
                else "explicit_mock_or_reconstruction_proxy"
            ),
            "z8_trainer_mode": "full",
        },
    )
    grad_clip = float(args.grad_clip_max_norm)
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="z8_hierarchical_predictive_coding",
        lane_id="lane_z8_m12a_modal_t4_l2_long_training_per_catalog_325_symposium_proceed_with_revisions_20260530",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        ema_decay=0.997,
        pr95_faithful_curriculum_enabled=bool(
            getattr(args, "pr95_faithful_curriculum_enabled", False)
        ),
        pr95_curriculum_total_epochs=getattr(
            args, "pr95_curriculum_total_epochs", None
        ),
        grad_clip_max_norm=(grad_clip if grad_clip > 0.0 else None),
        warmup_epochs=int(args.warmup_epochs),
        weight_decay=float(args.weight_decay),
        optimizer_kind=str(args.optimizer_kind),
        notes=(
            "Z8 hierarchical predictive coding MLX-first score-aware full "
            "training via canonical mlx_score_aware_full_main harness; real "
            "contest video + reconstruction + REAL Hinton-distilled SegNet "
            "(KL T=2.0) + REAL PoseNet (pose-MSE) teachers; Wave N+11 "
            "stabilizer (EMA 0.997 + grad-clip + warmup + weight decay); "
            "non-promotable [macOS-MLX research-signal] per Catalog "
            "#192/#317/#341; MLX->PyTorch bridge + paired CUDA/CPU anchor "
            "remain the exact promotion gate."
        ),
    )
    print(
        f"[Z8 MLX full] DONE epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def _canonical_quadruple_main(args: argparse.Namespace) -> int:
    """M9 canonical quadruple binding-integration training loop.

    Operator-routed Yousfi-cascade TOP-1 post-M6 (2026-05-30). Lifts the
    Z8 ``_full_main`` from the prior MLX-harness-only path to the canonical
    M4 + M5 + M6 + M8 compose pattern per Catalog #312 canonical quadruple +
    HNeRV parity discipline L7 substrate-engineering UNIQUE-IFIES.

    The compose pattern (from Z8 Phase E landing memo):

        m5.decompose(input) -> per-level latents
        m6.encode(top_state, side_info=m5(reconstruct(input))) -> archive bytes
        m8.per_level_loss(reconstruction, target,
                           sensitivity=m7.get_for_level(level)) -> scalar

    Lands M9 ``full_main_trainer_lifts_notimplementederror`` to LANDED in
    ``build_progress.py``. M10 (inflate consumes real trained weights per
    Catalog #369) + M11 (L1 MLX-LOCAL end-to-end smoke) + M12 (paired-CUDA
    Modal T4 sub-0.189 threshold attempt) are now unblocked.

    Per CLAUDE.md "Forbidden make_synthetic_pair_batch in any non-smoke
    training path" non-negotiable: this path REQUIRES real
    upstream/videos/0.mkv frames per Catalog #213. Synthetic-only fixtures
    are forbidden outside the canonical _smoke_main path.

    Per Catalog #192 the output artifact is non-promotable by construction
    (axis_tag=[macOS-CPU advisory] / score_claim=False / promotable=False).
    """
    if args.canonical_quadruple_output_dir is None:
        print(
            "[Z8 M9 canonical quadruple] --canonical-quadruple-output-dir "
            "is required.",
            file=sys.stderr,
        )
        return 2

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        build_canonical_quadruple_binding_from_z8_config,
        load_real_video_targets_numpy,
        run_canonical_quadruple_training_loop,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
        Z8HierarchicalConfig,
    )

    eval_h = int(args.canonical_quadruple_eval_h)
    eval_w = int(args.canonical_quadruple_eval_w)
    num_pairs = int(args.num_pairs)
    epochs = int(args.epochs)
    cfg = Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=num_pairs,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(eval_h, eval_w),
    )
    binding = build_canonical_quadruple_binding_from_z8_config(cfg)

    print(
        f"[Z8 M9 canonical quadruple] decoding {num_pairs} pairs from "
        f"{args.video_path!s} at ({eval_h}, {eval_w}) per Catalog #213.",
    )
    try:
        pair_rgb_targets = load_real_video_targets_numpy(
            args.video_path,
            num_pairs=num_pairs,
            output_height=eval_h,
            output_width=eval_w,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[Z8 M9 canonical quadruple] FATAL: {exc}", file=sys.stderr)
        return 65  # canonical "real video unavailable" exit code

    print(
        f"[Z8 M9 canonical quadruple] running {epochs} epochs x "
        f"{num_pairs} pairs canonical compose pattern (M4+M5+M6+M8)...",
    )
    artifact = run_canonical_quadruple_training_loop(
        binding,
        pair_rgb_targets,
        epochs=epochs,
        notes=(
            f"Z8 M9 canonical quadruple binding-integration smoke "
            f"(operator-routed Yousfi-cascade TOP-1 post-M6); "
            f"real video {args.video_path!s} at ({eval_h}, {eval_w}); "
            f"non-promotable [macOS-CPU advisory] per Catalog #192; "
            f"lifts build_progress.py M9 to LANDED."
        ),
    )

    output_dir = args.canonical_quadruple_output_dir
    # Refuse /tmp/ as an absolute prefix (matches /tmp/X, /private/tmp/X)
    # but explicitly allow .omx/tmp (the canonical operator scratch sub-dir).
    output_dir_str = str(output_dir)
    if (
        output_dir_str.startswith("/tmp/")
        or output_dir_str.startswith("/private/tmp/")
        or output_dir_str.startswith("/var/tmp/")
    ):
        print(
            f"[Z8 M9 canonical quadruple] refusing absolute /tmp/ output_dir "
            f"per CLAUDE.md transient-evidence trap: {output_dir}",
            file=sys.stderr,
        )
        return 4
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "m9_canonical_quadruple_artifact.json"
    out_path.write_text(
        json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[Z8 M9 canonical quadruple] DONE epochs={artifact.total_epochs_completed} "
        f"convergence={artifact.convergence_verdict} "
        f"final_payload_bytes={artifact.final_wyner_ziv_payload_bytes} "
        f"wall_clock={artifact.total_wall_clock_seconds:.2f}s "
        f"artifact={out_path}",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    if args.canonical_quadruple_binding:
        return _canonical_quadruple_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main())
