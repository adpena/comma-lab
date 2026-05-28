# SPDX-License-Identifier: MIT
"""Z5 Rao-Ballard MLX-FIRST score-aware trainer — L1 LONG-RUN MLX-LOCAL.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only
# OPTIMAL_FORM_DISPATCH_OK:scaffold_full_main_wires_canonical_mlx_score_aware_harness_via_Z5RaoBallardSubstrateMLX_per_t4_symposium_wave_n13_z5_first_landed_20260528

Per T4 SYMPOSIUM Wave N+13 verdict ``f5d3c6835`` op-routable #1 (Z5-first
among Z4/Z5/Z6/Z7/Z8 class-shift queue) + operator NON-NEGOTIABLE 2026-05-28
+ CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
symposium" + Catalog #311 ego-motion-conditioned predictive coding canonical
+ Catalog #312 hierarchical predictive coding canonical quadruple.

Z5 is FIRST in the Z4/Z5/Z6/Z7/Z8 class-shift queue per the Carmack
engineering-shortcut applied by the T4 symposium: Rao-Ballard 1999
hierarchical predictive coding is the **simplest paradigm with the clearest
empirical anchor candidate** because (a) the 2-level hierarchy admits the
canonical identity-predictor disambiguator probe (per Catalog #308
alternative-probe methodology), (b) the predictor block is small (~few KB
weights), and (c) the per-pair residual ``r_t = z_low_t - z_low_t_pred``
mechanism is directly observable empirically against the canonical
``score_aware_loss`` Lagrangian.

Z5 distinguishing primitive (per Catalog #272):

1. **2-level Rao-Ballard hierarchy**: per-pair ``z_low`` + per-pair ``z_high``
   + per-pair ``ego_motion`` (FoE prior).
2. **Predictor**: ``(z_high_t, ego_motion_t) -> z_low_t_pred``; 2-layer
   tanh+GELU MLP per architecture.py.
3. **Residual penalty**: ``||z_low_t - z_low_t_pred||_2^2`` in the
   training Lagrangian per Atick-Redlich 1990 cooperative-receiver bound
   (Catalog #311).

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD (Catalog #290):
Z5's OWN canonical engineering pass; the predictor + 2-level latent split
is Z5-specific (NOT shared with Z6 single-level FiLM NOR Z7-Mamba-2
state-space recurrence).

Canonical-vs-unique decision per layer (Catalog #290):

- ADOPT_CANONICAL: training loop / EMA / score-aware loss harness / Provenance
  / posterior anchor (``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``).
- ADOPT_CANONICAL: HNeRV-class base decoder backbone (Conv2d + PixelShuffle
  + GELU) — same as Z6-v2 + PACT-NeRV cascade per the empirically validated
  PR95/PR101/PR110 medal-class topology.
- FORK_BECAUSE_PRINCIPLED_MISMATCH: 2-level Rao-Ballard hierarchical
  predictor + EXPLICIT ``z_high`` + ego-motion FoE conditioning (Z5's UNIQUE
  substrate-distinguishing primitives per Catalog #272).

Dispatch gating (Catalog #325): MLX-LOCAL ONLY ($0 M5 Max); the matching
PyTorch-sister recipe stays ``dispatch_enabled: false`` + ``research_only: true``.
Output from this MLX-LOCAL trainer is non-promotable ``[macOS-MLX research-signal]``
per Catalog #192/#341. Per-substrate symposium per Catalog #325 + MLX->PyTorch
bridge + paired CUDA/CPU anchor remain DEFERRED to a sister L2 / paid-dispatch
path.

Usage
-----

Smoke (Apple Silicon, manifest only)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z5_mlx_local.py \\
        --output-dir experiments/results/z5_mlx_smoke_<utc> --smoke

Full LONG run (MLX-LOCAL M5 Max, real video, score-aware)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z5_mlx_local.py \\
        --full --output-dir experiments/results/z5_mlx_long_<utc> \\
        --epochs 50 --num-pairs 600

[verified-against: experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py canonical pattern]
[verified-against: experiments/train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py canonical pattern]
[verified-against: src/tac/substrates/_shared/mlx_score_aware/adapter.py harness contract]
[verified-against: src/tac/substrates/time_traveler_l5_z5/mlx_renderer.py Z5RaoBallardSubstrateMLX API]
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
        "env": "Z5_MLX_OUTPUT_DIR",
        "rationale": (
            "Output dir for Z5 MLX-local training artifacts: training_artifact "
            "JSON + EMA checkpoint + observability surface (NOT /tmp per "
            "Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "Z5_MLX_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. SHORT smoke runs (e.g. 50) "
            "for paired-comparison anchor; LONG production target ~2000 "
            "epochs per CLAUDE.md MLX-FIRST $0 GPU."
        ),
        "default": "100",
        "required_input_file": False,
    },
    "--num-pairs": {
        "env": "Z5_MLX_NUM_PAIRS",
        "rationale": (
            "Contest pair count for 600-pair MLX-LOCAL canonical anchor "
            "(Catalog #325 per-substrate symposium-evidence sister of Z6-v2 "
            "Wave N+5 anchor 3.74x pose-axis reduction at 50ep/600pair + "
            "Z7-Mamba-2 lr=1e-4 stabilized canonical 19.2% pose-axis baseline)."
        ),
        "default": "600",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "Z5_MLX_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full score-aware training (Catalog #114; "
            "real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _full_main(args: argparse.Namespace) -> int:
    """Run the canonical MLX-first score-aware ``_full_main`` body for Z5.

    Routes through the canonical substrate-AGNOSTIC harness
    ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
    (sister of Z6-v2 + Z7-Mamba-2 canonical patterns).

    Per CLAUDE.md "MLX portable-local-substrate authority": the harness auto-
    stamps the canonical non-promotable markers (``score_claim=False``,
    ``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``).

    Distillation defaults to ``0.5`` for SegNet + ``1.0`` for PoseNet (the
    canonical operating-point per CLAUDE.md "SegNet vs PoseNet importance" —
    pose is DOMINANT at the PR106 frontier operating point).
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
    from tac.substrates.time_traveler_l5_z5.architecture import Z5RaoBallardConfig
    from tac.substrates.time_traveler_l5_z5.archive_candidate import (
        export_z5_mlx_archive,
    )
    from tac.substrates.time_traveler_l5_z5.mlx_renderer import (
        Z5RaoBallardSubstrateMLX,
    )

    # Wave N+10 Slot 3 stabilizer telemetry per task #1481 + Z7-Mamba-2
    # canonical stabilized pattern (lr=1e-4 succeeded vs lr=3e-4 NaN at ep38).
    # Per CLAUDE.md "Max observability" non-negotiable: log every stabilizer
    # flag at training start so the empirical anchor JSON records the EXACT
    # stabilizer configuration even when the flags are not yet adapter-wired.
    stabilizer_telemetry = {
        "grad_clip_max_norm": (
            float(args.grad_clip_max_norm)
            if args.grad_clip_max_norm is not None
            else None
        ),
        "warmup_epochs": int(args.warmup_epochs),
        "effective_full_lr": float(args.full_lr),
        "stabilizer_status": (
            "ADAPTER_WIRING_DEFERRED_TO_SISTER_SUBAGENT"
            if (
                args.grad_clip_max_norm is not None
                or args.warmup_epochs > 0
            )
            else "BASELINE_NO_STABILIZER"
        ),
        "smallest_perturbation_this_turn": "reduced_full_lr_only",
    }
    print(
        f"[z5_mlx_local:_full_main] stabilizer_telemetry={json.dumps(stabilizer_telemetry, sort_keys=True)}",
        file=sys.stderr,
    )

    cfg = Z5RaoBallardConfig(num_pairs=int(args.num_pairs))
    model = Z5RaoBallardSubstrateMLX(cfg, seed=int(args.seed))
    out_h, out_w = int(cfg.output_height), int(cfg.output_width)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )

    # Canonical Hinton-distilled scorer surrogate wiring per Z6-v2 commit
    # c26647891 + Z7-Mamba-2 commit 2224eff58 + Catalog #164.
    #
    # Per 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27: this is
    # Z5's OWN engineering pass — the 2-level Rao-Ballard hierarchical
    # predictive coding paradigm under Hinton-distilled scorer-bound gradient
    # binds the canonical Hinton-Vinyals-Dean 2014 KL T=2.0 + pose-MSE
    # composition to Z5's specific substrate-distinguishing primitives per
    # Catalog #272.
    #
    # Cross-family hypothesis (operator-routable per the Catalog #344
    # canonical equation ``z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1``):
    # does the EXPLICIT 2-level hierarchy + Atick-Redlich cooperative-receiver
    # gradient binding produce empirically DIFFERENT convergence signature than
    # Z6-v2's single-level FiLM ego-motion conditioning AND Z7-Mamba-2's
    # state-space recurrence under the SAME Hinton-distilled scorer-bound
    # gradient at 600-pair MLX-LOCAL? Canonical equation anchor 0 -> 1.
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
            forward_convention="reconstruct_pair_nchw01",
            distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_dims=DEFAULT_POSE_DIMS,
        )
        scorer_teacher = build_mlx_segnet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",  # CLAUDE.md "MPS auth eval is NOISE" — CPU teacher only.
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
        # Z5 reconstructs per-pair via the canonical
        # ``reconstruct_pair_nchw01`` convention (mirrors Z7-Mamba-2 pattern;
        # the model.reconstruct_pair(idx) returns (rgb_0, rgb_1, residual)
        # NCHW in [0, 1]).
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=float(args.distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        pose_distillation_weight=pose_distillation_weight,
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=DEFAULT_POSE_DIMS,
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        export_archive_fn=export_z5_mlx_archive,
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="time_traveler_l5_z5_mlx_local",
        lane_id=(
            "lane_time_traveler_l5_z5_rao_ballard_mlx_local_scaffold_20260528"
        ),
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "Z5 Rao-Ballard 2-level hierarchical predictive coding MLX-FIRST "
            "score-aware training via canonical mlx_score_aware harness + "
            "Hinton-distilled SegNet + PoseNet teacher per Catalog #164; "
            "2-level Rao-Ballard hierarchy (z_low + z_high + predictor; "
            f"low_latent_dim={int(cfg.low_latent_dim)}, "
            f"high_latent_dim={int(cfg.high_latent_dim)}, "
            f"ego_dim={int(cfg.ego_dim)}) is the substrate-distinguishing "
            "primitive per Catalog #272; FIRST among Z4/Z5/Z6/Z7/Z8 class-"
            "shift queue per T4 SYMPOSIUM Wave N+13 verdict; sister of "
            "Z6-v2 single-level FiLM (commit c26647891) + Z7-Mamba-2 "
            "state-space (commit 2224eff58) within the cooperative-receiver "
            "paradigm class per Catalog #311 + #312 hierarchical predictive "
            "coding canonical quadruple; non-promotable [macOS-MLX research-"
            "signal] per Catalog #192/#317/#341; per-axis + MLX->PyTorch "
            "bridge + paired CUDA/CPU anchor DEFERRED to sister L2 + per-"
            "substrate symposium Catalog #325; canonical equation "
            "`z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1` "
            "anchor 0 -> 1 fires Catalog #371 auto-recalibration trigger "
            "after 2 more empirical anchors land."
        ),
    )
    print(
        f"[z5_mlx_local:_full_main] DONE "
        f"epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-local smoke manifest (config + renderer + 1 forward; no training).

    Validates that ``Z5RaoBallardSubstrateMLX`` constructs end-to-end and
    produces correctly-shaped output. Non-promotable per Catalog
    #192/#317/#341.
    """
    try:  # pragma: no cover — exercised on Apple Silicon with MLX installed.
        import mlx.core as mx
    except Exception as exc:
        print(
            f"ERROR: MLX is not available on this host: {exc!r}. The MLX-local "
            "smoke requires Apple Silicon with the ``mlx`` package installed.",
            file=sys.stderr,
        )
        return 2

    from tac.substrates.time_traveler_l5_z5.architecture import Z5RaoBallardConfig
    from tac.substrates.time_traveler_l5_z5.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        Z5RaoBallardSubstrateMLX,
    )

    cfg = Z5RaoBallardConfig(num_pairs=min(int(args.num_pairs), 8))
    model = Z5RaoBallardSubstrateMLX(cfg, seed=int(args.seed))
    num_params = int(model.num_parameters())
    # Single forward to validate the architecture binds end-to-end.
    import numpy as np

    idx_np = np.array(list(range(min(4, cfg.num_pairs))), dtype=np.int32)
    rgb_0, rgb_1, residual = model.reconstruct_pair(idx_np)
    mx.eval(rgb_0)  # type: ignore[union-attr]
    mx.eval(rgb_1)  # type: ignore[union-attr]
    mx.eval(residual)  # type: ignore[union-attr]
    out_shape_0 = tuple(int(s) for s in rgb_0.shape)
    out_shape_1 = tuple(int(s) for s in rgb_1.shape)
    residual_shape = tuple(int(s) for s in residual.shape)
    expected_shape = (
        min(4, cfg.num_pairs),
        3,
        int(cfg.output_height),
        int(cfg.output_width),
    )
    if out_shape_0 != expected_shape or out_shape_1 != expected_shape:
        print(
            f"ERROR: MLX renderer output shape mismatch — "
            f"got rgb_0={out_shape_0}, rgb_1={out_shape_1}, "
            f"expected {expected_shape}",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(
        args.output_dir
        or ".omx/research/z5_rao_ballard_mlx_local_smoke"
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
            "z5_rao_ballard_mlx_local_smoke_manifest_v1_20260528"
        ),
        "substrate_id": "time_traveler_l5_z5_mlx_local",
        "lane_id": (
            "lane_time_traveler_l5_z5_rao_ballard_mlx_local_scaffold_20260528"
        ),
        "renderer_module": (
            "tac.substrates.time_traveler_l5_z5.mlx_renderer"
        ),
        "renderer_schema_version": SCHEMA_VERSION,
        "renderer_num_parameters": num_params,
        "config": {
            "low_latent_dim": int(cfg.low_latent_dim),
            "high_latent_dim": int(cfg.high_latent_dim),
            "ego_dim": int(cfg.ego_dim),
            "embed_dim": int(cfg.embed_dim),
            "initial_grid_h": int(cfg.initial_grid_h),
            "initial_grid_w": int(cfg.initial_grid_w),
            "decoder_channels": list(cfg.decoder_channels),
            "num_upsample_blocks": int(cfg.num_upsample_blocks),
            "sin_frequency": float(cfg.sin_frequency),
            "film_generator_depth": int(cfg.film_generator_depth),
            "film_hidden_width": int(cfg.film_hidden_width),
            "num_pairs": int(cfg.num_pairs),
            "output_height": int(cfg.output_height),
            "output_width": int(cfg.output_width),
            "predictor_hidden_dim": int(cfg.predictor_hidden_dim),
            "predictor_num_layers": int(cfg.predictor_num_layers),
            "lambda_residual": float(cfg.lambda_residual),
            "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
        },
        "forward_smoke": {
            "input_indices": [int(v) for v in idx_np.tolist()],
            "rgb_0_shape": list(out_shape_0),
            "rgb_1_shape": list(out_shape_1),
            "residual_shape": list(residual_shape),
            "rgb_0_min": float(rgb_0.min()),  # type: ignore[union-attr]
            "rgb_0_max": float(rgb_0.max()),  # type: ignore[union-attr]
            "rgb_0_mean": float(rgb_0.mean()),  # type: ignore[union-attr]
            "rgb_1_min": float(rgb_1.min()),  # type: ignore[union-attr]
            "rgb_1_max": float(rgb_1.max()),  # type: ignore[union-attr]
            "rgb_1_mean": float(rgb_1.mean()),  # type: ignore[union-attr]
            "residual_l2": float(((residual * residual).sum()) ** 0.5),  # type: ignore[union-attr]
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
        "scaffold_state": (
            "L1_full_main_wired_via_canonical_mlx_score_aware_harness_"
            "z5_first_t4_symposium_landed_20260528"
        ),
        "operator_routable_landing_memo": (
            ".omx/research/"
            "z5_rao_ballard_mlx_local_scaffold_landed_20260528.md"
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
        f"[z5_mlx_local smoke] manifest written to: {manifest_path} "
        f"(num_params={num_params}; rgb_0_shape={out_shape_0}; "
        f"{MLX_EVIDENCE_GRADE}) non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Z5 Rao-Ballard 2-level hierarchical predictive coding MLX-first "
            "score-aware trainer (L1 LONG-RUN MLX-LOCAL 2026-05-28)."
        )
    )
    p.add_argument("--smoke", action="store_true", help="Emit smoke manifest only.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Run full MLX score-aware training via the canonical harness.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-pairs",
        type=int,
        default=600,
        help="Trainable pair count (600 for canonical anchor; smoke caps at 8).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="MLX score-aware epochs (--full).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (NOT /tmp per Catalog #208).",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full score-aware training (Catalog #114).",
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
        "--full-lr",
        type=float,
        default=1e-4,
        help=(
            "MLX optimizer learning rate (default 1e-4 per Z7-Mamba-2 commit "
            "2224eff58 stabilized pattern; lr=3e-4 produced NaN at ep38 for "
            "Z7-Mamba-2 sister at 600 pairs)."
        ),
    )
    p.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help=(
            "Weight on the gradient-reachable Hinton-KL T=2.0 SegNet scorer "
            "surrogate term in the --full score-aware loss (0.0 disables). "
            ">0 + NOT --allow-mock-scorer-teacher binds the REAL SegNet + "
            "REAL PoseNet teacher cache via canonical "
            "build_mlx_segnet_pair_teacher/build_mlx_posenet_pair_teacher per "
            "Catalog #164."
        ),
    )
    p.add_argument(
        "--pose-distillation-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the POSE-MSE distillation term per CLAUDE.md 'SegNet "
            "vs PoseNet importance' operating-point-dependent discipline "
            "(pose is DOMINANT at frontier). Default 1.0 wires both scorers."
        ),
    )
    p.add_argument(
        "--allow-mock-scorer-teacher",
        action="store_true",
        help=(
            "EXPLICIT opt-in to the scorer-BLIND deterministic-cosine mock "
            "teacher when --distillation-weight > 0 AND no real scorer_teacher "
            "is wired. Default OFF — the harness fails closed otherwise."
        ),
    )
    p.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=None,
        help=(
            "Adam stabilizer: gradient clipping max-norm. Per Loshchilov+"
            "Hutter 2019 (Adam stability), max_norm=1.0 is canonical for "
            "hierarchical-predictive-coding + Adam stability. STATUS THIS "
            "TURN: flag accepted + recorded; full wiring requires canonical "
            "adapter PR (sister territory). The effective fix this turn is "
            "via reduced --full-lr (sister Z7-Mamba-2 pattern)."
        ),
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help=(
            "Adam stabilizer: linear warmup 0->lr over N epochs. "
            "STATUS THIS TURN: flag accepted + recorded; full wiring requires "
            "canonical adapter PR (sister territory)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    _build_parser().print_help()
    return 1


__all__ = ["_full_main", "_smoke_main", "main"]


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
