#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PACT-NeRV-IA3 + Hinton-distilled scorer surrogate integration smoke.

INDIVIDUALLY-FRACTAL integration smoke per the 11th UNIQUE-AND-COMPLETE-PER-METHOD
standing directive: PACT-NeRV-IA3 is the smallest sister (56198 params) in the
PACT-NeRV cascade per `.omx/research/pact_nerv_ultimate_research_and_design_*`;
this smoke wires the canonical Hinton-distilled scorer surrogate
(`tac.substrates.hinton_distilled_scorer_surrogate`) onto the canonical
MLX-first score-aware harness
(`tac.substrates._shared.mlx_score_aware`) with the REAL SegNet teacher cache
+ REAL PoseNet teacher cache + learnable student heads BOUND on the PACT-NeRV-IA3
MLX renderer.

PURPOSE: produce empirical evidence that the canonical scorer-surrogate
architecture binds end-to-end on a sister PACT-NeRV cascade trainer at $0
MLX-LOCAL cost, demonstrating the unblock pattern for the rest of the cascade
(IA3 multi / V2 / V3 / V4 / VQ) BEFORE any sister wave commits to a sister
trainer-side wire-in.

# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341:
every artifact is `[macOS-MLX research-signal]` with `score_claim=False`,
`promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`. Reactivation
to contest-axis claim requires paired Linux x86_64 + NVIDIA on contest-compliant
hardware per Catalog #246.

Per Catalog #314/#340 sister-checkpoint guard: this smoke only ADDS a NEW tool
under `tools/`; it does NOT mutate the PACT-NeRV-IA3 substrate package, the
canonical Hinton surrogate substrate package, or the canonical MLX harness.

Canonical-vs-unique decision per layer (Catalog #290):

- ADOPT_CANONICAL_BECAUSE_SERVES: the entire integration stack — MLX renderer
  (`tac.substrates.pact_nerv_ia3.mlx_renderer.PactNervIa3SubstrateMLX`), real
  SegNet teacher cache (`build_mlx_segnet_pair_teacher`), real PoseNet teacher
  cache (`build_mlx_posenet_pair_teacher`), learnable SegNet 1x1-conv student
  head (`build_learnable_student_head`), learnable PoseNet pose student head
  (`build_learnable_pose_student_head`), bundle (`RendererBundle`), harness
  (`run_mlx_score_aware_full_main`), target decode (`decode_mlx_targets`).
- FORK_NONE: nothing is forked. This smoke is the canonical integration
  exemplar.

[verified-against: tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss
canonical KL T=2.0 + real SegNet/PoseNet teacher caches + learnable heads]
[verified-against: tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main
canonical substrate-AGNOSTIC harness]
[verified-against: tac.substrates.pact_nerv_ia3.mlx_renderer.PactNervIa3SubstrateMLX
canonical IA3 γ-only ego-pose modulated renderer]
[predicted: predicted-finite Hinton-KL T=2.0 + per-axis pose-MSE composition
converges below pure-pixel-MSE PACT-NeRV-IA3 baseline; predicted_axis_decomposition
is non-promotable observability-only per Catalog #341]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Catalog #287 placeholder-rationale rejection: this smoke runs ONLY on Apple
# Silicon with MLX. Catalog #1 + #317 fail-closed on a non-MLX host.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PACT-NeRV-IA3 + Hinton-distilled scorer surrogate (real SegNet "
            "+ real PoseNet teacher cache) MLX-local integration smoke. "
            "Non-promotable [macOS-MLX research-signal] per Catalog #192."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Canonical output dir for the smoke artifact (NOT /tmp per "
            "CLAUDE.md FORBIDDEN_PATTERN /tmp paths). The harness writes the "
            "TrainingArtifact + canonical posterior anchor here."
        ),
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help=(
            "Real contest video per Catalog #114 (NEVER make_synthetic_*). "
            "Targets are decoded at the PACT-NeRV-IA3 native (384, 512) "
            "resolution which is also the canonical SegNet/PoseNet eval size."
        ),
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=Path("upstream"),
        help="Upstream repo path containing the SegNet + PoseNet safetensors.",
    )
    p.add_argument(
        "--num-pairs",
        type=int,
        default=8,
        help="Trainable pair count (8 for the smoke; 32+ for cascade unblock).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Smoke epoch budget (5 for the integration smoke; longer for the cascade unblock).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed per Catalog #305 observability diff-able.",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="MLX optimizer LR (canonical 1e-3 per the harness).",
    )
    p.add_argument(
        "--distillation-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the Hinton-KL T=2.0 SegNet distillation term. "
            "1.0 = recon + KL roughly balanced (canonical default per the "
            "harness)."
        ),
    )
    p.add_argument(
        "--pose-distillation-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the POSE-MSE distillation term per CLAUDE.md "
            "'SegNet vs PoseNet importance' operating-point-dependent "
            "discipline (pose is DOMINANT at frontier). 1.0 wires both "
            "scorers (PoseNet REQUIRED unless allow_segnet_only_research=True)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Catalog #208 / CLAUDE.md /tmp path forbidden: refuse early.
    out = args.output_dir.resolve()
    if str(out).startswith(("/tmp/", "/private/tmp/")):
        print(
            f"ERROR: --output-dir {out} under /tmp per CLAUDE.md FORBIDDEN_PATTERN "
            "'Forbidden /tmp paths in any persisted artifact'",
            file=sys.stderr,
        )
        return 2
    out.mkdir(parents=True, exist_ok=True)

    # Import the canonical primitives. All discovered + verified above.
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        is_mlx_available,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        DEFAULT_POSE_DIMS,
        DEFAULT_SEGNET_CLASSES,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )
    from tac.substrates.pact_nerv_ia3.architecture import PactNervIa3Config
    from tac.substrates.pact_nerv_ia3.mlx_renderer import PactNervIa3SubstrateMLX

    if not is_mlx_available():
        print(
            "ERROR: MLX is not available on this host. The integration smoke "
            "requires Apple Silicon with the `mlx` package installed per "
            "Catalog #1 + #317 fail-closed.",
            file=sys.stderr,
        )
        return 2

    t_start = time.time()

    # ---- 1. Build the PACT-NeRV-IA3 MLX renderer (56198 params at the
    # canonical num_pairs=32 default; 56198 at any num_pairs since the
    # renderer is per-pair embedding + shared decoder).
    cfg = PactNervIa3Config(num_pairs=int(args.num_pairs))
    model = PactNervIa3SubstrateMLX(cfg)
    out_h, out_w = int(cfg.output_height), int(cfg.output_width)
    if (out_h, out_w) != (384, 512):
        print(
            f"ERROR: PACT-NeRV-IA3 renderer output {(out_h, out_w)} != (384, 512); "
            "real SegNet/PoseNet teacher caches require contest size.",
            file=sys.stderr,
        )
        return 2

    # ---- 2. Decode real contest video targets at SegNet/PoseNet size.
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )

    # ---- 3. Build the bundle WITHOUT scorer_teacher first (so the
    # build_*_teacher helpers can read .target_rgb_* off it).
    bundle_no_teacher = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="call_b2chw_255",
        distillation_weight=0.0,  # no distill yet; just for teacher-cache build.
        pose_distillation_weight=0.0,
        pose_dims=DEFAULT_POSE_DIMS,
    )

    t_segnet = time.time()
    segnet_teacher = build_mlx_segnet_pair_teacher(
        bundle_no_teacher,
        upstream_dir=str(args.upstream_dir),
        device="cpu",  # per CLAUDE.md "MPS auth eval is NOISE" — CPU only for teacher.
    )
    segnet_seconds = time.time() - t_segnet

    t_posenet = time.time()
    posenet_teacher = build_mlx_posenet_pair_teacher(
        bundle_no_teacher,
        upstream_dir=str(args.upstream_dir),
        device="cpu",
    )
    posenet_seconds = time.time() - t_posenet

    # ---- 4. Build the learnable student heads.
    student_head = build_learnable_student_head(
        num_classes=DEFAULT_SEGNET_CLASSES,
        in_channels=3,
        seed=int(args.seed),
    )
    pose_student_head = build_learnable_pose_student_head(
        pose_dims=DEFAULT_POSE_DIMS,
        seed=int(args.seed),
    )

    # ---- 5. Rebuild the bundle WITH scorer + pose teachers + student heads.
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="call_b2chw_255",
        distillation_weight=float(args.distillation_weight),
        scorer_teacher=segnet_teacher,
        learnable_student_head=student_head,
        pose_distillation_weight=float(args.pose_distillation_weight),
        pose_scorer_teacher=posenet_teacher,
        learnable_pose_student_head=pose_student_head,
        pose_dims=DEFAULT_POSE_DIMS,
    )

    # ---- 6. Route through the canonical substrate-AGNOSTIC harness. The
    # harness wraps the bundle in the adapter, builds a canonical
    # LongTrainingConfig, calls run_long_training (EMA + telemetry + posterior
    # anchor). MLX-LOCAL ONLY; the device gate fails closed on a non-MLX host.
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="pact_nerv_ia3_hinton_distill_real_scorer_teacher_integration_smoke",
        lane_id="lane_pact_nerv_ia3_hinton_distill_real_scorer_teacher_integration_smoke_20260528",
        output_dir=out,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 4),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        notes=(
            "PACT-NeRV-IA3 + canonical Hinton-distilled scorer surrogate "
            "(real SegNet + real PoseNet teacher cache + learnable student "
            "heads) integration smoke via canonical mlx_score_aware harness. "
            "Demonstrates end-to-end binding of the Hinton-distilled scorer "
            "surrogate (KL T=2.0 + pose-MSE) onto the PACT-NeRV cascade's "
            "smallest sister at $0 MLX-LOCAL cost per the 8th MLX-first + "
            "11th INDIVIDUALLY-FRACTAL standing directives. Non-promotable "
            "[macOS-MLX research-signal] per Catalog #192/#317/#341; paired "
            "Linux x86_64 + NVIDIA + per-substrate symposium per Catalog "
            "#325 + Catalog #246 REQUIRED for any contest-axis claim."
        ),
    )

    # ---- 7. Write the integration manifest summarizing the smoke.
    posenet_safetensors = Path(args.upstream_dir) / "models" / "posenet.safetensors"
    segnet_safetensors = Path(args.upstream_dir) / "models" / "segnet.safetensors"
    manifest = {
        "schema_version": (
            "pact_nerv_ia3_hinton_distill_integration_smoke_manifest_v1_20260528"
        ),
        "substrate_id": "pact_nerv_ia3_hinton_distill_real_scorer_teacher_integration_smoke",
        "lane_id": "lane_pact_nerv_ia3_hinton_distill_real_scorer_teacher_integration_smoke_20260528",
        "task_id": "1444",
        "renderer": {
            "module": "tac.substrates.pact_nerv_ia3.mlx_renderer",
            "class": "PactNervIa3SubstrateMLX",
            "num_pairs": int(args.num_pairs),
            "num_parameters": int(model.num_parameters()),
            "output_height": out_h,
            "output_width": out_w,
            "forward_convention": "call_b2chw_255",
        },
        "scorer_teacher": {
            "kind": "real_segnet_teacher_cache",
            "module": "tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss",
            "class": "RealSegNetTeacherLogitsCache",
            "build_seconds": round(segnet_seconds, 3),
            "num_classes": int(segnet_teacher.num_classes),
            "frame_count": int(segnet_teacher.frame_count),
            "segnet_safetensors_sha256": (
                hashlib.sha256(segnet_safetensors.read_bytes()).hexdigest()
                if segnet_safetensors.is_file()
                else None
            ),
        },
        "pose_scorer_teacher": {
            "kind": "real_posenet_teacher_cache",
            "module": "tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss",
            "class": "RealPoseNetTeacherCache",
            "build_seconds": round(posenet_seconds, 3),
            "num_pairs": int(posenet_teacher.num_pairs),
            "pose_dims": int(posenet_teacher.pose_dims),
            "upstream_posenet_safetensors_sha256": (
                hashlib.sha256(posenet_safetensors.read_bytes()).hexdigest()
                if posenet_safetensors.is_file()
                else None
            ),
        },
        "student_heads": {
            "segnet_1x1_conv": {
                "num_classes": DEFAULT_SEGNET_CLASSES,
                "in_channels": 3,
                "seed": int(args.seed),
            },
            "pose_head": {
                "pose_dims": DEFAULT_POSE_DIMS,
                "seed": int(args.seed),
            },
        },
        "loss": {
            "kind": "canonical_hinton_distilled_kl_t2_plus_pose_mse",
            "module": "tac.substrates._shared.mlx_score_aware.loss",
            "fn": "score_aware_loss",
            "distillation_temperature": 2.0,
            "distillation_weight": float(args.distillation_weight),
            "pose_distillation_weight": float(args.pose_distillation_weight),
        },
        "training": {
            "harness": "tac.substrates._shared.mlx_score_aware.harness",
            "fn": "run_mlx_score_aware_full_main",
            "epochs": int(artifact.total_epochs_completed),
            "batch_pair_indices_per_step": min(int(args.num_pairs), 4),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "total_wall_clock_seconds": round(
                float(artifact.total_wall_clock_seconds), 3
            ),
            "promotable": bool(artifact.promotable),
            "evidence_grade": getattr(artifact, "evidence_grade", "[macOS-MLX research-signal]"),
        },
        "axis_discipline": {
            "axis_tag": "[macOS-MLX research-signal]",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                "MLX-local research signal; canonical paired Linux x86_64 + "
                "NVIDIA paired auth_eval + per-substrate symposium per Catalog "
                "#325 REQUIRED for any contest-axis claim. Per Catalog "
                "#192/#317/#341 + CLAUDE.md 'MLX portable-local-substrate "
                "authority' non-promotable by construction."
            ),
        },
        "canonical_provenance": {
            "kind": "predicted_from_model",
            "evidence_grade": "predicted",
            "axis_tag": "[macOS-MLX research-signal]",
            "score_claim_valid": False,
            "promotable": False,
            "canonical_helper_invocation": (
                "tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main"
                " + tac.substrates.hinton_distilled_scorer_surrogate.build_real_segnet_teacher_cache"
                " + tac.substrates._shared.mlx_score_aware.build_mlx_posenet_pair_teacher"
            ),
            "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rationale": (
                "Integration smoke demonstrates canonical Hinton-distilled "
                "scorer surrogate (real SegNet + real PoseNet teacher cache "
                "+ learnable student heads) binds end-to-end onto "
                "PACT-NeRV-IA3 MLX renderer via canonical mlx_score_aware "
                "harness. Non-promotable by construction."
            ),
        },
        "total_wall_clock_seconds_including_teacher_build": round(
            time.time() - t_start, 3
        ),
    }

    manifest_path = out / "integration_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[pact_nerv_ia3_hinton_distill_real_scorer_teacher_integration_smoke] "
        f"DONE\n"
        f"  num_params      = {int(model.num_parameters())}\n"
        f"  num_pairs       = {int(args.num_pairs)}\n"
        f"  epochs          = {int(artifact.total_epochs_completed)}\n"
        f"  SegNet teacher  = {segnet_seconds:.2f}s ({int(segnet_teacher.frame_count)} pairs cached)\n"
        f"  PoseNet teacher = {posenet_seconds:.2f}s ({int(posenet_teacher.num_pairs)} pairs cached)\n"
        f"  wall clock      = {float(artifact.total_wall_clock_seconds):.2f}s (training)\n"
        f"  total wall      = {time.time() - t_start:.2f}s (incl teacher build)\n"
        f"  manifest        = {manifest_path}\n"
        f"  evidence_grade  = [macOS-MLX research-signal] non-promotable per Catalog #192/#317/#341"
    )
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
