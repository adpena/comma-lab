# SPDX-License-Identifier: MIT
"""PACT-NeRV-VQ MLX-first score-aware trainer — L1 LONG-RUN MLX-LOCAL.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

PACT-NERV-VQ-LONG-RUN-MLX-LOCAL 2026-05-28: dedicated MLX-LOCAL trainer sister
of ``experiments/train_substrate_pact_nerv_vq.py`` (the PyTorch sister). Per
the operator NON-NEGOTIABLE TOP-1 operator-routable post-SELECTOR-V4 verdict
(commit ``f013736de``): the SELECTOR-PARADIGM cascade has empirically
saturated at the 32-pair base-decoder floor; PACT-NeRV-VQ provides the
ORTHOGONAL paradigm (DISCRETE TOKENS via van den Oord VQ-VAE codebook +
per-pair index) per ULTIMATE STAIRCASE Step 15 PRIORITY 1.

CONTEXT — SELECTOR-PARADIGM saturation (PACT-NeRV cascade)
==========================================================

- IA3 L1 commit ``9ecc75a2d``:       140.0x / 0.00240 final
- SELECTOR-V2 L1 commit ``fee801ac7``: 196.5x / 0.00172 final
- SELECTOR-V3 L1 commit ``2f69d0ea6``: 231.1x / 0.00146 final
- SELECTOR-V4 L1 commit ``f013736de``: 201.3x / 0.001677 final

The 0.0014-0.0017 band is the stochastic-seed + AdamW-noise floor at 32-pair
scale. PACT-NeRV-VQ is the orthogonal-architectural pivot per portfolio
diversification discipline (Aaron van den Oord inner council seat aligned).

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD
=======================================================

This trainer is PACT-NeRV-VQ's OWN canonical MLX engineering pass per the
11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27. The trainer is
SEPARATE from the PyTorch sister (no shared-helper shortcut; per-method
optimization). The PyTorch ``experiments/train_substrate_pact_nerv_vq.py``
(if/when it lands as the sister L1 PyTorch trainer) continues to exist with
its CUDA-required ``_full_main``; this trainer is the dedicated MLX-LOCAL
engineering pass per the 8th MLX-first standing directive REINFORCED 2026-05-28
("you can fire everything and anything on MLX").

Canonical-vs-unique decision per layer (Catalog #290)
-----------------------------------------------------

- ADOPT_CANONICAL_BECAUSE_SERVES: training loop / EMA / score-aware loss /
  Provenance / posterior anchor (the canonical
  ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
  harness + ``run_long_training``).
- FORK_BECAUSE_PRINCIPLED_MISMATCH (this substrate's UNIQUE primitive): the
  VQ-VAE codebook + per-pair index quantizer per van den Oord 1711.00937
  (``mlx_renderer.PactNervVqSubstrateMLX``).

Dispatch gating (Catalog #325)
------------------------------

MLX-LOCAL ONLY ($0 M5 Max); the harness fails closed on a non-MLX host (NO
CPU/CUDA paid-dispatch leak per Catalog #1 + #317). The matching PyTorch-
sister recipe stays ``dispatch_enabled: false`` + ``research_only: true``;
output from this MLX-LOCAL trainer is non-promotable
``[macOS-MLX research-signal]`` per Catalog #192/#341. Per-substrate
symposium per Catalog #325 + MLX→PyTorch bridge + paired CUDA/CPU anchor
remain DEFERRED to the PyTorch sister L2 / paid-dispatch path; this
trainer is the FREE pre-paid-dispatch research signal generator.

Cross-references
----------------

- Canonical MLX renderer: :mod:`tac.substrates.pact_nerv_vq.mlx_renderer`
- Canonical PyTorch sister architecture:
  :mod:`tac.substrates.pact_nerv_vq.architecture`
- Canonical MLX score-aware harness:
  :mod:`tac.substrates._shared.mlx_score_aware`
- Sister IA3 MLX trainer reference:
  ``experiments/train_substrate_pact_nerv_ia3_mlx_local.py``
- ULTIMATE design memo:
  ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
- L0 SCAFFOLD design memo:
  ``.omx/research/pact_nerv_vq_l0_scaffold_design_20260520T211500Z.md``
- This landing memo:
  ``.omx/research/pact_nerv_vq_l1_long_run_mlx_landed_20260528.md``

Usage
-----

Smoke (CPU/MLX, 2 epochs, synthetic-free real video, manifest only)::

    .venv/bin/python experiments/train_substrate_pact_nerv_vq_mlx_local.py \\
        --output-dir experiments/results/pact_nerv_vq_mlx_smoke_<utc> \\
        --smoke

Full LONG run (MLX-LOCAL M5 Max, real video, score-aware via canonical harness)::

    .venv/bin/python experiments/train_substrate_pact_nerv_vq_mlx_local.py \\
        --full --output-dir experiments/results/pact_nerv_vq_mlx_long_<utc> \\
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
        "env": "PACT_NERV_VQ_MLX_OUTPUT_DIR",
        "rationale": (
            "Output dir for MLX-local training artifacts: training_artifact "
            "JSON + EMA checkpoint + observability surface (NOT /tmp per "
            "Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "PACT_NERV_VQ_MLX_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. The canonical LONG run is "
            "2000ep / 32 pairs matching the PACT-NeRV cascade reference "
            "(IA3/V2/V3/V4 all 2000ep) per Catalog #325."
        ),
        "default": "2000",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "PACT_NERV_VQ_MLX_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full score-aware training (Catalog "
            "#114; real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _full_main(args: argparse.Namespace) -> int:
    """Run the canonical MLX-first score-aware ``_full_main`` body.

    This routes through the canonical substrate-AGNOSTIC harness
    ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
    (sister of the IA3 / V2 / V3 / V4 MLX-LOCAL trainers).

    Per CLAUDE.md "MLX portable-local-substrate authority": the harness
    auto-stamps the canonical non-promotable markers
    (``score_claim=False``, ``promotion_eligible=False``,
    ``ready_for_exact_eval_dispatch=False``) on the ``TrainingArtifact``.

    Distillation defaults to ``0.0`` (pure reconstruction); the operator
    opts INTO the gradient-reachable Hinton-KL T=2.0 scorer surrogate via
    ``--distillation-weight``. Per Catalog #164 + the C6 IBPS / DreamerV3
    scorer-blindness lesson: any non-zero distillation weight MUST bind a
    real SegNet teacher (the harness fails closed otherwise unless
    ``--allow-mock-scorer-teacher`` is explicitly passed for a $0
    no-real-SegNet smoke).
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
    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig
    from tac.substrates.pact_nerv_vq.mlx_renderer import (
        PactNervVqSubstrateMLX,
    )

    cfg = PactNervVqConfig(num_pairs=int(args.num_pairs))
    model = PactNervVqSubstrateMLX(cfg)
    out_h, out_w = int(cfg.output_height), int(cfg.output_width)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )

    # ----------------------------------------------------------------
    # PACT-NeRV-VQ INDIVIDUALLY-FRACTAL export_state_dict_fn (CRITICAL).
    # ----------------------------------------------------------------
    # The canonical MLX harness fallback uses ``model.parameters()`` which
    # EXCLUDES the VQ codebook + EMA buffers (registered as bare mx.array
    # attributes per van den Oord §3.2 — NOT MLX nn.Parameter). The
    # MLX→PyTorch bridge tool requires these buffers to reconstruct the
    # PyTorch quantizer state at archive-pack time. Wire a substrate-specific
    # export that emits the FULL state in MLX-native HWIO Conv2d layout
    # (matches what the bridge tool's ``unpack_state_dict_numpy`` expects).
    # Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" + Catalog #290: this is
    # the substrate-specific FORK_BECAUSE_PRINCIPLED_MISMATCH per layer.
    def _export_vq_state_dict_with_buffers(m: Any, path: Path) -> None:
        import numpy as np

        from tac.substrates._shared.numpy_portable_inflate import (
            pack_state_dict_numpy,
        )

        # Build canonical MLX-HWIO state including VQ buffers; the canonical
        # ``model.export_state_dict()`` returns PyTorch-OIHW (transposed) so
        # we read the raw MLX-side attributes directly for the bridge's
        # HWIO→OIHW transpose to work.
        flat: dict[str, np.ndarray] = {}

        # Per-pair learnable latent (no Conv layout; 2-D).
        flat["latents"] = np.asarray(m.latents, dtype=np.float32).copy()

        # VQ buffers (registered as private mx.array attrs; NOT params).
        flat["quantizer.codebook"] = np.asarray(
            m.quantizer.codebook, dtype=np.float32
        ).copy()
        flat["quantizer.ema_cluster_size"] = np.asarray(
            m.quantizer.ema_cluster_size, dtype=np.float32
        ).copy()
        flat["quantizer.ema_w"] = np.asarray(
            m.quantizer.ema_w, dtype=np.float32
        ).copy()

        # Linear weights (no Conv layout).
        flat["latent_embed.weight"] = np.asarray(
            m.latent_embed.weight, dtype=np.float32
        ).copy()
        flat["latent_embed.bias"] = np.asarray(
            m.latent_embed.bias, dtype=np.float32
        ).copy()

        # Conv2d weights in MLX HWIO layout (out, kH, kW, in).
        for i, block in enumerate(m.blocks):
            d = block.dsc.depthwise
            p = block.dsc.pointwise
            flat[f"blocks.{i}.dsc.depthwise.weight"] = np.asarray(
                d.weight, dtype=np.float32
            ).copy()
            flat[f"blocks.{i}.dsc.depthwise.bias"] = np.asarray(
                d.bias, dtype=np.float32
            ).copy()
            flat[f"blocks.{i}.dsc.pointwise.weight"] = np.asarray(
                p.weight, dtype=np.float32
            ).copy()
            flat[f"blocks.{i}.dsc.pointwise.bias"] = np.asarray(
                p.bias, dtype=np.float32
            ).copy()

        for head_name in ("head_rgb_0", "head_rgb_1"):
            head = getattr(m, head_name)
            flat[f"{head_name}.weight"] = np.asarray(
                head.weight, dtype=np.float32
            ).copy()
            flat[f"{head_name}.bias"] = np.asarray(
                head.bias, dtype=np.float32
            ).copy()

        path.parent.mkdir(parents=True, exist_ok=True)
        blob_path = path.with_suffix(path.suffix + ".npsd")
        blob = pack_state_dict_numpy(flat, dtype="fp32")
        blob_path.write_bytes(blob)

    # Canonical Hinton-distilled scorer surrogate wiring per IA3 sister commit
    # b551bfd34 + SELECTOR-V3 sister commit ab650cc78 + canonical equation #1
    # hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1 sister cascade
    # batch fires Catalog #371 auto-recalibration trigger. When
    # --distillation-weight > 0 AND NOT --allow-mock-scorer-teacher we bind the
    # REAL SegNet + REAL PoseNet teacher caches + learnable student heads per
    # the canonical Hinton-Vinyals-Dean 2014 KL T=2.0 + pose-MSE composition.
    # Per CLAUDE.md "SegNet vs PoseNet importance" + Catalog #164 the harness
    # bundle.__post_init__ fail-closes on missing pose teacher when
    # distillation_weight > 0 (C6 IBPS / DreamerV3 scorer-blindness lesson).
    # Output (384, 512) matches the canonical SegNet/PoseNet eval resolution
    # exactly (zero adapter).
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
        export_state_dict_fn=_export_vq_state_dict_with_buffers,
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="pact_nerv_vq_mlx_local",
        lane_id="lane_pact_nerv_vq_l1_long_run_mlx_local_20260528",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "PACT-NeRV-VQ MLX-first score-aware LONG-RUN training via canonical "
            "mlx_score_aware harness; real contest video + reconstruction + "
            "optional Hinton-KL T=2.0 scorer surrogate; VQ-VAE codebook + "
            "per-pair discrete index (van den Oord 1711.00937 §3.1-3.2) is the "
            "substrate-distinguishing ORTHOGONAL primitive vs the SELECTOR-"
            "PARADIGM cascade (IA3/V2/V3/V4) saturated at 32-pair base-decoder "
            "floor; non-promotable [macOS-MLX research-signal] per Catalog "
            "#192/#317/#341; per-axis + MLX->PyTorch bridge + paired CUDA/CPU "
            "anchor DEFERRED to sister L2 + per-substrate symposium Catalog #325."
        ),
    )
    print(
        f"[pact_nerv_vq_mlx_local:_full_main] DONE "
        f"epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-local smoke manifest (config + renderer + 1 forward; no training).

    Emits non-promotable research-signal markers per Catalog #341 + Catalog
    #317. Output destination is operator-configurable. Validates the MLX
    renderer imports, the configured architecture instantiates, and a single
    forward pass produces the expected (B, 2, 3, H, W) output shape — the
    smoke gate per CLAUDE.md "MVP-first phasing" non-negotiable.
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

    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig
    from tac.substrates.pact_nerv_vq.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        PactNervVqSubstrateMLX,
    )

    cfg = PactNervVqConfig(num_pairs=min(int(args.num_pairs), 8))
    model = PactNervVqSubstrateMLX(cfg)
    num_params = int(model.num_parameters())
    idx = mx.array(list(range(min(4, cfg.num_pairs))), dtype=mx.int32)
    output = model(idx)
    mx.eval(output)
    output_shape = tuple(int(s) for s in output.shape)
    expected_shape = (
        min(4, cfg.num_pairs),
        2,
        3,
        int(cfg.output_height),
        int(cfg.output_width),
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
        or ".omx/research/pact_nerv_vq_mlx_local_smoke"
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
        "schema_version": "pact_nerv_vq_mlx_smoke_manifest_v1_20260528",
        "substrate_id": "pact_nerv_vq_mlx_local",
        "lane_id": "lane_pact_nerv_vq_l1_long_run_mlx_local_20260528",
        "renderer_module": "tac.substrates.pact_nerv_vq.mlx_renderer",
        "renderer_schema_version": SCHEMA_VERSION,
        "renderer_num_parameters": num_params,
        "config": {
            "latent_dim": int(cfg.latent_dim),
            "embed_dim": int(cfg.embed_dim),
            "initial_grid_h": int(cfg.initial_grid_h),
            "initial_grid_w": int(cfg.initial_grid_w),
            "decoder_channels": list(cfg.decoder_channels),
            "sin_frequency": float(cfg.sin_frequency),
            "num_upsample_blocks": int(cfg.num_upsample_blocks),
            "codebook_size": int(cfg.codebook_size),
            "codebook_decay": float(cfg.codebook_decay),
            "commitment_weight": float(cfg.commitment_weight),
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
        "vq_observability": {
            "last_commitment_loss": float(model.last_commitment_loss),
            "last_indices": [int(v) for v in model.last_indices.tolist()],
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
                "renderer construction + single forward pass + VQ observability "
                "only. Non-promotable by construction per Catalog #192/#317/#341."
            ),
        },
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[pact_nerv_vq_mlx_local smoke] manifest written to: {manifest_path} "
        f"(num_params={num_params}; output_shape={output_shape}; "
        f"codebook_size={cfg.codebook_size}) "
        f"{MLX_EVIDENCE_GRADE} non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PACT-NeRV-VQ MLX-first score-aware trainer "
            "(L1 LONG-RUN MLX-LOCAL 2026-05-28)."
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
        default=32,
        help="Trainable pair count (32 for LONG matching PACT-NeRV cascade; smoke caps at 8).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="MLX score-aware epochs (--full); canonical PACT-NeRV cascade LONG=2000.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Output dir (NOT /tmp)."
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full score-aware training (Catalog #114).",
    )
    p.add_argument("--full-lr", type=float, default=1e-3)
    p.add_argument(
        "--distillation-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the gradient-reachable Hinton-KL T=2.0 scorer surrogate "
            "term in the --full score-aware loss (0.0 disables). >0 + NOT "
            "--allow-mock-scorer-teacher binds the REAL SegNet + REAL PoseNet "
            "teacher cache via canonical "
            "build_mlx_segnet_pair_teacher/build_mlx_posenet_pair_teacher per "
            "the IA3 sister commit b551bfd34 + V3 sister commit ab650cc78 + "
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
        "--allow-mock-scorer-teacher",
        action="store_true",
        help=(
            "EXPLICIT opt-in to the scorer-BLIND deterministic-cosine mock "
            "teacher when --distillation-weight > 0 AND no real scorer_teacher "
            "is wired. Default OFF — the harness fails closed otherwise."
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
