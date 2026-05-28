# SPDX-License-Identifier: MIT
"""PACT-NeRV-SELECTOR-V3 MLX-first score-aware trainer — L1 LONG-RUN MLX-LOCAL.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

PACT-NERV-NEXT-VARIANT 2026-05-28: dedicated MLX-LOCAL trainer sister of
``experiments/train_substrate_pact_nerv_selector_v3.py`` (the PyTorch sister).
This trainer is the THIRD PACT-NeRV variant promoted from L0 SCAFFOLD to L1
LONG-RUN MLX-LOCAL via the canonical pattern landed by sister PACT-NeRV-IA3
(commit ``9ecc75a2d``) + sister PACT-NeRV-SELECTOR-V2 (commit ``fee801ac7``).

Variant selection rationale (per parent prompt's individually-fractal criteria)
------------------------------------------------------------------------------

After PACT-NeRV-SELECTOR-V2 (Stage 11) landed L1 LONG-RUN MLX-LOCAL via commit
``fee801ac7``, the next-highest-EV variant per the ULTIMATE design memo
(``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``)
is PACT-NeRV-SELECTOR-V3 (Variant #12 / STAIRCASE Step 12), PRIORITY 1 per
CROSS-CANDIDATE finding #1 empirical headroom + cascade continuation:

- (i) **Most-canonical "next"** per ULTIMATE STAIRCASE: Step 12 (SELECTOR
  -PARADIGM-EXTENSIONS class continuation) — sister next-pick after SELECTOR-V2
  per the SELECTOR-V2 landing memo's TOP-1 recommendation.
- (ii) **Highest predicted-ΔS-per-MLX-hour EV**: SELECTOR-V3 inherits the same
  FEC6 k=16 palette empirical headroom anchor as SELECTOR-V2 (+259 bytes →
  +0.00333 [contest-CPU] empirical anchor); the L1 MLX research-signal probes
  whether the Rice-Golomb coder (Golomb 1966 + Rice 1971; optimal for
  geometric-decay distributions) achieves even tighter code-lengths than
  SELECTOR-V2's arithmetic coder for the FEC6 mode-frequency distribution.
- (iii) **MLX-implementable at L1 ~3-6h**: SELECTOR-V3's base HNeRV decoder
  is structurally identical to SELECTOR-V2 (DepthSep + SIREN + PixelShuffle x7);
  the Rice-Golomb primitive operates at ARCHIVE-ENCODE TIME so the MLX
  renderer is the BASE HNeRV decoder without modulation (~430 LOC sister
  ``mlx_renderer.py`` landed at commit ``$(this commit)``).
- (iv) **DISJOINT from IA3 + SELECTOR-V2** at the primitive surface: Rice-Golomb
  unary+binary coding (optimal for geometric-decay) is a different fractional-
  bit-precision strategy than SELECTOR-V2's arithmetic coding (optimal for
  arbitrary distributions); SELECTOR-V3 is the next SELECTOR-PARADIGM-
  EXTENSIONS family member.

INDIVIDUALLY-FRACTAL per UNIQUE-AND-COMPLETE-PER-METHOD
-------------------------------------------------------

This is PACT-NeRV-SELECTOR-V3's OWN canonical MLX engineering pass per the
11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27. The trainer is
SEPARATE from the PyTorch sister (no shared-helper shortcut; per-method
optimization). The PyTorch ``experiments/train_substrate_pact_nerv_selector_v3.py``
continues to exist with its CUDA-required ``_full_main``; this trainer is
the dedicated MLX-LOCAL engineering pass per the 8th MLX-first standing
directive REINFORCED 2026-05-27 ("always prefer MLX first always").

Canonical-vs-unique decision per layer (Catalog #290)
-----------------------------------------------------

- ADOPT_CANONICAL_BECAUSE_SERVES: training loop / EMA / score-aware loss /
  Provenance / posterior anchor (the canonical
  ``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``
  harness + ``run_long_training``).
- ADOPT_CANONICAL_BECAUSE_SERVES: HNeRV-class base decoder backbone
  (DepthSep + SIREN + PixelShuffle x7) — same as PACT-NeRV-IA3 / SELECTOR-V2
  per the empirically validated PR95/PR101/PR110 medal-class topology.
- FORK_BECAUSE_PRINCIPLED_MISMATCH (this substrate's UNIQUE primitive): the
  per-pair difficulty-conditioned Rice-Golomb coder over k=16 palette per
  Golomb 1966 + Rice 1971 (the substrate-distinguishing primitive operating
  at ARCHIVE-ENCODE TIME — NOT in the MLX forward path; optimal for the
  geometric-decay distribution where mode 0 "none" dominates).

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

- Canonical MLX renderer:
  :mod:`tac.substrates.pact_nerv_selector_v3.mlx_renderer`
- Canonical PyTorch sister architecture:
  :mod:`tac.substrates.pact_nerv_selector_v3.architecture`
- Canonical MLX score-aware harness:
  :mod:`tac.substrates._shared.mlx_score_aware`
- ULTIMATE design memo (Step 12 / Variant #12):
  ``.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md``
- This landing memo:
  ``.omx/research/pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528.md``
- SELECTOR-V2 reference landing (cascade-continuation):
  ``.omx/research/pact_nerv_selector_v2_l1_long_run_mlx_landed_20260528.md``
- IA3 reference landing (canonical L1 promotion pattern):
  ``.omx/research/pact_nerv_long_run_mlx_local_closure_landed_20260528.md``

Usage
-----

Smoke (CPU/MLX, 2 epochs, synthetic-free real video, manifest only)::

    .venv/bin/python experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py \\
        --output-dir experiments/results/pact_nerv_selector_v3_mlx_smoke_<utc> \\
        --smoke

Full LONG run (MLX-LOCAL M5 Max, real video, score-aware via canonical harness)::

    .venv/bin/python experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py \\
        --full --output-dir experiments/results/pact_nerv_selector_v3_mlx_long_<utc> \\
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
        "env": "PACT_NERV_SELECTOR_V3_MLX_OUTPUT_DIR",
        "rationale": (
            "Output dir for MLX-local training artifacts: training_artifact "
            "JSON + EMA checkpoint + observability surface (NOT /tmp per "
            "Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "PACT_NERV_SELECTOR_V3_MLX_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. LONG run smoke -> short "
            "follow-up runs; 100-2000ep is the canonical pre-paid-dispatch "
            "research-signal window per Catalog #325."
        ),
        "default": "100",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "PACT_NERV_SELECTOR_V3_MLX_VIDEO_PATH",
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
    (sister of ``pact_nerv_ia3_mlx_local`` / ``pact_nerv_selector_v2_mlx_local``
    for the IA3 + SELECTOR-V2 sisters + ``dreamer_v3_rssm`` / ``coin_pp`` for
    other MLX-first substrates).

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
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
    )
    from tac.substrates.pact_nerv_selector_v3.archive_candidate import (
        export_pact_nerv_selector_v3_mlx_archive,
    )
    from tac.substrates.pact_nerv_selector_v3.mlx_renderer import (
        PactNervSelectorV3SubstrateMLX,
    )

    cfg = PactNervSelectorV3Config(num_pairs=int(args.num_pairs))
    model = PactNervSelectorV3SubstrateMLX(cfg)
    out_h, out_w = int(cfg.output_height), int(cfg.output_width)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="call_b2chw_255",
        distillation_weight=float(args.distillation_weight),
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        export_archive_fn=export_pact_nerv_selector_v3_mlx_archive,
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="pact_nerv_selector_v3_mlx_local",
        lane_id="lane_pact_nerv_selector_v3_l1_long_run_mlx_local_20260528",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "PACT-NeRV-SELECTOR-V3 MLX-first score-aware LONG-RUN training "
            "via canonical mlx_score_aware harness; real contest video + "
            "reconstruction + optional Hinton-KL T=2.0 scorer surrogate; "
            "per-pair difficulty-conditioned Rice-Golomb coder over k=16 "
            "palette (Golomb 1966 + Rice 1971; optimal for geometric-decay "
            "distributions) is the substrate-distinguishing primitive "
            "operating at ARCHIVE-ENCODE TIME (NOT the MLX forward path; "
            "sister of PACT-NeRV-IA3 + SELECTOR-V2 base HNeRV decoder "
            "backbone); non-promotable [macOS-MLX research-signal] per "
            "Catalog #192/#317/#341; per-axis + MLX->PyTorch bridge + "
            "paired CUDA/CPU anchor DEFERRED to sister L2 + per-substrate "
            "symposium Catalog #325."
        ),
    )
    print(
        f"[pact_nerv_selector_v3_mlx_local:_full_main] DONE "
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

    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
    )
    from tac.substrates.pact_nerv_selector_v3.mlx_renderer import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        PactNervSelectorV3SubstrateMLX,
    )

    cfg = PactNervSelectorV3Config(num_pairs=min(int(args.num_pairs), 8))
    model = PactNervSelectorV3SubstrateMLX(cfg)
    num_params = int(model.num_parameters())
    # Single forward to validate the architecture binds end-to-end.
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
        or ".omx/research/pact_nerv_selector_v3_mlx_local_smoke"
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
        "schema_version": "pact_nerv_selector_v3_mlx_smoke_manifest_v1_20260528",
        "substrate_id": "pact_nerv_selector_v3_mlx_local",
        "lane_id": "lane_pact_nerv_selector_v3_l1_long_run_mlx_local_20260528",
        "renderer_module": "tac.substrates.pact_nerv_selector_v3.mlx_renderer",
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
            "selector_palette_size": int(cfg.selector_palette_size),
            "rice_golomb_k": int(cfg.rice_golomb_k),
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
        f"[pact_nerv_selector_v3_mlx_local smoke] manifest written to: {manifest_path} "
        f"(num_params={num_params}; output_shape={output_shape}) "
        f"{MLX_EVIDENCE_GRADE} non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PACT-NeRV-SELECTOR-V3 MLX-first score-aware trainer "
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
        help="Trainable pair count (32 for LONG; smoke caps at 8).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="MLX score-aware epochs (--full).",
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
            "term in the --full score-aware loss (0.0 disables). >0 REQUIRES a "
            "real scorer_teacher OR --allow-mock-scorer-teacher per Catalog "
            "#164 + the C6 IBPS scorer-blindness lesson."
        ),
    )
    p.add_argument(
        "--allow-mock-scorer-teacher",
        action="store_true",
        help=(
            "EXPLICIT opt-in to the scorer-BLIND deterministic-cosine mock "
            "teacher when --distillation-weight > 0 AND no real scorer_teacher "
            "is wired. Default OFF — the harness fails closed otherwise. Set "
            "ONLY for a $0 no-real-SegNet smoke that explicitly accepts the "
            "result is reconstruction-proxy (NOT scorer-bound)."
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
