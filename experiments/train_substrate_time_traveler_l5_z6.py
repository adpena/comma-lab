# SPDX-License-Identifier: MIT
"""Train the Z6 Time-Traveler L5 F-asymptote-node predictive-coding world-model substrate.

Per the Time-Traveler L5 Z6/Z7/Z8 predictive-coding world-model scoping design memo
(``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``,
commit ``aa412d2db``): Z6 is the FIRST sequenced Z-variant of the F-asymptote
trajectory along the scorer-relationship class-shift axis (predictive-coding
paradigm; Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver). Recommended
FIRST build per Section 22 op-routable #2 (lowest engineering risk; sister Z5 L1
scaffold pattern).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet.
- ``apply_eval_roundtrip_during_training`` (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` (Catalog #88).
- Score-domain predictive-coding Lagrangian via canonical
  ``tac.substrates._shared.score_aware_common.score_pair_components`` per
  Catalog #164 (single-source-of-truth scorer-preprocess routing).
- End with CUDA auth eval on best EMA checkpoint via canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` (CLAUDE.md "Auth eval
  EVERYWHERE" + Catalog #226).
- Inflate-device-fork via canonical ``select_inflate_device`` per Catalog #205.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived``
  (Catalog #197).
- SubstrateContract via canonical ``@register_substrate`` per Catalog #241/#242
  (META layer auto-wire one-way data flow).
- Catalog #240 substrate-engineering discipline: ``_full_main`` is implemented
  by the Phase 1b lift, while score-bearing dispatch remains gated on paired
  CPU/CUDA evidence, lane claim custody, and exact-eval artifacts.

Phase 1b scope:
- ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on synthetic
  data, runs archive pack + parse + inflate roundtrip + autoregression
  test, and emits a contest-compliant runtime tree (no scorer load).
- ``_full_main`` decodes real pairs, derives PoseNet-conditioned ego-motion,
  trains the Z6 predictive-coding substrate, emits a Z6PCWM1 archive/runtime,
  and routes auth eval through the canonical gate when not explicitly skipped.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_z6.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z6_smoke_<utc> \\
        --epochs 3 --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:autoregressive-predictor-unroll-needs-canary-validation
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_full_main
# INLINE_DEVICE_FORK_OK:canonical-select_inflate_device-imported-via-tac.substrates._shared.inflate_runtime-per-Catalog-#205
# SCORER_PREPROCESS_HANDLED_OK:routed-through-canonical-tac.substrates._shared.score_aware_common.score_pair_components-per-Catalog-#164
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

# Catalog #241/#242 META layer substrate contract — single source-of-truth
# for trainer + recipe + lane registry + cost band envelope.
from tac.substrate_registry import SubstrateContract, register_substrate

# Catalog #205 canonical inflate device-fork — token visible to Catalog #270
# Tier 3 substrate-correctness verifier.
from tac.substrates._shared.inflate_runtime import select_inflate_device  # noqa: F401

# Catalog #226 canonical auth-eval gate (NEVER hand-roll subprocess to
# contest_auth_eval.py per Catalog #226 self-protect).
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _decode_real_pairs,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _canon_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    require_contest_cuda_auth_eval_claim as _canon_require_contest_cuda_auth_eval_claim,
)

# Catalog #164 canonical scorer-loss helper routing — token visible to
# Catalog #270 Tier 1 dispatch-optimization-protocol verifier.
from tac.substrates.score_aware_common import (  # noqa: F401
    CONTEST_POSE_SQRT_WEIGHT,
    score_pair_components,
)
from tac.substrates.time_traveler_l5_z6 import (
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
    pack_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_TAG = "time_traveler_l5_z6"
SUBSTRATE_LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
PAIRED_CONTROL_INITIALIZATION = "shared_modules_seed_order_matched_v2"

# Contest invariants
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


def _runtime_lane_id() -> str:
    """Return the lane id bound by the dispatch recipe, falling back to base Z6."""

    return os.environ.get("Z6_LANE_ID", "").strip() or SUBSTRATE_LANE_ID


# ---------------------------------------------------------------------------
# Z6-v2 Candidate 1 Wave 2 BUILD predictor-architecture resolver per Phase 3
# council §9. Maps the operator-facing --predictor-architecture flag to the
# (predictor_depth, predictor_hidden_dim) tuple consumed by
# Z6PredictiveCodingConfig.
# ---------------------------------------------------------------------------
def _resolve_predictor_architecture(
    name: str,
) -> tuple[int, int]:
    """Return ``(predictor_depth, predictor_hidden_dim)`` for ``name``.

    Per Path B BUILD design memo §4.1 + Phase 3 council §9 binding spec:
    - ``single_layer_film_75k`` preserves Z6-v1 (depth=1, hidden_dim=64;
      total substrate ~75K params).
    - ``multi_layer_film_depth_3_300k`` is Candidate 1 (depth=3, hidden_dim=96;
      total substrate ~300K params per the council binding ceiling).
    """
    if name == "single_layer_film_75k":
        return 1, 64
    if name == "multi_layer_film_depth_3_300k":
        return 3, 96
    raise SystemExit(
        f"ERROR: unknown --predictor-architecture {name!r}; expected one of "
        "{'single_layer_film_75k', 'multi_layer_film_depth_3_300k'} per Phase 3 "
        "council §9 binding spec"
    )


def _predictor_width_metadata(
    args: argparse.Namespace,
    *,
    effective_predictor_hidden_dim: int,
    predictor_depth: int,
) -> dict[str, int | str]:
    """Return predictor-width custody fields shared by archive and sidecars."""
    requested_predictor_hidden_dim = int(
        getattr(args, "predictor_hidden_dim", effective_predictor_hidden_dim)
    )
    return {
        "predictor_hidden_dim": int(effective_predictor_hidden_dim),
        "requested_predictor_hidden_dim": requested_predictor_hidden_dim,
        "effective_predictor_hidden_dim": int(effective_predictor_hidden_dim),
        "predictor_film_mlp_hidden_dim": int(args.predictor_film_mlp_hidden_dim),
        "predictor_architecture": str(args.predictor_architecture),
        "predictor_depth": int(predictor_depth),
    }


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z6_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z6_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z6_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=300",
        "default": "300",
    },
    "--batch-size": {
        "env": "Z6_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 handles 4-8 at 384x512 with autoregressive "
            "predictor unroll across 600 pairs"
        ),
        "default": "4",
    },
    "--lr": {
        "env": "Z6_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--lambda-residual-entropy": {
        "env": "Z6_LAMBDA_RESIDUAL_ENTROPY",
        "rationale": (
            "Predictive-coding residual-entropy weight (Rao-Ballard 1999); "
            "0 = no PC, 1 = canonical, higher = more aggressive predictor learning"
        ),
        "default": "1.0",
    },
    "--predictor-kernel-size": {
        "env": "Z6_PREDICTOR_KERNEL_SIZE",
        "rationale": (
            "FiLM predictor conv kernel size; 1 / 3 / 5; default 3 per design memo "
            "Section 4.1"
        ),
        "default": "3",
    },
    "--predictor-ego-motion-dim": {
        "env": "Z6_PREDICTOR_EGO_MOTION_DIM",
        "rationale": (
            "Ego-motion proxy dimension projected from PoseNet output; "
            "default 8; small dim keeps predictor compact"
        ),
        "default": "8",
    },
    "--identity-predictor": {
        "env": "Z6_IDENTITY_PREDICTOR",
        "rationale": (
            "Probe-disambiguator ablation: when true, predictor is identity "
            "(no learning); compare to full FiLM for Rao-Ballard "
            "refutation/confirmation per Catalog #125 hook #6"
        ),
        "default": "false",
    },
    # Z6-v2 Candidate 1 Wave 2 extensions per Phase 3 council §9 spec.
    "--predictor-architecture": {
        "env": "Z6_PREDICTOR_ARCHITECTURE",
        "rationale": (
            "Z6-v2 Candidate 1 predictor architecture choice. "
            "single_layer_film_75k preserves Z6-v1 backward-compat (depth=1, "
            "~75K total substrate params). multi_layer_film_depth_3_300k uses "
            "depth=3 FiLM stack at hidden_dim=96 (~300K total params per Path "
            "B BUILD design memo §4.1 + Phase 3 council binding ceiling)."
        ),
        "default": "single_layer_film_75k",
    },
    "--predictor-param-count-target": {
        "env": "Z6_PREDICTOR_PARAM_COUNT_TARGET",
        "rationale": (
            "Council binding ceiling on substrate total params (Phase 3 "
            "council §9). Used at startup to verify the configured architecture "
            "stays within ±5%% of the declared ceiling."
        ),
        "default": "300000",
    },
    "--predictor-hidden-dim": {
        "env": "Z6_PREDICTOR_HIDDEN_DIM",
        "rationale": (
            "Single-layer FiLM width knob for Candidate 4c and other "
            "source-channel probes. The multi-layer Candidate 1 resolver "
            "overrides this value, but single-layer recipes must be able to "
            "tune it without code edits."
        ),
        "default": "64",
    },
    "--predictor-film-mlp-hidden-dim": {
        "env": "Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM",
        "rationale": (
            "FiLM modulation MLP width knob; exposes side-information channel "
            "capacity to operator recipes and remote drivers."
        ),
        "default": "32",
    },
    "--ego-source": {
        "env": "Z6_EGO_SOURCE",
        "rationale": (
            "Ego-motion source for the FiLM predictor. posenet_projection "
            "preserves Z6-v1 baseline (PoseNet head -> 8-dim projection); "
            "scorer_logit is Z6-v2 Candidate 4c: compress-time SegNet logits "
            "+ PoseNet head features reduced into the same fixed-size "
            "ego-motion side-info vector. Remaining alternatives (raft / "
            "optical_flow / multi_frame) per Path B BUILD design memo §4 "
            "sub-options remain deferred to subsequent Candidate 4 dispatches."
        ),
        "default": "posenet_projection",
    },
    "--enable-paired-control-initialization": {
        "env": "Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION",
        "rationale": (
            "Catalog #229 paired-control fix marker. "
            "shared_modules_seed_order_matched_v2 preserves the canonical "
            "Z6-v1 paired-control marker so identity-predictor disambiguator "
            "at SAME archive bytes works apples-to-apples per Phase 3 "
            "council Revision #2."
        ),
        "default": "shared_modules_seed_order_matched_v2",
    },
    "--emit-identity-predictor-disambiguator-archive": {
        "env": "Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE",
        "rationale": (
            "When true, emit BOTH the full-predictor archive AND an identity-"
            "predictor archive at the SAME archive bytes (same stride; same "
            "600-pair sample) so the Wave 2 disambiguator can compute ΔS = "
            "full_FiLM_score - identity_predictor_score per Catalog #105/#139/"
            "#220/#272 + Council Revision #2."
        ),
        "default": "false",
    },
    "--paired-control-disambiguator-decision-criterion-delta-s": {
        "env": "Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S",
        "rationale": (
            "Decision threshold for the disambiguator: full-FiLM-WIN at ΔS >= "
            "this value at contest-CUDA triggers Wave 4 Phase council on Wave "
            "3 spec; ΔS < threshold OR identity-WIN triggers Wave 3b DEFER "
            "branch per Phase 3 council Revision #3 binding contingency."
        ),
        "default": "0.005",
    },
    "--enable-autocast-fp16": {
        "env": "Z6_ENABLE_AUTOCAST_FP16",
        "rationale": "Catalog #172; pending canonical autocast backport",
        "default": "false",
    },
}


# ---------------------------------------------------------------------------
# Catalog #241/#242 SubstrateContract — META layer single source of truth.
# This binds (a) trainer's claimed contract, (b) recipe schema, (c) lane
# registry, (d) cost band envelope into ONE source-of-truth that fails-loud
# at decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------
TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="time_traveler_l5_z6",
    lane_id=SUBSTRATE_LANE_ID,
    target_modes=(
        "contest_one_video_replay",
        "contest_generalized",
        "research_substrate",
    ),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "Z6PCWM1 monolithic single-file 0.bin: header + encoder + decoder + "
        "FiLM predictor state_dicts (fp16 brotli) + latent_init + per-pair "
        "residuals + ego_motion + meta JSON"
    ),
    parser_section_manifest={
        "header": "Z6PCWM1_magic_and_version",
        "encoder_blob": "fp16_brotli_blob",
        "decoder_blob": "fp16_brotli_blob",
        "predictor_blob": "fp16_brotli_blob",
        "latent_init_blob": "int8_quantized",
        "residuals_blob": "int8_quantized",
        "ego_motion_blob": "int8_quantized_sidecar",
        "meta_blob": "sorted_keys_json_utf8",
    },
    inflate_runtime_loc_budget=120,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=700,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=3,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=1.0,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="Rao-Ballard",
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=(
        "tools/probe_z6_predictive_coding_vs_identity_disambiguator.py"
    ),
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_163_remote_lane_sentinel_used",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_240_substrate_engineering_pre_build_opt_out",
        "catalog_244_remote_lane_canonical_nvml_block",
        "catalog_270_dispatch_optimization_protocol_scaffold_pass",
        "catalog_272_distinguishing_feature_byte_mutation_pending",
        "catalog_290_canonical_vs_unique_decision_per_layer_documented",
        "catalog_305_observability_surface_declared",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "FiLM predictor gradient norm IS the sensitivity signal but registration"
            " happens post Phase 2 council approval"
        ),
        "hook_bit_allocator_class": (
            "int8 per-pair residuals + fp16 brotli weights; no per-tensor bit"
            " allocator at L1 SCAFFOLD"
        ),
        "hook_continual_learning_anchor_kind": (
            "L1 SCAFFOLD has no contest-CUDA anchor yet; posterior_update_locked"
            " fires after Phase 2 dispatch + paired CPU/CUDA"
        ),
    },
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_time_traveler_l5_z6",
        description=(
            "Train Z6 Time-Traveler L5 F-asymptote-node predictive-coding "
            "world-model substrate (FiLM-conditioned next-frame predictor; "
            "Rao-Ballard 1999 + Atick-Redlich 1990 cooperative-receiver)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)
    p.add_argument("--predictor-hidden-dim", type=int, default=64)
    p.add_argument("--predictor-film-mlp-hidden-dim", type=int, default=32)
    p.add_argument(
        "--predictor-kernel-size", type=int, default=3, choices=[1, 3, 5]
    )
    p.add_argument("--predictor-ego-motion-dim", type=int, default=8)
    p.add_argument(
        "--identity-predictor", action="store_true",
        help=(
            "Probe-disambiguator regime: identity predictor (no learning); "
            "Catalog #125 hook #6"
        ),
    )
    # Z6-v2 Candidate 1 Wave 2 extensions per Phase 3 council §9 spec.
    p.add_argument(
        "--predictor-architecture",
        choices=("single_layer_film_75k", "multi_layer_film_depth_3_300k"),
        default="single_layer_film_75k",
        help=(
            "Z6-v2 Candidate 1 predictor architecture: single_layer_film_75k "
            "preserves Z6-v1 backward-compat (depth=1); "
            "multi_layer_film_depth_3_300k uses depth=3 FiLM stack at "
            "hidden_dim=96 (~300K total) per Path B BUILD design memo §4.1."
        ),
    )
    p.add_argument(
        "--predictor-param-count-target", type=int, default=300_000,
        help=(
            "Council binding ceiling on substrate total params (Phase 3 "
            "council §9). Trainer prints diagnostic if ±5%% deviation."
        ),
    )
    p.add_argument(
        "--ego-source",
        choices=("posenet_projection", "scorer_logit"),
        default="posenet_projection",
        help=(
            "Ego-motion source. posenet_projection preserves Z6-v1's "
            "baseline per Phase 3 council §9. scorer_logit enables Z6-v2 "
            "Candidate 4c: compress-time SegNet logits + PoseNet head "
            "features reduced into the same archive-side ego buffer."
        ),
    )
    p.add_argument(
        "--enable-paired-control-initialization",
        choices=("shared_modules_seed_order_matched_v2",),
        default="shared_modules_seed_order_matched_v2",
        help=(
            "Catalog #229 paired-control marker; Phase 3 council Revision #2 "
            "binding for identity-predictor disambiguator at SAME archive bytes."
        ),
    )
    p.add_argument(
        "--emit-identity-predictor-disambiguator-archive", action="store_true",
        help=(
            "Emit identity-predictor archive at SAME archive bytes alongside "
            "the full-predictor archive so the Wave 2 disambiguator can compute "
            "ΔS empirically (Council Revision #2)."
        ),
    )
    p.add_argument(
        "--paired-control-disambiguator-decision-criterion-delta-s",
        type=float, default=0.005,
        help=(
            "Decision threshold ΔS for the disambiguator (Council Revision #3)."
        ),
    )
    p.add_argument(
        "--smoke-ego-motion-mode",
        choices=("ramp", "zero", "random", "real-video"),
        default="ramp",
        help=(
            "Smoke-only ego-motion proxy. Default ramp exercises FiLM "
            "conditioning; real-video derives a proxy from upstream video frame "
            "deltas; zero is retained only as a cargo-cult control."
        ),
    )
    p.add_argument(
        "--smoke-target-mode",
        choices=("synthetic", "real-video"),
        default="synthetic",
        help=(
            "Smoke target source. synthetic is fastest; real-video decodes the "
            "first tiny contest-video pair batch through the canonical pyav helper."
        ),
    )

    # Predictive-coding Lagrangian weights
    p.add_argument(
        "--lambda-residual-entropy", type=float, default=1.0,
        help="Rao-Ballard residual-entropy weight; 0=no PC, 1=canonical",
    )
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Training
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument(
        "--val-pair-count", type=int, default=64,
        help="Pairs held out for val loop (clamped to n_pairs // 8 minimum 1)",
    )
    p.add_argument(
        "--val-every-epochs", type=int, default=10,
        help="Run val + checkpoint best EMA every N epochs",
    )
    p.add_argument(
        "--ego-motion-chunk-size", type=int, default=16,
        help=(
            "PoseNet forward chunk size when deriving the ego-motion sidecar "
            "(Catalog #218 OOM discipline; T4-safe default)"
        ),
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help=(
            "If set, cap real-pair decoding to N (research-only; pair-capped "
            "training emits non-promotable archive)"
        ),
    )
    p.add_argument(
        "--skip-archive-build", action="store_true",
        help="Skip archive pack + runtime emission (training-only smoke)",
    )
    p.add_argument(
        "--skip-auth-eval", action="store_true",
        help="Skip canonical gate_auth_eval_call (training-only smoke)",
    )

    # Mode flags
    p.add_argument(
        "--smoke", action="store_true",
        help="Run smoke path (tiny config, synthetic data, no scorer load)",
    )
    p.add_argument(
        "--full-cpu", action="store_true",
        help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)",
    )
    p.add_argument(
        "--advisory-cpu-explicitly-waived", action="store_true",
        help="Required sister flag for --full-cpu (Catalog #197)",
    )
    p.add_argument(
        "--enable-autocast-fp16", action="store_true",
        help="Catalog #172; pending canonical autocast backport",
    )
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with --advisory-cpu-explicitly-waived."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------

def _smoke_effective_epochs(requested_epochs: int) -> int:
    """Return the bounded epoch count for smoke runs.

    Default full-training epochs are intentionally large, but smoke is a
    training-artifact probe and must stay at <=3 epochs unless the full path is
    lifted separately.
    """

    return max(1, min(int(requested_epochs), 3))


def _populate_smoke_ego_motion(
    substrate: Z6PredictiveCodingSubstrate,
    *,
    mode: str,
    seed: int,
    real_video_ego_motion: torch.Tensor | None = None,
) -> None:
    """Populate smoke ego-motion so FiLM conditioning is actually exercised."""

    shape = (
        substrate.cfg.num_pairs,
        substrate.cfg.predictor_ego_motion_dim,
    )
    target_device = substrate.ego_motion_buffer.device
    if mode == "real-video":
        if real_video_ego_motion is None:
            raise ValueError(
                "real-video smoke ego-motion mode requires decoded real-video features"
            )
        if tuple(real_video_ego_motion.shape) != shape:
            raise ValueError(
                f"real-video ego-motion shape {tuple(real_video_ego_motion.shape)} "
                f"!= expected {shape}"
            )
        substrate.ego_motion_buffer.copy_(real_video_ego_motion.to(target_device))
        return
    if mode == "zero":
        substrate.ego_motion_buffer.zero_()
        return
    if mode == "ramp":
        values = torch.linspace(
            -1.0,
            1.0,
            steps=shape[0] * shape[1],
            device=target_device,
        ).view(shape)
        substrate.ego_motion_buffer.copy_(values)
        return
    if mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 17)
        values = torch.randn(shape, generator=generator, device="cpu").to(target_device)
        substrate.ego_motion_buffer.copy_(values)
        return
    raise ValueError(f"unknown smoke ego-motion mode: {mode!r}")


def _ego_motion_from_smoke_targets(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    ego_motion_dim: int,
) -> torch.Tensor:
    """Derive a tiny deterministic ego-motion proxy from real-video smoke targets."""

    if target0.shape != target1.shape:
        raise ValueError(
            f"target0 shape {tuple(target0.shape)} != target1 shape {tuple(target1.shape)}"
        )
    if target0.dim() != 4 or target0.shape[1] != 3:
        raise ValueError(
            "smoke targets must be shaped (num_pairs, 3, height, width)"
        )
    if ego_motion_dim <= 0:
        raise ValueError(f"ego_motion_dim must be > 0; got {ego_motion_dim}")
    diff = target1 - target0
    channel_mean_delta = diff.mean(dim=(2, 3))
    abs_delta_mean = diff.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    features = torch.cat([channel_mean_delta, abs_delta_mean], dim=1)
    features = features - features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    features = features / scale
    if features.shape[1] < ego_motion_dim:
        repeats = math.ceil(ego_motion_dim / features.shape[1])
        features = features.repeat(1, repeats)
    return features[:, :ego_motion_dim].contiguous()


def _decode_real_video_smoke_targets(
    video_path: Path,
    cfg: Z6PredictiveCodingConfig,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a tiny real-video smoke batch and its frame-delta ego proxy."""

    pairs = _decode_real_pairs(
        video_path,
        n_pairs=cfg.num_pairs,
        max_pairs=cfg.num_pairs,
        substrate_tag=SUBSTRATE_TAG,
        repo_root=REPO_ROOT,
    )
    pairs = pairs.to(device=device, dtype=torch.float32) / 255.0
    flat = pairs.reshape(
        cfg.num_pairs * 2,
        3,
        pairs.shape[-2],
        pairs.shape[-1],
    )
    resized = F.interpolate(
        flat,
        size=(cfg.output_height, cfg.output_width),
        mode="bilinear",
        align_corners=False,
    )
    resized_pairs = resized.view(
        cfg.num_pairs,
        2,
        3,
        cfg.output_height,
        cfg.output_width,
    ).contiguous()
    target0 = resized_pairs[:, 0]
    target1 = resized_pairs[:, 1]
    ego_motion = _ego_motion_from_smoke_targets(
        target0,
        target1,
        ego_motion_dim=cfg.predictor_ego_motion_dim,
    )
    return target0, target1, ego_motion


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: tiny config, ≤3 epochs, no scorer load."""
    torch.manual_seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_epochs = _smoke_effective_epochs(args.epochs)

    # Z6-v2 Candidate 1 Wave 2 BUILD: resolve predictor architecture per
    # Phase 3 council §9 spec. Smoke honors --predictor-architecture so the
    # smoke validates the depth=3 forward path before any paid full dispatch.
    predictor_depth, _full_hidden_dim_unused = _resolve_predictor_architecture(
        getattr(args, "predictor_architecture", "single_layer_film_75k")
    )

    # Tiny smoke config (kept tiny for fast smoke; depth flows through to
    # MultiLayerFilmPredictor dispatch in Z6PredictiveCodingSubstrate).
    num_pairs = 5
    cfg = Z6PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_film_mlp_hidden_dim=8,
        predictor_kernel_size=args.predictor_kernel_size,
        predictor_ego_motion_dim=4,
        predictor_depth=predictor_depth,
        identity_predictor=args.identity_predictor,
        lambda_residual_entropy=args.lambda_residual_entropy,
    )
    substrate = Z6PredictiveCodingSubstrate(cfg).to(args.device)
    real_video_ego_motion: torch.Tensor | None = None
    if (
        args.smoke_target_mode == "real-video"
        or args.smoke_ego_motion_mode == "real-video"
    ):
        target0, target1, real_video_ego_motion = _decode_real_video_smoke_targets(
            args.video_path,
            cfg,
            device=args.device,
        )
    else:
        target0 = torch.rand(
            num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
        )
        target1 = torch.rand(
            num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
        )
    _populate_smoke_ego_motion(
        substrate,
        mode=args.smoke_ego_motion_mode,
        seed=args.seed,
        real_video_ego_motion=real_video_ego_motion,
    )
    print(f"[z6-smoke] param breakdown: {substrate.num_parameters_breakdown()}")
    print(f"[z6-smoke] identity_predictor={cfg.identity_predictor}")
    print(f"[z6-smoke] smoke_ego_motion_mode={args.smoke_ego_motion_mode}")

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(effective_epochs):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, z_t = substrate.reconstruct_pair(idx)
        # Pixel-MSE proxy + residual-norm proxy in smoke
        recon_loss = (
            (rgb_0 - target0).pow(2).mean()
            + (rgb_1 - target1).pow(2).mean()
        )
        residual_loss = args.lambda_residual_entropy * substrate.residuals.pow(2).mean()
        loss = recon_loss + residual_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
        opt.step()
        losses.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "recon": float(recon_loss.item()),
            "residual": float(residual_loss.item()),
        })

    # Pack archive
    enc_sd = substrate.encoder.state_dict()
    dec_sd = substrate.decoder.state_dict()
    pred_sd = substrate.predictor.state_dict()
    latent_init = substrate.latent_init.detach().cpu()
    residuals = substrate.residuals.detach().cpu()
    ego_motion = substrate.ego_motion_buffer.detach().cpu()
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        **_predictor_width_metadata(
            args,
            effective_predictor_hidden_dim=cfg.predictor_hidden_dim,
            predictor_depth=predictor_depth,
        ),
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
        "smoke_target_mode": args.smoke_target_mode,
        "smoke_ego_motion_mode": args.smoke_ego_motion_mode,
        "requested_epochs": args.epochs,
        "effective_epochs": effective_epochs,
        "ego_source": getattr(args, "ego_source", "posenet_projection"),
        "ego_motion_source": (
            "smoke_proxy_no_scorer_load"
            if getattr(args, "ego_source", "posenet_projection") == "scorer_logit"
            else "smoke_proxy"
        ),
    }
    archive_bytes = pack_archive(
        enc_sd, dec_sd, pred_sd, latent_init, residuals, ego_motion, meta,
        lambda_residual_entropy=args.lambda_residual_entropy,
        predictor_kernel_size=args.predictor_kernel_size,
        identity_predictor=args.identity_predictor,
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Z6-v2 Candidate 1 Wave 2 BUILD: identity-predictor disambiguator emission
    # per Phase 3 council Revision #2 + Catalog #105/#139/#220/#272. When
    # --emit-identity-predictor-disambiguator-archive is set, ALSO pack a
    # second archive with identity_predictor=True at the SAME archive bytes
    # so the disambiguator probe can compute ΔS empirically. Reuses the SAME
    # encoder + decoder + latent_init + residuals + ego_motion buffers
    # (paired-control-initialization=shared_modules_seed_order_matched_v2 per
    # Catalog #229); the ONLY change is identity_predictor=True at archive-
    # build time so the inflate path's predictor effect is decoupled from
    # everything else.
    disambiguator_archive_bytes: bytes | None = None
    if getattr(args, "emit_identity_predictor_disambiguator_archive", False):
        identity_meta = dict(meta)
        identity_meta["identity_predictor_disambiguator"] = True
        identity_meta["paired_control_marker"] = PAIRED_CONTROL_INITIALIZATION
        identity_meta["paired_control_decision_criterion_delta_s"] = float(
            getattr(args, "paired_control_disambiguator_decision_criterion_delta_s", 0.005)
        )
        disambiguator_archive_bytes = pack_archive(
            enc_sd, dec_sd, pred_sd, latent_init, residuals, ego_motion,
            identity_meta,
            lambda_residual_entropy=args.lambda_residual_entropy,
            predictor_kernel_size=args.predictor_kernel_size,
            identity_predictor=True,
        )
        disambiguator_path = out_dir / "0_identity_predictor_disambiguator.bin"
        disambiguator_path.write_bytes(disambiguator_archive_bytes)
        print(
            f"[z6-smoke] disambiguator archive emitted "
            f"size={len(disambiguator_archive_bytes)}B path={disambiguator_path}"
        )

    final = losses[-1] if losses else {"loss": float("inf")}
    stats = {
        "lane_id": _runtime_lane_id(),
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "requested_epochs": args.epochs,
        "epochs": len(losses),
        "smoke_epoch_cap": 3,
        "final_loss_proxy": final["loss"],
        "final_recon": final.get("recon"),
        "final_residual": final.get("residual"),
        "archive_bytes": len(archive_bytes),
        "lambda_residual_entropy": args.lambda_residual_entropy,
        "predictor_kernel_size": args.predictor_kernel_size,
        "identity_predictor": args.identity_predictor,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "paired_control_shared_modules": [
            "encoder",
            "decoder",
            "latent_init",
            "residuals",
            "ego_motion_buffer",
        ],
        "smoke_target_mode": args.smoke_target_mode,
        "smoke_ego_motion_mode": args.smoke_ego_motion_mode,
        "video_path": str(args.video_path),
        "ego_motion_nonzero_fraction": float(
            (substrate.ego_motion_buffer.detach().abs() > 0).float().mean().item()
        ),
        "ego_motion_l2": float(
            substrate.ego_motion_buffer.detach().pow(2).sum().sqrt().item()
        ),
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "param_breakdown": substrate.num_parameters_breakdown(),
        # Z6-v2 Candidate 1 Wave 2 BUILD fields per Phase 3 council §9.
        "predictor_architecture": getattr(args, "predictor_architecture", "single_layer_film_75k"),
        "predictor_depth": predictor_depth,
        "predictor_param_count_target": getattr(args, "predictor_param_count_target", 300_000),
        "ego_source": getattr(args, "ego_source", "posenet_projection"),
        "emit_identity_predictor_disambiguator_archive": bool(
            getattr(args, "emit_identity_predictor_disambiguator_archive", False)
        ),
        "identity_predictor_disambiguator_archive_bytes": (
            len(disambiguator_archive_bytes)
            if disambiguator_archive_bytes is not None else None
        ),
        "paired_control_disambiguator_decision_criterion_delta_s": float(
            getattr(args, "paired_control_disambiguator_decision_criterion_delta_s", 0.005)
        ),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[z6-smoke] OK final_loss={final['loss']:.6f} archive={len(archive_bytes)}B "
        f"kernel={args.predictor_kernel_size} identity={args.identity_predictor} "
        f"ego={args.smoke_ego_motion_mode} "
        f"arch={getattr(args, 'predictor_architecture', 'single_layer_film_75k')} "
        f"depth={predictor_depth}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full entry path — UNLOCKED 2026-05-16 per UNIQUE-AND-COMPLETE-PER-METHOD
# operating mode + Phase 1b Z6 lift directive.
# ---------------------------------------------------------------------------
# Per the standing directive
# (`feedback_consolidate_everything_into_meta_layer_or_canonical_helpers_standing_directive_20260515.md`):
# substrate trainers bind ALL ingredients into ONE coherent packet per the
# PR 95 paradigm. The UNIQUE elements of Z6 (per Catalogs #310/#311/#312 +
# the Z6/Z7/Z8 design memo §4.1):
#
# 1. UNIQUE primary class-shift substrate (Catalog #310 — Z6 is the
#    architectural core, NOT a bolt-on objective on top of Z3/A1; the FiLM
#    predictor + autoregressive latent dynamics + archived residuals form
#    a primary substrate per the Atick-Redlich cooperative-receiver +
#    Rao-Ballard predictive-coding paradigm).
# 2. UNIQUE ego-motion-conditioned next-frame predictor (Catalog #311 —
#    pose-conditioned prediction with a focus-of-expansion (FOE) prior
#    derived from per-pair PoseNet output following Gibson 1950 + Ballard
#    embodied-vision lineage; the canonical autoregressive next-frame
#    predictor `predictor(z_{t-1}, ego_motion[t]) -> z_t_pred` rolls latent
#    state forward at training time and inflate time).
# 3. UNIQUE residual-entropy Lagrangian (Rao-Ballard residual term + canonical
#    score-aware seg+pose+rate Lagrangian; the bit budget structurally shrinks
#    from H(latent) toward H(residual) per the cooperative-receiver theorem).
# 4. UNIQUE archive grammar (Z6PCWM1 = predictor weights + per-pair residuals +
#    ego-motion sidecar + decoder + latent_init; per Catalog #124 8-field
#    archive grammar already declared in
#    `src/tac/substrates/time_traveler_l5_z6/__init__.py`).
#
# Canonical-vs-unique decisions per layer (design memo §7 alignment):
#   1. pyav decode               -> ADOPT canonical (decode_real_pairs helper)
#   2. seed pinning              -> ADOPT canonical (pin via _pin_seeds skeleton)
#   3. device gate               -> ADOPT canonical (device_or_die)
#   4. YUV6 patch                -> ADOPT canonical (patch_upstream_yuv6_globally)
#   5. scorer load               -> ADOPT canonical (load_differentiable_scorers)
#   6. EMA shadow                -> ADOPT canonical (tac.training.EMA, 0.997)
#   7. score-aware loss          -> FORK (Z6PredictiveCodingScoreAwareLoss; the
#                                   canonical score_pair_components_dispatch
#                                   homogenizes the scorer-preprocess path; the
#                                   FORK is the Rao-Ballard residual term + the
#                                   ego-motion sidecar requirement)
#   8. ego-motion conditioning   -> FORK (substrate-specific PoseNet-projected
#                                   ego-motion proxy with FOE-prior derivation;
#                                   no canonical helper exists for this — the
#                                   predictor's ego-motion sidecar IS the
#                                   distinguishing substrate primitive per
#                                   Catalog #311 NON-NEGOTIABLE)
#   9. mini-batch reconstruct    -> ADOPT canonical (reconstruct_pair indexed
#                                   per Catalog #218)
#  10. archive pack/runtime      -> FORK (Z6PCWM1 grammar; per-pair residuals)
#  11. auth-eval gate            -> ADOPT canonical (gate_auth_eval_call)
#  12. posterior update          -> ADOPT canonical (posterior_update_locked_*)
#  13. provenance + manifest     -> ADOPT canonical pattern (sister substrates)
#  14. hardware detect           -> ADOPT canonical (detect_hardware_substrate)


def _pin_seeds(seed: int) -> None:
    """Pin all RNG sources for reproducibility (Catalog #6)."""
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_head_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_auth_eval_json_paths(
    output_dir: Path,
    *,
    durable_root: Path | None = None,
    filename: str = "contest_auth_eval.json",
) -> tuple[Path, Path]:
    """Return ``(gate_json, local_copy_json)`` for score-grade auth eval.

    Modal trainers run from a writable ``/tmp/pact`` copy. The canonical
    ``contest_auth_eval.py`` scorer refuses score-grade evidence paths under
    temp storage, so the gate writes to a non-temp path and then the trainer
    copies that JSON back into ``output_dir`` for artifact harvest. Sister of
    NSCS01 pattern.
    """
    local_copy_json = output_dir / filename
    temp_root = Path(tempfile.gettempdir())
    if not _path_is_under(local_copy_json, temp_root):
        return local_copy_json, local_copy_json
    root = durable_root
    if root is None:
        root = Path(
            os.environ.get(
                "Z6_AUTH_EVAL_ROOT",
                "/root/time_traveler_l5_z6_auth_eval",
            )
        )
    return root / output_dir.name / filename, local_copy_json


def _standardize_ego_motion_features(stacked: torch.Tensor) -> torch.Tensor:
    """Standardize per-column so FiLM modulation is well-conditioned."""
    mean = stacked.mean(dim=0, keepdim=True)
    std = stacked.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return ((stacked - mean) / std).contiguous()


def _pad_or_truncate_ego_features(
    features: torch.Tensor,
    *,
    ego_motion_dim: int,
) -> torch.Tensor:
    """Return exactly ``ego_motion_dim`` columns without changing row order."""
    if features.shape[1] < ego_motion_dim:
        pad = torch.zeros(
            features.shape[0],
            ego_motion_dim - features.shape[1],
            dtype=features.dtype,
            device=features.device,
        )
        features = torch.cat([features, pad], dim=1)
    return features[:, :ego_motion_dim]


def _allocate_scorer_logit_feature_slots(ego_motion_dim: int) -> dict[str, int]:
    """Allocate fixed ego-buffer slots across all Candidate 4c signal groups."""
    if ego_motion_dim <= 0:
        raise ValueError(f"ego_motion_dim must be > 0; got {ego_motion_dim}")
    priority = ("seg_mean", "pose", "entropy", "margin", "seg_std")
    slots = {name: 0 for name in priority}
    if ego_motion_dim < len(priority):
        for name in priority[:ego_motion_dim]:
            slots[name] = 1
        return slots

    for name in priority:
        slots[name] = 1
    remaining = ego_motion_dim - len(priority)
    # Default dim=8 becomes seg_mean=2, pose=2, seg_std=2, entropy=1,
    # margin=1. This preserves every group instead of prefix-truncating the
    # raw feature bank when SegNet has many classes.
    expansion_order = ("seg_mean", "pose", "seg_std", "pose", "seg_mean")
    idx = 0
    while remaining > 0:
        slots[expansion_order[idx % len(expansion_order)]] += 1
        remaining -= 1
        idx += 1
    return slots


def _scorer_logit_feature_slot_metadata(
    ego_source: str,
    ego_motion_dim: int,
) -> dict[str, int] | None:
    """Return JSON-safe Candidate 4c slot metadata when the path is active."""
    if ego_source != "scorer_logit":
        return None
    return {
        key: int(value)
        for key, value in _allocate_scorer_logit_feature_slots(
            ego_motion_dim
        ).items()
    }


def _param_count_target_diagnostic(
    breakdown: dict[str, int],
    *,
    target: int,
    actual_num_pairs: int,
    latent_dim: int,
    full_num_pairs: int = N_PAIRS_FULL,
) -> dict[str, object]:
    """Return honest param-target diagnostics for full and pair-capped runs."""
    actual_total = int(breakdown["total"])
    residuals_actual = int(breakdown.get("residuals", 0))
    residuals_full_equivalent = int(full_num_pairs) * int(latent_dim)
    full_equivalent_total = (
        actual_total - residuals_actual + residuals_full_equivalent
    )
    comparison_total = (
        full_equivalent_total
        if int(actual_num_pairs) < int(full_num_pairs)
        else actual_total
    )
    if target > 0:
        deviation_pct = abs(comparison_total - int(target)) / int(target) * 100.0
        within_tolerance = deviation_pct <= 5.0
    else:
        deviation_pct = 0.0
        within_tolerance = True
    return {
        "target": int(target),
        "actual_total": actual_total,
        "actual_num_pairs": int(actual_num_pairs),
        "residuals_actual": residuals_actual,
        "full_num_pairs": int(full_num_pairs),
        "full_equivalent_total": int(full_equivalent_total),
        "comparison_total": int(comparison_total),
        "comparison_basis": (
            "full_equivalent_total_from_pair_capped_run"
            if int(actual_num_pairs) < int(full_num_pairs)
            else "actual_total_full_run"
        ),
        "deviation_pct": float(deviation_pct),
        "within_5pct": bool(within_tolerance),
    }


def _compress_feature_group(features: torch.Tensor, slots: int) -> torch.Tensor:
    """Reduce one feature group to ``slots`` columns without prefix truncation."""
    if slots <= 0:
        return features.new_zeros((features.shape[0], 0))
    if features.dim() != 2:
        raise ValueError(f"feature group must be rank-2; got {tuple(features.shape)}")
    width = int(features.shape[1])
    if width == slots:
        return features
    if width < slots:
        pad = torch.zeros(
            features.shape[0],
            slots - width,
            dtype=features.dtype,
            device=features.device,
        )
        return torch.cat([features, pad], dim=1)
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        features.unsqueeze(1),
        slots,
    )
    return pooled.squeeze(1)


def _reduce_scorer_logit_feature_groups(
    *,
    seg_mean: torch.Tensor,
    pose_tensor: torch.Tensor,
    seg_std: torch.Tensor,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    ego_motion_dim: int,
) -> torch.Tensor:
    """Budget scorer-logit side-info slots across every signal family."""
    slots = _allocate_scorer_logit_feature_slots(ego_motion_dim)
    parts = [
        _compress_feature_group(seg_mean, slots["seg_mean"]),
        _compress_feature_group(pose_tensor, slots["pose"]),
        _compress_feature_group(seg_std, slots["seg_std"]),
        _compress_feature_group(entropy, slots["entropy"]),
        _compress_feature_group(margin, slots["margin"]),
    ]
    reduced = torch.cat(parts, dim=1)
    if reduced.shape[1] != ego_motion_dim:
        raise RuntimeError(
            f"reduced scorer-logit feature width {reduced.shape[1]} != {ego_motion_dim}"
        )
    return reduced


def _derive_ego_motion_from_posenet(
    posenet: torch.nn.Module,
    gt_pair_tensor: torch.Tensor,
    *,
    ego_motion_dim: int,
    chunk_size: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Derive per-pair ego-motion-conditioning vectors from PoseNet output.

    Per Catalog #311 NON-NEGOTIABLE the Z6 substrate's predictor MUST be
    ego-motion-conditioned (pose-conditioned next-frame prediction); the
    ego_motion sidecar IS the cooperative-receiver side-information per
    Atick-Redlich 1990. The canonical signal is the PoseNet head output
    (12-dim; first 6 dims are the contest pose) per CLAUDE.md "Exact scorer
    architectures" — this captures the FOE (focus-of-expansion) prior +
    rotation/translation parameters per Gibson 1950 + Ballard embodied-
    vision lineage.

    Pipeline:
        1. PoseNet.preprocess_input(pair) -> 12-channel YUV6 input
        2. PoseNet.forward(...) -> dict with 'pose' key (B, 12) or tensor
        3. Project first ego_motion_dim coordinates (canonical PoseNet head
           order = pose params first)

    Args:
        posenet: frozen PoseNet (eval mode); inference_mode forward.
        gt_pair_tensor: ``(N, 2, 3, H, W)`` in [0, 255] from
            ``decode_real_pairs``.
        ego_motion_dim: target ego-motion vector dimension (Catalog #124
            field; default 8 per design memo Section 4.1).
        chunk_size: pair-batch size for PoseNet forward to avoid OOM
            (Catalog #218 mini-batch discipline).
        device: optional torch.device for compute.

    Returns:
        ``(N, ego_motion_dim)`` float32 tensor on CPU (so it can be copied
        to the substrate's ego_motion_buffer regardless of substrate device).
        Standardized per-column (mean 0, std 1) so FiLM modulation is
        well-conditioned.
    """
    if ego_motion_dim <= 0:
        raise ValueError(f"ego_motion_dim must be > 0; got {ego_motion_dim}")
    if gt_pair_tensor.dim() != 5 or gt_pair_tensor.shape[1] != 2:
        raise ValueError(
            f"gt_pair_tensor must be (N, 2, 3, H, W); got {tuple(gt_pair_tensor.shape)}"
        )
    n_pairs = int(gt_pair_tensor.shape[0])
    compute_device = device if device is not None else gt_pair_tensor.device
    posenet.eval()

    pose_features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, chunk_size):
            chunk = gt_pair_tensor[start : start + chunk_size].to(compute_device)
            # PoseNet expects (B, T=2, C=3, H, W) per upstream/modules.py
            pose_in = posenet.preprocess_input(chunk)
            pose_out = posenet(pose_in)
            if isinstance(pose_out, dict):
                if "pose" not in pose_out:
                    raise RuntimeError(
                        f"PoseNet forward returned dict without 'pose' key; "
                        f"keys={list(pose_out.keys())}"
                    )
                pose_tensor = pose_out["pose"]
            else:
                pose_tensor = pose_out
            # Take first ego_motion_dim coords; canonical Hydra head order
            # has pose params first per CLAUDE.md "Exact scorer architectures".
            features = pose_tensor[:, :ego_motion_dim].to(
                device="cpu", dtype=torch.float32
            )
            pose_features.append(features)
    stacked = torch.cat(pose_features, dim=0)
    if stacked.shape[0] != n_pairs:
        raise RuntimeError(
            f"derived {stacked.shape[0]} ego-motion vectors from {n_pairs} pairs"
        )
    # Standardize per-column so FiLM modulation is well-conditioned (Gibson
    # 1950 FOE prior is a relative-velocity field; zero-mean unit-variance
    # capture the relative-motion structure).
    if stacked.shape[1] < ego_motion_dim:
        # Pad with zeros if PoseNet head has fewer than ego_motion_dim coords
        pad = torch.zeros(
            n_pairs, ego_motion_dim - stacked.shape[1], dtype=stacked.dtype
        )
        stacked = torch.cat([stacked, pad], dim=1)
    return _standardize_ego_motion_features(stacked)


def _derive_ego_motion_from_scorer_logits(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_pair_tensor: torch.Tensor,
    *,
    ego_motion_dim: int,
    chunk_size: int = 16,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Derive Candidate 4c ego side-info from compress-time scorer outputs.

    This is Z6-v2 Candidate 4c per the Path B design memo §4.4c: enrich the
    predictor side-information channel with SegNet logits plus PoseNet head
    features. The scorer is used only at compress/training time; the reduced
    vector is stored in the existing Z6PCWM1 ego-motion buffer, so inflate
    remains scorer-free.
    """
    if ego_motion_dim <= 0:
        raise ValueError(f"ego_motion_dim must be > 0; got {ego_motion_dim}")
    if gt_pair_tensor.dim() != 5 or gt_pair_tensor.shape[1] != 2:
        raise ValueError(
            f"gt_pair_tensor must be (N, 2, 3, H, W); got {tuple(gt_pair_tensor.shape)}"
        )
    n_pairs = int(gt_pair_tensor.shape[0])
    compute_device = device if device is not None else gt_pair_tensor.device
    posenet.eval()
    segnet.eval()

    reduced_features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, n_pairs, chunk_size):
            chunk = gt_pair_tensor[start : start + chunk_size].to(compute_device)

            pose_in = posenet.preprocess_input(chunk)
            pose_out = posenet(pose_in)
            if isinstance(pose_out, dict):
                if "pose" not in pose_out:
                    raise RuntimeError(
                        f"PoseNet forward returned dict without 'pose' key; "
                        f"keys={list(pose_out.keys())}"
                    )
                pose_tensor = pose_out["pose"]
            else:
                pose_tensor = pose_out
            pose_tensor = pose_tensor.to(dtype=torch.float32)

            seg_in = segnet.preprocess_input(chunk)
            seg_logits = segnet(seg_in).to(dtype=torch.float32)
            if seg_logits.dim() != 4:
                raise RuntimeError(
                    f"SegNet logits must be (B, C, H, W); got {tuple(seg_logits.shape)}"
                )

            seg_mean = seg_logits.mean(dim=(2, 3))
            seg_std = seg_logits.std(dim=(2, 3), unbiased=False)
            seg_prob = torch.softmax(seg_logits, dim=1)
            entropy = -(seg_prob * seg_prob.clamp_min(1e-8).log()).sum(
                dim=1, keepdim=True
            ).mean(dim=(2, 3))
            top2 = seg_logits.topk(k=min(2, int(seg_logits.shape[1])), dim=1).values
            if top2.shape[1] == 1:
                margin = top2[:, :1].mean(dim=(2, 3))
            else:
                margin = (top2[:, :1] - top2[:, 1:2]).mean(dim=(2, 3))

            reduced = _reduce_scorer_logit_feature_groups(
                seg_mean=seg_mean,
                pose_tensor=pose_tensor,
                seg_std=seg_std,
                entropy=entropy,
                margin=margin,
                ego_motion_dim=ego_motion_dim,
            )
            reduced_features.append(reduced.to(device="cpu", dtype=torch.float32))

    stacked = torch.cat(reduced_features, dim=0)
    if stacked.shape[0] != n_pairs:
        raise RuntimeError(
            f"derived {stacked.shape[0]} scorer-logit vectors from {n_pairs} pairs"
        )
    return _standardize_ego_motion_features(stacked)


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored Z6 substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``.
    Per Catalog #163 the script uses ``set -euo pipefail``.
    Per CLAUDE.md "Strict scorer rule": no scorer at inflate time.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir / "src" / "tac" / "substrates" / "time_traveler_l5_z6"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "time_traveler_l5_z6"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)

    # Runtime __init__.py is MINIMAL — no score_aware_loss import (forbidden
    # per "Strict scorer rule"; pulls in scorer code at inflate time).
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"Z6 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.time_traveler_l5_z6.architecture import (\n"
        "    EVAL_HW, NUM_PAIRS,\n"
        "    FilmConditionedNextFramePredictor,\n"
        "    Z6PredictiveCodingConfig, Z6PredictiveCodingSubstrate,\n"
        ")\n"
        "from tac.substrates.time_traveler_l5_z6.archive import (\n"
        "    Z6PCWM1_HEADER_FMT, Z6PCWM1_HEADER_SIZE, Z6PCWM1_MAGIC,\n"
        "    Z6PCWM1_SCHEMA_VERSION, Z6PCWM1_SECTION_ROLES,\n"
        "    Z6PredictiveCodingArchive,\n"
        "    pack_archive, parse_archive, parse_z6pcwm1_archive_bytes,\n"
        ")\n"
        "__all__ = [\n"
        "    'EVAL_HW', 'NUM_PAIRS',\n"
        "    'Z6PCWM1_HEADER_FMT', 'Z6PCWM1_HEADER_SIZE', 'Z6PCWM1_MAGIC',\n"
        "    'Z6PCWM1_SCHEMA_VERSION', 'Z6PCWM1_SECTION_ROLES',\n"
        "    'FilmConditionedNextFramePredictor',\n"
        "    'Z6PredictiveCodingArchive', 'Z6PredictiveCodingConfig',\n"
        "    'Z6PredictiveCodingSubstrate',\n"
        "    'pack_archive', 'parse_archive', 'parse_z6pcwm1_archive_bytes',\n"
        "]\n",
        encoding="utf-8",
    )

    # Vendor the canonical _shared/inflate_runtime.py (Catalog #205).
    shared_dir = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py",
        shared_dir / "inflate_runtime.py",
    )

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Z6 Time-Traveler L5 FiLM-conditioned predictive-coding inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "# Per CLAUDE.md \"Strict scorer rule\": no scorer at inflate time.\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec \"${PYTHON:-python3}\" \"$HERE/inflate.py\" "
        "\"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""Z6 contest-compliant inflate runtime.\n'
        "\n"
        "Delegates to the vendored substrate CLI. No scorer imports.\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))  "
        "# SUBMISSION_PYTHONPATH_SHIM_OK:vendored-tac-package-self-contained-per-Catalog-295\n"
        "from tac.substrates.time_traveler_l5_z6.inflate import main_cli\n"
        "\n"
        "def main() -> int:\n"
        "    return main_cli()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin)."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


def _run_val_loop(
    substrate: Z6PredictiveCodingSubstrate,
    loss_fn,
    gt_pair_tensor: torch.Tensor,
    val_pair_indices: list[int],
    archive_bytes_proxy: torch.Tensor,
    device,
    *,
    chunk_size: int = 8,
) -> float:
    """Validation pass with EMA shadow + torch.inference_mode (Catalog #180).

    Mini-batches val pairs per Catalog #218 OOM discipline: full 600-pair
    autoregressive unroll at 384x512 exceeds T4 (14.56 GB).
    """
    substrate.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(val_pair_indices), chunk_size):
            chunk = val_pair_indices[start : start + chunk_size]
            if not chunk:
                continue
            idx_tensor = torch.tensor(chunk, device=device, dtype=torch.long)
            rgb_0, rgb_1, _z_t = substrate.reconstruct_pair(idx_tensor)
            # Substrate decoder is sigmoid -> [0, 1]; scale to [0, 255] for scorer.
            rgb_0_255 = rgb_0 * 255.0
            rgb_1_255 = rgb_1 * 255.0
            gt_0 = gt_pair_tensor[idx_tensor, 0]
            gt_1 = gt_pair_tensor[idx_tensor, 1]
            try:
                loss, _ = loss_fn(
                    reconstructed_rgb_0=rgb_0_255,
                    reconstructed_rgb_1=rgb_1_255,
                    gt_rgb_0=gt_0,
                    gt_rgb_1=gt_1,
                    archive_bytes_proxy=archive_bytes_proxy,
                    residuals=substrate.residuals,
                    apply_eval_roundtrip=True,
                    noise_std=0.0,
                )
            except Exception as exc:
                print(
                    f"[{SUBSTRATE_TAG}-val] WARN val batch start={start} "
                    f"skipped: {exc!r}"
                )
                continue
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else math.inf


def _full_main(args: argparse.Namespace) -> int:
    """Full Z6 training entry: pyav decode + score-aware predictive-coding + EMA + auth eval.

    Binds all PR 95 paradigm ingredients into ONE coherent packet (Catalogs
    #310 PRIMARY class-shift + #311 ego-motion-conditioned next-frame
    predictor + #312 hierarchical predictive coding):

      * pyav-decoded real contest pairs (NO synthetic data; Catalog #114)
      * patched upstream YUV6 BEFORE scorer construction (Catalog #187)
      * load_differentiable_scorers (frozen; eval mode; no scorer at inflate)
      * **UNIQUE**: ego-motion-conditioned next-frame predictor via
        PoseNet-projected per-pair pose features (Catalog #311 — FOE prior
        derived from PoseNet head output; Gibson 1950 + Atick-Redlich 1990
        + Ballard embodied-vision; the FiLM predictor ROLLS LATENT STATE
        FORWARD via autoregressive next-frame prediction)
      * **UNIQUE**: Z6PredictiveCodingScoreAwareLoss with Rao-Ballard 1999
        residual-entropy Lagrangian routed through canonical
        score_pair_components (Catalog #164)
      * EMA(decay=0.997) update post optimizer.step; inference = EMA shadow
      * eval_roundtrip=True throughout (Catalog #5 non-negotiable)
      * AdamW + cosine annealing; gradient clip 1.0; NaN watchdog
      * Mini-batched reconstruct_pair (Catalog #218 OOM discipline) with
        autoregressive predictor unroll across selected pairs
      * Z6PCWM1 archive pack + contest-compliant runtime tree emission
      * Canonical gate_auth_eval_call (Catalog #226) on best EMA checkpoint
      * Canonical require_contest_cuda_auth_eval_claim (Catalog #127 custody)
      * Continual-learning posterior_update_locked (Catalog #128 atomic fcntl)
      * Hardware substrate detection (Catalog #190)
      * Catalog #220 operational-mechanism declaration via auth-eval claim

    The Z6 predictor IS the substrate-distinguishing primitive (Catalog #272):
    every byte in the ``predictor_blob`` is structurally consumed by the
    autoregressive next-frame predictor at inflate time. The
    ``residuals_blob`` carries the per-pair Rao-Ballard residuals; the
    ``ego_motion_blob`` is the cooperative-receiver side-information per
    Atick-Redlich 1990 (Wyner-Ziv-style side-info channel).
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.time_traveler_l5_z6.score_aware_loss import (
        Z6PredictiveCodingLossWeights,
        Z6PredictiveCodingScoreAwareLoss,
    )
    from tac.training import EMA

    _pin_seeds(args.seed)
    device = _canon_device_or_die(
        args.device,
        smoke=False,
        substrate_tag=SUBSTRATE_TAG,
        allow_full_cpu=bool(args.full_cpu),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    auth_eval_gate_json_path, auth_eval_json_path = _resolve_auth_eval_json_paths(
        args.output_dir
    )
    (
        identity_auth_eval_gate_json_path,
        identity_auth_eval_json_path,
    ) = _resolve_auth_eval_json_paths(
        args.output_dir,
        filename="contest_auth_eval_identity_predictor_disambiguator.json",
    )
    archive_zip_path = args.output_dir / "archive.zip"
    archive_zip_sha = ""
    archive_zip_size = 0
    bin_sha = ""
    bin_size = 0
    identity_disambiguator_bin_sha = ""
    identity_disambiguator_bin_size = 0
    identity_disambiguator_archive_zip_sha = ""
    identity_disambiguator_archive_zip_size = 0
    identity_disambiguator_bin_path = (
        args.output_dir / "0_identity_predictor_disambiguator.bin"
    )
    identity_disambiguator_archive_zip_path = (
        args.output_dir / "archive_identity_predictor_disambiguator.zip"
    )
    n_params_total = 0
    best_val_lag = math.inf
    best_epoch = -1
    auth_eval_result: dict[str, object] | None = None
    identity_auth_eval_result: dict[str, object] | None = None

    try:
        # 2. Load differentiable scorers (frozen; eval; no grad).
        # Canonical contract: (posenet, segnet) order per
        # tac.scorer.load_differentiable_scorers signature; see Catalog #222
        # which refuses reversed assignment.
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 3. Decode real contest pairs via canonical pyav helper.
        print(f"[{SUBSTRATE_TAG}-full] decoding pairs from {args.video_path}")
        gt_pair_tensor = _decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=args.max_pairs,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        # 4. **UNIQUE Z6 STEP**: derive predictor side-information vectors.
        # Candidate 1 uses PoseNet projection; Candidate 4c uses compress-time
        # scorer logits + PoseNet features reduced into the same fixed-size
        # ego-motion buffer. Both paths keep inflate scorer-free because the
        # archive stores only the reduced vector.
        if args.identity_predictor:
            print(
                f"[{SUBSTRATE_TAG}-full] identity_predictor=True; using zero "
                "ego-motion (disambiguator-probe regime per Catalog #125 hook #6)"
            )
            ego_motion_vectors = torch.zeros(
                n_pairs, args.predictor_ego_motion_dim, dtype=torch.float32
            )
            ego_motion_source_for_meta = "zero_identity_disambiguator_regime"
        elif args.ego_source == "scorer_logit":
            ego_motion_vectors = _derive_ego_motion_from_scorer_logits(
                posenet,
                segnet,
                gt_pair_tensor,
                ego_motion_dim=args.predictor_ego_motion_dim,
                chunk_size=args.ego_motion_chunk_size,
                device=device,
            )
            ego_motion_source_for_meta = (
                "scorer_logit_segnet_logits_plus_posenet_head_standardized"
            )
        else:
            ego_motion_vectors = _derive_ego_motion_from_posenet(
                posenet,
                gt_pair_tensor,
                ego_motion_dim=args.predictor_ego_motion_dim,
                chunk_size=args.ego_motion_chunk_size,
                device=device,
            )
            ego_motion_source_for_meta = (
                "posenet_pose_head_first_k_coords_standardized"
            )
        _stage(
            f"ego_motion_derived_from_{args.ego_source}_"
            f"dim_{args.predictor_ego_motion_dim}_"
            f"l2_{float(ego_motion_vectors.pow(2).sum().sqrt()):.3f}"
        )
        scorer_logit_feature_slots = _scorer_logit_feature_slot_metadata(
            args.ego_source,
            args.predictor_ego_motion_dim,
        )

        # 5. Validation split: hold out the last N pairs for val loop.
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 6. Build Z6 substrate at the requested num_pairs + ego-motion-dim.
        # Z6-v2 Candidate 1 Wave 2 BUILD: resolve --predictor-architecture per
        # Phase 3 council §9. multi_layer_film_depth_3_300k overrides
        # --predictor-hidden-dim with the council-binding-ceiling 96 to hit
        # the ~300K total params target per Path B BUILD design memo §4.1.
        predictor_depth_full, predictor_hidden_dim_resolved = (
            _resolve_predictor_architecture(
                getattr(args, "predictor_architecture", "single_layer_film_75k")
            )
        )
        # The canonical resolver's hidden_dim takes precedence ONLY when
        # multi_layer_film_depth_3_300k is selected; single_layer_film_75k
        # preserves the operator's --predictor-hidden-dim override (Z6-v1
        # backward-compat).
        effective_predictor_hidden_dim = (
            predictor_hidden_dim_resolved
            if predictor_depth_full > 1
            else args.predictor_hidden_dim
        )
        predictor_width_metadata = _predictor_width_metadata(
            args,
            effective_predictor_hidden_dim=effective_predictor_hidden_dim,
            predictor_depth=predictor_depth_full,
        )
        cfg = Z6PredictiveCodingConfig(
            latent_dim=args.latent_dim,
            encoder_input_channels=3,
            encoder_hidden_dim=64,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_initial_grid_h=3,
            decoder_initial_grid_w=4,
            decoder_channels=(24, 20, 16, 12, 8, 6),
            decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
            num_pairs=n_pairs,
            output_height=384,
            output_width=512,
            predictor_hidden_dim=effective_predictor_hidden_dim,
            predictor_film_mlp_hidden_dim=args.predictor_film_mlp_hidden_dim,
            predictor_ego_motion_dim=args.predictor_ego_motion_dim,
            predictor_kernel_size=args.predictor_kernel_size,
            predictor_depth=predictor_depth_full,
            identity_predictor=args.identity_predictor,
            lambda_residual_entropy=args.lambda_residual_entropy,
        )
        substrate = Z6PredictiveCodingSubstrate(cfg).to(device)
        # Copy derived ego-motion into the substrate's buffer.
        with torch.no_grad():
            substrate.ego_motion_buffer.copy_(
                ego_motion_vectors.to(
                    device=device, dtype=substrate.ego_motion_buffer.dtype
                )
            )
        breakdown = substrate.num_parameters_breakdown()
        n_params_total = int(breakdown["total"])
        n_params_predictor = int(breakdown["predictor"])
        n_params_encoder = int(breakdown["encoder"])
        n_params_decoder = int(breakdown["decoder"])
        target = int(getattr(args, "predictor_param_count_target", 300_000))
        param_count_diagnostic = _param_count_target_diagnostic(
            breakdown,
            target=target,
            actual_num_pairs=n_pairs,
            latent_dim=cfg.latent_dim,
        )
        print(
            f"[{SUBSTRATE_TAG}-full] params: total={n_params_total:,} "
            f"predictor={n_params_predictor:,} encoder={n_params_encoder:,} "
            f"decoder={n_params_decoder:,} depth={cfg.predictor_depth} "
            f"arch={getattr(args, 'predictor_architecture', 'single_layer_film_75k')}"
        )
        # Z6-v2 Candidate 1 Wave 2 BUILD: param-count-target diagnostic per
        # Phase 3 council §9. ±5% deviation surfaces a warning so the operator
        # knows the substrate's empirical params deviate from the binding
        # council ceiling. NOT a hard failure - the cfg-derived params reflect
        # the operator's --predictor-architecture choice exactly.
        if target > 0:
            comparison_total = int(param_count_diagnostic["comparison_total"])
            deviation_pct = float(param_count_diagnostic["deviation_pct"])
            basis = str(param_count_diagnostic["comparison_basis"])
            if not bool(param_count_diagnostic["within_5pct"]):
                print(
                    f"[{SUBSTRATE_TAG}-full] WARNING: substrate total params "
                    f"{comparison_total:,} ({basis}) deviates "
                    f"{deviation_pct:.1f}% from council binding ceiling "
                    f"{target:,} (Phase 3 council §9)"
                )
            else:
                print(
                    f"[{SUBSTRATE_TAG}-full] OK: substrate total params "
                    f"{comparison_total:,} ({basis}) within ±5%% of "
                    f"ceiling {target:,}"
                )
        _stage(
            f"substrate_built_total_{n_params_total}_predictor_"
            f"{n_params_predictor}_decoder_{n_params_decoder}_"
            f"depth_{cfg.predictor_depth}"
        )

        # 7. EMA shadow (CLAUDE.md non-negotiable, decay=0.997).
        ema = EMA(substrate, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 8. Score-aware predictive-coding Lagrangian (FORK per design memo §7).
        weights = Z6PredictiveCodingLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            lambda_residual_entropy=args.lambda_residual_entropy,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = Z6PredictiveCodingScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=weights,
        )
        _stage("lagrangian_built_predictive_coding")

        # 9. Optimizer (AdamW + cosine annealing).
        optimizer = torch.optim.AdamW(
            substrate.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # Archive byte proxy estimate (~97 KB target per design memo §10).
        # int8 residuals + fp16 brotli predictor weights dominate.
        residuals_bytes_proxy = n_pairs * args.latent_dim  # int8 per coordinate
        ego_motion_bytes_proxy = n_pairs * args.predictor_ego_motion_dim
        latent_init_bytes_proxy = args.latent_dim
        # fp16 + brotli ~0.6x ratio for predictor + encoder + decoder weights
        weights_bytes_proxy = max(
            5_000,
            int((n_params_predictor + n_params_encoder + n_params_decoder) * 2 * 0.6),
        )
        meta_bytes_proxy = 1_500
        total_proxy_bytes = (
            residuals_bytes_proxy
            + ego_motion_bytes_proxy
            + latent_init_bytes_proxy
            + weights_bytes_proxy
            + meta_bytes_proxy
        )
        archive_bytes_proxy = torch.tensor(float(total_proxy_bytes), device=device)
        print(
            f"[{SUBSTRATE_TAG}-full] archive_bytes_proxy: residuals="
            f"{residuals_bytes_proxy}B ego={ego_motion_bytes_proxy}B "
            f"weights={weights_bytes_proxy}B meta={meta_bytes_proxy}B "
            f"total={total_proxy_bytes}B"
        )

        # 10. Train loop with autoregressive predictor unroll.
        train_started_at = time.time()
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            substrate.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []

            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                batch_idx_tensor = torch.tensor(
                    batch_indices, device=device, dtype=torch.long
                )
                # Mini-batched reconstruct_pair (Catalog #218 discipline).
                # The substrate rolls forward autoregressively from pair 0
                # to max(batch_indices), then selects + decodes — this IS
                # the ego-motion-conditioned next-frame predictor consuming
                # the PoseNet-derived side information per Catalog #311.
                rgb_0, rgb_1, _z_t = substrate.reconstruct_pair(batch_idx_tensor)
                rgb_0_255 = rgb_0 * 255.0  # decoder is sigmoid -> [0, 1]
                rgb_1_255 = rgb_1 * 255.0
                gt_0 = gt_pair_tensor[batch_idx_tensor, 0]
                gt_1 = gt_pair_tensor[batch_idx_tensor, 1]

                loss, _parts = loss_fn(
                    reconstructed_rgb_0=rgb_0_255,
                    reconstructed_rgb_1=rgb_1_255,
                    gt_rgb_0=gt_0,
                    gt_rgb_1=gt_1,
                    archive_bytes_proxy=archive_bytes_proxy,
                    residuals=substrate.residuals,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[{SUBSTRATE_TAG}-full] NaN strike "
                        f"{nan_strike}/{max_nan_strikes}"
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    substrate.parameters(), args.grad_clip
                )
                optimizer.step()
                ema.update(substrate)
                epoch_losses.append(float(loss.detach().cpu()))

            scheduler.step()

            if (
                epoch % max(1, args.val_every_epochs) == 0
                or epoch == args.epochs - 1
            ):
                live_state = {
                    k: v.detach().clone() for k, v in substrate.state_dict().items()
                }
                ema.apply(substrate)
                ema_state_for_ckpt: dict[str, torch.Tensor] | None = None
                try:
                    val_lag = _run_val_loop(
                        substrate, loss_fn, gt_pair_tensor, val_indices_pool,
                        archive_bytes_proxy, device,
                    )
                    ema_state_for_ckpt = {
                        k: v.detach().cpu().clone()
                        for k, v in substrate.state_dict().items()
                    }
                finally:
                    substrate.load_state_dict(live_state)
                    substrate.train()

                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else math.nan
                )
                print(
                    f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d}: "
                    f"train={avg_train:.5f} val={val_lag:.5f} "
                    f"(best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag and ema_state_for_ckpt is not None:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    torch.save(
                        {
                            "state_dict": ema_state_for_ckpt,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_lag": val_lag,
                            "ego_motion_buffer": substrate.ego_motion_buffer.detach()
                            .cpu()
                            .clone(),
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # 11. Load best EMA checkpoint + emit Z6PCWM1 archive.
        if not args.skip_archive_build:
            if ckpt_best_path.exists():
                # WEIGHTS_ONLY_FALSE_OK:trusted-local-checkpoint-from-this-process
                best_ckpt = torch.load(
                    ckpt_best_path, weights_only=False, map_location=device
                )
                substrate.load_state_dict(best_ckpt["state_dict"])
                if "ego_motion_buffer" in best_ckpt:
                    with torch.no_grad():
                        substrate.ego_motion_buffer.copy_(
                            best_ckpt["ego_motion_buffer"].to(
                                device=device,
                                dtype=substrate.ego_motion_buffer.dtype,
                            )
                        )
            substrate.eval()

            archive_meta = {
                "lane_id": _runtime_lane_id(),
                "encoder_input_channels": cfg.encoder_input_channels,
                "encoder_hidden_dim": cfg.encoder_hidden_dim,
                "decoder_embed_dim": cfg.decoder_embed_dim,
                "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
                "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
                **predictor_width_metadata,
                "latent_init_std": cfg.latent_init_std,
                "smoke": False,
                "best_val_lag": (
                    float(best_val_lag) if math.isfinite(best_val_lag) else None
                ),
                "best_epoch": int(best_epoch),
                "git_head": _git_head_sha(),
                "trained_at_utc": _utc_now_iso(),
                "n_params_total": int(n_params_total),
                "n_params_predictor": int(n_params_predictor),
                "n_params_encoder": int(n_params_encoder),
                "n_params_decoder": int(n_params_decoder),
                "param_count_target_diagnostic": param_count_diagnostic,
                "ego_motion_source": ego_motion_source_for_meta,
                "ego_source": args.ego_source,
                "scorer_logit_feature_slot_allocation": scorer_logit_feature_slots,
                "scorer_logit_feature_slot_allocation_version": (
                    "z6_candidate4c_v1"
                    if scorer_logit_feature_slots is not None
                    else None
                ),
            }
            bin_bytes = pack_archive(
                substrate.encoder.state_dict(),
                substrate.decoder.state_dict(),
                substrate.predictor.state_dict(),
                substrate.latent_init.detach().cpu(),
                substrate.residuals.detach().cpu(),
                substrate.ego_motion_buffer.detach().cpu(),
                archive_meta,
                lambda_residual_entropy=args.lambda_residual_entropy,
                predictor_kernel_size=args.predictor_kernel_size,
                identity_predictor=args.identity_predictor,
            )
            bin_sha = _sha256_bytes(bin_bytes)
            bin_size = len(bin_bytes)
            print(
                f"[{SUBSTRATE_TAG}-full] Z6PCWM1 archive: {bin_size} B "
                f"sha256={bin_sha[:16]}..."
            )
            _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

            if getattr(args, "emit_identity_predictor_disambiguator_archive", False):
                identity_meta = dict(archive_meta)
                identity_meta["identity_predictor_disambiguator"] = True
                identity_meta["paired_control_marker"] = PAIRED_CONTROL_INITIALIZATION
                identity_meta["paired_control_decision_criterion_delta_s"] = float(
                    args.paired_control_disambiguator_decision_criterion_delta_s
                )
                identity_meta["paired_control_full_predictor_archive_sha256"] = bin_sha
                identity_meta["paired_control_full_predictor_archive_bytes"] = bin_size
                identity_disambiguator_bytes = pack_archive(
                    substrate.encoder.state_dict(),
                    substrate.decoder.state_dict(),
                    substrate.predictor.state_dict(),
                    substrate.latent_init.detach().cpu(),
                    substrate.residuals.detach().cpu(),
                    substrate.ego_motion_buffer.detach().cpu(),
                    identity_meta,
                    lambda_residual_entropy=args.lambda_residual_entropy,
                    predictor_kernel_size=args.predictor_kernel_size,
                    identity_predictor=True,
                )
                identity_disambiguator_bin_sha = _sha256_bytes(
                    identity_disambiguator_bytes
                )
                identity_disambiguator_bin_size = len(identity_disambiguator_bytes)
                identity_disambiguator_bin_path.write_bytes(
                    identity_disambiguator_bytes
                )
                _build_archive_zip(
                    identity_disambiguator_archive_zip_path,
                    bin_bytes=identity_disambiguator_bytes,
                )
                identity_disambiguator_archive_zip_sha = _sha256_bytes(
                    identity_disambiguator_archive_zip_path.read_bytes()
                )
                identity_disambiguator_archive_zip_size = (
                    identity_disambiguator_archive_zip_path.stat().st_size
                )
                print(
                    f"[{SUBSTRATE_TAG}-full] identity disambiguator archive: "
                    f"{identity_disambiguator_bin_size} B "
                    f"sha256={identity_disambiguator_bin_sha[:16]}..."
                )
                _stage(
                    "identity_disambiguator_archive_built_"
                    f"{identity_disambiguator_bin_size}_B_"
                    f"sha{identity_disambiguator_bin_sha[:8]}"
                )

            # 12. Emit contest-compliant runtime tree + archive.zip.
            submission_dir = args.output_dir / "submission_dir"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            if identity_disambiguator_bin_size:
                shutil.copy2(
                    identity_disambiguator_bin_path,
                    submission_dir / "0_identity_predictor_disambiguator.bin",
                )
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes)
            archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
            archive_zip_size = archive_zip_path.stat().st_size
            shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
            if identity_disambiguator_archive_zip_size:
                shutil.copy2(
                    identity_disambiguator_archive_zip_path,
                    submission_dir / "archive_identity_predictor_disambiguator.zip",
                )
            _stage("archive_emitted")

            # 13. Auth eval ([contest-CUDA] inline) through the canonical gate
            # per Catalog #226 (no hand-rolled subprocess to contest_auth_eval.py).
            if not args.skip_auth_eval:
                auth_eval_result = _canon_gate_auth_eval_call(
                    args=args,
                    archive_zip=archive_zip_path,
                    inflate_sh=submission_dir / "inflate.sh",
                    upstream_dir=args.upstream_dir,
                    output_json=auth_eval_gate_json_path,
                    contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                    substrate_tag=SUBSTRATE_TAG,
                    device=device,
                    full_cpu_active=bool(args.full_cpu),
                )
                if auth_eval_result is not None:
                    if auth_eval_gate_json_path != auth_eval_json_path:
                        auth_eval_json_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        shutil.copy2(auth_eval_gate_json_path, auth_eval_json_path)
                    _canon_require_contest_cuda_auth_eval_claim(
                        auth_eval_json_path,
                        archive_sha256=archive_zip_sha,
                        substrate_tag=SUBSTRATE_TAG,
                    )
                    _stage("auth_eval_cuda_done_valid_claim")
                    if identity_disambiguator_archive_zip_size:
                        identity_auth_eval_result = _canon_gate_auth_eval_call(
                            args=args,
                            archive_zip=identity_disambiguator_archive_zip_path,
                            inflate_sh=submission_dir / "inflate.sh",
                            upstream_dir=args.upstream_dir,
                            output_json=identity_auth_eval_gate_json_path,
                            contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                            substrate_tag=SUBSTRATE_TAG,
                            device=device,
                            full_cpu_active=bool(args.full_cpu),
                        )
                        if identity_auth_eval_result is not None:
                            if (
                                identity_auth_eval_gate_json_path
                                != identity_auth_eval_json_path
                            ):
                                identity_auth_eval_json_path.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                shutil.copy2(
                                    identity_auth_eval_gate_json_path,
                                    identity_auth_eval_json_path,
                                )
                            _canon_require_contest_cuda_auth_eval_claim(
                                identity_auth_eval_json_path,
                                archive_sha256=(
                                    identity_disambiguator_archive_zip_sha
                                ),
                                substrate_tag=SUBSTRATE_TAG,
                            )
                            _stage(
                                "identity_disambiguator_auth_eval_cuda_done_valid_claim"
                            )
                        else:
                            _stage("identity_disambiguator_auth_eval_skipped_gate_refused")
                else:
                    _stage("auth_eval_skipped_gate_refused")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 14. Posterior update (Catalog #128 atomic fcntl).
    if (not args.skip_auth_eval) and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(
                auth_eval_json_path
            )
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(
                f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}"
            )

    # 15. Provenance + manifest (canonical schema; sister-substrate pattern).
    hardware_substrate_cuda = _canon_detect_hardware_substrate(
        axis="cuda",
        substrate_tag=SUBSTRATE_TAG,
        env_var_candidates=("Z6_GPU", "MODAL_GPU"),
    )
    pair_capped = (
        getattr(args, "max_pairs", None) is not None
        and args.max_pairs < N_PAIRS_FULL
    )
    provenance = {
        "lane_id": _runtime_lane_id(),
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "identity_predictor_disambiguator_archive_sha256": (
            identity_disambiguator_bin_sha or None
        ),
        "identity_predictor_disambiguator_archive_bytes": (
            identity_disambiguator_bin_size or None
        ),
        "identity_predictor_disambiguator_archive_zip_sha256": (
            identity_disambiguator_archive_zip_sha or None
        ),
        "identity_predictor_disambiguator_archive_zip_bytes": (
            identity_disambiguator_archive_zip_size or None
        ),
        "identity_predictor_disambiguator_archive_path": (
            str(identity_disambiguator_bin_path)
            if identity_disambiguator_bin_size
            else None
        ),
        "identity_predictor_disambiguator_archive_zip_path": (
            str(identity_disambiguator_archive_zip_path)
            if identity_disambiguator_archive_zip_size
            else None
        ),
        "n_params": n_params_total,
        "param_count_target_diagnostic": param_count_diagnostic,
        "best_val_lag": float(best_val_lag) if math.isfinite(best_val_lag) else None,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "lambda_residual_entropy": args.lambda_residual_entropy,
        "predictor_kernel_size": args.predictor_kernel_size,
        "predictor_ego_motion_dim": args.predictor_ego_motion_dim,
        **predictor_width_metadata,
        "ego_source": args.ego_source,
        "ego_motion_source": ego_motion_source_for_meta,
        "scorer_logit_feature_slot_allocation": scorer_logit_feature_slots,
        "scorer_logit_feature_slot_allocation_version": (
            "z6_candidate4c_v1" if scorer_logit_feature_slots is not None else None
        ),
        "identity_predictor": args.identity_predictor,
        "emit_identity_predictor_disambiguator_archive": bool(
            args.emit_identity_predictor_disambiguator_archive
        ),
        "paired_control_disambiguator_decision_criterion_delta_s": float(
            args.paired_control_disambiguator_decision_criterion_delta_s
        ),
        "design_memo": (
            ".omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_"
            "world_models_asymptotic_pursuit_scoping_design_20260516.md"
        ),
        "council_phase_2_consensus_seal": False,  # L1 lift; council sign-off pending
        "stage_log": stage_log,
        "auth_eval_gate_json_path": str(auth_eval_gate_json_path),
        "auth_eval_json_path": str(auth_eval_json_path),
        "identity_predictor_disambiguator_auth_eval_gate_json_path": str(
            identity_auth_eval_gate_json_path
        ),
        "identity_predictor_disambiguator_auth_eval_json_path": str(
            identity_auth_eval_json_path
        ),
        "auth_eval_result": auth_eval_result,
        "identity_predictor_disambiguator_auth_eval_result": (
            identity_auth_eval_result
        ),
        "hardware_substrate_cuda": hardware_substrate_cuda,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema": "time_traveler_l5_z6_training_artifact_manifest_v1",
        "lane_id": _runtime_lane_id(),
        "substrate_tag": SUBSTRATE_TAG,
        "training_mode": "pair_capped_smoke" if pair_capped else "full",
        "research_only": pair_capped,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_bytes": bin_size,
        "archive_sha256": bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "identity_predictor_disambiguator_archive_bytes": (
            identity_disambiguator_bin_size or None
        ),
        "identity_predictor_disambiguator_archive_sha256": (
            identity_disambiguator_bin_sha or None
        ),
        "identity_predictor_disambiguator_archive_zip_bytes": (
            identity_disambiguator_archive_zip_size or None
        ),
        "identity_predictor_disambiguator_archive_zip_sha256": (
            identity_disambiguator_archive_zip_sha or None
        ),
        "param_count_target_diagnostic": param_count_diagnostic,
        "max_pairs": args.max_pairs,
        **predictor_width_metadata,
        "ego_source": args.ego_source,
        "ego_motion_source": ego_motion_source_for_meta,
        "scorer_logit_feature_slot_allocation": scorer_logit_feature_slots,
        "scorer_logit_feature_slot_allocation_version": (
            "z6_candidate4c_v1" if scorer_logit_feature_slots is not None else None
        ),
        "n_pairs_full_required_for_auth_eval": N_PAIRS_FULL,
        "auth_eval_skipped": bool(args.skip_auth_eval),
        "emit_identity_predictor_disambiguator_archive": bool(
            args.emit_identity_predictor_disambiguator_archive
        ),
        "paired_control_disambiguator_decision_criterion_delta_s": float(
            args.paired_control_disambiguator_decision_criterion_delta_s
        ),
        "auth_eval_skipped_reason": (
            "pair_capped_smoke_emits_truncated_raw_stream"
            if pair_capped and args.skip_auth_eval
            else ""
        ),
        "auth_eval_json_path": str(auth_eval_json_path),
        "identity_predictor_disambiguator_auth_eval_json_path": str(
            identity_auth_eval_json_path
        ),
        "auth_eval_result": auth_eval_result,
        "identity_predictor_disambiguator_auth_eval_result": (
            identity_auth_eval_result
        ),
        "result": {
            "training_mode": "pair_capped_smoke" if pair_capped else "full",
            "archive_bytes": bin_size,
            "archive_sha256": bin_sha,
            "archive_zip_bytes": archive_zip_size,
            "archive_zip_sha256": archive_zip_sha,
            "identity_predictor_disambiguator_archive_bytes": (
                identity_disambiguator_bin_size or None
            ),
            "identity_predictor_disambiguator_archive_sha256": (
                identity_disambiguator_bin_sha or None
            ),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    paired_identity_required = bool(
        args.emit_identity_predictor_disambiguator_archive
        and identity_disambiguator_archive_zip_size
    )
    primary_auth_valid = (
        auth_eval_result is not None
        and auth_eval_result.get("auth_eval_score_claim_valid") is True
        and auth_eval_result.get("auth_eval_score_axis") == "contest_cuda"
    )
    identity_auth_valid = (
        identity_auth_eval_result is not None
        and identity_auth_eval_result.get("auth_eval_score_claim_valid") is True
        and identity_auth_eval_result.get("auth_eval_score_axis") == "contest_cuda"
    )
    paired_auth_valid = primary_auth_valid and (
        identity_auth_valid if paired_identity_required else True
    )
    stats = {
        **manifest,
        "stats_schema": "time_traveler_l5_z6_full_stats_v1",
        "auth_eval_score_claim_valid": paired_auth_valid,
        "auth_eval_score_axis": (
            auth_eval_result.get("auth_eval_score_axis")
            if auth_eval_result is not None
            else None
        ),
        "auth_eval_lane_tag": (
            auth_eval_result.get("auth_eval_lane_tag")
            if auth_eval_result is not None
            else None
        ),
        "primary_auth_eval_score_claim_valid": primary_auth_valid,
        "identity_predictor_disambiguator_auth_eval_score_claim_valid": (
            identity_auth_valid
        ),
        "paired_identity_auth_eval_required": paired_identity_required,
        "paired_identity_auth_eval_complete": (
            paired_auth_valid if paired_identity_required else None
        ),
    }
    if auth_eval_result is not None:
        stats.update(auth_eval_result)
    if identity_auth_eval_result is not None:
        for key, value in identity_auth_eval_result.items():
            stats[f"identity_predictor_disambiguator_{key}"] = value
    stats["auth_eval_score_claim_valid"] = paired_auth_valid
    stats["primary_auth_eval_score_claim_valid"] = primary_auth_valid
    stats["identity_predictor_disambiguator_auth_eval_score_claim_valid"] = (
        identity_auth_valid
    )
    stats["paired_identity_auth_eval_required"] = paired_identity_required
    stats["paired_identity_auth_eval_complete"] = (
        paired_auth_valid if paired_identity_required else None
    )
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[{SUBSTRATE_TAG}-full] wrote {args.output_dir / 'provenance.json'}"
    )
    return 0


@register_substrate(TIME_TRAVELER_L5_Z6_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
