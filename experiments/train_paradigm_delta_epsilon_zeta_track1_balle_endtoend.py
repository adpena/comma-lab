#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PARADIGM-δεζ Track 1 — Ballé hyperprior + 128K decoder end-to-end trainer.

This is the Phase 1 trainer for T1. It takes the frozen A1
latent table as input, trains:

  - a fresh 128K-parameter Quantizr-class FiLM decoder (
    :class:`tac.paradigm_delta_epsilon_zeta.Decoder128K`)
  - a Ballé 2018 ScaleHyperprior on the latent stream (
    :class:`tac.paradigm_delta_epsilon_zeta.BalleHyperpriorWrapper`)
  - jointly under a Boyd-style adaptive-ρ Lagrangian-ADMM coordinator (
    :class:`tac.paradigm_delta_epsilon_zeta.JointLagrangianADMM`)

against the Phase 1 pixel proxy. The emitted archive/runtime is now
contest-contract-shaped and byte-closed, but score promotion still requires
the normal dispatch-claim, exact-eval custody, and CUDA/CPU evidence gates.
When ``--enable-scorer-domain-loss`` is set, Phase 1 trains against the real
PoseNet/SegNet scorer-domain surrogate under the PR #95 eval-roundtrip/YUV6
inner-loop discipline. Phase 2 replaces the interim state-dict payloads with a
deterministic byte-optimised wire format.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT:
  archive_grammar: three ZIP members ``x`` / ``decoder.bin`` / ``balle.bin``
  parser_section_manifest: member names, SHA-256s, and byte sizes
  inflate_runtime_loc_budget: contest-shaped runtime, budget checked by packet compiler
  runtime_dep_closure: packet-local tac runtime subset + torch/brotli/compressai
  export_format: phase1_contest_contract
  score_aware_loss: Phase 1 real-pixel eval-roundtrip proxy; optional scorer-domain loss
  bolt_on_loc_budget: substrate_engineering
  no_op_detector_planned: exact old/new archive SHA plus inflate consumption

CLAUDE.md non-negotiables — wired through this file
---------------------------------------------------

- **EMA at decay 0.997 with snapshot+restore at eval time** — every
  evaluator call uses the EMA shadow temporarily; live weights are restored
  before training continues.
- **eval_roundtrip = True** — the proxy loss simulates the inflate roundtrip
  (384→874→uint8→384) so the proxy/auth gap stays bounded.
- **noise_std = 0.5** — straight-through quantisation noise applied to the
  decoder output to approximate the int8 cast.
- **NEVER MPS authoritative** — the trainer raises on ``--device mps`` and
  refuses to write ``[contest-CUDA]`` tags from any non-CUDA forward pass.
- **Auth eval is fail-closed through dispatch custody** — ``--auth-eval``
  requires an active lane dispatch claim and still cannot promote without
  exact CUDA/CPU evidence, archive SHA-256, runtime custody, and logs.
- **Predicted scores tagged** — the run manifest carries ``score_band:
  "[predicted; Phase 1 scaffold; not yet empirical]"`` until a
  contest-CUDA result lands.

Smoke mode (Phase 1 build verification)
---------------------------------------

``--smoke`` runs ONE epoch on ONE pair, builds a deterministic
``smoke_archive.zip`` (NOT submission-quality), and exits with rc=0 if the
end-to-end pipeline closed without error. Used by the dispatcher's local
preflight before any GPU spend.

CLI surface (DO NOT INVENT FLAGS — verified by tests)
-----------------------------------------------------

  --output-dir         where to write artifacts (required)
  --device             cuda|cpu (mps refused)
  --epochs             3000 default for Q-FAITHFUL
  --batch-size         16 default
  --learning-rate      1e-4 default
  --aux-learning-rate  1e-3 default (EntropyBottleneck quantile-loss)
  --ema-decay          0.997 default (CLAUDE.md non-negotiable)
  --rate-target-bytes  80000 default
  --seg-target         7e-4 default
  --pose-target        1.7e-4 default
  --rho-init           1.0 default
  --enable-scorer-domain-loss  train against differentiable PoseNet/SegNet
                       score-domain terms instead of pixel-L1 scaffold
  --segmentation-surrogate  soft_cosine|fisher_rao|sinkhorn (default sinkhorn)
  --pixel-l1-anchor-weight  optional pixel-L1 anchor during score-domain training
  --grad-clip-norm     optional global norm clip; remote score-domain default is 1.0
  --eval-every-epochs  100 default
  --auth-eval          refused until hermetic runtime/export custody is closed
  --video-path         real contest video for non-smoke target pixels
  --target-pixels-path optional pre-extracted real target tensor
  --max-target-pairs   optional real-target pair cap for local debug
  --smoke              build-verification mode (1 epoch, 1 pair)
  --seed               20 default (matches PYTHONHASHSEED canonical)
  --canonical-a1-relpath  override A1 canonical dir (default
                       experiments/results/A1_canonical)
  --enable-t13-sqrt-n-budget  opt-in Fridrich √n per-pair latent budget hook
                       (default False; backward-compat). When enabled, the
                       trainer queries `tac.joint_source_rd_bound.per_pair_sqrt_n_budget`
                       to compute the undetectable-bits-per-pair budget for
                       the latent stream and shrinks ``rate_target_bytes``
                       accordingly. Predicted impact tag:
                       ``[predicted; T13 Fridrich sqrt-n latent shrink]``.
                       See memory `feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`.
  --t13-alpha          Fridrich proportionality constant (default 1.0; see
                       Ker-Pevný-Fridrich 2008).
  --t13-current-bits-per-pair  caller-supplied estimate of the trainer's
                       current per-pair latent rate (bits/pair). Default 3.0
                       per A1-substrate empirical anchor (per memo §6).
                       Determines how much rate to reallocate.
  --enable-t19-adaptive-rho  opt-in Boyd §3.4.1 / He-Yang 2000 adaptive-ρ
                       update via the standalone
                       `tac.joint_admm_coordinator.adaptive_rho_step` helper
                       (default False; backward-compat per coherence council
                       recommendation). Predicted impact tag:
                       ``[predicted; T19 adaptive ρ 2-3× convergence speedup;
                       not direct score]``.
                       See memory `feedback_t11_t13_t19_free_lateral_leaps_landed_20260509`.
  --t19-tau-grow       T19 ρ-growth factor (default 2.0; Boyd canonical).
  --t19-tau-shrink     T19 ρ-shrink factor (default 0.5; Boyd canonical).
  --pr95-parity-profile  PR95 intake profile JSON recorded as T1 parity
                       evidence for score-domain training.

Lane class
----------

This trainer is a SUBSTRATE-ENGINEERING lane (per CLAUDE.md HNeRV parity
discipline lesson #7). It builds the score-aware substrate that downstream
representation/codec lanes consume; it does not itself emit a representation
into the contest packet without further bolt-ons. ``lane_class=substrate_engineering``
is recorded in the run manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.paradigm_delta_epsilon_zeta import (  # noqa: E402
    PARADIGM_DELTA_EPS_ZETA_PROVENANCE,
    PARADIGM_DELTA_EPS_ZETA_VERSION,
    BalleHyperpriorConfig,
    BalleHyperpriorWrapper,
    Decoder128K,
    Decoder128KConfig,
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    build_balle_hyperprior,
    build_decoder_128k,
    load_frozen_a1_encoder,
)
from tac.training import EMA  # canonical EMA class, decay default 0.997

# PR #95 binary-forensics replication. See
# `src/tac/differentiable_eval_roundtrip.py` and CLAUDE.md
# "Eval-roundtrip + autograd-YUV6 in training inner loop".
from tac.differentiable_eval_roundtrip import (  # noqa: E402
    Yuv6PatchToken,
    Yuv6RoutingMode,
    apply_eval_roundtrip_during_training,
    apply_mp4_codec_simulation_during_training,
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)
from tac.losses import (  # noqa: E402
    SEGMENTATION_SURROGATE_FISHER_RAO,
    SEGMENTATION_SURROGATE_SINKHORN,
    SEGMENTATION_SURROGATE_SOFT_COSINE,
    scorer_forward_pair,
    scorer_loss_terms_btchw,
    scorer_loss_terms_cached_btchw,
)
# T20 / T22 wire-in (memo
# feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509). Both
# are training-time additive losses; default OFF for backward-compat per
# T13/T19 pattern + coherence council.
from tac.kl_pose_distill import (  # noqa: E402
    DEFAULT_TEMPERATURE as KL_POSE_DEFAULT_TEMPERATURE,
    KLPoseDistillConfig,
    apply_kl_pose_distill,
)
from tac.temporal_consistency_regularizer import (  # noqa: E402
    DEFAULT_BOUNDARY_HANDLING as TC_DEFAULT_BOUNDARY_HANDLING,
    DEFAULT_LAMBDA_WEIGHT as TC_DEFAULT_LAMBDA_WEIGHT,
    TemporalConsistencyConfig,
    apply_temporal_consistency,
)

DEFAULT_SEGMENTATION_SURROGATE = SEGMENTATION_SURROGATE_SINKHORN


# Tier-1 operator-required CLI flags manifest. Catalog #151
# (`check_operator_wrapper_threads_trainer_tier_required_flags`) scans every
# wrapper / dispatch script (.sh under scripts/, .py under tools/ that
# subprocess-invokes a trainer) and refuses any that invokes THIS trainer
# without threading the env-var ladder for each flag below.
#
# Schema per grand council review 2026-05-12 (9/10 PROCEED with R1-R7):
#   "--cli-flag-name": {
#       "env":                  "WRAPPER_ENV_VAR_NAME",       # required
#       "rationale":            "operator-facing reason",     # required
#       "default":              "<str> or None if opt-in",    # optional
#       "satisfied_by_profile": ("profile1", "profile2"),     # R4: profile-equiv
#       "requires":             ("--upstream-flag",),         # MacKay: dep edges
#   }
#
# Acceptance rule (per council R3 + R4 + R5):
#   The check passes if the wrapper EITHER (a) threads the literal flag,
#   OR (b) threads an env-var-gated block referencing meta["env"],
#   OR (c) threads --profile X where X is in meta["satisfied_by_profile"],
#   OR (d) carries same-line `# TIER_REQUIRED_FLAG_WAIVED_OK:<flag>:<reason>`.
#
# Multi-tier: a trainer may also expose TIER_2_OPERATOR_REQUIRED_FLAGS etc.
# The check unions every TIER_N_OPERATOR_REQUIRED_FLAGS module-level Assign.
TIER_1_OPERATOR_REQUIRED_FLAGS = {
    "--enable-autocast-fp16": {
        "env": "T1_ENABLE_AUTOCAST_FP16",
        # F5 fix (non-arbitrariness sweep 2026-05-12): replace stale pre-Amdahl
        # multiplier "4-6x T4" with Amdahl-corrected single-component figure.
        "rationale": (
            "fp16 forward + GradScaler + fp32-Lagrangian cast; Amdahl-component "
            "~1.6x contribution to ~2.5-3.5x compound speedup "
            "(see feedback_adversarial_review_fixup_pass_1_landed_20260512.md "
            "for Amdahl decomposition; savings overlap with soft_cosine)"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--enable-mp4-codec-sim": {
        "env": "T1_ENABLE_MP4_CODEC_SIM",
        "rationale": (
            "differentiable BT.601 chroma 4:2:0 + optional DCT-quant noise STE; "
            "captures ~30% of real mp4 codec losses (chroma + Gaussian noise); "
            "does NOT simulate DCT quant tables, motion comp, deblocking, loop filter"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--mp4-codec-sim-noise-std": {
        "env": "T1_MP4_CODEC_SIM_NOISE_STD",
        "rationale": "per-block Gaussian noise std for mp4 codec sim",
        "default": "0.0",
        "satisfied_by_profile": (),
        "requires": ("--enable-mp4-codec-sim",),  # MacKay: dependency edge
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--enable-t20-kl-pose-distill": {
        "env": "T1_ENABLE_T20_KL_POSE_DISTILL",
        "rationale": (
            "T20 KL-distill on pose axis; teacher_pose_cache eliminates 1 of 4 "
            "scorer forwards per batch (~25% scorer-time reduction, ~1.25x Amdahl)"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--enable-t22-temporal-consistency": {
        "env": "T1_ENABLE_T22_TEMPORAL_CONSISTENCY",
        "rationale": (
            "T22 temporal-consistency regularizer; pose-axis stabilizer; "
            "marginal Δ-score not yet empirically anchored ([predicted])"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--segmentation-surrogate": {
        "env": "SEGMENTATION_SURROGATE",
        "rationale": (
            "Source-faithful T8 sinkhorn scorer-domain default; soft_cosine is "
            "an explicit speed override when the operator chooses approximate "
            "throughput over PR95/T1 parity"
        ),
        "default": DEFAULT_SEGMENTATION_SURROGATE,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--enable-t13-sqrt-n-budget": {
        "env": "T1_ENABLE_T13_SQRT_N_BUDGET",
        "rationale": (
            "Fridrich sqrt(n) latent-rate budget per Ker-Pevný-Fridrich 2008; "
            "rate-allocation reshape; T13 free $0 lateral-leap"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    "--enable-t19-adaptive-rho": {
        "env": "T1_ENABLE_T19_ADAPTIVE_RHO",
        "rationale": (
            "Boyd adaptive ρ-step per Boyd §3.4.1 / He-Yang 2000; "
            "predicted 2-3x ADMM-convergence speedup [predicted; not direct score]"
        ),
        "default": None,
        "satisfied_by_profile": (),
        "requires": (),
        "rationale_audit": ".omx/research/non_arbitrariness_sweep_20260512.md#f5",
    },
    # Catalog #152 (2026-05-12): required_input_file declares this flag's VALUE
    # must be an EXISTING FILE PATH at wrapper-dispatch time. Non-smoke
    # score-domain runs are FATAL without it (line ~2175 emits the blocker
    # `pr95_parity_profile_missing_run_experiments_profile_pr95_hnerv_muon_intake`).
    # Wrapper-side validator: `tools/validate_dispatch_required_inputs.py`.
    # Bug-class anchor: $0.016 burned on Modal A100 at 2026-05-12T17:12
    # (call_id fc-01KREJST89QHFRWJXHAKXD850C) crashed in 15s because the
    # default profile path did not exist on the Modal worker.
    "--pr95-parity-profile": {
        "env": "T1_PR95_PARITY_PROFILE",
        "rationale": (
            "PR95 HNeRV/Muon intake profile JSON; non-smoke score-domain runs "
            "FATAL without it. Generated by "
            "experiments/profile_pr95_hnerv_muon_intake.py."
        ),
        "default": (
            ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"
        ),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            ".venv/bin/python experiments/profile_pr95_hnerv_muon_intake.py "
            "--json-out .omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"
        ),
        "rationale_audit": "feedback_permanent_fix_required_input_validation_20260512.md",
    },
}


CONTEST_AUTH_EVAL_RELATIVE = "experiments/contest_auth_eval.py"
DISPATCH_CLAIMS_RELATIVE = ".omx/state/active_lane_dispatch_claims.md"
INFLATE_ROUNDTRIP_CAMERA_HW = (874, 1164)
EVAL_HW = (384, 512)
PHASE1_SCAFFOLD_ONLY = True
PHASE1_SCAFFOLD_BLOCKERS = (
    "auth_eval_custody_not_wired",
    "exact_cuda_score_not_run",
    "no_op_runtime_consumption_exact_one_video_pending",
    "state_dict_wire_format_not_rate_tightened",
)
A1_BASELINE_ARCHIVE_SHA256 = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_BASELINE_ARCHIVE_SIZE_BYTES = 178_262
CONTEST_UNCOMPRESSED_BYTES = 37_545_489
ARCHIVE_IN_LOOP_MANIFEST_SCHEMA = "t1_archive_in_loop_manifest_v1"
DEFAULT_PR95_PARITY_PROFILE = (
    REPO_ROOT
    / ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"
)
PR95_TRAINER_PARITY_SCHEMA = "pr95_hnerv_muon_t1_trainer_parity_v1"
SCORE_DOMAIN_OBJECTIVE_DIRECT = "direct_score"
SCORE_DOMAIN_OBJECTIVE_AUGMENTED = "augmented_lagrangian"
SCORE_DOMAIN_OBJECTIVE_CHOICES = (
    SCORE_DOMAIN_OBJECTIVE_DIRECT,
    SCORE_DOMAIN_OBJECTIVE_AUGMENTED,
)


def contest_rate_penalty_from_batch_bits(
    rate_bits: torch.Tensor,
    *,
    batch_pairs: int,
    total_pairs: int,
) -> torch.Tensor:
    """Estimate the contest byte term from a minibatch entropy-model output."""

    if batch_pairs <= 0:
        raise ValueError(f"batch_pairs must be positive, got {batch_pairs!r}")
    if total_pairs <= 0:
        raise ValueError(f"total_pairs must be positive, got {total_pairs!r}")
    if batch_pairs > total_pairs:
        raise ValueError(
            f"batch_pairs={batch_pairs!r} cannot exceed total_pairs={total_pairs!r}"
        )
    full_archive_rate_bits = rate_bits * (float(total_pairs) / float(batch_pairs))
    return 25.0 * (full_archive_rate_bits / 8.0) / CONTEST_UNCOMPRESSED_BYTES


def phase1_scaffold_blockers() -> list[str]:
    """Return the hard blockers that keep this trainer scaffold-only."""
    return list(PHASE1_SCAFFOLD_BLOCKERS)


def refuse_phase1_scaffold_path(path_name: str) -> None:
    """Fail closed for Phase 1 paths that can be mistaken for score work."""
    blockers = ", ".join(phase1_scaffold_blockers())
    raise SystemExit(
        f"[t1] {path_name} refused: T1/Ballé Phase 1 is scaffold-only and "
        f"not dispatch/eval/promotable yet. blockers={blockers}. Use --smoke "
        "only for local build verification. Non-smoke score-domain training "
        "requires --enable-scorer-domain-loss plus an active remote dispatch "
        "claim, but auth-eval and promotion remain blocked until the runtime is "
        "hermetic and exact CUDA custody is wired."
    )


def _repo_rel_or_str(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_pr95_trainer_parity_contract(profile_path: Path | None) -> dict:
    """Load the PR95 release-view parity contract for T1 provenance/preflight."""
    if profile_path is None:
        return {
            "status": "missing",
            "profile_path": None,
            "score_claim": False,
            "local_trainer_parity_preflight_passed": False,
            "ready_for_score_bearing_t1_hnerv_parity_dispatch": False,
            "score_bearing_dispatch_blockers": ["pr95_parity_profile_path_not_configured"],
        }
    resolved = profile_path if profile_path.is_absolute() else REPO_ROOT / profile_path
    base = {
        "profile_path": _repo_rel_or_str(resolved),
        "score_claim": False,
        "ready_for_score_bearing_t1_hnerv_parity_dispatch": False,
    }
    if not resolved.is_file():
        return {
            **base,
            "status": "missing",
            "local_trainer_parity_preflight_passed": False,
            "score_bearing_dispatch_blockers": [
                "pr95_parity_profile_missing_run_experiments_profile_pr95_hnerv_muon_intake",
            ],
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            **base,
            "status": "invalid_json",
            "local_trainer_parity_preflight_passed": False,
            "error": str(exc),
            "score_bearing_dispatch_blockers": ["pr95_parity_profile_invalid_json"],
        }
    contract = payload.get("trainer_parity_contract")
    if not isinstance(contract, dict):
        return {
            **base,
            "status": "missing_contract",
            "profile_schema": payload.get("schema"),
            "local_trainer_parity_preflight_passed": False,
            "score_bearing_dispatch_blockers": ["trainer_parity_contract_missing"],
        }
    preflight = contract.get("preflight_contract") or {}
    blockers = list(preflight.get("score_bearing_dispatch_blockers") or [])
    schema_matches = contract.get("schema") == PR95_TRAINER_PARITY_SCHEMA
    return {
        **base,
        "status": "loaded",
        "profile_schema": payload.get("schema"),
        "contract_schema": contract.get("schema"),
        "contract_schema_matches_expected": schema_matches,
        "source_tree_sha256": contract.get("source_tree_sha256"),
        "stage_schedule_digest": contract.get("stage_schedule_digest"),
        "source_stage_count": preflight.get("source_stage_count"),
        "stage_order_matches_release_view": preflight.get("stage_order_matches_release_view"),
        "local_trainer_parity_preflight_passed": bool(
            preflight.get("local_trainer_parity_preflight_passed") and schema_matches
        ),
        "t1_trainer_config": contract.get("t1_trainer_config"),
        "stage_schedule": contract.get("stage_schedule"),
        "score_bearing_dispatch_blockers": blockers,
    }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def eval_roundtrip_decoded(
    decoded: torch.Tensor,
    *,
    noise_std: float,
    enable_eval_roundtrip_in_training: bool = True,
) -> torch.Tensor:
    """Return decoded frames after the differentiable inflate roundtrip.

    decoded: (B, 2, 3, H_eval, W_eval) in [0, 255] (sigmoid * 255)

    Per CLAUDE.md "eval_roundtrip — non-negotiable", the proxy MUST simulate
    the contest's inflate→evaluate path unless an explicit ablation disables
    it. When enabled we:
      1. add small noise to mimic the int8 quantisation residual,
      2. run the PR #95-faithful roundtrip primitive
         (bicubic up + bilinear down + STE uint8 round).
    """
    if not enable_eval_roundtrip_in_training:
        return decoded

    B, P, C, H, W = decoded.shape
    flat = decoded.reshape(B * P, C, H, W)
    if noise_std > 0:
        flat = flat + noise_std * torch.randn_like(flat)
    down = apply_eval_roundtrip_during_training(
        flat,
        simulate_uint8=True,
        simulate_resize=True,
        ste_round=True,
        target_h=INFLATE_ROUNDTRIP_CAMERA_HW[0],
        target_w=INFLATE_ROUNDTRIP_CAMERA_HW[1],
    )
    return down.reshape(B, P, C, H, W)


def eval_roundtrip_pixel_l1(
    decoded: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    noise_std: float,
    enable_eval_roundtrip_in_training: bool = True,
) -> torch.Tensor:
    """Proxy pixel distortion after the PR #95-faithful eval roundtrip."""
    decoded_rt = eval_roundtrip_decoded(
        decoded,
        noise_std=noise_std,
        enable_eval_roundtrip_in_training=enable_eval_roundtrip_in_training,
    )
    return F.l1_loss(decoded_rt, target_pixels)


def _grad_l2_norm(params: list[torch.nn.Parameter]) -> tuple[float, bool]:
    """Return total grad L2 norm and whether every present grad is finite."""
    total_sq = 0.0
    saw_grad = False
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        saw_grad = True
        if not torch.isfinite(grad).all().item():
            return float("nan"), False
        total_sq += float(grad.detach().double().pow(2).sum().item())
    if not saw_grad:
        return 0.0, True
    return math.sqrt(total_sq), True


def assert_score_domain_gradient_reachability(
    *,
    decoder_params: list[torch.nn.Parameter],
    balle_main_params: list[torch.nn.Parameter],
) -> dict[str, float]:
    """Fail closed if scorer-domain loss did not reach trainable weights."""
    decoder_norm, decoder_finite = _grad_l2_norm(decoder_params)
    balle_norm, balle_finite = _grad_l2_norm(balle_main_params)
    if not decoder_finite or not balle_finite:
        raise RuntimeError(
            "[t1] score-domain gradient reachability failed: non-finite gradient"
        )
    if decoder_norm <= 0.0 or balle_norm <= 0.0:
        raise RuntimeError(
            "[t1] score-domain gradient reachability failed: "
            f"decoder_grad_l2={decoder_norm:.6g} balle_main_grad_l2={balle_norm:.6g}"
        )
    return {
        "decoder_grad_l2": float(decoder_norm),
        "balle_main_grad_l2": float(balle_norm),
    }


def assert_stable_train_loss(
    loss: torch.Tensor,
    *,
    max_abs: float,
    epoch: int,
    batch_index: int,
    objective: str,
) -> None:
    """Fail closed before a remote run burns hours in a numerically bad basin."""
    if max_abs <= 0:
        return
    detached = loss.detach()
    if not torch.isfinite(detached).all():
        raise RuntimeError(
            "[t1] non-finite training loss; refusing to update weights "
            f"(epoch={epoch + 1}, batch={batch_index + 1}, objective={objective})"
        )
    value = float(detached.abs().max().item())
    if value > float(max_abs):
        raise RuntimeError(
            "[t1] unstable training loss; refusing to continue remote work "
            f"(abs_loss={value:.6e} > max_abs={float(max_abs):.6e}, "
            f"epoch={epoch + 1}, batch={batch_index + 1}, objective={objective}). "
            "Use --score-domain-objective direct_score for warm-starts, or "
            "explicitly raise --max-stable-train-loss-abs only with a documented "
            "loss-scale justification."
        )


def _cache_tensor_on_cpu(tensor: torch.Tensor, *, pin_for_cuda: bool) -> torch.Tensor:
    """Store frozen scorer targets once, outside the hot scorer loop."""
    cached = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if pin_for_cuda:
        try:
            cached = cached.pin_memory()
        except RuntimeError:
            pass
    return cached


def _cached_batch_to_device(
    cache: torch.Tensor,
    idx: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return indexed cached scorer targets on the active training device."""
    idx_cpu = idx.detach().to(device="cpu", dtype=torch.long)
    batch = cache.index_select(0, idx_cpu)
    return batch.to(device=device, non_blocking=bool(cache.is_pinned()))


# ---------------------------------------------------------------------------
# Smoke target generator (used in --smoke mode without real video data)
# ---------------------------------------------------------------------------


def make_smoke_target(n_pairs: int, latent_dim: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic latent + target frame pair for build-verification only.

    PER CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", this MUST
    be reachable only behind ``--smoke`` and any score derived from it MUST
    be tagged ``[smoke synthetic; not measurable]``.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    latents = torch.randn((n_pairs, latent_dim), generator=g)
    targets = torch.randint(0, 256, (n_pairs, 2, 3, *EVAL_HW), generator=g).float()
    return latents, targets


def _load_upstream_yuv420_to_rgb():
    """Load upstream's PyAV RGB conversion helper without patching upstream."""
    import importlib.util

    frame_utils_path = REPO_ROOT / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location(
        "pact_t1_upstream_frame_utils", frame_utils_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load upstream frame_utils.py from {frame_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def load_real_target_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
) -> torch.Tensor:
    """Decode real contest frame pairs for non-smoke training.

    This mirrors the upstream AVVideoDataset pair order: non-overlapping
    `(0,1), (2,3), ...` pairs, resized to the trainer's `(384, 512)` proxy
    resolution and returned as `(N, 2, 3, H, W)` float32 in `[0, 255]`.
    Synthetic targets are forbidden outside `--smoke`.
    """
    if not video_path.is_file():
        raise FileNotFoundError(
            f"real target video not found: {video_path}. Non-smoke T1 training "
            "requires upstream/videos/0.mkv or --target-pixels-path."
        )
    try:
        import av  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("pyav (`av`) is required for non-smoke T1 training") from exc

    yuv420_to_rgb = _load_upstream_yuv420_to_rgb()
    target_pairs = n_pairs if max_pairs is None else min(n_pairs, max_pairs)
    frames_needed = target_pairs * 2
    frames_chw: list[torch.Tensor] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb_hwc = yuv420_to_rgb(frame)
            rgb_chw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(
                rgb_chw,
                size=EVAL_HW,
                mode="bilinear",
                align_corners=False,
            )
            frames_chw.append(resized.squeeze(0).contiguous())
            if len(frames_chw) >= frames_needed:
                break
    finally:
        container.close()
    if len(frames_chw) < frames_needed:
        raise RuntimeError(
            f"{video_path} yielded {len(frames_chw)} frame(s), need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


def load_target_pixels_from_path(path: Path) -> torch.Tensor:
    """Load pre-extracted real target pixels from a torch payload."""
    if not path.is_file():
        raise FileNotFoundError(f"--target-pixels-path not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("target_pixels", "targets", "frame_pairs", "pairs"):
            value = payload.get(key)
            if torch.is_tensor(value):
                payload = value
                break
    if not torch.is_tensor(payload):
        raise ValueError(
            f"{path} did not contain a target tensor or one of "
            "target_pixels/targets/frame_pairs/pairs"
        )
    tensor = payload.float()
    if tensor.ndim != 5:
        raise ValueError(f"target pixels must have 5 dims; got shape {tuple(tensor.shape)}")
    # Accept either (N, 2, H, W, 3) or (N, 2, 3, H, W).
    if tensor.shape[2] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 1, 4, 2, 3).contiguous()
    if tensor.shape[1:] != (2, 3, *EVAL_HW):
        raise ValueError(
            f"target pixels must have shape (N, 2, 3, {EVAL_HW[0]}, {EVAL_HW[1]}); "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "mps":
        raise SystemExit(
            "[t1] --device mps refused per CLAUDE.md MPS-NOISE rule. "
            "Use cuda for authoritative work or cpu for smoke."
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[t1] --device cuda requested but cuda not available")
    return torch.device(name)


def _canonical_dir_name_from_relpath(value: str) -> str:
    """Return the canonical A1 directory name from a repo-relative path.

    ``load_frozen_a1_encoder`` intentionally accepts a directory name under
    ``experiments/results``. The trainer CLI exposes the more operator-friendly
    relpath, so normalize here and fail closed on ambiguous paths.
    """
    path = Path(value)
    parts = path.parts
    if parts == (path.name,):
        return path.name
    expected_prefix = ("experiments", "results")
    if len(parts) == 3 and parts[:2] == expected_prefix:
        return parts[2]
    raise SystemExit(
        "--canonical-a1-relpath must be either A1_canonical or "
        "experiments/results/<canonical_name>; got "
        f"{value!r}"
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)


def _active_claim_rows(*, lane_id: str, claims_path: Path) -> list[dict]:
    """Return active dispatch-claim rows for ``lane_id`` via the canonical helper."""
    helper = REPO_ROOT / "tools" / "claim_lane_dispatch.py"
    if not helper.exists():
        raise SystemExit(f"[t1] dispatch-claim helper not found: {helper}")
    if not claims_path.exists():
        raise SystemExit(f"[t1] dispatch-claims ledger not found: {claims_path}")
    cmd = [
        sys.executable,
        str(helper),
        "summary",
        "--claims-path", str(claims_path),
        "--format", "json",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            "[t1] dispatch-claim summary failed "
            f"(rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[t1] dispatch-claim summary emitted invalid JSON: {exc}") from exc
    return [row for row in payload.get("active", []) if row.get("lane_id") == lane_id]


def require_active_dispatch_claim(*, lane_id: str | None, claims_path: Path) -> None:
    """Fail closed unless an active same-lane claim exists for auth eval."""
    if not lane_id:
        raise SystemExit(
            "[t1] --auth-eval requires --dispatch-lane-id. Claim the lane "
            "with tools/claim_lane_dispatch.py before running eval."
        )
    rows = _active_claim_rows(lane_id=lane_id, claims_path=claims_path)
    if not rows:
        raise SystemExit(
            f"[t1] --auth-eval refused: no active dispatch claim for lane_id={lane_id!r} "
            f"in {claims_path}. Claim the lane before eval."
        )


def _restore_state_dict_with_loose_buffers(module: torch.nn.Module, state: dict) -> None:
    """``load_state_dict`` variant tolerant of buffer-shape changes.

    The CompressAI ``EntropyBottleneck`` mutates buffer shapes inside
    ``update()``. The default ``load_state_dict`` refuses to load tensors of
    different shape, breaking snapshot+restore. We therefore reassign buffers
    individually (replacing the parameter object) so any shape works.
    """
    own_state = module.state_dict()
    for name, value in state.items():
        if name not in own_state:
            continue
        # Try direct copy_ when shapes match; otherwise replace the buffer.
        cur = own_state[name]
        if cur.shape == value.shape and cur.dtype == value.dtype:
            cur.detach().copy_(value)
        else:
            # Walk the module tree to find the parent and replace the buffer.
            *parents, leaf = name.split(".")
            mod = module
            for p in parents:
                mod = getattr(mod, p)
            if leaf in getattr(mod, "_buffers", {}):
                mod._buffers[leaf] = value.detach().clone()
            elif leaf in getattr(mod, "_parameters", {}):
                # Wrap in nn.Parameter to preserve grad behaviour.
                import torch.nn as nn
                mod._parameters[leaf] = nn.Parameter(
                    value.detach().clone(),
                    requires_grad=mod._parameters[leaf].requires_grad,
                )


def _apply_ema_with_loose_buffers(ema: EMA, module: torch.nn.Module) -> None:
    """``ema.apply()`` variant tolerant of buffer-shape changes.

    Mirrors :func:`_restore_state_dict_with_loose_buffers` but for the EMA
    shadow state (which may also have been recorded BEFORE ``update()``
    mutated buffer shapes).
    """
    _restore_state_dict_with_loose_buffers(module, ema.shadow)


def _eval_ema_proxy(
    *,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    latents: torch.Tensor,
    target_pixels: torch.Tensor,
    noise_std: float,
    enable_eval_roundtrip_in_training: bool,
    eval_batch_size: int,
) -> dict[str, float]:
    """Snapshot+restore eval (CLAUDE.md non-negotiable)."""
    if eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
    decoder_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    balle_state = {k: v.detach().clone() for k, v in balle.state_dict().items()}
    ema_decoder.apply(decoder)
    _apply_ema_with_loose_buffers(ema_balle, balle)
    decoder.eval()
    balle.eval()
    try:
        with torch.no_grad():
            rate_bits = 0.0
            pixel_l1_weighted = 0.0
            pixel_l1_weight = 0
            n_pairs = int(latents.shape[0])
            for start in range(0, n_pairs, int(eval_batch_size)):
                end = min(start + int(eval_batch_size), n_pairs)
                balle_out = balle(latents[start:end])
                decoded = decoder(balle_out["y_hat"])
                target_chunk = target_pixels[start:end]
                chunk_pixel_l1 = eval_roundtrip_pixel_l1(
                    decoded,
                    target_chunk,
                    noise_std=noise_std,
                    enable_eval_roundtrip_in_training=enable_eval_roundtrip_in_training,
                )
                weight = int(target_chunk.numel())
                pixel_l1_weighted += float(chunk_pixel_l1) * float(weight)
                pixel_l1_weight += weight
                rate_bits += float(balle_out["rate_total_bits"])
            pixel_l1 = pixel_l1_weighted / max(pixel_l1_weight, 1)
        return {
            "ema_proxy_pixel_l1": pixel_l1,
            "ema_proxy_rate_bits": rate_bits,
        }
    finally:
        decoder.load_state_dict(decoder_state)
        _restore_state_dict_with_loose_buffers(balle, balle_state)
        decoder.train()
        balle.train()


def _maybe_save_ema_checkpoint(
    *,
    output_dir: Path,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    coord: JointLagrangianADMM,
    proxy_score: float,
    epoch: int,
    best_proxy_score: float,
) -> float:
    """Save EMA shadow as best-proxy checkpoint when it improves."""
    if proxy_score >= best_proxy_score:
        return best_proxy_score
    ckpt = {
        "schema_version": 1,
        "epoch": epoch,
        "proxy_score": proxy_score,
        "decoder_ema_state_dict": ema_decoder.state_dict(),
        "balle_ema_state_dict": ema_balle.state_dict(),
        "coord_state_dict": coord.state_dict(),
        "tag": "[predicted; Phase 1 scaffold; not yet empirical]",
        "scaffold_version": PARADIGM_DELTA_EPS_ZETA_VERSION,
    }
    torch.save(ckpt, output_dir / "checkpoint_best_proxy_ema.pt")
    return proxy_score


def build_archive_from_ema(
    *,
    output_dir: Path,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    latents: torch.Tensor,
    decoder_config: Decoder128KConfig,
    balle_config: BalleHyperpriorConfig,
) -> Path:
    """Materialise an archive.zip + submission_dir for contest_auth_eval.

    Phase 1 contest-compliant emission (operator decision B 2026-05-09):

    Archive layout — three named ZIP members (parser_section_manifest input):
        ``x``           — Brotli-compressed Ballé latent strings, with a
                          ``<I:n_strings><I:len><bytes>...<I:len><bytes>``
                          length-prefixed binary preamble (NOT pickle; per
                          Q1 council Contrarian dissent honored).
        ``decoder.bin`` — torch.save() of EMA decoder state-dict, brotli q11.
        ``balle.bin``   — torch.save() of EMA balle state-dict, brotli q11.

    HNeRV parity discipline lesson 3 substrate-engineering exception: 3
    members (not monolithic single-file) is justified by ``lane_class=
    substrate_engineering`` + parser_section_manifest legibility. Recorded
    in build_manifest.json::hnerv_parity_manifest::archive_grammar=
    ``Phase1-three-member-x-decoder-bin-balle-bin``.
    """
    import brotli
    import io
    import zipfile

    submission_dir = output_dir / "submission_dir"
    submission_dir.mkdir(exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(exist_ok=True)

    # Snapshot+restore for EMA application.
    decoder_state = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    balle_state = {k: v.detach().clone() for k, v in balle.state_dict().items()}
    ema_decoder.apply(decoder)
    _apply_ema_with_loose_buffers(ema_balle, balle)
    decoder.eval()
    balle.eval()
    try:
        balle.update(force=True)
        # Move latents to balle's device for compression.
        device = next(balle.parameters()).device
        latents_dev = latents.to(device)
        with torch.no_grad():
            strings = balle.compress(latents_dev)

        # Q1 consensus: length-prefixed binary serialisation of the Ballé
        # strings list, then brotli q11 wrap. compressai's ``balle.compress``
        # returns ``list[list[bytes]]`` (one per string-stream level); we
        # serialise outer-then-inner length prefixes deterministically.
        x_payload_bytes = _serialise_balle_strings(strings)
        x_member_bytes = brotli.compress(x_payload_bytes, quality=11)

        # Decoder + Balle state-dicts via torch.save into an in-memory buffer,
        # then brotli q11 wrap.
        dec_buf = io.BytesIO()
        torch.save(ema_decoder.state_dict(), dec_buf)
        decoder_member_bytes = brotli.compress(dec_buf.getvalue(), quality=11)

        balle_buf = io.BytesIO()
        torch.save(ema_balle.state_dict(), balle_buf)
        balle_member_bytes = brotli.compress(balle_buf.getvalue(), quality=11)

        archive_path = output_dir / "archive.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, payload in (
                ("x", x_member_bytes),
                ("decoder.bin", decoder_member_bytes),
                ("balle.bin", balle_member_bytes),
            ):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                # Per packet compiler: NON_EXECUTABLE_MODE = 0o644.
                info.external_attr = (0o100000 | 0o644) << 16
                zf.writestr(info, payload)

        # Write inflate.py / inflate.sh / codec.py / model.py.
        _write_runtime(
            submission_dir=submission_dir,
            decoder_config=decoder_config,
            balle_config=balle_config,
        )

        # Place a copy of archive.zip alongside the submission_dir runtime
        # tree so contest_auth_eval can find it via the submission_dir's
        # parent (the canonical contest layout).
        sub_archive = submission_dir / "archive.zip"
        sub_archive.write_bytes(archive_path.read_bytes())
    finally:
        decoder.load_state_dict(decoder_state)
        _restore_state_dict_with_loose_buffers(balle, balle_state)
        decoder.train()
        balle.train()
    return archive_path


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _archive_in_loop_manifest_path(output_dir: Path) -> Path:
    return output_dir / "archive_builds_manifest.json"


def _load_archive_in_loop_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": ARCHIVE_IN_LOOP_MANIFEST_SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "rows": [],
        }
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != ARCHIVE_IN_LOOP_MANIFEST_SCHEMA:
        raise RuntimeError(
            f"archive-in-loop manifest schema mismatch: {path} "
            f"schema={payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("rows"), list):
        raise RuntimeError(f"archive-in-loop manifest rows missing/list-invalid: {path}")
    return payload


def _append_archive_in_loop_manifest(output_dir: Path, row: dict) -> Path:
    path = _archive_in_loop_manifest_path(output_dir)
    payload = _load_archive_in_loop_manifest(path)
    payload["rows"].append(row)
    payload["candidate_count"] = len(payload["rows"])
    payload["ready_for_exact_eval_dispatch"] = any(
        bool(r.get("exact_cuda_eligible")) for r in payload["rows"]
    )
    payload["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _compile_archive_in_loop_packet(
    *,
    candidate_dir: Path,
    output_dir: Path,
    baseline_archive_sha256: str,
    baseline_archive_size_bytes: int,
) -> tuple[Path, int, str]:
    packet_dir = output_dir / "packet_compiled"
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "tools" / "build_phase1_packet_compiler.py"),
        "--input-packet",
        str(candidate_dir / "submission_dir"),
        "--output-dir",
        str(packet_dir),
        "--mode",
        "optimize",
        "--target-mode",
        "contest_one_video_replay",
        "--runtime-dep-closure",
        "torch",
        "brotli",
        "compressai",
        "--export-format",
        "phase1_three_member_x_decoder_bin_balle_bin",
        "--bolt-on-loc-budget",
        "400",
        "--allow-existing-output-dir",
        "--score-affecting-payload-changed",
        "--baseline-archive-sha256",
        baseline_archive_sha256,
        "--baseline-archive-size-bytes",
        str(int(baseline_archive_size_bytes)),
        "--print-result-json",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = output_dir / "packet_compiler.log"
    log_path.write_text(proc.stdout)
    return packet_dir, int(proc.returncode), proc.stdout


def materialize_archive_in_loop_candidate(
    *,
    output_dir: Path,
    decoder: Decoder128K,
    balle: BalleHyperpriorWrapper,
    ema_decoder: EMA,
    ema_balle: EMA,
    latents: torch.Tensor,
    decoder_config: Decoder128KConfig,
    balle_config: BalleHyperpriorConfig,
    epoch: int,
    metrics: dict,
    best_proxy_improved: bool,
    max_candidate_archive_bytes: int,
    baseline_archive_sha256: str,
    baseline_archive_size_bytes: int,
) -> dict:
    """Emit one byte-closed candidate row without claiming score movement."""
    candidate_root = output_dir / "archive_candidates"
    candidate_dir = candidate_root / f"epoch_{int(epoch):06d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_archive_from_ema(
        output_dir=candidate_dir,
        decoder=decoder,
        balle=balle,
        ema_decoder=ema_decoder,
        ema_balle=ema_balle,
        latents=latents,
        decoder_config=decoder_config,
        balle_config=balle_config,
    )
    archive_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_path(archive_path)
    rate_cap_passed = int(archive_bytes) <= int(max_candidate_archive_bytes)
    compiler_blockers: list[str] = []
    packet_dir: Path | None = None
    packet_compile_rc: int | None = None
    if rate_cap_passed:
        packet_dir, packet_compile_rc, compiler_stdout = _compile_archive_in_loop_packet(
            candidate_dir=candidate_dir,
            output_dir=candidate_dir,
            baseline_archive_sha256=baseline_archive_sha256,
            baseline_archive_size_bytes=baseline_archive_size_bytes,
        )
        if packet_compile_rc != 0:
            compiler_blockers.append(f"packet_compiler_rc={packet_compile_rc}")
        try:
            build_manifest_path = packet_dir / "build_manifest.json"
            build_manifest = json.loads(build_manifest_path.read_text())
            compiler_blockers.extend(str(x) for x in build_manifest.get("blockers", []))
        except Exception as exc:
            compiler_blockers.append(f"packet_manifest_unreadable:{type(exc).__name__}")
        if not compiler_stdout:
            compiler_blockers.append("packet_compiler_stdout_empty")
    else:
        compiler_blockers.append(
            "rate_cap_exceeded:"
            f"{int(archive_bytes)}>{int(max_candidate_archive_bytes)}"
        )

    row = {
        "schema_version": "t1_archive_in_loop_candidate_v1",
        "epoch": int(epoch),
        "candidate_dir": candidate_dir.as_posix(),
        "archive_path": archive_path.as_posix(),
        "archive_bytes": int(archive_bytes),
        "archive_sha256": archive_sha256,
        "baseline_archive_sha256": baseline_archive_sha256,
        "baseline_archive_size_bytes": int(baseline_archive_size_bytes),
        "max_candidate_archive_bytes": int(max_candidate_archive_bytes),
        "rate_cap_passed": bool(rate_cap_passed),
        "best_proxy_improved": bool(best_proxy_improved),
        "proxy_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in metrics.items()
        },
        "compiled_packet_dir": packet_dir.as_posix() if packet_dir is not None else None,
        "packet_compile_rc": packet_compile_rc,
        "compiler_blockers": sorted(set(compiler_blockers)),
        "exact_cuda_eligible": bool(rate_cap_passed and not compiler_blockers),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = _append_archive_in_loop_manifest(output_dir, row)
    print(
        "[t1] archive-in-loop candidate "
        f"epoch={epoch} bytes={archive_bytes} "
        f"eligible={row['exact_cuda_eligible']} manifest={manifest_path}"
    )
    return row


def _serialise_balle_strings(strings: object) -> bytes:
    """Serialise BalleHyperpriorWrapper.compress(...) output deterministically.

    The wrapper returns ``dict[str, Any]`` with three fields:
        ``y_strings`` — ``list[bytes]`` (per-batch-element AC payload)
        ``z_strings`` — ``list[bytes]`` (per-batch-element hyperprior payload)
        ``z_shape``   — ``list[int]`` (4-D shape of the discretised z buffer)

    Wire format (little-endian uint32 length prefixes throughout):

        <I:n_y> [ <I:blen> <bytes> ]_n_y
        <I:n_z> [ <I:blen> <bytes> ]_n_z
        <I:n_shape> [ <i:int> ]_n_shape   # z_shape ints (signed for safety)

    The reader in inflate.py deserialises bit-exact via the inverse of this
    format and reconstructs the dict the wrapper's decompress() expects.
    """
    if not isinstance(strings, dict):
        raise RuntimeError(
            f"unexpected balle.compress() return type {type(strings)!r}; "
            "expected dict[str, Any]"
        )

    def pack_u32_count(name: str, value: int) -> bytes:
        if value < 0 or value > 0xFFFF_FFFF:
            raise RuntimeError(f"{name}={value} does not fit uint32")
        return struct.pack("<I", value)

    y_strings = strings.get("y_strings")
    z_strings = strings.get("z_strings")
    z_shape = strings.get("z_shape")
    for name, val, expected in (
        ("y_strings", y_strings, list),
        ("z_strings", z_strings, list),
        ("z_shape", z_shape, list),
    ):
        if not isinstance(val, expected):
            raise RuntimeError(
                f"balle.compress() field {name!r} has type "
                f"{type(val).__name__!r}; expected {expected.__name__}"
    )
    parts: list[bytes] = []
    for stream_name, byte_list in (("y_strings", y_strings), ("z_strings", z_strings)):
        parts.append(pack_u32_count(f"{stream_name}.count", len(byte_list)))
        for blob in byte_list:
            if not isinstance(blob, (bytes, bytearray)):
                raise RuntimeError(
                    f"balle.compress()[{stream_name!r}] element type "
                    f"{type(blob).__name__!r}; expected bytes"
                )
            parts.append(pack_u32_count(f"{stream_name}.blob_len", len(blob)))
            parts.append(bytes(blob))
    parts.append(pack_u32_count("z_shape.count", len(z_shape)))
    for dim in z_shape:
        if not isinstance(dim, int):
            raise RuntimeError(
                f"balle.compress()['z_shape'] element type "
                f"{type(dim).__name__!r}; expected int"
            )
        if dim < 0 or dim > 0x7FFF_FFFF:
            raise RuntimeError(
                f"balle.compress()['z_shape'] dim {dim} does not fit "
                "nonnegative int32"
            )
        parts.append(struct.pack("<i", dim))
    return b"".join(parts)


def _write_packet_local_runtime_modules(src: Path) -> None:
    """Vendor the exact T1 runtime modules needed by inflate.

    The contest packet must not search for the operator checkout at inflate
    time. We therefore copy the narrow decoder/hyperprior runtime subset into
    ``submission_dir/src/tac``. Third-party dependencies remain explicit in the
    packet compiler contract: ``torch``, ``brotli``, and ``compressai``.
    """

    package_root = src / "tac"
    runtime_pkg = package_root / "paradigm_delta_epsilon_zeta"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text(
        "\"\"\"Packet-local tac runtime subset for Phase 1 inflate.\"\"\"\n",
        encoding="utf-8",
    )
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"Packet-local PARADIGM-dezeta runtime modules.\"\"\"\n",
        encoding="utf-8",
    )

    source_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "tac"
        / "paradigm_delta_epsilon_zeta"
    )
    for name in ("decoder_128k.py", "balle_hyperprior.py"):
        text = (source_root / name).read_text(encoding="utf-8")
        if name == "balle_hyperprior.py":
            text = text.replace(
                "Install via `uv pip install compressai==1.2.8`.",
                "Ensure compressai is present in the contest runtime environment.",
            )
        (runtime_pkg / name).write_text(text, encoding="utf-8")


def _write_runtime(
    *,
    submission_dir: Path,
    decoder_config: Decoder128KConfig,
    balle_config: BalleHyperpriorConfig,
) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + src/{codec,model}.py.

    Operator decision B 2026-05-09 — replaces the prior research-only-no-export
    scaffold with a byte-closed contest-compliant runtime tree per the Phase 1
    packet compiler's contract (`tac.phase1_packet_compiler.compile_phase1_packet`).

    Council Q1-Q6 consensus (memo
    `feedback_phase1_trainer_write_runtime_fix_landed_20260509.md`):

    * Q1 — Ballé strings are length-prefixed binary (NOT pickle), then
      brotli q11 wrapped, stored as ZIP member ``x``.
    * Q2 — 3 named ZIP members (``x`` + ``decoder.bin`` + ``balle.bin``) per
      lane_class=substrate_engineering exception to HNeRV parity lesson 3.
    * Q3 — per-video ``<base>.raw`` output at camera-resolution (874, 1164)
      uint8 RGB per the contest contract (see upstream/submissions/baseline_fast/inflate.sh).
    * Q4 — runtime_dep_closure = (torch, brotli, compressai); 3 deps; ≤100
      LOC inflate.sh enforced by packet compiler.
    * Q5 — trainer is mode-agnostic; emits ONE canonical contest packet.
    * Q6 — inflate.py imports ZERO scorer code (no PoseNet/SegNet/rgb_to_yuv6).

    6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default"):

    1. Sensitivity-map: 3-member ZIP layout is the encoder-side dual of the
       packet compiler's parser_section_manifest. Sensitivity-driven
       allocations land as per-section byte sizes.
    2. Pareto frontier: enforces Pareto constraints implicitly (rate via
       brotli q11, archive size via per-section bytes).
    3. Bit-allocator hook: DIRECT — 3-member archive layout IS the bit
       allocator's encoder-side output.
    4. Cathedral autopilot: every Phase 1 trainer dispatch flows into
       ``compile_phase1_packet(mode="optimize")`` for byte-closure verification.
    5. Continual-learning posterior: empirical CUDA + CPU evals on the
       Phase 1 packet trigger per-architecture drift posterior updates.
    6. Probe-disambiguator: trainer is mode-agnostic; the packet compiler's
       identity/canonicalize/optimize modes are the disambiguator downstream.
    """
    src = submission_dir / "src"
    src.mkdir(exist_ok=True)
    _write_packet_local_runtime_modules(src)

    # Per Q3+Q4: 3-positional-arg inflate.sh per upstream/submissions/baseline_fast/inflate.sh.
    # Per CLAUDE.md `check_shell_set_e_present`: set -euo pipefail.
    # Per the packet compiler hermetic-runtime gate: no uv/pip/network
    # dependency resolution at inflate time; the runtime must use the contest
    # environment's already-installed interpreter and packaged sources.
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Phase 1 contest-compliant inflate (operator decision B 2026-05-09).\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list (newline-separated\n"
        "# video file basenames; one .raw output per name at camera resolution).\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec \"${PYTHON:-python3}\" "
        "\"$HERE/inflate.py\" \"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh)
    (submission_dir / "inflate.sh").chmod(0o755)

    # Per Q3+Q6: per-video loop, camera-resolution uint8 RGB output, no scorer
    # code imports. Per Q1: deserialise length-prefixed binary then call
    # balle.decompress(strings_list). Per Q2: read 3 ZIP members.
    # Aspirational ≤100 LOC per HNeRV parity discipline lesson 4.
    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"Phase 1 contest-compliant inflate runtime.\n"
        "\n"
        "Reads archive_dir (members: x, decoder.bin, balle.bin), reconstructs the\n"
        "Balle decompressor + 128K decoder once, loops over file_list, writes one\n"
        "<base>.raw per name at camera-resolution uint8 RGB. No scorer imports.\n"
        "\"\"\"\n"
        "import io, struct, sys, zipfile\n"
        "from pathlib import Path\n"
        "import brotli\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from model import Decoder128KRuntime, BalleRuntime\n"
        "CAMERA_H, CAMERA_W = 874, 1164\n"
        "EVAL_H, EVAL_W = 384, 512\n"
        "\n"
        "def _deserialise_strings(payload: bytes):\n"
        "    off = 0\n"
        "    def _read_byte_list(off):\n"
        "        n = struct.unpack_from('<I', payload, off)[0]; off += 4\n"
        "        items = []\n"
        "        for _ in range(n):\n"
        "            blen = struct.unpack_from('<I', payload, off)[0]; off += 4\n"
        "            items.append(payload[off:off + blen]); off += blen\n"
        "        return items, off\n"
        "    y_strings, off = _read_byte_list(off)\n"
        "    z_strings, off = _read_byte_list(off)\n"
        "    n_shape = struct.unpack_from('<I', payload, off)[0]; off += 4\n"
        "    z_shape = []\n"
        "    for _ in range(n_shape):\n"
        "        z_shape.append(struct.unpack_from('<i', payload, off)[0]); off += 4\n"
        "    return {'y_strings': y_strings, 'z_strings': z_strings, 'z_shape': z_shape}\n"
        "\n"
        "def _read_member(archive_dir: Path, name: str) -> bytes:\n"
        "    archive_zip = archive_dir / 'archive.zip'\n"
        "    if archive_zip.is_file():\n"
        "        with zipfile.ZipFile(archive_zip, 'r') as zf:\n"
        "            return zf.read(name)\n"
        "    member = archive_dir / name\n"
        "    if not member.is_file():\n"
        "        raise FileNotFoundError(f'missing archive member: {member}')\n"
        "    return member.read_bytes()\n"
        "\n"
        "def _load_models(archive_dir: Path, device):\n"
        "    x_bytes = _read_member(archive_dir, 'x')\n"
        "    dec_bytes = _read_member(archive_dir, 'decoder.bin')\n"
        "    balle_bytes = _read_member(archive_dir, 'balle.bin')\n"
        "    strings = _deserialise_strings(brotli.decompress(x_bytes))\n"
        "    decoder_sd = torch.load(io.BytesIO(brotli.decompress(dec_bytes)),\n"
        "                            map_location=device, weights_only=True)\n"
        "    balle_sd = torch.load(io.BytesIO(brotli.decompress(balle_bytes)),\n"
        "                          map_location=device, weights_only=False)\n"
        "    balle = BalleRuntime().to(device)\n"
        "    balle.load_state_dict(balle_sd)\n"
        "    balle.eval(); balle.update(force=True)\n"
        "    decoder = Decoder128KRuntime().to(device)\n"
        "    decoder.load_state_dict(decoder_sd)\n"
        "    decoder.eval()\n"
        "    return strings, decoder, balle\n"
        "\n"
        "def _decode_to_raw(decoder, balle, strings, dst: Path) -> int:\n"
        "    device = next(decoder.parameters()).device\n"
        "    with torch.no_grad():\n"
        "        latents = balle.decompress(strings).to(device)\n"
        "    n = 0\n"
        "    with torch.no_grad(), open(dst, 'wb') as fout:\n"
        "        for i in range(0, latents.shape[0], 16):\n"
        "            j = min(i + 16, latents.shape[0])\n"
        "            decoded = decoder(latents[i:j])\n"
        "            up = F.interpolate(decoded.reshape(-1, 3, EVAL_H, EVAL_W),\n"
        "                               size=(CAMERA_H, CAMERA_W),\n"
        "                               mode='bicubic', align_corners=False)\n"
        "            up = up.clamp(0, 255).round().to(torch.uint8)\n"
        "            up = up.permute(0, 2, 3, 1).contiguous().cpu().numpy()\n"
        "            fout.write(up.tobytes())\n"
        "            n += up.shape[0]\n"
        "    return n\n"
        "\n"
        "def main():\n"
        "    if len(sys.argv) != 4:\n"
        "        sys.exit('Usage: inflate.py <archive_dir> <output_dir> <file_list>')\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list = Path(sys.argv[3])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "    strings, decoder, balle = _load_models(archive_dir, device)\n"
        "    for line in file_list.read_text().splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        dst = output_dir / f'{base}.raw'\n"
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "        n = _decode_to_raw(decoder, balle, strings, dst)\n"
        "        print(f'saved {n} frames -> {dst}')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py)

    # codec.py: thin shim explaining the wire format. The deserialiser lives
    # inside inflate.py (per Q4 ≤100 LOC inflate budget); codec.py is here
    # for documentation + future Phase 2 evolution.
    codec_py = (
        "\"\"\"Phase 1 wire-format documentation (deserialiser lives in inflate.py).\n"
        "\n"
        "Archive layout (3 ZIP members):\n"
        "    'x'           : brotli(serialise(balle.compress(latents)))\n"
        "                    where serialise = <I:n_outer>[<I:n_inner>[<I:blen><bytes>]]\n"
        "    'decoder.bin' : brotli(torch.save(decoder_state_dict))\n"
        "    'balle.bin'   : brotli(torch.save(balle_state_dict))\n"
        "\n"
        "Phase 2 will replace the brotli-of-torch-save with a deterministic\n"
        "FP4/INT8 quantised wire format + custom CDF tables.\n"
        "\"\"\"\n"
    )
    (src / "codec.py").write_text(codec_py)

    # The runtime model file embeds the same Decoder128K + Balle configs the
    # trainer used so the runtime can re-instantiate identically. It imports
    # only from the packet-local `src/tac` subset emitted above, never from the
    # operator checkout.
    model_py = (
        "from tac.paradigm_delta_epsilon_zeta.decoder_128k import Decoder128K, Decoder128KConfig\n"
        "from tac.paradigm_delta_epsilon_zeta.balle_hyperprior import BalleHyperpriorWrapper, BalleHyperpriorConfig\n"
        f"_DEC_CFG = Decoder128KConfig(latent_dim={decoder_config.latent_dim}, base_channels={decoder_config.base_channels})\n"
        f"_BALLE_CFG = BalleHyperpriorConfig(y_channels={balle_config.y_channels}, z_channels={balle_config.z_channels}, hyper_hidden={balle_config.hyper_hidden})\n"
        "class Decoder128KRuntime(Decoder128K):\n"
        "    def __init__(self):\n"
        "        super().__init__(_DEC_CFG)\n"
        "class BalleRuntime(BalleHyperpriorWrapper):\n"
        "    def __init__(self):\n"
        "        super().__init__(_BALLE_CFG)\n"
    )
    (src / "model.py").write_text(model_py)


def write_provenance(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    encoder_provenance: dict | None,
    n_decoder_params: int,
    n_balle_params: int,
    started_at_utc: str,
    completed_at_utc: str | None,
    t13_bit_reallocation: dict | None = None,
    t19_adaptive_rho: dict | None = None,
    t20_kl_pose_distill: dict | None = None,
    t22_temporal_consistency: dict | None = None,
    score_domain_gradcheck: dict[str, float] | None = None,
    pr95_trainer_parity: dict | None = None,
) -> Path:
    p = {
        "schema_version": 1,
        "tool": "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
        "scaffold_version": PARADIGM_DELTA_EPS_ZETA_VERSION,
        "scaffold_provenance": PARADIGM_DELTA_EPS_ZETA_PROVENANCE,
        # Per CLAUDE.md HNeRV parity discipline lesson #7: this trainer is
        # a substrate-engineering lane, not a representation lane. Tagged
        # explicitly so downstream lane-registry / preflight gates know to
        # apply substrate-engineering review (not bolt-on review).
        "lane_class": "substrate_engineering",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "args": vars(args),
        "encoder_provenance": encoder_provenance,
        "decoder_param_count": n_decoder_params,
        "balle_param_count": n_balle_params,
        "score_band": (
            "[predicted; Phase 1 scorer-domain training; not yet empirical]"
            if getattr(args, "enable_scorer_domain_loss", False)
            else "[predicted; Phase 1 scaffold; not yet empirical]"
        ),
        "score_domain_loss": {
            "enabled": bool(getattr(args, "enable_scorer_domain_loss", False)),
            "objective": getattr(args, "score_domain_objective", None),
            "max_stable_train_loss_abs": getattr(
                args,
                "max_stable_train_loss_abs",
                None,
            ),
            "segmentation_surrogate": getattr(args, "segmentation_surrogate", None),
            "segmentation_temperature": getattr(args, "segmentation_temperature", None),
            "fisher_rao_eps": getattr(args, "fisher_rao_eps", None),
            "sinkhorn_max_positions_per_chunk": getattr(
                args,
                "sinkhorn_max_positions_per_chunk",
                None,
            ),
            "pixel_l1_anchor_weight": getattr(args, "pixel_l1_anchor_weight", None),
            "first_batch_gradcheck": score_domain_gradcheck,
            "trainable_signal": (
                (
                    "PoseNet/SegNet tensor losses optimize the direct "
                    "contest-score surrogate"
                )
                if getattr(args, "score_domain_objective", None)
                == SCORE_DOMAIN_OBJECTIVE_DIRECT
                else (
                    "PoseNet/SegNet tensor losses are wired into the ADMM "
                    "constraint residuals"
                )
                if getattr(args, "enable_scorer_domain_loss", False)
                else "pixel-L1 scaffold; seg/pose residuals are constants"
            ),
        },
        "pr95_hnerv_trainer_parity": pr95_trainer_parity,
        # T13 / T19 wire-in surfaces (memo
        # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509). Both
        # default to None when the corresponding --enable flag is OFF
        # (backward-compat preserved).
        "t13_bit_reallocation": t13_bit_reallocation,
        "t19_adaptive_rho": t19_adaptive_rho,
        # T20 / T22 wire-in surfaces (memo
        # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509).
        # Both default to None when the corresponding --enable flag is OFF
        # (backward-compat preserved per T13/T19 pattern).
        "t20_kl_pose_distill": t20_kl_pose_distill,
        "t22_temporal_consistency": t22_temporal_consistency,
        "platform": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    path = output_dir / "provenance.json"
    path.write_text(json.dumps(p, indent=2, default=str))
    return path


def maybe_run_auth_eval(
    *,
    archive_path: Path,
    submission_dir: Path,
    output_dir: Path,
    enabled: bool,
    dispatch_lane_id: str | None,
    dispatch_claims_path: Path,
) -> dict | None:
    if not enabled:
        return None
    if PHASE1_SCAFFOLD_ONLY:
        refuse_phase1_scaffold_path("--auth-eval")
    require_active_dispatch_claim(
        lane_id=dispatch_lane_id,
        claims_path=dispatch_claims_path,
    )
    auth_script = REPO_ROOT / CONTEST_AUTH_EVAL_RELATIVE
    if not auth_script.exists():
        raise SystemExit(f"[t1] auth eval failed: {auth_script} not found")
    work_dir = output_dir / "auth_eval_work"
    work_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        str(auth_script),
        "--archive", str(archive_path),
        "--inflate-sh", str(submission_dir / "inflate.sh"),
        "--upstream-dir", str(REPO_ROOT / "upstream"),
        "--device", "cuda",
        "--work-dir", str(work_dir),
        "--json-out", str(output_dir / "contest_auth_eval.json"),
        "--keep-work-dir",
    ]
    print(f"[t1] dispatching auth eval: {' '.join(cmd)}")
    rc = subprocess.run(cmd, check=False).returncode
    auth_json = output_dir / "contest_auth_eval.json"
    if rc != 0:
        raise SystemExit(f"[t1] auth eval failed rc={rc}; see {work_dir}")
    if not auth_json.exists():
        raise SystemExit(f"[t1] auth eval reported success but did not write {auth_json}")
    return {"returncode": rc, "auth_json_path": str(auth_json)}


def apply_t13_sqrt_n_budget(
    *,
    n_pairs: int,
    n_symbols_per_pair: int,
    current_bits_per_pair: float,
    rate_target_bytes: float,
    alpha: float = 1.0,
) -> dict:
    """Compute the T13 Fridrich √n latent reallocation envelope.

    Returns a dict suitable for emission into the run manifest. The dict
    carries:

    * ``bit_reallocation_t13_applied`` — always True (caller only invokes
      this when ``--enable-t13-sqrt-n-budget`` is set).
    * ``per_pair_undetectable_bits`` — closed-form Fridrich bound at α.
    * ``per_pair_current_bits`` — caller-supplied current rate.
    * ``per_pair_headroom_bits`` — undetectable - current. May be negative
      (current spends MORE than undetectable; reallocation is forbidden in
      that direction since the latent is already detectable; we clip to 0).
    * ``latent_bits_reduced`` / ``pose_bits_added`` — how the headroom was
      reallocated. Headroom is reallocated 1:1 from latent to pose stream
      per memo §6 lesson 6 ("re-allocate per-pair latent → per-pair pose
      budget per Fridrich √n bound").
    * ``rate_target_bytes_before`` / ``rate_target_bytes_after`` — only
      the LATENT portion of the rate target shrinks; pose is consumed by
      the same coordinator constraint set so we DO NOT add bytes back.
      The trainer's rate target is the ARCHIVE-level target; T13 shrinks
      the per-pair latent contribution and the saved bytes are consumed
      elsewhere downstream.
    * ``predicted_pose_distortion_decrease`` — qualitative tag (we cannot
      predict a numeric Δ without empirical anchor; per CLAUDE.md
      Forbidden Score Claims).
    * ``alpha`` / ``n_pairs`` / ``n_symbols_per_pair`` / ``notes`` —
      provenance.

    Per CLAUDE.md ``forbidden_dead_flag_wiring_pattern`` and the memo's
    integration log, this helper is callable standalone (covered by tests)
    and the trainer invokes it once at startup. The returned dict is
    written into ``provenance.json`` as ``t13_bit_reallocation``.

    Tagging: every numeric impact is ``[predicted; T13 Fridrich sqrt-n
    latent shrink]`` per CLAUDE.md Forbidden Score Claims.
    """
    from tac.joint_source_rd_bound import per_pair_sqrt_n_budget  # noqa: WPS433

    if n_symbols_per_pair <= 0:
        raise SystemExit(
            f"[t1] T13 requires positive n_symbols_per_pair; got "
            f"{n_symbols_per_pair!r}"
        )
    if current_bits_per_pair < 0:
        raise SystemExit(
            f"[t1] T13 requires non-negative current_bits_per_pair; got "
            f"{current_bits_per_pair!r}"
        )
    if rate_target_bytes <= 0:
        raise SystemExit(
            f"[t1] T13 requires positive rate_target_bytes; got "
            f"{rate_target_bytes!r}"
        )
    report = per_pair_sqrt_n_budget(
        n_pairs=n_pairs,
        n_symbols_per_pair=n_symbols_per_pair,
        alpha=alpha,
    )
    headroom_bits = report.undetectable_bits_per_pair - current_bits_per_pair
    # Reallocation is bounded: only positive headroom is realloc-able.
    # If the current per-pair rate already exceeds the undetectable budget,
    # we surface the negative gap as a warning but do not over-shrink the
    # rate target.
    realloc_per_pair_bits = max(0.0, headroom_bits)
    realloc_total_bits = realloc_per_pair_bits * n_pairs
    realloc_total_bytes = realloc_total_bits / 8.0
    rate_target_after = max(1.0, rate_target_bytes - realloc_total_bytes)
    return {
        "bit_reallocation_t13_applied": True,
        "per_pair_undetectable_bits": float(report.undetectable_bits_per_pair),
        "per_pair_current_bits": float(current_bits_per_pair),
        "per_pair_headroom_bits": float(headroom_bits),
        "latent_bits_reduced": float(realloc_total_bits),
        "pose_bits_added": float(realloc_total_bits),
        "rate_target_bytes_before": float(rate_target_bytes),
        "rate_target_bytes_after": float(rate_target_after),
        "predicted_pose_distortion_decrease": (
            "[predicted; T13 Fridrich sqrt-n latent shrink; "
            "qualitative direction only — empirical anchor required]"
        ),
        "alpha": float(alpha),
        "n_pairs": int(n_pairs),
        "n_symbols_per_pair": int(n_symbols_per_pair),
        "notes": list(report.notes),
        "tag": "[predicted; T13 Fridrich sqrt-n latent shrink]",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--epochs", type=int, default=3000)
    # Per engineering audit 2026-05-12: T4 has 40 SMs × 64 cores; batch_size=16
    # under-utilizes SM count for scorer forwards (SegNet EfficientNet-B2 amortizes
    # over batch>=32 for its BN+conv kernels). Default raised to 32 for better
    # SM utilization. Reduce manually with --batch-size 16 if T4 VRAM-bound at AMP+1080p.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--rate-target-bytes", type=float, default=80_000.0)
    parser.add_argument("--seg-target", type=float, default=7e-4)
    parser.add_argument("--pose-target", type=float, default=1.7e-4)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument(
        "--enable-scorer-domain-loss",
        action="store_true",
        default=False,
        help=(
            "Train the Phase 1 decoder/hyperprior against differentiable "
            "PoseNet/SegNet scorer-domain terms instead of the historical "
            "pixel-L1 proxy. This removes the scaffold-only constant seg/pose "
            "placeholders and wires the ADMM residuals to real scorer outputs."
        ),
    )
    parser.add_argument(
        "--segmentation-surrogate",
        choices=[
            SEGMENTATION_SURROGATE_SOFT_COSINE,
            SEGMENTATION_SURROGATE_FISHER_RAO,
            SEGMENTATION_SURROGATE_SINKHORN,
        ],
        default=DEFAULT_SEGMENTATION_SURROGATE,
        help=(
            "SegNet differentiable surrogate used when "
            "--enable-scorer-domain-loss is set. Default sinkhorn follows the "
            "T7/T8/T11 disambiguator verdict and the PR95/T1 parity contract; "
            "soft_cosine remains available as an explicit speed override."
        ),
    )
    parser.add_argument(
        "--segmentation-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for scorer-domain SegNet surrogate.",
    )
    parser.add_argument(
        "--fisher-rao-eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon used by the fisher_rao SegNet surrogate.",
    )
    parser.add_argument(
        "--sinkhorn-max-positions-per-chunk",
        type=int,
        default=2048,
        help=(
            "Maximum flattened SegNet rows per T8 Sinkhorn chunk in "
            "score-domain training. Keep small for T4 guard dispatches; "
            "raise only after memory telemetry proves headroom."
        ),
    )
    parser.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        default=False,
        help=(
            "Wrap heavy forward block (balle + decoder + scorer surrogates) in "
            "torch.autocast('cuda', dtype=torch.float16) and use GradScaler for "
            "backward. T4 FP16 throughput is 8× FP32 (65 TFLOPS vs 8 TFLOPS); "
            "A100 FP16 throughput is 5× FP32. Losses are explicitly cast back "
            "to FP32 before the Lagrangian step to avoid dual-update fp16 "
            "overflow. Per engineering audit 2026-05-12 'exploit hardware'. "
            "OFF by default for backward-compat; enable for new dispatches."
        ),
    )
    parser.add_argument(
        "--enable-mp4-codec-sim",
        action="store_true",
        default=False,
        help=(
            "Wire apply_mp4_codec_simulation_during_training (BT.601 chroma 4:2:0 "
            "+ optional DCT-quant noise STE) into the per-batch decoded_rt chain "
            "AFTER apply_eval_roundtrip_during_training. Closes the "
            "pixels→bytes→pixels fidelity gap: the eval roundtrip simulates "
            "uint8 quant but NOT mp4 chroma 4:2:0 / DCT-quant losses. Predicted "
            "proxy-auth gap closure ~0.5-2% PoseNet at PR106 r2. OFF by default "
            "for backward-compat; enable when training a new substrate where "
            "the eval pipeline includes mp4 reencoding."
        ),
    )
    parser.add_argument(
        "--mp4-codec-sim-noise-std",
        type=float,
        default=0.0,
        help=(
            "Stddev (in uint8 units) of per-block Gaussian DCT-quant noise added "
            "to chroma during mp4 codec sim. Default 0 (chroma 4:2:0 only; no "
            "block-level noise). 5-10 simulates CRF 23-30 mp4 quality."
        ),
    )
    parser.add_argument(
        "--pixel-l1-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary pixel-L1 anchor added to scorer-domain loss. "
            "Default 0.0 so score-domain terms are primary."
        ),
    )
    parser.add_argument(
        "--score-domain-objective",
        choices=SCORE_DOMAIN_OBJECTIVE_CHOICES,
        default=SCORE_DOMAIN_OBJECTIVE_DIRECT,
        help=(
            "Objective used when --enable-scorer-domain-loss is set. "
            "direct_score optimizes the differentiable contest-score surrogate "
            "plus optional T20/T22/anchor terms and is the stable warm-start "
            "default. augmented_lagrangian routes scorer residuals through the "
            "JointLagrangianADMM/T19 constraint coordinator and is opt-in after "
            "a direct-score basin exists."
        ),
    )
    parser.add_argument(
        "--max-stable-train-loss-abs",
        type=float,
        default=1e9,
        help=(
            "Fail closed if abs(training loss) exceeds this finite threshold. "
            "Prevents remote GPU jobs from burning hours after an ADMM/loss-scale "
            "configuration error. Set <=0 only for a documented diagnostic run."
        ),
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help=(
            "Optional global gradient norm clip after backward. 0 disables. "
            "Recommended for scorer-domain CUDA runs because early random "
            "decoder residuals can be large."
        ),
    )
    parser.add_argument("--eval-every-epochs", type=int, default=100)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "EMA proxy evaluation chunk size. Defaults to --batch-size. Keep at "
            "1 for full 600-pair T4 score-domain runs; unchunked eval can OOM "
            "before archive export."
        ),
    )
    parser.add_argument(
        "--archive-in-loop",
        action="store_true",
        default=False,
        help=(
            "Materialise byte-closed EMA archive candidates during the eval loop. "
            "Rows are written to archive_builds_manifest.json with score_claim=false; "
            "remote Stage 6 decides which eligible packets receive exact CUDA eval."
        ),
    )
    parser.add_argument(
        "--archive-every-epochs",
        type=int,
        default=100,
        help=(
            "When --archive-in-loop is enabled, build a candidate every N epochs "
            "in addition to every new-best proxy checkpoint and final epoch."
        ),
    )
    parser.add_argument(
        "--max-candidate-archive-bytes",
        type=int,
        default=190_000,
        help=(
            "Archive-in-loop rate cap. Candidates above this byte count are "
            "recorded but not packet-compiled or exact-CUDA eligible."
        ),
    )
    parser.add_argument(
        "--max-exact-cuda-candidates",
        type=int,
        default=1,
        help=(
            "Candidate-count budget surfaced to remote Stage 6. The trainer "
            "records it in the manifest; exact CUDA dispatch remains external."
        ),
    )
    parser.add_argument(
        "--baseline-archive-sha256",
        default=A1_BASELINE_ARCHIVE_SHA256,
        help="Baseline archive SHA-256 for archive-in-loop no-op proof.",
    )
    parser.add_argument(
        "--baseline-archive-size-bytes",
        type=int,
        default=A1_BASELINE_ARCHIVE_SIZE_BYTES,
        help="Baseline archive size for archive-in-loop no-op proof.",
    )
    parser.add_argument("--auth-eval", action="store_true", default=False)
    parser.add_argument("--no-auth-eval", dest="auth_eval", action="store_false")
    parser.add_argument(
        "--no-auth-eval-on-best",
        dest="auth_eval",
        action="store_false",
        help=(
            "Explicit Phase 1 auth-eval opt-out. The Phase 1 runtime is "
            "contest-contract-shaped, but promotion still requires dispatch "
            "claim custody plus exact CUDA/CPU evidence."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="1 epoch, 1 pair build verification")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Real contest video used for non-smoke target pixels.",
    )
    parser.add_argument(
        "--target-pixels-path",
        type=Path,
        default=None,
        help=(
            "Optional torch tensor containing real target pixels with shape "
            "(N, 2, 3, 384, 512) or (N, 2, 384, 512, 3). Overrides --video-path."
        ),
    )
    parser.add_argument(
        "--max-target-pairs",
        type=int,
        default=None,
        help="Optional cap on real frame pairs for local non-smoke debugging.",
    )
    parser.add_argument(
        "--canonical-a1-relpath",
        default="experiments/results/A1_canonical",
        help="Path under repo root to the canonical A1 symlink",
    )
    parser.add_argument(
        "--allow-missing-canonical-a1",
        action="store_true",
        help="Skip frozen-A1 load (smoke mode only)",
    )
    parser.add_argument(
        "--dispatch-lane-id",
        default=None,
        help="Required with --auth-eval; must have an active dispatch claim.",
    )
    parser.add_argument(
        "--dispatch-claims-path",
        type=Path,
        default=REPO_ROOT / DISPATCH_CLAIMS_RELATIVE,
        help="Dispatch-claim ledger checked before --auth-eval.",
    )
    parser.add_argument(
        "--pr95-parity-profile",
        type=Path,
        default=DEFAULT_PR95_PARITY_PROFILE,
        help=(
            "PR95 HNeRV/Muon intake profile JSON emitted by "
            "experiments/profile_pr95_hnerv_muon_intake.py. Non-smoke "
            "score-domain runs record its eight-stage release-view manifest "
            "as parity/preflight evidence."
        ),
    )
    # T13 — Fridrich √n per-pair latent budget (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t13-sqrt-n-budget",
        action="store_true",
        default=False,
        help="Opt-in T13 Fridrich sqrt(n) latent-budget hook. Default OFF for "
             "backward-compat. When ON, the trainer queries "
             "tac.joint_source_rd_bound.per_pair_sqrt_n_budget and shrinks "
             "rate_target_bytes by the reallocation envelope.",
    )
    parser.add_argument(
        "--t13-alpha",
        type=float,
        default=1.0,
        help="Fridrich proportionality constant (Ker-Pevný-Fridrich 2008). "
             "Default 1.0; values in [0.5, 2.0] cover the literature.",
    )
    parser.add_argument(
        "--t13-current-bits-per-pair",
        type=float,
        default=3.0,
        help="Caller-supplied estimate of the trainer's current per-pair "
             "latent rate (bits/pair). Default 3.0 per A1 substrate empirical "
             "anchor (memo §6).",
    )
    # T19 — adaptive ρ ADMM (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t19-adaptive-rho",
        action="store_true",
        default=False,
        help="Opt-in T19 Boyd §3.4.1 / He-Yang 2000 adaptive-ρ update via "
             "tac.joint_admm_coordinator.adaptive_rho_step. Default OFF for "
             "backward-compat per coherence council recommendation.",
    )
    parser.add_argument(
        "--t19-tau-grow",
        type=float,
        default=2.0,
        help="T19 ρ-growth factor (Boyd canonical 2.0). Must be > 1.",
    )
    parser.add_argument(
        "--t19-tau-shrink",
        type=float,
        default=0.5,
        help="T19 ρ-shrink factor (Boyd canonical 0.5). Must be in (0, 1).",
    )
    # T20 — KL pose-axis distillation loss (memo
    # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t20-kl-pose-distill",
        action="store_true",
        default=False,
        help="Opt-in T20 Hinton-Vinyals-Dean 2014 KL distillation on the "
             "PoseNet 12-dim head logits at T=2.0 (matches Quantizr's "
             "verified 0.33 archive recipe). Student = PoseNet(decoded_rt); "
             "teacher = PoseNet(target). Requires --enable-scorer-domain-loss. "
             "Default OFF for backward-compat per T13/T19 pattern.",
    )
    parser.add_argument(
        "--t20-temperature",
        type=float,
        default=KL_POSE_DEFAULT_TEMPERATURE,
        help="T20 Hinton softmax temperature (default 2.0; Quantizr canonical).",
    )
    parser.add_argument(
        "--t20-weight-pose",
        type=float,
        default=1.0,
        help="T20 multiplier added to the augmented Lagrangian. Default 1.0 "
             "(per T20 landing's `KLPoseDistillConfig.weight_pose` default).",
    )
    parser.add_argument(
        "--t20-mode",
        choices=["distill_softmax_full", "distill_softmax_first6", "regression_mse"],
        default="distill_softmax_full",
        help="T20 loss form: distill_softmax_full (canonical Hinton, default), "
             "distill_softmax_first6 (softmax restricted to scored region), or "
             "regression_mse (ablation baseline).",
    )
    # T22 — Temporal-consistency regularizer (memo
    # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509)
    parser.add_argument(
        "--enable-t22-temporal-consistency",
        action="store_true",
        default=False,
        help="Opt-in T22 Horn-Schunck 1981 brightness-constancy temporal "
             "regularizer on per-pair rendered frames (B, P=2, C, H, W). "
             "Identity-warp form (no flow): "
             "λ · mean(|R_{t+1} - R_t|²). Default OFF for backward-compat.",
    )
    parser.add_argument(
        "--t22-lambda-weight",
        type=float,
        default=TC_DEFAULT_LAMBDA_WEIGHT,
        help="T22 λ multiplier (default 0.1; T20 landing recommends 0.05-0.5).",
    )
    parser.add_argument(
        "--t22-boundary-handling",
        choices=["zeros", "border", "reflection"],
        default=TC_DEFAULT_BOUNDARY_HANDLING,
        help="T22 grid_sample padding_mode (default border; matches renderer "
             "convention per T22 landing's commit-pinned coordinate convention).",
    )
    # ----------------------------------------------------------------------
    # PR #95 binary-forensics replication (Finding A + Finding B)
    # See `src/tac/differentiable_eval_roundtrip.py`.
    # ----------------------------------------------------------------------
    parser.add_argument(
        "--enable-eval-roundtrip-in-training",
        dest="enable_eval_roundtrip_in_training",
        action="store_true",
        default=True,
        help=(
            "[default: True] Apply contest eval roundtrip "
            "(bicubic-up + bilinear-down + STE-round) inside training "
            "inner loop through apply_eval_roundtrip_during_training, "
            "so the proxy gradient matches contest-eval. Per CLAUDE.md "
            "eval_roundtrip non-negotiable."
        ),
    )
    parser.add_argument(
        "--disable-eval-roundtrip-in-training",
        dest="enable_eval_roundtrip_in_training",
        action="store_false",
        help="Ablation only — see --enable-eval-roundtrip-in-training.",
    )
    parser.add_argument(
        "--enable-differentiable-yuv6",
        dest="enable_differentiable_yuv6",
        action="store_true",
        default=True,
        help=(
            "[default: True] Apply autograd-preserving rgb_to_yuv6 "
            "monkey-patch (PR #95 Finding B). Without it, PoseNet "
            "gradients are zero through YUV6 op (pose plateau)."
        ),
    )
    parser.add_argument(
        "--disable-differentiable-yuv6",
        dest="enable_differentiable_yuv6",
        action="store_false",
        help="Ablation only — see --enable-differentiable-yuv6.",
    )
    parser.add_argument(
        "--yuv6-mode",
        choices=[m.value for m in Yuv6RoutingMode],
        default=Yuv6RoutingMode.AUTO.value,
        help=(
            "Which differentiable-yuv6 routing to use: "
            "'monkey_patch_global' (PR #95 verified-working recipe), "
            "'tac_differentiable_routing' (cleaner, requires per-call "
            "routing), 'auto' (default; runs probe-disambiguator). "
            "See tools/probe_yuv6_differentiability_disambiguator.py."
        ),
    )
    if argv is None:
        return parser.parse_args()
    return parser.parse_args(argv)


def _resolve_yuv6_mode_with_probe_t1(requested: str) -> Yuv6RoutingMode:
    """In-process arbitration for ``--yuv6-mode auto``."""
    mode = Yuv6RoutingMode(requested)
    if mode is not Yuv6RoutingMode.AUTO:
        return mode
    from tac.differentiable_eval_roundtrip import differentiable_rgb_to_yuv6

    rgb = (torch.rand((1, 3, 32, 32)) * 255.0).requires_grad_(True)
    out = differentiable_rgb_to_yuv6(rgb)
    out.sum().backward()
    if rgb.grad is None or rgb.grad.abs().sum().item() == 0.0:
        raise RuntimeError(
            "[t1] yuv6 auto-resolution: tac.differentiable_rgb_to_yuv6 "
            "produced zero gradient on calibration batch. CRITICAL invariant."
        )
    return Yuv6RoutingMode.MONKEY_PATCH_GLOBAL


def _activate_yuv6_mode_t1(
    mode: Yuv6RoutingMode, *, enabled: bool
) -> Yuv6PatchToken | None:
    """Activate the chosen YUV6 routing mode."""
    if not enabled:
        return None
    if mode is Yuv6RoutingMode.MONKEY_PATCH_GLOBAL:
        return patch_upstream_yuv6_globally()
    return Yuv6PatchToken(
        frame_utils_orig=None,
        modules_orig=None,
        frame_utils_was_patched=False,
        modules_was_patched=False,
    )


def main() -> int:
    args = parse_args()
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY and not args.enable_scorer_domain_loss:
        refuse_phase1_scaffold_path("non-smoke training")
    if args.allow_missing_canonical_a1 and not args.smoke:
        raise SystemExit(
            "[t1] --allow-missing-canonical-a1 is smoke-only. Non-smoke "
            "training requires the real frozen A1 latent table."
        )
    if args.max_target_pairs is not None and args.max_target_pairs <= 0:
        raise SystemExit("[t1] --max-target-pairs must be positive when set")
    if args.enable_scorer_domain_loss and not args.enable_eval_roundtrip_in_training:
        raise SystemExit(
            "[t1] --enable-scorer-domain-loss requires "
            "--enable-eval-roundtrip-in-training. Score-domain training without "
            "the PR #95 eval-roundtrip path recreates the known proxy/auth gap."
        )
    if args.enable_scorer_domain_loss and not args.enable_differentiable_yuv6:
        raise SystemExit(
            "[t1] --enable-scorer-domain-loss requires "
            "--enable-differentiable-yuv6. Disabling YUV6 differentiability in "
            "score-domain mode recreates the PR #95 PoseNet-gradient bug."
        )
    if args.enable_scorer_domain_loss and args.epochs <= 0:
        raise SystemExit(
            "[t1] --enable-scorer-domain-loss requires --epochs > 0 so the "
            "first-batch score-domain gradient reachability guard can run."
        )
    if (
        args.enable_scorer_domain_loss
        and args.enable_t19_adaptive_rho
        and args.score_domain_objective != SCORE_DOMAIN_OBJECTIVE_AUGMENTED
    ):
        raise SystemExit(
            "[t1] --enable-t19-adaptive-rho is only wired when "
            "--score-domain-objective augmented_lagrangian is selected. "
            "The stable default direct_score objective intentionally bypasses "
            "the ADMM/T19 coordinator until a direct-score warm-start basin "
            "exists."
        )
    pr95_trainer_parity = load_pr95_trainer_parity_contract(args.pr95_parity_profile)
    if (
        args.enable_scorer_domain_loss
        and not args.smoke
        and not pr95_trainer_parity["local_trainer_parity_preflight_passed"]
    ):
        raise SystemExit(
            "[t1] non-smoke score-domain training requires a valid "
            "--pr95-parity-profile emitted by "
            "experiments/profile_pr95_hnerv_muon_intake.py; "
            f"status={pr95_trainer_parity['status']} "
            f"blockers={pr95_trainer_parity['score_bearing_dispatch_blockers']}"
        )
    if args.pixel_l1_anchor_weight < 0:
        raise SystemExit("[t1] --pixel-l1-anchor-weight must be non-negative")
    if args.grad_clip_norm < 0:
        raise SystemExit("[t1] --grad-clip-norm must be non-negative")
    if not math.isfinite(float(args.max_stable_train_loss_abs)):
        raise SystemExit("[t1] --max-stable-train-loss-abs must be finite")
    if args.segmentation_temperature <= 0:
        raise SystemExit("[t1] --segmentation-temperature must be positive")
    if not (0.0 < args.fisher_rao_eps < 1e-3):
        raise SystemExit("[t1] --fisher-rao-eps must be in (0, 1e-3)")
    if args.sinkhorn_max_positions_per_chunk <= 0:
        raise SystemExit("[t1] --sinkhorn-max-positions-per-chunk must be positive")
    if args.archive_in_loop:
        if args.archive_every_epochs <= 0:
            raise SystemExit("[t1] --archive-every-epochs must be positive")
        if args.max_candidate_archive_bytes <= 0:
            raise SystemExit("[t1] --max-candidate-archive-bytes must be positive")
        if args.max_exact_cuda_candidates <= 0:
            raise SystemExit("[t1] --max-exact-cuda-candidates must be positive")
        if (
            not isinstance(args.baseline_archive_sha256, str)
            or len(args.baseline_archive_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in args.baseline_archive_sha256)
        ):
            raise SystemExit("[t1] --baseline-archive-sha256 must be 64 hex chars")
        if args.baseline_archive_size_bytes <= 0:
            raise SystemExit("[t1] --baseline-archive-size-bytes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    # ------------------------------------------------------------------
    # PR #95 binary-forensics replication: activate autograd-preserving
    # rgb_to_yuv6 BEFORE any scorer is loaded. See
    # `src/tac/differentiable_eval_roundtrip.py`.
    # ------------------------------------------------------------------
    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    yuv6_token = _activate_yuv6_mode_t1(
        yuv6_mode, enabled=args.enable_differentiable_yuv6
    )
    print(
        f"[t1][pr95-replication] enable_eval_roundtrip_in_training="
        f"{args.enable_eval_roundtrip_in_training} "
        f"enable_differentiable_yuv6={args.enable_differentiable_yuv6} "
        f"yuv6_mode={yuv6_mode.value}"
    )
    pr95_provenance = {
        "enable_eval_roundtrip_in_training": args.enable_eval_roundtrip_in_training,
        "enable_differentiable_yuv6": args.enable_differentiable_yuv6,
        "yuv6_mode": yuv6_mode.value,
        "yuv6_monkey_patched": bool(
            yuv6_token
            and (yuv6_token.frame_utils_was_patched or yuv6_token.modules_was_patched)
        ),
        "evidence_grade": "[predicted; PR #95 eval_roundtrip+yuv6 monkey-patch in training inner loop]",
        "score_claim": False,
        "binary_forensics_dossier": ".omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md",
        "trainer_parity_profile": pr95_trainer_parity,
        "note": (
            "When --enable-scorer-domain-loss is set, Phase 1 forwards "
            "through PoseNet/SegNet in the training inner loop, making the "
            "YUV6 patch load-bearing immediately. Without that flag, this "
            "remains a pixel-L1 scaffold/debug run."
        ),
    }
    (args.output_dir / "pr95_replication_provenance.json").write_text(
        json.dumps(pr95_provenance, indent=2)
    )

    print(f"[t1] scaffold version: {PARADIGM_DELTA_EPS_ZETA_VERSION}")
    print(f"[t1] device: {device}; smoke: {args.smoke}; epochs: {args.epochs}")

    # Load frozen A1 encoder (or smoke-synthesise).
    encoder_provenance = None
    if args.smoke and args.allow_missing_canonical_a1:
        print("[t1] smoke + allow-missing-canonical-a1 — synthesising latents")
        latents, target_pixels = make_smoke_target(
            n_pairs=1, latent_dim=28, seed=args.seed,
        )
    else:
        encoder = load_frozen_a1_encoder(
            repo_root=REPO_ROOT,
            canonical_dir_name=_canonical_dir_name_from_relpath(args.canonical_a1_relpath),
        )
        encoder_provenance = encoder.provenance
        latents = encoder.latents
        # In smoke we keep just one pair; in full mode we use all 600.
        if args.smoke:
            latents = latents[:1].clone()
        latents.requires_grad_(False)
        if args.smoke:
            # Build-verification only; non-smoke must use real frame targets.
            # The make_smoke_target n_pairs MUST match latents shape so the
            # broadcast shapes are aligned at loss time.
            _, target_pixels = make_smoke_target(
                n_pairs=int(latents.shape[0]),
                latent_dim=int(latents.shape[1]),
                seed=args.seed,
            )
        else:
            if args.target_pixels_path is not None:
                target_pixels = load_target_pixels_from_path(args.target_pixels_path)
                if args.max_target_pairs is not None:
                    target_pixels = target_pixels[: args.max_target_pairs].clone()
                    latents = latents[: target_pixels.shape[0]].clone()
            else:
                target_pixels = load_real_target_pairs(
                    args.video_path,
                    n_pairs=int(latents.shape[0]),
                    max_pairs=args.max_target_pairs,
                )
                latents = latents[: target_pixels.shape[0]].clone()
            if int(target_pixels.shape[0]) != int(latents.shape[0]):
                raise SystemExit(
                    "[t1] real target pair count mismatch: "
                    f"targets={target_pixels.shape[0]} latents={latents.shape[0]}. "
                    "Use --max-target-pairs to trim both explicitly."
                )

    latents = latents.to(device)
    target_pixels = target_pixels.to(device)

    posenet = None
    segnet = None
    if args.enable_scorer_domain_loss:
        from tac.scorer import load_differentiable_scorers  # noqa: WPS433

        posenet, segnet = load_differentiable_scorers(
            REPO_ROOT / "upstream",
            device=device,
        )
        posenet.eval()
        segnet.eval()
        for scorer_module in (posenet, segnet):
            for param in scorer_module.parameters():
                param.requires_grad_(False)
        print(
            f"[t1] scorer-domain loss enabled: surrogate="
            f"{args.segmentation_surrogate}; pixel_l1_anchor_weight="
            f"{args.pixel_l1_anchor_weight}"
        )

    # Frozen scorer-target cache. The target path tgt → PoseNet/SegNet is
    # gradient-free and deterministic for the whole run. Precomputing it once
    # turns every hot-loop batch from two scorer passes (predicted + target)
    # into one scorer pass (predicted only), while preserving the exact target
    # tensors used by the canonical scorer-domain loss.
    gt_pose_cache: torch.Tensor | None = None
    gt_seg_cache: torch.Tensor | None = None
    gt_seg_already_probs = False
    if args.enable_scorer_domain_loss:
        with torch.no_grad():
            pose_chunks: list[torch.Tensor] = []
            seg_chunks: list[torch.Tensor] = []
            cache_chunk_size = max(1, int(getattr(args, 'batch_size', 16)))
            for cstart in range(0, target_pixels.shape[0], cache_chunk_size):
                cend = min(cstart + cache_chunk_size, target_pixels.shape[0])
                tgt_chunk = target_pixels[cstart:cend].contiguous()
                pose_out_chunk, seg_logits_chunk = scorer_forward_pair(
                    tgt_chunk,
                    posenet,
                    segnet,
                )
                pose_chunks.append(pose_out_chunk["pose"].detach())
                if float(args.segmentation_temperature) == 1.0:
                    seg_chunks.append(F.softmax(seg_logits_chunk, dim=1).detach())
                    gt_seg_already_probs = True
                else:
                    seg_chunks.append(seg_logits_chunk.detach())
                    gt_seg_already_probs = False
            pin_cache = device.type == "cuda"
            gt_pose_cache = _cache_tensor_on_cpu(
                torch.cat(pose_chunks, dim=0),
                pin_for_cuda=pin_cache,
            )
            gt_seg_cache = _cache_tensor_on_cpu(
                torch.cat(seg_chunks, dim=0),
                pin_for_cuda=pin_cache,
            )
            cache_n = gt_pose_cache.shape[0]
            cache_bytes = (
                gt_pose_cache.element_size() * gt_pose_cache.numel()
                + gt_seg_cache.element_size() * gt_seg_cache.numel()
            )
            print(
                f"[t1] scorer_target_cache built: {cache_n} pairs; "
                f"pose_shape={tuple(gt_pose_cache.shape[1:])}; "
                f"seg_shape={tuple(gt_seg_cache.shape[1:])}; "
                f"seg_cache={'probs' if gt_seg_already_probs else 'logits'}; "
                f"{cache_bytes / (1024 * 1024):.1f}MB CPU"
                f"{' pinned' if gt_pose_cache.is_pinned() else ''}; "
                "saves one frozen PoseNet+SegNet target forward per batch"
            )

    # Build modules.
    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    n_decoder_params = sum(p.numel() for p in decoder.parameters())
    n_balle_params = sum(p.numel() for p in balle.parameters())
    print(f"[t1] decoder params: {n_decoder_params:,}; balle params: {n_balle_params:,}")

    # Optimisers (CompressAI requires SEPARATE aux optimiser).
    decoder_trainable_params = [p for p in decoder.parameters() if p.requires_grad]
    balle_main_trainable_params = [
        p for n, p in balle.named_parameters()
        if p.requires_grad and ("entropy_bottleneck" not in n or "quantiles" not in n)
    ]
    main_params = [
        *decoder_trainable_params,
        *balle_main_trainable_params,
    ]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(
        [p for p in main_params if p.requires_grad], lr=args.learning_rate
    )
    optim_aux = torch.optim.Adam(
        [p for p in aux_params if p.requires_grad], lr=args.aux_learning_rate
    ) if aux_params else None

    # EMA shadows (decay 0.997 per CLAUDE.md non-negotiable).
    ema_decoder = EMA(decoder, decay=args.ema_decay)
    ema_balle = EMA(balle, decay=args.ema_decay)

    # T13 — Fridrich √n per-pair latent budget hook (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509). When OFF,
    # the trainer behaves identically to its pre-T13 form (backward-compat).
    rate_target_bytes_effective = float(args.rate_target_bytes)
    t13_report: dict | None = None
    if args.enable_t13_sqrt_n_budget:
        t13_report = apply_t13_sqrt_n_budget(
            n_pairs=int(latents.shape[0]),
            n_symbols_per_pair=int(latents.shape[1]),
            current_bits_per_pair=float(args.t13_current_bits_per_pair),
            rate_target_bytes=float(args.rate_target_bytes),
            alpha=float(args.t13_alpha),
        )
        rate_target_bytes_effective = t13_report["rate_target_bytes_after"]
        print(
            f"[t1] T13 enabled: per-pair undetectable bits = "
            f"{t13_report['per_pair_undetectable_bits']:.3f}; "
            f"per-pair current = {t13_report['per_pair_current_bits']:.3f}; "
            f"headroom = {t13_report['per_pair_headroom_bits']:.3f}; "
            f"rate_target_bytes "
            f"{t13_report['rate_target_bytes_before']:.0f} -> "
            f"{t13_report['rate_target_bytes_after']:.0f} "
            f"[predicted; T13 Fridrich sqrt-n latent shrink]"
        )

    # Joint Lagrangian-ADMM coordinator. T19 (memo
    # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509) routes the
    # adaptive-ρ rule through tac.joint_admm_coordinator.adaptive_rho_step
    # when --enable-t19-adaptive-rho is set; otherwise the legacy windowed-
    # average grow/shrink rule is preserved (backward-compat per coherence
    # council recommendation).
    coord_kwargs: dict = {
        "rate_target_bytes": rate_target_bytes_effective,
        "seg_target": args.seg_target,
        "pose_target": args.pose_target,
        "rho_init": args.rho_init,
        "use_t19_adaptive_rho": bool(args.enable_t19_adaptive_rho),
        "t19_tau_grow": float(args.t19_tau_grow),
        "t19_tau_shrink": float(args.t19_tau_shrink),
    }
    if args.enable_t19_adaptive_rho:
        # The brief specifies rho_min=1e-3, rho_max=1e3 for T19 per memo
        # feedback_t11_t13_t19_free_lateral_leaps_landed_20260509.
        coord_kwargs["rho_min"] = 1e-3
        coord_kwargs["rho_max"] = 1e3
    coord_cfg = JointLagrangianADMMConfig(**coord_kwargs)
    coord = JointLagrangianADMM(coord_cfg)
    if args.enable_t19_adaptive_rho:
        print(
            f"[t1] T19 enabled: adaptive-ρ via "
            f"tac.joint_admm_coordinator.adaptive_rho_step "
            f"(tau_grow={args.t19_tau_grow}, tau_shrink={args.t19_tau_shrink}, "
            f"rho_min={coord_cfg.rho_min}, rho_max={coord_cfg.rho_max}) "
            f"[predicted; T19 adaptive ρ 2-3× convergence speedup; "
            f"not direct score]"
        )
    # T20 / T22 enable banners (memo
    # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509).
    if args.enable_t20_kl_pose_distill:
        print(
            f"[t1] T20 enabled: KL pose-axis distill at T={args.t20_temperature} "
            f"(weight_pose={args.t20_weight_pose}, mode={args.t20_mode}) "
            f"[predicted; T20 KL pose distill at T=2.0]"
        )
    if args.enable_t22_temporal_consistency:
        print(
            f"[t1] T22 enabled: temporal consistency "
            f"(λ={args.t22_lambda_weight}, boundary={args.t22_boundary_handling}) "
            f"[predicted; T22 temporal consistency penalty]"
        )

    n_pairs = int(latents.shape[0])
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else min(args.batch_size, n_pairs)
    eval_batch_size = batch_size if args.eval_batch_size is None else int(args.eval_batch_size)
    if eval_batch_size < 1:
        raise SystemExit(f"[t1] --eval-batch-size must be >= 1, got {eval_batch_size}")

    best_proxy = float("inf")
    history = []
    score_domain_gradcheck: dict[str, float] | None = None
    # T20 + T22 running totals for provenance summary (memo
    # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509). When
    # both flags OFF, both stay 0.0 and provenance emits an "enabled=false"
    # row per the T13/T19 backward-compat pattern.
    t20_loss_running: float = 0.0
    t22_loss_running: float = 0.0
    t20_t22_n_batches: int = 0

    # Autocast FP16 / GradScaler — per engineering audit 2026-05-12 "exploit
    # hardware". T4 has 8× FP16 throughput vs FP32; A100 has 5×. Heavy forward
    # block (balle + decoder + scorer surrogates) runs in autocast; losses
    # cast to FP32 before Lagrangian dual update to avoid fp16 overflow.
    autocast_enabled = bool(getattr(args, 'enable_autocast_fp16', False)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)
    aux_scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled) if optim_aux is not None else None
    if autocast_enabled:
        print(
            "[t1] autocast FP16 enabled: forward block wraps balle+decoder+scorers; "
            "losses cast to FP32 for Lagrangian dual; GradScaler ON"
        )

    for epoch in range(epochs):
        decoder.train()
        balle.train()
        # Shuffle pair indices each epoch.
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]

            with torch.autocast('cuda', dtype=torch.float16, enabled=autocast_enabled):
                balle_out = balle(y)
                decoded = decoder(balle_out["y_hat"])
                if args.enable_scorer_domain_loss:
                    if posenet is None or segnet is None:  # pragma: no cover - defensive
                        raise RuntimeError("scorer-domain loss requested without loaded scorers")
                    decoded_rt = eval_roundtrip_decoded(
                        decoded,
                        noise_std=args.noise_std,
                        enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                    )
                    # Optional mp4 codec sim wire-in — closes the pixels→bytes→pixels
                    # gap (eval roundtrip simulates uint8 quant but NOT mp4 chroma
                    # 4:2:0 / DCT-quant losses). Per engineering audit 2026-05-12.
                    if getattr(args, 'enable_mp4_codec_sim', False):
                        decoded_rt = apply_mp4_codec_simulation_during_training(
                            decoded_rt,
                            chroma_subsample=True,
                            block_quant_noise_std=float(getattr(args, 'mp4_codec_sim_noise_std', 0.0)),
                        )
                    if gt_pose_cache is not None and gt_seg_cache is not None:
                        gt_pose_batch = _cached_batch_to_device(
                            gt_pose_cache,
                            idx,
                            device=device,
                        )
                        gt_seg_batch = _cached_batch_to_device(
                            gt_seg_cache,
                            idx,
                            device=device,
                        )
                        scorer_distortion, pose_loss, seg_loss = (
                            scorer_loss_terms_cached_btchw(
                                decoded_rt,
                                gt_pose_batch,
                                gt_seg_batch,
                                posenet,
                                segnet,
                                segmentation_surrogate=args.segmentation_surrogate,
                                segmentation_temperature=args.segmentation_temperature,
                                fisher_rao_eps=args.fisher_rao_eps,
                                sinkhorn_max_positions_per_chunk=args.sinkhorn_max_positions_per_chunk,
                                gt_seg_already_probs=gt_seg_already_probs,
                            )
                        )
                    else:
                        scorer_distortion, pose_loss, seg_loss = scorer_loss_terms_btchw(
                            decoded_rt,
                            tgt,
                            posenet,
                            segnet,
                            segmentation_surrogate=args.segmentation_surrogate,
                            segmentation_temperature=args.segmentation_temperature,
                            fisher_rao_eps=args.fisher_rao_eps,
                            sinkhorn_max_positions_per_chunk=args.sinkhorn_max_positions_per_chunk,
                        )
                    if args.pixel_l1_anchor_weight > 0:
                        pixel_anchor = F.l1_loss(decoded_rt, tgt)
                        distortion = scorer_distortion + args.pixel_l1_anchor_weight * pixel_anchor
                    else:
                        distortion = scorer_distortion
                else:
                    distortion = eval_roundtrip_pixel_l1(
                        decoded,
                        tgt,
                        noise_std=args.noise_std,
                        enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                    )
                    # Scaffold/debug mode only. Real score-lowering non-smoke runs
                    # must use --enable-scorer-domain-loss so these ADMM residuals
                    # are actual PoseNet/SegNet tensors, not constants.
                    seg_loss = torch.tensor(args.seg_target, device=device)
                    pose_loss = torch.tensor(args.pose_target, device=device)
            # Cast losses to FP32 BEFORE Lagrangian dual update — fp16 overflow
            # in coord.step() ρ-rescaling causes non-finite augmented_lagrangian.
            distortion = distortion.float() if autocast_enabled else distortion
            seg_loss = seg_loss.float() if autocast_enabled else seg_loss
            pose_loss = pose_loss.float() if autocast_enabled else pose_loss
            rate_bits = balle_out["rate_total_bits"]
            if autocast_enabled and hasattr(rate_bits, 'float'):
                rate_bits = rate_bits.float()
            use_direct_score = (
                args.enable_scorer_domain_loss
                and args.score_domain_objective == SCORE_DOMAIN_OBJECTIVE_DIRECT
            )
            if use_direct_score:
                rate_penalty = contest_rate_penalty_from_batch_bits(
                    rate_bits,
                    batch_pairs=int(idx.numel()),
                    total_pairs=n_pairs,
                )
                total_loss = distortion + rate_penalty
            else:
                res = coord.step(
                    distortion=distortion,
                    rate_bits=rate_bits,
                    seg_loss=seg_loss,
                    pose_loss=pose_loss,
                )
                if not torch.isfinite(res.augmented_lagrangian).all():
                    raise RuntimeError(
                        "[t1] non-finite augmented Lagrangian; refusing to update weights"
                    )
                total_loss = res.augmented_lagrangian
            # T20 — KL pose-axis distillation (memo
            # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509).
            # Adds Hinton-Vinyals-Dean 2014 KL on PoseNet 12-dim head logits.
            # Requires --enable-scorer-domain-loss (needs posenet loaded).
            # When OFF, the base objective is unchanged (backward-compat).
            t20_loss_value: float = 0.0
            t22_loss_value: float = 0.0
            if args.enable_t20_kl_pose_distill:
                if not args.enable_scorer_domain_loss or posenet is None or segnet is None:
                    raise RuntimeError(
                        "[t1] --enable-t20-kl-pose-distill requires "
                        "--enable-scorer-domain-loss (PoseNet must be loaded). "
                        "Refusing silent skip per CLAUDE.md fail-loud rule."
                    )
                # Student logits (gradient-reaching): decoded_rt → PoseNet.
                # Teacher logits: precomputed cache lookup (deterministic on
                # frozen scorer weights). Per "exploit auth scorer eval" 2026-05-12
                # engineering audit, the teacher forward (tgt → PoseNet) is
                # cached once before the epoch loop in teacher_pose_cache; the
                # batch indexing replaces a full PoseNet forward with O(B)
                # memory bandwidth — ~25-40% wall-clock saving on T4 with T20.
                student_pose_out, _ = scorer_forward_pair(
                    decoded_rt, posenet, segnet
                )
                if gt_pose_cache is not None:
                    teacher_pose_out = {
                        "pose": _cached_batch_to_device(
                            gt_pose_cache,
                            idx,
                            device=device,
                        ).detach()
                    }
                else:
                    # Defensive fallback if cache wasn't built (e.g., T20
                    # toggled mid-run via legacy path). Keep behavior identical.
                    with torch.no_grad():
                        _teacher_full, _ = scorer_forward_pair(
                            tgt, posenet, segnet
                        )
                    teacher_pose_out = _teacher_full
                t20_cfg = KLPoseDistillConfig(
                    temperature=float(args.t20_temperature),
                    weight_pose=float(args.t20_weight_pose),
                    mode=args.t20_mode,
                )
                t20_loss = apply_kl_pose_distill(
                    student_pose_out["pose"],
                    teacher_pose_out["pose"],
                    t20_cfg,
                )
                if not torch.isfinite(t20_loss).all():
                    raise RuntimeError(
                        "[t1] T20 KL pose distill produced non-finite loss; "
                        "refusing to update weights"
                    )
                total_loss = total_loss + t20_loss
                t20_loss_value = float(t20_loss.detach())
            # T22 — Temporal-consistency regularizer. Identity-warp form on
            # per-pair (B, P=2, C, H, W) decoded frames. Works regardless of
            # whether scorer-domain loss is enabled (operates on decoded RGB).
            if args.enable_t22_temporal_consistency:
                # Use decoded_rt when scorer-domain loss is on (matches the
                # roundtrip-aware proxy); otherwise use raw decoded.
                t22_input = decoded_rt if args.enable_scorer_domain_loss else decoded
                t22_cfg = TemporalConsistencyConfig(
                    lambda_weight=float(args.t22_lambda_weight),
                    flow_source="identity",
                    boundary_handling=args.t22_boundary_handling,
                )
                t22_loss = apply_temporal_consistency(
                    t22_input,
                    flow=None,  # identity-warp: λ · mean(|R_{t+1} - R_t|²)
                    config=t22_cfg,
                )
                if not torch.isfinite(t22_loss).all():
                    raise RuntimeError(
                        "[t1] T22 temporal-consistency produced non-finite "
                        "loss; refusing to update weights"
                    )
                total_loss = total_loss + t22_loss
                t22_loss_value = float(t22_loss.detach())
            # Track per-batch contributions for provenance summary.
            t20_loss_running += t20_loss_value
            t22_loss_running += t22_loss_value
            assert_stable_train_loss(
                total_loss,
                max_abs=float(args.max_stable_train_loss_abs),
                epoch=epoch,
                batch_index=n_batches,
                objective=str(args.score_domain_objective),
            )
            optim_main.zero_grad()
            # Backward + step with GradScaler when autocast enabled. GradScaler
            # handles fp16 dynamic loss scaling to prevent gradient underflow.
            if autocast_enabled:
                scaler.scale(total_loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optim_main)
                    if args.enable_scorer_domain_loss and score_domain_gradcheck is None:
                        score_domain_gradcheck = assert_score_domain_gradient_reachability(
                            decoder_params=decoder_trainable_params,
                            balle_main_params=balle_main_trainable_params,
                        )
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in main_params if p.requires_grad],
                        max_norm=float(args.grad_clip_norm),
                    )
                elif args.enable_scorer_domain_loss and score_domain_gradcheck is None:
                    score_domain_gradcheck = assert_score_domain_gradient_reachability(
                        decoder_params=decoder_trainable_params,
                        balle_main_params=balle_main_trainable_params,
                    )
                scaler.step(optim_main)
                scaler.update()
            else:
                total_loss.backward()
                if args.grad_clip_norm > 0:
                    if args.enable_scorer_domain_loss and score_domain_gradcheck is None:
                        score_domain_gradcheck = assert_score_domain_gradient_reachability(
                            decoder_params=decoder_trainable_params,
                            balle_main_params=balle_main_trainable_params,
                        )
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in main_params if p.requires_grad],
                        max_norm=float(args.grad_clip_norm),
                    )
                elif args.enable_scorer_domain_loss and score_domain_gradcheck is None:
                    score_domain_gradcheck = assert_score_domain_gradient_reachability(
                        decoder_params=decoder_trainable_params,
                        balle_main_params=balle_main_trainable_params,
                    )
                optim_main.step()

            if optim_aux is not None:
                optim_aux.zero_grad()
                if autocast_enabled:
                    with torch.autocast('cuda', dtype=torch.float16, enabled=True):
                        aux = balle.aux_loss()
                    aux_scaler.scale(aux).backward()
                    aux_scaler.step(optim_aux)
                    aux_scaler.update()
                else:
                    aux = balle.aux_loss()
                    aux.backward()
                    optim_aux.step()

            ema_decoder.update(decoder)
            ema_balle.update(balle)

            epoch_loss += float(total_loss.detach())
            n_batches += 1
            t20_t22_n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            metrics = _eval_ema_proxy(
                decoder=decoder,
                balle=balle,
                ema_decoder=ema_decoder,
                ema_balle=ema_balle,
                latents=latents,
                target_pixels=target_pixels,
                noise_std=args.noise_std,
                enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                eval_batch_size=eval_batch_size,
            )
            print(
                f"[t1] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} "
                f"ema_pixel_l1={metrics['ema_proxy_pixel_l1']:.4f} "
                f"ema_rate_bits={metrics['ema_proxy_rate_bits']:.0f} "
                f"rho={coord.rho:.3f}"
            )
            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                **metrics,
                "rho": coord.rho,
                "lambdas": dict(coord.lambdas),
            })
            previous_best_proxy = best_proxy
            best_proxy = _maybe_save_ema_checkpoint(
                output_dir=args.output_dir,
                decoder=decoder,
                balle=balle,
                ema_decoder=ema_decoder,
                ema_balle=ema_balle,
                coord=coord,
                proxy_score=metrics["ema_proxy_pixel_l1"],
                epoch=epoch + 1,
                best_proxy_score=best_proxy,
            )
            best_proxy_improved = best_proxy < previous_best_proxy
            if args.archive_in_loop:
                should_build_candidate = (
                    best_proxy_improved
                    or (epoch + 1) % args.archive_every_epochs == 0
                    or epoch == epochs - 1
                )
                if should_build_candidate:
                    candidate_row = materialize_archive_in_loop_candidate(
                        output_dir=args.output_dir,
                        decoder=decoder,
                        balle=balle,
                        ema_decoder=ema_decoder,
                        ema_balle=ema_balle,
                        latents=latents,
                        decoder_config=decoder_config,
                        balle_config=balle_config,
                        epoch=epoch + 1,
                        metrics=metrics,
                        best_proxy_improved=best_proxy_improved,
                        max_candidate_archive_bytes=args.max_candidate_archive_bytes,
                        baseline_archive_sha256=args.baseline_archive_sha256.lower(),
                        baseline_archive_size_bytes=args.baseline_archive_size_bytes,
                    )
                    history[-1]["archive_in_loop_candidate"] = {
                        "candidate_dir": candidate_row["candidate_dir"],
                        "archive_bytes": candidate_row["archive_bytes"],
                        "archive_sha256": candidate_row["archive_sha256"],
                        "rate_cap_passed": candidate_row["rate_cap_passed"],
                        "exact_cuda_eligible": candidate_row["exact_cuda_eligible"],
                        "compiler_blockers": candidate_row["compiler_blockers"],
                    }

    if args.enable_scorer_domain_loss and score_domain_gradcheck is None:
        raise RuntimeError(
            "[t1] score-domain gradcheck missing after training loop; refusing "
            "archive emission because scorer-domain gradients were never proven"
        )

    # Build the EMA archive + maybe-run auth eval.
    archive_path = build_archive_from_ema(
        output_dir=args.output_dir,
        decoder=decoder,
        balle=balle,
        ema_decoder=ema_decoder,
        ema_balle=ema_balle,
        latents=latents,
        decoder_config=decoder_config,
        balle_config=balle_config,
    )
    print(f"[t1] wrote archive: {archive_path} ({archive_path.stat().st_size} bytes)")

    auth_eval_result = None
    if not args.smoke:  # never auth-eval the smoke-synthetic archive
        auth_eval_result = maybe_run_auth_eval(
            archive_path=archive_path,
            submission_dir=args.output_dir / "submission_dir",
            output_dir=args.output_dir,
            enabled=args.auth_eval,
            dispatch_lane_id=args.dispatch_lane_id,
            dispatch_claims_path=args.dispatch_claims_path,
        )

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # T19 — emit per-iteration ρ trajectory side-log (always, when the
    # coordinator has any entries; the helper appends only on actual ρ
    # changes). When --enable-t19-adaptive-rho is OFF the trajectory is
    # populated only when the legacy backend triggers a ρ update.
    t19_summary: dict | None = None
    if args.enable_t19_adaptive_rho or coord.rho_trajectory:
        rho_log_path = args.output_dir / "rho_trajectory.json"
        rho_log_path.write_text(
            json.dumps(coord.rho_trajectory, indent=2, default=str)
        )
        t19_summary = {
            "enabled": bool(args.enable_t19_adaptive_rho),
            "tau_grow": float(args.t19_tau_grow),
            "tau_shrink": float(args.t19_tau_shrink),
            "rho_min": float(coord.config.rho_min),
            "rho_max": float(coord.config.rho_max),
            "n_rho_updates": len(coord.rho_trajectory),
            "rho_final": float(coord.rho),
            "rho_trajectory_path": str(rho_log_path),
            "tag": (
                "[predicted; T19 adaptive ρ 2-3× convergence speedup; "
                "not direct score]"
                if args.enable_t19_adaptive_rho
                else "[legacy adaptive-ρ backend]"
            ),
        }
    if score_domain_gradcheck is not None:
        (args.output_dir / "score_domain_gradcheck.json").write_text(
            json.dumps(score_domain_gradcheck, indent=2)
        )
    # T20 — KL pose-axis distillation summary (memo
    # feedback_t20_t22_pose_axis_temporal_lateral_leaps_landed_20260509).
    t20_summary: dict | None = None
    if args.enable_t20_kl_pose_distill:
        t20_summary = {
            "enabled": True,
            "temperature": float(args.t20_temperature),
            "weight_pose": float(args.t20_weight_pose),
            "mode": args.t20_mode,
            "n_batches": int(t20_t22_n_batches),
            "loss_total": float(t20_loss_running),
            "loss_mean_per_batch": float(
                t20_loss_running / max(t20_t22_n_batches, 1)
            ),
            "tag": "[predicted; T20 KL pose distill at T=2.0]",
        }
    # T22 — Temporal-consistency summary (same memo).
    t22_summary: dict | None = None
    if args.enable_t22_temporal_consistency:
        t22_summary = {
            "enabled": True,
            "lambda_weight": float(args.t22_lambda_weight),
            "boundary_handling": args.t22_boundary_handling,
            "flow_source": "identity",  # this trainer wires the no-flow form
            "n_batches": int(t20_t22_n_batches),
            "loss_total": float(t22_loss_running),
            "loss_mean_per_batch": float(
                t22_loss_running / max(t20_t22_n_batches, 1)
            ),
            "tag": "[predicted; T22 temporal consistency penalty]",
        }
    write_provenance(
        output_dir=args.output_dir,
        args=args,
        encoder_provenance=encoder_provenance,
        n_decoder_params=n_decoder_params,
        n_balle_params=n_balle_params,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        t13_bit_reallocation=t13_report,
        t19_adaptive_rho=t19_summary,
        t20_kl_pose_distill=t20_summary,
        t22_temporal_consistency=t22_summary,
        score_domain_gradcheck=score_domain_gradcheck,
        pr95_trainer_parity=pr95_trainer_parity,
    )
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (args.output_dir / "auth_eval_summary.json").write_text(
        json.dumps(auth_eval_result or {"skipped": True}, indent=2)
    )

    # Revert global YUV6 monkey-patch on graceful exit so downstream
    # importers (test harness, adjudication tools) see upstream unchanged.
    if yuv6_token is not None:
        unpatch_upstream_yuv6(yuv6_token)

    print("[t1] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
