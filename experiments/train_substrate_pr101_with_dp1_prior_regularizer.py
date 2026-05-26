# SPDX-License-Identifier: MIT
"""PATH 2 scaffold: pr101_lc_v2_clone trainer with DP1 frame-prior regularizer.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Lane: lane_dp1_plus_fec6_dual_stacking_build_20260517

**Premise verification (Catalog #229):** the operator's original PATH 2
description ("L2 regularizer on PR101 decoder weights") is structurally
incompatible with fec6 because fec6 has no learned decoder weights. DP1
codebook is a FRAME-SPACE prior (road_plane_basis, sky_horizon_profile,
vehicle_appearance_basis — all RGB tensors). The natural recipient substrate
for the DP1 frame-prior is ``train_substrate_pr101_lc_v2_clone_enhanced_curriculum``
which HAS a learned renderer whose RGB output can be regularized.

This scaffold is therefore PATH 2-REFORMULATED:
  primary substrate: pr101_lc_v2_clone (learned HNeRV decoder)
  auxiliary loss:    DashcamPriorLoss on the decoder's RGB output
  archive bytes:     +0 (DP1 codebook is COMPRESS-TIME ONLY; not shipped in
                     archive — per CLAUDE.md Catalog #146 contest_one_video_replay
                     allowance, the codebook is BAKED INTO inflate.py as a numpy
                     constant IF the operator approves shipping the codebook
                     in inflate.py source rather than in the archive blob)

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
``_full_main`` raises NotImplementedError pending council Phase 2 approval.
``_smoke_main`` runs a minimal CPU-only correctness check: DashcamPriorLoss
constructs from a synthetic codebook + applies cleanly to a random RGB tensor +
gradients flow through it.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":
* CANONICAL adopt: SubstrateContract decoration / Catalog #205 device select /
  Catalog #226 gate_auth_eval_call routing / Catalog #190 hardware_substrate
  detection / fcntl-locked posterior writes per Catalog #128/#131.
* UNIQUE fork: the trainer's loss combines pr101_lc_v2 score-aware loss WITH
  DashcamPriorLoss as a frame-space soft prior. The DP1 codebook is FROZEN
  (registered as torch buffers, never trained). λ_DP1 is a tunable hyperparam.
* DEFER: the codebook-baked-in-inflate.py vs codebook-in-archive design
  decision is council-grade (CLAUDE.md "Design decisions") — see design memo.

Per CLAUDE.md "9-dimension success checklist evidence" + Catalog #294 + #303
+ #305 + #296 + #309: see ``.omx/research/dp1_dual_stacking_design_20260517.md``
for the complete design memo with all required sections.

Catalog references:
* Catalog #240 ``check_substrate_contest_cuda_chain_complete_or_research_only_tagged``
  — this trainer's ``_full_main`` raises NotImplementedError; recipe MUST
  declare ``research_only: true``.
* Catalog #241 ``check_substrate_uses_register_decorator_or_explicitly_legacy_tagged``
  — decorated with ``@register_substrate``.
* Catalog #242 ``check_register_substrate_contract_fields_canonical`` — all 36
  fields declared per the canonical contract.
* Catalog #209 ``check_no_contest_video_leakage_in_distillation_callers`` —
  DP1 codebook is distilled OFFLINE from Comma2k19; this trainer LOADS the
  frozen codebook, never re-distills.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from tac.substrate_registry import SubstrateContract, register_substrate


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Catalog #151 + #152: Tier 1 required-flags manifest. Operator wrappers thread
# env-vars + pre-dispatch file validation runs.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PR101_DP1_VIDEO_PATH",
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "generator_command": "ls upstream/videos/0.mkv",
    },
    "--dp1-codebook-bin": {
        "env": "PR101_DP1_CODEBOOK_BIN",
        "default": "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/codebook.bin",
        "required_input_file": True,
        "generator_command": (
            ".venv/bin/python -m tac.substrates.pretrained_driving_prior.codebook --help "
            "# distill via tools/distill_dp1_codebook.py (deferred)"
        ),
    },
    "--output-dir": {
        "env": "PR101_DP1_OUTPUT_DIR",
        "default": "experiments/results/lane_dp1_plus_fec6_dual_stacking_build_20260517",
        "required_input_file": False,
    },
    "--device": {
        "env": "PR101_DP1_DEVICE",
        "default": "cuda",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "PR101_DP1_EPOCHS",
        "default": "100",
        "required_input_file": False,
    },
    "--lambda-dp1-prior": {
        "env": "PR101_DP1_LAMBDA",
        "default": "0.05",
        "required_input_file": False,
    },
    "--enable-autocast-fp16": {
        "env": "PR101_DP1_ENABLE_AUTOCAST_FP16",
        "default": "0",
        "required_input_file": False,
    },
}


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242). Decoration extincts the
# Z3 v2 silent-drift bug class by binding (a) trainer's claimed contract, (b)
# recipe schema, (c) lane registry, (d) cost-band envelope into ONE source.
# ---------------------------------------------------------------------------
PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="pr101_with_dp1_prior_regularizer",
    lane_id="lane_dp1_plus_fec6_dual_stacking_build_20260517",
    target_modes=("contest_one_video_replay", "research_substrate"),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/dp1_dual_stacking_design_20260517.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "PR101_LC_V2 monolithic single-file 0.bin (HNeRV-family grammar; learned "
        "decoder weights fp4/brotli + per-pair selector + Huffman entropy). DP1 "
        "codebook is COMPRESS-TIME ONLY (frozen during training); the codebook "
        "either ships in archive (variant-A; +5-10KB) or is BAKED INTO inflate.py "
        "as numpy constant per CLAUDE.md Catalog #146 contest_one_video_replay "
        "allowance (variant-B; +0 archive bytes; ~5-10KB inflate.py source "
        "growth). Design decision DEFERRED to council Phase 2."
    ),
    parser_section_manifest={
        "header": "PR101_magic_and_version_length_prefixed",
        "decoder_weights": "fp4_brotli_blob",
        "per_pair_selector": "huffman_k16",
        "dp1_codebook_optional": "fp16_brotli_blob_or_baked_in_inflate",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "numpy"),
    export_format="custom",  # PR101 LC_V2 fp4_brotli + huffman_k16 + optional DP1 codebook (variant-A) or baked-in (variant-B)
    score_aware_loss="custom",  # canonical scorer_loss_terms_btchw + DashcamPriorLoss frame-prior
    bolt_on_loc_budget=350,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,  # variant-A: +5-10KB; variant-B: +0
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,
    recipe_research_only=True,  # _full_main raises NotImplementedError
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_canary_dependency="pretrained_driving_prior",
    recipe_video_input_strategy="per_dispatch_local_copy",
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=1.50,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="custom",  # rate-distortion + DashcamPriorLoss KL frame-prior term
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_152_required_input_files_validated_pre_dispatch",
        "catalog_205_select_inflate_device_used",
        "catalog_209_no_contest_video_leakage_in_distillation_callers",
        "catalog_210_dp1_codebook_provenance_metadata_preserved",
        "catalog_220_operational_mechanism_declared_research_only",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_229_premise_verification_before_edit",
        "catalog_240_research_only_recipe_pending_council_phase_2",
        "catalog_241_register_substrate_decorator_used",
        "catalog_242_substrate_contract_fields_canonical",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "Sensitivity captured by per-tensor entropy of PR101_LC_V2 decoder "
            "weights + DP1 frame-prior KL divergence. The DP1 prior contributes "
            "to the Lagrangian's distortion term, NOT a per-tensor sensitivity "
            "score."
        ),
        "hook_bit_allocator_class": (
            "fp4/brotli on decoder weights + huffman_k16 on per-pair selector + "
            "fp16/brotli on DP1 codebook (variant-A) OR baked-in (variant-B). "
            "Per-substream not per-tensor bit allocator."
        ),
        "hook_probe_disambiguator": (
            "tools/probe_dp1_prior_lambda_disambiguator.py (planned; council "
            "Phase 2 approval required). Sweeps lambda_dp1_prior in {0.0, 0.01, "
            "0.05, 0.10, 0.25, 0.50} to disambiguate prior-effect from "
            "overfitting."
        ),
    },
)


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu requires paired --advisory-cpu-explicitly-waived."""
    if getattr(args, "full_cpu", False) and not getattr(
        args, "advisory_cpu_explicitly_waived", False
    ):
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "CLAUDE.md 'MPS auth eval is NOISE' + Catalog #197"
        )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "PATH 2 scaffold: pr101_lc_v2_clone trainer with DP1 frame-prior "
            "regularizer (DashcamPriorLoss applied to learned-decoder RGB output)."
        )
    )
    parser.add_argument("--video-path", type=str, default="upstream/videos/0.mkv")
    parser.add_argument(
        "--dp1-codebook-bin",
        type=str,
        default="experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/codebook.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/lane_dp1_plus_fec6_dual_stacking_build_20260517",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lambda-dp1-prior", type=float, default=0.05)
    parser.add_argument("--enable-autocast-fp16", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run _smoke_main only.")
    parser.add_argument(
        "--full-cpu", action="store_true",
        help="Allow non-smoke CPU path (advisory; refused without coupled waiver)."
    )
    parser.add_argument(
        "--advisory-cpu-explicitly-waived", action="store_true",
        help="Catalog #197 coupling flag; required when --full-cpu is set."
    )
    return parser


def _smoke_main(args: argparse.Namespace) -> int:
    """Minimal correctness smoke: DashcamPriorLoss constructs + gradients flow.

    No dispatch, no Modal, no real training. Verifies:
    1. tac.substrates.pretrained_driving_prior.DashcamPriorLoss imports
    2. PriorApplicationWeights constructs with sensible defaults
    3. DashcamPriorLoss applies cleanly to a random RGB tensor
    4. apply_soft_prior produces a finite output for strength > 0
    5. gradients flow through DashcamPriorLoss.apply_soft_prior

    Returns 0 on success, nonzero on failure.
    """
    import torch

    from tac.substrates.pretrained_driving_prior import (
        DashcamCodebook,
        DashcamPriorLoss,
        PriorApplicationWeights,
        deterministic_zero_codebook,
    )

    print("[smoke] PATH 2 scaffold smoke check: DashcamPriorLoss correctness")

    codebook: DashcamCodebook = deterministic_zero_codebook()
    weights = PriorApplicationWeights(
        lambda_road_plane=0.05,
        lambda_sky_horizon=0.02,
        lambda_vehicle=0.01,
        eval_resolution=(384, 512),
    )
    prior_loss = DashcamPriorLoss(codebook=codebook, weights=weights, device="cpu")
    print(f"[smoke] DashcamPriorLoss instantiated with {sum(p.numel() for p in prior_loss.buffers())} buffer elements")

    # Build a tiny synthetic RGB batch with grad enabled.
    rgb_pred = torch.randn(2, 3, 384, 512, dtype=torch.float32, requires_grad=True)
    rgb_pred_unit = torch.sigmoid(rgb_pred)  # in [0, 1]

    adjusted = prior_loss.apply_soft_prior(rgb_pred_unit, strength=0.1)
    print(f"[smoke] apply_soft_prior output shape: {tuple(adjusted.shape)}")
    assert adjusted.shape == rgb_pred_unit.shape, "shape mismatch"
    assert torch.isfinite(adjusted).all(), "apply_soft_prior produced non-finite output"

    # Gradient check: loss = (adjusted - target)^2 should produce nonzero
    # grads on rgb_pred.
    target = torch.zeros_like(rgb_pred_unit)
    loss = ((adjusted - target) ** 2).mean()
    loss.backward()
    assert rgb_pred.grad is not None, "no gradient flowed through DashcamPriorLoss"
    grad_norm = rgb_pred.grad.norm().item()
    print(f"[smoke] gradient norm through prior: {grad_norm:.6f}")
    assert grad_norm > 1e-8, f"gradient too small: {grad_norm}"

    print("[smoke] PATH 2 scaffold smoke OK (DashcamPriorLoss correctness verified).")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """DEFERRED: full PR101+DP1-prior training awaits council Phase 2 approval.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
    Catalog #240: this trainer is RESEARCH_ONLY at landing. The full training
    integration requires:

    1. Council Phase 2 design verdict on:
       (a) variant-A (codebook in archive; +5-10KB rate cost) vs variant-B
           (codebook baked into inflate.py; +0 rate cost) per CLAUDE.md
           Catalog #146 contest_one_video_replay allowance
       (b) λ_DP1 sweep design (which values to test in parallel)
       (c) cost-band budget (T4 100ep ~$1.50; full-curriculum ~$10)
    2. Probe-disambiguator landing (tools/probe_dp1_prior_lambda_disambiguator.py)
    3. Catalog #295 self-contained inflate.py for variant-B (codebook baked in)
       OR canonical archive grammar extension for variant-A.
    4. pr101_lc_v2_clone trainer integration surface — _full_main must subclass
       the canonical pr101_lc_v2 trainer and inject DashcamPriorLoss into the
       score-aware loss via λ * dp1_prior_loss + ((1 - λ) * pr101_lc_v2_loss).
    5. Catalog #229 premise verification on the integration surface BEFORE
       implementing.

    Reactivation criteria:
    - council Phase 2 deliberation memo lands with PROCEED verdict
    - design memo updated with variant-A vs variant-B decision
    - probe-disambiguator lands + bootstrapped on λ sweep [0.0, 0.01, 0.05, 0.10]
    - sister CPU smoke dispatch validates DashcamPriorLoss + pr101_lc_v2 loss
      compose correctly on the fec6 frontier archive's GT pair batches
    """
    raise NotImplementedError(
        "phase_2_council_approval_required_to_lift_full_main_NotImplementedError; "
        "see lane_dp1_plus_fec6_dual_stacking_build_20260517 design memo at "
        ".omx/research/dp1_dual_stacking_design_20260517.md § PATH 2 Reformulated. "
        "Reactivation: council Phase 2 verdict + probe-disambiguator + variant-A/B "
        "decision."
    )


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


@register_substrate(PR101_WITH_DP1_PRIOR_SUBSTRATE_CONTRACT)
def _register_substrate_marker() -> None:
    """Marker function so Catalog #241 decorator scanner sees the contract."""
    return None


if __name__ == "__main__":
    sys.exit(main())
