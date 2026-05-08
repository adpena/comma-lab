#!/usr/bin/env python3
"""Training script for GPU-lane mask-conditioned renderer.

Pipeline:
    GT frames -> frozen SegNet -> 5-class masks (extracted once at startup)
    Training loop: mask pairs -> PairGenerator -> scorer_loss -> backprop

Usage:
    .venv/bin/python -m tac.experiments.train_renderer --profile mask_renderer_smoke --tag test
    .venv/bin/python -m tac.experiments.train_renderer --profile mask_renderer_full --tag v1
    .venv/bin/python -m tac.experiments.train_renderer --epochs 100 --lr 1e-3 --tag quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os as _os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# DX-fix 2026-04-25: line-buffer stdout/stderr so progress logs flush
# immediately when piped to log files (Python buffers ~8KB by default,
# making long-running scripts appear silent for hours per the optimize_poses
# incident on the A100 today).
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# ── Path setup ──────────────────────────────────────────────────────────

_script_dir = Path(__file__).resolve().parent
_repo = _script_dir.parent.parent.parent  # src/tac/experiments -> repo root
sys.path.insert(0, str(_repo / "src"))

_upstream = Path(_os.environ.get(
    "TAC_UPSTREAM_DIR",
    str(_repo / "workspace" / "upstream" / "comma_video_compression_challenge"),
))
if _upstream.exists() and str(_upstream) not in sys.path:
    sys.path.insert(0, str(_upstream))

# ── Imports (after path setup) ──────────────────────────────────────────

from tac.contrib.wavelet_renderer import build_wavelet_renderer  # noqa: E402
from tac.data import decode_video, pair_from_frames, pair_start_indices  # noqa: E402
from tac.feature_masking import FeatureMasker  # noqa: E402
from tac.fp4_quantize import (  # noqa: E402
    QATRendererFP4,
    dequantize_fp4,
    quantize_fp4,
)
from tac.fridrich_losses import dct_quant_loss  # noqa: E402
from tac.loss_t2_xpred import x_prediction_loss  # noqa: E402
from tac.losses import (  # noqa: E402
    _hwc_to_chw,
    eval_scorer_loss,
    frequency_aware_loss,
    kl_distill_segnet_only,
    scorer_forward_pair,
    scorer_loss,
    scorer_loss_cached,
    segnet_uncertainty_weighted_loss,
    uniward_quant_noise_loss,
)
from tac.mae_mask_aug import (  # noqa: E402
    MAEMaskAugConfig,
    MAEMaskAugmenter,
)
from tac.mask_codec import extract_masks, mask_pair_from_index  # noqa: E402
from tac.profiles import PROFILES  # noqa: E402
from tac.renderer import simulate_eval_roundtrip  # noqa: E402
from tac.scorer import detect_device, load_scorers  # noqa: E402
from tac.self_augmentation_v2 import (  # noqa: E402
    HighSigmaSampler,
    HighSigmaStrategyConfig,
    apply_sigma_noise_to_input,
)
from tac.training import EMA  # noqa: E402
from tac.utils import setup_signal_handlers, write_telemetry  # noqa: E402

# ── Variant routing ─────────────────────────────────────────────────────
#
# Codex R5-2 Finding #1 (2026-04-27): variant routing must agree between the
# build_renderer dispatch (line ~1051) AND the --auth-eval-on-best FP4A export
# guard (line ~2070). Pre-fix the guard hardcoded `variant in ('default', None)`,
# which silently reject-then-RuntimeError'd every profile that flows through
# build_renderer (dilated, psd, mask_renderer, plus ~20 others). Result: hours
# of training burned with no authoritative score, exactly the failure mode
# `--auth-eval-on-best` exists to prevent.
#
# Variants in `_VARIANTS_NON_BUILD_RENDERER` have their own builder branch in
# train() and produce checkpoints whose state_dict layout cannot be exported by
# the canonical AsymmetricPairGenerator FP4A path. For those we MUST disable
# the auth-eval-on-best guard at startup (not after training).
#
# Variants in `_VARIANTS_BUILD_RENDERER_FP4A_OK` flow through build_renderer
# and produce AsymmetricPairGenerator/PairGenerator state_dicts that the
# canonical export_asymmetric_checkpoint_fp4 path handles. The auth-eval-on-best
# block can FP4A-export their best checkpoints.
#
# Keep these two sets EXHAUSTIVE: every profile["variant"] value in profiles.py
# must appear in exactly one set. test_train_renderer_variant_routing_complete
# pins this.
_VARIANTS_NON_BUILD_RENDERER: frozenset[str] = frozenset({
    "wavelet_renderer",
    "coord_renderer",
    "coolchic_renderer",
    "c3_residual_renderer",
    "dp_sims",
    "vqvae",
    "diffusion_teacher",
})

_VARIANTS_BUILD_RENDERER_FP4A_OK: frozenset[str] = frozenset({
    # Empty string + None handled separately (legacy default).
    "default",
    "mask_renderer",
    "dilated",
    "psd",
    "channel_recurrent",
    "constrained_gen",
    "constrained_gen_pipeline",
    "content_adaptive",
    "counterpoint",
    "cross_disciplinary",
    "dct_midband",
    "depthwise_renderer",
    "distillation",
    "domain_solver",
    "dp_sims_v2",
    "dual_head",
    "film_qat",
    "gated_dilated",
    "lagrangian_dual",
    "luma_dilated",
    "pareto_trace",
    "pixelshuffle_dilated_v2",
    "pixelshuffle_upscale",
    "siren",
    "variational_gen",
})


def _variant_supports_fp4a_export(variant: str | None) -> bool:
    """True if the given variant flows through build_renderer and produces a
    checkpoint that export_asymmetric_checkpoint_fp4 can serialise."""
    if variant is None or variant == "":
        return True  # legacy default → build_renderer
    return variant in _VARIANTS_BUILD_RENDERER_FP4A_OK


# ── Argument parsing ────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mask-conditioned renderer (GPU lane)")

    # Profile
    renderer_profiles = [
        k for k in PROFILES
        if k.startswith(("mask_renderer", "wavelet_renderer", "coord_renderer", "coolchic_renderer", "c3_residual_renderer"))
    ]
    p.add_argument("--profile", type=str, default=None, choices=list(PROFILES.keys()),
                   help=f"Named profile. Renderer profiles: {renderer_profiles}")
    p.add_argument("--variant", type=str, default=None,
                   choices=[
                       "mask_renderer", "wavelet_renderer", "coord_renderer",
                       "coolchic_renderer", "c3_residual_renderer",
                       "dp_sims", "vqvae", "diffusion_teacher",
                   ],
                   help="Renderer variant (auto-detected from profile if not set)")

    # Architecture
    p.add_argument("--base-ch", type=int, default=None, help="MaskRenderer base channels")
    p.add_argument("--mid-ch", type=int, default=None, help="MaskRenderer bottleneck channels")
    p.add_argument("--embed-dim", type=int, default=None, help="Per-class embedding dim")
    p.add_argument("--motion-hidden", type=int, default=None, help="MotionPredictor hidden channels")
    p.add_argument("--depth", type=int, default=None, help="U-Net depth (1=single-scale, 2=two-scale)")
    p.add_argument("--latent-ch", type=int, default=None, help="Cool-Chic/C3 latent channels")
    p.add_argument("--residual-hidden", type=int, default=None, help="C3 residual head hidden width")
    p.add_argument("--residual-layers", type=int, default=None, help="C3 residual hidden layer count")
    p.add_argument("--residual-scale", type=float, default=None, help="C3 residual pixel bound")

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    # 2026-04-29 silent-default audit fix: argparse default=1.0 silently
    # overrode profile's grad_clip=10.0 in 2 profiles (KL distill bug class).
    # Pass None so _resolve() consults profile, falling back to 1.0 only when
    # neither CLI nor profile set it.
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--pretrain-epochs", type=int, default=None,
                   help="Phase 1 epochs: L1+edge loss, no scorer (default: from profile or 0). "
                        "LEGACY 2-phase API. New profiles should use --phase{1..5}-epochs.")
    # Quantizr-style 5-phase QAT schedule (R2 C3 architectural fix
    # 2026-04-26). Per Quantizr's published recipe:
    #   Phase 1 anchor: pretrain renderer on pixel L1 vs GT only.
    #   Phase 2 finetune: add SegNet hinge + PoseNet MSE.
    #   Phase 3 joint: + Fridrich (texture / l_inf / markov / dct_quant) + KL.
    #   Phase 4 QAT: enable FP4 fake-quant on weights, low LR.
    #   Phase 5 final: hard-pair-emphasis sampler at very low LR.
    # Defaults are None so the profile resolver wins; phases with 0 epochs
    # are skipped (preserving legacy 2-phase profiles).
    p.add_argument("--phase1-epochs", type=int, default=None,
                   help="Phase 1 (anchor) epoch count. Pixel L1 only.")
    p.add_argument("--phase2-epochs", type=int, default=None,
                   help="Phase 2 (finetune) epoch count. Pixel + scorer.")
    p.add_argument("--phase3-epochs", type=int, default=None,
                   help="Phase 3 (joint) epoch count. Full Fridrich + KL stack.")
    p.add_argument("--phase4-epochs", type=int, default=None,
                   help="Phase 4 (QAT) epoch count. Enables FP4 fake-quant.")
    p.add_argument("--phase5-epochs", type=int, default=None,
                   help="Phase 5 (final) epoch count. Hard-pair emphasis polish.")
    p.add_argument("--phase1-lr", type=float, default=None,
                   help="Phase 1 LR (anchor). Default 1e-3.")
    p.add_argument("--phase2-lr", type=float, default=None,
                   help="Phase 2 LR (finetune). Default 5e-4.")
    p.add_argument("--phase3-lr", type=float, default=None,
                   help="Phase 3 LR (joint). Default 3e-4.")
    p.add_argument("--phase4-lr", type=float, default=None,
                   help="Phase 4 LR (QAT). Default 5e-5 (10x reduction from Phase 3).")
    p.add_argument("--phase5-lr", type=float, default=None,
                   help="Phase 5 LR (final polish). Default 1e-5.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Truncate GT frames to first N (for trend / smoke runs). "
                        "Default: use all loaded frames (typically 1200).")

    # Mask augmentation: train against AV1-quantized masks so the renderer
    # learns to handle the masks it will actually see at inflate time.
    # Yousfi diagnosis (2026-04-25): the gating measurement showed CRF=63
    # masks caused 34x PoseNet explosion because the renderer was only ever
    # trained on clean GT masks. Mixing noisy variants during training closes
    # this distribution gap.
    p.add_argument("--mask-noise-mkv", type=str, default=None,
                   help="Path to a CRF-encoded masks.mkv to augment training. "
                        "Decoded once at startup; randomly swapped in per pair "
                        "with probability --mask-noise-prob.")
    p.add_argument("--mask-noise-prob", type=float, default=0.5,
                   help="Probability of using the noisy mask variant on each "
                        "training step (default 0.5 = 50/50 mix).")
    p.add_argument("--use-saug-v2", action="store_true", default=None,
                   help="Lane SAUG-V2: enable Cosmos HighSigmaStrategy-style "
                        "per-sample renderer-input noise.")
    p.add_argument("--saug-v2-redraw-fraction", type=float, default=None,
                   help="SAUG-V2 probability of drawing from the high-sigma "
                        "range (default 0.05).")
    p.add_argument("--saug-v2-high-sigma-min", type=float, default=None,
                   help="SAUG-V2 high-sigma log-uniform lower bound "
                        "(default 80.0).")
    p.add_argument("--saug-v2-high-sigma-max", type=float, default=None,
                   help="SAUG-V2 high-sigma log-uniform upper bound "
                        "(default 2000.0).")
    p.add_argument("--saug-v2-normal-sigma-min", type=float, default=None,
                   help="SAUG-V2 normal-sigma log-uniform lower bound "
                        "(default 0.5).")
    p.add_argument("--saug-v2-normal-sigma-max", type=float, default=None,
                   help="SAUG-V2 normal-sigma log-uniform upper bound "
                        "(default 80.0).")
    p.add_argument("--use-calibrated-positional-encoding", action="store_true",
                   default=None,
                   help="Lane CG: enable analytic per-pixel viewing-ray "
                        "positional encoding (src/tac/contrib/"
                        "calibrated_positional_encoding.py).")

    # Lane MAE-V: Masked Autoencoder Variant on input mask patches. Drops a
    # MAEMaskAugmenter (src/tac/mae_mask_aug.py) into the per-step pair
    # construction so a fraction of mask patches are replaced by a Gumbel-
    # softmax sample from a learnable categorical mask token. Eval-mode is
    # a passthrough, so the inference distribution matches what the contest
    # scorer sees. Predicted band [0.85, 1.10] per Cosmos research synthesis.
    p.add_argument("--use-mae-mask-aug", action="store_true", default=None,
                   help="Lane MAE-V: enable MAE-style random patch masking on "
                        "input masks during training (eval is passthrough).")
    p.add_argument("--mae-mask-ratio", type=float, default=None,
                   help="MAE-V probability per patch of being replaced by the "
                        "learnable mask token (default 0.25).")
    p.add_argument("--mae-patch-size", type=int, default=None,
                   help="MAE-V patch size in pixels (default 16).")

    # Half-frame mask simulation: replace mask_t with inverse-warp(mask_t1)
    # so the renderer learns the same mask distribution it sees at inflate
    # when the archive ships only odd-frame masks (Quantizr paradigm — the
    # main rate lever, saves ~50% of mask bytes).
    #
    # Without this, the model trains on (mask_t, mask_t1) where both are
    # ground-truth, but at inflate sees (warp(mask_t1), mask_t1) where
    # mask_t is approximated. This train/inflate mismatch costs 0.05-0.10
    # score points. Wiring closes the gap.
    #
    # Zoom scalars come from analytical lane-mark-speed estimation (no GT
    # poses needed — the masks themselves are sufficient per Hotz). The
    # warp uses tac.radial_zoom.RadialZoomWarp.warp_inverse_masks.
    # default=None so the profile resolver can override (Lane D 2026-04-27 fix).
    # With default=0.0 the profile value was DEAD CODE: _resolve() only falls
    # through to the profile when cli_val is None. Same class of bug as KL
    # distill (commit 38a250b8) and Yousfi #5 uncertainty loss (R2 C1 audit).
    p.add_argument("--mask-half-sim-prob", type=float, default=None,
                   help="Probability of replacing mask_t with inverse-warp("
                        "mask_t1) during training (default 0.0 = off). Set to "
                        "0.5 when shipping a half-frame archive (Quantizr "
                        "paradigm) so train/inflate distributions match.")
    # Lane V-V2 (2026-04-27) annealing schedule for the half-frame warp prob.
    # Format: "start_value,ramp_start_frac,end_value,ramp_end_frac" (4 floats).
    # Schedule: prob = start_value for epoch_frac < ramp_start_frac,
    #           linear ramp to end_value over [ramp_start_frac, ramp_end_frac],
    #           prob = end_value for epoch_frac > ramp_end_frac.
    # Example "0,0.3,1.0,0.7" = warmup full-frame for 30%, ramp 30%→70%,
    #                            half-frame 70%→100%.
    # When supplied, OVERRIDES the static --mask-half-sim-prob /
    # profile mask_half_sim_prob value.
    p.add_argument("--mask-half-sim-prob-schedule", type=str, default=None,
                   help="Lane V-V2 annealing schedule. Format: "
                        "'start,ramp_start_frac,end,ramp_end_frac'. Example: "
                        "'0,0.3,1.0,0.7' = full-frame warmup for first 30%% "
                        "of epochs, linear ramp to half-frame between 30%% "
                        "and 70%%, half-frame for last 30%%. Overrides the "
                        "static --mask-half-sim-prob value when supplied.")

    # KL-like distillation is explicit-scope only. Historical primary/full-
    # scorer KL collapsed PoseNet; train_renderer only permits the SegNet
    # auxiliary path, and only when --kl-distill-scope=segnet_aux is present
    # through CLI/profile. Positive weights without scope fail closed.
    p.add_argument("--kl-distill-weight", type=float, default=None,
                   help="Auxiliary KL distill loss weight (default: from profile, "
                        "else 0.0 = off). Requires --kl-distill-scope=segnet_aux "
                        "when >0.")
    p.add_argument("--kl-distill-temperature", type=float, default=None,
                   help="Softmax temperature for KL distillation (default: from "
                        "profile, else 2.0).")
    p.add_argument("--kl-distill-scope", type=str, default=None,
                   choices=["none", "segnet_aux", "primary_scorer"],
                   help="Required scope for KL-like auxiliaries. train_renderer "
                        "permits only 'segnet_aux'; 'primary_scorer' is "
                        "forensic-only and blocked here.")
    p.add_argument("--allow-high-kl-weight-forensic", action="store_true", default=None,
                   help="Explicit forensic opt-in for kl_distill_weight >= 0.1. "
                        "High-weight KL is non-promotable until scale review "
                        "and exact CUDA component gates exist.")
    # Lane J-JBL (Jack Cycle 1 TOP-1, 2026-04-28): swap the SegNet KL distill
    # auxiliary for Jaccard Metric Loss + Boundary Label Smoothing per
    # Wang et al. NeurIPS 2023 (arXiv 2302.05666). loss_mode="jbl" replaces
    # kl_distill_segnet_only with combined_jbl_distill_loss in the auxiliary
    # add. loss_mode="kl" selects SegNet-aux KL when the explicit scope and
    # positive weight are also present.
    # Default None lets the profile resolver win; effective default is
    # "standard" so unprofiled CLI runs are byte-identical to the legacy
    # path. See tac.losses_jbl for the math + .omx/research/jack_skunkworks
    # _segnet_rate_research_20260428.md §S1 for the wedge attribution.
    p.add_argument("--loss-mode", type=str, default=None,
                   choices=["standard", "kl", "jbl", "logit_margin"],
                   help="Auxiliary distillation loss family. 'kl' = "
                        "Hinton SegNet-aux KL when kl_distill_scope=segnet_aux. "
                        "'jbl' = Lane J-JBL Jaccard + Boundary Label Smoothing. "
                        "'logit_margin' = Lane 19 fragility-weighted CE that "
                        "concentrates gradient on boundary pixels where "
                        "top1-top2 logit gap < threshold (Fridrich UNIWARD "
                        "applied to segmentation).")
    p.add_argument("--boundary-weight", type=float, default=None,
                   help="Lane J-JBL: per-pixel weight multiplier on boundary "
                        "pixels inside the BLS+CE channel (default 3.0).")
    p.add_argument("--bls-smoothing", type=float, default=None,
                   help="Lane J-JBL: label-smoothing epsilon applied at "
                        "boundary pixels only (default 0.1).")
    # Lane 19 (SegNet logit-margin boundary loss).
    # SegNet score = argmax disagreement rate; only the top-2 logit ordering
    # matters per pixel. Standard CE wastes capacity on confident pixels;
    # Lane 19 weights CE by `(threshold - margin) / threshold` so confident
    # pixels (margin >= threshold) contribute zero loss and boundary pixels
    # (margin < threshold) contribute proportional to their ambiguity.
    # Fridrich UNIWARD principle (steganalysis): hide error in detector
    # blind spots; here, the SegNet's boundaries ARE the detector blind
    # spots (per CLAUDE.md "Exact scorer architectures").
    #
    # Wired as an AUXILIARY loss alongside scorer_loss (NOT a replacement)
    # so confident-wrong pixels are still caught by the underlying scorer.
    # Per council memo .omx/research/council_lane_19_logit_margin_design_20260430.md.
    p.add_argument("--logit-margin-weight", type=float, default=None,
                   help="Lane 19: scalar weight for the SegNet logit-margin "
                        "auxiliary loss. Default 0.0 = off (byte-identical "
                        "to Lane G v3). Profile LANE_19_LOGIT_MARGIN sets 0.1.")
    p.add_argument("--logit-margin-threshold", type=float, default=None,
                   help="Lane 19: top1-top2 logit-margin cutoff above which "
                        "fragility weight = 0 (confident; no learning signal). "
                        "Default 1.0. Typical range 0.5-2.0 depending on logit "
                        "scale.")
    # Lane PS (per-class SegNet weighting). Per memory
    # `project_research_survey_20260420` — research-grade, never
    # implemented. SegNet predicts 5-class segmentation in canonical contest
    # order (road, lane_markings, undrivable, movable, my_car); classes with
    # different scorer sensitivity are averaged uniformly today. When supplied,
    # the per-pixel SegNet contribution
    # of `scorer_loss` / `scorer_loss_cached` AND the auxiliary
    # `kl_distill_segnet_only` are multiplied by the weight at each
    # pixel's GT-argmax class. The CSV is parsed via
    # tac.losses.parse_class_weights_csv (5 floats, non-negative, not all
    # zero) so typo'd values fail loud at boot.
    p.add_argument("--segnet-class-weights", type=str, default=None,
                   help="Lane PS: CSV of 5 per-class weights for the "
                        "SegNet loss term and the auxiliary KL distill "
                        "(e.g., '1,5,5,1,1' to boost lane + boundary "
                        "classes). Default None / empty = uniform "
                        "weighting (byte-identical to baseline).")
    p.add_argument("--segmentation-surrogate", type=str,
                   default="soft_cosine",
                   choices=["soft_cosine", "fisher_rao"],
                   help="SegNet scorer-loss training surrogate. Default "
                        "'soft_cosine' preserves legacy behavior. "
                        "'fisher_rao' enables the opt-in normalized "
                        "Fisher-Rao simplex metric from the Hilbert/"
                        "information-geometry workstream.")
    p.add_argument("--segmentation-temperature", type=float, default=1.0,
                   help="Temperature for the SegNet training surrogate. "
                        "Cached-GT mode supports only 1.0 because it stores "
                        "softmax probabilities, not raw teacher logits.")
    p.add_argument("--fisher-rao-eps", type=float, default=1e-6,
                   help="Numerical epsilon for --segmentation-surrogate "
                        "fisher_rao. Must be in (0, 1e-3).")

    p.add_argument("--subsample", type=int, default=4,
                   help="Train on 1/N of pairs per epoch")
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--segnet-weight", type=float, default=None,
                   help="Weight for SegNet term in scorer_loss")
    # R36 fix: store_true with default=True is dead code (the flag is True
    # before parsing, so --use-qat is a no-op). Use set_defaults for the
    # default and let --use-qat / --no-qat both work as explicit toggles.
    p.add_argument("--use-qat", action="store_true",
                   help="Enable FP4 QAT (default: on; --no-qat to disable)")
    p.add_argument("--no-qat", dest="use_qat", action="store_false")
    p.set_defaults(use_qat=True)

    # R-FP4-fix CLI: opt-in robustness flags for FP4 QAT. Defaults match the
    # legacy behaviour to avoid surprising old runs. Set all three for residual
    # heads (Cool-Chic / C3) to close the float-FP4 gap.
    # 2026-04-29 silent-default audit fix: argparse default='default' silently
    # overrode profile's fp4_codebook='residual' in 14 profiles (1 profile sets
    # 'default', 14 set 'residual'). Pass None so _resolve consults profile,
    # falling back to 'default' only when neither CLI nor profile set it.
    # Without this fix, every 'residual'-profile training run was silently
    # using the wrong codebook (potentially 4x worse small-magnitude weight
    # preservation, which is exactly the bug the codebook was meant to fix).
    p.add_argument("--fp4-codebook", choices=("default", "residual"),
                   default=None,
                   help="FP4 codebook: 'default' = mask2mask uniform spacing, "
                        "'residual' = denser-near-zero (4x better small-mag preservation). "
                        "Use 'residual' for renderers with a correction head.")
    p.add_argument("--fp4-stochastic", action="store_true", default=False,
                   help="Stochastic rounding during training (unbiased dither). "
                        "Auto-disabled in eval mode for inflate determinism.")
    p.add_argument("--fp4-robust-scale", action="store_true", default=False,
                   help="Per-block scale via p99.5 quantile instead of max(|w|). "
                        "Protects small-magnitude tail from outlier-driven collapse.")

    # R39 fix: register CLI flags for the 9 R38-added profile resolvers so
    # operators can override from CLI AND the arity validator can enforce
    # them. dest=None on store_true means default to None (not False) so
    # _resolve() can detect "user did not pass" vs "user passed False".
    p.add_argument("--use-dilation", action="store_true", default=None,
                   help="Enable Conv2d dilation (profile-driven)")
    p.add_argument("--padding-mode", type=str, default=None,
                   choices=["zeros", "reflect", "replicate", "circular"],
                   help="Conv2d padding mode (profile default 'zeros'; renderer profiles use 'replicate')")
    p.add_argument("--use-dsconv", action="store_true", default=None,
                   help="Depthwise-separable convolutions")
    p.add_argument("--use-ghost", action="store_true", default=None,
                   help="Lane GH: GhostConv2d (Han et al. CVPR 2020) — halves "
                        "renderer params via primary conv + cheap depthwise ghost branch")
    p.add_argument("--use-zoom-flow", action="store_true", default=None,
                   help="GREEN profile: 4ch MotionPredictor + RadialZoomWarp")
    # Lane S (2026-04-27): self-compressing renderer (Szabolcs 2301.13142).
    # When True, AFTER build_renderer the eligible Conv2d layers are swapped
    # with SelfCompressingConv2d (per-channel learnable bit-depth via STE).
    # The Lagrangian rate penalty drives bit-depth toward target during
    # training. Protected layers (renderer.head, motion.head, FiLM, fuse_conv)
    # stay FP32 per Lane F's scorer-sensitivity finding.
    p.add_argument("--use-self-compress-codec", action="store_true", default=None,
                   help="Lane S: swap bulk Conv2d for SelfCompressingConv2d "
                        "with per-channel learnable bit-depth.")
    # Lane SG (2026-04-28 council EUREKA #5): re-scope which layers stay FP32
    # based on which scorer we are protecting. SegNet contributes 100x seg
    # vs sqrt(10*pose) for PoseNet — at our operating point SegNet is 2-5x
    # more impactful per unit-distortion. Lane SG protects SegNet-relevant
    # layers (decoder out_conv, decode_head, class affines) instead of the
    # PoseNet-relevant layers (FiLM, motion.head, renderer.head).
    p.add_argument("--protected-pattern-set", type=str, default="posenet_prior",
                   choices=["posenet_prior", "segnet_prior"],
                   help="Lane SG: which scorer-prior protection list to use "
                        "for SC codec swap. 'posenet_prior' = default "
                        "(SC_PROTECTED_NAME_PATTERNS); 'segnet_prior' = "
                        "Lane SG (SC_SEGNET_PROTECTED_NAME_PATTERNS).")
    p.add_argument("--self-compress-init-bits", type=float, default=None,
                   help="Initial bit-depth for SC layers (default 8.0 = full "
                        "precision; rate penalty anneals toward target).")
    p.add_argument("--self-compress-target-bits", type=float, default=None,
                   help="Target average bit-depth for SC weights (default 2.5). "
                        "Lagrangian penalty kicks in above this.")
    p.add_argument("--self-compress-lambda-start", type=float, default=None,
                   help="Initial Lagrangian rate-penalty multiplier (default 0.0).")
    p.add_argument("--self-compress-lambda-end", type=float, default=None,
                   help="Final Lagrangian rate-penalty multiplier (default 1.0).")
    p.add_argument("--self-compress-lambda-ramp-start-frac", type=float, default=None,
                   help="Fraction of training before lambda ramp begins (0.3 default).")
    # Lane S V2 (auto-warmup): replace the hand-coded ramp_start_frac
    # with the SAGA-style scorer-loss convergence detector
    # (`tac.training.scorer_loss_convergence_detector`). When enabled,
    # the per-epoch scorer loss is fed to the detector; once the
    # OLS-slope plateau is detected the Lagrangian rate ramp is allowed
    # to begin (instead of waiting for the static fraction). Default
    # OFF (False) for backward compat — every existing profile keeps
    # its --self-compress-lambda-ramp-start-frac semantics.
    p.add_argument("--auto-warmup-lambda", action="store_true", default=False,
                   help="Lane S V2: auto-detect scorer-loss convergence "
                        "and start the Lagrangian rate ramp THEN, instead "
                        "of at the hand-coded "
                        "--self-compress-lambda-ramp-start-frac fraction. "
                        "Uses tac.training.scorer_loss_convergence_detector "
                        "(OLS-slope plateau detector with a 50-epoch "
                        "sliding window, slope tolerance 1e-4 / epoch, "
                        "min_warmup_epochs floor of 50).")
    p.add_argument("--auto-warmup-window", type=int, default=50,
                   help="Lane S V2: sliding window size for the auto-"
                        "warmup detector (default 50 epochs).")
    p.add_argument("--auto-warmup-slope-tol", type=float, default=1e-4,
                   help="Lane S V2: slope tolerance (loss-units per epoch) "
                        "for the auto-warmup plateau test (default 1e-4).")
    p.add_argument("--auto-warmup-min-epochs", type=int, default=50,
                   help="Lane S V2: minimum epoch count before the "
                        "auto-warmup detector is allowed to fire (default "
                        "50). Hard floor; even on a synthetically-flat "
                        "loss curve, the rate ramp cannot start before "
                        "this epoch.")
    # Codex R5-2 Finding #2 (2026-04-27): the 5 build_renderer arch knobs below
    # were silent dead-resolvers — `getattr(args, "blend_mode", "scalar")` at
    # the build site read the wrong default on every run because parse_args
    # never copied the profile value into args. Same bug class as pose_dim
    # (commit 0746a803). Wire them through with the canonical _resolve()
    # pattern + matching CLI flags so operators can override at the command
    # line AND the dead-resolver scanner clears them.
    p.add_argument("--blend-mode", type=str, default=None,
                   choices=["scalar", "spatial", "none"],
                   help="MaskRenderer blend mode: scalar=learned alpha, "
                        "spatial=per-pixel, none=disable. Profile default "
                        "varies (mask_renderer / wavelet / dilated profiles "
                        "all use 'spatial').")
    p.add_argument("--noise-mode", type=str, default=None,
                   choices=["deterministic", "shared", "independent"],
                   help="MaskRenderer noise injection mode "
                        "(profile default 'deterministic').")
    p.add_argument("--motion-type", type=str, default=None,
                   choices=["depth_aware", "learned_cnn", "analytical", "none"],
                   help="MotionPredictor variant. Profile default varies "
                        "('depth_aware' for renderer profiles, 'learned_cnn' "
                        "elsewhere).")
    p.add_argument("--beta-start", type=float, default=None,
                   help="Diffusion teacher noise schedule lower bound "
                        "(diffusion_teacher variant only; default 1e-4).")
    p.add_argument("--beta-end", type=float, default=None,
                   help="Diffusion teacher noise schedule upper bound "
                        "(diffusion_teacher variant only; default 0.02).")
    p.add_argument("--use-texture-loss", action="store_true", default=None,
                   help="Fridrich UNIWARD: hide errors in textured regions")
    p.add_argument("--texture-loss-weight", type=float, default=None)
    p.add_argument("--use-linf-penalty", action="store_true", default=None,
                   help="Fridrich square root law: spread errors")
    p.add_argument("--linf-weight", type=float, default=None)
    p.add_argument("--use-markov-loss", action="store_true", default=None,
                   help="Fridrich HUGO: preserve local gradient statistics")
    p.add_argument("--markov-weight", type=float, default=None)
    # Yousfi #5 (council 5/0 vote 2026-04-26): ScanNet-style spatial uncertainty
    # weighting via inverse-SegNet-entropy. Profile keys used by WILDE/SHIRAZ
    # (weight 0.05) and DEN (weight 0.02 — kept light because DEN already runs
    # KL distill which subsumes ~80% of the same signal).
    p.add_argument("--use-uncertainty-loss", action="store_true", default=None,
                   help="Yousfi #5: weight L1 by inverse-SegNet-entropy (focus "
                        "loss on pixels SegNet is most confident about)")
    p.add_argument("--uncertainty-loss-weight", type=float, default=None,
                   help="Weight applied to segnet_uncertainty_weighted_loss "
                        "(default 0.0 = disabled). Profiles set 0.02-0.05.")
    p.add_argument("--dct-quant-weight", type=float, default=None,
                   help="Fridrich council #1: JPEG-Q-table-weighted DCT-domain "
                        "residual loss (0=disabled). Penalises low-freq leak; "
                        "lets renderer hide error in high-freq DCT bins the "
                        "scorer cannot see. Recommended 0.5 (similar to "
                        "texture_loss_weight).")
    p.add_argument("--use-per-class-weights", action="store_true", default=None)
    p.add_argument("--use-swa", action="store_true", default=None)
    p.add_argument("--even-frame-skip-seg", action="store_true", default=False,
                   help="Trick 3: skip SegNet loss when frame_t1 is even-indexed "
                   "(SegNet only evaluates odd frames in the scorer)")
    p.add_argument("--frequency-loss-weight", type=float, default=None,
                   help="Trick 2: wavelet frequency-domain loss weight (0=disabled)")
    # Lane T2-XPRED: Tuna-2 style x-prediction reconstruction objective.
    # Defaults are resolved after profile loading so named profiles can
    # activate the lane while unprofiled CLI runs default to disabled.
    p.add_argument("--use-t2-xpred-loss", action="store_true", default=None,
                   help="Lane T2-XPRED: replace the primary reconstruction "
                        "loss with x_prediction_loss.")
    p.add_argument("--t2-xpred-sigma", type=float, default=None,
                   help="Lane T2-XPRED: constant sigma for x_prediction_loss "
                        "(default 1.0).")
    p.add_argument("--t2-xpred-weighting", type=str, default=None,
                   choices=["v", "x"],
                   help="Lane T2-XPRED weighting mode: 'v' or 'x' "
                        "(default 'v').")
    # Lane T2-MASK: training-time bottleneck feature masking.
    p.add_argument("--use-t2-mask", action="store_true", default=None,
                   help="Lane T2-MASK: enable deterministic bottleneck feature "
                        "masking during training.")
    p.add_argument("--t2-mask-p", type=float, default=None,
                   help="Lane T2-MASK: probability of applying masking to a "
                        "training batch (default 0.5).")
    p.add_argument("--t2-mask-ratio", type=float, default=None,
                   help="Lane T2-MASK: spatial position mask ratio when "
                        "masking applies (default 0.15).")
    p.add_argument("--t2-mask-apply-fraction", type=float, default=None,
                   help="Lane T2-MASK: apply only in this final fraction of "
                        "training progress (default 0.4).")
    # Lane T2-DROP: encoder-free renderer ablation. This is informational-only
    # and not a stacking candidate; it replaces mask-encoder feature tensors
    # with zeros while preserving the model/state-dict structure.
    p.add_argument("--no-mask-encoder", action="store_true", default=False,
                   help="Lane T2-DROP informational-only ablation: zero the "
                        "mask encoder outputs instead of using encoded mask "
                        "features. Not a stacking candidate.")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resize chain in scorer loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")

    # Data
    p.add_argument("--precomputed", type=str,
                   default=str(_repo / "experiments" / "precomputed_local"),
                   help="Dir with gt_frames.pt (skip video decode)")
    p.add_argument("--video", type=str,
                   default=str(_upstream / "videos" / "0.mkv"),
                   help="GT video path (used if no precomputed)")
    p.add_argument("--mask-batch-size", type=int, default=4,
                   help="Batch size for SegNet mask extraction")

    # Resilience
    # 2026-04-29 silent-default audit fix: argparse default=0 (no limit)
    # silently overrode profile's wall_clock_timeout=39600 (11h cap) in 2
    # profiles. A modal lane configured for 11h timeout would silently get
    # no timeout — pre-running budget on dead-ended experiments. Pass None
    # so _resolve consults profile.
    p.add_argument("--wall-clock-timeout", type=int, default=None,
                   help="Max wall-clock seconds before emergency save + clean exit (0=no limit)")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to training_state_*.pt checkpoint to resume from")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible experiment replay")
    p.add_argument("--deterministic", action="store_true", default=None,
                   help="Use deterministic torch algorithms where available")
    p.add_argument("--nondeterministic", dest="deterministic", action="store_false",
                   help="Allow nondeterministic kernels for speed")
    # Codex R-Lane-D-Issue1 (2026-04-27): explicit override for pose_dim that
    # beats BOTH the profile resolver AND any checkpoint arch_meta autodetect.
    # Use this when you intentionally want to retrain a different pose_dim arch
    # from scratch (NOT a resume): pass the override + leave --resume-from
    # unset. With a resume, --force-pose-dim mismatching the saved arch will
    # raise a clear shape-mismatch error from load_state_dict — that is the
    # intended behaviour (silent shape drift wasted 1.2h on DEN; we never
    # accept a quiet shape mismatch again).
    p.add_argument("--force-pose-dim", type=int, default=None,
                   help="Override pose_dim, beating profile + checkpoint "
                        "arch_meta. Default unset = profile/checkpoint wins.")

    # Output
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--output-dir", type=str,
                   default=str(_repo / "experiments" / "postfilter_weights"))
    p.add_argument("--device", type=str, default=None)

    # Auth eval (CLAUDE.md non-negotiable: "Auth eval EVERYWHERE")
    # Council R2 (2026-04-26): default flipped to TRUE because opt-in violates
    # the CLAUDE.md non-negotiable. Every memory entry from the past 2 weeks
    # has a "wasted GPU because we didn't auth-eval" footnote. Use
    # --no-auth-eval-on-best for the rare smoke-test case.
    p.add_argument("--auth-eval-on-best", action="store_true", default=True,
                   help="At end of training, run CUDA auth eval against the best "
                        "fp4 checkpoint. DEFAULT TRUE per CLAUDE.md non-negotiable. "
                        "The proxy fp4_scorer is a TRAINING SIGNAL only — proxy-auth "
                        "gap can be 100-350x even on CUDA-CUDA (LANE-B 2026-04-26). "
                        "The auth result is the ONLY trustworthy score. Result is "
                        "saved as <out_dir>/auth_eval_on_best.json.")
    p.add_argument("--no-auth-eval-on-best", dest="auth_eval_on_best", action="store_false",
                   help="Skip the auth eval (smoke tests / dry runs only). Using this "
                        "flag means the run produces NO authoritative score — the "
                        "proxy fp4_scorer alone is not a measurement.")
    p.add_argument("--auth-eval-masks", type=str, default=None,
                   help="Path to masks.mkv for auth eval. Required with --auth-eval-on-best.")
    p.add_argument("--auth-eval-poses", type=str, default=None,
                   help="Path to optimized_poses.bin for FiLM models. If the trained "
                        "renderer has pose_dim>0, this MUST be passed (auth_eval_renderer.py "
                        "hard-fails on FiLM + no poses, see feedback_film_eval_no_poses_critical).")
    p.add_argument("--qfaithful-training-poses", type=str, default=None,
                   help="Q-FAITHFUL only: optimized pose stream consumed during "
                        "training. Required for variant=quantizr_faithful; the "
                        "shim refuses zero-pose fallback so training and inflate "
                        "share the same FiLM-pose contract.")
    p.add_argument("--auth-eval-upstream-dir", type=str,
                   default=str(_upstream),
                   help="Path to upstream/ for auth eval (defaults to repo upstream).")

    # Lane W (2026-04-27): hard-pair weighted self-compression. The
    # per-pair weight tensor is produced by experiments/profile_pair_sensitivity.py
    # and is a (N_pairs,) float32 tensor where the top-K hardest pairs (by
    # 100*seg + sqrt(10*pose) contribution) are weighted ``--hard-weight``
    # and the rest stay at 1.0. The per-pair scalar multiplies BOTH the
    # scorer loss AND any added Lagrangian rate penalty (Lane S SC codec)
    # so the per-channel learnable bit-depth allocation responds to
    # hard-pair gradient. See feedback_overfit_is_the_goal +
    # feedback_posenet_tracking + feedback_curriculum_must_use_full_score.
    p.add_argument("--pair-loss-weights", type=str, default=None,
                   help="Path to a (N_pairs,) float32 tensor produced by "
                        "experiments/profile_pair_sensitivity.py. When set, per-step "
                        "loss is scaled by pair_loss_weights[pair_idx_int]. The shape "
                        "MUST match the dataset's n_total pairs (typically 600).")
    p.add_argument("--pair-weights-path", type=str, default=None,
                   help="Lane WC alias for --pair-loss-weights. Intended for "
                        "results/lane_wc/pair_weights.pt produced by "
                        "experiments/fit_curator_outlier_weights.py from "
                        "SegNet feature geometry, not pair training loss.")

    # Lane W-V2 (2026-04-27): make --pair-loss-weights LEARNABLE. The per-
    # pair weights become a TRAINABLE parameter group (LearnablePairWeights)
    # whose values are warm-started from --pair-loss-weights. A Lagrangian
    # rate penalty drives sum(weights) toward N_pairs (mean=1) so the loss
    # scale stays comparable to the unweighted run while the optimiser
    # redistributes weight mass to the hardest pairs. See
    # tac.learnable_pair_weights and project_arbitrary_vs_learnable_taxonomy.
    p.add_argument("--learnable-pair-weights", action="store_true",
                   default=False,
                   help="Lane W-V2: turn the --pair-loss-weights tensor into "
                        "a LEARNABLE parameter group (LearnablePairWeights). "
                        "Requires --pair-loss-weights as the warm-start. "
                        "Adds a Lagrangian rate penalty so sum(weights) "
                        "≈ N_pairs (mean=1).")
    p.add_argument("--learnable-pair-weights-lr", type=float, default=1e-3,
                   help="Lane W-V2: learning rate for the LearnablePairWeights "
                        "parameter group. Default 1e-3 (small — these are "
                        "scalars, not weight tensors).")
    p.add_argument("--learnable-pair-weights-rate-lambda", type=float,
                   default=1e-4,
                   help="Lane W-V2: Lagrangian multiplier on the "
                        "sum(weights)≈N_pairs constraint. Small (1e-4) so "
                        "the constraint doesn't dominate the primary loss.")

    # Lane PS-V2 (2026-04-27): make --segnet-class-weights LEARNABLE. The
    # 5-class weight vector becomes a softmax-parameterised parameter
    # (LearnableClassWeights). A Lagrangian penalty equalises per-class
    # contribution variance — Pareto-optimal under the score formula. See
    # tac.learnable_class_weights and project_arbitrary_vs_learnable_taxonomy.
    p.add_argument("--learnable-segnet-class-weights", action="store_true",
                   default=False,
                   help="Lane PS-V2: replace the static --segnet-class-weights "
                        "CSV with a LEARNABLE softmax-parameterised "
                        "5-vector. When set, --segnet-class-weights provides "
                        "the warm-start; the optimiser then equalises "
                        "per-class distortion variance via a Lagrangian "
                        "penalty.")
    p.add_argument("--learnable-segnet-class-weights-lr", type=float,
                   default=1e-2,
                   help="Lane PS-V2: learning rate for the "
                        "LearnableClassWeights parameter group.")
    p.add_argument("--learnable-segnet-class-weights-var-lambda", type=float,
                   default=1.0,
                   help="Lane PS-V2: Lagrangian multiplier on the per-class "
                        "distortion variance penalty (equalisation term).")

    args = p.parse_args(argv)

    # Apply profile defaults, then CLI overrides
    profile_vals = {}
    if args.profile:
        profile_vals = dict(PROFILES[args.profile])

    def _resolve(cli_val, profile_key, default):
        if cli_val is not None:
            return cli_val
        return profile_vals.get(profile_key, default)

    # Resolve variant from CLI, profile, or default
    args.variant = _resolve(args.variant, "variant", "mask_renderer")

    # R38 fix: profiles canonically use "base_ch" key; "hidden" is a legacy
    # alias used by older postfilter profiles. Try the canonical key first.
    args.base_ch = _resolve(args.base_ch, "base_ch",
                            profile_vals.get("hidden", 36))
    args.mid_ch = _resolve(args.mid_ch, "mid_ch", 60)
    args.embed_dim = _resolve(args.embed_dim, "embed_dim", 6)
    args.motion_hidden = _resolve(args.motion_hidden, "motion_hidden", 32)
    args.depth = _resolve(args.depth, "depth", 1)
    # R38 fix: WILDE/SHIRAZ/GREEN profiles set use_dilation, padding_mode,
    # use_dsconv, use_zoom_flow but train_renderer.py's resolver dropped
    # them. Silent wrong-arch training. Add resolvers.
    args.use_dilation = _resolve(getattr(args, "use_dilation", None),
                                  "use_dilation", False)
    args.padding_mode = _resolve(getattr(args, "padding_mode", None),
                                  "padding_mode", "zeros")
    args.use_dsconv = _resolve(getattr(args, "use_dsconv", None),
                                "use_dsconv", False)
    # Lane GH 2026-04-27: GhostConv2d arch flag — same dead-resolver pattern
    # as use_dsconv. Without this resolver, profiles that set use_ghost=True
    # would silently train with use_ghost=False (288K dense conv instead of
    # 144K ghost conv) and the param-count smoke check would fail.
    args.use_ghost = _resolve(getattr(args, "use_ghost", None),
                                "use_ghost", False)
    args.use_zoom_flow = _resolve(getattr(args, "use_zoom_flow", None),
                                   "use_zoom_flow", False)
    args.use_saug_v2 = _resolve(
        getattr(args, "use_saug_v2", None),
        "use_saug_v2", False,
    )
    args.saug_v2_redraw_fraction = _resolve(
        getattr(args, "saug_v2_redraw_fraction", None),
        "saug_v2_redraw_fraction", 0.05,
    )
    args.saug_v2_high_sigma_min = _resolve(
        getattr(args, "saug_v2_high_sigma_min", None),
        "saug_v2_high_sigma_min", 80.0,
    )
    args.saug_v2_high_sigma_max = _resolve(
        getattr(args, "saug_v2_high_sigma_max", None),
        "saug_v2_high_sigma_max", 2000.0,
    )
    args.saug_v2_normal_sigma_min = _resolve(
        getattr(args, "saug_v2_normal_sigma_min", None),
        "saug_v2_normal_sigma_min", 0.5,
    )
    args.saug_v2_normal_sigma_max = _resolve(
        getattr(args, "saug_v2_normal_sigma_max", None),
        "saug_v2_normal_sigma_max", 80.0,
    )
    # Lane CG resolver (calibrated positional encoding orphan wire-up).
    # Importing the module triggers its monkey-patch of MaskRenderer so
    # downstream build_renderer accepts use_calibrated_positional_encoding.
    args.use_calibrated_positional_encoding = _resolve(
        getattr(args, "use_calibrated_positional_encoding", None),
        "use_calibrated_positional_encoding", False,
    )
    if args.use_calibrated_positional_encoding:
        # Trigger _patch_renderer_mask_renderer() side effect.
        from tac.contrib import calibrated_positional_encoding  # noqa: F401
    # Lane MAE-V resolvers (mirror SAUG-V2 wiring pattern).
    args.use_mae_mask_aug = _resolve(
        getattr(args, "use_mae_mask_aug", None),
        "use_mae_mask_aug", False,
    )
    args.mae_mask_ratio = _resolve(
        getattr(args, "mae_mask_ratio", None),
        "mae_mask_ratio", 0.25,
    )
    args.mae_patch_size = _resolve(
        getattr(args, "mae_patch_size", None),
        "mae_patch_size", 16,
    )
    # Lane S: SC codec resolution (2026-04-27)
    args.use_self_compress_codec = _resolve(
        getattr(args, "use_self_compress_codec", None),
        "use_self_compress_codec", False,
    )
    args.self_compress_init_bits = _resolve(
        getattr(args, "self_compress_init_bits", None),
        "self_compress_init_bits", 8.0,
    )
    args.self_compress_target_bits = _resolve(
        getattr(args, "self_compress_target_bits", None),
        "self_compress_target_bits", 2.5,
    )
    # 2026-04-29 silent-default audit fix: grad_clip default was 1.0 but
    # 2 profiles set 10.0 — the argparse default silently won. Now the CLI
    # default is None so _resolve consults the profile (10.0 when present)
    # before falling back to 1.0.
    args.grad_clip = _resolve(args.grad_clip, "grad_clip", 1.0)
    args.fp4_codebook = _resolve(args.fp4_codebook, "fp4_codebook", "default")
    args.wall_clock_timeout = _resolve(args.wall_clock_timeout, "wall_clock_timeout", 0)
    args.self_compress_lambda_start = _resolve(
        getattr(args, "self_compress_lambda_start", None),
        "self_compress_lambda_start", 0.0,
    )
    args.self_compress_lambda_end = _resolve(
        getattr(args, "self_compress_lambda_end", None),
        "self_compress_lambda_end", 1.0,
    )
    args.self_compress_lambda_ramp_start_frac = _resolve(
        getattr(args, "self_compress_lambda_ramp_start_frac", None),
        "self_compress_lambda_ramp_start_frac", 0.3,
    )
    # Codex R5-2 Finding #2 (2026-04-27): blend_mode / noise_mode / motion_type
    # were dead-resolvers — every getattr(args, "blend_mode", "scalar") at the
    # build sites silently returned the function default, ignoring profile
    # values like "spatial" (mask_renderer/wavelet/dilated profiles all set
    # spatial). Wire them through the standard _resolve() pattern. Defaults
    # match build_renderer's own defaults so unprofiled CLI runs keep working.
    args.blend_mode = _resolve(getattr(args, "blend_mode", None),
                                "blend_mode", "scalar")
    args.noise_mode = _resolve(getattr(args, "noise_mode", None),
                                "noise_mode", "deterministic")
    args.motion_type = _resolve(getattr(args, "motion_type", None),
                                 "motion_type", "learned_cnn")
    # Lane HM orphan wire-up: importing tac.contrib.homography_motion runs
    # _patch_renderer_dispatch(), which adds a homography_analytical case
    # to build_renderer if the renderer.py native dispatch is not yet
    # available. Safe to import unconditionally — the module's patch is
    # idempotent and a no-op when motion_type != 'homography_analytical'.
    if args.motion_type == "homography_analytical":
        from tac.contrib import homography_motion  # noqa: F401
    # Diffusion teacher noise schedule (only consumed by variant=='diffusion_teacher').
    # Defaults match tac.contrib.diffusion_renderer.build_diffusion_teacher.
    args.beta_start = _resolve(getattr(args, "beta_start", None),
                                "beta_start", 1e-4)
    args.beta_end = _resolve(getattr(args, "beta_end", None),
                              "beta_end", 0.02)
    # Lane D council 2026-04-27: pose_dim was being read from profile via
    # getattr() at the build site (line ~928) but never resolved on args,
    # making it silently invisible to anything inspecting the parsed
    # Namespace (tests, logs, snapshots). Same dead-resolver pattern that
    # killed Yousfi #5 uncertainty loss — fix it here so Lane D's required
    # pose_dim=6 (FiLM modulation, baseline arch parity) is observable.
    #
    # Codex R-Lane-D-Issue1 (2026-04-27): --force-pose-dim wins over both
    # profile and CLI-via-getattr. Used to (a) intentionally retrain a
    # different pose_dim arch from scratch, or (b) override a resume that
    # would otherwise honour the checkpoint's arch_meta (which the train()
    # path peeks BEFORE constructing the model, see _peek_checkpoint_arch_meta).
    if getattr(args, "force_pose_dim", None) is not None:
        args.pose_dim = int(args.force_pose_dim)
    else:
        args.pose_dim = _resolve(getattr(args, "pose_dim", None),
                                  "pose_dim", 0)
    # Fridrich inverse-steganalysis losses (WILDE / SHIRAZ "competitive
    # advantage" — were silently disabled by missing resolver):
    args.use_texture_loss = _resolve(getattr(args, "use_texture_loss", None),
                                      "use_texture_loss", False)
    args.texture_loss_weight = _resolve(getattr(args, "texture_loss_weight", None),
                                         "texture_loss_weight", 0.5)
    args.use_linf_penalty = _resolve(getattr(args, "use_linf_penalty", None),
                                      "use_linf_penalty", False)
    args.linf_weight = _resolve(getattr(args, "linf_weight", None),
                                 "linf_weight", 0.01)
    args.use_markov_loss = _resolve(getattr(args, "use_markov_loss", None),
                                     "use_markov_loss", False)
    args.markov_weight = _resolve(getattr(args, "markov_weight", None),
                                   "markov_weight", 0.1)
    # Fridrich council #1 (2026-04-26): JPEG-Q-table-weighted DCT loss.
    # Default 0.0 = disabled (zero overhead — call site is gated).
    args.dct_quant_weight = _resolve(getattr(args, "dct_quant_weight", None),
                                      "dct_quant_weight", 0.0)
    # Quantizr council #4 CRITICAL (2026-04-26): KL distillation T=2.0 on
    # SegNet was DEAD CODE because the resolver wasn't reading from profile.
    # DEN profile declared kl_distill_weight=1.0 but argparse default 0.0
    # always won. Every WILDE/SHIRAZ/DEN training run since the KL feature
    # landed has trained without KL distill — Quantizr's #1 SegNet trick.
    # Resolved via standard _resolve() pattern — profile key wins over
    # argparse default unless CLI explicitly passes the flag.
    args.kl_distill_weight = _resolve(getattr(args, "kl_distill_weight", None),
                                       "kl_distill_weight", 0.0)
    args.kl_distill_temperature = _resolve(getattr(args, "kl_distill_temperature", None),
                                            "kl_distill_temperature", 2.0)
    args.kl_distill_scope = _resolve(
        getattr(args, "kl_distill_scope", None),
        "kl_distill_scope",
        "none",
    )
    args.allow_high_kl_weight_forensic = _resolve(
        getattr(args, "allow_high_kl_weight_forensic", None),
        "allow_high_kl_weight_forensic",
        False,
    )
    args.promotion_eligible = _resolve(
        getattr(args, "promotion_eligible", None),
        "promotion_eligible",
        True,
    )
    _VALID_KL_DISTILL_SCOPES = ("none", "segnet_aux", "primary_scorer")
    if args.kl_distill_scope not in _VALID_KL_DISTILL_SCOPES:
        raise SystemExit(
            f"FATAL: --kl-distill-scope={args.kl_distill_scope!r} "
            f"unrecognised; valid: {_VALID_KL_DISTILL_SCOPES}."
        )
    if args.kl_distill_weight < 0.0:
        raise SystemExit(
            f"FATAL: --kl-distill-weight must be >= 0; got "
            f"{args.kl_distill_weight}."
        )
    if args.kl_distill_scope == "primary_scorer":
        raise SystemExit(
            "FATAL: train_renderer never permits primary/full-scorer KL "
            "distillation. Use kl_distill_scope='segnet_aux' for scoped "
            "SegNet auxiliary work, and require exact CUDA archive eval "
            "before any claim."
        )
    if args.kl_distill_weight > 0.0 and args.kl_distill_scope != "segnet_aux":
        raise SystemExit(
            "FATAL: positive kl_distill_weight requires explicit "
            "kl_distill_scope='segnet_aux'. This prevents stale profiles from "
            "silently enabling KL-like auxiliaries from weight alone."
        )
    if (
        args.kl_distill_weight >= 0.1
        and args.promotion_eligible is not False
        and not args.allow_high_kl_weight_forensic
    ):
        raise SystemExit(
            "FATAL: kl_distill_weight >= 0.1 is a high-scale KL configuration. "
            "It is forensic-only after the primary-KL/PoseNet-collapse review. "
            "Use a profile with promotion_eligible=False or pass "
            "--allow-high-kl-weight-forensic for non-promotable research."
        )
    if args.allow_high_kl_weight_forensic:
        args.promotion_eligible = False
    # Lane J-JBL (Jack Cycle 1 TOP-1) resolvers — see CLI add_argument block
    # for context. Default loss_mode "standard" keeps the auxiliary KL
    # distill path byte-identical to Lane G v3 unless a profile / CLI
    # opts in to "jbl".
    args.loss_mode = _resolve(getattr(args, "loss_mode", None),
                              "loss_mode", "standard")
    # 2026-04-28 widening: the original Lane J-JBL validator only allowed
    # ('standard','kl','jbl') which crashed every Lane D / Lane G v3 / Lane V /
    # Lane MAE-V profile (all of which inherit loss_mode='focal_ste' from
    # DILATED_H64_HALF_FRAME). The historical valid set in tac.cli line 112
    # is the source of truth — keep it in sync here.
    _VALID_LOSS_MODES = (
        "standard", "kl", "jbl",
        "temperature", "focal_ste", "kl_distill", "pcgrad",
        "feature_match",
        "posenet_embedding",
        "segnet_kl",
        "logit_margin",
    )
    # NOTE: keep the _VALID_LOSS_MODES tuple ABOVE this comment paren-free.
    # Preflight regex `_VALID_LOSS_MODES \s*=\s*\([^)]*\)` truncates at the
    # first `)`, so any closing paren inside the tuple body — including in
    # comments — masks later entries from the profile-loss-mode-allowlist
    # static scan. Lane M-V3 posenet_embedding + SC++/g_v3 segnet_kl were
    # added 2026-04-29 to fix that allowlist gap.
    if args.loss_mode not in _VALID_LOSS_MODES:
        raise SystemExit(
            f"FATAL: --loss-mode={args.loss_mode!r} unrecognised; "
            f"valid: {_VALID_LOSS_MODES}."
        )
    from tac.kl_config import (
        DistillationPolicyError,
        distillation_policy_sha256,
        normalize_distillation_policy,
    )
    try:
        _distillation_policy = normalize_distillation_policy(args)
        args.distillation_policy_provenance = _distillation_policy.to_provenance()
        args.distillation_policy_sha256 = distillation_policy_sha256(_distillation_policy)
    except DistillationPolicyError as exc:
        raise SystemExit(f"FATAL: invalid distillation policy: {exc}") from exc
    args.boundary_weight = _resolve(getattr(args, "boundary_weight", None),
                                     "boundary_weight", 3.0)
    args.bls_smoothing = _resolve(getattr(args, "bls_smoothing", None),
                                   "bls_smoothing", 0.1)
    # Lane 19 (logit-margin) resolvers. Default 0.0 weight = off (byte-
    # identical to Lane G v3 anchor). Profile LANE_19_LOGIT_MARGIN sets 0.1.
    args.logit_margin_weight = _resolve(
        getattr(args, "logit_margin_weight", None),
        "logit_margin_weight", 0.0,
    )
    args.logit_margin_threshold = _resolve(
        getattr(args, "logit_margin_threshold", None),
        "logit_margin_threshold", 1.0,
    )
    if args.logit_margin_weight < 0.0:
        raise SystemExit(
            f"FATAL: --logit-margin-weight must be >= 0; got "
            f"{args.logit_margin_weight}."
        )
    if args.logit_margin_threshold <= 0.0:
        raise SystemExit(
            f"FATAL: --logit-margin-threshold must be > 0; got "
            f"{args.logit_margin_threshold}."
        )
    # Lane PS (per-class SegNet weighting) resolver. Default sentinel is
    # ``None`` / empty string. Profile key + CLI flag share the standard
    # _resolve() rules (CLI > profile > default). Parsed lazily into a
    # tensor below — at parse time we run the same fail-loud validation
    # as `experiments/optimize_poses.py` so a typo'd CSV trips at boot,
    # not 30 min into training.
    args.segnet_class_weights = _resolve(
        getattr(args, "segnet_class_weights", None),
        "segnet_class_weights", None,
    )
    args.pair_loss_weights = _resolve(
        getattr(args, "pair_loss_weights", None),
        "pair_loss_weights", None,
    )
    args.pair_weights_path = _resolve(
        getattr(args, "pair_weights_path", None),
        "pair_weights_path", None,
    )
    if (
        args.pair_loss_weights is not None
        and args.pair_weights_path is not None
        and args.pair_loss_weights != args.pair_weights_path
    ):
        raise SystemExit(
            "FATAL: --pair-loss-weights and --pair-weights-path both set "
            "to different files; pass only one pair-weight source."
        )
    if args.pair_loss_weights is None:
        args.pair_loss_weights = args.pair_weights_path
    # Parse the CSV into a (5,) float tensor (or None for the disabled
    # path). Stored on args under a non-CLI attribute so the loss call
    # sites can fetch it without re-parsing every iteration.
    from tac.losses import parse_class_weights_csv as _parse_pcw
    args._segnet_class_weights_tensor = _parse_pcw(
        args.segnet_class_weights, num_classes=5,
    )
    from tac.losses import segnet_surrogate_per_pixel as _segnet_surrogate_probe
    # Fail bad Fisher-Rao/temperature config at boot. A one-pixel probe keeps
    # this scorer-free and deterministic while reusing the canonical validator.
    _probe_logits = torch.zeros(1, 5, 1, 1)
    _segnet_surrogate_probe(
        _probe_logits,
        _probe_logits,
        surrogate=args.segmentation_surrogate,
        temperature=args.segmentation_temperature,
        fisher_rao_eps=args.fisher_rao_eps,
    )
    if args._segnet_class_weights_tensor is not None:
        print(
            f"[lane-ps] SegNet per-class weights ACTIVE: "
            f"{args._segnet_class_weights_tensor.tolist()}",
            flush=True,
        )
    if args.segmentation_surrogate != "soft_cosine":
        print(
            f"[hilbert] SegNet segmentation surrogate ACTIVE: "
            f"{args.segmentation_surrogate} "
            f"(temperature={args.segmentation_temperature}, "
            f"fisher_rao_eps={args.fisher_rao_eps})",
            flush=True,
        )
    # Yousfi #5 council wiring (2026-04-26): WILDE/SHIRAZ/DEN profiles set
    # use_uncertainty_loss=True with weights 0.02-0.05 but train_renderer.py
    # had no resolver — feature was DEAD CODE in profile (same class of bug
    # as KL distill above). Per Fridrich R2 C1 audit, wire it through.
    # See module-level note in tac.losses.segnet_uncertainty_weighted_loss
    # for caveats (SegNet entropy is a proxy, not absolute uncertainty).
    args.use_uncertainty_loss = _resolve(getattr(args, "use_uncertainty_loss", None),
                                          "use_uncertainty_loss", False)
    args.uncertainty_loss_weight = _resolve(getattr(args, "uncertainty_loss_weight", None),
                                             "uncertainty_loss_weight", 0.0)
    # Codex R-Lane-D-Issue1 (2026-04-27, INCIDENTAL FIX): the call site at
    # line ~1614 reads args.uncertainty_loss_floor but this attribute was
    # NEVER resolved on args (no CLI flag, no resolver). Profiles like
    # WILDE/SHIRAZ/DEN/Lane D set uncertainty_loss_floor=0.1, which silently
    # never reached the helper — function default would have won at runtime
    # (and AttributeError'd before that since the resolver was missing).
    # Wire it through with the same _resolve() pattern.
    args.uncertainty_loss_floor = _resolve(getattr(args, "uncertainty_loss_floor", None),
                                            "uncertainty_loss_floor", 0.1)
    # Yousfi #3 (use_variance_noise / variance_noise_*) — RE-ENABLED on
    # 2026-04-26 after Fridrich R2 C2 fix: the box-filter variance estimator
    # has been replaced with an un-decimated Daubechies-8 sub-band energy
    # map (see tac.wavelet_variance.wavelet_variance_map). This is the
    # UNIWARD-correct construction per Holub & Fridrich 2014 §III.B.
    # Profiles default to mode='wavelet_db4'; the legacy 'box' / 'variance'
    # mode is still accepted for A/B comparison ONLY.
    args.use_variance_noise = _resolve(getattr(args, "use_variance_noise", None),
                                        "use_variance_noise", False)
    args.variance_noise_weight = _resolve(getattr(args, "variance_noise_weight", None),
                                           "variance_noise_weight", 0.1)
    args.variance_noise_base_std = _resolve(getattr(args, "variance_noise_base_std", None),
                                             "variance_noise_base_std", 2.0)
    args.variance_noise_kernel = _resolve(getattr(args, "variance_noise_kernel", None),
                                           "variance_noise_kernel", 8)
    args.variance_noise_mode = _resolve(getattr(args, "variance_noise_mode", None),
                                         "variance_noise_mode", "wavelet_db4")
    args.use_per_class_weights = _resolve(getattr(args, "use_per_class_weights", None),
                                           "use_per_class_weights", False)
    args.use_swa = _resolve(getattr(args, "use_swa", None), "use_swa", False)
    # Half-frame mask simulation (Lane D2). When enabled, training periodically
    # replaces mask_t with inverse_warp(mask_t1, analytical_zoom[k]) so the
    # renderer learns the same distribution it sees at inflate when the
    # archive ships only odd-frame masks.
    args.mask_half_sim_prob = _resolve(
        getattr(args, "mask_half_sim_prob", None),
        "mask_half_sim_prob", 0.0,
    )
    # Lane V-V2 (2026-04-27): optional annealing schedule for the half-frame
    # warp probability. When set, the per-epoch warp prob is computed via
    # mask_half_sim_prob_for_epoch() instead of using the static value.
    # CLI override path: --mask-half-sim-prob-schedule "S,RS,E,RE" parses
    # into the 4-key dict below. Profile path: the dict is read directly
    # from PROFILES[<name>]["mask_half_sim_prob_anneal"].
    cli_anneal_str = getattr(args, "mask_half_sim_prob_schedule", None)
    if cli_anneal_str:
        parts = [p.strip() for p in cli_anneal_str.split(",")]
        if len(parts) != 4:
            raise SystemExit(
                f"FATAL: --mask-half-sim-prob-schedule expects 4 comma-"
                f"separated floats (start_value,ramp_start_frac,end_value,"
                f"ramp_end_frac), got {cli_anneal_str!r}"
            )
        args.mask_half_sim_prob_anneal = {
            "start_value": float(parts[0]),
            "ramp_start_frac": float(parts[1]),
            "end_value": float(parts[2]),
            "ramp_end_frac": float(parts[3]),
        }
    else:
        args.mask_half_sim_prob_anneal = profile_vals.get("mask_half_sim_prob_anneal")
    if args.mask_half_sim_prob_anneal is not None:
        sched = args.mask_half_sim_prob_anneal
        required_keys = {
            "start_value", "ramp_start_frac",
            "end_value", "ramp_end_frac",
        }
        missing = required_keys - set(sched.keys())
        if missing:
            raise SystemExit(
                f"FATAL: mask_half_sim_prob_anneal missing required keys "
                f"{sorted(missing)}; have {sorted(sched.keys())}"
            )
        for key, val in sched.items():
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise SystemExit(
                    f"FATAL: mask_half_sim_prob_anneal[{key!r}]={val!r} "
                    f"must be a float in [0, 1]"
                )
        if sched["ramp_start_frac"] > sched["ramp_end_frac"]:
            raise SystemExit(
                f"FATAL: mask_half_sim_prob_anneal ramp_start_frac="
                f"{sched['ramp_start_frac']} > ramp_end_frac="
                f"{sched['ramp_end_frac']} — schedule would ramp backwards"
            )
    args.latent_ch = _resolve(args.latent_ch, "latent_ch", 8)
    args.latent_shapes = profile_vals.get("latent_shapes", ((6, 8), (12, 16), (24, 32)))
    args.residual_hidden = _resolve(args.residual_hidden, "residual_hidden", 32)
    args.residual_layers = _resolve(args.residual_layers, "residual_layers", 2)
    args.residual_scale = _resolve(args.residual_scale, "residual_scale", 16.0)
    args.epochs = _resolve(args.epochs, "epochs", 200)
    args.lr = _resolve(args.lr, "lr", 1e-3)
    args.ema_decay = _resolve(args.ema_decay, "ema_decay", 0.997)
    args.accum_steps = _resolve(args.accum_steps, "accum_steps", 2)
    args.eval_every = _resolve(args.eval_every, "eval_every", 10)
    args.segnet_weight = _resolve(args.segnet_weight, "segnet_loss_weight", 100.0)
    args.pretrain_epochs = _resolve(args.pretrain_epochs, "pretrain_epochs", 0)
    # 5-phase Quantizr-adapted QAT schedule (Quantizr R2 C3 architectural
    # fix 2026-04-26). Default 0 epochs = phase disabled. The legacy
    # `pretrain_epochs` / `epochs` keys still work when no phaseN_epochs are
    # declared (backwards-compat path). See _phase_for_epoch() below for
    # the dispatch + boundaries logic.
    args.phase1_epochs = _resolve(args.phase1_epochs, "phase1_epochs", 0)
    args.phase2_epochs = _resolve(args.phase2_epochs, "phase2_epochs", 0)
    args.phase3_epochs = _resolve(args.phase3_epochs, "phase3_epochs", 0)
    args.phase4_epochs = _resolve(args.phase4_epochs, "phase4_epochs", 0)
    args.phase5_epochs = _resolve(args.phase5_epochs, "phase5_epochs", 0)
    args.phase1_lr = _resolve(args.phase1_lr, "phase1_lr", 1e-3)
    args.phase2_lr = _resolve(args.phase2_lr, "phase2_lr", 5e-4)
    args.phase3_lr = _resolve(args.phase3_lr, "phase3_lr", 3e-4)
    args.phase4_lr = _resolve(args.phase4_lr, "phase4_lr", 5e-5)
    args.phase5_lr = _resolve(args.phase5_lr, "phase5_lr", 1e-5)
    args.seed = _resolve(args.seed, "seed", 42)
    args.deterministic = _resolve(args.deterministic, "deterministic", True)

    # Yousfi council tricks (resolve from profile if not set via CLI)
    if not args.even_frame_skip_seg:
        args.even_frame_skip_seg = profile_vals.get("even_frame_skip_seg", False)
    # R40 fix: was `if args.frequency_loss_weight == 0.0` which inverted
    # CLI-overrides-profile semantics: a user passing --frequency-loss-weight 0.0
    # against a profile that sets 0.1 would silently get 0.1 back. Use the
    # canonical _resolve() path with default=None argparse sentinel.
    args.frequency_loss_weight = _resolve(
        args.frequency_loss_weight, "frequency_loss_weight", 0.0,
    )
    args.use_t2_xpred_loss = _resolve(
        getattr(args, "use_t2_xpred_loss", None),
        "use_t2_xpred_loss", False,
    )
    args.t2_xpred_sigma = _resolve(
        getattr(args, "t2_xpred_sigma", None),
        "t2_xpred_sigma", 1.0,
    )
    args.t2_xpred_weighting = _resolve(
        getattr(args, "t2_xpred_weighting", None),
        "t2_xpred_weighting", "v",
    )
    if args.t2_xpred_weighting not in {"v", "x"}:
        raise SystemExit(
            f"FATAL: t2_xpred_weighting must be 'v' or 'x', got "
            f"{args.t2_xpred_weighting!r}"
        )
    if float(args.t2_xpred_sigma) <= 0.0:
        raise SystemExit(
            f"FATAL: t2_xpred_sigma must be positive, got {args.t2_xpred_sigma}"
        )

    args.use_t2_mask = _resolve(
        getattr(args, "use_t2_mask", None),
        "use_t2_mask", False,
    )
    args.t2_mask_p = _resolve(
        getattr(args, "t2_mask_p", None),
        "t2_mask_p", 0.5,
    )
    args.t2_mask_ratio = _resolve(
        getattr(args, "t2_mask_ratio", None),
        "t2_mask_ratio", 0.15,
    )
    args.t2_mask_apply_fraction = _resolve(
        getattr(args, "t2_mask_apply_fraction", None),
        "t2_mask_apply_fraction", 0.4,
    )
    for _name in ("t2_mask_p", "t2_mask_ratio", "t2_mask_apply_fraction"):
        _val = float(getattr(args, _name))
        if not (0.0 <= _val <= 1.0):
            raise SystemExit(f"FATAL: {_name} must be in [0, 1], got {_val}")

    args.use_entropy_bottleneck = profile_vals.get("use_entropy_bottleneck", False)
    args.eb_lambda = float(profile_vals.get("eb_lambda", 0.0))
    args.eb_num_channels = int(profile_vals.get("eb_num_channels", args.mid_ch))
    if args.eb_lambda < 0.0:
        raise SystemExit(f"FATAL: eb_lambda must be >= 0, got {args.eb_lambda}")
    if args.eb_num_channels <= 0:
        raise SystemExit(
            f"FATAL: eb_num_channels must be positive, got {args.eb_num_channels}"
        )

    return args


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Configure process-level reproducibility for local and remote runs."""
    if deterministic:
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


# ── Lane V-V2: half-frame warp probability annealing ─────────────────────


def mask_half_sim_prob_for_epoch(
    epoch: int,
    total_epochs: int,
    *,
    static_prob: float,
    schedule: dict | None,
) -> float:
    """Compute the per-epoch ``mask_half_sim_prob`` value.

    When ``schedule`` is None, returns ``static_prob`` unchanged (the V1
    behaviour). When supplied, reads the 4-key annealing schedule and
    returns:

      * ``start_value`` for ``epoch_frac < ramp_start_frac``
      * linear interpolation ``start_value → end_value`` for
        ``ramp_start_frac <= epoch_frac < ramp_end_frac``
      * ``end_value`` for ``epoch_frac >= ramp_end_frac``

    where ``epoch_frac = epoch / max(total_epochs, 1)``.

    The schedule was validated at parse-time (see ``parse_args`` /
    ``_resolve``); this function trusts the keys are present and the
    floats are in [0, 1].
    """
    if schedule is None:
        return float(static_prob)
    if total_epochs <= 0:
        # Degenerate: if total_epochs is 0 the loop won't run anyway, but
        # don't divide by zero — return the start value.
        return float(schedule["start_value"])
    epoch_frac = float(epoch) / float(total_epochs)
    rs = float(schedule["ramp_start_frac"])
    re = float(schedule["ramp_end_frac"])
    sv = float(schedule["start_value"])
    ev = float(schedule["end_value"])
    if epoch_frac < rs:
        return sv
    if epoch_frac >= re:
        return ev
    if re <= rs:
        # Defensive: validator forbids this but if it slipped through,
        # fall back to end_value (the operator's intended endpoint).
        return ev
    # Linear ramp.
    t = (epoch_frac - rs) / (re - rs)
    return sv + (ev - sv) * t


# ── Data loading ────────────────────────────────────────────────────────


def load_gt_frames(args: argparse.Namespace) -> list[torch.Tensor]:
    """Load GT frames from precomputed .pt or decode from video."""
    precomputed = Path(args.precomputed) / "gt_frames.pt"
    if precomputed.exists():
        print(f"[data] Loading precomputed GT frames from {precomputed}")
        frames_tensor = torch.load(precomputed, map_location="cpu", weights_only=True)
        # Convert (N, H, W, 3) tensor to list of (H, W, 3)
        if isinstance(frames_tensor, torch.Tensor):
            return [frames_tensor[i] for i in range(frames_tensor.shape[0])]
        return frames_tensor
    else:
        print(f"[data] Decoding GT video from {args.video}")
        return decode_video(args.video)


def training_mask_pair_from_index(
    masks: torch.Tensor,
    start_idx: int,
    *,
    pair_index_basis: str = "full_frame_index",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a renderer mask pair for full-frame or half-frame training masks."""

    if pair_index_basis == "full_frame_index":
        return mask_pair_from_index(masks, start_idx)
    if pair_index_basis == "half_frame_pair_index":
        pair_idx = int(start_idx) // 2
        if pair_idx < 0 or pair_idx >= int(masks.shape[0]):
            raise IndexError(
                f"half-frame mask pair index {pair_idx} out of bounds for "
                f"{int(masks.shape[0])} masks"
            )
        mask = masks[pair_idx].unsqueeze(0)
        return mask, mask
    raise ValueError(f"unknown training mask pair index basis: {pair_index_basis!r}")


# ── FP4 evaluation ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_fp4(
    model: nn.Module,
    ema: EMA,
    all_masks: torch.Tensor,
    gt_frames: list[torch.Tensor],
    pair_starts: list[int],
    posenet,
    segnet,
    device: torch.device,
    *,
    fp4_codebook: str = "default",
    fp4_robust_scale: bool = False,
    sim_zoom_warp=None,
    use_zoom_flow: bool = False,
    half_frame_mode: bool = False,
    qfaithful_pose_lookup: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """Evaluate the EMA model after FP4 round-trip quantization.

    R-FP4-fix: codebook + robust_scale must match the QAT wrapper used during
    training. Mismatched eval-time quantization gives a misleading FP4 scorer
    (that's how the trend report's 93.44 plateau hid the real float→FP4 gap).

    Lane D (2026-04-27): when the model was built with use_zoom_flow=True
    (AsymmetricPairGenerator with 4-channel motion output), forward() requires
    an ego_flow tensor. Pass the same RadialZoomWarp used during training so
    the FP4 evaluation reflects the inflate-time motion structure.

    Codex R-Lane-D-Issue2 (2026-04-27): when ``half_frame_mode=True`` we
    REPLACE mask_t with ``warp_inverse_masks(mask_t1, pair_idx)``, mirroring
    the inflate-side reconstruction path in
    ``submissions/robust_current/inflate_renderer.py:2452``. This is what the
    deployed model will actually see when only odd-frame masks ship in the
    archive (Quantizr paradigm). Without this mode, best-checkpoint selection
    optimises a distribution the deployed model never sees, and Lane D's
    predicted 0.55-0.75 score depends on the wrong checkpoint shipping.

    Returns: (scorer, avg_pose, avg_seg)
    """
    from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
    codebook = (RESIDUAL_CODEBOOK if fp4_codebook == "residual"
                else DEFAULT_CODEBOOK).clone()

    if half_frame_mode and (sim_zoom_warp is None):
        raise ValueError(
            "evaluate_fp4(half_frame_mode=True) requires sim_zoom_warp; "
            "without it the warp_inverse_masks() call has nothing to apply. "
            "If you intended a full-frame eval, pass half_frame_mode=False."
        )

    # Build a temporary model with FP4-quantized EMA weights
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load EMA weights
    ema.apply(model)

    # FP4 round-trip: quantize then dequantize using the SAME codebook + scale
    # path that QAT trained against.
    fp4_packed = quantize_fp4(
        model.state_dict(), codebook=codebook, robust_scale=fp4_robust_scale,
    )
    fp4_state = dequantize_fp4(fp4_packed)
    model.load_state_dict(fp4_state)
    model.eval()

    use_autocast = device.type == "cuda" and torch.cuda.is_available()
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_autocast)

    total_p, total_s, count = 0.0, 0.0, 0
    with autocast_ctx:
        for start in pair_starts:
            mask_t, mask_t1 = mask_pair_from_index(all_masks, start)
            mask_t = mask_t.to(device)
            mask_t1 = mask_t1.to(device)

            # Per-pair index (shared by half-frame warp AND ego_flow lookup).
            # Pairs are formed as (frame[2k], frame[2k+1]); pair_starts steps
            # by SEQ_LEN=2, so pair_idx = start // 2.
            pair_idx_int = start // 2
            pair_idx_t = torch.tensor([pair_idx_int], device=device, dtype=torch.long)
            qfaithful_pose = None
            if qfaithful_pose_lookup is not None:
                qfaithful_pose = qfaithful_pose_lookup[pair_idx_int:pair_idx_int + 1].to(
                    device=device, dtype=torch.float32,
                )

            # Codex R-Lane-D-Issue2: replicate inflate-side mask reconstruction
            # when the deployed archive will only ship odd-frame masks. This is
            # the EXACT call inflate_renderer.py:2452 makes — same warp object,
            # same nearest-neighbour resampling, same border class fill.
            if half_frame_mode:
                mask_t = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx_t)

            gt_pair = pair_from_frames(gt_frames, start).to(device)

            # Lane D: ego_flow plumbing for use_zoom_flow=True models. Eval
            # path mirrors training path — same RadialZoomWarp, same per-pair
            # scalar lookup. No flip aug at eval time so no flow mirroring.
            ego_flow = None
            if use_zoom_flow and sim_zoom_warp is not None:
                H_m, W_m = mask_t1.shape[-2], mask_t1.shape[-1]
                ego_flow = sim_zoom_warp(pair_idx_t, H_m, W_m)

            # See companion fix in train_loop ~line 3155: when use_zoom_flow=True
            # AND variant=quantizr_faithful, both ego_flow and qfaithful_pose
            # are non-None. Pass BOTH; the shim swallows ego_flow via **_kwargs
            # and still receives the mandatory pose= kwarg. Eval must mirror
            # training's forward contract or the proxy-auth gap re-opens.
            forward_kwargs: dict = {}
            if ego_flow is not None:
                forward_kwargs["ego_flow"] = ego_flow
            if qfaithful_pose is not None:
                forward_kwargs["pose"] = qfaithful_pose
            rendered_pair = model(mask_t, mask_t1, **forward_kwargs)
            # uint8 round-trip to match official scorer pipeline
            rendered_pair = rendered_pair.round().clamp(0, 255).to(torch.uint8).float()

            score, pd, sd = eval_scorer_loss(rendered_pair, gt_pair, posenet, segnet)
            total_p += pd
            total_s += sd
            count += 1

            del mask_t, mask_t1, gt_pair, rendered_pair
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_p = total_p / max(count, 1)
    avg_s = total_s / max(count, 1)
    scorer = 100.0 * avg_s + math.sqrt(10.0 * avg_p)

    # Restore original weights
    model.load_state_dict(orig_state)
    model.train()
    return scorer, avg_p, avg_s


# ── Pre-training loss ──────────────────────────────────────────────────


def pretrain_loss(rendered_pair: torch.Tensor, gt_pair: torch.Tensor) -> torch.Tensor:
    """L1 + edge-aware loss for renderer pre-training (Phase 1).

    No scorer in the loop -- just pixel reconstruction + edge matching.
    This teaches basic texture synthesis from masks before scorer fine-tuning.

    Args:
        rendered_pair: (B, 2, H, W, 3) in [0, 255]
        gt_pair: (B, 2, H, W, 3) in [0, 255]

    Returns:
        Scalar loss = L1 + 0.5 * edge_loss
    """
    r = rendered_pair / 255.0
    g = gt_pair.float() / 255.0
    # Renderer output is at mask resolution (384x512), GT may be at full resolution (874x1164)
    # Downscale GT to match renderer output if sizes differ
    if r.shape[2:4] != g.shape[2:4]:
        # (B, 2, H, W, 3) -> (B*2, 3, H, W) for interpolation, then back
        g_bchw_tmp = g.reshape(-1, *g.shape[2:]).permute(0, 3, 1, 2)
        g_bchw_tmp = F.interpolate(g_bchw_tmp, size=r.shape[2:4], mode="bilinear", align_corners=False)
        g = g_bchw_tmp.permute(0, 2, 3, 1).reshape(r.shape)
    l1 = F.l1_loss(r, g)

    # Simple edge loss via horizontal gradient magnitude
    r_bchw = r.reshape(-1, *r.shape[2:]).permute(0, 3, 1, 2)  # (B*2, 3, H, W)
    g_bchw = g.reshape(-1, *g.shape[2:]).permute(0, 3, 1, 2)
    edge_r = (r_bchw[:, :, :, 1:] - r_bchw[:, :, :, :-1]).abs().mean()
    edge_g = (g_bchw[:, :, :, 1:] - g_bchw[:, :, :, :-1]).abs().mean()
    edge_loss = F.l1_loss(edge_r, edge_g)

    return l1 + 0.5 * edge_loss


class _T2MaskHook:
    """Forward-hook controller for training-only bottleneck feature masking."""

    def __init__(self, masker: FeatureMasker) -> None:
        self.masker = masker
        self.training_progress = 0.0
        self.handle = None

    def __call__(self, _module, _inputs, output: torch.Tensor) -> torch.Tensor:
        return self.masker(output, training_progress=self.training_progress)

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(progress)


def _install_t2_feature_masker(
    model: nn.Module,
    *,
    channels: int,
    p: float,
    mask_ratio: float,
    apply_in_final_fraction: float,
    seed: int,
) -> _T2MaskHook:
    """Attach FeatureMasker to ``model.renderer.bottleneck`` as a hook.

    The hook keeps the original bottleneck module in place, so deployment
    checkpoints retain the standard renderer key layout. The train-time
    mask token is registered as ``renderer.t2_feature_masker`` so it receives
    gradients and is saved in training-state checkpoints.
    """
    renderer = getattr(model, "renderer", None)
    if renderer is None:
        renderer = model
    bottleneck = getattr(renderer, "bottleneck", None)
    if bottleneck is None:
        raise ValueError(
            "Lane T2-MASK requires a renderer with a bottleneck module; "
            f"got {type(model).__name__}"
        )
    if hasattr(renderer, "t2_feature_masker"):
        raise ValueError("Lane T2-MASK is already installed on this renderer")

    masker = FeatureMasker(
        channels=channels,
        p=p,
        mask_ratio=mask_ratio,
        apply_in_final_fraction=apply_in_final_fraction,
        seed=seed,
    )
    renderer.add_module("t2_feature_masker", masker)
    hook = _T2MaskHook(masker)
    hook.handle = bottleneck.register_forward_hook(hook)
    return hook


class _EntropyBottleneckHook:
    """Forward-hook controller for train-time entropy bottleneck regularization."""

    def __init__(self, entropy_bottleneck: nn.Module) -> None:
        self.entropy_bottleneck = entropy_bottleneck
        self.handle = None

    def __call__(self, _module, _inputs, output: torch.Tensor) -> torch.Tensor:
        y_hat, _bits = self.entropy_bottleneck(output)
        return y_hat


def _install_entropy_bottleneck(
    model: nn.Module,
    *,
    channels: int,
    init_scale: float = 10.0,
) -> tuple[nn.Module, _EntropyBottleneckHook]:
    """Attach EntropyBottleneck to ``model.renderer.bottleneck`` as a hook."""
    from tac.entropy_bottleneck import EntropyBottleneck

    renderer = getattr(model, "renderer", None)
    if renderer is None:
        renderer = model
    bottleneck = getattr(renderer, "bottleneck", None)
    if bottleneck is None:
        raise ValueError(
            "Lane EBR requires a renderer with a bottleneck module; "
            f"got {type(model).__name__}"
        )
    if hasattr(renderer, "entropy_bottleneck"):
        raise ValueError("Lane EBR entropy bottleneck is already installed")

    eb = EntropyBottleneck(num_channels=channels, init_scale=init_scale)
    renderer.add_module("entropy_bottleneck", eb)
    hook = _EntropyBottleneckHook(eb)
    hook.handle = bottleneck.register_forward_hook(hook)
    return eb, hook


def _strip_t2_training_only_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drop train-time-only latent regularizer keys before deploy export."""
    return {
        k: v for k, v in state.items()
        if ".t2_feature_masker." not in k
        and not k.startswith("t2_feature_masker.")
        and ".entropy_bottleneck." not in k
        and not k.startswith("entropy_bottleneck.")
    }


def _zero_mask_encoder_hook(
    _module: nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """Forward hook for Lane T2-DROP's informational-only encoder ablation."""
    return torch.zeros_like(output)


def install_no_mask_encoder_hooks(model: nn.Module) -> list:
    """Zero all 5-class mask embedding outputs without changing state dicts.

    The renderer and motion predictor use shared ``nn.Embedding(5, D)``
    modules as the mask encoder. Forward hooks keep those modules present for
    checkpoint compatibility while replacing their encoded features with zeros.
    """
    handles = []
    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings == 5:
            handles.append(module.register_forward_hook(_zero_mask_encoder_hook))
    if not handles:
        raise ValueError(
            "Lane T2-DROP --no-mask-encoder found no 5-class nn.Embedding "
            "mask encoders to ablate"
        )
    return handles


def apply_no_mask_encoder_if_requested(
    model: nn.Module,
    args: argparse.Namespace,
) -> list:
    """Install T2-DROP hooks when ``args.no_mask_encoder`` is set."""
    if not getattr(args, "no_mask_encoder", False):
        return []
    return install_no_mask_encoder_hooks(model)


# ── Checkpoint arch_meta peek (resume-time pose_dim resolution) ────────
#
# Codex R-Lane-D-Issue1 (2026-04-27). Before building the model from the
# profile we MUST peek the checkpoint to decide pose_dim. Three cases:
#
#   1. arch_meta present (NEW format): use arch_meta["pose_dim"] verbatim.
#      Profile pose_dim is overridden with a loud warning if they disagree.
#   2. arch_meta absent (LEGACY format) AND state_dict has any film_* keys:
#      checkpoint was trained with pose_dim>0, infer pose_dim=6 (the only
#      pose_dim ever used in production — DEN, SHIRAZ, WILDE, GREEN, Lane D
#      all set pose_dim=6).
#   3. arch_meta absent AND no film_* keys: legacy checkpoint with the dead
#      resolver (pose_dim was effectively 0). Force pose_dim=0 with WARN so
#      the resume succeeds, instead of crashing on strict load.
#
# --force-pose-dim N at the CLI beats all of the above; that path lives in
# parse_args() and is checked before this helper runs.

# Architecture-meta keys we know how to round-trip on resume. Adding a new
# one? It must (a) be saved in BOTH save_training_state() AND the FP4 + fp32
# best save sites, (b) be read here, and (c) be plumbed through to
# build_renderer() at the construction site.
_RESUMABLE_ARCH_KEYS = (
    "pose_dim", "base_ch", "mid_ch", "embed_dim", "motion_hidden", "depth",
    "use_zoom_flow", "use_dsconv", "use_ghost", "use_dilation", "padding_mode",
)


def _peek_checkpoint_arch_meta(
    ckpt_path: Path | str,
    device: torch.device | str = "cpu",
) -> dict | None:
    """Return the saved arch_meta dict if present, else None.

    Auto-detects legacy checkpoints that lack arch_meta and synthesises a
    minimal {"pose_dim": ...} based on whether film_* keys appear in the
    state_dict. Never raises — load failures are reported via stderr and
    return None so the caller can fall back to the profile.
    """
    try:
        state = torch.load(str(ckpt_path), map_location=device,
                           weights_only=False)
    except Exception as exc:
        print(f"[resume] WARN: could not peek {ckpt_path} ({exc!r}); "
              f"profile pose_dim will be used as-is.", file=sys.stderr)
        return None

    # NEW format: arch_meta stored as a sibling key.
    arch = state.get("arch_meta") if isinstance(state, dict) else None
    if isinstance(arch, dict) and "pose_dim" in arch:
        return arch

    # FP4 / fp32 best save format: __meta__ contains the same arch fields.
    meta = state.get("__meta__") if isinstance(state, dict) else None
    if isinstance(meta, dict) and "pose_dim" in meta:
        return {k: meta[k] for k in _RESUMABLE_ARCH_KEYS if k in meta}

    # LEGACY format: synthesise from state_dict film key presence.
    sd = None
    if isinstance(state, dict):
        sd = state.get("model") or state.get("model_state_dict")
        if sd is None and all(isinstance(v, torch.Tensor) for v in state.values()):
            sd = state  # raw state_dict
    if isinstance(sd, dict):
        has_film = any("film_" in k for k in sd)
        synth = {"pose_dim": 6 if has_film else 0,
                 "_legacy_no_arch_meta": True}
        return synth

    # Couldn't determine — let the caller fall back to profile.
    return None


def _resolve_pose_dim_for_resume(
    args: argparse.Namespace,
    ckpt_path: Path | str | None,
) -> tuple[int, str]:
    """Decide pose_dim for the model under construction. Returns (value, source).

    Priority:
        --force-pose-dim    > profile/checkpoint               (already
                              applied in parse_args; we just echo it here)
        checkpoint arch_meta > profile resolver
        legacy auto-detect (film keys present) > profile resolver
        legacy auto-detect (no film keys)      > FORCE 0 + warn
    """
    if getattr(args, "force_pose_dim", None) is not None:
        return int(args.force_pose_dim), "cli_force"
    if not ckpt_path or not Path(str(ckpt_path)).exists():
        return int(getattr(args, "pose_dim", 0) or 0), "profile"
    meta = _peek_checkpoint_arch_meta(ckpt_path)
    if meta is None:
        return int(getattr(args, "pose_dim", 0) or 0), "profile"
    ckpt_pd = int(meta.get("pose_dim", 0) or 0)
    profile_pd = int(getattr(args, "pose_dim", 0) or 0)
    src = "checkpoint_arch_meta"
    if meta.get("_legacy_no_arch_meta"):
        src = ("legacy_film_autodetect" if ckpt_pd > 0
               else "legacy_no_film_autodetect_zero")
    if ckpt_pd != profile_pd:
        print(
            f"[resume] WARN: profile pose_dim={profile_pd} OVERRIDDEN to "
            f"pose_dim={ckpt_pd} by checkpoint {src} "
            f"(legacy={'yes' if meta.get('_legacy_no_arch_meta') else 'no'}). "
            f"Pass --force-pose-dim {profile_pd} to override the override "
            f"(but expect strict load_state_dict failure if shapes mismatch).",
            file=sys.stderr,
        )
    return ckpt_pd, src


# ── 5-Phase Quantizr-style QAT schedule helpers ────────────────────────
#
# Quantizr's published recipe (project_quantizr_full_intel_20260421 +
# project_quantizr_definitive_binary_analysis):
#   Phase 1 (anchor):    pretrain on pixel L1 vs GT only.
#   Phase 2 (finetune):  add SegNet hinge + PoseNet MSE.
#   Phase 3 (joint):     add Fridrich (UNIWARD/L_inf/Markov/DCT) + KL distill.
#   Phase 4 (QAT):       enable FP4 fake-quant on weights, low LR.
#   Phase 5 (final):     hard-pair-emphasis sampler at very low LR.
# Each phase has its own LR + epoch count. Backwards-compat: if no
# phaseN_epochs are declared (all zero), the legacy two-phase loop
# (pretrain_epochs / epochs) is used unchanged.

PHASE_NAMES = {1: "anchor", 2: "finetune", 3: "joint", 4: "qat", 5: "final"}


def has_5phase_schedule(args: argparse.Namespace) -> bool:
    """True iff at least one phaseN_epochs > 0 — opts into the new schedule.

    Backwards-compat: when False, callers fall back to the legacy
    `pretrain_epochs` / `epochs` two-phase loop. The 5-phase loop NEVER
    activates implicitly — a profile must declare phase epochs explicitly.
    """
    return any(
        getattr(args, f"phase{i}_epochs", 0) > 0 for i in range(1, 6)
    )


def phase_boundaries(args: argparse.Namespace) -> list[int]:
    """Cumulative epoch boundaries for the 5-phase schedule.

    Returns [b1, b2, b3, b4, b5] where epoch < b1 → phase 1, b1 <= epoch < b2
    → phase 2, etc. b5 == total_epochs.
    """
    out = []
    cum = 0
    for i in range(1, 6):
        cum += int(getattr(args, f"phase{i}_epochs", 0))
        out.append(cum)
    return out


def current_phase(epoch: int, boundaries: list[int]) -> int:
    """Return 1..5 for `epoch`, or the last non-empty phase if past the end."""
    for phase_idx, b in enumerate(boundaries, start=1):
        if epoch < b:
            return phase_idx
    # Past the schedule: return the last phase with epochs > 0
    for phase_idx in range(5, 0, -1):
        if boundaries[phase_idx - 1] > (boundaries[phase_idx - 2] if phase_idx >= 2 else 0):
            return phase_idx
    return 5


def lr_for_phase(args: argparse.Namespace, phase: int) -> float:
    """Resolve per-phase LR from args (already populated by the resolver)."""
    return float(getattr(args, f"phase{phase}_lr"))


def phase_epoch_offset(epoch: int, phase: int, boundaries: list[int]) -> tuple[int, int]:
    """(epoch_within_phase, phase_total_epochs) — used for cosine annealing
    within a single phase (so each phase ramps its own LR independently)."""
    start = boundaries[phase - 2] if phase >= 2 else 0
    total = boundaries[phase - 1] - start
    return (epoch - start, max(1, total))


def cosine_lr(base_lr: float, step: int, total: int, eta_min: float = 1e-6) -> float:
    """Cosine annealing within a single phase."""
    if total <= 0:
        return base_lr
    step = max(0, min(step, total))
    cos = 0.5 * (1 + math.cos(math.pi * step / total))
    return eta_min + (base_lr - eta_min) * cos


def resize_pair_hwc(pair: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize a ``(B, 2, H, W, 3)`` pair tensor while preserving HWC layout."""
    if pair.shape[2:4] == (target_h, target_w):
        return pair
    bsz, frames, _h, _w, channels = pair.shape
    flat = pair.reshape(bsz * frames, _h, _w, channels).permute(0, 3, 1, 2).contiguous()
    flat = F.interpolate(flat.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
    return flat.permute(0, 2, 3, 1).contiguous().reshape(bsz, frames, target_h, target_w, channels)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_qfaithful_training_poses(
    args: argparse.Namespace,
    *,
    n_pairs: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict | None]:
    if getattr(args, "variant", None) != "quantizr_faithful":
        return None, None

    pose_path_value = (
        getattr(args, "qfaithful_training_poses", None)
        or getattr(args, "auth_eval_poses", None)
    )
    if not pose_path_value:
        raise SystemExit(
            "[lane-q-faithful] CONFIG ERROR: variant=quantizr_faithful requires "
            "--qfaithful-training-poses pointing at the deployed nonzero pose "
            "stream. Silent zero-pose fallback is forbidden because it trains a "
            "different FiLM contract than inflate/eval uses."
        )

    from tac.submission_archive import load_optimized_poses

    pose_path = Path(pose_path_value)
    pose_dim = int(getattr(args, "pose_dim", 0) or 6)
    poses = load_optimized_poses(
        pose_path,
        pose_dim=pose_dim,
        expected_n_pairs=n_pairs,
    ).to(dtype=torch.float32)
    if poses.ndim != 2 or poses.shape != (n_pairs, pose_dim):
        raise ValueError(
            f"[lane-q-faithful] pose stream must have shape "
            f"({n_pairs}, {pose_dim}); got {tuple(poses.shape)} from {pose_path}"
        )
    abs_sum = float(poses.abs().sum().item())
    if abs_sum <= 0.0:
        raise ValueError(
            f"[lane-q-faithful] pose stream at {pose_path} is all-zero; "
            "training must use the deployed nonzero pose stream."
        )

    sha = _sha256_file(pose_path)
    contract = {
        "schema_version": 1,
        "training_pose_contract": "qfaithful_nonzero_deployed_pose_stream_v1",
        "training_pose_contract_promotable": True,
        "training_uses_nonzero_pose_stream": True,
        "training_uses_deployed_pose_stream": True,
        "zero_pose_fallback_allowed": False,
        "pose_dim": pose_dim,
        "n_pairs": n_pairs,
        "pose_path": str(pose_path),
        "pose_sha256": sha,
        "pose_source_sha256": sha,
        "pose_abs_sum": abs_sum,
    }
    print(
        "[lane-q-faithful] training pose contract active: "
        f"path={pose_path} sha256={sha} shape={tuple(poses.shape)}"
    )
    print(
        "[lane-q-faithful] horizontal flip augmentation disabled: "
        "pose-conditioned flips require an audited pose transform."
    )
    return poses.to(device), contract


def _saug_v2_generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        return torch.Generator().manual_seed(seed)


def _apply_saug_v2_to_renderer_input(x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Apply SAUG-V2 to renderer inputs without changing renderer APIs.

    Floating point inputs receive sigmas as supplied. Current renderer masks are
    categorical class indices, so train_renderer scales the uint8-style Cosmos
    sigma recipe into class-index units and rounds/clamps back to valid labels.
    """
    if torch.is_floating_point(x):
        return apply_sigma_noise_to_input(x, sigmas)

    scaled_sigmas = sigmas.to(device=x.device, dtype=torch.float32) / 255.0
    noisy = apply_sigma_noise_to_input(x.to(torch.float32), scaled_sigmas)
    return noisy.round().clamp(0, 4).to(dtype=x.dtype)


# ── Training loop ───────────────────────────────────────────────────────


def train(args: argparse.Namespace):
    # Codex R5-2 Finding #1 (2026-04-27): EARLY-FAIL the operator before any
    # GPU spend if --auth-eval-on-best is on for a variant whose checkpoint
    # cannot be FP4A-exported. Pre-fix, the run completed (hours), THEN crashed
    # at the post-training auth-eval block. The CLAUDE.md non-negotiable
    # "Auth eval EVERYWHERE" exists to catch the proxy-auth gap; failing
    # silently after training is the exact failure mode it forbids.
    #
    # This check intentionally runs BEFORE configure_reproducibility/device
    # selection so we don't pay even the model-load cost.
    # Lane S: when SC codec is on, we cannot FP4A-export (the SC weights
    # are stored as full nn.Conv2d.weight + a per-channel bit_depth tensor;
    # FP4A would round-trip through full precision and lose all SC gain).
    # Disable auth_eval_on_best up-front so the masks/poses requirement
    # check below does not kill an otherwise valid SC training run. The
    # SCv1 inflate path is the long-term auth path for these models.
    if (
        getattr(args, "use_self_compress_codec", False)
        and getattr(args, "auth_eval_on_best", False)
    ):
        print(
            "[lane-s] startup gate: --auth-eval-on-best disabled because "
            "use_self_compress_codec=True. SC weights cannot be FP4A-exported "
            "without losing the rate gain. Use SCv1 export + a separate auth "
            "eval after training."
        )
        args.auth_eval_on_best = False

    if getattr(args, "auth_eval_on_best", False):
        if not _variant_supports_fp4a_export(args.variant):
            raise SystemExit(
                f"[train] CONFIG ERROR: --auth-eval-on-best is enabled "
                f"(default true) but variant={args.variant!r} is not "
                f"FP4A-exportable. The post-training auth eval would crash "
                f"after hours of GPU spend. Either:\n"
                f"  (1) pass --no-auth-eval-on-best and run a variant-"
                f"specific eval afterwards, OR\n"
                f"  (2) switch to a build_renderer-compatible variant "
                f"(see _VARIANTS_BUILD_RENDERER_FP4A_OK in train_renderer.py).\n"
                f"Profile: {getattr(args, 'profile', None)!r}. "
                f"Failure mode: codex R5-2 Finding #1."
            )
        if not args.auth_eval_masks or not args.auth_eval_poses:
            raise SystemExit(
                f"[train] CONFIG ERROR: --auth-eval-on-best=True requires "
                f"BOTH --auth-eval-masks and --auth-eval-poses. Without "
                f"them the post-training archive would be renderer-only "
                f"(~290KB) instead of the real ~700KB triple-joint archive — "
                f"systematically optimistic by ~2x on the rate term. Got: "
                f"masks={args.auth_eval_masks!r}, poses={args.auth_eval_poses!r}.\n"
                f"To skip the auth eval entirely, pass --no-auth-eval-on-best."
            )

    configure_reproducibility(args.seed, args.deterministic)
    device = torch.device(args.device) if args.device else detect_device()
    print(f"[train] Device: {device}")
    print(f"[train] Reproducibility: seed={args.seed}, deterministic={args.deterministic}")

    saug_v2_sampler = None
    if getattr(args, "use_saug_v2", False):
        saug_v2_config = HighSigmaStrategyConfig(
            redraw_fraction=float(args.saug_v2_redraw_fraction),
            high_sigma_min=float(args.saug_v2_high_sigma_min),
            high_sigma_max=float(args.saug_v2_high_sigma_max),
            normal_sigma_min=float(args.saug_v2_normal_sigma_min),
            normal_sigma_max=float(args.saug_v2_normal_sigma_max),
            enabled=True,
        )
        saug_v2_generator = _saug_v2_generator_for_device(
            device, int(args.seed) + 20_026,
        )
        saug_v2_sampler = HighSigmaSampler(
            saug_v2_config, generator=saug_v2_generator,
        )
        print(
            "[lane-saug-v2] HighSigmaStrategy ACTIVE: "
            f"redraw_fraction={saug_v2_config.redraw_fraction} "
            f"normal=[{saug_v2_config.normal_sigma_min}, "
            f"{saug_v2_config.normal_sigma_max}] "
            f"high=[{saug_v2_config.high_sigma_min}, "
            f"{saug_v2_config.high_sigma_max}]"
        )

    # Lane MAE-V: MAEMaskAugmenter on input mask patches (training-only).
    # Eval mode is a strict passthrough — see MAEMaskAugmenter.forward —
    # so the inference distribution exactly matches the contest scorer's.
    mae_mask_augmenter = None
    if getattr(args, "use_mae_mask_aug", False):
        mae_cfg = MAEMaskAugConfig(
            mask_ratio=float(args.mae_mask_ratio),
            patch_size=int(args.mae_patch_size),
            num_classes=5,
            enabled=True,
        )
        mae_generator = _saug_v2_generator_for_device(
            device, int(args.seed) + 30_028,
        )
        mae_mask_augmenter = MAEMaskAugmenter(
            mae_cfg, generator=mae_generator,
        ).to(device)
        mae_mask_augmenter.train()
        print(
            "[lane-mae-v] MAEMaskAugmenter ACTIVE: "
            f"mask_ratio={mae_cfg.mask_ratio} "
            f"patch_size={mae_cfg.patch_size}"
        )

    # Load scorer models (respect TAC_MODELS_DIR env var for Modal/remote deploys)
    _models_dir = Path(_os.environ.get("TAC_MODELS_DIR", str(_upstream / "models")))
    posenet_path = _models_dir / "posenet.safetensors"
    segnet_path = _models_dir / "segnet.safetensors"
    print(f"[train] Loading scorers from {posenet_path.parent}")
    posenet, segnet = load_scorers(
        posenet_path, segnet_path, device=device, upstream_dir=str(_upstream),
    )

    # Load GT frames
    gt_frames = load_gt_frames(args)
    # R-FP4-fix: --max-frames lets trend/smoke runs use a small slice without
    # repacking the precomputed file. Truncate to even count to preserve
    # pair-construction invariants (every pair = even+odd frame).
    if args.max_frames is not None and len(gt_frames) > args.max_frames:
        n_keep = (args.max_frames // 2) * 2
        gt_frames = gt_frames[:n_keep]
        print(f"[data] --max-frames={args.max_frames} → truncated to {n_keep}")
    n_frames = len(gt_frames)
    print(f"[data] {n_frames} GT frames loaded, shape {gt_frames[0].shape}")

    # Extract masks once at startup
    print(f"[masks] Extracting {n_frames} masks via frozen SegNet (batch_size={args.mask_batch_size})...")
    t0 = time.monotonic()
    all_masks = extract_masks(gt_frames, segnet, device=device, batch_size=args.mask_batch_size)
    print(f"[masks] Extracted {all_masks.shape} masks in {time.monotonic() - t0:.1f}s")
    # Keep masks on CPU, move per-pair to device during training
    all_masks = all_masks.cpu()

    # Optional noisy-mask augmentation: decode a CRF-encoded mkv and randomly
    # mix in during training so the renderer becomes robust to AV1 quantization
    # at inflate time. (Yousfi 2026-04-25 diagnosis after CRF=63 gating.)
    noisy_masks = None
    noisy_masks_pair_index_basis = "full_frame_index"
    if args.mask_noise_mkv is not None:
        from tac.mask_codec import decode_masks
        print(f"[masks] Decoding noisy masks from {args.mask_noise_mkv} ...")
        noisy_masks = decode_masks(args.mask_noise_mkv).cpu()
        if noisy_masks.shape[0] == all_masks.shape[0]:
            noisy_reference_masks = all_masks
        elif noisy_masks.shape[0] * 2 == all_masks.shape[0]:
            noisy_masks_pair_index_basis = "half_frame_pair_index"
            noisy_reference_masks = all_masks[1::2]
            print(
                f"[masks] Noisy masks are half-frame pair-indexed "
                f"({noisy_masks.shape[0]} pair masks for {all_masks.shape[0]} "
                "GT masks); training will feed the archived pair mask as "
                "mask_t1 and duplicate it for mask_t."
            )
        else:
            raise ValueError(
                f"Noisy mask count {noisy_masks.shape[0]} is incompatible with "
                f"GT mask count {all_masks.shape[0]}; expected full-frame count "
                "or half-frame pair count."
            )
        # Sanity-check resolution match
        if noisy_masks.shape[1:] != all_masks.shape[1:]:
            raise ValueError(
                f"Noisy mask resolution {noisy_masks.shape[1:]} != "
                f"GT mask resolution {all_masks.shape[1:]}. "
                f"Re-encode noisy masks at the renderer's input resolution."
            )
        disagree = float((noisy_masks != noisy_reference_masks).float().mean().item())
        print(f"[masks] Noisy-mask augmentation enabled "
              f"(prob={args.mask_noise_prob}, basis={noisy_masks_pair_index_basis}, "
              f"disagreement vs GT={disagree:.4f})")

    # Zoom warp precompute. The same RadialZoomWarp object serves TWO purposes:
    #
    #   (1) Training-time half-frame mask simulation (Lane D2) — when
    #       --mask-half-sim-prob > 0, periodically replace mask_t with
    #       inverse_warp(mask_t1, zoom[k]) so the renderer learns the same
    #       mask distribution it sees at inflate.
    #
    #   (2) ego_flow input to AsymmetricPairGenerator (Lane D, council 2026-04-27).
    #       When --use-zoom-flow is set, the model expects an ego_flow tensor
    #       in forward(). We compute it from the SAME zoom_warp on a per-pair
    #       basis so train and inflate distributions match exactly.
    #
    # Either trigger forces construction. The zoom scalars come from analytical
    # lane-mark-speed estimation (Hotz: lane markings encode forward speed →
    # radial zoom from FoE), no GT poses needed. RadialZoomWarp +
    # warp_inverse_masks come from tac.radial_zoom.
    sim_zoom_warp = None
    _need_zoom_warp = (
        getattr(args, "mask_half_sim_prob", 0.0) > 0
        or getattr(args, "use_zoom_flow", False)
    )
    if _need_zoom_warp:
        from tac.lane_mark_speed import zoom_from_masks
        from tac.radial_zoom import RadialZoomWarp
        n_sim_pairs = all_masks.shape[0] // 2
        sim_zooms = zoom_from_masks(all_masks)  # (n_pairs,) float32
        sim_zoom_warp = RadialZoomWarp(n_pairs=n_sim_pairs).to(device)
        with torch.no_grad():
            sim_zoom_warp.zoom_scalars.data.copy_(
                sim_zooms.clamp(-sim_zoom_warp.max_zoom_log, sim_zoom_warp.max_zoom_log)
            )
        # Freeze zoom scalars during training — they're analytical, not learned.
        # The renderer adapts to THEM, not the other way around. (Pose TTO at
        # compress time may further refine these scalars later.)
        for p in sim_zoom_warp.parameters():
            p.requires_grad_(False)
        sim_zoom_warp.eval()
        roles = []
        if getattr(args, "mask_half_sim_prob", 0.0) > 0:
            roles.append(f"half-frame sim (prob={args.mask_half_sim_prob})")
        if getattr(args, "use_zoom_flow", False):
            roles.append("ego_flow input")
        print(f"[masks] RadialZoomWarp constructed for: {', '.join(roles)} "
              f"— zoom mean={sim_zooms.mean():.4f}, std={sim_zooms.std():.4f}, "
              f"n_pairs={n_sim_pairs}")

    # Codex R-Lane-D-Issue1 (2026-04-27): peek the resume checkpoint BEFORE
    # building the model so the saved arch_meta (or legacy autodetect) wins
    # over the profile pose_dim. This fixes the regression where Lane D's
    # newly-active resolver promoted profile pose_dim=6 → builds FiLM layers
    # → strict load_state_dict crash on legacy checkpoints saved with the
    # dead resolver (effective pose_dim=0 weights).
    _resume_ckpt_path = (
        args.resume_from
        if getattr(args, "resume_from", None)
        and Path(args.resume_from).exists()
        else None
    )
    _resolved_pose_dim, _pose_dim_source = _resolve_pose_dim_for_resume(
        args, _resume_ckpt_path,
    )
    if _resolved_pose_dim != int(getattr(args, "pose_dim", 0) or 0):
        print(
            f"[arch] pose_dim resolved from {_pose_dim_source}: "
            f"profile/cli={getattr(args, 'pose_dim', 0)} → "
            f"final={_resolved_pose_dim}"
        )
    else:
        print(f"[arch] pose_dim={_resolved_pose_dim} (source={_pose_dim_source})")
    args.pose_dim = _resolved_pose_dim

    # Build model (dispatch by variant)
    if args.variant == "wavelet_renderer":
        model = build_wavelet_renderer(
            num_classes=5,
            embed_dim=args.embed_dim,
            hidden=args.base_ch,
            motion_hidden=args.motion_hidden,
        )
    elif args.variant == "coord_renderer":
        from tac.contrib.coord_renderer import build_coord_renderer
        model = build_coord_renderer(
            num_classes=5,
            class_embed_dim=args.embed_dim,
            hidden_dim=args.base_ch,
            motion_hidden=args.motion_hidden,
        )
    elif args.variant == "coolchic_renderer":
        from tac.contrib.coolchic_renderer import build_coolchic_renderer
        model = build_coolchic_renderer(
            num_classes=5,
            embed_dim=args.embed_dim,
            latent_ch=args.latent_ch,
            hidden=args.base_ch,
            motion_hidden=args.motion_hidden,
            latent_shapes=args.latent_shapes,
            blend_mode=args.blend_mode,
            noise_mode=args.noise_mode,
        )
    elif args.variant == "c3_residual_renderer":
        from tac.contrib.coolchic_renderer import build_c3_residual_renderer
        model = build_c3_residual_renderer(
            num_classes=5,
            embed_dim=args.embed_dim,
            latent_ch=args.latent_ch,
            hidden=args.base_ch,
            motion_hidden=args.motion_hidden,
            residual_hidden=args.residual_hidden,
            residual_layers=args.residual_layers,
            residual_scale=args.residual_scale,
            latent_shapes=args.latent_shapes,
            blend_mode=args.blend_mode,
            noise_mode=args.noise_mode,
        )
    elif args.variant == "dp_sims":
        from tac.dp_sims_renderer import build_dp_sims_renderer
        model = build_dp_sims_renderer(
            num_classes=5,
            motion_hidden=args.motion_hidden,
        )
    elif args.variant == "vqvae":
        from tac.contrib.vqvae_codec import build_vqvae_pair_generator
        model = build_vqvae_pair_generator()
    elif args.variant == "diffusion_teacher":
        from tac.contrib.diffusion_renderer import build_diffusion_teacher
        model = build_diffusion_teacher(
            num_classes=5,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        )
    elif args.variant == "quantizr_faithful":
        # Lane Q-FAITHFUL — TRUE 1:1 Quantizr PR #55 replica. Builds a
        # JointFrameGenerator (NO motion, NO warp, single-mask + FiLM-pose
        # dual-head) wrapped in a thin shim that exposes the AsymmetricPair
        # API so the rest of the training loop / loss / scorer plumbing
        # (which expects model(mask_t, mask_t1) -> (B, 2, H, W, 3) HWC) is
        # unchanged. The shim discards mask_t (per Quantizr's design — only
        # mask_t1 is used) and stacks the (frame1, frame2) outputs.
        from tac.quantizr_faithful_renderer import (
            build_quantizr_faithful_renderer,
        )

        class _QuantizrFaithfulShim(torch.nn.Module):
            """Adapt JointFrameGenerator to the (mask_t, mask_t1) -> pair
            API the training loop assumes.

            Quantizr's forward(mask2, pose6) -> (frame1, frame2) returns
            CHW frames in [0, 255]. The training loop expects HWC pairs
            in (B, 2, H, W, 3). We discard mask_t (Quantizr's premise: the
            odd-frame mask alone fully determines both reconstructions
            via the FiLM-conditioned dual head).
            """

            def __init__(self, gen):
                super().__init__()
                self.gen = gen
                # Surface the FiLM pose contract for downstream introspection.
                self.pose_dim = int(gen.pose_dim)
                # Mark this as the Q-FAITHFUL family so inflate / heuristics
                # can detect it without isinstance brittleness.
                self.q_faithful = True

            def forward(self, mask_t, mask_t1, pose=None, **_kwargs):
                # mask_t is intentionally unused — Quantizr's premise is
                # that mask_t1 (the odd-frame mask) plus pose6 fully
                # determines both frame reconstructions.
                _ = mask_t
                if pose is None:
                    raise RuntimeError(
                        "Q-FAITHFUL forward requires an explicit deployed "
                        "pose tensor. Zero-pose fallback is forbidden because "
                        "it trains a different FiLM contract than inflate/eval."
                    )
                pose = pose.to(device=mask_t1.device, dtype=torch.float32)
                f1, f2 = self.gen(mask_t1, pose)  # each (B, 3, H, W) [0, 255]
                # Stack to (B, 2, 3, H, W) then permute -> (B, 2, H, W, 3)
                pair = torch.stack([f1, f2], dim=1)
                pair = pair.permute(0, 1, 3, 4, 2).contiguous()
                return pair

        gen = build_quantizr_faithful_renderer(
            num_classes=5,
            pose_dim=int(getattr(args, "pose_dim", 0) or 6) or 6,
        )
        model = _QuantizrFaithfulShim(gen)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[lane-q-faithful] JointFrameGenerator built: {n_params:,} params "
            f"(target ~88K). NO motion, NO warp, single-mask + FiLM-pose.",
            flush=True,
        )
    else:
        # 2026-04-26 council fix (arch drift): pass ALL profile-resolved arch
        # flags. Previously use_zoom_flow / use_dsconv / padding_mode /
        # use_dilation / pose_dim were silently dropped here, causing the DEN
        # checkpoint to mismatch consumer expectations and waste 1.2h of GPU.
        # 2026-04-27 Lane D fix: every other variant in this dispatch imports
        # its build_* locally; the default `dilated`/`asym` path was missing
        # its import, causing UnboundLocalError on Lane D launch. The auth-eval
        # block at line ~2262 imports build_renderer separately; making this
        # block self-sufficient avoids a NameError before training begins.
        from tac.renderer import build_renderer
        model = build_renderer(
            num_classes=5,
            embed_dim=args.embed_dim,
            base_ch=args.base_ch,
            mid_ch=args.mid_ch,
            motion_hidden=args.motion_hidden,
            depth=args.depth,
            blend_mode=args.blend_mode,
            noise_mode=args.noise_mode,
            motion_type=args.motion_type,
            use_zoom_flow=args.use_zoom_flow,
            use_dsconv=args.use_dsconv,
            use_ghost=args.use_ghost,
            padding_mode=args.padding_mode,
            use_dilation=args.use_dilation,
            pose_dim=getattr(args, "pose_dim", 0) or 0,
        )

    no_mask_encoder_hooks = apply_no_mask_encoder_if_requested(model, args)
    if no_mask_encoder_hooks:
        print(
            "[lane-t2-drop] INFORMATIONAL-ONLY: --no-mask-encoder active; "
            "mask encoder outputs are zero-filled. This is not a stacking "
            "candidate.",
            flush=True,
        )

    t2_mask_hook = None
    if getattr(args, "use_t2_mask", False):
        t2_mask_hook = _install_t2_feature_masker(
            model,
            channels=int(args.mid_ch),
            p=float(args.t2_mask_p),
            mask_ratio=float(args.t2_mask_ratio),
            apply_in_final_fraction=float(args.t2_mask_apply_fraction),
            seed=int(args.seed),
        )
        print(
            f"[lane-t2-mask] enabled: p={args.t2_mask_p} "
            f"ratio={args.t2_mask_ratio} "
            f"apply_final_fraction={args.t2_mask_apply_fraction} "
            f"channels={args.mid_ch}",
            flush=True,
        )

    entropy_bottleneck = None
    if getattr(args, "use_entropy_bottleneck", False):
        entropy_bottleneck, _eb_hook = _install_entropy_bottleneck(
            model,
            channels=int(args.eb_num_channels),
        )
        print(
            f"[lane-ebr] entropy bottleneck enabled: "
            f"channels={args.eb_num_channels} lambda={args.eb_lambda}",
            flush=True,
        )

    # Lane S: post-construction SC codec swap. We do this BEFORE moving to
    # device so the new SelfCompressingConv2d submodules inherit the same
    # device move below. The swap is in-place — `model` is mutated.
    if getattr(args, "use_self_compress_codec", False):
        from tac.self_compress import (
            get_protected_patterns,
            renderer_average_bits_per_weight,
            swap_renderer_convs_with_self_compress,
        )
        # Lane SG (2026-04-28, hardened by Codex F3 2026-04-28): pick the
        # protection list per chosen scorer prior and pass it as a
        # REPLACEMENT (not extras). The previous additive wiring made
        # Lane SG protect PoseNet-prior layers AND SegNet-prior layers,
        # which dilutes the SegNet-only sensitivity signal Lane SG is
        # specifically testing. The two pattern sets are disjoint by
        # construction (see SC_SEGNET_PROTECTED_NAME_PATTERNS docstring),
        # so additive == both-sets-protected, replacement == only-chosen-set.
        pattern_set_name = getattr(args, "protected_pattern_set", "posenet_prior")
        chosen = get_protected_patterns(pattern_set_name)
        diag = swap_renderer_convs_with_self_compress(
            model,
            init_bits=float(args.self_compress_init_bits),
            protected_patterns=tuple(chosen),
        )
        print(
            f"[lane-s] protected_pattern_set={pattern_set_name!r} "
            f"REPLACES default — protected layers in active list: {len(chosen)} "
            f"({'PoseNet-prior' if pattern_set_name == 'posenet_prior' else 'SegNet-prior'})"
        )
        avg = renderer_average_bits_per_weight(model)
        print(
            f"[lane-s] SC codec ENABLED — swapped {len(diag['swapped'])} layers "
            f"({diag['total_swapped_params']:,} params) | "
            f"protected {len(diag['protected'])} | "
            f"skipped {len(diag['skipped'])} | "
            f"init_bits={args.self_compress_init_bits} avg={avg:.2f}"
        )
        if diag["protected"]:
            print(f"[lane-s] Protected (FP32): {diag['protected']}")
        # The auth-eval-on-best path exports via FP4A; if SC weights are not
        # convertible into the FP4A blob format we MUST disable auth eval at
        # the start (per the canonical pre-flight pattern), not crash hours
        # in. The SCv1 magic byte path is the long-term fix; until that
        # lands in pipeline.py we just disable.
        if getattr(args, "auth_eval_on_best", False):
            print(
                "[lane-s] WARN: --auth-eval-on-best is set but SC codec is on. "
                "FP4A export of SC weights would round-trip through full-precision "
                "and lose the SC rate gain. Use SCv1 export + pipeline.py auth "
                "after this run completes. Forcing auth_eval_on_best=False."
            )
            args.auth_eval_on_best = False

    model = model.to(device)
    # channels_last memory format: 10-30% speedup for conv2d on CUDA
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # 5-phase schedule detection (Quantizr R2 C3 architectural fix).
    # When a profile declares any phase{1..5}_epochs > 0 we activate the
    # 5-phase Quantizr-adapted QAT schedule; otherwise the legacy 2-phase
    # `pretrain_epochs` / `epochs` loop is preserved unchanged.
    use_5phase = has_5phase_schedule(args)
    boundaries: list[int] = phase_boundaries(args) if use_5phase else []
    if use_5phase:
        # Phase 4 may enable QAT lazily even when --no-qat was passed at the
        # CLI; phase 1-3 always run float (matches Quantizr's recipe).
        total_phase_epochs = boundaries[-1]
        # Override args.epochs to the schedule sum so resume / log accounting
        # stays consistent. The legacy loop reads args.epochs as the cap.
        args.epochs = total_phase_epochs
        print(
            f"[train] 5-phase Quantizr-style schedule active. "
            f"Boundaries: {boundaries} (total {total_phase_epochs} epochs). "
            + " | ".join(
                f"P{i}({PHASE_NAMES[i]}):"
                f"{getattr(args, f'phase{i}_epochs')}ep@{getattr(args, f'phase{i}_lr'):.1e}"
                for i in range(1, 6)
                if getattr(args, f"phase{i}_epochs") > 0
            )
        )

    # Wrap with FP4 QAT if enabled. R-FP4-fix: pass codebook + robustness flags
    # so the trend report's float→FP4 gap (93.44 plateau while float kept
    # improving) can be closed by --fp4-codebook=residual --fp4-stochastic
    # --fp4-robust-scale on residual-head renderers.
    #
    # 5-phase note: when use_5phase is True, QAT is deferred until Phase 4 by
    # default (Quantizr recipe). If the user passes --use-qat AND we're in
    # 5-phase mode, QAT activates immediately for backwards-compat.
    qat_wrapper = None
    qat_active = False
    qat_phase4_pending = use_5phase and args.use_qat and args.phase4_epochs > 0
    if args.use_qat and not qat_phase4_pending:
        from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
        codebook = (RESIDUAL_CODEBOOK if args.fp4_codebook == "residual"
                    else DEFAULT_CODEBOOK).clone()
        qat_wrapper = QATRendererFP4(
            model,
            codebook=codebook,
            stochastic=args.fp4_stochastic,
            robust_scale=args.fp4_robust_scale,
        ).to(device)
        qat_active = True
        print(f"[train] FP4 QAT enabled (codebook={args.fp4_codebook}, "
              f"stochastic={args.fp4_stochastic}, "
              f"robust_scale={args.fp4_robust_scale})")
    elif qat_phase4_pending:
        print(
            "[train] FP4 QAT deferred until Phase 4 "
            f"(starts epoch {boundaries[2] if len(boundaries) >= 3 else 0})"
        )

    # Training infrastructure
    ema = EMA(model, decay=args.ema_decay)
    if entropy_bottleneck is not None:
        eb_param_ids = {id(p) for p in entropy_bottleneck.parameters()}
        main_params = [p for p in model.parameters() if id(p) not in eb_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": main_params, "lr": args.lr, "weight_decay": 1e-4},
                {
                    "params": list(entropy_bottleneck.parameters()),
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                },
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # T_max accounts for both warmup and pretrain phases so the cosine
    # schedule covers only the scorer fine-tuning (Phase 2) epochs.
    _tmax = max(1, args.epochs - args.warmup_epochs - args.pretrain_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_tmax, eta_min=1e-5,
    )

    # Pair indices
    all_pair_starts = pair_start_indices(n_frames)
    n_total = len(all_pair_starts)
    qfaithful_training_poses, qfaithful_training_pose_contract = (
        _load_qfaithful_training_poses(args, n_pairs=n_total, device=device)
    )
    args.qfaithful_training_pose_contract = qfaithful_training_pose_contract
    train_size = max(1, n_total // args.subsample)
    print(f"[train] {args.epochs} epochs (pretrain={args.pretrain_epochs}), "
          f"{train_size}/{n_total} pairs/epoch, "
          f"accum={args.accum_steps}, lr={args.lr}, depth={args.depth}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "no_mask_encoder", False):
        (out_dir / "T2_DROP_INFORMATIONAL_ONLY.txt").write_text(
            "informational-only: Lane T2-DROP --no-mask-encoder ablation; "
            "not a stacking candidate.\n"
        )

    # P0: Precompute GT scorer outputs (constant -- frames and scorers are frozen)
    print("[train] P0: Precomputing GT scorer cache...")
    gt_scorer_cache = {}
    with torch.no_grad():
        for start in all_pair_starts:
            gt_pair = pair_from_frames(gt_frames, start).to(device)
            gx = _hwc_to_chw(gt_pair)
            gp_out, gs_out = scorer_forward_pair(gx, posenet, segnet)
            gt_scorer_cache[start] = {
                "pose_6": gp_out["pose"][..., :6].cpu(),
                "seg_soft": F.softmax(gs_out, dim=1).cpu(),
            }
            del gt_pair, gx, gp_out, gs_out
    cache_bytes = sum(
        v["pose_6"].numel() * v["pose_6"].element_size()
        + v["seg_soft"].numel() * v["seg_soft"].element_size()
        for v in gt_scorer_cache.values()
    )
    print(f"[train] P0: Cached {len(gt_scorer_cache)} GT scorer outputs ({cache_bytes / 1e6:.1f}MB)")
    if args.eval_roundtrip:
        print("[train] eval_roundtrip=True — GT cache DISABLED, roundtrip simulation active (noise_std=0.5)")
    else:
        print("[train] eval_roundtrip=False — using GT scorer cache (WARNING: proxy-auth gap will be large)")

    best_scorer = float("inf")
    best_epoch = -1
    start_epoch = 0
    baseline_pose = None
    baseline_seg = None
    start_wall_time = time.monotonic()

    # ── Training state save/resume (Feature 6) ────────────────────────
    current_epoch = 0  # updated each epoch for signal handler visibility

    def save_training_state(path=None):
        if path is None:
            path = out_dir / f"training_state_{args.tag}.pt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".pt.tmp")
        # Codex R-Lane-D-Issue1 (2026-04-27): persist arch_meta so the
        # _peek_checkpoint_arch_meta() resume path can override the profile
        # pose_dim (and friends) before model construction. Without this,
        # legacy resumes built FiLM layers and crashed on strict load.
        # schema_version bumps when an arch field is added/removed.
        arch_meta = {
            "schema_version": 1,
            "pose_dim": int(getattr(args, "pose_dim", 0) or 0),
            "base_ch": args.base_ch,
            "mid_ch": args.mid_ch,
            "embed_dim": args.embed_dim,
            "motion_hidden": args.motion_hidden,
            "depth": args.depth,
            "use_zoom_flow": bool(args.use_zoom_flow),
            "use_dsconv": bool(args.use_dsconv),
            "use_ghost": bool(args.use_ghost),
            "use_dilation": bool(args.use_dilation),
            "padding_mode": args.padding_mode,
            "variant": args.variant,
            "profile": getattr(args, "profile", None),
            "use_t2_xpred_loss": bool(getattr(args, "use_t2_xpred_loss", False)),
            "use_t2_mask": bool(getattr(args, "use_t2_mask", False)),
            "no_mask_encoder": bool(getattr(args, "no_mask_encoder", False)),
            "informational_only": bool(getattr(args, "no_mask_encoder", False)),
            "distillation_policy": getattr(args, "distillation_policy_provenance", None),
            "distillation_policy_sha256": getattr(args, "distillation_policy_sha256", None),
        }
        _qfaithful_pose_contract = getattr(args, "qfaithful_training_pose_contract", None)
        if _qfaithful_pose_contract is not None:
            arch_meta["qfaithful_training_pose_contract"] = _qfaithful_pose_contract
            arch_meta["training_pose_contract"] = _qfaithful_pose_contract
        torch.save({
            "epoch": current_epoch,
            "model": model.state_dict(),
            "ema_shadow": ema.shadow,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_scorer": best_scorer,
            "best_epoch": best_epoch,
            "baseline_pose": baseline_pose,
            "baseline_seg": baseline_seg,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "arch_meta": arch_meta,
            "qfaithful_training_pose_contract": _qfaithful_pose_contract,
            "training_pose_contract": _qfaithful_pose_contract,
            "distillation_policy": getattr(args, "distillation_policy_provenance", None),
            "distillation_policy_sha256": getattr(args, "distillation_policy_sha256", None),
        }, tmp_path)
        tmp_path.rename(path)  # atomic on POSIX

    # Resume from checkpoint if specified
    if args.resume_from and Path(args.resume_from).exists():
        # Round 11 Finding 1 fix (2026-04-28): defence-in-depth magic-byte
        # detection. Lane WC v1 piped renderer.bin (ASYM/FP4A binary) into
        # --resume-from; torch.load died with an opaque pickle error after
        # hours of feature extraction. Refuse those magics with a clear
        # actionable message BEFORE attempting torch.load.
        _RENDERER_BIN_MAGICS = (
            b"FP4A", b"ASYM", b"DPSM", b"I4LZ",
            b"CCh1", b"C3R1", b"SCv1", b"OMG1",
        )
        with open(args.resume_from, "rb") as _f:
            _magic = _f.read(4)
        if _magic in _RENDERER_BIN_MAGICS:
            raise ValueError(
                f"--resume-from {args.resume_from!r} is a quantised renderer "
                f"binary (magic={_magic!r}); cannot resume from quantised "
                f"binary. Use the float checkpoint "
                f"(training_state_*.pt or renderer_*_best_fp32.pt) produced "
                f"by the original training run instead. (See "
                f"feedback_dead_flag_wiring_pattern + Round 11 Finding 1.)"
            )
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        # R-resume-fix: support both "model" key (training_state save format) AND
        # "model_state_dict" key (fp32 best save format) — Bug A from council R2.
        # Falls back gracefully if either is missing.
        model_state = state.get("model", state.get("model_state_dict", state))
        model.load_state_dict(model_state)
        ema.shadow = {k: v.to(device) for k, v in state["ema_shadow"].items()}
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        else:
            print("[train] Note: checkpoint has no optimizer state, using fresh optimizer")
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        else:
            print("[train] Note: checkpoint has no scheduler state, using fresh scheduler")
        start_epoch = state.get("epoch", 0) + 1
        best_scorer = state.get("best_scorer", float("inf"))
        best_epoch = state.get("best_epoch", -1)
        baseline_pose = state.get("baseline_pose")
        baseline_seg = state.get("baseline_seg")
        print(f"[train] Resumed from epoch {start_epoch - 1}, best {best_scorer:.4f}")

    # ── Emergency save signal handlers (Feature 2) ────────────────────
    setup_signal_handlers(save_training_state)

    has_pretrain = args.pretrain_epochs > 0
    # Track per-pair difficulty for Phase 5 hard-pair emphasis sampling.
    # Initialized uniform; updated each epoch from per-step scorer loss.
    pair_difficulty = torch.ones(n_total, dtype=torch.float32)
    last_phase = 0

    # Lane W: optional per-pair loss weights (produced offline by
    # experiments/profile_pair_sensitivity.py). When None, no scaling is
    # applied (legacy behaviour). When loaded, the (n_total,) tensor must
    # match the pair count exactly — the per-step training loop indexes
    # via pair_idx_int (== start // 2, the canonical 0..n_total-1 pair id).
    pair_loss_weights: torch.Tensor | None = None
    pair_loss_weights_path = getattr(args, "pair_loss_weights", None)
    if pair_loss_weights_path:
        _plw_p = Path(pair_loss_weights_path)
        if not _plw_p.exists():
            raise FileNotFoundError(
                f"--pair-loss-weights file not found: {_plw_p}"
            )
        _plw = torch.load(_plw_p, map_location="cpu", weights_only=False)
        if not isinstance(_plw, torch.Tensor):
            raise TypeError(
                f"--pair-loss-weights at {_plw_p} must be a torch.Tensor, "
                f"got {type(_plw).__name__}"
            )
        if _plw.ndim != 1 or _plw.shape[0] != n_total:
            raise ValueError(
                f"--pair-loss-weights shape {tuple(_plw.shape)} does not match "
                f"dataset's n_total={n_total}. Re-run profile_pair_sensitivity.py "
                f"on the same masks/video pairing used for training."
            )
        if not torch.isfinite(_plw).all():
            raise ValueError(
                f"--pair-loss-weights contains non-finite entries (NaN/Inf) at "
                f"{_plw_p}; refuse to train against a corrupt weight vector."
            )
        if (_plw < 0).any():
            raise ValueError(
                f"--pair-loss-weights contains negative entries at {_plw_p}; "
                f"per-pair loss multipliers must be >= 0."
            )
        pair_loss_weights = _plw.to(torch.float32)
        print(
            f"[train] Lane W: loaded {n_total} per-pair weights from "
            f"{_plw_p}  min={pair_loss_weights.min().item():.3f} "
            f"max={pair_loss_weights.max().item():.3f} "
            f"mean={pair_loss_weights.mean().item():.3f} "
            f"hard={(pair_loss_weights > 1.0).sum().item()} pairs"
        )

    # Lane W-V2 (2026-04-27): wrap pair_loss_weights in a learnable
    # parameter group when --learnable-pair-weights is set. The static
    # tensor becomes the warm-start; the LearnablePairWeights module
    # provides differentiable per-pair scalars + a Lagrangian rate
    # penalty so the optimiser redistributes weight mass during training.
    learnable_pair_weights = None
    if getattr(args, "learnable_pair_weights", False):
        if pair_loss_weights is None:
            raise ValueError(
                "--learnable-pair-weights requires --pair-loss-weights "
                "as the warm-start (Lane W-V2 cannot bootstrap a "
                "continuous weight distribution from scratch)."
            )
        from tac.learnable_pair_weights import LearnablePairWeights
        learnable_pair_weights = LearnablePairWeights(
            n_pairs=n_total, warm_start=pair_loss_weights,
        ).to(device)
        print(
            f"[train] Lane W-V2: --learnable-pair-weights ACTIVE — "
            f"wrapped {n_total} per-pair weights in LearnablePairWeights "
            f"(lr={args.learnable_pair_weights_lr}, "
            f"rate_lambda={args.learnable_pair_weights_rate_lambda}). "
            f"Static --pair-loss-weights tensor IGNORED at runtime; "
            f"warm-start applied at module init."
        )

    # Lane PS-V2 (2026-04-27): wrap --segnet-class-weights in a learnable
    # softmax-parameterised parameter group. Warm-start = the parsed CSV
    # tensor (or uniform when no CSV supplied). The optimiser equalises
    # per-class distortion variance via a Lagrangian penalty.
    learnable_segnet_class_weights = None
    if getattr(args, "learnable_segnet_class_weights", False):
        from tac.learnable_class_weights import LearnableClassWeights
        ws = getattr(args, "_segnet_class_weights_tensor", None)
        learnable_segnet_class_weights = LearnableClassWeights(
            num_classes=5, warm_start=ws,
        ).to(device)
        print(
            f"[train] Lane PS-V2: --learnable-segnet-class-weights ACTIVE "
            f"— softmax-parameterised 5-vector "
            f"(lr={args.learnable_segnet_class_weights_lr}, "
            f"var_lambda={args.learnable_segnet_class_weights_var_lambda}, "
            f"warm_start={None if ws is None else ws.tolist()})."
        )

    # Round 11 Finding 2 fix (2026-04-28, anti-arbitrariness):
    # LearnablePairWeights and LearnableClassWeights expose ONLY buffers
    # (lambda_pair, lambda_class) — no nn.Parameter. The Round 10 rewrite
    # moved them from raw-Parameter softplus/softmax to projected
    # dual-ascent. Adding `module.parameters()` to the optimiser was a
    # silent NO-OP (empty list) that hid the fact that nothing learns via
    # backprop. We now drive adaptation via explicit dual_update() calls
    # in the per-step loop below (eta = the *_lr argument, reused as the
    # dual step size). The empty-param-group branch is removed so the
    # optimiser layout remains unchanged when the flags are off.
    if learnable_pair_weights is not None or learnable_segnet_class_weights is not None:
        print(
            "[train] Lane W-V2/PS-V2: dual-ascent adaptation ACTIVE — "
            "lambdas updated via explicit dual_update() per step "
            f"(pair_eta={float(getattr(args, 'learnable_pair_weights_lr', 0.0))}, "
            f"class_eta={float(getattr(args, 'learnable_segnet_class_weights_lr', 0.0))}). "
            "No optimizer param groups added (modules expose only buffers)."
        )

    # Lane S V2 (auto-warmup): scorer-loss convergence detector. Built only
    # when both --use-self-compress-codec AND --auto-warmup-lambda are set.
    # When constructed, takes precedence over the static
    # --self-compress-lambda-ramp-start-frac fraction (the per-epoch SC
    # block below checks `auto_warmup_detector` first, falling back to the
    # static fraction otherwise so existing profiles stay byte-identical).
    auto_warmup_detector = None
    if (
        getattr(args, "use_self_compress_codec", False)
        and getattr(args, "auto_warmup_lambda", False)
    ):
        # Note: helper lives at tac.scorer_loss_convergence_detector
        # because src/tac/training.py is a flat module; the canonical
        # `tac.training.ScorerLossConvergenceDetector` re-export is
        # active too (training.py bottom).
        from tac.scorer_loss_convergence_detector import (
            ScorerLossConvergenceDetector,
        )
        auto_warmup_detector = ScorerLossConvergenceDetector(
            window=int(args.auto_warmup_window),
            slope_tolerance=float(args.auto_warmup_slope_tol),
            min_warmup_epochs=int(args.auto_warmup_min_epochs),
            require_decreasing=True,
        )
        print(
            f"[lane-s-v2] AUTO-WARMUP LAGRANGIAN ACTIVE: "
            f"window={args.auto_warmup_window} "
            f"slope_tol={args.auto_warmup_slope_tol} "
            f"min_warmup_epochs={args.auto_warmup_min_epochs} "
            f"(static --self-compress-lambda-ramp-start-frac="
            f"{args.self_compress_lambda_ramp_start_frac} is now an "
            f"UPPER-BOUND fallback; the detector fires earlier when the "
            f"scorer loss has actually plateaued — Csefalvay 2023 §3.2 "
            f"+ Polyak-Juditsky 1992)",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        epoch_start = time.monotonic()
        model.train()
        in_pretrain = has_pretrain and epoch < args.pretrain_epochs
        if t2_mask_hook is not None:
            t2_mask_hook.set_training_progress(epoch / max(args.epochs, 1))

        # Lane V-V2 (2026-04-27): compute the per-epoch half-frame warp
        # probability ONCE per epoch (the inner step loop reads
        # `current_mask_half_sim_prob` so the schedule applies uniformly
        # across all batches in the epoch). When no schedule is set, this
        # equals the static profile/CLI value (V1 behaviour, byte-identical).
        current_mask_half_sim_prob = mask_half_sim_prob_for_epoch(
            epoch, args.epochs,
            static_prob=float(getattr(args, "mask_half_sim_prob", 0.0) or 0.0),
            schedule=getattr(args, "mask_half_sim_prob_anneal", None),
        )

        # 5-phase Quantizr schedule: derive phase + per-phase LR from cosine
        # within the current phase. Each phase has its own optimizer.lr
        # baseline (--phaseN-lr). Cosine annealing applies WITHIN the phase.
        if use_5phase:
            phase_idx = current_phase(epoch, boundaries)
            in_pretrain = phase_idx == 1   # Phase 1 still pixel-only
            # Phase transition logging + Phase 4 lazy QAT activation.
            if phase_idx != last_phase:
                print(
                    f"[train] === Phase {phase_idx} ({PHASE_NAMES[phase_idx]}) "
                    f"start at epoch {epoch} === lr={lr_for_phase(args, phase_idx):.2e}"
                )
                if phase_idx == 4 and qat_phase4_pending and not qat_active:
                    from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
                    codebook = (
                        RESIDUAL_CODEBOOK if args.fp4_codebook == "residual"
                        else DEFAULT_CODEBOOK
                    ).clone()
                    qat_wrapper = QATRendererFP4(
                        model,
                        codebook=codebook,
                        stochastic=args.fp4_stochastic,
                        robust_scale=args.fp4_robust_scale,
                    ).to(device)
                    qat_active = True
                    print(
                        "[train] === Phase 4 QAT activated "
                        f"(FakeQuantFP4 codebook={args.fp4_codebook}, "
                        f"stochastic={args.fp4_stochastic}, "
                        f"robust_scale={args.fp4_robust_scale}) ==="
                    )
                last_phase = phase_idx
            base_lr = lr_for_phase(args, phase_idx)
            phase_step, phase_total = phase_epoch_offset(
                epoch, phase_idx, boundaries,
            )
            lr = cosine_lr(base_lr, phase_step, phase_total)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            # Legacy 2-phase path (unchanged).
            phase_idx = 1 if in_pretrain else 2
            # Warmup LR
            if epoch < args.warmup_epochs:
                lr = args.lr * (epoch + 1) / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            # Log phase transition
            if has_pretrain and epoch == args.pretrain_epochs:
                print(f"[train] === Phase 2 start (epoch {epoch}) === switching to scorer loss")

        # Sample pairs for this epoch. Phase 5 (final): hard-pair emphasis.
        # Probability of sampling pair i is proportional to pair_difficulty[i].
        # In all other phases the legacy uniform-random sampling is used.
        if use_5phase and phase_idx == 5 and pair_difficulty.sum() > 0:
            probs = pair_difficulty.clamp(min=1e-6)
            probs = probs / probs.sum()
            perm = torch.multinomial(probs, train_size, replacement=False)
        else:
            perm = torch.randperm(n_total)[:train_size]

        total_loss, total_pose, total_seg = 0.0, 0.0, 0.0
        # Lane D-V3 (2026-04-27) Phase-0 instrumentation: count how many
        # batches in this epoch actually fire the half-frame warp branch
        # and accumulate motion-module activation stats. Verifies the
        # mechanism end-to-end (a triggered branch must produce a non-zero
        # warped-mask diff to be useful — if the warp returns identity or
        # zeros, the renderer is training on an unintended distribution).
        # Logged once per epoch in the [train] line below at log_every cadence.
        # Zero-overhead when sim_zoom_warp is None (Lane D / Lane V profiles
        # only).
        halfframe_branch_fires = 0
        halfframe_warp_diff_sum = 0.0
        halfframe_warp_diff_count = 0
        optimizer.zero_grad(set_to_none=True)
        accum = args.accum_steps

        for step, pair_idx in enumerate(perm):
            mask_t = mask_t1 = gt_pair = rendered_pair = None
            start = all_pair_starts[pair_idx.item()]

            # Load mask pair and GT pair. R-mask-aug 2026-04-25: with prob
            # --mask-noise-prob, swap to the AV1-quantized variant so the
            # renderer trains against the same mask distribution it sees at
            # inflate time. The GT pair is unchanged (we still measure the
            # renderer against the true frames — only the mask input is noisy).
            if (noisy_masks is not None
                    and random.random() < args.mask_noise_prob):
                mask_t, mask_t1 = training_mask_pair_from_index(
                    noisy_masks,
                    start,
                    pair_index_basis=noisy_masks_pair_index_basis,
                )
            else:
                mask_t, mask_t1 = mask_pair_from_index(all_masks, start)
            mask_t = mask_t.to(device)
            mask_t1 = mask_t1.to(device)

            # Compute pair index ONCE (shared by half-frame sim AND ego_flow).
            # Pairs are formed as (frame[2k], frame[2k+1]); pair_starts is
            # range(0, n_frames-1, SEQ_LEN=2), so pair_idx = start // 2.
            pair_idx_int = start // 2
            pair_idx_t = torch.tensor([pair_idx_int], device=device,
                                      dtype=torch.long)
            qfaithful_pose = None
            if qfaithful_training_poses is not None:
                qfaithful_pose = qfaithful_training_poses[pair_idx_int:pair_idx_int + 1]

            # Half-frame mask simulation (Quantizr paradigm, Lane D2). With prob
            # --mask-half-sim-prob, replace mask_t with inverse_warp(mask_t1)
            # so this pair simulates inflate-time conditions where only the
            # odd-frame mask was archived. The GT pair (gt_pair below) is
            # unchanged — we still measure renderer output against the true
            # frames; only the mask INPUT distribution shifts.
            #
            # Lane V-V2: ``current_mask_half_sim_prob`` is computed once per
            # epoch (above the step loop) — it equals the static profile/CLI
            # value when no annealing schedule is set, OR the per-epoch
            # interpolated value when ``mask_half_sim_prob_anneal`` is active.
            if (sim_zoom_warp is not None
                    and random.random() < current_mask_half_sim_prob):
                # Lane D-V3 instrumentation: capture pre-warp mask for diff stat.
                # mask_t at this point is from independent SegNet extraction
                # (or the noisy AV1 variant); mask_t (post-warp) replaces it
                # with warp_inverse(mask_t1). The renderer must learn to handle
                # the difference. If diff~0 the warp is degenerate (zero zoom,
                # identity motion) and the half-frame branch is a no-op.
                _premask_for_diff = mask_t.detach().float()
                mask_t = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx_t)
                halfframe_branch_fires += 1
                with torch.no_grad():
                    _diff = (mask_t.detach().float() - _premask_for_diff).abs().mean()
                    halfframe_warp_diff_sum += float(_diff.item())
                    halfframe_warp_diff_count += 1
                _premask_for_diff = None

            gt_pair = pair_from_frames(gt_frames, start).to(device)

            # Horizontal flip augmentation (50% probability)
            flipped_h = False
            if qfaithful_training_poses is None and random.random() < 0.5:
                # mask shape is (B, H, W) so W is dim -1;
                # gt_pair shape is (B, 2, H, W, 3) so W is dim -2
                # (trailing channel dim shifts the index by one)
                mask_t = mask_t.flip(-1)
                mask_t1 = mask_t1.flip(-1)
                gt_pair = gt_pair.flip(-2)
                flipped_h = True

            # ego_flow plumbing (Lane D, council 2026-04-27). When the model
            # was built with use_zoom_flow=True (AsymmetricPairGenerator with
            # 4-channel motion output), forward() REQUIRES an ego_flow tensor.
            # Compute it from the same RadialZoomWarp used for half-frame sim
            # so train and inflate see identical motion structure.
            #
            # Horizontal-flip handling: a mirrored pair has its motion x-axis
            # negated (motion that went right now goes left). We mirror the
            # flow field on dim=-1 (W) AND negate the x-component (channel 0).
            ego_flow = None
            if (sim_zoom_warp is not None
                    and getattr(args, "use_zoom_flow", False)):
                with torch.no_grad():
                    H_m, W_m = mask_t1.shape[-2], mask_t1.shape[-1]
                    ego_flow = sim_zoom_warp(pair_idx_t, H_m, W_m)  # (1, 2, H, W)
                if flipped_h:
                    ego_flow = ego_flow.flip(-1)
                    ego_flow = torch.stack(
                        [-ego_flow[:, 0], ego_flow[:, 1]], dim=1
                    )

            if saug_v2_sampler is not None:
                saug_v2_sigmas = saug_v2_sampler.sample_sigmas(
                    int(mask_t.shape[0]), device,
                )
                mask_t = _apply_saug_v2_to_renderer_input(mask_t, saug_v2_sigmas)
                mask_t1 = _apply_saug_v2_to_renderer_input(mask_t1, saug_v2_sigmas)

            # Lane MAE-V: train-only patch masking with learnable token.
            # Eval-mode passthrough is enforced inside the augmenter.
            # MAEMaskAugmenter expects (B, H, W) long; both mask_t and mask_t1
            # already have that shape after mask_pair_from_index.
            if mae_mask_augmenter is not None:
                mask_t, _ = mae_mask_augmenter(mask_t)
                mask_t1, _ = mae_mask_augmenter(mask_t1)

            # Forward: render pair from masks. Only AsymmetricPairGenerator
            # accepts ego_flow (see renderer.py:1035-1100); the legacy
            # PairGenerator.forward() takes only (mask_t, mask_t1). The
            # _QuantizrFaithfulShim REQUIRES `pose=` kwarg AND swallows
            # `ego_flow` via **_kwargs. The q_faithful_dilated_88k profile
            # sets BOTH use_zoom_flow=True (so ego_flow is computed) AND
            # variant=quantizr_faithful (so qfaithful_pose is loaded);
            # historically the ego_flow branch dropped pose and the shim
            # raised at the first forward step. Now: when both are present,
            # pass both — the shim's **_kwargs swallows ego_flow harmlessly
            # while still receiving the required pose. Branch on presence
            # rather than model class so this stays variant-agnostic.
            forward_kwargs: dict = {}
            if ego_flow is not None:
                forward_kwargs["ego_flow"] = ego_flow
            if qfaithful_pose is not None:
                forward_kwargs["pose"] = qfaithful_pose
            rendered_pair = model(mask_t, mask_t1, **forward_kwargs)

            if in_pretrain:
                # Phase 1: L1 + edge loss only -- no scorer, much faster
                if args.use_t2_xpred_loss:
                    loss = x_prediction_loss(
                        rendered_pair, gt_pair,
                        sigma=args.t2_xpred_sigma,
                        weighting=args.t2_xpred_weighting,
                    )
                else:
                    loss = pretrain_loss(rendered_pair, gt_pair)
                pd, sd = 0.0, 0.0
            else:
                # eval_roundtrip: simulate contest eval resize chain before scorer
                if args.eval_roundtrip:
                    from tac.camera import CAMERA_H, CAMERA_W
                    gt_pair = resize_pair_hwc(gt_pair, rendered_pair.shape[2], rendered_pair.shape[3])
                    # rendered_pair is (B, 2, H, W, 3) — flatten to (B*2, 3, H, W) for roundtrip
                    rp_flat = rendered_pair.reshape(-1, *rendered_pair.shape[2:]).permute(0, 3, 1, 2).contiguous()
                    rp_flat = simulate_eval_roundtrip(rp_flat, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5)
                    B_r, C_r, H_r, W_r = rp_flat.shape
                    rendered_pair = rp_flat.permute(0, 2, 3, 1).reshape(-1, 2, H_r, W_r, C_r)

                    gt_flat = gt_pair.reshape(-1, *gt_pair.shape[2:]).permute(0, 3, 1, 2).contiguous()
                    gt_flat = simulate_eval_roundtrip(gt_flat, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0)
                    gt_pair = gt_flat.permute(0, 2, 3, 1).reshape(-1, 2, H_r, W_r, C_r)

                # Phase 2: Scorer loss (use cached GT scorer outputs when available)
                # Skip GT cache when eval_roundtrip is on — cached values were
                # computed without roundtrip, so they don't match the roundtripped
                # gt_pair. Force recomputation through the roundtripped gt_pair.
                _cached_gt = None if args.eval_roundtrip else gt_scorer_cache.get(start)
                # Lane PS: per-class SegNet weights tensor (None = uniform
                # / byte-identical to baseline). Parsed once at args
                # resolver time so this is a cheap attribute read.
                _pcw = getattr(args, "_segnet_class_weights_tensor", None)
                # Round 11 Finding 2 (2026-04-28, anti-arbitrariness): if the
                # learnable class-weights module is active, OVERRIDE the
                # static tensor with its current adaptive weights so the
                # loss actually reflects the learned distribution. Without
                # this thread the LearnableClassWeights module would be
                # SILENTLY IGNORED — Lane PS-V2 deploys would have appeared
                # to learn while their weights never reached the loss.
                _scorer_aux = None
                if learnable_segnet_class_weights is not None:
                    _pcw = learnable_segnet_class_weights.weights().detach()
                _seg_kwargs = {
                    "segmentation_surrogate": args.segmentation_surrogate,
                    "segmentation_temperature": args.segmentation_temperature,
                    "fisher_rao_eps": args.fisher_rao_eps,
                }
                # When either learnable module is active, use the *_with_aux
                # variant which returns per-pair pose distortion and
                # per-class SegNet distortion (detached) for the explicit
                # dual_update calls below. Byte-identical to the legacy path
                # otherwise (loss/pd/sd identical, no extra compute).
                _need_aux = (
                    learnable_pair_weights is not None
                    or learnable_segnet_class_weights is not None
                )
                if args.use_t2_xpred_loss:
                    loss = x_prediction_loss(
                        rendered_pair, gt_pair,
                        sigma=args.t2_xpred_sigma,
                        weighting=args.t2_xpred_weighting,
                    )
                    pd, sd = 0.0, 0.0
                elif _cached_gt is not None:
                    _gt_pose_6 = _cached_gt["pose_6"].to(device)
                    _gt_seg_soft = _cached_gt["seg_soft"].to(device)
                    if _need_aux:
                        from tac.losses import scorer_loss_cached_with_aux
                        loss, pd, sd, _scorer_aux = scorer_loss_cached_with_aux(
                            rendered_pair, _gt_pose_6, _gt_seg_soft, posenet, segnet,
                            class_weights=_pcw,
                            **_seg_kwargs,
                        )
                    else:
                        loss, pd, sd = scorer_loss_cached(
                            rendered_pair, _gt_pose_6, _gt_seg_soft, posenet, segnet,
                            class_weights=_pcw,
                            **_seg_kwargs,
                        )
                    del _gt_pose_6, _gt_seg_soft
                else:
                    if _need_aux:
                        from tac.losses import scorer_loss_with_aux
                        loss, pd, sd, _scorer_aux = scorer_loss_with_aux(
                            rendered_pair, gt_pair, posenet, segnet,
                            class_weights=_pcw,
                            **_seg_kwargs,
                        )
                    else:
                        loss, pd, sd = scorer_loss(
                            rendered_pair, gt_pair, posenet, segnet,
                            class_weights=_pcw,
                            **_seg_kwargs,
                        )

            # Trick 3: Even-frame SegNet skip
            # If frame_t1 (start+1) is even-indexed, SegNet won't evaluate it.
            # Scale loss to reduce SegNet contribution (PoseNet-only emphasis).
            if not in_pretrain and args.even_frame_skip_seg and (start + 1) % 2 == 0:
                loss = loss * 0.5

            # Trick 2: Frequency-domain wavelet loss
            if not in_pretrain and args.frequency_loss_weight > 0:
                freq_loss = frequency_aware_loss(rendered_pair, gt_pair)
                loss = loss + args.frequency_loss_weight * freq_loss

            # Fridrich inverse-steganalysis losses (R-fridrich-wire 2026-04-25):
            # WILDE/SHIRAZ/GREEN profiles SET these flags but train_renderer.py
            # never consumed them. Per Fridrich council R41 audit: "every
            # renderer training has been UNIFORM-loss — equal error budget on
            # smooth sky pixels and textured curb pixels. Estimated leak:
            # SegNet 0.0024 → 0.0010 = -0.14 score." Wire them up:
            #   - texture_loss: UNIWARD wavelet cost — hide errors in textured regions
            #   - linf_penalty: square root law — spread small errors, don't concentrate
            #   - markov_loss: HUGO-style local gradient continuity
            #
            # 5-phase Quantizr schedule (Quantizr R2 C3, 2026-04-26): the full
            # Fridrich + KL + uncertainty stack is gated to Phase 3+ so Phase
            # 2 cleanly isolates "scorer-only" finetune (matches Quantizr's
            # published "finetune" stage). Legacy 2-phase mode keeps the
            # historical behaviour (active throughout post-pretrain).
            fridrich_active = (not in_pretrain) and (
                (not use_5phase) or phase_idx >= 3
            )
            if fridrich_active:
                if args.use_texture_loss and args.texture_loss_weight > 0:
                    # R-texture-fix: original implementation used .var(dim=-1)
                    # which is variance OVER 3 RGB CHANNELS at each pixel — NOT
                    # local spatial variance, so the "down-weight textured regions"
                    # claim was false. Compute true local variance via 8x8 box-
                    # filter on luma, then weight L1 reconstruction error by the
                    # inverse: textured regions (high local var) → low weight (errors
                    # there are invisible to the scorer, FREE bits per UNIWARD).
                    rgb_pred = rendered_pair[:, 1].float()  # (B, H, W, 3)
                    rgb_gt = gt_pair[:, 1].float()
                    # Per-pixel L1 (averaged over channels)
                    diff = (rgb_pred - rgb_gt).abs().mean(dim=-1)  # (B, H, W)
                    # Codex R5-2 Finding #2 (2026-04-27): dead import fix —
                    # `tac.fridrich.luma_local_variance` was a planned helper
                    # that never landed; this code path silently raised
                    # ImportError every time `use_texture_loss=True` (set by
                    # WILDE/SHIRAZ/GREEN/DEN/Lane D profiles). The semantically
                    # equivalent helper is `compute_texture_complexity` in
                    # `tac.fridrich_losses` (HWC-aware, BT.601 luma, sliding
                    # box variance — same UNIWARD-style local variance map).
                    # It returns (B, 1, H, W); we squeeze to (B, H, W) to
                    # match the broadcast against `diff` below.
                    #
                    # Kernel choice: 7 (odd) instead of 8 (even). Even kernel
                    # + reflect-pad in avg_pool2d gains 1 pixel on each axis
                    # (H+2*pad-k+1 = H+1), which then breaks the diff x inv_var
                    # broadcast. The original "8" was the source of the
                    # 1-pixel-asymmetry bug the docstring above warns about.
                    # Odd kernel of similar radius (k=7 → 3-pixel half-window
                    # vs k=8's 4-pixel) preserves shape exactly and gives the
                    # same UNIWARD signal within rounding error.
                    from tac.fridrich_losses import compute_texture_complexity
                    local_var = compute_texture_complexity(
                        rgb_gt, kernel_size=7,
                    ).squeeze(1)  # (B, H, W)
                    assert local_var.shape == diff.shape, (
                        f"texture_loss shape mismatch: local_var={local_var.shape} "
                        f"vs diff={diff.shape}. compute_texture_complexity "
                        f"contract violated."
                    )
                    # inv_var: high in smooth regions (errors visible, EXPENSIVE),
                    # low in textured regions (errors hidden, FREE). Normalized so
                    # the loss scale stays comparable to vanilla L1.
                    inv_var = 1.0 / (local_var.sqrt() + 8.0)  # +8 = pixel-scale floor
                    inv_var = inv_var / inv_var.mean().clamp(min=1e-8)  # mean=1
                    texture_loss = (diff * inv_var).mean()
                    loss = loss + args.texture_loss_weight * texture_loss
                if args.use_linf_penalty and args.linf_weight > 0:
                    # R-linf-fix: original .amax() reduced over the WHOLE batch
                    # → only 1 pixel got gradient per step (opposite of "spread").
                    # Per-pair top-32 mean = bound the worst pixels per pair so
                    # gradient distributes across pairs and worst regions, which
                    # is what the square-root law actually wants.
                    rgb_pred = rendered_pair.float()
                    rgb_gt = gt_pair.float()
                    pixel_err = (rgb_pred - rgb_gt).abs().mean(dim=-1)  # (B, T, H, W)
                    flat = pixel_err.flatten(start_dim=1)  # (B, T*H*W)
                    topk = flat.topk(min(32, flat.shape[-1]), dim=-1).values  # (B, 32)
                    loss = loss + args.linf_weight * topk.mean()
                if args.use_markov_loss and args.markov_weight > 0:
                    # First-order spatial gradient continuity (HUGO).
                    rgb_pred = rendered_pair[:, 1].float()
                    rgb_gt = gt_pair[:, 1].float()
                    grad_x_pred = rgb_pred[:, :, 1:, :] - rgb_pred[:, :, :-1, :]
                    grad_x_gt = rgb_gt[:, :, 1:, :] - rgb_gt[:, :, :-1, :]
                    grad_y_pred = rgb_pred[:, 1:, :, :] - rgb_pred[:, :-1, :, :]
                    grad_y_gt = rgb_gt[:, 1:, :, :] - rgb_gt[:, :-1, :, :]
                    markov_loss = (grad_x_pred - grad_x_gt).abs().mean() + \
                                  (grad_y_pred - grad_y_gt).abs().mean()
                    loss = loss + args.markov_weight * markov_loss
                # Fridrich council #1 (2026-04-26): JPEG-Q-table-weighted DCT
                # loss. Penalises low-frequency residual energy ~6x more than
                # high-frequency, hiding error in DCT bins the scorers cannot
                # see. Zero-overhead when weight=0 (call site is gated).
                if args.dct_quant_weight > 0:
                    loss = loss + args.dct_quant_weight * dct_quant_loss(
                        rendered_pair, gt_pair,
                    )

                # KL distillation auxiliary — SegNet ONLY (R-fix-double-count).
                # The original wiring used kl_distill_scorer_loss which returns
                # 100*seg_kl + sqrt(10*pose_dist), and adding that to scorer_loss
                # double-counts BOTH terms (200x SegNet, double PoseNet pressure).
                # That matches the historical "KL distill caused PoseNet collapse"
                # failure mode per CLAUDE.md "Critical Lessons". Use the SegNet-
                # only helper that returns ONLY T²-scaled seg_kl.
                #
                # Lane J-JBL (Jack Cycle 1 TOP-1, 2026-04-28): when loss_mode
                # is "jbl", swap the KL distill auxiliary for the combined
                # Jaccard Metric Loss + Boundary Label Smoothing distillation
                # (Wang et al. NeurIPS 2023). The kl_distill_weight knob is
                # repurposed as the JBL master scalar so the wiring stays
                # byte-identical to Lane G v3 except for the loss family.
                # See tac.losses_jbl + .omx/research/jack_skunkworks_segnet_
                # rate_research_20260428.md §S1 for the wedge attribution.
                if args.kl_distill_weight > 0 and args.kl_distill_scope == "segnet_aux":
                    if args.loss_mode == "jbl":
                        from tac.losses_jbl import combined_jbl_distill_loss
                        # Forward both student (renderer-rendered) and teacher
                        # (GT) frames through SegNet to obtain logits. Mirrors
                        # the no-grad teacher pattern in kl_distill_segnet_only.
                        # NOTE: _hwc_to_chw is imported at module level (L63);
                        # do NOT re-import inside train() — it would shadow the
                        # module-level binding and cause UnboundLocalError at
                        # earlier uses (L2357 P0 cache).
                        fx = _hwc_to_chw(rendered_pair)  # KL_RAW_PAIRS_OK:rendered_pair was reassigned to roundtripped output at L1642
                        gx = _hwc_to_chw(gt_pair)
                        fs_in = segnet.preprocess_input(fx)
                        with torch.no_grad():
                            gs_in = segnet.preprocess_input(gx)
                        student_logits = segnet(fs_in)
                        with torch.no_grad():
                            teacher_logits = segnet(gs_in)
                        # GT mask = teacher argmax (5-class). The teacher-as-GT
                        # pattern is the canonical contest-eval surrogate; the
                        # teacher itself is the upstream SegNet's argmax.
                        gt_mask = teacher_logits.argmax(dim=1)  # (B, H, W)
                        jbl_loss, _jbl_tel = combined_jbl_distill_loss(
                            student_logits, teacher_logits, gt_mask,
                            num_classes=5,
                            jaccard_weight=1.0,
                            bls_weight=0.5,
                            boundary_pixel_weight=args.boundary_weight,
                            bls_smoothing=args.bls_smoothing,
                            teacher_temperature=args.kl_distill_temperature,
                        )
                        loss = loss + args.kl_distill_weight * jbl_loss
                    else:
                        kl_loss, _kl_seg = kl_distill_segnet_only(  # KL_RAW_PAIRS_OK:rendered_pair was reassigned to roundtripped output at L1642
                            rendered_pair, gt_pair, segnet,
                            temperature=args.kl_distill_temperature,
                            class_weights=getattr(
                                args, "_segnet_class_weights_tensor", None,
                            ),
                        )
                        loss = loss + args.kl_distill_weight * kl_loss

                # Lane 19 (SegNet logit-margin boundary loss). Fires alongside
                # the KL distill aux (NOT a replacement — see council memo
                # .omx/research/council_lane_19_logit_margin_design_20260430.md
                # §2 Contrarian: confident-WRONG pixels contribute zero to
                # margin loss but the underlying scorer_loss still catches
                # them, so this is wired strictly as an auxiliary).
                #
                # Default weight 0.0 = byte-identical to Lane G v3. Profile
                # LANE_19_LOGIT_MARGIN sets 0.1 to enable. Tagged
                # KL_RAW_PAIRS_OK because rendered_pair was reassigned to
                # the eval-roundtripped output at L1642 (same convention as
                # the KL distill block above).
                if args.logit_margin_weight > 0.0:
                    from tac.losses_logit_margin import compute_segnet_logit_margin_aux
                    lm_loss = compute_segnet_logit_margin_aux(  # KL_RAW_PAIRS_OK:rendered_pair was reassigned to roundtripped output at L1642
                        rendered_pair=rendered_pair,
                        gt_pair=gt_pair,
                        segnet=segnet,
                        threshold=float(args.logit_margin_threshold),
                        reduction="mean",
                    )
                    loss = loss + float(args.logit_margin_weight) * lm_loss

                # Yousfi #5 (council 5/0 vote 2026-04-26): inverse-SegNet-
                # entropy weighted L1 on the SegNet-evaluated frame (index 1
                # of the pair — the upstream scorer takes x[:, -1, ...]).
                # Default weight is 0.0 — no overhead unless profile/CLI
                # explicitly enables it. See module-level note in
                # tac.losses.segnet_uncertainty_weighted_loss.
                if args.use_uncertainty_loss and args.uncertainty_loss_weight > 0:
                    # 2026-04-26 R3 Yousfi: thread uncertainty_loss_floor
                    # from the profile resolver. Without this, the profile
                    # value (e.g. WILDE/SHIRAZ/DEN's 0.1) was silently
                    # discarded — function default won. Same dead-code class
                    # as the kl_distill resolver bug Quantizr R1 fixed.
                    uncert_loss = segnet_uncertainty_weighted_loss(
                        rendered_pair[:, 1], gt_pair[:, 1], segnet,
                        weight_floor=args.uncertainty_loss_floor,
                    )
                    loss = loss + args.uncertainty_loss_weight * uncert_loss

                if args.use_variance_noise and args.variance_noise_weight > 0:
                    variance_loss = uniward_quant_noise_loss(
                        rendered_pair[:, 1],
                        gt_pair[:, 1],
                        base_std=args.variance_noise_base_std,
                        kernel_size=args.variance_noise_kernel,
                        mode=args.variance_noise_mode,
                    )
                    loss = loss + args.variance_noise_weight * variance_loss

            # Lane S: Lagrangian rate penalty on per-channel bit-depth.
            # Active only when the SC codec is on. λ ramps from start to end
            # over training so the renderer learns scorer-sensitive features
            # before being forced to compress them. The penalty itself is
            # normalized by total weights (bits/weight) so the magnitude
            # stays comparable across renderer sizes.
            if getattr(args, "use_self_compress_codec", False):
                from tac.self_compress import compute_renderer_rate_penalty
                _total_epochs = max(1, args.epochs)
                _progress = epoch / max(_total_epochs - 1, 1)
                # Lane S V2: when the auto-warmup detector has fired, the
                # ramp_start is the fraction at which the detector
                # triggered (computed once at fire-time and cached on the
                # detector object). Otherwise fall back to the static
                # --self-compress-lambda-ramp-start-frac.
                if (
                    auto_warmup_detector is not None
                    and auto_warmup_detector.converged_at is not None
                ):
                    _ramp_start = (
                        auto_warmup_detector.converged_at
                        / max(_total_epochs - 1, 1)
                    )
                else:
                    _ramp_start = float(args.self_compress_lambda_ramp_start_frac)
                if _progress < _ramp_start:
                    _lambda = float(args.self_compress_lambda_start)
                else:
                    _ramp_progress = (_progress - _ramp_start) / max(1.0 - _ramp_start, 1e-6)
                    _lambda = (
                        float(args.self_compress_lambda_start)
                        + (float(args.self_compress_lambda_end) - float(args.self_compress_lambda_start))
                        * min(1.0, _ramp_progress)
                    )
                _rate_pen = compute_renderer_rate_penalty(
                    model,
                    target_bits_per_weight=float(args.self_compress_target_bits),
                    lambda_rate=_lambda,
                )
                loss = loss + _rate_pen

            # Lane W (2026-04-27): per-pair loss weighting. Scales BOTH the
            # scorer loss AND the SC Lagrangian penalty (already added above)
            # by pair_loss_weights[pair_idx_int] so the per-channel learnable
            # bit-depth allocation is steered to protect the hardest pairs.
            # No-op when --pair-loss-weights wasn't passed.
            #
            # Lane W-V2 + Round 11 Finding 2 fix (2026-04-28): when
            # --learnable-pair-weights is active, the LearnablePairWeights
            # module exposes ONLY a buffer (lambda_pair) — no Parameter.
            # We therefore (a) multiply the loss by the DETACHED
            # `1+lambda_pair[idx]` scalar so the magnitude is correct
            # without backprop into the buffer, and (b) drive adaptation
            # via an EXPLICIT projected dual-ascent update on the observed
            # per-pair loss. Without (b) this whole branch was a silent
            # no-op (Codex Round 11 Finding 2: "weights never adapt").
            if learnable_pair_weights is not None:
                _w = learnable_pair_weights.weight_for_pair(pair_idx_int).detach()
                loss = loss * _w
                # Streaming dual-ascent: feed the observed scalar loss
                # for THIS pair into the module's running-mean dual update.
                # eta is the *_lr CLI arg (re-purposed; the buffer-only
                # module has no AdamW state, so the "lr" naming is the
                # dual step size).
                _observed = loss.detach().reshape(1)
                learnable_pair_weights.dual_update(
                    _observed,
                    eta=float(args.learnable_pair_weights_lr),
                    pair_idx=int(pair_idx_int),
                )
            elif pair_loss_weights is not None:
                _w = pair_loss_weights[pair_idx_int]
                # Cast on-device so we don't add a CPU/CUDA sync per step.
                loss = loss * _w.to(loss.device)

            # Round 11 Finding 2 fix (2026-04-28): explicit dual update for
            # LearnableClassWeights. The per-class distortion vector comes
            # from the *_with_aux scorer loss (`_scorer_aux`); when it's
            # missing (e.g. xpred path) we skip silently — the lambdas
            # simply don't update for that step, which is correct (no
            # SegNet signal that step).
            if (learnable_segnet_class_weights is not None
                    and _scorer_aux is not None
                    and "per_class_seg_distortion" in _scorer_aux):
                _per_class_d = _scorer_aux["per_class_seg_distortion"]
                if torch.isfinite(_per_class_d).all():
                    learnable_segnet_class_weights.dual_update(
                        _per_class_d,
                        eta=float(args.learnable_segnet_class_weights_lr),
                    )

            if entropy_bottleneck is not None and float(args.eb_lambda) > 0.0:
                loss = loss + float(args.eb_lambda) * entropy_bottleneck.rate_loss()

            scaled_loss = loss / accum

            try:
                scaled_loss.backward()
            except torch.cuda.OutOfMemoryError:
                print(f"[train] CUDA OOM at step {step}, skipping")
                mask_t = mask_t1 = gt_pair = rendered_pair = None
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss += loss.item()
            total_pose += pd
            total_seg += sd

            # Phase 5 hard-pair sampling: track per-pair loss as the
            # difficulty signal. EMA-blended (0.5) so noisy single-step
            # measurements don't dominate. Updated for ALL phases so that
            # by the time Phase 5 starts the difficulty estimate is warm.
            try:
                _pair_idx_int = int(pair_idx.item())
                pair_difficulty[_pair_idx_int] = (
                    0.5 * pair_difficulty[_pair_idx_int]
                    + 0.5 * float(loss.item())
                )
            except (TypeError, ValueError):
                pass

            # Gradient accumulation step (Feature 5: grad clipping already present)
            if (step + 1) % accum == 0 or (step + 1) == len(perm):
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                # Lane S: clamp bit-depth tensor to [0, 8] after each opt step
                if getattr(args, "use_self_compress_codec", False):
                    from tac.self_compress import list_self_compress_layers
                    with torch.no_grad():
                        for _name, _layer in list_self_compress_layers(model):
                            _layer.bit_depth.bits.clamp_(0.0, 8.0)

            mask_t = mask_t1 = gt_pair = rendered_pair = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LR scheduler. 5-phase mode manages LR explicitly per-epoch via
        # cosine_lr() above, so the legacy CosineAnnealingLR is bypassed.
        if not use_5phase and epoch >= args.warmup_epochs:
            scheduler.step()

        n_steps = len(perm)
        avg_loss = total_loss / max(n_steps, 1)
        avg_pose = total_pose / max(n_steps, 1)
        avg_seg = total_seg / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]

        # Lane S V2: feed the per-epoch scorer-loss observation to the
        # auto-warmup detector. Once the detector fires, the SC ramp
        # logic above starts the rate penalty on the NEXT epoch.
        if auto_warmup_detector is not None:
            _just_fired = auto_warmup_detector.observe(epoch, float(avg_loss))
            if (
                _just_fired
                and auto_warmup_detector.converged_at == epoch
            ):
                print(
                    f"[lane-s-v2] AUTO-WARMUP TRIGGERED at epoch {epoch} "
                    f"(slope={auto_warmup_detector.last_slope:.2e} < "
                    f"tol={args.auto_warmup_slope_tol}); Lagrangian rate "
                    f"ramp begins now (would have started at epoch "
                    f"{int(args.self_compress_lambda_ramp_start_frac * args.epochs)} "
                    f"under the static fraction).",
                    flush=True,
                )

        # Per-epoch timing (Feature 7)
        epoch_sec = time.monotonic() - epoch_start

        # Determine current phase label. 5-phase mode uses Quantizr's named
        # phases; legacy mode keeps the historical "pretrain"/"scorer" labels.
        phase = (
            f"phase{phase_idx}_{PHASE_NAMES[phase_idx]}"
            if use_5phase
            else "pretrain"
            if in_pretrain
            else "scorer"
        )

        # FP4 evaluation (skip during Phase 1 -- scorer scores are meaningless)
        is_eval_epoch = (not in_pretrain and
                         ((epoch + 1) % args.eval_every == 0
                          or epoch == args.epochs - 1
                          or epoch == max(start_epoch, args.pretrain_epochs)))
        eval_pose, eval_seg = 0.0, 0.0
        # Codex R-Lane-D-Issue2 (2026-04-27): when the deployed archive will
        # ship only odd-frame masks (Quantizr paradigm — gated on
        # use_zoom_flow=True OR mask_half_sim_prob > 0), evaluate BOTH the
        # full-frame eval (legacy) AND the half-frame eval (mirrors inflate).
        # The half-frame score is the one the deployed model is judged on, so
        # we gate best-checkpoint selection on it. Without this, the best
        # checkpoint optimises a distribution the deployed model never sees
        # and Lane D's predicted 0.55-0.75 score depends on the wrong file
        # shipping. Full-frame is still logged for diagnostic comparison.
        eval_pose_full = eval_seg_full = scorer_val_full = None
        eval_pose_half = eval_seg_half = scorer_val_half = None
        _halfframe_eval_active = (
            sim_zoom_warp is not None
            and (
                getattr(args, "use_zoom_flow", False)
                or float(getattr(args, "mask_half_sim_prob", 0.0)) > 0
            )
        )
        if is_eval_epoch:
            scorer_val_full, eval_pose_full, eval_seg_full = evaluate_fp4(
                model, ema, all_masks, gt_frames,
                all_pair_starts, posenet, segnet, device,
                fp4_codebook=args.fp4_codebook,
                fp4_robust_scale=args.fp4_robust_scale,
                # Lane D: pass the same RadialZoomWarp used during training
                # so the FP4 evaluator sees the same ego_flow structure that
                # the model was trained on (use_zoom_flow=True path).
                sim_zoom_warp=sim_zoom_warp,
                use_zoom_flow=getattr(args, "use_zoom_flow", False),
                half_frame_mode=False,
                qfaithful_pose_lookup=qfaithful_training_poses,
            )
            if _halfframe_eval_active:
                scorer_val_half, eval_pose_half, eval_seg_half = evaluate_fp4(
                    model, ema, all_masks, gt_frames,
                    all_pair_starts, posenet, segnet, device,
                    fp4_codebook=args.fp4_codebook,
                    fp4_robust_scale=args.fp4_robust_scale,
                    sim_zoom_warp=sim_zoom_warp,
                    use_zoom_flow=getattr(args, "use_zoom_flow", False),
                    half_frame_mode=True,
                    qfaithful_pose_lookup=qfaithful_training_poses,
                )
                # Best-gate on HALF-FRAME score (deployment metric).
                scorer_val = scorer_val_half
                eval_pose = eval_pose_half
                eval_seg = eval_seg_half
                print(
                    f"[eval] FULL-FRAME: scorer={scorer_val_full:.4f} "
                    f"pose={eval_pose_full:.6f} seg={eval_seg_full:.6f} | "
                    f"HALF-FRAME (gates best): scorer={scorer_val_half:.4f} "
                    f"pose={eval_pose_half:.6f} seg={eval_seg_half:.6f}"
                )
            else:
                # Legacy / baseline profiles — full-frame is the only path.
                scorer_val = scorer_val_full
                eval_pose = eval_pose_full
                eval_seg = eval_seg_full
        else:
            # scorer_val is stale (carried from last eval) on non-eval epochs
            scorer_val = best_scorer

        # Baseline watermark + regression alarm (Feature 4)
        if is_eval_epoch and baseline_pose is None:
            baseline_pose = eval_pose
            baseline_seg = eval_seg
            print(f"[eval] Baseline watermark: pose={baseline_pose:.6f}, seg={baseline_seg:.6f}")

        if is_eval_epoch and baseline_pose is not None and baseline_pose > 0:
            pose_ratio = eval_pose / baseline_pose
            if pose_ratio > 3.0:
                print(f"  WARNING: PoseNet {pose_ratio:.1f}x regression! "
                      f"pose={eval_pose:.6f} vs baseline {baseline_pose:.6f}")
            if pose_ratio > 5.0:
                print(f"  CRITICAL: PoseNet {pose_ratio:.0f}x baseline -- checkpoint NOT saved!")
                scorer_val = float("inf")

        # Log and save best
        marker = ""
        if scorer_val < best_scorer:
            best_scorer = scorer_val
            best_epoch = epoch
            marker = " *BEST*"
            # Codex R-Lane-D-Issue2: announce the gating metric so operators
            # know which distribution the saved checkpoint is best on.
            if _halfframe_eval_active:
                print(
                    f"[best] checkpoint @ ep{epoch} based on HALF-FRAME score: "
                    f"{best_scorer:.4f} (full-frame for comparison: "
                    f"{scorer_val_full:.4f})"
                )
            else:
                print(
                    f"[best] checkpoint @ ep{epoch} based on FULL-FRAME score: "
                    f"{best_scorer:.4f}"
                )

            # Save FP4 checkpoint from EMA weights. R-FP4-fix: pass codebook +
            # robust_scale to match what QAT trained against — mismatched
            # quantization at save time silently corrupts the deployed model.
            save_state = _strip_t2_training_only_state(ema.state_dict())
            fp4_path = out_dir / f"renderer_{args.tag}_best_fp4.pt"
            from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
            _save_codebook = (RESIDUAL_CODEBOOK if args.fp4_codebook == "residual"
                              else DEFAULT_CODEBOOK).clone()
            fp4_packed = quantize_fp4(
                save_state,
                codebook=_save_codebook,
                robust_scale=args.fp4_robust_scale,
            )
            # 2026-04-26 council fix: include EVERY arch field so consumers
            # (pipeline.py, auth_eval_renderer.py) can rebuild the exact same
            # AsymmetricPairGenerator vs PairGenerator class with matching
            # state_dict shapes. Without these, DEN crashed on load (1.2h burn).
            _arch_meta = {
                "fp4_codebook": args.fp4_codebook,
                "fp4_robust_scale": args.fp4_robust_scale,
                "fp4_stochastic": args.fp4_stochastic,
                "epoch": epoch,
                "scorer": best_scorer,
                "pose": eval_pose,
                "seg": eval_seg,
                # Architecture (the missing fields that caused the 2026-04-26
                # arch drift bug — DEN profile said use_zoom_flow=True but
                # train_renderer didn't pass it OR record it):
                "base_ch": args.base_ch,
                "mid_ch": args.mid_ch,
                "embed_dim": args.embed_dim,
                "motion_hidden": args.motion_hidden,
                "depth": args.depth,
                "pose_dim": getattr(args, "pose_dim", 0) or 0,
                "use_zoom_flow": args.use_zoom_flow,
                "use_dsconv": args.use_dsconv,
                "use_ghost": args.use_ghost,
                "use_dilation": args.use_dilation,
                "padding_mode": args.padding_mode,
                "blend_mode": args.blend_mode,
                "noise_mode": args.noise_mode,
                "motion_type": args.motion_type,
                # Variant + extras:
                "latent_ch": args.latent_ch,
                "latent_shapes": args.latent_shapes,
                "residual_hidden": args.residual_hidden,
                "residual_layers": args.residual_layers,
                "residual_scale": args.residual_scale,
                "variant": args.variant,
                "seed": args.seed,
                "deterministic": args.deterministic,
                # Class name so consumers know which to instantiate.
                "model_class": "AsymmetricPairGenerator" if args.use_zoom_flow else "PairGenerator",
                "profile": getattr(args, "profile", None),
                "use_t2_xpred_loss": bool(getattr(args, "use_t2_xpred_loss", False)),
                "t2_xpred_sigma": float(getattr(args, "t2_xpred_sigma", 1.0)),
                "t2_xpred_weighting": getattr(args, "t2_xpred_weighting", "v"),
                "use_t2_mask": bool(getattr(args, "use_t2_mask", False)),
                "t2_mask_p": float(getattr(args, "t2_mask_p", 0.5)),
                "t2_mask_ratio": float(getattr(args, "t2_mask_ratio", 0.15)),
                "t2_mask_apply_fraction": float(
                    getattr(args, "t2_mask_apply_fraction", 0.4)
                ),
                "no_mask_encoder": bool(getattr(args, "no_mask_encoder", False)),
                "informational_only": bool(getattr(args, "no_mask_encoder", False)),
                "distillation_policy": getattr(args, "distillation_policy_provenance", None),
                "distillation_policy_sha256": getattr(args, "distillation_policy_sha256", None),
            }
            _qfaithful_pose_contract = getattr(args, "qfaithful_training_pose_contract", None)
            if _qfaithful_pose_contract is not None:
                _arch_meta["qfaithful_training_pose_contract"] = _qfaithful_pose_contract
                _arch_meta["training_pose_contract"] = _qfaithful_pose_contract
            fp4_packed["__meta__"] = _arch_meta
            if _qfaithful_pose_contract is not None:
                fp4_packed["qfaithful_training_pose_contract"] = _qfaithful_pose_contract
                fp4_packed["training_pose_contract"] = _qfaithful_pose_contract
            fp4_tmp = fp4_path.with_suffix(".pt.tmp")
            torch.save(fp4_packed, fp4_tmp)
            fp4_tmp.rename(fp4_path)
            fp4_size = fp4_path.stat().st_size
            param_count = sum(p.numel() for p in model.parameters())
            print(f"[fp4] Saved {param_count:,} params to {fp4_path} ({fp4_size:,} bytes)")

            # Lane D 2026-04-27: when use_zoom_flow=True the inflate side
            # NEEDS the per-pair zoom scalars to compute ego_flow consistently
            # with how the renderer was trained. Without zoom_scalars in the
            # archive, inflate falls back to identity zoom (scalars=0) which
            # is a train/inflate mismatch that re-introduces the very bug
            # Lane D was built to fix. Save once next to the FP4 checkpoint
            # so the bootstrap script can copy it into the archive.
            if sim_zoom_warp is not None and args.use_zoom_flow:
                zoom_scalars_path = out_dir / "zoom_scalars.pt"
                # Save the full state_dict so the inflate-side
                # RadialZoomWarp.load_state_dict() works without surgery.
                torch.save(sim_zoom_warp.state_dict(), zoom_scalars_path)
                zs_size = zoom_scalars_path.stat().st_size
                print(f"[zoom] Saved zoom_scalars.pt to {zoom_scalars_path} "
                      f"({zs_size:,} bytes; n_pairs={sim_zoom_warp.n_pairs})")

            # Also save EMA fp32 for resuming
            fp32_path = out_dir / f"renderer_{args.tag}_best_fp32.pt"
            fp32_tmp = fp32_path.with_suffix(".pt.tmp")
            # R-fp4-export-fix: save FP4 metadata in the fp32 checkpoint too,
            # so pipeline.step_export reads it and uses the matching codebook
            # at re-export time. Without this, training writes residual codebook
            # in the fp4 file, but step_export's later re-export uses default.
            # 2026-04-26 council fix: fp32 __meta__ MUST include the same
            # arch dict as fp4 so consumers can rebuild the right model class.
            # Reuse the dict we just built above.
            fp32_payload = {
                "model_state_dict": save_state,
                "__meta__": dict(_arch_meta),
            }
            if _qfaithful_pose_contract is not None:
                fp32_payload["qfaithful_training_pose_contract"] = _qfaithful_pose_contract
                fp32_payload["training_pose_contract"] = _qfaithful_pose_contract
            torch.save(fp32_payload, fp32_tmp)
            fp32_tmp.rename(fp32_path)

            # Save metadata
            meta_path = out_dir / f"renderer_{args.tag}_best_meta.json"
            meta_path.write_text(json.dumps({
                "epoch": epoch,
                "scorer": best_scorer,
                "pose": eval_pose,
                "seg": eval_seg,
                "fp4_path": str(fp4_path),
                "fp4_size": fp4_size,
                "distillation_policy": getattr(args, "distillation_policy_provenance", None),
                "distillation_policy_sha256": getattr(args, "distillation_policy_sha256", None),
                "qfaithful_training_pose_contract": _qfaithful_pose_contract,
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "exact_cuda_auth_eval_required": True,
                "args": {
                    "base_ch": args.base_ch,
                    "mid_ch": args.mid_ch,
                    "embed_dim": args.embed_dim,
                    "motion_hidden": args.motion_hidden,
                    "depth": args.depth,
                    "pretrain_epochs": args.pretrain_epochs,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "profile": args.profile,
                    "tag": args.tag,
                    "variant": args.variant,
                    "latent_ch": args.latent_ch,
                    "latent_shapes": args.latent_shapes,
                    "residual_hidden": args.residual_hidden,
                    "residual_layers": args.residual_layers,
                    "residual_scale": args.residual_scale,
                    "seed": args.seed,
                    "deterministic": args.deterministic,
                    "no_mask_encoder": bool(getattr(args, "no_mask_encoder", False)),
                    "informational_only": bool(getattr(args, "no_mask_encoder", False)),
                    "stacking_candidate": not bool(getattr(args, "no_mask_encoder", False)),
                },
            }, indent=2))

        # Epoch log with timing (Feature 7)
        eta_hours = epoch_sec * (args.epochs - epoch - 1) / 3600
        phase_tag = f"P{phase_idx}" if use_5phase else ("P1" if in_pretrain else "P2")
        # Lane D-V3 (2026-04-27) Phase-0 instrumentation: per-epoch
        # half-frame branch trigger rate + warp-diff stats. Empty string when
        # the half-frame branch never fires (sim_zoom_warp is None or
        # current_mask_half_sim_prob == 0), so log lines for full-frame
        # profiles are byte-identical to pre-V3.
        if halfframe_branch_fires > 0:
            _hf_rate = halfframe_branch_fires / max(len(perm), 1)
            _hf_diff_mean = (halfframe_warp_diff_sum
                             / max(halfframe_warp_diff_count, 1))
            hf_metric = (
                f" hf_fires={halfframe_branch_fires}/{len(perm)} "
                f"({_hf_rate:.2f}) hf_warp_diff={_hf_diff_mean:.4f} "
                f"hf_target_prob={current_mask_half_sim_prob:.3f}"
            )
        else:
            hf_metric = ""
        if is_eval_epoch:
            print(f"[ep {epoch:4d}/{args.epochs} {phase_tag}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} "
                  f"fp4_scorer={scorer_val:.4f} best={best_scorer:.4f} "
                  f"lr={lr:.6f} {epoch_sec:.1f}s/ep ETA={eta_hours:.1f}h{hf_metric}{marker}")
        elif epoch % 10 == 0:
            print(f"[ep {epoch:4d}/{args.epochs} {phase_tag}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} lr={lr:.6f} "
                  f"{epoch_sec:.1f}s/ep ETA={eta_hours:.1f}h{hf_metric}")

        # JSONL telemetry (Feature 1)
        if is_eval_epoch:
            telemetry = {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "pose": round(avg_pose, 8),
                "seg": round(avg_seg, 8),
                "fp4_scorer": round(scorer_val, 6) if scorer_val != float("inf") else None,
                "best": round(best_scorer, 6) if best_scorer != float("inf") else None,
                "lr": round(lr, 8),
                "phase": phase,
                "tag": args.tag,
                "variant": args.variant,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "epoch_sec": round(epoch_sec, 2),
                "eval_pose": round(eval_pose, 8),
                "eval_seg": round(eval_seg, 8),
                "best_epoch": best_epoch,
                "timestamp": datetime.now(UTC).isoformat(),
                "distillation_policy": getattr(args, "distillation_policy_provenance", None),
                "distillation_policy_sha256": getattr(args, "distillation_policy_sha256", None),
                "score_claim_eligible": False,
                "promotion_eligible": False,
                "exact_cuda_auth_eval_required": True,
                # Codex R-Lane-D-Issue2: surface BOTH eval modes when in
                # half-frame mode so post-hoc analysis can spot proxy/auth
                # divergence between full-frame and half-frame distributions.
                "eval_full_frame": (
                    {
                        "scorer": round(scorer_val_full, 6),
                        "pose": round(eval_pose_full, 8),
                        "seg": round(eval_seg_full, 8),
                    }
                    if scorer_val_full is not None else None
                ),
                "eval_half_frame": (
                    {
                        "scorer": round(scorer_val_half, 6),
                        "pose": round(eval_pose_half, 8),
                        "seg": round(eval_seg_half, 8),
                    }
                    if scorer_val_half is not None else None
                ),
                "best_gate": "half_frame" if _halfframe_eval_active else "full_frame",
                # Lane D-V3 (2026-04-27) Phase-0 mechanism telemetry: pin
                # the per-epoch warp prob, trigger count, and warp-diff
                # stat in JSONL so post-hoc analysis can verify the half-
                # frame branch is actually firing AND producing meaningful
                # mask perturbations (not identity).
                "halfframe_target_prob": round(
                    float(current_mask_half_sim_prob), 6,
                ),
                "halfframe_branch_fires": halfframe_branch_fires,
                "halfframe_step_count": len(perm),
                "halfframe_warp_diff_mean": (
                    round(
                        halfframe_warp_diff_sum
                        / max(halfframe_warp_diff_count, 1),
                        6,
                    )
                    if halfframe_warp_diff_count > 0 else None
                ),
            }
            write_telemetry(out_dir / f"{args.tag}_telemetry.jsonl", telemetry)

        # Save training state every 50 epochs for crash recovery (Feature 6)
        if epoch % 50 == 0 and epoch > 0:
            save_training_state()

        # Wall-clock timeout (Feature 3)
        if args.wall_clock_timeout > 0:
            elapsed = time.monotonic() - start_wall_time
            if elapsed >= args.wall_clock_timeout:
                print(f"\n[train] WALL-CLOCK TIMEOUT at epoch {epoch} "
                      f"(elapsed {elapsed/3600:.1f}h, "
                      f"limit {args.wall_clock_timeout/3600:.1f}h)")
                save_training_state()
                print(f"[train] Timeout exit. Best: {best_scorer:.4f} at epoch {best_epoch}")
                break

    # Final save
    save_training_state()
    print(f"\n[train] Complete. Best FP4 scorer: {best_scorer:.4f} at epoch {best_epoch}")
    print(f"[train] Saved to: {out_dir}/renderer_{args.tag}_best_fp4.pt")

    # Clean up QAT hooks
    if qat_wrapper is not None:
        qat_wrapper.remove_hooks()

    # Auth eval on best (CLAUDE.md non-negotiable: "Auth eval EVERYWHERE").
    #
    # Council R3 caught the previous wiring as fake: it invented an
    # `--auth-eval-masks` flag for auth_eval_renderer.py that doesn't exist
    # (auth_eval_renderer recomputes SegNet masks from GT internally and
    # doesn't take a masks file). The previous wiring also didn't pass
    # `--archive-size-bytes`, so rate was computed from the renderer .pt
    # alone (~290KB) — systematically optimistic vs the real ~700KB
    # triple-joint archive. THIRD bug: the previous version skipped the
    # whole eval if `--auth-eval-masks` wasn't passed, meaning every
    # default chain silently no-op'd.
    #
    # Real wiring (mirrors experiments/pipeline.py:step_eval which works):
    #   1. Build a real submission archive (renderer + masks + poses) via
    #      tac.submission_archive.build_submission_archive
    #   2. Get the actual archive byte size from the built file
    #   3. Call auth_eval_renderer with --checkpoint + --archive-size-bytes
    #      + --poses (only flags that exist per its argparse)
    #
    # Falls back gracefully when no masks are provided (still produces a
    # renderer-only auth score, which is at least HONEST about the rate
    # bias rather than silently skipping).
    if getattr(args, "auth_eval_on_best", False):
        import subprocess
        best_fp4 = out_dir / f"renderer_{args.tag}_best_fp4.pt"
        if not best_fp4.exists():
            raise RuntimeError(
                f"[auth-eval] No best checkpoint at {best_fp4}. Cannot run "
                f"--auth-eval-on-best. If training was killed before any "
                f"*BEST* save, pass --no-auth-eval-on-best to bypass."
            )
        # FAIL LOUD: --auth-eval-on-best REQUIRES masks + poses. The previous
        # WARN-and-fall-back silently produced renderer-only-rate scores
        # that were ~2x optimistic vs the real triple-joint archive — exactly
        # the systematic-optimism bug the user has flagged repeatedly.
        # If the caller wants to skip, they pass --no-auth-eval-on-best.
        # If they want to run, they MUST supply --auth-eval-masks AND
        # --auth-eval-poses so the archive is built honestly.
        if not args.auth_eval_masks or not args.auth_eval_poses:
            raise RuntimeError(
                f"[auth-eval] --auth-eval-on-best requires BOTH "
                f"--auth-eval-masks and --auth-eval-poses to build a real "
                f"submission archive. Without them, the rate term would be "
                f"computed from renderer-only bytes (~290KB) instead of the "
                f"real ~700KB triple-joint archive — systematically optimistic "
                f"by ~2x. Got: "
                f"masks={args.auth_eval_masks!r}, poses={args.auth_eval_poses!r}. "
                f"To skip the eval entirely, pass --no-auth-eval-on-best."
            )

        # Codex R-Lane-D-followup 2026-04-27: best_fp4 (.pt) is a torch.save
        # dict of FP4-packed scales/indices, NOT the FP4A magic-byte .bin
        # format that auth_eval_renderer.py and build_submission_archive
        # require. Without this conversion the entire --auth-eval-on-best
        # block was a silent no-op (or hard-failed at archive validation).
        # Mirror the bootstrap pattern: load the FP32 .pt sibling + arch_meta,
        # build a fresh model, export real FP4A .bin via the canonical path.
        fp32_path = out_dir / f"renderer_{args.tag}_best_fp32.pt"
        if not fp32_path.exists():
            raise RuntimeError(
                f"[auth-eval] FP32 sidecar missing at {fp32_path}; cannot "
                f"export FP4A .bin (the .pt at {best_fp4} is FP4-packed but "
                f"lacks magic bytes). The save site at line ~1877 should "
                f"have written this file alongside the .pt — investigate."
            )
        fp32_payload = torch.load(str(fp32_path), map_location="cpu", weights_only=False)
        if not isinstance(fp32_payload, dict) or "model_state_dict" not in fp32_payload \
                or "__meta__" not in fp32_payload:
            raise RuntimeError(
                f"[auth-eval] FP32 sidecar at {fp32_path} is malformed "
                f"(expected dict with 'model_state_dict' and '__meta__' keys; "
                f"got keys={list(fp32_payload.keys()) if isinstance(fp32_payload, dict) else type(fp32_payload).__name__})"
            )
        arch = fp32_payload["__meta__"]
        state = fp32_payload["model_state_dict"]
        # Codex R5-2 Finding #1 (2026-04-27): pre-fix this hardcoded
        # `variant in ('default', None)` check rejected EVERY profile that
        # routes through build_renderer (dilated, psd, mask_renderer, plus
        # ~20 others — see _VARIANTS_BUILD_RENDERER_FP4A_OK), then
        # RuntimeError'd at the END of training. Result: 5h Lane D runs
        # produced no authoritative score, exactly the failure mode the
        # default-True --auth-eval-on-best was added to prevent. The
        # validation now lives early in train() (search for "Finding #1"),
        # but we keep this assertion as a defence-in-depth guard against
        # arch_meta drift between the parse-args check and the save site.
        _ckpt_variant = arch.get("variant", "default")
        if not _variant_supports_fp4a_export(_ckpt_variant):
            raise RuntimeError(
                f"[auth-eval] FP4A export does not support variant="
                f"{_ckpt_variant!r} (it lives in _VARIANTS_NON_BUILD_RENDERER "
                f"and uses a non-AsymmetricPairGenerator state_dict layout). "
                f"This should have been caught by the parse-args validation "
                f"earlier — investigate how the meta drifted vs. args.variant. "
                f"Workaround: pass --no-auth-eval-on-best and run a separate "
                f"eval suited to the variant."
            )
        from tac.renderer import build_renderer
        from tac.renderer_export import export_asymmetric_checkpoint_fp4
        bin_model = build_renderer(
            num_classes=5,
            embed_dim=arch["embed_dim"],
            base_ch=arch["base_ch"],
            mid_ch=arch["mid_ch"],
            motion_hidden=arch["motion_hidden"],
            depth=arch["depth"],
            blend_mode=arch.get("blend_mode", "scalar"),
            noise_mode=arch.get("noise_mode", "deterministic"),
            motion_type=arch.get("motion_type", "learned_cnn"),
            use_zoom_flow=arch["use_zoom_flow"],
            use_dsconv=arch["use_dsconv"],
            use_ghost=arch.get("use_ghost", False),
            padding_mode=arch["padding_mode"],
            use_dilation=arch["use_dilation"],
            pose_dim=arch.get("pose_dim", 0) or 0,
        )
        missing, unexpected = bin_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"[auth-eval] state_dict shape mismatch loading {fp32_path}: "
                f"missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}. "
                f"This is the SHIRAZ-class arch-drift bug — the saved arch_meta "
                f"does not match the model the .pt was trained with."
            )
        bin_path = out_dir / f"renderer_{args.tag}_best.bin"
        nbytes = export_asymmetric_checkpoint_fp4(
            bin_model, bin_path,
            codebook_name=arch.get("fp4_codebook", "default"),
            robust_scale=arch.get("fp4_robust_scale", False),
        )
        print(f"[auth-eval] Exported FP4A: {bin_path} ({nbytes:,} bytes; "
              f"codebook={arch.get('fp4_codebook')}, robust={arch.get('fp4_robust_scale')})")
        best_fp4 = bin_path  # downstream archive + auth eval consume the .bin

        # Build a real archive (mirrors experiments/pipeline.py:step_eval).
        from tac.submission_archive import build_submission_archive
        archive_path = out_dir / "auth_eval_on_best_archive.zip"
        poses_arg = args.auth_eval_poses
        poses_kwargs = (
            {"optimized_poses_bin": poses_arg}
            if str(poses_arg).endswith(".bin")
            else {"optimized_poses_pt": poses_arg}
        )
        build_submission_archive(
            output_path=archive_path,
            renderer_bin=str(best_fp4),
            masks_mkv=args.auth_eval_masks,
            validate=True,
            **poses_kwargs,
        )
        archive_bytes = archive_path.stat().st_size
        print(f"[auth-eval] Built archive: {archive_path} ({archive_bytes:,} bytes)")

        # Step 2: call auth_eval_renderer with REAL flags only. Path
        # resolution uses the established `_repo` constant (Council R3-4 fix).
        auth_eval_script = _repo / "experiments" / "auth_eval_renderer.py"
        if not auth_eval_script.exists():
            raise RuntimeError(f"[auth-eval] cannot find {auth_eval_script}")

        cmd = [
            sys.executable, "-u", str(auth_eval_script),
            "--checkpoint", str(best_fp4),
            "--upstream-dir", args.auth_eval_upstream_dir,
            "--device", "cuda",
            "--archive-size-bytes", str(archive_bytes),
            "--output-dir", str(out_dir),
            "--poses", str(args.auth_eval_poses),
        ]
        print("\n[auth-eval] Launching CUDA auth eval against best checkpoint...")
        print(f"[auth-eval] cmd: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            print(proc.stdout[-2000:])
            if proc.returncode != 0:
                print(f"[auth-eval] FAILED with returncode={proc.returncode}",
                      file=sys.stderr)
                print(proc.stderr[-2000:], file=sys.stderr)
                raise RuntimeError(
                    f"auth_eval_renderer exited {proc.returncode}; "
                    f"see {out_dir} for output and stderr"
                )
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    print(f"\n[auth-eval] {line}")
                    break
            else:
                raise RuntimeError(
                    "auth_eval_renderer completed but emitted no RESULT_JSON line — "
                    "the run produced no authoritative score (CLAUDE.md violation)"
                )
            print(f"[auth-eval] Result saved alongside checkpoint in {out_dir}")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "[auth-eval] TIMED OUT after 15min — auth_eval_renderer is hung. "
                "Investigate manually."
            ) from exc

    return best_scorer


def _list_profiles_table(describe: bool = False) -> int:
    """Print a friendly table of training profiles and exit.

    DX #7 (2026-04-26): the `--profile` help text dumps 130 names into the
    `choices=` list, which is unreadable. Operators want a one-shot table
    showing each renderer-relevant profile with its key knobs (variant,
    base_ch / mid_ch / depth, pose_dim).

    `--describe` adds a longer description column derived from the profile
    dict's `description` / `notes` field if present. Without it, only the
    name + variant tag + size summary are printed.
    """
    rows: list[tuple[str, str, str, str]] = []
    for name in sorted(PROFILES.keys()):
        prof = PROFILES[name]
        if not isinstance(prof, dict):
            continue
        # Heuristic: a "renderer profile" sets at least one of these keys.
        is_renderer = any(k in prof for k in (
            "base_ch", "mid_ch", "embed_dim", "motion_hidden", "depth",
            "variant", "use_zoom_flow",
        ))
        if not is_renderer:
            continue
        variant = str(prof.get("variant", "mask_renderer"))
        size = (
            f"base={prof.get('base_ch', '?')} mid={prof.get('mid_ch', '?')} "
            f"depth={prof.get('depth', '?')} pose={prof.get('pose_dim', 0)}"
        )
        notes = ""
        if describe:
            notes = str(
                prof.get("description")
                or prof.get("notes")
                or prof.get("doc")
                or ""
            )[:80]
        rows.append((name, variant, size, notes))

    name_w = max((len(r[0]) for r in rows), default=10)
    var_w = max((len(r[1]) for r in rows), default=10)
    size_w = max((len(r[2]) for r in rows), default=20)
    print(f"{'profile':<{name_w}}  {'variant':<{var_w}}  {'arch':<{size_w}}  notes")
    print("-" * (name_w + var_w + size_w + 12))
    for name, var, size, notes in rows:
        print(f"{name:<{name_w}}  {var:<{var_w}}  {size:<{size_w}}  {notes}")
    print()
    print(f"Total: {len(rows)} renderer profiles in tac.profiles.PROFILES")
    print("Run with --describe for the description column (if set in the profile).")
    return 0


def main():
    # DX #7 (2026-04-26): handle --list-profiles BEFORE argparse so the
    # operator does not have to satisfy --profile=<choice> first. We peek
    # at sys.argv directly; argparse never sees these flags.
    import sys as _sys
    if "--list-profiles" in _sys.argv:
        describe = "--describe" in _sys.argv
        raise SystemExit(_list_profiles_table(describe=describe))
    args = parse_args()
    _enforce_eval_roundtrip(args)
    return train(args)


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner."""
    if not args.eval_roundtrip:
        if _os.environ.get("TAC_ALLOW_NO_ROUNDTRIP") != "1":
            raise SystemExit(
                "FATAL: eval_roundtrip is False but TAC_ALLOW_NO_ROUNDTRIP=1 "
                "is not set. Set the env var explicitly for diagnostic ablation."
            )
        print(
            "\n" + "!" * 78 + "\n"
            "DANGER: eval_roundtrip is DISABLED via TAC_ALLOW_NO_ROUNDTRIP=1.\n"
            "  Proxy-auth gap will be 2-11x. Tag results [no-roundtrip-ablation].\n"
            + "!" * 78 + "\n",
            flush=True,
        )


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result is not None else 1)
