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

# DX-fix 2026-04-25: line-buffer stdout/stderr so progress logs flush
# immediately when piped to log files (Python buffers ~8KB by default,
# making long-running scripts appear silent for hours per the optimize_poses
# incident on the A100 today).
import sys as _dx_sys
try:
    _dx_sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    _dx_sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


import argparse
import json
import math
import os as _os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from tac.data import decode_video, pair_from_frames, pair_start_indices  # noqa: E402
from tac.fp4_quantize import (  # noqa: E402
    QATRendererFP4,
    dequantize_fp4,
    quantize_fp4,
)
from tac.losses import (  # noqa: E402
    _hwc_to_chw,
    eval_scorer_loss,
    frequency_aware_loss,
    kl_distill_segnet_only,
    scorer_forward_pair,
    scorer_loss,
    scorer_loss_cached,
    segnet_uncertainty_weighted_loss,
)
from tac.mask_codec import extract_masks, mask_pair_from_index  # noqa: E402
from tac.profiles import PROFILES  # noqa: E402
from tac.fridrich_losses import dct_quant_loss  # noqa: E402
from tac.renderer import build_renderer, simulate_eval_roundtrip  # noqa: E402
from tac.contrib.wavelet_renderer import build_wavelet_renderer  # noqa: E402
from tac.scorer import detect_device, load_scorers  # noqa: E402
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
    p.add_argument("--grad-clip", type=float, default=1.0)
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

    # Half-frame mask simulation: replace mask_t with inverse-warp(mask_t1)
    # so the renderer learns the same mask distribution it sees at inflate
    # when the archive ships only odd-frame masks (Quantizr paradigm — the
    # main rate lever, saves ~50% of mask bytes).
    #
    # Without this, the model trains on (mask_t, mask_t1) where both are
    # ground-truth, but at inflate sees (warp(mask_t1), mask_t1) where
    # mask_t is approximated. This train/inflate mismatch costs 0.05–0.10
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

    # KL distillation: Quantizr's secret sauce per CLAUDE.md. Adds Hinton-style
    # KL divergence on SegNet soft distributions ALONGSIDE the standard scorer
    # loss (NOT replacing it — that caused PoseNet collapse historically).
    # T=2.0 with weight ~1.0 is the proven recipe.
    # default=None so the profile resolver can override (Quantizr council
    # CRITICAL 2026-04-26): with default=0.0 the profile value was DEAD,
    # KL distill never fired in any production training run.
    p.add_argument("--kl-distill-weight", type=float, default=None,
                   help="Auxiliary KL distill loss weight (default: from profile, "
                        "else 0.0 = off). Quantizr uses 1.0 with T=2.0 — Phase 2 only.")
    p.add_argument("--kl-distill-temperature", type=float, default=None,
                   help="Softmax temperature for KL distillation (default: from "
                        "profile, else 2.0).")

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
    p.add_argument("--fp4-codebook", choices=("default", "residual"),
                   default="default",
                   help="FP4 codebook: 'default' = mask2mask uniform spacing, "
                        "'residual' = denser-near-zero (4× better small-mag preservation). "
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
    p.add_argument("--use-zoom-flow", action="store_true", default=None,
                   help="GREEN profile: 4ch MotionPredictor + RadialZoomWarp")
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
    p.add_argument("--wall-clock-timeout", type=int, default=0,
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
    p.add_argument("--auth-eval-upstream-dir", type=str,
                   default=str(_upstream),
                   help="Path to upstream/ for auth eval (defaults to repo upstream).")

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
    args.use_zoom_flow = _resolve(getattr(args, "use_zoom_flow", None),
                                   "use_zoom_flow", False)
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
            pair_idx_t = torch.tensor([start // 2], device=device, dtype=torch.long)

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

            if ego_flow is not None:
                rendered_pair = model(mask_t, mask_t1, ego_flow=ego_flow)
            else:
                rendered_pair = model(mask_t, mask_t1)  # (1, 2, H, W, 3)
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
    "use_zoom_flow", "use_dsconv", "use_dilation", "padding_mode",
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
    except Exception as exc:  # noqa: BLE001
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
        has_film = any("film_" in k for k in sd.keys())
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
    if args.mask_noise_mkv is not None:
        from tac.mask_codec import decode_masks
        print(f"[masks] Decoding noisy masks from {args.mask_noise_mkv} ...")
        noisy_masks = decode_masks(args.mask_noise_mkv).cpu()
        if noisy_masks.shape[0] != all_masks.shape[0]:
            # Truncate or pad — must match GT mask count
            n_match = min(noisy_masks.shape[0], all_masks.shape[0])
            if noisy_masks.shape[0] != all_masks.shape[0]:
                print(f"[masks] WARNING: noisy masks ({noisy_masks.shape[0]}) != "
                      f"GT masks ({all_masks.shape[0]}); truncating both to {n_match}")
                noisy_masks = noisy_masks[:n_match]
                all_masks = all_masks[:n_match]
        # Sanity-check resolution match
        if noisy_masks.shape[1:] != all_masks.shape[1:]:
            raise ValueError(
                f"Noisy mask resolution {noisy_masks.shape[1:]} != "
                f"GT mask resolution {all_masks.shape[1:]}. "
                f"Re-encode noisy masks at the renderer's input resolution."
            )
        disagree = float((noisy_masks != all_masks).float().mean().item())
        print(f"[masks] Noisy-mask augmentation enabled "
              f"(prob={args.mask_noise_prob}, disagreement vs GT={disagree:.4f})")

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
    sim_warp_cache: dict[int, torch.Tensor] = {}
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
            padding_mode=args.padding_mode,
            use_dilation=args.use_dilation,
            pose_dim=getattr(args, "pose_dim", 0) or 0,
        )
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
    train_size = max(1, n_total // args.subsample)
    print(f"[train] {args.epochs} epochs (pretrain={args.pretrain_epochs}), "
          f"{train_size}/{n_total} pairs/epoch, "
          f"accum={args.accum_steps}, lr={args.lr}, depth={args.depth}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "use_dilation": bool(args.use_dilation),
            "padding_mode": args.padding_mode,
            "variant": args.variant,
            "profile": getattr(args, "profile", None),
        }
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
        }, tmp_path)
        tmp_path.rename(path)  # atomic on POSIX

    # Resume from checkpoint if specified
    if args.resume_from and Path(args.resume_from).exists():
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

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        epoch_start = time.monotonic()
        model.train()
        in_pretrain = has_pretrain and epoch < args.pretrain_epochs

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
                mask_t, mask_t1 = mask_pair_from_index(noisy_masks, start)
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

            # Half-frame mask simulation (Quantizr paradigm, Lane D2). With prob
            # --mask-half-sim-prob, replace mask_t with inverse_warp(mask_t1)
            # so this pair simulates inflate-time conditions where only the
            # odd-frame mask was archived. The GT pair (gt_pair below) is
            # unchanged — we still measure renderer output against the true
            # frames; only the mask INPUT distribution shifts.
            if (sim_zoom_warp is not None
                    and random.random() < args.mask_half_sim_prob):
                mask_t = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx_t)

            gt_pair = pair_from_frames(gt_frames, start).to(device)

            # Horizontal flip augmentation (50% probability)
            flipped_h = False
            if random.random() < 0.5:
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

            # Forward: render pair from masks. Only AsymmetricPairGenerator
            # accepts ego_flow (see renderer.py:1035-1100); the legacy
            # PairGenerator.forward() takes only (mask_t, mask_t1). Branch on
            # presence of ego_flow rather than model class so this stays
            # variant-agnostic.
            if ego_flow is not None:
                rendered_pair = model(mask_t, mask_t1, ego_flow=ego_flow)
            else:
                rendered_pair = model(mask_t, mask_t1)  # (1, 2, H, W, 3)

            if in_pretrain:
                # Phase 1: L1 + edge loss only -- no scorer, much faster
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
                if _cached_gt is not None:
                    _gt_pose_6 = _cached_gt["pose_6"].to(device)
                    _gt_seg_soft = _cached_gt["seg_soft"].to(device)
                    loss, pd, sd = scorer_loss_cached(
                        rendered_pair, _gt_pose_6, _gt_seg_soft, posenet, segnet,
                    )
                    del _gt_pose_6, _gt_seg_soft
                else:
                    loss, pd, sd = scorer_loss(rendered_pair, gt_pair, posenet, segnet)

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
                    # (H+2*pad-k+1 = H+1), which then breaks the diff×inv_var
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
                # loss. Penalises low-frequency residual energy ~6× more than
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
                if args.kl_distill_weight > 0:
                    kl_loss, _kl_seg = kl_distill_segnet_only(  # KL_RAW_PAIRS_OK:rendered_pair was reassigned to roundtripped output at L1642
                        rendered_pair, gt_pair, segnet,
                        temperature=args.kl_distill_temperature,
                    )
                    loss = loss + args.kl_distill_weight * kl_loss

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

        # Per-epoch timing (Feature 7)
        epoch_sec = time.monotonic() - epoch_start

        # Determine current phase label. 5-phase mode uses Quantizr's named
        # phases; legacy mode keeps the historical "pretrain"/"scorer" labels.
        if use_5phase:
            phase = f"phase{phase_idx}_{PHASE_NAMES[phase_idx]}"
        else:
            phase = "pretrain" if in_pretrain else "scorer"

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
            save_state = ema.state_dict()
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
            }
            fp4_packed["__meta__"] = _arch_meta
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
                },
            }, indent=2))

        # Epoch log with timing (Feature 7)
        eta_hours = epoch_sec * (args.epochs - epoch - 1) / 3600
        phase_tag = f"P{phase_idx}" if use_5phase else ("P1" if in_pretrain else "P2")
        if is_eval_epoch:
            print(f"[ep {epoch:4d}/{args.epochs} {phase_tag}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} "
                  f"fp4_scorer={scorer_val:.4f} best={best_scorer:.4f} "
                  f"lr={lr:.6f} {epoch_sec:.1f}s/ep ETA={eta_hours:.1f}h{marker}")
        elif epoch % 10 == 0:
            print(f"[ep {epoch:4d}/{args.epochs} {phase_tag}] loss={avg_loss:.4f} "
                  f"pose={avg_pose:.6f} seg={avg_seg:.6f} lr={lr:.6f} "
                  f"{epoch_sec:.1f}s/ep ETA={eta_hours:.1f}h")

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        print(f"\n[auth-eval] Launching CUDA auth eval against best checkpoint...")
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
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "[auth-eval] TIMED OUT after 15min — auth_eval_renderer is hung. "
                "Investigate manually."
            )

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
