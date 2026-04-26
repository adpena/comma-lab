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
)
from tac.mask_codec import extract_masks, mask_pair_from_index  # noqa: E402
from tac.profiles import PROFILES  # noqa: E402
from tac.renderer import build_renderer, simulate_eval_roundtrip  # noqa: E402
from tac.contrib.wavelet_renderer import build_wavelet_renderer  # noqa: E402
from tac.scorer import detect_device, load_scorers  # noqa: E402
from tac.training import EMA  # noqa: E402
from tac.utils import setup_signal_handlers, write_telemetry  # noqa: E402


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
                   help="Phase 1 epochs: L1+edge loss, no scorer (default: from profile or 0)")
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
    p.add_argument("--mask-half-sim-prob", type=float, default=0.0,
                   help="Probability of replacing mask_t with inverse-warp("
                        "mask_t1) during training (default 0.0 = off). Set to "
                        "0.5 when shipping a half-frame archive (Quantizr "
                        "paradigm) so train/inflate distributions match.")

    # KL distillation: Quantizr's secret sauce per CLAUDE.md. Adds Hinton-style
    # KL divergence on SegNet soft distributions ALONGSIDE the standard scorer
    # loss (NOT replacing it — that caused PoseNet collapse historically).
    # T=2.0 with weight ~1.0 is the proven recipe.
    p.add_argument("--kl-distill-weight", type=float, default=0.0,
                   help="Auxiliary KL distill loss weight (default 0.0 = off). "
                        "Quantizr uses this with T=2.0 to crush SegNet to 0.001. "
                        "Recommended: 1.0 during Phase 2.")
    p.add_argument("--kl-distill-temperature", type=float, default=2.0,
                   help="Softmax temperature for KL distillation (default 2.0).")

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
    p.add_argument("--use-texture-loss", action="store_true", default=None,
                   help="Fridrich UNIWARD: hide errors in textured regions")
    p.add_argument("--texture-loss-weight", type=float, default=None)
    p.add_argument("--use-linf-penalty", action="store_true", default=None,
                   help="Fridrich square root law: spread errors")
    p.add_argument("--linf-weight", type=float, default=None)
    p.add_argument("--use-markov-loss", action="store_true", default=None,
                   help="Fridrich HUGO: preserve local gradient statistics")
    p.add_argument("--markov-weight", type=float, default=None)
    p.add_argument("--use-per-class-weights", action="store_true", default=None)
    p.add_argument("--use-swa", action="store_true", default=None)
    p.add_argument("--even-frame-skip-seg", action="store_true", default=False,
                   help="Trick 3: skip SegNet loss when frame_t1 is even-indexed "
                   "(SegNet only evaluates odd frames in the scorer)")
    p.add_argument("--frequency-loss-weight", type=float, default=None,
                   help="Trick 2: wavelet frequency-domain loss weight (0=disabled)")
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resize chain in scorer loss (default: on)")
    p.add_argument("--no-eval-roundtrip", dest="eval_roundtrip", action="store_false",
                   help="Disable eval roundtrip simulation")

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

    # Output
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--output-dir", type=str,
                   default=str(_repo / "experiments" / "postfilter_weights"))
    p.add_argument("--device", type=str, default=None)

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
) -> tuple[float, float, float]:
    """Evaluate the EMA model after FP4 round-trip quantization.

    R-FP4-fix: codebook + robust_scale must match the QAT wrapper used during
    training. Mismatched eval-time quantization gives a misleading FP4 scorer
    (that's how the trend report's 93.44 plateau hid the real float→FP4 gap).

    Returns: (scorer, avg_pose, avg_seg)
    """
    from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
    codebook = (RESIDUAL_CODEBOOK if fp4_codebook == "residual"
                else DEFAULT_CODEBOOK).clone()

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

            gt_pair = pair_from_frames(gt_frames, start).to(device)

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

    # Half-frame mask simulation precompute. When --mask-half-sim-prob > 0,
    # we periodically replace mask_t with inverse_warp(mask_t1, zoom[k]) so
    # the renderer learns the same distribution it sees at inflate. Compute
    # the analytical zoom scalars once from the masks themselves (Hotz: lane
    # markings encode forward speed → radial zoom from FoE), no GT poses
    # needed. RadialZoomWarp + warp_inverse_masks come from tac.radial_zoom.
    sim_zoom_warp = None
    sim_warp_cache: dict[int, torch.Tensor] = {}
    if getattr(args, "mask_half_sim_prob", 0.0) > 0:
        from tac.lane_mark_speed import zoom_from_masks
        from tac.radial_zoom import RadialZoomWarp
        n_sim_pairs = all_masks.shape[0] // 2
        sim_zooms = zoom_from_masks(all_masks)  # (n_pairs,) float32
        sim_zoom_warp = RadialZoomWarp(n_pairs=n_sim_pairs).to(device)
        with torch.no_grad():
            sim_zoom_warp.zoom_scalars.data.copy_(
                sim_zooms.clamp(-sim_zoom_warp.max_zoom_log, sim_zoom_warp.max_zoom_log)
            )
        print(f"[masks] Half-frame simulation enabled "
              f"(prob={args.mask_half_sim_prob}, "
              f"zoom mean={sim_zooms.mean():.4f}, std={sim_zooms.std():.4f}, "
              f"n_pairs={n_sim_pairs})")

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
            blend_mode=getattr(args, "blend_mode", "scalar"),
            noise_mode=getattr(args, "noise_mode", "deterministic"),
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
            blend_mode=getattr(args, "blend_mode", "scalar"),
            noise_mode=getattr(args, "noise_mode", "deterministic"),
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
            beta_start=getattr(args, "beta_start", 1e-4),
            beta_end=getattr(args, "beta_end", 0.02),
        )
    else:
        model = build_renderer(
            num_classes=5,
            embed_dim=args.embed_dim,
            base_ch=args.base_ch,
            mid_ch=args.mid_ch,
            motion_hidden=args.motion_hidden,
            depth=args.depth,
            blend_mode=getattr(args, "blend_mode", "scalar"),
            noise_mode=getattr(args, "noise_mode", "deterministic"),
            motion_type=getattr(args, "motion_type", "learned_cnn"),
        )
    model = model.to(device)
    # channels_last memory format: 10-30% speedup for conv2d on CUDA
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Wrap with FP4 QAT if enabled. R-FP4-fix: pass codebook + robustness flags
    # so the trend report's float→FP4 gap (93.44 plateau while float kept
    # improving) can be closed by --fp4-codebook=residual --fp4-stochastic
    # --fp4-robust-scale on residual-head renderers.
    qat_wrapper = None
    if args.use_qat:
        from tac.fp4_quantize import DEFAULT_CODEBOOK, RESIDUAL_CODEBOOK
        codebook = (RESIDUAL_CODEBOOK if args.fp4_codebook == "residual"
                    else DEFAULT_CODEBOOK).clone()
        qat_wrapper = QATRendererFP4(
            model,
            codebook=codebook,
            stochastic=args.fp4_stochastic,
            robust_scale=args.fp4_robust_scale,
        ).to(device)
        print(f"[train] FP4 QAT enabled (codebook={args.fp4_codebook}, "
              f"stochastic={args.fp4_stochastic}, "
              f"robust_scale={args.fp4_robust_scale})")

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

    for epoch in range(start_epoch, args.epochs):
        current_epoch = epoch
        epoch_start = time.monotonic()
        model.train()
        in_pretrain = has_pretrain and epoch < args.pretrain_epochs

        # Warmup LR
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Log phase transition
        if has_pretrain and epoch == args.pretrain_epochs:
            print(f"[train] === Phase 2 start (epoch {epoch}) === switching to scorer loss")

        # Sample pairs for this epoch
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

            # Half-frame mask simulation (Quantizr paradigm). With prob
            # --mask-half-sim-prob, replace mask_t with inverse_warp(mask_t1)
            # so this pair simulates inflate-time conditions where only the
            # odd-frame mask was archived. The GT pair (gt_pair below) is
            # unchanged — we still measure renderer output against the true
            # frames; only the mask INPUT distribution shifts.
            if (sim_zoom_warp is not None
                    and random.random() < args.mask_half_sim_prob):
                pair_idx_int = start // 2
                pair_idx_t = torch.tensor([pair_idx_int], device=device,
                                          dtype=torch.long)
                mask_t = sim_zoom_warp.warp_inverse_masks(mask_t1, pair_idx_t)

            gt_pair = pair_from_frames(gt_frames, start).to(device)

            # Horizontal flip augmentation (50% probability)
            if random.random() < 0.5:
                # mask shape is (B, H, W) so W is dim -1;
                # gt_pair shape is (B, 2, H, W, 3) so W is dim -2
                # (trailing channel dim shifts the index by one)
                mask_t = mask_t.flip(-1)
                mask_t1 = mask_t1.flip(-1)
                gt_pair = gt_pair.flip(-2)

            # Forward: render pair from masks
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
            if not in_pretrain:
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
                    # Luma proxy + local variance via 8x8 box filter (avg pool)
                    luma = rgb_gt.mean(dim=-1)  # (B, H, W)
                    luma_bchw = luma.unsqueeze(1)  # (B, 1, H, W)
                    local_mean = F.avg_pool2d(luma_bchw, kernel_size=8, stride=1, padding=4)
                    local_mean = local_mean[:, :, : luma_bchw.shape[2], : luma_bchw.shape[3]]
                    local_sqmean = F.avg_pool2d(luma_bchw.pow(2), kernel_size=8, stride=1, padding=4)
                    local_sqmean = local_sqmean[:, :, : luma_bchw.shape[2], : luma_bchw.shape[3]]
                    local_var = (local_sqmean - local_mean.pow(2)).clamp(min=0.0).squeeze(1)
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

                # KL distillation auxiliary — SegNet ONLY (R-fix-double-count).
                # The original wiring used kl_distill_scorer_loss which returns
                # 100*seg_kl + sqrt(10*pose_dist), and adding that to scorer_loss
                # double-counts BOTH terms (200x SegNet, double PoseNet pressure).
                # That matches the historical "KL distill caused PoseNet collapse"
                # failure mode per CLAUDE.md "Critical Lessons". Use the SegNet-
                # only helper that returns ONLY T²-scaled seg_kl.
                if args.kl_distill_weight > 0:
                    kl_loss, _kl_seg = kl_distill_segnet_only(
                        rendered_pair, gt_pair, segnet,
                        temperature=args.kl_distill_temperature,
                    )
                    loss = loss + args.kl_distill_weight * kl_loss

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

            # Gradient accumulation step (Feature 5: grad clipping already present)
            if (step + 1) % accum == 0 or (step + 1) == len(perm):
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            mask_t = mask_t1 = gt_pair = rendered_pair = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LR scheduler
        if epoch >= args.warmup_epochs:
            scheduler.step()

        n_steps = len(perm)
        avg_loss = total_loss / max(n_steps, 1)
        avg_pose = total_pose / max(n_steps, 1)
        avg_seg = total_seg / max(n_steps, 1)
        lr = optimizer.param_groups[0]["lr"]

        # Per-epoch timing (Feature 7)
        epoch_sec = time.monotonic() - epoch_start

        # Determine current phase label
        phase = "pretrain" if in_pretrain else "scorer"

        # FP4 evaluation (skip during Phase 1 -- scorer scores are meaningless)
        is_eval_epoch = (not in_pretrain and
                         ((epoch + 1) % args.eval_every == 0
                          or epoch == args.epochs - 1
                          or epoch == max(start_epoch, args.pretrain_epochs)))
        eval_pose, eval_seg = 0.0, 0.0
        if is_eval_epoch:
            scorer_val, eval_pose, eval_seg = evaluate_fp4(
                model, ema, all_masks, gt_frames,
                all_pair_starts, posenet, segnet, device,
                fp4_codebook=args.fp4_codebook,
                fp4_robust_scale=args.fp4_robust_scale,
            )
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
            fp4_packed["__meta__"] = {
                "fp4_codebook": args.fp4_codebook,
                "fp4_robust_scale": args.fp4_robust_scale,
                "fp4_stochastic": args.fp4_stochastic,
                "epoch": epoch,
                "scorer": best_scorer,
                "pose": eval_pose,
                "seg": eval_seg,
                "base_ch": args.base_ch,
                "mid_ch": args.mid_ch,
                "embed_dim": args.embed_dim,
                "motion_hidden": args.motion_hidden,
                "depth": args.depth,
                "latent_ch": args.latent_ch,
                "latent_shapes": args.latent_shapes,
                "residual_hidden": args.residual_hidden,
                "residual_layers": args.residual_layers,
                "residual_scale": args.residual_scale,
                "variant": args.variant,
                "seed": args.seed,
                "deterministic": args.deterministic,
            }
            fp4_tmp = fp4_path.with_suffix(".pt.tmp")
            torch.save(fp4_packed, fp4_tmp)
            fp4_tmp.rename(fp4_path)
            fp4_size = fp4_path.stat().st_size
            param_count = sum(p.numel() for p in model.parameters())
            print(f"[fp4] Saved {param_count:,} params to {fp4_path} ({fp4_size:,} bytes)")

            # Also save EMA fp32 for resuming
            fp32_path = out_dir / f"renderer_{args.tag}_best_fp32.pt"
            fp32_tmp = fp32_path.with_suffix(".pt.tmp")
            # R-fp4-export-fix: save FP4 metadata in the fp32 checkpoint too,
            # so pipeline.step_export reads it and uses the matching codebook
            # at re-export time. Without this, training writes residual codebook
            # in the fp4 file, but step_export's later re-export uses default.
            fp32_payload = {
                "model_state_dict": save_state,
                "__meta__": {
                    "fp4_codebook": args.fp4_codebook,
                    "fp4_robust_scale": args.fp4_robust_scale,
                    "fp4_stochastic": args.fp4_stochastic,
                    "epoch": epoch,
                    "variant": args.variant,
                    "seed": args.seed,
                },
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
        phase_tag = "P1" if in_pretrain else "P2"
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

    return best_scorer


def main():
    args = parse_args()
    return train(args)


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result is not None else 1)
