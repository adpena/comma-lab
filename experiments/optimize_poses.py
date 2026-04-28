#!/usr/bin/env python3
"""Pose-Space TTO: optimize FiLM conditioning vectors per pair.

Instead of pixel-level TTO (707M values, 40 min on 4090), optimize the
6D FiLM pose vectors (3,600 values total, should converge in seconds).

The "optimized poses" are not physically meaningful -- they are 6D instructions
to the FiLM layer for how to render each pair optimally for the scorers.

Archive: 600 x 6 x 4 = 14.4KB (same size as GT poses, just different values).
Contest-compliant: no scorers at inflate time, single forward pass.

Usage:
    # Smoke test (local MPS, 10 pairs, 100 steps):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device mps --smoke

    # Full run (4090):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device cuda

    # Extended conditioning (pose 6D + latent 16D = 22D):
    PYTHONPATH=src:upstream python experiments/optimize_poses.py \
        --checkpoint path/to/renderer_best.pt --device cuda --latent-dim 16
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
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup (must run before tac imports)
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ.get("TAC_RESULTS_DIR", ""))
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "pose_tto"
)


from tac.renderer import simulate_eval_roundtrip  # canonical impl (no local copy)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEGNET_INPUT_H, SEGNET_INPUT_W = 384, 512
NUM_FRAMES = 1200
NUM_PAIRS = NUM_FRAMES // 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose-Space TTO: optimize FiLM conditioning vectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to renderer .pt or .bin checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--n-frames", type=int, default=NUM_FRAMES,
                   help="Number of frames to process")
    p.add_argument("--steps", type=int, default=500,
                   help="Optimization steps per batch")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Adam learning rate for pose vectors")
    p.add_argument("--batch-pairs", type=int, default=50,
                   help="Pairs per optimization batch")
    p.add_argument("--seg-weight", type=float, default=100.0,
                   help="SegNet loss weight (hinge)")
    p.add_argument("--pose-weight", type=float, default=10.0,
                   help="PoseNet loss weight (MSE)")
    p.add_argument("--hinge-margin", type=float, default=0.5,
                   help="Margin for SegNet hinge loss")
    p.add_argument("--upstream", type=str, default=None,
                   help="Path to upstream repo (auto-detected if None)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: timestamped)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to GT video (default: upstream/videos/0.mkv)")
    # R38 fix: was optional default=None — silent pose-mask distribution
    # mismatch caused 27x PoseNet regression per CLAUDE.md. Make required.
    p.add_argument("--masks", type=str, required=True,
                   help="Path to pre-decoded masks (.pt or .mkv). "
                        "CRITICAL: must match the actual archive masks, not fresh SegNet. "
                        "Without this, poses are optimized against perfect masks but "
                        "deployed with lossy AV1 masks — 27x PoseNet regression.")
    p.add_argument("--gt-pose-targets", type=str, default=None,
                   help="Path to precomputed GT pose targets (.pt, shape [N_pairs, 6]). "
                        "Skips expensive PoseNet inference on all GT frame pairs. "
                        "Generate with: extract_gt_pose_targets() and torch.save().")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 20 frames, 100 steps")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resolution roundtrip in loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--posetto-noise-std", type=float, default=0.5,
                   help="Hotz STE: gaussian noise std added DURING simulate_eval_roundtrip "
                        "(default 0.5). 2026-04-26 Fridrich council CRITICAL: prior default "
                        "of 0.0 silently re-opened the proxy-CUDA gap up to 11x on PoseNet. "
                        "Set 0 only for diagnostic ablations.")
    p.add_argument("--early-stop-patience", type=int, default=100,
                   help="Stop if loss hasn't improved in this many steps")
    p.add_argument("--argmax-constraint", action="store_true",
                   help="Hard constraint: reject steps that flip SegNet argmax (projected GD)")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Max retries with halved LR when argmax flips (with --argmax-constraint)")
    p.add_argument("--flip-budget", type=float, default=0.0,
                   help="Fraction of pixels allowed to flip argmax (0.0=strict, 0.001=~3900 px budget)")
    p.add_argument("--latent-dim", type=int, default=0,
                   help="Extra latent dimensions per pair (0=pose only, 16=22D total)")
    p.add_argument("--gt-poses-path", type=str, default=None,
                   help="Path to pre-extracted GT poses (poses.pt). "
                        "If not provided, extracts from GT video.")
    p.add_argument("--seed-poses-path", type=str, default=None,
                   help="Lane OS-A: path to seed poses (.pt) produced by "
                        "experiments/seed_poses_from_openpilot.py — these "
                        "are used as the WARM-START init_poses tensor "
                        "instead of GT poses or PoseNet outputs. Takes "
                        "precedence over --gt-poses-path when both are set. "
                        "The shape must be (N_pairs, 6); the file is loaded "
                        "with weights_only=True. Per memory "
                        "project_openpilot_seeding_demo, supercombo at "
                        "compress time is contest-compliant — only the "
                        "(600, 6) seed tensor is consumed by this loop, "
                        "supercombo itself is never bundled in the archive.")
    p.add_argument("--optimize-embedding", action="store_true",
                   help="Also optimize the renderer's shared class embedding (30 values, 120 bytes). "
                        "Embedding is GLOBAL (shared across all pairs), optimized once before "
                        "per-pair pose optimization begins.")
    p.add_argument("--embedding-lr", type=float, default=0.005,
                   help="Adam learning rate for embedding optimization (with --optimize-embedding)")
    p.add_argument("--embedding-epochs", type=int, default=5,
                   help="Number of epochs over all pairs for embedding optimization")
    p.add_argument("--log-every", type=int, default=25,
                   help="Log metrics every N steps")
    p.add_argument("--kl-distill-weight", type=float, default=0.0,
                   help="SegNet KL-distill auxiliary loss weight (default 0 = "
                        "disabled). Lane G: stack soft-label SegNet KL on top "
                        "of the standard scorer loss to push the renderer "
                        "toward GT-frame logit distributions instead of just "
                        "argmax. Mirrors train_renderer.py wiring (uses "
                        "tac.losses.kl_distill_segnet_only — the canonical "
                        "helper that avoids the kl_distill_scorer_loss "
                        "double-count trap per CLAUDE.md).")
    p.add_argument("--kl-distill-temperature", type=float, default=2.0,
                   help="Softmax temperature for KL-distill (Hinton 2015 "
                        "default 2.0). Loss is multiplied by T² internally "
                        "to keep gradient magnitude consistent with T=1.")
    # Lane G V3-V2 (Lagrangian SNR): replaces the hand-derived
    # --kl-distill-weight constant with a multiplicative dual-ascent rule
    # that maintains a target SNR (KL contribution / scorer contribution).
    # See src/tac/lagrangian_kl_weight.py for the full convergence proof
    # (Boyd & Vandenberghe §5.4 strong duality + Kivinen & Warmuth 1997
    # exponentiated gradient on ratio constraints). Mutually exclusive
    # with --kl-distill-weight (validated below in main()).
    p.add_argument("--kl-distill-snr-target", type=float, default=None,
                   help="Lane G V3-V2: Lagrangian-controlled KL weight. "
                        "Targets the auxiliary KL contribution / scorer "
                        "contribution ratio (default 0.10 = canonical "
                        "Hinton 2015 auxiliary regime when set). When "
                        "supplied, REPLACES --kl-distill-weight with a "
                        "multiplicative dual-ascent controller "
                        "(LearnableKLWeight). Mutually exclusive with "
                        "--kl-distill-weight (one or the other, not "
                        "both). The controller's initial weight is the "
                        "value of --kl-distill-weight if it was passed, "
                        "else the canonical 0.002.")
    p.add_argument("--kl-distill-snr-eta", type=float, default=0.5,
                   help="Lane G V3-V2: dual-ascent step size in log-space "
                        "for the SNR controller. Default 0.5 — converges "
                        "geometrically (η ≤ 1 is contractive).")
    # Lane M: radial-zoom-only pose mode. Per memory
    # `project_posenet_rank1_discovery`, PoseNet's Jacobian is rank ≈ 1.008
    # with 99.8% variance in dim 0 — a scalar radial zoom from the
    # Focus-of-Expansion is the information-theoretic minimum. The 6-DOF
    # representation is grossly over-parameterized; the auxiliary 5 dims
    # add optimizer noise without scoring signal. When `radial-zoom` is
    # selected the optimizable parameter is (N, 1) (the canonical "z
    # forward" component) and is projected back to (N, 6) by zero-padding
    # the other 5 dims before the renderer call. The (N, 1) tensor is
    # persisted to disk; the inflate-side reader expands it back.
    p.add_argument("--pose-mode", type=str, default="full-6dof",
                   choices=["full-6dof", "radial-zoom"],
                   help="Pose representation. 'full-6dof' (default) "
                        "preserves the original (N, 6) optimizable. "
                        "'radial-zoom' optimizes (N, 1) radial-zoom "
                        "scalars only, projecting to 6D as "
                        "[zoom, 0, 0, 0, 0, 0] before render. "
                        "Per memory project_posenet_rank1_discovery the "
                        "Jacobian is rank-1 — the other 5 dims add noise.")
    # Lane N: Fridrich L∞ auxiliary penalty. Per memory
    # `project_fridrich_inverse_steganalysis` Principle 3 ("spread small
    # errors, don't concentrate large ones"), penalizing the L∞ norm of
    # the (current_pose - baseline_pose) delta keeps the per-dim
    # perturbation bounded — PoseNet's detector is sensitive to
    # concentrated changes, so a soft L∞-ball constraint biases the
    # optimizer toward uniformly small perturbations. The helper is
    # `tac.fridrich.linf_pose_penalty` (see memory
    # feedback_existing_fridrich_code — Fridrich code already exists,
    # don't rebuild). Defaults preserve the baseline call path
    # byte-identically.
    p.add_argument("--linf-pose-weight", type=float, default=0.0,
                   help="Lane N: Fridrich L∞ pose-perturbation penalty "
                        "weight (default 0.0 = disabled). When > 0, "
                        "penalizes per-dim |current_pose - baseline_pose| "
                        "above --linf-pose-budget via "
                        "tac.fridrich.linf_pose_penalty. Spreads "
                        "perturbations uniformly so PoseNet (which detects "
                        "concentrated changes) cannot single any one dim "
                        "out.")
    p.add_argument("--linf-pose-budget", type=float, default=0.05,
                   help="Lane N: per-dim L∞ ball radius around "
                        "baseline_pose. Default 0.05 — small enough not to "
                        "perturb the rank-1 dominant dim "
                        "(project_posenet_rank1_discovery), large enough "
                        "to allow exploration on aux dims. Only takes "
                        "effect when --linf-pose-weight > 0.")
    # Lane PS (per-class SegNet weighting). Per memory
    # `project_research_survey_20260420` — research-grade, never
    # implemented. SegNet predicts 5-class segmentation; cheap classes
    # (road, sky) and costly classes (lane mark, vehicle) are averaged
    # uniformly today. When this flag is supplied, the auxiliary KL
    # distillation loss (kl_distill_segnet_only) multiplies its per-pixel
    # contribution by the weight at each pixel's GT-argmax class, biasing
    # the optimiser toward costly classes. Default ``""`` (empty CSV) is a
    # no-op — byte-identical to baseline. The CSV is parsed via
    # tac.losses.parse_class_weights_csv (5 floats, non-negative, not all
    # zero) so the operator catches "5,5,5,5" → 4-class footgun at CLI
    # parse time, not 30 min into a run.
    p.add_argument("--segnet-class-weights", type=str, default="",
                   help="Lane PS: CSV of 5 per-class weights for the "
                        "auxiliary KL distill SegNet loss "
                        "(e.g., '1,5,5,1,1' to boost lane + boundary "
                        "classes). Default '' (empty) = uniform weighting "
                        "(byte-identical to baseline). Only takes effect "
                        "when --kl-distill-weight > 0; pose TTO's primary "
                        "SegNet hinge loss is unaffected (this auxiliary "
                        "is the costly-class lever).")
    # Lane PS-V2 (2026-04-27): replace the static --segnet-class-weights
    # CSV with a LEARNABLE softmax-parameterised 5-vector. The CSV (when
    # supplied) is the warm-start. The optimiser equalises per-class
    # distortion variance via a Lagrangian penalty during the standard
    # loss loop. See tac.learnable_class_weights and
    # project_arbitrary_vs_learnable_taxonomy.
    p.add_argument("--learnable-segnet-class-weights", action="store_true",
                   default=False,
                   help="Lane PS-V2: replace the static --segnet-class-weights "
                        "CSV with a LEARNABLE softmax-parameterised "
                        "5-vector (LearnableClassWeights). Warm-start = "
                        "the parsed --segnet-class-weights CSV (or "
                        "uniform when empty). Adds the weights to the "
                        "pose-TTO optimiser as an extra parameter group "
                        "with its own learning rate.")
    p.add_argument("--learnable-segnet-class-weights-lr", type=float,
                   default=1e-2,
                   help="Lane PS-V2: learning rate for the "
                        "LearnableClassWeights parameter group.")
    p.add_argument("--learnable-segnet-class-weights-var-lambda", type=float,
                   default=1.0,
                   help="Lane PS-V2: Lagrangian multiplier on the per-class "
                        "distortion variance penalty (equalisation term).")
    # Lane LR (Low-Rank pose adaptation). Per memory
    # `project_posenet_rank1_discovery` the PoseNet Jacobian is rank ≈ 1.008
    # — the (N_pairs, 6) pose tensor lives on a rank-1 manifold. Lane LR
    # parameterises poses as base + U @ V where base = warm-start
    # (frozen), U is (N_pairs, R), V is (R, 6). Optimising U + V instead
    # of the full (N_pairs, 6) tensor saves rate (R=1: 606 fp16 ≈ 1.2KB
    # vs 7.2KB baseline → ≈0.004 score improvement on rate alone, with
    # distortion predicted neutral by the rank-1 hypothesis).
    #
    # Implementation contract:
    #   --lora-rank 0   (default) → full-rank pose TTO, byte-identical to
    #                                pre-Lane-LR baseline.
    #   --lora-rank 1+  → all pairs optimised JOINTLY in a single pass
    #                     (LoRA U is per-pair, V is shared globally — batched
    #                      per-block optimisation would defeat the global
    #                      basis sharing).
    p.add_argument("--lora-rank", type=int, default=0,
                   help="Lane LR: rank of the U @ V LoRA factorisation of "
                        "the per-pair pose tensor. 0 (default) disables "
                        "LoRA — runs the full-rank baseline TTO loop. "
                        "1, 2, 3 enable LoRA at that rank. The (N, 6) pose "
                        "tensor is parameterised as base + U @ V where "
                        "base is the warm-start (frozen), U is (N_pairs, R) "
                        "and V is (R, 6). Per memory "
                        "project_posenet_rank1_discovery rank-1 captures "
                        "99.8%% of PoseNet variance; higher ranks add rate "
                        "without distortion benefit.")
    p.add_argument("--lora-steps", type=int, default=0,
                   help="Lane LR: optimisation steps for the GLOBAL LoRA "
                        "pass (only meaningful when --lora-rank > 0). "
                        "0 (default) means inherit from --steps. The LoRA "
                        "loop sees ALL pairs each step (vs the full-rank "
                        "loop's per-batch inner loop), so the per-step "
                        "gradient is much richer; 500 is usually enough.")
    # Lane LR-V2 (2026-04-27). The V1 --lora-rank is FROZEN (operator picks
    # the rank offline); V2 --learnable-lora-max-rank is LEARNABLE (the rank
    # is data-driven via per-rank gates that the optimiser can drive toward
    # zero). Per project_posenet_rank1_discovery the optimal rank is most
    # likely 1, but treating it as a hyperparameter the optimiser can prune
    # is strictly more general than picking it offline.
    #
    # Mutual exclusion: passing both --lora-rank > 0 AND
    # --learnable-lora-max-rank > 0 is a hard error (the V2 pass would
    # supersede the V1 pass and the operator's intent is ambiguous).
    p.add_argument("--learnable-lora-max-rank", type=int, default=0,
                   help="Lane LR-V2: maximum rank for the LEARNABLE-rank LoRA "
                        "pose factorisation. 0 (default) disables Lane LR-V2 "
                        "(use --lora-rank for the V1 frozen-rank path). "
                        "Recommended value: 6 (= pose_dim, the upper bound on "
                        "useful rank). Per-rank gates start at sigmoid(0)=0.5 "
                        "and are co-trained with U + V; ranks with final "
                        "gate < --lora-prune-threshold are pruned before "
                        "serialisation.")
    p.add_argument("--lora-prune-threshold", type=float, default=0.1,
                   help="Lane LR-V2: minimum final gate value for a rank to "
                        "survive pruning (default 0.1). Lower values keep "
                        "more ranks; higher values prune more aggressively. "
                        "Only meaningful when --learnable-lora-max-rank > 0.")
    p.add_argument("--lora-init-gate-logit", type=float, default=0.0,
                   help="Lane LR-V2: initial logit for every per-rank gate "
                        "(default 0.0 → sigmoid(0)=0.5 = half-on per rank). "
                        "Negative values bias toward sparsity, positive "
                        "values bias toward keeping ranks. Only meaningful "
                        "when --learnable-lora-max-rank > 0.")
    # Lane RM (Riemannian SE(3)). Per Absil-Mahony-Sepulchre 2008 + Boumal
    # 2023 + Bonnabel 2013: poses live on the SE(3) Lie group, NOT in flat
    # ℝ⁶. Standard Euclidean SGD/Adam on (axis-angle ω, translation t)
    # accumulates orthogonality drift on the rotation factor across steps;
    # a Riemannian optimiser uses the SE(3) exponential map as the
    # retraction so rotations stay in SO(3) by construction. Convergence
    # rate matches Euclidean SGD on smooth manifolds (Bonnabel 2013) but
    # the constant factor improves on the SO(3) submanifold.
    #
    # Pose-tensor convention when --optimizer=riemannian-sgd: each row of
    # the optimisable conditioning tensor is interpreted as
    # (ω_x, ω_y, ω_z, t_x, t_y, t_z) where ω is axis-angle (so(3)) and t
    # is the Cartesian translation. This matches the comma.ai PoseNet
    # 6-vector output (first three angular, last three translational).
    #
    # Mutual exclusion: --optimizer=riemannian-sgd requires
    # --pose-mode=full-6dof (the SE(3) optimiser needs a 6-DOF tensor) AND
    # --lora-rank=0 + --learnable-lora-max-rank=0 (the LoRA paths have
    # their own dedicated Adam optimisers and aren't on the SE(3)
    # manifold).
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "sgd", "riemannian-sgd"],
                   help="Per-batch optimiser for the (B, 6) pose tensor. "
                        "'adam' (default, backward-compatible) uses "
                        "torch.optim.Adam. 'sgd' uses Euclidean "
                        "torch.optim.SGD with momentum=0.9. "
                        "'riemannian-sgd' (Lane RM) treats each pose row "
                        "as an SE(3) element (ω: so(3), t: ℝ³) and steps "
                        "via the SE(3) exponential map "
                        "(tac.riemannian_pose_optimizer.RiemannianSGD). "
                        "Required: --pose-mode=full-6dof, --lora-rank=0, "
                        "--learnable-lora-max-rank=0. Predicted band "
                        "[1.05, 1.15] vs Lane A's 1.15 [contest-CUDA].")
    p.add_argument("--riemannian-momentum", type=float, default=0.9,
                   help="Polyak heavy-ball momentum for "
                        "--optimizer=riemannian-sgd (default 0.9, matches "
                        "torch.optim.SGD's typical setting). The velocity "
                        "buffer lives in se(3) coordinates; with the "
                        "left-invariant SE(3) metric, parallel transport "
                        "along the geodesic is the identity on the Lie "
                        "algebra (Sola 2018 Eq. 174) so no transport "
                        "correction is needed between steps. Only takes "
                        "effect when --optimizer=riemannian-sgd.")
    return p.parse_args()


def load_renderer(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load AsymmetricPairGenerator from checkpoint, .bin, or .pt."""
    from pathlib import Path

    raw = Path(checkpoint_path).read_bytes()

    # Auto-detect format by magic bytes
    if raw[:4] == b"ASYM":
        from tac.renderer_export import load_asymmetric_checkpoint
        model = load_asymmetric_checkpoint(raw, device=str(device))
        model = model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        pose_dim = model.pose_dim
        print(f"[renderer] Loaded ASYM: {n_params:,} params, pose_dim={pose_dim}", flush=True)
        return model

    if raw[:4] == b"FP4A":
        from tac.renderer_export import load_asymmetric_checkpoint_fp4
        model = load_asymmetric_checkpoint_fp4(raw, device=str(device))
        model = model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        pose_dim = model.pose_dim
        print(f"[renderer] Loaded FP4A: {n_params:,} params, pose_dim={pose_dim}", flush=True)
        return model

    # PyTorch .pt format
    from tac.renderer import AsymmetricPairGenerator
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", ckpt.get("config", {}))
    # Every arch field that AsymmetricPairGenerator accepts must be threaded
    # from model_cfg, including padding_mode and use_dilation. Missing either
    # silently changes Conv2d behavior at boundaries → wrong outputs → wrong
    # pose TTO targets. (Round 23 finding.)
    model = AsymmetricPairGenerator(
        num_classes=model_cfg.get("num_classes", 5),
        embed_dim=model_cfg.get("embed_dim", 6),
        base_ch=model_cfg.get("base_ch", 36),
        mid_ch=model_cfg.get("mid_ch", 60),
        motion_hidden=model_cfg.get("motion_hidden", 32),
        depth=model_cfg.get("depth", 1),
        max_flow_px=model_cfg.get("max_flow_px", 20.0),
        max_residual=model_cfg.get("max_residual", 20.0),
        flow_only=model_cfg.get("flow_only", False),
        pose_dim=model_cfg.get("pose_dim", 6),
        use_dsconv=model_cfg.get("use_dsconv", False),
        padding_mode=model_cfg.get("padding_mode", "zeros"),
        use_dilation=model_cfg.get("use_dilation", False),
        use_zoom_flow=model_cfg.get("use_zoom_flow", False),
    )

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.eval().to(device)
    # Freeze ALL renderer parameters
    for p_param in model.parameters():
        p_param.requires_grad = False

    n_params = sum(p_param.numel() for p_param in model.parameters())
    pose_dim = model_cfg.get("pose_dim", 0)
    print(f"[renderer] Loaded {n_params:,} params, pose_dim={pose_dim} from {checkpoint_path}", flush=True)

    if pose_dim == 0:
        print("[WARNING] Renderer has pose_dim=0 -- FiLM layers are disabled.", flush=True)
        print("[WARNING] Pose optimization will have NO effect on output.", flush=True)
        print("[WARNING] Consider training a renderer with --pose-dim 6.", flush=True)
    return model


def segnet_hinge_loss(
    logits: torch.Tensor,
    gt_masks: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Hinge loss on SegNet logits: penalize pixels at risk of argmax flip.

    For each pixel, the loss is max(0, margin - (correct_logit - max_other_logit)).
    This focuses gradient on boundary pixels where argmax might flip, which is
    much more efficient than cross-entropy (2-5x empirically).

    Args:
        logits: (B, C, H, W) raw SegNet output
        gt_masks: (B, H, W) long tensor of GT class indices
        margin: desired minimum gap between correct and runner-up logit

    Returns:
        Scalar mean hinge loss
    """
    B, C, H, W = logits.shape
    # Gather correct class logits
    correct = logits.gather(1, gt_masks.unsqueeze(1)).squeeze(1)  # (B, H, W)
    # Mask out correct class to find runner-up
    mask_inf = torch.zeros_like(logits)
    mask_inf.scatter_(1, gt_masks.unsqueeze(1), float("-inf"))
    runner_up = (logits + mask_inf).max(dim=1).values  # (B, H, W)
    # Hinge: penalize when gap < margin
    loss = F.relu(margin - (correct - runner_up))
    return loss.mean()


def optimize_poses_batch(
    renderer: torch.nn.Module,
    masks_t: torch.Tensor,
    masks_t1: torch.Tensor,
    gt_masks: torch.Tensor,
    pose_targets: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    init_poses: torch.Tensor,
    device: torch.device,
    steps: int = 500,
    lr: float = 0.01,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    hinge_margin: float = 0.5,
    eval_roundtrip: bool = True,
    posetto_noise_std: float = 0.5,
    early_stop_patience: int = 100,
    argmax_constraint: bool = False,
    max_retries: int = 3,
    flip_budget: float = 0.0,
    zoom_warp: torch.nn.Module | None = None,
    batch_pair_indices: torch.Tensor | None = None,
    latent_dim: int = 0,
    log_every: int = 25,
    gt_frames_pair: torch.Tensor | None = None,
    kl_distill_weight: float = 0.0,
    kl_distill_temperature: float = 2.0,
    pose_mode: str = "full-6dof",
    linf_pose_weight: float = 0.0,
    linf_pose_budget: float = 0.05,
    segnet_class_weights: torch.Tensor | None = None,
    kl_distill_snr_controller: "object | None" = None,
    optimizer_kind: str = "adam",
    riemannian_momentum: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """Optimize pose vectors (and optional latent codes) for a batch of pairs.

    Args:
        renderer: FROZEN AsymmetricPairGenerator
        masks_t, masks_t1: (B, H, W) even/odd masks
        gt_masks: (2*B, H, W) GT SegNet masks for the batch frames
        pose_targets: (B, 6) GT PoseNet outputs for these pairs
        posenet, segnet: FROZEN differentiable scorers
        init_poses: (B, 6) initial pose vectors (warm start from GT)
        device: computation device
        steps: optimization steps
        lr: Adam learning rate
        seg_weight: SegNet hinge loss weight
        pose_weight: PoseNet MSE loss weight
        hinge_margin: margin for hinge loss
        eval_roundtrip: simulate eval resolution roundtrip
        early_stop_patience: stop if no improvement for this many steps
        argmax_constraint: if True, reject steps that flip SegNet argmax
        max_retries: max retries with halved step when argmax flips
        flip_budget: fraction of pixels allowed to flip (0.0=strict)
        latent_dim: extra latent dimensions (0 = pose only)
        log_every: logging frequency
        gt_frames_pair: (B, 2, H, W, 3) float GT frames in 0-255 range,
            required when kl_distill_weight > 0. Mirrors train_renderer.py's
            `gt_pair` shape contract — segnet.preprocess_input handles
            normalization internally.
        kl_distill_weight: auxiliary SegNet KL-distillation loss weight
            (default 0 = disabled). Uses tac.losses.kl_distill_segnet_only,
            same canonical helper as train_renderer (avoids the
            kl_distill_scorer_loss double-count trap per CLAUDE.md).
        kl_distill_temperature: softmax temperature for the KL distill
            (default 2.0, Hinton 2015). Loss is multiplied by T² internally.
        pose_mode: 'full-6dof' (default) or 'radial-zoom'. With
            'radial-zoom', the optimizable parameter is (B, 1) (the
            canonical "z forward" component); it is projected to 6-DOF
            via [zoom, 0, 0, 0, 0, 0] before being passed to the
            renderer. The renderer was trained on 6-DOF input so the
            projection is mandatory. Per memory
            project_posenet_rank1_discovery the PoseNet Jacobian is
            rank ≈ 1.008 — only the radial-zoom dim carries scoring
            signal; optimizing 6 params adds noise.
        linf_pose_weight: Lane N — Fridrich L∞ penalty weight on
            (current_pose - baseline_pose). Default 0.0 = disabled.
            When > 0, an auxiliary penalty
            ``linf_pose_weight * sum(max(0, |delta| - budget))`` is
            added to total_loss to bias toward uniformly small
            perturbations (PoseNet detects concentrated changes).
        linf_pose_budget: per-dim L∞ ball radius for the Lane N
            penalty (default 0.05). Only takes effect when
            linf_pose_weight > 0.
        segnet_class_weights: Lane PS — (NUM_CLASSES,) optional per-class
            weights for the AUXILIARY KL distill SegNet loss (the
            ``kl_distill_segnet_only`` call). Default ``None`` = uniform
            weighting (byte-identical to baseline). Only takes effect when
            ``kl_distill_weight > 0`` AND ``gt_frames_pair is not None``;
            otherwise the KL block is skipped and the weights are unused.
            Pose TTO's primary SegNet hinge loss is left untouched — the
            costly-class lever is intentionally limited to the soft-label
            auxiliary so it cannot accidentally crater the per-pixel
            argmax-flip signal that drives pose convergence.
        kl_distill_snr_controller: Lane G V3-V2 — optional
            ``LearnableKLWeight`` controller (see
            ``tac.lagrangian_kl_weight``). When supplied, REPLACES the
            static ``kl_distill_weight`` argument: the per-step weight is
            pulled from the controller and updated after each step using
            the observed SNR. ``kl_distill_weight`` is ignored when this
            argument is non-None (mutually exclusive). Only takes effect
            when ``gt_frames_pair is not None``.

    Returns:
        (optimized_conditioning, metrics_dict)
        conditioning is (B, pose_dim_internal + latent_dim) where
        pose_dim_internal is 1 for pose_mode='radial-zoom' and pose_dim
        otherwise.
    """
    B = masks_t.shape[0]
    pose_dim = init_poses.shape[1]

    if latent_dim > 0:
        raise NotImplementedError(
            f"--latent-dim {latent_dim} is not yet supported: the renderer has no "
            f"latent FiLM layer, so latent dimensions receive zero gradient and "
            f"produce meaningless output. Remove --latent-dim or implement latent "
            f"conditioning in the renderer architecture first."
        )

    # Lane M: in 'radial-zoom' mode the optimizable param is (B, 1) — the
    # scalar radial-zoom component (canonical pose dim 0, "z forward" per
    # FastViT-T12 PoseNet convention). It is projected to (B, pose_dim)
    # before being fed to the renderer (the renderer was trained on
    # 6-DOF; projecting padding the other 5 dims with 0 keeps the FiLM
    # input shape contract). pose_dim_internal = number of optimizable
    # values per pair; pose_dim = number expected by the renderer's FiLM
    # input.
    if pose_mode == "radial-zoom":
        pose_dim_internal = 1
    elif pose_mode == "full-6dof":
        pose_dim_internal = pose_dim
    else:
        raise ValueError(
            f"Unknown pose_mode={pose_mode!r}. Expected 'full-6dof' or "
            f"'radial-zoom'."
        )

    cond_dim = pose_dim_internal + latent_dim

    # Initialize conditioning vector: [pose (warm start) | latent (zeros)]
    conditioning = torch.zeros(B, cond_dim, device=device, dtype=torch.float32)
    if pose_mode == "radial-zoom":
        # Warm-start the 1-DOF zoom from the canonical dim-0 of the GT
        # pose target. The remaining 5 GT dims are dropped (per the rank-1
        # discovery, they carry < 0.2% variance). This is also the
        # `RadialZoomWarp` zero-init convention (identity zoom).
        conditioning[:, 0] = init_poses[:B, 0].to(device)
    else:
        conditioning[:, :pose_dim] = init_poses[:B].to(device)
    conditioning.requires_grad_(True)

    # Lane N: cache the baseline pose tensor (in 6-DOF for full mode,
    # 1-DOF for radial-zoom mode — matches the optimizable shape so the
    # delta is well-defined). Detached so the L∞ penalty's gradient flows
    # only into `conditioning`.
    if pose_mode == "radial-zoom":
        baseline_pose_for_linf = init_poses[:B, 0:1].detach().to(device)
    else:
        baseline_pose_for_linf = init_poses[:B, :pose_dim].detach().to(device)

    # Lane RM: dispatch on optimizer_kind. The default 'adam' is
    # byte-identical to the pre-Lane-RM call path (single line above).
    # 'sgd' is the Euclidean baseline (used as an A/B control for the
    # Riemannian / Euclidean comparison). 'riemannian-sgd' uses the
    # SE(3) exponential map as the retraction so rotations stay in SO(3)
    # by construction (Absil-Mahony-Sepulchre §3.5; Boumal §10).
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam([conditioning], lr=lr)
    elif optimizer_kind == "sgd":
        optimizer = torch.optim.SGD([conditioning], lr=lr, momentum=0.9)
    elif optimizer_kind == "riemannian-sgd":
        # Pre-conditions the CLI validator already enforced — re-asserted
        # here so a programmatic caller (test, notebook) gets the same
        # fail-loud behaviour as a CLI invocation.
        if pose_mode != "full-6dof":
            raise ValueError(
                "optimizer_kind='riemannian-sgd' requires pose_mode='full-6dof'; "
                f"got pose_mode={pose_mode!r}."
            )
        if conditioning.shape[-1] != 6:
            raise ValueError(
                "optimizer_kind='riemannian-sgd' requires conditioning of "
                f"last-dim 6 (interpreted as (omega, t)); got "
                f"shape {tuple(conditioning.shape)}."
            )
        from tac.riemannian_pose_optimizer import RiemannianSGD
        optimizer = RiemannianSGD(
            [conditioning], lr=lr, momentum=riemannian_momentum,
        )
    else:
        raise ValueError(
            f"Unknown optimizer_kind={optimizer_kind!r}. Expected one of "
            "{'adam', 'sgd', 'riemannian-sgd'}."
        )

    best_loss = float("inf")
    best_cond = conditioning.detach().clone()
    best_seg_loss = 0.0
    best_pose_loss = 0.0
    patience_counter = 0
    argmax_rejections = 0

    # Compute reference SegNet argmax if using hard constraint
    if argmax_constraint:
        with torch.no_grad():
            ref_pose = init_poses.to(device)
            ref_kwargs = {"pose": ref_pose}
            if zoom_warp is not None and batch_pair_indices is not None:
                ref_kwargs["ego_flow"] = zoom_warp(batch_pair_indices, masks_t.shape[1], masks_t.shape[2])
            ref_pairs = renderer(masks_t, masks_t1, **ref_kwargs)
            ref_ft = ref_pairs[:, 0]
            ref_ft1 = ref_pairs[:, 1]
            ref_hwc = torch.cat([ref_ft, ref_ft1], dim=0)
            ref_chw = ref_hwc.permute(0, 3, 1, 2).contiguous()
            ref_seg_in = segnet.preprocess_input(ref_chw.unsqueeze(1))
            ref_seg_logits = segnet(ref_seg_in)
            ref_argmax = ref_seg_logits.argmax(dim=1)  # (2*B, H, W)
        flip_threshold = int(ref_argmax.numel() * flip_budget)
        print(f"  [argmax constraint] Reference argmax: {ref_argmax.shape}, "
              f"flip budget={flip_budget:.4f} ({flip_threshold} px)", flush=True)

    metrics = {
        "steps_run": 0,
        "final_loss": float("inf"),
        "final_pose_loss": float("inf"),
        "final_seg_loss": float("inf"),
        "initial_loss": float("inf"),
        "improvement_pct": 0.0,
        "argmax_rejections": 0,
    }

    def _project_to_renderer_pose(cond: torch.Tensor) -> torch.Tensor:
        """Lane M projection: extract optimizable pose and lift to 6-DOF.

        Renderer's FiLM layer was trained on (B, pose_dim) input. In
        radial-zoom mode the optimizable is (B, 1); we zero-pad the
        remaining ``pose_dim - 1`` dims so the FiLM activation matches
        the training distribution as closely as possible (only the dim
        that PoseNet's Jacobian actually responds to is non-zero).
        """
        opt_part = cond[:, :pose_dim_internal]
        if pose_mode == "radial-zoom" and pose_dim_internal != pose_dim:
            zeros_pad = torch.zeros(
                opt_part.shape[0],
                pose_dim - pose_dim_internal,
                device=opt_part.device,
                dtype=opt_part.dtype,
            )
            return torch.cat([opt_part, zeros_pad], dim=-1)
        return opt_part

    for step in range(steps):
        # Save state before step (for argmax constraint rollback)
        if argmax_constraint:
            pre_step_cond = conditioning.detach().clone()
            pre_step_state = optimizer.state_dict()

        optimizer.zero_grad()

        # Extract pose part of conditioning (projected to renderer's
        # native pose_dim — Lane M radial-zoom mode lifts (B, 1) → (B, 6)
        # by zero-padding the auxiliary dims).
        pose_part = _project_to_renderer_pose(conditioning)

        # Forward: renderer produces (B, 2, H, W, 3) HWC pairs
        fwd_kwargs = {"pose": pose_part}
        if zoom_warp is not None and batch_pair_indices is not None:
            fwd_kwargs["ego_flow"] = zoom_warp(batch_pair_indices, masks_t.shape[1], masks_t.shape[2])
        pairs = renderer(masks_t, masks_t1, **fwd_kwargs)  # (B, 2, H, W, 3)

        # Convert to CHW for scorer input
        frame_t = pairs[:, 0]   # (B, H, W, 3)
        frame_t1 = pairs[:, 1]  # (B, H, W, 3)
        frames_hwc = torch.cat([frame_t, frame_t1], dim=0)  # (2*B, H, W, 3)
        frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()  # (2*B, 3, H, W)

        # Optional eval roundtrip
        if eval_roundtrip:
            frames_chw = simulate_eval_roundtrip(frames_chw, noise_std=posetto_noise_std)

        # --- SegNet loss (hinge) ---
        seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
        seg_logits = segnet(seg_in)  # (2*B, 5, H, W)

        seg_loss = segnet_hinge_loss(seg_logits, gt_masks.to(device), margin=hinge_margin)

        # --- PoseNet loss (MSE on 6D output) ---
        pairs_chw = torch.stack([
            frame_t.permute(0, 3, 1, 2),
            frame_t1.permute(0, 3, 1, 2),
        ], dim=1)  # (B, 2, 3, H, W)

        if eval_roundtrip:
            B_p, T_p, C_p, H_p, W_p = pairs_chw.shape
            flat = pairs_chw.reshape(B_p * T_p, C_p, H_p, W_p)
            flat = simulate_eval_roundtrip(flat, noise_std=posetto_noise_std)
            pairs_chw = flat.reshape(B_p, T_p, C_p, H_p, W_p)

        pose_in = posenet.preprocess_input(pairs_chw)
        pose_out = posenet(pose_in)["pose"][..., :6]  # (B, 6)
        pose_loss = F.mse_loss(pose_out, pose_targets[:pose_out.shape[0]].to(device))

        # Combined loss
        total_loss = seg_weight * seg_loss + pose_weight * pose_loss

        # KL distillation auxiliary — SegNet ONLY (mirrors train_renderer.py
        # ~L1773-1778). Use the SegNet-only helper, NOT kl_distill_scorer_loss
        # (the latter returns 100*seg_kl + sqrt(10*pose_dist) and stacking it
        # double-counts the SegNet term 200x and adds extra PoseNet pressure,
        # historically causing PoseNet collapse per CLAUDE.md "Critical
        # Lessons"). Renderer pairs are float 0-255 (rendered_pair.round()
        # .clamp(0,255) contract); GT frames are uint8 0-255 cast to float
        # by the caller; segnet.preprocess_input normalizes both.
        #
        # 2026-04-27 codex R5-r6 #2 fix: feed the SAME eval-roundtripped
        # frames the SegNet scorer path sees — NOT raw `pairs`. The
        # standard SegNet loss above already calls
        # `simulate_eval_roundtrip(frames_chw, ...)` before
        # `segnet.preprocess_input`. The KL auxiliary previously passed
        # `pairs` (raw, NOT roundtripped) which optimised against logits
        # for a different rendered distribution than the scored loss path
        # — meaning Lane G's KL gradients pulled the renderer in the
        # wrong direction relative to the scoring distribution. Now we
        # reshape the roundtripped CHW back to HWC and pass that.
        # The roundtrip is a no-op when eval_roundtrip=False, so the
        # default-off path is unchanged.
        kl_loss_val = 0.0
        # Lane G V3-V2: when an SNR controller is supplied, the per-step
        # weight is pulled from the controller (and updated AFTER this
        # step from the observed scorer/kl values). Otherwise fall back
        # to the static --kl-distill-weight constant.
        _effective_kl_weight = (
            kl_distill_snr_controller.weight
            if kl_distill_snr_controller is not None
            else kl_distill_weight
        )
        if _effective_kl_weight > 0 and gt_frames_pair is not None:
            from tac.losses import kl_distill_segnet_only
            # frames_chw is (2*B, 3, H, W) — roundtripped above when
            # eval_roundtrip=True, raw otherwise. Permute back to HWC
            # then unflatten the (2*B, ...) axis into (B, 2, ...) so the
            # shape matches the kl_distill_segnet_only contract
            # (B, T, H, W, C). Use the SAME simulate_eval_roundtrip
            # output the SegNet path consumed — do NOT call it again
            # (would double-roundtrip and add 2x noise + 2x interp loss).
            B_kl = pairs.shape[0]
            frames_hwc_rt = frames_chw.permute(0, 2, 3, 1).contiguous()  # (2*B, H, W, 3)
            rendered_pair_hwc_rt = frames_hwc_rt.view(
                2, B_kl, frames_hwc_rt.shape[1], frames_hwc_rt.shape[2], 3
            ).permute(1, 0, 2, 3, 4).contiguous()  # (B, 2, H, W, 3)
            kl_loss, kl_loss_val = kl_distill_segnet_only(
                rendered_pair_hwc_rt, gt_frames_pair, segnet,
                temperature=kl_distill_temperature,
                class_weights=segnet_class_weights,
            )
            total_loss = total_loss + _effective_kl_weight * kl_loss
            # Lane G V3-V2: update the SNR controller AFTER the step has
            # been added to total_loss. Use the SCORER residual (without
            # the KL contribution) as the denominator — that's the
            # "primary signal" the controller is targeting a fixed
            # ratio against. ``seg_weight*seg_loss + pose_weight*pose_loss``
            # is the canonical scorer residual (== total_loss before the
            # KL was added).
            if kl_distill_snr_controller is not None:
                _scorer_residual = float(
                    seg_weight * seg_loss.item() + pose_weight * pose_loss.item()
                )
                kl_distill_snr_controller.update(
                    kl_value=float(kl_loss_val),
                    scorer_value=_scorer_residual,
                )

        # Lane N: Fridrich L∞ pose-perturbation penalty. Operates on the
        # OPTIMIZABLE pose tensor (1-DOF in radial-zoom mode, 6-DOF in
        # full mode) so the penalty shape matches the parameter shape and
        # the gradient flows back into `conditioning`. Helper:
        # tac.fridrich.linf_pose_penalty (per CLAUDE.md "Fridrich Code
        # Already Exists" — don't rebuild).
        linf_loss_val = 0.0
        if linf_pose_weight > 0:
            from tac.fridrich import linf_pose_penalty
            optimizable_pose = conditioning[:, :pose_dim_internal]
            linf_violation = linf_pose_penalty(
                optimizable_pose,
                baseline_pose_for_linf,
                budget=linf_pose_budget,
            )
            linf_loss_val = linf_violation.item()
            total_loss = total_loss + linf_pose_weight * linf_violation

        total_loss.backward()
        optimizer.step()

        # --- Argmax constraint: reject steps that flip SegNet argmax ---
        if argmax_constraint:
            with torch.no_grad():
                # Re-forward to check argmax after optimizer step
                check_pose = _project_to_renderer_pose(conditioning)
                check_kwargs = {"pose": check_pose}
                if zoom_warp is not None and batch_pair_indices is not None:
                    check_kwargs["ego_flow"] = zoom_warp(batch_pair_indices, masks_t.shape[1], masks_t.shape[2])
                check_pairs = renderer(masks_t, masks_t1, **check_kwargs)
                check_ft = check_pairs[:, 0]
                check_ft1 = check_pairs[:, 1]
                check_hwc = torch.cat([check_ft, check_ft1], dim=0)
                check_chw = check_hwc.permute(0, 3, 1, 2).contiguous()
                check_seg_in = segnet.preprocess_input(check_chw.unsqueeze(1))
                check_logits = segnet(check_seg_in)
                new_argmax = check_logits.argmax(dim=1)

                flipped_pixels = (new_argmax != ref_argmax).sum().item()
                total_pixels = ref_argmax.numel()
                flip_threshold = int(total_pixels * flip_budget)

                if flipped_pixels > flip_threshold:
                    # Reject: rollback and try with smaller effective step
                    argmax_rejections += 1
                    retry_accepted = False

                    for retry in range(max_retries):
                        # Interpolate: move halfway between pre-step and post-step
                        alpha = 0.5 ** (retry + 1)
                        with torch.no_grad():
                            conditioning.copy_(
                                pre_step_cond + alpha * (conditioning.detach() - pre_step_cond)
                            )
                        # Check argmax again
                        check_pose2 = _project_to_renderer_pose(conditioning)
                        check_kwargs2 = {"pose": check_pose2}
                        if zoom_warp is not None and batch_pair_indices is not None:
                            check_kwargs2["ego_flow"] = zoom_warp(batch_pair_indices, masks_t.shape[1], masks_t.shape[2])
                        check_pairs2 = renderer(masks_t, masks_t1, **check_kwargs2)
                        check_ft2 = check_pairs2[:, 0]
                        check_ft12 = check_pairs2[:, 1]
                        check_hwc2 = torch.cat([check_ft2, check_ft12], dim=0)
                        check_chw2 = check_hwc2.permute(0, 3, 1, 2).contiguous()
                        check_seg_in2 = segnet.preprocess_input(check_chw2.unsqueeze(1))
                        check_logits2 = segnet(check_seg_in2)
                        new_argmax2 = check_logits2.argmax(dim=1)
                        flipped2 = (new_argmax2 != ref_argmax).sum().item()

                        if flipped2 <= flip_threshold:
                            retry_accepted = True
                            if step % log_every == 0:
                                print(f"    [argmax] step {step}: rejected ({flipped_pixels}/{total_pixels} "
                                      f"flipped), accepted at alpha={alpha:.3f} after {retry + 1} retries", flush=True)
                            break

                    if not retry_accepted:
                        # Full rollback
                        with torch.no_grad():
                            conditioning.copy_(pre_step_cond)
                        optimizer.load_state_dict(pre_step_state)
                        if step % log_every == 0:
                            print(f"    [argmax] step {step}: FULL ROLLBACK ({flipped_pixels} px flipped, "
                                  f"all {max_retries} retries failed)", flush=True)

        loss_val = total_loss.item()

        if step == 0:
            metrics["initial_loss"] = loss_val

        if loss_val < best_loss:
            best_loss = loss_val
            best_cond = conditioning.detach().clone()
            best_seg_loss = seg_loss.item()
            best_pose_loss = pose_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if step % log_every == 0 or step == steps - 1:
            grad_norm = conditioning.grad.norm().item() if conditioning.grad is not None else 0.0
            # |dpose| is computed on the optimizable shape (1-DOF in
            # radial-zoom, full pose_dim otherwise) so the L2 norm is
            # well-defined regardless of pose_mode.
            pose_change = (
                conditioning[:, :pose_dim_internal].detach()
                - baseline_pose_for_linf
            ).norm(dim=1).mean().item()
            constraint_str = f" rejections={argmax_rejections}" if argmax_constraint else ""
            kl_str = f" kl={kl_loss_val:.6f}" if kl_distill_weight > 0 else ""
            linf_str = f" linf={linf_loss_val:.6f}" if linf_pose_weight > 0 else ""
            print(f"  step {step:4d}/{steps}: loss={loss_val:.6f} "
                  f"(seg={seg_loss.item():.6f}, pose={pose_loss.item():.6f}{kl_str}{linf_str}) "
                  f"|grad|={grad_norm:.4f} |dpose|={pose_change:.4f}{constraint_str}", flush=True)

        if patience_counter >= early_stop_patience:
            print(f"  Early stop at step {step} (no improvement for {early_stop_patience} steps)", flush=True)
            break

    metrics["steps_run"] = step + 1
    metrics["final_loss"] = best_loss
    metrics["final_seg_loss"] = best_seg_loss
    metrics["final_pose_loss"] = best_pose_loss
    metrics["argmax_rejections"] = argmax_rejections
    if metrics["initial_loss"] > 0:
        metrics["improvement_pct"] = (
            (metrics["initial_loss"] - best_loss) / metrics["initial_loss"] * 100
        )

    return best_cond.cpu(), metrics


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main():
    args = parse_args()

    # Lane RM (Riemannian SE(3)) preconditions — fail loud BEFORE any GPU
    # spend. The SE(3) optimiser requires a 6-DOF pose tensor (each row =
    # one SE(3) element split as (ω, t)) and is incompatible with the
    # LoRA paths (which run their own dedicated Adam loops on a different
    # parameterisation). Per CLAUDE.md "NEVER invent CLI flags" + "fail
    # loud not silent": these constraints are enforced here, not silently
    # bypassed.
    if getattr(args, "optimizer", "adam") == "riemannian-sgd":
        if args.pose_mode != "full-6dof":
            raise SystemExit(
                "FATAL: --optimizer=riemannian-sgd requires "
                f"--pose-mode=full-6dof (got {args.pose_mode!r}). The SE(3) "
                "optimiser interprets each pose row as (ω: so(3), t: ℝ³) "
                "which only makes sense for the 6-DOF parameterisation."
            )
        if args.lora_rank != 0:
            raise SystemExit(
                "FATAL: --optimizer=riemannian-sgd is mutually exclusive "
                f"with --lora-rank > 0 (got {args.lora_rank}). The LoRA path "
                "uses its own Adam optimiser on a base + U @ V "
                "parameterisation that is not on the SE(3) manifold."
            )
        if args.learnable_lora_max_rank != 0:
            raise SystemExit(
                "FATAL: --optimizer=riemannian-sgd is mutually exclusive "
                f"with --learnable-lora-max-rank > 0 (got "
                f"{args.learnable_lora_max_rank}). Lane LR-V2 uses its own "
                "Adam optimiser on a different parameterisation."
            )
        if args.latent_dim != 0:
            raise SystemExit(
                "FATAL: --optimizer=riemannian-sgd is mutually exclusive "
                f"with --latent-dim > 0 (got {args.latent_dim}). The SE(3) "
                "optimiser steps a (B, 6) pose-only tensor; latent codes "
                "would extend the parameter beyond the SE(3) manifold."
            )

    # Preflight: catch integration mismatches before burning GPU time
    from tac.preflight import preflight_check
    preflight_check(
        renderer_path=args.checkpoint,
        masks_path=args.masks,
        poses_path=None,  # poses don't exist yet — we're creating them
    )

    if args.smoke:
        args.n_frames = 20
        args.steps = 100
        args.batch_pairs = 10
        args.log_every = 10
        print("[smoke] Smoke test: 20 frames, 100 steps, 10 pairs/batch", flush=True)

    # Ensure even frame count
    args.n_frames = args.n_frames - (args.n_frames % 2)
    n_pairs = args.n_frames // 2

    device = torch.device(args.device)
    upstream = Path(args.upstream) if args.upstream else UPSTREAM_ROOT
    if upstream is None:
        print("ERROR: Cannot find upstream root. Set --upstream or TAC_UPSTREAM_DIR.", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.output_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.output_dir = str(RESULTS_DIR / f"pose_tto_{ts}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # codex R5-r6 #3: gate AFTER output_dir resolution so the
    # eval_roundtrip_gate.json sidecar lands in the resolved
    # (timestamped) run directory.
    _enforce_eval_roundtrip(args)

    video_path = args.video or str(upstream / "videos" / "0.mkv")
    # Lane M: pose_dim_internal is the OPTIMIZABLE width (1 for
    # radial-zoom, 6 for full-6dof). The renderer always consumes 6-DOF
    # pose; the projection happens inside optimize_poses_batch.
    pose_dim_internal = 1 if args.pose_mode == "radial-zoom" else 6
    cond_dim = pose_dim_internal + args.latent_dim

    print(f"[config] device={device}, n_frames={args.n_frames}, steps={args.steps}", flush=True)
    print(f"[config] lr={args.lr}, batch_pairs={args.batch_pairs}", flush=True)
    print(f"[config] seg_weight={args.seg_weight}, pose_weight={args.pose_weight}", flush=True)
    print(f"[config] hinge_margin={args.hinge_margin}, eval_roundtrip={args.eval_roundtrip}", flush=True)
    print(f"[config] argmax_constraint={args.argmax_constraint}, max_retries={args.max_retries}", flush=True)
    print(f"[config] latent_dim={args.latent_dim} (conditioning dim={cond_dim})", flush=True)
    print(f"[config] kl_distill_weight={args.kl_distill_weight}, "
          f"kl_distill_temperature={args.kl_distill_temperature}", flush=True)
    print(
        f"[config] optimizer={args.optimizer} "
        f"(riemannian_momentum={args.riemannian_momentum} "
        f"if optimizer=riemannian-sgd)",
        flush=True,
    )
    # Lane G V3-V2: --kl-distill-snr-target validation. Mutually exclusive
    # with a non-zero static --kl-distill-weight (the SNR controller's
    # initial weight comes from --kl-distill-weight if set, but the
    # operator must not also try to "lock" the weight statically — that
    # would silently disable the controller). Build the controller here
    # so any construction error fails-loud BEFORE the GPU-loaded
    # scorers/renderer are touched.
    kl_distill_snr_controller = None
    if getattr(args, "kl_distill_snr_target", None) is not None:
        # Operator opted into Lagrangian SNR control. The static weight
        # is now an INITIAL value, not the per-step value.
        from tac.lagrangian_kl_weight import LearnableKLWeight
        _initial_w = args.kl_distill_weight if args.kl_distill_weight > 0 else 0.002
        kl_distill_snr_controller = LearnableKLWeight(
            snr_target=float(args.kl_distill_snr_target),
            initial_weight=float(_initial_w),
            eta=float(args.kl_distill_snr_eta),
        )
        print(
            f"[lane-g-v3-v2] LAGRANGIAN KL WEIGHT ACTIVE: "
            f"snr_target={args.kl_distill_snr_target} "
            f"initial_weight={_initial_w} eta={args.kl_distill_snr_eta} "
            f"(static --kl-distill-weight is now the INITIAL value; "
            f"the controller updates it after each step toward the "
            f"observed-SNR=target fixed point per Boyd & Vandenberghe "
            f"§5.4 + Kivinen & Warmuth 1997)",
            flush=True,
        )
    # Lane M: explicit banner so the operator sees the pose-mode at-a-
    # glance (per CLAUDE.md "no wasted resources" — silent mode flips
    # are how 6h GPU runs go off-target).
    if args.pose_mode == "radial-zoom":
        print(f"[pose-mode] using radial-zoom (1-DOF, projected to 6-DOF before render)", flush=True)
    else:
        print(f"[pose-mode] using {args.pose_mode}", flush=True)
    # Lane N: Fridrich L∞ banner when active.
    if args.linf_pose_weight > 0:
        print(f"[linf] Fridrich L∞ pose penalty active: weight={args.linf_pose_weight}, "
              f"budget={args.linf_pose_budget}", flush=True)
    # Lane PS: parse the per-class SegNet weights CSV (fail loud at parse
    # time per CLAUDE.md "fail loud, not silent" — a typo in the CSV must
    # NOT silently fall back to uniform weighting). Sentinel: empty string
    # → None (disabled, byte-identical to baseline).
    from tac.losses import parse_class_weights_csv
    segnet_class_weights_t = parse_class_weights_csv(
        args.segnet_class_weights or None, num_classes=5,
    )
    if segnet_class_weights_t is not None:
        print(
            f"[lane-ps] SegNet per-class weights ACTIVE: "
            f"{segnet_class_weights_t.tolist()} "
            f"(applies to auxiliary KL distill loss when "
            f"--kl-distill-weight > 0)",
            flush=True,
        )
        if args.kl_distill_weight <= 0:
            print(
                "[lane-ps] WARNING: --segnet-class-weights supplied but "
                "--kl-distill-weight <= 0 — Lane PS only takes effect "
                "via the auxiliary KL distill path. The weights are "
                "currently a NO-OP. Set --kl-distill-weight > 0 (e.g., "
                "1.0 for Quantizr-style distillation) to activate.",
                flush=True,
            )

    # Lane PS-V2 (2026-04-27): wrap segnet_class_weights_t in a learnable
    # parameter group when --learnable-segnet-class-weights is set. The
    # static CSV becomes the warm-start; the LearnableClassWeights
    # module's softmax forward provides the differentiable per-class
    # tensor passed to the loss helpers.
    learnable_segnet_class_weights = None
    if getattr(args, "learnable_segnet_class_weights", False):
        from tac.learnable_class_weights import LearnableClassWeights
        learnable_segnet_class_weights = LearnableClassWeights(
            num_classes=5, warm_start=segnet_class_weights_t,
        ).to(device)
        # Override the static tensor: the loss helpers will receive the
        # learnable weights' forward output instead of the parsed CSV.
        segnet_class_weights_t = learnable_segnet_class_weights().detach()
        print(
            f"[lane-ps-v2] LEARNABLE SegNet per-class weights ACTIVE: "
            f"warm-start={segnet_class_weights_t.tolist()} "
            f"(lr={args.learnable_segnet_class_weights_lr}, "
            f"var_lambda={args.learnable_segnet_class_weights_var_lambda})",
            flush=True,
        )
        if args.kl_distill_weight <= 0:
            print(
                "[lane-ps-v2] WARNING: --learnable-segnet-class-weights "
                "supplied but --kl-distill-weight <= 0 — Lane PS only "
                "takes effect via the auxiliary KL distill path. The "
                "learnable weights will train but apply NO loss "
                "contribution. Set --kl-distill-weight > 0 to activate.",
                flush=True,
            )
    print(f"[config] checkpoint={args.checkpoint}", flush=True)
    print(f"[config] output_dir={output_dir}", flush=True)

    t_total = time.monotonic()

    # ── Step 1: Load scorers ─────────────────────────────────────────────
    print("\n[1/6] Loading differentiable scorers...", flush=True)
    t0 = time.monotonic()
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(upstream, device=str(device))
    print(f"[1/6] Scorers loaded in {time.monotonic() - t0:.1f}s", flush=True)

    # ── Step 2: Load renderer ────────────────────────────────────────────
    print("\n[2/6] Loading renderer...", flush=True)
    t0 = time.monotonic()
    renderer = load_renderer(args.checkpoint, device)
    print(f"[2/6] Renderer loaded in {time.monotonic() - t0:.1f}s", flush=True)

    # Load zoom warp for use_zoom_flow models
    zoom_warp = None
    if hasattr(renderer, 'use_zoom_flow') and renderer.use_zoom_flow:
        from tac.radial_zoom import RadialZoomWarp
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.suffix == ".pt":
            ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "zoom_warp_state_dict" in ckpt_data:
                n_p = ckpt_data["zoom_warp_state_dict"]["zoom_scalars"].shape[0]
                zoom_warp = RadialZoomWarp(n_pairs=n_p).to(device)
                zoom_warp.load_state_dict(ckpt_data["zoom_warp_state_dict"])
                print(f"  Loaded zoom scalars ({n_p} pairs) from checkpoint", flush=True)
        if zoom_warp is None:
            zoom_path = ckpt_path.parent / "zoom_scalars.bin"
            if zoom_path.exists():
                from tac.radial_zoom import load_zoom_scalars
                zoom_warp = load_zoom_scalars(zoom_path, device=str(device))
                print(f"  Loaded zoom scalars from {zoom_path.name}", flush=True)
        if zoom_warp is None:
            print(f"  WARNING: use_zoom_flow but no zoom scalars found. Using identity zoom.", flush=True)

    # Verify pose_dim compatibility
    renderer_pose_dim = getattr(renderer, "pose_dim", 0)
    if renderer_pose_dim == 0:
        print("\n" + "=" * 70, flush=True)
        print("FATAL: Renderer has pose_dim=0. FiLM layers are not present.", flush=True)
        print("Pose-space TTO requires a renderer trained with --pose-dim 6.", flush=True)
        print("Aborting.", flush=True)
        print("=" * 70, flush=True)
        sys.exit(1)

    # ── Step 3: Decode GT video + extract masks + pose targets ──────────
    print(f"\n[3/6] Decoding GT video ({args.n_frames} frames)...", flush=True)
    t0 = time.monotonic()
    from tac.data import load_gt_video
    gt_frames = load_gt_video(video_path, n_frames=args.n_frames)
    args.n_frames = len(gt_frames)
    n_pairs = args.n_frames // 2
    print(f"[3/6] Decoded {args.n_frames} frames in {time.monotonic() - t0:.1f}s", flush=True)

    print("\n[4/6] Loading/extracting masks and pose targets...", flush=True)
    t0 = time.monotonic()
    from tac.scorer import extract_gt_masks, extract_gt_pose_targets

    if args.masks:
        # Use pre-decoded masks (MUST match the archive masks for correct optimization)
        masks_path = Path(args.masks)
        if masks_path.suffix == ".pt":
            gt_masks = torch.load(str(masks_path), weights_only=True).long()
        elif masks_path.suffix in (".mkv", ".mp4"):
            # Decode from AV1 video (same path as inflate_renderer.py)
            import subprocess, numpy as np
            cmd = ["ffmpeg", "-v", "quiet", "-i", str(masks_path),
                   "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
            proc = subprocess.run(cmd, capture_output=True)
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0", str(masks_path)],
                capture_output=True, text=True,
            )
            w, h = map(int, probe.stdout.strip().split(","))
            pixels = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w)
            scale = 255 // 4
            masks_np = np.clip(np.round(pixels.astype(np.float32) / scale).astype(np.int64), 0, 4)
            gt_masks = torch.from_numpy(masks_np)
        else:
            raise ValueError(f"Unknown mask format: {masks_path.suffix}")
        print(f"  Loaded archive masks from {args.masks}: {gt_masks.shape}", flush=True)
        # Upsample to 384x512 if needed (same logic as inflate_renderer.py)
        if gt_masks.shape[1] < 384 or gt_masks.shape[2] < 512:
            import torch.nn.functional as F
            gt_masks = F.interpolate(
                gt_masks.float().unsqueeze(1), size=(384, 512), mode="nearest"
            ).squeeze(1).long()
            print(f"  Upsampled to {gt_masks.shape}", flush=True)
    else:
        # R39 fix: dead branch — argparse required=True guarantees args.masks
        # is set, so the `if args.masks:` above is always True. Keeping a
        # defensive raise here in case argparse ever gets relaxed.
        raise RuntimeError(
            "--masks is required (argparse required=True). If you reached "
            "this branch, the requirement was relaxed — restore it."
        )

    if args.gt_pose_targets and Path(args.gt_pose_targets).exists():
        pose_targets = torch.load(args.gt_pose_targets, map_location="cpu", weights_only=True).float()
        print(f"  Loaded precomputed pose targets from {args.gt_pose_targets}: {pose_targets.shape}", flush=True)
    else:
        pose_targets = extract_gt_pose_targets(gt_frames, posenet, device)
        # Save for reuse on future runs
        cache_path = Path(args.output_dir) / "gt_pose_targets.pt"
        torch.save(pose_targets, cache_path)
        print(f"  Computed pose targets and cached to {cache_path}", flush=True)
    print(f"[4/6] Masks: {gt_masks.shape}, Poses: {pose_targets.shape} in {time.monotonic() - t0:.1f}s", flush=True)

    # ── Step 4: Load or extract initial GT poses ────────────────────────
    # Lane OS-A: --seed-poses-path takes precedence over --gt-poses-path. The
    # seed file is produced by experiments/seed_poses_from_openpilot.py from
    # openpilot supercombo (or the lane_mark_pose fallback). It is a
    # PoseNet-distribution-calibrated warm start, expected to converge faster
    # than GT/baseline pose warm-start. Memory: project_openpilot_seeding_demo.
    print("\n[5/6] Loading initial pose vectors...", flush=True)
    if args.seed_poses_path and Path(args.seed_poses_path).exists():
        init_poses = torch.load(args.seed_poses_path, map_location="cpu", weights_only=True).float()
        print(f"  [Lane OS-A] Loaded SEED poses from {args.seed_poses_path}: {init_poses.shape}", flush=True)
    elif args.gt_poses_path and Path(args.gt_poses_path).exists():
        init_poses = torch.load(args.gt_poses_path, map_location="cpu", weights_only=True).float()
        print(f"  Loaded GT poses from {args.gt_poses_path}: {init_poses.shape}", flush=True)
    else:
        # Use pose_targets as warm start (these are PoseNet outputs on GT frames)
        init_poses = pose_targets.clone()
        print(f"  Using PoseNet GT targets as warm start: {init_poses.shape}", flush=True)
    init_poses = init_poses[:n_pairs]

    # ── Step 5a (optional): Global embedding optimization ────────────────
    if args.optimize_embedding:
        print(f"\n[5.5/6] Optimizing shared class embedding "
              f"({args.embedding_epochs} epochs, lr={args.embedding_lr})...", flush=True)
        t0 = time.monotonic()

        # Unfreeze ONLY the shared embedding
        shared_emb = renderer.renderer.embedding
        original_emb = shared_emb.weight.data.clone()
        shared_emb.weight.requires_grad_(True)

        emb_optimizer = torch.optim.Adam([shared_emb.weight], lr=args.embedding_lr)
        best_emb_loss = float("inf")
        best_emb_weights = shared_emb.weight.data.clone()
        n_emb_batches = math.ceil(n_pairs / args.batch_pairs)

        for emb_epoch in range(args.embedding_epochs):
            epoch_loss = 0.0
            epoch_count = 0
            for bi in range(n_emb_batches):
                ps = bi * args.batch_pairs
                pe = min(ps + args.batch_pairs, n_pairs)
                fs, fe = 2 * ps, 2 * pe

                mt = gt_masks[fs:fe:2].to(device)
                mt1 = gt_masks[fs + 1:fe + 1:2].to(device)
                gm = gt_masks[fs:fe].to(device)
                pt = pose_targets[ps:pe]

                bp = init_poses[ps:pe].to(device) if renderer_pose_dim > 0 else None

                emb_optimizer.zero_grad()
                pairs = renderer(mt, mt1, pose=bp)
                ft, ft1 = pairs[:, 0], pairs[:, 1]
                frames_hwc = torch.cat([ft, ft1], dim=0)
                frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()

                if args.eval_roundtrip:
                    frames_chw = simulate_eval_roundtrip(frames_chw, noise_std=args.posetto_noise_std)

                seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
                seg_logits = segnet(seg_in)
                s_loss = segnet_hinge_loss(seg_logits, gm, margin=args.hinge_margin)

                pairs_chw = torch.stack([
                    ft.permute(0, 3, 1, 2), ft1.permute(0, 3, 1, 2)
                ], dim=1)
                if args.eval_roundtrip:
                    Bp, Tp, Cp, Hp, Wp = pairs_chw.shape
                    flat = pairs_chw.reshape(Bp * Tp, Cp, Hp, Wp)
                    flat = simulate_eval_roundtrip(flat, noise_std=args.posetto_noise_std)
                    pairs_chw = flat.reshape(Bp, Tp, Cp, Hp, Wp)
                pose_in = posenet.preprocess_input(pairs_chw)
                pose_out = posenet(pose_in)["pose"][..., :6]
                p_loss = F.mse_loss(pose_out, pt.to(device))

                total = args.seg_weight * s_loss + args.pose_weight * p_loss
                total.backward()
                emb_optimizer.step()

                epoch_loss += total.item()
                epoch_count += 1

            avg = epoch_loss / max(epoch_count, 1)
            emb_delta = (shared_emb.weight.data - original_emb.to(device)).norm().item()
            print(f"    Embedding epoch {emb_epoch}: avg_loss={avg:.6f}, |dw|={emb_delta:.4f}", flush=True)

            if avg < best_emb_loss:
                best_emb_loss = avg
                best_emb_weights = shared_emb.weight.data.clone()

        # Restore best and re-freeze
        shared_emb.weight.data = best_emb_weights
        shared_emb.weight.requires_grad_(False)

        emb_delta_final = (best_emb_weights - original_emb.to(device)).norm().item()
        print(f"    Embedding optimization done in {time.monotonic() - t0:.1f}s, "
              f"|total_delta|={emb_delta_final:.4f}", flush=True)

        # Save optimized embedding separately (120 bytes fp32)
        torch.save(best_emb_weights.cpu(), output_dir / "optimized_embedding.pt")
        print(f"    Saved optimized_embedding.pt ({best_emb_weights.numel() * 4} bytes)", flush=True)

    # ── Lane LR-V2 mutual-exclusion gate ─────────────────────────────────
    # Picking BOTH frozen-rank (V1) and learnable-rank (V2) is ambiguous:
    # the V2 pass would supersede the V1 pass and the operator's intent
    # cannot be inferred. Hard-fail rather than silently dropping one.
    if args.lora_rank > 0 and args.learnable_lora_max_rank > 0:
        raise SystemExit(
            "FATAL: --lora-rank and --learnable-lora-max-rank are mutually "
            "exclusive (V1 frozen-rank vs V2 learnable-rank). Pass exactly "
            "one, or neither (which falls through to the full-rank baseline)."
        )

    # ── Step 5b-LoRA-V2: Lane LR-V2 — LEARNABLE-rank pose adaptation ──────
    # When --learnable-lora-max-rank > 0, the LoRA factorisation begins at
    # rank=max_rank with a per-rank gate (sigmoid of a learnable logit). Gates
    # co-train with U + V; ranks whose final gate < --lora-prune-threshold
    # are pruned at serialisation. This lets the optimiser DECIDE the
    # effective rank instead of the operator picking it offline.
    if args.learnable_lora_max_rank > 0:
        from tac.lora_pose_v2 import (
            LearnableRankLoRAPose,
            save_lora_v2_poses,
        )
        from tac.fridrich import linf_pose_penalty as _linf_pose_penalty_v2

        lora_steps_v2 = args.lora_steps if args.lora_steps > 0 else args.steps
        print(
            f"\n[6/6] Lane LR-V2: optimising LearnableRank LoRA "
            f"(max_rank={args.learnable_lora_max_rank}, "
            f"prune_threshold={args.lora_prune_threshold}) over {n_pairs} "
            f"pairs for {lora_steps_v2} steps "
            f"(micro-batch={args.batch_pairs})...",
            flush=True,
        )
        if args.pose_mode != "full-6dof":
            raise SystemExit(
                f"FATAL: --learnable-lora-max-rank > 0 is not compatible "
                f"with --pose-mode={args.pose_mode!r}. Lane LR-V2 already "
                f"provides a low-rank parameterisation of the full 6-DOF "
                f"pose; pick one of: --learnable-lora-max-rank N (Lane LR-V2) "
                f"OR --pose-mode radial-zoom (Lane M)."
            )

        base_poses_v2 = init_poses[:n_pairs].to(device)
        lora_v2 = LearnableRankLoRAPose(
            base=base_poses_v2,
            max_rank=args.learnable_lora_max_rank,
            init_gate_logit=args.lora_init_gate_logit,
        ).to(device)
        print(
            f"  [Lane LR-V2] params: U={tuple(lora_v2.U.shape)} + "
            f"V={tuple(lora_v2.V.shape)} + gate={tuple(lora_v2.logit_gate.shape)} "
            f"= {lora_v2.trainable_params} trainable scalars",
            flush=True,
        )
        print(
            f"  [Lane LR-V2] worst-case archive bytes (no pruning, fp16): "
            f"{lora_v2.archive_bytes_fp16():,}; baseline (N, 6) fp16 = "
            f"{n_pairs * 6 * 2:,}",
            flush=True,
        )

        # Joint optimiser over U, V, and gate logits. Same lr — the gate
        # gradient magnitudes are similar to U/V gradient magnitudes
        # (sigmoid * U.norm() ~ U gradient scale).
        lora_v2_optimizer = torch.optim.Adam(
            [lora_v2.U, lora_v2.V, lora_v2.logit_gate], lr=args.lr,
        )
        eval_roundtrip_v2 = bool(args.eval_roundtrip)
        rng_v2 = torch.Generator(device="cpu").manual_seed(0)
        lora_v2_metrics = {
            "max_rank": args.learnable_lora_max_rank,
            "prune_threshold": args.lora_prune_threshold,
            "init_gate_logit": args.lora_init_gate_logit,
            "lora_steps": lora_steps_v2,
            "trainable_params": lora_v2.trainable_params,
            "archive_bytes_fp16_worst_case": lora_v2.archive_bytes_fp16(),
            "step_losses": [],
            "best_loss": float("inf"),
            "gate_history": [],
        }
        best_state_v2 = (
            lora_v2.U.detach().clone(),
            lora_v2.V.detach().clone(),
            lora_v2.logit_gate.detach().clone(),
        )
        t_lora_v2 = time.monotonic()
        for step in range(lora_steps_v2):
            idx = torch.randperm(n_pairs, generator=rng_v2)[: args.batch_pairs]
            idx_sorted, _ = idx.sort()
            pair_idx = idx_sorted.to(device)
            frame_idx_t = (2 * pair_idx).cpu()
            frame_idx_t1 = (2 * pair_idx + 1).cpu()

            mt = gt_masks[frame_idx_t].to(device)
            mt1 = gt_masks[frame_idx_t1].to(device)
            gm = torch.stack([
                gt_masks[frame_idx_t],
                gt_masks[frame_idx_t1],
            ], dim=1).reshape(
                -1, gt_masks.shape[-2], gt_masks.shape[-1],
            ).to(device)
            pt = pose_targets[idx_sorted].to(device)

            lora_v2_optimizer.zero_grad()
            full_poses_v2 = lora_v2()                  # (N, 6)
            batch_pose_v2 = full_poses_v2[pair_idx]    # (B, 6)

            fwd_kwargs_v2 = {"pose": batch_pose_v2}
            if zoom_warp is not None:
                fwd_kwargs_v2["ego_flow"] = zoom_warp(
                    pair_idx, mt.shape[1], mt.shape[2],
                )
            pairs = renderer(mt, mt1, **fwd_kwargs_v2)
            ft, ft1 = pairs[:, 0], pairs[:, 1]
            frames_hwc = torch.cat([ft, ft1], dim=0)
            frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()
            if eval_roundtrip_v2:
                frames_chw = simulate_eval_roundtrip(
                    frames_chw, noise_std=args.posetto_noise_std,
                )

            seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
            seg_logits = segnet(seg_in)
            seg_loss = segnet_hinge_loss(
                seg_logits, gm, margin=args.hinge_margin,
            )

            pairs_chw = torch.stack(
                [ft.permute(0, 3, 1, 2), ft1.permute(0, 3, 1, 2)], dim=1,
            )
            if eval_roundtrip_v2:
                Bp, Tp, Cp, Hp, Wp = pairs_chw.shape
                flat = pairs_chw.reshape(Bp * Tp, Cp, Hp, Wp)
                flat = simulate_eval_roundtrip(
                    flat, noise_std=args.posetto_noise_std,
                )
                pairs_chw = flat.reshape(Bp, Tp, Cp, Hp, Wp)
            pose_in = posenet.preprocess_input(pairs_chw)
            pose_out = posenet(pose_in)["pose"][..., :6]
            pose_loss = F.mse_loss(pose_out, pt)

            total = (
                args.seg_weight * seg_loss + args.pose_weight * pose_loss
            )

            if args.linf_pose_weight > 0:
                penalty = _linf_pose_penalty_v2(
                    full_poses_v2, lora_v2.base.detach(),
                    budget=args.linf_pose_budget,
                )
                total = total + args.linf_pose_weight * penalty

            total.backward()
            lora_v2_optimizer.step()

            loss_val = total.item()
            lora_v2_metrics["step_losses"].append(loss_val)
            if loss_val < lora_v2_metrics["best_loss"]:
                lora_v2_metrics["best_loss"] = loss_val
                best_state_v2 = (
                    lora_v2.U.detach().clone(),
                    lora_v2.V.detach().clone(),
                    lora_v2.logit_gate.detach().clone(),
                )

            if step % args.log_every == 0 or step == lora_steps_v2 - 1:
                gate_now = lora_v2.gate.detach().cpu().tolist()
                lora_v2_metrics["gate_history"].append(
                    {"step": step, "gate": gate_now}
                )
                u_norm = lora_v2.U.detach().norm().item()
                v_norm = lora_v2.V.detach().norm().item()
                print(
                    f"  [Lane LR-V2] step {step:4d}/{lora_steps_v2}: "
                    f"loss={loss_val:.6f} (seg={seg_loss.item():.6f}, "
                    f"pose={pose_loss.item():.6f}) "
                    f"|U|={u_norm:.4f} |V|={v_norm:.4f} "
                    f"gate=[{', '.join(f'{g:.3f}' for g in gate_now)}]",
                    flush=True,
                )

        # Restore best state.
        with torch.no_grad():
            lora_v2.U.copy_(best_state_v2[0])
            lora_v2.V.copy_(best_state_v2[1])
            lora_v2.logit_gate.copy_(best_state_v2[2])

        # Prune + serialise. Surface the kept rank up front so an operator
        # tailing the log knows immediately how aggressive the pruning was.
        kept_indices_final = lora_v2.kept_indices(
            prune_threshold=args.lora_prune_threshold,
        )
        print(
            f"  [Lane LR-V2] pruning: kept {len(kept_indices_final)} of "
            f"{lora_v2.max_rank} ranks "
            f"(indices={kept_indices_final}, "
            f"threshold={args.lora_prune_threshold})",
            flush=True,
        )

        with torch.inference_mode():
            optimized_poses = lora_v2().detach().cpu()
        save_size_v2 = save_lora_v2_poses(
            lora_v2,
            output_dir / "optimized_poses.pt",
            prune_threshold=args.lora_prune_threshold,
        )
        import json as _json
        (output_dir / "optimized_poses.meta").write_text(_json.dumps({
            "n_pairs_complete": int(n_pairs),
            "n_pairs_total": int(n_pairs),
            "is_final": True,
            "pose_dim": 6,
            "pose_mode": "full-6dof",
            "lora_v2_max_rank": int(args.learnable_lora_max_rank),
            "lora_v2_kept_rank": len(kept_indices_final),
            "lora_v2_kept_indices": kept_indices_final,
            "lora_v2_prune_threshold": args.lora_prune_threshold,
            "lora_archive_bytes": save_size_v2,
        }))

        lora_v2_dt = time.monotonic() - t_lora_v2
        print(
            f"  [Lane LR-V2] training done in {lora_v2_dt:.1f}s, "
            f"best_loss={lora_v2_metrics['best_loss']:.6f}, "
            f"saved optimized_poses.pt = {save_size_v2:,} bytes "
            f"(vs (N, 6) fp16 baseline = {n_pairs * 6 * 2:,} bytes)",
            flush=True,
        )

        from tac.scorer import compute_proxy_score
        gt_pose_frames = _generate_frames(
            renderer, gt_masks, init_poses[:n_pairs], device,
            args.batch_pairs, zoom_warp=zoom_warp,
        )
        gt_score = compute_proxy_score(
            gt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
        )
        opt_pose_frames = _generate_frames(
            renderer, gt_masks, optimized_poses, device,
            args.batch_pairs, zoom_warp=zoom_warp,
        )
        opt_score = compute_proxy_score(
            opt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
        )

        total_time = time.monotonic() - t_total
        summary = {
            "config": vars(args),
            "gt_score": gt_score,
            "optimized_score": opt_score,
            "delta_score": opt_score["score"] - gt_score["score"],
            "total_time_s": total_time,
            "lora_v2_metrics": lora_v2_metrics,
            "lora_v2_kept_indices": kept_indices_final,
            "n_pairs": n_pairs,
            "archive_bytes_fp16_baseline": n_pairs * 6 * 2,
            "archive_bytes_lora_v2": save_size_v2,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"\n  GT poses score = {gt_score['score']:.4f}; "
            f"LoRA-V2(kept_rank={len(kept_indices_final)}) score = "
            f"{opt_score['score']:.4f}; "
            f"delta = {opt_score['score'] - gt_score['score']:+.4f}",
            flush=True,
        )
        print(f"  Total time: {total_time:.1f}s", flush=True)
        print(f"  Results saved to: {output_dir}", flush=True)
        return

    # ── Step 5b-LoRA: Lane LR — global low-rank pose adaptation ─────────
    # When --lora-rank > 0, we replace the per-batch full-rank loop with a
    # SINGLE global optimisation over (U, V) so the basis V is shared
    # across all pairs (the rank-1 hypothesis from
    # project_posenet_rank1_discovery — V is the radial-zoom direction,
    # U is the per-pair scalar coefficient). Each step samples a random
    # `--batch-pairs` micro-batch for VRAM fit; (U, V) get gradients only
    # for the sampled rows of U but the full V on every step.
    if args.lora_rank > 0:
        from tac.lora_pose import LoRAPose, save_lora_poses
        from tac.losses import kl_distill_segnet_only
        from tac.fridrich import linf_pose_penalty as _linf_pose_penalty

        lora_steps = args.lora_steps if args.lora_steps > 0 else args.steps
        print(f"\n[6/6] Lane LR: optimising LoRA(rank={args.lora_rank}) "
              f"over {n_pairs} pairs for {lora_steps} steps "
              f"(micro-batch={args.batch_pairs})...", flush=True)
        if args.pose_mode != "full-6dof":
            # Lane LR factorises the 6-DOF pose; combining it with the
            # radial-zoom (1-DOF) projection would double-restrict the
            # parameterisation in incompatible ways. Hard-fail rather
            # than silently produce poses that ignore one of the two
            # restrictions.
            raise SystemExit(
                f"FATAL: --lora-rank > 0 is not compatible with "
                f"--pose-mode={args.pose_mode!r}. Lane LR already provides "
                f"a low-rank parameterisation of the full 6-DOF pose; pick "
                f"one of: --lora-rank N (Lane LR) OR --pose-mode radial-zoom "
                f"(Lane M)."
            )

        base_poses = init_poses[:n_pairs].to(device)
        lora = LoRAPose(base=base_poses, rank=args.lora_rank).to(device)
        # Preview the rate budget so the operator sees the saving up front.
        print(
            f"  [Lane LR] LoRA params: U={tuple(lora.U.shape)} + "
            f"V={tuple(lora.V.shape)} = {lora.trainable_params} trainable "
            f"scalars", flush=True,
        )
        print(
            f"  [Lane LR] predicted archive bytes (fp16, U+V+base): "
            f"{lora.archive_bytes_fp16():,}; baseline (N, 6) fp16 = "
            f"{n_pairs * 6 * 2:,}", flush=True,
        )

        lora_optimizer = torch.optim.Adam(
            [lora.U, lora.V], lr=args.lr,
        )
        # eval-roundtrip is non-negotiable per CLAUDE.md and already
        # enforced by _enforce_eval_roundtrip(); we still read the flag so
        # local pyhamic disabling (TAC_ALLOW_NO_ROUNDTRIP=1) is honoured
        # symmetrically with the full-rank loop.
        eval_roundtrip = bool(args.eval_roundtrip)
        rng = torch.Generator(device="cpu").manual_seed(0)
        lora_metrics = {
            "lora_rank": args.lora_rank,
            "lora_steps": lora_steps,
            "trainable_params": lora.trainable_params,
            "archive_bytes_fp16": lora.archive_bytes_fp16(),
            "step_losses": [],
            "best_loss": float("inf"),
        }
        best_state = (lora.U.detach().clone(), lora.V.detach().clone())
        t_lora = time.monotonic()
        for step in range(lora_steps):
            # Random sub-batch of pair indices for this step.
            idx = torch.randperm(n_pairs, generator=rng)[: args.batch_pairs]
            idx_sorted, _ = idx.sort()
            pair_idx = idx_sorted.to(device)
            frame_idx_t = (2 * pair_idx).cpu()
            frame_idx_t1 = (2 * pair_idx + 1).cpu()

            mt = gt_masks[frame_idx_t].to(device)
            mt1 = gt_masks[frame_idx_t1].to(device)
            # Interleave for the SegNet 2*B contract (t, t1, t, t1, …).
            gm = torch.stack([
                gt_masks[frame_idx_t],
                gt_masks[frame_idx_t1],
            ], dim=1).reshape(-1, gt_masks.shape[-2], gt_masks.shape[-1]).to(device)
            pt = pose_targets[idx_sorted].to(device)

            lora_optimizer.zero_grad()
            full_poses = lora()  # (N, 6) materialised; gradient flows into U + V
            batch_pose = full_poses[pair_idx]  # (B, 6)

            fwd_kwargs = {"pose": batch_pose}
            if zoom_warp is not None:
                fwd_kwargs["ego_flow"] = zoom_warp(pair_idx, mt.shape[1], mt.shape[2])
            pairs = renderer(mt, mt1, **fwd_kwargs)
            ft, ft1 = pairs[:, 0], pairs[:, 1]
            frames_hwc = torch.cat([ft, ft1], dim=0)
            frames_chw = frames_hwc.permute(0, 3, 1, 2).contiguous()
            if eval_roundtrip:
                frames_chw = simulate_eval_roundtrip(
                    frames_chw, noise_std=args.posetto_noise_std,
                )

            seg_in = segnet.preprocess_input(frames_chw.unsqueeze(1))
            seg_logits = segnet(seg_in)
            seg_loss = segnet_hinge_loss(seg_logits, gm, margin=args.hinge_margin)

            pairs_chw = torch.stack(
                [ft.permute(0, 3, 1, 2), ft1.permute(0, 3, 1, 2)], dim=1,
            )
            if eval_roundtrip:
                Bp, Tp, Cp, Hp, Wp = pairs_chw.shape
                flat = pairs_chw.reshape(Bp * Tp, Cp, Hp, Wp)
                flat = simulate_eval_roundtrip(flat, noise_std=args.posetto_noise_std)
                pairs_chw = flat.reshape(Bp, Tp, Cp, Hp, Wp)
            pose_in = posenet.preprocess_input(pairs_chw)
            pose_out = posenet(pose_in)["pose"][..., :6]
            pose_loss = F.mse_loss(pose_out, pt)

            total = args.seg_weight * seg_loss + args.pose_weight * pose_loss

            # Optional Lane N L∞ penalty on the LoRA delta from base.
            if args.linf_pose_weight > 0:
                lora_delta = (full_poses - lora.base).detach()  # (N, 6)
                # Use the same helper as the full-rank path; baseline
                # for the delta is the LoRA base (warm start). Computed
                # over the FULL (N, 6) tensor so the penalty sees every
                # pair every step (LoRA's V is global).
                penalty = _linf_pose_penalty(
                    full_poses, lora.base.detach(),
                    budget=args.linf_pose_budget,
                )
                total = total + args.linf_pose_weight * penalty

            total.backward()
            lora_optimizer.step()

            loss_val = total.item()
            lora_metrics["step_losses"].append(loss_val)
            if loss_val < lora_metrics["best_loss"]:
                lora_metrics["best_loss"] = loss_val
                best_state = (lora.U.detach().clone(), lora.V.detach().clone())

            if step % args.log_every == 0 or step == lora_steps - 1:
                u_norm = lora.U.detach().norm().item()
                v_norm = lora.V.detach().norm().item()
                print(
                    f"  [Lane LR] step {step:4d}/{lora_steps}: "
                    f"loss={loss_val:.6f} (seg={seg_loss.item():.6f}, "
                    f"pose={pose_loss.item():.6f}) "
                    f"|U|={u_norm:.4f} |V|={v_norm:.4f}",
                    flush=True,
                )

        # Restore best state.
        with torch.no_grad():
            lora.U.copy_(best_state[0])
            lora.V.copy_(best_state[1])

        # Materialise final poses and persist BOTH the LoRA dict (.pt) and
        # the equivalent (N, 6) tensor (.pt with same name? No — keep
        # canonical: the .pt is the LoRA dict, downstream loaders detect
        # the format sentinel transparently). The .bin is NOT emitted in
        # LoRA mode because the .bin format is raw fp16 (N, 6) which
        # would defeat the rate saving — readers should consume the .pt.
        with torch.inference_mode():
            optimized_poses = lora().detach().cpu()
        save_size = save_lora_poses(lora, output_dir / "optimized_poses.pt")
        # Sentinel-friendly meta sidecar so external watchdogs can see the
        # LoRA mode at a glance without unpickling.
        import json as _json
        (output_dir / "optimized_poses.meta").write_text(_json.dumps({
            "n_pairs_complete": int(n_pairs),
            "n_pairs_total": int(n_pairs),
            "is_final": True,
            "pose_dim": 6,
            "pose_mode": "full-6dof",
            "lora_rank": args.lora_rank,
            "lora_archive_bytes": save_size,
        }))

        lora_dt = time.monotonic() - t_lora
        print(
            f"  [Lane LR] LoRA training done in {lora_dt:.1f}s, "
            f"best_loss={lora_metrics['best_loss']:.6f}, "
            f"saved optimized_poses.pt = {save_size:,} bytes "
            f"(vs (N, 6) fp16 baseline = {n_pairs * 6 * 2:,} bytes)",
            flush=True,
        )

        # Compute proxy scores for parity with the full-rank summary.
        from tac.scorer import compute_proxy_score
        gt_pose_frames = _generate_frames(
            renderer, gt_masks, init_poses[:n_pairs], device,
            args.batch_pairs, zoom_warp=zoom_warp,
        )
        gt_score = compute_proxy_score(
            gt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
        )
        opt_pose_frames = _generate_frames(
            renderer, gt_masks, optimized_poses, device,
            args.batch_pairs, zoom_warp=zoom_warp,
        )
        opt_score = compute_proxy_score(
            opt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
        )

        # Summary in the same shape as the full-rank path so downstream
        # reporters don't branch on lora_rank.
        total_time = time.monotonic() - t_total
        summary = {
            "config": vars(args),
            "gt_score": gt_score,
            "optimized_score": opt_score,
            "delta_score": opt_score["score"] - gt_score["score"],
            "total_time_s": total_time,
            "lora_metrics": lora_metrics,
            "n_pairs": n_pairs,
            "archive_bytes_fp16_baseline": n_pairs * 6 * 2,
            "archive_bytes_lora": save_size,
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(
            f"\n  GT poses score = {gt_score['score']:.4f}; "
            f"LoRA(rank={args.lora_rank}) score = {opt_score['score']:.4f}; "
            f"delta = {opt_score['score'] - gt_score['score']:+.4f}",
            flush=True,
        )
        print(f"  Total time: {total_time:.1f}s", flush=True)
        print(f"  Results saved to: {output_dir}", flush=True)
        return

    # ── Step 5b: Batched pose optimization ────────────────────────────────
    print(f"\n[6/6] Optimizing {n_pairs} pose vectors in batches of {args.batch_pairs}...", flush=True)
    n_batches = math.ceil(n_pairs / args.batch_pairs)

    all_optimized = torch.zeros(n_pairs, cond_dim)
    all_metrics = []
    total_steps = 0

    for batch_idx in range(n_batches):
        pair_start = batch_idx * args.batch_pairs
        pair_end = min(pair_start + args.batch_pairs, n_pairs)
        frame_start = 2 * pair_start
        frame_end = 2 * pair_end
        n_batch_pairs = pair_end - pair_start

        print(f"\n--- Batch {batch_idx + 1}/{n_batches}: pairs [{pair_start}:{pair_end}] "
              f"({n_batch_pairs} pairs, {n_batch_pairs * 2} frames) ---", flush=True)

        # Prepare batch data
        # Masks: even-indexed for mask_t, odd-indexed for mask_t1
        batch_masks_t = gt_masks[frame_start:frame_end:2].to(device)
        batch_masks_t1 = gt_masks[frame_start + 1:frame_end + 1:2].to(device)
        batch_gt_masks = gt_masks[frame_start:frame_end].to(device)
        batch_pose_targets = pose_targets[pair_start:pair_end]
        batch_init_poses = init_poses[pair_start:pair_end]

        # Lane G: build (B, 2, H, W, 3) GT-pair tensor for KL-distill auxiliary
        # loss. Only materialize when actually needed — gt_frames are uint8
        # (H, W, 3) per frame from load_gt_video; we cast to float so the
        # tensor matches the renderer's float-0-255 output range that
        # kl_distill_segnet_only's _hwc_to_chw expects.
        batch_gt_frames_pair = None
        if args.kl_distill_weight > 0:
            pair_t = torch.stack(
                [gt_frames[i].float() for i in range(frame_start, frame_end, 2)]
            )
            pair_t1 = torch.stack(
                [gt_frames[i].float() for i in range(frame_start + 1, frame_end + 1, 2)]
            )
            batch_gt_frames_pair = torch.stack([pair_t, pair_t1], dim=1).to(device)

        # Lane PS-V2: refresh the per-class weights tensor from the
        # learnable module each batch. The module's softmax forward is
        # differentiable, but we only need the tensor VALUE here — the
        # Lagrangian update on the learnable params happens AFTER the
        # batch using the per-class distortion measurements returned by
        # optimize_poses_batch.
        if learnable_segnet_class_weights is not None:
            with torch.no_grad():
                segnet_class_weights_t = learnable_segnet_class_weights().detach().cpu()

        t0 = time.monotonic()
        optimized_cond, batch_metrics = optimize_poses_batch(
            renderer=renderer,
            masks_t=batch_masks_t,
            masks_t1=batch_masks_t1,
            gt_masks=batch_gt_masks,
            pose_targets=batch_pose_targets,
            posenet=posenet,
            segnet=segnet,
            init_poses=batch_init_poses,
            device=device,
            steps=args.steps,
            lr=args.lr,
            seg_weight=args.seg_weight,
            pose_weight=args.pose_weight,
            hinge_margin=args.hinge_margin,
            eval_roundtrip=args.eval_roundtrip,
            posetto_noise_std=args.posetto_noise_std,
            early_stop_patience=args.early_stop_patience,
            argmax_constraint=args.argmax_constraint,
            max_retries=args.max_retries,
            flip_budget=args.flip_budget,
            latent_dim=args.latent_dim,
            log_every=args.log_every,
            zoom_warp=zoom_warp,
            batch_pair_indices=torch.arange(pair_start, pair_end, device=device),
            gt_frames_pair=batch_gt_frames_pair,
            kl_distill_weight=args.kl_distill_weight,
            kl_distill_temperature=args.kl_distill_temperature,
            pose_mode=args.pose_mode,
            linf_pose_weight=args.linf_pose_weight,
            linf_pose_budget=args.linf_pose_budget,
            segnet_class_weights=segnet_class_weights_t,
            kl_distill_snr_controller=kl_distill_snr_controller,
            optimizer_kind=getattr(args, "optimizer", "adam"),
            riemannian_momentum=getattr(args, "riemannian_momentum", 0.9),
        )

        # Lane PS-V2: dual-ascent step on the learnable per-class weights.
        # The variance-equalisation penalty pulls weight mass toward the
        # bottleneck classes (where per-class distortion is highest) and
        # away from solved classes (where per-class distortion is low).
        # Per-class distortion is measured from the batch metrics; if not
        # surfaced (legacy batch_metrics), we fall back to using the
        # batch's mean SegNet distortion as a single-class proxy so the
        # rate-Lagrangian still constrains the simplex.
        if learnable_segnet_class_weights is not None:
            from tac.learnable_class_weights import (
                compute_class_weight_equalisation_penalty,
            )
            per_class = batch_metrics.get("per_class_distortion")
            if per_class is None:
                # Single-class proxy: replicate the mean SegNet distortion
                # across all 5 classes — the variance is then 0 so the
                # penalty is 0 (no-op). This keeps the parameter in its
                # warm-start state until per-class measurements are
                # plumbed through batch_metrics.
                seg_d = float(batch_metrics.get("final_seg_distortion", 0.0))
                per_class_t = torch.full((5,), seg_d, dtype=torch.float32)
            else:
                per_class_t = torch.as_tensor(per_class, dtype=torch.float32)
                if per_class_t.numel() != 5:
                    per_class_t = torch.full((5,), float(per_class_t.mean()))
            cw_optim = torch.optim.Adam(
                learnable_segnet_class_weights.parameters(),
                lr=float(args.learnable_segnet_class_weights_lr),
            )
            cw_optim.zero_grad(set_to_none=True)
            pen = compute_class_weight_equalisation_penalty(
                learnable_segnet_class_weights,
                per_class_t.to(
                    next(learnable_segnet_class_weights.parameters()).device
                ),
                lambda_var=float(args.learnable_segnet_class_weights_var_lambda),
            )
            pen.backward()
            cw_optim.step()
        dt = time.monotonic() - t0

        all_optimized[pair_start:pair_end] = optimized_cond
        batch_metrics["batch_idx"] = batch_idx
        batch_metrics["time_s"] = dt
        all_metrics.append(batch_metrics)
        total_steps += batch_metrics["steps_run"]

        print(f"  Batch {batch_idx + 1} done in {dt:.1f}s "
              f"(improvement: {batch_metrics['improvement_pct']:.1f}%)", flush=True)

        # Save intermediate results. Emit BOTH .pt (for resume) and a .meta
        # sidecar so any consumer can verify completeness without trusting the
        # filename. 2026-04-26: a wrapper blindly used `optimized_poses_partial.pt`
        # as the archive's `optimized_poses.bin` — auth_eval_renderer crashed
        # 7 min later. The new `tac.submission_archive.load_optimized_poses(
        # ..., expected_n_pairs=N)` check catches this, but writing the meta
        # makes the partial-vs-final distinction first-class data, not a
        # naming convention.
        torch.save(all_optimized[:pair_end], output_dir / "optimized_poses_partial.pt")
        import json as _json
        (output_dir / "optimized_poses_partial.meta").write_text(_json.dumps({
            "n_pairs_complete": int(pair_end),
            "n_pairs_total": int(n_pairs),
            "is_final": False,
            "batch_idx": int(batch_idx),
        }))

        # Free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ── Step 6: Compare GT poses vs optimized poses ─────────────────────
    print("\n" + "=" * 70, flush=True)
    print("RESULTS: GT poses vs optimized poses", flush=True)
    print("=" * 70, flush=True)

    # Save optimized poses (both formats for compatibility). Lane M-V2
    # (2026-04-27 fix): in radial-zoom mode we save the FULL (N, 6) tensor
    # where dim 0 is the optimized radial-zoom scalar and dims 1-5 are
    # FROZEN at their baseline values (from `--gt-poses-path`). The
    # original Lane M-V1 saved (N, 1) and relied on the inflate side
    # zero-padding dims 1-5 — that was the V1 bug: dims 1-5 of the
    # baseline poses are NOT zero (they encode the auxiliary PoseNet
    # information the renderer was trained on); zero-padding them at
    # inflate destroyed PoseNet (V1 score 2.35 vs Lane A 1.15).
    # Saving (N, 6) here means the file is consumable by inflate /
    # downstream tooling without any pose_mode-aware adapter — the
    # bytes ARE the 6-DOF poses, and only the optimization side knew
    # about the 1-DOF representation.
    pose_part = all_optimized[:, :pose_dim_internal]  # (N, pose_dim_internal)
    if args.pose_mode == "radial-zoom":
        # Compose (N, 6): dim 0 = optimized scalar; dims 1-5 = frozen
        # baseline values from init_poses (which was loaded from
        # --gt-poses-path or extracted from GT frames). init_poses is
        # ALWAYS (N, 6) per the load block above.
        baseline_aux = init_poses[:n_pairs, 1:6].detach().cpu().to(pose_part.dtype)
        if baseline_aux.shape[1] != 5:
            raise SystemExit(
                "FATAL: Lane M-V2 expects baseline init_poses to be (N, 6) "
                f"so dims 1-5 can be frozen-padded, got shape {tuple(init_poses.shape)}. "
                "Verify --gt-poses-path points at a (N, 6) tensor (e.g., "
                "submissions/baseline_dilated_h64_0_90/optimized_poses.pt)."
            )
        optimized_poses = torch.cat([pose_part[:, :1].cpu(), baseline_aux], dim=-1)
        print(
            f"  [Lane M-V2] composed (N, 6) save tensor: dim0=optimized radial-zoom, "
            f"dims1-5=frozen baseline (from --gt-poses-path)",
            flush=True,
        )
    else:
        optimized_poses = pose_part
    torch.save(optimized_poses, output_dir / "optimized_poses.pt")
    pt_size = (output_dir / "optimized_poses.pt").stat().st_size
    # Also save compact binary format (raw fp16, ~53% smaller). This is the
    # canonical archive artifact — wrappers should ALWAYS reach for the .bin
    # and never rename a .pt to .bin (the 2026-04-26 SHIRAZ crash).
    from tac.submission_archive import save_poses_binary
    bin_size = save_poses_binary(optimized_poses, output_dir / "optimized_poses.bin")
    # Emit completion sidecar so consumers can distinguish final vs partial
    # without trusting filename conventions. Lane M-V2: pose_dim recorded
    # in meta is now ALWAYS 6 in radial-zoom mode (we save (N, 6) with
    # frozen baseline dims 1-5), but we still record `pose_mode` for
    # observability + paper provenance.
    import json as _json
    (output_dir / "optimized_poses.meta").write_text(_json.dumps({
        "n_pairs_complete": int(optimized_poses.shape[0]),
        "n_pairs_total": int(optimized_poses.shape[0]),
        "is_final": True,
        "pose_dim": int(optimized_poses.shape[1]),
        "pose_mode": args.pose_mode,
    }))
    # Atomically remove the partial markers — if both `optimized_poses.bin`
    # and `optimized_poses_partial.pt` exist, downstream tooling has to guess.
    # The final write is the source of truth; partials are debug-only.
    for stale in ("optimized_poses_partial.pt", "optimized_poses_partial.meta"):
        p = output_dir / stale
        if p.exists():
            p.unlink()
    print(f"  Saved optimized_poses.pt: {optimized_poses.shape} ({pt_size:,} bytes)", flush=True)
    print(f"  Saved optimized_poses.bin: {optimized_poses.shape} ({bin_size:,} bytes, -{100*(1-bin_size/pt_size):.0f}%)", flush=True)

    if args.latent_dim > 0:
        # Latent slot starts at pose_dim_internal (1 for radial-zoom, 6
        # for full-6dof). Note: optimize_poses_batch currently raises
        # NotImplementedError for latent_dim>0, but keep this slice
        # correct so when latent FiLM lands the save path works.
        optimized_latents = all_optimized[:, pose_dim_internal:]
        torch.save(optimized_latents, output_dir / "optimized_latents.pt")
        print(f"  Saved optimized_latents.pt: {optimized_latents.shape}", flush=True)

    # Compute proxy score with GT poses vs optimized poses
    print("\n  Computing proxy scores (GT poses vs optimized poses)...", flush=True)
    from tac.scorer import compute_proxy_score

    # Lane M-V2: optimized_poses is now ALWAYS (N, 6) — in radial-zoom
    # mode dim 0 is the optimized scalar and dims 1-5 are the frozen
    # baseline values (composed in the save block above). No projection
    # needed for the renderer call; we keep the (N, 1) defensive branch
    # only as a fallback for any future code path that bypasses the
    # save-block composition (it should never trigger on the canonical
    # path; if it does we pad with zeros to AT LEAST avoid a shape
    # mismatch — but the warning prints loud so the operator notices).
    if args.pose_mode == "radial-zoom" and optimized_poses.shape[1] == 1:
        print(
            "  [Lane M-V2 WARNING] optimized_poses is (N, 1) at proxy-score time. "
            "The save-block composition should have produced (N, 6) with frozen "
            "baseline dims 1-5. Falling back to ZERO pad for the in-memory render "
            "(this will undercount PoseNet — fix the save-block path).",
            flush=True,
        )
        zeros_pad = torch.zeros(
            optimized_poses.shape[0], 6 - 1, dtype=optimized_poses.dtype,
        )
        optimized_poses_for_render = torch.cat([optimized_poses, zeros_pad], dim=-1)
    else:
        optimized_poses_for_render = optimized_poses

    # Generate frames with GT poses (always 6-DOF — init_poses is GT)
    gt_pose_frames = _generate_frames(renderer, gt_masks, init_poses[:n_pairs], device, args.batch_pairs, zoom_warp=zoom_warp)
    gt_score = compute_proxy_score(
        gt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    # Generate frames with optimized poses (always 6-DOF after Lane M projection)
    opt_pose_frames = _generate_frames(renderer, gt_masks, optimized_poses_for_render, device, args.batch_pairs, zoom_warp=zoom_warp)
    opt_score = compute_proxy_score(
        opt_pose_frames, gt_frames, posenet, segnet, device, rate=0.0,
    )

    print(f"\n  GT Poses:        score={gt_score['score']:.4f} "
          f"(pose={gt_score['pose']:.6f}, seg={gt_score['seg']:.6f})", flush=True)
    print(f"  Optimized Poses: score={opt_score['score']:.4f} "
          f"(pose={opt_score['pose']:.6f}, seg={opt_score['seg']:.6f})", flush=True)
    print(f"  Delta:           {opt_score['score'] - gt_score['score']:+.4f} "
          f"(pose: {opt_score['pose'] - gt_score['pose']:+.6f}, "
          f"seg: {opt_score['seg'] - gt_score['seg']:+.6f})", flush=True)

    # Pose vector statistics — compare on the OPTIMIZABLE shape so
    # radial-zoom (N, 1) vs full-6dof (N, 6) both have well-defined L2
    # norms. Lane M-V2: optimized_poses is now ALWAYS (N, 6); in
    # radial-zoom mode dims 1-5 are FROZEN baseline values so their
    # contribution to the delta is zero, but we slice to the optimizable
    # dim so the printed L2 norm reflects actual optimizer movement.
    delta_dim = pose_dim_internal  # 1 for radial-zoom, 6 for full-6dof
    pose_delta = (
        optimized_poses[:, :delta_dim] - init_poses[:n_pairs, :delta_dim]
    ).norm(dim=1)
    print(f"\n  Pose delta: mean={pose_delta.mean():.4f}, "
          f"max={pose_delta.max():.4f}, min={pose_delta.min():.4f}", flush=True)

    # Archive size estimate
    archive_bytes = n_pairs * cond_dim * 4  # float32
    archive_fp16_bytes = n_pairs * cond_dim * 2  # float16
    print(f"\n  Archive size: {archive_bytes:,} bytes (fp32), {archive_fp16_bytes:,} bytes (fp16)", flush=True)

    # Save summary
    total_time = time.monotonic() - t_total
    summary = {
        "config": vars(args),
        "gt_score": gt_score,
        "optimized_score": opt_score,
        "delta_score": opt_score["score"] - gt_score["score"],
        "total_time_s": total_time,
        "total_steps": total_steps,
        "n_pairs": n_pairs,
        "pose_delta_mean": pose_delta.mean().item(),
        "pose_delta_max": pose_delta.max().item(),
        "batch_metrics": all_metrics,
        "archive_bytes_fp32": archive_bytes,
        "archive_bytes_fp16": archive_fp16_bytes,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Total time: {total_time:.1f}s", flush=True)
    print(f"  Results saved to: {output_dir}", flush=True)
    print("=" * 70, flush=True)


def _generate_frames(
    renderer: torch.nn.Module,
    masks: torch.Tensor,
    poses: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    zoom_warp: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Generate frames using renderer with pose conditioning.

    Args:
        renderer: AsymmetricPairGenerator
        masks: (N, H, W) long masks
        poses: (P, 6) pose vectors, P = N//2
        device: computation device
        batch_size: pairs per forward pass

    Returns:
        (N, H, W, 3) float tensor of rendered frames in [0, 255]
    """
    N = masks.shape[0]
    P = N // 2
    all_frames = []

    with torch.inference_mode():
        for start in range(0, P, batch_size):
            end = min(start + batch_size, P)
            mask_t = masks[2 * start:2 * end:2].to(device)
            mask_t1 = masks[2 * start + 1:2 * end + 1:2].to(device)
            batch_pose = poses[start:end].to(device)

            fwd_kwargs = {"pose": batch_pose}
            if zoom_warp is not None:
                pair_idx = torch.arange(start, end, device=device)
                fwd_kwargs["ego_flow"] = zoom_warp(pair_idx, mask_t.shape[1], mask_t.shape[2])
            pairs = renderer(mask_t, mask_t1, **fwd_kwargs)  # (B, 2, H, W, 3)
            f0 = pairs[:, 0]
            f1 = pairs[:, 1]
            B = f0.shape[0]
            interleaved = torch.stack([f0, f1], dim=1).reshape(2 * B, *f0.shape[1:])
            all_frames.append(interleaved.cpu())

    return torch.cat(all_frames, dim=0).float()


if __name__ == "__main__":
    main()
