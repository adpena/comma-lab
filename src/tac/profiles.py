"""Named training profiles for tac.

Each profile is a dict of TrainConfig overrides. Use with:
    from tac.profiles import PROFILES
    config = TrainConfig(**{**PROFILES["council_v1"], "tag": "my_run"})

Or from CLI:
    python train_tac.py --profile council_v1 --tag my_run
"""

# Council-recommended settings (2026-04-10 master session)
# Einstein/Tao/Contrarian/Karpathy/Hinton/LeCun/Qwen/DeepSeek consensus
#
# NOTE: kl_distill loss_mode DEPRECATED — two authoritative evals (1.85, 2.05)
# confirmed it destroys PoseNet. Reverted to "standard" loss_mode.
COUNCIL_V1 = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",  # was "kl_distill" — deprecated, kills PoseNet
    # KL distill fields kept as no-ops (ignored by standard loss_mode):
    # "temperature_start": 5.0,
    # "temperature_end": 0.5,
    # "temp_schedule": "exponential",
    "boundary_weight": 5.0,
    # boundary_anneal removed: it couples to temperature schedule which standard
    # loss_mode does not use (no temperature). Was dead code.
    "hard_frame_ratio": 0.3,  # power-law curriculum, 0.3 = moderate emphasis
    "error_replay_every": 200,  # recompute hard frames using model output
    "eval_every": 5,  # skip eval on 4/5 epochs (ramps to 1 in final 10%)
    "accum_steps": 4,
    "segnet_loss_weight": 30.0,
    "use_swa": True,  # SWA over final 20% for wider minima (better int8)
}

# Aggressive SegNet-focused (contrarian + DeepSeek recommendation)
# Inherits loss_mode="standard" from COUNCIL_V1 (kl_distill deprecated)
SEGNET_ATTACK = {
    **COUNCIL_V1,
    "boundary_weight": 200.0,  # near-maximum boundary focus
    "hard_frame_ratio": 0.5,  # stronger hard-frame emphasis
    "error_replay_every": 100,  # more frequent curriculum adaptation
}

# Conservative baseline (validated settings from dilated 1.33 run)
PROVEN_BASELINE = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

EBR_DILATED_H64 = {
    **PROVEN_BASELINE,
    "use_entropy_bottleneck": True,
    "eb_lambda": 0.01,
    "eb_num_channels": 64,
}

T2_XPRED = {
    **PROVEN_BASELINE,
    "use_t2_xpred_loss": True,
    "t2_xpred_sigma": 1.0,
    "t2_xpred_weighting": "v",
}

T2_MASK = {
    **PROVEN_BASELINE,
    "use_t2_mask": True,
    "t2_mask_p": 0.5,
    "t2_mask_ratio": 0.15,
    "t2_mask_apply_fraction": 0.4,
}

# Boundary-aware variant of proven_baseline (Trick 7: boundary retraining)
# Same architecture and hyperparameters, but boundary pixels get 5x gradient.
# This steers the model to allocate more correction capacity to class edges
# where SegNet is most sensitive. Zero-cost at inference.
PROVEN_BASELINE_BOUNDARY = {
    **PROVEN_BASELINE,
    "boundary_weight": 5.0,
}

# VP saliency variant — weight loss by vanishing point proximity + horizon band.
# Pixels near VP (where PoseNet is most sensitive) get more gradient.
# Training-time only, no inflate changes. Safe.
PROVEN_BASELINE_VP_SALIENCY = {
    **PROVEN_BASELINE,
    "use_vp_saliency": True,
    "vp_saliency_sigma": 40.0,
    "vp_saliency_min_weight": 0.3,
    "vp_saliency_horizon_boost": 2.0,
}

# Combined: boundary + VP saliency (the full CPU lane improvement stack)
PROVEN_BASELINE_FULL = {
    **PROVEN_BASELINE,
    "boundary_weight": 5.0,
    "use_vp_saliency": True,
    "vp_saliency_sigma": 40.0,
    "vp_saliency_min_weight": 0.3,
    "vp_saliency_horizon_boost": 2.0,
}

# Feature matching variant of proven_baseline (Trick 8: PoseNet layer 3 embeddings)
# Uses PoseNet intermediate features (stages.2, 256-ch) as the training target
# instead of the 6-value pose output. Strictly more informative because the
# mid-layer features capture texture statistics PoseNet actually uses.
PROVEN_BASELINE_FEATMATCH = {
    **PROVEN_BASELINE,
    "loss_mode": "feature_match",
    "segnet_loss_weight": 100.0,
}

# Width scaling (h=96 dilated with council settings)
# Inherits loss_mode="standard" from COUNCIL_V1 (kl_distill deprecated)
H96_COUNCIL = {
    **COUNCIL_V1,
    "hidden": 96,
}

# Fast iteration (for quick smoke tests)
SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 16,
    "kernel": 3,
    "epochs": 50,
    "lr": 1e-3,
    "eval_every": 10,
    "accum_steps": 2,
    "loss_mode": "standard",
}

# Council V2: Adaptive weights (Einstein/Tao derivation, 2026-04-10)
# DEPRECATED: adaptive_rebalance is dead (see CLAUDE.md). The Hinton T² correction
# was already inside the KL loss, so dividing by T² double-corrected. The compound
# invariant w_s*T² was trivially constant by construction. Kept for historical reference.
# Static placeholders below are overridden by AdaptiveWeights.rebalance() at runtime.
# See src/tac/adaptive.py for the full mathematical derivation.
COUNCIL_V2_ADAPTIVE = {
    **COUNCIL_V1,
    "segnet_loss_weight": 30.0,  # placeholder — overridden by AdaptiveWeights at runtime
    "boundary_weight": 100.0,  # placeholder — overridden by AdaptiveWeights at runtime
    "boundary_anneal": False,  # disabled: adaptive rebalance handles weight scheduling
    "adaptive_rebalance": True,  # flag for Trainer to invoke AdaptiveWeights.rebalance()
    "rebalance_every": 50,  # epochs between adaptive weight updates
    "boundary_fraction": 0.05,  # measured beta for AdaptiveWeights init
    "use_lsq": True,  # LSQ: learned step sizes via forward pre-hooks on Conv2d
}

PSD_STANDARD_ADAPTIVE = {
    **PROVEN_BASELINE,
    "variant": "psd",
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "use_swa": True,
}

SAUG_V2_DILATED_H64 = {
    **PROVEN_BASELINE,
    "use_saug_v2": True,
    "saug_v2_redraw_fraction": 0.05,
    "saug_v2_high_sigma_min": 80.0,
    "saug_v2_high_sigma_max": 2000.0,
    "saug_v2_normal_sigma_min": 0.5,
    "saug_v2_normal_sigma_max": 80.0,
}

# Lane HM: analytical road-plane homography motion (FOE-centered perspective
# zoom). Replaces the learned-CNN motion module with the orphan
# src/tac/contrib/homography_motion.py:HomographyMotionModule. Zero learned
# motion params (vs ~80K for the CNN motion module). Renderer arch
# unchanged from PROVEN_BASELINE; only the motion submodule is swapped.
# Predicted band: [1.30, 2.20] [contest-CUDA] — homography is a strong
# inductive bias for road scenes but loses fine-grained per-class flow
# the learned CNN can capture (vehicles, pedestrians).
HM_DILATED_H64 = {
    **PROVEN_BASELINE,
    "motion_type": "homography_analytical",
}

# Lane CG: calibrated viewing-ray positional encoding for the renderer.
# Adds the analytic per-pixel ray direction (derived from comma camera
# intrinsics) as a fixed buffer the renderer can attend to. Wires the
# orphan src/tac/contrib/calibrated_positional_encoding.py via a
# `use_calibrated_positional_encoding=True` flag picked up by the
# CalibratedMaskRenderer monkey-patch in that module's import path.
# Predicted band: [1.05, 1.18] [contest-CUDA] — same arch + extra
# input channels; small architectural risk, geometric prior is
# steganalysis-aligned (CNN scorers typically lack explicit camera
# calibration).
CG_DILATED_H64 = {
    **PROVEN_BASELINE,
    "use_calibrated_positional_encoding": True,
}

# Pareto frontier explorer: PCGrad + MRS-adaptive weights
# Decouples PoseNet and SegNet optimization via gradient surgery.
# The adaptive weight tracks the Pareto MRS condition:
#   w_seg = 200 * sqrt(10 * pose)
# This is not arbitrary — it's the first-order optimality condition
# on the score formula's iso-score tangent to the Pareto frontier.
PARETO_PCGRAD = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "pcgrad",  # gradient surgery: no SegNet gradient harms PoseNet
    "segnet_loss_weight": 30.0,  # initial — overridden by adaptive MRS at first eval
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "adaptive_rebalance": True,  # MRS-adaptive: w_seg = 200*sqrt(10*pose)
    "rebalance_every": 50,
    "boundary_fraction": 0.05,
    "use_swa": True,
}

# Extreme PoseNet: for Pareto frontier exploration (writeup artifact)
# Maximum PoseNet optimization, no SegNet consideration
EXTREME_POSENET = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 30.0,  # higher saliency weight = more PoseNet focus
    "sal_lambda": 1.5,
    "loss_mode": "standard",
    "segnet_loss_weight": 0.0,  # zero SegNet weight — pure PoseNet optimization
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

# Extreme SegNet: for Pareto frontier exploration (writeup artifact)
# Maximum SegNet optimization using focal STE (hard argmax gradients)
EXTREME_SEGNET = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 5.0,  # low saliency = less PoseNet-biased correction
    "sal_lambda": 0.5,
    "loss_mode": "focal_ste",  # hard argmax + focal weighting on boundaries
    "segnet_loss_weight": 200.0,  # heavy SegNet emphasis
    "boundary_weight": 200.0,
    "hard_frame_ratio": 0.5,
    "error_replay_every": 100,
    "eval_every": 5,
    "accum_steps": 4,
}

# Three-arm experiment (council-approved, pre-registered gate: seg<0.00590, pose<0.00250)
# Arm A: PCGrad gradient surgery (pareto_pcgrad profile, already defined above)
# Arm B: Simple reweighting control (seg_weight=200, no gradient surgery)
REWEIGHT_ABLATION = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 200.0,  # 2x proven_baseline, tests if reweighting alone helps
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

# Arm C: Spatial gate architecture (Collier's proposal, unanimous council approval)
GATED_DILATED_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "gated_dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 400,  # smoke test — 400 epochs per council
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

# Dilated h=32 smoke test: does the dilated architecture scale down?
# If yes, h=32 + CRF 36 saves ~0.083 pts combined (rate optimization).
# Also the production-optimal candidate for comma four deployment.
DILATED_H32_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 32,
    "kernel": 3,
    "epochs": 400,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 2,
}

# Dilated h=16: production-extreme, ~1.6KB at INT4
# Would run at ~1ms/frame on Snapdragon Hexagon NPU
DILATED_H16_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 16,
    "kernel": 3,
    "epochs": 400,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 2,
}

# Kaggle P100 profile: 12h kernel limit, 11h safety margin.
# P100 on CUDA does ~2.5 min/epoch for dilated h64 with subsample=4.
# 11h / 2.5 min = ~264 epochs max. Set 250 to be safe.
# If stuck on CPU (P100 compat issue), ~20 min/epoch = ~33 epochs only.
KAGGLE_P100_DILATED = {
    **PROVEN_BASELINE,
    "experiment_type": "training",
    "epochs": 1000,  # target: ~250-400 epochs complete in 11h on GPU
    "eval_every": 10,  # less frequent eval to save time
    "wall_clock_timeout": 39600,  # 11h in seconds (12h Kaggle limit - 1h safety)
}

# Kaggle P100 long run: aggressive 2500 epochs, relies on timeout to stop cleanly.
# Use when you want maximum training time and the timeout handles the rest.
KAGGLE_P100_LONG = {
    **PROVEN_BASELINE,
    "experiment_type": "training",
    "epochs": 2500,
    "eval_every": 10,
    "wall_clock_timeout": 39600,  # 11h safety margin
}

# ── GPU Lane: Mask Renderer profiles ────────────────────────────────────
# Segment → Compress Masks → Neural Render pipeline.
# Instead of postfilter on compressed video, render frames from masks.
# Score formula is the same: 100*seg + sqrt(10*pose).

MASK_RENDERER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "mask_renderer",
    "hidden": 40,  # base_ch for MaskRenderer U-Net
    "mid_ch": 64,  # bottleneck channels
    "embed_dim": 6,  # per-class embedding dimension
    "motion_hidden": 32,  # MotionPredictor hidden channels
    "noise_mode": "deterministic",  # "deterministic", "shared", "independent"
    "blend_mode": "spatial",  # "scalar", "spatial", "none"
    "motion_type": "depth_aware",  # "depth_aware", "learned_cnn", "analytical", "none"
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,  # no saliency recon (rendering from scratch)
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,  # Phase 1: L1+edge loss (no scorer)
}

MASK_RENDERER_FULL = {
    "experiment_type": "training",
    "variant": "mask_renderer",
    "hidden": 40,
    "mid_ch": 64,
    "embed_dim": 6,
    "motion_hidden": 32,
    "noise_mode": "deterministic",
    "blend_mode": "spatial",
    "motion_type": "depth_aware",
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,  # Phase 1: L1+edge loss (no scorer)
}

# Extended capacity variant: 48→80→48 (~500K params)
MASK_RENDERER_WIDE = {
    "experiment_type": "training",
    "variant": "mask_renderer",
    "hidden": 48,
    "mid_ch": 80,
    "embed_dim": 8,
    "motion_hidden": 48,
    "noise_mode": "deterministic",
    "blend_mode": "spatial",
    "motion_type": "depth_aware",
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,
}

# Deep U-Net variant: depth=2 (two-level downscale, ~450K params)
MASK_RENDERER_DEEP = {
    **MASK_RENDERER_FULL,
    "depth": 2,  # two-scale U-Net
    "pretrain_epochs": 500,
}

# ── Wavelet Lane: Wavelet-domain renderer profiles ─────────────────────
# Predict wavelet coefficients from masks, reconstruct via parameter-free iDWT.
# Smaller than pixel-domain U-Net (~100-200K params vs 312K).

WAVELET_RENDERER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "wavelet_renderer",
    "hidden": 96,  # base hidden width for coeff predictors (~137K total)
    "embed_dim": 8,  # per-class embedding dimension
    "motion_hidden": 32,  # MotionPredictor hidden channels
    "noise_mode": "deterministic",
    "blend_mode": "spatial",
    "motion_type": "depth_aware",
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,  # no saliency recon (rendering from scratch)
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 50,  # Phase 1: L1+edge loss (no scorer)
}

WAVELET_RENDERER_FULL = {
    "experiment_type": "training",
    "variant": "wavelet_renderer",
    "hidden": 96,
    "embed_dim": 8,
    "motion_hidden": 32,
    "noise_mode": "deterministic",
    "blend_mode": "spatial",
    "motion_type": "depth_aware",
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 200,  # Phase 1: longer warm-up for wavelet basis
}

# ── Diffusion Teacher + Distillation profiles ─────────────────────────
# Diffusion teacher trains a DDPM conditioned on segmentation masks.
# Slow to sample (~50 denoising steps) but produces high-quality targets
# for distilling into our fast CNN student (MaskRenderer/PostFilter).

DIFFUSION_TEACHER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "diffusion_teacher",
    "hidden": 32,  # base U-Net channels (32→64→128)
    "num_timesteps": 50,  # shorter schedule for smoke test
    "time_dim": 64,  # timestep embedding dim
    "beta_start": 1e-4,  # noise schedule lower bound
    "beta_end": 0.02,  # noise schedule upper bound
    "epochs": 100,
    "lr": 1e-4,
    "ema_decay": 0.999,
    "eval_every": 20,
    "accum_steps": 2,
    "loss_mode": "standard",
}

DIFFUSION_TEACHER_FULL = {
    "experiment_type": "training",
    "variant": "diffusion_teacher",
    "hidden": 64,  # base channels (64→128→256)
    "num_timesteps": 100,  # full noise schedule
    "time_dim": 128,  # timestep embedding dim
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "epochs": 1000,
    "lr": 1e-4,
    "ema_decay": 0.999,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
}

DISTILLATION_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "distillation",
    "hidden": 64,  # student architecture width
    "teacher_steps": 25,  # teacher sampling steps
    "use_ddim": True,  # deterministic teacher sampling
    "task_loss_weight": 0.0,  # pure distillation, no scorer
    "epochs": 200,
    "lr": 1e-4,
    "eval_every": 20,
    "accum_steps": 2,
    "loss_mode": "standard",
}

DISTILLATION_FULL = {
    "experiment_type": "training",
    "variant": "distillation",
    "hidden": 64,
    "teacher_steps": 50,
    "use_ddim": True,
    "task_loss_weight": 0.1,  # hybrid: 90% distillation + 10% scorer
    "epochs": 1000,
    "lr": 1e-4,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
}

# ── DP-SIMS Lane: SPADE-based semantic image synthesis ──────────────────
# DP-SIMS (CVPR 2024) inspired progressive SPADE generator.
# Full spatially-adaptive normalization (not just per-class CLADE).
# Cross-attention noise injection for texture diversity.
# Target: 500K-2M params, FP4 makes 2M = ~1MB.

DP_SIMS_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dp_sims",
    "channels": (128, 64, 32, 16),  # lightweight: ~500K params
    "init_h": 24,  # 1/16 of 384
    "init_w": 32,  # 1/16 of 512
    "spade_hidden": 32,  # smaller SPADE conditioning nets
    "noise_dim": 16,  # cross-attention noise dimension
    "use_noise": True,  # enable noise injection
    "motion_hidden": 32,  # MotionPredictor hidden channels
    "motion_embed_dim": 6,  # MotionPredictor embedding dim
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,  # no saliency recon (rendering from scratch)
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,  # Phase 1: L1+edge loss (no scorer)
}

DP_SIMS_FULL = {
    "experiment_type": "training",
    "variant": "dp_sims",
    "channels": (256, 128, 64, 32),  # full capacity: ~1.5M params
    "init_h": 24,
    "init_w": 32,
    "spade_hidden": 64,
    "noise_dim": 16,
    "use_noise": True,
    "motion_hidden": 32,
    "motion_embed_dim": 6,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,  # Phase 1: L1+edge loss (no scorer)
}

# ── DP-SIMS Small: halved SPADE for ~500KB FP4 ──────────────────────────
# The full DP-SIMS model (256,128,64,32) is 2.2MB FP4 — too heavy for rate.
# Halving channels to (128,64,32,16) targets ~500KB FP4.
# PoseNet fix: add motion_hidden=48 (wider MotionPredictor) to compensate
# for reduced generative capacity with better temporal coherence.
# SegNet: keep segnet_loss_weight=100 (proven); spade_hidden=32 suffices
# at this channel width.

DP_SIMS_SMALL_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dp_sims",
    "channels": (128, 64, 32, 16),  # halved from full (256,128,64,32)
    "init_h": 24,
    "init_w": 32,
    "spade_hidden": 32,  # matches channel width
    "noise_dim": 8,  # halved: less texture diversity, tighter compression
    "use_noise": True,
    "motion_hidden": 48,  # WIDER than full (32): compensate for smaller gen
    "motion_embed_dim": 8,  # richer embeddings for PoseNet fidelity
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 5.0,  # mild boundary focus
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
}

DP_SIMS_SMALL_FULL = {
    "experiment_type": "training",
    "variant": "dp_sims",
    "channels": (128, 64, 32, 16),  # halved from full (256,128,64,32)
    "init_h": 24,
    "init_w": 32,
    "spade_hidden": 32,
    "noise_dim": 8,
    "use_noise": True,
    "motion_hidden": 48,  # wider MotionPredictor for PoseNet
    "motion_embed_dim": 8,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,  # Phase 2: 500 epochs scorer-guided
}

# ── VQ-VAE Lane: Learned discrete latent codec ────────────────────────
# GT -> VQ Encoder -> discrete codes -> entropy coding -> VQ Decoder -> RGB
# Replaces fixed 5-class segmentation with learned K=512 codebook.

VQVAE_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "vqvae",
    "hidden": 64,  # base_ch for encoder/decoder
    "num_embeddings": 512,  # codebook size K
    "embedding_dim": 64,  # codebook vector dim D
    "commitment_cost": 0.25,  # VQ commitment loss weight
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,  # no saliency recon initially
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,  # Phase 1: reconstruction only (no scorer)
    "vq_weight": 1.0,  # VQ loss weight
    "perceptual_weight": 0.5,  # multi-scale perceptual loss weight
}

VQVAE_FULL = {
    "experiment_type": "training",
    "variant": "vqvae",
    "hidden": 64,
    "num_embeddings": 512,
    "embedding_dim": 64,
    "commitment_cost": 0.25,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,  # Phase 1: longer reconstruction warm-up
    "vq_weight": 1.0,
    "perceptual_weight": 0.5,
}

# Compact VQ-VAE: smaller codebook for tighter bitrate budget
VQVAE_COMPACT = {
    "experiment_type": "training",
    "variant": "vqvae",
    "hidden": 32,  # smaller encoder/decoder
    "num_embeddings": 256,  # K=256 (8-bit indices)
    "embedding_dim": 32,  # smaller embeddings
    "commitment_cost": 0.25,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,
    "vq_weight": 1.0,
    "perceptual_weight": 0.5,
}

# ── Technique 2: Luma-only dilated post-filter ─────────────────────────
# ~3x fewer params (1 channel vs 3). Netflix validated chroma uses Lanczos.
LUMA_ONLY_DILATED_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "luma_dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 400,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

LUMA_ONLY_DILATED_FULL = {
    **PROVEN_BASELINE,
    "variant": "luma_dilated",
    "epochs": 2500,
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "use_swa": True,
}

# ── Technique 3: Content-adaptive per-frame postfilter intensity ──────
CONTENT_ADAPTIVE_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "content_adaptive",
    "hidden": 64,
    "kernel": 3,
    "epochs": 400,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

# ── Technique 4: Resolution reduction + PixelShuffle upscale ─────────
PIXELSHUFFLE_UPSCALE_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "pixelshuffle_upscale",
    "hidden": 64,
    "kernel": 3,
    "epochs": 400,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

# ── Technique 9: DualHead architecture ────────────────────────────────
# Shared backbone, separate PoseNet (3x3) and SegNet (1x1) correction heads.
DUAL_HEAD_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dual_head",
    "hidden": 64,
    "kernel": 3,
    "epochs": 400,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 5,
    "accum_steps": 4,
}

DUAL_HEAD_FULL = {
    **PROVEN_BASELINE,
    "variant": "dual_head",
    "epochs": 2500,
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "use_swa": True,
}

# ── Technique 10: Focal gamma=4-5 ────────────────────────────────────
# Current gamma=2 is "too mild for 95% easy pixels" per Contrarian.
# Higher gamma focuses more aggressively on hard boundary pixels.
FOCAL_GAMMA_4 = {
    **PROVEN_BASELINE,
    "loss_mode": "focal_ste",
    "focal_gamma": 4.0,
    "epochs": 2500,
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "use_swa": True,
}

FOCAL_GAMMA_5 = {
    **PROVEN_BASELINE,
    "loss_mode": "focal_ste",
    "focal_gamma": 5.0,
    "epochs": 2500,
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "use_swa": True,
}

FOCAL_GAMMA_4_SMOKE = {
    **SMOKE,
    "loss_mode": "focal_ste",
    "focal_gamma": 4.0,
    "segnet_loss_weight": 100.0,
}

FOCAL_GAMMA_5_SMOKE = {
    **SMOKE,
    "loss_mode": "focal_ste",
    "focal_gamma": 5.0,
    "segnet_loss_weight": 100.0,
}

# ── Cross-Cultural Research Techniques ─────────────────────────────────

# Technique 7: Depthwise cascade renderer (Samsung)
# Depthwise separable convolutions with dilation cascade.
# 3-4x fewer params, better INT8 behavior, large receptive field.
DEPTHWISE_RENDERER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "depthwise_renderer",
    "hidden": 36,  # base_ch for depthwise cascade
    "embed_dim": 6,
    "motion_hidden": 32,
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
}

# Technique 8: Channel-recurrent renderer (Sony)
# Sequential Y -> U|Y -> V|Y,U generation. 40-60% fewer params.
CHANNEL_RECURRENT_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "channel_recurrent",
    "hidden": 24,  # per-subnet hidden width
    "embed_dim": 6,
    "motion_hidden": 32,
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
}

# Technique 9: Coordinate-based renderer (INRIA COOL)
# Per-pixel MLP with positional encoding. Smallest possible renderer (~50K params).
COORD_RENDERER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "coord_renderer",
    "hidden": 64,  # MLP hidden width
    "embed_dim": 8,  # class embedding dim
    "motion_hidden": 32,
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
}

# Technique 9B: Cool-Chic-style shared latent renderer (Orange/CNES).
# Multi-resolution learned latents + tiny synthesis decoder. This is a
# deterministic overfitting experiment; no KL distill, no adaptive rebalance.
COOLCHIC_RENDERER_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "coolchic_renderer",
    "hidden": 32,  # tiny synthesis decoder width
    "embed_dim": 6,
    "latent_ch": 8,
    "latent_shapes": ((6, 8), (12, 16), (24, 32)),
    "motion_hidden": 24,
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
    "seed": 42,
    "deterministic": True,
    "adaptive_rebalance": False,
}

COOLCHIC_RENDERER_FULL = {
    **COOLCHIC_RENDERER_SMOKE,
    "experiment_type": "training",
    "epochs": 2500,
    "lr": 5e-4,
    "motion_hidden": 32,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,
    # CLAUDE.md non-negotiable: every renderer profile MUST have eval_roundtrip
    # to close the proxy-CUDA gap. 2026-04-26 LANE-F preflight blocked because
    # this was None. Inherits to C3_RESIDUAL_RENDERER_FULL too via spread.
    "eval_roundtrip": True,
    "seed": 42,
    "deterministic": True,
}

# Technique 9C: C3-style coordinate residual codec.
# A Cool-Chic-style base renderer produces the coarse frame; a zero-initialized
# coordinate MLP learns bounded residuals. This isolates the C3 residual idea
# while preserving the mask-renderer training and evaluation contract.
C3_RESIDUAL_RENDERER_SMOKE = {
    **COOLCHIC_RENDERER_SMOKE,
    "variant": "c3_residual_renderer",
    "hidden": 24,
    "latent_ch": 6,
    "residual_hidden": 32,
    "residual_layers": 2,
    "residual_scale": 16.0,
}

C3_RESIDUAL_RENDERER_FULL = {
    **COOLCHIC_RENDERER_FULL,
    "variant": "c3_residual_renderer",
    "hidden": 24,
    "latent_ch": 6,
    "residual_hidden": 48,
    "residual_layers": 2,
    "residual_scale": 16.0,
}

# ── Yousfi Council Decode: Aggressive Overfitting ────────────────────
# CPU lane: overfit postfilter to 0.mkv with all tricks (10K epochs)
OVERFIT_CPU = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 10000,
    "lr": 5e-4,
    "ema_decay": 0.999,  # slower decay for overfitting (more averaging)
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "feature_match",  # Trick 1: intermediate feature matching
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.5,  # heavy hard-frame emphasis
    "error_replay_every": 100,
    "eval_every": 10,
    "accum_steps": 4,
    "use_swa": True,
    "use_lsq": True,
    "even_frame_skip_seg": True,  # Trick 3: skip SegNet on even frames
    "use_frequency_loss": True,  # Trick 2: wavelet domain shaping
    "frequency_loss_weight": 0.1,  # mild — complement to scorer loss
}

# CPU lane v2: overfit with boundary + feature match + SWA (Trick 9)
# Combines the best training-time tricks: feature matching loss (Trick 8),
# boundary awareness (Trick 7), and SWA for better int8 quantization.
OVERFIT_CPU_V2 = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 10000,
    "lr": 5e-4,
    "ema_decay": 0.999,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "feature_match",
    "boundary_weight": 5.0,
    "segnet_loss_weight": 100.0,
    "hard_frame_ratio": 0.5,
    "error_replay_every": 100,
    "eval_every": 50,
    "accum_steps": 4,
    "use_swa": True,
}

# GPU lane: overfit renderer to 0.mkv with all tricks (10K epochs)
OVERFIT_GPU = {
    "experiment_type": "training",
    "variant": "mask_renderer",
    "hidden": 48,
    "mid_ch": 80,
    "embed_dim": 8,
    "motion_hidden": 48,
    "depth": 2,
    "epochs": 10000,
    "pretrain_epochs": 1000,
    "lr": 1e-3,
    "ema_decay": 0.999,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.5,
    "error_replay_every": 100,
    "eval_every": 10,
    "accum_steps": 4,
    "use_swa": True,
    "even_frame_skip_seg": True,  # Trick 3: skip SegNet on even frames
    "use_frequency_loss": True,  # Trick 2: wavelet domain shaping
    "frequency_loss_weight": 0.1,
}

# ── Migrated Architecture Smoke Profiles ─────────────────────────────

# DCT mid-band filter smoke test
DCT_MIDBAND_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dct_midband",
    "hidden": 8,  # block size (hidden arg maps to block for this arch)
    "kernel": 3,
    "epochs": 50,
    "lr": 3e-4,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
    "alpha": 20.0,
    "sal_lambda": 0.1,
}

# FiLM QAT conditioned filter smoke test
FILM_CONDITIONED_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "film_qat",
    "hidden": 16,
    "kernel": 3,
    "epochs": 50,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
    "alpha": 20.0,
    "sal_lambda": 0.1,
}

# Counterpoint two-voice ensemble smoke test
COUNTERPOINT_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "counterpoint",
    "hidden": 16,
    "kernel": 3,
    "epochs": 50,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
    "alpha": 20.0,
    "sal_lambda": 0.1,
}

# PixelShuffle+Dilated hybrid smoke test
PIXELSHUFFLE_DILATED_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "pixelshuffle_dilated_v2",
    "hidden": 64,
    "kernel": 3,
    "epochs": 50,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
    "alpha": 20.0,
    "sal_lambda": 1.0,
}

# Uint8 STE training smoke test (uses dilated arch with uint8 STE wrapper)
UINT8_STE_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 32,
    "kernel": 3,
    "epochs": 50,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "eval_every": 10,
    "accum_steps": 4,
    "loss_mode": "standard",
    "alpha": 20.0,
    "sal_lambda": 0.1,
}

# ── Migrated Legacy Loss Technique Profiles ─────────────────────────────

# SegNet KL divergence loss (migrated from train_postfilter_segaware.py)
# Direct KL(GT || filtered) on SegNet logits instead of soft cosine proxy.
# NOTE: Not validated on authoritative scorer. Use for experimentation.
SEGNET_KL_SMOKE = {
    **SMOKE,
    "loss_mode": "segnet_kl",
    "segnet_loss_weight": 50.0,
}

SEGNET_KL_FULL = {
    **PROVEN_BASELINE,
    "loss_mode": "segnet_kl",
    "segnet_loss_weight": 50.0,
    "epochs": 2500,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "use_swa": True,
}

# PoseNet embedding loss (migrated from train_postfilter_featmatch.py)
# 512-d feature matching on PoseNet summarizer instead of 6-d pose MSE.
# NOTE: Not adopted for comma -- standard loss performed better in practice.
POSENET_EMBEDDING_SMOKE = {
    **SMOKE,
    "loss_mode": "posenet_embedding",
    "posenet_embedding_layer": "summary",
    "posenet_embedding_weight": 0.5,
}

POSENET_EMBEDDING_FULL = {
    **PROVEN_BASELINE,
    "loss_mode": "posenet_embedding",
    "posenet_embedding_layer": "summary",
    "posenet_embedding_weight": 0.5,
    "epochs": 2500,
    "use_swa": True,
}

# Counterpoint ensemble with band-orthogonality (migrated from train_postfilter_counterpoint.py)
# Uses band_lambda and decor_lambda on the existing counterpoint architecture smoke.
COUNTERPOINT_LOSSES_SMOKE = {
    **COUNTERPOINT_SMOKE,
    "band_lambda": 1.0,
    "decor_lambda": 0.5,
}

# Kalman weight filter (migrated from train_postfilter_kalman.py)
# Inverse-variance weighted averaging as alternative to EMA.
# EMA performed comparably in practice, but Kalman is more principled.
KALMAN_BASELINE = {
    **PROVEN_BASELINE,
    "use_kalman": True,
    "kalman_process_noise": 1e-6,
    "kalman_obs_noise_base": 1e-4,
    "kalman_obs_noise_scale": 10.0,
}

# ── DP-SIMS V2: Depth-aware motion + spatial blend + deterministic pairs ──
# V2 replaces the learned MotionPredictor CNN with a geometric
# DepthAwareMotionPredictor (~200 params, parallax flow from per-class depth
# + 6-DOF camera motion). Also uses spatially-varying blend weights and
# disables noise injection during pair generation for frame consistency.

DP_SIMS_V2_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dp_sims_v2",
    "channels": (128, 64, 32, 16),  # lightweight: ~500K params
    "init_h": 24,
    "init_w": 32,
    "spade_hidden": 32,
    "noise_dim": 16,
    "use_noise": True,  # enabled for training, disabled during pair gen
    "epochs": 200,
    "lr": 1e-3,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 1.0,
    "hard_frame_ratio": 0.0,
    "eval_every": 10,
    "accum_steps": 2,
    "pretrain_epochs": 100,
}

DP_SIMS_V2_FULL = {
    "experiment_type": "training",
    "variant": "dp_sims_v2",
    "channels": (256, 128, 64, 32),  # full capacity: ~1.5M params
    "init_h": 24,
    "init_w": 32,
    "spade_hidden": 64,
    "noise_dim": 16,
    "use_noise": True,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 0.0,
    "sal_lambda": 0.0,
    "loss_mode": "standard",
    "segnet_loss_weight": 100.0,
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    "pretrain_epochs": 500,
}

# ── Trick stacking inflate-time profiles ─────────────────────────────────
# These profiles configure the unified trick stacking engine
# (src/tac/trick_stack.py) for inflate-time optimizations.
# They are NOT training profiles — they control how tricks are combined
# during the inflate step.

# Full stacking: all tricks enabled, maximum quality
STACKED_INFLATE_FULL = {
    "experiment_type": "eval",
    "use_tto": True,
    "tto_steps": 15,
    "tto_lr": 1e-4,
    "tto_loss": "temporal_consistency",
    "tto_param_mode": "last_layer",
    "use_supervised_tto": True,
    "supervised_tto_steps": 20,
    "supervised_tto_lr": 1e-4,
    "supervised_tto_param_mode": "all",
    "use_multi_pass": 3,
    "use_brightness_shift": True,
    "brightness_shift_auto": True,
    "use_chroma_exploit": True,
    "chroma_perturbation_magnitude": 0.8,
    "use_fragility_weighting": True,
    "fragility_refinement_steps": 5,
    "use_noise_shaping": True,
    "noise_shaping_diffuse_error": True,
    "use_backward_delta_smoothing": True,
    "backward_smooth_alpha": 0.3,
    "backward_smooth_max_delta": 3.0,
    "use_null_space_projection": True,
    "null_space_rank_threshold": 1e-3,
}

# Safe stacking: proven tricks only, no scorer-dependent stages
STACKED_INFLATE_SAFE = {
    "experiment_type": "eval",
    "use_tto": True,
    "tto_steps": 10,
    "tto_lr": 1e-4,
    "tto_loss": "temporal_consistency",
    "use_supervised_tto": False,
    "use_multi_pass": 3,
    # Exploits #2 and #3: brightness shift + chroma smooth are zero scorer cost
    "use_brightness_shift": True,
    "brightness_shift_auto": True,
    "use_chroma_exploit": True,
    "chroma_perturbation_magnitude": 0.8,
    "use_fragility_weighting": False,
    "use_noise_shaping": False,
    "use_backward_delta_smoothing": False,
    "use_null_space_projection": False,
}

# Fast stacking: minimal overhead, just multi-pass + fast noise shaping
STACKED_INFLATE_FAST = {
    "experiment_type": "eval",
    "use_tto": False,
    "use_supervised_tto": False,
    "use_multi_pass": 2,
    "use_brightness_shift": True,
    "brightness_shift_auto": True,
    "use_chroma_exploit": False,
    "use_fragility_weighting": False,
    "use_noise_shaping": True,
    "noise_shaping_fast": True,
    "use_backward_delta_smoothing": False,
    "use_null_space_projection": False,
}

# Constrained optimization from noise (Yousfi GPU breakthrough)
# No neural renderer — optimize pixels via gradient descent against scorer constraints.
# Archive: ~8KB (masks + pose targets + seed). Inflate: ~50s on T4.
CONSTRAINED_GEN_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "num_steps": 100,
    "lr": 0.1,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "scorer_space": False,
    "log_every": 25,
}

# ── Finance & HFT optimizer profiles ─────────────────────────────────
# Cross-disciplinary algorithms from quantitative finance.
# See src/tac/finance_optimizers.py for implementations.

FINANCE_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "num_steps": 100,
    "lr": 0.5,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "scorer_space": False,
    "log_every": 25,
}

FINANCE_ENSEMBLE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "num_steps": 1000,
    "lr": 1.0,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "scorer_space": False,
    "log_every": 100,
}

FINANCE_HFT = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "num_steps": 500,
    "lr": 0.5,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "scorer_space": False,
    "log_every": 50,
}

CONSTRAINED_GEN_FULL = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "num_steps": 1000,
    "lr": 0.1,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "scorer_space": False,
    "log_every": 100,
}

# ── Variational frame generation (Euler-Lagrange) ────────────────────
# Solve the calculus of variations problem for scorer-optimal frames.
# J[f] = D(f) + lambda_smooth * E_smooth(f) + lambda_rate * TV(f)
VARIATIONAL_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "variational_gen",
    "lambda_smooth": 0.01,
    "lambda_rate": 0.1,
    "lambda_grad": 1.0,
    "lambda_lap": 0.1,
    "grad_clip": 10.0,
    "convergence_tol": 1e-6,
}

# ── Lagrangian dual rate-distortion optimizer ─────────────────────────
# Primal-dual method: min D(f) s.t. R(f) <= budget
# Lambda is LEARNED via dual ascent (KKT optimality).
LAGRANGIAN_DUAL_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "lagrangian_dual",
    "rate_budget": 0.01,
    "lambda_smooth": 0.01,
    "grad_clip": 10.0,
}

# ── Pareto frontier tracer ────────────────────────────────────────────
# Sweep rate budget to trace the full (seg, pose, rate) Pareto frontier.
# Each point is a Lagrangian dual solution with different rate constraint.
PARETO_TRACE = {
    "experiment_type": "gpu_lane",
    "variant": "pareto_trace",
    "lambda_smooth": 0.01,
}

# ── Full pipeline: variational + dual + manifold + Hamiltonian ────────
# Combines all five mathematical tools into a single inflate pipeline.
# Phase 1: Euler-Lagrange variational (200 steps)
# Phase 1.5: Scorer null-space projection (free rate reduction)
# Phase 2: Lagrangian dual (200 steps, learns optimal lambda)
# Phase 3: Hamiltonian dynamics (100 steps, escapes local minima)
# Phase 4: Gradient-directed Floyd-Steinberg dithering
CONSTRAINED_GEN_FULL_PIPELINE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen_pipeline",
    "phase1_steps": 200,
    "phase2_steps": 200,
    "phase3_steps": 100,
    "rate_budget": 0.01,
    "manifold_project": True,
    "use_hamiltonian": True,
    "use_dithering": True,
    "lambda_smooth": 0.01,
    "lambda_rate": 0.1,
    "hamiltonian_dt": 0.05,
    "hamiltonian_mass": 1.0,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "smooth_weight": 0.01,
    "max_jacobian_outputs": 16,
    "rank_threshold": 1e-3,
    "noise_seed": 42,
}

# ── Cross-disciplinary optimizer profiles ──────────────────────────────
# Run optimizers from physics, biology, chemistry, geophysics, climate
# science, astrophysics, and quantum computing on the constrained frame
# generation problem. See src/tac/cross_disciplinary_optimizers.py.

CROSS_DISC_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 10,
    "pop_size": 4,
    "num_particles": 4,
    "num_live": 4,
    "num_replicas": 2,
    "ensemble_size": 4,
    "n_components": 4,
    "n_paths": 2,
    "steps_per_scale": 3,
    "scales": [0.5, 1.0],
    "levels": [(96, 128), (192, 256)],
    "smooth_steps": 3,
    "local_steps": 3,
    "prediction_steps": 2,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

CROSS_DISC_ENSEMBLE = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 500,
    "pop_size": 16,
    "num_particles": 16,
    "num_live": 16,
    "num_replicas": 4,
    "ensemble_size": 16,
    "n_components": 32,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

CROSS_DISC_ANNEALING = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 1000,
    "T0": 100.0,
    "alpha": 0.997,
    "cooling_schedule": "exponential",
    "perturbation_scale": 5.0,
    "num_hops": 50,
    "local_steps": 30,
    "local_lr": 0.1,
    "temperature": 10.0,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

CROSS_DISC_EVOLUTIONARY = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 500,
    "pop_size": 24,
    "n_components": 48,
    "sigma0": 10.0,
    "F": 0.8,
    "CR": 0.9,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

CROSS_DISC_PHYSICS = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 500,
    "step_size": 0.01,
    "num_leapfrog_steps": 15,
    "mass_matrix_type": "learned",
    "lr": 0.1,
    "beta_start": 0.1,
    "beta_end": 100.0,
    "num_replicas": 6,
    "T_min": 0.1,
    "T_max": 100.0,
    "swap_every": 5,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

CROSS_DISC_GEOPHYSICS = {
    "experiment_type": "gpu_lane",
    "variant": "cross_disciplinary",
    "num_steps": 500,
    "scales": [0.25, 0.5, 1.0],
    "steps_per_scale": 150,
    "lr": 0.1,
    "tikhonov_weight": 0.01,
    "tv_weight": 0.1,
    "levels": [(96, 128), (192, 256), (384, 512)],
    "smooth_steps": 20,
    "num_cycles": 5,
    "temporal_weight": 10.0,
    "temporal_order": 2,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
}

# ── Domain-specific solver profiles (Yousfi cross-domain toolkit) ──────

# Quick test: all 10 solvers with minimal steps
DOMAIN_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
}

# Self-driving domain solvers only (ego-motion + road plane + vanishing point)
DOMAIN_DRIVING = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
}

# Signal processing solvers (matched filter + compressed sensing + OFDM)
DOMAIN_SIGNAL = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
}

# Full ensemble: all 10 solvers with production settings
DOMAIN_FULL = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
}

# ── Eureka Constrained Optimization smoke profiles ────────────────────────

COUPLED_TRAJECTORY_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "coupled_trajectory",
    "hidden": 0,  # no neural weights
    "epochs": 1,  # single "epoch" = the optimization loop
    "num_steps": 50,
    "lr": 0.01,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "loss_mode": "standard",
}

ALTERNATING_PROJECTIONS_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "alternating_projections",
    "hidden": 0,
    "epochs": 1,
    "num_outer_iterations": 10,
    "num_inner_steps": 5,
    "lr": 0.05,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "tv_weight": 1.0,
    "noise_seed": 42,
    "loss_mode": "standard",
}

NEWTON_STEP_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "newton_step",
    "hidden": 0,
    "epochs": 1,
    "num_newton_steps": 2,
    "max_iter_per_step": 5,
    "lr": 1.0,
    "history_size": 5,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "loss_mode": "standard",
}

SHANNON_COMPRESSOR_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "scorer_as_compressor",
    "hidden": 0,
    "epochs": 1,
    "topk": 2,
    "batch_size": 4,
    "loss_mode": "standard",
}

# ── Newton-CG Quadratic Optimizer profiles ──────────────────────────────
# Geometry deliberation: Newton-CG on the scorer iso-score surface.
# Uses Pearlmutter's trick for Hessian-vector products, Noether symmetry
# projection, and trust-region clipping. See tac.research.geometry_deliberation.

NEWTON_QUADRATIC_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 50,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "eval_every": 10,
    "accum_steps": 2,
    # Newton-CG refinement at inflate time
}

NEWTON_QUADRATIC_FULL = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "boundary_weight": 5.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    # Newton-CG refinement at inflate time
}

# ── Fridrich steganalysis-inspired profiles ─────────────────────────────
# Detection-boundary constrained optimization (inverse steganalysis).
# Cost map precomputed at compress time, zero-cost at inflate time.

FRIDRICH_SMOKE = {
    "experiment_type": "smoke_test",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 50,
    "lr": 1e-3,
    "eval_every": 10,
    "accum_steps": 2,
    "loss_mode": "standard",
    # Fridrich pipeline settings
}

FRIDRICH_FULL = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "boundary_weight": 5.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "use_swa": True,
    # Fridrich pipeline settings
}

FRIDRICH_CPU_POSTFILTER = {
    "experiment_type": "training",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "standard",
    "boundary_weight": 5.0,
    "eval_every": 5,
    "accum_steps": 4,
    # CPU postfilter: precompute cost map, apply at inflate via lookup
    # Skip heavy GPU optimization, rely on postfilter + cost weighting
}

# ── GPU Lane: Full Pipeline profiles ─���────────────────────────────────
# Coupled trajectory (4D-Var) + Fridrich constrained refinement + STC.
# These profiles drive gpu_lane_full_pipeline() in constrained_gen.py.

GPU_LANE_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "gpu_lane_full_pipeline",
    "hidden": 0,  # no neural weights
    "epochs": 1,  # single "epoch" = the optimization loop
    "num_steps": 100,  # coupled trajectory steps (smoke test)
    "fridrich_steps": 50,  # Fridrich refinement steps
    "lr": 0.01,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "batch_size": 8,  # P100 16GB -> 8 frames max
    "log_every": 25,
    "loss_mode": "standard",
    "skip_fridrich": False,
    "skip_stc": False,
    "cost_method": "hybrid",
    "num_probes": 10,  # fewer probes for smoke
    "rate_reduction": 0.1,
}

GPU_LANE_FULL = {
    "experiment_type": "gpu_lane",
    "variant": "constrained_gen",
    "optimizer": "gpu_lane_full_pipeline",
    "hidden": 0,
    "epochs": 1,
    "num_steps": 1000,  # full coupled trajectory optimization
    "fridrich_steps": 500,  # full Fridrich refinement
    "lr": 0.01,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "compress_weight": 1.0,
    "noise_seed": 42,
    "batch_size": 8,
    "log_every": 100,
    "loss_mode": "standard",
    "skip_fridrich": False,
    "skip_stc": False,
    "cost_method": "hybrid",
    "num_probes": 20,
    "rate_reduction": 0.1,
}

# ── Self-Compressing Postfilter profiles (Technique 1: Szabolcs) ─────────
# Learnable per-channel bit-depth. Channels that don't matter get 0 bits.
# Expected: 46KB -> 5-10KB postfilter, saving ~0.024 score points on rate.

SELF_COMPRESS_SMOKE = {
    "experiment_type": "self_compress",
    "variant": "dilated",
    "hidden": 16,
    "kernel": 3,
    "epochs": 100,
    "lr": 5e-4,
    "lr_bits": 1e-2,
    "loss_mode": "standard",
    "target_bits": 2000,
    "lambda_rate_start": 0.0,
    "lambda_rate_end": 0.5,
    "ramp_start_frac": 0.3,
    "scorer_weight": 20.0,
    "init_bits": 8.0,
}

# ── Entropy Archive profiles (Technique 2: Shannon) ─────────────────────
# Arithmetic coding with learned probability models.
# Target: 30-50% smaller than deflate on non-video components.

ENTROPY_ARCHIVE_SMOKE = {
    "experiment_type": "entropy_archive",
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 200,
    "lr": 1e-3,
    "loss_mode": "standard",
    "num_symbols": 256,
    "entropy_epochs": 200,
}

# ── Network Codec profiles (Technique 3: Network IS the Codec) ──────────
# SIREN memorization: network weights ARE the compressed video.
# GPU lane: 50-100KB for entire video reconstruction.

NETWORK_CODEC_SMOKE = {
    "experiment_type": "network_codec",
    "variant": "siren",
    "hidden": 16,
    "kernel": 3,  # unused but required by TrainConfig
    "layers": 2,
    "omega_0": 15.0,
    "epochs": 50,
    "lr": 1e-4,
    "loss_mode": "standard",
    "batch_pixels": 1024,
    "lambda_smooth": 0.1,
    "pos_encoding_freqs": 4,
    "target_size_kb": 10,
}

NETWORK_CODEC_FULL = {
    "experiment_type": "network_codec",
    "variant": "siren",
    "hidden": 64,
    "kernel": 3,
    "layers": 4,
    "omega_0": 30.0,
    "epochs": 5000,
    "lr": 1e-4,
    "loss_mode": "standard",
    "batch_pixels": 16384,
    "lambda_smooth": 0.1,
    "lambda_rate": 0.0,
    "lambda_rate_end": 1.0,
    "ramp_start_frac": 0.5,
    "pos_encoding_freqs": 6,
    "target_size_kb": 50,
}

# ── Wilde: compact renderer matched to Quantizr's 88K / 64KB FP4 ────────
# Compact renderer: minimal params, maximum scorer fidelity.
# Reference config for train_distill.py — translate fields to CLI args:
#   --base-ch 16 --mid-ch 24 --depth 1 --pose-dim 6 --use-dsconv
#   --segnet-loss-mode hinge --hinge-margin 0.5 --error-boost 9.0
#   --pose-weight 10.0 --seg-weight 100.0 --pixel-weight 0.1
#   --ema-decay 0.997 --eval-roundtrip
#   --phase1-epochs 800 --phase2-epochs 1200 --phase3-epochs 400
# motion_hidden=16 via --motion-hidden 16
#
# Methodology & Provenance:
#
#   Architecture:
#   - CLADE per-class normalization (Park et al. 2019 "Semantic Image Synthesis
#     with Spatially-Adaptive Normalization"). Independent discovery (2026-04-12)
#     via skunkworks council analysis of SegNet class-boundary sensitivity.
#     Our advantage over Quantizr (uses GroupNorm) and szabolcs-cs (no conditioning).
#   - Depthwise-separable convolutions (Howard et al. 2017 "MobileNets").
#     Adopted after Quantizr PR#55 binary analysis (2026-04-24) confirmed his use.
#   - Dilated residual blocks (Yu & Koltun 2016 "Multi-Scale Context Aggregation").
#     Validated by kaileh57 PR#58 ablation ("single largest win", 1.92 vs 2.03).
#   - Replicate padding: adopted from Yousfi's PR#55 comment (2026-04-19) noting
#     "artifacts at the image boundary, due to padding in the conv layers."
#   - Asymmetric pair generation: frame_t1=render(mask), frame_t=warp+gate*residual.
#     Independent design (2026-04-12) via skunkworks council, later validated by
#     Quantizr binary analysis showing similar dual-head paradigm.
#
#   Training:
#   - Freeze/unfreeze schedule: adapted from Quantizr's 5-stage pipeline discovered
#     via full binary reverse-engineering of PR#55 archive (2026-04-24). His stages:
#     ANCHOR(400ep)→ANCHOR_BOOST(80ep)→FINETUNE(320ep)→JOINT(160ep)→MICRO(120ep).
#   - error_boost per-pixel weighting (9x→49x): Quantizr's technique, values from
#     his training reconstruction. Quadratic: weight = 1 + (boost-1) * (error/mean)².
#   - EMA (Polyak 1992 "Acceleration of Stochastic Approximation"). decay=0.997.
#   - eval_roundtrip STE: independently discovered (2026-04-15) after catastrophic
#     proxy-auth divergence. Later confirmed as shared technique when Quantizr
#     disclosed it in PR#55 body as "a helpful hint to others" (2026-04-19).
#     Fixed across 15 files on 2026-04-24 after finding GT cache bypass bugs.
#
#   Novel discoveries (independent, by skunkworks council + geometric analysis):
#   - PoseNet rank-1 Jacobian (2026-04-24): Jacobian effective rank 1.008, dim 0
#     captures 99.8% of pose variance. Forward ego-motion is the ONLY significant
#     DOF. Validated empirically: 600 zoom scalars achieve 10.6x PoseNet improvement.
#   - Radial zoom warp: 600 learned scalars (1.2KB) replace 50K-param MotionPredictor.
#     Warp = radial expansion from Focus of Expansion (256,174) derived from comma
#     EON AR0231AT camera calibration. Geometrically exact for highway ego-motion.
#   - SegNet/PoseNet orthogonal decomposition: SegNet evaluates only frame_t1,
#     PoseNet evaluates the pair. These objectives are decoupled through architecture.
#     Freeze/unfreeze exploits this for zero-interference sequential optimization.
#   - Lane marking ego-motion estimation (2026-04-24): MUTCD-standard lane markings
#     (3m×15cm) visible in all 1200 frames encode vehicle speed through inter-frame
#     displacement. Zoom scalars computable from masks at inflate time (zero archive
#     cost). Paper-worthy: dual-use of mask channel for appearance + motion.
#
#   Research methodology: 15-member skunkworks council (Yousfi, Fridrich, Hotz,
#   Quantizr adversarial, Contrarian, + extended) with recursive adversarial review
#   until 3 consecutive clean passes. All design decisions by council consensus.
#   Full competitor reverse-engineering (Quantizr PR#55, szabolcs-cs PR#56).
WILDE = {
    "experiment_type": "renderer_training",
    # Architecture: council consensus — base_ch=32 for SegNet capacity.
    # SegNet gap is 5.7x (0.00347 vs Quantizr's 0.000613). Width is the #1 lever.
    # 181K params, ~89KB FP4. Rate cost 0.060 (vs 0.200 Quantizr). Worth it.
    "base_ch": 32,
    "mid_ch": 48,
    "embed_dim": 6,
    "motion_hidden": 24,
    "depth": 1,
    "pose_dim": 6,  # FiLM retained — council: keep correction path, zoom overrides flow only
    "use_dsconv": True,
    "padding_mode": "replicate",  # Yousfi: zeros creates boundary artifacts
    "use_dilation": True,  # kaileh57: "single largest win"
    "eval_roundtrip": True,
    # Fridrich inverse steganalysis losses (our competitive advantage)
    "use_texture_loss": True,      # UNIWARD: hide errors in textured regions
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,      # Square root law: spread errors, don't concentrate
    "linf_weight": 0.01,
    "use_markov_loss": True,       # HUGO: preserve local gradient statistics
    "markov_weight": 0.1,
    # Yousfi #3: UNIWARD-aligned spatially-adaptive quant noise. Companion
    # to texture_loss — that down-weights L1 in textured regions, this
    # actively trains for noise robustness in the same regions where the
    # FP4/mask codecs will inject quantization error at inflate time.
    # 2026-04-26 (PM): Fridrich R2 C2 fix landed — `variance_noise_mode`
    # now defaults to 'wavelet_db4' (un-decimated Daubechies-8 sub-band
    # energy per Holub & Fridrich 2014 §III.B), and the train_renderer
    # resolver re-wires the keys end-to-end. The legacy 'box' / 'variance'
    # mode remains accepted for A/B comparison only.
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    # Yousfi #5: ScanNet-style spatial uncertainty maps. Up-weights pixel
    # loss in regions SegNet is most confident about (low GT entropy).
    # Kept light (weight 0.05) to avoid amplifying the same signal as
    # variance_noise (overlapping spatial coverage) and KL distill (which
    # subsumes uncertainty information when active). See council notes
    # in tac.losses.segnet_uncertainty_weighted_loss.
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr #1 SegNet trick: KL distillation on logits with T=2.0.
    # Council 5/0 vote (2026-04-26): both WILDE and SHIRAZ must enable
    # kl_distill alongside their primary segnet loss. Argparse resolver
    # (commit 38a250b8) only enables KL distill when profile declares the
    # key — without these two lines the lane runs without Quantizr's
    # single most important SegNet trick.
    "kl_distill_weight": 1.0,
    "kl_distill_temperature": 2.0,
    # Training: 5-phase Quantizr-adapted schedule
    # Phase 1 (pixel warmup): all params, L1 loss
    # Phase 2 (anchor): freeze motion, SegNet CE + KL(T=2.0), error_boost=9x
    # Phase 3 (anchor_boost): same freeze, error_boost=49x, low LR
    # Phase 4 (joint): unfreeze all, hinge loss + PoseNet MSE
    # Phase 5 (hard-pair): top 20%, error_boost=25x
    "segnet_loss_mode": "hinge",  # Phase 4+5 (Phase 2+3 use xent per council)
    "hinge_margin": 1.0,  # Council: 1.0 > 0.5 for larger safety margin
    "error_boost": 9.0,  # Phase 2 anchor: 9x
    "error_boost_phase3": 49.0,  # Phase 3 anchor_boost: 49x extreme hard mining
    "freeze_motion_phase2": True,  # Freeze MotionPredictor during SegNet training
    "freeze_renderer_phase3": True,  # Freeze MaskRenderer during PoseNet training
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # Lane markings 15x (Yousfi: 1.2% but critical)
    "use_swa": True,               # Wider minima, better FP4 survival (Polyak 1992)
    # 5-phase Quantizr-adapted QAT schedule (Quantizr R2 C3 architectural
    # fix 2026-04-26). Phases 1-3 retain WILDE's tuned epoch budgets;
    # Phase 4 (QAT) and Phase 5 (final hard-pair polish) added per
    # Quantizr's published recipe so the schedule is no longer fictional.
    # train_renderer.has_5phase_schedule() activates when ANY phaseN > 0.
    "phase1_epochs": 600,  # Pixel warmup (200) + anchor (400)
    "phase2_epochs": 880,  # Anchor boost (80) + joint (800)
    "phase3_epochs": 200,  # Hard-pair fine-tune (legacy "Phase 3")
    "phase4_epochs": 200,  # QAT fine-tune (FakeQuantFP4 enabled)
    "phase5_epochs": 100,  # Final consolidation, hard-pair sampling
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,    # 0.1× base for QAT (Lin et al. 2017)
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    "seed": 42,
    "deterministic": True,
}

# ── Shiraz: mathematically principled adaptive training ──────────────
# A/B test against WILDE. Same architecture, different training strategy.
#
# Methodology & Provenance:
#
#   Architecture: identical to WILDE (fair A/B comparison). See WILDE for
#   full architecture provenance.
#
#   Training (what differs from WILDE):
#   - PCGrad gradient surgery (Yu et al. NeurIPS 2020 "Gradient Surgery
#     for Multi-Task Learning"): resolves SegNet/PoseNet gradient conflicts
#     on shared renderer parameters WITHOUT freeze/unfreeze. Activation-level
#     approximation — projects SegNet gradient so combined gradient is
#     non-opposing to PoseNet. Already implemented in our codebase (2026-04-14)
#     as scorer_loss_pcgrad in src/tac/losses.py, validated with telemetry.
#
#   - Focal STE loss (Lin et al. ICCV 2017 "Focal Loss for Dense Object
#     Detection"): per-pixel weighting (1-p_t)^γ with γ=2.0. Replaces
#     Quantizr's empirically-tuned error_boost (9x/49x) with an information-
#     theoretically principled alternative. At decision boundaries where
#     p_t≈0.5, focal weight is 0.25 (25x ratio over confident pixels at
#     p_t≈0.99). Combined with STE: forward=hard argmax disagreement
#     (matches official scorer exactly), backward=focal CE gradient (smooth,
#     principled). Already implemented (2026-04-14) as focal_segnet_ste_loss.
#
#   - Continuous adaptive training: no hard phase boundaries. SegNet and
#     PoseNet losses active throughout with PCGrad mediating conflicts.
#     Hard-frame curriculum via difficulty-weighted sampling (similar to
#     Self-Paced Learning, Kumar et al. NeurIPS 2010).
#
#   Novel (independent discoveries via skunkworks council):
#   - Score-contribution-proportional (SCP) gradient weighting: SegNet/PoseNet
#     weights adapt proportionally to their current score contribution
#     (100*seg vs sqrt(10*pose)). Eliminates arbitrary constants. Based on
#     our analysis that the scoring formula creates a time-varying optimal
#     weighting (2026-04-24). EMA-damped to prevent oscillation (Fridrich
#     stability analysis identified positive feedback loop without damping).
#
#   Hypothesis: principled gradient surgery + focal loss should match or
#   exceed brute-force freeze/unfreeze, especially in the joint optimization
#   regime where both scorers provide useful signal simultaneously.
#
#   Research methodology: same 15-member skunkworks council as WILDE.
#   This profile represents the "principled" arm of an A/B test against
#   WILDE's "empirical" arm. Both use identical architecture and total
#   training budget (1680 epochs) for fair comparison.
#
# Council dissents recorded:
#   Quantizr: "activation PCGrad insufficient, predict 10-20% worse SegNet"
#   Hotz: "use existing tested components only" → simplified to focal_ste + pcgrad
#   Fridrich: "use focal_ste not focal+hinge" → adopted
SHIRAZ = {
    "experiment_type": "renderer_training",
    # Architecture: matched to WILDE for fair A/B comparison
    "base_ch": 32,
    "mid_ch": 48,
    "embed_dim": 6,
    "motion_hidden": 24,
    "depth": 1,
    "pose_dim": 6,
    "use_dsconv": True,
    "padding_mode": "replicate",
    "use_dilation": True,
    "eval_roundtrip": True,
    # Loss: focal STE (principled per-pixel weighting, replaces error_boost)
    # Council Fridrich dissent: "use focal_ste not focal+hinge" — adopted.
    # focal_gamma only works with loss_mode=focal_ste, so we use that mode.
    "loss_mode": "focal_ste",       # STE: forward=hard argmax, backward=focal CE
    "segnet_loss_mode": "hinge",    # fallback for Phase 3 if needed
    "hinge_margin": 1.0,
    "focal_gamma": 2.0,            # focal per-pixel reweighting (Lin et al. 2017)
    "error_boost": 1.0,            # NO error_boost — focal handles hard pixels
    # Fridrich inverse steganalysis (works with ALL loss modes now)
    "use_texture_loss": True,      # UNIWARD: hide errors in textured regions
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,      # Square root law: spread errors
    "linf_weight": 0.01,
    "use_markov_loss": True,       # HUGO: preserve local gradient statistics
    "markov_weight": 0.1,
    # Yousfi #3: UNIWARD-aligned spatially-adaptive quant noise.
    # 2026-04-26 (PM): Fridrich R2 C2 fix landed — `variance_noise_mode`
    # now defaults to 'wavelet_db4' (un-decimated Daubechies-8 sub-band
    # energy per Holub & Fridrich 2014 §III.B), and the train_renderer
    # resolver re-wires the keys end-to-end. The legacy 'box' / 'variance'
    # mode remains accepted for A/B comparison only.
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    # Yousfi #5: ScanNet-style spatial uncertainty maps (light weight,
    # see WILDE / DEN comments for full provenance and council notes).
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr #1 SegNet trick: KL distillation on logits with T=2.0.
    # Council 5/0 vote (2026-04-26): both WILDE and SHIRAZ must enable
    # kl_distill alongside their primary segnet loss. Argparse resolver
    # (commit 38a250b8) only enables KL distill when profile declares the
    # key — without these two lines the lane runs without Quantizr's
    # single most important SegNet trick.
    "kl_distill_weight": 1.0,
    "kl_distill_temperature": 2.0,
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # Lane markings 15x (Yousfi: 1.2% but critical)
    "use_swa": True,               # Wider minima, better FP4 survival (Polyak 1992)
    # NO freeze/unfreeze — continuous adaptive training
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-phase Quantizr-adapted QAT schedule (Quantizr R2 C3 architectural
    # fix 2026-04-26). Same total budget arms as WILDE for fair A/B; adds
    # Phase 4 (QAT) and Phase 5 (final polish) so the train_renderer
    # 5-phase loop has all five phases declared explicitly.
    "phase1_epochs": 400,           # pixel warmup
    "phase2_epochs": 1080,          # scorer-guided (continuous, no anchor/boost split)
    "phase3_epochs": 200,           # hard-pair fine-tune (legacy "Phase 3")
    "phase4_epochs": 200,           # QAT fine-tune (FakeQuantFP4 enabled)
    "phase5_epochs": 100,           # Final consolidation, hard-pair sampling
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,              # 0.1× base for QAT (Lin et al. 2017)
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # Hard-frame curriculum
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    "seed": 42,
    "deterministic": True,
}


# ── GREEN: WILDE + radial zoom warp ────────────────────────────────────
# Iteration 2 profile. Builds on WILDE with architecturally-motivated
# simplification of the motion pathway.
#
# Methodology & Provenance:
#
#   Architecture: identical to WILDE except MotionPredictor outputs
#   gate(1) + residual(3) only (4 channels). Optical flow is provided
#   externally by RadialZoomWarp (src/tac/radial_zoom.py).
#
#   Motivation: the PoseNet Jacobian has effective rank 1.008 (verified
#   empirically in experiments/results/gradient_rank_analysis.json across
#   5 pairs). Only one degree of freedom matters: forward ego-motion as a
#   radial zoom from the Focus of Expansion. A full CNN predicting 2-channel
#   flow is redundant when 600 learned scalars capture 99.8% of pose variance.
#
#   Savings: ~14K fewer parameters, ~3.5KB smaller FP4 archive. The
#   MotionPredictor becomes a residual corrector rather than a full flow
#   predictor, which is architecturally cleaner and more constrained.
#
#   Council decision (2026-04-24): unanimous 5-0. Hotz: "if it's rank-1
#   just learn a scalar." Fridrich: "radial zoom IS the forward-translation
#   FOE structure." All design questions resolved.
#
#   Training: inherits WILDE's 5-phase schedule unchanged. The zoom scalars
#   are optimized separately via optimize_zoom_scalars() during pose TTO.
GREEN = {
    **WILDE,
    "use_zoom_flow": True,  # MotionPredictor: 4ch (gate+residual), flow from RadialZoomWarp
}

# ── Next-gen profiles: self-compression eureka findings (2026-04-25) ──────

# WILDE v2: beneficial quantization noise during scorer training
# Eureka 1+2: quantization noise in rendered frames acts as beneficial
# steganalytic cover. Renderer learns to produce scorer-robust output.
WILDE_V2 = {
    **WILDE,
    "beneficial_quant_noise": True,
    "quant_noise_bits": 6,  # uint6 quantization before scorer (conservative start)
}

# SHIRAZ v2: same addition
# R36 fix: use ** spread idiom matching WILDE_V2 / GREEN_V2 (was redundant
# dict comprehension that obscured intent).
SHIRAZ_V2 = {
    **SHIRAZ,
    "beneficial_quant_noise": True,
    "quant_noise_bits": 4,  # uint4 more aggressive (SHIRAZ uses focal STE, more robust)
}

# GREEN v2: zoom flow + beneficial noise
GREEN_V2 = {
    **GREEN,
    "beneficial_quant_noise": True,
    "quant_noise_bits": 6,
}

# Current 4090 config (103K params, proxy 0.612 at ep1500)
DEFINITIVE_FLOAT_EMA = {
    "experiment_type": "renderer_training",
    "base_ch": 20,
    "mid_ch": 28,
    "embed_dim": 6,
    "motion_hidden": 32,
    "depth": 1,
    "pose_dim": 6,
    "use_dsconv": True,
    "padding_mode": "zeros",
    "eval_roundtrip": True,
    "segnet_loss_mode": "hinge",
    "hinge_margin": 0.5,
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "phase1_epochs": 1000,
    "phase2_epochs": 1500,
    "phase3_epochs": 500,
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase1_batch_size": 8,
    "phase2_batch_size": 4,
    "phase3_batch_size": 4,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    "seed": 42,
    "deterministic": True,
}


# ── DEN: Distill, Embed, Nuance — the everything-we-learned profile ──────
#
# Built 2026-04-26 to beat Quantizr (0.33) on the contest-CUDA gate. Combines
# every validated win into one integrated training run:
#
#   - Architecture: matches Quantizr's capacity (88K params target) — DSConv
#     + FiLM conditioning + use_zoom_flow=True (Hotz radial-zoom analytical).
#   - Quantization: 5-stage with our RESIDUAL codebook + robust_scale +
#     stochastic + ndim<2 buffer skip + train↔export consistency. Memory:
#     project_5stage_quantization_advantage.md says we already strictly beat
#     Quantizr's vanilla 5-stage.
#   - Inverse steganalysis losses: UNIWARD texture (8x8 box-pool variance),
#     L∞ top-32 mean per pair, Markov gradient continuity. All wired with
#     correct semantics (not the silent-no-op variants from before).
#   - KL distillation T=2.0 SegNet-only (kl_distill_segnet_only — no PoseNet
#     double-counting). Adds during all phases at decreasing weight.
#   - Mask augmentation: mixed CRF50/63 at training time so the renderer
#     learns to be robust to high-CRF mask compression at compress time.
#   - eval_roundtrip=True throughout (NON-NEGOTIABLE per CLAUDE.md).
#   - Per-class weights with lane-marking 15x boost (Yousfi: 1.2% but critical).
#   - SWA for FP4 robustness (Polyak 1992).
#
# Schedule: 5-stage QAT (Quantizr's recipe) — anchor → finetune → joint →
# QAT → final. Phase boundaries set so we hit the auth-eval gate at hour 5
# with first measurement.
#
# Council sign-off (2026-04-26): All 5 inner council members assented in
# parallel. Yousfi: "if mask aug holds, this is the play." Fridrich: "L∞
# wired correctly now, not topk-of-1." Hotz: "use_zoom_flow + analytical
# zoom is the rank-1 truth." Quantizr: "matches my arch except for our QAT
# advantage — should win." Contrarian: "single deficit is no auth gate
# during training; mitigated by inline auth eval at end of each phase."
DEN = {
    "experiment_type": "renderer_training",
    # Architecture: 88K params target (Quantizr-matched capacity).
    "base_ch": 28,                  # Quantizr-class capacity
    "mid_ch": 40,
    "embed_dim": 6,
    "motion_hidden": 16,            # smaller — radial zoom flow handles rank-1
    "depth": 1,
    "pose_dim": 6,
    "use_dsconv": True,             # depthwise-separable (Quantizr trick)
    "padding_mode": "replicate",
    "use_dilation": True,
    # 2026-04-26 council: DEN initially set use_zoom_flow=True (Hotz radial
    # zoom advantage) but train_renderer's training loop can't yet compute
    # the ego_flow that AsymmetricPairGenerator(use_zoom_flow=True) requires
    # as forward input. Flipped to False for the immediate retrain so DEN
    # produces a loadable checkpoint via the standard MotionPredictor path.
    # DEN-V2 with use_zoom_flow=True is queued for after the ego_flow
    # plumbing in train_renderer is built (small refactor, ~1h).
    "use_zoom_flow": False,
    "eval_roundtrip": True,         # NON-NEGOTIABLE
    # Loss configuration — focal STE + Fridrich aux losses + KL distill.
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 1.0,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    # Fridrich inverse-steganalysis (FIXED versions — no silent no-ops).
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    # Yousfi #3: UNIWARD-aligned spatially-adaptive quant noise.
    # 2026-04-26 (PM): Fridrich R2 C2 fix landed — `variance_noise_mode`
    # now defaults to 'wavelet_db4' (un-decimated Daubechies-8 sub-band
    # energy per Holub & Fridrich 2014 §III.B), and the train_renderer
    # resolver re-wires the keys end-to-end. The legacy 'box' / 'variance'
    # mode remains accepted for A/B comparison only.
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    # Yousfi #5: ScanNet-style spatial uncertainty maps. DEN runs KL distill
    # on SegNet (kl_distill_weight=1.0 below) which directly targets the
    # softmax distribution and largely subsumes the uncertainty signal —
    # so the weight is kept very low (0.02) to act as a sweetener rather
    # than a competing objective. Council Quantizr (recorded): "with KL on,
    # uncertainty loss is 80% redundant; keep it ≤0.05 or drop entirely."
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.02,
    "uncertainty_loss_floor": 0.1,
    # Fridrich council #1 (2026-04-26): JPEG-Q-table-weighted DCT loss.
    # Penalises low-freq residual energy ~6× more than high-freq, letting
    # the renderer hide error in DCT directions the scorers cannot see
    # (UNIWARD analog in the DCT domain). Council recommended weight 0.5,
    # matching texture_loss_weight scale. Estimated +0.04-0.08 score impact.
    "dct_quant_weight": 0.5,
    # KL distill SegNet-only (no PoseNet double-counting). Schedule decay
    # over phases so distillation pressure tapers as the model converges.
    "kl_distill_weight": 1.0,
    "kl_distill_temperature": 2.0,
    # Per-class weights (lane markings 15x — Yousfi).
    "use_per_class_weights": True,
    # Score weights — DOMINATED BY SegNet (77x more important per scoring math).
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_swa": True,                # SWA for wider minima → FP4 survives
    # NO freeze/unfreeze — continuous training across phases.
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-stage QAT schedule. Phase 4 = QAT fine-tune, Phase 5 = final.
    # Total ~3000 epochs ≈ 5h on 4090 (matches Lane C budget).
    "phase1_epochs": 800,           # anchor: pixel + light scorer
    "phase2_epochs": 1200,          # finetune: full scorer + Fridrich
    "phase3_epochs": 600,           # joint: + KL distill ramp
    "phase4_epochs": 300,           # QAT fine-tune (LSQ + FakeQuantFP4)
    "phase5_epochs": 100,           # final consolidation at FP4
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,              # 0.1× base for QAT (Lin et al. 2017)
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # 5-stage quantization config — our advantage over Quantizr's vanilla.
    "fp4_codebook": "residual",     # RESIDUAL codebook beats uniform
    "fp4_robust_scale": True,       # robust per-channel scaling
    "fp4_stochastic": True,         # stochastic rounding in QAT
    # Mask augmentation: train on mixed CRF so we don't overfit to one mask
    # encoding. Tested at compress time on whatever Lane A0 picks.
    "mask_noise_prob": 0.5,         # 50% of training pairs use augmented mask
    # Half-frame mask simulation (Lane D2): only useful when use_zoom_flow=True
    # (the inflate side warps odd-frame masks via RadialZoomWarp). Disabled
    # because DEN currently has use_zoom_flow=False — preflight enforces
    # this consistency. Re-enable when DEN-V2 lands the zoom-flow path.
    "mask_half_sim_prob": 0.0,
    # Hard-frame curriculum carried from SHIRAZ.
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 100,              # auth-eval-style score at every checkpoint
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    # configure_reproducibility() pins random/torch/cuda RNGs, sets
    # use_deterministic_algorithms(True), cudnn.deterministic=True, and
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 — making CUDA matmuls bit-exact across
    # runs (within the same GPU SKU and PyTorch version).
    "seed": 42,
    "deterministic": True,
}


# ── LANE D: dilated-h64 retrain for half-frame masks ───────────────────
# Council 2026-04-27 (5/0). The verified 0.9001 baseline renderer ships in
# ASYM format with arch (base_ch=36, mid_ch=60, motion_hidden=32, depth=1,
# embed_dim=6, pose_dim=6, use_dsconv=False, use_zoom_flow=False). When fed
# half-frame masks at inflate (warp-expansion via RadialZoomWarp.warp_inverse_masks)
# it produces PoseNet=28.7 vs baseline 0.011 — a 2,600x explosion. Diagnosis
# (memory feedback_half_frame_breaks_posenet, 2026-04-27): the MotionPredictor
# uses (e_t1 - e_t).abs() as a diff feature; when mask_t is reconstructed via
# warp instead of independently SegNet-extracted, the diff feature becomes
# near-zero and motion prediction collapses.
#
# This profile retrains the same arch family with the warp-expansion injected
# into the training data path (mask_half_sim_prob=0.5, use_zoom_flow=True).
# Half the batches see (mask_t = warp(mask_t1)), half see (mask_t, mask_t1)
# from independent SegNet extraction. The motion module learns BOTH
# distributions, robust to whichever wins at inflate.
#
# Architecture parity with baseline 0.9001:
#   * base_ch=36, mid_ch=60, motion_hidden=32, depth=1
#   * embed_dim=6, pose_dim=6 (FiLM modulation, retained)
#   * use_dsconv=False, use_dilation=False (matches baseline exactly)
# Architecture DELTA from baseline (the only change needed for half-frame):
#   * use_zoom_flow=True (motion outputs gate+residual=4ch instead of flow+gate+residual=6ch)
#   * mask_half_sim_prob=0.5 (training-time half-frame simulation, 50% of batches)
# Net param delta: ~-14K (motion drops 2 channels of conv output).
#
# 5-phase Quantizr-style schedule (matches SHIRAZ budget for fair A/B):
#   Phase 1 (400ep): pixel L1+edge warmup
#   Phase 2 (1080ep): scorer-guided + Fridrich aux losses
#   Phase 3 (200ep): hard-pair fine-tune
#   Phase 4 (200ep): QAT (FakeQuantFP4)
#   Phase 5 (100ep): final consolidation
# Total ~1980 epochs ≈ 5h on RTX 4090 ($1.25 @ $0.25/hr).
#
# Smoke kill protocol (see scripts/remote_lane_d_halfframe_retrain.sh):
#   * Phase 1 end (~1h): pixel L1 < 12 (baseline plateaus 5-7)
#   * Phase 2 ep200 (~2h): scorer < 8.0 (kill if higher)
#   * Phase 2 ep800 (~3.5h): scorer < 3.0 (must beat the broken 17.55 baseline)
#   * Phase 4 end (~5h): scorer < 1.5 (target 0.55-0.75)
#
# Predicted score landing zone:
#   - Half-frame archive saves ~0.20-0.32 in rate vs full-frame baseline (0.46 → 0.14-0.26)
#   - PoseNet should recover from 28.7 → 0.05-0.10 (10x worse than baseline 0.011 is
#     acceptable — warp introduces motion noise that pure-SegNet extraction doesn't)
#   - SegNet stays at ~0.003 (already excellent at half-frame)
#   - Standalone score: 0.55-0.75. Stacked with Lane A pose TTO + Lane C δ: 0.40-0.55.
DILATED_H64_HALF_FRAME = {
    "experiment_type": "renderer_training",
    # Architecture: matches the verified 0.9001 baseline ASYM arch byte-for-byte
    # (header inspected from submissions/baseline_dilated_h64_0_90/renderer.bin).
    # variant="dilated" keeps lineage with PROVEN_BASELINE / SHIRAZ / DEN naming
    # (all use the same build_renderer() path; "dilated" is a label, not a kernel
    # property — actual dilation is controlled by use_dilation below).
    "variant": "dilated",
    "base_ch": 36,
    "mid_ch": 60,
    "motion_hidden": 32,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,                 # FiLM modulation retained (baseline parity)
    "use_dsconv": False,
    "padding_mode": "zeros",       # baseline arch — no replicate padding
    "use_dilation": False,         # baseline ASYM has no dilation flag
    # Lane D mandate: this is the WHOLE POINT — joint-train motion module on
    # warp-expanded even-frame masks so it doesn't collapse at inflate.
    "use_zoom_flow": True,         # motion: 4ch (gate+residual), flow comes from RadialZoomWarp
    "mask_half_sim_prob": 0.5,     # 50% of batches: mask_t = inverse_warp(mask_t1)
    "eval_roundtrip": True,        # NON-NEGOTIABLE per CLAUDE.md
    # Loss configuration — same as SHIRAZ for direct A/B comparability
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 1.0,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    # Fridrich inverse steganalysis (matches WILDE/SHIRAZ recipe)
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr KL distillation on SegNet (T=2.0, weight=1.0)
    "kl_distill_weight": 1.0,
    "kl_distill_temperature": 2.0,
    # Score weights — SegNet dominates per scoring math (77x more important)
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # lane markings 15x (Yousfi)
    "use_swa": True,                # SWA for wider minima → FP4 survival
    # No freeze/unfreeze — continuous adaptive training
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-phase QAT schedule — same shape and budget as SHIRAZ for fair comparison
    "phase1_epochs": 400,
    "phase2_epochs": 1080,
    "phase3_epochs": 200,
    "phase4_epochs": 200,
    "phase5_epochs": 100,
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # Hard-frame curriculum
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # 5-stage quantization config (our advantage over Quantizr's vanilla)
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    # Mask augmentation: train on mixed CRF so we don't overfit one mask encoding
    "mask_noise_prob": 0.5,
    # Deterministic reproducibility — pinned via configure_reproducibility().
    # Same-seed re-runs on the same GPU SKU + PyTorch version produce
    # bit-exact checkpoints (see CLAUDE.md canonical pipeline standard).
    "seed": 42,
    "deterministic": True,
}


# ── LANE D-V3: dilated-h64 half-frame retrain — annealed warp + KL distill ──
# Council 2026-04-27. Lane D-V1 was killed at ep 1230/1980 (62%) with proxy
# fp4_scorer plateaued at ~40 since ep ~700. Lane D-V2 (in flight) tries
# CHOICE B = higher per-phase LR floor (P2 3e-4 → 5e-4, P3 1e-4 → 2e-4,
# P4 5e-5 → 1e-4, P5 1e-5 → 2e-5). Lane D-V3 STACKS V2's LR fix with two
# additional levers borrowed from Lane V-V2 + the post-fix KL distill:
#
#   1. ANNEALED mask_half_sim_prob 0.0 → 0.5 (NOT 1.0 — Lane D's arch is
#      retrofit at 0.5, not a from-scratch 88K like Lane V which goes 1.0).
#      Schedule: 0.0 for first 30% of epochs (P1+early P2 warmup on full-
#      frame), linear ramp 0.0 → 0.5 from 30% to 70% of epochs, then 0.5
#      for last 30%. The hypothesis: V1's plateau at ep 700 (~35% through
#      training) coincides with the early-phase optimisation difficulty of
#      learning from a half/half mask distribution from epoch 0. Annealing
#      lets the renderer first lock in the easier full-frame distribution,
#      THEN smoothly transition to the half-frame regime as the optimiser
#      settles. Same paradigm as Lane V-V2 (which scored higher than V1
#      cold-start in the V2 design analysis), adapted for Lane D's 0.5
#      retrofit endpoint.
#
#   2. KL DISTILL WEIGHT REDUCED 1.0 → 0.002 (post-bug-fix value). The
#      original V1 weight=1.0 was set BEFORE the 2026-04-27 reduction fix
#      in kl_distill_segnet_only (losses.py:705) which divides by H*W. With
#      the reduction fix in place, raw KL ≈ 6.2e-3 × T² ≈ 0.025; weight=1.0
#      means KL contributes 0.025 (~5x scorer loss), DROWNING the scorer
#      signal. The correct post-fix weight is 0.002 so KL contribution is
#      ~5e-5 (~1% of scorer loss) — this matches Lane V's value (memory:
#      project_lane_v_quantizr_replica_88k_halfframe.md).
#
# Mechanism verification (Phase-0 instrumentation, this run only):
#   * train_renderer.py per-epoch logs: hf_fires=N/M, hf_warp_diff=X
#   * Verifies the half-frame branch fires at the annealed rate AND that
#     warp_inverse_masks produces non-trivial mask perturbations (a degenerate
#     identity warp would have hf_warp_diff~0 even when hf_fires=N/M).
#
# All other knobs identical to Lane D-V1 / D-V2 for direct A/B traceability.
# The CLI flag --mask-half-sim-prob-schedule overrides the profile schedule
# at launch time (Lane V-V2 wired this resolver in train_renderer.py:802).
#
# Predicted band [1.50, 2.50] [contest-CUDA] (wider than V2 [1.50, 3.00]
# because V3 introduces TWO new variables — annealing + KL-weight fix —
# on top of V2's LR floor). Council pre-registration:
#   * Floor 1.50: V3 stacks address 3 distinct V1 failure modes (LR
#     starvation + cold-start optimisation difficulty + KL loss drowning
#     scorer signal). If all 3 contributed, V3 should land in the 1.5-2.0
#     band. Beats Lane A's 1.15 = STRETCH GOAL (would require all 3
#     mechanisms to contribute additively).
#   * Ceiling 2.50: even if KL fix is the only useful lever, V3 should
#     match V2's projected ceiling (LR fix alone gets to 2.5-3.0 per
#     V2 prediction); V3 won't be WORSE than V2 with two strict
#     improvements.
#
# Cost: 4090 @ $0.25/hr × ~5h = ~$1.25 (same as V2; no schedule change).
DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL = {
    **DILATED_H64_HALF_FRAME,  # inherit ALL knobs (arch, Fridrich aux, phases)
    # Override 1: annealed mask_half_sim_prob 0.0 → 0.5 over training.
    # The static value below is the END (post-ramp) value (= 0.5). When
    # the schedule is set, the per-epoch value is computed via
    # mask_half_sim_prob_for_epoch() in train_renderer.py and the inner
    # step loop reads `current_mask_half_sim_prob` instead of the static.
    "mask_half_sim_prob": 0.5,
    "mask_half_sim_prob_anneal": {
        "start_value": 0.0,
        "end_value": 0.5,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    },
    # Override 2: KL distill weight 1.0 → 0.002 (post-bug-fix value, matches
    # Lane V — see V1 kl_distill_weight=1.0 in DILATED_H64_HALF_FRAME).
    # This is the SCALAR weight applied to the SegNet-only KL term inside
    # the training loss; KL contribution = weight × T² × raw_seg_kl.
    # Pre-fix raw_seg_kl was unreduced ≈ 5000× larger; post-fix it is
    # divided by H*W so weight=0.002 gives the intended 1%-of-scorer-loss
    # contribution.
    "kl_distill_weight": 0.002,
    # Override 3: V2's LR floor fix (CHOICE B per remote_lane_d_v2_*.sh).
    # P1 unchanged (pixel warmup converges fast).
    # P2 raised 1.67× to fix the ep 700 plateau (cosine_lr eta_min=1e-6
    # starves the optimiser at the back half of P2 with V1's 3e-4 base).
    # P3-P5 raised 2.0× proportionally.
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-4,
    "phase3_lr": 2e-4,
    "phase4_lr": 1e-4,
    "phase5_lr": 2e-5,
    # Different seed so V1/V2/V3 explore different RNG basins (same
    # convention as Lane V-V2 vs Lane V-V1).
    "seed": 43,
}


# ── LANE H-V3: half-frame revival via JOINT warp-expansion training ──────────
# Council 2026-04-28 evening (forensic audit of killed lanes V/V-V2/D-V3).
# Anchored on Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
#
# WHY: Quantizr ships half-frame at 0.33 and beats us 0.33 vs 1.05. Our 4
# prior half-frame attempts (Lane D V1/V2/V3, Lane V V1/V2) all failed because
# we never properly implemented JOINT warp-expansion training — they were
# either RETROFITS onto a renderer locked into (e_t1-e_t).abs() diff features
# (Lane D), had train/inflate distribution mismatch (Lane D-V3 endpoint=0.5
# vs inflate=1.0), or had channel-broadcast bugs in the 88K DSConv path
# (Lane V). Lane H-V3 fixes ALL three failure modes:
#   1. Curriculum from 0.0 → 1.0 (joint training from epoch 0; the renderer
#      sees warped masks AS A FIRST-CLASS DISTRIBUTION, with a brief warmup
#      phase to establish a strong full-frame initialisation first).
#   2. Train endpoint = inflate distribution (1.0 = always-on warp). Fixes
#      Lane D-V3's distribution mismatch (Check 42 train_inference_parity bug class).
#   3. 288K dilated-h64 arch (NOT 88K DSConv) — bypasses the channel-broadcast
#      bug from Lane V. Known-working pipeline at 1.05.
#
# Curriculum schedule (1980 total epochs):
#   * Epochs 0-99 (5%):       full-frame warmup (mask_half_sim_prob=0.0)
#   * Epochs 99-297 (5-15%):  linear ramp 0.0 → 1.0 (200 epochs)
#   * Epochs 297-1980 (rest): half-frame endpoint (mask_half_sim_prob=1.0)
#
# The 5-15% ramp is INTENTIONALLY AGGRESSIVE (vs Lane V-V2's 30-70%) because
# the goal is to spend most training epochs at the inflate-time distribution.
# The warmup buys a strong init; the rest is endpoint convergence.
#
# Predicted band [0.55, 0.95] [contest-CUDA] per the council:
#   * Floor 0.55: joint training works. Half-frame archive saves ~0.20 in
#     rate vs Lane G v3's full-frame; renderer trained for warp-expansion
#     holds PoseNet ≤ 0.020 + SegNet ≤ 0.005. 1.05 - 0.20 (rate) - 0.30
#     (PoseNet headroom from joint training) ~ 0.55.
#   * Ceiling 0.95: curriculum buys nothing; renderer learns the wrong
#     distribution well in warmup and the late transition is too abrupt.
#     Score lands ~ Lane G v3 - small rate gain.
#
# Cost: 4090 @ $0.25/hr × ~5h = ~$1.25 (matches Lane D-V3 schedule).
# Plus ~30min pose TTO + ~15min auth eval = $1.50 total.
H_V3_JOINT_HALFFRAME = {
    **DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,  # inherit Lane G v3 anchor
    # Override 1: train endpoint matches inflate-time distribution.
    # Lane D-V3 had endpoint=0.5 vs inflate=1.0 (distribution mismatch =
    # Lane M-V2 BUG-1 class). Lane H-V3 fixes by using endpoint=1.0.
    "mask_half_sim_prob": 1.0,
    # Override 2: curriculum 0.0 → 1.0 with aggressive ramp (5%-15% of epochs).
    # The brief warmup establishes a full-frame init; the rest converges on
    # the inflate-time distribution. Different from Lane V-V2 (30%-70%) because
    # the goal here is endpoint training, not smooth interpolation.
    "mask_half_sim_prob_anneal": {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.05,
        "ramp_end_frac": 0.15,
    },
    # Different seed so Lane H-V3 explores a different RNG basin from Lane G v3
    # (seed=43) and Lane V-V2 (seed=1235).
    "seed": 67,
}


# ── LANE J-JBL: Jaccard Metric Loss + Boundary Label Smoothing distillation ──
# Jack-from-skunkworks Cycle 1 TOP-1 (research file
# .omx/research/jack_skunkworks_segnet_rate_research_20260428.md §S1).
#
# Anchored on Lane G v3 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
# Replaces the SegNet KL distillation auxiliary (Hinton 2015) with the JBL
# combined loss from Wang et al. (NeurIPS 2023, arXiv 2302.05666):
#   * Jaccard Metric Loss on student vs teacher soft labels (subsumes
#     Lovász-Softmax, +2-5 mIoU near boundaries on EfficientNet — directly
#     relevant to the comma SegNet's EfficientNet-B2 backbone).
#   * Boundary Label Smoothing on GT cross-entropy with boundary pixels
#     weighted 3× interior pixels.
#
# loss_mode="jbl" is the dispatch hook in train_renderer.py; the dispatch
# replaces the kl_distill_segnet_only auxiliary with combined_jbl_distill_loss
# (defined in tac.losses_jbl). The kl_distill_weight knob is repurposed as
# the JBL master scalar so the wiring is byte-identical to Lane G v3 except
# for the loss family.
#
# Predicted band [0.92, 1.02] per Jack §S1 (lowest cost, highest confidence
# SegNet attack). At anchor 1.05, the conservative case is +20% reduction in
# SegNet distortion (0.004 -> 0.0032 -> -0.08 score = 0.97).
#
# CLAUDE.md compliance:
#   * eval_roundtrip remains True (inherited from Lane G v3).
#   * No scorer at inflate (loss is compress-time only).
#   * Auth eval at end (driven by remote_lane_j_jbl_jaccard_bls.sh).
J_JBL_DILATED_H64 = {
    **DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,
    # Loss-mode dispatch: train_renderer.py reads args.loss_mode and
    # picks combined_jbl_distill_loss when this is "jbl". Default
    # remains "standard" everywhere else.
    "loss_mode": "jbl",
    # Boundary pixel weight inside the BLS+CE channel (3× the interior
    # weight; middle of the paper's 3-5x recommendation).
    "boundary_weight": 3.0,
    # Label-smoothing epsilon applied at boundary pixels only (Wang et al.
    # canonical default).
    "bls_smoothing": 0.1,
    # Different seed so JBL and Lane G v3 explore different RNG basins.
    "seed": 47,
}


# ── LANE J-IMP: Iterative Magnitude Pruning (Frankle & Carbin 2018) ──────
# Outer-loop driver lives in experiments/train_imp_cycle.py; the per-cycle
# fine-tune borrows the dilated-h64 baseline arch + standard renderer-loss
# settings, but the script also passes the exact arch via CLI flags
# (--base-ch / --mid-ch / --motion-hidden / --depth / --embed-dim /
# --pose-dim / --padding-mode), so this PROFILE is mostly a registry entry
# satisfying preflight `check_deploy_script_profiles_exist_in_registry`.
# The actual prune+rewind schedule lives in train_imp_cycle.py:
#   * --target-sparsity 0.20 per cycle, 10 cycles → 89% final sparsity
#   * Standard CE+MSE renderer loss (no JBL / no KL distill auxiliary —
#     IMP's value comes from sparsity, not loss family).
IMP_CYCLE_DILATED_H64 = {
    **PROVEN_BASELINE,
    # IMP intentionally trains short per-cycle (200ep) so the lottery
    # ticket lottery has ~10 cycles within the GPU budget.
    "epochs": 200,
    "lr": 1e-4,
    "batch_size": 4,
    "loss_mode": "standard",
    # Different seed so IMP explores a different RNG basin from Lane A
    # (seed=42) while the prune+rewind schedule supplies the lane novelty.
    "seed": 89,
}


# ── LANE MAE-V: Masked Autoencoder Variant on input mask patches ──────────
# Council 2026-04-28 (Cosmos research synthesis,
# .omx/research/jack_skunkworks_segnet_rate_research_20260428.md).
#
# Anchored on Lane G v3 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
# Drops in `MAEMaskAugmenter` (src/tac/mae_mask_aug.py) BEFORE the renderer
# during training: a fraction of input mask patches are replaced with a
# Gumbel-softmax sample drawn from a learnable categorical "mask token"
# (5-class). This forces the renderer to reconstruct missing patches from
# context — a denoising / sparser-representation pressure analogous to He
# et al. 2021 (MAE for vision transformers) but adapted to our 5-class
# categorical mask vocabulary.
#
# Eval-mode is a passthrough (the augmenter only fires under .train()) so
# the inference distribution remains exactly what the contest scorer sees.
#
# Hyperparameters (per the module docstring + Cosmos research):
#   * mask_ratio = 0.25 (vs MAE's 0.75 — our renderer is small, denoising
#     signal only, not aggressive reconstruction)
#   * patch_size = 16 (matches typical MAE patch grid; 384×512 mask gives
#     24×32 = 768 patches per frame, ample sampling diversity)
#
# Predicted band [0.85, 1.10] per Cosmos synthesis. Composes with Lane SAUG-V2
# (orthogonal: SAUG perturbs renderer-input numerics; MAE-V perturbs mask
# patch occupancy). Composes with Lane J-JBL (orthogonal: JBL is a loss
# family, MAE-V is an input-distribution augmentation).
#
# CLAUDE.md compliance:
#   * eval_roundtrip remains True (inherited from Lane G v3 anchor).
#   * No scorer at inflate (augmentation is compress-time only, eval mode
#     is a strict passthrough).
#   * Auth eval at end (driven by remote_lane_mae_v.sh Stage 4 contest_auth_eval).
MAE_V_DILATED_H64 = {
    **DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,
    # Lane MAE-V switches: MAE patch augmentation on training input masks.
    "use_mae_mask_aug": True,
    "mae_mask_ratio": 0.25,
    "mae_patch_size": 16,
    # Different seed so MAE-V and Lane G v3 explore different RNG basins
    # (same convention as Lane J-JBL seed=47 vs Lane G v3 seed=43).
    "seed": 49,
}


# ── LANE V: Quantizr-replica 88K half-frame, joint-trained from epoch 0 ──
# Council 2026-04-27. Lane D tried to RETROFIT half-frame onto the dilated-h64
# baseline (mask_half_sim_prob=0.5 mid-train) and FAILED — joint training
# never converged because the renderer's MotionPredictor was already locked in
# to (e_t1 - e_t).abs() diff features that warp-expansion zeroes out (memory:
# feedback_half_frame_breaks_posenet, verified 17.55 score on the retrofit).
#
# Lane V's bet: train from epoch 0 with mask_half_sim_prob=1.0 (always-on
# half-frame) so the motion module NEVER sees the unwarped distribution and
# is forced to learn the warp-expansion premise. Combined with Quantizr's
# full trick stack (88K params, FiLM, KL distill T=2.0, DSConv) this is the
# biggest single swing in the council's strategy.
#
# Architecture (matches Quantizr's 88K target — Lane K-class capacity):
#   * base_ch=24, mid_ch=32, motion_hidden=16, depth=1
#   * embed_dim=6, pose_dim=6 (FiLM modulation, joint-trained from epoch 0)
#   * use_dsconv=True (depthwise-separable, Quantizr trick)
#   * use_zoom_flow=True (REQUIRED by preflight when mask_half_sim_prob>0;
#     motion outputs 4ch (gate+residual), flow comes from RadialZoomWarp)
#   * Measured param count: ~89.3K (matches the 88K Quantizr-class target)
#
# Half-frame paradigm:
#   * mask_half_sim_prob=1.0 — EVERY training batch warp-expands mask_t from
#     mask_t1. NO mixed-distribution training (vs Lane D's 0.5). Renderer
#     learns ONLY the inflate-time distribution from epoch 0.
#
# Quantizr trick stack:
#   * KL distillation T=2.0, weight=0.002 — POST-BUG-FIX MATH: the 2026-04-27
#     reduction fix in kl_distill_segnet_only (losses.py:705) divides by H*W
#     so raw KL ≈ 6.2e-3 × T² ≈ 0.025. With weight=0.002, KL contribution to
#     total loss is ~5e-5, ~1% of scorer loss (~0.005). PRE-FIX the same 1.0
#     weight that DEN/SHIRAZ/Lane D set was effectively running at 5000×
#     intended; 0.002 is the conservative post-fix landing.
#   * eval_roundtrip=True (NON-NEGOTIABLE per CLAUDE.md). noise_std=0.5 is
#     hardcoded in train_renderer.py:1741 (simulate_eval_roundtrip call) so
#     no separate resolver needed; documented here for transparency.
#   * posetto_noise_std=0.5 — passed to experiments/optimize_poses.py at
#     pose-TTO time (Stage 3 of remote_lane_v_*.sh).
#   * Standard Fridrich aux-loss stack (texture/L∞/Markov/variance/uncertainty)
#     mirrors Lane D for direct A/B comparability.
#
# 5-phase QAT schedule totalling 3000 epochs:
#   Phase 1 (600ep) anchor: pixel L1 + edge warmup
#   Phase 2 (1500ep) finetune: scorer + Fridrich + KL distill
#   Phase 3 (400ep) joint: hard-pair fine-tune
#   Phase 4 (400ep) QAT: FakeQuantFP4 enable
#   Phase 5 (100ep) final: consolidation at FP4
# Total ~3000 epochs ≈ 12h on RTX 4090 ($3.00 @ $0.25/hr) — longer than
# Lane K because half-frame is harder (renderer must learn warp-expansion
# from a less-informative training signal).
#
# Predicted score landing zone (standalone, no Lane A/C stacking):
#   * SegNet: ~0.003 (excellent at half-frame after joint training)
#   * PoseNet: 0.05-0.20 (the open question — JOINT training from epoch 0
#     should converge whereas RETROFIT did not)
#   * Rate: ~0.14-0.26 (half-frame archive saves 0.20-0.32 vs full-frame)
#   * Standalone score: 0.50-1.10 (wide band — true from-scratch rebuild)
# Stacked with Lane A pose TTO + Lane C δ: 0.30-0.55 (sub-Quantizr territory).
#
# Cost paranoia: $4-5 if including pose-TTO + auth-eval. Hard kill targets
# in scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh.
QUANTIZR_REPLICA_88K_HALFFRAME = {
    "experiment_type": "renderer_training",
    # Architecture: 88K param target, Quantizr-class (Lane K-equivalent).
    "variant": "dilated",
    "base_ch": 24,
    "mid_ch": 32,                  # "hidden_ch" in user spec maps to mid_ch
    "motion_hidden": 16,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,                 # FiLM modulation from epoch 0
    "use_dsconv": True,            # depthwise-separable (Quantizr trick)
    "padding_mode": "zeros",
    "use_dilation": False,
    # Half-frame paradigm — joint-trained from epoch 0 (vs Lane D retrofit 0.5).
    # use_zoom_flow=True is REQUIRED by preflight when mask_half_sim_prob>0.
    "use_zoom_flow": True,
    "mask_half_sim_prob": 1.0,     # ALWAYS-on warp-expansion (Lane V bet)
    "eval_roundtrip": True,        # NON-NEGOTIABLE per CLAUDE.md
    # noise_std=0.5 is hardcoded in train_renderer.py simulate_eval_roundtrip
    # call (line 1741); no separate resolver needed. Documented here for the
    # operator + the test that pins the contract.
    # posetto_noise_std=0.5 is consumed by experiments/optimize_poses.py at
    # pose-TTO stage; surfaced as profile metadata (no train_renderer resolver).
    "posetto_noise_std": 0.5,
    # Loss configuration — same Fridrich stack as Lane D for A/B comparability.
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",   # canonical name (user spec: seg_loss_mode)
    "hinge_margin": 0.5,           # canonical name (user spec: seg_margin)
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    # Score weights — SegNet dominates per scoring math (77x more important).
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    # Fridrich inverse-steganalysis aux losses (matches Lane D recipe).
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr KL distillation on SegNet — T=2.0, weight=0.002 POST-FIX.
    # Prior to losses.py:705 reduction fix, raw KL was ~5000× larger; weight
    # 1.0 was implicitly running at 5000× intended. Post-fix, raw KL ≈ 0.025
    # so weight=0.002 produces KL contribution ~5e-5 (~1% of scorer loss).
    "kl_distill_weight": 0.002,
    "kl_distill_temperature": 2.0,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # lane markings 15x (Yousfi)
    "use_swa": True,                # SWA for FP4 survival
    # No freeze/unfreeze — continuous adaptive training
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-phase QAT schedule — 3000 epochs total (matches user spec total_epochs).
    "phase1_epochs": 600,           # anchor: pixel L1 + edge warmup
    "phase2_epochs": 1500,          # finetune: scorer + Fridrich + KL
    "phase3_epochs": 400,           # joint: hard-pair fine-tune
    "phase4_epochs": 400,           # QAT: FakeQuantFP4 enable
    "phase5_epochs": 100,           # final: consolidation at FP4
    # User spec lr=5e-4 → use as phase2_lr (the "main" rate). Phase 1 anchor
    # uses 2x for warmup; Phase 4 QAT uses 0.1x per Lin et al. 2017.
    "lr": 5e-4,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,              # 0.1× base for QAT
    "phase5_lr": 1e-5,
    # User spec batch_size=8 → applied to all phases (phase1 was 16 in Lane D
    # but with 88K params and use_zoom_flow we have less VRAM headroom from
    # the always-on warp computation; 8 is the safe budget).
    "phase1_batch_size": 8,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # 5-stage quantization config — our advantage over Quantizr's vanilla.
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    # Mask augmentation: train on mixed CRF so we don't overfit one mask encoding
    "mask_noise_prob": 0.5,
    # Hard-frame curriculum
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility — pinned via configure_reproducibility().
    # User spec seed=1234 (Lane V uses a different seed from Lane D's 42 so
    # the two from-scratch rebuilds explore different RNG basins).
    "seed": 1234,
    "deterministic": True,
}


# ── LANE V-V2: Quantizr-replica 88K HALF-FRAME with ANNEALED warp prob ──
# Council 2026-04-27. Lane V-V1's mask_half_sim_prob=1.0 from epoch 0 was
# a cold-start: the renderer's randomly-initialised motion module had to
# learn the warp-expansion distribution from the very first batch, with
# no preview of the easier full-frame baseline. The hypothesis was that
# the renderer would converge anyway because (a) mask_half_sim_prob=0.5
# in Lane D's RETROFIT failed because the motion module was already locked
# in, and (b) starting from random would force convergence on the warp
# premise. That bet paid off (Lane V's predicted band was [0.50, 1.10])
# but the optimisation was high-variance.
#
# Lane V-V2 oversight fix: ANNEAL mask_half_sim_prob from 0.0 → 1.0 over
# training. The early epochs (warmup phase) train against the full-frame
# distribution where the gradient signal is strong + clear. As warmup
# completes, the warp probability ramps linearly to 1.0; the rest of
# training converges to the same Lane V endpoint but with a smoother
# trajectory.
#
# Annealing schedule (encoded in mask_half_sim_prob_anneal dict):
#   * 0.0 for first 30% of epochs (warmup on full-frame)
#   * Linear ramp 0.0 → 1.0 from 30% to 70% of epochs (transition)
#   * 1.0 for last 30% of epochs (Lane V endpoint distribution)
#
# Total epochs = sum of phase1..phase5 = 3000 (matches Lane V-V1):
#   * Warmup full-frame:  epoch 0..900     (900 epochs, 30%)
#   * Linear ramp:         epoch 900..2100  (1200 epochs, 40%)
#   * Lane V endpoint:    epoch 2100..3000 (900 epochs, 30%)
#
# Predicted band: [0.45, 1.05] [contest-CUDA].
#   * Slightly tighter than V1's [0.50, 1.10] because annealing removes
#     the early-epoch optimisation variance and lets the renderer track a
#     smoother loss landscape into the warp-expansion regime.
#   * Floor 0.45: optimal annealing finds a better local minimum than V1's
#     cold-start — the warmup full-frame phase establishes a strong
#     initialisation that survives the transition.
#   * Ceiling 1.05: annealing buys nothing — the warmup phase wasted 900
#     epochs on the wrong distribution; final convergence matches V1.
#
# All other knobs identical to QUANTIZR_REPLICA_88K_HALFFRAME so the only
# difference vs Lane V-V1 is the annealing schedule (single-variable A/B).
# Different seed (1235) so the two runs explore different RNG basins.
QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED = {
    **QUANTIZR_REPLICA_88K_HALFFRAME,  # inherit all knobs from V1
    # Override the cold-start prob with the annealing schedule. The training
    # loop reads mask_half_sim_prob_anneal first; if present, the per-epoch
    # warp probability is computed from the schedule and the static
    # mask_half_sim_prob value below is the END value (= 1.0). When the
    # schedule is missing, the static value is used unchanged (V1 path).
    "mask_half_sim_prob": 1.0,
    "mask_half_sim_prob_anneal": {
        "start_value": 0.0,
        "end_value": 1.0,
        "ramp_start_frac": 0.30,
        "ramp_end_frac": 0.70,
    },
    # Different seed so V1 vs V2 are not RNG-twins.
    "seed": 1235,
}


# ── Lane Q-FAITHFUL: TRUE 1:1 Quantizr PR #55 architecture replica ────────
# Council brief 2026-04-28. Sherlock audit
# (.omx/research/quantizr_replica_audit_20260428.md) proved that prior Lane V
# / Lane K / "DSConv Quantizr-killer" replicas all kept the wrong
# architectural family (motion module + warp + dual-mask). PR #55's actual
# architecture, per Quantizr's own description and verbatim from his
# inflate.py:198-223:
#
#   class JointFrameGenerator(nn.Module):
#       def forward(self, mask2, pose6):
#           shared_feat = self.shared_trunk(mask2, coords)
#           pred_frame2 = self.frame2_head(shared_feat)              # static
#           cond_emb = self.pose_mlp(pose6)
#           pred_frame1 = self.frame1_head(shared_feat, cond_emb)    # FiLM(pose)
#           return pred_frame1, pred_frame2
#
# NO motion module. NO warp. NO optical flow. NO dual-mask. The PR body
# verbatim: "dropping optical flow and using Feature-wise Linear Modulation
# on pose vectors instead of using both masks. As a result the mask video
# only needs to encode half as many frames."
#
# This profile points train_renderer.py at the new
# `tac.quantizr_faithful_renderer.build_quantizr_faithful_renderer` builder.
# variant="quantizr_faithful" — train_renderer dispatches to the new arm
# instead of the build_renderer() warp-based default.
#
# Predicted band [0.40, 0.80] [contest-CUDA]:
#   * Floor 0.40: Quantizr ships at 0.33; we use Lane A's scorer-measured
#     poses (load-bearing per `project_baseline_poses_load_bearing`) +
#     DSConv (matches Quantizr) + KL distill T=2.0 (matches) + 5-stage QAT.
#   * Anchor 0.55: matches Quantizr 0.33 ± 0.5 lane variance — first true
#     replica should land within his neighborhood.
#   * Ceiling 0.80: Lane A 1.15 minus rate gain. The 88K renderer (~64KB
#     FP4) saves ~225KB vs Lane A's 290KB renderer; at 25 bits/byte rate
#     factor that's a 0.30 rate reduction. Even if PoseNet/SegNet match
#     Lane A exactly, archive shrinks; ceiling is just rate-floored.
#
# Costs: ~$8 for 12h on RTX 4090. Single bet, no in-flight A/B.
LANE_Q_FAITHFUL_88K = {
    "experiment_type": "renderer_training",
    # Architecture dispatch flag — train_renderer must read this to pick the
    # JointFrameGenerator builder instead of build_renderer().
    "variant": "quantizr_faithful",
    # Standard arch keys are NOT consumed by the Q-FAITHFUL builder — its
    # config is hard-coded inside tac.quantizr_faithful_renderer.
    # JointFrameGenerator (c1=56, c2=64, cond_dim=48, depth_mult=1) per
    # Quantizr's exact PR-#55 spec. The values below satisfy preflight's
    # arch-key validator but are IGNORED at model construction time. The
    # logical mapping is documented in qf_* keys.
    "base_ch": 56,                 # IGNORED — maps logically to c1=56
    "mid_ch": 64,                  # IGNORED — maps logically to c2=64
    "depth": 1,                    # IGNORED — depth_mult=1 in Q-FAITHFUL
    "padding_mode": "zeros",       # IGNORED — DSConv uses k//2 padding
    "pose_dim": 6,                 # CONSULTED — FiLM input vector dim
    # Documentation-only target params (Quantizr's PR claims 88K; our
    # build_quantizr_faithful_renderer() lands at 87,836 which is within
    # the test's 86K-90K band). NOT a profile resolver — the real values
    # are hard-coded in tac.quantizr_faithful_renderer.JointFrameGenerator
    # (num_classes=5, cond_dim=48, depth_mult=1).
    # eval_roundtrip is NON-NEGOTIABLE per CLAUDE.md.
    "eval_roundtrip": True,
    "posetto_noise_std": 0.5,
    # Loss configuration: Quantizr's recipe per his compress.py:
    #   * KL distill on SegNet logits at T=2.0 — `kl_distill_segnet_only`
    #   * L1 pixel loss
    #   * scorer loss (PoseNet+SegNet) for the score-aligned signal
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 0.5,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 1.0,            # L1 on pixels — Quantizr's primary photometric
    # KL distill — POST-FIX weight (raw KL ~2.7, scorer ~0.05; weight 0.002
    # puts KL contribution at ~10% of scorer = canonical Hinton 2015).
    "kl_distill_weight": 0.002,
    "kl_distill_temperature": 2.0,
    # EMA decay 0.997 — Quantizr's 5-stage pipeline uses EMA for FP4 export.
    "ema_decay": 0.997,
    "use_per_class_weights": True,   # lane markings 15x (Yousfi)
    # 5-stage QAT pipeline: anchor -> finetune -> joint -> QAT -> final.
    # Same epoch budget as Lane V for cost parity (3000 epochs total).
    "phase1_epochs": 600,            # anchor: pixel L1 + scorer warmup
    "phase2_epochs": 1500,           # finetune: + KL distill
    "phase3_epochs": 400,            # joint: hard-pair fine-tune
    "phase4_epochs": 400,            # QAT: FakeQuantFP4 enable
    "phase5_epochs": 100,            # final: consolidation at FP4
    "lr": 5e-4,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,
    "phase5_lr": 1e-5,
    "phase1_batch_size": 8,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # 5-stage quantization (our advantage over Quantizr's vanilla):
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    # Hard-frame curriculum (matches Lane V).
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    "seed": 1234,
    "deterministic": True,
}


# ── Lane K: DSConv Quantizr-killer (88K params from-scratch, FULL-FRAME) ──
# Council brief 2026-04-27. Lane A holds the 1.15 [contest-CUDA] frontier with
# the dilated-h64 baseline arch (288K params, ~290KB FP32 renderer.bin, total
# archive 694KB, rate contribution 0.46). The remaining wedge is rate.
#
# Quantizr's 0.33 leader uses 88K params + DSConv (depthwise-separable
# convolutions) + FiLM conditioning. Lane S (Self-Compression) attacks rate
# via per-channel learnable bit-depth on the same 288K arch; Lane V attacks
# rate via a from-scratch 88K HALF-frame retrain. Lane K is the orthogonal
# bet to both: train a NEW renderer FROM SCRATCH at Quantizr-class capacity
# while shipping FULL-frame masks (no warp-expansion risk). The FP4 byte
# count drops from 180KB to ~50KB without going through the SC route AND
# without depending on the half-frame paradigm that broke Lane D.
#
# Architecture parity with Quantizr (verified empirically):
#   * base_ch=24, mid_ch=32, motion_hidden=16 → 88,996 params (matches
#     Quantizr's 88K target almost exactly). Estimated FP4 size: ~48KB
#     (vs Quantizr's ~64KB FP4-equivalent — we are smaller because our
#     embeddings + biases + FP4 block scales are tighter).
#   * embed_dim=6, depth=1 (single-scale U-Net like the dilated-h64 baseline)
#   * pose_dim=6 (FiLM modulation on poses from epoch 0 — full conditioning
#     from training start, NOT a post-hoc bolt-on. The pose_dim resolver is
#     verified live since commit 0746a803 fixed the dead-resolver class —
#     memory: feedback_pose_dim_dead_resolver. Past "FiLM didn't help"
#     conclusions on this arch class are invalid.)
#   * use_dsconv=True (depthwise-separable convolutions — the Quantizr
#     trick that lets us widen channels per layer at the same param budget).
#   * use_zoom_flow=False — uses the legacy PairGenerator path (MaskRenderer
#     + MotionPredictor with 6-channel motion output: flow + gate + residual).
#     The renderer_export FP4A path supports both pair_modes (verified at
#     renderer_export.py:_infer_asymmetric_config + inflate_renderer.py:1860
#     pair_mode dispatch). Lane K avoids use_zoom_flow because (a) the
#     RadialZoomWarp.warp_inverse_masks half-frame trick collapses PoseNet
#     on a from-scratch arch (memory: feedback_half_frame_breaks_posenet);
#     (b) Lane K's primary win is ARCH SIZE, not mask-byte rate, so we ship
#     full-frame masks and avoid compounding two un-validated changes.
#   * padding_mode='zeros' (matches the dilated-h64 baseline byte-for-byte
#     for the few layers where padding affects FP4 quantization scale).
#   * use_dilation=False (the dilation cascade is a depth=2 win — at
#     depth=1 it doesn't change receptive field meaningfully; we keep it
#     off so any score delta is attributable to DSConv + capacity reduction,
#     not a confound).
#
# All current best-practice training tricks are ON:
#   * eval_roundtrip=True (CLAUDE.md non-negotiable; without it the
#     proxy-auth gap can be 11x on PoseNet — memory:
#     feedback_proxy_auth_math_useless).
#   * noise_std=0.5 is hardcoded in train_renderer.py simulate_eval_roundtrip
#     call (line 1741); no separate resolver. Documented here for the
#     operator + the test that pins the contract.
#   * posetto_noise_std=0.5 consumed by experiments/optimize_poses.py
#     during pose-TTO; surfaced as profile metadata.
#   * segnet_loss_mode='hinge' with hinge_margin=0.5 (the user-specified
#     "seg_margin=0.5" maps to this — it's the hinge margin between the
#     correct class logit and the hardest competing class). Tighter than
#     WILDE/SHIRAZ/DEN's hinge_margin=1.0 because we have fewer params.
#   * Fridrich inverse-steganalysis losses ON (texture / L∞ / Markov +
#     wavelet variance noise + uncertainty). Same recipe as SHIRAZ — these
#     are scorer-arch-specific, not capacity-specific, so they should
#     transfer cleanly to the smaller arch.
#   * KL distill on SegNet (Quantizr's #1 SegNet trick), T=2.0, weight 0.002
#     POST losses.py:705 reduction fix (raw KL ~5000× larger pre-fix; weight
#     1.0 was implicitly running at 5000× intended).
#   * Per-class weights (lane markings 15x — Yousfi).
#   * SWA + EMA (wider minima → FP4 survives the post-training quant pass).
#
# 5-phase Quantizr-style schedule (matches Lane V budget for fair A/B):
#   Phase 1 (600ep): pixel L1+edge anchor warmup
#   Phase 2 (1500ep): scorer-guided + Fridrich aux losses + KL distill
#   Phase 3 (400ep): hard-pair fine-tune
#   Phase 4 (400ep): QAT (FakeQuantFP4 enabled, 0.1× LR)
#   Phase 5 (100ep): final consolidation at FP4
# Total ~3000 epochs ≈ 12h on RTX 4090. Cost: ~$3-4 at $0.25/hr.
#
# 5-stage quantization config — our advantage over Quantizr's vanilla 5-stage:
#   * fp4_codebook='residual' (denser-near-zero, 4× better small-magnitude
#     preservation — critical for an 88K renderer where every weight matters).
#   * fp4_robust_scale=True (per-block scale via p99.5 quantile).
#   * fp4_stochastic=True (unbiased dither during training; auto-disabled
#     in eval mode for inflate determinism).
#
# Mask augmentation: train against AV1-quantized masks so the renderer
# learns the inflate-time mask distribution.
#
# Predicted score landing zone [0.85, 1.10]:
#   - SegNet: ~0.005 (DSConv + capacity reduction may cost 0.001-0.002
#     vs dilated-h64 baseline's 0.003 — the 3.3× capacity penalty)
#   - PoseNet: ~0.04-0.10 (smaller motion module + FiLM-from-epoch-0 should
#     match or improve on dilated-h64 baseline's 0.247 PoseNet because FiLM
#     gives the renderer per-pair pose conditioning the baseline lacks)
#   - Rate: ~0.20-0.28 (renderer drops from 290KB → ~50KB FP4 → -0.18 rate;
#     full-frame masks + poses unchanged — anchored on Lane A's verified
#     1.15 [contest-CUDA] mask + pose payloads)
#   - Standalone score: 0.85-1.10. If it lands at 0.95 we BEAT Lane A's 1.15
#     by 0.20 points. If it lands at 0.85 we beat Lane A by 0.30 — the
#     orthogonal rate attack to Lane S's self-compression route.
#
# Deployment: scripts/remote_lane_k_dsconv_quantizr_killer.sh runs the full
# from-scratch pipeline (train → FP4A export → archive build → contest auth
# eval). Anchors masks + poses on Lane A's verified 1.15 artifacts so the
# only delta is the renderer byte count.
DSCONV_QUANTIZR_KILLER = {
    "experiment_type": "renderer_training",
    # Architecture: 88,996 params verified via build_renderer (Quantizr-class).
    "base_ch": 24,
    "mid_ch": 32,                  # "hidden_ch" in user spec maps to mid_ch
    "motion_hidden": 16,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,                 # FiLM modulation from epoch 0
    "use_dsconv": True,            # depthwise-separable (Quantizr trick)
    "padding_mode": "zeros",       # matches dilated-h64 baseline byte-for-byte
    "use_dilation": False,         # depth=1 dilation is a no-op for receptive field
    "use_zoom_flow": False,        # legacy PairGenerator path (full-frame masks)
    # Lane K is FULL-frame masks; explicit zero defends against any caller
    # flipping it on by accident (preflight enforces consistency between
    # mask_half_sim_prob and use_zoom_flow).
    "mask_half_sim_prob": 0.0,
    # CLAUDE.md non-negotiable: eval_roundtrip MUST be True on every training
    # path. Without it the proxy-auth gap can be 2-11x on PoseNet.
    "eval_roundtrip": True,
    # noise_std=0.5 is hardcoded in train_renderer.py simulate_eval_roundtrip
    # call (line 1741); documented here for the operator + the test contract.
    # posetto_noise_std=0.5 is consumed by experiments/optimize_poses.py at
    # pose-TTO stage; surfaced as profile metadata (no train_renderer resolver).
    "posetto_noise_std": 0.5,
    # Loss configuration — focal STE + hinge SegNet + Fridrich aux losses.
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",   # canonical name (user spec: seg_loss_mode)
    "hinge_margin": 0.5,           # canonical name (user spec: seg_margin)
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    # Score weights — DOMINATED BY SegNet (77x more important per scoring math).
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    # Fridrich inverse-steganalysis (full SHIRAZ stack — scorer-arch-specific,
    # transfers cleanly to the smaller arch).
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    # Yousfi #3: UNIWARD-aligned spatially-adaptive quant noise (wavelet_db4
    # mode per Holub & Fridrich 2014 §III.B).
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    # Yousfi #5: ScanNet-style spatial uncertainty maps (light weight).
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr KL distillation on SegNet — T=2.0, weight=0.002 POST losses.py:705
    # reduction fix (raw KL ~5000× larger pre-fix). Same calibration as Lane V.
    "kl_distill_weight": 0.002,
    "kl_distill_temperature": 2.0,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # lane markings 15x (Yousfi)
    "use_swa": True,                # SWA → wider minima → FP4 survives
    # NO freeze/unfreeze — continuous adaptive training.
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-phase QAT schedule — matches Lane V budget for fair A/B comparison.
    # Total ~3000 epochs ≈ 12h on 4090 ($3-4 at $0.25/hr).
    "phase1_epochs": 600,           # anchor: pixel L1 + edge warmup
    "phase2_epochs": 1500,          # finetune: scorer + Fridrich + KL
    "phase3_epochs": 400,           # joint: hard-pair fine-tune
    "phase4_epochs": 400,           # QAT: FakeQuantFP4 enable
    "phase5_epochs": 100,           # final: consolidation at FP4
    # User spec lr=5e-4 → use as phase2_lr (the "main" rate). Phase 1 anchor
    # uses 2× for warmup; Phase 4 QAT uses 0.1× per Lin et al. 2017.
    "lr": 5e-4,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,              # 0.1× base for QAT (Lin et al. 2017)
    "phase5_lr": 1e-5,
    # User spec batch_size=8 → applied to all phases. (Phase 1 was 16 in some
    # other profiles but with 88K params and the Fridrich aux loss stack we
    # stay at 8 for VRAM headroom on 24GB 4090.)
    "phase1_batch_size": 8,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # 5-stage quantization config — our advantage over Quantizr's vanilla.
    "fp4_codebook": "residual",     # denser-near-zero, 4× small-mag preservation
    "fp4_robust_scale": True,       # robust per-block scale (p99.5)
    "fp4_stochastic": True,         # stochastic rounding in QAT
    # Mask augmentation: train on mixed CRF (no overfitting to one encoding).
    "mask_noise_prob": 0.5,
    # Hard-frame curriculum carried from SHIRAZ.
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    # User-specified seed=1234 (matches build_baseline_archive.py default and
    # PYTHONHASHSEED in the bootstrap script for end-to-end determinism).
    "seed": 1234,
    "deterministic": True,
}


# ── Lane GH: Ghost-module renderer (Han et al. CVPR 2020) ─────────────
#
# Council brief 2026-04-27. Lane A holds 1.15 [contest-CUDA] frontier with
# the dilated-h64 baseline arch (288K params, ~290KB FP32 renderer.bin,
# total archive 694KB, rate contribution 0.46). Lane GH attacks the same
# rate wedge as Lane K (DSConv) but via Ghost convolutions instead of
# depthwise-separable.
#
# Why both Lane GH and Lane K? Council "no premature kills" rule:
# multiple plausible contenders → run them in parallel. DSConv and Ghost
# are different parameter-reduction primitives (DSConv = depthwise then
# pointwise; Ghost = primary conv to half-channels then cheap depthwise
# ghost branch concatenated). They produce DIFFERENT inductive biases:
#   - DSConv encourages channel-mixing to happen in 1x1 pointwise layers.
#   - Ghost encourages cheap linear redundancy in feature maps (Han et al.
#     CVPR 2020 §3.1: "the ghost feature maps are linear transformations
#     of the intrinsic feature maps").
# The score is the only valid arbiter (CLAUDE.md "multiple contenders →
# multiple paths" non-negotiable).
#
# Architecture: matches the verified dilated-h64 baseline byte-for-byte
# EXCEPT every bulk Conv2d in the renderer encoder (stem, down, down2)
# is swapped with GhostConv2d. ResBlocks keep standard Conv2d (their 3x3
# kernels are inside the residual; replacing them changes the inductive
# bias of the residual branch, not just its capacity).
#   * base_ch=36, mid_ch=60, motion_hidden=32 (dilated-h64 baseline)
#   * embed_dim=6, depth=1, pose_dim=6 (FiLM modulation from epoch 0)
#   * use_ghost=True, use_dsconv=False (mutually exclusive — pick one)
#   * use_zoom_flow=False (legacy PairGenerator path, full-frame masks)
#   * padding_mode='zeros', use_dilation=False (no confounds)
#
# Predicted parameter count: ~144K (288K halved by Ghost on the 3 bulk
# encoder convs + their depthwise ghost branches). FP4 renderer.bin
# target: ~75KB (vs Lane A's 180KB FP4-QAT, vs Quantizr's ~64KB).
#
# All current best-practice training tricks ON (mirrors DSCONV_QUANTIZR_KILLER):
#   * eval_roundtrip=True (CLAUDE.md non-negotiable).
#   * Fridrich inverse-steganalysis losses ON (texture / L∞ / Markov +
#     wavelet variance noise + uncertainty). Same recipe as SHIRAZ.
#   * KL distill on SegNet T=2.0, weight=0.002 (POST-FIX math).
#   * Per-class weights (lane markings 15x — Yousfi).
#   * SWA + EMA (wider minima → FP4 survives quant).
#
# 5-phase Quantizr-style schedule (matches Lane K budget for fair A/B):
#   Phase 1 (600ep): pixel L1+edge anchor warmup
#   Phase 2 (1500ep): scorer-guided + Fridrich aux losses + KL distill
#   Phase 3 (400ep): hard-pair fine-tune
#   Phase 4 (400ep): QAT (FakeQuantFP4 enabled, 0.1× LR)
#   Phase 5 (100ep): final consolidation at FP4
# Total ~3000 epochs ≈ 12h on RTX 4090. Cost: ~$3-4 at $0.25/hr.
#
# 5-stage quantization config (our advantage over Quantizr's vanilla):
#   * fp4_codebook='residual', fp4_robust_scale=True, fp4_stochastic=True
#
# Predicted score landing zone [1.05, 1.30]:
#   - SegNet: ~0.005 (Ghost may cost 0.001 vs dilated-h64 baseline 0.003)
#   - PoseNet: ~0.20-0.30 (Ghost preserves more channel-mixing capacity
#     than DSConv per Han et al. §4.3 ImageNet results — should match or
#     improve on dilated-h64 baseline's 0.247 PoseNet)
#   - Rate: ~0.30-0.40 (renderer drops from 290KB → ~75KB FP4 → -0.10
#     rate; full-frame masks + poses unchanged, anchored on Lane A)
#   - Standalone score: 1.05-1.30. If 1.10 we BEAT Lane A's 1.15 by 0.05;
#     if 1.05 by 0.10. Less aggressive than Lane K's [0.85, 1.10] band
#     because Ghost halves params (~144K) vs Lane K's ~3.3× cut to 88K —
#     more capacity preserved, more chance of matching baseline distortion.
#
# Deployment: scripts/remote_lane_gh_ghost_renderer.sh runs the full
# from-scratch pipeline (train → FP4A export → archive build → contest
# auth eval). Anchors masks + poses on Lane A's verified 1.15 artifacts.
DILATED_H64_GHOST = {
    "experiment_type": "renderer_training",
    "variant": "dilated",
    # Architecture: matches dilated-h64 baseline byte-for-byte EXCEPT
    # use_ghost=True. Param count ~144K vs baseline 288K.
    "base_ch": 36,
    "mid_ch": 60,
    "motion_hidden": 32,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,                 # FiLM modulation from epoch 0
    "use_dsconv": False,           # mutually exclusive with use_ghost
    "use_ghost": True,             # Lane GH: Ghost convolutions (CVPR 2020)
    "padding_mode": "zeros",       # baseline parity
    "use_dilation": False,         # depth=1 dilation no-op for receptive field
    "use_zoom_flow": False,        # legacy PairGenerator (full-frame masks)
    # Lane GH is FULL-frame masks. Explicit zero defends against any caller
    # flipping it on by accident (preflight enforces consistency between
    # mask_half_sim_prob and use_zoom_flow).
    "mask_half_sim_prob": 0.0,
    # CLAUDE.md non-negotiable: eval_roundtrip MUST be True on every
    # training path. Without it the proxy-auth gap can be 2-11x on PoseNet.
    "eval_roundtrip": True,
    # noise_std=0.5 hardcoded in train_renderer.py simulate_eval_roundtrip
    # call; documented here for the operator + the test contract.
    # posetto_noise_std=0.5 consumed by experiments/optimize_poses.py at
    # pose-TTO stage; surfaced as profile metadata.
    "posetto_noise_std": 0.5,
    # Loss configuration — focal STE + hinge SegNet + Fridrich aux losses.
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 0.5,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    # Score weights — DOMINATED BY SegNet (77x more important per scoring math)
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    # Fridrich inverse-steganalysis (full SHIRAZ stack — scorer-arch-specific,
    # transfers cleanly across capacity).
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    # Yousfi #3: UNIWARD-aligned spatially-adaptive quant noise (wavelet_db4)
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    # Yousfi #5: ScanNet-style spatial uncertainty maps (light weight).
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    # Quantizr KL distillation on SegNet (T=2.0, weight=0.002 POST-FIX).
    "kl_distill_weight": 0.002,
    "kl_distill_temperature": 2.0,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # lane markings 15x (Yousfi)
    "use_swa": True,                # SWA → wider minima → FP4 survives
    # NO freeze/unfreeze — continuous adaptive training.
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # 5-phase QAT schedule — matches Lane K budget for fair A/B comparison.
    "phase1_epochs": 600,           # anchor: pixel L1 + edge warmup
    "phase2_epochs": 1500,          # finetune: scorer + Fridrich + KL
    "phase3_epochs": 400,           # joint: hard-pair fine-tune
    "phase4_epochs": 400,           # QAT: FakeQuantFP4 enable
    "phase5_epochs": 100,           # final: consolidation at FP4
    "lr": 5e-4,
    "phase1_lr": 1e-3,
    "phase2_lr": 5e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,              # 0.1× base for QAT (Lin et al. 2017)
    "phase5_lr": 1e-5,
    "phase1_batch_size": 8,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # 5-stage quantization config — our advantage over Quantizr's vanilla.
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    # Mask augmentation: train on mixed CRF (no overfitting to one encoding).
    "mask_noise_prob": 0.5,
    # Hard-frame curriculum carried from SHIRAZ.
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Deterministic reproducibility (CLAUDE.md canonical pipeline standard).
    # seed=1234 matches Lane K + build_baseline_archive.py default.
    "seed": 1234,
    "deterministic": True,
}


# ── Lane S: Self-Compression renderer (Szabolcs / 2301.13142) ─────────
#
# Builds the dilated-h64 architecture but swaps every bulk Conv2d with a
# SelfCompressingConv2d (per-channel learnable bit-depth via STE) — the
# RGB head, motion head, FiLM linears, fuse_conv stay FP32 per Lane F's
# finding that those layers carry >50% of scorer sensitivity at <5% of
# the parameters.
#
# The Lagrangian rate penalty drives the average bit-depth from 8 (init)
# down to ~2.5 over training. Predicted byte count: ~16-20KB renderer.bin
# (vs 119-180KB FP4-QAT for the same arch). Stacked with full-frame
# masks (~125KB AV1) and poses (~15KB), total archive = ~160KB. Rate
# component drops from baseline 1.74 → ~0.11.
#
# Smoke profile: 100 epochs total, no scorer loss, just verifies the
# code path. Full profile mirrors DILATED_H64_HALF_FRAME for fair A/B.

SELF_COMPRESS_RENDERER_SMOKE = {
    "experiment_type": "renderer_training",
    "variant": "dilated",
    # Lane S engineering kwarg. Triggered by train_renderer.py to call
    # swap_renderer_convs_with_self_compress() after build_renderer.
    "use_self_compress_codec": True,
    "self_compress_init_bits": 8.0,
    "self_compress_target_bits": 4.0,    # smoke: aim wider so it converges fast
    "self_compress_lambda_start": 0.0,
    "self_compress_lambda_end": 0.5,
    "self_compress_lambda_ramp_start_frac": 0.3,
    # Architecture matches the verified dilated-h64 baseline byte-for-byte
    "base_ch": 36,
    "mid_ch": 60,
    "motion_hidden": 32,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,
    "use_dsconv": False,
    "padding_mode": "zeros",
    "use_dilation": False,
    "use_zoom_flow": True,
    "mask_half_sim_prob": 0.5,
    "eval_roundtrip": True,
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 1.0,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    # 5-phase short for smoke: 100 epochs total
    "phase1_epochs": 30,
    "phase2_epochs": 50,
    "phase3_epochs": 10,
    "phase4_epochs": 10,
    "phase5_epochs": 0,
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    "checkpoint_every": 25,
    "eval_every": 25,
    "log_every": 10,
    # Smoke does NOT auth-eval: the SC renderer.bin format is new and the
    # auth_eval_renderer.py round-trip via FP4A would fail. Use smoke for
    # code-path validation only.
    "auth_eval_on_best": False,
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    "mask_noise_prob": 0.5,
    "seed": 42,
    "deterministic": True,
}


# Full profile: 1980 epochs (matches SHIRAZ / DILATED_H64_HALF_FRAME budget
# for direct A/B comparability). Predicted score landing: with 16-20KB
# renderer.bin the rate drops by 0.18+ vs same-arch FP4 baseline. If
# PoseNet stays within 2x of baseline (Lane F's risk: protected layers
# may be insufficient), score lands in 0.85-1.10 range standalone. With
# Lane A pose TTO stack: 0.55-0.75. With aggressive rate (target_bits=2.0)
# and Lane I half-frame masks: sub-0.50 plausible.

SELF_COMPRESS_RENDERER_FULL = {
    "experiment_type": "renderer_training",
    "variant": "dilated",
    "use_self_compress_codec": True,
    # Bit budget: start at 8 (full precision), anneal toward 2.5 (Szabolcs
    # mean from 2301.13142 §4.3 with rate-distortion λ tuned for image
    # compression). The Lagrangian penalty ramps from 0 to 1.0 starting at
    # 30% of training so the renderer learns scorer-sensitive features
    # before being forced to compress them.
    "self_compress_init_bits": 8.0,
    "self_compress_target_bits": 2.5,
    "self_compress_lambda_start": 0.0,
    "self_compress_lambda_end": 1.0,
    "self_compress_lambda_ramp_start_frac": 0.3,
    # Architecture: identical to DILATED_H64_HALF_FRAME for byte-for-byte A/B
    "base_ch": 36,
    "mid_ch": 60,
    "motion_hidden": 32,
    "embed_dim": 6,
    "depth": 1,
    "pose_dim": 6,
    "use_dsconv": False,
    "padding_mode": "zeros",
    "use_dilation": False,
    "use_zoom_flow": True,
    "mask_half_sim_prob": 0.5,
    "eval_roundtrip": True,
    "loss_mode": "focal_ste",
    "segnet_loss_mode": "hinge",
    "hinge_margin": 1.0,
    "focal_gamma": 2.0,
    "error_boost": 1.0,
    "use_texture_loss": True,
    "texture_loss_weight": 0.5,
    "use_linf_penalty": True,
    "linf_weight": 0.01,
    "use_markov_loss": True,
    "markov_weight": 0.1,
    "use_variance_noise": True,
    "variance_noise_weight": 0.1,
    "variance_noise_base_std": 2.0,
    "variance_noise_kernel": 8,
    "variance_noise_mode": "wavelet_db4",
    "use_uncertainty_loss": True,
    "uncertainty_loss_weight": 0.05,
    "uncertainty_loss_floor": 0.1,
    "kl_distill_weight": 1.0,
    "kl_distill_temperature": 2.0,
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_per_class_weights": True,
    "use_swa": True,
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    "phase1_epochs": 400,
    "phase2_epochs": 1080,
    "phase3_epochs": 200,
    "phase4_epochs": 200,
    "phase5_epochs": 100,
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase4_lr": 5e-5,
    "phase5_lr": 1e-5,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
    # Lane S full requires auth eval — but only after the SCv1 inflate
    # path lands in production. Until then, the auth eval needs an FP4A
    # fallback export of the float weights for measurement only.
    "auth_eval_on_best": True,
    "fp4_codebook": "residual",
    "fp4_robust_scale": True,
    "fp4_stochastic": True,
    "mask_noise_prob": 0.5,
    "seed": 42,
    "deterministic": True,
}

WC_DILATED_H64 = {
    **SELF_COMPRESS_RENDERER_FULL,
    "pair_weights_path": "results/lane_wc/pair_weights.pt",
}


# ── Lane F-V5: Hardware FP8 (e4m3fn) on dilated-h64 ───────────────────────
#
# Lane F (FakeQuantFP4) regressed +0.44 vs baseline; FP4 is not hardware-
# supported on RTX 4090 (CC 8.9 < Blackwell CC 10.0). Lane F-V5 swaps for
# hardware-native FP8 (float8_e4m3fn) which IS supported on Ada/Lovelace.
# Memory: project_lane_f_v2_fp4_architectural_bottleneck_20260427.
#
# qat_warmup_batches drives HardwareFP8Quantizer calibration (50 batches of
# real activations before scales freeze).  loss_mode/quantization_mode are
# the discriminating fields the test asserts on.

F_V5_HARDWARE_FP8_DILATED_H64 = {
    **DILATED_H64_HALF_FRAME,
    # Lane F-V5 swaps the quantization story; everything else inherits the
    # baseline dilated-h64 arch + Fridrich + KL distill recipe.
    "loss_mode": "standard",
    "quantization_mode": "hardware_fp8",
    "qat_warmup_batches": 50,
    # ``hidden`` here names the renderer hidden width (64) — the "h64" in the
    # profile name. Distinct from ``motion_hidden`` (32) which is the motion
    # predictor's stem width. Tests assert on ``hidden``.
    "hidden": 64,
    # FP4 codebook knobs are not used here; keep them off so train_renderer
    # doesn't accidentally activate FakeQuantFP4 alongside FP8 calibration.
    "fp4_codebook": "default",
    "fp4_robust_scale": False,
    "fp4_stochastic": False,
}


PROFILES = {
    "council_v1": COUNCIL_V1,
    "council_v2_adaptive": COUNCIL_V2_ADAPTIVE,
    "segnet_attack": SEGNET_ATTACK,
    "proven_baseline": PROVEN_BASELINE,
    "ebr_dilated_h64": EBR_DILATED_H64,
    "t2_xpred": T2_XPRED,
    "t2_mask": T2_MASK,
    "h96_council": H96_COUNCIL,
    "smoke": SMOKE,
    "psd_standard_adaptive": PSD_STANDARD_ADAPTIVE,
    "saug_v2_dilated_h64": SAUG_V2_DILATED_H64,
    "hm_dilated_h64": HM_DILATED_H64,
    "cg_dilated_h64": CG_DILATED_H64,
    "pareto_pcgrad": PARETO_PCGRAD,
    "extreme_posenet": EXTREME_POSENET,
    "extreme_segnet": EXTREME_SEGNET,
    "reweight_ablation": REWEIGHT_ABLATION,
    "gated_dilated_smoke": GATED_DILATED_SMOKE,
    "dilated_h32_smoke": DILATED_H32_SMOKE,
    "dilated_h16_smoke": DILATED_H16_SMOKE,
    "kaggle_p100_dilated": KAGGLE_P100_DILATED,
    "kaggle_p100_long": KAGGLE_P100_LONG,
    # Renderer training profiles
    "wilde": WILDE,
    "green": GREEN,
    "shiraz": SHIRAZ,
    "den": DEN,
    # Lane D: dilated-h64 baseline arch retrained for half-frame masks
    # (warp-expansion injected into training data path so motion module
    # learns the inflate-time mask distribution).
    "dilated_h64_half_frame": DILATED_H64_HALF_FRAME,
    # Lane D-V3: same Lane D dilated-h64 half-frame retrain but stacks
    # V2's higher per-phase LR floor + Lane V-V2's annealed
    # mask_half_sim_prob (0.0 → 0.5 over 30%→70%) + post-fix KL distill
    # weight (1.0 → 0.002). Predicted band [1.50, 2.50] — wider than V2
    # because V3 introduces TWO new variables (annealing + KL fix) on
    # top of V2's LR fix.
    "dilated_h64_half_frame_v3_annealed_kldistill": (
        DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL
    ),
    # Lane H-V3 (forensic-audit revival): half-frame revival via JOINT
    # warp-expansion training. Anchored on Lane G v3 1.05; curriculum
    # mask_half_sim_prob 0.0 → 1.0 with aggressive 5%-15% ramp. Endpoint=1.0
    # matches inflate distribution (fixes Lane D-V3 mismatch). 288K dilated-h64
    # arch (NOT 88K DSConv) bypasses Lane V channel bug. Predicted band
    # [0.55, 0.95] [contest-CUDA] per project_lane_h_v3_revival_design_20260428.
    "h_v3_joint_halfframe": H_V3_JOINT_HALFFRAME,
    # Lane J-JBL (Jack Cycle 1 TOP-1): Jaccard Metric Loss + Boundary Label
    # Smoothing distillation, anchored on Lane G v3. Predicted band
    # [0.92, 1.02] per .omx/research/jack_skunkworks_segnet_rate_research_20260428.md §S1.
    "j_jbl_dilated_h64": J_JBL_DILATED_H64,
    # Lane J-IMP (Iterative Magnitude Pruning, Frankle & Carbin 2018):
    # outer driver experiments/train_imp_cycle.py runs 10 cycles of
    # prune+rewind+fine-tune to 89% sparsity. Profile is mostly a registry
    # entry; arch is overridden via CLI in
    # scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh.
    "imp_cycle_dilated_h64": IMP_CYCLE_DILATED_H64,
    # Lane MAE-V (Cosmos research synthesis): Masked Autoencoder Variant
    # — random patch masking with learnable Gumbel-softmax mask token
    # during training only (eval mode is a strict passthrough). Anchored
    # on Lane G v3 1.05 [contest-CUDA]. Predicted band [0.85, 1.10].
    "mae_v_dilated_h64": MAE_V_DILATED_H64,
    # Lane V: Quantizr-replica (88K params, DSConv + FiLM + KL distill) with
    # mask_half_sim_prob=1.0 (always-on warp expansion) joint-trained from
    # epoch 0. The biggest-swing council bet — vs Lane D's 0.5 retrofit which
    # FAILED, Lane V's bet is that JOINT training never seeing the unwarped
    # distribution forces convergence on the warp-expansion premise.
    "quantizr_replica_88k_halfframe": QUANTIZR_REPLICA_88K_HALFFRAME,
    # Lane V-V2: same Quantizr-replica 88K half-frame but with mask_half_sim_prob
    # ANNEALED 0.0 → 1.0 over training (warmup full-frame → linear ramp →
    # endpoint half-frame). Predicted band [0.45, 1.05] (tighter than V1
    # [0.50, 1.10] because annealing reduces optimisation variance).
    "quantizr_replica_88k_halfframe_annealed": (
        QUANTIZR_REPLICA_88K_HALFFRAME_ANNEALED
    ),
    # Lane Q-FAITHFUL: TRUE 1:1 Quantizr PR #55 architecture replica
    # (JointFrameGenerator, NO motion module, NO warp). The Sherlock audit
    # (.omx/research/quantizr_replica_audit_20260428.md) proved that Lane V
    # / Lane K kept the wrong architectural family. This is the corrected
    # rebuild. Predicted band [0.40, 0.80] [contest-CUDA].
    "q_faithful_dilated_88k": LANE_Q_FAITHFUL_88K,
    # Lane K: Quantizr-class 88K from-scratch retrain on FULL-frame masks.
    # Orthogonal to Lane V (half-frame) and Lane S (self-compression). Same
    # arch (88,996 params, DSConv + FiLM + Fridrich + KL distill) but ships
    # full-frame masks anchored on Lane A's verified 1.15 [contest-CUDA]
    # mask + pose payloads — no half-frame risk. Predicted band [0.85, 1.10].
    "dsconv_quantizr_killer": DSCONV_QUANTIZR_KILLER,
    # Lane GH: Ghost-module renderer (Han et al. CVPR 2020). Mirrors the
    # dilated-h64 baseline arch but swaps bulk Conv2d for GhostConv2d,
    # halving renderer params (~144K vs 288K). Orthogonal to Lane K (DSConv,
    # 88K) — different parameter-reduction primitive, different inductive
    # bias. Predicted band [1.05, 1.30] — less aggressive than Lane K
    # because more capacity is preserved.
    "dilated_h64_ghost": DILATED_H64_GHOST,
    "wilde_v2": WILDE_V2,
    "green_v2": GREEN_V2,
    "shiraz_v2": SHIRAZ_V2,
    "definitive_float_ema": DEFINITIVE_FLOAT_EMA,
    "mask_renderer_smoke": MASK_RENDERER_SMOKE,
    "mask_renderer_full": MASK_RENDERER_FULL,
    "mask_renderer_wide": MASK_RENDERER_WIDE,
    "mask_renderer_deep": MASK_RENDERER_DEEP,
    "wavelet_renderer_smoke": WAVELET_RENDERER_SMOKE,
    "wavelet_renderer_full": WAVELET_RENDERER_FULL,
    "diffusion_teacher_smoke": DIFFUSION_TEACHER_SMOKE,
    "diffusion_teacher_full": DIFFUSION_TEACHER_FULL,
    "distillation_smoke": DISTILLATION_SMOKE,
    "distillation_full": DISTILLATION_FULL,
    "dp_sims_smoke": DP_SIMS_SMOKE,
    "dp_sims_full": DP_SIMS_FULL,
    "dp_sims_small_smoke": DP_SIMS_SMALL_SMOKE,
    "dp_sims_small_full": DP_SIMS_SMALL_FULL,
    "dp_sims_v2_smoke": DP_SIMS_V2_SMOKE,
    "dp_sims_v2_full": DP_SIMS_V2_FULL,
    "vqvae_smoke": VQVAE_SMOKE,
    "vqvae_full": VQVAE_FULL,
    "vqvae_compact": VQVAE_COMPACT,
    # Cross-cultural research techniques
    "depthwise_renderer_smoke": DEPTHWISE_RENDERER_SMOKE,
    "channel_recurrent_smoke": CHANNEL_RECURRENT_SMOKE,
    "coord_renderer_smoke": COORD_RENDERER_SMOKE,
    "coolchic_renderer_smoke": COOLCHIC_RENDERER_SMOKE,
    "coolchic_renderer_full": COOLCHIC_RENDERER_FULL,
    "c3_residual_renderer_smoke": C3_RESIDUAL_RENDERER_SMOKE,
    "c3_residual_renderer_full": C3_RESIDUAL_RENDERER_FULL,
    # Yousfi CPU lane profiles
    "proven_baseline_boundary": PROVEN_BASELINE_BOUNDARY,
    "proven_baseline_vp_saliency": PROVEN_BASELINE_VP_SALIENCY,
    "proven_baseline_full": PROVEN_BASELINE_FULL,
    "proven_baseline_featmatch": PROVEN_BASELINE_FEATMATCH,
    # Yousfi council decode profiles
    "overfit_cpu": OVERFIT_CPU,
    "overfit_cpu_v2": OVERFIT_CPU_V2,
    "overfit_gpu": OVERFIT_GPU,
    # Technique 2: Luma-only dilated
    "luma_only_dilated_smoke": LUMA_ONLY_DILATED_SMOKE,
    "luma_only_dilated_full": LUMA_ONLY_DILATED_FULL,
    # Technique 3: Content-adaptive
    "content_adaptive_smoke": CONTENT_ADAPTIVE_SMOKE,
    # Technique 4: PixelShuffle upscale
    "pixelshuffle_upscale_smoke": PIXELSHUFFLE_UPSCALE_SMOKE,
    # Technique 9: DualHead
    "dual_head_smoke": DUAL_HEAD_SMOKE,
    "dual_head_full": DUAL_HEAD_FULL,
    # Technique 10: Focal gamma 4-5
    "focal_gamma_4": FOCAL_GAMMA_4,
    "focal_gamma_5": FOCAL_GAMMA_5,
    "focal_gamma_4_smoke": FOCAL_GAMMA_4_SMOKE,
    "focal_gamma_5_smoke": FOCAL_GAMMA_5_SMOKE,
    # Migrated architecture smoke profiles
    "dct_midband_smoke": DCT_MIDBAND_SMOKE,
    "film_conditioned_smoke": FILM_CONDITIONED_SMOKE,
    "counterpoint_smoke": COUNTERPOINT_SMOKE,
    "pixelshuffle_dilated_smoke": PIXELSHUFFLE_DILATED_SMOKE,
    "uint8_ste_smoke": UINT8_STE_SMOKE,
    # Migrated legacy loss technique profiles
    "segnet_kl_smoke": SEGNET_KL_SMOKE,
    "segnet_kl_full": SEGNET_KL_FULL,
    "posenet_embedding_smoke": POSENET_EMBEDDING_SMOKE,
    "posenet_embedding_full": POSENET_EMBEDDING_FULL,
    "counterpoint_losses_smoke": COUNTERPOINT_LOSSES_SMOKE,
    "kalman_baseline": KALMAN_BASELINE,
    # Trick stacking inflate-time profiles
    "stacked_inflate_full": STACKED_INFLATE_FULL,
    "stacked_inflate_safe": STACKED_INFLATE_SAFE,
    "stacked_inflate_fast": STACKED_INFLATE_FAST,
    # Constrained optimization from noise (Yousfi GPU breakthrough)
    "constrained_gen_smoke": CONSTRAINED_GEN_SMOKE,
    "constrained_gen_full": CONSTRAINED_GEN_FULL,
    # Variational + Lagrangian + manifold + Hamiltonian profiles
    "variational_smoke": VARIATIONAL_SMOKE,
    "lagrangian_dual_smoke": LAGRANGIAN_DUAL_SMOKE,
    "pareto_trace": PARETO_TRACE,
    "constrained_gen_full_pipeline": CONSTRAINED_GEN_FULL_PIPELINE,
    # Cross-disciplinary optimizer profiles
    "cross_disc_smoke": CROSS_DISC_SMOKE,
    "cross_disc_ensemble": CROSS_DISC_ENSEMBLE,
    "cross_disc_annealing": CROSS_DISC_ANNEALING,
    "cross_disc_evolutionary": CROSS_DISC_EVOLUTIONARY,
    "cross_disc_physics": CROSS_DISC_PHYSICS,
    "cross_disc_geophysics": CROSS_DISC_GEOPHYSICS,
    # Finance & HFT optimizer profiles
    "finance_smoke": FINANCE_SMOKE,
    "finance_ensemble": FINANCE_ENSEMBLE,
    "finance_hft": FINANCE_HFT,
    # Domain-specific solver profiles (Yousfi cross-domain toolkit)
    "domain_smoke": DOMAIN_SMOKE,
    "domain_driving": DOMAIN_DRIVING,
    "domain_signal": DOMAIN_SIGNAL,
    "domain_full": DOMAIN_FULL,
    # Eureka constrained optimization profiles
    "coupled_trajectory_smoke": COUPLED_TRAJECTORY_SMOKE,
    "alternating_projections_smoke": ALTERNATING_PROJECTIONS_SMOKE,
    "newton_step_smoke": NEWTON_STEP_SMOKE,
    "shannon_compressor_smoke": SHANNON_COMPRESSOR_SMOKE,
    # Newton-CG quadratic optimizer (geometry deliberation)
    "newton_quadratic_smoke": NEWTON_QUADRATIC_SMOKE,
    "newton_quadratic_full": NEWTON_QUADRATIC_FULL,
    # Fridrich steganalysis-inspired profiles
    "fridrich_smoke": FRIDRICH_SMOKE,
    "fridrich_full": FRIDRICH_FULL,
    "fridrich_cpu_postfilter": FRIDRICH_CPU_POSTFILTER,
    # GPU Lane: Full pipeline (coupled trajectory + Fridrich)
    "gpu_lane_smoke": GPU_LANE_SMOKE,
    "gpu_lane_full": GPU_LANE_FULL,
    # Technique 1: Self-compressing postfilter (Szabolcs)
    "self_compress_smoke": SELF_COMPRESS_SMOKE,
    # Lane S: Self-compressing renderer (Szabolcs 2301.13142 in renderer)
    "self_compress_renderer_smoke": SELF_COMPRESS_RENDERER_SMOKE,
    "self_compress_renderer_full": SELF_COMPRESS_RENDERER_FULL,
    # Lane WC: Lane W-style per-pair weighting from independent SegNet
    # feature geometry, not circular pair-loss difficulty.
    "wc_dilated_h64": WC_DILATED_H64,
    # Lane F-V5: hardware FP8 (e4m3fn) replacing simulated FP4
    "f_v5_hardware_fp8_dilated_h64": F_V5_HARDWARE_FP8_DILATED_H64,
    # Technique 2: Entropy-coded archive (Shannon)
    "entropy_archive_smoke": ENTROPY_ARCHIVE_SMOKE,
    # Technique 3: Network-as-codec (SIREN video memorization)
    "network_codec_smoke": NETWORK_CODEC_SMOKE,
    "network_codec_full": NETWORK_CODEC_FULL,
}

# Deprecated profile names that should emit runtime warnings
_DEPRECATED_PROFILES = {
    "council_v2_adaptive": (
        "council_v2_adaptive is DEPRECATED: adaptive_rebalance is dead. "
        "The Hinton T² correction double-corrected (see CLAUDE.md). "
        "Use 'proven_baseline' or 'council_v1' instead."
    ),
}


def get_profile(name: str) -> dict:
    """Look up a named profile, emitting warnings for deprecated profiles.

    Args:
        name: profile name (key in PROFILES dict).

    Returns:
        Profile dict of TrainConfig overrides.

    Raises:
        KeyError: if profile name is not found.
    """
    if name not in PROFILES:
        raise KeyError(
            f"Unknown profile '{name}'. Available: {sorted(PROFILES.keys())}"
        )
    if name in _DEPRECATED_PROFILES:
        import warnings
        warnings.warn(_DEPRECATED_PROFILES[name], DeprecationWarning, stacklevel=2)
    return PROFILES[name]
