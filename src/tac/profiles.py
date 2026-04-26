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
    "quantize_mode": "fp4",  # use FP4 instead of int8
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "use_uint8_ste": True,  # flag for Trainer to apply Uint8STE after model forward
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
    "quantize_mode": "fp4",
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
    "quantize_mode": "fp4",
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
    "finance_optimizer": "risk_parity",
    "finance_config": {"base_lr": 0.5, "ema_decay": 0.95},
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
    "finance_optimizer": "ensemble",
    "finance_config": {
        "optimizers": [
            {"name": "risk_parity", "weight": 2.0, "config": {"base_lr": 1.0}},
            {"name": "order_book", "weight": 1.5, "config": {"base_lr": 1.0, "top_fraction": 0.3}},
            {"name": "implied_vol", "weight": 1.0, "config": {"base_lr": 1.0, "power": 0.5}},
        ],
        "blend_mode": "weighted_average",
    },
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
    "finance_optimizer": "ensemble",
    "finance_config": {
        "optimizers": [
            {"name": "order_book", "weight": 2.0, "config": {"base_lr": 0.5, "top_fraction": 0.4}},
            {"name": "almgren_chriss", "weight": 1.0, "config": {"total_steps": 500, "base_lr": 0.5}},
            {"name": "momentum_reversion", "weight": 1.0, "config": {"base_lr": 0.5}},
        ],
        "blend_mode": "round_robin",
    },
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
    "variational_steps": 50,
    "variational_lr": 0.5,
    "lambda_smooth": 0.01,
    "lambda_rate": 0.1,
    "lambda_grad": 1.0,
    "lambda_lap": 0.1,
    "use_line_search": False,
    "grad_clip": 10.0,
    "convergence_tol": 1e-6,
}

# ── Lagrangian dual rate-distortion optimizer ─────────────────────────
# Primal-dual method: min D(f) s.t. R(f) <= budget
# Lambda is LEARNED via dual ascent (KKT optimality).
LAGRANGIAN_DUAL_SMOKE = {
    "experiment_type": "gpu_lane",
    "variant": "lagrangian_dual",
    "dual_steps": 100,
    "dual_primal_lr": 0.5,
    "dual_dual_lr": 0.01,
    "rate_budget": 0.01,
    "lambda_init": 25.0,
    "lambda_smooth": 0.01,
    "kkt_tol": 1e-4,
    "grad_clip": 10.0,
}

# ── Pareto frontier tracer ────────────────────────────────────────────
# Sweep rate budget to trace the full (seg, pose, rate) Pareto frontier.
# Each point is a Lagrangian dual solution with different rate constraint.
PARETO_TRACE = {
    "experiment_type": "gpu_lane",
    "variant": "pareto_trace",
    "num_pareto_points": 10,
    "rate_range_min": 0.001,
    "rate_range_max": 0.05,
    "dual_steps": 200,
    "dual_primal_lr": 0.3,
    "dual_dual_lr": 0.01,
    "lambda_init": 25.0,
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
    "optimizer_names": [
        "simulated_annealing", "hmc", "langevin", "replica_exchange",
        "cma_es", "differential_evolution", "pso",
        "metadynamics", "basin_hopping",
        "fwi", "seismic_multigrid",
        "enkf", "4dvar",
        "nested_sampling", "multigrid_relaxation",
        "quantum_annealing",
    ],
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
    "optimizer_names": [
        "simulated_annealing", "basin_hopping", "cma_es",
        "langevin", "4dvar", "fwi",
    ],
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
    "optimizer_names": ["simulated_annealing", "basin_hopping"],
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
    "optimizer_names": ["cma_es", "differential_evolution"],
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
    "optimizer_names": ["hmc", "langevin", "replica_exchange"],
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
    "optimizer_names": ["fwi", "seismic_multigrid", "4dvar"],
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
    "solver_names": [
        "ego_motion_flow", "road_plane_homography", "vanishing_point",
        "matched_filter", "kalman_smoother", "compressed_sensing",
        "trajectory", "adaptive_optics", "ofdm", "turbo_scorer",
    ],
    "ego_motion_config": {"num_steps": 5},
    "road_plane_config": {"num_steps": 5},
    "vanishing_point_config": {"vp_weight": 1.0},
    "matched_filter_config": {"num_iterations": 5, "max_jacobian_outputs": 8},
    "kalman_config": {"pca_components": 10, "num_iterations": 1},
    "compressed_sensing_config": {"num_iterations": 5, "sparsity_lambda": 0.01},
    "trajectory_config": {"num_shooting_iterations": 5},
    "adaptive_optics_config": {"num_modes": 10, "num_iterations": 5},
    "ofdm_config": {"num_frequencies": 32, "num_iterations": 5},
    "turbo_config": {"num_turbo_iterations": 2, "pose_steps": 3, "seg_steps": 3},
}

# Self-driving domain solvers only (ego-motion + road plane + vanishing point)
DOMAIN_DRIVING = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
    "solver_names": ["ego_motion_flow", "road_plane_homography", "vanishing_point"],
    "ego_motion_config": {
        "focal_length": 910.0,
        "principal_point": [256.0, 192.0],
        "num_steps": 100,
        "flow_weight": 10.0,
    },
    "road_plane_config": {
        "camera_height": 1.2,
        "camera_pitch": 0.02,
        "num_steps": 50,
    },
    "vanishing_point_config": {
        "vp_weight": 2.0,
        "min_line_length": 20,
        "angular_tolerance": 0.15,
    },
}

# Signal processing solvers (matched filter + compressed sensing + OFDM)
DOMAIN_SIGNAL = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
    "solver_names": ["matched_filter", "compressed_sensing", "ofdm"],
    "matched_filter_config": {
        "regularization_lambda": 1e-3,
        "step_size": 0.1,
        "num_iterations": 20,
    },
    "compressed_sensing_config": {
        "sparsity_lambda": 0.01,
        "num_iterations": 50,
        "wavelet_type": "haar",
    },
    "ofdm_config": {
        "num_frequencies": 256,
        "water_fill_power_budget": 1e6,
        "sensitivity_threshold": 0.01,
    },
}

# Full ensemble: all 10 solvers with production settings
DOMAIN_FULL = {
    "experiment_type": "gpu_lane",
    "variant": "domain_solver",
    "solver_names": [
        "ego_motion_flow", "road_plane_homography", "vanishing_point",
        "matched_filter", "kalman_smoother", "compressed_sensing",
        "trajectory", "adaptive_optics", "ofdm", "turbo_scorer",
    ],
    "ego_motion_config": {"num_steps": 100, "focal_length": 910.0},
    "road_plane_config": {"camera_height": 1.2, "num_steps": 50},
    "vanishing_point_config": {"vp_weight": 2.0},
    "matched_filter_config": {"num_iterations": 20, "step_size": 0.1},
    "kalman_config": {"pca_components": 50, "num_iterations": 3},
    "compressed_sensing_config": {"num_iterations": 50, "sparsity_lambda": 0.01},
    "trajectory_config": {"num_shooting_iterations": 30, "control_penalty": 0.01},
    "adaptive_optics_config": {"num_modes": 50, "num_iterations": 100},
    "ofdm_config": {"num_frequencies": 256},
    "turbo_config": {"num_turbo_iterations": 10, "damping_factor": 0.5},
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
    "newton_refine": True,
    "newton_max_steps": 5,
    "newton_cg_iters": 3,
    "newton_trust_radius": 0.05,
    "newton_max_step_norm": 0.5,
    "newton_top_k_pairs": 10,  # only refine hardest 10 pairs
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
    "newton_refine": True,
    "newton_max_steps": 10,
    "newton_cg_iters": 5,
    "newton_trust_radius": 0.05,
    "newton_max_step_norm": 0.5,
    "newton_top_k_pairs": 60,  # refine top 10% of 600 pairs
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
    "fridrich_cost_method": "uniward",  # fast, no Jacobian computation
    "fridrich_num_probes": 5,
    "fridrich_max_magnitude": 10.0,
    "fridrich_opt_steps": 50,
    "fridrich_opt_lr": 0.05,
    "fridrich_rate_reduction": 0.1,
    "fridrich_subsample_frames": 4,
    "fridrich_jacobian_probes": 2,
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
    "fridrich_cost_method": "hybrid",
    "fridrich_num_probes": 20,
    "fridrich_max_magnitude": 30.0,
    "fridrich_opt_steps": 500,
    "fridrich_opt_lr": 0.01,
    "fridrich_rate_reduction": 0.1,
    "fridrich_subsample_frames": 1,
    "fridrich_jacobian_probes": 8,
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
    "fridrich_cost_method": "hybrid",
    "fridrich_num_probes": 20,
    "fridrich_max_magnitude": 30.0,
    "fridrich_boundary_fraction": 0.8,
    "fridrich_rate_reduction": 0.1,
    "fridrich_subsample_frames": 2,
    "fridrich_jacobian_probes": 4,
    # Skip heavy GPU optimization, rely on postfilter + cost weighting
    "fridrich_skip_optimize": True,
    "fridrich_skip_stc": False,
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
    "entropy_context_size": 4,
    "entropy_hidden": 16,
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
    "use_self_compress_codec": False,
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
    "use_self_compress_codec": True,
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
    "phase1_epochs": 600,  # Pixel warmup (200) + anchor (400)
    "phase2_epochs": 880,  # Anchor boost (80) + joint (800)
    "phase3_epochs": 200,  # Hard-pair fine-tune
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
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
    "pose_weight": 10.0,
    "seg_weight": 100.0,
    "pixel_weight": 0.1,
    "ema_decay": 0.997,
    "use_per_class_weights": True,  # Lane markings 15x (Yousfi: 1.2% but critical)
    "use_swa": True,               # Wider minima, better FP4 survival (Polyak 1992)
    # NO freeze/unfreeze — continuous adaptive training
    "freeze_motion_phase2": False,
    "freeze_renderer_phase3": False,
    # Training schedule (same total epochs as WILDE for fair comparison)
    "phase1_epochs": 400,           # pixel warmup
    "phase2_epochs": 1080,          # scorer-guided (continuous, no anchor/boost split)
    "phase3_epochs": 200,           # hard-pair fine-tune
    "phase1_lr": 1e-3,
    "phase2_lr": 3e-4,
    "phase3_lr": 1e-4,
    "phase1_batch_size": 16,
    "phase2_batch_size": 8,
    "phase3_batch_size": 8,
    # Hard-frame curriculum
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 50,
    "log_every": 25,
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
    "use_zoom_flow": True,          # GREEN-style radial zoom flow (Hotz)
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
    "phase4_batch_size": 8,
    "phase5_batch_size": 8,
    # 5-stage quantization config — our advantage over Quantizr's vanilla.
    "fp4_codebook": "residual",     # RESIDUAL codebook beats uniform
    "fp4_robust_scale": True,       # robust per-channel scaling
    "fp4_stochastic": True,         # stochastic rounding in QAT
    # Mask augmentation: train on mixed CRF so we don't overfit to one mask
    # encoding. Tested at compress time on whatever Lane A0 picks.
    "mask_noise_prob": 0.5,         # 50% of training pairs use augmented mask
    # Hard-frame curriculum carried from SHIRAZ.
    "hard_frame_ratio": 0.3,
    "error_replay_every": 100,
    "checkpoint_every": 100,
    "eval_every": 100,              # auth-eval-style score at every checkpoint
    "log_every": 25,
}


PROFILES = {
    "council_v1": COUNCIL_V1,
    "council_v2_adaptive": COUNCIL_V2_ADAPTIVE,
    "segnet_attack": SEGNET_ATTACK,
    "proven_baseline": PROVEN_BASELINE,
    "h96_council": H96_COUNCIL,
    "smoke": SMOKE,
    "psd_standard_adaptive": PSD_STANDARD_ADAPTIVE,
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
