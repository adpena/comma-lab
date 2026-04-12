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
    "epochs": 1000,  # target: ~250-400 epochs complete in 11h on GPU
    "eval_every": 10,  # less frequent eval to save time
    "wall_clock_timeout": 39600,  # 11h in seconds (12h Kaggle limit - 1h safety)
}

# Kaggle P100 long run: aggressive 2500 epochs, relies on timeout to stop cleanly.
# Use when you want maximum training time and the timeout handles the rest.
KAGGLE_P100_LONG = {
    **PROVEN_BASELINE,
    "epochs": 2500,
    "eval_every": 10,
    "wall_clock_timeout": 39600,  # 11h safety margin
}

# ── GPU Lane: Mask Renderer profiles ────────────────────────────────────
# Segment → Compress Masks → Neural Render pipeline.
# Instead of postfilter on compressed video, render frames from masks.
# Score formula is the same: 100*seg + sqrt(10*pose).

MASK_RENDERER_SMOKE = {
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

# ── Yousfi Council Decode: Aggressive Overfitting ────────────────────
# CPU lane: overfit postfilter to 0.mkv with all tricks (10K epochs)
OVERFIT_CPU = {
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
    "use_tto": True,
    "tto_steps": 10,
    "tto_lr": 1e-4,
    "tto_loss": "temporal_consistency",
    "use_supervised_tto": False,
    "use_multi_pass": 3,
    "use_brightness_shift": False,
    "use_chroma_exploit": False,
    "use_fragility_weighting": False,
    "use_noise_shaping": False,
    "use_backward_delta_smoothing": False,
    "use_null_space_projection": False,
}

# Fast stacking: minimal overhead, just multi-pass + fast noise shaping
STACKED_INFLATE_FAST = {
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

CONSTRAINED_GEN_FULL = {
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
    # Yousfi CPU lane profiles
    "proven_baseline_boundary": PROVEN_BASELINE_BOUNDARY,
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
}
