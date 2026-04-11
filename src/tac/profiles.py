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
    "loss_mode": "standard",          # was "kl_distill" — deprecated, kills PoseNet
    # KL distill fields kept as no-ops (ignored by standard loss_mode):
    # "temperature_start": 5.0,
    # "temperature_end": 0.5,
    # "temp_schedule": "exponential",
    "boundary_weight": 5.0,
    "boundary_anneal": True,          # couple bw to temperature schedule
    "hard_frame_ratio": 0.3,         # power-law curriculum, 0.3 = moderate emphasis
    "error_replay_every": 200,        # recompute hard frames using model output
    "eval_every": 5,                  # skip eval on 4/5 epochs (ramps to 1 in final 10%)
    "accum_steps": 4,
    "segnet_loss_weight": 30.0,
    "use_swa": True,                  # SWA over final 20% for wider minima (better int8)
}

# Aggressive SegNet-focused (contrarian + DeepSeek recommendation)
# Inherits loss_mode="standard" from COUNCIL_V1 (kl_distill deprecated)
SEGNET_ATTACK = {
    **COUNCIL_V1,
    "boundary_weight": 200.0,         # near-maximum boundary focus
    "hard_frame_ratio": 0.5,          # stronger hard-frame emphasis
    "error_replay_every": 100,        # more frequent curriculum adaptation
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
    "segnet_loss_weight": 30.0,       # placeholder — overridden by AdaptiveWeights at runtime
    "boundary_weight": 100.0,         # placeholder — overridden by AdaptiveWeights at runtime
    "boundary_anneal": False,         # disabled: adaptive rebalance handles weight scheduling
    "adaptive_rebalance": True,       # flag for Trainer to invoke AdaptiveWeights.rebalance()
    "rebalance_every": 50,            # epochs between adaptive weight updates
    "boundary_fraction": 0.05,        # measured beta for AdaptiveWeights init
    "use_lsq": True,                  # LSQ: learned step sizes via forward pre-hooks on Conv2d
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
    "loss_mode": "pcgrad",              # gradient surgery: no SegNet gradient harms PoseNet
    "segnet_loss_weight": 30.0,         # initial — overridden by adaptive MRS at first eval
    "boundary_weight": 50.0,
    "hard_frame_ratio": 0.3,
    "error_replay_every": 200,
    "eval_every": 5,
    "accum_steps": 4,
    "adaptive_rebalance": True,         # MRS-adaptive: w_seg = 200*sqrt(10*pose)
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
    "alpha": 30.0,                      # higher saliency weight = more PoseNet focus
    "sal_lambda": 1.5,
    "loss_mode": "standard",
    "segnet_loss_weight": 0.0,          # zero SegNet weight — pure PoseNet optimization
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
    "alpha": 5.0,                       # low saliency = less PoseNet-biased correction
    "sal_lambda": 0.5,
    "loss_mode": "focal_ste",           # hard argmax + focal weighting on boundaries
    "segnet_loss_weight": 200.0,        # heavy SegNet emphasis
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
    "segnet_loss_weight": 200.0,        # 2x proven_baseline, tests if reweighting alone helps
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
    "epochs": 400,                       # smoke test — 400 epochs per council
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
}
