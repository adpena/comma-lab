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

PROFILES = {
    "council_v1": COUNCIL_V1,
    "council_v2_adaptive": COUNCIL_V2_ADAPTIVE,
    "segnet_attack": SEGNET_ATTACK,
    "proven_baseline": PROVEN_BASELINE,
    "h96_council": H96_COUNCIL,
    "smoke": SMOKE,
    "psd_standard_adaptive": PSD_STANDARD_ADAPTIVE,
}
