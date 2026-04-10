"""Named training profiles for tac.

Each profile is a dict of TrainConfig overrides. Use with:
    from tac.profiles import PROFILES
    config = TrainConfig(**{**PROFILES["council_v1"], "tag": "my_run"})

Or from CLI:
    python train_tac.py --profile council_v1 --tag my_run
"""

# Council-recommended settings (2026-04-10 master session)
# Einstein/Tao/Contrarian/Karpathy/Hinton/LeCun/Qwen/DeepSeek consensus
COUNCIL_V1 = {
    "variant": "dilated",
    "hidden": 64,
    "kernel": 3,
    "epochs": 2500,
    "lr": 5e-4,
    "ema_decay": 0.997,
    "alpha": 20.0,
    "sal_lambda": 1.0,
    "loss_mode": "kl_distill",
    "temperature_start": 5.0,
    "temperature_end": 0.5,          # sub-1.0 regime for argmax pressure (floor relaxed to 0.1)
    "boundary_weight": 150.0,        # was 5-10; optimal ~150 for 5% boundary fraction
    "boundary_anneal": True,          # couple bw to temperature schedule
    "hard_frame_ratio": 0.3,         # power-law curriculum, 0.3 = moderate emphasis
    "error_replay_every": 200,        # recompute hard frames using model output
    "eval_every": 5,                  # skip eval on 4/5 epochs (ramps to 1 in final 10%)
    "accum_steps": 4,
    "segnet_loss_weight": 100.0,      # formula-derived, matches 100*seg in score
}

# Aggressive SegNet-focused (contrarian + DeepSeek recommendation)
SEGNET_ATTACK = {
    **COUNCIL_V1,
    "temperature_end": 0.2,           # aggressive argmax pressure in final phase
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

PROFILES = {
    "council_v1": COUNCIL_V1,
    "segnet_attack": SEGNET_ATTACK,
    "proven_baseline": PROVEN_BASELINE,
    "h96_council": H96_COUNCIL,
    "smoke": SMOKE,
}
