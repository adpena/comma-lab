# Agent Onboarding — comma video compression challenge

## Current state (updated 2026-04-10 evening)

- **Score: 1.33** authoritative (#1 by 0.53 margin, deadline May 3)
- **Compliant archive**: 903KB (includes postfilter_int8.pt inside archive.zip)
- **Best trustworthy technique**: proven_baseline (standard loss, dilated h=64)
- **Experimental**: PSD (PixelShuffle+Dilated) with standard loss + 5 adaptive frontier items

## CRITICAL LESSONS LEARNED

1. **KL distill is DEAD.** Two authoritative evals confirmed PoseNet collapse: 1.85 and 2.05.
   Root cause: w_s·T² = 750, optimal was 2.95 (254x over-weighted). PoseNet regressed 26x.
   NEVER use KL distill loss_mode.

2. **Neural network artifacts MUST be inside archive.zip** per contest rules (affects rate calculation).

3. **The adaptive weight formula was retired** after discovering T² cancels in the derivation.
   Use standard loss with static weights.

4. **Lean 4 proofs** (`proofs/AdaptiveWeights.lean`) are mathematically correct with zero sorry obligations,
   but they verify a vacuous identity (T² cancels by construction). The per-channel quantization proof remains useful.

## Key files

- `src/tac/` — training library v1.0.0, 70 tests
- `src/tac/adaptive.py` — adaptive weights (retired; formula was vacuous)
- `src/tac/profiles.py` — named training profiles
- `proofs/AdaptiveWeights.lean` — formal verification
- `experiments/train_tac.py` — canonical training entry point
- `submissions/robust_current/` — the submission (compliant archive)

## How to train

```bash
# RECOMMENDED: proven baseline (produced the 1.33 authoritative score)
.venv/bin/python experiments/train_tac.py \
    --profile proven_baseline \
    --tag my_experiment \
    --precomputed experiments/precomputed_local

# EXPERIMENTAL: PSD architecture with adaptive frontier
.venv/bin/python experiments/train_tac.py \
    --profile psd_standard_adaptive \
    --tag my_experiment \
    --precomputed experiments/precomputed_local

# Available profiles: proven_baseline, psd_standard_adaptive, council_v1, segnet_attack, h96_council, smoke
```

## What NOT to do

- Do NOT use KL distill loss_mode -- two authoritative evals confirmed PoseNet collapse (1.85 and 2.05)
- Do NOT use adaptive_rebalance=True -- the formula was vacuous (T² cancels)
- Do NOT use segnet_loss_weight > 100 with any loss mode
- Do NOT add PoseNet gradient caps/clamps (caused 26x regression)
- Do NOT use alpha_seg > 500 (formula-derived optimal is ~200)
- Do NOT store postfilter_int8.pt outside archive.zip (compliance violation)
- Do NOT modify upstream scorer files

## Current frontier

- **PSD architecture** (PixelShuffle+Dilated) being tested with standard loss
- **5 adaptive frontier items** implemented: boundary dispatch, sin² ramp, replay gate, 3-phase eval, LR plateau
- **Multi-pass inflate** being tested
- **Profile**: `psd_standard_adaptive`

## Infrastructure

- Local: M5 Max 128GB — run 3-4 experiments in parallel with precomputed data
- Lightning AI: SSH working (`ssh s_...@ssh.lightning.ai`), T4 CUDA, code synced
- Modal: A10G serverless (currently stopped, ready for fresh deploy)
- Kaggle: 2 GPU sessions available (T4/P100)
- Precomputed: `experiments/precomputed_local/` (7GB, instant loading)
