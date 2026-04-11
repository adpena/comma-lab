# Agent Onboarding — comma video compression challenge

## Current state (updated 2026-04-10 evening)

- **Score: 1.33** authoritative (#1 by 0.53 margin, deadline May 3)
- **Compliant archive**: 903KB (includes postfilter_int8.pt inside archive.zip)
- **Best trustworthy technique**: proven_baseline (standard loss, dilated h=64)
- **Experimental**: council_v2_adaptive (first-ever run in progress on local MPS)

## CRITICAL LESSONS LEARNED

1. **KL distill proxy is UNTRUSTWORTHY.** A checkpoint with proxy 1.25 scored 1.85 authoritative.
   Root cause: w_s·T² = 750, optimal was 2.95 (254x over-weighted). PoseNet regressed 26x.
   NEVER promote a KL distill checkpoint without authoritative eval.

2. **Neural network artifacts MUST be inside archive.zip** per contest rules (affects rate calculation).

3. **The adaptive weight system** (`src/tac/adaptive.py`) derives optimal hyperparameters from the
   scoring formula instead of guessing. Use `--profile proven_baseline` to enable.

4. **Lean 4 proofs** (`proofs/AdaptiveWeights.lean`) formally verify the key equations.
   Zero sorry obligations.

## Key files

- `src/tac/` — training library v1.0.0, 70 tests
- `src/tac/adaptive.py` — mathematically-derived adaptive weights
- `src/tac/profiles.py` — named training profiles
- `proofs/AdaptiveWeights.lean` — formal verification
- `experiments/train_tac.py` — canonical training entry point
- `submissions/robust_current/` — the submission (compliant archive)

## How to train

```bash
# RECOMMENDED: adaptive weights (self-correcting, derived from scoring formula)
.venv/bin/python experiments/train_tac.py \
    --profile proven_baseline \
    --tag my_experiment \
    --precomputed experiments/precomputed_local

# SAFE FALLBACK: proven baseline (produced the 1.33 authoritative score)
.venv/bin/python experiments/train_tac.py \
    --profile proven_baseline \
    --tag my_experiment \
    --epochs 3500 \
    --precomputed experiments/precomputed_local

# Available profiles: council_v1, council_v2_adaptive, segnet_attack, proven_baseline, h96_council, smoke
```

## What NOT to do

- Do NOT use static segnet_loss_weight > 10 with KL distill (w_s·T² must be ~3, not 750)
- Do NOT add PoseNet gradient caps/clamps (caused 26x regression)
- Do NOT promote KL distill checkpoints without authoritative eval
- Do NOT use alpha_seg > 500 (formula-derived optimal is ~200)
- Do NOT store postfilter_int8.pt outside archive.zip (compliance violation)
- Do NOT modify upstream scorer files

## Infrastructure

- Local: M5 Max 128GB — run 3-4 experiments in parallel with precomputed data
- Lightning AI: SSH working (`ssh s_...@ssh.lightning.ai`), T4 CUDA, code synced
- Modal: A10G serverless (currently stopped, ready for fresh deploy)
- Kaggle: 2 GPU sessions available (T4/P100)
- Precomputed: `experiments/precomputed_local/` (7GB, instant loading)
