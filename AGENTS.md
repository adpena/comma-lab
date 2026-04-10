# Agent Onboarding — comma video compression challenge

## Current state (updated 2026-04-10)

- **Score: 1.33** (#1 by 0.55 margin, deadline May 3)
- Best training proxy: **~1.30** and dropping (KL distill + hard-frame on Modal A10G)
- Promoted checkpoint: `submissions/robust_current/postfilter_int8.pt` (dilated h=64)

## The approach

A 45KB int8 CNN post-filter corrects AV1-decoded frames by backpropagating through frozen PoseNet + SegNet scorers. Score = 100*seg + sqrt(10*pose) + 25*rate.

## Key files

- `src/tac/` — the training library (v0.9.0, 70 tests)
- `src/tac/profiles.py` — named training profiles
- `experiments/train_tac.py` — canonical training entry point
- `submissions/robust_current/` — the submission
- `docs/writeup_draft.md` — competition writeup
- `.omx/state/` — current focus, next experiments
- `.omx/research/findings.md` — all research findings

## How to train

```bash
# Use council_v1 profile (recommended default for all new runs)
.venv/bin/python experiments/train_tac.py \
    --profile council_v1 \
    --tag my_experiment \
    --precomputed experiments/precomputed_local

# Available profiles: council_v1, segnet_attack, proven_baseline, h96_council, smoke
# Profiles live in src/tac/profiles.py. CLI args override profile values.
```

## Council-recommended settings (council_v1 profile)

- **Architecture**: dilated h=64, kernel=3
- **Loss**: KL distillation T=5->0.5 with PoseNet gradient cap
- **Boundary weight**: 150 (5% boundary pixels need ~20x amplification)
- **Boundary anneal**: True (couples boundary_weight to temperature)
- **Hard-frame curriculum**: ratio=0.3 (power-law emphasis on worst SegNet pairs)
- **Error replay**: every 200 epochs (recomputes using current model output)
- **Quantization**: per-channel symmetric int8
- **Eval**: every 5 epochs (ramps to 1 in final 10%)

## What NOT to do

- Do NOT use alpha_seg=5000 (formula-derived optimal is ~200)
- Do NOT use standard loss for new experiments (KL distill is better)
- Do NOT run standard (non-dilated) architecture (superseded)
- Do NOT modify upstream scorer files

## Infrastructure

- Local: M5 Max 128GB — run 3+ experiments in parallel with precomputed data
- Lightning AI: SSH working, T4 CUDA, precomputed uploaded
- Modal: A10G serverless, precomputed on volume
- Precomputed: `experiments/precomputed_local/` (7GB, instant loading)
