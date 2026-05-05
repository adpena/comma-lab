---
name: Free Tier Priority — Modal Credits Exhausted
description: Modal free credits used up. Prioritize Lightning T4 + Kaggle P100 + M5 Max MLX. Modal only for critical auth evals.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Rule

Modal free credits are EXHAUSTED. Prioritize free compute:

1. **M5 Max MLX** — Phase 1 renderer pre-training (4.7x faster, zero cost)
2. **Lightning T4** — GPU training (free tier, SSH keys exist at ~/.ssh/lightning_rsa)
3. **Kaggle P100** — Long overnight runs (free 30h/week)
4. **AWS free tier** — CPU-only tasks (CRF sweep, ensemble, packaging)
5. **Modal A10G** — ONLY for critical auth evals or time-sensitive experiments

**Why:** First Modal bill received. Budget should be spent only where free alternatives don't exist (auth eval requires specific environment).

**How to apply:** Before launching any Modal run, ask: "Can this run on Lightning or Kaggle instead?" If yes, use the free platform.
