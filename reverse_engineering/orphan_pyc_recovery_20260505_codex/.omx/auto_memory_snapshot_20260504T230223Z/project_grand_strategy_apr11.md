---
name: Grand Strategy Council Apr 11
description: 13-member council verdict — CRF sweep on dilated first, PSD to Modal, kill 4/6 processes
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Verdict (unanimous on strategy, dissents on details)

### Priority 1: CRF sweep on dilated (TODAY)
- Run CRF 34-38 auth eval using hardened tools/auth_eval.sh
- Guaranteed 0.04-0.06 pts improvement from rate reduction
- Zero architecture risk, proven checkpoint
- Einstein insight: CRF controls quantization, which is PoseNet's fundamental limit

### Priority 2: Deploy PSD to Modal A10G
- CUDA training is 5-10x faster than MPS
- PSD just broke PoseNet 0.01 — let it cook to ep 1500-2000
- Auth eval PSD at ep 1200 on Kaggle
- If PSD reaches pose=0.003 + CRF 36: projected ~1.24

### Process management
- KILLED: PSD+hfr, PSD+bw50, h=32 smoke, proven_baseline 3500
- KEEP: Only PSD standard h=64 on M5 Max (now moved to Modal)
- M5 Max freed for CRF sweep auth evals

### Key insights from council
- MRS formula is DIAGNOSTIC ONLY — use between runs, not during training (Tao/Karpathy)
- Geometric mean scorer is paper content, not competition strategy (Contrarian)
- CRF is the most important parameter and was undertested (Bellard)
- SWA on existing dilated checkpoint is free improvement (Contrarian)
- x265 parameter sweep beyond CRF has untapped gains (Bellard)
- Stop starting new experiments, let existing ones finish (Collier)

### Dissents
- Collier: don't kill proven_baseline 3500, let it converge
- Einstein: write paper NOW while insights are fresh
- Contrarian: SWA implementation effort is underestimated
- Tao: need more auth evals to validate proxy-auth transfer (n=2 is too few)

**Why:** 22 days left, CRF sweep is highest Sharpe ratio bet.
**How to apply:** CRF sweep today, PSD to Modal, paper in week 3.
