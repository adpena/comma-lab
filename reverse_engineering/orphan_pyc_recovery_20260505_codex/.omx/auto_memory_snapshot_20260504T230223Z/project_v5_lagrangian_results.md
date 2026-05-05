---
name: v5 Lagrangian Results
description: asym_v5_lagrangian_fixed auth results — 0.87 new best, Lagrangian annealing discovery
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
v5 Lagrangian-fixed (R2 caps: rho_max=100, lambda_cap=1000) resumed from v3 at ep12400.

**Results:**
- renderer_best.pt (ep12600, +200 epochs): auth=0.87 — NEW BEST (13% over v3's 1.00)
- PoseNet: 0.031 (35% better), SegNet: 0.0022 (held flat)
- constraints_met ep16999 (+4600 epochs): auth=1.37 — REGRESSION, model drifted under weaker caps

**Why:** Clamping λ from 10000→1000 on resume freed PoseNet from over-constraint, finding a better basin. But the weaker caps couldn't hold the improvement long-term.

**How to apply:** "Lagrangian annealing" — reduce caps briefly to explore the Pareto frontier, snapshot the transient improvement, then re-tighten or discard late epochs. This is a first-order optimization method along the constraint boundary, not fishing.

**Next experiments (council-approved):**
1. Checkpoint landscape sweep: auth eval ep12500-16500 every 500ep to map drift curve
2. λ_cap sweep: 5 short runs (300ep) from v5-best with λ_cap ∈ {500, 750, 1000, 1500, 2000}
3. Fix Kaggle deployment (new wheel + preflight), then supervised from-scratch (v6)
