---
name: Lightning.ai $200/mo Pro ACTIVATED for writeup + paper + dashboard value (NOT raw $/hr)
description: 2026-04-30 ~1:30pm CDT user decision: "I decided to sign up for lightning because even though it is a little expensive I think it's worth it and will give us the best chance at the best score possible and the best writeup possible; we should include the use of lightning.ai in our writeup and paper and maybe our site or use it for a supplement or notebook deployment or something of our full pipeline and training script with comparison dashboards and graphics and viz and stuff". USER OVERRODE my cost-efficiency recommendation. Reason: writeup/paper/dashboard value, not raw GPU $/hr. Lightning Studios = persistent compute + Jupyter + matplotlib-friendly + H100 access + great for public-facing notebooks/dashboards.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The decision

User: "I decided to sign up for lightning because even though it is a little expensive I think it's worth it and will give us the best chance at the best score possible and the best writeup possible; we should include the use of lightning.ai in our writeup and paper and maybe our site or use it for a supplement or notebook deployment or something of our full pipeline and training script with comparison dashboards and graphics and viz and stuff"

## Why user activated despite my cost-efficiency caveat

I recommended SKIP for raw $/hr efficiency (RunPod 4090 $0.34/hr beats Lightning's effective $1-1.50/hr-equivalent, AWS spot $0.22 T4 free credits, etc.). User overrode because:

1. **Writeup value** — Lightning Studios are the canonical platform for ML research notebooks; arXiv-paper-supplement notebooks are best hosted on Lightning
2. **Paper supplement / notebook deployment** — interactive reproducibility for the comma.ai writeup
3. **Comparison dashboards + graphics + viz** — Lightning's persistent studios make matplotlib + plotly + per-experiment dashboards trivial
4. **Public-facing pipeline reproducibility** — Lightning studios can be shared as "click to reproduce"
5. **Best chance at best score** — H100 access for paradigm-shift training (Self-Compress NN, joint renderer-scorer, Mask payload overhaul training)

This is a STRATEGIC investment in the WRITEUP and PUBLIC DEMO infrastructure, not raw compute. The cost-efficiency calculation I did was incomplete because I didn't weight writeup-quality value.

## How to apply

### For paper / writeup / supplement
- Lightning Studio with the FULL PIPELINE (not just training, the entire compress→inflate→eval workflow)
- Interactive cells: load Lane G v3 anchor → apply paradigm shift X → measure score → display before/after
- Comparison dashboards: per-paradigm-shift score breakdown, stacked-bar archive byte composition, etc.
- Embed in arXiv paper as supplement: "click to reproduce on Lightning Studio"

### For Cloudflare site (per CLAUDE.md Strategic Secrecy Rule — DEFER until user approves disclosure)
- Studio URL is public; secrecy-gate the actual code repo until contest deadline
- Post-deadline: full Lightning Studio public on the site
- Pre-deadline: Lightning Studio is INTERNAL DEV ENVIRONMENT only

### For training (heavyweight paradigm shifts)
- Self-Compressing NN training (paradigm ε)
- Joint renderer-scorer training (paradigm δ)
- Mask payload overhaul training (paradigm α — NeRV/wavelet/VQ-VAE)
- Full Phase 4 stack assembly + comparison runs

### For HIGH-THROUGHPUT DISPATCH (#309 redo)
Lightning becomes a 4th platform alongside Vast.ai + Modal + bat00:
- Vast.ai: cheap burst (4090 $0.27/hr, NVDEC roulette)
- Modal: serverless burst (T4 $0.59/hr, A10G $1.10/hr)
- bat00: free local CUDA (RTX 2070S → 3090)
- Lightning: heavy-training + interactive-dev + paper-supplement (H100 $3-4/hr-equiv)

Wire `lightning_dispatch.py` analog to `modal_train_lane.py`.

## Public-facing visibility

User explicitly said: "include the use of lightning.ai in our writeup and paper and maybe our site"

This is APPROVED disclosure (vs CLAUDE.md Strategic Secrecy Rule which guards code details). The fact that we USE Lightning is publishable. The IMPLEMENTATION DETAILS of our paradigm shifts remain gated until user approves submission.

## Cost projection

- $200/mo subscription = base
- $240 included credits (offsetting H100/A100 hours)
- Effective: ~80-100 hours of H100 OR ~200 hours of A100 OR ~300 hours of A10G
- Run rate: amortized $0.07/hr for the studio + per-hour compute for runs

If we run paradigm-shift training (Self-Compress NN, joint training) on H100 = ~10-30 hours total = well within $200/mo.

## Cross-refs

- feedback_no_monetary_commit_20260430.md (no caps)
- feedback_priority_time_to_floor_with_final_approval_20260430.md (final-approval gate still applies for public publication)
- feedback_full_six_month_plan_aggressive_no_shortcuts_20260430.md (the all-in plan)
- CLAUDE.md "Strategic Secrecy Rule" (still gates IMPLEMENTATION DETAILS until contest deadline)
- CLAUDE.md "Cloudflare site URL restriction" (don't broadcast site URL until user approves)
