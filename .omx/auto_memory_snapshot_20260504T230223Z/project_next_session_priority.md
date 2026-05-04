---
name: Next Session Priority — Vast.ai FiLM + Hinge Experiments (2026-04-20)
description: Deploy hinge+eval roundtrip on Vast.ai. Implement FiLM in renderer training. Close gap to Quantizr 0.33.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## STALE NOTICE
Previous version (2026-04-15) said "Deploy renderer_tto.py on Modal." Updated 2026-04-20.
Modal deploy path is superseded by Vast.ai. Kaggle scorer path is fixed.

## IMMEDIATE PRIORITY: Close Gap to Quantizr (0.87 -> sub-0.40)

### Path 1: TTO with all breakthroughs (Unlimited Compute lane)
Deploy on Vast.ai 4090:
```
python scripts/check_vastai.py create
python scripts/check_vastai.py deploy <id>
python scripts/check_vastai.py run <id> tto_v7_hinge_roundtrip
```
Expected: auth 0.30-0.35 with hinge + eval roundtrip + embedding loss

### Path 2: Renderer retraining with FiLM + eval roundtrip (Contest lane)
- Add FiLM conditioning on pose vectors to AsymmetricPairGenerator
- Train with simulate_eval_roundtrip in loss (like Quantizr PR#55)
- This improves the CONTEST-COMPLIANT score (Lane 1)
- Expected: auth 0.50-0.60 (major improvement from 0.87)

### Path 3: Latent-conditioned renderer (Contest lane, stretch)
- 16-dim per-frame latent code optimized at compress time
- Inflate is deterministic (no scorers) -- contest compliant
- 9.6KB rate cost (negligible at 600 pairs)

## Blockers
- FiLM not yet integrated into inflate_renderer.py (forward path only)
- Vast.ai SSH key must be registered before first create
- Checkpoint must be verified: MD5 cff8dca4

## What's Working
- tto_step_curve.py fully threads hinge + lr_schedule (verified in code)
- experiments registry has tto_v7_hinge_roundtrip entry
- Checkpoint verification module prevents wrong-checkpoint bugs
- build_deploy_bundle.sh needs pyproject.toml added (FIX PENDING)
