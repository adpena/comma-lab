---
name: Compute Budget — Unlimited T4, Paid Credits Authorized
description: Unlimited T4 GPU time available. Paid Modal/cloud credits authorized. No need to ration GPU hours.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Unlimited T4 GPU time is available. User has authorized paying for credits (Modal, etc.).

**Why:** User wants the lowest possible score. GPU budget is no longer a constraint for T4-class hardware. Time is the bottleneck, not money.

**How to apply:**
- Run as many parallel T4 experiments as needed on Modal
- No need to ration GPU hours or wait for free tier windows
- Kaggle free T4 is still useful as bonus parallelism but not critical-path
- Still use T4 as default GPU (our 287K param model doesn't need A10G/A100)
- If a specific experiment genuinely needs more powerful GPU, can request — but T4 is sufficient for current architecture
- Paid credits authorized for Modal and other cloud platforms
- Run experiments in parallel, not sequentially — maximize iteration speed
