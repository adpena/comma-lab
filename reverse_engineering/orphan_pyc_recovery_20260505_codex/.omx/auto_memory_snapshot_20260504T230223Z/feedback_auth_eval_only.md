---
name: Auth Eval Only for Decisions
description: NEVER make strategic decisions based on proxy scores alone. Always confirm with upstream evaluate.py + DALI auth eval.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Proxy scores are for fast iteration. Auth eval (upstream evaluate.py with DALI) is the ONLY number that matters.

**Why:** Proxy-to-auth gap was 38% on our renderer (proxy 0.63 vs auth 0.87). The gap comes from:
1. Resolution mismatch: proxy at 384x512, auth after 874x1164 round-trip
2. DALI vs PyAV decode: different pixel values (29x PoseNet divergence documented)
3. uint8 quantization: proxy simulates but auth does it for real

**How to apply:**
- Never claim a win without auth eval
- Never promote a checkpoint based on proxy alone
- Never compare to Quantizr (0.60) using proxy numbers
- Auth eval is auto-chained in the Modal TTO pipeline — always wait for tto_auth_eval.json
- If proxy looks good but auth eval isn't available, explicitly state "proxy only, auth pending"
