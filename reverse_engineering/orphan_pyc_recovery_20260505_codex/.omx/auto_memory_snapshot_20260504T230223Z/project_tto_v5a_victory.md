---
name: TTO v5a/v5b — Auth 0.41 [UNLIMITED-COMPUTE]. Gradient fix validated.
description: v5b embedding loss auth 0.41 NEW BEST unlimited-compute. v5a output MSE auth 0.43. NOT contest-compliant (exceeds 30-min inflate budget).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## TTO v5b Results — NEW BEST [UNLIMITED-COMPUTE] (2026-04-15)
- **Auth: 0.41** (embedding loss 512D) -- NOT submittable (TTO takes 3+ hours)
- PoseNet: 0.00263 auth (contribution: 0.162)
- SegNet: 0.00148 auth (contribution: 0.148)
- Rate: 0.00401 (150KB archive, old format)
- Proxy: 0.263

## TTO v5a Results [UNLIMITED-COMPUTE] (2026-04-15)
- **Auth: 0.43** (output MSE 6D)
- PoseNet: 0.00295 auth (contribution: 0.172)
- SegNet: 0.00160 auth (contribution: 0.160)
- Subdir on Modal: `tto_v5a_output_mse/`

## Key Finding
Embedding loss (512D) is strictly superior to output MSE (6D) when PoseNet gradients work.
TTO v3 appeared to show the opposite, but that was THE GREAT GRADIENT BUG.

## Contest-Compliant Score (what actually matters)
The contest-compliant score is still **0.87** (renderer only, no TTO).
These TTO scores are for the paper's scalability story only.

## Config (v5b)
- lr=0.005, seg=100, pose=10, compress=0.5, patience=150
- use_embedding_loss=True, seg_odd_only=True
- 500 steps, 60 batches x 10 pairs
- Subdir on Modal: `tto_v5b_embedding/`
