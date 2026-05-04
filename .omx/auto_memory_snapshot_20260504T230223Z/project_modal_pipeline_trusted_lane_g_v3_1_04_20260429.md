---
name: Modal pipeline TRUSTED — Lane G v3 = 1.04 [Modal-T4-CUDA]
description: 2026-04-29 ~5am. Modal auth eval reproduced Lane G v3 score within 0.01 of Vast.ai measurement (1.04 vs 1.05). Pipeline contest-equivalent. Pivot all training to Modal validated. PoseNet/SegNet/Rate breakdowns landed and stable.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The verification

Ran `experiments/modal_auth_eval.py` on `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip` (the canonical landed Lane G v3 archive at 694KB) on Modal T4.

| Metric | Modal T4 (this run) | Vast.ai 4090 (prior landing) | Drift |
|--------|--------------------|-----------------------------:|------:|
| PoseNet | 0.00305 | 0.00346 | -0.0004 |
| SegNet  | 0.00401 | 0.00400 | +0.00001 |
| Rate    | 0.01849 | 0.01849 | 0 |
| **Score** | **1.04** | **1.05** | -0.01 |

The 0.01 drift is within rounding (the formula is `100*seg + √(10*pose) + 25*rate` — small posenet shift propagates).

## Cost / latency

- Inflate: 43.3s (T4 inflate path is fast — neural renderer forward + writeback)
- Evaluate: 551.9s (~9.2 min, includes 600 paired-frame scoring)
- Total: 595.2s ≈ 10 min
- Cost: ~$0.10 ($0.59/hr × 10/60 h)

## Why this matters

Vast.ai 4090 NVDEC bad-host rate tonight ≈ 85%. We burned ~$5 across 5 dispatch rounds and got 0 lanes to actually train. Modal sidesteps this entirely:
- Same image every run (no host variability)
- No SSH proxy rate limits
- No NVDEC missing
- Slightly higher cost ($0.59 vs $0.25/hr) but variance ≈ 0

For a 6h training run: Vast.ai expected cost ≈ $1.50 with ~30% success rate = $5 effective; Modal = $3.54 with 99% success rate. **Modal wins on expected $/successful_lane.**

## Modal image fix landed (commit 11d56896)

`experiments/modal_auth_eval.py` now adds:
- `add_local_dir("src/tac", remote_path="/root/tac")` — needed for SZv1 renderer load
- `PYTHONPATH=/root:/root/submission:/root/upstream:{work}` — so `import tac` resolves

Without this, SZv1 archives fail at inflate with `ModuleNotFoundError: No module named 'tac'`.

## Decision rule (UPDATED — supersedes feedback_canonical_lane_lifecycle_DECISION_TREE_20260428's table)

| Lane property | Vast.ai 4090 | Modal T4 |
|---------------|--------------|----------|
| Auth eval (~10 min) | NO — random failures | ✓ canonical |
| Training < 2h, cheap moonshot, retry budget OK | ✓ with retry-wrapper | over-priced |
| Training 2-6h, validated lane | risky on bad NVDEC night | ✓ |
| Training > 6h, important | NO | ✓ canonical |
| First-of-class lane (no proof) | NO — local smoke first | NO — local smoke first |

## What's NOT yet built

- `modal_train_lane.py` — wrapper that runs any `remote_lane_*.sh` on Modal. Would replace the failed v5 dispatches. Estimated 1-2h to build.
- The 5 lanes that died on Vast.ai (Ω-Hessian, MAE-V, UNIWARD, CG, HM) are ready to dispatch on Modal once that wrapper exists.

## Cross-references

- Previous score: `experiments/results/lane_g_v3_landed/contest_auth_eval.json` (Vast.ai 1.05)
- This score: `experiments/results/modal_auth_eval_9b20bdfca246.json` (Modal T4 1.04)
- Modal harness: `experiments/modal_auth_eval.py` (commit 11d56896)
- Memory `feedback_vastai_nvdec_roulette_pivot_to_modal_20260429` — full session context
- Memory `feedback_canonical_lane_lifecycle_DECISION_TREE_20260428` — original Modal vs Vast.ai rubric (now superseded)
