---
name: Sweep Results — H= and CRF
description: Canonical locations for Karpathy H= sweep and CRF sweep results; what's recorded vs missing
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## H= Sweep (Karpathy channel-width sweep for postfilter)

**Status: CANONICALLY RECORDED in `reports/results.jsonl`**

All results from 2026-04-08 through 2026-04-10:
- h=16, long500: score 1.99, PoseNet 0.0692
- h=16, long1000: score 1.92, PoseNet 0.0589
- h=32, long500: score 1.95, PoseNet 0.0622
- h=32, long1000: score 1.85, PoseNet 0.0481
- h=32, ensemble MC75/25: score 1.84, PoseNet 0.0468
- h=48: score 1.76, PoseNet 0.0374
- h=64, long1000: score 1.73, PoseNet 0.0332
- h=64, standard long2500: score 1.51, PoseNet 0.0123
- **h=64, dilated (Modal): score 1.33, PoseNet 0.00218 ← BEST, promoted floor**

Run IDs follow pattern: `robust_current-{variant}-h{size}-promoted-cpu-2026-04-{date}`

**Why:** Dilated h=64 is the canonical CPU postfilter ceiling. All subsequent GPU work targets beating 1.33 auth.

**How to apply:** Don't re-run H= sweep. h=64 dilated is the winner. Any new postfilter experiments start from h=64 dilated.

## CRF Sweep

**Status: PARTIALLY RECORDED**

- CRF 34 (production baseline): 1.366 — IN results.jsonl as `robust_current-av1-524x394-crf34-promoted-cpu-2026-04-06`
- CRF 35: 1.328 (projected savings -6.6%) — rejected run in results.jsonl
- CRF sweep April 12: **0 bytes — run started, results never written**. File at `reports/raw/crf_sweep_20260412/summary.jsonl`.

**Why:** CRF 34 is the current encoder setting. CRF 35 saves bytes but loses quality. The April 12 re-sweep never completed.

**How to apply:** CRF 34 is the production encoder setting. A future sweep could test fractional CRFs (33.5, 34.5) but this is a low-priority CPU lane refinement vs the GPU renderer path.

## Tiny Frame Predictor Experiments (April 12)

**Status: RAW ONLY — NOT in results.jsonl**

Located in `reports/raw/2026-04-12-commavq-*` directories. These test a GPT-style autoregressive token predictor on VQ-codebook frames (commavq dataset). Model weights are 1.6MB, self-compressed to ~100KB (4-bit). Not directly relevant to the GPU renderer pipeline.

The commavq/lossless track is separate from the renderer GPU pipeline. Currently untracked and uncommitted.
