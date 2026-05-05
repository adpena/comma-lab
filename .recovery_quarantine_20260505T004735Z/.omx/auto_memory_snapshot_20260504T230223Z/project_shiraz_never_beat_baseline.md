---
name: SHIRAZ never beat 0.90 baseline
description: SHIRAZ is a smaller experimental architecture; it has never matched the dilated-h64 0.90 baseline. The 0.90 is NOT a SHIRAZ score — common point of confusion in run_log.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
The 0.90 contest-CUDA baseline is the **pinned dilated h64 + CRF=50 + poses**
archive — a *different* architecture (~297KB renderer, much larger than SHIRAZ).
It is NOT a previous SHIRAZ score.

**SHIRAZ's actual score history (real CUDA contest evals):**
- 2026-04-25 standalone CUDA (stale dilated-h64 poses): 4.83 (poses didn't match the SHIRAZ renderer)
- 2026-04-26 first eval (BUG: pipeline.step_eval forgot --poses, zero-pose render): 2.80
- 2026-04-26 re-eval with proper --poses: **2.70** ← real number

**Why:** SHIRAZ was always a smaller experimental architecture (109-181K params)
trying to match the larger dilated-h64 baseline (~297KB renderer) via
post-training tricks (pose TTO + QAT). It hasn't worked; the smaller renderer
produces frames PoseNet finds 24× harder to read.

**How to apply:**
- When user/text references "0.9" or "the baseline", that's the dilated-h64
  model, NOT SHIRAZ.
- Do not present SHIRAZ scores as regressions vs 0.90 — they aren't comparable
  (different arch). SHIRAZ 4.83→2.70 is its OWN improvement.
- "Bigger may be better" — to attack the 0.30 target, the path is to *scale up
  or improve* the dilated-h64 architecture (or one like it), not chase smaller
  experimental renderers like SHIRAZ/DEN.
- Memory references: project_cuda_gate_result_20260425.md (the 0.90 measurement),
  project_session_20260425_deployment.md.
