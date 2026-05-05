---
name: FiLM Eval Without Poses CRITICAL
description: pipeline.step_eval silently dropped --poses to auth_eval_renderer; FiLM models scored 32x worse on PoseNet from zero-pose rendering. Hard-fail enforced 2026-04-26.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
CRITICAL: never let a FiLM-conditioned renderer (pose_dim>0) be evaluated
without optimized poses. The auth_eval_renderer.py used to merely WARN
when this happened ("Using zero poses — PoseNet score will be degraded"),
which was invisible in 250-min pipeline runs.

**Why:** SHIRAZ v4 (2026-04-26) ran a full 250-min pipeline (train + QAT
+ pose TTO + archive + eval) on a Vast.ai 4090. The eval reported a
"contest-compliant CUDA score" of 2.80 (vs verified baseline 0.90). Hard
investigation showed the optimized_poses.bin (7.2KB) was sitting in the
archive iter_dir but pipeline.step_eval never built `--poses` into the
auth_eval_renderer.py subprocess command. The renderer rendered with
zero pose conditioning, PoseNet's FastViT scored 0.342 distortion vs
the ~0.011 it would have scored with the actual poses. The 2.80 number
was MEANINGLESS — the cost of running an FiLM renderer with zeros.

**How to apply:**
- Whenever building any subprocess command for `auth_eval_renderer.py`,
  ALWAYS pass `--poses {optimized_poses.bin|pt}` if the model is
  FiLM-conditioned. Auto-discover from the archive's iter_dir AND the
  archive's parent dir (covers both `pipeline.py compress` and `pipeline.py
  eval` standalone).
- The `auth_eval_renderer.py` script now SystemExit's hard if it sees
  pose_dim>0 and `--poses None`. This is the safety net.
- Memory check: any time you see `pose_d` > 0.1 in a CUDA contest eval
  for a FiLM renderer, the FIRST hypothesis is "did we pass --poses?",
  not "is the renderer broken?". Real PoseNet on a working renderer is
  ~0.005-0.02. Anything above 0.1 means structural bug, not training.
- The fix lives at commit 63854f31: pipeline.py:step_eval auto-discovers
  + passes; auth_eval_renderer.py raises SystemExit on silent zero-pose.
- Adjacent lesson: WARNING messages in long pipelines are dead text. Use
  raise SystemExit / RuntimeError for any data-corruption-level issue.
