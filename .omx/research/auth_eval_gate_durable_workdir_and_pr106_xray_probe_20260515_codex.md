# Auth-Eval Gate Durable Workdir + PR106 XRay Probe - 2026-05-15

## Summary

This landing turns two repeated failure classes into executable guardrails:

1. Substrate trainers using the canonical `gate_auth_eval_call` now pass
   `--keep-work-dir --work-dir <output_json_stem>_work` into
   `experiments/contest_auth_eval.py`, so score-grade auth-eval work products
   remain inspectable after harvest.
2. Modal training-wrapper CPU auth eval remains explicitly advisory. When the
   wrapper sets `MODAL_AUTH_EVAL_ADVISORY_ONLY=1` and forces
   `AUTH_EVAL_DEVICE=cpu`, the gate passes `--allow-temp-work-dir` as a
   diagnostic bypass instead of producing a misleading custody claim.
3. `scripts/remote_lane_substrate_sane_hnerv.sh` now surfaces the trainer's
   canonical `contest_auth_eval_cuda.json` path, with legacy `auth_eval.json`
   fallback.
4. `substrate_sane_hnerv_modal_a100_dispatch` is now explicitly
   `smoke_only: true` with `smoke_validation_contract: training_artifact_v1`.
   This matches `modal_train_lane.py`, which forces `AUTH_EVAL_DEVICE=cpu` and
   advisory-only scoring because the training wrapper is not an exact-CUDA
   evaluator. Full paired eval must be a separate exact-eval dispatch.
5. The PR106 CUDA latent-correction probe planner landed as false-authority
   XRay-to-probe wiring. It consumes hard-pair hitlists and pair XRay JSON,
   chooses pair/mode rows under a byte budget, and refuses materialization or
   frontier language until a byte-closed archive plus paired CPU/CUDA eval
   exists.

## Why This Matters

The operator concern was correct: meta-Lagrangian, Pareto, autopilot, magic
codec, PacketIR, and XRay tooling only lowers score when it feeds a concrete
candidate-producing loop. The prior SANE Modal smoke red was not a model result;
it was a custody/workdir mismatch in the auth-eval wrapper path. The PR106
planner closes the next analysis-to-build gap by turning XRay hard pairs into
an explicit correction probe plan, while preserving false-authority fields.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_smoke_auth_eval_gate.py \
  src/tac/tests/test_run_modal_smoke_before_full.py \
  src/tac/tests/test_remote_lane_substrate_sane_hnerv_script.py \
  src/tac/tests/test_build_pr106_cuda_latent_correction_probe.py -q
```

Result: focused smoke/auth-eval suite passed after the recipe contract fix.

```bash
.venv/bin/ruff check \
  src/tac/substrates/_shared/smoke_auth_eval_gate.py \
  src/tac/tests/test_smoke_auth_eval_gate.py \
  src/tac/tests/test_remote_lane_substrate_sane_hnerv_script.py \
  tools/build_pr106_cuda_latent_correction_probe.py \
  src/tac/tests/test_build_pr106_cuda_latent_correction_probe.py
```

Result: passed.

```bash
.venv/bin/python -m py_compile \
  src/tac/substrates/_shared/smoke_auth_eval_gate.py \
  experiments/train_substrate_sane_hnerv.py \
  tools/build_pr106_cuda_latent_correction_probe.py
```

Result: passed.

## Public Frontier Refresh

Official leaderboard refresh on 2026-05-15 still has PR101 first at `0.193`,
then PR103/PR102/PR100 at `0.195`. PR107 remains `0.229`.

Post-deadline PR refresh found PR108 updated/closed 2026-05-11 with a reported
score of `3.59`; it is not a frontier intake target. PR107 has no new technical
comment beyond the job/internship invitation. A PR comment scan found no
`geohot`/George participation in this challenge repo.

## Next Gate

The next concrete step is a fail-closed PR106 correction materializer or
score-table actuator that consumes
`tools/build_pr106_cuda_latent_correction_probe.py` output and produces either:

- deterministic CUDA-in-loop pair/latent score table artifacts, or
- a byte-closed archive mutation with paired exact CPU/CUDA eval queued.

Until then the planner remains `score_claim=false`, `promotion_eligible=false`,
and `ready_for_exact_eval_dispatch=false`.
