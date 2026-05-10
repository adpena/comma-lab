# Red-team overclaim fixes

Generated: `2026-05-10T09:55:00Z`

## Verdict

The round of adversarial review after A1 regression and T1 guard failure found
five concrete overclaim traps. All five are fixed in code with focused tests.

## Fixed bug classes

1. Raw `contest_auth_eval.py` CUDA/T4 JSON no longer marks a result as
   promotion-eligible or rank/kill-eligible by itself. Raw auth eval records
   exact-score completion and schema validity; promotion remains a later
   compliance/submission-policy decision.
2. `tac.auth_eval_schema` now recomputes the official contest formula from
   `avg_segnet_dist`, `avg_posenet_dist`, and exact archive bytes, and rejects
   score/rate inconsistency.
3. `scripts/pre_submission_compliance_check.py` consumes auth-eval schema
   blockers instead of trusting raw `promotion_eligible` stamps.
4. Stale nonterminal dispatch claims now block same-lane new claims until a
   matching terminal `stale_*` closure row is written.
5. T1 Modal recover treats adjudicated `promotion_blockers` and
   `promotion_eligible=false` as score-claim blockers; unsupported A1 GPU-tier
   cost estimates now fail closed instead of falling back to T4 pricing.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_pre_submission_compliance_check.py src/tac/tests/test_claim_lane_dispatch.py src/tac/tests/test_modal_t1_balle_endtoend.py src/tac/tests/test_dispatch_phase_a1_score_gradient_pr101.py src/tac/tests/test_preflight_all_clean_cache.py src/tac/tests/test_check_lane_pre_registered_before_work_starts.py`
  -> `257 passed in 15.76s`
- `.venv/bin/python -m py_compile experiments/contest_auth_eval.py experiments/modal_t1_balle_endtoend.py scripts/pre_submission_compliance_check.py src/tac/auth_eval_schema.py src/tac/preflight.py tools/claim_lane_dispatch.py tools/dispatch_phase_a1_score_gradient_pr101.py`
  -> passed
- `.venv/bin/python -m tac.preflight --scope dev --timings-json experiments/results/preflight_dev_timing_swarm_integrated_20260510_codex.json`
  -> `PREFLIGHT PASSED`, wall about `10s`
- Red-team inconsistent-score repro now returns:
  `score_component_formula_mismatch` and
  `rate_unscaled_archive_bytes_mismatch`.
- Unsupported A1 `lightning/H100` cost estimate now raises
  `ValueError: unsupported GPU cost estimate key`.

## Score-lowering implication

These fixes do not lower score directly. They protect the score-lowering loop
from promoting or re-dispatching bad evidence. The immediate score-lowering
queue is unchanged after hardening:

1. fix and rerun T1 only after canonical A1 payload mounting is proven;
2. keep Kaggle/Optuna/CMA-ES as proxy-only candidate generation;
3. materialize proxy winners into byte-closed archive/runtime packets;
4. run exact CUDA only behind fresh dispatch claims and strict schema gates.
