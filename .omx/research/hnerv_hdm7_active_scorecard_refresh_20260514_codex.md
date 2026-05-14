# HNeRV HDM7 Active Scorecard Refresh - 2026-05-14

## Purpose

Refresh the active HNeRV routing surfaces after the HDM7 final-length-elision
exact CUDA anchor landed.

This is an internal optimizer-routing update, not a public promotion claim.
Score axis remains `[contest-CUDA]`.

## New Internal Score-Lowering Floor

- label: `PR106-R2-HDM7-HLM2-XMEMBER`
- score: `0.20636832361415344`
- archive bytes: `186405`
- archive sha256: `df2e07084233fd38b01c46a4ffebe244391b01a8bf2e04392a4e0348463f1bbb`
- auth eval JSON: `experiments/results/modal_auth_eval/hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/contest_auth_eval.json`
- result review packet: `.omx/research/hdm7_final_len_elided_exact_cuda_result_review_20260514_codex.json`
- exact CUDA job: `hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z`

## Refreshed Artifacts

- active scorecard: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm7_codex/scorecard.json`
- scorecard markdown: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm7_codex/scorecard.md`
- HDM7 section profile: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm7_codex/hdm7_section_profiles.json`
- entropy ranking: `experiments/results/hnerv_frontier_entropy_gap_ranking_20260514_hdm7_codex/frontier_entropy_gap_ranking.json`
- autopilot evidence row appended to `reports/cathedral_autopilot_evidence.jsonl`
- post-exact-eval candidate manifest:
  `.omx/research/hdm7_final_len_elided_candidate_post_exact_eval_manifest_20260514_codex.json`

## Routing Changes

- `src/tac/hnerv_frontier_defaults.py` now points active HNeRV routing at the
  HDM7 scorecard and entropy ranking artifacts.
- `ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE` is now
  `0.20636832361415344`.
- `tools/all_lanes_preflight.py` now requires the HDM7 exact eval row in the
  HNeRV scorecard gate.
- Focused tests assert the HDM7 required-eval and non-promotional CUDA
  reference cannot silently fall back to HDM6.
- The entropy planner now accepts HDM-style lossless decoder equivalence with
  runtime parse/equivalence proof even when byte-identical payload identity is
  intentionally false. This prevents post-eval HDM7 routing from being blocked
  by stale pre-dispatch `exact_cuda_auth_eval_missing` fields.

## Verification

```bash
.venv/bin/python tools/audit_hnerv_frontier_scorecard.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm7_codex/scorecard.json \
  --required-eval PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json \
  --required-eval PR106-R2-HDM4-HLM1-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm1_xmember_modal_t4_20260514/contest_auth_eval.json \
  --required-eval PR106-R2-HDM4-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM7-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/contest_auth_eval.json
```

Result: `PASS (16 rows, 3 payload groups, 48 follow-up targets, internal score-lowering=PR106-R2-HDM7-HLM2-XMEMBER (0.20636832361415344))`.

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_hnerv_frontier_entropy_ranking.py \
  src/tac/tests/test_build_hnerv_frontier_scorecard.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py \
  tests/test_parallel_dispatch_top_k_exact_ready_audit.py
```

Result: `45 passed in 1.08s`.

```bash
.venv/bin/python tools/all_lanes_preflight.py
```

Pre-stage result: Gate #6 passed with internal score-lowering frontier =
`PR106-R2-HDM7-HLM2-XMEMBER (0.20636832361415344)`.
Gate #10 failed only because this refresh's new research files were still
untracked; rerun after staging/commit is required before dispatch.

Post-stage result: `ALL 34 PREFLIGHT CHECKS PASSED`; Gate #6 reports
`internal score-lowering=PR106-R2-HDM7-HLM2-XMEMBER (0.20636832361415344)`.
