# HNeRV HDM6 Active Scorecard Refresh - 2026-05-14

## Purpose

Refresh the active HNeRV routing surfaces after the HDM6 tuned-Brotli exact CUDA
anchor landed.

This is an internal optimizer-routing update, not a public promotion claim.
Score axis remains `[contest-CUDA]`.

## New Internal Score-Lowering Floor

- label: `PR106-R2-HDM6-HLM2-XMEMBER`
- score: `0.2063703211910128`
- archive bytes: `186408`
- archive sha256: `f3941568035d40bc7cb9e6fc0a108a5ec8bedf33f7ae14f6c060e92f7f247593`
- auth eval JSON: `experiments/results/modal_auth_eval/hnerv_hdm6_hlm2_modal_t4_20260514T081819Z/contest_auth_eval.json`
- result review packet: `.omx/research/hdm6_tuned_brotli_exact_cuda_result_review_20260514_codex.json`

## Refreshed Artifacts

- active scorecard: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm6_codex/scorecard.json`
- scorecard markdown: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm6_codex/scorecard.md`
- HDM6 section profile: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm6_codex/hdm6_section_profiles.json`
- entropy ranking: `experiments/results/hnerv_frontier_entropy_gap_ranking_20260514_hdm6_codex/frontier_entropy_gap_ranking.json`

## Routing Changes

- `src/tac/hnerv_frontier_defaults.py` now points active HNeRV routing at the
  HDM6 scorecard and entropy ranking artifacts.
- `ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE` is now
  `0.2063703211910128`.
- `tools/all_lanes_preflight.py` now requires the HDM6 exact eval row in the
  HNeRV scorecard gate.
- Focused tests assert the HDM6 required-eval and non-promotional CUDA
  reference cannot silently fall back to HLM2.

## Verification

```bash
.venv/bin/python tools/audit_hnerv_frontier_scorecard.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm6_codex/scorecard.json \
  --required-eval PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json \
  --required-eval PR106-R2-HDM4-HLM1-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm1_xmember_modal_t4_20260514/contest_auth_eval.json \
  --required-eval PR106-R2-HDM4-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM6-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm6_hlm2_modal_t4_20260514T081819Z/contest_auth_eval.json
```

Result: `PASS (15 rows, 3 payload groups, 45 follow-up targets, internal score-lowering=PR106-R2-HDM6-HLM2-XMEMBER (0.2063703211910128))`.

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py \
  src/tac/tests/test_build_hnerv_frontier_scorecard.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py
```

Result: `45 passed in 1.02s`.

```bash
.venv/bin/python tools/all_lanes_preflight.py
```

Result: `ALL 34 PREFLIGHT CHECKS PASSED`; Gate #6 reports
`internal score-lowering=PR106-R2-HDM6-HLM2-XMEMBER (0.2063703211910128)`.
