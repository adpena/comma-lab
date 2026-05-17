# L5 v2 Z6 post-L1 proxy surface hardening

## Change

The L5-v2 asymptotic candidate surface now consumes the Z6 real-video
ego-proxy sweep as post-L1 evidence instead of only reporting that the Z6 L1
scaffold exists.

## Why

The measured no-scorer real-video sweep found that all tested Z6 full-FiLM ego
proxies still lose to the identity predictor control. Without wiring that
artifact into `l5_v2_asymptotic_pursuit_candidates()`, operator-facing surfaces
could keep suggesting Z6 full-FiLM work based only on the completed L1 scaffold.
That is a retread/local-minimum failure mode.

## Evidence Consumed

- Tool: `tools/probe_z6_real_video_ego_proxy_sweep.py`
- Artifact: `.omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.json`
- Verdict: `identity_dominates_all_tested_ego_proxies_real_video_smoke`
- Best proxy: `random_control`
- Best `identity_minus_full_loss_proxy`: `-5.4001808166503906e-05`
- Authority: `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, `ready_for_paid_dispatch=false`

## Routing Effect

`src/tac/optimization/l5_staircase_v2.py` now emits
`post_l1_proxy_evidence` for the Z6/Z7/Z8 candidate and adds the blocker
`z6_full_film_paid_dispatch_blocked_identity_dominates_real_video_proxy_sweep`.

The recommended post-L1 action is now:

`advance_z6_only_with_posenet_or_scorer_ego_proxy_or_skip_to_z7`

This blocks spend on the measured Z6-v1 full-FiLM configuration, but does not
kill the L5 staircase or later Z7/Z8/Rudin/Tishby candidates.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  tools/build_l5_v2_asymptotic_candidate_surface.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_l5_staircase_v2.py \
  -k 'asymptotic_candidate_surface or asymptotic_pursuit_candidates'

.venv/bin/python tools/build_l5_v2_asymptotic_candidate_surface.py
```

Focused result: `6 passed, 102 deselected`.

## Authority

This is planning-surface hardening only. It makes the measured Z6 proxy result
harder to ignore in downstream operator/autopilot views. It is not a score
claim, not a promotion result, not a rank/kill result, and not an exact-eval
dispatch authorization.
