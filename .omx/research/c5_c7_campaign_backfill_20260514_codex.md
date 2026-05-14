# C5-C7 Campaign Backfill Ledger - 2026-05-14

parent_id_or_session: codex_scope_c5_c7_campaign_backfill_20260514
inherited_directives:
- .omx/research/journal_lab_grade_documentation_standard_directive_20260514.md
- .omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md
- .omx/research/lane_grand_council_maximize_value_20260514_directive_zen_floor_field_medal_grade_20260514.md
final_checkpoint_status: complete
research_only: true
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
gpu_spend_usd: 0

## 1. Hypothesis

The five unbackfilled long-term campaigns C5, C7, C2, C4, and C3 must be visible in the canonical cooperative-receiver campaign ranker and autopilot manifest with explicit `lane_class`, `campaign_id`, cost band, timeline metadata, and proxy dispatch gates, otherwise the long-burn roadmap can be lost by downstream ranker/autopilot consumers.

## 2. Math

Every row remains prediction-only under the contest score formula:

`S = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / 37,545,489`

Predicted value is only an ordering prior:

`EV_per_dollar_proxy = abs(midpoint(predicted_delta_S_band)) / midpoint(cost_usd_band)`

No row has measured `d_seg`, `d_pose`, archive bytes, or exact CUDA evidence, so all rows are forced through the proxy false-authority contract:

`score_claim = promotion_eligible = ready_for_exact_eval_dispatch = false`

## 3. Citations

- Atick and Redlich 1990: cooperative receiver / efficient coding.
- Wyner and Ziv 1976; Slepian and Wolf 1973: decoder-side information and distributed source coding.
- Liu et al. 2019 DARTS; Pham et al. 2018 ENAS: differentiable architecture search.
- Shannon 1959 vector rate-distortion: multi-axis floor framing.
- Internal source: `.omx/research/long_term_multi_year_campaign_roadmap_20260514.md`.

## 4. Provenance

Changed code:
- `src/tac/optimization/cooperative_receiver_campaigns.py`
- `src/tac/optimization/cooperative_receiver_integration.py`
- `src/tac/tests/test_cooperative_receiver_campaigns.py`
- `src/tac/tests/test_cooperative_receiver_integration.py`

No archive was built. No provider job was launched. No lane dispatch claim was created because this was metadata/ranker backfill only.

## 5. Empirical Evidence Tag

`[empirical: .venv/bin/python -m pytest -q src/tac/tests/test_cooperative_receiver_campaigns.py src/tac/tests/test_cooperative_receiver_integration.py]` passed locally: 7 tests.

## 6. Repro Recipe

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_cooperative_receiver_campaigns.py \
  src/tac/tests/test_cooperative_receiver_integration.py
```

Expected: `7 passed`.

## 7. Sister Lanes

Backfilled campaign IDs:
- `c5_full_cooperative_receiver_substrate_campaign_20260514`
- `c7_darts_supernet_architecture_search_campaign_20260514`
- `c2_z7_mature_predictive_receiver_l5_campaign_20260514`
- `c4_queued_architectural_moves_campaign_20260514`
- `c3_multi_year_zen_floor_sub_005_campaign_20260514`

Existing campaign rows for Z3/Z4/Z5/C1/C6 were not modified.

## 8. Six-Hook Wire-In

1. Sensitivity map: N/A for this metadata backfill; campaign ledgers name the future map IDs.
2. Pareto constraint: `build_pareto_constraint_rows()` now carries lane/cost/timeline metadata for the five rows and remains non-binding.
3. Bit allocator: N/A until byte-closed archive sections exist.
4. Cathedral autopilot dispatch hook: `build_autopilot_rows()` now exposes `campaign_id`, `lane_id`, `lane_class`, cost metadata, timeline metadata, and proxy blockers.
5. Continual-learning posterior: `build_continual_learning_policy()` still blocks posterior updates until exact artifacts exist.
6. Probe disambiguator: campaign ledgers name C5/C7/C2/C3 probes; this backfill preserves those campaigns in the queue.

## 9. Stop/Continue Thresholds

Continue if the campaign queue remains proxy-valid and the autopilot manifest exposes all five rows.
Stop before dispatch unless an operator separately authorizes spend and a lane claim is created.
Exact-eval threshold: no exact eval is allowed from these proxy rows until byte-closed archive/runtime custody exists.

## 10. Reactivation

If a future ranker omits any C2-C5-C7 campaign, rerun the focused tests above. If metadata drift is intentional, update this ledger and the tests with the new canonical campaign IDs.

## 11. Operator-Routable Decisions

1. Keep C5 gated on D4 frame-0 success before any spend.
2. Keep C7 full search gated on at least one C5/C6 empirical anchor; optional Stage 0 smoke still requires operator authorization.
3. Keep C2 and C3 operator-decision-required because their cost/timeline bands exceed normal short-horizon campaign scope.

next_action: none; metadata backfill is complete and dispatch remains blocked.
