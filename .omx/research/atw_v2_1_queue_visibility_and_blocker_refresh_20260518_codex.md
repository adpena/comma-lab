# ATW V2-1 Queue Visibility And Blocker Refresh - 2026-05-18

## Scope

This is a readiness/control-plane landing, not a score claim and not dispatch
authority. The ATW V2-1 Faiss-IVF-PQ WIP now appears in the canonical
asymptotic-pursuit queue as its own substrate row instead of being hidden under
the older `atw_codec_v2` D4 blocker.

The queue row intentionally remains `DEFER`.

## Current Evidence

ATW V2-1 now has all of the following current-state surfaces:

- lane: `lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518`
- recipe: `.omx/operator_authorize_recipes/substrate_atw_v2_1_modal_t4_smoke_dispatch.yaml`
- trainer scaffold: `experiments/train_substrate_atw_v2_1.py`
- channel helper: `src/tac/optimization/faiss_ivf_pq_atw_channel.py`
- disambiguator: `tools/probe_atw_v2_1_faiss_pq_disambiguator.py`
- diagnostic result: `.omx/research/atw_v2_1_faiss_pq_disambiguator_probe_20260518_codex.json`

Diagnostic result summary:

```text
[diagnostic-CPU; ATW V2-1 Faiss-PQ side-info MI probe]
v3_pool_shared: WEAK_CONDITIONING, MI=0.121512378237, bytes=3114, rate=0.002073484780
v2_sparse_top_k: MEANINGFUL_CONDITIONING, MI=2.457397664695, bytes=7941, high-cardinality upper-bound only
v1_dense: MEANINGFUL_CONDITIONING, MI=2.457397664695, bytes=452799, high-cardinality upper-bound only and over budget
phase2_status=pq_variants_not_dispatch_authority_upper_bound_or_weak
recommended_next_gate=pivot_to_scorer_logit_compression_or_trained_atw_residual_probe
```

## Blocker Refresh

The recipe previously still carried the stale sequencing blocker:

```text
z6_wave_2_4c_outcome_pending_cross_pollination_per_atw_v2_symposium_revision_5
```

Candidate 4c's zero-epoch full-vs-identity paired exact eval has now landed on
both `[contest-CUDA]` and `[contest-CPU]`, but it did not validate the
scorer-logit channel at the required `delta_s >= 0.005`. The ATW V2-1 blocker
was therefore updated to the measured outcome:

```text
z6_wave_2_4c_zeroepoch_exact_outcome_did_not_validate_scorer_logit_channel_delta_below_0_005
```

This preserves the cross-pollination signal without pretending the dependency
is still pending.

## Queue Artifact

Generated artifact:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T112720Z.json
```

Targeted row:

```text
substrate_id=atw_codec_v2_1_faiss_ivf_pq
readiness_verdict=DEFER
horizon_class=unknown
predicted_score_band=null
```

First blockers:

```text
CATALOG_240_FULL_MAIN_BLOCKED:RAISES_NotImplementedError
CATALOG_315_COUNCIL_PROCEED_WITH_REVISIONS_NEEDS_ITERATION_TO_OPTIMAL_FORM
RECIPE_research_only=true_OPERATOR_NEEDS_TO_FLIP_AFTER_PHASE_2_COUNCIL
RECIPE_DISPATCH_BLOCKER:faiss_pq_disambiguator_completed_20260518_v3_pool_shared_only_weak_conditioning_mi_0_121512378237
RECIPE_DISPATCH_BLOCKER:faiss_pq_v2_sparse_top_k_meaningful_mi_high_cardinality_upper_bound_only_unique_fraction_1_0
RECIPE_DISPATCH_BLOCKER:faiss_pq_v1_dense_meaningful_mi_high_cardinality_upper_bound_only_and_over_budget_rate_cost_0_301500268115
RECIPE_DISPATCH_BLOCKER:faiss_pq_selected_next_gate_is_pivot_to_scorer_logit_compression_or_trained_atw_residual_probe
RECIPE_DISPATCH_BLOCKER:z6_wave_2_4c_zeroepoch_exact_outcome_did_not_validate_scorer_logit_channel_delta_below_0_005
```

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/train_substrate_atw_v2_1.py \
  src/tac/optimization/faiss_ivf_pq_atw_channel.py \
  tools/probe_atw_v2_1_faiss_pq_disambiguator.py \
  src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py \
  src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py
# rc=0

.venv/bin/python -m pytest -q \
  src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py \
  src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py
# 36 passed in 0.37s

.venv/bin/python -m py_compile \
  tools/asymptotic_pursuit_candidate_readiness_assessment.py \
  tools/asymptotic_pursuit_dispatch_queue.py \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# rc=0

.venv/bin/python -m pytest -q \
  src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py
# 63 passed in 14.55s

.venv/bin/python tools/asymptotic_pursuit_dispatch_queue.py --write-artifact --json \
  > /tmp/pact_queue_after_atw_v21.json
# wrote .omx/state/asymptotic_pursuit/dispatch_queue_20260518T112720Z.json
```

## Result Classification

- classification: queue/readiness artifact repair plus stale-blocker refresh
- provider_dispatch_attempted: false
- lane_claim_opened: false
- score_claim: false
- promotion_eligible: false

Next score-moving ATW action is not paid Faiss-PQ dispatch. It is either a
scorer-logit compression channel probe or a trained ATW residual probe that can
replace the weak/high-cardinality PQ channel with a byte-closed channel capable
of passing a new D4 `MEANINGFUL_CONDITIONING` gate.

## Codex Addendum: Scorer-Softmax Sketch Gate

The cached SegNet softmax arrays from the Faiss-PQ probe were converted into
five deterministic ATW21SI dictionary packets to test the immediately
available scorer-derived sketch family before spending on any ATW V2-1 run.

Artifact surfaces:

```text
tool: tools/probe_atw_v2_1_scorer_softmax_sketch.py
state: .omx/state/atw_v2_1_scorer_softmax_sketch_probe.json
research_json: .omx/research/atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.json
research_md: .omx/research/atw_v2_1_scorer_softmax_sketch_probe_20260518_codex.md
local_packets: experiments/results/atw_v2_1_scorer_softmax_sketch_probe_20260518T113825Z/
cached_softmax_source: experiments/results/atw_v2_1_faiss_pq_probe_20260518T100524Z/
```

Measured result:

```text
[diagnostic-CPU; ATW V2-1 scorer-softmax sketch MI probe]
global_mean_softmax_q3: WEAK_CONDITIONING, MI=0.022207682205, packet_bytes=204
global_top2_margin_q5: WEAK_CONDITIONING, MI=0.024223506458, packet_bytes=203
region16_entropy_anchor_q4: WEAK_CONDITIONING, MI=0.016672440118, packet_bytes=209
region16_presence_confmask_q4: WEAK_CONDITIONING, MI=0.026670502277, packet_bytes=213
region256_coarse_entropy_anchor_q4: WEAK_CONDITIONING, MI=0.076162617811, packet_bytes=378
phase2_status=scorer_softmax_sketches_only_weak_or_biased_conditioning
recommended_next_gate=trained_atw_residual_probe_or_raw_scorer_logit_head_design
```

All five packets fit the 2048-byte side-info budget and none triggered the
high-cardinality bias guard. The failure is therefore not byte closure; the
available softmax sketches are simply too weak under this MI gate. This closes
the cached-softmax sketch branch as dispatch authority and sharpens the next
ATW V2-1 action to a trained residual channel or a probe that captures raw
scorer logits before softmax compression.

Verification:

```bash
.venv/bin/python -m py_compile tools/probe_atw_v2_1_scorer_softmax_sketch.py
# rc=0

.venv/bin/python -m pytest src/tac/tests/test_probe_atw_v2_1_scorer_softmax_sketch.py -q
# 4 passed in 0.50s

.venv/bin/python tools/probe_atw_v2_1_scorer_softmax_sketch.py
# phase2_status=scorer_softmax_sketches_only_weak_or_biased_conditioning
```

Queue refresh after wiring the blocker:

```text
.omx/state/asymptotic_pursuit/dispatch_queue_20260518T114053Z.json
```

The canonical queue row remains `DEFER` and now exposes the new recipe blockers:

```text
RECIPE_DISPATCH_BLOCKER:scorer_softmax_sketch_completed_20260518_all_byte_closed_but_best_mi_0_076162617811_weak
RECIPE_DISPATCH_BLOCKER:selected_next_gate_is_trained_atw_residual_probe_or_raw_scorer_logit_head_design
```
