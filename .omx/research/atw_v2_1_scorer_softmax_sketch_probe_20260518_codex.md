# ATW V2-1 Scorer-Softmax Sketch Probe

- observed_at_utc: `2026-05-18T11:38:25+00:00`
- axis_label: `[diagnostic-CPU; ATW V2-1 scorer-softmax sketch MI probe]`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`
- provider_spend_attempted: `false`
- side_info_budget_bytes: `2048`
- phase2_status: `scorer_softmax_sketches_only_weak_or_biased_conditioning`
- recommended_next_gate: `trained_atw_residual_probe_or_raw_scorer_logit_head_design`

## Variant Results

| Variant | Packet bytes | Rate cost | Unique frac | MI bits/symbol | Verdict | Bias guard | Phase 2 action | Blockers |
|---|---:|---:|---:|---:|---|---|---|---|
| global_mean_softmax_q3 | 204 | 0.00013584 | 0.007 | 0.022207682205 | WEAK_CONDITIONING | false | pivot_to_raw_logit_head_or_trained_atw_residual_probe | scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold |
| global_top2_margin_q5 | 203 | 0.00013517 | 0.007 | 0.024223506458 | WEAK_CONDITIONING | false | pivot_to_raw_logit_head_or_trained_atw_residual_probe | scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold |
| region16_entropy_anchor_q4 | 209 | 0.00013916 | 0.007 | 0.016672440118 | WEAK_CONDITIONING | false | pivot_to_raw_logit_head_or_trained_atw_residual_probe | scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold |
| region16_presence_confmask_q4 | 213 | 0.00014183 | 0.007 | 0.026670502277 | WEAK_CONDITIONING | false | pivot_to_raw_logit_head_or_trained_atw_residual_probe | scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold |
| region256_coarse_entropy_anchor_q4 | 378 | 0.00025169 | 0.017 | 0.076162617811 | WEAK_CONDITIONING | false | pivot_to_raw_logit_head_or_trained_atw_residual_probe | scorer_softmax_sketch_did_not_reach_meaningful_mi_threshold |

## Verdict

Best sketch: `region256_coarse_entropy_anchor_q4` with verdict `WEAK_CONDITIONING`, MI `0.076162617811` bits/symbol, packet bytes `378`, high-cardinality guard `false`.

No scorer-softmax sketch is dispatch authority. The probe is either
weak, biased by high side-info cardinality, or over the configured
side-info budget. This keeps ATW V2-1 on the trained residual or raw
scorer-logit-head gate rather than paid dispatch.

## False-Authority Guard

This diagnostic artifact uses cached SegNet softmax arrays, not raw scorer logits.
It is not a contest score, not promotion evidence, and not provider-spend authority.

## Reproduction

- command: `.venv/bin/python tools/probe_atw_v2_1_scorer_softmax_sketch.py`
- output_dir: `experiments/results/atw_v2_1_scorer_softmax_sketch_probe_20260518T113825Z`
