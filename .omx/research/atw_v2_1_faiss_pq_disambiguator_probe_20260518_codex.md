# ATW V2-1 Faiss-PQ Disambiguator

- observed_at_utc: `2026-05-18T10:21:09+00:00`
- axis_label: `[diagnostic-CPU; ATW V2-1 Faiss-PQ side-info MI probe]`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`
- provider_spend_attempted: `false`
- phase2_status: `pq_variants_not_dispatch_authority_upper_bound_or_weak`
- recommended_next_gate: `pivot_to_scorer_logit_compression_or_trained_atw_residual_probe`

## Variant Results

| Variant | Archive bytes (Brotli) | Rate cost | Unique frac | MI bits/symbol | Verdict | Bias guard | Blockers |
|---|---:|---:|---:|---:|---|---|---|
| v3_pool_shared | 3114 | 0.002073 | 0.033 | 0.121512378237 | WEAK_CONDITIONING | false | pq_variant_did_not_reach_meaningful_conditioning_threshold |
| v2_sparse_top_k | 7941 | 0.005288 | 1.000 | 2.457397664695 | MEANINGFUL_CONDITIONING | true | pq_side_info_high_cardinality_plugin_mi_upper_bound_only, actual_pq_payload_exceeds_v3_shippable_5kb_target |
| v1_dense | 452799 | 0.301500 | 1.000 | 2.457397664695 | MEANINGFUL_CONDITIONING | true | pq_side_info_high_cardinality_plugin_mi_upper_bound_only, actual_pq_payload_exceeds_v3_shippable_5kb_target |

## Best Variant

`v2_sparse_top_k` produced MI `2.457397664695` bits/symbol with `7941` bytes. Bias guard triggered: `true`.

## False-Authority Guard

This diagnostic artifact is not a score claim and is not dispatch authority.
A paid ATW V2-1 run still requires a selected channel, a new D4 probe,
sextet council ratification, and paired contest CPU/CUDA harvest.
