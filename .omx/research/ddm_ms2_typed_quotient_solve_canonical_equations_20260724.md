# DDM MS2 typed quotient solve — canonical equations

Authority is structural only: `[macOS-CPU frozen-scorer advisory]`,
`score_claim=false`, pointer unchanged. All rows were registered exactly once
through `tac.canonical_equations.registry` against the preserved pre-PF2
receipt SHA-256
`9b17c5108e4b8d5a517ecb66276fc0e78162e54b53a9f4d819a48286989b98b6`.
The later PF2 MAIN landing and current receipt SHA-256
`04060edf9834b661f12a9794e50ceadf7dd4ab114baf55a15555537abc71e419`
are linked by locked append-only `domain_refined` events, not duplicate
registrations.

| Equation ID | Callable | Law |
|---|---|---|
| `ddm_ms2_visible_quotient_gauge_zero_v1` | `visible_quotient_counted_bytes` | \(B=B_{\mathrm{visible}}+0_{\mathrm{gauge}}\) |
| `ddm_ms2_scorer_metric_second_order_action_v1` | `scorer_metric_rate_action` | \(\mathcal A=\frac12\delta^\top H_{\mathrm{score},R}\delta+\frac{25}{N}B\) |
| `ddm_ms2_typed_block_dual_exchange_v1` | `typed_block_exchange_rate` | \(\rho_b=\Delta S_b/\Delta B_b\), with unpooled measured \(\lambda_b\) |
| `ddm_ms2_effective_quantum_admission_v1` | `effective_quantum` | \(q_i^{\mathrm{eff}}=\Delta u_i^{\mathrm{uint8}}s_i^{\mathrm{score}}\) |
| `ddm_ms2_skeleton_fiber_coder_race_v1` | `skeleton_fiber_coder_race` | choose SKELETON/FIBER by exact counted bytes after semantic parse-back |

The callable module is
`tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724`. Imports and
temporary locked-registry population are regression-tested. These laws do not
claim that the n600 solve ran; their domain excludes identity-Euclidean
verdicts, pooled/imputed duals, unlanded sister snapshots, and score/promotion
claims. Their named consumers are the train-decision SOLVE column and
`pf2r.metric_active_three_formulation_rerun`, the latter held under
`PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE`.
