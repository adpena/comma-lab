# Replay Observability Signal

- schema: `archive_signal_table_v1`
- evidence_grade: `diagnostic_observability`
- score_claim: `False`
- dispatch_performed: `False`
- baseline: `PR79/S2` bytes=`277321` score=`0.31453355357318635`

This table is offline planning signal. It may guide the next dispatch, but exact CUDA auth eval on identical archive bytes is required for any score claim.

## Top Signals

| rank | source | kind | name | bytes | priority | rate delta vs baseline | guidance |
|---:|---|---|---|---:|---:|---:|---|
| 1 | exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z | component_trace_summary | trace_score_delta_vs_baseline | 260866 | 1.204689421 | -0.010956709 | byte saving can absorb this much component worsening |
| 2 | exact_eval_pr79_mask_body_crf53_t4_2x_20260503T1824Z | component_trace_summary | trace_score_delta_vs_baseline | 260866 | 1.204689421 | -0.010956709 | byte saving can absorb this much component worsening |
| 3 | exact_eval_pr79_mask_body_crf52_t4_20260503T1835Z | component_trace_summary | trace_score_delta_vs_baseline | 273141 | 1.172384153 | -0.002783290 | byte saving can absorb this much component worsening |
| 4 | exact_eval_pr79_nextbyte_crf52_pr79action_flatpack_t4_20260503T1848Z | component_trace_summary | trace_score_delta_vs_baseline | 259288 | 1.154018339 | -0.012007435 | byte saving can absorb this much component worsening |
| 5 | exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z | component_trace_summary | trace_score_delta_vs_baseline | 270941 | 1.109481320 | -0.004248180 | byte saving can absorb this much component worsening |
| 6 | archive.zip | stream | x | 296689 | 0.197553027 | 0.012962942 | exact CUDA eval required before score use |
| 7 | archive.zip | stream | p | 277221 | 0.184590085 | 0.000000000 | exact CUDA eval required before score use |
| 8 | PR82 | stream | mask | 219472 | 0.146137396 | 0.012962942 | exact CUDA eval required before score use |
| 9 | archive.zip | stream | p | 215860 | 0.143732314 | -0.040857771 | byte saving can absorb this much component worsening |
| 10 | PR81 | stream | range_mask.qma9 | 159011 | 0.105878898 | -0.040857771 | byte saving can absorb this much component worsening |
| 11 | PR82 | stream | model | 57074 | 0.038003234 | 0.012962942 | exact CUDA eval required before score use |
| 12 | PR81 | stream | split_model_reordered.br_bundle | 55725 | 0.037104990 | -0.040857771 | byte saving can absorb this much component worsening |
| 13 | exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z | component_atom | pair_0230 |  | 0.033764290 | -0.004248180 | repair/allocate if a charged atom can buy this pair budget |
| 14 | exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z | component_atom | pair_0419 |  | 0.023422227 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 15 | exact_eval_pr79_mask_body_crf53_t4_2x_20260503T1824Z | component_atom | pair_0419 |  | 0.023422227 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 16 | exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z | component_atom | pair_0331 |  | 0.020753094 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 17 | exact_eval_pr79_mask_body_crf53_t4_2x_20260503T1824Z | component_atom | pair_0331 |  | 0.020753094 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 18 | exact_eval_pr79_mask_body_crf53_t4_20260503T1828Z | component_atom | pair_0218 |  | 0.017052902 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 19 | exact_eval_pr79_mask_body_crf53_t4_2x_20260503T1824Z | component_atom | pair_0218 |  | 0.017052902 | -0.010956709 | repair/allocate if a charged atom can buy this pair budget |
| 20 | exact_eval_pr79_nextbyte_crf51_pr79action_flatpack_t4_20260503T1852Z | component_atom | pair_0216 |  | 0.016481342 | -0.004248180 | repair/allocate if a charged atom can buy this pair budget |

## Source Count

- sources: `12`
- stream rows: `30`
- component rows: `147`
