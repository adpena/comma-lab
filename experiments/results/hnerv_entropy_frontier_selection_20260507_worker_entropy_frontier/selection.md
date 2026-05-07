# HNeRV Entropy Frontier Next-Candidate Selection

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`

## Active Excluded Candidate

- label: `active_pr103_pr106`
- archive_bytes: `185578`
- archive_sha256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- active_candidate_byte_floor: `185578`

## Selected Next Candidate

- none

## Ranked Candidates

| label | bytes | exact-evaluable after claim | blockers |
|---|---:|---|---|
| pr101_split_brotli | 185998 | `false` | static_exact_eval_packet_not_ready, pr101_split_brotli_runtime_adapter_not_yet_integrated, pr106_inflate_will_fail_on_pr101_decoder_format, not_below_active_candidate_byte_floor:185578 |
| pr101_schema | 186044 | `false` | static_exact_eval_packet_not_ready, pr101_schema_runtime_tree_parity_manifest_missing, pr101_schema_inflate_output_parity_missing, strict_pre_submission_compliance_json_missing, not_below_active_candidate_byte_floor:185578 |
| hdm3 | 186066 | `false` | not_below_active_candidate_byte_floor:185578 |
| pr106_q10 | 186088 | `false` | static_exact_eval_packet_not_ready, requires_archive_manifest_preflight, not_below_active_candidate_byte_floor:185578 |

## Dispatch Boundary

This manifest is local custody/readiness only. It does not claim a
score and does not authorize or perform GPU dispatch.
