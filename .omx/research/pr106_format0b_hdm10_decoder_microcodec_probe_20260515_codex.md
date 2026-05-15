# PR106 Format 0x0B HDM10 Decoder Microcodec Probe

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- verdict: `byte_positive_hdm10_planning_candidates_require_runtime_and_custody`

## Source

- archive: `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip`
- archive bytes: `186341`
- archive sha256: `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
- member: `x`

## HDM9 Decoder Tail

- charged decoder tail: `169946` bytes
- q payload: `169858` bytes
- scale low3: `84` bytes
- scale high mask: `4` bytes
- scale high mask hex: `81001c05`
- scale high mask popcount: `7`

## Candidate Rows

| candidate | raw equivalence | charged tail bytes | delta | runtime now | blockers |
|---|---:|---:|---:|---:|---|
| `hdm10_fixed_scale_mask_runtime_constant` | `true` | 169942 | -4 | `false` | `would_move_exact_scale_mask_payload_bytes_into_runtime_source`, `requires_new_decoder_microcodec_runtime`, `no_candidate_archive_emitted`, `full_frame_inflate_output_parity_missing`, `exact_cuda_dispatch_forbidden_in_this_task` |
| `hdm10_fixed_popcount_combinadic_scale_mask_rank` | `true` | 169945 | -1 | `false` | `requires_new_decoder_microcodec_runtime`, `fixed_popcount_is_payload_specific_runtime_constant`, `no_candidate_archive_emitted`, `full_frame_inflate_output_parity_missing`, `exact_cuda_dispatch_forbidden_in_this_task` |
| `hdm10_generic_popcount_plus_combinadic_scale_mask_rank` | `true` | 169946 | 0 | `false` | `no_charged_decoder_byte_drop`, `requires_new_decoder_microcodec_runtime`, `no_candidate_archive_emitted`, `full_frame_inflate_output_parity_missing` |
| `hdm9_current_tail_control` | `true` | 169946 | 0 | `true` | `control_candidate_no_charged_decoder_byte_drop`, `no_candidate_archive_emitted` |

## Decision

- byte-positive candidate count: `2`
- contest-safe byte-positive candidate count: `0`
- best byte-positive candidate: `hdm10_fixed_scale_mask_runtime_constant`
- best byte-positive delta bytes: `-4`

## Blockers

- `planning_probe_only_no_archive_emitted`
- `no_hdm10_runtime_decoder_implemented`
- `candidate_archive_manifest_missing`
- `full_frame_inflate_output_parity_missing`
- `score_claim_false_until_byte_closed_archive_runtime_exists`
- `ready_for_exact_eval_dispatch_false_by_task_scope`

## Recommendation

Do not dispatch. The only byte-positive rows require a new runtime decoder and payload-specific constants; route through a separate PacketIR design review before any archive builder exists.
