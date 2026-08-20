# SNeRV scorer-loop QAT local trainer

Schema: `snerv_scorer_loop_qat_local_trainer.v1`
Authority: `false_authority_macos_cpu_snerv_scorer_loop_no_contest_score_claim`
Axis: `[macOS-CPU advisory]`
Pairs: `4`
Search mode: `learned_random_subspace`
Component guard: `score_primary`
Adapter: `snerv_spectra_preserving_mfu_hfr_temporal_adapter_v1`
MFU scales: `(1, 2, 4)`
HFR gain: `0.0`
Decoder features: `9`
Evaluations: `17`
Baseline score: `1.1927130470929876`
Best score: `1.1654646854156452`
Accepted improvement: `True`
Receiver contract satisfied: `True`

## Blockers

- `local_smoke_only_not_full_600_pairs`
- `paired_contest_cpu_cuda_pass_missing`
- `mixed_precision_decoder_payload_grammar_not_byte_optimized`
- `full_600_pair_receiver_proof_missing`
- `official_snerv_mfu_hfr_tub_parity_not_proven`
