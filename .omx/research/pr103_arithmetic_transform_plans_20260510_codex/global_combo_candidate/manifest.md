# PR103 Arithmetic Histogram Candidate

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_archive_preflight: `false`
- ready_for_exact_eval_dispatch: `false`

## Archive Pair

- source: `experiments/results/public_pr_intake_full/public_pr103_intake_20260505_auto/archive.zip` / `178223` bytes / `31881b2d23d027e6619f2d8df2fe35d4d207d08882ec673d6c1b7ff119f18c30`
- candidate: `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/archive.zip` / `178211` bytes / `578c8f4e86eafc9dc04eefe61cc0e7f3f3f43e134ef4447cf9ef26fd23a23551`
- selection_mode: `global_combo_best`
- payload_byte_delta: `-12`
- archive_byte_delta: `-12`
- source_probe_delta_sum: `-13`
- non_additivity_delta: `1`
- semantic_stream_parity: `true` across `9` streams

## Runtime Constants To Derive

- `SCA_LEN`: `56`
- `BR_LEN`: `7097`
- `HIST_LEN`: `883`
- `MERGED_AC_LEN`: `153856`
- `LATENT_META_LEN`: `112`
- `LO_LEN`: `15537`
- `HI_HIST_LEN`: `15`
- `sidecar_corrections_brotli`: `payload_tail`
- `fixed_bytes_before_tail`: `177556`

## Section Diffs

- `ac_histograms_brotli`: `895` -> `883` (`-12`)
- `merged_range_coded_weights_and_hi_latents`: `153856` -> `153856` (`0`)

## Blockers

- `candidate_runtime_adapter_missing`
- `candidate_inflate_output_parity_missing`
- `strict_pre_submission_compliance_json_missing`
- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`
