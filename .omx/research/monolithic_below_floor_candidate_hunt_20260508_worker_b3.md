# Monolithic Below-Floor Candidate Hunt - Worker B3 - 2026-05-08

Evidence grade: `empirical_disk_custody_no_score`.
Score claim: false.
Dispatch or lane claim: none.
Scope: read-only hunt across the current monolithic bridge, runtime proof,
CodecOp materialization surfaces, CMA/Optuna outputs, and PR101/PR106
parser-proven sections.

## Active Floor

Active rate-only floor used by the monolithic closure gate:

- archive path:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- bytes: `185578`
- SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- matching local build paths:
  `experiments/results/pr103_repack_pr106_standalone_20260507/archive.zip`
  and
  `experiments/results/pr103_repack_pr106_composed_op1_op2_20260507/archive.zip`
- exact-eval custody path:
  `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`

This hunt did not inspect or quote score fields from the active-floor
adjudication. The only comparator used here is archive byte count.

## Current Monolithic Bridge

Built monolithic control:

- candidate archive:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/pr106x_lgblock16_monolithic_candidate_from_manifest.zip`
- bytes: `186079`
- SHA-256:
  `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`
- source archive:
  `experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`
- source bytes: `186080`
- source SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- replacement payload:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/decoder_packed_brotli_lgblock16.section`
- replacement payload bytes: `170126`
- replacement payload SHA-256:
  `a812f1e837afd0e463a7f133b680ea6c027339ff8816db7012dd41253435afbf`
- source section bytes: `170127`
- source section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- rebuilt member SHA-256:
  `0a83096defc59120ee551c45e73f69e089165df78ae706fbbe2be3e9bc284765`
- manifest:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest_from_replacement_manifest.json`
- replacement manifest:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/codec_op_replacement_manifest.json`

Runtime proof is present:

- proof path:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_proof_from_log_pr106x_lgblock16.json`
- proof SHA-256:
  `81f6ebbca792de92d7d075b0d1b5c65bbc8a9ca068a40f22099e79a2447093b8`
- runtime log:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/runtime_consumption_pr106x_lgblock16.log`
- runtime log SHA-256:
  `f7396c712e3ac4574710c8c3152d9c2b629edf890963fb6245762f229a136f23`
- `ready_for_exact_eval_runtime=true`
- runtime blockers: `[]`

Closure result:

- closure gate:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/closure_gate_with_runtime_proof_no_claim_pr106x_lgblock16.json`
- closure blockers:
  `active_lane_claim_missing`,
  `rate_only_candidate_not_below_active_pr103_pr106_a_plus_plus_floor:185578`
- byte gap to floor: `186079 - 185578 = 501` bytes

Parser/rate interpretation:

- PR106 logical parser proof:
  `parser_proof_strength=magic_len_and_brotli_streams`
- candidate logical sections:
  `ff_header` 4 bytes,
  `decoder_packed_brotli` 170126 bytes,
  `latents_and_sidecar_brotli` 15849 bytes
- Brotli equivalence proof:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/brotli_equivalence_proof.json`
- decoder raw payload equality: true
- latent/sidecar payload equality: true

Verdict: this is a valid monolithic runtime-consumed control, but it is not a
below-floor candidate and it is not a meaningful scorer-changing path. It is a
rate-only one-byte repack above the active floor.

## CMA And Optuna Payload Outputs

Searched the current CMA/Optuna output directories for actual materialized
payload paths and section/blob files. No `materialized_payload_path`,
`replacement_payload_path`, `payload_path`, or `blob_path` entries were present
in these reports, and no `.section`, `.bin`, or `.blob` files exist in the
reviewed CMA/Optuna output directories.

Best planning rows on disk:

| Report | Successful evals | Best bytes_out | Best eval | Params | Materialized payload paths |
| --- | ---: | ---: | ---: | --- | ---: |
| `experiments/results/cma_pr101_real_substrate_20260507T222605Z/cma_pr101_search_report.json` | 24 | 162156 | 15 | `{"brotli_lgblock":19,"brotli_lgwin":13,"brotli_quality":10}` | 0 |
| `experiments/results/cma_pr101_real_substrate_cmaes_20260507T223229Z/cma_pr101_search_report.json` | 28 | 162154 | 10 | `{"brotli_lgblock":18,"brotli_lgwin":13,"brotli_quality":11}` | 0 |
| `experiments/results/optuna_pr101_real_substrate_20260507T230716Z/optuna_search_report.json` | 60 | 162150 | 33 | `{"brotli_lgblock":19,"brotli_lgwin":16,"brotli_quality":11}` | 0 |
| `experiments/results/optuna_pr101_real_substrate_hardened_20260507_codex/optuna_search_report.json` | 60 | 162151 | 23 | `{"brotli_lgblock":18,"brotli_lgwin":16,"brotli_quality":10}` | 0 |
| `experiments/results/optuna_pr101_known_best_probe_20260507_codex/optuna_search_report.json` | 1 | 162150 | 0 | `{"brotli_lgblock":19,"brotli_lgwin":16,"brotli_quality":11}` | 0 |

All five reports keep:

- `ready_for_exact_eval_dispatch=false`
- `score_affecting_payload_changed=false`
- `score_claim=false`
- dispatch blockers including
  `cpu_only_codec_op_search`,
  `no_archive_substitution_performed`,
  `no_score_affecting_payload_change_proof`,
  `missing_byte_closed_archive_manifest`,
  `missing_exact_cuda_auth_eval`

Verdict: the CMA/Optuna searches identify promising PR101 decoder byte counts,
but no existing materialized payload from those reports can be fed into the
monolithic bridge or PR101 archive surgery today. The next useful action for
those exact params is a new no-score materialization rerun with
`--materialized-payload-output-dir`, followed by PR101 substitution or a
monolithic replacement manifest only if the emitted bytes match a parser-proven
section contract.

## Existing PR101 CodecOp Materialized Archives

### PR101 lgwin18 Candidate

This is the only reviewed existing materialized archive that is below the
active 185578-byte rate-only floor and has changed charged decoder bytes.

- candidate archive:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/archive.zip`
- full runtime packet archive copy:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/full_runtime_packet/archive.zip`
- bytes: `178258`
- SHA-256:
  `c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933`
- source archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
- source bytes: `178258`
- source SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- candidate inner member bytes: `178158`
- candidate inner member SHA-256:
  `a7df36c2b11f88ca640e1bb259a526f80190c86e474e9b06204bb54657d3356c`
- source inner member SHA-256:
  `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf`
- candidate decoder blob bytes: `162164`
- candidate decoder blob SHA-256:
  `ea7155370f0adaab5d0078ef9158de7abe8363ff7a09afd26cf699ed8b81600d`
- source decoder blob SHA-256:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
- latent blob bytes/SHA-256:
  `15387`,
  `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe`
- sidecar blob bytes/SHA-256:
  `607`,
  `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`
- validation:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/lgwin18_candidate_validation.json`
- local decoder parity:
  `decoder_state_dict_parity.passed=true`,
  `compared_tensors=28`
- runtime custody:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/full_runtime_packet/runtime_custody_manifest.json`
- runtime tree SHA-256:
  `5c822fbbf481a278b253d7099c1cf1f69795d828b0663f7b256b517661422ed3`
- current blockers:
  `active_level2_lane_dispatch_claim`,
  `exact_cuda_auth_eval`,
  `contest_auth_eval_adjudication`,
  `operator_score_claim_review`
- validation exact-eval blockers:
  exact runtime parity/runtime-tree custody not established by the CPU
  validator,
  matching active lane dispatch claim required before GPU dispatch,
  CUDA auth eval not run

Verdict: this is a real below-floor rate-only candidate archive by byte count
(`178258 < 185578`), and charged decoder bytes differ. It is not a
score/scorer-changing proof: the decoded PR101 decoder state matches the source
state exactly, latent and sidecar bytes are preserved, and no CUDA auth eval
has run on this candidate. It is a concrete next local build target, not a
promotion or dispatch target.

Concrete next no-dispatch build command:

```bash
.venv/bin/python tools/build_pr101_runtime_packet.py \
  --candidate-archive experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/archive.zip \
  --packet-dir experiments/results/pr101_codecop_lgwin18_candidate_20260508_worker_b3_runtime_parity/full_runtime_packet \
  --candidate-id pr101_lgwin18_hnerv_ft_microcodec_worker_b3 \
  --run-local-inflate-parity \
  --parity-dir experiments/results/pr101_codecop_lgwin18_candidate_20260508_worker_b3_runtime_parity/local_inflate_parity \
  --inflate-timeout-seconds 3600 \
  --force
```

This command builds a fresh bounded runtime-parity packet only. It does not
claim, dispatch, import contest scorers, or run CUDA auth eval.

### PR101 native op1 sweep archive

- candidate archive:
  `experiments/results/pr101_codecop_sweep_20260507_codex/substituted/pr101_native_op1_pr101_native_op1_auto_selectFalse_brotli_quality11/archive.zip`
- bytes: `178258`
- SHA-256:
  `87849d0097788c0295ad8954ef3f2e64db5a4fa504d5a8809d63c1e35ef3cf08`
- substitution report:
  `experiments/results/pr101_codecop_sweep_20260507_codex/substituted/pr101_native_op1_pr101_native_op1_auto_selectFalse_brotli_quality11/substitution_report.json`
- replacement decoder SHA-256:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
- source decoder SHA-256:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`

Verdict: below-floor by archive bytes, but not useful as a scorer-changing or
rate-changing candidate. The replacement decoder blob equals the source
decoder blob; the archive SHA differs without useful payload change.

## PR101 And PR106 Parser-Proven Sections

PR101 parser-proven source sections:

- archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
- bytes/SHA-256:
  `178258`,
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- member `x` bytes/SHA-256:
  `178158`,
  `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf`
- logical sections:
  `decoder_blob` offset 0 len 162164 SHA-256
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`;
  `latent_blob` offset 162164 len 15387 SHA-256
  `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe`;
  `sidecar_blob` offset 177551 len 607 SHA-256
  `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`

PR106/PR106x parser-proven source/control sections:

- PR106 source archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- PR106 source bytes/SHA-256:
  `186239`,
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- PR106 source logical sections:
  `ff_header` len 4 SHA-256
  `7939f08db7d18dd4176e8e11b1232a9be6b2371db7a4e63c1c0871fb520148b6`;
  `decoder_packed_brotli` len 170278 SHA-256
  `654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004`;
  `latents_and_sidecar_brotli` len 15849 SHA-256
  `94257b33cf3083c5daa0f3b1e127cb7c51bee42a6416b19763eea7bf9ecc3c32`
- PR106x lgblock16 control logical sections are recorded in
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/layout.json`

The PR106 section grammar is monolithic: logical edits must target
`decoder_packed_brotli` or `latents_and_sidecar_brotli` inside the single
stored member. No reviewed materialized payload other than the lgblock16
one-byte control exists for those sections.

## Final Determination

No existing materialized PR106/monolithic payload produces an archive below the
active `185578` byte floor. The only runtime-proven monolithic control is
`186079` bytes and blocked by the floor gate.

The CMA/Optuna outputs do not currently include materialized payload files, so
they cannot produce an archive from existing bytes despite better `bytes_out`
planning rows.

One existing PR101 CodecOp archive is a real below-floor rate-only candidate:
`pr101_lgwin18` at `178258` bytes,
SHA-256 `c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933`.
It is not a scorer-changing proof and remains blocked on local inflate parity,
Level-2 claim, exact CUDA auth eval, adjudication, and operator review.

No meaningful scorer-changing stack path is proven by existing materialized
payloads in this surface. The concrete next build is the no-dispatch PR101
runtime parity command above; the concrete next monolithic work is to
materialize the best Optuna/CMA PR101 params or a real PR106 section payload
with `--materialized-payload-output-dir`, then bridge only if the emitted bytes
bind to a parser-proven section.
