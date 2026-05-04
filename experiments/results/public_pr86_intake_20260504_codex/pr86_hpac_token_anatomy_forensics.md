# PR86 HPAC/token anatomy forensics

- `score_claim=false`
- `dispatch_performed=false`
- source archive: `experiments/results/public_pr86_intake_20260504_codex/archive.zip`
- source bytes/SHA-256: `207579` / `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef`
- expected identity match: `True`
- stable parse digest: `b7438ee66e096f5b2840e93e9793ff43ae7ccaa6d5d208175577fd8bc572ed7f`

## Member byte anatomy

| member | role | bytes | share | sha256 |
| --- | --- | ---: | ---: | --- |
| `master.pt.gz` | TokenRendererV62 master weights | 31144 | 0.150 | `3f3ee2b19ba5cf97017559750c0d64bc422c3f84fedaed1877741ee6c6bd5236` |
| `slave.pt.gz` | ShrinkSingleNeRV slave weights | 32287 | 0.156 | `817294dea0d940a8ef62c190bf96338f5a756930882f0f0d7f4d7c7eb87a82a8` |
| `hpac.pt.ppmd` | HPACMini entropy model weights | 28243 | 0.136 | `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd` |
| `tokens.bin` | constriction queue-coded token stream | 113900 | 0.549 | `14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225` |
| `meta.pt` | runtime metadata | 1499 | 0.007 | `848381e2da1b0f307670174f135a3925c43d8cdc73576b4bf05fadf833de4a08` |

## Fail-closed custody

- sidecar/member status: `passed_exact_required_member_set`
- duplicate members: `[]`
- missing required members: `[]`
- unexpected members: `[]`
- unsafe member names: `[]`

## Token/HPAC contract

- submitted token encoding: `raw_tokens`
- training objective: `residual_tokens`
- archive writes raw GT tokens: `True`
- inflate reconstructs residuals: `False`
- constriction queue decoder present: `True`
- Categorical perfect=False: `True`
- probability clip epsilon: `1e-7`

## Current exact replay branch

- status: `archive_validator_whitelist_blocked`
- evidence grade: `invalid`
- score claim from this report: `false`

## PR85 transplant opportunities

### `hpac_reencode_pr85_mask_tokens`

- target: Replace PR85 QMA9 mask segment with an HPAC-coded token stream.
- drop-in status: `not_drop_in`
- gross byte math: `{"gross_saved_bytes_if_same_contract": 15369, "pr85_mask_segment_bytes": 159011, "pr86_hpac_tokens_meta_bytes": 143642}`
- score claim: `False`

### `full_pr86_runtime_as_external_baseline`

- target: Treat PR86 as a full external neural-codec baseline, not a PR85 transplant.
- drop-in status: `full_runtime_replacement_only`
- gross byte math: `{"gross_saved_bytes_vs_pr85_archive": 28749, "pr85_archive_bytes": 236328, "pr86_archive_bytes": 207579, "pr86_model_stack_bytes": 63431}`
- score claim: `False`

### `hpac_probability_contract_port`

- target: Port the HPAC probability model/coder contract into an Apogee-owned PR85 mask coder.
- drop-in status: `design_prior_only`
- gross byte math: `{"pr86_hpac_model_bytes": 28243, "pr86_meta_bytes": 1499, "pr86_tokens_bytes": 113900}`
- score claim: `False`

### `pr86_model_stack_lessons_for_pr85_nonmask_bytes`

- target: Use master/slave neural-renderer compression ideas against PR85 model/post/pose bytes.
- drop-in status: `not_drop_in`
- gross byte math: `{"gross_saved_bytes_if_replacing_all_nonmask_bytes": 13886, "pr85_non_mask_archive_bytes_estimate": 77317, "pr86_master_slave_bytes": 63431}`
- score claim: `False`

## Current branch actions

- Current local mirror branch: archive validator whitelist blocked before score JSON.
- No score claim is available; do not compare PR86 numerically to PR85 from this run.
- Fix or preflight the member-format contract before any future exact replay attempt.
