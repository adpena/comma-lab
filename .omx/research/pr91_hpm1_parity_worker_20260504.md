# PR91 HPM1 Entropy Decode/Replay Parity Worker - 2026-05-04

## Scope

Owned lane: PR91 HPM1 local entropy decode/replay parity only. No GPU dispatch,
no scorer loads, and no exact-eval claim.

Owned code/artifacts:

- `src/tac/pr91_hpm1_codec.py`
- `src/tac/tests/test_pr91_hpm1_codec.py`
- `experiments/results/public_pr91_intake_20260504_codex/diagnostics/*`
- `.omx/research/pr91_hpm1_parity_worker_20260504.md`

## Result

Status: **blocked, fail-closed**.

PR91 HPM1 static custody is valid for the downloaded archive:

- archive: `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- archive bytes/SHA-256: `222404`, `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- member `x` bytes/SHA-256: `222304`, `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- HPM1 mask bytes/SHA-256: `145087`, `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- token bytes/SHA-256: `116796`, `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPAC PPMd model bytes/SHA-256: `28243`, `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`

The HPAC model is byte-identical to PR86, but the token stream is distinct:
PR91 tokens are 2,896 bytes larger than PR86 tokens, share only the first 164
bytes / 41 uint32 words, and then diverge.

## Decode Evidence

Local CPU decode under the source PR86/PR91 probability contract fails closed:

- variant: `source_float64_perfect_false`
- failure: `submitted_tokens_decode` / `hpac_entropy_decode_contract_mismatch`
- coordinate: `frame=0`, `group=10`, `symbol_in_group=191`
- decoded symbols before failure: `5951`

Off-contract probes also fail before frame 0 completes:

- `source_float32_perfect_false`: `frame=0`, `group=24`, `symbol_in_group=561`
- `source_float64_perfect_true`: `frame=0`, `group=15`, `symbol_in_group=1534`
- `source_float32_perfect_true`: `frame=0`, `group=15`, `symbol_in_group=191`

No tested probability variant produced even frame-0 local decode parity, so no
decode-to-reencode byte parity was attempted for the real PR91 stream.

## Runtime Contract

Downloaded PR91 runtime source confirms:

- HPM1 branch is present and delegates to `pr86_hpac.decompress_tokens_hpac`.
- Probability rows are clipped/renormalized `float64` and use
  `Categorical(..., perfect=False)`.
- The HPM1 decoder receives the main runtime device via `str(device)`.
- No explicit HPM1 CPU force was detected.
- No fallback on HPM1 entropy failure was detected.

This means local CPU failure is a real replay blocker for this lane, and CUDA
behavior remains unproven here by scope.

## Diagnostics

Artifacts written:

- `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_static_contract_20260504_codex.json`
- `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_runtime_source_contract_20260504_codex.json`
- `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_preflight_frame0_20260504_codex.json`
- `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_probability_variant_matrix_frame0_20260504_codex.json`

## Tests

Commands run:

- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q`

Final result: `10 passed`.

## Classification

PR91 is **not ready for exact eval** from this lane. Required unblock before
dispatch: full local decode of the submitted HPM1 token stream plus source-
contract byte-exact re-encode parity, or a separately proven CUDA-only replay
contract with archive/runtime custody. This worker performed no GPU dispatch.
