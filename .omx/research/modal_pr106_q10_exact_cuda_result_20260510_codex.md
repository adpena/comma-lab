# Modal PR106 Q10 Exact CUDA Result (2026-05-10)

## Scope

This ledger records the corrected Modal T4 exact auth-eval for
`pr106_q10_151byte_brotli`, a low-level HNeRV Brotli repack candidate. The
candidate is a real rate-positive result relative to the PR106 source replay,
but it does not beat the active frontier floor and is not promotion-eligible.

## Custody

- Lane claim: `pr106_q10_151byte_brotli`
- Dispatch job: `pr106_q10_exact_cuda_modal_20260510T173900Z`
- Modal run: `ap-gMe3JmpoFs7nYcNZVth6ey`
- Archive:
  `experiments/results/hnerv_lowlevel_repack_pr106_q10_packet_20260507_codex/release_surface/archive.zip`
- Archive bytes: `186088`
- Archive SHA-256:
  `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- Inflate script:
  `experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh`
- Runtime tree SHA-256:
  `dbec651821fa45e349d5f05ba4159b09fb7d9f57b9d9f0991f8ae40febbd45e8`
- Modal artifacts:
  `experiments/results/modal_auth_eval/pr106_q10_exact_cuda_modal_20260510T173900Z/`

## Result

- Hardware: `Tesla T4`
- Evidence axis: `[contest-CUDA]`
- Wrapper verdict: `passed=true`
- Canonical recomputed score: `0.20936498680571203`
- Reported rounded score: `0.21`
- PoseNet distance: `0.00003351`
- SegNet distance: `0.00067151`
- Archive bytes: `186088`
- Sample count: `600`
- Validation errors: `[]`
- Promotion eligible: `false`

The raw auth-eval result still retained normal promotion blockers:

- `raw_auth_eval_does_not_verify_submission_policy_gates`
- `cpu_leaderboard_reproduction_not_adjudicated`
- `pre_submission_compliance_check_not_recorded`

## Baseline Comparison

Baseline PR106 adapter replay:

- Baseline artifact:
  `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json`
- Baseline score: `0.20945673680571203`
- Baseline PoseNet distance: `0.00003351`
- Baseline SegNet distance: `0.00067142`
- Baseline archive bytes: `186239`

Candidate delta vs PR106 source replay:

- Score delta: `-0.00009175`
- Byte delta: `-151`
- Pose delta: `0.0`
- Seg delta: `+0.00000009`

The observed score delta is close to the rate-only expectation
`-0.00010054470192144788`; the small remaining gap is explained by the tiny
SegNet drift. This confirms the repack is a useful low-level byte anchor, but
not a frontier candidate.

Active floor comparison:

- Active frontier floor: `0.2089810755823297`
- Candidate minus active floor: `+0.00038391122338233`

## Bit/Byte Classification

- Source archive bytes: `186239`
- Candidate archive bytes: `186088`
- Total archive delta: `-151`
- Changed sections:
  - `packed_header_ff_len24`: byte delta `0`, control/metadata update.
  - `decoder_packed_brotli`: byte delta `-151`, decoder weight stream.
- Raw equivalence proof:
  - `decoder_packed_brotli` raw bytes match source after Brotli decode.
  - `latents_and_sidecar_brotli` raw bytes match source after Brotli decode.

Adversarial interpretation: this is the safe version of low-level repacking.
It changes the compressed representation while preserving decoded raw sections,
unlike destructive range-stream deletion. It validates grammar-preserving
custom-codec work as a real route, but the rate delta is too small to escape
the local basin by itself.

## Score-Lowering Consequences

1. Do not spend more exact CUDA on small PR106 Brotli quality variants unless
   they also compose with a lower-distortion substrate or remove substantially
   more bytes.
2. Preserve this as a conformance and drift anchor for the custom packet
   compiler: decoded raw equivalence plus exact CUDA near-rate-only movement is
   the right contract.
3. Near-term score lowering should move to train-time/substrate engineering
   (active T1 Ballé job, HNeRV parity with eval-roundtrip/YUV6, custom codec
   compiler over decoded tensors) rather than further raw Brotli shaving.
4. Future custom-codec work should target larger sections with grammar-aware
   recoding, tensor-level saliency guards, and exact raw/tensor equivalence
   proofs before CUDA spend.

## Next Action

Use `pr106_q10_151byte_brotli` as a positive byte-repack proof, not as a
frontier. Keep T1 harvest as the highest-EV active score-lowering lane, and
build the next custom-codec tranche around deterministic packet compilation:
identity decode, grammar-preserving recode, tensor-diff proof, runtime closure,
then exact CUDA.
