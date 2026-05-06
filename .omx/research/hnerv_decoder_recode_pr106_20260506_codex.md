# HNeRV Decoder Recode PR106 - 2026-05-06

## Summary

PR106 HNeRV decoder structural recode was profiled and converted into a
byte-proved low-level brotli repack candidate. This is a rate-only,
raw-equivalent candidate, not a score claim.

## Source

- source_label: `PR106`
- source_archive: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- source_archive_bytes: `186239`
- source_archive_sha256: `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- source_decoder_section_bytes: `170278`
- source_decoder_section_sha256: `654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004`
- source_decoder_raw_bytes: `229070`

## Candidate

- candidate_archive: `experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/pr106_hnerv_brotli_repack_candidate.zip`
- candidate_archive_bytes: `186088`
- candidate_archive_sha256: `626b1c76d318eaed45198dc26aea7ee98c8a05f685b840356cf5b621bcddeea7`
- candidate_section_codec: `brotli`
- candidate_section_quality: `10`
- candidate_decoder_section_bytes: `170127`
- section_byte_delta: `-151`
- archive_byte_delta: `-151`
- rate_score_delta_if_components_equal: `-0.000100544702`

The candidate manifest reports `ready_for_archive_preflight=true` and no
candidate diff audit blockers. It remains `ready_for_exact_eval_dispatch=false`
until archive manifest preflight, dispatch claim, and exact CUDA auth eval are
completed.

## Negative Range-Coding Result

The existing arithmetic/range-style structural variants were byte-negative for
this PR106 decoder section:

- `aq_global_q_stream_plus_raw_scales`: `188991` bytes, `+18713`
- `canonical_huffman_global_q_stream_plus_raw_scales`: `189453` bytes, `+19175`
- `aq_per_tensor_q_streams_plus_raw_scales`: `198141` bytes, `+27863`

This does not kill arithmetic/range coding broadly. It only narrows the current
HNeRV decoder-section path: PR106's packed decoder is already better served by
deterministic brotli q10 recode than by the current AQ/Huffman structural
containers.

## Commands

```bash
.venv/bin/python tools/profile_hnerv_decoder_structural_recode.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --source-label public_pr106_belt_and_suspenders \
  --json-out experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json

.venv/bin/python tools/build_hnerv_lowlevel_repack_candidate.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip \
  --source-label PR106 \
  --output-dir experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex \
  --target-section decoder_packed_brotli \
  --quality 10 \
  --lgwin default \
  --jobs 1 \
  --json-out experiments/results/hnerv_lowlevel_repack_pr106_q10_20260506_codex/manifest.json

.venv/bin/python -m pytest \
  src/tac/tests/test_hnerv_decoder_recode.py \
  src/tac/tests/test_hnerv_lowlevel_packer.py \
  -q
```

## Verification

- `src/tac/tests/test_hnerv_decoder_recode.py src/tac/tests/test_hnerv_lowlevel_packer.py`: `10 passed`
- `score_claim=false`
- `dispatch_attempted=false`
- `ready_for_exact_eval_dispatch=false`

## Next

Run the normal archive preflight and exact CUDA auth eval path on the candidate
only after claiming the lane. If exact eval preserves components, this is a
small but clean rate improvement to fold into the HNeRV frontier stack.
