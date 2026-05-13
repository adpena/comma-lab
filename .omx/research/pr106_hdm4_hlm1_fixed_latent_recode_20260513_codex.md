# PR106 HDM4 + HLM1 fixed-latent recode (2026-05-13, Codex)

## Summary

Built and wired a byte-closed HLM1 fixed-latent recode for PR106-style HNeRV
packets. This is a rate-only candidate, not a score claim.

HLM1 replaces the legacy fixed-latent Brotli section with:

- low-byte stream: deterministic Brotli choice;
- fp16 min/scale metadata: raw 112 bytes;
- high-byte stream: sparse delta positions for the 1 bits.

The high byte is binary by construction for the fixed PR106 latent grammar:
the quantized latent is uint8, first-order deltas are in `[-254, 254]`, and
zigzag(delta) therefore needs at most 9 bits. HLM1 exploits that invariant
without adding a `constriction` dependency to the inflate runtime.

## Current HDM4 Candidate

- Source archive: `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip`
- Source archive bytes: `186492`
- HLM1 candidate archive: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip`
- HLM1 candidate archive bytes: `186423`
- Archive byte delta: `-69`
- Rate-only score delta if components remain identical: `-0.00004594426776542982`
- Candidate archive SHA-256: `8801845d5099b957898fb6c6e58625bfb4cc065085ed2e3154c2cbc702dc91e0`
- PacketIR identity proof: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/packetir_identity.json`
- Runtime decode/sidecar consumption proof: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/runtime_decode_consumption.json`
- Same-runtime streaming prefix parity: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/same_runtime_prefix_parity.json`
- Entropy floor refresh: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/entropy_floor.json`
- Static release surface: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/static_release_surface/`
- Static compliance: `experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pre_submission_compliance.static_clean.json`

## Evidence Classification

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- `contest_axis_claim=false`
- `dispatch_attempted=false`

This candidate is parser-consumption and runtime-decode proven locally. It is
not a leaderboard score and must not be promoted until static release surface,
runtime-tree custody refresh, lane claim, and exact `[contest-CUDA]` auth eval
are complete.

## Landed Code

- `src/tac/packet_compiler/pr106_fixed_latent_recode.py`
- `src/tac/hnerv_hlm1_archive_candidate.py`
- `tools/build_pr106_hlm1_latent_candidate.py`
- `src/tac/pr103_pr106_runtime_closure.py` HLM1 decode support
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py` HLM1 runtime support
- `tools/pr106_entropy_floor_probe.py` HLM1 latent-section support
- Focused tests:
  - `src/tac/tests/test_pr106_fixed_latent_recode.py`
  - `src/tac/tests/test_pr106_entropy_floor_probe.py`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_fixed_latent_recode.py src/tac/tests/test_pr106_entropy_floor_probe.py src/tac/tests/test_pr103_pr106_runtime_closure.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
  - `28 passed in 26.39s`
- Targeted py_compile passed for the new primitive, candidate builder, CLI,
  runtime parser, and touched tests.
- Targeted fatal ruff gate passed with `--select E9,F63,F7,F82`.
- Static nonfinal pre-submission compliance passed on the clean release surface.
  The only failed check is the expected warning `auth_eval_optional_missing`;
  therefore the packet remains non-promotable until exact auth eval lands.
- Same-runtime CPU streaming prefix parity between HDM4 source and HDM4+HLM1
  candidate passed for `max_pairs=1`: `prefix_parity_claim=true`,
  `streaming_output_sha256_equal=true`, `score_claim=false`.

## Next Steps

1. Refresh runtime-tree custody after HLM1 runtime support lands in main.
2. Claim lane `hnerv_hlm1_fixed_latent_recode_exact_eval` before any GPU job.
3. Dispatch exact `[contest-CUDA]` auth eval only after the static packet is
   green.
