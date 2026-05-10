# HNeRV hidden-gem PR103-ac adversarial review addendum - 2026-05-10

Candidate: `pr103_ac_merged_range_drop_u32_at_160824_20260510_agent`

Artifact root: `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/`

## Custody

- Source archive: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- Source archive bytes/SHA-256: `185578` / `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Source payload member: `0.bin`, `185470` bytes, SHA-256 `3272ec95a2ea5ec68feb1a53fa53f6b14bdae3883fac38ee2261cdadb1b16357`
- Candidate archive: `experiments/results/hnerv_hidden_gem_pr103_ac_candidate_20260510_agent/archive.zip`
- Candidate archive bytes/SHA-256: `185574` / `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278`
- Candidate payload member: `0.bin`, `185466` bytes, SHA-256 `416bd1593acac0da78e9ccbbe880095d73985d5c1c97b30377190d491f6cf3a6`
- ZIP structure: one stored member, `0.bin`; local/central header method, size, CRC, and name match.

## Adversarial Findings

The candidate is byte-different and runtime-consumed, but the first-pass proof understated the mutation. It is not exactly `source_payload[:160824] + source_payload[160828:]`: byte `1` in the PR106 packed length header also changes. The source header bytes at payload offsets `1:4` are `919602` (`169617`), and the candidate header bytes are `8d9602` (`169613`). This is required because the decoder section shrinks by four bytes.

The manifest now records `packed_header_update_proof`, and `tests/test_hnerv_hidden_gem_pr103_candidate_custody.py` locks the exact transform:

1. update the packed decoder-length header from `169617` to `169613`;
2. delete source payload bytes `eafe480c` at offset `160824`;
3. require the resulting payload to match the candidate exactly.

Runtime/local evidence remains local-only: dependency check passed for brotli/constriction/numpy/torch, `inflate.sh` parses under `bash -n`, generated runtime parsing decodes 28 tensors and `(600, 28)` latents, latents are unchanged, and decoded state tensors change. This supports "not a cosmetic ZIP/provenance-only edit"; it does not prove distortion.

## Not A Score Claim

`score_claim=false` remains correct. No exact CUDA auth eval was dispatched or run in this turn. The decoded state changes in `blocks.4.weight`, `blocks.5.weight`, and `refine.0.bias`, so the rate-only byte reduction cannot be promoted without measuring SegNet/PoseNet distortion through the official exact eval path.

This artifact is therefore `[local_decode_and_byte_custody_only]`, not `[contest-CUDA]`.

## Next Exact-Eval Requirements

- Claim the lane with `tools/claim_lane_dispatch.py` before any future remote exact eval.
- Use the candidate archive SHA-256 `8274e88c0ab1d26a06470a0730d17fe004556afa564460cf1c05624ff6060278` and runtime adapter under the artifact root.
- Run official CUDA auth eval on contest-equivalent hardware and record archive bytes/SHA, runtime tree SHA, command, logs, hardware, sample count, component distances, recomputed score, and terminal dispatch-claim row.
- Treat any distortion regression as a measured configuration update, not a lane kill, unless repeated exact-eval evidence proves the byte deletion is structurally invalid.
