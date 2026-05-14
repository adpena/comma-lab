# HNeRV HLM2 latent microcodec

Date: 2026-05-14
Author: Codex

## Summary

HLM2 is a byte-closed, lossless successor to HLM1 for the PR106 fixed-latent
section. It preserves HLM1's low-byte Brotli stream, fp16 min/scale metadata,
and sparse high-byte delta-position stream, but removes two redundant u16 fields:
`hi_delta_varint_len` and `hi_nonzero_count`. Both are derivable from packet
boundaries when decoding the high-byte delta stream until EOF.

This is intentionally small. It is a real archive-byte reduction, not a score
claim and not a promotion claim.

## Candidate artifact

- Source archive: `experiments/results/pr106_r2_hdm4_hlm1_xmember_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm1_latent_candidate.zip`
- Source SHA-256: `391400008b69e66f8bd522f4eb2a53c465e58a17e536d171caf039f9e51e874f`
- Source bytes: `186415`
- Candidate archive: `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/pr106_r2_hdm4_hlm1_xmember_hlm2_latent_candidate.zip`
- Candidate SHA-256: `2c6e5f8d71f687227a28a9a378dc5edfc3215b762015042203b6bf58bfee9378`
- Candidate bytes: `186411`
- Archive byte delta: `-4`
- Candidate manifest: `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/hlm2_candidate_manifest.json`

Rate-only score delta if components are identical:

```text
-4 * 25 / 37,545,489 = -0.00000266344
```

## Local proofs

- Runtime consumption proof:
  `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/hlm2_runtime_consumption.json`
- Prefix same-runtime parity:
  `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/same_runtime_prefix8_parity.json`
- Full same-runtime parity:
  `experiments/results/pr106_r2_hdm4_hlm2_latent_candidate_20260514_codex/same_runtime_full_frame_parity.json`

Full same-runtime parity result:

- `full_frame_inflate_output_parity_claim=true`
- `n_pairs_hashed=600`
- `total_frames=1200`
- source streaming raw SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- candidate streaming raw SHA-256:
  `e6d9170b92997db1e6211eeb665187a3ac6a23c743dd3659c46633e509af329f`
- `score_claim=false`
- `contest_axis_claim=false`
- `ready_for_exact_eval_dispatch=false`

## Code surfaces

- `src/tac/packet_compiler/pr106_fixed_latent_recode.py`
  - adds `HLM2_MAGIC`, HLM2 encode/decode, and generic HLM dispatch.
- `src/tac/hnerv_hlm1_archive_candidate.py`
  - adds `latent_codec=hlm1|hlm2` while keeping HLM1 as the default.
- `submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py`
  - adds contest-runtime HLM2 decode support.
- `src/tac/packet_compiler/pr106_hlm1_runtime_consumption.py`
  - adds generic HLM runtime-consumption proof while preserving HLM1 compatibility fields.
- `tools/build_pr106_hlm1_latent_candidate.py`
  - adds `--latent-codec hlm2`.
- `tools/prove_pr106_hlm1_runtime_consumption.py`
  - adds `--allowed-codec hlm2`.
- `tools/pr106_entropy_floor_probe.py`
  - labels HLM2 latent sections distinctly.
- `tools/all_lanes_preflight.py`
  - refreshes the PR106 PR101-grammar runtime source-tree hash after adding HLM2 decode support:
    `059065e1cf77a63afd9f80a56cbddad140f61e12cfed57f11dbc8f68c5a5cf0a`.

## Exact-eval routing

Modal T4 exact CUDA auth eval was dispatched and recovered under claimed lane
`hnerv_hlm2_fixed_latent_recode_exact_eval`.

- Modal run URL: `https://modal.com/apps/adpena/main/ap-Ucg3vk9ctChN442vTPl2nN`
- Modal call id: `fc-01KRJMGQXSF5KYQJ831PAMMY4Q`
- Output directory:
  `experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z`
- Auth eval JSON:
  `experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json`
- Result-review packet:
  `.omx/research/hdm4_hlm2_xmember_exact_cuda_result_review_20260514_codex.json`
- Autopilot evidence row:
  `reports/hdm4_hlm2_xmember_exact_cuda_evidence_row_20260514_codex.json`
- Dispatch terminal status: `completed_contest_cuda_modal_auth_eval_recovered`

Exact `[contest-CUDA reviewed]` result:

- score: `0.20637231876787215`
- avg PoseNet dist: `0.00003236`
- avg SegNet dist: `0.00064260`
- archive bytes: `186411`
- archive SHA-256:
  `2c6e5f8d71f687227a28a9a378dc5edfc3215b762015042203b6bf58bfee9378`
- runtime tree SHA-256:
  `53b0f14fb4d0b1dbfe0866808699681aeb27a8bf6e9d3cdef36fdab097aa531f`
- runtime content tree SHA-256:
  `f277a317663fef67665a72ebc800bed2cc04e7d960bc3726b7994909e10aeb82`
- promotion eligible: `false` until explicit lane promotion

Against HLM1 xmember score `0.20637498220368466`, the measured delta is
`-0.00000266343581251`, matching the 4-byte rate-only prediction.

Scorecard refresh:

- `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hlm2_codex/scorecard.json`
- Current exact-CUDA row: `PR106-R2-HDM4-HLM2-XMEMBER`
- Next byte target remains `inner_decoder_packed_brotli` at `169990` bytes.

## Post-review hardening

Fresh-eye review found and closed three signal-loss risks:

- HNeRV scorecard preflight now fails closed when any required HLM1/HLM2 exact
  eval artifact is missing instead of silently omitting it from
  `--required-eval`.
- PR106 sidecar preflight now includes an HLM2-specific runtime-consumption
  proof requiring `latent_section_codec=hlm2`, canonical decode match, and a
  runtime-visible valid mutation.
- HNeRV routing defaults are centralized in `tac.hnerv_frontier_defaults` and
  now point to the HLM2 scorecard/entropy-ranking artifacts.

The HLM2 review packet and autopilot row now also carry runtime-content custody:
`runtime_content_tree_sha256`, `inflate_script_sha256`,
`inflated_output_manifest_sha256`, and `inflated_output_aggregate_sha256`.
