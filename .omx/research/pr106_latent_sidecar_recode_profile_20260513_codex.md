# PR106 latent sidecar recode profile — 2026-05-13

## Summary

I canonicalized the PR106 latent sidecar byte grammar into
`tac.packet_compiler.pr106_sidecar_packet` and added
`tools/profile_pr106_latent_sidecar_recode.py`, a planning-only profiler that
decodes the real `(dim, delta_q)` sidecar arrays and proves lossless
equivalence for alternative byte grammars.

This is **not** a score claim. The emitted profile is a byte-planning artifact:

- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- no candidate archive emitted
- no new exact eval dispatched

## Source artifact

- source sidecar:
  `experiments/results/pr106_latent_score_table_materialized_20260513_codex_clean/sidecar.bin`
- sidecar bytes: `575`
- sidecar SHA-256:
  `acf58d03f708b0d297f9d32c2f47cefda3526f6665e19e9d0dee8073874c9d27`
- profile JSON:
  `experiments/results/pr106_latent_sidecar_recode_profile_20260513_codex/profile.json`
- profile JSON SHA-256:
  `ae92fbc0be3082ba94ae820a8033ebad4e8e5488eb17676a49ba5053f5c4f8a4`
- profile Markdown:
  `experiments/results/pr106_latent_sidecar_recode_profile_20260513_codex/profile.md`
- profile Markdown SHA-256:
  `fd9a2a5bddb13852288aff9fd08bfc315502457c9cac2e058cb794c8d1df01da`

Semantic sidecar arrays:

- pairs: `600`
- corrected pairs: `600`
- no-op pairs: `0`
- delta vocabulary: `[-2, -1, 1, 2]`
- dim SHA-256:
  `a55bd9bdecb4a462e166cf0f8363bcb5578e8be61339a97639859173aa5098c3`
- delta SHA-256:
  `616ee8fbee1e4770ee5574372efc8637fbf8b6f4c450c3c4c9cd47ee20ad6a53`

## Candidate byte frontier

| candidate | charged bytes | delta vs current | rate score delta if consumed | runtime decoder |
|---|---:|---:|---:|---|
| `pr101_ranked_no_op_sidecar_format_0x02` | 533 | -42 | -0.0000279661 | yes |
| `vocab_bitpack_dim_delta_raw` | 539 | -36 | -0.0000239709 | no |
| `vocab_bitpack_dim_delta_brotli_q11` | 547 | -28 | -0.0000186441 | no |
| `current_pr100_dim_delta_brotli_q11` | 575 | 0 | 0 | yes |
| `split_dim_stream_delta_stream_brotli_q11` | 598 | +23 | +0.0000153148 | no |
| `sparse_indexed_nonzero_brotli_q11` | 1357 | +782 | +0.000520702 | no |

All applicable rows prove lossless semantic equivalence to the source
`(dim, delta_q)` arrays.

## Adversarial interpretation

The earlier partner finding that PR101 grammar saves `42` bytes on PR106-R2 is
validated on the current clean materialized sidecar. The independent
vocab-bitpack design is smaller than the current PR100-style Brotli sidecar but
still loses to the already implemented PR101 ranked/no-op grammar. That makes a
new sidecar grammar a low-EV branch unless it beats the 533-byte consumed
frontier.

The highest-EV score-lowering path is therefore not another arbitrary sidecar
payload recode. The next action should be one of:

1. apply the already consumed PR101 grammar to the current clean sidecar
   materialization and keep the packet byte-closed;
2. improve the semantic `(dim, delta_q)` search itself, then re-run this
   profiler to ensure the selected grammar is still byte-optimal;
3. move byte work to the larger decoder/latent payload sections or to a
   full PacketIR compiler pass that changes a runtime-consumed section.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_sidecar_recode.py`
  passed: 6 tests.
- `.venv/bin/python -m pytest -q src/tac/tests/test_pr106_latent_sidecar_recode.py src/tac/tests/test_pr106_latent_sidecar.py src/tac/tests/test_prove_pr106_packetir_identity_tool.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py`
  passed: 40 tests.
- `.venv/bin/python -m ruff check src/tac/packet_compiler/pr106_sidecar_packet.py src/tac/packet_compiler/__init__.py experiments/build_pr106_latent_sidecar.py tools/profile_pr106_latent_sidecar_recode.py src/tac/tests/test_pr106_latent_sidecar_recode.py`
  passed.

## Current status

This closes a byte-planning question and validates the best consumed sidecar
grammar, but it does not lower the exact contest score by itself. A future
score claim requires a runtime-consuming archive, no-op/runtime-consumption
proof, dispatch claim, exact contest eval artifact, and axis-labelled
component recomputation.
