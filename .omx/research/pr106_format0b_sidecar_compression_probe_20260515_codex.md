# PR106 Format 0x0B Sidecar Compression Probe

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- verdict: `small_exact_radix_runtime_format_candidate_found`

## Source

- archive: `experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip`
- archive bytes: `186341`
- archive sha256: `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
- member: `x`

## Sidecar Headroom

- current sidecar payload: `525` bytes
- current dim container: `375` bytes
- exact base-28 dim container: `361` bytes
- format0C candidate payload: `511` bytes
- runtime-supported format0C savings: `14` bytes
- rate-only score delta if components equal: `-0.000009322025`

## Semantic Stats

- dim entropy: `4.762747` bits/symbol
- delta entropy: `1.773593` bits/symbol
- ideal dim entropy bytes, no model cost: `358`
- ideal delta entropy bytes, no model cost: `134`

## Generic Compressor Best Rows

| source | codec | encoded bytes | delta vs source |
|---|---|---:|---:|
| `current_375b_dim_container` | `brotli` | 376 | 1 |
| `current_525b_sidecar_payload` | `brotli` | 529 | 4 |
| `exact_radix_511b_candidate_payload` | `brotli` | 515 | 4 |

## Blockers

- `planning_probe_only_no_archive_emitted`
- `requires_full_frame_parity_before_exact_eval`
- `requires_lane_dispatch_claim_before_exact_cuda`
- `generic_compressor_rows_do_not_have_runtime_decoder`

## Recommendation

Emit a byte-closed format0C archive and run same-runtime parity before any exact-eval dispatch; prioritize decoder/latent stream transforms because the sidecar has only 14 byte-closed bytes of headroom.
