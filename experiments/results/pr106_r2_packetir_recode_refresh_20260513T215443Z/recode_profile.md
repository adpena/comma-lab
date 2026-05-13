# PR106 Latent Sidecar Recode Profile

- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- source_mode: `sidecar_archive`
- source_path: `submissions/pr106_latent_sidecar_r2/archive.zip`
- current_charged_sidecar_bytes: `575`
- n_pairs: `600`
- delta_q_unique: `[-2, -1, 1, 2]`

## Candidates

| candidate | charged bytes | delta bytes | rate delta if consumed | runtime decoder | equivalence |
|---|---:|---:|---:|---|---|
| `pr101_ranked_no_op_sidecar_format_0x02` | 533 | -42 | -2.7966076031131196e-05 | `true` | `true` |
| `vocab_bitpack_dim_delta_raw` | 539 | -36 | -2.397092231239817e-05 | `false` | `true` |
| `vocab_bitpack_dim_delta_brotli_q11` | 547 | -28 | -1.8644050687420796e-05 | `false` | `true` |
| `current_pr100_dim_delta_brotli_q11` | 575 | 0 | 0.0 | `true` | `true` |
| `split_dim_stream_delta_stream_brotli_q11` | 598 | 23 | 1.5314755921809942e-05 | `false` | `true` |
| `sparse_indexed_nonzero_brotli_q11` | 1357 | 782 | 0.000520701701341538 | `false` | `true` |

## Adversarial Claim Check

- verdict: `planning_only_no_score_claim`

Negative or positive byte deltas here are sidecar payload-rate signals only. A score claim requires a runtime that consumes the candidate grammar, a byte-closed archive, no-op proof, and exact contest eval on the emitted packet.
