# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip` | 276342 | 0.184004794824 | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | 1 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `p` | stored | 276242 | 276242 | public_pr75_p6_segactions_payload | 0 | 0.183990145927 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_p6_delta_varint | `masks.mkv` | 219472 | brotli_av1_obu | brotli_av1_obu_mask_stream | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_p6_delta_varint | `renderer.bin` | 55965 | brotli_qzs3 | brotli_qzs3_renderer | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_p6_delta_varint | `seg_tile_actions.bin` | 116 | seg_tile_actions_delta_varint_v1 | binary_payload | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_p6_delta_varint | `optimized_poses.qp1` | 677 | public_qp1_brotli | brotli_qp1_pose | 0 |

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| medium | payload_segment | `masks.mkv` | 219472 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `optimized_poses.qp1` | 677 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `renderer.bin` | 55965 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `seg_tile_actions.bin` | 116 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| low | zip_member | `p` | 276242 | 0 | False | generic nested probes do not beat current byte count |
