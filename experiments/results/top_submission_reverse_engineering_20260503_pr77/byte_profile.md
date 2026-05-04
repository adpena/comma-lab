# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip` | 276551 | 0.184143959345 | `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af` | 1 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `p` | stored | 276451 | 276451 | json_like | 0 | 0.184129310448 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_fixed_slices | `masks.mkv` | 219472 | brotli_av1_obu | brotli_av1_obu_mask_stream | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_fixed_slices | `renderer.bin` | 55756 | brotli_qzs3 | brotli_qzs3_renderer | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_fixed_slices | `seg_tile_actions.bin` | 325 | brotli_seg_tile_actions_v1 | binary_payload | 0 |
| `archive.zip` | `p` | public_pr75_qzs3_qp1_segactions_fixed_slices | `optimized_poses.qp1` | 898 | public_qp1_brotli | brotli_qp1_pose | 0 |

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| medium | payload_segment | `masks.mkv` | 219472 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `optimized_poses.qp1` | 898 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `renderer.bin` | 55756 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `seg_tile_actions.bin` | 325 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| low | zip_member | `p` | 276451 | 0 | False | generic nested probes do not beat current byte count |
