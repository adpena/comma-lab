# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/public_pr_refresh_20260503_codex/pr76_qpose14_poseq6/archive.zip` | 288567 | 0.192144920526 | `8ad7435787a311cef82e4c919c4feae845a33dede43bae76116e88cf91e0c4da` | 1 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `p` | stored | 288467 | 288467 | json_like | 0 | 0.192130271629 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|
| `archive.zip` | `p` | public_pr63_qpose14_fixed_slices | `masks.mkv` | 219472 | brotli_av1_obu | brotli_av1_obu_mask_stream | 0 |
| `archive.zip` | `p` | public_pr63_qpose14_fixed_slices | `renderer.bin` | 66841 | brotli_torch_fp4 | json_like | 0 |
| `archive.zip` | `p` | public_pr63_qpose14_fixed_slices | `optimized_poses.bin` | 2154 | public_qpose14_uint16_brotli | binary_payload | 0 |

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| medium | payload_segment | `masks.mkv` | 219472 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `optimized_poses.bin` | 2154 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| medium | payload_segment | `renderer.bin` | 66841 | 0 | False | segment-level planning signal only; requires charged repack/decoder |
| low | zip_member | `p` | 288467 | 0 | False | generic nested probes do not beat current byte count |
