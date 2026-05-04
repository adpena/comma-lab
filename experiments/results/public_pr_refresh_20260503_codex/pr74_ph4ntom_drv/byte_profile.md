# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/public_pr_refresh_20260503_codex/pr74_ph4ntom_drv/archive.zip` | 321311 | 0.213947806087 | `a73d011120b68eda636d063ccbee5277ccbaefd8c52c41a3a43250525ff6c924` | 3 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `model.pt.br` | deflated | 89012 | 88997 | unknown | 0 | 0.059366652542 |
| `archive.zip` | `mask.obu.br` | deflated | 218642 | 218607 | brotli_av1_obu_mask_stream | 0 | 0.145681948636 |
| `archive.zip` | `pose.npy.br` | stored | 13185 | 13185 | unknown | 0 | 0.008876565704 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| low | zip_member | `mask.obu.br` | 218642 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `model.pt.br` | 89012 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `pose.npy.br` | 13185 | 0 | False | generic nested probes do not beat current byte count |
