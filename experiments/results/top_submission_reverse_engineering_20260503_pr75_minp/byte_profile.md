# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip` | 276481 | 0.184097349218 | `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746` | 1 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `p` | stored | 276381 | 276381 | json_like | 0 | 0.184082700321 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| low | zip_member | `p` | 276381 | 0 | False | generic nested probes do not beat current byte count |
