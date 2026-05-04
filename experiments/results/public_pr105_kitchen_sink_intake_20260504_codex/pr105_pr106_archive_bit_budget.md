# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/public_pr105_kitchen_sink_intake_20260504_codex/archive.zip` | 177857 | 0.118427675825 | `597ba0732810eba08cdae619b679d211d398bc0249b8831898f7096d5beece1d` | 1 |
| `/Users/adpena/Projects/pact/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip` | 186239 | 0.124008905571 | `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58` | 1 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `0.bin` | stored | 177749 | 177749 | binary_payload | 0 | 0.118413026928 |
| `archive.zip` | `0.bin` | stored | 186131 | 186131 | binary_payload | 0 | 0.123994256674 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| low | zip_member | `0.bin` | 177749 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `0.bin` | 186131 | 0 | False | generic nested probes do not beat current byte count |
