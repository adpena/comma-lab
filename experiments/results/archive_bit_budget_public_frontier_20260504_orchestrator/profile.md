# Archive Bit Budget Profile

This profile is byte/compressibility evidence only. It is not score evidence, not promotion evidence, and not a method retirement signal.

- schema: `archive_bit_budget_profile_v1`
- score claim: `False`
- promotion eligible: `False`
- score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA`

## Archives

| archive | bytes | rate contribution | sha256 | members |
|---|---:|---:|---|---:|
| `/Users/adpena/Projects/pact/experiments/results/public_pr85_intake_20260503_codex/archive.zip` | 236328 | 0.157361114673 | `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e` | 1 |
| `/Users/adpena/Projects/pact/experiments/results/public_pr86_intake_20260504_merged_refresh/archive.zip` | 207579 | 0.13821833563 | `e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef` | 5 |
| `/Users/adpena/Projects/pact/experiments/results/public_pr89_intake_20260504_codex/archive.zip` | 236676 | 0.157592833589 | `2f4d38f295792bebc3e722abaad7647ecdd647b8a5948285bbe0906a6883ee98` | 2 |

## ZIP Members

| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |
|---|---|---|---:|---:|---|---:|---:|
| `archive.zip` | `x` | stored | 236228 | 236228 | unknown | 0 | 0.157346465776 |
| `archive.zip` | `master.pt.gz` | stored | 31144 | 31144 | unknown | -11 | 0.020804097131 |
| `archive.zip` | `slave.pt.gz` | stored | 32287 | 32287 | unknown | 0 | 0.021563842197 |
| `archive.zip` | `hpac.pt.ppmd` | stored | 28243 | 28243 | unknown | 0 | 0.018872440308 |
| `archive.zip` | `tokens.bin` | stored | 113900 | 113900 | binary_payload | 0 | 0.07590525722 |
| `archive.zip` | `meta.pt` | stored | 1499 | 1499 | nested_zip_or_torch | -930 | 0.001058049877 |
| `archive.zip` | `x` | stored | 236228 | 236228 | unknown | 0 | 0.157346465776 |
| `archive.zip` | `fb` | deflated | 268 | 300 | unknown | -38 | 0.000231718916 |

## Payload Segments

| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |
|---|---|---|---|---:|---|---|---:|

## Candidate Self-Compression Opportunities

| priority | scope | name | bytes | best probe savings | directly deployable | reason |
|---|---|---|---:|---:|---|---|
| medium | zip_member | `meta.pt` | 1499 | 930 | False | candidate byte savings require a charged decoder contract |
| medium | zip_member | `fb` | 268 | 38 | False | candidate byte savings require a charged decoder contract |
| medium | zip_member | `master.pt.gz` | 31144 | 11 | False | candidate byte savings require a charged decoder contract |
| low | zip_member | `x` | 236228 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `hpac.pt.ppmd` | 28243 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `slave.pt.gz` | 32287 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `tokens.bin` | 113900 | 0 | False | generic nested probes do not beat current byte count |
| low | zip_member | `x` | 236228 | 0 | False | generic nested probes do not beat current byte count |
