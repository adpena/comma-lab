# PR85 Archive Bit-Budget Profile

- evidence_grade: `planning_only_static_archive_accounting`
- planning_only: `true`
- score_claim: `false`
- dispatch_performed: `false`
- archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- archive_bytes: `236328`

## Top Byte Surfaces

| rank | segment | bytes | share | protected | deletion-negative | basis |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | `mask` | 159011 | 0.672840 | `false` | `false` | QMA9 mask payload is the largest charged segment. |
| 2 | `model` | 57074 | 0.241503 | `false` | `false` | Joint frame model bytes are the second largest charged segment. |
| 3 | `randmulti` | 16101 | 0.068130 | `true` | `true` | Whole randmulti deletion is negative; target group-level recoding or water-fill, not deletion. |
| 4 | `pose` | 1487 | 0.006292 | `false` | `false` | Small charged stream; only deterministic parity-preserving byte work is justified. |
| 5 | `post` | 1400 | 0.005924 | `true` | `true` | Deletion/preflight negatives protect this stream; only parity-preserving recodes are planning-safe. |
| 6 | `region` | 273 | 0.001155 | `false` | `false` | Small charged stream; only deterministic parity-preserving byte work is justified. |
| 7 | `shift` | 226 | 0.000956 | `true` | `true` | Deletion/preflight negatives protect this stream; only parity-preserving recodes are planning-safe. |
| 8 | `bias` | 223 | 0.000944 | `false` | `false` | Small charged stream; only deterministic parity-preserving byte work is justified. |

## Segment Budget

| rank | segment | bytes | bits | cumulative share | formula-only rate score |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `mask` | 159011 | 1272088 | 0.672840 | 0.105878897995 |
| 2 | `model` | 57074 | 456592 | 0.914344 | 0.038003233890 |
| 3 | `randmulti` | 16101 | 128808 | 0.982474 | 0.010720995004 |
| 4 | `pose` | 1487 | 11896 | 0.988766 | 0.000990132263 |
| 5 | `post` | 1400 | 11200 | 0.994690 | 0.000932202534 |
| 6 | `region` | 273 | 2184 | 0.995845 | 0.000181779494 |
| 7 | `shift` | 226 | 1808 | 0.996801 | 0.000150484123 |
| 8 | `bias` | 223 | 1784 | 0.997745 | 0.000148486547 |
| 9 | `frac3` | 154 | 1232 | 0.998396 | 0.000102542279 |
| 10 | `frac2` | 149 | 1192 | 0.999027 | 0.000099212984 |
| 11 | `frac` | 106 | 848 | 0.999475 | 0.000070581049 |
