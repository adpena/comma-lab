# Lossless Transform Sample Benchmarks

Sample:
- source: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0.bin`
- prefix: first `16` segments in canonical order
- sample tokens: `2,459,664`
- sample bytes: `4,919,328`
- proxy backend: `lzma6`

Baseline:
- `position_major` raw stream
- encoded bytes: `1,766,724`
- compression ratio: `2.7844349202252303`

Measured reversible transforms:

| transform | encoded bytes | ratio | delta vs raw |
| --- | ---: | ---: | ---: |
| `frequency_remap` | `1,761,832` | `2.792166335950306` | `-4,892` bytes |
| `temporal_residual` | `2,267,988` | `2.1690273493510546` | `+501,264` bytes |
| `bitplane_split(low_bits=5)` | `2,572,818` | `1.9120388616684119` | `+806,094` bytes |
| `sustain_attack` | `2,108,800` | `2.332761760242792` | `+342,076` bytes |

Component notes:
- `frequency_remap` proxy = `lzma(remapped stream)` + raw inverse-map header (`2,052` bytes)
- `temporal_residual` proxy = `lzma(residual stream)` after fixing the `1025` sentinel collision in residual bodies
- `bitplane_split` proxy = `lzma` per packed high/low bitplane + `2` bytes of low-bit metadata
- `sustain_attack` proxy = `lzma(first values)` + `lzma(packed hold mask)` + `lzma(changed values)` + `16` bytes of fixed header

Observed outcome:
- `frequency_remap` is the only one of these four that helped on this real sample, and the gain is tiny.
- `temporal_residual`, `bitplane_split`, and `sustain_attack` all lost materially against the plain `position_major` stream under the same `lzma6` proxy.
- For this branch, these transforms are not competitive primary lanes versus the already-stronger `position_major + prev-symbol` and GPT arithmetic paths.
