# HNeRV HDC2 Context Model Compression Probe - 2026-05-06

## Scope

Read-only byte accounting against the existing HDC2 stream work product:

`experiments/results/hnerv_entropy_packet_discovery_20260506_codex/hdc2_stream_work_product/candidate_hdc2_global_prev_symbol_stream.bin`

No archive was built, no dispatch was attempted, and no score is claimed.

## Section Baseline

- current frontier section: `decoder_packed_brotli`
- current frontier section bytes: `170127`
- HDC2 stream bytes: `221381`
- HDC2 net delta now: `+51254`

## HDC2 Split

- record metadata bytes: `472`
- context model bytes: `40361`
- range payload bytes: `180429`
- scale bytes: `112`
- context count: `247`

## Context Model Brotli Probe

Compressing only the context model bytes gives:

| Brotli quality | context model bytes | projected total bytes | delta vs current frontier section |
|---:|---:|---:|---:|
| 1 | 18712 | 199732 | +29605 |
| 3 | 18270 | 199290 | +29163 |
| 5 | 18036 | 199056 | +28929 |
| 7 | 17946 | 198966 | +28839 |
| 9 | 17963 | 198983 | +28856 |
| 11 | 16376 | 197396 | +27269 |

## Interpretation

Context-model compression is worthwhile but insufficient. Even an idealized
HDC3-style container that Brotli-compresses the current context table remains
byte-negative by about `+27269` bytes at Brotli quality 11. The payload coding
gap must also move materially before an archive build is rational.

Acceptance threshold for the next implementation:

- raw decoder equality remains closed,
- model table overhead decreases,
- range payload bytes also decrease,
- projected total must beat `170127` bytes before archive construction.

