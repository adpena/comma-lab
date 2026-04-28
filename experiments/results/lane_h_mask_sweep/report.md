# Lane H — Mask Encoder Sweep

- Source: precomputed_local/masks.pt (1200 frames @ 384x512)
- Rate denominator: 37,545,489 bytes
- Rate score = 25 * bytes / denom

| Codec | Mode | CRF | Size (B) | Rate | Disagreement | Enc (s) |
|-------|------|-----|---------:|-----:|-------------:|--------:|
| av1_monochrome | full | 63 | 108,758 | 0.0724 | 0.0074 | 35.5 |
| av1_monochrome | half | 50 | 234,351 | 0.1560 | 0.0057 | 15.5 |
| av1_monochrome | full | 56 | 287,001 | 0.1911 | 0.0034 | 31.5 |
| av1_monochrome | full | 50 | 421,054 | 0.2804 | 0.0022 | 29.2 |
| entropy_lossless_lzma | half | — | 582,559 | 0.3879 | 0.0043 | 3.7 |
| entropy_lossless_brotli11 | half | — | 583,465 | 0.3885 | 0.0043 | 61.8 |
| entropy_lossless_brotli11 | full | — | 984,525 | 0.6556 | 0.0000 | 98.7 |
| entropy_lossless_lzma | full | — | 990,483 | 0.6595 | 0.0000 | 6.4 |
