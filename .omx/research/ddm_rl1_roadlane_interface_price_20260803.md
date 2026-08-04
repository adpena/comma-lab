# ddm_rl1 — Road↔Lane interface counted-correction price — 2026-08-03

| granularity (lossless serialized object) | real coder | measured n32 bytes | bytes / pair | extrapolated ×600 bytes | B / Road↔Lane flip | vs W = 1.273108 B/flip | projected byte margin vs 299,369 B |
|---|---:|---:|---:|---:|---:|---:|---:|
| full interface chain-code | zlib-9 | 20,986 | 655.81 | 393,488 | 1.6734 | **1.314× — DEAD** | −94,119 |
| full interface chain-code | lzma-9e | 22,148 | 692.12 | 415,275 | 1.7660 | **1.387× — DEAD** | −115,906 |
| full interface chain-code | brotli-q11 | 20,648 | 645.25 | 387,150 | 1.6464 | **1.293× — DEAD** | −87,781 |
| per-CLASS Lane-mask crop box | zlib-9 | 16,180 | 505.62 | 303,375 | 1.2901 | **1.013× — DEAD** | −4,006 |
| per-CLASS Lane-mask crop box | lzma-9e | 17,092 | 534.12 | 320,475 | 1.3629 | **1.071× — DEAD** | −21,106 |
| **per-CLASS Lane-mask crop box** | **brotli-q11** | **14,553** | **454.78** | **272,869** | **1.1604** | **0.911× — BEATS W (n32 projection)** | **+26,500** |
| per-COMPONENT Lane sparse indices | zlib-9 | 54,076 | 1,689.88 | 1,013,925 | 4.3119 | **3.387× — DEAD** | −714,556 |
| per-COMPONENT Lane sparse indices | lzma-9e | 37,972 | 1,186.62 | 711,975 | 3.0278 | **2.378× — DEAD** | −412,606 |
| per-COMPONENT Lane sparse indices | brotli-q11 | 49,735 | 1,554.22 | 932,531 | 3.9657 | **3.115× — DEAD** | −633,162 |

**Result:** surgical representation changes the dense-label verdict, but only one measured row clears
the price line: the per-class Lane crop under Brotli projects to **272,869 B**, **26,500 B (8.85%)
under** the `235,148 × W = 299,368.85 B` Road↔Lane budget. This is **19.8× smaller** than gt3's
measured counted dense-label stream (5,397,443 B). The win is coder-specific and thin: zlib on the
same exact arrays is 4,006 B over W. Therefore this is an n32 admission signal, not an n600 verdict.

**Selection mode:** `EVENLY_STRIDED_ROUNDED_LINSPACE_NOT_PREFIX`, 32 pair ids
`[0, 19, 39, 58, 77, 97, 116, 135, 155, 174, 193, 213, 232, 251, 271, 290, 309, 328, 348, 367, 386, 406, 425, 444, 464, 483, 502, 522, 541, 560, 580, 599]` from
`np.load('experiments/results/mlx_fleet_gt_cache/gt_n600.npz')['lstars']`; this is never a prefix.
Every ×600 value is `600 × measured n32 mean bytes/pair`; it is an extrapolation from n32 and does
not measure population tails or temporal coding.

**Exact objects priced with real coders only:** each pair is serialized and compressed independently
(so coder framing is counted). The interface object is the set of Lane pixels with a 4-neighbor Road
pixel, split into 8-connected components; a deterministic DFS spanning walk stores each start plus
packed 3-bit 8-neighbor chain directions and exactly reconstructs the interface-pixel set. The class
object stores one Lane bounding box plus the exact row-major packed Lane bitmap. The component object
uses `scipy.ndimage.label` (8-connectivity), storing each Lane component box plus exact little-endian
uint32 local linear pixel indices. Decode equality against the source Boolean object was asserted for
all 32 pairs before byte counts were accepted. Coders are Python `zlib` level 9, XZ/LZMA preset 9
extreme, and Brotli quality 11. No i.i.d. entropy estimate appears anywhere in the table.

**Scope honesty:** Road↔Lane is held at the supplied settled `235,148` flips = 46.23% of flips =
30.46% of gap, with `W = 1.2731082153320312 B/flip`. These are GT-geometry description prices only;
they do not prove a receiver can realize all 235,148 corrections without collateral. Pointer UNMOVED.
