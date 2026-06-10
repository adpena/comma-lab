# Binary / source audit — `tac-boundary-decode`

**Verdict: CLEAN.** The crate source is PURE DECODE LOGIC. No learned or
video-derived constant is embedded in the binary or the source. This satisfies
the CLAUDE.md "Native eval-time runtime discipline" payload-cleanliness contract:
the decoder is FIXED, rate-free code; the partition labels / contours the SegNet
derives masks from are carried in `archive.zip` (the LZMA payload bytes), NOT in
the binary.

## What each module does (decode logic only)

| module | logic | inputs (caller-supplied) | embedded constants |
|--------|-------|--------------------------|--------------------|
| `contour.rs` | RAW-LZMA2 decompress → raw `uint8` label bytes (mirror of `contour_codec.decode_partition`) | LZMA payload bytes (from archive), `(H, W)` shape | LZMA codec config only: preset level `9`, `lc=0`, `lp=0`, `pb=0` (= `contour_codec._LZMA_FILTERS`) |
| `dseg.rs` | popcount(XOR) of two label arrays → `flip_count` / `d_seg` rate | two `uint8` label arrays | none (loop counters / `n_pixels` arithmetic) |
| `components.rs` | raster-scan flood-fill → 4-connected `region_of` id map (mirror of `partition.connected_components`) | `uint8` argmax array, `(H, W)`, `n_classes` | 4-connectivity neighbour offsets `[(-1,0),(1,0),(0,-1),(0,1)]` |
| `conformance.rs` | golden-vector loader + SHA-256 parity helper | manifest JSON + produced bytes | none |
| `lib.rs` | error type + module glue | — | none |

## Source-level constant scan (every `const`/`static`)

```
components.rs:  const NEIGH: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];   # 4-connectivity (structural)
contour.rs:     const LZMA_PRESET_LEVEL_9: u32 = 9;   # LZMA codec config (structural)
contour.rs:     const LZMA_LC: u32 = 0;               # literal-context-bits (codec config)
contour.rs:     const LZMA_LP: u32 = 0;               # literal-position-bits (codec config)
contour.rs:     const LZMA_PB: u32 = 0;               # position-bits (codec config)
```

All five are **structural codec / connectivity configuration**, identical to the
Python oracle's `contour_codec._LZMA_FILTERS` and `partition._CONN4`. None encode
a video-derived label, a trained weight, a Fourier table, a per-pair mod code, or
any other answer-from-the-data.

## Binary-level scan

- `libtac_boundary_decode-*.rlib` = ~231 KB — consistent with pure code; no
  multi-KB float/data tables (a baked weight blob would balloon this).
- example binary `__DATA` segment = 16 KB (Rust/libstd runtime overhead, not a
  weight table); `__TEXT` (code) = 304 KB.
- Source has ZERO references to `.npz` / `.pt` / `weight` / `codebook` /
  `fourier` / `mod_vec` / `trained` / `generator_argmax` (grep clean).

## Why the generator-blob decode is NOT here (NO FAKE)

The lever-B score-native seg generator's int8+brotli "blob" (`tools/
lever_b_score_native_argmax_smoke.py::_quantize_blob_bytes`) is a **size
estimator only** — it quantizes-and-brotli's to MEASURE bytes; there is NO
inverse Python decode function (`decode_quantized_blob → params` does not
exist). Lowering it into Rust would require INVENTING the grammar, which would
either embed the answer or fabricate a contract with no oracle to match
bit-for-bit. Per the SUPREME RULE (NO FAKE IMPLEMENTATIONS) it is deliberately
deferred until the grammar is finalized (sister of task #57 luma carrier).
