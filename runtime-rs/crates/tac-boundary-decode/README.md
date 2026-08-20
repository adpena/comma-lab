# `tac-boundary-decode`

Native (Rust) DECODE-side port of the **settled** score-native carrier
primitives in `src/tac/boundary_math/`, per CLAUDE.md
"Native eval-time runtime discipline" + task #58.

The Python oracle stays canonical. This crate matches it **byte-for-byte** on
the committed golden vectors: the decoded-raw SHA-256 must equal the Python
oracle's. Promotion of a primitive = its parity test goes green
(`tests/golden_vector_parity.rs` → `conformance::assert_sha256_parity`).

## Primitives lowered (PROVEN-hot / SETTLED only — grammar-first)

| primitive | Rust fn | Python oracle | parity vector | parity |
|-----------|---------|---------------|---------------|--------|
| dense-raster LZMA decode | `contour::decode_partition_raw` / `decode_partition_hw` | `dense_raster_lzma_baseline.decode_partition` (RAW-LZMA2) | `contour_decode_{full,small}_v1` | bit-identical ✅ |
| d_seg popcount | `dseg::flip_count` / `dseg::d_seg` | `bitmask_dseg.flip_count` / `d_seg_reference` | `dseg_popcount_v1` | bit-identical ✅ |
| connected components | `components::connected_components` | `partition.connected_components` (4-conn) | `connected_components_v1` | bit-identical ✅ |

The dense-label decode + reshape IS the inflate-time "region rasterize / fill": the
decoded label map is the rasterized partition (each pixel's class label),
interiors reconstructed from constant-label runs.

## Wall-clock speedup (advisory — NOT a contest-score claim)

Apples-to-apples on the same fixture bytes/shapes (M5 Max, release):

| primitive | Python (ms/call) | Rust (ms/call) | speedup |
|-----------|------------------|----------------|---------|
| decode_partition 384x512 | 0.064 | 0.034 | 1.88× |
| flip_count 64x96 | 0.0013 | 0.0008 | 1.71× |
| connected_components 32x48 | 0.152 | 0.0094 | **16.2×** |

The dense-label decode + d_seg wins are bounded (both sides call the same liblzma C
core / numpy already vectorizes the XOR). The connected-components win is large
because the dedicated raster flood-fill replaces `scipy.ndimage.label`'s heavy
generic-label machinery.

## NOT lowered (out of scope — NO FAKE)

- **generator-blob int8+brotli decode**: the Python side has only a byte-SIZE
  ESTIMATOR (`_quantize_blob_bytes`), no inverse decode grammar. Lowering it
  would require inventing a contract with no oracle to match. Deferred until the
  grammar is finalized.
- **luma carrier decode (task #57)**: grammar not yet final. Deferred.

## Payload cleanliness

Per CLAUDE.md "Native eval-time runtime discipline": the binary is FIXED,
rate-free decode code. ZERO learned/video-derived constants are embedded — only
structural codec config (LZMA preset/lc/lp/pb) + 4-connectivity offsets. The
partition labels the SegNet derives masks from are carried in `archive.zip`
(the LZMA payload bytes), byte-charged in the rate term. See
`binary_source_audit.md` + `embedded_constants_audit.txt` +
`archive_payload_manifest.json`.

## Rebuild / prove

See `rebuild_instructions.md`. TL;DR:

```bash
.venv/bin/python crates/tac-boundary-decode/golden_vectors/generate_golden_vectors.py
cd runtime-rs && cargo test -p tac-boundary-decode      # 15 green (10 unit + 5 parity)
.venv/bin/python crates/tac-boundary-decode/python_reference_equivalence_test.py
```

`publish = false` (LOCAL-only; operator-gated per CLAUDE.md "Public Disclosure
Hygiene", mirroring the sibling `tac-packet-compiler` crate).
