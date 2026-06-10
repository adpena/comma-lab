# Rust runtime: lowered the settled score-native carrier DECODE primitives (task #58)

- **UTC:** 2026-06-10T12:40:30Z
- **Task:** #58 — operator: *"full authority … lower any and all into rust"* (the
  settled score-native carrier decode primitives).
- **Governing rule:** CLAUDE.md "Native eval-time runtime discipline" +
  SUPREME RULE (NO FAKE IMPLEMENTATIONS). Python oracle is canonical; the Rust
  decoder is FIXED rate-free code; NO learned/video-derived constant embedded.
- **Crate:** `runtime-rs/crates/tac-boundary-decode` (NEW workspace member;
  reuses the `tac-packet-compiler` golden-vector parity pattern + the same
  `liblzma 0.4` C-library binding Python's stdlib `lzma` wraps).
- **Evidence grade:** native-parity-verified (bit-identical to the Python oracle
  on committed golden vectors). Wall-clock is `[advisory]`, NOT a contest-score
  claim. `score_claim=false`, `promotion_eligible=false`.

## Which primitives lowered (grammar-first: PROVEN-hot / SETTLED only)

| primitive | Rust fn | Python oracle | golden vector | parity |
|-----------|---------|---------------|---------------|--------|
| contour codec decode | `contour::decode_partition_raw` / `decode_partition_hw` | `contour_codec.decode_partition` (RAW-LZMA2 → label map) | `contour_decode_full_v1` (384×512) + `contour_decode_small_v1` (16×24) | **bit-identical ✅** |
| region rasterize / fill | (= contour decode + reshape; the decoded label map IS the rasterized partition, interiors from constant-label runs) | — | (covered by contour vectors) | **bit-identical ✅** |
| d_seg popcount | `dseg::flip_count` / `dseg::d_seg` | `bitmask_dseg.flip_count` / `d_seg_reference` (popcount XOR) | `dseg_popcount_v1` (64×96) | **bit-identical ✅** |
| connected components (4-conn region map) | `components::connected_components` | `partition.connected_components` | `connected_components_v1` (32×48) | **bit-identical ✅** |

Profiling FIRST (grammar-first discipline): of the candidate primitives, the
genuinely hot/settled inflate-side ones are contour decode + d_seg popcount; the
RAG `connected_components` (the slow 4.8 ms/call scipy path) is compress-time but
worth lowering as the biggest speed win. The mask-bitmask popcount (#3) and
generator decode (#4) were assessed per scope — see "Deferred" below for #4.

## Parity-sha verdict (bit-identical: Rust == Python)

The parity gate (`tests/golden_vector_parity.rs` → `conformance::assert_sha256_parity`)
asserts the SHA-256 of the Rust **decoded raw output** equals the Python oracle's,
byte-for-byte. All 5 parity tests GREEN (+ 10 unit tests = 15/15):

| golden vector | pinned decoded-raw SHA-256 (prefix) | Rust == Python |
|---------------|--------------------------------------|----------------|
| `contour_decode_full_v1`  | `0fb4d848bd334c3442fb3714…` | YES |
| `contour_decode_small_v1` | `56f2b7292f6336e8b986669a…` | YES |
| `dseg_popcount_v1`        | `00608c5f0d4ac065bbfa765a…` | YES |
| `connected_components_v1` | `ab8cd4b51cf39712395bf44a…` | YES |

**Negative control (NO-FAKE proof the gate verifies BEHAVIOR not constants):**
corrupting one hex char of `contour_decode_full_v1`'s pinned sha flips the parity
test to FAILED with a `ShaMismatch { produced != expected }` diagnostic; regenerating
the vector via the oracle restores green. The test would NOT pass if the Rust
decode were replaced by a constant-returning stub.

**Python-side cross-check** (`python_reference_equivalence_test.py`): re-derives
the same SHAs from the REAL `tac.boundary_math` oracle on the committed fixtures
and asserts manifest agreement — ALL PASS. Both sides of the contract agree.

The Rust port discovered + reproduced an exact ordering invariant:
`scipy.ndimage.label` numbers 4-connected components in **first-pixel raster-scan
order**, identical to a raster-scan flood-fill (verified empirically before
implementing), so the Rust `region_of` int32 raster is byte-identical.

## Wall-clock speedup (advisory, M5 Max, release, same fixture bytes/shapes)

| primitive | Python ms/call | Rust ms/call | speedup |
|-----------|----------------|--------------|---------|
| decode_partition 384×512 | 0.064 | 0.034 | 1.88× |
| flip_count 64×96 | 0.0013 | 0.0008 | 1.71× |
| connected_components 32×48 | 0.152 | 0.0094 | **16.2×** |

Contour decode + d_seg wins are bounded (both call the same liblzma C core /
numpy already vectorizes the XOR-compare; the Rust win is removed Python/numpy
dispatch + reshape overhead). The connected-components win is large because the
dedicated raster flood-fill replaces scipy's heavy generic-label machinery.

## Payload-cleanliness verdict: CLEAN (zero embedded learned constants)

The 5-file audit bundle lives at `runtime-rs/crates/tac-boundary-decode/`:
`binary_source_audit.md`, `embedded_constants_audit.txt`,
`archive_payload_manifest.json`, `rebuild_instructions.md`,
`python_reference_equivalence_test.py`.

- Every `const`/`static` in the crate is **structural codec/connectivity config**:
  LZMA preset `9|EXTREME` + `lc=0/lp=0/pb=0` (= `contour_codec._LZMA_FILTERS`) and
  the 4-connectivity neighbour offsets (= `partition._CONN4`). ZERO learned
  weights, ZERO video-derived contours/labels, ZERO codebooks.
- rlib = ~231 KB (pure code, no data tables); example bin `__DATA` = 16 KB
  (runtime overhead, not a weight blob).
- The partition labels the SegNet derives masks from are carried in `archive.zip`
  (the RAW-LZMA2 payload bytes, rate-charged), NOT in the binary. The decoder
  READS the archive; it does not embed the answer.

## Deferred (out of scope — NO FAKE, no oracle to match bit-for-bit)

- **generator-blob int8+brotli decode (#4 in scope):** the Python side
  (`tools/lever_b_score_native_argmax_smoke.py::_quantize_blob_bytes`) is a byte
  **SIZE ESTIMATOR ONLY** — it quantizes+brotli's to MEASURE bytes; there is NO
  inverse decode function (`decode_quantized_blob → params` does not exist; the
  generator's real load path is `load_generator_npz`, a numpy `.npz` container,
  not a settled byte grammar). Lowering it would require INVENTING the grammar
  with no oracle to match bit-for-bit → forbidden. Defer until the score-native
  generator archive grammar is finalized.
- **luma carrier decode (task #57):** grammar not yet final (per scope). Defer.

## Handoff

1. When the score-native **generator archive grammar** settles (the int8+brotli
   blob gets a real `decode_quantized_blob → params` Python oracle), lower it
   into this crate as `generator::decode_blob` with a `generator_blob_v1` golden
   vector + the int8-dequant fixed code (the weights stay in `archive.zip`).
2. When **task #57** (luma carrier) settles, lower its decode into this crate
   the same way (golden vector + parity test).
3. Per-function promotion contract: a primitive is promoted iff its
   `assert_sha256_parity` test is green on the committed golden vector.

## 6-hook wire-in (CLAUDE.md "Subagent coherence-by-default")

This is a native-runtime speed layer (a reference-oracle-gated port), not a
score-moving substrate. Hook status:
- #1 sensitivity-map — N/A (no per-axis byte savings; decode parity only).
- #2 Pareto constraint — N/A (no archive bytes changed; decode is rate-free).
- #3 bit-allocator — N/A.
- #4 cathedral autopilot dispatch — N/A (not archive-deployable on its own; it
  is the inflate-side decode body for the boundary_math carrier once that
  carrier ships).
- #5 continual-learning posterior — N/A (no empirical score anchor; advisory
  wall-clock only).
- #6 probe-disambiguator — N/A (single deterministic interpretation; the parity
  gate IS the arbiter).
`research_only=false` (the primitives are real, parity-proven, reusable), but
`score_claim=false` / `promotion_eligible=false` (no contest bytes moved).
