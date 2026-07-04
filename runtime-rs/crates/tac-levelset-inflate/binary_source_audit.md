# Binary / source audit — `tac-levelset-inflate`

**Verdict: CLEAN.** The crate source is PURE DECODE LOGIC. No learned or video-derived
constant is embedded in the binary or the source. This satisfies the CLAUDE.md "Native
eval-time runtime discipline" payload-cleanliness contract: the decoder is FIXED, rate-free
code; the ξ pose-carrier payload the SegNet/PoseNet-scored frames are warped from is carried
in `archive.zip` (the range-coded ξ stream bytes), NOT in the binary (rule 118).

## What each module does (decode logic only)

| module | logic | inputs (caller-supplied) | embedded constants |
|--------|-------|--------------------------|--------------------|
| `range_decode.rs` | CACM static-frequency arithmetic range DECODE → symbol indices (mirror of `tac.lossless.range_coder.decode_static_symbols`) | range-coded stream bytes (from archive), uint32 frequency table, symbol `count` | range-coder state-bit literals only: `STATE_BITS=32`, `FULL_RANGE/HALF/QUARTER/THREE_QUARTERS` (= `range_coder.STATE_BITS/FULL_RANGE/…`) |
| `xi_column.rs` | arithmetic decode → `+lo` offset → prefix-sum → `seed` (mirror of `xi_pose_coder._channel_delta_decode`, sans brotli-of-counts) | stream, frequencies, `seed`, `lo`, `p_count` | none (integer loop) |
| `conformance.rs` | golden-vector loader + SHA-256 parity helper | manifest JSON + produced bytes | none |
| `lib.rs` | error type + module glue | — | none |

## Source-level constant scan

The only `const` declarations are in `range_decode.rs` and are the arithmetic coder's
32-bit range-state boundaries — structural codec parameters, byte-for-byte identical to the
Python oracle's module constants. There are ZERO magic tables, learned weights, per-frame
lookups, ξ values, homographies, palettes, or video-derived arrays anywhere in the source.
See `embedded_constants_audit.txt` for the exhaustive enumeration.

## Parity proof (NO FAKE — sha256-PROVEN on REAL data, not asserted)

`tests/golden_vector_parity.rs` decodes the committed golden-vector fixtures (a REAL #257
store-nothing ξ payload derived from the frozen contest `gt_poses`) and asserts the decoded
output's SHA-256 equals the Python-oracle digest pinned in the manifest. The gate is
non-vacuous: flipping a single bit of the input stream fails `range_decode_parity` with a
`ShaMismatch` (verified 2026-07-03). The Python side re-derives the same digests from the REAL
oracle (`python_reference_equivalence_test.py`, ALL PASS), so both sides agree byte-for-byte.

## Feasibility verdict — why ONLY the ξ decode is here (task #282, MEASURED 2026-07-03)

The question posed: can a Rust port of the level-set inflate FORWARD be BIT-EXACT to the
numpy-fp64 oracle (`levelset_rgb_forward_numpy`)? The measured answer:

- **The neural forward is NOT bit-exact-portable.** Its wall-clock hot path is the coord-INR
  (in_proj / 4×hidden / out_sdf / out_tex matmuls + the hosc `tanh(β·sin)` `_act`). Two hard,
  MEASURED blockers against the current oracle:
  - numpy fp64 `@` dispatches to **Apple Accelerate `dgemm`** (blocked + FMA + SIMD reduction
    order). A naive fma-off sequential Rust-equivalent matmul diverges on **4497/6144** elements
    at ~1e-14 for a representative `(64,96)@(96,96)`. A portable Rust kernel does not reproduce
    Accelerate's proprietary accumulation order → not bit-exact. (`np.einsum(optimize=False)`
    == `@` bit-for-bit — both go to BLAS — confirming the blocker is BLAS, not numpy glue.)
  - numpy fp64 `tanh` ≠ the scalar system `libm` `tanh`: **844/4096** elements differ at 1 ULP.
    A Rust `f64::tanh` (system libm) therefore misses the hosc `_act`. (numpy `sin`/`cos`/`exp`
    DO match the scalar libm bit-for-bit — those are not the blocker; `tanh` is.)
- **The pose-carrier warp is NOT bit-exact-portable either** — `np.linalg.inv(H)` (LAPACK) +
  `Hinv @ grid` (BLAS) block it, same root cause.
- **The ONE bit-exact-by-CONSTRUCTION piece is the #257 store-nothing ξ decode** (this crate):
  pure integer, zero FMA/rounding/reduction/transcendental dependence. `u64` holds every
  intermediate exactly (max `(2^32)·total ≈ 2^44.9` for the real payload). Bit-exact is
  provable, not hoped.

**Deferred, with the honest route:**
- The neural forward's ONLY viable fast Rust route is **fp32-TOLERANT, not bit-exact** — gated
  on task #281's fp32 score-preserving verdict (if fp32 preserves argmax `d_seg` / `d_pose`,
  a Rust fp32 forward is far easier and faster; it will NOT byte-match the fp64 oracle, so it
  cannot flip its parity test to `assert_sha256_parity` — it needs a score-equivalence gate).
- The **lane AA-SDF render-band rasterizer** (`_lane_coverage`) is MEASURED matmul- and
  transcendental-FREE (only `np.polyval` Horner + elementwise `/`, `max`, `abs`, `clip`, `mod`,
  comparisons) → it IS bit-exact-able AND is a real per-pixel render-res wall-clock contributor
  (rule-118 FREE generic rasterizer). It is the **highest-value NEXT bit-exact increment**.

## MEANS (pointer 0.19110 UNMOVED)

This crate is decode-time wall-clock infrastructure. The ξ decode is a ONE-TIME per-inflate
cost (~3.6k symbols over 600 pairs; Python ~11.4 ms, Rust ~0.16 ms, ~73×), negligible against
the ~16-min contest inflate. It moves no score; it establishes the bit-exact-provable Rust
inflate beachhead on the genuinely-portable piece and records the measured feasibility boundary.
