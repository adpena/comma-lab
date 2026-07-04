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
| `lane_coverage.rs` (#283) | LBND1 parse + AA-SDF range-dependent COVERAGE raster (`polyval` Horner + elementwise fp64) → `(H,W)` f32 (mirror of `_INFLATE_PY::_lane_coverage`) | LBND1 blob (per-pair lane coeffs from archive) + render `(rh,rw)` | wire magic `b"LBND1\x00"` + the AA-SDF/IPM STRUCTURAL scalars copied verbatim from the oracle (`1e-6` softness floor, `1e-3` forward-denom floor, `0.5` half-width floor / clip bias, `1.0` clip hi / row+range offsets, `5.0` far-range offset, `2.0` cx=W/2) |
| `ffi.rs` (#283) | C-ABI (`cdylib`) wrapper over `lane_coverage` for the numpy-inflate `ctypes` swap-in (`INFLATE_RUST_LANE=1`) | LBND1 blob, `rh`, `rw`, caller `f32` out buffer | error sentinels `-1..-4` only |
| `conformance.rs` | golden-vector loader + SHA-256 parity helper | manifest JSON + produced bytes | none |
| `lib.rs` | error type + module glue | — | none |

## Source-level constant scan

The `const` declarations are the arithmetic coder's 32-bit range-state boundaries
(`range_decode.rs`) and the lane wire-format magic + AA-SDF/IPM STRUCTURAL scalars
(`lane_coverage.rs`) — all copied byte-for-byte from the Python oracle, all FIXED codec /
rasterizer structure. There are ZERO magic tables, learned weights, per-frame lookups, ξ
values, homographies, palettes, lane coeffs, or video-derived arrays anywhere in the source.
The lane geometry (`cam_h/fx/fy/v_h/dash_forward_max_m/softness/cx`) and the per-pair lane
polynomial coeffs are RUNTIME INPUTS read from the LBND blob in `archive.zip` (COUNTED,
rule 118) — never embedded. See `embedded_constants_audit.txt` for the exhaustive enumeration.

## Parity proof (NO FAKE — sha256-PROVEN on REAL data, not asserted)

`tests/golden_vector_parity.rs` runs each Rust primitive on the committed REAL fixtures and
asserts the output's SHA-256 equals the Python-oracle digest pinned in the manifest:

- **ξ decode** (`range_decode_parity` / `xi_column_delta_parity`) on a REAL #257 store-nothing
  ξ payload derived from the frozen contest `gt_poses`. Negative control: a 1-bit stream flip
  fails with a `ShaMismatch` (verified 2026-07-03).
- **lane coverage** (`lane_coverage_parity`, #283) on REAL per-pair lane coeffs fitted from the
  frozen CPU-torch SegNet argmax (`gt_n96.npz['lstars'][:32]` → `build_lane_band_pairs_from_lstars`
  → `serialize_lane_band` LBND1). 32 pairs (>=24), 160 lines (158 dash → exercises the
  range-dependent dash gate), 39,846 non-zero coverage px. Negative control
  (`lane_coverage_negative_control_bit_flip_breaks_parity`): the test FIRST asserts the pristine
  blob's Rust coverage matches the oracle digest, THEN flips one coefficient byte and asserts the
  SHA no longer matches — so the gate is proven non-vacuous in both directions (verified 2026-07-04).

The Python side re-derives the same digests from the REAL oracle
(`python_reference_equivalence_test.py`, ALL PASS incl. lane), and the end-to-end swap-in harness
(`lane_end_to_end_parity.py`) proves the numpy inflate's decoded coverage is BYTE-IDENTICAL with
the Rust rasterizer swapped in via `ctypes` (numpy == rust == oracle SHA), at 384×512 over 32 real
pairs — so shipping the Rust path leaves the archive-scored bytes UNCHANGED.

## Portability — why THIS piece ships where fp32 (#281) and the neural forward (#282) cannot

The lane rasterizer is bit-exact AND **host-portable** because every op is a correctly-rounded
IEEE-754 fp64 basic operation in a FIXED order:

- `np.polyval` = Horner `y = y*x + c` — a per-element sequential mul-then-add with **no FMA**
  (Rust does not auto-contract `a*b+c`; we never call `mul_add`), so there is no cross-element
  reduction whose accumulation order a BLAS could permute. This is the exact structural difference
  from the neural forward, whose `numpy @` dispatches to Apple Accelerate `dgemm` (blocked+FMA
  reduction order — MEASURED 4497/6144 divergent, #282). No matmul here ⇒ no reduction-order
  ambiguity.
- elementwise `/  *  -  abs  max  min  clip  mod` + comparisons — all correctly rounded and
  transcendental-FREE (no `tanh`, whose numpy-vs-libm 1-ULP gap blocked #282).
- the `f64 → f32` cast is round-to-nearest-ties-to-even (IEEE default; Rust `as f32` == numpy
  `astype('<f4')`). Unlike the fp32 raster (#281), the fp64 math means no accumulated fp32 error
  can reach the Δ22-uint8 argmax-boundary flips that made fp32 non-portable.

Correctly-rounded IEEE-754 basic ops reproduce byte-for-byte on any conformant host — x86-64
(SSE2 scalar, true 64-bit; not x87 80-bit) and ARM64 — with FMA contraction off. The SHA parity
gate is itself cross-toolchain evidence: **two independent fp64 implementations** (CPython/numpy's
C ops and Rust/LLVM's) produce byte-identical `f64→f32` output on the same host, which is only
possible if every op is correctly-rounded IEEE-754 (no FMA, no libm approximation). A full second
host is the remaining confirmation, but the op-set argument + the cross-toolchain agreement make
this the genuinely-shippable Rust inflate piece.

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
- The **lane AA-SDF render-band rasterizer** (`_lane_coverage`) — **DONE (task #283, this crate,
  `lane_coverage.rs` + `ffi.rs`)**. MEASURED matmul- and transcendental-FREE (only `np.polyval`
  Horner + elementwise `/ * - abs max min clip mod` + comparisons) → bit-exact AND portable (see
  Portability above), proven byte-for-byte on REAL lane coeffs + swapped into the numpy inflate via
  `ctypes` with byte-identical output. This is the FIRST genuinely SHIP-ABLE Rust inflate raster.

## MEANS (pointer 0.19110 UNMOVED)

This crate is decode-time wall-clock infrastructure. It moves no score.

- **ξ decode**: ONE-TIME per-inflate (~3.6k symbols / 600 pairs; Python ~11.4 ms, Rust ~0.16 ms, ~73×).
- **lane coverage** (#283): the per-pixel raster hot path (once/pair, shared across the pair's 2
  frames). MEASURED end-to-end (`lane_end_to_end_parity.py`, 32 real pairs @ 384×512): numpy 1.55
  ms/pair → Rust 0.71 ms/pair = **2.18× on the raster** (release `cargo bench` 0.61 ms/pair). At
  n600 that is ~0.93 s (numpy) → ~0.43 s (Rust) of the ~16-min contest 4-core inflate — a small but
  real decode-wall-clock win. The value is the bit-exact + PORTABLE Rust ship candidate, not the ms.

**Ship-readiness (#283):** bit-exact + portable ⇒ SHIP-ABLE. Remaining ship-integration (follow-on):
(1) wire the opt-in `INFLATE_RUST_LANE=1` fast path into the shipped inflate (`_lane_coverage` calls
the `ctypes` FFI when set; default OFF = numpy, byte-identical — #205's byte-close is untouched);
(2) the LBND2 (Wave-F RD) parse mirror (the coverage raster is IDENTICAL for LBND1/LBND2 since both
dequantize to the same fp64 `LaneLine`s — only the parse differs; the Python side can parse LBND2
and feed a lines-based FFI, reusing this exact raster); (3) deterministic reproducible build +
runtime-tree custody + `inflate.sh` closure per CLAUDE.md "Native eval-time runtime discipline".
