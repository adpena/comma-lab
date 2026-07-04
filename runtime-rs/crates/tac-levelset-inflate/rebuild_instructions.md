# Rebuild instructions — `tac-levelset-inflate`

Deterministic reproducibility contract (CLAUDE.md): the golden-vector fixtures + manifests
are regenerable byte-for-byte from the frozen contest inputs; the Rust port is proven against
them; both the Python oracle and the Rust port agree on the same SHA-256 digests.

## 1. Regenerate the golden vectors (only when a coder/rasterizer grammar changes)

```bash
# ξ pose-carrier decode vectors (from gt_n600):
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/generate_golden_vectors.py
# lane AA-SDF coverage vector (#283, from gt_n96):
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/generate_lane_golden_vectors.py
```

- Source of truth: `experiments/results/mlx_fleet_gt_cache/gt_n{600,96}.npz` (frozen contest
  `gt_poses` / SegNet argmax `lstars`, READ-ONLY — also consumed by the SACRED #205 run).
- ξ pipeline (deterministic): `gt_poses` → `warp_real_luma_frame0.xi_from_pose_calibration`
  → `xi_pose_coder.quantize_xi` → `serialize_xi_payload(coder="delta_ar")` → per-channel
  `(stream, counts, seed, lo)` → `range_coder.decode_static_symbols`.
- lane pipeline (#283, deterministic): `lstars[:32]` → `build_lane_band_pairs_from_lstars`
  → `serialize_lane_band` (LBND1 blob) → the SHIPPED inflate `_lane_coverage` (extracted from
  `_INFLATE_PY`) → stacked `(32,384,512)` float32. The generator ALSO cross-checks the shipped
  oracle == the canonical `rasterize_lane_coverage_range_dependent` bit-for-bit.
- Writes `golden_vectors/levelset_lane_coverage_v1.json` (manifest: sha256 of the stacked
  coverage as `<f4` C-order) + `levelset_lane_coverage_v1_band.bin` (the REAL LBND1 blob).

The committed fixtures are the regeneration output; day-to-day build/test reads the committed
fixtures and never needs the multi-GB gt cache.

## 2. Build + prove parity (offline; $0 / CPU)

```bash
cd runtime-rs
cargo build -p tac-levelset-inflate --offline
cargo test  -p tac-levelset-inflate --offline      # 18 tests incl. 3 real-data SHA parity gates + lane negative control
```

`range_decode_parity` / `xi_column_delta_parity` / `lane_coverage_parity` assert the Rust
output's SHA-256 equals the Python-oracle digest. `lane_coverage_negative_control_bit_flip_breaks_parity`
proves the lane gate non-vacuous. `every_golden_vector_has_paired_parity_test` refuses an un-tested vector.

## 3. Python side of the parity contract (both sides must agree)

```bash
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/python_reference_equivalence_test.py
# -> [PASS] levelset_xi_range_decode_v1 / [PASS] levelset_xi_column_delta_v1 / [PASS] levelset_lane_coverage_v1
```

## 4. End-to-end swap-in parity + speedup (#283 — the ship proof)

```bash
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/lane_end_to_end_parity.py
# builds/loads the cdylib via ctypes; asserts numpy inflate coverage == Rust coverage == oracle SHA
# (BYTE-IDENTICAL, 32 real pairs @ 384x512); reports the ~2.18x raster speedup.
```

## 5. Wall-clock micro-bench (MEANS — decode wall-clock only)

```bash
cargo run --release -p tac-levelset-inflate --example bench_decode --offline
# range_decode ~0.026 ms/call ; 6-column ξ ~0.16 ms one-time ; lane_coverage ~0.61 ms/pair (~0.37 s @ n600).
```

## 6. Negative control (prove the gates are non-vacuous)

- ξ: flip one bit of `golden_vectors/levelset_xi_range_decode_v1_stream.bin`, rerun — `range_decode_parity`
  must FAIL with a `ShaMismatch`. Restore to pass.
- lane: `lane_coverage_negative_control_bit_flip_breaks_parity` does this in-test (flips a coeff byte of the
  LBND1 blob in memory, asserts the SHA breaks) so it runs every `cargo test`.

## Toolchain pinned at proof time

- rustc/cargo 1.96.0 (offline; deps already vendored in the cargo cache).
- numpy Accelerate BLAS (macOS) for the oracle regeneration — but the Rust port's bit-exactness
  is BLAS-INDEPENDENT (pure integer), so it reproduces the digests on any host/BLAS.
