# Rebuild instructions — `tac-levelset-inflate`

Deterministic reproducibility contract (CLAUDE.md): the golden-vector fixtures + manifests
are regenerable byte-for-byte from the frozen contest inputs; the Rust port is proven against
them; both the Python oracle and the Rust port agree on the same SHA-256 digests.

## 1. Regenerate the golden vectors (only when the ξ coder grammar changes)

```bash
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/generate_golden_vectors.py
```

- Source of truth: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (frozen contest
  `gt_poses`, READ-ONLY — also consumed by the SACRED #205 run).
- Pipeline (all deterministic): `gt_poses` → `warp_real_luma_frame0.xi_from_pose_calibration`
  → `xi_pose_coder.quantize_xi` → `xi_pose_coder.serialize_xi_payload(coder="delta_ar")`
  → per-channel `(stream, uint32 counts, seed, lo)` → `range_coder.decode_static_symbols`.
- Writes `golden_vectors/levelset_xi_{range_decode,column_delta}_v1.{json,_stream.bin,_frequencies.bin}`.
  The manifest `sha256` pins the decoded output as little-endian `<i8` C-order bytes.

The committed fixtures are the regeneration output; day-to-day build/test reads the committed
fixtures and never needs the 4.8 GB gt cache.

## 2. Build + prove parity (offline; $0 / CPU)

```bash
cd runtime-rs
cargo build -p tac-levelset-inflate --offline
cargo test  -p tac-levelset-inflate --offline      # 11 tests incl. 2 real-xi SHA parity gates
```

`range_decode_parity` / `xi_column_delta_parity` assert the Rust decode's SHA-256 equals the
Python-oracle digest. `every_golden_vector_has_paired_parity_test` refuses an un-tested vector.

## 3. Python side of the parity contract (both sides must agree)

```bash
.venv/bin/python runtime-rs/crates/tac-levelset-inflate/python_reference_equivalence_test.py
# -> [PASS] levelset_xi_range_decode_v1 / [PASS] levelset_xi_column_delta_v1
```

## 4. Wall-clock micro-bench (MEANS — decode wall-clock only)

```bash
cargo run --release -p tac-levelset-inflate --example bench_decode --offline
# range_decode ~0.026 ms/call ; full 6-column ξ decode ~0.16 ms one-time (Python oracle ~11.4 ms).
```

## 5. Negative control (prove the gate is non-vacuous)

Flip one bit of `golden_vectors/levelset_xi_range_decode_v1_stream.bin` and rerun the test:
`range_decode_parity` must FAIL with a `ShaMismatch`. Restore the byte to pass again.

## Toolchain pinned at proof time

- rustc/cargo 1.96.0 (offline; deps already vendored in the cargo cache).
- numpy Accelerate BLAS (macOS) for the oracle regeneration — but the Rust port's bit-exactness
  is BLAS-INDEPENDENT (pure integer), so it reproduces the digests on any host/BLAS.
