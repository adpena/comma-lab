# Rebuild instructions — `tac-boundary-decode`

Deterministic rebuild of the crate, the golden vectors, and the parity proof.
$0 / local / no network required for the build (deps vendored in
`runtime-rs/Cargo.lock`).

## 1. Regenerate the golden vectors from the Python ORACLE

The vectors are produced by the REAL `tac.boundary_math` functions, NOT
hand-authored. Run from the repo root:

```bash
.venv/bin/python runtime-rs/crates/tac-boundary-decode/golden_vectors/generate_golden_vectors.py
```

This writes 4 manifests + their input `.bin` fixtures under `golden_vectors/`:

- `contour_decode_full_v1`  — 384x512 partition (the contest seg target shape).
- `contour_decode_small_v1` — 16x24 partition (fast unit-scale parity).
- `dseg_popcount_v1`        — 64x96 candidate-vs-gt flip count + d_seg.
- `connected_components_v1` — 32x48 4-connectivity region map.

Each manifest pins the SHA-256 of the *decoded raw output* the Rust port must
reproduce; the sibling `.bin` files are the exact input bytes the oracle
consumed (so the Rust port never reimplements numpy's RNG).

## 2. Build the crate

```bash
cd runtime-rs
cargo build -p tac-boundary-decode            # debug
cargo build -p tac-boundary-decode --release  # optimized
```

## 3. Run the bit-identical parity gate (the proof)

```bash
cd runtime-rs
cargo test -p tac-boundary-decode
```

Expect: 10 unit tests + 5 parity tests (`contour_decode_full_parity`,
`contour_decode_small_parity`, `dseg_popcount_parity`,
`connected_components_parity`, `every_golden_vector_has_paired_parity_test`)
all green. Each parity test asserts the Rust decoded-raw SHA-256 equals the
Python-oracle SHA-256 byte-for-byte (`conformance::assert_sha256_parity`).

### Negative control (proves the gate catches a mismatch, not constants)

```bash
# Corrupt one hex char of a golden sha; the parity test MUST fail.
# (then re-run generate_golden_vectors.py to restore)
```

Verified 2026-06-10: a 1-char sha corruption flips
`contour_decode_full_parity` to FAILED with a `ShaMismatch { produced != expected }`
diagnostic, then `generate_golden_vectors.py` restores green. The test verifies
BEHAVIOR (the decode output), not a metadata constant.

## 4. Wall-clock micro-bench (advisory, not a score claim)

```bash
cd runtime-rs
cargo run -p tac-boundary-decode --example bench_decode --release
```

## Python equivalence cross-check

```bash
.venv/bin/python runtime-rs/crates/tac-boundary-decode/python_reference_equivalence_test.py
```

Re-derives the same decoded-raw SHA-256 from the Python oracle on the committed
fixtures and asserts it equals each manifest's pinned digest — the Python side
of the same parity contract the Rust side proves.

## Toolchain

- `cargo` / `rustc` 1.96.0 (rust-version floor 1.85).
- deps: `liblzma 0.4` (wraps the same liblzma C lib as Python's stdlib `lzma`),
  `sha2`, `serde`/`serde_json`, `hex` — all pinned in `runtime-rs/Cargo.lock`.
