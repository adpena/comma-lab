# HPAC/HPM1 Native Entropy Lowering Prototype

Date: 2026-05-04

Scope: local-only Rust/native entropy-coding prototype for PR86/PR91 HPAC/HPM1-style 5-symbol range streams. No remote dispatch, no secrets, no score claim.

## What Changed

- Added `runtime-rs/crates/hpac-codec/Cargo.toml`.
- Added `runtime-rs/crates/hpac-codec/src/lib.rs`.
- Added `crates/hpac-codec` to `runtime-rs/Cargo.toml`.
- Cargo generated/updated `runtime-rs/Cargo.lock` while testing the new crate.
- Added this ledger at `experiments/results/hpac_native_lowering_20260504_worker/ledger.md`.

## Existing Contract Inspected

- Python PR86 source uses `constriction.stream.queue.RangeEncoder` / `RangeDecoder` and per-symbol `constriction.stream.model.Categorical(probabilities=..., perfect=...)`.
- The source-contract variant is `source_float64_perfect_false`: clip and renormalize five softmax probabilities as numpy float64, then use `Categorical(..., perfect=False)`.
- PR91/HPM1 reuses the same HPAC token stream contract inside an `HPM1` mask segment with `tokens_blob` as little-endian `uint32` constriction words.
- The local Python package is `constriction 0.4.2`; its metadata and Rust crate state that the Rust and Python APIs are intended to be binary compatible. The Rust crate uses the same `DefaultRangeEncoder`, `DefaultRangeDecoder`, and `DefaultContiguousCategoricalEntropyModel` presets as the Python bindings for concrete non-lazy categorical models.

## Prototype

`hpac-codec` exposes:

- `encode_f64_probability_rows` / `decode_f64_probability_rows` for source-style float64 probability rows with `perfect=false`.
- option-bearing f64/f32 encode/decode helpers for the known Python probability variants.
- `Cdf5` for explicit 5-symbol 24-bit fixed-point CDFs/weights.
- `AdaptiveCdf5`, `encode_adaptive_symbols`, and `decode_adaptive_symbols` as a deterministic adaptive 5-symbol stream prototype.
- `words_to_le_bytes` / `words_from_le_bytes` for the `tokens.bin` little-endian `uint32` boundary.

The smallest byte-compatibility proof is a frozen Python constriction 0.4.2 fixture:

```text
symbols = [0, 4, 1]
rows = [
  [0.30, 0.10, 0.10, 0.30, 0.20],
  [0.10, 0.40, 0.20, 0.10, 0.20],
  [0.40, 0.20, 0.10, 0.20, 0.10],
]
python constriction queue words = [0x43958018]
little-endian bytes = 18809543
```

The Rust test reproduces those exact words and decodes them back to `[0, 4, 1]`.

## Feasibility

Feasible for the entropy-coding layer. The Rust constriction 0.4.2 crate can reproduce a Python constriction 0.4.2 queue-range fixture exactly for concrete float64 `Categorical(..., perfect=False)` rows, and fixed-CDF/adaptive 5-symbol roundtrips are deterministic.

Not yet a full PR86/PR91 runtime lowering. Python/PyTorch still owns HPACMini model loading, logits, softmax, group-mask traversal, and PR91/HPM1 segment parsing in the main parity rollout. This crate starts at the boundary where a caller already has symbols plus either probability rows or fixed CDFs.

## Compatibility Risks

- Full PR86/PR91 byte parity still depends on reproducing Python row materialization exactly: torch CPU logits, float32 softmax, numpy dtype conversion, clipping, left-to-right renormalization, group order, and previous-context updates.
- `source_float64_perfect_false` is the only promotable current contract. f32 and `perfect=true` helpers are included for variant probes, not as evidence that submitted streams use those variants.
- The fixed-CDF/adaptive path is deterministic and useful for native lowering experiments, but it is a new explicit-CDF contract. It cannot decode existing PR86/PR91 float-Categorical streams unless the same fixed CDFs are proven equivalent to constriction's float quantization for every symbol.
- Rust integration must keep `tokens.bin` little-endian `uint32` aligned. Byte blobs with non-multiple-of-4 length fail closed.
- This prototype does not touch `inflate.sh` or contest runtime. It is local-only until full archive parity exists.

## Verification

Command:

```bash
cd /Users/adpena/Projects/pact/runtime-rs
cargo test -p hpac-codec
```

Result:

```text
5 passed; 0 failed
```

Covered:

- Python constriction 0.4.2 f64 `perfect=false` word fixture.
- Fixed-CDF 5-symbol encode/decode roundtrip.
- Adaptive 5-symbol encode/decode roundtrip.
- Little-endian `uint32` byte helper roundtrip and fail-closed misalignment.
- Out-of-range symbol and malformed CDF rejection.

## Next Integration Steps

1. Add a Python-side fixture exporter owned by the main parity rollout: first N symbols plus normalized probability rows from PR86/PR91, not raw logits.
2. Add a Rust fixture test that consumes that exported row file and proves exact `tokens_blob` prefix words against Python constriction.
3. Only after prefix word parity, add a narrow FFI/CLI bridge that accepts little-endian `tokens.bin`, row fixtures, and symbol count for decode/re-encode timing.
4. Defer HPACMini native/PyTorch lowering until the entropy layer is locked; otherwise model numeric drift will hide range-coder regressions.
