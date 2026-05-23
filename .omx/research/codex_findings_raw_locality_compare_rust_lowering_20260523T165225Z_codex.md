# Codex Findings: Raw Locality Compare Rust Lowering

**UTC**: 2026-05-23T16:52:25Z
**Scope**: profile and lower the DQS1 selective-runtime raw triplet comparison hotspot.

## Findings

- The locality-control compare phase is SHA-bound, not Python loop-bound on the synthetic smoke. On a 201,326,592-byte triplet, `compare_raw_triplet(..., raw_compare_backend="python")` took 0.1288s, with 0.117s inside `_hashlib.HASH.update`.
- A naive native lowering with pure Rust `sha2` was a regression on macOS arm64. On a 402,653,184-byte triplet it took about 26.4s, because Python uses an optimized OpenSSL-backed SHA path.
- The landed Rust comparator uses memory maps plus `ring` SHA-256. On the same 402,653,184-byte triplet, Python measured 0.2658s, Rust release measured 0.4176s cold and 0.2725s warm via the Python auto path. That is comparable but not a universal win.

## Integration

- The native comparator lives at `runtime-rs/crates/raw-locality-compare` and is in the `runtime-rs` workspace.
- `tools/run_decoder_q_selective_runtime_locality_controls.py` now accepts `--raw-compare-backend {auto,python,rust}`. `python` remains the correctness oracle and fallback; `rust` is available for explicit native checks and future lower-level optimization.
- The invalid selected-frame path now hashes raw files instead of referencing an unbound `file_hashes` variable.

## Authority

This is a locality-control acceleration surface only. It does not run the scorer and it carries no score, promotion, rank, kill, or exact-eval authority.
