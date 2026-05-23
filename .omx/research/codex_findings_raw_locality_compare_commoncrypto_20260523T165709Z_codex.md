# Codex Findings: Raw Locality Compare CommonCrypto Backend

**UTC**: 2026-05-23T16:57:09Z
**Scope**: full-scale DQS1 raw locality compare after lowering the native SHA path to macOS CommonCrypto.

## Findings

- Full-scale rank024 locality artifacts are 3,662,409,600 bytes per raw file across parent, global-mutated, and selective outputs.
- The earlier native SHA path regressed badly at full scale: the Rust backend completed the rank024 compare in 302.7909s while still producing zero locality mismatches. The bottleneck was digest implementation / memory traversal, not locality logic.
- The macOS build now uses CommonCrypto SHA-256 through Rust FFI and keeps the non-macOS `ring` fallback. On the same rank024 artifacts with cached pages, the locality compare phase completed in 7.0129s through the Python runner, and the direct binary profile measured `real 6.96`, `user 6.69`, `sys 0.25`.
- The native result reports `raw_compare_backend="rust"` and `compare_engine={"crate":"raw-locality-compare","name":"rust","schema":"raw_locality_compare_engine.v1"}` with zero missing, size, selected-frame, or unselected-frame mismatches.

## Interpretation

This validates the lowered Rust path as a useful hot-path backend only when paired with the platform-accelerated digest. It also preserves the earlier caution: cold external-volume reads and cache state dominate wall time, so queue reports must keep phase timings instead of assuming a single backend speed.

## Authority

This remains `[locality-control no-score]` evidence only. It does not run the contest scorer and has no score, promotion, rank, kill, or exact-eval authority.
