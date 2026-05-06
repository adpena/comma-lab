# AQc1 Sparse Arithmetic Container

Date: 2026-05-06
Author: codex
Evidence grade: empirical codec fixture
Score claim: false
Dispatch attempted: false

## Context

The existing AQv1 arithmetic qint container transmits a dense
`uint32[num_symbols]` frequency table. That is fine for tiny ternary streams
but byte-wasteful for sparse large alphabets used by some sidechannel,
boundary, and joint-stack experiments.

## Change

Added `AQc1`, a sparse-table companion container:

- same deterministic arithmetic coder as AQv1
- observed-symbol frequency table only
- explicit little-endian symbol/count pairs
- decode path exposed through `decode_qints_arithmetic`
- profile path exposed through `profile_arithmetic_container`
- existing AQv1 encoder remains the default

## Boundary

This is not a score claim and does not promote any archive. It is a deterministic
codec primitive and audit surface. Promotion still requires a byte-closed
archive member, runtime loader parity, lane claim, and exact CUDA auth eval on
the exact archive bytes.

## Verification

Focused tests:

`src/tac/tests/test_arithmetic_qint_codec.py`

`src/tac/tests/test_joint_codec_stack_orchestrator.py`

Result: `26 passed`.
