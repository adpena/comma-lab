# Codex Findings: PR101 Pose-Axis Raw-Delta Builder

Date: 2026-05-19T08:44:39Z

## Scope

Continued
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`.

The previous PR101 OP-7 builder proved packet mechanics with a same-length
Brotli recompression that was raw-equivalent. This landing adds the next
required local step: an opt-in same-length component-moving mode that mutates
one decompressed decoder-stream byte, recompresses back to the same compressed
length, and preserves all no-score authority boundaries.

## Landing

- `src/tac/master_gradient_pr101_operator_candidate.py`
  - Adds `mutation_mode=raw_equivalent|raw_byte_delta`.
  - Keeps `raw_equivalent` as the default.
  - Adds opt-in raw-stream byte mutation before same-length Brotli recompression.
  - Emits explicit `component_moving_candidate`, raw SHA, raw mutation, and
    packet-proof fields.
- `tools/build_pr101_pose_axis_operator_candidate.py`
  - Adds CLI flags `--mutation-mode`, `--raw-byte-offset`, and
    `--raw-byte-delta`.
- `src/tac/tests/test_master_gradient_pr101_operator_candidate.py`
  - Covers default raw-equivalent behavior, opt-in component-moving behavior,
    and zero-delta fail-closed behavior.

## Local Artifact

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_pr101_pose_axis_operator_candidate.py \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --operator-manifest .omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json \
  --output-dir experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex \
  --candidate-id pr101-op7-rank1-raw-byte-delta-same-length \
  --mutation-mode raw_byte_delta \
  --raw-byte-delta -1
```

Result:

- candidate archive:
  `experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/archive.zip`
- candidate archive bytes: `178258`
- candidate archive SHA-256:
  `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`
- archive byte delta vs source: `0`
- selected OP-7 rank: `1`
- selected split-Brotli stream: `2`
- compressed span: `1970:115197`
- source stream compressed SHA-256:
  `2379a2910a96ef0350c7b4cdde5a2547f6b29c5b0b3c2ff77b704ab0ce8ea26c`
- replacement stream compressed SHA-256:
  `f95b065ec0874b7bc3876dbfa853650ced7072b4c811212881c3e789229e0bca`
- source stream raw SHA-256:
  `00c40fe042a34761fa2d334e0820aa8d7075632a0ce13c70492d5ca405b605fa`
- replacement stream raw SHA-256:
  `0877a3891cdc3faf113d388b191ff528393697f7261ef02f2bab6ec1acd3b272`
- raw mutation: offset `33803`, delta `-1`, value `19 -> 18`
- replacement quality/lgwin: `10` / `18`

Runtime-consumption proof:

- proof path:
  `experiments/results/pr101_pose_axis_operator_candidate_raw_delta_20260519T084439Z_codex/runtime_consumption_proof.json`
- proof SHA-256:
  `3f5899da99c47e0c2987634a050721cf92dbf94b621c023e4732c9d24426550d`
- runtime log SHA-256:
  `f7c4ba3a585cca64b17f6ae7fd9eee90617f11787f5be3bd4ce1c6ee292ad2e3`
- runtime proof ready: `true`
- runtime proof blockers: `[]`

## Authority Boundary

This is not a score claim, promotion claim, rank claim, kill claim, or provider
dispatch. It is a byte-closed, parser-proven, runtime-consumed local candidate
that changes decoder raw bytes while preserving archive size.

Remaining blockers:

- full inflate success/output manifest;
- score-response matrix for the component-moving mutation;
- active lane claim before any paid/provider dispatch;
- exact `[contest-CUDA]` and paired `[contest-CPU]` auth eval before score
  language.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/tac/master_gradient_pr101_operator_candidate.py \
  tools/build_pr101_pose_axis_operator_candidate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# 4 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_prove_monolithic_runtime_consumption.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# 8 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/master_gradient_pr101_operator_candidate.py \
  tools/build_pr101_pose_axis_operator_candidate.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# All checks passed
```

