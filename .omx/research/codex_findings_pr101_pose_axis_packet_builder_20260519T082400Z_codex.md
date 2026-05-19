# Codex Findings: PR101 Pose-Axis Packet Builder

Date: 2026-05-19T08:24:00Z

## Scope

Continued
`codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`
after the OP-7 manifest resolved the top PR101 pose-axis diagnostic rows onto
the parser-proven PR101 decoder section.

## Landing

- Added `src/tac/master_gradient_pr101_operator_candidate.py`.
- Added `tools/build_pr101_pose_axis_operator_candidate.py`.
- Added `src/tac/tests/test_master_gradient_pr101_operator_candidate.py`.
- Extended `tools/prove_monolithic_runtime_consumption.py` to consume
  `pr101_fixed_offset_hnerv_microcodec` packets, including split-Brotli
  decoder streams.
- Added PR101 runtime-proof coverage in
  `src/tac/tests/test_prove_monolithic_runtime_consumption.py`.
- Fixed the stale `Task-Aware Codec` expansion in
  `reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft`; canonical
  wording is `Task-Aware Compression`.

## Candidate Artifact

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/build_pr101_pose_axis_operator_candidate.py \
  --source-archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --operator-manifest .omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json \
  --output-dir experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex \
  --candidate-id pr101-op7-rank1-same-length-brotli-stream
```

Result:

- candidate archive:
  `experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/archive.zip`
- candidate archive bytes: `178258`
- candidate archive SHA-256:
  `959d1e3955b9f8835f3ffa1ad1945d40eb83af370cfc5dc50e137d001d35b17c`
- archive byte delta vs source: `0`
- selected OP-7 rank: `1`
- diagnostic byte index: `35773`
- section: `decoder_blob`
- selected split-Brotli stream: `2`
- stream compressed span: `1970:115197`
- source stream SHA-256:
  `2379a2910a96ef0350c7b4cdde5a2547f6b29c5b0b3c2ff77b704ab0ce8ea26c`
- replacement stream SHA-256:
  `7531ad9b3ca3166a5a361ce58ab08fd9c9e850675b7e7718f4dd57b8ad28b46e`
- replacement quality/lgwin: `10` / `18`
- replacement decoder-section SHA-256:
  `ba0b618b4eb373397c5ee6c91f7b2bdd0c2618ea94cd1c249e4732aadc3d808d`

## Packet Proofs

The candidate is a same-length, byte-different, lossless split-Brotli
recompression of the containing decoder stream. It is intentionally
score-neutral until a future lossy/score-response operator is materialized.

Proved:

- repacked archive: `true`
- ZIP headers updated/bound: `true`
- ZIP CRC valid: `true`
- parser reparse success: `true`
- structural section non-noop: `true`
- split-Brotli stream reparse success: `true`
- decoder Brotli raw equivalence: `true`

Runtime-consumption proof command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_monolithic_runtime_consumption.py \
  --candidate-manifest experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/candidate_manifest.json \
  --runtime-log-out experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/runtime_consumption.log \
  --json-out experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/runtime_consumption_proof.json \
  --command-text 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/prove_monolithic_runtime_consumption.py --candidate-manifest experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/candidate_manifest.json --runtime-log-out experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/runtime_consumption.log --json-out experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/runtime_consumption_proof.json' \
  --fail-if-not-ready
```

Runtime proof:

- proof path:
  `experiments/results/pr101_pose_axis_operator_candidate_20260519T082400Z_codex/runtime_consumption_proof.json`
- proof SHA-256:
  `44b45c31d55e9903be30b08a14d4a2c82924900f256d6b265b38af5bebd1eeb5`
- runtime log SHA-256:
  `1541d49bf6b30d281b18fb8bbee939c457e058c0c60f4b06491865718b7b2e4d`
- runtime grammar: `pr101_fixed_offset_hnerv_microcodec`
- consumed changed section: `decoder_blob`
- split-Brotli stream count: `7`
- decoded decoder raw bytes: `229014`
- proof blockers: `[]`
- `ready_for_exact_eval_runtime`: `true`

## Authority Boundary

This is not a score claim, not a promotion claim, and not an exact-dispatch row.
The changed compressed decoder bytes are consumed by the PR101 runtime grammar,
but they decompress to the same decoder raw bytes. Therefore the current
artifact closes packet mechanics and runtime-consumption proof for OP-7 rank 1;
it does not prove scorer movement.

Remaining blockers:

- full inflate output proof or exact same-runtime eval for semantic parity;
- score-response matrix for a genuinely component-moving operator;
- active lane claim before any paid/provider dispatch;
- exact `[contest-CPU]` / `[contest-CUDA]` auth eval before score language.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_master_gradient_operator_plan.py \
  src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py \
  src/tac/tests/test_master_gradient_consumers.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# 59 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_prove_monolithic_runtime_consumption.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# 6 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  tools/prove_monolithic_runtime_consumption.py \
  src/tac/tests/test_prove_monolithic_runtime_consumption.py \
  src/tac/master_gradient_pr101_operator_candidate.py \
  tools/build_pr101_pose_axis_operator_candidate.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_tac_terminology.py --strict --json
# finding_count=0

git diff --check -- \
  src/tac/master_gradient_pr101_operator_candidate.py \
  tools/build_pr101_pose_axis_operator_candidate.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py \
  tools/prove_monolithic_runtime_consumption.py \
  src/tac/tests/test_prove_monolithic_runtime_consumption.py \
  reports/oss_d3_d4_drafts_20260514/THIRD_PARTY_NOTICES.md.draft
# clean
```
