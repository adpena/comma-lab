# Codex Findings: Pose-Axis Master-Gradient Operator Resolution

**UTC:** 2026-05-19T07:45:00Z  
**Task:** `codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`  
**Status:** partial implementation landed; still blocked before mutation/probe authority.

## Finding

The OP-7 pose-axis master-gradient selector now has a grammar-aware lowering
step. Diagnostic `gradient_subject_byte_index` rows are resolved through a
parser-proven archive layout into typed `CandidateModificationSpec` rows with
`coordinate_system=grammar_aware_operator_response`.

This is not raw archive-byte authority. The new rows keep:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `raw_archive_byte_coordinates_allowed=false`

The packet-proof blockers remain explicit until a grammar-specific mutation
builder proves repack, ZIP metadata, CRC, inflate success, and no-op/byte
consumption closure.

## Implementation Surface

- `src/tac/master_gradient_operator_plan.py`
  - added `build_pose_axis_operator_candidates(...)`
  - supports nested `logical_layout.sections` and extract-layout top-level
    `sections`
  - treats headers, magic, raw sections, and framing metadata as unresolved
    non-mutation targets
- `tools/hoist_pose_bytes_from_master_gradient.py`
  - added `--layout-manifest`
  - embeds grammar-aware operator candidate resolution in the OP-7 manifest
  - changes smoke status from missing builder to missing packet proofs when at
    least one diagnostic entry resolves
- `src/tac/tests/test_master_gradient_operator_plan.py`
- `src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py`

## Empirical Artifact

Archive:
`experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`

Archive SHA-256:
`b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`

Layout extraction:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip \
  --detect-grammar-only \
  --layout-contract-output .omx/research/pose_axis_operator_pr101_layout_contract_20260519T074500Z.json
```

OP-7 manifest:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/hoist_pose_bytes_from_master_gradient.py \
  --archive-sha256 b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e \
  --top-k 8 \
  --axis-dominance-threshold 0.7 \
  --layout-manifest .omx/research/pose_axis_operator_pr101_layout_contract_20260519T074500Z.json \
  --output-dir .omx/research/pose_axis_operator_pr101_artifacts_20260519T074500Z \
  --manifest-path .omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json
```

Manifest result:

- selected pose-axis diagnostic entries: 8
- resolved grammar-aware operator candidates: 8
- unresolved entries: 0
- logical grammar: `pr101_lc_v2`
- first resolved candidate: diagnostic index `35773` -> `decoder`
  section, role `brotli_streams_int8`, operator
  `decoder_codec_coordinate_response`
- smoke status: `blocked_missing_packet_proofs`

Primary artifact paths:

- `.omx/research/pose_axis_operator_pr101_layout_contract_20260519T074500Z.json`
- `.omx/research/pose_axis_operator_pr101_manifest_20260519T074500Z.json`
- `.omx/research/pose_axis_operator_pr101_artifacts_20260519T074500Z/master_gradient_consumers/pose_axis_dominant_bytes_b83bf3488625_op7_manifest_v1.json`

## Authority Boundary

This finding is diagnostic/planning evidence only. It proves that the pose-axis
selector can be mapped into a grammar-aware operator coordinate system for a
real public PR101 archive. It does not prove that any byte mutation is valid,
score-lowering, or contest-promotable.

Remaining blockers:

- `packet_proofs_missing`
- `anchor_score_axis_dominance_not_persisted`
- `archive_not_repacked_after_mutation`
- `zip_headers_not_rebuilt`
- `zip_crc_not_rebuilt`
- `inflate_success_not_proven`

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_master_gradient_operator_plan.py \
  src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py \
  src/tac/tests/test_master_gradient_consumers.py
```

Result: `57 passed in 0.47s`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  src/tac/master_gradient_operator_plan.py \
  tools/hoist_pose_bytes_from_master_gradient.py \
  src/tac/tests/test_master_gradient_operator_plan.py \
  src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py
```

Result: `All checks passed!`

## Next Action

Build the first grammar-specific mutation builder for the highest-EV resolved
candidate, then rerun the OP-7 manifest with packet proofs available only after
repack, ZIP metadata, CRC, inflate success, and byte-consumption proofs exist.
