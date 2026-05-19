# Codex Findings: Master-Gradient Backfill + Trust-Region Compiler

Timestamp: 2026-05-19T11:48:40Z  
Owner: codex  
Task: `codex_routing_directive_v2_synthesis_followup_null_space_plus_hash_seed_plus_cross_stack_20260518::ITEM_7`

## What Changed

Closed the OP-7 producer-side authority gap without hand-editing historical state.

- Added `tac.master_gradient.append_score_axis_dominance_backfill(...)`, an append-only locked helper that derives `score_axis_dominance` from the stored gradient tensor and persists a corrected anchor row.
- Added `tools/backfill_master_gradient_score_axis_dominance.py` for operator-visible backfill manifests.
- Updated master-gradient anchor readers to prefer later append-only corrections when `measurement_utc` ties.
- Backfilled the PR101 lc-v2 OP-7 anchor through the locked helper.
- Refreshed the OP-7 pose-axis manifest so `anchor_score_axis_dominance_not_persisted` is gone.
- Added `tac.master_gradient_trust_region` plus `tools/build_master_gradient_trust_region_candidates.py` to convert the exact OP-7 regression into block-aware, score-claim-false candidate manifests.

## Live Artifacts

- `.omx/research/master_gradient_score_axis_backfill_pr101_lc_v2_20260519T114258Z_codex.json`
  - append performed: `true`
  - pose-axis dominant byte count: `346`
  - source anchor SHA before: `90ecee15b114b3956bd12c15d8834571b05f5b03fcee8220143e266e4fdebe65`
- `.omx/research/pose_axis_operator_pr101_manifest_20260519T114316Z_score_axis_backfilled_codex.json`
  - source anchor score-axis dominance: `available=true`, `source=anchor_field`
  - remaining blockers: packet proofs and inflate/runtime proofs only
- `.omx/research/master_gradient_trust_region_candidates_pr101_op7_20260519T114840Z_codex.json`
  - candidates: `uniform`, `gradient_weighted`, `stc_style`, `segnet_boundary_preserving`
  - iid assumption: `false`
  - temporal block size: `42` pairs
  - mutation intensity cap: `0.15471980751989775`
  - exact negative guard: same-length raw-delta candidate `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a` is forbidden for rerun/promotion

## Authority Boundary

These artifacts do not claim score movement. The trust-region compiler emits candidate manifests only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- blockers include packet materialization, packet proofs, inflate proof, no-op detection, score-response probe, and exact eval

The exact OP-7 raw-delta result remains a measured configuration regression. The family is reactivated only through a smaller, block-aware, grammar-aware trust-region packet with a new archive SHA.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_extract_master_gradient.py \
  src/tac/tests/test_master_gradient_consumers.py \
  src/tac/tests/test_hoist_pose_bytes_from_master_gradient.py \
  src/tac/tests/test_master_gradient_pr101_operator_candidate.py \
  src/tac/tests/test_master_gradient_pr101_score_response_matrix.py \
  src/tac/tests/test_backfill_master_gradient_score_axis_dominance_tool.py \
  src/tac/tests/test_master_gradient_trust_region.py
# 116 passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  src/tac/master_gradient.py \
  src/tac/master_gradient_consumers.py \
  src/tac/master_gradient_trust_region.py \
  tools/backfill_master_gradient_score_axis_dominance.py \
  tools/build_master_gradient_trust_region_candidates.py \
  src/tac/tests/test_extract_master_gradient.py \
  src/tac/tests/test_master_gradient_consumers.py \
  src/tac/tests/test_backfill_master_gradient_score_axis_dominance_tool.py \
  src/tac/tests/test_master_gradient_trust_region.py
# All checks passed
```
