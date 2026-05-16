# L5 v2 PR106 stack-cell selector

Date: 2026-05-16
Owner: Codex
Scope: L5 v2 staircase / PR106 PacketIR / stack-of-stacks candidate selector

## Purpose

The L5 v2 readiness payload now exposes PR106 PacketIR rows as concrete
TT5L composite-cell proposals. This is a planning selector only. It ranks
paired PR106 source rows so the next implementation pass has an explicit
candidate order, while preserving the rule that no composite packet exists
until it is materialized, consumption-proved, and paired exact-evaluated.

## Code Surfaces

- `src/tac/optimization/l5_staircase_v2.py`
  - added `l5_v2_pr106_stack_cell_candidates()`
  - added `pr106_stack_cell_candidates` to `l5_v2_dispatch_readiness()`
- `src/tac/tests/test_l5_staircase_v2.py`
  - verifies non-promotional blocked proposals
  - verifies `top_k`
  - verifies fail-closed behavior without the PR106 matrix artifact

## Current Top-3 Proposal Order

Generated with `l5_v2_pr106_stack_cell_candidates(top_k=3)`:

1. `time_traveler_l5_autonomy+format_0x0d_latent_score_table`
   - source max axis score: `0.22712591739832488`
2. `time_traveler_l5_autonomy+format_0x0c_exact_radix`
   - source max axis score: `0.22776488386973992`
3. `time_traveler_l5_autonomy+prefix_top_16_pr101grammar`
   - source max axis score: `0.23010997767094082`

These values are sorting keys from source PR106 paired rows, not composite
score claims.

Every proposed cell carries:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Every proposed cell remains blocked by:

- `requires_l5_v2_composite_archive_materialization`
- `requires_l5_v2_composite_paired_exact_eval`
- `requires_composite_sideinfo_consumption_proof`

## Verification

Commands run from repo root:

```bash
.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_pr106_packetir_candidate_matrix.py -q
.venv/bin/python - <<'PY'
from tac.optimization.l5_staircase_v2 import l5_v2_pr106_stack_cell_candidates
p = l5_v2_pr106_stack_cell_candidates(top_k=3)
print(p["schema"])
print("candidate_count", p["candidate_count"])
print("blockers", p["blockers"])
for row in p["candidates"]:
    print(row["cell_id"], row["source_max_axis_score"], row["blockers"])
PY
```

Observed results:

- `All checks passed!`
- `49 passed`
- selector schema `l5_v2_pr106_packetir_stack_cell_candidates_v1`
- top-k blockers `[]` at selector level; each cell still carries the required
  composite-materialization blockers listed above.

## Follow-Up

Materialize the top proposal as a byte-closed composite archive candidate, then
prove section consumption and run paired exact CPU/CUDA eval before any score
or promotion decision.
