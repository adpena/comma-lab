# L5-v2 PR106 PacketIR Exact-Evidence Reactivation - 2026-05-16

## Context

The PR106 PacketIR matrix was over-conservative in two ways that suppressed
useful L5-v2 planning signal:

1. Historical PR106 R2 and PR101-grammar exact-eval artifacts were classified
   invalid because exported component averages were rounded enough to drift by
   about `1e-7` from the stored canonical score. That created duplicate
   re-dispatch targets for evidence that already exists.
2. `format_0x0c_exact_radix` and `prefix_top_16_pr101grammar` had valid paired
   CPU/CUDA exact artifacts with matching `runtime_content_tree_sha256`, but the
   older runtime-consumption manifests did not carry that content hash. L5-v2
   therefore saw zero runtime-bound paired PR106 candidates.

## Fix

- PacketIR exact evidence now tolerates only legacy rounded-component score
  mismatches when the artifact's stored canonical score and
  `score_recomputed_from_components` agree within the strict exact-eval
  tolerance and the exported-component drift is within `1e-6`.
- Real score-formula mismatches remain blocked.
- Runtime consumption can derive `runtime_content_tree_sha256` from exact eval
  only when both CPU and CUDA axes are valid and their runtime-content hashes
  match exactly. Axis/runtime mismatches still block pairing.
- Regenerated `.omx/research/pr106_packetir_candidate_matrix_20260516_codex.*`
  and updated the L5-v2 pinned matrix SHA.

## Resulting State

- Candidate statuses:
  - `paired_exact_measured`: 2
  - `paired_exact_blocked`: 3
  - `single_axis_exact_measured_needs_pair`: 9
  - `runtime_consumed_needs_paired_exact_eval`: 2
- Next paired Modal exact-eval targets: 11, down from 13.
- L5-v2 PacketIR paired candidates: 2
  - `format_0x0c_exact_radix`
  - `prefix_top_16_pr101grammar`
- L5-v2 PR106 stack-cell proposal count: 2

All surfaces remain non-promotional:
`score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

## Evidence

- `.venv/bin/python -m ruff check src/tac/packet_compiler/pr106_candidate_matrix.py src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_operator_briefing.py`
  - PASS
- `PYTHONPATH=src:. .venv/bin/pytest src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_all_lanes_operator_briefing_gate.py -q`
  - PASS: 96 passed
- `tools/operator_briefing.py --json --top 10`
  - PacketIR matrix SHA matches pinned expected SHA.
  - Dispatch targets are not suppressed.
  - `next_exact_eval_target_count=11`
  - `packetir_paired_candidate_count=2`
  - `pr106_stack_cell_candidate_count=2`

## Remaining Blockers

The two PR106 PacketIR paired candidates are stack-planning inputs only. Any L5-v2
composition still requires a byte-closed composite archive, side-info
consumption proof, paired CPU/CUDA exact eval, and L5-v2 gate evidence before
rank or promotion language.
