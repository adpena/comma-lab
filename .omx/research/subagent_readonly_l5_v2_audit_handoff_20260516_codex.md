# Subagent Read-Only L5 v2 Audit Handoff

- date: `2026-05-16`
- scope: L5 v2 staircase, TT5L readiness, paired Modal dispatch planning
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Source

This ledger preserves read-only subagent findings returned during the
`commit and push all to origin/main with no signal loss` sweep. No subagent
edited files directly.

## Findings To Preserve

### Side-Info Gate Evidence

The canonical side-info gate rejects the current proof artifacts because the
inflate provenance entries lack log binding. The old local proof remains scoped
as toy/parser consumption. The contest full-frame proof is semantically useful
for byte-consumption, but stale for the current gate because baseline and mutated
inflate provenance need `log_path`, `log_sha256`, and `log_bytes`.

Recommended patch:

- add `--log-path` to `tools/build_tt5l_inflate_provenance.py`;
- pass it to `build_tt5l_inflate_provenance_manifest(log_path=...)`;
- regenerate baseline and mutated provenance using the existing side-info proof
  logs;
- rebuild the contest side-info consumption proof;
- update the committed expected proof hash;
- add a live readiness regression that verifies the committed proof is accepted.

### Move-Level Feasibility

The move-level feasibility surface can still be fooled by JSON shape. A real
artifact needs a generating tool, input artifact SHAs, projection transcript,
constraint matrix, witness variables, tool hash, residual vector, and negative
tests proving handwritten JSON is rejected.

Until that producer exists, the honest classification is
`move_level_feasibility_producer_missing`, not `TT5L feasible`.

### Paired Modal Dispatch Plan

The paired measurement planner should reuse:

- `tools/dispatch_modal_paired_auth_eval.py`;
- `src/tac/deploy/modal/paired_dispatch.py`;
- `src/tac/deploy/modal/paired_dispatch_contract.py`.

Do not emit standalone active lane preclaims ahead of Modal wrappers; the
wrappers own the claim lifecycle. Runtime custody should use
`--expected-runtime-tree-sha256 auto`, and top-level blockers must stay nonempty
while archive/runtime/score-custody placeholders remain unresolved.

Required false-authority flags:

- `planning_only: true`
- `score_claim: false`
- `score_claim_valid: false`
- `promotion_eligible: false`
- `ready_for_exact_eval_dispatch: false`
- `rank_or_kill_eligible: false`
- `dispatch_attempted: false`
- `adjudication_required: true`

## Next Artifact Order

1. Log-bound TT5L side-info provenance rebuild.
2. Real move-level feasibility producer or explicit blocker artifact.
3. C1/Z5/TT5L paired exact probe rows.
4. Paired CPU+CUDA dispatch plan through the canonical Modal paired dispatcher.
