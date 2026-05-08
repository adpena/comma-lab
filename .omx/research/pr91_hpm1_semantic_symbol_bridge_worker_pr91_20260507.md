# PR91/HPM1 Semantic-Symbol Bridge - Worker PR91 - 2026-05-07

## Scope

Worker PR91 stayed inside PR91/categorical readiness scope. No GPU dispatch,
Lightning state, generated experiment directories, JCSP, or HNeRV entropy
surfaces were touched.

## Current Evidence

- Current categorical candidate audit remains fail-closed:
  [empirical:experiments/results/categorical_openpilot_payload_candidate_hardened_20260507_codex/readiness_audit_hardened.json]
- HPM1 structural inventory is accepted only as structural parity:
  `structural_reencode_matches_source=true`, `full_decode_proven=false`,
  `byte_exact_semantic_reencode_proven=false`.
- HPM1 semantic parity fail-closed proof is accepted only as a non-dispatchable
  divergence guard:
  `divergence_caught_before_exact_eval=true`,
  `semantic_symbol_bridge_status=no_simple_pr85_qma9_to_pr91_symbol_bridge_for_prefix`.
- Newer phase-major prefix probe classifies the closest bridge blocker:
  [empirical:experiments/results/pr91_hpm1_phase_major_prefix_reencode_blocker_20260507_codex/phase_major_prefix_reencode_blocker.json]
  `blocker_class=phase_major_pr85_qma9_reference_symbols_do_not_reproduce_submitted_hpm1_stream`.

## Exact Fail-Closed Finding

The phase-major reference-symbol path is not byte-exact reencode evidence. It
fails before any exact-evaluable stacking claim:

- First local reference word mismatch: `first_1_symbols`, common prefix words
  `0`.
- First submitted/reference prefix mismatch: `first_8_symbols`, decoded symbol
  `2`, reference symbol `0`, symbol index `7`.
- Target failure reproduced under reference context at frame `0`, group `17`,
  symbol in group `437`, decoded symbols before failure `15989`.

## Patch Result

`src/tac/categorical_candidate_readiness.py` now requires semantic-symbol
bridge fail-closed evidence to include:

- the expected fail-closed bridge schema;
- local-only/non-dispatchable proof flags;
- completed prefix counts;
- the four tested bridge classes: identity, global permutation, constant mod5
  offset, and previous-frame mod5 residual;
- a concrete first mismatch.

This prevents a weak bridge note from being accepted as semantic parity closure
while preserving the current accepted fail-closed PR91 proof.

## Next Smallest Implementation Step

Recover one of the following before any exact-eval dispatch:

1. True PR91 encoder semantic symbols/probability rows for the seed prefix, then
   rerun the prefix reemit probe until submitted/reference symbols match through
   at least the first failing seed window.
2. A byte-exact range reemit proof for the submitted PR91 token stream using the
   recovered probability-row contract.

Until one exists, PR91/HPM1 categorical/clade-spade-openpilot stacking remains
blocked by full decode, byte-exact semantic reencode, and sidecar-free runtime
parity, not by static custody.
