# SDWL1 post-repair adversarial review and seal

date_utc: 2026-07-23
lane_id: lane_ddm_dv2_grammar_sentences_20260723
axis: `[macOS-CPU frozen-scorer advisory]`
score_claim: false
main_landing_review_required: true

## Disposition

`SEAL_FOR_ISOLATED_COMMIT`

This seal follows the round-one rejection and all repairs. It authorizes only an
isolated-branch serializer commit for MAIN review. It does not authorize a
candidate archive, launch, evaluator claim, promotion, or merge.

## Clean pass 1 — semantic and wire re-derivation

`CLEAN`

- Re-derived the 11-record/76-scalar accounting from row widths.
- Verified canonical-zero padding is now enforced at inventory construction.
- Reviewed all three layouts, absolute and modulo-\(2^{64}\) causal transforms,
  per-pair independent resets, production inference, lexicon pruning, strict
  section order, whole-schema canonical comparison, and complete outer-zlib
  handling.
- Confirmed every declared type has a named derivation entry and only
  measured-use vocabulary enters the base lexicon.
- Focused acceptance suite passed 17 tests.

## Clean pass 2 — custody, failure, and resumability

`CLEAN`

- Verified the complete 5,078,017,610-byte source SHA-256 before direct,
  read-only ZIP_STORED member mapping.
- Verified atomic custody, inventory, row, payload, and final-receipt stages.
- Found and repaired absolute temporary-worktree path leakage before this pass;
  the regenerated evidence uses repository-relative in-tree paths.
- Audited all 33 payload byte counts and SHA-256 values from disk.
- Ran strict `--resume`: source rehash, inventory binding, all-row arithmetic
  parse-back, and byte-identical final receipt SHA-256
  `efc43fcda1f12f28df2b6059cd5e51e7ee2509a356d99b59e317b253927a709c`.

## Clean pass 3 — scope, claims, and landing boundary

`CLEAN`

- Confirmed the owned implementation contains none of the prohibited vehicle
  identifiers or artifacts.
- Confirmed no scorer, provider, GPU, live-run, candidate-archive, or canonical
  pointer path is invoked.
- Confirmed every persisted result is
  `[macOS-CPU frozen-scorer advisory]`, `score_claim=false`,
  `promotion_eligible=false`, and pointer-unchanged.
- Confirmed no temporary evidence path, lockfile change, whitespace error,
  symlink, or unrelated tracked-file edit.
- Reconciled findings and equations against the final receipt: selected typed
  causal row 68,464 bytes; all four optional syntax dimensions pruned; no score
  or receiver inference.
- Ruff check and format-check passed on all three Python surfaces.

## Environment note

The bare Homebrew `python3` lacks `pytest`. With the repository virtualenv first
on `PATH`, the requested command
`python3 -m pytest -q src/tac/optimization/tests/test_ddm_dv2_sdwl1.py`
passes 17/17. This is an environment dependency fact, not a test failure.

## Remaining mandatory gate

MAIN must inspect the isolated commit diff and decide whether to merge. The
canonical frontier pointer remains unchanged regardless of that landing
decision.
