# CHECKPOINTS - ddm_vo2

## Checkpoint 1 - Intake

- Read `.omx/tmp/codex_runs/vo2_prompt.md`.
- Read `.omx/tmp/codex_runs/_common_contract.md`.
- Read `PROGRAM.md`.
- Read governing sections of `CLAUDE.md` / `AGENTS.md`, `docs/operating_manual_craft_handoff.md`,
  and `.omx/state/main_hot_state.md`.
- Confirmed scorer-free scope and `ddm_et2` scorer-slot ownership.

## Checkpoint 2 - Recall

- Read VO1, CA1, SW1, and DK1 receipts/ledgers.
- Verified `p0_instrument_fractal_audit_20260806` already exists in
  `.omx/state/operator_p0_ledger.jsonl`; did not edit the dirty shared ledger.
- Confirmed existing dirty protected-file state was unrelated and left untouched.

## Checkpoint 3 - Build

- Added registry builder.
- Added warn-only form-grade reference checker.
- Added focused tests and positive control.

## Checkpoint 4 - R1 Materialized

- Registry rows: 4,630.
- Unique instrument IDs: 4,630.
- R1 dry: false.
- Registry SHA-256: `947b7faaa3ba61dfad567b434075c8151028e8fd5e6dbe3c38cbcb4ccc43b936`.

## Checkpoint 5 - Verification

- `py_compile`: passed.
- Focused pytest: 3 passed.
- Positive control: 1 passed.
- Form-grade scoped scan: 4 round-0 receipts scanned, 0 direct registry-ID citations, 0 missing refs.
- The builder now uses a deterministic audit-date `last_graded` label by default; use `--last-graded`
  only for an intentional refresh.

No scorer, archive, training, paid dispatch, or upstream edit happened in this generation.

## Checkpoint 6 - Generation 2 R2 Resume

- Read `.omx/tmp/codex_runs/vo2r2_prompt.md`.
- Re-read `NEXT_IF_RESUMED.md`, `.omx/tmp/codex_runs/vo2_prompt.md`, and
  `.omx/tmp/codex_runs/_common_contract.md`.
- Re-ran `tools/build_ddm_vo2_instrument_registry.py --out-dir .omx/research/ddm_vo2_20260806`.
- Verified the rebuild matched the R1 manifest before R2:
  - registry SHA-256 `947b7faaa3ba61dfad567b434075c8151028e8fd5e6dbe3c38cbcb4ccc43b936`
  - summary SHA-256 `0b450d49d33d1ba8e756b1d16d031f144fd05a490ace2c4292b577d4bb2b4393`
  - row count 4,630.

## Checkpoint 7 - R2 Element Batch 1

- Added `.omx/research/ddm_vo2_20260806/R2_ELEMENT_DECOMPOSITION.jsonl`.
- R2 selected rows: 23, each with all ten charter elements present.
- Row groups: vo1-round0 1, ca1-round0 6, sw1-round0 8, dk1-round0 3, vo2-new 5.
- Source-candidate rows are explicitly `R2_SOURCE_CANDIDATE_GRADED_NOT_CONSUMER_PROOF`.
- New R3 lineage instrument surfaced:
  `src/tac/canonical_equations/trajectory_derived_stopping_20260805.py`.
- `ROUND_SUMMARY.json` now records `round_reached=R2-partial`, `r2_complete=false`, and
  `seal_ready=false`.

No scorer, archive, training, paid dispatch, or upstream edit happened in Generation 2.
