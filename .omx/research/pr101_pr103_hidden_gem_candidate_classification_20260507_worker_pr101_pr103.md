# PR101/PR103 Hidden-Gem Candidate Classification - 2026-05-07 Worker PR101/PR103

## Scope

This pass classifies only PR101/PR103 schema-manifest surfaces. It does not
dispatch, does not claim score, and does not edit
`tools/build_field_meta_dispatch_selection.py` or `src/tac/frontier_rows.py`.

## Candidate Table

| candidate | artifact | classification | stack | substitute | evidence semantics | next action |
|---|---|---|---:|---:|---|---|
| PR101 f32 schema on PR106x | `experiments/results/hnerv_pr101_schema_candidate_20260507_codex/pr101_schema_archive_candidate_manifest.json` | `stack_candidate` | true | false | empirical byte equivalence: raw decoder SHA preserved, no score claim | add runtime adapter, runtime-tree parity manifest, inflate-output parity, strict compliance |
| PR101 fp16 scale probe on PR106x | `experiments/results/hnerv_pr101_schema_candidate_20260507_codex/pr101_schema_archive_candidate_manifest.json` | `substitute_candidate_probe` | false | true | empirical q-stream roundtrip, but restored scales differ, no score claim | prove output parity or classify scorer delta, then exact CUDA only after lane claim |
| PR101 split-Brotli repack on PR106 | `experiments/results/pr101_repack_pr106_20260507T152608Z_claude/manifest.json` | `stack_candidate_requires_runtime_adapter` | true | false | prediction/rate-only manifest; PR106 runtime is expected to fail on PR101 decoder format | prove raw or output parity and integrate fail-closed adapter before any dispatch |
| PR103 LC/AC source schema | `experiments/results/hnerv_pr103_lc_ac_schema_20260507_codex/manifest.json` | `stack_source_candidate` | true | false | empirical byte-identical AC re-encode and schema-gap analysis; source score semantics remain invalid | port bounded AC contract into a byte-different candidate and close replay fidelity before substitute use |

## Evidence Semantics

All rows have `score_claim=false`, `dispatch_attempted=false`, and
`ready_for_exact_eval_dispatch=false`. Byte deltas and predicted rate terms are
planning signals only; they are not promotion, ranking, retirement, or paper
score evidence.

The PR103 source archive remains blocked as a substitute/frontier score row by
`replay_fidelity:public_leaderboard_score_mismatch`. Its useful value is the
LC/AC schema contract and stream-gap map, not its public score claim.

## Blockers

- PR101 f32 schema: runtime adapter, runtime-tree parity manifest,
  inflate-output parity, strict compliance JSON, lane claim, exact CUDA auth
  eval.
- PR101 fp16 scale: runtime adapter, output/scorer delta classification, exact
  CUDA auth eval.
- PR101 split-Brotli repack on PR106: raw or output parity proof, fail-closed
  runtime adapter, strict compliance JSON, lane claim, exact CUDA auth eval.
- PR103 LC/AC: replay fidelity mismatch, byte-different candidate archive,
  old/new archive SHA pair, runtime adapter, strict compliance JSON, lane claim,
  exact CUDA auth eval.
