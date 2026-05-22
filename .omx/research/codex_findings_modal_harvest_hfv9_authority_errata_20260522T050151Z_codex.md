# Codex Findings Errata: Modal Harvest HFV9 Authority

timestamp_utc: 2026-05-22T05:01:51Z
lane_id: lane_codex_modal_recovery_config_synergy_audit_20260522
author: codex
verdict: CORRECTED_BY_ERRATA

## Scope

Errata for
`codex_findings_modal_harvest_recovery_20260522T045109Z_codex.md` and earlier
HFV9 notes that described exact eval as missing.

## Correction

HFV9 exact auth eval now exists on both axes for archive
`9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac`
(`178553` bytes):

- `[contest-CPU]` / `contest_cpu`:
  `score_recomputed_from_components=0.32067828057415293`
- `[contest-CUDA]` / `contest_cuda`:
  `score_recomputed_from_components=0.33713201858942626`

The recovered CPU payload can carry `score_claim=true` and
`score_claim_valid=true` because it is strict auth-axis evidence. That does not
make the archive promotional:

- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- promotable: `false`

Both scores are hard regressions relative to the current FEC6 frontier. Treat
HFV9 as a signed negative exact-eval label and axis/custody sanity pair, not as
a stacking or promotion candidate.

## Follow-Up

Future Modal harvest memos should print the full authority block whenever a
strict auth-axis result has `score_claim=true`, so readers can distinguish
"valid score evidence" from "promotion/rank authority."
