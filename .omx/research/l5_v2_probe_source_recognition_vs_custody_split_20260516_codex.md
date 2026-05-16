# L5 v2 Probe Source Recognition Vs Custody Split

- date: `2026-05-16`
- agent: `codex`
- trigger: readonly L5-v2 adversarial review P1 finding
- score_claim: `false`
- promotion_eligible: `false`

## Finding

`accepted_for_observation` previously meant only that a source JSON was readable and recognizable as a candidate/axis row. That overstated partial custody rows, because a recognized TT5L CUDA evidence summary with missing command, devices, log, and inflated-output manifest could still appear accepted.

## Fix

Source records now distinguish:

- `recognized_for_observation`: candidate and exact axis are recognized.
- `custody_valid_for_observation`: full exact-eval custody checks pass for that row.
- `accepted_for_observation`: recognized and custody-valid.
- `custody_blockers`: exact missing custody fields for partial rows.

The grouped observation can still choose the richest available axis evidence, but the source ledger no longer masks partial custody as accepted evidence.

## Current TT5L Source Split

- `.omx/research/time_traveler_recovered_tt5l_25ep_exact_cuda_evidence_row_20260515_codex.json`: recognized but not custody-valid.
- `.omx/research/time_traveler_recovered_tt5l_25ep_exact_cuda_result_review_20260515_codex.json`: recognized and custody-valid.
- `experiments/results/modal_auth_eval/time_traveler_recovered_tt5l_25ep_exact_cuda_20260514T105300Z/contest_auth_eval.json`: recognized and custody-valid.

The TT5L architecture gate remains fail-closed on the same real blockers: missing paired CPU axis, predicate, side-info consumption, and score-delta binding.
