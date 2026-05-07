# WR01 PR106x Lightning Plan Disposition - 2026-05-07

## Decision

- Raw plan: `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/lightning_exact_eval_plan.json`
- Disposition: keep raw JSON local/private.
- Tracked surface: this sanitized disposition ledger only.
- Reason: the raw plan is a dry-run provider execution plan, not score evidence. It contains operator-local command strings and Lightning account/environment surfaces that should not be committed directly.

## Sanitized Custody Summary

- generated_at_utc: `2026-05-07T00:44:27Z`
- lane_id: `wr01_apply_pr106x_half`
- run_id: `exact_eval_wr01_apply_pr106x_half_20260506`
- source family: PR106 belt-and-suspenders x-repack follow-on wavelet apply-transform candidate.
- candidate_archive_relpath: `experiments/results/hnerv_wavelet_apply_transform_pr106x_1_2_20260506_codex/hnerv_wavelet_apply_transform_candidate.zip`
- candidate_archive_size_bytes: `186222`
- candidate_archive_sha256: `d2208ffa41297c40ce5f0bdbbe4767a9831301e382522afd2f6acf455a6b1628`
- baseline_reference_json: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json`
- baseline_archive_size_bytes: `186231`
- baseline_archive_sha256: `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e`
- dry_run_only: `true`
- submit_attempted: `false`
- dispatch_attempted_by_this_plan: `false`
- score_claim: `false`

## Notes

- The raw plan was not covered by the existing research-state classifier when scanned directly; it was reported as `manual_review`.
- The raw candidate archive remains covered by existing archive ignore policy.
- This ledger intentionally omits command strings, local absolute paths, account identifiers, provider URLs, and predicted score bands.
- Any future exact-eval dispatch for this candidate still needs a fresh dispatch claim and normal contest custody gates.
