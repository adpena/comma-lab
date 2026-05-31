# Codex Findings: Public PR Audit Contract Migration

- UTC: 2026-05-31T04:06:13Z
- Lane: `lane_codex_contract_first_adversarial_review_20260531`

## Finding

`PublicSubmissionAuditResult.as_dict()` emitted a clean public-submission audit
record without the shared archive-bound candidate contract. That left public PR
archive custody, local smoke evidence, and exact-dispatch blockers in a
parallel readiness dialect.

## Fix

Public PR audit payloads now embed `tac_archive_bound_candidate_contract.v1`
fields. The contract treats downloaded archive ZIP custody as a byte-closed
candidate input, records public-frontier tags and anti-pattern penalties, and
keeps exact dispatch authority false until a shared receiver proof and exact
CPU/CUDA gate exist.

## Protection

`src/tac/tests/test_public_submission_pr_audit.py` now checks that:

- no-network self-test audit remains exact-false and blocked on missing smoke;
- local inflate-smoke audit can record archive custody and adapter readiness;
- receiver contract satisfaction remains false without the shared proof path;
- public PR audit rows route through the same contract blockers as other
  archive emitters.

## Remaining Gap

Queue consumers should continue dropping direct reads of public-audit
`ready_for_exact_eval_dispatch` in favor of
`archive_bound_candidate_contract.ready_for_exact_eval_dispatch` and contract
blockers.
