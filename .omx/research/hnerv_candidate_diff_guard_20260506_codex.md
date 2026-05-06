# HNeRV Candidate Diff Guard - 2026-05-06

This is a byte-forensics/no-op-control ledger, not a score claim.

- module: `src/tac/archive_byte_profile.py`
- function: `build_candidate_diff_manifest`
- hidden-gem key: `hnerv_payload_scorecard_followups`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence_grade: `byte_profile_only`

## Contract

The candidate diff manifest profiles source and candidate ZIP archives without
extracting them to disk or loading scorer/runtime code. It records archive
bytes, archive SHA-256s, rate terms, member names, member payload SHA-256s, and
changed-member records.

It classifies:

- `byte_identical_archive_noop`
- `payload_identical_container_reemit_noop`
- `non_noop_payload_changed`

Both no-op classes add `candidate_is_noop` to dispatch blockers. A non-noop
payload diff still remains blocked by:

- `candidate_diff_manifest_is_byte_forensics_only`
- `requires_exact_cuda_auth_eval_on_candidate`

## Use

Every future HNeRV repack candidate should attach this diff manifest before
archive preflight or exact CUDA eval. Byte-identical archive copies and
payload-identical ZIP/container rewrites are local controls only, not frontier
evidence. Non-noop payload changes still require exact archive compliance,
runtime custody, and exact CUDA auth eval before score claims.
