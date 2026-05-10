# T1 Modal guarded dispatch

Generated: `2026-05-10T08:45:00Z`

## Verdict

T1 Ballé end-to-end is dispatched as a bounded guard run, not a score claim.
This run exists to test the newly wired score-domain trainer and contest packet
compiler path after the A1 score-gradient exact-CUDA regression.

## Dispatch custody

- Lane id: `t1_balle_128k_endtoend`
- Instance/job id: `t1_balle_modal_guard_a3311268_20260510T0831Z`
- Modal app id: `ap-vPBYzG1bLZRRfMubZKWkQK`
- Modal function call id: `fc-01KR8GACB3NCW5TNG1E9YFPXHM`
- Claim status at dispatch: `active_dispatching`
- Metadata: `experiments/results/t1_balle_modal_guard_a3311268_20260510T0831Z/modal_metadata.json`
- Recover command: `.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_a3311268_20260510T0831Z`

## Guard configuration

- GPU: Modal `T4`
- Epochs: `50`
- Batch size: `8`
- Max target pairs: `64`
- Train timeout: `2h`
- Overall timeout: `24h`
- Expected samples for exact eval after packet build: `600`
- Score claim: `false`
- Promotion eligible: `false`

## Source custody

The Modal metadata recorded a clean mounted-code snapshot:

- Git HEAD: `a3311268f0a7a7547a32ae16f85ad3466e6de579`
- Dirty mounted code: `false`
- Worktree diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Index diff SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Promotion rule

Do not promote from this dispatch unless recovery produces an exact
`[contest-CUDA]` auth-eval artifact with `auth_eval_schema` blockers at zero,
paired contest-CPU reproduction is planned or recorded, the lane registry
promotion gate is satisfied, and the operator submission policy gate is clear.

If the guard regresses, classify the measured configuration only. The family
reactivation path is the exact artifact: proxy loss trace, archive bytes, rate
growth, component movement, packet compiler logs, and runtime closure.
