The EMA law mismatch is cured and six configs are re-sealed, but the burn remains unfired pending MAIN’s scorer-owned resume check.

The typed DSL now makes constant decay canonical, permits warmup only as an explicit ablation, independently recomputes sealed/executed coefficients, and halts on mismatch. Catalog #412 is STRICT with zero live violations.

Evidence:

- [Complete handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_wc3_qbr1_ema_law_cure_20260903.md)
- [Sealed MAIN fire order](/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/SEALED_MAIN_FIRE_ORDER.json)
- [Build receipt](/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/BUILD_RECEIPT.json)
- Exact cured commit: `106d0dd0a094dd4c289eba69c8d2c5124e13eb02`
- Tests: 26 WC3, 61 QBT/WC3, 333 broad relevant, and 204 lane-gate tests passed.
- Developer preflight passed 25/27 gates. Full preflight is environment-blocked because the sandbox denies real process inspection.
- No scorer, Metal, Modal, contest evaluation, or live-burn-directory write occurred.
- Git writes were denied, so the implementation and memo are retained as verified bundles—not falsely reported as landed on main.

Own-vehicle frontier: UNMOVED. Current effective frontier remains **afr1, S 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600]**.

## NEXT_IF_RESUMED

- `BUNDLE-READY-MAIN-MUST-LAND` — owner: MAIN; consumer store: shared Git repository and `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/`; fire trigger: safely import commits `106d0dd0a094...` and `6e06cd329856...`, preserving unrelated index/worktree changes, then verify hashes.
- `BLOCKED-ENVIRONMENT-CAPABILITY` — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/preflight_all/`; fire trigger: an environment permitting real `ps -axo pid=,command=`, then run full preflight without mocks or skips.
- `SEALED-BLOCKED-ON-MAIN-SCORER-LANE` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/resume_smoke`; fire trigger: acquire a unique scorer claim and execute `bounded_resume_smoke.argv`, requiring cursor/live/EMA/archive equality.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wc2_qbr1_bug_wallclock_realization_audit/metal_stage_profile/`; fire trigger: the first authorized cured cell plus unique scorer and Metal claims.
- `SEALED-AWAITING-MAIN-LIVE-CLAIMS` — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs`; fire trigger: resume PASS, source and storage revalidation, and unique scorer/Metal claims; execute all six retained launcher arrays and then `adjudication_argv`.

## LIVE-HYPOTHESES

- The real B=16 resume smoke should pass all four identity checks because construction and restoration now share the same typed law.
- Removing the native proxy term may improve the realized endpoint in at least two paired seeds; no treatment result exists yet.
- Frozen scorers and realization likely dominate cell time, but synchronized Metal profiling remains necessary.

## DEAD-ENDS

- The old literal-`warmup=True` cell cannot represent or validate the sealed constant-decay intervention.
- WC2’s one-site patch is superseded by the typed class cure and STRICT guard.
- The resume smoke is not scorer-free or n=1; source inspection shows real scorers and B=16.
- Mocking process inspection to manufacture a green full preflight is rejected.
- Main HEAD is not the cured sealed source until MAIN lands and verifies the retained fallback commit.