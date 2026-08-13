# CHARTER — ddm_hg1_named_followons (2026-08-13)

THE NAMED-$0-FOLLOW-ON BATCH (P0 DEF CON class per #870: a named $0 follow-on never run is worse than an unwired build). Three small, fully-specified items from the 08-13 arc. Scorer-free; no Modal; no training.

**OPERATOR DOCTRINE 2026-08-13 verbatim, BINDING:** "no naive or toy or generic basis ever" — fixes here are STRUCTURAL (the detector zeroes on the cure), never procedural patches.

## OPTIMAL FORM
Hygiene/build arm (OPTIMAL_FORM_NA: no mechanism raced; reference form = two-landing discipline — fix + regression guard per item). Every fix lands with a test that fails on the pre-fix behavior.

## ITEM 1 — po1 recover-idempotence fix
The po1 dispatcher's recover path appends DUPLICATE terminal rows to its ledger on re-entry (hv1 NEXT_IF_RESUMED row; observed duplicate terminal ledger rows in `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/` ledgers). Cure: idempotent terminal-write (check-before-append keyed on run_id + terminal status), regression test with a double-recover scenario. Source: `experiments/ddm_po1_t4_error_feedback_pose_compensation.py`.

## ITEM 2 — queue-tool OSError guard
`tools/codex_arm_queue.py` crashes with OSError "File name too long" when an inline charter STRING is passed where a file PATH is expected (`Path(prompt).stat()` class — hit live 08-13, killed a spawn). Cure: guard the stat with an explicit is-this-a-path check → typed refusal naming the file-path contract (charters must be files), never a raw OSError. Test: inline-string prompt → clean refusal message, rc≠0, no traceback.

## ITEM 3 — pz4r semantic-portion salvage: verify + route
pz4r's fold left a live hypothesis: its low-rate SEMANTIC portion is reusable (seg moved only 18 flips, 50,394→50,412; failure was pose-only — d_pose 0.000147465→0.631014228). VERIFY the claim from the retained records at `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/` (report: `.omx/research/ddm_pz4r_full_n600_eval_20260813.md`): confirm the seg field's near-identity is attributable to the semantic sections (not to accidental cancellation), name the exact reusable sections + shas + byte counts, and ROUTE the pinned row into the re1 consumer store (`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/` — append a typed candidate-component row, do not disturb re1's live work). If the claim does NOT verify, say so plainly — a refuted salvage is a finding.

## OUTPUT
One memo `.omx/research/ddm_hg1_named_followons_20260813.md` (per-item verdict + receipts), code + tests committed via `tools/subagent_commit_serializer.py` with post-edit working-tree shas; message ends `[no-triality] [p0-ledger-ok]`. Honesty labels on every number. End with NEXT_IF_RESUMED + LIVE-HYPOTHESES + DEAD-ENDS.
