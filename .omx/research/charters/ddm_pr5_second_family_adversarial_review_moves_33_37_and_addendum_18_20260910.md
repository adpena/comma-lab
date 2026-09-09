# ddm_pr5 — SECOND-FAMILY adversarial review (gpt-5.6-sol, xhigh) of the day's five pointer moves (33–37) and of gs3 Addendum 18 + its correction: re-derive every S from components, re-check every projection-vs-exact residual, and hunt the confound class MAIN's own family just missed twice (attribution read as payload; unshipped field read as shipped) (charter, 2026-09-10)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-5.6-sol, xhigh — the second model family, by design: different families catch different things) · Spawned by MAIN 2026-09-10 (operator 2026-09-09: "Use Astra or sol and different effort levels as appropriate"; m29 per-landing pantheon review). Sources: the five pointer-move packets `.omx/research/*pointer_move_3[3-7]_20260909.md` (rc2, pc2, sj1 pass 4, cmp1, cmp2) with their Modal receipts under `experiments/results/modal_auth_eval_mirror/` and artifacts under `/Volumes/APDataStore/pact/ddm_*_t4_*_20260909/`; gs3 Addendum 18 + CORRECTION (`ae6a0b837`, `5e6f774be`); bnd1's audit (`a42e0d337`) which found the two misreadings; gb2's vacuous-bound closure (`4c803b565`); `.omx/state/canonical_frontier_pointer.json`. Axes: review; every number re-derived `[from receipts]`; `score_claim=false`.

## MANDATE
Five exact rows landed in one day, each projected before firing and each promoted on harvest by the same family (MAIN Fable + Opus arms). Within an hour of the last one, the same family wrote two false premises into a charter and the gestalt. A review by the same family has a correlated blind spot. Review as the OTHER family: (1) recompute each of the five S values from the receipt components (d_seg, d_pose, archive bytes → S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489) and confirm the pointer file; (2) for each move, compare the pre-fire projection to the exact row and classify the residual (optimistic / pessimistic / exact) with its cause; (3) read Addendum 18 §I–§IV against the primary memos it cites and list every claim that is an ATTRIBUTION, a PREFIX, a DIFFERENT-OBJECT, or a BOUND-DIRECTION misread (the four classes bnd1 exposed) — with the correct statement beside each; (4) read the five packets' "not claimed" sections and check nothing claimed elsewhere contradicts them; (5) verify PR #140 is 13 moves behind and that no public text carries an AI attribution.

## PRIOR-LAW PREDICTION (m38)
- All five S values reproduce to 1e-12; projection residuals: rc2 exact, pc2 +1.84e-6 (pose print), sj1 −1.09e-6 (pessimistic), cmp1 ~0, cmp2 ~0.
- Addendum 18 carries ≥ 2 further misreadings of the four classes beyond the two already corrected (prediction: the "36 KB flag mass" is also an attribution; "12.75× the corner" and "8.94×" are ratios measured on other objects).
- **FALSIFIER:** if every remaining claim in Addendum 18 survives at source, say so plainly — a clean review is a result. If any S does NOT reproduce from its components, that is P0: stop and report before anything else.

## SCOPE
Review only: no code, no measurement jobs, no pricing. Read receipts; write findings.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; no writes outside your memo and its evidence dir `.omx/research/ddm_pr5_20260910/`. Do not touch any live tree.
- Every finding cites file:line at source; every corrected number carries its derivation. No new claims without a receipt.
- The local SCORER LANE belongs to MAIN, always. Do NOT write who holds it into a charter (the #1210 stale-precondition genus, memo ddm_bz2_bornsmall_capacity_ceiling 2026-08-29).
- Serializer commits w/ post-edit `--expected-content-sha256`; if git object writes are refused, leave `landing.patch` + manifest and say so. Tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_pr5`.

## PRIOR NEGATIVE SIGNAL
- bnd1's four DEAD-ENDS are the four misreading classes; hunt them, do not repeat them.
- m106: stale headlines survive corrected bodies — check the packets' headline lines against their bodies.
- available-field-vs-authoritative-field: a number read from the wrong field is the #1 class.

## OPTIMAL FORM
- Reference form: the per-landing pantheon review (m29; `.omx/research/*pantheon*review*` latest) and bnd1's source audit (`a42e0d337`) as the reference for "verified at source". SCOPE reductions: none (all five moves, the whole addendum). MECHANISM reductions FORBIDDEN: no sampling of moves; no trusting a packet's own summary line.
- **PRIOR-LAW PREDICTION (falsifiable):** as above.

## DELIVERABLE
Memo `.omx/research/ddm_pr5_second_family_review_moves_33_37_20260910.md`: the five re-derivations, the residual table, the misreading table with corrections, PR #140 status. Commit via the serializer. End with the live frontier line.
