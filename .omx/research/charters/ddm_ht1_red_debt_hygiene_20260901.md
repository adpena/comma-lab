# ddm_ht1_red_debt_hygiene — burn down the pre-existing RED-test debt (#1282 + #1305) by BEHAVIOR-VS-FIXTURE adjudication, and land the 8 orphaned arm-authored .py files through the real review gate (tasks #1282 / #1305 / #1190-adjacent)

## MANDATE

Three hygiene debts sit on main and each one threatens an innocent commit or orphans landed
signal. (1) **#1282**: 11 `confound_gates` + 2 `contest_auth_eval` test failures, pre-existing
on HEAD (verified with edits stashed at filing time). (2) **#1305**: the gate4 live-count
drift test bounds active lanes at ≤4 while the registry carries 84 legacy NERV/SNeRV-era
lanes. (3) **The .py pile**: 8 arm-authored experiment files from landed 08-31/09-01 arms sit
untracked-or-modified because their commits require the review gate's two genuine passes —
leaving them is the #1190 orphan-signal genus. This arm adjudicates each RED at SOURCE and
lands the pile properly. The binding law is the **#1005 lesson: adjudicate behavior vs
fixture — NEVER fixture-edit a test to green without deciding which side (test or code) is
the correct contract.** A test that goes red on every honest state change is a defective
test; a test that catches a real regression is the gate working. Every disposition says
which, with evidence.

## SCOPE

1. **#1282 — the 13 REDs.** Reproduce each failure at HEAD first (stash nothing — main
   should show them cleanly; if any no longer reproduces, record PREMISE-STALE per the hy1
   precedent, the 6th stale-headline genus). Per failure: name the contract under test, name
   what drifted (code behavior vs test expectation vs fixture data), adjudicate WHICH is
   correct, fix at the correct side. Expected mix per the #1005/#1281 precedents: stale
   string-anchors, drifted refusal ORDER (behavior call needed), genuinely broken guards.
   A behavior change to make a test pass is FORBIDDEN unless the memo argues the behavior
   was the defect.
2. **#1305 — gate4 live-count drift.** The ≤4 bound vs 84 legacy lanes: decide whether
   (a) the bound is the live law and the 84 legacy lanes need honest disposition per the
   CLAUDE.md retirement discipline (archived/research_only tokens — likely a bulk metadata
   pass, NOT deletion), or (b) the bound itself is stale/wrong-object (e.g. it should count
   ACTIVE-era lanes only) and needs re-derivation with a one-line provenance. Do NOT just
   raise the number to 84 — that is the fixture-edit trap wearing a constant.
3. **The .py pile — review + commit through the real gate (2 genuine passes each):**
   untracked: `experiments/ddm_afc1_address_free_census.py` ·
   `experiments/ddm_br2_born_object_scorer_realization.py` ·
   `experiments/ddm_qx2_events_section_redesign.py` ·
   `experiments/ddm_qx3_receiver_closure.py` ·
   `experiments/ddm_qx4_decodable_conditioning_reprice.py` ·
   `experiments/ddm_wd3_sealed_3d9e021d07_runner.py` ·
   `experiments/tests/test_ddm_qx3_receiver_closure.py`; modified:
   `experiments/ddm_jg2_tail_reencode.py` · `experiments/ddm_pq2_compress_e2e.py`.
   These are LANDED-ARM artifacts: review passes are genuine reads (defects found get fixed
   or filed, not waved), `tools/review_tracker.py mark-file <f> --status reviewed` after
   each pass, then serializer commits. The two MODIFIED files: diff against HEAD first and
   state what the delta is and which arm authored it before committing.
4. **Typed disposition table** covering all 13 + 1 + 9 rows: {surface · what drifted ·
   adjudication (behavior/fixture/stale) · action taken · commit}.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. $0 — no scorer runs, no Modal, no training. The local SCORER LANE
  belongs to MAIN; nothing here needs it.
- Serializer commits w/ post-edit `--expected-content-sha256`; bundle-fallback on
  .git/objects denial (#1293). `.py` = 2 genuine review passes + review_tracker marks —
  NO REVIEW_GATE_OVERRIDE on .py, ever.
- The #1005 law binds every RED: adjudicate, don't fixture-edit. The #1281 precedent
  (string-anchor drift) and the hy1 precedent (premise-stale headline counts) are the
  reference dispositions.
- Do NOT delete or archive any lane registry ROW as part of #1305 without the memo stating
  the retirement-discipline token used (`archived`/`research_only`) and why — the registry
  is the record of everything considered, kills included.
- Tests touched must run GREEN at close (the touched subset, plus the modules they import);
  cite the exact pytest invocations + counts. No silent skips introduced.

## PRIOR NEGATIVE SIGNAL (bearing precedents this charter consumes)

- #1005: CPU-dispatcher REDs — refusal ORDER changed; the cure was a behavior CALL, not a
  fixture edit. The reference adjudication form.
- #1281: cold_root trainer-slice test — `_do_checkpoint` region drifted past string
  anchors; anchors were the defect.
- hy1 (#942): 15 stale expectations repaired STRUCTURALLY + 4 env-failures xfail'd with a
  named owner — the honest-subset pattern when an env can't run a test.
- #780: staleness-in-tests class — a test hard-coding a live ledger row status goes red on
  every honest state change; that CLASS is the enemy here, both directions.
- #1138: test_candidate_seal.py hardcoded a pointer baseline — same genus; check whether
  any of the 13 REDs is another instance.

## OPTIMAL FORM

- Family exemplar: hy1's 08-05 repair wave (15 structural repairs + honest xfails, 1023
  passing, no silent skips) — same bar; receipt
  `.omx/research/ddm_hy1_20260805/HY1_RECEIPT.md` sha
  `212e383c059f5452953600b69b705b09b0b69dcdc694091b6930cad48842d298`. Provenance pins:
  commit `945a9d4b54` (the #1281 cure — AST select-by-behavior + vacuity guard + executed
  mutation control, the reference behavior-vs-fixture disposition this arm imitates) + the
  review-gate law (CLAUDE.md "Review gate — non-negotiable") + task rows
  #1282 (`ddm_hy1_20260805/HY1_RECEIPT.md` is the owning-memo association for the
  stale-headline precedent) / #1305 / #1005 / #1281.
- SCOPE reductions legal: the 84-lane #1305 disposition may land as a bulk metadata pass
  with a spot-checked sample (n≥10, seeded random, never a prefix — m88) IF the memo states
  the rule applied uniformly. MECHANISM reductions FORBIDDEN: no xfail without a named
  owner; no bound raised without derivation; no .py committed without both passes.
- **PRIOR-LAW PREDICTION (falsifiable):** the staleness-in-tests genus predicts ≥8 of the
  13 REDs are stale-expectation/fixture defects (test side), ≤5 are live code defects.
  FALSIFIER: a majority turn out to be REAL code regressions — that inverts the burn-down
  into a bug wave and the memo must say so loudly (it changes what the gates have been
  missing).

## DELIVERABLE

`.omx/research/ddm_ht1_red_debt_hygiene_verdict_20260901.md` — the typed disposition table ·
per-fix commits (serializer) · the #1305 adjudication with its derivation-or-disposition ·
green pytest receipts for every touched surface · DEAD-ENDS + denominator (REDs examined /
cured / xfail'd-with-owner / premise-stale). End with the own-vehicle frontier line
(S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha cbb8d928…d405bf25 —
UNMOVED).
