# ddm_jcb1_scmdl_batched_suffix_pricing — execute rxc1's two queued rows: batch-of-proposals-per-suffix exact pricing + one bounded batched pass pricing all three XOV1 alternatives as joint G/M rows (owning memo: ddm_rxc1_gen3_gate1_verdict_20260901.md; ledger row: task #1374 in .omx state, resolved by that memo)

## MANDATE

Operator 20260902: *"Frontier score lowering?"* — routed finding: rxc1's GATE-1-PARTIAL
(`ddm_rxc1_gen3_gate1_verdict_20260901.md`): restartable exact coder exactness is PROVEN
(64/64 byte-identical suffix re-encodes, zero abs error on the n=32 AFR1 instance) but the
per-proposal outer loop is economically DEAD — best 450.5928640831262 s/proposal at stride
300 = 0.5019554561318141× the 897.675 s full re-encode. The memo's own adjudication: exact
incremental coding is structurally suffix-priced on this coder, so SCMDL's loop granularity
MUST be batch-of-proposals-per-suffix-re-encode, not per-proposal. Its two typed
NEXT_IF_RESUMED rows (QUEUED-BATCH-PRICE + QUEUED-STRUCTURAL-LOCALITY) are owner-assigned
by MAIN — this arm is that assignment. Build the batched pricer, then run ONE bounded
batched pass pricing the three XOV1 alternatives as separate joint G/M rows.

## SCOPE

1. Verify the fire trigger at source: the three XOV1 source hashes named in the rxc1
   receipts under `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` still match
   their live objects; refuse with a typed row if any drifted (do not re-derive around a
   drift).
2. Build the batched runner (`experiments/ddm_jcb1_batched_suffix_pricer.py`): ONE causal
   suffix traversal amortized across multiple candidate G/M alternatives — the memo's live
   hypothesis is that cost is dominated by deterministic suffix replay while the
   alternatives are known before the replay starts. Exact identity (byte-identical
   re-encode assertions, the proven 64/64 pattern) + retained payloads are the ADMISSION
   GATE for every priced row; a row without both is not a price.
3. Run the bounded batched pass: price (a) the born expert, (b) the generator-conditioned
   peel chain, (c) the 5,506-record directed support — each as a separate joint G/M rate
   row against the AFR1 body, with per-row wall-clock so the batching amortization is
   itself measured.
4. If batching does NOT amortize (falsifier below), build-and-screen the memo's named
   alternative instead: a counted bounded-reset causal-state form — noting it CHANGES the
   counted causal model and must earn its bytes; price the reset overhead honestly.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Do NOT write who currently holds it into a
  charter: an occupancy claim goes stale the moment that holder exits, and the arm has no way
  to learn it did (the #1210 stale-precondition genus — MEASURED 2026-08-29, when
  `ddm_bz2_bornsmall_capacity_ceiling` correctly refused to claim a capacity ceiling because
  a charter told it a since-released lane was taken). If this arm's work needs a scorer run,
  emit a typed fire order naming its trigger and let MAIN fire it; landing an honest partial
  plus a fire order is the CORRECT outcome, never a failure. This charter's pricing is
  RATE-ONLY (coder bytes + wall clock) — no scorer needed; distortion claims are FORBIDDEN
  here and route to the qxo1/realization lane.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`
  (the rxc1-named consumer store; per-candidate payloads retained, not only the winner's).
- DETACHED >30-MIN COMPUTE: any single compute step projected to exceed 30 minutes MUST
  launch outside the arm session with `nohup` + `disown`, a pidfile, crash-resumable stage
  checkpoints, and a durable done-receipt. The arm MONITORS that process; a successor or
  MAIN harvests the done-receipt. An in-session multi-hour compute loop is FORBIDDEN.
- CLOSED-FORM-FIRST (operator 2026-08-31 "All upstream can be closed form"): the scoring
  chain is frozen piecewise-analytic math with every non-analytic locus exactly known —
  derive/solve against the EXACT upstream operators (atlas:
  ddm_cfa1_closed_form_atlas_20260831.md) before any fit, surrogate, or sampled estimate;
  a fitted stage owes a one-line reason the closed form was not usable.
- File ownership: this arm owns `experiments/ddm_jcb1_*` + the ddm_jc1 store only; do not
  touch qxr1's files or stores.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- rxc1 GATE-1-PARTIAL itself (`ddm_rxc1_gen3_gate1_verdict_20260901.md`): per-proposal
  exact pricing DEAD at 450.59 s/proposal (7.51 min for ONE proposal); correlation
  authority NONE — Pearson/Spearman 1.0 were VACUOUS-BY-CONSTRUCTION and no headline may
  cite them as a gate.
- FCD2/FCD3 realized-transfer failure (fcd family, tasks #1319/#1320): the same token-GT
  selection principle behind the 5,506-record support failed realized transfer — the memo
  itself grades that alternative WEAK; price it anyway, expect little, say so plainly.
- The #1195/#1201 genus: canonical values are never overridden without reconciliation;
  every byte count in the pass reconciles against the retained AFR1 body constants
  (180,002 B archive; token subsystem constants per the hc1/mi1 receipts).

## OPTIMAL FORM

- Family exemplar: the restartable exact coder + its 64/64 byte-identity protocol is the
  reference — `ddm_rxc1_gen3_gate1_verdict_20260901.md` @ commit 7a523b5c03, receipts at
  `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; the batched form is a
  SCOPE change (loop granularity), not a mechanism reduction — the coder, contexts, and
  identity assertions are reused unmodified.
- SCOPE reductions declared per row (n=32 AFR1 instance carried forward from rxc1 is a
  declared SCOPE bound; no family claim beyond it). MECHANISM reductions FORBIDDEN.
- **PRIOR-LAW PREDICTION (falsifiable):** rxc1's live hypothesis predicts one suffix
  traversal amortizes across the three alternatives — total batched cost ≤ ~2× one
  single-proposal replay (~900 s) for all three prices. FALSIFIER: batched cost ≥ 3× a
  single replay ⇒ batching is dead on this coder, and the bounded-reset counted-model form
  becomes the only live SCMDL route — count it plainly.

## DELIVERABLE

`.omx/research/ddm_jcb1_scmdl_batched_suffix_pricing_20260902.md` — typed rows: (1) XOV1
hash-trigger verification; (2) batched-runner build + 2 review passes + identity-assertion
receipts; (3) three joint G/M price rows w/ per-row wall clock + the amortization
measurement vs the prediction; (4) route adjudication (batch lives / bounded-reset next /
SCMDL closes) + NEXT_IF_RESUMED + DEAD-ENDS. Commit via the serializer. End with the
own-vehicle frontier line.
