# ddm_pr12 — adjudicate the first-fire intent contract for receivers that cannot inherit a timing leg (charter, MAIN 2026-09-10)

## The cycle your family's ruling created
ddm_pr11 ruled T4-DIRECT: a receiver that has never run on contest T4 takes its timing authority
from its OWN completed cold n600 public-entrypoint T4 decode (`mode: "t4_direct"`, landed verbatim
0524522f0), and for rlc1's cure "prefer its own t4_direct fire". But the seal contract (dwc1,
86961e487) requires a validated decode-wall-clock leg BEFORE a fire, and `t4_direct` requires the
completed receipt that only the fire produces. ddm_rlc2 (re-basing rlc1's counted-rider cure onto
pointer move 42) hit exactly this and STOPPED, as chartered, with an executed refusal
(`decode_wall_clock: leg must be an object`; `receipt missing: …/MODAL_REMOTE_RESULT.json`) and a
written proposal: `.omx/research/ddm_rlc2_20260910/PROPOSED_CONTRACT.md` (a separately typed
PRE-FIRE INTENT with all non-timing gates frozen; MAIN's authorized first-measurement consumer;
the unchanged `build_t4_direct_leg` validator after harvest; a NEW completed seal naming the intent
digest and the receipt; no timing clearance, no score claim, no promotion by the intent).
Local calibration (the alternative route) is SUSPENDED after five refused runs (pr11 memo, MAIN
records). MAIN did not implement anything; your family decides.

## Read first
- `.omx/research/ddm_rlc2_20260910/PROPOSED_CONTRACT.md`, `CONTRACT_DECISION_FIRE_ORDER.json`, the rlc2
  STOP memo `.omx/research/ddm_rlc2_rebase_rule118_cure_onto_move42_20260910.md` (executed refusal table).
- `.omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md` (your T4-DIRECT contract
  and the rlc1 authority answer), `src/tac/decode_wall_clock.py` (t4_direct + inheritance as landed),
  `src/tac/candidate_seal.py` (`validate_seal`, `require_decode_wall_clock`, `SEAL_SCHEMA`),
  `tools/make_candidate_seal.py`, `tools/fire_modal_auth_eval.py` (`--seal` path; single-flight; claims).
- The dwc1 memo `.omx/research/ddm_dwc1_decode_wall_clock_seal_leg_20260910.md` (why the fire guard exists:
  tc4's T4 timeout at 1,800 s wasted a paid row; move 41 measured 1,336.7 s).
- rlc1's evidence for the candidate class: `.omx/research/ddm_rlc1_20260910/QUIESCED_TIMING_RECORD.json`
  (g3/g4 829/831 s local, DIAGNOSTIC only; projected ≈ 1,030 s on T4 via the refused ratio) and the move 40
  receiver's actual T4 978–990 s; the cure adds ~4 % locally (counted geometry maps).

## Questions
1. Is the proposal's separation sound: a pre-fire INTENT (typed, versioned, all non-timing gates frozen,
   no seconds, `score_claim=false`) that the normal seal validator REFUSES, consumed only by an expressly
   authorized first-measurement path, followed by the unchanged `t4_direct` validation and a NEW completed
   seal? Name any way it weakens dwc1's guard or lets a timing claim leak into a score claim.
2. What bounded evidence must the intent carry so the first fire is not a blind bet against the 1,800 s
   budget (the guard's purpose)? Candidates: receiver-identical-except-maps lineage to a receiver with a
   completed T4 leg + the maps' measured local cost fraction; a local DIAGNOSTIC decode under the frozen v3
   sampler (non-authority) with a stated ceiling; a budget/timeout binding on the dispatch itself. WRITE
   THE EXACT FIELDS AND COMPARISONS; MAIN implements your text verbatim.
3. Exact schema/CLI surface: `candidate_prefire_intent.v1` fields; how `tools/make_candidate_seal.py`
   emits it (`--first-fire-intent`?), how `tools/fire_modal_auth_eval.py` consumes it (an explicit
   `--first-measurement` mode bound to the intent digest, lane, axis, budget), what the harvest must do
   (build `t4_direct` from the exact receipt; issue the completed seal naming the intent digest), and the
   refusals (identity/pointer drift, unknown state, missing authorization, warm decode, timeout).
4. Prospective-freeze discipline: the contract text is hashed and committed BEFORE rlc2 resumes; the
   positive control must come from a REAL producer (rlc2's candidate) and MAIN's real harvest — no
   handwritten-fixture-only proof (dwc1's gate-with-no-door lesson).
5. Verdict for rlc2 specifically: may it resume under this contract (after implementation), and is its
   conditional row (180,178 B, S 0.137436553721997 if raw identity holds) worth a paid first fire?

## OPTIMAL FORM
- Reference form: your pr10/pr11 reviews (per-claim evidence table; verdict_scope on every negative; typed
  follow-on dispositions; EXACT contract text for MAIN to implement verbatim). Adjudication, not a build:
  no code, no receipts edited, no reruns.
- Provenance pins: 0524522f0 (t4_direct), 86961e487 (dwc1 guard), 6a857a1ec (v3 sampler), the rlc2 memo
  and proposal shas (record them), pointer move 42 d2803c214.

## Prior negatives accounted (operator 2026-08-15)
- dwc1: a validator only fixtures could satisfy; tc4: a fire that timed out — the guard must keep both
  from recurring under the new state.
- pr10/pr11: rules fixed after data are defects — this contract is frozen before rlc2's first fire.
- mv2/mv3/rlc2: arms STOP on exact clauses — write the contract so the implementer has no ambiguity.

## Deliverable
`.omx/research/ddm_pr12_adjudicate_first_fire_intent_contract_20260910.md` via the serializer
(`REVIEW_GATE_OVERRIDE=1`; tags `[no-triality] [p0-ledger-ok]`; no co-author trailer): verdict in
{APPROVE (exact contract text), APPROVE-WITH-CHANGES (exact text), REFUSE (with the admissible alternative)},
the exact fields/comparisons/refusals, and the rlc2 answer. Checkpoint as `ddm_pr12`. Final message: verdict,
the three strongest findings, every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
