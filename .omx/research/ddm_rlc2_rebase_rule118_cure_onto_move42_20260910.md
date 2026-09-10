# ddm_rlc2 — STOP at the pre-fire seal contract gap

**BLOCKED before rebase materialization, as expressly required by the charter's Read first clause.**
The exact pointer did not move. No rlc2 candidate, real encode, raw-identity proof or seal exists.
This is a contract refusal, not a result against the counted-rider mechanism.

`research_only=true; score_claim=false; verdict_scope=INSTANCE`

Tokens: `[no-triality] [p0-ledger-ok]`

## Exact STOP and executable evidence

The charter says: “if a pre-fire seal for a `t4_direct`-authority candidate is genuinely impossible
under the current contract, STOP at that exact point with the exact refusal text and the smallest
contract change”. This instruction is under Read first, before materialization. I stopped there;
an independent read-only agent agreed with both the implementation finding and this scope.

`ddm_rlc2_20260910/REFUSALS.json` retains four executed controls on the real software, with the
existing move42 tree used only as a read-only input control, never labeled an rlc2 candidate:

| Control | Actual refusal |
|---|---|
| `make_candidate_seal.py` without either timing flag | `make_candidate_seal.py: error: one of the arguments --decode-wall-clock --inherit-decode-wall-clock is required` (rc 2) |
| `validate_decode_wall_clock(None, ...)` | `decode_wall_clock: leg must be an object` |
| `build_seal`, original move42 smoke/bar and absent timing | `decode wall-clock refused: decode_wall_clock: leg must be an object` (`SealContractError`) |
| `build_t4_direct_leg` for the absent first-fire receipt | `decode_wall_clock: receipt missing: /Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/MODAL_REMOTE_RESULT.json` (`SealContractError`) |

These are `[macOS-CPU advisory; contract-only, scorer-free]` checks. The builder control reached the
timing gate after validating the real source smoke. No source smoke was rebound to a nonexistent
candidate, and no control seal was written. The original move42 bar is historical relative to
move40; this control does not assert that move42's old seal is a fresh fire authorization today.

Verified at source: `tools/make_candidate_seal.py:98-105` requires timing; its `:267-271` revalidation
also requires timing. `src/tac/candidate_seal.py:1028-1035` unconditionally validates timing before
building a seal; `:1439-1443` refuses missing timing at consumption.
`src/tac/decode_wall_clock.py:310-311` requires a completed successful T4 result and `:438` requires
receiver identity for inheritance. `tools/fire_modal_auth_eval.py:746-748` requires the timing leg
again before a sealed fire. These exact source files are hash-bound in `SOURCE_PINS.json`.

The current validator has measured, inherited and completed `t4_direct` paths, with no approved
pending-measurement state. The legacy unsealed fire interface is present, but using it cannot
satisfy this charter's explicit seal-before-fire/STOP instruction. The existing public-smoke
custody waiver does not waive timing. No waiver or alternate fire was attempted.

## What was measured and what was not

Fresh on-disk hashes and the canonical pointer agree:

| Existing input, exact serialized bytes | Bytes | SHA-256 |
|---|---:|---|
| move42 archive | 180,238 | `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f` |
| rlc1 archive on move40 | 180,173 | `8c2eaefa944ca8acba3db80829a5cd7a3d06a17b1fdb7ed5f39ce5a6fd9a124d` |

Move42 normalized receiver: `6726fd77a7c4fa80b70ce37accb30595c1004cdd91b9ef990bf9c12f21c296bf`.
Rlc1 normalized receiver: `b06e59a67b60f577eda2038353a9905550967a414e546e87162a33d9d60d1e2d`.
They differ, independently confirming the inheritance exclusion. Full seal-runtime digests are
`8c4b4fcebb4409e6403e309efc37bbdf21dde7dfcf4d1117837c7e96bfb67033` and
`191086a8868aad5470fee491dcec124d113cda34f10992515138066db737fae5`, respectively.
These are the named `tac.candidate_seal.measure_runtime_digest` definition, not T4 upload digests.

**Rlc2 candidate bytes/SHA: NOT MATERIALIZED. Raw identity: NOT MEASURED. Predicted candidate S:
NOT ESTABLISHED. Seal: NONE.** The candidate tree was absent at the recorded check. No 49-row
manifest was generated, no public smoke was rebound, and no new-tree literal census was claimed.

Conditional arithmetic only: if the real move42 re-encode eventually saves exactly 60 B and full
raw identity holds, then 180,178 B gives
`25*180178/37545489 + 100*0.00010637 + sqrt(10*4.66e-6) = 0.137436553721997`.
The conditional delta is `-0.00003995153718733028`, which would clear both the move42 score and
the source seal's `-2e-5` admission threshold, but not sub-0.12. No byte or distortion transfer was
measured for rlc2. The actual candidate must still be twin-priced and proved on all 600 pairs.
`CONDITIONAL_ARITHMETIC.json` keeps the assumptions beside the calculation and measured fields null.

## Smallest contract change for its owners to decide

`ddm_rlc2_20260910/PROPOSED_CONTRACT.md` is a concrete, unadopted two-stage lifecycle proposal:
freeze all existing identity, manifest, smoke, retention, falsifier and admission requirements in
a separately typed pre-fire intent; let only MAIN's expressly authorized first measurement consume
it; then require the fire's actual receipt to pass the unchanged strict `t4_direct` validator and
issue a new fully validated completed seal. The intent never claims timing clearance or promotion
eligibility. Failure, timeout, drift and time over 1,260 seconds prevent completion.

The change must be decided by MAIN and the second family, then committed and hash-frozen before
measurement. This arm has not amended the rule, implemented a pending mode, or granted clearance.
Independent same-family source review is corroboration, not the required second-family decision.

## RECALL EVIDENCE

Read the full charter/common contract, PROGRAM, operating handoff and live board; the governing
NO-FAKE, retention, timing, checkpoint and serializer clauses; rlc1 memo, final handoff, public twins,
manifest refresh and pr9 counted-config accounting; pr11's direct-T4 contract; the move42 packet
and seal. Read all four charter-named laws from the project memory source, plus the linked law
`validator_contract_no_producer_can_satisfy_is_a_forever_refusal_test_the_producer_on_the_pass_path_20260910`.
The charter and live board supersede the common contract's obsolete frontier paragraph.

Own content queries and complete outputs are in `RECALL_SEARCHES.json` and `recall_*.txt`:

- Research memos/receipts: `t4_direct|seal.before.fire|pre.fire.seal|receiver.change.*timing`.
- Canonical research index, DAG/FEEDs, docs, SPEC/design/charter files and canonical task ledger:
  `t4_direct|decode.wall.clock|receiver.*timing`.
- Lane registry: `rlc2|rlc1|rp1_round2`; no rlc2 registry hit was found in that query, while the live
  board expressly assigns the arm. No compute lane was claimed or consumed by this contract audit.
- Canonical equations: fresh `tools/list_canonical_equations.py --json`, 483 records, searched for
  `t4_direct|decode_wall_clock|decode_determinism_integer_arithmetic|wall.clock` (21 broad matches).
- Codex memory registry: `rlc2|rule118|move42`, no relevant hit in that bounded search.

Beyond the charter seeds, scg1's public-smoke contract proves its custody waiver cannot substitute
for timing. Dwc1's memo and task rows preserve prior receiver-specific timing failures; the linked
real-producer law requires a genuine positive producer/consumer path, not just synthetic passing
fixtures. These findings changed the proposed cure: a separate first-measurement lifecycle state
must preserve every non-timing gate and keep the completed timing validator unchanged. The
equation/DAG matches cover other receivers and general runtime budgeting; I did not find an
approved first-fire timing exception in the searched surfaces. No equation proves one.

## Boundaries and custody

No Modal, timing window, decoder, full-n600 scorer, GT conversion, training or paid dispatch ran.
No payload was materialized or discarded; no raw was copied, moved, deleted or hardlinked.
No bulk output or storage-routing exception was needed. Future decode still requires the charter's
SSD preflight, 8 GiB cap, resumability and detached launcher with done receipt; the raw-identity
certificate must precede any deduplication. Local checks confer neither T4 timing nor contest-CPU,
contest-CUDA score, cross-host identity, or rule-118 clearance on a new tree.

No `upstream/`, PR tree, sealed tree, receiver, archive, pointer, shared staged index, prohibited
common-contract file or existing receipt was edited. Only this arm's new report/evidence files and
canonical checkpoint/task appends are owned outputs. No `.py` file changed, so two Python review
passes and a Python review override are inapplicable. The serializer outcome is in
`ddm_rlc2_20260910/LANDING_STATUS.json`; a fallback commit must not be described as landed on main.

Triality: apparatus-only `[no-triality]`; no new equation, trainer lever or score anchor. Sensitivity,
Pareto/allocator and posterior-score hooks are inapplicable. The dispatch consumer is MAIN's
decision record and subsequent retained candidate. The four executable refusals disambiguate the
current lifecycle from the proposed one. Checkpoint owner is `ddm_rlc2`; the rebase remains blocked.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN and the second family; consumer store:
  `.omx/research/ddm_rlc2_20260910/CONTRACT_DECISION_FIRE_ORDER.json`; fire trigger: harvest this
  STOP packet. Decide the unadopted pre-fire lifecycle proposal. If approved, freeze its implemented
  contract before resuming the original rebase/proof/seal chain and MAIN's own cold T4 fire;
  revalidate the pointer and exact candidate at every dependent boundary. No local timing rerun.

`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.

## LIVE-HYPOTHESES

- The lossless counted-rider cure may retain roughly its ancestor's 60 B benefit on move42 because
  the same real coder and geometry mechanism can be applied to the updated field. The real twin
  encode and full raw comparison remain unperformed, so the ancestor saving is not banked here.
- A separately typed first-measurement intent may remove the seal/receipt cycle without weakening
  completed timing proof, because all non-timing candidate facts can be frozen before the fire.
  It remains subject to the second family's decision and real producer/consumer verification.

## DEAD-ENDS

- Current pre-fire `t4_direct` sealing is closed in this contract state: it requires a completed
  receipt that the first fire has not produced.
- Inheriting move42 timing for the cure is closed by freshly verified receiver-digest inequality.
- A sixth local calibration attempt is closed for this arm by the explicit suspension.
- Legacy unsealed fire or a smoke waiver is closed as an answer to this charter: neither satisfies
  its required timing seal, and the charter expressly forbids inventing a waiver.
- Reusing rlc1's bytes, raw proof or score projection as a measured move42 rebase is closed: those
  artifacts bind another archive. No rlc2 byte claim can precede real encode and full parse-back.
