# ddm_hv3 — 18 finished arms drained into owned exits

Date: 2026-08-27  
Actor: `ddm_hv3`  
Authority: persisted-final, keeper, task-ledger, later-consumer, and custody audit; no scorer, training, archive mutation, Metal, Modal, or evaluator run

## Outcome

The live keeper denominator was exactly **18 FINISHED-unharvested arms**. All 18 had `rc=0` `.done`
receipts, an indexed persisted final whose on-disk SHA-256 matched the index, an extracted
`NEXT_IF_RESUMED` block, and a retained custody receipt. Every follow-on now exits
`CONSUMED-ALREADY`, `ROUTED`, `CLOSED-dead-with-reason`, or `QUEUED-W-FIRE-ORDER`; there is no
`UNKNOWN` residue.

There are **0 unowned FIRE-NOW frontier heads**. The charter's prediction of at least one live head
among the 13 arms outside MAIN's five named heads is therefore **falsified on this 18-arm cohort**.
The apparent exception was JF2, but its owning memo has a later append-only MAIN addendum: all four
ordered n600 advisory rows were already measured, the trained diagonal was closed, and the WJ1
consumer was explicitly mooted. The final-message capture predates that addendum.

No exact score moved. The effective frontier remains GB1 at
`S=0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]`.

**FIRE-NOW rank at handoff: NONE.**

## MAIN-settled cross-check

| Settled head | Cross-check disposition |
|---|---|
| `fb2` | Memo and ledger content are in HEAD. Route 1 is the live W96 aligned burn, route 2 is dead, and route 3 is the sealed RB1 queue. No fourth route was found. |
| `d3b` | The measured formulation is `REFUSED_AT_FORMULATION_SCOPE`. Its only surviving row is a dormant, materially different zero-side-information reopen with a byte proof trigger; no candidate is fireable now. |
| `bs3` | MAIN's stage-40 adjudication measures the born-small route `+3.455 S` against `0.0526 S` rate credit and closes route 2. The learned-screen dependency dies with it. |
| `bs4x` | Its storage floor is now physically affordable, but MAIN de-prioritized the unlock because the later stage-40 distortion result makes the route dead. Storage green is not scientific reactivation. |
| `sr3` | Its reclaim was consumed by W96B; seed 20260815 r8 is live and seed 20260816 remains ordered behind it. |

## Per-arm typed dispositions

| Arm | Verdict summary | Follow-on exits | Payload custody check |
|---|---|---|---|
| `bs4` | Stage 0 passed identity but storage stopped the solve before any scorer work. Later BS4Y/stage-40 evidence closes the born-small route on distortion. | Exact solve `CLOSED-dead-with-reason`; learned screen `CLOSED-dead-with-reason`. | PASS: retained Stage-0 receipt `bfd33e8d…e2a7` exists. |
| `bs4x` | The Stage-0 cure worked and selected geometry raised the storage floor to 60,449,654,528 B. Later stage-40 evidence, not storage, kills the route. | Stages 1–4 `CLOSED-dead-with-reason`; Stage 5 `CLOSED-dead-with-reason`. | PASS: selected-storage receipt SHA `6a53dda5…5749` exists. |
| `d3b` | Nine exact receiver rows leave the best formulation 207 B over the token bar and 360 B over GB1. | Current formulation `CLOSED-dead-with-reason`; materially different generic predictor `QUEUED-W-FIRE-ORDER-body<=64000-or-64080-integrated`. | PASS: `RESULT.json` SHA `ce901a9c…e396` plus retained candidates exist. |
| `fb2` | Complete GB1 route table was consumed by MAIN: two live owned routes remain after born-small closure. | Patches `CONSUMED-ALREADY`; W96 `ROUTED-to-MAIN-r8-and-seed16`; born-small `CLOSED-dead-with-reason`; renderer `ROUTED-to-ddm_or1_renderer_born_small`. | PASS: fallback/cleanup certificate and verified bundles remain retained. |
| `pf4x` | Bare-round population was adjudicated 30 to 0 without weakening; later main history has advanced the full preflight chain beyond r60 to r99. | Source/evidence landings `CONSUMED-ALREADY`; r60 `CONSUMED-ALREADY`. | PASS: evidence bundle SHA `a3d7bec7…7729` exists. |
| `w96b` | Exact aligned implementation landed; SR3 removed storage as the blocking resource and MAIN fired the first seed. | Reclaim `CONSUMED-ALREADY`; seed15 `ROUTED-to-live-r8`; seed16 and screens `ROUTED-to-MAIN-post-endpoint-order`. | PASS: build/storage receipt and sealed fire order exist. |
| `jf2` | Three terminal byte winners were real, but MAIN later measured all ordered advisory rows and closed the trained diagonal; WJ1 has no live consumer. | Scorer order `CONSUMED-ALREADY`; WJ1 `CLOSED-dead-with-reason`; Git landing `CONSUMED-ALREADY`. | PASS: terminal receipt SHA `0a1cc640…ba5` and all four scorer result stores exist. |
| `pc2` | Complete pose/carrier drain found no live current-body candidate; D3B later closed the conditional-factor route rate-negative. | Landing `CONSUMED-ALREADY`; post-D3B re-account `CONSUMED-ALREADY`. | PASS: verified bundle SHA `0e3e455e…57b4` exists. |
| `rb1` | Two born-small renderer bodies are byte-admissible and four configurations are sealed, but the Metal order remains W96 first. | Training `QUEUED-W-FIRE-ORDER-after-both-W96-seeds-and-fresh-Metal-scorer-claims`. | PASS: build receipt SHA `26152c85…28b2` and birth checkpoints exist. |
| `sr3` | Two terminal trees were losslessly compressed and reconstructed; reclaimed capacity was consumed by W96B. | W96 fire `CONSUMED-ALREADY/ROUTED-to-live-r8`. | PASS: both reclaim certificates and retained archives exist. |
| `w96a` | The original storage and objective blockers were cured by W96B plus SR3; the aligned run is now MAIN-owned. | Implementation/reclaim `CONSUMED-ALREADY`; seeds and screens `ROUTED-to-MAIN-r8-post-endpoint-order`. | PASS: blocker, OFF replay, and W96B receipts remain retained. |
| `bs3` | The 101,150 B body is real, but the later fresh-solve stage-40 row is about 66x underwater on distortion versus rate credit. | Exact solve and learned screen `CLOSED-dead-with-reason`; source landing `CONSUMED-ALREADY`. | PASS: `BODY_RESULT.json`, `FIRE_ORDER.json`, and stage-40 payloads exist. |
| `d3c` | All 24 peel orders lose; the best screen is 111,318 B over the bar and D3B's stronger reference form still loses. | Rank-1 reference successor `CLOSED-dead-with-reason-no-credible-all-costs-in-path-below-127292`. | PASS: 128-file manifest SHA `d8098367…9724` exists. |
| `hd1` | Serializer-retention and pin-consistency cures are present in later HEAD content; per-runtime protection remains the canonical use-time gate. | Two landings `CONSUMED-ALREADY`; mismatched runtimes `ROUTED-to-runtime-custodians-repin-or-retire-before-use`. | PASS: control receipt and both verified bundles exist. |
| `mg1` | No-MPS-decision population is strict-clean and the task is completed in the canonical ledger; later HEAD contains the cured surface. | Fallback landing `CONSUMED-ALREADY`. | PASS: intended bundle SHA `510f7fdc…920a` exists. |
| `or1` | Standalone orthogonal packets are dominated; only a byte-gated global grammar and the already-built smaller renderer survive. | Renderer `ROUTED-to-rb1-MAIN-Metal-queue`; global grammar `QUEUED-W-FIRE-ORDER-double-decode-exact<=47696B`. | PASS: payload manifest SHA `61fe5f95…fb56` exists. |
| `pf2x` | r57's process-census blocker was later consumed; current HEAD shows the same full chain progressed through r99. | r58 continuation `CONSUMED-ALREADY`. | PASS: r57 red receipt SHA `50ddd7a5…8676` exists. |
| `fc1x` | Fat-clone repair and strict guard were consumed into later HEAD commits; evidence memo and completed task row are present. | Ordered bundles `CONSUMED-ALREADY`. | PASS: all three thin bundles remain retained. |

## FIRE-NOW ranks for MAIN

None. The denominator is **0 / 18 arms** and **0 / 13 arms outside the five MAIN-settled heads**.
No trigger-met, unowned frontier action survived the later-consumer join, so there is no cost or
falsifier row to rank. This is the charter prediction's bounded falsifier, not a claim that the
campaign has no future work.

## Surviving owned queue

| Order | Disposition | Owner | Consumer store | Fire trigger |
|---:|---|---|---|---|
| 1 | `ROUTED` | MAIN W96 custodian | `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/` | Observe r8 endpoint; then execute seed 20260816 with the sealed watermark environment and run per-seed n60 screens. |
| 2 | `QUEUED-W-FIRE-ORDER` | MAIN renderer-training successor | `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/` | Both W96 seeds terminal, Metal free, and fresh distinct scorer/Metal claims; then fire the four sealed RB1 configs. |
| 3 | `QUEUED-W-FIRE-ORDER` | MAIN | `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/followons/fr0/` | A zero-side-information generic predictor or integrated framing derivation proves the exact 64,000/64,080 B body gate before any encode. |
| 4 | `QUEUED-W-FIRE-ORDER` | MAIN boundary-grammar successor | `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/global_region_grammar_reference/` | A counted, double-decode-exact global grammar is at most 47,696 B before integration or scoring. |

These are not FIRE-NOW heads: rows 1–2 are already MAIN-owned and resource-ordered; rows 3–4 have
unmet construction/byte triggers rather than runnable candidates.

## Storage fact at write time

Live `df -k` at `2026-08-27T16:33:25Z` reported:

- APDataStore: **109,376,569,344 B free**.
- VertigoDataTier: **8,964,227,072 B free**.

The AP number exceeds BS4X's 60,449,654,528 B storage floor and RB1's 60,380,026,816 B floor.
Neither becomes FIRE-NOW from capacity alone: BS4X is scientifically dead by the later route-2
adjudication, while RB1 remains explicitly ordered behind both W96 seeds and the Metal governor.

## Ledger and keeper receipts

Canonical ledger: `.omx/research/ddm_hv2_harvest_consumption_ledger_20260826.jsonl`, extended rather
than forked. This round appends 18 `ddm_hv3.harvest_consumption.v1` rows, one per indexed final:
**82 before, 100 after, 18/18 unique arms**, SHA-256
`b883f380de9e0aaef5ca9c2614417606adb54dcd4ceb30dae6de7343a9cb15e3`.

Keeper marking was performed only through `tools/codex_arm_queue.py mark --status landed`; the exact
18 command receipts are the final 18 rows of `.omx/state/codex_arm_queue.jsonl`: **18/18 landed,
18/18 unique names**, keeper SHA-256
`2d70aed8efbd0f2b51422511c4a4d0c7364bffda612ae8db6cf7894abd8c7c67` at the pre-hv3-self-transition
verification snapshot. Post-mark status reports
zero FINISHED-unharvested arms; only this hv3 executor remains as a died/resumable row until its
serializer landing is complete.

## RECALL EVIDENCE

Recall was not charter-only. It covered:

- governing surfaces: `PROGRAM.md`, identical `CLAUDE.md`/`AGENTS.md`, the operating handoff, common
  contract, live hot state, canonical frontier pointer, and the NO-FAKE body;
- queue truth: `codex_arm_queue.py status --limit 100`, all 18 `.done` receipts, all 18 indexed final
  captures, all 18 extracted plan rows, and the final-message SHA index;
- content queries across `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, design/SPEC
  and canonical task-status surfaces for every arm name plus `born-small`, `aligned`, `terminal
  diagonal`, `class pyramid`, `orthogonal`, `serializer`, `bare round`, and `MPS`;
- all 449 entries in the canonical equations JSON, queried for harvest/consumption,
  rate/factorization, MPS, and orthogonality terms; no equation supplied a contrary runnable
  candidate;
- later HEAD and consumer evidence, including the route-2 adjudication, JF2 MAIN addendum, current
  preflight-chain commits, apparatus-equivalent landings, and live storage.

Beyond the charter seeds, the JF2 append-only addendum changed a would-be FIRE-NOW row to two closed
exits; current HEAD changed PF2X/PF4X continuation rows to consumed; later equivalent apparatus
landings changed HD1/MG1/FC1X bundle orders to consumed; and live AP capacity changed BS4X from
storage-blocked to storage-affordable but scientifically dead. These facts are why the prediction
was falsified rather than satisfied by a stale final-message headline.

## Verification boundary

- Measured here: the 18-arm keeper denominator; receipt/index SHA joins; extracted-plan counts;
  retained receipt existence and hashes; later-consumer references; live filesystem capacity; and
  post-mark keeper state.
- Recalled and re-derived from retained prior measurements: JF2's later advisory closure, route-2's
  stage-40 distortion closure, and the exact frontier pointer. They are not new hv3 measurements.
- Not measured here: any new candidate bytes, distortion, scorer output, archive score, runtime, or
  frontier delta.
- No scorer, training, Metal, Modal, archive mutation, dispatch, payload move/delete, or `upstream/`
  write occurred. The unrelated dirty lane-claim file and untracked WD3 runner were preserved.
- Primary Git landing is **BLOCKED**. The mandated serializer passed its content/base guards but
  `git add` failed before staging with `unable to create temporary file: Operation not permitted`
  and `failed to insert into database`; the shared index remained empty and HEAD stayed
  `d833e37743c6`. Its initial verified fallback is commit `05c6299a8fa2`, bundle SHA
  `c427454c…4316`, under
  `/Volumes/APDataStore/pact/ddm_hv3_done_arm_consumption/receipts/commit_serializer_fallbacks/20260827T163740.373647Z-44203/`.
  Because this paragraph changes the memo after that snapshot, the initial fallback is superseded;
  a final post-note serializer rerun captures the exact two-file contents under the same durable
  fallback root. Its exact child path and hashes belong in the terminal handoff rather than in this
  self-referential memo. No main-history commit is claimed, and hv3 itself remains resumable.

## NEXT_IF_RESUMED

- `ROUTED` — owner `MAIN W96 custodian`; consumer store `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`; fire trigger r8 endpoint; execute the sealed seed-20260816 order and the per-seed n60 screen.
- `QUEUED-W-FIRE-ORDER` — owner `MAIN renderer-training successor`; consumer store `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/`; fire trigger both W96 seeds terminal plus free, freshly claimed Metal/scorer lanes; run the four sealed RB1 configurations.
- `QUEUED-W-FIRE-ORDER` — owner `MAIN`; consumer store `/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization/followons/fr0/`; fire trigger a zero-side-information derivation proves the 64,000/64,080 B body gate; run one retained identity encode only then.
- `QUEUED-W-FIRE-ORDER` — owner `MAIN boundary-grammar successor`; consumer store `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/global_region_grammar_reference/`; fire trigger a counted double-decode-exact payload at or below 47,696 B; integrate only then.

## LIVE-HYPOTHESES

- W96's exact aligned loss may still beat its OFF controls because the mechanism has not completed on this vehicle; the current r8 burn is the direct test and retains stage outputs.
- RB1's changed-object, jointly pose-trained renderer may escape the born-small carrier failure because it changes the renderer/object rather than applying a fresh carrier to a distortion-dead body.
- A gentle, generic zero-side-information Road/Lane calibration may recover D3B's narrow 208-body-byte deficit because the tested absolute mixer overwrote a strong prior rather than calibrating it.
- A global topology grammar may beat local row starts because it can amortize shared 2D/temporal structure, but it remains plausible only below the strict 47,696 B double-decode gate.

## DEAD-ENDS

- Reopening born-small route 2 from newly green storage is closed: the later random-n32 stage-40 row is about 66x underwater on distortion versus its entire rate credit.
- Running JF2's stale scorer or WJ1 orders is closed: MAIN already measured the four rows, closed the trained diagonal, and recorded that WJ1 has no live consumer.
- Treating apparatus bundles as unconsumed solely because their original commit hashes are not HEAD ancestors is closed: later HEAD commits contain the cured surfaces and landed memos, and the full preflight chain has moved past the named rounds.
- Firing D3C's reduced-context ladder is closed: its best row is 111,318 B over the subsystem bar, while the stronger receiver-closed D3B reference still loses.
- Counting a clean `.done` receipt as an unowned frontier action is closed on this cohort: all 18 have content-routed exits and zero unknown residue.

Own-vehicle frontier: **GB1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4 n600]; unchanged by this consumption-only arm.**
