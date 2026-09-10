# ddm_rlc3 — move-42 rebase stopped before production by Git custody

**PARTIAL / BLOCKED_GIT_OBJECT_WRITE_BEFORE_CANDIDATE_PRODUCTION.** No candidate or intent was produced. The serializer returned **17**, retained source commit `fa74b3399649185d06cd029d046b417a14a030c0` in its fallback bundle, and did not land it in shared HEAD. This is a filesystem/source-custody blocker, **not** a refusal from the frozen pre-fire contract and not a candidate negative. No exact score moved.

`research_only=true; score_claim=false` · `[no-triality] [p0-ledger-ok]`

## Actual results and missing deliverables

| Surface | Result / authority |
|---|---|
| SSD preflight | PASS, 54,366,003,200 B free at probe; arm store writable. Exact filesystem observation. |
| Move-42 archive | 180,238 B; SHA-256 `f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f`, independently rehashed. |
| Move-42 token field | 117,964,800 B; SHA-256 `d5248c775e49d8ac4299d2b0b60ffe5cffbeaafe348ef57986a169d3f118322b`, independently rehashed. |
| New source | `experiments/ddm_rlc3_move42_trace.py`; two recorded source-review passes, syntax compile and CLI help PASS; heavy behavior untested. |
| Source custody | Verified two-file fallback bundle; all committed blobs equal working files; shared index unchanged; source absent from HEAD. |
| Candidate bytes / SHA | NOT MEASURED; no candidate archive exists in this arm. |
| Full cold public raw identity | NOT RUN; move42 receipt names raw sha `1db04341d5ae9972d6296831a9e822d50a5d0f25ef77af378eae75b69e917159`, but RLC3 produced no raw. |
| New-tree literal census / manifest / smokes | NOT RUN; no new candidate tree. No copied CLEAR verdict. |
| Timing-risk receipt | NOT BUILT; retained historical risk arithmetic is not a new receipt. |
| Intent path / file SHA / bytes / canonical digest | NONE. |
| Normal `validate_seal` / normal `--seal` typed refusal | NOT EXERCISED: there is no real intent to submit to these controls. |

## Exact blocker and custody

The serializer's actual stderr is retained verbatim in `SOURCE_SERIALIZER_RETRY_STDERR.txt`:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_rlc3_20260910/SOURCE_REVIEWS.json: failed to insert into database
error: unable to index file '.omx/research/ddm_rlc3_20260910/SOURCE_REVIEWS.json'
fatal: adding files failed
```

Its typed outcome is `BUNDLE_READY_MAIN_MUST_LAND`, rc 17. The source bundle is
`/Users/adpena/Projects/pact/.omx/research/ddm_rlc3_20260910/source_serializer/20260910T165928.816039Z-60110/intended-commit.bundle`,
8291 B, SHA-256 `495606d53a1334e2636dffe6e7079ceca236c6bacc70be9439e629e09c42b0eb`.
Base HEAD: `69d12b5bbf54856d2088b4723ad73e6befaf7eac`.
Independent bundle verification recovered exactly `experiments/ddm_rlc3_move42_trace.py` and
`.omx/research/ddm_rlc3_20260910/SOURCE_REVIEWS.json`, checked both blobs, and confirmed no co-author trailer.
The verification object store is only a bundle inspection artifact; it is not a substitute current HEAD.

The common contract requires serializer commits. pr12's intent names the producer source commit checked
out before production, and `_pf_contract` checks ancestry to the frozen implementation and current HEAD.
I therefore stopped before producing from unlanded source. I did not claim a contract defect or mutate
`candidate_seal.py`, `decode_wall_clock.py`, the contract manifest, or the freeze receipt.
An earlier rc 8 was a self-identity collision: label `ddm_rlc3_source` disagreed with checkpoint owner
`ddm_rlc3`. Using the actual owner label resolved it without bypassing any review/collision gate.

## Source prepared for resume

The new trace is an explicit rebase of the landed move40 trace: actual move42 archive and field pins;
source controls must reconstruct the real shipped stream; all payloads retained; full corrector,
mixer and twin arithmetic-encoder state every 20 frames; seed 20260910; deterministic Torch and one
compute thread; owned native builds; 8 GiB store cap plus 16 GiB SSD reserve; fail closed on changed
sources or checkpoint bytes. It refuses production unless its exact source is committed at HEAD.
This prepares only the **source trajectory**, not the cured rider, raw proof, or intent. Those stages
remain owed. No new decoder code was applied to any sealed tree.

## Conditional arithmetic, not a candidate row

If a real raw-identical archive is 180,178 B, then
`net_dS = 25 * (180178 - 180238) / 37545489 = -3.995153718733028e-05` and
`S = 100 * 0.00010637 + sqrt(10 * 4.66e-6) + 25 * 180178 / 37545489 = 0.13743655372199698`.
These use move42's packet components and an **assumed** candidate size. The strict `net_dS < -2e-5`
bar requires at least 31 B saved: maximum integer size **180,207 B**. No fire at 180,208 B or more.

Frozen implementation: `a475431997d0e0c66563448524e44c2ca8ddb384`. Freeze receipt SHA-256:
`59158b8fce89e12c06eb4ae3e1cb71f347daa78a2cc5e8a061e1899be1150c2a`. All frozen implementation-manifest source hashes matched live files
at intake. pr12 memo SHA-256 `50d00e3956dc7ae5d3b15379d2ae6f8704119817b0b58f50aa30413b97eacadc`; rlc2 STOP memo SHA-256
`76ba13cf2ef8f91d731a8526d65f3bd96d7d3799c7002e9a99ed56e77fa24caf`. Full references are in `SOURCE_PINS.json`.

## RECALL EVIDENCE

Queries and retained outputs are under `.omx/research/ddm_rlc3_20260910/`.
Searched all `.omx/research/` by content for `counted.{0,40}(rider|geometry)|prefire_intent|receiver.delta`;
docs/SPEC surfaces for `rule.?118|prefire|receiver.*timing`; canonical task status, lane registry and
active claims for `rlc[123]|prefire|counted.rider`; `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` for
`counted.rider|rule.?118|prefire|receiver.*timing`. Generated the complete canonical equations export
with `tools/list_canonical_equations.py --json` and selected the related rows into `equations_matches.json`.
The memory-registry query `rlc3|move42|rebase|prefire` found no relevant hit in that scope.

Beyond the charter seeds, index R19 and the DAG's free/counted boundary keep video-selected geometry
in counted content; the integer-determinism equation does not prove a new receiver or its timing.
The task ledger still contains earlier RLC1 timing orders, while live hot state assigns this resumed
intent chain and suspends local timing; those old orders were not treated as authority to launch timing.
The actual move42 parse-back receipt identifies a different token-field hash, requiring a fresh source
trajectory. Inspection of `ddm_tc3_receiver.stable_libraries` found a storage helper bound to another
arm; the new trace calls the existing explicit-path `ddm_tc1_public_proof.build_libraries` in its own
store. No duplicate active RLC3 lane ID was found in the lane/claim files searched; hot state names RLC3.

## Boundaries and wire-in

No Modal, authorization, first-measurement fire, completion, timing window, scorer, GT decode, candidate
materialization, or long job was launched. No source or sealed tree, PR tree, `upstream/`, sj1 pass6,
contract code or protected files were edited. No payload was deleted or moved. The SSD write probe
is retained. The shared staged index remained byte-identical. MAIN alone may authorize and fire.

All six solver hooks are N/A for this blocked scorer-free custody step: no measured sensitivity,
Pareto constraint, bit-allocation update, deployable archive, empirical score posterior, or competing
model interpretation exists. The existing charter and `RESUME_FIRE_ORDER.json` own the next consumer;
this trace is research-only and is not an independent frontier lane or score claim.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `.omx/research/ddm_rlc3_20260910/SOURCE_BUNDLE_VERIFICATION.json`; fire trigger: Git-writable MAIN harvests the verified source bundle. Land the exact two files through the serializer and verify current-HEAD hashes and contract ancestry.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: ddm_rlc3 successor; consumer store: `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42`; fire trigger: source landing plus unchanged move42 and frozen contract. Execute the remaining real charter chain through committed intent, without dispatch; all original byte, identity, receiver, census, smoke, retention and timing-risk gates remain binding.

Frontier unchanged: `composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)`.

## LIVE-HYPOTHESES

- The counted cure may still save about 60 B on move42: the mechanism and source receiver are unchanged, but the changed field requires real recoding before that transfer is established.
- The cured receiver may finish T4 within 1,260 s: retained source T4 and refused g3/g4 local diagnostics motivate the 1,032.725 s risk estimate; it is not candidate timing authority.

## DEAD-ENDS

- Using the move40 rider as a move42 rebase is closed: move42 has a different causal token field, so the rider must be re-encoded.
- Treating a verified fallback bundle as current-HEAD custody is closed: this bundle is real but unlanded, and cannot satisfy the required source/intent ancestry.
- Repeating the wrong serializer label is closed: the exact checkpoint owner is `ddm_rlc3`; that fixed rc 8, leaving the independent rc 17 filesystem blocker.
- Local calibration retries, inherited timing for the changed receiver, and self-authorized T4 fire remain closed by the charter and frozen pr12 contract.
