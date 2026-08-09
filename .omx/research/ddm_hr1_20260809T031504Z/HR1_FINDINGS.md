# HR1 harvest routing — complete, zero silent remainder

HR1 extracted **221** action-bearing rows from the named 29-arm population. It preserved all 221
as provenance rows, marked **66** semantic aliases (leaving **155** canonical
adjudication rows), routed **221/221**, and left **0/221 unrouted**. After terminal closure and
alias collapse, **94** genuinely live or conditionally reopenable items were backfilled as 18
per-arm blocks into `.omx/state/codex_arm_queue.next_if_resumed.jsonl`; one additional live item
already has the exact canonical task row `tz1_adaptive_percell_869_joint_remeasure_20260804`.
No row is unowned, `MAIN to route`, or `your call`.

This was scorer-free apparatus work. No scorer, eval, Metal/MPS/CUDA, dispatch, launch, archive
build, promotion, upstream edit, or public-PR-intake edit occurred. The own-vehicle and contest
frontiers did not move.

## Denominators

- Arms named / inspected: **29 / 29**.
- Raw action-bearing rows extracted: **221**.
- Semantic aliases retained but not duplicated into a second plan: **66**.
- Canonical adjudication rows after alias collapse: **155**.
- Dispositions: `FIRED=0`, `FOLDED=63`,
  `QUEUED-WITH-FIRE-ORDER=59`,
  `DEFERRED=37`, `SUPERSEDED=41`,
  `ALREADY-DONE=21`.
- Live/conditional raw rows: **96**;
  live/conditional rows after alias/existing-task collapse: **94**.
- Provenance rows with owner + disposition + consumer + trigger: **221 / 221**.
- Routed: **221 / 221**. Unrouted: **0 / 221**.

## Per-arm census

| Arm | Extracted | Aliases | Fired | Folded | Queued | Deferred | Superseded | Already done | New costate items |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddm_aa1` | 13 | 3 | 0 | 1 | 4 | 1 | 5 | 2 | 4 |
| `ddm_cf2` | 10 | 1 | 0 | 6 | 4 | 0 | 0 | 0 | 4 |
| `ddm_cr1` | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 |
| `ddm_gc21` | 7 | 0 | 0 | 3 | 1 | 1 | 1 | 1 | 2 |
| `ddm_gdl1` | 6 | 2 | 0 | 4 | 0 | 0 | 0 | 2 | 0 |
| `ddm_hb2_hpac_pack_roundtrip` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `ddm_lx1` | 26 | 2 | 0 | 15 | 6 | 4 | 1 | 0 | 9 |
| `ddm_m1c1` | 8 | 0 | 0 | 0 | 0 | 7 | 1 | 0 | 7 |
| `ddm_m1r2` | 3 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 |
| `ddm_m1r3` | 3 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| `ddm_m1r4a_mechanics` | 7 | 6 | 0 | 0 | 0 | 0 | 6 | 1 | 0 |
| `ddm_m1r4b_science` | 7 | 6 | 0 | 0 | 0 | 0 | 6 | 1 | 0 |
| `ddm_m1r4c_arith` | 13 | 12 | 0 | 0 | 0 | 0 | 12 | 1 | 0 |
| `ddm_m1r5a` | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 8 |
| `ddm_m1r5b` | 7 | 2 | 0 | 2 | 5 | 0 | 0 | 0 | 5 |
| `ddm_m1r5c` | 16 | 6 | 0 | 6 | 10 | 0 | 0 | 0 | 10 |
| `ddm_ng1` | 8 | 2 | 0 | 2 | 1 | 4 | 1 | 0 | 5 |
| `ddm_oh1` | 43 | 20 | 0 | 20 | 3 | 17 | 0 | 3 | 20 |
| `ddm_pk1` | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| `ddm_pk2` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ddm_rr16` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `ddm_rr17` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| `ddm_rr18` | 4 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 2 |
| `ddm_rv2` | 6 | 1 | 0 | 1 | 3 | 1 | 1 | 0 | 4 |
| `ddm_tr2_trot_crosswalk` | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `ddm_tr2p1` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| `ddm_wc1` | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| `ddm_wc2` | 5 | 0 | 0 | 1 | 3 | 0 | 1 | 0 | 3 |
| `ddm_zc1` | 8 | 0 | 0 | 0 | 7 | 0 | 0 | 1 | 7 |

`ddm_pk2` is the bounded zero: HR1 did not find an explicit future action or fire trigger in its
persisted final, `PK2_FINDINGS.md`, or the named commit. Its n120 result remains a
FORMULATION-scoped negative; HR1 did not invent a row from the statement that n600/CUDA was not
globally killed.

## Generator defect and cure

Measured instance: all **29/29** byte-faithful final-message captures exist and are indexed, but
`next_if_resumed_blocks()` found **0 files with a block / 0 blocks**. The extractor already
accepted exact Markdown headings and `persist_final_message()` already invoked it. The loss was
upstream: `tac.subagent_contract.RETAINED_REASONING` existed in the standard contract, but
`tools/codex_arm_queue.py::keeper_source()` generated only “read the charter + common contract.”
The common contract did not contain the machine-readable handoff clause, so every one of these
29 completions could state remaining work in prose without ever emitting the parser's heading.

Cure:

- `RETAINED_REASONING` now requires exact `## NEXT_IF_RESUMED` syntax only when future work
  exists, one bullet per action with disposition, owner, consumer store, and fire trigger.
- It requires omission of the heading when no future work exists, preventing phantom plan rows.
- `keeper_source()` now injects that clause into both first-generation and relay prompts.
- The direct `python3 tools/codex_arm_queue.py ...` entrypoint bootstraps the repository `src/`
  path before importing the contract; the adversarial pass caught and fixed the initial
  venv-only import regression.

Executed controls:

- `python3 tools/codex_arm_queue.py status`: PASS with no `PYTHONPATH` assistance.
- `.venv/bin/python -m pytest -q src/tac/tests/test_codex_arm_queue.py src/tac/tests/test_subagent_contract.py`:
  **98 passed**.
- Positive control: old unheaded `Remaining work:` prose yields zero blocks; a completion obeying
  the newly generated exact heading yields one block and retains its owner.
- Negative control: prose merely mentioning the extractor while declaring completion yields zero
  blocks; the contract also says to omit the heading for an empty plan.

This changes what the detector sees after cure: the old-form control remains 0, while the
new-contract control is 1. It is therefore a producer-contract cure, not a detector that reports
the same state before and after.

## Store delivery

The append-only queue moved from **67 to 85 rows**. Exactly **18** `hr1-backfill` plan rows were
written, one for each arm with a unique live survivor, carrying **94** item bullets. The verified
reader `tools/costate_digest.py::section_arm_next_if_resumed` reports all 85 rows and all 18 HR1
arm names. Terminal-only arms and `ddm_pk2` received no phantom blocks.

## Highest-stakes routes

1. **M1R5 cure + reseal before any M1 fire.** The R5 mechanics/science/arithmetic rows name real
   route, controller/child, transactionality, memory, grammar, receipt-schema, terminal-selector,
   EMA, threshold-unit, and hot-state defects. They are now owned by the M1 cure-and-seal roles,
   fire before any M1 execution, and remain gated on a fresh 3/3 clean seal.
2. **CR1 edge-conditioned receiver closure.** The byte-only probe reduced the same exact-decode
   support from 575,095 B to 464,557 B (−110,538 B), but made no receiver/scorer claim. The route
   fires only when the exact label/annulus residual object can prove decode equality plus receiver
   parse-back; owner is the #984 edge-stream receiver role.
3. **Current-object int8 post-hoc check.** OH1's high-stakes historical signal is folded into the
   existing canonical task `mh1_orphan_int8_posthoc_lowers_dseg_and_bytes`. The old witness-object
   number is not transferred to TR1. The named owner may reopen it only as a receiver-closed,
   identical-accounting comparison on the selected current object.

## No-signal-loss custody

Every source row carries arm, file, line/heading, current file SHA-256, and source commit. The
three independent censuses reconciled 79 + 111 + 31 rows. The required M1R5C and WC2 findings are
tracked in rescue commit `b08fd86f87`; all 34 unique primary source files in `HR1_ROUTING.jsonl`
are checked again in `RECEIPT.json`. Remaining population-associated dirty artifacts are not
silently treated as landed: the receipt classifies each as indexed capture, protected input,
sacred live-run evidence, post-arm benchmark evidence, or cleanup-blocked residue, with bytes and
SHA/tree hash. HR1 deleted or moved none of them.

## RECALL EVIDENCE

Queries run across the full corpus included:

- `29 landed arms follow-on routing orphan NEXT_IF_RESUMED`
- `M1R5 launch blocking fire route terminal selector provenance`
- `OH1 ZC1 orphan signal consumption plan fire order`
- `codex arm queue retained reasoning NEXT IF RESUMED extractor consumer`

Scopes searched: `.omx/research/` content and receipts, canonical research indexes, the long
`sub015_DAG_*` FEED surfaces, design/spec docs, `.omx/state/canonical_task_status.jsonl`,
`.omx/state/probe_outcomes.jsonl`, both arm-queue stores, and the canonical-equations registry
(429 entries at search time).

Beyond the charter seeds, HR1 found the NP1 implementation receipt at
`.omx/research/ddm_np1_20260805/NP1_RECEIPT.md`, the actual costate reader at
`tools/costate_digest.py::section_arm_next_if_resumed`, existing canonical-task consumers for the
OH1 split-bank/563/int8/secant/chroma/floor rows and #824/#869, and no post-R5 M1 repair in the
bounded current git/task/research scope. That changed the plan: existing task rows were reused,
review-round aliases were collapsed into the newest defects, missing live rows went to the
costate-readable queue, and the keeper producer—not the already-working parser—was repaired.

## Frontier honesty

No score was measured and no pointer was promoted. Own-vehicle frontier remains
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved.
