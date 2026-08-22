# DDM NR1 task-cell quotient prebuild — 2026-08-22

## Disposition

**EXECUTABLE PREBUILD MEASURED; RATE FALSIFIER SURVIVES; NOT RECEIVER-CLOSED OR SCORE-ADMITTED.** NR1 now
has a real four-surface fit/encode/decode implementation, strict packet receiver, retained n600 coder
race, and an actual DX2 semantic-renderer consumption smoke. The best retained row is **69,004 B** and
beats the 113,777 B token stream by **44,773 B**. It also clears the stricter 71,395 B token-only ceiling
by **2,391 B**. This is a rate result only: the best row changes 1,558,833 task tokens, the current
primary r9 endpoint is not frozen, no scorer ran, no full 3,662,409,600 B raw was materialized, and no
candidate is receiver-closed or score-admitted.

This arm remained scorer-free and CPU-only. It did not launch Modal or touch/read/lock the live r9 run
directory. It modified only its owned NR1 module, runner, test, and this memo; shared coder, EC2, RB1,
HT1, JO, and production receiver sources remained untouched. All three materialized rows and their coder
losers/repeats are retained below `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/`.

## Result

NR1 is a counted implicit task-cell quotient that replaces exact semantic-token reproduction with the
shortest receiver-closed description that preserves the frozen endpoint's evaluator cells. Its primary
teacher is the frozen r9 endpoint: SegNet logits, winning margins, argmax cells, and Pose6. Source GT and
older diagnostic products may challenge or stratify the teacher, but they may not silently replace it.

At the current dx2 realized distortion, the whole archive must be at most **137,986 B**. The current exact
archive is **180,368 B**, so a same-distortion candidate must cut **42,382 B**. The current semantic token
field alone is **113,777 B**. NR1 therefore has a falsifiable task: the complete active quotient description
must beat that lossless stream at matched realized distortion, and the whole archive must fit the dynamic
cap after every other actual section is charged.

The old 33.5 KB “model/context” number is retired. It is neither a budget nor a launch gate.

## Executable prebuild measurement

The implemented object is an 8x8 categorical task-cell quotient consumed at the existing semantic-token
renderer boundary. `QPARAM` is the learned tile dictionary; `QCTX` is the learned per-tile default;
`QPAIR` is the full n600 temporal/context choice stream; `QEVENT` is 8,192 low-margin live token
corrections selected from the secondary C1 field. The teacher field is retained as training-only and is
explicitly forbidden by the shipping allow-list. It is not the current primary endpoint.

| Row | QPARAM | QCTX | QPAIR | QEVENT | Complete packet | Token agreement | Token-only ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| K32 / E8192 | 239 B | 152 B | 52,124 B | 16,489 B | **69,004 B** | 0.986785609 | **−2,391 B** |
| K64 / E8192 | 327 B | 153 B | 62,900 B | 16,496 B | **79,876 B** | 0.987467312 | +8,481 B |
| K128 / E8192 | 492 B | 159 B | 74,531 B | 16,448 B | **91,630 B** | 0.988848165 | +20,235 B |

These are complete packet bytes, including the outer header and every per-section header; physical
attribution covers every byte exactly once. Each logical surface was raced through raw, zlib-9,
LZMA1-1MiB, and Brotli-q11, with all losers and deterministic repeats retained. Brotli-q11 won every
surface for all three rows.

K32 packet SHA-256 is
`a68765dc683fa8302b560ef3db0d4a1507eeeccc695322fb8b69f684ed6dab28`. Its deterministic single-stored-
member research ZIP is 69,104 B, SHA-256
`7989c13a4e5eda3baadf87c37350470b3a5032c575aa2c622e62ed091b69ffa7`. The full decoded token output and
independent repeat are each 117,964,800 B with identical SHA-256
`d416895a250ce79be7f485188d4f7dfd1690a269a250063c2f6bc5f48cf8b8d8`. Exact-once consumption is
`QPARAM=1, QCTX=1, QPAIR=1, QEVENT=1`. Pair-zero through the actual DX2 semantic weights changed 408,008
of 589,824 retained uint8 renderer-output values.

Holding every current non-token archive byte fixed, K32 mechanically projects an archive of
**135,595 B** (`180,368 - 113,777 + 69,004`), 2,391 B below the current-distortion cap. This projection
does not preserve current distortion: 1,558,833/117,964,800 tokens differ, so the primary-endpoint
matched-distortion test remains the decisive unresolved gate.

The retained result roots are:

- `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k32_e8192_v1/`
- `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k64_e8192_v1/`
- `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/vq8_k128_e8192_v1/`

The executable sources are `src/tac/optimization/nr1_taskcell_quotient.py` and
`experiments/ddm_nr1_taskcell_quotient_prebuild.py`; the adversarial suite is
`experiments/ddm_nr1_taskcell_quotient_prebuild_test.py`. The retained K32 producer hashes are
`66500b813eeafeaf264d57ecb47ef68360956ec1bdb040043456f3d6f101cbb6` (module) and
`44e8ac10d20ca9c6325d572ac44cd3be9b553409ef0c408d41074f5ef9d7847c` (runner).

## RECALL EVIDENCE

The required corpus was searched before fixing this specification. The load-bearing sources were:

- `PROGRAM.md`, `AGENTS.md`, `CLAUDE.md`, the task charter, and `_common_contract.md` for governance.
- `main_hot_state.md` for the live pointer, r9 custody boundary, and MAIN ownership.
- `ddm_tl1_teacher_ledger_20260822.md` for the teacher hierarchy, primary endpoint
  supervision, dynamic budget, and current dx2 byte ledger.
- `ddm_es1_end_state_characterization_20260821.md` for current stream interactions and the retirement of
  the old static envelope.
- `ddm_ig1_implicit_carriage_gestalt_20260821.md` plus the completed WS1 result for the conditional
  implicit/explicit routing law and the closure of a full explicit worldsheet.
- `ddm_ws0_worldsheet_grammar_price_20260821.md`,
  `ddm_ws1_optimal_worldsheet_grammar_20260821.md`, and the WS1 receipts for the measured explicit-family
  no-go.
- `ddm_r012_rate_representation_20260821.md` for the proof that the current body cannot reach sub-0.12 by
  distortion or small composable byte cuts.
- `ddm_jo2_solve_reseal_20260821.md`, `ddm_jo6_receiver_container_compat_20260822.md`, and
  `experiments/ddm_jo2_receiver_close.py` for real single-member, parse-back, corruption, deterministic
  repeat, and staged-shipping receiver discipline.
- `ddm_dc1_score_quotient_functional_contract_DAG_FEED_20260724.md`,
  `src/tac/optimization/ddm_score_quotient_functional_contract.py`, and the canonical-equation registry for
  the earlier structural quotient ABI and rate/score laws.

The full-corpus queries, beyond opening the named sources, were:

| Surface | Query |
|---|---|
| Research memos and receipts | content search for `task-cell`, `task cell`, `score quotient`, `quotient functional`, `worldsheet`, `lossless token`, `teacher logits`, `receiver-close`, `113777`, `42,382`, and `33.5` under `.omx/research/` |
| Canonical equations | `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for `quotient`, `rate`, `MDL`, `task RD`, `worldsheet`, `receiver`, and `sufficient statistic` |
| Research index and DAG | content search for `FEED-603`, `DC1`, `NR1`, `score quotient`, `task_rd`, and `worldsheet` over `.omx/research/CANONICAL_RESEARCH_INDEX*` and `.omx/research/sub015_DAG_*` |
| Design/spec/source surfaces | content search for `ScoreQuotient`, `event continuation`, `typed sections`, `inactive identity`, and `one consumer` under `docs/`, `.omx/research/`, and `src/tac/` |
| Task and live-state stores | content search for `#1187`, `1187`, `nr1`, and `quotient` in `main_hot_state.md`, `canonical_task_status.jsonl`, `harness_tasklist_bridge_20260803.jsonl`, and the extracted Codex arm queue |

The recall changed the plan in four concrete ways:

1. WS1 is no longer pending. Its optimal explicit full-worldsheet reference is closed at **918,904 B
   lossless** and **885,750 B at q2**. Only the hood specialist produced a local byte win. A full explicit
   geometry stream is therefore not an NR1 candidate.
2. The older DC1 quotient is a structural research fixture, not an implemented scientific vehicle. Its
   typed-section and inactive-identity ideas are reusable; its rank-one prospective family and constant-
   plane receiver fixture are not authority for NR1.
3. The current source contains an event-continuation engine, so optimal-form NR1 must use a verified
   continuation/control path rather than resurrecting a fixed stage schedule. Presence in source is not
   proof that it is wired to NR1.
4. Task `#1187` was not found in the bounded canonical task-status or harness-bridge stores searched here.
   Its ownership is therefore taken only from `main_hot_state.md` and the extracted Codex arm queue; this
   prebuild does not invent a second task row.
5. The common contract's embedded qo1 `0.7539807` advisory line is stale. The later live board, ES1, and
   the charter-pinned TL1 all identify dx2 as the current own-vehicle contest-CUDA frontier. This memo uses
   the live dx2 line and does not revive the borrowed-pointer classification.

Pinned inputs were verified before this specification was written:

| Input | SHA-256 |
|---|---|
| TL1 memo | `d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15` |
| ES1 memo | `789b00f237bac0a8d1bdb3f00ae0a3b83be7ab75edfea472baaf64dbf0f05e18` |
| IG1 memo | `8ec60069b33f2d19d9a39ea30c94acee66ac299d800b5e739f411a48aa42ce8b` |

## Authority and frozen teacher

The frozen r9 endpoint is the primary task teacher. Harvest is allowed only after r9 has emitted a terminal
endpoint receipt and MAIN has declared it frozen. Nothing in this prebuild reads, writes, locks, moves,
cleans, or snapshots the live run directory.

The teacher manifest must bind all of the following by path, bytes, SHA-256, source archive SHA, source
runtime/content hashes, sample order, tensor shape/dtype, and production command/config where applicable:

- frozen endpoint archive and exact decoded raw output;
- SegNet endpoint logits for the scored last frame of every pair;
- winner and runner-up class identities and their signed margins;
- endpoint argmax cells after the real resize/uint8/parse-back path;
- PoseNet official six-scalar output for every scored pair;
- the fresh endpoint dxi/carrier state used by the endpoint;
- C1, DM, and EC1 diagnostic sources, each labeled secondary and non-shipping.

The frozen endpoint wins conflicts by default. Source GT may be used only for explicit hard-positive or
collateral-negative treatment that is recorded as an ablation against the endpoint. C1 can contribute
only cells where it improves the frozen endpoint under the real receiver. DM contributes only its named
events. EC1 supplies both positive targets and the collateral cells that must not regress.

No prefix-only result is a verdict. Engineering smoke may use a small deterministic subset, but candidate
admission and the representation falsifier use the full 600-sample population.

## Quotient representation

### Object being encoded

NR1 encodes a quotient of the frozen evaluator cells, not a reconstruction of teacher tensors. Two source
states are equivalent for NR1 when the actual receiver maps them to the same scored SegNet argmax cells
and Pose6 behavior within the candidate's admitted realized-distortion cell. Dense RGB, dense logits, and
exact semantic token identity are deliberately outside this equivalence relation.

The active description has four counted logical streams:

| Stream | Meaning | Required consumer |
|---|---|---|
| `QPARAM` | Quantized parameters and deterministic-generator selections for the implicit task-cell map | task-cell renderer |
| `QCTX` | Counted entropy/context state required to decode the other quotient streams | quotient decoder |
| `QPAIR` | Per-pair low-dimensional latent/control sequence | task-cell renderer |
| `QEVENT` | Sparse admitted corrections for cells or pairs the implicit body cannot preserve | correction compositor |

These names define ownership, not four mandatory physical files. The actual coder may jointly code them
when joint coding wins. Every paid byte must still have exactly one declared consumer, and the retained
receipt must report both the complete coded payload and a deterministic attribution method. No semantic
credit is awarded to a substream that does not survive the final parse-back.

The task-cell renderer produces actual RGB frames through the legal generator/receiver path. Scorer
weights and teacher tensors are training-only; the shipping receiver does not invoke a scorer. Generic,
video-independent algorithms may live in `inflate.py`; every video-derived parameter, table, latent,
context, exception, or learned basis is counted in `archive.zip`.

### Objective and admission

The only authority objective is the exact score of the parsed candidate:

`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

Logit, margin, cell, Pose6, and collateral losses are proposal gradients, not score substitutes. Their
multipliers must be derived from the canonical score-marginal law and measured continuation behavior;
this memo does not guess fixed weights. Candidate selection is performed after quantization, actual coding,
fresh-process decode, and scorer evaluation on the parsed bytes.

Optimal form requires all of the following:

- quantization and the real receiver operator in the training/admission loop;
- actual coder bytes in the rate decision, not entropy estimates;
- event continuation or an equivalently verified active-set control, not a fixed copied curriculum;
- margin-weighted cell treatment plus EC1-style collateral negatives;
- a fresh terminal joint dxi/Pose finish for each admitted body, never a stale copied dxi;
- per-stage atomic checkpoints, deterministic seed/config, resume-from-disk, and preserved EMA state;
- retained payloads for every materialized candidate, including rejected candidates;
- full n600 selection before any family verdict or pointer claim.

Width-only scaling, a flat palette, photometric copying, post-hoc finishing KD, and a lossless token recode
are not acceptable NR1 mechanisms. A small implicit body must be born against the quotient objective and
allowed to allocate bits between `QPARAM`, `QPAIR`, and `QEVENT` according to realized score value.

### Falsifier

The representation prediction is falsified when, at matched or better realized distortion, the complete
actual-coded quotient description (`QPARAM + QCTX + QPAIR + QEVENT`) is **greater than or equal to
113,777 B**, the lossless semantic token stream it replaces. That comparison must use retained bytes from
the actual coder and a fresh parse-back.

The whole candidate has a separate and stronger success gate: it must satisfy the dynamic archive cap.
Beating 113,777 B does not make the candidate viable if other counted streams push the archive above
`B_max(D_student)`.

## Counted-byte ledger and dynamic budget

The current own-vehicle frontier and frozen-teacher comparison anchor dx2 is
`S=0.14821987563243377`, `D=0.028120227975693968`, archive **180,368 B**
`[contest-CUDA T4, n600]`.
Its exact archive SHA-256 is
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.

| Current stream or group | Bytes | Type | Relation to the 42,382 B cut |
|---|---:|---|---|
| Lossless semantic token field | 113,777 | MEASURED, current exact component | Primary NR1 replacement target |
| dxi carrier | 22,010 | MEASURED, current exact component | Re-solved fresh; never assumed free |
| compact residual | 96 | MEASURED, current exact component | Charged unless replaced |
| all remaining archive bytes | 44,485 | DERIVED from exact archive minus the three measured rows | Includes HPAC, renderer, headers, and ZIP framing |
| of remainder: HPAC | 13,515 | MEASURED, current exact section | May be absorbed into `QCTX`/`QEVENT`; never double-counted |
| of remainder: semantic renderer | 30,856 | MEASURED current exact section | Charged at its new actual size |
| of remainder: internal/ZIP framing | 114 | DERIVED residual | 14 B member header plus 100 B ZIP framing |
| **Exact current archive** | **180,368** | **MEASURED, contest-CUDA T4 n600** | **Must fall by 42,382 B at current D** |

The K32 row measures the complete quotient at 69,004 B: QPARAM 239 B, QCTX 152 B, QPAIR 52,124 B, and
QEVENT 16,489 B, including exact physical framing attribution. It clears the 71,395 B token-only ceiling
by 2,391 B, so the representation is not killed by rate. This does not establish matched distortion.

For every parsed student candidate:

`B_max(D_student) = ceil((0.12 - D_student) * 37_545_489 / 25) - 1`

and

`B_quotient_max = B_max(D_student) - B_all_other_actual`.

At the current distortion, `B_max=137,986 B`. If the current non-token/non-HPAC accounting were held fixed
only for a first-body comparison, the combined replacement for token+HPAC would have at most **84,910 B**:

`137,986 - (22,010 + 96 + 30,856 + 114) = 84,910`.

That is a **derived comparison ceiling**, not a reserved allocation. It implies the combined 127,292 B
token+HPAC pool must shrink by the same 42,382 B. Every renderer, dxi, correction, framing, and distortion
change forces a fresh calculation from actual bytes. At zero distortion the current 180,368 B archive
would still need a 150 B cut; there is no distortion-only route to sub-0.12.

## Teacher-manifest prohibition list

The training manifest and shipping manifest are distinct artifacts. The training manifest may reference
teacher material under external custody. The shipping manifest is an allow-list of counted candidate
sections and must fail closed if any training-only role appears.

The following are forbidden from `archive.zip`, `inflate.py`, `inflate.sh`, generated source, constants,
resources, vendored packages, or any receiver-reachable side channel:

- source or teacher RGB/YUV frames, decoded raw video, PNG trees, or frame patches;
- dense teacher logits, margins, argmax maps, probability planes, masks, cell tables, or dense token fields;
- SegNet, PoseNet, or any other scorer weights, activations, gradients, optimizer state, or scorer cache;
- teacher checkpoints, EMA shadows, training checkpoints, optimizer checkpoints, or resume state;
- GT partitions, GT-argmax tables, labels, oracle decisions, or teacher-vs-GT correction tables;
- C1, DM, EC1, PP1, SP1, WS1, TL1, ES1, or IG1 payloads and dense caches;
- old endpoint dxi, old compensator coefficients, stale residuals, or copied carrier state;
- precomputed per-frame generator outputs, video-derived codebooks/bases/tables disguised as free code;
- paths, hashes, archive members, or embedded blobs that cause the receiver to fetch training material;
- any unclassified byte whose unique runtime consumer is not declared.

Allowed shipping content is limited to actual-coded NR1 description bytes, the current candidate's freshly
trained dxi/residual bytes, strict format metadata, and generic receiver code. A SHA reference does not make
a forbidden artifact shippable; it is allowed only in the external training/custody receipt.

## Receiver contract

The NR1 receiver is not complete until a real active candidate satisfies every item below. A parser shell,
constant-plane fixture, or length/CRC test without the actual task-cell render is not an implementation.

1. **Outer container.** Use a single stored member named `p`, with a versioned NR1 header, explicit mode,
   exact section lengths, per-section integrity, and no tolerated trailing bytes. The canonical packer must
   reproduce the same member byte-for-byte from the parsed object.
2. **Inactive identity.** With NR1 absent or explicitly inactive, parse-back and full inflate must reproduce
   the frozen base archive output byte-identically. An empty active section is not silently treated as an
   inactive candidate.
3. **Active completeness.** Active mode must consume every declared paid section exactly once and run the
   real quotient decoder, task-cell renderer, correction compositor, and fresh carrier. Unknown, duplicate,
   missing, reordered where order is semantic, overlong, or underlong sections fail closed.
4. **Fresh-process proof.** Encoding and decoding occur in separate fresh processes. Decode may read only
   the archive, free receiver code, and contest runtime. It may not read the teacher manifest, worktree
   caches, environment-specific absolute paths, or the source video.
5. **Mutation proof.** The retained receipt must include deterministic refusal for corrupt magic/version,
   lengths, integrity fields, each payload section, truncation, appended bytes, and inactive/active confusion.
   It must also include one-bit mutation evidence for every paid section showing either a changed decoded
   output or a deterministic refusal.
6. **Determinism.** Two independent inflates of the exact archive produce byte-identical complete raw output
   and matching tree/content hashes. The packed archive is repeated where cheap and its SHA/bytes recorded.
7. **Staged shipping.** Receiver closure requires the exact staged `inflate.sh` path to materialize the full
   **3,662,409,600 B** raw output. The raw payload and all candidate payloads are retained on the SSD tier;
   a scalar-only receipt is forbidden.
8. **Authority separation.** Receiver closure proves bytes and behavior, not score. Scorer evaluation occurs
   only after closure, on the exact retained archive. macOS/MPS results are advisory; a frontier claim still
   requires the governed contest-CPU/CUDA lane and canonical pointer update.

The prebuild satisfies the packet-level portion only: strict canonical parse/repack, bad magic/version/
mode/section/coder/order/truncation/trailing refusal, absent and explicit-inactive base-payload identity,
parse-valid semantic mutations for all four surfaces, inert-consumer refusal, exact physical ownership,
an isolated staged-source process decode, deterministic coder/packet/ZIP repeats, full n600 token decode,
and one actual-renderer pair counterfactual. It does **not** satisfy full inactive archive identity, a
production `inflate.sh` active branch, minimal-environment full raw decode, two complete raw inflates, or
fresh endpoint carrier integration. Therefore `receiver_closed=false` is binding.

Every materialized NR1 payload and receipt goes under
`/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/<candidate_id>/retained/`, with bytes, SHA-256,
command/config/env, source hashes, seed, checkpoint lineage, coder version, parse-back result, decoded raw
hash/bytes, and disposition. If space preflight fails, the job blocks before launch; it does not discard.

## Sealed MAIN fire order

There is exactly one fire order: **`NR1_MAIN_FIRE_20260822_V1`**.

| Field | Sealed value |
|---|---|
| Disposition | `QUEUED-WITH-A-FIRE-ORDER / ARMED-NOT-FIREABLE` |
| Owner | `MAIN` |
| Consumer store | `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/retained/` |
| Fire trigger | r9 has stopped; MAIN has declared one terminal endpoint frozen; endpoint archive, full raw, logits, margins, argmax, Pose6, fresh dxi, receiver-closure, and provenance receipts all exist and verify; no competing NR1/scorer lane is active |
| First dispatch | Bind the now-executable K32/K64/K128 quotient to the newly frozen primary endpoint, refit QEVENT against primary margins, build the active production-runtime branch, and integrate a fresh terminal carrier; retain all rows and repeats |
| First scorer fire | Only after inactive full-archive byte identity, active fresh-process parse-back, mutations, deterministic repeat, rule-118 source closure, and full staged 3,662,409,600 B raw output; then MAIN grants one governed n600 scorer lane to the exact retained candidate set |
| Selection | Lowest exact parsed-candidate score among the retained candidates, subject to `archive_bytes <= B_max(D_student)` and the 113,777 B quotient falsifier |
| Stop | Stop the family if its optimal actual-coded quotient is >=113,777 B at matched distortion, if no candidate fits the dynamic whole-archive cap, or if receiver closure fails; retain all bytes and typed negatives |

The scorer-free prebuild command now exists and has run, but no scorer command is sealed because the
primary endpoint is not frozen and the current packets are not production-runtime closed. The existing
`NR1_MAIN_FIRE_20260822_V1` queue row already owns the future action; this arm did not append a duplicate.
All scorer work remains gated on the stronger full receiver proof above.

## Conclusions

- NR1 now has a real fitted `QPARAM/QCTX/QPAIR/QEVENT` packet and task-token receiver. The K32 row survives
  the actual-coder rate falsifier at 69,004 B and projects 135,595 B with all current non-token bytes held.
- The result is not matched-distortion evidence. K32 differs from DX2 at 1,558,833 tokens, and the secondary
  C1 margin field cannot stand in for the unfrozen primary r9 endpoint.
- Packet mechanics are verified; full shipping closure is not. No production active branch, complete raw,
  fresh terminal carrier, scorer row, exact score, or pointer move exists.
- QPAIR dominates K32 at 52,124 B; QEVENT is next at 16,489 B. Further rate work should target temporal
  assignment entropy or better task-weighted dictionary birth, not relabel framing or generic coder state.
- The exact same-distortion archive cut is 42,382 B, not 42,381 B. The direct core falsifier remains
  113,777 B at matched distortion; the current run resolves only the rate half.
- The sealed future action remains single-owner MAIN. The already-harvested queue row is authoritative and
  was not duplicated.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER / ARMED-NOT-FIREABLE. Owner: MAIN. Consumer store:**
  `/Volumes/APDataStore/pact/ddm_nr1_taskcell_quotient_prebuild/retained/`. **Fire trigger:** r9 is terminal,
  MAIN has frozen and verified one endpoint archive/raw/logit/margin/argmax/Pose6/fresh-dxi receipt, and no
  competing NR1/scorer lane is active. **Action:** execute `NR1_MAIN_FIRE_20260822_V1` by refitting the
  executable K32/K64/K128 quotient against the frozen primary endpoint, integrating the production active
  receiver plus fresh carrier, and withholding scoring until full receiver/rule-118 closure passes.

## LIVE-HYPOTHESES

- The K32 mismatches may be tolerable after primary-endpoint refit because a task-cell quotient need not
  preserve token identity, and the 8,192 current events already target the smallest secondary margins.
- A task-weighted dictionary can improve agreement without paying K64's full QPAIR entropy because K32's
  dictionary was selected by raw pattern frequency, not evaluator-cell value.
- QPAIR may fall materially below 52,124 B with a learned temporal state transition because its current
  symbols use only previous/default/direct choices while the full sequence has structured pair motion.
- K64 may remain viable as a token+HPAC joint replacement because its 79,876 B packet clears that combined
  ceiling by 5,034 B and raises token agreement over K32.
- Fresh terminal dxi may compose with the quotient because the actual semantic receiver consumes the new
  token field and can expose a new carrier optimum rather than inheriting stale compensation.

## DEAD-ENDS

- A specification-only NR1 is closed: this arm built and measured an executable packet, so successors must
  advance primary-endpoint fit or full closure rather than rewrite the format memo.
- The July quotient ABI is closed as a scientific shortcut: its rank-one renderer, no-op placement fixture,
  absent QCTX, and hard-tail n24 scope do not perform this task-cell quotient.
- K128 in the present frequency-born form is closed by rate: 91,630 B misses even the 84,910 B combined
  token+HPAC ceiling by 6,720 B.
- Reusing 33.5 KB as a model/context budget is closed because it ignores realized distortion and the actual
  complete packet.
- Full explicit worldsheet shipping is closed by WS1 at 918,904 B lossless and 885,750 B at q2.
- Integrity-only mutation, four names sharing one inert handler, prefix-only evidence, and exact token
  recoding are closed as receiver or science evidence.
- The three current rows are closed as score claims: none used the frozen primary endpoint, materialized a
  complete raw, integrated a fresh carrier, or ran the scorer.

S = 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]; archive SHA-256 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674; frontier unmoved.
