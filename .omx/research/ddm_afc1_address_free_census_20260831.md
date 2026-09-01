# ddm_afc1 — address-free census of the exact shipped lb1 body

Date: 2026-08-31 · owner: `ddm_afc1` · score claim: **none** · dispatch: **none**

## Membership predicate

A transform is **ADDRESS-FREE** exactly when the information it adds to the counted archive is a fixed decoder rule or a bounded set of global/model parameters whose length does **not** grow with the number of affected sites. It ships parameters, not positions.

Operational test: hold the transform family and parameter precision fixed, increase the number of sites it affects, and ask whether the transform-specific payload must grow. A bitmap, index list, run length, coordinate, exception/event stream, per-site key, or dense replacement-symbol stream fails the test. A deterministic global rule, bounded model table, learned low-dimensional generator, whole-section parameterisation, or decoder-derived context passes it. Generic coder replacement is classified nonmember here because its counted coded stream scales with the symbol population even when it stores no explicit coordinate list.

Borderline calls are made on the complete shipped representation, not on a convenient sub-piece. A polynomial lane core is address-free; a lossless packet that adds per-site exceptions is not. A global relabel rule is address-free as an algorithm; a candidate that still ships a dense changed-symbol stream is not. This is stricter than AF1's earlier “does it explicitly name a subset?” predicate and is the predicate the AFC1 charter requires.

## Answer

The exact shipped-body census has **23 enumerated transform families / 18 members / 17 members with a built implementation or sealed apparatus / 1 member whose direct integrated transform is absent**. Exact-current evidence must not be collapsed across generations:

- **1/18 member family has fresh full-n600 exact-lb1 physical experiments:** zero-stored receiver-derived conditioning. It now contains three retained same-body candidate instances: shipped `patch192` (−109 B), OC2 `miss_rank8` (−2 B), and AFC1 `tile48 × groupbin8` (−81 B).
- **3/18 are already instantiated in exact current sections:** semantic `plane2`, RR5 arithmetic-basis parameters, and DX2 coefficient parameters.
- **1/18 has an exact-current structural floor:** ZIP/RX1 fixed framing.
- Every predecessor, equivalent-section transfer, alternate-body row, and stale apparatus is excluded from that exact-body numerator and labelled below.

The charter's falsifiable prediction is a hit: RB1 is address-free, **BUILT AND SEALED, NOT LAUNCHED**, and has never produced a trained changed-object packet or distortion measurement. Its exact inherited lb1 semantic section is SHA-bound, but its target is a changed born-small body, not the unchanged lb1 archive (`.omx/research/ddm_rb1_born_small_renderer_build_20260826.md:7-30,34-46,61-74`). CL1 is a second built-but-unmeasured apparatus, but its binding is stale PR130 rather than exact lb1 and does not earn an exact-body claim.

## Exact body denominator

Pinned archive: `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/retained/candidate_lb1_joint22_patch192.zip`, **180,083 B**, SHA-256 `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9`, one ZIP member `p`.

| Physical span | Bytes | SHA-256 / status |
|---|---:|---|
| ZIP local header | 31 | framing |
| RX1 header | 14 | framing |
| HPAC model | 13,515 | `602115…` |
| semantic renderer | 30,856 | `39d1be…` |
| carrier | 22,010 | `932b97…` |
| compact residual | 96 | `8ab2fe…` |
| token stream | 113,492 | `8838e44f6498cd9b94f480ae04d9ea12d89b7020ff3c6f215ff83de177a3eac2` |
| ZIP central directory | 47 | framing |
| EOCD | 22 | framing |
| **total** | **180,083** | **zero remainder** |

The three headline non-token payloads total 66,381 B. The charter's published **66,591 B** residue is correct only after adding the 96 B compact residual and 114 B of ZIP/RX1 framing. The exact non-token sum is `31+14+13,515+30,856+22,010+96+47+22 = 66,591 B`; adding the 113,492 B token stream closes the archive with no remainder. The non-token payload hashes are byte-identical to the AR1B/DX2 anatomy rows (`.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md:58-83`), so those section boundaries transfer; experimental verdicts do not transfer merely because bytes do.

## RECALL EVIDENCE

The recall preceded adjudication and searched the full corpus rather than only the charter seeds:

- Research memos and receipts: content queries for `address-free`, `ships parameters`, `bitmap`, `coordinate`, `implicit`, `born-small`, `merge`, `plane2`, `rank`, `precision`, `reorder`, `conditioning`, `groupbin8`, `tile48`, `unpriced`, and the pinned archive SHA; exact body receipts and runtime parser sources were inspected under both SSD tiers.
- Canonical equation registry: `.venv/bin/python tools/list_canonical_equations.py --json`; the applicable score remained `100*d_seg + sqrt(10*d_pose) + 25*bytes/37,545,489`, with no equation licensing a cross-unit exchange-ratio splice.
- Research index/graph surfaces: `.omx/research/CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, `.omx/state/main_hot_state.md`, and task-ledger rows around #1213, #1214, #1220, #1234, #1239, #1244, #1254, #1258, #1261, #1262, #1264, #1265, #1283, #1304, and #1326.
- Source surface: the exact shipped parser and corrector under `/Volumes/APDataStore/pact/ddm_lb1_banked_lossless_joint_collect/runtime_candidate_native/runtime/`, plus existing full-n600 physical re-encode machinery.

Beyond the charter's seeds, MI1 explicitly left **`tile48 × groupbin8` unpriced** and called it worth one row (`.omx/research/ddm_mi1_indicator_model_axis_20260824.md:451-458`); GB1 independently repeated that it remained unpriced and cheap (`.omx/research/ddm_gb1_groupbin8_conditioning_20260824.md:426-427,481-482`). Both marginal axes were already present on lb1, while the 384-cell interaction was absent. This changed the plan from a memo-only census to the retained exact physical experiment below. Recall also found that the charter inverted the primary #1264 side labels; that correction changes the law statement below.

## Enumerated shipped-body census

`YES` in the exact-lb1 column means a full physical experiment on archive SHA `5b856e…`; `CURRENT` means the representation is already instantiated in byte-identical current sections; `FLOOR` is an exact-current structural bound; `EQUIV`, `PREDECESSOR`, `ALTERNATE`, and `STALE` never count as exact-body measurements.

| # | Candidate transform family | Member? | Built? | Exact lb1? | Receipt, boundary, and disposition |
|---:|---|:---:|:---:|:---:|---|
| 1 | ZIP/RX1 fixed framing collapse or alternate constant grammar | yes | yes | FLOOR | JT23's one-member structural race leaves a 0 B reducible floor; current framing is 114 B. **CLOSED**. |
| 2 | Semantic whole-section plane serialization (`plane2`/global `k`) | yes | yes | CURRENT | CK2 built and consumed `plane2`; exact current semantic section is 30,856 B. Other whole-section recodes were closed by BP1/JT23. |
| 3 | Semantic global bit-depth/depth/row prune or coarsen | yes | yes | EQUIV | AP1 measured 5,886/17,015/18,427 B credits but +8.54/+17.18/+17.85 score damage; zero admitted (`.omx/research/ddm_ap1_residue_purchase_scorer_20260823.md`). Exact affected-section bytes transfer, not an exact-lb1 rebuild. **REFUSED-FAMILY**. |
| 4 | Semantic born-small/narrow learned renderer | yes | yes | NO | RB1 sealed four configs and exact input bindings, but no trained packet or distortion. Target is a changed BS3 body. **QUEUED scorer-gated prediction hit**. |
| 5 | Carrier fixed whole-section plane serialization | yes | yes | PREDECESSOR | CK2 measured −44 B only on a compensated lattice and +41 B control; current carrier plane bit is off. **CLOSED unless body layout changes**. |
| 6 | Carrier RR5 arithmetic-basis parameters | yes | yes | CURRENT | Receiver parses RR5 parameters; exact current 22,010 B carrier already consumes them. **SHIPPED**. |
| 7 | Carrier DX2 CABAC coefficient parameters | yes | yes | CURRENT | Receiver parses DX2 coefficient parameters; exact current carrier already consumes them. **SHIPPED**. |
| 8 | Carrier global rank/dimension/quantization | yes | yes | EQUIV | AP1 measured 2,742/5,875/9,035 B credits with +0.3045/+0.7275/+3.922 score damage; zero admitted. Post-hoc rank/refit is owned by that refusal. |
| 9 | Carrier implicit/fresh learned resolver | yes | yes | ALTERNATE | BS3 built and measured an alternate born-small object; the learned implicit successor was not built and route-2 was refused. No exact-lb1 datum. **FOLDED into the existing born-small queue**. |
| 10 | HPAC structured shrink/prune/width/depth | yes | yes | PREDECESSOR | HM1/MP3 measured local shrink and member-prune knees; MP3 found 34 B at 16/19, below its bar. The generation-22 corrector pool differs, so no exact-body claim. **REFUSED-FAMILY**. |
| 11 | HPAC discrete-context or linear parametric replacement | yes | yes | PREDECESSOR | LM1's best W=0 complete archive was 193,065 B and its learned lower bound 150,903 B versus the 127,292 B subsystem break-even. **REFUSED-FORMULATION**. |
| 12 | HPAC nonlinear retrain/widen/fixed-topology capacity | yes | apparatus | STALE | CL1 preregistered apparatus but ran no rung; receipt remains blocked and tied to PR130 (`.omx/research/ddm_cl1_capacity_20260809/BLOCKED_RECEIPT.md`, `PREREGISTRATION.md`). **FOLDED; requires current-body re-derivation, not duplicate fire**. |
| 13 | Compact residual global precision/coarsening | yes | yes | EQUIV | AP1 found every precision level stayed 96 B while distortion failed catastrophically. Fixed-layout rate credit is zero. **REFUSED-FAMILY**. |
| 14 | Compact residual bounded parametric context-table redesign | yes | yes | ALTERNATE | HM1 built a correction-table ladder on another body. The current 25×5 table is bounded parameters, but no exact-lb1 receipt exists. **FOLDED into HPAC/table knee evidence**. |
| 15 | Zero-stored receiver-derived conditioning over token coder | yes | yes | **YES** | Shipped `patch192` −109 B; OC2 `miss_rank8` −2 B; AFC1 `tile48 × groupbin8` **−81 B**, all exact-lb1, full n600, zero changed tokens. Family is drained except the admitted native-port/identity work below. |
| 16 | Global 5→4 alphabet merge plus integrated model/refit | yes | **ABSENT** | NO | MA2 showed the coded symbol is the class, built/raced/lost the lossless Lane/worldsheet repair, and closed fold-only transfer; the direct integrated merged-alphabet HPAC/renderer transform itself was not built. **CLOSED by recall; do not count the related address-paying repair as this build**. |
| 17 | Analytic/born-small generator replacing token+HPAC with bounded parameters | yes | yes | ALTERNATE | GF1's FORM archive was 433,051 B/5.09× and did not verify byte-identity of its source field to the live pointer; BO2/BZ2 use alternate bodies. **REFUSED-INSTANCE; no exact-lb1 claim**. |
| 18 | Fixed global seeded reorder/permutation | yes | yes | CLOSED-BEFORE-MEASURE | RR9 proved exact DX2 order invariance, 113,777→113,777 B, and the group index is fused into training. Architecture transfers; another physical lb1 race would rename a closed route. |
| 19 | Generic coder swap/re-race | **no** | yes | n/a | Strict predicate: the resulting counted code stream scales with all positions. JT23 also closed the practical section-coder axis (HPAC +40, semantic 0, carrier +2, token +5 across 25 configs). |
| 20 | Tolerance/global relabel/diagonal while retaining a dense replacement-symbol stream | **no** | yes | n/a | The algorithm may be global, but the complete archive still ships one changed symbol/key per site. Existing tolerance and diagonal refusals own it; AFC1's stricter predicate corrects AF1's broader call. |
| 21 | Explicit bitmap/index/RLE/run/coordinate/per-site selector | **no** | yes | n/a | Fails the charter predicate directly. AE1 flags, AD2 QEVENT addresses, and sparse selector edits are measured examples. |
| 22 | Exact worldsheet/lane repair with curve coordinates and exception packets | **no** | yes | n/a | The polynomial core alone is a member; the complete lossless WS1/MA2 packet ships coordinates/exceptions and is not. Its measured rate refusal owns the route. |
| 23 | GF1/topology exact residual sidecar | **no** | yes | n/a | Per-site mismatch events/addresses/RLE scale with errors. GF1's residual was 385,448 B; ET1 topology scales with 1.94M leaves. Calling addresses “topology” does not remove the tax. |

**DENOMINATOR:** `23 enumerated / 18 strict members / 17 members built as implementation or sealed apparatus / 1 member direct-transform ABSENT`. Exact-current evidence is `1 fresh full-n600 physical experiment family + 3 already-instantiated current-section families + 1 structural floor`; the first family contains `3 exact-lb1 candidate instances`. No predecessor, equivalent-section transfer, alternate body, or stale apparatus is hidden in that numerator.

## New exact physical measurement

Candidate: add the zero-payload 384-cell interaction

`tile48=((y//64)*8+(x//64)); groupbin8=(((x%64)+2*(y%64))*8)//190; context=tile48*8+groupbin8`

after the exact shipped 22-member conditioning configuration. The decoder computes both axes from `(x,y)` and stores no new candidate parameter bytes. The run used the real RC64 path over all 600 frames and 117,964,800 decoded tokens, with 25-frame atomic checkpoints and all control/candidate/repeat streams, archives, ledgers, sources, builds, checkpoints, and receipts retained under `/Volumes/APDataStore/pact/ddm_afc1_address_free_census/tile48_groupbin8/`.

Axis: **[macOS-CPU advisory / scorer-free EXACT byte measurement]**. No scorer, MPS, Modal, or `upstream/` mutation was used. This is a physical rate datum, not a promotable score.

| Gate | Result |
|---|---|
| custody pin | exact lb1 180,083 B / `5b856e…`; exact decoded tokens 117,964,800 B / `cc10a7…`; runtime pin `CONSISTENT` |
| fresh null control | 113,492 B token stream / `8838e44…`, byte-identical to shipped; retained receipt `physical_v1/retained/S1_control_600.json` SHA `993baca9…` |
| candidate | 113,411 B stream / `5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3`; 180,002 B archive / `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| determinism repeat | stream and archive byte-identical to candidate |
| identity of decoded field | encoder receipt `tokens_changed=0`; independent native receiver identity still owed |
| exact delta | token **−81 B**; archive **−81 B**; `ΔS_rate = −0.00005393457520289588` |
| admission | clears the preregistered 30 B bar by 51 B; `ADMIT_NATIVE_PORT_AND_RECEIVER_IDENTITY_OWED` |

Authority receipt: `/Volumes/APDataStore/pact/ddm_afc1_address_free_census/tile48_groupbin8/measurement_v1/ADJUDICATION.json`, 2026-08-31 SHA-256 `9bda316e278e6bf37e762c6c1308cc014db2f76703ce327eef0bad064b6ed841`. Full retained-store manifest: `measurement_v1/MANIFEST.json`, SHA-256 `1e8a111e8f5d010d67ac34e212a81370341743c4e9ab148c14b2ceb22425a425`, 45,807,018 retained bytes at capture.

The first wrapper attempt incorrectly aimed its null-control stage at the patched candidate runtime. The shipped-stream pin caught the mismatch at 113,411 B; that trace and checkpoint remain retained and are explicitly `EXCLUDED_INSTRUMENTATION_ERROR`, never counted as a control or a second measurement. A separately staged unmodified runtime then passed the byte-identical control. Two earlier slow runtime-copy attempts were interrupted; their partial bytes are retained and manifest-listed rather than silently deleted.

The candidate-side native receiver identity did **not** run. Storage preflight observed 6,382,288,896 B free against 8,142,450,560 B required for 3,662,409,600 B raw output, the 117,964,800 B token checkpoint, 64 MiB scratch, and a 4 GiB post-run reserve: shortfall 1,760,161,664 B. Receipt: `measurement_v1/IDENTITY_PREFLIGHT.json`, SHA-256 `ddfd268f1406b2ba57b392fc88f753b21d11eac390ee700243ca07897fb5f6ed`, disposition `QUEUED_AFTER_APDATASTORE_IDENTITY_FLOOR`. Certify-or-block therefore forbids pretending the independent decode gate passed.

## #1264 law: corrected n and verdict

The AFC1 charter reverses the primary #1264 labels. AF1 itself says **all six old exchange-ladder rungs were address-free and zero were address-paying**, not the reverse (`.omx/research/ddm_af1_address_free_class_law_20260824.md:23-46,243-273`). It also proves the ladder mixes units: W72's published 35.5364× cell was an S-ratio, comparable exchange ratio 922×; NI1's 247.69× was seg/ceiling, comparable ratio 714×; and tba1-D3's 21.62× was a derived seg-only phantom for a transform never built (`:51-55,156-215`). Those cells are excluded, not reused.

Under AF1's original loose criterion, the heterogeneous lossy ladder remains **n=6 address-free / n=0 address-paying**, and its proposed addressing law remains **undefined**. The classes are also incommensurable there because the address-paying exact packets have zero distortion and therefore no damage/credit exchange ratio.

Under AFC1's stricter shipped-payload predicate, the exact-lb1 lossless surface now has **n=3 retained address-free candidate instances** (`patch192`, `miss_rank8`, `tile48 × groupbin8`) within **one receiver-conditioning family**, each with unchanged decoded symbols. The old “empty side” premise is therefore refuted on the practical exact-body surface. The law is **still undefined, not supported or refuted as a trend**: three pure-rate, zero-damage points from one family cannot be spliced into a heterogeneous lossy damage/credit ladder without changing the ordinate. The defensible statement is narrower: at least one strict address-free family produces real same-body bytes, and AFC1 replicated that sign with a new composite interaction.

## Prior-negative reconciliation

- **Sharp optimum:** the prediction “all directions lose” is falsified at the instance level by the new −81 B zero-damage interaction, consistent with FCD1's earlier win-win exception. It does not reopen semantic quantization, carrier quantization, HPAC shrink, residual precision, or alternate-body generator refusals.
- **Born-small:** membership never implied success. BO2's distortion refusal and GF1's 5.09× packet remain closed instances. RB1 is queued only because it is a different, sealed changed-object renderer whose distortion has never been measured.
- **Merge:** MA2 owns global merge/fold and lossless worldsheet repair. The direct integrated 5→4 transform is absent, not secretly built by the address-paying repair.
- **Reorder:** RR9's trained group-index fusion and exact order-invariance close the family before another lb1 measurement.
- **Coder:** JT23 closes generic reraces, while OC2 and AFC1 are model-context changes with the coder and decoded symbols held fixed. This is not coder renaming.
- **Packet promises:** AFC1 saves 81 B, beating GF1's cited 14 B residual opportunity but remaining tiny against the frontier demand. It is a bankable free rate row, not a route to sub-0.12 by itself.

## Typed dispositions and fire orders

- **QUEUED_AFTER_APDATASTORE_IDENTITY_FLOOR** · owner: `AFC1 receiver-closure successor` · consumer store: `/Volumes/APDataStore/pact/ddm_afc1_address_free_census/tile48_groupbin8/receiver_identity_v1/` · fire trigger: APDataStore free space is at least **8,142,450,560 B**; then port only `tile48_groupbin8` to the staged native C receiver, prove Python/C configuration parity, and run one retained full-n600 candidate receiver identity. No scorer.
- **QUEUED_AFTER_SR3_BS4_W96B** · owner: `MAIN` · consumer store: `/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/` · fire trigger: SR3 is green with at least **60,380,026,816 B** free, #1304 BS4 then W96B is complete, MAIN claims distinct scorer and Metal lanes, and the shared changed-object teacher/scorer cache is materialized; then fire RB1's four sealed configs without mutating their originals (`.omx/research/ddm_rb1_born_small_renderer_build_20260826.md:90-97`).
- **FOLDED** · owner: `MAIN` · consumer store: existing HPAC/CL1 evidence stores · fire trigger: none on current evidence. CL1's stale PR130 apparatus may re-enter only after a current-body formulation derives positive headroom not owned by LM1/HM1/MP3; do not duplicate-fire it as “address-free capacity.”

**OWN-VEHICLE FRONTIER: UNMOVED — S = 0.14803010583079396 @ 180,083 B [contest-CUDA T4, n600], exact lb1 SHA `5b856e…`; AFC1 measured a non-promotable 180,002 B scorer-free candidate and did not claim or move the pointer.**
