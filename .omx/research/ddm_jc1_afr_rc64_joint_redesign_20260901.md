# JC1 AFR-RC64 joint redesign — one mechanism class survives arithmetic; no candidate is yet measured

Date: 2026-09-01  
Arm: `ddm_jc1_afr_rc64_joint_redesign`  
Terminal verdict: **MECHANISM-NAMED**  
Mode: **`$0`, scorer-free**  
Authority: synthesis of retained exact bytes and prior full-n600 receipts; **no new score, scorer run, archive, field, model, or payload was materialized by JC1**

## Conclusion

The rank-3 successor space is **not closed**. One mechanism class remains genuinely unmeasured:
**scorer-cell minimum-description-length projection (`SCMDL`)**. It changes the final dense token
field, the causal decode schedule/context graph, and the probability model together, then accepts a
field only by realized Seg/Pose behavior through the byte-closed renderer and prices it only with the
exact RC64 coder. The update is address-free because the changed field is the coded object; there is
no coordinate, exception, or GT side stream.

This is not a candidate or a success claim. JC1 did not find a retained pose-valid field proposal that
instantiates the class. Re-encoding JF/FCD/WWC fields under a renamed schedule would repeat already
closed mechanisms, and building from token-GT labels would repeat WWC1's label-to-realized failure.
The honest Stage-1 output is therefore a staged build fire order, not invented bytes.

## Pins and authority boundary

| Object | Pin / fact | Use |
|---|---|---|
| JC1 charter | SHA-256 `4ba860df7661dea224aece0662deb8d44c37f6ad01765fc58e5cd64d83d5e5fa` | governing arm contract |
| common contract | SHA-256 `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` | governing common contract |
| NX1 route memo | content SHA-256 `17d1ca197e963802a6a0a09437f8132c1072d208fd894ee07c118748c4774bbe`; charter lineage commit `5a19419bc8` | rank-3 definition and affine constraint |
| QX1 → QX4 chain | content SHAs `067a3014…`, `3bcd01d2…`, `ac893d74…`, `bb98fe45…` | complete event/conditioning representation chain |
| AFR1 archive | 180,002 B; SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` | live exact contest-CUDA pointer object |
| AFR1 distortion | `d_seg=0.00020139`; `d_pose=6.37e-6` | exact pointer components; not remeasured here |

The common contract's embedded QO1 frontier is stale. The live hot state and AFR1 pointer receipt are
the authority used here.

## Stage 0 — denominator and exact arithmetic

The mechanism census denominator is **8 / 8 rank-3 representation/model classes**. The prior-negative
fold denominator is **6 / 6 named negatives**. QX4's representation denominator is **6 / 6 complete
decodable forms** over **600 / 600 pairs**, **117,964,800 / 117,964,800 conditioning sites**, and all
**17,926 / 17,926 events**. JC1 added **0 physical measurements** and **0 materialized payloads**.

AFR1's jointly mutable pool is exact:

```text
token stream                         113,411 B
HPAC model                            13,515 B
joint field+model pool               126,926 B
immutable renderer/carrier/residual/
framing remainder                     53,076 B
archive                              180,002 B

affine-floor archive ceiling         140,479.86 B
allowed joint field+model pool        87,403.86 B
required joint-pool cut               39,522.14 B
required cut / joint pool              31.137938641%
```

The most favorable measured current-field conditioning target is 2,162.126399 B. Even granting all of
it at zero model-byte cost would leave 177,839.873601 B, still **37,360.013601 B over** the affine
ceiling. This is a current-field scale diagnostic, not a theorem about a changed field or context graph.
It does establish that the surviving route cannot honestly be described as an AFR1 model tweak.

| # | Mechanism class within the rank-3 surface | Best retained scale or relaxed floor | Arithmetic / evidence verdict |
|---:|---|---:|---|
| 1 | Fixed field, coder/framing replacement | JT23 generic coder axis: 0 B; AFR1 already harvested 81 B | **FOLDED, CURRENT-BODY FORMULATION.** Does not jointly change field/model and is far below 39,522.14 B. |
| 2 | Fixed field, calibration/static conditioning/paid-model replacement | 2,162.126399 B measured conditioning excess; richest unconsumed free context 211.13 B gross | **FOLDED AS A STANDALONE ROUTE, CURRENT-FIELD FORMULATION.** Same-field rearrangement cannot change the field entropy; model approximation scale misses the demand. |
| 3 | Exogenous threshold/drop field plus refit under the unchanged causal topology | JF2 best subsystem delta −1,576 B; realized terminal rows lose `+0.064232` to `+0.719950 S` | **FOLDED, FORMULATION.** Five-arm sharp optimum, JF1's +7,554 B positive-control deficit, and the realized pose wall bind this topology. |
| 4 | Token-GT benefit cone plus refit or post-hoc scorer selector | FCD1 −3,756 B; JF2 cone −4,210 B; best pose-safe FCD3 −2,940 B but `Delta S=+0.0019433244` | **FOLDED, TOKEN-GT / CURRENT-SELECTOR FORMULATION.** Real rate direction exists, but label benefit does not transfer to realized Seg and the union is nonadditive. |
| 5 | Separate exception/event stream conditioned on a decoded base | QX4 best complete archive 147,327 B; event payload 33,435 B | **FOLDED, QX FORMULATION.** It is 6,847.14 B above the AFR affine ceiling (and 9,342 B above QX's stricter fixed-distortion cap). All six decoded-base forms were raced. |
| 6 | Explicit quotient, topology, shape, or Lane carriage | QBW2 188,860 B; topology 233,262 B; best lossy Lane packet 36,044 B versus its 21,699 B target | **FOLDED, MEASURED REPRESENTATIONS.** Address/shape storage violates the address-free corridor or misses its own byte target. |
| 7 | Changed dense field with the same HPAC causal topology, followed by model refit | JF/FCD are the retained instances; best byte scale 1,576–4,210 B and every realized scorer transfer tested is refused | **FOLDED FOR THE MEASURED PROPOSALS, NOT A GLOBAL FIELD THEOREM.** It does not include a changed causal graph. |
| 8 | **SCMDL: scorer-cell field projection + causal schedule/context graph + probability-model co-design** | relaxed archive lower bound 53,076 B if the entire 126,926 B joint pool vanished | **SURVIVES ARITHMETIC; UNMEASURED MECHANISM CLASS.** The relaxed floor is 87,403.86 B below the ceiling, so arithmetic cannot close it. |

The relaxed SCMDL floor is deliberately optimistic. It is used only as the charter's Stage-0 falsifier:
because it lies below 140,479.86 B, JC1 may not issue `SUCCESSOR-SPACE-CLOSED`. It is not evidence that
87,403.86 B is achievable.

## Why SCMDL is a distinct mechanism

Let `X` be the final five-class field, `G` the causal schedule/context graph, and `M` the counted HPAC
model. The actual optimization object is the two-part exact description length

```text
L(X,G,M) = bytes_RC64(X | G,M) + bytes_model(G,M)
```

subject to all of these hard constraints:

1. the public receiver reconstructs the intended `X` exactly and the current renderer remains
   byte-closed;
2. full-n600 realized `d_pose <= 6.37e-6`, or any larger pose is priced in the exact score law rather
   than called “preserved”;
3. Seg acceptance uses frozen-scorer output through `R`, never token-GT or coding-argmax labels;
4. the complete archive is no larger than 140,479.86 B for the AFR1-pose affine cell, and the terminal
   authority is still exact `S < 0.12`, not the affine projection;
5. all video-derived model state and field bytes remain counted, retained, hashed, and parsed back.

The distinction from the six negatives is the coupled variable. JF1/JF2 changed `X` under one fixed
causal topology. FCD/WWC selected changes using label-space benefit. CM1 searched surrogates for the
existing exact coder state. QX1–QX4 stored a separate event section. RR9 proved fixed within-group
reordering is byte-neutral but explicitly left cross-group causal-schedule redesign unmeasured. QX4
then supplied the missing law: changing the decoded conditioning field can reorder coder rankings.
SCMDL changes `X`, `G`, and `M` in the same accepted stage and refits/reprices after every field move.

## Stage 1 — build boundary

No survivor payload was built. The retained inputs contain exogenous-threshold fields and token-GT
cones, but no scorer-realized, pose-valid whole-field direction for SCMDL. Using either retained family
as the “new” field would duplicate a closed cell. A new causal schedule alone on AFR1's unchanged field
would also fail the joint-change requirement and remain bounded by same-field model approximation.

The first build is therefore split into two non-promotable gates:

1. **Exact incremental-coder gate.** Add restartable HPAC/RC64 state immediately before edited pairs.
   Its null control must reproduce the full AFR1 stream byte-for-byte. Its stratified-random `n>=32`
   screen must compare incremental deltas with full exact re-encodes and clear the preregistered
   Pearson/Spearman `>=0.9` direction gate. CM1's existing halo-0 result makes this plausible, but its
   897.675 s full-state cost does not satisfy the gate.
2. **Joint realized-field gate.** With an assigned scorer lane, generate whole-field proposals from
   realized Seg/Pose cells, refit `G,M`, and admit only exact full-reencode, full-n600, receiver-closed
   rows. Every proposal payload, model, stream, archive, decoded field, raw receiver output, and repeat
   must be retained.

Until gate 1 exists, an outer loop is computationally non-fireable. Until gate 2 exists, rate-only rows
cannot be called joint scorer/coder redesign. JC1 did not launch either and spent `$0`.

## PRIOR NEGATIVES — name-or-fold receipt

| Required negative | Fold |
|---|---|
| Sharp optimum in every direction, five arms (#1214) | Classes 2, 3, and 7 under the fixed AFR/DX2 causal topology; not transferred to changed `G`. |
| Diagonal entered and refused 686× (#1239) | Class 3; realized pose dominates the rate credit. |
| JF1 positive-control fail +7,554 B (#1221) | Class 3; the warm-start/refit instrument was weaker than shipped before field comparison. |
| WWC1 label → realized broken | Class 4; forbids token-GT acceptance in SCMDL. |
| CM1 no cheap differentiable rate surrogate | Stage-1 gate 1; exact restartable state is required before an outer loop. |
| FCD1 5,268 label-benefit edits shrink the archive, realized transfer broken | Class 4; retained as proof that field changes can lower real RC64 bytes, not as scorer feasibility. |

## RECALL EVIDENCE

The recall pass searched `.omx/research/` memos and arm finals, code, canonical equations, the canonical
research index, `sub015_DAG_*` FEED blocks, design/spec surfaces, and task-ledger rows. Content queries
included `joint field`, `field model`, `HPAC`, `RC64`, `causal schedule`, `reorder`, `conditioning`,
`restartable`, `exact coder`, `token-GT`, `event`, `exception`, `Lane`, `quotient`, `address-free`, and
the exact byte constants `39522`, `140479`, `113411`, and `13515`.

Beyond the charter seeds, recall found:

- RR9's scoped result: within-group fixed-field reorder is exactly 0 B, while cross-group reorder
  changes the causal mask/schedule and was explicitly not measured. This supplied `G` as the untouched
  part of the joint variable.
- MI1/recall5's 2,162.126399 B current-field conditioning target and 211.13 B best unconsumed free
  context. This removed a same-field model tweak from the survivor set.
- AFC1's section-complete AFR anatomy, which closes `180,002 - 113,411 - 13,515 = 53,076 B` exactly.
- JT23's 0 B generic-coder result, which removes a coder-only rename.
- FCD2/FCD3's realized full-n600 failures, which prohibit treating FCD1 label benefit as scorer benefit.
- QX4's six-form reordering under decoded-QBT conditioning, which requires refitting/repricing the model
  after each accepted field change rather than transferring a ranking from another field.

These findings changed the plan: the survivor is not “better HPAC,” “rerun FCD,” or “compress QX
events.” It is the coupled `X,G,M` projection above, with exact incremental coding as its first apparatus
gate and realized scorer cells as its only field-selection authority.

## Custody and program status

Machine-readable adjudication: `/Volumes/APDataStore/pact/ddm_jc1/RESULT.json`. No payload existed in
memory during JC1, so the P0 payload-retention denominator is `0 / 0`; nothing was measured and
discarded. No Modal call, scorer job, exact evaluation, or `upstream/` write occurred.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN`-assigned exact-coder builder; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/`; fire trigger: MAIN assigns the rank-3 owner with AFR1 pins unchanged and no duplicate active lane, then the builder must prove byte-identical AFR1 null replay and stratified-random `n>=32` incremental-versus-full exact delta correlation before exposing an outer-loop API.
- **QUEUED-CONDITIONAL** — owner: `MAIN` scorer-lane scheduler plus the rank-3 builder; consumer store: `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/`; fire trigger: the restartable exact-coder gate passes and MAIN explicitly assigns the single local scorer lane, then run whole-field `X,G,M` proposals with full-n600 realized Seg/Pose acceptance and retain every candidate payload and repeat.
- **QUEUED-CONDITIONAL** — owner: `MAIN` evaluation scheduler; consumer store: canonical candidate/evaluation ledgers plus `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/`; fire trigger: a receiver-closed repeat-identical archive is at most 140,479.86 B, has measured full-n600 pose/Seg components satisfying the exact score law, and passes the normal content-derived seals; only then request sequential contest authority evaluation.

**OWN-VEHICLE FRONTIER: UNMOVED — S = 0.14797617125559104 @ 180,002 B [contest-CUDA T4, n600], AFR1 archive SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25`; JC1 named an unmeasured mechanism and produced no score row.**
