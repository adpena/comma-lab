# DDM EU4 — fresh-eyes fractal composition after QS4

Date: 2026-08-13

Arm: `ddm_eu4_fresh_eyes_fractal_composition_20260813`

Authority: research, exact arithmetic, custodied-byte inspection, and source inspection only

Score claim: `false`

Pointer moved: `false`

## Conclusion first

The immediate path and the direction of travel are different.

- **Immediate path:** finish QS5 on the exact trimmed edit object, but require a complete-object
  Pose re-solve, a whole-container recount, and a net move beyond `1e-5 S`. The strict three-pair
  object with its observed 17 net flips is too weak by itself. With q11's measured `+12 B`, a
  doubled QS2 Pose-leakage allowance, and HP4's independently measured `-5 B` repack, it projects
  only `-9.524830487e-6 S`: still inside the declared reporting band.
- **First super-band composition:** add RE1's independently observed two Seg flips only after
  solving its frame-0 compensation as part of the same exact union object. If the union retains
  19 net flips, closes at at most `+8 B`, and keeps the total Pose-term increase at or below
  `2.252354e-7 S`, it projects at most `-1.055439254e-5 S`. At the nominal `+7 B` composition it
  projects `-1.122025149e-5 S`.
- **A cleaner QS5-only target:** recover at least 30 net flips on the three-pair support, keep the
  q11-plus-HP4 whole-container delta at `+7 B` or less, and keep Pose leakage at or below
  `2.252354e-7 S`. That projects `-2.054506703e-5 S`. QS4 observed only 17; the extra 13 flips are
  a real missing measurement, not a banked result.
- **Direction:** Pose wins the next major allocation. CP135's canonical Pose term is
  `0.0082945765413 S`, or 69.38% of the entire `0.0119551382782` gap to 0.15. Even perfect Pose
  still leaves 4,319 Seg flips or 5,498 saved bytes owed. Therefore the correct portfolio is a
  large Pose representation move plus a smaller Seg/rate move, not an indefinite sequence of
  `1e-5` semantic micro-edits.

The prior m38 prediction that banked pieces already contain a projected `|Delta S| >= 2e-5`
composition is **REFUTED under honest current-object accounting**. Such a magnitude becomes
available only after an unmeasured recovery, such as raising QS5 from 17 to at least 30 net flips.

The effective frontier remains CP135 at
**`S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`**, archive SHA-256
`6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
The own-vehicle row remains LC2 at
**`S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`**.
This research arm ran no scorer, renderer, Modal job, training job, or `upstream/evaluate.py`, so it
did not achieve goal progress.

## Authority and arithmetic boundaries

`upstream/evaluate.py` computes

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37,545,489`.

`upstream/modules.py` makes the physical coupling explicit: SegNet consumes only the last frame,
while PoseNet consumes both frames after the same 512x384 bilinear resize and RGB-to-YUV6 map.
Therefore frame 0 is Seg-free but not Pose-free, and a frame-1 Seg edit is never presumed
Pose-invisible.

The exact marginal constants used below are:

| Quantity | Value | Status / axis |
|---|---:|---|
| one net Seg flip | `8.477105034722222e-7 S` | derived exactly from `100/117,964,800` |
| one archive byte | `6.658589531221714e-7 S` | derived exactly from `25/37,545,489` |
| break-even density | `0.7854791823 flips/B` | derived; reciprocal is `1.2731082153 B/flip` |
| CP135 canonical Pose term | `0.008294576541331089 S` | derived from report-8dp `d_pose=6.88e-6` |
| CP135 Pose marginal | `602.8035277 S / unit d_pose` | canonical equation `score_marginal_lagrange_multipliers_v1` |
| QS2 exact-object Pose leakage | `+1.126177e-7 S` | measured, matched contest-CUDA T4 component instrument, n600 |

The CP135 absolute score is a report-8dp component reconstruction with a declared
`+/-3.514565443e-6` score bound. The precise component-worker value
`d_pose=6.885642960696714e-6` and 34,970-flip field are not substituted into that absolute score.
They are used only for matched component deltas. This prevents instrument mixing.

## Fresh derivation of the ideal solver

I derived this object from `upstream/evaluate.py`, `upstream/modules.py`, the exact CP135 ZIP, and
its retained runtime before comparing it with the live machinery.

The exact CP135 archive is one ZIP member `p`: `186,152 B` of member payload plus `100 B` ZIP
framing. Its decoded physical allocation is `70,825 B` model, `96 B` residual, and `115,231 B`
RC64 token stream. Same-state lossless work is already mature; a score mover must change the
task-space representation or a scorer-visible integer object.

For a candidate support `A`, frame-1 semantic edit `u`, frame-0 compensation codes `c`, model or
probability state `m`, and complete encoded archive `Z(A,u,c,m)`, the ideal encoder solves the
discrete joint problem

`min 100*DeltaE(A,u,c,m)/117,964,800`

`    + sqrt(10*(d_pose + DeltaP(A,u,c,m))) - sqrt(10*d_pose)`

`    + 25*(bytes(Z)-186,252)/37,545,489`,

subject to strict public parse-back, deterministic render, and exact counted-byte custody.

The important order is part of the solver:

1. choose an edit support and rendered representative;
2. quantize that exact frame-1 object;
3. solve frame-0 compensation for those exact bytes and exact lattice;
4. re-encode the complete archive, including any probability-state response;
5. measure Seg, Pose, and bytes on the same instrument and same object;
6. accept only a super-band complete-S improvement.

No compensation vector is a reusable asset across edit objects. What is reusable is the method,
the retained T4 response data, and the integer solver.

The ideal local subsolver is not another one-shot local Jacobian. For a small active pair set it is
an **exact-instrument integer trust region**: evaluate a bounded lattice stencil of complete
candidate frames on the retained T4 Pose-vector worker, fit the six residual coordinates from
those exact observations, select an integer step, realize it, and shrink or move the region using
the observed/predicted ratio. This is a different formulation from PO1's closed one-shot
local-Jacobian batch. Model-based derivative-free least-squares trust regions are a standard way
to build residual models from expensive function values ([Cartis and Roberts, DFO-GN](https://arxiv.org/abs/1710.11005)); mixed-integer trust regions supply the discrete-neighborhood analogue
([Torres et al., 2024](https://hdl.handle.net/11584/434005)). Here those papers motivate the
optimization form only; Pact admission still comes solely from retained exact worker outputs.

This source-derived solver agrees with the project mission's task-oriented representation premise:
rate is spent on the evaluated task variables, not RGB fidelity. The external rate-distortion
literature independently supports making semantic task distortion explicit rather than treating
pixel reconstruction as the objective ([Guo et al., semantic compression with side information](https://arxiv.org/abs/2208.06094)). It does not supply a contest score or transfer any number.

## Lens 1 — consume-first regrade

Ranks are by probability of producing a lower exact CP135 row soon, not by conceptual interest.

| Rank | Consumed asset | New-law regrade | What survives | Disposition |
|---:|---|---|---|---|
| 1 | QS4 -> QS5 | **UP, but only as an exact-object repair** | q11 measured `+12 B` on the three-pair object; exact field says 17 net flips; stale compensation explains the Pose loss | **FIRED externally / do not duplicate:** QS5 is live; require fresh binding and super-band gate |
| 2 | HP4 order-0 q11 repack | **UP as a composition primitive** | `-5 B` complete-container, exact state restoration, no separate distortion change | **FOLDED into every next score-bearing CP135 child; never fire alone** |
| 3 | RE1 Round 1 | **DOWN as a standalone row; UP as a compensated union atom** | two Seg flips at nominal 0 B are real; signed net is indeterminate because Pose moved one report ULP | **QUEUE only behind exact per-union Pose projection** |
| 4 | QS2 exact-step | **BANKED calibration, not a pointer candidate** | `-32` flips, `+34 B`, Pose `+1.126177e-7 S`, net `-4.374914e-6 S` | **FOLD as control and solver/coder receipt** |
| 5 | GCA1 incidence/divergence and graph energy | **UP as a selector feature, not a payload** | oriented class-edge flow can distinguish Road->Lane from Lane->Road collateral on the exact QS4 field | **FOLD P1/P3 into QS5 re-screen; queue P2 only with a current exact map** |
| 6 | EU3 joint semantic-token/representative/HP3 RDO | **SURVIVES, sharpened** | representative choice and probability object must be optimized with compensation and whole-container bytes | **QUEUE after the micro candidate; do not reuse the direct C1 head** |
| 7 | HY1 C1 carriage | **MECHANISM survives; direct representative dies** | C1 grammar is cheap and expressive; HC1 proved its direct dense representative wrong | **FOLD carriage into a learned scorer-solved representative family** |
| 8 | L28 isolated current-terminal transform | **STILL OPEN, zero counted bytes** | current terminal has never received the isolated same-bytes A/B | **QUEUE after QS5 terminal; no old witness number transfers** |
| 9 | GC21 reference receiver | **DOWN for this frontier** | event-driven stop and reference receiver are useful engineering lessons | **FOLD near-term fire order:** old n120 receiver and stale frontier do not produce a CP135 marginal row |
| 10 | PO1 / PZ4R / ps135 pass 4 / PK2 direct pose objects | **CLOSED at their stated instance or formulation scopes** | their failures define representation and instrument constraints | **DO NOT RETRY unchanged; build a different jointly learned pose representation** |

The consume-first pass changed the plan in three ways: HP4 became a mandatory free composition,
RE1 moved from candidate to compensated atom, and GCA1 moved from a separate research fire to an
internal selector for the exact QS field.

## Lens 2 — ideal versus running solver

| Solver property | Ideal form | Running state after QS4 | Gap grade |
|---|---|---|---|
| objective | complete nonlinear Seg + Pose + exact bytes | complete-score arithmetic exists, but screens still begin on component proxies | `MEDIUM` |
| edit object | immutable fingerprint binds semantic tokens, lattice, rendered frame, and compensation | QS5 is adding exact-object fingerprints after QS4 exposed the omission | `HIGH, actively closing` |
| compensation | solved after every quantization/support change | QS2 proved same-object cancellation; QS4 transferred stale codes to a trimmed object | `HIGH` |
| Pose instrument | exact T4 vectors or a model whose step is above its mismatch floor | QS1 local model transferred on its own object; PO1 showed below-floor one-shot steps can invert by `~112x` | `HIGH` |
| Seg selection | intervention effect after composition, not gross field attribution | QS4 has exact B/H attribution, but 57 modeled flips became 17 realized | `HIGH` |
| rate | whole-container recount after every joint change | CP135 and HP4 do this correctly; micro arms still live in separate child archives | `MEDIUM` |
| uncertainty | super-band gate with matched component precision | band law is now explicit; QS2 remains correctly banked rather than promoted | `LOW` |
| scale of gain | at least one move capable of millipoints | current live line aims at tens of micro-points | `CRITICAL` |

The largest solver error is no longer realization washout. QS4 realized 97.4% of edited cells. The
largest error is choosing and compensating the wrong **composed object**: collateral selection changes
the Seg result, and any object change invalidates the previous Pose solution.

## Lens 3 — eureka search

| Rank | Lead | Why it is plausible now | Exact falsifier | Route |
|---:|---|---|---|---|
| 1 | **Exact-instrument integer trust region for frame-0 compensation** | Only six Pose residuals are scored; QS4 has three active pairs; the retained worker can batch exact affected-pair vectors, avoiding PO1's local-model mismatch and stale transfer | No lattice point in the preregistered radius improves the exact Pose residual enough while preserving Seg and paying bytes | Build only if QS5's fresh local solve still misses the super-band gate |
| 2 | **Jointly learned frame-0 Pose representation with a hard payload budget** | Pose contains `0.0082946 S`; frame 0 is Seg-free; old direct-v6 and post-hoc forms failed because their representative was not jointly learned on this vehicle | At `<=1,000 B` added archive, matched T4 `d_pose` cannot beat `5.8197332e-6` without Seg harm; that is the exact break-even point | Start a new formulation, not a PZ4R/PO1 retry |
| 3 | **QS field flow selection: GCA1 oriented edge divergence + exact intervention response** | QS4 showed 60/76 harmful cells are neighbors and class directions matter; a signed edge-flow feature can target beneficial conversions instead of counting all changed sites alike | On a frozen exact field, the feature does not improve held-out realized flips/B over the current ranker | Fold into the QS5 re-screen, zero payload |
| 4 | **HP4 repack as an always-on entropy-position composition** | It is already a deterministic complete-container `-5 B` win on CP135 and touches a different physical section from the compensation overlay | The changed parent absorbs the five bytes or parse-back differs | Recount inside each child; no dedicated scorer fire |
| 5 | **Isolated L28 same-terminal A/B** | zero counted bytes and current terminal untested; RE1 has released its prior queue condition | same archive bytes plus L28 is non-improving on complete matched Seg/Pose | One exact A/B after QS5, not a broad receiver-treatment sweep |

The discontinuous opportunity is the second row, not the first. At CP135:

| Added pose payload | Required canonical `d_pose` at break-even, Seg fixed | Required Pose reduction |
|---:|---:|---:|
| `256 B` | `6.6001271e-6` | `4.07%` |
| `512 B` | `6.3260656e-6` | `8.05%` |
| `1,000 B` | `5.8197332e-6` | `15.41%` |
| `2,000 B` | `4.8481400e-6` | `29.53%` |
| `5,000 B` | `2.4654023e-6` | `64.17%` |

A `1,000 B` representation that halves `d_pose` projects `-0.0017635663 S` before any Seg
movement. That is roughly two orders of magnitude larger than the QS5 admission threshold. The
existing PK2 `23,384 B` object is beyond even the `~12,456 B` zero-Pose break-even budget and is not
the vehicle to retry.

## Lens 4 — fractal gap table

The same ideal-versus-running mismatch appears at every scale.

| Scale | Ideal object | Current object | Gap / grade | Correction |
|---|---|---|---|---|
| mission | one exact archive below 0.15 | CP135 is 0.0119551 high | `CRITICAL` | stop treating micro-bands as the main campaign |
| score | full-precision same-instrument components | canonical 8dp band for absolute rows; precise worker deltas elsewhere | `MEDIUM` | use worker deltas for selection and exact eval for promotion |
| archive | one jointly optimized ZIP | CP135 plus separate HP4/QS/RE1 children | `HIGH` | rebuild and recount the union, never add isolated byte deltas as verdict |
| section | representation-changing allocation across 70,825 B model, 96 B residual, 115,231 B token stream | same-state lossless race mostly closed | `HIGH` | learn a different task-space state or probability object |
| temporal pair | frame-1 Seg edit and frame-0 Pose antidote solved together | semantic edit first, compensation later | `HIGH` | exact-object joint solve |
| class interface | oriented beneficial conversion with neighbor collateral priced | gross B/H cells and unsigned proximity dominate selection | `HIGH` | signed class-edge flow plus held-out intervention validation |
| pixel/cell | net evaluator flip value | 97.4% edit survival but only 17 net flips on QS4 | `HIGH` | optimize conversions, not survival count |
| integer lattice | exact discrete residual model at the realized point | local continuous/STE model may be below its mismatch floor | `HIGH` | exact-worker stencil and trust-region ratio |
| entropy position | every child absorbs all compatible lossless wins | HP4 remains a separate `-5 B` child | `MEDIUM` | make HP4 an always-on build rung |
| dispatch | one retained multi-candidate component job then one exact row | one-candidate exact fires expose model errors late | `MEDIUM` | batch affected-pair vectors before the public front door |
| handoff | one owner and consumer per surviving action | several older memos still queue overlapping rows | `MEDIUM` | fire order below supersedes those overlaps only for EU4's scope |

The fractal invariant is: **condition on the exact child object before optimizing its parent-scale
compensation or coder**. QS4 violated this at the pair/lattice scale; additive memo arithmetic violates
it at the archive scale; post-hoc pose representations violate it at the vehicle scale.

## Lens 5 — composition inventory now

All rows are projections unless explicitly labeled measured. Every proposed child requires one actual
whole-container build and same-object component measurement.

| Candidate | Exact child specification | Waterfilled arithmetic | Status |
|---|---|---:|---|
| `C0_QS2_HP4_CONTROL` | QS2 exact six-pair lattice plus HP4 order-0 repack | `-32*g + 29*r + 1.126177e-7 = -7.704208771e-6 S` | **FOLDED:** sub-band control only |
| `C1_QS5_Q11_HP4_BASE` | QS4 strict three-pair support, q11 physical overlay, fresh exact-object compensation, HP4 repack | `-17*g + 7*r + 2.252354e-7 = -9.524830487e-6 S` | **FOLDED unless measurement improves a bound:** still sub-band |
| `C2_QS5_HP4_RE1_UNION` | C1 plus RE1 pair-96 event, with compensation re-solved for the full four-pair union; require >=19 net flips, total `DeltaB<=+8`, Pose-term `<=2.252354e-7` | nominal `DeltaB=+7`: `-1.122025149e-5 S`; ceiling `+8`: `-1.055439254e-5 S` | **QUEUED-WITH-FIRE-ORDER after QS5 terminal** |
| `C3_QS5_RECOVER30_HP4` | strict QS5 q11 child with intervention re-screen restoring total net flips to >=30; HP4 absorbed; total `DeltaB<=+7`, Pose-term `<=2.252354e-7` | at thresholds: `-2.054506703e-5 S` | **QUEUED-WITH-FIRE-ORDER if a retained exact-field re-screen supplies 30 flips** |
| `C4_POSE1000_HALF` | a new jointly learned frame-0 representation, `DeltaB<=1,000`, matched T4 `d_pose<=3.44e-6`, no Seg regression | `DeltaS <= -0.0017635663` | **QUEUED as the major route; representation unbuilt** |

`g=8.477105034722222e-7 S/flip` and `r=6.658589531221714e-7 S/B` above.

The composition dependencies are non-additive:

- C0 and C1 are alternative lattice objects, not summable rows.
- RE1's two flips may be composed only after a new union Pose solve; its observed one-ULP Pose move is
  not set to zero by memo arithmetic.
- HP4's five bytes are used only as a build target until the changed complete archive recounts them.
- GCA1 is a selector for C3, not another score delta.
- The old 57-flip QS4 model is not a component. The exact trimmed object measured 17.

### Waterfilled route to 0.15

The CP135 gap is `0.0119551382782 S`. Canonical Pose contributes `0.0082945765413 S`, or
`69.38%` of that gap. Perfect Pose at unchanged bytes still leaves `0.0036605617369 S`, which is
4,319 net Seg flips or 5,498 saved bytes. The allocation conclusion is therefore:

1. direct at least half of the next major representation spend to Pose;
2. preserve one compact Seg/rate line capable of the remaining ~31%;
3. use QS5 as the immediate exact-row attempt, not as evidence that micro-edits can close the target.

This **CONFIRMS** the prior prediction that a eureka route should direct at least 50% of the next
marginal spend to Pose. It **REFUTES** the separate prediction that already-banked pieces honestly
contain a `>=2e-5` composition.

## RECALL EVIDENCE

I searched all seven indexed stores—research, equations, memory, DAG, council, tasks, and docs—with
queries for:

- `QS5 compensation exact edit object QS4 collateral suppression`;
- `pose representation T4 feedback relinearize trust region frame 0`;
- `fractal composition waterfill marginal score archive bytes`;
- `GCA1 graph incidence divergence collateral semantic edit`;
- `L28 current terminal zero byte receiver treatment`;
- task identifiers `#1031` and `#992`.

I also inspected `upstream/evaluate.py`, `upstream/modules.py`, upstream rules 114 and 118, the exact
CP135 archive and runtime receipts, the canonical equation registry, the live hot state, active dispatch
ledger, RE1's terminal full-row erratum, PO1's terminal addendum, and the charters/memos named by EU4.

Findings beyond the charter seeds that changed the plan:

- HP4 had already produced a retained complete-container `-5 B` order-0 repack. It is too small to
  score alone but materially changes the QS5 band arithmetic.
- RE1's terminal row exists. Its two Seg flips were confirmed, but the signed complete delta is
  indeterminate under the 8dp band because Pose moved one report ULP. This changed RE1 from a standalone
  candidate to a compensated-union atom.
- PO1's terminal addendum closes the one-shot local-Jacobian batch at instance/formulation scope and
  names the model-mismatch floor. This changed the eureka solver to an exact-worker derivative-free
  integer trust region, not another local-J retry.
- QS2's q11/dead-zone alternatives change the lattice. The matched QS2 component vector cannot be
  transferred to them. This removed an initially tempting but fake `-11e-6` additive recode claim.
- The exact CP135 result JSON establishes that its absolute score is reconstructed from report-8dp
  components. This forced separate absolute-band and precise-component arithmetic.
- Indexed task search did not expose a one-to-one current row for `#1031` or `#992`; EU4 therefore
  inherits the charter's ownership labels and makes no task-status mutation.

The second pass crossed every surviving lead against the exact archive section it changes; the third
pass crossed every candidate against the compensation-specific and band laws. A final pass found no
additional banked, compatible score-bearing component. That is the dry-round criterion for this memo.

## Source and custody receipts

- CP135 archive inspected at
  `/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/candidates/hp3_step2/split_brotli_per_section_opt_cap1_metadata__rc64/archive.zip`:
  186,252 B, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`,
  ZIP integrity PASS.
- CP135 authority receipt:
  `experiments/results/modal_auth_eval/ddm_cp135_composed_paired_modal_auth_20260810T193605Z_cuda/contest_auth_eval.json`.
- QS2 matched component receipt: `.omx/research/ddm_qs2_r2_admitted_verdict_20260813.md`.
- QS4 exact-field and q11 receipts: `.omx/research/ddm_qs4_collateral_suppression_20260813.md`.
- RE1 terminal receipt: `.omx/research/ddm_re1_round1_full_auth_row_20260813.md`.
- HP4 retained winner:
  `/Volumes/VertigoDataTier/pact/ddm_hp4/retained/candidates/order0/brotli_q11/archive.zip`,
  186,247 B, SHA-256 `3d80f8ab16212abcbec213bf76036c2ca785284cbe924c092cca04549d00f7cc`.

No new payload was materialized by EU4, so the payload-retention rule did not create a new SSD artifact
obligation. No protected source file, upstream file, index state, task row, or shared ledger was edited.

## Verification and landing status

- Independent arithmetic assertions for C0-C3, the 4,319-flip residual, and the 5,498-byte residual:
  PASS.
- `git diff --check` on this memo: PASS.
- CP135 and HP4 archive hashes reverified; both ZIP integrity checks pass.
- Required heading/order check: `## RECALL EVIDENCE`, `## NEXT_IF_RESUMED`,
  `## LIVE-HYPOTHESES`, then `## DEAD-ENDS`: PASS.
- The required serializer was invoked for this one file with post-edit SHA, `base=new`,
  `--no-co-author`, `[no-triality] [p0-ledger-ok]`, and triality `none`.
- **LANDING BLOCKER:** serializer staging stopped before commit because `git add` returned 128:
  `unable to create temporary file: Operation not permitted` and
  `failed to insert into database`. The managed checkout exposes `.git` read-only to this arm.
  `HEAD` remained `e0e261668ad8c30920d94ad5f2ccdc0b1aed44d4`; the index remained empty; this memo is an
  untracked, uncommitted workspace artifact. No fallback commit is claimed.

## Fire order

1. Let the already-live QS5 exact-object repair reach a terminal receipt. Do not create a duplicate arm
   or scorer claim.
2. If QS5's sealed child satisfies C3's 30-flip thresholds, MAIN fires that one child first.
3. Otherwise compile C2, including the RE1 event only after solving compensation for the exact union,
   absorb HP4, and fire only if the build proves the `19 flips / +8 B / Pose <=2.252354e-7 S` gates.
4. In parallel with handoff, design the new <=1,000 B jointly learned frame-0 Pose representation. Do not
   wait for a long chain of micro rows before opening the major route.
5. If a fresh local exact-object solve still fails because of Pose transfer, build one bounded exact-worker
   integer trust-region stencil on the affected pairs. Do not retry PO1's one-shot local-J formulation.
6. Run isolated L28 on the unchanged current terminal only after QS5 releases the scorer lane and only as
   one same-bytes A/B.

## NEXT_IF_RESUMED

- **FIRED / MONITOR-TO-TERMINAL** — owner: `ddm_qs5 current arm owner`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/`; fire trigger: the existing QS5 process emits its
  terminal memo and sealed complete-object receipt; action: ingest it without starting a duplicate run.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN sole scorer-lane router`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_qs5_20260813/`; fire trigger: QS5 seals a receiver-closed C3 child
  with at least 30 net flips, `DeltaB<=+7`, Pose-term `<=2.252354e-7 S`, all payloads retained, and the
  scorer lane clear; action: run one matched dual-axis component row, then exact public-front-door row
  only if the component result remains super-band.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / next QS composition owner`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_eu4_qs5_re1_union_20260813/retained/`; fire trigger: QS5 is terminal
  below the C3 gate, a complete C2 union build proves at least 19 net flips, `DeltaB<=+8`, Pose-term
  `<=2.252354e-7 S`, HP4 parse-back, exact-object compensation binding, and deterministic repeat;
  action: fire the one retained union candidate.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / next pose-representation owner`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_eu4_pose1000_joint_20260813/retained/`; fire trigger: a typed,
  resumable, per-stage-checkpointed jointly learned frame-0 representation seals a receiver-closed
  `DeltaB<=1,000` child with a same-instrument prefire target `d_pose<=3.44e-6` and no Seg harm;
  action: measure retained T4 Pose vectors first and promote only a complete super-band row.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN / exact-worker compensation successor`; consumer store:
  `/Volumes/VertigoDataTier/pact/ddm_eu4_integer_pose_trust_region_20260813/retained/`; fire trigger: QS5's
  fresh exact-object local solve fails Pose admission but the Seg/rate legs still project at most
  `-1e-5 S`; action: batch one preregistered exact affected-pair integer stencil, retain every candidate,
  and take at most one realized trust-region step before remeasurement.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: `current-terminal receiver-treatment successor`; consumer store:
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/stage0_v14/free_receiver_treatments/l28/`;
  fire trigger: QS5 is terminal, no scorer lane is active, and incumbent/L28 children prove identical
  archive bytes plus actual receiver consumption; action: run one matched current-terminal A/B.

## LIVE-HYPOTHESES

- A QS5 q11 child can reach the super-band by recovering 13 additional net flips without increasing its
  seven-byte HP4-composed overhead. This is plausible because QS4's 97.4% edit survival says the carrier
  works; the loss is concentrated in neighbor/class collateral and selection, not decoder washout.
- RE1's two-flip event can become useful inside an exact compensated union. This is plausible because its
  Seg effect and zero-byte closure were real, while the failure was a small Pose interaction whose pair-96
  frame 0 remains Seg-free and can be solved as part of the union.
- An exact-worker integer trust region will outperform stale/local compensation on changed objects. This
  is plausible because it models the six scored Pose residuals from the actual T4 function values and
  never accepts a step below an imported model's mismatch floor.
- A <=1,000 B jointly learned frame-0 representation can reduce current Pose distortion by more than
  15.41% without Seg harm. This is plausible because frame 0 is structurally Seg-free and only six Pose
  outputs matter, while all closed pose attempts used a wrong direct/post-hoc representative or a
  below-floor local model rather than a jointly learned current-vehicle preimage.
- HP4's five-byte win will survive at least one QS child. This is plausible because it repacks the HPAC
  embedding while QS compensation edits the carrier overlay, but the complete changed container must
  still prove the interaction.
- Isolated L28 may move the current terminal at zero counted bytes. This is plausible only because the
  exact current-terminal isolated A/B has never run; the older witness negative prevents a stronger claim.

## DEAD-ENDS

- Bank-only `>=2e-5` composition: closed by exact arithmetic. The honest compatible bank tops out below
  that magnitude; at least 13 new QS5 flips or another unmeasured gain is required.
- Reusing the QS4 57-flip model as evidence: closed because the exact trimmed intervention produced 17
  net flips. It is a falsified predictor, not a banked component.
- Transferring QS2's matched components to q11/dead-zone lattices: closed because quantization changes the
  exact compensation object. Every changed lattice needs its own solve and measurement.
- Firing QS2 plus HP4 as a pointer candidate: closed because the projection is only
  `-7.704208771e-6 S`, inside the declared band.
- Firing QS4 q11 plus HP4 without another gain: closed because the projection is only
  `-9.524830487e-6 S`, also inside the band.
- Treating RE1's two flips as a signed win: closed by its terminal report-ULP Pose movement. It may reopen
  only as an exact pose-projected union atom.
- Retrying PO1's one-shot local-Jacobian batch: closed at the measured instance/formulation; realized Pose
  became 8.257x worse and the predicted steps were below the model-mismatch floor.
- Retrying PZ4R direct-v6: closed at instance scope by `d_pose=0.631014` and a `+2.47154 S` matched-local
  loss. A jointly learned different representation is not that retry.
- Direct dense C1 representative/head: closed by HC1's exact row; C1 grammar carriage survives only with a
  learned scorer-solved representative.
- Post-hoc AR1/AR2 or current-state prediction of the HP3 embedding: closed by HP4's complete-container
  losses. Only its order-0 five-byte repack survives.
- Same-state generic lossless recoding as the major route: closed on the named CP135 coder races. The
  remaining target needs a representation change, not another isolated compressor sweep.
- Decoder-time optimization using scorer or source information: closed by contest availability and rule
  118. Generic receiver algorithms are free, but video-derived learned content is counted and the decoder
  has neither GT nor free scorer weights.
