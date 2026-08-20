# ddm_jg5 — the seg edits cost 13x more pose than they buy seg, and the cure is a joint admission

**Date** 2026-08-19 · **Arm** ddm_jg5 · **Charter** re-solve the pose carrier of the composed
sub-0.15 candidate against its OWN edited renders, splice, verify, seal.

**Axis discipline.** Pose: `[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI-lineage GT]`.
Seg: measured on the jg1 DALI per-pair instrument, projected onto the T4 axis by that
instrument's OWN base leg. Rate: EXACT (archive `stat`). `score_claim=false`,
`promotable=false`. Only `upstream/evaluate.py` on contest hardware is a score.

---

## 1. The headline

The composed candidate `ddm_jg4` handed me was never a sub-0.15 candidate. Measured at n600
on the DALI instrument, its d_pose is **0.0032680447584351262** — **467.3x** the br1 pointer's
6.99315662169577e-06. Its composed S is **0.3192**, not 0.156.

The seg token edits bought −0.012847 S on seg and cost **+0.172 S on pose**: a 13.4x loss.
`ddm_jg3` solved seg ALONE, and the direction it moved along is seg-descending but not
pose-null. This is the penalty-vs-projection law
([[penalty-vs-projection-the-seg-pose-coupling-law]]) firing at full scale.

The cure is not a bigger carrier. It is to stop composing two finished candidates and solve
the admission jointly ([[edits_and_drops_are_one_waterfill_solve_jointly_20260819]]).

---

## 2. The control that makes the number mean something

The measurement is only worth its instrument, so both were pinned and both were proved.

| control | result |
|---|---|
| candidate archive sha256 | `4b0dc2724117aa3076a6c271e56f11476120cdce255e42cf6bcfc31b79c253e4`, 181,636 B |
| candidate decode sha256 | `fce05763d7056bd348bb3b5eda57f6497ceafdef83140025289731b6eb4d1b44`, 3,662,409,600 B |
| forward model reproduces the candidate's shipped frame 0 | **byte-exact**, max abs delta 0 over 18,312,048 px |
| the candidate's carrier codes are br1's solved codes | **True** |
| **SAME carrier, SAME GT, BASE odd frames → d_pose** | **6.9931715329553345e-06** |
| br1's banked n600 value | 6.993149776996103e-06 |

The last two rows are the load-bearing control. Swapping only the ODD frames back to the base
decode reproduces br1's banked d_pose to 6 significant figures (2.2e-11 absolute). So the
instrument is br1's instrument, and the entire 467x move is the token edit's effect on frame 1
— not a lineage change, not a decode difference, not a different scorer.

571 of the 573 edited pairs got worse on pose. Two got better.

A second control makes the DROP branch of §4 legitimate rather than assumed: on all 8 unedited
pairs checked (of 27), the odd frame is **byte-identical** between the base decode and the
candidate decode, while every edited pair checked differs. So a pair whose edit is dropped
ships exactly the base frame 1, and pricing it at its base pose value is a measurement, not a
model.

---

## 3. What the carrier CAN recover — and why the old ceiling did not transfer

`ddm_br1` measured a free 2304-dof pose ceiling of 0.7347 and concluded d_pose is not fully
cancellable. Carried onto this arm that constant would have predicted the carrier can remove at
most 27% of the damage, and would have closed the arm before it started.

It is a NEAR-FLOOR constant. br1 measured it where d_pose was already ~7e-6 and the binding
limit was lattice quantisation. At 3.27e-3 the residual is large, the Gauss-Newton step is
well determined, and the carrier drives the residual back down to the same lattice-limited
floor. Measured on the first pairs:

| pair | stale carrier | re-solved | base |
|---|---|---|---|
| 0 | 9.107625e-04 | **3.673352e-07** | 4.517761e-07 |
| 25 | 2.303058e-03 | 3.105076e-06 | — |
| 100 | 2.350858e-03 | **6.224899e-08** | — |
| 10 | 4.180251e-03 | 3.146689e-03 | — |
| 75 | 5.931603e-03 | 2.645546e-03 | — |

Pair 0 lands BELOW its own base value. Pair 10 barely moves. The recovery is **bimodal**, and
that bimodality is the whole design: the pairs that recover are worth keeping, the pairs that
do not are worth dropping. Transferring 0.7347 would have hidden that
([[cross-regime-constant-transfer-genus-finishing-stage]]).

---

## 4. Two truncations in the inherited solver, and a derived rule to replace them

`br1.gn_solve_pair` stops on two limits that read as convergence but are budgets.

**(a) `iterations=6` is a counter.** Measured over this arm's n600 solve, the GN
accepted-step histogram is `[98, 131, 146, 111, 63, 35, 16]` for k = 0…6. The 16 pairs
at k = 6 accepted an improving step on *every* iteration — they were stopped by the
counter, not by the geometry, and all 16 were still improving >2% on their last step.

**(b) The pass structure is GN-then-polish, once.** It never re-enters GN after the ±2
polish has moved the point, so a pair that polish carries into a region where the
Gauss-Newton direction still has slope never gets to use it.

This is the **fifth instance of cross-regime constant transfer**
([[cross-regime-constant-transfer-genus-finishing-stage]]). br1's stopping behaviour was
calibrated where start-d_pose ≈ 1e-5 and the lattice was the binding floor; there a
6-iteration cap never binds. At jg5's start d_pose ≈ 3e-3 the same cap truncates real
mass. It is the same shape as `ddm_up2`'s ±2 search radius — the very defect br1 was
built to escape ([[caps_genus_trajectory_stopping_20260805]]).

The first replacement I wrote was *also* arbitrary (a 0.5%-per-step relative tolerance
and a hand-set budget). Those rows are retained under
`retained/superseded_refine_arbitrary_tolerance/` with a `WHY_SUPERSEDED.txt` and are
merged into nothing. The live rule is **derived end to end**:

| quantity | value | provenance |
|---|---|---|
| `dS/dd_i` | `10 / (2·600·sqrt(10·m))` = 1.043784 | exact derivative of the contest S |
| `m` | 6.374058e-06 | measured mean d_pose of the current waterfill kept set |
| `DELTA_FLOOR_S` | 3.5e-6 / 600 = 5.833333e-09 | measured T4 band, equal-allocated over n |
| `remaining_dd` | `g_k · r/(1−r)`, `r = g_k/g_{k−1}` | each pair's OWN measured decay |
| **stop threshold** | **5.588639e-09** | `DELTA_FLOOR_S / (dS/dd_i)` |

Iterate while `remaining_dd · dS/dd_i > DELTA_FLOOR_S`. `r ≥ 1` means the pair is not
decaying geometrically, no extrapolation is admissible, and the solver keeps going
rather than guessing. Backstop budgets exist only so a non-convergent pair cannot run
forever; a non-empty `gn_iteration_budget` or `outer_round_budget` count in the receipt
would mean the rule never bound, and is reported. No hand tolerance survives anywhere in
the stopping decision.

**The 29 history-length-1 pairs are converged, not early-exited.** All 29 have
`demanded_code_units_max ≥ 0.5`, so the ladder produced a non-empty candidate block that
was realized, evaluated, and rejected by the receiver. None hit the lattice floor.

### 4b. The instrument's batch shape is part of the claim

Measured 2026-08-19 on this forward: it is **deterministic at a fixed batch shape**
(batch-1 repeats are bit-identical) but its **value moves with the shape** —

| pair | stored | batch 1 | batch 8 | batch 32 | spread |
|---|---|---|---|---|---|
| 7 | 1.06770922e-02 | 1.06774140e-02 | 1.06767703e-02 | 1.06770922e-02 | 6.0e-5 |
| 35 | 2.14588989e-03 | 2.14574567e-03 | 2.14588988e-03 | 2.14588989e-03 | 6.7e-5 |
| 100 | 6.22489857e-08 | 6.21033286e-08 | 6.21040435e-08 | 6.22489857e-08 | 2.4e-3 |
| 299 | 1.33437853e-09 | 1.34477475e-09 | 1.33617396e-09 | 1.33437853e-09 | **7.7e-3** |

The solver's rows come from mixed ladder/polish chunk widths, so comparing them against a
base vector measured at one shape is a cross-instrument comparison
([[batch_shape_is_part_of_the_forward_instrument_20260806]]). The final KEEP/DROP
decision is therefore made on per-pair values **re-measured at one declared shape**
(`measure_pose`, batch 8) for both code sets, not on the solver's own rows.

## 5. The joint waterfill

Each pair has exactly two admissible states and both legs of both states are measured:

* **DROP** — frame 1 reverts to the base render and the carrier reverts to br1's codes. The
  pair's pose is its base value and it costs no tokens.
* **KEEP** — frame 1 is the edited render and the carrier is this arm's re-solve.

The pose leg is sqrt-concave, so per-pair pose costs do not add in score units
([[concavity_helps_when_you_pay_the_axis_upward_20260818]]); a fixed-ratio greedy is wrong at
the margin. The admission is swept over a Lagrange multiplier on pose damage and every
candidate subset is scored through the exact formula.

The sweep's decision, on matched-shape per-pair values:

| | value |
|---|---|
| edits admitted | **455 of 573** |
| d_seg (jg1 instrument) | 2.01323e-04 |
| d_seg (T4-projected) | 2.01334e-04 |
| d_pose (DALI, batch 8) | **6.365684e-06** |
| seg leg | 0.020133 |
| pose leg | 0.007979 |
| rate leg (MODELLED) | 0.120214 |
| **S (modelled rate)** | **0.1483263** |
| net vs pointer | −0.007826 |

The operating point moved 6.374058e-06 → 6.365684e-06, a 0.07% change in the derived
threshold — not material, so the stop rule was not re-derived.

Two arithmetic controls anchor the sweep:

| subset | model | measured |
|---|---|---|
| drop everything | 0.15614834772046404 | pointer 0.15615242950573233 |
| keep all 573 | 0.31917539282632396 | 0.3191825455409289 |

(The 4.1e-6 offset on the first row is the pointer's d_seg reported at 8 dp; the 7.1e-6 on the
second is the modelled vs measured token rate. Both are quoted, neither is folded away.)

---

## 6. Results

### 6.1 The n600 carrier re-solve

Two full n600 solves were run on the candidate's own renders: the first with br1's
budget stopping, the second (warm-started from it) with the derived materiality rule.

| | budget-stopped | materiality-stopped |
|---|---|---|
| pairs with new codes | 571 / 600 | 571 / 600 |
| coordinates changed | 6,233 | 6,247 |
| stop reasons | n/a (fixed count) | **600/600 `no_improving_step`** |
| `gn_iteration_budget` hits | — | **0** |
| `outer_round_budget` hits | — | **0** |
| `converged_below_materiality_floor` | — | **0** |
| mean d_pose @ batch 8, all edits kept | 4.167177e-04 | **4.089281e-04** |

The refinement improved 338 of 600 pairs on the solver's own rows. Re-measured at the
matched batch shape, 35 pairs are strictly better, **0 are worse**, and the 565 pairs
whose codes are unchanged measure bit-identical — which is also an independent check of
the fixed-shape determinism claim in §4b.

**The derived rule never bound.** Every one of the 600 pairs stopped because the
receiver refused every proposed step, not because a tolerance or a counter fired. The
materiality floor was therefore never the operative limit — it only removed the
possibility of one. That is the honest reading: the solve is at the GN+polish fixed
point of the shipped basis and lattice, and what remains is a property of the carrier,
not of the search.

Biggest single recoveries from the refinement (prior → refined):
pair 240 2.5345e-03 → 5.4316e-06 (467×), pair 221 2.1402e-03 → 1.0997e-05 (195×).
Both flip DROP → KEEP.

Re-solving the carrier alone, with **all** 573 edits kept, takes d_pose from 3.268e-3 to
4.089e-4 — a 8.0× recovery, but still 58× the pointer, and S is still 0.2023. The
carrier cannot rescue the full edit set. The admission has to do the rest.

### 6.2 The admitted subset, byte-closed

The 455 admitted edits were re-encoded onto the br1 pointer body with
`ddm_jg2_tail_reencode` (`delta_trustworthy=true`, 8,654 tokens changed, **3.8373
measured bits per changed token**), then the mixed carrier was spliced in: this arm's
re-solved codes on the 455 kept pairs, br1's codes on the 145 dropped.

| stage | archive | bytes |
|---|---|---|
| br1 pointer | `44e9e650…` | 176,429 |
| jg4 full-edit composite | `4b0dc272…` | 181,636 |
| jg5 subset body (455 edits, br1 carrier) | `30d372ae…` | 180,580 |
| **jg5 final (subset + mixed carrier)** | **`f3bce5d2…`** | **180,625** |

The carrier splice costs **+45 B**, measured by building, not modelled.

| leg | value | authority |
|---|---|---|
| seg | 0.02013338 | d_seg 2.01333755e-04, T4-projected from the jg1 DALI instrument |
| pose | 0.00797852 | d_pose 6.36568419e-06 `[macOS-CPU advisory, DALI GT, batch 8]` |
| rate | 0.12027077 | 180,625 B — **EXACT** |
| **S** | **0.14838267** | net **−0.00776976** vs the pointer 0.15615243 |

Controls on the final build: the identity rebuild from the body's own codes reproduces
`30d372ae…` at 180,580 B byte-for-byte, and a determinism repeat of the final build is
**byte-identical** at `f3bce5d2…`.

`score_claim=false` — only `upstream/evaluate.py` on contest hardware is a score. The
seal at `/Volumes/APDataStore/pact/ddm_jg5/CANDIDATE_SEAL_jg5.json` (SEAL_VALID) is the
fire-ready object; **MAIN fires**.

### 6.3 d_seg invariance, proved per candidate

`ddm_up3_carrier_splice.build_archive` copies the hpac stream, the semantic stream and the
section tail verbatim and re-encodes only the carrier. `mode=close` byte-diffs all four
sections of every candidate it builds against the body it built from, and refuses the level if
any frame-1 section moved. Measured on **every one of the five levels** built against the
subset body: hpac identical, semantic identical, token tail identical,
`frame1_sections_all_identical = true`. Only the carrier stream moves. The seg leg
therefore cannot move, and this is proved per candidate rather than argued once.

---

## 7. What is owed

* **The end-to-end advisory decode of the final archive is still running** — it is the
  belt-and-suspenders check that one real inflate of `f3bce5d2…` reproduces the composed
  seg and pose. The composed values are assembled from two proved decodes (kept pairs
  ship the candidate's frame 1, dropped pairs provably ship the base frame 1, §2), and
  the token round-trip is proved by the re-encoder's own control, but the final archive
  has not yet been decoded as ONE object. Treat S=0.14838267 as advisory-projected until
  that lands.
* A canonical `tools/build_retention_manifest.py` — br1, jg4 and jg5 have now each hand-rolled
  one ([[least_hand_typing_law_20260815]]).
* The `ddm_rr5` CPR1 lossless rider re-measures itself on whatever body it is handed; its
  −183 B is proven to survive a TAIL edit but explicitly NOT proven across a carrier re-solve
  (`.omx/research/ddm_rr5_rider_prestage_20260819.md`:167-170). It must be re-run on the final
  body, last, and it needs its two-line receiver patch staged with the runtime.
