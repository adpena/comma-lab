# ddm_fs2 — the stale carrier under fs1's new frame 0 is worth 21.5e-06 of `d_pose` for ONE byte; byte-closed, measured, SEALED, and the pointer is UNMOVED until MAIN fires

Arm: `ddm_fs2_carrier_resolve_on_changed_pairs` (2026-09-04). Tokens: `[no-triality] [p0-ledger-ok]`.
Craft contract: `docs/operating_manual_craft_handoff.md`.
Axis of every pose row below: **`[macOS-CPU advisory]`, frozen CPU-torch PoseNet, DALI-lineage GT,
n600 batch 8**. Every byte row is EXACT and device-free. `score_claim=false`, `promotable=false`.

## ANSWER FIRST

1. **The candidate exists, it is byte-closed, and it is sealed.** `archive.zip` sha
   `a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6`, **180,023 B** (fs1's 180,022
   + **1 B**), sealed for `[contest-CUDA T4 n600]` at
   `/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/SEAL_fs2_carrier_resolve_alternation_contest_cuda.json`
   (`SEAL_VALID`, seal sha `532f24824c64f4aa79c69f6bd1a0216afcca5cde9076bc6fff95e5290c9f567e`).
   **MAIN fires; I did not.**

2. **The prior-law prediction HELD, and the falsifier did not fire.** The re-solve buys
   **2.147033e-05** of summed `d_pose` on the 21 stale pairs for **+1 archive byte**. MEASURED at
   n600 batch 8, composed against fs1's own row:

   | candidate | changed pairs | ΔB | Δ`S`_pose | Δ`S`_rate | **net Δ`S`** | projected `S` |
   |---|---:|---:|---:|---:|---:|---:|
   | C — carrier re-solve only | 15 | +1 | −2.155980e-05 | +6.658590e-07 | **−2.089394e-05** | 0.1478423012702068 |
   | **D — + one alternation step (SEALED)** | **15** | **+1** | **−2.281134e-05** | **+6.658590e-07** | **−2.214548e-05** | **0.14784104973157752** |

   D is **1.107× the −2e-05 admit bar** and **3.00×** the conservative two-row 8-dp report bound
   (7.376063e-06, DERIVED via `tac.report_8dp_bounds`, never typed). Both candidates are
   **ADMISSIBLE under the registered law's own rule** (`exchange_ratio_noise_floor_v1`:
   `Q_0.975(dS_b) < 0` over a seeded n600 pair bootstrap). D: point −2.214548e-05, 95% interval
   **[−4.656439e-05, −5.263627e-06]**.

3. **The alternation CONVERGED in one and a half rounds, and that is the mechanism finding.**
   Sweeping all 8 selector modes on all 21 pairs at the re-solved codes moved **exactly one** pair
   (259, mode 4 → 7). Its label change is **byte-free** — MEASURED, not assumed: an already-active
   pair keeps both the active count and the position set, so the blob is 34 B before and after. Its
   carrier then re-solved once more for another 5.834326e-07, also byte-free. A third round has
   nothing left to move on 20 of 21 pairs. **The joint (selector, carrier) optimum on this object is
   essentially a fixed point after ONE exchange.**

4. **The re-solve gain is owned by the pairs the carrier CAN reach, and 64.6% of the set's `d_pose`
   is in pairs it cannot.** Six of the 21 pairs changed **zero** coordinates — including **pair 70,
   which alone carries 1.624967e-04, 52.3% of the whole set's `d_pose`**. That is `ddm_pr1` §5's
   representation limit arriving on a second actuator: the residual points out of the 12-dim span,
   and re-aiming inside the span cannot reach it. The 2.147033e-05 gain is **6.91%** of the set's
   total `d_pose` but **19.5%** of the reachable 1.100e-04.

5. **The 585 untouched pairs are bit-identical at the SCORE level.** `max |Δd_pose| = 0.0` exactly,
   on both candidates. The whole n600 mean move is owned by the 15 pairs the build says it touched,
   and a pair that moved without the build touching it would have refused the compose.

6. **`d_seg` cannot move, proved the same two ways fs1 proved it.** STRUCTURAL: the CAP1 carrier and
   the selector both write `output[2 * frame_ids]` (`f26_inflate.py:133`) — frame `2p` — while SegNet
   scores `x[:, -1, ...]` (`upstream/modules.py:100`) — frame `2p+1`. BYTE-LEVEL: through the
   receiver's own `read_residual_archive`, the semantic section, HPAC model, RC64 token stream,
   residual payload, table codes and compensation blob are **byte-identical**; only the CAP1 carrier
   (and, on D, the 34-B selector blob's contents) differ.

7. **Two constants frozen to one generation would have quietly cost this row, and both are now
   un-frozen.** `ddm_pr1.build_instrument` hardcoded the afr1 body sha — it cannot measure its own
   successor. `ddm_up3.build_archive` hardcoded q=11/lgwin=24, which is a DIFFERENT generation's
   shipped container: on this body it costs **2 bytes MORE** and breaks the byte-identity control the
   whole splice rests on. Both are now named parameters with behaviour-preserving defaults. This is
   `[[binding-instruction-numbers-expire-and-nobody-rederives-them]]` in two live instruments.

**Pointer: UNMOVED.** No exact row was bought by this arm.

## 1. WHAT THIS ARM WAS HANDED

`ddm_fs1` (`.omx/research/ddm_fs1_frame0_selector_reselection_20260904.md`) bought the twenty-fourth
pointer move by re-selecting the frame-0 selector on 21 of 600 pairs, and named its own limit in §8:

> The selector was optimised against a FIXED carrier. `pr1` §12.1 already noted this is a LOWER
> bound on the axis: re-solving each changed pair's carrier against its new frame 0 can only help,
> and neither arm did it.

The staleness is mechanical, not speculative. `ddm_up2.render_frame0` is *carrier render THEN
selector op* (`ddm_up2_shipping_pose_solve.py:397-410`), so a pair whose selector op changed is being
scored on a frame its 12 carrier coefficients were never fitted for. Those 21 code rows are the only
rows in the body demonstrably off their own operating point.

**The scope is a DIFF, and it is derived rather than typed.** A pair is in scope iff the receiver
would apply a different pixel op to its frame 0 than before. Pairs that stayed active at the SAME
mode (60, 116, 241, 373) are not stale and are excluded — otherwise "re-solve on stale rows" would be
measuring rows that are not stale. MEASURED from the two bodies' own selector vectors: 5 active
before, 24 after, **21 changed** — 20 identity→active plus pair 85's 3→identity.

## 2. THE INSTRUMENT AND THE SOLVER — reused verbatim, no mechanism reduction

* **Solver: `ddm_jg5.refine_pair`** — br1's damped Gauss–Newton on the shipped 12-dim basis and the
  shipped signed-int12 lattice, alternated with the ±2 polish under jg5's DERIVED materiality stop.
  `ddm_pr1` §5 established this is the OPTIMAL FORM here: 100.0% of solved pairs demand more than
  `ddm_up2`'s ±2 radius, so running up2 alone would measure the SOLVER's truncation and report it as
  the CARRIER's ceiling ([[caps_genus_trajectory_stopping_20260805]]).
* **Instrument: `ddm_pr1.build_instrument`** — the same assembly that reproduced the contest-CUDA
  pose leg to 0.068%. The only change is the body sha it will gate on.
* **Materiality floor: DERIVED, not set.** `jg5.materiality_dd_threshold` evaluated at the operating
  point the solve aims AT — fs1's own candidate-B n600 batch-8 mean 6.169860284911831e-06 — gives
  `dd_threshold = 5.498392073694633e-09`. Evaluating it at a staler mean would raise the floor by
  `sqrt(stale/target)` and stop the solver early.
* **The one thing this module adds is an explicit PAIR LIST.** pr1's `solve` takes a COUNT and a
  seed, which is right for an unbiased population estimate and wrong here: 21 specific pairs are not
  a sample. An override outside the diff is REFUSED.

## 3. THE RE-SOLVE — 15 of 21 pairs moved, 62 coordinates, every one at a receiver refusal

**MEASURED**, 436.3 s, n600 instrument, one pair at a time (batch 1 by construction).

| | |
|---|---:|
| pairs solved | 21 |
| pairs where any coordinate changed | **15** |
| pairs improved | **15** (zero got worse) |
| coordinates changed, of 252 | **62** |
| summed `d_pose` before → after | 3.108581e-04 → 2.905634e-04 |
| summed gain | **2.029460e-05** |
| n600 MEAN gain | 3.382434e-08 |
| stop reason | **`no_improving_step` 21/21** |

Every row stops because the receiver evaluated a real proposal and refused it — a physical stop at
the basis and the lattice, not a tolerance. **The six pairs that changed nothing are the interesting
ones**: 70, 85, 372, 479, 514, 585. They carry 2.008400e-04 — **64.6%** of the set's `d_pose` — and
the carrier cannot move any of it.

## 4. THE ALTERNATION — one exchange, then a fixed point

`ddm_fs1` §9.3 asked for `selector` ↔ `solve` until neither moves. **MEASURED** (`ddm_pr1 selector`,
all 8 modes × all 21 pairs, at candidate C's re-solved codes, on candidate C's own archive):

| | |
|---|---:|
| pairs whose best mode changed | **1 of 21** (pair 259, mode 4 → 7) |
| gain at that move | 5.930219e-07 (ratio 1.0552) |
| blob length before / after | **34 B / 34 B** — MEASURED, byte-free |
| pairs unchanged | 20 |

Then the carrier re-solved once more for pair 259 at mode 7: 1.074306e-05 → 1.015962e-05, another
**5.834326e-07** for **6 more coordinates and 0 more bytes** (the Rice payload stayed at 78,634
bits). Nothing else in the sweep moved, so a third round has 20 of 21 pairs already at their joint
fixed point.

**Why the label change is free, stated as a mechanism rather than a coincidence.** The blob is
`header + colex-rank(active positions) + 3-bit labels` (`runtime/frame0_selector.py:96-107`). Pair
259 was already active, so neither the active count nor the position set moves; only a 3-bit label
changes inside an unchanged byte budget. **A selector re-aim on an already-active pair is a free
pose actuator.** That is a reusable fact about this receiver, not a fact about this candidate.

## 5. THE BYTE PRICE — anchored, +1 B, and the anchor is checked before the delta is quoted

`ddm_up2.price_full_resolve_bytes` refuses to return a delta unless its own control reproduces the
SHIPPED Rice payload from the SHIPPED codes. **MEASURED on this body:**

| | shipped | candidate C | candidate D |
|---|---:|---:|---:|
| Rice payload bits | **78,628** | 78,634 | 78,634 |
| Rice payload bytes | 9,829 | 9,830 | 9,830 |
| changed coordinates | — | 62 | 67 |
| **archive bytes** | **180,022** | **180,023** | **180,023** |

`control_reproduces_shipped_payload: true`. 62 changed int12 coordinates cost **6 bits**; the
alternation's 5 further coordinates cost **0**. For scale, `ddm_pr1` §8.3 priced a FULL 600-pair
re-solve on the sister body at 6,847 coordinates for +999 bits (+125 B) — the sparse edit here is
0.097 bits per changed coordinate against the full re-solve's 0.146, and the archive delta is 1/125th.

### 5.1 The container is this body's own shape, and that had to be said out loud

`ddm_up3`'s module constants are q=11/lgwin=24 — its OWN generation's shipped shape. On the F26
semantic-joint body `ddm_fs1` §3.3 MEASURED q=9/lgwin=16 as the minimum, with q=11 costing 2 bytes
MORE. Building at up3's default would have produced 180,024 B AND failed the byte-identity control,
so the "+1 B" would have been a mixture of the carrier and the container. The shape is now a named
parameter. **MEASURED container search on candidate D**, 12 encoder-only alternatives:

| ck2 carrier plane 2 | quality | lgwin | archive bytes |
|:--:|---:|---:|---:|
| **false (shipped)** | **9** | **16** | **180,023** |
| false | 9 | 24 | 180,023 |
| true | 9 | 16/24 | 180,025 |
| either | 10 or 11 | 16 or 24 | 180,025 |

The shipped shape is already the minimum. There is no orthogonal container credit to separate out,
so the one-variable comparison holds.

## 6. THE CONTROLS

**Control 1 — container identity.** Re-encoding the SHIPPED codes through `ddm_up3`'s CAP1 forward
chain at this body's shape reproduces fs1's `archive.zip` **bit for bit**: sha
`50fcaf1a…708cf`, 180,022 B, `packed_metadata_identical`, `rice_payload_identical`, Rice bits 78,628.
`run_build` REFUSES if it does not, because without it no byte delta would be attributable. The whole
chain runs backwards — CAP1 canonical blob → packed metadata → DX2 CABAC → RR5 arithmetic basis →
brotli → RX1 → ZIP — through the receiver's own modules, never a recalled copy.

**Control 2 — the no-op detector, through the shipped receiver's own parse.** `read_residual_archive`
on candidate vs base:

| section | C | D |
|---|:--:|:--:|
| semantic blob (sha + bytes) | identical | identical |
| HPAC model | identical | identical |
| RC64 token stream | identical | identical |
| residual payload + table codes + scale | identical | identical |
| compensation blob (`None` both) | identical | identical |
| **CAP1 carrier** | **DIFFERS** | **DIFFERS** |
| selector blob | identical | **DIFFERS** (34 B → 34 B, contents only) |

A selector that moved without being requested REFUSES; a requested selector that does not parse back
to the requested vector REFUSES; a CAP1 carrier that did NOT move when the codes did REFUSES.

**Control 3 — parse-back.** `up3.build_archive` decodes its own written bytes back through the
receiver and refuses unless they yield exactly the requested 600×12 codes. The builder cannot hand
out bytes it has not proved.

**Control 4 — the unchanged-pair control at the SCORE level.** `max |Δd_pose|` over the 585 pairs the
build did not touch is **0.0 exactly**, on both candidates. `run_compose` additionally refuses if ANY
untouched pair moved at all.

**Control 5 — the batch-1 → batch-8 cross-shape step.** The solve screens at batch 1; the score is a
batch-8 population mean. Summed gain screened **2.029460e-05**, measured **2.029398e-05** —
**0.0031% apart**, an order of magnitude tighter than fs1's 0.025% on the same step.

**Control 6 — the projection reproduces the measurement.** Projected net Δ`S` −2.0894606e-05,
measured −2.0893943e-05: relative difference **3.17e-05**, far inside the ±6% exchange-noise
shorthand and, more to the point, inside the registered law's own interval.

**Control 7 — the slack-vs-staleness control, and it is the cleanest result in this memo.**
The "the carrier was stale" claim only means something if the solver finds NOTHING on the same 21
pairs of the body whose frame 0 did NOT move. If it found a comparable amount there, this arm would
be harvesting the shipping chain's own unclaimed slack and the mechanism story would be wrong. So I
ran it: identical solver, identical stopping rule, identical instrument, identical 21 pairs, on the
BASE body (`cbb8d928…`, the shipped selector). **MEASURED, 103.1 s:**

| | base body (frame 0 UNCHANGED) | fs1 candidate B (frame 0 MOVED) |
|---|---:|---:|
| pairs where any coordinate changed | **0 / 21** | **15 / 21** |
| coordinates changed, of 252 | **0** | **62** |
| summed `d_pose` gain | **0.0 exactly** | 2.029460e-05 |
| stop reasons | `no_improving_step` 21/21 | `no_improving_step` 21/21 |

**Zero.** Not "small" — the solver proposed real steps on all 21 and the receiver refused every one.
The base carrier is exactly at its Gauss–Newton fixed point on precisely these pairs, so **100% of
this arm's gain is repair of selector-induced staleness and none of it is pre-existing slack.** This
is a matched control on the SAME pair set, strictly stronger than `ddm_pr1` §8.1's population
estimate (2/200 on a different, random set) — and it reproduces it.

## 7. PER-PAIR RECEIPTS — candidate D vs fs1's candidate B

**MEASURED**, n600 batch 8, one instrument, each row read out of its OWN archive.

| pair | mode | coords | base `d_pose` | candidate `d_pose` | gain | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 95 | 3 | 12 | 8.089830e-06 | 1.837811e-07 | 7.906048e-06 | **44.019×** |
| 555 | 1 | 1 | 3.837826e-06 | 1.274867e-06 | 2.562959e-06 | 3.010× |
| 161 | 5 | 2 | 1.252060e-05 | 1.044038e-05 | 2.080222e-06 | 1.199× |
| 259 | 4→**7** | 8 | 1.203837e-05 | 1.015963e-05 | 1.878740e-06 | 1.185× |
| 504 | 3 | 12 | 3.211123e-06 | 1.658817e-06 | 1.552306e-06 | 1.936× |
| 436 | 2 | 3 | 2.020226e-06 | 4.924729e-07 | 1.527753e-06 | 4.102× |
| 518 | 2 | 12 | 1.500895e-06 | 3.385229e-07 | 1.162372e-06 | 4.434× |
| 5 | 6 | 3 | 1.640698e-06 | 7.329675e-07 | 9.077309e-07 | 2.238× |
| 547 | 1 | 4 | 2.160957e-06 | 1.371159e-06 | 7.897977e-07 | 1.576× |
| 77 | 5 | 1 | 1.819457e-06 | 1.349678e-06 | 4.697786e-07 | 1.348× |
| 221 | 6 | 3 | 1.020972e-05 | 9.902180e-06 | 3.075381e-07 | 1.031× |
| 488 | 2 | 3 | 2.623540e-05 | 2.611991e-05 | 1.154953e-07 | 1.004× |
| 71 | 4 | 1 | 2.549592e-05 | 2.541382e-05 | 8.210080e-08 | 1.003× |
| 173 | 3 | 2 | 1.652183e-06 | 1.585517e-06 | 6.666611e-08 | 1.042× |
| 586 | 6 | 1 | 3.364287e-07 | 2.756120e-07 | 6.081671e-08 | 1.221× |
| **70** | 3 | **0** | 1.624967e-04 | 1.624967e-04 | **0** | 1.000× |
| **85** | 0 | **0** | 1.608251e-05 | 1.608251e-05 | **0** | 1.000× |
| **514** | 7 | **0** | 8.527818e-06 | 8.527818e-06 | **0** | 1.000× |
| **479** | 1 | **0** | 7.010674e-06 | 7.010674e-06 | **0** | 1.000× |
| **372** | 3 | **0** | 2.428734e-06 | 2.428734e-06 | **0** | 1.000× |
| **585** | 6 | **0** | 1.528960e-06 | 1.528960e-06 | **0** | 1.000× |

Summed gain **2.147033e-05**; n600 mean gain **3.578388e-08**.

**Read the last six rows first.** The single largest `d_pose` in the whole set — pair 70, at
1.624967e-04, 52.3% of the set — moved by exactly nothing, and it stopped at `no_improving_step`.
Pair 95, three orders of magnitude smaller, gave up 44× of itself. **The carrier's reach and the
pose leg's mass are anti-correlated**, which is why a per-pair actuator OUTSIDE the 12-dim span
(what `ddm_pr1` §12 called for) remains the live lever and "solve harder" does not.

## 8. THE CLOSING ARITHMETIC

Recomputed from components (`upstream/evaluate.py:90`), never from the 2-dp `Final score` display
(#877). The LEVEL is fs1's contest-CUDA T4 receipt; the DELTA is this arm's advisory
same-instrument difference. That composition is labelled and it is **not a score**.

| | fs1 candidate B (base) | **candidate D** |
|---|---:|---:|
| archive bytes | 180,022 | **180,023** |
| rate leg (exact) | 0.11986926045895953 | 0.11987591904849075 |
| `d_pose` (advisory, n600 batch 8) | 6.169860284911831e-06 | **6.134076407345324e-06** |
| pose leg (advisory) | 0.007854845819563762 | 0.007832034478566424 |
| Δ`S`_rate | — | **+6.658590e-07** |
| Δ`S`_pose | — | **−2.281134e-05** |
| Δ`S`_seg | — | **0** (structural) |
| **net Δ`S`** | — | **−2.214548e-05** |
| projected `S` `[macOS-CPU advisory projection]` | 0.14786319521362173 | **0.14784104973157752** |

**Resolution, DERIVED not typed** (`tac.report_8dp_bounds`): the two rows' pose bounds sum to
6.376063e-06; adding the seg leg's own 8-dp rounding twice gives the conservative 7.376063e-06. On a
`d_seg`-invariant edit the two rows print the same `d_seg`, so that term is the same number twice and
cancels — but only if the T4 does print the same `d_seg`, which is the seal's first falsifier. The
net is **3.47×** the pose-only bound and **3.00×** the conservative one. A landed net Δ`S` inside
`(−7.376063e-06, 0)` is **UNRESOLVED, not a win**, and the seal says so.

**Admissibility, under the registered law rather than a hand-rolled tolerance**
(`exchange_ratio_noise_floor_v1`, seed 20260903, 200 pair resamples, the same draw matrix on both
pose vectors so the pairing survives):

| candidate | point net Δ`S` | 95% interval | half-width | ADMISSIBLE |
|---|---:|---|---:|:--:|
| C | −2.089394e-05 | [−4.650203e-05, −4.879965e-06] | 2.081103e-05 | **yes** |
| **D** | **−2.214548e-05** | **[−4.656439e-05, −5.263627e-06]** | 2.065038e-05 | **yes** |

The half-width is **0.93×** the point estimate — even wider, relative to the win, than fs1's 0.599×.
fs1's fourth anchor to this law already recorded why: the pose mean is owned by a handful of pairs,
and resampling 600 pairs with replacement moves that tail. This arm's win is smaller than fs1's, so
the same dispersion eats proportionally more of it. **The interval still clears, and it clears on the
law's own rule, not on the point estimate.**

## 9. HONEST LIMITS AND `verdict_scope`

* **verdict_scope: INSTANCE.** One archive (`50fcaf1a…`), one adopted code set, one selector label.
  No family claim, no transfer.
* **No score was measured.** Every pose number is `[macOS-CPU advisory]`. Only `upstream/evaluate.py`
  on contest hardware, on these exact bytes, is a score.
* **The advisory→CUDA transfer is unmeasured for THIS edit, and the win is smaller than fs1's.**
  fs1's transfer landed within −2.6e-06 of its projection on a pure-pose edit, which is strong
  evidence but not a bound: −2.6e-06 is **11.7%** of this arm's whole net Δ`S`. The seal
  pre-registers a T4 net Δ`S ≥ 0` as a refutation.
* **The staleness attribution is MEASURED (§6 control 7), and its scope is these 21 pairs.** 0/21
  move on the base body, 15/21 on the re-selected one. It does not follow that every future selector
  change leaves the same amount on the table: the size of the repair depends on how far the chosen
  op moves frame 0, and this arm measured one adopted set.
* **I did not run a full inflate.** The archive is proved through the receiver's own strict
  `read_residual_archive` parse and its own CAP1/DX2/RR5 inverses, not through a 30-minute CPU
  decode. The T4 fire is the first end-to-end execution.
* **`[contest-CPU]` stays RECORD-WITH-REASON** (single-axis waiver in the seal notes): same-object
  pose-only edit, `d_seg` structurally identical, and the prior CPU attempt on this body timed out at
  the 1,800 s inflation budget. No CPU score is inherited.
* **C is retained, built, staged-free and unsealed.** D dominates it by 1.252e-06 at the same byte
  count, on the same instrument.
* **The alternation is declared converged on the evidence I have, not proved.** One more
  selector↔solve round on pair 259 could in principle move again; 20 of 21 pairs did not move on the
  round I ran, and the remaining pair's gains are already down to 5.8e-07.

## 10. THE EQUATIONS LEG (`tac.canonical_equations`)

**Consumed — `exchange_ratio_noise_floor_v1`** (ddm_xr1), through the law's own callables
(`draw_pair_indices`, `bootstrap_mean`, `delta_s_from_components`, `percentile_interval_95`,
`near_win_is_admissible`) rather than a re-implementation. Both candidates ADMIT.

**Appended — nothing.** This arm adds NO anchor. Its edit is the same shape as fs1's fourth anchor
(`fs1_frame0_selector_pure_pose_near_win_pair_bootstrap_20260904`) — pure pose, whole-archive
constant ΔB, Δ`d_seg` ≡ 0 — so a second anchor of the same shape would be a duplicate reading of one
regime, not a new one. The law is consumed and reported; the registry is not padded.

**Cited, and explicitly NOT anchored — `renderer_seg_pose_coupling_shipped_object_v1`.** Its
denominator is Δ`d_seg`, which is identically zero here, so `k` is undefined. Reporting a coupling on
this actuator would be exactly the cross-regime constant transfer that law's own domain block
forbids.

## 11. GESTALT-DELTA

1. **A named limit in a landed memo was worth 21.5e-06 of `d_pose` for one byte, and the matched
   control proves it existed ONLY because fs1 moved frame 0.** fs1 wrote "re-solving each changed pair's carrier can only help, and
   neither arm did it" — a correctly-labelled owed step, in the same document as the pointer move it
   qualified. The lesson is not that fs1 should have done it; it is that **an owed step named inside
   a WIN is the easiest kind to lose**, because the win is what gets read. The control (0/21 on the
   unchanged body, 15/21 on the changed one) turns "we should also re-solve" from a plausible-sounding
   owed step into a measured one: the staleness is CREATED by the actuator, so it recurs every time
   that actuator fires.
2. **Two instruments had a body sha and a container shape frozen to their own generation.** pr1
   could not measure its own successor; up3 would have silently built at a 2-byte-worse container and
   failed the identity control that anchors every byte claim it makes. Neither was a bug when it was
   written. Both became one the moment the pointer moved.
   [[binding-instruction-numbers-expire-and-nobody-rederives-them]] is usually told about a *number
   in an instruction*; here it is a number in a **working instrument**, where being wrong is silent.
3. **A free actuator existed inside a paid one.** The selector blob's length depends on the active
   COUNT and the position SET — never on the labels. Re-aiming an already-active pair is therefore a
   0-byte pose move, and the alternation found one worth 1.18e-06 of Δ`S`. Any per-pair actuator with
   a sparse-rank encoding has this same free interior, and it is worth asking of the next one.
4. **The carrier's reach and the pose leg's mass are anti-correlated, measured twice now.** Six of 21
   pairs moved zero coordinates and carry 64.6% of the set's `d_pose`; pair 70 alone carries 52.3%
   and stopped at `no_improving_step`. `ddm_pr1` §5 called this a REPRESENTATION limit from a
   different direction. Two independent actuators now agree, which promotes it from a finding about
   one solve to a property of this object: **the surviving pose leg is not under-solved, it is
   un-representable**, and only "represent more" or "actuate differently" can reach it.
5. **The alternation converged in one and a half rounds — cheaper than the framing suggested.**
   "Alternate until neither moves" reads like an open-ended loop; it cost one 21-pair sweep and one
   1-pair solve. Naming a loop does not make it long.

## NEXT_IF_RESUMED

0. **State.** All runs complete, control included. Store
   `/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/` (166 GiB free tier), retention manifest
   at `RETENTION_MANIFEST.json` (75 artifacts, sha256 + bytes).

1. **FIRE FIRST — MAIN fires the sealed candidate.** One governed T4 call:

   ```
   .venv/bin/python tools/fire_modal_auth_eval.py \
       --seal /Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/SEAL_fs2_carrier_resolve_alternation_contest_cuda.json \
       --output-dir <dir> --lane-id <lane> --instance-job-id <job>
   ```

   **PROMOTE IFF exact `S < 0.14786319521362173` on `[contest-CUDA T4 n600]`.** Expect
   `d_seg = 0.00020139` unchanged, 180,023 B, `d_pose ≈ 6.13e-06`.

2. **Whenever a per-pair actuator is re-aimed, re-solve the rows it moved.** The control makes this
   a standing order rather than a suggestion: on this object a changed frame-0 op leaves 15 of 21
   affected carrier rows off their fixed point, worth 2.15e-05 of `d_pose` for 1 byte, while the
   untouched rows have exactly nothing left. The step is cheap (436 s for 21 pairs) and it is
   invisible unless someone runs it.

3. **The free-interior question generalises and is cheap.** Any receiver section whose length is a
   function of a COUNT (not of the values) has a zero-byte re-aim inside it. The frame-0 selector is
   one. `residual_archive`'s table codes and the RC64 token stream should be checked for the same
   shape before another byte is spent buying what is already free.

4. **Do NOT re-run a full 600-pair carrier re-solve on this body.** `ddm_pr1` §8.1 measured the base
   carrier at the Gauss–Newton fixed point (2/200 pairs, 1.0045× mean recovery) and §8.3 priced a
   full re-solve at +125 B. At this operating point 125 B costs +8.32e-05 of `S` against a pose gain
   of order 1.8e-08 — strictly worse by three orders of magnitude, and this arm's 579 untouched pairs
   are the same fixed point.

5. **The 64.6% that the carrier cannot reach is the live target, and it is not a carrier problem.**
   Pair 70 (1.62e-04), 85 (1.61e-05), 514 (8.53e-06), 479 (7.01e-06): four pairs, 1.94e-04, immune to
   both the 12-dim span and (per the §4 sweep) all 8 selector modes at their current codes. Reaching
   them needs a third actuator on frame 0, not a better search over the two that exist.

## RECEIPTS

| artifact | what it is |
|---|---|
| `retained/solve_candB/{rows.jsonl,SUMMARY.json}` | the 21-pair Gauss–Newton re-solve, per-pair, resumable |
| `retained/control_base_body/{rows.jsonl,SUMMARY.json}` | the slack-vs-staleness control: the same 21 pairs on the base body, 0/21 move |
| `retained/solve_D_259/{rows.jsonl,SUMMARY.json}` | the alternation's second-round solve on pair 259 |
| `retained/fs2_selector_sweep_on_candC.json` | all 8 modes × 21 pairs at the re-solved codes |
| `retained/codes_fs2_{resolved21,D_alt}.npy` (+ `.json`) | the 600×12 tables + the anchored Rice price |
| `retained/candidate_{C_resolve21,D0_alt_selector,D_alternation}/archive.zip` | every candidate archive |
| `retained/fs2_build_{C_resolve21,D0,D_alternation}.json` | identity control, no-op detector, parse-back, container search |
| `retained/measure_cand{C,D}_*_n600.json` (+ `_payload/`) | the two n600 batch-8 pose rows, per-pair `.npy` with sha256 |
| `retained/fs2_compose_{C_resolve21,D_alternation}.json` | the closing arithmetic + the pair-bootstrap admissibility |
| `retained/fs2_stage_D_alternation.json` | the staged fire tree, the re-pin, the proved two-line receiver diff |
| `fire_runtime_D_alternation/` | the tree MAIN fires (41 files, 878,469 B) |
| `SEAL_fs2_carrier_resolve_alternation_contest_cuda.json` | `SEAL_VALID`, seal sha `532f2482…` |
| `RETENTION_MANIFEST.json` | 75 artifacts, sha256 + bytes (ALWAYS KEEP THE PAYLOAD) |
| `logs/{solve_candB,build_C,measure_C,sweep_C,solve_D_259,measure_D,control_base}/` | launch manifests, run logs, `safe_run` status |

Code: `experiments/ddm_fs2_carrier_resolve_on_changed_pairs.py` (solve / codes / build / compose),
`src/tac/tests/test_ddm_fs2_carrier_resolve_on_changed_pairs.py` (28 tests), plus the two
un-freezing changes to `experiments/ddm_pr1_pose_resolve_on_renderer_change.py`
(`--expect-archive-sha256`) and `experiments/ddm_up3_carrier_splice.py` (`container_options`).
Commit `500189019`.

## Own-vehicle frontier

**fs1 S 0.14786319521362173 @ 180,022 B `[contest-CUDA T4 n600]` — UNMOVED by this arm.**
Sealed candidate D, projected `S` **0.14784104973157752 @ 180,023 B**
`[macOS-CPU advisory projection]` — **not a score** until the T4 row lands.
