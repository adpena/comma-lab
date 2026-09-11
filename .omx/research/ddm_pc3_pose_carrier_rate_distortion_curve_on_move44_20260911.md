# ddm_pc3 — the pose carrier's rate/distortion curve on move 44, and the point it found at the far left of it

Tokens: `[no-triality] [p0-ledger-ok]`

Charter: `.omx/research/ddm_pc3_pose_carrier_rate_distortion_curve_on_move44_charter_20260911.md`.
Pointer at spawn and at write: **S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)**, archive sha
`04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e`. MAIN fires; this arm moves nothing.

## The answer first

The charter asked what MORE carrier capacity buys. Measured on move 44: **less than it costs, on every axis that
could be bought**, and the axes split cleanly by arithmetic before any of them needed a sweep. What the arm found
instead is at the OTHER end of the same curve — **the carrier's rate can be cut 160 B at exactly zero distortion
change**, because the shipped CAP1 AR(1)+bias predictor is not the rate-optimal one for the codes it codes.

| curve point | Δ carrier B | Δ archive B | d_pose | d_seg | projected S | net vs move 44 |
|---|---:|---:|---|---|---|---|
| **predictor refit (SEALED candidate)** | **−164** | **−160** | 4.59e-06 *(identical decode)* | 0.00010345 *(identical decode)* | **0.13713836673884064** | **−1.0653743e-04** |
| move 44 (the pointer) | 0 | 0 | 4.59e-06 | 0.00010345 | 0.1372449041713402 | — |
| one dimension's lattice ÷2 (cheapest rung) | +54 … +76 | (not built) | needs ≤ 4.5382e-06 | unchanged | — | break-even needs −1.06 % d_pose |
| global lattice ÷2 | +808 | (not built) | needs ≤ 3.8870e-06 | unchanged | — | break-even needs −15.26 % d_pose |
| global lattice ÷4 | +1,697 | (not built) | needs ≤ 3.1839e-06 | unchanged | — | break-even needs −30.59 % d_pose |
| global lattice ÷8 | +2,603 | (not built) | needs ≤ 2.5395e-06 | unchanged | — | break-even needs −44.63 % d_pose |
| rank ×2 (12 more atoms) | +12,277 | (not built) | — | unchanged | — | **dead by arithmetic**: costs 1.207× the entire pose term |
| rank ×4 | +36,831 | (not built) | — | unchanged | — | dead by arithmetic, 3.6× |
| *(bound)* continuous coefficients, no lattice at all | 0 | 0 | see §4 | unchanged | — | the floor of every lattice rung |

Every byte number in the first block is a REAL encode. Every "needs" is derived from the exact score arithmetic at
move 44's own operating point, not assumed.

## 1. The pose base, on the pointer's own configuration, on this instrument (the pose-base law)

`d_pose = 4.58676349183645e-06`, pose leg `0.006772564870000471`, n600, frozen CPU-torch PoseNet, DALI-lineage GT,
move 44's archive, and — the part that matters — **move 44's OWN cold public decode** as the source of frame 1
(`/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw`, sha `2b762eba…`, verified before
the measurement). Receipt: `base/POSE_BASE_move44.json`, per-pair sha `e0c735f37be75f2b…`.

**That reproduces the T4 print 4.59e-06 at its printed precision.** The sj1 pass-6 chain read `4.770320724991143e-06`
for the same carrier — 3.85 % higher — on an instrument whose odd frames are RE-RENDERED from the token field
(`pose/overlay_pass6`). Compared per pair: **510 of 600 pairs are bit-identical between the two instruments and 90
differ**, all 90 reading higher on the overlay (0.000742 vs 0.000632 summed over those pairs). So the 7.7e-06 pose-leg
spread the last three packets recorded as "the T4 pose-print class" is, on this row, an **overlay-reconstruction
artifact on 90 pairs, not CPU-vs-T4 numerics** — measuring the pose on the row's own cold decode closes it.

Scope: one row, one instrument pair. It is enough to say the gap on THIS row is the overlay's; it is not yet enough to
retire the pose-print class in general. The cheap test for the next carrier pass is to read the pose on the candidate's
own parse-back raw instead of a re-render, and see whether the projection error drops from ~2e-06 to ~1e-08.

## 2. The two walls, DERIVED before a single sweep

The pose term at move 44 is `sqrt(10 · 4.59e-06) = 0.00677495387438173`. Driving d_pose to **exactly zero** gains at
most that, and at the exchange `25/37,545,489 = 6.658589531221714e-07` S/B that is worth at most **10,174.758 B**.

Move 44's carrier, measured (`inspect_cap1` on the shipped blob): 18,915 B in the archive = 190 B of fixed frame +
**12,277 B basis payload** (27,648 five-bit symbols, Huffman, 3.552 bits/symbol) + **6,448 B Rice payload** (7,200
coefficient symbols, 7.164 bits/symbol, all twelve `k = 5`). So one more dozen atoms is ~12,277 B — **1.207× the whole
pose term, at d_pose = 0**. The rank axis is closed by arithmetic, needs no measurement, and a rank change is a
RECEIVER change besides (`up2.CARRIER_DIM`, `inflate.CARRIER_H/W`, `BASIS_BITS`, `COEFFICIENT_BITS` are all receiver
code). The same is true of basis precision and coefficient WIDTH.

What is NOT a receiver change: the twelve coefficient SCALES are float32 words inside the archive, and `max|code| = 154`
of the ±2047 field leaves 13.3× of headroom. Refining the lattice STEP is therefore pure data — the only capacity axis
this arm could have sealed by itself.

## 3. The rate leg — exact, with a byte-exact positive control

`experiments/ddm_pc3_carrier_rate.py`. Control: re-encode the SHIPPED codes through the SHIPPED predictor and require
the Rice payload back byte-for-byte. **PASSED** (51,581 bits, ks all 5). Every rung below is then the encoder's own
output, not an estimate:

| rung | max abs code | Rice payload B | Δ B | Δ S | break-even d_pose | required reduction |
|---|---:|---:|---:|---:|---|---:|
| shipped | 154 | 6,448 | 0 | 0 | — | — |
| global ÷2 | 308 | 7,256 | +808 | +5.380e-04 | 3.886962e-06 | 15.26 % |
| global ÷4 | 616 | 8,145 | +1,697 | +1.130e-03 | 3.183896e-06 | 30.59 % |
| global ÷8 | 1,232 | 9,051 | +2,603 | +1.733e-03 | 2.539489e-06 | 44.63 % |
| dim 3 ÷2 (cheapest) | — | 6,502 | +54 | +3.596e-05 | 4.538189e-06 | 1.06 % |
| dim 9 ÷2 (dearest) | — | 6,524 | +76 | +5.061e-05 | 4.518474e-06 | 1.49 % |
| other ten dims ÷2 | — | — | +65 … +70 | ~+4.4e-05 | ~4.527e-06 | ~1.3 % |

The naive price of a halving is one Rice bit per symbol (900 B globally, 75 B per dimension); the AR(1) predictor
absorbs part of it, which is why the measured numbers are 808 B and 54–76 B. **The per-dimension granularity is the
arm's own contribution to the actuator set**: the scales are independent float32 words and CAP1 carries an independent
Rice `k` per dimension, so a halving can be bought one dimension at a time — 12× finer on the cost axis than a global
halving, and the only granularity whose cheapest step (1.06 % of d_pose) is small enough to be interesting.

## 4. The distortion leg — a BOUND, not a sweep

Refining the lattice — smaller step, more coefficient bits, a better quantiser, in any combination — cannot beat the
**continuous optimum inside the shipped 12-dimensional span**. That optimum is measurable at zero archive cost with the
real renderer and the real frozen scorer, and it bounds the whole family in one measurement. `mode=ceiling` runs
`jg5.refine_pair`'s shape with exactly one thing removed: the projection onto the int12 lattice. It also carries
**pc2 ITEM 3's cure** — the polish neighbourhood is a COEFFICIENT radius, not a lattice one, so it does not shrink as
the lattice refines (that shrinkage is why pc2's ×1/8 rung landed 2,500× worse on pair 0, a SEARCH failure it named as
the only route that could reopen its refusal).

**PROVISIONAL at n = 135 of 600** (strided shards, resumable; the n600 run is still going at write time — see §8):

- mean per-pair gain `4.360e-07`, s.d. `1.6e-06` (a heavy tail), so d_pose `4.5868e-06 → 4.1506e-06`, leg
  `0.0067726 → 0.0064425`, **ΔS −3.300e-04 → payable 495.6 B**; 95 % optimistic edge 706.9 B.
- stop reasons: 117 `no_improving_step`, 18 `converged_below_materiality_floor` — the solver is stopping on physics,
  not on a budget.
- **the gain is not where the mass is**: the 5 highest-d_pose measured pairs carry 50.0 % of the base but only 19.5 %
  of the gain; the top 20 carry 88.2 % of the base and 55.4 % of the gain. The hardest pairs are REACH-limited, not
  lattice-limited — pair 88 (2.128e-04, the single worst) gains **exactly zero** from infinite coefficient precision.

Read against §3: the global ÷2 rung costs 808 B and the ENTIRE lattice family — at infinite precision — is worth
495.6 B (706.9 B at the optimistic edge). **The global rungs are closed by the bound.** The per-dimension rungs at
54–76 B are not closed by it, but they cannot capture more than their share of a gain that the ceiling shows is spread
across all twelve dimensions and all 600 pairs, so the honest statement at n = 135 is that they are *unlikely* and the
n600 number decides. A realized per-rung measurement (`mode=project`, one evaluation per pair per rung, minutes not
hours) is built and waiting on the ceiling rows.

## 5. What the arm actually found: the CAP1 predictor is not the rate-optimal one

The rate leg's SISTER control asks whether an exhaustive fit over the closed legal schema RECOVERS the shipped
predictor. The schema is small and closed — `coefficient_predictor.py` bounds the Q8 factor to [−512, 512] and the bias
to [−16, 16], so one dimension has 1,025 × 33 = 33,825 legal models and the twelve are independent under a per-dimension
Rice parameter. The fit enumerates all of them. It **failed, and failed smaller**:

```
shipped   51,581 Rice bits = 6,448 B   factors [167,178,173,216,205,146,149,146,186,181,233,177]
                                       biases  [ 16, 12, 16, 16,-16,-16, 15,-15,-16, 15, 16, 16]
refit     50,270 Rice bits = 6,284 B   factors [142,132,125, 94,161,145,141,114,141,120,146,148]
                                       biases  [  6,  3,  6, 16, -8, -5,  2, -4,-11, 15, 16,  8]
```

Ten of the twelve shipped biases sit at or one step off the ±16 clamp — the signature of a fit that was clipped rather
than rate-optimised. **−164 B of carrier body at bit-identical codes.**

The predictor is a pure coding choice: `decode_cap1` inverts it and hands the renderer the same canonical CPR1 bytes
either way. Measured: the canonical CPR1 is **byte-identical**, sha `68e4784c4eef9db27ca55475d3e7c9089d8ed06b3b857ae48fb2a7e3696d4bc0`
both sides. So frame 0 is the same image, d_seg and d_pose are the pointer's BY CONSTRUCTION, and the receiver is
untouched — the twelve factors and twelve biases are data fields the shipped receiver already reads and range-validates.
Rule 118 is satisfied in the direction that matters: the video-derived fitted values live in the COUNTED archive, not in
free receiver code (the opposite of the move-41 retraction).

Two container constraints, checked not assumed: `up3.pack_cap1_metadata` packs the factors as 7-bit offsets from their
own minimum and the Rice parameters as ONE bit over a u8 base. The refit spans 67 and 1. Both fit, and `build_archive`
fails closed on either.

**Prior art, searched before claiming novelty.** `ddm_dx1` raced 16 entropy coders on this exact symbol array with the
predictor held FIXED (ceiling −18 B, nothing banked). The loud `CLOSED(FAMILY)` verdicts on "carrier refit" (ra3/rr1,
r012 row 6, es1) are about refitting the COEFFICIENTS/rank — a distortion move — not the AR(1) model. The statements
that "generic CAP1 predictor refitting is closed" (rj2, s1a, ap1) are about *identity reproduction of the shipped
object*: you cannot refit and still claim byte-identity with the shipped archive. That is a control requirement, and
this arm satisfies it the other way round — the archive-identity control below is run with the SHIPPED predictor, so
the delta is attributable. `ddm_jo2` still files the refit as an open LIVE-HYPOTHESIS. Nobody had priced it.

### The candidate

- archive `180,246 B`, sha `145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a`
- **Δ archive −160 B** (the carrier body moves −164 B; Brotli gives 4 B back)
- ΔS `−1.0653743249954742e-04`, projected S `0.13713836673884064`, **5.33× the −2e-5 bar**
- d_seg and d_pose: unchanged, by construction and then by the decode

### Controls, all passed

| control | result |
|---|---|
| shipped-model re-encode reproduces the shipped Rice payload | byte-exact |
| canonical CPR1 identical between shipped and refit carrier | identical, sha `68e4784c…` |
| archive identity rebuild from the body's OWN predictor reproduces move 44's bytes | `04758c0d…`, exact |
| twin encodes | byte-identical, both 180,246 B |
| packed factor span within the 7-bit field | 67 ≤ 127 |
| packed Rice-k span within the 1-bit field | 1 ≤ 1 |
| staging changes exactly `{archive.zip, inflate.py, MANIFEST.sha256}` | yes |
| receiver CODE digest (manifest removed, same function both trees) | equal to the pointer's |

## 6. Pre-registered falsifiers

1. **Pose base.** The base leg on move 44's own configuration must round to the T4 print 4.59e-06 at its printed
   precision. → 4.58676349183645e-06. PASS.
2. **Distortion identity.** The cold n600 public decode of the candidate must be byte-identical to move 44's retained
   raw across all 3,662,409,600 bytes — not "identical outside the carrier frames", identical everywhere. Any single
   differing byte refuses the row, because the whole claim is that d_seg and d_pose cannot move.
3. **Twin identity.** Two independent encodes must agree byte-for-byte, and the identity rebuild (same codes, same
   predictor) must reproduce move 44's archive sha exactly, or the −160 B is the rebuild's and not the refit's.
4. **Ceiling honesty.** The continuous solver must stop on physics (`no_improving_step` /
   `converged_below_materiality_floor`), not on an iteration budget; a ceiling produced by a budget-limited search
   would UNDERSTATE the family's headroom and wrongly close it.
5. **Rate control.** The shipped-model re-encode must reproduce the shipped payload byte-for-byte before any rung
   price is quoted.
6. **Rank arithmetic.** If a rank rung is ever revisited, its basis cost must be compared against the WHOLE pose term
   (10,174.758 B at move 44), not against the marginal 738 S per unit d_pose — the marginal rate overstates the value
   of any large reduction.

## 7. Boundaries — what this arm does NOT claim

- **No score.** No exact eval ran here. `0.13713836673884064` is arithmetic on a MEASURED byte delta and a distortion
  pair that is unchanged by construction and confirmed by a byte-identical decode. `score_claim=false` until MAIN fires.
- **No axis transfer.** Every local pose number is `[macOS-CPU advisory]` and every comparison here is local-to-local.
  The T4 print is quoted for context and never differenced against a local number.
- **The ceiling's scope.** It is the continuous optimum in the BASIN of the shipped lattice point, found by a local
  solver. A different basin could be better; pc1 measured that coarser lattices let GN escape local optima, so basin
  structure is real on this object. It bounds "refine this lattice around where we are", which is what every rung in
  §3 does, and it does not bound a global re-solve from a different start.
- **The ceiling's n.** §4 is PROVISIONAL at n = 135 of 600 and the mean has drifted upward as pairs land
  (1.15e-07 at n = 24 → 3.17e-07 at n = 74 → 4.36e-07 at n = 135). The n600 run is resumable and finishes on its own;
  the verdict on the per-dimension rungs waits for it.
- **The pose-print finding.** One row, one instrument pair. Sufficient to attribute THIS row's 3.85 % gap to the
  overlay re-render; not sufficient to retire the class.
- **No rank/precision/width rung was built.** They are receiver changes and they are closed by arithmetic; neither
  needed a build, and neither could have been sealed by this arm.
- **Two attempts were refused before the decode and both are retained**
  (`candidate/public_refused_native_root_binding/`, `candidate/public_refused_compile_cache_dest/`), with
  `WHY_SUPERSEDED.json` in each. Zero bytes of output were produced under either.

## 8. Custody

Everything under `/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/`:

| what | path |
|---|---|
| pose base, per-pair + receipt | `base/pose_base_move44.npy`, `base/POSE_BASE_move44.json` |
| continuous ceiling, per-pair rows | `ceiling/ceiling_rows_*.jsonl`, `ceiling/CEILING_SHARD_*.json` |
| exact rung rate table | `rate/CARRIER_RATE.json` |
| refit carrier blob + receipt | `refit/refit_cap1_carrier.bin`, `refit/PREDICTOR_REFIT.json` |
| candidate archive | `refit/candidate_archive.zip` (sha `145e02e2…`, 180,246 B) |
| staged candidate tree | `candidate/candidate_runtime/`, `candidate/STAGE.json` |
| cold n600 public decode + raw identity | `candidate/public/RESULT.json`, `candidate/public/output/0.raw` |
| launch manifests and logs | `logs/{base,ceiling,ceiling_smoke,refit,public4}/` |

Producers: `experiments/ddm_pc3_pose_carrier_curve.py` (base / ceiling / reach / project / report),
`experiments/ddm_pc3_carrier_rate.py` (exact rung prices), `experiments/ddm_pc3_predictor_refit.py` (the fit and the
build), `experiments/ddm_pc3_public.py` (stage / cold public proof / smoke),
`experiments/ddm_pc3_ceiling_shards.sh`.

<!-- # FORMALIZATION_PENDING: the curve becomes an equations-leg law once the n600 ceiling closes the lattice family with its exact row -->
