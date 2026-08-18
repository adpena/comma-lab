# ddm_iv1 — the sa1 inversion: the semantic tensor as a POSE ACTUATOR (2026-08-18)

**VERDICT — the pose mechanism WORKS and is measured; the byte realization does NOT
yet exist; and the honest transfer number is ~7x smaller than the headline.**

Re-solving the already-shipped per-pair `frame_embed` codes in the d_pose-descent
direction reduced REALIZED d_pose by **-90.74%** on 48 seeded-random pairs
(1.8012e-4 -> 1.6676e-5), measured through the exact shipping decode and the frozen
CPU-torch PoseNet, with six passing instrument controls, a passing null control, and
near-zero seg cross-talk (+5 flipped pixels on 12 pairs). Three findings then cut the
headline down, and all three are measured:

1. **Effectiveness collapses as the base error shrinks** (Spearman -0.65). The T4
   instrument's error scale sits in the *worst* band (ratio 0.59-0.72, not 0.093),
   confirmed by a second independent 260-pair scan + 30-pair solve. Band-corrected
   DERIVED T4 projection: **ΔS ≈ -0.0010 to -0.0017**, not -0.0267.
2. **The byte leg is NOT closed.** The 36,040 B semantic body is a hard fixed-offset
   receiver contract; the solve overshoots it by +3 B, and the naive trim buys the
   contract back only by reverting 43 of 44 rows (-90.74% -> **-10.78%**). The
   "~zero byte" premise is **withdrawn**: rANS state cascade made a *single* changed
   row cost +38 B.
3. **A CPU/T4 target question can flip the sign entirely** (§8) and is unresolvable
   locally.

So: a real, well-controlled mechanism with a genuinely large advisory effect, an
unfinished engineering realization, and one cheap T4 eval standing between it and a
verdict. **No exact row moved. Frontier unmoved.**

Axis: `[macOS-CPU advisory, frozen CPU-torch PoseNet, seeded-random pair subset]` ·
`score_claim=false` · `promotable=false`. No Modal, no GPU, no MLX, no lane claim.

---

## 1. The inversion

sa1 measured three mechanistically distinct *lossy* edits of the rr4/cp135 semantic
block and refused all three: each paid **68-512x its rate credit in pose** while
d_seg moved at most +4.8e-6. sa1 read that as a family closure.

Read as an **actuator datasheet** the same measurement says something else: this
block is a knob that moves d_pose hard and moves d_seg almost not at all. sa1 pushed
it in the byte-buying direction, which costs pose. This arm pushes it in the
pose-descent direction.

The charter's premise was that this costs ~no bytes, because the values are already
shipped and only their settings change. **That premise survived in spirit and failed
in detail** — see §7: the count is small (tens of bytes) but the shipped container
imposes a hard *fixed-length* constraint that the solve violates.

## 2. Structural facts established by preflight (measured at source, not assumed)

| fact | value | where |
|---|---|---|
| frame_1 comes from the semantic renderer | `output[2p+1]`; `output[2p]` is the carrier | `cpr1/inflate.py:328,352` |
| the per-pair actuator | `frame_embed = nn.Embedding(600, 8)` — one row per pair | `cpr1/inflate.py:96` |
| its storage | 4-bit signed nibble codes x 8 per-column fp32 scales (0.374-0.518) | `renderer_weight_codec::pack_signed(codes,4)` |
| usable code grid | **[-7, +7]** — `-8` is a RESERVED symbol | `renderer_weight_codec.py:294` |
| shipped code occupancy | [-7, +7]; 8 of 4,800 entries sit at the +7 edge | measured |
| the actuator's stream | W4 stream #1, **1,915 B, rANS-coded** (raw nibbles would be 2,400 B) | `inspect_wans1` |
| semantic body length | **fixed at 36,040 B** — parsed at a fixed offset | `residual_archive.py:194,363-375` |
| escape hatch | bodies tagged `SD1M`/`SM3R` bypass the fixed length | `residual_archive.py:190` |

The per-pair table is **4,800 of the block's 66,339 values (7.2%)**; the other 92.8%
is global renderer weight. These are two different economic objects inside one
section and should not be reasoned about as one.

**Correction to a hypothesis I formed and then falsified.** I initially read sa1's
"damage linear in perturbed mass" as arithmetic from pruning a per-pair table. It is
not: `ddm_sm3_semantic_representation.PRUNE_NAMES` is
`{blocks.1.film.weight, blocks.2.film.weight, blocks.3.film.weight}` — global
per-channel FiLM rows, not `frame_embed`. sa1's linearity finding stands as written.
I record the falsification because the wrong reading was load-bearing for ~20 minutes.

## 3. Instrument controls — all six passed BEFORE any claim

1. **Render bit-identity.** My torch render of frame_1 reproduces the retained
   `0.raw` bytes **exactly** — max abs deviation **0**, 0 mismatched pixels of
   3x874x1164, on 4 seeded-random pairs. The renderer, the bilinear upsample to
   874x1164, the clamp and the round are the shipping path, not a reimplementation.
2. **Byte-path identity.** Re-encoding the untouched actuator through
   `encode_wans1 -> encode_f12_wans_body -> brotli q11 -> pack_rx1_model ->
   deterministic_zip` reproduces the base archive **byte-identically**:
   181,161 B, sha `35ac2b9beb7e6fa8...`. Delta **+0**.
3. **Actuator grid.** `values == codes * scales[None, :]` exactly (deviation 0.0).
4. **Pose instrument reproduction.** 60-pair seeded-random mean base d_pose
   **1.6583e-4** vs the sa1 harness n600 **1.4747e-4** — ratio **1.124** on a
   heavy-tailed distribution. (A 6-pair pilot read 4.5e-5; that was small-sample
   noise, not a discrepancy, and is why the 60-pair leg was run.)
5. **Seg instrument reproduction.** 12-pair base d_seg **4.1072e-4** vs harness n600
   **4.2714e-4** — ratio 0.96.
6. **Null control.** 12 held-out pairs were left untouched; all 12 re-render
   **bit-identically** (deviation 0). `frame_embed` is indexed by pair, so an edit
   cannot leak across pairs — asserted, not assumed.

PoseNet batch-invariance was checked separately: batch-1 vs batch-8 max deviation
**1.79e-7** on a pose vector of norm ~33.7, so batching is numerically safe here.

## 4. Why this is not pk4 (the generalization curse)

pk4 fit a **model** — linear per-pair overlays from Jacobians on train pairs — and
applied it to unseen pairs: 23/23 in-sample winners were 0/23 LOO.

Here the actuator is a per-pair **decision variable** the encoder sets using that
pair's own ground truth, and **every accepted move is a realized measurement** on the
real objective. Predictions only propose; measurements decide. There is no fitted
model to generalize.

That design choice was load-bearing, not ceremonial: **23 of 48 winners were the
exactly-measured best single-code move, beating all 8 lattice proposals.** The
additive superposition prediction is unreliable, exactly as the measured
central-difference asymmetry (~1.0, i.e. the second-order term is as large as the
first) predicted. Trusting the model would have shipped a worse candidate on half
the pairs.

## 5. The measured result

48 solve pairs / 12 held-out, seeded random (never a prefix — m88/m96 prefix-bias
law; pose prefixes measure 2.5-4.2x harder than the population).

| quantity | value |
|---|---|
| solve mean base d_pose | 1.8012e-4 |
| solve mean **realized** d_pose | **1.6676e-5** |
| relative change | **-90.74%** |
| pairs with an accepted improving move | 44 / 48 |
| winners that were the measured single move | 23 / 48 |
| held-out null control | 12/12 bit-identical |
| Delta d_seg (12 pairs) | **+2.1193e-6** = **+5 flipped pixels** (969 -> 974 of 2,359,296) |

The seg delta is **five pixels**. Stated as a ratio it looks like a clean small
number; stated as a count it is honestly near the resolution of a 12-pair sample, so
its relative uncertainty is large (a different 12 pairs could plausibly read +0 or
+12). It is carried in the arithmetic below at face value, which is the conservative
choice, but it is the weakest number in this memo and a 60-pair seg leg would cost
~4 minutes if MAIN wants it tightened.

Per-pair reductions run to 18x (e.g. pair 72: 6.101e-4 -> 1.705e-5; pair 586:
3.209e-6 -> 2.94e-8). Four pairs found no improving move and were left at base.

## 6. THE BINDING CORRECTION — effectiveness collapses at small base error

The aggregate -90.74% is dominated by a hard tail. Binned by base error:

| base d_pose band | n | mean base | mean realized | **realized/base** |
|---|---:|---:|---:|---:|
| **< 1e-5 (the T4-like scale)** | 11 | 4.297e-6 | 3.106e-6 | **0.723** |
| 1e-5 - 3e-5 | 6 | 1.995e-5 | 5.184e-6 | 0.260 |
| 3e-5 - 1e-4 | 12 | 5.556e-5 | 5.259e-6 | 0.095 |
| > 1e-4 (hard tail) | 19 | 4.112e-4 | 3.537e-5 | 0.086 |

Spearman(base, ratio) = **-0.651**. The mechanism is grid resolution: the actuator's
median single-code pose quantum is 0.0141 with a per-pair minimum of 0.00056-0.00194,
so once a pair's residual approaches the grid's reach there is nothing left to buy.

**This matters because the T4 instrument's mean d_pose is 6.88e-6 — squarely in the
worst band.** Quoting -90.74% as a T4 expectation would be the skewed-population
error in new clothes: the population I optimised (CPU errors) is 21.4x harder than
the population that ships.

**A second, independent low-band leg was run to tighten this** (`receipts/lowband.json`,
seed 20260819 — a fresh pair draw). It scanned **260 random pairs** and solved the
lowest-error ones:

- **Instrument control, much tighter than the 60-pair leg:** 260-pair mean base
  d_pose **1.5322e-4** vs harness n600 **1.4747e-4** — agreement to **4%**.
- 43 of 260 pairs (16.5%) sit below 1e-5.
- Solve on the 30 smallest (mean base 3.343e-6): ratio **0.594**, **24 of 30**
  accepted, and **19 of 30 winners were the measured single move** — the model-free
  decision rule mattered even more here than in the main solve (23/48).

| leg | n | mean base d_pose | ratio |
|---|---:|---:|---:|
| main solve, `< 1e-5` band | 11 | 4.297e-6 | 0.723 |
| low-band leg | 30 | 3.343e-6 | 0.594 |
| **pooled** | **41** | 3.599e-6 | **0.635** |

Both legs sample *below* the T4 mean of 6.88e-6, and the collapse trend says harder
pairs do better — so 0.723 (the band nearest the T4 scale) is the conservative
anchor. The two legs differ by more than either's internal precision, so the honest
statement is a **range, not a point**.

**DERIVED band-corrected T4 projection** (label DERIVED, not measured):

| term | conservative (0.723) | pooled (0.635) | low-band (0.594) |
|---|---:|---:|---:|
| T4 d_pose 6.88e-6 -> | 4.974e-6 | 4.369e-6 | 4.088e-6 |
| pose term 0.0082946 -> | 0.0070528 | 0.0066097 | 0.0063938 |
| ΔS_pose | -0.0012418 | -0.0016849 | -0.0019008 |
| ΔS_seg (measured, +5 px) | +0.0002119 | +0.0002119 | +0.0002119 |
| **net (before rate)** | **≈ -0.00103** | **≈ -0.00147** | **≈ -0.00169** |

That is **~294x to ~483x the -3.5e-6 bar**, and **13% to 22% of the 0.00771 gap** to
sub-0.15 — before the rate term, which §7 shows is not yet closed.

## 7. The byte leg — NOT CLOSED. The "~zero byte" premise is qualified.

This is where the arm's engineering premise took real damage, and the damage is
measured, not argued.

**(a) The body length is a hard receiver contract.** `WANS_BODY_BYTES == 36_040` is
parsed at a **fixed offset** (`residual_archive.py:363-375`), so any other length
mis-parses. `frame_embed`'s stream is rANS-coded, so its length moves with the codes:
the 44-row solve produced **36,043 B (+3)** and `rebuild_archive` correctly
**refused to build an archive**. A byte-count assumption would have hidden this.

**(b) The naive trim destroys the result.** Reverting the lowest-pose-gain rows until
the body lands on 36,040 required reverting **43 of 44 rows** — the greedy walk goes
almost all the way back to base before the length matches. Measured outcome:

| after naive trim | value |
|---|---|
| rows still changed | **1 of 44** |
| body bytes | 36,040 (contract met) |
| archive bytes | **181,199 (+38 B, +2.53e-5 S)** |
| solve-subset relative d_pose change | **-10.78%** (was -90.74%) |
| archive sha256 | `83841b969fa4c677...` |

**(c) rANS makes the byte cost quasi-random, not zero.** That +38 B came from a
**single changed row**. rANS is a stateful stream: changing one symbol rewrites every
byte after it, so brotli's response is an essentially unpredictable walk of order
tens of bytes. At ~2% of the band-corrected pose credit this is affordable — but
"~zero byte" is the wrong description and I withdraw it. The binding problem is the
**length**, not the size.

**Routes that remain (neither verified here).**
1. **Length-constrained solve.** Each pair had ~9 verified candidates, several
   improving. Swapping a few pairs to their 2nd-best k perturbs the stream length at
   small pose cost, so hitting exactly 1,915 B with most of the gain intact is very
   likely reachable — but it is a real constrained search, not a formality, and this
   arm did not run it (the per-pair candidate lists were not persisted).
2. **The `SD1M`/`SM3R` tagged container**, which the shipping receiver already
   accepts at **variable length** (`residual_archive.py:190`; sa1 verified parse-back
   deviation 0.0). Price: sa1's measured **+321 B** format penalty (+2.14e-4 S) —
   ~17% of the band-corrected pose credit, i.e. affordable. **Caveat not verified
   here:** SD1M re-quantises with its own scales, so it may not preserve the exact
   solved values; that must be checked before relying on this route.

**Consequence for the verdict:** the pose result is measured and stands; the
**byte-closed candidate does not yet exist**. Any S arithmetic below is therefore a
projection over an unrealised archive, and is labelled as such.

## 8. THE GATE — the CPU/T4 target question, which can flip the sign

On **identical frames** the CPU advisory instrument reads d_pose **21.4x higher**
than the sealed T4 leg (1.4747e-4 vs 6.88e-6); seg differs only 1.44x. I optimised
against the CPU-measured residual. Two hypotheses explain the gap and they predict
**opposite signs**:

- **(A) smooth per-image instrument bias.** It largely cancels in
  `pose(gen) - pose(gt)`, so driving the CPU residual down also drives the T4
  residual down. The projection in §6 holds.
- **(B) a GT-decode difference** (DALI/NVDEC vs PyAV — the named unresolved
  mechanism in CLAUDE.md's apples-to-apples section; the sister failure, PyAV rgb24,
  is documented at ~100x phantom pose). Then the CPU residual is
  `r_true + g` with `|g| >> |r_true|`, and minimising it moves the render toward the
  **wrong target**. T4 pose would **regress**, potentially by ~20x.

Nothing local resolves this: I cannot run DALI without CUDA, and I have no per-pair
T4 residuals. The asymmetry (pose 21.4x, seg 1.44x) is consistent with *both* — pose
is a near-cancelling continuous difference and so amplifies any small GT change,
while seg argmax is robust.

**One cheap T4 eval on one candidate archive resolves it decisively**, and the two
hypotheses are so far apart that the result is unambiguous either way. That is the
single highest-information spend available on this arm.

## 9. Falsifier dispositions (pre-registered, sha `6d5ceb8387a56140...`)

| id | disposition |
|---|---|
| **F1 RESOLUTION** | **did not fire.** 44/48 pairs found an improving realized move; the actuator has fine directions (per-pair minimum quantum 5.6e-4 - 1.9e-3, below the T4 residual norm 6.42e-3). My pre-registration framed this on the *median* quantum (0.0141), which was the wrong statistic — the response magnitudes span ~50x across the 8 dims. |
| **F2 LINEARITY** | **fired, softly, and was absorbed by design.** Central-difference asymmetry ~1.0; 23/48 winners were the measured single move, not a lattice proposal. Because measurements decide, this degraded the gain rather than corrupting the verdict. |
| **F3 T4 TRANSFER** | **fired in a stronger form than written.** I pre-registered a coarseness test on the minimum quantum, which passed. The real transfer risk is the §6 band collapse plus the §8 target question. Both are now quantified. |
| **F4 SEG** | **did not fire.** Delta d_seg +2.1193e-6 (12 pairs, **+5 flipped pixels**) = +2.12e-4 S, ~17% of the band-corrected pose credit. Real and affordable, but measured at a 5-pixel count — the weakest number here. |
| **F5 BYTES** | **fired.** Body 36,043 vs required 36,040; the build fail-closed. §7. |
| charter held-out falsifier | **inapplicable as written, and the correct control was run instead.** There is no fitted model here, so "held-out realized" has no generalization content; the meaningful control is the null control (12/12 bit-identical), which passed. The generalization-shaped risk migrated to §6 (band transfer), where it is measured. |

## 10. What this does NOT establish

1. **No score.** Advisory CPU instrument only. `score_claim=false`.
2. **No n600 solve.** 48 of 600 pairs solved; the projection assumes the per-pair
   reduction factor is band-stationary, which §6 shows it is **not** across bands —
   hence the band correction rather than the aggregate.
3. **No T4 transfer evidence.** §8 is an open gate with an unresolved sign.
4. **The global-tensor arm was not run.** The 92.8% of the block that is global
   renderer weight poses a genuine generalization question and is untouched. It is a
   real follow-on, not a completed leg.
5. Candidate B (encode-side Q3 projection) was not reached; Candidate A did not
   close early.

## 11. Custody (ALWAYS KEEP THE PAYLOAD)

Store `/Volumes/APDataStore/pact/ddm_iv1/`:

- `receipts/PRE_REGISTERED_FALSIFIERS.json` — sha `6d5ceb8387a56140b57ddc55b6992e5a63955fec81c4b0dfaa64a82863861fa3`, written before any probe
- `receipts/render_control.json` · `receipts/actuator_probe.json` ·
  `receipts/actuator_solve.json` · `receipts/finalize.json` · `receipts/trim.json` ·
  `receipts/lowband.json`
- `retained/base_frame_codes.npy` · `base_frame_scales.npy` ·
  `probe_responses.npy` (48x8x2x6 realized pose responses) · `probe_residuals.npy` ·
  `solved_frame_codes.npy` · `final_frame_codes.npy` · `trimmed_frame_codes.npy` ·
  `lowband_scan_base_d_pose.npy` (260-pair base scan) · `lowband_scan_pairs.npy` ·
  `candidate_archive.zip` (the TRIMMED archive, 181,199 B, sha `83841b96...` — the
  -10.78% row, retained because it is a real archive, NOT a candidate worth firing)
- `work/solve60.log` · `work/trim.py` · `work/trim.log` · `work/lowband.py` ·
  `work/lowband_run/` (canonical `tools/launch_detached_process.py` manifest + log)

Measured source: `experiments/ddm_iv1_pose_actuator.py` (stages
`control|probe|solve|finalize`), ruff-clean.

## 12. Routing

**Ordering note:** the byte leg (§7) is unfinished, so there is **no candidate worth
firing on T4 today**. The retained `candidate_archive.zip` is the trimmed -10.78% row
and is NOT the candidate — firing it would spend a T4 eval on a gutted edit and
answer nothing. Steps 1 and 2 are therefore in dependency order, not priority order.

1. **Close the byte leg first — local, $0.** Re-run the solve persisting each pair's
   full verified candidate list, then search alternates for a code set whose
   frame_embed stream lands on exactly 1,915 B. ~10 min of solve plus a cheap search.
   Fall back to the `SD1M` container (+321 B, sa1-measured) if the constrained search
   fails — but verify first that SD1M's re-quantisation preserves the solved values.
2. **Then the T4 leg on that candidate — the decisive spend.** It resolves §8, whose
   hypotheses differ by ~20x in opposite directions, and the answer is unambiguous
   either way. MAIN owns the fire.
3. If §8 resolves as (A): extend the solve to all 600 pairs under the same
   length-preserving constraint. ~10 s/pair local CPU (~100 min for n600, $0); the
   per-pair operation is identical — no new mechanism, only scope.
3. The **global-tensor arm** (92.8% of the block) is the untouched half and carries a
   real generalization question; it deserves its own charter with a true
   solve-vs-heldout split.
4. **Amendment owed to sa1's family verdict**: its reactivation criterion #1
   ("pose-COMPENSATED semantic edit") is written for edits made *for bytes*. This arm
   shows a third door — edits made *for pose at constant bytes* — which its
   three-point refusal table does not cover.
