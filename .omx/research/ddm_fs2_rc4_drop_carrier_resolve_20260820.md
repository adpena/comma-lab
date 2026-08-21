# ddm_fs2 — rung 4 dies on RATE, not on pose: the token drop's first-order credit is 91 % illusory

**Task #1173** · **date** 2026-08-20 · **arm** `ddm_fs2` (seventeenth-move candidate, routed by `ddm_fs1` §7)
**Axis** `[macOS-CPU advisory]` for every distortion figure; EXACT for every rate figure derived from
retained decoded fields. `score_claim=false`, `promotion_eligible=false`. No Modal dispatch.
**Store** `/Volumes/APDataStore/pact/ddm_fs2/`

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
`df7fd266…` — UNMOVED by this arm at the time of writing.**

---

## ANSWER FIRST

`ddm_rc4` refused rung 4 (the token drop) on pose by 517×. I re-aimed it at the live `rc2` body with a
carrier re-solve, per the charter. Four things came out, and two of them change the shape of the row.

1. **The drop does not need a receiver change at all.** rc4 framed it as a *skip-decode* rung: the
   receiver stops decoding above `p_max >= tau`. That needs a new branch in
   `decode_production_tokens`. But the identical decoded field is reachable by having the **ENCODER**
   write the model argmax into the token field above tau and coding it with the receiver exactly as it
   ships. On the LIVE rc2 body this **argmax-substitution** form captures **88.2 % of the
   skip-decode credit at u = 7.0** (rv17 W2-F7 re-measurement, receipt 91aeea3653; the
   originally-published 86.6 % mis-attributed its body — see the ledger addendum) and is
   byte-closeable *today* with the proven
   `ddm_jg2_tail_reencode` machinery. Call them PATH A and PATH B; they differ only in rate.
2. **My first instrument was wrong, and my own third control caught it.** The live rc2 decoder codes
   against `corrector.coding_row(state)` — a row the `NativeFreeCorrector` adapts **after** the two
   sha256 checkpoints `decode_production_tokens` records. My first replay priced the pre-corrector
   table, **reproduced both digests exactly**, and still overstated the stream by **2.07 %**
   (929,671 ideal bits against a shipped 910,776). On the retired hv1 body that ratio was 1.00000,
   which is why rc4 never had to see it. With the corrector stepped in the loop the ideal code length
   lands on the coder's own bit count at ratio **1.0000000371**. Digest matching pins the TABLES; only
   the bit total pins the CODER.
3. **The carrier's pose reach does not degrade with the size of the frame-1 damage.** Measured on
   `ddm_jg5`'s retained n600 arrays across a **277× span** of uncompensated damage: Spearman(damage,
   residual) = **0.100**, and the fraction of pairs landing at or below their own base pose after the
   re-solve is **0.509 – 0.678 in every amplitude bin**. In rc4's own amplitude band the median
   residual is a **credit** (−1.28e-07) and **60.5 %** of pairs land at or below base. rc4's 517× was
   measured with the carrier held byte-identical — it priced the *uncompensated* damage, and the
   compensated distribution is bimodal at every amplitude, not just at small ones.
4. **And then the rung died on the axis nobody was looking at.** The modelled ceiling was −1.6058e-03 S
   (459× the bar) at Path B's optimum `u = 7.75`. I built the field and had the **proven jg2 re-encoder
   price it for real**, against a control that reproduced the live token stream **byte-identically**.
   The measured saving is **1,022 archive bytes, not 11,716.7 — 8.72 % of the first-order model.**
   Realised **0.8979 bits per changed token** against a first-order 10.2928. The seg cost is unchanged,
   so with **pose entirely free** the rung is
   **+5.5153e-03 S — a LOSS, 1,576× the bar in the wrong direction.**

**VERDICT: rung 4 is REFUSED on RATE on the live body, before pose is ever reached.** It needs
9,305 B to break even against its own seg cost and it delivers 1,022. That is a **9.10×** shortfall,
and it retires the rung far more cleanly than rc4's pose refusal did — rc4 left "one door left" (the
Schur-compensated reach test). I opened that door: the reach is real (§5), the compensation would
probably work, and **it does not matter, because there is no rate credit to compensate for.**

The mechanism is measured, not guessed: **the model's misses are not payload, they are context.**
Substituting the argmax at a confident disagreement erases the one place the image departs from the
model's prior, and the autoregressive model then predicts the whole neighbourhood worse. The
second-order term recaptures **91.3 %** of the first-order credit. rc4 labelled its ladder
DERIVED-first-order and wrote that "the sign of the second-order term is not established." It is now
established: adverse, and an order of magnitude.

I did not produce a byte-closed exact row. The pointer is unmoved and this arm did not move it. §7 is
what survives.

---

## §1 THE OBJECT, LOCATED (charter leg 1)

| what | where | identity |
|---|---|---|
| rung-4 definition | `experiments/ddm_rc4_drop_ladder.py`; ladder in `DROP_LADDER.json` | "substitute the model argmax wherever `p_max >= tau`", parameterised by `u = -log2(1 - tau)` |
| byte credit (hv1) | `DROP_LADDER.json` `ladder[]` | 17,985.2 B at `u = 7.0` / `p_max >= 0.9921875`, 12,902 token flips |
| the pose refusal | `POSE_RESCORED_DALI.json` + `POSE_LEG.json` (`retained/pose_leg/pose_u7.0.json`, sha `c8d44ba626091576`) | `delta_d_pose` **3.327899e-03** absolute, authority-lineage GT, n=48 stratified-random; `dS_pose` **+0.174319** against an allowed 6.431e-6 → **517.5×** |
| seg amplification | `AMPLIFICATION.json` (sha `77518fb7ac584524`) | A = 0.78475 / 0.79844 / 0.80686 at u = 5.0 / 7.0 / 8.5, n=120 |
| the refusal's own scope | rc4 verdict §VERDICT | `verdict_scope: FORMULATION` — *uncompensated* drop on the hv1 vehicle; explicitly **not** a family kill, with the Schur-compensated reach test named as "the one door left" |

The refusal receipt says in its own words that the carrier was held byte-identical between arms so the
differential isolates frame_1. That is exactly the measurement whose premise this arm re-opens.

**Positive control on my re-derivation of rc4's ladder.** Rebuilding Path A from rc4's retained
`hist_bits` / `hist_bits_disagree` / `hist_n_disagree` histograms reproduces the published
`bytes_saved` **exactly** at every threshold — 29,807.3 / 17,985.2 / 11,901.1 B at u = 5.0 / 7.0 / 8.5,
with flip counts 25,619 / 12,902 / 7,791 matching to the unit. My ladder is rc4's ladder.

---

## §2 PATH A vs PATH B — the drop does not need a receiver change

rc4 §mechanism item 4 reads the rung as a receiver change: a skip-decode branch plus a threshold
exponent in the `RX1_MODEL_HEADER` reserved byte. That is one realisation. There is a second, and it
produces the **identical decoded field**:

| | PATH A (skip decode) | PATH B (argmax substitution) |
|---|---|---|
| who acts | the RECEIVER stops decoding above tau | the ENCODER writes the argmax above tau |
| receiver change | **required** (new branch + header bit) | **none** — ships as-is |
| rate credit | the whole code length above tau | the bits now spent at the confident **disagreements**, minus the cost of coding the argmax there |
| decoded field | identical | identical |
| seg / pose legs | identical | identical |
| byte-closeable today | no | **yes**, via `ddm_jg2_tail_reencode` |

Path B forfeits only the bits held by positions above tau that the model **already gets right** — and
those cost almost nothing each, because they are the confident ones. Measured on hv1 at u = 7.0:
Path A 17,985.2 B, Path B 15,576.5 B, **ratio 0.866**. The forfeited 2,408 B is 19,124 bits spread over
millions of agreeing positions.

This is why rung 4 sat unowned. rc4 priced it as a runtime project; it is an encoder project.

---

## §3 THE INSTRUMENT — and the bug my own control caught

`decode_production_tokens` (`runtime/residual_archive.py:660–690` in the rc2 runtime) computes
`corrected`, digests it, computes `probability = _probability_table(corrected, …)`, digests **that**,
and only then forms `state = corrector.group_state(probability, predicted, flat_positions)` and codes
against `corrector.coding_row(state)`.

My first replay stopped at the digest stage. It passed both digest controls and was wrong:

| | pre-corrector (SUPERSEDED) | corrector in loop (live) |
|---|---:|---:|
| `corrected_quantized_logit_sha256` match | ✅ | ✅ |
| `corrected_cdf_input_sha256` match | ✅ | ✅ |
| corrector kind = receipt's `NativeFreeCorrector` | not checked | ✅ |
| ideal code bits | 929,670.66 | **910,776.03** |
| shipped token bits (113,847 B) | 910,776 | 910,776 |
| **ideal / shipped** | **1.020746** | **1.0000000371** |

The corrector is worth **18,895 bits = 2,362 B** on this body. Pricing a token drop against
probabilities the coder does not use would have overstated the credit at exactly the high-confidence
positions a drop targets. The superseded run is retained under
`superseded_pre_corrector/` with a `WHY_SUPERSEDED.txt`, because it is the receipt for the bug.

**The cure is structural, not vigilance: CONTROL 3.** The summed ideal code length must land within the
RC64 quantisation tax of the receipt's own `decoder_bit_position`. It is in the instrument and it
fail-closes. Note the genus — *the digest checkpoint is not the coding stage*: two sha256 identities
both matched while the priced object was the wrong one
([[measured_object_vs_named_object_20260816]], [[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]).

**Control 2 also fired for real.** The first launch refused with *"corrector is FreeCorrector, receipt
says NativeFreeCorrector"* — the native library was not built in my process and the loader
fail-opened to Python. I built it with inflate.sh's own line (`-ffp-contract=off` is load-bearing) and
re-fired. Without that control the arm would have priced a different coder.

Live instrument receipt `FS2_TOKEN_RD_REPLAY.json`, 425.5 s, all four controls green.

---

## §4 THE LIVE BODY'S RATE STRUCTURE

Re-derived on `ddm_rc2_composed` (`df7fd266…`, 180,456 B), never inherited from hv1.

| quantity | live rc2 | retired hv1 (rc4) |
|---|---:|---:|
| token stream | **113,847 B** | 112,110 B |
| positions | 117,964,800 | 117,964,800 |
| model top-1 error | **0.00193134** (227,830) | 0.00189628 (223,694) |
| ideal code bits | **910,776.03** | 896,876.62 |
| coder efficiency | **1.0000000371** | 0.99999623 |

The live body's ladder, both paths (`FS2_DROP_LADDER.json`; A-prior from rc4's flat measured A):

| u | `p_max >=` | flips | PATH A bytes | PATH B bytes | A net ΔS (pose free) | B net ΔS (pose free) | B B/flip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 0.9687500 | 24,960 | 28,366.5 | 24,131.6 | −2.284e-03 | +5.361e-04 | 0.967 |
| 6.0 | 0.9843750 | 16,837 | 21,284.6 | 18,472.4 | −2.874e-03 | −1.002e-03 | 1.097 |
| **6.375** | 0.9879515 | 14,784 | **19,288.8** | — | **−2.891e-03** (A optimum) | — | — |
| 7.0 | 0.9921875 | 11,915 | 16,310.5 | 14,378.3 | −2.796e-03 | −1.509e-03 | 1.207 |
| **7.75** | **0.9953547** | **9,106** | 13,138.5 | **11,716.7** | — | **−1.606e-03** (B optimum) | **1.287** |
| 8.5 | 0.9972379 | 6,821 | 10,347.1 | 9,333.8 | −2.224e-03 | −1.550e-03 | 1.368 |
| 10.0 | 0.9990234 | 3,655 | 6,123.9 | 5,605.8 | −1.578e-03 | −1.233e-03 | 1.534 |

Exchange rates at the live operating point: **1 archive byte = 6.658589531e-07 S**, 1 net SegNet argmax
flip = 8.477105e-07 S, breakeven **1.273108 B per net seg flip**, and the −3.5e-6 admission bar is
**5.256 bytes**. The live S decomposes rate **81.0 %** / seg 13.6 % / pose 5.4 %, which is why fs1
routed the seventeenth move at the rate axis.

**Rate-leg label: DERIVED-first-order.** Substituting a token perturbs the autoregressive context of
every later position. The realised credit is whatever `ddm_jg2_tail_reencode` measures when it
re-encodes the field for real — that measurement is §6.

---

## §5 THE CARRIER'S REACH IS AMPLITUDE-INDEPENDENT (the finding that re-opens rc4)

`ddm_fs1` warned, correctly, that jg5's compensation factor is a cross-regime bracket and must not be
carried onto a different edit family. So I did not carry it. I measured the **structural** question
instead, which is the one that transfers: *does the carrier's reach degrade as the frame-1
perturbation grows?*

$0 re-read of four retained `ddm_jg5` n600 arrays, content-hashed at read time
(`FS2_CARRIER_REACH.json`). For each edited pair, uncompensated damage `u_i = candidate_i − base_i`,
compensated residual `r_i = refined_i − base_i`.

| uncompensated-damage bin | n | median `u` | median `r` | fraction `r <= 0` | Σu/Σr |
|---|---:|---:|---:|---:|---:|
| [−6.5e-07, 1.69e-04) | 115 | 5.007e-05 | −2.893e-07 | **0.652** | −785.3 |
| [1.69e-04, 8.32e-04) | 114 | 4.082e-04 | −2.079e-07 | **0.596** | 23.60 |
| [8.32e-04, 1.84e-03) | 115 | 1.223e-03 | −4.495e-07 | **0.678** | 15.81 |
| [1.84e-03, 5.49e-03) | 114 | 3.101e-03 | −1.843e-07 | **0.632** | 10.29 |
| [5.49e-03, 9.22e-03) | 57 | 6.742e-03 | −2.332e-09 | **0.509** | 10.81 |
| [9.22e-03, 4.06e-02) | 58 | 1.386e-02 | −2.179e-07 | **0.569** | 6.26 |

* **Spearman(u, r) = 0.100** over a **276.9×** span of median damage.
* The at-or-below-base fraction is **0.509 – 0.678** in every bin — no trend.
* In rc4's own band (2e-03 … 6e-03, median `u` 3.482e-03 against rc4's measured 3.328e-03): n = 124,
  median residual **−1.285e-07 (a credit)**, **60.5 %** at or below base, Σu/Σr = 9.59.

The recovery is **bimodal at every amplitude**. It is a property of the pair — of whether the 6 pose
equations are reachable from that pair's 12 carrier coefficients — not of how hard frame_1 was hit.
That is exactly the structure a per-pair waterfill is built to exploit, and it is why rc4's
uncompensated 517× does not settle the compensated question.

`verdict_scope: FORMULATION` — measured on the jg3 seg-edit family on the br1/rc2 body. It sets the
prior for rung 4; it does not substitute for measuring rung 4.

---

## §6 THE MEASUREMENT THAT KILLED IT

I did not stop at the model. I materialised the substituted token field at `u = 7.75` and had
`ddm_jg2_tail_reencode` — the re-encoder that priced the fifteenth pointer move — encode it for real
against the live rc2 body.

**The control first, because a delta against an unverified encoder is not a measurement.** The unedited
re-encode emitted **113,847 B, sha `b9243abd2e38f9ae…` — byte-identical to the shipped rc2 token
stream**, `byte_identical: true`, `prefix_bytes_matching: 113847`. Its `code_bits` of **910,775.917**
also agrees with my own independent replay's **910,776.034** to **0.117 bits over the whole stream** —
two instruments built from different code paths landing on the same number.

**Then the candidate.**

| quantity | first-order model | **MEASURED** | realised / modelled |
|---|---:|---:|---:|
| token stream | 113,847 → 102,130 B | 113,847 → **112,825 B** | |
| **archive** | 180,456 → 168,739 B | 180,456 → **179,434 B** | |
| **bytes saved** | **11,716.7** | **1,022** | **0.0872** |
| bits per changed token | 10.2936 | **0.8979** | 0.0872 |
| `dS_rate` | −7.8017e-03 | **−6.8051e-04** | |

Both legs, at the model's own optimum, with **pose entirely free**:

| leg | value |
|---|---:|
| rate (**MEASURED, exact archive `stat`**) | **−6.8051e-04** |
| seg (9,106 flips × A 0.80265, rc4 prior) | **+6.1959e-03** |
| **rate + seg** | **+5.5153e-03 — a LOSS, 1,576× the bar in the wrong direction** |

The rung needs **9,305.1 B** to break even against its own seg cost — **8.1749 bits per changed
token** — and delivers **0.8979**. A **9.10× shortfall**, with the pose leg not yet charged.

### Why: the model's misses are context, not payload

The positions a confidence drop targets are exactly the positions where the image departs from the
model's prior. Writing the argmax there erases that departure, and the autoregressive model — which
conditions on the decoded field within the frame and across frames — then predicts the whole
neighbourhood worse. The second-order term recaptures **91.3 %** of the first-order credit.

It is **not** merely the free corrector losing its adaptation: the corrector is worth 18,895 bits
(2,362 B) in total on this body (§3), and the shortfall is 10,694.7 B — **4.5× larger than the
corrector's entire contribution**. The recapture is in the HPAC model's own conditioning, and the
corrector effect can only be a part of it.

`ddm_rc4` labelled its ladder "DERIVED-first-order, not MEASURED-closed-loop" and wrote that "the sign
of the second-order term is not established." **It is now established: adverse, and an order of
magnitude.** That is the load-bearing correction this arm makes to the rung.

### A second threshold — this is a FORMULATION verdict, not an instance

One measured point refutes one row. I encoded a second, 6.3× sparser threshold to find out whether the
realised/modelled ratio is a property of the ladder or of that row.

| u | `p_max >=` | flips | modelled B | **MEASURED B** | realised/model | bits/token | ΔS_rate | ΔS_seg | **rate+seg** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7.75 | 0.9953547 | 9,106 | 11,716.7 | **1,022** | **0.0872** | 0.8979 | −6.805e-04 | +6.196e-03 | **+5.515e-03** |
| 12.0 | 0.9997559 | 1,440 | 2,546.1 | **−37** | **−0.0145** | −0.2056 | +2.464e-05 | +9.849e-04 | **+1.010e-03** |

**At `u = 12.0` the substitution does not merely under-deliver — it COSTS 37 bytes.** The recapture
exceeds 100 %. And the direction is the one the mechanism predicts: the sparser and more confident the
targeted miss, the more informative that departure is as context, so erasing it does more damage than
the bit it was worth. There is no "safe confident tail" to harvest on the rate axis either — the same
shape rc4 found on the seg axis, where A was flat at 0.785–0.807.

Two thresholds spanning 6.3× in flip count, both refused, monotone in the adverse direction. That is
enough to scope this at the FORMULATION level for the whole ladder rather than at one row.

### PATH A inherits the same defect

Skip-decode does not dodge it: the decoder writes the same argmax into the same field, so the
downstream contexts move identically. On the live body at `u = 7.0`, Path A's 16,310.5 B splits into
**14,378.3 B of disagreement bits** (exposed to the recapture) and **1,932.2 B of agreeing-position
bits** (not obviously exposed). Realising the disagreement half at the measured 0.0872 and — generously
— the agreeing half at 100 % gives **3,186.4 B** against the **12,111.6 B** its seg cost demands:
**still 3.80× short.** DERIVED projection, stated with its optimistic assumption; it is a routing
signal, not a measurement of Path A.

---

## §6b THE MODELLED ARITHMETIC (superseded by §6, retained for the audit trail)

Path B at `u = 7.75` on the live body, with A from rc4's flat prior (0.80265 interpolated):

| leg | value | label |
|---|---:|---|
| rate | **−7.8017e-03** | modelled 11,716.7 B; MEASURED value from jg2 pending |
| seg | **+6.1959e-03** | 9,106 flips × A 0.80265 = 7,308.9 net flips, A is an hv1 prior |
| rate + seg | **−1.6058e-03** | **459× the −3.5e-6 bar** |
| pose headroom that gain buys | **2.8212e-06** mean `d_pose` | exact sqrt inverse at the live base 6.37e-06 |

rc4's uncompensated `delta_d_pose` of 3.3279e-03 at 12,902 flips, flip-scaled to 9,106 flips, is
2.3488e-03 — so the **uncompensated** rung needs **832.6×** cancellation here. It will not get it from
a whole-set re-solve: jg5 measured a whole-set recovery of **8.0×** at this exact amplitude.

**The waterfill is the mechanism, not the carrier alone.** Keep pair *i* only if its own joint realised
ΔS is negative; a dropped pair reverts to base tokens and base carrier and costs nothing, exactly as in
jg5. Under the §5 reach measurement, kept pairs are the ones with `r_i <= 0`, so the pose leg is
non-positive by construction and the rate+seg gain scales with the keep fraction *f*:

| keep fraction *f* | rate+seg | multiple of the bar |
|---:|---:|---:|
| 1.000 | −1.6058e-03 | 459× |
| 0.800 | −1.2847e-03 | 367× |
| **0.605** (the §5 measured fraction) | **−9.7153e-04** | **278×** |
| 0.500 | −8.0292e-04 | 229× |
| 0.300 | −4.8175e-04 | 138× |
| 0.200 | −3.2117e-04 | 92× |

This is a **PROJECTION**, not a measurement: it transfers §5's keep fraction, which was measured on the
jg3 edit family, onto rung 4's substitution family. That is precisely the move `ddm_fs1` §4 withdrew
its own headline for. **It is written here as a sizing argument and must not be cited as a result.**
What it does establish honestly is that the arm survives an enormous adverse surprise: the keep
fraction would have to fall below **0.44 %** before the row stops clearing the bar on rate+seg alone.

### The materialised candidate

`FS2_FIELD_u7p75.json`. The substituted token field at `u = 7.75`:

| artifact | bytes | sha256 |
|---|---:|---|
| `retained/fields/tokens_substituted_u7p75.u8` | 117,964,800 | `243076b9dc45646e…` |
| `retained/fields/edits_u7p75.npz` (jg2 `--edits` shape) | 838,435 | `27ef5793053ebd07…` |
| source tokens (rc2 decode r2) | 117,964,800 | `cc10a7b09353c0af…` |

9,106 changed tokens over 600 pairs. The `.npz` is in exactly the `{pair: (384,512) uint8}` shape
`ddm_jg2_tail_reencode --edits` consumes, so the rate is priced by the proven re-encoder and never by
my model.

---

## §7 WHAT SURVIVES

Steps 2-8 of the chain I had queued (decode -> seg -> carrier re-solve -> waterfill -> close ->
advisory -> seal) are **CANCELLED**. They were all downstream of a rate credit that does not exist.
Running them would have measured a pose leg for a candidate that loses 5.5e-03 S before pose is
charged - the means-as-ends failure in its purest form. The honest move is to stop, and to say why in
numbers.

**What is CLOSED.**

* **Rung 4, both realisations, on the live rc2 body.** `verdict_scope: FORMULATION` - the
  confidence-threshold token drop, Path A and Path B, across the ladder on this vehicle's token
  field. Refused on **rate**, before pose. This supersedes rc4's pose-based refusal with a stronger
  and earlier one, and it closes the door rc4 explicitly left open ("the Schur-compensated reach
  test"): the reach is real, and there is nothing to reach for.
* **The compensator-as-gate framing, for this rung only.** rc4's NEXT_IF_RESUMED row 2 said every
  remaining frame-1 lever inherits rung 4's 517x pose exposure and must first characterise the
  compensator. Section 5 characterises it: the reach is amplitude-independent and bimodal, so it is a
  real gate but a passable one. It is not what stopped rung 4.

**What is OPEN, with its number.**

| row | why it survives | reactivation number |
|---|---|---|
| **The recapture law itself** | Any lever that edits the token field pays this second-order cost, and the SIGN of the error depends on the direction of the edit. `ddm_jg5`'s seg edits measured **3.8373** realised bits per changed token — **0.927× of the ACTUAL flat price 4.1379** (Series B, MEASURED: an 8.50% overcharge). The historically-paired **4.718** is jg3's **LogitPrice RANKER** — ordering-only, not a price, per its own docstring — so the old **0.877×** survives only as Series A (instrument ordering, never a trust factor). Rung 4 measures **0.0872x**, and **-0.0145x** at u = 12. jg5 moved tokens *away* from the model's argmax and paid nearly the modelled price; rung 4 moved them *toward* it and got almost nothing back. **That asymmetry is the reusable law and no equation we hold carries it.** (verdict_scope: formulation — −log2p-model pricing of token-field levers on this vehicle; constants restated in Series A/B form per the ERRATUM + ADDENDA below, which remain the derivation record) | price any future token-field lever by a REAL re-encode, never by a `-log2 p` model: vs the ACTUAL price the model overcharges ≤~12% moving away from the argmax (real/price ~0.93x) and overstates the credit ~11x moving toward it (real/model ~0.09x) |
| **Path A's agreeing-position bits** | The only part of the credit not obviously exposed to the recapture: 1,932.2 B at u = 7.0 on the live body, removable only by a skip-decode receiver | they need **12,111.6 B** to cover that threshold's seg cost, so they are **6.3x short on their own**. Dead unless a variant drops agreeing positions WITHOUT touching disagreeing ones - which is `p_max >= tau AND model-agrees`, i.e. exactly the zero-distortion oracle rc4 already proved costs more to signal than it saves (293,148 B identification floor) |
| **Section 5's reach measurement** | A real, transferable, amplitude-independent property of the frame-0 carrier that outlives this rung | any future frame-1 lever that DOES have a rate credit may assume ~60% of pairs are pose-recoverable at any damage amplitude, and should waterfill per pair rather than compensate globally |

**What a successor must NOT do.** Do not re-run rung 4 at another threshold hoping for a better ratio:
two thresholds 6.3x apart both refuse and the trend is monotone adverse. Do not cite the -1.6058e-03
modelled ceiling in Section 6b as a result; it is retained only so the 11.5x correction stays
inspectable.

## §8 BOUNDARIES

No `upstream/` or protected file changed. No Modal dispatch, no paid spend, no contest-CPU or
contest-CUDA row produced, no frozen `#1111` packet custody touched. Every distortion number quoted
here is either retained `ddm_jg5`/`ddm_rc4` measurement or advisory arithmetic over retained arrays.

**The rate numbers are the strongest thing in this memo and the ones the verdict rests on.** They are
exact archive `stat` deltas from a re-encoder whose unedited control reproduced the shipped token
stream **byte-identically** (113,847 B, sha `b9243abd…`), on a coder whose bit total my independent
replay reproduces to 0.117 bits. Nothing about the refusal depends on the advisory pose instrument,
on a GT lineage, or on a sampled subset.

**The seg leg is the one soft number in the refusal**, and it is soft in a way that cannot rescue it.
`A = 0.80265` is an **hv1 prior** from rc4, not measured on the live body. But the refusal survives A
falling to **zero**: at u = 12.0 the substitution *costs* 37 bytes, so `rate+seg > 0` for **any**
non-negative A. At u = 7.75 the rung would need `A <= 0.0882` — a **9.1× drop** from a quantity rc4
measured three times and found flat at 0.785–0.807. That is the falsifier, and it is nowhere near.

`verdict_scope` per claim: §2 **FORMULATION** (the substitution realisation, on this receiver) ·
§3 **INSTANCE** (my own superseded instrument) · §5 **FORMULATION** (the jg3 seg-edit family on the
br1/rc2 body) · §6 **FORMULATION** (rung 4, both realisations, across the ladder on the live rc2
token field) · §6's Path-A row **DERIVED projection**, stated with its optimistic assumption ·
§6b **SUPERSEDED, retained for audit**. No claim here is a family kill — the *paradigm* (buy rate by
spending seg flips) is untouched; what is refused is this rung's exchange rate on this body. None is
a score.

**The pointer did not move and this arm did not move it.** What the arm produced is a measured
refusal one axis earlier and an order of magnitude harder than the one it inherited, plus a
transferable law about which token-field edits the `-log2 p` model may be trusted to price.

---

## §9 OBSERVABILITY SURFACE

**Inspectable per layer** — per-pair `u`-histograms for bits, disagreement bits, counts and
disagreement counts; the full argmax field and quantised u-index field are retained, so any threshold
is a suffix sum and never a re-replay. **Decomposable per signal** — rate, seg and pose legs priced
separately, and Path A vs Path B separated at every threshold. **Diff-able across runs** — the
superseded pre-corrector run is retained beside the live one with its own receipt, so the 2.07 %
correction is inspectable rather than asserted. **Queryable post-hoc** — `FS2_DROP_LADDER.json` carries
all 386 u-bins × 2 paths. **Cite-able** — every input digest is computed at read time, never
hardcoded. **Counterfactual-able** — `field --u-threshold` re-materialises the candidate at any
threshold from retained arrays in seconds.

---

## §10 ARTIFACTS (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_fs2/` — **86 files, 0.991 GB**, every one with bytes +
sha256 in `FS2_RETENTION_MANIFEST.json`. Both refused candidates were BUILT and retained, not merely
priced: the u = 12.0 archive is **180,493 B against the base's 180,456 — visibly larger**, which is the
refusal in a single number a reader can `stat` for themselves.

(Building that manifest caught a transcription error in this memo's own artifact table — I had
attributed `argmax_field.npy`'s sha to the u = 12 substituted field. Corrected above. The manifest is
the authority; a hand-typed sha is not.)

| artifact | bytes | sha256 (first 16) |
|---|---:|---|
| `retained/token_rd/argmax_field.npy` | 117,964,928 | (recorded in `FS2_TOKEN_RD_REPLAY.json`) |
| `retained/token_rd/u_index_field.npy` | 235,929,728 | (recorded) |
| `retained/token_rd/pair_hist_{bits,bits_disagree,n,n_disagree}.npy` | 1,852,928 ea | (recorded) |
| `retained/fields/tokens_substituted_u7p75.u8` | 117,964,800 | `243076b9dc45646e` |
| `retained/fields/edits_u7p75.npz` | 838,435 | `27ef5793053ebd07` |
| `retained/fields/tokens_substituted_u12.u8` | 117,964,800 | `22e04a44b822b386` |
| `retained/fields/edits_u12.npz` | 698,803 | `c8064fda1d2cd431` |
| `retained/token_rd/argmax_field.npy` (live) | 117,964,928 | `93cdf71daedd3950` |
| `reencode/retained/candidate_fs2u7p75.zip` (the refused candidate, built not modelled) | **179,434** | `f2589101377199c7` |
| `reencode/retained/candidate_fs2u12.zip` (the u=12 candidate, **larger than the base**) | **180,493** | `533e182d8c92de6a` |
| `reencode/work/tail_control_600.bin` (byte-identity control) | 113,847 | `b9243abd2e38f9ae` |
| `build/f26_corrector_native.so` | — | `3ce68dfd4056f424` |
| `superseded_pre_corrector/` (the 2.07 % bug's own receipt) | — | with `WHY_SUPERSEDED.txt` |
| `reencode/` (jg2 control + both encodes, incl. per-frame code-bit arrays) | — | control emitted sha `b9243abd2e38f9ae` = the shipped stream |

Receipts: `FS2_TOKEN_RD_REPLAY.json` · `FS2_DROP_LADDER.json` · `FS2_FIELD_u7p75.json` ·
`FS2_FIELD_u12.json` · `FS2_CARRIER_REACH.json` · the jg2 control/encode receipts under `reencode/`.

Landed instruments: `experiments/ddm_fs2_rc2_token_replay.py` (per-pair replay, corrector in loop,
four fail-closed controls) · `experiments/ddm_fs2_drop_ladder.py` (both ladders, per-pair credit, field
+ edits materialisation) · `experiments/ddm_fs2_carrier_reach.py` (the amplitude-dependence
measurement) · `experiments/ddm_fs2_jg5_on_candidate.py` (repoints the whole jg5 toolchain at any
candidate by rebinding the function rather than the constants — built for the cancelled steps 2–8, and
still the right tool for the next frame-1 lever that has a rate credit).

Own-vehicle frontier: **S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED by ddm_fs2.**

---

## ERRATUM (2026-08-20, from rv17 wave-2 W2-F3/W2-F13 + ddm_fs3's source verification)

The recapture-law row prints **"0.877x"** beside `3.8373 / 4.718` — but that quotient is
**0.8133**; the 0.877 is jg2's three-pair figure, a different instrument. Worse, the 4.718
denominator is jg3's `LogitPrice` **ranker** — jg3's own docstring calls it "a RANKER, not a
price." The price jg3 actually charged was a flat **4.1379 bits/token**, and against the measured
3.813767 the true model overcharge is **8.50%** (ratio 0.9217), not 19%. The away-from-argmax
trust factor is therefore a per-instrument RANGE, not one number: **jg5 own 0.813 · jg2 (n=3)
0.877 · jg3 0.773 (flagged `delta_trustworthy: false`) · vs jg3's actual flat price 0.922**.
The toward-argmax ~0.09× and the direction-dependence law itself are UNCHANGED. Receipts:
`.omx/research/ddm_fs3_jg5_real_price_reopen_20260820.md` (03d39d67d9). verdict_scope: instance —
this corrects quoted constants; no verdict in this memo moves.

**ERRATUM ADDENDUM (2026-08-20, rv17 W2-F14 — the corrected range mixed denominators).** The
range "0.77–0.92" published above combines three ratios whose denominator is the DISQUALIFIED
4.718 ranker with one ratio whose denominator is the corrected actual price — four numbers
presented as comparable that answer two different questions. Corrected, as two labeled series:

- **Series A — vs jg3's LogitPrice RANKER (4.718; ordering signal only, NOT prices):**
  jg5 0.813 · jg2 0.877 · jg3 0.773.
- **Series B — vs the ACTUAL flat price (4.1379):** jg5 **0.927** (= the 8.50% overcharge; the
  only pair where numerator and true price are same-object MEASURED). Rescaling series A by
  4.718/4.1379 gives DERIVED values jg2 1.000 · jg3 0.881 (valid only under the assumption those
  ratios share the 4.718 denominator; rv17 round-4 receipt) — so the honest price-based statement
  is: the model overcharges away-from-argmax moves by **at most ~12% and as little as 0%**, not
  "8–23%". Direction-dependence is UNCHANGED (~10× away-vs-toward). The consumer rule is also
  unchanged and is the real law: price by REAL re-encode. verdict_scope: instance — corrected
  constants only.

**LEDGER TERMINATION ADDENDUM (2026-08-20, rv17 W2 round-7 seal refusal — the five
unterminated rows, each given a written ending):**

- **W2-F1 (MED, no shared receipt writer) — CARRIED INTO WAVE 3 BY NAME.** Cure = a canonical
  typed receipt-writer helper (schema-validated BEFORE write, the R15 trailing-comma class made
  unrepresentable). Owner: MAIN. Trigger: before the next `DOC_DIVERGENCE_RECEIPT` append (the
  packet-swap boundary). Interim guard: the R16 post-write schema check, live.
- **W2-F4 (law row lacked verdict_scope) — CURED in place** (scope annotation added to the row,
  this commit).
- **W2-F5 (2,362 B vs 2,354 B) — ADJUDICATED-NO-CHANGE.** The two figures are two INSTRUMENTS:
  18,895 bits (2,362 B) is the corrected replay's corrector worth; 18,834 bits (2,354 B) is the
  superseded pre-corrector run's own reading, preserved in `WHY_SUPERSEDED.txt` as the receipt
  for the bug it found. The 61-bit (0.32%) divergence IS the corrector-table effect that receipt
  exists to document. The memo's 2,362 B is authoritative; both stay, now labeled.
- **W2-F6 (superseded receipt points at live `retained/token_rd/`) — ADJUDICATED-NO-CHANGE.**
  The pointer is a reference into the SHARED retained tree, whose per-file lineage is governed by
  `FS3/FS2_RETENTION_MANIFEST.json`; a superseded receipt citing retained evidence is the
  append-only form working, not a custody claim. Receipts are never edited.
- **W2-F7 (86.6% attributed to the live body) — CURED in place** (line 22, this commit): the
  live-rc2-body figure is **88.2%** (rv17 round-7 re-measurement, receipt 91aeea3653); the
  original 86.6% was measured by this arm on its working state and mis-attributed — same
  wrong-object-constant defect as W2-F3, in the same memo, cured the same way.

verdict_scope: instance — ledger terminations and attribution corrections only; no verdict moves.
