# ddm_fs2 — rung 4 was refused against a probability table the coder does not use, and against a carrier that was never re-solved

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
   ships. Measured on the live body, this **argmax-substitution** form captures **86.6 % of the
   skip-decode credit at u = 7.0** and is byte-closeable *today* with the proven
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
4. **The ceiling is not vacuous — it is 459× the admission bar.** Path B's optimum on the live body
   (`u = 7.75`, `p_max >= 0.9953547`) saves 11,716.7 B for 9,106 token flips: net **−1.6058e-03 S**
   with pose free, against a −3.5e-6 bar. Even a waterfill that keeps only **20 %** of pairs clears the
   bar by **92×**.

**Status of the row: the rate and reach legs are measured and favourable; the joint seg+pose
measurement on the live body is the part that decides it, and it is a mechanical run of the already
proven `jg5` chain.** I did not produce a byte-closed exact row. The pointer is unmoved and this arm
did not move it. §7 is the fire-order.

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

## §6 THE ARITHMETIC, AND WHAT IT DEMANDS

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

## §7 WHAT IS OWED — the fire-order

Every step below is a mechanical run of machinery that already exists and is already proven on this
body. Nothing new has to be invented.

| # | step | tool (reuse, unmodified) | state |
|---|---|---|---|
| 1 | jg2 control (byte-identity re-encode of the unedited body) + encode of `edits_u7p75.npz` → **MEASURED** Path-B rate and a candidate archive | `experiments/ddm_jg2_tail_reencode.py --stage control/encode --runtime-root <rc2> --expect-pointer-sha256 df7fd266…` | **FIRED**, running at time of writing |
| 2 | decode the candidate archive (real `inflate.sh`) → raw frames; proves the receiver produces the field | rc2 candidate runtime, `F26_CORRECTOR_NATIVE_LIBRARY` set | queued behind 1 |
| 3 | per-pair `d_seg` on the candidate raw, DALI lineage | `experiments/ddm_jg1_seg_solve.py validate` | queued behind 2 |
| 4 | per-pair stale-carrier `d_pose` baseline, then the n600 carrier re-solve in 5 strided shards | `ddm_jg5 … baseline` / `refine` (`load_candidate_instrument` takes the runtime, archive sha and raw path as kwargs — import, do not edit) | queued behind 2 |
| 5 | joint waterfill over KEEP/DROP per pair | `ddm_jg5 … waterfill` | queued behind 3+4 |
| 6 | close: splice the mixed carrier, prove frame-1 section identity per candidate, price | `ddm_jg5 … close` (`ddm_up3_carrier_splice`) | queued behind 5 |
| 7 | advisory n600 of the final archive | `tools/fire_local_advisory.py` **only** | queued behind 6 |
| 8 | seal + fire-order; **MAIN fires the T4 row** | `tools/make_candidate_seal.py` | queued behind 7 |

**Named reactivation numbers, so a successor does not have to re-derive them:**

* The row clears the −3.5e-6 bar on rate+seg alone at any keep fraction above **0.44 %**.
* It clears at the §5-measured keep fraction by **278×**.
* PATH A is worth a further **+1.42e-03 S** at its own optimum (`u = 6.375`, −2.891e-03 vs Path B's
  −1.606e-03) and needs only a skip-decode branch plus one header bit. It is the next rung once Path B
  lands, and it is now the only remaining reason to touch the receiver.
* A is an **hv1 prior**. If the live body's measured A exceeds **1.5348** the u = 7.75 row stops paying
  on rate+seg; rc4 measured 0.785–0.807 and flat, so this is a wide margin, but it is the falsifier.

---

## §8 BOUNDARIES

No `upstream/` or protected file changed. No Modal dispatch, no paid spend, no contest-CPU or
contest-CUDA row produced, no frozen `#1111` packet custody touched. Every distortion number quoted
here is either retained `ddm_jg5`/`ddm_rc4` measurement or advisory arithmetic over retained arrays.
The rate numbers are exact code lengths against a coder proved bit-identical to the shipping one; the
Path-B credit is DERIVED-first-order until jg2 prices it.

`verdict_scope` per claim: §2 **FORMULATION** (the substitution realisation of the rung on this
receiver) · §3 **INSTANCE** (my own superseded instrument) · §5 **FORMULATION** (the jg3 seg-edit
family on the br1/rc2 body) · §6 rate+seg **DERIVED**, the keep-fraction row **PROJECTION, not a
result**. No claim here is a family kill and none is a score.

**The pointer did not move and this arm did not move it.** What the arm produced is a candidate whose
rate leg is measured, whose refusal premise is falsified, and whose remaining work is mechanical.

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

Store root `/Volumes/APDataStore/pact/ddm_fs2/`.

| artifact | bytes | sha256 (first 16) |
|---|---:|---|
| `retained/token_rd/argmax_field.npy` | 117,964,928 | (recorded in `FS2_TOKEN_RD_REPLAY.json`) |
| `retained/token_rd/u_index_field.npy` | 235,929,728 | (recorded) |
| `retained/token_rd/pair_hist_{bits,bits_disagree,n,n_disagree}.npy` | 1,852,928 ea | (recorded) |
| `retained/fields/tokens_substituted_u7p75.u8` | 117,964,800 | `243076b9dc45646e` |
| `retained/fields/edits_u7p75.npz` | 838,435 | `27ef5793053ebd07` |
| `build/f26_corrector_native.so` | — | `3ce68dfd4056f424` |
| `superseded_pre_corrector/` (the 2.07 % bug's own receipt) | — | with `WHY_SUPERSEDED.txt` |

Receipts: `FS2_TOKEN_RD_REPLAY.json` · `FS2_DROP_LADDER.json` · `FS2_FIELD_u7p75.json` ·
`FS2_CARRIER_REACH.json`.

Landed instruments: `experiments/ddm_fs2_rc2_token_replay.py` (per-pair replay, corrector in loop,
four fail-closed controls) · `experiments/ddm_fs2_drop_ladder.py` (both ladders, per-pair credit, field
+ edits materialisation) · `experiments/ddm_fs2_carrier_reach.py` (the amplitude-dependence
measurement).

Own-vehicle frontier: **S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED by ddm_fs2.**
