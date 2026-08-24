# ddm_dg2 — the diagonal cell is ENTERED: the control passes byte-identically, and jf1's "positive control" was never a control

**Date:** 2026-08-24
**Arm:** `ddm_dg2_diagonal_reentry`
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`
**Score claim:** false · **Promotion eligible:** false · **Scorer ran:** false

STORES CONSULTED: `.omx/research/ddm_fb1_sub012_feasibility_bound_20260823.md` (commit `9c137a91ed`,
the routing memo + the 42,382 B demand) · `ddm_sy2_composition_synergy_deep_pass_20260823.md`
(commit `fe2ba12dc2`, the object-change law + its epoch-24 in-flight figure) ·
`.omx/research/ddm_ar1b_archive_residue_purchase_20260822.md` (commit `e864cb4ab4`, the residue
census, opened at commit) · `.omx/research/ddm_w72_distortion_advisory_20260823.md` (commit
`637af0c8c1`, the renderer refusal, opened at commit) · `ddm_tx1_toolbox_crosswalk_20260819.md` §0
(exchange rate 6.658590e-07 S/B — **CITED, not re-derived**) · commit `db153b5073` (the
bidirectional-control precedent, *"both control directions executed"*) ·
`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/**` (jf1's complete, static, SACRED
read-only receipt tree) · `.omx/tmp/arm_receipts_local/ddm_ld1_lane_lossy_drop_exchange/**` (the
field-alone leg, re-verified against its own receipts) · MAIN relays #1 (`ddm_tac1`, `562b35b0f0`)
and #2 (`ddm_tba1`, `85f6741ff6`).

---

## Verdict first

**The bidirectional control PASSED, in both directions, executed.**

| direction | what it asserts | measured |
|---|---|---|
| **POSITIVE** | re-encoding the unedited field through the shipping receiver reproduces the KNOWN shipped stream | **113,777 B, sha `e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5` — BYTE-IDENTICAL**, full 600-frame run, 691.7 s, no refusal |
| **NEGATIVE** | the same instrument DETECTS a known perturbation | one token flipped out of 117,964,800 → prefix agreement collapses **1,641 B → 2 B (820.5×)** at a matched 8-frame budget |

Receipt: `/Volumes/APDataStore/pact/ddm_dg2_diagonal_reentry/CONTROL_RESULT.json`
(4,939 B, sha `c04a28f223b851ee…`), `control_passed_both_directions: true`.

**The cell was therefore entered.** The diagonal rows below are admissible.

---

## 1. Why jf1's control "failed" — diagnosed at source, not guessed

jf1 reported, honestly and as its first number, that its mandatory positive control missed by
**7,554 B**. The receipt is
`.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_SCOPE_E0002.json`
(69,484 B), fields `positive_control_passed: false` and
`positive_control_stream_deficit_bytes: 7554`.

**Named cause: the quantity jf1 called a "positive control" is not a control.** Read the definition
at source — `experiments/ddm_jf1_joint_field_model_refit.py`, `finalize()`:

```python
"positive_control_passed": null["positive_control_stream_deficit_bytes"] <= 0,
```

with, in `measure()`:

```python
"positive_control_stream_deficit_bytes": (
    stream_bytes - SHIPPED_STREAM_BYTES if tag == "null" else None
),
```

That is a **training-outcome bar**: *a model retrained for N epochs on the unchanged field must emit
a stream no larger than the shipped one.* A control must reproduce a KNOWN quantity through the
MEASUREMENT PATH. This quantity instead requires a stochastic optimisation to land on a particular
byte count — an outcome no instrument can certify, and one that can fail while every byte the
harness reports is exact.

Two facts make the misclassification decisive:

1. **jf1 measured it at `SCOPE_REDUCTION_EPOCH_2_OF_60`** (`fitting_epoch: 2`). The reference
   schedule is 60 epochs in two phases whose terminal phase is `discrete_qat`; the shipped model is
   a QAT-terminal deployable. At epoch 2 the model under test sits in the early *continuous* phase —
   it had not returned to the shipped model's operating regime. The bar was read on an
   under-trained model.
2. **The genuine instrument control already existed in jf1's own tree and had PASSED byte-identically.**
   `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/shipped_control/retained/S1_control_600.json`
   records `byte_identical: true`, `emitted_bytes: 113777`, `prefix_bytes_matching: 113777`,
   `full_run: true`. `ld1` independently ran the same control
   (`.../ddm_ld1_lane_lossy_drop_exchange/measurement_v1/rate/retained/S1_control_600.json`,
   `byte_identical: true`). With this arm, the instrument has now passed in **three** independent arms.

So jf1's harness was never in doubt. Its 7,554 B is a real and useful number — it is the
**refit-drift baseline at epoch 2**, not evidence about the measurement path.

### 1b. The correction this forces on MAIN's relay #1

MAIN relayed tac1's cheapest-deciding-measurement as: *"If the epoch-60 null still cannot reproduce
the shipped 113,777 B token stream, then the instrument is what is being measured and no diagonal
row is interpretable at any epoch — that is your CONTROL_FAILED verdict."*

**That inference is refuted by measurement, and adopting it would have manufactured a second false
CONTROL_FAILED.** Instrument fidelity is certified independently, by encoding the unedited field
with the SHIPPED model: it reproduces the shipped stream byte-for-byte. A null-refit shortfall
therefore measures the REFIT, never the instrument. A drifting null does not make bytes
uninterpretable — every row is read against the SHIPPED ABSOLUTE bar (127,292 B), which the
certified instrument measures exactly. It only means the refit apparatus costs bytes, a cost the
diagonal must also pay. This is precisely jf1's conflation, one level up.

### 1c. Two further corrections to the relayed state

- **jf1's training COMPLETED the full reference budget.** MAIN's "last receipt write 2026-08-23
  02:00:27" is the epoch-2 *measurement* JSON. Training continued and wrote
  `qat_stage_end_epoch_0060.pt` for **all seven tags** at 17:06–17:28 on 2026-08-23. Every one
  passes jf1's own pack gates (`schema` ok, `epoch: 60`, `phase: discrete_qat`,
  `deployment_weights: ema_shadow`, `profile: jf1_joint_refit`, causal-state hash ok). MEASURED.
- **jf1 sealed no epoch-60 rows.** Only the epoch-2 scope-reduced rows exist in its tree. There was
  nothing to "harvest" at epoch 60; the rows below were produced by this arm.
- jf1 is confirmed TERMINAL: all fifteen launcher PIDs across its seven tags are dead. Its tree was
  read only; `jf1.measurement_root` is monkey-patched in this arm's runner so no byte is written
  inside it.

---

## 2. The control, in full (the arm's gate)

Runner: `experiments/ddm_dg2_diagonal_reentry.py` (committed `f87d39631b`). It reuses jg2's
shipping-receiver encoder rather than building a new instrument (recall-before-build).

| leg | field | frames | emitted B | prefix agreeing with shipped | verdict |
|---|---|---:|---:|---:|---|
| A-full | unedited | 600 | **113,777** | **113,777** | `byte_identical: true` — POSITIVE PASSES |
| A-short | unedited | 8 | 1,642 | 1,641 | matched baseline |
| B-short | **one token flipped** | 8 | 1,643 | **2** | perturbation DETECTED |

Injection (MEASURED, re-read from disk and diffed): pair 0, row 192, col 256, class `2 → 3`,
`tokens_changed = 1` of 117,964,800.

A partial run cannot agree on its final byte — the range coder has not flushed the clip — so
A-short's 1,641/1,642 is full agreement, not a mismatch. My first predicate demanded exact equality
and would itself have manufactured a CONTROL_FAILED; it was caught by executing the control and
corrected before sealing. The fail-closed gate was then proven **by execution**: with a failed
control receipt in place, `diagonal()` refuses with `CONTROL_FAILED` and runs nothing.

---

## 3. The 2×2 — and what the joint move CHANGES about the object (sy2's test)

The bar is the shipped combined **127,292 B** = token stream 113,777 + HPAC model 13,515
(ar1b census, HPAC at archive offset `[45,13560)`).

| cell | field | model | source |
|---|---|---|---|
| baseline | shipped | shipped | 127,292 B |
| leg 1 | **moved** (lane→road at k top-cost positions) | shipped | ld1, re-verified below |
| leg 2 | shipped | **refit** | this arm, `null` @ e60 |
| **diagonal** | **moved** | **refit to the moved field** | this arm, k-tags @ e60 |

**Leg 1 re-verified against ld1's own receipts** (jf1's hardcoded constants match to the byte, all
six):

| tag | stream B | Δ vs 113,777 | combined | Δ vs 127,292 |
|---|---:|---:|---:|---:|
| k002500 | 113,973 | +196 | 127,488 | +196 |
| k005000 | 114,056 | +279 | 127,571 | +279 |
| k010000 | 114,601 | +824 | 128,116 | +824 |
| k020000 | 115,305 | +1,528 | 128,820 | +1,528 |
| k040000 | 114,375 | +598 | 127,890 | +598 |
| k060000 | 113,798 | +21 | 127,313 | +21 |

Every rung is POSITIVE — ld1's finding stands: with the model held fixed, every lossy Lane rung
makes the archive BIGGER.

**What the joint move changes about the object.** The stream's byte count is set by two things: the
SOURCE being coded, and the MODEL's conditional distribution. Leg 1 changed the source while the
model stayed fit to the *original* source, so every rung paid a cross-entropy penalty for coding a
modified field under a stale model — that is what the +21…+1,528 B column is. **Leg 1's closure was
priced under a model whose optimality was pinned to the unmodified field.** Refitting the model to
the moved field voids exactly that premise: it changes *which distribution the model is optimal
for*. That is an object change in sy2's sense, not two legs composed — and it is measurable as an
interaction term the two legs cannot produce separately:

```
interaction = (diagonal − leg2) − (leg1 − baseline)
```

It is also selector-free, which matters for MAIN's relay #2: tba1's `N·H(m/N)` naming ceiling binds
levers that must TRANSMIT a chosen subset. Nothing here is transmitted. The modified field is the
source; the receiver reconstructs it from model + stream alone. No position is ever named.

**Scope honesty on tba1's D3.** D3 names retraining on a *reduced alphabet*. These rungs reduce lane
*mass* (k lane tokens become road, k ≤ 60,000 against ~696,000 lane positions ≈ 8.6% at the deepest
rung); the alphabet stays 5 symbols. This is the partial-reduction interpolation of D3, not its
extreme. The full alphabet collapse is untested here and is separately suspect on distortion: lane
is 19% of flips at IoU 0.263 already.

---

## 4. The measured model leg (epoch 60)

Every refit model packs SMALLER than the shipped 13,515 B, and monotonically with lane-drop depth:

| tag | IHS1 raw B | best Brotli B | Δ vs 13,515 |
|---|---:|---:|---:|
| null | 17,767 | 13,463 | **−52** |
| k002500 | 17,804 | 13,487 | −28 |
| k005000 | 17,806 | 13,442 | −73 |
| k010000 | 17,774 | 13,442 | −73 |
| k020000 | 17,756 | 13,440 | −75 |
| k040000 | 17,767 | 13,438 | −77 |
| k060000 | 17,762 | **13,398** | **−117** |

MEASURED, twelve-quality Brotli race per tag with parse-back verified on every representation.

---

## 5. THE DIAGONAL, MEASURED AT THE REFERENCE BUDGET (epoch 60, all seven tags, n600)

`fitting_budget_scope: FULL_REFERENCE_60_EPOCHS` — **no scope reduction on epochs, none on tags,
none on pairs.** All 600 pairs, all seven fields, the reference two-phase schedule to its
`discrete_qat` terminal.

### 5a. Leg 2 — the model moves alone (the discriminator MAIN asked for first)

| quantity | epoch 2 (jf1) | **epoch 60 (this arm)** |
|---|---:|---:|
| null refit stream | 121,331 B | **114,143 B** |
| Δ vs shipped 113,777 B | **+7,554** | **+366** |
| null refit model | 13,348 B | 13,463 B |
| null combined | 134,679 B | **127,606 B** (+314 vs bar) |

**jf1's 7,554 B collapses to 366 B when the same measurement is taken at the reference budget —
20.6× smaller.** The under-training diagnosis is confirmed quantitatively, not just argued. The
residual +366 B is the honest refit-drift cost at epoch 60: real, small, and a property of the
retraining, not of the instrument.

### 5b. The diagonal — field AND model moving together

Bar = 127,292 B. `interaction = Δdiagonal − Δleg2 − Δleg1`, all in combined bytes.

| tag | refit stream | refit model | **combined** | **Δ vs bar** | leg 1 Δ | **interaction** | ΔS (rate leg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| k002500 | 113,637 | 13,487 | 127,124 | **−168** | +196 | **−678** | −1.119e-04 |
| k005000 | 113,889 | 13,442 | 127,331 | +39 | +279 | −554 | +2.597e-05 |
| k010000 | 114,236 | 13,442 | 127,678 | +386 | +824 | −752 | +2.570e-04 |
| k020000 | 114,632 | 13,440 | 128,072 | +780 | +1,528 | −1,062 | +5.194e-04 |
| k040000 | 112,912 | 13,438 | 126,350 | **−942** | +598 | **−1,854** | −6.272e-04 |
| **k060000** | **112,318** | **13,398** | **125,716** | **−1,576** | +21 | **−1,911** | **−1.049e-03** |

Exchange rate 6.658590e-07 S/B, CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0.

**Two results, both measured:**

1. **The interaction term is NEGATIVE at every one of the six rungs** (−554 to −1,911 B). The joint
   move is strictly more than the sum of its legs, everywhere. This is the object change of §3
   showing up as a number: refitting the model voids the stale-model penalty leg 1 was paying, and
   it does so by more than either leg delivers alone. sy2's law predicted that only an
   object-changing composition could survive; this is a measured instance of one.
2. **Three rungs land BELOW the shipped bar.** Best: **k060000 at 125,716 B, −1,576 B**,
   ΔS(rate) = **−1.0494e-03**. The field-alone leg at that same rung was **+21 B** — the sign
   flips, and the magnitude moves by 1,597 B, purely from refitting the model to the field it must
   code.

**And the size of it: −1,576 B is 3.719% of the 42,382 B demand.** Not a route. A contributor.

### 5c. Confirmed at the archive, not just in the section arithmetic

Each row builds a real `candidate_archive.zip` through the shipping packer. The archive deltas
reproduce the section arithmetic exactly — the framing absorbs nothing:

| tag | candidate archive B | Δ vs dx2 180,368 B | ΔS (rate leg) |
|---|---:|---:|---:|
| null (leg 2) | 180,682 | +314 | +2.0908e-04 |
| k002500 | 180,200 | −168 | −1.1186e-04 |
| k005000 | 180,407 | +39 | +2.5968e-05 |
| k010000 | 180,754 | +386 | +2.5702e-04 |
| k020000 | 181,148 | +780 | +5.1937e-04 |
| k040000 | 179,426 | −942 | −6.2724e-04 |
| **k060000** | **178,792** | **−1,576** | **−1.0494e-03** |

These are byte-closed archives on disk, not projections.

### 5d. Receiver identity — PROVEN on all seven rows

Every row's final gate is `measure()`'s decode of its own candidate archive back to its exact target
field, under its own refit model, through the shipping receiver. A row that fails it raises and
produces no `MEASURE_RESULT.json`, so no row can be reported without it.

**All seven rows sealed with `decoded_token_identity: true`**, each decoded sha256 equal to its
target field's sha256 (~1,360 s decode per row):

| tag | decoded token sha256 (prefix) | = target field sha | identity |
|---|---|---|---|
| null | `cc10a7b09353c0af…` | yes | true |
| k002500 | `c45979acb7a87bda…` | yes | true |
| k005000 | `6c210dd19eefb2b6…` | yes | true |
| k010000 | `297cee64f3e1438b…` | yes | true |
| k020000 | `7251367a078796a1…` | yes | true |
| k040000 | `03ce7bd8a8498ea2…` | yes | true |
| k060000 | `15018481bd8007dd…` | yes | true |

The joint move is therefore **receiver-consistent**: the modified field is exactly reconstructible
from the refit model plus the stream, with no side channel and no transmitted selector.

Sealed verdict: `.omx/tmp/arm_receipts_local/ddm_dg2_diagonal_reentry/DIAGONAL_RESULT.json` —
`control_passed_both_directions: true`, `fitting_budget_scope: FULL_REFERENCE_60_EPOCHS`,
`rows_measured: [null, k002500, k005000, k010000, k020000, k040000, k060000]`,
`any_diagonal_row_below_shipped_combined: true`.

---

## 6. The prior-law prediction, adjudicated

My charter's pre-registered prediction, deliberately pessimistic:

> **REFUSED** — the diagonal measures net-positive ΔS (worse), consistent with a genuine local
> optimum rather than an axis-probing artifact.

**On the rate leg: REFUTED.** Three of six rungs measure ΔS(rate) < 0; the best is −1.0494e-03 at
k060000. The prediction of a uniformly positive ΔS is false, and the sharp-optimum reading does not
extend to this direction: the five concordant arms probed axes, and the diagonal is not one of the
axes they probed.

**On realized ΔS: NOT ESTABLISHED, and I will not claim it.** The charter's falsifier requires a
*realized* ΔS < 0. These rungs are LOSSY by construction — they convert k Lane tokens to Road, and
Lane is 19% of dx2's seg flips at IoU 0.263. The distortion leg is not owned by this arm, was not
measured, and is not free. A −1,576 B rate credit is erased by a d_seg increase of 1.576e-05
(0.0000158), which is 7.8% of dx2's total d_seg — a small change on an axis where every measured
lossy Lane rung has so far cost more than it bought. **The honest state is: the rate leg of the
diagonal is negative and measured; the realized ΔS is unmeasured and the burden is on the seg leg.**

The correct next measurement is therefore not another byte row. It is d_seg on the k060000 and
k040000 candidate archives, which this arm has built, retained, and receiver-checked.

---

## 7. What is NOT claimed

- **No score claim, no promotion, no pointer move.** Byte leg only, `[macOS-CPU advisory]`.
- **d_seg and d_pose are UNMEASURED** on every row. `d_seg_per_class: null`, `d_pose: null`,
  `net_delta_S: null` in every sealed row, by construction.
- **Never interpolated between rungs.** The non-monotone shape (k002500 below the bar, k005000
  through k020000 above it, k040000 and k060000 below) is read only at the six measured points.
  The amplification exponent ~16.7 forbids reading between them.
- **Every rate number is a MEASURED re-encode** through the shipping receiver, never a −log2 p
  estimate (fs2's direction-dependence law).
- Scope of the negative-existence claim in §1: I searched jf1's complete receipt tree, its source
  at `experiments/ddm_jf1_joint_field_model_refit.py`, and `.omx/research/` for any epoch-60 jf1
  measurement row. I found none; jf1 sealed only epoch-2 rows.
- The alphabet-collapse extreme of tba1's D3 is untested here (§3).

---

## 8. Attacking my own conclusion

1. **"The −1,576 B is just the refit model being better."** No — the null row prices exactly that
   and it is **+314 B, the wrong way**. The refit-alone move COSTS bytes. The gain appears only when
   the field moves too, which is why the interaction column is the load-bearing one.
2. **"The interaction is an artifact of comparing across two different harnesses."** Leg 1 is ld1's
   number and legs 2/diagonal are mine. I re-verified all six of ld1's stream bytes against ld1's
   own receipts (they match jf1's constants to the byte), and ld1's own byte-identity control passed
   — the same control mine passed. Same encoder, same runtime, same field custody.
3. **"The control passed on the shipped model, but the diagonal uses a refit model — the control
   does not cover it."** Correct, and it is why the receiver-identity decode matters: every row must
   decode its own candidate archive back to its exact target field under the refit model. That gate
   is `measure()`'s `JF1Error` refusal, not a report field, so a row that fails it produces no row
   at all.
4. **"Three of six rungs are above the bar — is the win cherry-picked?"** The reported best is the
   best of six pre-registered rungs measured together, all six published above, including the three
   that lose. The non-monotonicity is real and unexplained; I do not claim a trend.
5. **My own first predicate was wrong** and would have produced a false CONTROL_FAILED (§2). That is
   the second instance in two arms of a control predicate, not an instrument, producing the failure
   — which is the actual recurring defect this cell kept tripping over.

---

## 9. Verdict

`verdict_scope`: **the BYTE/RATE leg of the field × model diagonal on the dx2 object (archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B), at the six retained
ld1 lane→road rungs, at the reference 60-epoch refit budget, n600, all pairs.** It does NOT extend
to: the distortion legs (unmeasured), rungs not in the retained ladder, other objects, or the
alphabet-collapse form of D3.

- **CONTROL: PASSED, both directions, executed.** Positive byte-identical at 113,777 B; negative
  detects 1 token in 117,964,800.
- **THE CELL IS ENTERED.** It was not entered before this arm.
- **Prior-law prediction REFUTED on the rate leg**; realized ΔS not established.
- **Best measured rate leg: k060000, 125,716 B combined, −1,576 B, ΔS(rate) −1.0494e-03, 3.719% of
  the demand.** A contributor, not a route.
- **The interaction is negative at all six rungs** — the first measured object-change composition on
  this object.

## 10. What this arm owes next (named, not deferred vaguely)

The candidate archives are built, retained and receiver-checked. The single deciding measurement is
now **d_seg on the k060000 and k040000 candidate runtimes**. If seg cost < 1.576e-05 at k060000 the
row is a genuine frontier contributor; if not, the diagonal joins the closed family with a
measured — not assumed — reason. That fire is MAIN's, not this arm's.

## 11. Own-vehicle frontier

dx2 — **S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`** — **UNMOVED by this arm.**
No scorer ran, no pointer moved, no score claimed. Gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed
distortion, or 150 B at zero distortion.
