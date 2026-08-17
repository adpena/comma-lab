# pz5 — STAGE 0 GATE FAILS: the −20,524 B pose lever does not exist on hv1, and does not exist on PR130 either

`date_utc: 2026-08-16` · `owner: ddm_pz5` · `axis: [macOS-CPU $0 re-derivation; no scorer, no dispatch]`
`score_claim: false` · `promotable: false` · `frontier_moved: false`
`verdict: STAGE_0_REFUSED — DO NOT BUILD`
receipts: `/Volumes/APDataStore/pact/ddm_pz5/retained/pz5_stage0_arithmetic.json`,
`/Volumes/APDataStore/pact/ddm_pz5/retained/pz5_stage0_carrier_identity.json`

## Verdict

The charter routed me to receiver-close a 1,817 B pose packet against the hv1 frontier on the
strength of a `−20,524 B` rate win worth 142% of the gap to 0.15. **The gate fails on four
independent grounds, any one of which is sufficient.** I did not build. Per the charter's own
instruction — *"If (c) shows the numbers are against different bases, that is a COMPLETE and
VALUABLE finding: write it up, stop, and do not build"* — this is that finding, and it is worse
than a base mismatch.

The headline in one line: **the section the arithmetic proposed to delete is the renderer for
frame_0, and the packet proposed to replace it stores PoseNet's own six output scalars — which
are not an image. Nothing in the archive can turn one into the other.**

## 1. The frontier, re-derived (MEASURED)

I recomputed the pointer from components rather than trusting it:

| quantity | value |
|---|---|
| `25/37,545,489` | `6.658590e-7` S/byte |
| rate at 182,759 B | `0.1216917164` |
| `seg + pose + rate` | `0.15959729291365` vs pointer `0.15959729295498598`, residual `−4.13e−11` |
| `d_pose = (0.0082945765)²/10` | `6.880e−6` |
| `d_seg = 0.029611/100` | `2.9611e−4` |
| gap to 0.15 | `+0.0095973` |

MAIN's arithmetic is **internally correct**: from `−20,524 B` I reproduce `ΔS_rate = −0.0136661`
and a reach-0.15 pose budget of `2.2217×` exactly. The defect is not the arithmetic. It is the
`−20,524 B`.

## 2. (a) What the 1,817 B packet is (MEASURED, at source)

`direct_p040_b10-5-3-3-0-3` — uniform scalar quantization of the **600×6 official DALI PoseNet
target tensor** at bits `[10,5,3,3,0,3]`, Brotli-q11'd. Quantization MSE `2.32888e-5`. The 2,860 B
sibling `direct_p092_b12-6-6-5-5-5` reaches qMSE `6.91224e-7`.

It encodes **the six numbers PoseNet outputs**. It encodes no pixels.

## 3. (b) What hv1 actually contains (MEASURED, parsed from the archive)

I parsed the RX1 container header of the frontier archive itself
(`/Volumes/VertigoDataTier/…/ep0634/retained/candidate/archive.zip`, sha `80d9c8c6…`, verified):

| section | bytes | share |
|---|---:|---:|
| ZIP overhead | 100 | 0.05% |
| RX1 header | 14 | 0.01% |
| HPAC model | 13,515 | 7.4% |
| semantic model | 34,763 | 19.0% |
| **CPR1/CAP1 carrier** | **22,161** | **12.1%** |
| residual table | 96 | 0.05% |
| token stream | 112,110 | 61.3% |
| **total** | **182,759** | 100% |

Model region `14 + 70,439 = 70,453` and tokens `112,110` both reconcile to the sealed fire order
exactly. **The carrier is 22,161 B — not 22,161 assumed, measured from the header.** There is no
23,384 B object in hv1.

Decoded, the carrier is a 12×3×24×32 signed spatial **RGB basis** (27,648 codes, range −15..15)
plus 600×12 int12 per-frame coefficients plus 96 B of scales. Byte attribution by Brotli-q11 on
each stream alone: basis **12,041 B (54.3%)**, coefficients **10,077 B (45.5%)**; the two sum to
22,118 B against the actual joint 22,161 B — within 43 B, so the split is tight, not a guess.

## 4. (c) The base mismatch — CONFIRMED, and it is the shallowest of the four problems

`−20,524 B` reproduces exactly on **PR130**: `191,052 − 23,384 + 2,860 = 170,528`. PR130's base
score is `0.172141297491896447` at 191,052 B. hv1 is `0.15959729295498598` at 182,759 B. Different
archives, different trained state, different section sizes.

Re-based onto hv1 the analogous delta is `2,860 − 22,161 = −19,301 B`, giving `ΔS_rate =
−0.0128517` and a reach-0.15 pose budget of **1.9387×**, not 2.22×. The transferred number
**overstates the rate gain by 6.3%**.

Sister credit: `hv2` flagged this today (`ddm_hv2_arm_final_harvest_20260816.md:204` — *"pz2
measured against PR130's 23,384 B carrier. Neither is the hv1 object"*) and wrote the
re-derivation gate into the task row. hv2 scoped it as a precision caveat — *"correct in form and
approximate in its last digit."* **It is not. Sections 5–7 are why.**

### The subtrahend was never a section at all

`ddm_pi135_pr135_intake_20260810.md:81` (MEASURED by that arm): *"23,054 B is PR130's raw
serialized CPR1 section. **23,384 B is the separately measured full-archive leave-one-out marginal
attributed to pose.**"* A leave-one-out marginal is not a deletable object. The pz2 hypothetical
subtracted a *sensitivity*, then added a packet, and called the difference an archive size.

## 5. The category error (DECISIVE, MEASURED at source in `cpr1/inflate.py::render_video`)

```python
output[2*(start+offset) + 1] = master_np[offset]   # frame_1 <- semantic(tokens, indices)
output[2*(start+offset)]     = slave_np[offset]    # frame_0 <- einsum(coefficients, basis)
```

- **frame_1** (the frame SegNet scores, `x[:, -1, ...]`) is rendered from the semantic model +
  token stream. The carrier never touches it.
- **frame_0** is rendered **entirely** by the carrier: `einsum("bk,kchw->bchw", coefficients,
  basis)`, scaled, bicubic-upsampled to 874×1164, clamped to uint8.

So the carrier's whole job is to **synthesize a full 874×1164×3 frame_0 image** that PoseNet then
consumes. The pz2 packet stores PoseNet's *outputs*. PoseNet is fed images, not its own outputs.
**No receiver in the archive — and none proposed anywhere — converts six stored scalars into the
frame_0 image.** pz2 states this itself and did not claim otherwise: `frame_parity:
NOT_RUN_NO_FRAME_REALIZATION`, `scorer_invoked: false`, projection axis literally `[TOY-BRACKET …
no receiver/scorer]`, and its conclusion names the only live route as *"a jointly trained,
byte-closed target-conditioned receiver."* The `−20,524 B` was never a rate saving. It is the
**price of an unbuilt receiver**, quoted as if the receiver were free.

### Adversarial check: is the carrier secretly the pose?

The carrier has 12 dimensions and PoseNet emits 12 outputs. I tested the coincidence rather than
dismissing it. Decoding the shipped coefficients and correlating against pz2's own retained
official DALI targets (sha `23ae28d2…`):

- max |corr| over all 72 (carrier-dim × pose-output) pairs = **0.6267**
- R² of the **best** linear map from *all 12* carrier dims to each pose output =
  `[0.344, 0.551, 0.692, 0.750, 0.651, 0.499]`

Re-derived under a standardized, well-conditioned fit (`cond` 11.9 → 3.6): identical to 4 decimals.
25–66% of every pose output's variance is unexplained by the entire carrier. The carrier correlates
with ego-motion because camera motion drives frame appearance — it is not a pose store.

## 6. (d) d_pose is MODELLED, never REPRODUCED

pz2's `6.91224e-7` is the MSE between quantized and true **targets**. `d_pose` is
`MSE(PoseNet(our frames), PoseNet(gt frames))`. Equating them assumes a receiver that renders
frames whose PoseNet output lands exactly on the stored target. That receiver does not exist, so
every projected S in the pz2 receipt is a bracket, correctly labelled as one by pz2.

## 7. The rate ladder on this carrier is already MEASURED CLOSED — by 45,375×

`ra2c` measured this on the **identical 182,759 B object** today, archive held byte-identical,
frame_0 rendering varied:

| rung | bytes returned | measured d_pose ratio | my re-derived hold-frontier bar | over by |
|---|---:|---:|---:|---:|
| α=0, delete whole carrier | 22,161 | **350,428×** | 7.7229× | **45,375×** |
| rank-4 | 14,662 (analytic credit) | **2,400.65×** | 4.7394× | 507× |

I reproduced ra2c's published `7.7229×` bar exactly from the frontier components, which is my
evidence that I am reading their instrument correctly, not just quoting it.

pz2's swap proposes to return **19,301 B** — 87% of the carrier. The only rung that returns *real*
bytes is α=0, and it costs 350,428× against a 6.50× bar. ra2c also proves d_seg is *structurally*
invariant to any frame_0 carrier edit (SegNet reads frame_1 only), so the carrier is a pure
(pose, rate) trade — which makes MAIN's budget arithmetic right in form, and makes the measured
miss unambiguous rather than confounded.

Eckart–Young makes rank truncation a **bound**, not an estimate: the rendered field `C·B` has
25.15% error at rank-4 and 4.23% at rank-11 against tolerances of 0.389% / 0.118%.

## 8. Attacking my own conclusion

**What would falsify my headline.** A receiver that renders frame_0 from a small stored
representation and holds d_pose within 1.94× of `6.880e−6` on n600 through the real PoseNet. That
is not refuted by anything here — it is simply **unbuilt**, and it is a *generator*, not a
substitution. My claim is strictly: the arithmetic swap "delete carrier, add target packet, keep
the scored terms" is void.

**Where I was wrong mid-flight.** I drafted a fifth layer arguing the carrier drives d_seg and
that the swap would blow the seg term. Reading `render_video` at source killed it: the carrier
renders frame_0 only, SegNet reads frame_1, so d_seg is invariant. ra2c was right and my draft was
wrong. I discarded it before reporting.

**Honest limits.** ra2c's axis is macOS-CPU advisory with base d_pose `1.4747e−4` against the T4
frontier's `6.880e−6` — a 21.4× instrument gap, bridged by ratio-transfer, which is most defensible
at α=0 where the perturbation dwarfs instrument noise. Not a score claim. ra2c's rank-4/rank-11
byte figures are analytic rate credits; only α=0 returns real payload bytes. My byte attribution
(54.3%/45.5%) is separate-stream Brotli, an upper bound on separable cost, validated by summing to
within 43 B of the joint.

**Verdict scopes.** `FAMILY` for *replacing the frame_0 carrier with a stored PoseNet-target
packet* — the defect is structural (targets are not images), not formulation-specific. `INSTANCE`
for the specific `−20,524 B` / `1,817 B` / `2,860 B` numbers, which are sound measurements of the
PR130 target tensor and remain valid as representation facts. **pz2 is not at fault**: it labelled
every projection a TOY-BRACKET, ran no scorer, and queued the candidate behind a receiver
fire-trigger. The failure was downstream over-reading of a correctly-hedged receipt.

## 9. What MAIN should carry forward instead

Corrected constants for any future carrier-attack pricing on hv1:

- carrier section = **22,161 B** (measured from the RX1 header, not inherited)
- returning `R` bytes: `ΔS_rate = −6.658590e−7 · R`
- reach-0.15 d_pose bar: `((0.0082945765 + (0.15 − S_after))² / 10) / 6.880e−6`
- full-carrier deletion (22,161 B) buys 2.63× at best and measures 350,428×

## NEXT_IF_RESUMED

1. **Do not fire any T4 row for this lever.** No candidate exists; Stage 1 and Stage 2 are void.
   Owner: MAIN. Fire-condition: none — permanently, unless item 3 produces a byte-closed receiver.
2. **Correct the rank-1 task row** that routes launches on pz2's 1,817 B. Owner: MAIN.
   Fire-condition: immediately — the row currently points at a lever that does not exist, and hv2's
   gate ("must be re-derived before any launch is routed on it") is hereby discharged: **REFUSED**.
3. **The pose-metric rank ladder** (ra2c §5, already named, owner MAIN): whiten the coefficient
   space by the measured PoseNet Jacobian, truncate *there*, re-price against the same bars.
   Fire-condition: one advisory slot, $0. Falsifier: if pose-metric rank-4 error also exceeds
   0.389%, the carrier family is closed in both metrics and the remaining pose route is exclusively
   the js1 joint line.
4. **The target-conditioned frame_0 generator** (pz2's own named live route). Owner: unassigned.
   Fire-condition: a charter that treats it as a *generator* to be trained and byte-closed, never
   as a section substitution. It must render frame_0 and be measured through the real PoseNet.

Frontier unchanged: **S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600]**, archive sha
`80d9c8c6…`. Not moved, not measured, not claimed.
