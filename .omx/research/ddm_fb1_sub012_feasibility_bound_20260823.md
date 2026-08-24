# ddm_fb1 — the sub-0.12 FEASIBILITY BOUND: no single axis, perfected, clears; and the renderer's two requirements are in measured tension

**Type: DERIVED** (arithmetic over MEASURED inputs), not itself a measurement. Every input is
cited to a measured receipt; the derivation is exact and reproducible from the numbers below.
This memo makes no score claim and moves no pointer.

`verdict_scope`: **the dx2 object** (archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B). A different
object has a different residue map and a different bound.

STORES CONSULTED: `ddm_ar1b_archive_residue_purchase_20260822.md` (the residue map, zero
remainder) · `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate 6.658590e-07 S/B —
CITED, not re-derived, per #1207) · `ddm_mst1` / task #1205 + #1211 (manufactured-seg
fractions) · `ddm_w72_distortion_advisory_20260823.md` (#1230) · `ddm_rj1_renderer_joint_move_20260823.md`
(#1224) · `ddm_wd3_n120_family_disposition_20260816.md` (the distillation family verdict) ·
`ddm_sy2_composition_synergy_deep_pass_20260823.md` (#1227 object-change law) · tasks
#1203 (two-currency demand) · #1212/#1214 (sharp-optimum law) · #1222 (renderer carries pose).

---

## 1. Why this memo exists

Six arms in two days each attacked ONE axis of the dx2 object and each measured a refusal.
Nobody had asked the prior question: **what would any winning route have to look like?**
That question is answerable at $0 from receipts already in hand, and its answer constrains
every future charter — including rj2, which I have been calling "the ONE live route."

## 2. The inputs (all MEASURED, all cited)

| quantity | value | source |
|---|---:|---|
| dx2 archive | 180,368 B | pointer, sha `976f706d…` |
| dx2 `d_seg` | 0.00020139 | dx2 authority row |
| dx2 `d_pose` | 0.00000637 | dx2 authority row |
| dx2 S (recomputed from components, #877) | 0.148219876 | rate 0.1200996 + seg 0.020139 + pose 0.0079812 |
| target archive for 0.12 at fixed distortion | ≤ 137,986 B | strict-inequality floor, verified both ends |
| demand | **42,382 B** | 180,368 − 137,986 |
| exchange rate | 6.658590e-07 S/B | tx1 §0, CITED |
| manufactured fraction of dx2 seg error | 90.47% | mst1 / #1205 |
| …of which appears at the NATIVE RENDER | 78.71% | #1211 |

**Residue map (ar1b, zero remainder — sums to 180,368 exactly):**

| block | bytes | % archive | **% of the 42,382 B demand** | measured status |
|---|---:|---:|---:|---|
| token stream | 113,777 | 63.08% | **268.5%** | SHARP local optimum, 5 concordant arms (#1212/#1214) |
| renderer | 30,856 | 17.11% | 72.8% | W72 −10,879 B ⇒ seg ×116.8 (#1230); rj1 = **NO-VERDICT, all rungs UNMEASURED** (see §9) |
| carrier | 22,010 | 12.20% | 51.9% | refuted 1,356× wrong direction (#1222) |
| HPAC model | 13,515 | 7.49% | 31.9% | part of the sharp optimum |
| framing | 114 | 0.06% | 0.3% | — |
| compact residual | 96 | 0.05% | 0.2% | — |

## 3. FINDING 1 — only one block is larger than the demand, and it is the closed one

The demand is **23.50% of the whole archive**. Exactly one block exceeds it: the token stream
at 268.5%. That is precisely the block five concordant arms (oe1, ld1, ae1, ni1, wj1) measured
at a **sharp** local optimum in every direction tested.

Every other block is individually insufficient **even at zero bytes**. This is not a claim about
difficulty — it is arithmetic. A renderer that costs nothing supplies 72.8% of the demand.

## 4. FINDING 2 — no single axis, perfected, reaches sub-0.12

Each row perfects ONE axis and holds the others at their measured dx2 values:

| scenario | archive | S | vs 0.12 |
|---|---:|---:|---:|
| do nothing (dx2) | 180,368 | 0.148220 | +0.028220 |
| **renderer → ZERO bytes** | 149,512 | 0.127674 | **+0.007674** |
| **ALL manufactured seg removed** (90.47%) | 180,368 | 0.130000 | **+0.010000** |
| …only the native-render part (78.71% of it) | 180,368 | 0.133879 | +0.013879 |
| **pose → ZERO** | 180,368 | 0.140239 | **+0.020239** |
| **BOTH distortions → ZERO** | 180,368 | 0.120100 | **+0.000100** |
| manufactured seg + renderer→0 (two axes) | 149,512 | 0.109454 | −0.010546 **PASS** |

Cross-check: "both distortions → 0" leaves **150 B**, independently reproducing #1203's
zero-distortion gap from a different direction. The arithmetic is consistent with the prior
measurement, not a fresh derivation that could be quietly wrong.

**Consequence:** any route to sub-0.12 from this object must move **at least two axes**. That is
the same conclusion sy2 (#1227) reached from causal structure — a closed leg survives only when
another leg first changes the object it was priced on — arrived at here from pure accounting.

## 5. FINDING 3 — the renderer axis has a corner, and its two halves are in measured tension

Solve the iso-0.12 surface for the renderer axis. Let `f` = fraction of the manufactured seg
error removed; `B` = renderer bytes that must ALSO be shed; pose held at its dx2 value.

| f | d_seg after | B needed | % of the whole renderer | feasible? |
|---:|---:|---:|---:|---|
| 0.00 | 0.00020139 | 42,381 | 137.4% | IMPOSSIBLE — exceeds the whole renderer |
| 0.25 | 0.00015584 | 35,540 | 115.2% | IMPOSSIBLE |
| 0.50 | 0.00011029 | 28,700 | 93.0% | possible |
| 0.75 | 0.00006474 | 21,859 | 70.8% | possible |
| 1.00 | 0.00001919 | 15,018 | **48.7%** | possible |

Minimum manufactured-seg repair for a renderer-only close: **f = 0.4212** — remove 38.11% of ALL
dx2 seg error AND shed the entire 30,856 B renderer.

**The adversarial best case** (deliberately generous, pose unchanged): shed HALF the renderer
(15,428 B) AND remove ALL manufactured seg ⇒ **S = 0.119727, clearing 0.12 by 0.000273 = 410 B
of margin.** So the renderer axis is **NOT arithmetically closed** — I said that first and it was
wrong. It clears, in a corner, by a razor.

**But the corner's two halves trade against each other, measured:**

- it needs ≥48.7% of the renderer's bytes shed;
- **W72 shed 35.3% of the renderer and multiplied seg by 116.8×** (#1230) — the wrong direction,
  at less than the required magnitude;
- rj1's rungs are **UNMEASURED** — see §9. The only measured W64 object is WD4's warm-lineage
  instance at `d_seg=0.03182023`, `d_pose=13.43292999`, `S≈14.8829`, which rj1 itself labels an
  INSTANCE negative it did not rerun;
- wd3's fresh-init scorer-aware distillation family is NEGATIVE at 65 ep, and its own WARM arm
  projected ~6× over its byte-derived bar.

Four measurements, all on the "shed renderer bytes" half, all costing distortion rather than
saving it. The corner requires *more* byte reduction than any of them attempted *and*
simultaneously near-perfect seg repair.

## 6. What this means for rj2 — a correction to my own routing

I have been calling rj2 "the ONE live route to sub-0.12." That was over-claimed on two counts:

1. **rj2 can move the frontier without reaching 0.12.** Any ΔS < 0 is progress. That value is
   real and is not what this memo questions.
2. **rj2 cannot be a complete route by itself** unless it lands in the §5 corner — which demands
   it shed ~half the renderer *while* repairing essentially all manufactured seg, against four
   measurements that say byte-shedding manufactures seg.

The honest re-aim: rj2's leverage is **not** the renderer's 30,856 bytes (72.8% of demand,
insufficient, and measured-expensive to shed). It is the **21,537 B of rate budget that the
renderer's native-render manufactured seg error is worth** (50.8% of the demand, #1211) — value
recoverable at *zero* byte change, in the currency the exchange rate makes fungible. A renderer
that manufactures less error buys budget without paying the W72 tax.

That is a different objective than the rj2 charter states, and it is testable independently.

## 7. What is NOT claimed

- This does **not** prove sub-0.12 unreachable. It bounds the SHAPE of any route: ≥2 axes, or
  the token stream's sharp optimum broken.
- It does **not** price any route's achievability. `f` is a perfection parameter; **no arm has
  achieved any f > 0**.
- The manufactured-seg fractions are mst1's measurement on dx2; the exchange rate is linear in S
  and exact; the residue map is ar1b's with zero remainder. Errors in those inputs propagate.
- Pose is held constant in §5 by construction. Moving it changes the surface (pose → 0 alone is
  worth 11,986 B = 28.3% of demand).

## 8. Own-vehicle frontier

dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]` — **UNMOVED by this memo**
($0, no measurement fired). Gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed distortion, or 150 B at
zero distortion.


---

## 9. CORRECTIONS (append-only, 2026-08-24) — two of my own claims, both refuted

**9.1 — The "rj1 W64 refused 3.51×, d_pose 97.70%" figures HAVE NO SOURCE. Retracted.**
`ddm_msr1` refused to repeat them; `ddm_mf1` (`b0c2869ce4` §202) had independently reached the same
conclusion earlier and pointedly declined to restate them as an RJ1 fact. MAIN then verified at
source: `ddm_rj1_renderer_joint_move_20260823.md` carries `Status: MECHANISM-INCOMPLETE-WITHHELD`,
`verdict_scope: NO-VERDICT:DX2_PRECOMPENSATION_REPRESENTATION_RUNGS`, and all three rungs
(`nested_group_dense_w72`, `pointwise_svd_w96_r32`, `film_amortized_flat_w96`) read **UNMEASURED in
every column** — realized d_seg, realized d_pose, B, H, W, joint ΔS. Neither `3.51` nor `97.70`
appears anywhere in it. The only measured W64 object rj1 cites is WD4's warm-lineage
salience-pruned dense instance (`d_seg=0.03182023`, `d_pose=13.43292999`, `S≈14.8829`), explicitly
"an instance negative, not a family result," which rj1 did NOT rerun. Neither figure reproduces
from it (pose is 77.9% of that S, not 97.7%; the ratio to the pointer is ~100×, not 3.51×).

SEARCH SCOPE for the negative-existence claim (m53): `.omx/research/*.md`. Within that scope the
two figures appear ONLY in this memo, in the `ddm_tac1` charter that inherited them from this memo,
and in the two arms flagging them. NOT searched: charters generally, `.json` receipts, `.py`
sources, the task ledger, SSD-only artifacts. Sister row `#1224` carries the same unsupported
headline and is corrected in the ledger.

**9.2 — §6's re-aim of rj2 toward "the 21,537 B the native-render manufactured seg error is worth"
is WITHDRAWN.** `ddm_msr1` (`7624816b02`) measured the manufactured mass and found it **90.12%
BALANCED two-way flow**: 98.4% is bidirectional exchange across five near-symmetric class
interfaces; 99.66% of manufactured pixels sit at Chebyshev distance 1 from a token boundary (38.4×
enriched, ZERO in any interior); the median frozen-head deficit is 0.162 against a 5.763
correct-margin mean (24.8× shallower); 63.4% had a 17×17 token window exactly equal to GT, which
EXONERATES the token field; 0 of 600 pairs are clean. The mechanism is zero-mean one-pixel boundary
jitter by the frozen head on a painted discontinuity.

The consequence is actuator-INDEPENDENT: any actuator moving interface (a,b) one way in a cell
fixes `a→b` and **one-for-one deepens `b→a`**, so balanced flow is unreachable by construction —
learned weights, palette bias, and shipped mask alike. Charging real collateral, an oracle over 260
searched (interface × direction × δ) combinations, choosing best direction and δ independently per
interface, nets **+1 pixel = 1.27 B = 0.0047% of the manufactured mass**. Nineteen of twenty rows
never net positive at any δ. The one open sub-family, boundary SHARPENING, has no address either:
rendered cross-boundary contrast at manufactured pixels is 0.972× that at correct pixels.

The 21,537 B remains a REAL ACCOUNTING IDENTITY and is an UNREACHABLE LEVER. §2's inputs are
unaffected — msr1 reproduced the whole decomposition at source from the primary fields and its
independent control recomputed `d_seg = 0.00020139058431`, equal to the dx2 authority row. What is
withdrawn is §6's routing conclusion, not §2's arithmetic.

**9.3 — What survives, and where the campaign now points.** §3, §4 and §5's arithmetic stand.
Combined with `ddm_tac1` (1 OPEN pair of 28 = `tokens × HPAC model`) and `ddm_tba1` (four of six
directions closed at $0; only "retrain the model on a reduced alphabet" survives), three arms with
disjoint methods now converge on ONE cell. Reopening the manufactured-seg route requires an
object-changing leg per sy2 (`#1227`), after which the bound must be RE-DERIVED, not transferred.
