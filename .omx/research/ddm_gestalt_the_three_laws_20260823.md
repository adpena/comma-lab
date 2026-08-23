# THE GESTALT — twenty negatives are three laws, and the winning move class was visible the whole time

`verdict_scope: FORMULATION:DX2_LINEAGE_SUB012_CAMPAIGN` — a synthesis over measured rows, not a new
measurement. Every number below is cited to an arm receipt. Author: MAIN. Cost: $0. Date: 2026-08-23.
Operator directive: *"Continue with all, search for all signal everywhere pointing to gestalt"* +
standing law `all-signal-informs-evolving-gestalt` (08-21).

## 0. The one sentence

**Cost and damage are co-located, so single-axis moves are structurally dominated — every campaign win
was a joint move, every recent loss was a single-axis perturbation, and the only untouched joint object
is the renderer that dominates all three score terms.**

## 1. LAW ONE — the converged-optimum law

Eight arms perturbed the dx2 optimum in eight directions over two days. All eight lost.

| arm | axis perturbed | result |
|---|---|---|
| `ld1` (#1212) | field (Lane, lossy rungs) | every rung makes the archive **BIGGER** |
| `ae1` (#1208) | model (anti-predicted excess, static recovery) | 26,645 B real, both routes lose net |
| `oe1` (#1214) | model (zero-stored causal form) | loses too — family CLOSED |
| `ap1` (#1220) | residue (4 groups × 3 depths, 12 rungs) | **12/12 net-positive**, 148×–81,548× |
| `to2` (#1201) | order (token reordering) | **−196%** |
| `ad2` (#1201) | order (a different reordering) | +34.5% — *but only as a context-model substitute* |
| `ef1` (#1202) | estimator | generic loses to the domain-tuned incumbent |
| `mp3` | model (subtraction) | pending |

**This is one fact, not eight.** A jointly-converged optimum loses in *every* direction by definition;
that is the content of "converged." The eight rows were predictable after the second, and the campaign
spent two days re-measuring the same property. The information in rows 3–8 is not "this direction is
bad" — it is "the optimum is sharp," which rows 1–2 already said.

**Corollary (the escape clause):** you cannot perturb your way off a converged optimum. The only exits
are (a) change the objective and RETRAIN, or (b) move more than one axis at once (see Law 2).

## 2. LAW TWO — the co-location law

| granularity | measurement | enrichment |
|---|---|---|
| section (`ar1b` #1213) | what costs most is what hurts most | two granularities, zero remainder |
| class (`bl1` #1206) | Lane = **33.56% of model bits at 0.59% of area** — *and* the worst distortion class (IoU 0.263, ~19% of flips) | 57.31× mean |
| position (`wj1` #1217) | cost × render-manufactured error | **90.96× count / 257.48× bit** over independence |

In a well-matched representation, bits are spent where they **buy** fidelity. Here they are spent where
fidelity **fails**. That is the signature of a **basis mismatch**, not a misallocation.

**Why this explains every Law-1 negative:** allocation moves (waterfill, drop, reorder, coarsen, prune)
redistribute bits *within* a basis. If the basis is mismatched, redistribution cannot help — the bits
are already at the positions that matter, they are simply the wrong *kind* of bits. Eight arms
confirmed this by exhaustion without naming it.

## 3. LAW THREE — the untouched dominant

| object | bytes | % of the 42,382 B demand | role |
|---|---:|---:|---|
| **renderer** | **30,856** | **72.8%** | 78.71% of manufactured seg error born at its native render (`mst1` #1211); **carries d_pose** (#1222 — PoseNet scores rendered FRAMES) |
| carrier | 22,010 | 51.9% | seg-orthogonal (1.7× across all depths); pose 1,527× |
| HPAC model | 13,515 | 31.9% | sharp optimum in every direction tested |

The renderer is the **only object that dominates all three score terms simultaneously** — rate (72.8% of
demand), seg (78.71% of manufactured error), pose (it makes the frames PoseNet scores). **No arm has
ever retrained it.** `ap1`'s semantic group coarsened it (2,169 → 1,513 → 1,451×, improving but
asymptotic) — that is making it *smaller and worse*, never *different*.

## 4. THE SYNTHESIS — and the routing law it implies

Two days of **allocation** work on a **converged** optimum whose **basis** is mismatched, while the
object that **dominates all three terms** sat untouched.

**The falsifiable core, from the campaign's own history:**

- Every **pointer move** this campaign made was a **JOINT** move — `jg2`/`jg3` (seg edit × carrier
  re-solve), `rc4` (token drop × carrier re-solve), `fs3`, `to1` (tail-override × splice).
- Every **loss** of the last two days was a **SINGLE-AXIS** perturbation (§1 table).

This is not coincidence — it is Law 2's operational consequence. **If cost and damage co-locate, a move
that touches only one must lose, and only a move that touches both can win.**

## 5. What this does to `#1221` (jf1)

`jf1`'s mandatory positive control failed by 7,554 B **at epoch 2 of 60**, against a fit that trained
~960 epochs. That is **under-training, published honestly as the arm's first number** — not evidence
against the diagonal.

Re-read under the three laws: `jf1`'s diagonal (field AND model move together) is the **2×2 cell
#1215 named as unentered** — and it is the **same shape as every move that ever won**. jf1 is not "the
last live route by elimination." It is **the only route in the correct class**, and its scary first
number is an artifact of the epoch it was measured at.

Live state at ep24/60: the diagonal leads its own null on **both** axes — tokens 123,387 vs 131,278 B
(gap 7,891 B, stable since ep16) and top1 0.001973 vs 0.002141. QAT engages ep30.

## 6. Named consequences (what fires from this)

1. **`ddm_rj1` — the renderer joint move.** Apply the *only winning move class* to the *dominant
   object*: renderer re-representation × in-compile compensation (qs5-PROVEN, d_pose went BELOW base)
   × carrier re-solve. The unentered cell for 72.8% of the demand.
2. **`ddm_gv1` — the gestalt validation.** Classify EVERY measured move in campaign history as
   {single-axis, joint} × {win, loss}. If the correlation is strong, §4 is a **routing law** and every
   future charter is graded by it. If weak, §4 is pattern-matching on noise and **this memo is wrong** —
   which is the point of running it.

## 7. Honesty boundaries

- §1–§3 are **MEASURED** (every row cited to an arm receipt).
- §4's correlation is **ASSERTED FROM RECALL**, not yet counted — that is exactly what `gv1` measures,
  and it may refute this memo.
- §5's re-read of jf1 is **DERIVED**, not measured: the ep-30 QAT transition is the discriminator, and
  the under-training reading and the won't-reach reading are BOTH still live.
- No pointer claim. No score claim.

## Own-vehicle frontier

**dx2 — S 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, sha `976f706d…` — UNMOVED.
Gap to 0.12: **0.028220**.
