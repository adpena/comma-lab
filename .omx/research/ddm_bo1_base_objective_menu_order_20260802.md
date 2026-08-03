# ddm_bo1 — the BASE OBJECTIVE: derivation, menu, and order

- arm: `ddm_bo1` (base objective) · date_utc: 2026-08-02
- scope: operator directives 2026-08-02 (acceptance rule · formulation-negative follow-ons are P0 ·
  coordinates/level/cell/contiguity/dimension) + `ddm_cv1` §1/§11/§12.
- axis: every number below is either a **frozen-scorer structural fact** (computed from
  `upstream/models/posenet.safetensors` + `upstream/modules.py`, no video, no vehicle, no run) or a
  quoted prior receipt carrying its own axis. `score_claim=false`, `promotable=false`.
  **Pointer UNMOVED.** This unit produced no exact row and does not claim one.
- receipts: `tools/ddm_bo1_posenet_pair_geometry.py` · `reports/ddm_bo1/posenet_pair_geometry.json` ·
  `tools/tests/test_ddm_bo1_posenet_pair_geometry.py` (24 tests).

---

## §0 The answer, first

> **§0 REVISED IN-SESSION after §7's reconciliation against `ddm_ja1` / `ddm_ea1` / `ddm_cr2r`.**
> The first draft justified the verdict partly on "pose is therefore repairable". `ddm_cr2r`
> **measured** the opposite on this vehicle (a seg-only base is pose-hostile by **6.36×** warp-base /
> **148.98×** v4c two-plane). Both are true and §7.2 resolves them. The verdict survives; its
> *reason* changed, and the change matters more than the verdict.

**The pair-coherence base term that `ddm_cv1` §1 named as the unexhausted move should NOT be built —
because the obstruction it was designed to fix is NOT in the base, and not in the scorer. It is in
the CARRIER's reach.** Three grounds:

1. **MEASURED (frozen-scorer structural):** across the frame_1 perturbation directions that carry
   pose energy, **0.00% lie in directions where frame_0 cannot buy the pose damage back** — for a
   δ_0 free to be *anything in range(A_0)*. Median cancellability 99.96%; at p99 the unrestricted
   optimum still removes 68.5%. **The scorer imposes almost no obstruction.**
2. **MEASURED (`ddm_cr2r` / `#889`), and this is the load-bearing one:** the *shipped* carrier
   nonetheless leaves 6.36–148.98× of pose debt on a seg-only base. Since (1) says the scorer is not
   the obstruction, **the obstruction is the gap between the shipped family F and range(A_0)**. A
   loss term added to the base cannot close a gap that lives in the carrier's degrees of freedom.
3. **Operator directive 2026-08-02:** seg and rate are the decision column; pose debt is admissible.

So the base objective is **seg + rate only**, and the derivation's two products are:

- a **placement law** — *which of the four control quarters each term may spend* (§3.2). Frame_1's
  Q3 (294,912 dims) is **exactly** seg-active and **exactly** pose-free, and no flag expresses it.
  This is not a new principle: it is the measured instance of the already-canonical
  *"gauge-split / range(A) projection FIRST — free, shrinks all downstream"* rule (§7.1).
- a **redirect** — the P0 pose work is **widening F toward range(A_0)**, not adding a base term.
  §5 F3 measures where F actually sits, and that number prices the whole redirect.

The single highest-value thing this unit found that is not already in the config census:

> **The base has never been told WHERE to spend.** 25% of the pair's post-resize control DOF (Q3:
> frame_1 yuv6-null, 294,912 dims) is **exactly** seg-active and **exactly** pose-free — a linear
> kernel, no linearisation, any amplitude. The burn spends seg wherever gradient descent happens to
> push, which is overwhelmingly Q4 (the one coupled quarter). Spending Q3 first is free by
> construction. **No flag in the 60-row config census expresses this choice.**

---

## §1 (A) The derivation — from the frozen scorers, not by analogy

### 1.1 The exact factorisation (VERIFIED_VIA_SOURCE_INSPECTION + numerically re-derived)

`PoseNet.preprocess_input` (`modules.py:70-74`) resizes each frame to (384,512), applies
`rgb_to_yuv6` per frame, then **channel-concatenates** the pair: `IN_CHANS = 6*2 = 12`. The vision
stem's first block (`vision.stem[0]`) is a `MobileOneBlock` whose branches are
`conv_kxk[0]` = Conv2d(12,64,3,stride 2,pad 1)+BN and `conv_scale` = Conv2d(12,64,1,stride 2)+BN,
with **`identity = None`** (in_chans ≠ out_chans). Both branches are Conv+BN, i.e. **affine in eval
mode**, and the 1×1 at stride 2 samples exactly the centre tap of the 3×3 window. Therefore the two
branches fold into **one 3×3 stride-2 kernel**, and the pre-activation is exactly

```
z  =  A_0 u_0  +  A_1 u_1  +  c ,        u_t := yuv6(R(frame_t))
```

where `A_0`, `A_1` are simply the two 6-channel blocks of that one folded kernel. Everything after is
a fixed map `Phi`. **Fold verified numerically against the live module: max abs err 1.07e-05** (fp32
module vs fp64 fold; relative ~1e-6 on pre-activations of magnitude ~10).

Three consequences, all DERIVED and exact:

- **(C1)** `z == z*` ⟹ `d_pose == 0` **EXACTLY**. No linearisation, no Jacobian, no metric. The pair
  constraint `A_0 δ_0 + A_1 δ_1 = 0` is a *sound sufficient condition*.
- **(C2)** A sum of per-frame objectives (`‖δ_0‖ + ‖δ_1‖`) is also sufficient but **strictly stronger
  than necessary**: it demands both errors vanish, where pose only demands they cancel. The unused
  freedom is exactly the difference between those two conditions. This is the precise form of §1's
  "a per-frame objective is structurally blind to the relation the pose scorer reads."
- **(C3)** `z` is the **deepest level at which the pair's information still survives to the pose
  readout** (directive 3, "meet it where it lives"). Any pair term written above the stem is written
  in the wrong coordinates; any written below it is redundant.

### 1.2 The four quarters — NOT re-derived here; this is prior art

`control_surface_exact_dof_quartering_q3_seg_only_pose_null_20260731` (+ its derivation
`.omx/research/ddm_control_surface_exact_quartering_20260731.md`) already MEASURED the domain split.
I re-derived only its rank claim as an independent check and it agrees exactly:

| quarter | dim / pair | reaches |
|---|---:|---|
| Q1 frame_0 yuv6-null | 294,912 | **NEITHER** |
| Q2 frame_0 non-null | 294,912 | POSE only |
| **Q3 frame_1 yuv6-null** | **294,912** | **SEG only** (`d_pose` exactly 0) |
| Q4 frame_1 non-null | 294,912 | BOTH |

Independent re-derivation (mine, finite-difference on the real `rgb_to_yuv6`, which is
`@torch.no_grad()` so autograd returns zero — a trap worth recording): the per-2×2-block Jacobian is
**6×12 of rank exactly 6**, so nullity 6 per block × 49,152 blocks = **294,912 per frame**. Matches.

What the quartering does **not** contain, and what this unit adds, is the **cross-quarter** question:
given that a seg objective spends Q4, how much of the resulting pose damage can Q2 buy back?

### 1.3 The cross-quarter measurement (new)

For a frame_1 perturbation `δ_1`, define the residual ratio in the stem metric

```
ratio(policy)  =  || A_0 δ_0 + A_1 δ_1 ||²  /  || A_1 δ_1 ||²
```

`ratio = 1` means "as good as shipping a perfect frame_0"; `< 1` means frame_0 actively cancelled
frame_1's pose damage; `> 1` means frame_0 *added* pose damage. Block-diagonalising the stride-2
convolution over the 192×256 output grid (each output frequency couples 4 input polyphase
components, giving a 64×24 block per frame per frequency) and averaging over an isotropic `δ_1`:

| δ_0 policy | mean | p10 | p50 | p90 | p99 | share of pose energy WORSE than a perfect frame_0 |
|---|---:|---:|---:|---:|---:|---:|
| **optimal, anywhere in range(A_0)** | **0.02196** | 0.00034 | **0.00041** | 0.0748 | 0.3150 | **0.00%** |
| δ_0 = +δ_1 (transport the error) | 3.5983 | 2.4153 | 3.7599 | 3.7665 | 4.6810 | **99.94%** |
| δ_0 = −δ_1 (sign-flipped) | 0.33569 | 0.0310 | 0.0318 | 0.6900 | 3.6553 | 9.89% |

Why: the stem's pair read is **common-mode-dominated**, not a difference. `cos(W_f0, W_f1) = +0.8295`;
symmetric part 24.4452 vs antisymmetric 7.4665. And it is **chroma-dominated** — V alone is **81.4%**
of frame_1's stem energy with `cos_V = +0.9767`, U a further 6.1% (`cos_U = +0.7913`), all four luma
phases together only 12.6% (cosines −0.28…+0.02).

**The chroma fact is load-bearing and, as far as I found, unstated:** PoseNet's only pair-mixing layer
spends **87.5% of its weight energy on chroma**, and its two frame-blocks are 97.7% aligned on V.
CLAUDE.md carries "Chroma is a d_seg lever"; at the stem, chroma is overwhelmingly the *pose* lever,
and it is the channel in which frame_0/frame_1 cancellation is nearly perfect.

### 1.4 The carrier's displacement is the real knob (new, and it corrected me)

A warp carrier `f_0 := a·warp_ξ(f_1) + b` does not give `δ_0 = +δ_1`; it gives `δ_0 = δ_1 DISPLACED by
the ego-motion image displacement`. Displacement only changes the cross-term's phase, so:

| carrier displacement (full-res px) | ratio, δ_0 = +transport | ratio, sign-flipped |
|---|---:|---:|
| 0 | 3.5983 | 0.3357 |
| 2 | 1.98 | 1.95 |
| ≥ 4 | **1.9670** | **1.9670** |

At displacement ≥4 px the cross-term averages to zero and both collapse to
`1 + ‖W_f0‖²/‖W_f1‖² = 1.9670` exactly — incoherent addition. **So the sign lever exists only where
the error field is spatially coherent across the ego displacement** — i.e. near the focus of
expansion — and is dead in the periphery. This is directive 3's "contiguous scene block / the same ξ
that already ships" showing up as a hard cutoff, not a metaphor.

### 1.5 What directive (1) changes about the derivation — stated explicitly

Before the directive I was deriving a term whose job was to **hold pose while seg improves**. That
target is gone. What survives is strictly weaker and strictly cheaper: the base does not need a pose
term at all, because §1.3 says the repair channel is geometrically almost unobstructed
(0.00% of pose energy irreducible). The coherence idea survives **only** as an *instrumentation*
quantity — the repair price — not as a loss term. Concretely:

> **DROP** `L_pair` from the base loss. **KEEP** `ratio` as a logged diagnostic so the terminal pose
> solve knows what it inherited. A logged scalar costs zero bytes, zero gradient, and zero risk;
> a loss term costs a hyperparameter, a stage boundary, and a confound.

This is also why directive (2) is satisfied without new training: the formulation-scoped
"joint pose is DEAD" negative's named follow-on was *this* family, and the follow-on's verdict is
"the family is unnecessary on this vehicle", reached at $0.

---

## §2 Round-1 adversarial review of my own derivation

I attacked this before reporting it. Two of my own claims did not survive.

**KILLED #1 — "the shipped carrier is anti-optimal at ratio 3.6."** I inferred this from the carrier's
*form*. It is not supported. `ddm_bp2` measured that splicing a **true-GT frame_0** into the decoded
pair gives `d_pose` **3.05–16.66** against the decoded pair's **0.0008** — the decoded pair is
thousands of times better than a *perfect* frame_0, i.e. our frame_0 is already deep in the
**cancelling** regime (ratio ≪ 1), not the transport regime (ratio ≈ 2–3.6). My inference was a
plausible-mechanism story that the campaign's own n600 receipt contradicts. **Retracted.** What
survives: the *landscape* is mapped; **where the shipped carrier sits on it is OWED** (§5, F3), and
bp2's ratio does not pin it because replacing frame_0 with GT is not a small perturbation at fixed
δ_1, so the two ratios are not the same quantity.

**KILLED #2 — the direction-extremal read.** My first robustness check computed min/max of the ratio
over *all* directions and got `optimal ∈ [1.9e-5, 1.0]`, which looks like "there exist directions
frame_0 cannot fix at all." That read is **degenerate**: the max is attained in directions carrying
essentially zero pose energy (a tiny effect staying tiny). Replaced by the energy-weighted spectrum
in §1.3, which is the defensible instrument. **Recorded because the degenerate read is the one a
reader would naturally reach for.**

**SURVIVING LIMITS, each of which bounds a claim above:**

1. **RUNG 2, not rung 3.** `δ_1` isotropic, L2 energy. The quartering memo's own ladder is
   *dimension count → L2 spectrum → margin/Fisher-weighted (THE object, UNMEASURED)*, and it
   explicitly warns that the coarse read overstated pose reach ~3× there. My numbers inherit that
   warning. **Asymmetric consequence, and this matters:** the *naive-transport-is-bad* row is tight
   across p10–p99 (2.42–4.68, 99.94% of energy) and survives almost any reweighting; the
   *optimal-is-0.0004* row spans ~900× (p10 3.4e-4 → p99 0.315) and does **not**. Quote the optimum
   as "median 99.96% cancellable, p99 68.5%", never as a single number.
2. **Stem metric ≠ d_pose.** `z == z*` ⟹ `d_pose == 0` is exact; *proportionality between the stem
   ratio and the d_pose ratio is ASSUMED* and untested. Every percentage here is in stem
   pre-activation energy.
3. **range(A_0) is not the shipped family.** The optimum is a **LOWER BOUND** over an unrestricted
   δ_0. The 194 B carrier reaches an ~8-dim submanifold of it (directive 3's dimension rule: pose
   quadratic ≤6-dim, head rank-4 — do not write a term with more DOF than that). The restricted
   optimum is UNMEASURED.
4. **Circular boundary** in the polyphase block-diagonalisation: exact on the interior, differs on the
   1-pixel border of the 192×256 grid (~1.8% of positions). The ±δ rows are cross-checked against
   boundary-free Frobenius ratios and agree to 1e-14.
5. **`rgb_to_yuv6` clamps** at 0/255; all rank/kernel statements are the unclamped regime.
6. **Cross-check coverage, stated so nobody over-trusts it.** The tool's internal closed-form check is
   a *Parseval identity*: it validates energy bookkeeping and is **invariant to the polyphase tap
   arrangement**, so a pure index-algebra error would pass it. The index algebra is instead validated
   by an independent spatial-domain reproduction of a real circular stride-2 conv
   (`test_polyphase_blocks_reproduce_a_real_stride2_conv`). Writing that test **found a real
   precondition bug**: the per-direction ratios only span the input space when `M_1` has full column
   rank, which the tool now checks fail-closed (real PoseNet: 64×24, worst conditioning 2.75e-3, so
   it holds — but it would not have held for a narrower stem, and the mean would have been silently
   wrong).
7. **Negative-existence, SCOPED.** I did not find a prior computation of the Q2↔Q4 cancellability, of
   the stem's frame_0/frame_1 channel split, or of the carrier-displacement ratio, in **five
   `corpus_query` passes (top-6…12 each)** over research(7398) / equations(869) / memory(2051) /
   dag(915) / council(292) / tasks(409) / docs(96). Repo-wide literal `grep -rl` **timed out at 100 s
   on 4 tokens** and is therefore not evidence either way. This is a bounded-search negative, not an
   exhaustive one.

---

## §3 (B) The base MENU

### 3.1 What I did NOT do

A base-config provenance census already exists and is good: `ddm_gd1_generic_default_census_20260731`
(19 OBJECT rows T1–T19 + 13 STRUCTURE rows S1–S13, each with a class in
`GENERIC-UNRACED / GENERIC-CHOSEN-UNRACED / GENERIC-SURROGATE / RACED / DERIVED / …`),
`ddm_gd1_undecided_defaults_audit_20260731` (14 ranked rows on a
`NEVER-DECIDED / INHERITED / DERIVED-RACED` axis), `ddm_b2b` (QA86 corrections),
`ddm_b4s` (the sealed burn-4 config), and `c1_config_differential_audit` (the levelset sibling
vehicle, 235 flags / 23 levers). **Re-listing them would be rediscovery.** The live base is
`qa24_composed_burn_program(...)` at `src/tac/witness_dsl/spec_tr1_renderer_20260728.py:665`,
compiling to `experiments/train_tr1_partition_renderer_mlx.py`.

### 3.2 The menu's MISSING DIMENSION — placement

Every row in the census answers *what value*. **Not one answers *which quarter*.** The base loss is
written per-pixel over frame_1's render and descends wherever the gradient points; the Q1–Q4 split is
invisible to it. That is the config expression of directive 3's "CELL, not pixel" and "the right
coordinates".

Three choice points that do not exist and should:

| new choice point | why it is a choice, not a value | cost to add |
|---|---|---|
| **`--seg-spend-quarter`** ∈ `{unconstrained, q3_first, q3_only}` | Q3 is 294,912 dims of **exactly** pose-free seg authority. `unconstrained` is not a neutral default — it is an unpriced decision to spend the coupled quarter first. | one projection in the loss; no new hyperparameter if `q3_first` is a *scheduling* rule, not a weight |
| **`--frame0-carrier-phase`** (sign / displacement policy) | §1.4: the carrier's pose ratio is a known function of its displacement, with a hard cutoff at ~4 px. Today the sign is implicit in the warp and was never chosen. | scalar; rule-118-free (the policy is generic code, only its scalar is counted) |
| **`--pose-repair-price` (telemetry, default ON)** | The repair price `ratio` is the quantity the terminal solve inherits. Read-only ⇒ per CLAUDE.md's "off is a tracked queue" rule, score-neutral telemetry **defaults ON**; it is not gate-able. | logging only, byte-identical |

### 3.3 What the derivation RE-RANKS in the existing census

| census row | current status (quoted) | what §1 changes |
|---|---|---|
| **pose-in-burn** (gd1 row 12) | rule DERIVED from the 5-formulation photometric wall, but *"its **precondition is contested in the record** and unowned"*; the burn-4 charter says pose-in-burn currently **REQUIRED** | **ADJUDICATED, and it is a $0 read as gd1 said.** With 0.00% of pose energy irreducible **and** directive (1), the burn is **seg + rate only**; pose is a terminal repair. The contested precondition resolves *against* pose-in-burn. This is the single highest-value row the derivation touches. |
| **`--renderer-head-mode`** `rgb` / `class_field` / `class_field_photo` (b2b, pre-registered A/B/C, **not measured**) | race BUILT not MEASURED | This is the *level* question in directive 3. `class_field` is the cell-level head; `rgb` is the pixel-level head. §1.2 says the pose-free authority (Q3) is a **chroma-at-fixed-luma** subspace, which a `class_field` head cannot express at all and `class_field_photo` can. **Re-rank: the head race is a placement race, and it should be scored on Q3 reach, not only on seg.** |
| **`--w-rate 0.05`** vs derived **0.0768** (`DERIVED-ESTIMATE`, b4s holds 0.05 on constants-are-poison) | live 0.05 ≈ 65% of the S-commensurate value | Untouched by this derivation. Flagged only because rate is now half the decision column and a 35% under-weight on the *only* directly-priced axis is a larger error under the new acceptance rule than it was under the old one. |
| **`--margin-weight-temp 1.0`** (`RACED-NOT-ASSERTED`) | the SCALE temp has no provenance rung | The margin field is the correct rung-3 weight for §1.3. If it is mis-scaled, the rung-3 re-measurement (§5 F1) inherits the error. **Sequence F1 after this row is settled, or report both.** |
| GENERIC-UNRACED render-path rows T1/T2/T10/S1/S2/S3 (activation, output squash, padding, lattice, upsample, topology) | all `GENERIC-UNRACED` | All live **in Q4** (they shape RGB directly). Under a placement law they become *lower* priority than the Q3 question, because Q4 spending is the only kind that incurs a repair bill. **Deprioritised, not killed.** |

### 3.4 The one menu row I would add to the P0 queue

**`--seg-spend-quarter q3_first`**, because it is the only row here whose value is bounded *below* by a
structural fact rather than by a guess: seg spent in Q3 costs **exactly zero** pose, at any amplitude,
on any base. Its risk is bounded too — the quartering memo's own **OPEN LIMIT 1** is that *reaching*
SegNet's input at amplitude 6.0/255 is not the same as *moving its argmax*, and **OPEN LIMIT 2** is
realizability at camera-res uint8 (needs `δY_cam ∈ ker(R)`; reachable fraction UNMEASURED; the
best pointwise-isoluminant integer step in ±12 is `(−6,+5,−10)` giving leverage 7083:1). Those two
limits are exactly the falsifier in §5 F2.

---

## §4 (C) The ORDER

The couplings measured above are **exact**, so the order is derived from them rather than asserted.
The three actuators are exactly orthogonal in their scorer effects:

- **frame_0 (Q1+Q2): pose-only.** Seg obligation exactly zero (`x[:, -1, ...]`).
- **frame_1 yuv6-null (Q3): seg-only.** Pose exactly zero (a linear kernel).
- **frame_1 yuv6-range (Q4): both.** The *only* coupled channel.

**⇒ the coupling is confined to ONE quarter, and the order follows:**

| # | stage | why it is here, and what it must NOT do |
|---|---|---|
| **0** | **format / grammar / codebook CHOICE** (NOT the coder stage) | The description *language* prices everything later, so it is chosen first. **The coder STAGE is terminal, after stage 3** — `ddm_ja1`'s physical order and `oc1` both put entropy-coding last, and my first draft conflated the choice with the stage (§7.3). `oc1` measured the coder saturated (*"RAW wins 50/50"*) and the binding stage as QUANTIZE, so this is a cheap decision. No ordering constraint from the pair geometry. |
| **1** | **seg in Q3 (pose-free)** | Exactly zero pose cost by construction. Any seg obtainable here is unconditionally free, so it must be exhausted before any priced move. **Constraint: must not leak into Q4** — the projection is what makes it free. |
| **2** | **seg in Q4 (priced)** | The only place seg spends pose. Under directive (1) it is **admissible to spend freely** — but it should be *instrumented* with the repair price (§3.2 row 3), because that is what stage 4 inherits. **Constraint: no pose term in the loss here.** |
| **3** | **realization / uint8 window solve** | Must come after 1–2: it solves for the uint8 preimage of a *chosen* field, so the field must be chosen first. `ddm_ll1` measured this closed (88→3 flips, solve 0.07 s/frame vs enumeration 2.04 s/frame). |
| **4** | **pose repair in frame_0 (Q1+Q2)** | **LAST, as a repair.** This stage is NOT mine — it is the canonical staging law (`pose_is_a_terminal_six_equation_solve_on_conditioned_seg_base_20260728`), and §1 supplies only its mechanism and magnitude. Zero seg cost — exactly, not approximately — so it cannot undo stages 1–3. **But see §7.2: the repair is only as good as the carrier's reach, and `ddm_cr2r` measured that reach short by 6.36–148.98×.** "Pose falls out" is a statement about the scorer's geometry, not a promise about the current carrier. |
| **5** | **resolver / base re-resolution** | `ddm_sf1`'s law: any SOLVED or FITTED quantity has a PARTNER it was solved against. Stage 4's solve is against stage 2's base; if the base moves, `(a,b)` are stale. `ddm_uv1` removed the capability limit (`BASES` was a hardcoded 2-entry dict). |

**Ordering claims I am NOT making.** I derived constraints 1→2, 2→3, 3→4 (or 2→4), and 4→5 from the
measured couplings. I did **not** derive stage 0's position from anything in §1 — rate's precedence is
a description-order argument, not a scorer-geometry one, and I am labelling it as such rather than
dressing it up. **Reconciliation against the prior `oc1` / `fc1` / `ddm_ar1` order passes is OWED**
(§5, F4): if any of them prescribes a different order, the difference must be adjudicated at their
sources, not by preferring mine.

**What §1/§11 CHANGE about a prior "dependency-optimal order":** any earlier order that treated
"improve seg" as a single stage is under-resolved. Seg splits into a **free** stage and a **priced**
stage, and only the priced one creates a downstream obligation. And §11's ratio table splits it
again: Road (44% of the seg residual) sits at ratio 1.00 — attacking it is a *phase-faithfulness*
problem, the same axis as the pair relation; Undriv+Movable (22% of the seg residual = 12.9% of the
total gap) are ordinary above-floor optimisation. **So stage 2's first target is Undriv+Movable, and
Road is not an early-stage target at all.**

---

## §5 Pre-registered falsifiers

Registered before measurement, with thresholds, per the standing discipline.

**F1 — rung-3 re-measurement of §1.3.** Replace isotropic `δ_1` with the **margin/Fisher-weighted**
`δ_1` from the live base (the quartering memo's rung 3; `ms3/ms4`'s metric-custody bundle is the
existing producer). *Falsifier:* if the energy-weighted median cancellability degrades from 99.96% to
**below 90%**, or the share of pose energy worse-than-a-perfect-frame_0 rises above **5%**, then §0's
"drop the pair term" verdict is REFUTED and the coherence term returns as a live candidate.
**Cost: $0 given cached margins; needs no scorer forward.**

**F2 — the Q3 seg actuator (the quartering memo's own ONE MEASUREMENT, unchanged).** Perturb frame_1
within Q3 ∩ range(R) ∩ uint8-realizable at a margin floor; measure `Δd_seg` through the real
R→uint8→SegNet path with `Δd_pose = 0` as the positive control. *Falsifier:* `|Δd_seg|` below the
device-tie noise floor ⇒ `--seg-spend-quarter` is inert and §3.4 is REFUTED; stage 1 of the order
collapses into stage 2. **BLOCKED-ON-SLOT** (`ddm_pg1` holds it). Owed measurement stated exactly so
it can be queued.

**F3 — locate the shipped carrier on the §1.4 landscape.** Compute
`‖A_0 δ_0 + A_1 δ_1‖² / ‖A_1 δ_1‖²` on cached decoded frames of the live base. This is **one folded
3×3 conv**, not a scorer forward — but it does need decoded frames, so: **BLOCKED-ON-CACHED-FRAMES,
not on the n600 scorer slot.** *Pre-registered prediction, from §2 KILLED #1:* the ratio is **< 1**
(cancelling regime), **not** ≈2–3.6. If it comes back > 1, the carrier is adding pose damage and
`--frame0-carrier-phase` becomes a P0 zero-byte lever; if < 0.01 the carrier is near-saturated and
that row is DEAD.

**F4 — order reconciliation. DISCHARGED in-session; see §7.** It found the canonical DAG I had not
cited (`ddm_ja1`), one real contradiction in my §4 stage 0, and the `ddm_cr2r` tension that revised
§0. Recorded here as fired rather than deleted, so the sequence is auditable.

**Matched control for any base-objective A/B.** `ddm_cr2r`'s protocol: same solver, **74 matched
pairs**, both arms resolved against their own base (`ddm_uv1`'s `resolve_base()` / `--base-archive`,
so the `BASES` capability limit no longer censors the comparison). Report **seg and rate as the
decision column and pose debt separately**, per directive (1). Do **not** accept a prefix-scoped
verdict: `ddm_bp2` measured a video-order prefix flip a −0.122 S "win" into a +0.152 S loss because
the prefix's mean d_pose was 5.1× the population's (memory `m88`).

---

## §6 S-arithmetic

Against the PR130 demonstrated floor, from `gap_decomposition_against_floor_20260802`
(total gap **0.7263025**; **1% of gap = 10,908 B**):

| axis | gap | share |
|---|---:|---:|
| seg | 0.4015190 | 55.3% |
| pose | 0.2120155 | 29.2% |
| rate | 0.1127679 | 15.5% |

Under directive (1) the decision column is **seg + rate = 70.8%** of the gap; pose's 29.2% is debt
whose repair channel §1.3 shows is geometrically almost unobstructed. Within seg, §11 says
Undriv+Movable = **12.9% of the total gap** is ordinary above-floor optimisation and Road's 44% of
the seg residual is a phase-faithfulness problem.

**This unit moved the pointer by zero.** It removed a planned loss term, adjudicated one contested
config row, named three absent ones, and derived an order. Those are means. The end is a lower exact
score, and it is not lower today.

---

## §7 F4 DISCHARGED — reconciliation against the existing order artifacts

I did not know about the canonical DAG when I wrote §4. Reading it changed three things.

### 7.1 What already existed, and what §4 must therefore stop claiming

- **`ddm_ja1_order_of_operations_dag_20260731.json` (+ `ddm_ja1_joint_atlas_waterfill_20260731.md`
  §6) is THE canonical dependency DAG**, commissioned by ledger row QA73. Physical order:
  `motion (ξ GN solve) → per-depth projection → photometric (a,b + rolling shutter) → uint8
  (through-R) → coder (SMEVR)`, with **measured invalidation edges and re-solve wall-clock**
  (`token_base` invalidates `pose_solve` + `photo_fit` + `selector`, ~1 h + ~30 min; `pose_solve`
  invalidates `photo_fit`, ~30 min; `photo_fit` invalidates nothing).
- **The staging law is canonical and predates me:**
  `pose_is_a_terminal_six_equation_solve_on_conditioned_seg_base_20260728` —
  *"scene carrier → seg corrections → ALL repaints → THEN `solve_pose(frozen_frames, t_p)` as the
  LAST stream written."* **My §4 stage 4 IS this law.** I am not the source of it; §1's contribution
  is the *mechanism and magnitude* under it (0.00% irreducible against range(A_0)), not the ordering.
- **My "placement law" is an instance of an existing rule.** The 2026-07-27 operator memory
  (`we_have_everything_combine_techniques_in_optimal_order…`) already says
  *"GAUGE-SPLIT / range(A) projection FIRST — Free; shrinks all downstream → THAT is why first."*
  Q3-first is that rule evaluated on the yuv6 kernel. **Independent arrival, not novelty** — which
  strengthens it and obliges me to cite rather than claim it.

### 7.2 The `ddm_cr2r` tension, and the resolution that revised §0

`ddm_cr2r`/`#889`: a seg-only base is pose-hostile by **6.36×** (warp base) / **148.98×** (v4c
two-plane), and *"post-hoc pose cannot recover what the burn never carried."* My §1.3: **0.00%** of
pose energy is irreducible. These look contradictory. They are not, and the difference is the finding:

> §1.3's optimum is over **range(A_0)** — δ_0 free to be anything. `cr2r`'s penalty is measured with
> the **shipped carrier family F** (~8-dim: warp ξ + a,b). Both being true means the 6.36–148.98×
> lives **entirely in the gap between F and range(A_0)**, not in the scorer's geometry.

Consequences, stated plainly:
- A **base loss term cannot close that gap** — it constrains δ_1, while the deficiency is in δ_0's
  reachable set. This is now the primary reason to drop the pair-coherence term, and it is a stronger
  reason than the one I first wrote.
- The P0 pose lever is **more carrier DOF**, spent in Q1+Q2 (zero seg cost, exactly). Directive 3's
  dimension rule bounds how much: pose quadratic ≤6-dim, head rank-4 — so the target is a *small*
  widening, not a free-for-all.
- **`ddm_cr2r`'s protocol stands unchanged** as the acceptance gate: matched base, ≥32 pairs
  (74 available), before composing any seg base with any pose stream.

### 7.3 The one place §4 was wrong

**§4 stage 0 ("rate / format / codebook first") contradicts `ja1`/`oc1`, which both put the CODER
LAST.** They are right about the object they name and I conflated two objects:

- the **format / grammar / codebook CHOICE** (what description language exists) must be fixed early —
  it prices every later description;
- the **coder STAGE** (running the entropy coder over a produced field) is necessarily terminal.

Corrected: **stage 0 = format/grammar CHOICE; the coder STAGE runs after stage 3, per `ja1`.**
`oc1` further measured that the coder stage is *"saturated (RAW wins 50/50)"* and that the binding
stage is **QUANTIZE, not PREDICT** — so stage 0 is a cheap decision, not a campaign.

### 7.4 The scope limit I must carry — `ddm_ea1`

`ddm_ea1_einsteinian_negative_audit_20260730` condemns *"polish sequence as global strategy (finish
seg, then pose, then rate as independent campaigns)"* while explicitly **not** condemning
*"the staging law WITHIN a burn."* Its reason is measured: bar-feasibility is a **JOINT co-location
constraint** (seg+pose < 0.055897 at Gate-B-class rate), and *"sequential polish composes to ≥7–10×
over from any measured endpoint."*

> **Therefore §4 is a WITHIN-BUILD order and nothing more.** It is NOT a campaign strategy, and it
> must not be read as "do seg this month, pose next month". The 2026-08-02 acceptance rule loosens
> the *judging* of a move, not the *joint* feasibility constraint. Anyone citing §4 across builds is
> citing it out of scope.

### 7.5 Gaps the reconciliation confirmed

- **The resolver has no placement in any prior order artifact** (it became a named surface only on
  2026-08-02, in `ddm_cv1` §2). §4 stage 5 is therefore new and unreviewed by any prior pass.
- **The realization/uint8 window solve has an exact measured geometry but no placement** in any
  dependency order. §4 stage 3 is likewise new.
- **A mis-citation to fix, not mine:** `ddm_ax1` §7 attributes the
  *"solve → gauge-split → predict → quantize → truncate → entropy-code"* chain to `fc1`; that chain
  is not in the fc1 memo (the token `gauge-split` does not occur in it) and belongs to `oc1` / the
  2026-07-27 operator memory. Flagged for whoever owns `ax1`.

---

## §8 NEXT-IF-RESUMED

1. **F3 first** — $0, one folded 3×3 conv on cached decoded frames. It is now the **highest-value**
   item in this memo, not the cheapest curiosity: §7.2 concluded the whole pose obstruction lives in
   the gap between the shipped family F and range(A_0), and F3 is the measurement that *prices that
   gap*. Prediction already published (ratio < 1, cancelling regime).
2. **F1** — $0, rung-3 re-measurement. Still the only thing that can overturn §1.3. Do it before
   anyone builds on §0.
3. **Widen F, not the loss** — the redirected P0. Add carrier DOF in Q1+Q2 (exactly zero seg cost),
   bounded by directive 3's dimension rule (pose quadratic ≤6-dim, head rank-4). Gate with
   `ddm_cr2r`'s matched-base protocol, ≥32 of the 74 pairs.
4. **F2** — queue behind `ddm_pg1`'s slot; it is the quartering memo's own owed measurement and it
   gates `--seg-spend-quarter`.
5. **Then** build `--seg-spend-quarter` as a DSL `Lever` factory in `spec_tr1_renderer_20260728.py`
   (never a hand-added trainer flag), default `unconstrained` with a recorded reason, registered with
   a duty-to-measure so "off" is a tracked queue state.
6. **Do NOT** build a pair-coherence loss term unless F1 refutes §1.3 *and* F3 shows F already at the
   bound (i.e. the obstruction really is in δ_1 after all). Both, not either.
7. **Hand `ddm_ax1`'s owner the §7.5 mis-citation.**

Cross-refs: `ddm_cv1_seven_surface_convocation_20260802` §1/§11/§12 ·
`control_surface_exact_dof_quartering_q3_seg_only_pose_null_20260731` +
`ddm_control_surface_exact_quartering_20260731` · `ddm_ja1_order_of_operations_dag_20260731` ·
`pose_is_a_terminal_six_equation_solve_on_conditioned_seg_base_20260728` ·
`ddm_ea1_einsteinian_negative_audit_20260730` · `ddm_oc1_xi_temporal_predict_measured_20260727` ·
`ddm_fc1_assembly_capstone_flip_entropy_and_compose_20260728` ·
`ddm_ar1_archetype_codec_priced_spec_20260728` · `ddm_ax1_all_axes_derivation_20260730` §7 ·
`ddm_gd1_generic_default_census_20260731` · `ddm_gd1_undecided_defaults_audit_20260731` ·
`ddm_b2b_burn2_composition_build_20260731` · `ddm_b4s_burn4_charter_20260731` ·
`ddm_cr2r` / `#889` · `ddm_bp2` · `ddm_sf1` · `ddm_uv1` · `ddm_fl1` ·
`src/tac/optimization/ddm_ll1_window_solve.py` · `upstream/modules.py` · `upstream/frame_utils.py`
