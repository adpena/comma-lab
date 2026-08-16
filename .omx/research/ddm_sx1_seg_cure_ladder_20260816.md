---
arm: ddm_sx1
title: "rt1 reproduces exactly from its receipts, and the charter's premise is three ways wrong: the manufactured seg is 0.028604 S not 0.028155 (96.58%, td1's model ran 1.6% low), there is NO per-stage ladder to build (rt1 measured the v14 taxonomy does not nest -- S1 negative, S2 a 35.4x ceiling, S3 exactly zero), and the cure is not un-built (rg1b landed the edge-weighted objective this morning). The manufacture site is not a pipeline stage: the stock objective spends gradient on the 1-px band in exact proportion to AREA (ratio 1.0016) while 99.22% of the debt lives there. Every seg lever is an alternative on ONE shared support, so the ladder is a MAX not a SUM; the byte-carrying rung is arithmetically dead (eta 1.0069 needed to close the gap, i.e. impossible at perfect realization AND perfect coding); the cheapest live rung is one CLI flag on a receipted tool"
utc: 2026-08-16
charter: "MAIN ddm_sx1 seg-axis charter 2026-08-16 (+ MAIN correction relay, 7 items)"
axis: "[macOS-CPU advisory] frozen CPU-torch SegNet -- NEVER a score. No number here is a score claim."
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "per-rung, stated on each row; no FAMILY verdict is issued that rt1/rg1b did not already carry"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_sx1 — the seg cure ladder

**STORES CONSULTED (read at source, never from a summary):** rt1 memo
`.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md` + its receipts on
`/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816/` (`RT1_INSTRUMENT_CHECK.json`,
`RT1_LEDGER.json`, `RT1_MARGIN.json`, `RT1_LEG_band_r1_a1.json`,
`eta_gate_null/ETA_GATE_{ROWS.jsonl,VERDICT_AGGREGATE.json}`, `eta_gate_free/ETA_GATE_ROWS.jsonl`) ·
rg1b `.omx/research/ddm_rg1b_band_objective_build_20260816.md` · gx1
`.omx/research/ddm_gx1_gap_closure_composition_table_20260816.md` · td1
`ddm_td1_token_drop_schur_arithmetic_20260816.md` · v14
`codex_findings_ddm_v14_realization_fidelity_20260722_codex.md` · rvs1
`ddm_rvs1_realization_survival_harvest_20260811.md` · rvs2
`ddm_rvs2_geometry_survival_crosswalk_20260811.md` · hr1 `ddm_hr1_realization_engineering_20260811.md` ·
hr2 `ddm_hr2_prestage_build_20260811.md` · qs5 `ddm_qs5_verdict_and_no_toy_enforcement_20260813.md` ·
et1 `ddm_et1_eta_on_the_priced_band_20260803.md` · st2 `ddm_st2_lane_stroke_recovery_20260804.md` ·
wd3 `ddm_wd3_n120_family_disposition_20260816.md` · AA reconciliation
`aa_feasibility_reconciliation_20260702.md` · `upstream/modules.py` · `upstream/frame_utils.py` ·
`src/tac/pr130_lift/train_semantic_quantized_resumable.py` ·
`src/tac/pr130_lift/lifted/semantic_renderer_oracle.py` · memories [[m88]] [[m89]] [[m91]] [[m95]] [[m96]].

## ANSWER FIRST

**rt1's decomposition reproduces exactly.** I recomputed every headline from the JSON receipts, not
the prose: 33,743 round-trip flips, 0.028604295518663194 S, 96.58% of the seg axis, advisory/CUDA
ratio 1.000213. No number moved.

**The charter's premise is wrong in three ways, and the third is the one that matters.**

1. **The figure is stale.** "95%, 0.028155 S" is td1's *modelled* value. rt1 *measured* **0.028604 S
   = 96.58%**. td1's model ran **1.6% low**. The real quantity is **2.980×** the gap, not 2.9×.
2. **There is no stage ladder to build.** The charter asks me to walk "each stage where seg error is
   manufactured" — exact mask → painted RGB → R → uint8 → argmax. rt1 **measured that this taxonomy
   does not nest on this vehicle**: S2 flat paint reads back **35.4× worse** than the trained render,
   so S1 is **negative** (−1,161,920 flips); S3 (the R operator) supplies **exactly 0 flips**. The
   stages are alternatives, not layers. A per-stage ladder would be an arithmetic on an object that
   does not exist.
3. **The cure is not un-built.** The charter says "Nobody has built the cure." **rg1b built it this
   morning** — `src/tac/pr130_lift/band_objective.py` + a debt table derived from rt1's own n600
   payloads, wired into the live trainer at `train_semantic_quantized_resumable.py:1179/1231/1259`
   behind `--band-objective-weight` (default 0.0). It ran once and did not descend, and rg1b
   adjudicated the **judge**, not the objective, as the binding defect.

**So where IS the error manufactured?** Not in a pipeline stage. **In the training objective's
capacity allocation.** rg1b measured it: the stock loss puts **2.161%** of its gradient mass on the
1-px label band, and that band is **2.157%** of pixels — ratio **1.0016**. The objective allocates
learning by **area** while **99.22%** of the debt sits on a curve. That is a **45.9× misallocation**.
The renderer is not failing to render; it is being told the wrong thing to care about, in exact
proportion to how much of the picture each pixel occupies.

**The ladder is a MAX, not a SUM.** Every seg lever — flat paint, band repaint, solved correction
channel, α-nudge, the band objective — acts on the *same* 2,551,464-pixel ring-0 support. They are
alternatives competing for one curve. Only two pools are genuinely disjoint: the round trip
(33,743 flips, 0.028604 S) and the label plane (1,717 flips, 0.001456 S).

**The byte-carrying rung is dead on arithmetic, and I can now close it harder than rt1 did.** The
score function permits **1.2731 bytes per flip recovered**. rt1's best *modelled* coder spends
0.8661 B/flip, which looks like room — until you carry it through: closing the whole gap needs
**η = 1.0069**. That exceeds 1. **The correction channel cannot close the gap at perfect realization
AND perfect free conditioning.** Its ceiling is −0.009396 S = **97.9%** of the gap with zero margin
for any realization loss whatsoever. Measured η is 0.6235.

**The cheapest live rung is one CLI flag on a receipted tool**: the **α-ladder** on the free-band
anchor nudge. rt1 ran `--alpha 1.0` only (confirmed: one leg on disk, `RT1_LEG_band_r1_a1.json`) and
measured +1.3808 S. The tool already exposes `--alpha` at
`experiments/ddm_rt1_seg_roundtrip_decomposition.py:871`. **The sign of ∂d_seg/∂α at α→0⁺ is
undetermined by every measurement we own**, it costs zero bytes, and one run settles a family.

**Pointer UNMOVED.** This unit measured and priced. It did not move the score and was not permitted to.

## §0 Prior-law prediction lines (stated before the arithmetic, per the anti-re-anchor law)

1. **et1's radius law** — break-even η rises monotonically with band radius (r=1 0.61491, r=2
   0.65755, r=3 0.67888): "the band cannot be widened into profitability." PREDICTION: any rung I
   price that widens support will lose, and my ladder should contain no dilation rung. **HELD** —
   rt1's own r-ladder independently found r=1 optimal (η 0.6216 vs r=2 0.3243).
2. **gx1's portable constant** — dS/dflip = 8.477105e-07, dS/dbyte = 6.658590e-07, breakeven
   0.785479 flips/B, and every pre-registered bar in this campaign is that constant restated.
   PREDICTION: rt1's η bars will re-derive from it exactly, with no free parameter. **HELD** — see
   §5.1; I reproduce rt1's 0.7531 to four digits from bytes alone.
3. **m91 / pc2 hub law** — seg is one graph with one hub. PREDICTION: the ladder's rungs will not be
   separable per class. **HELD, and strengthened**: they are not separable at all (§4).
4. **qs5** — frame-1 seg edits carry ~zero pose tax when compensation is solved in-compile.
   PREDICTION: pose will not be the binding veto on any seg rung I price. **HELD** — and rt1's η
   data adds a mechanism I did not predict (§5.2).

## §1 Verification — rt1 reproduces exactly

Recomputed from the receipts, independently of rt1's prose.

| quantity | rt1 memo | my recomputation | source |
|---|---:|---:|---|
| dS per flip = 100/117,964,800 | 8.4771e-07 | **8.477105e-07** | derived |
| round-trip flips | 33,743 | **33,743** | `RT1_INSTRUMENT_CHECK.json` |
| round-trip S | 0.028604295518663194 | **0.028604296** | flips × dS |
| seg axis S (advisory) | 0.0296173095703125 | **0.029617310** | 34,938 × dS |
| round-trip share of seg | 96.6% | **96.58%** | 33,743 / 34,938 |
| advisory / contest-CUDA | 1.000213 | **1.0002131** | 2.96173e-4 / 2.9611e-4 |
| td1 label control | 1,717 exact | **1,717** | scorer-free set diff |
| round trip / gap | 2.98× | **2.980×** | 0.028604 / 0.0095973 |

The instrument sits **0.021%** from the contest-CUDA seg term. That is the tightest CPU↔CUDA
agreement the seg axis has shown in this lineage, and it is what licenses the advisory rows below.
It remains `[macOS-CPU advisory]` — a yardstick, never a score.

**One thing I checked that rt1 did not claim.** rg1b independently recomputed rt1's band from the
trainer's own target field and hit **2,551,446 px vs rt1's 2,551,464** — a **72-px disagreement in
117,964,800 (6.10e-7)**, traced to a different GT cache. It is 7e-6 of the band. It changes nothing
here and I record it rather than smooth it.

## §2 The charter's premise, corrected

| charter says | measured | delta |
|---|---|---|
| "95% of seg error is manufactured" | **96.58%** | td1's model ran 1.6% low |
| "0.028155 S" | **0.028604 S** | +0.000449 S understated |
| "2.9× the entire remaining gap" | **2.980×** | — |
| "each stage where seg error is manufactured" | **the stage taxonomy does not nest** | S1 negative, S2 a ceiling, S3 exactly 0 |
| "Nobody has built the cure" | **rg1b built it 2026-08-16** | commit `60aefac081` |

The charter also asks me to consult **#1042 qs5**, **#932/#927 et1**, **#624 v14**, **#149**,
**#1016–#1019**. Per [[m89]] and MAIN's correction 7, those bare ids do **not** resolve in this
repo's ledger (`.omx/state/canonical_task_status.jsonl` holds 205 ids, max 1029; none of those
appear). They live in the harness TaskList only. Every one of them is cited by **memo filename** in
STORES CONSULTED above, and any successor charter should do the same.

## §3 The manufacture site — it is the objective, not a stage

rt1 answered *where on the picture*. rg1b answered *why it gets there*. Together they close the
mechanism question the charter asked.

**Where (rt1, MEASURED n600):**

| ring from transmitted label edge | 0 | 1 | 2 | 3 | ≥4 |
|---|---:|---:|---:|---:|---:|
| pixels | 2,551,464 | 2,132,982 | 1,965,832 | 1,842,841 | ~109.5M |
| flips vs GT | **34,666** | 212 | 37 | 14 | **9** |

The interior — 104,337,564 px, 88.4% of the field — carries **7 flips**. The boundary ring flips at
**203,000×** the interior rate. And the failures are not gross: **98.29%** of flips already have the
wanted class in **second place**, at a median logit deficit of **0.1051**, while the pixels we get
right sit at margins of **3–10** (95.36% of correct pixels are at margin ≥3).

**Why (rg1b, MEASURED on 8 real n600 label fields):**

| curriculum phase | stock grad mass on band | band area share | ratio | angle to debt-weighted gradient |
|---|---:|---:|---:|---:|
| ce (step 0) | 2.157% | 2.157% | 1.000 | **83.31°** |
| softplus_margin (400) | **2.161%** | 2.157% | **1.0016** | **83.30°** |
| expected_flip (560) | 2.166% | 2.157% | 1.004 | **83.29°** |

The trainer allocates gradient by **area**. The debt lives on a **curve**. 45.9× misallocation,
83.3° apart. rg1b's line — *"no learning rate rotates a gradient 83°"* — is the whole diagnosis.

⚠ **Scope, carried from rg1b:** the weight-mass and area figures are exact on real label fields; the
**83.3° uses synthetic logits**, so it bounds the direction error at a representative operating
point rather than measuring the trained gradient. `verdict_scope: INSTANCE`.

**This is why the stage taxonomy could never have nested.** There is no stage that "adds" error. The
render is a near-tie everywhere on one curve because nothing in its objective ever paid attention to
that curve. Flat paint is worse (35.4×) precisely because the trained render, misallocated as it is,
still put the scorer within 0.1 logits of the answer at half the failure sites — which no palette can.

## §4 The joint-support law — the ladder is a MAX, not a SUM

MAIN's correction 1 is the binding structural constraint on this unit, and it is sharper than
"do not overcount."

**Every seg lever in the corpus acts on ring 0.** Flat paint (§2.5 of rt1), band repaint (§2.7),
the solved correction channel (§6), the α-nudge (§6 here), et1's priced band, sq1's solved paint,
st2's stroke composites, and rg1b's band objective all take the *same* 2,551,464-pixel support as
their actuator. They are **alternatives on one curve**, not layers. Their recoveries **cannot be
summed**. The correct composition operator is **max**, and a successor that adds two band rungs has
double-counted the same pixels.

**Exactly two pools are disjoint**, and this is the whole additive structure of the seg axis:

| pool | flips | S | actuator plane | status |
|---|---:|---:|---|---|
| **round trip** (render argmax ≠ shipped label) | 33,743 | **0.028604** | the render / its objective | the ladder below |
| **label plane** (shipped label ≠ GT) | 1,717 | **0.001456** | the token plane | td1 priced ≈685 flips = 0.00058 S recoverable at r_obs 0.8492 |
| *(off-band residual, inside the round trip)* | 272 | 0.000231 | unaddressable by band support | noted, negligible |

Sum: 0.028604 + 0.001456 = 0.030060 ≈ the 34,938-flip advisory seg term (0.029617) plus the 1,458
double-counted sites rt1 measured in §3.4. **The two pools are the only legitimate addition on this
axis.** Everything inside the round-trip pool is a max.

**One more non-additivity the ladder must respect.** rg1b measured that **debt density ranks nothing
like flip share**: Road↔Lane is **#1 by flips (43.4%)** and **#8 by density**, because its band is
1,143,639 px — 44.8% of the whole band. Movable↔MyCar is **10× denser**. A rung that "targets
Road↔Lane" is targeting *share*, and a rung that weights by *density* will not target Road↔Lane at
all. These are different actuators and rc2 already refused the substitution. My ladder names which
one each rung uses.

## §5 THE CURE LADDER — priced

Common arithmetic, from gx1's portable constant (a property of the score function, so it transfers
exactly): `dS/dflip = 8.477105e-07` · `dS/dbyte = 6.658590e-07` · **breakeven = 1.2731 B per flip
recovered** (= 1/0.785479 flips/B).

**What the gap actually demands.** To close −0.0095973 on seg alone requires **11,321 flips** =
**32.40%** of the seg axis = **33.55%** of the round trip. From rt1's measured deficit histogram,
those are the cheapest 33.55%, which requires moving the SegNet logit by **0.0608** — against a
field whose correct pixels sit at margin 3–10, i.e. **49–164× larger**. *That asymmetry is the
reason any of this is conceivable.* (DERIVED: linear-in-bin inverse CDF over rt1's
`deficit_hist_at_flips` = [2384, 4268, 10614, 12261, 5162, 235, 14, 0] on bins
[0, .01, .03, .1, .3, 1, 3, 10].)

### §5.1 The byte-carrying rungs — DEAD, and now closed on a ceiling rather than a bar

| rung | bytes | η | ΔS seg | ΔS rate | ΔS pose | **NET** | % of gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 measured coder @ measured η | 33,235 | 0.6235 | −0.018323 | +0.022130 | −0.001289 | **+0.002519** | −26.2% |
| R1 measured coder @ η = 1.0 | 33,235 | 1.000 | −0.029387 | +0.022130 | 0 | **−0.007257** | 75.6% |
| R1 **best modelled** coder @ η = 1.0 | 30,023 | 1.000 | −0.029387 | +0.019991 | 0 | **−0.009396** | **97.9%** |

**η required at the best modelled coder** — to break even **0.6803** · to supply half the gap
**0.8436** · **to close the gap 1.0069**.

**η > 1 is impossible by construction.** η is the fraction of described flips actually realized
after collateral; it cannot exceed 1. So the correction channel, given a *perfect* solver and the
*theoretical ceiling* of all free conditioning, still **cannot close the gap**. Its ceiling is
97.9% with **zero margin** for realization loss. Measured η is 0.6235 (n=9, sd 0.071, **0 of 9**
above the bar; best single pair ever observed 0.7200).

This closes rt1's own **third reopening condition** (av3's "a better-tuned solver reaching η >
0.753") on arithmetic rather than on effort. A swept solver that reached 0.753 would buy break-even.
To become a half-gap supplier it would need **0.8436** — **+0.124 above the best single pair ever
measured**, on a quantity whose r-ladder optimum rt1 already located (r=1). I do not recommend the
sweep, and I state the bar so that anyone who fires it knows what it must clear.

**A correction I owe rt1.** Its §6.4 argued the curves "do not cross" by comparing a required
**12.09%** byte reduction against the **12.2%** free-conditioning ceiling — which reads as *barely*
not crossing. That comparison mismatches objects: the 12.2% applies to the **mask** (33,082 →
29,058 B); the 965 B target-class cost does **not** shrink with it. On the **total** the ceiling is
**9.66%** (33,235 → 30,023 B). rt1's conclusion is right and its stated margin **understates its own
case** by 2.4 percentage points.

`verdict_scope: FORMULATION` — post-hoc byte-carrying correction addressed on the ring-0 band, at
this byte model. Reopenable only by an addressing scheme with a structurally different cost law
(et1's ph1 block16 phase field is the named survivor at a 3.6× lower bar — and it is **pose-blocked**,
not seg-blocked).

### §5.2 Why relaxing the pose constraint does NOT help — a measured counter to the obvious move

qs5 PROVED that in-compile Schur compensation drives d_pose **below** base on frame-1 seg edits, so
the natural move is: drop rt1's pose-null projection, solve freely, compensate in-compile, and
recover η. **rt1's own paired rows refute it.**

| pair 33 | η_net | d_pose ratio | source |
|---|---:|---:|---|
| pose-null constrained | **+0.6531** | ×1.582 | `eta_gate_null/ETA_GATE_ROWS.jsonl` |
| free / unconstrained | **+0.5714** | **×396.2** | `eta_gate_free/ETA_GATE_ROWS.jsonl` |

Removing the constraint made η **worse**, not better. rt1's §6.3 gives the mechanism and it is not
about pose at all: **η is capped by collateral, and the pose-null projection happens to be a useful
regularizer** — it restricts the edit to directions the pose head cannot see, which also limits seg
collateral. qs5's compensation does **not** restrict direction; it repairs pose afterward. So it
would buy the free-solve η (0.5714), not the constrained one.

**Consequence:** qs5 removes the pose *veto* on seg rungs — MAIN's correction 6 stands and I rely on
it — but it does **not** raise η. Anyone who assumed qs5 unlocks the correction channel should read
this row. `verdict_scope: INSTANCE` (one pair, both modes, rt1's solver budget).

### §5.3 The zero-byte rungs — where the arithmetic is favourable

Bytes are the killer above. A zero-byte rung inverts the problem: **any** realized recovery is pure
gain, and the bar collapses from η > 0.68 to **η_effective > 0.3355 of the round trip**.

| rung | actuator | bytes | ΔS at stated recovery | status |
|---|---|---:|---:|---|
| **R2 flat prototype paint** | whole frame | 15 | **+0.985** (35.4× worse) | **CLOSED**, rt1 measured, `FORMULATION` (v14 #603 re-confirmed vs a 35× stronger competitor) |
| **R3 flat-anchor band repaint, α=1** | ring 0, class-mean palette | 15 | **+1.3808** (47.6× worse) | **CLOSED at α=1**, rt1 measured, `INSTANCE (α=1, r=1)` |
| **R4 α-ladder on R3** | ring 0, class-mean palette, α<1 | 15 | **UNMEASURED** | **the cheapest live rung — §6** |
| **R5 solved-prototype ordered camera paint** | whole frame, prototypes solved vs frozen head | ~15–60 | v14 measured **−0.00212 S** on its own vehicle (0.029593→0.027470) | **LIVE but ancestor-scoped** ([[m18]]: numbers do not transfer; the *mechanism* does) |
| **R6 band objective in training** | ring 0, debt-**density** weighted | **0** | ceiling **−0.028604** (298% of gap); needs only 33.55% realized | **BUILT (rg1b), judge defective** |
| **R7 sub-pixel / AA edge placement** | camera-res pre-R | 0 | — | **CLOSED for fixed prototypes** (st2 +0.1255 S; AA supersample −49% and 41.3 min > budget); rt1 measured **R supplies exactly 0**, so there is nothing for it to recover |

**R6 is the top of the ladder and the only rung whose ceiling exceeds the gap.** Zero bytes,
2.98× the gap available, and it needs to realize only **one third** of it. Its cost is not collateral
(a GT-aimed training objective is aligned with the metric, unlike a post-hoc edit) but **capacity**:
the renderer trades boundary margin against whatever it currently spends that capacity on. That
trade is measurable only through a real run.

**R6's blocker is the judge, not the objective.** rg1b ran it once (α=1, 600 steps) and measured
`improved_over_init = NO`. But it landed at **−0.871σ** on the 5-arm displacement law
`peak_flips = 118,563.2·‖Δw‖₁₀₀^0.457640` (r² 0.9969, σ_log 0.0728) — better than the worst stock arm
at +1.070σ. rg1b's adjudication, which I accept and do not re-derive: in this regime **flips are a
function of weight displacement alone**, so a 600-step flip-trajectory probe **cannot distinguish
objectives at all**. The probe design is invalidated; the objective is not. rg1b's own next step —
compare arms at **matched ‖Δw‖**, and re-score existing arms by their **residual off the law** at
zero cost — is the correct repair and it is **rg1b's to fire, not mine**.

⚠ **R7, stated honestly.** The charter (via v14/#149) expects a pre-R sub-pixel rung. There is none.
rt1 measured **S3 = 0 flips, bit-identical argmax fields** on pairs 0 and 7 — the nearest-neighbour
lift, the evaluator's bilinear downsample and uint8 are *transparent* to a piecewise-constant field.
The scope limit rt1 flagged is real (this is measured on paint, not on the textured render, and
there is no "before R" version of the render to score), but the direction of the evidence is
unambiguous and st2/AA both measured the composite forms losing badly. #149's placement law
**survives** and is **not a supplier** on hv1.

### §5.4 The disjoint pool — the label plane

The only lever that adds rather than competes. td1 priced it: **807 correction sites recover ≈685
flips = 0.00058 S** at `r_obs` 0.8492, *and* save bytes. **6.0% of the gap.** Not worth a row alone,
worth folding into any token-plane work. This is ra1/gx1 territory, not mine; I record it so the
composition table has the disjointness proof.

## §6 The cheapest rung, and its falsifier

**RUNG: R4 — the α-ladder on the free-band anchor nudge.**

**Why this one.** It is the cheapest measurement on the ladder by a wide margin: **one CLI flag on a
tool that already exists, is receipted, and has run at n600.** It costs **zero new bytes** (the 15 B
palette is already counted and already receiver-legal). It sits on the largest un-harvested quantity
in the campaign. And it is **genuinely unmeasured**: I confirmed on disk that exactly one band leg
was ever produced — `RT1_LEG_band_r1_a1.json` / `argmax_band_r1_a1.npy`, `alpha: 1.0` — while
`experiments/ddm_rt1_seg_roundtrip_decomposition.py:871` has exposed `--alpha` (default 1.0) the
whole time.

**Why the sign is genuinely open.** Two first-order terms fight, and every measurement we own sits
at α = 1 where only one of them is visible:

- **The benefit is first-order in α and directed.** 99.22% of flips are on ring 0; the round trip is
  by definition *the render disagreeing with its own transmitted label*; the receiver owns that
  label for free. Nudging toward it is aimed at the residual. And it is aimed correctly 95.1% of the
  time, since only 1,717 of 34,938 sites have a wrong label.
- **The damage is also first-order in α, and undirected.** A local flat patch destroys the texture
  SegNet's region-level reading depends on. rt1 measured the extreme form: at α=1 the band repaint
  (1,663,803 flips) is **worse than repainting the whole frame flat** (1,196,248) — a local edit is
  not local to the scorer.

A pure margin model of the collateral (correct pixels with margin < δ, from rt1's
`margin_hist_at_correct`) predicts **20,985 correct pixels at risk against 11,321 flips recovered**
at δ = 0.0608 — 1.85× against for an *undirected* nudge, but ~43:1 *in favour* for a label-directed
one, since a directed nudge can only break the ~263 sites where the render currently beats a wrong
label. **The measured α=1 point is far worse than either model predicts.** That gap *is* the
texture/region-evidence term, and nothing we own measures its exponent in α. If it is quadratic or
worse near zero, a small α wins; if it is linear with a large coefficient, no α wins. **One run
resolves it.**

**THE FALSIFIER (pre-registered, precise).**

> Run `experiments/ddm_rt1_seg_roundtrip_decomposition.py --leg band --radius 1 --alpha α` for
> **α ∈ {0.02, 0.05, 0.10, 0.20, 0.40}** on a **seeded-random or strided n ≥ 120** pair set
> (**never a prefix** — [[m88]]/[[m96]]; seg prefixes measure ≈0.96× easier), same instrument pins as
> rt1 (frozen CPU-torch SegNet, batch = 1 pair, `torch.set_num_threads(8)`, `preprocess_input`
> verbatim — [[et4]]: batch shape is part of the instrument). Baseline is the α = 0 row on the
> identical pair set.
>
> **REFUTED — the family is closed — iff `d_seg(α) > d_seg(0)` for every α tested**, i.e. the
> measured `∂d_seg/∂α` is positive on the whole ladder down to 0.02. That closes every uniform
> zero-byte band edit on this vehicle at first order, with `verdict_scope: FORMULATION` (uniform
> class-anchor nudge on ring 0), and rt1's α=1 point stops being a single point and becomes a curve.
>
> **SURVIVES iff any α gives `d_seg(α) < d_seg(0)`.** Then the rung is worth its full treatment,
> and the admission arithmetic is already fixed and needs no new bar: at 15 B of palette the rate
> cost is **9.99e-06 S**, so the rung **admits at any recovery above 12 flips** and **closes the
> entire gap at 11,321 flips = 33.55% of the round trip**.

**Expected value, stated honestly: LOW-to-MODERATE and I will not oversell it.** Three independent
measurements point negative — rt1's α=1 (+1.3808 S), sq1's truth paint (η −3.7640, 0/32 pairs
helped), et1's re-measured truth paint (η −2.9861). All three are α=1 **content replacements**. The
rung is worth firing not because it is likely, but because it is the **cheapest** unresolved question
on the largest quantity in the campaign, and because a NO converts three scattered α=1 points into
one closed family — which is a real result under the charter's own terms.

**The highest-EV rung is a different one, and it is not mine to fire.** That is **R6**, the band
objective, at rg1b's corrected judge (matched ‖Δw‖; residual-off-the-law re-scoring at zero cost).
Zero bytes, 2.98× the gap available, needs one third realized. rt1's follow-on #3 routes it "into the
live wd3 / ns1-P1 line as a named objective term, **not a new arm**" — and rg1b has already put it
there. **I recommend no new arm for it.**

## §7 What this unit did NOT establish

- **No new measurement.** Every number here is rt1's, rg1b's, or arithmetic on theirs. I verified;
  I did not measure. The three genuinely new quantities — 11,321 flips / 0.0608 logits / η = 1.0069 —
  are **DERIVED** from receipted measurements, not observed.
- **No sign on ∂d_seg/∂α.** §6 states the falsifier; it does not answer it. The rung is *named*, not
  *resolved*.
- **No claim that R6 will descend.** Its ceiling is measured; its realization is not. rg1b's one run
  is uninformative by rg1b's own adjudication, and I did not re-adjudicate it.
- **No exponent on the texture/collateral term.** I identify it as the gap between the margin model
  and the measured α=1 point. I did not measure its shape, and that shape is exactly what decides §6.
- **No verdict on the label plane.** td1's 0.00058 S stands unre-priced by me.
- **R7's scope limit is unclosed.** rt1's S3 = 0 is measured on piecewise-constant paint. R remains
  formally unmeasured for the textured render, and no experiment in this repo can separate it,
  because the render exists only at camera resolution.
- **No score.** `[macOS-CPU advisory]` throughout. **Pointer UNMOVED at S 0.15959729295498598
  @ 182,759 B [contest-CUDA T4 n600].**
