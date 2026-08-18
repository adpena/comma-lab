---
arm: ddm_nx1
title: "Audit of every negative and mixed verdict in the 2026-08-15/16 window. One row hides real value: hg1's 'the hinge cannot go on hv1's trainer' is scoped to the TOKEN trainer, while hv1's SEMANTIC-renderer trainer already runs the identical signed-margin term with a frozen SegNet in the loss and a HARDCODED tau that wastes 73.6-90.3% of its gradient mass -- so a sealed 8-hour Metal A/B is aimed at the tr1 vehicle on hv1's arithmetic. Everything else in the window is correctly scoped; the rate closures split into CEILING closures and DISTORTION closures, and conflating the two is the live error"
utc: 2026-08-16
axis: "[macOS-CPU advisory + local $0 source inspection and exact arithmetic over MEASURED receipts] -- NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 ep0634 S 0.15959729295498598 @ 182,759 B [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "stated inline per row; this unit issues no new family verdict"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_nx1 — the negative and mixed-signal audit

**Operator directive 2026-08-16:** *"Must audit all negative and mixed signal as well."* Context:
*"There were also other wins sitting there."* So the job is not a census. The job is to find
frontier-lowering value that a negative verdict is currently hiding.

**Scope of the sweep, named so my negative-existence claims are bounded.** I read every
`.omx/research/*2026081[56]*.md` — **75 files** (excluding this one). Three were untracked when I
started; `ddm_sr1_…` and `ddm_ra2crr_…` landed at 23:14/23:15 while I worked, leaving
`ddm_av3_…_against_all_charter` untracked. I added **15 rows** the assignment did not name.
Anything outside that glob (task ledger, `charters/`, `arm_final_messages/`, code comments) I did
not sweep, and I make no claim about it.

**Frontier, re-derived not quoted (MEASURED).** `.omx/state/canonical_frontier_pointer.json`:
`S = 0.15959729295498598` @ **182,759 B** `[contest-CUDA T4, n600]`, sha `80d9c8c6…`.
Components recompose: `0.029611 + 0.0082945765 + 25·182759/37545489 (= 0.12169171641365491)
= 0.1595972929136549`, agreeing with the pointer to `4.1e-11` (the seg leg's 6-s.f. rounding).
Per-byte `6.658589531221714e-07`. Gap `0.009597292954985986` = **14,413.402 B**. Byte invariant
**168,345.5977 B**.

---

## THE ANSWER, FIRST

**One row hides value, and it is currently mis-routing a Metal slot.**

`hg1` (2026-08-16 20:00) concluded: *"the hinge cannot be added to hv1's trainer at all — there
is no SegNet forward there to hinge on. It belongs in `train_tr1_partition_renderer_mlx.py`."*

**That is true of the trainer hg1 read, and false of hv1.** I verified both halves at source.

- **hg1 is right about the file it checked.** `tools/train_ddm_cl1_hpac_capacity_mps.py` matches
  `segnet|posenet|render|rgb` **0 times**. MEASURED (grep).
- **hv1's archive carries TWO learned objects with TWO trainers.** The token/HPAC half is that
  file. The half that manufactures **96.6%** of the seg debt — the semantic renderer, 34,763 B —
  is trained by `src/tac/pr130_lift/train_semantic_quantized_resumable.py`. That trainer
  **loads a frozen SegNet** (`:1036-1038`), **renders through the exact quantized path**
  (`:1217-1219`, `exact_path=True`), **forwards it** (`:1230`, `logits = segnet(frame)`), builds
  the loss from those logits (`:1255`, `qat.curriculum_loss`), and **backprops through it**
  (`:1293`). MEASURED by source inspection.
- **The term hg1 wants is already the shipped curriculum.**
  `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:174` defines
  `target_margin = logit[GT] − max_{c≠GT} logit` — term for term hg1's signed margin. `:181`
  runs `ce → softplus_margin (τ=0.20) → expected_flip (τ: 0.15 → 0.05)` on it. b2e ran a
  3,000-step burn on exactly this trainer **17 hours before hg1**.

**But hg1's derived defect IS present on hv1** — it lives in a hardcoded `τ`, not in a flag.
I measured it on hg1's own retained n96 margin field
(`/Volumes/APDataStore/pact/ddm_hg1_ring0_margin_hinge_20260816/HG1_MARGIN_FIELD_n96.npy`,
96x384x512 float32). Gradient mass = `|dL/dm|` summed over 18,874,368 pixels; 5,448 are flips.

| objective | active frac | grad mass on flips | **WASTE on already-correct** |
|---|---:|---:|---:|
| hard hinge `T=1.0` (hg1's named defect, tr1 default) | 1.2295% | 2.348% | **97.65%** |
| hv1 shipped `softplus_margin`, τ=0.20 | 30.198% | 15.911% | **84.09%** |
| hv1 shipped `expected_flip`, τ=0.15 (phase start) | 10.091% | 9.698% | **90.30%** |
| hv1 shipped `expected_flip`, τ=0.05 (phase end) | 2.245% | 26.364% | **73.64%** |
| hard hinge at hg1's DERIVED `m_safe = 0.03918` | 0.0384% | 75.114% | **24.89%** |

Row 1 reproduces hg1's **97.65%** exactly, from a different tool — that is my positive control.
`m_safe = 2·delta_R` and `delta_R = 0.019590163230895963` is VERIFIED at
`reports/delta_R_noise_floor.json` (p95 of the uint8-induced margin perturbation over the
annulus, n96).

**Consequence, stated as a claim.** hv1's shipped seg objective spends **73.6-90.3%** of its
gradient mass on pixels that are already correct. Moving to hg1's own derived scale would put
**3x more** of it on flips at the terminal stage (26.4% → 75.1%) and **4.7x more** at the
softplus stage. `grep -n tau` on `train_semantic_quantized_resumable.py` returns **nothing** — τ
is not a flag, so this is not even a tracked-off lever. It is an unexposed constant, the
`[[m21]] CONSTANTS→LAWS` class.

**And the slot is aimed at the wrong vehicle.** hg1 §4.1 justifies an 8-hour Metal slot with
**hv1's** arithmetic (seg 0.029611 = 3.085x the gap; 32.41% recovery closes sub-0.15). hg1 §8
then seals a 3-arm fire order whose *"base argv [is] inherited from the sealed lv1 tr1 ticket"* —
the **tr1** vehicle. Per `[[L18]]`, a number from another vehicle is a hypothesis here, not a
justification.

**This was recall, not reasoning.** hg1 cites `sx1`, `rg1b`, `b2e`, `band_objective`, and
`train_semantic_quantized_resumable` a combined **0 times** (grep). `sx1` had already routed the
seg cure to `--band-objective-weight` on that exact trainer 9h35m earlier; `ns1`'s P1 correction
had already written *"the mp2-edited tensors live in the SemanticTokenRenderer (trainer
`src/tac/pr130_lift/train_semantic_quantized_resumable.py`), NOT the cl1 token model"* 19 hours
earlier. This is `[[charter_recall_validation_is_apparatus_not_volition_20260816]]` firing again
on the same day.

**Named next measurement.** Expose `τ` (or a margin offset) on
`train_semantic_quantized_resumable.py`, resolved LIVE through the registered
`margin_band_satisficing_threshold_v1` law off `reports/delta_R_noise_floor.json` — never a
literal — and run hg1's A/B **on hv1's semantic trainer**, not on tr1. Cost: a small code change
($0) plus **the slot hg1's seal already requests, re-aimed.** hg1's pre-registered falsifier
(realized seg recovery below 25% of the ladder) transfers unchanged.

**Its prerequisite is also unowned, and it is the cheaper gate.** b2e NEXT #2: *"does any lr /
step budget move the burn-2 base beyond the noise floor of this instrument?"* b2e MEASURED a
3,000-step window moving `ΔS_adv` by **+0.000336** with a **9-byte** weight-entropy change. If
nothing moves the base, the τ A/B is **untestable, not informative** — and that is a $0-to-cheap
question that must be answered before an 8-hour slot is spent either way.

---

## FINDING 2 — the rate closures split in two, and conflating them is a live error

Every rate closure landed on 2026-08-16 is one of two kinds. They license different actions.

**CEILING closures — perfect execution cannot reach the bar. These are finished.**

| row | ceiling (perfect execution, free) | % of the 14,413.4 B bar |
|---|---:|---:|
| `dc1` entropy coder (rc64 → a perfect coder) | ≤ **7.8 B** | **0.05%** |
| `hm1` additive correction table (table given away free) | ≤ **356.1 B** | **2.47%** |
| `cl2` in-network HPAC growth at the measured 1.15 marginal | — | **9.45%** |

`cl2` is the tightest of the three and deserves its caveat: its ceiling at bit saturation is
1.33-1.77x the bar, but reaching it needs a **sustained** marginal of 2.588-3.109 held to
saturation while convexity guarantees decay, against a measured near-point marginal of **1.15**.
That is a ceiling that clears on paper and closes on the only measured slope.

**DISTORTION closures — the bytes ARE there. A frame change is what refuses.**

| row | bytes available | % of bar | why refused |
|---|---:|---:|---|
| carrier (`ra1` rank-4) | **14,709 B** | **102.1%** | 828x-3,139x over break-even on pose (`ra2crr`, whole sphere, 292/292 descents) |
| token drop (`rc4` optimum) | 17,985 B; rate+seg **−3.2430e-3** | **33.8%** | pose +0.174319, **517x** over the headroom it buys |

**`ra2crr` already caught the conflation once, and it is the model for this section.** `ra3`
closed the carrier on *"perfect execution returns 6.3-12.8% of the gap"*. That is the
**one-dimension** rung's ceiling, not the family's. `ra1` measured rank-4 returning 14,709 B =
102.1% of the gap. **The carrier family's ceiling clears the bar; its closure rests entirely on
distortion, and must be cited that way.**

**So the archive holds the required 14,413 B twice over, and one wall holds both pools:** the
pose cost of changing a rendered frame. The wall is not that pose is large — pose is only
0.0082946 of S in total. The wall is that `d_pose = 6.88e-06` and the square root is steep
there: the pose marginal is **602.80 S per unit d_pose** (`rc4`, DERIVED, and it reproduces from
`5/√(10·6.88e-06)`). That single number is what refuses both pools.

---

## FINDING 3 — `rc4`'s one remaining door, priced at $0 from evidence that landed 35 minutes after it

`rc4` (20:15) left exactly one live row: *"can the in-compile frame-0 Schur compensation absorb
a per-pair `delta_d_pose` of 3.4e-3?"* PASS needs residual `Δd_pose ≤ 6.431e-6` — **99.807%**
cancellation. `ra3` landed at 20:50 with the measurement that prices it.

**DERIVED (this unit).** The best residual any MEASURED compensator on this carrier has reached
is `ra3`'s per-pair trust-regioned realized-acceptance row: `Δd_pose_abs = 2.62864e-4` (n600,
authority-tracking GT, exact through the shipped chain).

```
ra3 best measured residual        2.62864e-04
rc4 PASS bar                      6.431e-06
                                  = 40.87x above the bar
rc4 net if the compensator lands in that class:
    -3.2430e-3 (rate+seg)  +  0.043642 (pose)  =  +0.040399 S   -> REFUSE by 12.5x
```

**Two supports and one caveat, all stated.**

- `jc1` MEASURED that a linear model used as a **designer** over-states its own control authority
  by **134x-1,065x** at step 0.240-0.247 of coefficient RMS. I estimate `rc4`'s required
  compensating step at **~0.17** coeff-RMS (DERIVED: the drop needs a pose6 move of ‖p‖≈0.14
  against `ra3`'s incumbent ‖p‖≈0.112 at step 0.1351). Same regime.
- `ra3` MEASURED that no global trust radius exists — the accepted-slot histogram spreads across
  four decades — so a single-radius solve is not the missing trick either.
- **CAVEAT, load-bearing.** `ra3` corrected its own rank-cut damage using **11** coefficients
  under a subspace constraint. `rc4`'s compensator would have **all 12** free with the basis
  intact — strictly **more** authority, and against an externally-caused perturbation rather than
  its own. So **40.87x is INDICATIVE, not a bound.** I did not measure it.

**Still worth firing, at a re-priced expectation.** `jc1` left an exact n600 `d_pose` instrument
that runs in **62 s** with no bulk artifact (`ddm_jc1_carrier_pose_jacobian.py --eval-coeff`),
validated by a no-op control at 2.6e-5 relative. So the decisive measurement is minutes of local
CPU, not a dispatch. Fire it — but expect a refusal, and do not build the drop encoder first.

### A $0 extension that closes a follow-on nobody should propose

`rc4` measured pose at **one** threshold. I extended its own ladder under the DERIVED assumption
`Δd_pose ∝ token flips`:

| bytes saved | token flips | rate+seg | pose (derived) | net | pose ÷ |rate+seg| |
|---:|---:|---:|---:|---:|---:|
| 81,321 | 123,772 | −2.2670e-02 | +0.55679 | +0.53412 | **24.6x** |
| 42,652 | 43,629 | −9.9080e-03 | +0.32727 | +0.31736 | 33.0x |
| 18,869 | 13,711 | −2.6850e-03 | +0.17995 | +0.17726 | 67.0x |
| 11,901 | 7,791 | −1.3200e-03 | +0.13371 | +0.13239 | 101.3x |
| 1,538 | 759 | −5.8790e-05 | +0.03672 | +0.03666 | **624.6x** |

**The ratio is monotone INCREASING as the threshold softens** — 24.6x at the most aggressive
rung, 624.6x at the gentlest. Pose enters as `√`, rate enters linearly, so a smaller drop is a
*worse* trade, not a safer one. **No threshold is closer to break-even than the one `rc4`
tested.** "Sweep more thresholds" is dead; the compensator really is the only door. This
STRENGTHENS `rc4` rather than reopening it.

---

## The per-row table — (a) scope · (b) denominator · (c) ceiling · (d) hidden win · (e) staleness

Rows 1-6 are the assignment. Rows 7-21 are my sweep additions. Every number sourced to the memo
that measured it unless marked *(mine)*.

| # | row | (a) scope I can defend | (b) denominator | (c) ceiling in S | (d) hidden win | (e) stale? |
|---|---|---|---|---|---|---|
| 1 | **dc1** coder axis CLOSED ⚠ **NARROWED 2026-08-18 (`ddm_hd1`, from `na9` F5 #4): read this row as coder-SWAP at FIXED probabilities. It does NOT close the probability-MODEL axis — `fx1` measured −560.07 B there, byte-closed at 180,601 B, 72× the ceiling below. This row's "FAMILY" promoted dc1's ceiling one level too wide and told arms not to look at a live win for a day.** | **FAMILY, and it earns the word.** Not one race — a bound: the free-table ORACLE at 21 taps bottoms at 144,167 B vs a shipped 112,110 B stream | Real. rc64 measured at **1.00000** of the model cross-entropy from a byte-identical replay; `rc4` reproduced it independently to 0.02 B | **≤7.8 B = 5.19e-6 S = 0.05% of gap** | The **inverse** row: the learned prior returns **+3.81 B per counted byte**. dc1 called this "the only live cell"; `hm1` closed the table branch and `cl2` closed the network branch the same night | dc1's claim that the 831.5 s anchor is "UNVERIFIABLE" is **WRONG** — `hm1` located the primary receipt. Corrected at source by hm1 |
| 2 | **b2e** `REGIME_THESIS_INSTANCE_REFUTED` | **FORMULATION**, exactly as b2e labels it. And b2e's own caution is the honest read: the window moved the base by 1.4e-7 seg / 7.9e-7 pose, so *"this window did not train, so it did not train for editability"* | Real and pre-registered (50x collapse). Measured 0.75-1.06x — 4 orders of margin over reporting precision. n600, **not a prefix** | Even a perfect window must collapse pose damage **~75x** before these edits pay (b2e's own §4) | **YES — b2e NEXT #2.** "Does any lr/step budget move this trainer's base?" It is the gate on the entire seg-renderer route (Finding 1) and it is **unowned** | No |
| 3 | **#1058** campaign close | QAT-leg: **INSTANCE** (correct). mp2: **FAMILY** on post-hoc semantic weight edits without joint re-descent — upheld by `ns1` §A and by `b2e` | Real: bar net < −3.5e-6; pose leg exceeds it ~10,000x | Smallest probe (−25 B) pays +0.0362 S. **1.4e-3 S per byte removed vs 6.66e-7 S/B of rate value — 3 orders over.** No magnitude clears | None. §3 route #2 (carrier rank, GATED) is now closed 5 ways | Route §3 item 2 is **stale**: it says carrier rank is "GATED, not fired". `ra2c`/`jc1`/`ra2`/`ra3`/`ra2crr` fired all five treatments the next day |
| 4 | **rung-2 truncation** REFUTED 189-278x (`ra2c`) | **INSTANCE→FAMILY, correctly escalated.** ra2c §8.1 replaced an inherited round-number ladder with the complete computed table: miss monotone in r, never below 32.2x, flat spectrum | Real. Priced on both the advisory (1.5731x) and T4 (4.7394x) bars — closes on the **loosest**, the stronger statement | ra2c's own §8.4 named a **1,854 B per free dimension = 12.9% of gap** upside if `K ≥ 1` | **Fired and CLOSED the same day.** `jc1` measured **K = 0** at every tolerance 1e-16→1e-1; cond(J_stack)=12.02, column spread 2.16x. No free direction, no cheap one | **`gestalt` (16:27) is STALE**: its live ladder lists rung 2′ as *"OPEN — the named successor"*. `jc1` closed it at 18:20; `ra3` re-closed it at 20:50. Do not resume off that table |
| 5 | **av3 F1** EMA-lag REFUTED | **INSTANCE, and av3 says so explicitly** ("the A2 and C0 arms, 600 steps, this trainer"). A model row | Real and asymmetric: the init's surviving weight is 3.29e-19 at t=600 — the confound is impossible, not merely absent | N/A — apparatus, not a supplier | The **replacement** is the value: `peak_flips ∝ ‖dw‖^0.458` over 250x displacement, and *"the control is not null"* (C0 at lr 2e-7 moves the judged metric +17.4%). It re-reads the recovery as **rotation within a fixed-radius shell**, not annealing | No. Cleanest row in the window |
| 6 | **fb1** bank union cannot fire | **CLOSED**, and fb1 labels its own key step **DERIVED not MEASURED** (§D.2). Correct | Real, and fb1 corrected the pool first: 2 of 4 listed members are **already inside the pointer**. True pool −5.5818e-6 = **55.8%** of its own 1e-5 bar; `gx1` derived it independently | 1.79x short **at perfect composability and perfect additivity** — the honest ceiling. Cannot be satisfied by waiting | Not a byte win — the **invariant** is: `sub-0.15 ⟺ archive ≤ 168,345.5977 B`, stale-proof under pure-rate moves. Now registered as `sub015_pure_rate_archive_byte_bar_20260816` | **This row IS the staleness finding.** 13 lines / 11 files / 4 arms carry `< 186,269 B`; at that bar a candidate passes while scoring +0.002337165 **worse** than what we ship = 667.8x the report band |
| 7 | **rc4** rung 4 REFUSED 517x | **FORMULATION** (uncompensated drop), correctly labelled | Real, and rc4 defends it: paired differential, GT-lineage swap moves it 1.7%, determinism repeat bit-identical | rate+seg **−3.2430e-3 = 33.8% of gap** — the largest live pool in the window | **YES — row 1, the compensator.** Priced by me at 40.87x short *(mine, INDICATIVE)*. Fire it on `jc1`'s 62 s instrument, expect refusal | Pose leg is n=48; its own base came out **0.49x** the population value, so ~2x sampling uncertainty. Verdict survives with 258x of margin; the **fire condition** tightens to ~99.90% |
| 8 | **hm1** correction table CLOSED | **FAMILY** on additive post-hoc correction tables and hand-designed conditioning. Earned: 9 realized rungs + 4 free-table oracles | Real. Break-even is slope < −1; 8 richer rungs all in [−0.469, −0.000] | **356.1 B = 2.371e-4 S = 2.47% of bar**, table free | hm1's **oracle rows refuted hm1's own design** and it reported that. The surviving product: the model's 5-vector output is **not summarizable** by hand-designed features (≥ +2,097 B even free) | No. It corrected dc1 |
| 9 | **cl2** rate_lambda REFUSED | **INSTANCE** on the sealed order (Gate 0 fails: 4 of 7 hashes drifted; VertigoDataTier at 893 MiB) + a ceiling argument on the family | **cl2 fixed the denominator** — the 3.810 is an AVERAGE over `[0, M0]` and `tokens(model)` is convex, so it is an **upper bound on the marginal**, never an estimate. Measured near-point marginal **1.15** | 9.45% of bar at the measured marginal; 1.33-1.77x only at an unattainable sustained 2.588-3.109 | None left on this branch. **This closes hm1's last open branch**, 2h21m after hm1 named it | No — it is among the freshest rows in the window |
| 10 | **hg1** ring-0 hinge | **INSTANCE on `tools/train_ddm_cl1_hpac_capacity_mps.py`** — *not* on "hv1's trainer". See Finding 1 | The 1.519x prize re-prices soundly off rn1's proxy. The 97.65% waste figure **reproduces exactly** *(mine)* — but it prices the **tr1 default**, not hv1's shipped τ | Seg = **3.085x the gap**; 32.41% recovery closes sub-0.15. Largest ceiling on the board | **YES, and it is the lead finding.** The term already runs on hv1 with SegNet in loop; the defect is a hardcoded τ (waste 73.6-90.3%) not a missing lever; the sealed slot is aimed at tr1 | Not stale — **mis-routed**, which is worse. It also cites zero of sx1/rg1b/b2e |
| 11 | **rt1** free-band channel NON-SUPPLIER | **INSTANCE**, per-leg, as rt1 labels it | rt1 priced on an **AVERAGE** η. That is the weak link | +0.0025 S after rt1 corrected its own pose-aggregation error | **ALREADY FIRING.** `sr1` (landed 23:15) re-prices **per cell**: +0.00269 S → **−0.000595 S = 6.2% of gap**, plus a 0-byte tridiagonal deconvolution of the exact `[0.101470, 0.797060, 0.101470]` operator. *sr1's claim, not verified by me* | rt1's "R supplies exactly zero" is explained mechanically by sr1: A is **DC-preserving**, so flat paint is an eigenvector. **The zero does not transfer to the render** |
| 12 | **ps1u / ps1u_r2** REFUSED +1.686e-2 | **INSTANCE** on one candidate archive, correctly | Real — a paid contest-CUDA row, repeat noise 0.0 | Pose→0 buys 0.0082946 = **86.4% of gap** and **structurally cannot close it alone** (`ra3` §8: leaves +0.0013027) | The mechanism finding: *"the CAP was not the constraint, the TARGET was"* — this closes `ns1` P3 and `#850` | `ns1` P3 (01:09) is **stale**: it lists the uncapped pose solve as a named unfinished measurement. ps1u ran it by 09:30 and it made pose **8.93x worse** |
| 13 | **wd2 / wd3** distillation | wd2 **INSTANCE** (prune+refit @60ep). wd3 **FAMILY @65ep** with a reactivation ladder — properly parked, not killed | Real: 8.2x over the seg bar, decelerating; n120 seeded stratified **non-prefix** | wd2's measured 17,372 B clears **both** the stale and live rate bars — it failed on **distortion** (Δd_seg 7.0059x), never on rate | The law: optimizer-state pose-carry (warm 3x) ⇒ **judge fresh students at the seg asymptote**, and a negative needs n120, never n60 | `ddm_wd2_…:95` carries the stale **15,157 B** rung. Stale in the SAFE direction (too tight). `fb1` verified no verdict flipped |
| 14 | **td1** token drop as a rate lever CLOSED | **INSTANCE→confirmed by rc4** on the live vehicle | td1's own headline ran **1.6% low** (0.028155 vs rt1's measured 0.028604) — corrected at source by `rt1`/`sx1` | Names a supplier **2.9x the entire gap**: ~95% of the seg term is render→SegNet round-trip loss, not label error | The negative that pays: **the qs3 57.1% beneficial prior is wrong by 158x**. A borrowed prior, measured dead — `[[m21]]` | Superseded by `rc4`'s full three-leg pricing. Cite rc4 |
| 15 | **mz1 / mz2** lossless model-section race | **INSTANCE** on the exact e480b RX2 object, correctly | Real: complete framed sections, all parse-back verified | **Exact savings 0 B.** The shipped split-Brotli q10/q11/q11 won its own race | mz1 dissolved a claimed **52,566 B gap** into an attribution error. `dc1` reproduced the same shape on hv1 (semantic + carrier recodes **byte-identical** to shipped) | No |
| 16 | **pz5** `STAGE_0_REFUSED` | **INSTANCE**, and it is a model of the genre — refused on **four** independent grounds and **did not build** | The charter's −20,524 B was a wrong-object number | N/A — the lever does not exist | The mechanism: *"the section the arithmetic proposed to delete is the renderer for frame_0, and the packet proposed to replace it stores PoseNet's own six output scalars — which are not an image"* | This row **is** the staleness cure: a charter premise re-derived before a build |
| 17 | **r6j** VERDICT WITHDRAWN | Withdrawn correctly. The null control cost **0.7 s at $0** and was free the whole time | **The denominator was missing** — no cross-arm control. Textbook `[[the_denominator_and_the_falsifier_can_both_be_vacuous]]` | N/A | The generalizable rule: a cosine between two arms that **share an objective** is the control every step-similarity claim owes | No |
| 18 | **sx1** seg cure ladder | **Per-rung**, no new family verdict — correct restraint | Real: the byte-carrying rung needs η **1.0069**, i.e. impossible at perfect realization AND perfect coding | Every seg lever is an alternative on ONE shared support ⇒ **the ladder is a MAX, not a SUM** | **The cheapest live rung is one CLI flag** (`--band-objective-weight`, default 0.0) on the semantic trainer. It is Finding 1's sibling and it is **also unfired** | No — sx1 is the row `hg1` should have read |
| 19 | **na8 / ns1** the two prior audits | na8 self-corrected its own headline (the 148 expired deferrals are a **drained leak**, all pre-2026-07-15). ns1 covers the pre-08-16 window | na8's 148 is one ledger only; ns1's P-list predates the whole 08-16 arc | N/A | na8's live product is §2: **fit-vs-solve generalization asymmetry** and the **support-set ceiling** | **ns1 is stale in 2 of 5 rows and superseded in a third.** P2 ("unmeasured at ANY drop level") → `rc4` measured it, REFUSED. P3 ("the un-run measurement") → `ps1u` ran it, REFUSED. P1's first window ran and was REFUTED at **formulation** scope by `b2e` — the thesis stands, its "est. reach multi-KB" framing does not. P4/P5 stand |
| 20 | **ra2crr** (landed 23:14) | Corrects `ra3` at family level | **The correction that matters here** | `ra3`'s 6.3-12.8% is the ONE-DIMENSION ceiling. `ra1` measured rank-4 at **14,709 B = 102.1% of gap** | The carrier closure is a **DISTORTION** closure, not a ceiling closure. Finding 2 is built on this | Names 5 distinct "carrier" byte objects (22,161 / 22,219 / 22,242 / 22,307 / 22,278) — the 22,032 B figure is stale by 123 B |
| 21 | **sr1** (landed 23:15) | INSTANCE, hv1 | — | — | The live positive. See row 11 | — |

---

## Genuinely closed — worth no further attention

Do not spend a slot, an arm, or a charter on any of these. Each is closed on a **ceiling** or on a
**bound**, not on a single failed attempt.

1. **The entropy CODER on every hv1 section** (`dc1`, `rc4`). Ceiling ≤7.8 B. Point new arms at
   `tools/audit_archive_coder_axis.py` rather than letting them inherit the closure.
   ⚠ **NARROWED 2026-08-18 (`ddm_hd1`).** This entry closes **coder-SWAP at fixed probabilities**,
   and at that scope it is correct and unrefuted. It does **NOT** close the **probability model**
   that feeds the coder: `ddm_fx1` (08-17) landed a fixed-point logistic log-odds mixer at
   **−560.07 B** on the n600 token field, byte-closed **180,601 B** (sha `65c75d7f…`, parse-back
   PASSED), **ΔS −3.72881e-4 = 72×** this ceiling. Per the legend below, a different probability
   law **is** a new mechanism, so this entry never barred `fx1` — but as written it read as though
   it did, which is how a live −3.73e-4 sat unlooked-at for a day. Do not inherit it any wider
   than "coder-swap".
2. **Additive correction tables and hand-designed conditioning** (`hm1`). Ceiling 2.47% of bar
   with the table free.
3. **Lossless re-coding of any model section** (`mz1`, `mz2`, `dc1`). Exact savings 0 B; the
   semantic and carrier sections re-code **byte-identically** to what ships.
4. **The carrier as a rate lever, all five treatments** (α=0, rank truncation, keep-set re-fit,
   subspace, subspace + trust-regioned re-fit). `K = 0` MEASURED; cheapest direction anywhere on
   the sphere costs 828x-3,139x break-even. Closed on **distortion**, per `ra2crr`.
5. **The banked micro-edit union** (`fb1`). 55.8% of its own bar at perfect composability, and
   two of its four listed members are already inside the pointer.
6. **Uncompensated token drop at any threshold** (`rc4` + my §Finding 3 extension). The trade gets
   monotonically **worse** as the threshold softens.
7. **Post-hoc semantic weight edits without joint re-descent** (`#1058`, `ns1` §A, `b2e`). Three
   orders over, located to `blocks_1` FiLM rows, with a $0 screen (`Δd_pose ≤ 5.1e-9·ΔB`).
8. **Sweeping more drop thresholds, more ranks, more radii, or a longer b2e window at the same
   settings.** Each is explicitly refused by the memo that measured it.

---

## Hides value — with a named next measurement and its cost

Ranked by (probability the measurement flips something) x (S at stake).

| # | row | the ONE measurement | cost | what it decides |
|---|---|---|---|---|
| 1 | **hg1 mis-routing** (Finding 1) | Expose `τ` / margin offset on `train_semantic_quantized_resumable.py`, resolved from `delta_R` through the registered law, and run hg1's sealed A/B **on hv1** instead of tr1 | $0 code + **the slot hg1 already asked for**, re-aimed | Whether the largest ceiling on the board (seg = 3.085x the gap) is reachable on the vehicle that carries the goal |
| 2 | **b2e NEXT #2** (prerequisite to #1) | Does any lr/step budget move the semantic trainer's base beyond this instrument's noise floor? | cheap; short local runs | Whether **any** seg-renderer training row is testable at all. If nothing moves, #1 is untestable, not refuted |
| 3 | **rc4 row 1** — the frame-0 Schur compensator | Per-pair in-compile 12-coefficient re-solve against the 6 PoseNet equations at `p_max ≥ 0.9921875`; measure residual `Δd_pose` on `jc1`'s exact instrument | **62 s per candidate**, local, $0 | The last frame_1 rate lever (33.8% of gap) — **and** the gate every future frame_1 lever inherits. Priced by me at **40.87x short** *(INDICATIVE)*; fire it to convert an indication into a measurement |
| 4 | **sx1's `--band-objective-weight` α-ladder** | One CLI flag on a receipted tool, already built with an inert-guard, default 0.0 | one flag | The sibling of #1 on the same trainer. `rg1b` records it *"ran once and did not descend"* — one run is not a ladder |
| 5 | **rt1 → sr1 per-cell re-pricing** | `sr1` landed at 23:15 and needs an adversarial read — nothing has reviewed it | $0 read | Whether a rt1-class negative priced on an average flips sign per cell (**claimed** −0.000595 S = 6.2% of gap) and whether a 0-byte tridiagonal deconvolution is real |
| 6 | **`cl2` Gate 0 hygiene** | 4 of 7 sealed hashes drifted; VertigoDataTier at 893 MiB free | $0 | Not score — but the seal cannot fire as written, and that is a blocker with an owner-shaped hole |

---

## Where I attacked my own conclusion

1. **My lead finding could be a distinction without a difference.** hv1's shipped `expected_flip`
   at τ=0.05 already puts 26.4% of gradient mass on flips; hg1's derived target puts 75.1% there.
   That is 2.85x, not 40x. **Gradient-mass concentration is not score**, and I measured no ΔS.
   The claim I defend is narrow: *the lever is routable on hv1's own vehicle, and the seal is
   aimed elsewhere.* The magnitude of the prize is hg1's ladder, not mine.
2. **My margin field is the DEPLOYED render's, and the trainer's target is the token field, not
   GT.** hg1 measured against GT. Per hg1 §6 the two agree to 8.48e-6 against a 2.9611e-4 axis, so
   my gradient shares sit within ~3% of the trainer's own quantity. n=96 seeded, not n600
   (`[[m96]]`: a subset may refute a bar, not license a live verdict).
3. **My `rc4` row-1 price transfers a residual across two different problems.** `ra3` had 11
   coefficients under a subspace constraint correcting its own damage; `rc4`'s compensator has 12
   free coefficients correcting an external perturbation — strictly more authority. I label it
   INDICATIVE and I recommend firing the measurement anyway.
4. **My ladder extension in Finding 3 assumes `Δd_pose ∝ token flips`.** `rc4` measured pose at
   one point only. The assumption is a linear extrapolation of a quantity `rc4` itself calls
   first-order. The *sign* of the monotonicity is safe (pose is `√`, rate is linear); the
   magnitudes are DERIVED.
5. **I did not verify `fb1`'s 296/27/13 sweep counts, `na8`'s 148-row census, or any receipt on
   `/Volumes/APDataStore` other than hg1's margin field.** Those are the memos' claims, cited as
   such.
6. **"Everything else in the window is correctly scoped" is bounded by my glob.** I read 75 files
   under `.omx/research/*2026081[56]*.md`. I did not sweep `charters/`, `arm_final_messages/`, the
   task ledger, or code. A mis-scoped negative living in any of those is outside what I checked —
   and `fb1` §D.3 already proved that a sweep proves nothing about phrasings it cannot see.
7. **Two rows landed while I audited** (`sr1` 23:15, `ra2crr` 23:14). A window that is still
   moving is a window I can only claim to have read at a point in time.

---

## The pattern, named

Three of the 21 rows share one shape, and it is not carelessness:

**A negative that is correct about the object it measured, and wrong about the object it names.**

- `hg1` measured the **token** trainer and named **"hv1's trainer"**.
- `ra3` measured the **one-dimension** rung's ceiling and named the **family's** (caught by
  `ra2crr`).
- `rt1` measured an **average** η and named the **channel** (re-priced per cell by `sr1`).

In all three the measurement is sound and the label over-reaches by exactly one level. That is
the `[[the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816]]` law
firing on the verdict surface rather than on the number surface. **The cheap detector is one
question at verdict time: *which object did I actually measure, and is that the object my
sentence names?***

---

## NEXT_IF_RESUMED

- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. Re-aim hg1's §8 sealed 3-arm order from the
  tr1 vehicle to `train_semantic_quantized_resumable.py` on hv1, after landing a `τ` / margin-offset
  flag resolved through `margin_band_satisficing_threshold_v1`. Fire trigger: **before the sealed
  hg1 slot is spent.** $0 code; the slot is already requested.
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner **MAIN**. b2e NEXT #2, the trainability gate. Fires
  **first**, before the τ A/B. Cheap local.
- **`QUEUED-WITH-A-FIRE-ORDER`** — owner: the qs5/pose successor. `rc4` row 1 on `jc1`'s 62 s
  exact instrument, with my 40.87x price recorded as the prior. $0 local.
- **`CLOSED`** — the eight rows listed under "Genuinely closed". No arm should re-open one without
  a NEW mechanism, and a new radius, rank, threshold, or window is not a new mechanism.
- **`OWED, unowned`** — `ra2crr`'s family-level correction to `ra3`'s closure ground (ceiling vs
  distortion) is load-bearing for Finding 2. It landed at 23:14 but nothing downstream cites it
  yet; `ra3` §5 ground 2 and `gestalt` §5 still read the other way.

**Own-vehicle frontier: hv1 ep0634, S 0.15959729295498598 @ 182,759 B `[contest-CUDA T4, n600]` —
UNMOVED by this unit.** This arm ran no scorer, launched nothing, and spent $0. It did not lower
the score. It found one mis-scoped negative that is currently mis-routing an 8-hour Metal slot off
the vehicle that carries the goal, priced one open door at $0, and closed a follow-on class that
nobody should propose again.
