---
title: "The cold optimizer's damage is born in sixteen updates and the EMA reports it two thousand later: 62.011% of the shipped object's terminal seg error sits in sites that were wrong before the first update and never moved (Lane enriched 51.50x, floor 12.75x the sub-0.12 corner), the run creates 2.21x as much error as it repairs, and prediction 1 HOLDS on the shadow while it would FAIL on the live forward — but prediction 2 is FALSIFIED in 20 of 20 comparisons: carried AdamW moments do not damp the same trajectory, they walk a different one that breaks a 37.7-64.8% disjoint site set at every step and ends +0.0187 S_hat ABOVE the cold control"
arm: ddm_md1
charter: .omx/research/charters/ddm_md1_micro_to_macro_dynamics_20260904.md
charter_commit: a5b58f1fa
preregistration: .omx/research/ddm_md1_prereg_20260904.md
preregistration_commit: 0af527a80
utc: 2026-09-04T13:30:00Z
verdict_scope: "[macOS-CPU advisory . exact argmax reconstructed from the retained 16-step checkpoints . frozen CPU-torch SegNet+PoseNet . QBF1-born vehicle . n32 sealed selection . seed 20260902 cold control + ng1 warm cell . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_md1 — micro → macro: how the born field's error is born, moves and dies

## The pre-registered predictions and falsifiers, READ OUT BEFORE THE NUMBERS

Committed at `0af527a80` (`.omx/research/ddm_md1_prereg_20260904.md`) while the sweep launched at
2026-09-04T12:09:13Z was still running.

> **Prediction 1** (charter, from gc1's capacity closure): `PERSISTENT` ≥ 60% of the TERMINAL
> d_seg, and the persistent set is Lane- and edge-concentrated.
> **FALSIFIER: `PERSISTENT` < 40%** — then optimization levers alone could plausibly reach the
> target and the capacity closure is re-graded.

> **Prediction 2**: the warm cell's excursion sites are a SUBSET of the cold cell's (moments damp,
> they do not redirect).
> **FALSIFIER: > 30% of the warm excursion sites are absent from the cold set.**

The pre-registration also recorded, before the answer, that prediction 1's premise is **weakly
supported**: gc1 is a static four-point capacity sweep on the GF1 analytic generator scored by
categorical Hamming mismatch — not d_seg, not this trained object, not a trajectory. Its transferable
content is that Lane's error is capacity-resistant (Lane 1.16× vs Road 2.59×, Undrivable 3.03× as
the packet grows 1.599×). The 60% threshold was a PRIOR, not a transfer.

## Verdicts

| prediction | reading | measured | verdict |
|---|---|---:|---|
| 1 — `PERSISTENT` ≥ 60% of terminal d_seg | **EMA shadow** (the object that ships), DALI | **62.011%** | **HOLDS**; falsifier does NOT fire |
| 1 — same | EMA shadow, PyAV (the vehicle's own target) | 61.670% | HOLDS |
| 1 — same | **live weights**, DALI | **35.779%** | would **FIRE** the falsifier |
| 1 — Lane- and edge-concentrated | shadow, terminal | **64.79%** of persistent terminal-wrong sites touch Lane; GT=Lane enriched **51.50×** | **HOLDS** |
| 2 — warm excursion ⊂ cold (falsifier: >30% of warm born sites absent from cold) | own-peak, shadow | **45.40%** absent | **FALSIFIER FIRES** |
| 2 — same | own-peak, live | **38.39%** absent | **FALSIFIER FIRES** |
| 2 — same, SAME-STEP control at 10 steps × 2 forwards | 20 of 20 comparisons | **37.69%–64.80%** absent | **FALSIFIER FIRES everywhere** |

**The reading travels with the number.** The archive is re-encoded from `ema.shadow`
(`experiments/ddm_qbr1_born_fairform_burn_prep.py:629-632`), so the shadow is the object a
submission would contain and is the authoritative reading; the charter's own quoted terminal value
(0.0027589) is the shadow milestone. The live forward gives 35.779% because the live weights are
noisy checkpoint-to-checkpoint and `CHURN` absorbs 62.347% of their terminal error. Both are
reported; neither is chosen after the fact.

## The finding, first

**The cold optimizer's damage is complete after sixteen updates. The milestone record shows it at
step 2,000 because the EMA shadow is a 1,086-update low-pass filter, not because the field got
worse for 2,000 steps.** In the first 16 updates the live field's exact `d_seg_hat` triples —
0.0025555929 → **0.0060334524** (2.3612×) — Lane is painted at **1.29558×** and Movable at
**1.11117×** their GT area, and **24,336** sites are newly wrong, **75.6%** of them painted as a
rare class over **70.4%** GT-Road ground. The live field then oscillates and **never returns to the
initialisation's quality** (minimum 0.0027205 at step 3,072, still +6.5% above the start). The
shadow, whose weight on any single update at step 5,000 is `(1−d)·d^(5000−s)`, spends 1,344 updates
converging onto that worse field, first meets it at step **1,344**, and turns over at step **2,304**.

**And the run creates 2.21× as much error as it repairs.** In exact integer HT numerators, over the
5,000 updates the shadow repaired **24,435** units of error that existed before the first update
(PERSISTENT −9,075, HEALED −15,360) and created **54,045** new units (CHURN +19,380,
NEW_PERSISTENT +34,665), for a net **+29,610** on a start of 301,470 — **+9.82%**.

**The persistent core barely moves and is Lane.** Of the 214,380 numerator units the field carried
into update 1, 5,000 updates removed **9,075 = 4.233%**. 30.61% of the persistent sites have GT
class Lane against Lane's 0.594% of the frame — a **51.50×** enrichment (Movable: 1.62×). And the
floor those sites impose, **0.0017403920**, is **12.75×** the sub-0.12 accuracy corner: even a
lever that removed every optimizer-reachable site would leave the accuracy half an order of
magnitude short.

## 1. What was measured, and the reference form

The exact per-site SegNet argmax at **71 checkpoints** (every 16 steps 0–512, every 64 to 2,048,
every 256 to 5,000, plus the checkpointed milestone steps 2,000 and 4,000 and the terminal state),
for **both** the live weights and the EMA shadow, on all 32 pairs of the sealed no2 selection,
through the trainer's own render → bicubic camera → uint8-STE → frozen CPU scorer path. 141 forwards
for the cold cell; 4,262.2 s of CPU at 4 threads, peak RSS 15.31 GiB.

**Verified at source — every premise this arm adds:**

| premise | where | label |
|---|---|---|
| the milestone forward runs inside `qbt.ema_scope`, so every retained milestone is the **EMA SHADOW**; the training objective is the **LIVE** forward | `ddm_qbr1_born_fairform_burn_prep.py:600` → `:612` | MEASURED (source) |
| the shipped archive is re-encoded FROM the shadow, so the shadow is the authoritative reading | `ddm_qbr1_born_fairform_burn_prep.py:629-632` (`state=ema.shadow`) | MEASURED (source) |
| `d_seg_hat = Σ_p w_p·d_seg_p / 600`, integer HT weights `(15.0,)*24 + (30.0,)*8` | `ddm_qbr1_born_fairform_burn_prep.py:447`; `ddm_qbt1_qbflow_trainer.py:112` | MEASURED (source) |
| one AdamW group over `model.parameters()`, constant lr, no scheduler | `ddm_qbr1_born_fairform_burn_prep.py:681` | MEASURED (source) |
| checkpoints are written tmp+fsync+`os.replace`, so a `periodic_*.pt` under its final name is COMPLETE — reading the LIVE warm cell is safe by construction | `ddm_qbt1_qbflow_trainer.py:272-290` (`atomic_bytes` ← `atomic_torch`) | MEASURED (source) |
| seven parameter roles | `ddm_qbt1_qbflow_trainer.py:1946` | MEASURED (source) |
| the training target is PyAV `gt_n600.npz`; DALI `gt_cache_dali.pt` is the authority | `ddm_qbt1_qbflow_trainer.py:123`, `:2067`; `ddm_ar1_aa_render_price.py:102` | MEASURED (source) |
| `δ_R` n600 = 0.021881818771362305, law-resolved from sd1, never retyped | `experiments/ddm_sd1_surrogate_exact_map.py::DELTA_R_N600` | TRANSFERRED (dr1) |
| the sub-0.12 accuracy corner `d_seg = 1.3646784205e-4` | `.omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md` | TRANSFERRED (DERIVED there, n600, at the falsifier pose, on the bound 106,626 B archive) |
| EMA decay 0.9990793899844618, `warmup=False`, `d^5000 = 0.010000000000000278` | cold `RESULT.json` `ema_law_provenance`; `src/tac/training.py:504` | MEASURED |
| the warm cell starts from the SAME state as the cold control | both step-0 milestones report `d_seg_hat` 0.002518335978190104 and `S_hat` 0.39876797285867277; `pair_0004.npz` `segnet_argmax_u8` is bit-identical, 0 differing sites | MEASURED |

The **five error classes plus `ALWAYS_CORRECT` PARTITION** every site, under a falling rule (CHURN
first, because it is a flip-COUNT rule that would otherwise be absorbed by an endpoint rule).
`HEALED` — wrong at step 0, not persistently wrong, ≤4 flips — is the fifth class a partition
requires and that the charter's four do not name; it is reported explicitly, and its predicate does
NOT require ending correct (519 of its 1,352 sites are still wrong at the terminal).

## 2. Apparatus honesty: the CPU reconstruction against the retained MPS argmax

The burn ran on Metal; this reconstruction runs on CPU. Measured at every milestone that has BOTH a
16-step checkpoint and a retained argmax (1,000 and 3,000 are not multiples of 16, so no weight
state exists for them):

| milestone | sites compared | CPU-vs-retained-MPS differing sites | site fraction | `d_seg_hat` CPU (PyAV) | `d_seg_hat` retained MPS | relative gap | recomputed − recorded |
|---:|---:|---:|---:|---|---|---:|---:|
| 0 | 6,291,456 | **51** | 8.106e-06 | 0.002519353230794271 | 0.002518335978190104 | **+0.0404%** | **0.000e+00** |
| 5,000 | 6,291,456 | **53** | 8.424e-06 | 0.002761077880859375 | 0.002758916219075521 | **+0.0784%** | **0.000e+00** |

Two readings. First, the HT recomputation from the retained argmax reproduces the sealed
`MILESTONE.json` `d_seg_hat` **exactly** — the estimator is reproduced, not approximated. Second,
the CPU-vs-MPS argmax residual is ~8e-6 of sites at both ends and does not grow across the run; it
is 0.31% of the wrong-site population. **Every class table below is computed entirely inside the CPU
series, so the trajectory classes carry no MPS/CPU contamination**; only comparisons to the recorded
milestone carry that residual, and they carry it explicitly.

## 3. THE MACRO BRIDGE — the calibration gate is an integer identity

`d_seg_hat(t) = W(t) / (600·384·512)` with `W(t) = Σ_p w_p·n_wrong(p,t)`, integer weights and
integer counts. The class numerators therefore sum to the total exactly; **`max_t |Σ_classes − total|
= 0` in integers at every one of the 71 checkpoints, on both forwards and both GT lineages.** No
float tolerance is involved.

Cold control, EMA shadow, DALI authority, denominator 117,964,800:

| step | total W(t) | PERSISTENT | NEW_PERSISTENT | CHURN | TRANSIENT_BORN | HEALED | `d_seg_hat` |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 301,470 | 214,380 | 0 | 62,085 | 0 | 25,005 | 0.0025555929 |
| 2,000 | 383,865 | 213,090 | 20,505 | 108,645 | 32,235 | 9,390 | 0.0032540639 |
| **2,304 (peak)** | **388,230** | 212,520 | 20,880 | 111,720 | 33,570 | 9,540 | **0.0032910665** |
| 4,000 | 350,625 | 208,680 | 22,770 | 92,730 | 17,670 | 8,775 | 0.0029722850 |
| **5,000** | **331,080** | **205,305** | **34,665** | **81,465** | **0** | **9,645** | **0.0028065999** |

**Read the columns, not the total.** `PERSISTENT` falls 214,380 → 205,305: 5,000 updates removed
**4.233%** of the error the field already had. Everything else in the excursion is manufactured and
partly returned: `CHURN` +49,635 to the peak then −30,255, `TRANSIENT_BORN` 0 → 33,570 → 0 by
construction, and `NEW_PERSISTENT` climbs monotonically to **34,665** — error the run created and
kept. The net +29,610 (+9.82%) decomposes as **repaired 24,435 vs created 54,045 = 2.2118×**.

### Terminal class table (shadow, DALI)

| class | sites | site fraction | terminal wrong sites | terminal d_seg contribution | share of terminal error |
|---|---:|---:|---:|---|---:|
| ALWAYS_CORRECT | 6,258,119 | 0.994701 | 0 | 0 | 0.000% |
| **PERSISTENT** | 11,842 | 0.001882 | 11,344 | 0.0017403920 | **62.0107%** |
| CHURN | 10,924 | 0.001736 | 4,323 | 0.0006905874 | 24.6058% |
| NEW_PERSISTENT | 1,792 | 0.000285 | 1,792 | 0.0002938588 | 10.4703% |
| HEALED | 1,352 | 0.000215 | 519 | 0.0000817617 | 2.9132% |
| TRANSIENT_BORN | 7,427 | 0.001180 | 0 | 0 | 0.000% |

The same table on the PyAV lineage gives PERSISTENT 61.670% / CHURN 24.800% / NEW_PERSISTENT
10.565% / HEALED 2.966% — the verdict is lineage-independent.

### The same table on the LIVE forward, and why it differs

| class | terminal wrong sites | share of terminal error |
|---|---:|---:|
| CHURN | 12,346 | **62.347%** |
| PERSISTENT | 7,291 | **35.779%** |
| NEW_PERSISTENT | 350 | 1.865% |
| HEALED | 1 | 0.008% |

On the live weights the falsifier would fire. The cause is measured, not speculative: past step 512
the live `d_seg_hat` has standard deviation **3.108e-4 = 9.56%** of its own mean and moves
**2.948e-4** between consecutive swept checkpoints (max 1.019e-3), against the shadow's 1.252e-4 =
4.05%. **40,290** sites therefore flip more than four times and land in CHURN rather than in an
endpoint class — 3.69× the shadow's 10,924. **The classes are
cadence- and forward-conditional by construction**, which is exactly why the equation registered
below excludes transferring the share across cadences.

### The birth is frame-selective; the persistent core is universal

Per-pair terminal `PERSISTENT` sites (shadow, DALI, 11,344 over 32 pairs): min 204, median 343,
max 677 — a spread of only **3.32×**, and the worst 8 of 32 pairs carry **32.55%** of it (a uniform
split would be 25%). The persistent core is therefore a per-frame Lane-boundary property present in
every frame, not a few pathological frames. The BIRTH is the opposite: six of the 32 pairs carry
**26.6%** of the 24,336 born sites, led by pair 573 with Lane painted at 1.683× its GT area. **What
the optimizer breaks is frame-selective; what the representation cannot do is universal.** Any probe
that picks a frame should pick from different lists for the two questions.

## 4. Reachability against the sub-0.12 accuracy corner

| reading | terminal `d_seg_hat` | × target | PERSISTENT floor | × target | optimizer-reachable share |
|---|---|---:|---|---:|---:|
| shadow, DALI | 0.0028065999 | **20.57×** | **0.0017403920** | **12.75×** | 37.99% |
| shadow, PyAV | 0.0027610779 | 20.23× | 0.0017027543 | 12.48× | 38.33% |
| live, DALI | 0.0030947367 | 22.68× | 0.0011072790 | 8.11× | 64.22% |

Target `1.3646784205e-4` (qn1, DERIVED, n600, at the falsifier pose on the bound 106,626 B archive).
The n32 HT estimator estimates that n600 population, and qn1's own caveat that **n32 → n600 is
untested on this vehicle** travels with every row above.

**The reading.** Remove every non-persistent terminal error — every churning site, every site the
run created, every partially-repaired site — and the shipped object still sits at 12.75× the
accuracy corner. No schedule, transition, τ-band or area-cap lever can reach the target on this
representation, because the error they can touch is 37.99% of a number that is 20.57× too large.
That is not a re-grading of gc1's capacity closure; it is an independent, trajectory-based
measurement that agrees with it on the right object.

## 5. The birth: what the first sixteen updates do

| quantity | step 0 | step 16 (live) | step 16 (shadow) |
|---|---|---|---|
| `d_seg_hat` (DALI) | 0.0025555929 | **0.0060334524** (2.3612×) | 0.0025788625 (1.0091×) |
| Lane predicted / GT area | 1.03994 | **1.29558** | 1.04747 |
| Movable predicted / GT area | 1.02429 | **1.11117** | 1.02561 |
| `d_pose_hat` (PyAV targets) | 0.00057497 | **0.00145287** (2.527×) | 0.00053592 |

**Over-paint birth** — first checkpoint at or above 1.05× GT area:

| class | live | shadow |
|---|---:|---:|
| Lane | **step 16** | step 32 |
| Movable | **step 16** | **step 1,920** |
| Road / Undrivable / MyCar | never | never |

The EMA delays Movable's crossing by **1,904 updates**. Read from the milestone record alone, the
Movable over-paint looks like a slow mid-run drift; it is a step-16 event seen through the filter.

**Where the born error lands (live, at the step-16 peak): 24,336 sites.**

| GT class of born sites | count | predicted class at the peak | count |
|---|---:|---|---:|
| Road | 17,132 (70.4%) | **Lane** | **11,118 (45.7%)** |
| Undrivable | 4,909 (20.2%) | **Movable** | **7,273 (29.9%)** |
| Lane | 1,412 | Road | 3,530 |
| Movable | 553 | Undrivable | 2,075 |
| MyCar | 330 | MyCar | 340 |

Top born edges: **Road→Lane 10,996**, Road→Movable 4,164, Undrivable→Movable 3,065,
Undrivable→Road 1,705, Road→Undrivable 1,699, Lane→Road 1,365. **75.6%** of the born sites are
painted as a rare class, over ground that is 90.6% Road or Undrivable. The rare-class over-paint is
**25,439 sites = 67.65% of the peak error**, of which 72.01% were born during the run and **63.49%
recover** by the terminal checkpoint. The shadow's own born set is 6,907 sites of which 57.90%
recover, 77.3% painted rare.

**Pairs leading the birth** (live, born sites | Lane over-paint at the peak): pair 573
(1,183 | 1.683×), 563 (1,151 | 1.710×), 456 (1,135 | 1.352×), 278 (1,110 | 1.465×),
187 (992 | 1.421×), 52 (915 | 1.242×). These six of the 32 carry 26.6% of the birth.

This is sd1's rare-class over-paint mechanism, localised: to the first sixteen updates, to two
edges, and to six frames.

## 6. The EMA is the reporting delay, and its window is derivable

`decay = 0.9990793899844618`, `warmup = False`; time constant `1/(1−d) = 1086.236` updates;
`d^5000 = 0.010000000000000278` (the sealed terminal seed fraction 0.01, reproduced). The weight the
terminal shadow puts on a single update at step `s` is `(1−d)·d^(5000−s)`:

| window | weight in the terminal shadow |
|---|---:|
| last 1,000 updates | **60.19%** |
| last 2,000 updates | 84.15% |
| first 500 updates | **0.585%** |
| the single update at step 16 | 9.34e-06 |

Two consequences, both measured against the swept series. (a) **The shadow's peak LOCATION is set
by the filter's time constant, not by anything happening near step 2,304.** `dθ̄/dt = (1−d)(θ − θ̄)`,
so the shadow stops moving away from the live field only once it has caught up to it — measured,
**the live `d_seg_hat` first falls at or below the shadow's at step 1,344 and the shadow's own peak
follows at step 2,304**, about one time constant later. Nothing distinguishes step 2,304 in the live
field; the milestone record's "peak at 2,000" is the filter's arrival time at a field that has been
worse since step 16.
(b) The terminal shadow excess is **not** memory of the birth: only 0.585% of its weight comes from
the first 500 updates. **The shadow ends above its start because the live field ends above its
start** — measured, the live `d_seg_hat` never once falls below the start's 0.0025555929 across all
70 swept live checkpoints (minimum 0.0027205 at step 3,072, still +6.45%). The EMA delayed the report; it did not
manufacture the loss. It also earns its keep: the live series' 9.56% checkpoint-to-checkpoint
scatter would otherwise dominate any single reading (the shadow's is 4.05%).

Peak live/shadow ratio **2.3396× at step 16** — this closes sd1's owed `LIVE-VS-SHADOW-RESIDUAL`
gap, which sd1 recorded as UNMEASURED because "it would need the live-weights logits, which the
milestone does not retain". The checkpoints retain the live weights, so the forward could simply be
re-run: the gap decays 2.3396× → 1.3707× (step 48) → 1.1653× (step 256) → 1.0587× (step 576) → 1.0
at step 1,344.

## 7. Optimizer micro: the cold transition is a first-64-update event

Per-16-step displacement `‖θ_t − θ_{t−16}‖₂` by parameter role (live weights):

| step | boundary_flow | coarse_partition | interior_field | rgb_renderer | **pose_head** |
|---:|---:|---:|---:|---:|---:|
| 16 | **0.128224** | **0.072274** | 0.050190 | 0.035873 | **0.000000** |
| 32 | 0.100808 | 0.045578 | 0.037595 | 0.026632 | 0.000000 |
| 64 | 0.052137 | 0.038534 | 0.018546 | 0.014097 | 0.000000 |
| 128 | 0.040282 | 0.036193 | 0.014100 | 0.011010 | 0.000000 |
| 256 | 0.039427 | 0.032856 | 0.012814 | 0.010099 | 0.000000 |
| 1,024 | 0.045636 | 0.039178 | 0.016482 | 0.013292 | 0.000000 |

The first 16 updates move `boundary_flow` **3.25×** its steady-state per-16-step displacement, and
`coarse_partition` by **0.568%** of its own norm — the largest relative move of any role. The excess
decays 0.128224 → 0.100808 → 0.052137 → **0.040282 by step 128**, where it is already at the steady
value it holds for the rest of the run (0.039–0.048). The mechanism is visible in AdamW's own state: `Σ exp_avg_sq` for
`boundary_flow` grows **0.00551892 → 0.20744971** between steps 16 and 1,792, a 37.58× growth whose
square root is **6.13×** — the same order as ng1's measured cold/warm first-step ratio of 6.4581×.
(Those are different quantities — ng1's warm arm carries r10's second moment from 10,010 updates,
not this run's at step 1,792 — so this is a magnitude agreement that names the mechanism, not a
confirmation of ng1's number.)

## 8. Two things the milestone record cannot show

**(a) The margin distribution WIDENS while `d_seg` worsens.** Shadow, DALI: mean margin
**4.186669 → 4.241569 (+1.311%)**, 1st percentile 0.247935 → 0.261678, and the count of sites
within 25·δ_R falls **133,151 → 125,785 (−5.53%)**. The field becomes more confident — including on
the sites it has newly got wrong. This composes with, and is additional to, sd1's τ-schedule
identity: the training surrogate falls both because τ shrinks 3× and because the margins it measures
genuinely widen.

**And the classes sit at different DEPTHS.** Terminal margin band membership (shadow, DALI):

| class | within δ_R | within 25·δ_R | beyond 25·δ_R | n |
|---|---:|---:|---:|---:|
| PERSISTENT | 4.70% | 66.31% | **33.69%** | 11,842 |
| NEW_PERSISTENT | 25.17% | **99.78%** | 0.22% | 1,792 |
| CHURN | 21.56% | 99.70% | 0.30% | 10,924 |
| TRANSIENT_BORN | 8.83% | 98.06% | 1.94% | 7,427 |
| HEALED | 14.20% | 98.45% | 1.55% | 1,352 |

**The error the run CREATED is shallow; the error it INHERITED is deep.** 99.78% of
`NEW_PERSISTENT` sites are still within 25·δ_R — a small logit change would move them — while
33.69% of `PERSISTENT` sites sit beyond 25·δ_R, where no plausible perturbation of this
representation reaches. That is the same conclusion §4 reaches from the reachability arithmetic,
arrived at independently from the margin field.

**(b) 1,836 parameters — 2.309% of the counted packet — receive zero gradient.** The
`fairform_objective` scores pose from PoseNet's forward on the rendered camera
(`ddm_qbt1_qbflow_trainer.py:533`), never from the model's own `pose12` head
(`QBFLOWTorch.forward:402` returns it and the objective ignores it). MEASURED across three sealed
cells (control seeds 20260902/20260903, treatment seed 20260902): AdamW never allocates state for
`params.pose_in_{w,b}` / `params.pose_out_{w,b}`, and `max|Δθ| = 0.000e+00` over all 5,000 updates
while other roles moved up to 0.103. Their measured coded cost in the shipped model section is
**2,014 bytes** (79,692 → 77,678 when their values are zeroed, ABI preserved), against an
ABI-breaking raw bound of 2,072 B: **1.889% of the 106,643-byte archive**. The encode path
reproduces the sealed encoder byte-exactly (model section 87,854 raw / 79,692 coded, matching the
step-5000 `section_facts` row), so this is an exact-byte measurement, not an estimate.
This answers to the packet schema, not to any training lever.

## 9. Warm versus cold — the moments damp the first sixteen updates and nothing after

The ng1 warm cell (same seed, same data order, one lever: r10's AdamW `exp_avg`/`exp_avg_sq`/`step`
carried in) completed at 14:15Z, 5,000 updates, 11,065.6 s. Its own step-0 milestone is bit-identical
to the cold control's, and its AdamW step counter reads **10,026** at step 16 — 10,010 carried plus
16 — so the single lever is confirmed live in the retained state, not just in the config.

**The warm cell's own milestone record (its axis, `[macOS-MPS n32 stratified advisory]`, read
read-only; this arm does not adjudicate ng1's verdict, it measures the trajectory):**

| step | warm `S_hat` | warm `d_seg_hat` | warm `d_pose_hat` | cold `S_hat` | warm − cold |
|---:|---|---|---|---|---:|
| 0 | 0.398768 | 0.0025183 | 0.0005757 | 0.398768 | 0.000000 |
| 1,000 | 0.439328 | 0.0029868 | 0.0004834 | 0.466875 | **−0.027547** |
| 2,000 | 0.467442 | 0.0031367 | 0.0006851 | 0.485677 | **−0.018235** |
| 3,000 | 0.461163 | 0.0030496 | 0.0007258 | 0.475383 | **−0.014220** |
| 4,000 | 0.458423 | 0.0030151 | 0.0007383 | 0.442190 | **+0.016233** |
| 5,000 | **0.443798** | **0.0028681** | 0.0007401 | 0.425149 | **+0.018649** |

**ng1's PRIMARY falsifier fires on both clauses.** Clause 1 (`S_hat(5,000) < 0.398768`) fails —
0.443798 is +11.3% above the warm start. Clause 2 (below the cold control at every milestone) fails
at 4,000 and 5,000. The warm cell leads for three milestones and then loses; its terminal
`d_seg_hat` is **+3.96%** worse than the cold control's and its `d_pose_hat` **+20.9%** worse.

**What this arm's micro data adds: the carried moments buy exactly the first ~96 updates.**
Total per-16-step displacement `‖θ_t − θ_{t−16}‖₂`, cold vs warm at identical steps (live weights,
same data order, so the optimizer state is the only cause available):

| step | cold ‖Δθ‖ | warm ‖Δθ‖ | cold/warm | cold `d_seg_hat` | warm `d_seg_hat` | warm − cold |
|---:|---:|---:|---:|---|---|---:|
| 16 | 0.162308 | 0.065990 | **2.460×** | 0.0060335 | 0.0034395 | **−0.0025940** |
| 32 | 0.121681 | 0.070582 | 1.724× | 0.0044853 | 0.0033442 | −0.0011411 |
| 48 | 0.086387 | 0.069902 | 1.236× | 0.0035971 | 0.0036743 | +0.0000772 |
| 64 | 0.070867 | 0.064446 | 1.100× | 0.0035940 | 0.0031461 | −0.0004478 |
| 96 | 0.062193 | 0.063864 | **0.974×** | 0.0035605 | 0.0032828 | −0.0002777 |
| 192 | 0.062244 | 0.060688 | 1.026× | 0.0032764 | 0.0036716 | +0.0003952 |
| 464 | 0.067442 | 0.069436 | 0.971× | 0.0038907 | 0.0035950 | −0.0002958 |

The damping is **2.460× at step 16, 1.100× by step 64, and gone by step 96** — after which the two
cells are indistinguishable in step size and their `d_seg` difference changes sign from checkpoint
to checkpoint. Carried second moments are a first-hundred-update intervention, exactly as §7's
`Σ exp_avg_sq` growth predicts, and they do not touch anything after that.

**The warm cell's whole trajectory is displaced, not scaled.** Its live `d_seg_hat` peaks at step
**224** (0.0040343), not at 16 (0.0034395); its shadow peaks at step **2,000** (0.0031752), not
2,304. Terminal shadow class table (DALI, 71 checkpoints, calibration gate 0): PERSISTENT
**59.009%** (numerator 202,590 of 343,320), CHURN 25.922%, NEW_PERSISTENT **11.399%** (39,135 —
13% MORE created-and-kept error than the cold cell's 34,665), HEALED 3.670%. Reachability floor
0.0017173771 = **12.58×** the target; terminal 21.33×.

### Prediction 2 — FALSIFIED, and not by the choice of comparison moment

At each cell's own d_seg peak (the pre-registered definition):

| forward | cold peak | warm peak | cold born | warm born | intersection | warm-only | warm-only fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| shadow | 2,304 | 2,000 | 6,907 | 6,617 | 3,613 | 3,004 | **45.40%** |
| live | 16 | 224 | 24,336 | 11,722 | 7,222 | 4,500 | **38.39%** |

Both exceed the 30% falsifier. Because the two peaks fall at different steps, part of that could in
principle be "different moment" rather than "different sites", so the same comparison was re-run at
**identical steps** in both cells (same seed, same data order, `born(s) = correct at 0 AND wrong at
s`):

| step | live warm-only | live Jaccard | shadow warm-only | shadow Jaccard |
|---:|---:|---:|---:|---:|
| 16 | 37.69% | 0.199 | 40.50% | 0.205 |
| 64 | 47.95% | 0.279 | 38.59% | 0.270 |
| 224 | 63.81% | 0.282 | 43.64% | 0.301 |
| 512 | 50.34% | 0.292 | 45.41% | 0.332 |
| 1,024 | 58.66% | 0.273 | 45.61% | 0.364 |
| 2,000 | 57.58% | 0.292 | 47.18% | 0.358 |
| 2,304 | 57.33% | 0.270 | 42.81% | 0.374 |
| 3,072 | 64.23% | 0.244 | 47.07% | 0.365 |
| 4,000 | 59.36% | 0.257 | 51.04% | 0.348 |
| 5,000 | 64.80% | 0.224 | 56.18% | 0.299 |

**20 of 20 comparisons fire the falsifier**, warm-only 37.69%–64.80%, Jaccard 0.199–0.374
throughout. **The carried moments do not damp a fixed trajectory — they put the run on a different
one.** Same seed, same data order, one lever, and after the first hundred updates the two cells
break substantially different sites at every step to the end. Whatever the warm start buys, "the
same excursion, smaller" is measurably not it.



## 10. GESTALT — micro → meso → macro

**Micro.** In the first sixteen updates a freshly-constructed AdamW, whose second moment is still
near zero, takes a step of size ≈`lr·sign(g)` on every parameter: `boundary_flow` moves 0.128224
(3.25× its own steady-state per-16-step displacement) and `coarse_partition` by 0.568% of its norm,
while `Σ exp_avg_sq` for `boundary_flow` is still 0.00552 of the 0.20745 it will reach. Those
sixteen steps make 24,336 sites wrong — 45.7% of them by painting Lane over Road, 29.9% by painting
Movable over Road or Undrivable — and push Lane to 1.29558× its GT area. The mechanism is the
recall-only dual with nothing capping area (sd1), fired through an optimizer with no history.

**Meso.** The live field never recovers to its initialisation: its `d_seg_hat` floor over the whole
run is 0.0027205, +6.45% above the start, and it scatters at 9.56%. The EMA shadow — the object the
archive is re-encoded from — is a 1,086-update low-pass over that field. It reports 1.0091× of the
damage at step 16, catches the live field at step 1,344, peaks at step 2,304, and lands at
0.0028066. **The milestone record's "excursion peaking at step 2,000" is the filter's arrival time,
not the event.** Meanwhile the margin distribution widens 1.311% and the count of sites near the
decision boundary falls 5.53%: the field becomes more confident, and confidently wrong on what it
newly broke — which is why the training surrogate falls monotonically while the exact argmax rises,
on top of sd1's τ-schedule identity.

**Macro.** In exact integer numerators the run repaired 24,435 units of pre-existing error and
created 54,045 — **2.21× as much as it repaired** — for `S_hat`'s d_seg leg to end +9.82% above its
start. And the error it could never touch dominates what remains: **62.011%** of the terminal
`d_seg_hat` sits in sites that were wrong before update 1 and stayed wrong, 64.79% of them on a Lane
edge, GT-Lane enriched **51.50×** over Lane's 0.594% of the frame, a third of them beyond 25·δ_R.
Delete every optimizer-reachable site and the shipped object is still **12.75×** the sub-0.12
accuracy corner.

**Which lever class each component answers to.**

| component | numerator at 5,000 | share | the lever class that can move it |
|---|---:|---:|---|
| PERSISTENT | 205,305 | 62.011% | **REPRESENTATION**, and specifically class-protected: gc1 measured class-blind capacity buying Road 2.59× and Undrivable 3.03× but Lane 1.16×, and this arm measures the survivors to be 64.79% Lane-touching and 33.69% beyond 25·δ_R |
| CHURN | 81,465 | 24.606% | **MARGIN / τ** — 21.56% sit inside δ_R, where the uint8 roundtrip decides the class, not the field; the rest is a commitment problem |
| NEW_PERSISTENT | 34,665 | 10.470% | **OPTIMIZER + OBJECTIVE** — the transition (ng1), the area cap (ng2), the τ band (ng3); 99.78% are still within 25·δ_R, so they are shallow and genuinely reachable |
| HEALED (residual) | 9,645 | 2.913% | the run's own unfinished repair; more updates, not a new lever |
| TRANSIENT_BORN | 0 | 0.000% | **SCHEDULE** — costs 5,000 updates of wall-clock and zero terminal score |
| dead pose head | — | 2,014 B = 1.889% of the archive | **PACKET SCHEMA (ABI)** — no training lever reaches it |

**And the warm cell shows what "aiming at the reachable share" actually buys.** It damps the first
hundred updates by up to 2.46×, leads the cold control by 0.0275 `S_hat` at 1,000 steps — and ends
+0.0187 ABOVE it, with 13% more created-and-kept error and a persistent floor of 12.58× the target.
It did not walk a smaller version of the cold trajectory; it walked a different one, breaking a
37.7–64.8% disjoint set of sites at every step.

The three sealed cells queued against this object — ng1's warm transition, ng2's area cap,
ng3's τ band — all aim at the 37.99% share. Their combined credit ceiling is a factor of 1.61 on
`d_seg`, against a factor of 20.57 needed. **They are the right levers for the error they address
and they cannot close the accuracy corner on this representation.** The corner needs a different
object, and this measurement says precisely which sites that object has to get right.

## 11. Equations leg

**Consumed, unchanged:**

* `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — the per-site scalar this arm partitions.
  Every `w[t, s]` in §3 is `1[margin < 0]` on the same margin field the law defines, and §8(a)'s
  band table is that margin measured against the δ_R ladder. No refinement claimed.
* `muon_finisher_schedule_warmstart_and_lr_anneal_v1` — SIBLING/PREDICTIVE, not in-domain (Muon +
  cosine on the MLX witness vs AdamW + constant LR here), exactly as ng1 recorded. §7 measures the
  same SHAPE on this vehicle — a cold first-order buffer producing an oversized first move that
  decays as the second moment accumulates — and reports it as a magnitude agreement, not as an
  anchor. **No anchor appended.**
* `persistence_topology_cldice_betti_island_recall_v1` — cited to mark a DISTINCTION, not a use.
  That law's persistence is topological (birth–death pairs of a scalar field); this arm's
  persistence is temporal (a site wrong at ≥90% of checkpoints). They are different objects and
  must not be conflated; the name collision is the reason this note exists.

**Registered (new):** `checkpoint_trajectory_error_partition_v1`
(`src/tac/canonical_equations/checkpoint_trajectory_error_partition_20260904.py`, exported from
`tac.canonical_equations`, appended to `.omx/state/canonical_equations_registry.jsonl`). It exports
`partition_is_exact` (the integer gate, which REFUSES an empty class map rather than passing
vacuously), `reachability_floor` (the credit ceiling), and `floor_clears_target`. Its
`domain_of_validity.excluded` forbids transferring the persistent SHARE across cadences, reading the
floor as a prediction, a float-tolerance gate, non-integral sample weights, and any score or
promotion use. Its single anchor carries BOTH readings (shadow 62.011%, live 35.779%) so the number
can never be quoted without its forward. Residual |0.6201069 − 0.60| = **0.0201069**.

## 12. Custody, apparatus, and what this cost

`$0`. CPU only, `torch.set_num_threads(4)`, `nice`; **0 Metal, 0 Modal, 0 contest eval**; nothing
written under any cell's `runs/`; `upstream/` and `submissions/` untouched; no `/tmp` path in any
artifact. Store `/Volumes/APDataStore/pact/ddm_md1_micro_macro/` with a `CUSTODY_MANIFEST.json`
listing every file with bytes and sha256:

| artifact | what it holds |
|---|---|
| `payloads/<cell>/<forward>_step_<n>.npz` | the 32-pair argmax (uint8) and δ_R-band code for every swept checkpoint — the payload every table here is derived from |
| `sweep_rows_<cell>.jsonl` | one row per (step, forward): both GT lineages' `d_seg_hat`, per-pair rows, per-class predicted area, margin histogram and quantiles, band counts, per-role weight/displacement norms, AdamW moment norms, checkpoint and payload facts with sha256 |
| `ANALYSIS_<cell>_<lineage>.json` | the bridge, the class tables, the terminal edges and bands, the excursion and reachability blocks |
| `site_classes_<cell>_<forward>_<lineage>.npz` | the per-site class code (32, 384, 512) |
| `excursion_<cell>_<forward>_<lineage>.npz` | the per-site trajectory code `at_zero + 2·at_peak + 4·terminal` |
| `REPORT.json`, `TABLES_cold_dali.md` | the rendered tables — every number in this memo is rendered from JSON, never retyped |
| `DEAD_PARAMETERS.json`, `DEAD_PARAMETER_BYTES.json` | the §8(b) receipts across three sealed cells |
| `EMA_MEMORY_WINDOW.json` | the §6 derivation |
| `SAME_STEP_BORN_OVERLAP.json` | the §9 robustness control: born-set overlap at ten identical steps × two forwards |
| `COMPARE_dali.json` | the pre-registered own-peak born-set comparison |
| `SUPERSEDED_smoke_sweep_receipt_cold.json` | the 3-checkpoint smoke receipt, content + sha preserved, superseded by the full pass |

341 files, 29,466,549 B. Both cells' 71-checkpoint sweeps: 282 forwards, ~2.4 h of CPU at 4 threads,
peak RSS 15.31 GiB. Instrument `experiments/ddm_md1_micro_to_macro.py` (commit `0af527a80`,
completed `c3bc0e033`, `81b2b6f4e`), 70 tests; the law's 29. ar1's GT loader, sd1's δ_R constant and the sealed trainer's
own forward/roundtrip/scorer are called, never rebuilt.

## NEXT_IF_RESUMED

1. **The accuracy corner needs a REPRESENTATION change on this object, and the persistent set names
   where.** 11,842 sites, 64.79% Lane-touching, GT=Lane enriched 51.50×, and 3,990 of them
   (33.69%) already beyond 25·δ_R at the terminal checkpoint — out of reach of any perturbation this
   representation can make. A class-protected or curve-domain generator aimed at
   exactly that set is the next thing worth building; gc1's class-blind square atoms are measured
   not to buy it.
2. **The first 64 updates are the right target and the warm start is not the right instrument.**
   §7 confines the excess displacement to steps 1–64 and §5 puts the whole rare-class over-paint
   inside step 16, so the target is real. But §9 measures that carried moments REDIRECT rather than
   damp — 37.7–64.8% of the warm cell's broken sites are absent from the cold cell's at every step,
   and the cell ends +0.0187 `S_hat` worse. A **gradient-norm clip or an LR ramp over the first ~64
   updates** is the same intervention without a donor state: it scales the first steps of THIS
   trajectory instead of substituting another one's history. Two lines, zero bytes, one cell.
   That is the cheapest untried thing this arm found.
3. **2,014 counted bytes are free.** The pose head cannot be reached by this objective. Removing it
   needs a QBF1 ABI change (`qbf1.expected_param_shapes`), which is a packet-schema decision, not a
   training one. 1.889% of the archive, 4.79% of the −42,016 B rate demand.
4. **The n32 → n600 question is still open** and every reachability row here inherits it. The qn1
   realization ticket is the instrument; it has never been executed.
5. **Owed by this arm:** a second SEED at the same cadence (the 20260903/20260904 controls retained
   the same 313-checkpoint record). The two instances measured here — cold 62.011% and warm 59.009%
   on the shadow, 35.779% and 35.336% on the live — agree to 3.0 pp and 0.44 pp, but they are the
   SAME seed; a different seed is what turns the persistent share into a seed-independent reading.
   ~2.4 h of CPU, `$0`, no new code. Nothing else in this memo needs new compute.
