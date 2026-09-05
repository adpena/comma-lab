---
title: "The burn default cuts the damage the run creates by 2.44x and does not move the wall: under τ band × carried duals the PERSISTENT share is 62.954% — HIGHER than the cold control's 62.011% — the two schedules fail on the SAME sites (Jaccard 0.8069, 69.4% of the way from within-pool chance to the attainable ceiling), the over-paint birth is still step 16, and the floor only falls from 12.753x to 11.671x the sub-0.12 corner"
arm: ddm_md2
charter: .omx/research/charters/ddm_md2_persistent_partition_under_the_burn_default_20260905.md
parent_arms: [ddm_md1, ddm_ng5, ddm_gs4]
utc: 2026-09-05T13:55:00Z
verdict_scope: "[macOS-CPU advisory . exact argmax reconstructed from the retained 16-step checkpoints . frozen CPU-torch SegNet+PoseNet . QBF1-born vehicle . n32 sealed selection . seed 20260902 ng5 τ-band × carried-duals cell vs md1's cold control . SHARED INITIALISATION (init sha 991a1cc6…, bit-identical) . NON-PROMOTABLE . no score claim . 0 Metal / 0 Modal / 0 contest eval]"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_md2 — does the burn default change WHICH d_seg sites are reachable?

## VERDICT: **STANDS**

`PERSISTENT` = **62.954%** of the terminal `d_seg` on the EMA shadow / DALI authority — the
authoritative reading, ≥ 60%. md1's capacity closure stands unchanged on the burn default of
record. The born route needs a mechanism that changes which sites are reachable, not a schedule,
exactly as gs4 §5 says.

## The prior-law prediction and both falsifier directions, read out before the numbers

The charter (committed before the sweep launched at 2026-09-05T12:51:58Z) predicted:

> **PREDICTION:** `PERSISTENT` under ng5 falls to **40–55%** of the (smaller) terminal `d_seg` — a
> real reduction of the unreachable set but NOT below md1's 40% falsifier; the Lane-touching share
> stays ≥ 60%.
> **FALSIFIER (either direction decides):** `PERSISTENT` < 40% → md1's falsifier FIRES on the burn
> default → the capacity closure (gc1/gf2) is RE-GRADED and a born trainer with Lane held in-loop
> becomes admissible. `PERSISTENT` ≥ 60% → the closure stands unchanged and the born route needs a
> new generator, not a schedule.

| claim | reading | measured | verdict |
|---|---|---:|---|
| `PERSISTENT` falls to 40–55% | **EMA shadow, DALI** | **62.954%** | **FALSIFIED — it ROSE 0.943 pp** |
| same | EMA shadow, PyAV | 62.346% | FALSIFIED (rose 0.676 pp) |
| same | live weights, DALI | 35.439% | below 40%, as on the cold control (35.779%) |
| same | live weights, PyAV | 35.112% | below 40%, as on the cold control (35.409%) |
| Lane-touching share ≥ 60% | shadow, DALI | **65.52%** | **HOLDS** (cold 64.79%) |
| falsifier `< 40%` → RE-GRADE | shadow (the shipped object) | 62.954% | **did NOT fire** |
| falsifier `≥ 60%` → closure stands | shadow | 62.954% | **FIRED** |

**The prediction was wrong in the direction that matters.** I expected the τ band to shrink the
unreachable set because it suppresses the over-paint that md1 measured at step 16. It shrank the
set in ABSOLUTE terms (11,842 → 11,019 sites; floor 0.0017403920 → 0.0015927633) and shrank the
terminal error more (0.0028065999 → 0.0025300344), so the unreachable SHARE went **up**. That
arithmetic is the finding: the burn default removes error the optimizer could already reach, and
leaves the part it cannot almost untouched.

**The reading travels with the number,** per md1: the archive is re-encoded from `ema.shadow`
(`experiments/ddm_qbr1_born_fairform_burn_prep.py:629-632`), so the shadow is the object a
submission would contain and is the authoritative reading. The live forward is reported beside it
on both cells; neither is chosen after the fact.

## The finding, first

**The best burn default we have repairs 2.44× better and moves the wall by 8.5%.** Over 5,000
updates the cold control created 2.2118× as much error as it repaired; ng5 creates **0.9053×** — it
repairs slightly more than it makes. The excursion peak falls from +28.78% to +7.44% of the start
on the shadow, and the shadow peak arrives at step **640** instead of 2,304. The terminal `d_seg`
ends **1.00% BELOW its start** where the cold control ended 9.82% above it.

**And none of that touches the wall.** The `PERSISTENT` floor — what remains if a lever removed
every non-persistent terminal error — falls only from **12.753×** to **11.671×** the sub-0.12
accuracy corner. The two schedules fail on substantially the SAME sites: Jaccard **0.8069**, with
**92.65%** of ng5's persistent sites also persistent in the cold control. The over-paint birth is
still a **step-16** event: Lane and Movable both cross 1.05× their GT area at step 16 on the live
forward, identically to the cold control; the band changes the size of the birth (live `d_seg`
peak 1.9824× the start vs the cold control's 2.3612×, −16.0%) and not its timing.

## 1. What was measured, and the reference form

md1's instrument, **UNCHANGED** (`experiments/ddm_md1_micro_to_macro.py`, `--mode sweep` then
`--mode analyze`), on ng5's run root, same cadence, same `--churn-flips 4`, same
`PERSISTENT_FRACTION 0.90`, same `--threads 4`, both GT lineages, both forwards.

| premise | verified at | label |
|---|---|---|
| ng5's cadence is IDENTICAL to md1's cold control | `available_steps` on both run roots returns the same 71 steps → 141 forwards | MEASURED |
| ng5 starts from the SAME state as the cold control | both sealed configs name `qbr1_from_r10_ema_state.pt`, sha `991a1cc653c786af…`, 398,687 B | MEASURED (source) |
| …and the reconstruction confirms it | step-0 `d_seg_hat` DALI **0.0025555929** on ng5 = md1's cold step-0 value, bit-for-bit; step-0 weighted numerator 301,470 on both | MEASURED |
| ng5's authorized config | `/Volumes/APDataStore/pact/ddm_ng5_tau_band_x_continuous_objective/authorized_configs/seed_20260902_tau_band_x_continuous_objective_control_native100.json`; `total_steps` 5000, seed 20260902, same sealed EMA law | MEASURED (source) |
| the run root retains 313 checkpoints (312 periodic + `stage_01_end.pt`) | directory listing | MEASURED |
| the comparator is md1's own store, never a re-typed number | every cold column below is read from `ANALYSIS_cold_control_seed_20260902_{dali,pyav}.json` and persisted to `COLD_COMPARATOR_extract.json` | MEASURED |
| `δ_R` n600 = 0.021881818771362305 | `experiments/ddm_sd1_surrogate_exact_map.py::DELTA_R_N600`, law-resolved, never retyped | TRANSFERRED (dr1) |
| the sub-0.12 accuracy corner `d_seg = 1.3646784205e-4` | `.omx/research/ddm_qn1_qbr1_n600_realization_ticket_20260903.md` | TRANSFERRED (DERIVED there, n600, at the falsifier pose, on the bound 106,626 B archive) |

**The instrument was validated against md1's published numbers before it was pointed at ng5.** Four
positive controls, all reproduced exactly on md1's own cold cell: Lane-touching **64.79%**
(7,350/11,344); GT=Lane enrichment **51.503511×**; the over-paint birth table (live Lane 16 /
Movable 16, shadow Lane 32 / Movable 1,920 — md1 §5 verbatim); `NEW_PERSISTENT` 1,792 sites. The
terminal-wrong mask read out of the excursion payload agrees with the independently computed class
rows on both forwards (11,344 shadow, 7,291 live).

**The n32 sealed selection, stated as md1 stated it.** All 32 pairs of the sealed no2 stratified
Horvitz–Thompson selection, integer weights `(15.0,)*24 + (30.0,)*8`, pair ids
`(4, 31, 49, 52, 62, 90, 100, 113, 128, 148, 173, 179, 186, 187, 214, 236, 256, 260, 268, 278, 326,
328, 341, 352, 368, 382, 444, 456, 483, 508, 563, 573)`. **The n32 HT estimator estimates the n600
population, and qn1's own caveat that n32 → n600 is untested on this vehicle travels with every row
below.** One correction to how that caveat is sometimes repeated: this selection is **stratified
across the video (ids 4…573), not a contiguous prefix**, so the prefix-bias genus
(`[[m88]]`: a prefix of a skewed population is a different population, pose 2.5–4.2× harder, seg
0.96× easier) is **not** the applicable caveat here. The applicable one is n32 → n600 transfer, and
it is untested.

## 2. THE TABLE — md1's partition on ng5, against the cold control

Every column is read from JSON; nothing is retyped. Denominator 117,964,800; sites 6,291,456.

### EMA shadow, DALI authority (the shipped object, the authoritative reading)

| class | ng5 sites | ng5 terminal-wrong | ng5 share | cold sites | cold terminal-wrong | cold share |
|---|---:|---:|---:|---:|---:|---:|
| **PERSISTENT** | **11,019** | **10,488** | **62.954%** | 11,842 | 11,344 | 62.011% |
| CHURN | 9,997 | 3,871 | 24.687% | 10,924 | 4,323 | 24.606% |
| NEW_PERSISTENT | 1,447 | 1,447 | 9.142% | 1,792 | 1,792 | 10.470% |
| HEALED | 1,687 | 484 | 3.217% | 1,352 | 519 | 2.913% |
| TRANSIENT_BORN | 6,270 | 0 | 0.000% | 7,427 | 0 | 0.000% |
| ALWAYS_CORRECT | 6,261,036 | 0 | 0.000% | 6,258,119 | 0 | 0.000% |

### EMA shadow, PyAV (the vehicle's own training target)

| class | ng5 sites | ng5 terminal-wrong | ng5 share | cold share |
|---|---:|---:|---:|---:|
| **PERSISTENT** | **10,759** | **10,230** | **62.346%** | 61.670% |
| CHURN | 9,999 | 3,864 | 25.050% | 24.800% |
| NEW_PERSISTENT | 1,446 | 1,446 | 9.302% | 10.565% |
| HEALED | 1,697 | 491 | 3.302% | 2.966% |

### live weights, DALI / PyAV

| class | ng5 DALI share | cold DALI share | ng5 PyAV share | cold PyAV share |
|---|---:|---:|---:|---:|
| CHURN | 59.275% | 62.347% | 59.514% | 62.661% |
| **PERSISTENT** | **35.439%** | 35.779% | **35.112%** | 35.409% |
| NEW_PERSISTENT | 5.263% | 1.865% | 5.350% | 1.921% |
| HEALED | 0.023% | 0.008% | 0.023% | 0.008% |

**The verdict is lineage-independent and forward-consistent with md1's.** On the shadow the share is
62.3–63.0% on both lineages; on the live weights it is 35.1–35.4% on both, for the same measured
reason md1 gave — the live weights are noisy checkpoint-to-checkpoint and `CHURN` absorbs 59.3% of
their terminal error. Both readings sit within 0.4 pp of the cold control's corresponding reading.

**The macro bridge is an exact integer identity here too:**
`max_t |Σ_classes − total| = 0` in integers at every one of the 71 checkpoints, on both forwards
and both GT lineages. No float tolerance is involved.

### Reachability against the sub-0.12 accuracy corner

| reading | terminal `d_seg_hat` | × target | PERSISTENT floor | × target | optimizer-reachable share |
|---|---|---:|---|---:|---:|
| **ng5 shadow, DALI** | **0.0025300344** | **18.539×** | **0.0015927633** | **11.671×** | 37.046% |
| cold shadow, DALI | 0.0028065999 | 20.566× | 0.0017403920 | 12.753× | 37.989% |
| ng5 shadow, PyAV | 0.0024878184 | 18.230× | 0.0015510559 | 11.366× | 37.654% |
| cold shadow, PyAV | 0.0027610779 | 20.232× | 0.0017027537 | 12.477× | 38.330% |
| ng5 live, DALI | 0.0027904510 | 20.448× | 0.0009888967 | 7.246× | 64.561% |
| cold live, DALI | 0.0030947367 | 22.677× | 0.0011072795 | 8.114× | 64.221% |

**Delete every optimizer-reachable terminal error under the best burn default we have and the
shipped object still sits at 11.671× the accuracy corner.** The lever moved the floor by
**8.48%**; it needs to move it by **91.4%**.

## 3. The integer bridge: what the burn default actually bought

ng5, EMA shadow, DALI, weighted integer numerators:

| class | step 0 | terminal | Δ |
|---|---:|---:|---:|
| PERSISTENT | 197,790 | 187,890 | **−9,900** |
| CHURN | 72,150 | 73,680 | +1,530 |
| NEW_PERSISTENT | 0 | 27,285 | +27,285 |
| HEALED | 31,530 | 9,600 | −21,930 |
| TRANSIENT_BORN | 0 | 0 | 0 |
| **total W** | **301,470** | **298,455** | **−3,015 (−1.00%)** |

**Repaired 31,830 vs created 28,815 = 0.9053×**, against the cold control's **2.2118×**. That is a
**2.44× improvement in burn quality** and it is the honest headline for what the τ band × carried
duals buys.

**The persistent core still barely moves.** 5,000 updates removed **5.005%** of the persistent
numerator (PyAV: 5.089%); the cold control removed **4.233%** (PyAV: 4.425%). The burn default buys
**0.77 pp** on the error the field carries into update 1.

## 4. THE SITE OVERLAP — same sites, and the honest null

The charter asked: same sites ⇒ capacity-limited regardless of schedule; different ⇒
optimizer-path-dependent and reachable.

| forward / lineage | ng5 persistent | cold persistent | intersection | **Jaccard** | ng5 ⊂ cold |
|---|---:|---:|---:|---:|---:|
| **shadow / DALI** | 11,019 | 11,842 | **10,209** | **0.8069** | **92.65%** |
| shadow / PyAV | 10,759 | 11,621 | 9,973 | 0.8038 | 92.69% |
| live / DALI | 6,703 | 7,396 | 6,294 | 0.8064 | 93.90% |
| live / PyAV | 6,552 | 7,253 | 6,165 | 0.8069 | 94.09% |

**The naive enrichment is 492× chance, and I am not reporting it as the answer, because the null it
uses is wrong.** Both cells share an initialisation, so both persistent sets are — by the class
definition, which requires the site to be wrong at step 0 — subsets of the SAME step-0 wrong pool.
MEASURED: that pool is **16,553 sites** (DALI; 16,327 PyAV) and is **bit-identical across the two
cells**; both persistent sets are verified subsets of it. The subset relation is DEFINITIONAL, not
a finding. Against a null of two independent draws of the observed sizes **from that pool**:

| forward / lineage | measured J | within-pool chance J | attainable max J (given the two sizes) | fraction of the way chance → max |
|---|---:|---:|---:|---:|
| **shadow / DALI** | **0.8069** | 0.5263 | 0.9305 | **69.4%** |
| shadow / PyAV | 0.8038 | 0.5202 | 0.9258 | 69.9% |
| live / DALI | 0.8064 | 0.2697 | 0.9063 | **84.3%** |
| live / PyAV | 0.8069 | 0.2672 | 0.9034 | 84.8% |

**Read that as: same sites, with the caveat welded on.** Two schedules that share a start fail on
overlapping sets that reach 69.4% (shadow) and 84.3% (live) of the way from within-pool chance to
the ceiling their sizes allow. The persistent set is a property of the object at initialisation
that the schedule barely edits — but this experiment cannot separate "the representation cannot fit
these sites" from "no schedule tried so far repairs them", because **every cell measured shares one
initialisation**. That separation needs a different init, and §7 prices it.

**The contrast inside the same table is the sharpest statement of the finding.** The classes the run
MANUFACTURES have low overlap; the class it INHERITS does not (shadow, DALI):

| class | ng5 | cold | intersection | Jaccard |
|---|---:|---:|---:|---:|
| **PERSISTENT** (inherited) | 11,019 | 11,842 | 10,209 | **0.8069** |
| CHURN (manufactured) | 9,997 | 10,924 | 5,541 | 0.3603 |
| TRANSIENT_BORN (manufactured) | 6,270 | 7,427 | 1,829 | 0.1541 |
| HEALED (manufactured) | 1,687 | 1,352 | 464 | 0.1802 |
| NEW_PERSISTENT (manufactured) | 1,447 | 1,792 | 283 | **0.0957** |

**What the optimizer breaks is path-dependent; what it cannot fix is not.** md1 measured the same
shape against the warm cell (37.7–64.8% of born sites disjoint at every step); md2 measures it
against a schedule lever and adds the other half — the inherited class is 0.8069 while the created
classes are 0.096–0.360.

## 5. Lane: MORE concentrated, not less

md1's two formulas, applied unchanged (Lane-touching = terminal edges of the class with `Lane` on
either side; enrichment = the class's GT=Lane site fraction ÷ Lane's HT GT area fraction 0.59436%):

| reading | Lane-touching share of persistent terminal-wrong | GT=Lane share of the class | enrichment |
|---|---:|---:|---:|
| **ng5 shadow, DALI** | **65.52%** (6,872/10,488) | 32.61% | **54.86×** |
| cold shadow, DALI | 64.79% (7,350/11,344) | 30.61% | 51.50× |
| ng5 shadow, PyAV | 66.81% | 33.26% | 55.99× |
| ng5 live, DALI | 71.71% | 50.66% | 85.24× |
| cold live, DALI | 71.53% | 46.53% | 78.28× |

The prediction "the Lane-touching share stays ≥ 60%" **HOLDS**, and understates it: the burn default
makes the residue **more** Lane, not less. Every site the τ band removes from the persistent set is
disproportionately non-Lane.

## 6. The birth is still sixteen updates

Over-paint birth — first swept step at or above 1.05× the class's HT GT area:

| cell / forward | Lane | Movable | Road / Undrivable / MyCar |
|---|---:|---:|---|
| **ng5, live** | **step 16** | **step 16** | never |
| cold, live | step 16 | step 16 | never |
| **ng5, shadow** | step 48 | **never** | never |
| cold, shadow | step 32 | step 1,920 | never |

**The band does not move the birth; it shrinks it.** On the live forward the crossing is at step 16
for both classes on both cells. What changes is magnitude and persistence: the live `d_seg_hat` peak
is **0.0050661723 at step 16 = 1.9824×** the shared start, against the cold control's
**0.0060334524 = 2.3612×** (−16.0%); 18,974 sites are born wrong at the peak against 24,336
(−22.0%), and **87.05%** of them recover by the terminal against 82.76%. On the shadow the Movable
over-paint never crosses 1.05× at all, where the cold control crossed it at step 1,920.

The `NEW_PERSISTENT` sites — the error the run creates and keeps — are NOT a step-16 population on
either cell: median first-wrong step 2,304 (ng5 shadow) and 896 (cold shadow), with only 3.8% and
6.2% first wrong at or before step 16. md1's "damage born in sixteen updates" is a statement about
the EXCURSION, and it survives here; it was never a statement about `NEW_PERSISTENT`.

Shadow excursion, both cells: ng5 peaks at **0.0027456919 at step 640** (1.0744× start) and ends
0.0025300344 (**0.9900×** start); the cold control peaks at 0.0032910665 at step 2,304 (1.2878×) and
ends 0.0028065999 (1.0982×).

## 7. Apparatus honesty, and what the shared initialisation forbids

The burn ran on Metal; this reconstruction runs on CPU. At the four milestones that are multiples of
the 16-step checkpoint period:

| milestone | `d_seg_hat` CPU (shadow, PyAV) | ng5's retained MPS milestone | relative gap |
|---:|---|---|---:|
| 0 | 0.002519353231 | 0.002518335978 | **+0.0404%** |
| 2,000 | 0.002664438883 | 0.002664311727 | +0.0048% |
| 4,000 | 0.002554957072 | 0.002556737264 | −0.0696% |
| 5,000 | 0.002487818400 | 0.002486928304 | **+0.0358%** |

Same magnitude as md1 measured on the cold control (+0.0404% / +0.0784%), and it does not grow
across the run. **Every class table above is computed entirely inside the CPU series**, so the
trajectory classes carry no MPS/CPU contamination; only the comparison to the recorded milestone
carries that residual, and it carries it explicitly.

**The limit that binds this arm's conclusion.** All three retained `wc3` seed controls
(`seed_20260902`, `seed_20260903`, `seed_20260904`) declare the **same** `initial_state.sha256`
`991a1cc653c786af…` — MEASURED from the three sealed configs. A "second seed" therefore varies the
data order and the stochastic draws, **not the initialisation**. It cannot test whether the
persistent pool is a property of the initialisation. That is a correction to md1's owed item #5,
which named a second seed as the thing that "turns the persistent share into a seed-independent
reading": it does that for the SHARE, and it cannot do it for the SITES.

## 8. What the verdict implies, priced

**The verdict STANDS, so the next step is not another schedule.** md2 narrows where a new mechanism
must act. The unreachable set is 11,019 sites — **0.175%** of the frame — drawn from a 16,553-site
pool that is fixed at step 0; it is 65.5% Lane-touching with Lane enriched 54.9×; and 5,000 updates
of the best schedule in the series repair 5.005% of its weighted error, 0.77 pp more than the cold
control did.

**The one next step: measure the partition on a cell with a DIFFERENT INITIALISATION.** That is the
cheapest experiment that can move the pool, and therefore the only cheap one that could still
re-grade the closure. It is the first measurement in this family that is not free, because no
retained artifact has a different init.

*Price (DERIVED from measured components):* a fresh QBF1 init (or an r10 re-derivation) + one
5,000-update burn on the Metal at ng5's own measured wall clock **10,055 s ≈ 2.8 h**, then the
partition at md2's measured **3,235 s ≈ 54 min of CPU at 4 threads, $0**, plus ~35 MB of payloads.
Both tools already exist and are committed; no new code. Total ≈ **3.7 h, one Metal slot, $0 spend.**
Falsifier to pre-register: if the step-0 wrong pool of the new init overlaps the old pool at
J > 0.6, the pool is a property of the generator family and the born route's accuracy half is closed
for the family, not just for this object; if J < 0.3 and the new cell's persistent share falls below
40%, the closure is re-graded and initialisation is a live, un-priced lever.

*Prerequisite that IS free and removes a confound first (~54 min CPU, $0, 0 new code):* run the same
instrument + `experiments/ddm_md2_persistent_site_overlap.py` on `seed_20260903`'s control — its 313
checkpoints and sealed config already exist. With the init held fixed it isolates DATA ORDER, and it
turns the persistent share into a three-schedule × two-order reading. It cannot answer the init
question; naming it as if it could would be the substitution this memo just corrected.

## 9. Equations leg (`tac.canonical_equations`)

**Consumed, unchanged:** `checkpoint_trajectory_error_partition_v1`
(`src/tac/canonical_equations/checkpoint_trajectory_error_partition_20260904.py`, registered by
md1). This arm is the reactivation that anchor's own `reactivation_criteria` names — *"a second cell
measured at the SAME cadence … turns the persistent share into a family reading"* — and it is
measured at the same cadence, so the law's `domain_of_validity.excluded` clause forbidding transfer
of the persistent SHARE across cadences is respected rather than tested. A **third empirical anchor**
is appended: `md2_ng5_tau_band_x_continuous_objective_seed_20260902_shadow_trajectory_partition_20260905`,
carrying both forwards (shadow 62.954%, live 35.439%), the integer calibration gate (residual 0), the
floor 0.0015927633 = 11.671× the corner, and the site-overlap block (J 0.8069, within-pool chance J
0.5263, attainable max 0.9305). Its `known_boundary` records the new limit this arm measured: three
cells, three schedules, **one initialisation** — the persistent SHARE now has three instances
agreeing within 4.0 pp on the shadow (62.011% cold / 59.009% warm / 62.954% ng5) and within 0.45 pp
on the live forward, and the persistent SITES are 80.7% Jaccard-shared, but every instance shares
init sha `991a1cc6…`.

**Also consumed, unchanged:** `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — the per-site
scalar this partition is built on; no refinement claimed.

**Not registered:** the Jaccard-against-the-within-pool-null construction. It is one arm's
measurement on one shared initialisation; a law would need the init varied, which §8 prices.

## 10. Custody

`$0`. CPU only, `--threads 4`, one process, launched through
`tools/launch_detached_process.py --done-receipt md2_partition`; **0 Metal, 0 Modal, 0 contest eval**
(the sweep receipt records all three as 0); nothing written under any cell's `runs/`; `upstream/`
untouched; no `/tmp` path in any artifact. Peak RSS declared from md1's own safe_run receipt
(`launch/resource_safe_run_status.json`, 15,679.516 MiB = **15.31 GiB**); md2's realized peak
**15.367 GiB** — the declaration was accurate. 141 forwards, **3,235.4 s**.

Store `/Volumes/APDataStore/pact/ddm_md2_persistent_under_burn_default/`:

| artifact | what it holds |
|---|---|
| `payloads/ng5_tau_band_seed_20260902/<forward>_step_<n>.npz` | the 32-pair argmax (uint8) and δ_R-band code for all 141 swept forwards — the payload every table here derives from |
| `sweep_rows_ng5_tau_band_seed_20260902.jsonl` | one row per (step, forward): both lineages' `d_seg_hat`, per-pair rows, per-class predicted area, margin histogram, band counts, per-role weight/displacement norms, AdamW moment norms, checkpoint + payload sha256 |
| `ANALYSIS_ng5_tau_band_seed_20260902_{dali,pyav}.json` | the integer bridge, class tables, terminal edges and bands, excursion and reachability blocks |
| `site_classes_…_{shadow,live}_{dali,pyav}.npz` | the per-site class code (32, 384, 512) |
| `excursion_…_{shadow,live}_{dali,pyav}.npz` | the per-site trajectory code `at_zero + 2·at_peak + 4·terminal` |
| `OVERLAP_ng5…_vs_cold_control…_{dali,pyav}.json` | the site-overlap block, Lane rows for both cells, over-paint birth, `d_seg` trajectories, NEW_PERSISTENT first-wrong steps |
| `JACCARD_NULL_MODELS.json` | §4's honest null: the step-0 pool, its identity across cells, within-pool chance J, attainable max J |
| `COLD_COMPARATOR_extract.json` | md1's cold rows as read from md1's store — the comparator, extracted not retyped |
| `control/OVERLAP_warm…_vs_cold….json` | the four positive controls that validated the tool before it was pointed at ng5 |
| `md2_driver.sh`, `launch/` | the exact recipe and the launch manifest + safe_run receipt |
| `SWEEP_RECEIPT_ng5_tau_band_seed_20260902.json` | 141 forwards, 3,235.4 s, peak 15.367 GiB, 0 Metal / 0 Modal / 0 contest eval |

Instrument `experiments/ddm_md1_micro_to_macro.py` **unmodified**; the new tool is
`experiments/ddm_md2_persistent_site_overlap.py` (commit `1c2f35d2f`, two review-gate passes,
ruff clean). md1's `--mode compare` and `--mode report` are hardcoded to the cold and warm cell
names, which is why the overlap is a separate tool rather than a flag — and md1's `compare` answers
a different question (BORN-site overlap), not this one.

## NEXT_IF_RESUMED

1. **§8's priced step: a different initialisation.** Everything else in this family is a schedule,
   and three schedules now agree.
2. **The free confound-removal in §8** (`seed_20260903`, ~54 min CPU) — worth doing first, and
   worth NOT over-claiming when it lands.
3. **The 11,019 sites are now an enumerated, persisted object**
   (`site_classes_ng5_tau_band_seed_20260902_shadow_dali.npz`, class code 2). Any arm that wants to
   price an explicit representation of the residue can read them directly instead of re-deriving
   the set.

`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`
