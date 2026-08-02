# ddm_ti1 — CONTINUED TEACHER ITERATION: a NON-REDUNDANT teacher exists, it is already BUILT, and it is built on the wrong trainer (2026-08-02)

**Arm ddm_ti1, operator-originated 2026-08-02** ("Also pursue the continued teacher iteration and
optimization"). Pointer honesty FIRST: submittable **0.1910828242 [contest-CPU] UNMOVED**; own-vehicle
frontier **v4d 0.9639878 UNMOVED**. Every number below is **[macOS-CPU advisory]**, `score_claim=false`,
`research_only=true`. **Scorer slot: NOT REQUESTED, NOT USED — zero SegNet/PoseNet forwards.** This unit
is MEANS, not end.

---

## ANSWER (lead)

`ddm_lr1` closed *"rebase THIS teacher"* at FAMILY scope and left open *"is there a teacher carrying
information GT does not."* **There is. It is the cross-pair GT temporal-instability field, and it passes
the admission test that lr1's teacher failed.** MEASURED at n600 (598 interior pairs, 117,571,584 px,
zero scorer forwards), with a calibrated instrument, five controls and two independent cross-checks.

**And the finding that actually matters is not the field — it is that the lever is already SHIPPED CODE.**
`--seg-spike-reweight` / `--seg-spike-downweight` / `--seg-coherent-upweight` (#274, commit `6e355170d`)
exist in `experiments/train_levelset_witness_realized_through_R_mlx.py:9467-9494`, with a DSL `Lever`
factory, a gauge default, and a ledger row recording `"ever_fired": false`. Its producer computes the
**exact** two sets I measured (`sp = dp & dn` "unfittable flicker", `coh = (dp|dn) & ~sp` "winnable
boundary"). It has **no counterpart in the live TR1 trainer** (`grep spike|flicker
experiments/train_tr1_partition_renderer_mlx.py` → 1 hit, an unrelated histogram comment at `:1917`).

So this is the `built_elsewhere_unwired_is_p0` + `designed_stub_is_orphan_signal` class, and the thing
that was missing was never the code — it was the evidence that would justify firing it. **That evidence is
what this unit produced**, and it also prices the lever's two scalars *asymmetrically*, which nobody had
measured.

**I claim no ΔS.** Nothing here moved the pointer and nothing here predicts that it will.

---

## §1 STORES CONSULTED (recall receipts; path + line where load-bearing)

- `.omx/research/ddm_lr1_lattice_solve_rebase_refuted_20260802.md` — the closure I start FROM; its
  criterion (*a teacher pays only to the extent it is NOT redundant with what the loss already sees*) is
  the thing I generalize; its §10 NEXT-IF-RESUMED (2) names the admission-gate discipline I follow.
- `.omx/research/ddm_dw1_qa75_distill_window_20260730.md` — the distill-window negative; its §2 8-guard
  window discipline and its §5 preregistered noise floor (2.99e-5) are what my proposed next measurement
  reuses verbatim.
- `.omx/research/ddm_pj1_projection_probe_20260730.md` (photometric CONFOUND, cross-vehicle range wall) ·
  `.omx/research/ddm_fp1_class_field_projection_20260731.md` (f′ receiver floor 0.008305).
- `.omx/research/ddm_fl1_perclass_flicker_floors_20260731.md` — the per-class GT-flicker floors; its
  625,297 spike-px total over 598 interior pairs is what my instrument reproduces EXACTLY (§4).
- `.omx/research/ddm_ru1_recursive_upstream_endpoint_typing_20260729.md` +
  `/Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz` (sha `facc82539d2017e8…`) — the ep399
  realized-flip atlas; `gt_flicker_receipt.json` records ru1's flicker definition as
  `lstars[i] != lstars[i+1]` (ADJACENT change, 1.2456%), which is **not** fl1's spike (0.005318) — I
  re-derive both rather than reuse either flag.
- `.omx/research/ddm_rg5_rate_gradient_sign_20260801.md` §5 — the rate-surrogate orthogonality record.
- `.omx/research/ddm_er1_realized_trip_in_the_describe_objective_20260802.md` §0 — the margin-vs-label
  record.
- `.omx/research/default_off_decision_table_20260710.jsonl:111` — `--seg-spike-reweight`,
  `"ever_fired": false`, `"ev": {"label": "UNMEASURED"}`.
- `experiments/train_tr1_partition_renderer_mlx.py` · `experiments/train_witness_realized_through_R_mlx.py`
  · `experiments/train_levelset_witness_realized_through_R_mlx.py` · `src/tac/optimization/lane_guard.py`
  · `/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_03/tr1_config.json` (the LIVE config).

**Prior-art sweep (denominators named).** A read-only sweep covered `.omx/research/**` (36,558 files;
6,951 depth-1 `*.md`), `.omx/state/**` (21,372), `src/**` (22,642), `tools/**/*.py` (2,077),
`experiments/*.py` (818 top-level; `experiments/results/**` deliberately SKIPPED as vendored intake),
`docs/**` (183), `CLAUDE.md`, `AGENTS.md`. **This is a scoped statement, not a claim that nothing exists
anywhere.** What it found is in §6; the load-bearing hit (#274) I then verified myself at file:line.

---

## §2 THE DERIVATION — where the non-redundancy frontier actually is

lr1's criterion is right but its null model is one field (GT margin) and one statistic (raw correlation).
The correct null is broader and it is **structural**:

**The live TR1 per-pixel seg loss is per-pair separable.** For pair `t` it reads `lstars[t]`,
`margins[t]`, and the student's own live logits — nothing from any other pair. Therefore every per-pixel
weight it can express from its GT inputs is a **measurable function of (class, GT margin)**. The right
null model is not "correlated with GT margin"; it is **"measurable w.r.t. σ(class, GT margin)"**.

Two consequences fall out immediately:

1. **Any single-pair candidate is at structural risk of redundancy.** This is why lr1's teacher died: it
   was a per-pair function of GT, so it could only ever be a (noisier) reparametrization of a field the
   loss already had.
2. **A CROSS-PAIR candidate is outside that σ-algebra by construction.** A field computed from
   `lstars[t-1]` and `lstars[t+1]` is not a function of anything the per-pair loss sees. Non-redundancy is
   then guaranteed a priori; what remains to be measured is whether it is *worth* anything.

That is the whole design: **project onto σ(class, GT margin) nonparametrically, then test the residual
against the student's realized error.** Non-redundancy alone is not sufficient — lr1's residual was
non-zero (16.6% of σ) and worthless because it was the solve's own realization *noise*. So the second leg
must ask whether the residual predicts error.

**The conservative direction matters here, and it is in my favour.** On the LIVE config
(`window_03/tr1_config.json`) the loss's actual use of GT margin is *narrower* than my null model: the
only live path is `lane_guard.pixel_weight_addend`, and that term is **Lane-restricted** —
`src/tac/optimization/lane_guard.py:877` multiplies the margin-floor deficit by `is_lane`. `hinge` is
hard-wired to `0.0` at the TR1 call site (`train_tr1_partition_renderer_mlx.py:1873`, third positional),
`focal_gamma` and `fisher_density_weight` are never passed, and the live `tau_softplus` branch
(`train_witness_realized_through_R_mlx.py:1454-1459`) does not reference `margin` at all. So I projected
out **more** information than the loss actually uses, which makes the measured non-redundancy a LOWER
bound.

---

## §3 THE MEASUREMENT (n600, 598 interior pairs, 117,571,584 px, ZERO scorer forwards)

Producer: `tools/ddm_ti1_teacher_nonredundancy_probe.py` (landed this unit).
Receipts (durable, committed): `.omx/research/ddm_ti1_nonredundancy_probe_receipt_bins{10,40}_20260802.json`.
Inputs: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` + the ru1 ep399 flip atlas
(sha `facc82539d2017e88c61ca74da25cae7f42196f518a38a9bd596d5ca830fa948`).

`residual_var_frac` = fraction of the candidate's variance NOT expressible as a function of
(class, margin-bin). `MH_RR` = Mantel-Haenszel pooled flip risk-ratio, hi vs lo, **within** each
(class, margin-bin) stratum. `ctrl` = the same statistic with the field taken from a different pair.

| candidate | residual_var_frac (10 / 40 bins) | crude RR | **MH_RR (10 / 40)** | ctrl MH_RR (10 / 40) | usable strata (40) |
|---|---:|---:|---:|---:|---:|
| **spike** (#274 `sp`) | 0.8153 / 0.7946 | 76.83 | **2.224 / 1.757** | 1.102 / 1.023 | 19/200 |
| **coherent** (#274 `coh`) | 0.8471 / 0.7679 | 47.72 | **2.128 / 1.299** | 1.153 / 1.004 | 34/200 |
| change (ru1 defn) | 0.7798 / 0.7228 | 77.81 | 2.723 / 1.737 | 1.128 / 1.013 | 29/200 |
| flickcount ≥ 1 (`sp ∪ coh`) | 0.6894 / 0.6075 | 116.66 | **4.230 / 2.071** | 1.150 / 1.010 | 37/200 |

**Every candidate's control FIRED at both bin counts.**

Prevalence and raw risk:

| set | pixels | share of px | flips | P(flip \| set) |
|---|---:|---:|---:|---:|
| spike | 625,297 | 0.53% | 133,124 | 0.2129 |
| coherent | 1,678,532 | 1.43% | 186,833 | 0.1113 |
| unstable (union) | 2,303,829 | **1.96%** | **319,957** | 0.1389 |
| stable | 115,267,755 | 98.04% | 137,225 | 0.00119 |

**70.0% of the endpoint's entire realized error mass sits on 1.96% of pixels.**

### §3a The two #274 knobs are NOT symmetric — this is new

At the finest conditioning (40 bins), `spike` carries **1.757** conditional lift while `coherent` carries
**1.299**. The spike set is both rarer and conditionally more error-dense (P(flip) 0.213 vs 0.111). The
lever's two scalars therefore do not deserve equal magnitudes, and nothing in the existing #274 record
prices them apart — `gauge.py` carries a single `SEG_SPIKE_DOWNWEIGHT_DEFAULT = 0.25` and the flags
default to a symmetric inert 1.0/1.0.

---

## §4 INSTRUMENT VALIDATION (no vacuous passes, and the positive is guarded)

A "there is signal everywhere" readout is exactly what a broken lift estimator prints, so the positive is
guarded from both sides:

| control | expectation | MEASURED | verdict |
|---|---|---|---|
| **spike count vs `ddm_fl1`, independent implementation** | must reproduce 625,297 | **625,297 — exact** | **CROSS-VALIDATED** |
| **atlas total vs the registered burn seg** | 457,182/117,571,584 should ≈ 0.0038892 | **0.00388854** (Δ = the 2 excluded endpoint pairs) | **CROSS-VALIDATED** |
| `margin_bin_CALIBRATION_exact` (stratum-measurable by construction) | residual must be EXACTLY 0 | **0.000000 / 0.000000**; 0 usable strata | **PROJECTION EXACT** |
| `hashnoise_CALIBRATION_null` (non-redundant, zero information, matched sparsity) | MH_RR must be ~1 | residual **1.000000 / 0.999998**; MH **1.0145 / 1.0227** on 33 / 122 usable strata | **ESTIMATOR DOES NOT MANUFACTURE LIFT** |
| `flip_CALIBRATION_oracle` | MH must be enormous | inf (zero flips on the low side) | **LIFT LEG SEES SIGNAL** |
| `margin_CALIBRATION_redundant` | residual must shrink as conditioning refines | 0.0813 → 0.0093 (10 → 40 bins) | **BEHAVES** |
| shuffled-pair control, every candidate | must collapse toward 1 | 1.10–1.15 → **1.004–1.023** | **FIRES** |
| guard controls (zero pairs / zero flips / degenerate stride / constant field) | must REFUSE | all raise | **REFUSE** |

The `hashnoise` null is the one the whole positive rests on: a field that is 100% non-redundant and
carries no information about student error returns **1.02** on 122 usable strata, while the candidates
return **1.30–2.07** on 19–37. The separation is not something the pooling produced.

**Bin-count sensitivity (mandatory for a positive, per the probe's own documented resolution limit).**
The lift ATTENUATES with finer conditioning (spike 2.22→1.76, flickcount 4.23→2.07) — as it must, since
finer strata absorb more of the margin correlation. **The honest reading is that the true conditional
lift is at or below the 40-bin value.** What matters is that it does not converge toward the null: at 40
bins the candidates sit at 1.30–2.07 while both the noise null and the shuffled control sit at 1.00–1.02.

---

## §5 DIRECTION — which way the lever should be turned (MEASURED, with its limit stated)

A non-redundant field is useless until you know the sign. The atlas answers it scorer-free, because it
carries `realized_class` per flip.

**On the 319,957 flips at temporally-unstable pixels, the student's wrong label equals a temporal
NEIGHBOUR's GT label 98.17% of the time** (pair-shuffled control 67.16%; control fires). Per class:
Road 0.9754 · Lane 0.9901 · Undrivable 0.9660 · Movable 0.9789 · MyCar 0.9743. Lane alone holds 151,189
of the 319,957 (47.3%), consistent with fl1's "Lane binds hardest (13.1× corner-C)".

**Reading:** where GT is temporally unstable, the student behaves as a temporally-SMOOTH witness — it
emits the neighbour-frame label and GT spikes away from it. This is a phase-faithfulness debt, not
random error.

**Limit, stated plainly (this is the weakest number in the memo).** At a locally-binary boundary — two
classes competing — "wrong" and "the neighbour's label" nearly coincide, so a large part of 98.17% is
tautological. The shuffled-pair control is the guard and it sits at 67.16%, so **the defensible statement
is the 31-point excess over a same-pixel, different-pair null**, not the raw 98%. The near-uniformity
across all five classes is itself evidence that the locally-binary component is large. A sharper follow-up
(not run): restrict to true spikes where `prev == nxt ≠ cur` and ask whether the realized label equals the
temporal-MAJORITY label specifically.

**Consequence for the lever:** the measurement supports BOTH of #274's knobs being live and does not by
itself pick a sign — down-weighting spikes concedes an error class the student is structurally producing,
up-weighting coherent-unstable pixels attacks the winnable half. It does say the two are asymmetric
(§3a), and it says the direction question is a two-arm race, not a guess.

---

## §6 THE OTHER VEINS — verdicts (so the charter's list is closed, not left open)

**(a) Margin-optimality rather than LABEL (#888) — DERIVED-REDUNDANT for TR1.**
The live TR1 seg form IS already a margin objective: `tau_softplus`
(`train_witness_realized_through_R_mlx.py:1454-1459`) minimises `τ·softplus(−m/τ)` on `_live_signed()`,
built entirely from the student's own logits (`:1447-1452`). A margin target is a per-pair function of the
student, so by §2 it is inside the loss's information set — and here it is not merely inside it, it *is*
the loss. The #888 finding (`ddm_er1…:35-47`) is a **describe-path** issue (`direct_description_joint_descent.py:2359`
optimises a CE surrogate and discards the realized argmax at `:2419`) and is owned there.
**verdict_scope: this closes the margin-vs-label vein FOR THE TR1 VEHICLE ONLY**; the describe path is
untouched and its own deliverable stands.

**(b) Rate-side teacher — NOT FOUND as a training target in the named scope; the live record is a
different object.** The sweep (denominators in §1) found no byte-cost oracle used as a training target;
every `rate oracle` / `byte oracle` hit is an offline receiver-side object, and
`canonical_equations/jrd_exact_coefficient_prefix_selection_20260712.py:4` says so explicitly
(*"an offline receiver/rate oracle. It does not authorize a trainer…"*). The adjacent live finding is
`ddm_rg5…§5`: `cos(entropy, smevr_surrogate) ≈ 0` (−0.066 … +0.020) across four fields — that is a
**rate-SURROGATE choice**, not a teacher, and its indicated action (build the `entropy + smevr_surrogate`
SUM arm, which `--rate-model` cannot currently express — `train_tr1_partition_renderer_mlx.py:1439` is a
2-choice enum over mutually exclusive branches) is a trainer edit owned by that lineage. **Recorded, not
claimed, not mine.**

**(c) Reachable-target teacher — already BUILT, plus a second unwired producer.**
`--margin-saliency-reachability` (LEVER-4) has a producer (`tools/precompute_sR_reachability.py`) and a
live loss consumer (`train_levelset_witness_realized_through_R_mlx.py:8760-8767`). Separately,
`ddm_sn1`'s per-pixel `STRUCTURALLY_HARD_IRREDUCIBLE` classification at n600 (635,011 px = 28.03%,
`.omx/research/ddm_sn1_error_source_tensor_n600_20260723/error_source_budget.md:13-19`) is consumed by
**no training code** — a second difficulty field with a producer and no consumer, i.e. the same P0 class
as #274. Named, not pursued here.

**(d) Self/born-again teacher (gc10 row 6) and cross-class KD (dw1 row 5) — still UNEXECUTED**, no run
dirs. Note a citation correction inherited from lr1: in the dw1 reformulation queue the born-again
self-teacher is row **(3)**, and row **(5)** is cross-class KD.

---

## §7 WHAT WOULD SHOW THIS LOWERS THE EXACT SCORE, AND WHAT WOULD FALSIFY IT (preregistered)

**The build is a PRODUCER, not new machinery.** The consumer already exists in TR1 and is already used:
`seg_pixel_w` is constructed at `train_tr1_partition_renderer_mlx.py:1850-1866` from `class_weight_lane`
and `lane_guard`, and multiplies the per-pixel seg map before the mean in every seg form. Porting #274's
producer (`train_levelset_witness_realized_through_R_mlx.py:9469-9493`) into that construction is an
additive, default-inert edit — 1.0/1.0 scalars ⇒ map ≡ 1.0 ⇒ byte-identical, which is exactly how the
levelset copy is already gated. **Do not build a new lever; move the one that exists onto the live
vehicle.** (This is the `built_new_machinery_instead_of_paying_identified_debt` guard applied to my own
proposal.)

**The measurement:** three matched governed windows from the LIVE endpoint
(`ddm_b4s_20260731/window_03/checkpoints/intra_seg_trunk_tau_ep00854.npz`), reusing dw1's §2 8-guard
discipline verbatim (matched config, argv-diff assertion, in-window noise floor stated before the split,
joint seg+rate, s/ep matched):
- **B** = control, plain continuation (the strongest honest baseline — dw1 measured that plain
  continuation still pays at this lineage's endpoints).
- **A** = `--seg-spike-reweight --seg-spike-downweight 0.25 --seg-coherent-upweight 1.0` (the gauge's
  existing DERIVED default; concede the spike set).
- **C** = `--seg-spike-reweight --seg-spike-downweight 1.0 --seg-coherent-upweight <derived>` (attack the
  coherent half). Per §3a the two scalars must be raced separately, not moved together.

Endpoint realized n600 d_seg (EMA) + counted bytes reported separately.

**FALSIFIER (preregistered):** if neither A nor C beats B's endpoint realized n600 d_seg by more than
B's OWN in-window gate-residual std (dw1's prior magnitude at this lineage was 2.99e-5 — cited as a scale,
NOT transferred as a constant), then the temporal-instability teacher is **FORMULATION-dead as a
per-pixel seg reweight at this endpoint**, and the vein's remaining live form is a representational one
(a carrier that resolves temporal phase) rather than a loss weight. A seg win bought by spending token
entropy is not a win: bytes must be matched or the win reported jointly.

**What I am NOT claiming:** no ΔS, no predicted band. The 0.2721 S of error mass sitting on the unstable
set (319,957/117,571,584 × 100) is a **mass**, not a recoverable quantity — the teacher does not promise
to remove any of it, and the fact that it is 70.0% of the ep399 seg residual (0.27214 of 0.38885 S) is a
statement about *where* the error is, not about what a reweight will do to it.

---

## §8 HONESTY LABELS + verdict_scope

- §3 all numbers, §4 all controls, §5 match rates: **MEASURED** (n600, cached frozen-authority GT +
  ep399 realized-flip atlas; zero scorer forwards).
- §2 per-pair separability and the σ-algebra argument, §3a asymmetry consequence, §6(a): **DERIVED**
  (from code read at the file:line cited).
- §5 mechanism reading ("behaves as a temporally-smooth witness"): **INFERRED**, and explicitly discounted
  by the locally-binary caveat.
- §7 falsifier: **PREREGISTERED, unmeasured.**

**VERDICT: the temporal-instability teacher PASSES the admission test** — 61–85% non-redundant with
σ(class, GT margin) and carrying a stratified flip-risk lift of 1.30–2.07 against a measured null of 1.02.
It is the first candidate teacher to pass since lr1 set the criterion.

- **verdict_scope of the POSITIVE: INSTANCE** — one endpoint (ep399 tb1, the ru1 atlas), one vehicle
  lineage (TR1), one error set. The non-redundancy half is structural and holds at any endpoint (it
  compares a GT-derived field to a GT-derived σ-algebra, neither of which moves); **the predictiveness
  half is endpoint-bound and must be re-measured at ep854** before any window is sealed on it. Per
  `ddm_fs1` staleness this is a fit against a partner that has moved (ep399 → ep854, a 23%+ descent).
- **verdict_scope of §6(a) NEGATIVE: FORMULATION × VEHICLE** — margin-as-target is redundant on TR1
  because TR1's live loss already is that objective; it says nothing about the describe path.
- **NOT closed by this unit:** whether the lift survives at ep854; the sign of the lever; whether any
  reweight converts conditional error concentration into realized d_seg; the self/born-again and
  cross-class KD teachers; the rate-surrogate SUM arm.

---

## §9 CUSTODY (durable paths + shas; no `/tmp`)

- Probe (landed, this unit): `tools/ddm_ti1_teacher_nonredundancy_probe.py`.
- Receipts (durable, COMMITTED — the instrument and its output travel together):
  `.omx/research/ddm_ti1_nonredundancy_probe_receipt_bins10_20260802.json` +
  `…_bins40_20260802.json` (schema `ddm_ti1_teacher_nonredundancy_probe.v1`, `scope=FULL`,
  598 interior pairs / 117,571,584 px).
- Run dir (logs + duplicate receipts): `/Volumes/VertigoDataTier/pact/ddm_ti1_20260802/`.
- READ-ONLY inputs, untouched: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`;
  `/Volumes/VertigoDataTier/pact/ddm_ru1_20260729/atlas_flat.npz`
  (sha `facc82539d2017e88c61ca74da25cae7f42196f518a38a9bd596d5ca830fa948`).
- **No artifact was created, moved, or deleted outside my own run dir; no run dir belonging to another
  arm was touched; no trainer was modified.**

**Observed apparatus debts (reported, not chased — not mine to fix):**
1. `ddm_fl1`'s driver `ddm_fl1_perclass_flicker.py` lives ONLY on the external volume
   (`/Volumes/VertigoDataTier/pact/ddm_fl1_20260731/`), not in the repo — the PERSIST-THE-INSTRUMENT debt.
   I re-derived its number independently rather than cite it, and my 625,297 matches exactly.
2. `train_tr1_partition_renderer_mlx.py:983` (`_basin_window_valid`) tests `x["stage"] ==
   "seg_trunk_tau"` by exact string. A run launched with `--seg-form-start tau_softplus` gets stage
   `"seg_trunk_tau_softplus"` from `:1776` and never passes through the knee at `:2137-2139`, so it runs
   the identical loss form and can never satisfy the basin predicate. Same form, different string.
3. On the live `window_03` config `margin_weighted_loss='on'` is set, but the `tau_softplus` branch
   applies no `apply_mw` (`train_witness_realized_through_R_mlx.py:1454-1459`) — the flag is inert for the
   stage the endpoint actually ran. Flagged for whoever owns the TR1 curriculum; it did not affect any
   number here (my null model already assumes the loss CAN see GT margin, which is the conservative side).

Pointer delta: **UNMOVED** (submittable 0.1910828242 [contest-CPU]; own-vehicle v4d 0.9639878).
