---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Mallat, Tishby, Tao, Boyd, Hinton, MacKay, Ballé, Hotz, PR95Author, Time-Traveler]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "Half this audit's 'cargo' rows were already certified with open questions attached (n600_final_config_cert Q1 literally asks whether 300/726 should be a plateau-detector). The derivation's value is real but the framing 'we never examined the schedule' overstates — we examined it piecewise and never UNIFIED it. Say that honestly, and do not let a beautiful control theory delay run-3 behind three BUILDs when the recalibrated event-trigger flags already exist."
  - member: Hotz
    verbatim: "The unified-energy θ* controller is the third beautiful thing this week. Run-3 needs exactly two changes that are measured: fire tau on a real plateau test instead of ep300, and stop three schedules from ending on the same epoch. Everything else is polish. Ship the two, bank the rest."
  - member: PR95Author
    verbatim: "For the record: my 8 stages were 29,650 epochs of hand-tuned HNeRV babysitting, not a theory. That it seeded your witness curriculum was a historical accident. The right inheritance from my schedule is the ORDER (form the partition, then sharpen, then polish) — the numbers were never meant to transfer."
council_assumption_adversary_verdict:
  - assumption: "The CE→tau boundary belongs at a fixed epoch (300) because the cert run's CE knee was ~ep280-300."
    classification: CARGO-CULTED
    rationale: "Cross-run transfer of a fixed epoch is the ancestor-numbers-don't-transfer violation one level up: run-2's CE trajectory (0.466→0.162→0.122→0.127 @ ep0/25/50/75, n600 verdict) is ~23× above the cert-arm trajectory at like epochs — its CE will NOT be plateaued at ep300. The boundary is a READINESS TRIGGER (plateau + per-class nucleus), not a wall-clock constant."
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
  - assumption: "The stage ORDER CE→tau→Muon is PR95 inheritance that needs replacing."
    classification: HARD-EARNED
    rationale: "The order was independently RE-derived by #284: CE=mirror-descent/NG encoding, tau=Maslov/Γ dequantization (MCF), Muon=weight-space spectral finisher outside the τ-continuum which measurably CANNOT nucleate a zero-mass class (facet-4 §2.1) — so nucleate-then-sharpen-then-polish is forced by the math. Keep the order; replace the NUMBERS with triggers."
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
  - assumption: "EMA 0.997 (Quantizr non-negotiable) is optimal for this vehicle's finisher."
    classification: CARGO-CULTED
    rationale: "0.997 = 333-step window (per-step ema.update verified, trainer:4759) = 4.4 epochs at 75 steps/ep — it averages only the last ~1.6% of a 274-ep finisher. The π-group window/stage-length is not preserved from Quantizr's/PR95's step budgets. Measured 78× early-stage shadow lag (cert C3) is the tracking half of the same violation. --ema-decay-finisher already EXISTS (θ* TIER-2 MUST-3, default None = bit-identical) — the fix is built, unfired, un-A/B'd."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "geometric τ-anneal shape is derived, so the schedule leg is done."
    classification: HARD-EARNED
    rationale: "facet-4 derived geometric = constant Fisher-Rao velocity = adiabatic dwell (CV≈0.39 flat info-per-octave, $0-confirmed) — but ONLY for the τ path. The hosc-β anneal (the ONLY interface sharpener now that render-τ holds at 1.0) is LINEAR (choices linear|cosine — geometric not even wired), and β is the activation's own dequantization scalar. The derivation transfers; the code does not yet."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "The rewarmup 20ep/cosine/floor-0.1 values are arbitrary PR95-flavored knobs."
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "20ep×75 steps/ep = 1500 steps ≥ 1/(1−β₂)=1000 = AdamW's second-moment re-accumulation memory — the window bound is DERIVED-satisfied (the cert's 8ep=600 steps was NOT). But cosine-vs-linear shape and floor 0.1 have no derivation or A/B; and whether the window law (≥β₂ memory) is the binding constant is unmeasured."
    empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE
council_decisions_recorded:
  - "op-routable #1 (SAFE-NOW): nothing fires on pid 39999; the ep300 watch list + ep726 Muon-restart observables below are the validation instrument for the derived hand-off law"
  - "op-routable #2 (RUN-3): adopt the derived-native schedule spec (§C.ii) — 2 flag-level changes ready now (recalibrated event-trigger + ep300 collision stagger), 4 BUILDs gated"
  - "op-routable #3 (θ*): unified-energy costate-controlled curriculum = design item under #218/#78/#247, NOT a hot patch"
  - "op-routable #4 (triality): CurriculumGauge landed (gauge.py APPEND); 4 laws registered via tools/register_lever_laws_curriculum_20260705.py; DAG FEED-05i appended"
---

# GRAND COUNCIL SYMPOSIUM — full-stack curriculum/schedule derivation (task #302)

**Convened 2026-07-05. Operator directive (verbatim intent):** *"We developed our level set witness
and solved the contest math, but still largely inherited our curriculum and schedule from PR95 —
different vehicle (HNeRV), implemented straight out of the paper. It accidentally worked; its
suboptimality inspired powerful levers; but we never until now did a comprehensive examination of
the FULL schedule and curriculum and full-stack synergy, bridging the arch/math with the optimal
training, all recursive fractal."*

**Containment honored:** live run pid 39999 (`levelset_n600_witness_20260705T015247Z`, ep75,
d_seg 0.127467 n600-verdict `[macOS-MLX research-signal]`) untouched; $0 study; every number below
is mined from EXISTING logs/memos or code-cited. **Pointer contest-CPU 0.19110 UNMOVED — this
entire symposium is MEANS.**

---

## A. THE INHERITANCE AUDIT (the ledger)

Classifications: **PR95-INH** (PR95-inherited) · **QTZ-INH** (Quantizr-inherited) · **MEAS**
(measured-ours) · **DER** (derived-ours) · **UNEX** (unexamined-default). Compound tags where honest.
Live config = run-2 `launch.sh` (recovered, in-run-dir). PR95 source = `profile_pr95_hnerv_muon_intake.md`
(8 stages, 29,650 ep: CE 3000 → tau_softplus 5650 → smooth 1500 → QAT 500 → l7 9000+2000+3000 → Muon 5000).

| # | element | run-2 value | class | evidence / provenance |
|---|---|---|---|---|
| 1 | stage SET {CE, tau_softplus, Muon} | 3 of PR95's 8 | **PR95-INH → MEAS-pruned** | smooth-stage RAISES d_seg (measured, CLAUDE.md lever-4); QAT n/a (no int8 export yet); l7 = measured DEFECT demoted to 1001 (L1). The surviving SET is PR95's spine minus measured harm. |
| 2 | stage ORDER CE→tau→Muon | — | **PR95-INH + DER-confirmed** | #284: CE=mirror-descent/NG, tau=Maslov/Γ-dequantization, Muon=finisher outside the τ-continuum; Muon CANNOT nucleate a zero-mass class (measured, facet-4 §2.1) ⟹ order is forced. Inherited, then independently re-derived correct. |
| 3 | **boundaries 300 / 726 / 1000 as FIXED EPOCHS** | tau@300, Muon@726, end 1000 | **CARGO (the worst row)** | cert A4: 300 = "just past the CE knee (temp 0.804)" — of the CERT arm. cert A7: 726 = "PR95 stage-8 placement" (verbatim) + proven ckpt. NOT proportional PR95 rescales (PR95 CE=10.1% of run, ours 30%; PR95 Muon@83%, ours 72.6%) — they are cross-run transfers of one trajectory's knees. Run-2's CE is ~23× above that trajectory at like epochs ⟹ ep300 will fire MID-DESCENT (prediction registered, §C.i). cert Q1 already asked this question; unanswered until now. |
| 4 | epoch budget 1000 | `--epochs 1000` | **MEAS-weak** | cert A2 "the proven TOTAL budget" — chosen from one prior arm + wall-clock, never π_train-checked (π_train = lr×steps×sharpness, Tao memo). |
| 5 | LR 1e-3 → 1e-4 cosine | `--lr/--lr-end` | **PR95-ECHO / UNEX** | 1e-3 = PR95 stage-1/2 AdamW value (= Adam paper default); 1e-4 = PR95 stage-3 drop. Our cosine is a smoothed echo of PR95's staircase (1e-3→1e-4→3e-5→1e-5). No witness-native derivation of either endpoint. |
| 6 | weight-decay 1e-4 | `--weight-decay` | **UNEX** | no memo/derivation found; common default. Note film-stiefel neutralizes WD's magnitude component on W (trainer comment, by design) — partial principled coverage, global value unexamined. |
| 7 | adam-β₂ 0.999 | `--adam-beta2` | **UNEX (deliberate, #222 pending)** | MLX default = bit-identical; the trainer's own help cites the DERIVED small-n optimum ~0.9999999 (1−β₂ ≲ (1−β₁⁵)/n^3.5, arXiv 2603.02092, n≈75 accum steps/ep) — derived alternative exists IN THE HELP TEXT, un-A/B'd. |
| 8 | EMA 0.997 + shadow-at-eval | `--ema-decay 0.997` | **QTZ-INH (π-violation flagged)** | Quantizr non-negotiable value. Cadence VERIFIED per-optimizer-step (trainer:4759) ⟹ window N=1/(1−ρ)=333 steps=4.4 ep. Measured 78× early shadow lag (cert C3). π_ema = window/stage-length not preserved across vehicles. `--ema-decay-finisher` (SWA-style wider finisher decay) already BUILT, default None. |
| 9 | accum 8 | `--accum-pairs 8` | **MEAS** | memory/throughput-driven (compute gate B=8; OOM discipline). |
| 10 | hosc β 1.0→5.134 linear | `--hosc-beta-*` | **DER endpoints / UNEX shape** | 5.134 DERIVED: β(ep726)=4.00 exactly under the Muon β-freeze (review M4; β=4 target from measured fixed-β divergence + annealed-hosc stability). Shape LINEAR unexamined — β is the activation's dequantization scalar; the τ=ε=ħ argument says GEOMETRIC (equal epochs per octave); `--hosc-beta-anneal` choices are linear|cosine only (geometric = BUILD). |
| 11 | render-τ path: constant 1.0, geometric shape | `--softmax-temp-* 1.0/1.0` | **DER** | π_τ=τ/h→1 Γ-optimal endpoint (Ch.4 + Tao π-ledger); geometric = Fisher-Rao geodesic/adiabatic (facet-4 §1.2, $0-confirmed CV≈0.39) — shape currently inert-exact at equal endpoints (review L2). |
| 12 | tau-softplus-tau 0.3 | `--tau-softplus-tau` | **MEAS-weak** | cert A5: proven value, low-priority A/B; dominant temperature lever is elsewhere. |
| 13 | eikonal ramp 0.05→0.10 | `--eikonal-weight(-end)` | **DER** | interface-width control (#286): survival knee σ0.8→94% vs σ1.5→49%; ramp holds π_int ≳ 1 as MCF narrows (facet-4 §2.2b). |
| 14 | length 0.001 (held small) | `--length-weight` | **DER** | length IS the MCF-erosion driver (V=−κ); keep small + per-class holds (facet-4 §2.2c — the exact inversion of "add smoothing"). |
| 15 | persistence warmup 300 | `--persistence-warmup-epochs` | **MEAS-coupled / wall-clock-anchored** | calibrated to tau@300; de-syncs if tau fires elsewhere (review M1 — class-level gap). |
| 16 | seed-anneal 300 cosine | `--seed-anneal-epochs 300` | **DER-timing / UNEX-shape** | crutch withdrawal by tau onset (derived); cosine shape unexamined. **NEW FINDING: run-2 still has a 3-way ep300 collision** — tau onset + persistence-warmup completion + seed-anneal completion all land on ep300 (only the band was deconflicted to 350). Ch.6's "one homotopy parameter at a time" was applied to 1 of 4 colliding schedules. |
| 17 | rewarmup 20ep cosine floor 0.1 + reset-moments | `--stage-transition-*` | **DER-window / UNEX-shape+floor** | reset-moments = measured stale-moment root cause (FEED-ft#3). Window: 20ep×75=1500 steps ≥ 1/(1−β₂)=1000 = the AdamW moment-memory bound (DERIVED-satisfied; the cert's earlier 8ep=600 was UNDER it). Cosine shape + floor 0.1: no derivation. Critical-slowing exponent (dwell ∝ 1/gap²) never matched — the 20ep cosine is a bound-satisfying guess, not the derived profile. |
| 18 | Muon: start 726 | `--muon-start-epoch` | **PR95-INH (verbatim)** | cert A7 cites "PR95 stage-8 placement". See row 3. |
| 19 | Muon: lr 2e-3 | `--muon-lr` | **MEAS** | NO-FAKE catch A8; 7.8× the frozen base-LR at ep726 — that ratio IS why Muon is the drop. |
| 20 | Muon: momentum 0.95, ns 5 | `--muon-momentum/--muon-ns-steps` | **UNEX-canonical / MEAS-literature** | canonical Muon defaults; ns k=5 SETTLED (NVIDIA 2606.00371: polar accuracy is not the lever; k3 vs k5 ≤0.15pp). |
| 21 | Muon: warm-start + lr-final-frac 0.1 | `--muon-warm-start-*` | **DER** | muon deep-dive GAP1+GAP2 (measured +0.000357 cold-switch spike; flat-LR-can't-self-reduce). ACTIVE in run-2's launch — #270's levers are live in this run. |
| 22 | lane-band start 350 | `--lane-band-start-epoch` | **DER** | Ch.6 L1 deconflict from tau@300 (measured collision bump 0.0056→0.020, 3.4×, 75+ep). |
| 23 | eval/ckpt cadence 25 | `--eval-every/--ckpt-every` | **UNEX → DER-confirmed here** | never derived; post-hoc check: closed-loop control lag = 1 eval = 25 ep ≪ ~100-ep erosion timescale (4× margin) — adequate by accident; now on the record as the control-cadence bound (Nyquist of the erosion mode). |
| 24 | w_seg 100 / w_pose 1.0 | `--w-seg/--w-pose` | **DER / UNEX** | 100 = the score's own d_seg coefficient (exact). w_pose=1 vs the score's √(10·d_pose) is NOT the score's local gradient (∂S/∂d_pose = 5/√(10·d_pose) → ∞ as d_pose→0) — a fixed linear weight under-weights pose exactly when pose gets good. Unexamined on this vehicle. |
| 25 | closed-loop scope | `--closed-loop-control` | **DER-partial** | READ (grep, not assumed): covers verdict-trajectory classification → bounded eikonal bump (≤2×0.05, cap 0.20) + early-stop arming. Does NOT cover: stage handoffs (separate `--curriculum-event-triggered`, dropped for run-2 per C1 miscalibration), Muon timing (not event-fireable — verified review "cross-lever hunt"), τ path, LR. |

**Headline (operator's centerpiece):** the schedule's *continuous* laws (τ path, eikonal ramp,
length, Muon tuning, band deconflict) are now largely DERIVED — the 2026-07-04 pass did that work.
What remains inherited/cargo is the *discrete event structure*: *WHEN things fire is still
wall-clock epochs transferred from other trajectories* (rows 3, 15, 16, 18), the *optimizer
constants* (rows 5–8, 20) are PR95/Quantizr/Adam defaults, and *run-2 still has a 3-way collision
at ep300* (row 16). The curriculum's remaining PR95-ness is its CLOCK, not its physics.

---

## B. THE DERIVATION — the witness-native curriculum from the ONE energy

The witness trains one continuously-annealed energy, rendered through R:

E_τ[φ] = ∫ data(softmax_τ(φ/τ), y*) + λ_eik(t)·(|∇φ|−1)² + ν·|∂{argmax}| + persistence/margin terms

Every "stage" is a discretization artifact of the continuation path (τ(t), β(t), λ_eik(t), η(t),
optimizer). The voices, each with the piece they own:

**B.1 Γ-convergence / Modica-Mortola (Daubechies + Mallat) — the τ(t) backbone.**
τ=ε=ħ (#284, PROVEN spine): the pointwise Maslov limit and the spatial Γ-limit coincide. The rate
law is facet-4's: constant Fisher-Rao velocity in the τ coordinate, g_ττ = Var_p(z)/τ⁴ ⟹ dwell
where flips are being decided (τ_c = 0.2421·m per pixel, $0-confirmed); a broadband margin spectrum
⟹ **geometric (log-spaced) τ descent, floored at π_τ = τ/h → 1**. Interface collapse is avoided iff
the co-anneal holds **π_int ≳ 1**: as the interface narrows, λ_eik must RISE (run-2's 0.05→0.10 ramp
= this law) while ν stays SMALL (ν is the MCF erosion driver). **NEW (this symposium): β_hosc is the
activation's own dequantization scalar and is now the ONLY interface sharpener (render-τ holds at
1.0) — the same law applies to it: β should ascend GEOMETRICALLY (equal epochs per octave of
sharpness), not linearly. Run-2's linear β is a derivation gap; geometric β = BUILD (choices are
linear|cosine).**

**B.2 Morse-Smale / persistence (Tao + Daubechies) — the HAND-OFF law (the study's core result).**
Coarse-to-fine = persistence order. The tau stage is sharp-limit MCF; Allen-Cahn's critical-nucleus
theorem says any class-region below critical size is ERASED, never grown (measured: #205 seeded
lane at part_frac 0 → d_seg CREPT 0.004752@300 → 0.006568@400; Muon cannot nucleate). Therefore
**CE→tau is admissible only when every scored class is ABOVE its critical nucleus** — π₁ = w/σ ≳ 5
(measured knee, resolution-portable) — AND the CE data term has plateaued (else you hand MCF a
half-formed partition to erode). **The boundary is a per-class READINESS TRIGGER, not an epoch:**

  fire(CE→tau) ⟺ [∀ scored class c: nucleus(c) satisfied (part_frac > 0 ∧ within-flip below
  threshold)] ∧ [CE plateau: |rel slope| ≤ 1e-4 over a 25-ep window, min-stage ≥ 250]

The plateau constants are the C1 recalibration (measured against #205's own CE trace: eps 1e-3
fires ep151 mid-descent = 15% CE-floor loss; 1e-4 separates ep275+ from ep150). The nucleus guard is
a BUILD (the existing event-trigger tests ep_loss only). This subsumes cert Q1.

**B.3 IB / Tishby annealing — which levers may engage when.** Changing the objective mid-anneal
invalidates the temperature schedule. Rule: lever engagements belong at stage boundaries (or in CE),
never mid-τ-descent. Run-2 grades: band@350 = a compromise (deconflicted off the boundary but still
mid-tau — accepted because the measured collision harm 3.4× exceeded the theoretical mid-anneal
harm); the ep300 3-way collision (tau + persistence-completion + seed-anneal-end) VIOLATES the
one-parameter-at-a-time rule — stagger in run-3 (§C.ii).

**B.4 Optimal control / costate (Boyd + the #247 meta-layer) — closed-loop vs open-loop.** The
curriculum is a hybrid Bolza problem (facet-5): continuous controls (τ, η, λ_eik) follow the smooth
singular arc = the DERIVED open-loop paths (B.1); discrete controls (stage switches, seed
injection, lever engagements) are bang-bang and fire when the switching function — the costate
λ = marginal-ΔS, read from per-class verdict attribution — crosses zero = the MEASURED triggers
(B.2). **Classification of every knob:** open-loop (derived path): τ(t), β(t), λ_eik(t), η(t)
cosine, seed-anneal shape. Closed-loop (measured trigger): CE→tau handoff, Muon engage, early-stop
(T is a free terminal time — fixed `--epochs` forfeits it), eikonal bump (already closed-loop),
lever engagements. Run-2 has the eikonal bump + early-stop closed; the handoffs remain open-loop
wall-clock = the audit's headline gap.

**B.5 Optimization theory (Shannon + Hinton) — WHERE the Adam→Muon switch belongs.** CE+softmax =
mirror descent = NG (#284 proven). As τ (effectively β_hosc) descends, the boundary-basin Hessian
anisotropy grows (interface curvature ∝ 1/width ⟹ κ grows as the partition sharpens; measured
κ≈19 at the basin). AdamW's diagonal preconditioner collapses on the correlated boundary Hessian;
Muon's polar step is the κ-buster (measured −32% from an identical fork). **Derived switch
criterion: engage Muon when (a) the partition is FORMED (no class below nucleus — Muon cannot
nucleate, measured) and (b) the tau-stage verdict has plateaued (the remaining error is
conditioning-limited, not formation-limited).** PR95's "stage 8" placement is the right ORDER for
the wrong REASON (their clock); ep726 is admissible only if the plateau happens to be there. With
warm-start momentum + LR anneal to 0.1 (GAP1+2, active in run-2). Transition dwell: rewarmup window
≥ AdamW moment memory 1/(1−β₂) steps (run-2's 20ep=1500 ✓); the critical-slowing exponent match
(dwell ∝ 1/gap²) is honestly OPEN — the cosine profile is a guess that satisfies the bound.

**B.6 EMA theory (MacKay + Ballé) — the π-group violation.** Verified: per-step updates, 75
steps/ep ⟹ ρ=0.997 is a 333-step = 4.4-ep window. PR95 ran 29,650 epochs; Quantizr's 0.997 was
tuned to their step budgets — the dimensionless group π_ema = N_ema/N_stage was never preserved.
Two regimes, one knob: early fast-descent wants TRACKING (small window; measured 78× shadow lag =
the violation's receipt); the finisher wants POLYAK AVERAGING over the settled tail (window ≈
0.1–0.3 × finisher steps ⟹ N_ema ≈ 2000–6000 ⟹ ρ_fin ≈ 0.9995–0.9998 — 0.997 averages only the
last 1.6% of a 274-ep finisher, too SHORT, not too long). **The fix is already BUILT:
`--ema-decay-finisher` (SWA-style, default None = bit-identical). Run-3: A/B 0.9995.** The 0.997
house value stays the default (deployed-checkpoint authority unchanged until a byte-closed A/B).

**B.7 π_train (Tao) — the budget check.** π_train = lr × steps × sharpness per phase; the 1000-ep
budget and 1e-3→1e-4 LR were never checked against it (row 4/5). Honest status: we lack a measured
sharpness trace per stage (the Hessian probe is a BUILD); π_train stays a registered FORM with the
constant pending — do not fake a number.

### The $0 log-mined evidence table (existing logs only; no new training)

| observation | run / source | value | what it anchors |
|---|---|---|---|
| plateau-eps miscalibration | #205 CE trace, review C1 | eps 1e-3 fires ep151 (rel slope −8.22e-4) while d_seg 0.005473@150→0.004752@300 still descending; −9.38e-5 only at ep250→275 | handoff law constants (1e-4 / 25 / 250) |
| tau-onset erosion (no seed) | #205 run.log | d_seg 0.004752@300 → 0.006568@400 CREEP while smooth loss fell | nucleus guard necessity; surrogate↔verdict decoupling |
| ep300 collision bump | FEED-ft (prior fresh run) | 0.0056→0.020 (3.4×), persistent 75+ep | one-homotopy-param rule; band@350 |
| Muon cold-switch transient | cert B5 (FIRE arm) | ep725 0.004316 → ep750 0.004674 (+8%), ~75 ep recovery | warm-start momentum (GAP2) |
| Muon vs AdamW | fork A/B 20260622 | −32% d_seg, gap widening | keep-Muon; κ-busting finisher |
| run-2 CE trajectory | pid 39999 run.log | 0.466→0.162→0.122→0.127 @ ep0/25/50/75 (n600 verdict) | ep300 will fire mid-descent (prediction, §C.i) |
| EMA lag | cert C3 | up to 78× early shadow lag | π_ema violation (tracking half) |
| islands not starved | focal calibration 2026-07-05 | islands 1.1% residual / 3.5% gradient; Lane flip 0.392→0.216 per 25ep | readiness trigger uses per-class within-flip, and TIME binds on bulk |

All `[macOS-MLX research-signal]` / advisory; none are scores.

---

## C. PROPOSALS — three horizons

### C.i SAFE-NOW (run-2, pid 39999): NOTHING FIRES. The watch list is the validation instrument.

- **ep300 (tau onset), the registered prediction:** CE will NOT be plateaued (extrapolating
  0.127@75; expect d_seg ≳ 0.05 at ep300) ⟹ the fixed-epoch boundary fires mid-formation. If
  post-300 the n600 verdict slope DEGRADES vs the pre-300 trend (or closed-loop classifies
  DIVERGING_ERASING → eikonal bumps engage), **the hand-off law is validated on live data**; if
  descent continues unharmed, the readiness criterion is too conservative — either way the run
  MEASURES the law for free. Confound on the record: seed-anneal completion and
  persistence-warmup completion land on the SAME epoch (the 3-way collision) — attribution between
  "MCF too early" and "crutch fully withdrawn" is partial; the closed-loop classification (erosion
  vs transient) is the disambiguator.
- **ep726 (Muon):** run-2 has `--muon-warm-start-momentum --muon-lr-final-frac 0.1` ACTIVE (the
  #270 levers). Observable: the #205/FIRE-arm +8% cold-switch transient should be ABSENT (spike
  ≤ noise band at ep726→750). Presence of a spike anyway falsifies the warm-start mechanism as
  the transient's cause.
- **Muon stage β-freeze check:** β(726) should be 4.00 (the 5.134 endpoint's whole purpose) —
  one-line log check at the boundary.

### C.ii RUN-3 CURRICULUM SPEC (the derived-native schedule; flags grep-verified, BUILDs labeled)

Diff vs run-2, in firing order:

1. **CE→tau handoff → trigger:** `--curriculum-event-triggered --curriculum-plateau-rel-eps 1e-4
   --curriculum-plateau-windows 25 --curriculum-min-stage-epochs 250` (the C1 recalibration; fixed
   epochs become CAPS). The C2 l7 guard is **already LANDED** (`7226d2651`: l7 never converge-fires
   when demoted, trainer:1204). **Remaining BUILD:** the per-class nucleus guard — fire only when
   every scored class has part_frac > 0 and within-flip below threshold (reads the existing
   per-class attribution; est ~60–120 LOC + tests). Event mode without it is admissible (the
   recalibrated plateau alone ≈ fires ep275–300) but the guard is what makes the law complete.
2. **Re-anchor wall-clock levers to the FIRED boundary** (review M1, class-level): persistence
   warmup, seed-anneal end, hosc-β anneal, band start all become boundary-relative (est ~40–80
   LOC). Until built, event-triggering stays gated.
3. **Stagger the ep300 3-way collision** (works even without event-triggering):
   `--seed-anneal-epochs 275` (crutch fully gone 25 ep BEFORE MCF onset) and persistence warmup
   completing ≥25 ep before the boundary (`--persistence-warmup-epochs 275`) — one parameter moves
   per epoch neighborhood. $0 flag-level change, READY NOW.
4. **Geometric β_hosc:** add `geometric` to `--hosc-beta-anneal` choices (est ~10 LOC + test);
   endpoints unchanged (1.0→5.134 stays derived).
5. **Muon engage → trigger:** fire the Muon switch on tau-stage verdict plateau + nucleus-complete
   instead of fixed 726, with 726→a CAP (BUILD, est ~80 LOC; Muon is currently NOT event-fireable
   — verified). Keep `--muon-lr 0.002 --muon-warm-start-momentum --muon-lr-final-frac 0.1`.
   Optional B5 refine: Muon-boundary LR rewarmup from a floor (~30 LOC; possibly redundant with
   warm-start — A/B).
6. **Finisher EMA:** `--ema-decay-finisher 0.9995` (flag EXISTS; A/B vs None; deployed-checkpoint
   authority unchanged until byte-closed comparison).
7. **β₂ arm (#222):** `--adam-beta2 0.9999` A/B (the help-text small-n law; NOT the primary arm).
8. **Unchanged (derived, keep):** τ constant 1.0 geometric · eikonal 0.05→0.10 · length 0.001 ·
   band 350 · rewarmup 20/0.1/cosine + reset-moments · l7 1001 · lr 1e-3→1e-4 cosine (echo
   acknowledged; an LR derivation needs the sharpness probe first — do not churn it blind).

### C.iii θ*/CAPSTONE — stages dissolve (design item, #218/#78/#247 lineage)

The unified-energy design: ONE E_τ with (a) active-contour region data term (energy≡loss, the
run-3 unification's completion — training loss IS the energy the flow minimizes); (b) τ(t)
integrated ONLINE from the measured per-class margin spectrum (constant-Fisher-velocity ODE in
closed loop, not a precomputed shape); (c) λ_eik(t), β(t) co-annealed to hold π_int ≳ 1 by
construction; (d) the costate controller (#247): per-class marginal-ΔS attribution = λ, discrete
events (Muon engage, lever engagements, STOP) fire on switching-function zero-crossings; (e) T
free (early-stop = choosing T). No stages — only one continuation with event-driven impulses.
Design doc owed before any build.

---

## D. TRIALITY UPDATES (landed with this memo)

- **equations:** `tools/register_lever_laws_curriculum_20260705.py` registers
  `curriculum_handoff_critical_nucleus_v1` (B.2; anchors: C1 miscalibration + #205 creep + knee
  π₁≈5) · `ema_window_pi_group_v1` (B.6; anchors: verified per-step cadence + measured 78× lag;
  finisher value PENDING A/B) · `muon_switch_conditioning_criterion_v1` (B.5; anchors: −32% fork,
  +8% cold-switch transient, 7.8× LR ratio; trigger-form constants FORMALIZATION-pending) ·
  `rewarmup_beta2_memory_window_v1` (B.5/row 17; anchor: spike-skip incident + 1500≥1000 check;
  exponent match honestly open).
- **DSL:** `CurriculumGauge` appended to `tac.witness_dsl.gauge` (APPEND-ONLY):
  `PR95_ECHO` (= () byte-identical, the honest name for run-2's fixed-epoch clock) ·
  `DERIVED_NATIVE` (emits the §C.ii stagger + finisher-EMA flags — `--seed-anneal-epochs 275
  --persistence-warmup-epochs 275 --ema-decay-finisher 0.9995` — and COMPOSES with
  `ControlSystemGauge.CONTROLLED` for the recalibrated CE→tau trigger; the nucleus guard /
  Muon trigger / geometric-β remain BUILDs named in its docstring) · `UNIFIED_ENERGY`
  (DESIGN-STAGE, NotImplementedError per the pose_training fail-closed pattern).
- **DAG:** FEED-05i appended (this study + the ep300/ep726 watch instrument).
- **council anchor:** appended via `tac.council_continual_learning.append_council_anchor`.

## Self-reflection (Catalog #363)

**Round 2** — re-classification after verification cycle: the "boundaries are cargo" claim was
UPGRADED from assumption to VERIFIED (cert A7's verbatim "PR95 stage-8 placement" + run-2's
measured trajectory divergence); the "rewarmup arbitrary" claim was DOWNGRADED — the window
satisfies a derived bound (β₂ memory); only shape/floor remain unexamined. The "we never examined
the schedule" framing was corrected per Contrarian: the 2026-06-30 cert examined every knob
PIECEWISE with open questions; what this symposium adds is the UNIFICATION (one energy, one
continuation, triggers-not-clocks) and the closure of cert Q1.
**Round 3** — verdicts depending on ASSUMED/INFERRED assumptions: the critical-slowing exponent
match (row 17) and π_train constants (B.7) are PROVISIONAL-PENDING-VERIFICATION — no run-3 knob is
churned on their basis (the spec's items 1–7 rest only on VERIFIED anchors). The ep300 prediction
is pre-registered so the live run resolves it mechanically.

**HARD GATE: pointer 0.19110 UNMOVED. Everything here is MEANS; the first milestone after run-2
converges remains a byte-closed `upstream/evaluate.py` n600 exact row.**
