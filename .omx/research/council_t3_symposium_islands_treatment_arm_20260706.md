---
council_tier: T3
council_topic: islands-on treatment arm for the level-set witness (vs the mod32cap clean baseline)
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, loss-red-team-lens, schedule-curriculum-lens]
council_quorum_met: true
council_verdict: REVISE
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
axis_tag: "[macOS-MLX advisory] NON-PROMOTABLE — pointer 0.19110 UNMOVED (MEANS, not a score row)"
---

# T3 SYMPOSIUM — islands-on treatment arm — VERDICT: REVISE (gated on one $0 probe)

Operator directive 2026-07-06: the treatment arm gets the SAME grand-council symposium
that designed the mod32cap clean baseline, with deep math + adversarial review. Two
independent lenses deliberated (loss red-team + schedule/curriculum), plus the recall
evidence. They CONVERGE on **REVISE**, and — crucially — they converge on the SAME $0
measurement gate while DISAGREEING on a load-bearing premise (which the gate resolves).

## The convergent verdict

**Neither arm launches today. The next action is a $0 checkpoint probe, not a GPU run.**

### Lens A — loss red-team: the "fundamental conflict" is REFUTED (deep-math)
- `d_seg` is LINEAR in flip count (uniform per-pixel argmax mean; no concentration
  penalty). Birthing an island near boundary B: Δd_seg ∝ (n_big3 − n_isl). **Net-positive
  iff n_isl > n_big3.**
- UNIFORM amplification (every measured net-negative arm: full-stack 0.121, paint-seed
  0.026) raises the island logit at big-3-correct pixels too → n_big3 ≳ n_isl → net-neg.
- COSTATE/margin-GATED support Ω (amplify only where big-3 margin preserved) → n_big3→0
  → net-positive **by construction**. The #300 `--witness-alone-island-loss` soft-gate is
  MEASURED absorbing (lane within-flip −45%) WHILE total d_seg descended 0.162→0.122.
- ⇒ the recall's "birth-vs-total is a Pareto conflict" is a **wrong-support artifact +
  the seed-starvation bug (#300)**, IMPLEMENTATION-falsified, paradigm intact. Recording
  it as a fundamental conflict would be premature-KILL / conservative bias.
- **Ceiling** = 100·island_share·d_seg: island_share ≈1% → ΔS ≤ **0.005** (noise);
  island_share ≈19% (lane un-born) → ΔS ≤ **0.094** (nearly sub-0.15 alone). The regime
  is UN-MEASURED — the crux.
- The explicit **hard** costate-gated arm is **DESIGNED-ONLY**: reachability sensor inert
  (msal_uni at chance, #268 exact-S_R owed); per-class-λ actuator intentionally NOT wired
  to descent (costate_estimator is campaign-level, structural-containment); #323
  along-tangent lane homotopy NOT wired (only isotropic dilation = proven NO-GO for a
  curve). So the PROCEED branch's best lever may be the **analytic lane render-band
  (#213, d_seg 0.00087, a REPRESENTATION win at zero d_seg-cost)**, not a training arm.

### Lens B — schedule/curriculum: the schedule is the ENABLER of the loss arm
- mod32cap runs a PR95-ECHO **clock** on a partly-derived continuum: `tau@300` is the
  "worst cargo row" (a CE-knee epoch transferred from a DIFFERENT trajectory), `Muon@726`
  verbatim PR95, `EMA 0.997` a π-group violation. The physics (τ-path, length, Muon
  tuning) is largely derived; WHEN stages fire is still wall-clock epochs.
- **Nucleus theorem (Allen-Cahn, τ=ε=ħ MCF, #284/#302):** any scored class-region below
  critical size w/σ≳5 is ERASED by the tau stage, never grown. **#205 MEASURED the erosion
  signature: d_seg 0.004752@ep300 → 0.006568@ep400 (+38% creep across tau).**
- ⇒ the loss arm's measured net-negative was measured **ON THE EROSIVE FIXED SCHEDULE**.
  Fixed tau@300 taxes any born-but-nascent island before it reaches critical nucleus.
- **The #315 event-trigger + per-class nucleus-guard is FULLY WIRED** (verified
  end-to-end, not parsed-only: `_evt_resolve_seg_form` L1508, MEASURED nucleus predicate
  L4417, resume-persisted, boundary re-anchor L1703, cap-ceiling ⇒ never hangs ⇒
  byte-identical to fixed-schedule when unfired). It holds tau until every scored class
  consolidates → protects born islands at ZERO total-d_seg cost.
- Schedule-solo on the birthless baseline = near-zero effect (nothing to protect). Loss-
  solo = measured net-negative (starvation + erosion). **The PAIR** (island-birth loss +
  nucleus-guarded hand-off) is the only config where birth pressure can pay.
- Two FREE, already-wired schedule wins for ANY next run: `--muon-warm-start-momentum
  --muon-lr-final-frac 0.1` (kills the MEASURED +8% cold-Muon transient) + `--handoff-
  readiness-telemetry` (byte-identical; harvests the validation data). DESIGNED-ONLY
  BUILDs (don't block): Muon event-trigger (~80 LOC), geometric β_hosc (~10 LOC).

## Assumption-Adversary / recursive self-reflection (#363) — the adversarial catch

**Load-bearing assumption "#205 formed Lane(80%)+Movable(98.75%) islands via plain CE" =
empirical-verification-status ASSUMED/INFERRED — and the two lenses DISAGREE on it.** The
loss red-team took it as given (it framed the 19% upside); the schedule lens COULD NOT
verify it and found it CONTRADICTS the measured record (part_frac lane=0, movable=0 from
ep0 under plain CE + no growth loss, in BOTH mod32cap and #205; the 80%/98.75% are likely
within-flip of the BIG classes or a growth-loss-ON run). Per #363: a verdict resting on an
ASSUMED-class assumption is **PROVISIONAL-PENDING-VERIFICATION**. The verdict below is
provisional until the probe resolves it.

## The GATE — one $0 measurement resolves everything

Run the trainer's own pure per-class functions (`_evt_nucleus_counts` L1599 →
`_evt_nucleus_stats` → `_evt_nucleus_satisfied`) offline on the baseline EMA-shadow
checkpoints — render → argmax → per-class part_frac + within-flip + big-3 d_seg vs
`gt_n600`. NO training, NO flags, reads the BEST/stage npz not the live process.

- **NOW (ep225 ckpt):** answers the loss red-team's crux (lane share ≈1% vs ≈19%) AND
  resolves the premise contradiction (does plain CE birth lane at all?).
- **Across ep275/300/325/350 (as `--stage-checkpoints` writes them):** answers the
  schedule lens's discriminator (does big-3 d_seg CREEP up across ep300 like #205's
  0.004752→0.006568 = tau erosion confirmed?).

### Reactivation / branch criteria
- **lane share ≥ ~10% (un-born) AND ep300 erosion signature present** → PROCEED-class:
  next arm = island-birth loss + `--curriculum-event-triggered --curriculum-nucleus-guard`
  (the PAIR), FRESH not warm-start (CE-converged basin has ≈0 island gradient), seg-only.
  Prefer the **analytic lane render-band #213** (representation, zero-cost) as the lane
  lever; wire #268 exact-S_R + #323 along-tangent only if the training arm is needed.
  Fold in the two free Muon schedule wins. Predicted ΔS ceiling up to ~0.094; A/B vs the
  live baseline at matched epochs. **This is a REVISE — the loss arm wasn't wrong, it was
  measured on the wrong (erosive) schedule.**
- **lane share ≤ ~2% (self-absorbing) OR no ep300 erosion** → DEFER-CORRECTED: islands
  are noise; the residual is ~98% BULK (Undrivable + Road). Redirect to bulk-CE deepening
  (tau nucleation #302) + Muon finishing (#270) + the analytic band. Hold all islands
  treatment. This DEFER is then MEASURED-reality, not conservatism.

## Warm-start verdict: FROM-SCRATCH
Both lenses + the recall agree: a CE-converged basin has ≈0 gradient on islands; birth must
happen during early plastic CE. Warm-starting mod32cap's islandless ep200+ state makes birth
fight a settled minimum AND lands near tau-onset with nothing to protect. Reserve warm-start
only for resuming a run that ALREADY has formed islands.

## Bottom line
The symposium REFUTED the DEFER's reasoning (loss red-team), found the ENABLER the loss-only
view missed (schedule lens), and caught a premise contradiction between the two (recursive
self-reflection) — all resolvable by ONE $0 checkpoint probe. No GPU launch is warranted
until that probe runs. Pointer 0.19110 UNMOVED — MEANS.
