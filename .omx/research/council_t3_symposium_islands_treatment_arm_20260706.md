---
council_tier: T3
council_topic: islands-on treatment arm for the level-set witness (vs the mod32cap clean baseline)
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, loss-red-team-lens, schedule-curriculum-lens]
council_quorum_met: true
council_verdict: REVISE
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
axis_tag: "[macOS-MLX advisory] NON-PROMOTABLE — pointer 0.19110 UNMOVED (MEANS, not a score row)"
# Catalog #300 v2-frontmatter backfill 2026-08-25: council_dissent +
# council_assumption_adversary_verdict + council_decisions_recorded transcribed VERBATIM from
# this memo's own sections "Lens A / Lens B", "Assumption-Adversary / recursive self-reflection
# (#363)", "The GATE", "Reactivation / branch criteria", "#323 LADDER", "Warm-start verdict".
# Frontmatter-only addition per the CLAUDE.md "Council hierarchy" backward-compatibility
# clause (NO body mutation).
council_dissent:
  - member: loss-red-team-lens
    verbatim: "took as given that #205 formed Lane(80%)+Movable(98.75%) islands via plain CE — this framed the 19% upside."
  - member: schedule-curriculum-lens
    verbatim: "COULD NOT verify the island-birth premise and found it CONTRADICTS the measured record: part_frac lane=0, movable=0 from ep0 under plain CE with no growth loss, in BOTH mod32cap and #205; the 80%/98.75% are likely within-flip of the BIG classes or a growth-loss-ON run."
council_assumption_adversary_verdict:
  - assumption: "#205 formed Lane(80%) + Movable(98.75%) islands via plain CE."
    classification: CARGO-CULTED
    rationale: "The two lenses DISAGREE on this load-bearing premise: the loss red-team took it as given, the schedule lens found it contradicts the measured part_frac record. Per Catalog #363 a verdict resting on an ASSUMED-class assumption is PROVISIONAL-PENDING-VERIFICATION — the REVISE verdict is provisional until the $0 checkpoint probe resolves it."
    empirical_verification_status: ASSUMED_AWAITING_VERIFICATION
council_decisions_recorded:
  - "Neither arm launches today. The next action is a $0 checkpoint probe, NOT a GPU run."
  - "THE GATE: run the trainer's own pure per-class functions (_evt_nucleus_counts L1599 -> _evt_nucleus_stats -> _evt_nucleus_satisfied) OFFLINE on the baseline EMA-shadow checkpoints — render -> argmax -> per-class part_frac + within-flip + big-3 d_seg vs gt_n600. NO training, NO flags, reads the BEST/stage npz not the live process. NOW at the ep225 ckpt, then across ep275/300/325/350."
  - "Branch A — lane share >= ~10% (un-born) AND ep300 erosion signature present -> PROCEED-class: next arm = island-birth loss + `--curriculum-event-triggered --curriculum-nucleus-guard` (the PAIR), FRESH not warm-start, seg-only; prefer the analytic lane render-band #213; A/B vs the live baseline at matched epochs. This is a REVISE — the loss arm wasn't wrong, it was measured on the wrong (erosive) schedule."
  - "Branch B — lane share <= ~2% (self-absorbing) OR no ep300 erosion -> DEFER-CORRECTED: islands are noise, the residual is ~98% BULK; redirect to bulk-CE deepening (tau nucleation #302) + Muon finishing (#270) + the analytic band. This DEFER is then MEASURED-reality, not conservatism." # MAGNITUDE_DISMISSAL_OK: historical 2026-07-06 council branch-table row on the retired islands/witness vehicle (pointer era 0.19110); the branch predicate was measurement-gated by design and the vehicle is superseded by the DDM campaign — preserved append-only, detector backfill 2026-08-25
  - "#323 LADDER island-birth PURSUE (operator 2026-07-06), split by the transfer proof: Movable = SDF-dilation homotopy -> PROVEN transfer, GO independent of the probe; Lane = manifold-preserving along-tangent + ξ-phase, headroom sized by the probe's lane-share."
  - "Warm-start verdict: FROM-SCRATCH. A CE-converged basin has ~0 gradient on islands; birth must happen during early plastic CE. Warm-start is reserved only for resuming a run that ALREADY has formed islands."
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
- **Ceiling** = 100·island_share·d_seg: island_share ≈1% → ΔS ≤ **0.005**;
  island_share ≈19% (lane un-born) → ΔS ≤ **0.094** (nearly sub-0.15 alone). The regime
  is UN-MEASURED — the crux.
- **CORRECTION (operator 2026-07-06): 0.005 is NOT noise at this frontier.** At S=0.19110
  → sub-0.15 the gap-to-goal is 0.041; a 0.005 exact drop clears ~12% of it and is a
  bankable row per "any score sub-0.19 is progress on the way down." The magnitude does
  NOT justify a DEFER — only the LEVER COST does. So the branch below is reframed:
  low-lane-share does not mean "don't pursue," it means "pursue via the CHEAPEST lever
  (analytic render-band #213, zero byte cost), not an expensive hard-costate build."
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

## #323 LADDER island-birth — PURSUE (operator 2026-07-06), split by the transfer proof
The symposium flagged #323's along-tangent lane homotopy as "designed-only / not wired."
Operator directs it be pursued. It is NOT one lever — the LADDER transfer proof splits it:
- **Movable = SDF-dilation homotopy → PROVEN transfer (1-Lipschitz ⇒ Hausdorff-continuous
  ⇒ forward-Euler ∂_t u=+|∇u|). GO independent of the probe** — movable island-birth is
  sound whatever lane-share turns out to be. This is the un-gated pursuit.
- **Lane = manifold-preserving along-tangent + ξ-phase.** The earlier "isotropic-of-a-curve
  NO-GO" is a DIFFERENT operator; `eased_targets.oriented_width_eased` (now openpilot
  VP-tangent grounded, commit d90a64466) widens ALONG the tangent → stays on the lane
  manifold → not under the NO-GO. Headroom sized by the probe's lane-share.
- **The wiring gap (NOW CLOSED, commit 705afea84):** `src/tac/witness_curriculum/eased_targets.py`
  WAS orphan (zero importers). "Pursue #323" wired it: `tac.boundary_math.island_protection.
  eased_island_masks` now consumes both `sdf_dilation_eased` (movable) + `oriented_width_eased`
  (lane VP-tangent), selected by the new `--seed-island-eased` flag (default-OFF ⇒ byte-identical
  when unfired; adversarially verified). It is NOT yet a DSL lever (`SeedIslandEased` factory owed
  under #332) — that is the remaining wiring, not the trainer binding.

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
