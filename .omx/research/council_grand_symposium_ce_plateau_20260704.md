---
council_tier: T3
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Hotz, Hinton, Tishby, MacKay, Ballé, PR95Author]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "You are all treating the CE plateau as the gate. #205's CE floor did NOT predict its final score — the erosion happened in tau. The fresh run's watch may be pointed at the wrong epoch. Prove CE-priming gates the final BEFORE spending the A/B on a CE number."
  - member: Tishby
    verbatim: "CE is not supposed to birth the boundary — annealing (tau) is. A flat CE d_seg at a sub-critical IB point is not evidence of a capacity wall; it may be the expected pre-anneal state. The mod-dim critical-width prediction is real, but only tau's temperature sweep reveals whether 19 is below it."
council_assumption_adversary_verdict:
  - assumption: "The CE d_seg plateau at 0.0287 is a capacity problem (mod-19)."
    classification: CARGO-CULTED
    rationale: "Three untested alternatives outrank it in prior-probability given our history: (a) a DEAD-GRADIENT BUG (the #205 elephant was exactly this — bare .round() zero-grad masqueraded as a capacity wall for days); (b) NON-COMPARABLE BASELINE (#205 had no seed-shield/paint; the fresh witness-alone verdict excludes the seed carrying lane mass, so 'fresh CE should beat #205 CE' compares different dynamical systems); (c) WRONG-LOSS (CE weights the 0.6% boundary at 0.6% of the gradient — the bulk drowns it, a loss-design not capacity problem). None measured yet."
  - assumption: "d_seg=0.0287 is boundary/lane-dominated."
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "Not decomposed per-class. Could be bulk-class boundaries (road/undrivable) still coarse, which is a different fix than lane. $0 on the ep50 ckpt."
# Catalog #300 v2-frontmatter backfill 2026-08-25: council_decisions_recorded transcribed
# VERBATIM from this memo's own sections "The REVISION (what the council changes)" +
# "Recursive self-reflection (Catalog #363, Round 2)". Frontmatter-only addition per the
# CLAUDE.md "Council hierarchy" backward-compatibility clause (NO body mutation).
council_decisions_recorded:
  - "REVISION: do NOT fire the n192 A/B yet — insert a $0 disambiguator battery on the live ep50/ep75 checkpoints FIRST: D1 gradient-flow (Rudin) / D2 overfit-one-pair mod-19-vs-mod-32 (Dykstra+Hotz) / D3 per-class d_seg decomposition of the 0.0287 (Yousfi) / D4 seed-shield comparability audit (Hotz+Assumption-Adversary) / D5 margin-weighted-CE micro-probe (Shannon+Yousfi)"
  - "Conditional routing on the battery: D1 shows dead-grad -> FIX THE BUG, no A/B; D2 shows mod-19 overfits fine -> convergence, pivot to convergence-support (margin-CE / freq-warmup) not the capacity A/B; D2 walls at 19 AND D3 says lane AND D4 confirms comparable -> fire arm B (cheap) before arm C (rate)"
  - "Re-point the primary watch from CE-floor to tau-onset boundary mass (per-class mass above the critical nucleus), NOT total CE d_seg (Tishby + Contrarian)"
  - "Round-3 resolution: the disambiguator agent VERIFIES each probe is runnable before claiming its verdict; any probe that cannot run on existing surfaces is downgraded to pending_harness rather than asserted. No verdict from an unrun probe."
---

# GRAND COUNCIL SYMPOSIUM — deep-math + engineering review AGAINST the CE plateau

**Convened:** 2026-07-04 (operator: "Convene grand council symposium to do deep math and
engineering review against plateau"). **Subject:** fresh seeded run CE d_seg FLAT 0.0287 (ep25→50,
loss −12%); mod-19 vs #205's mod-32; the 3-cause triage in `mod_dim_capacity_ab_scoping_20260704.md`.
**Pointer 0.19110 UNMOVED — this symposium is MEANS.**

## The verdict in one line
**PROCEED_WITH_REVISIONS: do NOT fire the n192 A/B yet.** The A/B tests the capacity/bandwidth
story, but the council identifies TWO higher-prior causes it does not separate (a dead-gradient BUG;
a non-comparable-baseline artifact) and a WRONG-LOSS cause it under-weights. Run a **$0 disambiguator
battery on the ep50/ep75 checkpoints FIRST** — it is cheaper, faster, and decisive about *which of
five* causes is real, and it may dissolve the plateau entirely (as the #205 "wall" dissolved into an
artifact four times). Re-point the watch from CE-floor to tau-onset boundary mass.

## The five candidate causes (the triage the council expands from 3 → 5)
1. **Isotropic capacity** (mod-19 < trainability-critical) — the scoping memo's cause 1.
2. **Along-tangent bandwidth** (n-dir-freqs 2, anti-alias-coupled) — cause 2.
3. **Convergence-limited** (mod-19 sufficient, CE not there yet) — cause 3.
4. **Dead-gradient BUG** — the #205-elephant class; the boundary params may be receiving ~0 gradient.
5. **Non-comparable baseline** — the seed-shield excludes the seed's lane mass from the witness-alone
   verdict; "fresh CE should beat #205 CE" may compare different systems. (+ WRONG-LOSS: CE gives the
   0.6% boundary 0.6% of the gradient — a loss-design cause spanning 3 and 4.)

## The voices (deep-math, each lens distinct)

**Shannon (LEAD) — indirect-RD / the CEO problem.** The plateau is the surrogate minimizing the
WRONG distortion. CE is per-pixel log-loss over the whole frame (bulk-dominated); d_seg is
argmax-disagreement on the codim-1 boundary. Once the bulk is fit, `∇CE` points at bulk refinement,
not boundary sharpening — I(reduce-CE ; reduce-d_seg) DECOUPLES. **The boundary contributes ~0.6% of
the CE gradient by pixel count** — it is drowned, not absent. This is a LOSS-DESIGN problem before it
is a capacity problem. Test: margin-weighted CE (#141) re-weights the gradient onto the boundary at
$0 config cost.

**Rudin (CO-LEAD) — explain, don't characterize; bug vs wall.** The decisive cheap test:
freeze ep50, compute `∂(d_seg-surrogate)/∂θ_boundary`. Gradient ≈ 0 → BUG (dead-grad, the elephant).
Gradient nonzero but d_seg not falling → landscape (capacity/conditioning). This gradient-flow probe
must run BEFORE any A/B — it separates cause 4 from 1/3 in minutes.

**Dykstra (CO-LEAD) — feasibility.** Is 0.0287 ON the mod-19 achievable frontier, or is the optimizer
stuck above it? Overfit test: fit a mod-19 witness to ONE pair's partition with NO rate constraint. If
it reaches d_seg ≪ 0.0287 → 19 HAS the capacity, the plateau is convergence (cause 3). If it walls →
capacity (cause 1). mod-32 same, as the reference. This is the capacity-vs-convergence disambiguator.

**Daubechies (CO-LEAD) — the basis is the rate.** The boundary is a C² cartoon edge; Fourier gives
O(N^-1/2), curvelets O(N^-2(logN)^3). mod-dim spends isotropic DOF on the cross-edge direction where a
1-D step suffices. The along-tangent axis (n-dir-freqs) is right, but the anti-alias coupling
(ndf 4 needs freq-across 8) is a symptom of single-scale thinking — **a proper multiscale curvelet
(n-scales) anti-aliased per octave beats a single-scale ndf bump.** Arm B should also carry the scale
axis, not only ndf.

**Yousfi — decompose before theorizing.** Is 0.0287 lane or bulk? The margin-saliency map (#141)
names the flipping pixels. $0 on the ep50 ckpt. If bulk-class boundaries, the whole capacity story is
mis-aimed. UNIWARD: put the gradient where the detector is sensitive (the boundary) — same conclusion
as Shannon's margin-weighted CE, from the steganalysis side.

**Hotz — stop theorizing, measure the cheapest thing.** Overfit-one-pair at mod-19 vs mod-32 is a
10-minute test. And CHECK THE EVAL: the witness-alone verdict excludes the seed — is the exclusion
correct? A verdict-path bug would look exactly like a plateau. Ship the cheap probe before the campaign.

**Hinton — teacher-student capacity test.** #205 (mod-32, reached 0.00475) is a TEACHER. Distill its
boundary into a mod-19 student. If the student MATCHES at mod-19 → 19 has capacity, the fresh plateau
is init/optimization. If it can't → capacity. A clean capacity oracle that reuses existing weights.

**Tishby (dissent) — the IB phase transition + the annealing reframe.** CE compresses X→T; d_seg is
I(T;Y_task). The plateau is a sub-optimal IB point: bits allocated to the bulk, not the boundary.
mod-dim is the bottleneck width and there IS a critical width below which the boundary can't be
represented — BUT the annealing (tau's temperature sweep) is what moves the system along the IB curve.
**A flat CE d_seg is the EXPECTED pre-anneal state, not a wall.** Only tau reveals whether 19 is
sub-critical. → re-point the watch to tau-onset.

**Contrarian (dissent) — the watch is pointed wrong.** #205's CE floor (0.00475) did NOT predict its
final (the erosion in tau did). We are panicking about a CE number that may not gate the final score.
Prove CE-priming gates the final before spending the A/B on it.

**Assumption-Adversary — the two load-bearing cargo-cults** (see frontmatter): "plateau = capacity"
(CARGO-CULTED; bug/non-comparable-baseline/wrong-loss all outrank it and are unmeasured) and
"0.0287 is lane" (UNVERIFIED; decompose first).

## The REVISION (what the council changes)
**Insert a $0 disambiguator battery BEFORE the n192 A/B**, run on the live ep50/ep75 checkpoints:

| # | probe | separates | who | cost |
|---|---|---|---|---|
| D1 | gradient-flow: `∂d_seg-surrogate/∂θ_boundary` at ep50 | BUG (4) vs wall (1/3) | Rudin | $0, min |
| D2 | overfit-one-pair, mod-19 vs mod-32, no rate | capacity (1) vs convergence (3) | Dykstra/Hotz | $0, ~15 min |
| D3 | per-class d_seg decomposition of the 0.0287 | lane vs bulk (aims the fix) | Yousfi | $0 |
| D4 | seed-shield audit: is the witness-alone verdict correct? + #205 witness-alone-equivalent | non-comparable baseline (5) | Hotz/Assumption-Adv | $0 |
| D5 | margin-weighted-CE micro-probe: does re-weighting the gradient onto the boundary move d_seg? | wrong-loss (Shannon) | Shannon/Yousfi | $0, ~15 min |

**Then, conditional on the battery:** if D1 shows dead-grad → FIX THE BUG (no A/B). If D2 shows 19
overfits fine → convergence, pivot to convergence-support (margin-CE/freq-warmup), not the capacity
A/B. If D2 walls at 19 AND D3 says lane AND D4 confirms comparable → THEN the capacity/bandwidth A/B
is the right tool, fire arm B (cheap) before arm C (rate). **Re-point the primary watch from CE-floor
to tau-onset boundary mass** (Tishby/Contrarian) — the CE-priming criterion (per-class mass above the
critical nucleus), not total CE d_seg.

## Recursive self-reflection (Catalog #363, Round 2)
The council's own Round-1 assumption — "a $0 battery can separate the causes on existing checkpoints"
— carries `empirical_verification_status: ASSUMED_AWAITING_VERIFICATION` for D1/D2/D5 (the trainer
may not expose `∂d_seg/∂θ_boundary` cleanly, and overfit-one-pair needs a scratch harness). Round-3
resolution: the disambiguator agent VERIFIES each probe is runnable before claiming its verdict;
any probe that can't run on existing surfaces is downgraded to `pending_harness` rather than asserted.
No verdict from an unrun probe.

## Mission alignment
frontier_breaking: the battery either dissolves the plateau (unblocking the pointer-mover run cheaply)
or correctly aims the A/B spend. It serves the exact-score mission by refusing to spend GPU on a
capacity story that four prior "walls" suggest could be an artifact. Pointer 0.19110 UNMOVED.

<!-- STORES CONSULTED (2026-08-25 backfill append; this 2026-07-04 council memo predates the recall-evidence discipline #713): MEMORY.md index · .omx/research corrections index (au1) · task ledger #1274 (#300 v2-frontmatter backfill). Consulted for the frontmatter/waiver-append pass only; the deliberation body is historical, append-only, unchanged. -->

---

<!-- # COUNCIL_ROSTER_INCOMPLETE_OK:under_rostered_T3_convocation_20260704_acknowledged_not_repaired_missing_inner_quantizr_selfcomp_postdates_20260519_roster_landing_so_no_era_exemption_applies_historical_attendee_list_preserved_unmutated_append_only_per_catalog_110_113_no_invented_attendance_arm_cr1_20260825 -->

## Catalog #346 roster note — appended 2026-08-25, APPEND-ONLY

**Honest record, not a repair.** This T3 deliberation (2026-07-04) was convened AFTER the
2026-05-19 canonical-roster landing, so no era exemption applies: every seat it owed already
existed. It did not seat 2 mandatory inner-council voice(s): **Quantizr, Selfcomp**.

The absent seats are inner-council sister voices (not co-leads); the 4-co-lead shared-leadership core WAS seated.

The `council_attendees` list above is the HISTORICAL RECORD of who actually deliberated and is
**NOT mutated** — no attendance is added retroactively (fabricated attendance would be a fake
council record per the CLAUDE.md NO-FAKE supreme rule). Per Catalog #110/#113 APPEND-ONLY
HISTORICAL_PROVENANCE the gap is recorded here rather than papered over. Per CLAUDE.md
"Forbidden premature KILL without research exhaustion" this is an acknowledgement, not a
retraction of the deliberation's content.

Verified by `tac.canonical_council_roster.validate_council_dispatch_roster` after the 2026-08-25
detector cures (attendee-name normalization + seat-availability era filter), so the seats named
above are genuinely absent — not spelling variants and not anachronistic demands.
Ledger: `.omx/research/ddm_cr1_check346_roster_backfill_20260825.md`.
