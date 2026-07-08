# Position V7 — S2 Dykstra (CO-LEAD): feasibility / continuation [no-triality]

STORES CONSULTED: CONVENING_T3_v7_design_symposium_20260708.md (my face) ·
DRAFT_v7_restart_config_synthesis_20260708.md (§1 caps · §2 spine · §4 A/B) ·
crucible_v7_authored_20260708.md (diff table · WIRING-GAP list · 0-naked gate) ·
witness_native_schedule_derivation_20260709.md (§2 turnpike · §3 codim-≥2 one-at-a-time · §4
derived structure) · tail_k_build_20260709.md (τ_k=max(τ_{k−1}·0.5, m_q/ln5) clamp ≥τ_end;
dwell-237 powerlaw_meat exit; k_max cap; PowerPlay stop) · ladder_full_homotopy_323_20260709.md
(per-class-λ soft-gate; eased→held→annealed; 1-Lipschitz stepper) · ORCHESTRATION_LEDGER.md
(reqs B/C/R/U; seal ep-ordering 600<675<726) · $0 PROBE: `ladder_homotopy.py` LadderArmSpec
defaults birth=60/hold=0/anneal=200 + `scheduled_radius` returns 0 after the window.

## The composition verdict: they COMPOSE (temporally staggered; each makes the next feasible)

MEASURED ordering (source + seal): τ-descent [1,~600] → held at τ*=0.31 → Muon cap 726 → TAIL
~1100+. LADDER arm windows (defaults) = birth+hold+anneal = 60+0+200 = **~260**, and
`scheduled_radius`→0 after the window **regardless of λ_c**. So at any epoch ≤2 continuations are
live (τ+LADDER on [1,260], both in the BENIGN high-τ regime where the interface is soft/
well-conditioned — NOT the stiff κ≈19 floor), never 3. **TAIL×LADDER oscillation is structurally
impossible**: LADDER amplitude is 0 for all ep ≥ muon_start, so a TAIL τ_k re-raise cannot
re-open the gate (the schedule ceiling, not the gate, terminates support). TAIL's τ_k is clamped
≥τ_end=0.31 and halving-contractive with k_max=2 → any re-raise is bounded/shrinking and never
re-enters the coarse-τ nucleation regime; it perturbs the Muon-conditioned annulus, not the class
partitions. **Alternating-projections read:** the composed feasible set at τ* is NONEMPTY *because*
of the LADDER — Γ-convergence lands the τ-continuation on the argmax minimizer, and the per-class
source (Esedoğlu–Otto/Baldo, the theorem-cure) is exactly what keeps C_ladder(lane) ≠ ∅ against
MCF minority-erasure (retention 1.00→0.13 without it). The three are a designed sequence, not
competing constraints. **One feasibility ceiling to name (not block):** if lane-dash scale <
τ*/2-px half-width, C_ladder(dash) ∩ C_τ(τ≥0.31) = ∅ at the floor (L65 dash_erasure) — the lane
arm's VP-tangent+dash-phase support is the attempt; bounds achievable d_seg, pre-registered.

## Position — the council_pending knobs

1. **Event-sensor choices + caps (§6.1): ACCEPT caps 726/500/450 as-authored; DEFER sensor
   wiring.** Feasibility says the caps are the *confound-safe* form: a fixed staggered ordering
   trivially satisfies one-continuation-at-a-time + the Muon "nucleation-complete" precondition
   (LADDER done ~260 ≪ 726). The event wirings are where the codim-2 / misfire risk lives (below).
2. **TAIL k_max (§6.2): ACCEPT 2; stop-marginal-s 1e-4 ACCEPT (economics = S1/S4).** τ_k halving →
   geometric yield decay; PowerPlay stop is the real terminator, k_max=2 the fail-safe cap. Both
   present = req-B satisfied.
3. **LADDER gate thresholds (§6.3): ACCEPT builder defaults** (λ-gate OPEN 0.0 movable per T3
   split; release-coeff 0.95; σ_eff 1.5). The "recalibrate from run-1 λ trace" option is MOOT —
   run-1 produced ~no trajectory; the soft-gate is self-calibrating on live λ_c anyway.
5. **run-1 stop (§6.5): stop at seal-complete** — cleanest A/B (both arms restart from
   structured-init = comparable; supports S5's comparability concern). Checkpoints preserved either way.

## Verdict contribution: PROCEED_WITH_REVISIONS

- **REV-A (cheap structural guard, req-B completion guarantee — the one that binds):** add a
  config/test assertion `max_arm(birth+hold+anneal) < muon_start_epoch` (default 260<726 PASSES).
  Not a bug today — a guard so a future config lengthening lane's anneal cannot silently create
  a codim-2-at-the-floor overlap OR a live-LADDER-during-TAIL resurrection.
- **REV-B (build-time, for the OWED wiring only — DORMANT at launch):** if/when the
  powerlaw_meat→Muon event is wired live, gate it on a nucleation-complete positive control (no
  per-class λ_c above birth threshold) so a LADDER anneal transient cannot be misread as
  meat-exhaustion → premature Muon before lane nucleates. AS-AUTHORED (fixed cap 726) the cap does
  not fire on a transient, so this confound is dormant — a run-2 requirement, not a launch blocker.
- **Launch-vs-build:** LAUNCH with the 3 tagged CAPs. Building sensors first ADDS the codim-2/
  misfire feasibility risk the caps avoid, and "events beat caps" is UNMEASURED (req R). Gate the
  launch on REV-A only.

## Assumption tags (#363)
- LADDER completes ~260 ≪ Muon 726 → inert during TAIL: **VERIFIED_VIA_SOURCE_INSPECTION**
  (`ladder_homotopy.py` LadderArmSpec defaults + `scheduled_radius`→0 after window).
- τ_k clamped ≥τ_end, halving-contractive, k_max=2 (bounded re-raise): **VERIFIED_VIA_SOURCE_INSPECTION**
  (tail_cycles `next_tau`; tail_k_build memo).
- codim-≥2 one-at-a-time + Γ/MBO feasible-set-nonempty: **INFERRED_FROM_DOMAIN_LITERATURE**
  (derivation §3/§4; Modica-Baldo, Esedoğlu-Otto — cited there, not re-verified here).
- dash-scale < τ*/2 ⇒ C_ladder(dash)=∅ at floor: **ASSUMED_AWAITING_VERIFICATION** (L65 anchor;
  the run's per-class d_seg trajectory measures it — a pre-registered ceiling, not a blocker).

Pointer 0.19110 [contest-CPU] UNMOVED — v7 is MEANS until its byte-closed n600 exact row. [no-triality]
