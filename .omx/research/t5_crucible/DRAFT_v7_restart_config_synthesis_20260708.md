# DRAFT v7 RESTART CONFIG — pre-council synthesis (2026-07-08)

status: DRAFT for T3 council review → seal → governed stop of run-1 → relaunch.
review_status: pre-registered-only (NOT reviewed; council + 3×3+structure seal owed).
author: main orchestrator. Baseline: sealed crucible_v6 (run-1, ep7 at draft time).
Pointer 0.19110 [contest-CPU] UNMOVED — this config is MEANS until its byte-closed n600 exact row.

## STORES CONSULTED
witness_native_schedule_derivation_20260709.md (the CONTINUOUS verdict + hybrid rec) ·
schedule_provenance_gate_20260709.md (5 NAKED violations = this to-fix spec) ·
tail_k_build_20260709.md · ladder_full_homotopy_323_20260709.md ·
dashboard_live_tab_v6_schema_driven_20260708.md · ORCHESTRATION_LEDGER.md (reqs N–V) ·
DRAFT_OPTIMAL_STACK_v6_20260708.md (v6.4/6.5 errata carry over) · memories: L67 (l7 = measured
DEFECT) · L78/L79 (Muon keep + finishing schedule) · L22 (schedule_readback SoT) ·
config_must_be_dsl_defined (req V) · elementwise-audits-launder (structure round binding).

## AUTHORING FORM (requirement V — binding)
v7 is authored as a typed DSL WitnessProgram (pydantic schema, LawRef compile, program
manifest). NO hand argv. Every schedule element declares its `schedule_governance` class.
This draft SPECIFIES the program; the DSL object is the artifact, this file is not.

## 1. The five naked-violation resolutions (the gate's to-fix spec)

| v6 naked trigger | v7 resolution | class |
|---|---|---|
| `--tau-softplus-start-epoch 300` | **DELETED.** `--seg-form-unify-tau` (unified L_τ coupled to render-τ, geometric anneal, floor τ*=0.31). No discrete switch exists. | (dissolved) |
| `--l7-start-epoch 3000` | **DELETED.** l7 is a MEASURED DEFECT (L∞ sharpening inside a viscosity flow, d_seg-decoupling; L67/CLAUDE.md capstone caveat). v6 already parked it at 3000 (= never); v7 removes the flag entirely. | (dissolved) |
| `--muon-start-epoch 726` | **EVENT-TRIGGERED**: Muon entry fires on `powerlaw_meat_exit` of the τ-descent phase (#315 sensor; dwell-gated, same detector TAIL_k reuses per-cycle) — enter the metric finisher when first-order progress is measured-exhausted, not at a scripted epoch. Fixed 726 retained ONLY as tagged FAIL_SAFE_CAP (req-B; sensor: powerlaw_meat, rationale: ν-law settle+floor derivation from v6 seal). | EVENT + CAP |
| `--lane-band-start-epoch 350` | **EVENT-TRIGGERED**: fires on lane-class nucleus guard (π₁≳5 per-class critical-nucleus sensor, #315) — the band engages when the lane island is born and survivable, which is its design intent (v6's 350 was a hand-guess of that moment). CAP at 500 (tagged; rationale: past the eased-seed window even in slow runs). | EVENT + CAP |
| `--seg-chroma-boundary-start-epoch 300` | **EVENT-TRIGGERED**: fires when the margin-band annulus fraction stabilizes (annulus-convergence telemetry #333 sensor: annulus_frac plateau) — the chroma sharpener needs a formed boundary to sharpen. CAP at 450 (tagged). | EVENT + CAP |

Gate outcome: 0 NAKED. Two triggers deleted outright; three converted to named-sensor events
with tagged caps. All sensors already built (#315 powerlaw/nucleus, #333 annulus telemetry).

## 2. Schedule spine (the derivation's continuous form)
- ONE τ-homotopy: geometric anneal 1.0 → τ*=0.31 (knee-derived, P-TAU2), `cosine_hold` →
  `geometric` per the derivation. Loss = render = SAME τ (`--seg-form-unify-tau`).
- CE→tau transition-easing (rewarmup/moment-reset) DISABLED for the dissolved boundary; kept
  for Muon entry (a real optimizer change).
- Muon finisher: warm-start momentum + lr-final-frac 0.1 (L79, carried from v6).
- **TAIL_k** post-Muon: `--tail-cycles-max` k>0 (council to set k; propose 2), halving 0.5,
  dwell 237 (settle_window_v1), cycle floor 387.09 (tail_cycle_floor_v1), stop-marginal-s 1e-4.
- **LADDER**: `--ladder-island-homotopy` ON — movable dilation-GO (critical_nucleus_release_v1
  ceiling) + lane curve-prior (VP-tangent + dash-phase), per-class λ_c soft-gated (#315
  sensor); eased→held→annealed; uniform amplification structurally never emitted.
- Everything else: v6.4/6.5 sealed values carry over UNCHANGED unless council objects —
  β-end 10.0, LR pin 1000/1.0, fused-R, reanchor-levers, min-stage 250, structured init,
  seed-islands eased, persistence/amplify stack, weight-entropy λ=15, verdict-batch 32.

## 3. Pose block — CARRIED OVER VERBATIM (verified working in run-1)
`--w-pose 1.0 --pose-carrier --pose-carrier-source generated --pose-carrier-residual-mode
table` (store-nothing ξ, s_t self-fit reproduced the measured 0.044/2.562 optimum; pose term
training: 11.47→1.28 by ep3). Do NOT redesign a verified-live block. d_pose through the
byte-closed decode (#238) remains the export-time authority.

## 4. A/B + falsification (pre-registered)
- Incumbent arm: run-1's trajectory to the stop point (discrete-stage baseline trace) +
  the prior through-R trace (CE 0.01045→0.005443, τ0.3→0.004563, ep300 3.4× bump FEED-ft).
- FALSIFY unified-L_τ if it is worse than discrete at the τ*-floor on through-R d_seg →
  revert to discrete, keep geometric only (derivation's own criterion).
- ν-refit checkpoint ~ep500 (band 0.5–2× of 0.012653) carries over.

## 5. Gate chain at relaunch (all structural, no discipline required)
DSL program manifest (req V) → schedule-provenance gate (0 naked) → memory preflight →
admission governor → dashboard renders the new stage kinds additively (schema-driven).

## 6. Council asks (T3, sextet + Dykstra/Daubechies/Rudin emphasis + structure round)
1. Approve/adjust the three event-sensor choices + cap values (§1).
2. Set TAIL k_max (propose 2) + confirm stop-marginal-s 1e-4 vs λ_bytes economics.
3. LADDER gate thresholds: accept builder defaults or recalibrate from run-1's per-class
   λ trace at stop time (fresher data than the #205 corpus).
4. STRUCTURE ROUND (binding, blinded): re-derive the schedule SHAPE from the level-set
   energy without reading §2; compare after. Divergence = REVISE, not rationalize.
5. Confirm run-1 stop point: propose at the first Muon-entry cap OR at seal-complete,
   whichever first (checkpoints preserved either way).
