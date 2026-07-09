# v7.5 OPTIMAL-FORM ACTUATION — BUILD LOG (2026-07-08)

Actuation of `v75_optimal_form_actuation_spec_20260708.md` (b685a4eba) into `crucible_v7`
(`tac.witness_autoconfig._build_crucible_v7`) as DSL WitnessProgram deltas. Operator directives:
FEED-v75actuate (94b2e540f) + the mid-build elevation steer (Item A triad → priority-1; dash-comb
defer OVERRIDDEN; item-3 = a MEASUREMENT, register owed, NO concurrent GPU probe).

**CONFIG actuation, NOT a launch.** Heavy/paid launch stays operator-GO. Live #205 run (pid 63069)
untouched. **Pointer 0.19110 UNMOVED** — only a byte-closed n600 exact row from `upstream/evaluate.py`
(contest-CPU/CUDA, NEVER MPS) moves it. Every Δd_seg here is [macOS-MLX research-signal] NON-PROMOTABLE.

## ⚠️ RE-SEAL REQUIRED
This config change (paint + drop-include-lane + dash-comb, and any B/C/D follow-ups) MUST pass a fresh
v7.5 SEAL adversarial review BEFORE any launch. I did NOT re-seal (out of scope; not my authority).

## ITEM A — d_seg surgical fixes — LANDED (commit 1)
Surface: `_build_crucible_v7` `base` dict + `levers` tuple + `_CRUCIBLE_V7_DSL_LEVERS`; test fixtures in
`test_crucible_v7_config.py`. All via the DSL WitnessProgram (the emitter stays the DSL).

| # | item | DSL leg | status | measured anchor |
|---|------|---------|--------|-----------------|
| A.1 | paint-then-SDF + drop dead include-lane | `base["--lane-prior-phi1-mode"]="paint"` + `base.pop("--structured-init-include-lane")` | LANDED | lane FN 0.00713→0.00211 (~3×), #291 |
| A.2 | along-tangent dash comb (#287) | `DashComb()` Lever → crucible_v7 `levers` (8 now) | LANDED, DEFAULT-ON | oracle 0.00695; eq `dash_erasure_homogenization_20260707` |
| A.3 | −48% directional verification | — (a MEASUREMENT, not config) | REGISTERED OWED | −48% circular-probe; production `--self-orient` realization UNVERIFIED |

- **A.1** — prereq `--seed-islands` already emitted by crucible_v7 (verified). `replace` was the #291
  MEASURED NO-OP; `paint` = paint-then-SDF nucleation (ep0 admission gate = MEASURED part_frac[lane]>0).
  Dropping `--structured-init-include-lane` (lane_px=0, inert) makes the argv honest (no orphaned no-op).
- **A.2** — the `DashComb` factory already existed in the DSL; this build COMPOSES it into crucible_v7.
  Under the committed `lane_offloaded` basis regime, lane rides the FREE rule-118 analytic band at
  freq_along≈6 (cartoon scale) which provably cannot represent the ~25-cyc dash comb; DashComb supplies
  that sub-δ dash structure analytically at build time, rule-118 FREE at decode. No schedule-WHEN epoch
  flag ⇒ zero schedule-gate interaction; composes with `--lane-render-band` (F-3 coherence gate already
  asserts the band). Default-ON per the operator override of the spec's defensible-defer.
- **A.3** — operator: the −48% was measured on a circular-probe vehicle; whether production `--self-orient`
  (already ON) delivers it is UNVERIFIED. Do NOT fire a concurrent n600 GPU probe (2nd heavy GPU workload
  trips the machine-crash P0 hard gate while #205 holds the box). OWED: run as the v7.5 run's OWN
  directional A/B (self-orient ON vs OFF, same seed, first ~300 ep) AFTER launch or gated behind the live
  run. Pre-registered: production self-orient must move Road/all-class boundary d_seg toward −48%; a null
  reclassifies the *realization* (verdict-scope: the self-orient realization, not the directional paradigm).

**Triality:** DSL = DashComb composed in the crucible_v7 WitnessProgram + `_CRUCIBLE_V7_DSL_LEVERS`;
DAG = FEED-v75Aactuated (`sub015_DAG_*`); equations = `dash_erasure_homogenization_20260707` (A.2) + #291
measured anchor (A.1). All three consistent (no new EmpiricalAnchor: activation ≠ a new n600 measurement).

**Verification:** `test_crucible_v7_config.py` 51/51 + `test_witness_autoconfig.py` + `test_v75_birth_counterforce.py`
= 149 passed; `test_lever_registry.py`+`test_feed07b_build_levers.py`+`test_p0_forces_phase2_build.py` = 52
passed. `lever_registry.completeness().unmapped` UNCHANGED at 120 (no new orphan). ruff F clean.
crucible_v7 still compiles + validates 0 WitnessProgram.validate violations + schedule-provenance gate 0
NAKED. (Pre-existing UNRELATED collection error in `tools/run_compact_renderer_mlx_spine_runner.py:885`
— a stray `<<<<<<< Updated upstream` merge marker — is NOT from this build.)

## ITEM B — HORIZON FIX — PARTIAL (owed as commit 2)
- **B.4 temporal-screw-consistency (#360)** — the `TemporalScrewConsistency` Lever factory EXISTS
  (`curriculum_dsl.py:2778`; trainer flag `--seg-temporal-screw-weight` default 0.0). WIREABLE. NOT landed
  in this turn: it introduces `--seg-temporal-screw-start-epoch` (the factory default 0 is schedule-gate-
  EXEMPT as always-on, but the factory's own semantics say `start >= l7` — needs a formed partition to
  warp; under `--seg-form-unify-tau` l7 is dissolved, so the correct start-epoch/governance is a real
  design decision deserving its own reviewed commit, not a rushed same-turn add). OWED as commit 2.
- **B.5 horizon-weighted margin (#169) + sky=rotation-only** — **GAP / NO BUILT TRAINER FLAG.** grep of
  the levelset trainer argparse finds NO `--*horizon*` margin flag; sky=rotation-only exists ONLY as a
  warp-mode dict entry (`witness_autoconfig.py:427`), not a trainer flag or Lever factory. Wiring B.5 as a
  DSL lever would require INVENTING flags (forbidden). Item B's chroma boundary IS already on
  (`--seg-chroma-boundary-weight 0.1`). RECOMMENDATION: B.5 needs a trainer-side build (the 0-byte
  horizon-weighted margin term + a sky rotation-only stratification flag) BEFORE it can be a DSL lever —
  a genuine owed build, not an actuation.

## ITEM C — dynamic curriculum residual — OWED (commit 3)
- **C.6 plateau_ok telemetry legibility** — trainer-side observability (stamp `in_stage_epochs`/`stage_start`
  + dense-plateau slope on readiness rows). Score-neutral. Owed (trainer edit).
- **C.7 DERIVE curriculum_min_stage_epochs** — currently the bare literal 250 in crucible_v7's base. Deriving
  off the value-provenance ladder requires a named source (the note at `schedule_governance` shows it is an
  INPUT to `derive_octave_max_dwell` from `--anneal-epochs`); a real derivation, owed.
- **C.8 VERIFY Muon event nucleation positive-control** — RESULT: `--muon-start-event powerlaw_meat` IS
  emitted by crucible_v7 (verified) AND its governance declares it FIRES gated on the S2 REV-B
  nucleation-complete positive control (all LADDER arms past birth+hold+anneal). The `--muon-start-epoch 726`
  is the fail-safe BACKSTOP (LOUD `cap_fired_before_event` if the sensor did not fire by 726). So the Muon
  event nucleation IS wired + satisfiable via the event-wiring; it does NOT silently fall to 726 unless the
  nucleation sensor genuinely never completes (which is the correct fail-safe, LOUD not silent).

## ITEM D — POSE SEQUENCING — OWED (commit 4)
- **D.9 terminal POSE-FINISH stage** — the StoreNothingPoseCarrier + `--pose-carrier-xi-from-ckpt` /
  `--pose-carrier-dxi-scale` connectors EXIST (FEED-238resolved: R1 dxi BANKED, d_pose 0.001610 → 0.127,
  7.2KB). The one STRUCTURAL change: add a terminal pose-finish curriculum stage to the crucible_v7
  WitnessProgram (converge d_seg under the full curriculum → w_pose-emphasized ~100-Muon-ep pose-finish,
  R1 recipe) SUPERSEDING co-train-pose-from-ep0. This is a curriculum-stage build (a real TypedStage /
  Lever), owed as its own reviewed commit.

## NET
Item A (the operator-elevated priority-1 d_seg triad) is LANDED + reviewed + triality-recorded + tests
green. A.3 registered OWED (measurement, no concurrent probe). B.4/C/D are wireable-or-buildable owed
follow-ups; B.5 is a genuine trainer-build GAP (no flag to wire). RE-SEAL required before launch.
Pointer 0.19110 UNMOVED.

## ITEMS B/C/D — LANDED (2026-07-09, follow-up build agent; commits d90962f3f · 6412f2cf8 · 13ac03e64 · a07da4378)
Operator correction folded: B.5 BUILT to completion (NOT deferred — 0.012–0.024 ΔS is significant near
sub-0.15; building a new trainer flag+term is legitimate, never-invent-flags forbids WIRING a nonexistent
flag not CREATING one). Order landed: B.4 → D.9 → B.5(build) → C.6 → C.7. Each its own reviewed + triality
commit. **Pointer 0.19110 UNMOVED** (advisory [macOS-MLX], NON-PROMOTABLE). Live #205 (pid 63069) untouched.

| # | item | disposition | commit | triality |
|---|------|-------------|--------|----------|
| B.4 | temporal-screw #360 | **COMPOSED ON** in crucible_v7 (9th DSL lever); EVENT-governed on the annulus_plateau formed-boundary sensor (unify-τ replacement for the dissolved-l7 gate); start-epoch = FAIL_SAFE_CAP 450. Built the honest event-wire (`--seg-temporal-screw-start-event` + `_temporal_screw_gate` EBGate mirroring chroma + resume-registry, P0). | d90962f3f | DSL `TemporalScrewConsistency.start_event` + compose + sensor reg · DAG FEED-v75B4actuated · eqs = p0_forces §FORCE 1 (reg OWED byte-close) |
| D.9 | terminal POSE-FINISH (R1 two-phase) | **COMPOSED ON** as a `pose_finish` TypedStage (sister of muon) at the muon cap; supersedes co-train-pose-from-ep0. New `--pose-finish-start-epoch` gates the effective pose weight (0 pose-blind until the muon switch, then --w-pose joint descent; frame0 seg-free ⇒ pose CANNOT disturb d_seg). CAP-governed backstopping --muon-start-event. | 6412f2cf8 | DSL `TerminalPoseFinish` factory HOLDS the flag + pose_finish TypedStage · DAG FEED-v75D9actuated · eqs = FEED-238resolved R1 anchor |
| B.5a | horizon-weighted margin #169 | **BUILT, REGISTERED default-OFF** (`horizon_weighted_margin_169` duty-to-measure). 0-byte one-sided relu(m_target-m_wit) hinge on the SHARED _signed, STRATIFIED to horizon rows [96,288) AND GT-margin [0.3,0.5] — pushes ONLY the REDUCIBLE confident-GT band, EXCLUDES the <0.05 IRREDUCIBLE label-noise. `HorizonWeightedMargin` DSL factory. Exit-criterion n600 A/B registered. | 13ac03e64 | DSL factory + registered-off · DAG FEED-v75B5built · eqs = dseg_reducibility_gt_margin_verdict (reg OWED byte-close) |
| B.5b | sky=rotation-only | **BUILT, REGISTERED default-OFF** (`temporal_screw_sky_rotation_only`). The sky is at infinity ⇒ warp H_rot=K·R·K⁻¹ (ξ translation ρ zeroed), spatial-blended into the temporal-screw sky rows. `TemporalScrewConsistency(sky_rotation_only=, sky_row_hi=)`. Exit-criterion A/B registered. | 13ac03e64 | DSL params + registered-off · DAG FEED-v75B5built · eqs = ground-homography design note |
| C.6 | plateau_ok telemetry legibility | **DEFAULT-ON** (score-neutral observability; byte-identity by construction). Stamped the handoff_readiness row with stage_start/in_stage_epochs/min_stage_epochs + a plateau_slope (`_dense_plateau_slope`, scale-free). Fixes the sparse-verdict-cadence blind spot that misled the earlier plateau_ok read. | a07da4378 | trainer telemetry · DAG FEED-v75C67actuated · eqs = N/A (observability) |
| C.7 | derive curriculum_min_stage_epochs | **HARDCODED-WITH-WAIVER** on the value-provenance ladder (250 is the draft's hand-set floor, not measured/derived; did NOT fake a derivation). Named the OWED derivation = the critical-slowing τ_relax measurement ⇒ DERIVED-AT-CONFIG when measured. Sourced from `_CRUCIBLE_MIN_STAGE_EPOCHS` + `crucible_v7_min_stage_epochs_provenance()` on `tail_constant_provenance`. Emitted value UNCHANGED 250 (byte-identical). | a07da4378 | named constant + provenance fn + artifact wire-in · DAG FEED-v75C67actuated · eqs = owed τ_relax law |

**B.5 build-or-defer RECOMMENDATION (operator asked, then corrected to BUILD):** BUILT both pieces to
completion. The horizon-margin and sky=rotation-only are BUILT + HELD + REGISTERED default-OFF (A/B arms,
NOT composed ON — they would perturb the sealed config and carry the label-noise / clip-specific risks the
derivations flag). Temporal-screw (B.4) captures the HIGH-VALUE horizon fix (the ~50× Undriv-jitter lever)
and IS composed ON; horizon-margin adds the orthogonal 0-byte reducible-flip lever with a strict
label-noise-vs-real-recovery exit criterion. The RE-SEAL/A/B decides whether to promote horizon-margin +
sky to ON. This is the honest "off is a tracked queue with duty-to-measure" disposition.

**BINDING VERIFICATION (all pass):** crucible_v7 compiles + validates 0 WitnessProgram violations · 0 NAKED
schedule-provenance · lever_registry.completeness().unmapped UNCHANGED 120 (ZERO new orphans) · antagonists
OFF (l7 absent · hosc-beta-end 3.177 annealed not fixed · phi1-mode paint) · 262 tests GREEN across
crucible_v7 / p0_forces / schedule_provenance / lever_registry / witness_autoconfig / store_nothing /
event_wirings / sealed_205 / feed07b. Resumability P0 preserved (temporal-screw gate in the resume
registry; pose-finish re-derives from the persisted muon gate; horizon/sky default-off = no new state).

## ⚠️ RE-SEAL-REQUIRED
All of B.4/D.9/B.5/C.6/C.7 MUST pass a fresh v7.5 SEAL adversarial review (per SPEC_v75 §8 fix-ALL seal, 3
clean passes, re-derive-don't-confirm) BEFORE any launch. This agent did NOT re-seal (out of scope; not its
authority). Heavy/paid launch stays operator-GO. Pointer 0.19110 UNMOVED.
