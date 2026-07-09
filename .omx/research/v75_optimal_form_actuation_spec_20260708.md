# v7.5 OPTIMAL-FORM ACTUATION SPEC — the consolidated "turn on all optimal via DSL, recorded in triality" checklist (2026-07-08)

STORES CONSULTED (the #367 3-audit synthesis + pose campaign + operator insights, all committed):
Audit A `v75_dseg_lever_audit` (ad3270bed) · Audit B `v75_dynamic_curriculum_audit` (f3ac265ad) ·
Audit C `v75_r1_crossrun_best_audit` (8d8bdc9d6) · FEED-shippable (a0d02105f, pose 0.106 SHIPPABLE) ·
FEED-posesegdynamics (0d5731cd4) · FEED-undrivrecall (3832e47c7) · FEED-v75actuate (94b2e540f, the
binding operator directive). Author: main synthesis. Pointer **0.19110 UNMOVED** — this is the actuation
plan; a byte-closed n600 exact row is the only thing that moves it. Every Δd_seg [macOS-MLX
research-signal] NON-PROMOTABLE, n-labeled.

## THE HEADLINE (from all 3 audits): v7.5's d_seg base is OPTIMAL; the actuation is SMALL + SURGICAL
Audit A: v7.5 turns on the two decisively-measured d_seg levers (all-class directional basis **−48%**,
Muon **−32%**) in the correct synergy order (basis-match BEFORE capacity) and correctly AVOIDS every
measured-harmful item (l7 defect, smooth-stage +6.8%, fixed-β hosc divergence, isotropic-capacity-first
+6%, GradNorm, raw-paint-seed). Audit C: mod-32 is the best measured base (beats mod-26 16%), all best
levers + the net-new Chan-Vese counter-force ON. Audit B: the curriculum is ALREADY ~80% event-driven
(the "inert" claim was a telemetry misread). So this is NOT a rebuild — it is a short surgical actuation.

## THE ACTUATION CHECKLIST — all via DSL `Lever`/WitnessProgram (never-invent-flags), triality-recorded

### A. d_seg surgical fixes (Audit A gaps — measured, un-wired)
1. **DROP the two lane no-op flags + wire the measured 3× fix.** `--lane-prior-phi1-mode replace` (#291
   measured NO-OP) AND `--structured-init-include-lane` (lane_px=0, no-op) are BOTH emitting but inert.
   Replace with **paint-then-SDF** (`--lane-prior-phi1-mode paint`, #291 built) — measured lane FN
   0.0058→0.0019 (**3×**). DSL: the LanePriorPhi1 lever's mode → paint; drop the dead include-lane.
2. **Activate the along-tangent COMB / 2nd-order scattering A/B** (#287, FEED-08c) — the measured fix for
   the 3.2× along-tangent deficit (oracle best 0.00695); `freq-along 4→6` is FLAT (measured), so the comb
   is the real lever. Defensible-to-defer only because lane is offloaded to the analytic band; but it is
   an un-activated measured lever → wire as a DSL Lever, A/B it.
3. **The −48% directional transfer is the run's own A/B** (measured on a circular probe vehicle;
   production `--self-orient` realization unverified) — not a config change, a verification the run does.

### B. HORIZON FIX ON (FEED-undrivrecall — the Undriv 0.082 gap; measured floor 0.0016)
4. **Temporal-screw-consistency (#360, built)** — removes the inter-frame Undriv/horizon JITTER (the
   clean-canonical finding: the bulk floor is POSE-EXPLAINABLE jitter; Undriv-sky floor 0.0016 vs live
   0.082 = ~50×). DSL: TemporalScrewConsistency lever ON.
5. **0-byte horizon-weighted margin (#169 surviving lever)** + **sky=rotation-only stratification** —
   the per-class regime + the horizon-band margin term. DSL levers ON.

### C. DYNAMIC CURRICULUM residual (Audit B — already ~80% done)
6. **Fix the `plateau_ok` telemetry legibility defect** (observability — the exact thing that misled the
   earlier read): stamp readiness rows with `in_stage_epochs`/`stage_start` + dense-plateau slope.
7. **DERIVE `curriculum_min_stage_epochs`** (the one bare literal 250 left) off the value-provenance
   ladder — widen the event window so the plateau (not the floor) dominates timing.
8. **VERIFY the Muon event nucleation positive-control** (`--muon-start-event powerlaw_meat`) is
   satisfiable, else Muon silently falls to the 726 backstop.

### D. POSE SEQUENCING (Audit C gap + FEED-posesegdynamics + FEED-shippable — the one structural change)
9. **Replace co-train-pose-from-ep0 with the R1 TWO-PHASE sequence.** Pose is orthogonal + benign (R1:
   d_pose 97→0.0011 while d_seg HELD; seg⊥pose 99.95% null) → it belongs AFTER d_seg converges on a
   COHERENT render, NOT co-trained from ep0 on an incoherent one (why v7.5's as-configured pose sits
   ~1.79). DSL: add a **TERMINAL POSE-FINISH stage** (converge d_seg under the full optimal curriculum →
   w_pose-emphasized pose-finish, ~100 Muon ep, the R1 recipe) → then #238-serialize the dxi (SHIPPABLE,
   pose contribution 0.106, 7.2 KB, VALIDATED end-to-end). This SUPERSEDES Audit A's stale "pose is the
   binding wall / cannot reach sub-0.19" aside (it cited the pre-campaign SPEC §1; pose is now BANKED).

## THE OPTIMAL COMBINATION (synergy, from Audit A) — do NOT turn the union on blindly
Basis-match BEFORE capacity (isotropic-first HURTS +6%). Margin-field drives amplify/persistence/chroma-
annulus (Fisher r 0.978 — ONE saliency map, #141). Chan-Vese (area) ⟂ tie-locus (placement); the
separatrix-PLACEMENT residual (100% of the P-A oracle floor 0.000910) is P0 Force-3's domain, correctly
deferred to a later increment. Keep OFF/fixed: l7, smooth-stage, fixed-β hosc (annealed 1→3.177 +
siren-init), raw-paint-seed (uses eased n323 ladder), GradNorm/per-step reweight.

## SEQUENCE
Actuate A–D as DSL Lever/WitnessProgram changes (each: build → review-gate 2 passes → triality: DSL leg +
DAG FEED + canonical equation where a measured lever) → re-seal the v7.5 optimal WitnessProgram → launch
on operator-GO (launch held; CONFIG actuation GO'd via FEED-v75actuate). The d_seg stack is optimal-enough
TODAY; A/B/C/D are the surgical completion. Pointer 0.19110 UNMOVED until a byte-closed n600 exact row.
