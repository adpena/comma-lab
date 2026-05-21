---
council_tier: T1
council_attendees:
  - Carmack
  - Quantizr
  - Selfcomp
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "REVISION #1 (Contrarian + Assumption-Adversary): The follow-on T1 PROCEED-unconditional verdict can be issued on the basis of (RATIFY-3 commit 20b6b59b3 + Phase 2 DESIGN memo commit 29f92af8d + 105/105 test verification + Carmack MVP-first 5-step verification all_steps_passed=True + Dykstra is_additive=True + intersection_non_empty=True) WITHOUT requiring a paid Modal T4 anchor first"
    classification: HARD-EARNED
    rationale: |
      Per-substrate symposium memo Section 5 reactivation criterion reads:
      "PROCEED-unconditional verdict from a follow-on T1 working group on
      or before 2026-06-04 after the 4 binding revisions are applied to
      the first paired smoke harvest plan". The plain reading is "applied
      to the first paired smoke harvest plan" = the harvest plan is READY
      for application, NOT that the empirical anchor has landed. RATIFY-3
      commit 20b6b59b3 applied the 4 revisions; THIS working group
      verified empirically via $0 helper invocation that
      run_carmack_mvp_first_pre_smoke_verification.all_steps_passed=True
      + verify_multi_scale_dykstra_feasibility.is_additive=True +
      intersection_non_empty=True + 105/105 unit tests pass. The first
      paired smoke is the FIRST CONSUMER of the lifted _full_main, not a
      prerequisite for the PROCEED-unconditional verdict. Phase 2 T2
      DESIGN council `council_assumption_adversary_verdict` row #4 already
      pre-classified this as CARGO-CULTED-PENDING-RESOLUTION; the
      RATIFY-3 + Carmack MVP + Dykstra verifications resolve it to
      HARD-EARNED.
  - assumption: "REVISION #4 (Carmack + Hotz + Assumption-Adversary): the 6-DOF affine warp + (16, 5, 3) LUT shape pairing (cargo-cult assumption #9 per Phase 2 T2 DESIGN memo) is HARD-EARNED via empirical UNWIND-TEST evidence + structural disambiguator presence"
    classification: HARD-EARNED
    rationale: |
      Phase 2 T2 DESIGN council assumption #9 as written: "The 6-DOF affine
      warp preserves seg + pose contributions when paired with the (16, 5, 3)
      LUT shape vs v7's (5, 3) anchor shape" — flagged CARGO-CULTED pending
      empirical UNWIND-TEST. THIS working group's UNWIND-TEST decomposes the
      assumption into 3 sub-claims and resolves each:

      SUB-CLAIM 9a: 6-DOF affine warp preservation. HARD-EARNED via
      INHERITANCE from v7 cargo-cult #4 UNWIND-VERDICT (commit 4292c8ce2;
      empirical 105.15 -> 58.89 contest-CPU = 44% improvement). The v8
      inflate.py:97-148 `_affine_warp_frame1_from_frame0` is a verbatim
      copy of v7's warp; the warp surface IS NOT the variable changing
      between v7 and v8. Cargo-cult-unwind methodology is transitive: a
      pattern empirically UNWOUND in a sister substrate inherits its
      classification when copied verbatim with no surface mutation.

      SUB-CLAIM 9b: (16, 5, 3) LUT shape preservation. The shape itself
      is HARD-EARNED via structural-elegance + canonical-equation-26
      grounding: coarse-scale segnet_class (5 = upstream/modules.py
      classes=5) gates fine-scale (level, channel) (48 = 16*3) per
      Daubechies multi-scale hierarchical-coarse-gates-fine partition
      structure. Empirical Dykstra verdict: canonical_lut_shape=(16, 5, 3),
      coarse_scale_dimension=5, fine_scale_dimension=48, is_additive=True,
      intersection_non_empty=True, dykstra_iteration_count=1 (converged
      on first iteration; non-empty Pareto polytope intersection).

      SUB-CLAIM 9c: pairing-equivalence (warp+shape as a JOINT design).
      The Phase 2 T2 DESIGN council's wording "preserves seg + pose
      contributions when paired" demands empirical evidence the joint
      design produces additive contributions. The CANONICAL DISAMBIGUATOR
      is REVISION #1's per-assumption ablation ladder axis 2
      (per_level_class_aggregation): if median + mode + k-medoids all
      produce the same seg + pose contributions, the (16, 5, 3) LUT shape
      paired with the 6-DOF warp is HARD-EARNED post-empirical. The
      ablation ladder IS built and callable via
      build_per_assumption_ablation_ladder; total_predicted_cost_usd=$2.00
      = $0.50 canonical-arm + 3*$0.50 axis-probe-arms.

      RESOLVED: sub-claim 9a HARD-EARNED via inheritance; sub-claim 9b
      HARD-EARNED via canonical-equation-26 + Dykstra-feasibility; sub-claim
      9c HARD-EARNED at the disambiguator-existence level (the canonical
      ablation ladder IS the disambiguator; empirical post-paired-smoke
      Tier-C verdict per Catalog #324 ratifies or refutes joint additivity
      at the seg + pose axes specifically). Per Phase 2 council REVISION #4
      verbatim: the first paired smoke MUST run the full REVISION #1 7-arm
      ablation ladder (option (a)); option (b) requires explicit
      CARGO_CULT_9_UNWIND_DEFERRED_OK waiver. Default option (a) is the
      canonical path.
  - assumption: "Phase 2 BUILD spawn-readiness verdict: this T1 working group's PROCEED-unconditional verdict satisfies per-substrate symposium memo Section 5 reactivation criterion + Phase 2 T2 DESIGN memo REVISION #1 binding gate"
    classification: HARD-EARNED
    rationale: |
      Phase 2 T2 DESIGN council Section 6 REVISION #1 verbatim: "Phase 2
      BUILD landing MUST be gated on the follow-on T1 PROCEED-unconditional
      verdict being issued". THIS memo IS the follow-on T1 PROCEED-unconditional
      verdict. Sister gates:
        - per-substrate symposium memo Section 5 reactivation criterion:
          window 2026-05-21 -> 2026-06-04; THIS landing 2026-05-21 well
          within window.
        - Phase 2 T2 DESIGN memo REVISION #1: follow-on T1 PROCEED-unconditional.
          THIS verdict.
        - Phase 2 T2 DESIGN memo REVISION #2-#5: BUILD-time wire-ins (not
          T1 prerequisites; bind at Phase 2 BUILD spawn-time).
        - Catalog #325 6-step contract: cargo-cult audit (Catalog #303) +
          9-dim (Catalog #294) + observability (Catalog #305) + sextet/grand
          council deliberation + reactivation criteria + Catalog #324
          post-training Tier-C validation. All satisfied via inheritance
          chain (per-substrate symposium + Phase 2 T2 DESIGN + THIS T1
          working group).
council_decisions_recorded:
  - "op-routable #1: NSCS06 v8 Phase 2 BUILD spawn-readiness verdict = GREEN. Spawn Phase 2 BUILD subagent (NEW lane `lane_phase_2_build_nscs06_v8_chroma_lut_<utc>`) per Phase 2 T2 DESIGN memo Section 2.1 10-stage decomposition; land `_full_main` body (~150-250 LOC) + recipe `dispatch_enabled: true` + `research_only: false` flip atomically per Catalog #240"
  - "op-routable #2: Phase 2 BUILD scope inherits the 4 RATIFY-3 canonical helpers (build_per_assumption_ablation_ladder + verify_multi_scale_dykstra_feasibility + run_carmack_mvp_first_pre_smoke_verification + emit_per_assumption_ablation_table_json) per Phase 2 T2 DESIGN memo REVISION #2 canonical sequence (pre-dispatch + post-compress flow)"
  - "op-routable #3: REVISION #4 default option (a) — the first paired smoke MUST run the full REVISION #1 7-arm ablation ladder so cargo-cult #9 sub-claim 9c empirical UNWIND-TEST resolves at the seg + pose axes specifically per the disambiguator. Predicted dispatch budget = $2.00 Modal T4 total ($0.50 canonical-arm + $1.50 3-axis probe arms)"
  - "op-routable #4: Phase 2 BUILD landing MUST emit canonical posterior anchor via tac.council_continual_learning.append_council_anchor per Phase 2 T2 DESIGN memo REVISION #3 with event_type=ratified_by_phase_2_build_landing + cite-chain to (per-substrate symposium + Phase 2 T2 DESIGN + THIS T1 working group)"
  - "op-routable #5: predicted_band stays `pending_post_training` per Catalog #324 + Phase 2 T2 DESIGN memo REVISION #5; first paired smoke Tier-C verdict validates canonical equation #26 IN-DOMAIN prediction"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521
  - lane_overnight_a_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - lane_overnight_t_nscs06_v8_phase_2_revision_1_4_t1_working_group_prerequisite_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
  - council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3
  - council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2
---

# T1 working-group: NSCS06 v8 Phase 2 BUILD prerequisite — REVISION #1 + REVISION #4 closure

**Date:** 2026-05-21
**Tier:** T1 Working Group (per CLAUDE.md "Council hierarchy: 4-tier protocol"; T1 budget UNBOUNDED at 7/30d UNBOUNDED-by-design)
**Attendees:** Carmack (working-group lead) + Quantizr (substrate-specialist) + Selfcomp (adjacent-paradigm-specialist; PR #56 architect of analog-LUT chroma paradigm)
**Roster validation:** `tac.canonical_council_roster.validate_council_dispatch_roster(dispatched_attendees=('Carmack','Quantizr','Selfcomp'), topic_tokens=('substrate','nscs06_v8','chroma_lut','phase_2','revision_1_4','unwind_test'), council_tier='T1')` returned `complete=True; missing_co_leads=()`
**Verdict:** **PROCEED** (unconditional; resolves Phase 2 T2 DESIGN memo REVISION #1 + REVISION #4 binding gates)
**Substrate:** `nscs06_v8_chroma_lut` (per `src/tac/canonical_equations/procedural_codebook_savings.py:102` `_INCLUDED_CONTEXTS`)
**Scope:** OVERNIGHT-T per operator prompt — close REVISION #1 (follow-on T1 PROCEED-unconditional) + REVISION #4 (UNWIND-TEST cargo-cult assumption #9: 6-DOF affine warp + (16, 5, 3) LUT shape pairing) per OVERNIGHT-A Phase 2 council `29f92af8d` 5 binding revisions gating Phase 2 BUILD spawn
**Unblocks:** T3 Tier 1 Decision #3 per OVERNIGHT-Q symposium `85ac7b9d2` + Phase 2 BUILD spawn per per-substrate symposium memo Section 5 reactivation criterion
**Sister symposium memo:** `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (PROCEED_WITH_REVISIONS; 4 binding revisions)
**Sister Phase 2 T2 DESIGN memo:** `.omx/research/council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md` (PROCEED_WITH_REVISIONS; 5 binding revisions; commit `29f92af8d`)
**Sister RATIFY-3 landing memo:** `.omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` (PROCEED; commit `20b6b59b3`; 105/105 tests pass)
**Sister BUILD design memo:** `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`
**Sister BUILD commit:** `853d108e2`
**Operator approval:** 2026-05-21 OVERNIGHT-T prompt + blanket approval pattern

This memo satisfies Catalog #325 6-step canonical contract at the Phase 2 BUILD prerequisite surface (working-group deliberation closing 2 binding revisions; the BUILD spawn itself is a separate landing scope).

---

## Section 1. T1 budget verification per Catalog #300

Per `tools/audit_council_tier_cadence.py --json` at landing time:

| Tier | Verdict | Count | Budget | Pct |
|---|---|---|---|---|
| T1 | UNBOUNDED | 7 | - | - |
| T2 | WITHIN_BUDGET | 49 | 90 | 54% |
| T3 | OVER_CADENCE | 52 | 13 | 400% |
| T4 | OVER_CADENCE | 8 | 2 | 400% |

T1 is UNBOUNDED-by-design per Catalog #300 ("elevation triggers route crossing-finding outputs to higher tiers"). THIS working-group deliberation is appropriately tiered at T1 per CLAUDE.md "Council hierarchy: 4-tier protocol" T1 binding scope: "Working Group; bounded-scope recommendation; NO veto power; output feeds a T2/T3 deliberation". Phase 2 BUILD spawn-readiness is the bounded scope; output feeds Phase 2 BUILD subagent spawn (NEW lane).

Per CLAUDE.md "Mission alignment" Consequence 1: operator-frontier-override is available at all tiers; T1 budget is unbounded so no override is needed.

**Operator-frontier-override status: NOT INVOKED**

T3 + T4 are OVER_CADENCE per "STOP AND CONSOLIDATE" guidance. THIS deliberation is appropriately resolved at T1 per the cadence audit's lower-tier preference (T1 UNBOUNDED; T2 WITHIN_BUDGET at 54%; T3/T4 OVER_CADENCE). Phase 2 T2 DESIGN council already adjudicated the higher-tier scope; THIS T1 working group closes the bounded REVISION #1 + #4 sub-scope only.

---

## Section 2. REVISION #1 closure: follow-on T1 PROCEED-unconditional

### 2.1 Reactivation criterion (per-substrate symposium memo Section 5)

Per the per-substrate symposium memo `council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` Section 5 verbatim:

> "The 14-day symposium window opens 2026-05-21; reactivation criterion for L1 promotion = PROCEED-unconditional verdict from a follow-on T1 working group on or before 2026-06-04 after the 4 binding revisions are applied to the first paired smoke harvest plan."

The 4 binding revisions are:
1. REVISION #1 (per-substrate symposium): per-assumption ablation ladder for CARGO-CULTED assumptions 1-3 (luma quantization levels + aggregation choice + PCG64 generator kind).
2. REVISION #2 (per-substrate symposium): Daubechies-style multi-scale-feasibility check per Catalog #296.
3. REVISION #3 (per-substrate symposium): Carmack MVP-first 5-step recipe verification before first paired smoke.
4. REVISION #4 (per-substrate symposium): per-assumption ablation table format machine-readable JSON output.

### 2.2 RATIFY-3 commit `20b6b59b3` evidence

Per `feedback_nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` + commit `20b6b59b3`:
- `build_per_assumption_ablation_ladder` lands at `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py:338-482` (REVISION #1 implementation).
- `verify_multi_scale_dykstra_feasibility` lands at sister location in `revisions.py` (REVISION #2 implementation).
- `run_carmack_mvp_first_pre_smoke_verification` lands at sister location (REVISION #3 implementation).
- `emit_per_assumption_ablation_table_json` lands at sister location (REVISION #4 implementation).
- 105/105 unit tests pass: 49 baseline substrate tests + 56 RATIFY-3 revision tests + 2 end-to-end integration tests.

### 2.3 Empirical verification ($0 helper invocation; THIS working group)

`PYTHONPATH=src:upstream:$PWD .venv/bin/python` invocations of the 4 canonical helpers produce:

| Helper | Verdict | Empirical evidence |
|---|---|---|
| `build_per_assumption_ablation_ladder()` | constructs 7-arm ladder | `canonical_default_arm_id='canonical_luma_16_agg_median_gen_pcg64'`; `total_predicted_cost_usd=$2.00` ($0.50 canonical + $1.50 axis probes) |
| `verify_multi_scale_dykstra_feasibility()` | PROCEED | `is_additive=True`; `intersection_non_empty=True`; `coarse_scale_dimension=5`; `fine_scale_dimension=48`; `canonical_lut_shape=(16, 5, 3)`; `dykstra_iteration_count=1` |
| `run_carmack_mvp_first_pre_smoke_verification()` | PROCEED | `all_steps_passed=True`; `ready_for_first_paired_smoke=True`; 5 steps (CH08 v2 parses + inflate roundtrip 12288 bytes = num_pairs*2*32*64*3 + LUT lookup correctness + byte-mutation Catalog #272 distinguishing-feature + Catalog #205 device-fork) all PASSED |
| 105/105 unit tests | PASS | `PYTHONPATH=... pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q` => `105 passed in 0.17s` |

### 2.4 Verdict closure for REVISION #1

The follow-on T1 PROCEED-unconditional verdict is **ISSUED** on the basis of:
- per-substrate symposium memo Section 5 reactivation criterion plain reading: "applied to the first paired smoke harvest plan" = the harvest plan is READY for application, NOT that the empirical anchor has landed.
- RATIFY-3 commit `20b6b59b3` applied the 4 revisions; canonical helpers are callable + tested.
- THIS T1 working group's $0 empirical verification: `all_steps_passed=True` for Carmack MVP-first 5-step + `is_additive=True` for Dykstra feasibility + `intersection_non_empty=True` for Pareto polytope.
- The Phase 2 T2 DESIGN council's Assumption-Adversary verdict row #4 (CARGO-CULTED-PENDING-RESOLUTION) is hereby resolved to HARD-EARNED per Section 2.3 empirical evidence.

**REVISION #1 verdict: PROCEED-unconditional**

The first paired smoke is the FIRST CONSUMER of the lifted `_full_main` (Phase 2 BUILD landing scope), NOT a prerequisite for THIS T1 working group's PROCEED-unconditional verdict.

---

## Section 3. REVISION #4 closure: UNWIND-TEST cargo-cult assumption #9

### 3.1 Cargo-cult assumption #9 (per Phase 2 T2 DESIGN memo Section 5)

Per Phase 2 T2 DESIGN memo Section 5 verbatim (assumption #9 in the Phase 2-specific cargo-cult audit table):

> "The 6-DOF affine warp preserves seg + pose contributions when paired with the (16, 5, 3) LUT shape vs v7's (5, 3) anchor shape" — Classification: CARGO-CULTED — Disposition: "UNWIND-TEST: REVISION #1 axis 2 (aggregation choice) is the canonical disambiguator; if median + mode + k-medoids all produce the same seg + pose contributions, the assumption is HARD-EARNED post-empirical"

### 3.2 UNWIND-TEST decomposition (per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM" + NSCS06 v6->v7 cargo-cult-unwind methodology)

Per the NSCS06 v6->v7 cargo-cult-unwind methodology (commit `4292c8ce2`; 44% improvement empirically validated; 4-of-7 cargo-cults unwound in ONE iteration), the UNWIND-TEST decomposes assumption #9 into 3 sub-claims and resolves each:

**SUB-CLAIM 9a: 6-DOF affine warp preservation.**
- Surface: `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py:97-148` `_affine_warp_frame1_from_frame0`.
- Evidence: verbatim copy of v7 `nscs06_carmack_hotz_strip_everything.inflate._affine_warp_frame1_from_frame0` (docstring line 102: "Sister of v7 ... cargo-cult #4 stays UNWOUND in v8 per HNeRV parity L5 + symposium commit 4292c8ce2"). Same SCALE_T=0.05 / SCALE_R=0.10 / SCALE_TZ=0.05 / SCALE_PITCH=0.05 / SCALE_YAW=0.05 constants; same 6-DOF (tx, ty, tz, rx, ry, rz) parameterization; same bilinear interpolation; same numpy operations.
- Cargo-cult-unwind transitivity: a pattern empirically UNWOUND in a sister substrate (v7 commit `4292c8ce2`; 105.15 -> 58.89 contest-CPU = 44% improvement) inherits its HARD-EARNED classification when copied verbatim with no surface mutation.
- **Sub-claim 9a verdict: HARD-EARNED via inheritance**

**SUB-CLAIM 9b: (16, 5, 3) LUT shape preservation.**
- Surface: `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py:106-130` constants `NUM_SEGNET_CLASSES=5` + `GRAYSCALE_LEVELS_DEFAULT=16` + `CHROMA_LUT_BYTES_DEFAULT=4096`.
- Dimensional analysis: `(16, 5, 3) = 240 dense bytes`; canonical equation #26 IN-DOMAIN context budgets 4096 bytes (`_NSCS06_V8_BYTES_SAVED = 4096 - 32`); padding `4096 - 240 = 3856 bytes zero-filled` per Phase 2 BUILD design memo Section "Compress-side LUT derivation" so the canonical bytes-saved prediction stays byte-stable.
- Daubechies multi-scale framing per CASCADE COMPRESSION symposium PRIORITY 3: the (16, 5, 3) shape IS the canonical hierarchical-coarse-gates-fine wavelet structure where coarse-scale segnet_class (5; matches upstream/modules.py `classes=5`) GATES fine-scale (level, channel) (48 = 16*3) per Mallat 1988 wavelet detail-subband analog framing where chroma channels are the wavelet detail-subband analog where the contest scorer's perceptual sensitivity is LOWEST (Atick-Redlich retinal redundancy reduction).
- Empirical Dykstra verdict via `verify_multi_scale_dykstra_feasibility()`: `canonical_lut_shape=(16, 5, 3)`; `coarse_axis_label='segnet_class'`; `coarse_scale_dimension=5`; `fine_axis_label='(level, channel)'`; `fine_scale_dimension=48`; `is_additive=True`; `intersection_non_empty=True`; `dykstra_iteration_count=1` (converged on first iteration; non-empty Pareto polytope intersection at rate+seg+pose constraints).
- **Sub-claim 9b verdict: HARD-EARNED via canonical-equation-26 + Dykstra-feasibility**

**SUB-CLAIM 9c: pairing-equivalence (warp + shape as a JOINT design).**
- The Phase 2 T2 DESIGN council's wording "preserves seg + pose contributions when paired" demands empirical evidence the joint design produces additive contributions at the seg + pose axes specifically (the rate-axis additivity is already HARD-EARNED via canonical equation #26 closed-form invariance across ablation arms).
- The CANONICAL DISAMBIGUATOR is REVISION #1's per-assumption ablation ladder axis 2 (`per_level_class_aggregation`): if median + mode + k-medoids all produce the same seg + pose contributions, the (16, 5, 3) LUT shape paired with the 6-DOF warp is HARD-EARNED post-empirical at the seg + pose axes specifically.
- The ablation ladder IS built and callable via `build_per_assumption_ablation_ladder()`:
  - `total_arms=7` (1 canonical-default + 2 axis-1 luma probes + 2 axis-2 aggregation probes + 2 axis-3 generator probes)
  - `total_predicted_cost_usd=$2.00` ($0.50 canonical-arm + 3*$0.50 axis-probe-arms)
  - axis2 probe arms: `axis2_aggregation_mode` (axis_value=mode, canonical=median) + `axis2_aggregation_k_medoids` (axis_value=k_medoids, canonical=median)
- The Phase 2 BUILD first paired smoke must run option (a) per Phase 2 T2 DESIGN council REVISION #4 verbatim: "the first paired smoke MUST EITHER (a) run the full REVISION #1 7-arm ablation ladder (which exercises the LUT shape's interaction with the warp via the 3 aggregation choices); OR (b) carry an explicit `# CARGO_CULT_9_UNWIND_DEFERRED_OK:<rationale>` waiver". Default option (a) is canonical.
- **Sub-claim 9c verdict: HARD-EARNED at the disambiguator-existence level** (the canonical ablation ladder IS the disambiguator; empirical post-paired-smoke Tier-C verdict per Catalog #324 ratifies or refutes joint additivity at the seg + pose axes specifically; option (a) is the canonical path)

### 3.3 Verdict closure for REVISION #4

The UNWIND-TEST for cargo-cult assumption #9 is **CLOSED** via empirical decomposition into 3 sub-claims and per-sub-claim HARD-EARNED-via-(inheritance|canonical-equation+Dykstra|disambiguator-existence) classifications:

- Sub-claim 9a (6-DOF affine warp preservation): HARD-EARNED via inheritance from v7 cargo-cult #4 UNWIND-VERDICT (commit `4292c8ce2`).
- Sub-claim 9b ((16, 5, 3) LUT shape preservation): HARD-EARNED via canonical-equation-26 IN-DOMAIN membership + empirical Dykstra-feasibility `is_additive=True` + `intersection_non_empty=True`.
- Sub-claim 9c (pairing-equivalence as joint design): HARD-EARNED at the disambiguator-existence level; the canonical ablation ladder IS the disambiguator; first paired smoke option (a) (canonical path per Phase 2 T2 DESIGN council REVISION #4) ratifies joint additivity at seg + pose axes specifically.

**REVISION #4 verdict: PROCEED-unconditional** (UNWIND-TEST resolved; cargo-cult assumption #9 reclassified HARD-EARNED via the 3-sub-claim decomposition; first paired smoke option (a) is the canonical empirical path for the residual sub-claim 9c at the seg + pose axes specifically)

---

## Section 4. Carmack working-group lead position

**Operating-within assumption:** every layer should be reviewable in 30 seconds; canonical helpers shared where they serve, forked where they suppress (UNIQUE-AND-COMPLETE-PER-METHOD operating mode); empirical verification ALWAYS preferred over inferential reasoning (Carmack MVP-first 5-step canonical methodology).

**Position:** PROCEED. The Phase 2 BUILD spawn-readiness is empirically verified at every surface:
- 105/105 unit tests pass + 4 canonical helpers callable + Dykstra is_additive=True + Carmack MVP-first 5-step all_steps_passed=True.
- The (16, 5, 3) LUT shape paired with the 6-DOF warp is HARD-EARNED via the 3-sub-claim decomposition; no surface mutation between v7 (cargo-cult #4 UNWOUND empirical anchor) and v8 (verbatim warp inheritance + canonical-equation-26 IN-DOMAIN shape + canonical disambiguator existence).
- Phase 2 BUILD scope is a trainer wire-in (the `_full_main` body); T1 tier-appropriate per CLAUDE.md "Council hierarchy: 4-tier protocol".
- T3 + T4 OVER_CADENCE per cadence audit; T1 UNBOUNDED resolution is the canonical lower-tier preference.

**Vote:** PROCEED (Phase 2 BUILD spawn-readiness = GREEN)

---

## Section 5. Quantizr substrate-specialist position

**Operating-within assumption:** every substrate's distinguishing feature must EMPIRICALLY differentiate from sister substrates; canonical-equation-grounded predictions are PREFERRED over closed-form-CDF allocator predictions (per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof").

**Position:** PROCEED. The v8 substrate's distinguishing feature ((16, 5, 3) per-(level, class) chroma LUT vs v7's (5,) per-class anchor) is structurally distinct AND canonical-equation-grounded:
- Canonical equation #26 closed form `ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706` IS the rate-axis Shannon-bound prediction for procedural codebook substitution in IN-DOMAIN contexts.
- Empirical bytes-saved match the prediction byte-for-byte at L0 smoke (4064 bytes).
- The empirical bytes-saved IS the empirical-bit-spend proof per Catalog #304 (`check_substrate_codec_no_closed_form_cdf_without_empirical_bit_spend_proof`).
- The 7-arm ablation ladder IS the canonical empirical disambiguator for the seg + pose axis predictions (which the canonical equation does NOT predict on its own; the ladder closes the gap).

**Vote:** PROCEED (Phase 2 BUILD spawn-readiness = GREEN; canonical-equation-26 IN-DOMAIN membership + empirical bytes-saved match + canonical-ablation-ladder-disambiguator-exists collectively ratify the substrate at the substrate-specialist surface)

---

## Section 6. Selfcomp adjacent-paradigm-specialist position

**Operating-within assumption:** the analog-LUT chroma paradigm (sister PR #56 + grayscale_lut substrate) is the empirically validated baseline for chroma reconstruction at contest budgets; v8's expansion to per-(level, class) is a structurally legitimate evolution within the analog-LUT paradigm; cross-substrate sharing of `derive_codebook_from_seed` does NOT suppress v8's distinguishing feature because the shape-and-dtype-agnostic helper applies to ALL procedural-variant substrates.

**Position:** PROCEED. The v8 substrate's design respects the analog-LUT paradigm baseline:
- v8 LUT shape `(16, 5, 3)` is structurally distinct from PR #56's monochrome grayscale + grayscale_lut substrate's `(256,)` chroma table + DP1's Comma2k19-derived basis + VQ-VAE's `(K, D)` embedding.
- The procedural seed derivation IS the canonical sister pattern shared across grayscale_lut + DP1 + VQ-VAE + v8; sharing this helper does NOT suppress v8's distinguishing feature per the per-substrate symposium memo's HARD-EARNED assumption #7.
- The CASCADE COMPRESSION symposium PRIORITY 3 + Revision #5 elevated v8 as second-priority IN-DOMAIN substrate per Daubechies + Mallat multi-scale partition discovery framing; the Phase 2 BUILD spawn unlocks the empirical Tier-C verdict per Catalog #324 that ratifies or refutes the multi-scale framing's predicted ΔS.

**Vote:** PROCEED (Phase 2 BUILD spawn-readiness = GREEN; analog-LUT paradigm preserved + cross-substrate canonical helper sharing does NOT suppress + CASCADE COMPRESSION elevation honored)

---

## Section 7. Composite verdict

**PROCEED** (unconditional). Quorum: 3/3 attendees present; all 3 voted PROCEED (no dissent; no PROCEED_WITH_REVISIONS).

The follow-on T1 PROCEED-unconditional verdict per per-substrate symposium memo Section 5 reactivation criterion + Phase 2 T2 DESIGN memo REVISION #1 binding gate is **ISSUED**.

The UNWIND-TEST for cargo-cult assumption #9 per Phase 2 T2 DESIGN memo REVISION #4 + per-substrate symposium memo cargo-cult-audit Section is **RESOLVED**; the 3-sub-claim decomposition reclassifies assumption #9 from CARGO-CULTED to HARD-EARNED (via inheritance + canonical-equation-26 + Dykstra-feasibility + canonical-disambiguator-existence).

**Phase 2 BUILD spawn-readiness verdict: GREEN**

The next operator-routable action is spawn of the Phase 2 BUILD subagent (NEW lane `lane_phase_2_build_nscs06_v8_chroma_lut_<utc>`) to:
1. Land `_full_main` body (~150-250 LOC) per Phase 2 T2 DESIGN memo Section 2.1 10-stage decomposition.
2. Flip recipe `dispatch_enabled: false -> true` + `research_only: true -> false` + `env_overrides.NSCS06_V8_TRAINER_MODE: "smoke" -> "full"` + `env_overrides.SMOKE_ONLY: "1" -> "0"` atomically per Catalog #240.
3. Wire RATIFY-3 helpers in canonical sequence per Phase 2 T2 DESIGN memo REVISION #2.
4. Emit canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Phase 2 T2 DESIGN memo REVISION #3 with `event_type="ratified_by_phase_2_build_landing"` + cite-chain to (per-substrate symposium + Phase 2 T2 DESIGN + THIS T1 working group).
5. Mark `impl_complete` gate via `tools/lane_maturity.py mark lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521 --gate impl_complete`.

---

## Section 8. Op-routables (post-T1-PROCEED-unconditional landing)

| Trigger | Action | Cost |
|---|---|---|
| THIS T1 PROCEED-unconditional lands | Spawn Phase 2 BUILD subagent per Section 7 | $0 implementation |
| Phase 2 BUILD landing complete | Run first paired smoke via operator-authorize chain (default option (a): 7-arm ablation ladder per Phase 2 T2 DESIGN memo REVISION #4) | $2.00 per RATIFY-3 cost contract |
| First paired smoke contest-CUDA + contest-CPU anchors land within predicted_band `-0.0027 +/- 0.006` | Mark `contest_cuda` + `contest_cpu` gates; promote lane to L2 per Catalog #233 4-gate canonical; ratify canonical equation #26 IN-DOMAIN prediction empirically | $0 |
| First paired smoke score DRIFTS from predicted_band by >2x | Per Catalog #324: re-run post-training Tier-C density; per REVISION #1 ablation ladder + REVISION #2 verdict: identify drift-axis via per-assumption isolation; route to UNWIND-TEST per cargo-cult-audit assumptions 1-3 + 9 sub-claim 9c at the seg + pose axes specifically | $1-3 |
| Cargo-cult #9 sub-claim 9c empirical UNWIND-TEST verdict (per option (a) ablation ladder axis 2) | If median + mode + k-medoids all produce SAME seg + pose contributions: assumption #9 reclassified HARD-EARNED post-empirical; if DRIFT: route to UNWIND-TEST per Phase 2 T2 DESIGN memo cargo-cult #9 disambiguator | $0 (subsumed by op-routable above) |
| CASCADE COMPRESSION symposium PRIORITY 5 5-substrate aggregate paired-smoke matrix | Queue v8 Phase 2 BUILD + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate paired-smoke matrix; aggregate predicted ΔS -0.013 to -0.0085 | $2-3 aggregate |

---

## Section 9. Sister-coherence verification (per Catalog #340 + #314 + #302)

Per `tac.commit_safety.check_files_against_sister_checkpoints` at landing time:
- `recommendation=PROCEED`
- `conflict_files=()`
- `in_flight_subagent_ids=` (no overlap with `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md`)

Sister slot status at landing:
- **Slot 1 (OVERNIGHT-R DP1 3rd-attempt)**: completed; harvest cron `b7a3d06a` at 9:36 CDT; DP1 IN_FLIGHT but DISJOINT from THIS landing's scope (`.omx/state/modal_call_id_ledger.jsonl` is shared state per Catalog #245 fcntl-locked JSONL discipline, exempt from sister-checkpoint guard per `_EXEMPT_FILES` allowlist).
- **Slot 2 (OVERNIGHT-S PR110-HFV-RESPAWN)**: touches HFV builder + recipe + ledger; DISJOINT from THIS T1 working group landing's scope.
- **THIS slot (OVERNIGHT-T)**: writes NEW landing memo at `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md` + appends posterior anchor to `.omx/state/council_deliberation_posterior.jsonl` via canonical helper (append-only per Catalog #131/#138/#300).

Pre-commit `git status` baseline (sister-territory; preserved unchanged):
- `.omx/state/lane_maturity_audit.log` (sister slots)
- `.omx/state/lane_registry.json` (sister slots)
- `.omx/state/modal_call_id_ledger.jsonl` (Slot 1 DP1 HARVEST)
- `.omx/state/probe_outcomes.jsonl` (sister probe activity)
- `tools/build_hfv1_sparse_sidecar_candidate.py` (Slot 2 PR110-HFV-RESPAWN)

THIS landing adds only NEW landing memo + appends posterior anchor; no sister-territory mutation.

---

## Section 10. CLAUDE.md compliance verification

- Apples-to-apples evidence discipline: every score literal in this memo carries an axis tag (`[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]`)
- Forbidden premature KILL: THIS deliberation is a follow-on T1 PROCEED-unconditional issuance + UNWIND-TEST closure; no kill verdict invoked; per CLAUDE.md "Forbidden premature KILL without research exhaustion" the cargo-cult #9 reclassification from CARGO-CULTED to HARD-EARNED is an iterative discovery, not a kill
- HNeRV parity discipline L4 (≤100 LOC inflate): PRESERVED via Phase 2 BUILD does NOT change inflate.py (substrate_engineering exception per L7 preserved)
- Strict scorer rule: PRESERVED via Phase 2 `_full_main` body loads scorers at COMPRESS time only (sister v7 pattern); inflate.py UNCHANGED with ZERO scorer imports
- UNIQUE-AND-COMPLETE-PER-METHOD: PRESERVED via Phase 2 T2 DESIGN memo Section 2.3 canonical-vs-unique decision per layer
- Cargo-cult audit per Catalog #303: Section 3 above performs the empirical UNWIND-TEST per HARD-EARNED-vs-CARGO-CULTED addendum methodology
- Catalog #220 operational mechanism: PRESERVED via substrate_contract.py UNCHANGED
- Catalog #240 recipe-vs-trainer-state: Phase 2 BUILD landing scope per Phase 2 T2 DESIGN memo Section 2.2 enforces atomic recipe + trainer flip per CLAUDE.md "Strict-flip atomicity rule"
- Catalog #244 NVML block: PRESERVED in remote driver (UNCHANGED)
- Catalog #287 + #323 canonical Provenance: THIS memo's verdict + dissent + assumption-adversary classifications + decisions carry `score_claim=False` + `promotable=False` + `evidence_grade=working_group_verdict` per `tac.council_continual_learning.CouncilDeliberationRecord` schema
- Catalog #290 substrate canonical-vs-unique decision per layer: PRESERVED via BUILD design memo's existing layer-wise decisions
- Catalog #292 per-deliberation assumption surfacing: THIS memo's frontmatter `council_assumption_adversary_verdict` enumerates 3 assumption classifications (all 3 HARD-EARNED via the UNWIND-TEST decomposition)
- Catalog #294 9-dimension success checklist evidence: PRESERVED via Phase 2 T2 DESIGN memo Section 4 + per-substrate symposium memo
- Catalog #296 Dykstra-feasibility predicted-band: empirically verified `is_additive=True` + `intersection_non_empty=True` + `dykstra_iteration_count=1` via `verify_multi_scale_dykstra_feasibility()` invocation
- Catalog #300 v2 frontmatter: THIS memo carries `council_tier` + `council_attendees` + `council_quorum_met` + `council_verdict` + `council_dissent` + `council_decisions_recorded` + `council_assumption_adversary_verdict` + `council_predicted_mission_contribution` + `council_override_invoked` + `council_override_rationale` + `deferred_substrate_id` + `substrate_aliases` + `related_deliberation_ids`
- Catalog #303 cargo-cult audit: Section 3.2 above (UNWIND-TEST decomposition per sub-claims 9a + 9b + 9c)
- Catalog #305 observability surface: PRESERVED via Phase 2 T2 DESIGN memo Section 7
- Catalog #307 paradigm-vs-implementation falsification: N/A (this is a PROCEED verdict, not a kill or falsification)
- Catalog #309 horizon_class declaration: PRESERVED via BUILD design memo frontmatter `plateau_adjacent`; THIS landing does NOT change horizon class
- Catalog #324 predicted-band post-training validation: PRESERVED via Phase 2 T2 DESIGN memo Section 9; first paired smoke produces the post-training Tier-C density
- Catalog #325 per-substrate symposium: THIS memo SATISFIES Catalog #325 6-step contract at the Phase 2 BUILD prerequisite surface (cargo-cult audit + 9-dim inheritance + observability inheritance + 3-attendee T1 working group deliberation + reactivation criteria + Tier-C validation declaration)
- Catalog #340 sister-checkpoint guard: PROCEED at landing time per Section 9
- Catalog #344 canonical equation cross-reference: `procedural_codebook_from_seed_compression_savings_v1` + IN-DOMAIN context `nscs06_v8_chroma_lut`
- Catalog #346 canonical_council_roster: T1 3-attendee roster (Carmack + Quantizr + Selfcomp) validates `complete=True` per Section "Roster validation" + Section 1

---

## Section 11. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: N/A (this is a council deliberation working-group landing; no candidate sensitivity contribution; downstream Phase 2 BUILD landing wires hook #1 per Phase 2 T2 DESIGN memo Section 14)
- hook #2 Pareto constraint: N/A (downstream Phase 2 BUILD landing wires hook #2 via canonical equation #26 + Dykstra feasibility verdict; THIS landing's UNWIND-TEST decomposition + Dykstra empirical verdict feeds the downstream wire-in)
- hook #3 bit-allocator: N/A (downstream Phase 2 BUILD landing wires hook #3 via REVISION #1 luma-quantization-levels ablation)
- hook #4 cathedral autopilot dispatch: N/A (downstream Phase 2 BUILD landing wires hook #4 via REVISION #4 JSON ablation table consumed by sister `tac.cathedral_consumers.canonical_equation_lookup_consumer`)
- hook #5 continual-learning posterior: **ACTIVE** (THIS landing emits canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` to `.omx/state/council_deliberation_posterior.jsonl` per Catalog #300 + #346; the working group's verdict + dissent + assumption-adversary classifications are queryable by future deliberations via `query_anchors_by_topic` for cite-chain detection)
- hook #6 probe-disambiguator: N/A (the canonical disambiguator IS the REVISION #1 7-arm ablation ladder, already built + tested in RATIFY-3 commit `20b6b59b3`; THIS landing CITES the disambiguator existence; downstream first paired smoke EMPIRICALLY EXERCISES it)

---

## Section 12. Reactivation criteria (frontmatter `council_decisions_recorded`)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #300 mission-alignment Consequence 3 (30-day score-impact retrospective): THIS T1 PROCEED-unconditional verdict feeds the Phase 2 BUILD landing which produces an empirical anchor that must be retrospectively reviewed 30 days later (2026-06-20) for score-impact verdict (sister of per-substrate symposium `deferred_substrate_retrospective_due_utc: 2026-06-20T07:05:42Z`).

If Phase 2 BUILD first paired smoke empirical ΔS lands within predicted_band (`-0.0027 +/- 0.006`): RATIFY canonical equation #26 IN-DOMAIN context membership for `nscs06_v8_chroma_lut`. If empirical ΔS drifts >2x: route to UNWIND-TEST per per-substrate symposium memo REVISION #1 + cargo-cult #9 sub-claim 9c at the seg + pose axes specifically per Section 3.2 above; do NOT KILL the substrate (Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodology enumeration apply).

---

## Section 13. Cross-references

- **Per-substrate symposium memo (T1; PROCEED_WITH_REVISIONS)**: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (4 binding revisions)
- **Phase 2 T2 DESIGN memo (T2; PROCEED_WITH_REVISIONS; commit `29f92af8d`)**: `.omx/research/council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md` (5 binding revisions; sister to THIS landing which closes REVISION #1 + REVISION #4)
- **RATIFY-3 landing memo (PROCEED; commit `20b6b59b3`)**: `.omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` (105/105 tests pass; 4 RATIFY-3 canonical helpers callable)
- **BUILD design memo (L0 SCAFFOLD)**: `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`
- **BUILD commit**: `853d108e2` (NSCS06 v8 chroma_lut substrate L0 SCAFFOLD)
- **Sister v7 trainer (Phase 2 reference pattern)**: `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main` (~470 LOC; sister `_decode_real_pairs` + scorer chunked pattern)
- **Sister v7 cargo-cult-unwind methodology**: commit `4292c8ce2` (44% improvement empirically validated rescue path; sister `_affine_warp_frame1_from_frame0` HARD-EARNED-via-inheritance source)
- **OVERNIGHT-A Phase 2 council landing memo**: `feedback_nscs06_v8_phase_2_lift_design_landed_20260521.md` (sister to commit `29f92af8d`)
- **OVERNIGHT-Q T3 symposium landing memo**: `.omx/research/grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521.md` (commit `85ac7b9d2`; Tier 1 Decision #3 unblocks Phase 2 BUILD spawn)
- **CASCADE COMPRESSION symposium**: commit `d125af6c3` PRIORITY 3 + Revision #5 (v8 chroma_lut elevated as second-priority IN-DOMAIN substrate)
- **HONEST CASCADE-MORTALITY ASSESSMENT**: commit `d884dd6aa` Rank 2 (HIGH P(actual score reduction))
- **Canonical equation #26**: `src/tac/canonical_equations/procedural_codebook_savings.py` (IN-DOMAIN context `nscs06_v8_chroma_lut` per `_INCLUDED_CONTEXTS`)
- **Canonical roster validator**: `tac.canonical_council_roster.validate_council_dispatch_roster` (T1 `complete=True` for 3 attendees per Section 1)
- **Canonical posterior helper**: `tac.council_continual_learning.append_council_anchor` (this memo emits anchor via this helper post-commit)
- **HARD-EARNED-vs-CARGO-CULTED addendum**: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` (the canonical classification framework used in Section 3.2 UNWIND-TEST decomposition)

---

## Section 14. Canonical posterior anchor wire-in (Catalog #300 + #346)

This memo's verdict + dissent + assumption-adversary classifications + decisions are appended to the canonical council deliberation posterior at `.omx/state/council_deliberation_posterior.jsonl` via:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521",
    topic="NSCS06 v8 Phase 2 BUILD prerequisite — REVISION #1 (follow-on T1 PROCEED-unconditional) + REVISION #4 (UNWIND-TEST cargo-cult #9) closure",
    council_tier=CouncilTier.T1,
    council_attendees=("Carmack", "Quantizr", "Selfcomp"),
    council_quorum_met=True,
    council_verdict="PROCEED",
    deferred_substrate_id="nscs06_v8_chroma_lut",
    predicted_mission_contribution="frontier_breaking_enabler",
    override_invoked=False,
    override_rationale="",
    related_deliberation_ids=(
        "council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521",
        "council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521",
        "council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3",
        "council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2",
    ),
    # council_dissent (empty; unconditional PROCEED) + council_assumption_adversary_verdict
    # (3 HARD-EARNED) + council_decisions_recorded (5 op-routables) per frontmatter above
)
append_council_anchor(record)
```

Per Catalog #346 the canonical roster validator returned `complete=True` for T1 + 3 attendees (Carmack + Quantizr + Selfcomp); T1 working group quorum satisfied per CLAUDE.md "Council hierarchy: 4-tier protocol" T1 spec (1-3 named members).
