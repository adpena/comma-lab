---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Quantizr
  - Selfcomp
  - MacKay
  - Balle
  - PR95Author
  - Carmack
  - Hotz
  - Mallat
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: |
      PROCEED on DESIGN-only scope is fine. PROCEED on lifting
      NotImplementedError requires (a) sister BUILD commit 853d108e2 +
      RATIFY-3 binding revisions commit 20b6b59b3 are PROVEN to bind to a
      real compress path AGAINST upstream/videos/0.mkv, not the synthetic
      4-pair smoke shape; (b) the PROCEED-unconditional follow-on T1
      working group per per-substrate symposium memo Section 5 has
      actually returned PROCEED-unconditional. Both prerequisites are
      DEFER-pending. Issuing PROCEED-unconditional on Phase 2 BUILD scope
      from this T2 design symposium short-circuits the per-substrate
      symposium's own reactivation criterion ("PROCEED-unconditional
      verdict from a follow-on T1 working group on or before 2026-06-04
      after the 4 binding revisions are applied"). The 4 revisions are
      applied; the follow-on T1 PROCEED-unconditional has NOT yet been
      issued. Revision #1 below makes this prerequisite a binding gate.
  - member: Assumption-Adversary
    verbatim: |
      The Phase 2 BUILD scope as drafted carries 3 NEW operating-within
      assumptions distinct from the BUILD + symposium + RATIFY-3 prior
      assumptions: (i) "the closed-form chroma LUT derivation from
      upstream/videos/0.mkv ground-truth pixels produces the SAME bytes
      every run" — HARD-EARNED (sister v7 build_chroma_palette is
      deterministic; build_chroma_lut_from_ground_truth uses per-bin
      median which is order-independent); (ii) "the 6-DOF affine warp
      preserves seg + pose contributions when paired with the (16, 5, 3)
      LUT shape vs v7's (5, 3) anchor shape" — CARGO-CULTED (no empirical
      evidence either way; UNWIND-TEST per Revision #4 below); (iii) "the
      Modal T4 dispatch wall-clock is bounded by the v7 compress-only
      pattern (~5-10 minutes for 600 pairs)" — HARD-EARNED (v7 sister
      trainer is the canonical reference; same scorer chunked pattern +
      same numpy aggregation; v8 adds per-bin LUT median computation
      which is sub-second per pair empirically).
council_assumption_adversary_verdict:
  - assumption: "The Phase 2 BUILD trainer can land WITHOUT touching the 5 canonical substrate files (architecture/archive/inflate/procedural_variant/substrate_contract.py)"
    classification: HARD-EARNED
    rationale: |
      RATIFY-3 commit 20b6b59b3 empirically demonstrated the pattern:
      revisions.py landed as NEW module + tests/test_revisions.py NEW
      file + only __init__.py re-exports appended. Phase 2 BUILD scope
      MUST follow the same discipline — the trainer's _full_main lifts
      via NEW logic in experiments/train_substrate_nscs06_v8_chroma_lut.py
      (replacing the NotImplementedError raise) + the canonical helpers
      in revisions.py + architecture.py + archive.py + inflate.py stay
      unchanged. Catalog #220 operational mechanism declaration +
      Catalog #241/#242 SubstrateContract canonical contract preserved.
  - assumption: "The recipe `dispatch_enabled: true` flip is a SEPARATE landing scope from the trainer `_full_main` lift"
    classification: HARD-EARNED
    rationale: |
      Per Catalog #240 recipe-vs-trainer-state consistency: the recipe
      `research_only: true` + `dispatch_enabled: false` is paired with
      the trainer's `_full_main` raising NotImplementedError. Lifting
      ONE without the OTHER breaks the consistency invariant. The
      symposium's reactivation criterion gates BOTH flips together via
      the PROCEED-unconditional verdict, not independently. The Phase 2
      BUILD landing scope is the trainer `_full_main` body + recipe flip
      atomically in the same commit batch per CLAUDE.md "Strict-flip
      atomicity rule" applied to recipe-vs-trainer-state.
  - assumption: "The 4 binding revisions per symposium memo Section 4 are sufficient pre-smoke-harvest-plan discipline"
    classification: HARD-EARNED
    rationale: |
      Per RATIFY-3 landing 105/105 tests pass + 56 new revision tests +
      2 end-to-end integration tests. The 4 revisions cover: REVISION #1
      per-assumption ablation ladder (7 arms) + REVISION #2 multi-scale
      Dykstra-feasibility check + REVISION #3 Carmack MVP-first 5-step
      verification + REVISION #4 JSON ablation table emitter. Each
      revision has its dedicated test class + the integration test
      verifies the 4 revisions compose end-to-end. The pre-smoke harvest
      plan is ready; Phase 2 BUILD wires it into the trainer's
      `_full_main` body.
  - assumption: "The follow-on T1 working group PROCEED-unconditional verdict can be issued WITHOUT a paid Modal T4 dispatch"
    classification: CARGO-CULTED
    rationale: |
      The symposium memo Section 5 says "PROCEED-unconditional verdict
      from a follow-on T1 working group on or before 2026-06-04 after
      the 4 binding revisions are applied to the first paired smoke
      harvest plan". The plain reading is "applied to the first paired
      smoke harvest plan" = the harvest plan is READY for application,
      NOT that the empirical anchor has landed. RATIFY-3 satisfies this
      via the canonical helpers being callable + the 105/105 test
      verification. UNWIND-TEST: the follow-on T1 PROCEED-unconditional
      MAY be issued on the basis of RATIFY-3 + this Phase 2 DESIGN memo
      sufficiency review WITHOUT requiring a paid Modal T4 anchor first.
      The first paired smoke is the FIRST CONSUMER of the lifted
      `_full_main`, not a prerequisite. Revision #1 below makes this
      explicit.
  - assumption: "The Phase 2 BUILD scope is a Phase 2 DESIGN-only deliverable from THIS T2 symposium; the BUILD landing is a SEPARATE wave"
    classification: HARD-EARNED
    rationale: |
      Per operator prompt OVERNIGHT-A verbatim: "Prepare Phase 2
      DESIGN-only memo proposing the lift of `_full_main raises
      NotImplementedError` per Catalog #240". The DESIGN scope is THIS
      memo + sister substrate_contract.py NOT mutated. The BUILD wave
      is a follow-on subagent landing scope per Catalog #325 14-day
      window opening 2026-06-04 (when the per-substrate symposium
      window CLOSES) OR after the follow-on T1 PROCEED-unconditional
      verdict per Revision #1, whichever first.
  - assumption: "T2 tier is appropriate for the Phase 2 DESIGN scope; T3 elevation is NOT required"
    classification: HARD-EARNED
    rationale: |
      Per CLAUDE.md "Council hierarchy: 4-tier protocol":
      T2 binding scope = "in-flight engineering tradeoffs; loss-function
      choices; architecture parameters; trainer wire-ins". Phase 2
      BUILD scope is a trainer wire-in (the `_full_main` body wires the
      4 canonical revisions + canonical compress helpers per the BUILD
      design memo's existing canonical-vs-unique decision per layer).
      T3 binding scope = "CLAUDE.md non-negotiable additions/amendments;
      cross-cutting wire-ins". This memo does NOT add a CLAUDE.md
      non-negotiable; the per-substrate symposium IS the non-negotiable
      gate (Catalog #325) and was already executed at T1. Lower-tier
      resolution is preferred per the audit's STOP-AND-CONSOLIDATE
      guidance (T3 OVER_CADENCE at 392%; T4 OVER_CADENCE at 400%; T2
      WITHIN_BUDGET at 53%).
council_decisions_recorded:
  - "op-routable #1: Phase 2 BUILD landing is GATED on follow-on T1 PROCEED-unconditional verdict per per-substrate symposium memo Section 5 reactivation criterion; this T2 DESIGN-symposium-PROCEED does NOT short-circuit that gate"
  - "op-routable #2: Phase 2 BUILD landing scope = NEW logic in experiments/train_substrate_nscs06_v8_chroma_lut.py `_full_main` body (~150-250 LOC) + recipe `dispatch_enabled: true` flip + `research_only: false` flip atomically per Catalog #240 + lane registry `impl_complete` gate marking"
  - "op-routable #3: Phase 2 BUILD landing MUST preserve the 5 canonical substrate files unchanged per Catalog #220 + #241/#242 + sister RATIFY-3 commit 20b6b59b3 pattern"
  - "op-routable #4: Phase 2 BUILD MUST wire the 4 RATIFY-3 canonical helpers (build_per_assumption_ablation_ladder + verify_multi_scale_dykstra_feasibility + run_carmack_mvp_first_pre_smoke_verification + emit_per_assumption_ablation_table_json) into `_full_main` pre-dispatch + post-compress flow"
  - "op-routable #5: predicted_band stays `pending_post_training` per Catalog #324; the first paired smoke harvest produces the post-training Tier-C density that validates the canonical equation #26 IN-DOMAIN prediction empirically"
  - "op-routable #6: Phase 2 BUILD predicted dispatch budget = `$0.50` baseline + `$1.50` 3-axis ablation ladder = `$2.00` total Modal T4 per RATIFY-3 symposium cost contract; sister CASCADE COMPRESSION 5-substrate aggregate paired-smoke matrix queues v8 + grayscale_lut + DP1 + VQ-VAE + ATW V2 at `$2-3` aggregate per PRIORITY 5"
  - "op-routable #7: REVISION #4 (Assumption-Adversary): the Phase 2 BUILD MUST include a UNWIND-TEST per cargo-cult-audit assumption #ii (6-DOF affine warp + (16, 5, 3) LUT shape pairing) — the per-assumption ablation ladder axis 2 (aggregation choice) is the canonical disambiguator; REVISION #1 of the per-substrate symposium already queues this"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut
deferred_substrate_retrospective_due_utc: 2026-06-20T07:05:42Z
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521
  - lane_overnight_a_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
  - council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3
  - council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2
---

# T2 Phase 2 DESIGN-only symposium: NSCS06 v8 chroma-LUT lift of `_full_main raises NotImplementedError`

**Date:** 2026-05-21
**Tier:** T2 Inner-Skunkworks (per CLAUDE.md "Council hierarchy: 4-tier protocol")
**Attendees:** 16 (4 co-leads + 6 inner-council sister + 6 specialist Carmack/Hotz/Mallat + grand council Quantizr/Selfcomp/MacKay/Balle/PR95Author)
**Roster validation:** `tac.canonical_council_roster.validate_council_dispatch_roster` returned `complete=True` for T2 with topic tokens `["substrate", "chroma_lut", "nscs06_v8", "phase_2", "lift_notimplementederror"]` per Catalog #346
**Verdict:** PROCEED_WITH_REVISIONS (5 binding revisions; see Section 6)
**Substrate:** `nscs06_v8_chroma_lut` (per `src/tac/canonical_equations/procedural_codebook_savings.py:102` `_INCLUDED_CONTEXTS`)
**Scope:** DESIGN-ONLY — does NOT mutate `substrate_contract.py` / trainer `_full_main` / recipe YAML
**Sister symposium memo:** `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (PROCEED_WITH_REVISIONS)
**Sister BUILD design memo:** `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`
**Sister RATIFY-3 landing memo:** `.omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` (commit `20b6b59b3`)
**Sister BUILD commit:** `853d108e2`
**Operator approval:** 2026-05-21 blanket approval (operator NON-NEGOTIABLE prompt OVERNIGHT-A)

This memo satisfies Catalog #325 6-step canonical contract at the Phase 2 DESIGN surface:

1. **Cargo-cult audit per Catalog #303** — see frontmatter `council_assumption_adversary_verdict` + Section 5 below
2. **9-dimension success checklist evidence per Catalog #294** — Section 4 below
3. **Observability surface declaration per Catalog #305** — Section 7 below
4. **T2 working-group deliberation** with all 16 attendees (4 co-leads + 12 inner-council sister + grand council) per Catalog #346 canonical roster
5. **Per-substrate Phase 2 reactivation criteria** — see frontmatter `council_decisions_recorded`
6. **Catalog #324 post-training Tier-C validation discipline** — Section 9 below

---

## Section 1. Operator-frontier-override status

Per the cadence audit at landing time:

| Tier | Verdict | Count | Budget | Pct |
|---|---|---|---|---|
| T1 | UNBOUNDED | 6 | - | - |
| T2 | WITHIN_BUDGET | 48 | 90 | 53% |
| T3 | **OVER_CADENCE** | 51 | 13 | **392%** |
| T4 | **OVER_CADENCE** | 8 | 2 | **400%** |

T3 + T4 OVER_CADENCE. Per CLAUDE.md "Council hierarchy: 4-tier protocol" guidance: STOP AND CONSOLIDATE; review whether a deliberation could be resolved at a LOWER tier. THIS deliberation is resolved at **T2** (within-budget) per CLAUDE.md T2 binding scope: "in-flight engineering tradeoffs; loss-function choices; architecture parameters; trainer wire-ins". Phase 2 BUILD scope is a trainer wire-in (the `_full_main` body); T3 elevation is NOT required.

**Operator-frontier-override status: NOT INVOKED**

Per CLAUDE.md "Mission alignment" Consequence 1: operator-frontier-override is available at all tiers, REQUIRES operator-verbatim quote in `council_override_rationale`, BYPASSES quorum + tie-break + recusal for the specific decision, PRESERVES maximum-signal preservation. THIS deliberation has full quorum (16 attendees; T2 sextet pact + sister + grand council; `validate_council_dispatch_roster` returned `complete=True`); override is NOT needed.

**operator approval recorded:** "OVERNIGHT-A: Convene NSCS06 v8 Phase 2 council symposium per Catalog #325 6-step contract per operator blanket approval 2026-05-21 (2nd round)" — this is operator authorization to CONVENE the symposium at T2 tier; it is NOT an operator-frontier-override of the per-substrate symposium's PROCEED-unconditional gate (which remains in force per Revision #1 below).

---

## Section 2. Phase 2 BUILD scope proposal

### 2.1 What `_full_main` needs to implement

The current `_full_main` raises NotImplementedError. Lifting it requires implementing the full compress + auth-eval flow that mirrors the sister NSCS06 v7 trainer pattern (`experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main` lines 574-...). The Phase 2 BUILD body, per the BUILD design memo's existing architecture + sister v7 reference, decomposes into 10 stages:

| Stage | What | Source of canonical pattern |
|---|---|---|
| 1 | seed pin + device-or-die + output_dir mkdir | sister v7 `_full_main` lines 605-609; canonical helper `tac.substrates._shared.trainer_skeleton.device_or_die` |
| 2 | upstream yuv6 patch + scorer load (compress-side ONLY) | sister v7 `_full_main` lines 615-628; canonical helpers `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` + `tac.scorer.load_differentiable_scorers` |
| 3 | decode real pairs from `upstream/videos/0.mkv` | sister v7 `_full_main` lines 631-637; pyav-based pair decode (sister `_decode_real_pairs`) |
| 4 | per-pixel SegNet argmax (chunked per Catalog #218 OOM fix) | sister v7 `_full_main` lines 675-706; chunk_size default 8 per Catalog #218 |
| 5 | per-pixel grayscale quantization at full resolution | NEW for v8: full-res grayscale (NOT lowres like v7) to feed the per-(level, class) LUT lookup |
| 6 | build chroma LUT via canonical `build_chroma_lut_from_ground_truth` | NEW for v8: per-bin median over (level, class) bins from GT pixels per BUILD design memo Section "Compress-side LUT derivation" |
| 7 | PoseNet at compress-side (chunked) | sister v7 `_full_main` lines 770-789; canonical Hydra dict `out["pose"][..., :POSE_DIMS]` slice |
| 8 | invoke RATIFY-3 canonical helpers pre-pack | new for v8 Phase 2: REVISION #3 5-step pre-smoke verification + REVISION #1 ablation ladder construction |
| 9 | pack CH08 v2 archive via canonical `pack_archive(chroma_seed=...)` | EXISTING canonical helper in `tac.substrates.nscs06_v8_chroma_lut.archive` |
| 10 | invoke canonical auth-eval helper `gate_auth_eval_call` + emit RATIFY-3 JSON ablation table per REVISION #4 | sister v7 + RATIFY-3 helpers |

**Estimated _full_main body LOC: ~150-250** (sister v7 _full_main is ~470 LOC; v8 is slimmer because no chroma_seed_mode branch, no class_label arith encoding (the v8 LUT directly indexes by class), no per-class anchor derivation; the (level, class) median aggregation is sub-second per pair so no chunking needed for stage 6).

### 2.2 Recipe `dispatch_enabled` flip per Catalog #240

Atomically with the trainer `_full_main` lift, the recipe `substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` flips:

- `dispatch_enabled: false` -> `dispatch_enabled: true`
- `research_only: true` -> `research_only: false`
- `env_overrides.NSCS06_V8_TRAINER_MODE: "smoke"` -> `"full"`
- `env_overrides.SMOKE_ONLY: "1"` -> `"0"`
- `dispatch_blockers`: remove `per_substrate_symposium_pending_per_catalog_325_window_20260521_to_20260604` and `canonical_equation_26_predicted_band_post_training_tier_c_validation_required_per_catalog_324` (per Catalog #324 the `predicted_band_validation_status: pending_post_training` STAYS — the first paired smoke is the post-training validator)

Per Catalog #240 recipe-vs-trainer-state consistency: BOTH flips MUST land in the same commit batch per CLAUDE.md "Strict-flip atomicity rule" applied at the recipe-vs-trainer-state surface.

### 2.3 What does NOT change

Per RATIFY-3 commit `20b6b59b3` discipline + Catalog #220 + #241/#242 invariants:

- `architecture.py` UNCHANGED (CH08 grammar + chroma LUT shape + build_chroma_lut_from_ground_truth + lookup_rgb_via_chroma_lut)
- `archive.py` UNCHANGED (CH08 v1 + v2 pack/parse; 4064-byte exact-match invariant preserved)
- `inflate.py` UNCHANGED (~120 LOC; numpy + Pillow only; substrate_engineering exception per HNeRV L7)
- `procedural_variant.py` UNCHANGED (canonical equation #26 IN-DOMAIN context + predicted_delta_s)
- `substrate_contract.py` UNCHANGED (Catalog #241/#242 SubstrateContract canonical contract)
- `revisions.py` UNCHANGED (1010 LOC; 4 RATIFY-3 canonical helpers callable from Phase 2 `_full_main`)
- `tests/test_substrate.py` UNCHANGED (49 baseline tests preserved)
- `tests/test_revisions.py` UNCHANGED (56 RATIFY-3 tests preserved)
- driver `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh` UNCHANGED (Catalog #244 NVML block preserved)

The Phase 2 BUILD landing changes:
- `experiments/train_substrate_nscs06_v8_chroma_lut.py` `_full_main` body (NotImplementedError -> ~150-250 LOC)
- `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` (4 fields flip atomically)
- NEW `experiments/train_substrate_nscs06_v8_chroma_lut.py` test additions for `_full_main` smoke (Catalog #229 premise verification)
- `tools/lane_maturity.py mark lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521 --gate impl_complete --evidence "<path>"`

---

## Section 3. Predicted_band per Catalog #324

### 3.1 Rate-axis prediction (HARD-EARNED)

Canonical equation #26 closed form `ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706` for the rate-axis only. EMPIRICAL bytes-saved match the prediction byte-for-byte at L0 smoke (4064 bytes); see RATIFY-3 landing memo Section "test result verification".

### 3.2 Seg-axis + pose-axis prediction (PENDING POST-TRAINING per Catalog #324)

The canonical equation #26 closed form does NOT predict seg or pose impact of the LUT-shape choice. The Phase 2 BUILD first paired smoke produces the post-training Tier-C density via `tools/mdl_scorer_conditional_ablation.py --tier c` against the landed paired smoke archive sha. Per Catalog #324: `predicted_band_validation_status: pending_post_training` STAYS in the recipe through Phase 2 BUILD landing; the post-training Tier-C measurement validates whether the (16, 5, 3) LUT shape's perceptual reconstruction matches v7's per-class anchor or drifts.

### 3.3 Composite predicted_band

| Axis | Predicted | Confidence | Mechanism |
|---|---|---|---|
| rate | `-0.002706 [prediction; canonical-equation-26-grounded]` | HARD-EARNED | canonical equation #26 IN-DOMAIN `nscs06_v8_chroma_lut`; 4064-byte exact empirical match at L0 smoke |
| seg | `[0.0, +0.005] [prediction; per-substrate-symposium-pending]` | CARGO-CULTED | The (16, 5, 3) LUT shape has MORE chroma capacity than v7's (5, 3) anchor; seg should NOT regress vs v7. UNWIND-TEST: first paired smoke + Tier-C |
| pose | `[0.0, +0.001] [prediction; per-substrate-symposium-pending]` | CARGO-CULTED | The 6-DOF affine warp is sister to v7 (cargo-cult #4 UNWOUND); pose should NOT regress. UNWIND-TEST: first paired smoke + Tier-C |
| **composite** | `-0.0027 +/- 0.006 [prediction; per-substrate-symposium-pending]` | LOW (CARGO-CULTED dominated) | sum of axis predictions; the wide CI reflects the CARGO-CULTED seg + pose components |

The wide composite CI reflects the operator-routable risk: if the empirical anchor lands within `[-0.009, +0.003]`, the canonical equation #26 IN-DOMAIN prediction is ratified; if it drifts >2x outside (>= `+0.012` or <= `-0.018`), the REVISION #1 7-arm ablation ladder per RATIFY-3 isolates the drift-axis per Catalog #324.

---

## Section 4. 9-dimension success checklist evidence

## 9-dimension success checklist evidence (per Catalog #294)

Per Catalog #294: the Phase 2 BUILD scope inherits the BUILD design memo's 9-dim evidence. Phase 2 BUILD additions:

| Dim | Status | Phase 2 BUILD evidence |
|---|---|---|
| 1. UNIQUENESS | PARTIAL (same as L0) | v8 is a refinement-class substrate; strictly more capable chroma reconstruction than v7 but operates within the chroma_lut_replacement axis manifold; Phase 2 BUILD does NOT change the uniqueness class |
| 2. BEAUTY + ELEGANCE | PASS | Phase 2 `_full_main` body ~150-250 LOC; reviewable in 30s; sister v7 trainer is the canonical reference pattern |
| 3. DISTINCTNESS | PASS (same as L0) | distinct from v7 + grayscale_lut + DP1 + VQ-VAE |
| 4. RIGOR | PASS | Phase 2 BUILD inherits RATIFY-3 105/105 tests + 4 canonical helpers + 2 end-to-end integration tests; Phase 2 BUILD adds NEW `_full_main` smoke test (real pyav decode + LUT derivation + pack); follow-on T1 PROCEED-unconditional review per Revision #1 |
| 5. OPTIMIZATION PER TECHNIQUE | PASS | Phase 2 BUILD canonical-vs-unique decision per layer: trainer body UNIQUE (sister v7 reference + RATIFY-3 helper invocations); canonical helpers ADOPT (scorer load + auth eval + device + NVML + mount) |
| 6. STACK-OF-STACKS COMPOSABILITY | PASS | Phase 2 BUILD preserves canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` membership; rate-axis ΔS stacks additively with sister procedural-variant substrates per CASCADE COMPRESSION symposium PRIORITY 5 |
| 7. DETERMINISTIC REPRODUCIBILITY | PASS | Phase 2 BUILD preserves byte-stable archive pack/parse roundtrip; seed-pinned via canonical `_pin_seeds` helper; numpy seed-pinned; CH08 grammar fixed at design-time |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | PASS | Phase 2 BUILD per-bin LUT median is O(N*H*W) closed-form; SegNet + PoseNet chunked per Catalog #218 OOM fix (sister v7 pattern); Modal T4 dispatch budgeted at ~5-10 minutes wall-clock (sister v7 reference) |
| 9. OPTIMAL MINIMAL CONTEST SCORE | DEFERRED-PENDING-FIRST-PAIRED-SMOKE | predicted ΔS = -0.0027 +/- 0.006; first paired smoke empirical anchor per Section 3 + Catalog #324 |

---

## Section 5. Cargo-cult audit per assumption

## Cargo-cult audit per assumption (per Catalog #303)

Per the per-substrate symposium memo's existing 7-assumption audit + 3 NEW Phase 2-specific assumptions per the Assumption-Adversary dissent above:

| # | Assumption | Classification | Phase 2 disposition |
|---|---|---|---|
| 1-7 | (inherit from per-substrate symposium memo) | (3 CARGO-CULTED queued for UNWIND-TEST per REVISION #1; 4 HARD-EARNED) | preserved; first paired smoke is the empirical test per Section 3 |
| 8 | The closed-form chroma LUT derivation from upstream/videos/0.mkv ground-truth pixels produces the SAME bytes every run | HARD-EARNED | per-bin median is order-independent; sister v7 build_chroma_palette is deterministic; numpy seed-pinning preserved |
| 9 | The 6-DOF affine warp preserves seg + pose contributions when paired with the (16, 5, 3) LUT shape vs v7's (5, 3) anchor shape | CARGO-CULTED | UNWIND-TEST: REVISION #1 axis 2 (aggregation choice) is the canonical disambiguator; if median + mode + k-medoids all produce the same seg + pose contributions, the assumption is HARD-EARNED post-empirical |
| 10 | The Modal T4 dispatch wall-clock is bounded by the v7 compress-only pattern (~5-10 minutes for 600 pairs) | HARD-EARNED | v7 sister trainer is the canonical reference; same scorer chunked pattern + same numpy aggregation; v8 adds per-bin LUT median which is sub-second per pair empirically |

3 NEW assumptions enumerated (1 CARGO-CULTED queued for UNWIND-TEST + 2 HARD-EARNED). No silent assumptions admitted.

---

## Section 6. Binding revisions (5)

### REVISION #1 (Contrarian + Assumption-Adversary): Phase 2 BUILD landing is GATED on follow-on T1 PROCEED-unconditional

The per-substrate symposium memo Section 5 reactivation criterion is: "PROCEED-unconditional verdict from a follow-on T1 working group on or before 2026-06-04 after the 4 binding revisions are applied to the first paired smoke harvest plan". RATIFY-3 commit `20b6b59b3` applies the 4 revisions; the follow-on T1 PROCEED-unconditional has NOT yet been issued. Phase 2 BUILD landing MUST be gated on the follow-on T1 PROCEED-unconditional verdict being issued; this T2 DESIGN-symposium-PROCEED does NOT short-circuit that gate. The follow-on T1 working group reviews: (a) the BUILD design memo's existing canonical-vs-unique decision per layer; (b) RATIFY-3 commit `20b6b59b3` 4 binding revisions applied; (c) THIS T2 Phase 2 DESIGN memo; (d) the 105/105 + 49 baseline + 56 RATIFY-3 + 2 end-to-end test verification.

### REVISION #2 (Quantizr + Selfcomp): Phase 2 BUILD MUST wire RATIFY-3 helpers pre-dispatch

The Phase 2 `_full_main` body MUST invoke the 4 RATIFY-3 canonical helpers in the canonical sequence:

1. `tac.substrates.nscs06_v8_chroma_lut.revisions.run_carmack_mvp_first_pre_smoke_verification(...)` — BEFORE firing the paid GPU meter on Modal T4; verifies the 5 invariants (CH08 v2 parses cleanly + inflate roundtrip + LUT lookup correctness + byte-mutation smoke + Catalog #205 device-fork)
2. `tac.substrates.nscs06_v8_chroma_lut.revisions.build_per_assumption_ablation_ladder(...)` — constructs the 7-arm ablation matrix per REVISION #1 of the per-substrate symposium memo
3. `tac.substrates.nscs06_v8_chroma_lut.revisions.verify_multi_scale_dykstra_feasibility(...)` — verifies the wavelet-style hierarchical-coarse-gates-fine LUT structure produces additive seg + pose contributions per Catalog #296
4. (post-compress + post-pack + post-auth-eval) `tac.substrates.nscs06_v8_chroma_lut.revisions.emit_per_assumption_ablation_table_json(...)` — emits the canonical JSON ablation table to `<repo>/.omx/state/nscs06_v8_per_assumption_ablation/...` per REVISION #4 of the per-substrate symposium memo

The first paired smoke verifies the helpers actually fire in the canonical sequence (Carmack MVP-first 5-step verification before paid GPU meter starts; if Step (e) fails, exit non-zero BEFORE dispatch).

### REVISION #3 (Daubechies + Mallat CO-LEAD): Phase 2 BUILD MUST emit per-substrate symposium ratification anchor to canonical posterior

Per Catalog #300 + #344: Phase 2 BUILD landing emits a canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` with `event_type="ratified_by_phase_2_build_landing"` + cross-reference to THIS T2 Phase 2 DESIGN memo + the follow-on T1 PROCEED-unconditional memo (per Revision #1) + RATIFY-3 commit `20b6b59b3`. The cite-chain enables future deliberations to trace the Phase 2 ratification provenance.

### REVISION #4 (Carmack + Hotz + Assumption-Adversary): Phase 2 BUILD first paired smoke MUST run with UNWIND-TEST per cargo-cult-audit assumption #9

Per the Assumption-Adversary dissent: the 6-DOF affine warp + (16, 5, 3) LUT shape pairing is CARGO-CULTED. The first paired smoke MUST EITHER (a) run the full REVISION #1 7-arm ablation ladder (which exercises the LUT shape's interaction with the warp via the 3 aggregation choices); OR (b) carry an explicit `# CARGO_CULT_9_UNWIND_DEFERRED_OK:<rationale>` waiver in the dispatch plan body documenting why the empirical isolation is deferred. Default: option (a) is the canonical path per REVISION #1 of the per-substrate symposium memo.

### REVISION #5 (Mallat + Balle + MacKay + PR95Author): Phase 2 BUILD predicted_band stays `pending_post_training` per Catalog #324

Per Catalog #324: the recipe `predicted_band_validation_status: pending_post_training` STAYS through Phase 2 BUILD landing. The first paired smoke harvest produces the post-training Tier-C density via `tools/mdl_scorer_conditional_ablation.py --tier c` against the landed paired smoke archive sha. The empirical Tier-C verdict validates whether the canonical equation #26 IN-DOMAIN prediction is ratified or refuted. Per CLAUDE.md "Apples-to-apples evidence discipline": no score CLAIM is asserted until the Tier-C verdict lands.

---

## Section 7. Observability surface

## Observability surface (per Catalog #305)

Phase 2 BUILD inherits the BUILD design memo's observability surface declaration. Phase 2-specific additions:

1. **Inspectable per layer (Phase 2 additions):**
   - The 10 Phase 2 `_full_main` stages are observability-named (`_stage("seed_pinned")`, `_stage("scorers_loaded_compress_side")`, etc.) per sister v7 trainer pattern; the `stage_log` is queryable from the smoke metadata JSON.
   - Each RATIFY-3 helper invocation surfaces a typed verdict dataclass (`CarmackMvpFirstPreSmokeVerificationVerdict`, `PerAssumptionAblationLadder`, `MultiScaleDykstraFeasibilityVerdict`) included in the JSON ablation table per REVISION #4.
2. **Decomposable per signal (Phase 2 additions):**
   - The Phase 2 `_full_main` emits per-stage wall-clock + memory footprint to `stage_log[*]["elapsed_seconds"]` + `peak_memory_bytes`.
   - The canonical equation #26 IN-DOMAIN context `nscs06_v8_chroma_lut` rate-axis prediction decomposes per-axis into `rate_contrib` + `seg_contrib` + `pose_contrib` in the post-compress metadata JSON.
3. **Diff-able across runs (Phase 2 additions):**
   - The Phase 2 BUILD first paired smoke + sister second + Nth paired smokes produce JSON ablation tables that are diff-able via `jq` or `tac.cathedral_consumers.canonical_equation_lookup_consumer` (per Catalog #335 + #344 + sister `procedural_codebook_generator_consumer`).
4. **Queryable post-hoc (Phase 2 additions):**
   - `experiments/results/<lane>/smoke_metadata.json` carries the full SubstrateContract-derived metadata + canonical posterior anchor reference + RATIFY-3 helper verdicts + stage_log + canonical equation #26 anchor.
5. **Cite-able (Phase 2 additions):**
   - Every Phase 2 BUILD prediction carries `[prediction; canonical-equation-26-grounded; phase_2_build]` axis tag per Catalog #287 + #323.
   - Every Phase 2 BUILD empirical anchor carries `[empirical:<artifact_path>]` axis tag.
   - The canonical posterior anchor per Revision #3 is the citation chain for downstream consumers.
6. **Counterfactual-able (Phase 2 additions):**
   - The REVISION #1 7-arm ablation ladder IS the canonical counterfactual: "what if luma quantization = 8 instead of 16?" etc.
   - The REVISION #2 multi-scale Dykstra-feasibility check produces the "what if seg + pose are non-additive?" counterfactual verdict.

---

## Section 8. Dykstra-feasibility (per Catalog #296)

Per the BUILD design memo's Dykstra-feasibility check: canonical equation #26 closed form `ΔS = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706` IS the rate-axis Pareto-frontier alternating-projections fixed point for procedural codebook substitution in IN-DOMAIN contexts. The Phase 2 BUILD first paired smoke is the FIRST EMPIRICAL TEST of whether the seg + pose contributions stack additively per the symposium's REVISION #2 verdict.

**Constraint intersection (rate <= R) AND (seg <= S) AND (pose <= P) with v8 contribution (rate -= 4064 bytes; seg += {0.0, +0.005} prediction; pose += {0.0, +0.001} prediction):**

| Component | feasibility verdict |
|---|---|
| rate-axis | NON-EMPTY (canonical equation #26 IS the fixed point) |
| seg-axis | UNKNOWN (per-substrate symposium REVISION #2 first paired smoke empirically tests) |
| pose-axis | UNKNOWN (sister UNWIND-TEST per REVISION #4 of THIS memo) |

If the first paired smoke empirically confirms `is_additive=True` for all 3 axes, the Phase 2 BUILD lands `composite predicted_band` ratified empirically. If `is_additive=False` for seg OR pose, route to UNWIND-TEST per REVISION #1 of the per-substrate symposium memo.

---

## Section 9. Catalog #324 post-training Tier-C validation

Per Catalog #324: `predicted_band_validation_status: pending_post_training` STAYS through Phase 2 BUILD landing. The first paired smoke harvest produces the post-training Tier-C density via:

```bash
.venv/bin/python tools/mdl_scorer_conditional_ablation.py \
    --archive experiments/results/<lane>/archive.zip \
    --tier c \
    --output .omx/state/nscs06_v8_phase_2_tier_c_<utc>.json
```

The Tier-C verdict (within-class density >= 0.70 / trending 0.50-0.70 / across-class <= 0.30 / indeterminate) classifies v8 vs the operator's existing canonical equations registry. Per RATIFY-3 REVISION #4: the JSON ablation table emitter ALREADY embeds `canonical_provenance` per Catalog #287 + #323; the Tier-C verdict appends to the same canonical anchor JSON via `update_equation_with_empirical_anchor` per Catalog #344.

---

## Section 10. Dispatch budget estimate

Per RATIFY-3 symposium cost contract:

| Item | Cost | Provenance |
|---|---|---|
| baseline smoke (1 compress pass + auth eval) | $0.50 | recipe `cost_band.hand_calibrated_fallback_p50_usd=0.50` |
| 3-axis ablation ladder (REVISION #1 incremental) | $1.50 | symposium memo verbatim "3 ablation arms x $0.50 each = $1.50" |
| **Phase 2 BUILD first paired smoke total** | **$2.00** | sister RATIFY-3 cost contract |
| sister CASCADE COMPRESSION 5-substrate aggregate paired-smoke matrix (PRIORITY 5; v8 + grayscale_lut + DP1 + VQ-VAE + ATW V2) | $2-3 aggregate | CASCADE COMPRESSION symposium memo |

Phase 2 BUILD first paired smoke fires post-PROCEED-unconditional verdict per REVISION #1; no paid GPU spend pre-PROCEED-unconditional.

---

## Section 11. Sister-coherence verification

**Slot 2 (DP1 HARVEST, OVERNIGHT-B)**: touches `.omx/state/modal_call_id_ledger.jsonl` + harvested artifacts; DISJOINT from THIS T2 DESIGN memo scope. Sister-checkpoint guard PROCEED at landing time.

**Slot 3 (HF dataset prep, OVERNIGHT-C)**: touches HF dataset infrastructure; DISJOINT from THIS T2 DESIGN memo scope. Sister-checkpoint guard PROCEED at landing time.

**THIS slot (OVERNIGHT-A)**: touches `.omx/research/council_t2_or_t3_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md` (NEW) + `.omx/state/council_deliberation_posterior.jsonl` (append-only via canonical helper per Catalog #131/#138/#300).

Pre-commit `git status` baseline at landing: `.omx/state/modal_call_id_ledger.jsonl` + `tools/build_hfv1_sparse_sidecar_candidate.py` already modified by sister slots. THIS landing adds only the NEW DESIGN memo + the posterior anchor append; no overlap.

---

## Section 12. Op-routables (post-T2-DESIGN-symposium-landing)

| Trigger | Action | Cost |
|---|---|---|
| Follow-on T1 working group reviews THIS T2 DESIGN memo + RATIFY-3 commit `20b6b59b3` + BUILD commit `853d108e2` + per-substrate symposium memo Section 5 reactivation criterion | Issue PROCEED-unconditional verdict OR PROCEED_WITH_REVISIONS OR DEFER_PENDING_EVIDENCE | $0 |
| Follow-on T1 PROCEED-unconditional issued | Spawn Phase 2 BUILD subagent (NEW lane `lane_phase_2_build_nscs06_v8_chroma_lut_20260521_or_later`); land `_full_main` body + recipe flip atomically per Catalog #240 | $0 implementation |
| Phase 2 BUILD landing complete | Run first paired smoke via operator-authorize chain (or operator dispatches manually); harvest 7-arm ablation matrix per REVISION #1 | $2.00 per RATIFY-3 cost contract |
| First paired smoke contest-CUDA + contest-CPU anchors land within predicted_band `-0.0027 +/- 0.006` | Mark `contest_cuda` + `contest_cpu` gates; promote lane to L2 per Catalog #233 4-gate canonical | $0 |
| First paired smoke score DRIFTS from predicted_band by >2x | Per Catalog #324: re-run post-training Tier-C density; per REVISION #1 ablation ladder + REVISION #2 verdict: identify drift-axis via per-assumption isolation; route to UNWIND-TEST per cargo-cult-audit assumptions 1-3 + 9 | $1-3 |
| CASCADE COMPRESSION symposium PRIORITY 5 5-substrate aggregate paired-smoke matrix | Queue v8 Phase 2 BUILD + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate paired-smoke matrix; aggregate predicted ΔS -0.013 to -0.0085 | $2-3 aggregate |

---

## Section 13. CLAUDE.md compliance verification

- Apples-to-apples evidence discipline: every prediction in this memo carries an axis tag (`[prediction; canonical-equation-26-grounded; phase_2_build_design]` / `[prediction; per-substrate-symposium-pending]`)
- Forbidden premature KILL: Phase 2 BUILD is a substrate scaffold lift, not a kill; per-substrate symposium PROCEED-unconditional reactivation criterion gates promotion per REVISION #1
- HNeRV parity discipline L4 (<=100 LOC inflate): Phase 2 BUILD does NOT change inflate.py; substrate_engineering exception per L7 preserved
- Strict scorer rule: Phase 2 `_full_main` loads scorers at COMPRESS time only (sister v7 pattern); inflate.py UNCHANGED with ZERO scorer imports
- UNIQUE-AND-COMPLETE-PER-METHOD: Phase 2 `_full_main` body is UNIQUE per substrate (sister v7 reference + RATIFY-3 helper invocations); canonical helpers ADOPT per Catalog #290
- Cargo-cult audit per Catalog #303: Section 5 above enumerates 3 NEW Phase 2-specific assumptions (1 CARGO-CULTED + 2 HARD-EARNED) + preserves the per-substrate symposium memo's 7 prior assumptions
- Catalog #220 operational mechanism: PRESERVED via substrate_contract.py UNCHANGED (`score_improvement_mechanism_status=SCAFFOLD_DEFERRED_INTEGRATION` flips to `OPERATIONAL` at Phase 2 BUILD landing per the recipe flip atomically)
- Catalog #240 recipe-vs-trainer-state: Section 2.2 above declares the atomic flip per CLAUDE.md "Strict-flip atomicity rule"
- Catalog #244 NVML block: PRESERVED in remote driver (UNCHANGED)
- Catalog #287 + #323 canonical Provenance: Phase 2 BUILD smoke metadata JSON carries `score_claim=False` + `promotable=False` + `evidence_grade=predicted` (pre-Tier-C) -> `evidence_grade=empirical` (post-Tier-C) + axis_tag literal
- Catalog #290 substrate canonical-vs-unique decision per layer: Section 2.3 above preserves the BUILD design memo's existing layer-wise decisions; Phase 2 `_full_main` body is a NEW UNIQUE layer (the trainer wire-in surface)
- Catalog #292 per-deliberation assumption surfacing: THIS memo's frontmatter `council_assumption_adversary_verdict` enumerates 6 assumption classifications
- Catalog #294 9-dimension success checklist evidence: Section 4 above declares Phase 2 BUILD additions per dimension
- Catalog #296 Dykstra-feasibility predicted-band: Section 8 above carries the constraint-intersection verdict per Phase 2 BUILD scope
- Catalog #300 v2 frontmatter: THIS memo carries `council_tier` + `council_attendees` + `council_quorum_met` + `council_verdict` + `council_dissent` + `council_decisions_recorded` + `council_assumption_adversary_verdict` + `council_predicted_mission_contribution` + `council_override_invoked` + `deferred_substrate_id` + `deferred_substrate_retrospective_due_utc`
- Catalog #303 cargo-cult audit: Section 5 above
- Catalog #305 observability surface: Section 7 above
- Catalog #309 horizon_class declaration: PRESERVED in BUILD design memo frontmatter `plateau_adjacent`; Phase 2 BUILD does NOT change horizon class
- Catalog #324 predicted-band post-training validation: Section 9 above
- Catalog #325 per-substrate symposium: THIS memo SATISFIES Catalog #325 6-step contract at the Phase 2 DESIGN surface (cargo-cult audit + 9-dim + observability + 16-attendee deliberation + reactivation criteria + Tier-C validation declaration)
- Catalog #340 sister-checkpoint guard: PROCEED at landing time per Section 11
- Catalog #344 canonical equation cross-reference: `procedural_codebook_from_seed_compression_savings_v1` + IN-DOMAIN context `nscs06_v8_chroma_lut`
- Catalog #346 canonical_council_roster: T2 16-attendee roster validates `complete=True` per Section "Verdict" + Section 1

---

## Section 14. 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map: ACTIVE (Phase 2 BUILD emits per-axis decomposition `predicted_d_seg_delta` + `predicted_d_pose_delta` + `predicted_archive_bytes_delta` in smoke metadata JSON per Catalog #356; canonical Provenance threaded per Catalog #323)
- hook #2 Pareto constraint: ACTIVE via canonical equation #26 IN-DOMAIN predicted rate-axis contribution to Pareto polytope (Section 8); first paired smoke empirically tests intersection-non-empty for seg + pose axes
- hook #3 bit-allocator: ACTIVE via REVISION #1 luma-quantization-levels ablation (8 / 16 / 32 levels are different bit-allocator regimes per sister RATIFY-3 wire-in declaration)
- hook #4 cathedral autopilot dispatch: ACTIVE via REVISION #4 JSON ablation table consumed by sister `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 + #344 + sister `tac.cathedral_consumers.procedural_codebook_generator_consumer`
- hook #5 continual-learning posterior: ACTIVE via REVISION #3 (Phase 2 BUILD landing emits canonical posterior anchor via `tac.council_continual_learning.append_council_anchor`) + first paired smoke feeds canonical equation #26 posterior update via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344
- hook #6 probe-disambiguator: ACTIVE — the REVISION #1 7-arm ablation ladder IS the canonical probe disambiguator per Catalog #308 alternative-probe-methodology enumeration; the planned path `tools/probe_nscs06_v8_chroma_lut_canonical_equation_26_in_domain_disambiguator.py` is the canonical disambiguator artifact

---

## Section 15. Phase 2 reactivation criteria (frontmatter `council_decisions_recorded`)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #300 mission-alignment Consequence 3 (30-day score-impact retrospective): Phase 2 BUILD landing produces an empirical anchor that must be retrospectively reviewed 30 days later (2026-06-20) for score-impact verdict. If empirical ΔS lands within predicted_band (`-0.0027 +/- 0.006`): RATIFY canonical equation #26 IN-DOMAIN context membership. If empirical ΔS drifts >2x: route to UNWIND-TEST per REVISION #1 of the per-substrate symposium memo; do NOT KILL the substrate (Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodology enumeration apply).

---

## Section 16. Cross-references

- **Per-substrate symposium memo (T1)**: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (PROCEED_WITH_REVISIONS; 4 binding revisions)
- **RATIFY-3 landing memo**: `.omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` (PROCEED; commit `20b6b59b3`; 105/105 tests pass)
- **BUILD design memo**: `.omx/research/nscs06_v8_chroma_lut_design_20260521.md` (L0 SCAFFOLD design)
- **BUILD commit**: `853d108e2` (NSCS06 v8 chroma_lut substrate L0 SCAFFOLD)
- **Sister v7 trainer (Phase 2 reference pattern)**: `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main` (~470 LOC; sister `_decode_real_pairs` + scorer chunked pattern + canonical Hydra dict pose slice)
- **Sister v7 cargo-cult-unwind methodology**: commit `4292c8ce2` (44% improvement empirically validated rescue path)
- **CASCADE COMPRESSION symposium**: commit `d125af6c3` PRIORITY 3 + Revision #5 (v8 chroma_lut elevated as second-priority IN-DOMAIN substrate)
- **HONEST CASCADE-MORTALITY ASSESSMENT**: commit `d884dd6aa` Rank 2 (HIGH P(actual score reduction))
- **Canonical equation #26**: `src/tac/canonical_equations/procedural_codebook_savings.py` (IN-DOMAIN context `nscs06_v8_chroma_lut`)
- **Canonical roster validator**: `tac.canonical_council_roster.validate_council_dispatch_roster` (T2 complete=True for 16 attendees)
- **Canonical posterior helper**: `tac.council_continual_learning.append_council_anchor` (this memo emits anchor via this helper post-commit)

---

## Section 17. Canonical posterior anchor wire-in (Catalog #300 + #346)

This memo's verdict + dissent + assumption-adversary classifications + decisions are appended to the canonical council deliberation posterior at `.omx/state/council_deliberation_posterior.jsonl` via:

```python
from tac.council_continual_learning import (
    CouncilDeliberationRecord, CouncilTier, append_council_anchor,
)

record = CouncilDeliberationRecord(
    deliberation_id="council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521",
    topic="NSCS06 v8 Phase 2 DESIGN-only symposium lifting `_full_main raises NotImplementedError`",
    council_tier=CouncilTier.T2,
    council_attendees=(
        "Shannon", "Dykstra", "Rudin", "Daubechies",
        "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary",
        "Quantizr", "Selfcomp", "MacKay", "Balle", "PR95Author",
        "Carmack", "Hotz", "Mallat",
    ),
    council_quorum_met=True,
    council_verdict="PROCEED_WITH_REVISIONS",
    deferred_substrate_id="nscs06_v8_chroma_lut",
    deferred_substrate_retrospective_due_utc="2026-06-20T07:05:42Z",
    predicted_mission_contribution="frontier_breaking_enabler",
    override_invoked=False,
    override_rationale="",
    related_deliberation_ids=(
        "council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521",
        "council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3",
        "council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2",
    ),
    # council_dissent + council_assumption_adversary_verdict + council_decisions_recorded
    # per frontmatter above
)
append_council_anchor(record)
```

Per Catalog #346 the canonical roster validator `tac.canonical_council_roster.validate_council_dispatch_roster(council_attendees, topic_tokens=['substrate', 'chroma_lut', 'nscs06_v8', 'phase_2', 'lift_notimplementederror'], council_tier='T2')` returns `complete=True` because all 16 inner council members are present (4 co-leads + 12 sister inner council seats; sextet pact + sister members satisfied).
