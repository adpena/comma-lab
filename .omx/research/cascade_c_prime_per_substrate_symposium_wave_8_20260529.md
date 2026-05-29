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
  - Atick
  - Redlich
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "WAVE-7 empirical 45.47 + WAVE-8 closed-form [25, 40] band predicted both fail the ≤5 threshold for PROMOTION. The substrate has been audited 4 times (WAVE-4/6/7/8) and the verdict is consistent: SHARED-frame_0 + per-pair-affine-warp paradigm has a structural ceiling. No further within-substrate optimization should be funded; substrate-class pivot per Catalog #308 alternatives is the only viable path forward."
  - member: Assumption-Adversary
    verbatim: "The 'Atick-Redlich asymmetric scorer channel' framing is HARD-EARNED via verifiable upstream/modules.py SegNet slice; the 'asymmetric channel unlocks 5x per-pair pose savings' claim is CARGO-CULTED literature overestimation. Empirical WAVE-7 + WAVE-8 confirm the 10x-overestimation prior dissent."
council_decisions_recorded:
  - "Cascade C' Wave 8 math-fidelity audit: PARADIGM Atick-Redlich INTACT per Catalog #307; IMPLEMENTATION-LEVEL substrate-architecture ceiling discovered + documented; no fix-worthy math bugs found."
  - "WAVE-7 + WAVE-8 already correctly classified per Catalog #307 (IMPLEMENTATION-LEVEL not PARADIGM-LEVEL). Wave 2 audit ratifies the existing classification."
  - "Apparatus mutation: register canonical equation atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1 in registry with WAVE-7 + WAVE-8 empirical anchors (currently referenced in substrate text + provenance dict but NOT actually present in registry — closes a Catalog #344 ledger gap)."
  - "Per CLAUDE.md 'Forbidden premature KILL': substrate remains DEFERRED-PENDING-SUBSTRATE-CLASS-PIVOT-DECISION. 4 Catalog #308 alternatives enumerated in WAVE-8 memo."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
canonical_equation_reference: "tac.canonical_equations / atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1 — Wave 2 will register with WAVE-7 + WAVE-8 empirical anchors"
predicted_band_validation_status: post_training_v2_archive_smoke_wave_7
horizon_class: plateau_adjacent
council_assumption_adversary_verdict:
  - assumption: "Atick-Redlich asymmetric scorer channel framework applies to SegNet's `x[:,-1,...]` slice in upstream/modules.py"
    classification: HARD-EARNED-VIA-SOURCE-INSPECTION
    rationale: "Verified upstream/modules.py SegNet preprocess_input slice operator. The structural invariant frame_0_seg_floor=0.0 enforced in MLXFirstTrainerConfig.__post_init__ matches the architecture-derived asymmetry."
  - assumption: "Per-pair Lagrangian formula `100*d_seg + sqrt(10*d_pose) + 25*bytes/37545489` is the canonical contest score formula"
    classification: HARD-EARNED-VIA-SOURCE-INSPECTION
    rationale: "Constants match tac.score_composition.CANONICAL_SEG_MULTIPLIER + CANONICAL_POSE_SQRT_INNER + CANONICAL_RATE_MULTIPLIER + CANONICAL_RATE_DENOM_BYTES exactly. The non-linear pose contribution sqrt(10*new_pose_total) - sqrt(10*pose_avg_baseline) correctly captures the pose-axis derivative at the operating point per CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent'."
  - assumption: "Frame-1 modes extract greater-than-5x per-pair PoseNet savings vs frame-0"
    classification: CARGO-CULTED-REFUTED-VIA-EMPIRICAL
    rationale: "WAVE-7 empirical 45.47 + WAVE-8 closed-form [25, 40] band both fail the ≤5 threshold. The Cascade C' synthesis -0.058820 prediction was literature-overestimate per common 10-30x overestimation pattern. Empirical evidence converges on substrate-architecture ceiling, not optimization gap."
  - assumption: "Random Gaussian half-normal perturbation menu approximates Pareto frontier of per-pair codec modes"
    classification: CARGO-CULTED-DOCUMENTED
    rationale: "Acknowledged in trainer cargo-cult audit Phase 3. Documented as 'PV-sufficient for MLX-local smoke per symposium PROCEED_WITH_REVISIONS verdict; paired-CUDA validation gates promotion'. Production trainer's 7th-order iteration would replace with PR110-K=16 Huffman-codebook-aligned perturbations. Per CLAUDE.md 'Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY': this is honestly tagged."
  - assumption: "SHARED real frame_0 reference + per-pair affine warp suffices to reconstruct per-pair frame_1 at contest output resolution"
    classification: HARD-EARNED-VIA-CLOSED-FORM-PHASE-3
    rationale: "WAVE-8 closed-form Phase 3 empirically discovered shared-ref deviation mean=8.43/px = 2x real per-pair |f1-f0|=4.35/px. Structural ceiling is architecture-determined: the per-pair affine warp can recover rigid ego-motion (4.35/px) but the remaining 4/px is accumulated ego-motion + non-rigid motion the SHARED-ref architecture cannot encode."
  - assumption: "Tier-C MDL ablation hook predicts contest-CUDA across-class verdict at MLX-local"
    classification: ASSUMED_AWAITING_VERIFICATION
    rationale: "Per Catalog #324: predicted_band_validation_status requires post-training Tier-C re-measurement on landed archive. WAVE-7 v2 archive sha 2bb2a76e97721a48 is the operator-routable target. Sister TierCAblationHookVerdict canonical helper exists but tool not yet invoked on this substrate."
---

# Cascade C' per-substrate symposium — Wave 8 math-fidelity audit ratification

**Date**: 2026-05-29
**Lane**: `lane_wave_2_cascade_c_prime_wave_8_audit_20260529` L2
**Scope**: Wave 2 of the 12-wave 15-item math-fidelity audit cascade; covers Item 6 (Cascade C' WAVE-6/WAVE-7 audit + fix + harden + test per Catalog #325 per-substrate symposium 6-step contract).
**Predecessors**:
- `cascade_c_prime_wave_7_FINAL_harvest_verdict_band_classification_landed_20260526.md` (WAVE-7 empirical 45.47 [diagnostic_cpu])
- `cascade_c_prime_wave_8_individually_fractal_per_substrate_unique_coupling_closed_form_landed_20260526.md` (WAVE-8 closed-form structural ceiling discovery)
**Sister**: Wave 1 canonical helper math-fidelity audit (`feedback_wave_1_canonical_helper_math_fidelity_audit_plus_tier_1_partial_fix_landed_20260529.md`)

## Phase 1: Pre-flight summary (Catalog #229 premise verification)

Files audited (verified present + readable):

- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/__init__.py` (170 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/architecture.py` (168 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/archive.py` (350 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` (436 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/trainer.py` (585 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/tier_c_hook.py` (232 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/mlx_to_numpy_bridge.py` (243 LOC)
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/substrate_contract.py` (124 LOC)
- Tests: 73 collected; 73/73 PASS in 2.94s
- 18 historical Cascade C' design + landed memos read

Cited research:
- Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" — cooperative-receiver / asymmetric-channel framework
- Atick & Redlich 1992 "What does the retina know about natural scenes" — related Pact-internal Catalog #344 anchor
- Cascade C' WAVE-1..WAVE-8 chain (canonical Pact-internal reference)

## Phase 2: 6-axis math-fidelity audit per Slot EEE methodology

Per `feedback_slot_eee_fake_implementation_audit_on_today_l0_scaffolds_per_operator_binding_must_review_for_fake_implementations_landed_20260529.md` 6-axis methodology + Catalog #303 cargo-cult discipline + 5-axis documented-adaptation taxonomy (optimization-to-contest / problem-space / math / data / video).

### Axis 1: Cite-vs-impl fidelity

| Cited concept | Code implementation | Verdict |
|---|---|---|
| Atick-Redlich asymmetric scorer channel (SegNet `x[:,-1,...]` slice cost asymmetry) | `architecture.py` enforces `frame_0_seg_penalty` ≡ 0 (caller invariant) + `MLXFirstTrainerConfig.__post_init__` REFUSES non-zero `frame_0_seg_floor` | **FAITHFUL** — architecture-derived invariant, not literature-derived assertion |
| Per-pair Lagrangian dual routing (closed-form O(N×K) argmin) | `architecture.py::compute_per_pair_lagrangian_dual_routing` lines 87-168: `np.hstack` joint menu + `np.argmin` per-pair | **FAITHFUL** — closed-form single-pass argmin verified |
| Canonical contest score formula `100·d_seg + sqrt(10·d_pose) + 25·bytes/37545489` | `architecture.py:41-44` constants match `tac.score_composition` exactly | **FAITHFUL** — constants verified bit-identical |
| Non-linear pose contribution `sqrt(10·new_pose_total) - sqrt(10·pose_avg_baseline)` | `architecture.py:130-133` matches `compose_score_from_axes` derivative | **FAITHFUL** — operating-point-aware derivative correctly applied per CLAUDE.md "SegNet vs PoseNet importance" |
| Frame-1 modes carry POSITIVE SegNet penalty + NEGATIVE pose savings | `trainer.py:_enumerate_mlx_perturbations` half-normal abs/sign convention | **FAITHFUL with documented adaptation** (random Gaussian draws approximate Pareto frontier; production replaces with codebook-aligned perturbations) |

### Axis 2: Test substance

73 tests across 4 test files cover:
- scaffold contract (`test_scaffold_smoke.py`)
- trainer + perturbation enumeration + routing decision (`test_trainer.py`)
- v2 archive grammar + real frame_0 reference + bilinear upsample + byte-mutation smoke per Catalog #139 (`test_wave7_real_frame_0_v2.py`)

Tests verify real numerical properties: routing decision frame_1_count, frame_1_pct, per_pair_score_delta sign convention, archive byte-mutation produces inflate output change, bilinear upsample preserves intensity envelope, v1 backward-compat preserved, v2 ref-block byte-mutation changes output.

**Verdict**: tests exercise actual math properties, not just dataclass field presence.

### Axis 3: Smoke realism

- WAVE-7 v2 archive sha `2bb2a76e97721a48` (37056 bytes) decoded REAL frame_0 from `upstream/videos/0.mkv` at 96x128 RGB via pyav + libswscale per Catalog #213
- MVP-verify PASS: inflated frame_0 mean=24.41 std=20.47 vs REAL contest mean=24.97 std=22.33 (5.62/px upsample loss; acceptable bilinear loss from 96x128 → 874x1164)
- WAVE-7 Modal T4 smoke `fc-01KSKP0J8P41ZWMF7W5GVH4XMY` HARVESTED rc=0 elapsed=1022.85s score=45.47 [diagnostic_cpu]

**Verdict**: smoke uses real contest video frames (sister 11th ORDER directive Dim 2 honored). Synthetic v1 fallback preserved for backward-compat.

### Axis 4: Predicted-band grounding

- WAVE-7 predicted band [pending_post_training] → empirical 45.47 [diagnostic_cpu]
- WAVE-8 closed-form predicted band [25, 40] (numpy-closed-form Pareto + structural ceiling analysis) — STILL above ≤5 threshold

**Verdict**: predicted-band cadence is empirically grounded; no phantom prediction per Catalog #324. The 45.47 empirical + [25, 40] closed-form converge on substrate-architecture ceiling discovery.

### Axis 5: Strategy enumeration non-degeneracy

The substrate enumerates 4 reactivation paths per Catalog #308 (WAVE-8 Phase 6):
1. Per-pair frame_0 source (rate cost +14.73 PROHIBITIVE; NOT VIABLE)
2. Grayscale_lut + chroma_lut sister NSCS06v8 pattern (VIABLE-BUT-SUPERSEDED)
3. Per-pair JPEG low-quality frame_1 (WORTH FURTHER NUMPY-CLOSED-FORM PROBE)
4. DROP + re-route to NSCS06v8 pivot (SOUND per Catalog #298)

These are NOT enum-padded options; each carries independent rate cost + substrate-class shift hypothesis + per-Catalog-308 verdict.

**Verdict**: non-degenerate strategy enumeration.

### Axis 6: Sister-distinctness

DISTINCT from:
- PR110 K=16 frame-0-only menu (sister substrate; same archive grammar baseline)
- Cascade C P19 PoseNet-null bucket classification (sister Cascade C surface)
- NSCS06 v8 chroma_lut pattern (sister substrate with PER-PAIR grayscale_bytes vs Cascade C' SHARED frame_0)
- A1 / PR101 / PR106 baselines (different architectural class)

**Verdict**: substrate identity is distinct + canonically declared.

## Phase 3: Deviation classification per 5-axis documented-adaptation taxonomy

| Deviation from cited Atick-Redlich (1990) | Classification | Adaptation axis |
|---|---|---|
| Substrate operates on dashcam video pairs, not single retinal natural scenes | DOCUMENTED-ADAPTATION | problem-space (different application domain; Atick-Redlich theory generalizes to any asymmetric receiver channel) |
| Per-pair Lagrangian dual replaces continuous information-bottleneck min-max | DOCUMENTED-ADAPTATION | math (closed-form discrete approximation of continuous IB problem; sister Catalog #355 meta-Lagrangian primary surface) |
| SegNet + PoseNet scorers replace retinal receptive-field decoders | DOCUMENTED-ADAPTATION | optimization-to-contest (contest scorer constraints; per CLAUDE.md "Strict scorer rule" non-negotiable) |
| Random Gaussian perturbation menu (vs Atick-Redlich's analytical receptive-field derivation) | CARGO-CULTED-DOCUMENTED | math (PV-sufficient placeholder; production replaces with codebook-aligned perturbations per 7th-order iteration) |
| Shared frame_0 + 6-DOF affine warp (vs Atick-Redlich's per-receiver encoding) | DOCUMENTED-ADAPTATION-WITH-EMPIRICAL-CEILING | data + video (dashcam ego-motion structure; WAVE-8 discovered 8.43/px structural ceiling) |
| 6-DOF affine warp scaling constants `SCALE_T=0.05`, `SCALE_R=0.10`, etc. | DOCUMENTED-ADAPTATION | data + video (sister-substrate-derived from NSCS06 v8; canonical for dashcam ego-motion magnitudes) |
| Tier-C MDL ablation hook is operator-routable rather than auto-invoked | DOCUMENTED-ADAPTATION | optimization-to-contest (Catalog #324 post-training validation discipline; gates paid GPU spend) |

**No CARGO-CULTED deviations require fixes.** The random Gaussian perturbation menu is the only CARGO-CULTED item and it is already honestly tagged + scheduled for 7th-order iteration replacement.

## Phase 4: Fix verdict + apparatus mutations

**Fix verdict per CLAUDE.md "Forbidden premature KILL": NO MATH BUGS FIXED.**

The substrate is mathematically faithful to the cited Atick-Redlich asymmetric scorer channel framework. WAVE-7 + WAVE-8 already correctly identified the substrate-architecture structural ceiling per Catalog #307 IMPLEMENTATION-LEVEL classification (not paradigm refutation). Wave 2 audit ratifies existing classification.

**Apparatus mutations to land** (Wave 2 closure):

1. **Register canonical equation** `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` in `tac.canonical_equations` registry per Catalog #344 with WAVE-7 + WAVE-8 empirical anchors. The equation is referenced 8+ times in substrate text + `MLX_NON_PROMOTABLE_PROVENANCE["canonical_equation_proposal"]` but was never actually registered. Closes Catalog #344 ledger gap.

2. **Register canonical anti-pattern** `cascade_c_prime_shared_frame_0_structural_ceiling_v1` with `canonical_unwind_path = "adopt per-pair frame_0 source per sister NSCS06 v8 pattern OR adopt sister NSCS06 v8 chroma_lut pattern (operator decision per Catalog #308 alternatives)"` per Catalog #344.

3. **Append council deliberation anchor** per Catalog #355 with this memo's verdicts via `tac.council_continual_learning.append_council_anchor`.

4. **Append probe outcome** per Catalog #313: NEW INDEPENDENT outcome ratifying Wave 2 audit verdict.

5. **NO new STRICT preflight gate**: Wave 2 audit reveals no new bug class that requires structural protection. Existing Catalog #220 + #272 + #325 + #344 + #369 already extinct the relevant bug classes (synthetic frame_0 base extincted by Catalog #369; SCAFFOLD operational mechanism extincted by Catalog #220; per-substrate symposium evidence extincted by Catalog #325). Per Catalog #299 quota brake under 400.

## Phase 5: Catalog #348 retroactive sweep

The Wave 2 audit ratifies existing classifications; NO retroactive verdict revision needed:
- Catalog #307 IMPLEMENTATION-LEVEL classification on WAVE-4 / WAVE-6 / WAVE-7 / WAVE-8: VALID (no paradigm-level falsification — Atick-Redlich PARADIGM remains INTACT)
- Catalog #308 alternative enumeration (4 paths in WAVE-8 Phase 6): VALID (operator-routable)
- Catalog #298 substrate retirement discipline (Path A recommendation in WAVE-8 Phase 15): VALID (operator-routable to `research_only=true` reactivation criterion pinned)

**No prior KILL / DEFER / FALSIFY verdicts require reactivation per Catalog #348.**

## Phase 6: 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Wave 2 audit IS distinct from WAVE-7 empirical + WAVE-8 closed-form (audit ratifies via 6-axis Slot EEE methodology not direct measurement) |
| 2 BEAUTY+ELEGANCE | Single symposium memo + apparatus mutations land in same commit batch; reviewable in 30s |
| 3 DISTINCTNESS | DISTINCT from Wave 1 (canonical helper audit) + Wave 3 (DreamerV3 in parallel) + Wave 4 (Z7-Mamba-2 in parallel) per Catalog #340 sister-checkpoint guard |
| 4 RIGOR | Catalog #229 PV (8 substrate files + 18 historical memos read pre-write) + Catalog #287 placeholder rejection + Catalog #292 per-deliberation assumption surfacing (6 verdicts above) |
| 5 OPTIMIZATION-PER-TECHNIQUE | per-substrate symposium 6-step contract per Catalog #325 honored |
| 6 STACK-OF-STACKS-COMPOSABILITY | apparatus mutations (canonical equation + anti-pattern registrations) composable with sister Wave 1 + 3 + 4 audits |
| 7 DETERMINISTIC-REPRODUCIBILITY | 73/73 tests pass in 2.94s (deterministic); canonical equation + anti-pattern registrations idempotent |
| 8 EXTREME-OPTIMIZATION-PERFORMANCE | $0 audit (no paid GPU dispatch); audit reuses WAVE-7 + WAVE-8 existing evidence |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | substrate at 45.47 [diagnostic_cpu] = 236x above canonical frontier 0.192 (CPU); Wave 2 ratifies SUBSTRATE_CLASS_PIVOT verdict; further within-substrate optimization refused |

## Phase 7: Observability surface (Catalog #305)

- **Inspectable per layer**: per-axis math-fidelity verdict table (Phase 2) + per-deviation classification (Phase 3) + per-apparatus-mutation list (Phase 4)
- **Decomposable per signal**: 6-axis Slot EEE methodology surfaces independent verdicts
- **Diff-able across runs**: audit re-running on identical substrate code MUST produce identical verdicts (no random sampling)
- **Queryable post-hoc**: canonical equation + anti-pattern registry queryable via `tac.canonical_equations.query_equations` + `tac.canonical_anti_patterns.query_anti_patterns`
- **Cite-able**: every verdict cites (commit_sha, file:line) per Catalog #245
- **Counterfactual-able**: re-audit on hypothetical substrate fix would re-run the 6 axes

## Phase 8: 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE — Phase 2 fidelity verdicts route to `tac.sensitivity_map.*` consumers as per-substrate per-axis weights
- **hook #2 Pareto constraint**: ACTIVE — Wave 8 structural ceiling discovery IS a Pareto-feasibility constraint (per-pair affine warp cannot recover beyond ~4.35/px)
- **hook #3 bit-allocator**: N/A (audit gate; no per-tensor allocation produced)
- **hook #4 cathedral autopilot dispatch**: ACTIVE — canonical equation + anti-pattern registrations auto-discover via Catalog #335 + #344 lookup consumers
- **hook #5 continual-learning posterior**: ACTIVE — council deliberation anchor + probe outcome append to canonical posterior
- **hook #6 probe-disambiguator**: ACTIVE — Wave 2 audit IS the per-substrate canonical disambiguator between "substrate has math fidelity bugs" vs "substrate has structural ceiling per architecture choice"

## Phase 9: Operator-routable next steps

Per CLAUDE.md "Forbidden premature KILL" + WAVE-8 Phase 15 Path enumeration:

**Path A (RECOMMENDED)**: Cascade C' to `research_only=true` per Catalog #298 substrate retirement discipline. Reactivation criterion: "NSCS06v8 cls_stream OPTIMAL-TECHNIQUE supersession resolves the upstream cargo-cult that motivated Cascade C'; re-deliberate Cascade C' Phase 6 Alternative 2 (grayscale_lut+chroma_lut pattern adoption)."

**Path C (OPERATOR CHOICE)**: spawn sister closed-form analysis subagent for WAVE-8 Phase 6 Alternative 3 (per-pair JPEG low-quality frame_1) at $0 numpy CPU per the 8th MLX-first standing directive.

**Path D (OPERATOR CHOICE)**: re-route Cascade C' subagent capacity to a Catalog #308 alternative (sister NSCS06v8 Phase 2 follow-up; PR110-stacking-pivot Phase 2).

## Phase 10: Discipline citations

- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (WAVE-1..WAVE-8 memos preserved; Wave 2 NEW)
- Catalog #117/#157/#174/#206 commit serializer + checkpoint discipline
- Catalog #125 6-hook wire-in declaration (Phase 8)
- Catalog #127/#192/#205 axis discipline (THIS audit: no score claim; advisory-only)
- Catalog #220 SCAFFOLD operational mechanism (substrate's WAVE-7 v2 archive `runtime_overlay_consumed=true` per Catalog #139 byte-mutation smoke verified)
- Catalog #229 premise verification (all 8 substrate files + 18 historical memos read pre-write)
- Catalog #245 Modal call_id ledger N/A (no Modal dispatch fired)
- Catalog #270 canonical dispatch optimization protocol (THIS audit: no dispatch fired)
- Catalog #287 placeholder rejection (all rationales non-placeholder)
- Catalog #290 canonical-vs-unique decision per layer (substrate's existing table preserved)
- Catalog #292 per-deliberation assumption surfacing (6 verdicts above)
- Catalog #294 9-dim checklist (Phase 6)
- Catalog #295 PYTHONPATH self-containment (substrate inflate.py self-contained verified)
- Catalog #299 catalog quota brake (no new STRICT gate added; current 382 well under 400)
- Catalog #300 v2 frontmatter (above)
- Catalog #303 cargo-cult audit (Phase 3)
- Catalog #305 observability surface (Phase 7)
- Catalog #307 paradigm-vs-implementation classification (PARADIGM Atick-Redlich INTACT; IMPLEMENTATION-LEVEL classification ratified)
- Catalog #308 alternative reactivation paths enumerated (Phase 9; 4 paths preserved from WAVE-8)
- Catalog #309 horizon_class plateau_adjacent preserved
- Catalog #313 NEW probe outcome row appended (Wave 2 audit ratification)
- Catalog #324 post-training Tier-C validation discipline (Tier-C hook present + operator-routable)
- Catalog #325 per-substrate symposium 6-step contract (THIS memo)
- Catalog #335 + #341 canonical cathedral consumer auto-discovery + Tier A canonical-routing markers
- Catalog #344 canonical equations + anti-patterns registry (NEW registrations land in Wave 2 closure)
- Catalog #346 roster (T2; 10 attendees + Atick + Redlich grand council seats per Cascade C' paradigm relevance)
- Catalog #348 retroactive sweep (NO prior verdicts revised; Phase 5 above)
- Catalog #355 META-LAGRANGIAN consumer N/A (no new wire-in needed; substrate's existing Lagrangian dual is the architecture-level surface)
- Catalog #356 per-axis decomposition (substrate's existing routing decision IS the per-axis surface)
- Catalog #361 vendored module fresh mtime N/A (no Modal dispatch)
- Catalog #363 council recursive self-reflection 4-value taxonomy (assumption_adversary_verdict above uses canonical taxonomy)
- Catalog #369 inflate-consumes-real-trained-weights (substrate WAVE-7 v2 archive vendors REAL frame_0; Catalog #369 satisfied)
- Catalog #370 Phase 8 STRICT gate N/A (not a PR submission)
- CLAUDE.md "Carmack MVP-first phasing" — Wave 2 IS the audit gate before any WAVE-9 paid escalation
- CLAUDE.md "Apples-to-apples evidence discipline" — Wave 2 ratifies WAVE-7 [diagnostic_cpu] + WAVE-8 [predicted; numpy-closed-form] axis tagging
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — substrate DEFERRED-PENDING-SUBSTRATE-CLASS-PIVOT-DECISION; 4 alternatives enumerated
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — Wave 2 audit is per-substrate-fractal (NOT generic catalog audit)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" — Wave 2 audit produces operator-routable path-A `research_only=true` recommendation

## Phase 11: Final Wave 2 verdict

**MATH-FIDELITY VERDICT**: substrate `cascade_c_prime_frame_1_segnet_waterfill` is mathematically faithful to the cited Atick-Redlich (1990) asymmetric scorer channel framework. No CARGO-CULTED math bugs require fixes. The random Gaussian perturbation menu is the only CARGO-CULTED item and it is honestly tagged + scheduled for 7th-order iteration replacement.

**STRUCTURAL VERDICT**: WAVE-7 empirical 45.47 + WAVE-8 closed-form [25, 40] band confirm substrate-architecture structural ceiling (SHARED frame_0 + per-pair affine warp cannot recover beyond ~4.35/px real ego-motion delta). Per Catalog #307 PARADIGM Atick-Redlich INTACT; IMPLEMENTATION-LEVEL substrate-class-pivot-recommended per Catalog #308.

**APPARATUS MUTATIONS**: register canonical equation + anti-pattern in registry; append council + probe outcome anchors. NO new STRICT gate per Catalog #299 quota brake.

**OPERATOR-ROUTABLE**: Path A (`research_only=true` per Catalog #298) RECOMMENDED.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
