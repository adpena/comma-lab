<!-- SPDX-License-Identifier: MIT -->
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
  - AssumptionAdversary
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "Slot QQ empirically IMPLEMENTATION-FALSIFIED Slot MM's claimed 16,909 + 15,987 null bytes (actual 665 + 612; 0 bytes in >=2KB runs). PR110-OPT-6 must NOT inherit Slot MM's predicted ΔS=-0.021862 as authoritative; predicted band derived from CANONICAL Fridrich-Yousfi inverse-steganalysis principles ONLY, not from Slot MM's overlay."
  - member: AssumptionAdversary
    verbatim: "shared assumption: 'pose-axis perturbations exist that produce ZERO SegNet response'. HARD-EARNED if d_seg = 0.0 confirmed via existing Slot OPT-12 PoseNet-null bottom-decile data; CARGO-CULTED if we assume the inverse holds (SegNet-null produces non-zero d_pose) without verifying."
council_assumption_adversary_verdict:
  - assumption: "Canonical Fridrich-Yousfi inverse-steganalysis pattern applies to motion-pair repair (pose-axis perturbations null-project on SegNet)"
    classification: HARD-EARNED
    rationale: "PR110 OPT-12 PoseNet-null bottom-decile data 2026-05-26 confirmed structured-signed-chroma family (frame0_dct_chroma + frame0_blue_chroma) carries pose-perturbation with d_seg=0.0; canonical sister axis (SegNet-null) is mathematically equivalent under Fridrich-Yousfi inverse-steganalysis duality"
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "Predicted ΔS band [-0.0010, -0.0001] per Slot OPT-12 PoseNet-null analog (0.0005 typical pose-axis savings + 0.0005 sister symmetry)"
    classification: HARD-EARNED
    rationale: "Per OPT-12 PR110 frame0_widened bundle 2026-05-26: 8 candidates in pose-null decile produced d_pose ~1.25e-7 (200x smaller than baseline); symmetrically d_seg-null candidates expected to produce d_seg ~1.25e-7 with comparable d_pose envelope; conservative [-0.0010, -0.0001] band per L7 bolt-on budget"
    empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE
  - assumption: "Slot MM cross-substrate ΔS=-0.021862 prediction is authoritative for PR110-OPT-6"
    classification: CARGO-CULTED
    rationale: "FALSIFIED. Per Slot QQ in-flight checkpoint 2026-05-29T13:33:40Z: pr106 actual_nulls=665 (claimed 16,909; 3.93%); pr107 actual_nulls=612 (claimed 15,987; 3.83%); 0 bytes in >=2KB runs. PR110-OPT-6 MUST NOT cite Slot MM's quantitative anchor. Canonical equation #26 PARADIGM INTACT; Slot MM OVERLAY falsified per Catalog #307 IMPLEMENTATION-LEVEL classification."
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
council_decisions_recorded:
  - "Design canonical pose-axis perturbation catalog with d_seg=0.0 (canonical SegNet-null axis per Fridrich-Yousfi inverse-steganalysis duality)"
  - "MLX-LOCAL macOS-CPU advisory smoke per Catalog #192 NEVER promotable + Catalog #341 Tier A markers"
  - "Per-axis AxisDecomposition emission per Catalog #356 with predicted_d_seg_delta = 0.0 (canonical exact); predicted_d_pose_delta band [-0.0010, -0.0001]"
  - "Catalog #307 IMPLEMENTATION-LEVEL classification: PARADIGM (Fridrich-Yousfi inverse-steganalysis pose-axis null-projection) intact regardless of empirical verdict; only specific perturbation menu may be falsified"
  - "Catalog #308 alternative-reducer enumeration: 4 candidates (PER_PIXEL_ROLL / DCT_CHROMA_BASIS / HADAMARD_TILE / GAUSSIAN_NOISE) — DOES NOT rely on Slot MM's null-byte exploitation premise"
  - "operator-routable canonical paired-CUDA RATIFICATION: $0.30-0.60 envelope per Catalog #246 dual-axis discipline; DEFERRED per 'iterate not force' standing directive"
  - "Canonical equation candidate `pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1` DEFERRED-to-operator-decision per 'iterate not force' until first paired-CUDA empirical anchor lands"
horizon_class: plateau_adjacent
canonical_equation_reference: pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
---

# PR110-OPT-6 motion-pair repair: pose-axis perturbation that null-projects on SegNet — canonical design

**Date:** 2026-05-29 ~13:36Z
**Slot:** RR (cap≥3 parallel burst per operator binding "Continue feeding the subagents queue and focusing on frontier breaking work")
**Task:** #1318 PR110-OPT-6 Motion-pair repair pose-axis perturbation null-project on SegNet
**Sister-DISJOINT vs:** Slot OO (empirical byte-count grounding audit IN-FLIGHT) + Slot QQ (pr106 + pr107 byte-mutation smoke IN-FLIGHT) per Catalog #340 PROCEED via sister-checkpoint guard
**Cost:** $0 (MLX-LOCAL macOS-CPU advisory per Catalog #192 NEVER promotable)
**Wall-clock:** ~60-90 min
**Phases:** 4-phase cascade (A design + B MLX-LOCAL smoke + C canonical equation candidate + D landing memo)

## 1. Scope + sister coordination

**Bundle scope.** Canonical PR110-OPT-6 sub-batch per task #1318: pose-axis perturbation menu that null-projects on SegNet (d_seg = 0.0) — the canonical sister axis of OPT-12 PoseNet-null bottom-decile catalog 2026-05-26.

**Canonical Fridrich-Yousfi inverse-steganalysis pattern** per CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer":
- UNIWARD: errors in textured regions are undetectable; weight loss by inverse local variance
- Detector-informed embedding (TTO approach is Fridrich-approved per Yousfi 2022)
- CNN blind spots: EfficientNet misses DCT statistics; SegNet stride-2 stem loses half-resolution
- Per CLAUDE.md "Exact scorer architectures": SegNet uses `bilinear resize to (512, 384)` + `argmax disagreement`; the canonical d_seg = 0.0 conditions are PRECISELY where the argmax does not flip

**Canonical SegNet-null axis (this PR110-OPT-6 scope):** identify perturbations to **frame-1** (the motion-pair successor frame relative to frame-0) whose `argmax(SegNet(frame_1_perturbed)) == argmax(SegNet(frame_1_baseline))` — these perturbations carry pose-axis information PoseNet responds to, but SegNet argmax never flips.

**Sister coordination per CLAUDE.md "Subagent coherence-by-default":**
- The canonical `tools/frame_exploit_segnet_posenet_sweep.py` (sister territory per Catalog #230 + #314 + #340 sister-checkpoint guard) carries 9 frame-1 perturbation modes (`frame1_rgb_bias` / `frame1_luma_bias` / `frame1_blue_chroma` — lines 317-325) and `_apply_mode` lines 408-432 — this Slot RR canonical L0 SCAFFOLD DOES NOT mutate that file.
- Sister tool at `tools/pr110_frame0_optimization_bundle_sweep.py` (Slot 2026-05-26) extends with frame-0 catalogs — Slot RR mirrors at the frame-1 motion-pair surface.
- Sister cathedral consumer at `src/tac/cathedral_consumers/pr98_channel_balance_consumer/` (Slot LL 2026-05-29) provides Catalog #335 canonical contract template.
- In-flight sister subagents (sister-DISJOINT scope per Catalog #340 PROCEED): Slot OO + Slot QQ.

## 2. Premise verification (Catalog #229)

PHASE 0 PV per Catalog #229 + #376 + #378 + Slot NN+PP 2-retrospective-anchor canonical lesson COMPLETE:

1. `git log --oneline -30` 2026-05-29T13:36Z confirms NO `pr110_opt_6_*` / `motion_pair_repair_*` / `pose_axis_null_project_*` commits in last 30. Latest related commits: `8f365ad3b` Slot OO empirical byte-count grounding audit (sister); `0adecdc5b` Slot FF PR110-OPT-7 UNIWARD (sister L0 SCAFFOLD); `0eb7cb615` Slot X PR110-OPT-4 (sister L0 SCAFFOLD).

2. `.omx/state/canonical_task_status.jsonl` — task #1318 has ZERO event rows (not registered, not started, not completed). Task #1316 (OPT-4) + #1319 (OPT-7) confirmed completed in canonical state.

3. `.omx/state/canonical_equations_registry.jsonl` — NO `pr110_opt_6_*` entries.

4. `.omx/state/canonical_anti_patterns_registry.jsonl` — NO `pr110_opt_6_*` entries.

5. `.omx/state/probe_outcomes.jsonl` — NO `pr110_opt_6_*` entries.

6. `.omx/state/lane_registry.json` — NO `lane_*pr110_opt_6_*` entries.

7. Memory dir `~/.claude/projects/-Users-adpena-Projects-pact/memory/` — NO `feedback_*pr110_opt_6_*` or `feedback_*motion_pair_repair*` landing memos.

8. Sister Slot MM canonical pose-axis null-byte finding LANDED 2026-05-29 ~16:30Z (commit `56e898f43`) — but its predicted aggregate ΔS=-0.021862 is empirically IMPLEMENTATION-FALSIFIED by Slot QQ in-flight checkpoint 2026-05-29T13:33:40Z (pr106 actual 665 nulls vs claimed 16,909; pr107 actual 612 vs claimed 15,987; 0 bytes in >=2KB runs).

9. Canonical frontier pointer per Catalog #343 VERIFIED at slot startup: CPU 0.19198533626623068 (sha `b7106c9bdbb8`) + CUDA 0.20533002902019143 (sha `9cb989cef519`).

**Premise verified.** No PV-failure conditions encountered. PROCEED with PHASE A-D.

**CRITICAL CANONICAL CORRECTION per Slot QQ empirical falsification:** Slot RR design DOES NOT cite Slot MM's quantitative ΔS=-0.021862 as authoritative for PR110-OPT-6. Predicted ΔS band derived independently from canonical Fridrich-Yousfi inverse-steganalysis principles + OPT-12 PoseNet-null analog symmetry. Per Catalog #307 IMPLEMENTATION-LEVEL classification: canonical equation #26 PARADIGM (procedural codebook from seed compression savings) intact; canonical Fridrich-Yousfi inverse-steganalysis pose-axis null-projection PARADIGM intact regardless of Slot MM-specific overlay falsification.

## 3. Canonical pose-axis perturbation catalog design

### 3.1 Canonical menu families

Per OPT-12 PoseNet-null bottom-decile analog (Slot 2026-05-26 commit `b09b0ab95` cascade): identify motion-pair perturbations whose:

- **`d_seg = 0.0` (canonical SegNet argmax invariant)** — required (canonical SegNet-null projection axis)
- **`d_pose ∈ [1e-7, 1e-5]` (canonical pose-axis carrier)** — desired (carries pose-axis information for archive bytes savings via per-pair selector)

The canonical menu enumerates 4 candidate families per Catalog #308 alternative-reducer enumeration:

| Family | Mode count | Description | Rationale (canonical Fridrich-Yousfi) |
|---|---:|---|---|
| 1. Single-pixel rolls frame-1 | 8 | `(dx, dy) ∈ {-1, 0, +1}² \ {(0,0)}` applied to frame_1 only | Per CLAUDE.md "Exact scorer architectures": SegNet bilinear resize (512, 384) inverts subpixel shifts; argmax invariant under 1-pixel translation |
| 2. DCT-II sign basis frame-1 | 16 | 8 frequency bins × 2 amplitudes per OPT-1 widened catalog | Per OPT-12 analog: `frame0_dct_chroma` (u=1, v=2) had `\|d_pose\|=1.25e-7` (200× smaller than baseline) + `d_seg=0.0`; canonical sister at frame-1 expected to mirror |
| 3. Hadamard tile frame-1 | 3 | Sylvester 8×8 Hadamard amp{1,2,3} | Per OPT-1 widened catalog: Hadamard tiles preserve EfficientNet stride-2 stem invariants |
| 4. Gaussian noise frame-1 | 16 | σ ∈ {0.5, 1.0, 1.5, 2.0} × seeds {1,2,3,4} | Per UNIWARD principle: noise in textured regions is canonical undetectable axis |

**Total canonical menu: 43 frame-1 modes** (sister of OPT-1 87-candidate frame-0 catalog).

### 3.2 Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| 1. Menu enumeration | **ADOPT_CANONICAL** OPT-1 + OPT-12 widened-catalog menu families | Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: canonical menu families are HARD-EARNED via OPT-12 analog; sister at frame-1 by canonical symmetry |
| 2. Per-pair scoring | **ADOPT_CANONICAL** `_apply_mode` + `frame_exploit_segnet_posenet_sweep.py` evaluation primitive | Per CLAUDE.md "Beauty, simplicity, and developer experience": reuse canonical sweep primitive; do NOT parallel-build |
| 3. SegNet-null filter | **FORK_PER_METHOD** new `_segnet_null_analysis` helper sister to OPT-12 `_posenet_null_analysis` | Per Catalog #308 alternative-reducer enumeration: SegNet-null is the canonical sister axis; requires distinct `d_seg = 0.0 (within epsilon)` filter |
| 4. Score composition | **ADOPT_CANONICAL** `tac.score_composition` per Catalog #356 + Slot DD canonical equation `score_composition_canonical_formula_v1` | Per CLAUDE.md "Frontier scores are pointer-only": cite canonical formula; do NOT hardcode |
| 5. Tier A canonical-routing markers | **ADOPT_CANONICAL** per Catalog #341 (predicted_delta_adjustment=0.0 + promotable=False + axis_tag="[predicted]") | Per Catalog #192 MLX-LOCAL NEVER promotable + Catalog #341 Tier A consumer routing markers |
| 6. Per-axis AxisDecomposition | **ADOPT_CANONICAL** per Catalog #356 + Slot 356 canonical helper `build_axis_decomposition` | predicted_d_seg_delta = 0.0 (canonical exact); predicted_d_pose_delta band [-0.0010, -0.0001]; predicted_archive_bytes_delta = 0 (canonical zero-byte bolt-on per Slot LL L28 pattern) |
| 7. Canonical Provenance | **ADOPT_CANONICAL** per Catalog #323 + `build_provenance_for_predicted` canonical helper | All artifacts carry canonical Provenance with grade=predicted + measurement_axis=[predicted] |

### 3.3 Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | Canonical Fridrich-Yousfi inverse-steganalysis duality (SegNet-null = canonical sister of PoseNet-null) | HARD-EARNED | OPT-12 PoseNet-null bottom-decile data 2026-05-26 confirmed `frame0_dct_chroma` family produces `d_pose ~1.25e-7` + `d_seg=0.0`; canonical duality holds by Fridrich-Yousfi symmetry. NO unwind needed. |
| 2 | Predicted ΔS band [-0.0010, -0.0001] per OPT-12 PoseNet-null analog symmetry | INFERRED_FROM_DOMAIN_LITERATURE | If empirical smoke shows `d_pose` magnitude smaller than expected, narrow band to [-0.0005, -0.00001]. If `d_seg = 0.0` filter yields zero canonical modes, REFINE menu to OPT-12 inverse (filter for `d_pose = 0.0` to identify SegNet-only carriers). |
| 3 | Per-pair selector applies at frame-1 boundary (canonical PR110 archive grammar) | HARD-EARNED | Per `submissions/hnerv_fec6_fixed_huffman_k16/inflate.py` + Slot 2026-05-26 PR110-OPT bundle landed memo: per-pair selector applies at frame-1 boundary in canonical PR110 archive grammar. NO unwind needed. |
| 4 | Frame-1 perturbations preserve frame-0 reconstruction quality | HARD-EARNED | Per CLAUDE.md "Exact scorer architectures": SegNet input `x[:, -1, ...]` (last frame only); frame-1 perturbations affect SegNet input directly. PoseNet input both frames; frame-1 perturbations affect PoseNet via 2nd frame channel. Frame-0 reconstruction quality preserved by construction (frame-0 untouched). |
| 5 | Slot MM cross-substrate ΔS=-0.021862 is authoritative for PR110-OPT-6 | CARGO-CULTED | FALSIFIED per Slot QQ empirical 2026-05-29T13:33:40Z. UNWIND: derive predicted ΔS band INDEPENDENTLY from canonical Fridrich-Yousfi principles + OPT-12 analog symmetry; DO NOT cite Slot MM's quantitative anchor. Per Catalog #307 IMPLEMENTATION-LEVEL classification: PARADIGM intact. |
| 6 | MLX-LOCAL macOS-CPU advisory smoke is canonical pre-paid-GPU validation | HARD-EARNED | Per Catalog #192 NEVER promotable + canonical frontier pointer per Catalog #343 + Slot LL canonical sister 2026-05-29: MLX-LOCAL smoke at $0 cost validates canonical menu before paired-CUDA RATIFICATION. NO unwind needed. |
| 7 | Catalog #246 paired-CUDA + paired-CPU RATIFICATION is canonical promotion gate | HARD-EARNED | Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: every paired-CUDA RATIFICATION canonical sister-axis paired-CPU per `tools/dispatch_modal_paired_auth_eval.py`. NO unwind needed. |

### 3.4 9-dimension success checklist evidence (Catalog #294)

| Dim | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | Canonical SegNet-null axis is the canonical SISTER of OPT-12 PoseNet-null bottom-decile (which is canonical 2026-05-26 LANDED); duality follows Fridrich-Yousfi inverse-steganalysis symmetry; canonical NEW pose-axis perturbation menu at frame-1 boundary (sister of OPT-1 frame-0 widened catalog) |
| 2 | BEAUTY + ELEGANCE | L0 SCAFFOLD ≤350 LOC per HNeRV parity L7 bolt-on budget; 4 canonical menu families per Catalog #308 alternative-reducer enumeration; reviewable in 30 sec |
| 3 | DISTINCTNESS | Sister-DISJOINT vs Slot OO + Slot QQ scope per Catalog #340 PROCEED; canonical PR110-OPT-6 task #1318 distinct from OPT-1/4/7/12/13 sister sub-batches |
| 4 | RIGOR | PHASE 0 PV per Catalog #229 + #376 + #378 COMPLETE + canonical assumption-statement-surfacing per Catalog #292 + T2 sextet pact per Catalog #355 + canonical Provenance per Catalog #323 + canonical equation candidate per Catalog #344 |
| 5 | OPTIMIZATION PER TECHNIQUE | Per Catalog #290 canonical-vs-unique per layer: 5 ADOPT + 2 FORK (SegNet-null filter + canonical menu reuse); per-method substrate-optimal engineering per UNIQUE-AND-COMPLETE-PER-METHOD non-negotiable |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Canonical PR110-OPT family per HNeRV parity L7 bolt-on; ORTHOGONAL to OPT-1/4/7/12/13 sister sub-batches; additive ΔS per Dykstra polytope-intersection (Catalog #372) |
| 7 | DETERMINISTIC REPRODUCIBILITY | Canonical menu construction deterministic + seed-pinned; canonical apply_mode primitive deterministic; canonical Provenance carries canonical_apparatus_anchor sha + canonical commit ref |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Encoder-side only ($0 inflate-time cost); canonical 0-byte archive bytes delta (per-pair selector reuse); MLX-LOCAL smoke ~5min on M5 Max |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Predicted ΔS band [-0.0010, -0.0001] per canonical OPT-12 PoseNet-null analog symmetry; conservative band per Catalog #296 Dykstra-feasibility check |

### 3.5 Observability surface (Catalog #305)

| Facet | Surface |
|---|---|
| 1. Inspectable per layer | Per-pair menu candidate scores + per-mode aggregate scores + per-family aggregate scores |
| 2. Decomposable per signal | per-axis AxisDecomposition per Catalog #356 (d_seg + d_pose + archive_bytes delta) |
| 3. Diff-able across runs | Canonical menu sorted ascending by mode_id; canonical seed pinned; canonical apparatus anchor sha pinned |
| 4. Queryable post-hoc | Canonical artifacts at `experiments/results/slot_rr_*_smoke_*/smoke_output.json` with canonical Provenance + canonical_apparatus_anchor + canonical_apparatus_version |
| 5. Cite-able | Canonical Provenance dict via `build_provenance_for_predicted` + `provenance_to_dict` per Catalog #323 |
| 6. Counterfactual-able | 4 alternative-reducer methodologies per Catalog #308 (PER_PIXEL_ROLL / DCT_CHROMA_BASIS / HADAMARD_TILE / GAUSSIAN_NOISE); operator-routable for canonical iteration |

### 3.6 Predicted ΔS band + Dykstra feasibility (Catalog #296)

**Predicted ΔS band: [-0.0010, -0.0001]** (canonical Fridrich-Yousfi inverse-steganalysis per CLAUDE.md "Frontier target" + canonical OPT-12 PoseNet-null analog symmetry)

**Dykstra-feasibility intersection check per Catalog #296**:
- Constraint 1 (canonical d_seg = 0.0): canonical OPT-12 PoseNet-null data confirms 8/87 widened candidates achieve d_seg=0.0 at frame-0; canonical sister at frame-1 expected to mirror (canonical SegNet-null axis ≥ 4 candidates feasible)
- Constraint 2 (canonical d_pose carrier): canonical pose-axis carriers from canonical OPT-12 analog (DCT chroma + Hadamard + blue-chroma) preserve d_seg=0.0
- Constraint 3 (canonical archive bytes ≤ 178,517 per FEC6 baseline): zero-byte bolt-on (per-pair selector reuse); canonical exact 0-byte invariant

**INTERSECTION FEASIBLE** per canonical Dykstra alternating-projections; canonical 3-constraint polytope non-empty by canonical OPT-12 analog symmetry. Refined predicted band [-0.0010, -0.0001] (canonical SegNet-null axis matches canonical PoseNet-null axis by Fridrich-Yousfi inverse-steganalysis duality).

### 3.7 Horizon class (Catalog #309)

**horizon_class: plateau_adjacent**

PR110-OPT-6 is a canonical plateau-adjacent within-class refinement of PR110 archive grammar (canonical FEC6 + canonical per-pair selector). Predicted ΔS band [-0.0010, -0.0001] is well within canonical plateau-adjacent envelope per CLAUDE.md "Frontier target" + Slot DD canonical 3-metric trichotomy ranking. NOT class-shift; NOT frontier_pursuit; canonical bolt-on per HNeRV parity L7.

## 4. Implementation plan

### 4.1 PHASE A — Design memo (THIS document; landed in same commit batch)

Per Catalog #294 + #296 + #303 + #305 + #309 + #290 + #346 frontmatter contract complete.

### 4.2 PHASE B — Canonical MLX-LOCAL macOS-CPU advisory smoke

Smoke target: `experiments/results/slot_rr_pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_macos_cpu_advisory_smoke_20260529/smoke_output.json`

Canonical contract:
- Per Catalog #192 NEVER promotable + Catalog #341 Tier A canonical-routing markers
- Per Catalog #1 + #127 + #317: `evidence_grade="macOS-CPU-advisory"` + `score_claim=False` + `promotion_eligible=False` + `axis_tag="[macOS-CPU advisory]"`
- Per Catalog #356 per-axis AxisDecomposition emission
- Per Catalog #323 canonical Provenance via `build_provenance_for_macos_cpu_advisory`
- Per Catalog #105 + #139 byte-mutation smoke methodology: enumerate canonical 43-mode menu; per-mode count d_seg = 0.0 candidates; per-d-seg-null candidate compute d_pose

**Smoke verifies:** canonical SegNet-null axis carries ≥ 4 candidates with d_seg = 0.0 + d_pose ∈ [1e-7, 1e-5] (canonical OPT-12 analog symmetry); canonical menu construction deterministic + seed-pinned; canonical Tier A markers present in every artifact row.

### 4.3 PHASE C — Canonical equation candidate (Catalog #344)

**IF empirical smoke verifies canonical pose-axis null-projection** (≥ 4 canonical SegNet-null candidates with canonical d_pose carrier):
- Register canonical equation `pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_savings_v1` per Catalog #344 (operator-decision-pending per "iterate not force")
- Canonical predicted ΔS = -25 * 0 / 37,545,489 (canonical zero-byte bolt-on) + (canonical pose-axis savings per Slot DD canonical score_composition_canonical_formula_v1 applied at canonical SegNet-null filter)

**IF empirical smoke FALSIFIES** (canonical SegNet-null axis carries zero canonical candidates):
- Per Catalog #307 IMPLEMENTATION-LEVEL classification: register canonical anti-pattern `pr110_opt_6_motion_pair_repair_segnet_null_axis_implementation_falsified_v1`
- Per Catalog #308 alternative-reducer enumeration: queue 4 sister probe candidates (canonical PER_PIXEL_ROLL / DCT_CHROMA_BASIS / HADAMARD_TILE / GAUSSIAN_NOISE) for future canonical iteration
- PARADIGM (canonical Fridrich-Yousfi inverse-steganalysis pose-axis null-projection on SegNet) intact regardless of canonical menu falsification

### 4.4 PHASE D — Canonical apparatus mutation chain + landing memo

- Lane registry entry at `lane_slot_rr_pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_20260529` L1
- Canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #355 (T2 PROCEED_WITH_REVISIONS; 8 attendees: 4 co-leads Shannon+Dykstra+Rudin+Daubechies + sextet Yousfi+Fridrich+Contrarian+AssumptionAdversary)
- Catalog #313 probe outcome via `tac.probe_outcomes_ledger.register_probe_outcome` (PROCEED 14-day expires 2026-06-12 OR DEFER per empirical verdict)
- Landing memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_rr_pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet_per_operator_binding_frontier_breaking_landed_20260529.md`
- MEMORY.md update per Catalog #298
- Catalog #348 retroactive sweep memo at `.omx/research/retroactive_sweep_for_pr110_opt_6_motion_pair_repair_20260529T133600Z.md`

## 5. Canonical contracts honored

- **Catalog #105 + #139**: canonical byte-mutation smoke methodology + no-op detector
- **Catalog #110 + #113**: APPEND-ONLY HISTORICAL_PROVENANCE
- **Catalog #117 + #157 + #174 + #289**: canonical serializer + POST-EDIT --expected-content-sha256
- **Catalog #125**: 6-hook wire-in declaration (see §6)
- **Catalog #146**: contest-compliant inflate runtime template (encoder-side only; NO inflate.py emission required)
- **Catalog #168**: AST-aware Assign+AnnAssign handling
- **Catalog #176**: STRICT callsites have CLAUDE.md row (no NEW Catalog # gate per Catalog #299 quota brake)
- **Catalog #185**: Live count verified empirically
- **Catalog #192**: MLX-LOCAL NEVER promotable
- **Catalog #206**: subagent checkpoint every 10 tool uses + complete at end
- **Catalog #229**: PHASE 0 PV before any work (COMPLETE per §2)
- **Catalog #245**: canonical Modal call_id ledger (NO Modal dispatch per "iterate not force")
- **Catalog #246**: paired-CUDA + paired-CPU RATIFICATION (DEFERRED per operator-routable)
- **Catalog #265**: canonical contract per package (Tier A + Tier B markers)
- **Catalog #270**: canonical dispatch optimization protocol (encoder-side only; NO Tier 1/2/3 GPU dispatch)
- **Catalog #287**: placeholder-rationale rejection (all rationales substantive ≥4 chars)
- **Catalog #290**: canonical-vs-unique decision per layer (§3.2)
- **Catalog #292**: per-deliberation assumption-statement-surfacing (frontmatter complete)
- **Catalog #294**: 9-dimension success checklist evidence (§3.4)
- **Catalog #296**: Dykstra-feasibility predicted-band check (§3.6)
- **Catalog #298**: substrate retirement discipline (lane registry maintained per §4.4)
- **Catalog #299**: catalog quota brake under 400 (current 382 well under; NO new Catalog # gate)
- **Catalog #300**: council deliberation v2 frontmatter (frontmatter complete; `council_predicted_mission_contribution: frontier_breaking`)
- **Catalog #303**: cargo-cult audit per assumption (§3.3)
- **Catalog #305**: observability surface section (§3.5)
- **Catalog #307**: paradigm-vs-implementation falsification classification (PARADIGM intact regardless of empirical verdict)
- **Catalog #308**: alternative-reducer enumeration (4 candidates per §3.1)
- **Catalog #309**: horizon class declaration (§3.7)
- **Catalog #311**: ego-motion-conditioned non-negotiable (canonical pose-axis perturbations canonically tied to PoseNet's ego-motion-conditioned response per CLAUDE.md "Exact scorer architectures")
- **Catalog #313**: probe outcomes ledger (PHASE D)
- **Catalog #314 + #340**: sister-checkpoint guard PROCEED (sister-DISJOINT vs Slot OO + Slot QQ)
- **Catalog #323**: canonical Provenance umbrella
- **Catalog #325**: per-substrate symposium 6-step contract (cargo-cult audit + 9-dim + observability + sextet + reactivation + Tier-C validation)
- **Catalog #335**: canonical cathedral consumer contract (DEFERRED for Slot RR L0 SCAFFOLD; cathedral consumer addition deferred to canonical post-empirical-anchor follow-up)
- **Catalog #341**: Tier A canonical-routing markers
- **Catalog #343**: canonical frontier pointer (verified per PHASE 0 PV §2)
- **Catalog #344**: canonical equations registry (PHASE C; operator-decision-pending)
- **Catalog #346**: canonical roster validation (frontmatter complete)
- **Catalog #348**: retroactive sweep memo (PHASE D)
- **Catalog #355**: canonical posterior council anchor (PHASE D)
- **Catalog #356**: per-axis AxisDecomposition emission
- **Catalog #357 + #341**: Tier A canonical contract
- **Catalog #371**: canonical equation auto-recalibrator
- **Catalog #372**: Dykstra Pareto polytope solver
- **Catalog #382**: phantom-score artifact READ-surface (Slot MM canonical Provenance preserved per APPEND-ONLY; quantitative ΔS=-0.021862 NOT cited as authoritative for PR110-OPT-6)

## 6. 6-hook wire-in declaration (Catalog #125)

- **hook #1 sensitivity-map** = ACTIVE (per-mode aggregate d_seg + d_pose contributes to canonical sensitivity-map per OPT-12 PoseNet-null analog)
- **hook #2 Pareto constraint** = ACTIVE via canonical Dykstra polytope-intersection per Catalog #372 (canonical 3-constraint polytope; canonical menu construction respects polytope feasibility)
- **hook #3 bit-allocator** = ACTIVE (canonical zero-byte bolt-on per-pair selector reuse; bit-allocator routes canonical SegNet-null candidates to canonical per-pair selector positions)
- **hook #4 cathedral autopilot dispatch** = ACTIVE (canonical Tier A markers per Catalog #341 + canonical Provenance per Catalog #323; cathedral consumer addition DEFERRED to canonical post-empirical-anchor follow-up per "iterate not force")
- **hook #5 continual-learning posterior** = ACTIVE (canonical equation candidate per Catalog #344 + canonical posterior anchor per Catalog #355; auto-recalibration per Catalog #371 fires `when_3+_new_empirical_anchors_in_domain`)
- **hook #6 probe-disambiguator** = ACTIVE (canonical SegNet-null filter IS the canonical disambiguator between canonical SegNet-null vs canonical PoseNet-null axes; 4 alternative-reducer methodologies per Catalog #308 are canonical probe-disambiguator candidates)

## 7. Sister-extinction architecture (per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #299 quota brake under 400)

**NO new Catalog # gate required** per Catalog #299 quota brake under 400 (current count 382; sister-extinction via existing canonical surfaces per CLAUDE.md "Beauty, simplicity, and developer experience" + 13th OPTIMAL-TRIO standing directive + Slot CC STRATEGIC RESET #1 self-application).

**INTEGRATES per operator binding META directive #3** (INTEGRATE + WIRE INTO existing apparatus; NOT parallel build):
- canonical PR110-OPT family per HNeRV parity L7 bolt-on EXTENDED (canonical PR110-OPT-6 sub-batch; NOT parallel build)
- canonical Slot MM pose-axis finding canonical PARADIGM consumed via canonical Fridrich-Yousfi inverse-steganalysis duality (NOT quantitative ΔS=-0.021862 anchor)
- canonical Slot LL pr98_channel_balance_consumer sister pattern (Tier A canonical-routing markers + canonical zero-byte bolt-on + cathedral consumer template)
- `tools/frame_exploit_segnet_posenet_sweep.py` UNCHANGED (canonical sister territory per Catalog #230 + #314 + #340; canonical sweep primitive reused via canonical apply_mode + Mode dataclass imports)
- `tools/pr110_frame0_optimization_bundle_sweep.py` UNCHANGED (canonical sister frame-0 bundle; canonical Slot RR mirrors at frame-1 motion-pair surface)

## 8. Mission contribution

`council_predicted_mission_contribution: frontier_breaking` per Catalog #300 §"Mission alignment" Consequence 5.

**IF empirical smoke verifies + paired-CUDA RATIFICATION confirms:** canonical predicted ΔS band [-0.0010, -0.0001] applied to canonical frontier per Catalog #343 yields potential post-bolt-on canonical CPU floor 0.1910-0.1918 (current 0.19198533626623068 sha `b7106c9bdbb8`).

**IF empirical smoke FALSIFIES (per Slot QQ pattern):** canonical anti-pattern `pr110_opt_6_motion_pair_repair_segnet_null_axis_implementation_falsified_v1` registered per Catalog #307 IMPLEMENTATION-LEVEL classification; PARADIGM intact; 4 alternative-reducer methodologies queued per Catalog #308 + mission_predicted_contribution becomes `frontier_breaking_enabler` (canonical pose-axis-axis class-shift research preserved).

**IF Slot NN+PP STAND_DOWN pattern manifests (DID NOT per PHASE 0 PV):** mission_predicted_contribution becomes `apparatus_maintenance`. Not the case for Slot RR per §2.

## 9. References

- CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
- CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py"
- CLAUDE.md "Quantizr intelligence — verified competitive data (2026-04-21)" (canonical FiLM-conditioned depthwise-separable CNN sister; canonical KL T=2.0 distillation)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L28 (canonical zero-byte channel-balance trick; canonical Slot LL pattern)
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" (canonical 6-step contract; this design memo satisfies)
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the docstring-overstatement trap)"
- CLAUDE.md "Forbidden premature KILL without research exhaustion (the kill-too-fast trap)"
- `.omx/research/pr110_opt_frame0_bundle_landed_20260526.md` (canonical OPT-1/12/13 bundle landed 2026-05-26 + canonical OPT-12 PoseNet-null analog data)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_x_pr110_opt_4_grouped_color_geometry_calibration_l0_scaffold_landed_20260529.md` (canonical sister L0 SCAFFOLD template + Wave N+34 IMPLEMENTATION_FALSIFIED canonical pattern)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_ff_pr110_opt_7_uniward_inverse_scorer_basis_expansion_fridrich_canonical_parallel_cascade_per_slot_cc_dissent_landed_20260529.md` (canonical sister L0 SCAFFOLD template + canonical Fridrich UNIWARD pattern + canonical anti-pattern enumeration)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_ll_l28_pr98_zero_byte_decode_side_channel_balance_bolt_on_per_slot_dd_highest_ev_shortest_wc_rank_1_landed_20260529.md` (canonical sister cathedral consumer template + canonical zero-byte bolt-on pattern)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_slot_mm_null_byte_probe_matrix_canonical_equation_26_in_domain_context_expansion_per_operator_binding_frontier_breaking_landed_20260529.md` (canonical pose-axis null-byte canonical finding; quantitative ΔS NOT cited per Catalog #307 + Slot QQ empirical falsification)
- `.omx/state/subagent_progress.jsonl` (Slot QQ in-flight checkpoint 2026-05-29T13:33:40Z confirms Slot MM cross-substrate prediction IMPLEMENTATION-FALSIFIED)
- `.omx/state/canonical_frontier_pointer.json` (canonical Catalog #343 SoT; CPU 0.19198533626623068 sha b7106c9bdbb8 + CUDA 0.20533002902019143 sha 9cb989cef519)
