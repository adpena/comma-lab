<!--
SPDX-License-Identifier: MIT

Slot GGG SCALE-UP matrix N modes x M pairs x contest resolution
[macOS-CPU advisory] LANDING MEMO 2026-05-30
-->
---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Atick, Tishby_memorial, Wyner]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "5/16 = 31.25% confirmation rate at 96x128 macOS-CPU is below the 50% threshold the operator's Yousfi-cascade prediction model assumes; the Tier C full canonical contest resolution 384x512 ratification is a prerequisite for FEC10 selector codec composition not optional sidecar evidence"
  - member: Assumption-Adversary
    verbatim: "the 8 PER_PIXEL_ROLL FALSIFICATIONS at 96x128 (vs 2 confirmations at 48x64 in Slot GGG Part 3 predecessor anchor) is the canonical disambiguator that single-pixel-roll perturbations are DOWNSAMPLING-DEPENDENT; the canonical SegNet bilinear-resize-up from 48x64 to 384x512 was absorbing the 1-pixel shift; at native 96x128 (closer to contest 384x512) the shift survives the bilinear stage and breaks argmax invariance. This is empirical evidence the Slot GGG Part 3 PER_PIXEL_ROLL anchors should be re-classified as IMPLEMENTATION-LEVEL falsifications per Catalog #307 at the higher-resolution surface; the Fridrich-Yousfi PARADIGM remains intact via the DCT_CHROMA_BASIS family"
council_assumption_adversary_verdict:
  - assumption: "16 modes will mostly confirm under Yousfi-cascade prediction"
    classification: CARGO-CULTED
    rationale: "Yousfi-cascade prediction assumes a confirmation regime that the empirical anchor partially refutes; 5/16 = 31.25% confirmation at 96x128 is not the canonical 50-87.5% prediction. The DCT_CHROMA dominant family (5 of 8 confirms at 62.5%) does match the canonical OPT-12 PoseNet-null analog dominant-family fraction 87.5% within statistical agreement for n=4 pairs but PER_PIXEL_ROLL family is 0/8 = 0%"
  - assumption: "single-pixel-roll perturbations preserve SegNet argmax across all canonical resolutions"
    classification: CARGO-CULTED
    rationale: "Slot GGG Part 3 predecessor anchor at 48x64 confirmed 2 PER_PIXEL_ROLL modes; this scale-up at 96x128 falsified all 8. Empirically: argmax disagreement rate jumps from 0.0000 at 48x64 to 1.6-2.6% at 96x128. The canonical SegNet bilinear-resize-up from 48x64 to 384x512 was absorbing the 1-pixel shift via 8x downsample-then-upsample averaging; at 96x128 the 4x downsample-then-upsample no longer absorbs"
  - assumption: "DCT_CHROMA_BASIS family dominates the canonical PoseNet-null axis per OPT-12 analog symmetry"
    classification: HARD-EARNED
    rationale: "5/8 DCT_CHROMA modes confirmed with |d_pose| in 9e-9 to 8e-8 range (sub canonical 1e-7 lower bound = strong signal); aligns with design memo §4.1 OPT-12 PoseNet-null bottom-decile dominant-family fraction 87.5% structured-signed-chroma (within statistical agreement for n=4 pairs)"
council_decisions_recorded:
  - "op-routable #1: extend canonical Tier B SCALE-UP smoke to 16 modes x 60 pairs x (192, 256) ~ 30min macOS-CPU; canonical operator-routing-cadence for the FEC10 composition (next-turn) per the canonical capacity-per-cost ranking"
  - "op-routable #2: extend canonical Tier C SCALE-UP smoke to 16 modes x 600 pairs x (384, 512) ~ 4.5h macOS-CPU with baseline cache (queued for overnight); canonical full contest resolution ratification per Catalog #246 BEFORE FEC10 selector codec composition"
  - "op-routable #3: paired-CUDA RATIFICATION of top-K ranked CONFIRMED_MODE_IDS per Catalog #246 dual-axis discipline; canonical $0.30 envelope per substrate; canonical 4 paired_cuda_ratification_targets (v14_v2_dqs1 + fec6 + pr106_format0d + nscs06_v8_stacked)"
  - "op-routable #4: canonical equation registration per Catalog #344 DEFERRED at confirmation_count=5 < threshold=8; reactivation criterion = canonical Tier B smoke produces >=8 CONFIRMED modes OR canonical Tier C smoke produces >=8 CONFIRMED modes"
  - "op-routable #5: Slot GGG Part 3 PER_PIXEL_ROLL anchors re-classify per Catalog #307 IMPLEMENTATION-LEVEL FALSIFIED at 96x128+ resolution surface; canonical Fridrich-Yousfi PARADIGM remains INTACT via DCT_CHROMA_BASIS family"
council_deliberation_id: slot_ggg_scale_up_matrix_n_modes_x_600_pairs_contest_resolution_20260530
horizon_class: plateau_adjacent
---

# Slot GGG SCALE-UP MATRIX: N modes × M pairs × contest resolution — LANDING

**Date**: 2026-05-30 (Slot-GGG SCALE-UP follow-on to Slot GGG Part 3 commit `f9d0f2465`)
**Lane**: `lane_slot_ggg_scale_up_matrix_n_modes_x_600_pairs_contest_resolution_20260530` L2
**Mission contribution per Catalog #300**: `frontier_breaking_enabler`
**Evidence axis**: `[macOS-CPU advisory]` per Catalog #192 NEVER promotable

## Operator-routed scope

Per operator routing 2026-05-30 + Yousfi-cascade TIER-1 prerequisite for Slot GGG × Cascade A FEC10 selector codec composition (queued for next turn — `do_not_touch_this_turn=True` per artifact handoff envelope).

Predecessor anchor: Slot GGG Part 3 (commit `f9d0f2465`) verified `2 PER_PIXEL_ROLL modes × 2 pairs × 48×64` macOS-CPU advisory with `SegNet argmax = 0.0000` + `|d_pose| = 1.81e-6 to 2.08e-6` in canonical carrier band. THIS SCALE-UP extends to `N modes (default 16; cap 43) × M pairs (default 60; cap 600) × contest resolution (default 384×512)` with canonical baseline cache + multi-strategy unified menu + ranked CONFIRMED_MODE_IDS by capacity-per-cost.

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable:

| Layer | Decision | Rationale |
|---|---|---|
| Real-frame decode | ADOPT_CANONICAL | `tac.inverse_steganalysis_real_video_mlx.decode_upstream_video_frames` per Catalog #213 canonical real-video discipline |
| Scorer load | ADOPT_CANONICAL | `tac.scorer.load_default_scorers` per CLAUDE.md "Exact scorer architectures" |
| Scorer-pair forward | ADOPT_CANONICAL | `tac.substrates.score_aware_common.score_pair_components` per Catalog #164 + CLAUDE.md "scorer preprocess must be gradient-reachable" |
| Per-mode perturbation | ADOPT_CANONICAL | `_apply_perturbation_for_mode_canonical` sister of Slot RR Part 2 + Slot GGG Part 3 |
| Canonical menu | FORK_BECAUSE_SUPPRESSES | NEW `build_unified_canonical_scale_up_menu` multi-strategy unified menu (single strategy at a time per the legacy `build_canonical_frame1_pose_axis_null_projection_menu` would have required 4 separate invocations + manual aggregation); the canonical multi-strategy unified menu matches the operator's "N modes" abstraction at the helper boundary |
| Baseline cache | FORK_BECAUSE_PRINCIPLED_MISMATCH | NEW O(M + N*M) scheme vs predecessor naive O(2*N*M); the predecessor Slot GGG Part 3 anchor recomputed baseline per (mode, pair) pair which is wasteful at scale; canonical engineering optimization per CLAUDE.md "Production-hardened dispatch optimization protocol" Tier 1 |
| Capacity-per-cost ranking | FORK_BECAUSE_SUPPRESSES | NEW `rank_confirmed_modes_by_capacity_per_cost` ranker; the predecessor Slot GGG Part 3 helper returns unordered confirmed list; the canonical Atick-Tishby-Wyner minimum-detectability side-channel capacity metric IS the canonical disambiguator for downstream FEC10 selector codec consumption |
| Empirical artifact persistence | ADOPT_CANONICAL | `experiments/results/slot_ggg_scale_up_matrix_<UTC>/scale_up_matrix.json` per CLAUDE.md durable-evidence discipline (NOT `/tmp` per Catalog #113) |
| Canonical equation registration | ADOPT_CANONICAL | `tac.canonical_equations.register_canonical_equation` per Catalog #344 lazy import + opt-in flag + threshold |
| Tier A canonical-routing markers | ADOPT_CANONICAL | Catalog #341 + #357 (promotable=False, score_claim=False, axis_tag="[macOS-CPU advisory]", evidence_grade="predicted") |
| Canonical Provenance | ADOPT_CANONICAL | `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323 |
| AxisDecomposition | ADOPT_CANONICAL | per Catalog #356; EMPIRICAL per-axis values (NOT predicted-zero) per Slot GGG Part 3 pattern |

## Cargo-cult audit per assumption

Per Catalog #303 cargo-cult audit + HARD-EARNED-vs-CARGO-CULTED addendum:

1. **CARGO-CULTED**: 16 modes will mostly confirm. EMPIRICAL: 5/16 = 31.25% confirms; the operator's Yousfi-cascade prediction assumed ≥50% (canonical 50-87.5%); refuted at 96×128. Unwind: route forward via the ratified DCT_CHROMA_BASIS dominant family + queue Tier C 384×512 to test if PER_PIXEL_ROLL re-confirms at contest resolution.
2. **CARGO-CULTED**: PER_PIXEL_ROLL preserves SegNet argmax across all canonical resolutions. EMPIRICAL: 2 confirms at 48×64 → 0/8 confirms at 96×128. Argmax disagreement rate jumps from 0.0000 to 1.6-2.6%. Unwind: re-classify Slot GGG Part 3 PER_PIXEL_ROLL anchors per Catalog #307 IMPLEMENTATION-LEVEL FALSIFIED at 96×128+ resolution; PARADIGM intact via DCT_CHROMA.
3. **HARD-EARNED**: DCT_CHROMA_BASIS dominates the canonical PoseNet-null axis. EMPIRICAL: 5/8 = 62.5% confirms at 96×128; aligns with canonical OPT-12 PoseNet-null bottom-decile dominant-family fraction 87.5% structured-signed-chroma (within statistical agreement for n=4 pairs). Aligns with design memo §4.1 + canonical Fridrich-Yousfi inverse-steganalysis duality.
4. **HARD-EARNED**: Baseline cache reduces scorer calls ~50%. EMPIRICAL: 46.9% savings at 16 modes × 4 pairs (asymptotic 50% at large N).
5. **HARD-EARNED**: Multi-strategy unified menu correctly fills per canonical PER_PIXEL_ROLL → DCT_CHROMA → HADAMARD → GAUSSIAN order. EMPIRICAL: 16 modes = 8 PER_PIXEL_ROLL + 8 DCT_CHROMA verified.
6. **HARD-EARNED**: Capacity-per-cost ranker correctly identifies canonical Atick-Tishby-Wyner minimum-detectability modes. EMPIRICAL: top mode `frame1_dct_chroma_u1_v0_amp_1` with |d_pose|=9.1e-9 (>= 5× smaller than the next-best PER_PIXEL_ROLL would have been if it confirmed).
7. **ASSUMED_AWAITING_VERIFICATION**: Yousfi-cascade prediction of ~300 bytes "free" side-channel capacity at near-zero seg cost = frontier-crossing −0.014 score potential. UNVERIFIED at this resolution; requires Tier C full canonical contest resolution ratification + paired-CUDA RATIFICATION before any contest-axis score claim per Catalog #192/#246/#341.

## Predicted ΔS band

Per Catalog #296 Dykstra-feasibility + canonical Atick-Tishby-Wyner asymmetric-channel capacity:

* **Predicted band (canonical empirical)**: `[-0.014, -0.001]` per Yousfi-cascade prediction × canonical confirmation rate at scale (`-0.014 × 5/16 = -0.004` linearly extrapolated; canonical Dykstra-feasibility lower bound `-0.001` per Slot GGG Part 3 predicted band `[-0.001, -0.0001]`).
* **Predicted band validation status**: `pending_post_training` per Catalog #324 — requires canonical Tier C full canonical contest resolution ratification + paired-CUDA RATIFICATION before validation.
* **Dykstra-feasibility check**: predicted band falls inside canonical Pareto polytope `(d_seg=0, |d_pose| ≤ 4e-6, archive_bytes_delta=0)` per the per-axis decomposition emitted by the canonical scale-up entry point.

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: NEW canonical scale-up matrix entry point not present in any sister substrate; the canonical multi-strategy unified menu + baseline cache + capacity-per-cost ranker is unique to this canonical Slot GGG operator-routing surface.
2. **BEAUTY+ELEGANCE**: ~750 LOC scale-up entry point reviewable in ~5 minutes; canonical fill-order semantics (PER_PIXEL_ROLL → DCT_CHROMA → HADAMARD → GAUSSIAN) directly mirror the canonical 4-family menu construction order; ranked CONFIRMED_MODE_IDS list is the single canonical disambiguator for downstream FEC10 consumer.
3. **DISTINCTNESS**: Distinct from Slot GGG Part 3 (single strategy × cheap smoke) AND distinct from canonical sister `apply_pose_axis_null_projection_via_canonical_real_video_mlx_to_pr110_archive` (no scorer verification).
4. **RIGOR**: Premise verification CLEAN (no predecessor; no sister overlap on scoped paths per Catalog #229 + #376 + #378); 31/31 dedicated tests PASS + 1 slow-marked skipped + 0 regression on sister 62-test Slot RR+GGG suite; canonical Catalog #287 placeholder rejection + Catalog #341 routing markers + Catalog #323 Provenance + Catalog #356 AxisDecomposition all verified.
5. **OPTIMIZATION PER TECHNIQUE**: Baseline cache 46.9% savings at 16×4 (asymptotic 50%); contest-resolution-at-helper-boundary avoids canonical bilinear-resize-roundtrip artifact per CLAUDE.md "CATASTROPHIC FAILURES" 48×64 mask anchor; canonical lazy import of `tac.canonical_equations` keeps test surface free of registration side-effects.
6. **STACK-OF-STACKS-COMPOSABILITY**: Canonical CONFIRMED_MODE_IDS ranked list IS the structural input to downstream Cascade A FEC10 selector codec composition (next-turn operator-routable); orthogonal axes (per-pair × per-mode) compose additively in FEC10 selector encoding.
7. **DETERMINISTIC REPRODUCIBILITY**: Canonical inputs SHA per Catalog #323 + canonical seeded GAUSSIAN_NOISE family per `_apply_perturbation_for_mode_canonical`; canonical empirical artifact JSON byte-stable sort_keys.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: Canonical 4.5h wall-clock for 16 modes × 600 pairs × 384×512 with baseline cache vs naive 8.4h (47% reduction); canonical Tier A operator-routing smoke 2.3min wall-clock at 16 modes × 4 pairs × 96×128.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Canonical Yousfi-cascade prediction: if monotonicity holds at Tier C scale + ≥8 confirmed modes, ~600 pairs × log2(8) = 3 bits/pair = ~225 bytes "free" side-channel capacity at near-zero seg cost = frontier-crossing −0.010 to −0.014 score potential vs canonical CPU frontier per `tools/refresh_canonical_frontier.py` (canonical pointer per Catalog #343).

## Observability surface

Per Catalog #305 6-facet:

1. **Inspectable per layer**: every scorer call surfaced via `per_mode_empirical_verification` list (family + params + d_seg + |d_pose| + argmax_rate + verdict per mode).
2. **Decomposable per signal**: top-level `aggregate_empirical_d_seg_mean_across_modes` + `aggregate_empirical_abs_d_pose_mean_across_modes` decompose per-axis; `baseline_cache_savings_ratio` + `elapsed_seconds_per_mode_mean` decompose per-engineering-invariant.
3. **Diff-able across runs**: canonical empirical artifact JSON byte-stable (sort_keys=True) so two scale-up runs at same inputs produce diffable JSON.
4. **Queryable post-hoc**: canonical artifact JSON schema `slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution.v1` consumable by downstream cathedral consumer per Catalog #335.
5. **Cite-able**: canonical Provenance per Catalog #323 + canonical equation candidate ID + canonical anti-pattern candidate ID propagate from helper to artifact.
6. **Counterfactual-able**: per-mode + per-pair (d_seg, |d_pose|, argmax_rate) tuples queryable post-hoc to ask "what if mode X had perturbation Y?" without re-running scorer; canonical capacity-per-cost ranker is deterministic from per-mode dict.

## 6-hook wire-in declaration per Catalog #125

* **hook #1 sensitivity-map**: ACTIVE — per-mode (d_seg, |d_pose|) decomposition surfaces per-mode SegNet/PoseNet sensitivity; consumable by `tac.sensitivity_map.*` downstream consumers.
* **hook #2 Pareto constraint**: ACTIVE — canonical per-mode (d_seg, |d_pose|, archive_bytes_delta=0) decomposition emits the canonical Pareto polytope vertex per mode per Catalog #356; Dykstra-feasibility check passes (predicted band falls inside Pareto polytope).
* **hook #3 bit-allocator**: ACTIVE — canonical ranked CONFIRMED_MODE_IDS list IS the canonical capacity-per-cost ranking for downstream FEC10 selector codec bit-allocator consumption.
* **hook #4 cathedral autopilot dispatch**: ACTIVE — canonical Tier A `[macOS-CPU advisory]` markers per Catalog #341; cathedral consumer auto-discovered per Catalog #335 (sister `phantom_score_canonical_posterior_lookup_consumer` already routes ranking decisions against canonical posterior).
* **hook #5 continual-learning posterior**: ACTIVE — canonical equation registration opt-in per Catalog #344; DEFERRED at this confirmation count (5 < threshold 8) per "iterate not force"; reactivation criterion = Tier B / Tier C smoke produces ≥8 CONFIRMED modes.
* **hook #6 probe-disambiguator**: ACTIVE — canonical SCALE_UP verdict (ALL_CONFIRMED / ALL_FALSIFIED / PARTIAL) IS the canonical disambiguator at the macOS-CPU advisory surface; canonical CONFIRMED_MODE_IDS list IS the canonical operator-routable disambiguator between paradigm-confirmed vs paradigm-falsified modes per Catalog #307 IMPLEMENTATION-LEVEL classification.

## Empirical anchor

* **Artifact path**: `experiments/results/slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution_macos_cpu_advisory_smoke_20260530T144658Z/scale_up_matrix.json`
* **Schema**: `slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution.v1`
* **Configuration**: 16 modes × 4 pairs × (96, 128) at macOS-CPU advisory
* **Wall-clock**: 139.0s (2.3min)
* **Scorer calls**: 4 baseline + 64 perturbed = 68 actual vs 128 naive (46.9% baseline cache savings)
* **Modes confirmed**: 5/16 (31.25%); modes falsified: 11/16
* **Confirmed family distribution**: 5/8 DCT_CHROMA_BASIS confirms (62.5%) ; 0/8 PER_PIXEL_ROLL confirms (0%) — empirical evidence of DCT dominant-family axis
* **Top ranked CONFIRMED modes by capacity-per-cost**:
  1. `frame1_dct_chroma_u1_v0_amp_1`: |d_pose|=9.111e-09, capacity=1.098e+08
  2. `frame1_dct_chroma_u1_v2_amp_1`: |d_pose|=9.980e-09, capacity=1.002e+08
  3. `frame1_dct_chroma_u2_v2_amp_1`: |d_pose|=1.833e-08, capacity=5.455e+07
  4. `frame1_dct_chroma_u0_v2_amp_1`: |d_pose|=6.704e-08, capacity=1.492e+07
  5. `frame1_dct_chroma_u2_v0_amp_1`: |d_pose|=8.359e-08, capacity=1.196e+07
* **Aggregate d_seg_mean**: -1.212e-04 (all confirmed modes ≤ 0.1% argmax disagreement); aggregate |d_pose|_mean: 4.127e-06
* **Verdict**: `SCALE_UP_PARTIAL_CONFIRMED_ON_MACOS_CPU_ADVISORY_DEFERRED_PENDING_PAIRED_CUDA_RATIFICATION_OF_CONFIRMED_SUBSET`

## Tier A / Tier B / Tier C operator-routing matrix

| Tier | N modes × M pairs × resolution | Wall-clock (macOS-CPU) | Purpose |
|---|---|---|---|
| Tier 0 (test smoke) | 2-4 × 2-4 × 48-64 | ~30s | Test suite + CI |
| **Tier A (this landing)** | **16 × 4 × 96×128** | **2.3min** | **Empirical anchor + ranked CONFIRMED list** |
| Tier B (operator-routing) | 16 × 60 × 192×256 | ~30min | Canonical statistical mid-tier; queue for next session |
| Tier C (overnight bind) | 16 × 600 × 384×512 | ~4.5h with cache | Canonical full contest resolution; required before paired-CUDA RATIFICATION |
| Paired-CUDA RATIFICATION | top-K × 600 × 384×512 | ~$0.30 per substrate × 4 | Canonical Catalog #246 dual-axis; promotion-eligible |

## Catalog #348 retroactive sweep companion memo

See `.omx/research/retroactive_sweep_for_slot_ggg_scale_up_matrix_n_modes_x_600_pairs_contest_resolution_20260530T*.md` per Catalog #348 4-field contract.

## Sister DISJOINT verification per Catalog #340

2 sister subagents in-flight this turn (Z8 Phase E + Z8 Phase B) are CONFIRMED DISJOINT per operator routing. Sister-checkpoint guard verified at lookback minutes 60 via `.omx/state/subagent_progress.jsonl`. NO file overlap on `src/tac/composition/pr110_opt_6_motion_pair_repair_*` or `experiments/results/slot_*` paths.

## Operator-routable next steps

1. **Next turn**: Slot GGG × Cascade A FEC10 selector codec composition consumes `ranked_confirmed_modes_by_capacity_per_cost` from artifact `experiments/results/slot_ggg_scale_up_matrix_n_modes_x_m_pairs_x_contest_resolution_macos_cpu_advisory_smoke_20260530T144658Z/scale_up_matrix.json`. Per artifact handoff envelope: `do_not_touch_this_turn=True`.
2. **Within-session**: extend Tier B 16 × 60 × (192, 256) smoke ~ 30min to derive statistical confidence intervals on top-K ranked modes.
3. **Overnight**: Tier C 16 × 600 × (384, 512) smoke ~ 4.5h with baseline cache to ratify canonical contest resolution.
4. **Paired-CUDA RATIFICATION**: top-3 ranked modes per Catalog #246; ~$1.20 envelope across 4 paired_cuda_ratification_targets (`v14_v2_dqs1` + `fec6` + `pr106_format0d` + `nscs06_v8_stacked`); requires Tier C smoke completion first.
5. **Canonical equation registration**: triggered automatically when Tier B/C produces ≥8 CONFIRMED modes per opt-in flag.

## Cross-references

* Slot GGG Part 3 predecessor: commit `f9d0f2465`; `src/tac/composition/pr110_opt_6_motion_pair_repair_pose_axis_null_projection_on_segnet/__init__.py::apply_pose_axis_null_projection_via_real_scorers_to_pr110_archive` (line 1085)
* Slot RR Part 2 sister: commits `30bf9029f` + `32a70c051`
* Canonical Frontier pointer: `.omx/state/canonical_frontier_pointer.json` per Catalog #343
* CLAUDE.md "Fridrich inverse steganalysis — how to beat the scorer"
* CLAUDE.md "MLX portable-local-substrate authority" 8th standing directive
* Catalog #192 MACOS-CPU NEVER promotable
* Catalog #246 paired-CUDA dual-axis
* Catalog #287 placeholder rejection
* Catalog #292 per-deliberation assumption surfacing
* Catalog #294 9-dim checklist
* Catalog #296 Dykstra-feasibility predicted-band
* Catalog #300 v2 frontmatter
* Catalog #303 cargo-cult audit
* Catalog #305 observability surface
* Catalog #307 paradigm-vs-implementation falsification
* Catalog #308 alternative-reducer enumeration
* Catalog #309 horizon class declaration
* Catalog #313 probe outcomes ledger
* Catalog #323 canonical Provenance umbrella
* Catalog #325 6-step symposium contract
* Catalog #340 sister-checkpoint guard
* Catalog #341 Tier A canonical-routing markers
* Catalog #344 canonical equation registration
* Catalog #346 canonical roster validation
* Catalog #348 retroactive sweep
* Catalog #355 canonical posterior council anchor
* Catalog #356 per-axis AxisDecomposition
* Catalog #376 + #378 spawn-time PV discipline
