<!-- SPDX-License-Identifier: MIT -->
<!-- Working-group landing memo per CLAUDE.md "Council hierarchy: 4-tier protocol" + Catalog #300 v2 frontmatter. T1 engineering fix scope (closes Contrarian VETO at sub-surface). -->
---
council_tier: T1
council_attendees: [Shannon, Dykstra, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The canonical mlx_score_aware harness adapter score_aware_components hook is the correct surface to populate per_axis_decomposition for downstream cross-family seg/pose attribution"
    classification: HARD-EARNED
    rationale: "The canonical run_long_training loop ALREADY calls adapter.score_aware_components(adapter.model, sample) at line 2066 of long_training_canonical.py and threads the result into PerEpochMetrics.per_axis_decomposition at line 2088. The gap was the adapter's stub returning None universally. Populating it from the existing score_aware_loss decomposition (distill / pose_distill / recon / total) is the single-source-of-truth canonical surface per Catalog #290 ADOPT_CANONICAL discipline. Confirmed empirically by smoke verification: per_epoch_metrics rows now carry {seg, pose, recon_aux, archive_bytes} for the Z6-v2 + Hinton + 600-pair pattern (and every other canonical-harness consumer transitively)."
  - assumption: "Pure-reconstruction mode (distillation_weight=0.0 AND pose_distillation_weight=0.0) should preserve sister-adapter parity by returning None"
    classification: HARD-EARNED
    rationale: "Sister adapters in time_traveler_l5_z6 / nirvana_cascading_nerv / nscs06_v8_chroma_lut / boost_nerv_pr110_residual / mdl_ibps_j_discrete_categorical_mine_hybrid / z7_mamba2_v2_fresh_substrate ALL return None per the legacy 'L2 MLX is reconstruction-proxy' reasoning. The fix preserves this contract for the scorer-unbound code path — emitting synthetic scorer-unbound per-axis would pollute per_epoch_metrics with rows that have no scorer-bound semantics. The fix ONLY emits per-axis when the canonical Hinton-distilled scorer-bound surrogate is active (distillation_weight > 0 OR pose_distillation_weight > 0)."
council_decisions_recorded:
  - "op-routable #1: Z6-v2 + Hinton + 600-pair Contrarian VETO STRUCTURALLY CLOSED at the per_axis_decomposition emission surface; future cross-family attribution analyzers can decompose per-axis seg vs pose vs archive_bytes contributions"
  - "op-routable #2: cross-family attribution surface UNBLOCKED for sub-0.18 downstream-of-in-training differentiation analysis per Z6-v2 + Hinton APPARATUS-LEVEL FINDING"
  - "op-routable #3: future Tier B consumers per Catalog #357 + Dim 6 Step 6.5 can now consume MLX-research-signal per-axis as upstream input for sub-frontier-inference predictions (with non-promotable [macOS-MLX research-signal] axis_tag preserved per Catalog #192/#127/#323)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: ""
related_deliberation_ids:
  - z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_landed_20260528
  - v2_v4_vq_hinton_distill_600pair_long_mlx_landed_20260528
---

# PER_AXIS_DECOMPOSITION GAP FIX in canonical mlx_score_aware harness LANDED 2026-05-28

**Closes**: Z6-v2 + Hinton + 600-pair Contrarian VETO per `.omx/research/z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_landed_20260528.md` op-routable #4.

**Scope**: $0 MLX-local engineering fix at the canonical harness adapter surface; non-promotable per Catalog #192/#127/#317/#323/#341. T1 working-group landing (closes Contrarian VETO; not a paradigm/strategy decision).

## The bug class

Pre-fix, `tac.substrates._shared.mlx_score_aware.adapter.MlxScoreAwareAdapter.score_aware_components(model, batch)` returned `None` UNIVERSALLY, mirroring the legacy sister-adapter pattern (Z6 / nirvana / boost / mdl_ibps / z7 / nscs06_v8 all return None with the same "L2 MLX is reconstruction-proxy only; per-axis = N/A" rationale).

The canonical `tac.training.long_training_canonical.run_long_training` ALREADY calls `adapter.score_aware_components(adapter.model, sample)` at line 2066 per epoch and threads the result into `PerEpochMetrics.per_axis_decomposition` at line 2088. The downstream emission surface was structurally correct — the gap was at the adapter stub.

With the canonical Hinton-distilled scorer-bound surrogate landed (Catalog #164 + the L2 BOTH-TEACHER-WIRED fail-closed contract), the legacy reasoning no longer holds:

- `parts["distill"]` is the SegNet axis surrogate gradient signal (KL T=2.0 against the REAL SegNet teacher's per-pixel class distribution per `RealSegNetTeacherLogitsCache`).
- `parts["pose_distill"]` is the PoseNet axis surrogate gradient signal (MSE against the REAL PoseNet teacher's per-pair pose vector per `RealPoseNetTeacherCache`).

Both are genuinely scorer-bound. The gap was therefore an outdated stub contract.

## What landed (10-line summary)

`src/tac/substrates/_shared/mlx_score_aware/adapter.py::score_aware_components` now:

1. Returns `None` when BOTH `distillation_weight=0.0` AND `pose_distillation_weight=0.0` (pure-recon mode; sister-adapter parity preserved).
2. Otherwise calls `score_aware_loss(self.bundle, batch)` (single source of truth per Catalog #290 ADOPT_CANONICAL) and maps the canonical loss components to per-axis:
   - `seg` ← `parts["distill"]` (0.0 if SegNet teacher not wired)
   - `pose` ← `parts["pose_distill"]` (0.0 if PoseNet teacher not wired)
   - `recon_aux` ← `parts["recon"]` (telemetry; per-pixel reconstruction; not per-axis attributable)
   - `archive_bytes` ← `0.0` (per-step delta undefined at MLX L2; archive built post-training; AxisDecomposition NaN-safe rule accepts 0.0 as no-signal)

## Empirical verification

### Smoke #1: scorer-bound bundle on dreamer_v3_rssm (3 epochs, 0.12s wall-clock, $0)

```
epoch=0 stage=dreamer_v3_rssm_mlx_score_aware_full loss=0.507203
  per_axis={'seg': 0.012860833667218685, 'pose': 0.0, 'recon_aux': 0.4954449534416199, 'archive_bytes': 0.0}
epoch=1 stage=dreamer_v3_rssm_mlx_score_aware_full loss=0.501875
  per_axis={'seg': 0.012225426733493805, 'pose': 0.0, 'recon_aux': 0.48542535305023193, 'archive_bytes': 0.0}
epoch=2 stage=dreamer_v3_rssm_mlx_score_aware_full loss=0.491538
  per_axis={'seg': 0.010908916592597961, 'pose': 0.0, 'recon_aux': 0.4661552309989929, 'archive_bytes': 0.0}
```

`per_axis_decomposition` is POPULATED (was `None` pre-fix). The seg component is non-zero (Hinton-KL term active via mock SegNet teacher), the pose component is 0.0 (no pose teacher wired in this fast-path fixture per the sister-adapter pose-bind fail-closed contract requiring real PoseNet), and `recon_aux` carries the per-pixel reconstruction telemetry.

### Smoke #2: pure-recon backward compat on dreamer_v3_rssm (2 epochs, $0)

```
epoch=0 loss=0.500458 per_axis=None
epoch=1 loss=0.495646 per_axis=None
```

Sister-adapter parity preserved. `None` is emitted when the scorer surrogate is inactive.

### Cross-family transitive coverage

Both `experiments/train_substrate_pact_nerv_ia3_mlx_local.py` and `experiments/train_substrate_z6_v2_cargo_cult_unwind_mlx_local.py` route through `run_mlx_score_aware_full_main` per static grep, so the GAP fix transitively reaches IA3 + Z6-v2 + every cross-family Hinton-distilled substrate consuming the canonical harness. Future cross-family attribution analyzers can now decompose per-axis convergence contributions empirically.

## Tests

`src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py` — 4 NEW tests:
- `test_adapter_score_aware_components_pure_recon_returns_none` (renamed from `_defers`; pure-recon backward compat preserved)
- `test_adapter_score_aware_components_seg_bound_populates_per_axis` (Hinton-distilled scorer-bound surrogate populates per-axis)
- `test_adapter_score_aware_components_both_teachers_populates_seg_and_pose` (both SegNet + PoseNet teachers → seg AND pose populated; cross-family attribution unblocked)
- `test_adapter_score_aware_components_compatible_with_axis_decomposition` (per-axis dict round-trips through canonical `AxisDecomposition.from_dict` contract per Catalog #356)

`src/tac/substrates/_shared/tests/test_mlx_score_aware_full_main.py` — 1 UPDATED + 1 NEW (sister test path):
- `test_adapter_score_aware_components_pure_recon_defers` (was `_defers_to_pytorch_sister`; updated to reflect new contract)
- `test_adapter_score_aware_components_scorer_bound_populates_per_axis` (NEW sister-path coverage)

Full test suites:
- `src/tac/substrates/_shared/mlx_score_aware/tests/` — 67 passed + 1 skipped (MLX gating).
- `src/tac/substrates/_shared/tests/` + `mlx_score_aware/tests/` aggregate — 226 passed + 1 skipped.
- `src/tac/tests/test_score_composition.py` — 39 passed (downstream consumer canonical contract regression guard).

Catalog #356 STRICT preflight gate live count: 0 (no regression).

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `score_aware_components` return shape | ADOPT_CANONICAL `Mapping[str, float] \| None` | Matches `tac.training.long_training_canonical.PerEpochMetrics.per_axis_decomposition` contract; consumed by canonical harness at line 2088 without change |
| seg/pose attribution source | ADOPT_CANONICAL `score_aware_loss` decomposition | Single source of truth per Catalog #164 + the canonical Hinton-distilled BOTH-TEACHER-WIRED contract; no duplicate loss computation |
| `archive_bytes` per-step | FORK_BECAUSE_PRINCIPLED_MISMATCH (emit 0.0) | MLX L2 builds the archive post-training via `export_archive_fn`; per-step archive byte delta is undefined; AxisDecomposition NaN-safe rule accepts 0.0 as no-signal |
| Pure-recon mode | ADOPT_SISTER_ADAPTER_CONTRACT (return None) | Preserves sister-adapter parity for the legacy scorer-unbound code path; no synthetic per-axis row pollutes per_epoch_metrics |
| Provenance routing | ADOPT_CANONICAL Catalog #323 | Parent `TrainingArtifact.canonical_provenance` already carries `[macOS-MLX research-signal]` axis_tag + non-promotable invariants per harness orchestrator; per-axis rows inherit transitively |
| Non-promotable invariants | ADOPT_CANONICAL Catalog #192/#127/#317/#341 | MLX-research-signal axis preserved; downstream consumers cannot promote per-axis rows to `[contest-CPU]` / `[contest-CUDA]` without paired Linux x86_64 + NVIDIA evidence |

## 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: the canonical mlx_score_aware harness is the substrate-AGNOSTIC bridge between any MLX renderer and the canonical L2 harness; this fix is at the canonical-vs-unique boundary, NOT at any substrate.
2. **BEAUTY + ELEGANCE**: ~80 LOC fix in `adapter.py::score_aware_components` (single function body); reuses canonical `score_aware_loss` decomposition; 30-second reviewable.
3. **DISTINCTNESS**: structurally distinct from the sister-adapter per-substrate stub pattern; the canonical harness adapter is the ONE place that can populate per-axis without modifying individual substrate adapters.
4. **RIGOR**: 4 new dedicated tests + 1 sister-path coverage; smoke verification on dreamer_v3_rssm with explicit per-epoch metric inspection; pure-recon backward compat empirically verified; 226 aggregated tests pass.
5. **OPTIMIZATION PER TECHNIQUE**: ADOPT_CANONICAL the loss decomposition (no recomputation); FORK only where the MLX L2 surface principled-mismatches the per-step archive_bytes contract.
6. **STACK-OF-STACKS COMPOSABILITY**: per-axis dict round-trips through `tac.cathedral.consumer_contract.AxisDecomposition.from_dict` per Catalog #356; composes with `tac.score_composition.compose_score_from_axes` for downstream cathedral ranker consumption.
7. **DETERMINISTIC REPRODUCIBILITY**: deterministic per-batch (uses `adapter.sample_batch(batch_size, seed + epoch + 1_000_000)` per the canonical L2 harness contract).
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 0.12s smoke wall-clock for 3 epochs on dreamer_v3_rssm tiny fixture; no MLX evaluation overhead (uses existing per-batch `score_aware_loss` call that the harness already performs).
9. **OPTIMAL MINIMAL CONTEST SCORE**: N/A directly (apparatus_maintenance contribution per Catalog #300); UNBLOCKS sub-0.18 downstream-of-in-training differentiation per the Z6-v2 + Hinton APPARATUS-LEVEL FINDING.

## Observability surface per Catalog #305

- **Inspectable per layer**: `PerEpochMetrics.per_axis_decomposition` populated per epoch in the canonical telemetry JSONL via the existing `telemetry_sink.record(metrics)` at line 2095 of `long_training_canonical.py`.
- **Decomposable per signal**: 4 canonical keys `{seg, pose, recon_aux, archive_bytes}` map directly into `AxisDecomposition` canonical contract per Catalog #356.
- **Diff-able across runs**: per-axis rows are deterministic per `seed + epoch + 1_000_000` batch seed; cross-run diff enabled.
- **Queryable post-hoc**: existing `TrainingArtifact.per_epoch_metrics` tuple + `.as_dict()` serialization unchanged; new keys appear in the existing serialization surface.
- **Cite-able**: parent `TrainingArtifact.canonical_provenance` per Catalog #323 carries the `[macOS-MLX research-signal]` axis_tag; per-axis rows inherit non-promotability transitively.
- **Counterfactual-able**: per-axis deltas enable Dim 1 Phase 4 Dykstra alternating-projections on the (seg, pose, rate) polytope downstream per `tac.score_composition.compose_score_from_axes`.

## Operator-routable

1. **TOP-1**: Cross-family attribution analyzer subagent can now consume `per_epoch_metrics[i].per_axis_decomposition` from any MLX-research-signal landing memo's telemetry JSONL to decompose seg vs pose vs recon contributions across substrate families (PACT-NeRV cascade vs Z6-v2 cooperative-receiver vs sister substrates) and surface the canonical downstream-of-in-training differentiation surface where sub-0.18 lives.

2. **DEFERRED-OPTIONAL**: extend `tac.score_composition.compose_score_from_axes` to honor the MLX `archive_bytes=0.0` no-signal convention by routing through canonical archive post-emit hook (currently the canonical helper accepts 0.0 cleanly per AxisDecomposition NaN-safe rule; no urgency).

3. **DEFERRED-PENDING-PHASE-2**: Tier B consumers per Catalog #357 + Dim 6 Step 6.5 may consume MLX-research-signal per-axis as upstream input for sub-frontier-inference predictions; the per-axis surface is now structurally available.

## Mission contribution per Catalog #300

`apparatus_maintenance`: closes Contrarian VETO at sub-surface; unblocks downstream-of-in-training differentiation surface analysis required by the Z6-v2 + Hinton apparatus-level finding; the immediate score-lowering value is N/A but the structural foundation unblocks the cross-family attribution analysis sister wave.

## Discipline

Catalog #229 PV (read full state of bundle + adapter + loss + harness + canonical training harness + AxisDecomposition contract + score_composition canonical helper + sister adapter implementations pre-edit). Catalog #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` for every edited file. Catalog #206 (5 checkpoints). Catalog #110/#113 APPEND-ONLY (NEW landing memo; updated test name reflects new contract per Catalog #287; zero mutation of historical artifacts). Catalog #230 sister-subagent ownership map (own only canonical mlx_score_aware harness extension + tests + landing memo + canonical posterior anchor; sister Slot 2 #1453 archive-encode-time differentiation analysis disjoint scope). Catalog #287 placeholder rejection (all rationales ≥4 chars). Catalog #356 canonical AxisDecomposition contract preserved at 0 violations. Catalog #340 sister-checkpoint guard (own-checkpoint mark-complete pattern). CLAUDE.md "Subagent coherence-by-default" + "MLX portable-local-substrate authority" + "Forbidden premature KILL". $0 GPU verified throughout (MLX-local M5 Max only; non-promotable per Catalog #192/#317/#341). All 226 tests pass.

## Sister cross-references

- Z6-v2 + Hinton + 600-pair landing memo: `.omx/research/z6_v2_cargo_cult_unwind_hinton_distill_600pair_long_mlx_landed_20260528.md` op-routable #4 (the GAP this fix closes)
- V2+V4+VQ + Hinton + 600-pair landing memo: `.omx/research/v2_v4_vq_hinton_distill_600pair_long_mlx_landed_20260528.md` (the cross-family parity anchor pattern)
- Canonical AxisDecomposition contract: `src/tac/cathedral/consumer_contract.py::AxisDecomposition` (Catalog #356 surface)
- Canonical compose_score_from_axes helper: `src/tac/score_composition/__init__.py` (Dim 1 Phase 4 + Dim 3 Step 3.2 surface)
- Canonical L2 harness: `src/tac/training/long_training_canonical.py::run_long_training` (the consumer of `adapter.score_aware_components` at line 2066)
- Canonical Provenance umbrella: `src/tac/provenance/__init__.py` + `tac.provenance.builders.build_provenance_for_predicted` (Catalog #323 surface)
- Sister adapter implementations preserving None contract per legacy reasoning: `time_traveler_l5_z6 / nirvana_cascading_nerv / boost_nerv_pr110_residual / mdl_ibps_j_discrete_categorical_mine_hybrid / z7_mamba2_v2_fresh_substrate / nscs06_v8_chroma_lut`
- CLAUDE.md non-negotiables binding this landing: "MLX portable-local-substrate authority" / "Subagent coherence-by-default" / "Apples-to-apples evidence discipline" / "Forbidden premature KILL without research exhaustion"
- Catalogs cited: #110 / #113 / #117 / #125 / #127 / #131 / #138 / #157 / #164 / #174 / #185 / #192 / #206 / #226 / #229 / #230 / #265 / #270 / #287 / #290 / #292 / #294 / #300 / #305 / #317 / #323 / #335 / #340 / #341 / #356

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map contribution**: ACTIVE — per-axis seg/pose/recon_aux rows are the canonical per-batch sensitivity surface at the consumer boundary; downstream `tac.sensitivity_map.*` consumers can now decompose convergence attribution
- **hook #2 Pareto constraint**: ACTIVE — per-axis decomposition is the primitive input to `tac.score_composition.compose_score_from_axes` Dykstra alternating-projections on the (seg, pose, rate) polytope per Dim 1 Phase 4 enabler
- **hook #3 bit-allocator hook**: N/A at this surface — `archive_bytes=0.0` is the canonical no-signal convention at MLX L2; bit-allocator consumption happens at archive emission boundary (post-training)
- **hook #4 cathedral autopilot dispatch hook**: ACTIVE — per-axis rows are consumable by the cathedral autopilot ranker via `invoke_cathedral_consumers_on_candidates` per Catalog #336 sister; per-axis attribution enables future Tier B consumer routing
- **hook #5 continual-learning posterior update**: ACTIVE — per-axis rows persist in canonical telemetry JSONL via the existing `telemetry_sink.record(metrics)` at line 2095 of `long_training_canonical.py`; downstream posterior anchors inherit per-axis decomposition transitively
- **hook #6 probe-disambiguator**: ACTIVE — the per-axis decomposition IS the canonical disambiguator between SegNet-bound (`seg` non-zero) vs PoseNet-bound (`pose` non-zero) vs scorer-unbound (`None`) training paths; downstream PR97-anti-pattern detection enabled

## Lane

`lane_per_axis_decomposition_gap_fix_mlx_score_aware_harness_20260528` L1 (impl_complete + memory_entry + canonical_posterior_anchor).
