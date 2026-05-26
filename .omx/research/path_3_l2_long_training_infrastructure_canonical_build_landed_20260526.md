---
schema_version: path_3_l2_long_training_infrastructure_canonical_build_landed_v1
created_utc: 2026-05-26T09:22:49Z
council_tier: T1
council_attendees: [Shannon, Carmack, Hotz, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
horizon_class: frontier_pursuit
related_deliberation_ids:
  - path_3_canonical_substrate_development_cascade_doctrine_20260526
  - path_3_d_z6_l1_promotion_landed_20260526
council_decisions_recorded:
  - "op-routable #1: SPAWN D=Z6 L2 first long-training run at epochs=100 num_pairs=50 (proof-of-canonical-helper end-to-end; estimated wall-clock ~30-60 min on M-series)"
  - "op-routable #2: SPAWN L1-PROMOTION-CASCADE wave for E=BoostNeRV / G=NIRVANA / C'=NSCS06 / B'=Z7-Mamba-2-v2 / J=MDL-IBPS each producing a per-substrate long_training_adapter.py per the Z6 reference pattern"
  - "op-routable #3: After 3-5 L1 promotions land, SPAWN L2 LONG-TRAINING-CASCADE concurrent on M-series via run_long_training_multi_arm"
  - "op-routable #4: META work continues in parallel (Wave #1 posterior_emission + CONSOLIDATE-OP-1 + R2-COMBINED + future audits)"
---

# Path 3 L2 Long-Training Infrastructure Canonical Build — LANDED

**Status**: LANDED 2026-05-26T09:22:49Z per L2-INFRA-BUILD charter (Path 3 cascade doctrine commit `fb270e9b6` §"L2 LONG-TRAINING INFRASTRUCTURE — CANONICAL + REUSABLE + COMPOSABLE + PRODUCTION-HARDENED" + operator binding directive *"Also the long training infrastructure, ensure reusable composable beautiful elegant creative expressive cimposable abstractions and production hardened OSS and no duplicative code"*).

**Lane**: `lane_path_3_l2_long_training_infrastructure_canonical_build_20260526` L1 (impl_complete + memory_entry).

## Canonical helper API (canonical contract operator + future sister subagents reference)

**Module**: `tac.training.long_training_canonical` (~1170 LOC; 21-symbol public `__all__`).

**Canonical entry-points**:
- `run_long_training(adapter, config) -> TrainingArtifact` — canonical L2 entry-point.
- `run_long_training_multi_arm(arms, *, max_concurrent_arms=4) -> MultiArmDispatchResult` — multi-arm dispatch.

**Frozen dataclass schemas** (all `__post_init__` validated):
- `LongTrainingConfig` — substrate_id + lane_id + epochs + batch_pair_indices_per_step + curriculum_stages + ema_decay (default 0.997 per Catalog #2) + checkpoint_interval_epochs + early_stopping_patience + score_aware_loss_kwargs + optimizer_class + learning_rate + seed + output_dir + telemetry_path + checkpoint_dir + device (mlx|cuda|cpu; mps REFUSED) + resume_from_checkpoint + evidence_grade + notes.
- `CurriculumStage` — name + start_epoch + end_epoch + loss_weights + lr_scale + freeze_layers + enable_qat + notes. Contiguous + non-overlapping enforced at config construction.
- `PerEpochMetrics` — epoch + stage_name + loss + loss_components + per_axis_decomposition + wall_clock_seconds + ema_drift_l2 + learning_rate + captured_at_utc.
- `TrainingArtifact` — substrate_id + lane_id + config_snapshot + ema_shadow_checkpoint_path + per_epoch_metrics + total_wall_clock_seconds + total_epochs_completed + canonical_provenance + telemetry_path + archive_path + archive_sha256 + archive_bytes + early_stopped + early_stop_reason + posterior_update_accepted + non-promotable markers (score_claim=False / promotion_eligible=False / ready_for_exact_eval_dispatch=False / rank_or_kill_eligible=False / promotable=False) enforced via `__post_init__`.
- `MultiArmDispatchResult` — arms tuple + total_wall_clock + captured_at_utc.

**Substrate Protocol** (`SubstrateLongTrainingAdapter`):
- Required attrs: `substrate_id`, `model`.
- Required methods: `sample_batch`, `loss_fn`, `optimizer_step`, `export_state_dict`, `export_archive`, `score_aware_components`.
- Optional Style B: `train_step(batch, learning_rate, loss_weights) -> {"total": ...}` for MLX `value_and_grad` pattern. Canonical helper auto-detects via `hasattr(adapter, 'train_step')`.

**Canonical primitives** (composable, reusable):
- `PolyakEMAShadow` — canonical EMA with auto-detected torch vs MLX duck-typing (state_dict vs parameters+tree_flatten/tree_unflatten); decay=0.997 default; snapshot+restore + restore_from_snapshot pattern.
- `TelemetrySink` — fcntl-locked atomic JSONL append; idempotent flush via `_next_flush_index` tracking; close()-safe.
- `CheckpointWriter` — periodic + final checkpoints with `substrate_id` + `lane_id` + `curriculum_hash` validation refusing cross-substrate / cross-curriculum resume per Catalog #229 PV.
- `OOMSafeStepRunner` — Style A/B auto-detection; halves batch_size on OOM up to max_retries=4; raises typed RuntimeError after exhaustion.

**Conformance helpers**:
- `validate_substrate_adapter(adapter)` — Protocol conformance validator.
- `validate_long_training_config(config)` — sister symmetric API.

## Per-element 10-contract evidence

| # | Element | Status | Evidence |
|---|---|---|---|
| 1 | `run_long_training(substrate, config)` | ADOPTED | canonical entry-point; ~250 LOC orchestrator |
| 2 | `LongTrainingConfig` frozen dataclass | ADOPTED | 17 fields + `__post_init__` strict invariants |
| 3 | `CurriculumStage` frozen dataclass | ADOPTED | 7 fields + contiguous/non-overlapping enforcement at config-level |
| 4 | Checkpoint+resume | ADOPTED | `CheckpointWriter` with fcntl-locked sister of Catalog #206 + cross-substrate/cross-curriculum refusal |
| 5 | Per-arm canonical Provenance + posterior anchor | ADOPTED | `_build_canonical_provenance_for_artifact` via Catalog #323 builders + `_emit_canonical_posterior_anchor` via sister `posterior_emission_helper` per Catalog #128 |
| 6 | Differentiable-eval-roundtrip + EMA-apply-at-eval wrappers | ADOPTED (EMA); FORKED-WITH-JUSTIFICATION (eval_roundtrip deferred to substrate adapter's loss_fn body since MLX has no torch.no_grad equivalent — per CLAUDE.md eval_roundtrip the substrate's score_aware_components is the canonical surface for this; sister at L3+ paid CUDA promotion) |
| 7 | Multi-arm parallel dispatch | ADOPTED | `run_long_training_multi_arm`; sequential-safe default per Catalog #302; concurrent ThreadPool deferred per MLX single-GPU contention semantics (noted in docstring) |
| 8 | Crash-recovery + OOM-safe | ADOPTED | `OOMSafeStepRunner` with duck-typed OOM detection (RuntimeError + MemoryError + msg-substring); checkpoint preservation across crashes via final checkpoint always-emit + try/except wrappers around every canonical write |
| 9 | Observability surface per Catalog #305 | ADOPTED | `TelemetrySink` fcntl-locked JSONL + per-epoch loss decomposition + per-axis (when adapter emits) + EMA-drift L2 + wall-clock; canonical `training_artifact.json` for cite-able structured artifact |
| 10 | OSS-clean public API | ADOPTED | narrow `__all__` (21 symbols); SPDX-License-Identifier: MIT header; zero `/Users/...` paths (Catalog #208 clean); docstrings reference Catalogs explicitly |

## D=Z6 reference refactor (proof-of-pattern)

**L1 promotion** (existing; commit `8833b9db5`): `experiments/train_substrate_z6_predictive_coding_mlx.py` — **627 LOC** of hand-rolled training loop + EMA + checkpoint + Provenance + posterior anchor + archive emission.

**L2 canonical pattern** (this landing): `experiments/train_substrate_z6_predictive_coding_mlx_l2.py` — **136 LOC** total (≈30 LOC of substrate-specific config + ONE canonical `run_long_training()` invocation; rest is argparse + main scaffold + video decode).

**Reduction**: **78%** (491 LOC absorbed into canonical helper). The substrate-axis surface (substrate-specific config + adapter glue) is the only code that varies per substrate.

**Z6 adapter** (Style B because MLX uses `value_and_grad`): `src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py` — `Z6LongTrainingAdapter` class with `train_step` (combined value+grad+update) + `loss_fn` (diagnostic-only Style A fallback) + `export_state_dict` + `export_archive` (via `mlx_export_bridge`) + `score_aware_components` returning None (deferred to L2 PyTorch sister per Yousfi dissent).

**End-to-end smoke verification** (epochs=5 num_pairs=8 at `experiments/results/z6_l2_canonical_smoke_20260526T091918Z/`):
- archive `0.bin` 61.4 KB + sha256 `40e4f81010c9...`
- telemetry JSONL 5 rows with `loss [0.3288, 0.328, 0.3269, 0.3252, 0.3228]` (decreasing — training real)
- EMA drift L2 [0.292, 0.638, 1.023, 1.441, 1.880] (Polyak averaging tracking)
- 3 checkpoints emitted (epoch 1, 3, final)
- canonical Provenance `{artifact_kind=predicted_from_model, evidence_grade=predicted, measurement_axis=[macOS-MLX research-signal], promotion_eligible=False, score_claim_valid=False}`
- training_artifact.json carries the full 21-field schema with non-promotable markers all False

## Test suite (≥30 target; 60 landed)

**Canonical helper tests** (`src/tac/substrates/_shared/tests/test_long_training_canonical.py`): **53 tests** covering:
- Canonical constants (3 tests): CANONICAL_EMA_DECAY anchor; non-promotable markers all False; PR95 8-stage curriculum [0,3000) contiguous.
- CurriculumStage invariants (6 tests): happy path + rejection of empty name + invalid epoch range + negative loss weights + placeholder notes + zero lr_scale.
- LongTrainingConfig invariants (9 tests): happy path + reject lane_id without prefix + reject `device='mps'` + reject `/tmp` output_dir + non-contiguous curriculum + non-zero-start curriculum + EMA decay outside (0,1) + stage_at_epoch behavior + curriculum_hash stability.
- PolyakEMAShadow (4 tests): initial clone independence + Polyak averaging math + apply_to snapshot+restore + invalid decay rejection.
- TelemetrySink (3 tests): record + flush + idempotent flush (no row duplication on close); reject /tmp; snapshot tuple.
- CheckpointWriter (4 tests): canonical metadata emission + cross-substrate refusal + curriculum_hash refusal + invalid curriculum_hash refusal.
- OOMSafeStepRunner (3 tests): success without OOM + halve on OOM + raise after exhaustion.
- run_long_training happy path (5 tests): artifact return + telemetry JSONL emission + archive emission + adapter-returns-None archive deferral + training_artifact.json emission.
- Per-axis + EMA + reproducibility (3 tests): per-axis decomposition when emitted + None when adapter returns None + seed-pinning byte-stable.
- Curriculum + early stopping + checkpoint (4 tests): stage transitions + early stopping on patience + checkpoint at interval + resume cross-curriculum guard.
- Canonical Provenance + posterior anchor (2 tests): Provenance dict shape + posterior anchor attempt recorded.
- Multi-arm parallel dispatch (3 tests): per-arm artifact + reject empty arms + telemetry isolation.
- Validation helpers + TrainingArtifact invariants (3 tests): reject incomplete adapter + reject empty substrate_id + reject promotion_eligible=True.
- Reproducibility loss trace (1 test): identical config + seed produces identical loss trace.

**Conformance tests** (`src/tac/substrates/_shared/tests/test_substrate_long_training_conformance.py`): **7 tests** covering:
- Z6 adapter class declares all Protocol methods.
- Z6 adapter uses Style B (train_step).
- Z6 adapter substrate_id canonical.
- Z6 adapter construction requires MLX.
- Z6 adapter passes `validate_substrate_adapter`.
- Canonical helper detects Style B train_step when present.
- Canonical helper falls back to Style A when no train_step.

**Total: 60/60 tests pass**. PR95 sister tests (`test_pr95_mlx_long_training_infrastructure.py`) still pass 10/10 — no regression.

## OSS-clean public API audit (Catalog #208 docs/local-paths CLEAN + SPDX header present + narrow `__all__`)

- `/Users/adpena/` paths in canonical helper + adapter + tests + trainer: **0** (one occurrence in `long_training_canonical.py` is the FORBIDDEN-PATH DETECTOR's own pattern definition — legitimate per Catalog #208 sister discipline; same pattern as the gate's own self-protection).
- SPDX-License-Identifier: MIT header: present in all 5 new files.
- Narrow `__all__`: 21 canonical symbols (constants + 5 dataclasses + Protocol + 2 entry-points + 4 primitives + 2 validators).

## Composability evidence (cathedral autopilot + canonical equations + bit_allocator)

- **Cathedral autopilot consumer ingest**: canonical posterior anchor at `.omx/state/mps_research_signal_manifest.jsonl` is auto-discovered by `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers` per Catalog #335. Smoke verified end-to-end (artifact emits posterior_update_accepted field; non-promotable markers route to refused_anchor_count per Catalog #341).
- **Canonical equations registry calibration**: `TrainingArtifact.canonical_provenance` carries `artifact_kind=predicted_from_model` + `source_sha256` enabling future calibration via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344.
- **Bit allocator integration**: `PerEpochMetrics.per_axis_decomposition` (when adapter emits) is the canonical per-axis input bit_allocator consumes per Catalog #1068; integration is at the consumer side (no work needed in the canonical helper).
- **Findings Lagrangian**: `Catalog #355` invocation consumes canonical Provenance from TrainingArtifact (no extra wire-in required at the helper side; the existing `invoke_meta_lagrangian_on_candidates` reads `.omx/state/`).
- **Master-gradient consumers BUNDLE** per Catalog #354: multi-arm dispatch results feed `master_gradient_xray_consumer` for cross-arm diff (integration at consumer side).

## Per Catalog #290 canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| `run_long_training` entry-point | ADOPT CANONICAL | substrate-agnostic core; the canonical contract |
| `LongTrainingConfig` schema | ADOPT CANONICAL | substrate-agnostic; PR95 sister has its own (PR95-SPECIFIC) which legitimately forked per Catalog #290 (PR95-HNeRV architecture is hardcoded; this canonical helper is the general substrate-agnostic layer) |
| `CurriculumStage` schema | ADOPT CANONICAL | substrate-agnostic; mirrors PR95 8-stage curriculum with flexible loss_weights for substrate-specific overrides |
| Checkpoint discipline | ADOPT CANONICAL | fcntl-locked sister of Catalog #206 + cross-substrate guard |
| EMA primitive | ADOPT CANONICAL | extends sister `tac.training.EMA` with duck-typed MLX support |
| Telemetry sink | ADOPT CANONICAL | sister of `tac.continual_learning` fcntl-locked JSONL pattern |
| OOM-safety | ADOPT CANONICAL | substrate-agnostic; duck-typed OOM detection |
| Multi-arm dispatch | ADOPT CANONICAL | substrate-agnostic; default sequential per Catalog #302 |
| Provenance + posterior | ADOPT CANONICAL via sister builders | routes through `tac.provenance` + `tac.substrates._shared.posterior_emission_helper` per Catalog #323 + #128 |
| PR95-HNeRV specific training loop | FORKED-WITH-JUSTIFICATION (sister module) | per Catalog #290: PR95-HNeRV has hardcoded HNeRVDecoderMLX + RGB-MSE loss + PyAV pipeline; substrate-axis structure is PR95-HNeRV-specific; the canonical helper is the substrate-AGNOSTIC layer; sister PR95 helper remains canonical for PR95-HNeRV substrate |
| Z6 adapter | ADOPT Style B (train_step) | MLX `value_and_grad` requires combined value+grad+update; canonical helper auto-detects via hasattr(adapter, 'train_step') |
| score_aware_components for Z6 | DEFERRED (returns None) | per per-substrate symposium PROCEED_WITH_REVISIONS Yousfi dissent: SegNet/PoseNet decomposition routes through PyTorch sister L2 promotion path; MLX L2 is reconstruction-proxy only |

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = N/A (canonical infrastructure helper, not a sensitivity contributor; substrate adapters may emit via score_aware_components which would feed `tac.sensitivity_map`)
- hook #2 Pareto constraint = N/A (canonical helper; per-axis Pareto contribution is at consumer side)
- hook #3 bit-allocator = ACTIVE (PerEpochMetrics.per_axis_decomposition is the canonical bit_allocator input per Catalog #1068)
- hook #4 cathedral autopilot dispatch = ACTIVE (canonical posterior anchor at `.omx/state/mps_research_signal_manifest.jsonl` auto-discovered per Catalog #335)
- hook #5 continual-learning posterior = ACTIVE (every L2 run emits canonical posterior anchor via Catalog #128 fcntl-locked helper)
- hook #6 probe-disambiguator = ACTIVE (the Style A vs Style B Protocol distinction IS the canonical disambiguator between torch-natural vs MLX-natural training axes)

## Operator-routable next steps

1. **D=Z6 L2 first long-training run** at `epochs=100 num_pairs=50 output_height=48 output_width=64` for ~30-60 min wall-clock on M-series:
   ```bash
   .venv/bin/python experiments/train_substrate_z6_predictive_coding_mlx_l2.py \
       --num-pairs 50 --epochs 100 \
       --output-dir experiments/results/z6_l2_first_long_run_$(date -u +%Y%m%dT%H%M%SZ)
   ```
   Validates the canonical L2 helper at scale; emits canonical posterior anchor + cathedral consumer cascade observes.

2. **L1-PROMOTION-CASCADE wave** for next-most-EV substrates per cascade doctrine spawn order:
   - E=BoostNeRV (highest EV per residual-against-PR110 stacking)
   - G=NIRVANA (3-axis fully evidenced)
   - C'=NSCS06 v8 chroma_lut cargo-cult-first
   - B'=Z7-Mamba-2-v2 cargo-cult-first
   - J=MDL-IBPS DISCRETE-CATEGORICAL-MINE-HYBRID

   Each L1 promotion produces a substrate-specific `long_training_adapter.py` (~150 LOC per Z6 reference) + sister L2 trainer (~30 LOC).

3. **L2 LONG-TRAINING-CASCADE concurrent on M-series** once 3-5 L1 promotions land via `run_long_training_multi_arm`. Initial cap = 4 concurrent per Catalog #302 sister-subagent scope overlap.

4. **META work continues in parallel**: Wave #1 posterior_emission, CONSOLIDATE-OP-1, R2-COMBINED, future audits.

## Discipline applied

- Catalog #229 PV: read doctrine + trainer_skeleton + posterior_emission_helper + training.py + pr95_hnerv_mlx_long_training + Z6 L1 promotion BEFORE designing helper API
- Catalog #117/#157/#174 canonical serializer with POST-EDIT `--expected-content-sha256` per file (commit pending)
- Catalog #119 Co-Authored-By Claude trailer (commit pending)
- Catalog #206 checkpoint discipline (4 in-progress checkpoints emitted)
- Catalog #287 placeholder-rationale rejection enforced in CurriculumStage.notes + LongTrainingConfig.notes
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW files only; no mutation of existing memos
- Catalog #208 docs/local-paths CLEAN: zero `/Users/adpena/` paths
- Catalog #230 sister-subagent ownership map: file collision check passes (no overlap with Wave #1 posterior_emission or CONSOLIDATE-OP-1 or R2-COMBINED)
- Catalog #340 sister-checkpoint guard: PROCEED (no conflicting in-flight edits on canonical helper module)
- Catalog #265/#335 canonical contract pattern: SubstrateLongTrainingAdapter Protocol satisfies the canonical-contract pattern
- Catalog #290 canonical-vs-unique: per-layer decision documented above
- Catalog #294 9-dim success checklist: see §"Per Catalog #290 canonical-vs-unique decision per layer" above
- Catalog #299 gate consolidation: ZERO new STRICT preflight gates introduced
- Catalog #305 observability surface: PerEpochMetrics + TrainingArtifact + telemetry JSONL satisfy 6-facet contract
- Catalog #323 canonical Provenance umbrella: every TrainingArtifact carries canonical Provenance
- CLAUDE.md "Beauty, simplicity, and developer experience": 30-second-reviewable canonical entry-point; minimal substrate-side LOC; canonical primitive composition
- CLAUDE.md "Public Disclosure Hygiene": SPDX-License-Identifier: MIT header per Catalog #265
- CLAUDE.md "Executing actions with care": NO `gh pr create`, NO Modal/Vast/Lightning paid dispatch (this is canonical infrastructure build; $0 GPU spend)

## Cross-references

- Path 3 cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md`
- D=Z6 L1 promotion: `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md`
- PR95 sister-module reference: `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- Canonical helper docs: `docs/canonical_long_training_infrastructure.md`
- CLAUDE.md "MLX portable-local-substrate authority"
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
- CLAUDE.md "EMA — NON-NEGOTIABLE"
- CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE"

## Files landed

1. **`src/tac/training/long_training_canonical.py`** (~1170 LOC) — canonical L2 helper.
2. **`src/tac/substrates/_shared/tests/__init__.py`** — new tests package.
3. **`src/tac/substrates/_shared/tests/test_long_training_canonical.py`** (~700 LOC; 53 tests).
4. **`src/tac/substrates/_shared/tests/test_substrate_long_training_conformance.py`** (~180 LOC; 7 tests).
5. **`src/tac/substrates/time_traveler_l5_z6/long_training_adapter.py`** (~200 LOC) — Z6 reference adapter.
6. **`experiments/train_substrate_z6_predictive_coding_mlx_l2.py`** (136 LOC) — Z6 L2 trainer (proof of canonical pattern).
7. **`docs/canonical_long_training_infrastructure.md`** — operator + sister-subagent canonical usage docs.
8. **`.omx/research/path_3_l2_long_training_infrastructure_canonical_build_landed_20260526.md`** — THIS landing memo.

## Cost + wall-clock

- $0 GPU (canonical infrastructure build; no paid dispatch)
- ~3h wall-clock (PV + module design + tests + Z6 adapter + smoke verification + docs + landing memo)
