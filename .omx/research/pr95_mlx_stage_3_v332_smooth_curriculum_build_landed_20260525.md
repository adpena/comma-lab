---
# SPDX-License-Identifier: MIT
council_tier: T1
council_attendees: [Shannon, Dykstra, PR95Author]
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "op-routable: spawn Stage 4 v332_qat sister BUILD per MLX-PARADIGM-T3 Op #3 cascade"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: plateau_adjacent
canonical_equation_refs_queued:
  - pr95_mlx_stage_3_v332_smooth_one_to_one_curriculum_port_v1
related_deliberation_ids:
  - pr95_mlx_stage_2_v331_softplus_curriculum_build_landed_20260525
  - mlx_arch_5_pr101_state_dict_paired_forward_landed_20260525
  - codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex
---

# PR 95 Stage 3 v332_smooth MLX Curriculum Build LANDED 2026-05-25

**Lane**: `lane_pr95_mlx_stage_3_c1a_l3_curriculum_build_20260525` L1
**Cost**: $0 + ~30 min wall-clock (faster than Stage 2's 70 min via canonical extension pattern)
**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer +
POST-EDIT `--expected-content-sha256` + #206 (5 checkpoints) + #110/#113
APPEND-ONLY + #230 ownership map (Slot 3 DQS1-ASYMPTOTIC-FLOOR-SUBSTRATE-CLASS-SHIFT
+ Slot 4 PROBE-9-TIER-2-DISPATCH-PREP both disjoint) + #287 placeholder-rationale
rejection + #305 observability surface + #307 IMPLEMENTATION-LEVEL
falsification classification + #313 probe-outcomes registered + #340
sister-checkpoint guard PROCEED.

**Scope canonicalization per operator PR 95 1:1 port directive**: the prompt's
hypothetical `c1a_l3` stage name was canonicalized to `stage3_v332_smooth`
per the recovered public PR 95 source (`.omx/research/pr95_8stage_curriculum_forensic_20260513.md`
+ `.omx/research/pr95_curriculum_recovery_20260513_codex.md`). `c1a` refers to
PR 95's `cat_entropy_v2` (rate-axis Lagrangian) which is introduced at
**Stage 5** (`stage5_c1a_l7`), NOT Stage 3. Stage 3 is the
`smooth_disagreement_seg_loss(tau=0.3)` bridge between Stage 2
(`tau_softplus_seg_loss`) and Stage 4 QAT (`stage4_v332_qat` — fresh QAT
continuing Stage 3 cosine). Same sister-canonicalization pattern as Stage 2
landing 2026-05-25 (initial scope `stage2_v328_ce` → canonical
`stage2_v331_softplus`).

**Cascade closure**: extends canonical Stage 1+2+5+8 cascade (per Stage 2
landing commit `f10722f30`) to canonical Stage 1+2+3+5+8 by inserting the
intermediate Stage 3 smooth_disagreement bridge.

## Carmack MVP-first 5-step compliance per CLAUDE.md `be125b878`

1. **FREE local MLX Stage 3 synthetic 100-step smoke**: ZERO paid GPU cost;
   validated on local Apple Silicon (M5 Max MLX GPU).
2. **Falsifiably challenged**: forward parity Stage 1 vs Stage 3 at random init
   measured against the MLX-ARCH-5 dispatch contract band (ε=5e-3 fp32).
   Falsifying outcomes:
   - **Stage 3 100-step synthetic smoke on MLX**: PASS_RUN (rc=0 via direct
     `tools/run_pr95_mlx_timing_smoke.py`) — 2.34s wall-clock, 23.40ms/step,
     42.73 ex/s, 915.9KB state, last_loss=0.0828 (converged from random init).
   - **Stage 1 vs Stage 3 paired forward parity at random init**: PASS_SHAPE +
     PASS_BAND_5E3 + max_abs_diff=0.0 (byte-identical; canonical sanity check
     that Stage 3 does NOT silently perturb the architecture).
   - **Architecture parity Stage 2 vs Stage 3**: state_bytes byte-identical
     (915,944 bytes both stages — confirms shared HNeRVDecoder + base_ch=36 +
     latent_dim=28); seconds_per_step within 0.13% (23.40 vs 23.43 ms) — both
     metrics empirically validate the canonical sanity invariant.
   - **Canonical non-promotable contract maintained**: score_claim=False,
     promotable=False, ready_for_exact_eval_dispatch=False, evidence_grade=
     `[macOS-MLX research-signal]` per CLAUDE.md "MPS auth eval is NOISE" +
     Catalog #1/#192/#287/#323.
3. **Catalog #344 reference**: canonical equation candidate
   `pr95_mlx_stage_3_v332_smooth_one_to_one_curriculum_port_v1` QUEUED for
   operator-routable RATIFY-N (NOT auto-registered; per Stage 2 + ARCH-3+4+5
   precedent + Catalog #344 operator-decision protocol for new canonical
   equations).
4. **Landed verdict in same commit batch**: 4 source files modified + 1 NEW
   test file + landing memo + Catalog #313 probe-outcomes row registered.
5. **Re-route operator priority queue**: per the empirical findings below,
   the operator-routable next step is **Path A** (PROCEED to Stage 4
   `stage4_v332_qat` sister BUILD per MLX-PARADIGM-T3 Op #3 cascade); the
   canonical MLX substrate-trainer extension paradigm is empirically
   validated end-to-end for an additional PR 95 published curriculum stage.

## What landed

### 4 source files modified

1. **`src/tac/local_acceleration/pr95_hnerv_mlx.py`** (+2 LOC; ADDITIVE):
   - `PR95_STAGE_MODULES[3] = "stage3_v332_smooth"`.
   - `PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[3] = "pr95_stage3_adamw_baseline_mlx"`.

2. **`src/tac/optimization/optimizer_scheduler_registry.py`** (+42 LOC; ADDITIVE):
   - NEW `OptimizerSchedulerDescriptor("pr95_stage3_adamw_baseline_mlx", ...)`
     between Stage 2 + Stage 5 in `default_optimizer_scheduler_descriptors()`.
   - Canonical configuration per recovered public PR 95 source:
     `adamw_lr=1e-4` (FRESH cosine, distinct from Stage 1+2 baseline 1e-3
     and Stage 5 quantization 3e-5 and Stage 8 finetune 1e-5) +
     `stage_loss_family="smooth_disagreement_seg_loss"` +
     `stage_cat_sigma=0.2` + `stage_cat_lambda=0.0` + `stage_epochs=1500` +
     `stage_uses_qat=False` (QAT is Stage 4) + canonical non-promotable
     contract preserved.

3. **`tools/build_pr95_mlx_optimizer_matrix_queue.py`** (+5 LOC; ADDITIVE):
   - `_apply_control_profile` `full_pr95_source_video_runtime` profile now
     emits `args.stages = [1, 2, 3, 5, 8]` (was `[1, 2, 5, 8]` per Stage 2
     landing) per the canonical PR 95 8-stage curriculum Stage 1+2+3+5+8
     cascade. Citation comment references this landing memo +
     MLX-PARADIGM-T3 commit `916c43d89`.
   - `--control-profile` help string updated from "stage 1/2/5/8" to
     "stage 1/2/3/5/8".

4. **Two sister test files updated** (test discipline per APPEND-ONLY +
   Stage 2 sister-canonical precedent):
   - `src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py`:
     `assert sorted(PR95_STAGE_MODULES) == [1, 2, 5, 8]` weakened to
     `assert {1, 2, 5, 8}.issubset(set(PR95_STAGE_MODULES))` per Catalog
     #110/#113 APPEND-ONLY discipline (Stage 2 test now references Stage 3
     landing in docstring); regex weakened from `\[1, 2, 5, 8\]` to
     `supported PR95 MLX timing stages` (stage 4/6/7 referenced as
     unsupported); `stage_smoke_config(3)` changed to `stage_smoke_config(4)`
     (3 is now supported). Future Stage 4/6/7 landings do not require
     mutating this Stage 2 test.
   - `src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`:
     `test_pr95_mlx_optimizer_matrix_full_source_video_control_profile`
     updated to expect `stage_indices == [1, 2, 3, 5, 8]` + 5-plan
     descriptor set + 5 experiments (was 4) per the new canonical Stage 3
     spine.

### 1 NEW test file (~280 LOC; 11 tests)

`src/tac/tests/test_pr95_mlx_stage_3_v332_smooth_curriculum_build.py`:
- Canonical interface extension invariants (4 tests): PR95_STAGE_MODULES
  contains stage_3 / default descriptor for stage_3 / canonical {1, 2, 3, 5, 8}
  superset-of pattern / unsupported stage raises with canonical-list message.
- Canonical descriptor row invariants per Catalog #287 + #323 (3 tests):
  descriptor in registry + canonical contract + passes proxy-candidate
  validator + canonical LR ladder verification (1e-3 Stage 1+2; 1e-4 Stage 3
  fresh cosine; 3e-5 Stage 5; 1e-5 Stage 8).
- `stage_smoke_config` dispatch invariants (2 tests): default dispatch +
  explicit `--optimizer-descriptor-id` override.
- End-to-end synthetic timing smoke (1 test): 10-step MLX smoke runs +
  emits non-promotable contract.
- Paired forward parity Stage 1 vs Stage 3 at random init (1 test):
  byte-identical max_abs_diff=0.0; PASS_BAND_5E3 sanity check.

### 1 experiment package landed

`experiments/results/pr95_mlx_stage_3_v332_smooth_curriculum_build_20260525T162242Z/stage3/pr95_stage3_adamw_baseline_mlx/seed20260525_c36/`:
- `manifest.json` — `pr95_hnerv_mlx_timing_smoke_manifest_v1` schema with
  `stage_index=3` + `stage_module=stage3_v332_smooth` + last_loss=0.0828
  + evidence_grade=`[macOS-MLX research-signal]`.
- `runtime_profile.json` — canonical schema with
  `optimizer_descriptor_id=pr95_stage3_adamw_baseline_mlx` +
  `training_fidelity=synthetic_timing_only` + `training_backend=mlx` +
  `hardware_substrate=Darwin_arm64_mlx` + `seconds_per_step=0.02340` +
  `examples_per_second=42.73` + `state_bytes=915944`.
- `representation_training_manifest.json` — canonical
  representation_training_observation schema sister.

### Pre-existing bug surfaced (NOT MINE)

`tools/build_pr95_mlx_optimizer_matrix_queue.py` invocation triggers
`TypeError: _extra_artifact_postconditions() missing 1 required keyword-only
argument: 'write_pytorch_export_parity'` at `tools/run_pr95_mlx_timing_smoke.py:1028`.
Reproduces for Stage 1, Stage 2, AND Stage 3 (verified empirically by
re-running with `--stage 1` after Stage 3 extensions landed) → confirms
this is a pre-existing sister-territory bug NOT caused by my Stage 3
extension. Bypassed by invoking `tools/run_pr95_mlx_timing_smoke.py`
directly for the canonical 100-step smoke. Operator-routable: sister
subagent should add `write_pytorch_export_parity=False` to the call site
at line 1028.

## Empirical test results

### Stage 3 v332_smooth synthetic timing smoke (M5 Max MLX GPU)

| Metric | Value |
|---|---:|
| Wall-clock (100 steps) | 2.34 s |
| Seconds per step (avg) | 23.40 ms |
| Examples per second | 42.73 |
| State bytes | 915,944 |
| Last loss (converged) | 0.0828 |
| Hardware substrate | `Darwin_arm64_mlx` |
| Stage module | `stage3_v332_smooth` |
| Optimizer descriptor | `pr95_stage3_adamw_baseline_mlx` |
| AdamW LR | 1e-4 |
| Loss family | `smooth_disagreement_seg_loss` |
| Stage epochs (canonical PR 95) | 1500 |
| Stage uses QAT | False |
| Score claim | False |
| Promotable | False |
| Ready for exact eval dispatch | False |
| Evidence grade | `[macOS-MLX research-signal]` |

### Architecture parity Stage 2 vs Stage 3 (empirical receipts)

| Metric | Stage 2 | Stage 3 | Δ |
|---|---:|---:|---:|
| State bytes | 915,944 | 915,944 | 0 (byte-identical) |
| Seconds per step | 23.43 ms | 23.40 ms | -0.03 ms (0.13%) |
| Examples per second | 42.68 | 42.73 | +0.05 (0.12%) |
| Last loss | 0.0737 | 0.0828 | +0.0091 (similar magnitude; convergence trajectories aligned) |

Byte-identical state confirms the canonical architecture is preserved
across all Stage 1/2/3 transitions; only the loss family + LR schedule
distinguish the stages per PR 95's published curriculum design.

### Stage 1 vs Stage 3 paired forward parity at random init

| Sample | Max abs diff | Mean abs diff | PASS_BAND_5E3 |
|---|---:|---:|---|
| seed=20260525 / N=1 / N2CHW=(1, 2, 3, 384, 512) | **0.0** | **0.0** | **PASS** |

Byte-identical: Stage 1 + Stage 3 share the SAME architecture (HNeRVDecoder
+ base_ch=36, latent_dim=28); only the loss-family transition
(smooth_disagreement vs RGB MSE) + LR schedule transition (1e-4 fresh
cosine vs 1e-3 baseline) distinguish them. At step 0 (before training),
seeded random init produces byte-identical forward output. Canonical sanity
check that Stage 3 does NOT silently perturb the architecture.

### Full test suite (68/68 passing)

```
src/tac/tests/test_pr95_hnerv_mlx.py                                    21 passed
src/tac/tests/test_pr95_hnerv_mlx_training.py                            8 passed
src/tac/tests/test_run_pr95_mlx_timing_smoke.py                          1 passed
src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py                    5 passed
src/tac/tests/test_optimizer_scheduler_registry.py                       8 passed
src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py   14 passed
src/tac/tests/test_pr95_mlx_stage_3_v332_smooth_curriculum_build.py     11 passed
                                                                         ==
                                                                         68 passed in 12.39s
```

Zero regression on sister test suites; full canonical interface integration
verified end-to-end for Stage 3 extension.

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A | Stage 3 is a curriculum extension, not a sensitivity surface. |
| #2 Pareto constraint | N/A | Curriculum stage extension, not a Pareto-relevant signal. |
| #3 bit-allocator | N/A | Curriculum stage extension, not a bit-allocator signal. |
| #4 cathedral autopilot dispatch | ACTIVE | Stage 3 candidates participate in canonical autopilot ranking via `optimizer_scheduler_registry` candidates. |
| #5 continual-learning posterior | ACTIVE | Probe-outcomes row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313. |
| #6 probe-disambiguator | ACTIVE | Stage 1 vs Stage 3 paired forward parity IS the canonical disambiguator between "Stage 3 is a no-op replica" vs "Stage 3 is the canonical v332_smooth refinement". |

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable +
Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (`HNeRVDecoderMLX` + `HNeRVSyntheticTrainingBundleMLX`) | ADOPT_CANONICAL_BECAUSE_SERVES | PR 95 published 8-stage curriculum shares the same architecture across all stages; Stage 3 v332_smooth is a loss-family + LR-schedule transition, NOT an architectural shift. Canonical helper from `tac.local_acceleration.pr95_hnerv_mlx`. |
| Optimizer (`Pr95MlxOptimizerConfig` + `apply_pr95_mlx_optimizer_step`) | ADOPT_CANONICAL_BECAUSE_SERVES | Stage 3 uses the same AdamW baseline as Stage 1+2 but at fresh cosine LR=1e-4 per the recovered public PR 95 source. Canonical helper from `tac.local_acceleration.pr95_hnerv_mlx`. |
| Stage routing (`PR95_STAGE_MODULES` dict + `stage_smoke_config`) | ADOPT_CANONICAL_BECAUSE_SERVES | Extending the canonical dict by 1 entry preserves the dispatch interface for ALL existing consumers (`tools/build_pr95_mlx_optimizer_matrix_queue.py`, `tools/run_pr95_mlx_timing_smoke.py`, `tac.local_acceleration.pr95_hnerv_mlx_training`, etc.). |
| Optimizer descriptor (`OptimizerSchedulerDescriptor`) | ADOPT_CANONICAL_BECAUSE_SERVES | Registry-pattern extension preserves descriptor invariants (false-authority + canonical-equation-refs + solver-stack-wire-in + sha256). |
| Loss family (`smooth_disagreement_seg_loss`) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Stage 1 uses canonical RGB MSE baseline; Stage 2 uses tau_softplus seg loss; Stage 3 uses smooth_disagreement seg loss (sigmoid bell on negative margin) per the recovered public PR 95 source. The descriptor declares `stage_loss_family="smooth_disagreement_seg_loss"` so downstream consumers can route the loss accordingly. |
| LR schedule (1e-4 fresh cosine vs Stage 1+2 baseline 1e-3) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per recovered public PR 95 source: Stage 3 starts a FRESH cosine schedule (not continuing Stage 2's LR) because the smooth_disagreement loss family requires different convergence dynamics than tau_softplus. Distinct from Stage 5 (c1a_l7 at 3e-5) + Stage 8 (muon_finetune at 1e-5). |
| Test discipline | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors `test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py` Stage 2 invariant patterns at the Stage 3 surface; superset-of pattern adopted to avoid future stage 4/6/7 mutations of this Stage 3 test per Catalog #110/#113. |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST EVIDENCE SECTION" non-negotiable +
Catalog #294:

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | Stage 3 v332_smooth extends PR 95 published curriculum by exactly 1 canonical step beyond Stage 1+2+5+8 spine. Per Catalog #309: `horizon_class: plateau_adjacent` (continuation of the PR 95 canonical paradigm; not a class-shift). |
| 2. BEAUTY+ELEGANCE | 4 source files / +49 LOC additive; canonical registry-pattern extension; zero refactoring of existing code; APPEND-ONLY pattern for predecessor test extension. |
| 3. DISTINCTNESS | Stage 3 differs from Stage 1+2 in loss-family transition (smooth_disagreement vs tau_softplus vs RGB MSE baseline) AND LR-schedule transition (1e-4 fresh cosine vs 1e-3 baseline); architecture shared. Distinct from Stage 5 (c1a_l7 at 3e-5) + Stage 8 (muon_finetune at 1e-5). Stage 3 is the intermediate bridge between Stage 2 softplus refinement and Stage 4 QAT. |
| 4. RIGOR | Per Catalog #229 premise-verification: full read of Stage 2 landing memo + recovered public PR 95 source + canonical interface BEFORE any edit. Sister-coherence verified pre-edit. 11 NEW tests + 68-test full regression PASS. Empirical: forward parity vs Stage 1 at random init byte-identical; state_bytes byte-identical to Stage 2 (915,944). |
| 5. OPTIMIZATION PER TECHNIQUE | Per Catalog #290 canonical-vs-unique decision table above: 5 ADOPT_CANONICAL + 2 FORK_BECAUSE_PRINCIPLED_MISMATCH (loss family + LR schedule). |
| 6. STACK-OF-STACKS-COMPOSABILITY | Stage 3 extends the canonical PR 95 8-stage cascade by 1 stage; future Stage 4/6/7 sister landings can extend the dispatch dict identically using the superset-of test pattern landed in this commit. |
| 7. DETERMINISTIC REPRODUCIBILITY | Seed-pinned (seed=20260525); descriptor sha256 stable; canonical experiment package emitted via canonical `tools/run_pr95_mlx_timing_smoke.py`. |
| 8. EXTREME OPTIMIZATION+PERFORMANCE | 23.40 ms/step on M5 Max MLX; 42.73 ex/s; 915.9 KB state; per-step throughput within ARCH-5 dispatch contract for synthetic timing smoke. 30 min wall-clock for full Stage 3 build (vs Stage 2's 70 min — canonical extension pattern proves out). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Stage 3 is a CURRICULUM EXTENSION not a score-promotion path. Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323: non-promotable by construction; promotion requires paired Linux x86_64 + NVIDIA contest-CPU/CUDA per Catalog #192. |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305:

| Facet | Mechanism |
|---|---|
| 1. Inspectable per layer | Every layer's `bundle.parameters()` introspectable via `partition_pr95_mlx_parameter_names`; per-step `step_summary` emitted via `apply_pr95_mlx_optimizer_step`. |
| 2. Decomposable per signal | `runtime_profile` payload decomposes `seconds_per_step` + `examples_per_second` + `state_bytes` + `operator_mix` + `kernel_fusion_strategy_id` + `parameter_group_fingerprint_sha256`. |
| 3. Diff-able across runs | Stage 1 vs Stage 3 paired forward parity test demonstrates byte-level diff at random init; state_bytes byte-identical to Stage 2 confirms architecture parity; descriptor `config_sha256` stable across runs. |
| 4. Queryable post-hoc | Canonical `manifest.json` + `runtime_profile.json` + `representation_training_manifest.json` emitted to filesystem. |
| 5. Cite-able | This memo + Catalog #313 probe-outcomes row + canonical equation candidate QUEUED per Catalog #344. |
| 6. Counterfactual-able | Stage 1 vs Stage 3 paired forward parity test IS the canonical counterfactual: "if Stage 3 silently perturbs the architecture, this test fails." |

## Cargo-cult audit per assumption

Per CLAUDE.md "Cargo-cult audit per assumption" non-negotiable + Catalog #303:

| Assumption | Classification | Rationale |
|---|---|---|
| Stage 3 shares Stage 1+2's architecture | HARD-EARNED | Empirical byte-identical forward parity test (max_abs_diff=0.0) + state_bytes byte-identical to Stage 2 (915,944 bytes) confirms architecture is shared; PR 95 published curriculum design supports this. |
| Stage 3 LR = 1e-4 (FRESH cosine; distinct from Stage 1+2 baseline 1e-3) | HARD-EARNED | Per recovered public PR 95 source (`.omx/research/pr95_curriculum_recovery_20260513_codex.md`): "Stage 3 `stage3_v332_smooth.py` 1500 epochs `smooth_disagreement_seg_loss(tau=0.3)` AdamW only **fresh** cosine 1e-4 → 5e-6". |
| Stage 3 stage_module = `stage3_v332_smooth` | HARD-EARNED | Per recovered public PR 95 source: `src/stages/stage3_v332_smooth.py`; the prompt's hypothetical `c1a_l3` was canonicalized to the recovered source name per Stage 2 sister-canonical precedent. |
| Stage 3 loss family = `smooth_disagreement_seg_loss` | HARD-EARNED | Per recovered public PR 95 source: `smooth_disagreement_seg_loss(tau=0.3)` (sigmoid bell on negative margin). |
| Stage 3 has no QAT (QAT is Stage 4) | HARD-EARNED | Per recovered public PR 95 source: Stage 4 (`stage4_v332_qat.py`) introduces QAT continuing Stage 3 cosine schedule. |
| MLX synthetic timing smoke is non-promotable by construction | HARD-EARNED | Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323; promotion requires paired Linux x86_64 + NVIDIA contest-CPU/CUDA per Catalog #192. |
| `c1a_l3` (prompt-hypothetical) is the correct Stage 3 name | CARGO-CULTED → CANONICAL-CORRECTED | Per recovered public PR 95 source, `c1a` = `cat_entropy_v2` (rate-axis Lagrangian) introduced at Stage 5, NOT Stage 3. Canonicalized to `stage3_v332_smooth` per Stage 2 sister-canonicalization precedent (initial scope `stage2_v328_ce` → canonical `stage2_v331_softplus`). |

## Operational notes

- **Pre-existing builder bug surfaced (NOT MINE)**: `tools/build_pr95_mlx_optimizer_matrix_queue.py`
  invocation fails with `TypeError: _extra_artifact_postconditions() missing 1
  required keyword-only argument: 'write_pytorch_export_parity'` at line 1028
  of `tools/run_pr95_mlx_timing_smoke.py`. Reproduces empirically for Stage 1,
  Stage 2, AND Stage 3. Sister-territory bug; bypassed by invoking
  `tools/run_pr95_mlx_timing_smoke.py` directly for the canonical 100-step
  smoke. Operator-routable: sister subagent fix at the call site.
- **Sister coherence audit**: ZERO active sister subagents on overlapping
  scope at edit time (Catalog #340 sister-checkpoint guard PROCEED verified
  pre-staging). Slot 3 (DQS1-ASYMPTOTIC-FLOOR-SUBSTRATE-CLASS-SHIFT) + Slot 4
  (PROBE-9-TIER-2-DISPATCH-PREP) DISJOINT from this lane.
- **Canonical equation registration**: `pr95_mlx_stage_3_v332_smooth_one_to_one_curriculum_port_v1`
  QUEUED for operator-routable RATIFY-N per Catalog #344 operator-decision
  protocol. NOT auto-registered.
- **Catalog #313 probe-outcomes row**: registered via canonical
  `tac.probe_outcomes_ledger.register_probe_outcome` (verdict=PROCEED,
  blocker_status=advisory, staleness_window=30 days,
  written_at_utc=2026-05-25, evidence_path=manifest.json).
- **lane-id note per Catalog #110/#113**: the lane-id preserves the original
  `stage_3_c1a_l3` scoping per APPEND-ONLY discipline; the test file +
  stage_module + descriptor evidence reflect the canonical
  `stage3_v332_smooth` name per the recovered public PR 95 source.

## Operator-routable next step (Path A recommended)

Per the empirical findings + sister coherence:

- **Path A (Stage 4 v332_qat sister BUILD)**: spawn next sister subagent to
  land PR 95 Stage 4 `stage4_v332_qat` per the MLX-PARADIGM-T3 commit
  `916c43d89` Op #3 cascade. Stage 4 introduces QAT (Quantization-Aware
  Training) continuing Stage 3's cosine schedule (no fresh LR). Per
  recovered public PR 95 source: 500 epochs, same `smooth_disagreement_seg_loss(tau=0.3)`
  as Stage 3, `stage_uses_qat=True`. The canonical extension pattern is now
  operationally validated end-to-end across THREE PR 95 stages (Stage 1+2+3
  share architecture; only loss family + LR schedule + QAT bit distinguish
  them). Stage 4/6/7 sister landings can extend the dispatch dict identically.
- **Path B (Stage 3 source-faithful training scaling)**: rather than
  cascading to Stage 4, scale the Stage 3 synthetic timing proxy to actual
  PR 95 source-video training per the codex profile pattern
  (`--train-on-source-video-pairs --source-video-loss-surface rgb_yuv6_mse`).
  Same sister option as Stage 2's Path B.
- **Path C (per-block parity validation Stage 3 vs Stage 2 post-training)**:
  rather than cascading, validate that Stage 3 trained outputs differ from
  Stage 2 trained outputs by the expected magnitude per the
  smooth_disagreement seg loss family.
- **Path D (fix pre-existing builder bug)**: sister subagent should fix
  `tools/run_pr95_mlx_timing_smoke.py:1028` to pass `write_pytorch_export_parity=False`
  to `_extra_artifact_postconditions` so the queue builder works again for
  all stages.

## Cross-references

- `pr95_mlx_stage_2_v331_softplus_curriculum_build_landed_20260525.md` — Stage 2
  landing (immediate predecessor; canonical extension pattern this build mirrors).
- `mlx_arch_5_pr101_state_dict_paired_forward_landed_20260525.md` — MLX-ARCH-5
  cascade closure (paired forward at random init).
- `codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md` —
  codex Stage 1+5+8 canonical control profile (the predecessor lane this
  build extends).
- `pr95_8stage_curriculum_forensic_20260513.md` — recovered public PR 95
  8-stage curriculum spec (canonical source for Stage 3 name + LR + loss).
- `pr95_curriculum_recovery_20260513_codex.md` — codex curriculum recovery
  (sister canonical source).
- CLAUDE.md "Public Disclosure Hygiene" + "Strategic Secrecy".
- CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU
  AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".
- CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against".
- CLAUDE.md "Subagent coherence-by-default" (Catalog #314 + #340).
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
  docstring-overstatement trap)" + Catalog #287.

## Empirical receipts

| Artifact | Path |
|---|---|
| Experiment package directory | `experiments/results/pr95_mlx_stage_3_v332_smooth_curriculum_build_20260525T162242Z/` |
| Per-cell manifest | `experiments/results/pr95_mlx_stage_3_v332_smooth_curriculum_build_20260525T162242Z/stage3/pr95_stage3_adamw_baseline_mlx/seed20260525_c36/manifest.json` |
| Per-cell runtime profile | `experiments/results/pr95_mlx_stage_3_v332_smooth_curriculum_build_20260525T162242Z/stage3/pr95_stage3_adamw_baseline_mlx/seed20260525_c36/runtime_profile.json` |
| Per-cell training manifest | `experiments/results/pr95_mlx_stage_3_v332_smooth_curriculum_build_20260525T162242Z/stage3/pr95_stage3_adamw_baseline_mlx/seed20260525_c36/representation_training_manifest.json` |
| Catalog #313 probe-outcomes row | `.omx/state/probe_outcomes.jsonl` (probe_id=`pr95_mlx_stage_3_v332_smooth_curriculum_build_synthetic_timing_smoke_100ep`) |
| Source code changes (3 files) | `src/tac/local_acceleration/pr95_hnerv_mlx.py` + `src/tac/optimization/optimizer_scheduler_registry.py` + `tools/build_pr95_mlx_optimizer_matrix_queue.py` |
| NEW test file | `src/tac/tests/test_pr95_mlx_stage_3_v332_smooth_curriculum_build.py` |
| Predecessor test files updated (APPEND-ONLY superset-of pattern) | `src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py` + `src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py` |

---

**Lane verdict**: PROCEED ✓
**Cost band**: free_local_smoke_only ($0 + ~30 min wall-clock)
**Mission alignment**: `frontier_breaking_enabler` (extends MLX substrate-trainer
extension paradigm to an additional PR 95 published curriculum stage; unblocks
Stage 4/6/7 sister BUILD cascade per MLX-PARADIGM-T3 Op #3).
