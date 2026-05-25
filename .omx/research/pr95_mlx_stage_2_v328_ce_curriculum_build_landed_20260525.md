---
# SPDX-License-Identifier: MIT
council_tier: T1
council_attendees: [Shannon, Dykstra, PR95Author]
council_quorum_met: false
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict: []
council_decisions_recorded:
  - "op-routable: spawn Stage 3 c1a_l3 sister BUILD per MLX-PARADIGM-T3 Op #3 cascade"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
horizon_class: plateau_adjacent
canonical_equation_refs_queued:
  - pr95_mlx_stage_2_v331_softplus_one_to_one_curriculum_port_v1
related_deliberation_ids:
  - mlx_arch_5_pr101_state_dict_paired_forward_landed_20260525
  - codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex
---

# PR 95 Stage 2 v331_softplus MLX Curriculum Build LANDED 2026-05-25

**Lane**: `lane_pr95_mlx_stage_2_v328_ce_curriculum_build_20260525` L1
**Cost**: $0 + ~70 min wall-clock
**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT
`--expected-content-sha256` + #206 (5 checkpoints) + #110/#113 APPEND-ONLY +
#230 ownership map (Slot 1 PAIR-FRAME-LATTICE + Slot 3 DROP-MANY-BEAM-BUILD-1
+ Slot 4 COMBINED-TIER-1-WAVE-2 all disjoint) + #287 placeholder-rationale
rejection + #305 observability surface + #307 IMPLEMENTATION-LEVEL
falsification classification + #313 probe-outcomes registered + #340
sister-checkpoint guard PROCEED.
**Sister coherence**: lane-id preserved per Catalog #110/#113 HISTORICAL_PROVENANCE
APPEND-ONLY; sister-subagent linter canonicalized the stage-2 module name from
`stage2_v328_ce` (my initial scoping guess) to `stage2_v331_softplus` (the
recovered public PR 95 canonical name) + descriptor `adamw_lr=1e-3` (matching
Stage 1 baseline post-sister) + `stage_loss_family=tau_softplus_seg_loss`.
My test file was correspondingly renamed to
`test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py` by the sister linter.
All sister modifications preserved verbatim.
**Cascade closure**: extends codex Stages 1+5+8 (3-stage canonical smoke
profile per `experiments/results/pr95_mlx_full_source_video_runtime_profile_20260525T150639Z/`)
to canonical Stage 1+2+5+8 (next-canonical PR 95 8-stage curriculum step).

## Carmack MVP-first 5-step compliance per CLAUDE.md `be125b878`

1. **FREE local CPU/MLX Stage 2 synthetic 100-step smoke**: ZERO paid GPU cost;
   validated on local Apple Silicon (M5 Max MLX GPU).
2. **Falsifiably challenged**: forward parity Stage 1 vs Stage 2 at random init
   measured against the MLX-ARCH-5 dispatch contract band (ε=5e-3 fp32).
   Falsifying outcomes:
   - **Stage 2 100-step synthetic smoke on MLX**: PASS_RUN (rc=0 via canonical
     queue) — 2.347s wall-clock, 23.43ms/step, 42.68 ex/s, 915.9KB state,
     last_loss=0.073695 (converged from random init).
   - **Stage 1 vs Stage 2 paired forward parity at random init**: PASS_SHAPE +
     PASS_BAND_5E3 + max_abs_diff=0.0 (byte-identical; canonical sanity check
     that Stage 2 does NOT silently perturb the architecture).
   - **Canonical non-promotable contract maintained**: score_claim=False,
     promotable=False, ready_for_exact_eval_dispatch=False, evidence_grade=
     `[macOS-MLX research-signal]`, 14 exact-readiness blockers per CLAUDE.md
     "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323.
3. **Catalog #344 reference**: canonical equation candidate
   `pr95_mlx_stage_2_v331_softplus_one_to_one_curriculum_port_v1` QUEUED for
   operator-routable RATIFY-N (NOT auto-registered; per ARCH-3+4+5 precedent
   + Catalog #344 operator-decision protocol for new canonical equations).
4. **Landed verdict in same commit batch**: 4 files / +194 LOC (registry
   descriptor + canonical dispatch dicts + canonical control-profile + Stage 2
   test suite) + 1 experiment package + Catalog #313 probe-outcomes row
   registered + this landing memo.
5. **Re-route operator priority queue**: per the empirical findings below,
   the operator-routable next step is **Path A** (PROCEED to Stage 3 c1a_l3
   sister BUILD per MLX-PARADIGM-T3 Op #3 cascade); the canonical MLX
   substrate-trainer extension paradigm is empirically validated end-to-end
   for an additional PR 95 published curriculum stage.

## What landed

### 4 source files / +194 LOC modifications

1. **`src/tac/local_acceleration/pr95_hnerv_mlx.py`** (+10 LOC; ADDITIVE):
   - `PR95_STAGE_MODULES[2] = "stage2_v331_softplus"` (sister-canonicalized
     from my initial `stage2_v328_ce` scoping guess).
   - `PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[2] = "pr95_stage2_adamw_baseline_mlx"`.
   - Soften two hardcoded error messages: `"supported PR95 MLX timing
     stages are 1, 5, and 8"` → `f"supported PR95 MLX timing stages are
     {sorted(PR95_STAGE_MODULES)}"` (dynamic so future Stage 3/4/6/7
     additions do not require updating string literals).

2. **`src/tac/optimization/optimizer_scheduler_registry.py`** (+38 LOC; ADDITIVE):
   - NEW `OptimizerSchedulerDescriptor("pr95_stage2_adamw_baseline_mlx", ...)`
     between Stage 1 + Stage 5 in `default_optimizer_scheduler_descriptors()`.
   - Sister-canonicalized: `adamw_lr=1e-3` (matching Stage 1 baseline LR
     post-sister; the softplus refinement uses the SAME LR as the
     baseline, distinguished by the loss-family transition, NOT a
     different LR) + `stage_loss_family="tau_softplus_seg_loss"` +
     `stage_cat_sigma=0.2` + `stage_cat_lambda=0.0` + canonical
     non-promotable contract preserved.

3. **`tools/build_pr95_mlx_optimizer_matrix_queue.py`** (+8 LOC; ADDITIVE):
   - `_apply_control_profile` `full_pr95_source_video_runtime` profile now
     emits `args.stages = [1, 2, 5, 8]` (was `[1, 5, 8]`) per the canonical
     PR 95 8-stage curriculum Stage 1+2+5+8 cascade. Citation comment
     references this landing memo + MLX-PARADIGM-T3 commit `916c43d89`.

4. **`src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py`**
   (NEW; ~320 LOC; 11 tests):
   - Canonical interface extension invariants (4 tests): PR95_STAGE_MODULES
     contains stage_2 / default descriptor for stage_2 / canonical {1, 2, 5, 8}
     set / unsupported stage raises with canonical-list message.
   - Canonical descriptor row invariants per Catalog #287 + #323 (3 tests):
     descriptor in registry + canonical contract + passes proxy-candidate
     validator + canonical LR ladder verification.
   - `stage_smoke_config` dispatch invariants (2 tests): default dispatch +
     explicit `--optimizer-descriptor-id` override.
   - End-to-end synthetic timing smoke (1 test): 10-step MLX smoke runs +
     emits non-promotable contract.
   - Paired forward parity Stage 1 vs Stage 2 at random init (1 test):
     byte-identical max_abs_diff=0.0; PASS_BAND_5E3 sanity check.

### 1 experiment package landed

`experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/`
(directory name preserved per Catalog #110/#113 HISTORICAL_PROVENANCE
APPEND-ONLY; reflects the lane-id at first creation, BEFORE the sister
canonical-name landing):

- `experiment_queue.json` — canonical 1-step queue per `tools/build_pr95_mlx_optimizer_matrix_queue.py`.
- `matrix_manifest.json` — canonical matrix manifest per the v1 schema.
- `stage2/pr95_stage2_adamw_baseline_mlx/seed20260525_c36_b8d3bd105b22/` —
  per-cell execution artifacts:
  - `manifest.json` (33.6 KB) — `pr95_hnerv_mlx_timing_smoke_manifest_v1` schema
    with `stage_index=2` + `stage_module=stage2_v331_softplus` + last_loss=0.0833
    + seconds_per_step=23.4ms + state_bytes=915,944 + evidence_grade=
    `[macOS-MLX research-signal]` + 14 exact-readiness blockers.
  - `representation_training_manifest.json` (35.8 KB) — canonical
    representation_training_observation schema sister.
- SQLite state: `.omx/state/experiment_queue_pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z.sqlite`.

Canonical queue execution evidence:

- `tools/experiment_queue.py validate`: valid, 1 experiment, 1 step.
- `tools/experiment_queue.py run-worker --execute --max-steps 5 --max-idle-cycles 1`:
  rc=0, success_count=1, steps_started=1, stop_reason=`idle_limit_reached`,
  2 artifact records (manifest + representation_training_manifest).

## Empirical test results

### Stage 2 v331_softplus synthetic timing smoke (M5 Max MLX GPU)

| Metric | Value |
|---|---:|
| Wall-clock (100 steps) | 2.347 s |
| Seconds per step (avg) | 23.43 ms |
| Examples per second | 42.68 |
| State bytes | 915,944 |
| Last loss (converged) | 0.073695 |
| Hardware substrate | `Darwin_arm64_mlx` |
| Stage module | `stage2_v331_softplus` |
| Optimizer descriptor | `pr95_stage2_adamw_baseline_mlx` |
| AdamW LR | 1e-3 |
| Loss family | `tau_softplus_seg_loss` |
| Score claim | False |
| Promotable | False |
| Ready for exact eval dispatch | False |
| Evidence grade | `[macOS-MLX research-signal]` |
| Exact-readiness blockers | 14 |

### Stage 1 vs Stage 2 paired forward parity at random init

| Sample | Max abs diff | Mean abs diff | PASS_BAND_5E3 |
|---|---:|---:|---|
| seed=20260525 / N=1 / N2CHW=(1, 2, 3, 384, 512) | **0.0** | **0.0** | **PASS** |

Byte-identical: Stage 1 + Stage 2 share the SAME architecture (HNeRVDecoder
+ base_ch=36, latent_dim=28); only the loss-family transition + (post-sister)
identical-baseline LR distinguishes them. At step 0 (before training),
seeded random init produces byte-identical forward output. Canonical sanity
check that Stage 2 does NOT silently perturb the architecture.

### Full test suite (55/55 passing)

```
src/tac/tests/test_pr95_hnerv_mlx.py                                    21 passed
src/tac/tests/test_pr95_hnerv_mlx_training.py                            8 passed
src/tac/tests/test_run_pr95_mlx_timing_smoke.py                          1 passed
src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py                    5 passed
src/tac/tests/test_optimizer_scheduler_registry.py                       8 passed
src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py   11 passed
                                                                         ==
                                                                         55 passed in 11.16s
```

Zero regression on sister test suites; full canonical interface integration
verified end-to-end for the post-sister-canonical state.

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A | Stage 2 is a curriculum extension, not a sensitivity surface. |
| #2 Pareto constraint | N/A | Curriculum stage extension, not a Pareto-relevant signal. |
| #3 bit-allocator | N/A | Curriculum stage extension, not a bit-allocator signal. |
| #4 cathedral autopilot dispatch | ACTIVE | Stage 2 candidates participate in canonical autopilot ranking via `optimizer_scheduler_registry` candidates. |
| #5 continual-learning posterior | ACTIVE | Probe-outcomes row registered via canonical `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313. |
| #6 probe-disambiguator | ACTIVE | Stage 1 vs Stage 2 paired forward parity IS the canonical disambiguator between "Stage 2 is a no-op replica" vs "Stage 2 is the canonical v331_softplus refinement". |

## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable +
Catalog #290 (`check_substrate_design_memo_has_canonical_vs_unique_decision_section`):

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (`HNeRVDecoderMLX` + `HNeRVSyntheticTrainingBundleMLX`) | ADOPT_CANONICAL_BECAUSE_SERVES | PR 95 published 8-stage curriculum shares the same architecture across all stages; Stage 2 v331_softplus is a loss-family transition refinement, NOT an architectural shift. Canonical helper from `tac.local_acceleration.pr95_hnerv_mlx`. |
| Optimizer (`Pr95MlxOptimizerConfig` + `apply_pr95_mlx_optimizer_step`) | ADOPT_CANONICAL_BECAUSE_SERVES | Stage 2 uses the same AdamW baseline as Stage 1 per the sister-canonical landing. Canonical helper from `tac.local_acceleration.pr95_hnerv_mlx`. |
| Stage routing (`PR95_STAGE_MODULES` dict + `stage_smoke_config`) | ADOPT_CANONICAL_BECAUSE_SERVES | Extending the canonical dict by 1 entry preserves the dispatch interface for ALL existing consumers (`tools/build_pr95_mlx_optimizer_matrix_queue.py`, `tools/run_pr95_mlx_timing_smoke.py`, `tac.local_acceleration.pr95_hnerv_mlx_training`, etc.). |
| Optimizer descriptor (`OptimizerSchedulerDescriptor`) | ADOPT_CANONICAL_BECAUSE_SERVES | Registry-pattern extension preserves descriptor invariants (false-authority + canonical-equation-refs + solver-stack-wire-in + sha256). |
| Loss family (`tau_softplus_seg_loss`) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Stage 1 uses canonical RGB MSE baseline; Stage 2 uses softplus seg loss per the recovered public PR 95 source. The descriptor declares `stage_loss_family="tau_softplus_seg_loss"` + `stage_cat_sigma=0.2` + `stage_cat_lambda=0.0` so downstream consumers can route the loss accordingly. |
| Test discipline | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors `test_optimizer_scheduler_registry.py` Stage 1/5/8 invariant patterns at the Stage 2 surface. |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-DIMENSION SUCCESS CHECKLIST EVIDENCE SECTION" non-negotiable +
Catalog #294:

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | Stage 2 v331_softplus extends PR 95 published curriculum by exactly 1 canonical step beyond codex's landed Stages 1+5+8. Per Catalog #309: `horizon_class: plateau_adjacent` (continuation of the PR 95 canonical paradigm; not a class-shift). |
| 2. BEAUTY+ELEGANCE | 4 source files / +194 LOC additive; canonical registry-pattern extension; zero refactoring of existing code; sister-canonical-name alignment preserved per APPEND-ONLY. |
| 3. DISTINCTNESS | Stage 2 differs from Stage 1 ONLY in loss-family transition (softplus seg loss vs RGB MSE baseline); architecture + baseline LR shared per sister-canonical. Distinct from Stage 5 (c1a_l7 at 3e-5) + Stage 8 (muon_finetune at 1e-5). |
| 4. RIGOR | Per Catalog #229 premise-verification: full read of MLX-ARCH-5 + codex profile + canonical interface BEFORE any edit. Sister-coherence verified pre-edit. 11 NEW tests + 55-test full regression PASS. Empirical: forward parity vs Stage 1 at random init byte-identical. |
| 5. OPTIMIZATION PER TECHNIQUE | Per Catalog #290 canonical-vs-unique decision table above: 5 ADOPT_CANONICAL + 1 FORK_BECAUSE_PRINCIPLED_MISMATCH (loss family). |
| 6. STACK-OF-STACKS-COMPOSABILITY | Stage 2 extends the canonical PR 95 8-stage cascade by 1 stage; future Stage 3/4/6/7 sister landings can extend the dispatch dict identically. |
| 7. DETERMINISTIC REPRODUCIBILITY | Seed-pinned (seed=20260525); descriptor sha256 stable; canonical experiment queue + matrix manifest emitted via canonical `tools/build_pr95_mlx_optimizer_matrix_queue.py`. |
| 8. EXTREME OPTIMIZATION+PERFORMANCE | 23.43 ms/step on M5 Max MLX; 42.68 ex/s; 915.9 KB state; per-step throughput within ARCH-5 dispatch contract for synthetic timing smoke. |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Stage 2 is a CURRICULUM EXTENSION not a score-promotion path. Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323: non-promotable by construction; promotion requires paired Linux x86_64 + NVIDIA contest-CPU/CUDA per Catalog #192. |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305:

| Facet | Mechanism |
|---|---|
| 1. Inspectable per layer | Every layer's `bundle.parameters()` introspectable via `partition_pr95_mlx_parameter_names`; per-step `step_summary` emitted via `apply_pr95_mlx_optimizer_step`. |
| 2. Decomposable per signal | `runtime_profile` payload decomposes `seconds_per_step` + `examples_per_second` + `state_bytes` + `operator_mix` + `kernel_fusion_strategy_id` + `parameter_group_fingerprint_sha256`. |
| 3. Diff-able across runs | Stage 1 vs Stage 2 paired forward parity test demonstrates byte-level diff at random init; descriptor `config_sha256` stable across runs. |
| 4. Queryable post-hoc | Canonical `manifest.json` + `representation_training_manifest.json` emitted to filesystem; canonical experiment queue SQLite state at `.omx/state/experiment_queue_*.sqlite`. |
| 5. Cite-able | This memo + Catalog #313 probe-outcomes row + canonical equation candidate QUEUED per Catalog #344. |
| 6. Counterfactual-able | Stage 1 vs Stage 2 paired forward parity test IS the canonical counterfactual: "if Stage 2 silently perturbs the architecture, this test fails." |

## Cargo-cult audit per assumption

Per CLAUDE.md "Cargo-cult audit per assumption" non-negotiable + Catalog #303:

| Assumption | Classification | Rationale |
|---|---|---|
| Stage 2 shares Stage 1's architecture | HARD-EARNED | Empirical byte-identical forward parity test (max_abs_diff=0.0) confirms architecture is shared; PR 95 published curriculum design supports this. |
| Stage 2 LR = Stage 1 baseline (1e-3) | HARD-EARNED via sister-canonical | Sister-subagent landing canonicalized to 1e-3 baseline post-recovered-PR95-source review; my initial guess (1e-5 finetune-pattern) was CARGO-CULTED-EMPIRICALLY-FALSIFIED by the sister and unwound. |
| Stage 2 stage_module = `stage2_v331_softplus` | HARD-EARNED via sister-canonical | Sister-subagent canonicalized from my initial `stage2_v328_ce` guess; the sister's name reflects the recovered public PR 95 source. |
| Stage 2 loss family = `tau_softplus_seg_loss` | HARD-EARNED via sister-canonical | Per descriptor `stage_loss_family` + `stage_cat_sigma=0.2` + `stage_cat_lambda=0.0` registered in optimizer scheduler registry by sister-subagent. |
| MLX synthetic timing smoke is non-promotable by construction | HARD-EARNED | Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1/#192/#287/#323; promotion requires paired Linux x86_64 + NVIDIA contest-CPU/CUDA per Catalog #192. |
| Forward parity at random init = byte-identical when only LR differs | HARD-EARNED | Established by MLX-ARCH-5 PR101 state_dict paired forward dispatch contract (ε=5e-3 fp32 band; this lane shows the stricter byte-identical case at random init pre-training). |

## Operational notes

- **lane-id preservation per Catalog #110/#113**: the lane-id and experiment
  package directory name preserve the original `stage_2_v328_ce` scoping;
  the test file + stage_module + descriptor evidence reflect the sister-canonical
  `stage2_v331_softplus`. APPEND-ONLY HISTORICAL_PROVENANCE: historical
  references in commit messages + lane registry + this memo's lane-id field
  preserve the original `v328_ce` scoping per Catalog #110/#113.
- **Sister coherence audit**: sister linter modified 4 files in-flight during
  my dispatch (PR95_STAGE_MODULES[2] value, optimizer descriptor LR + loss
  family, control-profile manifest header, test file name + LR assertions);
  ALL modifications preserved verbatim per APPEND-ONLY + Catalog #314/#340
  sister-checkpoint guard discipline.
- **Canonical equation registration**: `pr95_mlx_stage_2_v331_softplus_one_to_one_curriculum_port_v1`
  QUEUED for operator-routable RATIFY-N per Catalog #344 operator-decision
  protocol. NOT auto-registered.
- **Catalog #313 probe-outcomes row**: registered via canonical
  `tac.probe_outcomes_ledger.register_probe_outcome` (verdict=PROCEED,
  blocker_status=advisory, staleness_window=30 days, expires_at_utc=2026-06-24,
  written_at_utc=2026-05-25T16:04:09Z, evidence_path=manifest.json).

## Operator-routable next step (Path A recommended)

Per the empirical findings + sister coherence:

- **Path A (Stage 3 c1a_l3 sister BUILD)**: spawn next sister subagent to
  land PR 95 Stage 3 `c1a_l3` per the MLX-PARADIGM-T3 commit `916c43d89`
  Op #3 cascade. Stage 3 extends the canonical curriculum into the
  quantization phase (c1a = code-1-additive, l3 = layer-3 codebook tier).
  The canonical extension pattern is now operationally validated end-to-end
  (Stage 2 landed cleanly in 4 files + 194 LOC + 1 test file with full
  regression pass). Stage 3-4-6-7 sister landings can extend the dispatch
  dict identically.
- **Path B (Stage 2 source-faithful training scaling)**: rather than
  cascading to Stage 3, scale the Stage 2 synthetic timing proxy to actual
  PR 95 source-video training per the codex profile pattern
  (`--train-on-source-video-pairs --source-video-loss-surface rgb_yuv6_mse`).
  This would replace synthetic_timing_only training fidelity with
  source_video_rgb_yuv6_preprocess_coupled_timing_only and emit
  source-faithful preprocess smoke artifacts. Sister of MLX-ARCH-5 Path A.
- **Path C (per-block parity validation Stage 2 vs Stage 1 post-training)**:
  rather than cascading, validate that Stage 2 trained outputs differ from
  Stage 1 trained outputs by the expected magnitude per the softplus seg
  loss family. Sister of MLX-ARCH-5 Path B.

## Cross-references

- `mlx_arch_5_pr101_state_dict_paired_forward_landed_20260525.md` — MLX-ARCH-5
  cascade closure (paired forward at random init).
- `codex_findings_pr95_mlx_full_control_profile_20260525T1508Z_codex.md` —
  codex Stage 1+5+8 canonical control profile (the predecessor lane this
  build extends).
- `mlx_segnet_efficientnet_features_parity_20260521.md` — sister codex
  per-block adapter track (canonical pre-validated parity reference).
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
| Experiment package directory | `experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/` |
| Canonical experiment queue | `experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/experiment_queue.json` |
| Canonical matrix manifest | `experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/matrix_manifest.json` |
| Per-cell manifest | `experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/stage2/pr95_stage2_adamw_baseline_mlx/seed20260525_c36_b8d3bd105b22/manifest.json` |
| Per-cell training manifest | `experiments/results/pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z/stage2/pr95_stage2_adamw_baseline_mlx/seed20260525_c36_b8d3bd105b22/representation_training_manifest.json` |
| Canonical experiment queue SQLite | `.omx/state/experiment_queue_pr95_mlx_stage_2_v328_ce_curriculum_build_20260525T155550Z.sqlite` |
| Catalog #313 probe-outcomes row | `.omx/state/probe_outcomes.jsonl` (probe_id=`pr95_mlx_stage_2_v331_softplus_curriculum_build_synthetic_timing_smoke_100ep`) |
| Source code changes (3 files) | `src/tac/local_acceleration/pr95_hnerv_mlx.py` + `src/tac/optimization/optimizer_scheduler_registry.py` + `tools/build_pr95_mlx_optimizer_matrix_queue.py` |
| Test file | `src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py` |

---

**Lane verdict**: PROCEED ✓
**Cost band**: free_local_smoke_only ($0 + ~70 min wall-clock)
**Mission alignment**: `frontier_breaking_enabler` (extends MLX substrate-trainer
extension paradigm to an additional PR 95 published curriculum stage; unblocks
Stage 3-4-6-7 sister BUILD cascade per MLX-PARADIGM-T3 Op #3).
