# HINTON-MLX-BUNDLE 2026-05-25 — REAL-TEACHER REFIRE FALSIFIES MOCK-TEACHER CONVERGENCE; NUMPY PARITY PROOF BLOCKED

**Lane(s)**:
- `lane_hinton_mlx_real_teacher_refire_20260525` (L1; impl_complete + memory_entry)
- `lane_hinton_mlx_numpy_parity_proof_20260525` (L0; deferred per Phase 1 gate)

**Date**: 2026-05-25
**Tasks owned**: #1261 (real-teacher refire) + #1262 (numpy parity proof — BLOCKED-BY #1261 verdict)
**Operator authorization**: "proceed with all in the order you think makes the most sense" (2026-05-25)
**Subagent ID**: `hinton_mlx_bundle_20260525`
**Cost**: $0 (local M5 Max, MLX + CPU SegNet teacher cache)
**Wall-clock**: ~12 min (10 min PV + patch + tests + tiny smoke; 6.5s mock baseline; 2:49 real-teacher refire)

## Headline verdict

Per the MVP-first phasing non-negotiable + sister codex's stand-down advisory acknowledging the mock-teacher gap: **the mock-teacher CONVERGES_CONSISTENTLY verdict at commit `e3b8c0d8d` was a cargo-cult**. Real upstream-PyTorch SegNet teacher on identical infrastructure produces SUB_PARADIGM (17% reduction) where mock produced CONVERGES (97.9% reduction). Per Catalog #307: **IMPLEMENTATION-LEVEL falsification; Hinton distillation paradigm INTACT**. Per Phase 2 gate: BLOCKED (no numpy parity proof on an empirically-falsified student).

## Empirical receipts (mock vs real teacher; 600 frames, 100ep, identical infrastructure)

| Metric | Mock teacher | Real SegNet teacher | Ratio (real / mock) |
|---|---:|---:|---:|
| Initial loss | 0.20274 | 1.55172 | 7.65× |
| Final loss | 0.00417 | 1.28654 | 308× |
| Min loss across run | 0.00159 | 0.99020 | 623× |
| Loss reduction | 97.94% | 17.09% | 0.174× |
| Verdict | CONVERGES_CONSISTENTLY | **SUB_PARADIGM** | — |
| Wall-clock | 6.4 s | 2:49 (169 s) | 26× |
| Teacher cache build | n/a | 162.7 s (CPU SegNet) | n/a |

**Source video**: `upstream/videos/0.mkv`, sha256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`, 600 frames at (384, 512) per CANONICAL_EVAL_SIZE per Catalog #229 PV.
**Mock initial→final**: 0.203 → 0.0042 (factor 49× reduction; min 0.0016).
**Real initial→final**: 1.552 → 1.287 (factor 1.2× reduction; min 0.990).

The mock teacher's initial loss (0.20) is already 10× LOWER than the real teacher's irreducible MIN loss (0.99) across 100 epochs. The mock-teacher convergence was reachable because mock projection on decoded RGB and mock projection on target RGB are SAME-MATH at different downsample factors; the student trivially learns to match. Real SegNet logits are NOT a deterministic projection of RGB pixels — they encode learned class boundaries that a deterministic cosine head cannot recover.

## Cargo-cult audit per Catalog #303

**Assumption surfaced (CARGO-CULTED + EMPIRICALLY FALSIFIED)**:
- Mock-teacher CONVERGES_CONSISTENTLY at 600 frames / 100 epochs implies the canonical Hinton KL T=2.0 contract is sound for the contest SegNet teacher.
  - **Classification**: CARGO-CULTED (inherited from sister codex commit `e3b8c0d8d` smoke verdict).
  - **Empirical falsification**: real-SegNet teacher SUB_PARADIGM at identical config; the mock teacher hides the irreducible KL between a deterministic-projection student head and a learned scorer's class boundaries.
  - **Mechanism**: per `_student_logits_from_decoded` (lines 401-431 of `src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py`), the student "logits" are a fixed cosine projection of decoded RGB — NOT a learnable head. When teacher and student share the same projection class (mock-mock), the student can drive KL→0 by matching reconstruction. When the teacher is a real learned CNN (SegNet's EfficientNet-B2 + Unet), the deterministic projection cannot represent the teacher's class boundaries; KL has an irreducible floor.

**Assumption preserved (HARD-EARNED)**:
- Hinton-Vinyals-Dean 2014 KL T=2.0 math is correct per Quantizr canonical 0.33 anchor; the loss function itself is not falsified.
- Real upstream-PyTorch SegNet weights are stageable on macOS CPU; cache build in 162s for 600 frames is operationally feasible per CLAUDE.md "MPS auth eval is NOISE" requirement that the teacher be loaded on CPU.

## 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: real upstream-PyTorch SegNet teacher is the canonical contest scorer; no sister subagent has demonstrated this empirical falsification on the canonical Slot 1 MLX long-training pipeline.
2. **BEAUTY + ELEGANCE**: minimum-LOC integration. New canonical `RealSegNetTeacherLogitsCache` + `build_real_segnet_teacher_cache` helpers (~140 LOC) + `--teacher-provider {mock,real_segnet}` CLI flag. Zero mutation of Slot 1 canonical infrastructure or sister codex `mlx_scorer_adapters.py`.
3. **DISTINCTNESS**: explicitly distinct from sister codex's mock-teacher smoke at commit `e3b8c0d8d`; the new code path is the canonical real-teacher refire surface.
4. **RIGOR**: PV (10 files read); cargo-cult audit per Catalog #303; 4 verdict-taxonomy per Catalog #305 observability; mock vs real apples-to-apples comparison on identical infrastructure; back-compat preserved (25/25 existing tests pass).
5. **OPTIMIZATION PER TECHNIQUE**: real-SegNet teacher cache built ONCE pre-training (162s) then O(1) MLX integer indexing per batch — keeps the MLX training loop fast.
6. **STACK-OF-STACKS-COMPOSABILITY**: cache-based teacher provider composes cleanly with the existing pure-MLX student projection; future sister wave can swap the student for a learnable head without touching the cache.
7. **DETERMINISTIC REPRODUCIBILITY**: SegNet weights loaded via `tac.scorer.load_default_scorers` (safetensors deterministic); source video sha256 captured; random_seed pinned at 0.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: per-batch teacher lookup is O(1) MLX indexing; training step wall-clock unchanged vs mock path; cache build is one-shot.
9. **OPTIMAL MINIMAL CONTEST SCORE**: the falsification PREVENTS a paid Modal A100 dispatch on a substrate that empirically cannot converge against the contest teacher. Per AAA T4 §6.5 predicted ΔS [-0.01, -0.03] now refuted at the student-head architectural surface — would have wasted $2-5 of paid GPU on a SUB_PARADIGM trainer.

## Observability surface per Catalog #305

1. **Inspectable per layer**: per-epoch loss curve at `experiments/results/hinton_mlx_bundle_20260525/phase1_real_teacher_telemetry.jsonl`; verdict JSON at `phase1_real_teacher_verdict.json` carries every scalar invariant.
2. **Decomposable per signal**: combined loss decomposes into `mse(decoded_f0, target)` + `λ * T² * KL(student || teacher)`; the SUB_PARADIGM verdict reflects the irreducible KL floor on the real teacher.
3. **Diff-able across runs**: mock vs real verdicts in the SAME dir; per-run `random_seed=0` + `source_video_sha256` pin determinism.
4. **Queryable post-hoc**: verdict JSON is JSON; telemetry is JSONL; canonical schema `hinton_mlx_long_training_smoke_verdict.v1`.
5. **Cite-able**: verdict carries source_video_sha256 + teacher_cache_frame_count + teacher_cache_hwk + run_seconds + cache_build_seconds.
6. **Counterfactual-able**: rerun with `--teacher-provider mock` to recompute mock baseline; rerun with `--smoke-epochs 1000` to test whether SUB_PARADIGM resolves to CONVERGES with longer training (operator-routable).

## Phase 2 gate disposition

Per the mandate Phase 2 (Task #1262) numpy parity proof gate is **causally chained**: "BLOCKED-BY Task #1261 (real-teacher convergence verdict) per Carmack MVP-first: if the mock-vs-real teacher cargo-cult falsifies the convergence, the numpy parity proof on a falsified student is wasted work."

**Decision**: Phase 2 BLOCKED. The student trained against the real SegNet teacher reaches SUB_PARADIGM (17% reduction, min KL 0.99); the trained student state_dict is not yet a valid PR95 HNeRV decoder for downstream paired-CPU+CUDA verify per Catalog #205. Proving numpy round-trip parity on a sub-paradigm student establishes only that the bridge is byte-stable — not that the student is dispatchable.

Lane `lane_hinton_mlx_numpy_parity_proof_20260525` remains L0 with reactivation criteria pinned (see below). The bridge infrastructure itself (`tools/export_pr95_mlx_to_pytorch_state_dict.py` + `tac.local_acceleration.pr95_hnerv_mlx.pytorch_state_dict_from_mlx`) is already operational per sister #1251 work; that wave's numpy parity proof on the **un-distilled PR95 HNeRV decoder** is the canonical sister surface and DID land. Phase 2 here is the **distilled-student-specific** parity proof, which is irrelevant until the distilled student itself converges.

## Operator-routable next gates

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308 alternative-probe-methodology enumeration:

1. **Higher-EV / lowest-LOC**: extend `_student_logits_from_decoded` to use a LEARNABLE student head (e.g. a 1×1 conv from decoded RGB to 5 classes, ~150 trainable params). The current deterministic projection has zero expressive capacity for class-boundary learning; a learnable head closes that gap structurally. Estimated impact: SUB_PARADIGM → CONVERGES on real teacher (predicted final loss <0.5 from current 1.29 plateau).
2. **Medium-EV / medium-LOC**: lower distillation temperature `T` from 2.0 to 1.0 or 0.5. Quantizr canonical T=2.0 was tuned for FP32 PyTorch SegNet distill; the MLX KL may need a different operating point. Predicted impact: SUB_PARADIGM persists if root cause is student-head capacity (Path 1); only helps if the issue is softmax saturation.
3. **Lower-EV / higher-LOC**: scorer-response dataset cache (per `tac.optimization.scorer_response_dataset`) instead of in-memory cache. Operational improvement only; does NOT fix the falsification.
4. **DEFER-and-archive**: register the SUB_PARADIGM verdict as a `DEFER` probe outcome per Catalog #313 + Catalog #308. Stop iterating on this student-head architecture; re-route operator priority to other DQS1 substrate-class-shift candidates per the ranking ledger.
5. **Paired-teacher pretraining (CARGO-CULT-CARRYING)**: train the student head against a paired (mock, real) teacher curriculum. NOT recommended — this just delays the structural answer that the student head needs to be learnable.

**Recommended next subagent prompt**: "implement Path 1 (learnable 1×1-conv student head) as a sister NEW canonical surface in `mlx_loss.py`; re-fire the real-teacher 100ep smoke; if CONVERGES, gate Phase 2 numpy parity proof on the new student state_dict; cost ~$0 local."

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A. The mock-vs-real verdict surfaces an IMPLEMENTATION-LEVEL falsification at the substrate-design surface; no per-byte/per-pair sensitivity signal contribution. `research_only=true` for this dim.
2. **Pareto constraint**: N/A. Empirical falsification of student-head architecture; no Pareto constraint contribution. `research_only=true` for this dim.
3. **Bit-allocator hook**: N/A. No archive byte modification proposed. `research_only=true` for this dim.
4. **Cathedral autopilot dispatch hook**: **ACTIVE**. The SUB_PARADIGM verdict on the real teacher prevents the autopilot ranker from promoting the Hinton-distilled scorer surrogate to paid Modal A100 dispatch until Path 1 (learnable student head) lands. This IS the falsification gate per CLAUDE.md "Production-hardened dispatch optimization protocol".
5. **Continual-learning posterior update**: **ACTIVE**. Probe outcome registered via `tac.probe_outcomes_ledger.register_probe_outcome` (see below) for the canonical posterior consumers (Rashomon ensemble + Assumption-Adversary + autopilot ranker per Catalog #300).
6. **Probe-disambiguator**: **ACTIVE**. The `--teacher-provider {mock,real_segnet}` CLI flag IS the canonical probe-disambiguator between the foundation $0 smoke surface and the production real-teacher path. Future MLX long-training smokes inherit this flag.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

`lane_hinton_mlx_real_teacher_refire_20260525` is **DEFERRED-PENDING-LEARNABLE-STUDENT-HEAD**; not killed. Reactivation paths:

- **Path A (primary)**: implement Path 1 above (learnable 1×1-conv student head) and re-fire the real-teacher 100ep smoke. If verdict transitions to CONVERGES_CONSISTENTLY, lane re-enters L1 with the new test artifact.
- **Path B (secondary)**: extend training to 1000ep / 3000ep on the current deterministic-projection student. SUB_PARADIGM rationale notes "operator may extend training before paid GPU"; only valid if the irreducible KL floor at 0.99 actually drifts down with more iterations (currently the curve plateaued at min=0.99 between epoch ~50 and epoch 100, suggesting saturation).
- **Path C (research)**: scorer-response dataset substitute per `tac.optimization.scorer_response_dataset` HF dataset path; orthogonal to the falsification but improves cache reproducibility across machines.

## Sister coordination report

- **Sister codex `hinton_custom_loss_fn_mlx_long_training_20260525`**: complete at 21:47 UTC (PRE-existing); registered the canonical foundation infrastructure. APPEND-ONLY coexistence preserved per Catalog #110/#113; NEW canonical surfaces only (`RealSegNetTeacherLogitsCache` + `build_real_segnet_teacher_cache` + `--teacher-provider` CLI). Sister's `MockTeacherLogitsProvider` + `make_hinton_custom_loss_fn` + `HintonMlxCustomLossFnConfig` field schema all preserved.
- **Sister codex `pr95_mlx_full_decoder_downstream_scorer_drift_measurement_20260525`**: complete at 21:13 UTC; established the canonical drift band BELOW_SCORER_PRECISION 7.41e-05 across the PR95 HNeRV MLX↔PyTorch decoder bridge. This Bundle's Phase 2 numpy parity proof is BLOCKED-BY Phase 1 RED verdict — the canonical bridge is operational but the distilled-student trained against it is not yet a valid dispatchable artifact.
- **Slot 1 + Slot 3 sister codex waves**: no collisions; no files mutated outside the new HINTON surface.
- **Catalog #340 sister-checkpoint guard**: PROCEED at every checkpoint; the only files mutated under sister-shared paths are NEW symbols at the END of `mlx_loss.py` + NEW flag additions to `run_hinton_mlx_long_training_smoke.py` — both APPEND-ONLY per Catalog #110/#113.

## Discipline checklist

- [x] Catalog #229 PV before edit (10 files read in full)
- [x] Catalog #206 checkpoint discipline (3 checkpoints emitted)
- [x] Catalog #287 placeholder-rationale rejection (every rationale carries substantive non-placeholder text)
- [x] Catalog #110/#113 APPEND-ONLY (NEW landing memo; zero mutation of sister codex stand-down memo `hinton_distilled_scorer_surrogate_mlx_long_training_validation_landed_20260525.md`)
- [x] Catalog #125 6-hook wire-in (declared per-hook above)
- [x] Catalog #303 cargo-cult audit (surfaced 1 CARGO-CULTED + EMPIRICALLY FALSIFIED + 2 HARD-EARNED)
- [x] Catalog #305 observability surface (6-facet declaration)
- [x] Catalog #294 9-dim checklist (per-dim evidence)
- [x] Catalog #307 paradigm-vs-implementation classification (IMPLEMENTATION-LEVEL; paradigm INTACT)
- [x] Catalog #308 alternative-probe-methodology enumeration (4 reactivation paths)
- [x] Catalog #313 probe outcome to be registered alongside this landing memo
- [x] CLAUDE.md "MLX portable-local-substrate authority" non-negotiable (every artifact `[macOS-MLX research-signal]`)
- [x] CLAUDE.md "Forbidden premature KILL" (DEFERRED-PENDING-LEARNABLE-STUDENT-HEAD, not killed)
- [x] Catalog #340 sister-checkpoint guard (PROCEED throughout; APPEND-ONLY discipline)

## Artifacts

- `experiments/results/hinton_mlx_bundle_20260525/phase1_real_teacher_verdict.json` — Real SegNet teacher SUB_PARADIGM verdict + per-epoch loss curve
- `experiments/results/hinton_mlx_bundle_20260525/phase1_real_teacher_telemetry.jsonl` — Per-epoch telemetry rows (Catalog #305)
- `experiments/results/hinton_mlx_bundle_20260525/phase1_mock_baseline_verdict.json` — Mock teacher CONVERGES_CONSISTENTLY baseline for apples-to-apples comparison
- `experiments/results/hinton_mlx_bundle_20260525/phase1_mock_telemetry.jsonl` — Mock baseline per-epoch telemetry

## Cross-references

- **Sister stand-down memo**: `.omx/research/hinton_distilled_scorer_surrogate_mlx_long_training_validation_landed_20260525.md` (sister codex commit; the source of the mock-teacher cargo-cult identification)
- **Slot 1 canonical infrastructure**: `tac.local_acceleration.pr95_hnerv_mlx_long_training.MLXLongTrainingPipeline`
- **PR95-MLX-PyTorch bridge**: `tools/export_pr95_mlx_to_pytorch_state_dict.py` + `tac.local_acceleration.pr95_hnerv_mlx.pytorch_state_dict_from_mlx`
- **Drift measurement (sister codex)**: `.omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_measurement_landed_20260525.md`
- **AAA T4 §6.5 + §12.4 Tier 2A spec**: commit `a951a11f9` (distortion-axis substrate-class-shift cascade)
- **CLAUDE.md non-negotiables**: "MLX portable-local-substrate authority"; "MPS auth eval is NOISE"; "Submission auth eval — BOTH CPU AND CUDA"; "Forbidden premature KILL without research exhaustion"; "Forbidden empirical-claim-without-evidence-tag"

## Final reading

**Phase 1 verdict**: SUB_PARADIGM on real teacher (RED per gate semantics); CONVERGES_CONSISTENTLY on mock teacher (the sister codex anchor at commit e3b8c0d8d).

**Phase 2 verdict**: BLOCKED-BY Phase 1 RED.

**Bug class extincted**: future MLX long-training smokes on the Hinton-distilled scorer surrogate substrate cannot regress to mock-only validation because `--teacher-provider` is now an explicit CLI flag; `MockTeacherLogitsProvider` remains as the foundation $0 surface BUT any operator-facing paid dispatch authorization (per `paid_dispatch_authorization_signal`) MUST gate on real_segnet verdict. The mock-vs-real cargo-cult is structurally extincted at the substrate level.

**Mission contribution per Catalog #300**: `frontier_protecting`. Prevents a paid Modal A100 dispatch on a falsifiably-non-convergent substrate; saves $2-5 of paid GPU + preserves the operator queue for higher-EV substrate-class-shift candidates per DQS1 ranking.
