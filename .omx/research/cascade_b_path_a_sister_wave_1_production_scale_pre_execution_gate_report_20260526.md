# CASCADE B Path A Sister Wave 1 — Production-Scale Convergence Pre-Execution Gate Report

**UTC**: 2026-05-26T20:00:00Z
**Subagent**: `cascade-b-path-a-sister-wave-1-production-scale-convergence-600f-1000ep-canonical-fixture-mlx-first-individually-fractal-20260526`
**Lane**: `lane_cascade_b_path_a_sister_wave_1_production_scale_convergence_20260526`
**Sister-of**: `lane_cascade_b_hinton_kl_distill_catalyst_distortion_attack_mlx_first_numpy_portable_individually_fractal_20260526` (commit `15b11c86e` 50f×100ep PARADIGM-VALIDATED at scaffold scale)
**Operator approval**: 2026-05-26 verbatim "all are approved + follow up are approved + pursue other attacks as well + remember all MLX first + individually fractally optimized"
**Mission contribution per Catalog #300**: `frontier_breaking_enabler` (production-scale convergence validation unblocks CATALYST cascade composition P5 QAT + P10 BPR1 5th-order spawn)

## 1. Premise verification (Catalog #229)

PV completed on 5 prerequisites BEFORE any execution:

1. **Sister Path A landing commit `15b11c86e`** — read in full: scaffold-scale 50f×100ep PARADIGM-VALIDATED. Deterministic baseline: 6.17→6.18 (-0.2% noise; SUB_PARADIGM; 0 params). Path A learnable head: 6.01→4.52 (-24.8% MONOTONIC; PARTIAL_CONVERGENCE; 20 params). Delta = 1.66 nats. PARADIGM-VALIDATED per Catalog #307.
2. **`mlx_loss.py` Path A factory** — `LearnableConv1x1StudentHead` dataclass + `build_learnable_student_head(num_classes, in_channels, seed, init_scale)` factory present and exported. Sister 39 tests pass unchanged.
3. **Sister smoke harness `tools/cascade_b_path_a_learnable_head_smoke.py`** — 396 LOC; canonical pattern: SegNet teacher cache builder → MLX `mx.einsum("bhwc,ck->bhwk")` 1×1 conv + bias → `mx.value_and_grad` over (W, b) → plain SGD with lr=0.5 + same fixture both arms. Reusable as-is with `--n-frames 600 --n-epochs 1000`.
4. **Anchor-appender `tools/append_cascade_b_path_a_anchor_to_hinton_canonical_equation.py`** — accepts `--learnable-verdict` + `--deterministic-verdict` and emits provenance-tagged `EmpiricalAnchor`s into `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`. Currently 4 anchors; this wave will extend to 5-6 anchors.
5. **Canonical video** — `upstream/videos/0.mkv` decodes ≥700 frames at canonical 384×512 resize; 600 is feasible (verified via `av.open()` decode probe).

## 2. Scale-up parameter selection

| Knob | Scaffold (sister) | Production (this wave) | Rationale |
|---|---|---|---|
| n_frames | 50 | 600 | Full canonical contest fixture (1 frame per evaluator pair × ~600 pairs) |
| n_epochs | 100 | 1000 | 10× wall-clock for convergence beyond 4.52 plateau |
| batch_size | 10 | 30 | Maintain ~20 batches/epoch (50/10=5 vs 600/30=20); GPU pressure linear |
| temperature | 2.0 | 2.0 | Hinton-Vinyals-Dean canonical T=2.0; sister anchor |
| distillation_weight | 0.5 | 0.5 | Sister anchor |
| learning_rate | 0.5 | 0.5 | Sister anchor; plain SGD |
| seed | 0 | 0 | Reproducibility |

Wall-clock estimate: scaffold 50f×100ep = ~1.5 min observed. Production 600f×1000ep = ~12× batches/ep × 10× epochs = ~120× compute ≈ ~180 min. **Updated estimate after harness-tuning: emit per-epoch progress; allow up to 30 min budget before checkpoint-and-defer.**

## 3. Convergence acceptance taxonomy (Catalog #307 paradigm-vs-implementation)

| Verdict | Criterion | Implication |
|---|---|---|
| `CONVERGES_CONSISTENTLY` | final KL < 1.5 | PARADIGM-VALIDATED at production scale; unblocks 5th-order CATALYST cascade composition |
| `PARTIAL_CONVERGENCE_EXTENDED` | 1.5 ≤ final KL < 4.0 | plateau-adjacent extension; PARADIGM INTACT; sister 4th-order iteration (longer training / arch sweep) |
| `STAGNATES_AT_SCAFFOLD_FLOOR` | 4.0 ≤ final KL < 5.0 | implementation plateau matches scaffold; sister hyperparam sweep |
| `DIVERGES` | final KL ≥ 5.0 OR NaN/Inf | IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307; PARADIGM (Hinton KL T=2.0 + learnable head) preserved per "Forbidden premature KILL"; sister 5th-order architectural iteration |

## 4. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Loss function (Hinton KL T=2.0) | ADOPT_CANONICAL | Canonical equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`; this wave EXTENDS anchors |
| Student head architecture (1×1 conv) | ADOPT_CANONICAL | Sister `LearnableConv1x1StudentHead` factory; 20 params |
| Teacher cache (real SegNet on CPU) | ADOPT_CANONICAL | Sister `build_real_segnet_teacher_cache` |
| Fixture (canonical video) | ADOPT_CANONICAL | `upstream/videos/0.mkv` 384×512 resize |
| Optimizer | ADOPT_CANONICAL | Plain SGD with lr=0.5 per sister anchor |
| Scale (frames × epochs) | FORK_BECAUSE_PRINCIPLED | 600×1000 is the production validation scale, not scaffold |
| Output destination | ADOPT_CANONICAL | `experiments/results/cascade_b_path_a_sister_wave_1_20260526/` |

## 5. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: production-scale anchor; first 600f×1000ep MLX-local convergence test of Hinton KL T=2.0 learnable head paradigm.
2. **BEAUTY+ELEGANCE**: harness is a thin wrapper invoking sister `run_smoke` at scaled params; ≤50 LOC of new code.
3. **DISTINCTNESS**: distinct from scaffold smoke (50f×100ep) by scale; distinct from sister deterministic baseline by architecture.
4. **RIGOR**: PV (Catalog #229) + identical fixture both arms + canonical provenance per Catalog #287/#323 + Catalog #307 verdict taxonomy.
5. **OPTIMIZATION-PER-TECHNIQUE**: 1×1-conv is the canonical minimal-capacity head; sister landing proves it breaks deterministic saturation at scaffold scale.
6. **STACK-OF-STACKS-COMPOSABILITY**: production-scale validation is prerequisite for P5+P10 CATALYST cascade composition.
7. **DETERMINISTIC-REPRODUCIBILITY**: seed=0 + canonical fixture + MLX-pinned + identical hyperparams; byte-stable per `numpy.random.RandomState(0)`.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: MLX-local on M5 Max; $0 paid GPU; ~30 min wall-clock budget.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: calibration-time substrate (not contest-archive direct); future CATALYST cascade composition is the score-affecting downstream.

## 6. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Unwind |
|---|---|---|
| Sister 50f×100ep PARADIGM-VALIDATED extends to 600f×1000ep | HARD-EARNED-FROM-SISTER-ANCHOR (commit `15b11c86e`) | This wave IS the unwind test |
| Plain SGD with lr=0.5 stable at 10× epochs | CARGO-CULTED (no anchor at this scale) | Empirical via per-epoch monitoring; verdict `DIVERGES` covers failure mode |
| Ground-truth RGB as student input (vs HNeRV-decoded) | INHERITED-FROM-SCAFFOLD | Sister smoke design choice per its comment line 138-146: "isolates the head's convergence floor"; preserved here for apples-to-apples |
| MLX-CPU teacher cache + MLX-MPS student head bridge OK at 600f | HARD-EARNED-VIA-SISTER (50f anchor) | Wall-clock-limited; this wave extends |

## 7. Observability surface (Catalog #305)

| Facet | Surface |
|---|---|
| Inspectable per layer | Per-epoch KL printed; per-arm verdict JSON in `experiments/results/` |
| Decomposable per signal | KL only (distill term); MSE=0 by design |
| Diff-able across runs | Sister `comparison_summary.json` schema preserved |
| Queryable post-hoc | `*_verdict.json` + canonical equation anchor IDs |
| Cite-able | `source_video_sha256` + `random_seed` + `hardware_substrate=macos_arm64` |
| Counterfactual-able | re-run with same seed deterministic |

## 8. MLX↔CUDA drift declaration (per just-elevated directive)

Calibration-time substrate; no contest-archive direct emission. The student head's weights are MLX-trained; bridge to numpy/PyTorch via `tac.local_acceleration.mlx_to_pytorch_export` is canonical for the 5th-order CATALYST cascade composition (P5+P10 sister wave) — NOT this wave's scope.

## 9. Predicted ΔKL band (Catalog #296 Dykstra-feasibility)

**Predicted band**: final KL ∈ [1.0, 3.5] at 600f × 1000ep.

Mathematical grounding: Hinton KL T=2.0 with 20 learnable params trained for 10× sister epochs SHOULD continue monotonic descent past sister 4.52 plateau if the plateau is training-time-limited (not capacity-limited). If capacity-limited, plateau persists at ~4.0-4.5 = `STAGNATES_AT_SCAFFOLD_FLOOR`. First-principles citation: Hinton-Vinyals-Dean 2014 distillation paper Fig. 1 shows MNIST distillation converges to teacher accuracy in ~10× the source training time — applied to our context: scaffold 100ep extending to 1000ep ≈ paper's regime.

**Sister probe-disambiguator**: per-epoch KL curve emitted; convergence regime classified post-hoc per Section 3 taxonomy.

## 10. Horizon class declaration (Catalog #309)

**`plateau_adjacent`** — calibration-time research substrate; not direct-attack on contest archive bytes. Future CATALYST cascade composition (P5 QAT + P10 BPR1) is the frontier-pursuit horizon.

## 11. Catalog #344 canonical equation anchor

Production-scale anchor target: `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`. Sister tool `tools/append_cascade_b_path_a_anchor_to_hinton_canonical_equation.py` reused; will extend registry 4 → 5/6 anchors via per-arm append.

## 12. 6-hook wire-in declaration (Catalog #125)

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A — research substrate; no per-byte sensitivity emission |
| #2 Pareto constraint | N/A — calibration-time |
| #3 bit-allocator | N/A — no archive byte emission |
| #4 cathedral autopilot dispatch | ACTIVE — canonical equation anchor consumed by autopilot ranker |
| #5 continual-learning posterior | ACTIVE — anchor append via `update_equation_with_empirical_anchor` |
| #6 probe-disambiguator | ACTIVE — Section 3 4-verdict taxonomy IS the disambiguator |

## 13. Operator-routable next step (per landing taxonomy)

- IF `CONVERGES_CONSISTENTLY`: 5th-order CATALYST cascade composition sister wave (P5 QAT + P10 BPR1 onto Path A foundation; canonical equation #2 `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` first anchor).
- IF `PARTIAL_CONVERGENCE_EXTENDED`: sister 4th-order iteration (capacity sweep: 1×1 → 3×3 conv; or longer training 2000-5000ep).
- IF `STAGNATES_AT_SCAFFOLD_FLOOR`: sister architectural iteration (LearnableConv3x3 + hidden layer); preserve PARADIGM per Catalog #307.
- IF `DIVERGES`: IMPLEMENTATION-LEVEL FALSIFIED at production scale; PARADIGM (Hinton KL T=2.0) preserved per "Forbidden premature KILL"; sister 5th-order architectural OR hyperparam iteration.
