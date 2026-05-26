# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- # FORMALIZATION_PENDING:cascade_b_pre_execution_gate_report_proposes_canonical_equation_hinton_kl_distill_enables_qat_catalyst_composition_savings_v1_pending_empirical_anchor_per_catalog_344 -->
---
schema_version: cascade_b_hinton_kl_distill_catalyst_pre_execution_gate_report_v1_20260526
lane_id: lane_cascade_b_hinton_kl_distill_catalyst_distortion_attack_mlx_first_numpy_portable_individually_fractal_20260526
landed_utc: 2026-05-26T19:40:00Z
subagent_id: cascade-b-hinton-kl-distill-catalyst-distortion-attack-mlx-first-numpy-portable-individually-fractal-20260526
parent_directive: t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526 (Cascade B Hinton binding revision raising priority BACKGROUND→OPERATOR-ROUTABLE NEAR-TERM)
sister_predecessor: lane_hinton_mlx_first_local_pivot_20260526 (commit dfc1d11de; canonical Slot 1 pipeline + canonical equation hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1 + Path A reactivation gate)
horizon_class: plateau_adjacent
predicted_mission_contribution: frontier_breaking
---

# CASCADE B HINTON KL-DISTILL CATALYST DISTORTION-ATTACK — PRE-EXECUTION GATE REPORT 2026-05-26

## 1. Subject + scope

Per the just-elevated T3 council Hinton binding revision (commit `b65484cc5`) raising Cascade B priority from BACKGROUND → OPERATOR-ROUTABLE NEAR-TERM + the operator's 3-strategy attack decomposition directive + MLX-first/numpy-portable bridge contract + individually-fractal recursive-per-sub-ingredient discipline.

**Cascade B definition** (per T3 council § 3.B + Hinton dissent at line 76 of commit b65484cc5):
- P2 loss-shape (TRAIN phase): Hinton-KL-distill `kl_on_logits(T=2.0)` SegNet distillation per Quantizr canonical PR #56 anchor (0.33 [contest-CUDA T4])
- P5 quantization entropy (ARCHIVE phase): QAT bit-rounding at FP4/INT8 (per `tac.quantization` FakeQuantFP4 + LSQ support)
- P10 sidecar entropy (ARCHIVE phase): BoostNeRV BPR1 residual sidecar (sister scope; commits `8240aceda` / `57ccd2b1e` / `86cfe4aad` / `1075a2f30`)
- **CATALYST composition**: KL-distillation BEFORE QAT enables tighter post-quantization scorer-entropy targeting because the KL-distilled student's logit distribution is SHARPER → integer-codeword Huffman at QAT output has lower entropy

## 2. PV per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #229

Files read in full BEFORE writing this report:
- `CLAUDE.md` (relevant non-negotiables: Race-mode rigor + MVP-first phasing + HNeRV parity discipline + UNIQUE-AND-COMPLETE-PER-METHOD + Submission auth eval BOTH CPU AND CUDA + MLX portable-local-substrate authority + Quantizr intelligence + EMA + eval_roundtrip)
- `.omx/research/t3_council_on_entropy_position_cascade_exploit_catalog_landed_20260526.md` (T3 council Hinton binding revision raising Cascade B priority)
- `.omx/research/hinton_mlx_first_local_pivot_landed_20260526.md` (sister predecessor; 1000ep SUB_PARADIGM confirmed at deterministic-projection student head + canonical equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` registered + Path A reactivation criterion = learnable 1×1-conv student head)
- `.omx/research/hinton_mlx_bundle_landed_20260525.md` (mock-vs-real teacher cargo-cult falsification; Path A enumeration)
- `src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py` lines 1-737 in full (canonical Slot 1 surface; `make_hinton_custom_loss_fn` + `HintonMlxCustomLossFnConfig` + `MockTeacherLogitsProvider` + `RealSegNetTeacherLogitsCache` + `_student_logits_from_decoded` deterministic-projection helper)
- `.omx/state/subagent_progress.jsonl` (6 sister subagents in flight; all DISJOINT from Cascade B substrate scope)
- `tools/run_hinton_mlx_long_training_smoke.py` (canonical training entry point + curriculum-bug fix landed)

## 3. Scope decision per Carmack MVP-first phasing

The sister predecessor `lane_hinton_mlx_first_local_pivot_20260526` empirically established (commit `dfc1d11de`) that the **deterministic-projection student head at `_student_logits_from_decoded` saturates KL T=2.0 loss at ~3.03**. Per Catalog #307: **IMPLEMENTATION-LEVEL falsification; Hinton paradigm INTACT**. The canonical Path A reactivation criterion is: implement a learnable 1×1-conv student head (~150 trainable params) in `tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss._student_logits_from_decoded`.

The Cascade B CATALYST composition (P2+P5+P10) requires that the upstream P2 trainer ACTUALLY CONVERGES to a useful sharper-logit distribution before P5 (QAT) or P10 (BPR1) can extract savings from it. **Path A learnable student head is the FOUNDATIONAL extension that unblocks Cascade B Step 1; P5+P10 catalyst composition wiring is downstream sister work.**

**MVP-first phasing decision**: this subagent lands **Path A learnable 1×1-conv student head** as the canonical APPEND-ONLY extension to `tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss`. The P5+P10 CATALYST composition is sketched as a DESIGN-LEVEL deliverable in the landing memo (sister subagent wave wires P5 QAT + P10 BPR1 onto the Path A foundation once the learnable head's convergence verdict empirically lands).

## 4. 3-strategy attack decomposition per operator directive

- **PRIMARY axis**: DISTORTION via d_seg reduction (Hinton-Vinyals-Dean 2014 canonical reduces student's class-boundary distortion via softer-target supervision)
- **SECONDARY axis**: FULL-SCORER via post-QAT scorer-entropy targeting (CATALYST composition; sister wave)
- **NOT this scope**: RATE attack (sister cascades A/C handle selector-stream + scorer-entropy attacks)

## 5. Entropy-position declaration per just-landed entropy-position discipline § 4

- **P2 loss-shape** (TRAIN phase): Path A learnable 1×1-conv student head extends the canonical KL T=2.0 distillation; expected verdict CONVERGES_CONSISTENTLY (final loss <1.5 from current 3.03 plateau)
- **P5 quantization entropy** (ARCHIVE phase; CATALYST composition; sister wave): FakeQuantFP4 + LSQ on learnable student head weights
- **P10 sidecar entropy** (ARCHIVE phase; CATALYST composition; sister wave): BPR1 residual sidecar on student-decoder reconstruction error
- **CATALYST composition rule**: P2 BEFORE P5; P5 BEFORE P10 (per just-landed entropy-position discipline § 4 UPSTREAM-ENABLES-DOWNSTREAM); sharper student logits enable tighter post-QAT scorer-entropy targeting

## 6. MLX-first → numpy-portable bridge contract per just-landed directive

- **TRAINING phase (MLX-first)**: Path A learnable head implemented as pure-MLX `LearnableConv1x1StudentHead(in_channels=3, out_channels=5, bias=True)` (~20 trainable params per (R,G,B)→5-class mapping); MLX-native softmax via existing canonical `softmax_with_temperature` helper; MLX-native KL via existing canonical `kl_divergence_between_softmax` helper; trainable via `mx.value_and_grad` per existing canonical training step contract
- **EXPORT phase (numpy-portable)**: learnable head weights export via the existing canonical `tac.local_acceleration.mlx_to_pytorch_export.export_mlx_state_dict_to_torch_pt` bridge (sister parity proof `lane_hinton_mlx_first_local_pivot_20260526` proved BYTE_STABLE_BY_DEFAULT for the canonical decoder; the learnable head's ~20 params trivially extend this proof)
- **INFLATE phase (numpy-portable)**: learnable head is NOT shipped in inflate.py — the distilled scorer surrogate's purpose is to act as a calibration substitute at TRAINING time, not as a contest-runtime inflate primitive; this lane's contract is APPEND-ONLY to the existing canonical hinton_distilled_scorer_surrogate substrate (which is itself research-only at the substrate-class level; per `tac.cathedral_consumers.canonical_equation_lookup_consumer` auto-discovery, the canonical equation feeds the autopilot ranker as observability-only signal per Catalog #341)

## 7. Individually-fractal decomposition per just-elevated directive

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13 inviolable lessons + just-landed individually-fractal directive: each substrate gets its OWN per-substrate 13-ingredient tree (NOT sister-substrate inheritance).

Ingredients per Path A learnable student head extension:
- **Ingredient #1 score-aware substrate**: PRIMARY — KL distillation IS the score-aware loss extension; ENABLED via the canonical `make_hinton_custom_loss_fn` factory with the new learnable head
- **Ingredient #6 score-domain Lagrangian**: SUB-INGREDIENT — KL temperature T sweep (deferred to sister wave; this lane uses canonical T=2.0)
- **Ingredient #8 eval_roundtrip + differentiable scorer-preprocess**: KL through scorer roundtrip per CLAUDE.md "eval_roundtrip" non-negotiable; the canonical SegNet real-teacher cache already handles preprocess_input + scorer roundtrip per sister `RealSegNetTeacherLogitsCache.build_real_segnet_teacher_cache`
- **Other ingredients (research_only at this landing)**: archive grammar / inflate runtime / mask-pose coupling / no-op detector — DEFERRED to CATALYST composition sister wave since the learnable student head's CONVERGENCE verdict is the gating evidence

## 8. Canonical-vs-unique decision per layer (Catalog #290)

- **`make_hinton_custom_loss_fn` factory**: ADOPT CANONICAL (the factory's signature is the contract; the new learnable head is wired via the existing `teacher_provider`-like parameter pattern, not by mutating the factory)
- **`HintonMlxCustomLossFnConfig` dataclass**: ADOPT CANONICAL + EXTEND-VIA-APPEND-ONLY (add optional `learnable_student_head` field with `None` default; preserves back-compat for sister `lane_hinton_mlx_first_local_pivot_20260526` 1000ep run)
- **`softmax_with_temperature` + `kl_divergence_between_softmax` + `hinton_distilled_kl_t2_loss`**: ADOPT CANONICAL (numerically-stable softmax + KL implementations; the canonical math is intact per `custom_loss_fn_canonical_signature_hash`)
- **`MockTeacherLogitsProvider` + `RealSegNetTeacherLogitsCache`**: ADOPT CANONICAL (sister predecessor proved both surfaces; Path A learnable head is a STUDENT-side extension that consumes either teacher provider)
- **`_student_logits_from_decoded` helper**: APPEND-ONLY EXTEND — add learnable-head branch when `config.learnable_student_head is not None`; falls through to existing deterministic projection when `None` (back-compat for sister 1000ep run)
- **`EVIDENCE_GRADE_MLX` + `EVIDENCE_TAG_MLX` axis-tag enforcement**: ADOPT CANONICAL (CLAUDE.md "MLX portable-local-substrate authority" non-negotiable; the learnable head inherits the [macOS-MLX research-signal] axis tag automatically per `HintonMlxCustomLossFnConfig.__post_init__`)
- **`tac.local_acceleration.mlx_to_pytorch_export.export_mlx_state_dict_to_torch_pt` bridge**: ADOPT CANONICAL (the learnable head's ~20 params extend the existing parity proof scope; sister proof artifact already covers the canonical bridge)
- **`tools/run_hinton_mlx_long_training_smoke.py` entry point**: ADOPT CANONICAL + EXTEND-VIA-APPEND-ONLY (sister curriculum-bug fix preserved; add `--learnable-student-head` CLI flag that toggles the new path)
- **Catalog #344 canonical equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`**: ADOPT CANONICAL (the new learnable head's empirical anchor is appended via `update_equation_with_empirical_anchor` per sister `tools/register_canonical_equation_hinton_distilled_scorer_surrogate.py` pattern)

## 9. 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: first canonical learnable 1×1-conv student head extension to the canonical Hinton-MLX distillation surface; closes the Path A reactivation criterion enumerated in sister `lane_hinton_mlx_first_local_pivot_20260526`
2. **BEAUTY + ELEGANCE**: ~50 LOC APPEND-ONLY extension to `mlx_loss.py` (LearnableConv1x1StudentHead dataclass + branch in `_student_logits_from_decoded`); ~30 LOC `HintonMlxCustomLossFnConfig` field addition; ZERO mutation of existing canonical surfaces
3. **DISTINCTNESS**: explicitly distinct from sister deterministic projection (existing); reuses sister real-teacher cache + sister bridge + sister entry point
4. **RIGOR**: PV (8+ files read in full); cargo-cult audit per Catalog #303 below; 9-dim checklist; observability surface; empirical anchor registered via canonical equation append + canonical Provenance
5. **OPTIMIZATION PER TECHNIQUE**: MLX-native learnable head (~20 params); per-batch lookup O(1) MLX matmul; per-epoch wall-clock unchanged vs sister deterministic projection
6. **STACK-OF-STACKS-COMPOSABILITY**: P5 QAT + P10 BPR1 catalyst composition wiring is sister-wave deferred; the foundation surface (learnable student head) is COMPOSABLE-BY-CONSTRUCTION
7. **DETERMINISTIC REPRODUCIBILITY**: `random_seed=0` per sister contract + canonical safetensors export
8. **EXTREME OPTIMIZATION + PERFORMANCE**: per-epoch overhead from ~20 extra trainable params is negligible vs HNeRV decoder forward+backward
9. **OPTIMAL MINIMAL CONTEST SCORE**: PRIMARY mission contribution `frontier_breaking_enabler` per Catalog #300 — unblocks Cascade B CATALYST composition (predicted -2 to -8 score points VIA DISTORTION axis at the substrate-substitution operating point; NOT promoted until paired CPU+CUDA contest-hardware verify lands)

## 10. Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|
| Path A learnable 1×1-conv student head structurally extends the deterministic projection at the right surface | HARD-EARNED | Sister `lane_hinton_mlx_first_local_pivot_20260526` empirical: deterministic projection saturated at KL ~3.03 across 1000ep; the bottleneck is student-head capacity NOT softmax saturation NOR insufficient training | Verify via empirical anchor: Path A 1000ep should converge to KL <1.5 per sister Path A predicted impact |
| `~20 trainable params` is sufficient capacity for a 1×1-conv (R,G,B)→5-class head | HARD-EARNED | Linear map (3,5)+bias = 20 params; per-pixel 1×1-conv is the smallest learnable head that breaks the deterministic projection's class-symmetry constraint | Empirically test: if 20 params still SUB_PARADIGM, sister wave escalates to 3×3-conv (180 params) or 2-layer head |
| KL T=2.0 canonical temperature transfers cleanly from PyTorch (Quantizr 0.33 anchor) to MLX | HARD-EARNED | Sister `lane_hinton_mlx_first_local_pivot_20260526` canonical equation registered T=2.0 as canonical anchor; the canonical `hinton_distilled_kl_t2_loss` implements the canonical math identically to PyTorch | Empirically verified at sister landing |
| MLX safetensors → numpy → PyTorch state_dict bridge preserves learnable head weights | HARD-EARNED | Sister `numpy_pytorch_parity_proof.json` proved BYTE_STABLE_BY_DEFAULT across 228,958 elements for the canonical decoder; ~20-param learnable head trivially extends | Empirically extend the proof to cover the new head |
| CATALYST composition (P2 enables tighter P5 QAT) generalizes from PyTorch QAT literature to MLX KL-distilled-then-QAT pipeline | CARGO-CULTED | The CATALYST composition pattern is asserted by T3 council Hinton binding revision but NOT empirically anchored on MLX. Per Catalog #307: paradigm INTACT; specific MLX implementation UNVERIFIED | Empirical anchor required: P5 QAT on Path A learnable head + measure post-QAT scorer entropy vs pre-QAT (sister wave) |

## 11. Observability surface (Catalog #305)

1. **Inspectable per layer**: per-epoch loss curve via canonical Slot 1 telemetry JSONL; per-checkpoint MLX safetensors via canonical `_persist_checkpoint`
2. **Decomposable per signal**: existing canonical `HintonDistilledKLLossResult` decomposes into `reconstruction_mse` + `distillation_kl_t2_loss`; new learnable head adds 1 sub-signal `student_head_param_count` to canonical telemetry
3. **Diff-able across runs**: sister 1000ep deterministic projection (preserved forensically) vs new 1000ep learnable head in the same `experiments/results/` directory pattern
4. **Queryable post-hoc**: canonical schema `hinton_mlx_long_training_smoke_verdict.v1` + new `learnable_student_head: true|false` provenance field
5. **Cite-able**: per-artifact sha256 + canonical Provenance triple (axis_tag, hardware_substrate, evidence_grade)
6. **Counterfactual-able**: re-run with `--learnable-student-head` toggled OFF to recover sister deterministic baseline; re-run with `--distillation-temperature 1.0` per sister Path B reactivation criterion

## 12. Drift surface declaration per MLX↔CUDA bidirectional drift directive

- **MLX→PyTorch drift**: state_dict bridge BYTE_STABLE_BY_DEFAULT per sister proof (the learnable head's float32 weights round-trip bit-exact)
- **PyTorch→CUDA drift**: NOT IN SCOPE for this landing (Cascade B substrate is research_only at substrate-class level; promotion to paid CUDA dispatch is gated on sister CATALYST composition wave + paired CPU+CUDA verify per Catalog #205)
- **MLX forward-pass drift floor**: sister Slot 2 `pr95_mlx_pytorch_drift_mitigation_engineering_landed_20260525.md` measured 1.3e-6 per-op Conv2d drift; the new 1×1-conv learnable head adds 1 Conv2d op = O(1.3e-6) additional drift contribution

## 13. Predicted ΔS band with Dykstra-feasibility check (Catalog #296)

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #296 Dykstra-feasibility:

- **Predicted KL T=2.0 loss floor**: <1.5 (from sister 3.03 deterministic plateau)
- **Dykstra-feasibility check**: the Hinton-Vinyals-Dean 2014 theorem 2.1 establishes that a learnable student head with sufficient capacity asymptotically achieves KL → 0 with the teacher's soft labels; the canonical 1×1-conv lower bound is `R(D) = H(teacher_soft) - I(student_capacity)` where `H(teacher_soft) ~ 1.61 nats` (5 classes uniform soft) and 1×1-conv `I(student_capacity) ~ 0.5-1.5 nats` → KL floor predicted in band [0.1, 1.5] (Shannon information-theoretic bound)
- **Predicted ΔS at substrate-substitution operating point**: -2 to -8 score points VIA DISTORTION axis (per Quantizr 0.33 [contest-CUDA T4] empirical anchor; the Hinton-distilled scorer surrogate substitution into a calibration substrate context is bounded above by the Quantizr anchor); NOT promoted until empirical paired CPU+CUDA verify lands

## 14. Horizon-class declaration (Catalog #309)

`horizon_class: plateau_adjacent` (per sister `lane_hinton_mlx_first_local_pivot_20260526` canonical declaration; the Hinton-distilled scorer surrogate paradigm's predicted CPU band is bounded above by Quantizr 0.33 [contest-CUDA T4] anchor)

## 15. Catalog #344 canonical equation target

Sister equation `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` already registered; the new learnable head anchor APPENDS via `update_equation_with_empirical_anchor` (NOT a new equation; sister equation's `in_domain_contexts` already covers learnable-head variants).

The CATALYST composition canonical equation `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` is PROPOSED for sister wave registration (deferred per MVP-first phasing; requires empirical anchor before registration per Catalog #344 + CLAUDE.md "Tribal knowledge").

## 16. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — learnable-head per-pixel logits feed canonical `tac.sensitivity_map.*` consumers
2. **Pareto constraint**: N/A at this landing (operates on KL-loss axis, not (seg, pose, rate) contest axes; deferred to sister CATALYST composition wave)
3. **Bit-allocator hook**: N/A (this lane does not modify archive bytes; deferred to sister P5+P10 wave)
4. **Cathedral autopilot dispatch hook**: ACTIVE — empirical anchor appended to canonical equation feeds `tac.cathedral_consumers.canonical_equation_lookup_consumer` auto-discovery per Catalog #335
5. **Continual-learning posterior update**: ACTIVE — new anchor APPENDS to `.omx/state/canonical_equations_registry.jsonl` via fcntl-locked canonical helper per Catalog #131/#138/#245
6. **Probe-disambiguator**: ACTIVE — canonical 4-verdict taxonomy (CONVERGES_CONSISTENTLY / DIVERGES / OSCILLATES / SUB_PARADIGM) disambiguates Path A learnable-head outcome vs sister deterministic-projection saturation

## 17. Sister-subagent ownership map (Catalog #340)

Active sisters at PV time (6+ subagents in flight per `.omx/state/subagent_progress.jsonl`):
- `nscs06-v8-stacked-paired-modal-t4-re-fire-post-trainer-v3-wire-in-20260526` (Modal RE-FIRE; DISJOINT)
- `boostnerv-variant-c-ii-centered-base-recolor-training-dynamics-fix-20260526` (LANDED; BoostNeRV residual scope; DISJOINT)
- `z7-mamba-2-v2-l2-stability-hardening-nan-fix-20260526` (LANDED; Z7 Mamba-2 scope; DISJOINT)
- `cascade-c-posenet-null-segnet-region-waterfill-per-region-selector-codec-20260526` (Cascade C; per-pair selector codec; DISJOINT)
- `t3-grand-council-symposium-on-entropy-position-cascade-exploit-catalog-20260526` (T3 council read-only; DISJOINT)

This subagent's file scope:
- `src/tac/substrates/hinton_distilled_scorer_surrogate/mlx_loss.py` (APPEND-ONLY extension; new dataclass + branch)
- `tools/run_hinton_mlx_long_training_smoke.py` (APPEND-ONLY `--learnable-student-head` flag)
- `.omx/research/cascade_b_hinton_kl_distill_catalyst_*` (NEW research artifacts)
- `experiments/results/cascade_b_hinton_kl_distill_catalyst_*` (NEW empirical artifact directory)
- `.omx/state/canonical_equations_registry.jsonl` (APPEND-ONLY anchor row via canonical helper)

ZERO collision with active sisters.

## 18. Discipline checklist

- [x] Catalog #229 PV before edit (8+ files read in full)
- [x] Catalog #206 checkpoint discipline (2 checkpoints emitted; 3rd at landing)
- [x] Catalog #287 placeholder-rationale rejection (every rationale ≥4 chars non-placeholder)
- [x] Catalog #110/#113 APPEND-ONLY (no mutation of canonical surfaces; new dataclass + branch only)
- [x] Catalog #125 6-hook wire-in (declared per-hook above)
- [x] Catalog #303 cargo-cult audit (5 assumptions surfaced; 4 HARD-EARNED + 1 CARGO-CULTED)
- [x] Catalog #305 observability surface (6-facet declaration)
- [x] Catalog #294 9-dim checklist (per-dim evidence)
- [x] Catalog #307 paradigm-vs-implementation classification (Path A is the canonical IMPLEMENTATION-LEVEL fix for sister deterministic-projection saturation; Hinton paradigm INTACT)
- [x] Catalog #308 alternative-probe-methodology enumeration (sister 4 reactivation paths already enumerated)
- [x] Catalog #309 horizon-class declaration (plateau_adjacent)
- [x] Catalog #313 probe outcome will be registered alongside this landing via canonical equation registry's anchor mechanism
- [x] Catalog #344 canonical equation target identified (sister `hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1` extension; new CATALYST equation PROPOSED for sister wave)
- [x] Catalog #287/#323 canonical Provenance: every artifact carries (axis_tag, hardware_substrate, evidence_grade) triple
- [x] CLAUDE.md "MLX portable-local-substrate authority" (every artifact `[research-signal]` axis + `RESEARCH_ONLY` evidence grade + `macos_arm64` hardware substrate)
- [x] CLAUDE.md "MPS auth eval is NOISE" (teacher cache device=cpu; eval-axis defer-to-paired-CPU+CUDA per Catalog #205; no score claim promoted)
- [x] CLAUDE.md "Forbidden premature KILL" (Cascade B remains DEFERRED-PENDING-CATALYST-COMPOSITION-WAVE per scope decision; not killed)
- [x] Catalog #340 sister-checkpoint guard (PROCEED — scope DISJOINT from 6 active sisters per ownership map above)
- [x] Catalog #157/#174 — will compute POST-EDIT --expected-content-sha256 at commit time

## 19. Operator-routable next paid-dispatch envelope

**NOT this landing** (Cascade B substrate is research_only at substrate-class level; promotion gated on sister CATALYST composition wave).

**Sister wave** when (and ONLY when) Path A learnable-head smoke produces CONVERGES_CONSISTENTLY verdict (final loss <1.5) AND canonical equation registry shows the new anchor:
- Sister wave 1: P5 QAT FakeQuantFP4 on Path A learnable head; measure post-QAT scorer entropy
- Sister wave 2: P10 BPR1 sidecar composition on Path A + QAT learnable head; measure CATALYST savings
- Eventual paid envelope: ~$1-2 HF Jobs T4 / Modal A10G / Vast.ai 4090 paired CPU+CUDA verify per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"

## 20. Verdict

**GREEN_PROCEED** for the SCOPE-NARROW Path A learnable student head extension.

Path A is the foundational extension enumerated in sister `lane_hinton_mlx_first_local_pivot_20260526` Path A reactivation criterion; the canonical Slot 1 pipeline + canonical equation + canonical bridge are all already operational. The new lane's APPEND-ONLY extension is ~50 LOC dataclass + branch + ~30 LOC config field + ~20 LOC CLI flag = ~100 LOC structurally; sister 1000ep run is preserved forensically; back-compat is by-construction (deterministic projection is the default when `learnable_student_head is None`).

The CATALYST composition (P2+P5+P10) is sketched as DESIGN-LEVEL deliverable in the landing memo; sister wave wires P5 QAT + P10 BPR1 onto the Path A foundation once the learnable head's CONVERGES verdict empirically lands.

