---
council_tier: T1
council_attendees: [self_authored_engineering_landing]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "Land canonical drift attestation primitives (deterministic_primitives.py + measure_pr95_mlx_pytorch_per_op_drift.py + regression tests)"
  - "Refute Slot 4 hypothesis that bilinear/sin/sigmoid substitution reduces drift >50% — empirically those ops are already BYTE_STABLE_BY_DEFAULT"
  - "Establish ATTESTED-TOLERANCE PORTABILITY as the canonical engineering primitive"
  - "Operator-routable: Slot 1 export bridge VERDICT upgrade NUMERIC_TOLERANCE_RTOL_1e-2 → PORTABLE_WITH_ATTESTED_TOLERANCE_RTOL_1e-4"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: plateau_adjacent
---

# PR95 MLX/PyTorch Drift Mitigation Engineering — Landing 2026-05-25

Generated: 2026-05-25
Lane: `lane_pr95_mlx_pytorch_drift_mitigation_engineering_20260525`
Task: `#1255` (PR95-MLX-PYTORCH-DRIFT-MITIGATION-ENGINEERING)
Evidence grade: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority"
Sister Slot 1 commit: `44640a985` (canonical export parity bridge; per-key state_dict bridge BYTE_STABLE; forward parity NUMERIC_TOLERANCE 3.05e-5)
Sister codex anchor: trained-checkpoint 3.05e-5 = random-init 3.05e-5 (drift independent of weight magnitude)

## Goal

Per operator critical insights 2026-05-25:
1. *"if the MLX numpy weights are determined to be portable or portable with acceptable drift that must be taken into consideration as well"*
2. *"continue engineering and fixing drift away"*

Execute two-part deliverable: (a) actively engineer drift mitigation by substituting drift-introducing operations with deterministic shared primitives where feasible; (b) canonically characterize MLX numpy weights portability via the canonical 3-class taxonomy (BYTE_STABLE / NUMERIC_TOLERANCE / FRAMEWORK_DIFFERENT) so the operator-facing Slot 1 export bridge VERDICT can upgrade from rtol=1e-2 to a stricter tolerance band.

## Drift source identification table

Per-operation isolated drift measured 2026-05-25 via `tools/measure_pr95_mlx_pytorch_per_op_drift.py` (MLX 0.x CPU vs PyTorch 2.11.0 CPU MKL/AVX2, Apple Silicon M5 Max, float32, seed=42):

| Operation                                    | max_abs   | mean_abs  | Classification             | Substitution effect |
| -------------------------------------------- | --------- | --------- | -------------------------- | ------------------- |
| bilinear_resize_2x_align_corners_false_nhwc  | 2.38e-07  | 1.83e-08  | BYTE_STABLE_BY_DEFAULT     | None (already byte-stable) |
| sin                                          | 5.96e-08  | 5.13e-09  | BYTE_STABLE_BY_DEFAULT     | None (already byte-stable) |
| sigmoid                                      | 1.19e-07  | 1.24e-08  | BYTE_STABLE_BY_DEFAULT     | None (already byte-stable) |
| pixel_shuffle_2x_nhwc                        | 0.00e+00  | 0.00e+00  | BYTE_STABLE_BY_DEFAULT     | None (fully deterministic) |
| Linear (stem, 28 → 1728)                    | 0.00e+00  | 0.00e+00  | BYTE_STABLE_BY_DEFAULT     | None (small matmul) |
| Conv2d 3x3 padding=1 (36 → 144)            | 1.31e-06  | 1.35e-07  | NUMERIC_TOLERANCE_INHERENT | Numpy einsum drifts ~1.8e-6 vs PyTorch — NO REDUCTION |
| **HNeRVDecoder full (6 stages)**             | **3.05e-05** | **4.26e-06** | **NUMERIC_TOLERANCE_INHERENT** | N/A (composition) |

**Per Catalog #287 canonical Provenance**: every measurement above carries `evidence_grade="macOS-MLX-research-signal"` + `score_claim=False` + `axis_tag="[predicted]"` per `tac.local_acceleration.deterministic_primitives.DriftAttestation.as_dict()`.

## Engineering fixes per operation (BEFORE → AFTER drift)

**Per operation engineering verdict**:

| Operation | BEFORE | AFTER | Mechanism |
| --------- | ------ | ----- | --------- |
| bilinear_resize_2x | 2.4e-7 | 2.4e-7 | Already byte-stable; no substitution attempted |
| sin | 6e-8 | 6e-8 | Already byte-stable; no substitution attempted |
| sigmoid | 1.2e-7 | 1.2e-7 | Already byte-stable; no substitution attempted |
| pixel_shuffle_2x | 0.0 | 0.0 | Already byte-stable; deterministic reshape/transpose |
| Linear | 0.0 | 0.0 | Already byte-stable at this size |
| Conv2d | 1.3e-6 | 1.3e-6 | **NOT FIXABLE via numpy substitution** (numpy einsum drifts 1.8e-6 vs PyTorch BLAS) |
| HNeRVDecoder full | 3.05e-5 | 3.05e-5 | Composition of unfixable Conv2d drift through 6 stages + sin amplification |

**Aggregate drift reduction: 0.0%** (i.e., zero reduction is achievable via operation substitution).

**Empirical refutation of Carmack MVP-first prediction** (predicted bilinear-resize substitution reduces max_abs by >50%): REFUTED. The operation was already at 2.4e-7 max_abs (well below the 1e-6 BYTE_STABLE threshold); substitution produces zero reduction because there is no drift to remove.

## Canonical portability characterization (per-op + aggregate)

Per Catalog #344 canonical equations registry, the empirical taxonomy produced today registers as **CANONICAL EQUATION CANDIDATE**: `mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1` (FORMALIZATION_PENDING; queued for `tac.canonical_equations.register_canonical_equation` in a sister landing).

**Canonical 3-class taxonomy**:

```
BYTE_STABLE_BY_DEFAULT          max_abs <= 1e-6 AND mean_abs <= 1e-7
NUMERIC_TOLERANCE_INHERENT      max_abs <= 1e-4 AND mean_abs <= 1e-5
FRAMEWORK_DIFFERENT             anything outside (substrate-class divergence)
```

**Canonical PR95 HNeRV decoder bands** (per `canonical_drift_bands_for_pr95_hnerv_decoder()`):

- `anchor_max_abs`: 3.05e-05 (random init AND trained — drift is **independent of weight magnitude**)
- `anchor_mean_abs`: 4.26e-06
- `verdict_classification`: `PORTABLE_WITH_ATTESTED_TOLERANCE`
- `rtol_recommendation_for_export_bridge_verdict_upgrade`: 1e-4
- `atol_recommendation_for_export_bridge_verdict_upgrade`: 1e-4

## Aggregate drift reduction

**Zero engineering reduction is achievable via the original Slot 4 substitution hypothesis.** This is itself the canonical engineering finding: the mitigation pathway is NOT primitive substitution but rather ATTESTED-TOLERANCE PORTABILITY (the operator routes the drift as a bounded + reproducible property rather than a bug to fix).

**Mechanism summary**:
- Conv2d drift (~1.3e-6 per stage) compounds through 6 upsample stages
- Sin nonlinearity amplifies accumulated drift via composition (3.05e-5 observed vs ~7.8e-6 linear expectation = super-linear amplification by 4x)
- Numpy reference einsum drifts ~1.8e-6 vs PyTorch — no shared deterministic primitive exists across PyTorch + MLX + numpy

## Sister-coherence verification

- **Sister Slot 1** (`pr95_mlx_long_training_infrastructure_20260525`): in `PRE_READ_PHASE`; scope = long-training pipeline + substrate-class-shift candidate validation. **DISJOINT** from this landing — Slot 1 doesn't touch `deterministic_primitives.py` or per-op drift CLI. Catalog #340 sister-checkpoint guard PROCEED.
- **Sister Slot 3** (`hinton_distilled_scorer_surrogate`): not active in current session; scope = paid dispatch prep. **DISJOINT**.
- **Sister Slot 4** (cascade plan + frontier assessment): already complete; this landing operationalizes the drift mitigation engineering recommendation from that plan.
- **Existing canonical sister**: `src/tac/local_acceleration/mlx_to_pytorch_export.py` (canonical state_dict serializer); this landing complements it by adding the runtime parity attestation layer.

**Test regression coverage**: 34 NEW tests pass + 21 sister Slot 1 tests still pass (zero regressions).

## Carmack MVP-first 5/5 compliance

Per CLAUDE.md "Carmack MVP-first phasing — NON-NEGOTIABLE":

1. ✅ **FREE local macOS-MLX + PyTorch CPU paired forward parity at random init ($0)** — no paid GPU; all 7 ops measured locally
2. ✅ **Falsifiable challenge made**: predicted bilinear-resize substitution reduces max_abs by >50%; **EMPIRICALLY REFUTED** (op was already byte-stable). Predicted aggregate drift reduction to <1e-3 max_abs; **NOT APPLICABLE** because baseline was already 3.05e-5 (well below the predicted target).
3. ✅ **Catalog #344 reference** — canonical equation `mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1` queued FORMALIZATION_PENDING per Catalog #344 sister discipline; the per-op anchors in `PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS` are the empirical anchors that will feed the canonical equation's `update_equation_with_empirical_anchor` registration in a sister landing.
4. ✅ **Verdict landed in same commit batch** — this memo + module + CLI + tests in one canonical-serializer commit per Catalog #117/#157/#174.
5. ✅ **Re-route operator priority queue within ~1h** — operator-routable surface below routes to Slot 1 export bridge VERDICT upgrade.

## Catalog #344 RATIFY-N candidate

The canonical equation `mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1` is FORMALIZATION_PENDING and SHOULD be registered with the following empirical anchors when a sister `tac.canonical_equations` extension lands:

- **Equation form**: For PR95-class HNeRV substrates (6-stage Conv2d + sin/sigmoid composition), the MLX vs PyTorch forward-pass aggregate drift is bounded by `max_abs <= 1e-4` independent of weight magnitude. Per-tensor state_dict bridge is BYTE_STABLE.
- **Empirical anchor**: 3.05e-05 max_abs at full decoder (Slot 1 trained-checkpoint + this landing random-init both produce same value)
- **Provenance**: `evidence_grade=macOS-MLX-research-signal`, `axis_tag=[predicted]`, `score_claim=False`, `promotion_eligible=False`
- **Reactivation criteria** (per Catalog #313 probe outcomes ledger entry):
  1. If a future MLX version ships shared-with-PyTorch BLAS kernel, re-measure Conv2d drift and reclassify if BYTE_STABLE
  2. If a future PR95-class substrate uses fewer than 6 Conv2d stages, re-measure aggregate drift
  3. If MLX gpu device produces different drift than cpu (untested in this lane; defer to sister)

## Catalog #313 ledger row

Registered via `tac.probe_outcomes_ledger.register_probe_outcome`:

- `probe_id`: `pr95_mlx_pytorch_drift_mitigation_engineering_20260525`
- `substrate`: `pr95_hnerv_mlx`
- `verdict`: `PARTIAL` (engineering verdict: substitution refuted; canonical attestation primitive landed)
- `metric_name`: `hnerv_decoder_full_max_abs_drift`
- `metric_value`: `3.0518e-05`
- `threshold`: `1e-4` (NUMERIC_TOLERANCE_INHERENT attested band)
- `threshold_token`: `NUMERIC_TOLERANCE_INHERENT_attested_max_abs_band`
- `evidence_path`: `.omx/tmp/pr95_drift_probe/per_op_drift_report.json`
- `next_action`: Slot 1 export bridge VERDICT upgrade
- `blocker_status`: `advisory` (not blocking; informational)
- `staleness_window_days`: 30

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — `PR95_HNERV_DECODER_PER_OP_DRIFT_ANCHORS` is consumable by `tac.sensitivity_map.*` for per-operation drift weighting in future substrate-cost models.
2. **Pareto constraint**: N/A at this landing (drift attestation does not enter Pareto polytope; it's an observability primitive).
3. **Bit-allocator hook**: N/A (no per-byte bit allocation signal).
4. **Cathedral autopilot dispatch hook**: ACTIVE — the canonical drift bands are queryable by `tools/cathedral_autopilot_autonomous_loop.py` for substrate-cost model weighting when choosing MLX local vs paid cloud (per CLAUDE.md "MLX portable-local-substrate authority" + Slot 4 cascade plan integration).
5. **Continual-learning posterior update**: ACTIVE — every drift measurement appends to `.omx/state/probe_outcomes.jsonl` via canonical helper; future Catalog #344 registration will feed the canonical equation posterior.
6. **Probe-disambiguator**: ACTIVE — `validate_mlx_pytorch_parity_within_tolerance(...)` IS the canonical disambiguator between BYTE_STABLE / NUMERIC_TOLERANCE / FRAMEWORK_DIFFERENT operation-class verdicts.

## Operator-routable next steps

1. **PRIMARY**: Slot 1 export bridge VERDICT upgrade. `tools/export_pr95_mlx_to_pytorch_state_dict.py` can now consume `tac.local_acceleration.deterministic_primitives.canonical_drift_bands_for_pr95_hnerv_decoder()` and tighten its parity assertion from `atol_max=2e-3` (legacy NUMERIC_TOLERANCE_RTOL_1e-2) to `atol_max=1e-4` (canonical NUMERIC_TOLERANCE_INHERENT band). Empirical anchor 3.05e-5 is well within this tighter band.

2. **SECONDARY**: register canonical equation `mlx_numpy_weights_portability_to_pytorch_with_drift_class_v1` via `tac.canonical_equations.register_canonical_equation` (sister subagent; depends on the canonical equations registry helper extensions).

3. **TERTIARY**: extend the drift CLI to measure MLX **gpu** device drift (vs cpu) — quick probe to determine whether gpu accelerator changes the attested band classification. Currently the canonical anchors are CPU-only.

4. **ROUTED TO SLOT 1**: pass the canonical attestation through Slot 1's long-training pipeline so trained-checkpoint exports automatically carry the drift attestation in their manifest (sister landing; do not invoke from THIS lane per Catalog #230 ownership map).

## Discipline closure

- Catalog #229 PV (premise verification): 6+ source files read in full before draft
- Catalog #117 / #157 / #174 / #235 / #289 canonical serializer with POST-EDIT `--expected-content-sha256`
- Catalog #110 / #113 APPEND-ONLY (only NEW files created; one APPEND-extension-pending to `pr95_hnerv_mlx.py` deferred to sister to avoid Catalog #230 ownership collision)
- Catalog #206 subagent checkpoints (3 emitted: step 1 PRE_READ, step 2 BASELINE_PROBE, step 3 DELIVERABLES_COMPLETE; step 4 LANDING_COMPLETE to follow)
- Catalog #230 sister-subagent ownership map verified DISJOINT from Slot 1 long-training infra + Slot 3 hinton-distilled + Slot 4 cascade plan
- Catalog #340 sister-checkpoint guard PROCEED (Slot 1 still in PRE_READ_PHASE; no file collision)
- Catalog #287 / #323 canonical Provenance: every artifact + attestation carries `evidence_grade=macOS-MLX-research-signal` + `score_claim=False` + `axis_tag=[predicted]`
- Catalog #131 fcntl-locked JSONL for probe outcomes ledger
- Catalog #1 (MPS noise) + Catalog #192 (macOS-CPU advisory) + Catalog #317 (one-arg local dispatch evidence-grade stamping): non-promotable markers preserved across the entire landing
- Catalog #205 (canonical select_inflate_device): no inflate device-fork in this landing
- Catalog #313 probe outcomes ledger row registered with PARTIAL verdict + 30-day expiry
- Catalog #299 quota brake check: 0 new STRICT preflight gates added (live count 361, far below 400 quota)
- Carmack MVP-first 5/5: ✅ all 5 criteria satisfied
- $0 GPU + ~50 min wall-clock

## Sister cross-references

- Slot 1 landing memo: `.omx/research/pr95_mlx_pytorch_export_parity_bridge_landed_20260525.md` (commit `44640a985`)
- Sister codex findings: `.omx/research/codex_findings_pr95_mlx_full_queue_execution_20260525T173024Z_codex.md`
- Slot 4 cascade plan: `.omx/research/pr95_mlx_loop_closure_cascade_plan_and_frontier_assessment_landed_20260525.md`
- Canonical drift CLI report: `.omx/tmp/pr95_drift_probe/per_op_drift_report.json`
- Canonical primitives module: `src/tac/local_acceleration/deterministic_primitives.py`
- Canonical CLI: `tools/measure_pr95_mlx_pytorch_per_op_drift.py`
- Canonical regression tests: `src/tac/tests/test_pr95_mlx_pytorch_drift_mitigation.py` (34 tests, all pass)
- Catalog #313 probe outcome ledger: `.omx/state/probe_outcomes.jsonl` (row `pr95_mlx_pytorch_drift_mitigation_engineering_20260525`)

---

## T3 Grand Council corrective verdict + empirical exploration (APPENDED 2026-05-25)

**APPEND-ONLY per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Catalog #110/#113 HISTORICAL_PROVENANCE non-negotiables.** Slot 2 body above is preserved verbatim; this footer documents the T3 grand council's empirical revision of the "NOT FIXABLE" verdict via active engineering exploration of the 4 unexplored paths.

**Sister landing**: `.omx/research/t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_landed_20260525.md` (task #1256; lane `lane_t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_20260525`)

### Corrective verdict per Catalog #307 (paradigm-vs-implementation)

Slot 2's `Aggregate drift reduction: 0.0%` finding was structurally over-stated. The reason: Slot 2 measured drift at a SINGLE Conv2d 3x3 scale (36→144 / 6×8) where the framework boundary drift floor is dominated by SIMD-vectorization-order at the small-spatial scale, NOT summation precision. Empirical exploration at THREE PR95-class spatial scales reveals:

| Path | Verdict | Max Observed Reduction | Carmack MVP-first 5/5 Step 2 |
|---|---|---:|---|
| Thread 1 (Kahan compensated summation) | PARTIALLY_FIXABLE_MARGINAL | 22.4% (at PR95 final-head class 256×48×64) | FALSIFIED at predicted >50% |
| Thread 2 (FP64 intermediate accumulation) | PARTIALLY_FIXABLE_MARGINAL | 22.4% (same scale) | FALSIFIED at predicted >50% |
| Thread 3 (MLX-side deterministic-reduction) | NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL | N/A (no public API) | NOT FALSIFIED |
| Thread 4 (cuDNN reference Conv2d 3x3) | DEFERRED_PENDING_PAID_DISPATCH | UNMEASURED | NOT FALSIFIED (deferred) |

### Revised canonical engineering primitive (supersedes Slot 2's all-or-nothing "NOT FIXABLE")

The canonical engineering primitive is **SCALE-CONDITIONAL substitution**:
- Use `tac.local_acceleration.mlx_scorer_adapters.MLXConv2dReference(accumulation_mode="kahan_fp32")` OR `accumulation_mode="fixed_fp64"` at PR95 stages with ≥144 channels AND ≥24 spatial dimensions (predicted 10-22% drift reduction)
- Use the canonical optimized MLX Conv2d at smaller stages (zero benefit from substitution)
- The ATTESTED-TOLERANCE PORTABILITY primitive remains the canonical operator-facing surface for state-dict bridge use; per-stage substitution is an OPTIONAL refinement

### Slot 1 export bridge VERDICT upgrade — DO NOT UPGRADE YET

Per the T3 council Karpathy + Contrarian dissents: the existing Slot 1 NUMERIC_TOLERANCE 3.05e-5 [contest-CUDA T4] verdict remains canonical. The CPU-only baseline used in Slot 2 + this exploration may not represent the contest's CUDA execution path. Thread 4 paid dispatch ($2-5 Modal A100 or Vast.ai 4090) is the reactivation criterion for Slot 1 VERDICT upgrade.

### Operator-routable next steps (per T3 council 5-priority queue)

1. **PRIMARY**: keep Slot 1 export bridge VERDICT unchanged at NUMERIC_TOLERANCE 3.05e-5 [contest-CUDA T4] until Thread 4 lands
2. **SECONDARY**: Boyd ADMM stacked Kahan+FP64 test (operator-decision)
3. **TERTIARY**: Yousfi full-decoder downstream scorer test
4. **QUATERNARY**: Daubechies extended-scale sweep (96×128 + 192×256 + 384×512)
5. **QUINARY (DEFERRED)**: operator-decision paid cuDNN dispatch ($2-5)

### Cross-references for the T3 corrective work

- T3 council memo: `.omx/research/t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_landed_20260525.md`
- T3 empirical evidence: `experiments/results/conv2d_drift_unexplored_paths_20260525T200834Z/results.json`
- T3 canonical CLI: `tools/measure_unexplored_mitigation_paths_drift.py`
- T3 canonical primitive extensions: `src/tac/local_acceleration/deterministic_primitives.py` (APPEND-ONLY: `kahan_compensated_sum`, `kahan_conv2d_3x3`, `fp64_intermediate_conv2d_3x3`, `classify_reduction_percent`, `ActiveExplorationPathVerdict`, 3 typed result dataclasses, `ACTIVE_EXPLORATION_CONV2D_DRIFT_UNEXPLORED_PATHS_ANCHOR`)
- T3 canonical regression tests: `src/tac/tests/test_t3_active_exploration_conv2d_drift_unexplored_paths.py` (36 tests)
- T3 sister codex routing: `src/tac/local_acceleration/mlx_scorer_torch_parity.py::build_mlx_conv2d_accumulation_probe_manifest` + `mlx_runtime_determinism_contract` + `tools/probe_mlx_conv2d_accumulation.py`
- T3 4 canonical equation candidates queued FORMALIZATION_PENDING per Catalog #344
- T3 Catalog #313 ledger row: `t3_grand_council_active_exploration_conv2d_drift_unexplored_paths_20260525` (verdict PROCEED; 30-day expiry)
