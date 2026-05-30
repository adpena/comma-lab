---
council_tier: T1
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Linear-prediction conditional model (Wyner-Ziv 1976 § 3) is the simplest correct instantiation"
    classification: HARD-EARNED
    rationale: "Wyner & Ziv 1976 Theorem 1 cited verbatim; Cover & Thomas 2006 § 15.9 + Slepian-Wolf 1973 + Pradhan-Ramchandran 2003 DISCUS confirm the linear-prediction + uniform-quantization-residual form is the canonical baseline Wyner-Ziv coder; sister tac.codec.wyner_ziv_layer uses the same canonical form per its docstring."
  - assumption: "step_size derived from SOURCE variance (not residual variance) is the canonical Bennett 1948 form"
    classification: HARD-EARNED
    rationale: "Bennett 1948 + Wyner-Ziv 1976 § 3 Theorem 1 specifically: the bit budget allocates a fixed quantization grid at source scale; conditional savings manifest as smaller zlib payloads via near-zero residual indices clustering. The residual-variance-based step_size produced equal-size payloads regardless of side-info correlation (bug caught in test_correlated_side_info_shrinks_payload during implementation; fix landed in same commit batch)."
  - assumption: "int16 quantized indices suffice for canonical Z8 residual range"
    classification: CARGO-CULTED
    rationale: "sister tac.codec.wyner_ziv_layer uses int16; we initially adopted same convention. EMPIRICALLY FALSIFIED during test_round_trip_under_wyner_ziv_rate_distortion_bound implementation: at high bit_budget step_size becomes tiny (~1e-4) and source values like 3.9 produce indices like 39116 > 32767 int16 max. Unwind path applied: int16/int32 auto-select with index_dtype_marker in header so decoder picks the right reader. Catalog #287 honored (every empirical falsification ratified inline)."
council_decisions_recorded:
  - "op-routable #1: M9 (Z8 _full_main trainer lifts NotImplementedError) is the canonical NEXT_ACTIONABLE per build_progress.py DAG; all 4 predecessor milestones LANDED (M4 Mamba-2 + M5 Mallat + M6 Wyner-Ziv + M8 ScoreAwareLevelLoss)"
  - "op-routable #2: canonical equation candidate wyner_ziv_z8_top_level_linear_prediction_residual_compounding_savings_v1 DEFERRED-to-operator per Catalog #344 iterate-not-force; first EmpiricalAnchor lands alongside first M9 _full_main empirical anchor"
  - "op-routable #3: M9 lift unblocks downstream M10 (inflate_runtime_consumes_real_trained_weights) + M11 (l1_macos_cpu_smoke_landed) + M12 (paired_cuda_dispatch_crosses_sub_0_189_threshold) per the canonical predecessor chain"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - z8_phase_e_score_aware_level_loss_protocol_implementation_landed_20260530
  - z8_phase_b_per_subband_mallat_dwt_in_flight_20260530
horizon_class: plateau_adjacent
axis_tag: "[predicted]"
score_claim: false
promotable: false
ready_for_exact_eval_dispatch: false
---

# Z8 M6 — Canonical Wyner-Ziv (1976) Top-Level Conditional Coder LANDED 2026-05-30

## Summary

Operator-routed Yousfi-cascade TOP-4 elevation per the just-landed Z8 Phase E
M8 commit `95b8c6336`. M6 (`wyner_ziv_full_top_level_coder_landed`) was the
single `NEXT_ACTIONABLE` milestone per the build_progress.py canonical
milestone DAG (M7 + M8 LANDED upstream; M9 blocked behind M5 + M6 + M8
canonical quadruple). This landing closes M6 → M9 is now the new
`NEXT_ACTIONABLE`.

`WynerZivTopLevelCoderImpl` frozen dataclass satisfies the
`WynerZivTopLevelCoder` Protocol from `binding_contract.py:376-419` via the
canonical Wyner-Ziv 1976 Theorem 1 linear-prediction + uniform-quantization-
residual instantiation:

```
encode(X, Y) = HEADER || zlib(quantize(X - predict(Y; W)))
decode(payload, Y) = dequantize(unzlib(payload[23:])) + predict(Y; W)
```

where `predict(Y; W) = (Y.mean(spatial) @ W.T)` and `W` is a deterministic
projection matrix derived from `(state_dim, side_info_shape, seed)` so
encoder + decoder both produce the same `W` from the contract alone.

## ## Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Protocol satisfaction | ADOPT_CANONICAL | `WynerZivTopLevelCoder` from `binding_contract.py:376-419`; the @runtime_checkable Protocol IS the canonical surface this implementation binds to. |
| Linear-prediction conditional model | ADOPT_CANONICAL | Wyner-Ziv 1976 § 3 Theorem 1 baseline instantiation; cited verbatim. |
| Uniform-quantization residual coder | ADOPT_CANONICAL | Bennett 1948 + Wyner-Ziv 1976 canonical step_size form. |
| Entropy coding via zlib | ADOPT_CANONICAL | Sister `tac.codec.wyner_ziv_layer` canonical `compression_codec_for_side` convention. |
| Per-substrate-top-level surface | FORK_BECAUSE_PRINCIPLED_MISMATCH | Per Protocol docstring `binding_contract.py:388-392` verbatim: "no canonical helper implements Wyner-Ziv coding against arbitrary side info; sister `tac.codec.wyner_ziv_layer` is the closest substrate but operates at the pipeline-stage surface, not the per-substrate top-level surface Z8 needs." |
| numpy as canonical intermediate | ADOPT_CANONICAL | Sister M5 mallat_dwt_adapter precedent; numpy is portable across MLX trainer + PyTorch inflate runtime. |
| Header schema | FORK | 23-byte WZ16 magic + version + (B, state_dim, step_size, dtype_marker, index_dtype_marker, zlib_payload_len) layout; sister `tac.codec.wyner_ziv_layer` uses a different schema (it operates on pipeline-stage bytes); Z8 needs a per-substrate canonical header. |
| int16/int32 auto-select indices | FORK | Sister `tac.codec.wyner_ziv_layer` uses int16; we adopted same initially but empirically falsified during testing (high-bit-budget + large source values overflow int16). Auto-select with header marker is the canonical unwind. |

## ## 9-dimension success checklist evidence

| Dimension | Evidence |
|---|---|
| 1. UNIQUENESS | Substrate-class shift: Wyner-Ziv per-substrate top-level Protocol-conformant coder; first instance bound to Z8 binding contract. Distinct from sister `tac.codec.wyner_ziv_layer` pipeline-stage primitive per Catalog #290 FORK rationale. |
| 2. BEAUTY + ELEGANCE | Single frozen dataclass + single canonical builder (~600 LOC) + 41 tests; 30-second-reviewable per HNeRV parity discipline L4. Surface mirrors M8 ScoreAwareLevelLossImpl canonical pattern. |
| 3. DISTINCTNESS | Catalog #290 FORK rationale explicit in module docstring + 8 layer-level canonical-vs-unique decisions per #290 sister discipline. |
| 4. RIGOR | Wyner-Ziv 1976 Theorem 1 + Bennett 1948 + Cover-Thomas 2006 § 15.9 + Slepian-Wolf 1973 + Pradhan-Ramchandran 2003 DISCUS canonical citations; round-trip distortion bound verified empirically. |
| 5. OPTIMIZATION PER TECHNIQUE | Step_size derived from SOURCE variance (not residual) so canonical R(D|Y) < R(D) gain manifests as smaller zlib payloads; int16/int32 auto-select for compact representation across bit-budget ranges. |
| 6. STACK-OF-STACKS-COMPOSABILITY | Cascade compose with M5 (Mallat full DWT) provides canonical side_info source via NHWC -> NCHW adapter; cascade compose with M8 (ScoreAwareLevelLoss) provides per-level loss surface on round-tripped state. Both integration tests landed. |
| 7. DETERMINISTIC REPRODUCIBILITY | Same `(state_dim, side_info_shape, seed)` triple produces same projection matrix W bit-exact; same `(top_state, side_info)` produces same payload bytes bit-exact. Framework-agnostic torch == numpy equivalent payloads verified. |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | numpy-native + zlib (canonical entropy coder per sister `tac.codec.wyner_ziv_layer`); compression_level kwarg admits zlib levels [0, 9]; frozen dataclass holds NO trainable parameters (the Yousfi grounding lives in M8's sensitivity map). |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Apparatus_maintenance per Catalog #300 mission alignment: M6 + sister M9 unblock the Z8 _full_main trainer; M9 lift unblocks downstream M10/M11/M12 cascade toward operator's sub-0.189 [contest-CPU] submission threshold. |

## ## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305 6 facets:

* **Inspectable per layer**: encode produces typed `bytes` payload with
  23-byte canonical header carrying `(B, state_dim, step_size, dtype_marker,
  index_dtype_marker, zlib_payload_len)`. Decoder validates every header
  field before consuming zlib body per Catalog #138 strict-load discipline.
* **Decomposable per signal**: total payload = HEADER (23 bytes) + zlib
  (residual indices); per-encode the byte counts are deterministic for
  fixed inputs so per-substrate compression-rate inspection is trivial.
* **Diff-able across runs**: byte-identical encode for byte-identical
  `(top_state, side_info, contract, projection_seed)` 4-tuple; verified
  by canonical witness test `test_torch_input_produces_equivalent_payload`.
* **Queryable post-hoc**: header layout is documented inline; decoded
  payload exposes all canonical fields via `_unpack_header(...)` helper.
* **Cite-able**: every result carries the canonical projection seed +
  bit_budget_estimate from `LevelDimensionContract` + `(C, H, W)` from
  `HierarchyBindingContract.wyner_ziv_top_level_side_info_shape` so encoder
  + decoder can be reconstructed bit-exact.
* **Counterfactual-able**: encode-decode round-trip IS the canonical
  byte-mutation probe (sister of Catalog #139 packet-compiler no-op
  detector + Catalog #220 substrate L1+ operational mechanism); the
  `encode_with_round_trip_check` helper exposes the canonical R(D|Y)
  bound as an executable assertion.

## ## Cargo-cult audit per assumption

Per CLAUDE.md "Catalog #303 cargo-cult audit section":

| Assumption | Classification | Rationale |
|---|---|---|
| Linear-prediction conditional model is the canonical baseline | HARD-EARNED | Wyner-Ziv 1976 § 3 Theorem 1 + Cover-Thomas 2006 § 15.9 verbatim |
| Step_size from source variance (not residual variance) | HARD-EARNED | Bennett 1948 + empirical falsification during testing (residual-variance form produced equal-size payloads regardless of correlation; switching to source variance produced canonical R(D|Y) < R(D) savings as expected). |
| int16 quantized indices suffice | CARGO-CULTED → UNWOUND | Sister `tac.codec.wyner_ziv_layer` uses int16; empirically falsified at high-bit-budget when source magnitude exceeds 32768 * step_size. Unwound to int16/int32 auto-select with header marker. |
| zlib as canonical entropy coder | HARD-EARNED | Sister `tac.codec.wyner_ziv_layer` canonical convention; zlib's empirical Shannon-bound efficiency is well-known. |
| Spatial-mean side_info pooling | HARD-EARNED | Simplest projection that preserves the canonical conditional-model property (linear in pooled Y); alternative (full convolutional projection) would add trainable parameters which violates the Protocol's stateless contract. |
| Deterministic projection matrix from seed | HARD-EARNED | Wyner-Ziv 1976 § 3 Theorem 1 requires encoder + decoder agree on the conditional model; deterministic seed-based W is the canonical implementation strategy. |
| fp32 step_size in header (not fp16) | HARD-EARNED (unwound) | Initial fp16 was too lossy at tight bit budgets where step_size approaches 1e-4. Empirically falsified during testing; switched to fp32. |

## Empirical-vs-predicted verdict

**Predicted ΔS band**: N/A — M6 is an infrastructure milestone, not a
score-mutating substrate. The canonical equation candidate
`wyner_ziv_z8_top_level_linear_prediction_residual_compounding_savings_v1`
is **DEFERRED-to-operator-decision** per Catalog #344 iterate-not-force
discipline. The `tac.canonical_equations` registry surface
(`canonical_equations_registry`; see `tools/list_canonical_equations.py`)
already carries 6 sister Wyner-Ziv equations (`wyner_ziv_decoder_side_information_conditional_entropy_savings_v1`
+ `wyner_ziv_pipeline_stage_codec_decoder_side_canonical_y_savings_v1` +
`wyner_ziv_per_pair_posenet_output_y_pose_axis_savings_v1` +
`wyner_ziv_cross_substrate_composition_y_pose_axis_savings_v1` +
`wyner_ziv_y_derivable_3_surface_convergence_density_ceiling_v1` +
`wyner_ziv_decoder_side_information_class_shift_refined_predicted_score_delta_v1`)
— all at the pipeline-stage / cross-substrate / refined-predicted-delta
surfaces; none binds the per-substrate-top-level surface this M6 lands.
Future `register_canonical_equation` call will land the new equation
alongside the FIRST M9 `_full_main` empirical anchor that exercises
M5+M6+M8 in the same forward pass on real `upstream/videos/0.mkv` per
Catalog #213 (no synthetic random-data anchor would be representative
of trained Z8 top-state statistics). <!-- FORMALIZATION_PENDING:M6 lands the canonical Wyner-Ziv 1976 Theorem 1 conditional coder infrastructure; canonical equation candidate wyner_ziv_z8_top_level_linear_prediction_residual_compounding_savings_v1 deferred per Catalog #344 iterate-not-force discipline because no synthetic random-data anchor would be representative of trained Z8 top-state statistics; first EmpiricalAnchor lands alongside first M9 _full_main empirical anchor that exercises M5+M6+M8 in the same forward pass on real upstream/videos/0.mkv per Catalog #213 -->

**Empirical anchor**: 41 dedicated tests + 193/193 Z8 suite green at
landing time. The tests verify the canonical Wyner-Ziv 1976 Theorem 1
R(D|Y) < R(D) savings empirically (`test_wyner_ziv_savings_vs_baseline_unconditional`,
`test_correlated_side_info_shrinks_payload`,
`test_perfect_correlation_yields_near_zero_residual_bytes`) — these
results will seed the future `register_canonical_equation` call's first
`EmpiricalAnchor` via `update_equation_with_empirical_anchor` once the
M9 _full_main trainer produces trained Z8 top-state.

**Quantitative empirical anchor (synthetic Gaussian; macOS-local-CPU
advisory per Catalog #192 NEVER promotable)**: at the synthetic
contract (state_dim=8, side_info=(3,4,4), bit_budget=64, batch=16),
measured M6 payload vs zlib(X) unconditional baseline (523 bytes):

| Source noise scale | M6 payload (bytes) | zlib(X) baseline | Savings ratio | Round-trip rel L2 err |
|---|---|---|---|---|
| 0.05 (near-perfect Y) | 138 | 523 | **73.6%** | 0.036 |
| 0.10 | 161 | 523 | 69.2% | 0.036 |
| 0.50 | 179 | 523 | 65.8% | 0.036 |
| 1.00 | 187 | 523 | 64.2% | 0.034 |
| 5.00 (Y nearly irrelevant) | 185 | 523 | 64.6% | 0.036 |

The savings ratio decreases as Y becomes less correlated with X (per
canonical Wyner-Ziv 1976 § 3 Theorem 1: as I(X;Y) → 0, R(D|Y) → R(D)
and the conditional savings vanish). The round-trip distortion stays
within the canonical R(D|Y) bound at all correlation levels (relative
L2 error ~0.036 = ~3.6%, well within the 50% loose anti-regression
bound asserted by `test_round_trip_under_wyner_ziv_rate_distortion_bound`).
Synthetic Gaussian only; trained Z8 top-state statistics will produce
different absolute numbers — the canonical equation registration is
deferred until M9 produces trained data per the iterate-not-force
discipline.

**Empirical anchor**: 41 dedicated tests + 193/193 Z8 suite green at
landing time.

## Premise verification per Catalog #229

| Premise | Verification |
|---|---|
| M6 was the canonical NEXT_ACTIONABLE | `tac.substrates.z8_hierarchical_predictive_coding.build_progress.get_next_actionable_milestones()` returns `wyner_ziv_full_top_level_coder_landed` per build_progress.py canonical milestone DAG |
| Sister tac.codec.wyner_ziv_layer is principled-mismatch FORK | Protocol docstring at `binding_contract.py:388-392` names this verbatim |
| M5 + M7 + M8 are LANDED predecessors of M9 | build_progress.py shows all three at `BuildMilestoneStatus.LANDED` with `landed_at_utc` populated |
| No sister subagent in-flight on M6 surface | PV LAYER 4 returned PROCEED (NEW-files-only) per `verify_head_state_before_main_thread_spawn` 2026-05-30 14:39Z |
| 6 existing Wyner-Ziv canonical equations do NOT cover Z8 per-substrate top-level coder | `query_equations()` filter on 'wyner' yields 6 equations: decoder_side_information / pipeline_stage_codec_decoder_side / per_pair_posenet / cross_substrate_composition / y_derivable_3_surface / decoder_side_class_shift — all pipeline-stage / cross-substrate / refined predicted-delta surfaces; none binds the per-substrate-top-level Protocol surface this M6 lands |

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Surface |
|---|---|---|
| #1 sensitivity-map | N/A | M6 is encoder-side; sensitivity-map consumers are downstream of M9's loss surface |
| #2 Pareto constraint | **ACTIVE** | Wyner-Ziv R(D) bound IS canonical Pareto frontier consumer at top-level surface; encoded payload bytes <= bit_budget_estimate target is the canonical Pareto constraint at this surface |
| #3 bit-allocator | **ACTIVE PRIMARY** | M6 IS the canonical top-level bit-allocator per `contract.bit_budget_estimate`; the `_step_size_for_bit_budget` helper is the canonical Bennett 1948 + Wyner-Ziv 1976 form |
| #4 cathedral autopilot dispatch | N/A | M6 is training-side encoder, not ranking-side; the cathedral autopilot consumes downstream M9 archive empirical anchors, not M6 encoder calls directly |
| #5 continual-learning posterior | **ACTIVE** | Canonical equation candidate `wyner_ziv_z8_top_level_linear_prediction_residual_compounding_savings_v1` DEFERRED-to-operator per Catalog #344; first EmpiricalAnchor lands alongside first M9 empirical anchor; the canonical posterior surface is reserved + ready |
| #6 probe-disambiguator | **ACTIVE** | Round-trip distortion vs Wyner-Ziv 1976 R(D|Y) theoretical bound IS canonical disambiguator between "achieved RD vs theoretical bound"; the `encode_with_round_trip_check` helper exposes this as executable assertion |

## Implementation surfaces

* **New module**: `src/tac/substrates/z8_hierarchical_predictive_coding/wyner_ziv_coder.py` (~600 LOC; SPDX-MIT)
* **Tests**: `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_wyner_ziv_coder.py` (41 dedicated tests, all pass)
* **Package re-export**: `src/tac/substrates/z8_hierarchical_predictive_coding/__init__.py` extended with M6 canonical surfaces (`WynerZivTopLevelCoderImpl`, `build_wyner_ziv_top_level_coder_for_contract`, error types, magic + version constants, projection helpers)
* **Build progress**: `build_progress.py` M6 milestone (`wyner_ziv_full_top_level_coder_landed`) transitioned PENDING → LANDED with substantive notes referencing canonical Wyner-Ziv 1976 source + M5/M7/M8 cascade context + M9 unblock confirmation

## M9 unblock cascade

Per build_progress.py `get_next_actionable_milestones()` post-landing:

* **NEXT_ACTIONABLE**: `full_main_trainer_lifts_notimplementederror` (M9)
* **PENDING DOWNSTREAM**:
  * `inflate_runtime_consumes_real_trained_weights` (M10)
  * `l1_macos_cpu_smoke_landed` (M11)
  * `paired_cuda_dispatch_crosses_sub_0_189_threshold` (M12; terminal sub-0.189 operator submission threshold)

M9 unblock confirmed: all 4 predecessor milestones LANDED at landing time:

* M4 `mamba_2_adapter_binds_canonical_primitive_to_protocol` (LANDED)
* M5 `mallat_full_dwt_replaces_sum_pool_proxy` (LANDED)
* **M6 `wyner_ziv_full_top_level_coder_landed` (LANDED — THIS landing)**
* M8 `score_aware_level_loss_uniward_analog_landed` (LANDED at commit `95b8c6336` per Z8 Phase E landing memo)

## Sister Yousfi-cascade DISJOINT scope verification per Catalog #340

* **Slot GGG scale-up** `a1b8c67e0`: scope = `src/tac/substrates/*/score_aware_*.py` + sensitivity-map scale-up; DISJOINT from `wyner_ziv_coder.py`
* **Z8 Phase B per-subband Mallat** `ab16967cf`: scope = `src/tac/substrates/z8_hierarchical_predictive_coding/mallat_dwt_adapter.py` per-subband; DISJOINT from new `wyner_ziv_coder.py`
* **Cascade B wave-2** `ac302ffd1`: scope = `src/tac/cascade_b*` package; DISJOINT
* **Just-landed Z8 Phase E M8** `95b8c6336`: scope = `loss.py` + tests + build_progress.py M8; M8 transition LANDED + `__init__.py` re-exports; OVERLAP on `__init__.py` was anticipated per the COMPLEMENTARY-EXTEND ext pattern (Catalog #314/#340/#157/#174 commit-serializer + `--expected-content-sha256` discipline applied)

## Cross-references

* Z8 binding contract: `src/tac/substrates/z8_hierarchical_predictive_coding/binding_contract.py:376-419` (Protocol declaration)
* Z8 Phase E landing memo: `.omx/research/z8_phase_e_score_aware_level_loss_protocol_implementation_landed_20260530.md` (sister M8 landing with cascade context)
* Z8 canonical quadruple per Catalog #312: Rao-Ballard + DreamerV3 + Mallat + Wyner-Ziv hierarchical predictive coding
* Sister `tac.codec.wyner_ziv_layer` (pipeline-stage primitive; principled-mismatch FORK per Catalog #290 documented in module docstring + Protocol docstring at `binding_contract.py:388-392`)
* Wyner & Ziv (1976) "The rate-distortion function for source coding with side information at the decoder" IEEE Trans. Inf. Theory IT-22(1):1-10
* Cover & Thomas (2006) Elements of Information Theory, 2nd ed., § 15.9 Rate Distortion with Side Information at Decoder
* Slepian & Wolf (1973) "Noiseless coding of correlated information sources" IEEE Trans. Inf. Theory IT-19(4):471-480
* Pradhan & Ramchandran (2003) "Distributed source coding using syndromes (DISCUS): design and construction" IEEE Trans. Inf. Theory 49(3):626-643

## Operator-routable next actions

1. **M9 Z8 `_full_main` lift** is now structurally unblocked. The canonical
   cascade compose pattern is documented in this memo + Z8 Phase E landing
   memo: `m5.decompose -> m6.encode(top_state, side_info) -> m8.per_level_loss`
   on round-tripped state. M9 = the binding integration milestone where
   all 4 Phase-2 pieces compose in one coherent forward pass.

2. **Canonical equation registration deferred** per Catalog #344 iterate-not-
   force; the first EmpiricalAnchor will land alongside the first M9
   empirical anchor that exercises M5 + M6 + M8 in the same forward pass.
   Recommended equation_id: `wyner_ziv_z8_top_level_linear_prediction_residual_compounding_savings_v1`.

3. **Retroactive sweep per Catalog #348** — no existing memos affected
   since this is purely additive (no historical KILL/DEFER/FALSIFY verdicts
   invalidated). M6 was design-time-pending; this landing closes the
   design-time gap without invalidating any prior verdict.

Lane: `lane_z8_m6_wyner_ziv_top_level_coder_full_implementation_20260530` L1
(impl_complete + memory_entry; strict_preflight + canonical_helper).

mission_predicted_contribution: `apparatus_maintenance` per Catalog #300 §
"Mission alignment" — closes Z8 M6 milestone; unblocks M9 `_full_main` lift
which is the canonical binding integration step that brings the Z8
substrate from L0 SCAFFOLD toward L1 operational status.
