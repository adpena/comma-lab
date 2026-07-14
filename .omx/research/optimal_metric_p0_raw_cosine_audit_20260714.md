# Optimal-metric P0 raw-cosine fidelity audit — 2026-07-14

**Pointer status:** unchanged. `[contest-CPU Linux x86_64]` submittable
pointer `0.1910828242`; borrowed defensive bank `0.1880443979880752` remains
non-submission. This audit is `research_only=true`, `$0`, and has no score or
promotion authority.

**Canonical law:** `metric_id=argmax_native_vjp_fidelity_v1`.
Per-state receipt schema is `reachable_decision_geometry_fidelity.v1`; selector
schema is `reachable_decision_preconditioner_selection.v1`. They are not metric
identifiers.

## Audit method and scope

The exact source scan was:

```text
rg -l 'cosine_similarity|cosine_debt|vjp.*cosine|cosine.*vjp|costate.*cosine|cosine.*costate|fidelity.*cosine|cosine.*fidelity' src/tac --glob '!**/tests/**' tools
```

It returned the 36 production/canonical/tool files classified below. Test files
inherit the production classification and cannot create authority. A raw cosine
is retained only when it compares two implementations of the same tensor-valued
function in the same chart, is a mechanism canary, or is explicitly diagnostic.
It never again decides whether a teacher surrogate/provider matches the teacher.

Geometrically, `1-cos` is a curved representational divergence on a normalized
manifold, not the proper flat quadratic Bregman/Mahalanobis law used here. That
classification explains the mismatch; it does not erase legitimate same-chart
parity canaries below.

## Active teacher/provider paths — replaced or fail-closed

| Source | Previous role | Landed authority after this pass |
|---|---|---|
| `src/tac/boundary_math/segnet_gradient_replacement.py` | `measure_costate_agreement` exposed ambient RGB costate cosine/L2 without a type-level authority label. | Ambient record now says `ambient_input_costate_diagnostic_only`, `replacement_authority=false`; `measure_reachable_costate_agreement` emits the canonical receipt. |
| `src/tac/witness_dsl/scorer_gradient_policy.py` | Periodic student/cache modes compared ambient costate cosine/L2/norm thresholds. | Ambient values are diagnostics only. Their four legacy config fields remain optional for parse-back and new policies may omit them. Every replacement now requires a content-described renderer-gradient preconditioner, canonical reachable receipt, positive `rho` and `eta`, plus its existing content-bound exact teacher one-step check. Missing geometry falls back to the full teacher. |
| `src/tac/witness_dsl/onpolicy_scorer_surrogate_policy.py` + `tools/probe_onpolicy_scorer_surrogate.py` | The contract and probe could treat nonnegative ambient costate cosine as a decision predicate. | Contract compiles `raw_input_costate_cosine=AMBIENT_DIAGNOSTIC_ONLY`; probe no longer changes the exact CE/through-R decision from raw cosine. |
| `src/tac/scorer_surrogate/whole_teacher_distilled_student.py` | Full input-VJP cosine/L2 was named decisive. | Legacy rows are labeled ambient diagnostic only; new `reachable_vjp_pair_metrics` and `aggregate_reachable_vjp_pair_metrics` use the canonical metric. |
| `src/tac/witness_dsl/whole_teacher_distilled_student_policy.py` + `tools/probe_whole_teacher_distilled_student.py` | Training-gradient admission used worst-pair teacher/input-VJP cosine/L2. | Admission requires exactly 600 canonical metric rows with distinct law/state-schema/selector-schema custody, a byte-rehashed selected-M receipt, n600 fidelity aggregate, positive worst-pair `rho`/`eta`, NumPy/framework parity in the identical student chart, and a byte-rehashed exact functional receipt. Receipt schemas, law ID, state count, selected-M status, cross-receipt SHA, and aggregate values are verified at decision time. Missing geometry is `NO_VERDICT_DATA_CUSTODY`. The old ambient-fit driver is implementation-blocked from `fit-measure`. |
| `tools/probe_instant_projected_adjoint.py` | Renderer-gradient cosine was computed through the ambient costate helper. | Renderer gradients are already reachable pullbacks; the gate now calls `reachable_pullback_geometry_summary` with sealed `M=I`, reporting canonical `rho` and `eta`. |
| `src/tac/canonical_equations/argmax_native_vjp_fidelity_20260714.py` | A second hand-coded `M=I` reduction duplicated the law. | It delegates to `weighted_pullback_summary`; one implementation now owns the numerical law. |

The same canonical helper now exposes categorical finite Bregman/KL,
dual-Euclidean squared-Hessian distance, and a non-negative extended-KL
estimator. The dual distance is explicitly not VJP `rho` and cannot launder a
Fisher-natural cotangent solve.

The regression test `test_active_surrogate_admission_has_no_raw_costate_cosine_branch`
guards the three previously decisive raw-cosine branches.

## Legacy training objectives — visible but not authority

| Source | Classification | Required reformulation |
|---|---|---|
| `src/tac/scorer_surrogate/onpolicy_costate.py` | First-cut MSE + ambient cosine auxiliary. Exact on-policy functional checks, not this loss, decide admission. | Replace auxiliary loss with the selected reachable decision metric when real renderer/preconditioner custody exists. |
| `src/tac/scorer_surrogate/amortized_onpolicy_costate.py` | First-cut ambient auxiliary; EMA improvement is internal fit evidence only. | Consume the same selected metric and preserve exact functional validation. |
| `src/tac/scorer_surrogate/whole_teacher_distilled_student.py` (`cached_student_fit_loss_mlx`) | Legacy boundary-masked ambient VJP objective. It cannot run as the requested optimal-form measurement after this pass. | Add checkpoint-bound renderer pullbacks or matrix-free renderer VJP, selected `M`, value/Jacobian stages, and exact functional validation before re-enabling `fit-measure`. |

These are not called optimal-form failures. They are retained first-cut
optimization auxiliaries with no negative-verdict authority.

## Historical first-cut evidence — preserved, never reused as current authority

The following files encode the sealed round-2/round-3/round-4 experiments or
their historical equations/policies. Rewriting their recorded fields would
destroy provenance; current consumers must use the new canonical law instead:

- `src/tac/scorer_surrogate/frozen_replay_convex_head.py`
- `src/tac/scorer_surrogate/replace_round3_fidelity_wall.py`
- `src/tac/scorer_surrogate/replace_round4_support_ranking.py`
- `src/tac/witness_dsl/replace_round3_fidelity_wall_policy.py`
- `tools/probe_frozen_replay_convex_head.py`
- `tools/probe_grokking_ridge_round2.py`
- `tools/probe_replace_round3_fidelity_wall.py`
- `src/tac/canonical_equations/frozen_replay_convex_head_contraction_20260713.py`
- `src/tac/canonical_equations/replace_round3_fidelity_wall_20260713.py`
- `src/tac/canonical_equations/replace_round4_support_ranking_20260713.py`
- `src/tac/canonical_equations/whole_teacher_distilled_student_20260713.py`

Their raw-cosine conclusions are `INSTANCE-OF-FIRST-CUT-FORMULATION` only. The
decision/Fisher/functional family is intact.

## Ambient-justified or non-fidelity cosine uses

| Sources | Why raw cosine is genuinely scoped here |
|---|---|
| `src/tac/quantization_audit.py`, `src/tac/cuda_levelset_training.py` | Same-function fp32-vs-quantized or eager-vs-compiled tensor parity in one chart; argmax/exact-delta checks remain primary. |
| `tools/probe_segnet_costate_injection.py` | Exact-vs-injected and sign-reversed mechanism canaries prove the chain-rule seam; they do not admit an approximate surrogate. |
| `tools/adjudicate_p0_costate_reuse_k2.py`, `tools/probe_p0_costate_reuse_k2.py`, `tools/bench_custom_sparse_adjoint_kernel.py` | Costate cosine is a telemetry field; actual real-R CE/d_seg/d_pose guard outcomes and kernel/runtime receipts decide. |
| `tools/probe_costate_trust_region_economics.py`, `tools/probe_jacobian_drift_certificate.py` | Inherited fidelity/drift descriptors; exact teacher-step/trust/economics receipts decide provider use. |
| `tools/probe_seg_loss_surrogate_disambiguator.py`, `src/tac/substrates/pact_nerv_selector_v3/heterogeneous_bit_allocation.py` | Same-parameter-chart force-conflict/allocation diagnostics, not teacher-surrogate fidelity or score authority. |
| `src/tac/pr101_split_brotli_codec_derivers.py`, `tools/cpu_cuda_xray_substrate_class_classifier.py`, `src/tac/optimization/pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.py`, `src/tac/cathedral_consumers/substrate_fit_diagnostic_consumer/__init__.py` | Histogram, drift-signature, graph-signature, or bounded-ranking similarity; no gradient/costate replacement claim. |
| `src/tac/contrib/variational_gen.py` | Design-note pseudocode, not an executable evaluator or fidelity gate. |
| `tools/probe_surrogate_vjp_fidelity_metric.py` | Reads sealed legacy cosine reductions only to prove the locus artifact and reaggregate retained `M=I` evidence; advanced claims fail closed. |

## Measured-data boundary

Machine receipt:
`.omx/research/optimal_metric_p0_data_custody_receipt_20260714.json`.

- Whole-teacher optimal student: `n_pairs=0`, `fit_steps=0`, `teacher_calls=0`;
  strict manifest absent.
- Round 2: 120 real-n600-source heldout states retain exact/candidate 19-D
  pullbacks, but no matched finite real-R outcome or renderer range basis.
- K2: 600 rows retain exact through-R line-search outcomes, but no candidate
  surrogate pullback, logits/probabilities, or margin Jacobian.

Therefore categorical-Fisher-natural vs winner-rival-margin-Fisher-natural vs
identity selection, the centered-logit optimal-form student verdict, and the
null-removal/reweighting numerical decomposition are all
`NO_VERDICT_DATA_CUSTODY`. This is an admission classification, not an
`INSTANCE`, `FORMULATION`, `FAMILY`, or `PARADIGM` negative.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5/v8 canonical specs;
`reports/latest.md`; lane/subagent/task/gradient/modal/posterior canonical state;
latest sister findings/design/council/directive memos; retained round-2/round-3
receipts and cleanup manifests; K2 n600 rows; whole-teacher blocker receipt;
canonical GT-cache custody; live inbox through the latest checkpoint.
