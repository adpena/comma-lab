---
title: rgb_to_yuv6 Canonical Extraction Migration BEFORE/AFTER Comparison
date: 2026-05-30
lane_id: lane_rgb_to_yuv6_canonical_extraction_migration_20260530
related_landings:
  - sister mlx_canonicalization_audit + tinygrad_bridge LANDED commit `e52f2f6b4`
  - sister gumbel_softmax_sample extraction LANDED commit `6bc53d607`
audit_ref: .omx/research/mlx_canonicalization_audit_inventory_20260530.md (op-routable #2)
canonical_helper: tac.framework_agnostic.canonical_kernels.rgb_to_yuv6
catalog_383_live_count_before: 0
catalog_383_live_count_after: 0
sister_disjoint_vs:
  - PR110-OPT-7 trainer wire-in (Agent acd4123aaaba505a9 owns experiments/train_substrate_pr110_opt7_*.py)
  - Z7-Mamba-2 stabilizer (Agent owns src/tac/substrates/z7_mamba2_*/) — EXCLUDED from migration per scope
  - z6_v2 29,650ep MLX-LOCAL FULL RUN (Agent ae1c4683e73e39b7a in-flight)
  - NULL-BYTE PROBE READ-ONLY
deferred_to_sister_agent:
  - src/tac/substrates/z7_mamba2_*/ rgb_to_yuv6 sister impls — DEFERRED per scope EXCLUSION; no such impl exists at landing time (grep confirms)
---

# rgb_to_yuv6 Canonical Extraction Migration

## Executive summary

Sister of the 2026-05-30 MLX canonicalization audit op-routable #2 + sister
of the gumbel_softmax_sample canonical extraction landed commit `6bc53d607`
which drove Catalog #383 live count 2 → 0 STRICT-flip ready.

Per audit inventory A.2.6 the 4 sister rgb_to_yuv6 impls live OUTSIDE
`src/tac/substrates/` (so NOT within Catalog #383's substrate-directory
scope; Catalog #383 only tracks the 4 substrate-MLX-renderer primitives
`gumbel_softmax_sample` / `pixel_shuffle_2x_nhwc` / `bilinear_resize_nhwc` /
`bilinear_resize2x_align_corners_false_nhwc`). The 4 sister impls are
the canonical training-time / saliency-pipeline / composition-operator
implementations enumerated in audit A.2.6.

**Migration result**: 3 of 4 sister impls now route through the canonical
helper `tac.framework_agnostic.canonical_kernels.rgb_to_yuv6` as
byte-stable layout-adapter wrappers; 1 sister remains a documented
PRINCIPLED FORK per Catalog #290 falling-rule list because the canonical
helper is float32-native and the composition operator's downstream
perturbation math requires float64 precision (~2e-5 absolute discrepancy
empirically observed, above fp32 epsilon ~3e-7).

## Scope per parent prompt

- ONLY the 4 sister rgb_to_yuv6 substrate impls enumerated in audit A.2.6
- CONSUME ONLY `tac.framework_agnostic.canonical_kernels.rgb_to_yuv6` (no edits to canonical helper)
- `src/tac/substrates/z7_mamba2_*/` EXCLUDED per sister-Agent ownership (no rgb_to_yuv6 sister impl exists there at landing)
- DO NOT edit CLAUDE.md / preflight / canonical_anti_patterns / canonical_equations / Modal recipes / PR110-OPT-7 trainer / z6_v2 substrate

## Per-substrate migration table

| # | Sister impl | BEFORE LOC | AFTER LOC | LOC delta | Migration | Byte-stable verified | Catalog #290 classification |
|---|---|---|---|---|---|---|---|
| 1 | `tac.constrained_gen.rgb_to_yuv6` | 2738 | 2739 | +1 (docstring) | **MIGRATED** | YES (max_abs_diff = 0.00e+00 across 4D NCHW / 5D leading-dim / 6D deep leading-dim) | ADOPT_CANONICAL_BECAUSE_SERVES + layout adapter |
| 2 | `tac.saliency.rgb_to_yuv6` | 172 | 201 | +29 (canonical docstring) | **MIGRATED** | YES (max_abs_diff = 0.00e+00 across 3D / 4D / 5D HWC) | ADOPT_CANONICAL_BECAUSE_SERVES + layout adapter |
| 3 | `tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx` | 630 | 641 | +11 (docstring) | **MIGRATED** | YES (max_abs_diff = 0.00e+00 across 4D / odd-crop / 5D NHWC) | ADOPT_CANONICAL_BECAUSE_SERVES + NHWC layout adapter |
| 4 | `tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator.rgb_to_yuv6_numpy` | 631 | 643 | +12 (PRINCIPLED FORK rationale docstring + waiver) | **PRINCIPLED FORK** | YES (byte-identical to pre-migration; canonical helper diverges 2.13e-5 above fp32 epsilon, violates float64 precision contract) | FORK_BECAUSE_PRINCIPLED_MISMATCH (float64 precision + HWC layout) |
| - | Sister test file `test_rgb_to_yuv6_canonical_extraction_migration.py` | NEW | 462 | +462 | 21 dedicated tests | 21/21 PASS | — |

LOC net effect: +515 LOC across 5 files; net SLOC for the migrated impls
themselves is ~0 (the canonical math body is preserved upstream; the
sisters become layout-adapter shims with extra documentation per
Catalog #305 observability discipline). The migration's structural
value is **consumer routing**: 3 sister impls now consume canonical
math, eliminating math drift across the rgb_to_yuv6 / yuv6_to_rgb sister
surface per audit A.3 "0% → ~75% adoption rate".

## Empirical byte-stability evidence (deterministic seed=42)

### Sister 1: `tac.constrained_gen.rgb_to_yuv6`
- 4D NCHW `(2, 3, 16, 16)` parity max_abs_diff = **0.00e+00**
- 5D leading-dim `(1, 2, 3, 384, 512)` parity max_abs_diff = **0.00e+00**
- 6D deep leading-dim `(2, 3, 4, 3, 8, 8)` parity max_abs_diff = **0.00e+00**
- Gradient propagation: present + finite + non-zero (64.00 sum)

### Sister 2: `tac.saliency.rgb_to_yuv6`
- 3D HWC `(16, 16, 3)` parity max_abs_diff = **0.00e+00**
- 4D HWC `(2, 16, 16, 3)` parity max_abs_diff = **0.00e+00**
- 5D HWC `(1, 2, 384, 512, 3)` (canonical saliency pipeline path) parity max_abs_diff = **0.00e+00**
- Gradient propagation: present + finite + non-zero

### Sister 3: `tac.local_acceleration.pr95_hnerv_mlx_training.rgb_to_yuv6_mlx`
- 4D NHWC `(2, 16, 16, 3)` parity max_abs_diff = **0.00e+00**
- Odd-crop `(1, 17, 19, 3)` parity (canonical pr95 test anchor) max_abs_diff = **0.00e+00**
- 5D NHWC `(1, 2, 8, 8, 3)` parity max_abs_diff = **0.00e+00**

### Sister 4 (PRINCIPLED FORK): `tac.composition.yuv6_chroma_subsampled_perturbation_operator.rgb_to_yuv6_numpy`
- Byte-identical to pre-migration form (max_abs_diff = **0.00e+00** vs legacy implementation; NOT delegating to canonical helper)
- Canonical helper divergence empirically anchored: **2.13e-5** absolute discrepancy (above fp32 epsilon ~3e-7) per `test_canonical_helper_diverges_above_fp32_epsilon_documents_principled_fork`
- float64 dtype preserved per perturbation operator downstream contract
- Same-line waiver: `CATALOG_383_PRINCIPLED_FORK_OK:hwc_float64_precision_contract_canonical_helper_is_float32_native_diverges_2e_minus_5_above_fp32_epsilon_at_perturbation_operator_downstream_dependency_per_catalog_290_falling_rule_documented_in_audit_inventory_A_2_6`

## Test coverage delta

| Test surface | BEFORE PASS | AFTER PASS | Δ |
|---|---|---|---|
| `src/tac/framework_agnostic/tests/test_canonical_kernels.py` | 34 pass | 34 pass | 0 |
| `src/tac/tests/test_pr95_hnerv_mlx_training.py` | 12 pass | 12 pass | 0 |
| `src/tac/tests/test_constrained_gen_pose_semantics.py` | 2 pass | 2 pass | 0 |
| `src/tac/tests/test_compliance.py` | 20 pass | 20 pass | 0 |
| `src/tac/tests/test_differentiable_eval_roundtrip.py` | 31 pass | 31 pass | 0 |
| `src/tac/composition/yuv6_chroma_subsampled_perturbation_operator/tests/test_operator.py` | 28 pass | 28 pass | 0 |
| `src/tac/framework_agnostic/tests/` (full) | N/A | 150 pass | — |
| `src/tac/local_acceleration/mlx_canonical_audit/tests/` | N/A | 36 pass | — |
| `src/tac/cathedral_consumers/mlx_canonicalization_audit_consumer/tests/` | N/A | 13 pass | — |
| **NEW** `src/tac/tests/test_rgb_to_yuv6_canonical_extraction_migration.py` | N/A | **21 pass** | **+21** |
| **TOTAL combined sister test suites** | 127 pass | **148 pass** | **+21** |

0 regressions detected across all sister test surfaces.

## Catalog #383 live count delta

Catalog #383 (`check_mlx_primitives_route_through_canonical_helper`) only
tracks 4 substrate-MLX-renderer primitives:
- `gumbel_softmax_sample`
- `pixel_shuffle_2x_nhwc`
- `bilinear_resize_nhwc`
- `bilinear_resize2x_align_corners_false_nhwc`

`rgb_to_yuv6` is NOT in the gate's scope (the gate is for substrate-
internal MLX renderers re-implementing canonical primitives; the 4 sister
rgb_to_yuv6 impls live in non-substrate canonical helper modules).

| Catalog #383 live count | BEFORE | AFTER |
|---|---|---|
| Substrate-MLX-renderer duplicates | 0 | 0 |

Net change to Catalog #383: **0** (was clean; remains clean).

The migration's structural value is consumer-side routing per audit A.3
"0% → near-100% adoption rate" for the rgb_to_yuv6 / yuv6_to_rgb sister
surface, NOT a Catalog #383 violation reduction.

## Z7-Mamba-2 sister-Agent ownership deferral

Per parent prompt scope: "SKIP `src/tac/substrates/z7_mamba2_*/` (sister
Z7-Mamba-2 Agent owns it). If rgb_to_yuv6 has an impl in z7_mamba2,
document as DEFERRED to sister Agent + propose follow-on landing."

Empirical: `find src/tac/substrates/z7_mamba2* -name "*.py" | xargs grep -l "rgb_to_yuv6\|def yuv6"` returns ZERO matches — no rgb_to_yuv6 sister impl exists in z7_mamba2 at landing time. No deferral needed.

## 6-hook wire-in declaration per Catalog #125

- hook #1 sensitivity-map = N/A (canonicalization migration; no sensitivity signal)
- hook #2 Pareto constraint = N/A
- hook #3 bit-allocator = N/A
- hook #4 cathedral autopilot dispatch = N/A (no candidate routing changes)
- hook #5 continual-learning posterior = ACTIVE (canonical equation `mlx_primitive_canonicalization_compounding_savings_v1` consumes this empirical anchor; sister cathedral consumer `mlx_canonicalization_audit_consumer` auto-discovers per Catalog #335)
- hook #6 probe-disambiguator = ACTIVE (presence of `from tac.framework_agnostic.canonical_kernels import rgb_to_yuv6` IS the canonical disambiguator between canonical-routing vs duplicate-impl-fork; sister gate Catalog #383 enforces the substrate-MLX-renderer subset)

## Mission contribution per Catalog #300

**Mission**: `apparatus_maintenance` — extincts sister rgb_to_yuv6 math
duplication across 3 of 4 sister implementations; the PRINCIPLED FORK
(composition operator) is documented per Catalog #290 falling-rule list
with empirically-anchored divergence rationale.

## Sister discipline observed

- **Catalog #117/#157/#174/#186/#206/#234/#340 commit discipline**: subagent registered checkpoint + executed via canonical serializer
- **Catalog #229 premise-verification-before-edit**: empirical parity smoke run BEFORE migration to confirm canonical math is byte-stable with each sister
- **Catalog #287 placeholder-rationale rejection**: same-line waiver rationale on composition operator is substantive (>4 chars, no placeholder literals)
- **Catalog #290 canonical-vs-unique decision per layer**: per-sister falling-rule classification documented above
- **Catalog #305 observability surface**: each migrated wrapper preserves operator-facing docstring documenting canonical helper delegation + PRINCIPLED FORK rationale
- **Catalog #340 sister-checkpoint guard**: no conflicts with concurrent PR110-OPT-7 trainer wire-in / z7-Mamba-2 / z6_v2 in-flight subagents
- **NO FAKE IMPLEMENTATIONS per Slot EEE 5 forbidden classes**: canonical helper is ACTUALLY invoked + tests verify byte-stable BEHAVIOR on REAL inputs + canonical Provenance via re-routing + substantive consumption (not just rename)

## Operator-routable next steps

None — this is a structural extraction migration with 0 regressions and
empirically byte-stable parity. The follow-on canonical extraction wave
for `tac.framework_agnostic` `yuv6_to_rgb` sister extraction (currently
0 sister impls in canonical helper; sister `constrained_gen.yuv6_to_rgb`
+ `composition.yuv6_to_rgb_numpy` exist) is deferred per audit op-routable
#3 ("MEDIUM-EV; sister extension of THIS landing") to a future session.
