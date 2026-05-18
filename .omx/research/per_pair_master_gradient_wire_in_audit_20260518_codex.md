# Per-Pair Master-Gradient Wire-In Audit

Date: 2026-05-18
Author: Codex

## Scope

Audit requested by
`codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518.md`
and the operator clarification that per-pair master gradients are expected to
serve training time, compress time, and possibly inflate time. This memo is a
wire-in status ledger, not a score claim.

## Executive Verdict

Per-pair master-gradient signal is partially active and already consumed by the
canonical Cathedral autopilot path, but the stack is not yet a closed
end-to-end optimizer for every granularity surface.

| Surface | Status | Current authority | Closure requirement |
| --- | --- | --- | --- |
| Gradient extraction | ACTIVE for fec6/A1/PR101; FAIL-CLOSED for PR106/HNeRV/PR107 packed families | `tools/extract_master_gradient.py` | Add grammar-aware projectors for packed/length-prefixed families before emitting anchors. |
| Master-gradient consumers | ACTIVE for consumers 1/2/3/4/5/6/15 | `src/tac/master_gradient_consumers.py` | Add consumers 7/8/10/11/12/13/14 only when they produce typed planning payloads, not loose metrics. |
| Cathedral autopilot | ACTIVE canonical consumer | `tools/cathedral_autopilot_autonomous_loop.py` | Keep planner sidecars planning-only until byte-closed CandidateModificationSpec/runtime packets exist. |
| Training time | ACTIVE | `src/tac/training_curriculum/per_pair_master_gradient_wire_in.py` | Verify every trainer that claims gradient-aware curriculum imports the namespace adapter instead of bespoke pair weights. |
| Compress time | ACTIVE | `src/tac/compress_time_optimization/per_pair_master_gradient_wire_in.py` and pipeline auto-threading | Extend regression coverage to representative compressor policies beyond policy-dict plumbing. |
| Inflate time post-processing | DESIGN-ONLY | `inflate_time_post_processing` namespace is legal in `src/tac/optimization/per_pair_namespace_wire_in.py` | Build a concrete, scorer-free inflate-time adapter before any runtime path consumes this signal. |
| Hook 3 bit allocation | ACTIVE under modern namespace, DORMANT under literal old name | `src/tac/optimization/bit_allocator_end_to_end.py::allocate_per_pair_bits` | Do not resurrect `tac.bit_allocator.allocate_per_pair`; route callers through the canonical namespace adapter. |
| Field equation planner | DORMANT for per-pair fp64 master-gradient payloads | `src/tac/optimization/field_equation_planner.py::field_row` | Add optional optimal-plan/per-pair summary fields and KKT readiness annotations without changing action score authority. |
| Xray primitives | ACTIVE generally, DORMANT for direct master-gradient primitive | `src/tac/xray/registry.py`, `src/tac/xray/wire_in.py` | Add a typed `master_gradient_per_pair_difficulty` xray primitive if xray consumers need direct per-pair gradient access. |
| Continual learning posterior | DORMANT for per-pair keyed updates | `src/tac/continual_learning.py::posterior_update_locked` | Add a canonical per-pair anchor adapter or sister ledger; do not mutate posterior JSONL directly. |
| Rashomon ensemble | ACTIVE | `RashomonEnsembleRanker.update_all_from_master_gradient` | Keep disagreement queue diagnostic/planning-only unless exact eval closes a candidate. |

## Cathedral Autopilot Consumption

Cathedral autopilot is not merely report-aware. The default ranking path calls
`apply_z1_empirical_revision_to_candidate_delta`, which delegates into
`adjust_predicted_delta_for_venn_classification_v2`. The v2 cascade first tries
`tac.master_gradient_consumers.load_optimal_plan_for_archive`; when a matching
sidecar exists, it replaces the candidate's predicted delta with the
Lagrangian-dual optimal plan payload. The optional
`--include-master-gradient-optimal-plans` path also materializes planning-only
candidate rows from optimal-plan sidecars.

The important boundary: `rerank_candidates_via_master_gradient` is still
conservative. It annotates anchor/authority presence and refuses to invent score
movement without a candidate modification spec and packet proof. That is the
right contest-compliance posture.

## Training / Compress / Inflate Routing

Training and compress-time adapters already route through the shared namespace
helper rather than bespoke local formulas. That helper auto-loads per-pair
gradients, runs Wyner-Ziv classification, and delegates Hook 3 budgets to
`allocate_per_pair_bits`.

Inflate-time post-processing is not yet an implemented consumer in the inspected
surface. The namespace exists, which is useful, but runtime consumption should
fail closed until a concrete adapter proves scorer-free behavior, deterministic
decode, runtime budget, and no hidden side-channel dependence.

## Highest-Value Remaining Closures

1. Packed/length-prefixed grammar projectors for PR106/HNeRV/PR107 in
   `tools/extract_master_gradient.py`; this keeps ITEM_3 open.
2. A field-equation planner extension that carries per-pair optimal-plan
   summaries into KKT rows without changing score authority.
3. A per-pair continual-learning adapter that records empirical anchors by
   pair/category/region/label without direct posterior mutation.
4. A concrete inflate-time adapter only after a runtime-safe target exists.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_master_gradient.py src/tac/tests/test_master_gradient_consumers.py src/tac/tests/test_master_gradient_consumers_rashomon.py src/tac/tests/test_cathedral_autopilot_autonomous_loop.py]`
  - Result: `251 passed in 1.39s`

## Non-Claim

This memo does not promote any master-gradient sidecar to contest authority.
Existing A1/PR101 materializations remain diagnostic/planning anchors unless a
byte-closed archive/runtime packet and matching exact-eval custody close the
loop.
