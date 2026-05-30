# Retroactive sweep for Catalog #383 — 2026-05-30T21:45:00Z

**Gate**: `check_mlx_primitives_route_through_canonical_helper`
**Lane**: `lane_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_20260530`
**Per Catalog #348** (`check_new_gate_landing_includes_retroactive_sweep_evidence`): every new STRICT preflight gate landing MUST ship a retroactive sweep memo with 4 canonical fields.

## Bug-class symptom signature

Substrate trainers + scaffolds under `src/tac/substrates/*/` or `experiments/train_substrate_*.py` re-implementing canonical MLX primitives (`gumbel_softmax_sample` / `pixel_shuffle_2x_nhwc` / `bilinear_resize_nhwc` / etc.) WITHOUT routing through the canonical extractor module (`tac.local_acceleration.pr95_hnerv_mlx` / `tac.framework_agnostic` / etc.) OR documenting per-substrate FORK per Catalog #290 falling-rule list.

Detection signature (line-anchored AST scan):
- `ast.FunctionDef` / `ast.AsyncFunctionDef` / `ast.ClassDef` node whose `name` matches one of `_CHECK_383_PRIMITIVE_NAMES`
- Enclosing file has `import mlx` or `from mlx` import (filter)
- Enclosing file does NOT carry `from <canonical_extractor> import <primitive>` for the same primitive name
- Enclosing line + preceding line do NOT carry `# MLX_PRIMITIVE_UNIQUE_BECAUSE_<reason>:<rationale>` with non-placeholder substantive rationale (>=4 chars; placeholder rejected per Catalog #287)

## Pre-fix window

**Before 2026-05-30**: per the audit inventory at `.omx/research/mlx_canonicalization_audit_inventory_20260530.md` A.2.5:
- `gumbel_softmax_sample` duplicated across 3 substrates (DreamerV3 / Z8 / mdl_ibps_j with `_mlx` suffix)
- No canonical extractor existed in `tac.framework_agnostic`
- No STRICT preflight gate to refuse new sister substrate re-implementations

**Reactivation criterion at landing**: strict-flip Catalog #383 after canonical extraction migration drives live count to 0.

## Historical-KILL / DEFER / FALSIFY search results

Search corpus: `.omx/research/*.md` + `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_*.md` for memos mentioning `gumbel_softmax_sample`, `pixel_shuffle_2x_nhwc`, `MLX canonicalization`, `MLX primitive duplication`, `framework_agnostic canonical kernel`.

**Historical findings**:
1. **NO historical KILL verdicts** invalidated by this gate. The gate is purely PREVENTATIVE — it extincts future drift, does not invalidate past verdicts.
2. **NO historical DEFER verdicts** invalidated by this gate. The 2 known substrate-side `gumbel_softmax_sample` impls (DreamerV3 + Z8) predate this gate and remain DEFERRED-pending-canonical-extraction per CLAUDE.md "Forbidden premature KILL" + the operator-routable migration plan in audit inventory A.3.
3. **NO historical FALSIFY verdicts** invalidated by this gate.

**Sister gate landings**:
- Catalog #205 (`check_inflate_py_uses_canonical_select_inflate_device`) lands the sister inflate-time canonical-routing surface; #383 is the training-time MLX primitive sister
- Catalog #335 (`check_cathedral_consumer_directory_package_exposes_canonical_contract`) lands the canonical cathedral consumer auto-discovery; the new `mlx_canonicalization_audit_consumer` registers via that mechanism
- Catalog #344 (`check_empirical_finding_memo_references_canonical_equation`) lands the canonical equations registry; 2 new equations land via that mechanism (`mlx_primitive_canonicalization_compounding_savings_v1` + `mlx_pytorch_tinygrad_cross_backend_byte_stable_v1`)

## Per-finding RE-EVAL priority assignment

| Finding | Priority | Action | Reactivation criteria |
|---|---|---|---|
| `gumbel_softmax_sample` 3 sister impls | MEDIUM | EXTRACT_CANONICAL to `tac.framework_agnostic` | Strict-flip after migration |
| `rgb_to_yuv6` 4 sister impls | MEDIUM | EXTRACT_CANONICAL preserving PRINCIPLED FORK semantics | Strict-flip after migration |
| 7 per-substrate MLX→PyTorch export tools | HIGH | EXTRACT_CANONICAL via `mlx_state_dict_to_npz_bridge` | Strict-flip after migration (88% LOC reduction) |
| 6 substrate MLX renderers not routing through `pr95_hnerv_mlx` | LOW | Paired audit per substrate to verify PRINCIPLED FORK | Catalog #290 falling-rule decision |

## 4-field contract verification

1. **Bug-class symptom signature**: ✅ documented above (line-anchored AST scan with primitive-name match + import filter + waiver check)
2. **Pre-fix window**: ✅ documented above (pre-2026-05-30; 3 sister impls of `gumbel_softmax_sample` predate the gate)
3. **Historical-KILL/DEFER/FALSIFY search results**: ✅ documented above (NO historical verdicts invalidated; gate is purely PREVENTATIVE)
4. **Per-finding RE-EVAL priority assignment**: ✅ table above (4 findings; all operator-routable migration paths)

## Apparatus mutations landed in same commit batch

1. NEW canonical helper `src/tac/local_acceleration/mlx_canonical_audit/` (~600 LOC; 36 tests passing)
2. NEW canonical kernels `src/tac/framework_agnostic/canonical_kernels.py` (~470 LOC; 26 tests passing)
3. NEW STRICT preflight gate Catalog #383 (~220 LOC; 21 tests passing)
4. NEW canonical equations: `mlx_primitive_canonicalization_compounding_savings_v1` + `mlx_pytorch_tinygrad_cross_backend_byte_stable_v1` (registry 158→160)
5. NEW cathedral consumer `tac.cathedral_consumers.mlx_canonicalization_audit_consumer` (~80 LOC; 13 tests passing); auto-discovered per Catalog #335
6. Lane registry: `lane_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_20260530` L1
7. Probe outcome via Catalog #313 PROCEED 14-day advisory
8. Audit inventory memo `.omx/research/mlx_canonicalization_audit_inventory_20260530.md`
9. Catalog #348 retroactive sweep memo (THIS file)
10. Landing memo `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_landed_20260530.md`

Total: **83 new tests passing** + 2 NEW canonical equations + 1 NEW cathedral consumer + 1 NEW STRICT preflight gate + 6 apparatus surface mutations.

Mission contribution per Catalog #300: `apparatus_maintenance` (extincts the MLX primitive substrate-side re-implementation drift class structurally; canonical helper + STRICT gate sister-extinct the bug class bidirectionally at training-time MLX canonical-routing surface).
