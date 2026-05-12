# Wiring Audit (Dimension W) — 2026-05-12

**Pass**: FFF — Wiring + Integration + Arbitrariness (W/I/A) sweep
**Scope**: `src/tac/`, `tools/`, `experiments/` — public symbols added in last 30 days + cross-checked against landed primitives
**Author**: subagent (FFF pass), date 2026-05-12

## Methodology

1. `git log --since="30 days ago" --diff-filter=A --name-only` → 2,991 new public symbols (functions + classes)
2. For each, count external references via ripgrep across `.py` files (excluding own module + paired test file)
3. Filter to ZERO external refs → 973 candidates
4. Cross-check against `__init__.py` re-exports + string-token usage + intra-file usage → reveals which are truly orphan vs. internal helpers
5. Final pass: keep only symbols with no references ANYWHERE in `.py/.md/.json/.sh/.yaml` AND not even used inside their own def file

## Headline counts

| Bucket | Count |
|---|---|
| New public symbols (last 30d) | 2,991 |
| Zero external refs (.py only) | 973 |
| Zero refs (all source-type files) | 328 |
| Zero refs AND not used inside own file | **14** |

## True orphan candidates (14)

| Symbol | File:Line | Kind | Status / Proposed action |
|---|---|---|---|
| `verify_checkpoint_quality` | `src/tac/checkpoint.py:105` | function | PUBLIC-API-INTENDED (utility) — DEFERRED-pending-consumer-audit |
| `write_plan_manifest` | `src/tac/codec_stack_planner.py:1070` | function | PUBLIC-API-INTENDED (planner emit) — DEFERRED-pending-consumer-audit |
| `ControlSummationAdapter` | `src/tac/contrib/multi_control_hint_encoder.py:127` | class | PUBLIC-API-INTENDED (contrib namespace) — KEEP |
| `EC2InstanceSpec` | `src/tac/deploy/aws/ec2_client.py:119` | class | PUBLIC-API-INTENDED (AWS deploy contract) — KEEP |
| `adapt_for_aws` | `src/tac/deploy/aws/experiments.py:13` | function | PUBLIC-API-INTENDED (AWS deploy contract) — KEEP |
| `iter_layer_pairs` | `src/tac/frozen_bit_quant.py:294` | function | WIRE-NEEDED — quantization helper without consumer; could feed bit_allocator |
| `config_from_mapping` | `src/tac/hnerv_arch_schema.py:286` | function | PUBLIC-API-INTENDED (HNeRV schema) — KEEP |
| `load_pr101_archive_state_dict` | `src/tac/pr101_archive_state_loader.py:157` | function | PUBLIC-API-INTENDED (intake loader) — KEEP |
| `write_range_contract_json` | `src/tac/pr91_hpm1_range_contract.py:1040` | function | PUBLIC-API-INTENDED (contract emit) — KEEP |
| **`check_pose_fit_module_has_white_noise_test`** | `src/tac/preflight.py:27363` | function | **WIRE-NEEDED — preflight check never called from `preflight_all`. Live count: 0. Strict-safe.** |
| **`check_preflight_hook_supports_changed_files_mode`** | `src/tac/preflight.py:27493` | function | **WIRE-NEEDED — preflight check never called from `preflight_all`. Live count: 0. Strict-safe.** |
| **`check_renderer_codec_has_posenet_protection`** | `src/tac/preflight.py:27171` | function | **WIRE-NEEDED — preflight check never called from `preflight_all`. Live count: 27 (warn-only).** |
| `fp8_format_metadata` | `src/tac/quantization_fp8.py:365` | function | PUBLIC-API-INTENDED (metadata emit) — KEEP |
| `compute_rate_term` | `src/tac/submission_archive.py:1547` | function | WIRE-NEEDED — rate-term helper appears post-PR108; consumer audit needed |

## High-leverage findings

### W-1 (HIGH): three preflight checks defined but not wired into `preflight_all()`

Three preflight check functions exist in `src/tac/preflight.py` but are never called from `preflight_all()` (the canonical entry point). Live counts:

- `check_pose_fit_module_has_white_noise_test` — strict=True default in signature, 0 live violations → SAFE TO WIRE STRICT
- `check_preflight_hook_supports_changed_files_mode` — strict=True default in signature, 0 live violations → SAFE TO WIRE STRICT
- `check_renderer_codec_has_posenet_protection` — strict=False default, 27 live violations → SAFE TO WIRE WARN-ONLY

All three docstrings claim "Lands STRICT @ 0 violations" or "Lands WARN-ONLY initially. Promotion plan: per-module owner adds the appropriate tag, then flip strict=True". The promotion never happened — they were defined but not wired.

**Bug class**: classic "landed but not wired" failure mode. Sister to Catalog #151 (which catches it for TRAINER flag manifests) — but for PREFLIGHT CHECKS the symptom is dormant guards: a future regression in pose_fit / preflight_hook / renderer_codec would not be caught.

**Proposed fix**: wire the two zero-violation checks STRICT into `preflight_all()`, wire the warn-only one with `strict=False`. Trivial — adds 3 lines. Lands in this pass.

### W-2 (MEDIUM): packet_compiler primitives pr63/pr64/pr65/pr105 only consumed by tests

Per `feedback_packet_compiler_5_pr63_64_65_105_primitives_landed_20260512.md`, these primitives landed as port-byte-faithful from public PR intakes:

- `pr63_qpose14_codec` — only `__init__.py` re-export + test file
- `pr64_unified_brotli_pose_velocity` — only `__init__.py` re-export + test file
- `pr65_pq12_pose_codec` — only `__init__.py` re-export + test file
- `pr105_packed_state_schema` — only `__init__.py` re-export + test file

The mining-backlog row (`tools/build_public_pr_mining_expansion_backlog.py`) names them but no archive builder consumes them yet. The accompanying landing memo explicitly tagged `score_claim=False`, `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`, which is the correct posture per CLAUDE.md `forbidden_score_claim_with_byte_change_unless_inflate_consumes`.

**Status**: NOT-A-BUG — this is correct evidence discipline. The primitives are the BUILDING BLOCKS that downstream PacketIR composition / archive builders will consume. Surfaced here for completeness so the orphan signal isn't surprising in future audits.

### W-3 (LOW): autopilot autonomous loop claims docstring posterior wire-in but lacks the import

`tools/cathedral_autopilot_autonomous_loop.py:70` docstring says:
> :mod:`tac.continual_learning` — posterior consumed and updated by the loop

But `grep "from tac.continual_learning\|posterior_update_locked"` against the file returns 0 matches. The constant `POSTERIOR_REWEIGHT = "posterior_reweight"` (line 126) is defined but never referenced anywhere else in the file (or in any other repo file).

This is a real integration gap — see `integration_gap_audit_20260512.md` I-1 finding. Logged here for cross-reference.

### W-4 (NOT-A-BUG): 943 internal-helper symbols flagged by naive scan

A naive cross-file scan flagged 943 zero-ref symbols. After filtering for:

- intra-file use (`int_or_none`, `first_numeric`, helpers used in the same module via method calls)
- string-token references (registry lookups)
- `__init__.py` re-exports

…the genuine-orphan count dropped to 14. The original 943 is a false-positive trap that future audits should avoid by routing through the v3 scanner in this audit (see `wiring_audit_data_20260512.json` for the full data).

## Recommended actions

**LAND NOW (trivial, ≤ 5 LOC per fix)**:
- Wire 2 zero-violation preflight checks STRICT into `preflight_all()` (W-1.a, W-1.b)
- Wire 1 warn-only preflight check into `preflight_all()` (W-1.c)

**SURFACE FOR OPERATOR DECISION**:
- W-2 packet_compiler orphans: do we name a target archive builder for pr63/pr64/pr65/pr105 in the Phase 2 / Phase 3 plan, or do they stay as building blocks? Council-level question.
- W-3 autopilot ↔ continual-learning bridge: should it actually consume the posterior, or should the docstring be corrected? See I-1.

**DEFERRED-pending-consumer-audit**:
- `verify_checkpoint_quality`, `write_plan_manifest`, `iter_layer_pairs`, `compute_rate_term` — each needs a sibling-tool consumer audit; tagging as DEFERRED rather than DELETE per CLAUDE.md "KILL is LAST RESORT".

## Wire-in hook declarations (per CLAUDE.md Catalog #125)

1. **Sensitivity-map**: N/A — wiring audit does not touch sensitivity scoring.
2. **Pareto constraint**: N/A — no Pareto constraint added or relaxed.
3. **Bit-allocator**: N/A — `iter_layer_pairs` orphan call-out IS a potential bit-allocator feed, but no fix lands here.
4. **Cathedral autopilot dispatch hook**: relevant (W-3) — autopilot's docstring claim of posterior wire-in is the surfaced gap.
5. **Continual-learning posterior**: relevant (W-3 + I-1) — no posterior write in this audit (it's a meta audit).
6. **Probe-disambiguator**: N/A — findings are unambiguous code-presence; no math arbitration needed.

## References

- Data: `.omx/research/wiring_audit_data_20260512.json` (full 2,991-symbol scan)
- Refined: `.omx/research/wiring_audit_refined_20260512.json` (973 zero-ref after __init__/string filter)
- True orphans: `.omx/research/wiring_audit_truly_orphan_20260512.json` (328 candidates)
- External orphans: `.omx/research/wiring_audit_external_orphans_20260512.json` (14 candidates)
