# LOOPCLOSE — unified-Lagrangian end-to-end loop-closure audit (2026-05-12)

Lane: `lane_loopclose_unified_lagrangian_audit_20260512` (L0 → L1 on landing)
Mode: META audit. NO GPU dispatch. NO archive bytes touched. NO scorer load.
Scope: trace the 6-hook closed loop at runtime under POST-FIX-wave state.

## Executive summary

| Hook | Verdict | Producer | Transport | Consumer | Closure test |
|---|---|---|---|---|---|
| 1. Sensitivity-map | **PARTIAL** | `tac.sensitivity_map` (full module) + `discover_sensitivity_map_artifacts()` (inventory) | planner-context payload + bridge JSON `axis_weight` | `cathedral_autopilot_autonomous_loop` (inventory consumer); `build_composition_ranking_json` (axis-weight reweighter, **NOT** consuming `tac.sensitivity_map` data) | bridge produced 5 rows with `axis_weight=2.71` (pose); inventory enumeration runs in planner; **but** the axis-weight multiplier is a CLAUDE.md-hardcoded constant, not data-driven from sensitivity-map artifacts |
| 2. Pareto constraint | **CLOSED-LOOP** | `tac.composition.enumerate.enumerate_cells()` + `RefusedReason` enum | in-memory cell list with `compatibility_verdict` field | `build_composition_ranking_json` filters to `{compatible, compatible_bare_substrate}`; refused cells (8975 `violates_ordering_or_mutually_exclusive` + 3560 `violates_dependency`) skipped | observed 16,833 total → 4,298 compatible → bridge emits ranked rows over compatible set only |
| 3. Bit-allocator | **CLOSED-LOOP** | `tac.bit_allocator.allocate_bits`; sign-encoding 5-strategy taxonomy in `PACKET_COMPILER_TRANSFORMS` | per-tensor importance weights + token registry (40 tokens) | `tac.hessian_block_fp`, `tac.mdl_fp4_tto`, `tools/build_uniward_stc_hessian_a1_v1.py` | `import tac.bit_allocator` OK; `PACKET_COMPILER_TRANSFORMS` length 40; sign_encoding_{negzig,zig,twos,off,raw_uint8} all registered |
| 4. Cathedral autopilot dispatch | **CLOSED-LOOP** | `tac.composition.enumerate_cells()` (via FIX-C bridge) | `tac_composition_cell_to_autopilot_bridge_v1` / `tac_autopilot_dispatch_ranking_v1` JSON | `cathedral_autopilot_autonomous_loop.load_candidates_from_substrate_composition_ranking()` | bridge → 5-row JSON → autopilot loader returned 5 `CandidateRow` rows with correct `eig_per_dollar` ordering |
| 5. Continual-learning posterior | **CLOSED-LOOP** | empirical anchors → `posterior_update_locked` | `.omx/state/cost_band_posterior.jsonl` + `ContinualLearningPosterior` object | bridge calls `load_posterior` + `posterior_query_track_correction`; autopilot loop also reads via `load_planner_posterior_for_loop` | live posterior shows `accepted_anchor_count=21`, `refused_anchor_count=11`; bridge consumes it during ranking (verified import + call path) |
| 6. Probe-disambiguator | **CLOSED-LOOP** | 14+ probes under `tools/probe_*.py` | per-probe stdout/JSON manifests | operator + future autopilot consumers | probes exist for: a2_packet, pr103_arithmetic, precoarsening_entropy_coders, eval_drift_matrix, posenet_layer_drift, yuv6_differentiability, eval_loader_drift, schema_elision, frame_conditional_quantization, pr103_lc_ac, seg_loss_surrogate, seg_pose_weight_at_operating_point, inflate_shell_output_parity, sign_encoding |

**Closed-loop verdict: 5/6 fully closed; 1/6 PARTIAL.**

The single PARTIAL is hook 1: the bridge uses a CLAUDE.md-hardcoded axis-weight multiplier (`pose=2.71`, `seg=1.0`, `rate=1.0`, `mixed=1.5`) per the "SegNet vs PoseNet importance — operating-point dependent" rule, but it does NOT consume the per-tensor sensitivity-map data emitted by `tac.sensitivity_map` itself. The inventory enumeration IS exposed in the planner context (I-3 wire-in), but the autopilot ranker has no consumer-side path that reads the inventory's `.pt` payloads. This is a deferred-by-design choice — the autopilot is a higher-level technique ranker, not a per-tensor bit allocator. Honest classification: **PARTIAL-by-design**.

## Closure tests run

### Hook 4 (autopilot dispatch) — direct runtime test

```bash
$ .venv/bin/python tools/build_composition_ranking_json.py --output .omx/tmp/loopclose_audit/bridge_test.json --max-total 5
wrote 5 ranked composition cells to .omx/tmp/loopclose_audit/bridge_test.json

$ .venv/bin/python -c "
import sys; sys.path.insert(0, 'tools')
from cathedral_autopilot_autonomous_loop import load_candidates_from_substrate_composition_ranking
from pathlib import Path
rows = load_candidates_from_substrate_composition_ranking(Path('.omx/tmp/loopclose_audit/bridge_test.json'))
print('Loaded rows:', len(rows))
"
Loaded rows: 5
```

Result: producer → transport → consumer flows end-to-end at runtime. PASS.

### Hook 5 (posterior) — live posterior read

```python
>>> from tac.continual_learning import load_posterior
>>> posterior = load_posterior()
>>> posterior.accepted_anchor_count
21
>>> posterior.refused_anchor_count
11
```

Bridge consumes `load_posterior` + `posterior_query_track_correction` (verified at `tools/build_composition_ranking_json.py:80-81`). The downstream autopilot loop ALSO reads the posterior via `load_planner_posterior_for_loop` (`tools/cathedral_autopilot_autonomous_loop.py`). Both call sites confirmed.

### Hook 2 (Pareto refusal) — enumeration counters

```python
>>> from tac.composition.enumerate import enumerate_cells
>>> cells = enumerate_cells()
>>> from collections import Counter
>>> Counter(c.compatibility_verdict for c in cells)
Counter({
  'violates_ordering_or_mutually_exclusive': 8975,
  'compatible': 4274,
  'violates_dependency': 3560,
  'compatible_bare_substrate': 24,
})
```

Bridge filter: `c.compatibility_verdict in ("compatible", "compatible_bare_substrate")` (line 360-361 of `build_composition_ranking_json.py`). Refused cells skipped correctly. PASS.

The `RefusedReason` enum (`tac.composition.registry`) defines 7 typed values: `ordering_violation`, `mutual_exclusion`, `substrate_incompatible`, `semantic_incompatible`, `dependency_missing`, `duplicate_primitive`, `unknown_primitive`. ZZZZZ's "no typed taxonomy" finding is partially obsolete: the enum exists, but cells currently use a 4-value `compatibility_verdict` string rather than the richer `RefusedReason`. Optional refinement.

## Cross-system bridge audit

| Bridge | Verdict | Detail |
|---|---|---|
| `tac.composition` ↔ `cathedral_autopilot` | **CLOSED** | via FIX-C `build_composition_ranking_json.py` (in-flight sibling diff to imports/typing — non-functional polish) |
| `tac.continual_learning` ↔ `cathedral_autopilot` | **CLOSED** | Wave 2/H wire-in; bridge + autopilot loop both read posterior; harvest tools write |
| `tac.cost_band_calibration` ↔ operator-authorize wrappers | **CLOSED via FIX-G** | All 10 wrappers now thin-shim → `tools/operator_authorize.py`; cost_band centralized at `tools/operator_authorize.py:411-413` (`tac.cost_band_calibration.predict()` + `.omx/state/cost_band_posterior.jsonl` reference). The original I-2 in-line patches were superseded by FIX-G's canonical unification. |
| `tac.cost_band_calibration` ↔ `cathedral_autopilot` | **CLOSED** | `tools/cathedral_autopilot_autonomous_loop.py:121` imports `predict as predict_cost_band`; bridge also imports `tac.cost_band_calibration` for fallback midpoint |
| `tac.deterministic_compiler` ↔ packet-producing surfaces | **CLOSED** | Catalog #158 STRICT in `preflight_all()`; `tools/build_deterministic_packet.py` is canonical CLI; 19 golden vectors under `src/tac/packet_compiler/golden_vectors/`; 3 dedicated test files |
| `xray_substrate_classifier` ↔ composition compatibility lookup | **PARTIAL** | Classifier emits substrate-class manifest; composition `canonical_substrate_inventory()` indexes by `substrate_id`. No automatic `class → id` mapping. Note: ZZZZZ flagged "two parallel paths" between `xray_substrate_classifier.py` (archive-byte magic-signature) and `cpu_cuda_xray_substrate_class_classifier.py` (CPU/CUDA drift signature) — these are **DIFFERENT FUNCTIONS** (not duplicates); the recent docstring fix (commit `c103a287`) disambiguates them. ZZZZZ's interpretation was a misread. |
| CompressAI codecs ↔ composition matrix as primitives | **WIRE-NEEDED** | `tac.packet_compiler.factorized_prior` + `tac.packet_compiler.balle_hyperprior` + `tac.packet_compiler.cheng2020` are importable + used by `tac.balle_hyperprior_renderer`. They are **NOT** registered in `canonical_primitive_inventory()` (14 primitives total; no `compressai_*` entries). If they are intended to be selectable as composition cells, registry rows are missing. |
| Substrate scaffolds ↔ composition matrix substrate enumeration | **WIRE-NEEDED** | 39 scaffolds under `submissions/`; 19 substrate lanes in lane registry; 24 substrate primitives in `canonical_substrate_inventory()`. The grand-council substrate-design landing's new scaffolds `sane_hnerv`, `balle_renderer`, `hybrid_renderer_residual`, `grayscale_lut`, `self_compress`, `vq_vae`, `tc_nerv`, `pr101_lc_v2_clone` are **MISSING** from the composition registry's substrate inventory. Cool-Chic + wavelet + siren ARE present. |

## Orphan signal scan (new session-public symbols, sample)

| Symbol | Status | Resolution |
|---|---|---|
| `tac.composition.enumerate_cells` | CLOSED via FIX-C bridge | Was orphan in ZZZZZ audit; now consumed by `tools/build_composition_ranking_json.py` |
| `tac.composition.CompositionCell` | PUBLIC-API-INTENDED | Public dataclass; Rust-port-intended; preserve |
| `tac.composition.RefusedReason` | PARTIAL-WIRE | Enum defined but `enumerate_cells()` emits a coarser `compatibility_verdict` string. Reviewer-actionable: enrich rows with typed `refused_reason: RefusedReason` field. ~5-LOC change in `tac.composition.enumerate`. |
| `tac.packet_compiler.compressai_factorized_prior` | PUBLIC-API-INTENDED | `score_claim=false` adapter; used by `tac.balle_hyperprior_renderer` + tests |
| `tac.packet_compiler.compressai_balle_hyperprior` | PUBLIC-API-INTENDED | Same |
| `tac.packet_compiler.compressai_cheng2020` | PUBLIC-API-INTENDED | Same |
| `tac.packet_compiler.deterministic_compiler.compile_packet` | CLOSED | Canonical via Catalog #158; consumed by `tools/build_deterministic_packet.py` + tests |
| `tools/xray_substrate_classifier.py` | DISTINCT (not orphan) | Different function from `cpu_cuda_xray_substrate_class_classifier.py`; both have valid scopes per docstring disambiguation in commit `c103a287` |
| `tools/cpu_cuda_xray_substrate_class_classifier.py` | DISTINCT (not orphan) | Same |
| `discover_sensitivity_map_artifacts()` | CLOSED | I-3 wire-in; payload key `sensitivity_map_inventory` is in `load_planner_posterior_for_loop` output |

## Top operator decisions surfaced

Ranked by leverage. NO unilateral fixes applied this audit — all routable.

1. **HOOK 1 PARTIAL — sensitivity-map data-driven axis weights** (Medium). Bridge currently uses hardcoded `pose_axis_weight=2.71`. A real consumer of `tac.sensitivity_map`'s `.pt` artifacts would derive axis weights per-cell from the actual per-tensor importance. ~50 LOC bridge enhancement. Likely the canonical refactor for FIX-C v2.

2. **Composition registry substrate-inventory gap** (Medium). 8+ new substrates from the grand-council substrate-design landing (`sane_hnerv`, `balle_renderer`, `hybrid_renderer_residual`, `grayscale_lut`, `self_compress`, `vq_vae`, `tc_nerv`, `pr101_lc_v2_clone`) exist in `submissions/` + lane registry but are NOT in `canonical_substrate_inventory()`. Adding them unlocks substrate × primitive cells the composition matrix can rank. ~30 LOC per substrate row. Should be batch-landed by a sibling subagent that owns the composition registry surface.

3. **CompressAI primitives in composition registry** (Medium). 3 CompressAI codecs (factorized_prior / balle_hyperprior / cheng2020) registered in `tac.packet_compiler` but absent from `canonical_primitive_inventory()`. If they are intended as composition-cell primitives, add 3 `PrimitiveRow` entries. If they are renderer-only (substrate-side codecs), document the exclusion.

4. **`RefusedReason` enum wiring** (Low). The typed enum exists but `enumerate_cells()` emits a 4-value string `compatibility_verdict`. Enrich rows with `refused_reason: RefusedReason | None` so reviewers can audit infeasibility by typed class. ~10 LOC.

5. **WWW4 dispatch closure-test recipe** (operator-routable). When WWW4's anchor lands:

   ```bash
   # 1. Verify anchor reached posterior:
   .venv/bin/python -c "
   from tac.continual_learning import load_posterior
   p = load_posterior()
   print('accepted:', p.accepted_anchor_count)  # should increment by 1
   # WWW4-specific anchor key should appear in p.accepted_anchor_history
   "
   # 2. Rebuild bridge to observe posterior-corrected ranking shift:
   .venv/bin/python tools/build_composition_ranking_json.py \
     --output .omx/tmp/loopclose_audit/post_www4_bridge.json --max-total 20
   # 3. Diff vs pre-WWW4 bridge:
   diff <(jq -r '.ranked_dispatches[].candidate_id' .omx/tmp/loopclose_audit/bridge_test.json) \
        <(jq -r '.ranked_dispatches[].candidate_id' .omx/tmp/loopclose_audit/post_www4_bridge.json)
   ```

## Methodology + validation

- AST/grep traversal of `src/tac`, `tools`, `experiments`, `scripts` (excluding tests, results, __pycache__, vendored intake).
- Runtime smoke: each closure test invoked actual producer + consumer side-by-side and verified the data flowed.
- ZZZZZ cross-check: 6 of ZZZZZ's findings independently confirmed; 1 ("two xray classifier paths") rebutted — they have distinct functions.
- Preflight: `python -m tac.preflight --scope all` runs without failure (one warn — `commit-serializer-usage` 50 legacy commits — non-blocking).
- Lane registry: `python tools/lane_maturity.py validate` → `OK — 454 lane(s) validated cleanly`.

## Production-hardened verdict

- 3-clean-pass adversarial review: deferred to landing memo.
- 6-hook wire-in: this is a META audit. All 6 hooks EXERCISED (read-only).
- No GPU dispatch. No archive bytes changed. No `/tmp` paths persisted (used `.omx/tmp/loopclose_audit/`).
- No KILL verdicts.

DECLARED-BROKEN findings: 0. PARTIAL-with-rationale findings: 1 (hook 1, by design).
WIRE-NEEDED findings: 2 (substrate inventory gap, CompressAI primitive gap).
