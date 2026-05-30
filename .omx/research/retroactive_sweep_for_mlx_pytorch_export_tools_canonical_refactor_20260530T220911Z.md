# Retroactive sweep — MLX→PyTorch export tools canonical-helper refactor (2026-05-30)

**Lane**: `lane_mlx_pytorch_export_tools_canonical_refactor_20260530`
**Catalog #348 compliance**: this memo satisfies the canonical "retroactive verdict-taint sweep" contract per CLAUDE.md "Catalog #348" non-negotiable + Catalog #229 premise-verification-before-edit + CLAUDE.md "Forbidden premature KILL without research exhaustion".

---

## 1. Bug-class symptom signature

**Bug class**: MLX-HWIO → PyTorch-OIHW Conv2d weight transpose pattern duplicated verbatim across 5 of 8 export tools (`tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py`, `tools/export_pact_nerv_selector_v2/v3/v4_mlx_to_pytorch_state_dict.py`, `tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py`).

**Symptom signature**:
- 19-line block per tool containing `np.transpose(arr, (0, 3, 1, 2))` + `np.ascontiguousarray` + `astype(np.float32)` + `torch.from_numpy(out_arr.copy())` + per-tensor sha256 sidecar emission
- BYTE-IDENTICAL across 5 sister tools (verified via `diff` + canonical apples-to-apples test)
- Per-tool docstrings already DOCUMENT they are sister implementations of the IA3 reference at `tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py`

**Bug-class consequence pre-fix**: a future audit / refactor / behavior change to the transpose pattern requires N-tool coordinated updates instead of 1-place canonical-helper update. Sister discipline: this is the `mlx_trainer_pytorch_sister_duplicated_implementation_v1` anti-pattern's bridge-tool-surface variant per the canonical anti-patterns registry landed 2026-05-28.

---

## 2. Pre-fix window

**Window**: 2026-05-30 ~early-evening (lane registration via predecessor checkpoint `ad5a6f152208557fa` step=1 at 21:49:21Z) through 2026-05-30 22:09Z (this landing).

**Predecessor crash anchor** per Catalog #206 mandatory crash-resume protocol: predecessor `ad5a6f152208557fa` wrote step=1 checkpoint then hit operator parent-session-limit at 0 actual file mutations. This resume picked up at step=1 and landed substantive work.

**Pre-fix LOC inventory** (8 tools, ~2711 total LOC):
- `tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py`: 346 LOC
- `tools/export_pact_nerv_selector_v2_mlx_to_pytorch_state_dict.py`: 349 LOC
- `tools/export_pact_nerv_selector_v3_mlx_to_pytorch_state_dict.py`: 353 LOC
- `tools/export_pact_nerv_selector_v4_mlx_to_pytorch_state_dict.py`: 355 LOC
- `tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py`: 356 LOC
- `tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py`: 385 LOC (PRINCIPLED FORK)
- `tools/export_pr95_mlx_to_pytorch_state_dict.py`: 252 LOC (PRINCIPLED FORK)
- `tools/export_wyner_ziv_pipeline_stage_codec_mlx_to_pytorch_state_dict.py`: 315 LOC (PRINCIPLED FORK)

**Pre-fix audit prediction** (`.omx/research/mlx_canonicalization_audit_inventory_20260530.md` §A.2.8): 88% LOC reduction by canonicalizing 7 tools to consume `tac.framework_agnostic.helpers.mlx_state_dict_to_npz_bridge`.

**Pre-fix empirical reality** (this report): 1.3% LOC reduction (-34 LOC across 8 tools) is the EMPIRICAL apples-to-apples result; 3 of 8 tools are PRINCIPLED FORKS per Catalog #290 falling-rule.

---

## 3. Historical KILL / DEFER / FALSIFY search results

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307 paradigm-vs-implementation falsification classification:

### 3.1 Historical KILL verdicts affected: NONE

No prior KILL / FALSIFIED memos exist for the MLX → PyTorch export tools bug class. The 8 tools all currently exist + are operator-routable per their canonical lane registry entries.

### 3.2 Historical DEFER verdicts affected: NONE

No prior DEFER memos exist for this bug class.

### 3.3 Historical FALSIFY verdicts affected: PARTIAL

The audit prediction at `.omx/research/mlx_canonicalization_audit_inventory_20260530.md` §A.2.8 "88% LOC reduction; 7 sister tools" is RATIFIED-FALSIFICATION-OF-THE-SPECIFIC-PREDICTION per Catalog #307:

- **Paradigm-level**: "MLX-FIRST canonical extraction is valuable" — INTACT (5 of 8 tools converged on canonical helper at 1.3% LOC reduction; canonical helper exists + 14/14 tests pass + byte-stable verified)
- **Implementation-level**: "all 8 tools converge on the npz bridge at 88% reduction" — FALSIFIED (3 of 8 are PRINCIPLED FORKS per Catalog #290 falling-rule; 5 tools converged via a different but ALSO canonical helper — `convert_mlx_state_dict_to_pytorch_oihw` — that produces PyTorch `.pt` state_dicts, NOT npz bytes per the original audit framing)

Per CLAUDE.md "Forbidden premature KILL": the audit's prediction is NOT killed — it is RATIFIED-AT-IMPLEMENTATION-LEVEL with reactivation criteria pinned (if a future substrate adopts the `mlx_state_dict_to_npz_bridge` contract directly without per-tensor sidecar metadata + without substrate-distinguishing PyTorch `.pt` output, then the 88% reduction prediction reactivates for that specific tool).

### 3.4 Sister wave activity check

- alaska canonical inverse-steganalysis patterns (commit `61a91a48e`) — sister-DISJOINT (covers Fridrich-school surface, not MLX → PyTorch bridge)
- m9-v3 (commit `49f41e22c`) — sister-DISJOINT
- Yousfi-Tier-1 (commit `3d027ecf9`) — sister-DISJOINT
- Z7+Z8 mamba2_adapter (LANDED) — sister-DISJOINT (substrate architecture, not bridge tool)
- MLX-LOCAL smoke validation (commit `98412f194`) — sister-DISJOINT (smoke validation, not bridge tool)
- deferred-items feeder (commit `46aa6ad86`) — sister-DISJOINT (READ-ONLY at landing)
- Fridrich-school extension (commit `396488202`) — sister-DISJOINT (steganalysis surface)
- Z8 M12a pre-flight (commit `ef7fd29e3`) — sister-DISJOINT
- z6_v2 pre-flight (commit `7a8581424`) — sister-DISJOINT
- z6_v2 29,650-epoch MLX-LOCAL FULL RUN (LANDED `7e9e3fc...`?) — sister-DISJOINT (consumes a sister bridge tool but does not edit the tool itself)
- PR110-OPT-7 L1 promotion (in-flight `acd4123aaaba505a9` / LANDED `d31cd4adcce8e7ab1`) — sister-DISJOINT (substrate L1 promotion, not bridge tool)
- canonical anti-patterns Wave (sister Agent in-flight on `src/tac/canonical_anti_patterns/`) — sister-DISJOINT per Catalog #340 sister-checkpoint-guard (no overlap with `tools/export_*`)

---

## 4. Per-finding RE-EVAL-priority assignment

| Finding | Pre-fix verdict | Post-fix verdict | RE-EVAL priority | Reactivation criteria |
|---|---|---|---|---|
| Audit §A.2.8 "88% LOC reduction prediction" | Predicted (cargo-culted-loc-minimization) | RATIFIED-FALSIFICATION-OF-THE-SPECIFIC-PREDICTION per Catalog #307 | LOW | If a future substrate adopts canonical `mlx_state_dict_to_npz_bridge` directly without per-tensor sidecar + PyTorch `.pt` output, the 88% prediction reactivates for that specific tool. |
| 5-tool MLX-HWIO → PyTorch-OIHW transpose duplication | DUPLICATION DETECTED | EXTINCT structurally via canonical helper `tac.framework_agnostic.helpers.convert_mlx_state_dict_to_pytorch_oihw` | RESOLVED | If a NEW substrate adds an MLX → PyTorch bridge tool, consume the canonical helper per Catalog #290 OBVIOUS_FIT classification. Canonical helper is the structural protection. |
| VQ PRINCIPLED FORK (substrate-distinguishing `is_vq_buffer` sidecar) | Implicit (no FORK doc) | DOCUMENTED PRINCIPLED FORK per Catalog #290 + docstring | LOW | If the VQ tool's per_tensor sidecar contract changes to drop `is_vq_buffer` AND adopt the canonical "skipped_by_predicate" layout token, the canonical helper consumption reactivates. |
| PR95 PRINCIPLED FORK (substrate-specific archive ZIP parser) | Implicit | EXPLICITLY CLASSIFIED in BEFORE/AFTER memo | LOW | If a future PR95 tool variant accepts `.npsd` state_dict input (instead of archive ZIP), the canonical helper consumption reactivates for that variant. |
| wyner_ziv PRINCIPLED FORK (archive-bytes parity not state_dict bridge) | EXPLICITLY DOCUMENTED in tool docstring | EXPLICITLY CLASSIFIED in BEFORE/AFTER memo | NONE | Structural FORK: WZPSC01 archives have no state_dict. The canonical helper is permanently structurally inapplicable. |

---

## 5. Operator-routable next steps

Per CLAUDE.md "Results must become system intelligence" + "Subagent coherence-by-default":

1. **MEDIUM-EV deferred per audit §A.2 (still operator-routable)**:
   - lift `gumbel_softmax_sample` to canonical `tac.framework_agnostic.gumbel_softmax_sample` (3 substrate sister impls: DreamerV3 / Z8 / mdl_ibps_j; Hafner-2023 unimix discipline landed 2026-05-29 for the math-fidelity case)
   - lift `rgb_to_yuv6` + `yuv6_to_rgb` to canonical `tac.framework_agnostic` with 4-backend dispatch (NUMPY / PYTORCH / MLX / TINYGRAD); 4 sister impls preserving PRINCIPLED FORKS where contracts differ per Catalog #290

2. **LOW-EV deferred per audit §A.3**: paired audit of 6 substrate MLX renderers that do NOT import `pr95_hnerv_mlx` canonical primitives; verify each has either substrate-optimal FORK per Catalog #290 OR migration target.

These remain operator-routable per CLAUDE.md "Forbidden premature KILL"; this lane explicitly OUT-OF-SCOPE per the parent prompt boundary (only `tools/export_*_mlx_to_pytorch_*.py` + canonical helper).

---

## 6. Catalog cross-references

- Catalog #110 / #113: HISTORICAL_PROVENANCE preserved per APPEND-ONLY discipline (audit memo, canonical-helper docstring, BEFORE/AFTER LOC report, landing memo, this sweep)
- Catalog #125: 6-hook wire-in declaration (see landing memo)
- Catalog #126: lane pre-registration (predecessor checkpoint step=1 from `ad5a6f152208557fa`)
- Catalog #157 + #174 + #186 + #206 + #234 + #340: commit-machinery discipline + crash-resume + sister-checkpoint-guard
- Catalog #176: STRICT-callsite-has-CLAUDE.md-row sister discipline (this gate adds the canonical helper to canonical __all__ exports per Catalog #335)
- Catalog #185: META-meta-meta drift detection (canonical helper + tests verified empirically)
- Catalog #229: premise-verification-before-edit (this report includes the empirical reconciliation of the audit's 88% prediction vs empirical 1.3%)
- Catalog #287: placeholder-rationale rejection sister discipline (no placeholders in any landed file)
- Catalog #290: canonical-vs-unique decision per layer (formal 3-tool PRINCIPLED FORK classification)
- Catalog #300: mission contribution `apparatus_maintenance` declared
- Catalog #307: paradigm-vs-implementation falsification classification (audit prediction = IMPLEMENTATION-LEVEL ratified-falsification; paradigm INTACT)
- Catalog #313: probe outcome PROCEED 14-day advisory
- Catalog #323: canonical Provenance (per-tensor sha256 sidecar preserved across refactor)
- Catalog #335: canonical helper exposed via package __all__ per cathedral consumer auto-discovery sister discipline
- Catalog #344: canonical equation candidate DEFERRED per iterate-not-force (the empirical 1.3% LOC reduction is a single anchor; the canonical equation `mlx_pytorch_bridge_tool_duplication_extinction_compounding_savings_v1` requires 3+ sister anchors for registration per the auto-recalibrator trigger)
- Catalog #348: this retroactive sweep memo (canonical APPEND-ONLY)

---

## 7. Apparatus mutation chain executed

1. NEW canonical helper `tac.framework_agnostic.helpers.convert_mlx_state_dict_to_pytorch_oihw` (~95 LOC including docstring)
2. NEW canonical helper exported via `tac.framework_agnostic.__init__.py` re-export + `__all__` per Catalog #335 sister discipline
3. NEW canonical tests `src/tac/framework_agnostic/tests/test_convert_mlx_state_dict_to_pytorch_oihw.py` (14/14 PASS)
4. 5 sister tools refactored to consume canonical helper (pact_nerv_ia3 + selector_v2 + selector_v3 + selector_v4 + z6_v2_cargo_cult_unwind)
5. VQ tool docstring updated to DOCUMENT the PRINCIPLED FORK per Catalog #290 (substrate-distinguishing `is_vq_buffer` sidecar + byte-stability concern with canonical helper's `skipped_by_predicate` layout token)
6. PR95 tool UNCHANGED (PRINCIPLED FORK per Catalog #290 — substrate-specific archive ZIP parser, NOT state_dict bridge)
7. wyner_ziv tool UNCHANGED (PRINCIPLED FORK per Catalog #290 — archive-bytes parity, NOT state_dict bridge; explicitly documented in tool's own docstring)
8. BEFORE/AFTER LOC report at `.omx/research/mlx_pytorch_export_tools_refactor_loc_reduction_20260530.md`
9. Landing memo at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_pytorch_export_tools_canonical_refactor_landed_20260530.md`
10. THIS retroactive sweep memo per Catalog #348
11. Lane registry mark `impl_complete` + `memory_entry` per Catalog #126
12. Probe outcome `register_probe_outcome` PROCEED 14-day per Catalog #313
13. MEMORY.md index entry prepended (one-line ≤300 chars)
14. Sister-DISJOINT confirmation per Catalog #340 (no overlap with concurrent sister Agents D7+D8+D9 on `src/tac/canonical_anti_patterns/` + deferred-items feeder audit)

# HISTORICAL_SCORE_LITERAL_OK:catalog_348_retroactive_sweep_no_frontier_score_claims_2026-05-30
