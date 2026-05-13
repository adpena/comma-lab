# CANON-1.D — Hand-curated list anti-pattern systematic audit (META, 2026-05-12)

**Source ledger**: `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
CANON-1.D ("Hand-curated list anti-pattern systematic audit").

**Lane**: `lane_canon_b_c_remaining_decisions_20260512` (L1, Phase 2).

**Operator directives**:
- 2026-05-12 "canonicalize and clean up and deduplicate"
- 2026-05-12 "engineer for maintainability and useful and composable and discoverable and abstractions and extendable and stackable"
- 2026-05-12 lessons-learned Lesson 2: "Hand-curated lists are where the (N+1)th bug hides"

**Scope**: META audit — identify hand-curated lists in `src/tac/` and
`tools/` that are candidates for discovery-based replacement, classify each by
expected leverage / risk, recommend follow-up work for the high-leverage ones.
**No code modified**; recommendations only.

## Methodology

Static analysis of `src/tac/` and `tools/` for module-level constants matching
hand-curated-list patterns:

```
grep -rln '^[A-Z_]+_LIST\s*=\|^[A-Z_]+_TOKENS\s*=\|^[A-Z_]+_ROWS\s*=\|^[A-Z_]+_REGISTRY\s*='
```

then per-candidate manual triage by:
1. Is the list the canonical source-of-truth? (yes = candidate)
2. Are there N+ consumers that must agree with the source-of-truth?
3. Could the consumer INTROSPECT the source instead of duplicating it?
4. What is the empirical bug-class cost (this session: $0.016 for Modal mount
   list, ~30-min wall-clock for fixture-set test_known_transform_tokens)?

## Audit findings

### HIGH leverage (recommend follow-up work)

---

**HCL-1: `src/tac/preflight.py` catalog #N entries**

| Field | Value |
|---|---|
| Source-of-truth | `src/tac/preflight.py` orchestrator wire-in (`preflight_all()`) + each `def check_<name>(...strict=...)` definition |
| Hand-curated list | The CLAUDE.md "Meta-bug class catalog (strict-mode preflight)" table (~75+ entries with text describing each check, strict status, evidence count) |
| N consumers that must agree | 3+: (a) operator-facing CLAUDE.md table; (b) `preflight_all()` wire-in callsite list; (c) Catalog #159 `check_claude_md_catalog_text_matches_preflight_strict_value` (which auto-detects drift but cannot regenerate the table) |
| Existing self-protection | Catalog #118 (no-duplicate-numbers) + Catalog #159 (text matches code strictness) |
| Bug class | Drift between CLAUDE.md description and actual `strict=` value (UUU audit found 23 entries pre-WAVE-A-2; bulk text-fix cleared 8 in WAVE-A-2). The remaining ~15 entries land in CANON-1.K (separate ledger row); META-class is THIS one. |
| Recommendation | **Replace bulk text fix with a generator**: write `tools/generate_catalog_md_table.py` that introspects `src/tac/preflight.py` (via AST) for every `check_<name>` function + its `preflight_all()` wire-in `strict=` value, and emits the canonical CLAUDE.md catalog block. Operator runs the generator; CLAUDE.md is REWRITTEN; Catalog #118 + #159 verify integrity. |
| Cost | ~150 LOC tool + ~30 LOC integration test |
| Risk if not implemented | Catalog drift will recur as new strict-flips happen (the same 23-drift state UUU caught) |
| Status | DEFERRED-pending-operator-approval (touches src/tac/preflight.py introspection — currently safe; new tool can land independently) |

---

**HCL-2: `_REPRESENTATION_LANE_TOKENS` in Catalog #124**

| Field | Value |
|---|---|
| Source-of-truth | The 15 substrate-scaffold subpackages under `src/tac/substrates/<name>/` + the `<name>_as_renderer.py` modules under `src/tac/` (TRADITION 2). |
| Hand-curated list | `src/tac/preflight.py` `_REPRESENTATION_LANE_TOKENS` (case-insensitive substrate-family tokens like `nerv`, `hnerv`, `cool_chic`, `c3`, `wavelet`, `vqvae`, ...). Per CLAUDE.md Catalog #124. |
| N consumers | 1 (Catalog #124 STRICT scanner) — but WHEN a new substrate family lands, this token list MUST be updated or the gate silently misses the new family. |
| Bug class | New substrate family lands (e.g., "diffusion_renderer" added 2026-05-11), token list NOT updated, Catalog #124 silently passes the new lane through without checking the 8 archive-grammar fields. |
| Recommendation | **Discovery via `canonical_substrate_inventory()`**: Catalog #124 should iterate the inventory's `substrate_id` set + extract token roots, rather than maintain a hand-curated regex token list. Then ANY substrate appearing in the inventory automatically gates through Catalog #124. |
| Cost | ~30 LOC inside `check_representation_lane_has_archive_grammar_at_design_time` |
| Risk if not implemented | (N+1)th substrate family slips past Catalog #124. Already empirical (diffusion_renderer + dp_sims_renderer + mlx_mask_renderer added in WAVE-A-2 inventory — verify token coverage). |
| Status | DEFERRED-pending-preflight-edit-window (in-flight Modal source-staleness subagent has preflight.py dirty per WAVE-CANON-B-C parent-coordinator scope) |

---

**HCL-3: `tac.composition.registry.PRIMITIVE_ROWS` (extended FIX-D)**

| Field | Value |
|---|---|
| Source-of-truth | The packet-compiler primitive set in `tac.packet_compiler.PACKET_COMPILER_TRANSFORMS` + composition-ready substrates from `canonical_substrate_inventory()` |
| Hand-curated list | `src/tac/composition/registry.py` PRIMITIVE_ROWS (extended by FIX-D 2026-05-12 to include 42 codec primitives + 16,833 compatibility cells; FIX-D's DEPENDENCY_MISSING reclassification proved the matrix is sensitive to token coverage) |
| N consumers | 4+: (a) `enumerate_compositions()` enumerator; (b) cathedral autopilot ranking; (c) Pareto solver feasibility region; (d) bit-allocator initial weights |
| Bug class | New primitive lands in PACKET_COMPILER_TRANSFORMS (or new substrate in canonical_substrate_inventory()), PRIMITIVE_ROWS not updated, composition matrix silently misses N pairings. FIX-D found 12 such missing rows in this session. |
| Recommendation | **Discovery via dual-source introspection**: `PRIMITIVE_ROWS` should be COMPUTED from `PACKET_COMPILER_TRANSFORMS` + `canonical_substrate_inventory()` at module-load time. Adding either a new primitive OR a new substrate auto-extends the matrix. |
| Cost | ~80 LOC refactor + 10 regression tests |
| Risk if not implemented | (N+1)th primitive bug — the FIX-D 12-row miss is the empirical anchor |
| Status | DEFERRED-pending-operator-approval; touches `tac.composition.registry` which is dependency-rich (autopilot + Pareto + bit-allocator); recommend dedicated subagent post-cap-reset |

---

**HCL-4: Cathedral autopilot recipe registry**

| Field | Value |
|---|---|
| Source-of-truth | Hard-coded recipe catalog in `tac.optimization.autopilot_dispatch_ranking` (and sister `tac.optimization.cuda_cpu_axis_adaptive_analyzer`) — entries like `(film_pose, magic_codec)`, `(film_pose, hessian_block_fp)`, etc. |
| Hand-curated list | The recipe enumeration list (5-15 named pairings) inside the autopilot module |
| N consumers | 2: (a) the autopilot ranker itself; (b) `tools/build_b1_*.py` family (which cross-references the recipe names) |
| Bug class | New canonical pairing (e.g., a Wave-X composition) lands in PACKET_COMPILER_TRANSFORMS / PRIMITIVE_ROWS, autopilot recipe registry NOT updated, the autopilot can't rank/fire it |
| Recommendation | **Discovery via PRIMITIVE_ROWS**: if HCL-3 is implemented, autopilot recipes should be derived as the SUBSET of PRIMITIVE_ROWS pairings whose `is_composition=True` flag is set. No separate hand-curated list. |
| Cost | ~50 LOC refactor (dependent on HCL-3 first) |
| Risk if not implemented | Wave-X autopilot dispatch silently omits new pairings until a manual recipe-list update |
| Status | DEFERRED-pending-HCL-3 (depends on HCL-3 PRIMITIVE_ROWS canonicalization) |

---

### MEDIUM leverage (recommend documentation, defer implementation)

---

**HCL-5: Catalog #131 `_BARE_WRITE_CANONICAL_HELPERS` exempt list**

| Field | Value |
|---|---|
| Source-of-truth | Files implementing canonical fcntl-locked write helpers: `tac.deploy.lightning.active_jobs_state`, `tac.deploy.azure.active_vms_state`, `tac.continual_learning`, `tac.vastai_tracker` |
| Hand-curated list | `src/tac/preflight.py` `_BARE_WRITE_CANONICAL_HELPERS` set (extended by Catalog #133 META-meta gate when a previously-exempted file lacked the canonical lock pattern) |
| Bug class | Adding a file to the exempt list without confirming the file actually USES the canonical lock pattern (caught by Catalog #133). Already self-protected. |
| Recommendation | **No action**. Catalog #133 is the structural fix. The exempt list is small (~5 entries) and changes require adversarial review. |
| Status | NO-ACTION (already structurally protected) |

---

**HCL-6: Substrate test fixture token sets**

| Field | Value |
|---|---|
| Source-of-truth | `tac.packet_compiler.PACKET_COMPILER_TRANSFORMS` |
| Hand-curated list | `src/tac/substrates/*/tests/test_*` files often hardcode expected token strings |
| N consumers | 1 per test file (the test itself) |
| Bug class | Token added to PACKET_COMPILER_TRANSFORMS, individual sister test fixtures not updated, test silently passes against stale expected-set |
| Recommendation | **No action** beyond the canonical pattern: `test_phase1_packet_compiler_packet_compiler_transforms` already uses runtime introspection (per COUNCIL-E7 refactor — non-empty / no-duplicates / no-empty-or-whitespace / snake-case-ASCII). New test files should follow the SAME pattern, not hardcode tokens. |
| Status | NO-ACTION (canonical pattern documented; per-test compliance is reviewer responsibility) |

---

### LOW leverage (CLOSED — no action recommended)

---

**HCL-7: `vastai_active_instances.json` register_instance + remove_instance**

Already canonical — both write paths route through `tac.vastai_tracker` with
fcntl lock + strict load (Catalog #148). NO list to canonicalize.

**HCL-8: Lane registry `.omx/state/lane_registry.json`**

Already canonical — `tools/lane_maturity.py` is the single mutation surface
with file lock + Check 90 STRICT. NO list to canonicalize.

**HCL-9: Modal mount list (already extincted by FIX-I + Catalog #153)**

The Modal mount list bug class was canonicalized 2026-05-12 via
`tac.deploy.modal.mount_manifest.build_training_image()`. NO further work.

**HCL-10: Operator-authorize wrappers (already extincted by FIX-G + Catalog #162)**

The 10 operator-authorize-* shell scripts were unified to ONE canonical
`tools/operator_authorize.py --recipe` + 11 YAML recipes. NO further work.

## Verdict summary

| Candidate | Leverage | Status | Action |
|---|---|---|---|
| HCL-1 catalog #N entries | HIGH | DEFERRED-pending-operator | Build `tools/generate_catalog_md_table.py` |
| HCL-2 `_REPRESENTATION_LANE_TOKENS` | HIGH | DEFERRED-pending-preflight-window | Discovery via `canonical_substrate_inventory()` |
| HCL-3 `PRIMITIVE_ROWS` | HIGH | DEFERRED-pending-operator | Discovery via dual-source introspection |
| HCL-4 autopilot recipe registry | HIGH (dependent) | DEFERRED-pending-HCL-3 | Discovery via `PRIMITIVE_ROWS` |
| HCL-5 `_BARE_WRITE_CANONICAL_HELPERS` | MEDIUM | NO-ACTION | Catalog #133 covers |
| HCL-6 substrate test fixtures | MEDIUM | NO-ACTION | COUNCIL-E7 pattern documented |
| HCL-7 vastai tracker | LOW | CLOSED | Catalog #148 covers |
| HCL-8 lane registry | LOW | CLOSED | Check 90 covers |
| HCL-9 Modal mount list | LOW | CLOSED | FIX-I + Catalog #153 |
| HCL-10 operator-authorize | LOW | CLOSED | FIX-G + Catalog #162 |

**Net**: 4 HIGH-leverage candidates remain (HCL-1/2/3/4); all DEFERRED to
post-cap-reset operator-approval. 2 MEDIUM with NO-ACTION (already structurally
protected). 4 LOW already closed.

## 6-hook wire-in declaration (per Catalog #125)

This is a META audit memo, not an empirical landing. The 6 hooks are N/A by
construction:

1. **Sensitivity-map**: N/A — META audit; no per-tensor saliency contribution.
2. **Pareto constraint**: N/A — no new feasibility constraint introduced.
3. **Bit-allocator**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch**: N/A — META documentation; no archive
   variant produced.
5. **Continual-learning posterior**: N/A — no new empirical anchor.
6. **Probe-disambiguator**: N/A — no design tension between 2+ defensible
   interpretations; recommendations are strictly more general than current
   hand-curated patterns.

## Forbidden patterns honored

- ZERO `/tmp` paths.
- ZERO score claims (all candidates carry `score_claim=false` semantics by
  virtue of being META audit recommendations).
- ZERO MPS-derived strategic decisions.
- ZERO KILL verdicts (HIGH-leverage candidates are DEFERRED with reactivation
  criteria; LOW-leverage are CLOSED with explicit existing-protection refs).

## Cross-references

- `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md` (CANON-1.D source)
- `feedback_session_2026_05_12_lessons_learned_canonicalization_discipline.md` (Lesson 2)
- `feedback_modal_mount_manifest_consolidation_landed_20260512.md` (HCL-9 anchor)
- `feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512.md` (HCL-1 partial fix anchor)
- Catalog #118, #133, #148, #153, #159, #162 (existing structural protections)

## Operator decision

> Approve a follow-up subagent (post-cap-reset) to land HCL-1 (`tools/generate_catalog_md_table.py`)
> as the highest-leverage candidate? Estimated cost: 1 subagent, ~$0 GPU,
> ~1-1.5 hours wall-clock. The other 3 HIGH-leverage candidates can follow
> sequentially or in parallel depending on operator's wave priorities.
