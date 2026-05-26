# Path 3 FIX-WAVE-G-OP1: NIRVANA test expand for Wave #1 canonical exports LANDED 2026-05-26

**Lane:** `lane_path_3_fix_wave_r3_g_op1_nirvana_test_expand_20260526` L1
**Charter:** single-finding IMPLEMENTATION-LEVEL fix for G=NIRVANA test regression
caught by R3-COMBINED at commit `f0cd43237`.
**Operator approval:** 2026-05-26.
**Mission contribution:** `apparatus_maintenance` (test-suite restoration; no score
delta; unblocks G fresh R1' counter so the substrate can re-enter the recursive
adversarial review cycle per CLAUDE.md "Recursive adversarial review protocol").
**Cost:** $0. **Wall-clock:** ~10 min.

---

## 1. Finding (per Catalog #307 paradigm-vs-implementation classification)

R3-COMBINED 3-axis review (commit `f0cd43237`, memo
`.omx/research/path_3_g_recursive_adversarial_review_r3_combined_3_axis_20260526.md`)
caught that the strict-`__all__`-equality test at
`src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py:46-58` does NOT
include the 4 NEW canonical exports added by Wave #1 posterior emission wire-in
(commit `3d103dafd`).

**Empirical anchor (pre-fix):**

```
src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py::test_module_exposes_canonical_public_api FAILED
E       AssertionError: assert {'ARCHITECTUR..._BASE_H', ...} == {'ARCHIVE_GRA..._LEVELS', ...}
E         Extra items in the left set:
E         'SUBSTRATE_ID'
E         'emit_landing_posterior_anchor'
E         'ARCHITECTURE_CLASS'
E         'CANONICAL_EQUATION_IDS'
```

**Classification per Catalog #307:** **IMPLEMENTATION-LEVEL** falsification.
Wave #1's canonical narrow-public-API contract via `__all__` is correct per
Catalog #335 (cathedral consumer auto-discovery). G is the ONLY substrate with
this strict-equality test pattern (other 8 Path 3 substrates use subset-equality);
G's test simply needs to expand its expected set to match the actual `__all__`.
Wave #1 paradigm-INTACT; test surface needed catch-up only.

**Catalog #299 quota brake decision:** NO new STRICT preflight gate. Single-line
mechanical test fix; not a recurring bug class; current catalog # well under 400
quota but per Catalog #299 "evaluate META-meta gate >=3 sister cases BEFORE
landing additional gate" — only 1 case (G is the unique strict-equality test);
adding a gate would dilute Catalog #335's contract surface.

---

## 2. Fix (1-line diff)

**File:** `src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py`
**Lines:** 46-58 → 46-62 (4 entries added; alphabetical-ordering preserved per
existing test convention).

```diff
@@ -46,12 +46,16 @@
     expected = {
+        "ARCHITECTURE_CLASS",
         "ARCHIVE_GRAMMAR_FIELDS",
         "ARCHIVE_MAGIC",
         "ARCHIVE_VERSION",
+        "CANONICAL_EQUATION_IDS",
         "DEFAULT_BASE_H",
         "DEFAULT_BASE_W",
         "DEFAULT_NUM_LEVELS",
         "DEFAULT_PER_PAIR_LATENT_DIM",
         "NIRVANA1_HEADER_FMT",
         "NIRVANA1_HEADER_LEN",
         "NirvanaCascadingNervConfig",
         "SISTER_SUBSTRATES",
+        "SUBSTRATE_ID",
+        "emit_landing_posterior_anchor",
     }
     assert set(mod.__all__) == expected
```

**4 NEW canonical exports added** (sourced verbatim from G's `__init__.py:211-215`
Wave #1 wire-in commit `3d103dafd`):

1. `SUBSTRATE_ID` — `"nirvana_cascading_nerv"` constant for cathedral consumer
   auto-discovery per Catalog #335.
2. `ARCHITECTURE_CLASS` — `"nirvana_cascading_nerv_hierarchical_residual_decoder_l0_scaffold_mlx"`
   per Wave #1 canonical posterior schema.
3. `CANONICAL_EQUATION_IDS` — tuple containing
   `cascading_nerv_per_stage_residual_v1_proposed_per_audit_e757bb74c_op_routable_3`
   (proposed equation token per Catalog #344 lineage trace).
4. `emit_landing_posterior_anchor` — canonical helper function per
   `tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor`
   (Wave #1 charter commit `f6b432be1` + wire-in commit `3d103dafd`).

---

## 3. Test verification (POST-FIX)

```
src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py::test_module_exposes_canonical_public_api PASSED [100%]
============================== 1 passed in 0.12s ===============================
```

**Target test transitioned RED → GREEN.**

---

## 4. Collateral regression scan

### 4a. G's full test suite (27 tests)

```
============================== 27 passed in 0.56s ==============================
```

ALL 27 G tests PASS (test_basic.py covers structural import + Catalog
#124/#139/#240/#305 compliance + MLX↔numpy↔PyTorch parity + inflate runtime
+ NIRVANA1 archive grammar round-trip + estimate_archive_bytes Dykstra
feasibility). Zero collateral regression from the test expansion.

### 4b. Sister Wave #1 posterior emission helper tests (75 tests)

```
src/tac/tests/test_substrate_posterior_emission_helper.py + test_path_3_substrate_landing_posterior_emissions.py
============================== 75 passed in 0.74s ==============================
```

Including 9 parametrized G=`nirvana_cascading_nerv` rows across the canonical
suite:
- `test_substrate_emits_anchor_with_canonical_markers[nirvana_cascading_nerv]` PASS
- `test_substrate_posterior_refused_per_advisory_grade[nirvana_cascading_nerv]` PASS
- `test_substrate_manifest_row_emitted_with_canonical_extras[nirvana_cascading_nerv]` PASS
- `test_substrate_emit_is_idempotent_at_manifest_jsonl[nirvana_cascading_nerv]` PASS

Wave #1 wire-in remains coherent. Zero regression in the sister surface.

---

## 5. Advancement (per CLAUDE.md "Recursive adversarial review protocol")

**Before:** G counter RESET to 0/3 by R3-COMBINED finding.
**After:** G can now start fresh R1' counter on next review round. Wave #1
canonical posterior emission wire-in remains intact + test-suite-verified.

Per R3-COMBINED memo § "Verdict": this was the ONLY blocking finding for G;
other concerns flagged were AMBER/non-blocking. With the test fix landed, G
re-enters the standard 3-clean-pass recursive adversarial review cycle.

---

## 6. Sister coordination (Catalog #230 + #340)

**At task start:** ALL SISTERS COMPLETE (R3-COMBINED + L2-INFRA-BUILD + R1'' +
SISTER-#1265-GATE-Z6PCWM1 all landed pre-charter).

**Sister-checkpoint guard verdict:** PROCEED (0 in-flight sister subagents in
60-minute lookback window touching `test_basic.py` or sister test files).

**My scope:** G=NIRVANA test ONLY — DISJOINT from any sister surface. ZERO
files_touched overlap. Wave #1 wire-in's `__init__.py` (touched by sister
subagent earlier) NOT modified by this fix — only the test's expected-set
literal was expanded.

---

## 7. Discipline (per CLAUDE.md non-negotiables)

- **Catalog #229 PV:** read FULL `test_basic.py` (694 lines) + G's `__init__.py`
  (332 lines) + Wave #1 helper module + R3-COMBINED parent memo BEFORE any edit.
- **Catalog #117/#157/#174/#235/#289** canonical serializer with POST-EDIT
  `--expected-content-sha256` per file
  (test_basic.py post-edit sha `ef54ccb674a4c8512b77f89257d47bf714f4ac349ec6dc08785ef0d8c8678fba`).
- **Catalog #119** Co-Authored-By trailer (internal commit, NOT public-PR
  surface per user_pr_attribution standing rule).
- **Catalog #287** placeholder-rationale rejection (waiver tokens in this memo
  carry substantive rationale; no `<rationale>` / `<reason>` literals).
- **Catalog #110/#113** APPEND-ONLY HISTORICAL_PROVENANCE — test EXPAND not
  REPLACE; the original 11 expected entries are preserved verbatim in the new
  15-entry set (4 NEW added alphabetically).
- **Catalog #208** docs/local-paths — no `/Users/adpena/` paths in this memo;
  only repo-relative paths.
- **Catalog #230** sister-subagent ownership map — verified ALL SISTERS
  COMPLETE at task start; my edit DISJOINT from all sister files_touched.
- **Catalog #340** sister-checkpoint guard PROCEED verified before edit
  (`tools/check_sister_checkpoint_before_git_add.py` rc=0).
- **Catalog #307** paradigm-vs-implementation classification: IMPLEMENTATION-LEVEL
  documented + explicit Wave #1 paradigm INTACT verdict.
- **Catalog #335** canonical narrow-public-API contract via `__all__` preserved
  + test now matches the canonical contract.
- **Catalog #299** NO new STRICT gate (single-line mechanical fix; not a
  recurring bug class; <3 sister cases).
- **Catalog #206** crash-resume protocol — checkpoint emitted at task start +
  this landing memo serves as the final checkpoint record.

---

## 8. 6-hook wire-in declaration (per Catalog #125)

This is a test-suite-only fix; all 6 hooks N/A:

- Hook #1 sensitivity-map = N/A (no sensitivity signal).
- Hook #2 Pareto constraint = N/A (no Pareto-relevant signal).
- Hook #3 bit-allocator = N/A (no bit-allocator signal).
- Hook #4 cathedral autopilot dispatch = N/A (no candidate row contribution).
- Hook #5 continual-learning posterior = N/A (no posterior anchor; test
  verification only).
- Hook #6 probe-disambiguator = N/A (no disambiguation surface).

The fix RESTORES the Wave #1 wire-in's hook #4 + #5 active state (which
depend on the test passing for canonical narrow-public-API contract
verification).

---

## 9. Cross-references

- **Parent finding memo:** `.omx/research/path_3_g_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
- **Wave #1 charter:** `tac.substrates._shared.posterior_emission_helper` (commit `f6b432be1`)
- **Wave #1 wire-in commit:** `3d103dafd` (G `__init__.py:211-215`)
- **Lane registry:** `lane_path_3_fix_wave_r3_g_op1_nirvana_test_expand_20260526` L1
- **Sister Wave #1 tests:** `src/tac/tests/test_path_3_substrate_landing_posterior_emissions.py` (75 tests; 9 G-parametrized)
- **CLAUDE.md sections:** "Recursive adversarial review protocol" + "Forbidden
  premature KILL without research exhaustion" + Catalog #307 paradigm-vs-
  implementation falsification + Catalog #335 cathedral consumer auto-
  discovery canonical contract.

---

*Landed via canonical subagent commit serializer per Catalog #117/#157/#174/#235/#289.*
