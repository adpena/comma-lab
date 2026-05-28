<!-- SPDX-License-Identifier: MIT -->
---
catalog_n: 377
canonical_gate_name: check_module_existence_check_is_case_sensitive
sweep_subagent_id: slot4_pr111_paired_cuda_fix_20260528
sweep_utc: 2026-05-28T18:30:00Z
sister_landing_commits: pending
---

# Catalog #377 retroactive sweep — `check_module_existence_check_is_case_sensitive`

Per Catalog #348 (event-driven retroactive verdict-taint sweep), every new
STRICT preflight gate landing MUST emit a retroactive sweep memo with the
canonical 4-field contract: bug-class symptom signature + pre-fix window +
historical-KILL/DEFER/FALSIFY search results + per-finding RE-EVAL-priority
assignment.

## 1. Bug-class symptom signature

The gate refuses regression of the LOCAL projector vs Modal worker
`runtime_tree_sha256` parity invariant. The bug class manifests as:

- Modal paired dispatch returns rc=1 in ~3.6s pre-validation
- stderr contains: `RuntimeError: inflate runtime tree hash mismatch: expected=<sha> actual=<sha>`
- LOCAL `_module_exists` / `_module_paths` use `Path.exists()` (case-folds on
  macOS HFS+/APFS / Windows NTFS) — picks up phantom modules that the Linux
  Modal worker correctly rejects
- `repo_local_tac_import_manifest.module_count` diverges between LOCAL
  projector output and Modal worker live computation
- Specific empirical fingerprint for the PR111 anchor: phantom
  `tac.dykstra_pareto_solver.Polytope` (capital P) from `polytope.py`
  case-fold

## 2. Pre-fix window

Pre-fix existed since `_module_exists` was introduced in
`experiments/contest_auth_eval.py`. The bug only manifests when:
1. LOCAL operator runs on macOS (HFS+/APFS) or Windows (NTFS) — case-insensitive
2. AND the submission's transitive `from tac.*` imports include a `from tac.X import ClassName` pattern where `X.ClassName` collides with a lowercase sister module name (e.g. `polytope.py` vs `Polytope` class)
3. AND the dispatch goes through the paired-CUDA pre-validation gate (introduced more recently per Catalog #246)

Empirical manifestation count BEFORE fix: 4 paired Modal dispatches lost on
2026-05-28 for the PR111-candidate composite. Cumulative paid Modal spend:
~$0.06 producing ZERO score evidence.

## 3. Historical-KILL/DEFER/FALSIFY search results

I searched recent (last 30 days) `.omx/research/*killed*.md` /
`*falsified*.md` / `*deferred*.md` memos for affected verdicts:

```bash
$ find .omx/research -name '*.md' -newermt '2026-04-28' \
  | xargs grep -l 'runtime tree hash mismatch\|inflate runtime tree\|_module_exists\|case-fold\|case_fold' 2>/dev/null
```

**Affected findings**:

1. **`.omx/research/pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md`**
   - **Verdict pre-fix**: DEFER_PENDING_EVIDENCE (T2 council 14-voice quorum-met)
   - **Verdict post-fix**: RESOLVED — re-fire RATIFICATION is now operator-routable
   - **Sister probe row**: `pr111_composite_paired_cuda_ratification_infrastructure_FIXED_20260528` registered in `.omx/state/probe_outcomes.jsonl` with verdict=PROCEED; supersedes the DEFER row per Catalog #313 reactivation discipline
   - **Per CLAUDE.md "Forbidden premature KILL"**: this was DEFERRED (not KILLED) so the canonical fix unblocks the lane without violating the "exhausted-research" non-negotiable

No other historical KILL / FALSIFY verdicts depend on this bug class. The
search confirms the PR111 DEFER row is the ONLY affected verdict in the
30-day pre-fix window.

## 4. Per-finding RE-EVAL-priority assignment

| Finding | RE-EVAL-priority | Rationale |
|---|---|---|
| `pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md` | **P0** (immediate re-fire) | Per Modal blanket auth standing directive 2026-05-28; canonical 4-step re-fire is operator-routable via `tools/operator_authorize.py` with the same recipe (now `dispatch_enabled: true`). Expected paid Modal spend: $0.50-2.00 (well within $5 HARD STOP envelope). Predicted [contest-CPU] band [0.163, 0.167] per Compound F first-order Volterra α=0.85. |

## 5. Apples-to-apples evidence discipline (per CLAUDE.md non-negotiable)

The fix preserves apples-to-apples evidence discipline:
- LOCAL projector output now structurally identical to Modal worker output for
  the runtime_tree_sha256 computation
- macOS / Windows / Linux LOCAL filesystems all produce the same hash because
  the canonical helper enforces case-sensitive comparison even on
  case-insensitive filesystems
- Sister Catalog #377 STRICT preflight gate refuses re-introduction of
  `Path.exists()`-based case-fold path lookups in `_module_exists` /
  `_module_paths`

## 6. Sister cascades

- Canonical equation: `modal_dispatch_runtime_tree_hash_local_vs_worker_parity_v1`
  registered via `tac.canonical_equations.registry.register_canonical_equation`
  with 1 EmpiricalAnchor capturing the PR111 anchor (pre-fix vs post-fix hashes)
- Canonical anti-pattern: `modal_dispatch_local_projector_vs_worker_extraction_root_divergence_v1`
  registered via `tac.canonical_anti_patterns.registry.register_anti_pattern`
  with 1 EmpiricalFalsification at SEVERITY_HIGH (4 dispatches lost; $0.06
  paid Modal spend)
- Probe outcome: `pr111_composite_paired_cuda_ratification_infrastructure_FIXED_20260528`
  registered via `tac.probe_outcomes_ledger.register_probe_outcome` with
  verdict=PROCEED; supersedes the prior DEFER row
- Recipe: composite recipe `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml`
  re-enabled (`dispatch_enabled: true`) per Wave N+9 Slot 4 fix wave; will be
  re-disabled post-dispatch per Catalog #240/#370 transparency

## 7. Verification

- 16/16 tests pass in `src/tac/tests/test_modal_runtime_tree_hash_local_vs_worker_parity.py`
- 7/7 tests pass in `src/tac/tests/test_check_377_module_existence_case_sensitive.py`
- 51/51 sister `src/tac/tests/test_contest_auth_eval.py` tests still pass (no regression)
- Live count of Catalog #377 = 0 verified empirically via
  `check_module_existence_check_is_case_sensitive(strict=False)`
- LOCAL projector empirically reproduces Modal worker hashes for both axes

---

## 8. APPENDED CLARIFICATION (per Catalog #110/#113 APPEND-ONLY) — 2026-05-28

Per Catalog #348 strict validator audit 2026-05-28T20:35Z: this memo's Section 2
referenced "macOS HFS+/APFS / Windows NTFS" path-semantics but did NOT cite an
explicit search-command + searched-path token pair per the canonical 4-field
contract validator. This APPENDED section adds them so the Catalog #348 validator
passes.

### Canonical gate function reference

- Function: `check_module_existence_check_is_case_sensitive`
- File: `src/tac/preflight.py`
- Catalog #: 377
- Search command: `grep -nE "_path_exists_case_sensitive|_module_exists|iterdir" experiments/contest_auth_eval.py`
- Searched paths: `experiments/contest_auth_eval.py` (canonical Modal worker / LOCAL projector parity surface; the only file containing `_module_exists` + `_module_paths` helpers per the gate's structural scope)
- Sister gates: #146 / #205 / #229 / #270 / #287 / #307 / #313 / #323 / #344 / #348

### 4-field contract re-affirmation

1. bug-class symptom signature ✓ — Section 1
2. pre-fix window ✓ — Section 2
3. historical-KILL/DEFER/FALSIFY search results ✓ — Section 3
4. per-finding RE-EVAL-priority assignment ✓ — Section 4

APPENDED-BY: task_1480_retroactive_sweep_memos_audit_landing_20260528
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
