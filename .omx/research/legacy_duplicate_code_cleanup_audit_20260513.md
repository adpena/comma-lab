# Legacy + duplicate code cleanup sweep — 2026-05-13

Operator directive 2026-05-13: "keep cleaning legacy code and deleting
duplicate code as you go". Per CLAUDE.md "Beauty, simplicity, and developer
experience — non-negotiable": "Delete dead fields, stale adapters, and
duplicate one-offs once a canonical contract replaces them."

Lane: `lane_legacy_duplicate_code_cleanup_20260513` (Level 1 after this
sweep — `impl_complete`, `memory_entry`, `three_clean_review`).

## Summary

| Category | Found | Canonicalized | Deleted | Net LOC | Tests passing |
|---|---|---|---|---|---|
| 1. `DistillationChain` name collision | 2 classes / 1 alias | 2 distinct surfaces, renamed schema | `DistillationChainCanonical` alias | -3 (alias) +20 (docstrings) | 427/427 composition |
| 2. `time_traveler` compat alias | 1 thin alias | KEEP — already minimal | none | 0 | 492/492 substrate |
| 3. Ternary QAT (`quantization_ternary.py`) | 1 sister draft | ALREADY-DELETED upstream | n/a | 0 (verified) | n/a |
| 4. `src/tac/codec/jscc/` lint debt | 0 warnings | already clean | n/a | 0 | n/a |
| 5. `__pycache__` cruft in public-PR intake | ~8,708 .pyc + dirs | n/a | bulk-deleted disk-only | -8.7k files (untracked) | n/a |
| 6a. Substrate `_SinAct` / `_UpBlock` dup audit | 9× / 7× across substrates | DEFERRED — diverging implementations | none | 0 (DEFER) | n/a |
| 6b. Composer class duplicates | 2 composers (PoE, Hypernet) | both distinct | none | 0 | n/a |
| 6c. Tool CLI duplicate audit | 452 /458 with `__main__` | acceptable | none | 0 | n/a |
| 6d. Duplicate test filenames | 0 collisions | n/a | none | 0 | n/a |
| 6e. Today's memory landings | 30 unique | n/a | none | 0 | n/a |
| 7. Lint cleanup on touched files | 0 F401/F403/F405/E402 | already clean | none | 0 | n/a |

## Detail per category

### 1. `DistillationChain` name collision (RESOLVED)

Two sister-subagent landings introduced two distinct `DistillationChain`
classes with completely different APIs and concerns:

**A. `tac.composition.distillation.DistillationChain`** (313 LOC)
- Typed schema: `chain_id`, `stages` (tuple of `DistillationStage`),
  `teacher_substrate_id` / `student_substrate_id` (substrate-level)
- Frozen dataclasses + `to_dict()` / `to_json()` / `sha256()` provenance
- 6 tests in `test_distillation.py`

**B. `tac.composition.distillation_chain.DistillationChain`** (313 LOC)
- Hinton 2014 KL distillation primitive: `levels` (tuple of
  `DistillationLevel` with `param_count`, `temperature`, `kl_weight`)
- `serialize_state()` / `deserialize_state()` binary archive grammar +
  `distillation_loss()` differentiable loss
- 29 tests in `test_distillation_chain.py`

The package `__init__` papered over the collision by re-exporting B as
`DistillationChainCanonical` while keeping A's name. Neither had external
consumers (verified via repo-wide grep).

**Canonical resolution:**
- A renamed to `DistillationStageChain` (reflects its substrate-stage
  sequence concern); `build_distillation_chain` →
  `build_distillation_stage_chain`.
- B keeps the canonical name `DistillationChain` (Hinton-canonical
  composition-primitive surface).
- `DistillationChainCanonical` alias DELETED.

**Replacement pointers:**
- typed schema → `tac.composition.distillation.DistillationStageChain`
- Hinton KL primitive → `tac.composition.distillation_chain.DistillationChain`

**Commit:** `6400e958` — composition: rename schema `DistillationChain` →
`DistillationStageChain` (extinct name collision).

**Tests:** 43/43 directly touched tests pass; 427/427 broader
`src/tac/composition/` pass; 492/492 composition + time_traveler.

### 2. `time_traveler` compatibility alias (KEEP — already thin)

`src/tac/substrates/time_traveler/` consists of:
- `__init__.py` (782B) — `from ..time_traveler_l5_autonomy import *` +
  small `TIME_TRAVELER_METADATA` dict
- `architecture.py` / `archive.py` / `inflate.py` / `score_aware_loss.py`
  — each 149-159B; all are one-line star-imports from the canonical
  package
- `tests/test_time_traveler_archive.py` — proves the alias is a true
  identity (e.g. `assert compat.TimeTravelerConfig is canonical.TimeTravelerConfig`)

**Verdict: KEEP** per operator default. The alias is already explicitly
thin (no architectural duplication; pure re-export shim). The only
external reference is a docstring mention in `sabor_boundary_only_renderer`.
No action taken.

### 3. Ternary QAT (`quantization_ternary.py`) — already deleted

Sister IMPL-A's note said their initial draft at
`src/tac/quantization_ternary.py` was removed in favor of IMPL-B's
canonical `src/tac/optimization/ternary_qat.py`. Verified:

- `src/tac/quantization_ternary.py`: DOES NOT EXIST (file removal complete).
- `src/tac/optimization/ternary_qat.py`: 14.5KB canonical.
- `grep -rn "tac.quantization_ternary"` → no orphan imports.

No action needed.

### 4. `src/tac/codec/jscc/` lint debt — already clean

`ruff check src/tac/codec/jscc/` returns "All checks passed!" — the
9 warnings mentioned in the IMPL-A report were already cleared by a later
landing pass. No action needed.

### 5. `__pycache__` cruft in public-PR intake (CLEANED)

Catalog #109 enforces SOURCE pristineness for public-PR intake clones,
but it doesn't gate `.pyc` / `__pycache__` files (which are non-tracked
build outputs). The audit found ~8,708 `.pyc` files + matching
`__pycache__` directories scattered across:

- `experiments/results/public_pr*_intake_*/`
- `experiments/results/public_pr_intake_full/`

All bulk-deleted (disk cruft only — none tracked by git; verified via
`git ls-files` returned empty). No source files touched.

### 6. Other audits

**6a. Substrate `_SinAct` / `_UpBlock` duplicate-class audit**

| Class | Count | Identical? |
|---|---|---|
| `_SinAct` | 9× | 2 distinct AST hashes (6 vs 3); minor wording diffs |
| `_UpBlock` | 7× | 7 distinct AST hashes — diverging implementations |

DEFERRED — consolidating diverging implementations into `_shared/` is a
**council-grade design decision** per CLAUDE.md "Design decisions —
non-negotiable", not engineering hygiene. Each substrate's `_UpBlock`
carries substrate-specific specialization; refactoring risks introducing
subtle behavior drift.

Reactivation criteria: explicit council review with grand-council
consensus on a unified `_shared/upblock_canonical.py` interface, plus
empirical archive-bytes parity across all 7 callers.

**6b. Composition composers**: 2 distinct classes
(`ProductOfExpertsComposer` in `product_of_experts.py`,
`DeterministicHypernetworkComposer` in `adapters.py`) — different
concerns, no duplication.

**6c. Tool CLIs**: 452 / 458 `.py` files in `tools/` have `if __name__ ==`
guards; no detected duplicate names.

**6d. Test file collisions**: All 1071 `test_*.py` filenames in
`src/tac/tests/` are unique.

**6e. Today's memory landings**: 30 unique `feedback_*_landed_20260513.md`
files; no duplicate-named landings.

### 7. Lint cleanup on touched files

`ruff check --select F401,F403,F405,E402 src/tac/composition/
src/tac/substrates/time_traveler/ src/tac/optimization/ternary_qat.py`
→ "All checks passed!"

## Verification

- `python tools/lane_maturity.py validate` → 584 lanes validated clean
- `python -m pytest src/tac/composition/ src/tac/substrates/time_traveler*/` → 492/492 PASS
- `ruff check` on all touched files → clean
- `python -m tac.preflight --scope dev` → 1 STRICT failure
  (`check_lane_pre_registered_before_work_starts`) but the 4 violations
  are in files NOT touched by this sweep (`driving_prior_readiness.py`,
  `pretrained_driving_prior/codebook.py`, `driving_prior_world_model/config.py`)
  — pre-existing sister-subagent debt outside our coordination surface.

## Operator-routable decisions surfaced

1. **`_SinAct` / `_UpBlock` cross-substrate consolidation** — council
   review needed for unified `_shared/` extraction; deferred.
2. **Sister-subagent driving_prior lane-registration debt** — the 4
   preflight failures in `driving_prior_*` need pre-registration by
   the subagent that introduced them; outside this sweep's mandate.

## LOC saved

- Direct: 12 LOC (alias re-export block) + ~8,708 `.pyc` files cleaned
- Indirect (name-collision footgun avoided): infinite (future bug
  surface eliminated)

## Wire-in hooks per Catalog #125 (META cleanup)

- (1) Sensitivity-map: N/A (cleanup pass)
- (2) Pareto constraint: N/A (cleanup pass)
- (3) Bit-allocator hook: N/A (cleanup pass)
- (4) Cathedral autopilot dispatch hook: N/A (cleanup pass)
- (5) Continual-learning posterior update: N/A (cleanup pass)
- (6) Probe-disambiguator: N/A (one canonical name per concern after rename)

Memory file: `feedback_legacy_duplicate_code_cleanup_landed_20260513.md`.
