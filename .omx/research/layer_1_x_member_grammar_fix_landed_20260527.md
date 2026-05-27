# Layer 1 x-member archive-grammar fix LANDED 2026-05-27

**Lane**: `lane_layer_1_x_member_grammar_fix_20260527` L1 (impl_complete + memory_entry)

**Subagent**: `layer1_grammar_fix_1`

**Mission contribution per Catalog #300**: `apparatus_maintenance` /
`frontier_protecting`. Closes Phase 10 op-routable #1 — the canonical-submission-
pipeline Layer 1 archive-grammar `x`-member gap that drove the V14-V2 PR111
candidate end-to-end dry-run to exit 5 NAMED-BLOCKER. NO paid GPU; NO `gh`
commands. $0, ~5 min wall-clock.

## TL;DR

The canonical Layer 1 helper `discover_section_specs_from_archive` (and the
sister `ArchiveGrammarManifest.__post_init__` monolithic invariant) hardcoded
the member name `0.bin` as the sole monolithic-classification signal. The
canonical frontier archives PR101 / DQS1 use a single ZIP member named `x` (the
established frontier grammar). A structurally-monolithic single-`x`-member
archive was therefore misclassified as multi-file, failing HNeRV-parity-L3 and
driving the Phase 10 dry-run to exit 5.

The fix makes monolithic classification **member-name-agnostic**: ANY archive
with exactly ONE ZIP member is monolithic, recognizing the structural property
("exactly one member") rather than the literal name. Backward-compat for `0.bin`
and genuine multi-file (2+ members) preserved.

**Empirical re-run result**: the Phase 10 dry-run against the actual V14-V2
candidate archive (`0a3abfe6...`, single `x` member, 178446 bytes) now advances
**from exit 5 (Layer-1 NAMED-BLOCKER) to exit 2 (COMPLIANCE-ERRORS)** — Layer 0,
1, 2 (builder, inflate_py_loc 73 under budget), and 3 (linter clean) all PASS;
the remaining Layer 4 compliance errors are the operator-gated D3 (`gh release`)
+ D5 (`gh pr create`) artifacts the Phase 10 memo already documented as
operator-gated (NOT in this lane's scope). The Layer 1 gap is closed.

## The bug (root cause, two coupled surfaces)

`src/tac/submission_packet/archive_grammar.py`:

1. **`discover_section_specs_from_archive`** (the classifier): the monolithic
   branch was gated `if len(members) == 1 and members[0] == monolithic_member_name`
   where `monolithic_member_name` defaults to `CANONICAL_MONOLITHIC_MEMBER_NAME
   = "0.bin"`. A single-`x`-member archive fell through to the multi-file
   fallback and returned `derived_is_monolithic=False`.

2. **`ArchiveGrammarManifest.__post_init__`** (the validator): the
   `monolithic_single_file=True` branch required EVERY section's
   `member_name == "0.bin"`. Even if surface (1) were patched to return
   `derived_is_monolithic=True` for a single `x` member, the validator would
   then reject the resulting manifest because the section's member is `x` not
   `0.bin`.

The canonical CLI calls
`build_archive_grammar_from_compression_pipeline_result` with
`monolithic_single_file=True` (default); when `section_specs is None` it
auto-derives, and the line `if monolithic_single_file and not
derived_is_monolithic: monolithic_single_file = False` flipped the flag to
False, then `__post_init__` raised
`monolithic_single_file=False requires non-None multi_file_justification` →
exit 5.

## The fix (PRIMARY — member-name-agnostic structural classification)

Per HNeRV parity L3 ("monolithic single-file `0.bin` OR explicitly justified
multi-file" — a single member IS monolithic regardless of name):

- **`discover_section_specs_from_archive`**: classify `len(members) == 1` as
  monolithic and emit a section spec using the member's ACTUAL name. The
  `monolithic_member_name` kwarg is retained for backward compatibility but no
  longer gates classification.
- **`ArchiveGrammarManifest.__post_init__`**: `monolithic_single_file=True`
  now requires all sections to share ONE distinct member name (the structural
  single-file property — `{spec.member_name for spec in section_specs}` has
  cardinality ≤ 1), not specifically `0.bin`. 2+ DISTINCT members with
  `monolithic_single_file=True` is forbidden (that is multi-file and must carry
  `multi_file_justification` with `monolithic_single_file=False`).
- Field docstrings updated to describe the structural property + name the
  PR101/DQS1 `x` convention.

The ALTERNATIVE (`--multi-file-justification` flag path) was NOT taken: PRIMARY
is safe for all downstream consumers (verified by the 241-test pipeline
regression below) and is the cleaner reading per HNeRV parity L3 — a single `x`
member genuinely IS monolithic, so forcing the operator to supply a
"multi-file justification" for a single-file archive would be semantically
wrong.

## Test coverage

`src/tac/tests/test_archive_grammar.py` — net new tests + 1 replaced:

- `test_discover_single_x_member_is_monolithic` — the PR101/DQS1 `x`-member
  archive classifies monolithic (the exact Phase 10 NAMED-BLOCKER case).
- `test_discover_single_0bin_member_is_monolithic_regression` — `0.bin`
  backward-compat regression.
- `test_discover_single_arbitrary_member_is_monolithic` — any single-member
  name (`weights.bin`) classifies monolithic.
- `test_discover_genuine_multi_file_still_not_monolithic` — 2+-member archive
  still NOT monolithic (the fix does not over-classify).
- `test_build_canonical_entry_single_x_member_passes` — end-to-end through the
  canonical entry point: a single-`x`-member archive yields a valid
  ArchiveGrammarManifest with `monolithic_single_file=True` and NO
  `multi_file_justification` needed.
- `test_manifest_monolithic_accepts_any_single_member_name` (replaced the old
  `test_manifest_monolithic_requires_canonical_member_name`) — a single
  `weights.bin`-member spec with `monolithic_single_file=True` is now valid.
- `test_manifest_monolithic_rejects_multiple_distinct_members` — 2+ DISTINCT
  members with `monolithic_single_file=True` raises.
- `test_live_repo_v14_v2_dqs1_x_member_archive_is_monolithic` — regression
  against the ACTUAL on-disk V14-V2 candidate archive (existence-guarded so the
  suite stays green on clones without the candidate work dir).

`empty archive` edge case already covered by the pre-existing
`test_discover_empty_archive_raises` (raises `ArchiveGrammarError`).

**Result**: `src/tac/tests/test_archive_grammar.py` 94 passed (was ~88). Full
downstream pipeline regression
(`test_archive_grammar` + `test_submission_bundle` +
`test_operator_pr_submission_full_lifecycle_cli` + `test_compression_pipeline`)
**241 passed** — backward-compatible; no downstream consumer broke.

## Phase 10 dry-run re-run result (empirical)

`tools/operator_pr_submission_full_lifecycle.py --dry-run` against the V14-V2
candidate (scratch output to `.omx/tmp/`, cleaned post-run; NO `submissions/`
pollution; NO `gh`):

| Layer | Pre-fix | Post-fix |
|---|---|---|
| pre attribution self-lint | PASS | PASS |
| 0 compression_pipeline | PASS | PASS |
| 1 archive_grammar | **FAIL (exit 5)** | **PASS** (`ok: true`, sections: 1, sha `0a3abfe6...`) |
| 2 builder | NOT REACHED | **PASS** (inflate_py_loc 73, sidecar emitted) |
| 3 linter | NOT REACHED | **PASS** (0 errors, 0 warns) |
| 4 compliance | NOT REACHED | FAIL (operator-gated D3 + D5 per Phase 10 memo) |

**Exit code: 5 → 2.** Layer 1 fully unblocked. The remaining Layer 4
COMPLIANCE-ERRORS verdict is the expected next blocker — D3 hosting
(`gh release`) + D5 submission (`gh pr create`) are operator-gated per CLAUDE.md
"Executing actions with care" and the Phase 10 memo's strict-flip blocker chain
items 4-5 (NOT in this lane's scope).

## Pre-existing preflight note (NOT this lane)

`preflight_all()` flags two sister-modified `.omx/state/*.jsonl` ledgers
(`modal_call_id_ledger.jsonl` + `probe_outcomes.jsonl`) under the Catalog #113
artifact-lifecycle gate as UNKNOWN-classification. These are canonical-helper-
managed ledgers I did NOT touch (confirmed via `git status`); the gap is a
pre-existing artifact-classification issue orthogonal to this change. My edited
`.py` files are review-tracker `reviewed` and the targeted archive-grammar +
pipeline test suites are green.

## Canonical-vs-unique decision per layer

| Decision | Choice | Rationale |
|---|---|---|
| Structural single-member classification vs hardcoded name allow-set | FORK_BECAUSE_PRINCIPLED | HNeRV parity L3 means "exactly one member" structurally; an allow-set (`{"0.bin", "x"}`) would still break on the next frontier grammar's member name |
| PRIMARY (member-name-agnostic) vs ALTERNATIVE (`--multi-file-justification` flag) | ADOPT PRIMARY | a single `x` member genuinely IS monolithic; forcing a "multi-file justification" for a single-file archive is semantically wrong + the 241-test regression proves PRIMARY is safe downstream |
| Keep `monolithic_member_name` kwarg for backward compat | ADOPT_CANONICAL | preserves the existing API surface; the kwarg simply no longer gates classification |

## 9-dimension success checklist evidence

1. **UNIQUENESS** — closes the exact Phase 10 NAMED-BLOCKER.
2. **BEAUTY + ELEGANCE** — structural property (member count) replaces a magic
   string; fewer lines, clearer invariant.
3. **DISTINCTNESS** — orthogonal to sister Cascade B wave-2 work.
4. **RIGOR** — empirically re-ran the Phase 10 dry-run + regression-tested the
   actual on-disk V14-V2 archive; 241-test downstream regression green.
5. **OPTIMIZATION PER TECHNIQUE** — single coupled fix at the two surfaces that
   together produced the misclassification.
6. **STACK-OF-STACKS COMPOSABILITY** — Layers 2-3 now compose cleanly atop the
   fixed Layer 1.
7. **DETERMINISTIC REPRODUCIBILITY** — exit 2 deterministic; archive sha
   byte-stable.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — $0; ~5 min wall-clock; NO paid GPU.
9. **OPTIMAL MINIMAL CONTEST SCORE** — does NOT directly lower score;
   frontier-protecting fix that unblocks the canonical PR111 lifecycle for the
   already-frontier-anchored V14-V2 candidate.

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 SENSITIVITY_MAP** — N/A (apparatus fix; no signal contribution)
- **Hook #2 PARETO_CONSTRAINT** — N/A
- **Hook #3 BIT_ALLOCATOR** — N/A
- **Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH** — **ACTIVE** (Layer 1 now passes for
  PR101/DQS1-grammar candidates, so the `pr_submission_compliance_consumer` per
  Catalog #335 can weight V14-V2 PR111-readiness past the Layer 1 gate)
- **Hook #5 CONTINUAL_LEARNING_POSTERIOR** — **ACTIVE** (the exit 5 → 2 advance
  is the empirical anchor for the Phase 9
  `full_lifecycle_cli_consolidation_savings_v1` equation; it records that the
  Layer 1 `x`-member gap is closed, moving the 6-equation FORMALIZATION_PENDING
  promotion one blocker closer to a future PACKET-CLEAN run)
- **Hook #6 PROBE_DISAMBIGUATOR** — **ACTIVE** (the structural single-member
  classification IS the disambiguator: a single-`x`-member archive is
  monolithic, not multi-file; member name is not the classification signal)

## Sister coordination (Catalog #230 ownership map)

Sister `ac302ffd185e1543d` (Cascade B wave-2) owns `tools/cascade_b_*.py` +
`.omx/research/cascade_b_*.md` — SISTER-DISJOINT (this lane owns
`src/tac/submission_packet/archive_grammar.py` + its test + this memo). ZERO
file collision. The canonical serializer's POST-EDIT `--expected-content-sha256`
+ Catalog #340 sister-checkpoint guard protected the commit.

## Cross-references

- Phase 10 dry-run landing (the NAMED-BLOCKER source):
  `.omx/research/phase_10_pr111_candidate_dry_run_validation_landed_20260527.md`
- Phase 3 Layer 1 landing:
  `.omx/research/phase_9_operator_pr_submission_full_lifecycle_cli_landed_20260527.md`
- V14-V2 candidate report:
  `reports/pr111_candidate_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md`
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json`

**VERDICT: LAYER_1_X_MEMBER_GRAMMAR_FIX_LANDED — single-member archives now
classify monolithic member-name-agnostically; V14-V2 PR111 candidate dry-run
advances exit 5 → exit 2 (Layer 1 unblocked; remaining Layer 4 blockers are
operator-gated D3/D5); 241-test downstream regression green; backward-compat for
`0.bin` + genuine multi-file preserved.**

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
