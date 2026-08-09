# DG1 findings — #994 magnitude-dismissal false positive

Result: **CURED** on the shared classifier source of truth. This is scorer-free
apparatus work; it did not move either frontier.

## Root cause

The defect is a **broken claim-co-occurrence predicate**, not a special case for
`ORPHAN-LITERAL`.

Before the cure, `magnitude_dismissal_candidates` joined an arbitrary three-line
window and accepted it when `_DISMISSAL.search(passage)` and
`_MAGNITUDE.search(passage)` were both true. The two matches did not have to be
part of the same claim. In `M1R5C_REVIEW.md`, `ORPHAN` matched inside the <!-- # MAGNITUDE_DISMISSAL_OK: source-level diagnosis of a false positive, not a dismissal -->
provenance-class noun `ORPHAN-LITERAL` on source line 139, while `marginal`
matched an unrelated provenance description on source line 141. Their only
relationship was proximity.

The pre-cure output was:

```text
line_count 387
dismissal match: line 139, ORPHAN inside ORPHAN-LITERAL
magnitude match: line 141, marginal
candidate: line 140
.omx/research/ddm_m1r4_20260808/M1R5C_REVIEW.md:140: magnitude-based dismissal without relative significance or a measured-un-recoverability citation — "| `safety_bound_steps_by_key` | 2 | IMPORTED (3250) / ORPHAN-LITERAL (6500) | forecast versus unowned extension | | `executor.schedule` numeric/prose | 6 | BORROWED (base LR) / IMPORTED (250,3250) / D"
```

## Predicate before and after

Before, the operative predicate was:

```python
passage = "\n".join(lines[max(0, i - 1): i + 2])
if not (_DISMISSAL.search(passage) and _MAGNITUDE.search(passage)):
    continue
```

After, `tools/magnitude_dismissal_detector.py:201-230` requires linked claim
context. Same-line dismissal and magnitude matches remain positive. Cross-line
matches require an explicit causal phrase, an anaphoric or magnitude-led
continuation, or a colon introducing the reason; distinct Markdown table rows
cannot lend vocabulary to each other. The classifier now calls:

```python
window = lines[max(0, i - 1): i + 2]
if not _has_linked_cooccurrence(window):
    continue
```

No token was added to an exclusion list. The existing relative-significance,
measured-unrecoverability, waiver, discussion, fmtools, and fail-open behavior
was left intact.

## Corpus sweep

Scope: every readable `*.md` reached recursively below `.omx/research` in the
shared working tree. The primary pre-cure sweep reached **8,146/8,146 files**,
**1,281,880 lines**, with **0 unreadable files**.

- The detector vocabularies produced 34,027 raw regex hits.
- Uppercase hyphenated class labels accounted for 860 hits across 316 unique
  labels. The old predicate produced 687 candidates total; 44 candidates
  contained at least one such class-label hit. This broad stratum mixes real
  verdict labels with non-verdict provenance nouns, so 44 is not asserted as a
  false-positive count.
- The narrow provenance-shaped stratum requested by the charter (`*-LITERAL`,
  `*-ONLY`, and magnitude-prefixed class labels) contained **38 occurrences in
  16 files**. `ORPHAN-LITERAL` accounted for 16 occurrences. Exactly **1/38**
  occurrences induced an old-predicate candidate: M1R5C.
- A post-cure confirmation reached **8,147/8,147 files** and **1,281,968 lines**;
  one memo appeared concurrently in the shared worktree. It found **0**
  provenance-shaped induced candidates. Because the denominator changed by one
  file, pre/post total-candidate counts are diagnostic only, not a paired corpus
  experiment.

This is a population-level predicate defect even though the narrow live stratum <!-- # MAGNITUDE_DISMISSAL_OK: population diagnosis of false positives, not a dismissal -->
held one observed trigger. Independent recall found an earlier lexical false
positive in the RA1 source inventory: `orphan inventory` borrowed `label-noise` <!-- # MAGNITUDE_DISMISSAL_OK: quoted prior false-positive source list, not a dismissal -->
from the next source-list line. The cured predicate also returns no candidate on
that unchanged passage.

## Controls

### Positive — real dismissal still refuses

Input: `We defer horizon-margin #169 because its measured delta-S is weak.` <!-- # MAGNITUDE_DISMISSAL_OK: verbatim positive-control fixture, not a research verdict -->

```text
flags=['positive_control.md:1: magnitude-based dismissal without relative significance or a measured-un-recoverability citation — "We defer horizon-margin #169 because its measured delta-S is weak."']
decision=REFUSE
rc=1
```

### Negative — unchanged M1R5C review passes

```text
flags=[]
decision=PASS
rc=0
```

### Cure test

Would the detector report the same M1R5C candidate after only the predicate cure
was applied? **No.** The file bytes are unchanged at SHA-256
`75a6b4d455cf35de05ea5b2ac4cc394b2df94b98e15415a7c8393bb7d032d0b6`;
the pre-cure classifier emitted one flag and the cured classifier emitted none.

## Guard and verification

The regression guard executes the unchanged M1R5C file and the prior RA1 source
inventory, plus crafted table-row, same-row positive, causal-wrap, anaphoric, and
magnitude-led controls.

- `src/tac/tests/test_magnitude_dismissal_detector.py`: **34 passed**.
- #404/refusal-positive-control subset of `test_confound_gates.py`: **4 passed,
  159 deselected**.
- Full `test_confound_gates.py`: **158 passed, 5 failed** on unrelated live-tree
  bounds already present in dirty shared surfaces (hosc launch history, raw VM
  basis, observer-role split, designed-stub count, optimizer-state fixture).
  No #404 test failed.
- `py_compile`: rc=0.
- `ruff --select F`: rc=0.
- `git diff --check`: rc=0.
- Two `review_tracker.py mark-file` passes were executed for each changed Python
  file.

Two serialized landings:

1. `43c9754007c4d1b692186ff8e6d8bfc1b9630147` — predicate cure.
2. `fde131ae3e294c20f2f9a4a7e532e124fd9938e8` — regression guard.

## RECALL EVIDENCE

Sources and queries consulted:

- Full `.omx/research` content sweep for `magnitude-dismissal`,
  `ORPHAN-LITERAL`, provenance-class language, and lexical false positives.
- `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` content search for #404 and the
  detector lineage.
- `tools/list_canonical_equations.py --json`: 429 equations examined, 0 relevant
  entries for this apparatus predicate.
- Design/catalog sources: `magnitude_dismissal_hook_build_20260708.md`,
  `relative_significance_reaudit_20260708.md`, and Catalog #404 in
  `docs/meta_bug_class_catalog.md`.
- Task/queue stores: the current `ddm_dg1` queue rows were found; no separate
  canonical-task-status row for #994 was found in the searched stores.
- Detector history and the RA1 adjudication were read beyond the charter seed.

What changed in the plan: the prior RA1 false positive and the class-label
population ruled out an `ORPHAN-LITERAL` exemption. The implementation moved the
fix to linked claim context and added guards for both known false-positive
shapes.

## Boundaries

Measured: predicate behavior, corpus denominators, control return codes, tests,
file hashes, and commits. Not measured: scorer components, archive bytes, or any
score. No scorer slot, Metal/MPS/CUDA, launch, dispatch, promotion, upstream edit,
public-PR-intake edit, or protected-file edit occurred.

Own-vehicle frontier remains **S = 0.7534578126155775 @ 357,837 B
[macOS-CPU advisory]**. Contest pointer **0.19108 is borrowed and unmoved**.
