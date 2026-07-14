# Consolidation Code-Land Receipt — 2026-07-14

## Scope and authority

This receipt closes the drained-fleet consolidation arm. It covers the reviewed code, the three
shared source-of-truth registries, the narrow quarantine decisions, and the post-land regression
disposition. It does **not** authorize a launch, score, promotion, or frontier-pointer change.

The later main-supervisor directive at `2026-07-14T23:37:58.052511+00:00` superseded the request to
finish all 2,609 pytest files after main independently compared the landing against the
pre-consolidation baseline `f8074d6e1c`. The disposition below therefore distinguishes measured
partial-suite failures from consolidation regressions; it does not relabel red baseline gates as
green.

## Reviewed landings

Eighteen coherent serializer commits landed before this receipt:

| # | Commit | Reviewed group | Verification / disposition |
|---:|---|---|---|
| 1 | `f43554bbf6` | RIPO deterministic Fisher trust region | 34 focused tests passed |
| 2 | `3c38aed572` | Dormant M+Adam reference | 19 focused tests passed |
| 3 | `4925036623` | Polar-Fourier API and legacy parity | 39 focused tests passed |
| 4 | `ae5b57525b` | Retire unbound banked-pose verdict | 11 focused tests passed |
| 5 | `b70e51454e` | De-authorize ambient costate cosine | 25 focused tests passed |
| 6 | `7263015bd1` | Label ambient-costate authority | 24 focused tests passed |
| 7 | `b673d8fb85` | Fourier naming and resume-compatible aliases | 121 focused tests passed across 15 files |
| 8 | `4cf6be0bcd` | De-authorize dashboard ambient-R1 fallback | 11 focused tests passed |
| 9 | `d900c3cc1c` | Spatial-KL support normalization | 220 focused tests passed across 6 files |
| 10 | `7be3499ca6` | Memory-governor plateau admission | 196 focused tests passed across 6 files |
| 11 | `f8074d6e1c` | Live PoseNet throughput policy | 16 focused tests passed across 3 files |
| 12 | `84b5007a05` | Retire unreproduced directional claim | Claim removed rather than promoted without custody |
| 13 | `5bacd6d5b1` | Canonical-equations SoT reconciliation | Latest-wins loader and equation queries verified |
| 14 | `cc7c02f78b` | Lane-registry and audit reconciliation | Catalog #90 tests: 72 passed; 1,884 lanes validate |
| 15 | `587ed7f98a` | Canonical task-status reconciliation | Targeted tests: 7 passed; 168 rows strict-valid |
| 16 | `6bae3f73d3` | Portable docs-line preflight correction | 17 focused tests passed |
| 17 | `cca5f1a1af` | Truthful same-line lock descriptor annotation | Check 131: zero findings; ruff and compile passed |
| 18 | `0e9229ffd7` | Quarantine two implementation-orphaned tests | Missing implementations were absent on reachable refs; bytes remain in git history |

The review pruned 77 unsafe, false-custody, killed-arm, or otherwise unowned files; reverted 55
unsafe or broad mechanical tracked edits; and left no untracked code. Python files that landed were
read and marked through the review tracker; no Python review override was used.

## Source-of-truth reconciliation

### Canonical equations

- MEASURED: 701 append-only events resolve to 332 unique current law IDs.
- MEASURED: the #502 basis law has one registration plus two append-only supersession events.
- MEASURED: 21 orphan arm rows were pruned before landing.
- MEASURED: no new duplicate registration was introduced relative to `HEAD` at reconciliation.
- Historical provenance remains append-only: 60 pre-existing law IDs have multiple historical
  registration events, but latest-wins resolution exposes one current law per ID.

### Lane registry

- MEASURED: 24 additive lanes were retained; no lane was dropped.
- MEASURED: 1,884 lane IDs validate with zero duplicate IDs.
- Two stale #502 evidence claims were removed and that lane was demoted to L0 rather than retaining
  false measurement authority.

### Canonical task status

- MEASURED: 168 rows pass strict validation with one registration per task and no dangling
  consolidation transition.
- RIPO is completed at `f43554bbf6`; D41 is blocked after its drained owner; #502/#503 retain explicit
  quarantine notes.
- Thirteen historical memo-pointer violations exist on the baseline and none was introduced by this
  consolidation.

The `tac.canonical_equations` current-law query and `tac.witness_dsl.lever_registry` both loaded.
Lever-registry measurement: 80 unique factories, 398 trainer flags, 296 mapped flags, 102 visible
gaps, and zero stale emitted flags. The focused equation/lever verification passed 48 tests.

## Correctness and regression verdict

### Preflight

`.venv/bin/python -m tac.preflight` reached the codebase-drift gate after two pile-introduced scanner
issues were fixed. It remains **RED on 12 baseline violations**: nine `experiments/launch_*.py`
findings, two findings in `experiments/run_capstone_capacity_ablation_2x2.sh`, and one in
`experiments/manim_levelset/render.sh`. The preflight implementation, scanner tests, and implicated
paths were present before this consolidation. These findings require a separately governed
rebaseline or substantive fixes; no blanket exemption was added here.

### Pytest

The resumable post-land sharded run collected 42,287 nodes from 2,609 files. At the supervisor stop
point, 522 files had completed with:

- 8,540 collected nodes in completed shards;
- 8,291 passed, 122 failed, 11 skipped, and 0 errors;
- 64 failure files and 3 abnormal files;
- 1,809.845334 test-seconds.

Counts do not sum to collected nodes because terminated abnormal files had unreported remaining
nodes. Two pathological test mocks attempted multi-terabyte-scale `nn.Linear` allocations; those
shards were terminated after attribution. A stale dispatch test monkeypatched `subprocess.call` while
production uses `subprocess.run`, briefly reaching the local Modal CLI; it was stopped immediately.
The canonical ledger records `pre_spawn_fatal`, no Modal call ID, and `$0.00` actual cost. The one
synthetic test-pollution row was removed exactly. No paid or remote job launched.

Main-supervisor then compared failures against baseline `f8074d6e1c` and issued the binding verdict:
the surfaced failures are baseline repo-scan/candidate-builder debt. The only two tests that passed
on baseline but flipped at the landing were PR85 real-smoke probes whose subject code is byte-identical
between baseline and landing. They are flaky replay-runtime-support probes, not consolidation
regressions. Per that directive, the remaining 2,087 files were not ground through and baseline debt
was not expanded into this consolidation.

**Regression verdict: CLEAN versus pre-consolidation baseline `f8074d6e1c`; preflight and the partial
suite are not globally green.**

## Drift-gate disposition

No apparatus drift-gate commit was forced. The three diagnosed bugs remain OWED:

1. retry inherits a broken sandbox and lacks resumability;
2. in-flight workspace-write arms lack a retrofit/cap;
3. drain timeout can appear successful instead of exiting nonzero.

The first two require broader delegate-runtime design and compatibility work; the third owning source
was not established narrowly enough for a safe `$0` fix. They remain recorded in the consolidation
memo for separate two-landing fixes and strict tests.

## Closure

- Main branch remained the source of truth.
- Frontier pointer: unchanged.
- Paid/GPU launch: none; accidental local CLI reach was contained before spawn (`$0`, no call ID).
- Working tree: clean before receipt creation and required clean immediately after its serializer
  commit.

DAG FEED: `FEED-CONSOLIDATION-CODE-LAND-20260714` — 18 reviewed code/SoT/quarantine landings;
equations, lanes, tasks, and lever-registry current views verified; regression clean versus
`f8074d6e1c`; 12 preflight findings and partial-suite reds retained as baseline debt; three apparatus
drift fixes remain OWED; pointer unchanged.
