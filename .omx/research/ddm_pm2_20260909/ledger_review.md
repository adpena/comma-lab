# Gate 3 review receipt

Axis: [macOS-CPU static apparatus audit]. `score_claim=false`. No scorer, pricing,
encoding, candidate payload sweep, or frontier measurement was run.

Catalog claim #414: `check_ddm_ledger_before_optional_dump`, STRICT in
`preflight_all`. The #415 umbrella `check_instrument_binds_to_live_pointer` delegates
lazily to parent-owned `comma_lab.instrument_gates.audit_instrument_gate_wiring`.

The typed initial census found **14 call-site findings in 4/698 producers**:
rp1 9, nr1 2, pc2 2, tb2 1. Four producers is within the charter's predicted 2–6
instances when an instance denotes a producer; 14 call sites exceeds that range
when an instance denotes a call site. The final census is **0/698**, STRICT.
Both source-scoped receipts are adjacent JSON files. These are detector findings,
not 14 reproduced production failures. Raw broad discovery had 730 numpy
concatenate/stack call lines before typed accepted-set classification.

Fixes: rp1's older rank persists its bit ledger before assembling its optional
candidate NPZ, with typed empty arrays; its already-cured mixer remains numeric
behavior equivalent. nr1 refuses an empty accepted-event bank explicitly. tb2
refuses an empty top-cost candidate set after its existing bit-ledger writes.
pc2 persists incremental coarsen/hybrid price rows before optional coefficient
NPZ dumps, preserving final receipt schema and measured values.

## RECALL EVIDENCE

- Memory registry query: `ledger|preflight|catalog` in
  `/Users/adpena/.codex/memories/MEMORY.md`; lines 35–37 identify #899's same-line
  substantive-waiver precedent. Changed the plan by reusing the existing
  `PreflightError`/STRICT surface and strengthening comment parsing rather than
  treating text containing a waiver token as an actual comment.
- Full memo/receipt corpus bounded content query:
  `ledger.{0,80}(dump|persist)|concatenate\(\[\]\)|ledger loss` in `.omx/research/`
  markdown files; inspected `ddm_rxc1_restartable_exact_coder_20260901.md`,
  especially its per-frame-ledger-in-checkpoint custody. Beyond charter seeds,
  this demonstrated why NPZ checkpoints already carrying per-frame ledger data
  are persistence and must not be confused with optional candidate dumps.
- Equations registry: ran `tools/list_canonical_equations.py --json`, searched
  `ledger|persist|empty|dump`. Returned ledger/reporting metadata; did not find
  a ledger-before-dump or empty-accepted-set law in that bounded registry query.
- Research-index / DAG / design-doc / task surfaces searched with content query
  `ledger.{0,50}(persist|dump)|empty.{0,40}(candidate|accepted)` over
  `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, `docs/`, and
  `.omx/state/canonical_task_status.jsonl`. The DAG's empty-candidate-manifest
  discussion and older catalog hits did not add an executable replacement for
  the requested producer gate. This did not change the implementation plan.
- Live sibling source inspection found that rp1 mixer already has a ledger-first
  empty-safe cure, while its older `cmd_rank` does not. Ported the behavior to the
  older sibling without launching either coder.

## Verification and review

37 regression tests pass in `src/tac/tests/test_ddm_pm2_ledger_preflight.py`.
They include actual rp1 writer execution extracted from its production AST to
avoid importing a scorer/coder, empty typed outputs, a forced NPZ-write failure
which leaves the real ledger readable, and nonempty array-value equivalence.
Ruff is clean for all six Python files owned by this arm.

Review 1: candidate alias/parameter taint and expiry of nonempty proof after
`clear`/reassignment were adversarially checked and corrected. Review 2: literal
empty arguments, fake waiver strings, and conditional earlier ledger writes
were checked and corrected. The final static census took 10.19 seconds on this
host; it is not a scorer timing or contest authority claim.

## Boundaries

This is a bounded intra-function AST gate over top-level `experiments/ddm_*.py`,
not a general Python theorem prover. It recognizes `np`/`numpy` concatenate,
stack and NPZ calls; empty accepted/keep collections and aliases; candidate
collections visibly filtered by branches; and named ledger/rows/bit-array sinks.
Arbitrary import aliases, opaque helper-side persistence and differently named
unrelated collections are outside the current static signature. Guards are
checked lexically, with early-empty-exit and mutation expiry; arbitrary runtime
mutation through an unknown helper is not inferred. Fixed per-input result
collections and checkpoints that already contain the ledger are not optional
accepted-set dumps. No silent waiver was added to production sources.

LIVE-HYPOTHESES: None added by this bounded implementation; all required findings
have a producer fix and a tested STRICT consumer.

DEAD-ENDS: Treating every numpy stack/concatenate as a selected-candidate bag
conflates fixed-size outputs with optional acceptance; classifying a checkpoint
which itself persists the ledger as an optional pre-ledger dump is also wrong.
Both paths were closed by source inspection and negative controls.

Parent integration review found that a ledger-named `.append` could falsely count as persistence. All append methods are now excluded from durable sinks. Four further regression controls cover that incident and nested-loop ledger dominance. The strict census was repeated after this correction.
