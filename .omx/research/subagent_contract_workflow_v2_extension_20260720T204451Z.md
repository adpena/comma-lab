# Subagent contract workflow-v2 extension

**UTC:** 2026-07-20T20:44:51Z  
**Lane:** `workflow_contract_extension_20260720T203932Z`  
**Authority:** delegated apparatus-only build; no score, GPU, paid-dispatch, or pointer authority.

## Implementation specification

**DERIVED objective.** Extend `tac.subagent_contract.standard_contract()` additively with five
unconditional core doctrine blocks: `RESEARCH_AUTHORITY`, `DECOMPOSE_HEADLINE`,
`TIEBREAK_LEAST_COMPLEXITY`, `MASTER_THESIS_FRAMING`, and `VERDICT_SCOPE_LADDER`.

**DERIVED constraints.** Preserve every existing constant and key phrase; add each new constant to
`__all__`, `CONTRACT_CONSTANT_NAMES`, `KEY_PHRASES`, the hardcoded preflight required-constant
tuple, and the composer. The existing composer exposes only `review` and `triality` opt-outs, so no
new opt-out API is authorized. Extend focused tests to prove constant registration, load-bearing
phrases, unconditional composition, and integrity-gate coverage.

**UNKNOWN source caveat.** The delegated authority names
`docs/operating_workflow_v2_velocity_rigor_autonomy_20260720.md` as the first source, but that path
is absent in this worktree and absent at `main` according to `git cat-file`. The delegated prompt's
explicit five-block specification and all four named durable memory directives were available and
read in full; MAIN must confirm that the missing document would not alter the five enumerated
requirements.

**DERIVED acceptance.** The focused contract tests pass; a direct strict invocation of
`check_subagent_contract_module_integrity` passes; two adversarial review-tracker passes cover
every touched Python file; serializer HEAD-content verification passes.

## Verification receipts

**STORES CONSULTED:** delegated authority file (SHA-256
`57828c6b7ab07502849426addcf318c534a3bec0775f2bc911f7a6c8165f83ae`); `CLAUDE.md`;
`AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`; the existing contract,
integrity gate, and focused tests; the four named 2026-07-20/2026-07-08 memory directives; lane
registry; subagent-progress ledger; per-arm inbox; fleet broadcast through
`2026-07-19T19:48:01Z`. Deliberately not consulted: scorer/run artifacts, because this apparatus
landing makes no empirical or score claim.

**MEASURED focused-test receipt** (`fresh-eyes-reviewed(2)`):

```text
$ PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest src/tac/tests/test_subagent_contract.py src/tac/tests/test_subagent_contract_eightfold.py src/tac/tests/preflight/test_check_reasoning_echo_and_subagent_contract.py -q
......................................................................   [100%]
70 passed in 1.70s
```

**MEASURED strict integrity-gate receipt** (`fresh-eyes-reviewed(2)`):

```text
  [subagent-contract] check_subagent_contract_module_integrity: OK
STRICT_GATE_RESULT violations=[]
```

**MEASURED static/adversarial receipts** (`fresh-eyes-reviewed(2)`):

```text
ADVERSARIAL_MUTATION_MATRIX 5 constants x 3 failure modes = 15/15 caught
COMPOSER_VARIANTS 4/4 unconditional variants contain all five blocks
  [reasoning-echo] check_no_reasoning_echo_instructions: OK (106 prompt-surface file(s) scanned)
REASONING_ECHO_GATE_RESULT violations=[]
All checks passed!
```

**CONFIRMED review provenance.** Round 1 found an empty-composer fail-open in the inherited
truthiness guards; the fix removed both guards and added a regression, which reset the clean-pass
counter. After the fix, review-tracker pass `workflow-v2-round1-clean` covered every touched Python
file. A separate mutation-matrix pass then recorded `workflow-v2-round2-clean` for the same four
files: `src/tac/subagent_contract.py`, `src/tac/preflight.py`,
`src/tac/tests/test_subagent_contract.py`, and
`src/tac/tests/preflight/test_check_reasoning_echo_and_subagent_contract.py`.

**UNKNOWN unrelated repository gate.** `tools/lane_maturity.py validate` reports 110 inherited
missing-evidence paths on historical lanes. None names this new L0 lane; no attempt was made to
rewrite or waive those unrelated records. This landing's acceptance gate is the focused strict
contract integrity receipt above.

## Triality and authority

- **DSL/tooling leg:** this landing is the canonical composed-dispatch contract and its strict
  anti-rot gate.
- **DAG leg:** `FEED-subagent-contract-workflow-v2` is supplied in the sibling DAG FEED artifact.
- **Equations leg:** N/A — apparatus-only contract work; no measured law or empirical anchor is
  introduced.
- **Pointer delta:** none authorized or claimed.

## MAIN landing review

Required. MAIN must review the branch diff, the missing-source caveat, focused receipts, and exact
committed content before merging; this branch is not itself promotion authority.
