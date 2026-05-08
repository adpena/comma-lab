# Contest Packet Compiler Low-Level Hardening - Worker K - 2026-05-07

## Scope

Worker K advanced the deterministic contest packet compiler/checker lane in a
narrow Python-oracle slice. This is byte-custody and conformance tooling only.
It does not optimize score-affecting bytes, run scorers, dispatch remote work,
or claim score.

## Changed Surface

- `src/tac/submission_packet_compiler.py`
  - Added explicit ZIP wire metadata for future Rust/Zig/C/ASM parity:
    member order index, data offset, payload bytes/SHA-256, compressed payload
    SHA-256, local-header record, central-directory extra/comment lengths,
    create system, external attributes, and POSIX permissions.
  - Added fail-closed deterministic metadata blockers for noncanonical ZIP
    timestamps, permissions, create system, extras/comments, data descriptors,
    symlink entries, unsupported methods, encrypted members, duplicate names,
    and local/central name mismatches.
  - Added `runtime_tree_manifest.v1` with path/bytes/SHA-256/mode rows,
    mode-aware tree SHA-256, runtime path blockers, and a
    `--expected-runtime-tree-sha256` hook for stricter packet/release gates.
  - Added explicit `score_dispatch_gate` fields:
    `score_claim=false`, `promotion_eligible=false`, `dispatchable=false`,
    `ready_for_exact_eval_dispatch=false`, and blockers requiring exact CUDA
    auth eval, byte-closed archive custody, Level-2 dispatch claim, and strict
    pre-submission compliance before any promotion.
- `tools/contest_packet_compiler.py`
  - Added an operator-facing alias for the deterministic packet compiler.
    The CLI writes the same manifest schema and prints shape blockers plus
    score/dispatch blockers.
- `src/tac/tests/test_contest_packet_compiler.py`
  - Added focused coverage for deterministic ZIP metadata, runtime-tree
    manifest hooks, score/dispatch fail-closed semantics, noncanonical ZIP
    blockers, and the new CLI alias.
- `src/tac/tests/test_submission_packet_compiler.py`
  - Extended the fake native `zipwire` conformance payload helper so existing
    native parity tests compare the new core ZIP wire fields.

## Verification

```text
.venv/bin/python -m ruff check \
  src/tac/submission_packet_compiler.py \
  tools/contest_packet_compiler.py \
  src/tac/tests/test_submission_packet_compiler.py \
  src/tac/tests/test_contest_packet_compiler.py
```

Result: passed.

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_submission_packet_compiler.py \
  src/tac/tests/test_contest_packet_compiler.py -q
```

Result: 14 passed, 1 expected duplicate-ZIP-member warning from the regression
fixture.

## Compliance And Blockers

- Evidence grade remains `byte_custody_only`.
- `score_claim=false`; no SegNet/PoseNet component values were computed.
- `dispatchable=false`; no remote/GPU work was launched.
- Exact score truth remains `archive.zip -> inflate.sh -> upstream/evaluate.py`
  on CUDA through the canonical auth-eval path.
- Any future score or promotion use still requires a byte-closed archive,
  strict pre-submission compliance, Level-2 dispatch claim, exact CUDA auth
  eval, and a reviewed runtime-tree manifest match.

## Next Narrow Step

Use this Python oracle as the parity target for an opt-in native `zipwire`
inspector. The first native success criterion should be JSON field parity on
synthetic golden vectors and public archive custody samples, not adoption in a
contest inflate path.
