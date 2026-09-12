Implemented and landed the manifest-pin custody fix plus enforcement guard.

- FIX: `3e4645313` — transactional manifest rebinding through [receiver_manifest.py](/Users/adpena/Projects/pact/src/tac/receiver_manifest.py:33).
- GUARD: `62dce685e` — STRICT Catalog #420 in [preflight.py](/Users/adpena/Projects/pact/src/tac/preflight.py:95299).
- Automatic AGENTS/CLAUDE synchronization landed separately as `e41e81c94`.
- All three producers and five production callers are protected.
- Missing target rows refuse before mutation; failures restore both `inflate.py` and the manifest; no-ops preserve manifest bytes.
- Both valid archive-row policies remain supported.
- Verification: 111 tests passed; Ruff and `git diff --check` clean.
- Live guard: 0 violations, 3 producers, 5 callers. The production denominator was 7,085 during landing review and 7,086 post-commit due to concurrent shared-worktree activity.
- Serializer completed successfully with `[no-triality] [p0-ledger-ok]`; the index is empty.
- Full handoff: [ddm_mpg1_manifest_pin_guard_20260912.md](/Users/adpena/Projects/pact/.omx/research/ddm_mpg1_manifest_pin_guard_20260912.md).

No scorer, paid dispatch, candidate build, or long job ran. No candidate, sealed, upstream, or volume tree was modified. The frontier remains **S 0.13632299781031237 at 179,153 B `[contest-CUDA T4 n600]`**.

## LIVE-HYPOTHESES

- None within this completed charter.

## DEAD-ENDS

- Editing retained candidate/pass-6 trees directly: those are custody evidence.
- Rebinding manifests caller-by-caller: the producer must own the transaction.
- Hard-coding a 49-row manifest: valid receivers may include or exclude `archive.zip`.
- Treating post-materialization seal validation as prevention: Catalog #420 now blocks the defective producer class earlier.