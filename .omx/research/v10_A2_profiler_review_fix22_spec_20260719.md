# V10 A2 profiler FIX22 frozen implementation contract

Date: 2026-07-19  
Lane: `v10_A2_profiler_20260718`  
Input bundle: `working-tree-sha256:420ede1222cfd128fe7082f1bab1e1fc93743bd8007abca8168b599396046be8`  
Input review state: rounds 35-36 `CLEAN`, round 37 independently `NOT_CLEAN`; counter `0/3`  
Verdict scope: generic atomic preconstruction/publication custody only. No launch, score,
promotion, pointer movement, n600 measurement, or profiler-math verdict is authorized.

## Reproduced defect

FIX21 validates an absent-target transaction namespace, returns, and later constructs scratch and
transaction evidence without binding the transaction namespace observed by that first gate. Root
injected an empty transaction generation immediately after
`_validate_atomic_prepublication_namespace` returned. The empty file was accepted later as a strict
prefix of the newly constructed record; the desired target was published and the provenance-free
file was retired. Observed result:

```text
target_exists=True
target_payload={"state":"desired"}\n
active_transactions=[]
retained_count=2
```

This is the same P0 partial-only adoption class shifted into the gap between preflight and
construction. FIX21's post-record transaction fingerprint cannot distinguish roles that appeared
after preflight but before its first snapshot.

## Exact implementation boundary

Only these files may change:

- `src/tac/witness_control/segnet_head_feature_cache.py`
- `src/tac/tests/test_segnet_head_feature_cache.py`

The other six bundle files remain byte-identical.

## Required semantics

### A. Prepublication observation is a typed boundary, not a retained-only tuple

The read-only prepublication gate must return descriptor-bound fingerprints for all target-scoped
roles it observed:

- active prepared/generation scratch;
- active transaction generations, including complete records and their exact strict-prefix
  constructions;
- retained roles and active completion proof/construction roles.

For an absent target, partial transaction constructions are admissible only when the same
prepublication observation also contains exactly one structurally valid complete fresh authority
and its exact designated source. Partial-only evidence remains an immutable blocker.

Immediately after the gate returns and before creating any prepared/generation/transaction byte,
`_atomic_bytes` must re-enumerate the complete namespace and require byte-, inode-, name-, and role-
identical agreement with that typed boundary. The stable retained/completion fingerprint for an
absent target must also still be empty. A role injected after the gate therefore refuses before the
writer creates any byte.

### B. Construction admits only its own exact deltas

After the immediate boundary revalidation:

- scratch construction may add only the one exact prepared or complete-generation inode created by
  this call; every pre-existing scratch inode remains identical;
- transaction construction may either reuse the exact complete transaction already present at the
  prepublication boundary, or add exactly one complete transaction generation returned by this
  call;
- a strict-prefix transaction path is never newly admitted merely because it is a prefix of the
  new record. Such a path is authorized only if it was already fingerprinted with its complete
  authority at prepublication;
- before the first post-record boundary, compare the actual namespace to the prepublication set
  plus only those explicitly created exact deltas. Extra, missing, substituted, or duplicated
  paths refuse before target publication.

The existing after-hook boundary must then compare against this authorized construction result,
not against an arbitrary first snapshot that may already contain late roles.

### C. Required regressions

At minimum, tests must prove:

1. the exact root reproduction: inject `b""` transaction evidence after prepublication returns;
   the target stays absent and the tree equals the post-injection/pre-writer snapshot (no prepared,
   generation, completion, retention, or target created by the rejected call);
2. the same post-prepublication injection with an existing target preserves its prior bytes and
   inode;
3. a non-empty strict-prefix transaction injected after prepublication is not adopted;
4. an empty or non-empty strict-prefix transaction injected immediately after transaction writing
   but before the first FIX21 boundary refuses before target publication;
5. an active scratch role injected after prepublication is not silently admitted;
6. a valid interrupted fresh transaction with one complete authority, its exact designated source,
   and a pre-existing strict-prefix construction remains resumable;
7. all FIX20/FIX21 completion, late-role, absent-target, and profiler integration regressions remain
   green.

Tests assert namespace bytes plus inode preservation where applicable, not only exception text.

## Acceptance and seal

Run the exact combined three-module suite, Ruff check, Ruff format check, `py_compile`,
`git diff --check`, forbidden-transform scan, destructive-operation scan, and scorer-loader scan on
one frozen lexicographically hashed eight-file bundle. Verify `uv.lock` remains
`4e492923a868788516200c8fad14d9087012bdffb88a49be54327be4306db2d3`; do not touch the sacred
run. Then obtain three independent zero-finding reviews on those exact bytes. Any finding resets
the counter. Only `CLEAN x3` authorizes a source-freeze commit, and that commit still requires MAIN
landing review before merge or any governed n600 execution.

