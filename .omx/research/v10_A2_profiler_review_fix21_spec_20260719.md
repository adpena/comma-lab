# V10 A2 profiler FIX21 frozen implementation contract

Date: 2026-07-19  
Lane: `v10_A2_profiler_20260718`  
Input bundle: `working-tree-sha256:98e9e70936f3cb77c44025654b22b894b9b895c9c6d2b58245d6ea3431d4e023`  
Input review state: rounds 33 and 34 independently `NOT_CLEAN`; clean counter `0/3`  
Verdict scope: generic atomic metadata publication/restart custody only. No launch, score,
promotion, pointer movement, n600 measurement, or profiler-math verdict is authorized here.

## Reproduced defects

1. **Late active namespace role crosses publication.** The final authorization boundary in
   `_commit_bound_atomic_source` freezes only retained roles. A generation, transaction, or
   completion role injected after transaction construction can therefore survive the boundary;
   the target is published/exchanged and cleanup refuses only afterward. R33 reproduced a fresh
   target publication with a late foreign active generation.
2. **Partial-only transaction is adopted for an absent target.** The absent-target branch of
   `_validate_atomic_prepublication_namespace` ignores active transaction evidence. An empty
   partial transaction with no complete authority or designated source is later retired while a
   newly generated desired target is published. R34 reproduced this exactly.

Both are P0 resumability/custody failures because a rejected operation may mutate the authority
tree or convert provenance-free bytes into admitted history.

## Exact implementation boundary

Only these implementation/test files may change for FIX21:

- `src/tac/witness_control/segnet_head_feature_cache.py`
- `src/tac/tests/test_segnet_head_feature_cache.py`

Profiler code and the other six frozen bundle files remain byte-identical unless a new independent
review proves a direct integration defect.

## Required semantics

### A. Read-only absent-target transaction gate

Before creating any prepared/generation byte for an absent target:

- enumerate and descriptor-read every active transaction generation;
- reject partial-only evidence unless each partial is an exact strict prefix of one complete,
  structurally valid transaction authority;
- require all complete records to describe one consistent `FRESH_TARGET` transition for the exact
  desired bytes and exact consumer-authorization digest;
- require the transaction-designated source basename, inode identity, size, and SHA-256 to exist
  in the active scratch namespace and contain the exact desired payload;
- reject missing, duplicate, contradictory, malformed, or foreign transaction/source evidence
  without changing any pre-existing byte, inode, or target state.

A valid interrupted fresh transaction with its exact designated source remains resumable.

### B. Full final publication boundary

The final check immediately before rename-no-replace or atomic exchange must cover the entire
target-scoped authority namespace, not retained roles alone:

- retained scratch/transaction/completion evidence;
- active prepared/generation scratch;
- active transaction generations;
- active completion proofs and completion-construction generations.

Before transaction construction, descriptor-bind the admitted active scratch set. After the
durable transaction generation exists and again after the test hook, re-enumerate the namespace
read-only and require:

- the active scratch set is exactly the previously admitted set, with identical basenames,
  descriptor identities, lengths, and SHA-256 values, and the same designated source;
- every active transaction is the exact transaction record or an exact strict prefix construction
  of it; no contradictory complete record is accepted;
- the prepublication retained/completion namespace is byte-, identity-, and name-identical;
- no new, removed, substituted, malformed, or unadmitted target-scoped role exists.

Any mismatch refuses before the publication syscall. For an existing target, its exact prior bytes
and inode remain authoritative. For a fresh target, the target remains absent. The implementation
must not delete, replace, exchange, retain, or otherwise mutate evidence while making this refusal.

### C. Required regressions

At minimum, tests must prove:

1. late foreign active generation at the commit hook refuses with a fresh target still absent;
2. the same class on an existing target leaves prior bytes and inode unchanged;
3. late foreign active transaction refuses before publication;
4. late active completion proof or completion-construction role refuses before publication;
5. empty partial-only transaction plus absent target refuses with an exact flat-tree snapshot;
6. non-empty partial transaction lacking a complete authority refuses identically;
7. complete fresh transaction missing/substituting its designated source refuses before scratch
   creation;
8. a valid interrupted fresh transaction plus exact designated source remains resumable;
9. the existing late-retained-role and FIX20 completion-history regressions remain green.

Tests must assert both content and inode/namespace preservation where applicable, not merely an
exception string.

## Acceptance and seal

Run, on one frozen eight-file bundle:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_segnet_head_feature_cache.py \
  src/tac/tests/test_uint8_lattice_profile.py \
  src/tac/tests/test_uint8_lattice_seed_anchor.py
.venv/bin/ruff check <exact eight bundle files>
.venv/bin/ruff format --check <exact eight bundle files>
.venv/bin/python -m py_compile <exact eight bundle files>
git diff --check
```

Re-run the existing forbidden-transform, destructive-operation, and scorer-loader static scans;
recompute the exact eight-file bundle SHA with the existing lexicographic `shasum` algorithm; and
verify `uv.lock` plus the sacred 18-file tree are unchanged. Then dispatch three independent
zero-finding reviews of those exact bytes. Any finding resets the counter and requires a new frozen
spec. Only `CLEAN x3` authorizes the source-freeze commit, which still requires MAIN landing review.

