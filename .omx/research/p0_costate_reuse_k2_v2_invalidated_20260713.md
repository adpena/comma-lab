# P0 K2 v2 replay invalidation — no evidence

**Status:** `INVALIDATED_NO_EVIDENCE`  
**Invalidated:** `2026-07-13T23:23:44Z`  
**Scope:** every partial row or possible aggregate under
`experiments/results/p0_costate_reuse_k2_n600_v2_20260713/`

The v2 run contract included `git_head_at_launch` in strict semantic resume equality. An unrelated
sibling serializer commit advanced repository `HEAD` from
`789766ddbe4de3267f1fa358c90b046d6fe7abbe` to
`24b30e54cb9b296f99221433878337fa738a2663` while the source/input bytes pinned by the replay stayed
unchanged. The running process could continue, but a crash could not pass its own resume contract.
That violates the P0 resumability contract.

The run was interrupted after 59 atomic pair records. No stage manifest, aggregate admission, or
measurement receipt was complete. **None of those rows may be cited, migrated, re-signed, aggregated,
or used as warm-start evidence.** The local directory contains a matching machine-readable
`INVALIDATED_NO_EVIDENCE.json` binding run-contract SHA-256
`b9546120c172c3d78294c0dc0d64f95ef18afb29a399cf603519e0d54a9afb64`.

The v3 correction retains launch `HEAD` as provenance but excludes only that field from semantic
resume equality. Source, input, output directory, storage plan, objective, scorer, admission spec,
constants, and bounded/full-run mode remain byte-strict. v3 uses a new empty output directory and a
new storage-plan path, so v1/v2 rows cannot enter it by normal invocation.

`verdict_scope`: this invalidates the v2 evidence artifact, not the guarded K2 formulation. Pointer
delta is `NONE`; score claim is false; no training or paid dispatch occurred.
