# Meta-Lagrangian Archive Manifest Custody - 2026-05-06

## Context

The meta-Lagrangian allocator ranks planning atoms and marks which rows are
ready for stack review. Rows can carry `archive_manifest_path` and
`archive_manifest_sha256` so byte-closed candidate evidence can be connected to
planning records.

## Finding

The allocator previously treated a nonempty path plus nonempty SHA string as
`byte_closed_archive_manifest_attached=true`. It did not verify that the file
exists or that the file bytes match the declared SHA-256. A stale or invented
manifest path could therefore make a row appear archive-ready for stack review.

Evidence grade: `empirical` planning-custody hardening, not score evidence.

## Change

- Verify archive manifest path existence, file type, SHA-256 format, and
  SHA-256 match before setting `byte_closed_archive_manifest_attached=true`.
- Emit `archive_manifest_custody` with exact status fields and actual SHA when
  available.
- Add fail-closed blockers for missing, non-file, invalid-SHA, and mismatched
  archive manifests.
- Add regression tests for verified, mismatched, and missing manifests.

## Verification

Focused:

```text
.venv/bin/python -m pytest src/tac/tests/test_meta_lagrangian_allocator.py -q
```

Full preflight:

```text
.venv/bin/python tools/all_lanes_preflight.py --timings
```

## Promotion Status

This is planning-DX and custody hardening only. It does not dispatch GPU work
and does not convert any atom into score evidence. Exact CUDA auth eval remains
required for promotion.
