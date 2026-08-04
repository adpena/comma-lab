# COMMIT BLOCKER - ddm_uf1

Date: 2026-08-05. Arm: `uf1`.

Serializer command attempted with explicit `--files`, `--base-content-sha256 <file>=new`, and
post-edit `--expected-content-sha256 <file>=<sha256>` for the 11 UF1-owned files.

Result: blocked before commit. `git add` failed while indexing
`.omx/research/ddm_uf1_20260805/NEXT-IF-RESUMED.md`:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_uf1_20260805/NEXT-IF-RESUMED.md: failed to insert into database
error: unable to index file '.omx/research/ddm_uf1_20260805/NEXT-IF-RESUMED.md'
fatal: adding files failed
```

Post-failure inspection: `git diff --cached --name-status` was empty, so the failed serializer
attempt left no staged UF1 files.

Required harvest action: main operator or a git-write-capable harvest step should commit the UF1 file
set exactly, after recomputing post-edit shas:

- `src/tac/derived_upstream_refresh.py`
- `src/tac/tests/test_derived_upstream_refresh.py`
- `tools/build_ddm_uf1_refresh_registry.py`
- `tools/tests/test_build_ddm_uf1_refresh_registry.py`
- `.omx/research/ddm_uf1_20260805/UF1_RECEIPT.md`
- `.omx/research/ddm_uf1_20260805/NEXT-IF-RESUMED.md`
- `.omx/research/ddm_uf1_20260805/m66_gap_decomposition_qo1.json`
- `.omx/research/ddm_uf1_20260805/queued_refreshes.jsonl`
- `.omx/research/ddm_uf1_20260805/refresh_registry.jsonl`
- `.omx/research/ddm_uf1_20260805/refresh_summary.json`
- `.omx/research/ddm_uf1_20260805/transport_refreshes.jsonl`
- `.omx/research/ddm_uf1_20260805/COMMIT_BLOCKER.md`

No score claim. No promotion eligibility. Own-vehicle frontier unchanged:
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.
