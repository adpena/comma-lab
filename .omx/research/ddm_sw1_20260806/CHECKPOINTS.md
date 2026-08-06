# CHECKPOINTS - ddm_sw1

## Completed

- Charter and common contract read.
- Governing Pact files read before implementation.
- Parent tq1c custody verified by archive sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.
- Null basis constructed from SVD with `N.shape=[12,6]`, `max_abs_A_times_N=1.1796119636642288e-16`, `max_abs_NtN_minus_I=1.5585082678759737e-16`.
- Runner compiled with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile experiments/ddm_sw1_null_basis_phase_solve.py`.
- Bounded n=4 measurement completed with pairs `[0, 20, 32, 48]`.
- Bulk rows and summary written under `/Volumes/VertigoDataTier/pact/ddm_sw1_20260806`.
- Receipt copy written under `.omx/research/ddm_sw1_20260806/sw1_null_basis_summary.json`.
- Seam/metric ledger written with 16 rows and `score_claim=false` on every row.
- Seam/metric ledger JSONL validated after writing: 16 rows; classification counts match `RECEIPT.md`.
- Review tracker pass 1 marked 30 entities in `experiments/ddm_sw1_null_basis_phase_solve.py` as reviewed.
- Review tracker pass 2 marked 30 entities in `experiments/ddm_sw1_null_basis_phase_solve.py` as reviewed.

## Checksums

| path | sha256 |
|---|---|
| `/Volumes/VertigoDataTier/pact/ddm_sw1_20260806/sw1_null_basis_rows.jsonl` | `5994edf22d5af37a5cbfe17712d4ae9ad610aacb86453120c65a9ddd00d8026a` |
| `/Volumes/VertigoDataTier/pact/ddm_sw1_20260806/sw1_null_basis_summary.json` | `231c11e958731450c8821d70d494ff26e7d4959276dd085ac80bd4a8330f5ac1` |
| `.omx/research/ddm_sw1_20260806/sw1_null_basis_summary.json` | `231c11e958731450c8821d70d494ff26e7d4959276dd085ac80bd4a8330f5ac1` |

## Finalization

- Commit only the SW1 files through `tools/subagent_commit_serializer.py` with post-edit `--expected-content-sha256` declarations.
- If serializer is blocked by unrelated dirty work, report the exact blocker and leave SW1 artifacts unmodified.
