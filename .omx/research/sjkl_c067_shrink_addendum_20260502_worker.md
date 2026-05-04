# SJ-KL C067 Shrink Addendum - 2026-05-02

Scope: local deterministic SJ-KL payload shrink only. No GPU job was dispatched.
No score claim is made here; exact CUDA auth eval remains required before any
promotion, ranking, or paper claim.

## Code Change

- `src/tac/sjkl_basis.py` now writes compact quantized SJ-KL basis payloads by
  default (`basis_quant_bits=6`) while preserving backward-compatible unpack of
  legacy FP16 basis payloads via `basis_quant_bits=None`.
- `experiments/build_sjkl_c067_archive.py` now builds a runtime-compatible
  minimized RPK1 header and deterministically screens all four-member logical
  orders for the smallest Brotli payload. Source archive SHA custody remains in
  the manifest rather than in the runtime header.

## Local Candidate Built

Source frontier archive:

- path: `experiments/results/c067_breakthrough_candidate_matrix_20260502T1030Z/line_search_source_c067_fixedslice/archive.zip`
- bytes: `276214`
- SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`

Input SJ-KL rows:

- source: `experiments/results/sjkl_c067_trace_selected_repack_20260502T_local/trace_top16/repack/sjkl.bin`
- bytes: `422`
- SHA-256: `9ecc86c074e51239a8129d80af18692002bee0e535deba9f6a73921a610d44ac`

New local candidate:

- archive: `experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/pack/archive.zip`
- archive bytes: `276999`
- archive SHA-256: `7977f3ee5d6d744f5818d358c13424a1f19f6bf8b6604af56d37c13912159922`
- delta vs source archive: `+785` bytes
- `sjkl.bin`: `250` bytes, SHA-256 `13f605bfd9ad950807d410c8371f20fd2b1c3d9c04bb59cc6bf07d474dcc78bb`
- repack manifest: `experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/repack/sjkl_repack_manifest.json`
- archive manifest: `experiments/results/sjkl_c067_shrink_q6_minrpk1_20260502T_worker/pack/sjkl_c067_archive_manifest.json`

The archive manifest records `score_claim=false`, `promotion_eligible=false`,
and `evidence_grade=empirical_archive_candidate_until_exact_cuda`.

## Local Verification

- `.venv/bin/python -m pytest src/tac/tests/test_sjkl_basis.py src/tac/tests/test_build_sjkl_c067_archive.py src/tac/tests/test_plan_sjkl_trace_benefit_allocator.py`
  - result: `35 passed`
- Runtime unpack sanity check on the candidate archive extracted:
  `optimized_poses.bin`, `sjkl.bin`, `masks.mkv`, and `renderer.bin`.

## Evidence Boundary

This is a deterministic local byte/custody result only. The 6-bit basis
quantization changes the decoded SJ-KL basis relative to the prior FP16 basis,
so component behavior must be measured by exact CUDA auth eval before any
score or promotion claim.
