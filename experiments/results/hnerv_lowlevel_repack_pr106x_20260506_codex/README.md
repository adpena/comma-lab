# PR106x HNeRV Low-Level Repack Candidate

This directory records a byte-only HNeRV packed-payload repack candidate. It is
not a score claim and is not dispatch-ready without the normal archive preflight,
lane claim, and exact CUDA auth eval.

- source: `experiments/results/public_pr106_belt_and_suspenders_xrepack_20260504_codex/archive.zip`
- source label: `PR106x`
- source archive bytes: `186231`
- candidate archive bytes: `186080`
- byte delta: `-151`
- changed section: `decoder_packed_brotli`
- raw brotli equivalence: recorded in `result.json`
- candidate archive: `pr106x_hnerv_brotli_repack_candidate.zip`

Command:

```bash
.venv/bin/python tools/build_hnerv_lowlevel_repack_candidate.py \
  --source-archive experiments/results/public_pr106_belt_and_suspenders_xrepack_20260504_codex/archive.zip \
  --source-label PR106x \
  --output-dir experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex \
  --json-out experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/result.json
```

