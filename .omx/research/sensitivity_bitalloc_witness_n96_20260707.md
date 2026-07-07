# #336 Sensitivity bit-alloc APPLY on witness weights — n96 MEASURED row (FEED-07b row 5)

**Status:** RESULTS PENDING — this skeleton is committed only if the run completes; the numbers
section below is filled from `bitalloc_witness_n96.json` before commit. If you are reading
this line in a committed file, the harvest step failed — treat as a blocker, not a result.

Run: `tools/apply_sensitivity_bitalloc_witness.py` on the mod32cap live-run snapshot
(`snapshot_ema_BEST.npz`, ep425 best d_seg 0.0036364; run `levelset_n600_witness_mod32cap_20260706T115554Z`),
probe n16 / eval n96 evenly-spaced pairs, `--mean-bits 6 5`. Authority: [macOS-CPU advisory]
NON-PROMOTABLE — a bounded-subset compression measurement, never a score. Blob-grammar parity vs
the canonical `build_levelset_blob` (#202) was MEASURED byte-identical 2026-07-07 (base int8
72,695 B sha-match; code int8 38,400 B sha-match; brotli-11 streams byte-equal) and is pinned by
`src/tac/tests/test_bitalloc_witness_blob_parity.py`. Pointer 0.19110 UNMOVED (MEANS).
