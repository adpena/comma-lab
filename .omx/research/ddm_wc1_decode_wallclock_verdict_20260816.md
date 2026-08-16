# ddm_wc1 verdict — advisory decode wall clock 1,905 s → 517 s base / 370 s cached; bars MET

Date: 2026-08-16 · Owner: MAIN (#1072 close) · Axis: [M5-CPU scorer-free advisory decode — NEVER a
score] · Admission receipt: `ddm_wc1_advisory_fast_path_admission.v1` (`complete: true`,
`blockers: []`, `identity_pass: true`, `shipping_packet_touched: false`). Workspace:
`/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815`.

STORES CONSULTED: six run receipts (prefix_n4_r2 · prefix_n32 · base_optimized_n600_r3 ·
base_scalar_n600_r1 · micro_fresh_keep75mk87_r2 · micro_cached_keep75mk87_r1) + the finalize
admission receipt + `receipts/cache_parity.json` (sha 0402ed08…) + the mp2 prepare receipt
(manifest 2e170b1f…) + the operator steer #1072 ("Wall clock on decode side needs iteration and
optimization").

## The measured table (n600 advisory decode, M5 CPU, 4 workers)

| row | total s | token decode s | raw sha | identity |
|---|---|---|---|---|
| baseline (pre-wc1) | 1,905 | — | e5539653… | reference |
| base optimized r3 (native SIMD, cache r/w) | **516.8** | 146.5 | e5539653… | 4/4 vs reference |
| base scalar twin r1 (native scalar C) | 657.0 | 143.8 | e5539653… **IDENTICAL** | raw parity |
| micro fresh r2 (mp2 keep75∖keep87, cache off) | 509.7 | 144.8 | c881db66… | token PASS (own parse-back) |
| micro cached r1 (same candidate, cache r/w) | **370.4** | **5.9** | c881db66… **IDENTICAL to fresh** | cache parity |

## Adjudication

1. **Both bars MET.** The ≤700 s bar holds on every optimized row (3.69× vs baseline). The ≤450 s
   stretch bar holds on the cached path (370.4 s, 5.14×).
2. **Raw-parity leg PAID.** The scalar twin reproduces the optimized raw byte-for-byte
   (e5539653…), so the SIMD-optimized native HPAC decoder is byte-exact against the scalar C
   reference — the optimization is a pure speed change.
3. **Cache parity PROVEN, cross-candidate.** The cache was populated by the BASE archive run
   (80d9c8c6…) and was consumed by a DIFFERENT candidate archive (37194782…) because the cache
   key binds SECTION content hashes, which the mp2 differential candidate inherits. The cached
   decode reproduces the fresh decode's raw sha exactly (c881db66…). Token decode collapses
   144.8 s → 5.9 s (24.5×). Cross-candidate cache reuse is byte-safe on this instrument.
4. **The remaining wall is the RENDER, not the token decode.** Post-cache split: neural render
   318 s (86% of the cached total) + checkpoint copy 88 s + frame0/selector I/O 45 s. Any wc2
   successor targets the render (worker count is already auto-derived at 4; per-chunk worker RSS
   peaks ~2.5 GiB under the 4 GiB budget).
5. **Ops lesson (r1 crash, launch 73, rc=2 @5 s):** wc1 `run --source-generation` takes the
   ORIGINAL generation directory and stages the advisory runtime internally. The `prepare`d copy
   REPLACES `runtime/f26_inflate.py` with the advisory F26P runtime, so pointing `run` at it
   trips the base-identity pin (expected 4718834f…, observed 5558f95a…). The gate was CORRECT;
   the path was wrong. Recorded in hot-state; the successful base runs had always ridden the
   argparse default, so the explicit flag path had never been exercised.
6. **Consumption:** the ~370 s cached advisory decode is the new cost basis for every mp2/wd-line
   advisory n600 row on differential candidates — a 5.1× cheaper admission instrument for the
   #1058/#1067 chains. Prices quoted per row: 3.66 GB raw retained per run (payload kept),
   ~6 min/row cached vs ~32 min at the old baseline.

## Receipts

- Admission: emitted by `finalize` (consumer code SHAs pinned: builder 29eebcaf…, inflate driver
  5558f95a…, runtime 249770e4…, mp2 queue eee4691c…).
- `receipts/cache_parity.json` sha 0402ed08… · `receipts/native_assets.json` (optimized .so
  1cf0e61b…, scalar 64efe1e8…).
- All raw payloads + token fields retained under `$WC/runs/*/output/` per ALWAYS-KEEP-THE-PAYLOAD.
