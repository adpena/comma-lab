# Compact y-hat rate-distortion ladder — Task #548

Date: 2026-07-19 UTC
Lane: `lane_yhat_rd_ladder_20260719`
Status: `research_only=true`
Axis: `[macOS-CPU advisory n24] NON-PROMOTABLE`
Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**
Authority: local MEASURE only; no launch, paid dispatch, score, promotion, or pointer authority

## Verdict

**MEASURED:** a four-rung, two-distortion y-hat ladder now exists over 24 disjoint
real pairs from `gt_n600.npz`. Every one of the `56,623,104` rung-block solves was
`FEASIBLE_EXACT`; repair cells, proven-infeasible blocks, budget/heuristic blocks,
and realized-numerator error cells were all zero. The direct-plane formulations
remain far too large. The decisive byte result is that the ep725 generator archive
represents its byte-closed y-hat witness in **83,838 actual bytes**, whereas a
complete direct Brotli-Q11 description of the same n24 shared planes costs
**42,051,900 actual bytes** across the two independent n12 containers.

`verdict_scope`: the direct complete-plane payload formulations measured here are
not viable as contest payloads; the compact generator/state description family is
**OPEN**. This is a frame1-y-hat diagnostic with source frame0, not a complete
two-plane packet and not a contest score.

## Machine-readable result

Primary table: `.omx/research/yhat_rd_ladder_20260719_codex.json`
CSV: `.omx/research/yhat_rd_ladder_20260719_codex.csv`

| rung | counted-rate basis | actual bytes measured | bytes/pair | n600 bytes | n600 rate term | mean d_seg | mean d_pose |
|---|---|---:|---:|---:|---:|---:|---:|
| A `source_exact_i32` | two Brotli-Q11 complete-plane n12 containers | 37,746,958 / n24 | 1,572,789.92 | 943,673,950 **DERIVED** | 628.353748 | 0.000000211928 | 0.000000001144 |
| B `witness_byteclosed_i32` | actual full-n600 ep725 archive | 83,838 / n600 | 139.73 | 83,838 **MEASURED** | 0.055824 | 0.003455480 | 63.031066895 |
| C `witness_q6_u8` | two Brotli-Q11 complete-plane n12 containers | 6,187,537 / n24 | 257,814.04 | 154,688,425 **DERIVED** | 103.000673 | 0.003548092 | 63.606825829 |
| D `witness_q4_u8` | two Brotli-Q11 complete-plane n12 containers | 2,631,552 / n24 | 109,648.00 | 65,788,800 **DERIVED** | 43.806061 | 0.005516900 | 64.023759524 |

The B distortions are measured after serializing the exact shared-plane integer
numerators and choosing another exact bounded-uint8 preimage. Its secondary direct
plane custody is 42,051,900 Brotli-Q11 bytes / 45,188,817 zstd-19 bytes over n24.
Thus the `83,838 B` row is not a hidden plane sidecar: it is the actual full-n600
generator archive that deterministically renders the plane. The selected n24 raw
decode is measurement scratch on the SSD and is excluded from the byte count.

Actual zstd-19 complete-plane container totals for A/B/C/D are respectively
`41,138,237`, `45,188,817`, `6,096,559`, and `2,491,067` bytes. Every Brotli and
zstd stream was decompressed and its complete descriptor parsed back before the
row was admitted.

## Plane error, realization, and interaction custody

| rung | mean absolute shared-plane error vs source | RMSE vs source | exact blocks | repair/infeasible/error cells | mean lattice solve s/pair |
|---|---:|---:|---:|---:|---:|
| A source exact | 0 | 0 | 14,155,776 | 0 | 11.8343 |
| B witness exact | 18.420144 | 25.044688 | 14,155,776 | 0 | 12.4034 |
| C witness q6 | 18.397932 | 25.081797 | 14,155,776 | 0 | 12.3835 |
| D witness q4 | 19.094081 | 25.580230 | 14,155,776 | 0 | 12.3491 |

Frame0 policy is source `gt_f0`, external and zero-byte for this diagnostic. That
choice isolates frame1 y-hat but is **not contest-complete**. It also exposes the
large Pose interaction debt (`d_pose about 63`) of pairing source frame0 with the
c2 witness frame1; no claim of a good two-frame witness is made.

The shipped byte-closed camera frames themselves score `d_seg=0.003455691874`,
`d_pose=63.030915737`. Re-solving their exact y-hat changes the n24 means by
`-2.1191469e-7 d_seg` and `+0.0001511574 d_pose`. This is the honest alternate-
preimage interaction gap. The target y-hat has zero realized numerator error;
camera bytes are not claimed identical.

Rung A's nonzero `2.12e-7` mean Seg difference despite exact rational shared-plane
equality is the already-known frozen-fp32 tie/noise surface, not a lattice repair.
Its near-zero Pose result (`1.14e-9`) reproduces the earlier n6 mechanism at n24.

## Real custody and byte-close proof

- Source cache: real `gt_n600.npz`, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Checkpoint: sacred read-only `levelset_witness_ema_BEST.npz`, explicit EMA epoch
  725, `384x512`, SHA-256
  `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef`.
- Actual full-n600 archive: `83,838 B`, SHA-256
  `149fefd097c1fa85c4afb6cb2d8ab20311035d7ba8063f1e72137b843a9b89f3`.
- Exact selected code-row slice: pairs
  `{0,50,...,550} union {10,60,...,560}`; no requantization.
- Shipped receiver raw: `146,496,384 B`, SHA-256
  `3ffe6e11d672b3278c8d78ac2c562468d4fdffe97dc71777068b464fad9e57a8`,
  preserved on `/Volumes/VertigoDataTier/pact/`; two-pair/four-frame numpy-fp32
  oracle gate was bit-exact (`max_abs_uint8_diff=0`).
- Sacred run metadata hash was
  `0bdc3e39c5eac970625f91e6803b1bb33330e412514b78ce35dab2c4c351842c`
  before and after both prepare and chunk measurements.
- Full DistortionNet used frozen `modules.py`, PoseNet, and SegNet hashes from each
  chunk receipt. Batch geometry is one `[1,2,874,1164,3]` uint8 pair per CPU
  forward; seed `20260719`; deterministic algorithms enabled.

The measurement followed `docs/operating_manual_craft_handoff.md`: inspect the
sacred run, preserve exact bytes and hashes, use the SSD waterfall for bulk,
checkpoint each pair atomically, and keep negative verdicts formulation-scoped.

## Resumability and receipts

The run is two disjoint n12 chunks, each with an atomic state file and twelve
write-once per-pair stages. Resume re-derives all scientific fields from the frozen
cache, byte-closed raw, and scorer, then compares them to the preserved stage;
stored scientific output is never trusted as input. Runtime metadata is preserved
from the first successful stage and excluded from deterministic scientific equality.

| artifact | SHA-256 |
|---|---|
| witness prepare receipt | `350f38e53bba79f5b282b6726ca47a958f27f5759b612f9ffa984fd523486625` |
| chunk A receipt | `b2d35d864bda323be75805b1d4ae9f0299cddb620cbc067b8723f65001f906cf` |
| chunk B receipt | `349d977bac463976a658b0696753f11324a6cfdb00791c9aafcaa86804c0ea65` |
| composed JSON table | `74e04312e90330d1a4c03e49db5090b134c4cf5894ffd78165dd576ab5c796e3` |
| composed CSV table | `6c01c83b4e6da8d6c6b87d478f95ecd35156f4293b01a348dd1e403b4c77f256` |

## Labels, triality, and consumers

- **MEASURED:** n24 codec bytes, both distortions, plane errors, exact/repair
  counts, runtimes, full archive bytes, and byte-close hashes.
- **DERIVED:** direct-plane n600 byte/rate extrapolations from two measured disjoint
  n12 containers; no full-n600 direct-plane compression was rerun.
- **SPECULATIVE:** none in the table.
- Equation leg: consumes the ratified exact factor-2 bounded-uint8 resize-preimage
  law implemented by `src/tac/optimization/uint8_lattice_feasibility.py`.
- DSL/receiver leg: the ep725 row uses the shipped byte-close decoder; the new
  complete-plane descriptors are a measurement grammar, not yet a production V10
  DSL section or inflate consumer.
- DAG leg: this receipt closes §5 item 11 as an n24 advisory measurement and feeds
  `#536` shared-fidelity KKT/bit allocation plus `#543` production receiver work.
  No autopilot dispatch is authorized. MAIN must wire any adopted law after review.

## Remaining blockers

1. Build a compact frame0 description and a joint two-plane packet.
2. Build the production V10 descriptor/receiver; direct complete-plane bytes are
   diagnostic rate custody, not a shippable decoder format.
3. Measure any chosen compact ladder at full n600 and through exact contest CPU and
   CUDA on the same archive bytes.
4. MAIN must independently review this branch before landing or consuming it.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5 §8 and v8 SPEC; Task #548 delegated
authority; `docs/operating_manual_craft_handoff.md`;
`.omx/research/v10_capstone_state_review_20260719_codex.md`;
`.omx/research/v10_lattice_rate_verdict_and_composition_20260719.md`;
`src/tac/optimization/uint8_lattice_feasibility.py`;
`tools/measure_uint8_lattice_feasibility.py`;
`tools/levelset_byte_close_and_eval.py`; real n600 GT cache; sacred ep725 run;
delegation inbox and broadcast ledger.

**Pointer delta: 0. MAIN landing review is required.**
