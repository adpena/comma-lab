# Task #541 constructive solver and free-predictor floor

Date: 2026-07-19 UTC  
Lane: `constructive_solver_541_20260719`  
Status: `research_only=true`  
Axis: `[Darwin-arm64 CPU advisory real-cache n48] NON-PROMOTABLE`  
Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**  
Authority: isolated local build and $0 advisory measurement only; no launch,
paid dispatch, contest score, promotion, submission, or pointer authority.
MAIN landing review is required.

## Verdict

**MEASURED:** `spatial-smooth-121.v1` is the smallest of the three exact-uint8
predictor formulations on 48 real pairs. Four independently parseable n12
production sections total **31,872,098 B** including the charged frame-0
bootstrap. Its descriptor-plus-residual content is **16,115,021 B** through the
production inner Brotli-Q11 representation; the same canonical conditional
bytes compressed directly are **16,037,540 B Brotli-Q11** and **16,873,868 B
zstd-19**. No bootstrap is called free.

**MEASURED:** a single n48 production rung-E archive for that predictor is
**31,873,460 B**, parses back, inflates to `292,992,768 B`, and realizes both
scorer planes with all `56,623,104` integer numerator values exact. The external
native-float32 CPU hard oracle nevertheless measures mean
`d_seg=0.00012344784206814235` and
`d_pose=0.00005041551356414436`. This is the expected receiver-arithmetic
admissibility boundary: exact integer A-numerators do not authorize an exact
frozen-fp32 argmax claim.

`verdict_scope`: this closes only the zero-band, exact uint8-plane description
instance over predictor copy/affine6/smooth on pairs `0..47`. It does **not**
kill positive-band constructive solving, compact learned/generator
descriptions, other predictor families, or the complete v10 carrier family.
The rate is not contest-viable at this formulation; the family remains open.

## Predictor floor: global counted bytes

All byte columns are actual codec outputs that were decompressed and parsed
back. The n48 rows are sums of four independent canonical n12 streams, not a
full-n600 run. `Conditional raw` is descriptor plus exact signed-int16 RGB
residual; `production conditional` includes its internal production framing;
`complete production` additionally charges the full frame-0 bootstrap.

| predictor | conditional raw B | direct Brotli-Q11 B | direct zstd-19 B | production conditional Brotli-Q11 B | complete production B | residual nonzero values | residual absolute sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| `previous-plane-copy.v1` | 56,623,104 | 16,390,330 | 17,356,549 | 16,414,411 | 32,171,488 | 24,263,715 | 111,364,934 |
| `affine6-q12.v1` | 56,624,256 | 16,360,410 | 17,251,044 | 16,344,819 | 32,101,896 | 24,311,706 | 108,370,411 |
| `spatial-smooth-121.v1` | 56,623,104 | **16,037,540** | **16,873,868** | **16,115,021** | **31,872,098** | 24,431,486 | 102,556,724 |

The affine descriptor is exactly six little-endian signed Q12 coordinate
deltas, or 24 video-derived bytes per pair before compression. Copy and smooth
have no fitted descriptor. Smooth has more nonzero residual values but a lower
absolute sum and better actual coder output; nonzero count alone is therefore
not a sufficient rate proxy.

**DERIVED, not measured at n600:** smooth costs `664,002.0417 B/pair` complete,
`335,729.6042 B/pair` production-conditional, and `351,538.9167 B/pair` by
zstd-19. Linear multiplication gives `398,401,225 B`, `201,437,762.5 B`, and
`210,923,350 B` respectively at n600. Chunk/container effects make these
extrapolations non-authoritative.

## Best-predictor attribution

These are independently compressed attribution streams containing a packed
membership mask plus selected int16 RGB residual values. They are actual
coder bytes, not pro-rated global bytes, and must **not** be summed to recover
the global stream. The composed machine table retains the class and margin rows
for all three predictors; the compact tables below show the selected predictor.

### By cached class

| class id | selected pixels | attribution raw B | Brotli-Q11 B | zstd-19 B |
|---:|---:|---:|---:|---:|
| 0 | 2,147,560 | 14,069,128 | 3,141,680 | 3,352,259 |
| 1 | 60,390 | 1,546,108 | 172,924 | 203,044 |
| 2 | 4,667,815 | 29,190,658 | 8,239,803 | 8,699,185 |
| 3 | 142,696 | 2,039,944 | 295,969 | 340,867 |
| 4 | 2,418,723 | 15,696,106 | 4,429,885 | 4,632,164 |

### By cached winner-rival margin

| margin band | selected pixels | attribution raw B | Brotli-Q11 B | zstd-19 B |
|---|---:|---:|---:|---:|
| `[0,.1)` | 25,128 | 1,334,536 | 89,807 | 104,421 |
| `[.1,.25)` | 37,290 | 1,407,508 | 125,248 | 143,095 |
| `[.25,.5)` | 62,072 | 1,556,200 | 187,882 | 215,863 |
| `[.5,1)` | 116,772 | 1,884,400 | 316,092 | 358,909 |
| `[1,2)` | 192,018 | 2,335,876 | 463,752 | 523,673 |
| `[2,inf)` | 9,003,904 | 55,207,192 | 15,267,765 | 16,444,179 |

The observed mass is dominated by the high-margin interior. That observation
does not estimate the bytes of a positive-band solve; it only identifies where
the current exact residual values live under these standalone attribution
grammars.

## Rate-ladder comparison without false equivalence

The prior `yhat_rd_ladder_20260719_codex.md` direct source-exact i32 row measured
`1,572,789.92 B/pair`, while its compact lossy generator is an actual
`83,838 B/n600` archive. The new smooth row is a third surface: exact bounded
uint8 scorer planes in a complete two-plane packet.

- **DERIVED:** smooth production-conditional is `4.6847x` smaller per pair than
  the direct source-exact i32 row; complete smooth is `2.3687x` smaller. This is
  not fidelity-equivalent because the i32 row preserves the stricter rational
  numerator surface.
- **DERIVED:** the smooth complete n600 linear extrapolation is `4,752.0x`
  larger than the `83,838 B` generator archive. This is not distortion-equivalent:
  the generator is compact and lossy, whereas this row carries exact selected
  uint8 planes.
- **MEASURED:** one n48 archive costs 1,362 B more than the sum of four n12
  production sections because it has one distinct n48 packet/container shape.

The comparison therefore supports one narrow negative: transmitting exact
uint8 plane residuals under this predictor/grammar remains too expensive. It
does not support replacing exactness with the cited generator without a joint
Seg/Pose admission measurement.

## Rung-E custody and hard-oracle result

| field | value |
|---|---|
| archive | `31,873,460 B`; SHA-256 `7ebbe894efed6050e841e838c8eb05512e918fe986de940a8e6b5cd298bdfcd8` |
| predictor payload | SHA-256 `399232c6fa1c34f517f01c24d648ae23ff34704c5cdeb328e2ca5009ba5f2b51` |
| inflated raw | `292,992,768 B`; SHA-256 `d7c78c412c3d731b1d4961b734737edd4b4aa307c2bdfa57d203573b3da24064` |
| integer proof | both planes exact; `56,623,104` equal numerator values |
| native-fp32 Seg | 1,165 mismatched cells over 9,437,184 pair-cells; all 48 pairs nonzero; maximum 37 in one pair |
| mean distortions | `d_seg=0.00012344784206814235`; `d_pose=0.00005041551356414436` |
| composed predictor receipt | SHA-256 `31c9b04dab35a8058b88b6343befc7206eccab2cfe87f6dc1818b5e3a313b3ba` |
| cache | `5,078,017,610 B`; SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |

The production inflate import boundary contains no scorer, Torch, source
cache, cached labels, or margins. The hard oracle runs encode-side after
inflate. The archive and `292,992,768 B` inflated raw were certified
rebuildable and removed after the durable receipt was written: storage
preflight passed with `651,599,818,752 B` free against `854,420,992 B`
required; cleanup completed and temporary paths are redacted. The receipt
preserves archive/raw hashes plus cache, tool, codec, and composed-receipt
hashes. No operator-facing evidence cites temporary storage.

`verdict_scope` for the nonzero Seg result: the 1,165 mismatches are a
Darwin-arm64 native-fp32 hard-oracle observation for these 48 pairs and this
receiver arithmetic. They extend the instance evidence behind
`f32_receiver_arithmetic_exactness_admissibility_v1`; they are not a contest
CPU/CUDA equivalence claim. MAIN should append the anchor to the canonical law
only after reviewing this branch.

## Constructive solver and production receiver

`src/tac/optimization/v10_constructive_solver.py` implements the deterministic
NumPy-fp32 reference required by the flattened KKT derivation:

`delta(lambda) = clip(anchor + lambda*q/w, lower, upper)`

It projects onto the box/Seg half-space, then optionally applies the rank-6
pose ellipsoid through a six-dimensional Gram eigensolve and deterministic
Dykstra alternation. Zero-band mode fixes `lower=upper=0`, providing the exact
end-to-end control today. The hard-oracle contract freezes request/source
hashes, exact geometry, receiver arithmetic, non-boolean pair ids, and
post-callback frame bytes; a local q/J proposal cannot certify itself.

The optional sidecar loader consumes exactly `vjp_custody_pair.v1`. A real
pair-0 sidecar smoke used
`/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/chunk_000_011/pair_0000.vjp.npz`,
SHA-256 `f2daa9139afbe41f4f6fd7960573db15fe6da0af6ad12f28f26df6e0440d3af6`;
its `seg_q` is `[384,512,3]` float32 and `pose_j_y` is
`[6,2,384,512,3]` float32. This confirms schema compatibility only. No
positive band or VJP efficacy number is claimed.

`src/tac/codec/v10_predictor_residual.py` provides the closed, hash-checked
predictor packet. The additive production registry entries are
`predictor-residual-u8.v1` and `description-frame0.v1`. Parse-back refuses
unknown modes, malformed geometry/length/hash/dtype, duplicate/non-integer ids,
trailing content, or excessive aggregate decompression. Existing raw/Brotli
receivers remain compatible. Inflate realizes two distinct planes through the
landed factor-2 lattice instead of repeating frame1.

Chunk measurement is deterministic and resume-safe: four atomic state files
plus 48 write-once per-pair stage receipts. Resume re-derives scientific fields
from frozen inputs and refuses config/input drift.

## Durable receipts and verification

- Composed predictor table:
  `.omx/research/constructive_solver_541_predictor_floor_n48_20260719.json`,
  SHA-256 `31c9b04dab35a8058b88b6343befc7206eccab2cfe87f6dc1818b5e3a313b3ba`.
- Rung-E receipt:
  `.omx/research/constructive_solver_541_rung_e_n48_20260719.json`, SHA-256
  `d966e066bfd24deb0f7ad1fda865337ed2f108c03c93244c24dd592ac69682a9`.
- Chunk receipt SHA-256 values for `00..11`, `12..23`, `24..35`, and `36..47`:
  `390b757db01c2fd0215d660067f406934cd0d9466eba6e64e7ed9f91eac8527a`,
  `4ef9ca4e50acfa11a8c15b7843dac2097ba7d62eb55103c1d627e6554aac6382`,
  `36994916ea3cefbc2bd15a3fceada9709d7fdb36fab2c98dda316df7fb881ba5`,
  and `0c1768b674d5d3530692c922f28d458fc49303d82d0ba291992e0c09d59d4a05`.
- Focused new-code suite: 77 passed. Broader coder/projection/lattice/receiver
  dependency group: 209 passed and one existing MLX parity test deselected
  because this headless sandbox has no Metal device. Ruff and format checks
  passed on all eight changed Python files. Receipt invariants, JSON parsing,
  `git diff --check`, forbidden-file/sacred-tree non-mutation, and scratch
  cleanup all passed.

## Labels, triality, and review

- **MEASURED:** four n12 codec receipts; all global/class/margin coder bytes;
  predictor residual statistics; n48 rung archive/inflate bytes and hashes;
  exact integer proof; local hard-oracle distortions.
- **DERIVED:** n48 sums/best ordering; per-pair rates; n600 linear byte
  extrapolations; cross-row ratios.
- **INFERRED:** no quantitative result in the tables.
- DSL leg: closed predictor packet plus additive production codec/frame-0
  registries and exact two-plane inflate.
- Equations leg: box/half-space KKT projection plus optional rank-6 pose
  correction, implementing the flattened Lagrangian surface without claiming
  proposal authority.
- DAG leg: lane `constructive_solver_541_20260719`, chunk/state/stage receipts,
  composed table, and rung-E receipt. Adoption remains held for MAIN review;
  no autopilot/launcher/pointer consumer is authorized from this branch.

Exactly five bounded adversarial self-review rounds were used. They found and
fixed: an initially marker-only hard-oracle seam; float64 and zero-band pose
bypass; invalid Lipschitz/geometry/id acceptance; compressed-payload resource
bounds; affine portability; callback and returned-frame mutability; and the
actual rung-E `complete`/`completed` integration typo. Round five was clean
after the seam fix. Further independent review belongs to MAIN.

## Stores consulted

1. Delegated authority file, verified SHA-256
   `af8372f91f21c21cb22c4cdf4403908b3cdd2c24c002f3f7675957f6e030d267`.
2. `CLAUDE.md`, `AGENTS.md`, canonical lane/task/pointer surfaces, and the live
   delegation inboxes.
3. `.omx/research/v10_flattened_lagrangian_kkt_derivation_20260719.md` and
   registered laws `pose_plane_proximity_corollary_v1` and
   `f32_receiver_arithmetic_exactness_admissibility_v1`.
4. `docs/operating_manual_craft_handoff.md`, followed for sacred-input custody,
   exact hashes, SSD-first preflight, resumable stages, scoped negatives, and
   pointer honesty.
5. Landed production receiver, factor-2 lattice, range-A projection, and
   content-priced coder modules; the two live-arm forbidden files were neither
   edited nor invoked.
6. Read-only real `gt_n600.npz`, the actual pair-0 VJP custody sidecar, four
   chunk receipts, the composed n48 receipt, the rung-E receipt, and
   `.omx/research/yhat_rd_ladder_20260719_codex.md`.

## Pointer delta and MAIN handoff

Pointer delta is exactly **zero**. No contest score exists for these bytes.
MAIN must independently review codec accounting and decode freedom, the hard-
oracle/f32 interpretation, q/J proposal-only boundaries, all five review fixes,
the 1,362 B container difference, and the non-equivalent rate-ladder comparison
before merging or registering the new empirical law anchor.
