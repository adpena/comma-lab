# Seg-secant rate-distortion curve: the first measured `d_seg>0` axis

Date: 2026-07-19 UTC  
Lane: `lane_seg_secant_rd_curve_20260719`  
Status: `research_only=true`  
Axis: `[Darwin-arm64 CPU advisory real-cache n24] NON-PROMOTABLE`  
Pointer: `0.1910828242 [contest-CPU Linux x86_64]` **UNMOVED**  
Authority: isolated local build and measurement only; no launch, paid dispatch,
contest score, promotion, submission, or pointer authority. MAIN landing review
is required.

## Verdict

**MEASURED:** the zero-Seg face is open. Seven deliberately lossy operating
points on the same 24 real pairs all have `d_seg>0`. They span two distinct
families: four VJP-native-margin abandonment thresholds and three residual
precision truncations. Every row measures full native-float32 CPU-Torch SegNet
and PoseNet, five per-class Seg distortions, actual Brotli-Q11 and zstd-19
payload bytes after exact parse-back, and explicit repair/reject counts.

**DERIVED:** after correctly converting measured mean bytes/pair to the global
600-pair byte term, every emitted adjacent saving secant exceeds the KKT
break-even

`100 * 37,545,489 / 25 = 150,181,956 bytes / unit d_seg`

or about `150.182` global bytes per `1e-6 d_seg`. The equivalent threshold for
a mean bytes/pair curve is `0.25030326 bytes/pair per 1e-6 d_seg`.

**DERIVED:** the composed #536 solver now returns
`MEASURED_SECANT_KKT_CANDIDATE`, not the former
`INCONCLUSIVE_FLAT_OR_NOISY`. Its closest measured Brotli secants use the
`margin_m0p3` and `precision_drop1` endpoints; marginal gap
`4.142214713626108e-12` score/global-byte. This is a conditional
n600-equivalent range-payload allocation, not an archive optimum.

**MEASURED Pose qualification:** Pose is inactive/slack in `162/168` delivered
pair/point observations, not all 168. The six violations are confined to
precision depths 2 and 3. Both endpoints selected by #536 (`margin_m0p3` and
`precision_drop1`) have `0/48` Pose violations, so Pose is inactive on the
selected measured segment only. The unrestricted all-row inactivity hypothesis
is refuted at this n24 instance scope.

`verdict_scope`: these are selected real-cache n24, range-coordinate payload
measurements under a concrete camera-preimage policy. They are not receiver-
closed `archive.zip` bytes, not contest-Linux CPU/CUDA evidence, and not a score
claim. They do not settle other predictor, residual-transform, or repair
families.

## Measured curve

Bytes are mean two-frame counted residual bytes per pair. Each codec payload
was decompressed and compared byte-for-byte and value-for-value with its signed
little-endian int32 scorer-numerator residual. `rep/rej` is direct-transform
repair/reject count; neither family filters back to `d_seg=0`.

| point | family parameter | Brotli-Q11 B/pair | zstd-19 B/pair | `d_seg` | `d_pose` | Seg flips | rep/rej |
|---|---:|---:|---:|---:|---:|---:|---:|
| `margin_m0p01` | `m*=0.01` | 2,222,625.12 | 2,502,564.58 | 1.8649631e-5 | 1.7919215e-8 | 88 | 0/0 |
| `margin_m0p03` | `m*=0.03` | 2,222,173.38 | 2,501,966.79 | 2.0980835e-5 | 8.8257117e-8 | 99 | 0/0 |
| `margin_m0p1` | `m*=0.1` | 2,220,441.00 | 2,499,911.42 | 2.5643243e-5 | 3.0930624e-7 | 121 | 0/0 |
| `margin_m0p3` | `m*=0.3` | 2,214,597.46 | 2,493,139.21 | 3.9206611e-5 | 8.4393414e-7 | 185 | 0/0 |
| `precision_drop1` | drop 1 low bit | 1,770,993.33 | 1,991,528.79 | 1.6276042e-4 | 3.8682648e-5 | 768 | 0/0 |
| `precision_drop2` | drop 2 low bits | 1,313,066.92 | 1,480,862.88 | 1.8246969e-4 | 7.5275306e-5 | 861 | 0/0 |
| `precision_drop3` | drop 3 low bits | 1,145,117.33 | 1,290,867.29 | 1.7399258e-4 | 1.4154703e-4 | 821 | 0/0 |

The source reference is **MEASURED** at 2,222,946.21 Brotli bytes/pair,
2,502,807.96 zstd bytes/pair, `d_seg=0`, `d_pose=0`. Precision-drop 3 has fewer
bytes and lower mean `d_seg` than drop 2, but worse mean Pose; it therefore
dominates drop 2 only on the two-term Seg/rate projection, not on the full
three-term objective.

## Adjacent Seg secants and break-even sign

For a move from lower distortion/more bytes to higher distortion/fewer bytes,

`Delta S = 100*Delta d_seg - 25*bytes_saved/37,545,489`.

Therefore paying the higher `d_seg` improves the two-term score exactly when
`bytes_saved/Delta d_seg` is **above** 150,181,956 global bytes/unit. Equivalently,
if those bytes are interpreted as the extra cost to prevent the distortion,
prevention is worthwhile below the threshold and not worthwhile above it. This
re-derived sign is load-bearing; reversing it reverses every admit decision.

The table reports n600-equivalent global bytes saved per `1e-6 d_seg`, derived
as measured mean bytes/pair times 600. All emitted adjacent secants favor
accepting the higher distortion on this conditional range-payload surface.

| adjacent move | Brotli B / `1e-6` | zstd B / `1e-6` | versus 150.182 B threshold |
|---|---:|---:|---|
| source -> `margin_m0p01` | 10,329.963 | 7,829.914 | above; accept higher `d_seg` |
| `margin_m0p01` -> `margin_m0p03` | 116,270.397 | 153,858.271 | above; accept higher `d_seg` |
| `margin_m0p03` -> `margin_m0p1` | 222,937.386 | 264,503.892 | above; accept higher `d_seg` |
| `margin_m0p1` -> `margin_m0p3` | 258,499.584 | 299,580.826 | above; accept higher `d_seg` |
| source -> `precision_drop1` | 1,666,079.078 | 1,884,779.520 | above; accept higher `d_seg` |
| `precision_drop1` -> `precision_drop3` | 33,433,058.339 | 37,427,951.871 | above; accept higher `d_seg` |

No saving secant is emitted from precision-drop 3 to drop 2 because drop 2 has
both more bytes and higher mean `d_seg`; its distinct Pose behavior remains in
the full measured table.

## Per-class Seg result

Conditional `d_seg` uses the real GT pixels of each comma10k class
`[Road0,Lane1,Undrivable2,Movable3,MyCar4]` as denominator.

| point | Road0 | Lane1 | Undrivable2 | Movable3 | MyCar4 |
|---|---:|---:|---:|---:|---:|
| `margin_m0p01` | 3.3478e-5 | 7.10011e-4 | 4.288e-6 | 1.66267e-4 | 7.440e-6 |
| `margin_m0p03` | 3.8128e-5 | 8.11441e-4 | 3.860e-6 | 1.80123e-4 | 9.920e-6 |
| `margin_m0p1` | 4.4637e-5 | 8.45251e-4 | 5.575e-6 | 1.66267e-4 | 1.9014e-5 |
| `margin_m0p3` | 6.6956e-5 | 1.318592e-3 | 9.006e-6 | 2.07834e-4 | 3.1414e-5 |
| `precision_drop1` | 2.82704e-4 | 4.091017e-3 | 4.2027e-5 | 9.56036e-4 | 1.45497e-4 |
| `precision_drop2` | 3.50590e-4 | 4.361497e-3 | 6.0467e-5 | 1.011459e-3 | 1.16563e-4 |
| `precision_drop3` | 3.57100e-4 | 5.240559e-3 | 4.7601e-5 | 9.28325e-4 | 8.5975e-5 |

Lane is the most distortion-sensitive class at every measured point. This is a
measurement on this n24 corpus, not a universal class-order theorem.

## Pose activity and violations

The hard inactivity test is literal `d_pose < 2.5e-4` per pair/point.

- all four margin rows: `96/96` inactive;
- precision drop-1: `24/24` inactive;
- precision drop-2: `22/24` inactive, pair 12 `4.7202266e-4`, pair 21
  `3.0283153e-4`;
- precision drop-3: `20/24` inactive, pair 12 `5.8410417e-4`, pair 15
  `4.2018269e-4`, pair 22 `4.8707774e-4`, pair 23 `5.4366025e-4`.

Mean `d_pose` stays below the crossover at every aggregate point, but averaging
does not erase the six hard pair-level violations. The scoped correct verdict
is therefore: Pose is inactive for margin abandonment and precision drop-1,
including the selected #536 segment; it is not inactive over the whole
precision family.

## Construction and custody

The implementation composes settled instruments instead of rebuilding them:

- `generated_fill_predictor` and exact range numerators from
  `joint_seg_pose_rate` / `uint8_lattice_feasibility`;
- the native margin and winner arrays from all 24 immutable VJP sidecars;
- the existing native-float32 B=1 hard oracle from
  `measure_joint_seg_pose_rate.py`;
- the existing `solve_measured_waterfill` #536 implementation;
- the real `gt_n600.npz` cache, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.

Margin abandonment replaces complete low-margin disjoint resize blocks with
their generated-predictor blocks, so every result is a concrete reachable
uint8 preimage. Precision truncation drops low bit-planes from the signed
camera-preimage residual toward zero, then recomputes the exact range
numerators. The counted payload is only the signed int32 numerator residual;
camera/null-space bytes are not claimed free in a receiver archive. This is a
range-coordinate convention until a decoder grammar reproduces the chosen
preimage and parse-back closes on shipped bytes.

The runner is deterministic, seeded, capped at 12 pairs/invocation, and
checkpoints every pair/point with atomic state plus immutable stage JSON. The
two chunks cover pair IDs `[0..10,24]` and `[12..17,19..23,25]`. Pair 11 and 18
remain outside this corpus under their previously scoped VJP/hard-oracle
refusals; they are not silently replaced inside a measurement chunk.

## Durable receipts

- Chunk A receipt:
  `/Volumes/VertigoDataTier/pact/evidence/seg_secant_20260719/chunk_000_010_024_v1/receipt.json`,
  SHA-256 `48bd168cbc856c03e2c6f7f358e51be732a91609610e2c6c51e600adcbe0f927`.
- Chunk B receipt:
  `/Volumes/VertigoDataTier/pact/evidence/seg_secant_20260719/chunk_012_017_019_023_025_v1/receipt.json`,
  SHA-256 `e27e11e9ac6180634d2eaaeb89e6c6506f11da26fe8325312efd777a9f880a3f`.
- Machine-readable n24 curve:
  `.omx/research/seg_secant_rd_curve_n24_20260719.json`, SHA-256
  `7dba18dc37ee7ca6b7ae07c2aa4fc1eb72cd9dce024c97b6df7ae121c402cacc`.

Both raw receipts bind the exact executed tool hash
`36eec6668e5a044e7153b4f57ae7be44c8cda8a698022fe3349e2c4c37c7cb39`
and module hash
`90c11cdfc6ae3b46084ee0d59fe6fe4a603aaa76da749764d169444320492189`,
which re-derive exactly from commit `ac604dd71e`.

The first composition used mean bytes/pair directly in the global score term.
It is **INVALID for derived secant/KKT normalization**, though its raw measured
rows are identical. It is preserved, not cited as authority, at
`/Volumes/VertigoDataTier/pact/evidence/seg_secant_20260719/superseded_composed_per_pair_normalization_v1.json`,
SHA-256 `a4d367fa841a294b30d09d383ba912dbfcc78b0da1278da722684d47384a6986`.

## Adversarial self-review (maximum five rounds)

1. **Round 1 — findings fixed before n24:** real pair-0 smoke found that the
   custody loader incorrectly required a full-manifest pair list and hashed a
   cast int64 winner instead of the custodied int8 bytes. Both bugs were fixed;
   the smoke then measured all seven positive-Seg points with exact codec
   parse-back.
2. **Round 2 — critical normalization finding fixed:** review caught that
   mean bytes/pair had been inserted directly into the global archive-byte
   score term. The first composition was invalidated and preserved; code now
   multiplies by the declared 600-pair population before break-even and #536.
3. **Round 3 — clean custody pass:** 192 unique immutable stages, 768 codec
   parse-backs (two frames times two codecs), source-receipt hashes, stage
   hashes/configs, and 2,943 aggregate versus per-class mismatches all
   revalidated exactly.
4. **Round 4 — clean authority pass:** raw executed source hashes re-derived
   from commit `ac604dd71e`; both chunks report the sacred result tree unchanged;
   no source pair, VJP sidecar, upstream file, live run, or pointer was changed.
5. **Round 5 — clean code/regression pass:** 168 focused and composed tests
   pass; Ruff, `py_compile`, CLI help, and `git diff --check` pass. Review
   tracker marks all three Python files reviewed.

## Triality and system integration

- **Equations:** the adjacent finite-difference secants, global/per-pair byte
  normalization, exact break-even, and #536 KKT candidate are executable in
  `tac.optimization.seg_secant_rd_curve`.
- **DAG/evidence:** two resumable chunk receipts, 192 immutable stages, and the
  composed machine JSON are the durable evidence chain.
- **DSL/control:** no trainer or launch lever was added. This is a
  `research_only` measurement surface, so inventing a trainer flag would be a
  no-stray violation.
- **Sensitivity/allocator:** custodied VJP native margins order family 1; the
  existing #536 waterfill consumes the measured curve with corrected n600 byte
  normalization.
- **Bit allocator/autopilot:** the non-null candidate is emitted for later
  planning, but no dispatch is authorized because receiver/archive and contest
  axes remain absent.
- **Continual learning:** the machine curve and regression tests make the
  zero-face unlock and the 600x normalization reusable; the six Pose violations
  prevent future all-row inactivity claims.

This work follows `docs/operating_manual_craft_handoff.md`: each claim is
labeled by how it was obtained, the negative is scoped, the arithmetic is
re-derived from primary bytes, and the pointer delta is stated literally.

## Remaining blockers

- Bytes are a range-coordinate numerator payload, not a receiver-closed archive
  section. Header/grammar, preimage reconstruction, archive parse-back, runtime,
  and exact `upstream/evaluate.py` custody are absent.
- The n600 byte values used by secants/#536 are a **DERIVED conditional sum** of
  measured mean bytes/pair, not a measured n600 archive.
- Native macOS CPU advisory is neither contest-Linux x86_64 CPU nor contest-CUDA.
- Precision depths 2–3 violate the Pose crossover on six pair/point instances;
  any allocator using them must add a Pose guard/repair and remeasure its bytes.
- The non-null #536 result is an adjacent measured candidate, not a continuous
  optimum or promotion authorization.

## MAIN landing review required

MAIN must review the isolated branch before merge. Review emphasis:
(a) VJP-native margin ordering and complete-block abandonment semantics;
(b) signed truncation-toward-zero and exact recomputed numerator custody;
(c) actual Brotli/zstd parse-back and immutable resume stages; (d) the 600-pair
normalization and break-even sign; (e) per-class mismatch aggregation; (f) the
six Pose violations versus the narrower zero-violation selected segment; and
(g) strict separation of range payload, receiver/archive bytes, and contest
score authority.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; delegated authority prompt SHA-256
`f0e93f49f1c8bd00f14b478e995ada774269dd83a82dd20e4d3a0e465a4262a8`;
`PROGRAM.md`; `docs/operating_manual_craft_handoff.md`;
`v10_flattened_lagrangian_kkt_derivation_20260719.md`;
`vjp_custody_positive_bands_20260719_codex.md`;
`constructive_solver_541_20260719_codex.md`; v7.5/v8 operating SPECs; canonical
frontier, lane, task, and progress state; real `gt_n600.npz`; all 24 VJP
sidecars/manifests; both Seg-secant chunk receipts/stages; per-arm and broadcast
inboxes through the final checkpoint.

**Pointer delta:** none. `0.1910828242 [contest-CPU Linux x86_64]` remains
unchanged.
