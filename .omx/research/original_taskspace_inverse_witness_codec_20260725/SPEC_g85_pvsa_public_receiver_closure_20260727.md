# G85 — PVSA public receiver closure and exact-row falsification

Date: 2026-07-27  
Lane: `lane_g85_pvsa_public_receiver_closure_20260727`  
Authority: full-n600 repository-runtime decode plus private-runtime
`upstream/evaluate.py` on macOS CPU  
Status: fail-closed research result; not a public archive, contest score,
candidate, or pointer row

## Outcome first

G85 validates the compact G82 container and its bounded sparse receiver, but
falsifies G82 as a frontier candidate.

The exact G82 archive decodes all 1,200 frames twice to the same
3,662,409,600-byte raw object. The sparse receiver lowers wall time from
887.95 s to 184.76 s and peak RSS from 2,613 MiB to 843 MiB. It therefore
closes the receiver-correctness and bounded-scheduling questions in the
repository runtime.

The full 600-sample frozen scorer then returns:

```text
d_seg       0.02747120
d_pose      163.06130981
archive     129392 bytes
size_ratio  0.00344627
score       43.21 printed; 43.21412746 from reported components
```

This is not a near-frontier row. The base semantic P is a complete syntactic
600-pair stream, but not a source-wide evaluator-cell preimage. G82 adds only
two G74 atoms, both on pair 0 and both with selector `BOTH`; it cannot repair
population pose or segmentation. The exact row should have preceded public
receiver polishing.

The competitive pointer remains the official-leaderboard `0.172`.

## Triality

DSL:

```text
decode_G82(PVSA1(P, A_pair0)) -> raw[1200,874,1164,3] uint8
score(raw, frozen_targets) -> (d_seg, d_pose, size_ratio, S)
```

DAG:

```text
exact G82 archive/member custody
  -> strict PVSA1 parse
  -> render semantic P once
  -> render/stitch addressed pair 0 through canonical G74
  -> bounded batch-16 chronological raw write
  -> exact size/hash parse-back
  -> independent decode A/B byte equality
  -> private-runtime upstream/evaluate.py, n600
  -> component-level falsification
```

Equations:

```text
size_ratio = 129392 / 37545489
           = 0.00344627

rate_score = 25 * size_ratio
           = 0.08615682

seg_score  = 100 * 0.02747120
           = 2.74712000

pose_score = sqrt(10 * 163.06130981)
           = 40.38085064

S_reported_components = seg_score + pose_score + rate_score
                      = 43.21412746
```

`Compression Rate` in `evaluate.py` is the dimensionless size ratio. It is not
the rate contribution to the objective; that contribution is
`25 * size_ratio`.

At the same archive bytes, even an ancestor-like `d_pose=0.000034` would require
`d_seg < 0.00067404` to beat `0.172`. With `d_seg=0.0005`, pose would need
`d_pose < 0.00012847`. The measured G82 components are outside both cells by
orders of magnitude.

## Exact input custody

```text
G82 archive.zip
  bytes   129392
  sha256  b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd

PVSA1 member 0.bin
  bytes   133363
  sha256  d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31

semantic P
  bytes   133941
  sha256  759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df

counted G74 operand
  bytes   52
  sha256  5616799adc0d2ab942f37a20b070f7a0fa48119771e8f1b56c1f45e2605306ca
  atoms   2
  pairs   {0}
  selector BOTH
```

## Exact double-decode proof

Both repository-runtime decodes write the required ABI:

```text
shape   (1200,874,1164,3)
dtype   uint8
bytes   3662409600
sha256  436ce2b6965c859556a217df9b1cc17784d988f2af900c35201d3e3c7f372782
cmp     identical
```

Decode A is the canonical per-batch receiver:

```text
receiver elapsed  851.9686696669 s
safe_run wall     887.95 s
peak RSS          2613 MiB
```

Decode B renders unaddressed pairs through semantic P once and sends only
addressed pair 0 through the canonical G74 path:

```text
receiver elapsed  150.7509735420 s
safe_run wall     184.76 s
peak RSS          843 MiB
addressed pairs   1
```

The sparse schedule is 4.806x faster by safe-run wall time, 5.651x faster by
receiver timing, and uses 3.100x less peak RSS. Pair `(0,1)` was also proven
equal between canonical and sparse receivers before the population run:
12,208,032 bytes, SHA-256
`1e91105f88d348c2a08d9ea5f92056406cca0ca9e8bd8d1e87803eec3f6dbc42`.

## Exact evaluator evidence

Command:

```text
.venv/bin/python tools/safe_run.py \
  --rss-mb 24576 --projected-gib 4.0 --timeout 1800 \
  --label g85_g82_full_n600_batch16_upstream_cpu_private_runtime_advisory -- \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 \
  .venv/bin/python upstream/evaluate.py \
  --batch-size 16 --num-threads 2 --prefetch-queue-depth 2 \
  --submission-dir <SSD>/eval_private_r1 \
  --uncompressed-dir upstream/videos \
  --seed 1234 --device cpu \
  --report <SSD>/eval_private_r1/report.txt \
  --video-names-file upstream/public_test_video_names.txt
```

Evidence:

```text
samples            600
batches            38
safe_run exit      0
safe_run wall      576.09 s
safe_run peak RSS  10267 MiB
report SHA-256     75e85ae2a75748423f4c74e592982a9bf61d07e3b0ef7fee294934a71a6bde7c
```

The raw safe-run stdout was observed live but was not separately tee'd to a
durable file; no fake stdout-log hash is claimed. The evaluator report above is
the durable authoritative output of this run.

This is tagged `[macOS-CPU advisory/private-runtime]`: the raw video was
pre-inflated by the repository environment, and the evaluator ran from
`.venv`. It is not a public-entrypoint replay and not contest-CPU/CUDA
authority.

## Public runtime blocker stack

The three-argument staging entrypoint was run with the frozen upstream-default
system Python and failed before output:

```text
ModuleNotFoundError: No module named 'pydantic'
```

Dependency audit separates import baggage from executed semantics:

- `pydantic` and `scipy` are tree-shake/import-graph baggage;
- Brotli is materially executed in 12 retained sections; and
- `cv2.fillPoly(..., lineType=LINE_8, shift=0)` materially rasterizes all G1
  movable polygons.

The encoder-side portable recode replaces all 12 Brotli sections with registered
zlib frames and verifies identical decoded section bytes:

```text
portable semantic P  161915 bytes  sha256 f44dd0e1...
portable member      161337 bytes  sha256 47ffe158...
portable archive     144633 bytes  sha256 63654e8e...
archive delta        +15241 bytes
rate-score delta     +0.0101483563
```

The portable archive is a new noncandidate lineage. It is not output-equality
closed because the exact generic rasterizer is still missing.

The naive Pillow candidate was measured on every retained G1 polygon across all
600 frames:

```text
polygons                   2197
vertices                   19150
one/two/three-plus vertices 42 / 101 / 2054
differing pixels           28648
differing frames           600
maximum differing/frame    134
canonical mask SHA-256     42ecdc1d...
Pillow mask SHA-256        826d3a9d...
```

That verdict kills only this exact Pillow integer
point/line/polygon formulation. It does not kill a bit-exact generic
OpenCV-LINE_8 raster implementation.

## Structural lesson and next gate

The missing type was not another container layer. It was the distinction
between:

1. population-shaped syntax (`P` addresses 600 pairs),
2. population-shaped actuator coverage (video-specific selected preimages
   exist where needed), and
3. evaluator-cell closure (the decoded object actually satisfies full-n600
   Seg/Pose budgets).

Future receiver-closure work should be gated by a scorer-native population
certificate on the exact decoded object. A complete stream is not a useful
codec base merely because it parses and covers all pair indices.

## Verification

```text
G85 receiver tests          7 passed
portable recode tests       5 passed
G1 n600 raster tests        2 passed
ruff                        passed
py_compile                  passed
staging inflate.sh syntax   passed
```

## Pointer-delta honesty

The effective frontier is unchanged at official-leaderboard `0.172`.
G85 validates the compact wire and bounded receiver, but its exact scorer row
is `43.21`. G82 is invalidated as a candidate, and neither the original nor the
portable archive is promotion eligible.
