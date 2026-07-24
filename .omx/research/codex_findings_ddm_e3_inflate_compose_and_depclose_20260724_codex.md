# Codex findings — DDM E3 inflate composition and dependency closure — 2026-07-24

`research_only=true` · `score_claim=false` ·
`[macOS-CPU frozen-scorer advisory]` · pointer unchanged ·
MAIN landing review required.

## Verdict

**PASS PA1 receiver survival and Brotli dependency closure on the bounded local
axis. BLOCK promotion because no contest-CPU/CUDA replay was authorized.**

The governed exporter/receiver now derives the PA1 scorer-only affine from the
already-counted decoded E2 frames, applies the camera residual to frame 0 only,
and preserves both base and corrected 16-pair checkpoints. It imports no scorer
and carries zero amplitude payload bytes. The exact locked upstream run passed:

| Quantity | Result |
|---|---:|
| archive bytes | 439,303 |
| archive SHA-256 | `dd8fc5fed6ff11e532765dfe6104f02b3b97171b824123312a3ab469c1be6cbe` |
| raw bytes | 3,662,409,600 |
| raw SHA-256 | `4c553508b0bf92ccdc137e215799ae30a346b58e0617e5156441a7929302b4f1` |
| upstream d_pose (printed) | 147.49104309 |
| upstream d_seg (printed) | 0.02861482 |
| upstream score (printed) | 41.56 |
| harness wallclock | 842.944145 s |

All 38 source and all 38 corrected camera batch SHA-256 values equal the
immutable PA1 measurement checkpoints. That exact byte binding transfers the
PA1 full-precision row: `d_pose=147.49104204339514`,
`d_seg=0.02861480712890625`, 3,375,540 errors / 117,964,800 sites. E3 and E2
frame-1 streams are byte-identical at SHA
`bfe8f686e5da8578a86029287b0a78430431cf612457ab84abc302cd8ac2bca1`.
The exact E3-byte advisory score is `41.55855704359477`.

## Dependency and rate closure

The shipped receiver dependency list is now `["torch"]`; `lzma` is stdlib and
Brotli is absent. Real-stream measurements selected raw LZMA1 with a fixed
1 MiB dictionary (`lc=3, lp=0, pb=2`):

| Coder | Chart raw coded bytes | Semantic raw coded bytes | Total |
|---|---:|---:|---:|
| Brotli Q11 control | 18,412 | 315,033 | 333,445 |
| raw LZMA1, 1 MiB | 17,768 | 411,205 | 428,973 |
| XZ6 | 17,828 | 421,376 | 439,204 |
| zlib9 | 34,234 | 720,361 | 754,595 |

The dependency-closed archive is 95,837 bytes larger than E2. This increase is
COUNTED and costs `0.06381392449036953` score units. The PA1 transform itself
is FREE at exactly zero bytes. D2 and D5 remain NULL.

## Adversarial execution finding

The first locked run correctly failed closed. PA1 had been measured with a
fixed four-thread Torch CPU contract, while the initial packaged receiver
forced one thread. Torch bilinear interpolation resolved a small set of
float32 rounding ties differently, producing raw SHA
`884c3f78189c1df1bb7aa6ddd370273936ce508c4d3ec4d6e1008e21e087c8c5`
instead of the compiler-bound SHA. No score was cited. The failed packet,
partial raw, logs, and roughly 11 GB of resumable checkpoints were preserved
under:

`/Volumes/VertigoDataTier/pact/evidence/ddm_e3_inflate_compose_and_depclose_20260723/superseded_thread1_mismatch`

Its certification manifest is 1,668 bytes, SHA
`9429e0b7cb2fef5cf78612ade39519ef810aef114c8ddbde53a669100f6f4d89`.
The final compiler and receiver both seal four Torch threads, and the clean
checkpoint namespace then passed.

## Directive disposition

| Directive | Disposition |
|---|---|
| compose PA1 in the governed E2 surface | consumed as explicit E3 schema on the shared exporter/receiver |
| local n600 frozen CPU scorer only | PASS |
| preserve d_seg and reproduce PA1 d_pose | PASS by 38/38 batch byte identity and locked report |
| no amplitude payload bytes | PASS, FREE 0 bytes |
| close Brotli dependency with measured stdlib alternative | PASS, raw LZMA1 |
| full locked upstream `evaluate.sh` | PASS, exit 0 |
| no remote/paid/training/upstream edit | PASS |
| pointer unchanged | PASS |
| MAIN landing review | required and pending |

## Triality and custody

- DSL: `DDME3RuntimeExporterConfigV1`,
  `DDME3UpstreamHarnessConfigV1`, and `ddm_e3_runtime_archive.v1`.
- DAG: E2 counted packet → base decode checkpoints → official YUV6 moments →
  scorer-only target affine → frame-0 camera residual → corrected checkpoints
  → exact raw → locked upstream scorers.
- Equations: `ddm_e3_pa1_receiver_survival_v1` extends the PA1 law with exact
  38-batch, frame-1, and zero-payload predicates.

Machine receipts:

- `.omx/research/ddm_e3_inflate_compose_and_depclose_20260723/ddm_e3_runtime_export_receipt.json`
- `.omx/research/ddm_e3_inflate_compose_and_depclose_20260723/ddm_e3_upstream_harness_receipt.json`
- `.omx/research/ddm_e3_inflate_compose_and_depclose_20260723/ddm_e3_receiver_survival_receipt.json`

## Independent review record

Round 1 disposition: **PASS local evidence and implementation; BLOCK
promotion.** The review re-derived the transform boundary, inspected the
packaged runtime rather than trusting the exporter, and verified there is no
Brotli/scorer/GT import, no amplitude member, no changed frame-1 byte, no
unpriced archive byte, and no unscoped authority claim.

Three consecutive clean passes:

1. Ruff, Python compile, 24 focused tests, five canonical JSON checks,
   runtime-copy identity, archive/submission identity, and diff whitespace
   check: PASS.
2. Independent exporter process replayed the exact archive under
   overwrite-refusal; receiver resume rehashed/adopted all base/corrected
   stages with `render_seconds=0` and final SHA unchanged: PASS.
3. Re-derived all 38 PA1 source/corrected batch hashes, frame-1 SHA,
   1,830,500,832 changed channel values, 610,401,600 changed RGB pixels,
   archive homes, FREE/COUNTED/NULL partition, and score arithmetic from
   bytes; reran lint and 24 tests: PASS.

The review tracker records three reviewed marks for every tracked entity in
the receiver, exporter, runtime tests, equations module, and locked harness.
The only bug found during execution—the one-thread/four-thread interpolation
contract mismatch—was fixed, regression-scoped, and its failed evidence was
cold-stored rather than erased.

MAIN landing review is required.
