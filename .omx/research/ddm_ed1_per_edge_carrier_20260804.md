# DDM ED1 - Road/Lane Per-Edge Separatrix Carrier

## Survival Verdict

Realized survival vs charter falsifier `0.3956`: **NOT MEASURED**. ED1 did not
run SegNet/PoseNet because `.omx/research/scorer_batch_20260804.md` gives the
single full-n600 scorer slot to sg4/sb1. The exact row is queued in that scorer
batch.

Own byte-closed break-even, using the ED1 archive actually built here:
`0.6964303814` survival, assuming no pose movement and no collateral. This is
not the charter sg3 cheap-addressing break-even. The actual ED1 receiver stores
a counted degree-4 centerline stream plus pair-bitpacked innovations, and costs
169,149 section bytes / 169,351 archive bytes over `sub_final`.

## Artifact

Base: `/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip`

Candidate:
`/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/archive.zip`

| field | value |
|---|---:|
| base archive bytes | 358,084 |
| base archive sha256 | `ad5dd0e4fbe5b13ab53a5995a6d77cc558c25f40b63f894ea50ad336bd50fb66` |
| candidate archive bytes | 527,435 |
| candidate archive sha256 | `a18c1a8c1fe4cab5fe675f661f3433b4b0013c2b4f51e764119d819b2fd86b89` |
| candidate payload sha256 | `5c8254837b440cffad75d61356699900f80c836e99235714cb70e3e90dd7ee22` |
| ED1 section bytes | 169,149 |
| ED1 section sha256 | `e84e2a5860b8b6a12a164f2f0df8397f7bf6062b6af4464e724e1ceae4d576ad` |
| byte ledger | `/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_byte_ledger.json` |
| receiver smoke | `/Volumes/VertigoDataTier/pact/ddm_ed1_20260804/sub_final_per_edge_centerline/ed1_receiver_smoke.json` |

`zipinfo` closes the candidate as a one-member stored archive: 527,435 bytes
total, member `0.bin` 527,327 bytes. `unzip -tqq` passed.

## Carrier Ledger

Selection mode: n600 full population, scorer-free cache-derived target set.
Inputs:

- GT argmax:
  `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy`
  sha256 `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d`
- Current argmax:
  `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy`
  sha256 `5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be`

| component | bytes | codec |
|---|---:|---|
| centerline raw | 574,738 | raw |
| centerline coded | 52,636 | brotli11 |
| edit bitplanes raw | 335,356 | raw |
| edit bitplanes coded | 116,416 | lzma9 |

| count | value |
|---|---:|
| Road/Lane cached target cells | 235,148 |
| centerline-band captured target cells | 191,005 |
| cache capture fraction | 0.8122756732 |
| band pixels | 1,341,421 |
| nonzero pairs | 600 / 600 |
| max captured targets in one pair | 804 |

The receiver applies ED1 after rendering frame 1 and before warping frame 0.
It paints selected Road/Lane scorer-grid cells back through the same private
bilinear supports used by the F0PR proof. The counted archive carries the
centerline chart and innovation bits; the runtime does not ship scorer weights,
GT argmax tables, or margin tables.

Receiver smoke on pair 10 parsed 320 ED1 corrections, changed 1,280 camera
pixels, and changed frame1 sha256 from
`c6f47bb35d1f20d91ddd769fbed626a5d1f48c00ad06d18c3ddb3f02d12b5a94` to
`7bb87ee51baf0088cf6a381c432fef8e450ad2fc09174eae1b85b03eddddce93`.

## Projection, Not A Score

Baseline live best for this own-vehicle lane remains
`S = 0.7541459 @ 358,084 B [macOS-CPU advisory]`.

| projection field | value |
|---|---:|
| rate delta vs baseline | +0.1127638796 score units |
| 100% survival seg gain, no collateral | -0.1619169447 score units |
| predicted S at charter sg3 survival `0.3956`, no collateral | 0.8028554362 |
| predicted S at 100% survival, no collateral | 0.7049928349 |
| actual-byte break-even survival, no collateral | 0.6964303814 |

The charter's R3a GO-signal was `81,365 B / 161,547 flips / 0.3956`
break-even, with cg3 survival `0.555`. ED1 does not re-label that as measured
for this archive. This archive's priced receiver is heavier, captures a
different cache-derived target set, and must be judged by the queued exact row.

## Boundaries

- No SegNet/PoseNet run was performed.
- No realized survival or per-class flip delta was measured.
- No frontier pointer moved.
- No upstream files were edited.
- No `/tmp` evidence path is cited.
- `sub_auto_pairbit` was not used because qo1 is queued, not landed.
- This is byte-closed and receiver-consumed, but scorer-unvalidated.

## Queued Scorer Spec

The scorer-batch section `ED1 - Road/Lane per-edge separatrix carrier, queued
n600 verdict` now owns the exact n600 command and owed verdict fields:
exact d_seg, d_pose, bytes, recomputed S, realized Road/Lane survival against
both `0.3956` and `0.6964303814`, Road->Lane/Lane->Road flip deltas, and
collateral outside the Road/Lane target set.

## Tests

Focused receiver/coder tests:

```bash
.venv/bin/python -m pytest experiments/tests/test_ddm_ed1_per_edge_carrier.py
```

Result: 4 passed.

Syntax check:

```bash
.venv/bin/python -m py_compile experiments/ddm_ed1_per_edge_carrier.py experiments/tests/test_ddm_ed1_per_edge_carrier.py
```

Result: passed.
