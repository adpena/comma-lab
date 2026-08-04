# CQ1 comma10k chart-overlap receipt - 2026-08-04

Status: **GOOD-OVERLAP measured; tiny-student route remains live**.

Axis: `[macOS-CPU advisory / public-model chart-overlap scorer-free]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

Own-vehicle baseline from hot state: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`
for qo1 `sub_auto_pairbit`. Contest pointer remains borrowed/unmoved.

## Charter

CQ1 tested RF1's fallback route after qo1 receiver-field closure failed: use a
custodied public-data-only comma10k segmentation model on `<=32` qo1 generated
frame_1 images, without contest scorer forwards, to measure whether its Road/Lane
chart overlaps SE3's `cx1` stand-in Road/Lane band strongly enough to justify a
later tiny comma10k-only student.

Selection reused SE2's stratified random non-prefix n32 pair list, seed
`20260804`:

`31, 43, 62, 82, 94, 118, 147, 165, 167, 182, 185, 200, 237, 241, 247, 259, 272, 286, 288, 292, 296, 306, 327, 382, 390, 419, 473, 488, 525, 555, 560, 581`.

## Inputs And Controls

| item | value |
|---|---:|
| public model | `/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet/model.safetensors` |
| public model sha256 | `8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8` |
| eval preprocessing sha256 | `d260853fe0a993e23613ff38039fdce59264f5fe31f729c1fa65f8c3e5fde913` |
| qo1 inflated raw | `/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/inflated/0.raw` |
| qo1 inflated raw bytes | `3,662,409,600` |
| qo1 inflated raw sha256 | `3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31` |
| raw layout | `1200 x 874 x 1164 x 3 uint8`; frame_1 index = `pair*2+1` |
| cx1 argmax sha256 | `5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be` |
| GT argmax sha256 | `b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d` |
| class order | Road `0`, Lane `1`, Undrivable `2`, Movable `3`, MyCar `4` |

Preprocessing followed the custodied eval config: resize RGB frame to `384x512`
with linear interpolation, then tensor conversion. `albumentations` is not
installed in the Pact venv, so the script used an explicit OpenCV/PyTorch
equivalent and preserved the config's absence of a normalization transform.

## Receipts

| artifact | sha256 |
|---|---|
| `.omx/research/ddm_cq1_20260804/cq1_summary.json` | `a9de19ac35bf3bfd7f133b7887f72d722f929115c3fdfe305623b294eac5bc2e` |
| `.omx/research/ddm_cq1_20260804/cq1_pair_rows.jsonl` | `e270f3d939608b48ba61e0bc38ed97998f6fafad0aaab50edaa3329d51dc33b0` |
| `/Volumes/VertigoDataTier/pact/ddm_cq1_20260804/comma10k_public_model_argmax_pairs_n32.npy` | `249ad030e60025ad46464fa78ff7957753b056371761140e2891978107a0d44a` |
| `experiments/ddm_cq1_comma10k_chart_overlap.py` | `ab2655382bb93ed329c59109243d9e1f1ddf8566cc557b242ce38f6769812f86` |

Bulk output is on the SSD tier. No persisted evidence path uses `/tmp`.

## Measured Result

Denominators:

| denominator | value |
|---|---:|
| selected pairs | `32` |
| subset scorer cells | `6,291,456` |
| subset Road/Lane target flips | `12,407` |

Primary SE3 r1 overlap:

| metric | value |
|---|---:|
| SE3 r1 captured target flips, cx1 chart | `8,670` |
| those flips also inside comma10k public-model r1 band | `8,644` |
| decisive overlap fraction | `0.9970011534` |
| GOOD threshold | `0.80` |

Radius sweep:

| radius | SE3 captured | micro captured | overlap of SE3 captured | overlap fraction | band IoU vs SE3 |
|---:|---:|---:|---:|---:|---:|
| `1` | `8,670` | `8,655` | `8,644` | `0.997001` | `0.995768` |
| `2` | `9,152` | `9,125` | `9,121` | `0.996613` | `0.996800` |
| `3` | `9,429` | `9,395` | `9,393` | `0.996182` | `0.996938` |

Per-class IoU, public-model chart against `cx1`:

| class | IoU |
|---|---:|
| Road | `0.999742` |
| Lane | `0.994872` |
| Undrivable | `0.999898` |
| Movable | `0.999784` |
| MyCar | `0.999362` |

Per-class IoU, public-model chart against GT:

| class | IoU |
|---|---:|
| Road | `0.984535` |
| Lane | `0.689915` |
| Undrivable | `0.991395` |
| Movable | `0.922109` |
| MyCar | `0.994703` |

## Verdict

**GOOD-OVERLAP**, verdict_scope
`FORMULATION: public comma10k-segnet receiver-chart proxy on qo1 n32 stratified frame_1 subset; not n600, not contest authority`.

The tiny-student route is live. The next unit is **CQ2 comma10k-only distillation
sizing**, queued with this fire order:

1. Train or select only a tiny comma10k-public-data student from a frozen public-data recipe.
2. Count the student bytes economically against the SE3 `81,365 B` side-implied and `100,904 B`
   explicit-direction stream prices; do not call public-code bytes free if weights ship.
3. Re-run CQ1 overlap with the tiny student before any archive work.
4. If CQ2 remains GOOD-OVERLAP and bytes clear the break-even envelope, then run receiver-field
   closure against a receiver-consumed chart and only then queue scorer validation.

## Boundaries

- No contest SegNet/PoseNet forward was run.
- No `upstream/evaluate.py` run was performed.
- No `archive.zip` was built.
- No public model training, finetuning, distillation, adaptation, or checkpoint selection was performed.
- The public 38,502,740 B model is not a shipping proposal. If any public-model or student weights
  ship, their bytes are counted.
- This is n32 stratified-random chart overlap, not n600 authority and not a frontier move.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`;
contest pointer borrowed/unmoved.
