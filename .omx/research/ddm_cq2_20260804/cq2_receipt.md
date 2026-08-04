# CQ2 comma10k-only tiny-student sizing receipt - 2026-08-04

Status: **BLOCKED_DATASET_ABSENT_OR_INCOMPLETE**.

Axis: `[macOS-CPU advisory / public-data custody preflight / scorer-free]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

Own-vehicle baseline from hot state: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`
for qo1 `sub_auto_pairbit`. Contest pointer remains borrowed/unmoved.

## Charter

CQ2 was chartered to train a comma10k-public-data-only tiny segmentation student
at roughly 25 KB / 75 KB / 150 KB counted weight sizes, choose the smallest
student by comma10k-val Road/Lane fidelity against the public teacher, and only
then re-run CQ1's frozen-candidate n32 Road/Lane band-overlap measurement.

The charter explicitly requires stopping with a typed blocker if the comma10k
dataset is absent. This receipt is that stop.

## Recall Inputs

CQ1 predecessor result: GOOD-OVERLAP for the full public comma10k-segnet teacher
on the matched n32 subset:

| metric | value |
|---|---:|
| SE3 r1 captured target flips, cx1 chart | `8,670` |
| those flips also inside public-model r1 band | `8,644` |
| overlap fraction | `0.9970011534` |
| Road IoU vs cx1 | `0.999742` |
| Lane IoU vs cx1 | `0.994872` |
| teacher argmax npy sha256 | `249ad030e60025ad46464fa78ff7957753b056371761140e2891978107a0d44a` |

SE3 priced stream targets:

| stream row | stream bytes | captured flips | break-even survival |
|---|---:|---:|---:|
| `road_lane_band_r1_edit_bits_side_implied` | `81,365` | `161,660` | `0.395339` |
| `road_lane_band_r1_edit_plus_direction_bits` | `100,904` | `161,660` | `0.490276` |
| ED1 section baseline | `169,149` | `191,005` | `0.696430` |

Rate exchange: `W = 1.2731082153320312 B/flip`.

## Custody Preflight

Repo HEAD at preflight: `2c5fda118298429b7a262b1c13048818ee614634`.

Teacher path verified:

| file | sha256 |
|---|---|
| `/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet/model.safetensors` | `8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8` |
| `/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet/albumentations_config_eval.json` | `d260853fe0a993e23613ff38039fdce59264f5fe31f729c1fa65f8c3e5fde913` |
| `/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet/config.json` | `2b8f16dbad9bd85386609386a9cb5dedc6e0c518253a9af484e0a128d9463c88` |

Teacher custody manifest: `/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet/CUSTODY_MANIFEST.json`.

Dataset path checked:
`/Volumes/VertigoDataTier/pact/public_datasets/comma10k`.

Dataset findings:

| check | result |
|---|---|
| remote URL | `https://github.com/commaai/comma10k` |
| `git rev-parse HEAD` | `fatal: ambiguous argument 'HEAD': unknown revision or path not in the working tree.` |
| `git symbolic-ref -q HEAD` | `fatal: No such ref: HEAD` |
| maxdepth-2 file count | `4` |
| maxdepth-4 file count | `20` |
| non-git data files found in maxdepth-4 listing | `0` |
| lock marker | `.git/shallow.lock` present |
| pack marker | `.git/objects/pack/tmp_pack_xkG6pP` present |

The path is an incomplete or in-progress git clone, not a usable comma10k
dataset. No train/val split can be made and no legal public-data-only
distillation can launch from this state.

## Derived Thresholds Only

These rows are arithmetic for the next run, not measured student results.
Formula:
`break_even_survival = (student_counted_bytes + stream_bytes) / (W * 161660)`.

| hypothetical student bytes | side-implied total bytes | side-implied break-even survival | explicit-direction total bytes | explicit-direction break-even survival |
|---:|---:|---:|---:|---:|
| `25,000` | `106,365` | `0.516810` | `125,904` | `0.611747` |
| `75,000` | `156,365` | `0.759752` | `175,904` | `0.854688` |
| `150,000` | `231,365` | `1.124164` | `250,904` | `1.219101` |

Interpretation: a measured 25 KB student would require realizer survival above
about `0.517` on the side-implied stream and `0.612` on the explicit-direction
stream. A measured 75 KB student would still fit under the ED1 bytes only for
the side-implied row, but would require about `0.760` survival. A measured
150 KB student cannot clear the SE3 break-even on these streams without more
captured flips or a better coder.

Live realizer context from the charter/hot state remains unchanged: se2's
paint ceiling `0.263-0.407` does not cover the 25 KB side-implied threshold;
sq2's solved-field eta is the live candidate if its convergence survives.
Composition verdict remains MAIN's after CQ2 and sq2 both return measured rows.

## Measurements Not Performed

- No comma10k train/val split was made.
- No public-data student was trained, selected, quantized, or compressed.
- No comma10k-val Road/Lane IoU was measured.
- No CQ1 overlap re-measurement with a tiny student was run.
- No contest SegNet/PoseNet forward was run.
- No `upstream/evaluate.py` run was performed.
- No `archive.zip` was built.

## Verdict

**BLOCKED**, verdict_scope
`INSTANCE: /Volumes/VertigoDataTier/pact/public_datasets/comma10k is present only as incomplete git metadata at 2026-08-04T22:27:28Z`.

This is not a negative verdict on the tiny-student formulation. It is a custody
preflight blocker required by the charter.

## Follow-On Disposition

QUEUED-WITH-FIRE-ORDER:

1. When the comma10k clone finishes, re-run the dataset preflight before any
   training: record git HEAD, remote URL, file counts, top-level tree shape, and
   whether image/label assets are present.
2. If the dataset is complete, create the fixed train/val split from the
   conventional repo val list if present; otherwise use a seeded split and
   record the seed.
3. Train the public-data-only student size curve and choose the candidate by
   comma10k-val metrics before any contest-side CQ1 overlap read.
4. Then run the frozen-candidate n32 overlap measurement exactly once for the
   selected candidate.

Do not download data in this arm and do not silently symlink around the
incomplete SSD clone.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`;
contest pointer borrowed/unmoved.
