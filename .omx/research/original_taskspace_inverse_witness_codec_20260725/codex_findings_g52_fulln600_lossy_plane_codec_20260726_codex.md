# G52 full-n600 lossy selected-plane codec findings

Date: 2026-07-26  
Lane: `lane_g52_fulln600_lossy_selected_plane_codec_20260726`  
Axis: `[Darwin-arm64 CPU advisory] NON-PROMOTABLE`  
Authority: `research_only=true`, `candidate_lineage_allowed=false`

## Forest-level result

The rate representation is promising; the tested color representation is
catastrophically wrong.

A scorer-asymmetric `Y1` base plus conditional `Y0|Y1` layer reduced exact
stored bundle bytes by more than half relative to direct pair interleaving.
After fixing encoder determinism, the 46 kb/s layered row occupied **150,585
bytes**, below the canonical 154,523-byte sub-0.15 planning coordinate.
However, the full n600 frozen batch-16 scorer measured:

| exact quantity | value |
|---|---:|
| bundle bytes | 150,585 |
| bundle SHA-256 | `2f5fc3cf28ee15a803a71d8125c38a771cc9eef5348ff43af85782393a6eba7d` |
| `d_seg` | 0.019747933281792536 |
| `d_pose` | 89.4822183466414 |
| `100*d_seg` | 1.9747933281792536 |
| `sqrt(10*d_pose)` | 29.91357858007654 |
| distortion term | 31.88837190825579 |
| exact bundle rate term | 0.10026837045590217 |
| advisory total | **31.988640278711692** |
| margin to dynamic 0.172 target | -31.81664027871169 |

This is a decisive negative for **this historical-C1-oracle,
SVT-AV1/yuv420p, equal-rate two-layer formulation only**. It is not a
negative for temporal codecs, the task-layer split, AV1 as a family, or
selected-preimage coding.

The dominant structural error is explicit: `yuv420p` subsamples semantic RGB
colors whose channel differences encode evaluator classes. Plane RGB MSE
already exposed a nearly flat floor: deterministic high-rate layered joint MSE
41.90 versus 41.65 at the slightly lower 46 kb/s point. More bits did not
restore the destroyed chroma geometry. The scorer then amplified that into
`d_pose=89.48`; Lane alone had `d_seg=0.624526`.

## Exact rate and determinism rows

The first SVT runs used FFmpeg `-threads 1` but did not bind SVT's internal
logical processors. A full n600 replay changed encoded **and decoded** hashes.
Those rows are reproducibility-falsified and cannot be retained as candidate
results.

The actual deterministic control is `svtav1-params ...:lp=1`. Two independent
full-n600 48 kb/s runs then matched all 15 bitstreams, all decoded hashes, and
both final bundles byte-for-byte:

| arm | exact bytes | exact bundle SHA-256 |
|---|---:|---|
| `DIRECT_INTERLEAVED_RGB` | 365,195 | `a6fb45727edc8c0f7b6475d230790dedda92276b911e2ecb60676fe8fb5b96f7` |
| `TASK_LAYERED` | 157,868 | `2da49f18d3366bf9eed80a1d034a4270322306bffa7e5bedc946973180c5d2d1` |

The secant-informed 46 kb/s row reached 150,585 bytes. It was then decoded from
the exact stored bitstreams, reconstructed into coupled `uint8` scorer planes,
realized and reverified through
`tac.witness_dsl.v10_production_receiver.realize_pair_frame1`, and scored in
38 immutable batch-16 stages.

## Macro action and costate substrate

Every experiment preserves exact reserialized `ZIP_STORED` operating points:

- full bundle;
- all-enhancement-off with typed `Y0_hat=Y1_hat` fallback;
- each 120-pair enhancement-layer eviction;
- each 120-pair base-layer eviction, marked decoder-open until a fallback is
  selected;
- each whole 120-pair segment eviction.

The full scorer receipt retains all 600 pair rows, all five-class debts, and
2,329,561 exact Seg error events. This is the correct costate source for
nonuniform bitrate allocation. RGB MSE and equal 23 kb/s segment allocation
are no longer admissible selectors.

The current final payload also pays ten independent stream resets: five
120-frame `Y1` base streams plus five 120-frame conditional streams, each with
its own keyframe/context/container overhead. Production should preserve the
five immutable encoder checkpoints, then perform one resumable final recode
into two chronological 600-frame streams (or another exact long-GOP
consolidation) before archive closure.

## Custody and sharp edges closed

Canonical dynamic composition receipt:

`/Volumes/VertigoDataTier/pact/evidence/g52_fulln600_lossy_selected_plane_scoring_frontier46_layered_20260726/scorer/composition_receipt_v2.json`

SHA-256:
`7de13052d2758f48aeade2141d15e8e7e9703d4bfed6920beb6b780ed686f64b`

It reopens the live effective pointer and binds pointer path, SHA, axis,
source, and target. Pointer SHA was
`2a61b052be496d3a9a1be1a9c230c8d179a788e61fd03472e50fc85832da94c6`;
the effective target was the official leaderboard 0.172 row.

Two metadata traps were caught without losing scientific work:

1. `score_coupled_witness_raw_debt.py` defaults to a C1 contest reference and
   has no CLI null spelling. A typed wrapper called `run()` with
   `contest_reference=None`, preventing a delayed post-stage false refusal.
2. The first composition config hardcoded 0.172. Its v1 receipt is preserved
   but superseded. V2 reopens the validated canonical pointer dynamically and
   records the scorer's actual advisory axis.

Source/bridge custody:

- V10 bridge receipt SHA:
  `9410bcaee31bdde3ffe8c1bcd82e6e81e42654ff318bb257c18f9ae866d91b6b`
- realized raw SHA:
  `948aca13f76b92142c60d9ac1b041a13fcd381f0d9a9a4c8d50ec7804b9e3aac`
- full scorer receipt file SHA:
  `533f9227673591324a53ef2bf6de82a9fcc37ccc5817386e6b7a7defda7d2459`
- scorer internal receipt SHA:
  `cffe44586ee5333cf8763a11fe6f03a02a0d721e838a4510931bef9ce0eeefb5`

## Fresh-lineage boundary and next production move

All rows above read the historical C1 archive and are permanently diagnostic.
No bitstream, plane, raw file, or target value from this run may become
candidate lineage.

Freshness must be proven by **inputs and derivation**, not by output hash
novelty. A fresh compiler may deterministically emit scorer planes equal to a
historical hash if it derives them anew from sealed `gt_f0/gt_f1` under current
batch-16 label custody, labels the source-cache poses advisory-only, and proves
that no historical archive or plane file was an input.

G51 owns that fresh exact-plane operand iterator. G52's generic codec core now
accepts three explicit encoder contracts:

- deterministic `libsvtav1` / `yuv420p` / IVF, diagnostic only for semantic
  color until channel packing is applied;
- `libx265` / `yuv444p` / raw HEVC;
- RGB-native `libx264rgb` / `rgb24` / raw H.264.

Local FFmpeg lacks `libaom-av1`; public closure must not assume it. The next
whole-population race should consume only G51's fresh iterator and compare
scorer-closed chroma-faithful representations. Unit fixtures may debug
execution, but no n12/n24/subset result is decision evidence. Full n600 exact
score selects the arm.

## Triality and six hooks

**DSL:** strict research/candidate-lineage flags; source archive hashes;
encoder/container/pixel/color/GOP/thread contract; five exact 120-pair stages;
named direct and conditional transforms.

**DAG:** source custody → plane codec → double decode → exact operating-point
bundles → V10 integer realization → n600 batch-16 scorer → dynamic pointer
composition.

**Equations:** `E=clip(round((Y0-Y1)/2)+128)` and
`Y0_hat=clip(Y1_hat+2*(E_hat-128))`; exact score
`100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489`.

1. Exact-eval: not run; historical input makes promotion illegal.
2. Serialization: deterministic ZIP_STORED bundles plus immutable JSON stages.
3. Reload: every stream reopened; decoder doubled; full LP1 A/B proved exact.
4. `uint8`: reconstruction and V10 realization are explicit and hash-bound.
5. Activation: not applicable to this post-solve codec layer.
6. Curriculum: codec rate is selected by exact byte/scorer costate, not a
   training schedule.

Pointer delta: **UNMOVED**. Advisory score 31.99 is nowhere near the frontier.

## Fresh production core now closed

The successor implementation is no longer coupled to `SourcePlanes` or the
historical C1 loader:

- `src/tac/witness_dsl/taskspace_fresh_selected_plane_codec_v1.py`
- `tools/run_taskspace_fresh_selected_plane_codec_n600.py`
- `src/tac/witness_dsl/tests/test_taskspace_fresh_selected_plane_codec_v1.py`
- `tools/tests/test_run_taskspace_fresh_selected_plane_codec_n600.py`

The fresh core enforces the G51 loader-shaped provider contract, exact
chronological pair ids, current target-label arrays, explicitly advisory
source-cache poses, five immutable stages, and a separate atomic final recode
into two long streams. Stage resume revalidates the current operand hashes;
final resume revalidates both stream and counted-bundle custody. The stage
artifacts are never discarded. Pose authority remains the final upstream
evaluation.

Public decoding is now PyAV-authoritative, not FFmpeg-CLI-authoritative. Every
stream is decoded twice through the exact PyAV path, and the bundle manifest
binds PyAV/library versions, decoded stream hashes, reconstructed `Y0/Y1`
hashes, and the exact factor-2 camera raw SHA/byte count derived from those
PyAV bytes. The generic public receiver therefore has a byte target produced
by its available dependency surface.

Production derives `av==17.0.0` from the authoritative `upstream/uv.lock` and
refuses a version mismatch. The x264rgb path requires actual native `gbrp` and
extracts the G/B/R planes directly into RGB, avoiding libswscale. A fixture
stream decoded to the same SHA
`6f6878f329589e4968e341c4f2cbcdb1c84c6d1a0c53cd4bfcf978065b13fe09`
under local PyAV 17.1.0 and authoritative PyAV 17.0.0. Requested codec/pixel
format and actual PyAV codec/native format are separately recorded.

The implementation explicitly distinguishes two types:

- `DIRECT_TASK_LAYERED`: implemented, fresh `Y1` base plus fresh conditional
  `Y0|Y1` enhancement.
- `PROGRAM_RESIDUAL_LAYERED`: blocked because G51 does not supply a fresh V15
  semantic predictor/base byte stream. No V15 composition claim is made.

Focused verification:

```text
uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_fresh_selected_plane_codec_v1.py \
  tools/tests/test_run_taskspace_fresh_selected_plane_codec_n600.py
5 passed
```

This closes the software-side candidate codec seam, not the score. The next
score-moving action is exact and singular: bind G51's immutable aggregate
receipt, run the full n600 x264rgb/x265-444 race, pass the winning two-stream
bundle through G55's PyAV/factor-2 public closure, and measure the exact public
archive with `upstream/evaluate.py`. Until then, pointer delta remains
**UNMOVED**.
