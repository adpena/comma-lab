# CQ2 NEXT-IF-RESUMED

CQ2 is blocked before training by dataset custody:
`/Volumes/VertigoDataTier/pact/public_datasets/comma10k` is an incomplete or
in-progress git clone, with unresolved `HEAD`, `.git/shallow.lock` present, and
no non-git image/label data found in the maxdepth-4 preflight at
`2026-08-04T22:27:28Z`.

Do not train, select, quantize, or overlap-test a student until the public
comma10k dataset is complete. Do not download in this arm and do not symlink
around the SSD path.

Fire order when the clone is complete:

1. Re-run dataset custody preflight: record remote URL, git HEAD, file counts,
   top-level tree shape, and image/label asset counts.
2. Use the conventional comma10k val list if present; otherwise create a seeded
   train/val split and record the seed.
3. Train only public-data-only students against the frozen public teacher, with
   Road/Lane-weighted distillation and resumable checkpoints on
   `/Volumes/VertigoDataTier/pact/ddm_cq2_20260804/`.
4. Measure comma10k-val Road/Lane IoU and counted student bytes for the size
   curve. Choose the smallest candidate meeting the pre-registered comma10k-val
   bar before any contest-side read.
5. Run CQ1's n32 overlap measurement once for the frozen selected student.

Derived thresholds to keep in view:

| student bytes | side-implied stream total / break-even | explicit-direction stream total / break-even |
|---:|---:|---:|
| `25,000` | `106,365 B / 0.516810` | `125,904 B / 0.611747` |
| `75,000` | `156,365 B / 0.759752` | `175,904 B / 0.854688` |
| `150,000` | `231,365 B / 1.124164` | `250,904 B / 1.219101` |

Boundaries stay live: no contest SegNet/PoseNet forward, no archive score
claim, no `/tmp` persisted evidence, and candidate selection must be based on
comma10k-val metrics before the final n32 overlap read.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`;
contest pointer borrowed/unmoved.
