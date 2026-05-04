# C091-Relative PR75/PR77 Action/Pose Packer Matrix - 2026-05-03 Codex

Scope: local-only C091-relative action/pose packer review for PR75/PR77
sub-0.314 candidates. No remote GPU job was dispatched. `.omx/state` active
claims were not read or modified.

## Inputs

- C091 PR75 public replay anchor:
  - bytes: `276481`
  - SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
  - score: `0.31516575028285976`
  - components: SegNet `0.060804000000000004`, PoseNet
    `0.07026450028285976`, component total `0.13106850028285977`
- PR77 public replay exact T4 evidence:
  - archive:
    `experiments/results/lightning_batch/exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/contest_auth_eval.json`
  - score: `0.31537874750377204`
  - component total: `0.13123474750377203`
  - component delta vs C091: `+0.00016624722091226085` score worse
- Candidate matrix:
  - `experiments/results/c091_relative_pr75_pr77_action_pose_20260503_codex/candidate_matrix.json`
- Tool:
  - `experiments/build_c091_relative_pr75_pr77_action_pose_matrix.py`

## Candidate Matrix

All rows are local empirical byte-screen or prior exact-eval context only.
Every row has `score_claim=false`, `promotion_eligible=false`, and
`sub314_dispatch_worthy=false`.

| rank | candidate | bytes | SHA-256 | delta vs C091 | sub-0.314 component gain needed | readiness |
|---:|---|---:|---|---:|---:|---|
| 1 | `pr77_actions_pr75mask_renderer_c089pose_fixedslice` | `276329` | `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8` | `-152` | `0.0010645397219851693` / `1599` byte-equivalent | already queued separately as `exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z`; ready only after dispatch claim |
| 2 | `c067_pr75_actions_pose_safe_positive_ampminus1_p6` | `276317` | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | `-164` | `0.0010565494145477472` / `1587` byte-equivalent | local archive valid; future exact eval only after claim |
| 3 | `c067_pr75_actions_positive_poseharm_ampminus1_p6` | `276389` | `244c366a7d07ff185091dbbcf7ecb1e0308d11d9e7467bc2ae2eb2f8b6bd0a6a` | `-92` | `0.001104491259172502` / `1659` byte-equivalent | local archive valid; future exact eval only after claim |
| - | `pr77_actions_c089mask_pr75renderer_c089pose_p3` | `276332` | `a0e4b86baf01838d42083d66841a037bca7c6d98ab38c139085a23287cab3b37` | `-149` | `0.0010665372988445387` / `1602` byte-equivalent | fallback only if fixed-slice row fails runtime/parser custody |
| - | `pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe` | `276337` | `686af16796bc3f3a8f3e8d050492ba77a86237ae2b4eb9df0238b221329619c8` | `-144` | `0.0010698665936101914` / `1607` byte-equivalent | fail-closed pending action-order raw-output parity |

## Findings

1. No new local candidate is sub-0.314 dispatch-worthy.
   The best byte saves are only `152` to `164` bytes versus C091, worth about
   `0.00010` score. Sub-0.314 still needs about `0.00106` component-score
   improvement after rate savings.

2. PR77 is not positive exact-eval evidence yet.
   The public PR77 replay exact T4 result is worse than C091 by
   `0.0002130` total score and `0.0001662` component score. The fixed-slice
   PR77/C089-pose row is still a valid exact-eval probe because it changes the
   pose stream and bytes, but the local math does not support a sub-0.314
   expectation.

3. PR75 action-policy rows have real action changes but insufficient modeled
   scale.
   `pose_safe_positive_ampminus1_p6` changes `28` action ids with positive
   selected pose/seg trace sums and no duplicate pair/tile records. Its
   selected combined trace sum is `0.0001781321228804055`, far below the
   `0.0010565494145477472` C091-relative sub-0.314 break-even.
   `positive_poseharm_ampminus1_p6` has larger selected SegNet trace
   (`0.00026194259407930076`) but needs `0.001104491259172502` component score.

4. The sorted PR77 P6 probe remains non-dispatchable.
   It changes decoded action record order; it needs local raw-output parity or
   an equivalent reviewed proof before any exact eval can be considered.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile experiments/build_c091_relative_pr75_pr77_action_pose_matrix.py src/tac/tests/test_build_c091_relative_pr75_pr77_action_pose_matrix.py
.venv/bin/python -m pytest src/tac/tests/test_build_c091_relative_pr75_pr77_action_pose_matrix.py -q
.venv/bin/python experiments/build_c091_relative_pr75_pr77_action_pose_matrix.py
```

Results:

- focused tests: `3 passed`
- matrix rows emitted: `5`
- all candidate archive profiles in the matrix passed single stored `p` member
  custody checks
- no remote dispatch performed
