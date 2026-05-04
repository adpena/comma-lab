# PR77 Action / C089 Pose Mixed Container Candidates - 2026-05-03 Codex

Scope: local-only deterministic archive-packer/action-pose candidate builder for
PR75/PR77/C089/C091. No remote GPU job was dispatched. No Lightning state or
dispatch state was touched.

## Inputs

- C-091 PR75 replay:
  - archive: `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
  - bytes: `276481`
  - SHA-256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
  - score: `0.31516575028285976`
- C089 `c067_pr75_qp1_top40_p6`:
  - archive: `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
  - bytes: `276342`
  - SHA-256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
  - score: `0.3154707273953505`
- PR77 action-delta public archive:
  - archive: `experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip`
  - bytes: `276551`
  - SHA-256: `f90880383c95e14d82704f99db9b20944786ae6452a844348638b06c439972af`

## Tooling Added

- Builder: `experiments/build_pr77_action_pose_mixed_container_candidates.py`
- Focused tests: `src/tac/tests/test_build_pr77_action_pose_mixed_container_candidates.py`
- Output matrix:
  `experiments/results/pr77_action_pose_mixed_container_20260503_codex/candidate_matrix.json`

The builder reads only single-member `p` archives, rejects unsafe ZIP members,
uses `submissions/robust_current/unpack_renderer_payload.py` as parse truth,
writes deterministic stored ZIPs, records no-op/payload-rewrite status, and
keeps `score_claim=false` and `promotion_eligible=false`.

## Candidate Matrix

All rows are empirical byte-screen artifacts only. Deltas are archive bytes.
Sub-0.314 math is shown against the C-091 score if component terms were
unchanged; exact CUDA auth eval is required for any score claim.

| candidate | bytes | SHA-256 | delta vs C091 | delta vs C089 | delta vs PR77 | sub-0.314 remaining vs C091 | dispatch status |
|---|---:|---|---:|---:|---:|---:|---|
| `pr77_actions_pr75mask_renderer_c089pose_fixedslice` | `276329` | `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8` | `-152` | `-13` | `-222` | `1599` byte-equivalent / `0.0010645397219851693` score | exact-eval ready after claim |
| `pr77_actions_c089mask_pr75renderer_c089pose_p3` | `276332` | `a0e4b86baf01838d42083d66841a037bca7c6d98ab38c139085a23287cab3b37` | `-149` | `-10` | `-219` | `1602` byte-equivalent / `0.0010665373066628697` score | exact-eval ready after claim |
| `pr77_actions_sorted_c089mask_pr75renderer_c089pose_p6_probe` | `276337` | `686af16796bc3f3a8f3e8d050492ba77a86237ae2b4eb9df0238b221329619c8` | `-144` | `-5` | `-214` | `1607` byte-equivalent / `0.0010698665936101914` score | fail-closed; needs action-order raw-output parity |
| `c091_pr75_replay_noop_control` | `276481` | `2227c1cdc64defc88e88a31c2410a1ba88edac204274624d2aeae6ca9b4b1b3f` | `0` | `+139` | `-70` | `1751` byte-equivalent / `0.0011657502828597703` score | do not dispatch; payload-identical ZIP rewrite |
| `pr77_actions_c089mask_renderer_pose_p3_isolation` | `276541` | `ab27e04e28e86a558b5089ccce3fee3508608b2b3ca0eef7d4e18ed5c6ec7020` | `+60` | `+199` | `-10` | `1811` byte-equivalent / `0.0012057019764137684` score | exact-eval ready after claim, low byte priority |
| `pr77_replay_noop_control` | `276551` | `8f60c64b9dff70a0f53387ca108d69c768157d2deec8c8f25bfe953a9a39a360` | `+70` | `+209` | `0` | `1821` byte-equivalent / `0.0012123605570064028` score | do not dispatch; payload-identical ZIP rewrite |

## Findings

1. Highest-EV local candidate is
   `pr77_actions_pr75mask_renderer_c089pose_fixedslice`:
   - carries PR77's exact source-order action stream, so no P6 action
     reordering is needed;
   - reuses C089's smaller QP1 pose stream (`676` Brotli bytes vs PR77/C091
     `898`);
   - keeps PR75/C091 mask and renderer fixed-slice boundaries so current
     robust runtime parses it as
     `public_pr75_qzs3_qp1_segactions_fixed_slices`;
   - robust parser decoded closure passed and manifest records raw wire order
     `masks.mkv`, `renderer.bin`, `seg_tile_actions.bin`,
     `optimized_poses.qp1`.

2. Byte-only packer/action/pose work is not enough for sub-0.314:
   - best row saves `152` bytes vs C-091, worth about `0.00010121056087457006`;
   - it still needs about `1599` byte-equivalent or `0.0010645397219851693`
     component-score improvement to cross `0.314`;
   - therefore this is an action/component-improvement candidate, not a
     byte-only sub-0.314 path.

3. PR77 action stream facts:
   - encoded action bytes: `325`
   - encoded SHA-256:
     `d8c75e4f3725bbcf608434f0a78f5b37a9ce86bd8177c71092fd727d7e2af75a`
   - decoded runtime records: `147`
   - decoded SHA-256:
     `8ac9a01caad973096c58b42daf2b1a8e476ad68cf285d443baa4ac94fdb42255`
   - pair order is not nondecreasing, so source-order PR77 actions cannot be
     carried by P6 without reordering.

4. Sorted P6 probe is fail-closed:
   - PR77 has no duplicate pair/tile records, so sorted records target
     disjoint tiles structurally;
   - decoded action record order changes, so it is not dispatchable until
     local raw-output parity or an equivalent reviewed runtime proof exists.

5. No-op controls are payload-identical but ZIP-metadata rewrites:
   - C091 payload rewrite keeps archive bytes `276481` but changes SHA to
     `2227c1cdc64defc88e88a31c2410a1ba88edac204274624d2aeae6ca9b4b1b3f`;
   - PR77 payload rewrite keeps archive bytes `276551` but changes SHA to
     `8f60c64b9dff70a0f53387ca108d69c768157d2deec8c8f25bfe953a9a39a360`;
   - both are marked `payload_identical_zip_metadata_changed` and must not be
     dispatched.

## Exact Next Dispatch Recommendations

1. First dispatch choice, if a local-only selection is later allowed to move to
   exact eval:
   - candidate:
     `experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip`
   - bytes/SHA:
     `276329` /
     `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`
   - reason: exact PR77 action order, best byte count, parser-validated
     fixed-slice closure, and C089 pose byte savings.

2. Fallback if fixed-slice mixed-total parsing is considered too brittle:
   - candidate:
     `experiments/results/pr77_action_pose_mixed_container_20260503_codex/pr77_actions_c089mask_pr75renderer_c089pose_p3/archive.zip`
   - bytes/SHA:
     `276332` /
     `a0e4b86baf01838d42083d66841a037bca7c6d98ab38c139085a23287cab3b37`
   - reason: self-describing P3 header costs `+3` bytes versus the fixed-slice
     winner but avoids relying on inferred fixed-slice lengths.

3. Do not dispatch the sorted P6 probe until an action-order parity artifact
   exists. Do not dispatch no-op controls.

Any exact eval dispatch must first claim the lane with
`tools/claim_lane_dispatch.py claim ...`, then run the canonical CUDA path
`archive.zip -> inflate.sh -> upstream/evaluate.py` via
`experiments/contest_auth_eval.py --device cuda`, preserving
`contest_auth_eval.json`, logs, archive SHA/bytes, runtime tree hash, and
component gates.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile experiments/build_pr77_action_pose_mixed_container_candidates.py src/tac/tests/test_build_pr77_action_pose_mixed_container_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr77_action_pose_mixed_container_candidates.py -q
.venv/bin/python experiments/build_pr77_action_pose_mixed_container_candidates.py --force
```

Results:
- focused tests: `2 passed`
- real local build: `6` candidates emitted
- no remote dispatch performed
