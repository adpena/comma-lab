# Public Floor Next Breakthrough Worker - 2026-05-03

Evidence grade: `empirical_planning_only`.

No remote GPU job was dispatched and no `.omx/state` dispatch claim was read or edited.

## Anchor

- C091 score: `0.31516575028285976`.
- C091 bytes/SHA-256: `276481`, `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`.
- Strict unchanged-component bytes needed for `<0.314`: `1751`.

## Recommendation

`do_not_dispatch`: no new non-queued opportunity is both byte-sufficient or component-break-even and locally safe; parent-queued lanes are explicitly excluded

## Excluded Parent-Queued Lanes

- `c091_native_cem_pose_waterfill_top128_s025`: bytes `276489`, needs component gain `0.0011710771544847232`; excluded to avoid duplicating the parent queue.
- `c091_pr65_pose_qp1_c089_actions_p6`: bytes `276346`, needs component gain `0.0010758593241882441`; excluded to avoid duplicating the parent queue.

## Ranked Non-Queued Opportunities

1. `qzs3_b0064_c091_p3_preserved_minp_slices` - `blocked_pose_safety`
   - bytes/SHA: `274840`, `1af9c5db6e9990e368c600c8246111e53de81aff9b72c3feb0a94b01b86c896b`
   - unchanged-component score: `0.3140730757407863`; component gain still needed: `7.307574078629919e-05`
   - safety: safe=`False`, mean/rms/max=`7.243117809295654`/`11.985550127767786`/`198.90521240234375`
   - why not: needs 110 more byte-equivalent savings or component gain after rate; renderer output parity failed local pose-safety
2. `qzs3_b0096_c091_p3_preserved_minp_slices` - `blocked_pose_safety`
   - bytes/SHA: `274166`, `711af4b78976f9302d1d01c5ef0359faab9316f2a1df20e9e240a0170061c516`
   - unchanged-component score: `0.31362428680638194`; component gain still needed: `0.0`
   - safety: safe=`False`, mean/rms/max=`8.704894065856934`/`14.295000730970163`/`231.69911193847656`
   - why not: renderer output parity failed local pose-safety
3. `public_renderer_c089_p6_lossless_stream_resweep` - `blocked_break_even_and_renderer_gate`
   - bytes/SHA: `276124`, `337b04040bee1316375bae8b2cfc2f08acddc235e4ee02d37ecfd62c0c831d95`
   - unchanged-component score: `0.31492803863659513`; component gain still needed: `0.0009280386365951299`
   - why not: saves only 357 bytes versus C091; still needs about 0.000928 component-score gain; renderer transplant gate remains unresolved
4. `c091_native_action_atom_upper_bound` - `blocked_component_break_even`
   - bytes/SHA: `276317`, `n/a`
   - unchanged-component score: `0.31505654941454775`; component gain still needed: `0.0010565494145477472`
   - why not: component proxy upper bound 0.000127120115 is below needed 0.001056549415; no exact-eval dispatch: all observed exact global action candidates are worse than C091, PR77/C089-pose fixed-slice is strongly pose-toxic, and the conservative atom upper bound does not clear the sub-0.314 component break-even
5. `c091_mask_lossless_brotli_resweep` - `safe_but_too_small`
   - bytes/SHA: `276474`, `n/a`
   - unchanged-component score: `0.3151610892701879`; component gain still needed: `0.0011610892701878761`
   - why not: saves only 7 mask-stream bytes; self-describing header overhead dominates standalone use
6. `geometry_bounded_mask_reencode_family` - `blocked_geometry_safety`
   - bytes/SHA: `133099`, `912b391a3199b6bc4358cd2b04bfbcf632e5255d454e70a5f0b22c3ba98f7f44`
   - unchanged-component score: `0.21969356186629657`; component gain still needed: `0.0`
   - why not: large byte saves change decoded mask classes; exact CDO1 repair overlays cost hundreds of KB after compression
7. `pr77_actions_c089_pose_fixedslice_exact_negative` - `measured_negative`
   - bytes/SHA: `276329`, `27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8`
   - unchanged-component score: `0.31506453972198517`; component gain still needed: `0.0010645397219851693`
   - why not: exact T4 score is worse than C091 by more than 0.003

## Verification

- Planner artifacts are deterministic JSON/Markdown.
- Score claims remain `false`; promotion eligible remains `false`.
