# PR103-on-PR106 Arithmetic Decoder Repack Exact Eval

Date: 2026-05-07
Owner: codex
Evidence grade: A++ contest T4

## Result

Supersession note, 2026-05-10: this remains an exact CUDA score anchor for
planning under the delegated runtime used by the eval, but it is not a
contest-final upload-packet proof for the tracked static wrapper. The later
runtime-custody hardening memo
`.omx/research/pr103_static_release_runtime_custody_hardening_20260507_codex.md`
shows the static wrapper fails `submission_runtime_tree_matches_auth_eval`
until the wrapper itself is exact-evaluated or a reviewed delegate-root proof is
encoded and re-evaluated.

The PR103 arithmetic-coded decoder packing was transplanted into the PR106
frontier envelope and evaluated through the canonical exact CUDA path.

- Archive: `experiments/results/pr103_repack_pr106_standalone_20260507/exact_eval_static_release_surface/archive.zip`
- Archive bytes: `185578`
- Archive SHA-256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Packed member: `0.bin`
- Packed member SHA-256: `3272ec95a2ea5ec68feb1a53fa53f6b14bdae3883fac38ee2261cdadb1b16357`
- Runtime: `submissions/pr103_pr106_final_runtime/inflate.sh`
- Exact eval artifact: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json`
- Final compliance artifact: `experiments/results/pr103_repack_pr106_standalone_20260507/pre_submission_compliance.contest_final.json`

The direct Lightning exact-eval JSON is local custody and intentionally ignored
by git. The tracked clean-checkout score snapshot is the contest-final
compliance JSON above; preflight validates its `passed` state, check list,
strict formula score, report-reconstructed score, bytes, archive SHA-256,
self-contained anchor proof, and A++ auth-eval record before accepting the
anchor.

Contest CUDA metrics:

- Strict formula score: `0.2089810755823297`
- Report-reconstructed score: `0.20898105277982337`
- Rounded contest display: `0.21`
- SegNet distance: `0.00067082`
- PoseNet distance: `0.0000336`
- Samples: `600`
- Device: `cuda`
- GPU: `Tesla T4`
- Runtime tree SHA-256: `54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6`
- Score delta vs `0.20935` baseline: `-0.0003689244176703122`

The strict formula score recomputes `100*seg + sqrt(10*pose) +
25*archive_bytes/37,545,489` from the displayed component distances and exact
charged archive bytes. The report-reconstructed score is lower by
`2.280250632611279e-08` because `upstream/evaluate.py` prints compression rate
to 8 decimal places and `contest_auth_eval.py` reconstructs its JSON score from
that rounded report field.

The strict contest-final compliance gate passed after harvest. It verified the
archive SHA/bytes, single safe ZIP member, manifest, report, T4-equivalent auth
eval, runtime-tree SHA, strict formula proof, score reconstruction, and terminal
dispatch claim.

## Dispatch Custody

Lane claim closed:

- Lane: `hnerv_pr103_pr106_ac_repack_exact_eval`
- Platform: `lightning`
- Job: `pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z`
- Terminal status: `completed_score_0_2089810528`

Duplicate reservation closed:

- Lane: `pr103_pr106_standalone`
- Job: `pending-gpu-billing-20260507`
- Terminal status: `stale_superseded_by_codex_exact_cuda_eval`

This prevents the same exact archive from remaining as an active phantom GPU
reservation in `.omx/state/active_lane_dispatch_claims.md`.

## Interpretation

This is a confirmed rate-only win over the PR106 low-level Brotli baseline, but
not a public-top-3 score. The observed score is close to the predicted
rate-only expectation, which means the arithmetic-coded decoder packing did not
silently collapse the renderer, masks, or poses. It should become the current
strict A++ local floor for HNeRV envelope repack work until a lower exact CUDA
score lands.

The immediate implication is that smaller rate-only candidates such as
`lgblock16`, HDM3 fixed-schema recode, and PR101 f32 schema recode should not be
dispatched for score lowering unless they either stack on this packet, beat
`185578` charged bytes, or provide needed proof coverage after a failure class
appears. The next high-EV work should move to scorer-changing or larger
representation-changing axes: categorical/self-compression labels, LA-pose and
telescopic foveation, sensitivity-aware byte allocation, HDC2/range/ANS entropy
gaps, and cross-paradigm stack composition.

## Next Tranche

1. Feed this exact result into the Pareto/meta-Lagrangian frontier as an A++
   archive-byte anchor.
2. Require all next HNeRV byte candidates to compare against `185578` bytes,
   not the older `186239`/`186080` floor.
3. Continue exact-evaluable hidden-gem search only when the candidate changes
   payload bytes, records old/new SHA-256, and has a strict packet readiness
   path.
4. Prioritize cross-paradigm score-changing work over further one-byte Brotli
   saturation: categorical label priors, LA-pose/foveation transport fields,
   entropy-rate decomposition, and joint field-equation atom allocation.
5. Preserve this result as paper/OSS evidence only with the A++ qualifier and
   artifact links above; do not generalize it into a broad proof that PR103
   arithmetic coding dominates all HNeRV variants.
