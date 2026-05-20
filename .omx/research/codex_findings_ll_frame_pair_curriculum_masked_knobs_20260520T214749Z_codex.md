# Codex Findings: LL Frame/Pair Curriculum and Masked Knob Plan

Timestamp: 2026-05-20T21:47:49Z
Lane: `lane_master_gradient_frame_decomposition_20260520`

## Verdict

Extended the per-frame master-gradient landing into an actionable curriculum
and exact-frame correction-knob queue. The design follows the scorer topology:
SegNet is trained/searched per last frame; PoseNet is trained/searched per
canonical pair.

This remains non-promotional. All outputs carry `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Landed implementation

- `src/tac/optimization/frame_pair_curriculum.py`
  - Converts `frame_axis_l1[frame, axis]` into:
    - SegNet frame sampling probabilities.
    - PoseNet frame and pair sampling probabilities.
    - Pair axis mixes.
    - Photoshop-style masked adjustment layers.
  - Keeps the sparse-residual over-budget prohibition from the LL response
    plan and prefers byte-neutral masked runtime knobs before charged sparse
    pixels.
- `tools/build_ll_frame_pair_curriculum.py`
  - Builds JSON/Markdown curriculum artifacts from the per-frame `.npy`
    output plus optional decomposition and LL response-plan JSON.
- `src/tac/optimization/inflate_postprocess_surface.py`
  - Added exact `frame_indices` targeting to raw postprocess specs.
  - Existing all/even/odd behavior is unchanged.
- `tools/run_inflate_postprocess_smoke.py`
  - Added `--custom-spec-json`, so exact-frame specs can be smoked by the
    existing raw postprocess path.
- `tools/build_masked_knob_postprocess_specs.py`
  - Converts curriculum adjustment layers into concrete exact-frame
    `PostprocessSpec` JSON rows.
- Tests:
  - `src/tac/tests/test_frame_pair_curriculum.py`
  - `src/tac/tests/test_masked_knob_postprocess_specs.py`
  - Extended `src/tac/tests/test_inflate_postprocess_surface.py`

## Materialized artifacts

Curriculum:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_20260520_codex.json`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/ll_frame_pair_curriculum_20260520_codex.md`

Masked postprocess specs:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/masked_knob_postprocess_specs_20260520_codex.json`

## Empirical/procedural output

The curriculum was built from:

- `master_gradient_frame_axis_l1_20260520_codex.npy`
- `master_gradient_frame_decomposition_20260520_codex.json`
- `ll_scorer_response_next_probe_plan_20260520_codex.json`

Sampling probability checks all sum to `1.0`:

- `seg_frame_prob_sum=1.0`
- `pose_frame_prob_sum=1.0`
- `total_frame_prob_sum=1.0`
- `pair_pose_prob_sum=1.0`
- `pair_total_prob_sum=1.0`

Top pair ordering:

1. Pair `7` frames `(14,15)`: total `51.8510`, seg share `0.636`, pose share `0.364`
2. Pair `6` frames `(12,13)`: total `48.0112`, seg share `0.670`, pose share `0.330`
3. Pair `4` frames `(8,9)`: total `34.7058`, seg share `0.791`, pose share `0.209`
4. Pair `2` frames `(4,5)`: total `34.6151`, seg share `0.826`, pose share `0.174`
5. Pair `5` frames `(10,11)`: total `30.7876`, seg share `0.879`, pose share `0.121`
6. Pair `3` frames `(6,7)`: total `27.5818`, seg share `0.906`, pose share `0.094`
7. Pair `1` frames `(2,3)`: total `27.1909`, seg share `0.885`, pose share `0.115`
8. Pair `0` frames `(0,1)`: total `26.0508`, seg share `0.992`, pose share `0.008`

Generated adjustment layers:

- `11` curriculum layers.
- Pairs `7` and `6` get both SegNet boundary/tone layers and PoseNet global
  pair layers.
- Remaining top pairs are SegNet-dominant and get last-frame boundary/tone
  layers.

Generated smoke specs:

- `12` exact-frame raw postprocess specs.
- `3`-spec top subset for visible-change smoke:
  - `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/masked_knob_postprocess_specs_top3_20260520_codex.json`
- Examples:
  - `pair_0007_seg_boundary_last_frame_rgb_bias_m1`: channel bias on frame `15`.
  - `pair_0007_seg_boundary_last_frame_rgb_bias_p1`: channel bias on frame `15`.
  - `pair_0007_pose_global_pair_temporal_blend_a1_8`: temporal blend on frames `14,15`.
  - `pair_0006_seg_boundary_last_frame_rgb_bias_m1`: channel bias on frame `13`.

Visible-change smoke:

- Output:
  - `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/masked_knob_postprocess_top3_visible_smoke_20260520_codex/summary.json`
- Candidate count: `3`
- Visible changes: `3`
- Advisory scoring: intentionally not run.
- Full-size candidate raws left behind: `0` (`--cleanup-candidate-raw`)
- Per-spec raw mutation:
  - `pair_0007_seg_boundary_last_frame_rgb_bias_m1`: changed `1` frame, `2805309` bytes, max abs delta `1`.
  - `pair_0007_seg_boundary_last_frame_rgb_bias_p1`: changed `1` frame, `3047401` bytes, max abs delta `1`.
  - `pair_0007_pose_global_pair_temporal_blend_a1_8`: changed `2` frames, `5763773` bytes, max abs delta `32`.

Top-3 advisory scoring:

- Output:
  - `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/masked_knob_postprocess_top3_advisory_20260520_codex/summary.json`
- Baseline score: `0.19206142414659494`
- Candidate count: `3`
- Advisory successes: `3`
- Improved count: `0`
- Best row: `pair_0007_seg_boundary_last_frame_rgb_bias_p1`
  - score `0.19206733847162177`
  - delta `+0.000005914325026834533`
- Other scored rows:
  - `pair_0007_seg_boundary_last_frame_rgb_bias_m1`: score `0.19207716563717325`
  - `pair_0007_pose_global_pair_temporal_blend_a1_8`: regressed more than the one-frame bias rows
- Full-size candidate raws left behind: `0` (`--cleanup-candidate-raw`)

Response dataset refreshed after top-3 advisory:

- `row_count=29`
- `family_counts`: decoder_q `21`, inflate_postprocess `6`, scorer_gradient_sparse_residual `1`, sparse_residual_oracle `1`
- `improved_total_score_count=0`
- Best total row is now the exact-frame masked knob `+1` bias on frame `15`, still a regression.
- Sparse-residual widening remains blocked by the same negative byte-budget margin.

## Interpretation

This operationalizes the "Photoshop layer" model:

- SegNet-visible odd frames are not just harder frames; they are the natural
  surface for masked boundary/color adjustments.
- PoseNet-relevant pairs need coherent pair-level low-frequency or temporal
  knobs, not isolated last-frame pixel tweaks.
- Sparse residual widening remains blocked until the byte budget break-even
  gate is satisfied. The current path is therefore exact-frame, byte-neutral
  runtime knob smoke first.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_frame_pair_curriculum.py src/tac/tests/test_master_gradient_frame_decomposition.py src/tac/tests/test_inflate_postprocess_surface.py src/tac/tests/test_masked_knob_postprocess_specs.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_scorer_gradient_sparse_residual.py src/tac/tests/test_sparse_residual_oracle.py`
  - `33 passed`
- `.venv/bin/python -m py_compile src/tac/optimization/frame_pair_curriculum.py tools/build_ll_frame_pair_curriculum.py src/tac/optimization/inflate_postprocess_surface.py tools/run_inflate_postprocess_smoke.py tools/build_masked_knob_postprocess_specs.py`
  - passed
- `git diff --check -- <touched files>`
  - passed
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1080 lane(s) validated cleanly`
- `.venv/bin/python tools/run_inflate_postprocess_smoke.py ... --custom-spec-json masked_knob_postprocess_specs_top3_20260520_codex.json --cleanup-candidate-raw`
  - `candidate_count=3`, `visible_change_count=3`, no `0.raw` files retained
- `.venv/bin/python tools/run_inflate_postprocess_smoke.py ... --run-advisory --baseline-score 0.19206142414659494 --custom-spec-json masked_knob_postprocess_specs_top3_20260520_codex.json --cleanup-candidate-raw`
  - `advisory_success_count=3`, `improved_count=0`, best delta `+0.000005914325026834533`

## Next action

Do not promote the current top-3 masked knobs. They are useful response labels
because the best row is only a `+5.91e-06` regression, but not a candidate.
Next frontier-moving branch is the complementary null-space path: partition
bytes by max-over-pairs master-gradient magnitude, identify low-sensitivity
bytes, and route procedural re-derivation candidates through
`tac.null_space_exploiter` / `tac.procedural_codebook_generator` rather than
adding sparse residual payload.
