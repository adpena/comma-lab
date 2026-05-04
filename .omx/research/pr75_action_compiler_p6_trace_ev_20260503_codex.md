# PR75 Action Compiler P6 Trace-EV Matrix - 2026-05-03

## Scope

Local-only continuation of the PR75 tile-action/action-dictionary search. No
remote GPU, Lightning, Modal, Vast.ai, or other dispatch was performed. The
outputs below are deterministic archive candidates and planning artifacts only;
they are `byte_and_trace_planning_only_until_exact_cuda` until the exact
`archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA auth path measures the
exact bytes.

## Inputs

- C067/T4 baseline archive and trace:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z`
- PR75 reverse-engineered archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip`
- PR75 reverse-engineering action records and stream anatomy:
  `experiments/results/top_submission_reverse_engineering_20260503_deep_codex/comparisons/action_records.csv`
  `experiments/results/top_submission_reverse_engineering_20260503_deep_codex/comparisons/stream_analysis.json`
- Exact T4 action traces used for ranking/calibration:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_only_p3_t4_20260503T0401Z/component_trace.json`
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/component_trace.json`
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_p3_t4_20260503T0440Z/component_trace.json`
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top25_ampminus1_p3_t4_20260503T0520Z/component_trace.json`
- Diagnostic CUDA trace used only as a non-T4 transform clue:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_actions_top40_ampminus2_p3_l40sdiag_20260503T0512Z/component_trace.json`

## Evidence Read

The latest T4 replay invalidates the earlier RTX PRO ordering for
`top25_ampminus1`: on T4 it scores worse than both `top25_p3` and `top40_p3`.
The active local opportunity is therefore not another P3/P5 dictionary sweep by
default; it is to combine exact T4 per-pair action traces with the P6
pair-delta-varint encoding already supported by the runtime, then reserve a
single charged P5 custom-dictionary probe as a nonlinear escape check.

The builder now emits explicit `noop`, `source_preserving`, and
`source_preservation` flags. `all:p6` is the source-preserving PR75 action-body
control; all subset and transformed rows are marked as non-source-preserving.

## Output Matrix

Matrix path:
`experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/candidate_matrix.json`

Matrix SHA-256:
`911d24786c6c0ed583d27f22beb612917758154b27109d70a6bd5a0c45099b0b`

| priority | candidate | bytes | sha256 | records | expected score | trace tradeoff | guards |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| 1 | `c067_pr75_actions_lag_eval_top67_p6/archive.zip` | 276352 | `d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972` | 44 | 0.315442604958 | seg +0.000224643, pose +0.000041610, rate +0.000091883 | noop=false, source_preserving=false |
| 2 | `c067_pr75_actions_lag_eval_pose2_top67_p6/archive.zip` | 276338 | `af7a34cb1c051b1accebe2768245a44f55280e2596b315f8e4809a73a23926cd` | 38 | 0.315454715933 | seg +0.000183953, pose +0.000060867, rate +0.000082563 | noop=false, source_preserving=false |
| 3 | `c067_pr75_actions_lag_eval_pose4_top67_p6/archive.zip` | 276338 | `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef` | 38 | 0.315460650152 | seg +0.000173781, pose +0.000065105, rate +0.000082563 | noop=false, source_preserving=false |
| 4 | `c067_pr75_actions_top49_p6/archive.zip` | 276368 | `bc2db878509e1ae30f09ab9e28d1feeb28cbff9c14eac6afea64309c68a880ca` | 49 | 0.315468090190 | seg +0.000225491, pose +0.000025930, rate +0.000102576 | noop=false, source_preserving=false |
| 5 | `c067_pr75_actions_top40_p6/archive.zip` | 276342 | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | 40 | 0.315468893593 | seg +0.000205994, pose +0.000027312, rate +0.000085226 | noop=false, source_preserving=false |
| 6 | `c067_pr75_actions_top40_ampminus2_p6/archive.zip` | 276347 | `f573c8a93b8d2146459cfc96109c6d84ee0ed71508621ddaa248683740cc36f8` | 40 | 0.315472222888 | seg +0.000205994, pose +0.000027312, rate +0.000088555 | noop=false, source_preserving=false |
| 7 | `c067_pr75_actions_beam_pose4_top55_p6/archive.zip` | 276334 | `9e7ccb5577e14bab5e6e730a0b42712fdf522baa70251e3f81b40b735d13f450` | 37 | 0.315477424325 | seg +0.000181410, pose +0.000038038, rate +0.000079899 | noop=false, source_preserving=false |
| 8 | `c067_pr75_actions_positive_p6/archive.zip` | 276395 | `20d0c59f83b10d1006a2a8b61e09d732573680b549d39510eaaac44349fa0a0a` | 60 | 0.315479007865 | seg +0.000261943, pose -0.000003461, rate +0.000120521 | noop=false, source_preserving=false |
| 9 | `c067_pr75_actions_all_p6/archive.zip` | 276417 | `93d3de00e595c793d745539c823ec5731a98bc3999b306e29dce5197f83c78f0` | 67 | 0.315502108501 | seg +0.000274658, pose -0.000024628, rate +0.000135172 | noop=false, source_preserving=true |
| 10 | `c067_pr75_actions_top40_custompose125_p5/archive.zip` | 276550 | `a45a52c9aee24ecbe0d2bbd68044c76900b4c860f0f3e217cf0c4fe199fffb4f` | 40 | 0.315607392255 | seg +0.000205994, pose +0.000027312, rate +0.000223738 | noop=false, source_preserving=false, custom_dictionary=true |

Expected scores are C067 T4 component score minus selected trace benefit plus
archive-rate delta. They are not score truth; exact T4 replay can reorder the
matrix.

## Verification

- `.venv/bin/python -m py_compile experiments/build_pr75_tile_action_subset_candidates.py src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`
- `.venv/bin/python -m pytest src/tac/tests/test_build_pr75_tile_action_subset_candidates.py`
- Local unpack smoke parsed the top three P6 candidates as
  `public_pr75_qzs3_qp1_segactions_p6_delta_varint` and the P5 probe as
  `public_pr75_qzs3_qp1_segactions_p5_packed_custom_dict`.

No remote dispatch was performed.

## Exact T4 follow-up status - 2026-05-03T07:00Z

The P6/action planning matrix is now partially exact-evaluated under the
QP1-float32 runtime.

Confirmed A++ frontier from this family:

- `c082_qp1_p6_delta_varint_actions_stream_resweep`
  - Job:
    `exact_eval_c082_qp1_p6_delta_varint_actions_stream_resweep_t4_20260503T0626Z`
  - Score: `0.3154889937553647`
  - Bytes: `276394`
  - SHA-256:
    `9b78333dd39c12c986ca7fc02a4bb35a6a61ef5a1cc0c9c9c840820eb840058a`
  - PoseNet: `0.00049675`
  - SegNet: `0.00060969`
  - Evidence: `A++ contest T4`, `promotion_eligible=true`

Active exact-eval jobs:

| job | candidate | expected bytes | expected SHA-256 | status |
|---|---|---:|---|---|
| `exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z` | `top40_p6` | `276342` | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | running |
| `exact_eval_c067_pr75_qp1_lag_eval_pose4_top67_p6_t4_20260503T0626Z` | `lag_eval_pose4_top67_p6` | `276338` | `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef` | running |
| `exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z` | `pose_safe_positive_ampminus1_p6` | `276317` | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | running |
| `exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z` | smaller decoded-stream-preserving C082 repack | `276333` | `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681` | pending |

The current exact result shows that T4 component noise and runtime custody are
large enough to reorder trace predictions at the `1e-5` level. Continue to use
the trace-EV matrix for dispatch ordering, but not for claims or retirement.

## Exact T4 harvest update - 2026-05-03T07:14Z

Three P6/action candidates completed and were harvested through state-derived
SSH custody:

| candidate | job | score | bytes | SHA-256 | status |
|---|---|---:|---:|---|---|
| `top40_p6` | `exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z` | `0.3154707273953505` | `276342` | `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8` | A++ frontier |
| `lag_eval_pose4_top67_p6` | `exact_eval_c067_pr75_qp1_lag_eval_pose4_top67_p6_t4_20260503T0626Z` | `0.31552758560163663` | `276338` | `de7d549ec4437b3ed9508c1f62f8a79bef88440ba3db0f4a7de5a2c83fe160ef` | A++ non-frontier |
| `pose_safe_positive_ampminus1_p6` | `exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z` | `0.31556196759570776` | `276317` | `6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796` | A++ non-frontier |

The smaller decoded-stream-preserving C082 repack also completed:

- Job: `exact_eval_c082_p6_stream_resweep_276333_t4_20260503T0705Z`
- Score: `0.3154874419767294`
- Bytes: `276333`
- SHA-256:
  `30932c684c6b09a7bd2bd248e17fd66a4bc448ed2fe5747f92e74bcceec66681`
- Evidence: `A++ contest T4`, `promotion_eligible=true`
- Interpretation: it beats the older C082 row but not the new `top40_p6`
  frontier.

The L40S duplicate for the same C082 bytes scored `0.31536316025521677` and
is useful only as diagnostic hardware-variance evidence; the T4 packet is the
promotion authority and did not move the current frontier.

Current exact best after this harvest:

- `c067_pr75_qp1_top40_p6`
- Score `0.3154707273953505`
- Bytes `276342`
- SHA-256
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`

Dispatch implication: do not spend more T4 capacity on action-only micro-variants
unless their byte-screen or component trace implies a plausible `>1e-4` score
move. The active renderer-shrink exact eval and renderer self-compression burns
remain higher EV for crossing `0.314`.
