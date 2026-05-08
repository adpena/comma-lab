# Modal Phase A1 Wrong Source Root Failure — 2026-05-08

## Evidence

- Lane: `track1_phase_a1_score_gradient`
- Modal call: `fc-01KR4TVY14SWW0VN07XT1B4Y2Q`
- Instance/job id: `track1_phase_a1_score_gradient_20260508T222043Z_modal`
- Local artifacts:
  `experiments/results/track1_phase_a1_score_gradient_20260508T222043Z_modal/`
- Terminal claim status:
  `failed_archive_build_wrong_pr101_source_root`

## What happened

The opt-in Modal fallback correctly continued past the DALI/NVDEC exact-eval
preflight failure and completed CUDA training:

- Stage 1 train return code: `0`
- Stage 1 elapsed: `1825.34` seconds
- Training manifest:
  `experiments/results/track1_phase_a1_score_gradient_20260508T222043Z_modal/harvested_artifacts/train__build_manifest.json`

Archive build then failed:

```text
FATAL: PR101 source missing:
/tmp/modal_phase_a1/track1_phase_a1_score_gradient_20260508T222043Z_modal/inputs/pr101_src/codec.py
```

Root cause: local dispatch packaged the detached checkout parent
`.../source/` instead of the PR101 runtime source root
`.../source/submissions/hnerv_ft_microcodec/src`.

The bad path uploaded a `387849273` B source snapshot, versus the correct
runtime source snapshot of `19137` B.

## Score and promotion status

No candidate archive was produced. No exact CUDA or CPU eval ran. This is
not score evidence.

Training-side diagnostic only:

- `initial_pose_dist`: `0.10996793210506439`
- `final_pose_dist`: `0.33592045307159424`
- `initial_seg_dist`: `0.039058029651641846`
- `final_seg_dist`: `0.03631831705570221`
- `pose_delta_pct`: `-205.47`
- `seg_delta_pct`: `+7.01`

This diagnostic does not promote or retire A1, because it is pre-archive,
pre-exact-eval, and used the failed build path.

## Guard added

`experiments/modal_phase_a1_score_gradient_pr101.py` now validates that
`--pr101-source-dir` contains `codec.py` and `model.py` at the root before
it creates the in-memory source zip. If the caller passes the parent
`source/` directory, it fails locally before any lane claim or Modal spend
and prints the likely corrected nested path.

## Reactivation criteria

1. Relaunch with:
   `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src`
2. Preserve `--continue-after-nvdec-failure` only if the Modal DALI/NVDEC
   probe still fails; otherwise run exact CUDA in the same Modal chain.
3. Promote only from a candidate archive with paired exact CUDA and CPU
   eval artifacts.
