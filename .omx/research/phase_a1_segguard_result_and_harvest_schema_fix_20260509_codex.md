# Phase A1 SegGuard Result + Harvest Schema Fix — 2026-05-09

## Verdict

Measured configuration retired. The A1 guarded schedule is not a score-lowering
candidate on either measured axis.

- Lane: `track1_phase_a1_score_gradient`
- Job: `track1_phase_a1_score_gradient_segguard_kl0p5_l1p02_40e_20260509T052414Z_modal`
- Archive bytes: `178279`
- Archive SHA-256: `f220be0350675eb06399e38a39d5849f9d57b6b0ebdac1c8295845d32e6cf6ca`
- Runtime-tree SHA-256, Modal CUDA: `134f9ace728e8c9a32e7e7c0de7f1c0260326939eb00f1194fe159db6775db71`
- Runtime-tree SHA-256, local macOS CPU advisory: `a674905a5770a2b6edd709ad1b4a09a13dec34a530e1102786c2a80659a20981`

## Exact CUDA Result

- Evidence: `[contest-CUDA]`, Modal T4, `600` samples
- Canonical score: `0.22655968711150934`
- PoseNet: `0.00017099`
- SegNet: `0.000665`
- Rate term: `0.11870875`
- Auth eval JSON:
  `experiments/results/track1_phase_a1_score_gradient_segguard_kl0p5_l1p02_40e_20260509T052414Z_modal/harvested_artifacts/eval_work/contest_auth_eval.json`
- Review packet:
  `.omx/research/artifacts/a1_segguard_kl0p5_l1p02_cuda_result_review_20260509_codex.json`

Baseline comparison: current A1 CUDA anchor is `0.2263520234784395`, so this
run regresses by about `+0.000208`.

## macOS CPU Advisory Screen

- Evidence: `[macOS-CPU advisory]`, not contest-CPU
- Canonical score: `0.19309483549345535`
- PoseNet: `0.00003287`
- SegNet: `0.00056256`
- Rate term: `0.11870875`
- Auth eval JSON:
  `experiments/results/a1_segguard_kl0p5_l1p02_macos_cpu_advisory_20260509_codex/work/contest_auth_eval.json`
- Review packet:
  `.omx/research/artifacts/a1_segguard_kl0p5_l1p02_macos_cpu_advisory_review_20260509_codex.json`

Baseline comparison: current A1 Linux contest-CPU anchor is
`0.19284757743677347`, so this local CPU screen regresses by about `+0.000247`.

## Wrapper Bug Fixed

The Modal recover path initially printed `score=None` despite a valid exact
CUDA result. Root cause: `experiments/modal_phase_a1_score_gradient_pr101.py`
only read legacy `score` / `total_score` / `score_components` fields, while
the canonical evaluator emits `canonical_score`,
`score_recomputed_from_components`, `avg_posenet_dist`, `avg_segnet_dist`, and
`score_rate_contribution`.

Fix:

- Added `_eval_metric_summary()` to normalize both canonical and legacy auth
  eval schemas.
- Wired normalized metrics into `_finish_remote()`, remote build manifests,
  recover printing, and local `harvest_summary.json`.
- Added tests covering both canonical and legacy schemas.

Focused tests:

- `.venv/bin/python -m pytest src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py tests/test_modal_phase_a1_recover_paths.py -q`
- Result: `12 passed`

Strict preflight:

- `.venv/bin/python tools/audit_untracked_source_artifacts.py --repo-root . --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json --strict`
- `.venv/bin/python tools/all_lanes_preflight.py`
- Result: `ALL 29 PREFLIGHT CHECKS PASSED`
- Source-custody disposition: added exact `ignore_rebuildable` entries for the
  four harvested A1 runtime source files under `experiments/results/.../submission_dir/`;
  the `experiments/results/` runtime-source prefix baseline remains stable.

## Classification

- Result class: `measured_config_retired`
- Falsification scope: this exact guarded `kl=0.5`, `pixel_l1=0.02`, 40x8,
  `lr=2e-6` schedule only.
- Family falsified: `false`
- Method family retired: `false`

## Follow-up

- Do not relaunch this exact KL/L1 schedule.
- Future A1 work needs true validation selection or early stopping on a
  score-domain/SegNet component proxy, not just stronger auxiliary KL/L1
  regularization.
- The current best A1 exact anchor remains the latent-aligned importpathfix run:
  `178262 B`, `[contest-CPU] 0.19284757743677347`, `[contest-CUDA] 0.2263520234784395`.
