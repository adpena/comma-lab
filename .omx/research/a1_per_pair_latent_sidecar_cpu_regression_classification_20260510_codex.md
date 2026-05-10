# A1 per-pair latent sidecar CPU regression classification (2026-05-10)

<!-- generated_at: 2026-05-10T13:18:40Z -->
<!-- research_only=true -->
<!-- score_claim=false -->
<!-- no_remote_or_gpu_dispatch=true -->

## Classification

The full-600 `proxy_mse` A1 per-pair latent sidecar packet is **retired for the
measured implementation** after exact `[contest-CPU]` evaluation. This is not a
CUDA score claim and not a proxy claim.

## Candidate custody

- Manifest:
  `experiments/results/a1_sidecar_resumable_codex_20260509T_local/sidecar_manifest.json`
- Classification artifact:
  `experiments/results/a1_sidecar_resumable_codex_20260509T_local/exact_eval_classification.json`
- Classification artifact SHA-256:
  `1fd2dc04fe81bde71ae3dccedf97371fb362eb871581a650e72d5772683b2ba7`
- Archive SHA-256:
  `c7f3d88e1ad23bf8cda987583e702ac57e293b64bc7bfea77902e835d19cea10`
- Archive bytes: `178316`
- Runtime-tree SHA-256:
  `3497c774d94fe202563bccba2af4a5f90925cb8d9b2e982cf4428d0efbea0190`
- Sidecar choice state: full 600/600 coverage, CPU `proxy_mse`, scalar-safe
  pair records.
- No-op proof: old/new sidecar SHA changed and bounded runtime output changed.
- Exact inflate smoke: contest `inflate.sh archive_dir output_dir file_list`
  surface passed before evaluation.

## Eval evidence

- Eval artifact:
  `experiments/results/a1_sidecar_resampled_proxy_mse_20260510T053453Z_cpu_eval_gha/contest_auth_eval.adjudicated.json`
- Axis: `[contest-CPU]` on GitHub Actions Linux x86_64.
- Samples: `600`
- Score from returned components: `0.20962552129271272`
- A1 `[contest-CPU]` baseline: `0.19284757743677347`
- Delta vs baseline: `+0.016777943855939254`
- SegNet distortion: `0.00071063`
- PoseNet distortion: `0.00003932`

## Terminal blocker

`tools/build_a1_per_pair_latent_correction_sidecar.py --classify-eval-result`
now writes a structured `a1_sidecar_exact_eval_classification_v1` record and
updates the manifest to:

- `post_eval_status=measured_contest_cpu_regression_retired`
- `ready_for_exact_eval_dispatch=false`
- `score_claim=false`
- `dispatch_blockers=["measured_implementation_regression_retired"]`

Do not redispatch this exact packet. CUDA remains unproven, and the CPU
regression is sufficient to retire this measured `proxy_mse` implementation.

## Solver Wire-In

- Sensitivity-map contribution: no positive empirical anchor; no update.
- Pareto constraint: this measured `proxy_mse` sidecar point is dominated by A1
  on `[contest-CPU]`.
- Bit-allocator hook: no hook; measured sidecar policy regressed.
- Cathedral autopilot dispatch hook: blocked by terminal classifier.
- Continual-learning posterior update: negative CPU anchor only; do not promote
  as CUDA evidence.
- Probe-disambiguator: N/A. The returned artifact is a same-archive candidate
  with a single measured classification.
