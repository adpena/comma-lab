# Codex Findings: Z8 full-video surface guard

- Timestamp UTC: 2026-05-31T17:11:38Z
- Scope: Z8 joint P18/P19 coefficient water-fill and relinearized search.
- Authority: `[macOS-CPU advisory]` only; no contest CPU/CUDA score claim,
  promotion, rank/kill, or exact dispatch authority.

## Finding

The operator is right: repair, gradient backprop, water-fill, and quantizer
allocation are mathematically a full-video joint action. A pair-broadcast or
single-frame surface can be useful as a smoke, but it must not masquerade as
the full-video gradient field.

## Fix

- Z8 coefficient materialization now requires full archive pair-grid surface
  coverage by default.
- Both `joint_weight` and `rate_attack_deadzone_mask` must declare enough pair
  rows for the archive, or the materializer/search fails closed with
  `joint_p18_p19_surface_does_not_cover_full_archive_pair_grid`.
- The iterative search manifest records
  `z8_joint_p18_p19_full_video_surface_coverage.v1` for every candidate.
- CLIs include explicit `--allow-broadcast-surface` for exploratory smokes only.

## Full-Video Smoke

Input archive:
`experiments/results/z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_20260530T161526Z/submission/archive/0.bin`

Output manifest:
`.omx/research/z8_joint_p18_p19_full_video_relinearized_search_smoke_20260531T171138Z/candidate/z8_joint_p18_p19_relinearized_search_manifest.json`

Observed local advisory deltas:

- Surface coverage: `required_pair_count=4`, `joint_surface_declared_pair_count=4`,
  `rate_attack_deadzone_mask_declared_pair_count=4`, blocker `null`.
- Z8HPC1 archive bytes: `92,408 -> 7,735` (`archive_rate_ratio=0.08370`).
- Iterations accepted: `2`.
- Cumulative small-receiver distortion:
  `mse=0.001073921`, `mae=0.0239236`, `max_abs_delta=0.204564`.
- Exact blocker preserved:
  `receiver_proof_and_contest_cpu_cuda_eval_not_executed`.

The remaining upstream gap is to generate those full-video surfaces from real
MLX scorer VJP plus PoseNet Mahalanobis/null-subset measurements after each
accepted candidate, rather than synthetic smoke surfaces.
