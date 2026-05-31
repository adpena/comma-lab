# Codex findings: Z8 P19 proxy and decode-benchmark authority

UTC: 2026-05-31T19:49:12Z
Agent: codex

## Verdict

The full-video Z8 MLX VJP lane may carry exact gradient-reduction authority
after all pair shards reduce, but it must not carry budget-spend or optimizer
authority while the PoseNet term is the scalar first-6 pose-MSE VJP proxy.

That proxy is useful ranking signal, but it is not the promised P19
per-axis PoseNet Jacobian / Mahalanobis null-subset surface. Treating it as a
true null detector can spend rate into pose-sensitive atoms.

## Changes landed

- `full_video_vjp_acquisition.py` now tags MLX shards with
  `pose_surface_kind=scalar_first6_pose_mse_vjp_proxy_v1`,
  `pose_jacobian_abs_is_true_jacobian=false`, and blocker
  `p19_pose_surface_not_true_per_axis_jacobian`.
- Bundle assembly keeps `gradient_reduction_authority=true` for exact
  full-video reductions, but sets `budget_spend_authority=false`,
  `optimizer_update_authority=false`, and
  `implicit_allocator_authority=false` unless every shard declares a true
  `per_axis_posenet_jacobian_mahalanobis_v1` surface.
- The Z8 wavelet decode benchmark is now explicitly parser-only:
  `z8_wavelet_blob_parse_benchmark.v1`,
  `wavelet_blob_parse_seconds_*`, and blocker
  `full_contest_inflate_sh_runtime_not_measured`.

## Remaining implementation target

Build the true P19 surface: six-axis PoseNet VJP/JVP or finite-difference
null-subset probe with contest inverse-variance weighting. Only that surface
may promote Z8 P18/P19 budget-spend authority.
