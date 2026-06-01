# Codex Findings - HPRC Baseline MLX Response Reuse

UTC: 2026-06-01T01:49:25Z
Author: Codex
Authority: MLX-local advisory tooling; no contest CPU/CUDA score claim

## Finding

The HPRC profile loop was still recomputing the unchanged baseline scorer
response for every residual sweep.  That made the last pair-scoped full-video
run spend most wall time in CPU MLX response generation even after direct HPRC
cache materialization removed raw inflate as the bottleneck.

This landing adds a provenance-checked baseline reuse path:

- source profile schema must match `hprc_mlx_component_neutralization_profile.v1`
- baseline `hprc_0bin_sha256` must match the current candidate baseline
- reference cache path must match
- requested `max_pairs` must match
- prior baseline response and component arrays must exist
- reused baseline response is copied into the new output tree
- archive byte fields are retargeted to the current baseline archive bytes
- false-authority fields remain false

## Smoke

Command shape:

`tools/profile_hprc_mlx_component_neutralization.py --reuse-baseline-profile ... --sections --max-pairs 600`

Evidence directory:

`.omx/research/hprc_baseline_reuse_smoke_20260601T014925Z_codex/`

Result:

- variants: `1` baseline only
- elapsed seconds: `0.42223525047302246`
- reused source cache: `/Volumes/VertigoDataTier/pact/hprc_pair_scoped_residual_full600_20260601T011601Z/mlx_caches/baseline`
- score claim: `false`
- promotion eligible: `false`

This converts the next HPRC residual sweeps from baseline plus variants into
variant-only scorer work whenever the baseline identity is unchanged.

## Next Action

Wire this option into bounded HPRC residual sweeps by default when a matching
baseline profile exists, then attack the remaining variant scorer-response hot
path with component-cache reuse, pair-window incremental scoring, and native or
vectorized kernels if profiling still shows one-core CPU saturation.
