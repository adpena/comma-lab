# PR95 Full-600 Section-Value Baseline-Reuse Findings

**Author:** Codex  
**UTC:** 2026-06-02T04:44:10Z  
**Axis:** `[macOS-MLX research-signal]`; false-authority, not exact CPU/CUDA  
**Archive SHA-256:** `09eacafaba78d98bca869a0f224d0edfa971af5d7c9293b8e7ce72b7f58d36a0`

## What Landed

`tools/profile_pr95_hnerv_mlx_section_value.py` now supports an explicit external baseline reuse path:

- `--baseline-cache-dir`
- `--baseline-cache-report`
- `--baseline-mlx-response`

Reuse is accepted only when archive bytes, archive SHA-256, cache manifest, pair coverage, scorer batch shape, and false-authority flags match the current baseline variant. The reused response is copied into the new profile output so downstream reports still have a local provenance path.

## Full-Video Evidence

Profile:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_full600_section_value_with_baseline_reuse_20260602T043459Z/hprc_mlx_component_neutralization_profile.json`

Bounded-runner plan:

`/Volumes/VertigoDataTier/pact/hprc_section_value_profiles/pr95_hnerv_full600_section_value_with_baseline_reuse_20260602T043459Z/hprc_spine_bounded_runner_plan.json`

Baseline reuse status:

`accepted_archive_hash_and_pair_shape_match`

Baseline full-600 MLX advisory score:

`0.1989702560519192`

Measured section neutralizations:

| Section | Bytes Removed | Delta Nonrate | Delta Rate | Delta Total | Verdict |
|---|---:|---:|---:|---:|---|
| `decoder_qw` | 161,907 | +90.5216488783 | -0.1078072255 | +90.4138416528 | protect |
| `latents_rc` | 15,850 | +4.6640917128 | -0.0105538644 | +4.6535378484 | protect |

Both decoder weights and latents have measured score value per removed KiB far above the fixed rate price. This artifact is not a candidate for section cutting; the right next action is byte recoding or architecture training, not deleting payload.

## Runner State

The bounded runner consumed the full-video profile plus receiver proof. It selects:

`shrink_or_recode_compact_base_under_ceiling`

Current hard blocker:

- `contest_cpu_cuda_exact_eval_not_executed`
- `no_full_coverage_candidate_under_any_hard_ceiling`

The candidate is receiver-proven, full coverage, but `178,357` bytes: `357` bytes above the strict `178,000` ceiling. The plan correctly queues byte sweep/recode rather than section cuts.

## Discipline

No exact score claim is made. MLX remains advisory. The full profile output is large but lives on SSD and retains cache/report/provenance artifacts for deterministic replay.
