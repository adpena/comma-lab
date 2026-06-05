# SNeRV Skip-High Mode Comparison

Schema: `snerv_skip_high_mode_comparison.v1`
Verdict: `NO_CURRENT_SKIP_HIGH_MODE_READY_FOR_EXACT_EVAL`
Axis: `[macOS-CPU/MLX planning:false-authority]`

## Binary Profiles

| label | codec | archive bytes | stored shape | stored raw bytes | under cap | scalar collapse |
|---|---:|---:|---|---:|---:|---:|
| scalar_mean | scalar_mean_float64 | 91445 | `[1, 1, 1, 1]` | 8 | True | True |
| shared_mean | shared_mean_float64 | 436084 | `[1, 3, 192, 256]` | 1179648 | False | False |

## Prefilter Profiles

| label | score | Seg term | Pose term | local replay | OOD |
|---|---:|---:|---:|---:|---:|
| scalar_epoch003199 | 90.8645 | 50.4825 | 40.3197 | False | True |

## Crux

- rate-admissible scalar skip-high is cheap (91445 bytes) but collapses stored skip-high to 8 raw bytes.
- non-scalar skip-high preserves more value-domain structure but the best attached profile is 436084 bytes (258084 vs hard ceiling).
- attached scorer prefilter evidence is out of distribution; do not promote or exact-dispatch from these local scores.

## Next Actions

- block Modal/exact auth eval until a byte-closed candidate also passes local scorer-input and cache-quality gates
- run the next SNeRV local skip-high smoke on a non-scalar storage mode only after current MLX claims clear
- record frame-1 SegNet, two-frame PoseNet, archive bytes, and skip-high storage shape for every mode
- do not use scalar_mean as the promotion path unless a receiver value-domain xray disproves the collapse mechanism
- treat current local prefilter rows as acquisition/falsification evidence only

## Blockers

- `snerv_skip_high_mode_comparison_false_authority`
- `no_skip_high_mode_with_both_byte_cap_and_non_scalar_storage`
- `skip_high_prefilter_scorer_input_out_of_distribution`
- `no_skip_high_prefilter_profile_admissible_for_local_replay`
