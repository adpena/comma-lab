# PR85 Residual Sufficient-Program Profile

- planning_only: true
- score_claim: false
- dispatch_performed: false
- charged_qma9_mask_bytes: 159011
- charged_qma9_bits_per_token: 0.010783624
- token_source_sha256: `c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a`
- render_order_sha256: `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`

## Residual Programs

| rank | predictor | zero frac | nonzero frac | best lb bytes | est saved bytes | rate-score delta lb | row spans |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `left_zero_border` | 0.995599 | 0.004401 | 721370.025 | -562359.025 | 0.374451792 | 222328 |
| 2 | `up_zero_border` | 0.986566 | 0.013434 | 787567.456 | -628556.456 | 0.418529944 | 67551 |
| 3 | `time_prev_then_left_border` | 0.986295 | 0.013705 | 807767.491 | -648756.491 | 0.431980318 | 73339 |
| 4 | `time_prev_zero_first` | 0.986269 | 0.013731 | 808820.771 | -649809.771 | 0.432681654 | 73394 |
| 5 | `absolute_zero` | 0.232344 | 0.767656 | 13622761.632 | -13463750.632 | 8.964958901 | 222328 |

## Recommendations

- `do_not_dispatch_residual_program_without_model_or_atom_refinement`: best deterministic residual sufficient-statistic lower bound comes from left_zero_border
- `if_training_on_fast_gpu_use_this_profile_as_lossless_target_density`: top changed frames, row spans, and residual nonzero maps provide a non-arbitrary curriculum/atom density field for learned mask coders

These are local sufficient-statistic bounds only. They can select a coder or training curriculum, but cannot claim score until a byte-closed archive passes exact CUDA auth eval.
