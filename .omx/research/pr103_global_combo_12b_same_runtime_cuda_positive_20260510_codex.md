# PR103 global-combo -12B same-runtime CUDA result (2026-05-10)

## Verdict

The PR103 global-combo histogram packet is a real same-runtime CUDA rate win.
This supersedes the earlier broad interpretation of the greedy `-8B` packet as
a method negative: that conclusion compared against an older replay rather than
a same-runtime source baseline.

## Candidate

- archive:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/packet/archive.zip`
- archive SHA-256:
  `578c8f4e86eafc9dc04eefe61cc0e7f3f3f43e134ef4447cf9ef26fd23a23551`
- archive bytes: `178211`
- packet runtime tree SHA-256:
  `bf43663559e88b89f1bc0a1fa14b5093b7195da64f5aa7ed1cac696cb60caa02`
- full CPU same-runtime frame parity:
  `true` over `3,662,409,600` rendered bytes, output SHA-256
  `074f834f14ba4611f9358bb0a3f8e729bb43e4ea673be23e2acf85e7448dd1e5`

## Same-runtime Modal T4 pair

| packet | archive bytes | score | seg | pose | runtime tree |
|---|---:|---:|---:|---:|---|
| PR103 source | `178223` | `0.22777817708207615` | `0.00067635` | `0.00017199` | `ea25380a6eee64b7f57a30a5e9c745fa6bd8867c728f92a293945cfd6dce5d42` |
| global-combo candidate | `178211` | `0.22777017708207614` | `0.00067635` | `0.00017199` | `87804ddc539f3c122f5c8ed886cbbef3d61956c0c602a09304fc7a9394297b45` |

Delta:

- bytes: `-12`
- score: `-0.000008000000000008`
- component deltas: seg `0`, pose `0`
- classification: exact same-runtime CUDA positive, rate-only

## Artifacts

- candidate eval:
  `experiments/results/modal_auth_eval/pr103_global_combo_12b_exact_cuda_modal_20260510T2257Z/`
- source baseline eval:
  `experiments/results/modal_auth_eval/pr103_source_same_runtime_cuda_baseline_modal_20260510T2300Z/`
- candidate frame parity:
  `.omx/research/pr103_arithmetic_transform_plans_20260510_codex/global_combo_candidate/frame_parity_probe_full_cpu.json`

## Implication

For PR103 histogram transforms, compare packets against a same-runtime source
baseline before declaring tiny CUDA deltas positive or negative. The earlier
old-replay comparison was not strong enough. The next score-lowering step is to
generalize the global-combo optimizer beyond q8 histogram sideband changes
while keeping full frame parity and same-runtime source/candidate CUDA pairs.
