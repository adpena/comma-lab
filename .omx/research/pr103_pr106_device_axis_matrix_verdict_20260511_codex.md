# PR103/PR106 device-axis matrix verdict (2026-05-11)

Status: diagnostic, non-promotional.  
Primary artifact: `.omx/research/artifacts/pr103_pr106_device_axis_matrix_analysis_20260511_codex/analysis.json`

## Inputs

- Archive: `experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/archive.zip`
- Archive SHA-256: `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- Current runtime: `submissions/pr103_pr106_final_runtime` at commit `43a14ef1`
- Matrix planner: `.omx/research/artifacts/pr103_pr106_device_axis_matrix_plan_20260511_codex/plan.json`

## Results

| Label | Axis | Scorer | Inflate policy | Score | Raw aggregate SHA prefix | Runtime SHA prefix |
| --- | --- | --- | --- | ---: | --- | --- |
| `cuda_auto_old` | `contest_cuda` | `cuda` | `auto` | `0.20898305277982338` | `99141b32678d` | `f2ebe56a408a` |
| `cpu_auto_old` | `contest_cpu` | `cpu` | `auto` | `0.22966566346263317` | `1e5fdfa06090` | `f2ebe56a408a` |
| `cuda_forced_cpu_inflate` | `diagnostic_cuda` | `cuda` | `cpu` | `0.20898205277982337` | `dfd6c0199456` | `f83830a81b2e` |
| `cuda_auto_current` | `contest_cuda` | `cuda` | `auto` | `0.20898305277982338` | `99141b32678d` | `f83830a81b2e` |
| `cpu_scorer_cuda_inflate` | `diagnostic_cpu` | `cpu` | `cuda` | `0.2296698981058085` | `99141b32678d` | `f83830a81b2e` |

## Verdict

1. The current runtime patch did not perturb default CUDA output: `cuda_auto_old` and `cuda_auto_current` have identical score and identical raw aggregate SHA despite different runtime tree SHA.
2. Forced CPU inflate on the current CUDA host is real but tiny on the CUDA scorer: raw aggregate differs from CUDA-auto, while score improves by only `-0.000001000000000001`.
3. The missing quadrant is now closed. `cpu_scorer_cuda_inflate` has the same raw aggregate SHA and same runtime tree SHA as `cuda_auto_current`, but scores `+0.020686845325985137` worse because the scorer axis is CPU. This localizes the large PR103-on-PR106 gap primarily to evaluator/scorer math, not inflate output bytes.
4. The large pure CPU-auto gap remains per-submission, not universal: `cpu_auto_old` is `+0.0206826106828098` worse than `cuda_auto_current`; its raw output differs, but the CPU-scorer/CUDA-inflate quadrant proves identical raw bytes still retain the CPU-score basin.
5. The only inflate-device movement observed here is tiny on the CUDA scorer (`-0.000001000000000001`) and non-promotable because it requires a forced diagnostic inflate policy.

## Guardrails

- `cuda_forced_cpu_inflate` is diagnostic only: `score_axis=diagnostic_cuda`, `score_claim=false`, `promotion_eligible=false`, diagnostic blocker `inflate_device_policy_cpu`.
- `cpu_scorer_cuda_inflate` is diagnostic only: `score_axis=diagnostic_cpu`, `score_claim=false`, `promotion_eligible=false`, diagnostic blocker `inflate_device_policy_cuda`.
- Do not infer CPU or CUDA is globally better. This matrix only proves the PR103-on-PR106 archive/runtime behavior.
- Any score-lowering use of forced inflate-device policy requires a contest-compliant default runtime path or a byte-changing packet whose default inflate path consumes the intended policy without external environment overrides.
