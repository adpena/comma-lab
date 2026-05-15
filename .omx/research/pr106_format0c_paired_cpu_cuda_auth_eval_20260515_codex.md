# PR106 format0C paired CPU/CUDA auth eval - 2026-05-15

## Custody

- Archive:
  `experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip`
- Archive bytes: `186327`
- Archive SHA-256:
  `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- Runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh`
- Pair group: `pair_pr106_format0c_exact_radix_20260515`
- Dispatch plan:
  `experiments/results/pr106_format0c_exact_radix_paired_20260515T0918Z_plan.json`

## Results

| Axis | Modal call | Output dir | Score | SegNet | PoseNet |
| --- | --- | --- | ---: | ---: | ---: |
| `[contest-CUDA]` | `fc-01KRNET3305N9S3ZSCJKF9M9EK` | `experiments/results/modal_auth_eval/pr106_format0c_exact_radix_paired_20260515T0918Z_cuda` | `0.2063163866158099` | `0.0006426` | `0.00003236` |
| `[contest-CPU]` | `fc-01KRNETMJ026XDDD6KC28503VR` | `experiments/results/modal_auth_eval_cpu/pr106_format0c_exact_radix_paired_20260515T0918Z_cpu` | `0.22776488386973992` | `0.00063198` | `0.00016402` |

Classification: legitimate paired exact-eval result, not frontier. The CPU
axis is worse than CUDA for this archive/runtime pair, driven mainly by PoseNet
distance. Do not infer either axis from the other.

## Engineering Findings

- Before this run, there was no exact Linux x86 `[contest-CPU]` artifact for
  this PR106 format0C archive SHA.
- The first CPU spawn attempt exposed a Modal source packaging race: generated
  `__pycache__` bytecode under mounted `src/` could change during Modal image
  build and abort before remote spawn. Fixed in commit `b32c6685f` by applying
  a generated-file mount ignore helper to Modal auth-eval mounts.
- Commit `b32c6685f` also makes Modal auth eval paired-by-default. Direct
  CPU/CUDA wrappers now require a shared `pair_group_id` or an explicit
  `single_axis_waiver_reason`; the default score-bearing entry point is
  `tools/dispatch_modal_paired_auth_eval.py`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_build_selector_cuda_transfer_calibration.py -q`
  -> `45 passed`
- `.venv/bin/python -m ruff check src/tac/deploy/modal/auth_eval.py src/tac/deploy/modal/mount_ignore.py experiments/modal_auth_eval.py experiments/modal_auth_eval_cpu.py tools/dispatch_modal_paired_auth_eval.py src/tac/tests/test_modal_auth_eval.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py tools/build_selector_cuda_transfer_calibration.py src/tac/tests/test_build_selector_cuda_transfer_calibration.py`
  -> clean
