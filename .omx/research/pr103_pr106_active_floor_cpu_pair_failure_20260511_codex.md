# PR103-on-PR106 active floor CPU-pair attempt failure (2026-05-11)

## Summary

The paired `[contest-CPU]` replay for the active internal `[contest-CUDA]`
floor was attempted through the canonical fork GitHub Actions CPU dispatcher.
It failed before scoring because the runtime intentionally requires CUDA during
inflate.

This is a runtime-portability blocker and a negative custody result, not a
score result.

## Custody

- lane id: `pr103_pr106_active_floor_cpu_pair`
- dispatch claim close status: `failed_runtime_cuda_required_on_cpu_axis`
- fork PR: `https://github.com/adpena/comma_video_compression_challenge/pull/28`
- GHA run: `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25646285524`
- submission name: `pr103_pr106_active_floor_cpu_pair_20260511T015517Z`
- archive bytes: `185578`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- downloaded artifact: archive only, SHA-256 matches planned archive
- score claim: `false`

## Failure

The fork workflow reached the `Evaluate` step, installed dependencies, extracted
`archive/0.bin`, and passed the runtime dependency check:

- `brotli=1.2.0`
- `constriction=0.4.2`
- `numpy=2.3.4`
- `torch=2.10.0+cpu`

Inflate then exited:

```text
pr103_pr106_final_runtime inflate requires CUDA for contest-faithful output
```

The guard is in `submissions/pr103_pr106_final_runtime/inflate.py` and prevents
this runtime from producing a CPU-axis output. The failure therefore does not
measure the archive's `[contest-CPU]` score.

## Implication

The current PR103-on-PR106 active CUDA floor remains:

- `[contest-CUDA]` score `0.20898105277982337`
- archive SHA-256
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- runtime tree SHA-256
  `54db9e5ddee85ae7f486fae900ff3907932efb1c8d3062bc264b0e5c7456d8f6`

It remains missing a CPU pair.

## Follow-up implementation

After this failure classification, `submissions/pr103_pr106_final_runtime/inflate.py`
was changed to select CUDA when available and CPU otherwise. This creates a new
runtime tree and does not retroactively update the existing CUDA score evidence.

The correct next action is explicit CUDA/CPU output-parity or drift
classification, followed by fresh paired CPU and CUDA auth evals for the new
dual-device runtime tree. Do not call the failed GHA run a CPU regression, and
do not promote a CPU pair from it.
