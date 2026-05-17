# L5 v2 TT5L Paired Anchor Pair Canonicalization - 2026-05-17

## Scope

Adversarial review found a no-signal-loss bug in the TT5L paired exact-eval
surface: the historical per-axis CPU evidence row still carried
`contest_cuda_pending_for_internal_promotion` even after the CUDA axis had been
harvested and reviewed. A consumer that read only the CPU row could incorrectly
conclude the paired anchor was incomplete.

## Fix

The paired state is now owned by the machine-readable pair artifact:

- `.omx/research/l5_v2_tt5l_paired_exact_anchor_pair_20260516_codex.json`
- `.omx/research/l5_v2_tt5l_paired_exact_contest_cpu_inflated_outputs_manifest_summary_20260516_codex.json`
- `.omx/research/l5_v2_tt5l_paired_exact_contest_cuda_inflated_outputs_manifest_summary_20260516_codex.json`

`src/tac/optimization/l5_staircase_v2.py` discovers that artifact as the
canonical `exact_anchor_or_diagnostic_pair` evidence source. The per-axis rows
remain preserved as historical evidence, but they no longer own pair-level
truth.

## Classification

This is not a score claim and not promotion evidence. The paired exact CPU/CUDA
anchor is a non-promotional measured-config failure: both axes were harvested
for the same archive/runtime-content tree, but the TT5L side-info stream was
all zero and the score collapsed. The artifact exists to preserve the exact
paired signal and prevent stale-axis ambiguity.

## Regression Coverage

`test_l5_v2_dispatch_readiness_consumes_pair_anchor_artifact_over_stale_axis_rows`
asserts that a stale CPU-axis `contest_cuda_pending_for_internal_promotion`
remnant cannot override the canonical paired artifact.
