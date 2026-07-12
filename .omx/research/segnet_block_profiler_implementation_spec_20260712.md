# Frozen-SegNet block profiler implementation spec

Date: 2026-07-12  
Lane: `lane_frozen_segnet_gradient_replacement_elm_headsolve_20260712`  
Authority: `[macOS-CPU advisory profiling]`; `score_claim=false`; `promotion_eligible=false`.

## Purpose

Turn the already-measured inline pair-0 block-hook experiment into a deterministic, bounded,
committed executable. The output must support the optimality audit's layer-wise forward,
input-backward, and saved-activation attribution without claiming profiler-instrumented wall time
is an uninstrumented throughput benchmark.

## Owned surfaces

- `tools/profile_segnet_blocks.py`
- `src/tac/tests/test_profile_segnet_blocks.py`
- one ignored local JSON receipt under `experiments/results/segnet_block_profile_*`

No trainer, scorer implementation, live-arm result, DSL, or shared-state edit is authorized.

## Required contract

1. Load the canonical frozen `upstream/modules.py::SegNet` and exact safetensor weights; freeze all
   parameters and record their count plus weight SHA-256.
2. Decode one explicitly selected real video pair and follow the canonical last-frame plus
   bilinear-384x512 preprocessing path. Record video path, bytes, SHA-256, pair index, decoded
   frame identity, dtype/range, and model-input shape.
3. Seed Python/NumPy/Torch from one recorded seed, set an explicit Torch thread count, enable
   deterministic algorithms, and record Python/NumPy/Torch/platform/git/tool-source custody.
   Because the worktree is shared and may be dirty, also content-address `upstream/modules.py`,
   `upstream/frame_utils.py`, `pyproject.toml`, and `uv.lock`, and record the imported PyAV,
   segmentation-models-pytorch, timm, and safetensors versions rather than treating HEAD as
   sufficient dependency custody.
4. Attribute only non-overlapping top-level encoder stem/blocks, the post-stem encoder `bn1`,
   decoder blocks, and segmentation head. Measure one or more explicit warmups/samples. Report raw
   per-sample forward/backward times, medians, paired-sample shares, selected timing coverage, and
   unattributed time/saved storage, plus the hook/profiler overhead warning. Never label a ratio of
   independently aggregated medians as paired-sample coverage.
5. Use `torch.autograd.graph.saved_tensors_hooks` to record logical saved bytes and a declared,
   deterministic owner-attribution rule. Distinguish summed logical bytes from unique storage;
   never call them peak RSS.
6. Write JSON atomically to a durable repo/SSD path and refuse `/tmp`/transient output. The receipt
   must preserve the exact rerun command and every parsed config value.
7. Tests use a tiny CPU network/synthetic input to validate block selection, accounting schema,
   deterministic aggregation, and atomic receipt behavior; they must not load the real 37 MB model.

## Verdict scope

This executable makes the block table reproducible. It does not establish Metal/CUDA ordering,
training-loop speedup, d_seg movement, or score movement. The canonical uninstrumented grouped
backward measurements remain the throughput authority for the Apple training path.
