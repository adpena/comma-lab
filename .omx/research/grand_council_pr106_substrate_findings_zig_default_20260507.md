# Grand Council - PR106 Substrate Findings Correction

Date: 2026-05-07

Scope: Canonicalize the PR106 substrate composition signal without carrying
forward a stale draft conclusion.

Evidence:

- `experiments/results/lane_codec_pipeline_full_stack_pr106_20260507T172731Z/composition_matrix_pr106.json`
- `experiments/results/lane_per_tensor_shannon_pr106_20260507T173436Z/per_tensor_shannon.json`
- `experiments/results/pr103_repack_pr106_standalone_20260507/manifest.json`
- `experiments/results/pr103_repack_pr106_standalone_20260507/runtime_closure.json`
- `.omx/research/pr103_pr106_runtime_closure_20260507_codex.md`

## Correction

An earlier draft asserted that PR103 AC regressed on PR106 tensors 0/2/4 by
`+11,498` bytes and that auto-fallback would actively recover those bytes on
the current PR106 repack. That is not supported by the current artifacts.

The current per-tensor Shannon audit reports:

- `total_brotli_bytes = 170096`
- `total_ac_bytes_pr103_indices_only = 161388`
- `ac_regression_total_bytes = 0`
- `ac_regression_tensors = []`

The current PR103-on-PR106 archive manifest reports:

- source archive bytes: `186239`
- candidate archive bytes: `185578`
- archive delta: `-661` bytes
- candidate SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- score claim: `false`

Therefore, per-tensor auto-fallback remains the correct production safety gate,
but it is inactive for this exact PR106 repack audit.

## Durable Findings

1. PR103 AC on PR106 is the immediate rate-only frontier path currently visible
   in the planner. It is byte-different and smaller, but not a score claim.
2. The runtime closure proof binds section lengths, decoder SHA-256, latents
   SHA-256, and fallback routing. It remains blocked on final runtime packet,
   dependency custody, strict compliance, lane claim, and exact CUDA auth eval.
3. Wire-format honesty remains mandatory. Composition experiments that store
   multiple intermediate blobs are not production byte wins unless the final
   shipped archive pays only one charged representation.
4. `auto_select` and `ac_auto_fallback` are safety gates against substrate
   mismatch. They are not score evidence by themselves.

## Next Move

Promote PR103-on-PR106 only after:

1. A self-contained PR103-aware submission runtime packet exists.
2. `brotli` and `constriction` dependency custody is closed for that runtime.
3. Strict submission compliance passes on the exact packet.
4. The lane is claimed.
5. Exact CUDA auth eval lands through
   `archive.zip -> inflate.sh -> upstream/evaluate.py`.

Strategic follow-on remains joint substrate training: reduce the native
description length of the HNeRV/sidechannel system instead of relying only on
post-hoc entropy recodes.
