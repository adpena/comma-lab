# FEC8 Static Second-Order Exact Auth Eval Landed

Date: 2026-05-26
Agent: codex
Lane: `lane_pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526`
Archive SHA-256: `b44da5d54d34ce094c8fca7a37172b9ea6546d8a56eb960afa86252f56d11844`
Archive bytes: 178507

## Summary

The FEC8 static second-order Markov selector codec was wired into the PR101
frame-exploit packet builder and built from a clean pushed worktree. The clean
archive is byte-identical to the pre-commit smoke artifact and is 10 bytes
smaller than the FEC6 fixed-Huffman K16 packet while preserving the same source
payload and selector code sequence.

Exact Modal paired auth eval was dispatched and recovered on both contest axes.
The result is not promotable.

## Exact Results

- `[contest-CPU]`: `0.19204465829157438`
  - `avg_posenet_dist=0.00002943`
  - `avg_segnet_dist=0.00056029`
  - Modal call: `fc-01KSJXFSBP4EAR56WARK8D2EEJ`
  - Result: `experiments/results/modal_auth_eval_cpu/lane_pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526_paired_modal_auth_20260526T195114Z_cpu/modal_cpu_auth_eval_result.json`
- `[contest-CUDA]`: `0.22620336310396671`
  - `avg_posenet_dist=0.00016846`
  - `avg_segnet_dist=0.00066299`
  - Modal call: `fc-01KSJXF4SNCPAHR6J6EATE2HE9`
  - Result: `experiments/results/modal_auth_eval/lane_pr101_frame_exploit_selector_fec8_static_second_order_k16_20260526_paired_modal_auth_20260526T195114Z_cuda/modal_cuda_auth_eval_result.json`

## Verdict

`promotion_eligible=false`. Current best anchors remain:

- `[contest-CPU]`: `0.1920282830`
- `[contest-CUDA]`: `0.2053300290`

The CPU result is a small regression versus the CPU frontier, and the CUDA
result is a large regression. Treat the FEC8 static second-order codec as a
valid byte-saving materializer only under an exact-score gate. The materializer
is useful for selector-stream entropy modeling and future chained packet work,
but this PR101 selector perturbation payload is not exact-frontier safe.

## Durable Signal

- The byte-side result validates the P11/P13 selector-stream entropy model:
  static second-order context coding can remove 10 archive bytes versus FEC6 on
  this fixed K16 selector sequence.
- The scorer-side result falsifies the local/proxy promotion hypothesis for
  this exact PR101 packet. Do not dispatch siblings from this family without a
  stronger scorer-response gate, especially a CUDA-axis guard.
- Future action should compose selector entropy savings with a selector policy
  learned against exact scorer response, not just local advisory deltas.
