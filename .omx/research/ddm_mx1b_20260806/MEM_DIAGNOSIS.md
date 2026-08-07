# ddm_mx1b Memory Diagnosis

borrowed_substrate_accounting: no PR130 renderer mechanism changed. This is load-phase engineering and launch safety.

## Named Allocators

1. Full-600 cache materialization before subsetting: CONFIRMED in the pre-fix baseline blob at `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 382-385. The old path loaded `input_tokens_all` and `target_tokens_all` with `.long()` over the whole 600-pair cache, then sliced `pair_ids`.

2. Torch-load plus selected NumPy/MLX double residency: CONFIRMED as the main CPU-side load class. Current fix lives at `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 253-314 and `run_mlx_train` lines 647-664. It still must deserialize the monolithic `.pt`, but only the selected rows survive past the helper; the full payload is deleted before MLX setup.

3. Camera-resolution expansion of all pairs: NOT FOUND in the inspected mlx-train path. The current forward uses selected `conditioning` and `pair_idx` only at lines 703-713 and 754-756. This absence is bounded to `experiments/ddm_mx1_pr130_semantic_renderer.py`; it is not a global claim about other trainers.

4. Lazy MLX graph accumulation during setup: PLAUSIBLE/UNMEASURED on this sandbox, fixed structurally. The pre-fix baseline had no setup barriers before the first train-step eval. Current barriers are at lines 678, 696, 706-713, 723, and the train step sync at line 775.

5. Upstream scorer eager load: CONFIRMED but necessary for this train path. Current source loads CPU SegNet at line 719, converts to MLX at 721-723, then deletes the torch scorer at 724-727. MAIN must measure the Metal allocator delta for this stage.

6. MLX allocator/cache pressure: UNKNOWN locally. The sandbox fails at `require_mlx(device="cpu")` with no Metal device before any MLX active/cache sample can exist. MAIN Metal mem-probe must supply the authority receipt.

## CPU-Side Receipt Summary

Measured receipt path: `.omx/research/ddm_mx1b_20260806/mem_probe_cpu/mem_probe_receipt.json`
Receipt SHA-256: `79fc2fb2ee2a430caf48695d73ed4497771d4c347a4b6bf4d02200cb336a8ae3`

| allocator stage | measured local effect |
|---|---|
| init checkpoint load | RSS +0.001129 GiB |
| selected input cache clone from 943,720,090 B cache | RSS +0.974930 GiB from start |
| selected target cache clone from 943,720,076 B cache | RSS +1.021820 GiB from start |
| selected NumPy copy plus torch full-cache free | RSS +1.068787 GiB from start |
| MLX require/model/scorer/steps | blocked by no Metal device |

Interpretation: the local CPU-side fix bounds retained RSS around 1.27 GiB after the selected clone/free path. It does not prove the Metal load-phase is safe; the original incident's 65 GiB to 0 GiB swing remains a MAIN-only measurement until a passed `mem_probe_receipt.json` includes MLX active/cache/peak and the required final mem-probe train-step sample.

## Code Pointers

- Memory telemetry: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 81-192.
- Selected cache clone/free helper: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 253-314.
- Memory budget derivation and MLX limit calls: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 548-608.
- mlx-train setup order and barriers: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 642-727.
- Training-step eval barrier: `experiments/ddm_mx1_pr130_semantic_renderer.py` line 775.
- mem-probe receipt builder: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 866-954.
- launch ticket receipt/scheduling fields: `experiments/ddm_mx1_pr130_semantic_renderer.py` lines 1001-1037.
- focused tests: `experiments/tests/test_ddm_mx1_memory_probe.py` lines 52-123.

## Boundary

This diagnosis did not run a full MLX training step locally, did not run any scorer job, did not produce a d_seg/d_pose verdict, did not touch `upstream/`, and did not move the pointer.
