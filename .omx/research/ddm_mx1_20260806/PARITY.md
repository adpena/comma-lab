# ddm_mx1 parity receipt

Axis labels:
- Lifted torch reference: `[macOS-CPU advisory torch upstream SegNet]`.
- MLX port/parity: `[torch-CPU reference vs MLX host parity]`, locally blocked by MLX runtime.
- No result here is a contest score or n600 scorer row.

## Verdicts First

| gate | verdict | measured values |
| --- | --- | --- |
| lifted PR130 torch CPU forward/curriculum behavior | PASS | focused tests `4 passed`; output shape `(2,3,12,16)`, bounded `[0,255]`, deterministic same-seed output equal |
| torch CPU -> MLX-CPU raw frame parity | BLOCKED-LOCAL-RUNTIME | max-abs not measured; local MLX probe fails before array execution with `[metal::load_device] No Metal device available` |
| torch CPU -> MLX-CPU scorer argmax parity | BLOCKED-LOCAL-RUNTIME | argmax diff count not measured for same reason; MAIN must report exact count on real cache pairs because the default MLX conv adapter has a known #855 hazard |
| torch CPU -> MLX-CPU one-step loss parity | BLOCKED-LOCAL-RUNTIME | loss delta not measured for same reason |
| torch CPU -> MLX-CPU gradient parity | NOT-CLAIMED | no gradient-parity result exists locally; any later training claim must add it or remain scoped to forward/loss research-signal |
| MLX-CPU n=2 smoke | BLOCKED-LOCAL-RUNTIME | `--mode mlx-train` exits fail-closed with rc 2 and writes blocker JSON |
| lifted torch CPU n=2 smoke on real OUR label payloads | PASS-SMOKE-ONLY | loss `0.004268820863217115 -> 0.004265481140464544`; batch d_seg stayed `0.0042317709885537624`; `4.949445009231567` s/step; resume load OK |

## Evidence

Local MLX probe:

```json
{"status":"blocked","device_request":"cpu","error_type":"MlxUnavailableError","error":"[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible."}
```

Torch smoke command:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode torch-smoke --pairs 2 --steps 2 --scorer upstream --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/torch_smoke_n2 --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/torch_smoke_n2/result.json
```

Torch smoke artifacts:

| artifact | bytes | sha256 |
| --- | ---: | --- |
| `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/torch_smoke_n2/result.json` | 7690 | `d3384be3b55e093c5af18bf467e8c0912fcc61431ad0a63b46393852dd2f8cca` |
| `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/torch_smoke_n2/torch_smoke.latest.pt` | 842713 | `7a8a0bf7406daa55cec93192bd82f2ddf55fa6749388661781603537472f9cda` |

MLX blocked artifacts:

| artifact | bytes | sha256 |
| --- | ---: | --- |
| `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/mlx_parity_probe/result.json` | 6517 | `682cb5a2a35064fc9dd9a686a77270a52d6cc100d9b4d2fcd901ef4b9fa7f0db` |
| `/Volumes/VertigoDataTier/pact/ddm_mx1_20260806/mlx_block_probe/result.json` | 6527 | `16f31e4d6c03296f00656db42a46dc7fa9d0d1feead6c1329c30ba7bf6573675` |

## MAIN Amendment Boundaries

The MAIN amendment in `.omx/research/ddm_mx1_20260806/CHARTER_AMENDMENT_MAIN.md` is honored as follows:

- Parity PASS is not claimed locally.
- The implemented `mlx-parity` mode uses real built label caches, not synthetic tensors.
- The implemented `mlx-parity` JSON reports token/scorer batch shapes, the MLX SegNet adapter name, `seg_argmax_equal`, and `seg_argmax_diff_count`.
- Gradient parity is not claimed by this mode; any later training result must either add one real-input gradient parity receipt or remain explicitly scoped to forward/loss research-signal.

## Source Closure

PR130 source root: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo`

Source HEAD: `2f94596bb0136d342254022a5c9584756eae0468`

| lifted file | source sha256 |
| --- | --- |
| `code/semantic_renderer_oracle.py` | `2bf3a6a8621334723fec1c3e596665d1f049a55f311c5348dd5a4c588873f25b` |
| `code/train_semantic_full.py` | `2d7a3575e422dc2b5823b97e52101ad5632e5fa04a98e0af8a83d85b7c2176b8` |
| `code/train_semantic_quantized.py` | `4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d` |
| `code/evaluate_semantic_quantization.py` | `5bbd2136174bfa2c99219d73f45103d4293f60e3f4eced5ee188e38053923962` |

Mechanism anchors checked:
- Architecture/token conditioning: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:86-168`.
- Curriculum loss and exact R/uint8 path: `src/tac/pr130_lift/lifted/semantic_renderer_oracle.py:178-202`.
- QAT and quantized exact-path render: `src/tac/pr130_lift/lifted/train_semantic_quantized.py:37-71`.
- Full/QAT train loops, scheduler, and selection: `src/tac/pr130_lift/lifted/train_semantic_full.py:98-180`, `src/tac/pr130_lift/lifted/train_semantic_quantized.py:234-260`.
- Existing MLX scorer adapter and exact-R substrate: `src/tac/local_acceleration/mlx_scorer_adapters.py:1468-1471`, `src/tac/local_acceleration/pr95_hnerv_mlx_training.py:126-155`.

## Verification

Commands passed:

```bash
.venv/bin/python -m py_compile src/tac/pr130_lift/mlx_semantic_renderer.py experiments/ddm_mx1_pr130_semantic_renderer.py src/tac/pr130_lift/lifted/semantic_renderer_oracle.py src/tac/pr130_lift/lifted/train_semantic_full.py src/tac/pr130_lift/lifted/train_semantic_quantized.py src/tac/pr130_lift/lifted/evaluate_semantic_quantization.py
.venv/bin/python -m pytest src/tac/pr130_lift/tests/test_mx1_pr130_lift.py -q
```

Test output: `4 passed in 0.67s`. The MLX package still emits the same atexit no-device error after tests; this is part of the local MLX blocker, not a parity pass.
