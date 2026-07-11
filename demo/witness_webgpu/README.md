# Witness · WebGPU + WebNN demo (Kernel B, client-side)

A **real forward pass of a trained level-set witness** running in the browser: the covariant
class partition `argmax_k φ_k` (Kernel B — `witness_forward`) ported to a WebGPU compute shader,
with an optional WebNN trunk-matmul path. Scrub the drive; the partition re-solves live on the GPU.

**AUTHORITY:** `[WebGPU/WebNN demo — NON-AUTHORITY]`. WebGPU/WebNN are new compute substrates in
the same class as MPS/MLX — the **numpy-fp32 reference is the bit-identical authority**; a browser
"d_seg"/partition is a **visualization, never a contest score**. Only `upstream/evaluate.py` on
byte-closed `archive.zip` bytes is a score. The contest-CPU pointer **0.19108282 is UNMOVED** — this
is reach/showcase, not a pointer-mover.

## Files
| file | what |
|---|---|
| `export_fixture.py` | loads a live EMA-best checkpoint, reconstructs the curvelet + self-orient front-end with the repo's own functions, computes the numpy-fp32 reference partition, writes `fixture.json` / `feats.bin` / `reference.bin`. |
| `witness_forward.wgsl` | Kernel B as a WebGPU compute shader — faithful port of `levelset_sdf_argmax_mlx` (in_proj → FiLM hidden×N → out_sdf → argmax). |
| `parity_shader_model.py` | numpy fp32 shader-model vs the numpy-fp32 authority → per-frame pixel-match parity (advisory). |
| `index.html` | the client-side demo (WebGPU forward + WebNN trunk + live parity badge). |

## Regenerate the fixture (from a real checkpoint)
```bash
.venv/bin/python demo/witness_webgpu/export_fixture.py \
  --ckpt experiments/results/levelset_n600_witness_20260705T015247Z/levelset_witness_ema_BEST.npz \
  --grid-h 96 --grid-w 128 --frames 0 199 399 599 799 999
```

## Verify parity (WGSL algorithm vs numpy-fp32 authority)
```bash
.venv/bin/python demo/witness_webgpu/parity_shader_model.py
# -> overall_pixel_match 1.0, verdict PASS (6/6 frames, 0 mismatched px)
```
The parity contract is **WGSL forward vs numpy-fp32 forward on the identical shipped `(feats, weights)`**,
so the port-fidelity claim is exact. A WebGPU browser reproduces the shader-model by construction.

## Run the demo
```bash
cd demo/witness_webgpu && python -m http.server 8000
# open http://localhost:8000/index.html in a WebGPU browser (Chrome/Edge 113+, Safari TP)
```
No network, no CDN, no build step. WebNN (`navigator.ml`) is used when present for the trunk
projection `feats @ Wᵢₙᵀ + b` and checked against the CPU reference; otherwise WebGPU covers the
whole forward.

## Honest scope
- The MLP forward (features → FiLM → argmax) is a faithful port; parity is 100% on the shipped inputs.
- The **front-end features** (`feats.bin`) are computed by the numpy-fp32 authority and shipped — this
  mirrors report 010's **W3** (export the trunk; the geometric/scipy front-end stays on the authority /
  intrinsic lane). The demo is a real witness forward on real trained weights, not the identical live
  self-orient feats to the last bit (the cheap-partition fixed point is reconstructed from documented
  byte-closeable helpers) — disclosed, and immaterial to the WGSL-vs-numpy parity claim.
- Headless WebGPU is not available in the dev environment, so browser execution is
  **verified-by-construction + the parity model**, driven by the operator in a WebGPU browser.
