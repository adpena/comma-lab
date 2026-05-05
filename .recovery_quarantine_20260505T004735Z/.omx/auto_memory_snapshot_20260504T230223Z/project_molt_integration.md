---
name: molt compiler integration — tinygrad compiled renderer + openpilot models
description: adpena/molt (Rust Python compiler for tinygrad) will compile our renderer to standalone binaries and WASM. Conv2d + GroupNorm being implemented by partner agent. ONNX support exists. Expect these ops to land and be usable in pact/tac.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## What's Coming from molt

A partner agent is implementing in adpena/molt:
1. Conv2d (standard, depthwise-separable, dilated, with padding_mode)
2. GroupNorm (for CLADE normalization)
3. ConvTranspose2d (for U-Net upsampling)
4. grid_sample (for radial zoom warp + flow-based warping)
5. F.interpolate (bilinear, for eval_roundtrip simulation)

Once landed, molt can compile our renderer (103K-181K params) into:
- Standalone native binary (Metal on Mac, CUDA on Linux)
- WASM binary (for browser demo on Cloudflare site)
- ONNX import path (already supported)

## How to Use in pact/tac

### Inflate-time (contest submission)
```
# Instead of: python inflate_renderer.py archive/ output/ names.txt
# Could be:   ./renderer_compiled < masks.mkv > frames.raw
```
No Python. No pip install. No PyTorch. Just a binary + masks + zoom scalars.

### Compress-time (openpilot integration)
molt can compile openpilot's supercombo model → lane detection binary
→ zoom scalar computation → no full openpilot stack needed

### Paper/portfolio (WASM demo)
Compile renderer to WASM → embed in Cloudflare site → visitors render
frames live in browser from the actual trained weights

## Integration Points in tac

- src/tac/renderer_export.py — add molt/tinygrad export alongside FP4/int4+LZMA2
- experiments/pipeline.py — add step_compile_molt after weight compression
- submissions/robust_current/inflate_renderer.py — detect compiled binary format
- Quarto/marimo paper — embed WASM demo

## Current molt State

- Rust-based Python AOT compiler
- 26 tinygrad primitives implemented
- Metal/WASM/WebGPU/CPU backends
- ONNX import supported
- openpilot supercombo demo exists (shape validation)
- Gap: Conv2d, GroupNorm, ConvTranspose2d not yet in Python nn module
- Estimated implementation: partner agent working on it now

## Timeline

- Conv2d + GroupNorm: expect within days (partner agent)
- Renderer compilation test: as soon as ops land
- WASM demo: after competition (for paper/portfolio)
- Production deployment: longer term (needs edge hardware testing)
