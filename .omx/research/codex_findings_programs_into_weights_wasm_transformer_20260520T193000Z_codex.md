# Codex Findings: Programs Into Weights / WASM Transformer Executor

UTC: 2026-05-20T19:30:00Z
Owner: Codex
Scope: Online source review for the claim that WebAssembly/program execution can be compiled into neural-network weights, producing deterministic computation inside a transformer.

## Primary Identification

The team/person most directly associated with the recent claim is **Percepta AI**, with the public post by **Christos Tzamos together with others at Percepta**, "Can LLMs Be Computers?", published 2026-03-11.

Percepta's public post claims a transformer can execute programs internally rather than calling an external tool. The public page is sparse in direct crawl form, but mirrored text attributes the core construction to:

- implementing a RAM-style computer inside a transformer;
- compiling C to WebAssembly-like execution traces;
- using a standard PyTorch transformer with `d_model=36`, `n_heads=18`, `n_layers=7`, gated FFNs, and 2D heads;
- replacing long linear attention scans in the executor regime with geometric 2D-head retrieval, described as logarithmic-time supporting-point/convex-hull lookup;
- treating weights as a deployment target for software, where compiled program logic becomes internal model circuitry.

## Public Reproduction / Closest Inspectable Artifact

The most concrete public artifact I found is Hugging Face user **`eastlondoner`**:

- Model: `eastlondoner/wasm-interpreter-transformer`
- License: MIT
- Claimed design: a hand-compiled transformer that executes WebAssembly bytecode through real forward passes; weights set by a compiler, not gradient descent.
- Scale: about 316k parameters, 8 layers, 30 total heads, `d_model=100`, `d_head=2`, `d_ffn=100`, vocab size 260.
- Claimed tests: 112/112 passing, covering arithmetic, comparisons, memory/local variables, filesystem I/O, loops, and termination.
- Supported operations include `i32.const`, arithmetic/logical ops, comparisons, `i32.load/store`, local variables, `fd_open/read/write/close`, loops, `br_if`, output, and halt.

This is important because Percepta's original post is more of a research announcement; the Hugging Face model card is a concrete, downloadable, inspectable artifact.

## Secondary Reviews / Skepticism

Secondary commentary is broadly consistent:

- The work is technically interesting because it compiles deterministic execution into weights rather than training it.
- The practical caveat is severe: a compiled WASM interpreter in weights may be slower and less maintainable than simply running a WASM runtime unless differentiability or in-model integration matters.
- Training was not demonstrated in the core claim; the weights are compiled/directly written.
- The differentiability claim is not fully proven for hard/average-hard attention variants.
- Benchmarks against native WASM, Python tool-calling, or task-specific CPU code are incomplete.

One independent write-up by `austegard.com` reports rebuilding the idea from the blog post, validating the parabolic-key/2D-head lookup primitives, and compiling a small stack-machine executor into a PyTorch `nn.Module`. That reproduction explicitly says the compile path worked while gradient training struggled to learn exact arithmetic.

## What Matters For Pact

This is not immediately a PR110 candidate path. It is a design pattern for deterministic neural circuitry:

1. **Program-as-weights is real enough to treat as a substrate idea.** We can compile exact operations into model weights when the operation is small, deterministic, and byte-cheap.
2. **It is not free compression.** A general WASM interpreter would likely cost far more bytes than PR110's remaining budget unless extremely specialized.
3. **The useful contest analogue is not "ship a full interpreter."** The useful analogue is compiling tiny deterministic correction circuits into already-present decoder weights or small sidecar-controlled neural modules.
4. **Best fit for current decoder-q lane:** represent local deterministic repair logic as constrained q-symbol edits or tiny compiled gates inside existing tensors, then verify by byte rebuild + raw inflate + advisory/exact eval.
5. **Do not cite as PR evidence without caution.** The Percepta result supports the general concept of compiled computation in weights, not any specific video-compression score claim.

## Recommended Pact Follow-Up

- Add `programs_into_weights` as a research-only inspiration tag for the decoder-q observable lattice and future waterbucket planners.
- Keep all generated candidates byte-closed; do not introduce an interpreter runtime into PR110.
- If we experiment, start with a tiny exact compiled circuit such as a channel-gated affine correction in the final RGB head, not a full WASM interpreter.
- Treat any "differentiable execution" claim as unproven until reproduced locally with gradients through the actual operation.

## Source Map

- Percepta official page: `https://www.percepta.ai/blog/can-llms-be-computers`
- Percepta mirrored text: `https://tool.lu/en_US/article/7HF/preview`
- Hugging Face artifact: `https://huggingface.co/eastlondoner/wasm-interpreter-transformer`
- Independent reproduction/write-up: `https://whtwnd.com/austegard.com/3mgxahx5axp2c`
- Critical secondary summary: `https://awesomeagents.ai/news/percepta-transformer-computer-wasm-deterministic/`
- Discussion mirror: `https://tildes.net/~comp/1t6j/executing_programs_inside_transformers_with_exponentially_faster_inference`
