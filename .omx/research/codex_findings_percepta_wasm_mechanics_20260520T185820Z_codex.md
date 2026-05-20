# Codex Findings: Percepta / Wasm-In-Transformer Mechanics

UTC: 2026-05-20T18:58:20Z
Owner: Codex
Lane: lane_percepta_wasm_mechanics_reverse_engineering_20260520
Scope: primary artifact/code inspection for the "Wasm interpreter in transformer weights" claim.
Authority: research_only=true; score_claim=false; promotion_eligible=false.

## Executive Verdict

The real, inspected mechanism is not "an LLM learned to run arbitrary
WebAssembly." It is a hand-compiled, sparse, deterministic transformer-like
circuit whose weights implement a tiny stack-machine interpreter. The public
Percepta page is a blog claim; I found no official Percepta model weights,
compiler, test suite, or C-to-Wasm artifact. The concrete public artifact is an
independent Hugging Face implementation by `eastlondoner`, explicitly inspired
by Percepta:

- `eastlondoner/wasm-transformer`, repo sha
  `553102d707b44f8bfa39584a7f2fe478d953db08`.
- `eastlondoner/wasm-interpreter-transformer`, repo sha
  `a34f874f09c9c10126eae127ef928deba6442858`.

The idea is real enough to learn from: small deterministic state machines can
be compiled into weights. The artifact is not a general Wasm runtime and should
not be cited as evidence that arbitrary C, full Wasm, exact i32 semantics, or
portable differentiable execution is solved.

## Inspected Primary Artifacts

Downloaded HF artifacts to `/tmp` for inspection only; durable identifiers:

| Artifact | Evidence |
|---|---|
| Percepta official page | `https://www.percepta.ai/blog/can-llms-be-computers`; crawl-visible page has title/author/date only. A mirror contains the long post text. |
| HF repo with code | `https://huggingface.co/eastlondoner/wasm-transformer`, sha `553102d...`; includes `configuration_wasm_transformer.py`, `modeling_wasm_transformer.py`, `run_inference.py`, `load_model.py`, `model.safetensors`. |
| Older HF repo | `https://huggingface.co/eastlondoner/wasm-interpreter-transformer`, sha `a34f874...`; lacks the custom model code and says 112/112 tests rather than 115/115. |
| `wasm-transformer/model.safetensors` | sha256 `2b5f958f84aa0f1896389e7eb2473fd60d12315212cff857889123e2e887c71a`, 146 tensors, 316,000 F32 params, 1,277,942 bytes. |
| `wasm-transformer/modeling_wasm_transformer.py` | sha256 `48d31ea375539d861b50be6b2e6b04ed5d80d6249f090674fe5d7f32640cc7f0`, 32,670 bytes. |
| `wasm-transformer/run_inference.py` | sha256 `cb96d24d9d1c314da77de38d752d648437683f62d42f575111a60d0d41587197`, 33,471 bytes. |

Local verification: `run_inference.py` passes the 9 bundled examples. The
claimed 115-test suite is not present in the HF repo, so I could not reproduce
the 115/115 claim from primary files.

## Architecture

The HF `config.json` defines:

- `d_model=100`, `n_layers=8`, `heads_per_layer=[13,1,0,8,2,1,1,4]`.
- 30 total attention heads, `d_head=2`, `d_ffn=100`, `vocab_size=260`.
- Special tokens: 256 program start, 257 program end, 258 branch taken, 259 halt.
- Attention modes: hard-max retrieval, one sum-mode head, and four filesystem cross-attention heads.
- Residual stream layout: byte value, program/trace flags, instruction index and square, opcode flags, computed stack-depth fields, fetched opcode flags, runtime stack/result fields, bit channels, local/memory fields, and filesystem fields.

The weights are sparse and visibly hand-authored:

- Overall density by loaded tensors is tiny: layer densities range from about
  0.03% to 0.91%; embedding density is about 6.0%; unembedding density is about
  2.0%.
- `embedding.weight` maps byte tokens to numeric byte values, constant-one, one-hot opcode flags, and bit channels.
- `unembed.weight` implements the quadratic scoring pattern: token `t` has
  coefficients roughly `-t^2` on the constant dimension and `2t` on result.
- Layer 1's sum head has value weights for stack deltas and writes cumulative
  depth into the depth channels.
- Layer 4's FFN gate/value/output matrices implement arithmetic, comparisons,
  and bitwise reconstruction using ReLU-gated products and fixed output
  coefficients.

The Hugging Face `WasmTransformerModel` sets parameters with
`requires_grad=False`. This is compiled circuitry, not a trained model.

## Program And State Representation

The artifact does not parse `.wasm` binaries. It uses a toy instruction object
with `(op, operand)` and a fixed 5-byte token encoding:

- `run_inference.py`: `[op, operand_low, operand>>8, operand>>16, operand>>24]`.
- HF model code `run_program`: `[op, operand, 0, 0, 0]`, so the library-facing
  helper effectively exposes one operand byte unless callers are careful.

Execution state is append-only transformer trace state plus Python-side driver
state:

- Program tokens are first fed into the KV cache.
- Each execution step feeds the prior output token plus a program-specific PE.
- Attention reconstructs operands from previous program/trace positions.
- Sum-attention accumulates stack depth from prior value vectors.
- Hard-max heads retrieve stack top/second, local sources, memory stores, bit
  positions, and filesystem bytes.
- FFNs compute the result channel and halt channel.
- The unembedding chooses the next byte/special token via argmax.

Important host-side participation:

- `analyze_program()` is Python code, not weights. It walks the program and
  computes loop metadata and filesystem cursor hints.
- `run_program()` / `run_transformer()` are Python control loops. They choose
  the current instruction index, skip structural loop markers, enforce max loop
  counts, branch the virtual instruction pointer after `br_if`, update the
  simulated filesystem, and provide dynamic filesystem cursor overrides.
- Filesystem values live in an external KV table built by Python:
  4 file descriptors x 32 bytes. `fd_write` mutates this external table.

So the model does perform real matrix/attention/FFN computation, but the public
artifact is not a standalone neural CPU. It is a compiled neural circuit plus a
custom execution harness.

## Supported Instruction Subset

Configured/supported subset:

- Arithmetic/logic: `i32.const`, `i32.add`, `i32.sub`, `i32.mul`, `i32.and`,
  `i32.or`.
- Comparisons: `i32.eq`, `i32.ne`, `i32.lt_s`, `i32.gt_s`, `i32.le_s`,
  `i32.ge_s`.
- Storage: `i32.load`, `i32.store`, `local.get`, `local.set`, `local.tee`.
- Filesystem: `fd_open`, `fd_read`, `fd_write`, `fd_close`, limited to 4 fds
  and 32 bytes each.
- Control: `loop`, `end_loop`, `br_if`, bounded by 256 loop iterations.
- Output/termination: `output`, `halt`.

Not supported as a real Wasm runtime:

- No `.wasm` module parser, validation, sections, imports/exports, functions,
  tables, globals, call stack, traps, real WASI, SIMD, floats, i64, or memory
  growth.
- `br`, `call`, and `return` appear in `run_inference.py`'s parser/reference
  enum, but they are not in `config.json`'s supported operations and failed my
  transformer probes.
- The arithmetic is byte-oriented, despite `i32` names. The reference VM wraps
  arithmetic to `& 0xFF`, and output tokens are byte tokens.

Boundary probes:

- Bundled examples: 9/9 passed locally.
- `i32.const 250; i32.const 10; i32.add; output; halt` failed:
  reference `[4]`, transformer `[255]`.
- `i32.const 20; i32.const 13; i32.mul; output; halt` failed:
  reference `[4]`, transformer `[255]`.
- `i32.const 1; br 0; output; halt` failed:
  reference `[]`, transformer `[0]`.
- `i32.const 5; call 0; output; halt` failed:
  reference `[5]`, transformer `[0]`.
- Memory and locals are real for small byte examples:
  repeated stores/loads and repeated `local.set` passed my custom probes.

The failure mode is coherent: arithmetic that produces an internal result above
the byte token range does not implement modulo arithmetic; quadratic unembedding
selects the highest available byte token near the out-of-range result.

## Determinism Limits

Determinism is strong only inside the artifact's intended numeric envelope:

- Hard-max and `argmax` make outputs deterministic for a fixed implementation,
  but tie handling is an implementation detail. The code uses "last argmax" in
  several places and a custom convex-hull path in HF model code.
- The construction relies on large exact-ish constants such as 10,000, 30,000,
  and 100,000 in F32 tensors. Device/compiler changes can perturb hard
  threshold margins if they are close.
- The fast path is nonstandard from a Transformers perspective: `ConvexHullKV`
  replaces a linear hard-max scan for non-sum, non-cross heads. That is a
  custom inference data structure, not vanilla HF text-generation.
- The "differentiable execution" story is not established by the artifact:
  parameters are frozen, attention is hard argmax / hull query, and driver
  control flow updates Python state.
- "No runtime values in PE" is too strong for the inspected code. The driver
  supplies filesystem cursor overrides, and the Python harness controls the
  instruction pointer and loop state.

## What Is Real Versus Marketing

Real:

- The weights are hand-compiled and sparse.
- The forward pass uses real matrix multiplies, attention projections, FFN
  gating, and argmax unembedding.
- The architecture implements useful finite stack-machine mechanics: operand
  retrieval, stack-depth accumulation, local/memory lookup, byte bitwise ops,
  output, halt, bounded loops, and small external filesystem reads/writes.
- This is a concrete example of "programs into weights" for tiny deterministic
  circuits.

Marketing or overreach:

- "Complete WASM interpreter" is false for the inspectable artifact. It is a
  restricted bytecode DSL with Wasm-flavored names.
- "Arbitrary C code" is not publicly demonstrated by primary artifact files.
  No public C compiler path, Wasm binary parser, or full test corpus was found.
- "115/115 tests" is not reproducible from the HF repo as provided.
- "The model executes by itself" hides the amount of Python execution harness:
  program analysis, control-flow stepping, filesystem mutation, and cursor
  overrides are outside the weights.
- The Percepta blog's d_model=36 / n_heads=18 / n_layers=7 vanilla transformer
  claim is separate from eastlondoner's d_model=100 / 8-layer / custom-code HF
  artifact. Do not conflate them.

## Pact Implementation Lessons

Safe lessons for Pact:

- Compile tiny deterministic byte-level correction circuits into decoder
  tensors when the circuit is small, bounded, and consumed by the existing
  inflate path.
- Good candidates: affine channel gates, fixed residual selectors, byte
  lookup/codebook selection, sign/magnitude gates, bounded finite-state
  predictors, fixed hard-pair patch selectors, tiny local arithmetic on q-symbol
  streams, and deterministic unembedding/LUT-style decoders.
- If using attention-like retrieval, keep it byte-closed: fixed packet grammar,
  explicit tensor section, deterministic reconstruction, no scorer at inflate
  time, no hidden host-side analysis, raw output parity, and CPU/CUDA axis
  separation.
- Treat compiled-weight circuits as a way to replace a few dozen lines of
  deterministic decoder logic, not as a reason to ship an interpreter.

Unsafe or unfaithful for Pact:

- Shipping a general Wasm interpreter or Python-controlled VM in the runtime.
  The byte cost and compliance risk dominate any likely score gain.
- Letting an offline analyzer compute semantic state that the submitted archive
  does not encode, then claiming the weights performed that computation.
- Relying on hard argmax/hull tie behavior without exact parity tests on the
  target contest runtime.
- Claiming signed i32, C, or Wasm correctness from byte-domain toy op tests.
- Introducing external mutable filesystem/state side channels at inflate time.
- Treating a differentiability claim as useful for score training until the
  actual operation is gradient-reachable through the Pact scorer path.

Near-term Pact use: research tag only. A credible first experiment would be a
tiny fixed correction circuit embedded into an already-present decoder tensor,
with a no-op detector and byte/output parity proof. A full interpreter is not a
frontier-moving path unless a separate byte-closed artifact proves it beats the
current simpler codec mechanisms.

## Source Links

- Percepta official page: `https://www.percepta.ai/blog/can-llms-be-computers`
- Mirrored Percepta text inspected for missing body content:
  `https://tool.lu/en_US/article/7HF/preview`
- HF model with code:
  `https://huggingface.co/eastlondoner/wasm-transformer`
- HF older model:
  `https://huggingface.co/eastlondoner/wasm-interpreter-transformer`
- Prior Codex broad source review:
  `.omx/research/codex_findings_programs_into_weights_wasm_transformer_20260520T193000Z_codex.md`
