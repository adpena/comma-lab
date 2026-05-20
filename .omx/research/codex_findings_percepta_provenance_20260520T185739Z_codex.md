# Codex Findings: Percepta Provenance For Programs-Into-Transformer-Weights Claim

UTC: 2026-05-20T18:57:39Z
Owner: Codex
Lane: lane_percepta_provenance_adversarial_research_20260520
Scope: Online provenance and factual-claim audit for "coding WebAssembly/programs into transformer weights so it can do deterministic computer things."
Edits: research-only memo plus lane registry bookkeeping; no PR110 live submission files touched.

## Bottom Line

The Percepta claim is real enough to cite as a research direction, but only with careful wording. The supported version is:

> Percepta published a March 2026 research blog series and an Apache-2.0 repo showing an analytically constructed autoregressive transformer that simulates a WebAssembly virtual machine / subset execution stack, with a 2D hull KV-cache fast path for structured hardmax attention lookups.

Do not phrase it as "a trained LLM learned to run arbitrary programs" or "a peer-reviewed result proves LLMs can compute." The official implementation is a code-to-weights / analytic-construction system, not a large pretrained language model that acquired computation through gradient descent.

## Team / Person / Company

- Company: Percepta. Official company page lists founders Hirsh Jain (CEO), Thomas Mathew (CTO), Athul Paul Jacob (Chief AI Officer), Costis Daskalakis (Chief Scientist), Radha Jain, and Michael Rochlin. This identifies company leadership, not necessarily authorship of the transformer-vm work.
- Primary named author for the first post: Christos Tzamos, "together with others at Percepta" on the official March 11 post.
- Follow-up / code-release author: Percepta Research Team on the official March 25 post.
- Official GitHub repo owner: Percepta-Core. Initial commit author in GitHub API: Athul Jacob, commit `6cfee30dd7a8f5bffd76d0b0fcf2932fdd41fc97`, 2026-03-25T20:06:02Z.

## Primary Source Timeline

- 2026-03-11: Percepta publishes "Can LLMs Be Computers?" by Christos Tzamos with others at Percepta. The official page and blog index describe "executing arbitrary C programs for millions of steps" and "exponentially faster inference via 2D attention heads." The full article content is delivered via Next.js chunks, but the official page metadata/byline/date are visible in server HTML and the blog index.
- 2026-03-12/13: Hacker News and Tildes discussions appear. These are useful for critique, not as proof.
- 2026-03-13: Oskar Austegard publishes an independent validation write-up claiming a from-scratch CPU/PyTorch toy implementation based only on Percepta's public post.
- 2026-03-25: Percepta publishes "Constructing an LLM-Computer: From programs into transformer weights." This is the key corrective source: it says they released the full implementation and describes ALM/CALM, a compiler from gate graphs to weights via MILP scheduling, a WebAssembly interpreter, and program specialization.
- 2026-03-25: `Percepta-Core/transformer-vm` is created/pushed. API and `git ls-remote` agree on `main` at `6cfee30dd7a8f5bffd76d0b0fcf2932fdd41fc97`. No GitHub releases exist as of this memo.

## Official Public Artifacts

1. Official blog post:
   - `https://www.percepta.ai/blog/can-llms-be-computers`
   - Claims: in-model execution trace, WebAssembly interpreter inside transformer weights, C -> WASM/token path, more than 30k tokens/sec on CPU, 2D lookup heads / log-time structured decoding, Sudoku/min-cost matching demos.

2. Official construction/code-release post:
   - `https://www.percepta.ai/blog/constructing-llm-computer`
   - Claims: full implementation released; Append-only Lookup Machine (ALM), CALM DSL, LookUp/ReGLU gate graph, MILP schedule to transformer weights, WASM interpreter in CALM, specialization/Futamura projection by baking fixed programs into FFN weights.

3. Official GitHub repo:
   - `https://github.com/Percepta-Core/transformer-vm`
   - License: Apache-2.0.
   - Repo description: "Compile programs directly into transformer weights. Includes a 2D convex-hull KV cache with O(log n) inference."
   - README claim: standard softmax-ReGLU transformer whose weights are computed analytically and correctly simulates a WebAssembly VM on arbitrary programs.
   - Commands: `wasm-run`, `wasm-eval`, `wasm-compile`, `wasm-build`, `wasm-specialize`, `wasm-reference`.
   - Examples manifest: `hello`, `addition`, `collatz`, `fibonacci`, `min_cost_matching`, `sudoku`.
   - README caveat: supported WASM opcodes are finite; unsupported ops such as MUL/DIV/MOD/AND/OR/XOR/SHL/SHR are lowered at compile time.
   - No official model release/package observed; the repo builds weights and C++ inference locally.

## Independent Reproductions / Related Public Models

1. Oskar Austegard / `oaustegard/llm-as-computer`
   - Blog: `https://whtwnd.com/austegard.com/3mgxahx5axp2c`
   - Repo: `https://github.com/oaustegard/llm-as-computer`
   - License: MIT.
   - Created 2026-03-12; current `main` observed at `41dcb62a05c7f0b4205355e7db3e1c63368b84b7`.
   - Claims independent validation from the blog post without Percepta code/weights, with a toy compiled executor, parabolic 2D addressing, loops/conditionals, and a 55-opcode WebAssembly-like stack ISA. This is supportive but not peer review and not an exact reproduction of Percepta's official repo.

2. Hugging Face `eastlondoner/wasm-transformer`
   - URL: `https://huggingface.co/eastlondoner/wasm-transformer`
   - License: MIT.
   - Created 2026-03-15; last modified 2026-04-05 via HF API.
   - Model card: 316k-parameter hand-compiled transformer, 8 layers, 30 heads, `d_model=100`, hardmax/sum/cross-attention, 115/115 test programs passing, explicitly "not by gradient descent."
   - The card says it is inspired by Percepta, so treat as related independent artifact, not Percepta official evidence.

3. Hugging Face `eastlondoner/wasm-interpreter-transformer`
   - URL: `https://huggingface.co/eastlondoner/wasm-interpreter-transformer`
   - License: MIT.
   - Created 2026-03-12; last modified 2026-04-05 via HF API.
   - Similar/overlapping card and safetensors artifact. Treat as related independent model, not official Percepta artifact.

## Critiques / Caution Signals

- Hacker News correctly flagged early that the first artifact was a blog post, not a paper. That changed partly on March 25 because code was released, but a formal paper is still not obvious from public search.
- HN/Tildes critiques focus on practical value: why put a computer inside model weights instead of calling a normal interpreter, how to update/debug/access-control such a VM, and whether the system can integrate with natural-language capability.
- Differentiability remains a weak claim. The March 11 post says the execution trace is part of the forward pass and can support gradients, but hardmax/argmax/hull-cache retrieval is not differentiable in the ordinary soft-attention sense. I found no public Percepta experiment proving a large hybrid LLM can train end-to-end through this mechanism.
- Benchmarking remains narrow. "More than 30k tokens/sec on CPU" is an execution-trace throughput claim for their implementation, not proof it beats native WASM, Python tools, or task-specific CPU code. The official repo README says programs run at approximately 30K tok/s and gives a Sudoku trace around 900K tokens, but no broad native-runtime comparison is presented on the primary pages I checked.

## Claims We Should Not Repeat

- Do not say "a trained LLM learned WebAssembly." The official construction is analytic/code-to-weights, and independent HF artifacts explicitly say no training data/loss/optimizer.
- Do not say "arbitrary C programs" without a scope caveat. Official prose uses broad language, but the repo implements a subset/lowering path and has concrete examples. Safer: "C programs that fit the released WASM subset/lowering pipeline."
- Do not say "fully differentiable computer inside an LLM" as an established fact. Safer: "Percepta proposes differentiable/hybrid integration as a direction; exact hardmax execution is not shown as trainable end-to-end."
- Do not say "peer-reviewed paper" or "formal theorem released" for this Percepta work. I found blog posts and code, not a formal paper/arXiv.
- Do not say "MIT/UW-Madison team" as the team label. Official surfaces say Percepta, Christos Tzamos with others at Percepta, and Percepta Research Team. Individual academic affiliations are separate provenance.
- Do not cite "33K tokens/sec" as a primary Percepta number. Primary wording I verified is "more than 30k" / approximately 30K. "33K" appears in secondary summaries.
- Do not present this as immediate Pact score evidence. It is a programs-into-weights design pattern, not a video-compression result, not a PR110 claim, and not an exact-eval archive substrate.

## Pact Interpretation

The usable signal is narrow: compiled deterministic circuits inside weights are real enough to inspire tiny, byte-closed decoder/correction circuits. The bad interpretation is shipping a general interpreter or citing Percepta as proof that a learned model can compress arbitrary exact computation. For Pact, use this as research-only inspiration for small deterministic tensor/decoder gates, with explicit archive-byte accounting and exact eval. Do not wire it into PR110 public surfaces without a local byte-closed prototype.

## Source Map

- Percepta original post: `https://www.percepta.ai/blog/can-llms-be-computers`
- Percepta blog index: `https://www.percepta.ai/blog`
- Percepta construction/code-release post: `https://www.percepta.ai/blog/constructing-llm-computer`
- Percepta company page: `https://www.percepta.ai/company`
- Official repo: `https://github.com/Percepta-Core/transformer-vm`
- Official repo API: `https://api.github.com/repos/Percepta-Core/transformer-vm`
- Official repo commits API: `https://api.github.com/repos/Percepta-Core/transformer-vm/commits?per_page=5`
- Official repo README raw: `https://raw.githubusercontent.com/Percepta-Core/transformer-vm/main/README.md`
- Oskar Austegard write-up: `https://whtwnd.com/austegard.com/3mgxahx5axp2c`
- Oskar Austegard repo: `https://github.com/oaustegard/llm-as-computer`
- Hugging Face `eastlondoner/wasm-transformer`: `https://huggingface.co/eastlondoner/wasm-transformer`
- Hugging Face `eastlondoner/wasm-interpreter-transformer`: `https://huggingface.co/eastlondoner/wasm-interpreter-transformer`
- Hacker News discussion: `https://news.ycombinator.com/item?id=47348275`
- Tildes discussion: `https://tildes.net/~comp/1t6j/executing_programs_inside_transformers_with_exponentially_faster_inference`
