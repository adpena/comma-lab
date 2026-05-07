---
name: Codegen + genetic + evolutionary research synthesis (subagent deepening)
date: 2026-05-07
type: research-synthesis
parent_memo: feedback_codegen_genetic_evolutionary_research_direction_20260507.md
evidence_grade: "[predicted-band only — NO contest-CUDA claims]"
---

# Codegen / Genetic / Evolutionary — research synthesis for the cathedral

Operator directive (2026-05-07): "what about codegen and genetic and
evolutionary?" The parent memo named three orthogonal vectors and queued the
deeper exploration here. This memo (a) surveys real literature per vector,
(b) maps each vector to specific cathedral integration points with LOC
estimates, (c) issues a WIRE/STUDY/PASS verdict per vector, and (d) sketches
an implementation pilot for the highest-WIRE pick. Every score-impact band
below is tagged `[predicted-band only]` per CLAUDE.md rules.

---

## 1. Three vectors at a glance

**Codegen — LLM-as-codec-designer.** Spawn an LLM (Claude, Gemini, or local)
with the `tac.codec_pipeline.CodecOp` Protocol + a substrate description +
the `contest_score_decomposition()` formula, and ask it to generate new
CodecOp source code. Surviving candidates that pass the protocol-conformance
test suite (`test_codec_pipeline.py`) and beat a baseline blob enter the
sweep manifest as fresh atoms. This is the highest-ceiling vector but also
the one with the deepest review burden — generated code lands in the
contest archive pipeline, so it must clear `review_tracker.py` like any
human PR. DeepMind's FunSearch (Nature 2023) and AlphaEvolve (May 2025)
are the canonical exemplars.

**Genetic algorithms (CMA-ES-class).** Population-based black-box search
over CodecOp continuous knobs (`brotli_quality`, `n_components`,
`sidecar_dim_codebook`) and discrete choices (`byte_map` permutations,
`conv4_perm` orderings). CMA-ES (Hansen 2016) handles continuous + ill-
conditioned non-separable landscapes; mainstream Python via `pycma` or
`cmaes` (CyberAgent, 2024 paper). Cathedral fit is mechanical: replace
`--param-grid` with a CMA-ES ask-and-tell loop around the existing sweep
manifest's evaluation harness. The cathedral's hot path is brotli at ~22ms
per call; 1000 evals = ~22s wall-clock, fully tractable.

**Evolutionary strategies + neuroevolution.** Broader umbrella covering
OpenAI ES (Salimans 2017, scalable RL alternative), NEAT (Stanley 2002,
topology + weight evolution with historical innovation markings), and
quality-diversity methods like MAP-Elites. For our domain, the relevant
specialization is **topology evolution over CodecPipeline compositions**:
discrete genetic search over which ops compose, in what order, with what
parameter bindings. This is what the four-way stack does manually; an
evolutionary search would automate the substitutional vs substrate-transform
vs decorator-mode composition decisions and surface non-obvious compositions
the human council didn't enumerate.

---

## 2. Literature hits

### 2.1 Codegen (LLM-as-codec-designer)

- **Romera-Paredes et al, "Mathematical discoveries from program search with
  large language models" (FunSearch), Nature 2023** — pretrained LLM proposes
  Python programs; an evaluator filters by an objective; surviving programs
  feed the next prompt. Discovered new cap-set lower bounds (first
  LLM-attributed mathematical discovery). Key insight: the LLM is prompted
  with the *best-so-far* programs as in-context examples — it operates as
  a creative mutation operator, not a from-scratch generator.
  ([deepmind.google/funsearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/))

- **Novikov et al, "AlphaEvolve: A coding agent for scientific and
  algorithmic discovery", arXiv 2506.13131 (May 2025)** — successor to
  FunSearch; evolves whole code files (not single functions); used Gemini
  2.0 Flash + Pro in tandem (cheap-fast for breadth + slow-careful for
  deep edits); discovered a 48-mul algorithm for 4×4 complex matrix
  multiplication (first improvement over Strassen in 56 years); recovered
  0.7% of Google's data-center fleet capacity by evolving a scheduling
  heuristic. Direct relevance: shows LLM-evolution scales to performance-
  critical code that gets shipped to production.
  ([arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131))

- **Chen et al, "EvoPrompting: Language Models for Code-Level Neural
  Architecture Search", NeurIPS 2023** — LLMs as adaptive
  mutation/crossover operators in evolutionary NAS. Demonstrates the LLM
  is a *strictly better* mutation operator than Gaussian noise on
  architecture genotypes. Applies directly to our CodecOp topology
  evolution.
  ([arxiv.org/abs/2302.14838](https://arxiv.org/abs/2302.14838))

- **Nasir et al, "LLMatic: Neural Architecture Search via Large Language
  Models and Quality Diversity Optimization", 2024** — combines LLM-
  generated architecture proposals with MAP-Elites quality-diversity
  archive. Generates an *archive* of high-performing diverse codecs,
  not a single point. Highly relevant to a portfolio-of-archives strategy.
  ([arxiv.org/abs/2306.01102](https://arxiv.org/html/2306.01102v8))

- **Li et al, "Competition-level code generation with AlphaCode", Science
  2022** — generate massive numbers of candidates, filter by tests. The
  test-filter pattern is exactly what the cathedral's
  `test_codec_pipeline.py` enforces.
  ([science.org/AlphaCode](https://www.science.org/doi/10.1126/science.abq1158))

### 2.2 Genetic / CMA-ES

- **Hansen, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772
  (2016)** — canonical reference. CMA-ES adapts a multivariate Gaussian
  sampling distribution by tracking historical step-direction successes;
  invariant to monotone transformations; handles ill-conditioned
  landscapes (10⁵+ condition numbers) where finite-difference gradient
  methods fail.

- **Nomura et al, "cmaes: A Simple yet Practical Python Library for
  CMA-ES", arXiv:2402.01373 (2024)** — modern ask-and-tell API, integrated
  with Optuna; supports CMA-ES + sep-CMA-ES + IPOP/BIPOP variants and
  warm-starts. Pip-installable; CyberAgent maintained.
  ([arxiv.org/abs/2402.01373](https://arxiv.org/html/2402.01373v2))

- **Hansen et al, "pycma" repo (active 2025)** — older but more
  feature-complete: handles equality + inequality constraints, mixed-
  integer (CMA-ESwM), restart strategies. Slightly heavier API but
  superset of `cmaes`. ([github.com/CMA-ES/pycma](https://github.com/CMA-ES/pycma))

- **Fortin et al, "DEAP: Evolutionary Algorithms Made Easy", JMLR 2012
  (still actively maintained 2025)** — the framework for genetic
  programming, multi-objective NSGA-II/SPEA2, particle swarm. Critical
  for *topology* evolution where individuals are compositions, not
  vectors. ([github.com/DEAP/deap](https://github.com/DEAP/deap))

- **Beyer & Schwefel, "Evolution Strategies — A Comprehensive
  Introduction", Natural Computing 2002** — foundational reference for
  the (μ + λ) and (μ, λ) selection schemes that DEAP and pycma both
  implement.

### 2.3 Evolutionary / neuroevolution

- **Salimans et al, "Evolution Strategies as a Scalable Alternative to
  Reinforcement Learning", arXiv:1703.03864 (2017)** — OpenAI ES;
  scales to 1000+ workers via common-random-numbers communication;
  solves 3D humanoid in 10 min. Directly applicable when the fitness
  function is expensive *and* parallelizable.
  ([openai.com/index/evolution-strategies/](https://openai.com/index/evolution-strategies/))

- **Stanley & Miikkulainen, "Evolving Neural Networks through Augmenting
  Topologies (NEAT)", Evolutionary Computation 2002** — historical
  innovation numbers + speciation enable variable-genome crossover. The
  *innovation-number* trick is the right primitive for evolving the
  **CodecPipeline op-graph**: each op-add/op-swap/parameter-mutate gets a
  unique ID, and crossover preserves shared ancestry without expensive
  topological alignment.
  ([nn.cs.utexas.edu/stanley.ec02.pdf](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))

- **Mouret & Clune, "Illuminating search spaces by mapping elites
  (MAP-Elites)", arXiv:1504.04909 (2015)** — quality-diversity archive
  indexed by behavioral descriptors. For codecs, descriptors could be
  (rate_share, seg_share, pose_share) — illuminates the rate-distortion
  triangle directly. Frontier alternative to single-point CMA-ES.

- **Real et al, "Regularized Evolution for Image Classifier Architecture
  Search" (AmoebaNet), AAAI 2019** — tournament selection with aging;
  beat hand-designed and RL-NAS architectures on ImageNet at the time.
  Confirms evolutionary search competes with gradient-based NAS when the
  search space is discrete + structured (which CodecPipeline ops are).

- **Zhang et al, "Vector Quantized-Elites", arXiv:2504.08057 (2025)** —
  modern QD that learns its own behavior descriptors via VQ-VAE.
  Side-benefit: removes the hand-designed-descriptor burden that plagues
  classic MAP-Elites.

---

## 3. Cathedral fit per vector

### 3.1 Codegen → wire-up sketch

- **New file:** `tools/codec_op_llm_codegen.py` (~250 LOC)
- **Reuses:** `src/tac/codec_pipeline.CodecOp` Protocol + existing
  `tests/test_codec_pipeline.py` conformance suite + atom ledger writer
  + `contest_score_decomposition()`
- **External dep:** `anthropic` Python SDK (already installed) or
  `claude-code-sdk` for subagent dispatch
- **Cost model:** ~$0.05–0.15 per LLM-generated CodecOp candidate (Claude
  Opus 4.7 input ~3K tokens substrate description, output ~500 tokens
  Python); ~$5–15 per 100-candidate generation pass
- **Predicted score-impact band [predicted-band only]:** −0.001 to −0.008
  off PR106 frontier (0.20935 → 0.201–0.208) for one good
  LLM-generated rate-side op (e.g., a smarter variable-length encoder
  for the residual delta). Higher-ceiling outcomes possible but
  unproven for codec-design specifically (FunSearch's prior is
  combinatorial-math, not codec engineering).

### 3.2 Genetic / CMA-ES → wire-up sketch

- **New file:** `tools/codec_op_genetic_search.py` (~120 LOC, CMA-ES
  flavor) or `tools/codec_op_cma_search.py` (~80 LOC, thinner)
- **Reuses:** sweep manifest's `build_sweep_candidates()` +
  `to_meta_lagrangian_candidates()` + atom ledger schema
- **External dep:** `cmaes` (CyberAgent) — single `pip install cmaes`,
  pure-Python, no native deps
- **Wire-up:** ask-and-tell loop emits ~20 candidates per generation;
  each candidate runs the existing CodecPipeline encode path (CPU,
  ~50–200 ms per eval depending on op stack); harvest emits one
  ledger row per evaluation; `cmaes.CMA.tell()` updates the
  distribution. ~30–50 generations to convergence on a 4–6 dim
  parameter sub-space.
- **Predicted score-impact band [predicted-band only]:** −0.0005 to
  −0.003 off PR106 frontier on a single op tuning pass. CMA-ES wins
  most when the manual grid was coarse — current sweep manifest
  defaults are coarse, so there's real room. Loses when the landscape
  is convex enough that the existing meta-Lagrangian search already
  finds the optimum (likely true for `n_components` alone but
  unlikely for joint `n_components × brotli_quality × byte_map`).

### 3.3 Evolutionary / topology evolution → wire-up sketch

- **New file:** `tools/codec_op_topology_evolution.py` (~250 LOC,
  DEAP-flavor with NSGA-II selection + NEAT-style innovation
  numbers)
- **Reuses:** `CodecPipeline` instantiation + `contest_score_decomposition()`
  as multi-objective fitness + atom ledger writer
- **External dep:** `deap` (single pip install); optionally
  `pymoo` for NSGA-III if we want better diversity at the
  3-objective Pareto front (rate, seg, pose)
- **Genome:** ordered list of (op_class_name, op_params_dict)
  tuples; mutations = insert / delete / swap / parameter-mutate;
  crossover = innovation-number-aligned two-point splice
- **Predicted score-impact band [predicted-band only]:** −0.002 to
  −0.010 off PR106 frontier IF a non-obvious 5-or-6-op composition
  exists that beats the manual 4-way stack. The *manual* 4-way
  stack already encodes the four highest-prior ops the council
  enumerated, so the marginal gain depends on whether mixing in
  decorator-mode ops (e.g., Op2.5 inference tuning at multiple
  insertion points) plus parameter-tuned variants yields
  super-additive savings. Most likely outcome: a 5-op composition
  that beats the 4-way by 2–4 KB but stays inside the same band
  (i.e., evolutionary search confirms council's pick was already
  near-optimal — *that* is also valuable signal).

---

## 4. Risks / killers per vector

**Codegen risks:**
1. **API-surface hallucination.** LLM invents `CodecOp.encode_async()`
   or imports `tac.codec_pipeline.LegacyOp`. Mitigation: every
   generated file gates on `test_codec_pipeline.py::test_protocol_conformance`
   *before* it's allowed into the manifest. Per CLAUDE.md
   `forbidden_dead_flag_wiring`, also auto-grep generated code for
   non-existent flag names.
2. **Code-review burden + license/IP.** Generated code lands in the
   contest archive pipeline; per the strategic-secrecy rule, public
   PRs disclose codec design. Need an explicit `review_tracker.py
   mark-file <generated.py> --status reviewed` before any
   contest-CUDA dispatch consumes it.
3. **Reward-hacking via Goodhart.** LLM finds a degenerate codec that
   scores well on `contest_score_decomposition` but fails the actual
   contest evaluator (e.g., produces an archive that's bit-faithful in
   our test harness but trips upstream's `inflate.sh` size validator).
   Mitigation: **EVERY** LLM-generated candidate must pass the
   `lane_codec_pipeline_mask_*` smoke test (the existing harness)
   before any GPU dispatch.

**GA / CMA-ES risks:**
1. **Per-eval cost is the bottleneck.** Hot path = brotli at ~22 ms;
   on a 6-dim CMA-ES with population 12 over 50 generations =
   ~13 s wall-clock, totally fine. But if anyone wires CMA-ES around
   a *full pipeline encode* (which can be 200–500 ms when Op3 is
   active), 50 gens × 12 pop = 5 min — still fine but starts to
   matter when run in parallel with the meta-Lagrangian sweep.
2. **Pareto-front collapse.** Single-objective CMA-ES converges to
   one point; at the contest score's three-axis frontier this loses
   the ability to surface "rate-cheaper but seg-worse" alternatives.
   Mitigation: use multi-objective CMA-ES (MO-CMA-ES) or NSGA-II via
   DEAP/pymoo when the operator wants Pareto-illumination.
3. **Premature saturation on convex sub-spaces.** If the search
   space happens to be convex (e.g., `brotli_quality` alone), a
   3-point grid sweep finds the optimum faster than CMA-ES. Pilot
   with a small 2-D grid first to characterize the landscape.

**ES / topology-evolution risks:**
1. **Local-minimum convergence.** Classic ES has no guarantee of
   beating CMA-ES on convex regions; on rugged regions it can
   stagnate. Mitigation: warm-start from the manual 4-way stack
   genome and only let mutations propose *additive* changes (which
   matches the cathedral's "stackable bolt-on" philosophy).
2. **Fitness-evaluation noise.** If two dispatches of the same
   genome yield different scores (they shouldn't, but bytes-out
   *might* drift across brotli versions), evolution chases noise.
   Mitigation: pin brotli version + record `op_state` SHA in
   ledger.
3. **Diversity vs quality trade-off.** Topology evolution will
   produce many bizarre 8-op compositions that nominally pass
   validate() but produce 50-KB-bigger archives. Without a hard
   bytes-budget filter, the population fills with junk. Mitigation:
   pre-screen with a `bytes_out < 1.05 * baseline` constraint
   before scoring.

---

## 5. Operator action items (verdict per vector)

| Vector | Verdict | Rationale | LOC | First step |
|---|---|---|---|---|
| Genetic / CMA-ES | **WIRE** | Lowest integration risk, mechanical fit with sweep manifest, real GP+CMA-ES literature, immediate value on coarse param grids | ~120 LOC + tests | `pip install cmaes` then write `tools/codec_op_cma_search.py` |
| Topology / NEAT-style | **STUDY** | Real upside but speculative; warm-start from 4-way stack first; needs DEAP wiring + 3-axis Pareto fitness | ~250 LOC + tests | Queue research memo on op-graph genome design before code |
| LLM codegen | **STUDY** | Highest ceiling but highest review burden + reward-hacking risk; AlphaEvolve precedent valid only at scale | ~250 LOC + review-gate plumbing | Pilot via single Claude subagent dispatch generating *one* CodecOp variant; gate on test_codec_pipeline; iterate |

No vector is **PASS** — all three have plausible value at the Shannon-floor
operating point. The differentiation is wire-up cost vs ceiling.

---

## 6. Pilot plan — CMA-ES wrapper (highest WIRE pick)

**File:** `tools/codec_op_cma_search.py` (new)

**Purpose:** Replace the hand-curated `--param-grid` JSON with a CMA-ES
ask-and-tell loop that proposes 12 candidates per generation, evaluates
each via the existing `CodecPipeline.encode()` path, writes one row per
eval to the atom ledger, and updates the CMA-ES distribution. Terminates
when generation-best plateau < ε for K generations OR a wall-clock budget
is exhausted.

**Function signatures:**

```python
def run_cma_search(
    *,
    module: str,                          # e.g. "tac.pr101_split_brotli_codec"
    class_name: str,                      # e.g. "Op1_PR101SplitBrotli"
    state_dict_path: Path,
    param_bounds: dict[str, tuple[float, float]],  # {"brotli_quality": (1, 11), ...}
    discrete_params: dict[str, list],      # {"byte_map": ["uint8", "uint16", ...]}
    anchor: ContestAnchor,                 # seg/pose/bytes baseline
    population_size: int = 12,
    max_generations: int = 50,
    plateau_tolerance: float = 1e-4,
    plateau_window: int = 5,
    budget_seconds: float = 300.0,
    output_ledger: Path,
) -> CMASearchResult:
    """Run CMA-ES over CodecOp param landscape; emit ledger rows; return best."""

def _evaluate_candidate(
    op_instance: CodecOp,
    state_dict: dict[str, torch.Tensor],
    anchor: ContestAnchor,
) -> CandidateMeasurement:
    """Encode + measure bytes_out + project contest_score_decomposition."""
```

**Fitness function:** the existing `contest_score_decomposition()`.
Single-objective for v1: `total = seg_term + pose_term + rate_term`.
Multi-objective (MO-CMA-ES) deferred to v2 if v1 saturates.

**Termination criteria (any of):**
1. `gen_best_total - gen_(best-window)_total < plateau_tolerance` for
   `plateau_window` consecutive generations
2. `wall_clock > budget_seconds`
3. `generation > max_generations`

**Atom-ledger row per eval:** one JSONL line per `(op_class, params, bytes_out,
seg_proj, pose_proj, rate_proj, total_proj, evidence_grade=[CMA-ES-CPU-prep])`,
appendable to `experiments/results/bilevel_atom_ledger.jsonl`. Ledger
consumers (3-axis Pareto + meta-Lagrangian search) get the new candidates
for free.

**Test surface:** `tests/test_codec_op_cma_search.py` with at least:
1. CMA-ES converges on a 2-D quadratic toy fitness within 30 generations
   (sanity)
2. CMA-ES on `Op1_PR101SplitBrotli` over (`brotli_quality`,
   `auto_select_threshold`) finds a candidate within 1% of the
   hand-curated grid optimum within 25 generations
3. Atom ledger receives one row per evaluation, each row passes the
   ledger schema check (the existing `tests/test_atom_ledger_schema.py`
   if present, else inline schema validation)
4. Termination criteria exercised: plateau-detection, wall-clock-budget,
   max-generations

**Pilot rollout:**
1. Wire up CMA-ES on a single op (`Op1_PR101SplitBrotli`) over
   2 continuous knobs (`brotli_quality`, `tail_lz4_quality` if
   present; else just `brotli_quality`). Verify ledger rows land.
2. Compare best-found vs current sweep-manifest grid optimum. If
   CMA-ES beats grid by >1% bytes saved, expand to 4-D
   (add discrete `byte_map` choice as binary one-hot).
3. If 4-D pilot converges to a non-trivial improvement, generalize
   to all wired ops and queue a multi-objective NSGA-II variant
   for the Pareto-illumination case.

**No GPU dispatch from the pilot.** All evaluation is CPU-only on the
existing CodecOp encode path. Operator approval gate is only for
*translating* a top-K CMA-ES finding into a contest-CUDA dispatch via
the canonical `tools/parallel_dispatch_top_k.py` actuator (CLAUDE.md
race-mode rule).

---

## 7. Composability with cathedral

Each vector feeds back into existing cathedral primitives without
displacement:

**Atom ledger** (`experiments/results/bilevel_atom_ledger.jsonl`).
- CMA-ES: one row per `(op_class, params, bytes_proj, seg_proj,
  pose_proj, total_proj, evidence_grade=[CMA-ES-CPU-prep])`. Direct
  drop-in; no schema change.
- Topology evolution: one row per genome with the *composition*
  serialized into a new field `pipeline_genome=[op1, op2, op3]`
  (additive schema field). Existing consumers ignore unknown fields.
- LLM codegen: one row per generated CodecOp with
  `evidence_grade=[llm-generated-CPU-prep]` and a
  `provenance.generator_model=claude-opus-4-7` field — keeps generated
  code traceable per the strict-scorer rule.

**3-axis Pareto frontier** (`tools/contest_score_pareto_3axis.py`).
- All three vectors emit candidates that already speak the
  (rate, seg, pose) triple the Pareto tool consumes. No tool change
  needed; the atom-ledger schema field gives the lineage.

**Joint-ADMM coordinator.** ADMM cycles on the dual variables (λ_seg,
λ_pose, λ_rate); search vectors operate one level *above* — they
propose new primal candidates that ADMM then dual-balances. CMA-ES /
ES / codegen all sit at the "atom-generation" layer, ADMM at the
"atom-selection" layer. Clean separation.

**Bilevel optimization 7-phase trajectory.** The bilevel loop's outer
phase is "select next candidate to evaluate"; CMA-ES/topology
evolution/codegen are all candidate-proposers for that outer
selection. They feed candidates into the inner-phase (encode-decode-
project) and harvest the projected score back into the ledger.
The bilevel coordinator itself doesn't change; it just sees more
diverse candidates per iteration, which is exactly what the
"fan-out is a first-class deliverable" rule prescribes (CLAUDE.md
Race-mode rule 1).

**Parallel-dispatch actuator** (`tools/parallel_dispatch_top_k.py`).
After any of the three vectors produces a top-K ranked atom set, the
canonical parallel actuator consumes the ranking and fans out paid-GPU
contest-CUDA evaluations with cost gating. **No vector is allowed to
short-circuit the actuator** — even an LLM-generated CodecOp that
scores well in projection must go through `parallel_dispatch_top_k.py`
to land a `[contest-CUDA]` row. This preserves the empirical
calibration loop (`tools/harvest_and_reseed.py`).

---

## 8. Scope discipline + tags

- All score-impact bands tagged `[predicted-band only]`. Zero
  contest-CUDA claims in this memo.
- Pilot is CPU-only by design. No GPU spend until the operator
  approves a CMA-ES top-K -> parallel_dispatch handoff.
- All three vectors keep tac/comma-lab separation: CMA-ES wrapper
  lives in `tools/`, calls `tac.codec_pipeline.CodecOp` (reusable
  library); LLM codegen wrapper lives in `tools/`, calls Claude
  via `anthropic` SDK; topology evolution lives in `tools/`,
  delegates genome encoding to `tac.codec_pipeline_genome` (a thin
  dataclass module to be added if/when topology evolution is
  promoted from STUDY to WIRE).
- Per CLAUDE.md `forbidden_premature_kill_without_research_exhaustion`:
  if the CMA-ES pilot underperforms the grid sweep, the verdict is
  DEFERRED-pending-research, not KILLED — the 4-D and topology
  variants are unexplored.

---

## 9. Cross-refs

- Parent memo (research-direction queue):
  `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_codegen_genetic_evolutionary_research_direction_20260507.md`
- Hot-path audit (~22 ms brotli budget):
  `feedback_hotpath_audit_brotli_dominant_20260507.md`
- Cathedral CodecOp Protocol:
  `/Users/adpena/Projects/pact/src/tac/codec_pipeline.py`
- Sweep manifest (CMA-ES wraps this):
  `/Users/adpena/Projects/pact/tools/codec_op_param_sweep_manifest.py`
- Atom ledger schema:
  `/Users/adpena/Projects/pact/experiments/results/bilevel_atom_ledger.jsonl`
- 3-axis Pareto:
  `/Users/adpena/Projects/pact/tools/contest_score_pareto_3axis.py`
- Contest score decomposition function:
  `/Users/adpena/Projects/pact/src/tac/contest_rate_distortion_system.py:164`
- Canonical parallel actuator (terminal sink for all three vectors):
  `/Users/adpena/Projects/pact/tools/parallel_dispatch_top_k.py`
