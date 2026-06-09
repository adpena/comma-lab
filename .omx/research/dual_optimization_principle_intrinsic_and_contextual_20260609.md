# The Dual-Optimization Principle: intrinsic AND contextual, at every layer

UTC 2026-06-09 · claude · operator binding directive: "all elements and atoms and dimensions and
layers of all get full optimization themselves AND ALSO contextual optimization within the greater
ecosystem of the stack/pipeline of which they are part." A binding meta-principle, sister to
non-arbitrariness.

## The law
Every element E in the stack must be optimized on TWO axes:
1. **INTRINSIC** — E is at its own frontier in isolation (its best possible form for what it does).
2. **CONTEXTUAL** — E is optimal *for its role in the composition* (accounting for how it interacts
   with every other element it composes with).

Neither alone is sufficient, because the objective is **non-separable**: when components interact,
`S(compose(E1..En)) != sum/compose of S(Ei) optimized alone`. A locally-perfect atom can be globally
harmful (bad cross-terms); a globally-tuned-but-locally-sloppy element wastes intrinsic potential. The
correct optimization shape is **coordinate-ascent-with-interaction-terms over intrinsically-maxed
components** — push each to its frontier, then optimize the composition's cross-terms, iterate.

## Why this is forced by the contest law
`S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/N` is non-separable across components:
- the `sqrt(10·d_pose)` term is **nonlinear** — marginal value depends on the operating point;
- the rate term shares **one byte budget** across all tensors/atoms (allocation is joint);
- actions **do not commute** — `ΔS(a∘b) != ΔS(a) + ΔS(b)`;
- the evaluator couples seg (frame1 argmax) and pose (both-frame YUV6) through the shared frame1.
So intrinsic-only (greedy-local) provably leaves score on the table; contextual is mandatory.

## The same law at every layer (each is one instance)
| layer | INTRINSIC (element at its frontier) | CONTEXTUAL (optimal in composition) |
|---|---|---|
| codec / bit | each tensor's best per-channel quant | joint waterfill under the shared byte budget by evaluator sensitivity |
| optimizer | each group's right optimizer (AdamW/Muon/Aurora) | composed operators jointly (comp-Muon; PixelShuffle∘conv; selector×menu) |
| action atom | each atom minimizes its own ΔS | commutator planner: ΔS(a∘b) − ΔS(a) − ΔS(b) cross-terms |
| curriculum stage | each stage does its job (recon fit / seg anneal / pose anneal) | stage ordering + held anchors so stages compose without collapse |
| vehicle (carrier) | each of HiNeRV/SNeRV/PACT-VQ/PR110++ at its own best | V3 composes them: which mixture minimizes total S |
| evidence row | each row intrinsically valid (authority_tier × metric_family) | competes in the portfolio under EV-per-wallclock-per-byte |

## The V3 schema hook (wire when the portfolio exists)
Every `CandidateActionEvaluation` should carry BOTH:
- `intrinsic_optimality_status` ∈ {at_frontier, improvable, unknown} — is the element maxed in itself?
- `contextual_delta_s` — its exact ΔS *measured in the current composition* (not in isolation).
And the commutator field `comm(a,b) = ΔS(a∘b) − ΔS(a) − ΔS(b)` for any pair that co-occurs. The
waterfiller then admits not by intrinsic ΔS alone but by **contextual** ΔS with commutator corrections
— the operationalization of this principle. (Gated on a portfolio of >1 candidate; premature now.)

## Stack audit against the principle (honest, today)
- **Clean PR95 baseline (running):** the INTRINSIC-recipe proof (the proven element in isolation). It
  does NOT yet test contextual composition with atoms/selectors — that's V3's later job. Correct order:
  prove the element intrinsically (clean PR95) before composing it contextually.
- **R1/R2/R3 (off-spec):** failed the CONTEXTUAL axis — score-aware applied from epoch 0 was a bad
  *composition* of stages (the elements may have been fine; the ordering/interaction was wrong).
- **Arm A (21.74 dB):** INTRINSIC capacity proof of the carrier element; says nothing contextual (not score).
- **Waterfiller + commutator planner (designed, task #30):** the CONTEXTUAL mechanism — it exists in
  the plan precisely because intrinsic atom optimization is insufficient under non-separability.
- **Adaptive codec (gated):** the dual law at the bit layer — per-tensor intrinsic precision × joint
  budget contextual allocation.

## The discipline (binding)
1. Build every element to its intrinsic frontier — never ship a sloppy-local element.
2. NEVER assume intrinsic optima compose — measure the contextual ΔS + the commutators by exact eval.
3. Order the work intrinsic-first, contextual-second (prove the element, then compose it) — but treat
   the contextual measurement as mandatory, not optional polish.
4. A candidate is "optimal" only when BOTH axes are satisfied: at its own frontier AND ΔS-negative in
   the live composition.

## One sentence
**Optimize every element to its own frontier AND optimize the composition's cross-terms — because the
score is non-separable, so the shortest evaluator-equivalent program is found by coordinate-ascent with
interaction terms over intrinsically-maxed parts, never by greedy-local or pure-global alone.**
