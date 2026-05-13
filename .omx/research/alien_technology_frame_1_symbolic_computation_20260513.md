# Frame 1 — Symbolic-computation lineage (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §1.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).
**Persona**: blank-slate alien-tech researcher.

## Worldview

A civilization that discovered formal logic (Frege 1879, Russell 1903, Hilbert 1900) and theorem-proving (Gödel 1931, Church 1936, Turing 1936) — but **never** the perceptron (1957) or backprop (1986). Their AI is symbolic-algebraic. They distrust any output that doesn't carry a checkable witness.

## Core inductive bias

**Proof certificates over fit quality.** Their archive is a **PROGRAM** with a **CORRECTNESS PROOF**, not a tensor of weights with empirical validation.

## Concrete technique 1A: SMT-encoded archive synthesis

```
Variables: A[0..N-1] ∈ {0,1}^8           # archive bytes
Constraints:
  1. score(inflate(A)) ≤ S_target
  2. |A| ≤ B_target
  3. A satisfies ZIP/runtime schema constraints

Goal: minimize |A| subject to (1)-(3)
```

Solvers: [Z3](https://github.com/Z3Prover/z3), [CVC5](https://cvc5.github.io/), [Bitwuzla](https://github.com/bitwuzla/bitwuzla).

**Tractability barrier**: 800K boolean variables for naive byte search. **Decomposition**: solve only for the LATENT axis (15 KB → ~120K bits), holding decoder weights fixed.

**Empirical hope**: SMT-certificate-based feasibility proofs. Even a NEGATIVE result (no archive of size ≤ N achieves S_target) is a **lower bound** we currently lack.

**[unknown-unknown]**: SMT solvers HAVE solved similar combinatorial problems in cryptography / formal verification. The question is whether the contest's specific structure admits a tractable encoding.

## Concrete technique 1B: Symbolic regression of scorer surface

[PySR](https://github.com/MilesCranmer/PySR), [SR-bench](https://github.com/cavalab/srbench), [QLattice](https://abzu.ai/).

For the contest video, fit per-region symbolic expressions:

```
yuv6_patch(x, y, t) = c_1 * T_2(x/W) * T_2(y/H) + c_2 * sin(2π * t/T)
                      + c_3 * smoothstep(...) + residual_NN(x, y, t)
```

where `T_k` are Chebyshev polynomials. The archive stores `c_1, c_2, c_3` per patch + small residual NN.

**[mathematical-derivation]**: A degree-6 bivariate Chebyshev expansion per 32×32 patch is 28 coefficients. For 1164×874 frame ÷ 32×32 patch = 5089 patches. Per-frame: 5089 × 28 × 2B = 285 KB. Per 1200 frames at 99% temporal redundancy: ~3 KB/frame after delta-coding. Total: ~3.6 MB raw, ~50 KB after arithmetic coding.

**Verdict**: comparable to existing wavelet codecs. NOT a paradigm shift — but a **byte-faithful symbolic representation** unlocks the symbolic-computation lineage's next steps.

## Concrete technique 1C: Theorem-prover-driven archive synthesis

[Coq](https://coq.inria.fr/) / [Lean](https://lean-lang.org/) statement:

```lean
theorem archive_exists : ∃ (A : Vector Byte N) (_ : N ≤ B_target),
  score (inflate A) V_GT ≤ S_target := by
    -- proof construction
    sorry
```

Curry-Howard: an inhabitant of this type IS an archive. Extract via `lean --extract`.

**Tractability**: zero progress without years of investment. **Long-game value**: **judges could verify the score without running the GPU.**

## Concrete technique 1D: Inductive logic programming

[Progol](https://www.doc.ic.ac.uk/~shm/progol.html), [FOIL](https://en.wikipedia.org/wiki/First-order_inductive_learner).

Learn Horn-clause rules describing each frame:

```
is_road(x, y, t) :- is_lane_marker(x, y, t), in_belt(y, 400, 700).
is_sky(x, y, t)  :- ¬is_road(x, y, t), y < 200.
ego_motion(t)    :- prev_pose(t-1) + Δ_pose(t), Δ_pose(t) ∈ canonical_actions.
```

Archive = ~500 Horn clauses ≈ 5-10 KB.

**Catch**: pixels-from-Horn-clauses is hard. Hybrid: ILP for STRUCTURE; tiny NN for textures.

## Closest extant work

- AlphaCode [Li et al. 2022](https://www.deepmind.com/blog/competitive-programming-with-alphacode)
- DreamCoder [Ellis et al. 2021](https://arxiv.org/abs/2006.08381)
- ARC-AGI [Chollet 2019](https://arxiv.org/abs/1911.01547) — symbolic-vs-statistical benchmark
- Microsoft Project Verona [Verona language](https://github.com/microsoft/verona)

## Wire-in declaration

All 6 hooks: N/A (research-only frame ledger).

## Research-only tag

`research_only=true` (per CLAUDE.md HNeRV parity discipline lesson 2). NOT yet a packetizable archive grammar.
