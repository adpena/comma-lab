# Paper disposition: Strong Stochastic Flow Maps

UTC: 2026-07-20  
lane: `lane_einstein_kolmogorov_crux_20260719`  
research_only: `true`  
pointer: `0.1910828242 [contest-CPU Linux x86_64]`, UNMOVED  
review_status: recovery-written-UNREVIEWED-BY-MAIN

## Citation and intake custody

Sam McCallum, Zander W. Blasingame, Timothy Herschell, Niklas Rindtorff,
Alexander Tong, and James Foster (2026), *Strong Stochastic Flow Maps*,
arXiv:2606.01086v1. The arXiv title, authors, identifier, and 34-page PDF were
resolved and checked directly. The methods, proofs, training appendix,
experimental appendix, and limitations were read; this disposition is not
based on the abstract.

The prior not-relevant/watch ledger was consulted first. It had no entry for
arXiv:2606.01086. That ledger is outside this worktree's write authority, so
this append-only repository memo is the durable row MAIN should mirror.

## Method actually imported

The paper learns the strong solution, or Itô, map of an additive-noise SDE from
an initial state and a realization of the Brownian path. Its construction has
three load-bearing pieces:

1. an Euler-Maruyama-shaped finite-time map satisfying the diagonal tangent
   condition;
2. a semigroup self-distillation loss whose exact minimizer is the Itô map
   under the stated regularity and uniqueness assumptions; and
3. a shifted-Legendre representation of Brownian motion with independent
   Gaussian coefficients, closed-form Chen relations, and almost-sure
   convergence in alpha-Holder rough-path distance for alpha below one half.

The empirical claim is a few-step stochastic generator for images and
molecular conformations. It is not an inverse scorer, integer-lattice solver,
rate allocator, or deterministic archive codec.

## Divergence from the frozen contest problem

**VERDICT: NOT-A-DECODE-LEVER; COMPRESS-TIME WATCH ONLY.** The paper's useful
object is a learned stochastic solution map. The contest decoder must instead
be deterministic and bit-exact, may not ship scorer weights, and must charge
all video-derived learned state. Brownian coefficients would be counted
random-path data and would make byte identity path-dependent. The learned
network weights would also be counted unless they were genuinely generic;
neither condition closes the current 216--244 KB distortion-dependent box. No stochastic generator
was built or measured.

Two ideas survive the assumption fork:

- **Strong versus weak state recovery is a conceptual guard.** Matching the
  distribution of scorer outputs is insufficient; the receiver must recover
  the declared uint8 state through `R`, just as a strong solution is tied to
  one driving path rather than only a marginal law. This restates the existing
  realized-through-R rule and creates no new authority.
- **A deterministic flow-map accelerator is a compress-time watch item.** If a
  level-set/PDE solve becomes a measured encode-time bottleneck, one may learn
  or derive its deterministic solution map and validate it against the exact
  terminal state. That would collapse integration steps during compression,
  not enter `inflate.py`. The present Einstein--Kolmogorov blocker is counted
  numerator/description rate, not solver wall time, so this transfer has zero
  current expected score value.

## Measured disposition

- **MEASURED:** no new archive, `d_seg`, `d_pose`, or byte result came from this
  paper.
- **DERIVED:** stochastic decode conflicts with deterministic byte identity and
  counted learned-state custody.
- **INFERRED/WATCH:** deterministic solution-map acceleration could reduce
  compress time only if a future profile makes integration dominant.
- **UNKNOWN:** whether a generic, scorer-free deterministic map can ever be
  shared across clips without paid video-derived weights.

No code path, launch, or pointer consumer is authorized by this paper.
