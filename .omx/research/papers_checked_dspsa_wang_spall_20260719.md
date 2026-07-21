# Papers checked — bounded DSPSA for the PDW1 palette lattice (2026-07-19)

research_only: `true`  
axis: `[macOS-CPU advisory]`  
lane_id: `lane_einstein_kolmogorov_crux_20260719`  
verdict_scope: method intake for a 15-dimensional per-pair uint8 fill search; no general optimizer ranking

## New primary sources checked

1. James C. Spall (1992), *Multivariate stochastic approximation using a simultaneous
   perturbation gradient approximation*, IEEE Transactions on Automatic Control 37(3),
   DOI `10.1109/9.119632`. Full method source resolved and checked. Imported result:
   one Bernoulli simultaneous perturbation obtains a full-dimensional stochastic
   direction from two objective evaluations. Disposition: `RELEVANT_METHOD_ANCESTOR`,
   but continuous SPSA alone is not the integer algorithm used here.
2. Qi Wang (2013), *Optimization with Discrete Simultaneous Perturbation Stochastic
   Approximation Using Noisy Loss Function Measurements*, arXiv `1311.0042`, Chapter 2.
   Full chapter derivation and bounded algorithm checked. Imported result: maintain a
   real iterate, evaluate the two integer corners around the half-integer middle point,
   project to the bounded domain, and preserve explicit gain schedules. Disposition:
   `RELEVANT_IMPLEMENTED_TOURNAMENT_ARM`.

## Local substitution and limits

- Domain substituted: `[0,255]^15` fill bytes for each fixed-label PDW1 pair.
- Objective substituted: exact singleton CPU-Torch SegNet hard mismatch after factor-2
  uint8 realization, not a differentiable proxy.
- Determinism added locally: perturbations are derived from the typed seed and iteration;
  every iteration has an immutable checkpoint; a second full same-seed n24 run must
  reproduce candidate bytes and pair rows exactly.
- The convergence hypotheses in Wang are not established for this frozen nonconvex
  scorer composition. DSPSA is therefore a measured candidate family, never a proof of
  global optimality.

This file is the worktree-local append-only paper-ledger update. The separately named
Claude memory ledger is outside this managed worktree's write authority; MAIN may mirror
these two rows there after landing review.
