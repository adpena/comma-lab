# Papers checked — adversarial dynamical systems and information-model lower bounds (2026-07-20)

research_only: `true`  
axis: `[theory intake; no score axis]`  
lane_id: `lane_einstein_kolmogorov_crux_20260719`  
verdict_scope: lower-bound hygiene for the Einstein--Kolmogorov crux; no codec or score result

## Primary source checked

Matthew J. Colbrook, Igor Mezic, and Alexei Stepanenko (2026), *Adversarial
dynamical systems characterize when data-driven learning succeeds or fails*, Nature
Communications 17, 5397, DOI `10.1038/s41467-026-74220-8`, published 2026-07-14.

- Article: `https://www.nature.com/articles/s41467-026-74220-8`
- Supplementary information:
  `https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-026-74220-8/MediaObjects/41467_2026_74220_MOESM1_ESM.pdf`
- Supplement checked end-to-end: 36 pages, 7,448,874 bytes, SHA-256
  `750fa7aa6b3f27cc8dc972af993c7281559bdd0c50f95ed303142a71880ec9dd`.

## What transfers

The paper's transferable result is methodological. A computability or lower-bound
claim must first declare the computational problem: the admissible primary class,
target map, metric, and allowed evaluation set. Its negative results then construct
systems that remain indistinguishable to every finite set of allowed evaluations while
the target spectrum or spectral type changes. The associated upper bounds use explicit
towers of limits, matrix pencils, pseudospectral distance functions, and verification
sets; they do not infer learnability from finite-data fit alone.

For this lane, that gives a strict review guard:

1. A positive instance `K`-floor requires a declared scorer-quotient source class,
   target functional, metric, and allowed observations.
2. It must additionally provide either a contest-specific counting/entropy theorem or
   an adversarial indistinguishability construction that preserves those observations
   while changing the minimum legal program length.
3. Otherwise the numerical instance floor remains `UNKNOWN` (with only the trivial
   `K>=0`), regardless of how compelling an n24 optimization trace looks.
4. An n24 Seg-only component receipt cannot identify the best n600 joint Seg/Pose/rate
   family because Pose and full-archive observations are outside that receipt's
   evaluation set.

## What does not transfer

The frozen contest scorer, receiver, and finite source are fully specified; this lane is
not learning the spectrum of an unknown infinite-dimensional dynamical system from
snapshots. The paper proves no byte lower bound, no Kolmogorov-complexity value, no
archive-size cap, and no ordering among the palette, topology, or xi families. Koopman
spectral algorithms, EDMD conditioning diagnostics, and the paper's Solvability
Complexity Index classifications are therefore `NON_LEVER_FOR_CURRENT_BYTES`.

Disposition: `RELEVANT_LOWER_BOUND_GUARD_ONLY`. No implementation arm, paid dispatch,
score claim, or frontier-pointer change is authorized by this source.
