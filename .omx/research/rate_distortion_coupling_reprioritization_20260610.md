# Rate–distortion coupling: the reprioritization (operator insight, 2026-06-10)

Operator: "as long as distortion is not zero there is still meat on the bone, but the rate term seems
necessary but not sufficient." Both exactly right; this reprioritizes the offensive. Binding input to
Lever F (floor) and the offensive-research lever.

## The exact decomposition of the current frontier (0.19109982 [contest-CPU], recoded-R3, 177,169 B)
- rate  = 25·177169/37,545,489 = 0.117970  (61.7%)  — necessary constraint, lossless-FLOORED
- d_seg = 100·0.00055978       = 0.055978  (29.3%)  — LINEAR, the dominant RECOVERABLE pool
- d_pose= sqrt(10·0.00002942)  = 0.017153  ( 9.0%)  — sqrt: steep marginal near zero; E3 cross-pair fungible
"If rate→0" floor of THIS representation = d_seg+d_pose = 0.0731 — i.e. even a free archive of the
current carrier is 0.073; the representation, not the coder, sets that.

## The two principles
1. **Rate is necessary-not-sufficient.** 62% of score but lossless-exhausted (latents at per-dim
   marginal, cross-pair MI=0; decoder 98.6% iid Shannon — cite before any rate re-attempt). Further
   rate cuts COUPLE into distortion (smaller bytes ⇒ worse reconstruction unless the representation
   class changes). Treat rate as a CONSTRAINT TO HOLD (no bloat), not a currency to chase. Pure-rate
   wins are noise-margin + non-innovative (PR#112).
2. **Distortion is the unbounded meat while nonzero.** d_seg LINEAR + dominant ⇒ halving 5.6e-4→2.8e-4
   = −0.028 score (≈1000× the recoded-R3 rate win). d_pose sqrt ⇒ small now, steep marginal toward 0.

## The reprioritization (changes the lever ranking)
- PRIMARY currency = d_seg reduction (linear, biggest pool), then d_pose. NOT rate.
- The ONLY move that lowers rate AND distortion together = a representation CLASS SHIFT that dominates
  the R–D curve (the offensive levers A–F). This is the mathematical reason the innovation gate IS the
  score path, not a values nicety.
- d_seg's cheapest mechanism = WEIGHT-SPACE (decoder amortizes a weight change across all pixels at
  ZERO per-flip byte cost). This is why the Class-3 sidecar failed (1.525 B/flip > break-even) but a
  trained decoder pays nothing per flip — the AFSR-1 paradigm (fresh-init, null-space-primary) is the
  d_seg lever, not the sidecar.

## Consequences for the running levers
- **F (floor)**: T_floor is a JOINT rate–distortion floor, NOT a byte floor. The current 177KB archive
  is a SUBOPTIMAL point OFF the achievable R–D frontier (it spends 0.118 rate to hold 0.073 distortion;
  a class shift can move to lower-rate AND lower-distortion simultaneously). Derive the floor as the
  R–D frontier's minimum of S, and report how far 0.191 sits ABOVE it (the headroom is the gap to the
  frontier, not just to zero bytes).
- **Offensive research**: rank d_seg-reducing class shifts (weight-space / score-native / quotient)
  ABOVE any rate play; rate plays are stack-on constraints, not primary. The #1 lever should be the one
  with the steepest d_seg/effort, because d_seg is linear and dominant.
