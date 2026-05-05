---
name: Jacobian Nonlinearity of PoseNet
description: PoseNet's honest linear trust radius is ~0.0001 pixels, its Jacobian is effectively rank-1, and our winning CNN spreads corrections densely in mid-frequency space — all measured 2026-04-09
type: project
---

Measured 2026-04-09 during mathematical investigation of the 1.845 floor.

**Trust radius**: PoseNet's relative linearization error exceeds 50% even at
perturbation magnitudes of 0.0001 pixels (measured via random-direction sweep
across 10 frame pairs × 3 directions × 11 alpha values). The knee of the
`||exact - linear|| / ||exact||` curve is at or below the smallest alpha we
tested. Single-step Newton, quasi-Newton, or any closed-form correction is
therefore mathematically dead on arrival — step sizes must be below 0.0001
pixels per iteration, making 10^4+ iterations required to move 1 LSB.

**Jacobian rank**: the per-pair Jacobian J = dPose/dPixel has effective rank
~1.008 across 30 sampled pairs (measured via entropy of normalized squared
singular values). Top singular value is 45x larger than the second. 98% of
PoseNet's pixel sensitivity lies along ONE direction per frame. Condition
number is ~399.

**CNN residual structure**: our winning h=32 long-1000 QAT+EMA CNN has a very
different correction strategy from the theoretical minimum-norm delta:

- CNN moves 56.6% of pixels (Jacobian linear solution moves 0.0024%)
- CNN mean absolute correction is 0.83 (Jacobian 0.0044)
- CNN puts 90.3% of luma residual energy in the mid-frequency DCT band
- Jacobian linear solution is roughly uniform across frequency bands
- The CNN's strategy is "dense, large-amplitude, mid-frequency biased"

**Why**: PoseNet is piecewise linear in theory (ReLU) but the piece boundaries
are so close to every input that any correction crosses many of them. A
single-step delta concentrated on a handful of pixels blows through ReLU
boundaries and lands in a different linear piece, invalidating the Jacobian
that was computed at the starting point. The CNN's dense mid-frequency
strategy stays inside the ReLU regions because small residuals applied at
every pixel stay numerically close to the linearization point.

**Consequence**: any future optimization idea that requires "apply a closed
form derived from J" is dead. The path forward is:

1. Better CNN training (longer, wider, better EMA, scorer-faithful losses)
2. Parametrizations that bias toward mid-frequency corrections (DCT basis)
3. Gradient-free search in weight space (CMA-ES)
4. SegNet attack — it is a separate non-linearity with its own leverage
