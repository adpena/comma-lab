# MAIN adversarial review of hpr1's R1 "DO-NOT-FIRE" verdict (2026-09-11 ~19:40Z) — relative significance stated, verdict scope INSTANCE

# FORMALIZATION_PENDING: a review note on a single arm verdict; the equations leg is the score identity S = 100·d_seg + sqrt(10·d_pose) + 25·B/37,545,489 already registered.

Operating point: move 45, S 0.1371383667388406; remaining gap to sub-0.12 = 0.0171383667.

| object | ΔB (exact) | ΔS (rate only; distortion unchanged, output-lossless) | ΔS / remaining gap |
|---|---:|---:|---:|
| R1 (conv_past dilation 2, retrained 60 ep) vs move 45 | −75 B | −4.994e-5 | **0.29 %** |
| CONTROL (shipped geometry, retrained 60 ep) vs move 45 | **−887 B** | **−5.906e-4** | **3.45 %** |
| R1 vs its own control (the shape leg isolated) | **+812 B** (Δmodel +12, Δtail +800) | +5.41e-4 | −3.16 % (a loss) |

Review. R1 is not dismissed on magnitude: it clears the −2e-5 bar (2.5×) and its 0.29 % of the gap would be banked under the
byte-closed-rows-at-cadence law if it stood alone. It is dismissed on DOMINANCE: the control reaches the same object with 812 B
fewer bytes, no receiver change (normal seal instead of the first-measurement chain), and the same output-lossless proof, so firing
R1 would bank the retrain's win while paying the shape's loss. The correct action is to fire the CONTROL, and hpr1 says so ("the
control is FIRE-ELIGIBLE"). Verdict scope: INSTANCE — one shape rung (spatial dilation of conv_past) is falsified at exact bytes;
hpr1's own atlas says the unexploited information is on the TEMPORAL axis (every temporal set's MDL net is positive), so the shape
family is not closed by this rung. Un-recoverability is not claimed for R1; it is superseded, which is a different exit.
Consequence for sequencing (MAIN): the control and ntb2's frame_even (−245 B, move-46 intent in flight) are two edits to the same
HPAC prior (rounding vs replacing the frame embedding) and do not add; the control subsumes the rounding, and hpr1 states the
rounding can be re-applied on top of the retrained prior. Sequence: move 46 = frame_even (first-measurement chain, gives a measured
leg); move 47 = the retrained control rebased onto move 46 (same token field and carrier; normal seal inheriting move 46's leg),
then the rounding re-applied on top as a further row if it still prices negative.
