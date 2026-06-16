# Production-readiness "bind ALL ingredients" spec — small-basis Track-A sub-0.15 run

**Operator 2026-06-16: "does this include all final production techniques such as QAT and all" +
"dig deep on quantization, PR95, final rate attack, layer 3, and all" + "score > training time
always."** This is the canonical bind-all checklist for the FINAL production run, per CLAUDE.md
"HNeRV / leaderboard-implementation parity discipline" (bind architecture + score-aware training +
archive grammar + inflate runtime + export + curriculum + QAT + rate-attack + L3 into ONE package).
`[contest-CPU advisory]` until byte-closed dual exact eval; pointer 0.19110 UNMOVED.

## Status table (audited from the code, 2026-06-16)
| Technique | Status | Note |
|---|---|---|
| eval_roundtrip (bicubic↑874→bilinear↓→uint8-STE) | ✅ always-on | 1:1 port of contest `common.py` (driver L1174-1179) |
| EMA (+warmup) | ✅ on | `ema_warmup` hook |
| QAT (score-aware Lever-4 + uniform fallback) | ✅ in refinement stage | `spec.use_qat` L1140; only the SINGLE refinement stage, not progressive |
| Muon (final-stage orthogonalized) | ✅ in refinement | `use_muon` |
| oomph sharp soft_cosine seg lever | ✅ | the d_seg lever (T0.3→0.05+renorm+tight τ) |
| FiLM-v2 pose decouple + pose cadence | ✅ | + APGC controller (bc448da84); per score>time deploy **k=1 (pose every epoch)**, APGC=safety net |
| PR95 codec grammar (build_archive/parse_archive/cat_entropy_v2) | ✅ vendored | G2-proven byte-close (b4e42061c) |
| solved byte-neutral taper [22,16,15,14,15,14,10] | ✅ solved+verified | 7dd5b5188; −0.55% bytes; feeds the d_seg-critical HIGH band |
| **Full PR95 8-stage curriculum** (τ/smooth/progressive-QAT/C1a-buildup/λ-sweep/σ-sweep) | ❌ not run | basin is `stage1_v328_ce` only; 29,650-ep from-scratch is ~40-60d on MPS → INFEASIBLE locally |
| **Final rate attack** (variable-level codec / Lever-4 sensitivity levels) | ◐ wired, GATED-OFF | `variable_level_waterfill_enabled=False` → ENABLE for production |
| **L3 distortion finishing-kit** (PR98 bias / T10 affine / S12 mask / Lever-D) | ◐ wired, GATED-OFF | `distortion_finishing_kit=None` → ENABLE/apply for production |
| **L3 rate recodes** (D1 cross-pair latent dedup, D2/3/4) | ❌ #106 partial | D1 (biggest) unbuilt |
| **Hungriest-tensor / boundary head** | ❌ not built | rgb_1 is the OUTPUT layer (sensitivity≠under-capacity); taper already feeds its upstream band; see below |

## The wall-clock resolution (KD-warm-start — the key enabler)
The full 29k-epoch PR95 curriculum from-scratch is infeasible on MPS. The basin (stage-1 CE) +
rich refinement (QAT+Muon+C1a+σ) captures most of the curriculum's VALUE via warm-start — but the
solved taper's different channel shapes block warm-starting the vendored basin. **Resolution:
KD-warm-start** — distill the converged vendored-taper basin (teacher) into the solved-taper
student (fast convergence to a solved-taper basin), THEN the rich refinement with all levers. This
binds QAT/oomph/FiLM/taper/rate/L3 without the months-long from-scratch curriculum. (Alternative:
paid GPU for the full curriculum — $ + days; reserve for the final byte-closed exact eval, G3.)

## The bind-all production config (the run that aims at sub-0.15)
1. **Architecture:** solved taper [22,16,15,14,15,14,10] (winner of the cheap A/B vs vendored).
2. **Warm-start:** KD from the vendored-taper basin → solved-taper student (recovers the head-start).
3. **Refinement (the levers, all on):** oomph sharp soft_cosine seg + FiLM-v2 pose + **pose every
   epoch (k=1; APGC tight as safety)** + progressive-QAT + Muon + C1a coder-aware reg + σ noise.
4. **Always-on:** eval_roundtrip + EMA-warmup.
5. **Rate attack:** ENABLE variable-level codec (Lever-4 sensitivity levels) + the full PR95 entropy
   grammar (per-tensor maps / split brotli / raw-LZMA latents / temporal-delta / fp16 scales / q=11).
6. **Layer-3:** apply PR98/T10/S12/Lever-D finishing-kit (post-round, where each pays) + D1 latent dedup.
7. **Pose:** the real knob is λ_pose + completing the FiLM-v2 TRUNK decoupling (orthogonal pose),
   NOT cadence — cadence is a time-proxy (per [[score-over-training-time-always-pose-throttle-is-score-negative]]).
8. **Close:** byte-close (G2-proven) → dual CPU/CUDA exact eval (G3) → pointer.

## Hungriest tensor (rgb_1, sensitivity-density 3.88) — optimal handling
rgb_1 is hungriest largely because it's the OUTPUT layer (writes the scored pixels) — perturbation-
sensitivity of the output is expected and ≠ under-capacity (the review's perturbation-asymmetry).
Optimal, ranked: (1) feed its UPSTREAM high-res band (stage4/5) — the solved taper ALREADY does
this; (2) train its boundary harder (oomph + boundary-weighted head loss, 0 bytes); (3) if measured
capacity-starved, BREAK THE QUADRATIC (depthwise-separable refine, O(C) not O(C²)) rather than widen
final; (4) boundary-conditioned low-rank head residual (spend capacity only at the ~440 contested
argmax pixels; ties to Lever-D + boundary-math seg core #52). `--final-cap 11` is a cheap EMPIRICAL
check, NOT the expected win (output-layer sensitivity ≠ under-capacity).

## WIRED + INTEGRATED + OPTIMAL + SYNERGISTIC (operator 2026-06-16) — the system, not the parts
The ingredients are not independent bolt-ons; built right they REINFORCE. The architecture:

**The shared spine — ONE sensitivity map drives the whole stack.** The gate-2 d_seg-sensitivity
map (per-tensor Δd_seg/param, measured on the converged tapered model) is the SAME signal that
should drive, computed ONCE and fanned out (no triple-recompute):
- **taper** (allocate CHANNELS by sensitivity — done: [22,16,15,14,15,14,10]),
- **QAT / Lever-4** (allocate BITS by sensitivity — `levels_from_sensitivity_for_codec`, the SAME rank-norm band),
- **variable-level codec / rate-attack** (allocate quant LEVELS by sensitivity),
- **boundary head loss + L3 PR98/Lever-D** (allocate PIXEL weights by boundary saliency).
⇒ wire `tac.sensitivity_map` (or the gate-2 probe output) as the single shared input to taper +
QAT + codec + head. This is the synergistic core: capacity, bits, levels, and pixel-weights all
follow one coherent d_seg-geometry, instead of four separately-tuned heuristics.

**Synergy table (where pieces reinforce):**
| pair | synergy | integration requirement |
|---|---|---|
| taper ↔ QAT/Lever-4 ↔ rate-codec | all allocate by the SAME sensitivity | re-measure sensitivity on the TAPERED model; feed all three |
| oomph(seg) ↔ FiLM-v2(pose) | FiLM carries pose orthogonally → oomph cranks seg with NO pose cost | **complete the FiLM-v2 TRUNK decoupling** (head-only today → the synergy LEAKS → the measured pose drift) |
| pose-every-epoch ↔ FiLM-v2 | tight pose hold AND cheap (orthogonal) | k=1 + complete decoupling |
| eval_roundtrip ↔ QAT ↔ L3-PR98 | jointly close the train→deploy gap (resize / weight-quant / channel-bias) | PR98 offset applied POST-round (driver L2428 ✓) |
| KD-warm ↔ basin | basin knowledge → re-tapered student (recovers head-start) | KD loss composes with the score-aware loss (inherit fidelity AND score-awareness) |

**Optimal-as-a-whole (joint, not greedy):** the whole stack minimizes ONE objective
S = 100·d_seg + √(10·d_pose) + 25·B/N. The taper (d_seg, byte-neutral), oomph (d_seg via loss),
FiLM+pose-every-epoch (hold d_pose), QAT (preserve d_seg/d_pose under quant), rate-attack+L3
(min B + recover quant distortion) are coupled through S. The Lagrangian λ's + the shared
sensitivity spine are the coordination mechanism — solve jointly, not lever-by-lever. **The
byte budget is JOINT:** taper (channels) + rate-attack (levels) together set the final archive
bytes → verify byte-neutrality at the POST-int8-brotli archive (review R4), not param-count.

**The integration that unlocks the biggest synergy: complete the FiLM-v2 trunk decoupling.** The
measured pose drift IS the symptom of the incomplete oomph↔FiLM synergy (FiLM decouples the rgb_0
head but the shared trunk still couples pose↔seg). Completing it (carry pose fully via the FiLM
path + stored scalars so the trunk is FiLM-clean for d_seg) makes oomph and pose REINFORCE instead
of compete — the single highest-leverage integration for the score-optimal pose stack.

## Gaps to close before production (turn into tasks)
- Build the KD-warm-start (basin teacher → re-tapered student) + composed refinement actuator.
- Enable + verify the variable-level codec rate attack (post-int8-brotli byte-neutral check).
- Enable/apply the L3 finishing-kit + build D1 latent dedup.
- (optional) boundary-conditioned head arm if the A/B shows the head is the binding lever.
