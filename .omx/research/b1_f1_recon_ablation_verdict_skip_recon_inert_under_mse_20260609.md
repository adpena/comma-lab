# F1 recon-ablation VERDICT — the PR95 skip is recon-INERT under pure MSE (H1-under-MSE FALSIFIED)

UTC 2026-06-09 · claude · `[macOS-MLX research-signal]` recon-PSNR (NOT d_seg, NOT contest score) →
`mechanism_update_eligible` only. A pivotal NEGATIVE result, recorded with the rigorous interpretation
(APPEND-ONLY per Catalog #110/#113 — does NOT mutate the F1-landed memo `b1_f1_bilinear_skip_canonical_
primitive_landed_20260609.md`, which PREDICTED the break; this row supersedes that prediction). DEFERRED,
not KILLED (Forbidden premature KILL — see "what this does NOT falsify").

## The result (recon-fit capacity probe, N=600, pure RGB-MSE)
| arm | config | plateau PSNR |
|---|---|---|
| control | skip-OFF, w=30 | 21.74 dB |
| A | skip-ON, **w=30** | 21.73 dB (flat ep50→100) |
| B | skip-ON, **w=1** (PR95-faithful) | 21.77 dB (flat ep50→150) |

**Both skip arms equal the skip-OFF control.** The PR95 bilinear-skip + terminal refine adds ~0 recon
capacity at EITHER frequency. The w=30 alias-trap hypothesis (H4) was real (arm A) but is NOT the whole
story — even at the coherent w=1, the skip does not move recon PSNR.

## What this FALSIFIES (narrow + clean)
**H1-under-MSE — "the missing residual HF path is the binding constraint on the carrier's RECON capacity"
— is FALSIFIED.** The 21.74 dB recon ceiling of the 229K decoder + 600 per-pair latents is NOT caused by
the missing skip; adding it changes nothing under pure MSE. So "patch the skip → recon breaks → d_seg
follows" (the F1-landed prediction + the deep-review H1) does not hold for our MLX carrier.

## What this does NOT falsify (the rigorous nuance — the real lesson)
The recon-fit probe optimizes **pure RGB-MSE**, whose global minimizer is the conditional MEAN (blur). MSE
does not reward the high-frequency boundary structure the skip enables — so MSE will choose the mean-field
REGARDLESS of whether the skip is present. **The recon-fit probe therefore isolates Mistake-A under an
objective that is itself Mistake-B-contaminated.** It cleanly shows the skip is inert *under MSE*; it does
NOT and cannot test the actual hypothesis of interest:

> **skip (architecture, Mistake-A fix) + nonzero scorer objective (Mistake-B fix) → low d_seg.**

That full hypothesis requires BOTH levers ON together. Neither F1 arm had the scorer objective; pact_nerv_vq
had the objective but no skip (→ d_seg≈0.5, Mistake-A binds GIVEN the objective). **The ONLY carrier that
tests skip + scorer-objective together is SNeRV-B** (Path B official conv U-Net WITH the residual skip +
the real-PoseNet/SegNet direct-live VJP objective, just launched, PID 42886). SNeRV-B's early exact d_seg
probe is the decisive test of the full hypothesis — not the recon-fit.

### Methodological lesson (system intelligence)
Mistake-A and Mistake-B INTERACT: a residual HF path only pays off under an objective that rewards HF.
Isolating Mistake-A "cleanly" under pure MSE is a category error — MSE suppresses exactly the signal the
architecture fix enables. The cheap recon-fit was the right FIRST probe (it falsified "the skip alone fixes
recon capacity" for ~$0) but the wrong probe to prove the skip's evaluator value. The clean Mistake-A
isolation needs the scorer objective ON (which is why SNeRV-B, not the recon-fit, is the real test).

## Strategic redirect (evidence-gated, per the Vehicle OS)
1. **Do NOT make "patch the skip into pact_nerv_vq/hi_nerv" the primary fix.** The reference-carrier
   comparison's RANK-1 (patch pact) was EXPLICITLY contingent on "the F1 ablation shows the skip breaks the
   plateau" — it did not. That contingency now routes to its documented fallback.
2. **SNeRV-B is the decisive running experiment** — the only carrier with skip + real-scorer objective. Its
   early TRUE d_seg probe (target 0.71→<0.2) is the next signal that matters most.
3. **The reference comparison's vendor path RISES to primary IF SNeRV-B also fails:** a clean ~300-500 LOC
   PR95-HNeRV MLX port (proven to reach ~0.193) rather than continuing to patch our 388 KB-scaffolded
   carrier whose recon ceiling the skip cannot move — the "deeper bug than the missing skip" branch.
4. **The 21.74 dB recon ceiling itself is now the open question** (independent of the skip): candidates are
   capacity (229K decoder / 28-d per-pair latents too small for 600 diverse pairs), coordinate conditioning
   (grid-PE OFF — F2), latent injection geometry, the optimizer, or an MLX-stack bug. Per the OS, prefer
   vendor-a-faithful-carrier over endlessly patching the sketch.

## OS bookkeeping
- HiNeRV `vehicle_fidelity_manifest`: `bilinear_skip` / `terminal_refine` = implemented-default-OFF, and now
  annotated **recon-inert-under-MSE; scorer-objective benefit UNTESTED**. The skip does NOT lift HiNeRV to
  L2 (it did not pass an intrinsic-optimization bar). DEFERRED-pending-SNeRV-B (the skip+objective test).
- The canonical `bilinear_skip_residual_canonical` / `terminal_hf_refine_canonical` kernels remain valid +
  tested primitives (the FINDING is about their effect under MSE, not their correctness).
- Authority: recon PSNR is `[macOS-MLX research-signal]`; this updates MECHANISM routing only, never the
  score roadmap.

## Cross-refs
`b1_f1_bilinear_skip_canonical_primitive_landed_20260609.md` (the superseded prediction) ·
`reference_carrier_comparison_20260609.md` (the patch-vs-vendor fork this result resolves toward vendor) ·
`principled_frequency_basis_synthesis_20260609.md` (w is still arbitrary; but the skip's inertness shows the
basis problem is deeper than frequency) · `snerv_fullstack_extreme_scrutiny_*` + SNeRV-B (the decisive
skip+objective test).
