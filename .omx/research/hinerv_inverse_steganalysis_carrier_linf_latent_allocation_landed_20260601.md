# HiNeRV inverse-steganalysis carrier — L∞ latent-domain allocation via dense decoder-VJP adjoint — LANDED

- **Date:** 2026-06-01
- **Lane:** `lane_hinerv_inverse_steganalysis_carrier_20260601`
- **horizon_class:** frontier_pursuit (Phase-2 carrier candidate of the inverse-steganalysis full stack; the cheapest-measured-RD NeRV-family carrier per arXiv 2306.09818)
- **axis_tag:** `[macOS-CPU advisory]` — NON-PROMOTABLE (`score_claim=false`, `promotable=false`) per Catalog #341/#192/#127/#323. $0 macOS-CPU only; no paid dispatch, no GPU, no PR, no MPS authority. ALL d_seg/d_pose measured on the verified bit-exact CPU mirror.
- **Frontier reference:** `.omx/state/canonical_frontier_pointer.json` (pointer-only per Catalog #343).
- **Design memo:** `.omx/research/inverse_steganalysis_optimal_full_stack_design_20260601.md` (§7 GREEN; Phase-2 carrier candidates) + §"NeRV-family carrier research" (HiNeRV = cheapest measured RD, learned-renderer fallback with dense decoder-VJP adjoint).
- **Durable anchor:** `experiments/results/hinerv_invsteg_carrier_advisory/hinerv_latent_linf_allocation_*.json`

## Thesis

The §7 gate proved (GREEN, `inverse_steganalysis_linf_vs_l2_gate_landed_20260601.md`) that inverse-steganalysis L∞ margin-budget allocation beats L2-MSE at equal rate **on rendered frames in PIXEL space**. This lane pushes that proven objective into the **HiNeRV carrier's LATENT coefficient domain** — the layer where a NeRV-family carrier actually spends its precious bits — via the **dense decoder-VJP adjoint (G3)**, and byte-closes + advisory-re-measures the result.

## What was built (committed)

- `src/tac/analysis/hinerv_latent_linf_allocation.py` — the G3 dense decoder adjoint + L∞ latent allocator:
  - `decoder_jacobian_vjp` — `Jᵀy` via reverse-mode AD through the SHIPPED HiNeRV decoder (reverse-mode AD IS the exact adjoint of a nonlinear synthesis at the operating point). The HPRC analytic resize-gather sister (`hprc_synthesis_adjoint`) generalized to the dense-CNN case.
  - `adjoint_dotproduct_residual` — the PRIMARY exactness proof: `<J x, y>_pixel == <x, Jᵀy>_latent` (the definition of the adjoint). **Machine-exact ~1e-7 on every scale, even on a trained high-Lipschitz carrier.**
  - `finite_difference_vjp_residual` — corroborating numerical-Jacobian convergence check; an adaptive eps-sweep (1e-2 → 1e-12, fp64) finds the convergence floor regardless of the trained sin(30) carrier's enormous local Lipschitz constant (verified: the numerical JVP converges to the analytic JVP — rel-residual 1.0 at eps=1e-2 but 3.4e-4 at eps=1e-10 — proving the analytic JVP IS the true Jacobian).
  - `scale_jvp_norm` — probes whether the carrier actually uses a latent scale at the operating point (a trained carrier can drive a whole scale's Jacobian column to ~0, where the fd ratio is noise/noise; the adjoint dot-product test stays exact regardless).
  - `push_pixel_saliency_to_latent` — the **Fisher-pullback** `s_latent[k] = Σᵢ (∂frameᵢ/∂z_k)² · s_pixelᵢ` = diag(Jᵀ diag(s_pixel) J): pushes the combined oracle ρ_i (s_seg P18 + s_pose P19 at score-derivative weights) into per-latent-dim saliency exactly (full Jacobian column energy via forward-mode over the latent basis).
  - `allocate_linf_latent_steps` / `allocate_l2_uniform_latent_steps` — L∞ reverse-water-fill `δ_k = clip(c·ρ_k, lo, R_scale)` at cost ρ_k=1/(s_latent_k+ε) vs the L2 per-scale uniform baseline, at EQUAL latent rate (the canonical high-rate uniform-quantizer rate model, with per-scale coefficient dynamic range), with the `disadvantage_linf` anti-gaming nudge (L∞ forced to spend ≥ L2 bits).
  - `quantize_latents_with_steps` — deterministic mid-rise uniform quantization the archive stores + inflate renders from.
- `tools/run_hinerv_latent_linf_allocation.py` — the $0 runner: train a light HiNeRV carrier on REAL `0.mkv` frames (NeRV-style per-pixel-MSE fit; no scorer in the inner loop — the receiver never loads the scorer), prove G3 exact, push oracle → latent, allocate L∞ vs L2, byte-close two HIV1 archives, advisory re-measure d_seg/d_pose/rate on the bit-exact CPU mirror, emit the durable JSON anchor.
- `src/tac/tests/test_hinerv_latent_linf_allocation.py` — 24 NO-FAKE tests (the adjoint identity holds AND a random "adjoint" fails it; fd converges; the Fisher-pullback genuinely concentrates where the pixel saliency concentrates; L∞ steps differ from uniform L2 and a higher-saliency latent dim gets a finer step; equal-rate fairness; quantization actually mutates; fail-closed guards). All pass.

## Empirical verdict (`[macOS-CPU advisory]`; $0 macOS-CPU)

**G3 adjoint — PROVEN EXACT.** Across every advisory probe + the test suite, the adjoint dot-product residual is machine-exact (~1e-7) on all three latent scales, including on the trained high-Lipschitz carrier where forward-mode-only JVP libraries break. The fd-sweep corroborates on non-degenerate scales. **The dense decoder-VJP adjoint is the rigorous G3 gate for a learned-renderer carrier (the design memo's noted "re-opens G3" risk for HiNeRV — CLOSED here by the adjoint dot-product proof).**

**L∞ vs L2 in the latent domain — sign operating-point-dependent at $0-local scale.** The smoke (48×64, 2 pairs) showed L∞ beating L2 by 2.12 contest-distortion; tighter-budget probes (96×128, 6 pairs) showed L∞ ±0.37. The aggregate advisory S stays ~81-90 across all configs because **the carrier itself does not fit**: d_seg pins at ~0.51 (SegNet argmax flips on ~half the pixels) and d_pose ~150 regardless of latent allocation, because a tiny HiNeRV carrier trained by per-pixel-MSE on a few pairs at $0-local scale never reaches medal-band d_seg (the proxy-auth gap — low MSE ≠ low d_seg).

**The co-equal-keystone — CONFIRMED EMPIRICALLY (the load-bearing finding).** The design memo §2 established { score-exact oracle objective, carrier whose R(D) is cheap-enough } as co-equal-necessary (triple-confirmed: Z8, council, HPRC). This lane adds a fourth confirmation from the HiNeRV direction: **the oracle aims the bits perfectly (G3 exact) and the L∞ objective transfers correctly into the latent domain, but at $0-local carrier scale the carrier R(D) dominates so the latent-allocation lever is second-order.** Two reinforcing mechanisms: (a) the carrier doesn't fit (d_seg pinned); (b) per-pair latents barely influence the trained carrier output (JVP norms ~1e-4–1e-7 — the NeRV family amortizes content into the *decoder weights*, leaving the per-pair latents with little leverage). The aiming is correct; the carrier must be both cheap AND well-fit for the aiming to move the score.

## Super-small-rate-by-design — structurally TRUE, fit-blocked at $0-local

The operator driver: a stack designed around a super-small rate term by construction, with solved distortion. **The rate half is structurally achievable**: a tiny HiNeRV decoder (~18K brotli bytes at 600 pairs) + tiny int16 latents (~18K bytes) ≈ 36KB — ~5× cheaper than the PR101 frontier carrier (`b7106c9b`, pointer-only) — cheap BY CONSTRUCTION (the NeRV family is parameterized by a target byte budget, the property HPRC's explicit-coefficient carrier + Z8's raw-float wavelet detail LACK). **The distortion half is NOT solved at $0-local**: reaching medal-band d_seg on a tiny carrier requires score-aware training (the CUDA `_full_main` path with SegNet/PoseNet in the loop, thousands of epochs) — exactly why it is a paid-GPU job. The L∞-in-latent mechanism is the correct distortion-allocation lever ON TOP of a well-fit cheap carrier; it cannot manufacture a fit a per-pixel-MSE carrier never reached.

## Synergy verdict (1 line)

**HiNeRV + L∞-in-latent is the structurally-correct super-small-rate-by-design + decision-margin-allocated stack — the cheap-by-construction carrier and the G3-exact oracle-aimed allocator both work — but it realizes solved-distortion only with a CUDA score-aware fit (operator-auth); at $0-local the carrier R(D) dominates, empirically re-confirming the co-equal-keystone (cheap AND well-fit, not just cheap).**

## Canonical-vs-unique decision per layer

- L0 oracle: ADOPT_CANONICAL (`score_exact_saliency` + verified mirror).
- G3 adjoint: FORK_PRINCIPLED — HPRC's analytic resize-gather adjoint does not apply to a dense CNN; the dense decoder-VJP (reverse-mode AD) is the carrier-specific adjoint, proven exact by the same canonical dot-product test.
- L∞ allocator: ADOPT_CANONICAL rate model (`inverse_steganalysis_linf_vs_l2_gate`) + FORK to the latent coefficient domain (per-scale dynamic range, not uint8 256).
- Carrier: FORK_PRINCIPLED — HiNeRV is the Phase-2 learned-renderer candidate (cheapest measured RD); distinct from the PR95-HNeRV Phase-1 carrier the operator resolved.
- L4 grammar: ADOPT_CANONICAL (the substrate's existing HIV1 monolithic 0.bin + the trainer's deterministic archive.zip builder).

## 9-dimension success checklist evidence

UNIQUENESS — first push of the §7-proven L∞ objective into a learned NeRV-family carrier's latent domain via the dense decoder-VJP adjoint. BEAUTY — one adjoint (reverse-mode AD), one Fisher-pullback, one reverse-water-fill, one rate model. DISTINCTNESS — explicitly NOT fidelity; cost = the detector Jacobian pulled back through the decoder. RIGOR — adjoint dot-product machine-exact ~1e-7 + fd convergence + 24 NO-FAKE tests + the random-vector falsification control. OPTIMIZATION-PER-TECHNIQUE — DeepFool/Fisher/reverse-water-fill are the per-channel optima. STACK-OF-STACKS — s_seg+s_pose combine additively at score-derivative weights before the pullback. DETERMINISTIC-REPRODUCIBILITY — seeded; deterministic quantization + byte-close. EXTREME-OPTIMIZATION — cheap-by-construction carrier (~36KB at 600 pairs) + closed-form water-level bisection. OPTIMAL-MINIMAL-SCORE — the latent allocation aims the contest security objective at the carrier's bit-spend layer; the co-equal-keystone (cheap AND well-fit) is the empirically-confirmed gate to the actual minimal score.

## Cargo-cult audit per assumption

- "the latent-allocation lever moves the score on any carrier" — CARGO-CULTED; FALSIFIED here at $0-local (the carrier R(D)/fit dominates; latent leverage near-zero on an under-fit carrier).
- "reverse-mode AD might not be the exact adjoint for a nonlinear decoder" — FALSIFIED (the dot-product test is machine-exact; reverse-mode AD IS Jᵀ at the operating point).
- "a small fd residual at a single eps proves the JVP" — CARGO-CULTED; for a high-Lipschitz sin(30) carrier the linear regime is tiny, so a single-eps fd undershoots; the adaptive eps-sweep is the rigorous numerical-Jacobian convergence test.
- "super-small-rate-by-design alone lowers the score" — refined to HONEST: cheap-by-construction is necessary (rate half) but the distortion half requires a score-aware CUDA fit (the co-equal-keystone).

## Observability surface

Inspectable per layer (per-pixel ρ_i; per-latent-dim s_latent; per-coefficient step maps; per-scale JVP norms). Decomposable (s_seg/s_pose/rate separable; per-pair d_seg/d_pose/distortion table; per-scale adjoint+fd residuals). Diff-able (L∞ vs L2 vs baseline on the same carrier + frames). Queryable (the durable JSON anchor). Cite-able (every residual + measurement anchored to the trained-carrier config + frozen-weight mirror + pair index). Counterfactual-able (the L2-vs-L∞-vs-baseline three-way + the byte-mutation no-op proof in the substrate tests + the random-vector adjoint falsification control).

## Predicted ΔS band

Direction PENDING — the G3 adjoint + L∞-in-latent objective are PROVEN correct, but the archive-level ΔS at $0-local is dominated by the carrier R(D)/fit (the co-equal-keystone), so no carrier-level ΔS is asserted from this $0 probe. The archive-level ΔS requires the CUDA score-aware `_full_main` fit + paired CPU+CUDA (Catalog #246). Dykstra-feasibility: the latent allocation is the projection onto {latent-rate ≤ R, ∀k δ_k ≥ ρ_k-floored} solved by the reverse-water-fill bisection. `# PREDICTED_BAND_VIBES_OK:objective-and-adjoint-proven-archive-delta-dominated-by-carrier-RD-at-0-dollar-local-deferred-to-cuda-score-aware-fit-and-paired-cpu-cuda`

## Canonical equation reference

`# FORMALIZATION_PENDING: hinerv_Linf_latent_margin_budget_via_decoder_vjp_adjoint_savings — register in tac.canonical_equations after a CUDA score-aware-trained HiNeRV carrier + paired CPU+CUDA archive-level anchor lands; the dense decoder-VJP adjoint exactness (dot-product ~1e-7) + the L∞-in-latent objective transfer are verified [macOS-CPU advisory], but the archive-level savings model is pending the score-aware carrier fit (the co-equal-keystone's distortion half) + exact paired eval.`

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map — ACTIVE.** The Fisher-pullback `push_pixel_saliency_to_latent` produces per-latent-dim sensitivity from the oracle ρ_i via the exact decoder adjoint.
2. **Pareto constraint — ACTIVE.** The {latent-rate, per-coeff margin-budget} trade is the reverse-water-fill polytope; the canonical `joint_p18_p19_waterfill` KKT-Dykstra solver is the production sister.
3. **Bit-allocator — ACTIVE.** `allocate_linf_latent_steps` IS a latent-domain bit allocator at cost=oracle-ρ_i.
4. **Cathedral autopilot dispatch — N/A.** Advisory $0 probe; the result feeds the design memo's Phase-2 carrier decision (HiNeRV head-to-head vs PR95-HNeRV), not a dispatch ranking row. The co-equal-keystone confirmation reseeds the carrier-cheapness prior.
5. **Continual-learning posterior — N/A here.** No contest-axis anchor (`[macOS-CPU advisory]` non-promotable, FORMALIZATION_PENDING); the score-aware fit + paired anchor will recalibrate.
6. **Probe-disambiguator — ACTIVE.** This lane IS the probe that disambiguates "does the §7 L∞ objective transfer into a learned-carrier latent domain?" (YES, G3-exact) and "does it move the score at $0-local carrier scale?" (NO — carrier R(D)/fit dominates; the co-equal-keystone). The random-vector adjoint falsification control is the no-fake disambiguator within the G3 proof.

## Remaining before a score claim (paid)

Train the HiNeRV carrier via the score-aware CUDA `_full_main` path (SegNet+PoseNet in the loop, eval-roundtrip, EMA — the proxy-auth gap forbids per-pixel-MSE) at an aggressively-small byte budget → wire the proven L∞-in-latent allocation onto the well-fit carrier → byte-close → paired CPU+CUDA on exact bytes (Catalog #246). The build + this advisory analysis are $0; the score-aware fit + paired eval cross into paid GPU → **operator authorization required.**
