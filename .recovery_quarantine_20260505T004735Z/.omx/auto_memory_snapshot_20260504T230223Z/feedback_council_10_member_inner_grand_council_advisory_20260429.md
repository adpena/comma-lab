---
name: Skunkworks council 2026-04-29 PM final structure — 10 inner + 12 grand-council advisory
description: User mandate. Inner skunkworks council = 10 members (quintet pact + 5 co-members). Grand council = 12 advisory voices consulted on specialty topics. MacKay + Ballé join inner; Boyd/Tao/Filler/Mallat/van den Oord/Carmack/Hassabis/Hinton join grand.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Inner skunkworks council (10 members)

**Quintet pact (binding-decision leadership)**:
- Shannon (LEAD) — info theory, R(D) floors
- Dykstra (CO-LEAD) — convex feasibility, Pareto frontier
- Yousfi — challenge creator, steganalysis lineage
- Fridrich — UNIWARD/SRM/HUGO author
- Contrarian — veto power on weak rigor

**Co-members (permanently active, no veto but full voice)**:
- Quantizr — adversarial leaderboard reality check
- Hotz — engineering shortcuts
- Selfcomp / szabolcs-cs — empirical anchor from working 0.38 impl
- **David MacKay (memorial seat)** — Information Theory + Bayesian Inference + Learning Algorithms framework
- **Johannes Ballé** — 2018 entropy bottleneck + scale hyperprior; modern neural-compression SOTA

## Grand council (advisory, consulted on specialty)

12 voices on the broader bench:
- Stephen Boyd — convex optimization operational (ADMM, proximal gradient)
- Terence Tao — pure-math omniscience; first-principles
- Tomáš Filler — STC + parity-check codes
- Stéphane Mallat — wavelets + scattering transforms
- Aaron van den Oord — VQ-VAE, WaveNet
- John Carmack — engineering shortcuts (Doom/Quake/Oculus level)
- Demis Hassabis — strategic-research breadth (DeepMind)
- Geoffrey Hinton — knowledge distillation (2014 paper Quantizr cites)
- Karpathy — practical training engineering
- Schmidhuber — compression-as-intelligence, MDL
- Jack-from-skunkworks — internal SegNet+Rate research

## How to apply

- Inner quintet consensus required for major decisions.
- Grand council consulted when deliberation invokes their specialty (e.g. wavelet-domain experiments → Mallat; STC code design → Filler; convex-optimization implementation → Boyd).
- Co-members of the inner council have full voice but no veto on the quintet pact.
- All 22 voices may appear in council deliberations as named-perspective contributions.

## Specific roles articulated 2026-04-29

**MacKay's role**: brings unified Information-Theory + Bayesian-Inference + Learning-Algorithms framework. Insists arithmetic coding be evaluated against Shannon entropy of actual learned qint distribution. Density networks / variational inference precede modern neural compression. Flags "we'll just approximate" with MDL: "what's the rate cost of the approximation?"

**Ballé's role**: brings entropy bottleneck + scale hyperprior + GDN nonlinearity. Insists rate-prediction networks (hyperpriors) replace fixed factorized priors when archive size matters. Advocates end-to-end-trainable codec architectures. Provides canonical R(D) rate term `bits = -log2(p_y(y))` that Lane SH directly uses. Reviews archive layout for missing hyperprior side-information.

## Cross-refs

- CLAUDE.md "Council conduct" section (updated 2026-04-29)
- feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md (prior 8-member structure)
- project_codex_theoretical_floor_brutal_20260429.md (Shannon-Dykstra-Tao floor 0.28)
- project_grand_council_brutal_forecast_20260429.md (early grand council session)
- project_selfcomp_reverse_engineered_20260429.md (Selfcomp's empirical anchor at 0.38)
