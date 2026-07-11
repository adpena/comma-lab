# Papers-checked: NVFP4-RL (humansand.ai blog) / QeRL / Sol-RL "FP4 Explore, BF16 Train" (arXiv 2604.06916) — CONFIRM-of-our-discipline + ONE exploration lever; hardware half NOT-APPLICABLE

Date: 2026-07-11 · operator-supplied link (blog 403s; substance = QeRL + Sol-RL per coordinator
relay) · anti-re-research ledger entry (sister of `papers_checked_arxiv_2601_20498_20260710`,
`paper_harvest_v9cgauge_20260711`). Routed during the #426 costate-organ build.

**Verdicts (label per claim):**

1. **Explore-cheap / commit-high-fidelity decoupling (Sol-RL) = CONFIRM-NOT-LEVER, sharpened.**
   [INFERRED from relayed method] Generate a large candidate pool with cheap low-precision
   rollouts; regenerate + optimize only the selected candidates at high fidelity. This IS our
   standing discipline (MLX/proxy explores → numpy-fp32/exact-eval commits; #396 EKI candidate
   pool → exact-through-R verdict; #319 K>1 emission when the band spans 0). NEW sharpening folded
   into the #426 costate organ's exploration design: the SENSE/DECIDE layer (λ-field + duty queue)
   is the CHEAP proposal tier — it may rank MANY duty-to-measure candidates per unit compute and
   reserve through-R/exact measurement for the selected few; the routing benchmark's cost-tiered
   cascade (RouterBench lineage) is the same shape. Recorded in
   `amortized_operator_pontryagin_loop_cluster_20260711.md` §6-addendum.

2. **Quantization-noise-AS-exploration (QeRL) = WATCH-lever for #396 + the organ's exploration
   policy.** [MEASURED-by-paper per relay] Deliberate low-precision noise IMPROVES exploration
   rather than merely being tolerated. Transfer: structured noise as an exploration OPERATOR in
   the #396 gradient-free terminal finisher (EKI ensembles already inject perturbations — QeRL
   says shape them as exploration, e.g. anneal ensemble noise like a quantization schedule) and
   in the duty-queue proposal sampling. SPECULATIVE-UNTIL-PROBED on our substrate; probe rides
   #396's existing $0 EKI probe (no new task).

3. **NVFP4 hardware systems-speedup = NOT-APPLICABLE on our substrate.** [HONEST CAVEAT, binding]
   NVFP4 is a Blackwell/H100 tensor-core FORMAT; we run Apple M5 Max (MLX/Metal). Only the
   ALGORITHM (decouple + noise-as-exploration) transfers; do NOT design as if we have FP4 tensor
   cores. If a paid NVIDIA dispatch happens (≤$20 Modal envelope, operator-GO), NVFP4-RL could
   accelerate THAT run — different substrate, separate ticket.

**verdict_scope:** read-level triage (formulation); no internal lane killed or opened beyond the
#396-probe note. Pointer 0.19108282 UNMOVED (training-efficiency/exploration design = MEANS).
