# DAG FEED — operator-paper intake cluster (2026-07-13), verdict-scoped

**Merge target:** `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (standalone FEED to avoid
serializer collision with the running throughput arms; main merges).

**Pointer:** submittable **0.19108** [contest-CPU], borrowed non-submission bank **0.18804** — UNMOVED.
All below is paper-analysis → hypotheses; NO measurement, NO pointer move. Ladder declared per node.

---

## Node FEED-madam — arXiv 2607.10611 (M+Adam) + github Anima-Lab/M-Adam. Task #496.

M+Adam = hybrid additive+multiplicative low-precision-TRAINING optimizer (BF16/FP8/FP4 master weights,
monotone-descent proof, fixes additive stall-at-large-magnitude under coarse mantissa; LLaMA 60M–1B).

**Verdict = NOT a kill; INSTANCE-scoped hypotheses + a live reformulation queue.** Narrowest level the
evidence (paper analysis only, no n600) supports = **INSTANCE** for every claim below.

**Reformulation queue (untested formulations to measure — the required "alternatives"):**
- AXIS-1 (RATE): train witness at FP4/FP8 (M+Adam) → ship low-precision weights (≤8× weight-payload cut)
  IF d_seg/d_pose hold. Untested vs the incumbent = train-fp32-then-post-hoc-KKT-waterfill (#157/#311/#406).
- AXIS-2 (CONVERGENCE+DISTORTION): M+Adam multiplicative/exponent-space branch = scale-invariant update
  → applied to the witness multi-scale Fourier basis to converge the under-trained high-freq lane bands
  (measured 3.2× along-tangent deficit = a d_seg residual). Untested vs Adam/Muon per-band; and vs the
  incumbent multi-scale levers FreSh (#448) + FINER++ (#310) + along-tangent (#277).

**Scoping statements (NOT family kills — each is a mechanism-transfer boundary WITH its reason):**
- The *core* M+Adam convergence fix (coarse-mantissa additive-stall) is **low-precision-specific** → does
  not explain our **fp32** unequal-class-convergence (that stays the loss-geometry/imbalance thread:
  recos ordinal + Minkowski/Wulff σ_cc′ #382). FORMULATION-boundary, not a kill of low-precision training.
- Monotone-descent guarantee is **precision-regime-specific** → does not fix our fp32 **loss-geometry**
  instabilities (eikonal re-entry #316, MCF thin-lane erasure). Scoping, not a kill.
- Distortion benefit is **RD-coupled** to AXIS-1 (preservation-under-quantization), not an independent
  distortion win. Observation, not a verdict.
- FP8-GEMM *speed* needs Hopper/Transformer-Engine → on MLX/Metal only the **algorithm** transfers.
- OFF-TARGET for the throughput P0 (frozen-scorer INFERENCE) and unrelated to the L70 determinism wall
  — M+Adam optimizes weights *being trained*, not a *frozen* forward. INSTANCE, correct by construction.

**Transfer caveat (both axes):** validated on LLM PRETRAINING (generalization); our witness = single-video
INR (memorization) → not obvious, MEASURE before believing. $0-probe-gated; no heavy dispatch until verdict.

**Deep-math tie:** multiplicative/exponent-space branch lives in the log/tropical space of our argmax
(#284) + logit-adjust + softmax → reinforces MD-Decoupling (#175) + manifold-Muon (#469).

---

## Node FEED-recos — arXiv 2602.05266 (recos, ordinal concordance). Routed to frontier-math arm.

Off-domain (STS embeddings); one on-target idea: **ordinal-concordance ↔ argmax preservation**. Elevated
from throughput-curio to candidate **d_seg lever** (potential cure for unequal class convergence).
Reformulation queue: (a) recos-style full-ranking as a complement to the interval-arithmetic argmax
certificate; (b) ordinal/margin-concentrated loss vs arithmetic CE — per-class convergence-RATE A/B.
Narrowest level: INSTANCE (no measurement). Adversarial temper: geometric causes (MCF erasure, flicker
floor 0.005318) may dominate → the arm must verdict contributory/dominant/inert, not assume.

## Node FEED-minkowski-kempner — Minkowski surface area (IMG_7018) + Kempner weight-decay-plasticity.

Minkowski content `lim_{ε↓0}[Vol(K+εB₂)−Vol(K)]/ε` = the mathematical foundation of σ_cc′ (#382): Steiner
expansion unifies perimeter (ε¹, the energy) + integrated mean curvature (ε², the MCF erasure velocity);
isotropic B₂ = scalar-length erasure trap, anisotropic Wulff/Finsler = class-balanced σ_cc′. Kempner
weight-decay-for-plasticity → distinct $0 witness A/B (per-class convergence + effective rank + rate);
transfer caveat: LM-pretraining (generalization) vs INR (memorization). Both INSTANCE-scoped, routed to
frontier-math arm. IMG_7017 content not delivered to this session (flagged to operator).
