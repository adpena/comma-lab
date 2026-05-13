# Pre-Mortem Analysis — Final Submission Lands at 0.25

**Date:** 2026-05-13
**Lane:** `lane_expert_team_aerospace_stealth_analytic_alien_tech_20260513`
**Method:** Klein 2007 "Performing a Project Premortem" (Harvard Business Review).
**Evidence grade:** `[structured-analytic-technique]`.

---

## 0. Pre-mortem scenario

**The hypothetical:** Imagine 14 days from now (2026-05-27). The contest is over (or the
race window has elapsed). Our final submission scored **0.25 [contest-CUDA]** — below PR101's
0.193 baseline but above the medal band (~0.195). We achieved approximately the 11th-place
band the team hit on April 30. **The submission FAILED — what happened?**

Per Klein's method: each council member independently lists possible causes; the team then
deduplicates and rank-orders failure modes. The pre-mortem is COMPLETE when every council
member has contributed and consensus on top-N failure modes emerges.

---

## 1. Failure modes — independent council member contributions

### Shannon (LEAD)
1. **F-IT (Information-Theoretic floor was too optimistic)**: the 0.10±0.03 floor estimate
   was a Bayesian aggregation. We anchored on the LOW end and overcommitted to score-
   lowering substrates that were never theoretically reachable.
2. **F-EE (Effective-equivalence-class smaller than theoretical)**: while the mathematical
   equivalence class has O(10^9) free dimensions, the EFFECTIVE achievable equivalence
   class (within Brotli/zstd entropy budget and within Python decoder LOC budget) was much
   smaller. Submarine of dim 10^9 doesn't help if you can't address it.

### Dykstra (CO-LEAD)
3. **F-PCS (Pareto-constraint-set under-tightened)**: we modeled F_seg ∩ F_pose ∩ F_rate but
   missed F_runtime (the inflate.sh 30-min T4 budget). When training time per substrate
   exceeded budget, we couldn't iterate.
4. **F-DKS (Dykstra didn't converge)**: alternating projections require the constraint sets
   be CONVEX. Argmax-preservation is DISCRETE (non-convex). Our Pareto frontier estimate
   was wrong for that constraint.

### Yousfi (steganalysis expert)
5. **F-AAG (Auth-Auth Gap — model on Linux x86_64 production)**: our internal CUDA measurement
   on Vast.ai 4090 did NOT match the contest GHA Linux x86_64 CUDA. We discovered a 0.020-
   0.040 gap LATE. Score 0.193 internally → 0.225+ publicly.
6. **F-CCG (CPU-CUDA gap not measured)**: per PR102 showing +0.033 CUDA-CPU gap, our
   submission would benefit from the public CPU axis but we never measured. Wasted score.

### Fridrich
7. **F-AVL (Auth-eval value-of-loss assumption wrong)**: our proxy loss correlated with
   auth on training video `0.mkv` but NOT on the held-out contest video. Score = 0.193
   on `0.mkv` → 0.265 on contest video.
8. **F-OBP (Over-Boosted Proxy)**: gradient-pushing the proxy created adversarial
   examples that fool the scorer ON the training video but fail to generalize.

### Contrarian
9. **F-LL (Local minimum — HNeRV basin not escaped)**: the council recommended TRIPLET φ
   (SABOR + S2SBS + PAYIC) but the dispatch team built TRIPLET E (HNeRV-family-internal).
   We hit the HNeRV ceiling at 0.171; everyone celebrated, but it wasn't medal-band.
10. **F-AC (Anchoring on Council theoretical floor)**: the 0.10±0.03 floor turned out to
    be aspirational; actual achievable floor was 0.18. We over-promised.

### Quantizr
11. **F-QU (QAT discipline never applied)**: every substrate landed at FP32; QAT
    finetune was deferred for "next iteration" that never came. Net Δ left on the table:
    0.001-0.004.
12. **F-EMA (EMA shadow inflated)**: we built archives from FP32 live weights rather
    than EMA shadow. 1-3% performance loss per CLAUDE.md "EMA non-negotiable".

### Hotz
13. **F-RACE (We optimized for elegance, not speed)**: every substrate took 7-14 days; in
    a 4-hour-race the medal-band moved from 0.21 to 0.193 while we built infrastructure.
    Our 0.225 archive existed but was an order behind.
14. **F-COMP (We tried to be smart instead of fast)**: small bolt-ons (241 LOC silver)
    beat large architectures (1776 LOC kitchen sink). We trended large.

### Selfcomp
15. **F-CGS (Compose-Generate-Stack failure)**: we built 10 substrates but never
    composed them. Each substrate scored ~0.193-0.220. Single-substrate-best didn't beat
    PR101.
16. **F-NB (Non-binding hypothesis)**: we built 14 substrates per ACH but only had time
    to dispatch 5; some substrates were never auth-eval'd.

### MacKay (memorial)
17. **F-Bayes (Bayesian prior on substrate feasibility miscalibrated)**: we believed
    PAYIC would work with 0.65 probability; actual yes-probability was 0.20. We invested
    in a probable-failure.
18. **F-OF (Operating-point fragility)**: PR106 r2 pose_avg = 3.4e-5 (pose marginal = 271).
    Substrates optimized for this operating point CATASTROPHICALLY fail when the operating
    point shifts (auth eval at slightly different pose_avg).

### Ballé
19. **F-NBE (Neural-Bottleneck-Entropy miscalibrated)**: scale-hyperprior over HNeRV
    latents is supposed to add -0.002 to -0.006. Actual contribution was +0.002 (it
    increased rate). The scale hyperprior overfit the training set.

### Hassabis (grand council)
20. **F-NXG (Next-generation architecture never built)**: we kept iterating on HNeRV when
    the breakout was always going to require a new architecture class (e.g. Cool-Chic, C3).
    The PUBLIC corpus had no sub-0.20 in PR50-115 except PR105 — strong signal we needed
    novel substrate.

### Carmack (grand council)
21. **F-LOC (Lines-of-code budget violated)**: our cascade renderer exceeded 200 LOC
    when contest budget is 100 LOC (200 with waiver). The submission failed contest
    closure gate.
22. **F-DEPS (Dependency closure failed)**: a critical sidecar `.br` decoder wasn't
    installed in the Modal/Vast.ai bootstrap; auth eval crashed silently.

---

## 2. Dedupe and rank — top failure modes by likely-frequency × impact

| Rank | Failure mode | Likelihood | Impact | Mitigation |
|---:|---|:---:|:---:|---|
| 1 | **F-AAG/CCG**: Auth-Auth or CUDA-CPU gap not measured | 0.35 | -0.040 to -0.080 | DUAL-EVAL mandate on every shipping archive (CLAUDE.md non-negotiable in place; ensure compliance) |
| 2 | **F-RACE**: Optimized for elegance, not speed | 0.30 | -0.030 to -0.060 | Per CLAUDE.md "Race-mode rigor inversion" + "Strategic-rigor inversion at leaderboard moves": when race active, smallest credible bolt-on wins |
| 3 | **F-LL**: Local minimum (HNeRV basin) not escaped | 0.30 | -0.020 to -0.050 | Dispatch matrix MUST include at least one HNeRV-external substrate per ACH H15 + H7 + H8 |
| 4 | **F-AVL**: Auth-eval value-of-loss assumption fails on held-out video | 0.25 | -0.020 to -0.060 | Pre-mortem requires hold-out probe; train on `0.mkv` but validate against pseudo-hold-out (videos 1-9) |
| 5 | **F-CGS**: Substrate composition never landed | 0.40 | -0.010 to -0.030 | `tac.composition.registry` PRE-DISPATCH: every substrate must declare composition partners |
| 6 | **F-LOC**: Lines-of-code budget violated | 0.20 | -0.020 to -0.050 (closure failure → unscoreable) | CLAUDE.md "≤350 LOC bolt-on" + Catalog #146 contest-compliant runtime gate |
| 7 | **F-EMA/QU**: EMA + QAT not applied | 0.40 | -0.005 to -0.010 each = -0.010 to -0.020 combined | Already CLAUDE.md non-negotiable; verify substrate-side compliance |
| 8 | **F-OF**: Operating-point fragility | 0.30 | -0.005 to -0.025 | F1 multi-spectral Lagrangian sweep + F5 coupled component-balance (ledger 06) |
| 9 | **F-IT**: Theoretical floor was too optimistic | 0.30 | -0.020 to -0.080 (gap between 0.10 estimate and actual 0.18 achievable) | Update prior to reflect HNeRV ceiling = 0.17-0.19 as MOST LIKELY achievable; treat 0.10 as ASPIRATIONAL not PLANNING basis |
| 10 | **F-NXG**: New architecture class never built | 0.20 | -0.040 to -0.080 | Maintain at least 1 long-pole substrate-engineering lane (PR106 r2 latent sidecar / SIREN family) |
| 11 | **F-OBP**: Over-boosted proxy creates non-generalizing adversarial | 0.20 | -0.010 to -0.040 | Eval-roundtrip + scorer-preprocess differentiable (CLAUDE.md non-negotiable) — verify substrates comply |
| 12 | **F-PCS/DKS**: Pareto constraints under-tightened OR non-convex | 0.15 | -0.005 to -0.030 | Update meta-Lagrangian solver to track F_runtime + handle discrete argmax constraint via relaxation |
| 13 | **F-DEPS**: Dependency closure failed | 0.10 | catastrophic (unscoreable) | Catalog #2 (set -e) + remote_archive_only_eval canonical bootstrap + Phase 1 packet-compiler runtime closure check |
| 14 | **F-Bayes**: PAYIC yes-probability overestimated | 0.40 | -0.005 to -0.015 (lost investment) | φ2 PAYIC probe at $0-5 BEFORE substrate build (ledger 02 B4 cheap-probe discipline) |
| 15 | **F-NBE**: Scale-hyperprior overfits training | 0.20 | -0.002 to -0.006 | Hyperprior trained with regularization; held-out validation |
| 16 | **F-COMP/F-AC/F-NB**: process failures | 0.30 | -0.005 to -0.015 each | Already addressed via continual-learning posterior + cathedral autopilot |

---

## 3. Mitigation matrix — what we can do TODAY to prevent each failure mode

| Failure mode | Action this session (READ-ONLY) | Action next ≤2 days | Action next ≤7 days |
|---|---|---|---|
| F-AAG | Audit ACH for evidence anchors; ensure dual-CUDA-CPU is tagged on every shipping anchor | Dual-eval audit | Re-verify on T4 + GHA Linux |
| F-RACE | Audit dispatch matrix for elegance-vs-speed bias | Drop large substrates if no urgent dispatch | Cron-fire small bolt-ons hourly during race window |
| F-LL | Recommend H15 + H7 + H8 (ACH winners) | Build PR103-on-PR106 sidecar | Dispatch H15 + cheap-probe H7 |
| F-AVL | Document hold-out-video risk in council | Add pseudo-hold-out validation | Sweep across videos 1-9 |
| F-CGS | Verify `tac.composition.registry` accepts H15+H7 composition | Composition test fixture | Composition dispatch |
| F-LOC | Re-verify Catalog #146 contest-runtime gate active | Substrate LOC audit | Hard-cap enforcement |
| F-EMA/QU | Catalog #87 + #88 EMA + QAT discipline review | Substrate-side compliance check | Re-train any non-compliant substrate |
| F-OF | Audit existing substrates for operating-point fragility | Add F1 + F5 sweep training | Multi-spectral Lagrangian dispatch |
| F-IT | Update planning prior to reflect HNeRV ceiling | Re-rank substrates by likelihood, not theoretical max | Diversify dispatch matrix |
| F-NXG | Allocate at least 1 substrate-engineering lane for new architecture class | Begin PR105/PR106 latent sidecar design | Dispatch substrate-engineering canary |
| F-OBP | Audit substrate trainers for eval-roundtrip compliance | Differentiable scorer preprocess verify | Adversarial-validation gate |
| F-PCS/DKS | Update solver to handle non-convex argmax | Code sketch in `tac.meta_lagrangian` | Wire into autopilot |
| F-DEPS | Catalog #11 + #14 runtime closure check | Pre-dispatch dependency closure | Modal/Vast bootstrap audit |
| F-Bayes | φ2 PAYIC probe at $0-5 budget allocated | Dispatch probe | Decision: invest or defer |
| F-NBE | Scale-hyperprior regularization audit | Held-out validation | Re-train if overfit |
| F-COMP/F-AC/F-NB | Continual-learning posterior update | Cathedral autopilot dispatch journal | Substrate dispatch fan-out |

---

## 4. Council adversarial review of the pre-mortem

- **Klein (memorial seat).** "30% improvement in failure-mode identification is the
  empirical pre-mortem ROI. You found 22 modes; you'd have found ~15 without the pre-mortem
  structure. The framework worked." → **ENDORSE**.

- **Heuer (memorial seat).** "F-CCG (CPU-CUDA gap not measured) is a HIGH-PROBABILITY,
  HIGH-IMPACT failure mode. The CLAUDE.md dual-eval mandate is in place — verify substrate
  teams are following it." → **STRONG ENDORSE; flag as operator-attention top item**.

- **Hotz.** "These pre-mortems are EXACTLY the failure-mode list from the May 4 race
  retrospective. We're re-discovering the same lessons. The mitigation is to DO the
  per-CLAUDE.md disciplines, not re-discover them." → **CONFIRM.** This pre-mortem
  validates existing CLAUDE.md non-negotiables.

- **Contrarian.** "Pre-mortem itself can be a procrastination tool. We've now written 6
  ledgers + ACH + pre-mortem + master memo = 100+ KB of memos. Time to DISPATCH something."
  → **STRONG CHALLENGE.** Operator should consider: this session's output is research-only
  per CLAUDE.md; the actionable items are the H15 + H7 dispatches recommended in the ACH.

---

## 5. Final ranked failure modes (operator-attention list)

For operator decision-attention, the **top 5 failure modes** to actively defend against:

1. **F-AAG/CCG (auth-auth or CUDA-CPU gap)** — likelihood 0.35, impact -0.040 to -0.080.
   **Mitigation: verify dual-eval mandate compliance on every shipping anchor**.
2. **F-RACE (optimized for elegance not speed)** — likelihood 0.30, impact -0.030 to -0.060.
   **Mitigation: in race window, fire smallest credible bolt-on per CLAUDE.md "Race-mode rigor inversion"**.
3. **F-LL (HNeRV local minimum not escaped)** — likelihood 0.30, impact -0.020 to -0.050.
   **Mitigation: ACH H15 + H7 + H8 dispatches (HNeRV-external substrates)**.
4. **F-CGS (substrate composition never landed)** — likelihood 0.40, impact -0.010 to -0.030.
   **Mitigation: pre-dispatch `tac.composition.registry` audit**.
5. **F-EMA/QU (EMA + QAT not applied)** — likelihood 0.40, impact -0.010 to -0.020.
   **Mitigation: Catalog #87 + #88 STRICT gate enforcement**.

---

**End pre-mortem.**
