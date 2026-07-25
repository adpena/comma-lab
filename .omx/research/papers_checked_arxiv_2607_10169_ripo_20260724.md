# Papers-checked — arXiv/alphaXiv 2607.10169 "Beyond Euclidean Clipping: … Riemannian Isometric Policy Optimization" (Cai, Guo, Wu, Wang, Ma, Zhang, Zhou)

UTC: 2026-07-24 · Harvested by: MAIN (Fable, inline) · $0
Evidence class: MEASURED_EXTERNAL (LLM-RL benchmarks — never our contest axis). Lessons-only.
No RL enters our live line (organ stays advisory); the harvestable content is the GEOMETRIC
CLIP LAW, not the RL algorithm.

## What the paper shows

PPO-Clip's fixed Euclidean clip on importance ratios is geometrically inconsistent with the
Fisher/KL metric of the policy manifold: true geometric distance ∝ π_old·(r−1)², so a FLAT
threshold is over-conservative for RARE actions and over-aggressive for FREQUENT ones →
exploration collapse. Cure (RIPO): distribution-dependent clip ε(π_old)=√(δ/π_old) — every
update takes the SAME geometric step. Side theorem: isometric steps induce homoscedastic
importance-sampling variance. External wins up to +60% rel over GRPO (AIME24), transfers to
PPO/coding/search.

## Crosswalk vs live surfaces (4 rows)

| # | Their lesson | Our surface | Disposition |
|---|---|---|---|
| 1 | A FIXED Euclidean threshold on a manifold whose natural metric is Fisher is a DEFECT CLASS; correct radius ∝ 1/√(local metric weight) | Dual-metric law (Euclid-vs-Fisher cosine SIGN-FLIP, never one alone) + the trust-region stack: v16/v17 validity-radius · g2f amplitude trust regions · j4/j5 per-step realized trust | **ADOPT-AS-RADIUS-LAW CANDIDATE (post-J8F, raced not assumed)**: the #366 fitting engine's realized-acceptance trust radii are currently magnitude thresholds applied uniformly across buckets whose margin-Fisher weights differ by orders of magnitude. The ms3/ms4 custody bundle (bucket-complete rank-4/margin-Fisher block Grams) is EXACTLY the producer for per-bucket metric weights → candidate refinement: per-bucket proposal-trust radius ∝ 1/√(Gram diagonal). Constants-are-poison discipline: DERIVE the radius law from the landed Grams + A/B vs the flat radius in the engine smoke — never adopt the constant by citation. IMPORTANT SCOPE CUT: the quarter-quantum caps are NOT in scope — those derive from uint8 LATTICE realizability (a realization constraint, legitimately flat in lattice units), not from a distance metric |
| 2 | Flat thresholds systematically UNDER-update rare events, OVER-update frequent ones | The measured rare-class story: Lane 0.59% area · lane-erasure ∝ 1/persistence · #208 rare-class-protected init · per-class λ | **CORROBORATION**: external math for why per-class/per-bucket scaling is geometrically REQUIRED, not a heuristic. No action |
| 3 | Isometric steps ⇒ homoscedastic sampling variance | Acceptance-noise floors in realized-acceptance loops (v19 family) + the #385/#387 noise-floor column | **NOTE**: metric-scaled radii would also EQUALIZE verdict noise across buckets — a second, independent reason to prefer them if row 1's A/B fires. No action now |
| 4 | RIPO as an RL algorithm | No RL in the counted path; organ advisory-only | **N/A** |

## Verdict

`LESSONS_HARVESTED_INLINE; ONE_RADIUS-LAW_CANDIDATE_QUEUED (post-J8F, race-gated, ms4-Gram-fed);
NO_ARM_SPAWNED`. Coherence check: NOVELTY — the √(δ/p) functional form + homoscedasticity
theorem are new external datums on our dual-metric law; DERIVATION — row 1 maps to the named
flat-radius surfaces in the live engine, fed by an ALREADY-LANDED producer (ms4 bundle);
DISTANCE — joins the post-J8F wave as an engine-config refinement, no new critical path.
Pointer 0.1910828242 [contest-CPU] UNMOVED — this is means.

STORES CONSULTED: dual_metric_readback memory · ms3/ms4 bundle receipts (BUNDLE-COMPLETE via
rg3) · v16/v17 validity-radius law · g2f trust regions · j4/j5 realized-trust rows ·
realization-quantization-gated law (quarter-quantum scope cut) · #208/per-class-λ rare-class
rows · papers_checked_* precedent.
