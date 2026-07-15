# RIPO categorical-Fisher trust-region: the binary-transfer falsification, MEASURED — 2026-07-14

**Mode:** `research_only=true`, `$0`, cached-only (one cheap local SegNet forward on the
already-decoded GT cache — no training, no dispatch). **Authority:** `[macOS-CPU advisory /
NumPy-fp32; no score authority]`. **Pointer UNMOVED:** 0.19108 submittable / 0.18804 borrowed
bank. All below is MEANS.

**Task:** the D42 + RIPO route-now $0 probe — validate the DERIVED categorical-Fisher directional
trust-region law on cached receipts. The RIPO memo
(`[[ripo_categorical_fisher_trust_region_falsification_20260714]]`) and the codex memo
(`codex_premise_falsification_ripo_multiclass_20260714_codex.md`) ASSERT the falsification and give
it at **two hand-picked p-vectors** (2.108× / 9.03×) but never MEASURED it on the real SegNet
distribution. This probe measures it.

## The two laws (code-verified, not memo-paraphrased)

Source: `src/tac/optimization/ripo_fisher_trust_region.py::winner_rival_curvature` /
`winner_rival_radius` (lines 270–312). `radius = sqrt(4·delta_quad / C_wr)`, `delta_quad = 2·delta_kl`
⇒ `|t| = sqrt(8·delta_kl / C_wr)`.

- **CORRECT (directional, K=5 categorical Fisher):** `|t|/sqrt(delta_kl) = 2/sqrt(C_wr)`,
  `C_wr = p_w + p_r − (p_w − p_r)²` (winner–runner-up curvature of the log-partition Hessian
  `diag(p) − p pᵀ` along `e_w − e_r`).
- **FALSE (binary intake transfer):** `||Δlogit||/sqrt(delta) = 1/sqrt(p_w)` — RIPO Eq.10, a
  single sampled-action importance-ratio bound. It depends on `p_w` alone; it is missing `1−p_w`,
  the tail mass, and the direction. Name-preserving cargo-cult of a binary result into a K=5 softmax.

## What is / isn't cached (honest data custody)

- **Real GT frames ARE cached:** `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` holds
  `gt_f1` (96 pairs, 874×1164×3 uint8, the LAST frame SegNet reads) + `lstars` (cached SegNet
  argmax). ⇒ real K=5 logits are reproducible for **$0** through the authority forward
  (`upstream/modules.py::SegNet`).
- **The RIPO fixed-head capture is store-nothing** (`large_artifacts_written=False`): it retained
  per-pair `scorer_max_probability` (p_w), `scorer_probability_margin` (p_w−p_r),
  `scorer_logit_margin`, and `exact_correction_tie_kl` **quantiles** — but NOT raw per-pixel
  joint logits. So the per-pixel joint (p_w, p_r) needed for the full ratio distribution comes from
  the cheap real forward; the cached per-pair quantiles are the cross-check.

## MEASURED — real SegNet, n96 (18,874,368 real pixels)

Probe: `experiments/probe_ripo_categorical_fisher_binary_vs_directional_20260714.py`.
Receipt: `.omx/research/ripo_categorical_fisher_binary_vs_directional_measured_20260714.json`.

**Validation (round-1 review requirement — are these REAL K=5 SegNet logits?):**
argmax(reproduced logits) == cached `lstars` on **18,874,368 / 18,874,368 pixels = 1.000000**.
The forward is the authority SegNet (right weights/preprocess/[0,255] range), not a proxy.
Operating point p_w median **0.9941** (q01 0.5899) — matches the cached RIPO n600
`scorer_max_probability` (median-of-pair-q50 **0.9974**, q01 0.5379): SAME confident distribution
with a thin near-tie annulus tail. The n96 measurement is representative of n600.

**The falsification (measured distribution, not two points):**

| statistic | value | reading |
|---|---|---|
| **Spearman rank corr `r_bin` vs `r_dir`** | **−0.9601** | the two laws order pixels **almost perfectly OPPOSITELY** — the structural falsification |
| ratio `r_dir/r_bin = 2·sqrt(p_w/C_wr)` | median **16.34×**, q01 1.57×, **max 1025×** | binary massively under-sizes the true Fisher radius at confident pixels (C_wr→0 as p_w→1) |
| binary over-admit frac (`r_bin > r_dir`) | **0** | binary is uniformly ≤ the true Fisher radius (conservative in magnitude) but **mis-ranked** |

**The rank reversal, MEASURED (annulus vs interior):**

| region | n pixels | median p_w | median C_wr | median `r_bin` (1/√p_w) | median `r_dir` (2/√C_wr) |
|---|---|---|---|---|---|
| **annulus** (logit margin < 0.5) | 249,301 | ~0.56 | ~0.98 | **1.337** | **2.018** |
| **interior** (logit margin > 4) | 16,976,420 | ~0.994 | ~0.012 | **1.003** | **17.10** |

The **binary** law assigns the annulus a **wider** absolute radius than the interior (1.337 > 1.003)
— exactly the reversed claim the codex memo flagged. The **directional** law assigns the interior a
radius **8.5× wider** than the annulus (17.10 ≫ 2.018). This is the two-point codex counterexample
(near-tie 2.108×, confident 9.03× at p_w=0.98) now confirmed as a distribution — and STRONGER: at the
real interior median p_w≈0.994, `r_dir/√δ ≈ 17`, well past the codex hand-picked 9.03×.

**Verdict scope:** FALSIFICATION-CONFIRMED at FORMULATION level — the scalar `sqrt(δ/p_1)` transfer is
structurally wrong for a K=5 softmax (Spearman −0.96). NOT a verdict on Fisher/KL trust regions as a
family (the directional law is the corrected member).

## Deliverable 2 — surrogate re-admission under the correct locus = BLOCKER (honest)

The task's D42 probe (recompute whole-teacher / on-policy surrogate ADMISSION under the correct
directional predicate instead of raw cosine) **cannot be executed on cached receipts.** The
whole-teacher surrogate remeasurement (`surrogate_vjp_fidelity_metric_remeasurement_20260714.json`)
records every advanced-locus metric as NOT-MEASURABLE by store-nothing deletion:

- `winner_runner_up_margin_directional_or_flip_preservation` = `NOT_MEASURED_LOGITS_MARGINS_AND_PERTURBATION_OUTCOMES_NOT_RETAINED`
- `softmax_kl_bregman` = `NOT_MEASURED_PROBABILITIES_NOT_RETAINED`
- `categorical_fisher_primal_or_dual` = `NOT_MEASURED_CENTERED_LOGITS_PROBABILITIES_AND_JACOBIANS_NOT_RETAINED`

The surrogate's per-state probabilities / centered logits / directional Jacobians were not retained,
so the correct-locus ADMIT/REJECT of the distilled SegNet-surrogate cannot be re-decided from disk.
The codebase-level disposition is already
`global_disposition = RAW_COSINE_WALL_IS_A_LOCUS_ARTIFACT; RETAINED_FIRST_CUT_INSTANCES_ARE_BELOW_THE_STATIC_LICENSE_GATE`
— i.e. the raw-cosine NOT-ADMITTED verdict was confirmed a locus artifact (consistent with D42), and
cosine has already been replaced by the reachable `argmax_native_vjp_fidelity_v1` metric across the
active surrogate/provider paths (`optimal_metric_p0_raw_cosine_audit_20260714.md`).

## Deliverable 3 — does D42 reopen?

**No positive re-admission; D42 stays OPEN-CUSTODY (not a negative).** The falsification HARDENS D42's
premise — the binary/cosine wall carries zero authority (Spearman −0.96, over-admit frac 0) — but the
correct-locus RE-ADMISSION requires **re-generating the retained receipts** (n600 surrogate + teacher
centered-logits / probabilities / directional Jacobians), which were deleted. `n=0 receipt is not a
negative.` The reopen is a RE-CAPTURE task, not a cached recompute.

## OWED (serialized follow-ups; NOT done here — a sibling arm owns the registries)

- **Canonical-equation registration OWED:** candidate id `categorical_fisher_trust_region_winner_rival_v1`
  (`|t| = sqrt(8·δ_kl/C_wr)`, `C_wr = p_w+p_r−(p_w−p_r)²`) — NOT registered here (merge collision:
  a sibling arm holds `src/tac/canonical_equations/`). This memo + the code-verified law are the
  provenance for a serialized registration.
- **D42 re-capture OWED:** to re-decide the surrogate admission, re-run the whole-teacher capture with
  the advanced-locus receipts RETAINED (centered logits, probabilities, directional Jacobians,
  perturbation outcomes), then apply the directional predicate. Not a $0 probe.

## Stores consulted

`[[ripo_categorical_fisher_trust_region_falsification_20260714]]`,
`codex_premise_falsification_ripo_multiclass_20260714_codex.md`,
`codex_findings_ripo_fisher_trust_region_20260714_codex.md`,
`naive_nogo_rescoping_audit_498_20260715.md` (D42 row),
`optimal_metric_p0_raw_cosine_audit_20260714.md`,
`surrogate_vjp_fidelity_metric_remeasurement_20260714.json`,
`src/tac/optimization/ripo_fisher_trust_region.py`, `upstream/modules.py`,
`experiments/results/mlx_fleet_gt_cache/gt_n96.npz`,
`experiments/results/ripo_fisher_trust_region_500_20260714/fresh_sequential_capture/progress.json`.

Pointer delta: 0.0000000000.
