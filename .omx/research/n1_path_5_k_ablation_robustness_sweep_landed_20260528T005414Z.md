---
council_tier: T1
council_attendees: [Claude-N1-PATH-5-K-ablation-subagent]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "operator NON-NEGOTIABLE 2026-05-27 verbatim 'All MLX unless auth eval or high signal or artifact, otherwise all are approved and you decide'; T1 working-group K-ablation reactivation per predecessor T1 (commit a2cd12274) reactivation criterion; per-T3-grand-council reactivation path #5 K-ablation sweep prerequisite per Catalog #315 BEFORE paired-CUDA invocation per Catalog #246"
canonical_equation_reference: pose_axis_score_direction_matching_paradigm_savings_v1
horizon_class: asymptotic_pursuit
council_dissent:
  - member: Claude-N1-PATH-5-K-ablation-subagent
    verbatim: "the classifier verdict ROBUST is single-seed evidence; the bootstrap variance verdict at K=10 across seeds {0,7,13} shows SNR=0.18 paired Δpose mean=+1.54 std=8.59 — the +5.02 to -11.23 K-ablation signal at seed=0 IS within noise of zero across 3 seeds; the honest verdict is DUAL: classifier-ROBUST + bootstrap-SEED-DEPENDENT. The K-ablation classifier was naive about seed variance; the bootstrap verdict is the higher-order finding."
council_assumption_adversary_verdict:
  - assumption: "K-ablation single-seed pattern ≥3 of 5 composable IS sufficient empirical evidence of robustness"
    classification: CARGO-CULTED
    rationale: "the 5-K-ablation single-seed pattern was the operator-spec robustness threshold but the bootstrap-3-seed variance verdict empirically reveals SNR=0.18 — the K-pattern is determined by seed init not by structural K-dependence. Per Catalog #292 per-deliberation assumption surfacing + CLAUDE.md 'Apples-to-apples evidence discipline' the single-seed verdict is structurally insufficient for paired-CUDA spend justification."
  - assumption: "paired-CUDA spend warranted ONLY when classifier verdict ROBUST"
    classification: CARGO-CULTED
    rationale: "the spec mapped classifier verdict ROBUST → paired-CUDA TOP-1 surfaced; the bootstrap verdict supersedes this mapping. Per CLAUDE.md 'Forbidden empirical-claim-without-evidence-tag': the K-classifier verdict cannot be elevated to paired-CUDA justification without bootstrap-variance-passing the signal-magnitude > 2*sigma threshold. SNR=0.18 ≪ 2."
council_decisions_recorded:
  - "op-routable #1: NO PAIRED-CUDA invocation at K=5 despite classifier verdict ROBUST. Bootstrap SNR=0.18 verdict supersedes per CLAUDE.md 'Apples-to-apples evidence discipline'."
  - "op-routable #2: canonical equation pose_axis_score_direction_matching_paradigm_savings_v1 status remains FORMALIZATION_PENDING per Catalog #344; refined with K-ablation EmpiricalAnchor rows (5 K + 2 bootstrap variance = 7 new anchors); composable_with_segnet_partial_confirmation_delta_pose field NOT updated (the K=10 anchor stands as historical but is bootstrap-seed-dependent)."
  - "op-routable #3: Catalog #313 probe-outcomes ledger row updated NEW row probe_id mlx_pose_axis_score_direction_matching_path5_n1_kablation_20260528 verdict DEFER (NOT PROMOTE despite classifier verdict ROBUST) with reactivation criterion: 8-seed bootstrap variance sweep (seeds {0,7,13,21,42,73,101,133}) at K=5 to determine whether the K=5 BEST-SAMPLE-POINT signal exists outside the seed-{0,7,13} 3-sample bootstrap or whether seed=0 is the only seed where K=5 produces composable signal."
  - "op-routable #4: 8th MLX-first standing directive honored — entire sweep $0 MLX-local; faithful YUV6 via real MLX SegNet+PoseNet adapters; no synthetic teacher; non-promotable per Catalog #192/#127/#323."
---

# N1 PATH 5 D-REGIME K-ABLATION ROBUSTNESS SWEEP (LANDED)

**Predecessor**: T1 N1 PATH 5 score-DIRECTION matching probe (commit `a2cd12274`,
memo `.omx/research/n1_path_5_score_direction_matching_pose_binding_landed_20260527T225708Z.md`)
measured single-sample-point D regime (SegNet + pose-cos-dir) at REFRESH_K=10
producing Δpose=−5.02 composable signal but flagged this as UNCONFIRMED for
robustness to teacher-direction staleness. The predecessor explicitly named the
K-ablation sweep as the canonical prerequisite per Catalog #315 BEFORE any
paired-CUDA invocation per Catalog #246.

**THIS PROBE** executes the K-ablation sweep across REFRESH_K ∈ {1, 5, 10, 30,
100} plus a 3-seed bootstrap variance check at K=10 (seeds {0, 7, 13}) per the
operator mandate, classifying the D-regime signal into one of 4 robustness
verdicts and producing a paired-CUDA conditional command sheet for the operator.

## 6-K-value × seed=0 D-regime empirical results

(faithful YUV6 distortion measurement via real MLX SegNet+PoseNet adapters;
100ep AdamW LR=1e-2; NP=16; SEG_W=POSE_DIR_W=0.5; PyTorch first-order CPU teacher
gradient cache; refresh count derived from `STEPS // REFRESH_K`)

| K | pose | seg | recon | Δpose vs B (154.0147) | Δseg vs B | refresh_count | teacher_grad_absmax | wall_s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | 151.2571 | 0.4667 | 0.00641 | **−2.76** | −0.0511 | 100 | 0.449 | 340 |
| **5** | **142.7844** | 0.5034 | 0.00640 | **−11.23 ← BEST** | −0.0144 | 20 | 0.417 | 88 |
| **10** | 148.9949 | 0.5069 | 0.00641 | **−5.02 (predecessor reproduces)** | −0.0109 | 10 | 0.376 | 56 |
| **30** | 146.7308 | 0.5037 | 0.00641 | **−7.28** | −0.0141 | 4 | 0.415 | 37 |
| **100** | 148.4852 | 0.4970 | 0.00640 | **−5.53** | −0.0209 | 1 | 0.614 | 27 |

**REGRESSION TEST PASS**: D K=10 seed=0 reproduces predecessor anchor seg=0.5069
+ pose=148.9949 within tolerance ε_seg=0.005 + ε_pose=1.0 (per Catalog #229
premise verification). The probe scaffold is byte-faithful.

**All 5 K values composable** (Δpose < 0 vs B at all K); **all 5 seg axes stay
≤ B baseline** (seg-axis is NOT traded for pose-axis gain — the canonical
Pareto-frontier-expansion property holds across all K). At single-seed evidence
the classifier verdict is **ROBUST**.

## Bootstrap variance check at K=10 (seeds 0, 7, 13)

| Seed | B baseline pose | D K=10 pose | Paired Δpose (D−B) |
|:----:|---:|---:|---:|
| 0 | 154.0147 | 148.9949 | **−5.02** (composable; predecessor anchor) |
| 7 | 169.5484 | 180.8031 | **+11.26** (worsens; signal reverses) |
| 13 | 142.1776 | 140.5504 | **−1.63** (composable but smaller magnitude) |

| Statistic | Value |
|---|---:|
| D K=10 pose mean (n=3) | 156.78 |
| D K=10 pose std (n=3, ddof=1) | 21.23 |
| B baseline pose mean (n=3) | 155.25 |
| B baseline pose std (n=3, ddof=1) | 13.73 |
| **Paired Δpose D−B mean (n=3 pairs)** | **+1.54** (positive — net worsens across seeds) |
| **Paired Δpose D−B std (n=3 pairs)** | **8.59** |
| **Signal-to-noise (|mean| / std)** | **0.18** (≪ 1 — noise dominates) |

The bootstrap-3-seed verdict empirically shows that the K=10 Δpose signal at
seed=0 (−5.02) is within the noise distribution at K=10 across 3 seeds (std=8.59,
mean=+1.54). The SNR=0.18 is ~5.6× below the conventional 1.0 SNR threshold for
"real effect" and ~11× below the 2.0 SNR threshold for "confident effect."

## Honest dual verdict per Catalog #307 paradigm-vs-implementation discipline

**Two distinct signals emerge from the empirical data:**

### Signal 1 (classifier verdict): D-regime is COMPOSABLE at all 5 K values, seed=0

- 5/5 K values composable (Δpose < 0 vs B at all K).
- 5/5 K values seg axis stays ≤ B baseline (Pareto-frontier-expansion holds).
- K=5 BEST sample point at Δpose=−11.23 (~7.3% relative pose improvement on B's 154.01).
- Per the mandate's pre-specified classifier rule (≥3 of 5 composable AND seg ≤ B
  threshold ≥3): verdict = **ROBUST**.

### Signal 2 (bootstrap verdict): the K=10 anchor IS SEED-DEPENDENT

- Paired Δpose at K=10 across seeds {0, 7, 13}: −5.02, +11.26, −1.63.
- Mean=+1.54 (POSITIVE — net worsens across 3 seeds).
- Std=8.59 (~5.7× larger than predecessor's −5.02 anchor magnitude).
- SNR=0.18 ≪ 1.0 (noise dominates the signal at K=10).
- Implication: the −5.02 anchor at seed=0 is structurally indistinguishable
  from a noise realization within the 3-seed bootstrap distribution.

### Combined verdict (per Catalog #307 IMPLEMENTATION-LEVEL classification)

The K-ablation single-seed pattern at seed=0 (all 5 K composable, monotonic-
non-monotonic) IS REAL at seed=0 but DOES NOT REPLICATE across seeds. The
predecessor's −5.02 anchor + this probe's −11.23 K=5 BEST are seed-specific
realizations; the underlying paradigm (cos-distance pose-direction matching
composable with SegNet binding) DOES produce composable signal at the K-ablation
sample mean (seed=0) but DOES NOT produce a stable signal across seed
initialization.

**Per Catalog #307**: this is **IMPLEMENTATION-LEVEL PARTIAL** — the paradigm
mathematical structure (Cauchy-Schwarz scale-invariance of cos-distance) is
INTACT; the IMPLEMENTATION REALIZATION at NP=16 / STEPS=100 / SEG_W=POSE_DIR_W=0.5
is SEED-DEPENDENT and not robust to initialization noise. The structural-ceiling
verdict from the parent T3 council ("MLX surrogate pose-binding paradigm
empirically DEFER per 5+ falsified mechanisms") is REINFORCED at the bootstrap
verdict surface.

### Per Catalog #246 paired-CUDA gate

**NO PAIRED-CUDA invocation surfaced as TOP-1 routable** despite classifier
verdict ROBUST. The bootstrap SNR=0.18 verdict empirically falsifies the
single-seed classifier verdict's promotability to paired-CUDA spend
justification. Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog
#192 "macOS-MLX research-signal NEVER promotable without paired Linux x86_64
+ NVIDIA CUDA validation" + Catalog #287 "Forbidden empirical-claim-without-
evidence-tag" — the K=5 BEST sample-point at Δpose=−11.23 is a single-seed
realization within a 3-seed bootstrap noise distribution whose mean is +1.54.
A paired-CUDA dispatch at K=5 would burn ~$0.30-0.50 on a signal that does not
exceed the noise-floor.

## Operator-routable conditional command sheet (NOT invoked from this subagent)

Per CLAUDE.md "Executing actions with care" non-negotiable: the operator decides.
The mandate authorized "you decide" but the bootstrap-variance verdict
supersedes the classifier verdict per the assumption-adversary verdict above.
The honest operator-routable cascade is:

### Cascade A (RECOMMENDED): NO paired-CUDA spend; extend bootstrap to 8 seeds

```bash
# Extend bootstrap variance check from 3 seeds to 8 seeds at K=5 (the BEST
# single-seed K-sample-point) to determine whether K=5 composability is
# seed-dependent (most likely outcome) or reproducible across seed init.
# Estimated cost: 8 seeds × (D K=5 + B at same seed) × 2 × 88s ≈ 24 min wall-clock
# at $0 MLX-local; non-promotable per Catalog #192.
.venv/bin/python .omx/tmp/pose_score_direction_matching_e2e_kablation_8seed_bootstrap_at_k5.py
```

If 8-seed bootstrap at K=5 produces SNR ≥ 1.0: revisit paired-CUDA decision.
If 8-seed bootstrap SNR < 1.0: declare paradigm IMPLEMENTATION-LEVEL DEFER per
Catalog #307 + queue alternative-reducer enumeration per Catalog #308 (4
canonical alternatives: per-pair class HISTOGRAM / per-region class HISTOGRAM /
per-segment-class / per-temporal-window).

### Cascade B (NOT RECOMMENDED): paired-CUDA at K=5 despite bootstrap warning

```bash
# OPERATOR OVERRIDE: paired-CUDA at K=5 (BEST single-seed K-sample-point)
# despite SNR=0.18 bootstrap warning. Expected: ~$0.30-0.50 PAID with high
# probability of NULL or WORSEN on CUDA-T4 (seed-dependent signal does not
# transfer to different hardware substrate per Catalog #192 standing pattern).
.venv/bin/python tools/operator_authorize.py \\
    --recipe substrate_dreamer_v3_rssm_modal_t4_dispatch \\
    --override-config '{"bind_pose_cos_direction": true, "POSE_DIR_W": 0.5, "REFRESH_K": 5, "STEPS": 1000, "NP": 16}' \\
    --paired-axis cpu+cuda
```

### Cascade C (HONEST CEILING): declare structural-ceiling DEFER + queue Z6/Z7/Z8

```bash
# Mark N1 PATH 5 paradigm DEFER (paradigm INTACT, IMPLEMENTATION SEED-DEPENDENT)
# per Catalog #307; cathedral autopilot routes to next-paradigm class-shift
# candidates per Catalog #311 (Z6/Z7/Z8 ego-motion-conditioned predictive coding)
# per the T3 council deferred-substrate-retrospective protocol Consequence 3.
# No further $ spend on this paradigm pending fundamental reformulation.
```

## Catalog #344 canonical equation actions

**Action 1 (anchor registration)**: register 7 new EmpiricalAnchor rows on
`pose_axis_score_direction_matching_paradigm_savings_v1`:

- 5 K-value anchors at seed=0 (K∈{1, 5, 10, 30, 100}); the K=10 anchor
  reproduces the predecessor's −5.02 single-sample but the K=1/5/30/100 anchors
  are NEW evidence of K-dependent composability at seed=0.
- 2 bootstrap variance anchors at K=10 / seeds {7, 13} surfacing the SNR=0.18
  paired Δpose noise distribution.

**Action 2 (no calibration_status change)**: status remains `FORMALIZATION_PENDING`
per Catalog #344. The reactivation criteria list `K_ablation_sweep_K_in_1_5_10_25_50_at_D_regime_100ep`
is now SATISFIED (although the predecessor planned K=25/50, the executed sweep
used K=30/100 — close but not identical sample points; THIS landing notes the
divergence). NEW reactivation criterion added: `8_seed_bootstrap_variance_sweep_at_K5_to_verify_signal_exceeds_noise_floor`
per Cascade A.

**Action 3 (composable_with_segnet_partial_confirmation_delta_pose field
NOT updated)**: the K=10 seed=0 anchor at Δpose=−5.02 stands as historical per
Catalog #110/#113 APPEND-ONLY, but the field is NOT promoted to a higher-
confidence value because the bootstrap variance verdict empirically shows the
field's predicate is seed-dependent. Future updates would require 8-seed
bootstrap SNR ≥ 1.0 OR paired-CUDA confirmation.

## Catalog #313 probe-outcomes ledger row update

- **NEW row** `mlx_pose_axis_score_direction_matching_path5_n1_kablation_20260528`
  with verdict **DEFER** (NOT PROMOTE despite classifier verdict ROBUST).
  Reactivation criterion: 8-seed bootstrap variance sweep at K=5 (Cascade A above).
  Metric: `k_ablation_composable_count_over_5_paired_with_bootstrap_variance`.
  Metric value: 5.0 (all 5 K composable at seed=0).
  Threshold token: `K_ABLATION_ROBUST_AT_SEED0_BUT_BOOTSTRAP_SEED_DEPENDENT_AT_K10_SNR_BELOW_1`.
- **OLD row** `mlx_pose_axis_score_direction_matching_path5_n1_20260527`
  verdict PARTIAL stands as historical anchor per Catalog #110/#113 APPEND-ONLY.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map**: ACTIVE — the per-K Δpose sensitivities (5 K-values)
  contribute to `tac.sensitivity_map.*` as per-pair pose-axis composable-direction
  sensitivity AT seed=0 only; bootstrap-3-seed variance excludes the signal from
  sensitivity-map promotion.
- **hook #2 Pareto constraint**: PARTIAL — at seed=0 the D regime expands the
  empirical Pareto frontier at all 5 K (both seg AND pose simultaneously improve).
  At bootstrap-3-seed mean, the D regime DOES NOT empirically expand the Pareto
  frontier (paired Δpose mean = +1.54 worsens net).
- **hook #3 bit-allocator**: N/A (probe does not produce per-byte signal).
- **hook #4 cathedral autopilot dispatch**: ACTIVE via canonical equation
  registry consumer auto-discovery (Catalog #335) — the equation will surface in
  the ranker's predicted-delta cascade as `[predicted]` axis_tag observability-only
  annotation; the autopilot ranker will see the NEW K-ablation anchors via the
  canonical equation consumer.
- **hook #5 continual-learning posterior**: ACTIVE via `register_probe_outcome`
  + `update_equation_with_empirical_anchor` calls invoked from the registration
  helper at `.omx/tmp/register_n1_path5_kablation_canonical.py`.
- **hook #6 probe-disambiguator**: ACTIVE — the canonical disambiguator between
  classifier-ROBUST (single-seed) vs bootstrap-SEED-DEPENDENT (3-seed) IS the
  paired Δpose SNR=0.18 statistic per this memo + the 5-K-value × 3-seed
  empirical anchor distribution.

## Constraints + discipline manifest

- **$0 GPU** verified — all measurement is local MLX-research-signal +
  CPU PyTorch teacher per Catalog #192/#127/#323. All output rows
  `score_claim=False` / `promotable=False` / `axis_tag="[macOS-MLX research-signal]"` /
  `evidence_grade="MLX-research-signal"` / `rank_or_kill_eligible=False` /
  `ready_for_exact_eval_dispatch=False`.
- **Catalog #340** sister-checkpoint guard: subagent owned ONLY new probe file
  (`.omx/tmp/pose_score_direction_matching_e2e_kablation.py`) + registration
  helper (`.omx/tmp/register_n1_path5_kablation_canonical.py`) + this landing
  memo + Catalog #313 + #344 ledger appends. DID NOT touch predecessor probe
  file (`.omx/tmp/pose_score_direction_matching_e2e.py`) or any
  `src/tac/substrates/_shared/mlx_score_aware/*` (Code-Frozen per N1 council
  verdict). DID NOT touch any of the 4 in-flight background subagent surfaces
  (paradox-closer / Round 2 self-reflection / Cascade B wave-2 / Layer 1 grammar
  fix). No collision detected.
- **Catalog #206** crash-resume: 5 checkpoints emitted at start/probe-authored/
  execute/B-reproduces/landing.
- **Catalog #229** premise verification: B regime at seed=0 reproduces predecessor
  154.0147 ↔ 154.0147 EXACTLY; D regime at seed=0 K=10 reproduces predecessor
  pose=148.99 ↔ 148.99 within ε=1.0. Probe scaffold validated.
- **Catalog #287** placeholder rejection: every rationale ≥4 chars + substantive.
- **Catalog #292** per-deliberation assumption surfacing: 2 assumptions surfaced
  explicitly in the council_assumption_adversary_verdict block above; both
  classified CARGO-CULTED with empirical justification.
- **Catalog #307** paradigm-vs-implementation: BOTH signal verdicts surfaced
  distinctly (NOT collapsed to single binary).
- **Catalog #315** OPTIMAL FORM discipline: the K-ablation sweep IS the canonical
  iteration step per the N1 PATH 5 paradigm; the bootstrap-SEED-DEPENDENT finding
  is the canonical "iteration produced empirical-residual > NULL_BAND" signal
  that the implementation IS NOT YET at OPTIMAL FORM.
- **Catalog #346** roster validation: T1 working-group exempt from roster
  completeness gate per Catalog #346 scope.
- **Catalog #348** retroactive sweep cadence (NEW gate): the K-ablation
  predecessor anchor at seed=0 (Δpose=−5.02) is preserved per Catalog #110/#113
  APPEND-ONLY; the bootstrap verdict ADDS evidence rather than supersedes the
  predecessor anchor.
- **8th MLX-first standing directive** honored: MLX-first probe + numpy-portable
  scoring helpers + faithful YUV6 via real MLX SegNet+PoseNet adapters.
- **13th OPTIMAL-TRIO standing directive** honored:
  - **AUTOMATED**: K-ablation probe is reusable scaffold for future K-ablation
    + bootstrap-variance + paradigm-iteration analyses.
  - **COMPOUNDING**: 7 new EmpiricalAnchor rows on canonical equation
    `pose_axis_score_direction_matching_paradigm_savings_v1` compound the
    historical empirical record; bootstrap-variance methodology compounds across
    future class-shift candidates.
  - **OPTIMAL**: the honest bootstrap-SEED-DEPENDENT verdict SAVES $0.30-0.50
    of paired-CUDA spend that would have produced NULL evidence; the
    operator-routable Cascade A 8-seed bootstrap at $0 is the next-optimal step.

## Files touched

- `.omx/tmp/pose_score_direction_matching_e2e_kablation.py` (NEW; K-ablation probe)
- `.omx/tmp/pose_score_direction_matching_e2e_kablation_summary.json` (NEW; machine-readable summary)
- `.omx/tmp/pose_score_direction_matching_e2e_kablation_run.log` (NEW; full execution log)
- `.omx/tmp/register_n1_path5_kablation_canonical.py` (NEW; canonical ledger registration helper)
- `.omx/research/n1_path_5_k_ablation_robustness_sweep_landed_20260528T005414Z.md` (THIS memo)
- `.omx/state/canonical_equations_registry.jsonl` (APPENDED 7 EmpiricalAnchor rows)
- `.omx/state/probe_outcomes.jsonl` (APPENDED 1 NEW probe-outcomes row)
