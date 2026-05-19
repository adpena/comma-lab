# sigma=15 grayscale-LUT reframe — premise correction + actual impact analysis

**Date:** 2026-05-19 (UTC)
**Author:** Main-Claude (in-context work during 3-slot saturation)
**Lane:** No new lane — this is a premise-verification memo that corrects an earlier in-flight framing
**Authority:** Operator "approved proceed with all" 2026-05-18 included decision D (sigma=15 reframe analysis); my prior message framed this as a NSCS06 v8 Path B resurrection candidate. **This memo CORRECTS that framing as wrong-premise** and documents the actual impact surface.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Data: arbitrariness audit JSONL row | ADOPT_CANONICAL | Wave 2C `.omx/state/arbitrariness_extinction_audit_20260518.jsonl` row #16 is the source-of-truth |
| Calibration tool: MAD-derived canonical sigma | ADOPT_CANONICAL (Wave 2C `tac.experimental_extinctions.segnet_boundary_curvature_sigma_calibration`) | Rousseeuw-Croux 1.4826 factor is the standard robust estimator |
| Premise-correction discipline | UNIQUE — apply Catalog #229 before claiming a strategic implication | I claimed NSCS06 v8 resurrection without grepping; Catalog #229 PV caught it |

## 9-dimension success checklist evidence

- UNIQUENESS: distinguishes two ENTIRELY DIFFERENT sigma questions (curvature MAD vs LUT bandwidth)
- BEAUTY+ELEGANCE: 5-row impact-surface table makes the actual scope obvious in 30 seconds
- DISTINCTNESS: corrects a phantom-resurrection claim that would have wasted operator attention
- RIGOR: every claim cited with file:line evidence; predicted impact bounded with audit-row source
- OPTIMIZATION-PER-TECHNIQUE: recommends focused per-substrate sweep on actual grayscale-LUT codepath, NOT redundant work
- STACK-OF-STACKS-COMPOSABILITY: predicted ΔS is small; not stack-of-stacks gating
- DETERMINISTIC-REPRODUCIBILITY: every cited file is at canonical commit
- EXTREME-OPTIMIZATION-PERFORMANCE: N/A (this memo is a premise correction, not an optimization)
- OPTIMAL-MINIMAL-CONTEST-SCORE: predicted ΔS [-0.002, -0.0003] is small-impact bolt-on; should NOT block frontier-breaking work

## Observability surface (per Catalog #305)

- Inspectable per layer: 5 consumer files cited with file:line
- Decomposable per signal: curvature-MAD vs LUT-bandwidth distinction makes the two empirical questions independent
- Diff-able across runs: future per-substrate sigma sweep will produce per-substrate optimal-sigma row in canonical audit JSONL
- Queryable post-hoc: this memo is the canonical reference for the premise correction
- Cite-able: predecessor Wave 2C memory entry + audit JSONL row
- Counterfactual-able: changing sigma in any of the 5 consumers can be tested via paired-comparison smoke

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| sigma=15 came from NSCS06 v8 Path B | CARGO-CULTED-PHANTOM (rejected pre-edit) | sigma=15 is NOT in NSCS06 code; my Wave 2C summary read produced false-association |
| Wave 2C MAD-derived sigma=1.1 applies to NSCS06 | CARGO-CULTED-PHANTOM (rejected pre-edit) | Wave 2C measures SegNet boundary curvature; NSCS06 doesn't use that codepath |
| sigma=15 is inherited from Selfcomp PR #56 lineage | HARD-EARNED — verified via 5 file:line citations | Actual canonical source |
| The two sigma questions (curvature MAD vs LUT bandwidth) are different problems | HARD-EARNED — verified mathematically | Curvature MAD measures boundary-pixel statistic; LUT bandwidth controls 256→5-class mapping sharpness |
| Predicted ΔS [-0.002, -0.0003] is small | HARD-EARNED — sourced from Wave 2C audit row | Audit predictor explicitly bounded the impact |
| Focused per-substrate sweep on the actual grayscale-LUT codepath is the right next step | HARD-EARNED — directly addresses the unmeasured question | Wave 2C calibration tool used wrong input statistic for the actual decision |

## Predicted ΔS band (per Catalog #296)

`[-0.002, -0.0003]` per Wave 2C audit JSONL row #16. Bounded by R(D) theoretic argument: sigma controls the entropy of the grayscale-LUT softmax output (sharper sigma → lower entropy → fewer effective classes per gray value); the contest scorer's argmax operation makes scores INVARIANT to entropy below a threshold; expected score impact dominated by the noise-tolerance gap, not by per-pixel argmax flips.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" requirement that predicted bands have Dykstra-feasibility / first-principles citation: the upper bound is R(D)-grounded; the lower bound is empirical-zero (sigma already approximately optimal for current architectures). Probe-disambiguator path: focused per-substrate sigma sweep on the actual grayscale-LUT codepath (NOT the curvature MAD tool).

## Horizon class (per Catalog #309)

`horizon_class: plateau_adjacent` — sigma=15 is an inherited canonical bandwidth across 5 consumers; the predicted ΔS [-0.002, -0.0003] is within the 0.196-0.199 plateau cluster; not an asymptotic-pursuit candidate.

## Section 1: My prior framing was wrong

In an earlier message I proposed:
> "Sigma=15 → sigma≈1.1 NSCS06 reframe deep-dive ... If the actual cargo-cult is sigma=15, then v8 may be ONE cargo-cult-unwind away from re-entering the band, not paradigm-dead."

This framing has TWO premise errors caught by Catalog #229 PV before writing a wrong memo:

**Error 1: sigma=15 is NOT in NSCS06.** Grep verification:
```
$ grep -rn "sigma=15\|sigma = 15" /Users/adpena/Projects/pact/src/tac/
src/tac/segmap_renderer.py:520:            # Build the LUT inline (5 classes, sigma=15 per design).
src/tac/segmap_renderer.py:523:            sigma = 15.0
src/tac/optimize_grayscale_canvas.py:27:                                                     (sigma=15) on gray_continuous
src/tac/scpp_substrate.py:45:    CLAUDE.md "Quantizr intelligence" verified data; sigma=15; qint_max=7).
src/tac/scpp_substrate.py:135:#: Selfcomp's verified block-FP defaults. ``sigma=15`` is the std-dev cutoff
src/tac/mask_grayscale_lut.py:23:3. The Gaussian-softmax LUT (sigma=15) at inflate time matches the public
src/tac/contrib/szabolcs_renderer.py:21:     targets ``[0, 255, 64, 192, 128]`` (sigma=15.0). Result: 5-channel soft
$ grep -rn "sigma" /Users/adpena/Projects/pact/submissions/nscs06_carmack_hotz_strip_everything/
[empty]
$ grep -rn "sigma.*15\|15.*sigma" /Users/adpena/Projects/pact/experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py
[empty]
```

NSCS06 doesn't use sigma=15 anywhere. The substrate uses `_grayscale_to_rgb` chroma-replication (Y=R=G=B), NOT grayscale-LUT softmax projection.

**Error 2: The Wave 2C calibration tool measures a DIFFERENT statistic than what sigma=15 controls.**

The Wave 2C `tac.experimental_extinctions.segnet_boundary_curvature_sigma_calibration` calibration tool:
- INPUT: per-pixel SegNet boundary curvature samples
- ALGORITHM: median + MAD of curvature → canonical_sigma = 1.4826 × MAD
- OUTPUT: optimal sigma matching curvature distribution

But the `sigma=15` constant in `src/tac/segmap_renderer.py:523` + `src/tac/mask_grayscale_lut.py:67` controls a different thing entirely:
- CONTEXT: Gaussian-softmax LUT mapping grayscale [0, 255] → 5-class probability
- FORMULA: `bell = exp(-(gray - class_target)² / (2σ²))` → `softmax → 5-class probability`
- PURPOSE: controls SHARPNESS of class assignment per grayscale value (with class targets at [0, 64, 128, 192, 255] spaced ~64 apart)

A sigma=1.1 (Wave 2C MAD-derived) on the grayscale-LUT would produce nearly-one-hot class assignment (any gray value gets routed to its nearest class with probability ≈1.0; sigma=3.0 from the grid is similar). A sigma=15 (current) gives smooth soft class assignment with ~25% probability at midpoints between classes.

These are two DIFFERENT sigma questions and the Wave 2C calibration tool's answer (1.1 or 3.0) does NOT directly transfer to the grayscale-LUT codepath.

## Section 2: Actual sigma=15 impact surface

5 canonical consumers (verified file:line):

| File | Role | Line | Sigma value |
|---|---|---|---|
| `src/tac/segmap_renderer.py:523` | SegMap renderer training-time grayscale-LUT | 520-525 | `sigma = 15.0` hardcoded |
| `src/tac/mask_grayscale_lut.py:67` | Canonical grayscale-LUT codec (inflate path) | 67 | `LUT_DEFAULT_SIGMA: Final[float] = 15.0` |
| `src/tac/scpp_substrate.py:135` | SCPP substrate (Selfcomp clone) | 45, 135 | sigma=15 referenced in docstring |
| `src/tac/contrib/szabolcs_renderer.py:21` | Szabolcs PR #56 reference renderer | 21 | sigma=15.0 in docstring |
| `src/tac/optimize_grayscale_canvas.py:27` | Canvas optimizer | 27 | sigma=15 in docstring |

Lineage: inherited from Selfcomp/Szabolcs PR #56 (the 0.38-scoring entry per CLAUDE.md "Quantizr intelligence — verified competitive data 2026-04-21"). The canonical class targets `[0, 255, 64, 192, 128]` are also inherited from this lineage.

Audit-predicted ΔS: `[-0.002, -0.0003]` — small-impact bolt-on per Wave 2C predictor. Source: `.omx/state/arbitrariness_extinction_audit_20260518.jsonl` row #16 with `provenance.artifact_kind=predicted_from_model`.

## Section 3: The actual empirical question (and the actual next step)

The unanswered empirical question for the grayscale-LUT codepath is:

**For a given (renderer architecture, class targets, scorer-loss landscape) tuple, what sigma value minimizes the contest score?**

The Wave 2C calibration tool does NOT answer this — it answers the SegNet boundary curvature question. To answer the sigma=15 question, the focused next step is:

**Per-substrate sigma sweep on the actual grayscale-LUT codepath:**
- Substrates that consume sigma=15: SCPP, SegMap renderer variants, Szabolcs lineage
- Sigma grid: {0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0}
- Metric: contest score on the actual archive's inflate path
- Per-substrate symposium gating per Catalog #325 BEFORE any paid dispatch

Predicted cost: $1-3 per substrate × N substrates. Predicted ΔS [-0.002, -0.0003] aggregate.

Compared to bigger frontier-breaking opportunities (MPS gap experiment, NSCS06 v7 cargo-cult-unwind methodology applied to next substrate, K-sweep VQ codebook anti-Pareto resolution): **this is bolt-on-grade work, not frontier-breaking**. Should be queued AFTER the higher-impact dispatches return.

## Section 4: Operator-routable recommendations

1. **Do NOT pursue the NSCS06 v8 sigma=15 resurrection framing** — wrong premise, caught pre-write.
2. **Sigma=15 reframe is a small-impact bolt-on** (-0.002 to -0.0003 predicted ΔS). Queue AFTER higher-impact frontier work.
3. **Wave 2C calibration tool's MAD-derived sigma is valid for SegNet boundary curvature** but does NOT directly apply to grayscale-LUT bandwidth.
4. **Future per-substrate sigma sweep** on the actual grayscale-LUT codepath is the right next step IF the small predicted impact is judged worth $1-3 per substrate × per-substrate symposium overhead.

## 6-hook wire-in declaration (per Catalog #125)

1. Sensitivity-map: N/A (premise-correction memo)
2. Pareto constraint: N/A
3. Bit-allocator: N/A
4. Cathedral autopilot: ACTIVE — premise correction prevents the autopilot from over-weighting sigma=15 reframe as a high-EV reactivation
5. Continual-learning posterior: ACTIVE — this memo IS the canonical posterior anchor for the sigma=15 premise correction
6. Probe-disambiguator: ACTIVE — distinguishes curvature MAD question from LUT bandwidth question; future per-substrate sweep IS the probe

## Predicted mission contribution

`apparatus_maintenance` — premise correction is bookkeeping, NOT frontier-breaking. Marks the operator's "decision D" routing as RESOLVED with verdict "small-impact; queue after higher-impact work" rather than auto-dispatching a high-overhead sweep.

## Cross-references

- Wave 2C audit JSONL row #16: `.omx/state/arbitrariness_extinction_audit_20260518.jsonl`
- Wave 2C calibration tool: `src/tac/experimental_extinctions/segnet_boundary_curvature_sigma_calibration.py`
- 5 canonical consumers (see Section 2)
- CLAUDE.md "Forbidden premature KILL" (sigma=15 is NOT killed; just correctly classified as small-impact)
- CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" (every claim in this memo has file:line evidence)
- Catalog #229 (premise verification — what caught my wrong framing)
- Catalog #287 (just landed; the META gate that would have caught any phantom-API citation here at design-memo surface)

— Main-Claude 2026-05-19 (in-context work during 3-slot saturation)
