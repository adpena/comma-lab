# Codex Source Doc Structure and Compliance Map

Date: 2026-04-30

## Purpose

This document maps Codex implementation progress to the source-of-truth
planning documents. It is used to determine whether work is:

- scientifically aligned with the Grand Council plan,
- contest-grade under the evidence audit,
- ready for shortest-wall-clock execution,
- sufficiently rigorous for adversarial review.

## Source Documents

| Document | Role | Compliance Question |
|---|---|---|
| `grand_council_paradigm_shift_to_shannon_floor_20260430.md` | Scientific and mathematical strategy | Does the implementation match the intended alpha/beta/gamma hypothesis and score math? |
| `contest_grade_all_lane_results_audit_20260430.md` | Evidence grading and 1:1 contest compliance | Is this result allowed to rank, promote, kill, or anchor floor claims? |
| `shannon_floor_execution_readiness_20260430.md` | Dispatch and wall-clock ordering | What should run next, and what can parallelize? |
| `external_research_intake_shannon_floor_20260430.md` | Papers/OSS intake | Is an outside idea Copy, Translate, or Watch for this contest? |
| `shannon_floor_paper_rigor_writeup_blueprint_20260430.md` | Paper/writeup rigor blueprint | Are claims mapped to evidence grades, formulas, artifacts, ablations, source docs, and review gates before publication? |

## Progress Ledgers

| Progress Ledger | Mirrors | Current Use |
|---|---|---|
| `grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md` | Grand Council paradigm plan | Tracks implementation against alpha/beta/gamma strategy. |
| `contest_grade_all_lane_results_audit_20260430_codex_progress.md` | Contest-grade audit | Tracks evidence impact and grade changes. |
| `shannon_floor_execution_readiness_20260430_codex_progress.md` | Execution readiness | Tracks active blockers and next-turn dispatch plan. |

## Writeup Blueprint

| Blueprint | Mirrors | Current Use |
|---|---|---|
| `shannon_floor_paper_rigor_writeup_blueprint_20260430.md` | Contest audit, execution plan, Grand Council strategy, external intake | Defines the publication-quality claim matrix, reproducibility artifact pack, adversarial review gates, ablation table shells, and non-claim boundaries for the Shannon-floor push. |

## Current Cross-Reference

| Work Item | Grand Council Alignment | Contest Audit Status | Execution Status |
|---|---|---|---|
| Sensitivity-map artifact | Beta foundation: sensitivity-aware everything | Implementation only, no score evidence | Landed and tested; needs CUDA artifact generator/converter |
| OWV3 mixed-channel archive | Beta: protect PoseNet-sensitive channels while recovering OWV2 rate | Implementation-smoke only; not Grade A | Encoder/decoder, builder, provenance, registry, and inflate dispatch landed; needs real CUDA sensitivity artifact and exact eval |
| `.nrv` auth whitelist | Alpha: mask payload overhaul | Compliance unblock only; not score evidence | Landed and tested |
| `.nrv` mask resolver and canonical manifest | Alpha: prevents SegNet fallback and lets canonical archive validation accept `masks.nrv` | Compliance unblock only; not score evidence | Landed and tested; clean contest dependency closure still needs remote proof |
| PFP16 exact eval | Pose-stream deterministic crumb | Empirical/predicted until CUDA eval | Highest-priority exact eval |
| Lane 12 NeRV full run | Alpha highest rate leverage | Empirical-only until exact CUDA archive eval | Run after clean dependency closure |
| Lane 17 IMP | Renderer branch / hidden-gem recovery | Empirical-only until exact archive eval | Harvest active dispatch |
| Gamma ADMM / hyperprior-lite | Gamma coordinator | Not ready for score claims | Defer until measured alpha + beta streams exist |

## Determination Rules

1. **Complete implementation** means:
   - code exists,
   - deterministic builder exists where an archive is needed,
   - payload closure is proven,
   - tests cover success and failure modes,
   - provenance is emitted,
   - exact archive eval path is callable.

2. **Scientifically rigorous** means:
   - objective maps to the contest score formula,
   - parameters and thresholds have derivation, sweep, or adversarial-review
     justification,
   - claims are tagged as derivation/prediction/synthetic/empirical/score-grade,
   - kill and promote criteria are falsifiable.

3. **Contest compliant** means:
   - exact `archive.zip` is preserved,
   - SHA-256 is recorded,
   - manifest contains only known files,
   - no local sidecars are used at inflate,
   - upstream scorer is unmodified,
   - 600 samples score recomputes,
   - CUDA/T4-equivalent status is recorded.

4. **Optimized for shortest wall-clock** means:
   - deterministic small wins run immediately,
   - independent hypotheses run in parallel,
   - stack experiments wait for individually measured components,
   - failed lanes are classified as scientific kills or engineering bugs,
   - hidden-gem recovery runs where failure was infrastructure, not hypothesis.

## Adversarial Review Requirements

Before any strategic promotion or kill:

1. Shannon/math pass: recompute score, rate slopes, and distortion tradeoff.
2. Scorer-sensitivity pass: inspect PoseNet/SegNet failure modes.
3. Artifact-custody pass: verify archive, hash, manifest, and logs.
4. Engineering pass: inspect silent defaults, fallbacks, sidecars, dependency
   closure, and no-op encoders.
5. Optimization pass: compare against marginal-score-per-byte alternatives and
   stack conflicts.

Three consecutive clean passes are required. Any issue resets the counter.
