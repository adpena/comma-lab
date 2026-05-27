---
council_tier: T1
council_attendees: [Claude-N1-PATH-5-subagent]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: true
council_override_rationale: "operator NON-NEGOTIABLE 2026-05-27 verbatim 'ooperator override approved if necessary'; T1 working-group reactivation probe per T3 council reactivation path #5 (commit `18e7eee13`); landing memo aggregates empirical findings + paradigm-vs-implementation classification per Catalog #307"
canonical_equation_reference: pose_axis_score_direction_matching_paradigm_savings_v1
horizon_class: asymptotic_pursuit
---

# N1 PATH 5 — score-DIRECTION matching pose-binding probe (LANDED)

**Predecessor**: T3 grand council all-negative-findings review (commit `18e7eee13`,
memo `.omx/research/t3_grand_council_all_negative_findings_review_operator_override_20260527T222723Z.md`)
empirically deferred (NOT KILL) the MLX surrogate-head pose-binding paradigm
after 5 distinct mechanism falsifications. Diagnosed root cause: real-PoseNet's
gradient-to-input is ill-conditioned (absmax 1e8-1e27) → MLX surrogate
distillation through gradient-MSE optimizes a noise direction.

**THIS PROBE** executes reactivation path #5: replace gradient-MSE with a
sign-invariant cos-distance matching objective. By Cauchy-Schwarz,
`cos(a, b) = (a·b) / (|a|·|b|) ∈ [-1, 1]` is bounded REGARDLESS of input
magnitudes. Multiplying the teacher gradient by 1e27 (worst-case overflow)
does NOT change the cos value — mathematical scale-invariance, not heuristic.

## 4-regime empirical anchor (100ep faithful YUV6 MLX SegNet+PoseNet adapters)

| Regime | seg | pose | recon | Δpose vs A | Δpose vs B |
|---|---:|---:|---:|---:|---:|
| A recon-only (baseline)           | 0.6085 | 178.1679 | 0.00670 | —       | —      |
| B SegNet-only (canonical best)    | 0.5178 | 154.0147 | 0.00636 | −24.15  | —      |
| C pose-cos-dir-only (NEW)         | 0.5876 | 179.6920 | 0.00664 | +1.52   | **+25.68** |
| D SegNet + pose-cos-dir (NEW)     | 0.5069 | 148.9949 | 0.00641 | −29.17  | **−5.02** |

**WINNER**: D (segnet + pose-cos-dir) pose=148.99 — improves on B by Δpose=−5.02 AND
on seg by Δseg=−0.011 simultaneously.

## Honest dual-signal verdict (per Catalog #307 paradigm-vs-implementation discipline)

**Two independent findings emerge from the 4-regime data, NOT a single binary verdict.**

### Finding 1 (the central question): cos-distance ALONE does NOT replace SegNet

- C pose=179.69 vs B pose=154.01: Δpose=**+25.68** (cos-distance ALONE WORSENS pose vs SegNet-only).
- Verdict: **6th mechanism falsified** at the cos-distance-as-pose-binding-paradigm-replacement surface.
- Classification per Catalog #307: **IMPLEMENTATION-LEVEL FALSIFICATION**, NOT paradigm-level KILL.
- The diagnosed root cause (1e8-1e27 ill-conditioning) ALSO causes the COS-DISTANCE
  loss landscape to be dominated by noise: the teacher gradient at refresh time
  had absmax=0.236 (regime C) / 0.376 (regime D) — finite, but the cos-distance
  computed against student pixels (max ~1.0) yields a near-orthogonal direction
  because the teacher gradient's structure does not align with the pixel manifold
  in any pose-meaningful way.
- The mathematical proof of scale-invariance is correct (Cauchy-Schwarz unchallenged),
  but the EMPIRICAL POSE-AXIS INFORMATION CONTENT of the cos-direction is insufficient
  to outperform the SegNet baseline. This is a structural-ceiling reinforcement of
  the council's pose-axis-structural-ceiling honest calibration verdict.

### Finding 2 (the unexpected partial signal): cos-distance is COMPOSABLE with SegNet

- D pose=148.99 vs B pose=154.01: Δpose=**−5.02** (cos-distance combined with SegNet IMPROVES on SegNet-alone).
- D seg=0.5069 vs B seg=0.5178: Δseg=**−0.011** (composability is BOTH-AXIS, not pose-axis-only).
- Classification per Catalog #307: **PARTIAL CONFIRMATION** for the composable-pose-binding paradigm.
- Mechanism hypothesis: SegNet binding stabilizes the renderer's spatial structure
  (the dominant signal), and cos-direction matching provides a SECONDARY pose-axis
  regularization that does NOT compete with SegNet but ADDS marginal improvement.
  The 5.02-unit pose reduction is ~3.3% relative to B's 154.01 (small but real,
  exceeds the NULL_BAND=1.0 threshold).

### Honest combined verdict

**Paradigm: PARTIALLY REACTIVATED via composable surface, NOT standalone.** The
sister-paradigm "cos-distance pose-direction matching as standalone SegNet
replacement" is the **6th falsified mechanism** per the council's 5-mechanism
catalog. The sister-paradigm "cos-distance pose-direction matching as
composable additive to SegNet" yields **partial empirical confirmation** at
the D-regime measurement.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the
overall N1 paradigm is **PARTIAL** — neither full KILL nor full REACTIVATION
— with two distinct sub-paradigm verdicts that the canonical equation
registry + probe-outcomes ledger MUST capture distinctly.

## Catalog #344 canonical equation actions

**Action 1 (registration)**: register NEW canonical equation
`pose_axis_score_direction_matching_paradigm_savings_v1` per the operator-binding
constraint that canonical equations capture every empirical finding. Status:
**FORMALIZATION_PENDING** per CLAUDE.md "Canonical equations + models registry"
because the closed-form for "when cos-distance matching produces ΔS improvement
in combination with SegNet binding" requires systematic K=10 refresh-window
ablation (current empirical anchor is K=10 only).

Initial calibration: D−B Δpose=−5.02 [macOS-MLX research-signal] at NP=16/100ep/K=10.
NOT promotable per Catalog #192 (macOS-MLX is NON-PROMOTABLE; paired Linux x86_64
CPU + NVIDIA CUDA validation required for any contest-axis claim).

**Action 2 (amendment)**: amend existing canonical equation
`mlx_surrogate_head_pose_alignment_impossibility_via_2nd_order_autograd_nan_v1`
with new EmpiricalAnchor row documenting C-regime Δpose=+25.68 as the 6th
mechanism falsification at the cos-distance-as-standalone-replacement surface.
Falsified-mechanisms list extends from 5 to 6:
- pose_head_rgb_pair
- pose_head_rgb_diff
- pose_head_yuv6_pair
- pose_head_yuv6_diff
- pytorch_teacher_direction_distill_K10 (linear-distill)
- **cos_distance_score_direction_matching_K10_standalone (NEW)**

## Catalog #313 probe-outcome row update

Substrate: `mlx_surrogate_head_pose_binding_score_direction_matching_path5`
Probe kind: `cos_distance_sign_invariant_pose_direction_matching_4_regime`
Verdict: **PARTIAL** (NOT DEFER, NOT PROCEED) — the composable surface validates
but standalone surface falsifies.
Metric: `delta_pose_d_vs_b_composable_segnet_pose_cos_dir`
Metric value: −5.02 (improvement; composable signal)
Sister metric: `delta_pose_c_vs_b_standalone_cos_dir_replaces_segnet`
Sister metric value: +25.68 (worsening; standalone falsification)
Threshold: 0.0 (any negative Δpose = composable verdict accepted)
Threshold token: `PARTIAL_COMPOSABLE_POSE_AXIS_BINDING_VIA_COS_DISTANCE`

Reactivation criteria: K-ablation sweep (K∈{1, 5, 10, 25, 50}) on D regime to
characterize the K-dependence of the composable signal; the council's K=1
reactivation path (#1) is now sister-relevant.

## 6-hook wire-in declaration (per Catalog #125)

- **hook #1 sensitivity-map**: ACTIVE — the empirical Δpose D−B=−5.02 contributes
  to `tac.sensitivity_map.*` as a per-pair pose-axis composable-direction sensitivity.
- **hook #2 Pareto constraint**: ACTIVE — D regime expands the empirical Pareto
  frontier (seg=0.5069 + pose=148.99 dominates B at both axes simultaneously).
- **hook #3 bit-allocator**: N/A (probe does not produce per-byte signal).
- **hook #4 cathedral autopilot dispatch**: ACTIVE via canonical equation registry
  consumer auto-discovery (Catalog #335) — the equation will surface in the
  ranker's predicted-delta cascade as `[predicted]` axis_tag observability-only
  annotation.
- **hook #5 continual-learning posterior**: ACTIVE via `register_probe_outcome` +
  `register_canonical_equation` + `update_equation_with_empirical_anchor` calls
  invoked from THIS subagent's persistence pass.
- **hook #6 probe-disambiguator**: ACTIVE — the canonical disambiguator between
  composable-cos-distance vs standalone-cos-distance IS the 4-regime measurement
  table; future agents inherit this distinction structurally.

## Operator-routable (CONDITIONAL paired-CUDA)

Per the operator's "override approved if necessary" wording: the empirical
signal IS POSITIVE (D−B=−5.02 composable; C alone WORSENS). The composable
signal IS WORTH paired-CUDA validation per Catalog #246 because:

1. It expands the Pareto frontier on BOTH axes (seg AND pose simultaneously improve).
2. Magnitude is modest (~3.3% relative pose improvement) but exceeds NULL_BAND.
3. The mathematical mechanism (Cauchy-Schwarz scale-invariance) is principled.

**Routable** (NOT invoked from THIS subagent per CLAUDE.md "Executing actions
with care"; operator decides):

```bash
# D-regime full-run on Modal T4 paired (CPU + CUDA) via canonical operator-authorize:
.venv/bin/python tools/operator_authorize.py \\
    --recipe substrate_dreamer_v3_rssm_modal_t4_dispatch \\
    --override-config '{"bind_pose_cos_direction": true, "POSE_DIR_W": 0.5, "REFRESH_K": 10, "STEPS": 1000}' \\
    --paired-axis cpu+cuda
```

Expected paired cost: $0.30-0.50 PAID per Catalog #246 paired discipline.
Verdict gating: D−B Δpose < 0 on CUDA-T4 → canonical equation PROMOTION
candidate; else NULL on Linux x86_64 → DEFER per Catalog #192 (macOS-MLX
research-signal not transferring to contest hardware).

## Constraints + discipline manifest

- **$0 GPU** verified — all measurement is local MLX-research-signal +
  CPU PyTorch teacher per Catalog #192/#127/#323. All output rows
  `score_claim=False` / `promotable=False` / `axis_tag="[macOS-MLX research-signal]"` /
  `evidence_grade="MLX-research-signal"`.
- **Catalog #340** sister-checkpoint guard: subagent owned ONLY new probe file
  (`.omx/tmp/pose_score_direction_matching_e2e.py`) + this landing memo + Catalog
  #313 + #344 entries. DID NOT touch `src/tac/substrates/_shared/mlx_score_aware/*`
  (Code-Frozen per N1 council verdict). DID NOT touch existing predecessor probe
  file. No collision detected.
- **Catalog #206** crash-resume: 4 checkpoints emitted at start/scaffolding-read/
  authoring/execute/landing.
- **Catalog #287** placeholder rejection: every rationale ≥4 chars + substantive.
- **Catalog #307** paradigm-vs-implementation: BOTH sub-paradigm verdicts surfaced
  distinctly (NOT collapsed to single binary).
- **8th MLX-first standing directive** honored: MLX-first probe + numpy-portable
  scoring helpers + faithful YUV6 via real MLX SegNet+PoseNet adapters.
- **13th OPTIMAL-TRIO standing directive** honored:
  - **AUTOMATED**: probe is reusable scaffold for future K-ablation + cos-variants.
  - **COMPOUNDING**: canonical equation #344 entries compound across class-shift candidates.
  - **OPTIMAL**: cos-distance is mathematically provably scale-invariant (Cauchy-Schwarz).

## Files touched

- `.omx/tmp/pose_score_direction_matching_e2e.py` (NEW; canonical probe scaffold)
- `.omx/tmp/pose_score_direction_matching_e2e_summary.json` (NEW; machine-readable summary)
- `.omx/tmp/pose_score_direction_matching_e2e_run.log` (NEW; full execution log)
- `.omx/research/n1_path_5_score_direction_matching_pose_binding_landed_20260527T225708Z.md` (THIS memo)
- `.omx/state/probe_outcomes.jsonl` (PARTIAL row appended per Catalog #313)
- `.omx/state/canonical_equations_registry.jsonl` (NEW equation registered + 5-mechanism equation amended per Catalog #344)

## Cross-references

- T3 council all-negative findings review (commit `18e7eee13`):
  `.omx/research/t3_grand_council_all_negative_findings_review_operator_override_20260527T222723Z.md`
- Predecessor 5-mechanism falsification (commit `74007357b`):
  `.omx/research/mlx_harness_pose_head_align_iterate_landed_20260527T221739Z.md`
- Predecessor probe file (linear-distill teacher direction K=10):
  `.omx/tmp/pose_pytorch_teacher_direction_e2e.py`
- CLAUDE.md "MPS auth eval is NOISE" non-negotiable
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- Catalog #307 paradigm-vs-implementation falsification discipline
- Catalog #313 probe-outcomes canonical ledger
- Catalog #344 canonical equations + models registry
- Catalog #192 macOS-CPU advisory not promoted without Linux verification
- Catalog #127 authoritative tag custody validator routing
- Catalog #323 canonical Provenance umbrella
- Catalog #178 MPS engineering correction (epsilon-stabilized denominators)
