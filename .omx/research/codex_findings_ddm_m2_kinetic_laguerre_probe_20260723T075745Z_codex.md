---
title: Codex findings - DDM M2 kinetic Laguerre at-tolerance probe
date_utc: 2026-07-23T07:57:45Z
lane_id: lane_ddm_m2_kinetic_laguerre_probe_20260723
research_only: true
score_claim: false
promotion_eligible: false
not_a_candidate: true
main_landing_review_required: true
---

# Verdict

`KINETIC_LAGUERRE_REGISTERED_LADDER_FORMULATION_FALSIFIED_STAGE_A`

MEASURED: every one of the 72 preregistered n600 cells crossed the exact
v19b matched-fidelity error bound before completing the video. The smallest
lower bound was 3,137,421 errors, already 215 errors worse than v19b's
3,137,206 and 22.9278 times the Stage-A ceiling of 136,839. No Stage-A
winner exists.

The negative is deliberately scoped to
`FORMULATION:KINETIC_ANISOTROPIC_LAGUERRE_REGISTERED_LADDER`.
It is not a generator-family or operations-grammar closure. The broader family
remains open under the operator's optimal-form discipline.

# Authority and custody

- Delegated authority:
  `/Users/adpena/Projects/pact/.omx/tmp/codex_runs/ddm_m2_kinetic_laguerre_probe_20260723T064512Z.wrapped.prompt.txt`
- Authority SHA-256:
  `cb5bc9d90cc9285407cd830dc9c4f310aabb706ce437cfa4effab3c5499248a8`
- Typed config SHA-256:
  `a385e8c30dd41bc63d814e0c725137e8df124ed63a61261df5798ad46552a373`
- Final receipt:
  `.omx/research/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723T000000Z/ddm_m2_kinetic_laguerre_probe_receipt.json`
- Final receipt SHA-256:
  `cd5783261128ae4a748c4129342b6401d4696d6fab3a79fd9e63481cbb108ef6`
- Input target cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- Evidence axis: `[macOS-CPU cached-label advisory]` for Stage A.
- Pointer: `0.1910828242 [contest-CPU]`, unchanged.

The final output contains 292 files and 17,656,620 bytes: 144 immutable
programs, 144 per-cell checkpoints, two rung aggregates, the final receipt,
and its directory structure. Every cell reports real coder bytes, double
decode identity, a deterministic prediction-state hash, and sampled
NumPy-fp32 versus cKDTree bit identity.

# Measured matrix

| rung | registered cells | disposed | within 100,099 B | admitted |
|---|---:|---:|---:|---:|
| n64 compute-integrity | 72 | 72 | 72 | 0 |
| n600 cached-label | 72 | 72 | 45 | 0 |

MEASURED closest registered kinetic row:

| field | value |
|---|---|
| cell | `k512_d3_shared_chart_anisotropic_spd_spline_sites_weights_plus_sparse_regular_triangulation_flips` |
| exact program bytes | 83,992 |
| selected coder | Brotli q11 |
| exact error lower bound | 3,137,421 |
| frames needed to falsify matched fidelity | 295 / 600 |
| selected temporal segments | 9 |
| charged regular-triangulation events | 14,774 |
| Stage-A admission | false |

MEASURED minimum-byte n600 row was the K64 degree-1 shared-SPD kinetic
cell at 53,827 bytes. It crossed 3,146,232 errors after 134 frames, so its
rate advantage does not buy usable partition fidelity.

# Describe-line race and receiver disposition

The matched race is
`KINETIC_DID_NOT_REACH_V19B_MATCH_FIDELITY`.
No kinetic row reached equal-or-better d_seg than v19b, so no kinetic byte
winner exists to compare against the 100,099-byte describe-line home.
The live describe-line remains optimal among these measured forms.

Stage B is exactly `NOT_RUN_STAGE_A_GATE_CLOSED`:

- kinetic-only `(bytes, d_seg, d_pose)`: N/A;
- composed-with-v19b-corrections `(bytes, d_seg, d_pose)`: N/A;
- receiver materialization and frozen scorer forward: not executed;
- vehicle repoint: false.

This is a gate result, not missing work. Running the receiver or adding
correction credit after Stage-A failure would violate the preregistered
contract.

# Build-time defects caught and extincted

1. A first fire exposed rank-deficient six-dimensional pose regression in very
   short temporal segments. The final implementation uses centered Tikhonov
   regression, requires at least eight frames per segment, and refuses
   nonfinite or overflowing quantization.
2. macOS Accelerate emitted false floating warnings for small finite GEMMs even
   when results were bit-equal to scalar contractions. The final implementation
   uses explicit deterministic `einsum` contractions. An actual n600
   warning-as-error diagnostic passed all eight waterfill segment counts.
3. Qhull's randomized joggle path was removed. Exact lifted degeneracy now uses
   deterministic index-keyed perturbation.

The two pre-fix runs were stopped and preserved under
`.omx/tmp/codex_quarantine/` with `INVALID` names. They are not cited as
evidence and are not part of the landing.

# Reformulation queue - family remains open

The next kinetic-generator test must change the representation, not merely
increase this ladder:

1. fit generators directly against Fisher/margin-ranked boundary debt rather
   than per-class area/Morton quantiles;
2. use curvelet/shearlet boundary coordinates and the corrected inner-Jacobian
   realization law from the operator's 2026-07-19 directives;
3. permit birth/death and local site-count allocation at Road-Lane saddles
   instead of a global fixed-K prefix;
4. jointly fit a thin receiver-closed correction vocabulary in the common
   exact-R receiver before judging the broader operations-grammar family.

These are unmeasured optimal-form reformulations, not implied winners.

# Triality and system disposition

- DSL: typed `DDMKineticLaguerreAtToleranceProbeV1` config and strict execution
  authority receipt.
- DAG: `.omx/research/ddm_m2_kinetic_laguerre_probe_DAG_FEED_20260723.md`.
- Equation: N/A. No non-vacuous matched-error generator-rate law stabilized,
  so registering one would create false authority.
- Continual learning: one `KILL` probe-outcome row scoped to this exact
  formulation, with the reformulation queue as reactivation criteria.
- #366/J5: proceed unchanged; no vehicle repoint.

# Verification and re-derivation

Focused verification:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_probe_ddm_kinetic_laguerre_at_tolerance.py
# 18 passed
.venv/bin/python -m ruff check tools/probe_ddm_kinetic_laguerre_at_tolerance.py tests/test_probe_ddm_kinetic_laguerre_at_tolerance.py
# All checks passed
```

Exact re-derivation:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python tools/probe_ddm_kinetic_laguerre_at_tolerance.py --config .omx/research/configs/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723.json --output-directory .omx/research/ddm_m1_kinetic_laguerre_at_tolerance_probe_20260723T000000Z
```

# MAIN landing review

MAIN must re-derive the authority/config/input hashes, verify all 72 n600 cell
dispositions and immutable program hashes, confirm Stage B remained closed,
recompute the matched-fidelity race, inspect the false-authority labels, and
review the complete `7588b9c008..HEAD` branch diff before merge.
