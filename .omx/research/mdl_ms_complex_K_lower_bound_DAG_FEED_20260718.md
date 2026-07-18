---
title: "FEED-MDL-K-20260718 — concrete n600 digital-complex descriptions do not lower-bound universal K"
date: 2026-07-18
lane_id: lane_mdl_ms_complex_k_lower_bound_20260718
research_only: true
authority: "[macOS-CPU advisory] NON-PROMOTABLE"
score_claim: false
promotion_claim: false
pointer_moved: false
execution_used: false
verdict: FALSIFIED_AT_CLAIM_LEVEL__UNIVERSAL_K_THRESHOLD_INCONCLUSIVE
producer:
  - tools/measure_mdl_ms_complex_k_lower_bound.py
  - .omx/research/mdl_ms_complex_K_lower_bound_20260718.json
memo: .omx/research/mdl_ms_complex_K_lower_bound_20260718.md
---

# DAG FEED — MDL / K inequality correction and exact-description custody

## Feed payload

The requested edge

`measured MDL(MS complex) -> numeric lower bound on universal individual K`

is **REJECTED [DERIVED]** because its inequality is reversed. For fixed decoder `D_C`, the valid
candidate relation is

\[
K_U(T\mid D_C) \le L_C(T)+O_U(1).
\]

For evaluator `E`, exact witness `Y`, and evaluator target `T_E=(S,P)` with frozen PoseNet output
`P`, the true but uncomputable statistic complexity satisfies

\[
K_U(T_E\mid E) \le K_U(Y\mid E)+O_U(1).
\]

No concrete measured codelength may be substituted for the left-hand term in the second relation.
The measured carrier object `(S,xi_quantized)` is distinct: temporal ξ is not emitted by `E`.

## Producer → transform → consumer graph

```text
gt_n600.npz[lstars, gt_poses] (read-only, frozen scorer)
  + necessity exact-eps0 calibration (read-only)
  + n600 temporal-xi byte-close receipt (read-only)
  + context_partition_codec (existing exact Seg codec)
                         |
                         v
measure_mdl_ms_complex_k_lower_bound.py
  |-- target/class/hash custody
  |-- exact emitted Seg code-family upper bound
  |-- inherited optimistic digital-complex model audit
  |-- temporal-xi custody and nonzero-pose-distortion guard
  |-- strict integer sub-0.15 ceiling
  |-- universal-K premise falsification
                         |
       +-----------------+------------------+-------------------+
       v                                    v                   v
v10 / K-bracket design consumers   #536 / reverse-waterfill   equation review
consume achievable-from-above      consume ceiling only;      retire false MDL<=K;
description evidence only          K verdict INCONCLUSIVE     review corrected candidate
       |                                    |                   |
       +-----------------+------------------+-------------------+
                         v
exact parse-back MS/digital-complex codec + exact receiver closure
(OPEN BLOCKER; required before score/adoption/promotion)
```

## Typed feed rows

| feed id | producer fact | label | consumer action | authorization |
|---|---|---|---|---|
| `mdl-k-1` | `K_U(T|D_C) <= L_C(T)+O(1)` | DERIVED theorem candidate | canonical-equation reviewer | REVIEW ONLY; MAIN required |
| `mdl-k-2` | universal numeric K lower bound is nontrivial-unmeasured/uncomputable | DERIVED | all K-bracket consumers | remove numeric lower-rung claims |
| `mdl-k-3` | exact n600 temporal-context Seg payload is measured by the result receipt | MEASURED declared-family upper bound | v10/code-family search | evidence from above only; no promotion |
| `mdl-k-4` | `228764 B` is a post-Brotli `/2` estimate, not emitted shared-edge bytes | MEASURED code audit + DERIVED classification | necessity/K-ladder consumers | forbid “lossless K seed” promotion |
| `mdl-k-5` | optimistic Seg+ξ project model is `235974 B` | DERIVED | reverse-waterfill | one family misses ceiling; universal optimum inconclusive |
| `mdl-k-6` | strict largest integer rate-only byte ceiling is `225272 B` | DERIVED | rate-budget consumers | safe arithmetic constant only |
| `mdl-k-7` | ξ section is `7195 B` with measured nonzero `d_pose` | MEASURED | pose/K-bracket consumers | forbid `(0,0,K)` custody claim |
| `mdl-k-8` | raster argmax is a digital scorer-cell complex, not proven classical MS | DERIVED from available fields | MS/Laguerre consumers | require potential/Hessian/transversality for classical claim |

Every numeric row above is advisory and must be read with the detailed machine-readable receipt.

## Blockers and reactivation gates

1. **Emitted shared-edge codec [OPEN]:** frame/class/component framing, decoder, parse-back, and exact
   byte count must replace the nonlinear-compression `/2` estimate.
2. **Continuous-complex semantics [OPEN]:** exact logits/tie maxima or an explicit scalar potential,
   plus critical-point/nondegeneracy/transversality custody, are required for a classical
   Morse–Smale claim. Digital adjacency remains valid without this promotion.
3. **Receiver closure [OPEN]:** the seed must generate legal RGB bytes whose frozen evaluator statistic
   exactly matches the target; partition reconstruction alone is insufficient.
4. **Pose closure [OPEN]:** temporal ξ must close the realized Pose term to the claimed corner; the
   current measured section has nonzero `d_pose`.
5. **Model-restricted minimality [OPTIONAL OPEN]:** a lower bound inside a declared finite grammar
   requires exhaustive/proved minimality. It would remain grammar-restricted, not universal K.

## Triality and apparatus disposition

- **Equation:** candidate/debt only; MAIN must review the corrected inequality before registration.
- **DAG:** this file is the durable feed; no hot shared DAG was edited from the isolated lane.
- **DSL:** N/A — no runtime/trainer/launcher configuration exists in this cached measurement.
- **Sensitivity map / Pareto / bit allocator / autopilot:** no mutation. The result is a theorem
  correction plus advisory code-family evidence, not a scored marginal or dispatch authorization.
- **Continual learning:** consume the premise falsification and exact blockers through this feed and the
  main memo; do not ingest `235974 B` as K or as an attainable archive.
- **Probe disambiguator:** the executable measurement receipt separates real emitted Seg code,
  heuristic MS-model bytes, cached target poses, and temporal ξ. The remaining ambiguity is a theorem
  question resolved above, not a two-mode implementation choice.

## Pointer-delta honesty

`pointer_delta=NONE` · `score_claim=false` · `paid_dispatch=false` · `training=false` ·
`sacred_run_mutation=false`.

The feed changes the interpretation of prior description-length evidence, not the contest frontier.
