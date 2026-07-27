# DAG FEED — closed scorer variational DE

UTC 2026-07-21T17:36:54Z · `research_only=true` · pointer unchanged · MAIN review required.

Append under the existing Einstein-Kolmogorov `SOLVE -> DESCRIBE -> BYTE-CLOSE -> EXACT-EVAL`
bridge. This feed does not authorize dispatch.

```text
FEED-CLOSED-SCORER-DE-20260721
  D1 TASK-SPACE CLOSURE [MEASURED advisory]
    inputs:
      segnet_head_rank4_linear_flipdist_v1
      fisher_curvature_equals_categorical_fisher_trace_caustic_v1
      frozen SegNet + gt_n600, seed=1234
    equation: closed_scorer_taskspace_variational_functional_v1
    receipt: closed_scorer_variational_de_20260721T173654Z.json
    gate: 20 held-out real tiles; native-f32 power and Bregman residual both 0
    scope: final-head task geometry only; no inverse/receiver/rate/score authority
    |
    v
  U1 CLOSED STATIONARITY DE [DERIVED, witness owed]
    equation: closed_scorer_viscosity_kkt_stationarity_v1
    operators:
      Seg: viscosity/subgradient HJ on Laguerre separatrices
      Pose: calibrated SE(3) geodesic pullback in xi
      Rate: exact MDL gradient; lambda_rate=25/37545489; hard-cap mu_B distinct
    required gates:
      flip_margin_step_law_v1 corrected secant/QP realization
      decoder/R/PoseNet pullback custody
      exact entropy-to-archive residual
    |
    v
  U2 LEGAL-DESCRIPTION LOWER BOUND [OPEN]
    equation: closed_scorer_archive_reachability_bound_v1
    exact arithmetic:
      B_cap=154600
      rate_at_cap=0.10294179415268769
      distortion_allowance_for_sub015=0.04705820584731231
    REFUSE:
      proxy entropy == archive bytes
      0.118 empirical achiever == exact infimum
      177169-byte witness as feasible under 154600 cap
    owed:
      explicit legal description language + valid relaxation/lower bound
    |
    v
  U3 S_STAR / REACHABILITY [UNRESOLVED_REQUIRES_BYTE_CLOSED_WITNESS]
    required evidence:
      archive.zip <=154600 bytes + sha256
      receiver parse-back + deterministic inflate + runtime custody
      exact realized-through-R n600 Seg/Pose
      contest-CPU and contest-CUDA kept as separate axes
    terminal outputs:
      exact S* bound/witness OR typed unresolved blocker
```

Reverse-waterfill admission is surgical: target only nonzero necessity strata, rank on Fisher/margin,
use corrected realization secants, use curvelet/shearlet for any residual carrier, and stop when
marginal score improvement per byte falls below `25/37,545,489`.

Pointer delta: **ZERO**. This feed becomes launch-relevant only after typed DSL/LawRef binding and the
ordinary lane/SSD/resume/per-stage-checkpoint preflights.
