# sub015 DAG FEED — exponential-linear witness warm-start

UTC: 2026-07-14T17:28:43Z  
Lane: `lane_exp_linear_reparam_warmstart_20260714`  
Contract: `PAPER_WARM_START_FROM_DIVERGENCE`  
`research_only=true`

## Executable dependency graph

```text
[paper/source custody]
          |
          v
[paper-vs-witness divergence fork]
          |
          v
[actual ep650 checkpoint + real GT + feature-state hashes]
          |
          v
[anchored mismatch inverse: exact W/rate/d_seg identity] ---- MEASURED
          |
          v
[deterministic 2x2 MLX: AdamW, Muon, SEL+AdamW, SEL+Muon] -- BLOCKED: Metal custody
          |
          v
[CPU/numpy through-R batch32 verdict at each boundary]
          |
          +---------------------+
          |                     |
          v                     v
[matched-d_seg step gate]  [int8+Brotli + distribution gate]
          |                     |
          +----------+----------+
                     v
        [SEL-over-Muon additive/redundant verdict]
                     |
                     v
    [typed DSL + canonical equation legs after drain] ------ HELD
                     |
                     v
       [resume/fold/parse-back receiver closure]
```

## Node state

| Node | State | Receipt / blocker |
|---|---|---|
| paper/source custody | complete | arXiv `2607.09967` |
| divergence fork | complete | findings memo |
| real artifact custody | complete | checkpoint, GT, and feature-state SHA-256 in common-start JSON |
| anchored conversion | complete | exact effective identity, exact 62,087-byte identity, exact through-R d_seg identity |
| probe implementation | complete | `tools/probe_exp_linear_reparam_warmstart_mlx.py` plus unit tests |
| 2×2 execution | blocked | headless MLX cannot load a Metal device, including MLX CPU selection |
| matched-step verdict | waiting | requires all four traces |
| terminal rate verdict | waiting | requires all four traces |
| DSL/equation implementation | held | contested shared trees; V9 provenance owner retains custody |
| promotion / heavy launch | forbidden | advisory containment and no measured landing |

## Preregistered transitions

1. Execute/resume all four arms to step 24 with CPU/numpy batch32 d_seg every two steps.
2. Require that each non-SEL control's terminal d_seg strictly improves from the common start, then set that terminal value as its matched target.
3. Record first step at or below target. `fewer_steps` requires a strict step reduction and equal-or-better terminal d_seg.
4. Classify `ADDITIVE_ON_THIS_SMOKE` only when SEL+Muon passes step 3 over Muon.
5. Compare exact int8+Brotli bytes and magnitude statistics at each arm's first matched-basin row; retain fixed-step terminal deltas as secondary telemetry.
6. If both fewer-step and rate/objective admission gates pass, route the held typed lever to the provenance owner after the shared-tree drain.

## Canonical consumers after a measured anchor

- Sensitivity map: treatment effect on matched d_seg step count and exact blob bytes.
- Pareto surface: `(steps_to_target, d_seg, blob_bytes)` with axis and formulation token.
- Bit allocator: terminal magnitude/entropy delta by parameter group.
- Cathedral/autopilot: admit only a typed stage-boundary action with resume and fold receipts.
- Continual learning: append the scoped contrast, including a negative or blocker token; do not infer across formulation or hardware axes.
- Probe disambiguator: keep `fixed` and `annealed` scale-LR schedules as explicit modes if the fixed treatment is negative.

No consumer is wired from the present blocker because doing so would turn an unmeasured prior into false authority.

## Triality / pointer delta

- DSL leg: spec present, implementation **HELD**.
- DAG leg: this file.
- Equation leg: `ΔW≈-ηJ_fJ_fᵀ∇_W L`, with local metric `(J_fJ_fᵀ)⁻¹`; canonical code **HELD**.
- Pointer delta: **NONE**.
