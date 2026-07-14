# Third-order boundary and finite receiver-space descent-filler — n600 MEASURE

**Date:** 2026-07-13/14 UTC

**Role:** `third_order_descent_filler` · `$0` cached analysis · no training · no paid/remote/live-run actuation

**Axis:** `[macOS-CPU advisory]` NumPy authority for the measured arrays and deterministic byte codecs; **not** contest-CPU/CUDA score authority

**Primary receipt:** `.omx/research/third_order_descent_filler_20260713.json`

**Reproducer:** `tools/measure_third_order_descent_filler.py`

> **{ARM-1 third-order: `M3,max = 0.922745805 m^-2` MEASURED on 2,945 real-n600 Road/Lane quartic-check fits; the all-boundary strict no-flip tolerance is `epsilon=0 px`, while the Road/Lane scoped strict tolerance is `2.1457672e-6 px` · the incumbent 3-jet/current coherent LBND2 is `41,303 B`; a 2-jet LBND2 is `34,402 B` unmatched, but exact correction to the same cubic Lane mask raises it to `164,757 B`, so cubic wins this pre-receiver matched control by `123,454 B` · third-order saving is REAL only as an incumbent-vs-ablation label-space fact and NO as a new V9 lever, because the current coder already stores `c3` and no jet stream has a V9 RGB-through-R consumer. ARM-2 V9·CGauge descent-filler: the requested pointwise `d_cov+d_gauge` gluing gap remains NOT-TYPED; the closest receiver-shaped PHAS1 candidate has `1,264,814` nonzero quantized residual states on `1,287,364` ground straddles and `4,285/6,703` triple-junction sites, costs `1,006,377 B = 54.5403x` the `147,616-bit / 18,452-B` D36 quantity, and its xi predictor worsens the residual stream by `4.7866%` · this candidate is NOT D36, NOT a minimal filler, and INERT. Overall neither arm supplies a new byte-close-worthy candidate; the score pointer is unchanged.}**

## 0. Authority correction and stores consulted

The live inbox directive at `2026-07-14T00:15:35Z` is acknowledged and applied. The live vehicle is **V9·CGauge**, one covariant trunk. The old v8 per-class texture-carrier framing is not used: texture was dropped; Movable is predominantly event content, while the remaining debt is the gauge zero-mode. ARM 2 therefore asks whether the single V9 covariant section can be glued across the registered `d_seg=d_cov+d_gauge` split.

`STORES CONSULTED` before deciding:

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, and the v7.5/v8 canonical specs;
- `.omx/research/local_global_descent_dig_20260713.md` and `.omx/research/local_global_descent_dig_DAG_FEED_20260713.md`;
- `.omx/research/ladder_owed_measurables_20260713.md` and its DAG FEED (D36–D39);
- `.omx/research/bousfield_deep_read_20260713.md`;
- `.omx/research/dpose_covariance_mirror_audit_20260711.md`, `.omx/research/v9_cgauge_truly_optimal_design_20260712.md`, L87 in `.omx/state/canonical_equations_registry.jsonl`, and `tac.canonical_equations.einstein_pass_covariance_laws_20260710.dseg_covariant_residual`;
- `.omx/research/residual_kit_deshare_curverel_build_20260709.md`, `.omx/research/v8_roadlane_geometric_rate_20260709.md`, and `.omx/research/v8_roadlane_ego_compensated_rate_20260709.md`;
- the real n600 frozen-SegNet cache and the preserved V9 checkpoint listed in §5.

The key recalled type distinction remains binding: D36 is a measured operational conditional codelength for a code-table proxy. D38 asks whether local sections are globally effective. An infeasible gluing choice has no finite codelength; only a typed, nonempty filler space can be packed and measured.

## 1. Measurement contract

The tool uses all 600 cached scorer states, `600*384*512 = 117,964,800` scorer-grid sites. It performs no scorer forward and makes no score claim.

1. **#275 flip radius.** At every genuine-V right/down straddle, the cached margin ratio is

   `t = m_p / (m_p + m_q + 1e-6)`,

   and the linearized displacement that reaches the nearest sampled endpoint is

   `epsilon_i = min(t, 1-t)` pixels.

2. **Boundary derivatives.** Real Road/Lane clusters are mapped to the incumbent ground-frame chart. Cubic fits measure `|y'''|=|6 c3|`. Independent degree-4 fits expose the missing remainder: `M3_i = sup_[a,b] |24 c4 s + 6 c3|`, `M4_i=|24 c4|`, and numerical arclength `L_i`.

3. **Actual bytes.** The current coherent-slot LBND2 grammar is used with Brotli quality 11. Quadratic and cubic outputs are round-tripped through the real decoder. An ABS2 correction flips the quadratic Lane mask exactly to the cubic Lane mask, producing a true matched **pre-receiver label-map** control.

4. **V9 phase candidate.** PHAS1 is sourced from the real n600 tie fields and the actual V9 effective twist `pose_carrier.xi_stored + pose_carrier.dxi`. The twist is passed directly to `encode_phase_carrier`; it is already rho-first se(3), so PoseNet-output calibration is not re-applied.

The exact command was:

```bash
.venv/bin/python tools/measure_third_order_descent_filler.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --v9-ckpt experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz \
  --out .omx/research/third_order_descent_filler_20260713.json
```

## 2. ARM 1 — third order

### 2.1 `M3` and `epsilon`

| Quantity | Status | n600 result |
|---|---|---:|
| all-class genuine-V straddles | MEASURED | 1,325,485 |
| strict all-class `epsilon_min` | MEASURED | **0 px** |
| Road/Lane genuine-V straddles | MEASURED | 617,212 |
| strict Road/Lane `epsilon_min` | MEASURED | **2.1457672e-6 px** |
| Road/Lane `epsilon` p0.1 / p1 / median | MEASURED | 0.000769579 / 0.007707323 / 0.293566674 px |
| incumbent cubic `|6c3|` median / p99 / max | MEASURED | 9.34414e-5 / 0.0410998 / 0.237931 `m^-2` |
| quartic-check `M3=sup|y'''|` median / p99 / max | MEASURED | 0.000844247 / 0.154376 / **0.922746 `m^-2`** |
| quartic-check `M4=|y''''|` median / p99 / max | MEASURED | 3.17879e-5 / 0.0320277 / 0.391734 `m^-3` |

The all-class uniform tolerance is zero because at least one real boundary is already at the sampled tie. Therefore **no finite uniform all-separatrix Taylor count follows**. This is not a numerical nuisance; it is the argmax discontinuity the requested tolerance must respect. The Road/Lane subproblem has a positive empirical minimum and supports a scoped count.

This `M3` is not falsely promoted to the entire V9 separatrix. It is the real n600 Road/Lane ground-frame family consumed by the incumbent lane coder. Movable/event boundaries, horizon-only boundaries, and codimension-2 junction singularities are outside this derivative-fit scope.

### 2.2 Taylor count law, now with the third-order remainder typed

For a quadratic local approximation and a cubic local approximation on the same arc,

`epsilon_m,i = epsilon_px * forward_min,i / fx` (the conservative near-end IPM scale),

`N2_i >= L_i * (M3_i / (6 epsilon_m,i))^(1/3)`, and

`N3_i >= L_i * (M4_i / (24 epsilon_m,i))^(1/4)`.

The IPM conversion is necessary because the measured flip radius is in render pixels while the fitted curve is in ground metres. The third line is also necessary: comparing 2-jet and 3-jet counts from `M3` alone would be mathematically incomplete because the cubic remainder is controlled by `M4`.

| Road/Lane tolerance | Status | `sum N2` | `sum N3` | quadratic terms `3N2` | cubic terms `4N3` | cubic/quadratic term ratio |
|---|---|---:|---:|---:|---:|---:|
| strict empirical min `2.14577e-6 px` | DERIVED | 2,172,734 | 318,419 | 6,518,202 | 1,273,676 | **0.1954** |
| distributional p0.1 `0.000769579 px` | DERIVED, not uniform | 307,072 | 74,298 | 921,216 | 297,192 | 0.3226 |
| distributional p1 `0.00770732 px` | DERIVED, not uniform | 143,248 | 42,405 | 429,744 | 169,620 | 0.3947 |

The local Taylor accounting predicts fewer cubic terms. That does **not** establish a rate win: the present grammar is one global polynomial per detected line, not the theorem's piecewise local cover, and it already includes the cubic coefficient.

### 2.3 Actual byte measurement

| Code | Counted bytes | Pre-receiver Lane-mask error vs GT | Match status | `d_seg` through R |
|---|---:|---:|---|---|
| 2-jet coherent LBND2 | **34,402 B** | 577,961 states / 0.004899436 | not matched to cubic | **NOT MEASURED** |
| 3-jet coherent LBND2 | **41,303 B** | 559,011 states / 0.004738795 | current cubic target | **NOT MEASURED** |
| current coder | **41,303 B** | same as 3-jet | current coder **is** the 3-jet row | **NOT MEASURED** |
| 2-jet + exact ABS2 correction to cubic Lane mask | **164,757 B** (`34,402 + 130,355`) | bit-identical to cubic Lane mask | matched pre-receiver | **NOT MEASURED** |

The quadratic and cubic decoded Lane masks disagree on 207,022 states (0.001754947 of all n600 states). At exactly matched cubic label-mask fidelity, the current cubic stream is **123,454 B smaller** than quadratic plus correction. The extra `c3` stream is therefore rate-efficient in this control. It is not a new saving: the 41,303-B current coder already banks it.

**ARM-1 scoped verdict — `NO_NEW_LEVER / REAL_INCUMBENT_ABLATION`.**

`verdict_scope = FORMULATION x PRE_RECEIVER_ROAD_LANE x n600 x current coherent LBND2`. Third order is a real reason to retain the incumbent cubic code, but no additional third-order V9 candidate exists and matched `d_seg` through R is not measured.

`req-R to reopen as a new lever:` define a distinct receiver-consumed jet grammar, render it into V9 RGB, and produce a same-archive-budget n600 R+SegNet A/B. This negative does not kill higher-order boundary coding outside the incumbent Road/Lane grammar.

## 3. ARM 2 — finite receiver-space descent filler in V9·CGauge

### 3.1 What the current V9 state can and cannot type

L87 registers the scalar distortion identity

`d_seg = d_cov + d_gauge`, with `d_cov = max(0, d_total-d_gauge)`.

It does **not** expose two pointwise decoded fields, transition functions, overlap restrictions, or a receiver projection whose disagreement support can be evaluated. Consequently the requested finite gluing gap over the `d_cov+d_gauge` split is still **NOT-TYPED**. Subtracting two scalar distortions cannot manufacture a spatial obstruction support.

The closest landed receiver-shaped object is PHAS1: a quantized tie-coordinate residual on decoder-derivable ground straddles, transported by V9's stored twist. It is measured below as a candidate upper bound, never as the obstruction or its minimum code.

### 3.2 Candidate support and bytes

| Quantity | Status | n600 result |
|---|---|---:|
| ground-straddle support | MEASURED | 1,287,364 states = 1.091312% of all states |
| nonzero quantized residual support | MEASURED | **1,264,814** = 98.248359% of active = 1.072196% of all |
| active support Road / Lane / Undrivable | MEASURED | 649,336 / 337,094 / 300,934 |
| 2x2 triple-junction sites | MEASURED | 6,703 |
| triple junctions on active ground support | MEASURED | 6,542 |
| triple junctions with nonzero ground residual | MEASURED | **4,285** |
| PHAS1 section | MEASURED | **1,006,377 B** |
| residual bytes with V9 xi predictor | MEASURED | 990,037 B |
| residual bytes with constant/raw-tie predictor | MEASURED | 944,813 B |
| xi amortization ratio | MEASURED | **1.0478656** (predictor is 4.7866% worse) |
| q-step reconstruction | MEASURED | bit-identical; tie RMSE 0.0718903 px |
| recovered `d_seg` after RGB -> R -> SegNet | NOT MEASURED | PHAS1 consumer is absent / explicitly OWED |

The support is real, including codimension-2 concentration, but it is a **pre-receiver tie-residual support**, not the support of a measured difference between pointwise `d_cov` and `d_gauge` sections. The V9 twist does not amortize it; it adds 45,224 residual bytes relative to the constant predictor before section overhead.

### 3.3 D36 comparison

| Object | Bytes | Typed meaning |
|---|---:|---|
| D36 `H(q_G|U_proxy)` operational upper-bound stream | **18,452 B / 147,616 bits** | conditional code-table proxy; receiver predictor not admitted |
| V9-twist PHAS1 candidate section | **1,006,377 B** | pre-receiver tie residual; no RGB consumer |
| difference | **+987,925 B** | arithmetic only; not an obstruction identity |
| ratio | **54.5402666x** | arithmetic only; not evidence of equality |

There is no typed map `q_G -> filler`, no sufficiency theorem, and no minimality proof. Hence:

- **not free:** the measured PHAS1 candidate is nonzero;
- **not D36:** it is 54.54x larger and codes a different random variable;
- **not the finite obstruction:** the true filler space and receiver projection remain absent;
- **inert as an engineering candidate:** it costs more than an entire contest archive and has no measured `d_seg` recovery.

**ARM-2 scoped verdict — `INERT_CANDIDATE / OBSTRUCTION_STILL_NOT_TYPED`.**

`verdict_scope = INSTANCE x PHAS1_v1 x V9_EFFECTIVE_TWIST x PRE_RECEIVER x n600`. This kills dispatch or byte-close work on the current PHAS1 packet, not the finite receiver-space descent-filler family.

`req-R to reopen the family:` land a pointwise, decoder-derived V9 `(covariant section, gauge zero-mode)` projection plus overlap restrictions; demonstrate a nonempty filler choice; then pack it and measure its exact receiver-consumed RGB/R/SegNet delta. No equality to D36 is admissible without a typed map and minimal-code proof.

## 4. Triality and apparatus wire-in

### Equation leg

The measured inputs now make the conditional Taylor pair well-typed:

`epsilon_m,i = epsilon_px forward_min,i/fx`,

`N2(epsilon) >= sum_i L_i (M3_i/(6 epsilon_m,i))^(1/3)`, and

`N3(epsilon) >= sum_i L_i (M4_i/(24 epsilon_m,i))^(1/4)`.

No canonical equation was registered. The law is only scoped to the measured Road/Lane quartic-fit cover, while full-boundary `epsilon_min=0` and receiver-consumed rate are unresolved. Registering a universal boundary-rate law would overstate the evidence.

### DAG leg

```text
gt_n600 {lstars,margins}
  +-- #275 tie radius ----------------------> epsilon distribution
  +-- Road/Lane ground chart
        +-- degree-4 remainder -------------> M3, M4, L -> N2/N3 bounds
        +-- degree-2 LBND2 -----------------> 34,402 B --+
        +-- degree-3 coherent LBND2 --------> 41,303 B <-+ exact-mask control

V9 checkpoint {xi_stored,dxi}
  + gt_n600 tie fields
        +-- PHAS1 direct twist encode ------> 1,006,377 B
        +-- no V9 RGB consumer -------------> recovered d_seg = OWED

d_seg=d_cov+d_gauge scalar law
  +-- no pointwise state/restrictions ------> finite filler = NOT-TYPED
```

No separate canonical DAG FEED was appended because neither arm produced a clean receiver-closed law. This memo and its JSON receipt are the durable `research_only=true` terminal for the measurement.

### DSL / action leg

- No training scalar, flag, or launch config was introduced.
- No paid dispatch, GPU, live trainer, or evaluator was touched.
- The measurement is deterministic, n600, and outputs only a small JSON receipt; it creates no bulky scratch requiring cold-store cleanup.
- Sensitivity contribution: ARM 1 isolates 207,022 Lane-mask states attributable to removing `c3`; ARM 2 isolates 1,264,814 nonzero phase-residual states and 4,285 triple-junction sites.
- Pareto constraint / bit allocator: reject current PHAS1 because rate is +1,006,377 B with unknown distortion benefit; retain incumbent cubic because the matched pre-receiver ablation costs +123,454 B.
- Cathedral/autopilot hook: **no dispatch row**; receiver-closure prerequisites are explicit fail-closed gates.
- Continual learning: bank “Taylor term reduction does not imply a new rate lever when the incumbent already carries the term” and “a scalar covariance split cannot type a spatial filler.”
- Probe disambiguator: the quadratic-plus-ABS2 exact-mask control arbitrates the only two defensible boundary-order readings; the current cubic grammar wins that control.

## 5. Provenance and pointer delta

| Field | Value |
|---|---|
| measurement git SHA | `a05a4b18c3c4d951ef48e41b4f7dc3060e174c95` (shared worktree dirty; no unrelated files absorbed) |
| GT cache | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` |
| GT cache SHA-256 | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` |
| V9 checkpoint | `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_BEST.npz` |
| V9 checkpoint SHA-256 | `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c` |
| tool SHA-256 | `21d67072c7682ecde2013496d4453b9956c99210af775e37ed56a875093bd1c5` |
| receipt SHA-256 | `d3b0f02a5be594babbc023926118ce42b4e43ab053c16d67ea0be52f70593c5b` |
| NumPy | `1.26.4` |
| elapsed | 70.9 s CPU |

Verification:

- full n600 reproducer: **PASS**, exit 0, 70.9 s;
- Ruff + `py_compile`: **PASS**;
- CPU-targeted boundary/codec tests: **64 passed** (`47 + 17`), with MLX cases deselected;
- broad four-file test attempt: **64 passed / 17 failed**, where every failure was the same headless `No Metal device available` environment blocker; no CPU/NumPy assertion failed.

**Pointer delta honesty:** no archive was built or evaluated; no exact CPU/CUDA row exists; the frontier pointer is unchanged. **Neither arm is byte-close-worthy in its present form.** ARM 1 is already banked in the current cubic coder; ARM 2 fails the rate stop rule by more than fiftyfold against D36 before receiver closure.
