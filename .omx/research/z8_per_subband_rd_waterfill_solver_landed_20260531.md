# Z8 per-subband RD water-fill solver LANDED 2026-05-31

`[macOS-CPU advisory]` — NON-PROMOTABLE planner artifact per Catalog #192/#341/#127/#323.
$0 macOS-CPU, no GPU, no PR.

## What landed

`src/tac/substrates/z8_hierarchical_predictive_coding/per_subband_rd_waterfill_solver.py`
(~330 LOC) + 27 NO-FAKE tests. The canonical Lagrangian rate-distortion
water-fill solver for the Z8 wavelet detail payload — the "only open knob" the
detail-coeff entropy-headroom report (`z8_detail_entropy_headroom_20260531T185438Z.json`,
sister landing `7066bf63c`) identified as the remaining Z8 rate lever.

This is the rate-axis per-subband Δ component of #1591 (unified Lagrangian
water-filling solver) / #1592 (joint P18/P19 water-fill). It is a pure **extend**
of the existing executable actuator
(`joint_coefficient_waterfill.py::apply_joint_p18_p19_deadzone_to_z8_archive`):
the solver emits that actuator's existing `entropy_detail_quantization_steps`
input rather than re-implementing any coefficient mutation or archive packing.

## Mechanism

1. `load_subband_rd_curves_from_report` — parse the 6 per-(level,orientation)
   RD curves (`L0_hh/hl/lh`, `L1_hh/hl/lh`); each curve = keep-raw baseline +
   5 measured `quant_sweep` operating points (Δ, bytes/coeff, MSE).
2. `_pareto_frontier` — lower-convex-hull reduction (Shoham-Gersho 1988). The
   raw RD points are NON-MONOTONE in Δ (the live codec switches method at coarse
   Δ → bytes go UP at Δ=0.25 vs Δ=0.125), so dominated points are dropped and
   only convex-hull vertices survive — the only points a Lagrangian can select.
3. `solve_per_subband_waterfill` — minimize `D_total + λ·R_total`; per subband
   pick the hull point minimizing `mse_i·n_i + λ·bytes_i·n_i`; bisect λ to hit a
   byte budget OR a distortion ceiling (both monotone in λ). Distortion is the
   Parseval coefficient-count-weighted mean MSE.
4. `emit_actuator_quant_steps` — frame-agnostic Δ applied to both
   `frame_0_details`/`frame_1_details`; keep-raw subbands omitted; every key
   round-trips through the actuator's own `_parse_entropy_detail_step_key`.

## Empirical RD operating points (REAL report; baseline raw-f32 detail = 939,178 B/pair)

| operating point | detail bytes | % of raw | wMSE |
|---|---|---|---|
| near-lossless (wMSE≤2e-5) | 210,531 | 22.4% | 1.97e-5 |
| low-dist (wMSE≤5e-5) | 172,381 | 18.4% | 4.76e-5 |
| mid (wMSE≤2e-4) | 123,089 | 13.1% | 1.96e-4 |

The solver makes heterogeneous per-subband decisions (near-lossless keeps
`L1_hl`/`L1_lh` raw while quantizing all L0 subbands) — the optimum the prior
single-global-Δ actuator could not express.

## Honest scope

This solver allocates the RATE axis (per-subband Δ) from the report's measured
RD curves. It does NOT itself measure the contest scorer — the emitted Δ map is
a proposal; the full-video inflate/eval replay through the actuator ratifies, and
exact CPU/CUDA auth-eval signs any score. The detail blob is ~99.5% of the Z8
archive but Z8 remains ~24-546× from the contest frontier per the sister
dead-zone landing `ad73c2863` / 600-pair landing `76eea75e5`; this lever cannot
close the structural gap alone. DEFER-pending: compose with the joint P18/P19
scorer-saliency surface (`joint_coefficient_waterfill.py` already wires it) so
the RD allocation is scorer-protection-aware, not magnitude-only.

## 6-hook wire-in declaration per Catalog #125

- #1 sensitivity-map = N/A (consumes the report's measured RD curves directly)
- #2 Pareto constraint = ACTIVE (the solver IS a discrete-RD Pareto allocator)
- #3 bit-allocator = ACTIVE PRIMARY (per-subband Δ IS the bit allocation)
- #4 cathedral autopilot dispatch = N/A (advisory planner; non-promotable)
- #5 continual-learning posterior = N/A (no score claim emitted)
- #6 probe-disambiguator = ACTIVE (byte-budget vs distortion-ceiling modes are
  the canonical disambiguator between rate-target and quality-target operating
  points)

mission_predicted_contribution = `frontier_breaking_enabler`;
horizon_class = `frontier_pursuit`. Sister-DISJOINT per Catalog #340 (only the
new solver file + its test + this memo; consumes the existing actuator + report
read-only).
