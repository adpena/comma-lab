---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "micro-batch-pairs is a 2-4x speed lever (the #261/#293 borrowed number transfers to the live config)"
    classification: CARGO-CULTED
    rationale: "MEASURED FALSE at the LIVE config: n24 A/B (B in {1,2,4,8}, wa+seed ON, 384x512, full lever stack) shows FLAT-to-SLOWER t_step (no monotone B-trend for B<=4; B=8 clearly slower ~16s vs ~7-11s). The 384x512 SegNet forward already saturates the Apple GPU at B=1, so batching yields no utilization headroom while adding lazy-graph + wa-2nd-forward overhead. The recursive-review AXIS-9 lesson: measure the SCORED/timed quantity at the real config, never trust the ancestor number."
  - assumption: "batching B>1 busts memory at n600 (waterfill excludes B>1)"
    classification: HARD-EARNED-WITH-SCOPE
    rationale: "MEASURED benign at n24: peak_rss FLAT across B (3393/3388/3361/3422 MiB for B=1/2/4/8). The batched forward activation is a small transient freed per accum group; resident cf_cache+gt+verdict dominate. n600 waterfill (B-independent resident model) = 67.6 GiB SAFE. Micro-batch adds a bounded forward transient projected negligible; not the binding constraint. Binding constraint is throughput (no win), not memory."
council_decisions_recorded:
  - "op-routable 1: micro-batch-pairs is now UNBLOCKED (routes wa/focal/bd/eik-stab) but NOT recommended for relaunch — n24 shows no speedup. Leave OFF (B=1) until a clean single-workload multi-epoch median (or a smaller-render regime) demonstrates a >15% win."
  - "op-routable 2: the equivalence contract holds (loss <=6.4e-8, grad <=1.8e-7 MLX-CPU, batch=1==single BITWISE EXACT) — correctness is not the blocker; throughput is."
  - "op-routable 3: still fail-closed vs micro-batch (out of scope, correct): --margin-saliency-reachability, --seg-spike-reweight."
---

# MICRO-BATCH TWIN ROUTING (task #313) — 4 levers routed, equivalence proven, n24 shows NO speedup

**2026-07-05. $0-ish local (n24 governed probes, GPU idle single-workload). Axis: all timing rows
`[macOS-MLX advisory] NON-PROMOTABLE`. Pointer contest-CPU 0.19110 UNMOVED — everything here is
MEANS (a compute lever), never a score.** Sibling #315 (event/curriculum) edits the SAME trainer in a
disjoint region (curriculum/stage-boundary); this task owns the micro-batch twin + its fail-close block.

## What was blocked (the binding wall)
`--micro-batch-pairs` (#261/#293) fail-CLOSED against the LIVE v4 lever set: the trainer raised
`NotImplementedError` for `--witness-alone-island-loss` (THE live-config blocker), `--seg-focal-gamma>0`,
`--boundary-distance-weight>0`, and the EIK-STAB stabilizers (`--eikonal-viscosity` /
`--eikonal-steik-weight`) because the importable batched twin
`tac.boundary_math.levelset_micro_batch_loss` did not carry those legs. #293 had already built the
seed-co-grad batched path; wa-island was the last binding blocker for the live config.

## What was built (4 legs routed, faithfully)
Each leg mirrors `total_loss_fn` op-for-op; trainer-local helpers are passed as callables in
`LeverConfig` (avoids a tac<-trainer import cycle, keeps the math bit-identical to the canonical fn):

1. **FOCAL** (`--seg-focal-gamma`): the `focal_pixel_weight_mlx` reweight folds into EVERY seg form's
   per-pixel map BEFORE the mean, exactly as `make_loss_fn` does. Passed the SAME callable => one math.
2. **BOUNDARY-DISTANCE** (`--boundary-distance-weight`): SDF-native `boundary_distance_term_mlx` on
   `model.sdf(cf,c1)` + the per-pair band provider. Per-pair, batches trivially.
3. **EIK-STAB** (`--eikonal-viscosity` + `--eikonal-steik-weight`): ViscoReg residual REPLACES the
   eikonal residual while eps>0 (not additive — no double-count); StEik damping ADDS. The `_eik_stab`
   dict is passed BY REFERENCE so the batched twin reads the per-epoch viscosity anneal live.
4. **WITNESS-ALONE ISLAND** (`--witness-alone-island-loss` #300a): when the flag is set + a
   seed-excluded `render_fn_wa` is supplied + an island lever (amplify/persist) is engaged, a SECOND
   batched SegNet forward over the K witness-alone f1 frames feeds the island levers' margin/logits;
   base + non-island levers keep the seed-composed forward. Off => island levers alias the composed
   forward (byte-identical, no 2nd forward). Symmetric in `single_realized_loss` for the A/B baseline.

The 4 `NotImplementedError` fail-closes are removed. STILL fail-closed (out of scope, correct):
`--margin-saliency-reachability`, `--seg-spike-reweight`.

## Equivalence contract — MEASURED (the NO-FAKE gate)
Bar (honest): NOT bitwise-vs-serial (batched changes fp summation ORDER); instead (a) per-pair loss
bitwise-equal at batch=1; (b) accumulated grad within a MEASURED fp tolerance; (c) trajectory A/B.

- **(a) batch=1 == single: BITWISE EXACT** (167088.890625 == 167088.890625, MLX-CPU).
- **(b) batched grad == mean-of-per-pair grad** (MLX-CPU, batch-independent scorer):

  | leg | loss rel-err | grad L2 rel-err |
  |---|---|---|
  | base ce/tau/l7/margin_hinge (K4) | <=2.6e-8 | <=6.0e-8 |
  | focal (K3) | 2.2e-8 | 1.5e-7 |
  | boundary-distance (K4) | 2.8e-8 | 4.6e-8 |
  | eik-viscosity (K3) | 6.3e-8 | 1.3e-7 |
  | eik-steik (K3) | 6.2e-8 | 1.8e-7 |
  | wa-amplify (K3) | 6.4e-8 | 9.1e-8 |
  | wa-persist (K3) | 2.5e-8 | 1.4e-7 |
  | ALL-4 stacked (K3) | 6.3e-8 | 1.6e-7 |

  All grad L2 rel-err <= 1.8e-7 — well inside fp32 and far below any optimizer step scale (lr 1e-3,
  grad-clip 1.0). Justification: the only re-order is the mean-over-B; the frozen SegNet/PoseNet are
  batch-independent in eval mode (running-stat BN, per-pixel conv, per-frame head).
- **Tests: 47 pass** (`src/tac/tests/test_levelset_micro_batch_loss.py`; 21 pre-existing + 26 new — every
  routed leg's equivalence, a canonical-`make_loss_fn` focal-parity gate, "leg actually moves the loss"
  NO-FAKE guards, wa "routing is real not aliased" + "wa-off byte-identical regardless of render_fn_wa"
  + "no island lever => no 2nd forward"). The #293 real-path proof (`test_batched_seed_cograd.py`, real
  gt + real frozen scorer) still passes 7/7 (no regression).
- **(c) n24 trajectory + throughput A/B (real trainer, wa+seed live-config levers, GPU idle):** see below.

## Throughput A/B — n24, real trainer, wa+seed ON `[macOS-MLX advisory] NON-PROMOTABLE`
`t_step_fwd_bwd_opt_ema_s` (verdict excluded), 6 ep, muon@4, `experiments/results/mb_twin_ab_20260705/`:

| B | ep3 (pre-muon) | ep6 (muon) | peak_rss |
|---|---|---|---|
| 1 (serial) | 6.98 | 11.37 | 3393 MiB |
| 2 | 11.16 | 7.18 | 3388 MiB |
| 4 | 11.44 | 11.58 | 3361 MiB |
| 8 | 15.51 | 17.51 | 3422 MiB |

**HEADLINE (honest): NO measurable speedup.** For B<=4 the t_step is NOISE-swamped (range 6.98-11.58 s,
no monotone B-trend — B=1 and B=2 literally flip which epoch is faster; GPU thermal/warmup variance,
R_isolated ref_fwd swings 2.3-6.0 ms/frame between runs). B=8 is clearly SLOWER (~16-17.5 s). The
original #261 "2-4x" does NOT reproduce at the LIVE config. Mechanism: the 384x512 SegNet forward already
saturates the Apple GPU at B=1 (large single-frame conv), so batching gives no utilization headroom while
adding lazy-graph size + (with wa) a 2nd batched forward. This is the recursive-review AXIS-9 lesson in
action — the batched twin was FAITHFULLY built and PROVEN correct, but the ancestor speed number did not
transfer to the real config; measure the timed quantity at the real config, never inherit it.

**Memory: FLAT + SAFE.** peak_rss is flat across B (+-1%); the batched forward activation is a small
transient freed per accum group. n600 waterfill projection (B-independent resident model,
`tools/witness_memory_preflight.project_peak_rss_gib`) = **67.61 GiB SAFE** at the 0.85/128 GiB ceiling;
micro-batch adds a projected-negligible transient. Memory is NOT the binding constraint — throughput is
(there is none to gain here).

## A/B spec for promotion (GO-gated; the honest gate)
Because the lever is trajectory-affecting (batched fp reduction; grad==serial only within ~1.8e-7 on
CPU, ~1e-3 on GPU per the module note) a promotion A/B is owed IF a throughput win ever appears. Arms:
serial `--micro-batch-pairs 1` vs batched `--micro-batch-pairs {2,4}` on the SAME seed, SAME argv, ~50 ep,
single-workload (no concurrent GPU). Pre-registered equivalence-of-outcome gate:
**(i) d_seg at a matched epoch within the run-to-run noise band; AND (ii) batched median t_step (over
>=10 clean epochs) < serial median t_step by >15%.** Given the n24 evidence, prediction: (ii) FAILS at
the live 384x512+wa config => micro-batch stays OFF (B=1). Re-open ONLY if a smaller-render regime or a
cleaner multi-epoch median shows a win. NOT a relaunch flag now.

## Legs routed vs still fail-closed (summary)
- ROUTED (fail-close removed, equivalence proven): witness-alone-island, focal, boundary-distance,
  eik-viscosity + eik-steik.
- STILL fail-closed vs micro-batch (out of scope): margin-saliency-reachability, seg-spike-reweight.

## Observability surface
Per-leg equivalence + tolerances: `src/tac/tests/test_levelset_micro_batch_loss.py` (47 tests) +
the measured table above (reproducible via the test harness helpers). n24 A/B run.logs (profile_timing
rows) under `experiments/results/mb_twin_ab_20260705/B{1,2,4,8}/run.log` (checkpoint bulk auto-cleaned;
logs kept). Diff-able across B; cite-able (run dirs + this memo + commit); counterfactual via the
`--micro-batch-pairs` toggle recorded in each run's safe_run header.

## 6-hook wire-in (Catalog #125)
sensitivity-map: N/A (compute lever, not a score axis) · Pareto: ACTIVE (feeds the launch runway —
micro-batch is NOT on the streamline; the n600 waterfill 67.6 GiB SAFE row is unchanged) ·
bit-allocator: N/A · cathedral autopilot: N/A (advisory timing) · continual-learning: this memo +
DAG FEED row + the CARGO-CULTED classification of "2-4x transfers" · probe-disambiguator: the n24 A/B
IS the disambiguator (resolved: no live-config speedup).
