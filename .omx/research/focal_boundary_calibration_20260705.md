# FOCAL-γ CALIBRATION + FOCAL/BOUNDARY-DISTANCE BUILD — RESULTS (council deliverables 1+2)

**Date:** 2026-07-05 · **Mandate:** `council_grand_symposium_levelset_loss_geometry_20260705.md`
(PROCEED_WITH_REVISIONS: measure before surgery; build default-OFF, READY, NOT deployed).
**Axis:** `[macOS-CPU/MLX advisory] NON-PROMOTABLE` — frozen MLX SegNet (parity-gated twin of the
CPU-torch authority), witness-alone deploy surface, 12-pair gt_n24 subset (stride-2), TWO paired
checkpoints (ep50 + ep75) of the LIVE nucleation-fix run `levelset_n600_witness_20260705T015247Z`
(read-only snapshots; the run was never touched). **Pointer 0.19110 UNMOVED — MEANS.**

## Headline (one paragraph)

**γ\* = 0 — HOLD. Focal-γ is measurably NOT an island lever on this vehicle at this stage, and the
symposium's premise (island-starved gradient) is FALSE on the nucleation-fix run.** Measured on the
real checkpoints: the islands (Lane+Movable) carry only **1.9% (ep50) → 1.1% (ep75)** of the d_seg
residual while receiving **3.1% → 3.5%** of the seg-loss gradient (over-allocated relative to
residual, Shannon equalization already satisfied-plus), and they are ABSORBING on time alone (Lane
within-flip **0.392 → 0.216** in 25 epochs; Movable ~solved at 0.9–1.3%). The residual is ~98%
BULK (Undrivable 62–65% + Road 33–36%, both mid-formation during a steeply-descending CE:
n600 verdict 0.162→0.122 per 25ep, |slope| 0.040 ≫ the 0.02 fire threshold). Focal-γ's measured
authority is WEAK and points at the wrong region: the island weight share is **NON-monotone in γ**
(peak ~γ=1, then FALLS below the pixel share by γ=3) because the hardest-p tail is bulk, not
islands — focal mildly feeds the bulk-BOUNDARY band instead (+0.5–1.2pp). The pre-registered fire
criterion is **doubly unmet** (slope steep AND islands ≪50% of residual). Both levers are BUILT,
tested, byte-identity-proven, ready — and should stay OFF.

## 1. The measured share table (deliverable 1a+1b)

Island = GT classes {1,3} (Lane, Movable; canonical comma10k order). Bulk-boundary = ≤2px from a
GT inter-class edge, non-island. Gradient = per-pixel |d seg_loss / d frame| mass through the real
R + frozen MLX SegNet on the witness-alone render; loss = the live config's CE +
`1+4·exp(−margin)` GT-margin weight; focal = stop-grad mean-1 `(1−p_y)^γ` (the build's exact
semantics). Every number below is from an executed computation (probe
`experiments/probe_focal_gamma_calibration.py`; merged JSONs in the probe sidecars).

**ep50 (12 pairs; subset d_seg 0.1412 vs logged n600 0.1217 — first-12-even-pairs sampling, same
drift class the 2026-07-04 disambiguator validated):**

| variant | island grad share | bulk-boundary | bulk-interior | island weight share (analytic) |
|---|---|---|---|---|
| current (γ=0) | **0.0312** | 0.0917 | 0.8770 | 0.0217 (= pixel share) |
| focal γ=0.5 | 0.0323 | 0.0950 | 0.8726 | 0.0312 |
| focal γ=1 | **0.0325** | 0.0975 | 0.8700 | **0.0323 (peak)** |
| focal γ=2 | 0.0317 | 0.1011 | 0.8672 | 0.0263 |
| focal γ=3 | 0.0305 | **0.1036** | 0.8659 | 0.0214 (≈ back to pixel share) |

**ep75 (same 12 pairs; subset d_seg 0.1466):**

| variant | island grad share | bulk-boundary | bulk-interior | island weight share (analytic) |
|---|---|---|---|---|
| current (γ=0) | **0.0354** | 0.0970 | 0.8677 | — |
| focal γ=0.5 | 0.0354 | 0.0979 | 0.8668 | 0.0335 |
| focal γ=1 | 0.0350 | 0.0984 | 0.8666 | 0.0329 |
| focal γ=2 | 0.0339 | 0.0998 | 0.8664 | 0.0222 |
| focal γ=3 | 0.0328 | 0.1014 | 0.8658 | **0.0151 (falling)** |

Reading: at ep75 focal is *monotonically anti-island* (the islands' p rose as they absorbed, so
(1−p)^γ down-weights them). The symposium's `share_isl(γ) monotone in γ` claim holds only when
islands are the uniformly-hardest tail — measured, they are not. Registered as the
`focal_gradient_concentration_v1` anchor (non-monotone, peak ~γ=1, |residual| 0.011 at γ=3).

## 2. d_seg decomposition + the paired ep50→ep75 slope (deliverable 1c)

| quantity | ep50 | ep75 | paired slope (same pairs, same harness) |
|---|---|---|---|
| Lane (1) within-flip | 0.392 | **0.216** | **absorbing fast (−45%/25ep)** |
| Movable (3) within-flip | 0.0086 | 0.0133 | ~solved (noise band) |
| island share of flips | 0.0186 | 0.0108 | falling |
| island within-flip | 0.121 | 0.073 | falling |
| Road (0) within-flip | 0.202 | 0.234 | mid-formation churn |
| Undrivable (2) within-flip | 0.186 | 0.184 | flat-high (the residual's core) |
| bulk-boundary within-flip | 0.422 | 0.503 | the 2px band is the hard band |
| share of d_seg: Undriv/Road/islands | 65.2 / 32.6 / 1.9 % | 62.1 / 36.4 / 1.1 % | bulk ≈ 98% |
| subset total d_seg | 0.1412 | 0.1466 | subset wiggle; n600 log: 0.162→0.122 (falling) |

Cross-run contrast (the nucleation-fix WORKS): the 2026-07-04 disambiguator measured the OLD fresh
run at ep75 with Lane and Movable at **100%** within-flip (71% of its plateau). This run at ep75:
Lane 21.6%, Movable 1.3%. The paint-seed + witness-alone-island-loss + seed-anneal absorption
pathway is delivering exactly what it was built for.

## 3. The loss-vs-basis verdict (NOT confounded — two timepoints measured)

The question as posed ("island gradient already substantial but d_seg[islands] not moving → basis;
island gradient starved → loss") has a measured third answer on this run: **island gradient is
substantial (3.1→3.5%, ≥ 1.6× the island residual share) AND d_seg[islands] IS moving (Lane flip
−45% per 25ep). Neither the loss nor the basis binds for the islands at this stage; TIME binds,
and it binds on the BULK (Road↔Undrivable mid-CE formation), which owns ~98% of the residual.**
The two-timepoint paired decomposition removes the `partially_confounded` escape — the verdict is
measured. (Caveat retained: the gradient surface is the witness-alone render; the seed-composed
surface is not reconstructable offline — no seed params in any checkpoint — and the wa island
share is an UPPER bound on the composed-surface share, which only strengthens "not starved" →
"not starved" is the conservative direction for the γ decision but note the composed-surface CE
gradient on islands is smaller; the ABSORPTION evidence (Lane flip falling) is surface-free.)

## 4. γ\* (Shannon equalization) and the fire decision

Equalization target: gradient share ∝ residual share. Measured (ep75): islands get 3.5% grad for
1.1% residual (OVER-allocated 3.2×); bulk-boundary gets 9.7% for 16.6% (under-allocated); interior
86.8% for 82.3%. Focal-γ cannot fix the one genuine under-allocation (bulk-boundary): its whole
measured authority is +0.4pp per γ-unit (9.7%→10.1% at γ=3 vs the 16.6% target), and it pays for
that with island share. **γ\* = 0 (HOLD).** If a focal arm is ever fired anyway, γ=1 is the
measured least-harmful value (island-share peak, mild boundary gain); γ≥2 is measured
anti-island. The pre-registered fire criterion (ep50→100 |Δd_seg| < 0.02/25ep AND islands >50% of
residual) is doubly unmet: slope 0.040 (steep), islands 1.1%. The calibration PROCEDURE is the
durable deliverable: re-run `probe_focal_gamma_calibration.py` on any later checkpoint (chunked
`--pair-list` invocations; ~35s/pair/5-variants, CPU, ≤5 GB RSS) and re-derive γ\* from the shares
— never guess it.

## 5. The build (deliverable 2 — READY, NOT deployed)

Commits `d3c4771ac` (probe) + `535e142be` (levers + tests). Live run untouched (no new flags in
its argv; its process runs pre-change code).

- **`--seg-focal-gamma` (default 0.0 = byte-identical):** stop-grad, mean-1-renormalized
  `(1−p_y)^γ` per-pixel reweight folded into the base loss's `seg_pixel_w` hook
  (`focal_pixel_weight_mlx`, base trainer) — applies to EVERY seg form on the SAME `seg_logits`
  surface the base form reads (the render_fn-composed frame, i.e. seed-composed when seeding;
  the #300 witness-alone island-lever routing is untouched). Composes multiplicatively with the
  spike-reweight. Rudin observability: per-epoch `{"stage":"focal","island_grad_share":...}` row
  (a TRUE measured grad share on one rotating pair, post-R surface — verified firing in the
  engaged smoke: 0.011/0.034 on n1).
- **`--boundary-distance-weight` (default 0.0 = byte-identical):** SDF-native Kervadec boundary
  placement — per-pair GT inter-class-edge distance transform (`boundary_distance_band_map`,
  2px linear band = the measured 1–2px flip band, computed ONCE per pair, ~472 MB at n600 when
  ON), loss = band-weighted mean |φ_GT − φ_runner| on `model.sdf(cf, c1)` (frame1; the contour
  DOF the witness owns; one extra trunk forward per pair when ON).
- **Fail-closed:** both levers raise with `--micro-batch-pairs>1` (the batched twin would
  silently drop them). Both registered in the resume lever-drift guard
  (`_resume_lever_divergences`) + resume sidecar cfg.
- **Byte-identity PROVEN (not asserted):** golden test on real gt_n6 + real frozen MLX SegNet:
  loss AND every grad leaf bitwise identical across {no-kwarg current path, `focal_gamma=0.0`,
  a re-implementation of the pre-change CE expression}; γ=2 changes loss/grads; `seg_pixel_w=ones`
  composes bitwise. Structural: both branches are `if flag > 0.0:` python gates — no mx ops at 0.
- **Tests:** `experiments/test_focal_boundary_levers.py` — 10/10 green (5^γ ratio law exact;
  mean-1 + stop-grad; monotone concentration on a synthetic hardest-tail case; band-map exact
  geometry; boundary term monotone-in-offset + differentiable; drift guard; argparse; the slow
  real-scorer byte-identity). Siblings green: seed-absorption 9/9 (incl. slow), cograd fast 6/6.
  ruff: 0 new (75→75 pre-existing). Engaged-path smoke (n1, 2ep, CPU, both flags ON): completes,
  `boundary_distance` + `focal` rows fire, final checkpoint written.
- **Deployment caveats (parent-owned):** (a) mid-run engagement via `--resume-from` requires
  `--resume-allow-lever-drift` and adds a loss jump the spike-guard may skip-step on — prefer a
  fresh arm or accept the re-treat; (b) Tishby window: engage at CE (≤ep300) or wait for run 3;
  (c) `TAC_MLX_CUSTOM_GROUPED_BACKWARD=0` required for any `--mlx-device cpu` invocation
  (pre-existing Metal-kernel constraint, hit + worked around in the smoke).

## 6. Recommended arm flags

**Recommendation: NO ARM — HOLD (γ\*=0; fire criterion doubly unmet; islands absorbing on time).**
If the parent overrides and fires anyway at the ep100 checkpoint: `--seg-focal-gamma 1.0`
(measured least-harmful; expect near-inert island effect) and/or `--boundary-distance-weight 0.05`
(UNMEASURED effect — a genuine A/B arm at the eikonal-weight scale, engage at CE; the boundary
band is where the one real under-allocation lives, and this lever targets it geometrically rather
than by p-hardness). Any arm re-runs the calibration probe on its checkpoint first.

## 7. Triality propagation

- **equations:** `focal_gradient_concentration_v1` registered
  (`tools/register_lever_laws_20260705.py`; module
  `tac/canonical_equations/focal_gradient_concentration_20260705.py`) with the MEASURED
  non-monotone anchor. `levelset_energy_loss_equivalence_v1` NOT registered (design-stage, run-3
  item per symposium resolution 3 — no measured anchor; FORMALIZATION deferred with that design).
- **DSL:** SegLossGauge NOT added this pass — the calibrated verdict is HOLD/γ\*=0, so there is no
  live chart to encode beyond BASELINE; the flags are grep-verified in the trainer argparse and
  the gauge addition belongs with an actual firing decision (avoids a DSL chart that no run uses).
  Named follow-up if fired: add `SegLossGauge{BASELINE, FOCAL(γ=1), BOUNDARY_DIST(0.05)}`.
- **DAG:** FEED-05e appended (`sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`).

## 8. Self-reflection (Catalog #363)

- Share tables + decomposition + slope: `VERIFIED_VIA_EMPIRICAL_ANCHOR` (executed probe, two
  checkpoints, merged from per-chunk raw masses; kill-durable sidecars).
- "wa share is an upper bound on composed-surface share": `VERIFIED_VIA_SOURCE_INSPECTION`
  (disambiguator mechanism + compose code) — the seed satisfies island loss on the composed frame.
- n600 ep75 verdict: pending async at memo time — the fire-criterion slope uses the logged
  ep25→ep50 n600 rows (0.162→0.122); subset totals are labeled advisory.
- Probe-harness validation: subset-vs-logged drift (0.141 vs 0.122) matches the disambiguator's
  validated drift class (0.0307 vs 0.0289) in sign/magnitude; per-class STRUCTURE is the signal.
- Adversarial self-challenge: the ep75 second timepoint was taken specifically because it could
  have falsified the "absorbing on time" reading (islands stalling would have flipped the verdict
  to basis-binds). It did not — Lane flip fell 45%.

**Pointer 0.19110 UNMOVED.** All numbers advisory calibration; the only authority is a byte-closed
`upstream/evaluate.py` row.
