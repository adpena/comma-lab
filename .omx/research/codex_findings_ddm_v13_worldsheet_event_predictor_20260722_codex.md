---
schema: codex_findings_ddm_v13_worldsheet_event_predictor.v1
date_utc: 2026-07-22
lane_id: ddm_v13_worldsheet_event_predictor
axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
execution_allowed: false
score_claim: false
d_seg_claim: false
main_landing_review_required: true
---

# DDM v13 worldsheet/event predictor — outcome first

The measured round-2 V13 baseline materially improves the bound n600 predictor but fails its
preregistered INSTANCE falsifier. Operator addenda received at 19:16--19:26 UTC supersede the
measured Lane grammar and extend the Movable successor; therefore this row is a SHA-bound baseline,
not evidence against those successor formulations.
The selected `islands` rung is 132,606 B with `d_seg=0.029592759874`, official frozen-Pose
`d_pose=163.016398660918`, and Movable-conditional `d_seg=0.481331895297`. The Movable half of
the falsifier clears (`<=0.5`), but total `d_seg` remains above `0.01`; binding mechanism is
`receiver_projection`. Verdict:
`ADVISORY_V13_INSTANCE_FALSIFIER_TRIGGERED_FORMULATION_ONLY`.

No R6 exact-eval request is authorized. Pointer `0.1910828242 [contest-CPU]` is unchanged.

The later G2 correction was also consumed with a separate receiver-closed phase-only ablation.
On n600, 130 raw q8 phase symbols add 2,128 B and improve Lane-conditional d_seg by
`-0.029228004790`, but worsen total d_seg by `+0.000280736287` and advisory action by
`+0.028907928562`. The raw-symbol instance is rejected for total-objective harm. The Lane sign
is nevertheless a measured mechanism signal; the distinct anisotropic AR(1)-whitened BEV
successor remains unmeasured and open.

## Measured receiver-closed ladder

| rung | archive B | d_seg | d_pose | Movable | Lane | objective | disposition |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 102,105 | 0.034502249824 | 163.039648911962 | 0.988264941023 | 0.424611121005 | 43.896380982393 | inherited control |
| islands | 132,606 | 0.029592759874 | 163.016398660918 | 0.481331895297 | 0.434971091989 | 43.422862186555 | selected; pays |
| lane | 104,999 | 0.038188722399 | 163.010374426555 | 0.995240100663 | 0.613172728444 | 44.263330034174 | reject |
| both | 135,004 | 0.034225607978 | 162.963739223740 | 0.540649512951 | 0.620308149409 | 43.881221954195 | pays weakly vs base, dominated by islands |

All rows are MEASURED advisory values through the receiver, uint8/resize path, frozen SegNet,
and official frozen PoseNet. They are not contest-CPU/CUDA scores.

Two-part-code decisions relative to base:

- G1 islands: +30,501 archive B, delta-S `-0.473518795838`,
  `-1.552469741445e-05` per added byte; admit.
- Lane: +2,894 archive B, delta-S `+0.366949051781`; reject.
- Both: +32,899 archive B, delta-S `-0.015159028198`; positive but dominated.

## G1 adoption and the blocker delta

The operator's late G1 table was consumed before round 2. The productionized v13 encoder exactly
reproduces G1's selected Movable derivation: 29,810 B,
SHA-256 `1066081727229e605462e67b8fdd26937d5e3552c13cb66a7444ea3b7360366f`, 180 births,
2,017 persists, 158 deaths, ten slots, and 19,150 vertices. Its semantic mask parse-back has
33,378 errors, DERIVED clean-rest `d_seg=0.00028294881184895833`.

The exact mask grammar is therefore not the active Movable representation blocker. Painting that
decoded worldsheet through the inherited receiver improves Movable from 0.9883 to 0.4813, but
does not realize the mask-level bound. Cross-stratum/RGB/resize projection is the owed mechanism.
The next build should optimize receiver-visible class birth/paint under Road/Lane ordering, not
invent another Movable atom vocabulary.

Lane is an independent negative on this pre-addendum formulation. One xi-driven dash phase, polynomial drift
knots, width profile, and visibility add 2,894 archive B and worsen Lane from 0.4246 to 0.6132.
Counted derivation fields decompose as dash phase 284 B, geometry 1,040 B, thin-structure/visibility
308 B, shared packet address/header 420 B; the remaining 842 B is outer manifest/ZIP home cost.

## G2 phase-symbol falsification

The G2 ledger's four corrections are binding: spend is ranked by frozen-scorer flip/margin rather
than pixel energy; Lane is birth-dominated and therefore keeps dash-comb plus event structure;
Movable is persistence-dominated and requires relative velocity plus deviations rather than
xi-only transport; and phase/skip claims had no authority before this receiver test.

| window | phase symbols | added B | delta d_seg | delta Lane d_seg | delta action | disposition |
|---|---:|---:|---:|---:|---:|---|
| n64 | 23 | 521 | -0.000217199325 | -0.036247253996 | -0.021258003342 | sanity sign helps |
| n600 | 130 | 2,128 | +0.000280736287 | -0.029228004790 | +0.028907928562 | reject raw q8 instance |

Both rows compare the same Lane production archive with and without phase-only knots. The added
knots carry no centerline geometry or width fields. The n600 split verdict means raw phase symbols
repair Lane pixels but spill enough error outside Lane to lose globally; it neither validates nor
refutes the later whitened-process formulation.

## G3 broad-debt allocation correction

G3 merge `e472310d93` landed before commit and invalidates any per-pair top-k successor. Measured
joint score mass is broad: top 10 carry 1.9785%, top 100 carry 18.7039%, and Seg top 100 carry
26.8544%. V13 allocation is therefore constrained to shared grammar/templates/process priors paid
once across n600. Per-pair streams may only be near-uniform-cheap innovations; top24 (`r=0.5953`)
is a screening surface, never verdict authority. Every ladder verdict remains full n600.

The current counted G1 EVENT worldsheet and six Lane programs have representation support at all
three G3 proxies: pair 279 has 819 Movable pixels, pair 286 has 1,014, and pair 452 has 10,404;
all have six active Lane programs. This is support coverage, not semantic ground truth or measured
event benefit. Receipt: `ddm_v13_g3_allocation_consumption_20260722.json`.

## Implementation and custody

- `direct_description_g1_worldsheet.py` is the window-generic exact G1 EVENT/CENTROID/SHAPE
  encoder/decoder with real Brotli/LZMA/zlib selection and semantic parse-back.
- V13 archives carry the counted `.g1s` derivation as its own unique byte home. They reject mixing
  it with the superseded moment-ellipse worldsheet.
- Lane stays one phase per coherent object, never one event per dash, and is projected against the
  inherited Road support.
- The runner preserves extraction, exact G1 payload, composition, and each measured rung as atomic
  checkpoints; each invocation measures at most one missing rung.
- n64 sanity and n600 primary use batch16 release. No scorer weights, GT table, pixel/RGB patch,
  paid dispatch, Modal call, or candidate archive is present.
- The append-only `ddm_describe_line_rate_distortion_bracket_v1` registry row now has a fifth,
  SHA-bound V13 empirical anchor.

## Round-1 adversarial findings

1. The initial local moment/ellipse worldsheet did not satisfy the late G1 adopt-as-is directive.
   It was superseded before round 2 by an exact-byte G1 polygon production.
2. Accelerate emitted spurious warnings for tiny 2-D matmuls in the superseded extractor; explicit
   scalar projection removed the warning class.
3. The first derived receipt mislabeled the binder as `track_fidelity`; re-derivation showed the
   exact G1 mask is already below the box while through-R is not. V2 receipts correct this to
   `receiver_projection`; V1 receipts remain historical and are not authoritative.
4. Lane is net harmful at n64 and n600. It remains an INSTANCE-scoped negative, not a family death.
5. The 19:16--19:26 operator addenda arrived after this grammar was formed. The old Lane rung is
   now explicitly `measured_pre_20260722T1916Z_operator_addenda_baseline_only`; it cannot be
   promoted as the required BEV-curvature, range-gated dash-comb, anisotropic-volatility AR(1)
   innovation process. G1 remains exact evidence, but projective depth normalization, shared shape
   templates, aspect/rotation morph, and sparse residual events are an unmeasured successor.
6. The 20:15 G2 ledger invalidated pixel-energy ranking and left the phase hypothesis untested.
   A separate batch16 receiver-closed ablation measured the split result above: Lane helps, total
   d_seg harms. The raw q8 instance is rejected; the scoped successor stays open.
7. The 21:13 G3 atlas falsified heavy-tail allocation. The canonical law now refuses per-pair
   top-k as a closing form, limits top24 to screening, and requires full-n600 verdicts plus
   amortized shared structure.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`
- v7.5 operating contract and v8 per-class decomposition spec
- V6, V9-V12 code, receipts, findings, DAGs, and equation notes
- G1 merge `fbc24fb5ab`, compact table, memo, and receipt SHA
  `aeeb916f973523d5ffa3389ee8d744901fe9477cc149af7e756726e2ead907f6`
- frozen n600 target cache SHA
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- `reports/latest.md`, lane registry, canonical task/subagent state, per-arm inbox, broadcast ledger
- operator 2026-07-19 EV/Fisher/inner-Jacobian/curvelet/xi directives and 2026-07-22 grammar supplement
- operator 2026-07-22 19:16--19:26 Lane jitter, anisotropic projective stretch, projective
  Movable lifecycle, and openpilot road-polytope/far-near/angular-chart addenda
- G2 ledger merge `b8f3833d09` and the 2026-07-22 20:15 metric/topology/transport/phase correction
- G3 atlas merge `e472310d93`, compact receipt SHA
  `6c4157092a7bdf7ba44b458cd470725cc470d84a8fc77ed7d3dedb59160734f5`, and hard-pair registry
  SHA `0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4`
- existing `analytic_lane_render_band.py`, `dash_phase_carrier.py`,
  `lane_ground_factorization.py`, and `lane_track_and_smooth.py` reuse surfaces

MAIN landing review must verify the exact G1 SHA match, receiver paint/overwrite order, V2 receipt
bindings, phase receipt SHAs, and that neither the pre-addendum Lane baseline nor the raw q8 phase
negative is mistaken for the required successor or a family verdict. No contest score or R6
request may be inferred.
