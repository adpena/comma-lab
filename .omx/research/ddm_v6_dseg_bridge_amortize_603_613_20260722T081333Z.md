---
schema: ddm_v6_dseg_bridge_amortize_landing.v1
task: 603
feeds_task: 613
master_task: 578
lane_id: lane_ddm_v5_route_fix_compose_603_613_20260722
evidence_axis: "[macOS-CPU frozen-SegNet advisory]"
score_claim: false
candidate_archive: false
producer_commit: 0e7be4bbb6a73957aa80784681f52e697b6803c3
main_landing_review_required: true
---

# DDM v6 evaluator bridge and temporal amortization

## Outcome

**MEASURED:** the v5 composed member is not close to the evaluator knee.  The exact n64 control is
119,335 B at advisory `d_seg=0.044353087743`; the exact n256 control is 324,260 B at
`d_seg=0.038300534089`.  Both miss Task #613's `0.00116` bar and the S4 `0.016 @ 216,207 B` knee.
Official CPU-Torch PoseNet with its two-frame YUV6 preprocessing was feasible and returned
`d_pose=159.108007460341` (n64) and `157.800897199751` (n256).  Pose6 payload completeness remains
1.0, but payload completeness is not confused with evaluator `d_pose`.

**MEASURED rate mechanism / negative efficacy:** unit-root AR(1) hold (`x_t=x_{t-1}` between
24-pair key refreshes) and counted-Pose6/xi-adaptive keying reduce the n64-to-n256 marginal from
`1067.317708 B/pair` to `107.958333` and `107.937500 B/pair`.  The requested `<=300 B/pair` test is
green.  The same formulations remain red on absolute `d_seg`; this is a representation-rate success,
not a promotion or score success.

## Receiver-closed points

All rows are exact final composed ZIP lengths.  `membership` is MEASURED only for the unchanged v5
control.  Amortized rows show a DERIVED interval from the settled C1-to-GT match
`0.999873638153`: `1-d_seg-epsilon <= membership <= 1-d_seg+epsilon`,
`epsilon=0.000126361847`.

| mode | P | bytes | membership | advisory d_seg | advisory d_pose |
|---|---:|---:|---:|---:|---:|
| v5 exact | 64 | 119,335 | **0.955627997716 MEASURED** | 0.044353087743 | 159.108007460341 |
| fixed AR(1)-hold24 | 64 | 50,155 | [0.954587141673, 0.954839865367] DERIVED | 0.045286496480 | 159.104827981350 |
| xi/Pose6 AR(1)-hold24 | 64 | 50,192 | [0.954782724380, 0.955035448074] DERIVED | 0.045090913773 | 159.094020290442 |
| residual-zero/static-once | 64 | 50,661 | [0.955247163773, 0.955499887467] DERIVED | 0.044626474380 | 159.068233847613 |
| v5 exact | 256 | 324,260 | **0.961685518424 MEASURED** | 0.038300534089 | 157.800897199751 |
| fixed AR(1)-hold24 | 256 | 70,883 | [0.959704418977, 0.959957142671] DERIVED | 0.040169219176 | 157.798907948748 |
| xi/Pose6 AR(1)-hold24 | 256 | 70,916 | [0.960068106651, 0.960320830345] DERIVED | 0.039805531502 | 157.796182090838 |
| residual-zero/static-once | 256 | 71,828 | [0.958458364010, 0.958711087704] DERIVED | 0.041415274143 | 157.683205618006 |

No membership interval is silently promoted to a measurement.  Full ordered per-pair rows and all
target-class/topology/margin strata live in the SHA-bound candidate checkpoints referenced by the
cross-window receipt.

## Decisive SegNet strata

| row | Road | Lane | Undrivable | Movable | MyCar | boundary codim-1 | interior |
|---|---:|---:|---:|---:|---:|---:|---:|
| v5 exact n64 | .091077797700 | .417607251976 | .002855959553 | .999747692002 | .000103722180 | .486618757886 | .033771179003 |
| v5 exact n256 | .076487078184 | .366618614685 | .004886740363 | .987887081494 | .001106591890 | .469455175602 | .028580256805 |
| xi hold24 n256 | .080395329615 | .440499380501 | .004843907703 | .999742410076 | .001114585118 | .476654438103 | .029956878385 |

The failure localizes to Movable, Lane, Road, and codimension-1 boundaries.  MyCar and Undrivable
are already low-error.  The margin decomposition agrees: for exact n256, errors are
`.550133388884`, `.500395372066`, `.411799036795`, `.026158371608` over margin bands
`[0,.1)`, `[.1,.5)`, `[.5,1)`, `[1,inf)`.  This supports Fisher/margin-ranked follow-on allocation;
it does not license blanket spending.

Per-pair `d_seg` distributions:

| row | min | q25 | median | q75 | max |
|---|---:|---:|---:|---:|---:|
| v5 exact n64 | .023605346680 | .033093770345 | .039454142253 | .055141448974 | .073501586914 |
| v5 exact n256 | .020660400391 | .028799692790 | .035382588704 | .043262481689 | .073501586914 |
| xi hold24 n256 | .021728515625 | .030618031820 | .036145528157 | .045017242432 | .073282877604 |

## Marginal bytes

| formulation | n64 | n256 | delta / 192 added pairs | <=300 B/pair |
|---|---:|---:|---:|---:|
| v5 exact | 119,335 | 324,260 | 1067.317708 | no |
| fixed AR(1)-hold24 | 50,155 | 70,883 | **107.958333** | yes |
| xi/Pose6 AR(1)-hold24 | 50,192 | 70,916 | **107.937500** | yes |
| residual-zero/static-once | 50,661 | 71,828 | **110.244792** | yes |

### Chart stream marginal homes

| stream | v5 exact B/pair | fixed hold24 B/pair |
|---|---:|---:|
| global anchors | 1.244792 | 0.281250 |
| axial gradients | 3.744792 | 0.583333 |
| low residual | 180.583333 | 10.968750 |
| mid residual | 235.854167 | 13.369792 |
| high residual | 248.479167 | 13.453125 |
| Pose6 exact stream | 5.750000 | 5.750000 |

### Outer structured-member marginal homes

| home | v5 exact B/pair | fixed hold24 B/pair |
|---|---:|---:|
| chart.zip | 675.656250 | 44.406250 |
| Undrivable events / components | 82.229167 / 27.885417 | 16.885417 / 4.234375 |
| Road events / components | 149.609375 / 89.687500 | 20.968750 / 12.015625 |
| Lane events / components | 37.505208 / 4.713542 | 7.442708 / 1.979167 |
| Road PXQ1 / Lane LBND2 / MyCar hood static | 0 / 0 / 0 | 0 / 0 / 0 |

The settled static surfaces were already static-once.  The measured gain comes from amortizing chart
residuals and dynamic S4 event/component streams, not from double-spending static canonicalization.
The xi schedules were derived from counted Pose6 L1 motion: n64 keys `[0,21,42]`; n256 keys
`[0,23,47,70,89,110,131,155,178,202,226,248]`.

## Round-1 adversarial review

Finding: the first implementation emitted decisive `d_seg`/`d_pose` but omitted new same-C1
membership values.  Fix: v5 controls keep the prior MEASURED membership; amortized candidates receive
only the rigorous triangle-inequality interval, explicitly labeled DERIVED.  Verification after the
fix: Ruff clean; focused suite **39 passed**.  No score weights are present in any archive; sources on
SSD stayed read-only; every candidate archive and measurement receipt is atomically preserved.

## Blocker delta and next routing

- Canonical #603 register count stays **8/19 on this branch**; the appended v6 row is a MAIN-review
  draft.  MAIN decides the count transition when registering the evaluator-bridge closure.
- GREEN: actual v5 advisory `d_seg`; official-YUV6 advisory `d_pose`; per-pair and per-stratum tables.
- GREEN: temporal marginal test, from 1067.32 to 107.94-110.24 B/pair.
- RED, formulation-scoped: all measured rows miss `d_seg<=0.016`, therefore also `<=0.00116`.
- RED: n600 evaluator bridge not run within the bounded arm; contest CPU/CUDA not authorized.
- Next highest-EV action: spend only on measured Movable/Lane/Road boundary error under the
  Fisher/margin ranker, with the now-green amortized representation as the rate carrier.

## Bounded re-derivation argv

Each command completed under ten minutes on this host and resumes from preserved candidate stages:

```text
python3 tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v6_dseg_bridge_amortize_n64_603_613_20260722T075903Z.config.json --output-dir .omx/research/ddm_v6_dseg_bridge_amortize_n64_603_613_20260722T075903Z --execution-allowed false
python3 tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v6_dseg_bridge_amortize_n256_603_613_20260722T075903Z.config.json --output-dir .omx/research/ddm_v6_dseg_bridge_amortize_n256_603_613_20260722T075903Z --execution-allowed false
```

## STORES CONSULTED

- `docs/operating_manual_craft_handoff.md`
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`
- v5 SHA-bound n64/n256 receipts and exact composed archives
- settled `gt_n600` `lstars`, `margins`, and `gt_poses`
- canonical S4 archive/runtime, read-only
- 2026-07-19 reverse-waterfill, Fisher/margin, corrected-Jacobian, curvelet, and xi directives

Pointer honesty: **0.1910828242 [contest-CPU] — unchanged.**
