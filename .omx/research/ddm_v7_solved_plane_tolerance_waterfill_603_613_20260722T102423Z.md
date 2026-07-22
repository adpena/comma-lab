---
schema: ddm_v7_solved_plane_tolerance_waterfill_landing.v1
task: 603
feeds_task: 613
master_task: 578
lane_id: ddm_v7_solved_plane_tolerance_waterfill
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
research_only: true
score_claim: false
d_seg_claim: false
d_pose_claim: false
candidate_archive: false
producer_commit: e16e25b025324c475f1b6152049f55366350160f
main_landing_review_required: true
---

# DDM v7 solved-plane tolerance waterfill

## Outcome

**MEASURED, FORMULATION-scoped:** losslessly describing the solved C1 planes restores both advisory
scorer legs, but the opaque site/value correction formulation is roughly three orders of magnitude
outside its preregistered 200 KB box. The only rows satisfying `d_seg <= 0.00116` are `exact_all`:

| window | cheapest feasible policy | exact bytes | multiple of 200 KB | advisory d_seg | advisory d_pose |
|---|---|---:|---:|---:|---:|
| n64 `[448,512)` | exact_all | 43,112,153 | 215.560765x | 0.000171422958 | 0.000081666650 |
| n256 `[344,600)` | exact_all | 171,332,654 | 856.663270x | 0.000154534976 | 0.000104117518 |

Both exact receivers reproduce their solved-plane windows bit-for-bit. This is a decisive rate wall
for opaque exact site/value corrections over the v6 predictor, not a negative verdict on learned,
analytic, curvelet/shearlet, or evaluator-inverse descriptions of the same solved planes.

## Receiver-closed tolerance ladder

All rows below are **MEASURED** on `[macOS-CPU frozen-scorer advisory]`; exact final ZIP lengths are
counted, and `score_claim=false`, `d_seg_claim=false`, `d_pose_claim=false` throughout.

| policy | n64 bytes | n64 d_seg | n64 d_pose | n256 bytes | n256 d_seg | n256 d_pose |
|---|---:|---:|---:|---:|---:|---:|
| exact_all | 43,112,153 | 0.000171422958 | 0.000081666650 | 171,332,654 | 0.000154534976 | 0.000104117518 |
| q4_all | 27,479,944 | 0.001374244690 | 0.002485906775 | 108,637,789 | 0.001218597094 | 0.004350660189 |
| waterfill_sensitive_exact | 4,179,132 | 0.012644211451 | 8.479886633718 | 15,571,267 | 0.010529518127 | 7.994121936780 |
| waterfill_balanced | 3,534,452 | 0.012602090836 | 8.451518574362 | 13,250,967 | 0.010495305061 | 7.990826227167 |
| q16_all | 10,761,809 | 0.005078713099 | 0.060651052613 | 42,253,555 | 0.004773457845 | 0.093786208900 |
| q64_all | 1,608,262 | 0.033052523931 | 8.929935045545 | 6,264,275 | 0.025728523731 | 10.969005841337 |
| fixed_hold24_all | 2,239,211 | 0.027928272883 | 0.121842519507 | 7,796,629 | 0.026270667712 | 0.229579410911 |
| xi_hold24_all | 2,218,850 | 0.028261661530 | 0.097639616130 | 8,501,332 | 0.025154093901 | 0.198460411219 |
| drop_to_predictor_all | 52,697 | 0.045286496480 | 159.104827981350 | 73,426 | 0.040169219176 | 157.798907948748 |

The q4 rows nearly reach the d_seg bar but remain enormous: n256 misses by `0.000058597094` at
108.6 MB. The mixed policies are not useful Pareto points. Dropping the nominally tolerant
Undrivable/MyCar corrections exposes large image and Pose debt through the inherited predictor;
the v6 coarse-paint sensitivity ranking does not transfer into a license to omit their solved-plane
signal.

## Per-stratum decomposition

The exact-rung ZIP homes identify the rate crux:

| stratum | n64 bytes | n256 bytes | n64 exact d_seg | n256 exact d_seg |
|---|---:|---:|---:|---:|
| Road | 8,606,120 | 34,212,776 | 0.000313013968 | 0.000279025110 |
| Lane | 94,284 | 392,068 | 0.004153725728 | 0.003706867157 |
| Undrivable | 21,551,775 | 86,082,475 | 0.000079782654 | 0.000064048086 |
| Movable | 985,456 | 3,409,572 | 0.000744515403 | 0.000866889168 |
| MyCar | 10,459,480 | 42,001,617 | 0.000076691552 | 0.000090511552 |
| Boundary | 1,362,996 | 5,161,372 | 0.007240733120 | 0.006924412700 |

Undrivable, MyCar, and Road dominate exact bytes even though Lane and the boundary band dominate
the residual advisory error. This separates the rate binder from the d_seg binder: reducing Lane
and boundary error is necessary for the evaluator bar, while replacing opaque region-wide exact
values is necessary for rate.

The two SHA-bound window receipts contain the complete six-stream byte table and advisory d_seg by
all five target classes, boundary/interior topology, and Fisher/margin band for **every** policy—not
only the exact rows summarized above:

- n64 receipt SHA-256 `8db93c4ef90e6d7f29943b4334a0d441f5cfbe226bf68fceb7ed03e59730970b`
- n256 receipt SHA-256 `d68f1d9ead9401173160b8cc4ec7fb9d49753a6bb0f298af23de293bc28d4274`
- cross-window receipt SHA-256 `64658a05a8975707f98db308223cefff78b5352975bb59cc2aa8a4ff2f8d50fb`

## Discrete reverse-waterfill

The route is derived from measurements: sort by exact archive bytes, retain only strict distortion
record improvements for `D=100*d_seg+sqrt(10*d_pose)`, then compare each positive-byte marginal to
`25/37,545,489 = 0.000000665858953` score units per byte.

- n64 Pareto route: predictor -> q64 -> xi_hold24 -> q16 -> q4 -> exact. First rate break:
  `xi_hold24_all -> q16_all`, measured marginal `0.000000295873431` per byte.
- n256 Pareto route: predictor -> q64 -> fixed_hold24 -> xi_hold24 -> q16 -> q4 -> exact. First rate
  break: `fixed_hold24_all -> xi_hold24_all`, measured marginal `0.000000309471389` per byte.

The rate-optimal stopping points do not satisfy the d_seg constraint. Conversely, the feasible knee
is far beyond rate break-even. There is no admissible opaque-correction point near 200 KB.

## Receiver, custody, and resumability

- Six opaque correction sections are counted in the archive. Stratum semantics and any ground-truth
  argmax table remain external; no SegNet/PoseNet weights are shipped.
- Brotli-Q11 and LZMA-XZ preset-9-extreme compete per section; exact parseback determines admission.
- Boundary precedence is explicit; the five roles are self-detected from the inherited route receipt
  and never luma-sorted.
- Compiler x2, parse/re-encode identity, deterministic receiver replay, unique ZIP byte homes, and
  exact target reconstruction pass.
- Seven rung checkpoints and nine candidate checkpoints are atomically preserved per window.
- Local bulk totals 183,020,989 B (n64) and 719,471,409 B (n256). The delegated boundary makes the
  SSD read-only, so certify-or-block keeps these bytes locally; no deletion or cold-store move was
  authorized. The cross-window receipt records counts, sizes, rebuild authority, and blocker reason.
- Fresh builds measured 727.27 seconds (n64) and 2,831.91 seconds (n256). Rebuilding only the final
  receipt from preserved stages measured 7.85 and 25.62 seconds. Sealed-receipt validation now hashes
  every rung frame, candidate checkpoint, and candidate archive in 0.81 and 1.05 seconds.
- n600 was not run: the required n256 scale check already took 47 minutes and falsified the rate
  formulation by 856.7x. This is a time-bound omission, not an inferred n600 result.

## Round-1 adversarial review

Disposition: `PASS_AFTER_THREE_REPORT_AND_RESUME_FIXES`.

1. A handwritten waterfill order produced a negative added-byte step. Fixed by deriving the strict
   discrete Pareto envelope from measurements; regression rejects dominated/non-monotone routes.
2. One candidate scope inherited the older v6 SegNet-only axis. Fixed with a single v7 joint-scorer
   scope constructor and checkpoint evidence-axis validation.
3. Rebuilding a final receipt changed live free-space telemetry and refused deterministic replay.
   Fixed by validating and returning the sealed final receipt against committed producer custody,
   candidate-table hash, every candidate archive/checkpoint, and every rung frame.

The invalid round-0 n64 report is preserved, SHA-256
`217bd957e2d1e24ab623f6bfc8464e4c135fadc3f07fb0f86070448d77511062`; it is not an evidence
source. Ruff, pycompile, and the focused suite are green: **45 passed**.

## Blocker delta and next routing

- GREEN, local advisory: exact solved-plane receiver, per-stratum ladder, joint Seg/Pose bridge,
  exact archive bytes, and bounded disk resume.
- GREEN only at infeasible rate: `d_seg <= 0.00116` on both exact rows.
- RED, formulation-scoped: exact opaque residual `<=200 KB`; 43.1 MB at n64 and 171.3 MB at n256.
- RED: n600, contest-CPU/CUDA, candidate archive, score claim, and frontier promotion.
- Canonical #603 register remains `8/19` on this branch. MAIN reviews the append-only draft row.
- Highest-EV successor: replace region-wide opaque site/value payloads with a structured analytic or
  learned solved-plane description, preserving exact/fine Lane and boundary obligations while
  factoring bulk Road/Undrivable/MyCar interiors and Pose via the registered Fisher/inner-Jacobian,
  curvelet/shearlet, and xi laws. The failed opaque formulation must not be retried at n600.

## Bounded re-derivation argv

Both same-output invocations validate the sealed receipts in under ten minutes on this host:

```text
.venv/bin/python tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v7_solved_plane_tolerance_waterfill_n64_603_613_20260722T085852Z.config.json --output-dir .omx/research/ddm_v7_solved_plane_tolerance_waterfill_n64_603_613_20260722T085852Z --execution-allowed false
.venv/bin/python tools/run_direct_description_entropy_priced_member.py --config .omx/research/ddm_v7_solved_plane_tolerance_waterfill_n256_603_613_20260722T085852Z.config.json --output-dir .omx/research/ddm_v7_solved_plane_tolerance_waterfill_n256_603_613_20260722T085852Z --execution-allowed false
```

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`
- target receipt `a8d94f0f8338036fb3224a92078eff1f1fb5fd2eb598ed994a1f965b6561efb2`
- SHA-bound v5/v6 receipts, exact receiver archives, and coder survey
- 2026-07-19 reverse-waterfill, Fisher/margin, inner-Jacobian, curvelet/shearlet, and xi directives

Pointer honesty: **0.1910828242 [contest-CPU] — unchanged.**
