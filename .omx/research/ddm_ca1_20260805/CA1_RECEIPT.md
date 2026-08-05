---
schema: ddm_ca1_cap_artifact_wall_sweep.v1
date_utc: 2026-08-05
arm: ddm_ca1
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
axis: "[scorer-free static/source/receipt audit]"
tokens: [no-triality, p0-ledger-ok]
---

# CA1 - cap-artifact wall sweep

## Answer First

CA1 landed the #874 instrumentation cure and refreshed the live source census. It did
not move a score row, did not run scorer forwards, did not launch, and did not run
`upstream/evaluate.py`.

Live-source denominator after excluding archived `experiments/results` bundles:

| field | value |
|---|---:|
| Python files scanned | 6,209 |
| files matched with cap defaults | 84 |
| cap-default sites | 89 |
| scanner-clean stop-reason sites | 6 |
| silent cap-default sites | 83 |
| parse errors | 0 |

CA1 classification over the 89 sites:

| class | count | meaning |
|---|---:|---|
| A_reports_stop_reason | 6 | scanner found existing stop-reason vocabulary |
| A_local_convergence_flag_scanner_warns_until_migrated | 1 | OS1 body found a real local convergence flag, but the generic scanner still warns until migrated |
| B_live_or_reopen_risk | 6 | current or plausible reopen path where a cap can affect a load-bearing decision |
| C_dormant_or_hygiene_backlog | 76 | no current load-bearing capped verdict found in the bounded recall scope |

No additional scorer-free solver rerun was fired in CA1. The ranked B rows either need
frozen scorer/PoseNet/SegNet forwards, are owned by an active lane (`q3x`/`od9` ordering),
or are stale/off-chain. Firing them here would have violated the common contract's
scorer-free and single-writer boundaries.

What changed in code:

- Added `CapStopReceipt` / `build_cap_stop_receipt` in `tac.optimization.trajectory_stopping`
  with the required payload shape:
  `{stop_reason: converged|cap_bound|failed, steps_run, cap, still_descending}`.
- Taught `tools/check_no_silent_cap_defaults.py` that `CapStopReceipt` is a reporting
  surface and removed archived `experiments/results` source bundles from the live denominator.
- Added a warn-only `tac.preflight.check_no_silent_cap_defaults` wrapper and wired it into
  `preflight_all`.
- Added tests proving the positive control flags a censored cap site and the trajectory
  stop-receipt consumer is not flagged.

## Recall Evidence

Bounded recall changed the classification in three places:

- `cw1` proved the class on `sq1`: the old 25-step row was a cap artifact, the 50-step
  rerun improved all 32 rows, and the stop census was still `31/32 iteration_cap_best_at_cap`,
  `1/32 iteration_cap_before_plateau`, `0/32 converged`.
- `sq2` extended the same `#935` denominator to 100 steps: eta rose to
  `0.9112579957356077`, but the stop census remained `21/32 iteration_cap_best_at_cap`,
  `11/32 iteration_cap_before_plateau`, `0/32 converged`, and the pose-accounted gate failed.
  This is a higher floor, not convergence.
- `od3` closed the OD1/OD2 Stage-1 cap artifact on its n32 set: 32/32 semantic stops,
  0/32 `safety_bound_REPORTED`, max actual 75 under a derived 100-step ceiling, eta
  `0.604882865092900`.
- `ddm_os1_optimization_sweep_termination_census_20260802.md` supersedes the original
  "31/31 files" headline as a count: the count is flag-set dependent. Its body says to
  rank by recorded stop/cost facts, not by grep shape. CA1 follows that body.
- `ddm_os1` also found `experiments/multi_pass_inflate_optimizer.py --max-iters 5`
  already has a genuine local convergence flag. CA1 leaves it visible in the generic warning
  backlog until it migrates to `CapStopReceipt`, rather than calling it a live bug.
- `q31` is explicit cap-bound, not silent: 32/32 `iteration_cap_best_at_cap`.

## Inventory

Machine-readable inventory:

| path | bytes | sha256 |
|---|---:|---|
| `.omx/research/ddm_ca1_20260805/ca1_silent_cap_inventory_20260805.json` | 31,627 | `bb825efa32add301a34e0f0b31962d9f1613f9fc5c95aebe6ad04b7fe7e13f58` |
| `.omx/research/ddm_ca1_20260805/ca1_classified_inventory_20260805.json` | 44,775 | `e3930b599150cbcf974264a905601ca36a51068af803e42b99bfe7a0da032aa2` |

The classified sidecar contains all 89 cap-default sites with `path`, `line`,
`scope`, `flag`, `default`, scanner `status`, CA1 class, and rationale.

Scanner-clean sites:

| site | reason |
|---|---|
| `experiments/ddm_cq2_comma10k_tiny_student.py --max-steps 160` | reports stop reason |
| `experiments/ddm_lr2_realization_ladder.py --steps 30` | reports stop reason |
| `experiments/ddm_lr2_realization_ladder.py --max-steps 150` | reports stop reason |
| `experiments/ddm_q31_q3_constrained_solve.py --steps 50` | reports stop reason |
| `experiments/ddm_sq1_stage_decomposition_and_solved_paint.py --steps 25` | reports stop reason |
| `tools/run_repair_campaign_autonomous_floor_loop.py --max-iterations 1` | reports stop reason |

Class-B live or reopen-risk rows:

| rank | site | why it matters | action |
|---:|---|---|---|
| 1 | `experiments/ddm_q3x_q3_convergence_measurement.py --steps 25` | active Q3 gate source; cap can change a seg/pose routing decision | QUEUED behind current owner/scorer order; do not relaunch from CA1 |
| 2 | `experiments/ddm_et1_block16_realization.py --steps 25` | phase-field/coupling input; retained eta and pose spend are score-weighted | QUEUED behind scorer slot and et1/q3x ordering |
| 3 | `experiments/ddm_sq1_pose_null_constrained_paint.py --steps 15` | Q3/pose-null family predecessor; bounded result can misroute the Q3 family if revived | QUEUED only if Q3 branch reopens |
| 4 | `tools/pose_frame0_inverse_solve_probe.py --max-iter 8` | #850-style inverse-solve cap prior; current corpus says stale/off-chain | HOLD as historical audit target; no live rerun |
| 5 | `tools/probe_onpolicy_scorer_surrogate.py --steps 40` | on-policy scorer surrogate can rank a reopened costate path | HOLD until a lane consumes it |
| 6 | `tools/probe_yopo_first_layer_costate.py --steps 4` | costate probe appears in multiple research bundles; cap can truncate support ranking | HOLD until a lane consumes it |

Class C is the 76-site dormant/hygiene remainder in the classified JSON. CA1 did not
find a current load-bearing capped verdict for those sites in the bounded recall scope;
they remain warn-only cleanup targets when touched.

## Rerun Results

No solver reruns were performed. CA1's executed work was scorer-free validation and
instrumentation:

| command | result |
|---|---|
| `.venv/bin/python tools/check_no_silent_cap_defaults.py --write-baseline .omx/research/ddm_ca1_20260805/ca1_silent_cap_inventory_20260805.json` | 6,209 scanned, 89 cap defaults, 83 silent |
| `.venv/bin/python -c "from tac.preflight import check_no_silent_cap_defaults; ..."` | 83 warn-only violations, 0 parse errors |
| `.venv/bin/python -m pytest tools/tests/test_check_no_silent_cap_defaults.py src/tac/optimization/tests/test_trajectory_stopping.py` | 12 passed |
| `.venv/bin/ruff check --isolated --select F821 ...` | passed |

The scanner emits one existing Python `SyntaxWarning` from
`src/tac/composition/alaska_inverse_steganalysis_patterns/__init__.py:18`; AST parsing still
completed with zero parse errors.

## Fire Orders

1. `q3x`: the next cap-sensitive live rerun is owned by the Q3 lane. It must report
   `CapStopReceipt` or an equivalent stop census before any verdict is treated as convergence.
2. `et1`: if the scorer slot opens and q3x ordering permits, rerun the block16 realization at
   a higher/semantic stopping budget with stop receipts. State retained eta, pose spend, and
   subset/n600 denominator.
3. `sq1/#935`: do not duplicate `sq2`. The current receipt is 100-step n32, still cap-bound,
   and pose-accounted negative. Any further uncapping needs a pose-safe formulation before n600.
4. `pose_frame0_inverse_solve_probe`: keep stale/off-chain unless a current vehicle consumes it.
   If reopened, the first landing is stop-reason instrumentation, not a score claim.
5. Dormant 76-site hygiene backlog: migrate to `CapStopReceipt` when touched. Do not silence a
   warning with a marker string unless the real output reports the stop reason.

## NEXT_IF_RESUMED

Run this exact sequence:

1. Re-run `tools/check_no_silent_cap_defaults.py --json` and compare counts to the sidecar.
2. Inspect any count delta before trusting the prior classification.
3. If a live-risk site is now owned by this lane and scorer access is free, add `CapStopReceipt`
   first, then rerun to semantic stop or a higher cap with a recorded `still_descending` flag.
4. If only hygiene sites changed, require the same positive control: known censored site flags,
   trajectory-stop consumer does not flag.

## Evidence And SHA Table

| path | bytes | sha256 |
|---|---:|---|
| `.omx/tmp/codex_runs/ca1_prompt.md` | 4,483 | `ccaa824d8777d317c0ba9b2c76a1e0aeb3192115bac49d95afb9fba620e56545` |
| `.omx/tmp/codex_runs/_common_contract.md` | 4,124 | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` |
| `tools/check_no_silent_cap_defaults.py` | 9,497 | `76ddaa45cc0db0590ac92911a328634c6512f6547a4e2c67d26c7c39f9c502d7` |
| `tools/tests/test_check_no_silent_cap_defaults.py` | 5,311 | `9ded51043335974f6650a0233788f733e46094bac094a34c5de62dc5dc024b45` |
| `src/tac/optimization/trajectory_stopping.py` | 21,770 | `2b4e32f7fdaea90d4589da39d10089e73f7d4095df319f0efb4391173c893a1b` |
| `src/tac/preflight.py` | 3,964,735 | `bace89aa7f49532995c77f4ce26da1c0fd7f60794f60f5bb02e242e7e9736b6f` |

## Boundaries

- No scorer forward pass was run.
- No exact archive was built.
- No `upstream/evaluate.py` run was made.
- No contest-CPU/CUDA claim was made.
- No protected file was edited.
- The existing staged index was not used as an input and was not rewritten by CA1 before the
  serializer step.
- The own-vehicle frontier is unchanged.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
