# ZC1 Leg Receipt: MH1 Recover Lane-Skipband Arm C #524

Exit: DONE-with-artifact for recovery and routing.
Selection blocker: `NO_RECEIVER_CLOSED_DSEG_AB_RECEIPT_FOR_LANE_SKIPBAND`.
Axis: source/receipt read and no-scorer routing; no scorer run.
Score claim: false.
Promotion eligible: false.
Verdict scope: FORMULATION, lane-skipband as a current renderer/ARM candidate after recovery.

## RECALL EVIDENCE

Searches performed:

- `rg -n "#524|lane_skipband|skipband|skip-band|ARM-CAP|ARM-VEH|probe_lane_skipband_bindingness|LaneSkipBand" .omx/research src tools experiments`
- Targeted read of `.omx/research/arm_c_skiplever_ema_comparator_build_20260717.md`.
- Targeted read of `tools/probe_lane_skipband_bindingness.py`.
- Targeted reads of `src/tac/boundary_math/lane_skipband.py`, `src/tac/tests/test_lane_skipband.py`, and trainer flag references in `experiments/train_levelset_witness_realized_through_R_mlx.py`.
- Targeted read of `.omx/research/ddm_mh1_month_harvest_20260803.md` for the orphan-harvest framing.

Found beyond the charter seed:

- #524 is not phantom. Code, tests, DSL wiring, trainer flags, and the bindingness probe exist.
- The branch/build receipt records task #524, default-off safe wiring, fail-closed micro-batch handling, and `LaneSkipBand` DSL exposure.
- The n24 bindingness probe records `binds_when_enabled=true`, term-on mean around `1.549e-3`, gradient L2 mean around `1.05e-3`, and witness skip-band energy around ten percent of GT skip-band energy.
- The same receipt is explicit that this is not a d_seg score claim and not an end-to-end trainer A/B.
- Provider precompute was estimated around 236 MB at n600, so a later run needs storage/custody preflight.

What this changed:

- MH1's recovery row should not be treated as missing or dead.
- It also cannot be routed as already receiver-closed for ARM-CAP/ARM-VEH selection. The missing artifact is a current-object d_seg A/B receipt.

## Verdict

Recovery DONE-with-artifact. Selection remains blocked by missing receiver-closed d_seg A/B evidence.

Recovered surfaces:

- `src/tac/boundary_math/lane_skipband.py`
- `src/tac/tests/test_lane_skipband.py`
- `tools/probe_lane_skipband_bindingness.py`
- trainer flags for `lane_skipband_weight`, start epoch, dilation, and micro-batch guard
- receipt: `.omx/research/arm_c_skiplever_ema_comparator_build_20260717.md`

## Follow-On Disposition

QUEUED-WITH-A-FIRE-ORDER:

1. Run a default-off-safe n6 trainer smoke on the current vehicle with lane-skipband enabled at a small positive weight, lane-render-band OFF, and micro-batch pairs fixed to one.
2. Include storage/custody preflight for the provider precompute.
3. If the smoke confirms binding and default-off inertness, schedule a governed n600 current-object A/B.
4. Feed ARM-CAP/ARM-VEH selection only after receiver-closed d_seg evidence exists.

Own-vehicle frontier line: unchanged, `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
