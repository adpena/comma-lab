# NA5 — representative-population rerun of four pose verdicts

Date UTC: 2026-08-09  
Axis: `[macOS-CPU frozen-scorer advisory]`, but **no new scorer forward was run**  
Score claim: `false`  
Promotion eligible: `false`

## Outcome first

**Survived: 0/4. Flipped: 0/4. Indeterminate: 4/4.** This is not a draw: the representative rerun did not execute. All four source formulations require the frozen CPU PoseNet (one also requires SegNet), and the charter says to stop and report `blocked-on-slot` rather than consume a scorer slot. The exact original scratch harnesses/sidecars are also absent in the bounded searched scope, and the landed UB1 recovery changes two load-bearing mechanisms. Running it would not be the charter's “same four formulations, unchanged.”

The exact contest pointer did not move. No eval, dispatch, launch, archive build, promotion, Metal/MPS/CUDA operation, upstream mutation, public-PR-intake mutation, or scorer forward occurred.

## Per-formulation three-way table

| formulation | original video-order prefix `[macOS-CPU advisory]` | seeded stratified n120, seed 20260805 | whole-drive strided n120, stride 5 | verdict |
|---|---|---|---|---|
| `pose_l2_truedepth` | n24: HPLAN_REAL 0.878, L2_REAL 1.296, L2_WITNESS 171.8 (medians) | not run; selection ratio 1.005754 | not run; selection ratio 0.968897 | **INDETERMINATE** |
| `pose_carrier_arms` | n8 pairs 0–7: store_nothing 1.995, real-f0+witness-f1 10.42, warp-real-luma 37.4 (means) | not run; selection ratio 1.005754 | not run; selection ratio 0.968897 | **INDETERMINATE** |
| `pose_mladder_depthwarp` | n24 A0/A2 medians 1.685/1.486; n8 A2+ median 1.223 | not run; selection ratio 1.005754 | not run; selection ratio 0.968897 | **INDETERMINATE** |
| `pose_stratified_texture` | n24 A1T-best median 2.608; n8 diagnostic | not run; selection ratio 1.005754 | not run; selection ratio 0.968897 | **INDETERMINATE** |

Selection ratios are measured on the D2 `d_pose_shipped_f16` governing field, not on any of the four missing reruns. They establish that the prepared samples represent the D2 population; they do not substitute for formulation d_pose.

## The #931 magnitude is now re-derived, with one correction

From the current 600-row raw D2 JSONL (SHA-256 `d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5`), population mean `d_pose_shipped_f16 = 0.15950891917937635`:

- n24 prefix/population = **2.535475579649216×**.
- n96 prefix/population = **4.206770932037034×**.
- Therefore the cited **2.54–4.21× range is reproduced for n24 through n96**.
- The n8 prefix/population ratio is **1.2789847387169815×**, not 2.54×. Its direction is still harder than population, but the larger quoted range must not be attached to the n8 receipt.
- The prepared n120 stratified-random ratio is **1.0057539935665503×**; the matched-n strided ratio is **0.9688974305106292×**.

Exact pair IDs are in `SAMPLES.json`.

## Noise floor

No legitimate repeatability noise floor for any nonzero formulation row was found in the source receipts or recovered artifacts. The original positive controls—`1.2e-12`, `2.1e-12`, and `5.8e-12`—show that PoseNet reproduces its own GT target, but they are not a repeatability floor for arms with d_pose from 0.878 to 171.8. An IID standard error from the serial D2 rows would be invalid because the corpus already records serial dependence. Therefore no survive/flip call is forced from the old point estimates.

## Source provenance each new row would supersede

| formulation | exact verdict source | SHA-256 | commit |
|---|---|---|---|
| L2 true depth | `pose_l2_truedepth_probe_measured_20260708.md:28-47,49-68,121-141` | `39b84b28...cb12` | `c2aa76cf04b01cf8d8e4cf87147fe9fc4c7dbd2a` |
| carrier arms | `pose_carrier_arms_measured_20260708.md:16-48,50-76` | `d2ebf07e...8821` | `16030e6bf65c36108842817c8dfccac0f9ce074a` |
| M-ladder | `pose_mladder_depthwarp_measured_20260708.md:22-44,60-93,103-125` | `259bf0bd...0706` | `70649531f6f3c4f1613872b6431f050743734e92` |
| stratified texture | `pose_stratified_texture_probe_measured_20260708.md:19-41,43-90,102-119` | `026dbcf5...cd6e8` | `7d2784fc9511f1f464391fe6b853e92f938b0a80` |

## Why the recovered harness cannot be fired as the reference form

The landed UB1 harness at commit `f27f04f98b80ddc9b7a2c9c54fbe0e329090c1c6` is useful custody evidence but not an unchanged rerun:

- `build_render_cache` builds a level-set blob without a pose-carrier section and caches the plain INR `witness_f0` (`experiments/ddm_ub1_pose_family_923_harness.py:282-300`). Its `pose_carrier_arms` `store_nothing` branch feeds that unwarped `witness_f0` directly (`:355-358`). The original row defines store-nothing as `warp(witness f0 render, calibrated H)` (`pose_carrier_arms...:21-25,35-47`).
- UB1 loads checkpoint `xi_stored + dxi` (`:157-166`) and uses it for the real-luma and A0-shaped warp (`:359-372`). The original M-ladder A0 derives ξ with `xi_from_pose_calibration(..., s_t=0.16, s_r=1.0, pitch=0.02)` (`pose_mladder...:60-64`).
- UB1 truthfully refuses L2 and texture because their source bytes are missing (`:333-337`) and truthfully exposes only A0, not A2/A2+ (`:368-369`).

Those are mechanism changes, not legal scope reductions. No TOY row was run.

## Blocker classification

1. `BLOCKED_ON_SCORER_SLOT_BY_CHARTER`: all four require frozen scorer forwards; the charter orders STOP.
2. `BLOCKED_ON_SOURCE_CUSTODY`: exact-name search did not find the original scratch harnesses/caches in `/Users/adpena/Projects/pact`, Git history, `/Volumes/VertigoDataTier/pact`, or `/private/tmp`.
3. `BLOCKED_ON_REFERENCE_PARITY`: the partial UB1 reconstruction changes the source mechanism and cannot supersede the original verdicts.
4. `BLOCKED_ON_NOISE_FLOOR`: source receipts contain positive controls but no nonzero-arm repeatability floor.

## OPTIMAL FORM

The reference form remains the four unchanged formulations on both prepared n120 samples. No formulation, retuning, prefix, or aggregate-only mechanism reduction was admitted. Preparing selectors is a legal scope-preserving prerequisite. Execution stopped before the first scorer forward because reference parity and slot authority were absent.

## RECALL EVIDENCE

Queries/surfaces read: the charter and common contract; `PROGRAM.md`; mirrored `CLAUDE.md`/`AGENTS.md`; craft handoff; live hot state; NB1 rank table; NA2 §§2.5.3–2.5.7; all four source receipts line by line; canonical task row 476; all canonical equations filtered for pose/stratified parallax; the canonical research index and DAG content queries for `stored pose`, `post-hoc`, `pose carrier`, and the four receipt IDs; prior NA3/UB1/NA5 receipts; exact-name searches across repo, Git history, SSD, and private tmp; and the live probe/task stores.

Beyond the charter seeds, recall found the prior `ddm_na5_20260805` interrupted attempt, the landed UB1 partial harness, the equation registry's separate A0T aperture anchor, and live DALI-vs-AV pose-target ambiguity concentrated in pose dimension 0. The first two changed the plan: rather than repeat an unbounded cache build or trust a same-name harness, this arm traced source parity and stopped. The decoder-axis finding adds a future reporting requirement—pin the PyAV/macOS-CPU target used by the originals and report per-dimension pose errors—but does not change the four formulations.

## Routing

`NA5_ROWS.jsonl` contains 4/4 rows with owner, `QUEUED-WITH-A-FIRE-ORDER`, consumer `OD1 Stage-2 pose-recovery adjudication`, and consumer store `.omx/state/probe_outcomes.jsonl`. Four blocking `DEFER` events are appended at physical rows 671–674 under the listed `ledger_probe_id` values. The shared ledger had seven unrelated uncommitted rows at NA5 start and gained an eighth concurrent MAIN row before this append; this arm does not absorb or commit any of those eight rows.

## Cheapest next measurement

The cheapest next measurement is `pose_carrier_arms`: reconstruct the original calibrated-ξ path, reproduce the three n8 means first, establish a repeatability floor, then run the exact three arms on both n120 samples in one MAIN-owned frozen-scorer slot. It needs no depth model, A2 solver, or texture grid.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose arm / MAIN scorer-slot owner. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2. Fire trigger:** a source-faithful `pose_carrier_arms` reconstruction reproduces all three n8 rows under calibrated ξ with a declared repeatability floor; then score both prepared n120 samples.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose depth-recovery arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2. Fire trigger:** the original L2 depth cache/harness is recovered or rebuilt and reproduces HPLAN/L2 n24 controls before a scorer slot is granted.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose solve arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2. Fire trigger:** A0/A2/A2+ source parity is restored, including calibrated ξ and the original solver acceptance rule, before a scorer slot is granted.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: pose texture arm. Consumer store: `.omx/state/probe_outcomes.jsonl` and OD1 Stage-2. Fire trigger:** the texture/aperture grids and per-cell warp diagnostics reproduce the source n24/n8 rows before a scorer slot is granted.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN shared-ledger owner. Consumer store: `.omx/state/probe_outcomes.jsonl`. Fire trigger:** the eight unrelated dirty rows 663–670 land or receive separate custody; then commit only NA5 rows 671–674 with patch-intent staging so no sibling row is absorbed.

## LIVE-HYPOTHESES

- At least one absolute d_pose wall may weaken on representative pairs because the source n24 prefix is 2.535× harder than the measured D2 population while both prepared n120 samples are within 3.2% of the population mean.
- The `pose_carrier_arms` ranking may survive even if its absolute values fall: its n8 gaps are 5.2× and 18.7×, much larger than the measured 1.279× n8 selection distortion, but this remains untested without a source-faithful rerun.
- Per-dimension reporting may expose a selection mismatch hidden by scalar d_pose because current decoder-difference evidence concentrates almost entirely in pose dimension 0.

## DEAD-ENDS

- Re-running the landed UB1 harness as “the same formulation” is closed: source inspection proves it skips the original store-nothing warp and substitutes checkpoint ξ for calibrated ξ.
- Repeating the prior unbounded render-cache build is closed under this charter: it is not resumable/stage-checkpointed and cannot produce the two missing L2/texture formulations.
- Inferring survive/flip from the D2 sample ratios alone is closed: those ratios calibrate selection, not formulation d_pose.
- Treating the positive-control values near 1e-12 as the nonzero-arm noise floor is closed: they test target self-reproduction, not run-to-run variation.

S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]; unchanged by NA5.
