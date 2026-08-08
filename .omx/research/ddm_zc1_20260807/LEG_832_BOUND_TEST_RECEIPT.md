# ZC1 Leg Receipt: #832 Interpolation-Free Bound Test

Exit: DONE-with-artifact.
Axis: source-recalled scorer-free pricing and bound-test receipts; no ZC1 scorer run.
Score claim: false.
Promotion eligible: false.
Verdict scope: INSTANCE, the named #832 interpolation-free bound test as recorded by WD1/DC1.

## RECALL EVIDENCE

Searches performed:

- `rg -n "#832|interpolation-free|12.44|sign swings|wr2|wd1|dc1|ba31" .omx/research .omx/state src tools`
- Targeted reads of `.omx/research/ddm_wd1_pose_wiring_falsified_and_correction_minimum_scale_20260802.md`.
- Targeted reads of `.omx/research/ddm_dc1_correction_label_cost_and_qa03_censoring_20260801.md` and its JSON companion.
- Targeted reads of `.omx/research/ddm_ba31_negative_surfaces_20260731.md`.

Found beyond the charter seed:

- WD1 says DC1 had already run the ba31 named cheap test on 2026-08-01 and both controls passed.
- The waterline used there was `W = 1.2731082153 B/flip`.
- The label-free ba31 row was already the interpolation-free bound: `0.9822 B/flip`, `499,579` bytes, rate term `+0.332649 S`, seg term `-0.431179 S`, net `-0.098530 S`, equal to `12.44%` of the old gap in that receipt's baseline.
- DC1 superseded the label-free-only framing by measuring label cost and showing the correction stream still has a minimum viable scale requirement.

What this changed:

- ZC1 did not rerun #832. The named procedure is not missing; it is already consumed by WD1/DC1.
- The honest ZC1 action is to fold #832 into its existing receipts and avoid relabeling it as a fresh OH1 orphan.

## Verdict

The #832 interpolation-free bound test is DONE-with-artifact by prior receipt. The bound was win-confirmed in the scoped pricing sense, then sharpened by DC1's correction-label-cost analysis.

Important scoped values, as source-recalled rather than newly measured by ZC1:

- Bound row: `0.9822 B/flip`, `499,579` bytes, `-0.098530 S` net on the receipt baseline.
- Live reanchor in WD1/DC1: ja1/v4c base `0.8507 B/flip`, `0.668x W`, net `-0.143049 S`, `18.45%` of gap `0.7754681`.
- Burn ep854 reanchor: `0.8837 B/flip`, `0.694x W`, net `-0.120605 S`, `15.55%` of the same recalled gap.
- Correction stream viability threshold: `f* = 0.2860`; QA03 at `0.3669%` residual is about `78x` below that viable scale in the source receipt.

## Follow-On Disposition

FOLDED for #832. The bound test is no longer a dangling free test.

QUEUED-WITH-A-FIRE-ORDER for the successor correction stream: only reopen with the DC1/WD1 uncapped correction/QA03 procedure, explicit label-cost accounting, and the same-source baseline named. ZC1 does not allocate a scorer slot or launch that successor.

Own-vehicle frontier line: unchanged, `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
