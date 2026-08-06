# ddm_cvg1 - Next If Resumed

Status after this arm: `lw1_control_variate_gate_replay` was FIRED on the
banked jd4/jd7/jd8q3 gate36 rows and did not produce an admissible correction.

## Dispositions

| Follow-on | Disposition | Fire order |
|---|---|---|
| `lw1_control_variate_gate_replay` | FIRED / SCOPED NEGATIVE | Do not wire a correction from this replay. Cite `RECEIPT.md` and `replay_table.json`. |
| `lw1_sideinfo_residual_correlation_admission` | FIRED / NEGATIVE-OR-INSUFFICIENT | `d_seg` controls failed the joint RMSE+sign bar. `d_pose` was numerically suggestive but has constant control deltas, so admission is insufficient. |
| `lw1_large_scale_license_gate` | FOLDED FOR THIS INSTANCE | No en1/Q3/GN/trust-region scale increase may cite this replay as a positive side-info license. |

## If Reopened

Only reopen with new banked rows, not by re-fitting these rows.

Required row shape:

1. At least two additional independent endpoint windows with `d_pose` gate
   telemetry and a previous-endpoint control different from jd6.
2. Same run/window/basis/adapter/pair-list joins; no inference from filenames.
3. Controls known before target n600 truth is measured.
4. Per-axis leave-one-window-out plus bootstrap bands.
5. Accept only if held-out RMSE/MAE and sign-decision errors improve and the
   control residual correlation is defined and nonzero.

Rate remains blocked until a same-object population-rate row exists. Do not use
`total_counted_bytes` as a sampled rate gate.

Own-vehicle frontier remains:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
borrowed/unmoved.
