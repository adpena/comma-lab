# ddm_lw1 - Next If Resumed

Resume objective: turn the lw1 control-variate crosswalk into a measured
scorer-free replay over already-banked gate-vs-n600 rows. Do not run a scorer,
launch training, touch upstream, or claim score movement under this resume unless
a new charter explicitly grants that slot.

## Immediate Fire Order

1. Read `RECEIPT.md`, `_common_contract.md`, and `.omx/state/main_hot_state.md`.
2. Locate joinable rows in this priority order:
   - JD endpoint n600 both-bases receipts produced by
     `experiments/ddm_jd4_endpoint_n600_both_bases.py`.
   - A1/a1_gate or gate36 telemetry rows from the same run, basis, adapter, and
     pair list.
   - NA4-style rate rows only if the stream/coder/full-population denominator is
     the same object.
3. Build a read-only replay table with columns:
   `run_id`, `window`, `basis`, `axis`, `selection_mode`, `pair_ids`,
   `gate_estimate`, `truth_n600`, `control_name`, `control_subset`,
   `control_n600`.
4. If any join key is missing, stop with
   `BLOCKED_NO_JOINABLE_GATE_ROWS` and list the missing key/path. Do not infer
   row identity from nearby filenames.
5. Fit the scalar control-variate correction:
   `gate_estimate - beta * (control_subset - control_n600)`.
6. Validate leave-one-run/window-out, separately for `d_seg`, `d_pose`, and rate.
7. Report raw vs corrected RMSE, MAE, sign-decision error, coverage against n600,
   and whether prefix remains a different population.

## Queued Follow-Ons

| Status | Item | Fire order |
|---|---|---|
| QUEUED-WITH-FIRE-ORDER | `lw1_control_variate_gate_replay` | Execute the immediate fire order above. Accept only held-out residual reduction. |
| QUEUED-WITH-FIRE-ORDER | `lw1_sideinfo_residual_correlation_admission` | In the replay output, every side-information candidate reports held-out residual correlation and an axis-specific verdict. |
| QUEUED-WITH-FIRE-ORDER | `lw1_large_scale_license_gate` | Before any en1/Q3/GN/trust-region scale increase cites Lam/Wang, require a positive side-information admission row from the replay. |
| FOLDED | `lw1_direct_eo_plus_archive_vehicle` | No archive vehicle exists in this paper. |
| FOLDED | `lw1_blind_perturbation_campaign` | Use as a pruning lesson only. |

## Non-Negotiable Boundaries

- `score_claim=false` unless a new exact-eval charter supersedes this file.
- No `/tmp` persisted evidence.
- No `upstream/` edits.
- No protected-file edits named in `_common_contract.md`.
- No staged-index touch except the final serializer commit if a resume lands new
  artifacts.
- Keep axes separate: `[macOS-CPU advisory]`, `[contest-CPU]`, and
  `[contest-CUDA]` are never inferred equivalent.

Own-vehicle frontier line at lw1 landing:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer
borrowed/unmoved.
