# Wiring-pass: close the 3 CRITICAL un-wiring gaps (2026-06-17/18 session)

**Operator: close the CRITICAL un-wiring gaps a completeness audit found in today's work.**
Per the SoT `.omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md` §"NO-SIGNAL-LOSS / wire-in status"
subsections (a)/(b)/(c). All registrations went through the CANONICAL HELPERS (no invented APIs/schemas).
`$0`; `[contest-CPU advisory]` NON-PROMOTABLE throughout; exact pointer UNMOVED at 0.19110.

Disciplines closed: "Results must become system intelligence" / "no orphaned signals" / "Canonical
equations registry NON-NEGOTIABLE" / Catalog #331 (canonical task status ledger) / Catalog #313
(probe-outcome continual-learning).

## (1) Canonical equation — the stretched-exponential d_seg model
`dseg_stretched_exponential_anneal_trajectory_v1`: `d_seg(ep) = 0.00566·exp(−(ep/4263)^0.860)`.
- Registry: 410 → **411** equations (read-back: `tools/list_canonical_equations.py --equation-id dseg_stretched_exponential_anneal_trajectory_v1 --verbose`).
- 1 EmpiricalAnchor (CE-baseline fit): predicted d_seg(ep1000)=0.004248 vs empirical 0.004355 → residual 0.0246 (well-calibrated).
- Producer: re-audit (`tac.canonical_equations.builtins`). Consumers: `experiments.launch_bind_all_taper_ab` (the running long-train thesis), `tools.list_canonical_equations` (the sub-0.15 projection surface).
- Source: `closure_reaudit_round2_synthesis_audited_20260618.md` (16× better fit, SSE 2.7e-8 vs power-law 4.3e-7) + CE-control log `run_CE_baseline_ep3700.log` (ep1000 d_seg=0.004355, ep3700=0.002370). Domain EXCLUDES the power-law model class + bc28+ capacity scaling. It is the model the "sub-0.15 d_seg=0.000322 feasible at ~14.5k ep" thesis rests on (was prose-only = tribal-knowledge violation).

## (2) Canonical task status — 7 tasks (Catalog #331; ledger was frozen 2026-06-10)
Registered via `register_task` + `update_status` honoring `VALID_TRANSITIONS` (pending→in_progress→completed). Ledger strict-valid, 56 rows; **none of the 7 new tasks appear in the no-dangling-transition violations** (the 6 pre-existing violations are unrelated 2026-05-30 rows w/ missing source memos).

| TaskCreate | canonical task_id | status |
|---|---|---|
| #127 | `g3_dual_exact_row_bc20_first_byte_closed_20260618` | completed |
| #134 | `final_fine_tune_converged_checkpoint_to_exact_row_20260617` | pending |
| #136 | `fp_shrink_qat_rate_lever_smoke_20260617` | in_progress |
| #137 | `boundary_flip_sidecar_native_grid_in_cell_repair_20260617` | pending |
| #138 | `lane_poly_geometric_spatial_prior_iou_gate_20260617` | pending |
| #139 | `ego_hood_per_frame_mask_re_measurement_reopened_20260617` | pending |
| #140 | `pose_low_rank_radial_zoom_codec_build_20260617` | in_progress |

## (3) Probe outcomes — 8 decisive verdicts + 1 supersession cross-ref (Catalog #313)
Registered via `register_probe_outcome` (verdict vocabulary: KILL/DEFER/PROCEED — there is NO `FALSIFY` token, so the cost-allocation falsification maps to `KILL`). Ledger strict-loads (612 → **621** rows).

| probe_id | verdict | meaning |
|---|---|---|
| `yousfi_detector_cost_blindspot_b_20260617` | KILL | d_seg=irreducible boundary residual; per-pixel detector cost-alloc free-lever FALSIFIED (0% interior-avoidable) |
| `blindspot_probe_c_measurement_trust_dseg_dpose_20260617` | PROCEED | d_seg wall is REAL (not EMA-shadow artifact); pose eval-noise = FiLM carrier |
| `sufficient_statistic_floor_probe_20260617` | PROCEED | SS-store floor 0.2429 → learned decoder is the cheaper SS carrier |
| `frontier_rate_cut_vs_small_basis_anchoring_probe_20260617` | PROCEED | frontier rate AT entropy floor → FP-shrink must be LOSSY/QAT, not a recode |
| `compress_time_seed_and_solve_dseg_20260617` | DEFER | seed+solve NOT faster than descent → don't pivot to a solver finisher |
| `accel1_margin_hinge_flip_targeting_dseg_exponent_20260617` | PROCEED | margin-hinge BENDS d_seg exponent (0.787 vs CE 0.608) — the seg lever |
| `ego_hood_per_frame_mask_region_corrected_reopen_20260617` | DEFER | region-corrected REOPEN (7.36% per-frame hood band); survival-gated |
| `pose_lowrank_corrected_fidelity_radial_zoom_20260617` | PROCEED | REOPENED at MSE≤d_pose; rank-2 SVD 2.7× smaller; 1-DOF radial-zoom |

**#137 supersession cross-ref** appended (EVENT_BACKFILL, DEFER preserved) to
`partition_store_realization_gate_20260616`: the non-neural partition STORE is superseded as a lever by
the better-posed roundtrip-real #137 native-grid in-cell repair / boundary sidecar (per round2 reopen #4).

## NO-FAKE / honest-labeling notes
- Equation `python_callable_module_path` points at `tac.canonical_equations.builtins` (a real module, satisfying the dotted-path contract); the model has no production callable yet — the running long-train IS the empirical validator (the next anchor recalibrates per RECALIBRATE_ON_NEW_ANCHORS).
- Anchor/equation `inputs_sha256` are sha256 content-ids of identifying research-artifact strings (NOT byte-closed archive bytes), honestly labeled — these are advisory model-fits/verdicts, not exact-eval rows.
- The script that performed the registrations was ephemeral scratch (`.omx/tmp/`, not committed); only the 3 canonical ledgers + this memo are committed.
