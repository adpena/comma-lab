# Pose instrument gate wiring — ddm_pm2

Apparatus only; score_claim=false. No scorer or pricing job ran. The scorer-free mocked execution suite passed 20 tests; ruff passed all nine Python files listed below. Numerical routines are unchanged: admission adds refusals and records, never a correction to values.

## Implemented scope

- Population producers: `ddm_sj1_joint_admission.cmd_pose --tag base`, `ddm_rp1_pose.cmd_pose --tag base`, `ddm_fe1_admit_and_build.cmd_base_pose`. Capture pointer before measurement, compare measured n600 mean to its promoted component, refuse before NPY/base report creation, retain gate receipt.
- Pair producers: `ddm_fe1_pose_price.cmd_price`, `ddm_fe1_admit_and_build.cmd_pose`, `ddm_fe1_rebase.cmd_rebase`, `ddm_fe1_cmp1_rebuild.main`, `ddm_fe1_pass4_race.main`, `ddm_rw1_renderer_edge_foldback.cmd_pose`. They now require `--base-pose-reference` (n600 per-pair NPY). `prepare_pose_reference` checks supplied population mean and reference population mean against live pointer before measurement; `evaluate_base_codes` measures unchanged code, verifies pointer identity, and compares to the SAME pair's reference. Pair-to-population comparison would be invalid. The source reference bytes are read once and hashed; its receipt and per-pair gate result travel in the saved rows.
- `--pose-base-differs-because` is the explicit non-placeholder waiver; missing pair reference with rationale remains visibly waived, never passed. `TAC_INSTRUMENT_GATES=0` preserves the legacy live-process route and produces an invalid bypass receipt. Existing artifact names and measured numbers stay unchanged.
- Unit tests live at `src/tac/tests/test_ddm_pm2_pose_wiring.py`, because review_tracker excludes root tests/.

## RECALL EVIDENCE

Searched `.omx/research/` content for `pose.base.*(magnitude|500)`, `base measured.*overlay`, `same.instrument.*pose`; canonical equations via `tools/list_canonical_equations.py --json` with pose/instrument/prefix terms; `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*`; docs SPEC surfaces and state task/ledger surfaces with pose/base/instrument terms. A prior memory registry entry (MEMORY.md:31) reinforced consumed-field controls and explicit census denominators; code and contemporary memos were the current evidence.

Beyond the charter seeds: source inspection found fe1 has a separate n600 producer in `ddm_fe1_admit_and_build.cmd_base_pose` and four more fe1 pair-producing entrypoints, plus rw1's per-pair pose producer. All joined this gate. The research index documents historical pose-lineage/vehicle changes, so imposing the current pointer's numerical band on every historical vehicle would be an invalid control rather than extra coverage.

`ddm_sj1_multipass_token_predistortion_20260905.md` §19c records the original no-overlay d_pose 2.559e-3 and correct base 5.090165e-6; the wrong object is about 502.7 times the correct base. It also explains why even an existing overlay can name the wrong subset (full 445 pairs versus admitted 370). Magnitude is a gross-confound gate, not proof of exact decode identity.

## Historical band evidence, bounded

Read-back, not new measurements: sj1 memo lines456–457 gives measured base 5.7675e-6 versus seal 5.77e-6, ratio 0.99957. §19c gives base 5.090165e-6 versus pc2 scalar 5.09276404439735e-6, ratio 0.99949 (versus rounded T4 5.1e-6, 0.99807). rp1 memo §10 reports an independently measured move37 base 5.049765771412152e-6, bit-identical to sj1 pass4 CLOSE.json; cmp1/cmp2 preserved the pose body. These support the band but do not constitute an independently verified seven-of-seven census of promoted move31–37 base receipts. No falsifier found in this bounded scope.

## Census boundaries

`pose_base_census.json`: 698 `experiments/ddm_*.py` parsed, zero parse errors. Ten assignments contain `base` in the target and a descendant call named measure_pose/evaluate_codes/evaluate_base_codes. Six current pair producers are guarded. Four older distinct-vehicle producers remain outside the current-family gate: bs4y_stage_executor, jg5_pose_resolve_on_edited_renders, qs1_frame0_schur_coupled_solve, qs5_resolve_compensation. The heuristic is intentionally labeled: generic `per_pair` outputs selected by tags are not in its denominator, and the three current population producers above were found by direct inspection. It is not an exhaustive repository proof.

## Review

Review 1 traced refusal ordering and preservation of numerical values. Findings fixed before clean marks: reference byte hashing now covers exactly the bytes parsed; malformed/missing references produce typed alarms; every pair row retains population provenance; pass4 local `price` name collision avoided with `instrument_price` alias.
Review 2 traced snapshot lifetime, explicit waiver and environment bypass, all CLI flags, and scorer-free tests. Shared assumption challenge: an n600 reference cannot stand in for any individual pair; matching per-pair controls are therefore required. Magnitude cannot detect small wrong-tree differences within the band, so existing archive/carrier identity checks remain necessary.
