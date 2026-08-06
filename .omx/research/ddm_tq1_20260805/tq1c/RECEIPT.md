# ddm_tq1c receipt

Axis: `[macOS-CPU frozen-scorer advisory]`. `score_claim=false`; no contest-CPU/CUDA promotion row is claimed.

## Result

- Base: tq1b final archive `75df9cc341d8188b47fe38337d2ec996600c245112a008c3f333957a6522db59`, 357,837 B.
- Phase A: generated 1,133 new-base menu rows; priced the 26-move wall-budget prefix.
- Phase B: realized 26 queued moves through receiver -> R -> uint8 -> frozen CPU scorer; accepted 6.
- Accepted moves: `snap_r00_c23_L12`, `snap_r00_c25_L12`, `snap_r00_c26_L12`, `snap_r00_c28_L12`, `snap_r00_c30_L12`, `snap_r00_c12_L13`.
- Baseline measured row: `S=0.7537933983374265`, `d_seg=0.004308641222`, `d_pose=0.00071673674`, bytes `357837`.
- Final advisory row: `S=0.7534578126155775`, `d_seg=0.004305419922`, `d_pose=0.000716508925`, bytes `357837`.
- Realized delta vs measured tq1b-final baseline: `delta_S=-0.000335585722`, `delta_d_seg=-0.000003221300`, `delta_d_pose=-0.000000227815`, `delta_bytes=+0`.
- Final archive: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`, sha256 `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`.

## Saturation

State: `SATURATED_ACCEPTED_PREFIX`.

Scope: queued 26-move Phase A price-ledger prefix only. The new-base generated menu has 1,133 rows; only rows 1-26 were priced and realized. The parent qo1 menu had 1,140 rows, but that is not the tq1c closure denominator.

Resume point: regenerate or extend the Phase A price ledger from new-base menu index 27, candidate `snap_r01_c13_L12`, starting from the held final archive `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`. Do not claim full-menu closure until all 1,133 generated rows are priced and realized or explicitly folded.

## Artifacts

- Phase A receipt: `.omx/research/ddm_tq1_20260805/tq1c/phase_a_receipt.json`.
- Phase B receipt: `.omx/research/ddm_tq1_20260805/tq1c/phase_b_realized_acceptance_receipt.json`.
- Restage manifest: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/tq1c_base/restage_manifest.json`, sha256 `584f06e5ea7b9ac19ccfbb09101ec0e4dea3db4768319f8f28e561bccc67687a`.
- Candidate menu: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form_tq1c/tq1_phase_a_candidate_menu.jsonl`, 1,133 rows, sha256 `ad26e7811a782f9f85abbfbaa111ac30e5aab229bc1d1e35e4b0962cd28410d7`.
- Price ledger: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/optimal_form_tq1c/tq1_phase_a_candidate_prices.jsonl`, 26 rows, sha256 `3eb8879f6efbc292a5a520486d12e594a15c89ab6caebe2905ffbcb67d921b6b`.
- Realized measurement JSONL: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/tq1_phase_b_realized_measurements.jsonl`, 26 rows, sha256 `e0427f27beae82340dcdc9e17d472d51d32984416e409ebda3c76e15078ca15c`.
- Accepted-move ledger: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/tq1_phase_b_accepted_move_ledger.jsonl`, 6 rows, sha256 `a992f0112d5e7e02d3a773cf3daa0bd6e1c726c26325168b77a8770186497216`.
- Scorer checkpoints: `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/stage_checkpoints`.

SSD footprint: `optimal_form_tq1c` 2.0M, `phase_b_realized_tq1c` 23M, `tq1c_base` 2.1M.

## Recall Evidence

- Read charter files: `.omx/tmp/codex_runs/tq1c_prompt.md` and `.omx/tmp/codex_runs/_common_contract.md`.
- Read governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Searched canonical equations for token/waterfill/IX2/score/archive/receiver terms; relevant hits included frontier pointer law, score marginal Lagrange, correction-stream label-cost exclusions, and IX2 token/coder equations.
- Searched `.omx/research` for `tq1`, token edit, `snap_sublattice`, IX2, `#969`, and saturation strings.
- Beyond-charter findings: `.omx/research/ddm_tq1_preempted_by_rt1_and_sl2_composition_20260805.md` shows the stale tq1 blanket-map negative was not optimal-form token closure; `.omx/research/ddm_sv2_smevr_base_rule_race_20260803.md` identifies IX2TOK01 as the live token field; `.omx/research/ddm_en1_20260805/EN1_RECEIPT.md` keeps entropy pricing as triage only; `.omx/research/ddm_xo1_20260805/XO1_RECEIPT.md` blocks control-head ordering shortcuts; `.omx/research/ddm_lm1_20260805/RECEIPT.md` names tq1c as the consumer for accepted-rate vs rank yield evidence.

Contest pointer unchanged: `S=0.1910828242`, borrowed/unmoved.
