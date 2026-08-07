---
schema: ddm_rv2_findings.v1
arm: ddm_rv2
date: 2026-08-07
axis: "[CPU-only corpus regrade; no scorer slot; no Metal; no upstream/evaluate.py]"
score_claim: false
pointer_moved: false
---

# RV2 Findings - reactivation re-grade against the composed-vehicle preconditions

## Verdict

RV2 produced the required full-corpus reactivation table:

- Table: `.omx/research/ddm_rv2_20260807/RV2_REGRADE_TABLE.jsonl`
- Rows: 21 typed verdict rows.
- Dispositions: 6 `REOPEN-with-named-retest`, 9 `ALREADY-COVERED-BY(...)`, 6 `HONEST-NON-REACTIVATION`.
- Scorer/eval status: no scorer run, no archive mutation, no `upstream/evaluate.py`, no contest-CPU/CUDA row.

The reopens are not blanket reactivations. They are narrowly tied to P1-P5:
semantic tokens, trained semantic renderer, HPAC/label coder, semantic-pose route,
and terminal tq1/GN/MC finishers.

## Ranked REOPEN Fire Order

| rank | verdict_id | why it ranks there | next retest |
|---:|---|---|---|
| 1 | `rv2_r07_q3_889_pose_placement_semantic_base` | Highest Delta-S reach: #889 is placement-conditional, and a pose-held semantic burn can unblock composition without reviving post-hoc repair. | n>=120 random/stratified Q3-first or pose-null burn on selected semantic/pose-carrying base; matched pose control required. |
| 2 | `rv2_r04_per_class_carriers_cb1_semantic_stream` | Cheap byte-only first pass with direct P1/P3 contact; can retire or promote without scorer. | Class-conditional PP1-KT/HPAC vs monolithic semantic-label stream with exact decode equality. |
| 3 | `rv2_r01_per_pixel_sidecar_semantic_token_region` | Old sidecar negatives were camera-grid formulations; semantic-token edit atoms are a different rate surface. | Top SN1/error-source semantic-token region edits, byte-priced before scorer. |
| 4 | `rv2_r05_pe3_edge_partition_labels` | Edge labels become renderer conditioning rather than post-hoc pixel sidecars. | Paired label/no-label semantic renderer control, then n>=120 only if parse-back passes. |
| 5 | `rv2_r20_terminal_solver_gn_mc_finishers` | P5 reopens implementation/cap negatives, but standalone search quality measured tiny on one TR1 instance. | Attach GN/MC only to a high-reach reopened row with non-tail controls and cap telemetry. |
| 6 | `rv2_r19_sn1_menu_ranks_881_889_894_896_918_927_928` | The charter's numeric cluster is open-price menu evidence, not a closed task owner. | Realize selected SN1 rows as semantic-token edits and coder-price receiver-closed bytes. |

## Folded / Already Covered

- AA1 already covers #869/#933 waterfill on a PR130-class stream, HPAC model self-compress, direct PR130 blind-coordinate audit, and the #827 composition attack sheet.
- HB1 already owns the target-payload semantic-label byte race; PP1-KT remains the measured incumbent until HPAC is trained on OUR labels with exact decode equality.
- WL1 already ported the viable witness-line levers into PORT-NOW/RACE rows; RV2 does not duplicate them.
- ET1/BZ1/DQ1 already establish: the phase-field pose axis is not the blocker, the specified legal seg carriage does not byte-close, and #827 should be aimed with frame_0 pose repair rather than v4c ep854 post-hoc solve.
- Literal camera-grid sidecars, post-hoc finite pose repair on a damaged seg-only base, loss-only per-class tilts, old directional proxy routing, and KS1's measured endpoint are not reactivated.

## RECALL EVIDENCE

Governing reads:

- `.omx/research/ddm_rv2_20260807/CHARTER.md` and `.omx/tmp/codex_runs/_common_contract.md` were read in full.
- `CLAUDE.md` and `AGENTS.md` were checked byte-identical, then targeted sections were read: v7.5/v8 operating contract, anti-forgetfulness, NO FAKE, goal/frontier, serializer/review gate, negative-scope discipline.
- `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md` were read. Current own-vehicle frontier from hot state is `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.

Required corpus surfaces consumed:

- RV1: `.omx/research/ddm_rv1_conditional_validity_regrade_20260728.md` and DAG feed. Used its conditional-validity method and prior R/X rows, especially per-pixel sidecar, token-stream coder, KD, lane-channel, and closed literal sidecar rows.
- Negative registry/scope ladder: `.omx/research/negative_findings_register_20260709/*`, `negative_findings_reaudit_20260710.md`, `negative_cure_join_table_20260710.md`, and `adversarial_review_all_negative_findings_20260707.md`. Used the INSTANCE/FORMULATION/FAMILY/PARADIGM ladder and TOY-NAMED implementation rescope notes.
- WL1: `.omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md`, `RECEIPT.md`, and `NEXT_IF_RESUMED.md`. Used PORT-NOW/RACE/DEAD/PRECONDITION-CHANGED rows to avoid duplicate witness-line reactivations.
- Costate organ rows: `.omx/research/ddm_co7_organ_round7_20260728.md` and `.omx/research/ddm_co8_organ_round8_20260728.md`. Used the conditional precondition schema and rv1 proposal consumption rule.
- AU1 corrections/headline index: `.omx/research/ddm_au1_20260805/au1_corrections_index.jsonl` was scanned for PR130, semantic, HPAC, sidecar, pose, KD, distill, Q3, band, waterfill, and et1; this confirmed correction density and stale/refuted headline risk.
- Probe ledger: `.omx/state/probe_outcomes.jsonl` and `.omx/research/probe_outcomes_canonical_ledger_landed_20260516.md` were scanned. The ledger contains many scoped blocking/advisory entries; RV2 did not promote any probe-only row to a global kill.
- KS1: `.omx/research/codex_findings_ddm_ks1_knee_member_realization_20260725T142706Z_codex.md` and `ddm_ks1_knee_member_batch32_remeasure.json` were consumed as an instance advisory endpoint, not a paradigm kill.
- Current campaign surfaces: `ddm_aa1_20260807`, `ddm_hb1_20260806`, `ddm_rr4_20260806`, `ddm_et5_20260807`, `ddm_et1_eta_on_the_priced_band_20260803.md`, `ddm_dq1_20260805/RECEIPT.md`, `ddm_bz1_phase_field_byteclose_20260804.md`, `ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md`, `ddm_bo1_base_objective_menu_order_20260802.md`, `ddm_bo1_seg_base_objective_menu_order_20260803.md`, and `ddm_q31_20260804/Q31_Q3_CONSTRAINED_SOLVE_RECEIPT_20260804.md`.
- Canonical equations: `tools/list_canonical_equations.py --json` was queried for semantic, HPAC, waterfill, band, pose, token, PR130, lane, per-class, distill, and KD. Notable hits included `ddm_hb1_semantic_label_incumbent_transfer_v1`, `rate_mdl_cosmological_constant_reverse_waterfill_v1`, `waterfill_annulus_through_r_store_realization_vs_witness_capacity_v1`, and `ddm_pp1_correction_stream_position_band_v1`.
- Research index: `CANONICAL_RESEARCH_INDEX*` was searched for PR130, semantic, HPAC, #869, sidecar, KD, per-class, Q3, band, curvelet, waterfill, #827, #933, IX2TOK01, PE3, cb1, and et1. This confirmed old D1/D2/D4/D7/R21/W8 rows and the store-the-flips NO-GO x3.

Task/status distinction:

- `canonical_task_status.jsonl` did not expose the charter's bare numeric cluster as owned `task_id`s. The explicit relevant row found was `xa1_seg_token_solve_uncap_and_multistart_ab`, later priced by `sm1_seg_search_headroom_threshold`.
- The bare numbers 881/889/894/896/918/927/928 were found as SN1 menu ranks in `.omx/research/ddm_sn1_error_source_tensor_n600_20260723/error_source_solve_menu.jsonl`. They are `DESCRIBED_BUT_REALIZATION_LOST` / `OPEN_PRICE` menu rows, so RV2 treats them as receiver-price candidates, not as already-owned tasks.

## Boundaries

- CPU-only document/data regrade. No Metal, no MPS authority, no scorer slot, no archive build, no `upstream/evaluate.py`.
- No protected files were edited.
- No `.py` files were edited; review-tracker x2 and `.py` review gate do not apply.
- No persisted evidence path points at `/tmp`.
- The input charter remains unchanged.

## Follow-Ons

- FIRED: RV2 table and memo were written.
- FOLDED: literal camera-grid sidecars, post-hoc finite pose repair on seg-only damaged bases, loss-only class tilts, old directional proxy routing, KS1 endpoint rerun, and ET5 restricted patch sidecar.
- QUEUED-WITH-FIRE-ORDER: the six REOPEN rows in the ranked table above.
- ALREADY-COVERED: AA1, HB1, WL1, BZ1/DQ1/ET1, and live-row routes named in the JSONL table.

## Frontier

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]` from `.omx/state/main_hot_state.md`; contest pointer remains the borrowed contest-CPU row and is not moved by RV2.
