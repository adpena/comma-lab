# DDM ZC1 Findings

Status: COMPLETE for the report-only ZC1 scope.
Axis: local CPU/read-only recall and byte/pricing artifact inspection; no scorer slot, no remote job, no paid dispatch.
Score claim: false.
Promotion eligible: false.
Pointer moved: false.

Live own-vehicle authority was read from `.omx/state/main_hot_state.md`, which supersedes the copied frontier text in the common contract:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
Contest pointer remains borrowed and unmoved by this run.

## Summary

| leg | exit | artifact |
| --- | --- | --- |
| `#832_interpolation_free_bound_test` | DONE-with-artifact | `LEG_832_BOUND_TEST_RECEIPT.md` |
| `p1a_phi_composite_r_adjoint` | BLOCKED-with-named-blocker | `LEG_P1A_PHI_COMPOSITE_R_ADJOINT_RECEIPT.md` |
| `mh1_split_bank_gate_per_receipt` | BLOCKED-with-named-blocker | `LEG_MH1_SPLIT_BANK_GATE_RECEIPT.md` |
| `p1a_tw1_repriced_knee_live_vehicle` | BLOCKED-with-named-breakage | `LEG_P1A_TW1_LIVE_VEHICLE_RECEIPT.md` |
| `p2a_t0_harness_drain_six_rows` | DONE-with-artifact | `LEG_P2A_T0_DRAIN_RECEIPT.md` |
| `mh1_recover_lane_skipband_arm_c_524` | DONE-with-artifact | `LEG_MH1_524_LANE_SKIPBAND_RECEIPT.md` |

ZC1 made no source-code edits, no protected-file edits, no live run-dir writes, no scorer/evaluator calls, and no bulky artifacts. The only ZC1 outputs are small Markdown receipts in this directory.

## RECALL EVIDENCE

Searches performed before adjudication:

- Memory registry: `rg -n "ddm_zc1|zc1|20260807" /Users/adpena/.codex/memories/MEMORY.md`.
- Governance and live board: `PROGRAM.md`, `CLAUDE.md` / `AGENTS.md` front matter, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- OH1 seeds: `.omx/research/ddm_oh1_20260807/OH1_FINDINGS.md`, `.omx/research/ddm_oh1_20260807/OH1_CONSUMPTION_PLAN.jsonl`.
- Bound-test recall: searches for `#832`, `interpolation-free`, `ddm_wd1`, `ddm_dc1`, `ba31`, `wr2`, and `12.44`.
- Composite-R recall: searches for `p1a_phi`, `composite-R adjoint`, `ms4d`, `BUNDLE-COMPLETE`, `phi`, `D+`, `D-`, and `#984`.
- Split-bank recall: searches for `split bank`, `split-bank`, `extension_slots`, `active_member`, `ddm_runtime_exporter`, and `ddm_runtime_receiver`.
- TW1 recall: searches for `tw1`, `token_waterfill`, `wr1_cell_records`, `pfs1`, `tq1c`, and current-vehicle archive/cell-record references.
- P2A recall: searches for `#375`, `#841`, `#536`, `#873`, `#862`, `#824`, `canonical_task_status`, and P2A row titles.
- Lane-skipband recall: searches for `#524`, `lane_skipband`, `skipband`, `ARM-CAP`, `ARM-VEH`, `probe_lane_skipband_bindingness`, and trainer flags.

What changed beyond the charter seeds:

- `#832` was not a fresh orphan. WD1/DC1 already consumed the named interpolation-free bound and converted it into a stronger correction-label-cost finding.
- The ms4d bundle is complete, but it is a direct intrinsic/adjoint bundle with no scalar phi, reset-coordinate denominator, or counted actuator basis. That blocks P1A item 1 from being ranked into #984.
- Split-bank evidence is real as a custody requirement, but the current runtime receiver rejects active extension slots. A patch that pretends to consume banks without a producer schema would be apparatus fiction.
- The TW1 driver is intact, but its defaults are pfs1/wr1-specific. No tq1c/current-vehicle cell-record input was found in the searched scope, so rerunning the stale defaults would not satisfy the charter.
- P2A's #873 row is stale in the P2A memo: current canonical status shows completion by `ddm_pj2`. #824 remains pending and cannot be closed from a partial receipt basename.
- #524 lane-skipband was not phantom. Source, trainer flags, DSL wiring, tests, and a bindingness probe exist. What is missing is receiver-closed d_seg A/B evidence for current ARM selection.

## Follow-On Dispositions

- `#832`: FOLDED. The bound test is consumed by WD1/DC1; successor work is the uncapped correction/QA03 scale path, not a rerun of #832.
- `p1a_phi_composite_r_adjoint`: QUEUED-WITH-A-FIRE-ORDER. Build a read-only preflight that maps composite-R adjoints to a declared counted reset-coordinate basis and emits scalar phi with denominator and verdict scope before D+ / D- ordering.
- `mh1_split_bank_gate_per_receipt`: QUEUED-WITH-A-FIRE-ORDER. First real split-bank producer must add `receipt_path`, `member_bytes`, `member_sha256`, `parseback_equal`, and `consumer_row_id`, then receiver/manifest tests can enforce consumption.
- `p1a_tw1_repriced_knee_live_vehicle`: QUEUED-WITH-A-FIRE-ORDER. Materialize current tq1c or ARM selection token lattice, cell records, and descent receipt, then run the existing TW1 driver with explicit arguments.
- `p2a_t0_harness_drain_six_rows`: FOLDED for rows closed/superseded by current evidence; QUEUED-WITH-A-FIRE-ORDER for #536 and #824 as listed in the P2A receipt.
- `mh1_recover_lane_skipband_arm_c_524`: QUEUED-WITH-A-FIRE-ORDER. Run the default-off-safe n6 trainer smoke, then governed n600 A/B only if smoke and memory/custody checks pass.

## Boundaries

No new number in this ZC1 batch is a score. Recalled pricing, status, and bindingness values remain scoped to their source receipts. ZC1 did not move the own-vehicle pointer.

Own-vehicle frontier line: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
